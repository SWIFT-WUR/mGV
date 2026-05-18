function initialize_routing_model(param_file, elev)
    println("Initializing Kinematic Wave Routing...")
    println("  -> Source: $param_file")

    if !isfile(param_file)
        error("Routing parameter file not found: $param_file")
    end

    ds = NCDataset(param_file)

    # 1. Load raw data (CPU) 
    # Helper: Load and sanitize
    function load_safe(varname, T, fallback)
        data = ds[varname][:, :]
        return T.(replace(data, missing => fallback))
    end

    println("  -> Loading variables...")
    # IDs: Unique identifier for each cell. Missing = -1.
    raw_ids = load_safe("downstream_id", Int32, -1)
    # Targets: The ID of the cell downstream. Missing = -1.
    raw_target = load_safe("downstream", Int32, -1)

    # Physics parameters. Missing = NaN.
    raw_dist = load_safe("cell_dist", FloatType, ft(NaN))
    raw_area = load_safe("cell_area", FloatType, ft(NaN))
    raw_acc = load_safe("accumulation", FloatType, ft(NaN))

    close(ds)

    nx, ny = size(raw_ids)
    n_total = nx * ny
    println("  -> Grid: $nx x $ny ($n_total pixels)")

    # 2. Build Topology (with lookup table)
    # We must map ID -> Linear Index.
    println("  -> Building connectivity graph...")

    id_to_index = Dict{Int32,Int32}()

    # A. Map every valid ID to its 1-based linear index
    for i in 1:n_total
        id_val = raw_ids[i]
        if id_val != -1
            id_to_index[id_val] = Int32(i)
        end
    end

    # B. Build the downstream pointer array
    cpu_downstream = fill(Int32(-1), n_total)

    for i in 1:n_total
        # Where does cell i want to go?
        target_val = raw_target[i]

        # If target is valid and exists in our grid, save the index
        # If target is -1 (missing) or not found (outside domain), keep as -1
        if target_val != -1 && haskey(id_to_index, target_val)
            dest_idx = id_to_index[target_val]

            # Prevent self-loops (infinite accumulation)
            if dest_idx != i
                cpu_downstream[i] = dest_idx
            end
        end
    end

    # 3. Physics & geometry 
    flat_dist = vec(raw_dist)
    flat_area = vec(raw_area)
    flat_acc  = vec(raw_acc)

    # Calculate width ≈ C * sqrt(Accumulation)
    # Convert accumulation from m2 -> km2 for the formula
    acc_km2 = flat_acc ./ ft(1.0e6)

    # Width formula: 7 * sqrt(Area_km2). Clamp to [2m, 2000m]
    # Handle NaNs by defaulting to minimum width
    flat_width = map(x -> isnan(x) ? ft(2.0) : clamp(ft(7.0) * sqrt(x), ft(2.0), ft(2000.0)), acc_km2)

    # Slope is constructed from elevation difference over distance
    flat_elev = FloatType.(vec(elev))
    flat_slope = fill(MIN_SLOPE, n_total)
    
    for i in 1:n_total
        dest_idx = cpu_downstream[i]
        
        if dest_idx != -1
            dist = flat_dist[i]
            
            # Simple downstream slope calculation bounded by minimum slope and missing data
            if dist > ft(0) && !isnan(dist) && !isnan(flat_elev[i]) && !isnan(flat_elev[dest_idx])
                s = (flat_elev[i] - flat_elev[dest_idx]) / dist
                flat_slope[i] = max(s, MIN_SLOPE)
            end
        end
    end

    # 4. Allocate GPU state 
    println("  -> Allocating Routing State on GPU...")

    r_state = RoutingState(
        # Transfer CPU arrays to GPU
        ArrayType(cpu_downstream), 
        ArrayType(flat_dist),      
        ArrayType(flat_slope),
        ArrayType(flat_width),
        ArrayType(flat_area),
        ArrayType(flat_acc),
        
        # State vectors initialized to zero
        # alloc(n_total) defaults to FloatType
        alloc(n_total), # area_gpu
        alloc(n_total), # discharge_gpu
        alloc(n_total), # travel_time_gpu
        alloc(n_total), # cfl_gpu
        alloc(n_total), # inflow_current
        alloc(n_total), # inflow_next
        
        # Violation (of max speed) counter (1-element Int32 array)
        alloc(Int32, 1)
    )

    println("  -> Routing Initialized Successfully.")
    return r_state
end