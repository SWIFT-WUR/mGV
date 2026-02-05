function initialize_routing_model(param_file)
    println("Initializing Kinematic Wave Routing...")
    println("  -> Source: $param_file")

    if !isfile(param_file)
        error("Routing parameter file not found: $param_file")
    end
    
    ds = NCDataset(param_file)
    
    # --- 1. Load Raw Data (CPU) ---
    # We use a helper to replace 'missing' (FillValue) with safe sentinels.
    
    # Helper: Load and sanitize
    function load_safe(varname, T, fallback)
        data = ds[varname][:,:]
        return T.(replace(data, missing => fallback))
    end

    println("  -> Loading variables...")
    # IDs: Unique identifier for each cell. Missing = -1.
    raw_ids = load_safe("downstream_id", Int32, -1)
    
    # Targets: The ID of the cell downstream. Missing = -1.
    raw_target = load_safe("downstream", Int32, -1)
    
    # Physics parameters. Missing = NaN.
    raw_dist = load_safe("cell_dist", Float32, NaN32)
    raw_area = load_safe("cell_area", Float32, NaN32)
    raw_acc  = load_safe("accumulation", Float32, NaN32)

    close(ds)
    
    nx, ny = size(raw_ids)
    n_total = nx * ny
    println("  -> Grid: $nx x $ny ($n_total pixels)")

    # --- 2. Build Topology (Lookup Table) ---
    # VIC IDs are arbitrary integers. We must map ID -> Linear Index.
    println("  -> Building connectivity graph...")
    
    id_to_index = Dict{Int32, Int32}()
    
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
        
        # If target is valid and exists in our grid, save the index.
        # If target is -1 (ocean) or not found (outside domain), keep as -1.
        if target_val != -1 && haskey(id_to_index, target_val)
            dest_idx = id_to_index[target_val]
            
            # Prevent self-loops (infinite accumulation)
            if dest_idx != i
                cpu_downstream[i] = dest_idx
            end
        end
    end

    # --- 3. Physics & Geometry ---
    flat_dist = vec(raw_dist)
    flat_area = vec(raw_area)
    flat_acc  = vec(raw_acc)

    # Calculate Width ≈ C * sqrt(Accumulation)
    # Convert accumulation from m2 -> km2 for the formula
    acc_km2 = flat_acc ./ 1.0f6
    
    # Width formula: 7 * sqrt(Area_km2). Clamp to [2m, 2000m].
    # Handle NaNs by defaulting to minimum width
    flat_width = map(x -> isnan(x) ? 2.0f0 : clamp(7.0f0 * sqrt(x), 2.0f0, 2000.0f0), acc_km2)

    # Slope defaults to MIN_SLOPE (defined in routing.jl)
    flat_slope = fill(MIN_SLOPE, n_total)

    # --- 4. Allocate GPU State ---
    println("  -> Allocating Routing State on GPU...")
    
    r_state = RoutingState(
        CuArray(cpu_downstream),
        CuArray(flat_dist),
        CuArray(flat_slope),
        CuArray(flat_width),
        CuArray(flat_area),
        CuArray(flat_acc),             # Store Accumulation
        CUDA.zeros(Float32, n_total),  # area_gpu
        CUDA.zeros(Float32, n_total),  # discharge_gpu
        CUDA.zeros(Float32, n_total),  # travel_time_gpu       
        CUDA.zeros(Float32, n_total),  # cfl_gpu               
        CUDA.zeros(Float32, n_total),  # river_temperature     
        CUDA.zeros(Float32, n_total),  # energy_inflow_current 
        CUDA.zeros(Float32, n_total),  # energy_inflow_next    
        CUDA.zeros(Float32, n_total),  # inflow_current
        CUDA.zeros(Float32, n_total)   # inflow_next

    )
    
    println("  -> Routing Initialized Successfully.")
    return r_state
end