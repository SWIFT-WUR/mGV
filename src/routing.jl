const MIN_SLOPE = 1f-4    # Minimum channel slope [m/m]
const MANNING_N = 3.5f-2
const ROUTING_DT = 600f0  # timestep in seconds
const MAX_RIVER_VELOCITY = 6f0  # Cap at 6.0 m/s so wave celerity (5/3 * v) is max 10.0 m/s

using Atomix: @atomic


struct RoutingState{F <: AbstractVector, I <:AbstractVector}
    # --- Topography ---
    # Downstream index is integer
    downstream_idx::I

    # Static parameters (geometry)
    length::F
    slope::F
    width::F
    cell_area::F
    accumulation::F

    # --- State ---
    area::F
    discharge::F
    travel_time::F

    # --- CFL Diagnostics ---
    cfl::F

    # --- Buffers ---
    inflow_current::F
    inflow_next::F

    # --- Diagnostic counter ---
    violation_counter::I
end

@adapt_structure RoutingState

function RoutingState(config, elevation)
    param_file = config.input.paths.routing_param_file
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
    raw_dist = load_safe("cell_dist", Float32, NaN32)
    raw_area = load_safe("cell_area", Float32, NaN32)
    raw_acc = load_safe("accumulation", Float32, NaN32)

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
    acc_km2 = flat_acc ./ 1f6

    # Width formula: 7 * sqrt(Area_km2). Clamp to [2m, 2000m]
    # Handle NaNs by defaulting to minimum width
    flat_width = map(x -> isnan(x) ? 2f0 : clamp(7f0 * sqrt(x), 2f0, 2f3), acc_km2)

    # Slope is constructed from elevation difference over distance
    flat_elev = Float32.(vec(elevation))
    flat_slope = fill(MIN_SLOPE, n_total)
    
    for i in 1:n_total
        dest_idx = cpu_downstream[i]
        
        if dest_idx != -1
            dist = flat_dist[i]
            
            # Simple downstream slope calculation bounded by minimum slope and missing data
            if dist > 0f0 && !isnan(dist) && !isnan(flat_elev[i]) && !isnan(flat_elev[dest_idx])
                s = (flat_elev[i] - flat_elev[dest_idx]) / dist
                flat_slope[i] = max(s, MIN_SLOPE)
            end
        end
    end

    # 4. Allocate GPU state 
    println("  -> Allocating Routing State on GPU...")

    r_state = RoutingState(
        cpu_downstream, 
        Float32.(flat_dist),      
        Float32.(flat_slope),
        Float32.(flat_width),
        Float32.(flat_area),
        Float32.(flat_acc),
        # State vectors initialized to zero
        # alloc(n_total) defaults to FloatType
        zeros(Float32, n_total), # area
        zeros(Float32, n_total), # discharge
        zeros(Float32, n_total), # travel_time
        zeros(Float32, n_total), # cfl
        zeros(Float32, n_total), # inflow_current
        zeros(Float32, n_total), # inflow_next
        # Violation (of max speed) counter (1-element Int32 array)
        zeros(Int32,1)
    )

    println("  -> Routing Initialized Successfully.")
    return r_state
end

@kernel function kinematic_wave_kernel!(area, discharge, inflow_next, inflow_current,
    cfl_buffer, travel_time_buffer,
    runoff_forcing_flat, downstream_idx, lengths,
    slopes, widths, cell_areas, dt, n)
    # violation_counter)

    # Backend-agnostic indexing
    i = @index(Global, Linear)

    if i <= n
        # Load Data
        A_old = area[i] # Water (area) stored in channel (in this cell)
        Q_old = discharge[i]
        Q_in = inflow_current[i] # Inflow from upstream cell(s)

        # Lateral inflow from runoff
        # 1/86400000 (mm/day -> m/s) is roughly 1.15740741e-8 
        runoff_m3s = (runoff_forcing_flat[i] * cell_areas[i]) * 1.15740741f-8
        Q_total_in = Q_in + runoff_m3s

        # Mass balance
        dAdt = (Q_total_in - Q_old) / lengths[i] # Rate of change of stored water
        A_new = max(A_old + dAdt * dt, 0f0) # Update amount of water in channel

        # Momentum (Manning's equation)
        width = widths[i]
        slope = slopes[i]
        alpha = (sqrt(slope) / MANNING_N) * (width^-0.66666667f0) # -2/3
        Q_new = alpha * (A_new^1.66666667f0)  # 5/3

        # Velocity capping
        # We calculate the theoretical velocity
        v = 0f0
        if A_new > 1f-6
            v = Q_new / A_new
            
            # Check against the limit             
            if v > MAX_RIVER_VELOCITY
                v = MAX_RIVER_VELOCITY # Clamp velocity
                Q_new = v * A_new # Recalculate Q to maintain mass consistency (Q = V * A)
                # KernelAbstractions.@atomic violation_counter[1] += 1 # Diagnostic
                # violation_counter += 1
            end
        end

        # CFL and travel time     
        current_cfl = 0f0
        t_time = NaN32

        if A_new > 1f-4 # Threshold for "water exists"
            v = Q_new / A_new # Speed of the water
            c = 1.66666667f0 * v # Wave celerity c = 5/3 * v
            current_cfl = (c * dt) / lengths[i]

            # Calculate travel time
            if v > 1f-6
                t_time = lengths[i] / v
            end
        end

        cfl_buffer[i] = current_cfl
        travel_time_buffer[i] = t_time

        # Update states
        area[i] = A_new
        discharge[i] = Q_new

        # Scatter / routing
        dest = downstream_idx[i]

        # Atomic add for safety since multiple upstream cells
        # can simultaneously flow into the same downstream cell
        if dest > 0
            @atomic inflow_next[dest] += Q_new
        end
    end
end

function update_routing!(model)
    (; routing) = model
    (; total_runoff) = model.soil_variables
    dt = Int32(model.clock.dt.value)
    
    routing.violation_counter .= 0
    n_pixels = Int32(length(routing.downstream_idx))
    n_substeps = Int(ceil(dt / ROUTING_DT))
    dt_step = dt / Float32(n_substeps)
    runoff_flat = reshape(total_runoff, :)

    kernel_launcher! = kinematic_wave_kernel!(device_backend)

    for t in 1:n_substeps

        kernel_launcher!(
            routing.area,
            routing.discharge,
            routing.inflow_next,
            routing.inflow_current,
            routing.cfl,
            routing.travel_time,
            runoff_flat,
            routing.downstream_idx,
            routing.length,
            routing.slope,
            routing.width,
            routing.cell_area,
            dt_step,
            n_pixels,
            # routing.violation_counter;
            ndrange=n_pixels # Define the total number of items to process
        )

        # Ensure GPU finishes this step before we swap buffers
        KernelAbstractions.synchronize(device_backend)

        # # Diagnostics: copy the single integer from GPU to CPU to print it
        # if routing.violation_counter[1] > 0
        #     @warn "Velocity capped!" substep=t count=routing.violation_counter[1] max_allowed=MAX_RIVER_VELOCITY
        # end

        copyto!(routing.inflow_current, routing.inflow_next)
        fill!(routing.inflow_next, 0f0)
    end
    return nothing
end
