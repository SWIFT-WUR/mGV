const MIN_SLOPE = 1.0f-4    # Minimum channel slope [m/m]
const MANNING_N = 3.5f-2
const ROUTING_DT = 600.0f0  # timestep in seconds
const MAX_RIVER_VELOCITY = 6.0f0  # Cap at 6.0 m/s so wave celerity (5/3 * v) is max 10.0 m/s

struct RoutingState{F <: AbstractVector, I <: AbstractVector}
    # --- Topography ---
    # Downstream index is integer
    downstream_idx::I
    # Upstream cells of cell i, in CSR format: upstream_idx[upstream_ptr[i]:(upstream_ptr[i + 1] - 1)]
    upstream_ptr::I
    upstream_idx::I

    # Static parameters (geometry)
    length::F
    slope::F
    width::F
    cell_area::F
    accumulation::F
    alpha::F  # Manning coefficient sqrt(slope) / n * width^(-2/3), Q = alpha * A^(5/3)

    # --- State ---
    area::F
    discharge::F
    travel_time::F

    # --- CFL Diagnostics ---
    cfl::F

    # --- Buffers ---
    discharge_next::F  # Discharge of the next sub-step, swapped with discharge
    lateral_inflow::F  # Runoff into the channel [m3/s], constant over the sub-steps

    # --- Diagnostic counter ---
    violation_counter::I
end

@adapt_structure RoutingState

"""
Initialize the routing for the active cells of `domain`. The connectivity and slopes are
built on the full grid, so a cell that flows into an inactive cell (e.g. the ocean) gets
its slope from that cell's elevation, and then becomes an outlet.
"""
function RoutingState(config, domain::Domain)
    param_file = config.input.paths.routing_param_file
    println("Initializing Kinematic Wave Routing...")
    println("  -> Source: $param_file")

    ds = open_param_source(param_file)

    # 1. Load raw data (CPU)
    # Helper: Load and sanitize
    function load_safe(varname, T, fallback)
        return T.(readfull(ds, varname, fallback))
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

    close_param_source(ds)

    # Elevation of all grid cells, including the inactive ones
    ds_params = open_param_source(config.input.paths.input_param_file)
    elevation = readfull(ds_params, config.input.names.elevation, 0.0)
    close_param_source(ds_params)

    nx, ny = size(raw_ids)
    (nx, ny) == domain.grid_size ||
        error("Routing grid of size $((nx, ny)) does not match the parameter grid $(domain.grid_size)")
    n_total = nx * ny
    println("  -> Grid: $nx x $ny ($n_total pixels), $(ncells(domain)) active")

    # 2. Build Topology (with lookup table)
    # We must map ID -> Linear Index.
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
    flat_acc = vec(raw_acc)

    # Calculate width ≈ C * sqrt(Accumulation)
    # Convert accumulation from m2 -> km2 for the formula
    acc_km2 = flat_acc ./ 1.0f6

    # Width formula: 7 * sqrt(Area_km2). Clamp to [2m, 2000m]
    # Handle NaNs by defaulting to minimum width
    flat_width = map(x -> isnan(x) ? 2.0f0 : clamp(7.0f0 * sqrt(x), 2.0f0, 2.0f3), acc_km2)

    # Slope is constructed from elevation difference over distance
    flat_elev = Float32.(vec(elevation))
    flat_slope = fill(MIN_SLOPE, n_total)

    for i in 1:n_total
        dest_idx = cpu_downstream[i]

        if dest_idx != -1
            dist = flat_dist[i]

            # Simple downstream slope calculation bounded by minimum slope and missing data
            if dist > 0.0f0 &&
                    !isnan(dist) &&
                    !isnan(flat_elev[i]) &&
                    !isnan(flat_elev[dest_idx])
                s = (flat_elev[i] - flat_elev[dest_idx]) / dist
                flat_slope[i] = max(s, MIN_SLOPE)
            end
        end
    end

    # Manning coefficient only depends on static geometry, so compute it once here
    # instead of in every routing sub-step
    flat_alpha = @. (sqrt(flat_slope) / MANNING_N) * (flat_width^-0.66666667f0) # -2/3

    # 4. Keep the active cells. Cells flowing into an inactive cell become outlets.
    (; cells) = domain
    cell_index = zeros(Int32, n_total)  # index of each grid cell among the active cells, 0 if inactive
    cell_index[cells] .= 1:length(cells)
    downstream = [
        d == -1 || cell_index[d] == 0 ? Int32(-1) : cell_index[d] for d in cpu_downstream[cells]
    ]

    # Upstream list, so each cell can gather its inflow without atomics.
    # Upstream cells are in increasing index order, which fixes the summation order.
    upstream_ptr, upstream_idx = upstream_csr(downstream)
    n_cells = length(cells)

    # 5. Allocate GPU state
    println("  -> Allocating Routing State on GPU...")

    r_state = RoutingState(
        downstream,
        upstream_ptr,
        upstream_idx,
        Float32.(flat_dist[cells]),
        Float32.(flat_slope[cells]),
        Float32.(flat_width[cells]),
        Float32.(flat_area[cells]),
        Float32.(flat_acc[cells]),
        Float32.(flat_alpha[cells]),
        # State vectors initialized to zero
        zeros(Float32, n_cells), # area
        zeros(Float32, n_cells), # discharge
        zeros(Float32, n_cells), # travel_time
        zeros(Float32, n_cells), # cfl
        zeros(Float32, n_cells), # discharge_next
        zeros(Float32, n_cells), # lateral_inflow
        # Violation (of max speed) counter (1-element Int32 array)
        zeros(Int32, 1),
    )

    println("  -> Routing Initialized Successfully.")
    return r_state
end

"""
Upstream cells of every cell, from the `downstream` index of each cell (-1 for outlets),
in CSR format: the upstream cells of cell `i` are `idx[ptr[i]:(ptr[i + 1] - 1)]`, in
increasing order.
"""
function upstream_csr(downstream::Vector{Int32})
    n = length(downstream)
    counts = zeros(Int32, n)
    for d in downstream
        d == -1 || (counts[d] += 1)
    end
    ptr = Int32.(cumsum([1; counts]))
    idx = zeros(Int32, ptr[end] - 1)
    next = ptr[1:n]
    for (i, d) in enumerate(downstream)
        d == -1 && continue
        idx[next[d]] = i
        next[d] += 1
    end
    return ptr, idx
end

@kernel function lateral_inflow_kernel!(lateral_inflow, runoff_forcing_flat, cell_areas, n)
    i = @index(Global, Linear)

    if i <= n
        # 1/86400000 (mm/day -> m/s) is roughly 1.15740741e-8
        lateral_inflow[i] = (runoff_forcing_flat[i] * cell_areas[i]) * 1.15740741f-8
    end
end

@kernel function kinematic_wave_kernel!(
        area,
        discharge_next,
        @Const(discharge),
        lateral_inflow,
        upstream_ptr,
        upstream_idx,
        lengths,
        alphas,
        dt,
        n,
    )
    # violation_counter)

    # Backend-agnostic indexing
    i = @index(Global, Linear)

    if i <= n
        # Load Data
        A_old = area[i] # Water (area) stored in channel (in this cell)
        Q_old = discharge[i]

        # Gather inflow from upstream cell(s), from the previous sub-step.
        # No atomics, and a fixed summation order, so results are reproducible.
        Q_in = 0.0f0
        for k in upstream_ptr[i]:(upstream_ptr[i + 1] - Int32(1))
            # In bounds: upstream_csr only stores indices of existing cells
            Q_in += @inbounds discharge[upstream_idx[k]]
        end

        # Lateral inflow from runoff
        Q_total_in = Q_in + lateral_inflow[i]

        # Mass balance
        dAdt = (Q_total_in - Q_old) / lengths[i] # Rate of change of stored water
        A_new = max(A_old + dAdt * dt, 0.0f0) # Update amount of water in channel

        # Momentum (Manning's equation)
        # A_new^(5/3) as A_new * cbrt(A_new)^2: ~5x cheaper than pow on GPU
        Q_new = alphas[i] * (A_new * cbrt(A_new)^2)

        # Velocity capping
        # We calculate the theoretical velocity
        if A_new > 1.0f-6
            v = Q_new / A_new

            # Check against the limit
            if v > MAX_RIVER_VELOCITY
                v = MAX_RIVER_VELOCITY # Clamp velocity
                Q_new = v * A_new # Recalculate Q to maintain mass consistency (Q = V * A)
                # KernelAbstractions.@atomic violation_counter[1] += 1 # Diagnostic
                # violation_counter += 1
            end
        end

        # Update states
        area[i] = A_new
        discharge_next[i] = Q_new
    end
end

# CFL and travel time, from the state after the last sub-step
@kernel function routing_diagnostics_kernel!(
        cfl_buffer,
        travel_time_buffer,
        area,
        discharge,
        lengths,
        dt,
        n,
    )
    i = @index(Global, Linear)

    if i <= n
        A = area[i]
        current_cfl = 0.0f0
        t_time = NaN32

        if A > 1.0f-4 # Threshold for "water exists"
            v = discharge[i] / A # Speed of the water
            c = 1.66666667f0 * v # Wave celerity c = 5/3 * v
            current_cfl = (c * dt) / lengths[i]

            # Calculate travel time
            if v > 1.0f-6
                t_time = lengths[i] / v
            end
        end

        cfl_buffer[i] = current_cfl
        travel_time_buffer[i] = t_time
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
    backend = get_backend(total_runoff)

    # Runoff is constant over the sub-steps, so convert it to channel inflow once
    lateral_inflow_kernel!(backend)(
        routing.lateral_inflow, total_runoff, routing.cell_area, n_pixels; ndrange = n_pixels
    )

    kernel_launcher! = kinematic_wave_kernel!(backend)

    # Swap discharge buffers: each sub-step reads the previous discharge and writes the next
    discharge, discharge_next = routing.discharge, routing.discharge_next
    for t in 1:n_substeps

        kernel_launcher!(
            routing.area,
            discharge_next,
            discharge,
            routing.lateral_inflow,
            routing.upstream_ptr,
            routing.upstream_idx,
            routing.length,
            routing.alpha,
            dt_step,
            n_pixels,
            # routing.violation_counter;
            ndrange = n_pixels, # Define the total number of items to process
        )

        # # Diagnostics: copy the single integer from GPU to CPU to print it
        # if routing.violation_counter[1] > 0
        #     @warn "Velocity capped!" substep=t count=routing.violation_counter[1] max_allowed=MAX_RIVER_VELOCITY
        # end

        discharge, discharge_next = discharge_next, discharge
    end

    # With an odd number of sub-steps the latest discharge ends up in routing.discharge_next
    if discharge !== routing.discharge
        copyto!(routing.discharge, discharge)
    end

    routing_diagnostics_kernel!(backend)(
        routing.cfl, routing.travel_time, routing.area, routing.discharge, routing.length,
        dt_step, n_pixels; ndrange = n_pixels
    )
    return nothing
end
