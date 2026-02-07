const MIN_SLOPE = ft(0.0001)    # Minimum channel slope [m/m]
const MANNING_N = ft(0.035)
const ROUTING_DT = ft(28800.0)  # timestep in seconds
const MAX_RIVER_VELOCITY = ft(6.0)  # Cap at 6.0 m/s so wave celerity (5/3 * v) is max 10.0 m/s

struct RoutingState
    # --- Topography ---
    # Downstream index is integer
    downstream_idx::ArrayType{Int32,1}

    # Static parameters (geometry)
    length_gpu::ArrayType{FloatType,1}
    slope_gpu::ArrayType{FloatType,1}
    width_gpu::ArrayType{FloatType,1}
    cell_area_gpu::ArrayType{FloatType,1}
    accumulation_gpu::ArrayType{FloatType,1}

    # --- State ---
    area_gpu::ArrayType{FloatType,1}
    discharge_gpu::ArrayType{FloatType,1}
    travel_time_gpu::ArrayType{FloatType,1}

    # --- CFL Diagnostics ---
    cfl_gpu::ArrayType{FloatType,1}

    # --- Buffers ---
    inflow_current::ArrayType{FloatType,1}
    inflow_next::ArrayType{FloatType,1}

    # --- Diagnostic counter ---
    violation_counter::ArrayType{Int32,1}
end

@kernel function kinematic_wave_kernel!(area, discharge, inflow_next, inflow_current,
    cfl_buffer, travel_time_buffer,
    runoff_forcing_flat, downstream_idx, lengths,
    slopes, widths, cell_areas, dt, n,
    violation_counter)

    # Backend-agnostic indexing
    i = @index(Global, Linear)

    if i <= n
        # Load Data
        A_old = area[i] # Water (area) stored in channel (in this cell)
        Q_old = discharge[i]
        Q_in = inflow_current[i] # Inflow from upstream cell(s)

        # Lateral inflow from runoff
        # 1/86400000 (mm/day -> m/s) is roughly 1.15740741e-8 
        runoff_m3s = (runoff_forcing_flat[i] * cell_areas[i]) * ft(1.15740741e-8)
        Q_total_in = Q_in + runoff_m3s

        # Mass balance
        dAdt = (Q_total_in - Q_old) / lengths[i] # Rate of change of stored water
        A_new = max(A_old + dAdt * dt, ft(0)) # Update amount of water in channel

        # Momentum (Manning's equation)
        width = widths[i]
        slope = slopes[i]
        alpha = (sqrt(slope) / MANNING_N) * (width^ft(-0.66666667)) # -2/3
        Q_new = alpha * (A_new^ft(1.66666667))  # 5/3

        # Velocity capping
        # We calculate the theoretical velocity
        v = ft(0)
        if A_new > ft(1.0e-6)
            v = Q_new / A_new
            
            # Check against the limit             
            if v > MAX_RIVER_VELOCITY
                v = MAX_RIVER_VELOCITY # Clamp velocity
                Q_new = v * A_new # Recalculate Q to maintain mass consistency (Q = V * A)
                KernelAbstractions.@atomic violation_counter[1] += 1 # Diagnostic
            end
        end

        # CFL and travel time     
        current_cfl = ft(0)
        t_time = ft(NaN)

        if A_new > ft(1.0e-4) # Threshold for "water exists"
            v = Q_new / A_new # Speed of the water
            c = ft(1.66666667) * v # Wave celerity c = 5/3 * v
            current_cfl = (c * dt) / lengths[i]

            # Calculate travel time
            if v > ft(1.0e-6)
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
            KernelAbstractions.@atomic inflow_next[dest] += Q_new
        end
    end
end

function run_routing_step!(r_state::RoutingState, total_runoff_mm, dt_day_sec)
    
    n_pixels = length(r_state.downstream_idx)
    n_substeps = Int(ceil(dt_day_sec / ROUTING_DT))
    dt_step = Float32(dt_day_sec) / Float32(n_substeps)
    runoff_flat = reshape(total_runoff_mm, :)

    kernel_launcher! = kinematic_wave_kernel!(device_backend)

    for t in 1:n_substeps
        # We call the 'launcher', not the original function name
        kernel_launcher!(
            r_state.area_gpu,
            r_state.discharge_gpu,
            r_state.inflow_next,
            r_state.inflow_current,
            r_state.cfl_gpu,
            r_state.travel_time_gpu,
            runoff_flat,
            r_state.downstream_idx,
            r_state.length_gpu,
            r_state.slope_gpu,
            r_state.width_gpu,
            r_state.cell_area_gpu,
            dt_step,
            n_pixels,
            r_state.violation_counter;
            ndrange=n_pixels # Define the total number of items to process
        )

        # Ensure GPU finishes this step before we swap buffers
        KernelAbstractions.synchronize(device_backend)

        # Diagnostics: copy the single integer from GPU to CPU to print it
        n_capped = Array(r_state.violation_counter)[1]
        if n_capped > 0
        #    @warn "Velocity capped!" substep=t count=n_capped max_allowed=MAX_RIVER_VELOCITY
        end

        copyto!(r_state.inflow_current, r_state.inflow_next)
        fill!(r_state.inflow_next, ft(0))
    end
    return nothing
end