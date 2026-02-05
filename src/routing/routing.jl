# src/routing/routing.jl

const MIN_SLOPE   = 0.0001f0    # Minimum channel slope [m/m]
const MANNING_N   = 0.035f0
const ROUTING_DT  = 900.0f0 #timestep in seconds

struct RoutingState
    # --- Topography ---
    downstream_idx::CuArray{Int32, 1}
    length_gpu::CuArray{Float32, 1}
    slope_gpu::CuArray{Float32, 1}
    width_gpu::CuArray{Float32, 1}
    cell_area_gpu::CuArray{Float32, 1}
    accumulation_gpu::CuArray{Float32, 1}
    
    # --- State ---
    area_gpu::CuArray{Float32, 1}
    discharge_gpu::CuArray{Float32, 1}
    travel_time_gpu::CuArray{Float32, 1}

    # --- CFL Diagnostics ---
    cfl_gpu::CuArray{Float32, 1}    
    
    # --- Buffers ---
    inflow_current::CuArray{Float32, 1} 
    inflow_next::CuArray{Float32, 1}
end

function kinematic_wave_kernel!(area, discharge, inflow_next, inflow_current, 
                                cfl_buffer, travel_time_buffer,
                                runoff_forcing_flat, downstream_idx, lengths, 
                                slopes, widths, cell_areas, dt::Float32, n::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > n; return; end

    # [Load Data]
    A_old = area[i]
    Q_old = discharge[i]
    Q_in  = inflow_current[i]
    
    # [Lateral Inflow]
    runoff_m3s = (runoff_forcing_flat[i] * cell_areas[i]) * 1.15740741f-8 
    Q_total_in = Q_in + runoff_m3s

    # [Continuity]
    dAdt  = (Q_total_in - Q_old) / lengths[i]
    A_new = max(A_old + dAdt * dt, 0.0f0)

    # [Momentum]
    width = widths[i]
    slope = slopes[i]
    alpha = (sqrt(slope) / MANNING_N) * (width ^ -0.6666666f0)
    Q_new = alpha * (A_new ^ 1.6666666f0)

    # --- CFL & TRAVEL TIME CALCULATION ---    
    # Celerity c = dQ/dA = (5/3) * v
    current_cfl = 0.0f0
    t_time      = NaN32 # Default to NaN if no water
    if A_new > 1.0f-4 # Avoid division by zero
        v = Q_new / A_new
        c = 1.6666666f0 * v  # 5/3 * v
        current_cfl = (c * dt) / lengths[i]

        # Calculate Travel Time: T = L / v
        # Guard against extremely small velocity to prevent infinity
        if v > 1.0f-6
            t_time = lengths[i] / v
        end
    end

    cfl_buffer[i] = current_cfl
    travel_time_buffer[i] = t_time

    # [Update State]
    area[i]      = A_new
    discharge[i] = Q_new

    # [Scatter]
    dest = downstream_idx[i]
    if dest > 0
        CUDA.atomic_add!(pointer(inflow_next, dest), Q_new)
    end
    return nothing
end

function run_routing_step!(r_state::RoutingState, total_runoff_mm, dt_day_sec)
    n_pixels = length(r_state.downstream_idx)
    n_substeps = Int(ceil(dt_day_sec / ROUTING_DT))
    dt_step    = Float32(dt_day_sec) / Float32(n_substeps)

    runoff_flat = reshape(total_runoff_mm, :)
    threads = 256
    blocks  = cld(n_pixels, threads)

    for t in 1:n_substeps
        @cuda threads=threads blocks=blocks kinematic_wave_kernel!(
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
            n_pixels
        )
        
        # --- CFL CHECK ---
        # Find max CFL in the grid (GPU Reduction)
        max_cfl = maximum(r_state.cfl_gpu)
        
        if max_cfl > 1.0f0
            @warn "CFL Violation Detected!" substep=t max_courant=max_cfl threshold=1.0
            # Optional: You could break or throw error here
        end

        copyto!(r_state.inflow_current, r_state.inflow_next)
        fill!(r_state.inflow_next, 0.0f0)
    end
    return nothing
end