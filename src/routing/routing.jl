# src/routing/routing.jl

const MIN_SLOPE   = 0.0001f0    # Minimum channel slope [m/m]
const MANNING_N   = 0.035f0
const ROUTING_DT  = 90.0f0 #timestep in seconds

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

    # --- Thermal State ---
    river_temperature::CuArray{Float32, 1}       # River Temperature [C]
    energy_inflow_current::CuArray{Float32, 1} # Energy entering cell [m3*C/s]
    energy_inflow_next::CuArray{Float32, 1}    # Buffer for next step
    
    # --- Buffers ---
    inflow_current::CuArray{Float32, 1} 
    inflow_next::CuArray{Float32, 1}
end

function kinematic_wave_kernel!(
    area, discharge, river_temperature,
    inflow_next, inflow_current,
    energy_inflow_next, energy_inflow_current,
    cfl_buffer, travel_time_buffer,
    
    # Forcing Inputs (Flattened)
    surface_runoff_flat, subsurface_runoff_flat, 
    tair_flat, tsoil_deep_flat, 
    sw_flat, lw_flat, wind_flat, vp_flat, press_flat,
    
    downstream_idx, lengths, slopes, widths, cell_areas, dt::Float32, n::Int
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > n; return; end

    # [1. Load Water & Energy Inflow from Upstream]
    A_old      = area[i]
    Q_old      = discharge[i]
    Q_river_in = inflow_current[i]
    E_river_in = energy_inflow_current[i] # Units: m3/s * degC

    # [2. Lateral Inflow (Runoff)]
    # Convert mm/day -> m3/s
    # Factor: (Area_m2 / 1000mm) / 86400s = 1.15740741e-8
    factor = cell_areas[i] * 1.15740741f-8
    
    Q_surf = surface_runoff_flat[i] * factor
    Q_sub  = subsurface_runoff_flat[i] * factor
    
    # Determine Runoff Temps
    # Surface runoff assumes Air Temp (fast response)
    T_surf = tair_flat[i] 
    # Subsurface assumes Deep Soil Temp (stable)
    T_sub  = tsoil_deep_flat[i]

    # Calculate Energy from Runoff
    E_runoff = (Q_surf * T_surf) + (Q_sub * T_sub)
    Q_runoff = Q_surf + Q_sub

    # [3. Mixing]
    Q_total_in = Q_river_in + Q_runoff
    E_total_in = E_river_in + E_runoff
    
    # Initial Water Temp (Before Sun hits it)
    # If no water, assume T_air
    T_mix = (Q_total_in > 1f-6) ? (E_total_in / Q_total_in) : T_surf

    # [4. Hydraulic Routing (Continuity)]
    # (Same as your old code)
    dAdt  = (Q_total_in - Q_old) / lengths[i]
    A_new = max(A_old + dAdt * dt, 0.0f0)

    # [5. Momentum (Manning)]
    width = widths[i]
    slope = slopes[i]
    alpha = (sqrt(slope) / MANNING_N) * (width ^ -0.6666666f0)
    Q_new = alpha * (A_new ^ 1.6666666f0)

    # [6. Energy Balance (Solar Heating)]
    # Calculate depth (Hydraulic Radius)
    depth = (width > 0f0) ? (A_new / width) : 0.0f0
    
    if depth > 0.01f0
        # Calculate Flux [W/m2] using river_physics.jl
        Phi_net = calculate_river_heat_flux(
            T_mix, tair_flat[i], 
            sw_flat[i], lw_flat[i], 
            wind_flat[i], press_flat[i], vp_flat[i]
        )
        
        # Calculate Temperature Change
        # dT = (Flux * dt) / (rho * Cp * depth)
        dT = (Phi_net * dt) / (rho_w * c_p_water * depth)
        
        T_new = T_mix + dT
    else
        T_new = T_surf # Dry bed -> Air temp
    end

    # [7. CFL & Travel Time]
    # (Same as your old code)
    if A_new > 1.0f-4
        v = Q_new / A_new
        current_cfl = (1.6666666f0 * v * dt) / lengths[i]
        cfl_buffer[i] = current_cfl
        
        if v > 1.0f-6
            travel_time_buffer[i] = lengths[i] / v
        else
            travel_time_buffer[i] = NaN32
        end
    else
        cfl_buffer[i] = 0f0
        travel_time_buffer[i] = NaN32
    end

    # [8. Store State]
    area[i]              = A_new
    discharge[i]         = Q_new
    river_temperature[i] = T_new

    # [9. Scatter (Water + Energy)]
    dest = downstream_idx[i]
    if dest > 0
        CUDA.atomic_add!(pointer(inflow_next, dest), Q_new)
        
        # Advect Energy: Q_new * T_new
        E_out = Q_new * T_new
        CUDA.atomic_add!(pointer(energy_inflow_next, dest), E_out)
    end
    return nothing
end

function run_routing_step!(
    r_state::RoutingState, 
    surface_runoff, subsurface_runoff, # Separate inputs!
    tair, tsoil_deep, sw, lw, wind, vp, press, # Forcing
    dt_day_sec
)
    n_pixels = length(r_state.downstream_idx)
    n_substeps = Int(ceil(dt_day_sec / ROUTING_DT))
    dt_step    = Float32(dt_day_sec) / Float32(n_substeps)

    threads = 256
    blocks  = cld(n_pixels, threads)

    # Flatten inputs
    surf_flat  = reshape(surface_runoff, :)
    sub_flat   = reshape(subsurface_runoff, :)
    tair_flat  = reshape(tair, :)
    tsoil_flat = reshape(tsoil_deep, :)
    sw_flat    = reshape(sw, :)
    lw_flat    = reshape(lw, :)
    wind_flat  = reshape(wind, :)
    vp_flat    = reshape(vp, :)
    press_flat = reshape(press, :)

    for t in 1:n_substeps
        @cuda threads=threads blocks=blocks kinematic_wave_kernel!(
            r_state.area_gpu,
            r_state.discharge_gpu,
            r_state.river_temperature,
            r_state.inflow_next,
            r_state.inflow_current,
            r_state.energy_inflow_next,
            r_state.energy_inflow_current,
            r_state.cfl_gpu,
            r_state.travel_time_gpu,
            # Pass flattened forcing
            surf_flat, sub_flat, tair_flat, tsoil_flat,
            sw_flat, lw_flat, wind_flat, vp_flat, press_flat,

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

        # Swap Water Buffers
        copyto!(r_state.inflow_current, r_state.inflow_next)
        fill!(r_state.inflow_next, 0.0f0)
        
        # Swap Energy Buffers
        copyto!(r_state.energy_inflow_current, r_state.energy_inflow_next)
        fill!(r_state.energy_inflow_next, 0.0f0)

    end
    return nothing
end