# --- 1. The Scalar Physics Kernel (Compiles to GPU Code) ---
@inline function surface_temp_kernel(
    tsurf_val, T_soil_1, T_soil_2, albedo, Rs, RL, ra, 
    kap, D_1, D_2, D_3, Cs_val, total_et_val, Ta, psurf, 
    delta_t # Scalar arguments
)
    # Type aliases for stability
    F = Float32
    Zero = F(0)
    
    # Constants
    Sigma   = F(5.67e-8)
    Emis    = F(0.98) # Assuming emissivity is const, pass as arg if variable
    Hvap_Tb = F(2.26e6)
    Tb      = F(373.15)
    Tc      = F(647.096)
    n_exp   = F(0.38)
    denom_L = Tc - Tb
    Rho_w   = F(1000.0) # rho_w
    Cp_air  = F(1004.0) # c_p_air
    Kelvin  = F(273.15)

    # --- Pre-Loop Math (Done in Registers) ---
    Ta_K = Ta + Kelvin
    
    # Air Density
    air_dens = F(0.003486) * psurf * F(1000.0) / (Kelvin + Ta) # pa_per_kpa hardcoded
    
    # Depths (D1 in original code was sum of layer 1 and 2 depths?)
    # Based on original: D1 = depth[1] + depth[2], D2 = depth[3]
    D_combined_1 = D_1 + D_2
    D_combined_2 = D_3

    # Heat Transfer Terms
    # base_term = (kap / D2) + ...
    term_A = (kap / D_combined_2) + (Cs_val * D_combined_2 / (2 * delta_t))
    
    # denom_ht = 1 + (D1 / D2) + ...
    denom_ht = 1 + (D_combined_1 / D_combined_2) + (Cs_val * D_combined_1 * D_combined_2 / (2 * delta_t * kap))
    
    ht_term = term_A / denom_ht

    # Num_t6
    T1_K = T_soil_1 + Kelvin
    T2_K = T_soil_2 + Kelvin
    num_t6 = (kap * T2_K / D_combined_2) + (Cs_val * D_combined_2 * T1_K / (2 * delta_t))
    term6  = num_t6 / denom_ht

    # Air terms
    z_a = F(10.0)
    air_storage = (air_dens * Cp_air * z_a) / (2 * delta_t)
    air_cond    = air_dens * Cp_air / max(ra, F(1e-3))

    # Note: original code used tsurf in term5 *before* the loop?
    # "term5 = air_storage .* (tsurf .+ T(273.15))"
    # This implies tsurf is the OLD timestep temperature for the storage term.
    term5 = air_storage * (tsurf_val + Kelvin)

    RHS_const = (1 - albedo) * Rs + Emis * RL + air_cond * Ta_K + term5 + term6
    LHS_coeff = ht_term + air_cond + air_storage
    et_factor = total_et_val / (delta_t * F(1000.0)) # mm_in_m

    # --- Newton-Raphson Loop (In Registers) ---
    current_tsurf = tsurf_val
    
    for i in 1:3
        Tk = current_tsurf + Kelvin
        
        # Term 4 (Latent Heat)
        # L_v calculation inlined
        # term4 = rho_w * (2.501e6 - 2370.0 * (Tk - 273.15)) * et_factor
        term4 = Rho_w * (F(2.501e6) - F(2370.0) * current_tsurf) * et_factor
        
        # Function Value
        f_val = (Emis * Sigma * (Tk^4) + LHS_coeff * Tk) - (RHS_const - term4)
        
        # Derivative
        # Logic for Hvap derivative
        if Tk < Tc
            ratio = max((Tc - Tk) / denom_L, F(1e-6))
            lv_deriv = Hvap_Tb * n_exp * (ratio ^ (n_exp - 1)) * (-1 / denom_L)
        else
            lv_deriv = Zero
        end
        
        df_val = 4 * Emis * Sigma * (Tk^3) + LHS_coeff - (Rho_w * lv_deriv * et_factor)
        
        # Step
        step = (abs(df_val) >= 1e-10) ? (f_val / df_val) : Zero
        step = clamp(step, F(-10.0), F(10.0))
        
        current_tsurf = clamp(current_tsurf - step, F(-100.0), F(100.0))
    end
    
    # Final cleanup check
    if current_tsurf == F(-100.0) || current_tsurf == F(100.0)
        return Zero
    end
    
    return current_tsurf
end

# --- 2. The Optimized Driver Function ---
function solve_surface_temperature!(
    tsurf, 
    soil_temperature, albedo_gpu, Rs, RL, 
    ra_eff, kappa, depth_gpu, delta_t, Cs, total_et, 
    T_a, cv_gpu, psurf_gpu
)
    T = eltype(tsurf)
    
    # 1. Albedo Reduction (Keep this pre-calculation)
    albedo_grid = sum_with_nan_handling(cv_gpu .* albedo_gpu, 4)
    # Sanitize albedo in-place using broadcast
    @. albedo_grid = ifelse(isnan(albedo_grid) | (abs(albedo_grid) > 1e30), 0f0, albedo_grid)
    
    # Sanitize RA in-place
    @. ra_eff = ifelse(isnan(ra_eff) | (abs(ra_eff) > 1e30), 0f0, ra_eff)

    # 2. THE MEGA-BROADCAST
    # We pass views of the 3D/4D arrays so the kernel sees 2D inputs.
    # Note: depth_gpu is (nx, ny, 3)
    #       soil_temperature is (nx, ny, 3)
    #       kappa is (nx, ny, 3) --> Original used kappa[:,:,1]
    #       Cs is (nx, ny, 3)    --> Original used Cs[:,:,1]

    @views @. tsurf = surface_temp_kernel(
        tsurf,                      # tsurf_val (Old)
        soil_temperature[:,:,2],    # T_soil_1 (Layer 2)
        soil_temperature[:,:,3],    # T_soil_2 (Layer 3)
        albedo_grid,                # albedo
        Rs, RL, ra_eff,             # forcings
        kappa[:,:,1],               # kap
        depth_gpu[:,:,1],           # D_1
        depth_gpu[:,:,2],           # D_2
        depth_gpu[:,:,3],           # D_3
        Cs[:,:,1],                  # Cs_val
        total_et,                   # total_et_val
        T_a,                        # Ta
        psurf_gpu,                  # psurf
        Float32(delta_t)            # delta_t (Scalar)
    )

    return nothing
end



function estimate_layer_temperature(depth_gpu, dp_gpu, tsurf, soil_temperature, Tavg_gpu)
    # Based on Liang et al. (1999): Modeling ground heat flux in land surface parameterization schemes

    # Assign inputs
    topsoil_temperature = sum(soil_temperature[:, :, 1:2], dims=3) ./ 2  # Average of layers 1-2

    # Model layer 1 (my layers 1-2): Average of Tsurf and topsoil_temperature
    soil_temperature[:, :, 1:1] = 0.5 .* (tsurf .+ topsoil_temperature)
    soil_temperature[:, :, 2:2] = 0.5 .* (tsurf .+ topsoil_temperature)

    # Model layer 2 (my layer 3)
    soil_temperature[:, :, 3:3] = Tavg_gpu .- (dp_gpu ./ depth_gpu[:, :, 3:3]) .* 
                                   (topsoil_temperature .- Tavg_gpu) .* 
                                   (exp.(-(depth_gpu[:, :, 2:2] .+ depth_gpu[:, :, 3:3]) ./ dp_gpu) .- 
                                    exp.(-depth_gpu[:, :, 2:2] ./ dp_gpu))

    return soil_temperature
end