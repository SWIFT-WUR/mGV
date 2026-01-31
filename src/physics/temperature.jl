@inline function surface_temp_kernel(
    tsurf_val, T_soil_1, T_soil_2, albedo, Rs, RL, ra, 
    kap, D_1, D_2, D_3, Cs_val, total_et_val, Ta, psurf, 
    delta_t 
)
    # --- Constants (Strict Float32 via f0) ---
    # No casting needed if these are written as f0
    Sigma   = 5.67f-8
    Emis    = 0.98f0 
    Hvap_Tb = 2.26f6
    Tb      = 373.15f0
    Tc      = 647.096f0
    n_exp   = 0.38f0
    denom_L = Tc - Tb        # Calculated once at compile time
    Rho_w   = 1000.0f0 
    Cp_air  = 1004.0f0 
    Kelvin  = 273.15f0

    # --- Calculations ---
    Ta_K = Ta + Kelvin
    
    # Air Density
    air_dens = 0.003486f0 * psurf * 1000.0f0 / (Kelvin + Ta)
    
    D_combined_1 = D_1 + D_2
    D_combined_2 = D_3

    # Heat Transfer Terms
    term_A = (kap / D_combined_2) + (Cs_val * D_combined_2 / (2.0f0 * delta_t))
    
    # Note: 1.0f0 ensures the whole denominator is Float32
    denom_ht = 1.0f0 + (D_combined_1 / D_combined_2) + (Cs_val * D_combined_1 * D_combined_2 / (2.0f0 * delta_t * kap))
    
    ht_term = term_A / denom_ht

    # Soil temp terms
    T1_K = T_soil_1 + Kelvin
    T2_K = T_soil_2 + Kelvin
    num_t6 = (kap * T2_K / D_combined_2) + (Cs_val * D_combined_2 * T1_K / (2.0f0 * delta_t))
    term6  = num_t6 / denom_ht

    # Air terms
    z_a = 10.0f0
    air_storage = (air_dens * Cp_air * z_a) / (2.0f0 * delta_t)
    air_cond    = air_dens * Cp_air / max(ra, 1f-3)

    term5 = air_storage * (tsurf_val + Kelvin)

    RHS_const = (1.0f0 - albedo) * Rs + Emis * RL + air_cond * Ta_K + term5 + term6
    LHS_coeff = ht_term + air_cond + air_storage
    et_factor = total_et_val / (delta_t * 1000.0f0)

    # --- Newton-Raphson Loop ---
    current_tsurf = tsurf_val
    
    for i in 1:3
        Tk = current_tsurf + Kelvin
        
        # Latent Heat
        term4 = Rho_w * (2.501f6 - 2370.0f0 * current_tsurf) * et_factor
        
        # Function Value
        f_val = (Emis * Sigma * (Tk^4) + LHS_coeff * Tk) - (RHS_const - term4)
        
        # Derivative calculation
        if Tk < Tc
            ratio = max((Tc - Tk) / denom_L, 1f-6)
            lv_deriv = Hvap_Tb * n_exp * (ratio ^ (n_exp - 1.0f0)) * (-1.0f0 / denom_L)
        else
            lv_deriv = 0.0f0
        end
        
        df_val = 4.0f0 * Emis * Sigma * (Tk^3) + LHS_coeff - (Rho_w * lv_deriv * et_factor)
        
        # Step with safety check
        step = (abs(df_val) >= 1f-10) ? (f_val / df_val) : 0.0f0
        step = clamp(step, -10.0f0, 10.0f0)
        
        current_tsurf = clamp(current_tsurf - step, -100.0f0, 100.0f0)
    end
    
    return (current_tsurf <= -99.0f0 || current_tsurf >= 99.0f0) ? 0.0f0 : current_tsurf
end

# --- 2. The Optimized Driver Function ---
function solve_surface_temperature!(
    tsurf, 
    soil_temperature, albedo_gpu, Rs, RL, 
    aerodynamic_resistance, 
    kappa, depth_gpu, delta_t, Cs, total_et, 
    T_a, cv_gpu, psurf_gpu
)

    @assert delta_t isa Float32 "delta_t must be Float32"
    @assert eltype(tsurf) == Float32

    # 1. Albedo Reduction
    # (Matches existing logic)
    albedo_grid = sum_with_nan_handling(cv_gpu .* albedo_gpu, 4)
    @. albedo_grid = ifelse(isnan(albedo_grid) | (abs(albedo_grid) > 1f30), 0f0, albedo_grid)
    
    # 2. Aerodynamic Resistance Reduction (MOVED HERE)
    # Calculate effective inverse resistance: sum(cv / ra)
    # We use Float32 literals and avoid explicit T() casts
    ra_eff_inv = sum(cv_gpu ./ aerodynamic_resistance, dims=4)
    
    # Invert back to resistance: ra_eff = 1 / ra_eff_inv
    # Add epsilon to avoid division by zero
    ra_eff = 1.0f0 ./ max.(ra_eff_inv, 1f-9)
    
    # Sanitize RA in-place
    @. ra_eff = ifelse(isnan(ra_eff) | (abs(ra_eff) > 1f30), 0f0, ra_eff)

    # 3. THE MEGA-BROADCAST
    # We pass views of the 3D/4D arrays so the kernel sees 2D inputs.
    @views @. tsurf = surface_temp_kernel(
        tsurf,                      
        soil_temperature[:,:,2],    
        soil_temperature[:,:,3],    
        albedo_grid,                
        Rs, RL, ra_eff,             # Using the locally calculated ra_eff
        kappa[:,:,1],               
        depth_gpu[:,:,1],           
        depth_gpu[:,:,2],           
        depth_gpu[:,:,3],           
        Cs[:,:,1],                  
        total_et,                   
        T_a,                        
        psurf_gpu,       
        delta_t            
    )

    return nothing
end



function estimate_layer_temperature!(soil_temperature, depth_gpu, dp_gpu, tsurf, Tavg_gpu)
    
    # Define views for clarity
    T_L1 = @view soil_temperature[:, :, 1]
    T_L2 = @view soil_temperature[:, :, 2]
    T_L3 = @view soil_temperature[:, :, 3]
    
    D_L2 = @view depth_gpu[:, :, 2]
    D_L3 = @view depth_gpu[:, :, 3]

    # --- 1. Update Layer 3 ---
    # Must be done FIRST because it depends on the OLD values of L1 and L2.
    # We inline the calculation of top_avg = (L1 + L2) * 0.5
    @. T_L3 = Tavg_gpu - (dp_gpu / D_L3) * (((T_L1 + T_L2) * 0.5f0) - Tavg_gpu) * (exp(-(D_L2 + D_L3) / dp_gpu) - exp(-D_L2 / dp_gpu))

    # --- 2. Update Layer 1 ---
    # We inline top_avg again. 
    # This is safe: for every pixel, (T_L1 + T_L2) reads the OLD values before T_L1 is overwritten.
    @. T_L1 = 0.5f0 * (tsurf + ((T_L1 + T_L2) * 0.5f0))

    # --- 3. Update Layer 2 ---
    # L2 is modeled identically to L1, so we just copy the NEW L1 values.
    @. T_L2 = T_L1
    
    return nothing
end