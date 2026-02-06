@inline function surface_temp_kernel(
    tsurf_val, T_soil_1, T_soil_2, albedo, 
    swdown, lwdown, ra, 
    kap, D_1, D_2, D_3, Cs_val, total_et_val, Ta, psurf, 
    delta_t 
)
    # --- Empirical Newton-Raphson Constants ---
    Hvap_Tb = 2.26f6
    Tb      = 373.15f0
    Tc      = 647.096f0
    n_exp   = 0.38f0
    denom_L = Tc - Tb
    
    # --- Atmospheric Calculations ---
    Ta_K = Ta + t_freeze # Using t_freeze from PhysicalConstants 
    
    # Air Density using pa_per_kpa 
    air_dens = 0.003486f0 * psurf * pa_per_kpa / (t_freeze + Ta) 
    
    # Combined soil depths 
    D_combined_1 = D_1 + D_2
    D_combined_2 = D_3

    # Heat Transfer 
    term_A = (kap / D_combined_2) + (Cs_val * D_combined_2 / (2.0f0 * delta_t))
    denom_ht = 1.0f0 + (D_combined_1 / D_combined_2) + (Cs_val * D_combined_1 * D_combined_2 / (2.0f0 * delta_t * kap))
    ht_term = term_A / denom_ht

    # Soil temp terms (L2 and L3) 
    T1_K = T_soil_1 + t_freeze
    T2_K = T_soil_2 + t_freeze
    num_t6 = (kap * T2_K / D_combined_2) + (Cs_val * D_combined_2 * T1_K / (2.0f0 * delta_t))
    term6  = num_t6 / denom_ht

    # Air resistance and storage using c_p_air 
    z_a = 10.0f0
    air_storage = (air_dens * c_p_air * z_a) / (2.0f0 * delta_t)
    air_cond    = air_dens * c_p_air / max(ra, 1f-3)

    term5 = air_storage * (tsurf_val + t_freeze)

    # --- Energy Balance (RHS) ---
    # Using emissivity and swdown/lwdown directly 
    RHS_const = (1.0f0 - albedo) * swdown + emissivity * lwdown + air_cond * Ta_K + term5 + term6
    LHS_coeff = ht_term + air_cond + air_storage
    et_factor = total_et_val / (delta_t * 1000.0f0) 

    # --- Newton-Raphson Loop ---
    current_tsurf = tsurf_val
    
    for i in 1:3
        Tk = current_tsurf + t_freeze
        
        # Latent Heat of Vaporization using rho_w 
        term4 = rho_w * (2.501f6 - 2370.0f0 * current_tsurf) * et_factor
        
        # Function Value using sigma 
        f_val = (emissivity * sigma * (Tk^4) + LHS_coeff * Tk) - (RHS_const - term4)
        
        # Derivative 
        if Tk < Tc
            ratio = max((Tc - Tk) / denom_L, 1f-6)
            lv_deriv = Hvap_Tb * n_exp * (ratio ^ (n_exp - 1.0f0)) * (-1.0f0 / denom_L)
        else
            lv_deriv = 0.0f0
        end
        
        df_val = 4.0f0 * emissivity * sigma * (Tk^3) + LHS_coeff - (rho_w * lv_deriv * et_factor)
        
        # Step 
        step = (abs(df_val) >= 1f-10) ? (f_val / df_val) : 0.0f0
        step = clamp(step, -10.0f0, 10.0f0)
        
        current_tsurf = clamp(current_tsurf - step, -100.0f0, 100.0f0)
    end
    
    return (current_tsurf <= -99.0f0 || current_tsurf >= 99.0f0) ? 0.0f0 : current_tsurf
end

function solve_surface_temperature!(
    tsurf, 
    soil_temperature, albedo_gpu, swdown_gpu, lwdown_gpu,
    aerodynamic_resistance, 
    kappa, depth_gpu, delta_t, Cs, total_et, 
    T_a, cv_gpu, psurf_gpu
)
    # 1. Calculate weighted Albedo correctly across ALL tiles (Veg + Soil)
    # This ensures the bare soil albedo is included 
    albedo_grid = sum(cv_gpu .* albedo_gpu, dims=4) 
    
    # 2. Calculate ra_eff correctly (Inverse weighted sum)
    ra_eff_inv = sum(cv_gpu ./ max.(aerodynamic_resistance, 1f-9), dims=4)
    ra_eff = 1.0f0 ./ max.(ra_eff_inv, 1f-9)

    # 3. Call the mega-broadcast
    @views @. tsurf = surface_temp_kernel(
        tsurf,                      
        soil_temperature[:,:,2],    
        soil_temperature[:,:,3],    
        albedo_grid,                
        swdown_gpu, lwdown_gpu, ra_eff,
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