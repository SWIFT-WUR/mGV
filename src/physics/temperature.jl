@inline function surface_temp_kernel(
    tsurf_val, T_soil_1, T_soil_2, albedo, 
    swdown, lwdown, ra, 
    kap, D_1, D_2, D_3, Cs_val, total_et_val, Ta, psurf, 
    delta_t 
)
    # --- Empirical Newton-Raphson Constants ---
    Hvap_Tb = ft(2.26e6)
    Tb      = ft(373.15)
    Tc      = ft(647.096)
    n_exp   = ft(0.38)
    denom_L = Tc - Tb
    
    # --- Atmospheric Calculations ---
    Ta_K = Ta + t_freeze # Using t_freeze from PhysicalConstants 
    
    # Air Density using pa_per_kpa 
    air_dens = ft(0.003486) * psurf * pa_per_kpa / (t_freeze + Ta) 
    
    # Combined soil depths 
    D_combined_1 = D_1 + D_2
    D_combined_2 = D_3

    # Heat Transfer 
    term_A = (kap / D_combined_2) + (Cs_val * D_combined_2 / (ft(2.0) * delta_t))
    denom_ht = ft(1.0) + (D_combined_1 / D_combined_2) + (Cs_val * D_combined_1 * D_combined_2 / (ft(2.0) * delta_t * kap))
    ht_term = term_A / denom_ht

    # Soil temp terms (L2 and L3) 
    T1_K = T_soil_1 + t_freeze
    T2_K = T_soil_2 + t_freeze
    num_t6 = (kap * T2_K / D_combined_2) + (Cs_val * D_combined_2 * T1_K / (ft(2.0) * delta_t))
    term6  = num_t6 / denom_ht

    # Air resistance and storage using c_p_air 
    z_a = ft(10.0)
    air_storage = (air_dens * c_p_air * z_a) / (ft(2.0) * delta_t)
    air_cond    = air_dens * c_p_air / max(ra, ft(1.0e-3))

    term5 = air_storage * (tsurf_val + t_freeze)

    # --- Energy Balance (RHS) ---
    # Using emissivity and swdown/lwdown directly 
    RHS_const = (ft(1.0) - albedo) * swdown + emissivity * lwdown + air_cond * Ta_K + term5 + term6
    LHS_coeff = ht_term + air_cond + air_storage
    et_factor = total_et_val / (delta_t * ft(1000.0)) 

    # --- Newton-Raphson Loop ---
    current_tsurf = tsurf_val
    
    for i in 1:3
        Tk = current_tsurf + t_freeze
        
        # Latent Heat of Vaporization using rho_w 
        term4 = rho_w * (ft(2.501e6) - ft(2370.0) * current_tsurf) * et_factor
        
        # Function Value using sigma 
        f_val = (emissivity * sigma * (Tk^4) + LHS_coeff * Tk) - (RHS_const - term4)
        
        # Derivative 
        if Tk < Tc
            ratio = max((Tc - Tk) / denom_L, ft(1.0e-6))
            lv_deriv = Hvap_Tb * n_exp * (ratio ^ (n_exp - ft(1.0))) * (-ft(1.0) / denom_L)
        else
            lv_deriv = ft(0.0)
        end
        
        df_val = ft(4.0) * emissivity * sigma * (Tk^3) + LHS_coeff - (rho_w * lv_deriv * et_factor)
        
        # Step 
        step = (abs(df_val) >= ft(1.0e-10)) ? (f_val / df_val) : ft(0.0)
        step = clamp(step, -ft(10.0), ft(10.0))
        
        current_tsurf = clamp(current_tsurf - step, -ft(100.0), ft(100.0))
    end
    
    return (current_tsurf <= -ft(99.0) || current_tsurf >= ft(99.0)) ? ft(0.0) : current_tsurf
end

function solve_surface_temperature!(
    tsurf, tsurf_old,
    soil_temperature, albedo_gpu, swdown_gpu, lwdown_gpu,
    aerodynamic_resistance, 
    kappa, depth_gpu, delta_t, Cs, total_et, 
    T_a, cv_gpu, psurf_gpu
)
    # Store the previous timestep's tsurf for the Newton-Raphson initial guess
    tsurf_old .= tsurf
    
    # Zero out the accumulator array
    fill!(tsurf, eltype(tsurf)(0.0))
    
    nveg = size(cv_gpu, 4)
    
    # Compute tsurf iteratively for each vegetation tile and accumulate the area-weighted average
    for v in 1:nveg
        @views @. tsurf += cv_gpu[:, :, 1, v] * surface_temp_kernel(
            tsurf_old,                      
            soil_temperature[:,:,2],    
            soil_temperature[:,:,3],    
            albedo_gpu[:,:,1,v],                
            swdown_gpu, lwdown_gpu, aerodynamic_resistance[:,:,1,v],
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
    @. T_L3 = Tavg_gpu - (dp_gpu / D_L3) * (((T_L1 + T_L2) * ft(0.5)) - Tavg_gpu) * (exp(-(D_L2 + D_L3) / dp_gpu) - exp(-D_L2 / dp_gpu))

    # --- 2. Update Layer 1 ---
    # We inline top_avg again. 
    # This is safe: for every pixel, (T_L1 + T_L2) reads the OLD values before T_L1 is overwritten.
    @. T_L1 = ft(0.5) * (tsurf + ((T_L1 + T_L2) * ft(0.5)))

    # --- 3. Update Layer 2 ---
    # L2 is modeled identically to L1, so we just copy the NEW L1 values.
    @. T_L2 = T_L1
    
    return nothing
end