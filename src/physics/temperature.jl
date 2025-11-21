function solve_surface_temperature!(
    tsurf, # <--- Modified In-Place
    soil_temperature, albedo_gpu, Rs, RL, 
    ra_eff, kappa, depth_gpu, delta_t, Cs, total_et, 
    T_a, cv_gpu, psurf_gpu
)
    T = eltype(tsurf) 

    # --- 1. Preprocessing & Constants ---
    sigma_   = T(5.67f-8)
    emis_    = T(emissivity) 
    Hvap_Tb  = T(2.26e6)
    Tb       = T(373.15)
    Tc       = T(647.096)
    n_exp    = T(0.38)
    denom_L  = Tc - Tb
    rho_w_   = T(rho_w)
    
    # Flatten albedo (4D -> 2D Grid Average)
    albedo_grid = sum_with_nan_handling(cv_gpu .* albedo_gpu, 4)
    albedo_grid .= ifelse.(isnan.(albedo_grid) .| (abs.(albedo_grid) .> T(1e30)), T(0), albedo_grid)
    ra_eff .= ifelse.(isnan.(ra_eff) .| (abs.(ra_eff) .> T(1e30)), T(0), ra_eff)
    
    # --- 2. Soil & Air Terms ---
    @views begin
        T1_K = soil_temperature[:, :, 2] .+ T(273.15)
        T2_K = soil_temperature[:, :, 3] .+ T(273.15)
        D1   = depth_gpu[:, :, 1] .+ depth_gpu[:, :, 2]
        D2   = depth_gpu[:, :, 3]
        kap  = kappa[:, :, 1]
        Cs_t = Cs[:, :, 1]
    end
    
    T_a_K    = T_a .+ T(273.15)
    air_dens = T(0.003486) .* psurf_gpu .* T(pa_per_kpa) ./ (T(273.15) .+ T_a)
    
    # --- 3. Construct Constant Forcing Term ---
    base_term = (kap ./ D2) .+ (Cs_t .* D2 ./ (2 * delta_t))
    denom_ht  = 1 .+ (D1 ./ D2) .+ (Cs_t .* D1 .* D2 ./ (2 * delta_t .* kap))
    ht_term   = base_term ./ denom_ht
    
    num_t6    = (kap .* T2_K ./ D2) .+ (Cs_t .* D2 .* T1_K ./ (2 * delta_t))
    term6     = num_t6 ./ denom_ht
    
    z_a = T(10.0)
    air_storage = (air_dens .* T(c_p_air) .* z_a) ./ (2 * delta_t)
    air_cond    = air_dens .* T(c_p_air) ./ max.(ra_eff, T(1e-3))
    
    term5 = air_storage .* (tsurf .+ T(273.15)) 
    
    RHS_const = (1 .- albedo_grid) .* Rs .+ 
                emis_ .* RL .+ 
                air_cond .* T_a_K .+ 
                term5 .+ 
                term6
                
    LHS_coeff = ht_term .+ air_cond .+ air_storage
    et_factor = total_et ./ (delta_t * mm_in_m)

    # --- 4. PRE-ALLOCATE ALL SCRATCH ARRAYS (Crucial!) ---
    # These prevent UndefVarError and avoid allocation inside the loop
    Tk      = similar(tsurf)  
    step    = similar(tsurf)
    term4   = similar(tsurf)  # Holds Latent Heat Flux
    f_val   = similar(tsurf)  # Holds Residual
    df_val  = similar(tsurf)  # Holds Derivative

    # --- 5. Newton-Raphson Loop ---
    for iter in 1:3
        
        # Step A: Calculate Temperature in Kelvin
        @. Tk = tsurf + T(273.15)

        # Step B: Calculate Terms (Inlined L_v logic to avoid 'L_v' array)
        @. begin
            # 1. Calculate Term4 (Latent Heat Flux) directly
            # L_v = (2.501e6 - 2370.0 * T_c)
            term4 = rho_w_ * (2.501e6 - 2370.0 * (Tk - 273.15)) * et_factor
            
            # 2. Calculate Residual (f)
            f_val = (emis_ * sigma_ * Tk^4 + LHS_coeff * Tk) - (RHS_const - term4)
            
            # 3. Calculate Derivative (df)
            # We inline 'ratio' and 'Lv_deriv' here to save memory
            # ratio = (Tc - Tk) / denom_L
            # Lv_deriv logic handled by ifelse
            # term4_deriv = rho_w * Lv_deriv * et_factor
            
            df_val = 4 * emis_ * sigma_ * Tk^3 + LHS_coeff - (
                rho_w_ * ifelse(Tk < Tc, 
                       Hvap_Tb * n_exp * (max((Tc - Tk) / denom_L, T(1e-6)) ^ (n_exp - 1)) * (-1 / denom_L), 
                       T(0)
                ) * et_factor
            )
            
            # 4. Calculate Step
            step = ifelse(abs(df_val) >= 1e-10, f_val / df_val, T(0))
            step = clamp(step, T(-10.0), T(10.0))
        end
        
        # Step C: Apply Update
        @. tsurf = clamp(tsurf - step, T(-100.0), T(100.0))
    end
    
    # Final Cleanup
    @. tsurf = ifelse((tsurf == -100.0) | (tsurf == 100.0), T(0), tsurf)
    
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