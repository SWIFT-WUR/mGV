function solve_surface_temperature(tsurf, soil_temperature, albedo, Rs, RL, rh, kappa, depth_gpu, delta_t, Cs, total_et, T_a, cv_gpu, psurf_gpu)

    #### --- Preprocessing / NaN handling (same logic as original) --- ####

    albedo = sum_with_nan_handling(cv_gpu .* albedo, 4)
    albedo .= ifelse.(isnan.(albedo) .| (abs.(albedo) .> 1e30), 0.0, albedo)

    rh    .= ifelse.(isnan.(rh)    .| (abs.(rh)    .> 1e30), 0.0, rh)
    kappa .= ifelse.(isnan.(kappa) .| (abs.(kappa) .> 1e30), 0.0, kappa)
    Cs    .= ifelse.(isnan.(Cs)    .| (abs.(Cs)    .> 1e30), 0.0, Cs)

    # (optional) average of layers 1–2
    # T1_K = (soil_temperature[:, :, 1:1] .+ soil_temperature[:, :, 2:2]) ./ 2 .+ 273.15
    T1_K = soil_temperature[:, :, 2:2] .+ 273.15
    T2_K = soil_temperature[:, :, 3:3] .+ 273.15
    T1_K .= ifelse.(isnan.(T1_K) .| (abs.(T1_K) .> 1e15), 0.0, T1_K)
    T2_K .= ifelse.(isnan.(T2_K) .| (abs.(T2_K) .> 1e15), 0.0, T2_K)

    D1 = sum(depth_gpu[:, :, 1:2], dims = 3)
    D2 = depth_gpu[:, :, 3:3]

    T_a_K = T_a .+ 273.15
    T_a_K .= ifelse.(isnan.(T_a_K) .| (abs.(T_a_K) .> 1e15), 0.0, T_a_K)

    kappa_top = kappa[:, :, 1]
    Cs_top    = Cs[:, :, 1]

    base_term = (kappa_top ./ D2) .+ (Cs_top .* D2 ./ (2 * delta_t))
    heat_transfer_term = base_term ./ (1 .+ (D1 ./ D2) .+ (Cs_top .* D1 .* D2 ./ (2.0 * delta_t .* kappa_top)))

    # Precompute constant “layer 1–2” conductive term (term6)
    num = (kappa_top .* T2_K ./ D2) .+ (Cs_top .* D2 .* T1_K ./ (2 * delta_t))
    den = 1 .+ (D1 ./ D2) .+ (Cs_top .* D1 .* D2 ./ (2 * delta_t .* kappa_top))
    term6_const = num ./ den

    #### --- Air-layer storage term (eq. 35) --- ####

    z_a = 10.0
    air_dens    = (0.003486) .* (psurf_gpu) .* pa_per_kpa ./ ((273.15) .+ (T_a))
    air_storage = (air_dens .* c_p_air .* z_a) ./ (2.0 * delta_t)

    air_term    = (air_dens .* c_p_air ./ max.(rh, 1e-3)) .+ air_storage
    common_term = heat_transfer_term .+ air_term

    #### --- Terms that do not depend on Ts_new --- ####

    term1 = (1 .- albedo) .* Rs
    term2 = emissivity .* RL
    term3 = (air_dens .* c_p_air ./ max.(rh, 1e-3)) .* T_a_K

    Ts_prev = tsurf           # previous time step Ts (fixed during Newton)
    Ts_old  = Ts_prev
    Ts_new  = tsurf

    Ts_old_K = Ts_old .+ 273.15

    term5 = air_storage .* Ts_old_K

    et_flux = total_et ./ (delta_t .* mm_in_m)

    #### --- constants for latent-heat derivative --- ####

    Hvap_Tb = 2.26e6
    Tb      = 373.15
    Tc      = 647.096
    n       = 0.38
    denom_L = Tc - Tb

    #### --- residual and derivative (no nested broadcast of calculate_latent_heat) --- ####

    function f(Ts_new, Ts_old)
        Ts_new_K = Ts_new .+ 273.15
        Ts_old_K = Ts_old .+ 273.15

        lhs = emissivity .* sigma .* Ts_new_K.^4 .+ common_term .* Ts_new_K

        # IMPORTANT: call calculate_latent_heat separately to avoid GPU nested broadcast
        L_v   = calculate_latent_heat(Ts_new_K)       # same function you use elsewhere
        term4 = rho_w .* L_v .* et_flux               # now a simple broadcast

        return lhs .- (term1 .+ term2 .+ term3 .- term4 .+ term5 .+ term6_const)
    end

    function df_dTs_new(Ts_new)
        Ts_new_K = Ts_new .+ 273.15

        ratio = (Tc .- Ts_new_K) ./ denom_L
        ratio = max.(ratio, 1e-6)

        L_v_deriv = ifelse.(Ts_new_K .< Tc,
                            Hvap_Tb .* n .* (ratio .^ (n - 1)) .* (-1 ./ denom_L),
                            0.0)

        et_flux = total_et ./ (delta_t .* mm_in_m)
        term4_deriv = rho_w .* L_v_deriv .* et_flux

        return 4.0 .* emissivity .* sigma .* Ts_new_K.^3 .+ common_term .- term4_deriv
    end

    #### --- Newton iterations (same logic as your original) --- ####

    tolerance = 1e-3
    max_iter  = 3

    for iter in 1:max_iter
        residual   = f(Ts_new, Ts_old)   # Ts_old stays fixed = previous time step
        derivative = df_dTs_new(Ts_new)

        delta_Ts = ifelse.(abs.(derivative) .>= 1e-10, residual ./ derivative, 0.0)
        delta_Ts = clamp.(delta_Ts, -10.0, 10.0)

        converged = abs.(delta_Ts) .< tolerance
        if all(converged)
            @debug println("Converged after $iter iterations"); break
        end

        delta_Ts = ifelse.(converged, 0.0, delta_Ts)
        Ts_new   = clamp.(Ts_new .- delta_Ts, -100.0, 100.0)

        @debug println("Iteration $iter: Number of converged points = ", sum(converged))
        @debug println("Iteration $iter: Ts_new min/max: ", minimum(Ts_new), " / ", maximum(Ts_new))
        # NOTE: do NOT update Ts_old here
    end

    Ts_new = ifelse.((Ts_new .== -100.0) .| (Ts_new .== 100.0), 0.0, Ts_new)
    return Ts_new
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