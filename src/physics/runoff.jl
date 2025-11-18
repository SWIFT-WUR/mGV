function calculate_surface_runoff(prec_gpu, throughfall, soil_moisture_old, soil_moisture_max, b_i, cv_gpu)
    T   = eltype(soil_moisture_old)
    EPS = T(1e-9)
    
    # Sum top soil layers (equivalent to C code's loop over Nlayer-1)
    topsoil_moisture     = sum(soil_moisture_old[:, :, 1:2], dims=3)[:, :, 1]
    topsoil_moisture_max = sum(soil_moisture_max[:, :, 1:2], dims=3)[:, :, 1]
    
    # Clamp topsoil_moisture to not exceed max (as in C code)
    topsoil_moisture = min.(topsoil_moisture, topsoil_moisture_max)
    
    # Calculate ratio (add small epsilon to denominator to avoid division by zero)
    ratio = topsoil_moisture ./ (topsoil_moisture_max .+ EPS)
    ratio = clamp.(ratio, T(0), T(1))
    
    # Calculate A (saturated area) - USE CORRECT EXPONENT!
    # C code: ex = b_infilt / (1.0 + b_infilt)
    ex = b_i ./ (T(1) .+ b_i)
    A_sat = T(1) .- ((T(1) .- ratio) .^ ex)
    
    # Maximum infiltration capacity
    max_infil = (T(1) .+ b_i) .* topsoil_moisture_max
    
    # Initial infiltration
    i_0 = max_infil .* (T(1) .- ((T(1) .- A_sat) .^ (T(1) ./ b_i)))
    
    # Total water input (inflow in C code) - CONVERT TO CORRECT TYPE
    inflow = T.(sum_with_nan_handling(throughfall, 4))
    
    # Calculate runoff following C code logic exactly
    # Equation 3a: runoff = inflow - top_max_moist + top_moist
    runoff_3a = inflow .- topsoil_moisture_max .+ topsoil_moisture
    
    # Equation 3b: runoff = inflow - top_max_moist + top_moist + 
    #                       top_max_moist * (1 - (i_0 + inflow)/max_infil)^(1 + b_infilt)
    max_infil_safe = max.(max_infil, EPS)
    basis = T(1) .- (i_0 .+ inflow) ./ max_infil_safe
    runoff_3b = inflow .- topsoil_moisture_max .+ topsoil_moisture .+ 
                topsoil_moisture_max .* (basis .^ (T(1) .+ b_i))
    
    # Apply conditional logic matching C code
    runoff = ifelse.(inflow .<= EPS,
                     T(0),                                    # inflow == 0
                     ifelse.(max_infil .<= EPS,
                            inflow,                           # max_infil == 0
                            ifelse.((i_0 .+ inflow) .> max_infil,
                                   runoff_3a,                 # Eq. 3a
                                   runoff_3b)))               # Eq. 3b
    
    # Clamp to non-negative and not exceed input
    runoff = clamp.(runoff, T(0), inflow)
    
    return runoff, A_sat
end


function calculate_subsurface_runoff(soil_moisture_old, soil_moisture_max, Ds_gpu, Dsmax_gpu, Ws_gpu)
    bottomsoil_moisture = soil_moisture_old[:, :, 3:3]  # W_2^-[N+1], shape (204, 180, 1)
    bottomsoil_moisture_max = soil_moisture_max[:, :, 3:3]   # W_2^c, 
    Ws_fraction = Ws_gpu .* bottomsoil_moisture_max         # W_s * W_2^c, shape (204, 180, 1)

    # Initialize subsurface runoff (Q_b * Δt, assuming Δt = 1 day)
    Q_b = CUDA.zeros(float_type, size(bottomsoil_moisture, 1), size(bottomsoil_moisture, 2), size(bottomsoil_moisture, 3))

    # Compute subsurface runoff using ifelse for Eq. 21a and 21b
    Q_b = ifelse.(
        bottomsoil_moisture .<= Ws_fraction,
        # Eq. 21a: Linear drainage
        (Ds_gpu .* Dsmax_gpu ./ Ws_fraction) .* bottomsoil_moisture,
        # Eq. 21b: Nonlinear drainage
        (Ds_gpu .* Dsmax_gpu ./ Ws_fraction) .* bottomsoil_moisture .+
        (Dsmax_gpu .- (Ds_gpu .* Dsmax_gpu ./ Ws_gpu)) .* 
        ((bottomsoil_moisture .- Ws_fraction) ./ (bottomsoil_moisture_max .- Ws_fraction)) .^ 2
    )

    Q_b = max.(Q_b, 0.0)     # Ensure non-negative runoff

    return Q_b
end

# Eq. (24): Total runoff
function calculate_total_runoff(surface_runoff, subsurface_runoff, cv_gpu)

    surface_runoff .= ifelse.(isnan.(surface_runoff) .| (abs.(surface_runoff) .> fillvalue_threshold), 0.0, surface_runoff) # Q_d[n]
    subsurface_runoff .= ifelse.(isnan.(subsurface_runoff) .| (abs.(subsurface_runoff) .> fillvalue_threshold), 0.0, subsurface_runoff) # Q_b[n]


    # Sum surface and subsurface runoff, weighted by coverage
 #   total_runoff = sum_with_nan_handling(cv_gpu .* (surface_runoff .+ subsurface_runoff), 4) #./ 14. # C_v[n]
    total_runoff = (surface_runoff .+ subsurface_runoff) # TODO: with or without cv_gpu?

    return total_runoff
end