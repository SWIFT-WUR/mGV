@kernel function surface_temp_kernel!(
    tsurf_val, T_soil_1, T_soil_2, albedo, 
    swdown, lwdown, ra, 
    kap, D_1, D_2, D_3, Cs_val, total_et_val, Ta, psurf, 
    delta_t
)
    (; T_FREEZE, PA_PER_KPA, EMISSIVITY, SIGMA, RHO_W) = Constants
    I = @index(Global)

    # --- Empirical Newton-Raphson Constants ---
    Hvap_Tb = 2.26f6
    Tb      = 373.15f0
    Tc      = 647.096f0
    n_exp   = 0.38f0
    denom_L = Tc - Tb
    
    # --- Atmospheric Calculations ---
    Ta_K = Ta[I] + T_FREEZE # Using T_FREEZE from PhysicalConstants 
    
    # Air Density using PA_PER_KPA 
    air_dens = 0.003486f0 * psurf[I] * PA_PER_KPA / (T_FREEZE + Ta[I]) 
    
    # Combined soil depths 
    D_combined_1 = D_1[I] + D_2[I]
    D_combined_2 = D_3[I]

    # Heat Transfer 
    term_A = kap[I] / D_combined_2 + Cs_val[I] * D_combined_2 / (2f0 * delta_t)
    denom_ht = 1f0 + (D_combined_1 / D_combined_2) + (Cs_val[I] * D_combined_1 * D_combined_2 / (2f0 * delta_t * kap[I]))
    ht_term = term_A / denom_ht

    # Soil temp terms (L2 and L3) 
    T1_K = T_soil_1[I] + T_FREEZE
    T2_K = T_soil_2[I] + T_FREEZE
    num_t6 = (kap[I] * T2_K / D_combined_2) + (Cs_val[I] * D_combined_2 * T1_K / (2f0 * delta_t))
    term6  = num_t6 / denom_ht

    # Air resistance and storage using C_P_AIR 
    z_a = 10f0
    air_storage = (air_dens * C_P_AIR * z_a) / (2f0 * delta_t)
    air_cond    = air_dens * C_P_AIR / max(ra[I], 1f-3)

    term5 = air_storage * (tsurf_val[I] + T_FREEZE)

    # --- Energy Balance (RHS) ---
    # Using EMISSIVITY and swdown/lwdown directly 
    RHS_const = (1f0 - albedo[I]) * swdown[I] + EMISSIVITY * lwdown[I] + air_cond * Ta_K + term5 + term6
    LHS_coeff = ht_term + air_cond + air_storage
    et_factor = total_et_val[I] / (delta_t * 1f3) 

    # --- Newton-Raphson Loop ---
    current_tsurf = tsurf_val[I]
    
    for i in 1:3
        Tk = current_tsurf + T_FREEZE
        
        # Latent Heat of Vaporization using RHO_W 
        term4 = RHO_W * (2.501f6 - 2370f0 * current_tsurf) * et_factor
        
        # Function value using SIGMA 
        f_val = (EMISSIVITY * SIGMA * (Tk^4) + LHS_coeff * Tk) - (RHS_const - term4)
        
        # Derivative 
        if Tk < Tc
            ratio = max((Tc - Tk) / denom_L, 1f-6)
            lv_deriv = Hvap_Tb * n_exp * (ratio ^ (n_exp - 1f0)) * (-1f0 / denom_L)
        else
            lv_deriv = 0f0
        end
        
        df_val = 4f0 * EMISSIVITY * SIGMA * (Tk^3) + LHS_coeff - (RHO_W * lv_deriv * et_factor)
        
        # Step 
        step = (abs(df_val) >= 1f-10) ? (f_val / df_val) : 0f0
        step = clamp(step, -10f0, 10f0)
        
        current_tsurf = clamp(current_tsurf - step, -100f0, 100f0)
    end
    
    tsurf_val[I] = (current_tsurf <= -99f0 || current_tsurf >= 99f0) ? 0f0 : current_tsurf
end

"""
Compute the grid-cell albedo and effective aerodynamic resistance, weighted by
the snow band area fraction and vegetation fraction of every tile.
"""
@kernel function grid_albedo_resistance_kernel!(
    albedo_grid, ra_eff,
    @Const(AreaFract), @Const(cv), @Const(albedo), @Const(ra)
)
    i, j = @index(Global, NTuple)

    acc_albedo = 0f0
    acc_ra_inv = 0f0
    for v in 1:size(ra, 4), b in 1:size(ra, 3)
        w = AreaFract[i, j, b] * cv[i, j, 1, v]
        acc_albedo += w * albedo[i, j, 1, v]
        acc_ra_inv += w / max(ra[i, j, b, v], 1f-9)
    end

    albedo_grid[i, j] = acc_albedo
    ra_eff[i, j] = 1f0 / max(acc_ra_inv, 1f-9)
end

"""

"""
function update_surface_temperature!(model)

    (;
        surface_temperature, aerodynamic_resistance, total_evapotranspiration,
        grid_albedo, grid_aerodynamic_resistance
    ) = model.surface_energy_variables
    (; air_temperature, surface_pressure, shortwave_down, longwave_down) = model.forcing_variables
    (; thermal_conductivity, heat_capacity) = model.soil_variables
    (; depth) = model.soil_parameters
    (; vegetation_fraction) = model.vegetation_parameters
    (; snow_band_area_fraction) = model.grid_parameters

    soil_temperature = model.soil_variables.temperature
    albedo = model.vegetation_parameters.albedo

    # 1. Calculate weighted albedo correctly across all tiles (Veg + Soil)
    # This ensures the bare soil albedo is included 
    # 2. Calculate ra_eff correctly (Inverse weighted sum)
    grid_albedo_resistance_kernel!(device_backend)(
        grid_albedo, grid_aerodynamic_resistance,
        snow_band_area_fraction, vegetation_fraction, albedo, aerodynamic_resistance;
        ndrange = size(surface_temperature)
    )

    # 3. Call the kernel
    kernel = surface_temp_kernel!(device_backend)
    
    @views kernel(
        surface_temperature,
        soil_temperature[:,:,2],
        soil_temperature[:,:,3],
        grid_albedo,
        shortwave_down,
        longwave_down,
        grid_aerodynamic_resistance,
        thermal_conductivity[:,:,1],
        depth[:,:,1],
        depth[:,:,2],
        depth[:,:,3],
        heat_capacity[:,:,1],
        total_evapotranspiration,
        air_temperature,
        surface_pressure,
        Int32(model.clock.dt.value),
        ndrange = length(surface_temperature)
    )
end
