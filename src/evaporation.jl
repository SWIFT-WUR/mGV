function canopy_evap_physics(
    ws, max_ws, pot_evap, ra, rarc, 
    prec, lai, tair, elev, rmin
)

    # --- 1. Physics Calculations ---
    # Using your existing project functions
    slope = calculate_svp_slope(tair)
    latent_heat = calculate_latent_heat(tair)

    scale_height = calculate_scale_height(tair, elev)
    surface_pressure = 101325.0f0 * exp(-elev / scale_height)
    gamma_val = 1628.6f0 * surface_pressure / latent_heat

    # --- 2. Resistances ---
    rc = rmin / max(lai, 1f-6)
    inv_ra = 1f0 / max(ra, 1f-6)

    # --- 3. Denominators (Penman-Monteith) ---
    den_w = slope + gamma_val * (1f0 + rarc * inv_ra)
    den_rc = den_w + (gamma_val * rc * inv_ra)

    E_p_wet = pot_evap * (den_rc / max(den_w, 1f-6))

    # --- 4. VIC Evaporative Scaling (Branchless) ---
    # The original VIC model uses conditional logic to evaluate if the canopy is fully wet or dry.
    # We replace that with a continuous mathematical formulation:
    # 
    #   Wratio: Evaluates the current canopy water storage as a fraction of its maximum capacity.
    #           It is safely clamped between 0.0 (completely dry) and 1.0 (fully wet).
    #
    #   ra_ratio: Computes the aerodynamic resistance modifier.
    # 
    #   canopy_evap_star: Calculates the potential canopy evaporation scaled non-linearly 
    #                     (to the 2/3rds power) based on how much water is currently present.
    Wratio = clamp(ws / max(max_ws, 1f-6), 0f0, 1f0)
    ra_ratio = ra / max(ra + rarc, 1f-6)
    
    canopy_evap_star = (Wratio ^ (2f0 / 3f0)) * E_p_wet * ra_ratio

    # --- 5. Fraction Calculation (f_n) ---
    # Determines what fraction of the timestep the canopy will be evaporating.
    # If the available water (storage + new precipitation) is less than the theoretical 
    # evaporation amount (canopy_evap_star), the canopy will dry out partway through the timestep (f_n_val < 1.0).
    f_n_val = clamp((ws + prec) / max(canopy_evap_star, 1f-6), 0f0, 1f0)

    if canopy_evap_star <= 1f-6
        f_n_val = 1f0
    end

    # --- 6. Final Evaporation ---
    # The physical evaporation that occurs during the wet fraction of the timestep.
    evap = f_n_val * canopy_evap_star
    
    # Sanitize outputs to prevent NaN propagation
    if isnan(evap) | (abs(evap) > 1f15)
        evap = 0f0
    end

    return evap, f_n_val
end

function calculate_max_water_storage(lai, canopy_coverage, fillvalue_threshold)
    (; K_L) = Constants

    max_water_storage = ifelse(canopy_coverage > 1.0f-5, (K_L * lai) / canopy_coverage, 0f0)
    max_water_storage = ifelse(
        isnan(max_water_storage) | (abs(max_water_storage) > fillvalue_threshold), 
        0f0, 
        max_water_storage
    )
    return max_water_storage
end

@kernel function canopy_evaporation_kernel!(
    lai,
    canopy_coverage,
    fillvalue_threshold,
    maximum_water_storage,
    canopy_evaporation,
    wet_fraction,
    water_storage,
    potential_evaporation,
    aerodynamic_resistance,
    architectural_resistance,
    precipitation,
    air_temperature,
    elevation,
    minimum_resistance
)
    # Cartesian indexing lets each input keep its natural shape instead of
    # being `repeat`d up to (nx, ny, nbands, nveg), which cost ~13 GiB/step.
    I = @index(Global, Cartesian)
    i, j, band, tile = Tuple(I)

    lai_I = lai[i, j, 1, tile]

    maximum_water_storage[I] = calculate_max_water_storage(
        lai_I, canopy_coverage[i, j, 1, tile], fillvalue_threshold
    )

    canopy_evaporation[I], wet_fraction[I] = canopy_evap_physics(
        water_storage[I],
        maximum_water_storage[I],
        potential_evaporation[I],
        aerodynamic_resistance[I],
        architectural_resistance[i, j, 1, tile],
        precipitation[i, j],
        lai_I,
        air_temperature[i, j],
        elevation[i, j],
        minimum_resistance[i, j, 1, tile]
    )
end

function update_canopy_evaporation!(model::Model)
    (; air_temperature, precipitation) = model.forcing_variables
    (; elevation) = model.grid_parameters
    (; potential_evaporation, aerodynamic_resistance) = model.surface_energy_variables
    (; canopy_evaporation, wet_fraction, water_storage, maximum_water_storage) = model.canopy_variables
    (; lai, canopy_coverage, minimum_resistance, architectural_resistance) = model.vegetation_parameters

    # Inputs keep their natural shapes -- (nx, ny, nbands, nveg) state,
    # (nx, ny, 1, nveg) vegetation parameters, (nx, ny) forcings -- and the
    # kernel broadcasts them via Cartesian indexing.
    canopy_evaporation_kernel!(device_backend)(
        lai,
        canopy_coverage,
        model.config.fillvalue_threshold,
        maximum_water_storage,
        canopy_evaporation,
        wet_fraction,
        water_storage,
        potential_evaporation,
        aerodynamic_resistance,
        architectural_resistance,
        precipitation,
        air_temperature,
        elevation,
        minimum_resistance,
        ndrange = size(canopy_evaporation)
    )

    # 3. Post-Process: Zero out bare soil (last index)
    @view(canopy_evaporation[:, :, :, end]) .= 0f0
    return nothing
end

@kernel function transpiration_kernel!(
    # Outputs
    transpiration_full,
    transpiration_layers,
    # Inputs
    potential_evaporation, 
    water_storage, 
    max_water_storage, 
    soil_moisture,       
    soil_moisture_critical,  
    wilting_point,           
    root_gpu, 
    cv_gpu, 
    f_n,
    AreaFract,
    tair_gpu,    # 2D: air temperature [°C] (for Tfactor)
    vp_gpu       # 2D: vapour pressure [kPa] (for VPDfactor)
)
    i, j = @index(Global, NTuple)

    # Boundary Check
    if i <= size(transpiration_full, 1) && j <= size(transpiration_full, 2)
        
        # Constants
        EPS  = 1f-9
        ZERO = 0f0
        ONE  = 1f0
        
        # --- 1. SOIL STRESS (g1, g2) ---
        # Load Layer 1
        W1   = soil_moisture[i,j,1]
        Wcr1 = soil_moisture_critical[i,j,1]
        Wwp1 = wilting_point[i,j,1]
        
        # Load Layer 2
        W2   = soil_moisture[i,j,2]
        Wcr2 = soil_moisture_critical[i,j,2]
        Wwp2 = wilting_point[i,j,2]

        # g1 = clamp((W1 - Wwp1) / (Wcr1 - Wwp1 + EPS), 0, 1)
        g1 = clamp((W1 - Wwp1) / (Wcr1 - Wwp1 + EPS), ZERO, ONE)
        g2 = clamp((W2 - Wwp2) / (Wcr2 - Wwp2 + EPS), ZERO, ONE)

        # --- 2. VEGETATION LOOP ---
        nveg = size(root_gpu, 4)
        nbands = size(transpiration_full, 3)
        
        for k in 1:nveg
            # Load Root Fractions
            f1 = root_gpu[i,j,1,k]
            f2 = root_gpu[i,j,2,k]
            
            # Branchless root-fraction accumulation
            W_root_sum   = ifelse(f1 > ZERO, W1, ZERO) + ifelse(f2 > ZERO, W2, ZERO)
            Wcr_root_sum = ifelse(f1 > ZERO, Wcr1, ZERO) + ifelse(f2 > ZERO, Wcr2, ZERO)
            
            share_moist = (W_root_sum >= Wcr_root_sum) & (W_root_sum > ZERO)
            
            moist1_wet = W1 >= Wcr1
            moist2_wet = W2 >= Wcr2
            
            g_sw_veg = ifelse(share_moist, ONE, clamp((f1 * g1 + f2 * g2) / (f1 + f2 + EPS), ZERO, ONE))

            e1_total = ZERO
            e2_total = ZERO

            for b in 1:nbands
                # --- Canopy Wetness / Dry Time Factor ---
                ws   = water_storage[i,j,b,k]
                max_ws = max_water_storage[i,j,b,k]
                cv   = cv_gpu[i,j,1,k]
                fn_val = f_n[i,j,b,k]
                pe   = potential_evaporation[i,j,b,k]

                term_inner = clamp((ws / max(cv, EPS)) / max(max_ws, EPS), ZERO, ONE)
                dry_time_factor = clamp(ONE - fn_val * (term_inner ^ (2f0/3f0)), ZERO, ONE)

                dry_time_factor = ifelse(k == nveg, ONE, dry_time_factor)

                # --- Jarvis temperature + VPD factor for transpiration ---
                # Tfactor: reduces transpiration in cold/hot extremes (max=1 at T=25°C)
                # VPDfactor: CANOPY_CLOSURE=13000 Pa. Using tair_grid for PE slope
                # gives better transpiration distribution (95.3% in run49).
                tair_c_v     = tair_gpu[i, j]
                Tfactor_raw  = 0.08f0 * tair_c_v - 0.0016f0 * tair_c_v * tair_c_v
                Tfact        = Tfactor_raw < 1.0f-10 ? 1.0f-10 : (Tfactor_raw > ONE ? ONE : Tfactor_raw)
                svp_val_t    = SVP_A * exp((SVP_B * tair_c_v) / (SVP_C + tair_c_v))  # kPa
                vpd_pa_t     = max(svp_val_t - vp_gpu[i, j], 0f0) * PA_PER_KPA  # Pa
                raw_vpd_t    = 1f0 - vpd_pa_t / 13000.0f0
                VPDfact      = raw_vpd_t < 0.7f0 ? 0.7f0 : (raw_vpd_t > ONE ? ONE : raw_vpd_t)
                rc_factor    = Tfact * VPDfact

                # --- Transpiration Calculation ---
                # Output aggregation in io_writer sums trans_val directly (no extra Cv weight).
                trans_val = clamp(cv * dry_time_factor * pe * g_sw_veg * rc_factor, ZERO, Inf32)
                
                # =========================================================
                # --- Layer Apportionment (E1, E2) without branching ---
                # =========================================================
                # If moisture is limited in one layer, roots can pull from another.
                
                # Setup base root + moisture weights
                weight1      = f1 * g1  # root fraction * moisture stress factor
                weight2      = f2 * g2
                total_weight = weight1 + weight2 + EPS

                # Option A: Standard Weighted Apportionment
                # Used when moisture is severely limited (share_moist is false).
                # Demand is strictly partitioned by the available moisture weight.
                e1_weighted_demand = trans_val * (weight1 / total_weight)
                e2_weighted_demand = trans_val * (weight2 / total_weight)

                # Option B: Shared Moisture Apportionment (Roots redistribute uptake)
                # Used when at least one layer has sufficient moisture (above critical point).
                
                # 1. Base uptake demand under shared moisture:
                # If a layer is fully "wet", it fulfills its entire root fraction demand.
                # If a layer is "dry", it only fulfills a stress-reduced fraction.
                e1_shared_base = ifelse(moist1_wet, trans_val * f1, trans_val * g1 * f1)
                e2_shared_base = ifelse(moist2_wet, trans_val * f2, trans_val * g2 * f2)
                
                # 2. Calculate "Spare" ET demand:
                # This is the unmet demand from dry layers. Because roots are connected,
                # the plant will try to pull this missing water from the wet layers instead.
                spare_demand_from_1 = ifelse(moist1_wet, ZERO, trans_val * f1 * (ONE - g1))
                spare_demand_from_2 = ifelse(moist2_wet, ZERO, trans_val * f2 * (ONE - g2))
                total_spare_demand  = spare_demand_from_1 + spare_demand_from_2
                
                # 3. Calculate wet-layer capacity to receive the spare demand
                # We identify which layers have surplus moisture capacity to fulfill the spare demand.
                wet_root_frac_sum    = ifelse(moist1_wet, f1, ZERO) + ifelse(moist2_wet, f2, ZERO)
                can_distribute_spare = (total_spare_demand > ZERO) & (wet_root_frac_sum > ZERO)
                
                # 4. Add the redistributed spare demand back into the wet layers proportionately
                # The spare demand is sliced up according to the root mass present in the wet layers.
                spare_share_e1 = ifelse(moist1_wet & can_distribute_spare, total_spare_demand * (f1 / max(wet_root_frac_sum, EPS)), ZERO)
                spare_share_e2 = ifelse(moist2_wet & can_distribute_spare, total_spare_demand * (f2 / max(wet_root_frac_sum, EPS)), ZERO)
                
                e1_shared_total = e1_shared_base + spare_share_e1
                e2_shared_total = e2_shared_base + spare_share_e2
                
                # Finally, select between Option A and Option B seamlessly using a branchless ternary.
                # This eliminates massive GPU thread divergence caused by unpredictable if/else block stalling.
                e1_val = ifelse(share_moist, e1_shared_total, e1_weighted_demand)
                e2_val = ifelse(share_moist, e2_shared_total, e2_weighted_demand)

                trans_val = ifelse(k == nveg, ZERO, trans_val)
                e1_val    = ifelse(k == nveg, ZERO, e1_val)
                e2_val    = ifelse(k == nveg, ZERO, e2_val)

                # --- WRITE OUTPUTS ---
                transpiration_full[i,j,b,k] = trans_val
                
                # Accumulate layers
                e1_total += e1_val * AreaFract[i,j,b]
                e2_total += e2_val * AreaFract[i,j,b]
            end
            
            # 2. Layer distributed transpiration
            transpiration_layers[i,j,1,k] = e1_total
            transpiration_layers[i,j,2,k] = e2_total
            transpiration_layers[i,j,3,k] = ZERO
        end
    end
end


function update_transpiration!(model::Model)

    (; air_temperature, vapor_pressure) = model.forcing_variables
    (; potential_evaporation) = model.surface_energy_variables
    (; 
        water_storage, maximum_water_storage, transpiration, 
        transpiration_layers, wet_fraction
    ) = model.canopy_variables
    (; root_fraction, vegetation_fraction) = model.vegetation_parameters
    (; snow_band_area_fraction) = model.grid_parameters
    (; critical_moisture, wilting_point) = model.soil_parameters
    soil_moisture = model.soil_variables.moisture

    # 1. Configuration
    kernel_launcher! = transpiration_kernel!(device_backend)    
    nx, ny = size(transpiration)

    # 2. Launch
    kernel_launcher!(
        transpiration, 
        transpiration_layers,
        potential_evaporation, 
        water_storage, 
        maximum_water_storage, 
        soil_moisture, 
        critical_moisture, 
        wilting_point, 
        root_fraction, 
        vegetation_fraction, 
        wet_fraction,
        snow_band_area_fraction,
        air_temperature,
        vapor_pressure;
        ndrange = (nx, ny)
    )

    return nothing
end

function update_water_canopy_storage!(model::Model)
    # Band-adjusted precipitation (Pfactor / AreaFract), not the grid-cell mean
    (; band_precipitation) = model.snow_variables
    (; water_storage, maximum_water_storage, throughfall, canopy_evaporation
    ) = model.canopy_variables

    coverage_this_month = model.vegetation_parameters.canopy_coverage

    # 1. Update Throughfall FIRST
    # We calculate the 'excess' logic on the fly using the *current* (old) water_storage.
    # Logic: excess = max(0, (W + P - E) - Wm)
    # Throughfall = (excess * coverage) + (precipitation * (1 - coverage))
    @. throughfall = (max(0f0, water_storage + band_precipitation - canopy_evaporation - maximum_water_storage) * coverage_this_month) +
                     (band_precipitation * (1f0 - coverage_this_month))

    # 2. Update Water Storage SECOND
    # Now we can safely mutate water_storage.
    # Logic: clamped new storage
    @. water_storage = clamp(water_storage + band_precipitation - canopy_evaporation, 0f0, maximum_water_storage)

    return nothing
end

# Eq. (23): Total evapotranspiration
function update_total_evapotranspiration!(model)
    (; total_evapotranspiration) = model.surface_energy_variables
    (; transpiration, canopy_evaporation) = model.canopy_variables
    soil_evaporation = model.soil_variables.evaporation
    (; vegetation_fraction, canopy_coverage) = model.vegetation_parameters
    (; snow_band_area_fraction) = model.grid_parameters

    coverage_this_month = canopy_coverage

    # 1. Initialize with Soil Evaporation
    @. total_evapotranspiration = soil_evaporation

    # 2. Accumulate Vegetation Fluxes
    # We loop over tiles to avoid allocating massive intermediate arrays.
    for i in 1:size(canopy_evaporation, 4)
        for b in 1:size(canopy_evaporation, 3)
            @views @. total_evapotranspiration += (
                canopy_evaporation[:,:,b,i] * vegetation_fraction[:,:,1,i] + transpiration[:,:,b,i]
            ) * coverage_this_month[:,:,1,i] * snow_band_area_fraction[:,:,b]
        end
    end

    return nothing
end
