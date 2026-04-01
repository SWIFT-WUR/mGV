# --- Scalar Physics Kernel (Runs on every pixel) ---
function aerodynamic_kernel(z0, d0, tsurf, tair, wind, z2, Kt, gt, Tf, Ric, z_floor, d_floor, w_floor, L2_min, ra_min, ra_max)
    # 1. Roughness & Effective Height
    rough = max(z0, z_floor)
    d_eff = max(z2 - d0, d_floor)

    # 2. Log-law terms
    ratio = clamp(d_eff / rough, ft(1.0e-6), ft(1.0e6))
    L     = log(ratio)
    L2    = max(L^2, L2_min)
    a_sq  = (Kt^2) / L2
    ccoef = ft(49.82) * a_sq * sqrt(ratio)

    # 3. Stability (Richardson Number)
    w_spd = max(wind, w_floor)
    Tmean = max(((tair + Tf) + (tsurf + Tf)) * ft(0.5), ft(100.0))
    
    Ri_B  = gt * (tair - tsurf) * d_eff / (Tmean * w_spd^2)
    Ri_B  = clamp(Ri_B, -ft(0.5), Ric)

    # 4. Friction Factor (Fw)
    Fw_neg = ft(1.0) - (ft(9.4) * Ri_B) / (ft(1.0) + ccoef * sqrt(abs(Ri_B)))
    Fw_pos = ft(1.0) / (ft(1.0) + ft(4.7) * Ri_B)^2
    Fw     = ifelse(Ri_B < ft(0.0), Fw_neg, Fw_pos)
    Fw     = clamp(Fw, ft(1.0e-3), ft(10.0))

    # 5. Final Resistance (Scaled identically bridging the 0.5C gap mapped from VIC geometry bounds natively)
    C_H = max(ft(1.0) * a_sq * Fw, ft(1.0e-6))
    ra_val = ft(1.0) / (C_H * w_spd)
    
    return clamp(ra_val, ra_min, ra_max)
end


function compute_aerodynamic_resistance!(
    ra, 
    z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu
)
    # --- 1. Hard Validation (Debugging/Safety) ---
    # This costs zero time if types are correct but saves you from Float64 lag.
    @assert eltype(ra)       == FloatType "Output 'ra' must be FloatType"
    @assert eltype(tsurf)    == FloatType "Input 'tsurf' must be FloatType"
    @assert von_karman       isa FloatType "Constant 'von_karman' in SimConstants must be FloatType"
    @assert g                isa FloatType "Constant 'g' in SimConstants must be FloatType"

    # --- 2. Local Constants ---
    z_floor = ft(1e-3)
    d_floor = ft(1e-2)
    w_floor = ft(0.1)
    ra_min  = ft(1.0)
    ra_max  = ft(1e5)
    
    # Pre-calculate log expression
    L2_min  = ft(log(1.01)^2)
    
    # Cast scalar z2 once if it isn't already
    z2T = ft(z2)

    # --- 3. Grid Dimensions ---
    N_all   = size(ra, 4)
    veg_dim = max(N_all - 1, 0)

    # ========================================================================
    # 1. SOIL TILES (Last Index)
    # ========================================================================
    # We pass the global constants (von_karman, g, t_freeze, Ri_cr) directly.
    @views @. ra[:, :, :, N_all:N_all] = aerodynamic_kernel(
        z0soil_gpu,                
        d0_gpu[:,:,:,N_all:N_all], 
        tsurf,                     
        tair_gpu,                  
        wind_gpu,                  
        z2T, von_karman, g, t_freeze, Ri_cr, 
        z_floor, d_floor, w_floor, L2_min, ra_min, ra_max
    )

    # ========================================================================
    # 2. VEGETATION TILES (Indices 1:veg_dim)
    # ========================================================================
    if veg_dim > 0
        @views @. ra[:, :, :, 1:veg_dim] = aerodynamic_kernel(
            z0_gpu[:,:,:,1:veg_dim],   
            d0_gpu[:,:,:,1:veg_dim],   
            tsurf,
            tair_gpu,
            wind_gpu,
            z2T, von_karman, g, t_freeze, Ri_cr, 
            z_floor, d_floor, w_floor, L2_min, ra_min, ra_max
        )
    end

    return nothing
end


function calculate_net_radiation!(net_rad, swdown_gpu, lwdown_gpu, albedo_gpu, tsurf,
                                  snow_coverage_gpu=nothing, snow_albedo_gpu=nothing, snow_surf_temp_gpu=nothing)
    if snow_coverage_gpu === nothing
        @. net_rad = (ft(1.0) - albedo_gpu) * swdown_gpu + lwdown_gpu - emissivity * sigma * (tsurf + ft(273.15))^4
    else
        eff_alb(alb, sc, s_alb) = (isnan(sc) || sc <= ft(0.0)) ? alb : (sc * s_alb + (ft(1.0) - sc) * alb)
        eff_t(ts, sc, s_ts) = (isnan(sc) || sc <= ft(0.0)) ? ts : (sc * s_ts + (ft(1.0) - sc) * ts)
        
        @. net_rad = (ft(1.0) - eff_alb(albedo_gpu, snow_coverage_gpu, snow_albedo_gpu)) * swdown_gpu + lwdown_gpu - emissivity * sigma * (eff_t(tsurf, snow_coverage_gpu, snow_surf_temp_gpu) + ft(273.15))^4
    end
    
    return nothing
end

function calculate_potential_evaporation!(
    pe,
    tair_gpu, tair_grid, psurf_gpu, vp_gpu, elev_gpu,
    net_radiation, aerodynamic_resistance, rarc_gpu, rmin_gpu, LAI_gpu
)
    # 1. Setup Type
    T = eltype(pe)
    
    # --- Local Coefficients ---
    G_COEFF = ft(1628.6)    
    AIR_C   = ft(0.003486)  
    SOIL_RC = ft(100.0)     
    EPS     = ft(1.0e-6)      

    # 2. Pre-calculate 2D Meteorological Terms (Fused)
    # We reduce allocations by inlining 'vpd', 'scale_h', and 'p_sfc' 
    # directly into the terms that need them.

    # Array 1: Slope [Pa/K] (Must be stored, used in both num/den)
    slope = @. calculate_svp_slope(tair_gpu)
    
    # Array 2: Latent Heat [J/kg] (Must be stored, used in gamma and final calc)
    latent_heat = @. calculate_latent_heat(tair_gpu + t_freeze)
    
    # Array 3: Gamma [Pa/K]
    # FUSED: We calculate scale_height and p_sfc *inside* this kernel.
    # p_sfc = p_std * exp(-elev / scale_h)
    gamma_ = @. G_COEFF * (p_std * exp(-elev_gpu / calculate_scale_height(tair_gpu, elev_gpu))) / latent_heat
    
    # Array 4: Air Density Term [J/m3/s * Pa]
    # FUSED: We calculate VPD *inside* this kernel.
    # Note: calculate_vpd returns [Pa] now, so no extra conversion needed.
    # We use tair_grid instead of tair_gpu so our VPD mimics VIC's unlapsed VPD parity exactly.
    air_dens_term = @. ((AIR_C * psurf_gpu * pa_per_kpa) / (t_freeze + tair_gpu)) * c_p_air * calculate_vpd(tair_grid, vp_gpu) * day_sec

    # 3. Apply Logic (Tile Level)
    # Define the range for vegetation tiles
    veg_indices = 1:(nveg - 1)
    
    # Single fused kernel launch for all vegetation tiles
    @views @. pe[:, :, :, veg_indices] = max(
        (slope * (net_radiation[:, :, :, veg_indices] * day_sec) + 
         (air_dens_term / aerodynamic_resistance[:, :, :, veg_indices])) / 
        (latent_heat * (slope + gamma_ * (ft(1.0) + 
         ((rmin_gpu[:, :, :, veg_indices] / max(LAI_gpu[:, :, :, veg_indices], EPS)) + 
          rarc_gpu[:, :, :, veg_indices]) / aerodynamic_resistance[:, :, :, veg_indices]))),
        ft(0.0)
    )

    # --- Bare Soil Tile (Index 14) ---
    @views @. pe[:, :, :, nveg] = max(
        (slope * (net_radiation[:, :, :, nveg] * day_sec) + (air_dens_term / aerodynamic_resistance[:, :, :, nveg])) / 
        (latent_heat * (slope + gamma_ * (ft(1.0) + SOIL_RC / aerodynamic_resistance[:, :, :, nveg]))),
        ft(0.0)
    )

    return nothing
end


function calculate_max_water_storage!(max_water_storage, LAI_gpu, coverage_gpu)

    @. max_water_storage = ifelse(coverage_gpu > ft(1.0e-5), (K_L * LAI_gpu) / coverage_gpu, ft(0.0))

    @. max_water_storage = ifelse(
        isnan(max_water_storage) | (abs(max_water_storage) > fillvalue_threshold), 
        ft(0.0), 
        max_water_storage
    )

    return nothing
end


@inline function canopy_evap_physics_kernel(
    ws, max_ws, pot_evap, ra, rarc, 
    prec, lai, tair, elev, rmin
)

    # --- 1. Physics Calculations ---
    # Using your existing project functions
    slope = calculate_svp_slope(tair)
    latent_heat = calculate_latent_heat(tair + t_freeze)

    scale_height = calculate_scale_height(tair, elev)
    surface_pressure = ft(101325.0) * exp(-elev / scale_height)
    gamma_val = ft(1628.6) * surface_pressure / latent_heat

    # --- 2. Resistances ---
    rc = rmin / max(lai, ft(1e-6))
    inv_ra = ft(1.0) / max(ra, ft(1e-6))

    # --- 3. Denominators (Penman-Monteith) ---
    den_w = slope + gamma_val * (ft(1.0) + rarc * inv_ra)
    den_rc = den_w + (gamma_val * rc * inv_ra)

    E_p_wet = pot_evap * (den_rc / max(den_w, ft(1e-6)))

    # --- 4. VIC Equations ---
    Wratio = clamp(ws / max(max_ws, ft(1e-6)), ft(0.0), ft(1.0))
    ra_ratio = ra / max(ra + rarc, ft(1e-6))
    
    canopy_evap_star = (Wratio ^ (ft(2.0) / ft(3.0))) * E_p_wet * ra_ratio

    # --- 5. Fraction Calculation (f_n) ---
    f_n_val = clamp((ws + prec) / max(canopy_evap_star, ft(1e-6)), ft(0.0), ft(1.0))
    f_n_val = ifelse(canopy_evap_star <= ft(1e-6), ft(1.0), f_n_val)

    # --- 6. Final Evaporation ---
    evap = f_n_val * canopy_evap_star
    
    # Sanitize
    evap = ifelse(isnan(evap) || abs(evap) > ft(1e15), ft(0.0), evap)

    return evap, f_n_val
end

# Helper functions (must be global) 
@inline pick_evap(args...) = canopy_evap_physics_kernel(args...)[1]
@inline pick_fn(args...)   = canopy_evap_physics_kernel(args...)[2]

function calculate_canopy_evaporation!(
    canopy_evaporation, f_n, 
    water_storage, max_water_storage, potential_evaporation,
    aerodynamic_resistance, rarc, prec_gpu, cv_gpu, rmin_gpu, LAI_gpu,
    tair_gpu, elev_gpu
)

    # 1. Update canopy_evaporation
    @. canopy_evaporation = pick_evap(
        water_storage, max_water_storage, potential_evaporation,
        aerodynamic_resistance, rarc, 
        prec_gpu, LAI_gpu, tair_gpu, elev_gpu, rmin_gpu
    )

    # 2. Update f_n
    @. f_n = pick_fn(
        water_storage, max_water_storage, potential_evaporation,
        aerodynamic_resistance, rarc, 
        prec_gpu, LAI_gpu, tair_gpu, elev_gpu, rmin_gpu
    )

    # 3. Post-Process: Zero out bare soil (last index)
    last_veg = size(canopy_evaporation, 4)
    view(canopy_evaporation, :, :, :, last_veg) .= ft(0.0)

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
    soil_moisture_old,       
    soil_moisture_critical,  
    wilting_point,           
    root_gpu, 
    cv_gpu, 
    f_n,
    AreaFract
)
    i, j = @index(Global, NTuple)

    # Boundary Check
    if i <= size(transpiration_full, 1) && j <= size(transpiration_full, 2)
        
        # Constants
        EPS  = ft(1e-9)
        ZERO = ft(0.0)
        ONE  = ft(1.0)
        
        # --- 1. SOIL STRESS (g1, g2) ---
        # Load Layer 1
        W1   = soil_moisture_old[i,j,1]
        Wcr1 = soil_moisture_critical[i,j,1]
        Wwp1 = wilting_point[i,j,1]
        
        # Load Layer 2
        W2   = soil_moisture_old[i,j,2]
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
            
            W_root_sum = ZERO
            Wcr_root_sum = ZERO
            
            if f1 > ZERO
                W_root_sum += W1
                Wcr_root_sum += Wcr1
            end
            if f2 > ZERO
                W_root_sum += W2
                Wcr_root_sum += Wcr2
            end
            
            share_moist = (W_root_sum >= Wcr_root_sum) && (W_root_sum > ZERO)
            
            moist1_wet = W1 >= Wcr1
            moist2_wet = W2 >= Wcr2
            
            if share_moist
                g_sw_veg = ONE
            else
                g_sw_veg = clamp((f1 * g1 + f2 * g2) / (f1 + f2 + EPS), ZERO, ONE)
            end

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
                dry_time_factor = clamp(ONE - fn_val * (term_inner ^ (ft(2.0)/ft(3.0))), ZERO, ONE)

                if k == nveg
                    dry_time_factor = ONE
                end

                # --- Transpiration Calculation ---
                trans_val = clamp(cv * dry_time_factor * pe * g_sw_veg, ZERO, ft(Inf))
                
                # --- Layer Apportionment (E1, E2) ---
                if share_moist
                    root_sum = ZERO
                    spare_transp = ZERO
                    
                    if moist1_wet
                        e1_val = trans_val * f1
                        root_sum += f1
                    else
                        e1_val = trans_val * g1 * f1
                        spare_transp += trans_val * f1 * (ONE - g1)
                    end
                    
                    if moist2_wet
                        e2_val = trans_val * f2
                        root_sum += f2
                    else
                        e2_val = trans_val * g2 * f2
                        spare_transp += trans_val * f2 * (ONE - g2)
                    end
                    
                    if spare_transp > ZERO && root_sum > ZERO
                        if moist1_wet
                            e1_val += spare_transp * (f1 / root_sum)
                        end
                        if moist2_wet
                            e2_val += spare_transp * (f2 / root_sum)
                        end
                    end
                else
                    weight1 = f1 * g1
                    weight2 = f2 * g2
                    total_denom = weight1 + weight2 + EPS
                    
                    e1_val = trans_val * (weight1 / total_denom)
                    e2_val = trans_val * (weight2 / total_denom)
                end

                if k == nveg
                     trans_val = ZERO
                     e1_val = ZERO
                     e2_val = ZERO
                end

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

function calculate_transpiration!(
    # Outputs
    transpiration_full, 
    transpiration_layers, 
    # Inputs
    potential_evaporation, 
    water_storage, 
    max_water_storage, 
    soil_moisture_old, 
    soil_moisture_critical, 
    wilting_point, 
    root_gpu, 
    cv_gpu, 
    f_n,
    AreaFract
)
    # 1. Configuration
    kernel_launcher! = transpiration_kernel!(device_backend)    
    nx, ny = size(transpiration_full)

    # 2. Launch
    kernel_launcher!(
        transpiration_full, 
        transpiration_layers,
        potential_evaporation, 
        water_storage, 
        max_water_storage, 
        soil_moisture_old, 
        soil_moisture_critical, 
        wilting_point, 
        root_gpu, 
        cv_gpu, 
        f_n,
        AreaFract;
        ndrange = (nx, ny)
    )

    return nothing
end



function calculate_soil_evaporation!(
    soil_evap, # <--- Output array (Modified in-place, 2D Grid)
    soil_moisture, soil_moisture_max, potential_evaporation, 
    b_infilt_gpu, cv_gpu, coverage_gpu, residual_moisture, AreaFract_gpu
)
    T = eltype(soil_evap)
    
    # Clear the output array first (since we accumulate into it)
    fill!(soil_evap, ft(0.0))

    # --- 1. The Scalar Physics Kernel (Inner Function) ---
    function soil_evap_kernel(sm_top, sm_max_top, resid_top, pe, b_i, cv, cov)
        # 1. Calculate Max Infiltration
        max_infil = (ft(1.0) + b_i) * sm_max_top
        
        # 2. Moisture Ratio
        ratio = clamp(ft(1.0) - sm_top / sm_max_top, ft(0.0), ft(1.0))
        
        # 3. Handle b_i == -1.0 case
        ratio_adj = (b_i == -ft(1.0)) ? ratio : ratio ^ (ft(1.0) / (b_i + ft(1.0)))
        
        tmp = max_infil * (ft(1.0) - ratio_adj)
        if b_i == -ft(1.0)
            tmp = max_infil
        end

        # 4. Saturation Check
        is_saturated = tmp >= max_infil
        
        # 5. ARNO Evaporation Logic
        ratio_unsat = clamp(ft(1.0) - (tmp / max_infil), ft(0.0), ft(1.0))
        
        ratio_powered = (ratio_unsat > ft(0.0)) ? ratio_unsat ^ b_i : ft(0.0)
        as_val = ft(1.0) - ratio_powered
        
        ratio_beta = (ratio_powered > ft(0.0)) ? ratio_powered ^ (ft(1.0) / b_i) : ft(0.0)
        
        # 6. Series Expansion (Replaces the 40 lines of unrolled code)
        # Using a simple loop here uses registers, not VRAM arrays!
        dummy = ft(1.0)
        ratio_pow_term = ratio_beta # Starts as x^1
        
        # Loop 40 times (matches your manual unrolling)
        for k in 1:40
            # dummy += b * (ratio_beta^k) / (b + k)
            dummy += (b_i * ratio_pow_term) / (b_i + FloatType(k))
            ratio_pow_term *= ratio_beta # Increment power for next loop
        end

        beta_asp = as_val + (ft(1.0) - as_val) * (ft(1.0) - ratio_beta) * dummy
        
        # 7. Final Calculation
        # If saturated, use PE. Else, scale by beta_asp
        esoil = is_saturated ? pe : pe * beta_asp
        
        # Apply weights
        esoil = esoil * (ft(1.0) - cov) * cv
        
        # 8. Cap at Available Moisture
        avail = max(sm_top - resid_top, ft(0.0))
        esoil = clamp(esoil, ft(0.0), avail)
        
        return esoil
    end

    # --- 2. Apply Logic (Accumulate over Veg Types and Bands) ---
    N_veg = size(cv_gpu, 4)
    N_bands = size(AreaFract_gpu, 3)
    
    for i in 1:N_veg
        for b in 1:N_bands
            # FIX: Slice 4D arrays appropriately, and pass b_infilt_gpu directly
            @views @. soil_evap += soil_evap_kernel(
                soil_moisture[:,:,1],           
                soil_moisture_max[:,:,1],       
                residual_moisture[:,:,1],       
                potential_evaporation[:,:,b,i], # loop over band b
                b_infilt_gpu,                   
                cv_gpu[:,:,1,i] * AreaFract_gpu[:,:,b], # Weight by band fraction
                coverage_gpu[:,:,1,i]           
            )
        end
    end
    
    return nothing
end

function update_water_canopy_storage!(
    water_storage, throughfall,  # Targets for mutation
    prec, cv, canopy_evap, Wm, coverage
)
    # We use @. (dot broadcast) to fuse the loop and avoid allocating 'new_storage' or 'excess' arrays.
    
    # 1. Update Throughfall FIRST
    # We calculate the 'excess' logic on the fly using the *current* (old) water_storage.
    # Logic: excess = max(0, (W + P - E) - Wm)
    # Throughfall = (excess * coverage) + (prec * (1 - coverage))
    @. throughfall = (max(ft(0.0), water_storage + prec - canopy_evap - Wm) * coverage) + 
                     (prec * (ft(1.0) - coverage))

    # 2. Update Water Storage SECOND
    # Now we can safely mutate water_storage.
    # Logic: clamped new storage
    @. water_storage = clamp(water_storage + prec - canopy_evap, ft(0.0), Wm)

    return nothing
end




# Eq. (23): Total evapotranspiration
function calculate_total_evapotranspiration!(
    total_et,    # Mutated Output
    canopy_evap, transp, soil_evap, cv, coverage, AreaFract
)
    # 1. Initialize with Soil Evaporation
    @. total_et = soil_evap

    # 2. Accumulate Vegetation Fluxes
    # We loop over tiles to avoid allocating massive intermediate arrays.
    for i in 1:size(canopy_evap, 4)
        for b in 1:size(canopy_evap, 3)
            @views @. total_et += (canopy_evap[:,:,b,i] * cv[:,:,1,i] + transp[:,:,b,i]) * coverage[:,:,1,i] * AreaFract[:,:,b]
        end
    end
    
    return nothing
end