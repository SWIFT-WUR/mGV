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

    # 5. Final Resistance
    C_H = max(ft(1.351) * a_sq * Fw, ft(1.0e-6))
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
    @assert K                isa FloatType "Constant 'K' in SimConstants must be FloatType"
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
    # We pass the global constants (K, g, t_freeze, Ri_cr) directly.
    @views @. ra[:, :, :, N_all:N_all] = aerodynamic_kernel(
        z0soil_gpu,                
        d0_gpu[:,:,:,N_all:N_all], 
        tsurf,                     
        tair_gpu,                  
        wind_gpu,                  
        z2T, K, g, t_freeze, Ri_cr, 
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
            z2T, K, g, t_freeze, Ri_cr, 
            z_floor, d_floor, w_floor, L2_min, ra_min, ra_max
        )
    end

    return nothing
end

#more vic like:
#function compute_aerodynamic_resistance(z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu)
#    # Use one element type everywhere
#    T = eltype(cv_gpu)
#    
#    # Constants from C code
#    K2 = T(0.4^2)           # von Karman constant squared
#    factor = T(1.0/0.63 - 1.0)  # ≈ 0.5873
#    z_ref = T(2.0)          # reference height [m]
#    huge_resist = T(1e5)    # resistance when wind ≤ 0
#    
#    # Numeric safeties
#    z_floor = T(1e-3)       # min roughness [m]
#    d_floor = T(0.0)        # min displacement [m]
#    wind_floor = T(1e-6)    # min wind to avoid division by zero
#    ra_min = T(1.0)         # clamp bounds [s/m]
#    ra_max = T(1e5)
#    
#    # Roughness per tile (veg tiles from z0, last tile from soil z0)
#    roughness = similar(cv_gpu)
#    roughness[:, :, :, 1:end-1] .= max.(T.(z0_gpu[:, :, :, 1:end-1]), z_floor)
#    roughness[:, :, :, end:end] .= max.(T.(z0soil_gpu), z_floor)
#    
#    # Displacement per tile
#    displacement = max.(T.(d0_gpu), d_floor)
#    
#    # C code formula: Ra[0] = log((2. + (1.0/0.63 - 1.0) * d_Lower) / Z0_Lower) *
#    #                         log((2. + (1.0/0.63 - 1.0) * d_Lower) / (0.1 * Z0_Lower)) / K2
#    numerator = z_ref .+ factor .* displacement
#    
#    # Avoid log of numbers ≤ 1
#    arg1 = max.(numerator ./ roughness, T(1.001))
#    arg2 = max.(numerator ./ (T(0.1) .* roughness), T(1.001))
#    
#    term1 = log.(arg1)
#    term2 = log.(arg2)
#    
#    ra = (term1 .* term2) ./ K2
#    
#    # Scale by wind speed (or set to huge resistance if wind ≤ 0)
#    w = T.(wind_gpu)
#    ra = ifelse.(w .> T(0), 
#                 ra ./ max.(w, wind_floor),
#                 huge_resist)
#    
#    # Clamp to reasonable bounds
#    ra = clamp.(ra, ra_min, ra_max)
#    
#    return ra
#end


function calculate_net_radiation!(net_rad, swdown_gpu, lwdown_gpu, albedo_gpu, tsurf)
    @. net_rad = (ft(1.0) - albedo_gpu) * swdown_gpu + lwdown_gpu - emissivity * sigma * (tsurf + ft(273.15))^4
    
    return nothing
end

function calculate_potential_evaporation!(
    pe,
    tair_gpu, psurf_gpu, vp_gpu, elev_gpu,
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
    air_dens_term = @. ((AIR_C * psurf_gpu * pa_per_kpa) / (t_freeze + tair_gpu)) * c_p_air * calculate_vpd(tair_gpu, vp_gpu) * day_sec

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


function calculate_max_water_storage!(max_water_storage, LAI_gpu)

    @. max_water_storage = K * LAI_gpu 

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
    f_n 
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
        # Loop over the 4th dimension (vegetation tiles)
        nveg = size(root_gpu, 4)
        
        # Determine actual vegetation limit (exclude bare soil if needed)
        # Original code implied bare soil is the last index and should be zeroed.
        # We loop through all, but apply zero logic at the end.
        
        for k in 1:nveg
            # Load Root Fractions
            f1 = root_gpu[i,j,1,k]
            f2 = root_gpu[i,j,2,k]
            
            # --- Vegetation Conductance (g_sw_veg) ---
            # Original: clamp((f1 * g1) / (f1 + EPS), 0, 1)
            g_sw_veg = clamp((f1 * g1) / (f1 + EPS), ZERO, ONE)

            # --- Canopy Wetness / Dry Time Factor ---
            # Inputs are 4D (nx,ny,1,veg). Access [i,j,1,k]
            ws   = water_storage[i,j,1,k]
            max_ws = max_water_storage[i,j,1,k]
            cv   = cv_gpu[i,j,1,k]
            fn_val = f_n[i,j,1,k]
            pe   = potential_evaporation[i,j,1,k]

            # Logic: clamp(1 - f_n * ( (Ws/Cv) / MaxWs )^(2/3), 0, 1)
            # Be careful with parenthesis from original code
            term_inner = (ws / max(cv, EPS)) / max(max_ws, EPS)
            term_inner = clamp(term_inner, ZERO, ONE)
            
            dry_time_factor = clamp(ONE - fn_val * (term_inner ^ (ft(2.0)/ft(3.0))), ZERO, ONE)

            # Fix: Original code forced dry_time_factor to 1.0 for the last index (bare soil)
            # We handle that generally by checking if we are at the last index
            if k == nveg
                dry_time_factor = ONE
            end

            # --- Transpiration Calculation ---
            # T = Cv * DryFactor * PotEvap * Conductance
            trans_val = clamp(cv * dry_time_factor * pe * g_sw_veg, ZERO, ft(Inf))
            
            # --- Layer Weighting (E1, E2) ---
            # Original: trans * (f1*g1) / (f1*g1 + f2*g2 + EPS)
            weight1 = f1 * g1
            weight2 = f2 * g2
            total_denom = weight1 + weight2 + EPS
            
            e1_val = trans_val * (weight1 / total_denom)
            e2_val = trans_val * (weight2 / total_denom)

            # --- Bare Soil Check ---
            # Original code zeros out the last index (Bare Soil)
            # "if nveg > veg_dim ... end:end .= 0"
            # Assuming the last index is always bare soil in this context:
            if k == nveg
                 trans_val = ZERO
                 e1_val = ZERO
                 e2_val = ZERO
            end

            # --- WRITE OUTPUTS ---
            # 1. Total Transpiration
            transpiration_full[i,j,1,k] = trans_val
            
            # 2. Layer distributed transpiration
            transpiration_layers[i,j,1,k] = e1_val
            transpiration_layers[i,j,2,k] = e2_val
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
    f_n
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
        f_n;
        ndrange = (nx, ny)
    )

    return nothing
end



function calculate_soil_evaporation!(
    soil_evap, # <--- Output array (Modified in-place, 2D Grid)
    soil_moisture, soil_moisture_max, potential_evaporation, 
    b_infilt_gpu, cv_gpu, coverage_gpu, residual_moisture
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

    # --- 2. Apply Logic (Accumulate over Veg Types) ---
    N_veg = size(cv_gpu, 4)
    
    for i in 1:N_veg
        # FIX: Slice 4D arrays down to 2D (nx, ny) using [:,:,1,i]
        # FIX: Pass b_infilt_gpu directly (It is 2D, don't slice it!)
        @views @. soil_evap += soil_evap_kernel(
            soil_moisture[:,:,1],           
            soil_moisture_max[:,:,1],       
            residual_moisture[:,:,1],       
            potential_evaporation[:,:,1,i], # <--- Slice dim 3 to ensure 2D
            b_infilt_gpu,                   # <--- CORRECTED: No slicing!
            cv_gpu[:,:,1,i],                # <--- Slice dim 3
            coverage_gpu[:,:,1,i]           # <--- Slice dim 3
        )
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
    canopy_evap, transp, soil_evap, cv, coverage
)
    # 1. Initialize with Soil Evaporation
    @. total_et = soil_evap

    # 2. Accumulate Vegetation Fluxes
    # We loop over tiles to avoid allocating massive 4D intermediate arrays.
    # The @views macro ensures slicing (e.g., [:,:,:,i]) is zero-allocation.
    for i in 1:size(canopy_evap, 4)
        @views @. total_et += (canopy_evap[:,:,:,i] * cv[:,:,:,i] + transp[:,:,:,i]) * coverage[:,:,:,i]
    end
    
    return nothing
end