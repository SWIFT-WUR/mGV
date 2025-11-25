# --- Scalar Physics Kernel (Runs on every pixel) ---
function aerodynamic_kernel(z0, d0, tsurf, tair, wind, z2, Kt, gt, Tf, Ric, z_floor, d_floor, w_floor, L2_min, ra_min, ra_max)
    # 1. Roughness & Effective Height
    rough = max(z0, z_floor)
    d_eff = max(z2 - d0, d_floor)

    # 2. Log-law terms
    ratio = clamp(d_eff / rough, 1f-6, 1f6)
    L     = log(ratio)
    L2    = max(L^2, L2_min)
    a_sq  = (Kt^2) / L2
    ccoef = 49.82f0 * a_sq * sqrt(ratio)

    # 3. Stability (Richardson Number)
    w_spd = max(wind, w_floor)
    Tmean = max(((tair + Tf) + (tsurf + Tf)) * 0.5f0, 100.0f0)
    
    Ri_B  = gt * (tair - tsurf) * d_eff / (Tmean * w_spd^2)
    Ri_B  = clamp(Ri_B, -0.5f0, Ric)

    # 4. Friction Factor (Fw)
    Fw_neg = 1.0f0 - (9.4f0 * Ri_B) / (1.0f0 + ccoef * sqrt(abs(Ri_B)))
    Fw_pos = 1.0f0 / (1.0f0 + 4.7f0 * Ri_B)^2
    Fw     = ifelse(Ri_B < 0f0, Fw_neg, Fw_pos)
    Fw     = clamp(Fw, 1f-3, 10.0f0)

    # 5. Final Resistance
    C_H = max(1.351f0 * a_sq * Fw, 1f-6)
    ra_val = 1.0f0 / (C_H * w_spd)
    
    return clamp(ra_val, ra_min, ra_max)
end


function compute_aerodynamic_resistance!(
    ra, 
    z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu
)
    T = eltype(ra)

    # --- Constants (Float32) ---
    # We pass these into the kernel to ensure type stability
    Kt      = T(K)
    gt      = T(g)
    Tf      = T(t_freeze)
    Ric     = T(Ri_cr)
    z_floor = T(1e-3)
    d_floor = T(1e-2)
    w_floor = T(0.1)
    L2_min  = T(log(1.01)^2)
    ra_min  = T(1.0)
    ra_max  = T(1e5)
    z2T     = T(z2)
    
    N_all   = size(ra, 4)
    veg_dim = max(N_all - 1, 0)

    # ========================================================================
    # 1. SOIL TILES (Last Index)
    # ========================================================================
    # We broadcast the kernel. 
    # Note: tsurf, tair_gpu, wind_gpu are 2D. ra is 4D slice. 
    # Julia broadcasts (36,36) -> (36,36,1) automatically.
    @views @. ra[:, :, :, N_all:N_all] = aerodynamic_kernel(
        T(z0soil_gpu),          # z0 input (Soil specific)
        d0_gpu[:,:,:,N_all:N_all], # d0 input
        tsurf,                  # tsurf (2D)
        tair_gpu,               # tair (2D)
        wind_gpu,               # wind (2D)
        z2T, Kt, gt, Tf, Ric, z_floor, d_floor, w_floor, L2_min, ra_min, ra_max
    )

    # ========================================================================
    # 2. VEGETATION TILES (Indices 1:veg_dim)
    # ========================================================================
    if veg_dim > 0
        @views @. ra[:, :, :, 1:veg_dim] = aerodynamic_kernel(
            z0_gpu[:,:,:,1:veg_dim],   # z0 input (Veg specific)
            d0_gpu[:,:,:,1:veg_dim],   # d0 input
            tsurf,
            tair_gpu,
            wind_gpu,
            z2T, Kt, gt, Tf, Ric, z_floor, d_floor, w_floor, L2_min, ra_min, ra_max
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


function compute_partial_canopy_resistance(rmin_gpu, LAI_gpu)
    # Canopy resistance based on soil moisture (Eq. 6), without gsm multiplication; done in evapotranspiration calculation step   
    return rmin_gpu ./ LAI_gpu
end

#function calculate_net_radiation(swdown_gpu, lwdown_gpu, albedo_gpu, tsurf)
#    return (1.0f0 .- albedo_gpu) .* swdown_gpu .+ lwdown_gpu .- emissivity .* sigma .* (tsurf .+ 273.15f0).^4
#end

function calculate_net_radiation!(net_rad, swdown_gpu, lwdown_gpu, albedo_gpu, tsurf)
    @. net_rad = (1.0f0 - albedo_gpu) * swdown_gpu + lwdown_gpu - emissivity * sigma * (tsurf + 273.15f0)^4
    
    return nothing
end

function calculate_potential_evaporation!(
    pe,
    tair_gpu, psurf_gpu, vp_gpu, elev_gpu,
    net_radiation, aerodynamic_resistance, rarc_gpu, rmin_gpu, LAI_gpu
)
    # 1. Setup Type
    T = eltype(pe)
    
    # --- Local Coefficients (Float32 Literals) ---
    G_COEFF = 1628.6f0    
    AIR_C   = 0.003486f0  
    SOIL_RC = 100.0f0     
    EPS     = 1.0f-6      

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
    # Loop over vegetation tiles (Indices 1 to 13)
    for i in 1:(nveg - 1)
        @. pe[:, :, :, i] = max(
            (slope * (net_radiation[:, :, :, i] * day_sec) + (air_dens_term / aerodynamic_resistance[:, :, :, i])) / 
            (latent_heat * (slope + gamma_ * (T(1) + ( (rmin_gpu[:, :, :, i] / max(LAI_gpu[:, :, :, i], EPS)) + rarc_gpu[:, :, :, i]) / aerodynamic_resistance[:, :, :, i]))),
            T(0)
        )
    end

    # --- Bare Soil Tile (Index 14) ---
    @. pe[:, :, :, nveg] = max(
        (slope * (net_radiation[:, :, :, nveg] * day_sec) + (air_dens_term / aerodynamic_resistance[:, :, :, nveg])) / 
        (latent_heat * (slope + gamma_ * (T(1) + SOIL_RC / aerodynamic_resistance[:, :, :, nveg]))),
        T(0)
    )

    return nothing
end


function calculate_max_water_storage!(max_water_storage, LAI_gpu, cv_gpu, coverage_gpu)

    @. max_water_storage = K * LAI_gpu * cv_gpu #TODO should we multiply by .* cv_gpu ?

    @. max_water_storage = ifelse(
        isnan(max_water_storage) | (abs(max_water_storage) > fillvalue_threshold), 
        0.0f0, 
        max_water_storage
    )

    return nothing
end


function calculate_canopy_evaporation!(
    canopy_evaporation, f_n, # <--- Mutated Outputs (4D Arrays)
    water_storage, max_water_storage, potential_evaporation,
    aerodynamic_resistance, rarc, prec_gpu, cv_gpu, rmin, LAI_gpu,
    tair_gpu, elev_gpu,
)
    # Constants (Float32 literals)
    tiny = 1.0f-6
    
    # ---- Sanitize Inputs (In-place) ----
    # Using Float32 literals (0.0f0, 1.0f6) directly
    @. potential_evaporation  = ifelse(isnan(potential_evaporation) | (abs(potential_evaporation) > fillvalue_threshold), 0.0f0, potential_evaporation)
    @. water_storage          = ifelse(isnan(water_storage)         | (abs(water_storage)         > fillvalue_threshold), 0.0f0, water_storage)
    @. max_water_storage      = ifelse(isnan(max_water_storage)     | (abs(max_water_storage)     > fillvalue_threshold), 0.0f0, max_water_storage)
    @. aerodynamic_resistance = ifelse(isnan(aerodynamic_resistance)| (abs(aerodynamic_resistance)> fillvalue_threshold), 1.0f6, aerodynamic_resistance)
    @. rarc                   = ifelse(isnan(rarc)                  | (abs(rarc)                  > fillvalue_threshold), 0.0f0, rarc)

    
    # ---- Δ and γ ----
    # We use @. to broadcast scalar physics functions
    slope = @. calculate_svp_slope(tair_gpu)
    
    latent_heat = @. calculate_latent_heat(tair_gpu + t_freeze)

    scale_height = @. calculate_scale_height(tair_gpu, elev_gpu)
    
    # Calculate Pressure and Gamma
    # p_std is Float32 from SimConstants
    surface_pressure = @. p_std * exp(-elev_gpu / scale_height)
    gamma_ = @. 1628.6f0 * surface_pressure / latent_heat

    # =========================================================================
    # DEBUG BLOCK: Check Types (Canopy Evap)
    # =========================================================================
    if rand() < 0.01 
        println("\n--- [DEBUG] calculate_canopy_evaporation! Type Check ---")
        println("INPUTS (Sanitized):")
        println("  potential_evaporation: ", eltype(potential_evaporation))
        println("  water_storage:         ", eltype(water_storage))
        println("  max_water_storage:     ", eltype(max_water_storage))
        println("  aerodynamic_resistance:", eltype(aerodynamic_resistance))
        println("  rarc:                  ", eltype(rarc))
        println("INTERMEDIATES:")
        println("  slope:                 ", eltype(slope))
        println("  latent_heat:           ", eltype(latent_heat))
        println("  scale_height:          ", eltype(scale_height))
        println("  surface_pressure:      ", eltype(surface_pressure))
        println("  gamma_:                ", eltype(gamma_))
        println("-----------------------------------------------------------\n")
    end
    # =========================================================================

    # ---- Resistances ----
    rc = @. rmin / max(LAI_gpu, 1.0f-6)
    ra = aerodynamic_resistance # Alias

    RALPHA_MIN = nothing
    ralpha = (RALPHA_MIN === nothing) ? rarc : max.(rarc, RALPHA_MIN)

    # ---- Denominators & E_p_wet ----
    den_rc  = @. slope + gamma_ * (1.0f0 + (rc + ralpha) / ra)
    den_w   = @. slope + gamma_ * (1.0f0 + ralpha / ra)
    
    E_p_wet = @. potential_evaporation * (den_rc / max(den_w, tiny))

    # ---- VIC Eq. (1) ----
    Wratio   = @. clamp(water_storage / max(max_water_storage, tiny), 0.0f0, 1.0f0)
    ra_ratio = @. ra / max(ra + ralpha, tiny)
    
    # Note: 2/3 must be calculated as Float32 (2.0f0/3.0f0)
    canopy_evaporation_star = @. (Wratio ^ (2.0f0 / 3.0f0)) * E_p_wet * ra_ratio

    # ---- Update f_n (In-Place) ----
    # 1. Calculate raw fraction
    @. f_n = clamp((water_storage + prec_gpu) / max(canopy_evaporation_star, tiny), 0.0f0, 1.0f0)
    
    # 2. Apply masking logic directly to f_n
    # Logic: if star <= tiny, f_n = 1.0, else keep f_n
    @. f_n = ifelse(canopy_evaporation_star <= tiny, 1.0f0, f_n)

    # ---- Update canopy_evaporation (In-Place) ----
    @. canopy_evaporation = f_n * canopy_evaporation_star
    
    # Final sanitize
    @. canopy_evaporation = ifelse(isnan(canopy_evaporation) | (abs(canopy_evaporation) > fillvalue_threshold), 0.0f0, canopy_evaporation)

    # Zero out bare soil tile (last index)
    canopy_evaporation[:, :, :, end:end] .= 0.0f0

    return nothing
end


function calculate_transpiration(
    potential_evaporation::CuArray, aerodynamic_resistance::CuArray, rarc_gpu::CuArray,
    water_storage::CuArray, max_water_storage::CuArray, soil_moisture_old::CuArray,
    soil_moisture_critical::CuArray, wilting_point::CuArray, root_gpu::CuArray,
    rmin_gpu::CuArray, LAI_gpu::CuArray, cv_gpu, f_n
)
    # -------- One dtype everywhere --------
    T   = eltype(potential_evaporation)
    F0  = T(0); F1 = T(1); EPS = T(1e-9)

    # Cast only the arrays used below that may be Float64
    Wcr_T  = T.(soil_moisture_critical)      # (ny,nx,layer)
    Wwp_T  = T.(wilting_point)               # (ny,nx,layer)
    Wmax_T = T.(max_water_storage)           # (ny,nx,1,nveg)
    W_T    = T.(water_storage)               # (ny,nx,1,nveg)
    cv_T   = T.(cv_gpu)                      # (ny,nx,1,nveg)
    PE_T   = T.(potential_evaporation)       # (ny,nx,1,nveg)

    # -------- soil-moisture stress per layer (assume L=2) --------
    W1   = @view soil_moisture_old[:, :, 1]
    W2   = @view soil_moisture_old[:, :, 2]
    Wcr1 = @view Wcr_T[:, :, 1]
    Wcr2 = @view Wcr_T[:, :, 2]
    Wwp1 = @view Wwp_T[:, :, 1]
    Wwp2 = @view Wwp_T[:, :, 2]

    g1 = clamp.((W1 .- Wwp1) ./ (Wcr1 .- Wwp1 .+ EPS), F0, F1)
    g2 = clamp.((W2 .- Wwp2) ./ (Wcr2 .- Wwp2 .+ EPS), F0, F1)

    ny, nx = size(g1)
    veg_dim = size(root_gpu, 4)   # vegetation tiles (exclude bare)
    nveg    = size(cv_T, 4)       # vegetation + bare

    f1  = reshape(@view(root_gpu[:, :, 1, :]), ny, nx, 1, veg_dim)
    f2  = reshape(@view(root_gpu[:, :, 2, :]), ny, nx, 1, veg_dim)
    g1b = reshape(g1, ny, nx, 1, 1)
    g2b = reshape(g2, ny, nx, 1, 1)

#    sumf     = sum(root_gpu, dims=3)                          # (ny,nx,1,veg_dim)
#    g_sw_veg = clamp.((f1 .* g1b .+ f2 .* g2b) ./ (sumf .+ EPS), F0, F1)

    g_sw_veg = clamp.((f1 .* g1b ) ./ (f1 .+ EPS), F0, F1)


    # -------- canopy wetness (per canopy) --------
    cv_safe = max.(cv_T, EPS)
    W_can   = W_T ./ cv_safe                                   # canopy-area basis
    Wratio  = clamp.(W_can ./ max.(Wmax_T, EPS), F0, F1)
    wetFrac = Wratio .^ (T(2)/T(3))
    dry_time_factor = clamp.(F1 .- T.(f_n) .* wetFrac, F0, F1)
    dry_time_factor[:, :, :, end:end] .= F1                             # bare soil unaffected

    # -------- transpiration for vegetation tiles only --------
    transpiration_veg =
    cv_T[:, :, :, 1:veg_dim] .*
    dry_time_factor[:, :, :, 1:veg_dim] .*
    PE_T[:, :, :, 1:veg_dim] .*
    g_sw_veg

    transpiration_veg = clamp.(transpiration_veg, F0, T(Inf))

    # split to layers (weights)
    denom_veg = f1 .* g1b .+ f2 .* g2b .+ EPS
    E_1_t_veg = transpiration_veg .* (f1 .* g1b) ./ denom_veg
    E_2_t_veg = transpiration_veg .* (f2 .* g2b) ./ denom_veg

    # GPU-safe NaN guards
    E_1_t_veg = ifelse.(isnan.(E_1_t_veg), F0, E_1_t_veg)
    E_2_t_veg = ifelse.(isnan.(E_2_t_veg), F0, E_2_t_veg)

    # -------- assemble full-tile arrays with bare slice = 0 --------
    transpiration_full = CUDA.zeros(T, ny, nx, 1, nveg)
    E_1_t_full         = CUDA.zeros(T, ny, nx, 1, nveg)
    E_2_t_full         = CUDA.zeros(T, ny, nx, 1, nveg)

    transpiration_full[:, :, :, 1:veg_dim] .= transpiration_veg
    E_1_t_full[:, :, :, 1:veg_dim]        .= E_1_t_veg
    E_2_t_full[:, :, :, 1:veg_dim]        .= E_2_t_veg

    # -------- ADDED: Create transpiration_layers array (e1t, e2t, 0) --------
    # This array will have dimensions (ny, nx, 3, nveg)
    E_0_t_full = CUDA.zeros(T, ny, nx, 1, nveg) # Create zero layer
    transpiration_layers = cat(E_1_t_full, E_2_t_full, E_0_t_full; dims=3)


    return transpiration_full, transpiration_layers, E_1_t_full, E_2_t_full, g1, g2, g_sw_veg, dry_time_factor
end

function calculate_soil_evaporation!(
    soil_evap, # <--- Output array (Modified in-place, 2D Grid)
    soil_moisture, soil_moisture_max, potential_evaporation, 
    b_infilt_gpu, cv_gpu, coverage_gpu, residual_moisture
)
    T = eltype(soil_evap)
    
    # Clear the output array first (since we accumulate into it)
    fill!(soil_evap, 0.0f0)

    # --- 1. The Scalar Physics Kernel (Inner Function) ---
    function soil_evap_kernel(sm_top, sm_max_top, resid_top, pe, b_i, cv, cov)
        # 1. Calculate Max Infiltration
        max_infil = (1.0f0 + b_i) * sm_max_top
        
        # 2. Moisture Ratio
        ratio = clamp(1.0f0 - sm_top / sm_max_top, 0.0f0, 1.0f0)
        
        # 3. Handle b_i == -1.0 case
        ratio_adj = (b_i == -1.0f0) ? ratio : ratio ^ (1.0f0 / (b_i + 1.0f0))
        
        tmp = max_infil * (1.0f0 - ratio_adj)
        if b_i == -1.0f0
            tmp = max_infil
        end

        # 4. Saturation Check
        is_saturated = tmp >= max_infil
        
        # 5. ARNO Evaporation Logic
        ratio_unsat = clamp(1.0f0 - (tmp / max_infil), 0.0f0, 1.0f0)
        
        ratio_powered = (ratio_unsat > 0.0f0) ? ratio_unsat ^ b_i : 0.0f0
        as_val = 1.0f0 - ratio_powered
        
        ratio_beta = (ratio_powered > 0.0f0) ? ratio_powered ^ (1.0f0 / b_i) : 0.0f0
        
        # 6. Series Expansion (Replaces the 40 lines of unrolled code)
        # Using a simple loop here uses registers, not VRAM arrays!
        dummy = 1.0f0
        ratio_pow_term = ratio_beta # Starts as x^1
        
        # Loop 40 times (matches your manual unrolling)
        for k in 1:40
            # dummy += b * (ratio_beta^k) / (b + k)
            dummy += (b_i * ratio_pow_term) / (b_i + Float32(k))
            ratio_pow_term *= ratio_beta # Increment power for next loop
        end

        beta_asp = as_val + (1.0f0 - as_val) * (1.0f0 - ratio_beta) * dummy
        
        # 7. Final Calculation
        # If saturated, use PE. Else, scale by beta_asp
        esoil = is_saturated ? pe : pe * beta_asp
        
        # Apply weights
        esoil = esoil * (1.0f0 - cov) * cv
        
        # 8. Cap at Available Moisture
        avail = max(sm_top - resid_top, 0.0f0)
        esoil = clamp(esoil, 0.0f0, avail)
        
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
    # Throughfall = (excess * coverage * cv) + (prec * (1 - coverage) * cv)
    @. throughfall = (cv * max(0.0f0, water_storage + prec - canopy_evap - Wm) * coverage) + 
                     (prec * (1.0f0 - coverage) * cv)

    # 2. Update Water Storage SECOND
    # Now we can safely mutate water_storage.
    # Logic: clamped new storage * cv
    @. water_storage = clamp(water_storage + prec - canopy_evap, 0.0f0, Wm) * cv

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