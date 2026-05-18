# ==============================================================================
# Snow Physics Module
# ==============================================================================
# The state is 4-dimensional: (nx, ny, nbands, nveg). 
# This allows tracking an independent snowpack for every elevation band and 
# vegetation tile within each spatial grid cell.
# ==============================================================================

# Solve surface energy balance via Newton-Raphson to find snow surface temp (max 0°C).
@inline function snow_surface_temp_nr(
    tsurf_init,   # Initial temperature guess from previous timestep [°C] (OldTSurf)
    Ta,           # Air temperature [°C]
    sw_in,        # Incoming shortwave radiation [W/m²]
    lw_in,        # Incoming longwave radiation [W/m²]
    albedo,       # Albedo of the snow surface [-]
    psurf,        # Surface atmospheric pressure [kPa]
    ra,           # Aerodynamic resistance [s/m]
    vp_air,       # Vapor pressure of the air [kPa]
    swe_surf_m    # Surface layer SWE [m] (for deltaCC thermal inertia)
)
    # 1. Atmospheric Properties (Air Density & Specific Humidity)
    # Air density from ideal gas: ρ = P / (Rd * Tk)
    Ta_K     = Ta + ft(273.15)
    psurf_Pa = psurf * ft(1000.0)   # kPa → Pa
    rho_air  = psurf_Pa / (ft(287.0) * Ta_K)
    Cp_air   = ft(1004.0)
    ha       = rho_air * Cp_air / max(ra, ft(1.0))   # Heat transfer coefficient [W/(m²·K)]

    # 2. Radiative Forcing Variables
    sw_net = sw_in * (one(FloatType) - albedo)
    ps_eff = max(psurf_Pa, ft(50000.0))   # Capped surface pressure [Pa]

    # Vapour pressure: cap EactAir at es(Ta) to match VPD=0 condensation suppression
    EactAir_raw = clamp(vp_air * ft(1000.0), zero(FloatType), ft(15000.0))
    es_Ta       = ft(611.0) * exp(ft(17.27) * Ta / max(ft(237.3) + Ta, ft(1.0)))
    EactAir     = min(EactAir_raw, es_Ta)  # [Pa]

    # 3. Latent Heat Parameters
    L_sub    = ft(2.845e6)   # Latent heat of sublimation [J/kg]
    Ls_Ra    = rho_air * L_sub / max(ra, ft(1.0)) * ft(0.622) / ps_eff  # [W/m²/Pa]
    es0      = ft(611.0)     # Saturation vapor pressure at 0°C [Pa]
    LE_sub_0 = Ls_Ra * (es0 - EactAir)   # Latent energy at Ts=0°C (for melt energy)

    # 4. Energy Balance Formulation
    # Base RHS without LE term (LE will be in the residual):
    rhs_base = sw_net + lw_in + ha * Ta   # SW_net + LW_in + H_sens(Ta) [W/m²]

    # Melt energy is computed at Ts=0 using LE0 (standard convention)
    rhs_melt = rhs_base - LE_sub_0 
    
    # Longwave emission coefficient: SIGMA * epsilon_snow
    sig_eps = SIGMA * ft(0.97)   # param.EMISS_SNOW = 0.97

    # 5. Newton-Raphson Solver
    # f(ts) = LW_out(ts) + h_sens(ts,Ta) - LE(min(ts,0)) - rhs_base = 0
    ts = tsurf_init
    for _ in 1:12
        Ts_K   = ts + ft(273.15)
        lw_out = sig_eps * (Ts_K ^ ft(4.0))
        h_sens = ha * ts
        
        # Compute LE at capped surface temp (snow surface can't be > 0°C)
        ts_cap = min(ts, zero(FloatType))
        es_ts_cap = ft(611.0) * exp(ft(21.87) * ts_cap / max(ft(265.5) + ts_cap, ft(1.0)))
        le_ts = Ls_Ra * max(es_ts_cap - EactAir, zero(FloatType))
        
        # Objective function and derivative
        f_val  = lw_out + h_sens - le_ts - rhs_base
        dles_dts = ft(21.87) * es_ts_cap / max(ft(265.5) + ts_cap, ft(1.0))
        dle_dts = ifelse(ts < zero(FloatType), Ls_Ra * dles_dts, zero(FloatType))
        df_val = ft(4.0) * sig_eps * (Ts_K ^ ft(3.0)) + ha - dle_dts
        
        # Step update with clamping for stability
        step   = f_val / max(abs(df_val), ft(1e-6))
        step   = clamp(step, -ft(10.0), ft(10.0))
        ts     = ts - step
    end
    ts = clamp(ts, -ft(60.0), ft(50.0))

    # 6. Evaluate Melt Potential
    # Melt energy at Ts=0 (using LE at 0°C, per standard convention)
    Ts0_K           = ft(273.15)
    lw_out0         = sig_eps * (Ts0_K ^ ft(4.0))
    melt_energy_net = rhs_melt - lw_out0   # Melt energy using LE_sub_0
    ts_melt         = zero(FloatType)
    melt_heat_out   = max(melt_energy_net, zero(FloatType))

    # 7. Sublimation Calculation (Non-melting state)
    ts_no_melt = min(ts, zero(FloatType))
    is_melting = ts > zero(FloatType)

    # Sublimation mass is computed from the energy balance residual at ts_no_melt.
    Ts0_no_melt_K    = ts_no_melt + ft(273.15)
    lw_out_no_melt   = sig_eps * (Ts0_no_melt_K ^ ft(4.0))
    le_eb_no_melt    = rhs_base - lw_out_no_melt - ha * ts_no_melt  # LE from energy balance
    sub_flux_Wm2     = max(le_eb_no_melt, zero(FloatType))             # Positive LE = sublimation
    sub_mass_mm_cold = sub_flux_Wm2 / L_sub * ft(86400.0) * ft(1000.0)

    # 8. Merge Execution Paths
    final_ts          = ifelse(is_melting, ts_melt, ts_no_melt)
    final_melt_energy = ifelse(is_melting, melt_heat_out, zero(FloatType))
    sub_mass_mm       = ifelse(is_melting, zero(FloatType), sub_mass_mm_cold)

    return final_ts, final_melt_energy, sub_mass_mm
end

# ------------------------------------------------------------------------------
# Kernel Implementations
# ------------------------------------------------------------------------------

# Computing the daily mass and energy balance of the 4D snowpack grid.
@kernel function snow_dynamics_kernel!(
    # 4D Snowpack State (in/out)
    swe,                    # Snow water equivalent [mm]
    surf_water,             # Surface layer liquid water [mm]
    pack_water,             # Deep pack layer liquid water [mm]
    snow_depth,             # Snow depth [mm]
    snow_albedo,            # Surface albedo [-]
    snow_surf_temp,         # Snow surface temperature [°C]
    snow_coverage,          # Fractional snow coverage [-]
    melt_out,               # Melt runoff generated [mm/day]
    last_snow,              # Days since last significant snowfall [days]
    cold_content,           # Surface layer cold content [J/m²]
    pack_cold_content,      # Deep pack layer cold content (SWE > 125mm) [J/m²]
    melting_flag,           # Binary flag indicating active melting [-]
    store_snow,             # Sub-grid snow accumulation tracker
    snow_distrib_slope,     # Sub-grid distribution slope tracker
    store_swq,              # Tracked SWE for sub-grid distribution
    store_coverage,         # Tracked coverage for sub-grid distribution
    max_snow_depth,         # Maximum observed snow depth [m]
    
    # Atmospheric & Canopy Forcings
    @Const(throughfall_4d), # Throughfall from vegetation canopy [mm/day]
    @Const(tair_band),      # Air temperature at elevation band [°C]
    @Const(swdown_2d),      # Downwelling shortwave radiation [W/m²]
    @Const(lwdown_2d),      # Downwelling longwave radiation [W/m²]
    @Const(psurf_2d),       # Surface atmospheric pressure [kPa]
    @Const(vp_2d),          # Atmospheric vapor pressure [kPa]
    @Const(wind_2d),        # Wind speed [m/s]
    @Const(AreaFract),      # Elevation band area fractions [-]
    @Const(cv_4d),          # Vegetation cover fractions [-]
    @Const(annual_prec_2d), # Annual mean precipitation [mm/yr]
    
    # Temporal Context
    day_of_year,            # Current day of the year [1-366]
    lat_positive            # Hemisphere flag (1 = Northern, 0 = Southern)
)
    i, j, b, v = @index(Global, NTuple)

    area   = AreaFract[i, j, b]
    cv_wt  = cv_4d[i, j, 1, v]
    t_band = tair_band[i, j, b]
    tf_val = throughfall_4d[i, j, b, v]

    # Active mask for branchless execution
    active = (!isnan(area) & (area > zero(FloatType)) &
              !isnan(cv_wt) & (cv_wt > zero(FloatType)) &
              !isnan(t_band) & !isnan(tf_val))

    # --------------------------------------------------------------------------
    # 0. Precipitation Partitioning
    # --------------------------------------------------------------------------
    t_avg = t_band
    MAX_SNOW_TEMP = ft(0.5)
    MIN_RAIN_TEMP = ft(-0.5)
    
    rain_frac = clamp((t_avg - MIN_RAIN_TEMP) / max(MAX_SNOW_TEMP - MIN_RAIN_TEMP, ft(1e-6)), zero(FloatType), one(FloatType))
    p_snow = tf_val * (one(FloatType) - rain_frac)
    p_rain = tf_val * rain_frac

    # --------------------------------------------------------------------------
    # 1. State Loading & Initialization
    # --------------------------------------------------------------------------
    current_swe  = swe[i, j, b, v]
    c_surf_water = surf_water[i, j, b, v]
    c_pack_water = pack_water[i, j, b, v]
    
    old_coverage = snow_coverage[i, j, b, v]
    old_coverage = ifelse(isnan(old_coverage), zero(FloatType), clamp(old_coverage, zero(FloatType), one(FloatType)))
    
    # Coverage instantly becomes 1.0 if it is snowing
    is_p_snow = p_snow > zero(FloatType)
    temp_coverage = ifelse(is_p_snow, one(FloatType), old_coverage)

    p_rain_snowpack = p_rain * temp_coverage
    p_rain_bare     = p_rain * (one(FloatType) - temp_coverage)

    old_depth_m  = snow_depth[i, j, b, v] / ft(1000.0)

    current_cc    = cold_content[i, j, b, v]
    prior_cc_orig = current_cc
    current_pcc   = pack_cold_content[i, j, b, v]
    current_pcc   = ifelse(isnan(current_pcc), zero(FloatType), current_pcc)
    lsnow         = last_snow[i, j, b, v]
    melt_flag     = melting_flag[i, j, b, v]
    st_snow       = store_snow[i, j, b, v]
    st_swq        = store_swq[i, j, b, v]
    st_cov        = store_coverage[i, j, b, v]
    dslope        = snow_distrib_slope[i, j, b, v]
    mx_depth      = max_snow_depth[i, j, b, v]

    sw_in  = swdown_2d[i, j]
    lw_in  = lwdown_2d[i, j]
    ps     = psurf_2d[i, j]
    vp_air = vp_2d[i, j]
    wind   = wind_2d[i, j]

    # Aerodynamic resistance: Wind-based log-law
    z0_snow = ft(0.0005)
    ln_z_z0 = log((ft(2.0) + z0_snow) / z0_snow)
    u_2m    = max(wind, ft(0.1)) * ft(0.8375)  # Scale 10m → 2m
    ra = (ln_z_z0 * ln_z_z0) / (ft(0.16) * u_2m)   # s/m, neutral stability
    ra = clamp(ra, ft(50.0), ft(300.0))        # Clip at bounds

    ann_prec = annual_prec_2d[i, j]
    max_distrib_slope = ifelse(isnan(ann_prec) | (ann_prec <= zero(FloatType)), ft(0.4), ann_prec / ft(500.0))

    # --------------------------------------------------------------------------
    # 2. Accumulation & Cold Content Dynamics
    # --------------------------------------------------------------------------
    old_swq_pre = current_swe
    current_swe += p_snow + p_rain_snowpack
    
    # Adjust effective snowfall temp for daily steps to retain nighttime cold
    SNW_DTR_HALF = ft(4.0)
    sf_temp = min(t_avg - SNW_DTR_HALF, zero(FloatType))
    sf_cc = ifelse(p_snow > zero(FloatType), SNW_VCPICE_WQ * p_snow * sf_temp, zero(FloatType))
    current_cc = min(current_cc + sf_cc, zero(FloatType))

    # Add rain heat energy
    rain_heat = ifelse((p_rain > zero(FloatType)) & (current_swe > zero(FloatType)) & (t_avg > zero(FloatType)), 
                       ft(4186.0) * (p_rain / ft(1000.0)) * t_avg, 
                       zero(FloatType))
    current_cc = min(current_cc + rain_heat, zero(FloatType))

    # --------------------------------------------------------------------------
    # 3. Albedo Evolution
    # --------------------------------------------------------------------------
    is_trace  = p_snow > SNW_NEW_SNOW_THRESH_MM
    has_swe   = current_swe > zero(FloatType)

    # Counter reset on any meaningful snowfall
    lsnow = ifelse(is_trace, Int32(0), ifelse(has_swe, lsnow + Int32(1), Int32(0)))
    ls_f  = ft(lsnow)

    alb_accum = SNW_NEW_SNOW_ALB * (SNW_ALB_ACCUM_A ^ (ls_f ^ SNW_ALB_ACCUM_B))
    alb_thaw  = SNW_NEW_SNOW_ALB * (SNW_ALB_THAW_A  ^ (ls_f ^ SNW_ALB_THAW_B))
    is_accum  = (current_cc < zero(FloatType)) & (melt_flag == Int32(0))

    alb_age = ifelse(is_accum, alb_accum, alb_thaw)

    # Albedo resets to max ONLY when new snow falls on a cold pack
    pack_is_cold = current_cc < zero(FloatType)
    is_new = is_trace & pack_is_cold
    alb = ifelse(is_new, SNW_NEW_SNOW_ALB, ifelse(has_swe, alb_age, ft(NaN)))

    # --------------------------------------------------------------------------
    # 4. Seasonal Melting State Transition
    # --------------------------------------------------------------------------
    in_melt_season = ifelse(lat_positive == Int32(1),
                            (day_of_year > Int32(60)) & (day_of_year < Int32(273)),
                            (day_of_year < Int32(60)) | (day_of_year > Int32(273)))
                            
    SNW_MELT_RESET_THRESH_MM = ft(5.0)
    CC_MELT_DEADBAND = zero(FloatType)
    flag_cond1 = (current_cc >= CC_MELT_DEADBAND) & in_melt_season
    
    # Snow_reset: THAW --> ACCUM only if snowfall AND pack is truly cold
    pack_is_cold_for_reset = current_cc < zero(FloatType)
    snow_reset = is_trace & (melt_flag == Int32(1)) & pack_is_cold_for_reset
    
    melt_flag = ifelse(has_swe,
        ifelse(snow_reset, Int32(0), ifelse(flag_cond1, Int32(1), melt_flag)),
        Int32(0))

    # --------------------------------------------------------------------------
    # 5. Energy Balance & Melt Generation
    # --------------------------------------------------------------------------
    melt = zero(FloatType)
    prev_ts = snow_surf_temp[i, j, b, v]
    prev_ts = ifelse(isnan(prev_ts), zero(FloatType), prev_ts)
    
    # Initial guess for the NR solver
    t_s = ifelse((prev_ts >= t_avg - ft(5.0)) & (prev_ts <= zero(FloatType)), prev_ts, min(t_avg, zero(FloatType)))

    eff_alb = ifelse(isnan(alb), SNW_NEW_SNOW_ALB, alb)
    swe_surf_m_nr = min(current_swe / ft(1000.0), SNW_MAX_SURFACE_SWE_MM / ft(1000.0))

    ts_solved, melt_energy_at_zero, sub_mass_mm = snow_surface_temp_nr(t_s, t_avg, sw_in, lw_in, eff_alb, ps, ra, vp_air, swe_surf_m_nr)
    t_s = ifelse(has_swe, ts_solved, ft(NaN))
    
    melt_J = ifelse(has_swe & (melt_energy_at_zero > zero(FloatType)), melt_energy_at_zero * ft(86400.0), zero(FloatType))
    
    # --------------------------------------------------------------------------
    # 5a. Satisfy Cold Content
    # --------------------------------------------------------------------------
    energy_needed_sfc = max(-current_cc, zero(FloatType))
    melt_apply_sfc = min(melt_J, energy_needed_sfc)
    melt_J -= melt_apply_sfc
    current_cc = min(current_cc + melt_apply_sfc, zero(FloatType))

    energy_needed_pack = max(-current_pcc, zero(FloatType))
    melt_apply_pack = min(melt_J, energy_needed_pack)
    melt_J -= melt_apply_pack
    current_pcc = min(current_pcc + melt_apply_pack, zero(FloatType))

    # --------------------------------------------------------------------------
    # 5b. Phase Change and Liquid Generation
    # --------------------------------------------------------------------------
    phase_melt = ifelse(has_swe & (melt_energy_at_zero > zero(FloatType)), melt_J / (SNW_LATICE * SNW_RHOFW) * ft(1000.0), zero(FloatType))
    
    # Clamp phase melt to available solid ice
    swe_ice = max(current_swe - c_surf_water - c_pack_water, zero(FloatType))
    phase_melt = min(phase_melt, swe_ice)

    # Refreeze liquid if cold content wasn't fully satisfied
    refreeze_energy_sfc = min(-current_cc, c_surf_water * (SNW_LATICE * SNW_RHOFW) / ft(1000.0))
    refreeze_surf_val   = refreeze_energy_sfc / (SNW_LATICE * SNW_RHOFW) * ft(1000.0)
    c_surf_water       -= refreeze_surf_val
    current_cc         += refreeze_energy_sfc
    
    refreeze_energy_pack = min(-current_pcc, c_pack_water * (SNW_LATICE * SNW_RHOFW) / ft(1000.0))
    refreeze_pack_val    = refreeze_energy_pack / (SNW_LATICE * SNW_RHOFW) * ft(1000.0)
    c_pack_water        -= refreeze_pack_val
    current_pcc         += refreeze_energy_pack

    # Add new melt and rain to surface liquid storage
    c_surf_water += p_rain_snowpack + phase_melt

    # --------------------------------------------------------------------------
    # 5c. Pack Drainage
    # --------------------------------------------------------------------------
    swe_surf_m = min(current_swe / ft(1000.0), SNW_MAX_SURFACE_SWE_MM / ft(1000.0))
    swe_pack_m = max(current_swe / ft(1000.0) - SNW_MAX_SURFACE_SWE_MM / ft(1000.0), zero(FloatType))
    
    SNW_LIQUID_WATER_CAPACITY = ft(0.03)
    
    max_liq_surf = SNW_LIQUID_WATER_CAPACITY * (swe_surf_m * ft(1000.0))
    surf_drain = max(c_surf_water - max_liq_surf, zero(FloatType))
    c_surf_water -= surf_drain
    
    # Surface drainage feeds the deep pack
    c_pack_water += surf_drain
    
    # Pack layer refreeze
    refreeze_pack2_val = min(c_pack_water, max(-current_pcc / (SNW_LATICE * SNW_RHOFW) * ft(1000.0), zero(FloatType)))
    c_pack_water -= refreeze_pack2_val
    current_pcc  = min(current_pcc + refreeze_pack2_val * (SNW_LATICE * SNW_RHOFW) / ft(1000.0), zero(FloatType))

    # Pack layer outflow (Melt Runoff)
    max_liq_pack = SNW_LIQUID_WATER_CAPACITY * (swe_pack_m * ft(1000.0))
    pack_drain = max(c_pack_water - max_liq_pack, zero(FloatType))
    c_pack_water -= pack_drain

    # Update SWE accounting for sublimation mass loss
    ice_remaining = max(swe_ice - phase_melt - sub_mass_mm, zero(FloatType))
    current_swe   = max(ice_remaining + c_surf_water + c_pack_water, zero(FloatType))
    melt          = pack_drain
    melt_out_val  = pack_drain

    # --------------------------------------------------------------------------
    # 5d. Thermal Inertia State Updates
    # --------------------------------------------------------------------------
    swe_surf_m = min(current_swe / ft(1000.0), SNW_MAX_SURFACE_SWE_MM / ft(1000.0))
    swe_pack_m = max(current_swe / ft(1000.0) - SNW_MAX_SURFACE_SWE_MM / ft(1000.0), zero(FloatType))
    
    cc_melt_branch = min(prior_cc_orig * ft(0.01), zero(FloatType))
    pcc_melt_branch = ifelse(swe_pack_m > zero(FloatType), SNW_VCPICE_WQ * (swe_pack_m * ft(1000.0)) * (t_s * ft(0.5)), current_pcc)
    pcc_melt_branch = min(pcc_melt_branch, zero(FloatType))
    ts_for_cc = t_s
    cc_nomelt_branch = min(SNW_VCPICE_WQ * (swe_surf_m * ft(1000.0)) * ts_for_cc, zero(FloatType))
    pcc_nomelt_branch = current_pcc

    is_melting_step = melt > zero(FloatType)
    
    SNW_DEEP_SWE_MM   = ft(100.0)
    SNW_DTR_HALF_CC   = ft(4.0)
    sf_temp_night     = min(t_avg - SNW_DTR_HALF_CC, zero(FloatType))
    SNW_CC_MIN_T      = ft(-0.5)
    cc_min_thin       = SNW_VCPICE_WQ * (swe_surf_m * ft(1000.0)) * SNW_CC_MIN_T
    cc_melt_night     = min(SNW_VCPICE_WQ * (swe_surf_m * ft(1000.0)) * sf_temp_night, cc_min_thin)
    
    f_deep = clamp(current_swe / SNW_DEEP_SWE_MM, zero(FloatType), one(FloatType))
    cc_melt_physical = current_cc
    cc_melt_eff = f_deep * cc_melt_physical + (one(FloatType) - f_deep) * cc_melt_night
    
    current_cc = ifelse(is_melting_step, cc_melt_eff, ifelse(t_s < zero(FloatType), cc_nomelt_branch, current_cc))
    current_pcc = ifelse(is_melting_step, pcc_melt_branch, ifelse(t_s < zero(FloatType), pcc_nomelt_branch, current_pcc))

    in_off_season = !in_melt_season
    cc_reset_flag = in_off_season & (current_cc < zero(FloatType))
    melt_flag = ifelse(has_swe,
        ifelse(cc_reset_flag,
            Int32(0),
            ifelse(is_melting_step & in_melt_season & !snow_reset, Int32(1), melt_flag)),
        Int32(0))

    # --------------------------------------------------------------------------
    # 6. Physical Dimensions & Trace Pruning
    # --------------------------------------------------------------------------
    above_trace = current_swe >= SNW_TRACESNOW_MM
    current_swe  = ifelse(above_trace, current_swe, zero(FloatType))
    c_surf_water = ifelse(above_trace, c_surf_water, zero(FloatType))
    c_pack_water = ifelse(above_trace, c_pack_water, zero(FloatType))
    melt         = ifelse(above_trace, melt, zero(FloatType))

    current_cc  = ifelse(above_trace, current_cc,  zero(FloatType))
    current_pcc = ifelse(above_trace, current_pcc, zero(FloatType))
    t_s = ifelse(above_trace, t_s, ft(NaN))

    current_depth_m = (current_swe / ft(1000.0)) * (SNW_RHOFW / SNW_DENSITY)

    # Coverage: binary model (options.SPATIAL_SNOW = false)
    new_coverage = ifelse(current_swe > SNW_TRACESNOW_MM, one(FloatType), zero(FloatType))

    # --------------------------------------------------------------------------
    # 7. Write Result States
    # --------------------------------------------------------------------------
    swe[i, j, b, v]                  = ifelse(active, current_swe, zero(FloatType))
    surf_water[i, j, b, v]           = ifelse(active, ifelse(current_swe > zero(FloatType), c_surf_water, zero(FloatType)), zero(FloatType))
    pack_water[i, j, b, v]           = ifelse(active, ifelse(current_swe > zero(FloatType), c_pack_water, zero(FloatType)), zero(FloatType))
    snow_depth[i, j, b, v]           = ifelse(active, ifelse(current_swe > zero(FloatType), current_depth_m * ft(1000.0), zero(FloatType)), zero(FloatType))
    snow_albedo[i, j, b, v]          = ifelse(active, ifelse((current_swe > zero(FloatType)) & !isnan(alb), alb, ft(NaN)), ft(NaN))
    
    swe_surf_mm_out = min(current_swe, SNW_MAX_SURFACE_SWE_MM)
    t_s_out = ifelse(swe_surf_mm_out > zero(FloatType),
                     current_cc / (SNW_VCPICE_WQ * swe_surf_mm_out),
                     zero(FloatType))
    t_s_out = min(t_s_out, zero(FloatType))
    
    snow_surf_temp[i, j, b, v]       = ifelse(active, ifelse(current_swe > zero(FloatType), t_s_out, ft(NaN)), ft(NaN))
    snow_coverage[i, j, b, v]        = ifelse(active, new_coverage, zero(FloatType))
    melt_out[i, j, b, v]             = ifelse(active, melt_out_val, zero(FloatType))
    last_snow[i, j, b, v]            = ifelse(active, ifelse(current_swe > zero(FloatType), lsnow, Int32(0)), Int32(0))
    cold_content[i, j, b, v]         = ifelse(active, ifelse(current_swe > zero(FloatType), current_cc, zero(FloatType)), zero(FloatType))
    pack_cold_content[i, j, b, v]    = ifelse(active, ifelse(current_swe > zero(FloatType), current_pcc, zero(FloatType)), zero(FloatType))
    melting_flag[i, j, b, v]         = ifelse(active, ifelse(current_swe > zero(FloatType), melt_flag, Int32(0)), Int32(0))
    store_snow[i, j, b, v]           = ifelse(active, ifelse(current_swe > zero(FloatType), st_snow, Int32(0)), Int32(0))
    snow_distrib_slope[i, j, b, v]   = ifelse(active, ifelse(current_swe > zero(FloatType), dslope, zero(FloatType)), zero(FloatType))
    store_swq[i, j, b, v]            = ifelse(active, ifelse(current_swe > zero(FloatType), st_swq, zero(FloatType)), zero(FloatType))
    store_coverage[i, j, b, v]       = ifelse(active, ifelse(current_swe > zero(FloatType), st_cov, zero(FloatType)), zero(FloatType))
    max_snow_depth[i, j, b, v]       = ifelse(active, ifelse(current_swe > zero(FloatType), mx_depth, zero(FloatType)), zero(FloatType))

    nothing
end

# Wrapper to dispatch the `snow_dynamics_kernel!` across the GPU backend and sync.
function calculate_snow_dynamics!(
    swe_gpu, surf_water_gpu, pack_water_gpu, snow_depth_gpu, snow_albedo_gpu, snow_surf_temp_gpu,
    snow_coverage_gpu, snow_melt_gpu,
    last_snow_gpu, cold_content_gpu, pack_cc_gpu, melting_flag_gpu,
    store_snow_gpu, snow_distrib_slope_gpu,
    store_swq_gpu, store_coverage_gpu, max_snow_depth_gpu,
    throughfall_4d, tair_3d, swdown_gpu, lwdown_gpu, psurf_gpu, vp_gpu, wind_gpu,
    AreaFract_gpu, cv_gpu, annual_prec_gpu,
    day_of_year::Int, lat_mean::FloatType
)
    # Determine hemispheric context for seasonality checks
    lat_pos = Int32(lat_mean >= 0.0 ? 1 : 0)

    # Dispatch the compute kernel
    kernel! = snow_dynamics_kernel!(device_backend)
    kernel!(
        swe_gpu, surf_water_gpu, pack_water_gpu, snow_depth_gpu, snow_albedo_gpu, snow_surf_temp_gpu,
        snow_coverage_gpu, snow_melt_gpu,
        last_snow_gpu, cold_content_gpu, pack_cc_gpu, melting_flag_gpu,
        store_snow_gpu, snow_distrib_slope_gpu,
        store_swq_gpu, store_coverage_gpu, max_snow_depth_gpu,
        throughfall_4d, tair_3d, swdown_gpu, lwdown_gpu, psurf_gpu, vp_gpu, wind_gpu,
        AreaFract_gpu, cv_gpu, annual_prec_gpu,
        Int32(day_of_year), lat_pos;
        ndrange=size(swe_gpu)
    )
    
    KernelAbstractions.synchronize(device_backend)
end
