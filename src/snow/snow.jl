# =============================================================================
# SNOW PHYSICS MODULE — VIC-faithful implementation
# Mirrors: solve_snow.c, snow_utility.c, calc_snow_coverage.c, snow_melt.c
#
# Architecture: 4D state (nx, ny, nbands, nveg) — one snowpack per
# (elevation band × vegetation tile), matching VIC's per-tile approach.
# =============================================================================

# ---------------------------------------------------------------------------
# VIC Snow Parameters (from initialize_parameters.c)
# ---------------------------------------------------------------------------
const SNW_NEW_SNOW_ALB       = FloatType(0.85)    # new snow albedo
const SNW_ALB_ACCUM_A        = FloatType(0.94)    # accumulation-season decay A
const SNW_ALB_ACCUM_B        = FloatType(0.58)    # accumulation-season decay B
const SNW_ALB_THAW_A         = FloatType(0.82)    # melt-season decay A
const SNW_ALB_THAW_B         = FloatType(0.46)    # melt-season decay B
const SNW_TRACESNOW_MM       = FloatType(0.001)   # mm — minimum SWE for active snowpack pruning
# VIC: albedo resets to new_snow_alb on ANY snowfall (store_snowfall > 0 in solve_snow.c).
# Use TRACESNOW threshold to catch tiny snowfall events that sub-daily VIC accumulates.
const SNW_NEW_SNOW_THRESH_MM = FloatType(0.001)   # mm — match VIC SNOW_TRACESNOW (not 0.03mm)
const SNW_LIQUID_WATER_CAP   = FloatType(0.035)   # fraction liquid water capacity
const SNW_MAX_SURFACE_SWE_MM = FloatType(125.0)   # mm — surface layer limit
const SNW_VCPICE_WQ          = FloatType(2117.27)  # J/(kg·K) per water-equiv: VIC CONST_CPICE × CONST_RHOFW/1000
const SNW_LATICE             = FloatType(334000.0) # J/kg — latent heat of fusion
const SNW_RHOFW              = FloatType(1000.0)   # kg/m³ — density of water
const SNW_DENSITY            = FloatType(250.0)    # kg/m³ — typical snow density

# ---------------------------------------------------------------------------
# Newton-Raphson snow surface temperature solver (inline, GPU-compatible)
# Finds Tsurf that satisfies the snow surface energy balance for Tsurf <= 0°C.
# ---------------------------------------------------------------------------
@inline function snow_surface_temp_nr(
    tsurf_init,   # initial guess (°C), previous timestep = OldTSurf in VIC
    Ta,           # air temperature (°C)
    sw_in,        # incoming SW (W/m²)
    lw_in,        # incoming LW (W/m²)
    albedo,       # snow albedo
    psurf,        # surface pressure (kPa, converted internally)
    ra,           # aerodynamic resistance (s/m)
    vp_air,       # vapor pressure of air (kPa)
    swe_surf_m,   # surface layer SWE (m) — for deltaCC thermal inertia
    T_Type
)
    # Air density from ideal gas: ρ = P / (Rd * Tk)
    # NOTE: psurf is in kPa in the mGV forcing (same as temperature.jl uses pa_per_kpa)
    Ta_K     = Ta + T_Type(273.15)
    psurf_Pa = psurf * T_Type(1000.0)   # kPa → Pa
    rho_air  = psurf_Pa / (T_Type(287.0) * Ta_K)
    Cp_air   = T_Type(1004.0)
    ha       = rho_air * Cp_air / max(ra, T_Type(1.0))   # W/(m²·K)

    sw_net = sw_in * (one(T_Type) - albedo)
    ps_eff = max(psurf_Pa, T_Type(50000.0))   # Pa, high-elevation clamp

    # Vapour pressure: cap EactAir at es(Ta) to match VIC VPD=0 condensation suppression
    # (VIC latent_heat_from_snow.c: if Vpd==0 && SurfaceMassFlux<0, set MassFlux=0)
    EactAir_raw = clamp(vp_air * T_Type(1000.0), zero(T_Type), T_Type(15000.0))
    es_Ta       = T_Type(611.0) * exp(T_Type(17.27) * Ta / max(T_Type(237.3) + Ta, T_Type(1.0)))
    EactAir     = min(EactAir_raw, es_Ta)  # Pa

    # LE parameters
    L_sub    = T_Type(2.845e6)   # J/kg
    Ls_Ra    = rho_air * L_sub / max(ra, T_Type(1.0)) * T_Type(0.622) / ps_eff  # W/m²/Pa
    es0      = T_Type(611.0)     # saturation VP at 0°C
    LE_sub_0 = Ls_Ra * (es0 - EactAir)   # LE at Ts=0°C (for melt energy)

    # Temperature-dependent LE for NR: compute LE at ts_capped = min(ts, 0) each iteration
    # This gives accurate surface temperature (less cold bias than LE0 approach)
    # Base RHS without LE term (LE will be in the residual):
    rhs_base  = sw_net + lw_in + ha * Ta   # SW_net + LW_in + H_sens(Ta) [W/m²]

    # Melt energy is computed at Ts=0 using LE0 (VIC convention - correct physical approach)
    rhs_melt  = rhs_base - LE_sub_0  # rhs at Ts=0 using LE at 0°C
    # LW emission: sigma * epsilon_snow
    sig_eps = sigma * T_Type(0.97)   # VIC param.EMISS_SNOW = 0.97

    # Newton-Raphson: f(ts) = LW_out(ts) + h_sens(ts,Ta) - LE(min(ts,0)) - rhs_base = 0
    # LE is temperature-dependent: LE(ts_cap) = Ls_Ra * max(es(ts_cap) - EactAir, 0)
    ts = tsurf_init
    for _ in 1:12
        Ts_K   = ts + T_Type(273.15)
        lw_out = sig_eps * (Ts_K ^ T_Type(4.0))
        h_sens = ha * ts
        # Compute LE at capped surface temp (snow surface can't be > 0°C)
        ts_cap = min(ts, zero(T_Type))
        es_ts_cap = T_Type(611.0) * exp(T_Type(21.87) * ts_cap / max(T_Type(265.5) + ts_cap, T_Type(1.0)))
        le_ts = Ls_Ra * max(es_ts_cap - EactAir, zero(T_Type))
        # Residual: lw_out + h_sens - le_ts = rhs_base
        f_val  = lw_out + h_sens - le_ts - rhs_base
        # Jacobian: d(lw_out)/dts + d(h_sens)/dts - d(le)/dts (if ts<0, es varies)
        dles_dts = T_Type(21.87) * es_ts_cap / max(T_Type(265.5) + ts_cap, T_Type(1.0))
        dle_dts = ifelse(ts < zero(T_Type), Ls_Ra * dles_dts, zero(T_Type))
        df_val = T_Type(4.0) * sig_eps * (Ts_K ^ T_Type(3.0)) + ha - dle_dts
        step   = f_val / max(abs(df_val), T_Type(1e-6))
        step   = clamp(step, -T_Type(10.0), T_Type(10.0))
        ts     = ts - step
    end
    ts = clamp(ts, T_Type(-60.0), T_Type(50.0))

    # Melt energy at Ts=0 (using LE at 0°C, per VIC convention)
    Ts0_K          = T_Type(273.15)
    lw_out0        = sig_eps * (Ts0_K ^ T_Type(4.0))
    melt_energy_net = rhs_melt - lw_out0   # melt energy using LE_sub_0
    ts_melt        = zero(T_Type)
    melt_heat_out  = max(melt_energy_net, zero(T_Type))


    # Path B: no melt
    ts_no_melt = min(ts, zero(T_Type))
    is_melting = ts > zero(T_Type)

    # Sublimation mass: computed from the energy balance residual at ts_no_melt.
    # VIC (latent_heat_from_snow.c): VaporMassFlux = ErrorE / Ls where ErrorE is the LE
    # component of the energy balance. Computing sub from VPD × Ls_Ra independently
    # can give unrealistically large values (14+ mm/day) when es(ts) ≈ ea at high elevation.
    # Correct approach: LE = rhs_base - LW_out(ts) - ha*ts (total EB residual at ts_no_melt).
    # When NR has converged, LE is exactly what the energy balance requires.
    # Sub occurs when LE > 0; condensation when LE < 0 (net moisture flux onto pack).
    Ts0_no_melt_K   = ts_no_melt + T_Type(273.15)
    lw_out_no_melt  = sig_eps * (Ts0_no_melt_K ^ T_Type(4.0))
    le_eb_no_melt   = rhs_base - lw_out_no_melt - ha * ts_no_melt  # LE from energy balance
    sub_flux_Wm2    = max(le_eb_no_melt, zero(T_Type))  # positive LE = sublimation
    sub_mass_mm_cold = sub_flux_Wm2 / L_sub * T_Type(86400.0) * T_Type(1000.0)




    # Merge paths
    final_ts          = ifelse(is_melting, ts_melt, ts_no_melt)
    final_melt_energy = ifelse(is_melting, melt_heat_out, zero(T_Type))
    sub_mass_mm       = ifelse(is_melting, zero(T_Type), sub_mass_mm_cold)

    return final_ts, final_melt_energy, sub_mass_mm
end

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# VIC-faithful 4D snow dynamics kernel
# ---------------------------------------------------------------------------
@kernel function snow_dynamics_kernel!(
    # 4D State (in/out): (nx, ny, nbands, nveg)
    swe, surf_water, pack_water, snow_depth, snow_albedo, snow_surf_temp, snow_coverage, melt_out,
    last_snow, cold_content, pack_cold_content, melting_flag,
    store_snow, snow_distrib_slope, store_swq, store_coverage, max_snow_depth,
    @Const(throughfall_4d), @Const(tair_band), @Const(swdown_2d), @Const(lwdown_2d), 
    @Const(psurf_2d), @Const(vp_2d), @Const(wind_2d), @Const(AreaFract), @Const(cv_4d), @Const(annual_prec_2d),
    day_of_year, lat_positive
)
    i, j, b, v = @index(Global, NTuple)
    T_Type = eltype(swe)

    area   = AreaFract[i, j, b]
    cv_wt  = cv_4d[i, j, 1, v]
    t_band = tair_band[i, j, b]
    tf_val = throughfall_4d[i, j, b, v]

    # Active mask for branchless execution
    active = (!isnan(area) & (area > zero(T_Type)) &
              !isnan(cv_wt) & (cv_wt > zero(T_Type)) &
              !isnan(t_band) & !isnan(tf_val))

    t_avg = t_band
    MAX_SNOW_TEMP = T_Type(0.5)
    MIN_RAIN_TEMP = T_Type(-0.5)
    rain_frac = clamp((t_avg - MIN_RAIN_TEMP) / max(MAX_SNOW_TEMP - MIN_RAIN_TEMP, T_Type(1e-6)), zero(T_Type), one(T_Type))

    p_snow = tf_val * (one(T_Type) - rain_frac)
    p_rain = tf_val * rain_frac

    current_swe  = swe[i, j, b, v]
    c_surf_water = surf_water[i, j, b, v]
    c_pack_water = pack_water[i, j, b, v]
    
    old_coverage = snow_coverage[i, j, b, v]
    old_coverage = ifelse(isnan(old_coverage), zero(T_Type), clamp(old_coverage, zero(T_Type), one(T_Type)))
    
    # Coverage instantly becomes 1.0 if it is snowing
    is_p_snow = p_snow > zero(T_Type)
    temp_coverage = ifelse(is_p_snow, one(T_Type), old_coverage)

    p_rain_snowpack = p_rain * temp_coverage
    p_rain_bare     = p_rain * (one(T_Type) - temp_coverage)

    old_depth_m  = snow_depth[i, j, b, v] / T_Type(1000.0)

    current_cc   = cold_content[i, j, b, v]
    prior_cc_orig = current_cc
    current_pcc  = pack_cold_content[i, j, b, v]
    current_pcc  = ifelse(isnan(current_pcc), zero(T_Type), current_pcc)
    lsnow        = last_snow[i, j, b, v]
    melt_flag    = melting_flag[i, j, b, v]
    st_snow      = store_snow[i, j, b, v]
    st_swq       = store_swq[i, j, b, v]
    st_cov       = store_coverage[i, j, b, v]
    dslope       = snow_distrib_slope[i, j, b, v]
    mx_depth     = max_snow_depth[i, j, b, v]

    sw_in  = swdown_2d[i, j]
    lw_in  = lwdown_2d[i, j]
    ps     = psurf_2d[i, j]
    vp_air = vp_2d[i, j]
    wind   = wind_2d[i, j]

    # Aerodynamic resistance: Wind-based log-law (VIC CalcAerodynamic.c, snow understory)
    # z0_snow = 0.0005m, z = 2m ref height, Ka = 0.4, U_2m = U_10m * 0.8375
    # Neutral stability Ra = ln((z+z0)/z0)^2 / (Ka^2 * U_2m) = 68.7 / (0.134 * U_10m)
    # Result: ~300 s/m at 1.7m/s, ~170 s/m at 3m/s, ~100 s/m at 5.1m/s
    z0_snow = T_Type(0.0005)
    ln_z_z0 = log((T_Type(2.0) + z0_snow) / z0_snow)  # ln(4001) ≈ 8.294
    u_2m    = max(wind, T_Type(0.1)) * T_Type(0.8375)  # scale 10m → 2m
    ra = (ln_z_z0 * ln_z_z0) / (T_Type(0.16) * u_2m)  # s/m, neutral stability
    ra = clamp(ra, T_Type(50.0), T_Type(300.0))        # clip: 300 s/m neutral stability

    ann_prec = annual_prec_2d[i, j]
    max_distrib_slope = ifelse(isnan(ann_prec) | (ann_prec <= zero(T_Type)), T_Type(0.4), ann_prec / T_Type(500.0))

    # 1. Add snowfall and rain to SWE and update cold content (Branchless)
    old_swq_pre = current_swe
    current_swe += p_snow + p_rain_snowpack
    # VIC: CC from snowfall = VCPICE * snowfall * Tair (at each 3-hourly sub-step)
    # mGV uses daily-mean Ta, which is often near 0°C for autumn/spring events.
    # VIC's nighttime steps (Ta 5-8°C below daily mean) give fresh snow significant CC.
    # Correction: use min(t_avg - SNW_DTR_HALF, 0) as effective snowfall temp so autumn
    # snow arriving at Ta=+2°C still gets CC from the nighttime cold (effective T=-2°C).
    # SNW_DTR_HALF = 4°C = assumed half-amplitude of the diurnal temperature range.
    SNW_DTR_HALF = T_Type(4.0)
    sf_temp = min(t_avg - SNW_DTR_HALF, zero(T_Type))  # nighttime-adjusted temp for CC
    sf_cc = ifelse(p_snow > zero(T_Type), SNW_VCPICE_WQ * p_snow * sf_temp, zero(T_Type))
    current_cc = min(current_cc + sf_cc, zero(T_Type))

    rain_heat = ifelse((p_rain > zero(T_Type)) & (current_swe > zero(T_Type)) & (t_avg > zero(T_Type)), T_Type(4186.0) * (p_rain / T_Type(1000.0)) * t_avg, zero(T_Type))
    current_cc = min(current_cc + rain_heat, zero(T_Type))

    # 2. Albedo — VIC-faithful (snow_utility.c lines 264-283, solve_snow.c lines 311-327)
    is_trace  = p_snow > SNW_NEW_SNOW_THRESH_MM   # any snowfall > TRACESNOW
    has_swe   = current_swe > zero(T_Type)

    # lsnow counter: reset on any snowfall (VIC solve_snow.c line 324: last_snow=0 on snowfall)
    lsnow = ifelse(is_trace, Int32(0), ifelse(has_swe, lsnow + Int32(1), Int32(0)))
    ls_f  = T_Type(lsnow)

    alb_accum = SNW_NEW_SNOW_ALB * (SNW_ALB_ACCUM_A ^ (ls_f ^ SNW_ALB_ACCUM_B))
    alb_thaw  = SNW_NEW_SNOW_ALB * (SNW_ALB_THAW_A  ^ (ls_f ^ SNW_ALB_THAW_B))
    is_accum  = (current_cc < zero(T_Type)) & (melt_flag == Int32(0))

    alb_age = ifelse(is_accum, alb_accum, alb_thaw)

    # VIC snow_utility.c line 265: hard-reset to 0.85 ONLY when new_snow > TRACESNOW AND CC < 0.
    # If CC >= 0 (warm/melting pack) with fresh snowfall: continue aging from lsnow=0 (no 0.85 reset).
    # This prevents spurious albedo spikes during summer melt when daily-mean temperature
    # partitions rain as snow but VIC's sub-daily steps classify it mostly as rain.
    pack_is_cold = current_cc < zero(T_Type)
    is_new = is_trace & pack_is_cold   # hard reset only for cold-pack snowfall
    alb = ifelse(is_new, SNW_NEW_SNOW_ALB, ifelse(has_swe, alb_age, T_Type(NaN)))


    # 3. Melting flag
    in_melt_season = ifelse(lat_positive == Int32(1),
                            (day_of_year > Int32(60)) & (day_of_year < Int32(273)),
                            (day_of_year < Int32(60)) | (day_of_year > Int32(273)))
    # VIC: MELTING flag triggers when CC >= 0 AND in melt season (solve_snow.c:376-386).
    # VIC-faithful snowfall reset: snowfall > TRACESNOW while MELTING → MELTING=false.
    # However, VIC runs 3-hourly: a tiny snowfall (0.5mm) briefly resets MELTING=false,
    # then IMMEDIATELY re-triggers THAW at the next sub-hourly step (daytime melt).
    # In mGV's daily timestepping, a 0.5mm snowfall locks ACCUM mode for the FULL day,
    # causing albedo feedback divergence. Fix: use higher threshold for snow_reset (5mm)
    # so only substantial snowfall resets melt_flag in the daily model.
    # The lsnow counter (albedo aging) still resets on all snowfall > 0.001mm.
    SNW_MELT_RESET_THRESH_MM = T_Type(5.0)  # kept for reference
    CC_MELT_DEADBAND = zero(T_Type)  # No deadband: CC >= 0 triggers THAW (VIC-faithful)
    flag_cond1 = (current_cc >= CC_MELT_DEADBAND) & in_melt_season
    # Snow_reset: THAW → ACCUM only if snowfall AND pack truly cold (CC < 0).
    # VIC 3-hourly: a tiny snowfall briefly resets MELTING=false, but daytime melt immediately
    # re-triggers it. In mGV daily, we gate by CC: CC >= 0 (warm pack) means snowfall won't
    # keep pack cold for a full day → don't reset melt_flag.
    # CC < 0 (cold pack): snowfall can genuinely cool it further → allow ACCUM reset.
    pack_is_cold_for_reset = current_cc < zero(T_Type)
    snow_reset = is_trace & (melt_flag == Int32(1)) & pack_is_cold_for_reset
    # Apply: set flag=1 if flag_cond1; reset to 0 if genuine cold-pack snowfall; otherwise retain
    melt_flag = ifelse(has_swe,
        ifelse(snow_reset, Int32(0), ifelse(flag_cond1, Int32(1), melt_flag)),
        Int32(0))


    # 4. Surface temp / melt (Branchless)
    melt = zero(T_Type)
    prev_ts = snow_surf_temp[i, j, b, v]
    prev_ts = ifelse(isnan(prev_ts), zero(T_Type), prev_ts)
    # Initial guess for NR: use prev_ts if close to air temp, otherwise start from 0 or t_avg
    # This prevents the solver from being trapped near 0°C when the true equilibrium is cold
    t_s = ifelse((prev_ts >= t_avg - T_Type(5.0)) & (prev_ts <= zero(T_Type)), prev_ts, min(t_avg, zero(T_Type)))

    eff_alb = ifelse(isnan(alb), SNW_NEW_SNOW_ALB, alb)
    # Surface SWE (m) for deltaCC thermal inertia (VIC: SweSurfaceLayer)
    swe_surf_m_nr = min(current_swe / T_Type(1000.0), SNW_MAX_SURFACE_SWE_MM / T_Type(1000.0))

    ts_solved, melt_energy_at_zero, sub_mass_mm = snow_surface_temp_nr(t_s, t_avg, sw_in, lw_in, eff_alb, ps, ra, vp_air, swe_surf_m_nr, T_Type)
    t_s = ifelse(has_swe, ts_solved, T_Type(NaN))
    
    melt_J = ifelse(has_swe & (melt_energy_at_zero > zero(T_Type)), melt_energy_at_zero * T_Type(86400.0), zero(T_Type))
    
    # Diurnal melt fraction (seasonal): VIC's 3-hourly steps show melt only during
    # daytime hours at marginal Ta. Outside the main melt season (DOY 60-250), thin
    # autumn/spring snow is at risk of being over-melted by daily-mean SW.
    # We apply a daytime fraction correction ONLY in the off-season (DOY <60 or DOY>250).
    # During peak melt season (DOY 60-250), full melt rate applies.
    # f_day at Ta=0°C → 0.71 (29% reduction); at Ta>+DTR_half → 1.0; at Ta<-DTR_half → 0.
    SNW_DTR_HALF_MELT   = T_Type(3.0)
    in_off_season = (day_of_year < Int32(60)) | (day_of_year > Int32(250))
    f_day = sqrt(clamp((t_avg + SNW_DTR_HALF_MELT) / (T_Type(2.0) * SNW_DTR_HALF_MELT), zero(T_Type), one(T_Type)))
    f_melt = ifelse(in_off_season, f_day, one(T_Type))
    melt_J = melt_J * f_melt
    
    # Pack CC satisfying
    energy_needed_sfc = max(-current_cc, zero(T_Type))
    melt_apply_sfc = min(melt_J, energy_needed_sfc)
    melt_J -= melt_apply_sfc
    current_cc = min(current_cc + melt_apply_sfc, zero(T_Type))

    energy_needed_pack = max(-current_pcc, zero(T_Type))
    melt_apply_pack = min(melt_J, energy_needed_pack)
    melt_J -= melt_apply_pack
    current_pcc = min(current_pcc + melt_apply_pack, zero(T_Type))

    # Phase change out of remaining melt_J
    # phase_melt converts ice -> liquid water (stays in pack until it drains)
    phase_melt = ifelse(has_swe & (melt_energy_at_zero > zero(T_Type)), melt_J / (SNW_LATICE * SNW_RHOFW) * T_Type(1000.0), zero(T_Type))
    
    # Clamp phase melt to available ice  (ice = SWE - existing liquid)
    swe_ice = max(current_swe - c_surf_water - c_pack_water, zero(T_Type))
    phase_melt = min(phase_melt, swe_ice)

    # Refreeze liquid if cold content wasn't fully satisfied by melt
    refreeze_energy_sfc = min(-current_cc, c_surf_water * (SNW_LATICE * SNW_RHOFW) / T_Type(1000.0))
    refreeze_surf_val   = refreeze_energy_sfc / (SNW_LATICE * SNW_RHOFW) * T_Type(1000.0)
    c_surf_water       -= refreeze_surf_val
    current_cc         += refreeze_energy_sfc
    
    refreeze_energy_pack = min(-current_pcc, c_pack_water * (SNW_LATICE * SNW_RHOFW) / T_Type(1000.0))
    refreeze_pack_val    = refreeze_energy_pack / (SNW_LATICE * SNW_RHOFW) * T_Type(1000.0)
    c_pack_water        -= refreeze_pack_val
    current_pcc         += refreeze_energy_pack

    # Melted ice becomes liquid in surface layer (VIC: surf_water += SnowMelt)
    # phase_melt converts ice -> liquid; SWE stays the same (liquid still inside pack)
    c_surf_water += p_rain_snowpack + phase_melt

    # ---- Surface layer drainage ----
    # VIC: MaxLiquidWater = SNOW_LIQUID_WATER_CAPACITY * SurfaceSwq
    #      melt[0] = surf_water - MaxLiquidWater  (surface outflow to pack)
    swe_surf_m = min(current_swe / T_Type(1000.0), SNW_MAX_SURFACE_SWE_MM / T_Type(1000.0))
    swe_pack_m = max(current_swe / T_Type(1000.0) - SNW_MAX_SURFACE_SWE_MM / T_Type(1000.0), zero(T_Type))
    
    SNW_LIQUID_WATER_CAPACITY = T_Type(0.03) # VIC default 3% of ICE SWE
    
    max_liq_surf = SNW_LIQUID_WATER_CAPACITY * (swe_surf_m * T_Type(1000.0))
    surf_drain = max(c_surf_water - max_liq_surf, zero(T_Type))
    c_surf_water -= surf_drain
    
    # Surface outflow feeds pack layer (VIC: pack_water += melt[0])
    c_pack_water += surf_drain
    
    # ---- Pack layer refreeze (if pack cold content still negative) ----
    # VIC: if PackCC < -PackRefreezeEnergy, refreeze all; else partial refreeze
    # Simple branchless approximation: refreeze whatever cold content demands
    refreeze_pack2_val = min(c_pack_water, max(-current_pcc / (SNW_LATICE * SNW_RHOFW) * T_Type(1000.0), zero(T_Type)))
    c_pack_water -= refreeze_pack2_val
    current_pcc  = min(current_pcc + refreeze_pack2_val * (SNW_LATICE * SNW_RHOFW) / T_Type(1000.0), zero(T_Type))

    # ---- Pack layer drainage ----
    # VIC: MaxLiquidWater = SNOW_LIQUID_WATER_CAPACITY * PackSwq
    #      melt[0] = pack_water - MaxLiquidWater  (pack outflow = melt)
    max_liq_pack = SNW_LIQUID_WATER_CAPACITY * (swe_pack_m * T_Type(1000.0))
    pack_drain = max(c_pack_water - max_liq_pack, zero(T_Type))
    c_pack_water -= pack_drain

    # ---- SWE update: subtract sublimated ice ----
    ice_remaining = max(swe_ice - phase_melt - sub_mass_mm, zero(T_Type))
    current_swe   = max(ice_remaining + c_surf_water + c_pack_water, zero(T_Type))
    melt          = pack_drain   # melt output = liquid leaving the snowpack
    
    # The output variable OUT_SNOW_MELT is only the snowpack outflow (pack drain)
    # VIC: snow->melt = store_melt (pack outflow only; bare rain goes to ppt separately)
    melt_out_val = pack_drain

    # Distribute current_swe bounds update
    swe_surf_m = min(current_swe / T_Type(1000.0), SNW_MAX_SURFACE_SWE_MM / T_Type(1000.0))
    swe_pack_m = max(current_swe / T_Type(1000.0) - SNW_MAX_SURFACE_SWE_MM / T_Type(1000.0), zero(T_Type))
    
    cc_melt_branch = min(prior_cc_orig * T_Type(0.01), zero(T_Type))
    pcc_melt_branch = ifelse(swe_pack_m > zero(T_Type), SNW_VCPICE_WQ * (swe_pack_m * T_Type(1000.0)) * (t_s * T_Type(0.5)), current_pcc)
    pcc_melt_branch = min(pcc_melt_branch, zero(T_Type))
    ts_for_cc = t_s  # Use raw ts from NR solver (VCPICE now correctly calibrated)
    cc_nomelt_branch = min(SNW_VCPICE_WQ * (swe_surf_m * T_Type(1000.0)) * ts_for_cc, zero(T_Type))
    # Eliminate fake thermal inertia drift for the deep pack; VIC only cools deep pack via mass transfer
    pcc_nomelt_branch = current_pcc

    is_melting_step = melt > zero(T_Type)
    
    # Cold content after a melt step:
    # VIC: CC gets satisfied by melt energy in lines 290-298 (current_cc → 0 when fully melting).
    # mGV must NOT override this with cc_melt_branch (which would reset CC back to tiny negative).
    # Correct approach:
    # - Thick packs (f_deep ≈ 1): use CC from energy-balance satisfaction (lines 290-298).
    #   When CC is fully satisfied → CC = 0 → melt_flag can trigger next step.
    # - Thin packs (f_deep < 1, SWE < 100mm): apply nighttime CC correction to prevent
    #   runaway autumn/off-season albedo feedback from tiny melt events at marginal temps.
    SNW_DEEP_SWE_MM   = T_Type(100.0)
    SNW_DTR_HALF_CC   = T_Type(4.0)
    sf_temp_night     = min(t_avg - SNW_DTR_HALF_CC, zero(T_Type))  # proxy nighttime T
    cc_melt_night     = min(SNW_VCPICE_WQ * (swe_surf_m * T_Type(1000.0)) * sf_temp_night, zero(T_Type))
    # f_deep blends between nighttime-fix (thin) and physical-CC (thick)
    f_deep = clamp(current_swe / SNW_DEEP_SWE_MM, zero(T_Type), one(T_Type))
    # For thick packs: preserve energy-balance CC (current_cc from lines 290-298).
    # For thin packs: use nighttime-proxy CC to prevent spurious off-season THAW.
    # The blend: at f_deep=1 → use current_cc (physical); at f_deep=0 → nighttime fix.
    cc_melt_physical = current_cc  # already updated by energy balance (lines 290-298)
    cc_melt_eff = f_deep * cc_melt_physical + (one(T_Type) - f_deep) * cc_melt_night
    
    current_cc = ifelse(is_melting_step, cc_melt_eff, ifelse(t_s < zero(T_Type), cc_nomelt_branch, current_cc))
    current_pcc = ifelse(is_melting_step, pcc_melt_branch, ifelse(t_s < zero(T_Type), pcc_nomelt_branch, current_pcc))

    # Update melt_flag post-NR: based on FINAL cold content (post energy-balance update).
    # VIC (3-hourly): recalculates MELTING from coldcontent >= 0 at EACH sub-step.
    # In the off-season (DOY < 60 or DOY > 250): if CC is restored negative by nighttime fix,
    # reset melt_flag → 0 (ACCUM mode). This prevents THAW-albedo decay from small autumn
    # melt events at marginal temperatures (matching VIC's nighttime CC behavior).
    # In the main melt season (DOY 60-250): flag_cond1 triggers THAW when CC >= 0 (VIC-faithful).
    # CC reaches 0 naturally when thick-pack energy balance fully satisfies cold content.
    cc_reset_flag = in_off_season & (current_cc < zero(T_Type))
    melt_flag = ifelse(has_swe,
        ifelse(cc_reset_flag,
            Int32(0),  # Off-season + CC negative → ACCUM mode
            ifelse(is_melting_step & in_melt_season & !snow_reset, Int32(1), melt_flag)),
        Int32(0))




    # VIC trace-snow pruning: if SWE drops below the minimum meaningful threshold,
    # completely zero out the snowpack. This prevents ghost-snow cells (0.001-0.010 mm)
    # that would otherwise inflate basin-mean SWE by ~5x relative to VIC.
    # VIC applies this implicitly via its minimum accumulation threshold.
    above_trace = current_swe >= SNW_TRACESNOW_MM
    current_swe  = ifelse(above_trace, current_swe, zero(T_Type))
    c_surf_water = ifelse(above_trace, c_surf_water, zero(T_Type))
    c_pack_water = ifelse(above_trace, c_pack_water, zero(T_Type))
    melt         = ifelse(above_trace, melt, zero(T_Type))

    current_cc  = ifelse(above_trace, current_cc,  zero(T_Type))
    current_pcc = ifelse(above_trace, current_pcc, zero(T_Type))
    t_s = ifelse(above_trace, t_s, T_Type(NaN))

    # 5. Snow Depth
    current_depth_m = (current_swe / T_Type(1000.0)) * (SNW_RHOFW / SNW_DENSITY)

    # 6. Coverage: VIC binary model (options.SPATIAL_SNOW = false)
    # VIC solve_snow.c:402-408: if swq > 0 → coverage = 1; else coverage = 0
    # No sub-grid distribution needed; entire cell is covered when snow is present.
    new_coverage = ifelse(current_swe > SNW_TRACESNOW_MM, one(T_Type), zero(T_Type))

    # 7. Write outputs (Branchless Masked)
    swe[i, j, b, v]                  = ifelse(active, current_swe, zero(T_Type))
    surf_water[i, j, b, v]           = ifelse(active, ifelse(current_swe > zero(T_Type), c_surf_water, zero(T_Type)), zero(T_Type))
    pack_water[i, j, b, v]           = ifelse(active, ifelse(current_swe > zero(T_Type), c_pack_water, zero(T_Type)), zero(T_Type))
    snow_depth[i, j, b, v]           = ifelse(active, ifelse(current_swe > zero(T_Type), current_depth_m * T_Type(1000.0), zero(T_Type)), zero(T_Type))
    snow_albedo[i, j, b, v]          = ifelse(active, ifelse((current_swe > zero(T_Type)) & !isnan(alb), alb, T_Type(NaN)), T_Type(NaN))
    snow_surf_temp[i, j, b, v]       = ifelse(active, ifelse(current_swe > zero(T_Type), t_s, T_Type(NaN)), T_Type(NaN))
    snow_coverage[i, j, b, v]        = ifelse(active, new_coverage, zero(T_Type))
    melt_out[i, j, b, v]             = ifelse(active, melt_out_val, zero(T_Type))
    last_snow[i, j, b, v]            = ifelse(active, ifelse(current_swe > zero(T_Type), lsnow, Int32(0)), Int32(0))
    cold_content[i, j, b, v]         = ifelse(active, ifelse(current_swe > zero(T_Type), current_cc, zero(T_Type)), zero(T_Type))
    pack_cold_content[i, j, b, v]    = ifelse(active, ifelse(current_swe > zero(T_Type), current_pcc, zero(T_Type)), zero(T_Type))
    melting_flag[i, j, b, v]         = ifelse(active, ifelse(current_swe > zero(T_Type), melt_flag, Int32(0)), Int32(0))
    store_snow[i, j, b, v]           = ifelse(active, ifelse(current_swe > zero(T_Type), st_snow, Int32(0)), Int32(0))
    snow_distrib_slope[i, j, b, v]   = ifelse(active, ifelse(current_swe > zero(T_Type), dslope, zero(T_Type)), zero(T_Type))
    store_swq[i, j, b, v]            = ifelse(active, ifelse(current_swe > zero(T_Type), st_swq, zero(T_Type)), zero(T_Type))
    store_coverage[i, j, b, v]       = ifelse(active, ifelse(current_swe > zero(T_Type), st_cov, zero(T_Type)), zero(T_Type))
    max_snow_depth[i, j, b, v]       = ifelse(active, ifelse(current_swe > zero(T_Type), mx_depth, zero(T_Type)), zero(T_Type))

    nothing
end

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
function calculate_snow_dynamics!(
    swe_gpu, surf_water_gpu, pack_water_gpu, snow_depth_gpu, snow_albedo_gpu, snow_surf_temp_gpu,
    snow_coverage_gpu, snow_melt_gpu,
    last_snow_gpu, cold_content_gpu, pack_cc_gpu, melting_flag_gpu,
    store_snow_gpu, snow_distrib_slope_gpu,
    store_swq_gpu, store_coverage_gpu, max_snow_depth_gpu,
    throughfall_4d, tair_3d, swdown_gpu, lwdown_gpu, psurf_gpu, vp_gpu, wind_gpu,
    AreaFract_gpu, cv_gpu, annual_prec_gpu,
    day_of_year::Int, lat_mean::Float64
)
    lat_pos = Int32(lat_mean >= 0.0 ? 1 : 0)

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
