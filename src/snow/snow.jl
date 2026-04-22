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
const SNW_TRACESNOW_MM       = FloatType(0.001)   # mm — trace snowfall threshold
const SNW_LIQUID_WATER_CAP   = FloatType(0.035)   # fraction liquid water capacity
const SNW_MAX_SURFACE_SWE_MM = FloatType(125.0)   # mm — surface layer limit
const SNW_VCPICE_WQ          = FloatType(2062.0)  # J/(kg·K): heat capacity ice
const SNW_LATICE             = FloatType(334000.0) # J/kg — latent heat of fusion
const SNW_RHOFW              = FloatType(1000.0)   # kg/m³ — density of water
const SNW_DENSITY            = FloatType(250.0)    # kg/m³ — typical snow density

# ---------------------------------------------------------------------------
# Newton-Raphson snow surface temperature solver (inline, GPU-compatible)
# Finds Tsurf that satisfies the snow surface energy balance for Tsurf <= 0°C.
# ---------------------------------------------------------------------------
@inline function snow_surface_temp_nr(
    tsurf_init,   # initial guess (°C), previous timestep
    Ta,           # air temperature (°C)
    sw_in,        # incoming SW (W/m²)
    lw_in,        # incoming LW (W/m²)
    albedo,       # snow albedo
    psurf,        # surface pressure (Pa)
    ra,           # aerodynamic resistance (s/m)
    vp_air,       # vapor pressure of air (Pa)
    T_Type
)
    # Air density from ideal gas: ρ = P / (Rd * Tk)
    Ta_K    = Ta + T_Type(273.15)
    rho_air = psurf / (T_Type(287.0) * Ta_K)
    Cp_air  = T_Type(1004.0)
    ha      = rho_air * Cp_air / max(ra, T_Type(1.0))   # W/(m²·K)

    # Constant part of energy balance RHS (NO ground flux — used for melt trigger)
    sw_net    = sw_in * (one(T_Type) - albedo)
    
    # Pre-compute sublimation latent heat at 0°C to offset the energy balance
    ps_eff = max(psurf, T_Type(60000.0))
    vp_a_pa = clamp(vp_air * T_Type(1000.0), zero(T_Type), T_Type(15000.0))
    q_air  = T_Type(0.622) * vp_a_pa / (ps_eff - T_Type(0.378) * vp_a_pa)
    
    es0    = T_Type(611.0)
    q_sat0 = T_Type(0.622) * es0 / (ps_eff - T_Type(0.378) * es0)
    L_sub  = T_Type(2.845e6)
    LE_sub_0 = rho_air * L_sub / max(ra, T_Type(1.0)) * (q_sat0 - q_air)
    
    rhs_const = sw_net + lw_in + ha * Ta - LE_sub_0

    # LW emission coefficient: σ·ε (snow emissivity ≈ 0.99)
    eps_snow = T_Type(0.99)
    sig_eps  = sigma * eps_snow

    # Newton-Raphson: 12 iterations (more than enough for T^4 + linear convergence)
    # Step clamped to 10°C per iteration to avoid divergence
    ts = tsurf_init
    for _ in 1:12
        Ts_K = ts + T_Type(273.15)
        lw_out  = sig_eps * (Ts_K ^ T_Type(4.0))
        h_sens  = ha * ts
        f_val   = lw_out + h_sens - rhs_const
        df_val  = T_Type(4.0) * sig_eps * (Ts_K ^ T_Type(3.0)) + ha
        step    = f_val / max(abs(df_val), T_Type(1e-6))
        step    = clamp(step, -T_Type(10.0), T_Type(10.0))
        ts      = ts - step
    end

    # --- Branchless Evaluation ---
    # Path A: ts > 0 (Melt occurs) — energy at Ts=0°C goes to phase change
    Ts0_K    = T_Type(273.15)
    lw_out0  = sig_eps * (Ts0_K ^ T_Type(4.0))
    # Include ground heat flux at Ts=0: Qg = SNOW_CONDUCT * density^norm * (Tg=0 - 0) / depth = 0
    melt_energy_net = rhs_const - lw_out0   # h_sens=0 at Ts=0, Qg=0 at Ts=0
    
    ts_melt = zero(T_Type)
    melt_heat_out = max(melt_energy_net, zero(T_Type))

    # Path B: ts <= 0 (No melt)
    # Ground heat flux warms snow from below: Qg = SNOW_CONDUCT * (T_ground=0 - ts) / depth
    # VIC default SNOW_CONDUCT = 0.31 W/m/K, typical depth ~0.5m, density factor ~1
    # Effective conductance ≈ 0.31/0.5 = 0.62 W/m²/K → Qg ≈ 0.62 * (-ts)
    # This warms the snow surface (reduces cold content), matching VIC's deltaCC correction
    Qg_conductance = T_Type(0.62)   # W/m²/K — VIC SNOW_CONDUCT/depth 
    Qg_from_ground = Qg_conductance * (-ts)    # > 0 when ts < 0 (warming from ground)
    
    # Correct ts for ground heating: ts_corrected satisfies full energy balance with Qg
    Ts_K_final = ts + T_Type(273.15)
    dLW_dT = T_Type(4.0) * sig_eps * (max(Ts_K_final, T_Type(1.0)) ^ T_Type(3.0))
    # Solving: dF/dT * dts = Qg_from_ground
    ts_no_melt = min(ts + Qg_from_ground / max(dLW_dT + ha + Qg_conductance, T_Type(1.0)), zero(T_Type))
    
    # Merge
    is_melting = ts > zero(T_Type)
    final_ts = ifelse(is_melting, ts_melt, ts_no_melt)
    final_melt_energy = ifelse(is_melting, melt_heat_out, zero(T_Type))

    return final_ts, final_melt_energy
end

# ---------------------------------------------------------------------------
# VIC-faithful 4D snow dynamics kernel
# ---------------------------------------------------------------------------
@kernel function snow_dynamics_kernel!(
    # 4D State (in/out): (nx, ny, nbands, nveg)
    swe, surf_water, pack_water, snow_depth, snow_albedo, snow_surf_temp, snow_coverage, melt_out,
    last_snow, cold_content, pack_cold_content, melting_flag,
    store_snow, snow_distrib_slope, store_swq, store_coverage, max_snow_depth,
    @Const(throughfall_4d), @Const(tair_band), @Const(swdown_2d), @Const(lwdown_2d), 
    @Const(psurf_2d), @Const(vp_2d), @Const(AreaFract), @Const(cv_4d), @Const(annual_prec_2d),
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

    ann_prec = annual_prec_2d[i, j]
    max_distrib_slope = ifelse(isnan(ann_prec) | (ann_prec <= zero(T_Type)), T_Type(0.4), ann_prec / T_Type(500.0))

    # 1. Add snowfall and rain to SWE and update cold content (Branchless)
    old_swq_pre = current_swe
    current_swe += p_snow + p_rain_snowpack
    sf_cc = ifelse((p_snow > zero(T_Type)) & (t_avg < zero(T_Type)), SNW_VCPICE_WQ * (p_snow / T_Type(1000.0)) * t_avg, zero(T_Type))
    current_cc = min(current_cc + sf_cc, zero(T_Type))

    rain_heat = ifelse((p_rain > zero(T_Type)) & (current_swe > zero(T_Type)) & (t_avg > zero(T_Type)), T_Type(4186.0) * (p_rain / T_Type(1000.0)) * t_avg, zero(T_Type))
    current_cc = min(current_cc + rain_heat, zero(T_Type))

    # 2. Albedo (Branchless)
    is_trace = p_snow > SNW_TRACESNOW_MM
    is_new = is_trace & (current_cc < zero(T_Type))
    has_swe = current_swe > zero(T_Type)

    lsnow = ifelse(is_new, Int32(0), ifelse(has_swe, lsnow + Int32(1), Int32(0)))
    ls_f = T_Type(lsnow)
    
    alb_accum = SNW_NEW_SNOW_ALB * (SNW_ALB_ACCUM_A ^ (ls_f ^ SNW_ALB_ACCUM_B))
    alb_thaw = SNW_NEW_SNOW_ALB * (SNW_ALB_THAW_A  ^ (ls_f ^ SNW_ALB_THAW_B))
    is_accum = (current_cc < zero(T_Type)) & (melt_flag == Int32(0))

    alb_age = ifelse(is_accum, alb_accum, alb_thaw)
    alb = ifelse(is_new, SNW_NEW_SNOW_ALB, ifelse(has_swe, max(alb_age, T_Type(0.45)), T_Type(NaN)))

    # 3. melting flag
    in_melt_season = ifelse(lat_positive == Int32(1), 
                            (day_of_year > Int32(60)) & (day_of_year < Int32(273)),
                            (day_of_year < Int32(60)) | (day_of_year > Int32(273)))
    flag_cond1 = (current_cc >= zero(T_Type)) & in_melt_season
    flag_cond2 = (melt_flag == Int32(1)) & (p_snow > SNW_TRACESNOW_MM)
    melt_flag = ifelse(has_swe, ifelse(flag_cond1, Int32(1), ifelse(flag_cond2, Int32(0), melt_flag)), Int32(0))

    # 4. Surface temp / melt (Branchless)
    melt = zero(T_Type)
    prev_ts = snow_surf_temp[i, j, b, v]
    prev_ts = ifelse(isnan(prev_ts), zero(T_Type), prev_ts)
    # Initial guess for NR: use prev_ts if close to air temp, otherwise start from 0 or t_avg
    # This prevents the solver from being trapped near 0°C when the true equilibrium is cold
    t_s = ifelse((prev_ts >= t_avg - T_Type(5.0)) & (prev_ts <= zero(T_Type)), prev_ts, min(t_avg, zero(T_Type)))

    eff_alb = ifelse(isnan(alb), SNW_NEW_SNOW_ALB, alb)
    ra = T_Type(100.0)

    ts_solved, melt_energy_at_zero = snow_surface_temp_nr(t_s, t_avg, sw_in, lw_in, eff_alb, ps, ra, vp_air, T_Type)
    t_s = ifelse(has_swe, ts_solved, T_Type(NaN))
    
    melt_J = ifelse(has_swe & (melt_energy_at_zero > zero(T_Type)), melt_energy_at_zero * T_Type(86400.0), zero(T_Type))
    
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

    # ---- SWE update (matches VIC: swq = Ice + pack_water + surf_water) ----
    # Ice was reduced by phase_melt; SWE = ice_remaining + all_liquid
    ice_remaining = swe_ice - phase_melt
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
    cc_nomelt_branch = min(SNW_VCPICE_WQ * (swe_surf_m * T_Type(1000.0)) * t_s, zero(T_Type))
    # Eliminate fake thermal inertia drift for the deep pack; VIC only cools deep pack via mass transfer
    pcc_nomelt_branch = current_pcc

    is_melting_step = melt > zero(T_Type)
    current_cc = ifelse(is_melting_step, cc_melt_branch, ifelse(t_s < zero(T_Type), cc_nomelt_branch, current_cc))
    current_pcc = ifelse(is_melting_step, pcc_melt_branch, ifelse(t_s < zero(T_Type), pcc_nomelt_branch, current_pcc))

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

    # 6. Branchless Coverage Update
    is_p_snow = p_snow > zero(T_Type)
    is_melt = (melt > zero(T_Type)) & (!is_p_snow)
    
    # Path A:
    cov_snw = one(T_Type)
    st_snow_cond1 = old_coverage < one(T_Type)
    new_st_snow_A = ifelse(st_snow == Int32(0), ifelse(st_snow_cond1, Int32(1), st_snow), st_snow)
    new_st_swq_A = ifelse(st_snow == Int32(0), ifelse(st_snow_cond1, current_swe - old_swq_pre, st_swq), st_swq)
    new_st_cov_A = ifelse(st_snow == Int32(0), ifelse(st_snow_cond1, old_coverage, st_cov), st_cov)

    st_snow_cond2 = (st_snow != Int32(0)) & (st_swq == zero(T_Type))
    new_st_cov_A2 = ifelse(st_snow_cond2, ifelse(old_coverage < one(T_Type), old_coverage, one(T_Type)), new_st_cov_A)
    
    st_swq_A2_add = current_swe - old_swq_pre
    new_st_swq_A2 = ifelse(st_snow != Int32(0), st_swq + st_swq_A2_add, new_st_swq_A)

    st_snow_cond3 = (current_depth_m >= max_distrib_slope / T_Type(2.0)) & (st_snow != Int32(0))
    new_st_snow_A3 = ifelse(st_snow_cond3, Int32(0), new_st_snow_A)
    new_st_swq_A3 = ifelse(st_snow_cond3, zero(T_Type), new_st_swq_A2)
    new_dslope_A3 = ifelse(st_snow_cond3, zero(T_Type), dslope)
    new_st_cov_A3 = ifelse(st_snow_cond3, one(T_Type), new_st_cov_A2)

    # Path B:
    st_swq_cond1 = (st_swq > zero(T_Type)) & (current_swe < old_swq_pre)
    st_swq_B1 = ifelse(st_swq_cond1, st_swq + (current_swe - old_swq_pre), st_swq)
    st_swq_cond2 = st_swq_B1 <= zero(T_Type)
    new_st_swq_B  = ifelse(st_swq_cond1 & st_swq_cond2, zero(T_Type), st_swq_B1)
    new_old_cov_B = ifelse(st_swq_cond1 & st_swq_cond2, st_cov, old_coverage)
    new_st_cov_B  = ifelse(st_swq_cond1 & st_swq_cond2, one(T_Type), st_cov)

    dslope_cond1 = (new_st_swq_B == zero(T_Type)) & (dslope == zero(T_Type))
    cap_depth = min(max_distrib_slope, T_Type(2.0) * old_depth_m)
    new_dslope_B = ifelse(dslope_cond1, -cap_depth, dslope)
    mx_depth_B   = ifelse(dslope_cond1, cap_depth, mx_depth)
    new_st_snow_B = ifelse(dslope_cond1, Int32(1), st_snow)

    old_mx_depth_B = mx_depth_B
    new_mx_depth_B2 = ifelse(new_st_swq_B == zero(T_Type), T_Type(2.0) * current_depth_m, mx_depth_B)

    cov_cond_B = (new_mx_depth_B2 < old_mx_depth_B) | (old_mx_depth_B == zero(T_Type))
    raw_cov = ifelse(new_dslope_B != zero(T_Type), -new_mx_depth_B2 / new_dslope_B, one(T_Type))
    cov_snw_B = ifelse((new_st_swq_B == zero(T_Type)) & cov_cond_B, clamp(raw_cov, zero(T_Type), one(T_Type)), new_old_cov_B)

    # Combine A and B
    new_coverage = ifelse(is_p_snow, cov_snw, ifelse(is_melt, cov_snw_B, old_coverage))
    st_snow = ifelse(is_p_snow, new_st_snow_A3, ifelse(is_melt, new_st_snow_B, st_snow))
    st_swq  = ifelse(is_p_snow, new_st_swq_A3, ifelse(is_melt, new_st_swq_B, st_swq))
    st_cov  = ifelse(is_p_snow, new_st_cov_A3, ifelse(is_melt, new_st_cov_B, st_cov))
    dslope  = ifelse(is_p_snow, new_dslope_A3, ifelse(is_melt, new_dslope_B, dslope))
    mx_depth = ifelse(is_p_snow, mx_depth, ifelse(is_melt, new_mx_depth_B2, mx_depth))
    
    new_coverage = ifelse(current_swe <= zero(T_Type), zero(T_Type), new_coverage)

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
    throughfall_4d, tair_3d, swdown_gpu, lwdown_gpu, psurf_gpu, vp_gpu,
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
        throughfall_4d, tair_3d, swdown_gpu, lwdown_gpu, psurf_gpu, vp_gpu,
        AreaFract_gpu, cv_gpu, annual_prec_gpu,
        Int32(day_of_year), lat_pos;
        ndrange=size(swe_gpu)
    )
    KernelAbstractions.synchronize(device_backend)
end
