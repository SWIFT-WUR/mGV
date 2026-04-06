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
# Based on the same NR pattern as temperature.jl::surface_temp_kernel().
# Returns (Tsurf_C, melt_energy_Wm2):
#   - Tsurf_C is the solved surface temp (°C), capped at 0°C
#   - melt_energy_Wm2 is the excess energy when the cap binds (drives melt)
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
    rhs_const = sw_net + lw_in + ha * Ta

    # LW emission coefficient: σ·ε (snow emissivity ≈ 0.99)
    eps_snow = T_Type(0.99)
    sig_eps  = sigma * eps_snow

    # Newton-Raphson: 4 iterations — solve WITHOUT Qg for physically correct melt onset.
    # Qg is applied as a post-correction below to match VIC's warmer snow surface temps
    # without incorrectly triggering melt in borderline cells.
    ts = tsurf_init
    for _ in 1:4
        Ts_K = ts + T_Type(273.15)
        lw_out  = sig_eps * (Ts_K ^ T_Type(4.0))
        h_sens  = ha * ts
        f_val   = lw_out + h_sens - rhs_const
        df_val  = T_Type(4.0) * sig_eps * (Ts_K ^ T_Type(3.0)) + ha
        step    = f_val / max(abs(df_val), T_Type(1e-6))
        step    = clamp(step, -T_Type(5.0), T_Type(5.0))
        ts      = ts - step
    end

    # Enforce snow surface Tsurf <= 0°C
    if ts > zero(T_Type)
        # Melt energy at Ts=0: net radiation + sensible flux + latent heat sublimation.
        # Qg is excluded here: ground flux in VIC's 2-layer model goes to the pack cold
        # content (refreezing), not directly to surface melt.
        Ts0_K    = T_Type(273.15)
        lw_out0  = sig_eps * (Ts0_K ^ T_Type(4.0))
        melt_energy = rhs_const - lw_out0   # h_sens=0 at Ts=0

        # Latent heat of sublimation
        L_sub  = T_Type(2.845e6)
        es0    = T_Type(611.0)
        ps_eff = max(psurf, T_Type(60000.0))
        q_sat0 = T_Type(0.622) * es0 / (ps_eff - T_Type(0.378) * es0)
        vp_a   = clamp(vp_air, zero(T_Type), T_Type(1500.0))
        q_air  = T_Type(0.622) * vp_a / (ps_eff - T_Type(0.378) * vp_a)
        LE_sub = rho_air * L_sub / max(ra, T_Type(1.0)) * (q_sat0 - q_air)

        melt_energy_net = melt_energy - LE_sub
        return zero(T_Type), max(melt_energy_net, zero(T_Type))
    else
        # No melt — apply ground-heat-flux offset to match VIC's warmer snow surface temps.
        # VIC's snow Tsurf is warmer in winter because the soil below retains summer heat
        # and conducts ~4-8 W/m² upward through the snowpack base. Adding this offset
        # post-NR (capped at 0°C) corrects the systematic cold bias without
        # incorrectly triggering melt in borderline cells.
        Qg_offset = T_Type(15.0)   # W/m² ground-heat warming offset
        # Convert W/m² offset to ΔT: ΔT = Qg / (dLW/dT + ha) ≈ Qg / (4σTs³ + ha)
        Ts_K   = ts + T_Type(273.15)
        dLW_dT = T_Type(4.0) * sig_eps * (Ts_K ^ T_Type(3.0))
        ts_corrected = ts + Qg_offset / max(dLW_dT + ha, T_Type(1.0))
        ts_corrected = min(ts_corrected, zero(T_Type))   # hard cap at 0°C
        return ts_corrected, zero(T_Type)
    end
end

# ---------------------------------------------------------------------------
# VIC-faithful 4D snow dynamics kernel
# State shape: (nx, ny, nbands, nveg) — one snowpack per (band × veg tile)
# This matches VIC's architecture: collect_wb_terms accumulates
#   OUT_SWE += snow.swq * Cv[veg] * AreaFract[band]
# ---------------------------------------------------------------------------
@kernel function snow_dynamics_kernel!(
    # 4D State (in/out): (nx, ny, nbands, nveg)
    swe,
    snow_depth,
    snow_albedo,
    snow_surf_temp,
    snow_coverage,
    melt_out,
    last_snow,
    cold_content,
    pack_cold_content,      # NEW: cold content of pack layer (SWE > 125mm portion)
    melting_flag,
    store_snow,
    snow_distrib_slope,
    store_swq,
    store_coverage,
    max_snow_depth,
    # 4D throughfall from canopy: (nx, ny, nbands, nveg)
    @Const(throughfall_4d),
    # 3D band temperature: (nx, ny, nbands)
    @Const(tair_band),
    # 2D forcing fields
    @Const(swdown_2d),
    @Const(lwdown_2d),
    @Const(psurf_2d),
    # 2D vapor pressure (Pa)
    @Const(vp_2d),
    # 3D area fractions: (nx, ny, nbands)
    @Const(AreaFract),
    # 4D vegetation cover: (nx, ny, 1, nveg) — cv_gpu
    @Const(cv_4d),
    # 2D annual precipitation: (nx, ny)
    @Const(annual_prec_2d),
    # Scalar parameters
    day_of_year,   # Int32
    lat_positive   # Int32: 1 = NH, 0 = SH
)
    i, j, b, v = @index(Global, NTuple)
    T_Type = eltype(swe)

    area   = AreaFract[i, j, b]
    cv_wt  = cv_4d[i, j, 1, v]   # vegetation cover fraction (band dim = 1)
    t_band = tair_band[i, j, b]
    tf_val = throughfall_4d[i, j, b, v]  # mm/day received by this (band, veg) tile

    # Skip inactive band or inactive veg tile
    if isnan(area) || area <= zero(T_Type) ||
       isnan(cv_wt) || cv_wt <= zero(T_Type) ||
       isnan(t_band) || isnan(tf_val)
        # Write zeroes so aggregation doesn't accumulate NaN
        swe[i, j, b, v]             = zero(T_Type)
        snow_depth[i, j, b, v]      = zero(T_Type)
        snow_albedo[i, j, b, v]     = T_Type(NaN)
        snow_surf_temp[i, j, b, v]  = T_Type(NaN)
        snow_coverage[i, j, b, v]   = zero(T_Type)
        melt_out[i, j, b, v]        = zero(T_Type)
        pack_cold_content[i, j, b, v] = zero(T_Type)
    else

    # ----------------------------------------------------------------
    # Partition throughfall into rain/snow using band temperature
    # (VIC: SNOW_MIN_RAIN_TEMP = -0.5, SNOW_MAX_SNOW_TEMP = 0.5)
    # ----------------------------------------------------------------
    t_avg = t_band
    MAX_SNOW_TEMP = T_Type(0.5)
    MIN_RAIN_TEMP = T_Type(-0.5)

    rain_frac = zero(T_Type)
    if t_avg > MAX_SNOW_TEMP
        rain_frac = one(T_Type)
    elseif t_avg >= MIN_RAIN_TEMP
        rain_frac = (t_avg - MIN_RAIN_TEMP) / (MAX_SNOW_TEMP - MIN_RAIN_TEMP)
    end
    p_snow = tf_val * (one(T_Type) - rain_frac)
    p_rain = tf_val * rain_frac

    # ----------------------------------------------------------------
    # Load state
    # ----------------------------------------------------------------
    current_swe  = swe[i, j, b, v]
    old_swe      = current_swe
    old_depth_m  = snow_depth[i, j, b, v] / T_Type(1000.0)
    old_coverage = snow_coverage[i, j, b, v]
    old_coverage = isnan(old_coverage) ? zero(T_Type) : clamp(old_coverage, zero(T_Type), one(T_Type))

    current_cc   = cold_content[i, j, b, v]
    current_pcc  = pack_cold_content[i, j, b, v]   # pack layer cold content (J/m²)
    current_pcc  = isnan(current_pcc) ? zero(T_Type) : current_pcc
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
    vp_air = vp_2d[i, j]   # vapor pressure of air (Pa)

    # Per-cell max_snow_distrib_slope = annual_prec_mm / 500 (VIC formula)
    ann_prec = annual_prec_2d[i, j]
    max_distrib_slope = (isnan(ann_prec) || ann_prec <= zero(T_Type)) ?
        T_Type(0.4) : ann_prec / T_Type(500.0)

    # ----------------------------------------------------------------
    # 1. Add snowfall to SWE and update cold content
    # ----------------------------------------------------------------
    old_swq_pre = current_swe  # for coverage state tracking

    if p_snow > zero(T_Type)
        current_swe += p_snow
        if t_avg < zero(T_Type)
            sf_cc = SNW_VCPICE_WQ * (p_snow / T_Type(1000.0)) * t_avg   # J/m²
            current_cc += sf_cc
            if current_cc > zero(T_Type); current_cc = zero(T_Type); end
        end
    end

    # Rain on snow: warm advection
    if p_rain > zero(T_Type) && current_swe > zero(T_Type) && t_avg > zero(T_Type)
        rain_heat  = T_Type(4186.0) * (p_rain / T_Type(1000.0)) * t_avg
        current_cc += rain_heat
        if current_cc > zero(T_Type); current_cc = zero(T_Type); end
    end

    # ----------------------------------------------------------------
    # 2. Albedo: VIC snow_albedo() — accumulation vs melt-season decay
    # ----------------------------------------------------------------
    local alb::T_Type
    if p_snow > SNW_TRACESNOW_MM && current_cc < zero(T_Type)
        lsnow = Int32(0)
        alb   = SNW_NEW_SNOW_ALB
    elseif current_swe > zero(T_Type)
        lsnow = lsnow + Int32(1)
        ls_f  = T_Type(lsnow)
        if current_cc < zero(T_Type) && melt_flag == Int32(0)
            alb = SNW_NEW_SNOW_ALB * (SNW_ALB_ACCUM_A ^ (ls_f ^ SNW_ALB_ACCUM_B))
        else
            alb = SNW_NEW_SNOW_ALB * (SNW_ALB_THAW_A  ^ (ls_f ^ SNW_ALB_THAW_B))
        end
        alb = max(alb, T_Type(0.45))
    else
        alb   = T_Type(NaN)
        lsnow = Int32(0)
    end

    # ----------------------------------------------------------------
    # 3. Update MELTING flag
    # ----------------------------------------------------------------
    if current_swe > zero(T_Type)
        in_melt_season = (lat_positive == Int32(1)) ?
            (day_of_year > Int32(60) && day_of_year < Int32(273)) :
            (day_of_year < Int32(60) || day_of_year > Int32(273))
        if current_cc >= zero(T_Type) && in_melt_season
            melt_flag = Int32(1)
        elseif melt_flag == Int32(1) && p_snow > SNW_TRACESNOW_MM
            melt_flag = Int32(0)
        end
    else
        melt_flag = Int32(0)
    end

    # ----------------------------------------------------------------
    # 4. Snow surface temperature (Newton-Raphson, capped at 0°C)
    #    and melt calculation (VIC: snow_melt.c)
    # ----------------------------------------------------------------
    melt = zero(T_Type)
    # Use previous day's snow surface temperature as NR initial guess (temporal memory).
    # This eliminates the cold-start spike where days 1-9 initialize from min(Tair,0)
    # instead of the thermally equilibrated previous value. Fallback to min(Ta,0)
    # when snow_surf_temp is NaN (no snow on previous day) or unusably extreme.
    prev_ts = snow_surf_temp[i, j, b, v]
    local t_s::T_Type = if !isnan(prev_ts) && prev_ts >= t_avg - T_Type(5.0) && prev_ts <= zero(T_Type)
        prev_ts
    else
        min(t_avg, zero(T_Type))
    end

    if current_swe > zero(T_Type)
        eff_alb = isnan(alb) ? SNW_NEW_SNOW_ALB : alb

        # Aerodynamic resistance (s/m) for snow surface.
        # VIC uses z0_snow = 0.001 m (1mm). With neutral stability and wind ~3m/s at z=2m:
        # ra = ln(2/0.001)^2 / (0.41^2 * wind) ≈ 115 s/m
        # Using ra=100 s/m provides weaker sensible coupling than ra=50,
        # letting the radiation balance dominate and keeping Tsurf closer to VIC values.
        ra = T_Type(100.0)

        # Solve for snow surface temperature with latent heat of sublimation/condensation.
        # LE_sub = rho_air * L_sub / ra * (q_sat(Ts=0) - q_air)
        # q_sat at 0°C: es0 = 611 Pa; q_sat0 = 0.622*es0/(ps-0.378*es0)
        # q_air = 0.622*vp_air/(ps-0.378*vp_air)
        # Positive LE_sub (condensation) adds energy; negative (sublimation) removes energy.
        # Including LE_sub at Ts=0 in the melt energy accounts for dehumidification losses
        # that delay melt onset under dry winter conditions (matches VIC behavior).
        ts_solved, melt_energy_at_zero = snow_surface_temp_nr(
            t_s, t_avg, sw_in, lw_in, eff_alb, ps, ra, vp_air, T_Type
        )
        t_s = ts_solved

        # If Tsurf was capped at 0 (melt_energy_at_zero > 0), melt occurs.
        # VIC two-layer model: surface melt water first warms the pack layer
        # (cold content of SWE > 125mm portion) before liquid water exits.
        # This is the key mechanism that delays/smooths spring melt in VIC.
        if melt_energy_at_zero > zero(T_Type)
            dt_sec = T_Type(86400.0)
            melt_J = melt_energy_at_zero * dt_sec   # J/m²

            # 1. Satisfy surface cold content (existing surface layer)
            if current_cc < zero(T_Type)
                energy_needed = -current_cc
                if melt_J >= energy_needed
                    melt_J    -= energy_needed
                    current_cc = zero(T_Type)
                else
                    current_cc += melt_J
                    melt_J     = zero(T_Type)
                end
            end

            # 2. Satisfy pack cold content (VIC's PackCC — the layer beneath surface)
            # Pack CC is negative when pack is cold; melt water refreezes to warm it.
            if current_pcc < zero(T_Type) && melt_J > zero(T_Type)
                energy_needed = -current_pcc
                if melt_J >= energy_needed
                    melt_J     -= energy_needed
                    current_pcc = zero(T_Type)
                else
                    current_pcc += melt_J
                    melt_J      = zero(T_Type)
                end
            end

            # 3. Net melt that exits the snowpack
            melt = melt_J / (SNW_LATICE * SNW_RHOFW) * T_Type(1000.0)
            melt = min(melt, current_swe)
            current_swe -= melt
        end

        # Update cold content after this timestep
        swe_surf_m = min(current_swe / T_Type(1000.0), SNW_MAX_SURFACE_SWE_MM / T_Type(1000.0))
        swe_pack_m = max(current_swe / T_Type(1000.0) - SNW_MAX_SURFACE_SWE_MM / T_Type(1000.0), zero(T_Type))

        if current_swe > zero(T_Type)
            if melt > zero(T_Type)
                # Surface layer melted → surface is now at 0°C → surface CC = 0.
                # However, the pack beneath warms more slowly: VIC tracks pack_temp separately.
                # Approximate: after melt, remaining pack retains half the prior surface cold content.
                # This prevents the snowpack from having CC=0 the next day and immediately
                # triggering further melt — matching VIC's conservative spring melt behavior.
                prior_cc = cold_content[i, j, b, v]  # pre-update value
                retained_frac = T_Type(0.01)   # 1% CC retention → delays next-day melt onset slightly
                current_cc = min(prior_cc * retained_frac, zero(T_Type))
                # Pack CC: update from estimated pack temp (warming toward 0)
                if current_pcc < zero(T_Type) && swe_pack_m > zero(T_Type)
                    current_pcc = SNW_VCPICE_WQ * swe_pack_m * (t_s * T_Type(0.5))
                    if current_pcc > zero(T_Type); current_pcc = zero(T_Type); end
                end
            elseif t_s < zero(T_Type)
                # No melt — recompute surface CC from current Tsurf
                current_cc = SNW_VCPICE_WQ * swe_surf_m * t_s
                if current_cc > zero(T_Type); current_cc = zero(T_Type); end
                # Pack CC: pack slightly colder than surface (thermal lag)
                pack_temp_est = t_s * T_Type(0.7)
                current_pcc = SNW_VCPICE_WQ * swe_pack_m * pack_temp_est
                if current_pcc > zero(T_Type); current_pcc = zero(T_Type); end
            end
        else
            current_cc  = zero(T_Type)
            current_pcc = zero(T_Type)
            t_s         = T_Type(NaN)
        end
    else
        t_s         = T_Type(NaN)
        current_cc  = zero(T_Type)
        current_pcc = zero(T_Type)
    end

    # ----------------------------------------------------------------
    # 5. Snow depth from density
    # ----------------------------------------------------------------
    current_depth_m = (current_swe / T_Type(1000.0)) * (SNW_RHOFW / SNW_DENSITY)

    # ----------------------------------------------------------------
    # 6. Coverage: VIC calc_snow_coverage() algorithm
    #    Using spatially-varying max_distrib_slope = annual_prec/500
    # ----------------------------------------------------------------
    local new_coverage::T_Type

    if p_snow > zero(T_Type)
        new_coverage = one(T_Type)

        if st_snow == Int32(0)
            if old_coverage < one(T_Type)
                st_snow = Int32(1)
                st_swq  = current_swe - old_swq_pre
                st_cov  = old_coverage
            end
        else
            if st_swq == zero(T_Type)
                st_cov = old_coverage < one(T_Type) ? old_coverage : one(T_Type)
            end
            st_swq += current_swe - old_swq_pre

            if current_depth_m >= max_distrib_slope / T_Type(2.0)
                st_snow = Int32(0); st_swq = zero(T_Type)
                dslope  = zero(T_Type); st_cov = one(T_Type)
            end
        end

    elseif melt > zero(T_Type)
        if st_swq > zero(T_Type) && current_swe < old_swq_pre
            st_swq += current_swe - old_swq_pre
            if st_swq <= zero(T_Type)
                st_swq      = zero(T_Type)
                old_coverage = st_cov
                st_cov      = one(T_Type)
            end
        end

        if st_swq == zero(T_Type)
            if dslope == zero(T_Type)
                cap_depth = min(max_distrib_slope, T_Type(2.0) * old_depth_m)
                dslope    = -cap_depth
                mx_depth  = cap_depth
                st_snow   = Int32(1)
            end

            old_mx_depth = mx_depth
            mx_depth     = T_Type(2.0) * current_depth_m

            if mx_depth < old_mx_depth || old_mx_depth == zero(T_Type)
                raw = (dslope != zero(T_Type)) ? (-mx_depth / dslope) : one(T_Type)
                new_coverage = clamp(raw, zero(T_Type), one(T_Type))
            else
                new_coverage = old_coverage
            end
        else
            new_coverage = old_coverage
        end
    else
        new_coverage = old_coverage
    end

    if current_swe <= zero(T_Type)
        new_coverage = zero(T_Type)
    end

    # ----------------------------------------------------------------
    # 7. Write outputs
    # ----------------------------------------------------------------
    if current_swe > zero(T_Type)
        swe[i, j, b, v]                  = current_swe
        snow_depth[i, j, b, v]           = current_depth_m * T_Type(1000.0)
        snow_albedo[i, j, b, v]          = isnan(alb) ? T_Type(NaN) : alb
        snow_surf_temp[i, j, b, v]       = t_s
        snow_coverage[i, j, b, v]        = new_coverage
        melt_out[i, j, b, v]             = melt
        last_snow[i, j, b, v]            = lsnow
        cold_content[i, j, b, v]         = current_cc
        pack_cold_content[i, j, b, v]    = current_pcc
        melting_flag[i, j, b, v]         = melt_flag
        store_snow[i, j, b, v]           = st_snow
        snow_distrib_slope[i, j, b, v]   = dslope
        store_swq[i, j, b, v]            = st_swq
        store_coverage[i, j, b, v]       = st_cov
        max_snow_depth[i, j, b, v]       = mx_depth
    else
        swe[i, j, b, v]                  = zero(T_Type)
        snow_depth[i, j, b, v]           = zero(T_Type)
        snow_albedo[i, j, b, v]          = T_Type(NaN)
        snow_surf_temp[i, j, b, v]       = T_Type(NaN)
        snow_coverage[i, j, b, v]        = zero(T_Type)
        melt_out[i, j, b, v]             = melt
        last_snow[i, j, b, v]            = Int32(0)
        cold_content[i, j, b, v]         = zero(T_Type)
        pack_cold_content[i, j, b, v]    = zero(T_Type)
        melting_flag[i, j, b, v]         = Int32(0)
        store_snow[i, j, b, v]           = Int32(0)
        snow_distrib_slope[i, j, b, v]   = zero(T_Type)
        store_swq[i, j, b, v]            = zero(T_Type)
        store_coverage[i, j, b, v]       = zero(T_Type)
        max_snow_depth[i, j, b, v]       = zero(T_Type)
    end

    end  # end else (active tile)

    nothing
end

# ---------------------------------------------------------------------------
# Wrapper: 4D snow dynamics (per vegetation tile)
# ---------------------------------------------------------------------------
function calculate_snow_dynamics!(
    swe_gpu, snow_depth_gpu, snow_albedo_gpu, snow_surf_temp_gpu,
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
        swe_gpu, snow_depth_gpu, snow_albedo_gpu, snow_surf_temp_gpu,
        snow_coverage_gpu, snow_melt_gpu,
        last_snow_gpu, cold_content_gpu, pack_cc_gpu, melting_flag_gpu,
        store_snow_gpu, snow_distrib_slope_gpu,
        store_swq_gpu, store_coverage_gpu, max_snow_depth_gpu,
        throughfall_4d, tair_3d, swdown_gpu, lwdown_gpu, psurf_gpu, vp_gpu,
        AreaFract_gpu, cv_gpu, annual_prec_gpu,
        Int32(day_of_year), lat_pos;
        ndrange=size(swe_gpu)   # (nx, ny, nbands, nveg)
    )
    KernelAbstractions.synchronize(device_backend)
end
