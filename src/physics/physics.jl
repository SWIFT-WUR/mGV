# Vapor Pressure Deficit calculation
function calculate_vpd(tair, vp)
    # 1. Tetens Equation (standard over water)
    svp = svp_a * exp((svp_b * tair) / (svp_c + tair))

    # 2. Sub-zero correction (Murray 1967 / standard VIC logic)
    # Lower saturation vapor pressure over ice compared to water
    if tair < ft(0.0)
        svp = svp * (ft(1.0) + ft(0.00972) * tair + ft(0.000042) * tair^2)
    end

    # 3. Calculate VPD
    return max(svp - vp, ft(0.0)) * pa_per_kpa # [Pa]
end

# Calculates the slope of the SVP curve
@inline function calculate_svp_slope(tair)
    # Re-calculate SVP part locally (scalar)
    svp_part = svp_a * exp((svp_b * tair) / (svp_c + tair))
    
    # Calculate Slope
    slope_kpa = (svp_b * svp_c * svp_part) / ((svp_c + tair)^2)
    
    return slope_kpa * pa_per_kpa # [Pa/°C]
end

# Calculates atmospheric scale height [m] with lapse rate correction
@inline function calculate_scale_height(tair, elev)
    # (R / g) * T_avg
    # T_avg approx: T_air + t_freeze + (0.5 * elev * lapse_rate)
    return (r_air / g) * ((tair + t_freeze) + ft(0.5) * elev * lapse_rate) # [m]
end

# Calculates latent heat of vaporization (J/kg)
@inline function calculate_latent_heat(tair_kelvin)
    tc = tair_kelvin - t_freeze
    return ft(2.501e6) - ft(2361.0) * tc # [K/kg]
end
