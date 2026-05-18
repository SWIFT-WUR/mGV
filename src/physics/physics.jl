@inline function calculate_svp(tair)
    # 1. Tetens Equation (standard over water)
    svp = SVP_A * exp((SVP_B * tair) / (SVP_C + tair))

    # 2. Sub-zero correction (Murray 1967 / standard VIC logic)
    # Lower saturation vapor pressure over ice compared to water
    if tair < ft(0.0)
        svp = svp * (ft(1.0) + ft(0.00972) * tair + ft(0.000042) * tair^2)
    end
    return svp
end

# Vapor Pressure Deficit calculation
function calculate_vpd(tair, vp)
    svp_val = calculate_svp(tair)
    # 3. Calculate VPD
    return max(svp_val - vp, ft(0.0)) * PA_PER_KPA # [Pa]
end

# Calculates the slope of the SVP curve
@inline function calculate_svp_slope(tair)
    # Re-calculate SVP part locally (scalar)
    svp_part = SVP_A * exp((SVP_B * tair) / (SVP_C + tair))
    
    # Calculate Slope
    slope_kpa = (SVP_B * SVP_C * svp_part) / ((SVP_C + tair)^2)
    
    return slope_kpa * PA_PER_KPA # [Pa/°C]
end

# Calculates atmospheric scale height [m] with lapse rate correction
@inline function calculate_scale_height(tair, elev)
    # (R / G) * T_avg
    # T_avg approx: T_air + T_FREEZE + (0.5 * elev * LAPSE_RATE)
    return (R_AIR / G) * ((tair + T_FREEZE) + ft(0.5) * elev * LAPSE_RATE) # [m]
end

# Calculates latent heat of vaporization (J/kg)
@inline function calculate_latent_heat(tair_kelvin)
    tc = tair_kelvin - T_FREEZE
    return ft(2.501e6) - ft(2361.0) * tc # [K/kg]
end
