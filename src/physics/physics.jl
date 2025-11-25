using .PhysicalConstants

# Vapor Pressure Deficit Calculation
function calculate_vpd(tair, vp)
    # 1. Tetens Equation (Standard over water)
    svp = svp_a * exp((svp_b * tair) / (svp_c + tair))

    # 2. Sub-zero correction (Murray 1967 / standard VIC logic)
    # Lower saturation vapor pressure over ice compared to water
    if tair < 0.0f0
        svp = svp * (1.0f0 + 0.00972f0 * tair + 0.000042f0 * tair^2)
    end

    # 3. Calculate VPD
    # Ensure non-negative
    return max(svp - vp, 0.0f0) * pa_per_kpa # [Pa]
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
    return (r_air / g) * ((tair + t_freeze) + 0.5f0 * elev * lapse_rate) # [m]
end

# Calculates Latent Heat of Vaporization (J/kg)
@inline function calculate_latent_heat(tair_kelvin)
    tc = tair_kelvin - t_freeze
    return 2.501f6 - 2361.0f0 * tc # [K/kg]
end

# Watson correlation
#function calculate_latent_heat(T)
#    # Parameters for water
#    Hvap_Tb = 2.26e6  # Latent heat at boiling point (J/kg)
#    Tb = 373.15       # Boiling point (K)
#    Tc = 647.096      # Critical temperature (K)
#    n = 0.38          # Watson exponent
#
#    # Compute element-wise ratio
#    ratio = (Tc .- T) ./ (Tc - Tb)
#    ratio = clamp.(ratio, 1e-6, 1.0)  # Prevent unphysical values
#    Hvap = Hvap_Tb .* (ratio .^ n)
#
#    # Ensure latent heat is positive, element-wise
#    return max.(Hvap, 1e3)  # Minimum 1000 J/kg for numerical stability
#end