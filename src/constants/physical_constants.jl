module PhysicalConstants
    export rho_a, rho_w, svp_a, svp_b, svp_c, pa_per_kpa, 
           k_b, n_a, r_gas, mw_air, r_air, 
           t_freeze, lapse_rate, lat_vap, g, sigma, 
           p_std, c_p_air, day_sec, mm_in_m

    # Air and water density Constants       
    const rho_a = 1.225f0       # Density of air (TODO: make temperature dependent?)
    const rho_w = 1000.0f0      # Density of liquid water (TODO: make temperature dependent?)

    # Saturation Vapor Pressure Constants
    const svp_a = 0.61078f0     # Empirical coefficient; gives SVP at 0 °C (kPa)
    const svp_b = 17.269f0      # Dimensionless empirical constant
    const svp_c = 237.3f0       # Dimensionally same as temperature (used with T in °C)
    const pa_per_kpa = 1000.0f0 # Converted to float to avoid mixed-type math

    # Universal Physical Constants
    const k_b = 1.38065f-23     # Boltzmann's constant (J/K)
    const n_a = 6.02214f26      # Avogadro's number (molecules/kmole)
    
    # These derived constants will automatically be Float32 since inputs are Float32
    const r_gas = n_a * k_b     # Universal gas constant (J/K/kmole)
    const mw_air = 28.966f0     # Molecular weight of dry air (kg/kmole)
    const r_air = r_gas / mw_air # Dry air gas constant (J/K/kg)

    # Temperature and Environmental Constants
    const t_freeze = 273.15f0   # Freezing temperature (K)
    const lapse_rate = 0.0065f0 # Lapse rate (K/m)

    # Energy and Radiation Constants
    const lat_vap = 2.501f6     # Latent heat of vaporization (J/kg)
    const g = 9.81f0            # Gravitational acceleration (m/s²)
    const sigma = 5.67f-8       # Stefan-Boltzmann constant (W/m²K⁴)

    # Atmospheric Constants
    const p_std = 101325.0f0    # Standard pressure (Pa)
    const c_p_air = 1013.0f0    # Specific heat of moist air (J/kg·K)

    # Unit Conversion Constants
    const day_sec = 86400.0f0   # Seconds in a day 
    const mm_in_m = 1000.0f0    # Conversion factor from mm to m 
end