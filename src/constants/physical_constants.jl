# Air and water density Constants       
const rho_a = ft(1.225)       # Density of air (TODO: make temperature dependent?)
const rho_w = ft(1000.0)      # Density of liquid water (TODO: make temperature dependent?)

# Saturation Vapor Pressure Constants
const svp_a = ft(0.61078)     # Empirical coefficient; gives SVP at 0 °C (kPa)
const svp_b = ft(17.269)      # Dimensionless empirical constant
const svp_c = ft(237.3)       # Dimensionally same as temperature (used with T in °C)
const pa_per_kpa = ft(1000.0) # Converted to float to avoid mixed-type math

# Universal Physical Constants
const k_b = ft(1.38065e-23)     # Boltzmann's constant (J/K)
const n_a = ft(6.02214e26)      # Avogadro's number (molecules/kmole)
const r_gas = n_a * k_b         # Universal gas constant (J/K/kmole)
const mw_air = ft(28.966)       # Molecular weight of dry air (kg/kmole)
const r_air = r_gas / mw_air    # Dry air gas constant (J/K/kg)

# Temperature and Environmental Constants
const t_freeze = ft(273.15)   # Freezing temperature (K)
const lapse_rate = ft(0.0065) # Lapse rate (K/m)

# Energy and Radiation Constants
const lat_vap = ft(2.501e6) # Latent heat of vaporization (J/kg)
const g = ft(9.81)          # Gravitational acceleration (m/s²)
const sigma = ft(5.67e-8)       # Stefan-Boltzmann constant (W/m²K⁴)

# Atmospheric Constants
const p_std = ft(101325.0)    # Standard pressure (Pa)
const c_p_air = ft(1013.0)    # Specific heat of moist air (J/kg·K)

# Unit Conversion Constants
const day_sec = ft(86400.0)   # Seconds in a day 
const mm_in_m = ft(1000.0)    # Conversion factor from mm to m

# ==============================================================================
# VIC SNOW PHYSICAL CONSTANTS
# ==============================================================================
const CONST_PI        = ft(3.14159265358979323846)
const CONST_G         = ft(9.80616)
const CONST_EPS       = ft(18.016 / 28.97)
const CONST_TKFRZ     = ft(273.15)
const CONST_RHOFW     = ft(1000.0)
const CONST_RHOICE    = ft(917.0)
const CONST_CPFW      = ft(4188.0)
const CONST_CPICE     = ft(2117.27)
const CONST_VCPICE_WQ = (CONST_CPICE * CONST_RHOFW)
const CONST_LATICE    = ft(333700.0)
const CONST_LATVAP    = ft(2.501e6)
const CONST_LATSUB    = (CONST_LATICE + CONST_LATVAP)