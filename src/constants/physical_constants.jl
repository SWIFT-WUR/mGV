# Air and water density Constants       
const RHO_A = ft(1.225)       # Density of air (TODO: make temperature dependent?)
const RHO_W = ft(1000.0)      # Density of liquid water (TODO: make temperature dependent?)

# Saturation Vapor Pressure Constants
const SVP_A = ft(0.61078)     # Empirical coefficient; gives SVP at 0 °C (kPa)
const SVP_B = ft(17.269)      # Dimensionless empirical constant
const SVP_C = ft(237.3)       # Dimensionally same as temperature (used with T in °C)
const PA_PER_KPA = ft(1000.0) # Converted to float to avoid mixed-type math

# Universal Physical Constants
const K_B = ft(1.38065e-23)     # Boltzmann's constant (J/K)
const N_A = ft(6.02214e26)      # Avogadro's number (molecules/kmole)
const R_GAS = N_A * K_B         # Universal gas constant (J/K/kmole)
const MW_AIR = ft(28.966)       # Molecular weight of dry air (kg/kmole)
const R_AIR = R_GAS / MW_AIR    # Dry air gas constant (J/K/kg)

# Temperature and Environmental Constants
const T_FREEZE = ft(273.15)   # Freezing temperature (K)
const LAPSE_RATE = ft(0.0065) # Lapse rate (K/m)

# Energy and Radiation Constants
const LAT_VAP = ft(2.501e6)     # Latent heat of vaporization (J/kg)
const G = ft(9.81)              # Gravitational acceleration (m/s²)
const SIGMA = ft(5.67e-8)       # Stefan-Boltzmann constant (W/m²K⁴)

# Atmospheric Constants
const P_STD = ft(101325.0)    # Standard pressure (Pa)
const C_P_AIR = ft(1013.0)    # Specific heat of moist air (J/kg·K)

# Unit Conversion Constants
const DAY_SEC = ft(86400.0)   # Seconds in a day 
const MM_IN_M = ft(1000.0)    # Conversion factor from mm to m 

# Evapotranspiration Constants
const G_COEFF = ft(1628.6)    # Psychrometric / Evaporation coefficient
const AIR_C = ft(0.003486)    # Air density coefficient (1 / R_air)