const K_L = ft(0.1)
const von_karman = ft(0.4)
const z2 = ft(10.0)
const Ri_cr = ft(0.2)
const emissivity = ft(0.97)

# Thermal conductivity constants
const Ki = ft(2.2)   # Thermal conductivity of ice (W/mK)
const Kw = ft(0.57)  # Thermal conductivity of water (W/mK)
const Kdry_org = ft(0.05)  # Dry thermal conductivity of organic fraction (W/mK)
const Ks_org = ft(0.25)  # Thermal conductivity of organic solid (W/mK)

# Ground composition constants
const organic_frac = ft(0.0)
const bulk_dens_org = ft(0.0)
const soil_dens_org = ft(0.0)

# ==============================================================================
# VIC SNOW SIMULATION CONSTANTS
# ==============================================================================
const SNOW_MIN_SWQ_EB_THRES      = ft(0.01)     # (m) minimum SWQ for energy balance calculation
const SNOW_MAX_SURFACE_SWE       = ft(0.0125)   # (m) maximum surface layer SWE
const SNOW_LIQUID_WATER_CAPACITY = ft(0.035)    # Maximum liquid water capacity of snowpack
const MIN_RAIN_TEMP              = ft(-0.5)     # Temp below which precip is all snow
const MAX_SNOW_TEMP              = ft(0.5)      # Temp above which precip is all rain