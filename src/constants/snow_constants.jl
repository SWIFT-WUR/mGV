# Albedo decay parameters
const SNW_NEW_SNOW_ALB       = ft(0.85)     # Albedo of newly fallen snow [-]
const SNW_ALB_ACCUM_A        = ft(0.94)     # Accumulation-season decay factor A [-]
const SNW_ALB_ACCUM_B        = ft(0.58)     # Accumulation-season decay factor B [-]
const SNW_ALB_THAW_A         = ft(0.82)     # Melt-season decay factor A [-]
const SNW_ALB_THAW_B         = ft(0.46)     # Melt-season decay factor B [-]

# Capacities and physical limits
const SNW_TRACESNOW_MM       = ft(0.001)    # Minimum SWE for active snowpack pruning [mm]
const SNW_NEW_SNOW_THRESH_MM = ft(0.001)    # Match TRACESNOW [mm]
const SNW_LIQUID_WATER_CAP   = ft(0.035)    # Liquid water holding capacity fraction [-]
const SNW_MAX_SURFACE_SWE_MM = ft(125.0)    # Maximum snow water equivalent (SWE) in the surface layer [mm]

# Thermodynamic properties
const SNW_VCPICE_WQ          = ft(2117.27)  # Specific heat capacity of ice per water-equiv [J/(kg·K)]
const SNW_LATICE             = ft(334000.0) # Latent heat of fusion for water [J/kg]
const SNW_RHOFW              = ft(1000.0)   # Density of fresh water [kg/m³]
const SNW_DENSITY            = ft(250.0)    # Typical density of the snowpack [kg/m³]
