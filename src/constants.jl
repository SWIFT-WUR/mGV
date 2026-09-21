module Constants

export PhysConsts, SimConsts, SnowConsts

using Parameters

@consts begin
    # Air and water density Constants       
    RHO_A::Float32 = 1.225       # Density of air (TODO: make temperature dependent?)
    RHO_W::Float32 = 1000.0      # Density of liquid water (TODO: make temperature dependent?)

    # Saturation Vapor Pressure Constants
    SVP_A::Float32 = 0.61078     # Empirical coefficient; gives SVP at 0 °C (kPa)
    SVP_B::Float32 = 17.269      # Dimensionless empirical constant
    SVP_C::Float32 = 237.3       # Dimensionally same as temperature (used with T in °C)
    PA_PER_KPA::Float32 = 1000.0 # Converted to float to avoid mixed-type math

    # Universal Physical Constants
    K_B::Float32 = 1.38065e-23       # Boltzmann's constant (J/K)
    N_A::Float32 = 6.02214e26       # Avogadro's number (molecules/kmole)
    R_GAS = K_B * N_A               # Universal gas constant (J/K/kmole)
    MW_AIR::Float32 = 28.966        # Molecular weight of dry air (kg/kmole)
    R_AIR = R_GAS / MW_AIR  # Dry air gas constant (J/K/kg)

    # Temperature and Environmental Constants
    T_FREEZE::Float32 = 273.15   # Freezing temperature (K)
    LAPSE_RATE::Float32 = 0.0065 # Lapse rate (K/m)

    # Energy and Radiation Constants
    LAT_VAP::Float32 = 2.501e6     # Latent heat of vaporization (J/kg)
    G::Float32 = 9.81              # Gravitational acceleration (m/s²)
    SIGMA::Float32 = 5.67e-8       # Stefan-Boltzmann constant (W/m²K⁴)

    # Atmospheric Constants
    P_STD::Float32 = 101325.0    # Standard pressure (Pa)
    C_P_AIR::Float32 = 1013.0    # Specific heat of moist air (J/kg·K)

    # Unit Conversion Constants
    DAY_SEC::Float32 = 86400.0   # Seconds in a day 
    MM_IN_M::Float32 = 1000.0    # Conversion factor from mm to m 

    # Evapotranspiration Constants
    G_COEFF::Float32 = 1628.6    # Psychrometric / Evaporation coefficient
    AIR_C::Float32 = 0.003486    # Air density coefficient (1 / R_air)
end

PhysConsts = (;
    RHO_A, RHO_W, SVP_A, SVP_B, SVP_C, PA_PER_KPA, 
    K_B, N_A, R_GAS, MW_AIR, R_AIR, T_FREEZE, LAPSE_RATE, 
    LAT_VAP, G, SIGMA, P_STD, C_P_AIR, DAY_SEC, MM_IN_M, G_COEFF, AIR_C
)

@consts begin
    K_L::Float32 = 0.1
    VON_KARMAN::Float32 = 0.4
    Z2::Float32 = 10.0
    RI_CR::Float32 = 0.2
    EMISSIVITY::Float32 = 0.97

    # Thermal conductivity constants
    KI::Float32 = 2.2         # Thermal conductivity of ice (W/mK)
    KW::Float32 = 0.57        # Thermal conductivity of water (W/mK)
    KDRY_ORG::Float32 = 0.05  # Dry thermal conductivity of organic fraction (W/mK)
    KS_ORG::Float32 = 0.25    # Thermal conductivity of organic solid (W/mK)

    # Ground composition constants
    ORGANIC_FRAC::Float32 = 0.0
    BULK_DENS_ORG::Float32 = 0.0
    SOIL_DENS_ORG::Float32 = 0.0
end

SimConsts = (; 
    K_L, VON_KARMAN, Z2, RI_CR, EMISSIVITY, KI, KW, KDRY_ORG, KS_ORG,
    ORGANIC_FRAC, BULK_DENS_ORG, SOIL_DENS_ORG
)

@consts begin
    # Albedo decay parameters
    NEW_SNOW_ALB::Float32       = 0.85     # Albedo of newly fallen snow [-]
    ALB_ACCUM_A::Float32        = 0.94     # Accumulation-season decay factor A [-]
    ALB_ACCUM_B::Float32        = 0.58     # Accumulation-season decay factor B [-]
    ALB_THAW_A::Float32         = 0.82     # Melt-season decay factor A [-]
    ALB_THAW_B::Float32         = 0.46     # Melt-season decay factor B [-]

    # Capacities and physical limits
    TRACESNOW_MM::Float32       = 0.001    # Minimum SWE for active snowpack pruning [mm]
    NEW_SNOW_THRESH_MM::Float32 = 0.001    # Match TRACESNOW [mm]
    LIQUID_WATER_CAP::Float32   = 0.035    # Liquid water holding capacity fraction [-]
    MAX_SURFACE_SWE_MM::Float32 = 125.0    # Maximum snow water equivalent (SWE) in the surface layer [mm]

    # Thermodynamic properties
    VCPICE_WQ::Float32          = 2117.27  # Specific heat capacity of ice per water-equiv [J/(kg·K)]
    LATICE::Float32             = 334000.0 # Latent heat of fusion for water [J/kg]
    RHOFW::Float32              = 1000.0   # Density of fresh water [kg/m³]
    DENSITY::Float32            = 250.0    # Typical density of the snowpack [kg/m³]
end

SnowConsts = (;
    NEW_SNOW_ALB, ALB_ACCUM_A, ALB_ACCUM_B, ALB_THAW_A, ALB_THAW_B,
    TRACESNOW_MM, NEW_SNOW_THRESH_MM, LIQUID_WATER_CAP, MAX_SURFACE_SWE_MM,
    VCPICE_WQ, LATICE, RHOFW, DENSITY
)

end