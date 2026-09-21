struct SurfaceEnergyVariables{M <: AbstractMatrix, T <: AbstractArray}
    # State
    surface_temperature::M

    # Fluxes
    net_radiation::T
    potential_evaporation::T
    soil_potential_evaporation::T
    total_evapotranspiration::M

    # Derived/intermediate
    aerodynamic_resistance::T

    # Errors
    energy_error::M
    water_error::M
end

@adapt_structure SurfaceEnergyVariables

function SurfaceEnergyVariables(grid_dims, tile_dims)
    return SurfaceEnergyVariables(
        zeros(Float32, grid_dims),
        zeros(Float32, tile_dims),
        zeros(Float32, tile_dims),
        zeros(Float32, tile_dims),
        zeros(Float32, grid_dims),
        zeros(Float32, tile_dims),
        zeros(Float32, grid_dims),
        zeros(Float32, grid_dims)
    )
end

struct CanopyVariables{T <: AbstractArray}
    # State
    water_storage::T

    # Fluxes
    throughfall::T
    canopy_evaporation::T
    transpiration::T
    transpiration_layers::T

    # Derived/intermediate
    maximum_water_storage::T
    wet_fraction::T
end

@adapt_structure CanopyVariables

function CanopyVariables(dims)
    return CanopyVariables(
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims),
        zeros(Float32, dims)
    )
end


struct SoilVariables{M <: AbstractMatrix, T <: AbstractArray}
    # State
    moisture::T
    temperature::T
    ice_fraction::T

    # Fluxes
    evaporation::M
    infiltration::M
    surface_runoff::M
    subsurface_runoff::M
    total_runoff::M
    interlayer_drainage::T

    # Derived/intermediate
    thermal_conductivity::T
    heat_capacity::T
    saturated_fraction::M
end

@adapt_structure SoilVariables

function SoilVariables(grid_dims, soil_dims)
    return SoilVariables(
        zeros(Float32, soil_dims),
        zeros(Float32, soil_dims),
        zeros(Float32, soil_dims),
        zeros(Float32, grid_dims),
        zeros(Float32, grid_dims),
        zeros(Float32, grid_dims),
        zeros(Float32, grid_dims),
        zeros(Float32, grid_dims),
        zeros(Float32, (grid_dims[1], grid_dims[2], soil_dims[3]-1)),
        zeros(Float32, soil_dims),
        zeros(Float32, soil_dims),
        zeros(Float32, grid_dims)
    )
end

using .Constants: SVP_A, SVP_B, SVP_C, G, PA_PER_KPA, R_AIR, T_FREEZE, LAPSE_RATE

"""
    calculate_svp(air_temperature)

Compute the saturation vapor pressure (kPa?) based on the air
temperature (°C).
"""
function calculate_svp(air_temperature)
    # 1. Tetens Equation (standard over water)
    svp = SVP_A * exp((SVP_B * air_temperature) / (SVP_C + air_temperature))

    # 2. Sub-zero correction (Murray 1967 / standard VIC logic)
    # Lower saturation vapor pressure over ice compared to water
    if air_temperature < 0.0f0
        svp = svp * (1.0f0 + 0.00972f0 * air_temperature + 0.000042f0 * air_temperature^2)
    end
    return svp
end

"""
    calculate_vpd(air_temperature, vapor_pressure)

Compute the vapor pressure deficit (Pa?) based on the air temperature (°C)
and actual vapor pressure (kPa?).
"""
function calculate_vpd(air_temperature, vapor_pressure)
    return max(
        calculate_svp(air_temperature) - vapor_pressure,
        0.0f0
    ) * PA_PER_KPA # [Pa]
end

"""
Compute the slope of the saturation vapor pressure curve.

Note: input temperature should be in degC
"""
function calculate_svp_slope(air_temperature)
    # Re-calculate SVP part locally (scalar)
    svp_part = SVP_A * exp((SVP_B * air_temperature) / (SVP_C + air_temperature))
    
    # Calculate Slope
    slope_kpa = (SVP_B * SVP_C * svp_part) / ((SVP_C + air_temperature)^2)
    
    svp_slope = slope_kpa * PA_PER_KPA # [Pa/°C]
    return svp_slope
end

"""
Compute the atmospheric scale height [m] with lapse rate correction
"""
function calculate_scale_height(air_temperature, elevation)
    return (R_AIR / G) * (
        (air_temperature + T_FREEZE) + 0.5f0 * elevation * LAPSE_RATE
    )
end

"""
Compute the latent heat of vaporization (J/kg)

Note: input temperature should be in degC
"""
function calculate_latent_heat(air_temperature)
    return 2.501f6 - 2361.0f0 * air_temperature # [K/kg]
end
