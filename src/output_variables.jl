"""
All available output variables. Each entry gives the field in `Results`, the variable name in
the output file (used in `[output].variables`), a description, the unit, and the config flag
of the module the variable needs (`nothing` when always available). Also used to generate
the "Model output" documentation page.
"""
const OUTPUT_VARIABLES = (
    (
        field = :surface_temperature, name = "tsurf", unit = "°C", requires = nothing,
        description = "Surface temperature, blended with the snow surface temperature according to snow coverage",
    ),
    (
        field = :air_temperature, name = "tair", unit = "°C", requires = nothing,
        description = "Air temperature (from the forcing)",
    ),
    (
        field = :precipitation, name = "precipitation", unit = "mm/day", requires = nothing,
        description = "Precipitation (from the forcing)",
    ),
    (
        field = :total_evapotranspiration, name = "total_et", unit = "mm/day", requires = nothing,
        description = "Total evapotranspiration",
    ),
    (
        field = :surface_runoff, name = "surface_runoff", unit = "mm/day", requires = nothing,
        description = "Surface runoff",
    ),
    (
        field = :total_runoff, name = "total_runoff", unit = "mm/day", requires = nothing,
        description = "Total runoff (surface and subsurface)",
    ),
    (
        field = :discharge, name = "discharge", unit = "m³/s", requires = :enable_routing,
        description = "River discharge",
    ),
    (
        field = :travel_time, name = "travel_time", unit = "s", requires = :enable_routing,
        description = "Time for water to flow through the grid cell",
    ),
    (
        field = :potential_evaporation, name = "potential_evaporation", unit = "mm/day", requires = nothing,
        description = "Potential evaporation, weighted sum over vegetation types and snow bands",
    ),
    (
        field = :net_radiation, name = "net_radiation", unit = "W/m²", requires = nothing,
        description = "Net radiation, weighted sum over vegetation types and snow bands",
    ),
    (
        field = :transpiration, name = "transpiration", unit = "mm/day", requires = nothing,
        description = "Transpiration, weighted sum over vegetation types and snow bands",
    ),
    (
        field = :canopy_evaporation, name = "canopy_evaporation", unit = "mm/day", requires = nothing,
        description = "Evaporation of water held on the vegetation canopy, weighted sum over vegetation types and snow bands",
    ),
    (
        field = :water_storage, name = "water_storage", unit = "mm", requires = nothing,
        description = "Water held on the vegetation canopy, weighted sum over vegetation types and snow bands",
    ),
    (
        field = :snow_water_equivalent, name = "swe", unit = "mm", requires = :enable_snow,
        description = "Snow water equivalent",
    ),
    (
        field = :snow_albedo, name = "snow_albedo", unit = "fraction (0–1)", requires = :enable_snow,
        description = "Snow albedo, only where there is snow",
    ),
    (
        field = :snow_surface_temperature, name = "snow_surf_temp", unit = "°C", requires = :enable_snow,
        description = "Snow surface temperature, only where there is snow",
    ),
    (
        field = :snow_coverage, name = "snow_coverage", unit = "fraction (0–1)", requires = :enable_snow,
        description = "Fraction of the grid cell covered by snow",
    ),
    (
        field = :snow_melt, name = "snow_melt", unit = "mm/day", requires = :enable_snow,
        description = "Snowmelt",
    ),
    (
        field = :soil_evaporation, name = "soil_evaporation", unit = "mm/day", requires = nothing,
        description = "Evaporation from bare soil",
    ),
    (
        field = :soil_moisture, name = "soil_moisture", unit = "mm", requires = nothing,
        description = "Soil moisture, per soil layer",
    ),
)
