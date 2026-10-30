"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl/blob/main/Wflow/src/standard_name/standard_name_sbm.jl 
MIT License, Copyright (c) 2020 Deltares and contributors
"""

"""
Mapping of grid identifier to a key, to get some context and later on to retrieve 
the active indices of the model domain as in Wflow, active_indices(network, key::AbstractString).
"""
const GRIDS = Dict{String, Int}(
    "2D_grid" => 0,
    "soil_grid" => 1,
    "inter_soil_grid" => 2,
    "vegetation_snow_grid" => 3,
)
const GRIDS_2D = [GRIDS["2D_grid"], GRIDS["soil_grid"], GRIDS["inter_soil_grid"]]

"""
Mapping of (CSDMS) standard names to the metadata associated with the corresponding
variable or parameter. For more details and default values see `ParameterMetadata`.
"""
const standard_name_map = OrderedDict{String, ParameterMetadata}(
    #Surface energy variables
    # > 2D
    "land_surface__radiation~net~upward_energy_flux" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.net_radiation),
        unit = Unit(; J = 1, s = -1, m = -2),
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables_band],
    ),
    "land_surface__evaporation~potential" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.potential_evaporation),
        unit = Unit(; mm = 1, d = -1),
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables_band],
    ),
    "land_surface__evaporation~potential~soil" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.soil_potential_evaporation),
        unit = Unit(; mm = 1, d = -1),
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables_band],
    ),
    "atmosphere_bottom_air__resistance~aerodynamic" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.aerodynamic_resistance),
        unit = Unit(; s = 1, m = -1),
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables_band],
    ),

    # 2D
    "land_surface__temperature" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.surface_temperature),
        unit = Unit(; degC = 1),
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables],
    ),
    "land_surface__evaporation~total" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.total_evapotranspiration),
        unit = Unit(; mm = 1, d = -1),
        description = "Total evapotranspiration",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables],
        output_name = "total_et",
    ),
    #TODO do we want to expose these variables?
    # "energy_error" => ParameterMetadata(;
    #     lens = @optic(_.surface_energy_variables.energy_error),
    #     unit = Unit(),
    #     default = 1.0,
    #     description = "",
    #     gridtype = GRIDS["2D_grid"],
    #     input = false,
    #     output = true,
    #     tags = [:surface_energy_variables],
    # ),
    # "water_error" => ParameterMetadata(;
    #     lens = @optic(_.surface_energy_variables.water_error),
    #     unit = Unit(),
    #     default = 1.0,
    #     description = "",
    #     gridtype = GRIDS["2D_grid"],
    #     input = false,
    #     output = true,
    #     tags = [:surface_energy_variables],
    # ),

    #Soil variables > 2D
    "soil__moisture" => ParameterMetadata(;
        lens = @optic(_.soil_variables.moisture),
        unit = Unit(; mm = 1),
        description = "Soil moisture, as water depth per layer.",
        gridtype = GRIDS["soil_grid"],
        input = true,
        output = true,
        tags = [:soil_variables_layer],
        output_name = "soil_moisture",
    ),
    "soil__moisture~maximum" => ParameterMetadata(;
        lens = @optic(_.soil_parameters.maximum_moisture),
        unit = Unit(; mm = 1),
        description = (
            "Maximum soil moisture, as water depth per layer. " *
                "Note that the critical_moisture, field_capacity and wilting_point are " *
                "derived from this variable at initialization. So when modifying this " *
                "variable, consider also modifying those."
        ),
        gridtype = GRIDS["soil_grid"],
        input = true,
        output = false,
        tags = [:soil_variables_layer],
    ),
    "soil__temperature" => ParameterMetadata(;
        lens = @optic(_.soil_variables.temperature),
        unit = Unit(; degC = 1),
        description = "",
        gridtype = GRIDS["soil_grid"],
        input = false,
        output = true,
        tags = [:soil_variables_layer],
    ),
    "soil__ice_fraction" => ParameterMetadata(;
        lens = @optic(_.soil_variables.ice_fraction),
        unit = Unit(),
        description = "",
        gridtype = GRIDS["soil_grid"],
        input = false,
        output = true,
        tags = [:soil_variables_layer],
    ),
    "soil__drainage~interlayer" => ParameterMetadata(;
        lens = @optic(_.soil_variables.interlayer_drainage),
        unit = Unit(; mm = 1, d = -1),
        description = "",
        gridtype = GRIDS["inter_soil_grid"],
        input = false,
        output = true,
        tags = [:soil_variables_interlayer],
    ),
    "soil__thermal_conductivity" => ParameterMetadata(;
        lens = @optic(_.soil_variables.thermal_conductivity),
        unit = Unit(; J = 1, s = -1, m = -1, K = -1),
        description = "",
        gridtype = GRIDS["soil_grid"],
        input = false,
        output = true,
        tags = [:soil_variables_layer],
    ),

    #2D
    "soil__evaporation" => ParameterMetadata(;
        lens = @optic(_.soil_variables.evaporation),
        unit = Unit(; mm = 1, d = -1),
        description = "Evaporation from bare soil",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],
        output_name = "soil_evaporation",
    ),
    "soil__infiltration" => ParameterMetadata(;
        lens = @optic(_.soil_variables.infiltration),
        unit = Unit(; mm = 1, d = -1),
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],
    ),
    "land_surface__runoff" => ParameterMetadata(;
        lens = @optic(_.soil_variables.surface_runoff),
        unit = Unit(; mm = 1, d = -1),
        description = "Surface runoff",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],
        output_name = "surface_runoff",
    ),
    "land_subsurface__runoff" => ParameterMetadata(;
        lens = @optic(_.soil_variables.subsurface_runoff),
        unit = Unit(; mm = 1, d = -1),
        description = "Baseflow from the bottom soil layer. Acts as groundwater recharge when `external_groundwater_routing = true`",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],
    ),
    "land__runoff~total" => ParameterMetadata(;
        lens = @optic(_.soil_variables.total_runoff),
        unit = Unit(; mm = 1, d = -1),
        description = "Runoff into the river routing. Excludes subsurface runoff when `external_groundwater_routing = true`",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],
        output_name = "total_runoff",
    ),

    #Canopy variables >2d
    "land_vegetation_canopy__evaporation" => ParameterMetadata(;
        lens = @optic(_.canopy_variables.canopy_evaporation),
        unit = Unit(; mm = 1, d = -1),
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:canopy_variables_band],
    ),
    "land_vegetation_canopy__water_storage" => ParameterMetadata(;
        lens = @optic(_.canopy_variables.water_storage),
        unit = Unit(),  # TODO: add unit
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = true,
        output = true,
        tags = [:canopy_variables_band],
    )
)

const standard_routing_map = OrderedDict{String, ParameterMetadata}(
    #Routing parameters
    "channel__length" => ParameterMetadata(;
        lens = @optic(_.routing.length),
        unit = Unit(; m = 1),
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:routing_parameters],
    ),
    "channel__width" => ParameterMetadata(;
        lens = @optic(_.routing.width),
        unit = Unit(; m = 1),
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:routing_parameters],
    ),
    "channel__cell_area" => ParameterMetadata(;
        lens = @optic(_.routing.cell_area),
        unit = Unit(; m = 2),
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:routing_parameters],
    ),
    "channel_water__area~cross-sectional" => ParameterMetadata(;
        lens = @optic(_.routing.area),
        unit = Unit(; m = 2),
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = true,
        output = true,
        tags = [:routing_parameters],
    ),
    "channel_water__volume_flow_rate" => ParameterMetadata(;
        lens = @optic(_.routing.discharge),
        unit = Unit(; m = 3, s = -1),
        description = "Routing cell discharge",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:routing_parameters],
        output_name = "discharge",
    ),
    "channel_water__travel_time" => ParameterMetadata(;
        lens = @optic(_.routing.travel_time),
        unit = Unit(; s = 1),
        description = "Time for water to flow through the grid cell",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:routing_parameters],
        output_name = "travel_time",
    ),
)

"""
Mapping of (CSDMS) standard names to the metadata of the variables that are only computed for
the output (see `process_daily_outputs`), or that are not model state otherwise exposed through
BMI. `~cell-average` marks the grid cell averages of variables that are also available per
vegetation type and snow band.
"""
const standard_output_map = OrderedDict{String, ParameterMetadata}(
    # Forcing
    "atmosphere_bottom_air__temperature" => ParameterMetadata(;
        lens = @optic(_.forcing_variables.air_temperature),
        unit = Unit(; degC = 1),
        description = "Air temperature (from the forcing)",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:forcing_variables],
        output_name = "tair",
    ),
    "atmosphere_water__precipitation_leq-volume_flux" => ParameterMetadata(;
        lens = @optic(_.forcing_variables.precipitation),
        unit = Unit(; mm = 1, d = -1),
        description = "Precipitation (from the forcing)",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:forcing_variables],
        output_name = "precipitation",
    ),

    # Grid cell averages, computed by the postprocessing kernel
    "land_surface__temperature~cell-average" => ParameterMetadata(;
        lens = @optic(_.output_buffers.surface_temperature),
        unit = Unit(; degC = 1),
        description = "Surface temperature, blended with the snow surface temperature according to snow coverage",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:output_variables],
        output_name = "tsurf",
    ),
    "land_surface__evaporation~potential~cell-average" => ParameterMetadata(;
        lens = @optic(_.output_buffers.potential_evaporation),
        unit = Unit(; mm = 1, d = -1),
        description = "Potential evaporation, weighted sum over vegetation types and snow bands",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:output_variables],
        output_name = "potential_evaporation",
    ),
    "land_surface__radiation~net~upward_energy_flux~cell-average" => ParameterMetadata(;
        lens = @optic(_.output_buffers.net_radiation),
        unit = Unit(; J = 1, s = -1, m = -2),
        description = "Net radiation, weighted sum over vegetation types and snow bands",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:output_variables],
        output_name = "net_radiation",
    ),
    "land_vegetation__transpiration" => ParameterMetadata(;
        lens = @optic(_.output_buffers.transpiration),
        unit = Unit(; mm = 1, d = -1),
        description = "Transpiration, weighted sum over vegetation types and snow bands",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:output_variables],
        output_name = "transpiration",
    ),
    "land_vegetation_canopy__evaporation~cell-average" => ParameterMetadata(;
        lens = @optic(_.output_buffers.canopy_evaporation),
        unit = Unit(; mm = 1, d = -1),
        description = "Evaporation of water held on the vegetation canopy, weighted sum over vegetation types and snow bands",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:output_variables],
        output_name = "canopy_evaporation",
    ),
    "land_vegetation_canopy_water__depth" => ParameterMetadata(;
        lens = @optic(_.output_buffers.water_storage),
        unit = Unit(; mm = 1),
        description = "Water held on the vegetation canopy, weighted sum over vegetation types and snow bands",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:output_variables],
        output_name = "water_storage",
    ),

    # Snow grid cell averages, only available when `enable_snow = true`
    "snowpack__liquid-equivalent_depth" => ParameterMetadata(;
        lens = @optic(_.output_buffers.snow_water_equivalent),
        unit = Unit(; mm = 1),
        description = "Snow water equivalent",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:snow_variables],
        output_name = "swe",
    ),
    "snowpack_surface__albedo" => ParameterMetadata(;
        lens = @optic(_.output_buffers.snow_albedo),
        unit = Unit(),
        description = "Snow albedo, only where there is snow",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:snow_variables],
        output_name = "snow_albedo",
    ),
    "snowpack_surface__temperature" => ParameterMetadata(;
        lens = @optic(_.output_buffers.snow_surface_temperature),
        unit = Unit(; degC = 1),
        description = "Snow surface temperature, only where there is snow",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:snow_variables],
        output_name = "snow_surf_temp",
    ),
    "snowpack__area_fraction" => ParameterMetadata(;
        lens = @optic(_.output_buffers.snow_coverage),
        unit = Unit(),
        description = "Fraction of the grid cell covered by snow",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:snow_variables],
        output_name = "snow_coverage",
    ),
    "snowpack__melt_volume_flux" => ParameterMetadata(;
        lens = @optic(_.output_buffers.snow_melt),
        unit = Unit(; mm = 1, d = -1),
        description = "Snowmelt",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:snow_variables],
        output_name = "snow_melt",
    ),
)

#TODO decide which ones to include
const standard_param_map = OrderedDict{String, ParameterMetadata}(
    #Grid
    "elevation" => ParameterMetadata(;
        # (cell)
        lens = @optic(_.grid_parameters.elevation),
        unit = Unit(; m = 1),
        description = "Elevation of grid cell center",
        tags = [:grid_parameters],
    ),
    "average_temperature" => ParameterMetadata(;
        # (cell)
        lens = @optic(_.grid_parameters.average_temperature),
        unit = Unit(; degC = 1),
        description = "average temperature",
        tags = [:grid_parameters],
    ),
    "annual_precipitation" => ParameterMetadata(;
        # (cell)
        lens = @optic(_.grid_parameters.annual_precipitation),
        unit = Unit(; mm = 1, y = -1),
        description = "annual precipitation",
        tags = [:grid_parameters],
    ),
    "snow_band_area_fraction" => ParameterMetadata(;
        #(snow_band, cell)
        lens = @optic(_.grid_parameters.snow_band_area_fraction),
        unit = Unit(),
        description = "Fraction of grid cell area in each snow band",
        tags = [:grid_parameters],
    ),
    "snow_band_elevation" => ParameterMetadata(;
        #(snow_band, cell)
        lens = @optic(_.grid_parameters.snow_band_elevation),
        unit = Unit(; m = 1),
        description = "Elevation of snow bands",
        tags = [:grid_parameters],
    ),
    "snow_band_precipitation_factor" => ParameterMetadata(;
        #(snow_band, cell)
        lens = @optic(_.grid_parameters.snow_band_precipitation_factor),
        unit = Unit(),
        description = "Fraction of cell precipitation that falls on each elevation band",
        tags = [:grid_parameters],
    ),

    #Vegetation  (static)
    "vegetation__root_fraction" => ParameterMetadata(;
        #(veg_class, root_zone, cell)
        lens = @optic(_.vegetation_parameters.root_fraction),
        unit = Unit(),
        description = "Root zone fraction",
        input = true,
        output = false,
        tags = [:vegetation_parameters],
    ),
    "vegetation__fraction" => ParameterMetadata(;
        #(veg_class, cell)
        lens = @optic(_.vegetation_parameters.vegetation_fraction),
        unit = Unit(),
        description = "Vegetation fraction",
        tags = [:vegetation_parameters],
    ),
    "minimum_resistance" => ParameterMetadata(;
        #(veg_class, cell)
        lens = @optic(_.vegetation_parameters.minimum_resistance),
        unit = Unit(; s = 1, m = -1),
        description = "Minimum stomatal resistance",
        tags = [:vegetation_parameters],
    ),
    "architectural_resistance" => ParameterMetadata(;
        #(veg_class, cell)
        lens = @optic(_.vegetation_parameters.architectural_resistance),
        unit = Unit(; m = (2, 2)),
        description = "Architectural resistance",
        tags = [:vegetation_parameters],
    ),
    #active monthly val
    "displacement_height" => ParameterMetadata(;
        #(veg_class, month, cell)
        lens = @optic(_.vegetation_parameters.displacement_height),
        unit = Unit(; m = 1),
        description = "Vegetation displacement",
        tags = [:vegetation_parameters],
    ),
    "roughness_length" => ParameterMetadata(;
        #(veg_class, month, cell)
        lens = @optic(_.vegetation_parameters.roughness_length),
        unit = Unit(; m = 1),
        description = "vegetation roughness length",
        tags = [:vegetation_parameters],
    ),
    "vegetation_canopy__leaf_area_index" => ParameterMetadata(;
        #(veg_class, month, cell)
        lens = @optic(_.vegetation_parameters.lai),
        unit = Unit(; m = (2, 2)),
        description = "leaf area index",
        tags = [:vegetation_parameters],
    ),
    "albedo" => ParameterMetadata(;
        #(veg_class, month, cell)
        lens = @optic(_.vegetation_parameters.albedo),
        unit = Unit(),
        description = "albedo",
        tags = [:vegetation_parameters],
    ),
    "canopy_coverage" => ParameterMetadata(;
        #(veg_class, month, cell)
        lens = @optic(_.vegetation_parameters.canopy_coverage),
        unit = Unit(),
        description = "canopy coverage",
        tags = [:vegetation_parameters],
    ),


    #Soil
    "hydraulic_conductivity" => ParameterMetadata(;
        lens = @optic(_.soil_parameters.hydraulic_conductivity),
        unit = Unit(; mm = 1, d = -1),
        description = "Saturated hydrologic conductivity",
        tags = [:soil_input],
    ),
    "nijssen_nonlin_reservoir" => ParameterMetadata(;
        lens = @optic(_.soil_parameters.nijssen_nonlin_reservoir),
        # TODO: unit is likely wrong. As input (Nijssen D2) it is mm^(1-c) day-1, so mm-1 day-1
        # for c = 2. But `convert_nijssen2001_to_arno!` overwrites this field at initialization
        # with the ARNO Dsmax...
        unit = Unit(; mm = -2, d = -1),
        description = "Nonlinear reservoir coefficient for Nijssen baseflow",
        tags = [:soil_input],
    ),
    "nijssen_infilt_b" => ParameterMetadata(;
        lens = @optic(_.soil_parameters.nijssen_infilt_b),
        unit = Unit(),
        description = "Variable infiltration curve parameter (binfilt) for Nijssen baseflow.",
        tags = [:soil_input],
    )
)

"""
Merge the standard name maps, leaving out the variables of the snow and routing modules when
these are disabled.
"""
function standard_name_maps(; enable_snow = true, enable_routing = true)
    maps = [standard_name_map, standard_output_map]
    enable_routing && push!(maps, standard_routing_map)
    merged = merge(maps...)
    enable_snow || filter!(((_, meta),) -> !(:snow_variables in meta.tags), merged)
    return merged
end

"""
Metadata of the variables that can be written to the output file (those with an `output_name`),
keyed by standard name. Users select which of these to write in the config file.
"""
function output_variables(; kwargs...)
    return filter(((_, meta),) -> !isnothing(meta.output_name), standard_name_maps(; kwargs...))
end
