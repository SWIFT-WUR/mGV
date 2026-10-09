"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
Mapping of grid identifier to a key, to get some context and later on to retrieve 
the active indices of the model domain as in Wflow, active_indices(network, key::AbstractString).
"""
const GRIDS = Dict{String, Int}(
    "2D_grid" => 0,
    "soil_grid" => 1,
    "inter_soil_grid" => 2,
    "vegetation_snow_grid" => 3,
)

"""
Mapping of (CSDMS) standard names to the metadata associated with the corresponding
variable or parameter. For more details and default values see `ParameterMetadata`.
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
const standard_name_map = OrderedDict{String, ParameterMetadata}(
    #Surface energy variables
    # > 2D
    "land_surface__radiation~net~upward_energy_flux" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.net_radiation),
        unit = Unit(; J = 1, s = -1, m = -2),
        default = 1.0,
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables_band],    
    ),
    "land_surface__evaporation~potential" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.potential_evaporation),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables_band],    
    ),
    "land_surface__evaporation~potential~soil" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.soil_potential_evaporation),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables_band],    
    ),
    "atmosphere_bottom_air__resistance~aerodynamic" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.aerodynamic_resistance),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables_band],    
    ),

    # 2D
    "land_surface__temperature" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.surface_temperature),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables],    
    ),
    "land_surface__evaporation~total" => ParameterMetadata(;
        lens = @optic(_.surface_energy_variables.total_evapotranspiration),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:surface_energy_variables],    
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
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["soil_grid"],
        input = true,
        output = true,
        tags = [:soil_variables_layer],    
    ),
    "soil__temperature" => ParameterMetadata(;
        lens = @optic(_.soil_variables.temperature),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["soil_grid"],
        input = false,
        output = true,
        tags = [:soil_variables_layer],    
    ),
    "soil__ice_fraction" => ParameterMetadata(;
        lens = @optic(_.soil_variables.ice_fraction),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["soil_grid"],
        input = false,
        output = true,
        tags = [:soil_variables_layer],    
    ),
    "soil__drainage~interlayer" => ParameterMetadata(;
        lens = @optic(_.soil_variables.interlayer_drainage),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["inter_soil_grid"],
        input = false,
        output = true,
        tags = [:soil_variables_interlayer],    
    ),
    "soil__thermal_conductivity" => ParameterMetadata(;
        lens = @optic(_.soil_variables.thermal_conductivity),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["soil_grid"],
        input = false,
        output = true,
        tags = [:soil_variables_layer],    
    ),

    #2D
    "soil__evaporation" => ParameterMetadata(;
        lens = @optic(_.soil_variables.evaporation),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],    
    ),
    "soil__infiltration" => ParameterMetadata(;
        lens = @optic(_.soil_variables.infiltration),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],    
    ),
    "land_surface__runoff" => ParameterMetadata(;
        lens = @optic(_.soil_variables.surface_runoff),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],    
    ),
    "land_subsurface__runoff" => ParameterMetadata(;
        lens = @optic(_.soil_variables.subsurface_runoff),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],    
    ),
    "land__runoff~total" => ParameterMetadata(;
        lens = @optic(_.soil_variables.total_runoff),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:soil_variables],    
    ),

    #Canopy variables >2d
    "land_vegetation_canopy__evaporation" => ParameterMetadata(;
        lens = @optic(_.canopy_variables.canopy_evaporation),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["vegetation_snow_grid"],
        input = false,
        output = true,
        tags = [:canopy_variables_band],    
    ),

    #Routing parameters
    "channel__length" => ParameterMetadata(;
        lens = @optic(_.routing.length),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:routing_parameters],    
    ),
    "channel__cell_area" => ParameterMetadata(;
        lens = @optic(_.routing.cell_area),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:routing_parameters],    
    ),
    "channel_water__area~cross-sectional" => ParameterMetadata(;
        lens = @optic(_.routing.area),
        unit = Unit(),
        default = 1.0,
        description = "",
        gridtype = GRIDS["2D_grid"],
        input = false,
        output = true,
        tags = [:routing_parameters],    
    ),
)

#TODO decide which ones to include
const standard_param_map = OrderedDict{String, ParameterMetadata}(
    #Grid 
    "elevation" => ParameterMetadata(; #(lat, lon) 
        lens = @optic(_.grid_parameters.elevation),
        unit = Unit(; m = 1),
        default = 1.0,
        description = "Elevation of grid cell center",
        tags = [:grid_parameters],
    ),
    "average_temperature" => ParameterMetadata(; # (lat, lon) 
        lens = @optic(_.grid_parameters.average_temperature),
        unit = Unit(; degC = 1),
        default = 1.0,
        description = "average temperature",
        tags = [:grid_parameters],
    ),
    "annual_precipitation" => ParameterMetadata(; # (lat, lon) 
        lens = @optic(_.grid_parameters.annual_precipitation),
        unit = Unit(; mm = 1, y = -1),
        default = 1.0,
        description = "annual precipitation",
        tags = [:grid_parameters],
    ),
    "snow_band_area_fraction" => ParameterMetadata(; #(snow_band, lat, lon)
        lens = @optic(_.grid_parameters.snow_band_area_fraction),
        unit = Unit(),
        default = 1.0,
        description = "Fraction of grid cell area in each snow band",
        tags = [:grid_parameters],
    ),
    "snow_band_elevation" => ParameterMetadata(; #(snow_band, lat, lon)
        lens = @optic(_.grid_parameters.snow_band_elevation),
        unit = Unit(; m=1),
        default = 1.0,
        description = "Elevation of snow bands",
        tags = [:grid_parameters],
    ),
    "snow_band_precipitation_factor" => ParameterMetadata(; #(snow_band, lat, lon)
        lens = @optic(_.grid_parameters.snow_band_precipitation_factor),
        unit = Unit(),
        default = 1.0,
        description = "Fraction of cell precipitation that falls on each elevation band",
        tags = [:grid_parameters],
    ),

    #Vegetation  (static)
    "vegetation__root_fraction" => ParameterMetadata(; #(veg_class, root_zone, lat, lon)
        lens = @optic(_.vegetation_parameters.root_fraction),
        unit = Unit(),
        default = 1.0,
        description = "Root zone fraction",
        input = true,
        output = false,
        tags = [:vegetation_parameters],
    ),
    "vegetation__fraction" => ParameterMetadata(; #(veg_class, lat, lon)
        lens = @optic(_.vegetation_parameters.vegetation_fraction),
        unit = Unit(),
        default = 1.0,
        description = "Vegetation fraction",
        tags = [:vegetation_parameters],
    ),
    "minimum_resistance" => ParameterMetadata(; #(veg_class, lat, lon)
        lens = @optic(_.vegetation_parameters.minimum_resistance),
        unit = Unit(; s=1, m=-1),
        default = 1.0,
        description = "Minimum stomatal resistance",
        tags = [:vegetation_parameters],
    ),
    "architectural_resistance" => ParameterMetadata(; #(veg_class, lat, lon)
        lens = @optic(_.vegetation_parameters.architectural_resistance),
        unit = Unit(; m = (2, 2)),
        default = 1.0,
        description = "Architectural resistance",
        tags = [:vegetation_parameters],
    ),
    #active monthly val
    "displacement_height" => ParameterMetadata(; #(veg_class, month, lat, lon)
        lens = @optic(_.vegetation_parameters.displacement_height),
        unit = Unit(; m=1),
        default = 1.0,
        description = "Vegetation displacement",
        tags = [:vegetation_parameters],
    ),
    "roughness_length" => ParameterMetadata(; #(veg_class, month, lat, lon)
        lens = @optic(_.vegetation_parameters.roughness_length),
        unit = Unit(; m=1),
        default = 1.0,
        description = "vegetation roughness length",
        tags = [:vegetation_parameters],
    ),
    "vegetation_canopy__leaf_area_index" => ParameterMetadata(; #(veg_class, month, lat, lon)
        lens = @optic(_.vegetation_parameters.lai),
        unit = Unit(; m = (2, 2)),
        default = 1.0,
        description = "leaf area index",
        tags = [:vegetation_parameters],
    ),
    "albedo" => ParameterMetadata(; #(veg_class, month, lat, lon)
        lens = @optic(_.vegetation_parameters.albedo),
        unit = Unit(),
        default = 1.0,
        description = "albedo",
        tags = [:vegetation_parameters],
    ),
    "canopy_coverage" => ParameterMetadata(; #(veg_class, month, lat, lon) 
        lens = @optic(_.vegetation_parameters.canopy_coverage),
        unit = Unit(),
        default = 1.0,
        description = "canopy coverage",
        tags = [:vegetation_parameters],
    ),


    #Soil
    "hydraulic_conductivity" => ParameterMetadata(;
        lens = @optic(_.soil_parameters.hydraulic_conductivity),
        unit = Unit(; mm = 1, d = -1),
        default = 1.0,
        description = "Saturated hydrologic conductivity",
        tags = [:soil_input],
    ),
    "nijssen_nonlin_reservoir" => ParameterMetadata(;
        lens = @optic(_.soil_parameters.nijssen_nonlin_reservoir),
        unit = Unit(; mm = -2, d = -1),
        default = 1.0,
        description = "Nonlinear reservoir coefficient for Nijssen baseflow",
        tags = [:soil_input],    
    ),
    "nijssen_infilt_b" => ParameterMetadata(;
        lens = @optic(_.soil_parameters.nijssen_infilt_b),
        unit = Unit(),
        default = 1.0,
        description = "Variable infiltration curve parameter (binfilt) for Nijssen baseflow.",
        tags = [:soil_input],    
    )
)