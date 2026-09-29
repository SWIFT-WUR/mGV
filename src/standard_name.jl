const standard_name_map = OrderedDict{String, ParameterMetadata}(

    #Grid parameters
    "latitude" => ParameterMetadata(; 
        lens = @optic(_.grid_parameters.latitude),
        unit = Unit(; deg = 1),
        default = 1.0,
        description = "Latitude",
        tags = [:grid_parameters],
    ),
    "longitude" => ParameterMetadata(;
        lens = @optic(_.grid_parameters.longitude),
        unit = Unit(; deg = 1),
        default = 1.0,
        description = "longitude",
        tags = [:grid_parameters],
    ),
    "time" => ParameterMetadata(;
        lens = @optic(_.grid_parameters.time),
        unit = Unit(; s = 1),
        default = 1.0,
        description = "time",
        tags = [:grid_parameters],
    ),
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

    #per-snowband variables 
    "snow_band_area_fraction_layer" => ParameterMetadata(; #(snow_band, lat, lon)
        lens = @optic(_.grid_parameters.snow_band_area_fraction),
        unit = Unit(),
        default = 1.0,
        description = "Fraction of grid cell area in each snow band",
        tags = [:grid_parameters],
    ),
    "snow_band_elevation_layer" => ParameterMetadata(; #(snow_band, lat, lon)
        lens = @optic(_.grid_parameters.snow_band_elevation),
        unit = Unit(; m=1),
        default = 1.0,
        description = "Elevation of snow bands",
        tags = [:grid_parameters],
    ),
    "snow_band_precipitation_factor_layer" => ParameterMetadata(; #(snow_band, lat, lon)
        lens = @optic(_.grid_parameters.snow_band_precipitation_factor),
        unit = Unit(),
        default = 1.0,
        description = "Fraction of cell precipitation that falls on each elevation band",
        tags = [:grid_parameters],
    ),






    #Vegetation (static)
    "root_fraction" => ParameterMetadata(; #(veg_class, root_zone, lat, lon)
        lens = @optic(_.vegetation_parameters.root_fraction),
        unit = Unit(),
        default = 1.0,
        description = "Root zone fraction",
        tags = [:vegetation_parameters],
    ),
    "vegetation_fraction" => ParameterMetadata(; #(veg_class, lat, lon)
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
    "lai" => ParameterMetadata(; #(veg_class, month, lat, lon)
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
    ),

    "infiltration" => ParameterMetadata(;
        lens = @optic(_.soil_variables.infiltration),
        unit = Unit(),
        default = 1.0,
        description = "",
        tags = [:soil_variables],    
    ),
)


