const standard_name_map = OrderedDict{String, ParameterMetadata}(
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
    )
)