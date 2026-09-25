


const standard_name_map = OrderedDict{String, ParameterMetadata}(
    "hydraulic_conductivity" => ParameterMetadata(;
        lens = @optic(_.soil_parameters.hydraulic_conductivity),
        unit = Unit(; mm = 1, d = -1),
        default = 1.0,
        description = "Saturated hydrologic conductivity",
        tags = [:soil_input],
    )
)