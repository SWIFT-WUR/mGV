"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl/blob/main/Wflow/src/standard_name/standard_name_utils.jl 
MIT License, Copyright (c) 2020 Deltares and contributors
"""

"""
Metadata associated with parameters and variables.

# Arguments
- `lens`: The path in the model data structure to the parameter/variable if it exists
- `unit`: The unit of the parameter/variable in the Wflow input
- `description`: The description of the parameter/variable provided in the Wflow docs
- 'gridtype': The grid type of the paremeter/variable if it exists
- 'input': flag needed to set a variable
- 'output': flag needed to get a variable
- `tags`: Identifiers to filter parameters/variables for specific tables in the docs
- `output_name`: Name of the variable in the output file, or `nothing` if it cannot be written to file
"""
@kwdef struct ParameterMetadata{L}
    lens::L = nothing
    unit::Unit = EMPTY_UNIT
    description::String = ""
    gridtype::Int = 0
    input::Bool = false
    output::Bool = false
    tags::Vector{Symbol} = []
    output_name::Union{String, Nothing} = nothing
    function ParameterMetadata(
            lens::L,
            unit,
            description,
            gridtype,
            input,
            output,
            flags,
            output_name,
        ) where {L}
        return new{L}(lens, unit, description, gridtype, input, output, flags, output_name)
    end
end

function Base.:(==)(a::ParameterMetadata, b::ParameterMetadata)
    return all(getfield(a, f) == getfield(b, f) for f in fieldnames(ParameterMetadata))
end
