"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl/blob/main/Wflow/src/standard_name/standard_name_utils.jl 
MIT License, Copyright (c) 2020 Deltares and contributors
"""

"""
How an output variable is stored as Int16: a value `x` is stored as
`round((x - offset) * scale)`, rounded to nearest even, so in steps of `1 / scale` from
`offset - 32767 / scale` to `offset + 32767 / scale`. Values outside that range are clipped
to it, and NaN is stored as `typemin(Int16)`. This is the Zarr v3 `scale_offset` codec
followed by `cast_value` (see ZarrScaleCast.jl); NetCDF has the equivalent CF attributes
`scale_factor = 1 / scale` and `add_offset = offset`.

Choose `scale` from the precision that is needed: a lower scale compresses better.
"""
struct Packing
    scale::Float32
    offset::Float32
end

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
- `packing`: How the output is stored as Int16 (see [`Packing`](@ref)), or `nothing` to store
  it as Float32
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
    packing::Union{Packing, Nothing} = nothing
    function ParameterMetadata(
            lens::L,
            unit,
            description,
            gridtype,
            input,
            output,
            flags,
            output_name,
            packing,
        ) where {L}
        return new{L}(lens, unit, description, gridtype, input, output, flags, output_name, packing)
    end
end

function Base.:(==)(a::ParameterMetadata, b::ParameterMetadata)
    return all(getfield(a, f) == getfield(b, f) for f in fieldnames(ParameterMetadata))
end
