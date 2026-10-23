"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl/blob/main/Wflow/src/standard_name/standard_name_utils.jl 
MIT License, Copyright (c) 2020 Deltares and contributors
"""


# wrapper methods for standard name mapping
function get_standard_name_map(model::Model)
    if isnothing(model.routing)
        return standard_name_map
    else
        name_map = merge(standard_name_map, standard_routing_map)
        return name_map
    end
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
"""
@kwdef struct ParameterMetadata{L}
    lens::L = nothing
    unit::Unit = EMPTY_UNIT
    description::String = ""
    gridtype::Int = 0
    input::Bool = false
    output::Bool = false
    tags::Vector{Symbol} = []
    function ParameterMetadata(
            lens::L,
            unit,
            description,
            gridtype,
            input,
            output,
            flags,
        ) where {L}
        return new{L}(lens, unit, description, gridtype, input, output, flags)
    end
end

function Base.:(==)(a::ParameterMetadata, b::ParameterMetadata)
    return all(getfield(a, f) == getfield(b, f) for f in fieldnames(ParameterMetadata))
end

function metadata_from_lens_string(
        lens_string::AbstractString,
        standard_name_map::OrderedDict{String, ParameterMetadata},
    )::Union{ParameterMetadata, Nothing}
    for metadata_candidate in values(standard_name_map)
        if string(metadata_candidate.lens)[7:(end - 1)] == lens_string
            return metadata_candidate
        end
    end
    return nothing
end

function get_metadata(
        name::AbstractString,
        types::Vararg{Type};
        model::Model,
    )::Union{ParameterMetadata, Nothing}
    metadata = nothing
    for type in types
        standard_name_map = get_standard_name_map(model)
        # First see whether 'name' is a standard name within the standard name map
        # corresponding to 'type'
        metadata_candidate = get(standard_name_map, name, nothing)
        # If not, see whether 'name' is a path in the model object which matches
        # a lens in the standard name map
        if isnothing(metadata_candidate)
            metadata_candidate = metadata_from_lens_string(name, standard_name_map)
        end

        if !isnothing(metadata_candidate) && (metadata != metadata_candidate)
            # Metadata was found; check whether the lens matches
            if !isnothing(metadata_candidate.lens)
                found_matching_lens = false
                try
                    metadata_candidate.lens(model)
                    found_matching_lens = true
                catch
                end
                if found_matching_lens
                    !isnothing(metadata) && error(
                        "Ambiguity found for obtaining metadata for '$name'; this key is in the standard nampe map for at least 2 of $types with a fitting lens.",
                    )
                    metadata = metadata_candidate
                end
            else
                # If model or lens was not provided assume that the metadata matches
                !isnothing(metadata) && error(
                    "Ambiguity found for obtaining metadata for '$name'; this key is in the standard name map for at least 2 of $types and there was no model provided to disambiguate.",
                )
                metadata = metadata_candidate
            end
        end
    end
    return metadata
end

function get_metadata(
        name::AbstractString;
        model::Model,
        kwargs...,
    )::Union{ParameterMetadata, Nothing}
    # Check whether it is a land variable first
    metadata = get(get_standard_name_map(model), name, nothing)

    if isnothing(metadata)
        error("Unable to find metadata related to variable $name")
    end
    return metadata
end

get_metadata(name::AbstractString, model) = get_metadata(name, typeof(model); model)

function get_field_in_model(model, name::AbstractString)
    metadata = get_metadata(name; model)

    return if !isnothing(metadata)
        # If metadata was found, `str` is a standard name or a path in the model object which matches a lens
        metadata.lens(model), metadata
    else
        error("Couldn't obtain a field from this model specified by '$name'.")
    end
end
