# BMI helper functions.
"""
Return the standard name representing a layered variable 3D 
and the layer index `layer_index` based on a standard `name`.
"""
function soil_layer_standard_name(name::AbstractString)
    # Parse new naming format: soil_layer_N_water_... where N is the layer number
    parts = split(name, "_")
    layer_index = tryparse(Int, parts[end])
    if !isnothing(layer_index)
        # Remove the layer number to get the base name
        name_layered = join(parts[1:(end-2)], "_")
        return name_layered, layer_index
    end
    # Fallback for unexpected format
    @warn "Unable to parse layer standard name: $name"
    return name, nothing
end

"""
Return the standard name representing a snow-band variable 3D 
and the corresponding index `band_index` based on a standard `name`.
"""
function snow_band_standard_name(name::AbstractString)
    # Parse new naming format: snow_band_N where N is the layer number
    parts = split(name, "_")
    band_index = tryparse(Int, parts[end])
    if !isnothing(band_index)
        # Remove the layer number to get the base name
        name_without_band = join(parts[1:(end-2)], "_")
        return name_without_band, band_index
    end
    # Fallback for unexpected format
    @warn "Unable to parse layer standard name: $name"
    return name, nothing
end

"""
Expands the input/output variables name received through API section of the config file. 
The variables are expandend attaching 'layer_N' or 'band_n' at the end of the name. 
The ParameterMetadata input/output flags are used to filter the variables accordingly. 
"""
function io_var_checker(model::Model, is_input::Bool)
    config_vars = model.config.API.variables
    standard_map = get_standard_name_map(model)
    keep_names = String[]

    filtered = filter(name -> haskey(standard_map, name), config_vars)
    no_match = filter(name -> !haskey(standard_map, name), config_vars)
    if is_input
        filter!(name -> standard_map[name].input, filtered)
    else
        filter!(name -> standard_map[name].output, filtered)
    end
    if isempty(filtered)
        return String[]
    end

    for var in filtered
        metadata = get_metadata(var, model)
        if :soil_variables_layer in metadata.tags
            for j = 1:size(model.soil_parameters.depth, 3)
                push!(keep_names, var*"_layer_$j")
            end
        elseif :soil_variables_interlayer in metadata.tags
            for j = 1:(size(model.soil_parameters.depth, 3)-1)
                push!(keep_names, var*"_layer_$j")
            end
        elseif (
            :surface_energy_variables_band in metadata.tags ||
            :canopy_variables_band in metadata.tags
        )
            for j = 1:model.config.nbands
                push!(keep_names, var*"_band_$j")
            end
        else
            push!(keep_names, var)
        end
    end
    return keep_names
end

function check_name(model::Model, name::String)
    if occursin(r"_layer_\d", name)
        name_2d, = soil_layer_standard_name(name)
        name = name_2d
    elseif occursin(r"_band_\d", name)
        name_2d, = snow_band_standard_name(name)
        name = name_2d
    end
    return get_metadata(name, model)
end

"""
Checks if a given variable can be setted looking at the ParameterMetadata input flags. 
"""
function check_if_settable(model::Model, name::String)
    (; input) = check_name(model, name)
    return input
end