import BasicModelInterface as BMI

"""
    BMI.initialize(::Type{<:Model}, config_file::AbstractString)

Initialize the model. Reads the input settings and data as defined in the Config object
generated from the configuration file `config_file`. Will return a Model that is ready to
run.
"""
function BMI.initialize(::Type{<:Model}, config_file::AbstractString)
    return Model(config_file)
end

function BMI.update(model::Model)
    update!(model)
    return model
end

function BMI.update_until(model::Model, time::Float64)
    end_time = time - Float64(model.clock.dt.value)
    while datetime2unix(model.clock.time) < end_time
        update!(model)
    end

end

# TODO 
function BMI.finalize(model::Model)
    if !isnothing(model.writer)
        # close output
        stop_async_service(model.writer.io_service)
        close_output(model.writer.store)
        @set model.writer = nothing
    end
    return nothing
end


# INPUT, OUTPUT VARIABLE AND MODEL NAME
function BMI.get_component_name(model::Model)
    return "mGV"
end

function BMI.get_input_item_count(model::Model)
    return length(BMI.get_input_var_names(model))
end

"""
    BMI.get_input_var_names(model::Model)

Returns model input variables, based on the `API` section in the model configuration file.
The layer and band variables are expanded, using the ParameterMetadata flag, adding _layer_N 
and _band_N at the end of the name.
The input flag in ParameterMetadata is used to filter the input variables. 
"""
function BMI.get_input_var_names(model::Model)
    return io_var_checker(model,true)
end

function BMI.get_output_item_count(model::Model)
    return length(BMI.get_output_var_names(model))
end

"""
    BMI.get_input_var_names(model::Model)

Returns model output variables, based on the `API` section in the model configuration file.
The layer and band variables are expanded, using the ParameterMetadata flag, adding _layer_N and _band_N at the end of the name.
The output flag in ParameterMetadata is used to filter the output variables. 
"""
function BMI.get_output_var_names(model::Model)
    return io_var_checker(model,false)
end

#VAR GRID, GRID INFO
function BMI.get_var_grid(model::Model, name::String)
    if occursin(r"_layer_\d",name)
        name_2d,  = soil_layer_standard_name(name)
        name = name_2d
    elseif occursin(r"_band_\d",name)
        name_2d, = snow_band_standard_name(name)
        name = name_2d
    end
    metadata = get_metadata(name, model)
    if !isnothing(metadata)
        return metadata.gridtype
    else 
        error("No grid type specification for $name")
    end
end


function BMI.get_grid_type(model::Model, grid::Int)
    if grid == 0
        return "rectilinear" #2D variables (lat,long)
    elseif grid == 1
        return "rectilinear" #2D variables (lat,long) x soil_layers
    elseif grid == 2
        return "rectilinear" #3D variables (lat,long) x (nbsoil_layers-1)
    elseif grid == 3
        return "rectilinear" #3D variables (lat,long,nveg) x nbands
    else
        error("unknown grid type $grid")
    end
end

function BMI.get_grid_rank(model::Model, grid::Int)
    if grid == 0 || grid ==1 || grid == 2
        return 2
    elseif grid == 3
        return 3
    else 
        error("unknown grid type")
    end
end

"""
    BMI.get_grid_shape(model::Model, grid::Int)

Note that this function (as well as the other grid functions) returns information ordered with “ij”
indexing (as opposed to “xy”). For example, consider a two-dimensional rectilinear grid with four 
columns (nx = 4) and three rows (ny = 3). The get_grid_shape function would return a shape of [ny, nx],
or [3,4]. If there were a third dimension, the length of the z-dimension, nz, would be listed first.
"""
function BMI.get_grid_shape(model::Model, grid::Int)
    shape = zeros(Int, BMI.get_grid_rank(model, grid))
    return BMI.get_grid_shape(model, grid, shape)
end

function BMI.get_grid_shape(model::Model, grid::Int, shape::DenseVector{Int})
    if grid == 0 || grid == 1 || grid == 2
        shape[1] = length(model.grid_parameters.latitude)
        shape[2] = length(model.grid_parameters.longitude) 
    elseif grid == 3
        shape[1] = length(model.grid_parameters.vegetation)
        shape[2] = length(model.grid_parameters.latitude)
        shape[3] = length(model.grid_parameters.longitude)
    end
    return shape
end

# RemoteBMI uses outdated BasicModelInterface.jl definition;
function BMI.get_grid_x(model::Model, grid::Int)
    x = zeros(Float64, BMI.get_grid_shape(model, grid)[2])
    return BMI.get_grid_x(model, grid, x)
end

function BMI.get_grid_x(model::Model, grid::Int, x::Vector{Float64})
    if grid <= 3
        copyto!(x,model.grid_parameters.longitude)
    else
        error("unknown grid type $grid")
    end
    return x
end

# RemoteBMI uses outdated BasicModelInterface.jl definition;
function BMI.get_grid_y(model::Model, grid::Int)
    y = zeros(Float64, BMI.get_grid_shape(model, grid)[1])
    return BMI.get_grid_y(model, grid, y)
end

function BMI.get_grid_y(model::Model, grid::Int, y::Vector{Float64})
    if grid <= 3
        copyto!(y,model.grid_parameters.latitude)
    else
        error("unknown grid type $grid")
    end
    return y
end

function BMI.get_grid_z(model::Model, grid::Int)
    if grid == 0 || grid == 1 || grid == 2
        error("accessing z-coordinate in 2-rank grid")
    end
    z=zeros(Int, BMI.get_grid_shape(model,grid)[1])
    return BMI.get_grid_z(model, grid, z)
end


function BMI.get_grid_z(model::Model, grid::Int, z::Vector{Int})
    if grid == 3
        return copyto!(z,model.grid_parameters.vegetation)
    else 
        error("No z-coordinadate for grid value $grid")
    end
end

function BMI.get_grid_size(model::Model, grid::Int)
    y_size = length(model.grid_parameters.latitude)
    x_size = length(model.grid_parameters.longitude)
    if grid != 3         
        return x_size*y_size
    else 
        z_size = length(model.grid_parameters.vegetation)
        return x_size*y_size*z_size
    end
    error("unknown grid type $grid")
end




"""
    BMI.get_var_type(model::Model, name::String)

Return the variable's type with name name
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function BMI.get_var_type(model::Model, name::String)
    value = BMI.get_value_ptr(model, name)
    return repr(eltype(eltype(value)))
end


"""
    BMI.get_var_units(model::Model, name::String)

Return the variable's unit
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function BMI.get_var_units(model::Model, name::String)
    metadata = get_metadata(name, model)
    # return to_string(to_SI(metadata.unit); BMI_standard = true)
    return to_string(to_SI(metadata.unit); BMI_standard = true)
end

"""
    BMI.get_var_itemsize(model::Model, name::String)

Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function BMI.get_var_itemsize(model::Model, name::String)
    value = BMI.get_value_ptr(model, name)
    return sizeof(eltype(eltype(value)))
end

"""
    BMI.get_var_nbytes(model::Model, name::String)

Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function BMI.get_var_nbytes(model::Model, name::String)
    return sizeof(BMI.get_value_ptr(model, name))
end

# TODO 
function BMI.get_var_location(model::Model, name::String)
    error("Get var location is not implemented for this model")
end


#TIME 
function BMI.get_current_time(model::Model)
    return datetime2unix(model.clock.time)
end

function BMI.get_start_time(model::Model)
    return datetime2unix(DateTime(model.config.start_year))
end

function BMI.get_end_time(model::Model)
    return datetime2unix(DateTime(model.config.end_year, 12, 31))
end

function BMI.get_time_units(::Model)
    return "seconds since 1970-01-01T00:00+00:00"
end

function BMI.get_time_step(model::Model)
    return Float64(model.config.timestep)
end


"""
    BMI.get_value(model::Model, name::String, dest)

Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function BMI.get_value(model::Model, name::String, dest)
    copyto!(dest,copy(BMI.get_value_ptr(model, name)))
    return dest
end


"""
    BMI.get_value_ptr(model::Model, name::String)

Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function BMI.get_value_ptr(model::Model, name::String)
    if occursin(r"_layer_\d",name)
        name_2d, ind = soil_layer_standard_name(name)
        model_vals, (;gridtype) = get_field_in_model(model, name_2d)
        if gridtype != 3
            return @view model_vals[:,:,ind]
        else 
            return @view model_vals[:,:,ind,:]
        end
    elseif occursin(r"_band_\d",name)
        name_2d, ind = snow_band_standard_name(name)
        model_vals, (;gridtype) = get_field_in_model(model, name_2d)
        if gridtype != 3
            return @view model_vals[:,:,ind]
        else 
            return @view model_vals[:,:,ind,:]
        end
    else 
        (; lens) = get_metadata(name; model)
        if isnothing(lens)
            error("Accessing '$name' is not supported.")
        else
            dest = lens(model)
            if ndims(dest)>1
                dest = vec(dest)
            end
            n = length(dest)
            return @view(dest[1:n])
        end
    end
end

"""
    BMI.get_value_at_indices(model::Model, name::String)

Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function BMI.get_value_at_indices(
        model::Model,
        name::String,
        dest,
        inds::Vector{Int},
    )
    src = BMI.get_value_ptr(model, name)
    copyto!(dest, src[inds])
    return dest
end


"""
    BMI.set_value(model::Model, name::String, src)

Set a model variable `name` to the values in `src`, overwriting the current contents.
The type and size of `src` must match the model's internal array.
Based on Wflow.jl https://github.com/Deltares/Wflow.jl 
"""
function BMI.set_value(model::Model, name::String, src)
    l_src = length(src)
    if !check_if_settable(model, name)
        @warn "Variable $name is not settable, returning nothing"
        return nothing
    end
    mod_var = BMI.get_value_ptr(model, name)
    l_model = length(mod_var)
    if l_src != l_model
        error("Length mismatch between model $(l_model) and src $(l_src) variable $name\n 
        Probably you're trying to access a variable without specifying a given layer \n
        Ratio between lenghts $(l_model/l_src)")
    end
    return copyto!(mod_var,src)
end

"""
    BMI.set_value_at_indices(model::Model, name::String, inds::Vector{Int}, src)

Set a model variable `name` to the values in `src`, at indices `inds`.
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function BMI.set_value_at_indices(
        model::Model,
        name::String,
        inds::Vector{Int},
        src,
    )
    dest = BMI.get_value_ptr(model, name)
    dest[inds] = src
    return dest
end






# BMI helper functions.
"""
Return the standard name representing a layered variable 3D 
and the layer index `layer_index` based on a standard `name`.
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function soil_layer_standard_name(name::AbstractString)
    # Parse new naming format: soil_layer_N_water_... where N is the layer number
    parts = split(name, "_")
    layer_index = tryparse(Int, parts[end])
    if !isnothing(layer_index)
        # Remove the layer number to get the base name
        name_layered = join(parts[1:end-2], "_") 
        return name_layered, layer_index
    end
    # Fallback for unexpected format
    @warn "Unable to parse layer standard name: $name"
    return name, nothing
end

"""
Return the standard name representing a snow-band variable 3D 
and the corresponding index `band_index` based on a standard `name`.
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function snow_band_standard_name(name::AbstractString)
    # Parse new naming format: snow_band_N where N is the layer number
    parts = split(name, "_")
    band_index = tryparse(Int, parts[end])
    if !isnothing(band_index)
        # Remove the layer number to get the base name
        name_without_band = join(parts[1:end-2], "_") 
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
    standard_map = get_standard_name_map()
    keep_names = String[]

    filtered = filter(name -> haskey(standard_map, name), config_vars)
    no_match = filter(name -> !haskey(standard_map, name), config_vars)
    for var in no_match 
        @warn("$var is not listed as variable for BMI exchange and removed from input list") 
    end
    if is_input
        filter!(name -> standard_map[name].input, filtered)
    else 
        filter!(name -> standard_map[name].output, filtered)
    end 
    if isempty(filtered)
        return []
    end
        
    for var in filtered
        metadata = get_metadata(var,model)
        if :soil_variables_layer in metadata.tags
            for j in 1:size(model.soil_parameters.depth,3)
                push!(keep_names, var*"_layer_$j")
            end
        elseif :soil_variables_interlayer in metadata.tags
            for j in 1:(size(model.soil_parameters.depth,3)-1)
                push!(keep_names, var*"_layer_$j")
            end
        elseif (:surface_energy_variables_band in metadata.tags ||
                :canopy_variables_band in metadata.tags)
            for j in 1:model.config.nbands
                push!(keep_names, var*"_band_$j")
            end
        else
            push!(keep_names,var)
        end
    end
    return keep_names
end

"""
Checks if a given variable can be setted looking at the ParameterMetadata input flags. 
"""
function check_if_settable(model::Model, name::String)
    if occursin(r"_layer_\d",name)
        name_2d, _ = soil_layer_standard_name(name)
        (;input) = get_metadata(name_2d, model)
        return input
    elseif occursin(r"_band_\d",name)
        name_2d, _ = snow_band_standard_name(name)
        (;input) = get_metadata(name_2d, model)
        return input
    else 
        (; input) = get_metadata(name; model)
        return input
    end
end