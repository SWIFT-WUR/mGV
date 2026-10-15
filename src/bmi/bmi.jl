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
    return nothing
end

function BMI.update_until(model::Model, time::Float64)
    model_time = BMI.get_current_time(model)
    end_time = BMI.get_end_time(model)
    if time == model_time
        warn("Model is already at time=$time")
        return nothing
    elseif time < model_time
        error("Time must be in the model's future")
    elseif time > end_time
        error("Time can't exceed the current end time: $end_time")
    end

    _div, _rem = divrem(time-model_time, BMI.get_time_step(model))
    steps = Int(_div)
    if abs(_rem) > eps()
        error(
            "Provided `time` $time minus timestamp $t is not a multiple of $(model.clock.dt.value)",
        )
    end

    for _ = 1:steps
        update!(model)
    end
    return nothing
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
    return io_var_checker(model, true)
end

function BMI.get_output_item_count(model::Model)
    return length(BMI.get_output_var_names(model))
end

"""
    BMI.get_output_var_names(model::Model)

Returns model output variables, based on the `API` section in the model configuration file.
The layer and band variables are expanded, using the ParameterMetadata flag, adding _layer_N and _band_N at the end of the name.
The output flag in ParameterMetadata is used to filter the output variables. 
"""
function BMI.get_output_var_names(model::Model)
    return io_var_checker(model, false)
end

#VAR GRID, GRID INFO
function BMI.get_var_grid(model::Model, name::String)
    metadata = check_name(model, name)
    if !isnothing(metadata)
        return metadata.gridtype
    else
        error("No grid type specification for $name")
    end
end

function BMI.get_grid_type(model::Model, grid::Int)
    if grid == GRIDS["2D_grid"]
        return "rectilinear" #2D variables (lat,long)
    elseif grid == GRIDS["soil_grid"]
        return "rectilinear" #2D variables (lat,long) x soil_layers
    elseif grid == GRIDS["inter_soil_grid"]
        return "rectilinear" #3D variables (lat,long) x (nbsoil_layers-1)
    elseif grid == GRIDS["vegetation_snow_grid"]
        return "rectilinear" #3D variables (lat,long,nveg) x nbands
    else
        error("unknown grid type $grid")
    end
end

function BMI.get_grid_rank(model::Model, grid::Int)
    if grid in GRIDS_2D
        return 2
    elseif grid == GRIDS["vegetation_snow_grid"]
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
    if grid in GRIDS_2D
        shape[1] = length(model.grid_parameters.latitude)
        shape[2] = length(model.grid_parameters.longitude)
    elseif grid == GRIDS["vegetation_snow_grid"]
        shape[1] = model.config.nveg
        shape[2] = length(model.grid_parameters.latitude)
        shape[3] = length(model.grid_parameters.longitude)
    end
    return shape
end

# RemoteBMI uses outdated BasicModelInterface.jl definition;
function BMI.get_grid_x(model::Model, grid::Int)
    x = zeros(Float64, BMI.get_grid_shape(model, grid)[end])
    return BMI.get_grid_x(model, grid, x)
end

function BMI.get_grid_x(model::Model, grid::Int, x::Vector{Float64})
    if grid in values(GRIDS)
        copyto!(x, model.grid_parameters.longitude)
    else
        error("unknown grid type $grid")
    end
    return x
end

# RemoteBMI uses outdated BasicModelInterface.jl definition;
function BMI.get_grid_y(model::Model, grid::Int)
    y = zeros(Float64, BMI.get_grid_shape(model, grid)[end-1])
    return BMI.get_grid_y(model, grid, y)
end

function BMI.get_grid_y(model::Model, grid::Int, y::Vector{Float64})
    if grid in values(GRIDS)
        copyto!(y, model.grid_parameters.latitude)
    else
        error("unknown grid type $grid")
    end
    return y
end

function BMI.get_grid_z(model::Model, grid::Int)
    if grid == GRIDS["vegetation_snow_grid"] 
        z=zeros(Float64, BMI.get_grid_shape(model, grid)[begin])
        return BMI.get_grid_z(model, grid, z)
    else
        error("accessing z-coordinate in 2-rank grid")
    end
end


function BMI.get_grid_z(model::Model, grid::Int, z::Vector{Float64})
    if grid == GRIDS["vegetation_snow_grid"]
        return copyto!(z, 1:BMI.get_grid_shape(model, grid)[1])
    else
        error("No z-coordinate for grid value $grid")
    end
end

function BMI.get_grid_size(model::Model, grid::Int)
    shape = BMI.get_grid_shape(model, grid)
    return prod(shape)
    error("unknown grid type $grid")
end




"""
    BMI.get_var_type(model::Model, name::String)

Return the variable's type with name name
"""
function BMI.get_var_type(model::Model, name::String)
    value = BMI.get_value_ptr(model, name)
    return repr(eltype(eltype(value)))
end


"""
    BMI.get_var_units(model::Model, name::String)

Return the variable's unit
"""
function BMI.get_var_units(model::Model, name::String)
    metadata = check_name(model, name)
    return to_string(to_SI(metadata.unit); BMI_standard = true)
end

"""
    BMI.get_var_itemsize(model::Model, name::String)
"""
function BMI.get_var_itemsize(model::Model, name::String)
    value = BMI.get_value_ptr(model, name)
    return sizeof(eltype(eltype(value)))
end

"""
    BMI.get_var_nbytes(model::Model, name::String)
"""
function BMI.get_var_nbytes(model::Model, name::String)
    return sizeof(BMI.get_value_ptr(model, name))
end

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
"""
function BMI.get_value(model::Model, name::String, dest)
    if occursin(r"_band_\d", name)
        #the extra copy is needed to create a dense vector from 
        # non-contiguous gpu arrays (nbands)
        copyto!(dest, copy(BMI.get_value_ptr(model, name)))
    else
        copyto!(dest, BMI.get_value_ptr(model, name))
    end
    return dest
end


"""
    BMI.get_value_ptr(model::Model, name::String)
"""
function BMI.get_value_ptr(model::Model, name::String)
    if occursin(r"_layer_\d", name)
        name_2d, ind = soil_layer_standard_name(name)
        model_vals, (; gridtype) = get_field_in_model(model, name_2d)
        if gridtype != GRIDS["vegetation_snow_grid"]
            return @view model_vals[:, :, ind]
        else
            return @view model_vals[:, :, ind, :]
        end
    elseif occursin(r"_band_\d", name)
        name_2d, ind = snow_band_standard_name(name)
        model_vals, (; gridtype) = get_field_in_model(model, name_2d)
        if gridtype != GRIDS["vegetation_snow_grid"]
            return @view model_vals[:, :, ind]
        else
            return @view model_vals[:, :, ind, :]
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
"""
function BMI.get_value_at_indices(model::Model, name::String, dest, inds::Vector{Int})
    src = BMI.get_value_ptr(model, name)
    copyto!(dest, src[inds])
    return dest
end


"""
    BMI.set_value(model::Model, name::String, src)

Set a model variable `name` to the values in `src`, overwriting the current contents.
The type and size of `src` must match the model's internal array.
"""
function BMI.set_value(model::Model, name::String, src)
    l_src = length(src)
    if !check_if_settable(model, name)
        error("Variable $name is not settable")
    end

    grid = BMI.get_var_grid(model, name)
    l_model = BMI.get_grid_size(model, grid)
    if l_src != l_model
        error(
            "Length mismatch between model $(l_model) and src $(l_src) ",
            "variable $name. Probably you're trying to access a variable without ",
            "specifying a given layer.\n",
            "Ratio between lengths: $(l_model/l_src)"
        )
    end
    mod_var = BMI.get_value_ptr(model, name)
    copyto!(mod_var, src)
    return nothing
end

"""
    BMI.set_value_at_indices(model::Model, name::String, inds::Vector{Int}, src)

Set a model variable `name` to the values in `src`, at indices `inds`.
"""
function BMI.set_value_at_indices(model::Model, name::String, inds::Vector{Int}, src)
    dest = BMI.get_value_ptr(model, name)
    dest[inds] = src
    return dest
end
