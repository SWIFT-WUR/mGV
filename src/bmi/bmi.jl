import BasicModelInterface as BMI


"""
    BMI.initialize(::Type{<:Model}, config_file::AbstractString)

Initialize the model. Reads the input settings and data as defined in the Config object
generated from the configuration file `config_file`. Will return a Model that is ready to
run.

The backend is selected automatically, unless the `MGV_BACKEND` environment variable is
set to a backend name (see [`select_backend`](@ref)). This allows BMI hosts that can only
call `initialize` to pick a backend.
"""
function BMI.initialize(::Type{<:Model}, config_file::AbstractString)
    backend = get(ENV, "MGV_BACKEND", "")
    return Model(config_file; backend = isempty(backend) ? "auto" : backend)
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

    _div, _rem = divrem(time - model_time, BMI.get_time_step(model))
    steps = Int(_div)
    if abs(_rem) > eps()
        error(
            "Provided `time` $time minus timestamp $t is not a multiple of $(model.clock.dt.value)",
        )
    end

    for _ in 1:steps
        update!(model)
    end
    return nothing
end

function BMI.finalize(model::Model)
    finalize!(model)
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

"""
    BMI.get_grid_type(model::Model, grid::Int)

All grids are unstructured: the model only holds the active (land) cells of the input grid,
see [`Domain`](@ref). Their coordinates are given by `get_grid_x` and `get_grid_y`.
"""
function BMI.get_grid_type(model::Model, grid::Int)
    if grid == GRIDS["2D_grid"]
        return "unstructured" # cells
    elseif grid == GRIDS["soil_grid"]
        return "unstructured" # cells, per soil layer
    elseif grid == GRIDS["inter_soil_grid"]
        return "unstructured" # cells, per soil layer interface (nbsoil_layers-1)
    elseif grid == GRIDS["vegetation_snow_grid"]
        return "unstructured" # (cells, nveg), per snow band
    else
        error("unknown grid type $grid")
    end
end

"""
    BMI.get_grid_rank(model::Model, grid::Int)

The number of coordinate dimensions: x and y (`get_grid_x`, `get_grid_y`), and for the
vegetation grid also z, the vegetation type (`get_grid_z`).
"""
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
    BMI.get_grid_node_count(model::Model, grid::Int)

The nodes of the unstructured grids are the centers of the active cells. On the vegetation
grid, every cell has a node per vegetation type (the z coordinate). The grids have no edges
or faces: the values of all variables are located on the nodes.
"""
function BMI.get_grid_node_count(model::Model, grid::Int)
    grid in values(GRIDS) || error("unknown grid type $grid")
    n = ncells(model.domain)
    return grid == GRIDS["vegetation_snow_grid"] ? model.config.nveg * n : n
end

function BMI.get_grid_edge_count(model::Model, grid::Int)
    grid in values(GRIDS) || error("unknown grid type $grid")
    return 0
end

function BMI.get_grid_face_count(model::Model, grid::Int)
    grid in values(GRIDS) || error("unknown grid type $grid")
    return 0
end

# The grids have no edges or faces, so the connectivity arrays are empty
function BMI.get_grid_edge_nodes(model::Model, grid::Int, edge_nodes::Vector{Int})
    grid in values(GRIDS) || error("unknown grid type $grid")
    return edge_nodes
end

function BMI.get_grid_face_edges(model::Model, grid::Int, face_edges::Vector{Int})
    grid in values(GRIDS) || error("unknown grid type $grid")
    return face_edges
end

function BMI.get_grid_face_nodes(model::Model, grid::Int, face_nodes::Vector{Int})
    grid in values(GRIDS) || error("unknown grid type $grid")
    return face_nodes
end

function BMI.get_grid_nodes_per_face(model::Model, grid::Int, nodes_per_face::Vector{Int})
    grid in values(GRIDS) || error("unknown grid type $grid")
    return nodes_per_face
end

"""
Longitude and latitude of each node of `grid`: the active cells, repeated for every
vegetation type on the vegetation grid.
"""
function node_coordinates(model::Model, grid::Int)
    grid in values(GRIDS) || error("unknown grid type $grid")
    (; domain) = model
    cells = CartesianIndices(domain.grid_size)[domain.cells]
    lon = Array(model.grid_parameters.longitude)
    lat = Array(model.grid_parameters.latitude)
    nz = BMI.get_grid_node_count(model, grid) ÷ length(cells)
    return repeat([lon[I[1]] for I in cells], nz), repeat([lat[I[2]] for I in cells], nz)
end

# RemoteBMI uses outdated BasicModelInterface.jl definition;
function BMI.get_grid_x(model::Model, grid::Int)
    x = zeros(Float64, BMI.get_grid_node_count(model, grid))
    return BMI.get_grid_x(model, grid, x)
end

function BMI.get_grid_x(model::Model, grid::Int, x::Vector{Float64})
    return copyto!(x, first(node_coordinates(model, grid)))
end

# RemoteBMI uses outdated BasicModelInterface.jl definition;
function BMI.get_grid_y(model::Model, grid::Int)
    y = zeros(Float64, BMI.get_grid_node_count(model, grid))
    return BMI.get_grid_y(model, grid, y)
end

function BMI.get_grid_y(model::Model, grid::Int, y::Vector{Float64})
    return copyto!(y, last(node_coordinates(model, grid)))
end

function BMI.get_grid_z(model::Model, grid::Int)
    if grid == GRIDS["vegetation_snow_grid"]
        z = zeros(Float64, BMI.get_grid_node_count(model, grid))
        return BMI.get_grid_z(model, grid, z)
    else
        error("accessing z-coordinate in 2-rank grid")
    end
end

"""
    BMI.get_grid_z(model::Model, grid::Int, z::Vector{Float64})

The vegetation type of each node of the vegetation grid.
"""
function BMI.get_grid_z(model::Model, grid::Int, z::Vector{Float64})
    if grid == GRIDS["vegetation_snow_grid"]
        return copyto!(z, repeat(1:model.config.nveg; inner = ncells(model.domain)))
    else
        error("No z-coordinate for grid value $grid")
    end
end

function BMI.get_grid_size(model::Model, grid::Int)
    return BMI.get_grid_node_count(model, grid)
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

Return the variable's unit.
"""
function BMI.get_var_units(model::Model, name::String)
    metadata = check_name(model, name)
    return to_string(metadata.unit)
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

"""
    BMI.get_var_location(model::Model, name::String)

All values are located on the nodes of the grid: the active cells.
"""
function BMI.get_var_location(model::Model, name::String)
    check_name(model, name)  # errors for unknown variables
    return "node"
end


#TIME
function BMI.get_current_time(model::Model)
    return datetime2unix(model.clock.time)
end

function BMI.get_start_time(model::Model)
    return datetime2unix(DateTime(model.config.start_year))
end

function BMI.get_end_time(model::Model)
    return datetime2unix(DateTime(model.config.end_year + 1))
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
    copyto!(dest, BMI.get_value_ptr(model, name))
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
            return @view model_vals[:, ind]
        else
            return @view model_vals[:, ind, :]
        end
    elseif occursin(r"_band_\d", name)
        name_2d, ind = snow_band_standard_name(name)
        model_vals, (; gridtype) = get_field_in_model(model, name_2d)
        if gridtype != GRIDS["vegetation_snow_grid"]
            return @view model_vals[:, ind]
        else
            return @view model_vals[:, ind, :]
        end
    end
    (; lens) = get_metadata(name; model)
    if isnothing(lens)
        error("Accessing '$name' is not supported")
    end
    dest = lens(model)
    if ndims(dest) > 1
        dest = vec(dest)
    end
    n = length(dest)
    return @view(dest[1:n])
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
            "Ratio between lengths: $(l_model / l_src)"
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
    if !check_if_settable(model, name)
        error("Variable $name is not settable")
    end
    dest = BMI.get_value_ptr(model, name)
    dest[inds] = src
    return nothing
end
