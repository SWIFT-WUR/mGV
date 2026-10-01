import BasicModelInterface as BMI


const OUTvars = ["moisture",
                "temperature",
                "ice_fraction",
                "interlayer_drainage",
                "evaporation",
                "infiltration",
                "surface_runoff",
                "subsurface_runoff",
                "total_runoff",
                "surface_temperature",
                "total_evapotranspiration",
                "energy_error",
                "water_error"]



function BMI.initialize(::Type{<:Model}, config_file::AbstractString)
    return Model(config_file)
end

function BMI.update(model::Model)
    update!(model)
    return model
end

function BMI.update_until(model::Model, time::Float64)
    end_time = time - model.clock.dt
    while model.clock.time < end_time
        update!(model)
    end

end

# TODO 
# function BMI.finalize(model::Model)
#     if !isnothing(model.writer)
#         write_results(model.writer.io_service, model.clock, process_daily_outputs(model))
#     end

#     # next time step will be new year; close file output
#     if !isnothing(model.writer) && year(model.clock.time + model.clock.dt) > year(model.clock.time)
#         # close output
#         stop_async_service(model.writer.io_service)
#         close_output(model.writer.store)

#         if year(model.clock.time) == model.config.end_year
#             # if model has reached end time step set writer to nothing
#             @set model.writer = nothing
    
#         else
#             # otherwise set up new output file
#             println("Starting new io service...")
#             new_writer = start_io_service(
#                 model.config, model.grid_parameters, year(model.clock.time) + 1, model.clock.dt
#             )
#             model.writer.io_service = new_writer.io_service
#             model.writer.store = new_writer.store
#         end
#     end
# end


# INPUT, OUTPUT VARIABLE AND MODEL NAME
function BMI.get_component_name(model::Model)
    return "mGV"
end
function BMI.get_input_item_count(model::Model)
    return length(BMI.get_input_var_names(model))
end
function BMI.get_input_var_names(model::Model)
    return Vector{String}()
end

function BMI.get_output_item_count(model::Model)
    return length(BMI.get_output_var_names(model))
end
function BMI.get_output_var_names(model::Model)
    return OUTvars
end

#VAR GRID, GRID INFO
function BMI.get_var_grid(model::Model, name::String)
    resLat_min, resLat_max = extrema(diff(model.grid_parameters.latitude))
    resLon_min, resLon_max = extrema(diff(model.grid_parameters.longitude))
    if all(isapprox.(resLat_min, resLat_max)) && all(isapprox.(resLon_min, resLon_max))
        return 0 #Uniform rectilinear
    else 
        return 1 #Rectilinear
    end
    return nothing 
end

function BMI.get_grid_type(model::Model, grid::Int)
    if grid == 0
        return "rectilinear"
    else
        error("unknown grid type $grid")
    end
end

function BMI.get_grid_rank(model::Model, grid::Int)
    if grid == 0
        return 2
    else
        error("unknown grid type $grid")
    end
end

# RemoteBMI uses outdated BasicModelInterface.jl definition;
function BMI.get_grid_shape(model::Model, grid::Int)
    shape = zeros(Int, BMI.get_grid_rank(model, grid))
    return BMI.get_grid_shape(model, grid, shape)
end

function BMI.get_grid_shape(model::Model, grid::Int, shape::DenseVector{Int})
    if grid==0
        shape[1] = length(model.grid_parameters.latitude)
        shape[2] = length(model.grid_parameters.longitude) 
        return shape
    end
    error("unknown grid type $grid")
end

# RemoteBMI uses outdated BasicModelInterface.jl definition;
function BMI.get_grid_x(model::Model, grid::Int)
    x = zeros(Float64, BMI.get_grid_shape(model, grid)[2])
    return BMI.get_grid_x(model, grid, x)
end

function BMI.get_grid_x(model::Model, grid::Int, x::Vector{Float64})
    if grid == 0
        # x[:] = model.grid_parameters.longitude
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
    if grid == 0
        # y[:] = model.grid_parameters.latitude
        copyto!(y,model.grid_parameters.latitude)
    else
        error("unknown grid type $grid")
    end
    return y
end

function BMI.get_grid_z(model::Model, grid::Int, z::Vector{Float64})
    error("No grid z-coordinate")
end

# function BMI.get_grid_edge_count(model::Model, grid::Int)
#     (; domain) = model
#     if grid == 3
#         return ne(domain.river.network.graph)
#     elseif grid == 4
#         return length(domain.land.network.edge_indices.ind_x_up)
#     elseif grid == 5
#         return length(domain.land.network.edge_indices.ind_y_up)
#     elseif grid in 0:2 || grid == 6
#         @warn("edges are not provided for grid type $grid (variables are located at nodes)")
#     else
#         error("unknown grid type $grid")
#     end
# end

# function BMI.get_grid_edge_nodes(model::Model, grid::Int, edge_nodes::Vector{Int})
#     (; domain) = model
#     n = length(edge_nodes)
#     m = div(n, 2)
#     # inactive nodes (boundary/ghost points) are set at -999
#     if grid == 3
#         nodes_at_edge = adjacent_nodes_at_edge(domain.river.network.graph)
#         nodes_at_edge.dst[nodes_at_edge.dst .== m + 1] .= -999
#         edge_nodes[range(1, n; step = 2)] = nodes_at_edge.src
#         edge_nodes[range(2, n; step = 2)] = nodes_at_edge.dst
#         return edge_nodes
#     elseif grid == 4
#         ind_x_up = domain.land.network.edge_indices.ind_x_up
#         edge_nodes[range(1, n; step = 2)] = 1:m
#         ind_x_up[ind_x_up .== m + 1] .= -999
#         edge_nodes[range(2, n; step = 2)] = ind_x_up
#         return edge_nodes
#     elseif grid == 5
#         ind_y_up = domain.land.network.edge_indices.ind_y_up
#         edge_nodes[range(1, n; step = 2)] = 1:m
#         ind_y_up[ind_y_up .== m + 1] .= -999
#         edge_nodes[range(2, n; step = 2)] = ind_y_up
#         return edge_nodes
#     elseif grid in 0:2 || grid == 6
#         @warn("edges are not provided for grid type $grid (variables are located at nodes)")
#     else
#         error("unknown grid type $grid")
#     end
# end

# function BMI.get_grid_node_count(model::Model, grid::Int)
#     return length(active_indices(model.domain, GRIDS[grid]))
# end

function BMI.get_grid_size(model::Model, grid::Int)
    if grid==0
        y_size = length(model.grid_parameters.latitude)
        x_size = length(model.grid_parameters.longitude)
        return x_size*y_size
    end
    error("unknown grid type $grid")
end

# """
#     grid_element_type(model, lens::ComposedFunction)
#     grid_element_type(::T, var::PropertyLens)
#     grid_element_type(model, var::PropertyLens)

# Return the grid element type of a model variable (PropertyLens `var`) based on a `lens`. A
# `lens` allows access to a nested model variable.
# """
# function grid_element_type(
#         ::T,
#         var::PropertyLens,
#     ) where {T <: Union{RiverFlowModel{<:LocalInertial}, OverlandFlowModel{<:LocalInertial}}}
#     vars = (PropertyLens(x) for x in (:q, :q_average, :qx, :qy))
#     element_type = if var in vars
#         "edge"
#     else
#         "node"
#     end
#     return element_type
# end

# grid_element_type(model, var::PropertyLens) = "node"

# function grid_element_type(model::Model, lens::ComposedFunction)
#     lens_components = decompose(lens)
#     var = lens_components[1]
#     element_type = if PropertyLens(:river_flow) in lens_components
#         grid_element_type(model.routing.river_flow, var)
#     elseif PropertyLens(:overland_flow) in lens_components
#         grid_element_type(model.routing.overland_flow, var)
#     else
#         grid_element_type(model, var)
#     end
#     return element_type
# end






#VAR TYPE, UNITS, SIZE, NBYTES AND LOCATION
function BMI.get_var_type(model::Model, name::String)
    value = BMI.get_value_ptr(model, name)
    return repr(eltype(eltype(value)))
end

# TODO 
function BMI.get_var_units(model::Model, name::String)
#     (; land) = model
#     metadata = get_metadata(name, land; model)
#     return to_string(to_SI(metadata.unit); BMI_standard = true)
    return nothing
end

function BMI.get_var_itemsize(model::Model, name::String)
    value = BMI.get_value_ptr(model, name)
    return sizeof(eltype(eltype(value)))
end

function BMI.get_var_nbytes(model::Model, name::String)
    return sizeof(BMI.get_value_ptr(model, name))
end

# TODO 
function BMI.get_var_location(model::Model, name::String)
#     (; lens) = get_metadata(name; model)
#     element_type = grid_element_type(model, lens)
#     return element_type
    return nothing
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


#GET VALUES
function BMI.get_value(model::Model, name::String, dest)
    copyto!(dest,copy(BMI.get_value_ptr(model, name)))
    return dest
end

function BMI.get_value_ptr(model::Model, name::String)
    if occursin(r"_layer_\d",name)
        name_2d, ind = soil_layer_standard_name(name)
        model_vals, _ = get_field_in_model(model, name_2d)
        # println("Got index ", ind, " 2d name ", name_2d)
        return @view model_vals[:,:,ind]
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


#SET VALUES
"""
    BMI.set_value(model::Model, name::String, src)

Set a model variable `name` to the values in `src`, overwriting the current contents.
The type and size of `src` must match the model's internal array.
"""
function BMI.set_value(model::Model, name::String, src)
    if length(src) != length(BMI.get_value_ptr(model, name))
        error("Length mismatch between model and src varible $name")
    end
    return copyto!(BMI.get_value_ptr(model, name),src)
end

"""
    BMI.set_value_at_indices(model::Model, name::String, inds::Vector{Int}, src)

    Set a model variable `name` to the values in `src`, at indices `inds`.
"""
function BMI.set_value_at_indices(
        model::Model,
        name::String,
        inds::Vector{Int},
        src,
    )
    dest = BMI.get_value_ptr(model, name)
    dest[inds] = src
    println(typeof(dest))
    # return copyto!(dest[inds],src)
    return dest
end






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
        name_layered = join(parts[1:end-2], "_") 
        return name_layered, layer_index
    end
    # Fallback for unexpected format
    @warn "Unable to parse layer standard name: $name"
    return name, nothing
end