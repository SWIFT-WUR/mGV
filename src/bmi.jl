import BasicModelInterface as BMI

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


# <<<<<< to be done
# should return the model name/type

function BMI.get_component_name(model::Model)
    # return string(model.config.model.type)
    return nothing
end
#number of input features 
function BMI.get_input_item_count(model::Model)
    # return length(BMI.get_input_var_names(model))
    return nothing
end
#number of output features
function BMI.get_output_item_count(model::Model)
    # return length(BMI.get_output_var_names(model))
    return nothing
end


function BMI.get_input_var_names(model::Model)
    # Gets an array of names for the variables the model can use from other 
    # models implementing a BMI. The length of the array is given by 
    # get_input_item_count. The names are preferably in the form of CSDMS 
    # Standard Names. Standard Names enable a modeling framework to 
    # determine whether an input variable in one model is equivalent to,
    # or compatible with, an output variable in another model. This
    #  allows the framework to automatically connect components. 
    # Standard Names do not have to be used within the model.
    return nothing
end

function BMI.get_output_var_names(model::Model)
    return nothing
end


function BMI.get_var_grid(model::Model)
    resLat = diff(model.grid_parameters.latitude) 
    resLon = diff(model.grid_parameters.longitude)
    if all(isapprox.(resLat, resLat[1])) && all(isapprox.(resLon, resLon[1]))
        return 0 #Uniform rectilinear
    else 
        return 1 #Rectilinear
    end
    # return 2 #Structured quadrilateral
    # return 3 #Unstructured grid
    return nothing 
end

function BMI.get_var_type(model::Model, name::String)
    value = BMI.get_value_ptr(model, name)
    return repr(eltype(first(value)))
end




# Need to implement something similar to metadata in Wflow
function BMI.get_var_units(model::Model, name::String)
#     (; land) = model
#     metadata = get_metadata(name, land; model)
#     return to_string(to_SI(metadata.unit); BMI_standard = true)
    return nothing
end


function BMI.get_var_itemsize(model::Model, name::String)
    value = BMI.get_value_ptr(model, name)
    return sizeof(eltype(first(value)))
end

function BMI.get_var_nbytes(model::Model, name::String)
    return sizeof(BMI.get_value_ptr(model, name))
end

# # Need to implement something similar to metadata in Wflow
function BMI.get_var_location(model::Model, name::String)
#     (; lens) = get_metadata(name; model)
#     element_type = grid_element_type(model, lens)
#     return element_type
    return nothing
end

function BMI.get_current_time(model::Model)
    return model.clock.time
end

function BMI.get_start_time(model::Model)
    return DateTime(model.config.start_year)
end

function BMI.get_end_time(model::Model)
    return model.config.end_year
end

function BMI.get_time_units(::Model)
    return "y-m-d-h-m-s-ms"
end

function BMI.get_time_step(model::Model)
    return Second(model.config.timestep)
end

function BMI.get_value(model::Model, name::String, dest::Vector{Float64})
    dest .= copy(BMI.get_value_ptr(model, name))
    return dest
end

function BMI.get_value_ptr(model::Model, name::String)
    (; lens) = get_metadata(name; model)
    if isnothing(lens)
        error("Accessing '$name' is not supported.")
    else
        vec = lens(model)
        return @view(vec[1:n])
    end
end






# function BMI.get_value_at_indices(
#         model::Model,
#         name::String,
#         dest::Vector{Float64},
#         inds::Vector{Int},
#     )
#     dest .= BMI.get_value_ptr(model, name)[inds]
#     return dest
# end

# """
#     BMI.set_value(model::Model, name::String, src::Vector{Float64})

# Set a model variable `name` to the values in vector `src`, overwriting the current contents.
# The type and size of `src` must match the model's internal array.
# """
# function BMI.set_value(model::Model, name::String, src::Vector{Float64})
#     return BMI.get_value_ptr(model, name) .= src
# end

# """
#     BMI.set_value_at_indices(model::Model, name::String, inds::Vector{Int}, src::Vector{Float64})

#     Set a model variable `name` to the values in vector `src`, at indices `inds`.
# """
# function BMI.set_value_at_indices(
#         model::Model,
#         name::String,
#         inds::Vector{Int},
#         src::Vector{Float64},
#     )
#     return BMI.get_value_ptr(model, name)[inds] .= src
# end

# function BMI.get_grid_type(model::Model, grid::Int)
#     if grid in 0:2
#         return "points"
#     elseif grid in 3:6
#         return "unstructured"
#     else
#         error("unknown grid type $grid")
#     end
# end

# function BMI.get_grid_rank(model::Model, grid::Int)
#     if grid in 0:6
#         return 2
#     else
#         error("unknown grid type $grid")
#     end
# end

# function BMI.get_grid_x(model::Model, grid::Int, x::Vector{Float64})
#     (; reader, domain) = model
#     (; dataset) = reader
#     sel = active_indices(domain, GRIDS[grid])
#     inds = [sel[i][1] for i in eachindex(sel)]
#     x_nc = read_x_axis(dataset)
#     x .= x_nc[inds]
#     return x
# end

# function BMI.get_grid_y(model::Model, grid::Int, y::Vector{Float64})
#     (; reader, domain) = model
#     (; dataset) = reader
#     sel = active_indices(domain, GRIDS[grid])
#     inds = [sel[i][2] for i in eachindex(sel)]
#     y_nc = read_y_axis(dataset)
#     y .= y_nc[inds]
#     return y
# end

# function BMI.get_grid_node_count(model::Model, grid::Int)
#     return length(active_indices(model.domain, GRIDS[grid]))
# end

# function BMI.get_grid_size(model::Model, grid::Int)
#     return length(active_indices(model.domain, GRIDS[grid]))
# end

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

# # Extension of BMI functions (state handling and start time), required for OpenDA coupling.
# # May also be useful for other external software packages.
# function load_state(model::Model)
#     set_states!(model)
#     return nothing
# end

# function save_state(model::Model)
#     (; endstate_writer) = model.writer
#     (; output_path, output_dataset) = endstate_writer
#     if !isnothing(output_path)
#         @info "Write output states to netCDF file `$output_path`."
#     end
#     write_netcdf_timestep(model, endstate_writer)
#     close(output_dataset)
#     return nothing
# end

# function get_start_unix_time(model::Model)
#     return datetime2unix(model.config.time.starttime)
# end

# # BMI helper functions.
# """
# Return the standard name `name_layered` representing a layered soil model variable (vector
# of svectors) and the layer index `layer_index` based on a standard `name` representing a
# layer of a layered soil model variable.
# """
# function soil_layer_standard_name(name::AbstractString)
#     # Parse new naming format: soil_layer_N_water_... where N is the layer number
#     parts = split(name, "_")
#     if length(parts) >= 3 && parts[1] == "soil" && parts[2] == "layer"
#         layer_number = parts[3]
#         layer_index = tryparse(Int, layer_number)
#         if !isnothing(layer_index)
#             # Remove the layer number to get the base name
#             name_layered = join([parts[1], parts[2], parts[4:end]...], "_")
#             return name_layered, layer_index
#         end
#     end
#     # Fallback for unexpected format
#     @warn "Unable to parse layer standard name: $name"
#     return name, nothing
# end

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

