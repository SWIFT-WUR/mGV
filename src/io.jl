include("async_writer.jl")

"""
Create a `TransferBuffer` with pinned host memory for the `selected_outputs`, keyed by their
name in the output file.
"""
function create_transfer_buffer(nx, ny, nlayers, selected_outputs, backend::Backend = CPU())
    function make_pinned(dims...)
        A = zeros(Float32, dims...)
        pin_memory!(backend, A)
        return A
    end

    return TransferBuffer(
        meta.output_name => is_layered(meta) ? make_pinned(nx, ny, nlayers) : make_pinned(nx, ny)
            for meta in selected_outputs
    )
end

"""Whether an output variable has an additional soil layer dimension."""
is_layered(meta::ParameterMetadata) = meta.gridtype == GRIDS["soil_grid"]

"""Attributes of an output variable in the output file."""
output_attributes(meta::ParameterMetadata) =
    Dict{String, Any}("units" => to_string(meta.unit), "long_name" => meta.description)

const OutputArrays = Vector{Pair{String, Any}}

struct ZarrOutputStore
    data::OutputArrays
end

struct NetCDFOutputStore
    ds::NCDataset
    data::OutputArrays
end

close_output(store::ZarrOutputStore) = nothing # No action needed for Zarr
close_output(store::NetCDFOutputStore) = close(store.ds)


function create_output_zarr(output_path::String, year, nx, ny, nt, nlayers, lat_cpu, lon_cpu, selected_outputs)
    println("Initializing Zarr store at: $output_path")
    isdir(output_path) && rm(output_path, recursive = true)
    mkdir(output_path)

    # Initialize the group
    group = zgroup(output_path)

    # Shuffle options: shuffle = 0 = none, 1 = byte shuffle, 2 = bitshuffle, -1 = auto
    compressor = Zarr.BloscCompressor(cname = "lz4", clevel = 1, shuffle = 1)

    chunk_2d = (nx, ny, 1)
    chunk_3d_layer = (nx, ny, 1, nlayers)

    function make_zarr(name, dims, chunks, dim_names; attrs = Dict())
        # Convert input dict to Dict{String, Any}
        # This allows it to hold both Strings ("degrees_north") and Vectors (["lat", "lon"])
        full_attrs = Dict{String, Any}(attrs)
        full_attrs["_ARRAY_DIMENSIONS"] = dim_names

        # Pass the attributes dict into zcreate
        arr = zcreate(
            Float32, group, name, dims...;
            chunks = chunks,
            compressor = compressor,
            fill_value = NaN32,
            attrs = full_attrs
        )

        return arr
    end

    # Coords and time
    time_values = collect(0:(nt - 1))
    z_time = make_zarr(
        "time", (nt,), (nt,), ["time"];
        attrs = Dict("units" => "days since $year-01-01", "calendar" => "proleptic_gregorian")
    )
    z_time[:] = time_values
    z_lat = make_zarr("lat", (length(lat_cpu),), (length(lat_cpu),), ["lat"]; attrs = Dict("units" => "degrees_north", "axis" => "Y"))
    z_lat[:] = lat_cpu
    z_lon = make_zarr("lon", (length(lon_cpu),), (length(lon_cpu),), ["lon"]; attrs = Dict("units" => "degrees_east", "axis" => "X"))
    z_lon[:] = lon_cpu

    # Reversed names to match Python's Row-Major read order
    dim_2d = ["time", "lat", "lon"]
    dim_3d_layer = ["layer", "time", "lat", "lon"]

    store = ZarrOutputStore(
        OutputArrays(
            [
                meta.output_name => if is_layered(meta)
                    make_zarr(meta.output_name, (nx, ny, nt, nlayers), chunk_3d_layer, dim_3d_layer; attrs = output_attributes(meta))
                else
                    make_zarr(meta.output_name, (nx, ny, nt), chunk_2d, dim_2d; attrs = output_attributes(meta))
                end
                    for meta in selected_outputs
            ]
        )
    )

    return store
end

function create_output_netcdf(output_file::String, nx, ny, nt, nlayers, lat_cpu, lon_cpu, selected_outputs)
    println("Creating NetCDF output file at: $output_file")
    out_ds = NCDataset(output_file, "c")

    # Dimensions
    defDim(out_ds, "lon", nx); defDim(out_ds, "lat", ny); defDim(out_ds, "time", nt)
    defDim(out_ds, "qlayers", 2); defDim(out_ds, "layer", nlayers); defDim(out_ds, "top_layer", 1)

    # Chunks
    chunk_2d = (nx, ny, 1)
    chunk_3d_layer = (nx, ny, 1, nlayers)

    function def_fast_var(name, dims; chunks = nothing, attrib = Dict())
        return defVar(out_ds, name, Float32, dims; chunksizes = chunks, deflatelevel = 0, shuffle = false, attrib)
    end

    # Coords
    lat = defVar(out_ds, "lat", Float32, ("lat",)); lat[:] = lat_cpu; lat.attrib["axis"] = "Y"
    lon = defVar(out_ds, "lon", Float32, ("lon",)); lon[:] = lon_cpu; lon.attrib["axis"] = "X"

    # Store
    store = NetCDFOutputStore(
        out_ds, OutputArrays(
            [
                meta.output_name => if is_layered(meta)
                    def_fast_var(meta.output_name, ("lon", "lat", "time", "layer"); chunks = chunk_3d_layer, attrib = output_attributes(meta))
                else
                    def_fast_var(meta.output_name, ("lon", "lat", "time"); chunks = chunk_2d, attrib = output_attributes(meta))
                end
                    for meta in selected_outputs
            ]
        )
    )

    return store
end

"""
Copy the selected outputs from the `model` (on the GPU) to the CPU buffer `buf`.
"""
function async_transfer!(model, buf::TransferBuffer, selected_outputs)
    for meta in selected_outputs
        copyto!(buf[meta.output_name], meta.lens(model))
    end
    return nothing
end

function write_field!(array, buf::TransferBuffer, name, time_index)
    data = buf[name]
    if ndims(data) == 3  # Variables with an additional layer dimension
        array[:, :, time_index, :] = data
    else
        array[:, :, time_index] = data
    end
    return nothing
end

function write_slice!(time_index, buf::TransferBuffer, store::ZarrOutputStore)
    Threads.@sync for (name, array) in store.data
        Threads.@spawn write_field!(array, buf, name, time_index)
    end
    return nothing
end

function write_slice!(time_index, buf::TransferBuffer, store::NetCDFOutputStore)
    for (name, array) in store.data
        write_field!(array, buf, name, time_index)
    end
    return nothing
end

mutable struct OutputWriter
    io_service
    store
end

function start_io_service(
        config::Cfg,
        grid_parameters::GridParameters,
        year,
        dt,
        selected_outputs
    )
    nt = (DateTime(year + 1) - DateTime(year)) ÷ dt

    backend = get_backend(grid_parameters.elevation)
    grid_parameters = adapt(Array, grid_parameters)
    nx = length(grid_parameters.longitude)
    ny = length(grid_parameters.latitude)

    nlayers = 3  # hardcoded 3 soil layers

    if lowercase(config.output.format) == "netcdf"
        output_path = joinpath(config.output.dir, "$(config.output.file_prefix)$(year).nc")
        # Create a buffer pool
        output_store = create_output_netcdf(output_path, nx, ny, nt, nlayers, grid_parameters.latitude, grid_parameters.longitude, selected_outputs)
    else
        output_path = joinpath(config.output.dir, "$(config.output.file_prefix)$(year).zarr")
        output_store = create_output_zarr(output_path, year, nx, ny, nt, nlayers, grid_parameters.latitude, grid_parameters.longitude, selected_outputs)
    end

    # Start the async pool
    println("Starting Async I/O Service...")
    io_service = start_async_service(nx, ny, nlayers, output_store, selected_outputs, backend, 6)
    return OutputWriter(io_service, output_store)
end


function write_results(io_service::AsyncBufferService, clock, model)
    time_index = (clock.time - DateTime(year(clock.time))) ÷ clock.dt + 1
    # Get a free buffer from the pool
    # (Instant unless disk is >4 days behind)
    local current_buf
    current_buf = get_free_buffer(io_service)

    # Transfer GPU -> CPU (RAM copy)
    async_transfer!(model, current_buf, io_service.selected_outputs)

    # Hand off to background thread and continue simulation immediately.
    submit_buffer(io_service, time_index, current_buf)
    return nothing
end
