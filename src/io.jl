include("async_writer.jl")

"""
One pool buffer: pinned host memory for the selected `fields`, and a 1-element
placeholder for the rest, which nothing reads or writes.
"""
function create_transfer_buffer(nx, ny, nlayers, fields)
    function make_pinned(dims...)
        A = zeros(Float32, dims...)
        pin_memory!(A)
        return A
    end

    slice(field) = field in fields ? make_pinned(nx, ny) : zeros(Float32, 1, 1)
    layered(field) = field in fields ? make_pinned(nx, ny, nlayers) : zeros(Float32, 1, 1, 1)

    Results(
        slice(:surface_temperature),
        slice(:air_temperature),
        slice(:precipitation),
        slice(:total_evapotranspiration),
        slice(:surface_runoff),
        slice(:total_runoff),
        slice(:discharge),
        slice(:travel_time),
        slice(:potential_evaporation),
        slice(:net_radiation),
        slice(:transpiration),
        slice(:canopy_evaporation),
        slice(:water_storage),
        slice(:snow_water_equivalent),
        slice(:snow_albedo),
        slice(:snow_surface_temperature),
        slice(:snow_coverage),
        slice(:snow_melt),
        slice(:soil_evaporation),
        layered(:soil_moisture),
    )
end

# The selected `Results` fields, each with the array it is written to in the
# output file, in `OUTPUT_VARIABLES` order.
const OutputArrays = Vector{Pair{Symbol, Any}}

struct ZarrOutputStore
    data::OutputArrays
end

struct NetCDFOutputStore
    ds::NCDataset
    data::OutputArrays
end

close_output(store::ZarrOutputStore) = nothing # No action needed for Zarr
close_output(store::NetCDFOutputStore) = close(store.ds)


function create_output_zarr(output_path::String, year, nx, ny, nt, nlayers, lat_cpu, lon_cpu, fields)
    println("Initializing Zarr store at: $output_path")
    isdir(output_path) && rm(output_path, recursive=true)
    mkdir(output_path)

    # Initialize the group
    group = zgroup(output_path)

    # Shuffle options: shuffle = 0 = none, 1 = byte shuffle, 2 = bitshuffle, -1 = auto
    compressor = Zarr.BloscCompressor(cname="lz4", clevel=1, shuffle=1)

    chunk_2d = (nx, ny, 1)
    chunk_3d_layer = (nx, ny, 1, nlayers)

    function make_zarr(name, dims, chunks, dim_names; attrs=Dict())
        # Convert input dict to Dict{String, Any} 
        # This allows it to hold both Strings ("degrees_north") and Vectors (["lat", "lon"])
        full_attrs = Dict{String, Any}(attrs)
        full_attrs["_ARRAY_DIMENSIONS"] = dim_names
    
        # Pass the attributes dict into zcreate
        arr = zcreate(Float32, group, name, dims...; 
                      chunks=chunks, 
                      compressor=compressor, 
                      fill_value=NaN32,
                      attrs=full_attrs)
                      
        return arr
    end

    # Coords and time
    time_values = collect(0:nt-1) 
    z_time = make_zarr("time", (nt,), (nt,), ["time"]; 
                   attrs=Dict("units"=>"days since $year-01-01", "calendar"=>"proleptic_gregorian"))
    z_time[:] = time_values
    z_lat = make_zarr("lat", (length(lat_cpu),), (length(lat_cpu),), ["lat"]; attrs=Dict("units"=>"degrees_north", "axis"=>"Y"))
    z_lat[:] = lat_cpu
    z_lon = make_zarr("lon", (length(lon_cpu),), (length(lon_cpu),), ["lon"]; attrs=Dict("units"=>"degrees_east", "axis"=>"X"))
    z_lon[:] = lon_cpu

    # Reversed names to match Python's Row-Major read order
    dim_2d = ["time", "lat", "lon"] 
    dim_3d_layer  = ["layer", "time", "lat", "lon"]

    attrs = Dict(:travel_time => Dict("units" => "s", "long_name" => "River travel time"))
    store = ZarrOutputStore(OutputArrays([
        field => if field == :soil_moisture
            make_zarr(output_name(field), (nx, ny, nt, nlayers), chunk_3d_layer, dim_3d_layer)
        else
            make_zarr(output_name(field), (nx, ny, nt), chunk_2d, dim_2d; attrs=get(attrs, field, Dict()))
        end
        for field in fields
    ]))

    return store
end

function create_output_netcdf(output_file::String, nx, ny, nt, nlayers, lat_cpu, lon_cpu, fields)
    println("Creating NetCDF output file at: $output_file")
    out_ds = NCDataset(output_file, "c")
    
    # Dimensions
    defDim(out_ds, "lon", nx); defDim(out_ds, "lat", ny); defDim(out_ds, "time", nt)
    defDim(out_ds, "qlayers", 2); defDim(out_ds, "layer", nlayers); defDim(out_ds, "top_layer", 1)

    # Chunks
    chunk_2d = (nx, ny, 1)
    chunk_3d_layer = (nx, ny, 1, nlayers)

    function def_fast_var(name, dims; chunks=nothing)
        defVar(out_ds, name, Float32, dims; chunksizes=chunks, deflatelevel=0, shuffle=false)
    end

    # Coords
    lat = defVar(out_ds, "lat", Float32, ("lat",)); lat[:] = lat_cpu; lat.attrib["axis"] = "Y"
    lon = defVar(out_ds, "lon", Float32, ("lon",)); lon[:] = lon_cpu; lon.attrib["axis"] = "X"

    # Store
    store = NetCDFOutputStore(out_ds, OutputArrays([
        field => if field == :soil_moisture
            def_fast_var(output_name(field), ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer)
        else
            def_fast_var(output_name(field), ("lon", "lat", "time"); chunks=chunk_2d)
        end
        for field in fields
    ]))

    return store
end

function async_transfer!(processed_data, buf::TransferBuffer, fields)
    # Copy the selected fields from GPU (processed_data) to CPU (buffer).
    # copyto! detects pinned memory and optimizes automatically on CUDA/AMDGPU
    for field in fields
        copyto!(getfield(buf, field), getfield(processed_data, field))
    end
    return nothing
end

"""Write one field of `buf` into time slice `time_index` of its output array."""
function write_field!(array, buf::TransferBuffer, field, time_index)
    if field == :soil_moisture
        array[:, :, time_index, :] = buf.soil_moisture
    else
        array[:, :, time_index] = getfield(buf, field)
    end
    return nothing
end

function write_slice!(time_index, buf::TransferBuffer, store::ZarrOutputStore)
    Threads.@sync for (field, array) in store.data
        Threads.@spawn write_field!(array, buf, field, time_index)
    end
end

function write_slice!(time_index, buf::TransferBuffer, store::NetCDFOutputStore)
    for (field, array) in store.data
        write_field!(array, buf, field, time_index)
    end
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
    fields
)
    nt = (DateTime(year + 1) - DateTime(year)) ÷ dt

    grid_parameters = adapt(Array, grid_parameters)
    nx = length(grid_parameters.longitude)
    ny = length(grid_parameters.latitude)

    nlayers = 3  # hardcoded 3 soil layers
    
    if lowercase(config.output.format) == "netcdf"
        output_path = joinpath(config.output.dir, "$(config.output.file_prefix)$(year).nc")
        # Create a buffer pool 
        output_store = create_output_netcdf(output_path, nx, ny, nt, nlayers, grid_parameters.latitude, grid_parameters.longitude, fields)
    else
        output_path = joinpath(config.output.dir, "$(config.output.file_prefix)$(year).zarr")
        output_store = create_output_zarr(output_path, year, nx, ny, nt, nlayers, grid_parameters.latitude, grid_parameters.longitude, fields)
    end

    # Start the async pool 
    println("Starting Async I/O Service...")
    io_service = start_async_service(nx, ny, nlayers, output_store, fields, 6)
    return OutputWriter(io_service, output_store)
end


function write_results(io_service::AsyncBufferService, clock, results::Results)
    time_index = (clock.time - DateTime(year(clock.time))) ÷ clock.dt + 1
    # Get a free buffer from the pool 
    # (Instant unless disk is >4 days behind)
    local current_buf
    current_buf = get_free_buffer(io_service)

    # Transfer GPU -> CPU (RAM copy)
    async_transfer!(results, current_buf, io_service.fields)

    # Hand off to background thread and continue simulation immediately.
    submit_buffer(io_service, time_index, current_buf)

end