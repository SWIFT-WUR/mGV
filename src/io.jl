include("async_writer.jl")

function create_transfer_buffer(nx, ny, nlayers)
    function make_pinned(dims...)
        A = zeros(Float32, dims...)
        pin_memory!(A)
        return A
    end

    Results(
        make_pinned(nx, ny),  # tsurf
        make_pinned(nx, ny),  # tair
        make_pinned(nx, ny),  # prec
        make_pinned(nx, ny),  # total_et
        make_pinned(nx, ny),  # surface_runoff
        make_pinned(nx, ny),  # total_runoff
        make_pinned(nx, ny),  # discharge 
        make_pinned(nx, ny),  # travel_time
        make_pinned(nx, ny),  # pe_summed
        make_pinned(nx, ny),  # nr_summed
        make_pinned(nx, ny),  # tr_summed
        make_pinned(nx, ny),  # ce_summed
        make_pinned(nx, ny),  # ws_summed
        make_pinned(nx, ny),  # swe_summed
        make_pinned(nx, ny),  # snow_albedo
        make_pinned(nx, ny),  # snow_surf_temp
        make_pinned(nx, ny),  # snow_coverage
        make_pinned(nx, ny),  # snow_melt
        make_pinned(nx, ny),  # soil_evaporation
        make_pinned(nx, ny, nlayers)  # soil_moisture
    )
end

struct ZarrOutputStore
    data::Results
end

struct NetCDFOutputStore
    ds::NCDataset
    data::Results
end

close_output(store::ZarrOutputStore) = nothing # No action needed for Zarr
close_output(store::NetCDFOutputStore) = close(store.ds)


function create_output_zarr(output_path::String, year, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
    println("Initializing Zarr store at: $output_path")
    isdir(output_path) && rm(output_path, recursive=true)
    mkpath(output_path)

    # Initialize the group
    group = zgroup(output_path)

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

    store = ZarrOutputStore(
        Results(
            make_zarr("tsurf_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("tair_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("precipitation_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("total_et_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("surface_runoff_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("total_runoff_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("discharge_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("travel_time_output", (nx, ny, nt), chunk_2d, dim_2d; 
                    attrs=Dict("units"=>"s", "long_name"=>"River travel time")),
            make_zarr("potential_evaporation_summed_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("net_radiation_summed_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("transpiration_summed_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("canopy_evaporation_summed_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("water_storage_summed_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("swe_summed_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("snow_albedo_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("snow_surf_temp_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("snow_coverage_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("snow_melt_output", (nx, ny, nt), chunk_2d, dim_2d),
            
            # 4D Variables
            make_zarr("soil_evaporation_output", (nx, ny, nt), chunk_2d, dim_2d),
            make_zarr("soil_moisture_output", (nx, ny, nt, nlayers), chunk_3d_layer, dim_3d_layer)
        )
    )
    
    return store, create_transfer_buffer(nx, ny, nlayers)
end

function create_output_netcdf(output_file::String, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
    println("Creating NetCDF output file at: $output_file")
    mkpath(dirname(output_file))
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
    store = NetCDFOutputStore(
        out_ds,
        Results(
            def_fast_var("tsurf_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("tair_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("precipitation_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("total_et_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("surface_runoff_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("total_runoff_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("discharge_output", ("lon", "lat", "time"); chunks=chunk_2d), 
            def_fast_var("travel_time_output", ("lon", "lat", "time"); chunks=chunk_2d), 
            def_fast_var("potential_evaporation_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("net_radiation_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("transpiration_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("canopy_evaporation_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("water_storage_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("swe_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("snow_albedo_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("snow_surf_temp_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("snow_coverage_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("snow_melt_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("soil_evaporation_output", ("lon", "lat", "time"); chunks=chunk_2d),
            def_fast_var("soil_moisture_output", ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer)
        )
    )

    return store, create_transfer_buffer(nx, ny, nlayers)
end

function async_transfer!(processed_data, buf::TransferBuffer)
    # Helper to copy from GPU (processed_data fields) to CPU (buffer fields)
    # copyto! detects pinned memory and optimizes automatically on CUDA/AMDGPU
    dma!(dest, src) = copyto!(dest, src)

    dma!(buf.surface_temperature,      processed_data.surface_temperature)
    dma!(buf.air_temperature,          processed_data.air_temperature)
    dma!(buf.precipitation,            processed_data.precipitation)
    dma!(buf.total_evapotranspiration, processed_data.total_evapotranspiration)
    dma!(buf.surface_runoff,           processed_data.surface_runoff)
    dma!(buf.total_runoff,             processed_data.total_runoff)
    dma!(buf.discharge,                processed_data.discharge)
    dma!(buf.travel_time,              processed_data.travel_time)

    dma!(buf.potential_evaporation,    processed_data.potential_evaporation)
    dma!(buf.net_radiation,            processed_data.net_radiation)
    dma!(buf.transpiration,            processed_data.transpiration)
    dma!(buf.canopy_evaporation,       processed_data.canopy_evaporation)
    dma!(buf.water_storage,            processed_data.water_storage)
    dma!(buf.snow_water_equivalent,    processed_data.snow_water_equivalent)
    dma!(buf.snow_albedo,              processed_data.snow_albedo)
    dma!(buf.snow_surface_temperature, processed_data.snow_surface_temperature)
    dma!(buf.snow_coverage,            processed_data.snow_coverage)
    dma!(buf.snow_melt,                processed_data.snow_melt)
    
    dma!(buf.soil_evaporation, processed_data.soil_evaporation)
    dma!(buf.soil_moisture,    processed_data.soil_moisture)

    return nothing
end

function write_slice!(time_index, buf::TransferBuffer, store::ZarrOutputStore)
    Threads.@sync begin
        Threads.@spawn store.data.surface_temperature[:, :, time_index]      = buf.surface_temperature
        Threads.@spawn store.data.air_temperature[:, :, time_index]          = buf.air_temperature
        Threads.@spawn store.data.precipitation[:, :, time_index]            = buf.precipitation
        Threads.@spawn store.data.total_evapotranspiration[:, :, time_index] = buf.total_evapotranspiration
        Threads.@spawn store.data.surface_runoff[:, :, time_index]           = buf.surface_runoff
        Threads.@spawn store.data.total_runoff[:, :, time_index]             = buf.total_runoff
        Threads.@spawn store.data.discharge[:, :, time_index]                = buf.discharge
        Threads.@spawn store.data.travel_time[:, :, time_index]              = buf.travel_time
        Threads.@spawn store.data.potential_evaporation[:, :, time_index]    = buf.potential_evaporation
        Threads.@spawn store.data.net_radiation[:, :, time_index]            = buf.net_radiation
        Threads.@spawn store.data.transpiration[:, :, time_index]            = buf.transpiration
        Threads.@spawn store.data.canopy_evaporation[:, :, time_index]       = buf.canopy_evaporation
        Threads.@spawn store.data.water_storage[:, :, time_index]            = buf.water_storage
        Threads.@spawn store.data.snow_water_equivalent[:, :, time_index]    = buf.snow_water_equivalent
        Threads.@spawn store.data.snow_albedo[:, :, time_index]              = buf.snow_albedo
        Threads.@spawn store.data.snow_surface_temperature[:, :, time_index] = buf.snow_surface_temperature
        Threads.@spawn store.data.snow_coverage[:, :, time_index]            = buf.snow_coverage
        Threads.@spawn store.data.snow_melt[:, :, time_index]                = buf.snow_melt
        Threads.@spawn store.data.soil_evaporation[:, :, time_index]         = buf.soil_evaporation
        Threads.@spawn store.data.soil_moisture[:, :, time_index, :]         = buf.soil_moisture
    end
end

function write_slice!(time_index, buf::TransferBuffer, store::NetCDFOutputStore)
    store.data.surface_temperature[:, :, time_index]      = buf.surface_temperature
    store.data.air_temperature[:, :, time_index]          = buf.air_temperature
    store.data.precipitation[:, :, time_index]            = buf.precipitation
    store.data.total_evapotranspiration[:, :, time_index] = buf.total_evapotranspiration
    store.data.surface_runoff[:, :, time_index]           = buf.surface_runoff
    store.data.total_runoff[:, :, time_index]             = buf.total_runoff
    store.data.discharge[:, :, time_index]                = buf.discharge
    store.data.travel_time[:, :, time_index]              = buf.travel_time
    store.data.potential_evaporation[:, :, time_index]    = buf.potential_evaporation
    store.data.net_radiation[:, :, time_index]            = buf.net_radiation
    store.data.transpiration[:, :, time_index]            = buf.transpiration
    store.data.canopy_evaporation[:, :, time_index]       = buf.canopy_evaporation
    store.data.water_storage[:, :, time_index]            = buf.water_storage
    store.data.snow_water_equivalent[:, :, time_index]    = buf.snow_water_equivalent
    store.data.snow_albedo[:, :, time_index]              = buf.snow_albedo
    store.data.snow_surface_temperature[:, :, time_index] = buf.snow_surface_temperature
    store.data.snow_coverage[:, :, time_index]            = buf.snow_coverage
    store.data.snow_melt[:, :, time_index]                = buf.snow_melt
    store.data.soil_evaporation[:, :, time_index]         = buf.soil_evaporation
    store.data.soil_moisture[:, :, time_index, :]         = buf.soil_moisture
end

mutable struct OutputWriter
    io_service
    store
end

function start_io_service(
    config::Cfg,
    grid_parameters::GridParameters,
    year,
    dt
)
    nt = (DateTime(year + 1) - DateTime(year)) ÷ dt

    grid_parameters = adapt(Array, grid_parameters)
    nx = length(grid_parameters.longitude)
    ny = length(grid_parameters.latitude)

    nlayers = 3  # hardcoded 3 soil layers
    
    if lowercase(config.output.format) == "netcdf"
        output_path = joinpath(config.output.dir, "$(config.output.file_prefix)$(year).nc")
        # Create a buffer pool 
        output_store, _ = create_output_netcdf(output_path, nx, ny, nt, nlayers, grid_parameters.latitude, grid_parameters.longitude)
    else
        output_path = joinpath(config.output.dir, "$(config.output.file_prefix)$(year).zarr")
        output_store, _ = create_output_zarr(output_path, year, nx, ny, nt, nlayers, grid_parameters.latitude, grid_parameters.longitude)
    end

    # Start the async pool 
    println("Starting Async I/O Service...")
    io_service = start_async_service(nx, ny, nlayers, output_store, 6)
    return OutputWriter(io_service, output_store)
end


function write_results(io_service::AsyncBufferService, clock, results::Results)
    time_index = (clock.time - DateTime(year(clock.time))) ÷ clock.dt + 1
    # Get a free buffer from the pool 
    # (Instant unless disk is >4 days behind)
    local current_buf
    current_buf = get_free_buffer(io_service)

    # Transfer GPU -> CPU (RAM copy)
    async_transfer!(results, current_buf)

    # Hand off to background thread and continue simulation immediately.
    submit_buffer(io_service, time_index, current_buf)

end