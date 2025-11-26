using Zarr
using CUDA

# ============================================================================
# 1. TRANSFER BUFFER
# ============================================================================
# Matches the original struct exactly to ensure main.jl compatibility
struct TransferBuffer
    tsurf::Matrix{Float32}
    tair::Matrix{Float32}
    prec::Matrix{Float32}
    total_et::Matrix{Float32}
    surface_runoff::Matrix{Float32}
    total_runoff::Matrix{Float32}
    pe_summed::Matrix{Float32}
    nr_summed::Matrix{Float32}
    tr_summed::Matrix{Float32}
    ce_summed::Matrix{Float32}
    soil_evaporation::Array{Float32, 3}
    soil_moisture::Array{Float32, 3}
end

function create_transfer_buffer(nx, ny, nlayers)
    function make_pinned(dims...)
        A = zeros(Float32, dims...)
        CUDA.Mem.pin(A)
        return A
    end

    TransferBuffer(
        make_pinned(nx, ny),       # tsurf
        make_pinned(nx, ny),       # tair
        make_pinned(nx, ny),       # prec
        make_pinned(nx, ny),       # total_et
        make_pinned(nx, ny),       # surface_runoff
        make_pinned(nx, ny),       # total_runoff
        make_pinned(nx, ny),       # pe_summed
        make_pinned(nx, ny),       # nr_summed
        make_pinned(nx, ny),       # tr_summed
        make_pinned(nx, ny),       # ce_summed
        make_pinned(nx, ny, 1),    # soil_evaporation (top layer)
        make_pinned(nx, ny, nlayers) # soil_moisture
    )
end

# ============================================================================
# 2. ZARR STORE STRUCT (OPTIMIZED)
# ============================================================================
# A3 = Type for 3D Arrays (2D + Time)
# A4 = Type for 4D Arrays (3D + Time)
# This strict typing prevents memory allocations during writing.
struct ZarrOutputStore{A3, A4}
    tsurf::A3
    tair::A3
    prec::A3
    total_et::A3
    surface_runoff::A3
    total_runoff::A3
    pe_summed::A3
    nr_summed::A3
    tr_summed::A3
    ce_summed::A3
    
    Q12::A4
    soil_evaporation::A4
    soil_temperature::A4
    soil_moisture::A4
end

# ============================================================================
# 3. INITIALIZE ZARR
# ============================================================================
function create_output_zarr(output_path::String, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
    println("Initializing Zarr store at: $output_path")
    
    isdir(output_path) && rm(output_path, recursive=true)
    mkpath(output_path)

    # Use LZ4 for maximum speed (low CPU, decent compression)
    compressor = Zarr.BloscCompressor(cname="lz4", clevel=1, shuffle=false)

    chunk_2d = (nx, ny, 1)
    chunk_3d_qlayer = (nx, ny, 1, 2)
    chunk_3d_layer = (nx, ny, 1, 3)
    chunk_3d_top = (nx, ny, 1, 1)

    # Helper to create arrays with attributes
    function make_zarr(name, dims, chunks, dim_names; attrs=Dict())
        path = joinpath(output_path, name)
        arr = zcreate(Float32, dims...; path=path, chunks=chunks, compressor=compressor, fill_value=NaN32)
        arr.attrs["_ARRAY_DIMENSIONS"] = dim_names
        for (k, v) in attrs; arr.attrs[k] = v; end
        return arr
    end

    # Create Coordinate Arrays
    z_lat = zcreate(Float32, length(lat_cpu), path=joinpath(output_path, "lat"), compressor=compressor)
    z_lat[:] = lat_cpu
    z_lat.attrs["_ARRAY_DIMENSIONS"] = ["lat"]
    z_lat.attrs["axis"] = "Y"; z_lat.attrs["units"] = "degrees_north"

    z_lon = zcreate(Float32, length(lon_cpu), path=joinpath(output_path, "lon"), compressor=compressor)
    z_lon[:] = lon_cpu
    z_lon.attrs["_ARRAY_DIMENSIONS"] = ["lon"]
    z_lon.attrs["axis"] = "X"; z_lon.attrs["units"] = "degrees_east"

    # Create the Store
    # Julia will automatically detect the types A3 and A4 from these calls
    store = ZarrOutputStore(
        make_zarr("tsurf_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"°C")),
        make_zarr("tair_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"°C")),
        make_zarr("precipitation_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"mm/day")),
        make_zarr("total_et_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"mm")),
        make_zarr("surface_runoff_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"mm")),
        make_zarr("total_runoff_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"mm")),
        make_zarr("potential_evaporation_summed_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"mm")),
        make_zarr("net_radiation_summed_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"W/m^2")),
        make_zarr("transpiration_summed_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"mm")),
        make_zarr("canopy_evaporation_summed_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]; attrs=Dict("units"=>"mm")),
        
        # 4D Variables
        make_zarr("Q12_output", (nx, ny, nt, 2), chunk_3d_qlayer, ["lon", "lat", "time", "qlayers"]; attrs=Dict("units"=>"mm")),
        make_zarr("soil_evaporation_output", (nx, ny, nt, 1), chunk_3d_top, ["lon", "lat", "time", "top_layer"]; attrs=Dict("units"=>"mm")),
        make_zarr("soil_temperature_output", (nx, ny, nt, 3), chunk_3d_layer, ["lon", "lat", "time", "layer"]; attrs=Dict("units"=>"°C")),
        make_zarr("soil_moisture_output", (nx, ny, nt, 3), chunk_3d_layer, ["lon", "lat", "time", "layer"]; attrs=Dict("units"=>"kg/m^3"))
    )
    
    transfer_buf = create_transfer_buffer(nx, ny, 3)
    return store, transfer_buf
end

# ============================================================================
# 4. ASYNC TRANSFER (Phase 1)
# ============================================================================
# Moves data from GPU arrays to Pinned CPU Memory (Buffer)
# Returns immediately.
function async_transfer!(
    processed_data, 
    buf::TransferBuffer, 
    stream::CuStream
)
    function async_copy(dest, src)
        CUDA.unsafe_copyto!(pointer(dest), pointer(src), length(dest), stream=stream)
    end

    # --- 2D Raw ---
    async_copy(buf.tsurf,          processed_data.tsurf)
    async_copy(buf.tair,           processed_data.tair)
    async_copy(buf.prec,           processed_data.prec)
    async_copy(buf.total_et,       processed_data.total_et)
    async_copy(buf.surface_runoff, processed_data.surface_runoff)
    async_copy(buf.total_runoff,   processed_data.total_runoff)

    # --- 2D Processed Sums ---
    async_copy(buf.pe_summed,      processed_data.pe_summed)
    async_copy(buf.nr_summed,      processed_data.nr_summed)
    async_copy(buf.tr_summed,      processed_data.tr_summed)
    async_copy(buf.ce_summed,      processed_data.ce_summed)

    # --- 3D Layers ---
    async_copy(buf.soil_evaporation, processed_data.soil_evaporation)
    async_copy(buf.soil_moisture,    processed_data.soil_moisture_new)
    
    return nothing
end

# ============================================================================
# 5. WRITE SLICE (Phase 2)
# ============================================================================
# Synchronizes stream and writes from Pinned Buffer to Zarr Store
function write_zarr_slice!(
    day,
    buf::TransferBuffer,
    stream::CuStream,
    store::ZarrOutputStore
)
    # 1. Wait for GPU to finish copying to the buffer
    synchronize(stream)


    # 2. Parallel Write to Disk
    # 'Threads.@sync' ensures we wait for ALL variables to finish writing 
    # before we return. This prevents the buffer from being overwritten early.
    Threads.@sync begin
        
        # 2D Writes
        Threads.@spawn store.tsurf[:, :, day]          = buf.tsurf
        Threads.@spawn store.tair[:, :, day]           = buf.tair
        Threads.@spawn store.prec[:, :, day]           = buf.prec
        Threads.@spawn store.total_et[:, :, day]       = buf.total_et
        Threads.@spawn store.surface_runoff[:, :, day] = buf.surface_runoff
        Threads.@spawn store.total_runoff[:, :, day]   = buf.total_runoff
        Threads.@spawn store.pe_summed[:, :, day]      = buf.pe_summed
        Threads.@spawn store.nr_summed[:, :, day]      = buf.nr_summed
        Threads.@spawn store.tr_summed[:, :, day]      = buf.tr_summed
        Threads.@spawn store.ce_summed[:, :, day]      = buf.ce_summed

        # 4D Writes (Layered vars)
        Threads.@spawn store.soil_evaporation[:, :, day, :] = buf.soil_evaporation
        Threads.@spawn store.soil_moisture[:, :, day, :]    = buf.soil_moisture
    end

    # NOTE: Q12 and SoilTemperature are in the store, but your original
    # logic did not copy them to the buffer, so we leave them as NaN/Empty.
    
    return nothing
end

# ============================================================================
# HELPER: PRE-PROCESS DAILY OUTPUTS
# ============================================================================
function preprocess_daily_outputs(
    day, tsurf, tair_gpu, prec_gpu, 
    total_et, surface_runoff, total_runoff,
    soil_evaporation, soil_moisture_new,
    potential_evaporation, net_radiation, transpiration, canopy_evaporation,
    coverage_gpu, cv_gpu, fillvalue_threshold
)
    # --- Helper Functions ---
    san_nan = A -> begin
        T = eltype(A)
        thr = T(fillvalue_threshold)
        rep = T(NaN)
        ifelse.(isnan.(A) .| (abs.(A) .> thr), rep, A)
    end
    
    # Helper to broadcast convert cv_gpu to match input array type
    convcv = A -> convert.(eltype(A), cv_gpu)

    # --- Calculations ---
    
    # 1. Potential Evaporation
    pe_processed = san_nan(potential_evaporation)
    pe_summed    = sum_with_nan_handling(convcv(pe_processed) .* pe_processed, 4)

    # 2. Net Radiation
    nr_processed = san_nan(net_radiation)
    nr_summed    = sum_with_nan_handling(convcv(nr_processed) .* nr_processed, 4)

    # 3. Transpiration
    tr_processed = san_nan(transpiration)
    tr_gc        = tr_processed .* coverage_gpu
    tr_summed    = sum_with_nan_handling(tr_gc, 4)

    # 4. Canopy Evaporation
    ce_processed = san_nan(canopy_evaporation)
    ce_gc        = convcv(ce_processed) .* ce_processed .* coverage_gpu
    ce_summed    = sum_with_nan_handling(ce_gc, 4)

    # --- Packaging ---
    return (
        # Direct 2D
        tsurf          = tsurf,
        tair           = tair_gpu,
        prec           = prec_gpu,
        total_et       = total_et,
        surface_runoff = surface_runoff,
        total_runoff   = total_runoff,
        
        # Direct 3D (Layers)
        soil_evaporation  = soil_evaporation,
        soil_moisture_new = soil_moisture_new,

        # Calculated Sums
        pe_summed = pe_summed,
        nr_summed = nr_summed,
        tr_summed = tr_summed,
        ce_summed = ce_summed
    )
end