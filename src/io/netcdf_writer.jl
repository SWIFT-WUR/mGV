using Zarr
using NetCDF
using NCDatasets
using CUDA

# ============================================================================
# 1. TRANSFER BUFFER (Common)
# ============================================================================
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
        make_pinned(nx, ny, 1),    # soil_evaporation
        make_pinned(nx, ny, nlayers) # soil_moisture
    )
end

# ============================================================================
# 2. OUTPUT STORES (Polymorphic)
# ============================================================================

# --- Zarr Store ---
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

# --- NetCDF Store ---
# We wrap the NCDataset and the variables to mimic the Zarr struct structure
struct NetCDFOutputStore
    ds::NCDataset
    tsurf::Any
    tair::Any
    prec::Any
    total_et::Any
    surface_runoff::Any
    total_runoff::Any
    pe_summed::Any
    nr_summed::Any
    tr_summed::Any
    ce_summed::Any
    Q12::Any
    soil_evaporation::Any
    soil_temperature::Any
    soil_moisture::Any
end

# ============================================================================
# 3. INITIALIZATION FUNCTIONS
# ============================================================================

# --- ZARR INITIALIZATION ---
function create_output_zarr(output_path::String, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
    println("Initializing Zarr store at: $output_path")
    isdir(output_path) && rm(output_path, recursive=true)
    mkpath(output_path)

    compressor = Zarr.BloscCompressor(cname="lz4", clevel=1, shuffle=false)

    chunk_2d = (nx, ny, 1)
    chunk_3d_qlayer = (nx, ny, 1, 2)
    chunk_3d_layer = (nx, ny, 1, nlayers)
    chunk_3d_top = (nx, ny, 1, 1)

    function make_zarr(name, dims, chunks, dim_names; attrs=Dict())
        path = joinpath(output_path, name)
        arr = zcreate(Float32, dims...; path=path, chunks=chunks, compressor=compressor, fill_value=NaN32)
        arr.attrs["_ARRAY_DIMENSIONS"] = dim_names
        for (k, v) in attrs; arr.attrs[k] = v; end
        return arr
    end

    # Coords
    z_lat = make_zarr("lat", (length(lat_cpu),), (length(lat_cpu),), ["lat"]; attrs=Dict("units"=>"degrees_north", "axis"=>"Y"))
    z_lat[:] = lat_cpu
    z_lon = make_zarr("lon", (length(lon_cpu),), (length(lon_cpu),), ["lon"]; attrs=Dict("units"=>"degrees_east", "axis"=>"X"))
    z_lon[:] = lon_cpu

    store = ZarrOutputStore(
        make_zarr("tsurf_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("tair_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("precipitation_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("total_et_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("surface_runoff_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("total_runoff_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("potential_evaporation_summed_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("net_radiation_summed_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("transpiration_summed_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("canopy_evaporation_summed_output", (nx, ny, nt), chunk_2d, ["lon", "lat", "time"]),
        make_zarr("Q12_output", (nx, ny, nt, 2), chunk_3d_qlayer, ["lon", "lat", "time", "qlayers"]),
        make_zarr("soil_evaporation_output", (nx, ny, nt, 1), chunk_3d_top, ["lon", "lat", "time", "top_layer"]),
        make_zarr("soil_temperature_output", (nx, ny, nt, nlayers), chunk_3d_layer, ["lon", "lat", "time", "layer"]),
        make_zarr("soil_moisture_output", (nx, ny, nt, nlayers), chunk_3d_layer, ["lon", "lat", "time", "layer"])
    )
    
    return store, create_transfer_buffer(nx, ny, nlayers)
end

# --- NETCDF INITIALIZATION ---
function create_output_netcdf(output_file::String, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
    println("Creating NetCDF output file at: $output_file")
    out_ds = NCDataset(output_file, "c")
    
    # Dimensions
    defDim(out_ds, "lon", nx); defDim(out_ds, "lat", ny); defDim(out_ds, "time", nt)
    defDim(out_ds, "qlayers", 2); defDim(out_ds, "layer", nlayers); defDim(out_ds, "top_layer", 1)

    # Chunks
    chunk_2d = (nx, ny, 1)
    chunk_3d_qlayer = (nx, ny, 1, 2)
    chunk_3d_layer = (nx, ny, 1, nlayers)
    chunk_3d_top = (nx, ny, 1, 1)

    function def_fast_var(name, dims; chunks=nothing)
        defVar(out_ds, name, Float32, dims; chunksizes=chunks, deflatelevel=0, shuffle=false)
    end

    # Coords
    lat = defVar(out_ds, "lat", Float32, ("lat",)); lat[:] = lat_cpu; lat.attrib["axis"] = "Y"
    lon = defVar(out_ds, "lon", Float32, ("lon",)); lon[:] = lon_cpu; lon.attrib["axis"] = "X"

    # Store
    store = NetCDFOutputStore(
        out_ds,
        def_fast_var("tsurf_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("tair_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("precipitation_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("total_et_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("surface_runoff_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("total_runoff_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("potential_evaporation_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("net_radiation_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("transpiration_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("canopy_evaporation_summed_output", ("lon", "lat", "time"); chunks=chunk_2d),
        def_fast_var("Q12_output", ("lon", "lat", "time", "qlayers"); chunks=chunk_3d_qlayer),
        def_fast_var("soil_evaporation_output", ("lon", "lat", "time", "top_layer"); chunks=chunk_3d_top),
        def_fast_var("soil_temperature_output", ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer),
        def_fast_var("soil_moisture_output", ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer)
    )

    return store, create_transfer_buffer(nx, ny, nlayers)
end

# ============================================================================
# 4. DATA TRANSFER & WRITING (Dispatch based on Store Type)
# ============================================================================

# Phase 1: Async Transfer (Common to both)
function async_transfer!(processed_data, buf::TransferBuffer, stream::CuStream)
    function async_copy(dest, src)
        CUDA.stream!(stream) do
            copyto!(dest, src)
        end
    end
    # 2D
    async_copy(buf.tsurf,          processed_data.tsurf)
    async_copy(buf.tair,           processed_data.tair)
    async_copy(buf.prec,           processed_data.prec)
    async_copy(buf.total_et,       processed_data.total_et)
    async_copy(buf.surface_runoff, processed_data.surface_runoff)
    async_copy(buf.total_runoff,   processed_data.total_runoff)
    async_copy(buf.pe_summed,      processed_data.pe_summed)
    async_copy(buf.nr_summed,      processed_data.nr_summed)
    async_copy(buf.tr_summed,      processed_data.tr_summed)
    async_copy(buf.ce_summed,      processed_data.ce_summed)
    # 3D
    async_copy(buf.soil_evaporation, processed_data.soil_evaporation)
    async_copy(buf.soil_moisture,    processed_data.soil_moisture_new)
    return nothing
end

# Phase 2a: ZARR Parallel Write
function write_slice!(day, buf::TransferBuffer, stream::CuStream, store::ZarrOutputStore)
    synchronize(stream)
    Threads.@sync begin
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
        Threads.@spawn store.soil_evaporation[:, :, day, :] = buf.soil_evaporation
        Threads.@spawn store.soil_moisture[:, :, day, :]    = buf.soil_moisture
    end
end

# Phase 2b: NETCDF Serial Write
function write_slice!(day, buf::TransferBuffer, stream::CuStream, store::NetCDFOutputStore)
    synchronize(stream)
    # NetCDF writes are not thread-safe for the same dataset, so we do this serially
    store.tsurf[:, :, day]          = buf.tsurf
    store.tair[:, :, day]           = buf.tair
    store.prec[:, :, day]           = buf.prec
    store.total_et[:, :, day]       = buf.total_et
    store.surface_runoff[:, :, day] = buf.surface_runoff
    store.total_runoff[:, :, day]   = buf.total_runoff
    store.pe_summed[:, :, day]      = buf.pe_summed
    store.nr_summed[:, :, day]      = buf.nr_summed
    store.tr_summed[:, :, day]      = buf.tr_summed
    store.ce_summed[:, :, day]      = buf.ce_summed
    store.soil_evaporation[:, :, day, :] = buf.soil_evaporation
    store.soil_moisture[:, :, day, :]    = buf.soil_moisture
end

# ============================================================================
# 5. CLOSING
# ============================================================================
close_output(store::ZarrOutputStore) = nothing # No action needed for Zarr
close_output(store::NetCDFOutputStore) = close(store.ds)

# Copy the preprocess_daily_outputs function from the old file here...
function preprocess_daily_outputs(
    day, tsurf, tair_gpu, prec_gpu, 
    total_et, surface_runoff, total_runoff,
    soil_evaporation, soil_moisture_new,
    potential_evaporation, net_radiation, transpiration, canopy_evaporation,
    coverage_gpu, cv_gpu, fillvalue_threshold
)
    # ... (Keep the existing implementation)
    # [Insert content from lines 104-123 of original netcdf_writer.jl here]
    # For brevity, I assume you have this function available.
    
    san_nan = A -> begin
        T = eltype(A)
        thr = T(fillvalue_threshold)
        rep = T(NaN)
        ifelse.(isnan.(A) .| (abs.(A) .> thr), rep, A)
    end
    convcv = A -> convert.(eltype(A), cv_gpu)

    pe_processed = san_nan(potential_evaporation)
    pe_summed    = sum_with_nan_handling(convcv(pe_processed) .* pe_processed, 4)
    nr_processed = san_nan(net_radiation)
    nr_summed    = sum_with_nan_handling(convcv(nr_processed) .* nr_processed, 4)
    tr_processed = san_nan(transpiration)
    tr_gc        = tr_processed .* coverage_gpu
    tr_summed    = sum_with_nan_handling(tr_gc, 4)
    ce_processed = san_nan(canopy_evaporation)
    ce_gc        = convcv(ce_processed) .* ce_processed .* coverage_gpu
    ce_summed    = sum_with_nan_handling(ce_gc, 4)

    return (
        tsurf = tsurf, tair = tair_gpu, prec = prec_gpu,
        total_et = total_et, surface_runoff = surface_runoff, total_runoff = total_runoff,
        soil_evaporation = soil_evaporation, soil_moisture_new = soil_moisture_new,
        pe_summed = pe_summed, nr_summed = nr_summed, tr_summed = tr_summed, ce_summed = ce_summed
    )
end