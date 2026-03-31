# ============================================================================
# 1. TRANSFER BUFFER 
# ============================================================================
struct TransferBuffer
    tsurf::Matrix{FloatType}
    tair::Matrix{FloatType}
    prec::Matrix{FloatType}
    total_et::Matrix{FloatType}
    surface_runoff::Matrix{FloatType}
    total_runoff::Matrix{FloatType}
    discharge::Matrix{FloatType}
    travel_time::Matrix{FloatType}
    pe_summed::Matrix{FloatType}
    nr_summed::Matrix{FloatType}
    tr_summed::Matrix{FloatType}
    ce_summed::Matrix{FloatType}
    ws_summed::Matrix{FloatType}
    swe_summed::Matrix{FloatType}
    snow_albedo::Matrix{FloatType}
    snow_surf_temp::Matrix{FloatType}
    snow_coverage::Matrix{FloatType}
    snow_melt::Matrix{FloatType}
    soil_evaporation::Array{FloatType, 3}
    soil_moisture::Array{FloatType, 3}
end

function create_transfer_buffer(nx, ny, nlayers)
    function make_pinned(dims...)
        A = zeros(FloatType, dims...)
        Main.pin_memory!(A)
        return A
    end

    TransferBuffer(
        make_pinned(nx, ny),       # tsurf
        make_pinned(nx, ny),       # tair
        make_pinned(nx, ny),       # prec
        make_pinned(nx, ny),       # total_et
        make_pinned(nx, ny),       # surface_runoff
        make_pinned(nx, ny),       # total_runoff
        make_pinned(nx, ny),       # discharge 
        make_pinned(nx, ny),       # travel_time
        make_pinned(nx, ny),       # pe_summed
        make_pinned(nx, ny),       # nr_summed
        make_pinned(nx, ny),       # tr_summed
        make_pinned(nx, ny),       # ce_summed
        make_pinned(nx, ny),       # ws_summed
        make_pinned(nx, ny),       # swe_summed
        make_pinned(nx, ny),       # snow_albedo
        make_pinned(nx, ny),       # snow_surf_temp
        make_pinned(nx, ny),       # snow_coverage
        make_pinned(nx, ny),       # snow_melt
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
    discharge::A3
    travel_time::A3
    pe_summed::A3
    nr_summed::A3
    tr_summed::A3
    ce_summed::A3
    ws_summed::A3
    swe_summed::A3
    snow_albedo::A3
    snow_surf_temp::A3
    snow_coverage::A3
    snow_melt::A3
    Q12::A4
    soil_evaporation::A4
    soil_temperature::A4
    soil_moisture::A4
end

# --- NetCDF Store ---
struct NetCDFOutputStore
    ds::NCDataset
    tsurf::Any
    tair::Any
    prec::Any
    total_et::Any
    surface_runoff::Any
    total_runoff::Any
    discharge::Any
    travel_time::Any
    pe_summed::Any
    nr_summed::Any
    tr_summed::Any
    ce_summed::Any
    ws_summed::Any
    swe_summed::Any
    snow_albedo::Any
    snow_surf_temp::Any
    snow_coverage::Any
    snow_melt::Any
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

    # Initialize the group
    group = zgroup(output_path)

    compressor = Zarr.BloscCompressor(cname="lz4", clevel=1, shuffle=false)
    #compressor = Zarr.NoCompressor()

    chunk_2d = (nx, ny, 1)
    chunk_3d_qlayer = (nx, ny, 1, 2)
    chunk_3d_layer = (nx, ny, 1, nlayers)
    chunk_3d_top = (nx, ny, 1, 1)

    function make_zarr(name, dims, chunks, dim_names; attrs=Dict())
        # Convert input dict to Dict{String, Any} 
        # This allows it to hold both Strings ("degrees_north") and Vectors (["lat", "lon"])
        full_attrs = Dict{String, Any}(attrs)
        full_attrs["_ARRAY_DIMENSIONS"] = dim_names
    
        # Pass the attributes dict into zcreate
        arr = zcreate(FloatType, group, name, dims...; 
                      chunks=chunks, 
                      compressor=compressor, 
                      fill_value=NaN32,
                      attrs=full_attrs)
                      
        return arr
    end

    # Coords and time
    time_values = collect(0:nt-1) 
    z_time = make_zarr("time", (nt,), (nt,), ["time"]; 
                   attrs=Dict("units"=>"days since 1979-01-01", "calendar"=>"proleptic_gregorian"))
    z_time[:] = time_values
    z_lat = make_zarr("lat", (length(lat_cpu),), (length(lat_cpu),), ["lat"]; attrs=Dict("units"=>"degrees_north", "axis"=>"Y"))
    z_lat[:] = lat_cpu
    z_lon = make_zarr("lon", (length(lon_cpu),), (length(lon_cpu),), ["lon"]; attrs=Dict("units"=>"degrees_east", "axis"=>"X"))
    z_lon[:] = lon_cpu

    # Reversed names to match Python's Row-Major read order
    dim_2d = ["time", "lat", "lon"] 
    dim_3d_qlayer = ["qlayers", "time", "lat", "lon"] 
    dim_3d_top    = ["top_layer", "time", "lat", "lon"]
    dim_3d_layer  = ["layer", "time", "lat", "lon"]

    store = ZarrOutputStore(
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
        make_zarr("Q12_output", (nx, ny, nt, 2), chunk_3d_qlayer, dim_3d_qlayer),
        make_zarr("soil_evaporation_output", (nx, ny, nt, 1), chunk_3d_top, dim_3d_top),
        make_zarr("soil_temperature_output", (nx, ny, nt, nlayers), chunk_3d_layer, dim_3d_layer),
        make_zarr("soil_moisture_output", (nx, ny, nt, nlayers), chunk_3d_layer, dim_3d_layer)
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
        defVar(out_ds, name, FloatType, dims; chunksizes=chunks, deflatelevel=0, shuffle=false)
    end

    # Coords
    lat = defVar(out_ds, "lat", FloatType, ("lat",)); lat[:] = lat_cpu; lat.attrib["axis"] = "Y"
    lon = defVar(out_ds, "lon", FloatType, ("lon",)); lon[:] = lon_cpu; lon.attrib["axis"] = "X"

    # Store
    store = NetCDFOutputStore(
        out_ds,
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
        def_fast_var("Q12_output", ("lon", "lat", "time", "qlayers"); chunks=chunk_3d_qlayer),
        def_fast_var("soil_evaporation_output", ("lon", "lat", "time", "top_layer"); chunks=chunk_3d_top),
        def_fast_var("soil_temperature_output", ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer),
        def_fast_var("soil_moisture_output", ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer)
    )

    return store, create_transfer_buffer(nx, ny, nlayers)
end

# ============================================================================
# 4. DATA TRANSFER & WRITING (Dispatch based on store type)
# ============================================================================

# Phase 1: Transfer to buffer
function async_transfer!(processed_data, buf::TransferBuffer)
    
    # Helper to copy from GPU (processed_data fields) to CPU (buffer fields)
    # copyto! detects pinned memory and optimizes automatically on CUDA/AMDGPU
    dma!(dest, src) = copyto!(dest, src)

    dma!(buf.tsurf,          processed_data.tsurf)
    dma!(buf.tair,           processed_data.tair)
    dma!(buf.prec,           processed_data.prec)
    dma!(buf.total_et,       processed_data.total_et)
    dma!(buf.surface_runoff, processed_data.surface_runoff)
    dma!(buf.total_runoff,   processed_data.total_runoff)
    dma!(buf.discharge,      processed_data.discharge)
    dma!(buf.travel_time,    processed_data.travel_time)

    dma!(buf.pe_summed,      processed_data.pe_summed)
    dma!(buf.nr_summed,      processed_data.nr_summed)
    dma!(buf.tr_summed,      processed_data.tr_summed)
    dma!(buf.ce_summed,      processed_data.ce_summed)
    dma!(buf.ws_summed,      processed_data.ws_summed)
    dma!(buf.swe_summed,     processed_data.swe_summed)
    dma!(buf.snow_albedo,    processed_data.snow_albedo)
    dma!(buf.snow_surf_temp, processed_data.snow_surf_temp)
    dma!(buf.snow_coverage,  processed_data.snow_coverage)
    dma!(buf.snow_melt,      processed_data.snow_melt)
    
    dma!(buf.soil_evaporation, processed_data.soil_evaporation)
    dma!(buf.soil_moisture,    processed_data.soil_moisture)

    return nothing
end

# Phase 2a: ZARR Parallel write
function write_slice!(day, buf::TransferBuffer, store::ZarrOutputStore)

    Threads.@sync begin
        Threads.@spawn store.tsurf[:, :, day]          = buf.tsurf
        Threads.@spawn store.tair[:, :, day]           = buf.tair
        Threads.@spawn store.prec[:, :, day]           = buf.prec
        Threads.@spawn store.total_et[:, :, day]       = buf.total_et
        Threads.@spawn store.surface_runoff[:, :, day] = buf.surface_runoff
        Threads.@spawn store.total_runoff[:, :, day]   = buf.total_runoff
        Threads.@spawn store.discharge[:, :, day]      = buf.discharge
        Threads.@spawn store.travel_time[:, :, day]    = buf.travel_time
        Threads.@spawn store.pe_summed[:, :, day]      = buf.pe_summed
        Threads.@spawn store.nr_summed[:, :, day]      = buf.nr_summed
        Threads.@spawn store.tr_summed[:, :, day]      = buf.tr_summed
        Threads.@spawn store.ce_summed[:, :, day]      = buf.ce_summed
        Threads.@spawn store.ws_summed[:, :, day]      = buf.ws_summed
        Threads.@spawn store.swe_summed[:, :, day]     = buf.swe_summed
        Threads.@spawn store.snow_albedo[:, :, day]    = buf.snow_albedo
        Threads.@spawn store.snow_surf_temp[:, :, day] = buf.snow_surf_temp
        Threads.@spawn store.snow_coverage[:, :, day]  = buf.snow_coverage
        Threads.@spawn store.snow_melt[:, :, day]      = buf.snow_melt
        Threads.@spawn store.soil_evaporation[:, :, day, :] = buf.soil_evaporation
        Threads.@spawn store.soil_moisture[:, :, day, :]    = buf.soil_moisture
    end
end

# Phase 2b: NETCDF Serial write
function write_slice!(day, buf::TransferBuffer, store::NetCDFOutputStore)
    
    store.tsurf[:, :, day]          = buf.tsurf
    store.tair[:, :, day]           = buf.tair
    store.prec[:, :, day]           = buf.prec
    store.total_et[:, :, day]       = buf.total_et
    store.surface_runoff[:, :, day] = buf.surface_runoff
    store.total_runoff[:, :, day]   = buf.total_runoff
    store.discharge[:, :, day]      = buf.discharge
    store.travel_time[:, :, day]    = buf.travel_time
    store.pe_summed[:, :, day]      = buf.pe_summed
    store.nr_summed[:, :, day]      = buf.nr_summed
    store.tr_summed[:, :, day]      = buf.tr_summed
    store.ce_summed[:, :, day]      = buf.ce_summed
    store.ws_summed[:, :, day]      = buf.ws_summed
    store.swe_summed[:, :, day]     = buf.swe_summed
    store.snow_albedo[:, :, day]    = buf.snow_albedo
    store.snow_surf_temp[:, :, day] = buf.snow_surf_temp
    store.snow_coverage[:, :, day]  = buf.snow_coverage
    store.snow_melt[:, :, day]      = buf.snow_melt
    store.soil_evaporation[:, :, day, :] = buf.soil_evaporation
    store.soil_moisture[:, :, day, :]    = buf.soil_moisture
end

# ============================================================================
# 5. CLOSING
# ============================================================================
close_output(store::ZarrOutputStore) = nothing # No action needed for Zarr
close_output(store::NetCDFOutputStore) = close(store.ds)


# ============================================================================
# 6. OPTIMIZED PREPROCESSING KERNELS
# ============================================================================

@kernel function fused_preprocess_kernel!(
    pe_out, nr_out, tr_out, ce_out, ws_out, # 2D Outputs
    @Const(pe_in), @Const(nr_in),       # 4D Inputs
    @Const(tr_in), @Const(ce_in), @Const(ws_in), # 4D Inputs
    @Const(coverage), @Const(cv),       # 4D Weights
    threshold, fill_val                 # Scalars
)
    i, j = @index(Global, NTuple)

    # 1. Initialize Accumulators (sums for this grid cell)
    # We use the type of the output array to ensure stability
    acc_pe = zero(eltype(pe_out))
    acc_nr = zero(eltype(nr_out))
    acc_tr = zero(eltype(tr_out))
    acc_ce = zero(eltype(ce_out))
    acc_ws = zero(eltype(ws_out))

    # 2. Iterate over Vegetation Tiles (4th Dimension)
    # We assume inputs are (nx, ny, 1, n_tiles)
    n_tiles = size(pe_in, 4)
    total_cv = zero(eltype(pe_out))

    for k in 1:n_tiles
        # A. Shared Weights
        _cv_raw = cv[i, j, 1, k]
        w_cv = isnan(_cv_raw) ? zero(eltype(pe_out)) : eltype(pe_out)(_cv_raw)
        
        _cov_raw = coverage[i, j, 1, k]
        w_cov = isnan(_cov_raw) ? zero(eltype(pe_out)) : eltype(pe_out)(_cov_raw)
        
        total_cv += w_cv

        # B. Potential Evaporation (PE)
        val = pe_in[i, j, 1, k]
        if !isnan(val) && abs(val) <= threshold
            acc_pe += w_cv * val
        end

        # C. Net Radiation (NR)
        val = nr_in[i, j, 1, k]
        if !isnan(val) && abs(val) <= threshold
            acc_nr += w_cv * val
        end

        # D. Transpiration (TR)
        val = tr_in[i, j, 1, k]
        if !isnan(val) && abs(val) <= threshold
            acc_tr += w_cov * val
        end

        # E. Canopy Evaporation (CE)
        val = ce_in[i, j, 1, k]
        if !isnan(val) && abs(val) <= threshold
            acc_ce += w_cv * w_cov * val
        end

        # F. Water Storage (WS)
        val = ws_in[i, j, 1, k]
        if !isnan(val) && abs(val) <= threshold
            acc_ws += w_cv * w_cov * val
        end
    end

    # 3. Write Final Sums or NaN to Global Memory
    if isnan(total_cv) || total_cv < eltype(pe_out)(1e-6)
        pe_out[i, j] = fill_val
        nr_out[i, j] = fill_val
        tr_out[i, j] = fill_val
        ce_out[i, j] = fill_val
        ws_out[i, j] = fill_val
    else
        pe_out[i, j] = acc_pe
        nr_out[i, j] = acc_nr
        tr_out[i, j] = acc_tr
        ce_out[i, j] = acc_ce
        ws_out[i, j] = acc_ws
    end
end

function preprocess_daily_outputs(
    day, tsurf, tair_gpu, prec_gpu, 
    total_et, surface_runoff, total_runoff,
    soil_evaporation, soil_moisture,
    potential_evaporation, net_radiation, transpiration, canopy_evaporation, water_storage,
    coverage_gpu, cv_gpu, fillvalue_threshold,
    swe_gpu, snow_albedo_gpu, snow_surf_temp_gpu, snow_coverage_gpu, snow_melt_gpu
)
    # 1. Allocate Output Arrays (2D)
    nx, ny = size(tsurf)[1:2]
    
    pe_summed = similar(tsurf, nx, ny)
    nr_summed = similar(tsurf, nx, ny)
    tr_summed = similar(tsurf, nx, ny)
    ce_summed = similar(tsurf, nx, ny)
    ws_summed = similar(tsurf, nx, ny)

    # 2. Launch the Fused Kernel
    kernel_launcher! = fused_preprocess_kernel!(device_backend)
    
    # Launch with 2D range
    kernel_launcher!(
        pe_summed, nr_summed, tr_summed, ce_summed, ws_summed,
        potential_evaporation, net_radiation, transpiration, canopy_evaporation, water_storage,
        coverage_gpu, cv_gpu,
        FloatType(fillvalue_threshold), FloatType(NaN);
        ndrange=(nx, ny)
    )
    
    # 3. Handle reshapes (metadata only, instant)
    if Main.enable_routing
        discharge_2d = reshape(Main.routing_state.discharge_gpu, size(total_runoff))
        travel_time_2d = reshape(Main.routing_state.travel_time_gpu, size(total_runoff)) 
    else
        discharge_2d = similar(total_runoff)
        travel_time_2d = similar(total_runoff)
        fill!(discharge_2d, FloatType(0.0))
        fill!(travel_time_2d, FloatType(0.0))
    end 

    # 4. Package and return
    return (
        # Direct 2D
        tsurf          = tsurf,
        tair           = tair_gpu,
        prec           = prec_gpu,
        total_et       = total_et,
        surface_runoff = surface_runoff,
        total_runoff   = total_runoff,
        discharge      = discharge_2d,
        travel_time    = travel_time_2d,

        # Direct 3D
        soil_evaporation  = soil_evaporation,
        soil_moisture     = soil_moisture,

        # Calculate sums
        pe_summed = pe_summed,
        nr_summed = nr_summed,
        tr_summed = tr_summed,
        ce_summed = ce_summed,
        ws_summed = ws_summed,
        
        # Direct Snow Overrides
        swe_summed = swe_gpu,
        snow_albedo = snow_albedo_gpu,
        snow_surf_temp = snow_surf_temp_gpu,
        snow_coverage = snow_coverage_gpu,
        snow_melt = snow_melt_gpu
    )
end