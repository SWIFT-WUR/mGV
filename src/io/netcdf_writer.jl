# ============================================================================
# 1. TRANSFER BUFFER STRUCT
# ============================================================================

struct TransferBuffer
    # --- 2D Variables ---
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

    # --- 3D Variables ---
    soil_evaporation::Array{Float32, 3} # (lon, lat, top_layer)
    soil_moisture::Array{Float32, 3}    # (lon, lat, layer)
end

function create_transfer_buffer(nx, ny, nveg, nlayers)
    # Helper to alloc and pin
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
# 2. CREATE OUTPUT NETCDF
# ============================================================================

function create_output_netcdf(output_file::String, reference_array, reference_array2, float_type, lat_cpu, lon_cpu)
    println("Creating NetCDF output file with Optimized Chunking...")
    out_ds = NCDataset(output_file, "c")
    
    # ========================================================================
    # 1. DIMENSIONS & CHUNK CONFIGURATION
    # ========================================================================
    nx = size(reference_array, 1) 
    ny = size(reference_array, 2)
    nt = size(reference_array, 3)
    nveg = size(reference_array2, 4)
    
    defDim(out_ds, "lon",   nx)
    defDim(out_ds, "lat",   ny)
    defDim(out_ds, "time",  nt)
    defDim(out_ds, "nveg",  nveg)
    defDim(out_ds, "qlayers", 2)
    defDim(out_ds, "layer", 3)
    defDim(out_ds, "top_layer", 1)

    # OPTIMIZATION 1: Define Optimal Chunks
    # We want chunks to be the size of one write operation (one day)
    chunk_2d = (nx, ny, 1)
    chunk_3d_veg = (nx, ny, 1, nveg)
    chunk_3d_layer = (nx, ny, 1, 3)
    chunk_3d_qlayer = (nx, ny, 1, 2)
    chunk_3d_toplayer = (nx, ny, 1, 1)

    # Helper to define variable with SPEED settings (No compression, explicit chunks)
    function def_fast_var(ds, name, type, dims; chunks=nothing)
        v = defVar(ds, name, type, dims; 
                   chunksizes=chunks, 
                   deflatelevel=0, # COMPRESSION level
                   shuffle=false)
        return v
    end

    # ========================================================================
    # 2. STATIC VARIABLES (Lat/Lon)
    # ========================================================================
    lat = defVar(out_ds, "lat", float_type, ("lat",))
    lat.attrib["axis"] = "Y"
    lat.attrib["long_name"] = "latitude"
    lat.attrib["standard_name"] = "latitude"
    lat.attrib["units"] = "degrees_north"
    lat[:] = lat_cpu

    lon = defVar(out_ds, "lon", float_type, ("lon",))
    lon.attrib["axis"] = "X"
    lon.attrib["long_name"] = "longitude"
    lon.attrib["standard_name"] = "longitude"
    lon.attrib["units"] = "degrees_east"
    lon[:] = lon_cpu

    # ========================================================================
    # 3. OUTPUT VARIABLES (Optimized)
    # ========================================================================
    
    # --- 2D + Time Variables (Use chunk_2d) ---
    precipitation_output = def_fast_var(out_ds, "precipitation_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    precipitation_output.attrib["units"] = "mm/day"
    precipitation_output.attrib["description"] = "Daily precipitation"

    tair_output = def_fast_var(out_ds, "tair_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    tair_output.attrib["units"] = "°C"
    tair_output.attrib["description"] = "Air temperature at reference height"
    tair_output.attrib["_FillValue"] = float_type(1.e20)
    tair_output.attrib["missing_value"] = float_type(1.e20)

    tsurf_output = def_fast_var(out_ds, "tsurf_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    tsurf_output.attrib["units"] = "°C"
    tsurf_output.attrib["description"] = "Surface temperature per vegetation"

    canopy_evaporation_summed_output = def_fast_var(out_ds, "canopy_evaporation_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    canopy_evaporation_summed_output.attrib["units"] = "mm"
    canopy_evaporation_summed_output.attrib["description"] = "Total evaporation from canopy interception"

    transpiration_summed_output = def_fast_var(out_ds, "transpiration_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    transpiration_summed_output.attrib["units"] = "mm"
    transpiration_summed_output.attrib["description"] = "Total plant transpiration"

    potential_evaporation_summed_output = def_fast_var(out_ds, "potential_evaporation_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    potential_evaporation_summed_output.attrib["units"] = "mm"
    potential_evaporation_summed_output.attrib["description"] = "Potential evaporation"

    net_radiation_summed_output = def_fast_var(out_ds, "net_radiation_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    net_radiation_summed_output.attrib["units"] = "W/m^2"
    net_radiation_summed_output.attrib["description"] = "Net radiation"

    total_et_output = def_fast_var(out_ds, "total_et_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    total_et_output.attrib["units"] = "mm"
    total_et_output.attrib["description"] = "Total evapotranspiration"
    
    surface_runoff_output = def_fast_var(out_ds, "surface_runoff_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    surface_runoff_output.attrib["units"] = "mm"
    surface_runoff_output.attrib["description"] = "Surface runoff"

    total_runoff_output = def_fast_var(out_ds, "total_runoff_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    total_runoff_output.attrib["units"] = "mm"
    total_runoff_output.attrib["description"] = "Total runoff"

    # --- 3D + Time Variables (Layers) ---
    
    Q12_output = def_fast_var(out_ds, "Q12_output", float_type, ("lon", "lat", "time", "qlayers"); chunks=chunk_3d_qlayer)
    Q12_output.attrib["units"] = "mm"
    Q12_output.attrib["description"] = "Interlayer drainage"

    soil_evaporation_output = def_fast_var(out_ds, "soil_evaporation_output", float_type, ("lon", "lat", "time", "top_layer"); chunks=chunk_3d_toplayer)
    soil_evaporation_output.attrib["units"] = "mm"
    soil_evaporation_output.attrib["description"] = "Evaporation from the soil surface per top soil layer"

    soil_temperature_output = def_fast_var(out_ds, "soil_temperature_output", float_type, ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer)
    soil_temperature_output.attrib["units"] = "°C"
    soil_temperature_output.attrib["description"] = "Soil temperature per layer"

    soil_moisture_output = def_fast_var(out_ds, "soil_moisture_output", float_type, ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer)
    soil_moisture_output.attrib["units"] = "kg/m^3"
    soil_moisture_output.attrib["description"] = "Volumetric soil moisture content per layer"


    # ========================================================================
    # 4. RETURN
    # ========================================================================
    
    # Initialize Transfer Buffer (Crucial for matching the optimized write function)
    transfer_buf = create_transfer_buffer(nx, ny, nveg, 3)
    
    return out_ds, transfer_buf, 
           # 2D Summed/Standard Outputs
           precipitation_output, #water_storage_summed_output, 
           tair_output, tsurf_output, 
           canopy_evaporation_summed_output, transpiration_summed_output, 
           potential_evaporation_summed_output, # Note: You left this one active in your code
           net_radiation_summed_output, 
           
           # More 2D Outputs
           total_et_output, surface_runoff_output, total_runoff_output,
           
           # 3D / Layer Outputs
           Q12_output, soil_evaporation_output, soil_temperature_output, soil_moisture_output#,
           
end

# ============================================================================
# 3. PRE-PROCESS DAILY OUTPUTS
# ============================================================================

function preprocess_daily_outputs(
    day, tsurf, tair_gpu, prec_gpu, 
    total_et, surface_runoff, total_runoff,
    soil_evaporation, soil_moisture_new,
    potential_evaporation, net_radiation, transpiration, canopy_evaporation,
    coverage_gpu, cv_gpu, fillvalue_threshold
)

    # ========================================================================
    # DEBUG: TYPE CHECKER (Runs on Day 1 only)
    # ========================================================================
    if day == 1 
        println("\n" * "="^60)
        println("🔍 VARIABLE TYPE DIAGNOSTIC (Day $day)")
        println("="^60)
        vars_to_check = Dict(
            "tsurf"                 => tsurf,
            "tair_gpu"              => tair_gpu,
            "prec_gpu"              => prec_gpu,
            "total_et"              => total_et,
            "surface_runoff"        => surface_runoff,
            "total_runoff"          => total_runoff,
            "soil_evaporation"      => soil_evaporation,
            "soil_moisture_new"     => soil_moisture_new,
            "potential_evaporation" => potential_evaporation,
            "net_radiation"         => net_radiation,
            "transpiration"         => transpiration,
            "canopy_evaporation"    => canopy_evaporation,
            "coverage_gpu"          => coverage_gpu,
            "cv_gpu"                => cv_gpu
        )
        for (name, var) in sort(collect(vars_to_check), by=x->x[1])
            d_type = eltype(var)
            status = (d_type == Float64) ? "⚠️  Float64" : "✅ Float32"
            println("$(rpad(name, 25)) : $(rpad(string(d_type), 10)) | $status")
        end
        println("="^60 * "\n")
    end

    # ========================================================================
    # GPU HELPER FUNCTIONS
    # ========================================================================
    san_nan = A -> begin
        T = eltype(A)
        thr = T(fillvalue_threshold)
        rep = T(NaN)
        ifelse.(isnan.(A) .| (abs.(A) .> thr), rep, A)
    end
    
    # Helper to broadcast convert cv_gpu to match input array type
    convcv = A -> convert.(eltype(A), cv_gpu)

    # ========================================================================
    # CALCULATION & SANITIZATION
    # ========================================================================
    
    # --- Complex Processed Variables (Summing & Weighting) ---
    
    # 1. Potential Evaporation
    pe_processed = san_nan(potential_evaporation)
    pe_summed    = sum_with_nan_handling(convcv(pe_processed) .* pe_processed, 4)

    # 2. Net Radiation
    nr_processed = san_nan(net_radiation)
    nr_summed    = sum_with_nan_handling(convcv(nr_processed) .* nr_processed, 4)

    # 3. Transpiration
    tr_processed = san_nan(transpiration)
    # Weight by coverage
    tr_gc        = tr_processed .* coverage_gpu
    tr_summed    = sum_with_nan_handling(tr_gc, 4)

    # 4. Canopy Evaporation
    ce_processed = san_nan(canopy_evaporation)
    # Weight by coverage AND cv
    ce_gc        = convcv(ce_processed) .* ce_processed .* coverage_gpu
    ce_summed    = sum_with_nan_handling(ce_gc, 4)

    # ========================================================================
    # PACKAGING
    # ========================================================================
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

        # Calculated Sums (2D results from 3D inputs)
        pe_summed = pe_summed,
        nr_summed = nr_summed,
        tr_summed = tr_summed,
        ce_summed = ce_summed
    )
end



# ============================================================================
# 4. SPLIT TRANSFER AND WRITE-OUTPUTS (ASYNC OPTIMIZATION)
# ============================================================================

"""
Phase 1: Queue the copies on the GPU stream. 
This returns immediately, allowing the CPU to proceed.
"""
function async_transfer!(
    processed_data, 
    buf::TransferBuffer, 
    stream::CuStream
)
    # Helper for low-level async copy
    # Note: unsafe_copyto! requires contiguous arrays. 
    function async_copy(dest, src)
        # We use unsafe_copyto! to bypass Julia's default blocking mechanisms
        # The stream argument tells CUDA to queue this and move on.
        #CUDA.unsafe_copyto!(pointer(dest), pointer(src), length(dest), stream=stream)
        CUDA.stream!(stream) do
            copyto!(dest, src)
        end
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

"""
Phase 2: Synchronize stream (ensure data is in RAM) and write to Disk.
"""
function finalize_write!(
    day,
    buf::TransferBuffer,
    stream::CuStream,
    # Output Arrays (NetCDF vars)
    tsurf_out, tair_out, prec_out, 
    et_out, s_run_out, t_run_out,
    soil_evap_out, soil_mst_out,
    pe_out, nr_out, tr_out, ce_out
)
    # 1. WAIT for the specific stream to finish the copy.
    # While we wait here, the GPU can actually be busy calculating the NEXT day 
    # if we structure the main loop correctly.
    synchronize(stream)

    # 2. WRITE to Disk (Standard CPU blocking I/O)
    tsurf_out[:, :, day]        = buf.tsurf
    tair_out[:, :, day]         = buf.tair
    prec_out[:, :, day]         = buf.prec
    et_out[:, :, day]           = buf.total_et
    s_run_out[:, :, day]        = buf.surface_runoff
    t_run_out[:, :, day]        = buf.total_runoff
    
    pe_out[:, :, day]           = buf.pe_summed
    nr_out[:, :, day]           = buf.nr_summed
    tr_out[:, :, day]           = buf.tr_summed
    ce_out[:, :, day]           = buf.ce_summed
    
    soil_evap_out[:, :, day, :] = buf.soil_evaporation
    soil_mst_out[:, :, day, :]  = buf.soil_moisture
    
    return nothing
end