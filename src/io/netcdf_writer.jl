# ============================================================================
# 1. TRANSFER BUFFER STRUCT
# ============================================================================

struct TransferBuffer
    # Buffers for 2D variables (lon, lat)
    buf_2d::Matrix{Float32}
    
    # Buffers for 3D variables
    buf_3d_veg::Array{Float32, 3}      # (lon, lat, nveg)
    buf_3d_layer::Array{Float32, 3}    # (lon, lat, layer=3)
    buf_3d_qlayer::Array{Float32, 3}   # (lon, lat, qlayers=2)
    buf_3d_toplayer::Array{Float32, 3} # (lon, lat, top_layer=1)
end

function create_transfer_buffer(nx, ny, nveg, nlayers)
    # 1. Create standard arrays
    b2d = zeros(Float32, nx, ny)
    b3d_v = zeros(Float32, nx, ny, nveg)
    b3d_l = zeros(Float32, nx, ny, nlayers)
    b3d_q = zeros(Float32, nx, ny, 2)
    b3d_t = zeros(Float32, nx, ny, 1)

    # 2. Pin them (This locks them in physical RAM)
    CUDA.Mem.pin(b2d)
    CUDA.Mem.pin(b3d_v)
    CUDA.Mem.pin(b3d_l)
    CUDA.Mem.pin(b3d_q)
    CUDA.Mem.pin(b3d_t)

    # 3. Return struct
    TransferBuffer(b2d, b3d_v, b3d_l, b3d_q, b3d_t)
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
    chunk_2d = (nx ÷ 2, ny ÷ 2, 1)
    chunk_3d_veg = (nx ÷ 2, ny ÷ 2, 1, nveg)
    chunk_3d_layer = (nx ÷ 2, ny ÷ 2, 1, 3)
    chunk_3d_qlayer = (nx ÷ 2, ny ÷ 2, 1, 2)
    chunk_3d_toplayer = (nx ÷ 2, ny ÷ 2, 1, 1)

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

#    throughfall_summed_output = def_fast_var(out_ds, "throughfall_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    throughfall_summed_output.attrib["units"] = "mm/day"
#    throughfall_summed_output.attrib["description"] = "Total daily throughfall"

#    water_storage_summed_output = def_fast_var(out_ds, "water_storage_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    water_storage_summed_output.attrib["units"] = "mm"
#    water_storage_summed_output.attrib["description"] = "Total water stored in the canopy"

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

#    aerodynamic_resistance_summed_output = def_fast_var(out_ds, "aerodynamic_resistance_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    aerodynamic_resistance_summed_output.attrib["units"] = "s/m"
#    aerodynamic_resistance_summed_output.attrib["description"] = "Total aerodynamic resistance"    

    potential_evaporation_summed_output = def_fast_var(out_ds, "potential_evaporation_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    potential_evaporation_summed_output.attrib["units"] = "mm"
    potential_evaporation_summed_output.attrib["description"] = "Potential evaporation"

    net_radiation_summed_output = def_fast_var(out_ds, "net_radiation_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    net_radiation_summed_output.attrib["units"] = "W/m^2"
    net_radiation_summed_output.attrib["description"] = "Net radiation"

#    max_water_storage_summed_output = def_fast_var(out_ds, "max_water_storage_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    max_water_storage_summed_output.attrib["units"] = "mm"
#    max_water_storage_summed_output.attrib["description"] = "The maximum amount of water intercepted by the canopy"

    total_et_output = def_fast_var(out_ds, "total_et_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    total_et_output.attrib["units"] = "mm"
    total_et_output.attrib["description"] = "Total evapotranspiration"
    
    surface_runoff_output = def_fast_var(out_ds, "surface_runoff_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    surface_runoff_output.attrib["units"] = "mm"
    surface_runoff_output.attrib["description"] = "Surface runoff"

    total_runoff_output = def_fast_var(out_ds, "total_runoff_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    total_runoff_output.attrib["units"] = "mm"
    total_runoff_output.attrib["description"] = "Total runoff"

#    g_sw_summed_output = def_fast_var(out_ds, "g_sw_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    g_sw_summed_output.attrib["units"] = ""
#    g_sw_summed_output.attrib["description"] = "g_sw_summed_output"
#
#    g_sw_1_summed_output = def_fast_var(out_ds, "g_sw_1_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    g_sw_1_summed_output.attrib["units"] = ""
#    g_sw_1_summed_output.attrib["description"] = "g_sw_1_summed_output"
#
#    g_sw_2_summed_output = def_fast_var(out_ds, "g_sw_2_summed_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    g_sw_2_summed_output.attrib["units"] = ""
#    g_sw_2_summed_output.attrib["description"] = "g_sw_2_summed_output"
#    
#    g_sw_1_output = def_fast_var(out_ds, "g_sw_1_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    g_sw_2_output = def_fast_var(out_ds, "g_sw_2_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
    
#    asat_output = def_fast_var(out_ds, "asat_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    asat_output.attrib["units"] = "fraction"
#    asat_output.attrib["description"] = "Fraction of saturated area"

#    vp_output = def_fast_var(out_ds, "vp_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    vp_output.attrib["units"] = "kPa"
#    vp_output.attrib["description"] = "Vapor pressure"
#
#    vpd_output = def_fast_var(out_ds, "vpd_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    vpd_output.attrib["units"] = "Pa"
#    vpd_output.attrib["description"] = "Vapor pressure deficit"

#    density_output = def_fast_var(out_ds, "density_output", float_type, ("lon", "lat", "time"); chunks=chunk_2d)
#    density_output.attrib["units"] = "kg/m^3"
#    density_output.attrib["description"] = "Air density"

    # --- 3D + Time Variables (Vegetation) (Use chunk_3d_veg) ---
    
#    throughfall_output = def_fast_var(out_ds, "throughfall_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    throughfall_output.attrib["units"] = "mm/day"
#    throughfall_output.attrib["description"] = "Daily throughfall per vegetation"

#    water_storage_output = def_fast_var(out_ds, "water_storage_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    water_storage_output.attrib["units"] = "mm"
#    water_storage_output.attrib["description"] = "Water stored in the canopy per vegetation"

#    canopy_evaporation_output = def_fast_var(out_ds, "canopy_evaporation_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    canopy_evaporation_output.attrib["units"] = "mm"
#    canopy_evaporation_output.attrib["description"] = "Evaporation from canopy interception per vegetation"

#    transpiration_output = def_fast_var(out_ds, "transpiration_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    transpiration_output.attrib["units"] = "mm"
#    transpiration_output.attrib["description"] = "Plant transpiration per vegetation"

#    aerodynamic_resistance_output = def_fast_var(out_ds, "aerodynamic_resistance_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    aerodynamic_resistance_output.attrib["units"] = "s/m"
#    aerodynamic_resistance_output.attrib["description"] = "Aerodynamic resistance per vegetation"    

#    potential_evaporation_output = def_fast_var(out_ds, "potential_evaporation_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    potential_evaporation_output.attrib["units"] = "mm"
#    potential_evaporation_output.attrib["description"] = "Potential evaporation per vegetation"

#    net_radiation_output = def_fast_var(out_ds, "net_radiation_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    net_radiation_output.attrib["units"] = "W/m^2"
#    net_radiation_output.attrib["description"] = "Net radiation, per vegetation"

#    max_water_storage_output = def_fast_var(out_ds, "max_water_storage_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    max_water_storage_output.attrib["units"] = "mm"
#    max_water_storage_output.attrib["description"] = "The maximum amount of water intercepted by the canopy per vegetation"

#    E_1_t_output = def_fast_var(out_ds, "E_1_t_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    E_2_t_output = def_fast_var(out_ds, "E_2_t_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
    
#    g_sw_output = def_fast_var(out_ds, "g_sw_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    g_sw_output.attrib["units"] = ""
#    g_sw_output.attrib["description"] = "g_sw_output"

#    dry_time_factor_output = def_fast_var(out_ds, "dry_time_factor_output", float_type, ("lon", "lat", "time", "nveg"); chunks=chunk_3d_veg)
#    dry_time_factor_output.attrib["units"] = ""
#    dry_time_factor_output.attrib["description"] = "dry_time_factor_output"


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

#    kappa_array_output = def_fast_var(out_ds, "kappa_array_output", float_type, ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer)
#    cs_array_output = def_fast_var(out_ds, "cs_array_output", float_type, ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer)
#    residual_moisture_output = def_fast_var(out_ds, "residual_moisture_output", float_type, ("lon", "lat", "time", "layer"); chunks=chunk_3d_layer)

    # --- Time-Invariant Variables (Can use standard defVar or static chunks) ---
    
#    wilting_point_output = defVar(out_ds, "wilting_point_output", float_type, ("lon", "lat", "layer"))
#    soil_moisture_max_output = defVar(out_ds, "soil_moisture_max_output", float_type, ("lon", "lat", "layer"))
#    soil_moisture_critical_output = defVar(out_ds, "soil_moisture_critical_output", float_type, ("lon", "lat", "layer"))

    # ========================================================================
    # 4. RETURN
    # ========================================================================
    
    # Initialize Transfer Buffer (Crucial for matching the optimized write function)
    transfer_buf = create_transfer_buffer(nx, ny, nveg, 3)

    # return out_ds, transfer_buf, precipitation_output, water_storage_output, water_storage_summed_output, Q12_output, 
    #        tair_output, tsurf_output, canopy_evaporation_output,
    #        canopy_evaporation_summed_output, transpiration_output, transpiration_summed_output, aerodynamic_resistance_output, aerodynamic_resistance_summed_output,
    #        potential_evaporation_output, potential_evaporation_summed_output, net_radiation_output,
    #        net_radiation_summed_output, max_water_storage_output, max_water_storage_summed_output,
    #        soil_evaporation_output, soil_temperature_output, soil_moisture_output,  total_et_output, surface_runoff_output, total_runoff_output,
    #        kappa_array_output, cs_array_output, wilting_point_output, soil_moisture_max_output, soil_moisture_critical_output,
    #        E_1_t_output, E_2_t_output, g_sw_1_output, g_sw_2_output, g_sw_output, residual_moisture_output, 
    #        throughfall_output, throughfall_summed_output,
    #        surfstor_output, 
    #        asat_output, vp_output, vpd_output,
    #        density_output, g_sw_output, g_sw_summed_output, dry_time_factor_output, g_sw_1_summed_output, g_sw_2_summed_output
    
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
           
           # Soil Properties / Fixed Fields
           #soil_moisture_max_output, soil_moisture_critical_output,
           
           # G_SW variables (You had these active)
           #g_sw_1_output, g_sw_2_output, g_sw_output,
           #g_sw_summed_output, g_sw_1_summed_output, g_sw_2_summed_output,
           
           # Other active 2D variables
           #throughfall_summed_output, 
           #asat_output, vp_output, vpd_output,
           #density_output
end


# ============================================================================
# 3. WRITE DAILY OUTPUTS (OPTIMIZED)
# ============================================================================

# ============================================================================
# 3. WRITE DAILY OUTPUTS (OPTIMIZED DEFINITION)
# ============================================================================

function write_daily_outputs(
                            # --- INPUTS (These match the data passed in) ---
                            day, tsurf, aerodynamic_resistance, ra_eff, 
                            transpiration, tair_gpu, prec_gpu, throughfall,
                            asat, 
                            vp_gpu, vpd, Q12, soil_evaporation, 
                            soil_temperature, soil_moisture_new, total_et,
                            surface_runoff, total_runoff, kappa_array, cs_array, 
                            potential_evaporation, water_storage, net_radiation,
                            canopy_evaporation, max_water_storage, wilting_point,
                            soil_moisture_critical, soil_moisture_max, E_1_t, 
                            E_2_t, residual_moisture, cv_gpu, coverage_gpu, g_sw, dry_time_factor, #g_sw_1, g_sw_2,
                            
                            # --- OUTPUTS (Updated to match your new shortened list) ---
                            tsurf_output, 
                            transpiration_summed_output,
                            tair_output, 
                            precipitation_output, 
                            #throughfall_summed_output, 
                            #asat_output, 
                            #vp_output,
                            #vpd_output, 
                            #density_output,
                            Q12_output, 
                            soil_evaporation_output,
                            soil_temperature_output, 
                            soil_moisture_output,
                            total_et_output, 
                            surface_runoff_output, 
                            total_runoff_output,
                            potential_evaporation_summed_output,
                            #water_storage_summed_output,
                            net_radiation_summed_output,
                            canopy_evaporation_summed_output,
                            #soil_moisture_critical_output, 
                            #soil_moisture_max_output, 
                            #g_sw_output, 
                            #g_sw_summed_output, 
                            #g_sw_1_output, 
                            #g_sw_2_output, 
                            #g_sw_1_summed_output, 
                            #g_sw_2_summed_output,
                            
                            # --- BUFFER ---
                            transfer_buf)
    
    # ========================================================================
    # 🔍 DEBUG: TYPE CHECKER (Runs on Day 1 only)
    # ========================================================================
    if day == 1 
        println("\n" * "="^60)
        println("🔍  VARIABLE TYPE DIAGNOSTIC (Day $day)")
        println("="^60)
        
        # Dictionary of ACTIVE variables currently being written below
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
            "coverage_gpu"          => coverage_gpu, # Used in calc
            "cv_gpu"                => cv_gpu        # Used in calc
        )

        # Iterate and Print
        for (name, var) in sort(collect(vars_to_check), by=x->x[1])
            d_type = eltype(var)
            
            # Add a visual flag if it's Float64 (usually unexpected in GPU->NetCDF pipelines)
            status = (d_type == Float64) ? "⚠️  Float64" : "✅ Float32"
            
            println("$(rpad(name, 25)) : $(rpad(string(d_type), 10)) | $status")
        end
        println("="^60 * "\n")
    end
    # ========================================================================


    # GPU-safe sanitizers
    san_nan = A -> begin
        T = eltype(A)
        thr = T(fillvalue_threshold)
        rep = T(NaN)
        ifelse.(isnan.(A) .| (abs.(A) .> thr), rep, A)
    end
    
    san_zero = A -> begin
        T = eltype(A)
        thr = T(fillvalue_threshold)
        rep = T(0.0)
        ifelse.(isnan.(A) .| (abs.(A) .> thr), rep, A)
    end
    
    convcv = A -> convert.(eltype(A), cv_gpu)

    # ========================================================================
    # 2D Variables (Use buf_2d)
    # ========================================================================
    copyto!(transfer_buf.buf_2d, tsurf)
    tsurf_output[:, :, day] = transfer_buf.buf_2d

#    copyto!(transfer_buf.buf_2d, ra_eff)
#    aerodynamic_resistance_summed_output[:, :, day] = transfer_buf.buf_2d
    
    copyto!(transfer_buf.buf_2d, tair_gpu)
    tair_output[:, :, day] = transfer_buf.buf_2d
    
    copyto!(transfer_buf.buf_2d, prec_gpu)
    precipitation_output[:, :, day] = transfer_buf.buf_2d

#    copyto!(transfer_buf.buf_2d, sum_with_nan_handling(throughfall, 4))
#    throughfall_summed_output[:, :, day] = transfer_buf.buf_2d
    
#    copyto!(transfer_buf.buf_2d, asat)
#    asat_output[:, :, day] = transfer_buf.buf_2d
    
#    copyto!(transfer_buf.buf_2d, vp_gpu)
#    vp_output[:, :, day] = transfer_buf.buf_2d
#    
#    copyto!(transfer_buf.buf_2d, vpd)
#    vpd_output[:, :, day] = transfer_buf.buf_2d
#
#    # Special case for density (filled scalar)
#    fill!(transfer_buf.buf_2d, eltype(tsurf)(rho_a)) 
#    density_output[:, :, day] = transfer_buf.buf_2d
    
    copyto!(transfer_buf.buf_2d, total_et)
    total_et_output[:, :, day] = transfer_buf.buf_2d

    copyto!(transfer_buf.buf_2d, surface_runoff)
    surface_runoff_output[:, :, day] = transfer_buf.buf_2d

    copyto!(transfer_buf.buf_2d, total_runoff)
    total_runoff_output[:, :, day] = transfer_buf.buf_2d

    # ========================================================================
    # 3D Vegetation Variables (Use buf_3d_veg)
    # ========================================================================
#    copyto!(transfer_buf.buf_3d_veg, aerodynamic_resistance)
#    aerodynamic_resistance_output[:, :, day, :] = transfer_buf.buf_3d_veg

#    copyto!(transfer_buf.buf_3d_veg, throughfall)
#    throughfall_output[:, :, day, :] = transfer_buf.buf_3d_veg

#    copyto!(transfer_buf.buf_3d_veg, dry_time_factor)
#    dry_time_factor_output[:, :, day, :] = transfer_buf.buf_3d_veg

    # E_1_t and E_2_t
#    copyto!(transfer_buf.buf_3d_veg, E_1_t)
#    E_1_t_output[:, :, day, :] = transfer_buf.buf_3d_veg
#    copyto!(transfer_buf.buf_3d_veg, E_2_t)
#    E_2_t_output[:, :, day, :] = transfer_buf.buf_3d_veg

    # Processed Variables (Sanitization needed first)
    # Note: We sanitize into the buffer to avoid allocating a new GPU array
    
    # Q12 (Uses buf_3d_qlayer)
#    copyto!(transfer_buf.buf_3d_qlayer, san_zero(Q12))
#    Q12_output[:, :, day, :] = transfer_buf.buf_3d_qlayer
    
    # Soil Evaporation (Uses buf_3d_toplayer)
    copyto!(transfer_buf.buf_3d_toplayer, soil_evaporation)
    soil_evaporation_output[:, :, day, :] = transfer_buf.buf_3d_toplayer

    # Soil Properties (Uses buf_3d_layer)
#    copyto!(transfer_buf.buf_3d_layer, soil_temperature)
#    soil_temperature_output[:, :, day, :] = transfer_buf.buf_3d_layer
    
    copyto!(transfer_buf.buf_3d_layer, soil_moisture_new)
    soil_moisture_output[:, :, day, :] = transfer_buf.buf_3d_layer
    
#    copyto!(transfer_buf.buf_3d_layer, kappa_array)
#    kappa_array_output[:, :, day, :] = transfer_buf.buf_3d_layer
    
#    copyto!(transfer_buf.buf_3d_layer, cs_array)
#    cs_array_output[:, :, day, :] = transfer_buf.buf_3d_layer

#    copyto!(transfer_buf.buf_3d_layer, residual_moisture)
#    residual_moisture_output[:, :, day, :] = transfer_buf.buf_3d_layer

    # Complex Processed outputs (Sanitize + Sum)
    # Strategy: Sanitize to GPU temporary (if needed) or stream via buffer if simple
    
    # G_SW
#    g_sw_processed = san_nan(g_sw)
#    copyto!(transfer_buf.buf_3d_veg, g_sw_processed)
#    g_sw_output[:, :, day, :] = transfer_buf.buf_3d_veg
    
#    copyto!(transfer_buf.buf_2d, sum_with_nan_handling(convcv(g_sw_processed) .* g_sw_processed, 4))
#    g_sw_summed_output[:, :, day] = transfer_buf.buf_2d

    # G_SW 1 & 2
#    g_sw_1_processed = san_nan(g_sw_1)
#    copyto!(transfer_buf.buf_2d, g_sw_1_processed)
#    g_sw_1_output[:, :, day] = transfer_buf.buf_2d
    # If you need summed output for g_sw_1, calculate and copy here too

#    g_sw_2_processed = san_nan(g_sw_2)
#    copyto!(transfer_buf.buf_2d, g_sw_2_processed)
#    g_sw_2_output[:, :, day] = transfer_buf.buf_2d

    # Potential Evaporation
    pe_processed = san_nan(potential_evaporation)
#    copyto!(transfer_buf.buf_3d_veg, pe_processed)
#    potential_evaporation_output[:, :, day, :] = transfer_buf.buf_3d_veg
    
    copyto!(transfer_buf.buf_2d, sum_with_nan_handling(convcv(pe_processed) .* pe_processed, 4))
    potential_evaporation_summed_output[:, :, day] = transfer_buf.buf_2d
    
    # Water Storage
#    ws_processed = san_nan(water_storage)
#    copyto!(transfer_buf.buf_3d_veg, ws_processed)
#    water_storage_output[:, :, day, :] = transfer_buf.buf_3d_veg
    
#    copyto!(transfer_buf.buf_2d, sum_with_nan_handling(ws_processed, 4))
#    water_storage_summed_output[:, :, day] = transfer_buf.buf_2d
    
    # Net Radiation
    nr_processed = san_nan(net_radiation)
#    copyto!(transfer_buf.buf_3d_veg, nr_processed)
#    net_radiation_output[:, :, day, :] = transfer_buf.buf_3d_veg
    
    copyto!(transfer_buf.buf_2d, sum_with_nan_handling(convcv(nr_processed) .* nr_processed, 4))
    net_radiation_summed_output[:, :, day] = transfer_buf.buf_2d

    # Transpiration
    tr_processed = san_nan(transpiration)
    tr_gc = tr_processed .* coverage_gpu
#    copyto!(transfer_buf.buf_3d_veg, tr_gc)
#    transpiration_output[:, :, day, :] = transfer_buf.buf_3d_veg
    
    copyto!(transfer_buf.buf_2d, sum_with_nan_handling(tr_gc, 4))
    transpiration_summed_output[:, :, day] = transfer_buf.buf_2d
    
    # Canopy Evaporation
    ce_processed = san_nan(canopy_evaporation)
    ce_gc = convcv(ce_processed) .* ce_processed .* coverage_gpu
#    copyto!(transfer_buf.buf_3d_veg, ce_gc)
#    canopy_evaporation_output[:, :, day, :] = transfer_buf.buf_3d_veg
    
    copyto!(transfer_buf.buf_2d, sum_with_nan_handling(ce_gc, 4))
    canopy_evaporation_summed_output[:, :, day] = transfer_buf.buf_2d
    
    # Max Water Storage
#    mws_processed = san_nan(max_water_storage)
#    copyto!(transfer_buf.buf_3d_veg, mws_processed)
#    max_water_storage_output[:, :, day, :] = transfer_buf.buf_3d_veg
    
#    copyto!(transfer_buf.buf_2d, sum_with_nan_handling(convcv(mws_processed) .* mws_processed, 4))
#    max_water_storage_summed_output[:, :, day] = transfer_buf.buf_2d
    
    # Soil properties (Time invariant daily write)
    # Note: Dimensions are usually just (lon, lat, layer), so we can't use day index.
    # Assuming these are 3D arrays on GPU being written to 3D vars in NC
#    copyto!(transfer_buf.buf_3d_layer, wilting_point)
#    wilting_point_output[:, :, :] = transfer_buf.buf_3d_layer

#    copyto!(transfer_buf.buf_3d_layer, soil_moisture_critical)
#    soil_moisture_critical_output[:, :, :] = transfer_buf.buf_3d_layer
    
#    copyto!(transfer_buf.buf_3d_layer, soil_moisture_max)
#    soil_moisture_max_output[:, :, :] = transfer_buf.buf_3d_layer
end