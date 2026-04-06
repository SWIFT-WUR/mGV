# ============================================================================
# INITIALIZATION
# ============================================================================
const to = TimerOutputs.TimerOutput()

println("Loading parameter data and allocating memory...")

@time begin
    @load_params(
        lat, lon, d0, z0, z0soil, LAI, albedo, rmin, rarc, cv, elev,
        ksat, residmoist, init_moist, root, Wcr, Wfc, Wpwp, depth,
        quartz, bulk_dens, soil_dens, expt, coverage, b_infilt,
        Ds, Dsmax, Ws, dp, Tavg, c_expt,
        AreaFract, elevation, Pfactor, annual_prec
    )
end

@timeit to "gpu_load_static_inputs" @time gpu_load_static_inputs(@vars(
    rmin, rarc, cv, elev, ksat, residmoist, init_moist, root, Wcr, Wfc, Wpwp,
    depth, quartz, bulk_dens, soil_dens, expt, b_infilt, Ds, Dsmax, Ws, dp, Tavg, z0soil, c_expt,
    AreaFract, elevation, Pfactor, annual_prec
)...)

@timeit to "init_routing" begin
    # Conditionally initialize the routing state; if disabled, assign nothing to avoid undefined variable
    const routing_state = enable_routing ? initialize_routing_model(routing_param_file, elev_cpu) : nothing
end

@timeit to "reshaping_inputs" begin
    reshape_static_inputs!()
end

# ============================================================================
# INITIALIZE GPU STATE ARRAYS
# ============================================================================
println("Allocating State Arrays on: $backend_name")

@timeit to "initialize_GPU_arrays" begin
    
    # --- 1. Define shapes for readability and consistency ---
    dim_grid = size(Tavg_gpu)                        # (nx, ny)
    
    global AreaFract_gpu, elevation_gpu, Pfactor_gpu
    if isnothing(AreaFract_gpu)
        AreaFract_gpu = alloc(dim_grid..., 1)
        fill!(AreaFract_gpu, FloatType(1.0))
        elevation_gpu = alloc(dim_grid..., 1)
        @. elevation_gpu = elev_gpu
        Pfactor_gpu = alloc(dim_grid..., 1)
        fill!(Pfactor_gpu, FloatType(1.0))
    end
    
    nbands = size(AreaFract_gpu, 3)
    dim_tile = (size(coverage_gpu, 1), size(coverage_gpu, 2), nbands, size(coverage_gpu, 4))
    dim_soil = size(soil_dens_gpu)                   # (nx, ny, nlayers)
    dim_veg  = size(root_gpu, 4)                     # Number of vegetation tiles: nveg
    n_layers = dim_soil[3]                           # Extract layer count (usually 3)

    # --- 2. Per-Tile States (Vegetation/Canopy) ---
    # Shape: (nx, ny, 1, nveg)
    const water_storage          = alloc(dim_tile...)
    const max_water_storage      = alloc(dim_tile...)
    const throughfall            = alloc(dim_tile...)
    const canopy_evaporation     = alloc(dim_tile...)
    const f_n                    = alloc(dim_tile...)
    const net_radiation          = alloc(dim_tile...)
    const potential_evaporation  = alloc(dim_tile...)  # Step 1 PE: for transpiration, canopy evap
    const pe_soil                = alloc(dim_tile...)  # Step 2 PE: for soil evaporation (snow-blended energy)
    const aerodynamic_resistance = alloc(dim_tile...)
    const transpiration          = alloc(dim_tile...)
    const E_1_t                  = alloc(dim_tile...)
    const E_2_t                  = alloc(dim_tile...)
    const dry_time_factor        = alloc(dim_tile...)

    # --- 3. Per-Pixel States (Grid Averages) ---
    # Shape: (nx, ny)
    const tsurf             = alloc(dim_grid...)
    const Q_12              = alloc(dim_grid...)
    const soil_evaporation  = alloc(dim_grid...)
    const total_et          = alloc(dim_grid...)
    const infiltration      = alloc(dim_grid...)
    const surface_runoff    = alloc(dim_grid...)
    const asat              = alloc(dim_grid...)
    const subsurface_runoff = alloc(dim_grid...)
    const total_runoff      = alloc(dim_grid...)
    const g1_buf            = alloc(dim_grid...)
    const g2_buf            = alloc(dim_grid...)
    
    # --- 3b. Snow Per-(Band × Veg) States — 4D to match VIC architecture ---
    # Shape: (nx, ny, nbands, nveg)  — one snowpack per (elevation band × vegetation tile)
    # VIC's collect_wb_terms accumulates: OUT_SWE += snow.swq * Cv[veg] * AreaFract[band]
    const dim_snow          = (dim_grid[1], dim_grid[2], nbands, dim_veg)
    const swe_gpu               = alloc(dim_snow...)
    const snow_depth_gpu        = alloc(dim_snow...)
    const snow_albedo_gpu       = alloc(dim_snow...)
    const snow_surf_temp_gpu    = alloc(dim_snow...)
    const snow_coverage_gpu     = alloc(dim_snow...)
    const snow_melt_gpu         = alloc(dim_snow...)
    # VIC-faithful snow state variables
    const last_snow_gpu         = alloc(Int32, dim_snow...)   # days since last snowfall
    const cold_content_gpu      = alloc(dim_snow...)           # J/m² surface layer cold content
    const pack_cc_gpu           = alloc(dim_snow...)           # J/m² pack layer cold content (VIC 2-layer)
    const melting_flag_gpu      = alloc(Int32, dim_snow...)   # melt-season flag
    const store_snow_gpu        = alloc(Int32, dim_snow...)   # coverage state (1 = store)
    const snow_distrib_slope_gpu= alloc(dim_snow...)           # depth distribution slope (m)
    const store_swq_gpu         = alloc(dim_snow...)           # stored SWE for coverage (mm)
    const store_coverage_gpu    = alloc(dim_snow...)           # stored coverage fraction
    const max_snow_depth_gpu    = alloc(dim_snow...)           # max depth for coverage (m)
    # Per-band (3D) buffer for melt aggregated across veg tiles (for soil input)
    const melt_band_gpu         = alloc(dim_grid[1], dim_grid[2], nbands)
    const rain_band_gpu         = alloc(dim_grid[1], dim_grid[2], nbands)
    const ppt_gpu               = alloc(dim_grid[1], dim_grid[2], nbands)

    # --- 3. Forcings Buffers ---
    const tair_band              = alloc(dim_grid[1], dim_grid[2], nbands)
    const prec_band              = alloc(dim_grid[1], dim_grid[2], nbands)

    # --- 4. Soil Properties & States ---
    # Shape: (nx, ny, 3) or matched to soil density
    const soil_moisture          = alloc(dim_soil...)
    const soil_moisture_max      = alloc(dim_soil...)
    const soil_moisture_critical = alloc(dim_soil...)
    const kappa_array            = alloc(dim_soil...)
    const cs_array               = alloc(dim_soil...)
    const field_capacity         = alloc(dim_soil...)
    const wilting_point          = alloc(dim_soil...)
    const residual_moisture      = alloc(dim_soil...)
    const ice_frac               = alloc(dim_soil...)
    const soil_temperature       = alloc(dim_soil...) 
    
    const organic_frac_gpu       = alloc(dim_soil...)
    fill!(organic_frac_gpu, FloatType(organic_frac))

    # Soil constant maps (Shape: nx, ny, 3)
    const bulk_dens_min = alloc(size(bulk_dens_gpu)...)
    const soil_dens_min = alloc(size(bulk_dens_gpu)...)
    const porosity      = alloc(size(bulk_dens_gpu)...)
    const Lsum          = alloc(size(soil_dens_gpu)...)

    # --- 5. Complex Shapes ---
    const interlayer_drainage  = alloc(dim_grid[1], dim_grid[2], 2)
    const transpiration_layers = alloc(dim_grid[1], dim_grid[2], n_layers, dim_veg)
    const g_sw_veg_buf         = alloc(dim_grid[1], dim_grid[2], 1, dim_veg)

    # --- 6. Initialization ---
    # Initialize soil temperature with Tavg (broadcast across the ground layers, usually 3)
    for k in 1:n_layers
        view(soil_temperature, :, :, k) .= Tavg_gpu
    end

    # Explicitly initialize all allocated states to 0.0 to prevent propagating mem alloc() garbage (NaNs/Infs)
    let arrays_to_zero = (
        water_storage, max_water_storage, throughfall, canopy_evaporation,
        f_n, net_radiation, potential_evaporation, aerodynamic_resistance,
        transpiration, E_1_t, E_2_t, dry_time_factor, tsurf, Q_12,
        soil_evaporation, total_et, infiltration, surface_runoff, asat,
        subsurface_runoff, total_runoff, g1_buf, g2_buf,
        soil_moisture_max, soil_moisture_critical,
        kappa_array, cs_array, field_capacity, wilting_point,
        residual_moisture, ice_frac, bulk_dens_min, soil_dens_min,
        porosity, Lsum, interlayer_drainage, transpiration_layers,
        g_sw_veg_buf,
        swe_gpu, snow_depth_gpu, snow_albedo_gpu, snow_surf_temp_gpu,
        snow_coverage_gpu, snow_melt_gpu,
        cold_content_gpu, pack_cc_gpu, snow_distrib_slope_gpu,
        store_swq_gpu, store_coverage_gpu, max_snow_depth_gpu,
        melt_band_gpu, rain_band_gpu, ppt_gpu
    )
        for arr in arrays_to_zero
            fill!(arr, FloatType(0.0))
        end
    end
    # Initialize store_snow to 0 (VIC default: store_snow = false)
    # store_coverage and store_swq stay at 0.0 (already zeroed above)
end


# ============================================================================
# CALCULATE SOIL PROPERTIES
# ============================================================================

@timeit to "calculate_soil_properties" begin
    # Calculate properties directly into the pre-allocated global arrays
    calculate_soil_properties!(
        # Destinations (global arrays)
        bulk_dens_min, soil_dens_min, porosity,
        soil_moisture_max, soil_moisture_critical,
        field_capacity, wilting_point, residual_moisture,

        # Inputs
        bulk_dens_gpu, soil_dens_gpu, depth_gpu,
        Wcr_gpu, Wfc_gpu, Wpwp_gpu, residmoist_gpu,

        # Constants
        organic_frac, bulk_dens_org, soil_dens_org
    )

    soil_moisture .= clamp.(init_moist_gpu, residual_moisture, soil_moisture_max)
    
    # Translate baseflow parameters from NIJSSEN2001 to ARNO format
    convert_nijssen2001_to_arno!(Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu, soil_moisture_max)
end

# ============================================================================
# PROCESS YEAR FUNCTION
# ============================================================================

function process_year(year)
    println("============ Start run for year: $year ============")
    output_format = get_output_format()
    println("Output format selected: $output_format")

    # ------------------------------------------------------------------------
    # Load forcing data
    # ------------------------------------------------------------------------
    println("Loading forcing data and allocating memory...")
    @timeit to "load_forcing" begin
        @load_forcing year prec tair wind vp swdown lwdown psurf
    end

    # ------------------------------------------------------------------------
    # Create output file
    # ------------------------------------------------------------------------
    @debug println("Initializing output store...")
    local output_store

    @timeit to "create_output" begin
        nx, ny = size(prec_cpu, 1), size(prec_cpu, 2)
        nt = size(prec_cpu, 3)
        nlayers = 3 

        if output_format == :netcdf
            output_path = joinpath(output_dir, "$(output_file_prefix)$(year).nc")
            # Ignore the single buffer returned; we will create a pool instead.
            output_store, _ = create_output_netcdf(output_path, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
        else
            output_path = joinpath(output_dir, "$(output_file_prefix)$(year).zarr")
            output_store, _ = create_output_zarr(output_path, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
        end

        # Start the ssync pool 
        println("Starting Async I/O Service...")
        io_service = start_async_service(nx, ny, nlayers, output_store, 6)
    end

    # ------------------------------------------------------------------------
    # Daily timestep loop
    # ------------------------------------------------------------------------
    println("Running...")
    num_days = size(prec_cpu, 3)
    day_prev = 0
    month_prev = 0

    @showprogress "Processing year $year (GPU)..." for day in 1:num_days
        @timeit to "process_year" begin
            month = day_to_month(day, year)

            # ============================================================
            # Load daily/monthly inputs to GPU
            # ============================================================
            @timeit to "gpu_load_monthly_inputs" begin
                gpu_load_monthly_inputs(month, month_prev,
                    @vars(d0, z0, LAI, albedo, coverage)...)
            end

            @timeit to "gpu_load_daily_inputs" begin
                gpu_load_daily_inputs(day, day_prev,
                    @vars(prec, tair, wind, vp, swdown, lwdown, psurf)...)
            end

            # Initialize surface temperature on first timestep
            if day == 1 && year == start_year
                tsurf .= tair_gpu
            end

            # ============================================================
            # Energy balance and atmospheric calculations
            # ============================================================
            @timeit to "calculate_band_forcings" begin
                @. tair_band = tair_gpu + FloatType(-0.0065) * (elevation_gpu - elev_gpu)
                # VIC Pfactor fix: NC file stores precipitation FRACTIONS, not multipliers.
                # VIC divides by AreaFract to get the true per-band precipitation multiplier.
                # When Pfactor_nc[b] == AreaFract[b] (common case: uniform precip per unit area),
                # the true multiplier = 1.0, so each band gets the FULL grid-cell precipitation.
                # prec_band[b] = prec * (Pfactor_nc[b] / AreaFract[b])
                @. prec_band = prec_gpu * ifelse(
                    AreaFract_gpu > FloatType(1e-6),
                    Pfactor_gpu / AreaFract_gpu,
                    FloatType(0.0)
                )
            end

            @timeit to "compute_aerodynamic_resistance" begin
                compute_aerodynamic_resistance!(
                    aerodynamic_resistance,
                    z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_band, wind_gpu, cv_gpu
                )
            end

            @timeit to "calculate_net_radiation" begin
                # Step 1: compute WITHOUT snow for PE (matching VIC's compute_pot_evap convention)
                # Use grid-level tair_gpu for LW emission (consistent with VIC's Penman LW term).
                calculate_net_radiation!(
                    net_radiation, swdown_gpu, lwdown_gpu, albedo_gpu, tair_gpu
                )
            end

            @timeit to "calculate_potential_evaporation" begin
                calculate_potential_evaporation!(
                    potential_evaporation,
                    tair_band, tair_gpu, psurf_gpu, vp_gpu, elev_gpu, net_radiation,
                    aerodynamic_resistance, rarc_gpu, rmin_gpu, LAI_gpu
                )
            end

            @timeit to "calculate_net_radiation_snow" begin
                # Step 2: recompute WITH snow for the full energy balance
                calculate_net_radiation!(
                    net_radiation, swdown_gpu, lwdown_gpu, albedo_gpu, tsurf,
                    snow_coverage_gpu, snow_albedo_gpu, snow_surf_temp_gpu
                )
            end

            @timeit to "calculate_potential_evaporation_soil" begin
                # Step 2 PE for soil evaporation: snow-blended net_rad gives lower
                # PE in energy-limited (snow/winter) months, reducing mGV's 22%
                # summer overestimate vs VIC. (See run55-59 for validation.)
                calculate_potential_evaporation!(
                    pe_soil,
                    tair_band, tair_gpu, psurf_gpu, vp_gpu, elev_gpu, net_radiation,
                    aerodynamic_resistance, rarc_gpu, rmin_gpu, LAI_gpu
                )
            end

            # ============================================================
            # Canopy processes
            # ============================================================
            @timeit to "calculate_max_water_storage" begin
                calculate_max_water_storage!(max_water_storage, LAI_gpu, coverage_gpu)
            end

            @timeit to "calculate_canopy_evaporation" begin
                calculate_canopy_evaporation!(
                    canopy_evaporation, f_n,
                    water_storage, max_water_storage, potential_evaporation,
                    aerodynamic_resistance, rarc_gpu, prec_band, cv_gpu,
                    rmin_gpu, LAI_gpu, tair_band, elev_gpu
                )
            end

            # ============================================================
            # Transpiration
            # ============================================================
            @timeit to "calculate_transpiration" begin
                calculate_transpiration!(
                    # Outputs
                    transpiration,
                    transpiration_layers,
                    # Inputs
                    potential_evaporation, 
                    water_storage, 
                    max_water_storage, 
                    soil_moisture,
                    soil_moisture_critical, 
                    wilting_point, 
                    root_gpu, 
                    cv_gpu, 
                    f_n,
                    AreaFract_gpu,
                    tair_gpu,
                    vp_gpu
                )
            end

            # ============================================================
            # Water balance: throughfall (must run BEFORE snow dynamics so
            # the snow kernel sees today's precipitation, not yesterday's)
            # ============================================================
            @timeit to "update_water_canopy_storage" begin
                update_water_canopy_storage!(
                    water_storage, throughfall,
                    prec_band, cv_gpu, canopy_evaporation,
                    max_water_storage, coverage_gpu
                )
            end

            # ============================================================
            # Snow Dynamics — 4D per-(band × veg) tile, matching VIC architecture
            # ============================================================
            @timeit to "calculate_snow_dynamics!" begin
                if enable_snow
                    # Compute mean latitude for hemisphere detection
                    lat_mean_val = mean(lat_cpu)

                    # 4D snow kernel: partitions throughfall[b,v] per tile internally
                    calculate_snow_dynamics!(
                        swe_gpu, snow_depth_gpu, snow_albedo_gpu, snow_surf_temp_gpu,
                        snow_coverage_gpu, snow_melt_gpu,
                        last_snow_gpu, cold_content_gpu, pack_cc_gpu, melting_flag_gpu,
                        store_snow_gpu, snow_distrib_slope_gpu,
                        store_swq_gpu, store_coverage_gpu, max_snow_depth_gpu,
                        throughfall, tair_band, swdown_gpu, lwdown_gpu, psurf_gpu, vp_gpu,
                        AreaFract_gpu, cv_gpu, annual_prec_gpu,
                        day, Float64(lat_mean_val)
                    )

                    # Aggregate 4D→3D for soil model input:
                    # ppt_gpu[i,j,b] = sum_v( (rain[b,v] + melt[b,v]) * Cv[v] )
                    # where rain[b,v] = throughfall_pre_snow[b,v] * rain_frac(tair_band[b])
                    # We saved throughfall BEFORE snow (it was already overwritten above
                    # by the canopy step), so rain = throughfall * rain_frac
                    ft = FloatType

                    # rain_frac per band (3D): same formula as inside kernel
                    _rf_3d = clamp.(
                        (tair_band .- ft(-0.5)) ./ ft(1.0),
                        ft(0.0), ft(1.0)
                    )

                    # rain 4D: throughfall[b,v] * rain_frac[b]  (broadcast last dim)
                    _rain_4d = throughfall .* reshape(_rf_3d, size(_rf_3d, 1), size(_rf_3d, 2), size(_rf_3d, 3), 1)

                    # Cv-weighted aggregation over veg dim (dim=4)
                    # cv_gpu is (nx,ny,1,nveg), broadcasts over band dim automatically
                    rain_band_gpu .= dropdims(
                        sum(ifelse.(isnan.(_rain_4d .* cv_gpu), ft(0.0), _rain_4d .* cv_gpu), dims=4),
                        dims=4)
                    melt_band_gpu .= dropdims(
                        sum(ifelse.(isnan.(snow_melt_gpu .* cv_gpu), ft(0.0), snow_melt_gpu .* cv_gpu), dims=4),
                        dims=4)

                    # Total per-band soil influx: rain + melt
                    ppt_gpu .= rain_band_gpu .+ melt_band_gpu

                    # Broadcast back to 4D throughfall for downstream soil/runoff modules
                    # (they expect throughfall[b,v] = same water input for all veg tiles)
                    nx_s, ny_s, nb_s = size(ppt_gpu)
                    nv_s = size(throughfall, 4)
                    throughfall .= repeat(reshape(ppt_gpu, nx_s, ny_s, nb_s, 1), 1, 1, 1, nv_s)
                else
                    @. ppt_gpu = sum(throughfall, dims=(3,4))
                end
            end

            # Removed erroneous 13-argument calculate_infiltration! block

            # ============================================================
            # Soil evaporation
            # ============================================================
            @timeit to "calculate_soil_evaporation" begin
                calculate_soil_evaporation!(
                    soil_evaporation,
                    soil_moisture, soil_moisture_max, pe_soil,  # Step 2 (snow-blended) PE
                    b_infilt_gpu, cv_gpu, coverage_gpu, residual_moisture, AreaFract_gpu
                )
            end

            # (update_water_canopy_storage! already ran before snow dynamics above)

            @timeit to "calculate_surface_runoff" begin
                calculate_surface_runoff!(
                    surface_runoff, asat,
                    throughfall, soil_moisture,
                    soil_moisture_max, b_infilt_gpu, cv_gpu, AreaFract_gpu
                )
            end

            @timeit to "calculate_infiltration" begin
                calculate_infiltration!(infiltration, throughfall, surface_runoff, cv_gpu)
            end

            # ============================================================
            # Soil moisture update
            # ============================================================
            @timeit to "transpiration_grid" begin
                transpiration_grid = sum(transpiration_layers .* coverage_gpu, dims=4)
            end

            @timeit to "solve_runoff_and_drainage" begin
                solve_runoff_and_drainage!(
                    soil_moisture, subsurface_runoff, surface_runoff, interlayer_drainage,
                    infiltration, soil_evaporation, transpiration_grid,
                    soil_moisture_max, ksat_gpu, residual_moisture, expt_gpu,
                    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu
                )
            end

            # ============================================================
            # Total fluxes
            # ============================================================
            @timeit to "compute_total_fluxes" begin
                calculate_total_evapotranspiration!(
                    total_et,
                    canopy_evaporation, transpiration, soil_evaporation,
                    cv_gpu, coverage_gpu, AreaFract_gpu
                )
                calculate_total_runoff!(
                    total_runoff,
                    surface_runoff,
                    subsurface_runoff
                )
            end

            # ============================================================
            # Routing
            # ============================================================
            if enable_routing
                @timeit to "run_routing" begin
                    run_routing_step!(
                        routing_state,
                        total_runoff,
                        day_sec
                    )
                end
            end

            # ============================================================
            # Soil thermal properties & soil temperature
            # ============================================================
            @timeit to "soil_conductivity" begin
                soil_conductivity!(
                    kappa_array, soil_moisture, ice_frac,
                    soil_dens_min, bulk_dens_min, quartz_gpu,
                    organic_frac_gpu, porosity
                )
            end

            @timeit to "volumetric_heat_capacity" begin
                volumetric_heat_capacity!(
                    cs_array, bulk_dens_gpu, soil_dens_gpu, soil_moisture,
                    rho_w, ice_frac, organic_frac
                )
            end

            @timeit to "estimate_layer_temperature" begin
                estimate_layer_temperature!(
                    soil_temperature, depth_gpu, dp_gpu, tsurf, Tavg_gpu
                )
            end

            # ============================================================
            # Surface temperature 
            # ============================================================
            if day == 1 && year == start_year
                @timeit to "solve_surface_temperature_init" begin
                    solve_surface_temperature!(
                        tsurf,
                        soil_temperature, albedo_gpu, swdown_gpu, lwdown_gpu,
                        aerodynamic_resistance,
                        kappa_array, depth_gpu, day_sec, cs_array, total_et,
                        tair_gpu, cv_gpu, psurf_gpu, AreaFract_gpu
                    )
                end

                @timeit to "compute_aerodynamic_resistance" begin
                    compute_aerodynamic_resistance!(
                        aerodynamic_resistance,
                        z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_band, wind_gpu, cv_gpu
                    )
                end
            end

            @timeit to "solve_surface_temperature" begin
                solve_surface_temperature!(
                    tsurf, soil_temperature, albedo_gpu, swdown_gpu, lwdown_gpu,
                    aerodynamic_resistance,
                    kappa_array, depth_gpu, day_sec, cs_array, total_et,
                    tair_gpu, cv_gpu, psurf_gpu, AreaFract_gpu
                )
            end

            # ============================================================
            # Preprocess & Write Outputs
            # ============================================================
            # Correct logging outputs by recalculating Radiation dynamically closing the explicit current-day temperature offset mirroring the VIC closures.
            @timeit to "calculate_net_radiation_post_closure" begin
                calculate_net_radiation!(
                    net_radiation, swdown_gpu, lwdown_gpu, albedo_gpu, tsurf,
                    snow_coverage_gpu, snow_albedo_gpu, snow_surf_temp_gpu
                )
            end
            @timeit to "preprocess_daily_data" begin
                gpu_results = preprocess_daily_outputs(
                    day, tsurf, tair_gpu, prec_gpu,
                    total_et, surface_runoff, total_runoff,
                    soil_evaporation, soil_moisture,
                    potential_evaporation, net_radiation, transpiration, canopy_evaporation, water_storage,
                    coverage_gpu, cv_gpu, fillvalue_threshold,
                    swe_gpu, snow_albedo_gpu, snow_surf_temp_gpu, snow_coverage_gpu, snow_melt_gpu,
                    AreaFract_gpu
                )
            end

            @timeit to "outputs" begin
                
                # Get a free buffer from the pool 
                # (Instant unless disk is >4 days behind)
                local current_buf
                @timeit to "wait_for_buffer" begin
                    current_buf = get_free_buffer(io_service)
                end

                # Transfer GPU -> CPU (RAM copy)
                @timeit to "gpu_transfer" begin
                    async_transfer!(gpu_results, current_buf)
                end
            
                # Hand off to background thread and continue simulation immediately.
                @timeit to "async_submit" begin
                    submit_buffer(io_service, day, current_buf)
                end
            end

            day_prev = day
            month_prev = month
        end
    end

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------
    println("Waiting for pending background writes...")
    stop_async_service(io_service) # Waits for the last few days to be written
    
    println("Closing output file...")
    @timeit to "closing outputfile" close_output(output_store)

    println("============ Completed run for year: $year ============\n")
    flush(stdout)
end

# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================

for year in start_year:end_year
    if has_input_files(year)
        process_year(year)

        show(to)

        @timeit to "garbage collection" begin
            Base.GC.gc()

            if backend_name == "CUDA"
                CUDA.reclaim()
            end
        end
    else
        println("Skipping year $year due to missing input files.")
    end
end

println("Done!")