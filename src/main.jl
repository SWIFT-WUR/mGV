using .SimConstants

const to = TimerOutputs.TimerOutput()

# ============================================================================
# INITIALIZATION
# ============================================================================

println("Loading parameter data and allocating memory...")

@time begin
    @load_params(
        lat, lon, d0, z0, z0soil, LAI, albedo, rmin, rarc, cv, elev,
        ksat, residmoist, init_moist, root, Wcr, Wfc, Wpwp, depth,
        quartz, bulk_dens, soil_dens, expt, coverage, b_infilt,
        Ds, Dsmax, Ws, dp, Tavg, c_expt
    )
end

@timeit to "gpu_load_static_inputs" @time gpu_load_static_inputs(@vars(
    rmin, rarc, cv, elev, ksat, residmoist, init_moist, root, Wcr, Wfc, Wpwp,
    depth, quartz, bulk_dens, soil_dens, expt, b_infilt, Ds, Dsmax, Ws, dp, Tavg, z0soil
)...)

@timeit to "init_routing" begin
    const routing_state = initialize_routing_model(routing_param_file)
end

@timeit to "reshaping_inputs" begin
    reshape_static_inputs!()
end

# ============================================================================
# INITIALIZE GPU STATE ARRAYS
# ============================================================================
println("Julia has started with $(Threads.nthreads()) threads.")
println("Allocating State Arrays on: $backend_name")

@timeit to "initialize_GPU_arrays" begin

    # Helper for brevity
    alloc(dims...) = KernelAbstractions.zeros(device_backend, FloatType, dims...)

    # Canopy and surface states
    const water_storage = alloc(size(coverage_gpu))
    const max_water_storage = alloc(size(coverage_gpu))
    const throughfall = alloc(size(coverage_gpu))
    const canopy_evaporation = alloc(size(coverage_gpu))
    const f_n = alloc(size(coverage_gpu))

    const net_radiation = alloc(size(coverage_gpu))
    const Q_12 = alloc(size(Tavg_gpu))
    const potential_evaporation = alloc(size(coverage_gpu))
    const aerodynamic_resistance = alloc(size(coverage_gpu))
    const tsurf = alloc(size(Tavg_gpu))

    const soil_evaporation = alloc(size(Tavg_gpu))
    const total_et = alloc(size(Tavg_gpu))
    const infiltration = alloc(size(Tavg_gpu))

    const surface_runoff = alloc(size(Tavg_gpu))
    const asat = alloc(size(Tavg_gpu))

    const subsurface_runoff = alloc(size(Tavg_gpu))
    const interlayer_drainage = alloc(size(Tavg_gpu, 1), size(Tavg_gpu, 2), 2)
    const total_runoff = alloc(size(Tavg_gpu))

    # Soil property arrays
    const bulk_dens_min = alloc(size(bulk_dens_gpu))
    const soil_dens_min = alloc(size(bulk_dens_gpu))
    const porosity = alloc(size(bulk_dens_gpu))
    const Lsum = alloc(size(soil_dens_gpu))

    # Soil temperature initialization
    const soil_temperature = alloc(size(soil_dens_gpu))
    soil_temperature[:, :, 1:1] .= Tavg_gpu
    soil_temperature[:, :, 2:2] .= Tavg_gpu
    soil_temperature[:, :, 3:3] .= Tavg_gpu

    # Soil moisture arrays (3D: lon, lat, layer)
    soil_dims = (size(soil_dens_gpu, 1), size(soil_dens_gpu, 2), size(soil_dens_gpu, 3))

    const soil_moisture = alloc(soil_dims...)
    const soil_moisture_max = alloc(soil_dims...)
    const soil_moisture_critical = alloc(soil_dims...)

    const kappa_array = alloc(soil_dims...)
    const cs_array = alloc(soil_dims...)

    const field_capacity = alloc(soil_dims...)
    const wilting_point = alloc(soil_dims...)
    const residual_moisture = alloc(soil_dims...)

    const ice_frac = alloc(size(soil_moisture))
    const organic_frac_gpu = alloc(size(soil_moisture))
    fill!(organic_frac_gpu, FloatType(organic_frac))

    # --- Pre allocation for transpiration ---
    const transpiration = alloc(size(coverage_gpu))

    # transpiration_layers: (ny, nx, 3, nveg)
    const transpiration_layers = alloc(size(coverage_gpu, 1), size(coverage_gpu, 2), 3, size(coverage_gpu, 4))

    # Layer splits: (ny, nx, 1, nveg)
    const E_1_t = alloc(size(coverage_gpu))
    const E_2_t = alloc(size(coverage_gpu))

    # Intermediate buffers for stress and conductance
    const g1_buf = alloc(size(Tavg_gpu))
    const g2_buf = alloc(size(Tavg_gpu))

    # g_sw_veg: (ny, nx, 1, nveg-1)
    veg_dim = size(root_gpu, 4)
    const g_sw_veg_buf = alloc(size(coverage_gpu, 1), size(coverage_gpu, 2), 1, veg_dim)

    const dry_time_factor = alloc(size(coverage_gpu))
    # ---
end

# ============================================================================
# CALCULATE SOIL PROPERTIES
# ============================================================================

@timeit to "calculate_soil_properties" begin
    _bulk_dens_min, _soil_dens_min, _porosity, _soil_moisture_max,
    _soil_moisture_critical, _field_capacity, _wilting_point, _residual_moisture =
        calculate_soil_properties(
            bulk_dens_gpu, soil_dens_gpu, depth_gpu,
            Wcr_gpu, Wfc_gpu, Wpwp_gpu, residmoist_gpu
        )
end


@timeit to "copy_soil_properties" begin
    # Copy calculated values into constant/global GPU arrays
    bulk_dens_min .= _bulk_dens_min
    soil_dens_min .= _soil_dens_min
    porosity .= _porosity
    soil_moisture_max .= _soil_moisture_max
    soil_moisture_critical .= _soil_moisture_critical
    field_capacity .= _field_capacity
    wilting_point .= _wilting_point
    residual_moisture .= _residual_moisture

    # Initialize soil moisture within physical bounds
    soil_moisture .= min.(init_moist_gpu, soil_moisture_max)
    soil_moisture .= max.(soil_moisture, residual_moisture)
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
    local transfer_buf

    @timeit to "create_output" begin
        nx, ny = size(prec_cpu, 1), size(prec_cpu, 2)
        nt = size(prec_cpu, 3)
        nlayers = 3

        if output_format == :netcdf
            output_path = joinpath(output_dir, "$(output_file_prefix)$(year).nc")
            # Returns NetCDFOutputStore and Buffer
            output_store, transfer_buf = create_output_netcdf(output_path, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
        else
            output_path = joinpath(output_dir, "$(output_file_prefix)$(year).zarr")
            # Returns ZarrOutputStore and Buffer
            output_store, transfer_buf = create_output_zarr(output_path, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
        end
    end

    # Create a stream for data transfers
    transfer_stream = create_stream()

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
            @timeit to "compute_aerodynamic_resistance" begin
                compute_aerodynamic_resistance!(
                    aerodynamic_resistance,
                    z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu
                )
            end

            @timeit to "calculate_net_radiation" begin
                calculate_net_radiation!(
                    net_radiation, swdown_gpu, lwdown_gpu, albedo_gpu, tsurf
                )
            end

            @timeit to "calculate_potential_evaporation" begin
                calculate_potential_evaporation!(
                    potential_evaporation,
                    tair_gpu, psurf_gpu, vp_gpu, elev_gpu, net_radiation,
                    aerodynamic_resistance, rarc_gpu, rmin_gpu, LAI_gpu
                )
            end

            # ============================================================
            # Canopy processes
            # ============================================================
            @timeit to "calculate_max_water_storage" begin
                calculate_max_water_storage!(max_water_storage, LAI_gpu, cv_gpu, coverage_gpu)
            end

            @timeit to "calculate_canopy_evaporation" begin
                calculate_canopy_evaporation!(
                    canopy_evaporation, f_n,
                    water_storage, max_water_storage, potential_evaporation,
                    aerodynamic_resistance, rarc_gpu, prec_gpu, cv_gpu,
                    rmin_gpu, LAI_gpu, tair_gpu, elev_gpu
                )
            end

            # ============================================================
            # Transpiration
            # ============================================================
            @timeit to "calculate_transpiration" begin
                calculate_transpiration!(
                    # Outputs
                    transpiration, transpiration_layers, E_1_t, E_2_t,
                    g1_buf, g2_buf, g_sw_veg_buf, dry_time_factor,
                    # Inputs
                    potential_evaporation, aerodynamic_resistance, rarc_gpu,
                    water_storage, max_water_storage, soil_moisture,
                    soil_moisture_critical, wilting_point, root_gpu,
                    rmin_gpu, LAI_gpu, cv_gpu, f_n
                )
            end

            # ============================================================
            # Soil evaporation
            # ============================================================
            @timeit to "calculate_soil_evaporation" begin
                calculate_soil_evaporation!(
                    soil_evaporation,
                    soil_moisture, soil_moisture_max, potential_evaporation,
                    b_infilt_gpu, cv_gpu, coverage_gpu, residual_moisture
                )
            end

            # ============================================================
            # Water balance: throughfall and runoff
            # ============================================================
            @timeit to "update_water_canopy_storage" begin
                update_water_canopy_storage!(
                    water_storage, throughfall,
                    prec_gpu, cv_gpu, canopy_evaporation,
                    max_water_storage, coverage_gpu
                )
            end

            @timeit to "calculate_surface_runoff" begin
                calculate_surface_runoff!(
                    surface_runoff, asat,
                    prec_gpu, throughfall, soil_moisture,
                    soil_moisture_max, b_infilt_gpu, cv_gpu
                )
            end

            calculate_infiltration!(infiltration, throughfall, surface_runoff)

            # ============================================================
            # Soil moisture update
            # ============================================================
            # Weight for soil water removal
            transpiration_grid = sum(transpiration .* coverage_gpu, dims=4)
            #transpiration_grid = sum(transpiration_layers .* coverage_gpu, dims=4)

            @timeit to "solve_runoff_and_drainage" begin
                solve_runoff_and_drainage!(
                    soil_moisture, subsurface_runoff, interlayer_drainage,
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
                    cv_gpu, coverage_gpu
                )
                calculate_total_runoff!(
                    total_runoff,
                    surface_runoff,
                    subsurface_runoff,
                    fillvalue_threshold
                )
            end

            # ============================================================
            # Routing
            # ============================================================
            @timeit to "run_routing" begin
                run_routing_step!(
                    routing_state,
                    total_runoff,
                    day_sec
                )
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
                        sum_with_nan_handling(cv_gpu .* aerodynamic_resistance, 4),
                        kappa_array, depth_gpu, day_sec, cs_array, total_et,
                        tair_gpu, cv_gpu, psurf_gpu
                    )
                end

                @timeit to "compute_aerodynamic_resistance" begin
                    compute_aerodynamic_resistance!(
                        aerodynamic_resistance,
                        z2, d0_gpu, z0_gpu, z0soil_gpu, tsurf, tair_gpu, wind_gpu, cv_gpu
                    )
                end
            end

            @timeit to "solve_surface_temperature" begin
                solve_surface_temperature!(
                    tsurf, soil_temperature, albedo_gpu, swdown_gpu, lwdown_gpu,
                    aerodynamic_resistance,
                    kappa_array, depth_gpu, day_sec, cs_array, total_et,
                    tair_gpu, cv_gpu, psurf_gpu
                )
            end

            # ============================================================
            # Preprocess & Write Outputs
            # ============================================================
            @timeit to "preprocess_daily_data" begin
                gpu_results = preprocess_daily_outputs(
                    day, tsurf, tair_gpu, prec_gpu,
                    total_et, surface_runoff, total_runoff,
                    soil_evaporation, soil_moisture,
                    potential_evaporation, net_radiation, transpiration, canopy_evaporation,
                    coverage_gpu, cv_gpu, fillvalue_threshold
                )
            end

            @timeit to "outputs" begin
                async_transfer!(gpu_results, transfer_buf, transfer_stream)
                write_slice!(day, transfer_buf, transfer_stream, output_store)
            end

            day_prev = day
            month_prev = month
        end 
    end 

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------
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
            elseif backend_name == "AMDGPU"
                AMDGPU.reclaim()
            end
        end
    else
        println("Skipping year $year due to missing input files.")
    end
end

println("Done!")