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

@timeit to "reshaping_inputs" begin
    reshape_static_inputs!()
end

# ============================================================================
# INITIALIZE GPU STATE ARRAYS
# ============================================================================
println("DEBUG: Julia has started with $(Threads.nthreads()) threads.")

@timeit to "initialize_GPU_arrays" begin

# Canopy and surface states
global water_storage = CUDA.zeros(float_type, size(coverage_gpu))
global max_water_storage = CUDA.zeros(float_type, size(coverage_gpu))
global throughfall = CUDA.zeros(float_type, size(coverage_gpu))
global canopy_evaporation = CUDA.zeros(float_type, size(coverage_gpu))
global f_n = CUDA.zeros(float_type, size(coverage_gpu)) 

global net_radiation = CUDA.zeros(float_type, size(coverage_gpu))
global Q_12 = CUDA.zeros(float_type, size(Tavg_gpu))
global potential_evaporation = CUDA.zeros(float_type, size(coverage_gpu))
global aerodynamic_resistance = CUDA.zeros(float_type, size(coverage_gpu))
global tsurf = CUDA.zeros(float_type, size(Tavg_gpu))


global soil_evaporation = CUDA.zeros(float_type, size(Tavg_gpu))
global total_et = CUDA.zeros(float_type, size(Tavg_gpu))
global infiltration = CUDA.zeros(float_type, size(Tavg_gpu))

# Soil property arrays
global bulk_dens_min = CUDA.zeros(float_type, size(bulk_dens_gpu))
global soil_dens_min = CUDA.zeros(float_type, size(bulk_dens_gpu))
global porosity = CUDA.zeros(float_type, size(bulk_dens_gpu))
global Lsum = CUDA.zeros(float_type, size(soil_dens_gpu))

# Soil temperature initialization
global soil_temperature = CUDA.zeros(float_type, size(soil_dens_gpu))
soil_temperature[:, :, 1:1] = Tavg_gpu
soil_temperature[:, :, 2:2] = Tavg_gpu
soil_temperature[:, :, 3:3] = Tavg_gpu

# Soil moisture arrays (3D: lon, lat, layer)
soil_dims = (size(soil_dens_gpu, 1), size(soil_dens_gpu, 2), size(soil_dens_gpu, 3))
global soil_moisture_old = CUDA.zeros(float_type, soil_dims...)
global soil_moisture_new = CUDA.zeros(float_type, soil_dims...)
global soil_moisture_max = CUDA.zeros(float_type, soil_dims...)
global soil_moisture_critical = CUDA.zeros(float_type, soil_dims...)
global field_capacity = CUDA.zeros(float_type, soil_dims...)
global wilting_point = CUDA.zeros(float_type, soil_dims...)
global residual_moisture = CUDA.zeros(float_type, soil_dims...)

global ice_frac = CUDA.zeros(float_type, size(soil_moisture_old))
global organic_frac_gpu = CUDA.fill(float_type(organic_frac), size(soil_moisture_old))
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

# Copy calculated values into global GPU arrays
bulk_dens_min .= _bulk_dens_min
soil_dens_min .= _soil_dens_min
porosity .= _porosity
soil_moisture_max .= _soil_moisture_max
soil_moisture_critical .= _soil_moisture_critical
field_capacity .= _field_capacity
wilting_point .= _wilting_point
residual_moisture .= _residual_moisture

# Initialize soil moisture within physical bounds
soil_moisture_old .= min.(init_moist_gpu, soil_moisture_max)
soil_moisture_old .= max.(soil_moisture_old, residual_moisture)
soil_moisture_new .= copy(soil_moisture_old)

end

# Print diagnostics
@debug println("Wmax (mean) [mm]: L1=", mean(Array(soil_moisture_max[:,:,1])),
                         " L2=", mean(Array(soil_moisture_max[:,:,2])),
                         " L3=", mean(Array(soil_moisture_max[:,:,3])))
@debug println("Ksat (median) [mm/day]: L1=", median(Array(ksat_gpu[:,:,1])),
                                   " L2=", median(Array(ksat_gpu[:,:,2])),
                                   " L3=", median(Array(ksat_gpu[:,:,3])))

# ============================================================================
# PROCESS YEAR FUNCTION
# ============================================================================

function process_year(year)
    # Ensure we're modifying global variables
    @timeit to "take_in_global_arrays" begin

    global water_storage, throughfall, canopy_evaporation, bulk_dens_min, soil_dens_min
    global porosity, soil_moisture_old, Q_12, soil_moisture_new, soil_moisture_max
    global soil_moisture_critical, field_capacity, wilting_point, residual_moisture
    global soil_temperature, Lsum, tsurf
    global net_radiation
    global potential_evaporation, aerodynamic_resistance
    global max_water_storage
    global ice_frac, organic_frac_gpu
    global total_et
    
    end

    println("============ Start run for year: $year ============")
    
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
    @debug println("Opening output Zarr store...")
    @timeit to "create_output_zarr" begin
        # Change extension to .zarr
        output_path = joinpath(output_dir, "$(output_file_prefix)$(year).zarr")
        
        nx, ny = size(prec_cpu, 1), size(prec_cpu, 2)
        nt = size(prec_cpu, 3)
        nlayers = 3 # Hardcoded based on your usage
        
        # Initialize Store and Buffer
        zarr_store, transfer_buf = create_output_zarr(output_path, nx, ny, nt, nlayers, lat_cpu, lon_cpu)
    end
   

    # Create a stream for data transfers
    transfer_stream = CuStream()

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
            
            if GPU_USE == true
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
                    transpiration, transpiration_layers, E_1_t, E_2_t, g_sw_1, g_sw_2, g_sw, dry_time_factor = 
                        calculate_transpiration(
                            potential_evaporation, aerodynamic_resistance, rarc_gpu, 
                            water_storage, max_water_storage, soil_moisture_old, 
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
                        soil_moisture_old, soil_moisture_max, potential_evaporation, 
                        b_infilt_gpu, cv_gpu, coverage_gpu, residual_moisture
                    )
                end

                # ============================================================
                # Water balance: throughfall and runoff
                # ============================================================
                @timeit to "update_water_canopy_storage" begin
                    # Mutating call: updates water_storage and throughfall in-place
                    update_water_canopy_storage!(
                        water_storage, throughfall, 
                        prec_gpu, cv_gpu, canopy_evaporation, 
                        max_water_storage, coverage_gpu
                    )
                end

                # Calculate surface runoff
                surface_runoff, asat = calculate_surface_runoff(
                    prec_gpu, throughfall, soil_moisture_old, 
                    soil_moisture_max, b_infilt_gpu, cv_gpu
                )

                # Calculate infiltration
                calculate_infiltration!(infiltration, throughfall, surface_runoff)

#                total_input = sum_with_nan_handling(throughfall, 4)
#                infiltration_raw = total_input .- surface_runoff
#                infiltration = max.(infiltration_raw, zero(eltype(infiltration_raw)))

                @debug println("neg. infiltration cells = ", count(<(0), Array(infiltration_raw)))

                # ============================================================
                # Soil moisture update
                # ============================================================
                # Weight for soil water removal
                transpiration_grid = sum(transpiration .* coverage_gpu, dims=4)
                #transpiration_grid = sum(transpiration_layers .* coverage_gpu, dims=4)

                soil_moisture_new, subsurface_runoff, Q12 = solve_runoff_and_drainage(
                    infiltration, soil_evaporation, transpiration_grid, soil_moisture_old,
                    soil_moisture_max, ksat_gpu, residual_moisture, expt_gpu,
                    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu
                )

                soil_moisture_old = soil_moisture_new

                # ============================================================
                # Total fluxes
                # ============================================================
                @timeit to "compute_total_fluxes" begin
                    calculate_total_evapotranspiration!(
                        total_et,
                        canopy_evaporation, transpiration, soil_evaporation, 
                        cv_gpu, coverage_gpu
                    )
                    total_runoff = calculate_total_runoff(
                        surface_runoff, subsurface_runoff, cv_gpu
                    )
                end

                # ============================================================
                # Soil thermal properties
                # ============================================================
                @timeit to "soil_conductivity" begin
                    kappa_array = soil_conductivity(
                        soil_moisture_new, ice_frac, soil_dens_min, bulk_dens_min, 
                        quartz_gpu, organic_frac_gpu, porosity
                    )
                end

                @timeit to "volumetric_heat_capacity" begin
                    cs_array = volumetric_heat_capacity(
                        bulk_dens_gpu ./ soil_dens_gpu, 
                        soil_moisture_new ./ rho_w, ice_frac, organic_frac
                    )
                end

                @timeit to "estimate_layer_temperature" begin
                    soil_temperature = estimate_layer_temperature(
                        depth_gpu, dp_gpu, tsurf, soil_temperature, Tavg_gpu
                    )
                end

                # ============================================================
                # Surface temperature solution
                # ============================================================
                if day == 1 && year == start_year
                    @timeit to "solve_surface_temperature" begin
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

                # Calculate effective aerodynamic resistance
                ra_eff_inv = sum(cv_gpu ./ aerodynamic_resistance, dims=4)
                ra_eff = 1.0 ./ max.(ra_eff_inv, eps(eltype(ra_eff_inv)))

                @timeit to "solve_surface_temperature" begin
                    solve_surface_temperature!(
                        tsurf,
                        soil_temperature, albedo_gpu, swdown_gpu, lwdown_gpu,
                        ra_eff, 
                        kappa_array, depth_gpu, day_sec, cs_array, total_et, 
                        tair_gpu, cv_gpu, psurf_gpu
                    )
                end

                # ============================================================
                # Spike diagnostics (selected days)
                # ============================================================
                if day in [20, 120, 200, 320]
                    run_spike_diagnostics(day, transpiration, soil_moisture_old, 
                                        soil_moisture_critical, wilting_point, root_gpu, 
                                        cv_gpu, water_storage, max_water_storage, f_n, 
                                        potential_evaporation)
                end

                # External debug for day 120
                if day == 120
                    run_external_debug(day, g_sw_1, g_sw_2, root_gpu, transpiration)
                end

                @timeit to "preprocess_daily_data" begin gpu_results = preprocess_daily_outputs(
                        day, tsurf, tair_gpu, prec_gpu, 
                        total_et, surface_runoff, total_runoff,
                        soil_evaporation, soil_moisture_new,
                        potential_evaporation, net_radiation, transpiration, canopy_evaporation,
                        coverage_gpu, cv_gpu, fillvalue_threshold
                    )
                end

                # ============================================================
                # Write outputs
                # ============================================================
                @debug println("Writing outputs")
                @timeit to "outputs" begin    
                    
                    # 1. Start the transfer for TODAY (Day N)
                    # This queues commands to the GPU stream and returns immediately.
                    async_transfer!(gpu_results, transfer_buf, transfer_stream)
        
                    # 2. Synchronize and Write to Disk
                    # This waits for the GPU transfer to finish, then writes to Zarr.
                    # Because Zarr writes are so fast (chunked), we can do this simply here.
                    write_zarr_slice!(
                        day, 
                        transfer_buf, 
                        transfer_stream,
                        zarr_store
                    )
                end
                
            end # GPU_USE

            day_prev = day
            month_prev = month
        end # timeit process_year
    end # daily loop

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------
    #println("Closing output file...")
    #@timeit to "closing outputfile" close(out_ds)
    
    println("============ Completed run for year: $year ============\n")
    #println("Postprocessing for year $year...")
    #@timeit to "compress_file_async call" compress_file_async(output_file, 1)
    flush(stdout)
end

# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================

for year in start_year:end_year
    if has_input_files(year)
        process_year(year)
        
        show(to)  # Print profiling data
        
        @timeit to "garbage collection" begin
            Base.GC.gc()
            CUDA.reclaim()
        end
    else
        println("Skipping year $year due to missing input files.")
    end
end

println("Done!")