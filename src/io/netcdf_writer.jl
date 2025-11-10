function create_output_netcdf(output_file::String, reference_array, reference_array2, float_type, lat_cpu, lon_cpu)
    println("Creating NetCDF output file...")
    out_ds = NCDataset(output_file, "c")
    
    # Define dimensions based on the reference array’s shape
    defDim(out_ds, "lon",   size(reference_array, 1))
    defDim(out_ds, "lat",   size(reference_array, 2))
    defDim(out_ds, "time",  size(reference_array, 3))
    defDim(out_ds, "nveg",  size(reference_array2, 4))
    defDim(out_ds, "qlayers", 2)
    defDim(out_ds, "layer", 3)
    defDim(out_ds, "top_layer", 1)

    # Define latitude variable and assign values
    lat = defVar(out_ds, "lat", float_type, ("lat",))
    lat.attrib["axis"] = "Y"
    lat.attrib["long_name"] = "latitude"
    lat.attrib["standard_name"] = "latitude"
    lat.attrib["units"] = "degrees_north"
    lat[:] = lat_cpu  # Assign the latitude values

    # Define longitude variable and assign values
    lon = defVar(out_ds, "lon", float_type, ("lon",))
    lon.attrib["axis"] = "X"
    lon.attrib["long_name"] = "longitude"
    lon.attrib["standard_name"] = "longitude"
    lon.attrib["units"] = "degrees_east"
    lon[:] = lon_cpu  # Assign the longitude values

    # Define the output variables to be written
    precipitation_output = defVar(out_ds, "precipitation_output", float_type, ("lon", "lat", "time"))
    precipitation_output.attrib["units"]       = "mm/day"
    precipitation_output.attrib["description"] = "Daily precipitation"
    #precipitation_output.attrib["_FillValue"] = float_type(NaN)

    throughfall_output = defVar(out_ds, "throughfall_output", float_type, ("lon", "lat", "time", "nveg"))
    throughfall_output.attrib["units"]       = "mm/day"
    throughfall_output.attrib["description"] = "Daily throughfall per vegetation"

    throughfall_summed_output = defVar(out_ds, "throughfall_summed_output", float_type, ("lon", "lat", "time"))
    throughfall_summed_output.attrib["units"]       = "mm/day"
    throughfall_summed_output.attrib["description"] = "Total daily throughfall"

    water_storage_output = defVar(out_ds, "water_storage_output", float_type, ("lon", "lat", "time", "nveg"))
    water_storage_output.attrib["units"] = "mm"
    water_storage_output.attrib["description"] = "Water stored in the canopy per vegetation"
    
    water_storage_summed_output = defVar(out_ds, "water_storage_summed_output", float_type, ("lon", "lat", "time"))
    water_storage_summed_output.attrib["units"] = "mm"
    water_storage_summed_output.attrib["description"] = "Total water stored in the canopy"

    Q12_output = defVar(out_ds, "Q12_output", float_type, ("lon", "lat", "time", "qlayers"))
    Q12_output.attrib["units"] = "mm"
    Q12_output.attrib["description"] = "Interlayer drainage"
    
    tair_output = defVar(out_ds, "tair_output", float_type, ("lon", "lat", "time"), chunksizes = (36, 36, 1))
    tair_output.attrib["units"] = "°C"
    tair_output.attrib["description"] = "Air temperature at reference height"
    tair_output.attrib["_FillValue"] = float_type(1.e20)
    tair_output.attrib["missing_value"] = float_type(1.e20)

    tsurf_output = defVar(out_ds, "tsurf_output", float_type, ("lon", "lat", "time"))
    tsurf_output.attrib["units"] = "°C"
    tsurf_output.attrib["description"] = "Surface temperature per vegetation"
    
#    tsurf_summed_output = defVar(out_ds, "tsurf_summed_output", float_type, ("lon", "lat", "time"))
#    tsurf_summed_output.attrib["units"] = "°C"
#    tsurf_summed_output.attrib["description"] = "Summed surface temperature"
    
    canopy_evaporation_output = defVar(out_ds, "canopy_evaporation_output", float_type, ("lon", "lat", "time", "nveg"))
    canopy_evaporation_output.attrib["units"] = "mm"
    canopy_evaporation_output.attrib["description"] = "Evaporation from canopy interception per vegetation"
    
    canopy_evaporation_summed_output = defVar(out_ds, "canopy_evaporation_summed_output", float_type, ("lon", "lat", "time"))
    canopy_evaporation_summed_output.attrib["units"] = "mm"
    canopy_evaporation_summed_output.attrib["description"] = "Total evaporation from canopy interception"

    transpiration_output = defVar(out_ds, "transpiration_output", float_type, ("lon", "lat", "time", "nveg"))
    transpiration_output.attrib["units"] = "mm"
    transpiration_output.attrib["description"] = "Plant transpiration per vegetation"
    
    transpiration_summed_output = defVar(out_ds, "transpiration_summed_output", float_type, ("lon", "lat", "time"))
    transpiration_summed_output.attrib["units"] = "mm"
    transpiration_summed_output.attrib["description"] = "Total plant transpiration"

    aerodynamic_resistance_output = defVar(out_ds, "aerodynamic_resistance_output", float_type, ("lon", "lat", "time", "nveg"))
    aerodynamic_resistance_output.attrib["units"] = "s/m"
    aerodynamic_resistance_output.attrib["description"] = "Aerodynamic resistance per vegetation"    

    aerodynamic_resistance_summed_output = defVar(out_ds, "aerodynamic_resistance_summed_output", float_type, ("lon", "lat", "time"))
    aerodynamic_resistance_summed_output.attrib["units"] = "s/m"
    aerodynamic_resistance_summed_output.attrib["description"] = "Total aerodynamic resistance"    

    potential_evaporation_output = defVar(out_ds, "potential_evaporation_output", float_type, ("lon", "lat", "time", "nveg"))
    potential_evaporation_output.attrib["units"] = "mm"
    potential_evaporation_output.attrib["description"] = "Potential evaporation per vegetation"

    potential_evaporation_summed_output = defVar(out_ds, "potential_evaporation_summed_output", float_type, ("lon", "lat", "time"))
    potential_evaporation_summed_output.attrib["units"] = "mm"
    potential_evaporation_summed_output.attrib["description"] = "Potential evaporation"

    net_radiation_output = defVar(out_ds, "net_radiation_output", float_type, ("lon", "lat", "time", "nveg"))
    net_radiation_output.attrib["units"] = "W/m^2"
    net_radiation_output.attrib["description"] = "Net radiation, per vegetation"

    net_radiation_summed_output = defVar(out_ds, "net_radiation_summed_output", float_type, ("lon", "lat", "time"))
    net_radiation_summed_output.attrib["units"] = "W/m^2"
    net_radiation_summed_output.attrib["description"] = "Net radiation"

    max_water_storage_output = defVar(out_ds, "max_water_storage_output", float_type, ("lon", "lat", "time", "nveg"))
    max_water_storage_output.attrib["units"] = "mm"
    max_water_storage_output.attrib["description"] = "The maximum amount of water intercepted by the canopy per vegetation"

    max_water_storage_summed_output = defVar(out_ds, "max_water_storage_summed_output", float_type, ("lon", "lat", "time"))
    max_water_storage_summed_output.attrib["units"] = "mm"
    max_water_storage_summed_output.attrib["description"] = "The maximum amount of water intercepted by the canopy"
    
    soil_evaporation_output = defVar(out_ds, "soil_evaporation_output", float_type, ("lon", "lat", "time", "top_layer"))
    soil_evaporation_output.attrib["units"] = "mm"
    soil_evaporation_output.attrib["description"] = "Evaporation from the soil surface per top soil layer"
    
    soil_temperature_output = defVar(out_ds, "soil_temperature_output", float_type, ("lon", "lat", "time", "layer"))
    soil_temperature_output.attrib["units"] = "°C"
    soil_temperature_output.attrib["description"] = "Soil temperature per layer"
    
    soil_moisture_output = defVar(out_ds, "soil_moisture_output", float_type, ("lon", "lat", "time", "layer"))
    soil_moisture_output.attrib["units"] = "kg/m^3"
    soil_moisture_output.attrib["description"] = "Volumetric soil moisture content per layer"

    total_et_output = defVar(out_ds, "total_et_output", float_type, ("lon", "lat", "time"))
    total_et_output.attrib["units"] = "mm"
    total_et_output.attrib["description"] = "Total evapotranspiration, weighted sum of canopy evaporation, transpiration, and bare soil evaporation across all cover classes"
    
    surface_runoff_output = defVar(out_ds, "surface_runoff_output", float_type, ("lon", "lat", "time"))
    surface_runoff_output.attrib["units"] = "mm"
    surface_runoff_output.attrib["description"] = "Surface runoff, weighted sum of surface runoff across all cover classes"

    total_runoff_output = defVar(out_ds, "total_runoff_output", float_type, ("lon", "lat", "time"))
    total_runoff_output.attrib["units"] = "mm"
    total_runoff_output.attrib["description"] = "Total runoff, weighted sum of surface and subsurface runoff across all cover classes"

    kappa_array_output = defVar(out_ds, "kappa_array_output", float_type, ("lon", "lat", "time", "layer"))
    cs_array_output = defVar(out_ds, "cs_array_output", float_type, ("lon", "lat", "time", "layer"))

    wilting_point_output = defVar(out_ds, "wilting_point_output", float_type, ("lon", "lat", "layer"))
    soil_moisture_max_output = defVar(out_ds, "soil_moisture_max_output", float_type, ("lon", "lat", "layer"))
    soil_moisture_critical_output = defVar(out_ds, "soil_moisture_critical_output", float_type, ("lon", "lat", "layer"))

    residual_moisture_output = defVar(out_ds, "residual_moisture_output", float_type, ("lon", "lat", "time", "layer"))

    E_1_t_output = defVar(out_ds, "E_1_t_output", float_type, ("lon", "lat", "time", "nveg"))
    E_2_t_output = defVar(out_ds, "E_2_t_output", float_type, ("lon", "lat", "time", "nveg"))
    g_sw_1_output = defVar(out_ds, "g_sw_1_output", float_type, ("lon", "lat", "time"))
    g_sw_2_output = defVar(out_ds, "g_sw_2_output", float_type, ("lon", "lat", "time"))

    g_sw_output = defVar(out_ds, "g_sw_output", float_type, ("lon", "lat", "time", "nveg"))
    g_sw_output.attrib["units"] = ""
    g_sw_output.attrib["description"] = "g_sw_output"

    g_sw_summed_output = defVar(out_ds, "g_sw_summed_output", float_type, ("lon", "lat", "time"))
    g_sw_summed_output.attrib["units"] = ""
    g_sw_summed_output.attrib["description"] = "g_sw_summed_output"

    g_sw_1_summed_output = defVar(out_ds, "g_sw_1_summed_output", float_type, ("lon", "lat", "time"))
    g_sw_1_summed_output.attrib["units"] = ""
    g_sw_1_summed_output.attrib["description"] = "g_sw_1_summed_output"

    g_sw_2_summed_output = defVar(out_ds, "g_sw_2_summed_output", float_type, ("lon", "lat", "time"))
    g_sw_2_summed_output.attrib["units"] = ""
    g_sw_2_summed_output.attrib["description"] = "g_sw_2_summed_output"

    

    dry_time_factor_output = defVar(out_ds, "dry_time_factor_output", float_type, ("lon", "lat", "time", "nveg"))
    dry_time_factor_output.attrib["units"] = ""
    dry_time_factor_output.attrib["description"] = "dry_time_factor_output"

    topsoil_moisture_addition_output = defVar(out_ds, "topsoil_moisture_addition_output", float_type, ("lon", "lat", "time"))

    # New output variables
    delintercept_output = defVar(out_ds, "delintercept_output", float_type, ("lon", "lat", "time", "nveg"))
    delintercept_output.attrib["units"] = "mm"
    delintercept_output.attrib["description"] = "Change in canopy interception storage"

    inflow_output = defVar(out_ds, "inflow_output", float_type, ("lon", "lat", "time", "nveg"))
    inflow_output.attrib["units"] = "mm"
    inflow_output.attrib["description"] = "Water inflow to soil layers"

    surfstor_output = defVar(out_ds, "surfstor_output", float_type, ("lon", "lat", "time", "nveg"))
    surfstor_output.attrib["units"] = "mm"
    surfstor_output.attrib["description"] = "Surface water storage"

    delsurfstor_output = defVar(out_ds, "delsurfstor_output", float_type, ("lon", "lat", "time", "nveg"))
    delsurfstor_output.attrib["units"] = "mm"
    delsurfstor_output.attrib["description"] = "Change in surface water storage"

    delsoilmoist_output = defVar(out_ds, "delsoilmoist_output", float_type, ("lon", "lat", "time", "layer"))
    delsoilmoist_output.attrib["units"] = "kg/m^3"
    delsoilmoist_output.attrib["description"] = "Change in soil moisture"

    asat_output = defVar(out_ds, "asat_output", float_type, ("lon", "lat", "time"))
    asat_output.attrib["units"] = "fraction"
    asat_output.attrib["description"] = "Fraction of saturated area"

    latent_output = defVar(out_ds, "latent_output", float_type, ("lon", "lat", "time", "nveg"))
    latent_output.attrib["units"] = "W/m^2"
    latent_output.attrib["description"] = "Latent heat flux"

    sensible_output = defVar(out_ds, "sensible_output", float_type, ("lon", "lat", "time", "nveg"))
    sensible_output.attrib["units"] = "W/m^2"
    sensible_output.attrib["description"] = "Sensible heat flux"

    grnd_flux_output = defVar(out_ds, "grnd_flux_output", float_type, ("lon", "lat", "time", "nveg"))
    grnd_flux_output.attrib["units"] = "W/m^2"
    grnd_flux_output.attrib["description"] = "Ground heat flux"

    vp_output = defVar(out_ds, "vp_output", float_type, ("lon", "lat", "time"))
    vp_output.attrib["units"] = "kPa"
    vp_output.attrib["description"] = "Vapor pressure"

    vpd_output = defVar(out_ds, "vpd_output", float_type, ("lon", "lat", "time"))
    vpd_output.attrib["units"] = "Pa"
    vpd_output.attrib["description"] = "Vapor pressure deficit"

    surf_cond_output = defVar(out_ds, "surf_cond_output", float_type, ("lon", "lat", "time", "nveg"))
    surf_cond_output.attrib["units"] = "m/s"
    surf_cond_output.attrib["description"] = "Surface conductance"

    density_output = defVar(out_ds, "density_output", float_type, ("lon", "lat", "time"))
    density_output.attrib["units"] = "kg/m^3"
    density_output.attrib["description"] = "Air density"




    return out_ds, precipitation_output, water_storage_output, water_storage_summed_output, Q12_output, 
           tair_output, tsurf_output, canopy_evaporation_output,
           canopy_evaporation_summed_output, transpiration_output, transpiration_summed_output, aerodynamic_resistance_output, aerodynamic_resistance_summed_output,
           potential_evaporation_output, potential_evaporation_summed_output, net_radiation_output,
           net_radiation_summed_output, max_water_storage_output, max_water_storage_summed_output,
           soil_evaporation_output, soil_temperature_output, soil_moisture_output,  total_et_output, surface_runoff_output, total_runoff_output,
           kappa_array_output, cs_array_output, wilting_point_output, soil_moisture_max_output, soil_moisture_critical_output,
           E_1_t_output, E_2_t_output, g_sw_1_output, g_sw_2_output, g_sw_output, residual_moisture_output, 
           throughfall_output, throughfall_summed_output, topsoil_moisture_addition_output,
           delintercept_output, inflow_output, surfstor_output, delsurfstor_output, delsoilmoist_output,
           asat_output, latent_output, sensible_output, grnd_flux_output, vp_output, vpd_output,
           surf_cond_output, density_output, g_sw_output, g_sw_summed_output, dry_time_factor_output, g_sw_1_summed_output, g_sw_2_summed_output

end


function write_daily_outputs(day, tsurf, aerodynamic_resistance, ra_eff, 
                            transpiration, tair_gpu, prec_gpu, throughfall,
                            delintercept, inflow, surfstor, delsurfstor, 
                            delsoilmoist, asat, latent, sensible, grnd_flux, 
                            vp_gpu, vpd, surf_cond, Q12, soil_evaporation, 
                            soil_temperature, soil_moisture_new, total_et,
                            surface_runoff, total_runoff, kappa_array, cs_array, 
                            potential_evaporation, water_storage, net_radiation,
                            canopy_evaporation, max_water_storage, wilting_point,
                            soil_moisture_critical, soil_moisture_max, E_1_t, 
                            E_2_t, residual_moisture, cv_gpu, coverage_gpu, g_sw, dry_time_factor, g_sw_1, g_sw_2,
                            # Output array references
                            tsurf_output, aerodynamic_resistance_output,
                            aerodynamic_resistance_summed_output, 
                            transpiration_output, transpiration_summed_output,
                            tair_output, precipitation_output, throughfall_output,
                            throughfall_summed_output, delintercept_output,
                            inflow_output, surfstor_output, delsurfstor_output,
                            delsoilmoist_output, asat_output, latent_output,
                            sensible_output, grnd_flux_output, vp_output,
                            vpd_output, surf_cond_output, density_output,
                            Q12_output, soil_evaporation_output,
                            soil_temperature_output, soil_moisture_output,
                            total_et_output, surface_runoff_output, total_runoff_output,
                            kappa_array_output, cs_array_output,
                            potential_evaporation_output,
                            potential_evaporation_summed_output,
                            water_storage_output, water_storage_summed_output,
                            net_radiation_output, net_radiation_summed_output,
                            canopy_evaporation_output,
                            canopy_evaporation_summed_output,
                            max_water_storage_output,
                            max_water_storage_summed_output,
                            wilting_point_output, soil_moisture_critical_output,
                            soil_moisture_max_output, E_1_t_output, E_2_t_output,
                            residual_moisture_output, g_sw_output, g_sw_summed_output, dry_time_factor_output, g_sw_1_output, g_sw_2_output, g_sw_1_summed_output, g_sw_2_summed_output)
    


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

    # Direct outputs
    tsurf_output[:, :, day] = Array(tsurf)
    aerodynamic_resistance_output[:, :, day, :] = Array(aerodynamic_resistance)
    aerodynamic_resistance_summed_output[:, :, day] = Array(ra_eff)
    
    tair_output[:, :, day] = Array(tair_gpu)
    precipitation_output[:, :, day] = Array(prec_gpu)
    throughfall_output[:, :, day, :] = Array(throughfall)
    throughfall_summed_output[:, :, day] = Array(sum_with_nan_handling(throughfall, 4))
    
    # Diagnostic outputs
    delintercept_output[:, :, day, :] = Array(delintercept)
    inflow_output[:, :, day, :] = Array(inflow)
    surfstor_output[:, :, day, :] = Array(surfstor)
    delsurfstor_output[:, :, day, :] = Array(delsurfstor)
    delsoilmoist_output[:, :, day, :] = Array(delsoilmoist)
    asat_output[:, :, day] = Array(asat)
    
    # Energy balance outputs
    latent_output[:, :, day, :] = Array(latent)
    sensible_output[:, :, day, :] = Array(sensible)
    grnd_flux_output[:, :, day, :] = Array(grnd_flux)
    
    # Atmospheric outputs
    vp_output[:, :, day] = Array(vp_gpu)
    vpd_output[:, :, day] = Array(vpd)
    surf_cond_output[:, :, day, :] = Array(surf_cond)
    density_output[:, :, day] = fill(
        float_type(rho_a), size(tair_gpu, 1), size(tair_gpu, 2)
    )
    
    # Processed outputs with sanitization
    Q12_processed = san_zero(Q12)
    Q12_output[:, :, day, :] = Array(Q12_processed)
    
    soil_evaporation_output[:, :, day, :] = Array(soil_evaporation)
    soil_temperature_output[:, :, day, :] = Array(soil_temperature)
    soil_moisture_output[:, :, day, :] = Array(soil_moisture_new)
    
    total_et_output[:, :, day] = Array(total_et)
    surface_runoff_output[:, :, day] = Array(surface_runoff)

    total_runoff_output[:, :, day] = Array(total_runoff)
    kappa_array_output[:, :, day, :] = Array(kappa_array)
    cs_array_output[:, :, day, :] = Array(cs_array)
    
    g_sw_processed = san_nan(g_sw)
    g_sw_output[:, :, day, :] = Array(g_sw_processed)
    g_sw_summed_output[:, :, day] = Array(
        sum_with_nan_handling(
            convcv(g_sw_processed) .* g_sw_processed, 4
        )
    )

    g_sw_1_processed = san_nan(g_sw_1)
    g_sw_1_output[:, :, day] = Array(g_sw_1_processed)
#    g_sw_1_summed_output[:, :, day] = Array(
#        sum_with_nan_handling(
#            convcv(g_sw_1_processed) .* g_sw_1_processed, 4
#        )
#    )

    g_sw_2_processed = san_nan(g_sw_2)
    g_sw_2_output[:, :, day] = Array(g_sw_2_processed)
 #   g_sw_2_summed_output[:, :, day] = Array(
 #       sum_with_nan_handling(
 #           convcv(g_sw_2_processed) .* g_sw_2_processed, 4
 #       )
 #   )

    dry_time_factor_output[:, :, day, :] = Array(dry_time_factor)

    # Potential evaporation
    potential_evaporation_processed = san_nan(potential_evaporation)
    potential_evaporation_output[:, :, day, :] = Array(potential_evaporation_processed)
    potential_evaporation_summed_output[:, :, day] = Array(
        sum_with_nan_handling(
            convcv(potential_evaporation_processed) .* potential_evaporation_processed, 4
        )
    )
    
    # Water storage
    water_storage_processed = san_nan(water_storage)
    water_storage_output[:, :, day, :] = Array(water_storage_processed)
    water_storage_summed_output[:, :, day] = Array(
        sum_with_nan_handling(
            water_storage_processed, 4
        )
    )
    
    # Net radiation
    net_radiation_processed = san_nan(net_radiation)
    net_radiation_output[:, :, day, :] = Array(net_radiation_processed)
    net_radiation_summed_output[:, :, day] = Array(
        sum_with_nan_handling(
            convcv(net_radiation_processed) .* net_radiation_processed, 4
        )
    )

    # Transpiration
    transpiration_processed = san_nan(transpiration)
    transpiration_gc = transpiration_processed .* coverage_gpu # transpiration is already cv weighted
    transpiration_output[:, :, day, :] = Array(transpiration_gc)
    transpiration_summed_output[:, :, day] = Array(
        sum_with_nan_handling(transpiration_gc, 4)
    )
    
    # Canopy evaporation (grid-cell field)
    canopy_evaporation_processed = san_nan(canopy_evaporation)
    canopy_evaporation_gc = convcv(canopy_evaporation_processed) .* 
                           canopy_evaporation_processed .* coverage_gpu
    canopy_evaporation_output[:, :, day, :] = Array(canopy_evaporation_gc)
    canopy_evaporation_summed_output[:, :, day] = Array(
        sum_with_nan_handling(canopy_evaporation_gc, 4)
    )
    
    # Max water storage
    max_water_storage_processed = san_nan(max_water_storage)
    max_water_storage_output[:, :, day, :] = Array(max_water_storage_processed)
    max_water_storage_summed_output[:, :, day] = Array(
        sum_with_nan_handling(
            convcv(max_water_storage_processed) .* max_water_storage_processed, 4
        )
    )
    
    # Soil properties (some are time-invariant but written each day for now)
    wilting_point_output[:, :, :] = Array(wilting_point)
    soil_moisture_critical_output[:, :, :] = Array(soil_moisture_critical)
    soil_moisture_max_output[:, :, :] = Array(soil_moisture_max)
    E_1_t_output[:, :, day, :] = Array(E_1_t)
    E_2_t_output[:, :, day, :] = Array(E_2_t)
    residual_moisture_output[:, :, day, :] = Array(residual_moisture)
end