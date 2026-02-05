CASE, start_year_arg, end_year_arg = parse_case_args()

# Choose Float64, Float32 or other for operations and output
global float_type = Float32

lat_var = "lat"
lon_var = "lon"

# Load the configuration matching the input argument:
if CASE == "global"

    println("Loading configuration for 'global'...")
    global nveg = 14
    global fillvalue_threshold = 1f30

    # =========================== GLOBAL CONFIGURATION ===========================
        
    # Input file paths/names
    input_param_file       = "./input_data/global/vic_global_5min_params_fix2.nc"
    routing_param_file = "./params/VIC_rout_params_ISIMIP3b.nc"

    input_prec_prefix      = "./input_data/global/5arcmin/forcing/prec/prec_WFDE5_CRU+GPCC_v2.0_5arcmin_"
    input_tair_prefix      = "./input_data/global/5arcmin/forcing/tair/tair_WFDE5_v2.0_5arcmin_"
    input_wind_prefix      = "./input_data/global/5arcmin/forcing/wind/wind_WFDE5_v2.0_5arcmin_"
    input_vp_prefix        = "./input_data/global/5arcmin/forcing/vp/vp_WFDE5_v2.0_5arcmin_"
    input_swdown_prefix    = "./input_data/global/5arcmin/forcing/swdown/swdown_WFDE5_v2.0_5arcmin_"
    input_lwdown_prefix    = "./input_data/global/5arcmin/forcing/lwdown/lwdown_WFDE5_v2.0_5arcmin_"
    input_psurf_prefix     = "./input_data/global/5arcmin/forcing/psurf/psurf_WFDE5_v2.0_5arcmin_"

    # Input variable names (as specified in the input files' metadata)
    d0_var = "displacement"
    z0_var = "veg_rough"
    z0soil_var = "rough"
    LAI_var = "LAI"
    rmin_var = "rmin"
    rarc_var = "rarc"
    cv_var = "Cv"
    elev_var = "elev"
    residmoist_var = "resid_moist"
    init_moist_var = "init_moist"
    c_expt_var = "c"

    ksat_var = "Ksat"
    albedo_var = "albedo"
    root_var = "root_fract" # root_fract(veg_class, root_zone, lat, lon) ;
    #root_fract_layer1 = root_fract[:, 0, :, :]
    #root_fract_layer2 = root_fract[:, 1, :, :]
    
    # === Field Capacity, Wilting Point, and Critical Moisture related variables ===
    Wcr_var = "Wcr_FRACT" #Wcr_FRACT(nlayer, lat, lon) 
    Wfc_var = "Wfc_FRACT" #Wfc_FRACT(nlayer, lat, lon) 
    Wpwp_var = "Wpwp_FRACT" #Wpwp_FRACT(nlayer, lat, lon) 
    coverage_var = "fcanopy" #fcanopy(veg_class, month, lat, lon) # "canopy coverage"
    quartz_var = "quartz" #quartz(nlayer, lat, lon)

    # === Extract Soil Parameters ===
    depth_var = "depth" #depth(nlayer, lat, lon)
    bulk_dens_var = "bulk_density" #bulk_density(nlayer, lat, lon)
    soil_dens_var = "soil_density" #soil_density(nlayer, lat, lon) 
    expt_var = "expt"
    b_infilt_var = "infilt"

    # === Subsurface Parameters ===
    Ds_var = "Ds" #fraction
    Dsmax_var = "Dsmax" #mm/day
    Ws_var = "Ws" #fraction
    Tavg_var = "avg_T" 
    dp_var = "dp"

    prec_var = "prec"
    tair_var = "tair"
    wind_var = "wind" 
    vp_var = "vp"
    swdown_var = "swdown"
    lwdown_var = "lwdown"
    psurf_var = "psurf"

    # Output file paths/names
    output_dir             = "./output_data/global/"
    output_file_prefix     = "outputfile_global_"
    
    # Set default simulation years if no command-line arguments are provided
    start_year             = isnothing(start_year_arg) ? 1979 : start_year_arg
    end_year               = isnothing(end_year_arg)   ? 2019 : end_year_arg
    
    # ========================= END GLOBAL CONFIGURATION ==========================

    ensure_output_directory(output_dir)
    println("Running from year $start_year to year $end_year.\n")

elseif CASE == "indus"

    println("Loading configuration for 'indus'...")
    global nveg = 22
    global fillvalue_threshold = 1f30

    # ============================ INDUS CONFIGURATION ============================
   
    # Input file paths/names
    input_param_file       = "./input_data/indus/VIC_params_Mirca_calibrated_Indus.nc"
    routing_param_file     = "./input_data/indus/routing/VIC_rout_params_Indus.nc"

    input_prec_prefix      = "./input_data/indus/forcing/pr_daily_GFDL-ESM4adj_historical/pr_daily_GFDL-ESM4adj_historical_"
    input_tair_prefix      = "./input_data/indus/forcing/tas_daily_GFDL-ESM4adj_historical/tas_daily_GFDL-ESM4adj_historical_"
    input_wind_prefix      = "./input_data/indus/forcing/wind10_daily_GFDL-ESM4_historical/wind10_daily_GFDL-ESM4_historical_"
    input_vp_prefix        = "./input_data/indus/forcing/vp_daily_GFDL-ESM4_historical/vp_daily_GFDL-ESM4_historical_"
    input_swdown_prefix    = "./input_data/indus/forcing/swdown_daily_GFDL-ESM4adj_historical/swdown_daily_GFDL-ESM4adj_historical_"
    input_lwdown_prefix    = "./input_data/indus/forcing/lwdown_daily_GFDL-ESM4adj_historical/lwdown_daily_GFDL-ESM4adj_historical_"
    input_psurf_prefix     = "./input_data/indus/forcing/psurf_daily_GFDL-ESM4_historical/psurf_daily_GFDL-ESM4_historical_"
    
    # Input variable names (as specified in the input files' metadata)
    d0_var = "displacement"
    z0_var = "veg_rough"
    z0soil_var = "rough"
    LAI_var = "LAI"
    rmin_var = "rmin"
    rarc_var = "rarc"
    cv_var = "Cv"
    elev_var = "elev"
    residmoist_var = "resid_moist"
    init_moist_var = "init_moist"
    c_expt_var = "c"

    ksat_var = "Ksat"
    albedo_var = "albedo"
    root_var = "root_fract" # root_fract(veg_class, root_zone, lat, lon) ;
    #root_fract_layer1 = root_fract[:, 0, :, :]
    #root_fract_layer2 = root_fract[:, 1, :, :]
    
    # === Field Capacity, Wilting Point, and Critical Moisture related variables ===
    Wcr_var = "Wcr_FRACT" #Wcr_FRACT(nlayer, lat, lon) 
    Wfc_var = "Wfc_FRACT" #Wfc_FRACT(nlayer, lat, lon) 
    Wpwp_var = "Wpwp_FRACT" #Wpwp_FRACT(nlayer, lat, lon) 
    coverage_var = "fcanopy" #fcanopy(veg_class, month, lat, lon) # "canopy coverage"
    quartz_var = "quartz" #quartz(nlayer, lat, lon)

    # === Extract Soil Parameters ===
    depth_var = "depth" #depth(nlayer, lat, lon)
    bulk_dens_var = "bulk_density" #bulk_density(nlayer, lat, lon)
    soil_dens_var = "soil_density" #soil_density(nlayer, lat, lon) 
    expt_var = "expt"
    b_infilt_var = "infilt"

    # === Subsurface Parameters ===
    Ds_var = "Ds" #fraction
    Dsmax_var = "Dsmax" #mm/day
    Ws_var = "Ws" #fraction
    Tavg_var = "avg_T" 
    dp_var = "dp"

    prec_var = "pr"
    tair_var = "tas"
    wind_var = "wind10" 
    vp_var = "vp"
    swdown_var = "swdown"
    lwdown_var = "lwdown"
    psurf_var = "psurf"

    # Output file paths/names
    output_dir             = "./output_data/indus/"
    output_file_prefix     = "outputfile_indus_"
    
    # Set default simulation years if no command-line arguments are provided
    start_year             = isnothing(start_year_arg) ? 1979 : start_year_arg
    end_year               = isnothing(end_year_arg)   ? 2010 : end_year_arg
    
    # ========================== END INDUS CONFIGURATION ==========================

    ensure_output_directory(output_dir)
    println("Running from year $start_year to year $end_year. \n")

elseif CASE == "mekong"

    println("Loading configuration for 'mekong'...")
    global nveg = 14
    global fillvalue_threshold = 9998.0f0
    # ============================ INDUS CONFIGURATION ============================
   
    # Input file paths/names
    input_param_file       = "./input_data/mekong/vic_mekong_5min_params.nc"
    routing_param_file     = "./input_data/mekong/routing/VIC_rout_params_Mekong.nc"

    input_prec_prefix      = "./input_data/mekong/forcing/prec/prec_WFDE5_CRU+GPCC_v2.0_5arcmin_"
    input_tair_prefix      = "./input_data/mekong/forcing/tair/tair_WFDE5_v2.0_5arcmin_"
    input_wind_prefix      = "./input_data/mekong/forcing/wind/wind_WFDE5_v2.0_5arcmin_"
    input_vp_prefix        = "./input_data/mekong/forcing/vp/vp_WFDE5_v2.0_5arcmin_"
    input_swdown_prefix    = "./input_data/mekong/forcing/swdown/swdown_WFDE5_v2.0_5arcmin_"
    input_lwdown_prefix    = "./input_data/mekong/forcing/lwdown/lwdown_WFDE5_v2.0_5arcmin_"
    input_psurf_prefix     = "./input_data/mekong/forcing/psurf/psurf_WFDE5_v2.0_5arcmin_"

    # Input variable names (as specified in the input files' metadata)
    d0_var = "displacement"
    z0_var = "veg_rough"
    z0soil_var = "rough"
    LAI_var = "LAI"
    rmin_var = "rmin"
    rarc_var = "rarc"
    cv_var = "Cv"
    elev_var = "elev"
    residmoist_var = "resid_moist"
    init_moist_var = "init_moist"
    c_expt_var = "c"

    ksat_var = "Ksat"
    albedo_var = "albedo"
    root_var = "root_fract" # root_fract(veg_class, root_zone, lat, lon) ;
    #root_fract_layer1 = root_fract[:, 0, :, :]
    #root_fract_layer2 = root_fract[:, 1, :, :]
    
    # === Field Capacity, Wilting Point, and Critical Moisture related variables ===
    Wcr_var = "Wcr_FRACT" #Wcr_FRACT(nlayer, lat, lon) 
    Wfc_var = "Wfc_FRACT" #Wfc_FRACT(nlayer, lat, lon) 
    Wpwp_var = "Wpwp_FRACT" #Wpwp_FRACT(nlayer, lat, lon) 
    coverage_var = "fcanopy" #fcanopy(veg_class, month, lat, lon) # "canopy coverage"
    quartz_var = "quartz" #quartz(nlayer, lat, lon)

    # === Extract Soil Parameters ===
    depth_var = "depth" #depth(nlayer, lat, lon)
    bulk_dens_var = "bulk_density" #bulk_density(nlayer, lat, lon)
    soil_dens_var = "soil_density" #soil_density(nlayer, lat, lon) 
    expt_var = "expt"
    b_infilt_var = "infilt"

    # === Subsurface Parameters ===
    Ds_var = "Ds" #fraction
    Dsmax_var = "Dsmax" #mm/day
    Ws_var = "Ws" #fraction
    Tavg_var = "avg_T" 
    dp_var = "dp"

    prec_var = "prec"
    tair_var = "tair"
    wind_var = "wind" 
    vp_var = "vp"
    swdown_var = "swdown"
    lwdown_var = "lwdown"
    psurf_var = "psurf"

    # Output file paths/names
    output_dir             = "./output_data/mekong/"
    output_file_prefix     = "outputfile_mekong_"
    
    # Set default simulation years if no command-line arguments are provided
    start_year             = isnothing(start_year_arg) ? 1979 : start_year_arg
    end_year               = isnothing(end_year_arg)   ? 1984 : end_year_arg
    
    # ========================== END INDUS CONFIGURATION ==========================

    ensure_output_directory(output_dir)
    println("Running from year $start_year to year $end_year. \n")

else
    error("Unknown CASE: '$CASE'. Please provide 'global' or 'indus' (or any other case defined in init.jl) as the first argument.")
end

