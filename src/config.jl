using Configurations

@option "paths" struct InputPaths
    input_param_file::String
    coverage_file::String
    routing_param_file::String
    input_prec_prefix::String
    input_tair_prefix::String
    input_wind_prefix::String
    input_vp_prefix::String
    input_swdown_prefix::String
    input_lwdown_prefix::String
    input_psurf_prefix::String
end

@option "names" struct InputNames
    lat::String = "lat"
    lon::String = "lon"
    d0::String = "displacement"
    z0::String = "veg_rough"
    z0soil::String = "rough"
    LAI::String = "LAI"
    rmin::String = "rmin"
    rarc::String = "rarc"
    cv::String = "Cv"
    elev::String = "elev"
    residmoist::String = "resid_moist"
    init_moist::String = "init_moist"
    c_expt::String = "c"
    AreaFract::String = "AreaFract"
    elevation::String = "elevation"
    Pfactor::String = "Pfactor"
    ksat::String = "Ksat"
    albedo::String = "albedo"
    root::String = "root_fract"
    Wcr::String = "Wcr_FRACT"
    Wfc::String = "Wfc_FRACT"
    Wpwp::String = "Wpwp_FRACT"
    coverage::String = "fcanopy"
    quartz::String = "quartz"
    depth::String = "depth"
    bulk_dens::String = "bulk_density"
    soil_dens::String = "soil_density"
    expt::String = "expt"
    b_infilt::String = "infilt"
    Ds::String = "Ds"
    Dsmax::String = "Dsmax"
    Ws::String = "Ws"
    Tavg::String = "avg_T" 
    dp::String = "dp"
    annual_prec::String = "annual_prec"
    prec::String = "prec"
    tair::String = "tair"
    wind::String = "wind" 
    vp::String = "vp"
    swdown::String = "swdown"
    lwdown::String = "lwdown"
    psurf::String = "psurf"
end

@option "input" struct InputCfg
    paths::InputPaths
    names::InputNames
end

@option "output" struct OutputCfg
    dir::String 
    file_prefix::String
end

@option "config" struct Config
    nveg::Int = 14
    enable_routing::Bool = true
    lat_var::String = "lat"
    lon_var::String = "lon"
    enable_snow::Bool = true
    fillvalue_threshold::Float32 = 1f15
    start_year::Int
    end_year::Int
    input::InputCfg
    output::OutputCfg
end
