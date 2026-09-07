module Config

export load_config

using Configurations
using TOML


@option "paths" struct InputPaths
    input_param_file::String
    coverage_file::String
    routing_param_file::String
    precipitation_file::String
    air_temperature_file::String
    wind_speed_file::String
    vapor_pressure_file::String
    shortwave_down_file::String
    longwave_down_file::String
    surface_pressure_file::String
end

@option "names" struct InputNames
    # Grid Parameters
    latitude::String = "lat"
    longitude::String = "lon"
    time::String = "time"
    elevation::String = "elev"
    average_temperature::String = "avg_T"
    annual_precipitation::String = "annual_prec"
    snow_band_area_fraction::String = "AreaFract"
    snow_band_elevation::String = "elevation"
    snow_band_precipitation_factor::String = "Pfactor"

    # Vegetation 
    #  (static)
    root_fraction::String = "root_fract"
    vegetation_fraction::String = "Cv"
    minimum_resistance::String = "rmin"
    architectural_resistance::String = "rarc"
    #  (active monthly vals)
    displacement_height::String = "displacement"
    roughness_length::String = "veg_rough"
    lai::String = "LAI"
    albedo::String = "albedo"
    canopy_coverage::String = "fcanopy"

    # Soil
    hydraulic_conductivity::String = "Ksat"
    depth::String = "depth"
    initial_moisture::String = "init_moist"
    maximum_moisture::String = "max_moist"
    residual_moisture_fraction::String = "resid_moist"
    critical_moisture_fraction::String = "Wcr_FRACT"
    field_capacity_fraction::String = "Wfc_FRACT"
    wilting_point_fraction::String = "Wpwp_FRACT"
    quartz_content::String = "quartz"
    bare_roughness::String = "rough"
    bulk_density::String = "bulk_density"
    particle_density::String = "soil_density"
    campbell_n::String = "expt"
    nijssen_infilt_b::String = "infilt"
    nijssen_lin_reservoir::String = "Ds"
    nijssen_nonlin_reservoir::String = "Dsmax"
    moisture_depth_baseflow_transition::String = "Ws"
    column_depth::String = "dp"
    baseflow_curve_exp::String = "c"

    # Forcing
    precipitation::String = "prec"
    air_temperature::String = "tair"
    wind_speed::String = "wind" 
    vapor_pressure::String = "vp"
    shortwave_down::String = "swdown"
    longwave_down::String = "lwdown"
    surface_pressure::String = "psurf"
end

@option "input" struct InputCfg
    paths::InputPaths
    names::InputNames
end

@option "output" struct OutputCfg
    format::String = "zarr"
    dir::String
    file_prefix::String
end

@option "config" struct Cfg
    nveg::Int = 14  # number of vegetation types
    enable_routing::Bool = true
    lat_var::String = "lat"
    lon_var::String = "lon"
    enable_snow::Bool = true
    nbands::Int = 5  # snow bands (alternatively, derive from input data)
    fillvalue_threshold::Float32 = 1f15
    start_year::Int
    end_year::Int
    timestep::Int = 86400  # seconds
    input::InputCfg
    output::OutputCfg
end

"""Validate the path of a file relative to the given directory."""
function validate_path(file, dir)
    file = abspath(joinpath(dir, file))
    if endswith(file, "_")
        files = readdir(dirname(file))
        n_matching_files = sum(startswith.(files, basename(file)))
        if n_matching_files < 1
            error("No files found in ", dirname(file), "starting with", basename(file))
        end
    elseif !isfile(file)
        error("Cannot find file '$file'")
    end
    return file
end


function load_config(config_file)
    cfg_dict = TOML.parsefile(config_file)

    # Make all input paths absolute, make relative path abs to config.
    # The routing parameter file is only needed when routing is enabled, so do
    # not demand it otherwise -- a run with `enable_routing = false` should not
    # require a file it never opens.
    routing_enabled = get(cfg_dict, "enable_routing", true)
    for (key, path) in cfg_dict["input"]["paths"]
        if key == "routing_param_file" && !routing_enabled
            continue
        end
        cfg_dict["input"]["paths"][key] = validate_path(path, dirname(config_file))
    end

    # Make output dir path absolute, relative to the config file location
    if !isabspath(cfg_dict["output"]["dir"])
        cfg_dict["output"]["dir"] = abspath(joinpath(dirname(config_file), cfg_dict["output"]["dir"]))
    end

    return from_dict(Cfg, cfg_dict)
end


end
