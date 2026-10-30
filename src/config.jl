module Config

export load_config

using Configurations
using TOML

import ..output_variables


@option "paths" struct InputPaths
    input_param_file::String
    coverage_file::String
    routing_param_file::String
    # One file or store with all forcing variables; the per-variable entries below
    # default to it, and override it when given.
    forcing_file::Union{Nothing, String} = nothing
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
    variables::Union{Nothing, Vector{String}} = nothing  # variables to write; nothing = all, [] = none
end

@option "API" struct APISection
    variables::Vector{String} = []
    # Set to `true` to exclude subsurface runoff (baseflow) from the total runoff
    #  that feeds the river routing. The subsurface runoff is then expected to be
    #  routed by an external groundwater model (e.g. MODFLOW) coupled via BMI.
    external_groundwater_routing::Bool = false
end

@option "config" struct Cfg
    nveg::Int = 14  # number of vegetation types
    enable_routing::Bool = true
    lat_var::String = "lat"
    lon_var::String = "lon"
    enable_snow::Bool = true
    nbands::Int = 5  # snow bands (alternatively, derive from input data)
    fillvalue_threshold::Float32 = 1.0f15
    start_year::Int
    end_year::Int
    timestep::Int = 86400  # seconds
    input::InputCfg
    output::OutputCfg
    API::APISection = APISection()
end

"""
Check that `path` is a Zarr store: a `.zarr` directory with Zarr metadata at its
root (`.zgroup` or `.zarray` for Zarr v2, `zarr.json` for v3).
"""
function validate_zarr_store(path)
    isdir(path) && endswith(path, ".zarr") || return false
    return any(isfile(joinpath(path, name)) for name in (".zgroup", ".zarray", "zarr.json"))
end

const FORCING_FILE_KEYS = [
    "precipitation_file",
    "air_temperature_file",
    "wind_speed_file",
    "vapor_pressure_file",
    "shortwave_down_file",
    "longwave_down_file",
    "surface_pressure_file",
]

"""
Use `forcing_file` for every forcing variable without its own file entry.
"""
function fill_forcing_paths!(paths)
    for key in FORCING_FILE_KEYS
        haskey(paths, key) && continue
        haskey(paths, "forcing_file") ||
            error("No input path for '$key': set it, or set 'forcing_file' for all forcing variables")
        paths[key] = paths["forcing_file"]
    end
    return paths
end

"""Validate the path of a file relative to the given directory."""
function validate_path(file, dir, years)
    file = abspath(joinpath(dir, file))
    for year_file in unique(replace(file, "{year}" => string(year)) for year in years)
        if !isfile(year_file) && !validate_zarr_store(year_file)
            error("Cannot find file or Zarr store '$year_file'")
        end
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
    years = cfg_dict["start_year"]:cfg_dict["end_year"]
    fill_forcing_paths!(cfg_dict["input"]["paths"])
    for (key, path) in cfg_dict["input"]["paths"]
        if key == "routing_param_file" && !routing_enabled
            continue
        end
        cfg_dict["input"]["paths"][key] = validate_path(path, dirname(config_file), years)
    end

    # Make output dir path absolute, relative to the config file location
    if !isabspath(cfg_dict["output"]["dir"])
        cfg_dict["output"]["dir"] =
            abspath(joinpath(dirname(config_file), cfg_dict["output"]["dir"]))
    end
    if !isdir(cfg_dict["output"]["dir"])
        error("Output directory '$(cfg_dict["output"]["dir"])' does not exist")
    end

    cfg = from_dict(Cfg, cfg_dict)
    validate_output_variables(cfg)
    return cfg
end

"""
Check the names in `[output].variables`: throw an error for unknown names, and warn about
variables that will not be written because the module they need is disabled.
"""
function validate_output_variables(cfg::Cfg)
    requested = cfg.output.variables
    isnothing(requested) && return nothing

    output_names(; kwargs...) = [meta.output_name for meta in values(output_variables(; kwargs...))]
    all_names = output_names()
    for name in requested
        name in all_names || error(
            "Unknown output variable '$name' in [output].variables. Valid names: " *
                join(all_names, ", ")
        )
    end

    for setting in (:enable_snow, :enable_routing)
        getfield(cfg, setting) && continue
        skipped = filter(!in(output_names(; setting => false)), requested)
        isempty(skipped) || @warn "Not writing $(join(skipped, ", ")), because $setting = false"
    end
    return nothing
end


end
