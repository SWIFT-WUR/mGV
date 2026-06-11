using TOML


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


config_file = parse_args()

println("Loading configuration file...")

cfg_dict = TOML.parsefile(config_file)

# Make all input paths absolute, make relative path abs to config
for (key, path) in cfg_dict["input"]["paths"]
    cfg_dict["input"]["paths"][key] = validate_path(path, dirname(config_file))
end

# Check that parent directory of output dir exists, make relative path abs to config
if !isabspath(cfg_dict["output"]["dir"])
    output_dir = joinpath(dirname(config_file), dirname(cfg_dict["output"]["dir"]))
    if !isdir(output_dir)
        error(
            "Output parent directory '$output_dir' does not exist or is not a directory."
        )
    end
    cfg_dict["output"]["dir"] = joinpath(output_dir, basename(cfg_dict["output"]["dir"]))
end

cfg = from_dict(Config, cfg_dict)

start_year = cfg.start_year
end_year   = cfg.end_year

lat_var = cfg.input.names.lat
lon_var = cfg.input.names.lon

global nveg = cfg.nveg
global enable_routing = cfg.enable_routing

enable_snow         = cfg.enable_snow
fillvalue_threshold = cfg.fillvalue_threshold

input_param_file    = cfg.input.paths.input_param_file
coverage_file       = cfg.input.paths.coverage_file
routing_param_file  = cfg.input.paths.routing_param_file
input_prec_prefix   = cfg.input.paths.input_prec_prefix
input_tair_prefix   = cfg.input.paths.input_tair_prefix
input_wind_prefix   = cfg.input.paths.input_wind_prefix
input_vp_prefix     = cfg.input.paths.input_vp_prefix
input_swdown_prefix = cfg.input.paths.input_swdown_prefix
input_lwdown_prefix = cfg.input.paths.input_lwdown_prefix
input_psurf_prefix  = cfg.input.paths.input_psurf_prefix

d0_var = cfg.input.names.d0
z0_var = cfg.input.names.z0
z0soil_var = cfg.input.names.z0soil
LAI_var = cfg.input.names.LAI
rmin_var = cfg.input.names.rmin
rarc_var = cfg.input.names.rarc
cv_var = cfg.input.names.cv
elev_var = cfg.input.names.elev
residmoist_var = cfg.input.names.residmoist
init_moist_var = cfg.input.names.init_moist
c_expt_var = cfg.input.names.c_expt
AreaFract_var = cfg.input.names.AreaFract
elevation_var = cfg.input.names.elevation
Pfactor_var = cfg.input.names.Pfactor
ksat_var = cfg.input.names.ksat
albedo_var = cfg.input.names.albedo
root_var = cfg.input.names.root
Wcr_var = cfg.input.names.Wcr
Wfc_var = cfg.input.names.Wfc
Wpwp_var = cfg.input.names.Wpwp
coverage_var = cfg.input.names.coverage
quartz_var = cfg.input.names.quartz
depth_var = cfg.input.names.depth
bulk_dens_var = cfg.input.names.bulk_dens
soil_dens_var = cfg.input.names.soil_dens
expt_var = cfg.input.names.expt
b_infilt_var = cfg.input.names.b_infilt
Ds_var = cfg.input.names.Ds
Dsmax_var = cfg.input.names.Dsmax
Ws_var = cfg.input.names.Ws
Tavg_var = cfg.input.names.Tavg
dp_var = cfg.input.names.dp
annual_prec_var = cfg.input.names.annual_prec
prec_var = cfg.input.names.prec
tair_var = cfg.input.names.tair
wind_var = cfg.input.names.wind
vp_var = cfg.input.names.vp
swdown_var = cfg.input.names.swdown
lwdown_var = cfg.input.names.lwdown
psurf_var = cfg.input.names.psurf

output_dir             = cfg.output.dir
output_file_prefix     = cfg.output.file_prefix

ensure_output_directory(output_dir)
println("Running from year $start_year to year $end_year. \n")
