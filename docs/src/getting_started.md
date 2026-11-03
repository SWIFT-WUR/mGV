# Getting started

## Installation
### How to install Julia on linux
```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz
tar -xvzf julia-1.10.5-linux-x86_64.tar.gz
sudo mv julia-1.10.5 /opt/julia
sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia
julia --version
```
### Activate the project and install the dependencies
```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```
Please note that the command has to be done only once. 

## Running the test case
```bash
julia --project=. -e 'using mGV; mGV.run()' 'configs/mekong_config.toml
```

This will provide output netCDF files for the Mekong region, years 1979 to 1984. Currently we only provide forcing and landsurface parameter files for the (small) Mekong region in this repo, due to file-size considerations. Data for the entire globe and the indus region will be made available at a later point in development, or can be made available upon request.

## GPU Acceleration

GPU support is provided through package extensions, so no GPU packages are installed by default.
To run on a GPU, add the package for your hardware to your environment and load it together with mGV:

```julia
using CUDA, mGV  # or AMDGPU, Metal, oneAPI, OpenCL
mGV.run("configs/mekong_config.toml")
```

By default, (`backend="auto"`), the model runs on the first loaded GPU package that is functional, and otherwise falls back to CPU.
A specific backend can be selected with the `backend` keyword, one of `"auto"`, `"cpu"`, `"cuda"`, `"rocm"`, `"metal"`, `"oneapi"`, `"opencl"`:

```julia
using CUDA, mGV
mGV.run("configs/mekong_config.toml"; backend="cuda")
model = mGV.Model("configs/mekong_config.toml"; backend="cpu")
```

Explicitly requesting a GPU backend whose package is not loaded or not functional raises an error.

When the model is initialized through BMI (`BMI.initialize`), the backend can be selected with the `MGV_BACKEND` environment variable, e.g. `MGV_BACKEND=cpu`.
If it is not set, `"auto"` is used.
Any other KernelAbstractions backend can be used by passing it directly: `mGV.Model(config_file; backend=MyBackend())`.

GPU acceleration is confirmed to work and actively supported for Nvidia and AMD hardware.

- Apple Metal is currently untested due to a lack of testing hardware.

## Configuration file
### Time information
The temporal starting and ending point of the simulation are specified in the `config.toml` file. 
```toml
start_year = 1979
end_year = 1980
```
### Routing and snow routines
The implementations of routing and snow can be activated using the corresponding flags.
```toml
enable_routing = true
enable_snow = true
```

### Output section
The format, the name and the path of the output file are specified as follows:
```toml
[output]
dir         = "../output_data/mekong/"
file_prefix = "outputfile_mekong_"
format      = "netcdf"
```
Currently, only `netcdf` and `zarr` formats are supported. 

### Input section
The `input` section contains information regarding the forcing, model parameters and routing files.
Also in this case, the input files can be either `netcdf` or `zarr`.
More information about the structure and the preparation of the input files are available in this repo [mGV-input data](https://github.com/SWIFT-WUR/mGV-input-data).
The paths are stored in the '[input.paths]' section:
```toml
[input]
[input.paths]
input_param_file       = "../input_data/mekong/vic_mekong_5min_params.nc"
coverage_file          = "../input_data/mekong/vic_mekong_5min_domain.nc"
routing_param_file     = "../input_data/mekong/routing/VIC_rout_params_Mekong.nc"

# Forcing data:
precipitation_file    = "../input_data/mekong/forcing/prec/prec_WFDE5_CRU+GPCC_v2.0_5arcmin_{year}.nc"
air_temperature_file  = "../input_data/mekong/forcing/tair/tair_WFDE5_v2.0_5arcmin_{year}.nc"
wind_speed_file       = "../input_data/mekong/forcing/wind/wind_WFDE5_v2.0_5arcmin_{year}.nc"
vapor_pressure_file   = "../input_data/mekong/forcing/vp/vp_WFDE5_v2.0_5arcmin_{year}.nc"
shortwave_down_file   = "../input_data/mekong/forcing/swdown/swdown_WFDE5_v2.0_5arcmin_{year}.nc"
longwave_down_file    = "../input_data/mekong/forcing/lwdown/lwdown_WFDE5_v2.0_5arcmin_{year}.nc"
surface_pressure_file = "../input_data/mekong/forcing/psurf/psurf_WFDE5_v2.0_5arcmin_{year}.nc"
```

### BMI variables
All the variables exposed through the BMI implementation of mGV must be specified in the `[API]` section, within the `variables` array of strings. 
The variable names are based on the [CSDMS Standard Names](https://csdms.colorado.edu/wiki/CSDMS_Standard_Names).
```toml
variables = [
    "land_surface__radiation~net~upward_energy_flux",
    "land_surface__evaporation~potential",
    "land_surface__evaporation~potential~soil",
    "atmosphere_bottom_air__resistance~aerodynamic",
    "land_surface__temperature"
]
```

When coupling mGV with an external groundwater model, to avoid double counting the baseflow of the bottom soil layer, the following flag must be set to true. 
```toml
[API]
# set the following to `true` to manually solve subsurface routing via BMI
#  with a groundwater model (e.g. MODFLOW).
external_groundwater_routing = true
```
Indeed, mGV computes subsurface runoff (baseflow from the bottom soil layer) in the land module and by default adds it to the total runoff (`land__runoff~total`), which feeds the river routing.

## BMI implementation

BMI is a set of standard controls and query functions that can be added to a numerical model
to make it easier to learn and to couple it with other softwares. 
For more information and the spec list see: [BMI specifications](https://bmi.csdms.io/en/stable/).
mGV, based on the Julia programming language, uses the following [BMI for Julia](https://github.com/Deltares/BasicModelInterface.jl) specifications based on BMI 2.0 version. 

The BMI implementation in mGV uses rectilinear grids, and is based on three different coordinates:
longitude, latitude and vegetation types. 
Variables depending on soil layers as well as on snow bands are exposed as two-dimensional grids.
To set of get these variables the index of the layer/band is required by adding `_layer_N` / `_band_N` at the end of the variable name.

Currently, the variables exposed through the BMI are model state and output variables.
Their names are based on the [CSDMS Standard Names](https://csdms.colorado.edu/wiki/CSDMS_Standard_Names).

Variables having a dependence on soil layers or snow band are expanded adding the corresponding layer or band at the end of the name. For example, adding `soil__thermal_conductivity` to the ``` [API] ``` section of the config file, will automatically add `soil__thermal_conductivity_layer_1`, ..., `soil__thermal_conductivity_layer_N` (depending on the actual number of soil layers) to the list of the output variables. 

### Exposed variables
The full list of the variables and parameters exposed or accessible through BMI is contained in `standard_name.jl` file.
Currently, there are three `OrderedDict`: `standard_name_map` contains the list of variables beside those related to routing; `standard_routing_map` contains the routing variables; `standard_param_map` contains some parameters but is not actually used (for now).
Each entry of the dictionaries corresponds to a variable/parameter with some metadata like the corresponding units of measurement, a brief description and a tag. 

### Units

Values are exchanged in mGV's native units, as reported by `get_var_units`. These are not
necessarily SI units: for example, runoff is in mm day⁻¹ and temperatures are in °C. The
coupling framework is responsible for converting them to the units of the other model.

### Remote BMI controller
The mGV model, as any other model coupled with the BMI, can be controlled with the [BMI controller](https://github.com/BSchilperoort/bmi-controller), which is a browser-based controller for [Remote BMI](https://github.com/eWaterCycle/remotebmi) models.
The controller needs [Node.js](https://nodejs.org/en) 20+ and [pnpm](https://pnpm.io/).

#### Setup
```bash
pnpm install
```

#### Development

```bash
pnpm dev
```

Opens at `http://localhost:5173`. The dev server will connect to `http://localhost:50051` upon initialization, so the Remote BMI server must be running before you interact with the UI.

To use the controller a small Julia script like the following is needed:
```julia
import RemoteBMI.Server: run_bmi_server
using mGV
run_bmi_server(mGV.Model, "127.0.0.1", 8000)
```
Assuming that the script is called `server_script.jl`, to launch mGV with the BMI controller is sufficient to type `julia --project=. path/to/server_script.jl` within the mGV folder. 
This has to be done after launching `pnpm dev` within the `bmi-controller` folder.
Afterwards, it is possible to initialize the model and control it through the UI.
Currently, the following features are available:
- Initialize, step, run-until, and finalize a BMI model
- Inspect input/output variables
- Time-series plot for any output variable
- Grid viewer with per-variable color mapping (2 and 3D variables)