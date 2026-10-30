This repo contains a high-performance global macroscale hydrologic model programmed in Julia, based on the VIC framework by Liang et al. (1994). The code in this repo is a work-in-progress and in it's early development phase. 

## How to run

### Install Julia on Linux (if not available yet on your machine)

```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz
tar -xvzf julia-1.10.5-linux-x86_64.tar.gz
sudo mv julia-1.10.5 /opt/julia
sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia
julia --version
```

---

### Activate the project and install dependencies
Needs to be done only once.

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```


### Run command

To run the program:

```bash
julia --project=. --threads=auto run.jl configs/mekong_config.toml
```

Starting Julia with multiple threads (`--threads=auto`, or e.g. `--threads=8`) has a large effect on performance: on CPU the model kernels run multithreaded, and Zarr forcing is read with one thread per forcing variable (so up to 7 threads help there).

This will provide output netCDF files for the Mekong region, years 1979 to 1984. Currently we only provide forcing and landsurface parameter files for the (small) Mekong region in this repo, due to file-size considerations. Data for the entire globe and the indus region will be made available at a later point in development, or can be made available upon request.

### GPU Acceleration

GPU support is provided through package extensions, so no GPU packages are installed by default.
To run on a GPU, add the package for your hardware to your environment and load it together with mGV:

```julia
using CUDA, mGV  # or AMDGPU, Metal, oneAPI, OpenCL
mGV.run("configs/mekong_config.toml")
```

By default (`backend="auto"`) the model runs on the first loaded GPU package that is functional, and otherwise falls back to CPU.
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

### Configuration file

The model uses configuration files for initialization and loading input data.
Start and end years for a model run can be configured there.

Note that all paths in the configuration file should be;
1. Absolute (e.g. `/home/username/mgv/data/mekong/...`)
2. or relative to the *config file*, to aid in portability of data+configs.
