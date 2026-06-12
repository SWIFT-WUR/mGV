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
julia --project=. run.jl configs/mekong_config.toml
```

This will provide output netCDF files for the Mekong region, years 1979 to 1984. Currently we only provide forcing and landsurface parameter files for the (small) Mekong region in this repo, due to file-size considerations. Data for the entire globe and the indus region will be made available at a later point in development, or can be made available upon request.

### GPU Acceleration

If a GPU is available the code will try to run on GPU instead of CPU. Currently there is no configuration available here (such as device or backend selection).
GPU acceleration is confirmed to work and actively supported for Nvidia and AMD hardware.

- Apple Metal is currently not supported due to a lack of testing hardware.
- Intel OneAPI is not supported due to [a bug in OneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl/issues/575).

### Configuration file

The model uses configuration files for initialization and loading input data.
Start and end years for a model run can be configured there.

Note that all paths in the configuration file should be;
1. Absolute (e.g. `/home/username/mgv/data/mekong/...`)
2. or relative to the *config file*, to aid in portability of data+configs.
