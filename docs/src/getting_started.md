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
please note that the command has to be done only once. 

### Running the test case
```bash
julia --project=. -e 'using mGV; mGV.run()' 'configs/mekong_config.toml
```

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


### Runic formatter
#### Installation

In Julia v1.12 and later, Runic can be installed as a
[Pkg app](https://pkgdocs.julialang.org/dev/apps/):

```sh
julia -e 'using Pkg; Pkg.Apps.add("Runic")'
```

Assuming `~/.julia/bin` is in your `PATH` you can now invoke `runic` from your shell, e.g.:

```sh
runic --version # Show version info
runic --help    # Show documentation
```

```sh
# Format all files in-place in the current directory (recursively)
runic --inplace .
```

For more instruction see https://github.com/fredrikekre/Runic.jl and https://github.com/fredrikekre/runic-action.

