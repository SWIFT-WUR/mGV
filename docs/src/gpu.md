# Running on GPU

PU support is provided through package extensions, so no GPU packages are installed by default.
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
