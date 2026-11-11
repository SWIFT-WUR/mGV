"""
GPU backends are provided by package extensions (see `ext/`), which are loaded when the
corresponding GPU package is loaded alongside mGV (e.g. `using CUDA, mGV`). Each
extension registers its backend through [`register_backend!`](@ref) in its `__init__`.
"""
struct BackendEntry
    package::String         # package that provides the backend, e.g. "CUDA"
    is_functional::Base.Callable # () -> Bool
    construct::Base.Callable     # () -> KernelAbstractions.Backend
end

# Known backend names and the package providing them, in order of preference for "auto".
const KNOWN_BACKENDS = OrderedDict(
    "cuda" => "CUDA",
    "rocm" => "AMDGPU",
    "metal" => "Metal",
    "oneapi" => "oneAPI",
    "opencl" => "OpenCL",
)

const BACKEND_REGISTRY = Dict{String, BackendEntry}()

"""
    register_backend!(name, package, is_functional, construct)

Register a GPU backend under `name`, so it can be selected with `Model(config_file; backend=name)`.
Called by the package extensions on load.
"""
function register_backend!(name::AbstractString, package::AbstractString, is_functional, construct)
    BACKEND_REGISTRY[lowercase(name)] = BackendEntry(package, is_functional, construct)
    return nothing
end

backend_name(backend::Backend) = string(nameof(typeof(backend)))

# GPU backend names in order of preference: known backends first, then any others registered.
gpu_backend_names() = unique([collect(keys(KNOWN_BACKENDS)); collect(keys(BACKEND_REGISTRY))])

backend_options() = ["auto"; "cpu"; gpu_backend_names()]

"""
    select_backend(name="auto")

Select the KernelAbstractions backend to run the model on. This is evaluated at
runtime (not during precompilation), so the same precompiled package can be used
on machines with and without a GPU.

`name` is one of `"auto"`, `"cpu"`, $(join(repr.(collect(keys(KNOWN_BACKENDS))), ", ")).
GPU backends are only available once their package is loaded (e.g. `using CUDA`).
With `"auto"` the first loaded and functional GPU backend is used, falling back to
CPU. Explicitly requesting a GPU backend that is not loaded or not functional throws
an error.

A `KernelAbstractions.Backend` can also be passed directly, which is returned as is.
"""
select_backend(backend::Backend) = backend

function select_backend(name::AbstractString = "auto")
    name = lowercase(name)
    if name == "auto"
        for key in gpu_backend_names()
            entry = get(BACKEND_REGISTRY, key, nothing)
            if !isnothing(entry) && entry.is_functional()
                return entry.construct()
            end
        end
        return CPU()
    elseif name == "cpu"
        return CPU()
    elseif haskey(BACKEND_REGISTRY, name)
        entry = BACKEND_REGISTRY[name]
        entry.is_functional() || error(
            "Backend '$name' was requested, but $(entry.package) is not functional on this machine.",
        )
        return entry.construct()
    elseif haskey(KNOWN_BACKENDS, name)
        package = KNOWN_BACKENDS[name]
        error(
            "Backend '$name' requires $package.jl. Add it to your environment and load it " *
                "alongside mGV, e.g. `using $package, mGV`.",
        )
    else
        error("Unknown backend '$name'. Valid options are: $(join(backend_options(), ", ")).")
    end
end

"""
Page-lock (pin) host memory, so that device transfers can run asynchronously. Uses
`KernelAbstractions.pagelock!`, which the GPU packages implement for their backend (e.g.
CUDA and AMDGPU). Returns whether `arr` was pinned: `false` for backends that do not
implement it (`pagelock!` then returns `missing`), whose transfers still work, but can block
the task that makes them until they are done.
"""
pin_memory!(backend::Backend, arr) = !ismissing(KernelAbstractions.pagelock!(backend, arr))

"""
Undo [`pin_memory!`](@ref), for backends that do not unpin memory when it is freed (e.g.
AMDGPU). Does nothing unless a backend extension adds a method for its backend.
"""
unpin_memory!(::Backend, arr) = nothing

"""
    host_copier(backend)

An object that copies device arrays to host memory asynchronously, on a stream of its own,
for the output writer (see [`record_output!`](@ref)). Backend extensions that provide one
implement, for their copier `c`:

- `stage_copy!(c, dst, src)`: copy device array `src` to device array `dst` (of the same
  length), in the model's stream, after the model's kernels and before those that follow.
  Before it, waits (on the device) until the last host copy is done, so that `dst` can be reused.
- `start_host_copy!(c)`: makes the following host copies wait (on the device) for the model's
  stream, i.e. for the staged copies.
- `copy_to_host!(c, dst, doffs, src, soffs, n)`: copy `n` elements from device array `src`
  (from index `soffs`) to host array `dst` (from index `doffs`), without blocking the host.
- `finish_host_copy!(c)`: marks the end of the host copies of a timestep.
- `wait_host_copy(c)`: blocks the calling task until the host copies are done.

Returns `nothing` (the default) if the backend has none; the writer then copies on a
separate task instead.
"""
host_copier(::Backend) = nothing

function stage_copy! end
function start_host_copy! end
function copy_to_host! end
function finish_host_copy! end
function wait_host_copy end
