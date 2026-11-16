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
Page-lock host memory to speed up device transfers. Does nothing unless a backend
extension adds a method for its backend.
"""
pin_memory!(::Backend, arr) = nothing

"""
    load_gpu_backend!(name)

Load the package of GPU backend `name`, so that its extension registers the backend. With
`"auto"`, load the packages of the known backends that are installed, until one is functional.
Used by the app, which cannot load packages with `using` at runtime. The package must be
installed in the app's environment, see [`add_backend!`](@ref).
"""
function load_gpu_backend!(name::AbstractString)
    name = lowercase(name)
    candidates = name == "auto" ? collect(keys(KNOWN_BACKENDS)) :
        haskey(KNOWN_BACKENDS, name) ? [name] : String[]
    for key in candidates
        haskey(BACKEND_REGISTRY, key) && continue          # already loaded
        package = KNOWN_BACKENDS[key]
        pkg = Base.identify_package(package)               # nothing if not installed
        if isnothing(pkg)
            name == "auto" && continue
            error("Backend '$key' needs $package.jl, which is not installed. Install it with:\n" *
                "    mGV --add-backend $key")
        end
        try
            Base.require(pkg)                              # loads e.g. AMDGPUExt, which registers the backend
        catch err
            name == "auto" || rethrow()
            @warn "Could not load $package, skipping backend '$key'" exception = err
            continue
        end
        if !haskey(BACKEND_REGISTRY, key)
            msg = "$package was loaded, but the mGV extension for it was not (see the error above). " *
                "If mGV runs as an app, run `mGV --add-backend $key` again."
            name == "auto" ? (@warn msg; continue) : error(msg)
        end
        # for auto: stop at the first backend that is functional on this machine
        name == "auto" && Base.invokelatest(BACKEND_REGISTRY[key].is_functional) && break
    end
    return nothing
end

"""
    add_backend!(name)

Install the package of GPU backend `name` (e.g. AMDGPU for `"rocm"`) into the active
environment, such that `load_gpu_backend!` can load it. For the app this is the app's own
environment, so that Pkg resolves the GPU package together with mGV, within mGV's compat
bounds.

Pkg apps run in an environment with a copy of mGV's `Project.toml`, without the `ext/`
folder next to it. Julia looks for the extensions of the root project there, so copy it in.
Updating the app recreates its environment: run this again afterwards.
"""
function add_backend!(name::AbstractString)
    name = lowercase(name)
    haskey(KNOWN_BACKENDS, name) || error(
        "Unknown GPU backend '$name'. Valid options are: $(join(keys(KNOWN_BACKENDS), ", ")).",
    )
    project_file = Base.active_project()
    env = dirname(project_file)
    source = pkgdir(@__MODULE__)
    if samefile(env, source)
        # `Pkg.Apps.develop`: the app runs in the mGV repository, whose Project.toml we leave alone
        error(
            "mGV runs from its source folder ($source), e.g. after `Pkg.Apps.develop`. Install " *
                "$(KNOWN_BACKENDS[name]) in an environment of your own instead, and run mGV from " *
                "there: julia --project=<env> -m mGV config.toml -b $name",
        )
    end
    if Base.project_file_name_uuid(project_file, "") == Base.PkgId(@__MODULE__)
        cp(joinpath(source, "ext"), joinpath(env, "ext"); force = true)
    end
    Pkg.activate(env) do
        Pkg.add(KNOWN_BACKENDS[name])
    end
    return nothing
end
