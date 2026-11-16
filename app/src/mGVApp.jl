"""
The `mGV` command-line app.

GPU backends are package extensions of mGV, loaded when their GPU package (e.g. CUDA) is
loaded. The app runs in an environment of its own, without GPU packages. So the app loads the
GPU package of the requested backend from the general Julia environment (e.g. `@v1.12`), which
then loads mGV's extension for it.

mGV is a dependency of this package rather than the app itself, because Julia cannot find the
extensions of the package an app is made of: Pkg only copies its `Project.toml`, not `ext/`.
"""
module mGVApp

using ArgParse: ArgParse, ArgParseSettings, @add_arg_table!, parse_args
using mGV: mGV

const EPILOG = """
Examples:
    mGV config.toml
    mGV config.toml -b cuda

Number of threads:
  Threads are a Julia option, so pass them before `--`, and the app's arguments after it:

    mGV --threads=8 -- config.toml -b cpu
    mGV --threads=auto -- config.toml       (one thread per CPU thread, the default)

GPU backends:
  A GPU backend needs its package installed in the general Julia environment, e.g. for cuda:

    julia -e 'using Pkg; Pkg.add("CUDA")'

  The packages: $(join(("$k: $p" for (k, p) in mGV.KNOWN_BACKENDS), ", ")).
"""

function parse_cli(args)
    settings = ArgParseSettings(
        prog = "mGV",
        description = "Run the mGV hydrological model",
        epilog = EPILOG,
        preformatted_epilog = true,
    )
    @add_arg_table! settings begin
        "config"
        help = "path to the config.toml file"
        "--backend", "-b"
        help = "backend: " * join(mGV.backend_options(), ", ") * ". With auto, the first " *
            "installed GPU backend that works on this machine, or else cpu"
        default = "auto"
    end
    return parse_args(args, settings), settings
end

"""
    load_gpu_package!(name)

Load the GPU package of backend `name` from the general Julia environment, which loads mGV's
extension for it, which registers the backend. With `"auto"`, load the installed GPU packages
in order of preference, until one works on this machine. Does nothing for `"cpu"`.
"""
function load_gpu_package!(name::AbstractString)
    name = lowercase(name)
    candidates = name == "auto" ? collect(keys(mGV.KNOWN_BACKENDS)) :
        haskey(mGV.KNOWN_BACKENDS, name) ? [name] : String[]
    isempty(candidates) && return nothing
    # The general environment, after the app's own, which comes first
    "@v#.#" in LOAD_PATH || push!(LOAD_PATH, "@v#.#")
    for key in candidates
        package = mGV.KNOWN_BACKENDS[key]
        if !haskey(mGV.BACKEND_REGISTRY, key)
            pkg = Base.identify_package(package)  # nothing if not installed
            if isnothing(pkg)
                name == "auto" && continue
                error(
                    "Backend '$key' needs $package.jl. Install it in the general Julia " *
                        "environment:\n    julia -e 'using Pkg; Pkg.add(\"$package\")'",
                )
            end
            try
                Base.require(pkg)
            catch err
                name == "auto" || rethrow()
                @warn "Could not load $package, skipping backend '$key'" exception = err
                continue
            end
            if !haskey(mGV.BACKEND_REGISTRY, key)
                msg = "$package was loaded, but the mGV extension for it was not (see the error above)"
                name == "auto" || error(msg)
                @warn msg
                continue
            end
        end
        name == "auto" && Base.invokelatest(mGV.BACKEND_REGISTRY[key].is_functional) && break
    end
    return nothing
end

function (@main)(args)
    opts, settings = parse_cli(args)
    if isnothing(opts["config"])
        ArgParse.show_help(settings; exit_when_done = false)
        return 1
    end
    load_gpu_package!(opts["backend"])
    # The extension of the GPU package adds methods to mGV after this function was compiled
    Base.invokelatest(mGV.run, opts["config"]; backend = opts["backend"])
    return 0
end

end
