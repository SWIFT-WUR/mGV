using NCDatasets.CommonDataModel: CFVariable, MFCFVariable
using Dates: DateTime, Millisecond

const FORCING_VARS = [
    "precipitation",
    "air_temperature",
    "wind_speed",
    "vapor_pressure",
    "shortwave_down",
    "longwave_down",
    "surface_pressure"
]

# Cache RAM budget; cached timesteps = this / grid size. 2 GiB measured 25-38%
# faster than 256 MiB at global resolution, with no gain going higher.
const FORCING_CACHE_BUDGET_BYTES = 2 * 1024^3

# Zarr has no date type, so its time axis counts milliseconds from here.
# Milliseconds are exact for a DateTime, so dates survive the round trip.
const ZARR_TIME_EPOCH = DateTime(1970, 1, 1)

function getval(data::Any, name::String)
    return getfield(data, Symbol(name))
end

@kwdef struct ForcingVariables{T <: AbstractMatrix}
    precipitation::T
    air_temperature::T
    wind_speed::T
    vapor_pressure::T
    shortwave_down::T
    longwave_down::T
    surface_pressure::T
end

@adapt_structure ForcingVariables

const ForcingVar = Union{CFVariable, MFCFVariable}

"""
One variable's per-year Zarr stores as a single time axis. `offsets[i]` is the
run-wide index of the first timestep in `arrays[i]`.
"""
struct ZarrForcingVar{A}
    arrays::Vector{A}
    offsets::Vector{Int}
end

mutable struct ForcingReaders{S}
    # NetCDF variables or Zarr stores; only `load_block!` knows which.
    sources::Dict{String, S}
    # Read many timesteps per disk hit, then serve single days from memory.
    times::Vector{DateTime}
    cache::Dict{String, Vector{Matrix{Float32}}}
    cache_start::Int   # index of the first cached timestep, 0 when empty
    cache_len::Int     # number of valid timesteps currently cached
    capacity::Int      # how many timesteps the cache can hold
    slice_size::Tuple{Int, Int}  # (nx, ny) of one timestep's grid
end

"""
Open the per-year NetCDF files of every forcing variable as one aggregated
time series.
"""
function open_forcing_netcdf(config_file::AbstractString, cfg::Cfg)
    years = cfg.start_year:cfg.end_year
    var_prefixes = [getval(cfg.input.paths, "$(var)_file") for var in FORCING_VARS]
    files = Vector{String}(undef, length(years))
    datasets = Vector{Any}(undef,  length(var_prefixes))

    for i = eachindex(var_prefixes)
        for j = eachindex(years)
            ncfile = validate_path(
                "$(var_prefixes[i])$(years[j]).nc",
                dirname(config_file)
            )
            files[j] = ncfile
        end
        datasets[i] = NCDataset(unique(files), aggdim = "time", deferopen = false)
    end

    sources = Dict{String, ForcingVar}(
        var => datasets[i][getval(cfg.input.names, var)]
        for (i, var) in enumerate(FORCING_VARS)
    )

    times = collect(DateTime, datasets[1][getval(cfg.input.names, "time")][:])
    nx, ny, _ = size(sources[FORCING_VARS[1]])
    return sources, times, (nx, ny)
end

"""
Path to the Zarr store for one forcing variable's prefix and year, as
written by scripts/convert_forcing_to_zarr.jl.
"""
zarr_store_path(config_dir, prefix, year) = abspath(joinpath(config_dir, "$(prefix)$(year).zarr"))

"""
True if every forcing variable has a Zarr store on disk for every configured
year. Used to decide what `forcing_format = "auto"` should do.
"""
function zarr_forcing_available(config_file::AbstractString, cfg::Cfg)
    config_dir = dirname(config_file)
    years = cfg.start_year:cfg.end_year
    for var in FORCING_VARS
        prefix = getval(cfg.input.paths, "$(var)_file")
        for year in years
            isdir(zarr_store_path(config_dir, prefix, year)) || return false
        end
    end
    return true
end

"""
Open the per-year Zarr stores of every forcing variable, written by
scripts/convert_forcing_to_zarr.jl.
"""
function open_forcing_zarr(config_file::AbstractString, cfg::Cfg)
    years = cfg.start_year:cfg.end_year
    config_dir = dirname(config_file)

    sources = Dict{String, ZarrForcingVar}()
    times = DateTime[]

    for var in FORCING_VARS
        prefix = getval(cfg.input.paths, "$(var)_file")
        groups = map(years) do year
            path = zarr_store_path(config_dir, prefix, year)
            isdir(path) || error("Cannot find Zarr forcing store '$path'")
            zopen(path)
        end

        arrays = [group[getval(cfg.input.names, var)] for group in groups]
        offsets = Int[]
        next = 1
        for array in arrays
            push!(offsets, next)
            next += size(array, 3)
        end
        sources[var] = ZarrForcingVar(arrays, offsets)

        # Every variable shares one time axis, so only read it once.
        if isempty(times)
            for group in groups
                append!(times, ZARR_TIME_EPOCH .+ Millisecond.(group["time"][:]))
            end
        end
    end

    nx, ny, _ = size(first(sources[FORCING_VARS[1]].arrays))
    return sources, times, (nx, ny)
end

"""
Decide whether to read forcing as Zarr or NetCDF, from `cfg.input.forcing_format`.

"zarr" and "netcdf" pick that format directly. "auto" (the default) checks
whether a complete set of Zarr stores exists for the configured years and
uses it if so, otherwise falls back to NetCDF. Either way it prints which
one was picked, so the choice is never silent.
"""
function use_zarr_forcing(config_file::AbstractString, cfg::Cfg)
    format = lowercase(cfg.input.forcing_format)
    if format == "zarr"
        return true
    elseif format == "netcdf"
        return false
    elseif format == "auto"
        found = zarr_forcing_available(config_file, cfg)
        if found
            println("forcing_format=auto: found Zarr forcing stores, using zarr.")
        else
            println("forcing_format=auto: no complete set of Zarr forcing stores found, using netcdf.")
        end
        return found
    else
        error("Unknown forcing_format '$(cfg.input.forcing_format)'. Use \"auto\", \"zarr\", or \"netcdf\".")
    end
end

"""
Open the forcing input data files to prepare for stepwise data
loading.
"""
function open_forcing(config_file::AbstractString, cfg::Cfg)
    sources, times, (nx, ny) = if use_zarr_forcing(config_file, cfg)
        open_forcing_zarr(config_file, cfg)
    else
        open_forcing_netcdf(config_file, cfg)
    end

    bytes_per_step = nx * ny * sizeof(Float32) * length(FORCING_VARS)
    capacity = clamp(FORCING_CACHE_BUDGET_BYTES ÷ max(bytes_per_step, 1), 1, length(times))

    cache = Dict{String, Vector{Matrix{Float32}}}(
        var => [Matrix{Float32}(undef, nx, ny) for _ in 1:capacity] for var in FORCING_VARS
    )

    return ForcingReaders(
        sources,
        times,
        cache,
        0,
        0,
        capacity,
        (nx, ny),
    )
end

"""
Index of the timestep closest to `time`, mirroring the approximate time
matching the per-step `@select` used to do.
"""
function nearest_time_index(times::Vector{DateTime}, time::DateTime)
    i = searchsortedfirst(times, time)
    i <= 1 && return 1
    i > length(times) && return length(times)
    return (time - times[i - 1]) <= (times[i] - time) ? i - 1 : i
end

"""
Read `len` timesteps starting at `start` from a NetCDF variable into `buffers`.
"""
function load_block!(buffers::Vector{Matrix{Float32}}, src::ForcingVar, start::Int, len::Int)
    raw = src[:, :, start:(start + len - 1)]
    # A declared _FillValue yields Union{Missing,Float32}; the model wants NaN.
    block = raw isa Array{Float32, 3} ? raw : Array{Float32, 3}(coalesce.(raw, NaN32))
    for k in 1:len
        copyto!(buffers[k], view(block, :, :, k))
    end
    return nothing
end

"""
Read `len` timesteps from `start` into `buffers`, crossing year boundaries.
"""
function load_block!(buffers::Vector{Matrix{Float32}}, src::ZarrForcingVar, start::Int, len::Int)
    nx, ny = size(first(buffers))
    for k in 1:len
        global_index = start + k - 1
        i = searchsortedlast(src.offsets, global_index)
        local_index = global_index - src.offsets[i] + 1
        # One chunk per timestep, so this decompresses straight into the cache.
        Zarr.readblock!(
            reshape(buffers[k], nx, ny, 1),
            src.arrays[i],
            CartesianIndices((1:nx, 1:ny, local_index:local_index)),
        )
    end
    return nothing
end

"""
Load the block of timesteps starting at `start` into the host cache.
"""
function fill_forcing_cache!(readers::ForcingReaders, start::Int)
    len = min(readers.capacity, length(readers.times) - start + 1)
    for var in FORCING_VARS
        load_block!(readers.cache[var], readers.sources[var], start, len)
    end
    readers.cache_start = start
    readers.cache_len = len
    return nothing
end

"""
Copy the forcing field for `time` into `dest`, refilling the block cache when
the requested timestep falls outside it.
"""
function read_var!(dest, time::DateTime, readers::ForcingReaders, var::String)
    idx = nearest_time_index(readers.times, time)
    if idx < readers.cache_start || idx > readers.cache_start + readers.cache_len - 1
        fill_forcing_cache!(readers, idx)
    end
    copyto!(dest, readers.cache[var][idx - readers.cache_start + 1])
    return nothing
end

"""
Read in forcing data at a certain point in time.
"""
function read_var(time, readers::ForcingReaders, var::String)
    dest = Matrix{Float32}(undef, readers.slice_size)
    read_var!(dest, time, readers, var)
    return dest
end

"""
Initialize forcing reader and read the first timestep of
forcing data.
"""
function initialize_forcing(config_file::AbstractString, cfg::Cfg)
    forcing_readers = open_forcing(config_file, cfg)
    start_time = DateTime(cfg.start_year,1,1)
    forcing_vars = ForcingVariables(;
        ((Symbol(var) => read_var(start_time, forcing_readers, var)) for var in FORCING_VARS)...
    )
    return forcing_readers, forcing_vars
end

"""
Update the forcing variables to the data representing a new time step.

Note: will update CPU as well as GPU arrays in-place.
"""
function update_forcing!(time, readers::ForcingReaders, vars::ForcingVariables)
    read_var!(vars.precipitation, time, readers, "precipitation")
    read_var!(vars.air_temperature, time, readers, "air_temperature")
    read_var!(vars.wind_speed, time, readers, "wind_speed")
    read_var!(vars.vapor_pressure, time, readers, "vapor_pressure")
    read_var!(vars.shortwave_down, time, readers, "shortwave_down")
    read_var!(vars.longwave_down, time, readers, "longwave_down")
    read_var!(vars.surface_pressure, time, readers, "surface_pressure")
    return nothing
end
