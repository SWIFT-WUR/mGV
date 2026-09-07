using NCDatasets.CommonDataModel: CFVariable, MFCFVariable
using Dates: DateTime

const FORCING_COORDS = [
    "latitude",
    "longitude",
    "time",
]

const FORCING_VARS = [
    "precipitation",
    "air_temperature",
    "wind_speed",
    "vapor_pressure",
    "shortwave_down",
    "longwave_down",
    "surface_pressure"
]

# Host memory the forcing cache is allowed to occupy across all variables. The
# number of timesteps held in memory is derived from this and the grid size, so
# a small basin caches a long block while a large grid falls back to short ones.
# 256 MiB gave a cache capacity of exactly 1 timestep at global (5 arcmin)
# resolution -- i.e. no real caching at all, just a refill every single day.
# 2 GiB (~10 cached days at that resolution) measured 25-38% faster
# update_forcing! with no further gain going higher.
const FORCING_CACHE_BUDGET_BYTES = 2 * 1024^3

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

mutable struct ForcingReaders
    time::ForcingVar
    latitude::ForcingVar
    longitude::ForcingVar
    precipitation::ForcingVar
    air_temperature::ForcingVar
    wind_speed::ForcingVar
    vapor_pressure::ForcingVar
    shortwave_down::ForcingVar
    longwave_down::ForcingVar
    surface_pressure::ForcingVar
    # Block cache: reading one timestep at a time costs a NetCDF round trip per
    # variable per step, which dominated the run loop. Instead read a block of
    # timesteps once and serve the individual steps from host memory.
    times::Vector{DateTime}
    cache::Dict{String, Vector{Matrix{Float32}}}
    cache_start::Int   # index of the first cached timestep, 0 when empty
    cache_len::Int     # number of valid timesteps currently cached
    capacity::Int      # timesteps per block
    slice_size::Tuple{Int, Int}
end

"""
Open the forcing input data files to prepare for stepwise data
loading.
"""
function open_forcing(config_file::AbstractString, cfg::Cfg)
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

    vars_dict = Dict()

    for i in eachindex(FORCING_VARS)
        var = FORCING_VARS[i]
        vars_dict[var] = datasets[i][getval(cfg.input.names, var)]
    end
    for var in FORCING_COORDS
        vars_dict[var] = datasets[1][getval(cfg.input.names, var)]
    end

    times = collect(DateTime, vars_dict["time"][:])
    nx, ny, _ = size(vars_dict[FORCING_VARS[1]])
    bytes_per_step = nx * ny * sizeof(Float32) * length(FORCING_VARS)
    capacity = clamp(FORCING_CACHE_BUDGET_BYTES ÷ max(bytes_per_step, 1), 1, length(times))

    cache = Dict{String, Vector{Matrix{Float32}}}(
        var => [Matrix{Float32}(undef, nx, ny) for _ in 1:capacity] for var in FORCING_VARS
    )

    return ForcingReaders(
        vars_dict["time"],
        vars_dict["latitude"],
        vars_dict["longitude"],
        vars_dict["precipitation"],
        vars_dict["air_temperature"],
        vars_dict["wind_speed"],
        vars_dict["vapor_pressure"],
        vars_dict["shortwave_down"],
        vars_dict["longwave_down"],
        vars_dict["surface_pressure"],
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
Load the block of timesteps starting at `start` into the host cache.
"""
function fill_forcing_cache!(readers::ForcingReaders, start::Int)
    len = min(readers.capacity, length(readers.times) - start + 1)
    stop = start + len - 1
    for var in FORCING_VARS
        raw = getval(readers, var)[:, :, start:stop]
        # A declared _FillValue makes NCDatasets hand back a Union{Missing,Float32}
        # array; the rest of the model represents absent data as NaN.
        block = raw isa Array{Float32, 3} ? raw : Array{Float32, 3}(coalesce.(raw, NaN32))
        buffers = readers.cache[var]
        for k in 1:len
            copyto!(buffers[k], view(block, :, :, k))
        end
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
