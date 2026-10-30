using NCDatasets.CommonDataModel: CFVariable, MFCFVariable
using NCDatasets.CFTime: timedecode
using Dates: DateTime

const FORCING_VARS = [
    "precipitation",
    "air_temperature",
    "wind_speed",
    "vapor_pressure",
    "shortwave_down",
    "longwave_down",
    "surface_pressure",
]

# RAM for caching forcing timesteps (all 7 variables per step). 2 GiB measured
# 25-38% faster than 256 MiB at global resolution, with no gain going higher.
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

"""
One forcing variable stored as one Zarr array per year (or one for all years).
`arrays[i]` holds year `i` of the run, and `offsets[i]` is the run timestep of
its first day. E.g. for a 1979-1980 daily run: `offsets = [1, 366]`.
Every array is (lon, lat, time), see `check_zarr_dims`.
"""
struct ZarrForcingVar{A}
    arrays::Vector{A}
    offsets::Vector{Int}
end

mutable struct ForcingReaders{S}
    # NetCDF variables or ZarrForcingVars; `load_block!` has a version for each.
    sources::Dict{String, S}
    # Cache multiple forcing time steps to reduce read overhead
    times::Vector{DateTime}
    cache::Dict{String, Array{Float32, 3}}  # (nx, ny, capacity) per variable
    cache_start::Int   # index of the first cached timestep, 0 when empty
    cache_len::Int     # number of valid timesteps currently cached
    capacity::Int      # how many timesteps the cache can hold
    time_chunk::Int    # timesteps per chunk in the forcing (1 for NetCDF)
    slice_size::Tuple{Int, Int}  # (nx, ny) of one timestep's grid
end

"""
Open the NetCDF forcing files of all years so they can be read as if they were
one file. E.g. day 366 of a 1979-1980 run is 1 Jan 1980.
"""
function open_forcing_netcdf(cfg::Cfg)
    years = cfg.start_year:cfg.end_year
    var_paths = [getval(cfg.input.paths, "$(var)_file") for var in FORCING_VARS]
    datasets = Vector{Any}(undef, length(var_paths))

    for i in eachindex(var_paths)
        files = unique(replace(var_paths[i], "{year}" => string(year)) for year in years)
        datasets[i] = NCDataset(files, aggdim = "time", deferopen = false)
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
Open the Zarr forcing stores of all years so they can be read as if they were
one store (see `ZarrForcingVar`). E.g. day 366 of a 1979-1980 run is 1 Jan 1980.
A path without `{year}` is a single store that holds all years.
"""
function open_forcing_zarr(cfg::Cfg)
    years = cfg.start_year:cfg.end_year

    # Let Blosc decompress in several threads at once (for `load_all_vars!`)
    ENV["BLOSC_NOLOCK"] = "1"

    sources = Dict{String, ZarrForcingVar}()
    times = DateTime[]

    for var in FORCING_VARS
        path = getval(cfg.input.paths, "$(var)_file")
        paths = unique(replace(path, "{year}" => string(year)) for year in years)
        groups = [zopen(p) for p in paths]

        arrays = [group[getval(cfg.input.names, var)] for group in groups]
        for (array, p) in zip(arrays, paths)
            check_zarr_dims(array, cfg.input.names, "$var in '$p'")
        end
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
                append!(times, read_zarr_time(group[cfg.input.names.time]))
            end
        end
    end

    check_zarr_forcing(sources, length(times))
    nx, ny, _ = size(first(sources[FORCING_VARS[1]].arrays))
    return sources, times, (nx, ny)
end

"""
Decode a Zarr time coordinate to `DateTime`s, using its CF `units` (e.g.
"days since 1970-01-01") and `calendar` attributes.
"""
function read_zarr_time(time_array)
    haskey(time_array.attrs, "units") ||
        error("Zarr time coordinate '$(time_array.path)' has no 'units' attribute")
    units = time_array.attrs["units"]
    calendar = get(time_array.attrs, "calendar", "standard")
    return collect(DateTime, timedecode(time_array[:], units, calendar))
end

"""
Dimension names of a Zarr array in Julia order (the reverse of the order in the
store), or `nothing` if the store does not name them. Zarr v2 stores them in the
`_ARRAY_DIMENSIONS` attribute (xarray convention), v3 as `dimension_names`.
"""
function zarr_dimnames(array::ZArray)
    names = if array.metadata.zarr_format == 3
        metadata = Zarr.JSON.parse(String(array.storage[array.path, "zarr.json"]))
        get(metadata, "dimension_names", nothing)
    else
        get(array.attrs, "_ARRAY_DIMENSIONS", nothing)
    end
    return isnothing(names) ? nothing : reverse(String.(names))
end

"""
Check that a forcing array is ordered (lon, lat, time), as the cache is. Arrays
without dimension names are assumed to be.
"""
function check_zarr_dims(array::ZArray, names, description)
    ndims(array) == 3 || error("Forcing $description must have 3 dimensions, has $(ndims(array))")
    dims = zarr_dimnames(array)
    expected = [names.longitude, names.latitude, names.time]
    if !isnothing(dims) && dims != expected
        error(
            "Forcing $description has dimensions $dims (in Julia order), expected $expected. " *
                "In Python (C order) this is (time, lat, lon)."
        )
    end
    return nothing
end

"""
Check that all forcing arrays share a grid and cover every timestep, and warn
when their time chunks differ (reading stays correct, but some chunks are then
decompressed more than once).
"""
function check_zarr_forcing(sources::Dict{String, ZarrForcingVar}, ntimes::Int)
    arrays = [array for var in FORCING_VARS for array in sources[var].arrays]
    grids = unique(size(array)[1:2] for array in arrays)
    length(grids) == 1 || error("Zarr forcing arrays have different grid sizes: $grids")
    for var in FORCING_VARS
        n = sum(array -> size(array, 3), sources[var].arrays)
        n == ntimes || error("Zarr forcing variable $var has $n timesteps, the time coordinate $ntimes")
    end
    chunks = unique(array.metadata.chunks[3] for array in arrays)
    length(chunks) == 1 ||
        @warn "Zarr forcing arrays have different time chunk lengths ($chunks), reading will be slower"
    return nothing
end

"""
Open the forcing files, as NetCDF or Zarr depending on their file extension,
and set up the cache that the model reads its daily forcing from.
"""
function open_forcing(cfg::Cfg)
    # Paths ending in ".zarr" are Zarr stores, anything else is NetCDF
    is_zarr = [endswith(getval(cfg.input.paths, "$(var)_file"), ".zarr") for var in FORCING_VARS]
    if !(all(is_zarr) || !any(is_zarr))
        error("Forcing paths must either all be Zarr stores (.zarr) or all NetCDF files")
    end
    sources, times, (nx, ny) = all(is_zarr) ? open_forcing_zarr(cfg) : open_forcing_netcdf(cfg)

    bytes_per_step = nx * ny * sizeof(Float32) * length(FORCING_VARS)
    capacity = clamp(FORCING_CACHE_BUDGET_BYTES ÷ max(bytes_per_step, 1), 1, length(times))

    # Round the cache to whole Zarr chunks, at least one even if over budget
    time_chunk = time_chunk_length(sources)
    capacity = min(max(capacity ÷ time_chunk, 1) * time_chunk, length(times))

    cache = Dict{String, Array{Float32, 3}}(
        var => Array{Float32, 3}(undef, nx, ny, capacity) for var in FORCING_VARS
    )

    return ForcingReaders(
        sources,
        times,
        cache,
        0,
        0,
        capacity,
        time_chunk,
        (nx, ny),
    )
end

"""
Number of timesteps in one chunk, from the Zarr metadata of the first forcing
variable (time is its third axis, see `check_zarr_dims`). 1 for NetCDF, which
is read per timestep.
"""
time_chunk_length(sources) = 1
time_chunk_length(sources::Dict{String, ZarrForcingVar}) =
    first(sources[FORCING_VARS[1]].arrays).metadata.chunks[3]

"""
Timestep at which the chunk holding timestep `idx` starts. Chunks are counted
from the first timestep of each store (e.g. each year).
"""
chunk_start(readers::ForcingReaders, idx::Int) =
    chunk_start(readers.sources[FORCING_VARS[1]], idx, readers.time_chunk)
chunk_start(src, idx::Int, time_chunk::Int) = idx
function chunk_start(src::ZarrForcingVar, idx::Int, time_chunk::Int)
    offset = src.offsets[searchsortedlast(src.offsets, idx)]
    return offset + (idx - offset) ÷ time_chunk * time_chunk
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
Read `len` timesteps, starting at `start`, from a NetCDF variable into the cache.
"""
function load_block!(cache::Array{Float32, 3}, src::ForcingVar, start::Int, len::Int)
    raw = src[:, :, start:(start + len - 1)]
    # Missing values in the forcing file (its _FillValue) are read as `missing`: replace
    # them with NaN and convert to a plain Float32 array.
    block = raw isa Array{Float32, 3} ? raw : Array{Float32, 3}(coalesce.(raw, NaN32))
    copyto!(view(cache, :, :, 1:len), block)
    return nothing
end

"""
Read `len` timesteps, starting at `start`, from a Zarr variable into the cache,
one chunk per read. Handles crossing year boundaries. Time is the third axis
(checked by `check_zarr_dims`).
"""
function load_block!(cache::Array{Float32, 3}, src::ZarrForcingVar, start::Int, len::Int)
    nx, ny, _ = size(cache)
    stop = start + len - 1
    step = start  # next timestep to read
    for (array, offset) in zip(src.arrays, src.offsets)
        time_chunk = array.metadata.chunks[3]
        store_stop = offset + size(array, 3) - 1
        while offset <= step <= min(stop, store_stop)
            # Read up to the end of the chunk that holds `step`
            chunk_stop = offset + ((step - offset) ÷ time_chunk + 1) * time_chunk - 1
            n = min(chunk_stop, stop, store_stop) - step + 1
            # Plain Array, not a view: Zarr then unpacks straight into the cache (2-3x faster)
            GC.@preserve cache begin
                dest = unsafe_wrap(Array, pointer(cache, (step - start) * nx * ny + 1), (nx, ny, n))
                Zarr.readblock!(
                    dest, array,
                    CartesianIndices((1:nx, 1:ny, (step - offset + 1):(step - offset + n)))
                )
            end
            step += n
        end
    end
    return nothing
end

"""
Fill the cache, starting at the beginning of the chunk that holds timestep `idx`.
"""
function fill_forcing_cache!(readers::ForcingReaders, idx::Int)
    start = chunk_start(readers, idx)
    len = min(readers.capacity, length(readers.times) - start + 1)
    load_all_vars!(readers, start, len)
    readers.cache_start = start
    readers.cache_len = len
    return nothing
end

"""For NetCDF, read `len` timesteps of all forcing variables into the cache, one by one."""
function load_all_vars!(readers::ForcingReaders, start::Int, len::Int)
    for var in FORCING_VARS
        load_block!(readers.cache[var], readers.sources[var], start, len)
    end
    return nothing
end

"""For Zarr, read all variables at the same time, one task each (needs `julia -t` threads)."""
function load_all_vars!(readers::ForcingReaders{ZarrForcingVar}, start::Int, len::Int)
    @sync for var in FORCING_VARS
        Threads.@spawn load_block!(readers.cache[var], readers.sources[var], start, len)
    end
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
    # Copy by position: GPU arrays (tested for CUDA, AMDGPU) can't copy a view in one go and error on the element-by-element fallback
    n = length(dest)
    copyto!(dest, 1, readers.cache[var], (idx - readers.cache_start) * n + 1, n)
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
function initialize_forcing(cfg::Cfg)
    forcing_readers = open_forcing(cfg)
    start_time = DateTime(cfg.start_year, 1, 1)
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
