using Zarr: Blosc
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
# Zarr forcing splits it over the cache and the next block, which is read ahead
# (prefetched) while the model computes.
const FORCING_CACHE_BUDGET_BYTES = 2 * 1024^3

function getval(data::Any, name::String)
    return getfield(data, Symbol(name))
end

@kwdef struct ForcingVariables{T <: AbstractVector}
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

"""
One forcing variable stored for the land cells only: one Zarr array of size (cell, time)
per year (or one for all years), see `scripts/convert_forcing_landonly.jl`. `arrays` and
`offsets` are as for `ZarrForcingVar`. `rows[k]` is the model cell of row `k` of the
arrays, or 0 if that cell is not in the domain. Values are unpacked as
`value * scale + offset`, and `fill` (the missing value in the store) becomes NaN.
"""
struct LandZarrForcingVar{A}
    arrays::Vector{A}
    offsets::Vector{Int}
    rows::Vector{Int32}
    scale::Float32
    offset::Float32
    fill::Float64
end

const AnyZarrForcingVar = Union{ZarrForcingVar, LandZarrForcingVar}

mutable struct ForcingReaders{S}
    # NetCDF variables, ZarrForcingVars or LandZarrForcingVars; `load_all_vars!` has a
    # version for each.
    sources::Dict{String, S}
    # Cache multiple forcing time steps to reduce read overhead
    times::Vector{DateTime}
    cache::Dict{String, Matrix{Float32}}  # (ncells, capacity) per variable, active cells only
    # Gridded (nx, ny, time_chunk) buffer per variable that Zarr chunks are read into,
    # before the active cells are copied to the cache. Empty for NetCDF and land-only Zarr.
    staging::Dict{String, Array{Float32, 3}}
    cells::Vector{Int}  # linear grid index of each active cell, see `Domain`
    cache_start::Int   # index of the first cached timestep, 0 when empty
    cache_len::Int     # number of valid timesteps currently cached
    capacity::Int      # how many timesteps the cache can hold
    time_chunk::Int    # timesteps per chunk in the forcing (1 for NetCDF)
    # Prefetching (Zarr only): `next_cache` is filled with the block after the cache by the
    # task `prefetch`, while the model computes. Empty and `nothing` for NetCDF.
    next_cache::Dict{String, Matrix{Float32}}
    prefetch::Union{Task, Nothing}
    prefetch_start::Int
    prefetch_len::Int
    # Reused decompression buffers for land-only Zarr
    buffers::Channel{Vector{UInt8}}
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
function open_forcing_zarr(cfg::Cfg, domain::Domain)
    years = cfg.start_year:cfg.end_year

    # Let Blosc decompress in several threads at once (for `load_all_vars!`)
    ENV["BLOSC_NOLOCK"] = "1"

    sources = Dict{String, AnyZarrForcingVar}()
    times = DateTime[]
    land = nothing  # (rows, grid size) of land-only stores, shared by all variables

    for var in FORCING_VARS
        path = getval(cfg.input.paths, "$(var)_file")
        paths = unique(replace(path, "{year}" => string(year)) for year in years)
        groups = [zopen(p) for p in paths]

        arrays = [group[getval(cfg.input.names, var)] for group in groups]
        is_land = ndims(first(arrays)) == 2
        for (array, p) in zip(arrays, paths)
            if is_land
                check_land_zarr_dims(array, cfg.input.names, "$var in '$p'")
            else
                check_zarr_dims(array, cfg.input.names, "$var in '$p'")
            end
        end
        offsets = Int[]
        next = 1
        for array in arrays
            push!(offsets, next)
            next += size(array, ndims(array))
        end
        if is_land
            if isnothing(land)
                land = land_rows(groups, paths, cfg.input.names, domain)
            end
            sources[var] = land_forcing_var(arrays, offsets, first(land))
        else
            sources[var] = ZarrForcingVar(arrays, offsets)
        end

        # Every variable shares one time axis, so only read it once.
        if isempty(times)
            for group in groups
                append!(times, read_zarr_time(group[cfg.input.names.time]))
            end
        end
    end

    kinds = unique(typeof(src) <: LandZarrForcingVar for src in values(sources))
    length(kinds) == 1 || error("Zarr forcing variables must either all be gridded or all be land-only")
    if only(kinds)
        sources = Dict{String, LandZarrForcingVar}(sources)
        check_zarr_forcing(sources, length(times))
        return sources, times, last(land)
    end
    sources = Dict{String, ZarrForcingVar}(sources)
    check_zarr_forcing(sources, length(times))
    nx, ny, _ = size(first(sources[FORCING_VARS[1]].arrays))
    return sources, times, (nx, ny)
end

"""
Map the rows of the land-only stores `groups` to the cells of `domain`, from the 0-based
grid indices of each row (`lon_index`, `lat_index`). Returns the model cell of every row (0
for rows outside the domain), and the grid size from the `lon` and `lat` coordinates. Every
store must hold the same cells, and every cell of the domain must be in them.
"""
function land_rows(groups, paths, names, domain::Domain)
    rows = nothing
    grid_size = (0, 0)
    for (group, p) in zip(groups, paths)
        for key in ("lon_index", "lat_index", names.longitude, names.latitude)
            haskey(group.arrays, key) || error("Land-only forcing '$p' has no '$key' array")
        end
        grid_size = (length(group[names.longitude]), length(group[names.latitude]))
        lon_index, lat_index = group["lon_index"][:], group["lat_index"][:]
        # Linear index of each row in the (nx, ny) grid, as in `Domain`
        index = Int.(lon_index) .+ 1 .+ Int.(lat_index) .* first(grid_size)
        if isnothing(rows)
            rows = index
        elseif rows != index
            error("Land-only forcing '$p' holds different cells than '$(first(paths))'")
        end
    end

    grid_size == domain.grid_size ||
        error("Forcing grid of size $grid_size does not match the parameter grid $(domain.grid_size)")
    cell_of = zeros(Int32, prod(grid_size))
    cell_of[domain.cells] .= 1:ncells(domain)
    model_rows = cell_of[rows]
    found = count(>(0), model_rows)
    found == ncells(domain) == length(unique(filter(>(0), model_rows))) || error(
        "Land-only forcing '$(first(paths))' is missing $(ncells(domain) - found) of the " *
            "$(ncells(domain)) active cells, or holds some of them more than once"
    )
    return model_rows, grid_size
end

"""
A `LandZarrForcingVar`, with the CF `scale_factor` and `add_offset` of the arrays, and their
`_FillValue` attribute or Zarr fill value as missing value.
"""
function land_forcing_var(arrays, offsets, rows)
    attrs = first(arrays).attrs
    scale = Float32(get(attrs, "scale_factor", 1))
    offset = Float32(get(attrs, "add_offset", 0))
    fill = get(attrs, "_FillValue", first(arrays).metadata.fill_value)
    fill = isnothing(fill) ? NaN : Float64(fill)
    for array in arrays
        a = array.attrs
        (Float32(get(a, "scale_factor", 1)), Float32(get(a, "add_offset", 0))) == (scale, offset) ||
            error("Land-only forcing arrays of one variable must share scale_factor and add_offset")
    end
    return LandZarrForcingVar(arrays, offsets, rows, scale, offset, fill)
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
Check that a land-only forcing array is ordered (cell, time). Arrays without dimension names
are assumed to be.
"""
function check_land_zarr_dims(array::ZArray, names, description)
    dims = zarr_dimnames(array)
    expected = ["cell", names.time]
    if !isnothing(dims) && dims != expected
        error(
            "Land-only forcing $description has dimensions $dims (in Julia order), expected " *
                "$expected. In Python (C order) this is (time, cell)."
        )
    end
    return nothing
end

"""
Check that all forcing arrays share a grid and cover every timestep, and that no
time chunk is longer than that of the first array, which sets the size of the
staging buffer (see `time_chunk_length`). Warn when the time chunks differ (reading
stays correct, but some chunks are then decompressed more than once).
"""
function check_zarr_forcing(sources::Dict{String, <:AnyZarrForcingVar}, ntimes::Int)
    arrays = [array for var in FORCING_VARS for array in sources[var].arrays]
    grids = unique(size(array)[1:(end - 1)] for array in arrays)
    length(grids) == 1 || error("Zarr forcing arrays have different grid sizes: $grids")
    for var in FORCING_VARS
        n = sum(array -> size(array, ndims(array)), sources[var].arrays)
        n == ntimes || error("Zarr forcing variable $var has $n timesteps, the time coordinate $ntimes")
    end
    chunks = unique(last(array.metadata.chunks) for array in arrays)
    time_chunk = time_chunk_length(sources)
    maximum(chunks) <= time_chunk || error(
        "Zarr forcing arrays have time chunks of up to $(maximum(chunks)) steps, longer than " *
            "the $time_chunk steps of the first $(FORCING_VARS[1]) store, which sets the " *
            "size of the read buffer. Rechunk the forcing to equal time chunks."
    )
    length(chunks) == 1 ||
        @warn "Zarr forcing arrays have different time chunk lengths ($chunks), reading will be slower"
    return nothing
end

"""
Open the forcing files, as NetCDF or Zarr depending on their file extension,
and set up the cache that the model reads its daily forcing from.
"""
function open_forcing(cfg::Cfg, domain::Domain; cache_bytes = FORCING_CACHE_BUDGET_BYTES)
    # Paths ending in ".zarr" are Zarr stores, anything else is NetCDF
    is_zarr = [endswith(getval(cfg.input.paths, "$(var)_file"), ".zarr") for var in FORCING_VARS]
    if !(all(is_zarr) || !any(is_zarr))
        error("Forcing paths must either all be Zarr stores (.zarr) or all NetCDF files")
    end
    sources, times, grid_size = all(is_zarr) ? open_forcing_zarr(cfg, domain) : open_forcing_netcdf(cfg)
    grid_size == domain.grid_size ||
        error("Forcing grid of size $grid_size does not match the parameter grid $(domain.grid_size)")
    nx, ny = grid_size
    n = ncells(domain)

    # Zarr is read ahead into a second cache, so each gets half of the budget
    prefetch = all(is_zarr)
    budget = prefetch ? cache_bytes ÷ 2 : cache_bytes
    bytes_per_step = n * sizeof(Float32) * length(FORCING_VARS)
    capacity = clamp(budget ÷ max(bytes_per_step, 1), 1, length(times))

    # Round the cache to whole Zarr chunks, at least one even if over budget
    time_chunk = time_chunk_length(sources)
    capacity = min(max(capacity ÷ time_chunk, 1) * time_chunk, length(times))

    cache = Dict{String, Matrix{Float32}}(
        var => Matrix{Float32}(undef, n, capacity) for var in FORCING_VARS
    )
    gridded_zarr = valtype(sources) <: ZarrForcingVar
    staging = Dict{String, Array{Float32, 3}}(
        var => Array{Float32, 3}(undef, nx, ny, gridded_zarr ? time_chunk : 0)
            for var in FORCING_VARS
    )
    next_cache = Dict{String, Matrix{Float32}}(
        var => Matrix{Float32}(undef, n, capacity) for var in FORCING_VARS if prefetch
    )

    return ForcingReaders(
        sources,
        times,
        cache,
        staging,
        domain.cells,
        0,
        0,
        capacity,
        time_chunk,
        next_cache,
        nothing,
        0,
        0,
        Channel{Vector{UInt8}}(Inf),
    )
end

"""
Number of timesteps in one chunk, from the Zarr metadata of the first forcing
variable (time is its third axis, see `check_zarr_dims`). 1 for NetCDF, which
is read per timestep.
"""
time_chunk_length(sources) = 1
time_chunk_length(sources::Dict{String, <:AnyZarrForcingVar}) =
    last(first(sources[FORCING_VARS[1]].arrays).metadata.chunks)

"""
Timestep at which the chunk holding timestep `idx` starts. Chunks are counted
from the first timestep of each store (e.g. each year).
"""
chunk_start(readers::ForcingReaders, idx::Int) =
    chunk_start(readers.sources[FORCING_VARS[1]], idx, readers.time_chunk)
chunk_start(src, idx::Int, time_chunk::Int) = idx
function chunk_start(src::AnyZarrForcingVar, idx::Int, time_chunk::Int)
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
Copy the active `cells` of the gridded (nx, ny, n) `block` to the cache, from timestep
`t0` of the cache on.
"""
function gather_cells!(cache::Matrix{Float32}, block::AbstractArray{<:Any, 3}, cells, t0::Int)
    nx, ny, n = size(block)
    flat = reshape(block, nx * ny, n)
    for t in 1:n, (c, cell) in enumerate(cells)
        @inbounds cache[c, t0 + t - 1] = flat[cell, t]
    end
    return nothing
end

"""
Read `len` timesteps, starting at `start`, from a NetCDF variable into the cache.
"""
function load_block!(cache::Matrix{Float32}, staging, cells, src::ForcingVar, start::Int, len::Int)
    raw = src[:, :, start:(start + len - 1)]
    # Missing values in the forcing file (its _FillValue) are read as `missing`: replace
    # them with NaN.
    block = raw isa Array{Float32, 3} ? raw : coalesce.(raw, NaN32)
    gather_cells!(cache, block, cells, 1)
    return nothing
end

"""
Read `len` timesteps, starting at `start`, from a Zarr variable into the cache,
one chunk per read, via the gridded `staging` buffer. Handles crossing year
boundaries. Time is the third axis (checked by `check_zarr_dims`).
"""
function load_block!(cache::Matrix{Float32}, staging::Array{Float32, 3}, cells, src::ZarrForcingVar, start::Int, len::Int)
    nx, ny, _ = size(staging)
    stop = start + len - 1
    step = start  # next timestep to read
    for (array, offset) in zip(src.arrays, src.offsets)
        time_chunk = array.metadata.chunks[3]
        store_stop = offset + size(array, 3) - 1
        while offset <= step <= min(stop, store_stop)
            # Read up to the end of the chunk that holds `step`
            chunk_stop = offset + ((step - offset) ÷ time_chunk + 1) * time_chunk - 1
            n = min(chunk_stop, stop, store_stop) - step + 1
            # unsafe_wrap does no bounds check: a larger read would write past the buffer
            n <= size(staging, 3) ||
                error("Zarr time chunk of $n steps does not fit the staging buffer of $(size(staging, 3)) steps")
            # Plain Array, not a view: Zarr then unpacks straight into the buffer (2-3x faster)
            GC.@preserve staging begin
                dest = unsafe_wrap(Array, pointer(staging), (nx, ny, n))
                Zarr.readblock!(
                    dest, array,
                    CartesianIndices((1:nx, 1:ny, (step - offset + 1):(step - offset + n)))
                )
                gather_cells!(cache, dest, cells, step - start + 1)
            end
            step += n
        end
    end
    return nothing
end

"""
Fill the cache, starting at the beginning of the chunk that holds timestep `idx`. When that
block was prefetched, this waits for the prefetch and swaps it in. Afterwards, the block
after the cache is prefetched.
"""
function fill_forcing_cache!(readers::ForcingReaders, idx::Int)
    start = chunk_start(readers, idx)
    len = min(readers.capacity, length(readers.times) - start + 1)

    task = readers.prefetch
    readers.prefetch = nothing
    if !isnothing(task) && (readers.prefetch_start, readers.prefetch_len) == (start, len)
        fetch(task)  # rethrows a failed read
        readers.cache, readers.next_cache = readers.next_cache, readers.cache
    else
        # A prefetch of another block also uses the staging buffers: let it finish first
        isnothing(task) || fetch(task)
        load_all_vars!(readers, readers.cache, start, len)
    end
    readers.cache_start = start
    readers.cache_len = len

    start_prefetch!(readers)
    return nothing
end

"""
Start reading the block after the cache into `next_cache`, in a task that runs while the
model computes. Only for Zarr forcing (NetCDF can only be read by one thread at a time).
"""
function start_prefetch!(readers::ForcingReaders)
    isempty(readers.next_cache) && return nothing
    next = readers.cache_start + readers.cache_len
    next > length(readers.times) && return nothing
    start = chunk_start(readers, next)
    len = min(readers.capacity, length(readers.times) - start + 1)
    readers.prefetch_start = start
    readers.prefetch_len = len
    readers.prefetch = Threads.@spawn load_all_vars!(readers, readers.next_cache, start, len)
    return nothing
end

"""For NetCDF, read `len` timesteps of all forcing variables into `cache`, one by one."""
function load_all_vars!(readers::ForcingReaders, cache, start::Int, len::Int)
    for var in FORCING_VARS
        load_block!(cache[var], readers.staging[var], readers.cells, readers.sources[var], start, len)
    end
    return nothing
end

"""For Zarr, read all variables at the same time, one task each (needs `julia -t` threads)."""
function load_all_vars!(readers::ForcingReaders{ZarrForcingVar}, cache, start::Int, len::Int)
    @sync for var in FORCING_VARS
        Threads.@spawn load_block!(cache[var], readers.staging[var], readers.cells, readers.sources[var], start, len)
    end
    return nothing
end

"""
For land-only Zarr, read every chunk of every variable in its own task (needs `julia -t`
threads), and unpack it straight into the cache.
"""
function load_all_vars!(readers::ForcingReaders{LandZarrForcingVar}, cache, start::Int, len::Int)
    stop = start + len - 1
    @sync for var in FORCING_VARS
        src = readers.sources[var]
        for (array, offset) in zip(src.arrays, src.offsets)
            nrows, nt = size(array)
            row_chunk, time_chunk = array.metadata.chunks
            # Timesteps of this array within start:stop, as indices into the array
            first_t = max(start, offset) - offset + 1
            last_t = min(stop, offset + nt - 1) - offset + 1
            first_t > last_t && continue
            for tc in cld(first_t, time_chunk):cld(last_t, time_chunk), rc in 1:cld(nrows, row_chunk)
                Threads.@spawn load_land_chunk!(
                    cache[var], src, array, readers.buffers, rc, tc,
                    max(first_t, (tc - 1) * time_chunk + 1):min(last_t, tc * time_chunk),
                    offset - start,
                )
            end
        end
    end
    return nothing
end

"""Whether the chunks of `array` are only Blosc compressed, so they can be decompressed directly."""
plain_blosc(p::Zarr.V2Pipeline) = p.compressor isa Zarr.BloscCompressor && isnothing(p.filters)
plain_blosc(p::Zarr.V3Pipeline) =
    isempty(p.array_array) && p.array_bytes isa Zarr.Codecs.V3Codecs.BytesCodec &&
    length(p.bytes_bytes) == 1 && only(p.bytes_bytes) isa Zarr.Codecs.V3Codecs.BloscV3Codec &&
    ENDIAN_BOM == 0x04030201  # the bytes codec is little endian; so is the host
plain_blosc(p) = false

"""
Read chunk (`rc`, `tc`) of the land-only `array` of `src`, and unpack its timesteps `ts`
(indices into the array) into `cache`. Timestep `t` of the array goes to cache column
`t + col_offset`.
"""
function load_land_chunk!(cache, src::LandZarrForcingVar, array, buffers, rc, tc, ts, col_offset)
    T = eltype(array)
    row_chunk, time_chunk = array.metadata.chunks
    first_row = (rc - 1) * row_chunk + 1
    rows = first_row:min(rc * row_chunk, size(array, 1))
    bytes = Zarr.store_readchunk(array.storage, array.path, CartesianIndex(rc, tc), array.metadata.chunk_key_encoding)
    if isnothing(bytes)  # chunk not written: all missing
        for t in ts, r in rows
            c = src.rows[r]
            c > 0 && (cache[c, t + col_offset] = NaN32)
        end
        return nothing
    end

    pipeline = Zarr.get_pipeline(array.metadata)
    if plain_blosc(pipeline)
        buf = isready(buffers) ? take!(buffers) : UInt8[]
        Blosc.decompress!(buf, bytes)
        length(buf) == row_chunk * time_chunk * sizeof(T) ||
            error("Land-only forcing chunk $((rc, tc)) of '$(array.path)' has an unexpected size")
        GC.@preserve buf begin
            values = unsafe_wrap(Array, Ptr{T}(pointer(buf)), (row_chunk, time_chunk))
            unpack_land_chunk!(cache, values, src, rows, tc, ts, col_offset)
        end
        put!(buffers, buf)
    else
        values = Array{T}(undef, row_chunk, time_chunk)
        Zarr.pipeline_decode!(pipeline, values, bytes)
        unpack_land_chunk!(cache, values, src, rows, tc, ts, col_offset)
    end
    return nothing
end

"""Unpack the `rows` and timesteps `ts` of a decompressed chunk into the cache."""
function unpack_land_chunk!(cache::Matrix{Float32}, values::Matrix{T}, src::LandZarrForcingVar, rows, tc, ts, col_offset) where {T}
    (; scale, offset) = src
    has_fill = !isnan(src.fill)
    fill = has_fill ? T(src.fill) : zero(T)
    row_chunk, time_chunk = size(values)
    r0 = first(rows) - 1
    @inbounds for t in ts
        col = t + col_offset
        tl = t - (tc - 1) * time_chunk  # timestep within the chunk
        for j in 1:length(rows)
            c = src.rows[r0 + j]
            c > 0 || continue
            x = values[j, tl]
            cache[c, col] = has_fill & (x == fill) ? NaN32 : Float32(x) * scale + offset
        end
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
    dest = Vector{Float32}(undef, size(readers.cache[var], 1))
    read_var!(dest, time, readers, var)
    return dest
end

"""
Initialize forcing reader and read the first timestep of
forcing data.
"""
function initialize_forcing(cfg::Cfg, domain::Domain)
    forcing_readers = open_forcing(cfg, domain)
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
