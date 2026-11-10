include("async_writer.jl")

const OUTPUT_COMPRESSOR = Zarr.BloscCompressor(cname = "lz4", clevel = 1, shuffle = 1)

# Target size of an (uncompressed) output chunk. Chunks are compressed in parallel, so
# they should be small enough to give every thread work, and large enough to keep the
# number of files manageable.
const OUTPUT_CHUNK_BYTES = 16 * 1024^2

const OUTPUT_FORMATS = Config.OUTPUT_FORMATS

"""Whether an output variable has an additional soil layer dimension."""
is_layered(meta::ParameterMetadata) = meta.gridtype == GRIDS["soil_grid"]

"""Attributes of an output variable in the output file."""
output_attributes(meta::ParameterMetadata) =
    Dict{String, Any}("units" => to_string(meta.unit), "long_name" => meta.description)

"""
An output file or store for one year. Outputs are written one block of timesteps at a
time by [`write_block!`](@ref), on the writer task.
"""
abstract type OutputStore end

close_output(::OutputStore) = nothing

"""
Gridded chunks of `rows` latitude rows (and every longitude). `cells[b]` are the active
cells (indices into `domain.cells`) in band `b` of rows.
"""
struct GridBands
    domain::Domain
    rows::Int
    cells::Vector{UnitRange{Int}}
end

function GridBands(domain::Domain, time_chunk)
    nx, ny = domain.grid_size
    rows = clamp(OUTPUT_CHUNK_BYTES ÷ (nx * time_chunk * sizeof(Float32)), 1, ny)
    # The cells are in grid order, so the cells of a band are contiguous
    cells = [
        searchsortedfirst(domain.cells, (b - 1) * rows * nx + 1):searchsortedlast(domain.cells, min(b * rows, ny) * nx)
            for b in 1:cld(ny, rows)
    ]
    return GridBands(domain, rows, cells)
end

"""Gridded Zarr store: (lon, lat, time) arrays, or (lon, lat, time, layer)."""
struct GriddedZarrStore <: OutputStore
    arrays::Vector{ZArray}  # one per output, in the order of the selected outputs
    bands::GridBands
    time_chunk::Int
end

"""
Land-only Zarr store: (cell, time) arrays, or (cell, time, layer), with only the active
cells. The same layout as the land-only forcing, see `scripts/convert_forcing_landonly.jl`.
"""
struct LandZarrStore <: OutputStore
    arrays::Vector{ZArray}
    cell_chunk::Int
    time_chunk::Int
end

"""Gridded NetCDF file, (lon, lat, time) or (lon, lat, time, layer). Written serially."""
struct NetCDFStore <: OutputStore
    ds::NCDataset
    arrays::Vector{Any}
    bands::GridBands
    time_chunk::Int
    grid::Array{Float32, 4}  # buffer for a band of rows: (nx, rows, time_chunk, nlayers)
end

close_output(store::NetCDFStore) = close(store.ds)

"""
Open a new output store for `year` at `path` (removing an existing one), in `format` (one
of $(join(repr.(OUTPUT_FORMATS), ", "))). Timesteps are chunked by `time_chunk`.
"""
function create_output_store(format, path, year, domain, nt, time_chunk, nlayers, lat, lon, outputs)
    time_chunk = min(time_chunk, nt)
    if format == "netcdf"
        return create_output_netcdf(path, domain, nt, time_chunk, nlayers, lat, lon, outputs)
    end

    println("Initializing Zarr store at: $path")
    isdir(path) && rm(path, recursive = true)
    group = zgroup(path)

    function make_zarr(name, T, dims, chunks, dim_names; attrs = Dict{String, Any}(), compress = false)
        full_attrs = Dict{String, Any}(attrs)
        full_attrs["_ARRAY_DIMENSIONS"] = dim_names
        return zcreate(
            T, group, name, dims...;
            chunks, fill_value = T <: AbstractFloat ? T(NaN) : nothing, attrs = full_attrs,
            compressor = compress ? OUTPUT_COMPRESSOR : Zarr.NoCompressor(),
        )
    end
    function write_coord(name, values, dim_names; attrs = Dict{String, Any}())
        make_zarr(name, eltype(values), (length(values),), (length(values),), dim_names; attrs)[:] = values
        return nothing
    end

    write_coord(
        "time", Float32.(0:(nt - 1)), ["time"];
        attrs = Dict("units" => "days since $year-01-01", "calendar" => "proleptic_gregorian"),
    )
    write_coord("lat", lat, ["lat"]; attrs = Dict("units" => "degrees_north", "axis" => "Y"))
    write_coord("lon", lon, ["lon"]; attrs = Dict("units" => "degrees_east", "axis" => "X"))

    # Dimension names are reversed, to match Python's row-major order
    if format == "zarr"
        nx, ny = domain.grid_size
        bands = GridBands(domain, time_chunk)
        arrays = map(outputs, nlayers) do meta, n
            is_layered(meta) ?
                make_zarr(
                    meta.output_name, Float32, (nx, ny, nt, n), (nx, bands.rows, time_chunk, n),
                    ["layer", "time", "lat", "lon"]; attrs = output_attributes(meta), compress = true,
                ) :
                make_zarr(
                    meta.output_name, Float32, (nx, ny, nt), (nx, bands.rows, time_chunk),
                    ["time", "lat", "lon"]; attrs = output_attributes(meta), compress = true,
                )
        end
        return GriddedZarrStore(arrays, bands, time_chunk)
    elseif format == "zarr_landonly"
        n = ncells(domain)
        nx = domain.grid_size[1]
        write_coord("lon_index", Int32.((domain.cells .- 1) .% nx), ["cell"])
        write_coord("lat_index", Int32.((domain.cells .- 1) .÷ nx), ["cell"])
        cell_chunk = clamp(OUTPUT_CHUNK_BYTES ÷ (time_chunk * sizeof(Float32)), 1, n)
        arrays = map(outputs, nlayers) do meta, nl
            is_layered(meta) ?
                make_zarr(
                    meta.output_name, Float32, (n, nt, nl), (cell_chunk, time_chunk, nl),
                    ["layer", "time", "cell"]; attrs = output_attributes(meta), compress = true,
                ) :
                make_zarr(
                    meta.output_name, Float32, (n, nt), (cell_chunk, time_chunk),
                    ["time", "cell"]; attrs = output_attributes(meta), compress = true,
                )
        end
        return LandZarrStore(arrays, cell_chunk, time_chunk)
    else
        error("Unknown output format '$format', valid formats are: $(join(OUTPUT_FORMATS, ", "))")
    end
end

function create_output_netcdf(path, domain, nt, time_chunk, nlayers, lat, lon, outputs)
    nx, ny = domain.grid_size
    println("Creating NetCDF output file at: $path")
    ds = NCDataset(path, "c")

    defDim(ds, "lon", nx); defDim(ds, "lat", ny); defDim(ds, "time", nt)
    defDim(ds, "layer", maximum(nlayers))

    v = defVar(ds, "lat", Float32, ("lat",)); v[:] = lat; v.attrib["axis"] = "Y"
    v = defVar(ds, "lon", Float32, ("lon",)); v[:] = lon; v.attrib["axis"] = "X"

    bands = GridBands(domain, time_chunk)
    arrays = map(outputs, nlayers) do meta, n
        dims, chunks = is_layered(meta) ?
            (("lon", "lat", "time", "layer"), (nx, bands.rows, time_chunk, n)) :
            (("lon", "lat", "time"), (nx, bands.rows, time_chunk))
        defVar(
            ds, meta.output_name, Float32, dims;
            chunksizes = chunks, deflatelevel = 0, shuffle = false, attrib = output_attributes(meta),
        )
    end
    grid = Array{Float32}(undef, nx, bands.rows, time_chunk, maximum(nlayers))
    return NetCDFStore(ds, arrays, bands, time_chunk, grid)
end

"""
Scatter the timesteps of `block` in band `b` into `buf`, as an (nx, rows, time_chunk,
nlayers) gridded chunk. Inactive cells, and timesteps after the block, are NaN.
"""
function fill_grid_chunk!(buf::AbstractArray{Float32}, data::Array{Float32, 3}, len, bands::GridBands, b, time_chunk)
    nx = bands.domain.grid_size[1]
    plane = nx * bands.rows
    nlayers = size(data, 3)
    fill!(view(buf, 1:(plane * time_chunk * nlayers)), NaN32)
    offset = (b - 1) * plane  # linear grid index of the band start, minus one
    cells = bands.domain.cells
    @inbounds for l in 1:nlayers, t in 1:len
        base = ((l - 1) * time_chunk + t - 1) * plane - offset
        for k in bands.cells[b]
            buf[base + cells[k]] = data[k, t, l]
        end
    end
    return buf
end

"""
Copy the timesteps of `block` for the cells `rows` into `buf`, as a (cell_chunk,
time_chunk, nlayers) land-only chunk. Missing cells and timesteps are NaN.
"""
function fill_cell_chunk!(buf::Vector{Float32}, data::Array{Float32, 3}, len, rows, cell_chunk, time_chunk)
    n, block_length, nlayers = size(data)
    fill!(buf, NaN32)
    for l in 1:nlayers, t in 1:len
        copyto!(
            buf, ((l - 1) * time_chunk + t - 1) * cell_chunk + 1,
            data, ((l - 1) * block_length + t - 1) * n + first(rows), length(rows),
        )
    end
    return buf
end

"""
Compress the chunk `buf` and write it as chunk `index` of the Zarr `array`, from any
thread. All-NaN chunks are not written: they read as the fill value (NaN).
"""
function write_chunk!(array::ZArray, buf::Vector{Float32}, index::CartesianIndex, buffers::ChunkBuffers)
    all(isnan, buf) && return nothing
    bytes = acquire!(buffers.bytes)
    resize!(bytes, sizeof(buf) + Blosc.MAX_OVERHEAD)
    n = Blosc.compress!(bytes, buf; level = OUTPUT_COMPRESSOR.clevel, shuffle = OUTPUT_COMPRESSOR.shuffle)
    Zarr.store_writechunk(array.storage, resize!(bytes, n), array.path, index, array.metadata.chunk_key_encoding)
    release!(buffers.bytes, bytes)
    return nothing
end

"""Index of the time chunk that starts at time index `first`."""
function time_chunk_index(first, time_chunk)
    (first - 1) % time_chunk == 0 || error("Output block at timestep $first is not aligned with the time chunks")
    return (first - 1) ÷ time_chunk + 1
end

"""
Write a block of timesteps to a gridded Zarr store: every band of every output is a chunk,
which is compressed and written in its own task (needs `julia -t` threads).
"""
function write_block!(store::GriddedZarrStore, block::OutputBlock, buffers::ChunkBuffers)
    tc = time_chunk_index(block.first, store.time_chunk)
    @sync for (array, data) in zip(store.arrays, block.data), b in eachindex(store.bands.cells)
        isempty(store.bands.cells[b]) && continue  # no land: the chunk stays NaN
        Threads.@spawn begin
            buf = acquire!(buffers.values)
            chunk = prod(array.metadata.chunks)
            fill_grid_chunk!(resize!(buf, chunk), data, block.len, store.bands, b, store.time_chunk)
            index = ndims(array) == 4 ? CartesianIndex(1, b, tc, 1) : CartesianIndex(1, b, tc)
            write_chunk!(array, buf, index, buffers)
            release!(buffers.values, buf)
        end
    end
    return nothing
end

"""
Write a block of timesteps to a land-only Zarr store: every chunk of cells of every output
is compressed and written in its own task (needs `julia -t` threads).
"""
function write_block!(store::LandZarrStore, block::OutputBlock, buffers::ChunkBuffers)
    tc = time_chunk_index(block.first, store.time_chunk)
    n = size(first(block.data), 1)
    @sync for (array, data) in zip(store.arrays, block.data), c in 1:cld(n, store.cell_chunk)
        Threads.@spawn begin
            buf = acquire!(buffers.values)
            rows = ((c - 1) * store.cell_chunk + 1):min(c * store.cell_chunk, n)
            resize!(buf, prod(array.metadata.chunks))
            fill_cell_chunk!(buf, data, block.len, rows, store.cell_chunk, store.time_chunk)
            index = ndims(array) == 3 ? CartesianIndex(c, tc, 1) : CartesianIndex(c, tc)
            write_chunk!(array, buf, index, buffers)
            release!(buffers.values, buf)
        end
    end
    return nothing
end

"""Write a block of timesteps to a NetCDF file, one band of rows at a time (serially)."""
function write_block!(store::NetCDFStore, block::OutputBlock, ::ChunkBuffers)
    ny = store.bands.domain.grid_size[2]
    (; rows) = store.bands
    ts = block.first:(block.first + block.len - 1)
    for (array, data) in zip(store.arrays, block.data), b in eachindex(store.bands.cells)
        fill_grid_chunk!(store.grid, data, block.len, store.bands, b, store.time_chunk)
        r = ((b - 1) * rows + 1):min(b * rows, ny)
        # The chunk fills the first layers of the buffer; the last band can have fewer rows
        if ndims(array) == 4
            array[:, r, ts, :] = view(store.grid, :, 1:length(r), 1:block.len, 1:size(data, 3))
        else
            array[:, r, ts] = view(store.grid, :, 1:length(r), 1:block.len, 1)
        end
    end
    return nothing
end

"""
Start the output writer for the `selected_outputs`, writing to the store of `year`. Each
year gets its own file or store in `config.output.dir`.
"""
function start_output_writer(config::Cfg, domain::Domain, grid_parameters, soil_layers, year, dt, selected_outputs)
    format = lowercase(config.output.format)
    lat, lon = domain.latitude, domain.longitude  # coordinates of the grid
    block_length = config.output.write_interval
    nlayers = [is_layered(meta) ? soil_layers : 1 for meta in selected_outputs]

    function open_store(year)
        nt = (DateTime(year + 1) - DateTime(year)) ÷ dt
        extension = format == "netcdf" ? "nc" : "zarr"
        path = joinpath(config.output.dir, "$(config.output.file_prefix)$(year).$extension")
        return create_output_store(format, path, year, domain, nt, block_length, nlayers, lat, lon, selected_outputs)
    end

    println("Starting output writer...")
    backend = get_backend(grid_parameters.elevation)
    return OutputWriter(open_store, year, selected_outputs, ncells(domain), nlayers, block_length, backend)
end
