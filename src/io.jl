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

"""
Gridded NetCDF file, (lon, lat, time) or (lon, lat, time, layer). Written serially. Packed
outputs are written as their raw Int16 values (see [`Packing`](@ref)).
"""
struct NetCDFStore <: OutputStore
    ds::NCDataset
    arrays::Vector{Any}
    bands::GridBands
    time_chunk::Int
    grids::Dict{DataType, Array}  # per stored type, a buffer for a band of rows: (nx, rows, time_chunk, nlayers)
end

close_output(store::NetCDFStore) = close(store.ds)

"""
Create the Zarr v3 array `name` in `store`, of element type `T`, size `dims` and chunks of
size `chunks`, with dimension names `dim_names` (in Python's row-major order). Outputs are
`compress`ed with [`OUTPUT_COMPRESSOR`](@ref), and stored as Int16 if they have a `packing`:
they still read as Float32 (with ZarrScaleCast.jl, or another reader that knows the
`scale_offset` and `cast_value` codecs). mGV compresses and writes the chunks itself (see
[`write_chunk!`](@ref)), so the codecs here have to match what it writes.
"""
function create_zarr_v3(store, name, T, dims, chunks, dim_names; attrs = Dict{String, Any}(), compress = false, packing = nothing)
    array_array = isnothing(packing) ? () : (
            ScaleOffsetCodec(offset = packing.offset, scale = packing.scale),
            CastValueCodec(
                Int16; rounding = RoundNearest, out_of_range = :clamp,
                encode_map = [NaN => PACKED_NAN], decode_map = [PACKED_NAN => NaN],
            ),
        )
    bytes_bytes = compress ? (
            Zarr.Codecs.V3Codecs.BloscV3Codec(
                OUTPUT_COMPRESSOR.cname, OUTPUT_COMPRESSOR.clevel, OUTPUT_COMPRESSOR.shuffle, 0,
                sizeof(stored_type(packing)),
            ),
        ) : ()
    pipeline = Zarr.V3Pipeline(array_array, Zarr.Codecs.V3Codecs.BytesCodec(), bytes_bytes)
    fill = T <: AbstractFloat ? T(NaN) : zero(T)
    N = length(dims)
    metadata = Zarr.MetadataV3{T, N, typeof(pipeline)}(
        3, "array", Tuple(dims), Tuple(chunks), Zarr.typestr3(T), pipeline, fill, Zarr.ChunkKeyEncoding('/', true),
    )
    # Zarr.jl 0.10 writes neither the attributes nor the dimension names with the metadata
    json = JSON.lower(metadata)
    json["attributes"] = attrs
    json["dimension_names"] = dim_names
    store[name, "zarr.json"] = Vector{UInt8}(JSON.json(json))
    return ZArray(metadata, store, name, attrs, true)
end

"""
Open a new output store for `year` at `path` (removing an existing one), in `format` (one
of $(join(repr.(OUTPUT_FORMATS), ", "))). Timesteps are chunked by `time_chunk`. Output `i`
is stored with `packings[i]`. Zarr stores are Zarr v3.
"""
function create_output_store(format, path, year, domain, nt, time_chunk, nlayers, lat, lon, outputs, packings)
    time_chunk = min(time_chunk, nt)
    if format == "netcdf"
        return create_output_netcdf(path, domain, nt, time_chunk, nlayers, lat, lon, outputs, packings)
    end

    println("Initializing Zarr store at: $path")
    isdir(path) && rm(path, recursive = true)
    store = Zarr.DirectoryStore(path)
    zgroup(store, "", Zarr.ZarrFormat(3))

    make_zarr(name, T, dims, chunks, dim_names; kwargs...) = create_zarr_v3(store, name, T, dims, chunks, dim_names; kwargs...)
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
        arrays = map(outputs, nlayers, packings) do meta, n, packing
            is_layered(meta) ?
                make_zarr(
                    meta.output_name, Float32, (nx, ny, nt, n), (nx, bands.rows, time_chunk, n),
                    ["layer", "time", "lat", "lon"]; attrs = output_attributes(meta), compress = true, packing,
                ) :
                make_zarr(
                    meta.output_name, Float32, (nx, ny, nt), (nx, bands.rows, time_chunk),
                    ["time", "lat", "lon"]; attrs = output_attributes(meta), compress = true, packing,
                )
        end
        return GriddedZarrStore(arrays, bands, time_chunk)
    elseif format == "zarr_landonly"
        n = ncells(domain)
        nx = domain.grid_size[1]
        write_coord("lon_index", Int32.((domain.cells .- 1) .% nx), ["cell"])
        write_coord("lat_index", Int32.((domain.cells .- 1) .÷ nx), ["cell"])
        cell_chunk = clamp(OUTPUT_CHUNK_BYTES ÷ (time_chunk * sizeof(Float32)), 1, n)
        arrays = map(outputs, nlayers, packings) do meta, nl, packing
            is_layered(meta) ?
                make_zarr(
                    meta.output_name, Float32, (n, nt, nl), (cell_chunk, time_chunk, nl),
                    ["layer", "time", "cell"]; attrs = output_attributes(meta), compress = true, packing,
                ) :
                make_zarr(
                    meta.output_name, Float32, (n, nt), (cell_chunk, time_chunk),
                    ["time", "cell"]; attrs = output_attributes(meta), compress = true, packing,
                )
        end
        return LandZarrStore(arrays, cell_chunk, time_chunk)
    else
        error("Unknown output format '$format', valid formats are: $(join(OUTPUT_FORMATS, ", "))")
    end
end

function create_output_netcdf(path, domain, nt, time_chunk, nlayers, lat, lon, outputs, packings)
    nx, ny = domain.grid_size
    println("Creating NetCDF output file at: $path")
    ds = NCDataset(path, "c")

    defDim(ds, "lon", nx); defDim(ds, "lat", ny); defDim(ds, "time", nt)
    defDim(ds, "layer", maximum(nlayers))

    v = defVar(ds, "lat", Float32, ("lat",)); v[:] = lat; v.attrib["axis"] = "Y"
    v = defVar(ds, "lon", Float32, ("lon",)); v[:] = lon; v.attrib["axis"] = "X"

    bands = GridBands(domain, time_chunk)
    arrays = map(outputs, nlayers, packings) do meta, n, packing
        dims, chunks = is_layered(meta) ?
            (("lon", "lat", "time", "layer"), (nx, bands.rows, time_chunk, n)) :
            (("lon", "lat", "time"), (nx, bands.rows, time_chunk))
        attrib = output_attributes(meta)
        T = stored_type(packing)
        if !isnothing(packing)
            # CF packing: value = stored * scale_factor + add_offset
            attrib["scale_factor"] = 1 / packing.scale
            attrib["add_offset"] = packing.offset
        end
        v = defVar(
            ds, meta.output_name, T, dims;
            chunksizes = chunks, deflatelevel = 0, shuffle = false, attrib, fillvalue = fill_value(T),
        )
        # Write the stored values as they are, without NCDatasets' (un)packing
        return v.var
    end
    grids = Dict{DataType, Array}(
        T => Array{T}(undef, nx, bands.rows, time_chunk, maximum(nlayers)) for T in unique(stored_type.(packings))
    )
    return NetCDFStore(ds, arrays, bands, time_chunk, grids)
end

"""
Scatter the timesteps of `block` in band `b` into `buf`, as an (nx, rows, time_chunk,
nlayers) gridded chunk. Inactive cells, and timesteps after the block, are the fill value.
"""
function fill_grid_chunk!(buf::AbstractArray{T}, data::Array{T, 3}, len, bands::GridBands, b, time_chunk) where {T}
    nx = bands.domain.grid_size[1]
    plane = nx * bands.rows
    nlayers = size(data, 3)
    fill!(view(buf, 1:(plane * time_chunk * nlayers)), fill_value(T))
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
time_chunk, nlayers) land-only chunk. Missing cells and timesteps are the fill value.
"""
function fill_cell_chunk!(buf::Vector{T}, data::Array{T, 3}, len, rows, cell_chunk, time_chunk) where {T}
    n, block_length, nlayers = size(data)
    fill!(buf, fill_value(T))
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
thread. Chunks of only the fill value are not written: they read as NaN.
"""
function write_chunk!(array::ZArray, buf::Vector{T}, index::CartesianIndex, buffers::ChunkBuffers) where {T}
    all(isequal(fill_value(T)), buf) && return nothing
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
        Threads.@spawn write_grid_chunk!(array, data, block.len, store.bands, b, store.time_chunk, tc, buffers)
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
        Threads.@spawn write_cell_chunk!(array, data, block.len, c, store.cell_chunk, store.time_chunk, tc, buffers)
    end
    return nothing
end

"""Write chunk `(1, b, tc)` of a gridded Zarr `array`, from `data` of a block of length `len`."""
function write_grid_chunk!(array, data::Array{T, 3}, len, bands, b, time_chunk, tc, buffers) where {T}
    pool = value_pool(buffers, T)
    buf = acquire!(pool)
    fill_grid_chunk!(resize!(buf, prod(array.metadata.chunks)), data, len, bands, b, time_chunk)
    index = ndims(array) == 4 ? CartesianIndex(1, b, tc, 1) : CartesianIndex(1, b, tc)
    write_chunk!(array, buf, index, buffers)
    release!(pool, buf)
    return nothing
end

"""Write chunk `(c, tc)` of a land-only Zarr `array`, from `data` of a block of length `len`."""
function write_cell_chunk!(array, data::Array{T, 3}, len, c, cell_chunk, time_chunk, tc, buffers) where {T}
    pool = value_pool(buffers, T)
    buf = acquire!(pool)
    n = size(data, 1)
    rows = ((c - 1) * cell_chunk + 1):min(c * cell_chunk, n)
    resize!(buf, prod(array.metadata.chunks))
    fill_cell_chunk!(buf, data, len, rows, cell_chunk, time_chunk)
    index = ndims(array) == 3 ? CartesianIndex(c, tc, 1) : CartesianIndex(c, tc)
    write_chunk!(array, buf, index, buffers)
    release!(pool, buf)
    return nothing
end

"""Write a block of timesteps to a NetCDF file, one band of rows at a time (serially)."""
function write_block!(store::NetCDFStore, block::OutputBlock, ::ChunkBuffers)
    ny = store.bands.domain.grid_size[2]
    (; rows) = store.bands
    ts = block.first:(block.first + block.len - 1)
    for (array, data) in zip(store.arrays, block.data), b in eachindex(store.bands.cells)
        grid = store.grids[eltype(data)]
        fill_grid_chunk!(grid, data, block.len, store.bands, b, store.time_chunk)
        r = ((b - 1) * rows + 1):min(b * rows, ny)
        # The chunk fills the first layers of the buffer; the last band can have fewer rows
        if ndims(array) == 4
            array[:, r, ts, :] = view(grid, :, 1:length(r), 1:block.len, 1:size(data, 3))
        else
            array[:, r, ts] = view(grid, :, 1:length(r), 1:block.len, 1)
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
    packings = Union{Packing, Nothing}[
        config.output.packing ? meta.packing : nothing for meta in selected_outputs
    ]

    function open_store(year)
        nt = (DateTime(year + 1) - DateTime(year)) ÷ dt
        extension = format == "netcdf" ? "nc" : "zarr"
        path = joinpath(config.output.dir, "$(config.output.file_prefix)$(year).$extension")
        return create_output_store(format, path, year, domain, nt, block_length, nlayers, lat, lon, selected_outputs, packings)
    end

    println("Starting output writer...")
    backend = get_backend(grid_parameters.elevation)
    return OutputWriter(open_store, year, selected_outputs, packings, ncells(domain), nlayers, block_length, backend)
end
