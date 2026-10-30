# Convert gridded Zarr forcing (e.g. WFDE5) to land-only stores, one per year, which mGV reads
# much faster: only the active cells of the parameter file (run_cell > 0) are stored, as
# (cell, time) arrays.
#
#   julia --project=<env with mGV> -t 8 scripts/convert_forcing_landonly.jl \
#       <gridded forcing.zarr> <parameter file> <output_{year}.zarr> <first year> [<last year>]
#
# Every (lon, lat, time) array of the source is converted, keeping its type and encoding (e.g.
# int16 with CF scale_factor and add_offset), compressed with Blosc zstd. Each store also holds
# the lon/lat coordinates of the grid and the 0-based grid indices of every cell (lon_index,
# lat_index), which mGV uses to match the cells to its domain. CELL_CHUNK sets the number of
# cells per chunk (default 32768); the time chunks are those of the source.
using mGV, Dates
using mGV: Zarr
using .Zarr: zopen, zgroup, zcreate

const CELL_CHUNK = parse(Int, get(ENV, "CELL_CHUNK", "32768"))
const COMPRESSOR = Zarr.BloscCompressor(cname = "zstd", clevel = 5, shuffle = 1)

# Let Blosc compress in several threads at once
ENV["BLOSC_NOLOCK"] = "1"

"""The active cells of the parameter file, as linear indices into its (nx, ny) grid."""
function active_cells(param_file)
    ds = mGV.open_param_source(param_file)
    run_cell = mGV.readfull(ds, "run_cell", 0)
    mGV.close_param_source(ds)
    return findall(vec(run_cell .> 0)), size(run_cell)
end

"""Attributes of a source array, without `_FillValue` (the Zarr fill value takes its place)."""
function copy_attrs(array, dims)
    attrs = Dict{String, Any}(k => v for (k, v) in array.attrs if k != "_FillValue")
    attrs["_ARRAY_DIMENSIONS"] = dims
    return attrs
end

function write_vector(group, name, values, dims; attrs = Dict{String, Any}())
    attrs = merge(attrs, Dict{String, Any}("_ARRAY_DIMENSIONS" => dims))
    z = zcreate(eltype(values), group, name, length(values); chunks = (length(values),), attrs)
    z[:] = values
    return z
end

"""Copy timesteps `ts` of the gridded `src` array, at the `cells`, to the land-only `dest`."""
function convert_var!(dest, src, cells, ts)
    nx, ny, _ = size(src)
    tchunk = last(dest.metadata.chunks)
    grid = Array{eltype(src)}(undef, nx, ny, tchunk)
    land = Matrix{eltype(src)}(undef, length(cells), tchunk)
    for b in 1:tchunk:length(ts)
        m = min(tchunk, length(ts) - b + 1)
        block = m == tchunk ? grid : Array{eltype(src)}(undef, nx, ny, m)
        Zarr.readblock!(block, src, CartesianIndices((1:nx, 1:ny, ts[b:(b + m - 1)])))
        flat = reshape(block, nx * ny, m)
        for t in 1:m, (k, c) in enumerate(cells)
            @inbounds land[k, t] = flat[c, t]
        end
        dest[:, b:(b + m - 1)] = view(land, :, 1:m)
    end
    return nothing
end

function convert_year(source_path, source, cells, year, out)
    if isdir(out)
        println("Skipping existing: $out")
        return
    end
    times = mGV.read_zarr_time(source["time"])
    ts = findall(t -> Dates.year(t) == year, times)
    isempty(ts) && error("'$source_path' has no timesteps in $year")
    ts = first(ts):last(ts)
    vars = sort([name for (name, a) in source.arrays if ndims(a) == 3])
    nx = length(source["lon"])
    println("$year: $(length(ts)) days of $(join(vars, ", ")) -> $out")

    tmp = out * ".tmp"
    isdir(tmp) && rm(tmp; recursive = true)
    group = zgroup(tmp; attrs = Dict("source" => source_path, "description" => "Land-only forcing for mGV"))

    write_vector(group, "lon", source["lon"][:], ["lon"]; attrs = copy_attrs(source["lon"], ["lon"]))
    write_vector(group, "lat", source["lat"][:], ["lat"]; attrs = copy_attrs(source["lat"], ["lat"]))
    write_vector(group, "lon_index", Int32.((cells .- 1) .% nx), ["cell"])
    write_vector(group, "lat_index", Int32.((cells .- 1) .÷ nx), ["cell"])
    write_vector(group, "time", source["time"][ts], ["time"]; attrs = copy_attrs(source["time"], ["time"]))

    # Create the arrays first (the group is not thread-safe), then fill them in parallel
    dests = map(vars) do name
        src = source[name]
        fill = get(src.attrs, "_FillValue", src.metadata.fill_value)
        zcreate(
            eltype(src), group, name, length(cells), length(ts);
            chunks = (CELL_CHUNK, last(src.metadata.chunks)), compressor = COMPRESSOR,
            fill_value = isnothing(fill) ? nothing : eltype(src)(fill),
            attrs = copy_attrs(src, ["time", "cell"]),
        )
    end
    @sync for (name, dest) in zip(vars, dests)
        Threads.@spawn convert_var!(dest, source[name], cells, ts)
    end
    mv(tmp, out)
    return nothing
end

function main(args)
    length(args) in (4, 5) || error(
        "Usage: convert_forcing_landonly.jl <gridded forcing.zarr> <parameter file> " *
            "<output_{year}.zarr> <first year> [<last year>]"
    )
    source_path, param_file, out_pattern = args[1:3]
    years = parse(Int, args[4]):parse(Int, get(args, 5, args[4]))
    occursin("{year}", out_pattern) || length(years) == 1 ||
        error("The output path needs '{year}' to convert more than one year")

    source = zopen(source_path)
    cells, grid_size = active_cells(param_file)
    grid_size == (length(source["lon"]), length(source["lat"])) ||
        error("Parameter grid $grid_size does not match the forcing grid")
    println("$(length(cells)) active cells of the $grid_size grid")
    for year in years
        t = @elapsed convert_year(source_path, source, cells, year, replace(out_pattern, "{year}" => string(year)))
        println("$year done in $(round(t, digits = 1)) s")
    end
    return nothing
end

main(ARGS)
