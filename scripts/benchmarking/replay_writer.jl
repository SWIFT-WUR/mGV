# Replay the output writer on a block of real output, without running the model.
#
#   julia -t auto scripts/replay_writer.jl SOURCE TIME_CHUNK OUT_DIR [FORMAT] [NREP]
#
# Rebuilds the block of time chunk TIME_CHUNK (1-based) from the land-only Zarr store SOURCE
# written by mGV, in the element types it was stored with (Int16 when packed, else Float32).
# Then writes it NREP times (default 10) with `mGV.write_block!` to a new store of FORMAT
# (default "zarr_landonly") in OUT_DIR, and reports the time per block and the GC counters.
# Run it under a profiler with `MGV_PROFILE=1`.

using mGV, Printf, Statistics
using Zarr: Zarr, JSON, Blosc

source, tc = ARGS[1], parse(Int, ARGS[2])
out_dir = ARGS[3]
format = length(ARGS) >= 4 ? ARGS[4] : "zarr_landonly"
nrep = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 10

function decompress(T, bytes)
    nbytes, _, _ = Blosc.cbuffer_sizes(bytes)
    out = Vector{T}(undef, nbytes ÷ sizeof(T))
    n = ccall((:blosc_decompress_ctx, Blosc.libblosc), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint), bytes, out, nbytes, 1)
    n == nbytes || error("Decompression failed")
    return out
end

# The domain, from the cell indices in the store
lon_index = Zarr.zopen(joinpath(source, "lon_index"))[:]
lat_index = Zarr.zopen(joinpath(source, "lat_index"))[:]
lon = Zarr.zopen(joinpath(source, "lon"))[:]
lat = Zarr.zopen(joinpath(source, "lat"))[:]
nx, ny = length(lon), length(lat)
domain = mGV.Domain((nx, ny), Int.(lon_index) .+ 1 .+ Int.(lat_index) .* nx, Float64.(lon), Float64.(lat))
n = mGV.ncells(domain)

metas = Dict(meta.output_name => meta for meta in values(mGV.output_variables()))
outputs, packings, data, nlayers = mGV.ParameterMetadata[], Union{mGV.Packing, Nothing}[], Array[], Int[]
block_length = 0
for name in sort(collect(keys(metas)))
    dir = joinpath(source, name)
    isfile(joinpath(dir, "zarr.json")) || continue
    meta = JSON.parsefile(joinpath(dir, "zarr.json"))
    packed = any(c -> c["name"] == "scale_offset", meta["codecs"])
    T = packed ? Int16 : Float32
    shape = reverse(meta["shape"])  # (cell, time[, layer])
    chunks = reverse(meta["chunk_grid"]["configuration"]["chunk_shape"])
    cc, tlen = chunks[1], chunks[2]
    nl = length(shape) == 3 ? shape[3] : 1
    global block_length = tlen
    a = fill(mGV.fill_value(T), n, tlen, nl)
    for c in 1:cld(n, cc)
        key = length(shape) == 3 ? "c/0/$(tc - 1)/$(c - 1)" : "c/$(tc - 1)/$(c - 1)"
        file = joinpath(dir, key)
        isfile(file) || continue  # all fill value
        chunk = reshape(decompress(T, read(file)), cc, tlen, nl)
        rows = ((c - 1) * cc + 1):min(c * cc, n)
        a[rows, :, :] .= view(chunk, 1:length(rows), :, :)
    end
    push!(outputs, metas[name])
    push!(packings, packed ? metas[name].packing : nothing)
    push!(data, a)
    push!(nlayers, nl)
end
bytes = sum(sizeof, data)
@printf("Block: %d outputs, %d cells, %d timesteps, %.0f MiB\n", length(outputs), n, block_length, bytes / 2^20)

mkpath(out_dir)
# What `OutputWriter` sets up in mGV versions that compress with the global Blosc functions
ENV["BLOSC_NOLOCK"] = "1"
Blosc.set_compressor(mGV.OUTPUT_COMPRESSOR.cname)
path = joinpath(out_dir, "replay.$(format == "netcdf" ? "nc" : "zarr")")
nt = 365
store = mGV.create_output_store(format, path, 1979, domain, nt, block_length, nlayers, lat, lon, outputs, packings)
block = mGV.OutputBlock(data, zeros(Int32, length(outputs), block_length), nothing, store, 1, block_length)
buffers = mGV.ChunkBuffers()

mGV.write_block!(store, block, buffers)  # compile
times = Float64[]
GC.gc()
gc0 = Base.gc_num()
profile = get(ENV, "MGV_PROFILE", "") == "1"
if profile
    @eval using Profile
    Profile.init(n = 10^7, delay = 0.0005)
end
for i in 1:nrep
    # Different time chunks, as in a run (keeps the page cache from reusing the same files)
    block.first = (i % (nt ÷ block_length)) * block_length + 1
    if profile
        push!(times, @elapsed @eval Profile.@profile mGV.write_block!(store, block, buffers))
    else
        push!(times, @elapsed mGV.write_block!(store, block, buffers))
    end
end
gc1 = Base.gc_num()
mGV.close_output(store)
@printf(
    "Per block: median %.1f ms, min %.1f ms, max %.1f ms (%.2f GB/s of block data)\n",
    1.0e3 * median(times), 1.0e3 * minimum(times), 1.0e3 * maximum(times), bytes / median(times) / 1.0e9
)
@printf(
    "GC: %.3f s total, %.3f s to safepoints, %.2f GiB allocated\n",
    (gc1.total_time - gc0.total_time) / 1.0e9, (gc1.total_time_to_safepoint - gc0.total_time_to_safepoint) / 1.0e9,
    (gc1.allocd - gc0.allocd + gc1.deferred_alloc - gc0.deferred_alloc) / 2^30
)
if profile
    @eval Profile.print(format = :flat, sortedby = :count, mincount = 20, noisefloor = 2)
end
