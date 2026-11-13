# Benchmark Blosc codecs on the chunks of one output block, offline.
#
#   julia -t auto scripts/benchmark_codecs.jl STORE TIME_CHUNK [NTASKS]
#
# Reads the chunks of time chunk TIME_CHUNK (1-based) of every output array in the Zarr
# STORE written by mGV (land-only or gridded, packed or not), and decompresses them: that is
# the input of the compressor. Then compresses all of them in NTASKS parallel tasks (default:
# the number of threads) with each codec, and reports the compression ratio and the
# throughput (uncompressed bytes per second of wall time), in total and per variable.

using Zarr, Printf
using Zarr: JSON
using Zarr: Blosc
const libblosc = Blosc.libblosc

store, tc = ARGS[1], parse(Int, ARGS[2])
ntasks = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : Threads.nthreads()

"""All chunk files of array `dir` with time chunk index `tc` (dimension names give its position)."""
function chunk_files(dir, tc)
    meta = JSON.parsefile(joinpath(dir, "zarr.json"))
    dims = meta["dimension_names"]  # in row-major order, so the chunk key order
    itime = findfirst(==("time"), dims)
    isnothing(itime) && return String[], 0  # a coordinate
    typesize = 4
    for c in meta["codecs"]
        c["name"] == "blosc" && (typesize = c["configuration"]["typesize"])
    end
    files = String[]
    for (root, _, names) in walkdir(joinpath(dir, "c")), name in names
        key = split(relpath(joinpath(root, name), joinpath(dir, "c")), '/')
        parse(Int, key[itime]) == tc - 1 && push!(files, joinpath(root, name))
    end
    return files, typesize
end

function decompress(bytes)
    nbytes, _, _ = Blosc.cbuffer_sizes(bytes)
    out = Vector{UInt8}(undef, nbytes)
    n = ccall((:blosc_decompress_ctx, libblosc), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint), bytes, out, nbytes, 1)
    n == nbytes || error("Decompression failed")
    return out
end

chunks = Tuple{String, Vector{UInt8}, Int}[]  # (variable, uncompressed bytes, typesize)
for name in sort(readdir(store))
    dir = joinpath(store, name)
    isfile(joinpath(dir, "zarr.json")) && isdir(joinpath(dir, "c")) || continue
    files, typesize = chunk_files(dir, tc)
    for f in files
        push!(chunks, (name, decompress(read(f)), typesize))
    end
end
total = sum(length(c[2]) for c in chunks)
@printf("%d chunks, %.1f MiB uncompressed, %d tasks\n", length(chunks), total / 2^20, ntasks)

function compress(chunk, cname, clevel, shuffle, dest)
    _, src, typesize = chunk
    n = ccall(
        (:blosc_compress_ctx, libblosc), Cint,
        (Cint, Cint, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cstring, Csize_t, Cint),
        clevel, shuffle, typesize, length(src), src, dest, length(dest), cname, 0, 1,
    )
    n > 0 || error("Compression failed")
    return Int(n)
end

"""Compress every chunk once, in `ntasks` tasks; returns the wall time and the sizes."""
function run_codec(cname, clevel, shuffle)
    sizes = zeros(Int, length(chunks))
    next = Threads.Atomic{Int}(1)
    t = @elapsed @sync for _ in 1:ntasks
        Threads.@spawn begin
            dest = Vector{UInt8}(undef, maximum(length(c[2]) for c in chunks) + 64)
            while (i = Threads.atomic_add!(next, 1)) <= length(chunks)
                sizes[i] = compress(chunks[i], cname, clevel, shuffle, dest)
            end
        end
    end
    return t, sizes
end

codecs = [
    ("lz4", 1), ("lz4", 5), ("blosclz", 1), ("blosclz", 5), ("zstd", 1), ("zstd", 2), ("zstd", 3),
]
names = unique(first.(chunks))
results = []
println(rpad("codec", 22), "ratio   GB/s   MB per step (", total ÷ 2^20, " MiB in)")
for (cname, clevel) in codecs, (shuffle, sname) in ((1, "shuffle"), (2, "bitshuffle"))
    run_codec(cname, clevel, shuffle)  # warm up
    t, sizes = minimum(_ -> run_codec(cname, clevel, shuffle), 1:3)
    ratio = total / sum(sizes)
    push!(results, (cname, clevel, sname, ratio, total / t / 1.0e9, sizes))
    @printf("%-22s %5.2f  %5.1f\n", "$cname $clevel $sname", ratio, total / t / 1.0e9)
end

println("\nCompression ratio per variable")
println(rpad("variable", 24), join([rpad("$(r[1]) $(r[2]) $(r[3][1:3])", 13) for r in results]))
for name in names
    idx = findall(c -> c[1] == name, chunks)
    raw = sum(length(chunks[i][2]) for i in idx)
    print(rpad(name, 24))
    for r in results
        print(rpad(@sprintf("%.2f", raw / sum(r[6][idx])), 13))
    end
    @printf("  (%.0f MiB)\n", raw / 2^20)
end
