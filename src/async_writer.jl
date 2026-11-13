"""The Int16 that stands for NaN in packed output, see [`Packing`](@ref)."""
const PACKED_NAN = typemin(Int16)

"""The element type in which an output with `packing` is stored."""
stored_type(::Nothing) = Float32
stored_type(::Packing) = Int16

"""The fill value of stored outputs of element type `T`: where there is no data."""
fill_value(::Type{Float32}) = NaN32
fill_value(::Type{Int16}) = PACKED_NAN

"""
Pack `x` for storage with `packing` (see [`Packing`](@ref)). Returns the stored value, and
whether `x` was clipped to the range of the packing. Without a packing, `x` is stored as is.

Gives the same values as the Zarr `scale_offset` and `cast_value` codecs (in Float32, rounded
to nearest even), except that values below the range become -32767 rather than
`typemin(Int16)`, which stands for NaN.
"""
@inline pack_value(x::Float32, ::Nothing) = (x, false)
@inline function pack_value(x::Float32, p::Packing)
    y = round((x - p.offset) * p.scale)
    nan = isnan(y)
    clipped = !nan & !(abs(y) <= 32767.0f0)  # also ±Inf
    v = unsafe_trunc(Int16, ifelse(nan, 0.0f0, clamp(y, -32767.0f0, 32767.0f0)))
    return ifelse(isnan(x), PACKED_NAN, v), clipped
end

"""
Pack the outputs `src`, of size (n, nlayers) or (n,), into `dst`: layer `l` goes to
`dst[doffs + (l - 1) * dstride .+ (1:n)]`. Sets `flags[fidx]` to 1 when a value is clipped.
"""
@kernel function pack_output_kernel!(dst, doffs, dstride, @Const(src), n, packing, flags, fidx)
    i, l = @index(Global, NTuple)
    @inbounds v, clipped = pack_value(src[(l - 1) * n + i], packing)
    @inbounds dst[doffs + (l - 1) * dstride + i] = v
    if clipped
        @inbounds flags[fidx] = Int32(1)
    end
end

function pack_output!(backend, dst, doffs, dstride, src, packing, flags, fidx)
    n = size(src, 1)
    nlayers = length(src) ÷ n
    pack_output_kernel!(backend)(dst, doffs, dstride, src, n, packing, flags, fidx; ndrange = (n, nlayers))
    return nothing
end

"""
A block of consecutive output timesteps, accumulated in (pinned) host memory before it is
written to `store` on the writer task. `data[i]` holds output `i`, of size
(ncells, block length, nlayers), with nlayers = 1 for outputs without a layer dimension, in
its stored element type (see [`stored_type`](@ref)). Timestep `t` of the block is timestep
`first + t - 1` of the store. `clipped[i, t]` is nonzero when values of output `i` were
clipped to the range of its packing in timestep `t`. With a [`host_copier`](@ref), `copied`
is the event that marks the end of the copies to the block; the writer task waits for it.
"""
mutable struct OutputBlock
    const data::Vector{Array}
    const clipped::Matrix{Int32}
    const copied::Any  # an event of the copier, or `nothing`
    store::Any   # the `OutputStore` the block is written to
    first::Int   # time index in the store of the first timestep of the block
    len::Int     # number of timesteps in the block
end

"""Reusable buffers, shared by the tasks that write chunks in parallel."""
struct BufferPool{T}
    lock::ReentrantLock
    items::Vector{Vector{T}}
end

BufferPool{T}() where {T} = BufferPool{T}(ReentrantLock(), Vector{T}[])

acquire!(pool::BufferPool{T}) where {T} =
    @lock pool.lock (isempty(pool.items) ? T[] : pop!(pool.items))
release!(pool::BufferPool{T}, buf::Vector{T}) where {T} = (@lock pool.lock push!(pool.items, buf); nothing)

"""Buffers for the uncompressed (of each stored type) and the compressed chunks."""
struct ChunkBuffers
    float32::BufferPool{Float32}
    int16::BufferPool{Int16}
    bytes::BufferPool{UInt8}
end

ChunkBuffers() = ChunkBuffers(BufferPool{Float32}(), BufferPool{Int16}(), BufferPool{UInt8}())

value_pool(buffers::ChunkBuffers, ::Type{Float32}) = buffers.float32
value_pool(buffers::ChunkBuffers, ::Type{Int16}) = buffers.int16

"""
Writes the model output, one block of `block_length` timesteps at a time. Every timestep,
[`record_output!`](@ref) copies the outputs from the model (on the GPU) into the current
block. Full blocks are written by a background task (`task`), so writing happens while the
model computes the next block. There are `nblocks` blocks: while one is written, the model
fills another.

Outputs with a packing (`packings`) are packed to Int16 by a kernel on the model's device;
see [`Packing`](@ref). On a GPU, that kernel writes the outputs of a timestep to `staging`
buffers on the device (also those that stay Float32), which are copied to the (pinned) host
block while the model computes the next timestep. If the backend has a
[`host_copier`](@ref) (`copier`), it queues these copies on a stream of its own without
blocking the host; otherwise they run on a separate task (`copy`), which has its own stream.
That task runs in the interactive threadpool. Julia also runs the main task there, so with
one interactive thread the copy only starts when the model task yields: start Julia with
two interactive threads (`-t N,2`) for backends without a copier.

With a copier, a full block is handed to the writer task right away; the writer task waits
until the copies to it are done (the block's `copied` event). The model task only waits when
there is no free block.

The writer persists over the years of the run; every year gets its own store, see
[`next_store!`](@ref).
"""
mutable struct OutputWriter
    const outputs::Vector{ParameterMetadata}  # output variables to write
    const packings::Vector{Union{Packing, Nothing}}  # how each output is stored
    const free::Channel{OutputBlock}  # blocks ready to be filled
    const jobs::Channel{OutputBlock}  # full blocks, waiting to be written
    const task::Task                  # the writer task
    const nblocks::Int
    const block_length::Int
    const open_store::Function        # year -> new `OutputStore` for that year
    const busy::Base.RefValue{Float64}  # seconds spent writing, by the writer task
    const backend::Backend
    store::Any                        # the store of the current year
    block::Union{OutputBlock, Nothing}  # the block being filled, if any
    staging::Vector{Any}              # device copies of the outputs of a timestep (GPU only)
    flags::Any                        # device copy of the clipping flags of a timestep (GPU only)
    const copier::Any                 # the backend's `host_copier`, or `nothing`
    copy::Union{Task, Nothing}        # the task copying `staging` to the block, if any (without `copier`)
end

"""
Start the output writer for `outputs`, writing to the store of `year` from `open_store`.
The blocks are allocated on the host, and pinned when the backend supports it. `nlayers[i]`
is the number of layers of output `i` (1 for outputs without a layer dimension), and
`packings[i]` how it is stored.
"""
function OutputWriter(open_store, year, outputs, packings, ncells, nlayers, block_length, backend; nblocks = 3)
    free = Channel{OutputBlock}(nblocks)
    jobs = Channel{OutputBlock}(nblocks)

    copier = backend isa CPU ? nothing : host_copier(backend)
    types = stored_type.(packings)
    bytes = nblocks * block_length * ncells * sum(nlayers .* sizeof.(types))
    println("  -> Allocating $nblocks output blocks of $block_length timesteps ($(Base.format_bytes(bytes)))")
    for _ in 1:nblocks
        data = Array[Array{T}(undef, ncells, block_length, n) for (T, n) in zip(types, nlayers)]
        clipped = zeros(Int32, length(outputs), block_length)
        foreach(a -> pin_memory!(backend, a), data)
        pin_memory!(backend, clipped)
        put!(free, OutputBlock(data, clipped, isnothing(copier) ? nothing : block_event(copier), nothing, 0, 0))
    end

    store = open_store(year)
    busy = Ref(0.0)
    buffers = ChunkBuffers()
    task = Threads.@spawn begin
        warned = Set{Int}()  # outputs with a warning about clipping, for the current store
        current = nothing
        for block in jobs
            # Here, rather than in the model task, so that the model can queue more work
            isnothing(block.copied) || wait_host_copy(copier, block.copied)
            block.store === current || (current = block.store; empty!(warned))
            warn_clipped!(warned, block, outputs, packings)
            busy[] += @elapsed write_block!(block.store, block, buffers)
            block.store = nothing
            put!(free, block)
        end
    end
    # Rethrow errors of the writer task in the model, instead of waiting for it forever
    bind(free, task)
    bind(jobs, task)

    return OutputWriter(
        outputs, packings, free, jobs, task, nblocks, block_length, open_store, busy, backend, store, nothing,
        Any[], nothing, copier, nothing,
    )
end

"""Warn (once per output in `warned`) about outputs with values clipped in `block`."""
function warn_clipped!(warned, block::OutputBlock, outputs, packings)
    for (i, (meta, p)) in enumerate(zip(outputs, packings))
        i in warned && continue
        t = findfirst(!=(0), view(block.clipped, i, 1:block.len))
        isnothing(t) && continue
        lo, hi = p.offset - 32767 / p.scale, p.offset + 32767 / p.scale
        @warn "Output $(meta.output_name) has values outside the range of its Int16 packing ($lo to $hi), " *
            "which are clipped. First at time index $(block.first + t - 1); not reported again for this file."
        push!(warned, i)
    end
    return nothing
end

"""The time index of the current model time in the output store of its year."""
output_time_index(clock) = (clock.time - DateTime(year(clock.time))) ÷ clock.dt + 1

"""
Copy the outputs of the current timestep from the `model` (on the GPU) into the current
block, and hand the block to the writer task when it is full. Waits for a free block if
the writer is behind.

On a GPU the copy to the host is asynchronous: the outputs are packed into device buffers,
which are copied to the block while the model continues (by the backend's `host_copier`, or
else a separate task). The block is handed to the writer once its copies are done, see
[`finish_copy!`](@ref).
"""
function record_output!(writer::OutputWriter, model)
    finish_copy!(writer)
    t = output_time_index(model.clock)
    block = writer.block
    if isnothing(block)
        block = @timeit to "wait for output writer" take!(writer.free)
        block.store = writer.store
        block.first = t
        block.len = 0
        writer.block = block
    end
    t == block.first + block.len || error("Output timestep $t does not follow timestep $(block.first + block.len - 1)")
    block.len += 1
    t = block.len
    sources = [meta.lens(model) for meta in writer.outputs]
    backend = writer.backend
    nout = length(sources)
    if backend isa CPU
        # Straight into the block
        block.clipped[:, t] .= 0
        for (i, (data, src, p)) in enumerate(zip(block.data, sources, writer.packings))
            n, len, _ = size(data)
            pack_output!(backend, data, (t - 1) * n, n * len, src, p, block.clipped, (t - 1) * nout + i)
        end
        KernelAbstractions.synchronize(backend)
        finish_copy!(writer)
        return nothing
    end

    if isempty(writer.staging)
        append!(writer.staging, [KernelAbstractions.allocate(backend, eltype(data), length(src)) for (data, src) in zip(block.data, sources)])
        writer.flags = KernelAbstractions.zeros(backend, Int32, nout)
    end
    copier = writer.copier
    # In the model's stream: after the kernels of this timestep, and before those of the next.
    # With a copier, once the host copies of the last timestep are done (on the device).
    isnothing(copier) || begin_staging!(copier)
    fill!(writer.flags, 0)
    for (i, (staged, src, p)) in enumerate(zip(writer.staging, sources, writer.packings))
        pack_output!(backend, staged, 0, size(src, 1), src, p, writer.flags, i)
    end
    isnothing(copier) || return copy_with_copier!(writer, block)

    KernelAbstractions.synchronize(backend)
    staging, flags, len = writer.staging, writer.flags, block.len
    device = KernelAbstractions.device(backend)
    # In the interactive threadpool, so that the copy is not queued behind the many
    # (compression) tasks of the writer. A new task gets its own stream, on the default device.
    writer.copy = Threads.@spawn :interactive begin
        KernelAbstractions.device!(backend, device)
        foreach((data, src) -> copy_timestep!(data, len, src), block.data, staging)
        copyto!(block.clipped, (len - 1) * nout + 1, flags, 1, nout)
    end
    return nothing
end

"""
Queue the copies of the staged outputs (and clipping flags) of a timestep to timestep
`block.len` of `block` with the writer's `copier`, without blocking the host: in the copier's
stream, after the staging in the model's stream.
"""
function copy_with_copier!(writer::OutputWriter, block)
    copier = writer.copier
    start_host_copy!(copier)
    copy = (dst, doffs, src, soffs, n) -> copy_to_host!(copier, dst, doffs, src, soffs, n)
    foreach((data, src) -> copy_timestep!(data, block.len, src; copy), block.data, writer.staging)
    nout = length(writer.staging)
    copy_to_host!(copier, block.clipped, (block.len - 1) * nout + 1, writer.flags, 1, nout)
    finish_host_copy!(copier, block.copied)
    return nothing
end

"""
Hand the current block to the writer task if it is full. Without a `copier`, first wait
until the outputs of the last timestep are copied to it. With a `copier`, the writer task
waits for the copies (see [`OutputBlock`](@ref)), so the model never does.
"""
function finish_copy!(writer::OutputWriter)
    if !isnothing(writer.copy)
        task = writer.copy
        writer.copy = nothing
        @timeit to "wait for output copy" wait(task)
    end
    block = writer.block
    !isnothing(block) && block.len == writer.block_length && submit_block!(writer)
    return nothing
end

"""
Copy `src`, of size (ncells, nlayers) or (ncells,), to timestep `t` of the block `data`, of
size (ncells, block length, nlayers). One copy per layer, without intermediate buffers.
"""
function copy_timestep!(data::Array{T, 3}, t, src::AbstractArray{T}; copy = copyto!) where {T}
    n, len, nlayers = size(data)
    length(src) == n * nlayers ||
        error("Output of size $(size(src)) does not match the block size $((n, nlayers))")
    for l in 1:nlayers
        copy(data, ((l - 1) * len + t - 1) * n + 1, src, (l - 1) * n + 1, n)
    end
    return nothing
end

"""Hand the current block (if any) to the writer task."""
function submit_block!(writer::OutputWriter)
    isnothing(writer.block) && return nothing
    put!(writer.jobs, writer.block)
    writer.block = nothing
    return nothing
end

"""Wait until the writer task has written every submitted block."""
function wait_for_writer(writer::OutputWriter)
    @timeit to "wait for output writer" begin
        blocks = [take!(writer.free) for _ in 1:writer.nblocks]
        foreach(block -> put!(writer.free, block), blocks)
    end
    return nothing
end

"""
Submit the current block, and continue with the store of `year`. Zarr stores need no closing,
so the writer can still be writing the old one. NetCDF is not thread-safe: the writer has to
be done with the old file before it is closed and the new one is created.
"""
function next_store!(writer::OutputWriter, year)
    finish_copy!(writer)
    submit_block!(writer)
    if writer.store isa NetCDFStore
        wait_for_writer(writer)
        close_output(writer.store)
    end
    writer.store = writer.open_store(year)
    return nothing
end

"""Write all output, close the store and stop the writer task."""
function Base.close(writer::OutputWriter)
    finish_copy!(writer)
    submit_block!(writer)
    close(writer.jobs)  # the writer task stops when it has written every block
    wait(writer.task)
    # Every block is back in `free` once the writer task is done
    while isready(writer.free)
        block = take!(writer.free)
        foreach(a -> unpin_memory!(writer.backend, a), block.data)
        unpin_memory!(writer.backend, block.clipped)
    end
    close_output(writer.store)
    println("Output writer: $(round(writer.busy[]; digits = 1)) s writing")
    return nothing
end
