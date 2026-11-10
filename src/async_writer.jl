"""
A block of consecutive output timesteps, accumulated in (pinned) host memory before it is
written to `store` on the writer task. `data[i]` holds output `i`, of size
(ncells, block length, nlayers), with nlayers = 1 for outputs without a layer dimension.
Timestep `t` of the block is timestep `first + t - 1` of the store.
"""
mutable struct OutputBlock
    const data::Vector{Array{Float32, 3}}
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

"""Buffers for the uncompressed and the compressed chunks."""
struct ChunkBuffers
    values::BufferPool{Float32}
    bytes::BufferPool{UInt8}
end

ChunkBuffers() = ChunkBuffers(BufferPool{Float32}(), BufferPool{UInt8}())

"""
Writes the model output, one block of `block_length` timesteps at a time. Every timestep,
[`record_output!`](@ref) copies the outputs from the model (on the GPU) into the current
block. Full blocks are written by a background task (`task`), so writing happens while the
model computes the next block. There are `nblocks` blocks: while one is written, the model
fills another.

The writer persists over the years of the run; every year gets its own store, see
[`next_store!`](@ref).
"""
mutable struct OutputWriter
    const outputs::Vector{ParameterMetadata}  # output variables to write
    const free::Channel{OutputBlock}  # blocks ready to be filled
    const jobs::Channel{OutputBlock}  # full blocks, waiting to be written
    const task::Task                  # the writer task
    const nblocks::Int
    const block_length::Int
    const open_store::Function        # year -> new `OutputStore` for that year
    const busy::Base.RefValue{Float64}  # seconds spent writing, by the writer task
    store::Any                        # the store of the current year
    block::Union{OutputBlock, Nothing}  # the block being filled, if any
end

"""
Start the output writer for `outputs`, writing to the store of `year` from `open_store`.
The blocks are allocated on the host, and pinned when the backend supports it. `nlayers[i]`
is the number of layers of output `i` (1 for outputs without a layer dimension).
"""
function OutputWriter(open_store, year, outputs, ncells, nlayers, block_length, backend; nblocks = 2)
    free = Channel{OutputBlock}(nblocks)
    jobs = Channel{OutputBlock}(nblocks)

    bytes = nblocks * block_length * ncells * sum(nlayers) * sizeof(Float32)
    println("  -> Allocating $nblocks output blocks of $block_length timesteps ($(Base.format_bytes(bytes)))")
    for _ in 1:nblocks
        data = [Array{Float32}(undef, ncells, block_length, n) for n in nlayers]
        foreach(a -> pin_memory!(backend, a), data)
        put!(free, OutputBlock(data, nothing, 0, 0))
    end

    # Let Blosc compress in several threads at once (for `write_block!`). The compressor is
    # a global Blosc setting; every output store uses the same one.
    ENV["BLOSC_NOLOCK"] = "1"
    Blosc.set_compressor(OUTPUT_COMPRESSOR.cname)

    store = open_store(year)
    busy = Ref(0.0)
    buffers = ChunkBuffers()
    task = Threads.@spawn for block in jobs
        busy[] += @elapsed write_block!(block.store, block, buffers)
        block.store = nothing
        put!(free, block)
    end
    # Rethrow errors of the writer task in the model, instead of waiting for it forever
    bind(free, task)
    bind(jobs, task)

    return OutputWriter(outputs, free, jobs, task, nblocks, block_length, open_store, busy, store, nothing)
end

"""The time index of the current model time in the output store of its year."""
output_time_index(clock) = (clock.time - DateTime(year(clock.time))) ÷ clock.dt + 1

"""
Copy the outputs of the current timestep from the `model` (on the GPU) into the current
block, and hand the block to the writer task when it is full. Waits for a free block if
the writer is behind.
"""
function record_output!(writer::OutputWriter, model)
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
    for (meta, data) in zip(writer.outputs, block.data)
        copy_timestep!(data, block.len, meta.lens(model))
    end
    block.len == writer.block_length && submit_block!(writer)
    return nothing
end

"""
Copy `src`, of size (ncells, nlayers) or (ncells,), to timestep `t` of the block `data`, of
size (ncells, block length, nlayers). One copy per layer, without intermediate buffers.
"""
function copy_timestep!(data::Array{Float32, 3}, t, src::AbstractArray)
    n, len, nlayers = size(data)
    length(src) == n * nlayers ||
        error("Output of size $(size(src)) does not match the block size $((n, nlayers))")
    for l in 1:nlayers
        copyto!(data, ((l - 1) * len + t - 1) * n + 1, src, (l - 1) * n + 1, n)
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
    submit_block!(writer)
    close(writer.jobs)  # the writer task stops when it has written every block
    wait(writer.task)
    close_output(writer.store)
    println("Output writer: $(round(writer.busy[]; digits = 1)) s writing")
    return nothing
end
