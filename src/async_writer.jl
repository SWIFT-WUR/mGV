# async_writer.jl
using Base.Threads

struct AsyncBufferService
    free_pool::Channel{TransferBuffer}             # Buffers ready for the GPU to fill
    job_queue::Channel{Tuple{Int, TransferBuffer}} # Buffers full of data, waiting for Disk
    writer_task::Task                              # The background thread handle
end

"""
Starts the Async Service.
- Allocates `n_buffers` (default 4) to absorb disk latency spikes.
- Starts the background thread that handles the actual writing.
"""
function start_async_service(nx, ny, nlayers, output_store, n_buffers=4)
    # 1. Create Channels with fixed capacity
    free_pool = Channel{TransferBuffer}(n_buffers)
    job_queue = Channel{Tuple{Int, TransferBuffer}}(n_buffers)

    # 2. Allocate Pinned Memory Buffers
    # We use your existing create_transfer_buffer (which pins memory)
    println("  -> 💾 Allocating $n_buffers pinned buffers for Async Pool...")
    for _ in 1:n_buffers
        buf = create_transfer_buffer(nx, ny, nlayers)
        put!(free_pool, buf)
    end

    # 3. Start the Background Writer Thread
    # This runs on a separate thread (e.g., Core 2) and handles all disk I/O
    task = Threads.@spawn begin
        println("  -> 💾 Async Writer: STARTED on Thread $(Threads.threadid())")
        try
            # Loop forever: Wait for a job, write it, recycle the buffer.
            for (day, buf) in job_queue
                
                # A. Write to Disk (Sequential & Safe)
                # This calls the function in io_writer.jl
                write_slice!(day, buf, output_store)
                
                # B. Recycle Buffer
                # Send the buffer back to the free_pool so the GPU can use it again.
                put!(free_pool, buf)
            end
        catch e
            println("⚠️ ASYNC WRITER CRASHED: $e")
            rethrow(e)
        end
        println("  -> 💾 Async Writer: FINISHED")
    end

    return AsyncBufferService(free_pool, job_queue, task)
end

"""
Get a free buffer from the pool.
- Returns immediately if a buffer is ready.
- Blocks (waits) if the disk is too slow and all 4 buffers are full.
"""
function get_free_buffer(service::AsyncBufferService)
    return take!(service.free_pool)
end

"""
Hand off a full buffer to the background thread.
- Returns immediately (microseconds).
"""
function submit_buffer(service::AsyncBufferService, day, buf)
    put!(service.job_queue, (day, buf))
end

"""
Stops the service gracefully.
- Waits for the background thread to finish writing all pending days.
"""
function stop_async_service(service::AsyncBufferService)
    close(service.job_queue) # Tells the loop to stop when empty
    wait(service.writer_task) # Wait for the thread to finish
end