struct AsyncBufferService
    free_pool::Channel{TransferBuffer}
    job_queue::Channel{Tuple{Int, TransferBuffer}}
    writer_task::Task
end

function start_async_service(nx, ny, nlayers, output_store, n_buffers=4)
    # Create Channels
    free_pool = Channel{TransferBuffer}(n_buffers)
    job_queue = Channel{Tuple{Int, TransferBuffer}}(n_buffers)

    println("  -> 💾 Allocating $n_buffers buffers for Async Pool...")
    for _ in 1:n_buffers
        buf = create_transfer_buffer(nx, ny, nlayers)
        put!(free_pool, buf)
    end

    # START WRITER TASK
    # We use @async (runs on local thread but yields) or @spawn. 
    # For safety with Zarr/HDF5, let's try pinning it to a thread via sticky=false
    task = Threads.@spawn begin
        println("  -> 💾 Async Writer: STARTED on Thread $(Threads.threadid())")
        try
            for (day, buf) in job_queue
                # 1. Write
                write_slice!(day, buf, output_store)
                
                # 2. Recycle
                put!(free_pool, buf)
            end
        catch e
            # === CRITICAL ERROR HANDLING ===
            # Print to Stderr so it bypasses any buffer
            println(stderr, "\n\n🔴🔴🔴 ASYNC WRITER CRASHED ON DAY 🔴🔴🔴")
            showerror(stderr, e, catch_backtrace())
            println(stderr, "\n")
            
            # Close the pool. This forces the Main Thread to crash 
            # with an "InvalidStateException" instead of hanging forever.
            close(free_pool) 
            rethrow(e)
        end
        println("  -> 💾 Async Writer: FINISHED")
    end

    return AsyncBufferService(free_pool, job_queue, task)
end

function get_free_buffer(service::AsyncBufferService)
    # This will now throw an error if the writer crashed and closed the pool
    return take!(service.free_pool)
end

function submit_buffer(service::AsyncBufferService, day, buf)
    put!(service.job_queue, (day, buf))
end

function stop_async_service(service::AsyncBufferService)
    close(service.job_queue)
    wait(service.writer_task)
end
