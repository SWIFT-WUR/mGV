module CUDAExt

using CUDA: CUDA, CUDABackend, CuArray, CuContext, CuStream, CuEvent
using KernelAbstractions: KernelAbstractions
import mGV

mGV.device_free_memory(::CUDABackend) = CUDA.free_memory()

# No `unpin_memory!`: `KernelAbstractions.pagelock!` pins with `CUDA.pin`, which unpins the
# memory itself, after the array is garbage collected

"""
Copies output from the device to the host without blocking the host, with the asynchronous
`KernelAbstractions.copyto!` (`cuMemcpyDtoHAsync`, to the pinned output blocks).

Copies to the host run in a stream of their own (`stream`), ordered against the model's
stream (the stream of the calling task) with events: `staged` marks the end of the staging
(by kernels in the model's stream), `done` the end of the host copies. CUDA.jl also orders
the use of the staging buffers on the two streams itself. It learns from these events that
the streams already wait for each other, and otherwise makes the stream wait on the device,
so that does not block the host either.
"""
struct HostCopier
    context::CuContext  # of the device of the model
    stream::CuStream
    staged::CuEvent
    done::CuEvent
end

new_event() = CuEvent(CUDA.EVENT_DISABLE_TIMING)

function HostCopier()
    stream = CuStream()
    done = new_event()
    # On the still empty stream, so that the first `begin_staging!` has nothing to wait for
    CUDA.record(done, stream)
    return HostCopier(CUDA.context(), stream, new_event(), done)
end

mGV.host_copier(::CUDABackend) = HostCopier()

# The staging buffers are free once the last host copies are done
mGV.begin_staging!(c::HostCopier) = (CUDA.wait(c.done, CUDA.stream()); nothing)

function mGV.start_host_copy!(c::HostCopier)
    CUDA.record(c.staged, CUDA.stream())
    CUDA.wait(c.staged, c.stream)
    return nothing
end

function mGV.copy_to_host!(c::HostCopier, dst::Array{T}, doffs, src::CuArray{T}, soffs, n) where {T}
    dst_view = view(dst, doffs:(doffs + n - 1))
    src_view = view(src, soffs:(soffs + n - 1))
    # `copyto!` copies in the stream of the task, without waiting for the copy: make that
    # the copier's stream for this copy (which does not synchronize either stream)
    CUDA.stream!(c.stream) do
        KernelAbstractions.copyto!(CUDABackend(), dst_view, src_view)
    end
    return nothing
end

function mGV.finish_host_copy!(c::HostCopier, event::CuEvent)
    CUDA.record(c.done, c.stream)
    CUDA.record(event, c.stream)
    return nothing
end

mGV.block_event(::HostCopier) = new_event()

# Lets other Julia tasks (e.g. the model) run while waiting. The writer task calls this: a
# new task uses the default device, which may not be the model's.
mGV.wait_host_copy(c::HostCopier, event::CuEvent) = CUDA.context!(() -> CUDA.synchronize(event), c.context)

function __init__()
    return mGV.register_backend!("cuda", "CUDA", CUDA.functional, CUDABackend)
end

end
