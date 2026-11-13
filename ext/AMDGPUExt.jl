module AMDGPUExt

using AMDGPU: AMDGPU, ROCBackend, ROCArray
using AMDGPU.HIP: HIP, HIPStream, HIPEvent
import mGV

# Of the current device, which `KernelAbstractions.allocate` allocates on
mGV.device_free_memory(::ROCBackend) = first(AMDGPU.info())

# `KernelAbstractions.pagelock!` pins with `hipHostRegister`, which is not undone when the array is freed
function mGV.unpin_memory!(::ROCBackend, arr::Array)
    isdefined(AMDGPU.Mem, :is_registered) || return nothing  # older AMDGPU versions
    AMDGPU.Mem.is_registered(Ptr{Cvoid}(pointer(arr))) && AMDGPU.Mem.unpin(pointer(arr))
    return nothing
end

"""
Copies output from the device to the host with `hipMemcpyAsync`, which (unlike the
`hipMemcpyWithStream` of `copyto!`) does not block the host until the copy is done.

Copies to the host run in a stream of their own (`stream`), ordered against the model's
stream (the stream of the calling task) with events: `staged` marks the end of the staging
(by kernels in the model's stream), `done` the end of the host copies.
"""
struct HostCopier
    stream::HIPStream
    staged::HIPEvent
    done::HIPEvent
end

function HostCopier()
    stream = HIPStream()
    # Not recorded yet: waiting for an event that was never recorded returns immediately
    staged = HIPEvent(stream; do_record = false)
    done = HIPEvent(stream; do_record = false)
    return HostCopier(stream, staged, done)
end

mGV.host_copier(::ROCBackend) = HostCopier()

function memcpy_async(dst, src, bytes, kind, stream)
    HIP.hipMemcpyAsync(Ptr{Cvoid}(dst), Ptr{Cvoid}(src), bytes, kind, stream)
    return nothing
end

# The staging buffers are free once the last host copies are done
mGV.begin_staging!(c::HostCopier) = (HIP.hipStreamWaitEvent(AMDGPU.stream(), c.done, 0); nothing)

function mGV.start_host_copy!(c::HostCopier)
    HIP.hipEventRecord(c.staged, AMDGPU.stream())
    HIP.hipStreamWaitEvent(c.stream, c.staged, 0)
    return nothing
end

function mGV.copy_to_host!(c::HostCopier, dst::Array{T}, doffs, src::ROCArray{T}, soffs, n) where {T}
    checkbounds(dst, doffs + n - 1)
    checkbounds(src, soffs + n - 1)
    GC.@preserve dst src memcpy_async(
        pointer(dst, doffs), pointer(src, soffs), n * sizeof(T), HIP.hipMemcpyDeviceToHost, c.stream,
    )
    return nothing
end

function mGV.finish_host_copy!(c::HostCopier, event::HIPEvent)
    HIP.hipEventRecord(c.done, c.stream)
    HIP.hipEventRecord(event, c.stream)
    return nothing
end

mGV.block_event(c::HostCopier) = HIPEvent(c.stream; do_record = false)

# Lets other Julia tasks (e.g. the writer) run while waiting
mGV.wait_host_copy(::HostCopier, event::HIPEvent) = HIP.synchronize(event)

function __init__()
    return mGV.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end

end
