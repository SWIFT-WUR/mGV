module AMDGPUExt

using AMDGPU: AMDGPU, ROCBackend, ROCArray
using AMDGPU.HIP: HIP, HIPStream, HIPEvent
import mGV

# `KernelAbstractions.pagelock!` pins with `hipHostRegister`, which is not undone when the array is freed
function mGV.unpin_memory!(::ROCBackend, arr::Array)
    isdefined(AMDGPU.Mem, :is_registered) || return nothing  # older AMDGPU versions
    AMDGPU.Mem.is_registered(Ptr{Cvoid}(pointer(arr))) && AMDGPU.Mem.unpin(pointer(arr))
    return nothing
end

"""
Copies device arrays to the host with `hipMemcpyAsync`, which (unlike the
`hipMemcpyWithStream` of `copyto!`) does not block the host until the copy is done. The
copies run in a stream of their own (`stream`), ordered against the model's stream (the
stream of the calling task) with events: `staged` marks the end of the staging copies in the
model's stream, `done` the end of the host copies.
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

function mGV.stage_copy!(c::HostCopier, dst::ROCArray{T}, src::ROCArray{T}) where {T}
    length(dst) == length(src) || error("Staging buffer of length $(length(dst)) for an array of length $(length(src))")
    model = AMDGPU.stream()
    HIP.hipStreamWaitEvent(model, c.done, 0)  # `dst` is free once the last host copy is done
    GC.@preserve dst src memcpy_async(pointer(dst), pointer(src), sizeof(dst), HIP.hipMemcpyDeviceToDevice, model)
    return nothing
end

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

mGV.finish_host_copy!(c::HostCopier) = (HIP.hipEventRecord(c.done, c.stream); nothing)

# Lets other Julia tasks (e.g. the writer) run while waiting
mGV.wait_host_copy(c::HostCopier) = HIP.synchronize(c.done)

function __init__()
    return mGV.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end

end
