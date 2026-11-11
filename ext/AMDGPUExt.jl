module AMDGPUExt

using AMDGPU: AMDGPU, ROCBackend
import mGV

# `KernelAbstractions.pagelock!` pins with `hipHostRegister`, which is not undone when the array is freed
function mGV.unpin_memory!(::ROCBackend, arr::Array)
    isdefined(AMDGPU.Mem, :is_registered) || return nothing  # older AMDGPU versions
    AMDGPU.Mem.is_registered(Ptr{Cvoid}(pointer(arr))) && AMDGPU.Mem.unpin(pointer(arr))
    return nothing
end

function __init__()
    return mGV.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end

end
