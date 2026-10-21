module AMDGPUExt

using AMDGPU: AMDGPU, ROCBackend
import mGV

function __init__()
    return mGV.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end

end
