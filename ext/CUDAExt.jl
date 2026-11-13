module CUDAExt

using CUDA: CUDA, CUDABackend
import mGV

mGV.device_free_memory(::CUDABackend) = CUDA.free_memory()

function __init__()
    return mGV.register_backend!("cuda", "CUDA", CUDA.functional, CUDABackend)
end

end
