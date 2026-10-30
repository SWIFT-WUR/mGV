module CUDAExt

using CUDA: CUDA, CUDABackend
import mGV

mGV.pin_memory!(::CUDABackend, arr) = CUDA.pin(arr)

function __init__()
    return mGV.register_backend!("cuda", "CUDA", CUDA.functional, CUDABackend)
end

end
