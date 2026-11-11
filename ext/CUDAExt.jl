module CUDAExt

using CUDA: CUDA, CUDABackend
import mGV

function __init__()
    return mGV.register_backend!("cuda", "CUDA", CUDA.functional, CUDABackend)
end

end
