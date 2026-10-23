module OpenCLExt

using OpenCL: OpenCL, OpenCLBackend
import mGV

# OpenCL.jl has no `functional()`; check whether a device can be used instead.
function opencl_functional()
    try
        OpenCL.zeros(Float32, 1)
        return true
    catch
        return false
    end
end

function __init__()
    return mGV.register_backend!("opencl", "OpenCL", opencl_functional, OpenCLBackend)
end

end
