const HAS_CUDA   = try using CUDA;   true catch; false end
const HAS_AMDGPU = try using AMDGPU; true catch; false end
const HAS_METAL  = try using Metal;  true catch; false end
const HAS_OPENCL = try using OpenCL; true catch; false end
const OPENCL_FUNCTIONAL = try OpenCL.zeros(1); true catch; false end

if HAS_CUDA && CUDA.functional()
    const device_backend = CUDABackend()
    const ArrayType = CuArray
    const backend_name = "CUDA"

    const StreamType = CUDA.CuStream
    create_stream() = CUDA.CuStream()
    
    pin_memory!(arr) = CUDA.pin(arr)
    println("✅ Active device: NVIDIA GPU (CUDA)")

elseif HAS_AMDGPU && AMDGPU.functional()
    const device_backend = ROCBackend()
    const ArrayType = ROCArray
    const backend_name = "AMDGPU"
    
    const StreamType = AMDGPU.HIPStream
    create_stream() = AMDGPU.HIPStream()
    
    pin_memory!(arr) = nothing
    println("✅ Active device: AMD GPU (ROCm)")

elseif HAS_OPENCL && OPENCL_FUNCTIONAL
    const device_backend = OpenCLBackend()
    const ArrayType = CLArray
    const backend_name = "OpenCL"

    const StreamType = Nothing
    create_stream() = nothing

    local platforms = OpenCL.cl.platforms()
    # grab first device from first platform
    local device = OpenCL.cl.devices(OpenCL.cl.platforms()[1])[1]
    
    pin_memory!(arr) = nothing
    println("✅ Active device: OpenCL ($(device.name))")

elseif HAS_METAL && Metal.functional()
    const device_backend = MetalBackend()
    const ArrayType = MtlArray
    const backend_name = "Metal"

    # For now, run sequentially without asynchronous streams (CommandQueues) for Metal
    const StreamType = Nothing 
    create_stream() = nothing
    
    pin_memory!(arr) = nothing 
    println("✅ Active device: Apple Silicon (Metal)")

else
    const device_backend = CPU()
    const ArrayType = Array
    const backend_name = "CPU"

    # CPU has no streams
    const StreamType = Nothing
    create_stream() = nothing
    
    pin_memory!(arr) = nothing
    println("⚠️  GPU not found. Active device: CPU")
end
