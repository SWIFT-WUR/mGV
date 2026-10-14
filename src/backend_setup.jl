const HAS_CUDA   = try using CUDA;   true catch; false end
const HAS_AMDGPU = try using AMDGPU; true catch; false end
const HAS_METAL  = try using Metal;  true catch; false end
const HAS_ONEAPI = try using oneAPI; true catch; false end
const HAS_OPENCL = try using OpenCL; true catch; false end
const OPENCL_FUNCTIONAL = try OpenCL.zeros(1); true catch; false end

# oneAPI.jl skips driver initialization while precompiling, so `oneAPI.functional()`
# is always false at that point. As the backend is selected at precompile time, probe
# for a working device in a separate (non-precompiling) process instead.
function oneapi_functional()
    ccall(:jl_generating_output, Cint, ()) == 0 && return oneAPI.functional()
    probe = `$(Base.julia_cmd()) --startup-file=no -e 'using oneAPI; exit(oneAPI.functional() ? 0 : 1)'`
    return success(pipeline(probe; stdout=devnull, stderr=devnull))
end

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

elseif HAS_ONEAPI && oneapi_functional()
    const device_backend = oneAPIBackend()
    const ArrayType = oneArray
    const backend_name = "oneAPI"

    # For now, run sequentially without asynchronous streams (CommandQueues) for oneAPI
    const StreamType = Nothing 
    create_stream() = nothing
    
    pin_memory!(arr) = nothing

    # Work around oneAPI.jl (<= 2.9.2) returning the flat launch index for
    # `@index(Global)`, which is wrong when the workgroups do not exactly tile a
    # multidimensional ndrange (some items are skipped, others go out of bounds).
    # Derive it from the Cartesian index instead, like CUDA.jl does. The `ctx` type
    # restriction makes this more specific than oneAPI's own method, as overwriting
    # methods is not allowed during precompilation.
    Base.Experimental.@overlay oneAPI.method_table @inline function KernelAbstractions.__index_Global_Linear(
        ctx::KernelAbstractions.CompilerMetadata
    )
        I = KernelAbstractions.__index_Global_Cartesian(ctx)
        return @inbounds LinearIndices(KernelAbstractions.__ndrange(ctx))[I]
    end

    println("✅ Active device: Intel Graphics (oneAPI)")

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
