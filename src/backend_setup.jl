using KernelAbstractions

global float_type = Float32 # User settable float precision

# ===========================================================================
# 1. PRECISION CONFIGURATION
# ===========================================================================
if !isdefined(Main, :float_type)
    println("⚠️ User didn't specify a type in the configuration file. Defaulting to Float32.")
    const FloatType = Float32
else
    const FloatType = float_type
end

ft(x) = FloatType(x)

println("Precision set to: $FloatType")

# ===========================================================================
# 2. PACKAGE LOADING
# ===========================================================================
const HAS_CUDA   = try using CUDA;   true catch; false end
const HAS_AMDGPU = try using AMDGPU; true catch; false end
const HAS_METAL  = try using Metal;  true catch; false end

# ===========================================================================
# 3. DEVICE (GPU type or CPU) & ARRAY CONFIGURATION
# ===========================================================================

if HAS_CUDA && CUDA.functional()
    const device_backend = CUDABackend()
    const ArrayType = CuArray
    const backend_name = "CUDA"

    const StreamType = CUDA.CuStream
    create_stream() = CUDA.CuStream()
    
    pin_memory!(arr) = CUDA.Mem.pin(arr)
    println("✅ Active Device: NVIDIA GPU (CUDA)")

elseif HAS_AMDGPU && AMDGPU.functional()
    const device_backend = ROCBackend()
    const ArrayType = ROCArray
    const backend_name = "AMDGPU"
    
    const StreamType = AMDGPU.ROCStream
    create_stream() = AMDGPU.ROCStream()
    
    pin_memory!(arr) = AMDGPU.Mem.pin(arr)
    println("✅ Active Device: AMD GPU (ROCm)")

elseif HAS_METAL && Metal.functional()
    const device_backend = MetalBackend()
    const ArrayType = MtlArray
    const backend_name = "Metal"

    # For now, run sequentially without asynchronous streams (CommandQueues) for Metal
    const StreamType = Nothing 
    create_stream() = nothing
    
    pin_memory!(arr) = nothing 
    println("✅ Active Device: Apple Silicon (Metal)")

else
    const device_backend = CPU()
    const ArrayType = Array
    const backend_name = "CPU"

    # CPU has no streams
    const StreamType = Nothing
    create_stream() = nothing
    
    pin_memory!(arr) = nothing
    println("⚠️  GPU not found. Active Device: CPU")
end

# ===========================================================================
# 4. MEMORY ALLOCATION HELPER
# ===========================================================================

# 1. Generic version: User specifies the type (e.g., alloc(Int32, 5))
alloc(T::DataType, dims...) = KernelAbstractions.zeros(device_backend, T, dims...)

# 2. Physics version: Defaults to FloatType (e.g., alloc(nx, ny))
alloc(dims...) = alloc(FloatType, dims...)

println("✅ Memory allocator configured for: $backend_name")