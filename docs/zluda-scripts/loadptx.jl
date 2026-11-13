# Loads a PTX file with ZLUDA's cuModuleLoadData / cuModuleLoadDataEx, without CUDA.jl. For finding
# out why ZLUDA rejects a kernel (section 3.3 of docs/zluda-cuda-run.md).
#   source env.sh; julia --startup-file=no loadptx.jl kernel.ptx
# With ZLUDA's tracing layer (writes the parser's complaints to $ZLUDA_WORK/trace/julia*/module_*.log):
#   mkdir -p $ZLUDA_WORK/libcuda-trace $ZLUDA_WORK/trace
#   ln -sfn $ZLUDA_DIR/trace/libcuda.so.1 $ZLUDA_WORK/libcuda-trace/libcuda.so.1
#   ZLUDA_CUDA_LIB=$ZLUDA_DIR/libcuda.so LD_LIBRARY_PATH=$ZLUDA_WORK/libcuda-trace \
#     ZLUDA_LOG_DIR=$ZLUDA_WORK/trace julia --startup-file=no loadptx.jl kernel.ptx
using Libdl
h = dlopen("libcuda.so.1"); f(s) = dlsym(h, s)
ccall(f(:cuInit), Cint, (Cuint,), 0)
ctx = Ref{Ptr{Cvoid}}(); ccall(f(:cuDevicePrimaryCtxRetain), Cint, (Ptr{Ptr{Cvoid}}, Cint), ctx, 0); ccall(f(:cuCtxSetCurrent), Cint, (Ptr{Cvoid},), ctx[])
data = read(ARGS[1]); data[end] == 0 || push!(data, 0)
m = Ref{Ptr{Cvoid}}()
println("cuModuleLoadData=", ccall(f(:cuModuleLoadData), Cint, (Ptr{Ptr{Cvoid}}, Ptr{UInt8}), m, data))
println("cuModuleLoadDataEx(0 opts)=", ccall(f(:cuModuleLoadDataEx), Cint, (Ptr{Ptr{Cvoid}}, Ptr{UInt8}, Cuint, Ptr{Cint}, Ptr{Ptr{Cvoid}}), m, data, 0, C_NULL, C_NULL))
