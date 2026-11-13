# Queries ZLUDA's libcuda directly, without CUDA.jl: driver version, devices (name and compute
# capability), and the two context calls CUDA.jl needs (cuCtxGetId is not supported by ZLUDA
# v7-preview.21).  Run:  source env.sh; julia --startup-file=no driver_query.jl
using Libdl
h = dlopen("libcuda.so.1"); println("driver library: ", dlpath(h))
f(s) = dlsym(h, s)
v = Ref{Cint}(); println("cuDriverGetVersion: status ", ccall(f(:cuDriverGetVersion), Cint, (Ptr{Cint},), v), ", version ", v[])
println("cuInit: ", ccall(f(:cuInit), Cint, (Cuint,), 0))
n = Ref{Cint}(); ccall(f(:cuDeviceGetCount), Cint, (Ptr{Cint},), n); println("devices: ", n[])
for i in 0:n[]-1
    d = Ref{Cint}(); ccall(f(:cuDeviceGet), Cint, (Ptr{Cint}, Cint), d, i)
    buf = zeros(UInt8, 256); ccall(f(:cuDeviceGetName), Cint, (Ptr{UInt8}, Cint, Cint), buf, 256, d[])
    ma = Ref{Cint}(); mi = Ref{Cint}()
    ccall(f(:cuDeviceGetAttribute), Cint, (Ptr{Cint}, Cint, Cint), ma, 75, d[])   # compute capability major
    ccall(f(:cuDeviceGetAttribute), Cint, (Ptr{Cint}, Cint, Cint), mi, 76, d[])   # ... minor
    println("  ", i, ": ", unsafe_string(pointer(buf)), ", compute capability ", ma[], ".", mi[])
end
ctx = Ref{Ptr{Cvoid}}()
println("cuDevicePrimaryCtxRetain: ", ccall(f(:cuDevicePrimaryCtxRetain), Cint, (Ptr{Ptr{Cvoid}}, Cint), ctx, 0))
println("cuCtxSetCurrent: ", ccall(f(:cuCtxSetCurrent), Cint, (Ptr{Cvoid},), ctx[]))
id = Ref{Culonglong}(0)
println("cuCtxGetId: ", ccall(f(:cuCtxGetId), Cint, (Ptr{Cvoid}, Ptr{Culonglong}), ctx[], id), "   (801 = CUDA_ERROR_NOT_SUPPORTED)")
flags = Ref{Cuint}(0)
println("cuCtxGetFlags: ", ccall(f(:cuCtxGetFlags), Cint, (Ptr{Cuint},), flags), "   (801 = CUDA_ERROR_NOT_SUPPORTED)")
for l in dllist(); occursin(r"amdhip|hsa-runtime|comgr", l) && println("loaded: ", l); end
