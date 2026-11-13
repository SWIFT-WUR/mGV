# Baseline checks of CUDA.jl on ZLUDA, without mGV (section 4.1 of docs/zluda-cuda-run.md).
# Run:  source env.sh; julia --project=$ZLUDA_JULIA_ENV step1.jl
# Each stage prints [OK] or [FAIL] with the error and the top of the stack trace.
using CUDA, KernelAbstractions, Libdl
const KA = KernelAbstractions
stage(f, name) = try
    flush(stdout)
    r = f(); flush(stdout); println("[OK]   ", name, r === nothing ? "" : " => $r")
catch e
    println("[FAIL] ", name, ": ", sprint(showerror, e)[1:min(end, 600)])
    for (i, fr) in enumerate(stacktrace(catch_backtrace())[1:min(end, 5)]); println("         ", fr); end
end
stage("functional") do; CUDA.functional(true); end
stage("versioninfo") do; CUDA.versioninfo(); nothing; end
stage("device") do; d = CUDA.device(); "$(CUDA.name(d)), cap $(CUDA.capability(d)), $(round(CUDA.totalmem(d)/2^30, digits=1)) GiB, driver $(CUDA.driver_version()), runtime $(CUDA.runtime_version())"; end
stage("loaded libs") do; join(filter(l -> occursin(r"libcuda\.|nvcuda|amdhip|libcudart|nvml"i, l), Libdl.dllist()), "\n          "); end
a = nothing
stage("CuArray alloc + roundtrip") do; global a = CuArray(Float32.(1:1000)); Array(a) == Float32.(1:1000) || error("roundtrip mismatch"); end
stage("CUDA.zeros/fill") do; b = CUDA.fill(3f0, 100); all(Array(b) .== 3f0) || error("fill mismatch"); end
stage("broadcast") do; b = a .* 2f0 .+ 1f0; Array(b) == Float32.(1:1000) .* 2f0 .+ 1f0 || error("broadcast mismatch"); end
stage("broadcast math (sin/exp/sqrt, Float32)") do; b = Array(sin.(a) .+ exp.(a ./ 1000f0) .+ sqrt.(a)); r = sin.(Float32.(1:1000)) .+ exp.(Float32.(1:1000) ./ 1000f0) .+ sqrt.(Float32.(1:1000)); "max abs diff vs CPU = $(maximum(abs.(b .- r)))"; end
stage("broadcast Float64 pow/log") do; x = CuArray(collect(1.0:1000.0)); b = Array(x .^ 0.37 .+ log.(x)); r = collect(1.0:1000.0) .^ 0.37 .+ log.(1.0:1000.0); "max rel diff vs CPU = $(maximum(abs.(b .- r) ./ abs.(r)))"; end
stage("sum (mapreduce)") do; s = sum(a); s == sum(Float32.(1:1000)) || error("sum mismatch $s"); s; end
function k!(a, s); i = (blockIdx().x - 1) * blockDim().x + threadIdx().x; if i <= length(a); @inbounds a[i] = s * i; end; return; end
stage("@cuda kernel") do; c = CUDA.zeros(Float32, 1000); @cuda threads=256 blocks=4 k!(c, 2f0); CUDA.synchronize(); Array(c) == 2f0 .* (1:1000) || error("kernel result mismatch"); end
@kernel function kak!(c, @Const(a), s); i = @index(Global); @inbounds c[i] = a[i] * s + i; end
stage("KA @kernel on CUDABackend") do; be = CUDABackend(); c = KA.zeros(be, Float32, 1000); kak!(be)(c, a, 3f0; ndrange = length(c)); KA.synchronize(be); Array(c) == Float32.(1:1000) .* 3f0 .+ (1:1000) || error("KA kernel mismatch"); end
@kernel function kak2!(c, @Const(a)); i, j = @index(Global, NTuple); @inbounds c[i, j] = a[i, j] + i * 1000 + j; end
stage("KA 2D kernel, workgroupsize") do; be = CUDABackend(); x = rand(Float32, 37, 53); c = KA.zeros(be, Float32, 37, 53); kak2!(be, 64)(c, CuArray(x); ndrange = size(c)); KA.synchronize(be); Array(c) == [x[i, j] + i * 1000 + j for i in 1:37, j in 1:53] || error("mismatch"); end
stage("CUDA.synchronize / KA.synchronize") do; CUDA.synchronize(); KA.synchronize(CUDABackend()); CUDA.device_synchronize(); nothing; end
stage("CUDA.@allowscalar / findall / any") do; CUDA.@allowscalar(a[5]) == 5f0 || error("scalar"); any(a .> 999f0) || error("any"); nothing; end
flush(stdout)
