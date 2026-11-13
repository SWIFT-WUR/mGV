# Runs the Mekong test domain on a GPU backend and on the CPU (land-only Zarr output, unpacked
# Float32) and prints the differences per output variable (section 4.4 of docs/zluda-cuda-run.md).
#   source env.sh; julia -t 4,2 --project=$ZLUDA_JULIA_ENV mgv_compare.jl 10      # CUDA on ZLUDA
#   MGV_GPU=rocm ... (see rocm_baseline.sh)                                        # native AMDGPU
# MGV_SAVE=<file> stores the GPU output (Serialization) for cmp_jls.jl.
const GPU = get(ENV, "MGV_GPU", "cuda")
if GPU == "cuda"
    using CUDA
else
    using AMDGPU
end
using KernelAbstractions, Statistics
include(joinpath(@__DIR__, "mgv_common.jl"))
ndays = parse(Int, get(ARGS, 1, "10"))
gpu = mGV.select_backend(GPU)
println("GPU backend: ", mGV.backend_name(gpu), " on ", GPU == "cuda" ? CUDA.name(CUDA.device()) : string(AMDGPU.device())); flush(stdout)
g, tg = run_landonly(gpu, mktempdir(ENV["MGV_OUT"]; cleanup = false); ndays)
c, tc = run_landonly(mGV.select_backend("cpu"), mktempdir(ENV["MGV_OUT"]; cleanup = false); ndays)
println("\n$ndays timesteps: $GPU $(round(tg, digits=2)) s (incl. kernel compilation), CPU $(round(tc, digits=2)) s")
println("variables: GPU $(length(g)), CPU $(length(c)), same names: ", keys(g) == keys(c))
println(rpad("variable", 24), rpad("size", 15), rpad("NaN mismatch", 13), rpad("identical", 10), rpad("max abs diff", 14), rpad("/ max|cpu|", 15), rpad("max |cpu|", 13), rpad("frac differing", 15), "frac with rel diff > 1e-4 (|cpu| > 1e-3 max)")
nident = 0
for name in sort(collect(keys(c)))
    a, b = c[name], g[name]
    nanmis = count(isnan.(a) .!= isnan.(b))
    ok = .!isnan.(a) .& .!isnan.(b)
    ident = isequal(a, b); global nident += ident
    if any(ok)
        x = Float64.(a[ok]); y = Float64.(b[ok]); d = abs.(x .- y); mx = maximum(abs, x)
        sig = abs.(x) .> 1e-3 * mx
        big = count((d[sig] ./ abs.(x[sig])) .> 1e-4) / max(count(sig), 1)
        println(rpad(name, 24), rpad(string(size(a)), 15), rpad(nanmis, 13), rpad(ident, 10), rpad(Float32(maximum(d)), 14), rpad(Float32(maximum(d) / max(mx, 1e-30)), 15), rpad(Float32(mx), 13), rpad(round(count(>(0), d) / length(d), digits = 4), 15), round(big, digits = 5))
    else
        println(rpad(name, 24), rpad(string(size(a)), 15), rpad(nanmis, 13), rpad(ident, 10), "(all NaN)")
    end
end
println("\nbit-identical variables: $nident of $(length(c))")
if haskey(ENV, "MGV_SAVE")
    using Serialization; serialize(ENV["MGV_SAVE"], g)
end
