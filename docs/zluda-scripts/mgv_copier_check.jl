# With a mGV copy that has a CUDA host copier: shows that the copier is in use, whether the output
# blocks can be pinned, and compares 60 timesteps with and without it (section 4.5 of
# docs/zluda-cuda-run.md).  Run:  source env.sh; julia -t 4,2 --project=$ZLUDA_JULIA_ENV mgv_copier_check.jl
using CUDA, KernelAbstractions
include(joinpath(@__DIR__, "mgv_common.jl"))
be = mGV.select_backend("cuda")
c = mGV.host_copier(be)
println("host_copier(CUDABackend()) :: ", typeof(c))
println("pin_memory!(backend, Array) returns ", mGV.pin_memory!(be, zeros(Float32, 1000)), "  (false = output blocks are pageable on ZLUDA)")
run_landonly(be, mktempdir(ENV["MGV_OUT"]; cleanup = false); ndays = 4)   # compile
a, ta = run_landonly(be, mktempdir(ENV["MGV_OUT"]; cleanup = false); ndays = 60, write_interval = 8)
@eval mGV.host_copier(::CUDABackend) = nothing
b, tb = run_landonly(be, mktempdir(ENV["MGV_OUT"]; cleanup = false); ndays = 60, write_interval = 8)
println("60 timesteps, write_interval 8: with copier $(round(ta, digits=3)) s, without copier (copy task) $(round(tb, digits=3)) s")
println("outputs identical with/without copier: ", all(isequal(a[k], b[k]) for k in keys(a)), " (", length(a), " variables)")
