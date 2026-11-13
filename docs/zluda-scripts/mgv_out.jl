# mGV on the CUDA backend with output, 3 timesteps. With ZLUDA_PAGELOCK_THROW=1 this shows where
# mGV stops when pinning fails (workaround 7 in docs/zluda-cuda-run.md).
# Run:  source env.sh; ZLUDA_PAGELOCK_THROW=1 julia -t 4,2 --project=$ZLUDA_JULIA_ENV mgv_out.jl
using CUDA
include(joinpath(@__DIR__, "mgv_common.jl"))
backend = mGV.select_backend("cuda")
try
    out, t = run_landonly(backend, mktempdir(ENV["MGV_OUT"]; cleanup = false); ndays = 3)
    println("ran with output: ", length(out), " variables, ", round(t, digits = 2), " s")
catch e
    println("FAILED: ", first(sprint(showerror, e), 300))
    for fr in stacktrace(catch_backtrace())[1:min(end, 12)]; println("    ", fr); end
end
