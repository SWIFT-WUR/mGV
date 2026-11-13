# mGV on the CUDA backend without output (variables = []), a few timesteps: the smallest test of
# mGV's kernels.  Run:  source env.sh; julia -t 4,2 --project=$ZLUDA_JULIA_ENV mgv_noout.jl 3
# Set ZLUDA_PTX_DUMP_DIR=<dir> to keep the PTX of every compiled kernel (before the fix-ups).
using CUDA
include(joinpath(@__DIR__, "mgv_common.jl"))
ndays = parse(Int, get(ARGS, 1, "3"))
backend = mGV.select_backend("cuda")
println("backend: ", mGV.backend_name(backend), " on ", CUDA.name(CUDA.device())); flush(stdout)
dir = mktempdir(ENV["MGV_OUT"]; cleanup = false)
file = make_config(dir; variables = String[])
t0 = time(); model = mGV.Model(file; backend); println("Model constructed in $(round(time() - t0, digits=1)) s"); flush(stdout)
for d in 1:ndays
    t = @elapsed (mGV.update!(model); CUDA.synchronize())
    println("update! $d: $(round(t, digits=2)) s"); flush(stdout)
end
mGV.finalize!(model)
println("finalize! done")
