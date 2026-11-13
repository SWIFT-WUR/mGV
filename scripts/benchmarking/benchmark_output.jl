# Benchmark a model run including its output, on a GPU or the CPU.
#
#   julia --project -t auto scripts/benchmark_output.jl CONFIG NSTEPS LABEL [BACKEND] [RESULTS]
#
# Runs NSTEPS timesteps of the model in CONFIG. The wall time is measured from the start of
# the second step (the first one compiles) up to and including `finalize!` and a last
# `KernelAbstractions.synchronize`, so that output still being written, and GPU work still
# queued, is included. Also records the garbage collections in that interval, and samples
# `Dirty` and `Writeback` from /proc/meminfo every second (in a separate process, so that it
# is not held up by the Julia scheduler).
#
# Appends a line of results to RESULTS (default `benchmark_output.tsv`) and writes the step
# times and the /proc/meminfo samples next to it. Set `MGV_NO_COPIER=1` to turn off the
# backend's host copier (see `mGV.host_copier`).

using mGV, KernelAbstractions, Statistics, Dates, Printf
const BACKEND = length(ARGS) >= 4 ? ARGS[4] : "rocm"
BACKEND == "rocm" && @eval using AMDGPU
BACKEND == "cuda" && @eval using CUDA

config, nsteps, label = ARGS[1], parse(Int, ARGS[2]), ARGS[3]
results = length(ARGS) >= 5 ? ARGS[5] : "benchmark_output.tsv"
prefix = splitext(results)[1] * "_" * label

if get(ENV, "MGV_NO_COPIER", "") == "1"
    # On the backend's own type, to override the method of its extension (e.g. for ROCBackend)
    @eval mGV.host_copier(::$(typeof(mGV.select_backend(BACKEND)))) = nothing
end

t_init = @elapsed m = mGV.Model(config; backend = BACKEND)
backend = KernelAbstractions.get_backend(m.forcing_variables.air_temperature)

steps = zeros(nsteps)
steps[1] = @elapsed mGV.update!(m)
KernelAbstractions.synchronize(backend)
mGV.TimerOutputs.reset_timer!(mGV.to)

meminfo = prefix * "_meminfo.tsv"
sampler = run(
    pipeline(
        `sh -c 'while :; do printf "%s\t" "$(date +%s.%N)"; awk "/^(Dirty|Writeback):/ {printf \"%s\t\", \$2}" /proc/meminfo; echo; sleep 1; done'`;
        stdout = meminfo,
    );
    wait = false,
)

"""Steps 2 up to `nsteps`, then `finalize!` and a synchronize; returns the times and GC counts."""
function timed_run!(steps, m, backend)
    GC.gc()
    gc0 = Base.gc_num()
    t0 = time_ns()
    for i in 2:length(steps)
        steps[i] = @elapsed mGV.update!(m)
    end
    t_steps = (time_ns() - t0) / 1.0e9
    t_finalize = @elapsed begin
        mGV.finalize!(m)
        KernelAbstractions.synchronize(backend)
    end
    wall = (time_ns() - t0) / 1.0e9
    return wall, t_steps, t_finalize, gc0, Base.gc_num()
end

wall, t_steps, t_finalize, gc0, gc1 = timed_run!(steps, m, backend)
gc = Base.GC_Diff(gc1, gc0)
kill(sampler)

samples = [parse.(Float64, split(strip(l))) for l in eachline(meminfo) if length(split(strip(l))) == 3]
dirty_max = isempty(samples) ? NaN : maximum(s[2] for s in samples) / 2^20  # kB to GiB
writeback_max = isempty(samples) ? NaN : maximum(s[3] for s in samples) / 2^20

open(prefix * "_steps.txt", "w") do io
    foreach(t -> println(io, t), steps)
end
show(stdout, mGV.to; compact = true)
println()

rest = steps[2:end] .* 1.0e3
row = (
    label = label,
    date = string(now()),
    nsteps = nsteps,
    threads = "$(Threads.nthreads(:default)),$(Threads.nthreads(:interactive))",
    init_s = t_init,
    first_step_s = steps[1],
    wall_s = wall,
    steps_s = t_steps,
    finalize_s = t_finalize,
    per_step_ms = wall / (nsteps - 1) * 1.0e3,
    median_step_ms = median(rest),
    p90_step_ms = quantile(rest, 0.9),
    gc_time_s = gc.total_time / 1.0e9,
    gc_pauses = gc.pause,
    gc_full = gc.full_sweep,
    time_to_safepoint_s = (gc1.total_time_to_safepoint - gc0.total_time_to_safepoint) / 1.0e9,
    max_time_to_safepoint_ms = gc1.max_time_to_safepoint / 1.0e6,
    gc_allocd_GiB = gc.allocd / 2^30,
    dirty_max_GiB = dirty_max,
    writeback_max_GiB = writeback_max,
)
isfile(results) || open(io -> println(io, join(keys(row), '\t')), results, "w")
fmt(x) = x isa AbstractFloat ? @sprintf("%.4g", x) : string(x)
open(io -> println(io, join(fmt.(values(row)), '\t')), results, "a")
for (k, v) in pairs(row)
    println(rpad(k, 22), fmt(v))
end
