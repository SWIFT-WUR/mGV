# Driver features the CUDA host copier needs, each tested separately (section 4.2 of
# docs/zluda-cuda-run.md).  Run:  source env.sh; julia -t 4,2 --project=$ZLUDA_JULIA_ENV step2.jl
# With the patched CUDACore, KernelAbstractions.pagelock! returns `missing` instead of throwing;
# set ZLUDA_PAGELOCK_THROW=1 to see the original error of cuMemHostRegister_v2 there as well.
# The last stage runs the copier sequence with whatever pagelock! gives (pageable on ZLUDA);
# step2b.jl times that sequence for pageable and for driver-allocated pinned memory.
using CUDA, KernelAbstractions
const KA = KernelAbstractions
stage(f, name) = try
    flush(stdout); r = f(); println("[OK]   ", name, r === nothing ? "" : " => $r")
catch e
    println("[FAIL] ", name, ": ", first(sprint(showerror, e), 400))
    for fr in stacktrace(catch_backtrace())[1:min(end, 6)]; println("         ", fr); end
end
be = CUDABackend()
N = 4_000_000
dev = CuArray(Float32.(1:N)); ref = Float32.(1:N)
host = zeros(Float32, N)
stage("CUDA.pin(Array)") do; h = zeros(Float32, N); CUDA.pin(h); "is_pinned=$(CUDA.CUDACore.is_pinned(pointer(h)))"; end
stage("KA.pagelock!") do; KA.pagelock!(be, host); "is_pinned=$(CUDA.CUDACore.is_pinned(pointer(host)))"; end
stage("is_pinned of unpinned array") do; "is_pinned=$(CUDA.CUDACore.is_pinned(pointer(zeros(Float32, N))))"; end
stage("KA.copyto!(be, host, dev) + synchronize, data check  [host: after pagelock!, which does not pin on ZLUDA]") do
    fill!(host, 0); KA.copyto!(be, host, dev); KA.synchronize(be); host == ref || error("data mismatch: $(count(host .!= ref)) of $N differ"); "data equal"
end
stage("KA.copyto! into a view of host (offset), data check") do
    fill!(host, 0); KA.copyto!(be, view(host, 1001:2000), view(dev, 5001:6000)); KA.synchronize(be)
    host[1001:2000] == ref[5001:6000] && all(iszero, host[1:1000]) && all(iszero, host[2001:end]) || error("data mismatch"); "data equal"
end
stage("KA.copyto!(be, fresh pageable host, dev) + synchronize, data check") do
    h = zeros(Float32, N); KA.copyto!(be, h, dev); KA.synchronize(be); h == ref || error("data mismatch"); "data equal"
end
s2 = nothing
stage("CuStream()") do; global s2 = CuStream(); string(s2); end
stage("copy on second stream (stream! do ... KA.copyto!), sync that stream, data check") do
    fill!(host, 0); CUDA.stream!(s2) do; KA.copyto!(be, host, dev); end
    CUDA.synchronize(s2); host == ref || error("data mismatch: $(count(host .!= ref)) differ"); "data equal"
end
ev = nothing
stage("CuEvent(EVENT_DISABLE_TIMING)") do; global ev = CuEvent(CUDA.EVENT_DISABLE_TIMING); nothing; end
stage("CUDA.record(event, task stream)") do; CUDA.record(ev, CUDA.stream()); nothing; end
stage("CUDA.wait(event, second stream)  [cuStreamWaitEvent]") do; CUDA.wait(ev, s2); nothing; end
stage("CUDA.synchronize(event) from host") do; CUDA.synchronize(ev); nothing; end
stage("CUDA.isdone / query(event)") do; isdefined(CUDA, :isdone) ? "isdone=$(CUDA.isdone(ev))" : "n/a"; end
stage("record on fresh empty stream, then wait on task stream (HostCopier constructor + begin_staging!)") do
    s = CuStream(); d = CuEvent(CUDA.EVENT_DISABLE_TIMING); CUDA.record(d, s); CUDA.wait(d, CUDA.stream()); CUDA.synchronize(); nothing
end
stage("CUDA.synchronize(event) from another task with context!") do
    ctx = CUDA.context(); e = CuEvent(CUDA.EVENT_DISABLE_TIMING); CUDA.record(e, s2)
    fetch(Threads.@spawn CUDA.context!(() -> CUDA.synchronize(e), ctx)); nothing
end

# ordering semantics: does a stream that waits for an event really run after the work before the event?
@kernel function slow!(a, n)
    i = @index(Global)
    acc = 0f0
    for k in 1:n; acc += sin(Float32(k) * 1f-3 + Float32(i)); end
    @inbounds a[i] = acc
end
big = KA.zeros(be, Float32, 1_000_000)
slow!(be)(big, 10; ndrange = length(big)); KA.synchronize(be)  # compile
tk = @elapsed (slow!(be)(big, 3000; ndrange = length(big)); KA.synchronize(be))
println("slow kernel takes $(round(tk * 1000, digits=1)) ms when synchronized")
stage("full copier sequence: kernel on model stream, record staged, copier stream waits, async copy, record done, synchronize(done)") do
    fill!(big, -1f0); CUDA.synchronize()
    h = zeros(Float32, length(big)); KA.pagelock!(be, h)
    staged = CuEvent(CUDA.EVENT_DISABLE_TIMING); done = CuEvent(CUDA.EVENT_DISABLE_TIMING)
    t0 = time()
    slow!(be)(big, 3000; ndrange = length(big))            # model stream, not synchronized
    t1 = time(); CUDA.record(staged, CUDA.stream()); CUDA.wait(staged, s2)
    t2 = time(); CUDA.stream!(s2) do; KA.copyto!(be, h, big); end
    t3 = time(); CUDA.record(done, s2)
    t4 = time(); CUDA.synchronize(done)
    t5 = time()
    stale = count(==(-1f0), h)
    expected = Array(big)
    ms(a, b) = round((b - a) * 1000, digits = 2)
    "launch $(ms(t0,t1)) ms, record+wait $(ms(t1,t2)) ms, copyto! $(ms(t2,t3)) ms, record $(ms(t3,t4)) ms, synchronize(done) $(ms(t4,t5)) ms; stale (pre-kernel) values in host copy: $stale of $(length(h)); equal to device: $(h == expected)"
end
flush(stdout)
