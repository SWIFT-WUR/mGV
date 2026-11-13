# Run:  source env.sh; julia -t 4,2 --project=$ZLUDA_JULIA_ENV step2b.jl   (section 4.3 of docs/zluda-cuda-run.md)
# Timing of KernelAbstractions.copyto! (device -> host) while a long kernel is queued, on ZLUDA,
# for pageable host memory and for pinned memory allocated by the driver (cuMemHostAlloc), and of
# the full sequence of calls the CUDA host copier makes.
using CUDA, KernelAbstractions, Statistics
const KA = KernelAbstractions
const CC = CUDA.CUDACore
be = CUDABackend()
@kernel function slow!(a, n)
    i = @index(Global)
    acc = 0f0
    for k in 1:n; acc += sin(Float32(k) * 1f-3 + Float32(i)); end
    @inbounds a[i] = acc
end
M = 1_000_000
big = KA.zeros(be, Float32, M)
slow!(be)(big, 10; ndrange = M); KA.synchronize(be)
NITER = 20000
tk = minimum(@elapsed((slow!(be)(big, NITER; ndrange = M); KA.synchronize(be))) for _ in 1:3)
println("long kernel: $(round(tk * 1000, digits=1)) ms (launch + synchronize)")
tl = minimum(@elapsed(slow!(be)(big, NITER; ndrange = M)) for _ in 1:3); KA.synchronize(be)
println("long kernel launch only (no synchronize): $(round(tl * 1000, digits=3)) ms")

src = CuArray(Float32.(1:M)); ref = Float32.(1:M)   # 4 MB, independent of the kernel's array
pageable = zeros(Float32, M)
pinned_ok = try KA.pagelock!(be, zeros(Float32, M)); true catch e; println("KA.pagelock!/CUDA.pin: FAILS: ", first(sprint(showerror, e), 120)); false end
# pinned memory the only way ZLUDA offers it: allocated by the driver (cuMemHostAlloc)
hostmem = try CC.alloc(CC.HostMemory, M * sizeof(Float32)) catch e; println("cuMemHostAlloc FAILS: ", first(sprint(showerror, e), 120)); nothing end
hostalloc = hostmem === nothing ? nothing : unsafe_wrap(Array, convert(Ptr{Float32}, hostmem), M)
hostalloc === nothing || println("cuMemHostAlloc array: is_pinned = ", CC.is_pinned(pointer(hostalloc)))
s2 = CuStream()
ms(x) = round(x * 1000, digits = 3)
function trial(host, label; other_stream = false, queued = true)
    rs = Float64[]; ss = Float64[]; ok = true
    for _ in 1:5
        fill!(host, 0); CUDA.synchronize(); CUDA.synchronize(s2)
        queued && slow!(be)(big, NITER; ndrange = M)       # queued on the task's stream, not synchronized
        t0 = time()
        if other_stream
            CUDA.stream!(s2) do; KA.copyto!(be, host, src); end
        else
            KA.copyto!(be, host, src)
        end
        t1 = time()
        other_stream ? CUDA.synchronize(s2) : KA.synchronize(be)
        t2 = time()
        KA.synchronize(be)
        ok &= host == ref
        push!(rs, t1 - t0); push!(ss, t2 - t1)
    end
    println(rpad(label, 62), "copyto! returns after ", lpad(ms(median(rs)), 8), " ms (min ", ms(minimum(rs)), ", max ", ms(maximum(rs)), "); then synchronize ", lpad(ms(median(ss)), 8), " ms; data ok: ", ok)
end
println("\n-- idle device (no kernel queued)")
trial(pageable, "pageable, task stream"; queued = false)
hostalloc === nothing || trial(hostalloc, "pinned (cuMemHostAlloc), task stream"; queued = false)
println("\n-- long kernel queued on the task stream, copy on the SAME stream")
trial(pageable, "pageable, same stream")
hostalloc === nothing || trial(hostalloc, "pinned (cuMemHostAlloc), same stream")
println("\n-- long kernel queued on the task stream, copy on a SECOND stream (src not touched by the kernel)")
trial(pageable, "pageable, second stream"; other_stream = true)
hostalloc === nothing || trial(hostalloc, "pinned (cuMemHostAlloc), second stream"; other_stream = true)

println("\n-- copier sequence (as ext/CUDAExt.jl): kernel on model stream; record(staged); wait(staged, copier stream); copyto! of the kernel's OUTPUT on copier stream; record(done); synchronize(done)")
for (host, label) in ((zeros(Float32, M), "pageable"), (hostalloc, "pinned (cuMemHostAlloc)"))
    host === nothing && continue
    for rep in 1:3
        fill!(big, -1f0); fill!(host, 0); CUDA.synchronize(); CUDA.synchronize(s2)
        staged = CuEvent(CUDA.EVENT_DISABLE_TIMING); done = CuEvent(CUDA.EVENT_DISABLE_TIMING)
        t0 = time(); slow!(be)(big, NITER; ndrange = M)
        t1 = time(); CUDA.record(staged, CUDA.stream()); CUDA.wait(staged, s2)
        t2 = time(); CUDA.stream!(s2) do; KA.copyto!(be, host, big); end
        t3 = time(); CUDA.record(done, s2)
        t4 = time(); CUDA.synchronize(done)
        t5 = time()
        stale = count(==(-1f0), host); zero = count(==(0f0), host)
        CUDA.synchronize(); expected = Array(big)
        println(rpad(label, 26), "launch ", ms(t1 - t0), "  record+wait ", ms(t2 - t1), "  copyto! ", ms(t3 - t2), "  record(done) ", ms(t4 - t3), "  synchronize(done) ", ms(t5 - t4), " ms | stale values ", stale, ", untouched zeros ", zero, ", equal to device ", host == expected)
    end
end
flush(stdout)
