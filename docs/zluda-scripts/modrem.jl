# Float64 rem/mod on ZLUDA (silently wrong, section 5.1 of docs/zluda-cuda-run.md), and whether an
# exception thrown in a kernel is still reported on the host with the reduced exception runtime.
# Run:  source env.sh; julia --project=$ZLUDA_JULIA_ENV modrem.jl
using CUDA
x = Float64.(Float32.(range(0.1, 20, length=4096))); 
for (n, f) in (("rem(x,7.0)", x -> rem(x, 7.0)), ("mod(x,3.0)", x -> mod(x, 3.0)), ("rem(x,3.0)", x -> rem(x, 3.0)), ("x - 3floor(x/3)", x -> x - 3.0 * floor(x / 3.0)), ("rem Float32", x -> Float64(rem(Float32(x), 3f0))))
    g = Array(f.(CuArray(x))); c = f.(x); bad = findall(g .!= c)
    println(rpad(n, 18), "mismatches: ", length(bad), "/", length(x), isempty(bad) ? "" : "  e.g. x=$(x[bad[1]]) gpu=$(g[bad[1]]) cpu=$(c[bad[1]]); x=$(x[bad[end]]) gpu=$(g[bad[end]]) cpu=$(c[bad[end]])")
end
# exception detection
try
    y = sqrt.(CuArray([1f0, -1f0])); CUDA.synchronize(); println("sqrt(-1) on GPU: no exception raised on host, result ", Array(y))
catch e
    println("sqrt(-1) on GPU raised: ", first(sprint(showerror, e), 150))
end
# pattern of the error: which quotients floor(x / y) are affected, and what the wrong values are
let x = collect(range(0.0, 30.0, length = 30001))
    for y in (3.0, 7.0)
        g = Array(rem.(CuArray(x), y)); c = rem.(x, y); bad = g .!= c
        q = floor.(Int, x ./ y)
        println("rem(x, $y) on 30001 values in [0, 30]: wrong ", count(bad), "; per quotient floor(x/y): ",
            join(["$k: $(count(bad[q .== k]))/$(count(q .== k))" for k in 0:maximum(q)], ", "))
        any(bad) && println("   wrong GPU values range from ", minimum(g[bad]), " to ", maximum(g[bad]),
            "; smallest CPU remainder that is wrong: ", minimum(c[bad]), ", largest that is right (x >= y): ", maximum(c[.!bad .& (x .>= y)]))
    end
end
