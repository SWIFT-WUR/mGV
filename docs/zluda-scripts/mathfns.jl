# Compares 28 elementary functions in broadcasts on the GPU with the CPU, for Float32 and Float64
# (section 4.1 of docs/zluda-cuda-run.md).  Run:  source env.sh; julia --project=$ZLUDA_JULIA_ENV mathfns.jl
using CUDA
x32 = Float32.(range(0.1, 20, length=4096)); x64 = Float64.(x32)
fns = [("sin", sin), ("cos", cos), ("tan", tan), ("exp", exp), ("exp2", exp2), ("exp10", exp10), ("expm1", expm1), ("log", log), ("log10", log10), ("log1p", log1p), ("sqrt", sqrt), ("cbrt", cbrt), ("tanh", tanh), ("atan", atan), ("asin∘(x/21)", x -> asin(x / 21)), ("x^2.5", x -> x^oftype(x, 2.5)), ("x^3", x -> x^3), ("abs/min/max/clamp", x -> clamp(abs(x - 5), 1, 3)), ("floor/round", x -> floor(x) + round(x)), ("fma/muladd", x -> muladd(x, x, x)), ("mod/rem", x -> mod(x, oftype(x, 3)) + rem(x, oftype(x, 7))), ("Int32(floor)", x -> Float32(unsafe_trunc(Int32, x))), ("ifelse/isnan", x -> ifelse(isnan(x), zero(x), x)), ("sign/copysign", x -> copysign(x, -one(x)) + sign(x)), ("hypot", x -> hypot(x, one(x))), ("atan2", x -> atan(x, one(x))), ("sinh/cosh", x -> sinh(x / 10) + cosh(x / 10)), ("fld/div Int", x -> Float32(fld(unsafe_trunc(Int32, x), Int32(3))))]
for (T, x) in ((Float32, x32), (Float64, x64)), (name, f) in fns
    try
        g = Array(f.(CuArray(x))); c = f.(x)
        d = maximum(abs.(g .- c) ./ max.(abs.(c), floatmin(T)))
        println(rpad("$T $name", 30), d == 0 ? "exact" : "max rel diff $(Float32(d)) ($(round(d / eps(T), digits=1)) eps)")
    catch e
        println(rpad("$T $name", 30), "FAIL ", first(sprint(showerror, e), 80))
    end
    flush(stdout)
end
