# Compares two outputs stored by mgv_compare.jl (MGV_SAVE):  julia cmp_jls.jl a.jls b.jls
using Serialization
a = deserialize(ARGS[1]); b = deserialize(ARGS[2])
println(rpad("variable", 24), rpad("identical", 10), rpad("NaN mismatch", 13), rpad("max abs diff", 14), rpad("/ max|a|", 15), "frac differing")
n = 0
for name in sort(collect(keys(a)))
    x, y = a[name], b[name]; id = isequal(x, y); global n += id
    ok = .!isnan.(x) .& .!isnan.(y)
    if any(ok)
        d = abs.(Float64.(x[ok]) .- Float64.(y[ok])); mx = maximum(abs, x[ok])
        println(rpad(name, 24), rpad(id, 10), rpad(count(isnan.(x) .!= isnan.(y)), 13), rpad(Float32(maximum(d)), 14), rpad(Float32(maximum(d) / max(mx, 1f-30)), 15), round(count(>(0), d) / length(d), digits = 4))
    else
        println(rpad(name, 24), rpad(id, 10), rpad(count(isnan.(x) .!= isnan.(y)), 13), "(all NaN)")
    end
end
println("bit-identical: $n of $(length(a))")
