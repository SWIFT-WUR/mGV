using Test
using KernelAbstractions
using mGV: fastexp2, fastlog2, fastlog10, fastpow, polynomial_exp2, polynomial_log2, polynomial_log10

# error of `y` in units in the last place of the exact (Float64) result
ulp_error(y::Float32, exact::Float64) = abs(Float64(y) - exact) / Float64(eps(Float32(exact)))

# every `step`th Float32 between `lo` and `hi` (both of the same sign)
function float32_range(lo::Float32, hi::Float32, step::Integer)
    a, b = minmax(reinterpret(UInt32, lo), reinterpret(UInt32, hi))
    return (reinterpret(Float32, u) for u in a:UInt32(step):b)
end

function max_ulp_error(f, exact, xs)
    worst = 0.0
    for x in xs
        e = exact(Float64(x))
        # only compare normal, finite results
        floatmin(Float32) < abs(e) < floatmax(Float32) || continue
        worst = max(worst, ulp_error(f(x), e))
    end
    return worst
end

# calls the fast math functions from a kernel
@kernel function fastmath_kernel!(e, l, l10, @Const(x))
    i = @index(Global, Linear)
    @inbounds begin
        e[i] = fastexp2(x[i] / 8.0f0)
        l[i] = fastlog2(x[i])
        l10[i] = fastlog10(x[i])
    end
end

@testset "Fast math" begin
    @testset "Accuracy" begin
        @test max_ulp_error(polynomial_exp2, exp2, float32_range(0.0f0, 127.99f0, 997)) <= 1.0
        @test max_ulp_error(polynomial_exp2, exp2, float32_range(-0.0f0, -125.99f0, 997)) <= 1.0
        positive = float32_range(nextfloat(0.0f0), floatmax(Float32), 997)
        @test max_ulp_error(polynomial_log2, log2, positive) <= 1.8
        @test max_ulp_error(polynomial_log10, log10, positive) <= 3.1
        # exact for powers of two
        @test all(polynomial_exp2(Float32(n)) == exp2(Float32(n)) for n in -149:127)
        @test all(polynomial_log2(exp2(Float32(n))) == n for n in -149:127)
    end

    @testset "Special values" begin
        @test polynomial_exp2(0.0f0) === 1.0f0
        @test polynomial_exp2(Inf32) === Inf32
        @test polynomial_exp2(128.0f0) === Inf32
        @test polynomial_exp2(-Inf32) === 0.0f0
        @test polynomial_exp2(-200.0f0) === 0.0f0
        @test polynomial_exp2(-149.0f0) === nextfloat(0.0f0)
        @test isnan(polynomial_exp2(NaN32))

        @test polynomial_log2(1.0f0) === 0.0f0
        @test polynomial_log2(0.0f0) === -Inf32
        @test polynomial_log2(Inf32) === Inf32
        @test isnan(polynomial_log2(-1.0f0))
        @test isnan(polynomial_log2(NaN32))
        @test polynomial_log10(0.0f0) === -Inf32
    end

    @testset "fastpow" begin
        @test fastpow(2.0f0, 0.0f0) === 1.0f0
        @test fastpow(0.0f0, 0.0f0) === 1.0f0
        @test fastpow(0.0f0, 1.5f0) === 0.0f0
        @test_throws DomainError fastpow(-1.0f0, 0.5f0)  # on the host; `NaN` in kernels
        for x in (1.0f-6, 0.3f0, 1.0f0, 7.7f0, 1234.5f0), y in (-2.5f0, 2.0f0 / 3.0f0, 1.7f0, 5.0f0 / 3.0f0)
            @test fastpow(x, y) ≈ Float32(Float64(x)^Float64(y)) rtol = 1.0f-5
        end
        @test fastpow(2.0, 0.5) == exp2(0.5 * log2(2.0))
    end

    @testset "Polynomial versions only on PoCL" begin
        # inputs for which each polynomial version differs from Base in the last bit
        xs = Float32[1.067888, 1.691513, 2.4074345, 4.3182216, 5.1314287, 6.311327]
        @test all(polynomial_exp2.(xs ./ 8.0f0) .!= exp2.(xs ./ 8.0f0))
        @test all(polynomial_log2.(xs) .!= log2.(xs))
        @test all(polynomial_log10.(xs) .!= log10.(xs))

        # host code uses Base
        @test fastexp2.(xs ./ 8.0f0) == exp2.(xs ./ 8.0f0)
        @test fastlog2.(xs) == log2.(xs)
        @test fastlog10.(xs) == log10.(xs)

        # kernels on the CPU backend (PoCL) use the polynomial versions
        e, l, l10 = similar(xs), similar(xs), similar(xs)
        backend = CPU()
        fastmath_kernel!(backend)(e, l, l10, xs; ndrange = length(xs))
        KernelAbstractions.synchronize(backend)
        @test e == polynomial_exp2.(xs ./ 8.0f0)
        @test l == polynomial_log2.(xs)
        @test l10 == polynomial_log10.(xs)
    end
end
