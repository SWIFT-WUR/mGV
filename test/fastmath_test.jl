using Test
using mGV: fastexp2, fastlog2, fastlog10, fastpow

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

@testset "Fast math" begin
    @testset "Accuracy" begin
        @test max_ulp_error(fastexp2, exp2, float32_range(0.0f0, 127.99f0, 997)) <= 1.0
        @test max_ulp_error(fastexp2, exp2, float32_range(-0.0f0, -125.99f0, 997)) <= 1.0
        positive = float32_range(nextfloat(0.0f0), floatmax(Float32), 997)
        @test max_ulp_error(fastlog2, log2, positive) <= 1.8
        @test max_ulp_error(fastlog10, log10, positive) <= 3.1
        # exact for powers of two
        @test all(fastexp2(Float32(n)) == exp2(Float32(n)) for n in -149:127)
        @test all(fastlog2(exp2(Float32(n))) == n for n in -149:127)
    end

    @testset "Special values" begin
        @test fastexp2(0.0f0) === 1.0f0
        @test fastexp2(Inf32) === Inf32
        @test fastexp2(128.0f0) === Inf32
        @test fastexp2(-Inf32) === 0.0f0
        @test fastexp2(-200.0f0) === 0.0f0
        @test fastexp2(-149.0f0) === nextfloat(0.0f0)
        @test isnan(fastexp2(NaN32))

        @test fastlog2(1.0f0) === 0.0f0
        @test fastlog2(0.0f0) === -Inf32
        @test fastlog2(Inf32) === Inf32
        @test isnan(fastlog2(-1.0f0))
        @test isnan(fastlog2(NaN32))
        @test fastlog10(0.0f0) === -Inf32
    end

    @testset "fastpow" begin
        @test fastpow(2.0f0, 0.0f0) === 1.0f0
        @test fastpow(0.0f0, 0.0f0) === 1.0f0
        @test fastpow(0.0f0, 1.5f0) === 0.0f0
        @test isnan(fastpow(-1.0f0, 0.5f0))
        for x in (1.0f-6, 0.3f0, 1.0f0, 7.7f0, 1234.5f0), y in (-2.5f0, 2.0f0 / 3.0f0, 1.7f0, 5.0f0 / 3.0f0)
            @test fastpow(x, y) ≈ Float32(Float64(x)^Float64(y)) rtol = 1.0f-5
        end
        # other types fall back to Base
        @test fastpow(2.0, 0.5) == exp2(0.5 * log2(2.0))
    end
end
