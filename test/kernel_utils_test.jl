using Test
using mGV
using KernelAbstractions

# the n-th Fibonacci number and sum of 1:n, for n = 30 steps (unrolled on the CPU backend)
fibonacci_sum(x) = mGV.foldsteps((x, x, 0), Val(30)) do step, (a, b, s)
    @inline
    return (b, a + b, s + step)
end

@kernel inbounds=true function foldsteps_kernel!(fib, sums, @Const(x))
    i = @index(Global, Linear)
    a, _, s = fibonacci_sum(x[i])
    fib[i] = a
    sums[i] = s
end

@testset "foldsteps" begin
    @test fibonacci_sum(1) == (1346269, 2178309, 465)
    x = collect(1:100)
    fib, sums = similar(x), similar(x)
    backend = CPU()
    foldsteps_kernel!(backend)(fib, sums, x; ndrange = length(x))
    KernelAbstractions.synchronize(backend)
    @test fib == [fibonacci_sum(xi)[1] for xi in x]
    @test all(==(465), sums)
end
