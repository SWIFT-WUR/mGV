using Test
using mGV
using KernelAbstractions

mGV.@kernel_inbounds function double_kernel!(y, @Const(x))
    i = @index(Global, Linear)
    y[i] = 2.0f0 * x[i]
end

# the n-th Fibonacci number and sum of 1:n, for n = 30 steps (unrolled on the CPU backend)
fibonacci_sum(x) = mGV.foldsteps((x, x, 0), Val(30)) do step, (a, b, s)
    @inline
    return (b, a + b, s + step)
end

mGV.@kernel_inbounds function foldsteps_kernel!(fib, sums, @Const(x))
    i = @index(Global, Linear)
    a, _, s = fibonacci_sum(x[i])
    fib[i] = a
    sums[i] = s
end

@testset "@kernel_inbounds" begin
    # `@kernel` rewrites the kernel body; check that bounds checking stays disabled in it
    ex = @macroexpand mGV.@kernel_inbounds function k!(a)
        i = @index(Global, Linear)
        a[i] = 1.0f0
    end
    found = Ref(false)
    walk(e) = e isa Expr && (e == Expr(:inbounds, true) ? (found[] = true) : foreach(walk, e.args))
    walk(ex)
    @test found[]

    @test_throws ArgumentError @macroexpand mGV.@kernel_inbounds f!(a) = nothing

    x = collect(Float32, 1:1000)
    y = similar(x)
    backend = CPU()
    double_kernel!(backend)(y, x; ndrange = length(x))
    KernelAbstractions.synchronize(backend)
    @test y == 2 .* x
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
