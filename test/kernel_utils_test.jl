using Test
using mGV
using KernelAbstractions

mGV.@kernel_inbounds function double_kernel!(y, @Const(x))
    i = @index(Global, Linear)
    y[i] = 2.0f0 * x[i]
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
