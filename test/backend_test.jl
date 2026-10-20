using Test
using mGV
using KernelAbstractions: CPU

@testset "Backend selection" begin
    @test isempty(mGV.BACKEND_REGISTRY)  # no GPU package loaded in the test environment
    @test mGV.select_backend() isa CPU
    @test mGV.select_backend("CPU") isa CPU
    @test mGV.select_backend(CPU()) isa CPU
    @test_throws "requires CUDA.jl" mGV.select_backend("cuda")
    @test_throws "Unknown backend 'foo'" mGV.select_backend("foo")
    @test mGV.backend_options()[1:3] == ["auto", "cpu", "cuda"]

    # Fake registrations to test the dispatch on registered backends
    constructed = String[]
    try
        mGV.register_backend!("rocm", "AMDGPU", () -> false, () -> (push!(constructed, "rocm"); CPU()))
        mGV.register_backend!("metal", "Metal", () -> true, () -> (push!(constructed, "metal"); CPU()))
        mGV.register_backend!("opencl", "OpenCL", () -> true, () -> (push!(constructed, "opencl"); CPU()))

        @test_throws "AMDGPU is not functional" mGV.select_backend("rocm")
        # "auto" skips the non-functional rocm backend and prefers metal over opencl
        mGV.select_backend("auto")
        @test constructed == ["metal"]
        mGV.select_backend("OpenCL")
        @test constructed == ["metal", "opencl"]
    finally
        empty!(mGV.BACKEND_REGISTRY)
    end
end
