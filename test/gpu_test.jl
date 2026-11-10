# Checks the BMI getters and setters on the layer views (views into the 2D model
# arrays) on a locally available GPU backend. Not part of runtests.jl, as it requires a GPU.
# Run it from an environment with mGV, BasicModelInterface and CUDA and/or AMDGPU:
#   julia --project=<env> test/gpu_test.jl
# By default the first functional GPU backend is used, and the tests are skipped if there is
# none. Set MGV_TEST_BACKEND (e.g. "cuda" or "rocm") to require a specific backend.
using Test
using mGV
import BasicModelInterface as BMI

const MEKONG_CFG = joinpath(@__DIR__, "../configs/mekong_config.toml")

# Load the GPU packages that are available, which registers their backends with mGV
for pkg in ("CUDA", "AMDGPU")
    isnothing(Base.find_package(pkg)) || @eval using $(Symbol(pkg))
end

const backend = mGV.select_backend(get(ENV, "MGV_TEST_BACKEND", "auto"))

include("bmi_views.jl")

if mGV.backend_name(backend) == "CPU"
    @warn "No functional GPU backend (CUDA or AMDGPU) found, skipping the GPU tests"
else
    @testset "$(mGV.backend_name(backend)) BMI layer views" begin
        model = mGV.Model(MEKONG_CFG; backend)
        @test !(model.grid_parameters.latitude isa Array)  # the model arrays are on the device
        test_layered_views(model)
        BMI.finalize(model)
    end
end
