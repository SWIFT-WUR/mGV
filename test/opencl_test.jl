# Runs the model on the OpenCL backend and compares the results against the CPU backend,
# and checks the BMI getters and setters on (non-contiguous) device array views.
# Not part of runtests.jl, as it requires OpenCL.jl and an OpenCL driver. In CI this is
# run on the CPU through pocl (see .github/workflows/opencl.yml):
#   julia --project=<env with mGV, OpenCL, pocl_jll> test/opencl_test.jl
using Test
using pocl_jll, OpenCL
using mGV
using TOML
import BasicModelInterface as BMI

# Use the pocl platform, even if other OpenCL drivers are installed.
let platform = only(filter(p -> occursin("Portable", p.name), cl.platforms()))
    cl.platform!(platform)
end
OpenCL.versioninfo()

const NSTEPS = parse(Int, get(ENV, "MGV_TEST_STEPS", "30"))
const VARIABLES = [
    "soil__moisture_layer_1",
    "soil__moisture_layer_3",
    "land_surface__temperature",
    "land_surface__evaporation~total",
    "land__runoff~total",
    "channel_water__volume_flow_rate",
]

function get_values(model, name)
    dest = zeros(Float64, BMI.get_grid_size(model, BMI.get_var_grid(model, name)))
    BMI.get_value(model, name, dest)
    return dest
end

# Copy of the Mekong config that writes its output to a separate directory per backend.
function make_config(backend)
    config_dir = normpath(joinpath(@__DIR__, "../configs"))
    config = TOML.parsefile(joinpath(config_dir, "mekong_config.toml"))
    paths = config["input"]["paths"]
    for (key, path) in paths
        paths[key] = joinpath(config_dir, path)
    end
    dir = mktempdir()
    config["output"]["dir"] = mkpath(joinpath(dir, "output_$backend"))
    tomlpath = joinpath(dir, "config_$backend.toml")
    open(io -> TOML.print(io, config), tomlpath, "w")
    return tomlpath
end

function run_steps(backend)
    model = mGV.Model(make_config(backend); backend)
    for _ in 1:NSTEPS
        BMI.update(model)
    end
    values = Dict(name => get_values(model, name) for name in VARIABLES)
    BMI.finalize(model)
    return model, values
end

@testset "OpenCL backend (pocl)" begin
    @test mGV.select_backend("opencl") isa OpenCLBackend

    gpu_model, gpu_values = run_steps("opencl")
    @test gpu_model.grid_parameters.latitude isa CLArray
    _, cpu_values = run_steps("cpu")

    for name in VARIABLES
        gpu, cpu = gpu_values[name], cpu_values[name]
        valid = .!isnan.(cpu) .& .!isnan.(gpu)
        @testset "$name" begin
            @test any(valid)
            @test all(isfinite, gpu[valid])
            @test isapprox(gpu[valid], cpu[valid]; rtol = 1.0e-3)
        end
    end
end

include("bmi_views.jl")

@testset "OpenCL BMI layer and band views" begin
    model = mGV.Model(make_config("opencl"); backend = "opencl")
    @test model.grid_parameters.latitude isa CLArray
    test_layered_views(model)
    BMI.finalize(model)
end
