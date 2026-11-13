# Checks the BMI getters and setters on the layer views (views into the 2D model
# arrays) on a locally available GPU backend. Not part of runtests.jl, as it requires a GPU.
# Run it from an environment with mGV, BasicModelInterface and CUDA and/or AMDGPU:
#   julia --project=<env> test/gpu_test.jl
# By default the first functional GPU backend is used, and the tests are skipped if there is
# none. Set MGV_TEST_BACKEND (e.g. "cuda" or "rocm") to require a specific backend.
using Test
using mGV
import BasicModelInterface as BMI
using TOML
using Zarr: zopen

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

"""
Run the Mekong model on `backend` for `ndays` with land-only Zarr output in blocks of
`write_interval` timesteps, packed or not (`packing`), and return the output arrays by name.
"""
function landonly_output(backend; ndays = 20, write_interval = 8, packing = true)
    dir = mktempdir()
    configs_dir = joinpath(@__DIR__, "../configs")
    cfg = TOML.parsefile(MEKONG_CFG)
    for (key, path) in cfg["input"]["paths"]
        cfg["input"]["paths"][key] = abspath(joinpath(configs_dir, path))
    end
    merge!(cfg["output"], Dict("dir" => dir, "format" => "zarr_landonly", "write_interval" => write_interval, "packing" => packing))
    config_file = joinpath(dir, "config.toml")
    open(io -> TOML.print(io, cfg), config_file, "w")

    model = mGV.Model(config_file; backend)
    for _ in 1:ndays
        mGV.update!(model)
    end
    mGV.finalize!(model)
    store = zopen(joinpath(dir, only(filter(endswith(".zarr"), readdir(dir)))))
    # Size (cells, time) or (cells, time, layers); the coordinates have one dimension
    return Dict(name => collect(selectdim(a[ntuple(_ -> Colon(), ndims(a))...], 2, 1:ndays)) for (name, a) in store.arrays if ndims(a) >= 2)
end

# Last in this file: turning off the copier can not be undone
if mGV.backend_name(backend) != "CPU"
    # 20 days in blocks of 8: the last block is partial
    with_copier = landonly_output(backend)
    unpacked_with_copier = landonly_output(backend; packing = false)
    @eval mGV.host_copier(::$(typeof(backend))) = nothing
    without_copier = landonly_output(backend)
    unpacked_without_copier = landonly_output(backend; packing = false)

    @testset "$(mGV.backend_name(backend)) output with and without the host copier" begin
        @test !isempty(with_copier)
        @test keys(with_copier) == keys(without_copier)
        for (name, values) in with_copier
            @test isequal(values, without_copier[name])
        end
        # Not all NaN (the snow variables can be, in the Mekong)
        @test any(!isnan, with_copier["soil_moisture"])
        @test any(!isnan, with_copier["total_runoff"])
    end

    @testset "$(mGV.backend_name(backend)) unpacked output with and without the host copier" begin
        @test keys(unpacked_with_copier) == keys(with_copier)
        packings = Dict(meta.output_name => meta.packing for meta in values(mGV.output_variables()))
        for (name, values) in unpacked_with_copier
            @test eltype(values) == Float32
            @test isequal(values, unpacked_without_copier[name])
            # The packed output is the unpacked output, quantised (where it is in range)
            p = packings[name]
            isnothing(p) && continue
            packed = with_copier[name]
            @test isequal(isnan.(packed), isnan.(values))
            ok = .!isnan.(values) .& (abs.((values .- p.offset) .* p.scale) .< 32767)
            tolerance = 0.5f0 / p.scale .+ 4 .* eps.(abs(p.offset) .+ abs.(values[ok]))
            @test all(abs.(packed[ok] .- values[ok]) .<= tolerance)
        end
        @test unpacked_with_copier["total_runoff"] != with_copier["total_runoff"]  # not quantised
    end
end
