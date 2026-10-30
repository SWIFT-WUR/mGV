using Test

import BasicModelInterface as BMI
using mGV
using TOML

const COORDS = ["lat", "lon", "time"]

"""
Write a copy of the Mekong config to `dir` with Zarr output in `dir`, and `variables` in the
`[output]` section (left out when `nothing`). Other settings can be overridden with `kwargs`.
"""
function write_test_config(dir, variables; kwargs...)
    configs_dir = joinpath(@__DIR__, "../configs")
    cfg = TOML.parsefile(joinpath(configs_dir, "mekong_config.toml"))
    for (key, path) in cfg["input"]["paths"]
        cfg["input"]["paths"][key] = abspath(joinpath(configs_dir, path))
    end
    cfg["output"]["dir"] = dir
    cfg["output"]["format"] = "zarr"
    isnothing(variables) || (cfg["output"]["variables"] = variables)
    for (key, value) in kwargs
        cfg[string(key)] = value
    end
    config_file = joinpath(dir, "config.toml")
    open(io -> TOML.print(io, cfg), config_file, "w")
    return config_file
end

"""Run the model for a few time steps, and return the variables in the output store (if any)."""
function written_variables(variables; kwargs...)
    dir = mktempdir()
    model = BMI.initialize(mGV.Model, write_test_config(dir, variables; kwargs...))
    for _ in 1:2
        BMI.update(model)
    end
    BMI.finalize(model)

    stores = filter(endswith(".zarr"), readdir(dir))
    isempty(stores) && return nothing
    store = joinpath(dir, only(stores))
    return filter(name -> isdir(joinpath(store, name)) && !(name in COORDS), readdir(store))
end

all_names = [var.name for var in mGV.OUTPUT_VARIABLES]

@testset "Output variable selection" begin
    @testset "All variables are written by default" begin
        @test sort(written_variables(nothing)) == sort(all_names)
    end

    @testset "No output file without variables" begin
        @test isnothing(written_variables(String[]))
    end

    @testset "Only the selected variables are written" begin
        @test sort(written_variables(["discharge", "soil_moisture", "swe"])) ==
            ["discharge", "soil_moisture", "swe"]
    end

    @testset "Variables of disabled modules are skipped" begin
        selected = ["swe", "total_runoff"]
        @test_logs (:warn, r"Not writing swe") match_mode = :any begin
            @test written_variables(selected; enable_snow = false) == ["total_runoff"]
        end
    end

    @testset "Unknown variable names are rejected when loading the config" begin
        config_file = write_test_config(mktempdir(), ["not_a_variable"])
        @test_throws r"Unknown output variable 'not_a_variable'" mGV.Config.load_config(config_file)
    end
end
