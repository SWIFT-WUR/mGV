using Test

import BasicModelInterface as BMI
using mGV
using TOML
using Zarr: zopen

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

all_names = [meta.output_name for meta in values(mGV.output_variables())]

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

    @testset "Output is gridded, with NaN outside the active cells" begin
        dir = mktempdir()
        config_file = write_test_config(dir, ["soil_moisture", "total_runoff"])
        model = BMI.initialize(mGV.Model, config_file)
        BMI.update(model)
        (; domain) = model
        moisture = Array(model.soil_variables.moisture)
        BMI.finalize(model)

        store = zopen(joinpath(dir, only(filter(endswith(".zarr"), readdir(dir)))))
        runoff = store["total_runoff"][:, :, 1]
        @test size(runoff) == domain.grid_size == (36, 36)
        active = falses(domain.grid_size)
        active[domain.cells] .= true
        @test all(isfinite, runoff[active])
        @test all(isnan, runoff[.!active])
        # Layered output: the active cells of each layer hold the model values
        layers = store["soil_moisture"][:, :, 1, :]
        @test mGV.to_cells(domain, layers) == moisture
        @test all(isnan, reshape(layers, :, size(layers, 3))[vec(.!active), :])
    end

    @testset "Unknown variable names are rejected when loading the config" begin
        config_file = write_test_config(mktempdir(), ["not_a_variable"])
        @test_throws r"Unknown output variable 'not_a_variable'" mGV.Config.load_config(config_file)
    end

    @testset "Postprocessed variables are available through BMI without writing them" begin
        name = "land_surface__evaporation~potential~cell-average"
        dir = mktempdir()
        config_file = write_test_config(dir, String[]; API = Dict("variables" => [name]))
        model = BMI.initialize(mGV.Model, config_file)
        BMI.update(model)
        @test name in BMI.get_output_var_names(model)
        dest = fill(NaN32, BMI.get_grid_size(model, BMI.get_var_grid(model, name)))
        BMI.get_value(model, name, dest)
        @test any(isfinite, dest)
        @test !any(endswith(".zarr"), readdir(dir))
    end
end

@testset "Domain" begin
    domain = mGV.Domain((3, 2), [2, 3, 5])
    @test mGV.ncells(domain) == 3
    grid = reshape(collect(1.0:12.0), 3, 2, 2)
    cells = mGV.to_cells(domain, grid)
    @test cells == [2.0 8.0; 3.0 9.0; 5.0 11.0]
    @test mGV.to_cells(domain, grid[:, :, 1]) == [2.0, 3.0, 5.0]
    out = fill(NaN, 3, 2, 2)
    mGV.to_grid!(out, domain, cells)
    @test isequal(out, ifelse.(reshape([0, 1, 1, 0, 1, 0], 3, 2) .== 1, grid, NaN))
    @test_throws "does not match the domain grid" mGV.to_cells(domain, zeros(2, 3))
end
