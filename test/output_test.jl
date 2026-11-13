using Test

import BasicModelInterface as BMI
using mGV
using TOML
using Zarr: zopen
using NCDatasets: NCDataset
using ZarrScaleCast: ScaleOffsetCodec, CastValueCodec
import Zarr

const COORDS = ["lat", "lon", "time"]

"""
Write a copy of the Mekong config to `dir` with Zarr output in `dir`, and `variables` in the
`[output]` section (left out when `nothing`). Other settings can be overridden with `kwargs`,
and settings of the `[output]` section with `output`.
"""
function write_test_config(dir, variables; output = Dict(), kwargs...)
    configs_dir = joinpath(@__DIR__, "../configs")
    cfg = TOML.parsefile(joinpath(configs_dir, "mekong_config.toml"))
    for (key, path) in cfg["input"]["paths"]
        cfg["input"]["paths"][key] = abspath(joinpath(configs_dir, path))
    end
    cfg["output"]["dir"] = dir
    cfg["output"]["format"] = "zarr"
    isnothing(variables) || (cfg["output"]["variables"] = variables)
    merge!(cfg["output"], output)
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
        config_file = write_test_config(dir, ["soil_moisture", "total_runoff"]; output = Dict("packing" => false))
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

    @testset "Output is written in blocks of timesteps ($format, packing $packing)" for format in ["zarr", "zarr_landonly", "netcdf"],
            packing in [false, true]
        # 7 timesteps in blocks of 3: two full blocks, and a partial one written by finalize
        dir = mktempdir()
        variables = ["soil_moisture", "total_runoff"]
        config_file = write_test_config(dir, variables; output = Dict("format" => format, "write_interval" => 3, "packing" => packing))
        model = BMI.initialize(mGV.Model, config_file)
        metas = Dict(meta.output_name => meta for meta in model.selected_outputs)
        expected = Dict(name => [] for name in variables)
        for _ in 1:7
            BMI.update(model)
            for name in variables
                push!(expected[name], Array(metas[name].lens(model)))
            end
        end
        (; domain) = model
        BMI.finalize(model)

        path = joinpath(dir, only(filter(startswith("outputfile"), readdir(dir))))
        if format == "netcdf"
            ds = NCDataset(path)
            # The stored values, unpacked here to test the CF attributes
            function read_nc(name, idx...)
                var = ds[name]
                raw = var.var[idx...]
                packing || return raw
                @test eltype(raw) == Int16
                sf, ao = var.attrib["scale_factor"], var.attrib["add_offset"]
                return ifelse.(raw .== typemin(Int16), NaN32, Float32.(raw .* sf .+ ao))
            end
            moisture = [mGV.to_cells(domain, read_nc("soil_moisture", :, :, t, :)) for t in 1:7]
            runoff = [mGV.to_cells(domain, read_nc("total_runoff", :, :, t)) for t in 1:7]
            close(ds)
        else
            store = zopen(path)
            @test eltype(store["total_runoff"]) == Float32  # also when packed
            # Packed: stored as Int16 with the scale_offset and cast_value codecs
            codecs = store["total_runoff"].metadata.pipeline.array_array
            @test any(c -> c isa ScaleOffsetCodec, codecs) == packing
            @test any(c -> c isa CastValueCodec, codecs) == packing
            if format == "zarr"
                @test store["total_runoff"].metadata.chunks == (36, 36, 3)
                moisture = [mGV.to_cells(domain, store["soil_moisture"][:, :, t, :]) for t in 1:7]
                runoff = [mGV.to_cells(domain, store["total_runoff"][:, :, t]) for t in 1:7]
                @test all(isnan, store["total_runoff"][:, :, 8:end])  # not simulated
            else
                # Land-only: (cell, time) arrays, with the grid indices of each cell
                @test size(store["total_runoff"]) == (mGV.ncells(domain), 365)
                nx = domain.grid_size[1]
                @test store["lon_index"][:] .+ 1 .+ store["lat_index"][:] .* nx == domain.cells
                moisture = [store["soil_moisture"][:, t, :] for t in 1:7]
                runoff = [store["total_runoff"][:, t] for t in 1:7]
                @test all(isnan, store["total_runoff"][:, 8:end])
            end
        end
        if !packing
            @test moisture == expected["soil_moisture"]
            @test runoff == expected["total_runoff"]
        else
            # At most half a step off, plus the rounding of the decoding in Float32
            for (name, values) in (("soil_moisture", moisture), ("total_runoff", runoff))
                p = metas[name].packing
                for (v, e) in zip(values, expected[name])
                    @test isequal(isnan.(v), isnan.(e))
                    ok = .!isnan.(e)
                    tolerance = 0.5f0 / p.scale .+ 4 .* eps.(abs(p.offset) .+ abs.(e[ok]))
                    @test all(abs.(v[ok] .- e[ok]) .<= tolerance)
                end
            end
        end
    end

    @testset "The writer continues in the store of the next year ($format)" for format in ["zarr", "netcdf"]
        dir = mktempdir()
        config_file = write_test_config(dir, ["total_runoff"]; output = Dict("format" => format, "write_interval" => 7, "packing" => false))
        model = BMI.initialize(mGV.Model, config_file)
        lens = only(model.selected_outputs).lens
        runoff = []
        for _ in 1:367  # 1979 has 365 days, which is not a multiple of 7
            BMI.update(model)
            push!(runoff, Array(lens(model)))
        end
        (; domain) = model
        BMI.finalize(model)
        @test isnothing(model.writer)

        extension = format == "netcdf" ? "nc" : "zarr"
        paths = [joinpath(dir, "outputfile_mekong_$year.$extension") for year in (1979, 1980)]
        datasets = format == "netcdf" ? [NCDataset(p) for p in paths] : [zopen(p) for p in paths]
        read_step(ds, t) = mGV.to_cells(domain, format == "netcdf" ? ds["total_runoff"].var[:, :, t] : ds["total_runoff"][:, :, t])
        @test size(datasets[1]["total_runoff"], 3) == 365
        @test size(datasets[2]["total_runoff"], 3) == 366
        @test read_step(datasets[1], 365) == runoff[365]
        @test read_step(datasets[2], 1) == runoff[366]
        @test read_step(datasets[2], 2) == runoff[367]
        format == "netcdf" && foreach(close, datasets)
    end

    @testset "Invalid output settings are rejected when loading the config" begin
        config_file = write_test_config(mktempdir(), nothing; output = Dict("format" => "grib"))
        @test_throws r"Unknown output format 'grib'" mGV.Config.load_config(config_file)
        config_file = write_test_config(mktempdir(), nothing; output = Dict("write_interval" => 0))
        @test_throws r"write_interval must be at least 1" mGV.Config.load_config(config_file)
        config_file = write_test_config(mktempdir(), nothing; output = Dict("packing" => "int16"))
        @test_throws r"packing must be true or false, got \"int16\"" mGV.Config.load_config(config_file)
        @test mGV.Config.load_config(write_test_config(mktempdir(), nothing)).output.packing  # the default
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

@testset "Int16 packing" begin
    p = mGV.Packing(100, 3)
    lo, hi = 3 - 327.67f0, 3 + 327.67f0
    x = Float32[
        randn(1000) .* 100; 3; 3.005; 3.015; -3.005; 2.995; 0; -0.0;  # including ties
        lo; hi; lo - 0.004; hi + 0.004; lo - 0.006; hi + 0.006; -1.0f4; 1.0f4; -Inf; Inf; NaN;
    ]
    packed = mGV.pack_value.(x, Ref(p))
    values, clipped = first.(packed), last.(packed)
    @test eltype(values) == Int16

    @testset "Same values as the Zarr codecs" begin
        cast = CastValueCodec(
            Int16; rounding = RoundNearest, out_of_range = :clamp,
            encode_map = [NaN => typemin(Int16)], decode_map = [typemin(Int16) => NaN],
        )
        codec = Zarr.Codecs.V3Codecs.codec_encode
        expected = codec(cast, codec(ScaleOffsetCodec(offset = p.offset, scale = p.scale), x))
        # Except below the range: -32768 would read as NaN
        expected[expected .== typemin(Int16) .&& .!isnan.(x)] .= -32767
        @test values == expected
    end

    @testset "Clipping" begin
        @test clipped == (.!isnan.(x) .& .!(lo - 0.005f0 .<= x .<= hi + 0.005f0))
        @test values[x .== Inf] == [32767]
        @test values[x .== -Inf] == [-32767]
        @test values[isnan.(x)] == [typemin(Int16)]
    end

    @testset "Kernel" begin
        src = reshape(x[1:1000], 500, 2)
        dst = zeros(Int16, 1200)
        flags = zeros(Int32, 3)
        mGV.pack_output!(mGV.CPU(), dst, 100, 600, src, p, flags, 2)
        mGV.KernelAbstractions.synchronize(mGV.CPU())
        @test dst[101:600] == values[1:500]
        @test dst[701:1200] == values[501:1000]
        @test flags == [0, any(clipped[1:1000]), 0]  # random values can be out of range
        fill!(flags, 0)
        mGV.pack_output!(mGV.CPU(), dst, 0, 1, Float32[1.0f0], p, flags, 2)
        mGV.KernelAbstractions.synchronize(mGV.CPU())
        @test flags == [0, 0, 0]
        mGV.pack_output!(mGV.CPU(), dst, 0, 1, Float32[1.0f6], p, flags, 2)
        mGV.KernelAbstractions.synchronize(mGV.CPU())
        @test flags == [0, 1, 0]
        mGV.pack_output!(mGV.CPU(), Float32[0], 0, 1, Float32[1.0f6], nothing, flags, 3)
        mGV.KernelAbstractions.synchronize(mGV.CPU())
        @test flags == [0, 1, 0]  # not packed
    end
end

@testset "Domain" begin
    domain = mGV.Domain((3, 2), [2, 3, 5], [100.0, 101.0, 102.0], [-10.0, 20.0])
    @test mGV.ncells(domain) == 3
    grid = reshape(collect(1.0:12.0), 3, 2, 2)
    cells = mGV.to_cells(domain, grid)
    @test cells == [2.0 8.0; 3.0 9.0; 5.0 11.0]
    @test mGV.to_cells(domain, grid[:, :, 1]) == [2.0, 3.0, 5.0]
    out = fill(NaN, 3, 2, 2)
    mGV.to_grid!(out, domain, cells)
    @test isequal(out, ifelse.(reshape([0, 1, 1, 0, 1, 0], 3, 2) .== 1, grid, NaN))
    @test_throws "does not match the domain grid" mGV.to_cells(domain, zeros(2, 3))
    @test mGV.cell_coordinates(domain) == ([101.0, 102.0, 101.0], [-10.0, -10.0, 20.0])
end
