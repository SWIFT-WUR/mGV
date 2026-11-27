using Test
using mGV
using mGV: NCDatasets
using KernelAbstractions

# low vegetation (overstory = 0) buried by snow does
# not transpire, its leaves do not evaporate, and the water on its leaves falls through to
# the snowpack. Overstory (forests) stays above the snow and is not affected.

const MEKONG_CFG = joinpath(@__DIR__, "../configs/mekong_config.toml")

"""
Write a copy of the Mekong config next to the original (its paths are relative to it),
with the output in a temporary directory, the overstory variable named `overstory` and
the parameter file `param_file` (`nothing` keeps the original). Returns its path.
"""
function mekong_test_config(; overstory = "overstory", param_file = nothing)
    toml = replace(
        read(MEKONG_CFG, String),
        "../output_data/mekong/" => escape_string(mktempdir()),
        "[input.names]" => "[input.names]\noverstory = \"$overstory\"",
    )
    if !isnothing(param_file)
        toml = replace(toml, "../input_data/mekong/vic_mekong_5min_params.nc" => escape_string(param_file))
    end
    path, io = mktemp(dirname(MEKONG_CFG))
    write(io, toml)
    close(io)
    return path
end

@testset "buried_fraction" begin
    bf = mGV.buried_fraction
    # Low vegetation: buried by the snow cover
    @test bf(0.0f0, 0.0f0) == 0
    @test bf(1.0f0, 0.0f0) == 1
    @test bf(0.4f0, 0.0f0) == 0.4f0
    # Overstory: never buried
    @test bf(1.0f0, 1.0f0) == 0
    # No snow data, or cover out of range
    @test bf(NaN32, 0.0f0) == 0
    @test bf(1.5f0, 0.0f0) == 1
    @test bf(-0.2f0, 0.0f0) == 0
end

@testset "Leaf water storage under snow (kernel)" begin
    # Three tiles in one band, all with the same leaves and precipitation:
    #   1: grass without snow, 2: grass under snow, 3: forest under snow
    c = fill(0.8f0, 3)               # canopy coverage
    mx = fill(0.5f0, 3)              # maximum leaf storage [mm per canopy area]
    w0 = fill(0.3f0, 3)              # leaf storage at the start of the step
    p = [2.0f0]                      # band precipitation [mm]
    band = Int32[1, 1, 1]
    overstory = [0.0f0, 0.0f0, 1.0f0]
    snow = [0.0f0, 1.0f0, 1.0f0]

    function run_storage(e)
        w = copy(w0)
        throughfall = zeros(Float32, 3)
        mGV.water_canopy_storage_kernel!(CPU())(
            throughfall, w, mx, e, p, c, band, overstory, snow; ndrange = 3
        )
        return throughfall, w
    end

    # Without leaf evaporation: hand-computed values
    throughfall, w = run_storage(zeros(Float32, 3))
    @test w[1] ≈ 0.5                       # storage fills up
    @test throughfall[1] ≈ 1.84            # (2.3 - 0.5) * 0.8 + 2 * 0.2
    @test w[2] == 0                        # buried: storage empties
    @test throughfall[2] ≈ 2.24            # p + w * c = 2 + 0.3 * 0.8
    @test w[3] == w[1]                     # forest: as without snow
    @test throughfall[3] == throughfall[1]

    # Water balance: what falls through, plus the change in storage, plus the leaf
    # evaporation, is the precipitation (storage and evaporation per canopy area).
    # As in the model: buried leaves (tile 2) do not evaporate, and the evaporation does not
    # exceed the water available on the leaves (w + p = 2.3 here; 2.3 dries them out).
    for x in (0.0f0, 0.4f0, 2.3f0)
        e = [x, 0.0f0, x]
        throughfall, w = run_storage(e)
        for t in 1:3
            @test throughfall[t] + c[t] * (w[t] - w0[t] + e[t]) ≈ p[1] atol = 1.0f-5
        end
    end
end

@testset "Vegetation under snow (Mekong)" begin
    cfg = mekong_test_config()
    model = mGV.Model(cfg; backend = "cpu")
    try
        # Realistic soil moisture and leaf water
        for _ in 1:10
            mGV.update!(model)
        end

        veg = Array(model.tiles.veg)
        nveg = model.tiles.nveg
        overstory = Array(model.vegetation_parameters.overstory)

        @testset "overstory read from the parameter file" begin
            @test length(overstory) == mGV.ntiles(model.tiles)
            @test eltype(overstory) == eltype(model.vegetation_parameters.minimum_resistance)
            # In the Mekong file, the forest classes 1-5 are overstory
            @test all(overstory[veg .<= 5] .== 1)
            @test all(overstory[veg .> 5] .== 0)
        end

        cv = model.canopy_variables
        snow_cover = model.snow_variables.coverage
        cv.water_storage .= 0.5f0 .* cv.maximum_water_storage  # wet leaves
        w0 = copy(cv.water_storage)

        # Run the vegetation updates of one step with a given snow cover. update!
        # would recompute the snow cover (and Mekong has no snow), so call them directly.
        function vegetation_fluxes(cover)
            snow_cover .= cover
            cv.water_storage .= w0
            mGV.update_canopy_evaporation!(model)
            mGV.update_transpiration!(model)
            mGV.update_water_canopy_storage!(model)
            return (
                evaporation = Array(cv.canopy_evaporation),
                transpiration = Array(cv.transpiration),
                layers = Array(cv.transpiration_layers),
                storage = Array(cv.water_storage),
                throughfall = Array(cv.throughfall),
            )
        end

        free = vegetation_fluxes(0.0f0)
        buried = vegetation_fluxes(1.0f0)
        nodata = vegetation_fluxes(NaN32)

        low = (overstory .== 0) .& (veg .< nveg)
        high = overstory .== 1
        bare = veg .== nveg

        @testset "the test is meaningful" begin
            @test any(free.transpiration[low] .> 0)
            @test any(free.evaporation[low] .> 0)
            @test any(high)
        end

        @testset "low vegetation under snow" begin
            @test all(buried.transpiration[low] .== 0)
            @test all(buried.layers[low, :] .== 0)
            @test all(buried.evaporation[low] .== 0)
            @test all(buried.storage[low] .== 0)
            # All precipitation and the water on the leaves fall through
            c = Array(model.vegetation_parameters.canopy_coverage)
            p = Array(model.snow_variables.band_precipitation)[Array(model.tiles.band)]
            @test buried.throughfall[low] ≈ p[low] .+ w0[low] .* c[low]
        end

        @testset "overstory under snow is not affected" begin
            for name in keys(free)
                @test getfield(buried, name)[high, :] == getfield(free, name)[high, :]
            end
        end

        @testset "no snow data counts as no snow" begin
            @test nodata == free
        end

        @testset "bare soil" begin
            @test all(buried.transpiration[bare] .== 0)
            @test all(buried.evaporation[bare] .== 0)
        end

        @testset "a full step with snow stays finite" begin
            # A consistent snow state: Mekong has no snow, so the snow albedo and surface
            # temperature are NaN, and with a snow cover of 1 they would enter the
            # snow-blended net radiation and soil evaporation
            snow_cover .= 1.0f0
            model.snow_variables.albedo .= 0.8f0
            model.snow_variables.surface_temperature .= -1.0f0
            mGV.update!(model)
            @test all(isfinite, model.soil_variables.moisture)
            @test all(isfinite, model.surface_energy_variables.total_evapotranspiration)
            @test all(isfinite, model.surface_energy_variables.surface_temperature)
            @test all(isfinite, model.snow_variables.snow_water_equivalent)
        end
    finally
        mGV.finalize!(model)
        rm(cfg)
    end
end

@testset "Parameter file without overstory" begin
    cfg = mekong_test_config(overstory = "no_overstory")
    try
        model = @test_logs (:warn, r"No 'no_overstory'") match_mode = :any mGV.Model(cfg; backend = "cpu")
        # All vegetation is treated as low: it can be buried by snow
        @test all(Array(model.vegetation_parameters.overstory) .== 0)
        mGV.finalize!(model)
    finally
        rm(cfg)
    end
end
