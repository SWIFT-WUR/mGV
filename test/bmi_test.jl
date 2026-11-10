using Test

import BasicModelInterface as BMI
using Statistics: mean
using mGV
using Dates

MEKONG_CFG = joinpath(@__DIR__, "../configs/mekong_config.toml")


const MaxType = Dict(
    "Float16" => floatmax(Float16),
    "Float32" => floatmax(Float32),
    "Float64" => floatmax(Float64),
    "UInt8" => typemax(UInt8),
    "UInt16" => typemax(UInt16),
    "UInt32" => typemax(UInt32),
    "Int16" => typemax(Int16),
    "Int32" => typemax(Int32),
    "Int64" => typemax(Int64),
)
function _zeros(type::String, size::Int)
    return zeros(mGV.TYPES[type], size)
end

function get_var_grid_size_type(m::mGV.Model, name::String)
    g = BMI.get_var_grid(m, name)
    s = BMI.get_grid_size(m, g)
    t = BMI.get_var_type(m, name)
    return g, s, t
end

function get_value_mean(var_name::String, m::mGV.Model)
    _, size, var_type = get_var_grid_size_type(m, var_name)
    dest = _zeros(var_type, size)
    BMI.get_value(m, var_name, dest)
    return mean(dest)
end

function get_initialized_model(conf_path::String)
    model = BMI.initialize(mGV.Model, conf_path)
    return model
end

@testset "Simple BMI test" begin
    model = get_initialized_model(MEKONG_CFG)

    for i in 1:10
        BMI.update(model)
    end

    sml3 = "soil__moisture_layer_3"

    @test sml3 in BMI.get_input_var_names(model)
    @test sml3 in BMI.get_output_var_names(model)

    sm_grid, sm_size, sm_type = get_var_grid_size_type(model, sml3)
    dest = _zeros(sm_type, sm_size)

    # soil moisture should have values >=0
    BMI.get_value(model, sml3, dest)
    bad = findall(x -> !(x >= 0), dest)
    @test all(x -> isnan(x) || x >= 0, dest) # on CPU NaN might be present (ocean)
    @test !all(dest .== 0.0)  # not all values can be 0.0

    # set the soil moisture in layer 3 to 0:
    BMI.set_value(model, sml3, _zeros(sm_type, sm_size))
    @test !all(dest .== 0.0) # make sure dest didn't change
    dest = _zeros(sm_type, sm_size)
    BMI.get_value(model, sml3, dest)
    @test all(dest .== 0.0) # make sure all values in the model are now 0.0

    BMI.update(model)

    # After updating moisture should be back in layer 3
    dest = _zeros(sm_type, sm_size)
    BMI.get_value(model, sml3, dest)
    @test !all(dest .== 0.0)

    BMI.finalize(model)
    @test isnothing(model.writer)
end

@testset "Full BMI tests" begin

    model = get_initialized_model(MEKONG_CFG)

    # initialization and time functions
    @test BMI.get_time_units(model) == "seconds since 1970-01-01T00:00+00:00"
    @test BMI.get_time_step(model) == Float64(86400)
    @test BMI.get_start_time(model) == Float64(2.839968e8)
    @test BMI.get_current_time(model) == Float64(2.839968e8)
    @test BMI.get_end_time(model) == Float64(3.471552e8)

    # "model information functions" begin
    @test BMI.get_component_name(model) == "mGV"
    # 894 active (land) cells of the 36 x 36 grid
    grid = BMI.get_var_grid(model, "land_surface__temperature")
    @test BMI.get_grid_type(model, grid) == "unstructured"
    @test BMI.get_grid_rank(model, grid) == 2  # x and y
    @test BMI.get_grid_node_count(model, grid) == BMI.get_grid_size(model, grid) == 894
    @test BMI.get_grid_edge_count(model, grid) == 0
    @test BMI.get_grid_face_count(model, grid) == 0
    @test BMI.get_grid_edge_nodes(model, grid, Int[]) == Int[]
    @test BMI.get_grid_nodes_per_face(model, grid, Int[]) == Int[]
    @test BMI.get_var_location(model, "land_surface__temperature") == "node"
    @test BMI.get_var_location(model, "soil__moisture_layer_1") == "node"
    x = BMI.get_grid_x(model, grid)
    y = BMI.get_grid_y(model, grid)
    @test length(x) == length(y) == 894
    # Coordinates of the active cells, in grid order (longitude varies fastest)
    lon, lat = model.grid_parameters.longitude, model.grid_parameters.latitude
    @test all(in(lon), x) && all(in(lat), y)
    @test issorted(collect(zip(y, x)))
    n_inputs = count(meta -> meta.input, values(mGV.standard_name_map))
    n_outputs = count(meta -> meta.output, values(mGV.standard_name_map))
    @test BMI.get_input_item_count(model) >= n_inputs
    @test BMI.get_output_item_count(model) >= n_outputs
    # Tile grid: a node per snow band and vegetation type (z) with area in a cell. The Mekong
    # cells have a single snow band each, and on average 4 of the 14 vegetation types.
    tile_grid = BMI.get_var_grid(model, "land_vegetation_canopy__evaporation")
    tile_cell = Array(model.tiles.cell)
    @test BMI.get_grid_rank(model, tile_grid) == 3
    @test BMI.get_grid_node_count(model, tile_grid) == 3580
    @test issorted(tile_cell) && unique(tile_cell) == 1:894
    @test BMI.get_grid_z(model, tile_grid) == Array(model.tiles.veg)
    @test BMI.get_grid_x(model, tile_grid) == x[tile_cell]
    @test BMI.get_grid_y(model, tile_grid) == y[tile_cell]
    @test length(BMI.get_value_ptr(model, "land_vegetation_canopy__evaporation")) == 3580

    out_vars = BMI.get_output_var_names(model)
    in_vars = BMI.get_input_var_names(model)
    for i in 1:10
        BMI.update(model)
        if i == 1
            @test BMI.get_current_time(model) == Float64(2.840832e8)
        end
    end

    for name in out_vars
        grid, size, type = get_var_grid_size_type(model, name)
        dest = fill(MaxType[type], size)
        BMI.get_value(model, name, dest)
        @test all(dest .!= MaxType[type])

        dest = fill(MaxType[type], 3)
        BMI.get_value_at_indices(model, name, dest, [1, 100, size])
        @test all(dest .!= MaxType[type])
    end

    for name in in_vars
        grid, size, type = get_var_grid_size_type(model, name)
        dest = fill(MaxType[type], 3)
        BMI.set_value_at_indices(model, name, [1, 100, size], dest)
        BMI.get_value_at_indices(model, name, dest, [1, 100, size])
        @test all(dest .== MaxType[type])

        dest = fill(MaxType[type], size)
        BMI.set_value(model, name, dest)
        BMI.get_value(model, name, dest)
        @test all(dest .== MaxType[type])
    end

    time = BMI.get_current_time(model) + (200 * BMI.get_time_step(model))
    BMI.update_until(model, time)
    @test BMI.get_current_time(model) == time
    @test_logs (:warn, r"already at time") BMI.update_until(model, time)
    @test_throws r"not a multiple" BMI.update_until(model, time + 1.0)

    BMI.finalize(model)
end


@testset "BMI units" begin
    model = get_initialized_model(MEKONG_CFG)
    BMI.update(model)

    # Values are exchanged in the model's native units
    @test BMI.get_var_units(model, "land_subsurface__runoff") == "mm day-1"
    @test BMI.get_var_units(model, "land_surface__temperature") == "degC"
    @test BMI.get_var_units(model, "soil__moisture_layer_1") == "mm"
    @test BMI.get_var_units(model, "channel_water__volume_flow_rate") == "m3 s-1"
    @test BMI.get_var_units(model, "soil__ice_fraction_layer_1") == "-"

    name = "land_subsurface__runoff"
    dest = zeros(Float32, length(BMI.get_value_ptr(model, name)))
    BMI.get_value(model, name, dest)
    @test isequal(dest, Array(BMI.get_value_ptr(model, name)))

    BMI.finalize(model)
end


@testset "External groundwater routing" begin
    # Paths in the config are relative to the config file, so write the modified
    #  config next to the original one. Use a separate output directory to avoid
    #  clashing with output files still opened by other models.
    toml = replace(
        read(MEKONG_CFG, String),
        "external_groundwater_routing = false" => "external_groundwater_routing = true",
        "../output_data/mekong/" => escape_string(mktempdir()),
    )
    gw_tomlpath, io = mktemp(dirname(MEKONG_CFG))
    write(io, toml)
    close(io)

    try
        model = BMI.initialize(mGV.Model, gw_tomlpath)
        @test model.config.API.external_groundwater_routing

        for i in 1:10
            BMI.update(model)
        end

        get_var(name) = Array(BMI.get_value_ptr(model, name))
        surface = get_var("land_surface__runoff")
        subsurface = get_var("land_subsurface__runoff")
        total = get_var("land__runoff~total")

        # subsurface runoff is computed, but not passed on to the routing
        @test any(x -> !isnan(x) && x > 0, subsurface)
        @test isequal(total, surface)

        BMI.finalize(model)
    finally
        rm(gw_tomlpath)
    end
end
