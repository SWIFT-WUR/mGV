
using Test 

import BasicModelInterface as BMI
using Statistics: mean
using mGV
using Dates

const TYPES = Dict(
    "Float16" => Float16,
    "Float32" => Float32,
    "Float64" => Float64,
    "UInt8" => UInt8,
    "UInt16" => UInt16,
    "UInt32" => UInt32,
    "Int16" => Int16,
    "Int32" => Int32,
    "Int64" => Int64,
)
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
    return zeros(TYPES[type], size)
end

function get_value_mean(var_name::String, m::mGV.Model)
    grid = BMI.get_var_grid(m,var_name)
    var_type = BMI.get_var_type(m, var_name)
    dest = _zeros(var_type, BMI.get_grid_size(m, grid))
    BMI.get_value(m, var_name, dest)
    return mean(dest)
end

@testset "Simple BMI test" begin
    tomlpath = "../configs/mekong_config.toml" 
    model = BMI.initialize(mGV.Model, tomlpath)

    for i in 1:10
        BMI.update(model)
    end

    sml3 = "soil__moisture_layer_3"

    @test sml3 in BMI.get_input_var_names(model)
    @test sml3 in BMI.get_output_var_names(model)

    sm_grid = BMI.get_var_grid(model, sml3)
    sm_size = BMI.get_grid_size(model, sm_grid)
    sm_type = BMI.get_var_type(model, sml3)
    dest = _zeros(sm_type, sm_size)

    # soil moisture should have values >=0
    BMI.get_value(model, sml3, dest)
    bad = findall(x -> !(x >= 0), dest)
    @test all(x -> isnan(x) || x>=0, dest) # on CPU NaN might be present (ocean)
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
    model = nothing
    GC.gc()
    sleep(3)
end



@testset "Full BMI tests" begin
    
    tomlpath = "../configs/mekong_config.toml" 
    model = BMI.initialize(mGV.Model, tomlpath)
    config = mGV.load_config(tomlpath)
    clock = mGV.Clock(config)

    # initialization and time functions
    @test BMI.get_time_units(model) == "seconds since 1970-01-01T00:00+00:00"
    @test BMI.get_time_step(model) == Float64(clock.dt.value)
    @test BMI.get_start_time(model) == datetime2unix(clock.time)
    @test BMI.get_current_time(model) == datetime2unix(clock.time)
    @test BMI.get_end_time(model) == datetime2unix(DateTime(config.end_year, 12, 31))

    # "model information functions" begin
    @test BMI.get_component_name(model) == "mGV"
    @test BMI.get_grid_shape(model,BMI.get_var_grid(model,"land_vegetation_canopy__evaporation")) == [14,36,36]
    n_inputs = count(meta -> meta.input, values(mGV.standard_name_map))
    n_outputs = count(meta -> meta.output, values(mGV.standard_name_map))
    @test BMI.get_input_item_count(model) >= n_inputs
    @test BMI.get_output_item_count(model) >= n_outputs
    @test length(BMI.get_grid_z(model, BMI.get_var_grid(model,"land_vegetation_canopy__evaporation"))) == model.config.nveg

    out_vars = BMI.get_output_var_names(model)
    in_vars = BMI.get_input_var_names(model)
    for i in 1:10
        BMI.update(model)
        if i == 1
            @test BMI.get_current_time(model) == datetime2unix(clock.time) 
        end
    end

    for name in out_vars
        grid = BMI.get_var_grid(model, name)
        size = BMI.get_grid_size(model, grid)
        type = BMI.get_var_type(model, name)
        dest = fill(MaxType[type], size)
        BMI.get_value(model, name, dest)
        @test all(dest .!= MaxType[type])
    end

    for name in in_vars
        grid = BMI.get_var_grid(model, name)
        size = BMI.get_grid_size(model, grid)
        type = BMI.get_var_type(model, name)
        dest = fill(MaxType[type], size)
        BMI.set_value(model, name, dest)
        @test all(dest .== MaxType[type])
    end 
    println(BMI.get_var_units(model,"land_surface__radiation~net~upward_energy_flux"))

    time = BMI.get_current_time(model) + (200*BMI.get_time_step(model))
    BMI.update_until(model, time)
    @test BMI.get_current_time(model) == time - BMI.get_time_step(model)
end