
using Test 

import BasicModelInterface as BMI
using Statistics: mean
using mGV
using Dates

const TYPES = Dict(
    "Float16" => Float16,
    "Float32" => Float32,
    "Float64" => Float64,
    "Int16" => Int16,
    "Int32" => Int32,
    "Int64" => Int64,
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

function set_value_to!(var_name::String, m::mGV.Model, n::Number)
    grid = BMI.get_var_grid(m,var_name)
    BMI.set_value(m, var_name, fill(n, BMI.get_grid_size(m, grid)))
    return nothing
end

function test_set_value_at_indeces(var_name::String, m::mGV.Model)
    BMI.set_value_at_indices(model, var_name, [1, 2, 3], [0.1, 0.15, 0.2])
    if [0.1, 0.15, 0.2] == BMI.get_value_at_indices(model, var_name, zeros(3), [1, 2, 3])
        return true
    else 
        return false
    end
end



@testset "BMI tests" begin
    
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
    @test BMI.get_input_item_count(model) == 0
    @test BMI.get_output_item_count(model) == 13
    to_check = [
        "infiltration",
        "ice_fraction",
        "interlayer_drainage"
    ]
    # retrieved_vars = BMI.get_input_var_names(model)
    # @test all(x -> x in retrieved_vars, to_check)
    retrieved_vars = BMI.get_output_var_names(model)
    @test all(x -> x in retrieved_vars, to_check)

    #"variable information functions" begin
    @test BMI.get_var_grid(model, "") == 0
    # @test BMI.get_var_type(model, "reservoir_water__incoming_volume_flow_rate") ==
    #     "Float64"
    # @test BMI.get_var_units(model, "river_water__volume_flow_rate") == "m3 s-1"
    # @test BMI.get_var_itemsize(model, "subsurface_water__volume_flow_rate") ==
    #     sizeof(Float64)
    # @test BMI.get_var_nbytes(model, "river_water__instantaneous_volume_flow_rate") ==
    #     length(model.routing.river_flow.variables.q) * sizeof(Float64)
    # @test BMI.get_var_location(model, "river_water__volume_flow_rate") == "node"



    # "BMI update and get and set functions" begin
    BMI.update(model)
    #Note: at iteration 0 the clock is not advanced in time,
    #to have the model start time at iteration 1.
    @test BMI.get_current_time(model) == datetime2unix(clock.time) 

    #Check paramaters (no more needed)
    # @test isapprox(get_value_mean("nijssen_nonlin_reservoir", model),mean(model.soil_parameters.nijssen_nonlin_reservoir))
    # @test isapprox(get_value_mean("nijssen_infilt_b", model),mean(model.soil_parameters.nijssen_infilt_b))
    # @test isapprox(get_value_mean("snow_band_area_fraction_layer_1", model),mean(model.grid_parameters.snow_band_area_fraction[:,:,1]))
    # @test isapprox(get_value_mean("snow_band_elevation_layer_2", model),mean(model.grid_parameters.snow_band_elevation[:,:, 2]))
    # @test isapprox(get_value_mean("snow_band_precipitation_factor_layer_3", model),mean(model.grid_parameters.snow_band_precipitation_factor[:,:, 3]))
    
    @test isapprox(get_value_mean("infiltration", model),mean(model.soil_variables.infiltration))
    @test isapprox(get_value_mean("ice_fraction_layer_3", model),mean(model.soil_variables.ice_fraction[:,:,3]))
    @test isapprox(get_value_mean("interlayer_drainage_layer_2", model),mean(model.soil_variables.interlayer_drainage[:,:,2]))

    #NaN are causing errors
    # @test isapprox(get_value_mean("moisture_layer_1", model),mean(model.soil_variables.moisture[:,:,1]))
    # @test isapprox(get_value_mean("temperature_layer_2", model),mean(model.soil_variables.temperature[:,:,2]))
    # @test isapprox(get_value_mean("surface_temperature", model),mean(model.surface_energy_variables.surface_temperature))

    var_name = "infiltration"
    BMI.set_value_at_indices(model, var_name, [1, 2, 3], Float32[0.1, 0.15, 0.2])
    @test isapprox(Float32[0.1, 0.15, 0.2], BMI.get_value_at_indices(model, var_name, zeros(3), [1, 2, 3]))
    set_value_to!(var_name, model, Float32(300.0))  
    @test get_value_mean(var_name, model) == Float32(300)

    var_name = "ice_fraction_layer_3"
    BMI.set_value_at_indices(model, var_name, [1, 2, 3], Float32[0.1, 0.15, 0.2])
    @test isapprox(Float32[0.1, 0.15, 0.2], BMI.get_value_at_indices(model, var_name, zeros(3), [1, 2, 3]))
    set_value_to!(var_name, model, Float32(300.0))  
    @test get_value_mean(var_name, model) == Float32(300)
end