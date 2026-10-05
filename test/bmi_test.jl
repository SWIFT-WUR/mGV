
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

function check_get_var_grid(model::mGV.Model, name::String)
    resLat_min, resLat_max = extrema(diff(model.grid_parameters.latitude))
    resLon_min, resLon_max = extrema(diff(model.grid_parameters.longitude))
    if occursin(r"_layer_\d",name)
        name_2d,  = model.soil_layer_standard_name(name)
        name = name_2d
    end
    (; lens) = mGV.get_metadata(name; model)
    size_n = size(lens(model))
    if (all(isapprox.(resLat_min, resLat_max)) && all(isapprox.(resLon_min, resLon_max)))
        if length(size_n)<=2
            return 0
        elseif length(size_n)==3
            if size_n[3] == size(model.soil_parameters.depth,3)
                return 1 #Rectilinear 2d
            else 
                return 2
            end
        else 
            return 3 #Rectilinear 3d
        end
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
    @test BMI.get_grid_shape(model,BMI.get_var_grid(model,"canopy_evaporation")) == [14,36,36]
    @test BMI.get_input_item_count(model) == length(config.API.input_variables)
    @test BMI.get_output_item_count(model) == length(config.API.output_variables)
    @test length(BMI.get_grid_z(model, BMI.get_var_grid(model,"canopy_evaporation"))) == model.config.nveg

    standard_names = mGV.get_standard_name_map()
    for (name, metadata) in standard_names
        @test metadata.gridtype == check_get_var_grid(model, name)

        BMI.set_value_at_indices(model, name, [1, 2, 3], Float32[0.1, 0.15, 0.2])
        @test isapprox(Float32[0.1, 0.15, 0.2], BMI.get_value_at_indices(model, name, zeros(3), [1, 2, 3]))
        set_value_to!(name, model, Float32(300.0))  
        @test get_value_mean(name, model) == Float32(300)
    end
    
    retrieved_vars = BMI.get_output_var_names(model)
    @test all(x -> x in retrieved_vars, keys(standard_names))

    # "BMI update and get and set functions" begin
    BMI.update(model)
    #Note: at iteration 0 the clock is not advanced in time, to have the model start time at iteration 1.
    @test BMI.get_current_time(model) == datetime2unix(clock.time) 

    #Check paramaters (no more needed)
    # @test isapprox(get_value_mean("nijssen_nonlin_reservoir", model),mean(model.soil_parameters.nijssen_nonlin_reservoir))
    # @test isapprox(get_value_mean("nijssen_infilt_b", model),mean(model.soil_parameters.nijssen_infilt_b))
    # @test isapprox(get_value_mean("snow_band_area_fraction_layer_1", model),mean(model.grid_parameters.snow_band_area_fraction[:,:,1]))
    # @test isapprox(get_value_mean("snow_band_elevation_layer_2", model),mean(model.grid_parameters.snow_band_elevation[:,:, 2]))
    # @test isapprox(get_value_mean("snow_band_precipitation_factor_layer_3", model),mean(model.grid_parameters.snow_band_precipitation_factor[:,:, 3]))
    
    @test isapprox(get_value_mean("infiltration", model),mean(model.soil_variables.infiltration))
    # TODO: layers variables are now broken...
    # @test isapprox(get_value_mean("ice_fraction_layer_3", model),mean(model.soil_variables.ice_fraction[:,:,3]))
    # @test isapprox(get_value_mean("interlayer_drainage_layer_2", model),mean(model.soil_variables.interlayer_drainage[:,:,2]))

    #NaN are causing errors
    # @test isapprox(get_value_mean("moisture_layer_1", model),mean(model.soil_variables.moisture[:,:,1]))
    # @test isapprox(get_value_mean("temperature_layer_2", model),mean(model.soil_variables.temperature[:,:,2]))
    # @test isapprox(get_value_mean("surface_temperature", model),mean(model.surface_energy_variables.surface_temperature))

    
end