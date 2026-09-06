module mGV

using NetCDF
using NCDatasets
using Zarr
using ProgressMeter
using Dates
using LinearAlgebra  # Need this?
using TimerOutputs
using MacroTools
using Printf
using Statistics
using KernelAbstractions
using Adapt: adapt, @adapt_structure
using Accessors: @set

include("config.jl")
using .Config: load_config, Cfg
include("backend_setup.jl")
include("constants.jl")
using .Constants: PhysConsts, SimConsts, SnowConsts

include("reader.jl")
include("parameters.jl")
include("physics.jl")
include("snow.jl")
include("soil.jl")
include("routing.jl")
include("temperature.jl")
include("postprocess.jl")
include("io.jl")

const to = TimerOutputs.TimerOutput()

"""Validate the path of a file relative to the given directory."""
function validate_path(file, dir)
    file = abspath(joinpath(dir, file))
    if endswith(file, "_")
        files = readdir(dirname(file))
        n_matching_files = sum(startswith.(files, basename(file)))
        if n_matching_files < 1
            error("No files found in ", dirname(file), "starting with", basename(file))
        end
    elseif !isfile(file)
        error("Cannot find file '$file'")
    end
    return file
end

"""
Clock struct for timekeeping.

Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
mutable struct Clock{T}
    time::T
    iteration::Int
    dt::Second
end

"""
Advance clock one step in time.

Note: at iteration 0 the clock is not advanced in time,
to have the model start time at iteration 1.
"""
function advance!(clock)
    if clock.iteration != 0
        clock.time += clock.dt
    end
    clock.iteration += 1
    return nothing
end

"""Initialize clock based on config"""
function Clock(config::Cfg)
    return Clock(
        DateTime(config.start_year),
        0,
        Second(config.timestep)
    )
end


"""
mGV model state.

The model state includes all information the model's update functions
require.
"""
@kwdef struct Model
    config::Cfg    # all configuration options
    clock::Clock   # to keep track of simulation time
    grid_parameters::GridParameters
    vegetation_parameters::VegetationParameters
    monthly_vegetation_parameters::MonthlyVegetationParameters
    soil_parameters::SoilParameters
    surface_energy_variables::SurfaceEnergyVariables
    canopy_variables::CanopyVariables
    soil_variables::SoilVariables
    snow_variables::SnowVariables
    forcing_variables::ForcingVariables
    forcing_readers::ForcingReaders
    routing::Union{RoutingState, Nothing}  # nothing when routing is disabled
    writer::Union{OutputWriter, Nothing} # writes model output
end

include("energy_balance.jl")
include("evaporation.jl")

function Model(config_file::AbstractString)
    config = load_config(config_file)

    clock = Clock(config)
    forcing_readers, forcing_variables = initialize_forcing(config_file, config)
    grid_parameters, vegetation_parameters, soil_parameters, monthly_vegetation_parameters =
        read_parameters(config)

    # Check dim order!
    nx = length(grid_parameters.longitude)
    ny = length(grid_parameters.latitude)
    nveg = config.nveg  # vegetation types
    nbands = config.nbands  # snow bands
    nlayers = size(soil_parameters.depth, 3) # derive soil layers from input data
    grid_dims = (nx, ny)
    tile_dims = (nx, ny, nbands, nveg)
    soil_dims = (nx, ny, nlayers)

    surface_energy_variables = SurfaceEnergyVariables(grid_dims, tile_dims)
    canopy_variables = CanopyVariables(tile_dims)
    soil_variables = SoilVariables(grid_dims, soil_dims)
    snow_variables = SnowVariables(nx, ny, nbands, nveg)
    # Only build routing state when it is actually used; otherwise the run
    # would require a routing parameter file it never reads.
    routing = config.enable_routing ? RoutingState(config, grid_parameters.elevation) : nothing

    # Move data to backend during model initialization
    if backend_name != "CPU"
        grid_parameters = adapt(ArrayType, grid_parameters)
        vegetation_parameters = adapt(ArrayType, vegetation_parameters)
        soil_parameters = adapt(ArrayType, soil_parameters)
        surface_energy_variables = adapt(ArrayType, surface_energy_variables)
        canopy_variables = adapt(ArrayType, canopy_variables)
        soil_variables = adapt(ArrayType, soil_variables)
        snow_variables = adapt(ArrayType, snow_variables)
        forcing_variables = adapt(ArrayType, forcing_variables)
        if !isnothing(routing)
            routing = adapt(ArrayType, routing)
        end
    end

    # Seed the single-month vegetation buffers with the starting month so the
    # first timestep sees real parameters rather than zeros.
    load_monthly_parameters!(
        vegetation_parameters, monthly_vegetation_parameters, month(clock.time)
    )

    derive_soil_parameters!(soil_parameters)
    convert_nijssen2001_to_arno!(soil_parameters)

    # Initialize soil moisture and clamp between 0 and maximum_moisture
    @. soil_variables.moisture = clamp(soil_parameters.initial_moisture, 0.0f0, soil_parameters.maximum_moisture)

    # Seed soil column temperature from the annual mean/deep soil temperature parameter,
    # not 0degC -- otherwise the surface/soil thermal solve starts from a physically wrong
    # state and biases tsurf, PE and ET from day 1.
    for layer in axes(soil_variables.temperature, 3)
        @views soil_variables.temperature[:, :, layer] .= grid_parameters.average_temperature
    end

    writer = start_io_service(config, grid_parameters, year(clock.time), clock.dt)

    return Model(
        ;
        config,
        clock,
        grid_parameters,
        vegetation_parameters,
        monthly_vegetation_parameters,
        soil_parameters,
        surface_energy_variables,
        canopy_variables,
        soil_variables,
        snow_variables,
        forcing_variables,
        forcing_readers,
        routing,
        writer
    )
end

@timeit_all to function update!(model::Model)
    advance!(model.clock)

    update_forcing!(model.clock.time, model.forcing_readers, model.forcing_variables)

    # Refresh the active-month vegetation parameters
    load_monthly_parameters!(
        model.vegetation_parameters,
        model.monthly_vegetation_parameters,
        month(model.clock.time),
    )

    # Initialize surface temperature on first timestep
    if model.clock.iteration == 1
        model.surface_energy_variables.surface_temperature .= 
        model.forcing_variables.air_temperature
    end

    # Energy balance and atmospheric calculations
    update_energy_balance!(model)

    # Compute canopy evaporation
    update_canopy_evaporation!(model)

    # Transpiration
    update_transpiration!(model)
    
    # Water balance: throughfall (must run BEFORE snow dynamics so
    #   the snow kernel sees today's precipitation, not yesterday's)
    update_water_canopy_storage!(model)

    update_snow!(model)

    update_soil!(model)

    # update total fluxes
    update_total_evapotranspiration!(model)
    update_total_runoff!(model)

    # run routing
    #  Note: fix violation_counter
    if model.config.enable_routing
        update_routing!(model)
    end

    update_soil_conductivity!(model)
    update_soil_volumetric_heat_capacity!(model)
    estimate_soil_layer_temperature!(model)

    if model.clock.iteration == 1
        update_surface_temperature!(model)
        update_aerodynamic_resistance!(model)
    end
    update_surface_temperature!(model)

    update_net_radiation_post_closure!(model)

    # post process
    results = process_daily_outputs(model)

    # write away results if data writer is available
    if !isnothing(model.writer)
        write_results(model.writer.io_service, model.clock, results)
    end

    # next time step will be new year; close file output
    if !isnothing(model.writer) && year(model.clock.time + model.clock.dt) > year(model.clock.time)
        # close output
        stop_async_service(model.writer.io_service)
        close_output(model.writer.store)

        if year(model.clock.time) == model.config.end_year
            # if model has reached end time step set writer to nothing
            @set model.writer = nothing
    
        else
            # otherwise set up new output file
            println("Starting new io service...")
            new_writer = start_io_service(
                model.config, model.grid_parameters, year(model.clock.time) + 1, model.clock.dt
            )
            model.writer.io_service = new_writer.io_service
            model.writer.store = new_writer.store
        end
    end
    return nothing
end

function run(config_file::AbstractString)
    m = Model(config_file)

    end_time = DateTime(m.config.end_year + 1) - m.clock.dt
    while m.clock.time < end_time
        update!(m)
    end
    show(to)
    return m
end

function run()
    usage = "Usage: julia -e 'using mGV; mGV.run()' 'path/to/config.toml'"
    n = length(ARGS)
    if n != 1
        throw(ArgumentError(usage))
    end
    cfg_path = only(ARGS)
    if !isfile(cfg_path)
        throw(ArgumentError("Config file not found: $(cfg_path)\n"))
    end
    return run(cfg_path)
end

end # module end
