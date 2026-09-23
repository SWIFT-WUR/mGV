struct GridParameters{V <: AbstractVector, M <: AbstractMatrix, T <: AbstractArray}
    # Static
    latitude::V
    longitude::V
    elevation::M
    average_temperature::M
    annual_precipitation::M
    snow_band_area_fraction::T
    snow_band_elevation::T
    snow_band_precipitation_factor::T
end

@adapt_structure GridParameters

struct VegetationParameters{T <: AbstractArray}
    # Static
    root_fraction::T
    vegetation_fraction::T
    minimum_resistance::T
    architectural_resistance::T
    # Active monthly values
    displacement_height::T
    roughness_length::T
    lai::T
    albedo::T
    canopy_coverage::T
end

@adapt_structure VegetationParameters

struct SoilParameters{M <: AbstractMatrix, T <: AbstractArray}
    # Static soil and baseflow parameters
    hydraulic_conductivity::T
    depth::T
    initial_moisture::T
    maximum_moisture::T
    residual_moisture_fraction::T  # intermediary
    residual_moisture::T
    critical_moisture_fraction::T # intermediary
    critical_moisture::T
    field_capacity_fraction::T # intermediary
    field_capacity::T
    wilting_point_fraction::T # intermediary
    wilting_point::T
    quartz_content::T
    bare_roughness::M
    bulk_density::T
    minimum_bulk_density::T
    particle_density::T
    minimum_particle_density::T
    porosity::T
    campbell_n::T
    nijssen_infilt_b::M
    nijssen_lin_reservoir::M
    nijssen_nonlin_reservoir::M
    moisture_depth_baseflow_transition::M
    column_depth::M
    baseflow_curve_exp::M
end

@adapt_structure SoilParameters

"""
Full 12-month vegetation parameters

Due to storage constraints, storing all months of data on GPU uses too
much VRAM. Instead, copy over the active monthly values at the start of
each month.
"""
mutable struct MonthlyVegetationParameters{T <: AbstractArray}
    displacement_height::T
    roughness_length::T
    lai::T
    albedo::T
    canopy_coverage::T
    loaded_month::Int
end

"""Allocate the single-month device buffer matching a (nx, ny, 12, nveg) field."""
function current_month_buffer(monthly_field::AbstractArray)
    return zeros(
        eltype(monthly_field),
        size(monthly_field, 1), size(monthly_field, 2), 1, size(monthly_field, 4)
    )
end

"""
    load_monthly_parameters!(veg_params, monthly, current_month)

Copy `current_month`'s slice of every monthly vegetation parameter into the
single-month buffers held by `veg_params`. Returns immediately when the month
has not changed, so this costs one transfer per simulated month rather than
one per timestep.
"""
function load_monthly_parameters!(
    veg_params::VegetationParameters,
    monthly::MonthlyVegetationParameters,
    current_month::Integer,
)
    current_month == monthly.loaded_month && return nothing

    copy_month!(veg_params.displacement_height, monthly.displacement_height, current_month)
    copy_month!(veg_params.roughness_length,    monthly.roughness_length,    current_month)
    copy_month!(veg_params.lai,                 monthly.lai,                 current_month)
    copy_month!(veg_params.albedo,              monthly.albedo,              current_month)
    copy_month!(veg_params.canopy_coverage,     monthly.canopy_coverage,     current_month)

    monthly.loaded_month = current_month
    return nothing
end

"""
    copy_month!(dest, src, month)

Copy `src[:, :, month, :]` of a (nx, ny, 12, nveg) host array into the
(nx, ny, 1, nveg) buffer `dest`. Each (nx, ny) block is contiguous in `src`, so
it is copied directly with one `copyto!` per vegetation class. This avoids the
host temporary of slicing and the device temporary + kernel of `dest[:] = ...`.
"""
function copy_month!(dest::AbstractArray, src::Array, month::Integer)
    nx, ny, _, nveg = size(src)
    @assert size(dest) == (nx, ny, 1, nveg)
    n = nx * ny
    # Loop through vegetation index as it's not contiguous in memory.
    # Reordering input data to (nx, ny, nveg, month) would simplify this, but does
    # not match input data.
    for v in 1:nveg
        copyto!(dest, (v - 1) * n + 1, src, LinearIndices(src)[1, 1, month, v], n)
    end
    return dest
end

"""Open a static input file: Zarr if `path` ends in ".zarr", NetCDF otherwise."""
open_param_source(path::AbstractString) =
    endswith(path, ".zarr") ? zopen(path) : NCDataset(path)

"""
Read one variable, e.g. "elev" or "downstream_id", from the land surface
parameter file or the routing parameter file into memory. For NetCDF, missing
values are replaced by `fallback` (e.g. 0 or NaN). Zarr stores are assumed to
have no missing values, so they are read as-is.
"""
readfull(ds::NCDataset, name::AbstractString, fallback) = nomissing(Array(ds[name]), fallback)
readfull(ds, name::AbstractString, fallback) = Array(ds[name])

close_param_source(ds::NCDataset) = close(ds)
close_param_source(ds) = nothing

function read_parameters(config::Cfg)
    ds_params = open_param_source(config.input.paths.input_param_file)
    grid_params = GridParameters(
        readfull(ds_params, config.input.names.latitude, 0.0),
        readfull(ds_params, config.input.names.longitude, 0.0),
        readfull(ds_params, config.input.names.elevation, 0.0),
        readfull(ds_params, config.input.names.average_temperature, 0.0),
        readfull(ds_params, config.input.names.annual_precipitation, 0.0),
        readfull(ds_params, config.input.names.snow_band_area_fraction, 0.0),
        readfull(ds_params, config.input.names.snow_band_elevation, 0.0),
        readfull(ds_params, config.input.names.snow_band_precipitation_factor, 0.0),
    )

    # Reshape some 3D inputs to 4D
    cv = readfull(ds_params, config.input.names.vegetation_fraction, 0.0)
    vegetation_fraction = ndims(cv) == 3 ? reshape(cv, size(cv, 1), size(cv, 2), 1, size(cv, 3)) : cv
    rmin = readfull(ds_params, config.input.names.minimum_resistance, 0.0)
    minimum_resistance = ndims(rmin) == 3 ? reshape(rmin, size(rmin, 1), size(rmin, 2), 1, size(rmin, 3)) : rmin
    rarc = readfull(ds_params, config.input.names.architectural_resistance, 0.0)
    architectural_resistance = ndims(rarc) == 3 ? reshape(rarc, size(rarc, 1), size(rarc, 2), 1, size(rarc, 3)) : rarc

    monthly_veg_params = MonthlyVegetationParameters(
        readfull(ds_params, config.input.names.displacement_height, 0.0),
        readfull(ds_params, config.input.names.roughness_length, 0.0),
        readfull(ds_params, config.input.names.lai, 0.0),
        readfull(ds_params, config.input.names.albedo, 0.0),
        readfull(ds_params, config.input.names.canopy_coverage, 0.0),
        0,  # no month loaded yet
    )

    veg_params = VegetationParameters(
        readfull(ds_params, config.input.names.root_fraction, 0.0),
        vegetation_fraction,
        minimum_resistance,
        architectural_resistance,
        current_month_buffer(monthly_veg_params.displacement_height),
        current_month_buffer(monthly_veg_params.roughness_length),
        current_month_buffer(monthly_veg_params.lai),
        current_month_buffer(monthly_veg_params.albedo),
        current_month_buffer(monthly_veg_params.canopy_coverage)
    )

    bulk_density = readfull(ds_params, config.input.names.bulk_density, 0.0)

    # Quartz content can vary per dataset (e.g., 2D in global, 3D in mekong), so we dynamically check and expand to 3D if needed
    quartz_full = readfull(ds_params, config.input.names.quartz_content, 0.0)
    quartz_3d = if ndims(quartz_full) == 2
        repeat(quartz_full, 1, 1, size(bulk_density, 3))
    else
        quartz_full
    end

    soil_params = SoilParameters(
        readfull(ds_params, config.input.names.hydraulic_conductivity, 0.0),
        readfull(ds_params, config.input.names.depth, 0.0),
        readfull(ds_params, config.input.names.initial_moisture, 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),  # calculated later
        readfull(ds_params, config.input.names.residual_moisture_fraction, 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        readfull(ds_params, config.input.names.critical_moisture_fraction, 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        readfull(ds_params, config.input.names.field_capacity_fraction, 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        readfull(ds_params, config.input.names.wilting_point_fraction, 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        quartz_3d,
        readfull(ds_params, config.input.names.bare_roughness, 0.0),
        bulk_density,
        zeros(eltype(bulk_density), size(bulk_density)),  # calculated later
        readfull(ds_params, config.input.names.particle_density, 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        zeros(eltype(bulk_density), size(bulk_density)),
        readfull(ds_params, config.input.names.campbell_n, 0.0),
        readfull(ds_params, config.input.names.nijssen_infilt_b, 0.0),
        readfull(ds_params, config.input.names.nijssen_lin_reservoir, 0.0),
        readfull(ds_params, config.input.names.nijssen_nonlin_reservoir, 0.0),
        readfull(ds_params, config.input.names.moisture_depth_baseflow_transition, 0.0),
        readfull(ds_params, config.input.names.column_depth, 0.0),
        readfull(ds_params, config.input.names.baseflow_curve_exp, 0.0)
    )
    close_param_source(ds_params)
    return grid_params, veg_params, soil_params, monthly_veg_params
end

function read_and_allocate_parameter(varname::String)
    println("Loading $varname parameter input...")

    # 1) Open netCDF file 
    dataset = NetCDF.open(input_param_file)
    
    if !haskey(dataset.vars, varname)
        println("  -> WARNING: Variable $varname not found in $input_param_file. Returning nothing.")
        return nothing, nothing
    end

    var_dims = size(dataset[varname])

    # 2) Read data sequentially into memory and format
    slicing_indices = repeat([:], length(var_dims))
    cpu_preload = dataset[varname][slicing_indices...]

    # NetCDF.jl natively replaces _FillValue with Float NaN. This causes math corruption cascades. Re-cast to 0.0.
    cpu_preload[isnan.(cpu_preload)] .= eltype(cpu_preload)(0.0)

    # 3) Print array sizes for diagnostics
    if length(var_dims) <= 4
        println("Element type for $(length(var_dims))D: ", eltype(cpu_preload))
    end
    println("Full size of $varname: ", size(cpu_preload))

    # 4) Optimizations for data transfer
    # Locks memory pages if using NVIDIA/AMD; does nothing on CPU/Metal.
    try
        pin_memory!(cpu_preload)
    catch e
        println("  -> WARNING: Failed to pin CPU memory. Transfer will be slower. Error: $e")
    end

    # 5) Allocate device memory
    # Handle 4D reshaping logic to only pre-allocate memory daily for monthly (vegetation) tiles 
    adjusted_dims = if length(var_dims) == 4
        (var_dims[1], var_dims[2], (var_dims[3] == 12 ? 1 : var_dims[3]), var_dims[4])
    else
        var_dims
    end

    # 6) Pre-allocating memory on the active device (VRAM/RAM)
    device_arr = alloc(FloatType, adjusted_dims...)
    println("Allocated $backend_name array of size: ", size(device_arr))

    return cpu_preload, device_arr
end

