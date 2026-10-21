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
        size(monthly_field, 1),
        size(monthly_field, 2),
        1,
        size(monthly_field, 4),
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
    copy_month!(veg_params.roughness_length, monthly.roughness_length, current_month)
    copy_month!(veg_params.lai, monthly.lai, current_month)
    copy_month!(veg_params.albedo, monthly.albedo, current_month)
    copy_month!(veg_params.canopy_coverage, monthly.canopy_coverage, current_month)

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

function read_parameters(config::Cfg)
    ds_params = NCDataset(config.input.paths.input_param_file)
    grid_params = GridParameters(
        nomissing(ds_params[config.input.names.latitude][:], 0.0),
        nomissing(ds_params[config.input.names.longitude][:], 0.0),
        nomissing(ds_params[config.input.names.elevation][:, :], 0.0),
        nomissing(ds_params[config.input.names.average_temperature][:, :], 0.0),
        nomissing(ds_params[config.input.names.annual_precipitation][:, :], 0.0),
        nomissing(ds_params[config.input.names.snow_band_area_fraction][:, :, :], 0.0),
        nomissing(ds_params[config.input.names.snow_band_elevation][:, :, :], 0.0),
        nomissing(
            ds_params[config.input.names.snow_band_precipitation_factor][:, :, :],
            0.0,
        ),
    )

    # Reshape some 3D inputs to 4D
    cv = nomissing(ds_params[config.input.names.vegetation_fraction][:, :, :], 0.0)
    vegetation_fraction =
        ndims(cv) == 3 ? reshape(cv, size(cv, 1), size(cv, 2), 1, size(cv, 3)) : cv
    rmin = nomissing(ds_params[config.input.names.minimum_resistance][:, :, :], 0.0)
    minimum_resistance =
        ndims(rmin) == 3 ? reshape(rmin, size(rmin, 1), size(rmin, 2), 1, size(rmin, 3)) :
        rmin
    rarc = nomissing(ds_params[config.input.names.architectural_resistance][:, :, :], 0.0)
    architectural_resistance =
        ndims(rarc) == 3 ? reshape(rarc, size(rarc, 1), size(rarc, 2), 1, size(rarc, 3)) :
        rarc

    monthly_veg_params = MonthlyVegetationParameters(
        nomissing(ds_params[config.input.names.displacement_height][:, :, :, :], 0.0),
        nomissing(ds_params[config.input.names.roughness_length][:, :, :, :], 0.0),
        nomissing(ds_params[config.input.names.lai][:, :, :, :], 0.0),
        nomissing(ds_params[config.input.names.albedo][:, :, :, :], 0.0),
        nomissing(ds_params[config.input.names.canopy_coverage][:, :, :, :], 0.0),
        0,  # no month loaded yet
    )

    veg_params = VegetationParameters(
        nomissing(ds_params[config.input.names.root_fraction][:, :, :, :], 0.0),
        vegetation_fraction,
        minimum_resistance,
        architectural_resistance,
        current_month_buffer(monthly_veg_params.displacement_height),
        current_month_buffer(monthly_veg_params.roughness_length),
        current_month_buffer(monthly_veg_params.lai),
        current_month_buffer(monthly_veg_params.albedo),
        current_month_buffer(monthly_veg_params.canopy_coverage),
    )

    bulk_density = nomissing(ds_params[config.input.names.bulk_density][:, :, :], 0.0)

    # Quartz content can vary per dataset (e.g., 2D in global, 3D in mekong), so we dynamically check and expand to 3D if needed
    quartz_raw = ds_params[config.input.names.quartz_content]
    quartz_3d = if ndims(quartz_raw) == 2
        repeat(nomissing(quartz_raw[:, :], 0.0), 1, 1, size(bulk_density, 3))
    else
        nomissing(quartz_raw[:, :, :], 0.0)
    end

    soil_params = SoilParameters(
        nomissing(ds_params[config.input.names.hydraulic_conductivity][:, :, :], 0.0),
        nomissing(ds_params[config.input.names.depth][:, :, :], 0.0),
        nomissing(ds_params[config.input.names.initial_moisture][:, :, :], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),  # calculated later
        nomissing(ds_params[config.input.names.residual_moisture_fraction][:, :, :], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        nomissing(ds_params[config.input.names.critical_moisture_fraction][:, :, :], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        nomissing(ds_params[config.input.names.field_capacity_fraction][:, :, :], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        nomissing(ds_params[config.input.names.wilting_point_fraction][:, :, :], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        quartz_3d,
        nomissing(ds_params[config.input.names.bare_roughness][:, :], 0.0),
        bulk_density,
        zeros(eltype(bulk_density), size(bulk_density)),  # calculated later
        nomissing(ds_params[config.input.names.particle_density][:, :, :], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        zeros(eltype(bulk_density), size(bulk_density)),
        nomissing(ds_params[config.input.names.campbell_n][:, :, :], 0.0),
        nomissing(ds_params[config.input.names.nijssen_infilt_b][:, :], 0.0),
        nomissing(ds_params[config.input.names.nijssen_lin_reservoir][:, :], 0.0),
        nomissing(ds_params[config.input.names.nijssen_nonlin_reservoir][:, :], 0.0),
        nomissing(
            ds_params[config.input.names.moisture_depth_baseflow_transition][:, :],
            0.0,
        ),
        nomissing(ds_params[config.input.names.column_depth][:, :], 0.0),
        nomissing(ds_params[config.input.names.baseflow_curve_exp][:, :], 0.0),
    )
    return grid_params, veg_params, soil_params, monthly_veg_params
end
