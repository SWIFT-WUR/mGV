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
    # Active monthly values. These hold ONLY the currently simulated month,
    # with shape (nx, ny, 1, nveg); `load_monthly_parameters!` refreshes them
    # from the host-side `MonthlyVegetationParameters` on each month change.
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
Host-side store of the full 12-month vegetation parameter cycle.

Only the active month is mirrored onto the compute device (see
`VegetationParameters`). At global 5 arcmin resolution one
(nx, ny, 12, nveg) field is 4.5 GiB, so keeping all five resident on the
device would cost 22.7 GiB of memory for data that is 11/12 unused on any
given day. This struct deliberately has no `@adapt_structure`: it must stay
on the host.
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

    field_pairs = (
        (monthly.displacement_height, veg_params.displacement_height),
        (monthly.roughness_length,    veg_params.roughness_length),
        (monthly.lai,                 veg_params.lai),
        (monthly.albedo,              veg_params.albedo),
        (monthly.canopy_coverage,     veg_params.canopy_coverage),
    )

    for (host, device) in field_pairs
        copyto!(device, host[:, :, current_month:current_month, :])
    end

    monthly.loaded_month = current_month
    return nothing
end

function read_parameters(config::Cfg)
    ds_params = NCDataset(config.input.paths.input_param_file)
    grid_params = GridParameters(
        nomissing(ds_params[config.input.names.latitude][:], 0.0),
        nomissing(ds_params[config.input.names.longitude][:], 0.0),
        nomissing(ds_params[config.input.names.elevation][:,:], 0.0),
        nomissing(ds_params[config.input.names.average_temperature][:,:], 0.0),
        nomissing(ds_params[config.input.names.annual_precipitation][:,:], 0.0),
        nomissing(ds_params[config.input.names.snow_band_area_fraction][:,:,:], 0.0),
        nomissing(ds_params[config.input.names.snow_band_elevation][:,:,:], 0.0),
        nomissing(ds_params[config.input.names.snow_band_precipitation_factor][:,:,:], 0.0),
    )

    # Reshape some 3D inputs to 4D
    cv = nomissing(ds_params[config.input.names.vegetation_fraction][:,:,:], 0.0)
    vegetation_fraction = ndims(cv) == 3 ? reshape(cv, size(cv, 1), size(cv, 2), 1, size(cv, 3)) : cv
    rmin = nomissing(ds_params[config.input.names.minimum_resistance][:,:,:], 0.0)
    minimum_resistance = ndims(rmin) == 3 ? reshape(rmin, size(rmin, 1), size(rmin, 2), 1, size(rmin, 3)) : rmin
    rarc = nomissing(ds_params[config.input.names.architectural_resistance][:,:,:], 0.0)
    architectural_resistance = ndims(rarc) == 3 ? reshape(rarc, size(rarc, 1), size(rarc, 2), 1, size(rarc, 3)) : rarc

    # The 12-month vegetation cycle stays on the host; only the active month is
    # mirrored into the device-resident `VegetationParameters` buffers below.
    monthly_veg_params = MonthlyVegetationParameters(
        nomissing(ds_params[config.input.names.displacement_height][:,:,:,:], 0.0),
        nomissing(ds_params[config.input.names.roughness_length][:,:,:,:], 0.0),
        nomissing(ds_params[config.input.names.lai][:,:,:,:], 0.0),
        nomissing(ds_params[config.input.names.albedo][:,:,:,:], 0.0),
        nomissing(ds_params[config.input.names.canopy_coverage][:,:,:,:], 0.0),
        0,  # no month loaded yet
    )

    veg_params = VegetationParameters(
        nomissing(ds_params[config.input.names.root_fraction][:,:,:,:], 0.0),
        vegetation_fraction,
        minimum_resistance,
        architectural_resistance,
        current_month_buffer(monthly_veg_params.displacement_height),
        current_month_buffer(monthly_veg_params.roughness_length),
        current_month_buffer(monthly_veg_params.lai),
        current_month_buffer(monthly_veg_params.albedo),
        current_month_buffer(monthly_veg_params.canopy_coverage)
    )

    bulk_density = nomissing(ds_params[config.input.names.bulk_density][:,:,:], 0.0)
    
    # Quartz content can vary per dataset (e.g., 2D in global, 3D in mekong), so we dynamically check and expand to 3D if needed
    quartz_raw = ds_params[config.input.names.quartz_content]
    quartz_3d = if ndims(quartz_raw) == 2
        repeat(nomissing(quartz_raw[:,:], 0.0), 1, 1, size(bulk_density, 3))
    else
        nomissing(quartz_raw[:,:,:], 0.0)
    end

    soil_params = SoilParameters(
        nomissing(ds_params[config.input.names.hydraulic_conductivity][:,:,:], 0.0),
        nomissing(ds_params[config.input.names.depth][:,:,:], 0.0),
        nomissing(ds_params[config.input.names.initial_moisture][:,:,:], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),  # calculated later
        nomissing(ds_params[config.input.names.residual_moisture_fraction][:,:,:], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        nomissing(ds_params[config.input.names.critical_moisture_fraction][:,:,:], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        nomissing(ds_params[config.input.names.field_capacity_fraction][:,:,:], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        nomissing(ds_params[config.input.names.wilting_point_fraction][:,:,:], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        quartz_3d,
        nomissing(ds_params[config.input.names.bare_roughness][:,:], 0.0),
        bulk_density,
        zeros(eltype(bulk_density), size(bulk_density)),  # calculated later
        nomissing(ds_params[config.input.names.particle_density][:,:,:], 0.0),
        zeros(eltype(bulk_density), size(bulk_density)),
        zeros(eltype(bulk_density), size(bulk_density)),
        nomissing(ds_params[config.input.names.campbell_n][:,:,:], 0.0),
        nomissing(ds_params[config.input.names.nijssen_infilt_b][:,:], 0.0),
        nomissing(ds_params[config.input.names.nijssen_lin_reservoir][:,:], 0.0),
        nomissing(ds_params[config.input.names.nijssen_nonlin_reservoir][:,:], 0.0),
        nomissing(ds_params[config.input.names.moisture_depth_baseflow_transition][:,:], 0.0),
        nomissing(ds_params[config.input.names.column_depth][:,:], 0.0),
        nomissing(ds_params[config.input.names.baseflow_curve_exp][:,:], 0.0)
    )
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

