"""Store `v` at `x[i, j]`, or do nothing when `x` is `nothing` (output not selected)."""
@inline maybe_store!(::Nothing, i, j, v) = nothing
@inline maybe_store!(x::AbstractArray, i, j, v) = (x[i, j] = v; nothing)

"""Whether a sum is needed: an output array, `nothing` (not selected), or a precomputed flag."""
@inline want(::Nothing) = false
@inline want(::AbstractArray) = true
@inline want(flag::Bool) = flag

"""Add `w * x[i, j, b, k]` to `acc`, skipping NaN and values beyond `threshold`; no-op unless `want(out)`."""
@inline function add_weighted(acc, out, x, i, j, b, k, w, threshold = Inf32)
    want(out) || return acc
    val = x[i, j, b, k]
    return acc + ifelse(isnan(val) | (abs(val) > threshold), 0.0f0, w * val)
end

@kernel function fused_postprocess_kernel!(
        pe_out, nr_out, tr_out, ce_out, ws_out, # 2D Outputs
        swe_out, sc_out, sm_out,                # 2D Snow outputs
        salb_out, sts_out, tsurf_out,           # 2D Snow-weighted outputs
        @Const(pe_in), @Const(nr_in),       # 4D Inputs
        @Const(tr_in), @Const(ce_in), @Const(ws_in), # 4D Inputs
        @Const(swe_in), @Const(sc_in), @Const(sm_in), # 4D Snow inputs
        @Const(salb_in), @Const(sts_in),    # 4D Snow inputs
        @Const(tsurf_in),                   # 2D Surface temperature
        @Const(coverage), @Const(cv), @Const(AreaFract), # Weights
        threshold, fill_val,                # Scalars
    )
    i, j = @index(Global, NTuple)
    nz(x) = ifelse(isnan(x), 0.0f0, x)

    # Sums needed by the selected outputs, directly or via another output
    want_swe = want(swe_out) | want(salb_out) | want(sts_out) | want(tsurf_out)  # SWE gives the snow mask
    want_sc = want(sc_out) | want(tsurf_out)
    want_sts = want(sts_out) | want(tsurf_out)

    # 1. Initialize Accumulators
    acc_pe = 0.0f0
    acc_nr = 0.0f0
    acc_tr = 0.0f0
    acc_ce = 0.0f0
    acc_ws = 0.0f0

    acc_swe = 0.0f0
    acc_sc = 0.0f0
    acc_sm = 0.0f0
    acc_salb = 0.0f0  # weighted by the snow-covered tile weight only
    acc_sts = 0.0f0
    snow_w = 0.0f0    # sum of the snow-covered tile weights
    cv_sum = 0.0f0    # land mask

    n_bands = size(pe_in, 3)
    n_tiles = size(pe_in, 4)
    total_w = 0.0f0

    for k in 1:n_tiles
        w_cv = nz(Float32(cv[i, j, 1, k]))
        cv_sum += w_cv
        w_cov = nz(Float32(coverage[i, j, 1, k]))

        for b in 1:n_bands
            w_af = nz(Float32(AreaFract[i, j, b]))
            w_total = w_cv * w_af
            w_total_cov = w_cv * w_cov * w_af
            total_w += w_total # Accumulate total weight (should be close to 1)

            acc_pe = add_weighted(acc_pe, pe_out, pe_in, i, j, b, k, w_total, threshold)
            acc_nr = add_weighted(acc_nr, nr_out, nr_in, i, j, b, k, w_total, threshold)
            acc_tr = add_weighted(acc_tr, tr_out, tr_in, i, j, b, k, w_cov * w_af, threshold) # tr is weighted by coverage in physics
            acc_ce = add_weighted(acc_ce, ce_out, ce_in, i, j, b, k, w_total_cov, threshold)
            acc_ws = add_weighted(acc_ws, ws_out, ws_in, i, j, b, k, w_total_cov, threshold)

            # Snow: sum_{b,v}(state[b,v] * Cv[v] * AreaFract[b])
            if want_swe
                swe = swe_in[i, j, b, k]
                acc_swe += nz(swe) * w_total

                # Albedo/surf_temp: weighted only by the tiles where snow is present
                w_snow = ifelse(swe > 0.0f0, w_total, 0.0f0)
                snow_w += w_snow
                acc_salb = add_weighted(acc_salb, salb_out, salb_in, i, j, b, k, w_snow)
                acc_sts = add_weighted(acc_sts, want_sts, sts_in, i, j, b, k, w_snow)
            end
            acc_sc = add_weighted(acc_sc, want_sc, sc_in, i, j, b, k, w_total)
            acc_sm = add_weighted(acc_sm, sm_out, sm_in, i, j, b, k, w_total)
        end
    end

    active = !isnan(total_w) & (total_w >= 1.0f-6)
    maybe_store!(pe_out, i, j, ifelse(active, acc_pe, fill_val))
    maybe_store!(nr_out, i, j, ifelse(active, acc_nr, fill_val))
    maybe_store!(tr_out, i, j, ifelse(active, acc_tr, fill_val))
    maybe_store!(ce_out, i, j, ifelse(active, acc_ce, fill_val))
    maybe_store!(ws_out, i, j, ifelse(active, acc_ws, fill_val))

    # Land mask: cells with no active bands/veg → NaN (ocean cells)
    land = cv_sum > 1.0f-6
    swe_masked = ifelse(land, acc_swe, NaN32)
    coverage_masked = ifelse(land, acc_sc, NaN32)
    maybe_store!(swe_out, i, j, swe_masked)
    maybe_store!(sc_out, i, j, coverage_masked)
    maybe_store!(sm_out, i, j, ifelse(land, acc_sm, NaN32))

    # Snow-presence mask: cells with meaningful snow coverage (NaN compares false)
    present = swe_masked > 0.0f0
    snow_w = max(snow_w, 1.0f-6)
    snow_albedo = ifelse(present, acc_salb / snow_w, NaN32)
    snow_surf_temp = ifelse(present, acc_sts / snow_w, NaN32)
    maybe_store!(salb_out, i, j, snow_albedo)
    maybe_store!(sts_out, i, j, snow_surf_temp)

    # -----------------------------------------------------------------------
    # Blend tsurf with snow surface temperature (matches VIC's OUT_SURF_TEMP)
    # VIC: energy.Tsurf = snow.surf_temp when snow is present → the reported
    # surface temperature is cold (near 0°C) over snow, not the bare-soil temp.
    # Our tsurf is solved from the vegetation/soil energy balance; it stays
    # warm even when the cell is snow-covered. We correct this by blending:
    #   tsurf_out = snow_cov * snow_surf_temp + (1 - snow_cov) * bare_tsurf. TODO: is this reasonable?
    # -----------------------------------------------------------------------
    tsurf = tsurf_in[i, j]
    snow_cov_safe = ifelse(present, coverage_masked, 0.0f0)
    snow_t_safe = ifelse(present, snow_surf_temp, tsurf)
    maybe_store!(tsurf_out, i, j, snow_cov_safe * snow_t_safe + (1.0f0 - snow_cov_safe) * tsurf)
end

@kwdef struct Results{M, T}
    surface_temperature::Union{M, Nothing} # tsurf
    air_temperature::Union{M, Nothing} # tair
    precipitation::Union{M, Nothing} # prec
    total_evapotranspiration::Union{M, Nothing} # total_et
    surface_runoff::Union{M, Nothing} # surface_runoff
    total_runoff::Union{M, Nothing}
    discharge::Union{M, Nothing} # discharge
    travel_time::Union{M, Nothing} # travel_time
    potential_evaporation::Union{M, Nothing} # pe_summed
    net_radiation::Union{M, Nothing} # nr_summed
    transpiration::Union{M, Nothing} # tr_summed
    canopy_evaporation::Union{M, Nothing} # ce_summed
    water_storage::Union{M, Nothing} # ws_summed
    snow_water_equivalent::Union{M, Nothing} # swe_summed
    snow_albedo::Union{M, Nothing}
    snow_surface_temperature::Union{M, Nothing}
    snow_coverage::Union{M, Nothing}
    snow_melt::Union{M, Nothing}
    soil_evaporation::Union{M, Nothing}
    soil_moisture::Union{T, Nothing}  # 3D (nx, ny, nlayers)
end

"""
Return the fields of the output variables to write, based on `[output].variables` set by the user
in the config file (all variables when not set). Variables of disabled modules are skipped.
"""
function select_outputs(config::Cfg)
    requested = config.output.variables
    return [
        var.field for var in OUTPUT_VARIABLES
            if (isnothing(requested) || var.name in requested) &&
            (isnothing(var.requires) || getfield(config, var.requires))
    ]
end

"""Return the name of a variable in the output file."""
output_name(field::Symbol) = only(var.name for var in OUTPUT_VARIABLES if var.field == field)

"""
Preallocated device buffers for the 2D daily outputs that are not model state.
Buffers for outputs that are not selected are `nothing`.
"""
struct OutputBuffers{M}
    surface_temperature::Union{M, Nothing}
    potential_evaporation::Union{M, Nothing}
    net_radiation::Union{M, Nothing}
    transpiration::Union{M, Nothing}
    canopy_evaporation::Union{M, Nothing}
    water_storage::Union{M, Nothing}
    snow_water_equivalent::Union{M, Nothing}
    snow_albedo::Union{M, Nothing}
    snow_surface_temperature::Union{M, Nothing}
    snow_coverage::Union{M, Nothing}
    snow_melt::Union{M, Nothing}
end

"""Allocate buffers for the `selected_outputs`, with the same type and size as the 2D `template`."""
function OutputBuffers(template::AbstractMatrix, selected_outputs)
    buffer(field) = field in selected_outputs ? similar(template) : nothing
    return OutputBuffers{typeof(template)}(
        buffer(:surface_temperature), buffer(:potential_evaporation), buffer(:net_radiation),
        buffer(:transpiration), buffer(:canopy_evaporation), buffer(:water_storage),
        buffer(:snow_water_equivalent), buffer(:snow_albedo), buffer(:snow_surface_temperature),
        buffer(:snow_coverage), buffer(:snow_melt),
    )
end


function process_daily_outputs(model)

    (;
        surface_temperature, total_evapotranspiration, potential_evaporation, net_radiation,
    ) = model.surface_energy_variables
    (; air_temperature, precipitation) = model.forcing_variables
    (; surface_runoff, total_runoff) = model.soil_variables
    (; transpiration, canopy_evaporation, water_storage) = model.canopy_variables
    (; snow_band_area_fraction) = model.grid_parameters

    (; snow_water_equivalent, melt) = model.snow_variables
    snow_albedo = model.snow_variables.albedo
    snow_surface_temperature = model.snow_variables.surface_temperature
    snow_coverage = model.snow_variables.coverage

    soil_moisture = model.soil_variables.moisture
    soil_evaporation = model.soil_variables.evaporation

    (; vegetation_fraction, canopy_coverage) = model.vegetation_parameters

    (; fillvalue_threshold) = model.config
    out = model.output_buffers

    # 1. Launch the Fused Kernel, writing the 2D aggregates in one pass
    outputs = (
        out.potential_evaporation, out.net_radiation, out.transpiration,
        out.canopy_evaporation, out.water_storage,
        out.snow_water_equivalent, out.snow_coverage, out.snow_melt,
        out.snow_albedo, out.snow_surface_temperature, out.surface_temperature,
    )
    if any(!isnothing, outputs)
        kernel_launcher! = fused_postprocess_kernel!(get_backend(surface_temperature))
        kernel_launcher!(
            outputs...,
            potential_evaporation, net_radiation, transpiration, canopy_evaporation, water_storage,
            snow_water_equivalent, snow_coverage, melt, snow_albedo, snow_surface_temperature,
            surface_temperature,
            canopy_coverage, vegetation_fraction, snow_band_area_fraction,
            fillvalue_threshold, NaN32;
            ndrange = size(surface_temperature)
        )
    end

    # 2. Handle reshapes (metadata only, instant)
    if model.config.enable_routing
        discharge_2d = reshape(model.routing.discharge, size(total_runoff))
        travel_time_2d = reshape(model.routing.travel_time, size(total_runoff))
    else
        discharge_2d = nothing
        travel_time_2d = nothing
    end

    return Results(
        surface_temperature = out.surface_temperature,
        air_temperature = air_temperature,
        precipitation = precipitation,
        total_evapotranspiration = total_evapotranspiration,
        surface_runoff = surface_runoff,
        total_runoff = total_runoff,
        discharge = discharge_2d,
        travel_time = travel_time_2d,
        potential_evaporation = out.potential_evaporation,
        net_radiation = out.net_radiation,
        transpiration = out.transpiration,
        canopy_evaporation = out.canopy_evaporation,
        water_storage = out.water_storage,
        snow_water_equivalent = out.snow_water_equivalent,
        snow_albedo = out.snow_albedo,
        snow_surface_temperature = out.snow_surface_temperature,
        snow_coverage = out.snow_coverage,
        snow_melt = out.snow_melt,
        soil_evaporation = soil_evaporation,
        soil_moisture = soil_moisture,
    )
end
