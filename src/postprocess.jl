"""Store `v` at `x[i, j]`, or do nothing when `x` is `nothing` (output not selected)."""
@inline maybe_store!(::Nothing, i, j, v) = nothing
@inline maybe_store!(x::AbstractArray, i, j, v) = (x[i, j] = v; nothing)

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
    nz(x) = ifelse(isnan(x), 0f0, x)

    # Sums needed by the selected outputs, directly or via another output
    want(x) = !isnothing(x)
    want_swe = want(swe_out) | want(salb_out) | want(sts_out) | want(tsurf_out)  # SWE gives the snow mask
    want_sc = want(sc_out) | want(tsurf_out)
    want_sts = want(sts_out) | want(tsurf_out)

    # 1. Initialize Accumulators
    acc_pe = 0f0
    acc_nr = 0f0
    acc_tr = 0f0
    acc_ce = 0f0
    acc_ws = 0f0

    acc_swe = 0f0
    acc_sc = 0f0
    acc_sm = 0f0
    acc_salb = 0f0  # weighted by the snow-covered tile weight only
    acc_sts = 0f0
    snow_w = 0f0    # sum of the snow-covered tile weights
    cv_sum = 0f0    # land mask

    n_bands = size(pe_in, 3)
    n_tiles = size(pe_in, 4)
    total_w = 0f0

    for k in 1:n_tiles
        _cv_raw = cv[i, j, 1, k]
        w_cv = ifelse(isnan(_cv_raw), 0f0, Float32(_cv_raw))
        cv_sum += w_cv

        _cov_raw = coverage[i, j, 1, k]
        w_cov = ifelse(isnan(_cov_raw), 0f0, Float32(_cov_raw))

        for b in 1:n_bands
            _af_raw = AreaFract[i, j, b]
            w_af = ifelse(isnan(_af_raw), 0f0, Float32(_af_raw))

            w_total = w_cv * w_af
            w_total_cov = w_cv * w_cov * w_af

            total_w += w_total # Accumulate total weight (should be close to 1)

            if want(pe_out)
                val = pe_in[i, j, b, k]
                acc_pe += ifelse(isnan(val) | (abs(val) > threshold), 0f0, w_total * val)
            end
            if want(nr_out)
                val = nr_in[i, j, b, k]
                acc_nr += ifelse(isnan(val) | (abs(val) > threshold), 0f0, w_total * val)
            end
            if want(tr_out)
                val = tr_in[i, j, b, k]
                acc_tr += ifelse(isnan(val) | (abs(val) > threshold), 0f0, w_cov * w_af * val) # tr is weighted by coverage in physics
            end
            if want(ce_out)
                val = ce_in[i, j, b, k]
                acc_ce += ifelse(isnan(val) | (abs(val) > threshold), 0f0, w_total_cov * val)
            end
            if want(ws_out)
                val = ws_in[i, j, b, k]
                acc_ws += ifelse(isnan(val) | (abs(val) > threshold), 0f0, w_total_cov * val)
            end

            # Snow: sum_{b,v}(state[b,v] * Cv[v] * AreaFract[b])
            if want_swe
                swe = swe_in[i, j, b, k]
                acc_swe += nz(swe) * w_total

                # Albedo/surf_temp: weighted only by the tiles where snow is present
                w_snow = ifelse(swe > 0f0, w_total, 0f0)
                snow_w += w_snow
                if want(salb_out)
                    acc_salb += nz(salb_in[i, j, b, k]) * w_snow
                end
                if want_sts
                    acc_sts += nz(sts_in[i, j, b, k]) * w_snow
                end
            end
            if want_sc
                acc_sc += nz(sc_in[i, j, b, k]) * w_total
            end
            if want(sm_out)
                acc_sm += nz(sm_in[i, j, b, k]) * w_total
            end
        end
    end

    active = !isnan(total_w) & (total_w >= 1f-6)
    maybe_store!(pe_out, i, j, ifelse(active, acc_pe, fill_val))
    maybe_store!(nr_out, i, j, ifelse(active, acc_nr, fill_val))
    maybe_store!(tr_out, i, j, ifelse(active, acc_tr, fill_val))
    maybe_store!(ce_out, i, j, ifelse(active, acc_ce, fill_val))
    maybe_store!(ws_out, i, j, ifelse(active, acc_ws, fill_val))

    # Land mask: cells with no active bands/veg → NaN (ocean cells)
    land = cv_sum > 1f-6
    swe_masked = ifelse(land, acc_swe, NaN32)
    coverage_masked = ifelse(land, acc_sc, NaN32)
    maybe_store!(swe_out, i, j, swe_masked)
    maybe_store!(sc_out, i, j, coverage_masked)
    maybe_store!(sm_out, i, j, ifelse(land, acc_sm, NaN32))

    # Snow-presence mask: cells with meaningful snow coverage (NaN compares false)
    present = swe_masked > 0f0
    snow_w = max(snow_w, 1f-6)
    snow_albedo = ifelse(present, acc_salb / snow_w, NaN32)
    snow_surf_temp = ifelse(present, acc_sts / snow_w, NaN32)
    maybe_store!(salb_out, i, j, snow_albedo)
    maybe_store!(sts_out, i, j, snow_surf_temp)

    if want(tsurf_out)
        # -----------------------------------------------------------------------
        # Blend tsurf with snow surface temperature (matches VIC's OUT_SURF_TEMP)
        # VIC: energy.Tsurf = snow.surf_temp when snow is present → the reported  
        # surface temperature is cold (near 0°C) over snow, not the bare-soil temp.
        # Our tsurf is solved from the vegetation/soil energy balance; it stays
        # warm even when the cell is snow-covered. We correct this by blending:
        #   tsurf_out = snow_cov * snow_surf_temp + (1 - snow_cov) * bare_tsurf. TODO: is this reasonable?
        # -----------------------------------------------------------------------
        tsurf = tsurf_in[i, j]
        snow_cov_safe = ifelse(present, coverage_masked, 0f0)
        snow_t_safe = ifelse(present, snow_surf_temp, tsurf)
        tsurf_out[i, j] = snow_cov_safe * snow_t_safe + (1f0 - snow_cov_safe) * tsurf
    end
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

@adapt_structure Results

"""
All available output variables, mapping each variable to its name in the output
file. Users select which variables to output in the config file.
"""
const OUTPUT_VARIABLES = (
    (:surface_temperature,      "tsurf_output"),
    (:air_temperature,          "tair_output"),
    (:precipitation,            "precipitation_output"),
    (:total_evapotranspiration, "total_et_output"),
    (:surface_runoff,           "surface_runoff_output"),
    (:total_runoff,             "total_runoff_output"),
    (:discharge,                "discharge_output"),
    (:travel_time,              "travel_time_output"),
    (:potential_evaporation,    "potential_evaporation_summed_output"),
    (:net_radiation,            "net_radiation_summed_output"),
    (:transpiration,            "transpiration_summed_output"),
    (:canopy_evaporation,       "canopy_evaporation_summed_output"),
    (:water_storage,            "water_storage_summed_output"),
    (:snow_water_equivalent,    "swe_summed_output"),
    (:snow_albedo,              "snow_albedo_output"),
    (:snow_surface_temperature, "snow_surf_temp_output"),
    (:snow_coverage,            "snow_coverage_output"),
    (:snow_melt,                "snow_melt_output"),
    (:soil_evaporation,         "soil_evaporation_output"),
    (:soil_moisture,            "soil_moisture_output"),
)

"""
Return the list of output variables to write, based on `[output].variables` set by the user in
the config file. Throws an error for unknown variable names. Snow and routing outputs are
skipped when those modules are disabled.
"""
function select_outputs(config::Cfg)
    all_names = [name for (_, name) in OUTPUT_VARIABLES]
    requested = config.output.variables
    if !isnothing(requested)
        for name in requested
            name in all_names || error(
                "Unknown output variable '$name' in [output].variables. Valid names: " *
                join(all_names, ", ")
            )
        end
    end

    # Skip outputs of disabled modules; warn only if they were asked for explicitly
    snow_outputs = ["swe_summed_output", "snow_albedo_output", "snow_surf_temp_output",
                    "snow_coverage_output", "snow_melt_output"]
    routing_outputs = ["discharge_output", "travel_time_output"]
    disabled = String[]
    for (enabled, outputs, setting) in ((config.enable_snow, snow_outputs, "enable_snow"),
                                        (config.enable_routing, routing_outputs, "enable_routing"))
        enabled && continue
        append!(disabled, outputs)
        skipped = isnothing(requested) ? String[] : filter(in(outputs), requested)
        isempty(skipped) || @warn "Not writing $(join(skipped, ", ")), because $setting = false"
    end

    wanted = isnothing(requested) ? all_names : requested
    return [field for (field, name) in OUTPUT_VARIABLES if name in wanted && !(name in disabled)]
end

"""Return the name of a variable in the output file."""
output_name(field::Symbol) = only(name for (f, name) in OUTPUT_VARIABLES if f == field)

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

@adapt_structure OutputBuffers

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
        surface_temperature, total_evapotranspiration, potential_evaporation, net_radiation
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
            ndrange=size(surface_temperature)
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
        surface_temperature=out.surface_temperature,
        air_temperature=air_temperature,
        precipitation=precipitation,
        total_evapotranspiration=total_evapotranspiration,
        surface_runoff=surface_runoff,
        total_runoff=total_runoff,
        discharge=discharge_2d,
        travel_time=travel_time_2d,
        potential_evaporation=out.potential_evaporation,
        net_radiation=out.net_radiation,
        transpiration=out.transpiration,
        canopy_evaporation=out.canopy_evaporation,
        water_storage=out.water_storage,
        snow_water_equivalent=out.snow_water_equivalent,
        snow_albedo=out.snow_albedo,
        snow_surface_temperature=out.snow_surface_temperature,
        snow_coverage=out.snow_coverage,
        snow_melt=out.snow_melt,
        soil_evaporation=soil_evaporation,
        soil_moisture=soil_moisture,
    )
end
