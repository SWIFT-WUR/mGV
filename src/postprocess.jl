"""Store `v` at `x[i]`, or do nothing when `x` is `nothing` (output not selected)."""
@inline maybe_store!(::Nothing, i, v) = nothing
@inline maybe_store!(x::AbstractArray, i, v) = (x[i] = v; nothing)

"""Whether a sum is needed: an output array, `nothing` (not selected), or a precomputed flag."""
@inline want(::Nothing) = false
@inline want(::AbstractArray) = true
@inline want(flag::Bool) = flag

"""Add `w * x[t]` to `acc`, skipping NaN and values beyond `threshold`; no-op unless `want(out)`."""
@inline function add_weighted(acc, out, x, t, w, threshold = Inf32)
    want(out) || return acc
    val = x[t]
    return acc + ifelse(isnan(val) | (abs(val) > threshold), 0.0f0, w * val)
end

@kernel function fused_postprocess_kernel!(
        pe_out, nr_out, tr_out, ce_out, ws_out, # Per-cell outputs
        swe_out, sc_out, sm_out,                # Per-cell snow outputs
        salb_out, sts_out, tsurf_out,           # Per-cell snow-weighted outputs
        @Const(pe_in), @Const(nr_in),       # Per-tile inputs
        @Const(tr_in), @Const(ce_in), @Const(ws_in), # Per-tile inputs
        @Const(swe_in), @Const(sc_in), @Const(sm_in), # Per-tile snow inputs
        @Const(salb_in), @Const(sts_in),    # Per-tile snow inputs
        @Const(tsurf_in),                   # Per-cell surface temperature
        @Const(coverage), @Const(cv), @Const(AreaFract), # Weights
        @Const(cell_tiles), @Const(band),   # Tile layout
        threshold, fill_val,                # Scalars
    )
    i = @index(Global, Linear)
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

    total_w = 0.0f0

    for t in cell_tiles[i]:(cell_tiles[i + 1] - 1)
        w_cv = nz(Float32(cv[t]))
        w_cov = nz(Float32(coverage[t]))
        w_af = nz(Float32(AreaFract[band[t]]))
        w_total = w_cv * w_af
        w_total_cov = w_cv * w_cov * w_af
        total_w += w_total # Accumulate total weight (should be close to 1)

        acc_pe = add_weighted(acc_pe, pe_out, pe_in, t, w_total, threshold)
        acc_nr = add_weighted(acc_nr, nr_out, nr_in, t, w_total, threshold)
        acc_tr = add_weighted(acc_tr, tr_out, tr_in, t, w_cov * w_af, threshold) # tr is weighted by coverage in physics
        acc_ce = add_weighted(acc_ce, ce_out, ce_in, t, w_total_cov, threshold)
        acc_ws = add_weighted(acc_ws, ws_out, ws_in, t, w_total_cov, threshold)

        # Snow: sum_{b,v}(state[b,v] * Cv[v] * AreaFract[b])
        if want_swe
            swe = swe_in[t]
            acc_swe += nz(swe) * w_total

            # Albedo/surf_temp: weighted only by the tiles where snow is present
            w_snow = ifelse(swe > 0.0f0, w_total, 0.0f0)
            snow_w += w_snow
            acc_salb = add_weighted(acc_salb, salb_out, salb_in, t, w_snow)
            acc_sts = add_weighted(acc_sts, want_sts, sts_in, t, w_snow)
        end
        acc_sc = add_weighted(acc_sc, want_sc, sc_in, t, w_total)
        acc_sm = add_weighted(acc_sm, sm_out, sm_in, t, w_total)
    end

    active = !isnan(total_w) & (total_w >= 1.0f-6)
    maybe_store!(pe_out, i, ifelse(active, acc_pe, fill_val))
    maybe_store!(nr_out, i, ifelse(active, acc_nr, fill_val))
    maybe_store!(tr_out, i, ifelse(active, acc_tr, fill_val))
    maybe_store!(ce_out, i, ifelse(active, acc_ce, fill_val))
    maybe_store!(ws_out, i, ifelse(active, acc_ws, fill_val))

    # Land mask: cells without tiles (no vegetation or snow band area) → NaN
    land = cell_tiles[i + 1] > cell_tiles[i]
    swe_masked = ifelse(land, acc_swe, NaN32)
    coverage_masked = ifelse(land, acc_sc, NaN32)
    maybe_store!(swe_out, i, swe_masked)
    maybe_store!(sc_out, i, coverage_masked)
    maybe_store!(sm_out, i, ifelse(land, acc_sm, NaN32))

    # Snow-presence mask: cells with meaningful snow coverage (NaN compares false)
    present = swe_masked > 0.0f0
    snow_w = max(snow_w, 1.0f-6)
    snow_albedo = ifelse(present, acc_salb / snow_w, NaN32)
    snow_surf_temp = ifelse(present, acc_sts / snow_w, NaN32)
    maybe_store!(salb_out, i, snow_albedo)
    maybe_store!(sts_out, i, snow_surf_temp)

    # -----------------------------------------------------------------------
    # Blend tsurf with snow surface temperature (matches VIC's OUT_SURF_TEMP)
    # VIC: energy.Tsurf = snow.surf_temp when snow is present → the reported
    # surface temperature is cold (near 0°C) over snow, not the bare-soil temp.
    # Our tsurf is solved from the vegetation/soil energy balance; it stays
    # warm even when the cell is snow-covered. We correct this by blending:
    #   tsurf_out = snow_cov * snow_surf_temp + (1 - snow_cov) * bare_tsurf. TODO: is this reasonable?
    # -----------------------------------------------------------------------
    tsurf = tsurf_in[i]
    snow_cov_safe = ifelse(present, coverage_masked, 0.0f0)
    snow_t_safe = ifelse(present, snow_surf_temp, tsurf)
    maybe_store!(tsurf_out, i, snow_cov_safe * snow_t_safe + (1.0f0 - snow_cov_safe) * tsurf)
end

"""
Return the metadata of the output variables to write, based on `[output].variables` set by the
user in the config file (all variables when not set). Variables of disabled modules are skipped.
"""
function select_outputs(config::Cfg)
    requested = config.output.variables
    available = output_variables(; config.enable_snow, config.enable_routing)
    return [meta for meta in values(available) if isnothing(requested) || meta.output_name in requested]
end

"""
Preallocated device buffers for the per-cell daily outputs that are not model state.
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

"""
Allocate buffers for the outputs written to file (`selected_outputs`) or exposed through BMI
(`[API].variables` in the config), with the same type and size as the per-cell `template`.
"""
function OutputBuffers(template::AbstractVector, selected_outputs, config::Cfg)
    available = output_variables(; config.enable_snow, config.enable_routing)
    api_outputs = [available[name] for name in config.API.variables if haskey(available, name)]
    lenses = [meta.lens for meta in [selected_outputs; api_outputs]]
    buffer(lens) = lens in lenses ? similar(template) : nothing
    return OutputBuffers{typeof(template)}(
        buffer(@optic(_.output_buffers.surface_temperature)),
        buffer(@optic(_.output_buffers.potential_evaporation)),
        buffer(@optic(_.output_buffers.net_radiation)),
        buffer(@optic(_.output_buffers.transpiration)),
        buffer(@optic(_.output_buffers.canopy_evaporation)),
        buffer(@optic(_.output_buffers.water_storage)),
        buffer(@optic(_.output_buffers.snow_water_equivalent)),
        buffer(@optic(_.output_buffers.snow_albedo)),
        buffer(@optic(_.output_buffers.snow_surface_temperature)),
        buffer(@optic(_.output_buffers.snow_coverage)),
        buffer(@optic(_.output_buffers.snow_melt)),
    )
end


"""
Compute the grid cell averages in `model.output_buffers` that are allocated (selected for
output or BMI).
"""
function process_daily_outputs!(model)
    (; surface_temperature, potential_evaporation, net_radiation) = model.surface_energy_variables
    (; transpiration, canopy_evaporation, water_storage) = model.canopy_variables
    (; snow_band_area_fraction) = model.grid_parameters

    (; snow_water_equivalent, melt) = model.snow_variables
    snow_albedo = model.snow_variables.albedo
    snow_surface_temperature = model.snow_variables.surface_temperature
    snow_coverage = model.snow_variables.coverage

    (; vegetation_fraction, canopy_coverage) = model.vegetation_parameters

    (; fillvalue_threshold) = model.config
    out = model.output_buffers

    # Launch the Fused Kernel, writing the per-cell aggregates in one pass
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
            model.tiles.cell_tiles, model.tiles.band,
            fillvalue_threshold, NaN32;
            ndrange = size(surface_temperature)
        )
    end
    return nothing
end
