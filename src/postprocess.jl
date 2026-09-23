@kernel function fused_preprocess_kernel!(
    pe_out, nr_out, tr_out, ce_out, ws_out, # 2D Outputs
    swe_out, sc_out, sm_out,                # 2D Snow outputs
    salb_out, sts_out, tsurf_out,           # 2D Snow-weighted outputs
    @Const(pe_in), @Const(nr_in),       # 4D Inputs
    @Const(tr_in), @Const(ce_in), @Const(ws_in), # 4D Inputs
    @Const(swe_in), @Const(sc_in), @Const(sm_in), # 4D Snow inputs
    @Const(salb_in), @Const(sts_in),    # 4D Snow inputs
    @Const(tsurf_in),                   # 2D Surface temperature
    @Const(coverage), @Const(cv), @Const(AreaFract), # Weights
    threshold, fill_val                 # Scalars
)
    i, j = @index(Global, NTuple)
    nz(x) = ifelse(isnan(x), 0f0, x)

    # 1. Initialize Accumulators
    acc_pe = zero(eltype(pe_out))
    acc_nr = zero(eltype(nr_out))
    acc_tr = zero(eltype(tr_out))
    acc_ce = zero(eltype(ce_out))
    acc_ws = zero(eltype(ws_out))

    acc_swe = 0f0
    acc_sc = 0f0
    acc_sm = 0f0
    acc_salb = 0f0  # weighted by the snow-covered tile weight only
    acc_sts = 0f0
    snow_w = 0f0    # sum of the snow-covered tile weights
    cv_sum = 0f0    # land mask

    n_bands = size(pe_in, 3)
    n_tiles = size(pe_in, 4)
    total_w = zero(eltype(pe_out))

    for k in 1:n_tiles
        _cv_raw = cv[i, j, 1, k]
        w_cv = ifelse(isnan(_cv_raw),  zero(eltype(pe_out)), eltype(pe_out)(_cv_raw))
        cv_sum += w_cv

        _cov_raw = coverage[i, j, 1, k]
        w_cov = ifelse(isnan(_cov_raw), zero(eltype(pe_out)), eltype(pe_out)(_cov_raw))

        for b in 1:n_bands
            _af_raw = AreaFract[i, j, b]
            w_af = ifelse(isnan(_af_raw), zero(eltype(pe_out)), eltype(pe_out)(_af_raw))
            
            w_total = w_cv * w_af
            w_total_cov = w_cv * w_cov * w_af

            total_w += w_total # Accumulate total weight (should be close to 1)

            val = pe_in[i, j, b, k]
            acc_pe += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(pe_out)), w_total * val)

            val = nr_in[i, j, b, k]
            acc_nr += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(nr_out)), w_total * val)

            val = tr_in[i, j, b, k]
            acc_tr += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(tr_out)), w_cov * w_af * val) # tr is weighted by coverage in physics

            val = ce_in[i, j, b, k]
            acc_ce += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(ce_out)), w_total_cov * val)

            val = ws_in[i, j, b, k]
            acc_ws += ifelse(isnan(val) | (abs(val) > threshold), zero(eltype(ws_out)), w_total_cov * val)

            # Snow: sum_{b,v}(state[b,v] * Cv[v] * AreaFract[b])
            swe = swe_in[i, j, b, k]
            acc_swe += nz(swe) * w_total
            acc_sc += nz(sc_in[i, j, b, k]) * w_total
            acc_sm += nz(sm_in[i, j, b, k]) * w_total

            # Albedo/surf_temp: weighted only by the tiles where snow is present
            w_snow = ifelse(swe > 0f0, w_total, 0f0)
            snow_w += w_snow
            acc_salb += nz(salb_in[i, j, b, k]) * w_snow
            acc_sts += nz(sts_in[i, j, b, k]) * w_snow
        end
    end

    active = !isnan(total_w) & (total_w >= eltype(pe_out)(1f-6))
    pe_out[i, j] = ifelse(active, acc_pe, fill_val)
    nr_out[i, j] = ifelse(active, acc_nr, fill_val)
    tr_out[i, j] = ifelse(active, acc_tr, fill_val)
    ce_out[i, j] = ifelse(active, acc_ce, fill_val)
    ws_out[i, j] = ifelse(active, acc_ws, fill_val)

    # Land mask: cells with no active bands/veg → NaN (ocean cells)
    land = cv_sum > 1f-6
    swe_masked = ifelse(land, acc_swe, NaN32)
    coverage_masked = ifelse(land, acc_sc, NaN32)
    swe_out[i, j] = swe_masked
    sc_out[i, j] = coverage_masked
    sm_out[i, j] = ifelse(land, acc_sm, NaN32)

    # Snow-presence mask: cells with meaningful snow coverage (NaN compares false)
    present = swe_masked > 0f0
    snow_w = max(snow_w, 1f-6)
    snow_albedo = ifelse(present, acc_salb / snow_w, NaN32)
    snow_surf_temp = ifelse(present, acc_sts / snow_w, NaN32)
    salb_out[i, j] = snow_albedo
    sts_out[i, j] = snow_surf_temp

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

@kwdef struct Results{M, T}
    surface_temperature::M # tsurf
    air_temperature::M # tair
    precipitation::M # prec
    total_evapotranspiration::M # total_et
    surface_runoff::M # surface_runoff
    total_runoff::M
    discharge::M # discharge
    travel_time::M # travel_time
    potential_evaporation::M # pe_summed
    net_radiation::M # nr_summed
    transpiration::M # tr_summed
    canopy_evaporation::M # ce_summed
    water_storage::M # ws_summed
    snow_water_equivalent::M # swe_summed
    snow_albedo::M
    snow_surface_temperature::M
    snow_coverage::M
    snow_melt::M
    soil_evaporation::M
    soil_moisture::T  # 3D (nx, ny, nlayers)
end

@adapt_structure Results

"""
Preallocated device buffers for the 2D daily outputs that are not model state.
"""
struct OutputBuffers{M}
    surface_temperature::M
    discharge::M    # zeros, used when routing is disabled
    travel_time::M  # zeros, used when routing is disabled
    potential_evaporation::M
    net_radiation::M
    transpiration::M
    canopy_evaporation::M
    water_storage::M
    snow_water_equivalent::M
    snow_albedo::M
    snow_surface_temperature::M
    snow_coverage::M
    snow_melt::M
end

@adapt_structure OutputBuffers

"""Allocate output buffers with the same type and size as the 2D `template`."""
function OutputBuffers(template::AbstractMatrix)
    buffer() = similar(template)
    zero_buffer() = fill!(similar(template), 0f0)
    return OutputBuffers(
        buffer(), zero_buffer(), zero_buffer(),
        buffer(), buffer(), buffer(), buffer(), buffer(),
        buffer(), buffer(), buffer(), buffer(), buffer(),
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

    # 1. Launch the Fused Kernel, writing all 2D aggregates in one pass
    kernel_launcher! = fused_preprocess_kernel!(device_backend)
    kernel_launcher!(
        out.potential_evaporation, out.net_radiation, out.transpiration,
        out.canopy_evaporation, out.water_storage,
        out.snow_water_equivalent, out.snow_coverage, out.snow_melt,
        out.snow_albedo, out.snow_surface_temperature, out.surface_temperature,
        potential_evaporation, net_radiation, transpiration, canopy_evaporation, water_storage,
        snow_water_equivalent, snow_coverage, melt, snow_albedo, snow_surface_temperature,
        surface_temperature,
        canopy_coverage, vegetation_fraction, snow_band_area_fraction,
        fillvalue_threshold, NaN32;
        ndrange=size(surface_temperature)
    )

    # 2. Handle reshapes (metadata only, instant)
    if model.config.enable_routing
        discharge_2d = reshape(model.routing.discharge, size(total_runoff))
        travel_time_2d = reshape(model.routing.travel_time, size(total_runoff)) 
    else
        discharge_2d = out.discharge
        travel_time_2d = out.travel_time
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
