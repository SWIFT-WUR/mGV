using .Constants: T_FREEZE, PA_PER_KPA, P_STD, C_P_AIR, DAY_SEC

@kernel function band_forcings_kernel!(
        band_air_temperature, band_precipitation,
        @Const(air_temperature), @Const(precipitation), @Const(elevation),
        @Const(snow_band_elevation), @Const(snow_band_area_fraction),
        @Const(snow_band_precipitation_factor), @Const(band_cell)
    )
    j = @index(Global, Linear)
    i = band_cell[j]

    band_air_temperature[j] = air_temperature[i] - 0.0065f0 * (snow_band_elevation[j] - elevation[i])

    # can't the result of this if-else statement be pre-calculated?
    area = snow_band_area_fraction[j]
    band_precipitation[j] = precipitation[i] * ifelse(
        area > 1.0f-6,
        snow_band_precipitation_factor[j] / area,
        0.0f0
    )
end

"""
Calculate per-snow-band forcings.
"""
function calculate_band_forcings!(
        grid_parameters::GridParameters,
        forcing_variables::ForcingVariables,
        snow_variables::SnowVariables,
        tiles::Tiles
    )
    (; elevation, snow_band_elevation, snow_band_area_fraction, snow_band_precipitation_factor) = grid_parameters
    (; air_temperature, precipitation) = forcing_variables
    (; band_air_temperature, band_precipitation) = snow_variables

    band_forcings_kernel!(get_backend(band_air_temperature))(
        band_air_temperature, band_precipitation,
        air_temperature, precipitation, elevation,
        snow_band_elevation, snow_band_area_fraction, snow_band_precipitation_factor,
        tiles.band_cell;
        ndrange = size(band_air_temperature)
    )
    return nothing
end

"""
Function containing the aerodynamic resistance computation of a single tile.
"""
function aerodynamic_kernel(z0, d0, tsurf, tair, wind, Z2, Kt, gt, Tf, Ric, z_floor, d_floor, w_floor, L2_min, ra_min, ra_max)
    # 1. Roughness & Effective Height
    rough = max(z0, z_floor)
    d_eff = max(Z2 - d0, d_floor)

    # 2. Log-law terms
    ratio = clamp(d_eff / rough, 1.0f-6, 1.0f6)
    L = log(ratio)
    L2 = max(L^2, L2_min)
    a_sq = (Kt^2) / L2
    ccoef = 49.82f0 * a_sq * sqrt(ratio)

    # 3. Stability (Richardson Number)
    w_spd = max(wind, w_floor)
    Tmean = max(((tair + Tf) + (tsurf + Tf)) * 0.5f0, 100.0f0)

    Ri_B = gt * (tair - tsurf) * d_eff / (Tmean * w_spd^2)
    Ri_B = clamp(Ri_B, -0.5f0, Ric)

    # 4. Friction Factor (Fw)
    Fw_neg = 1.0f0 - (9.4f0 * Ri_B) / (1.0f0 + ccoef * sqrt(abs(Ri_B)))
    Fw_pos = 1.0f0 / (1.0f0 + 4.7f0 * Ri_B)^2
    Fw = ifelse(Ri_B < 0.0f0, Fw_neg, Fw_pos)
    Fw = clamp(Fw, 1.0f-3, 10.0f0)

    # 5. Final Resistance
    C_H = max(1.0f0 * a_sq * Fw, 1.0f-6)
    ra_val = 1.0f0 / (C_H * w_spd)

    return clamp(ra_val, ra_min, ra_max)
end

@kernel function aerodynamic_resistance_kernel!(
        aerodynamic_resistance,
        @Const(roughness_length), @Const(displacement_height), @Const(bare_roughness),
        @Const(surface_temperature), @Const(air_temperature), @Const(wind_speed),
        @Const(cell), @Const(veg), nveg
    )
    (; Z2, VON_KARMAN, G, T_FREEZE, RI_CR) = Constants
    t = @index(Global, Linear)
    i = cell[t]

    # Local constants
    z_floor = 1.0f-3
    d_floor = 1.0f-2
    w_floor = 0.1f0
    ra_min = 1.0f0
    ra_max = 1.0f5

    # Pre-calculated log expression
    L2_min = 9.901f-5 # = log(1.01)^2

    # Bare soil (the last vegetation type) has the roughness of the soil
    z0 = veg[t] == nveg ? Float32(bare_roughness[i]) : Float32(roughness_length[t])

    aerodynamic_resistance[t] = aerodynamic_kernel(
        z0,
        displacement_height[t],
        surface_temperature[i],
        air_temperature[i],
        wind_speed[i],
        Z2, VON_KARMAN, G, T_FREEZE, RI_CR,
        z_floor, d_floor, w_floor, L2_min, ra_min, ra_max
    )
end

function update_aerodynamic_resistance!(model)
    (; displacement_height, roughness_length) = model.vegetation_parameters
    (; surface_temperature, aerodynamic_resistance) = model.surface_energy_variables
    (; air_temperature, wind_speed) = model.forcing_variables
    (; bare_roughness) = model.soil_parameters
    (; tiles) = model

    aerodynamic_resistance_kernel!(get_backend(aerodynamic_resistance))(
        aerodynamic_resistance,
        roughness_length, displacement_height, bare_roughness,
        surface_temperature, air_temperature, wind_speed,
        tiles.cell, tiles.veg, tiles.nveg;
        ndrange = size(aerodynamic_resistance)
    )
    return nothing
end

"""
Blend the albedo and surface temperature of tile `t` with the snow values where snow is
present. Without snow (`coverage` is `nothing`) they are returned unchanged.
"""
@inline snow_blend(alb, ts, ::Nothing, snow_albedo, snow_surface_temperature, t) = (alb, ts)
@inline function snow_blend(alb, ts, coverage::AbstractArray, snow_albedo, snow_surface_temperature, t)
    sc = coverage[t]
    snowy = !(isnan(sc) || sc <= 0.0f0)
    eff_alb = snowy ? sc * snow_albedo[t] + (1.0f0 - sc) * alb : alb
    eff_t = snowy ? sc * snow_surface_temperature[t] + (1.0f0 - sc) * ts : ts
    return eff_alb, eff_t
end

@kernel function net_radiation_kernel!(
        net_radiation,
        @Const(albedo), @Const(surface_temperature),
        @Const(shortwave_down), @Const(longwave_down), @Const(cell),
        coverage, snow_albedo, snow_surface_temperature  # `nothing` to ignore snow
    )
    (; EMISSIVITY, SIGMA) = Constants
    t = @index(Global, Linear)
    i = cell[t]

    eff_alb, eff_t = snow_blend(
        albedo[t], surface_temperature[i],
        coverage, snow_albedo, snow_surface_temperature, t
    )

    net_radiation[t] = (
        (1.0f0 - eff_alb) * shortwave_down[i] +
            longwave_down[i] -
            EMISSIVITY * SIGMA * @fastmath((eff_t + 273.15f0)^4)
    )
end

"""
Compute the net radiation (ignoring snow).
"""
function calculate_net_radiation!(
        forcing_variables::ForcingVariables,
        surface_energy_variables::SurfaceEnergyVariables,
        albedo,
        tiles::Tiles
    )
    (; shortwave_down, longwave_down) = forcing_variables
    (; surface_temperature, net_radiation) = surface_energy_variables

    net_radiation_kernel!(get_backend(net_radiation))(
        net_radiation,
        albedo, surface_temperature,
        shortwave_down, longwave_down, tiles.cell,
        nothing, nothing, nothing;
        ndrange = size(net_radiation)
    )
    return nothing
end

"""
Compute the net radiation, including the effect of snow cover
"""
function calculate_net_radiation!(
        forcing_variables::ForcingVariables,
        surface_energy_variables::SurfaceEnergyVariables,
        snow_variables::SnowVariables,
        albedo,
        tiles::Tiles
    )
    (; shortwave_down, longwave_down) = forcing_variables
    (; surface_temperature, net_radiation) = surface_energy_variables
    (; coverage) = snow_variables

    net_radiation_kernel!(get_backend(net_radiation))(
        net_radiation,
        albedo, surface_temperature,
        shortwave_down, longwave_down, tiles.cell,
        coverage, snow_variables.albedo, snow_variables.surface_temperature;
        ndrange = size(net_radiation)
    )
    return nothing
end


@kernel function potential_evaporation_kernel!(
        potential_evaporation,
        air_temperature,
        surface_pressure,
        vapor_pressure,
        elevation,
        net_radiation,
        aerodynamic_resistance,
        architectural_resistance,
        minimum_resistance,
        lai,
        cell_tiles,
        veg,
        nveg
    )
    i = @index(Global, Linear)

    # Local coefficients
    G_COEFF = 1628.6f0
    AIR_C = 0.003486f0
    EPS = 1.0f-6

    # Meteo terms of this grid cell, the same for all its tiles
    tair = air_temperature[i]
    slope = calculate_svp_slope(tair)
    latent_heat = calculate_latent_heat(tair)
    scale_height = calculate_scale_height(tair, elevation[i])
    gamma = G_COEFF * (
        P_STD * exp(-elevation[i] / scale_height)
    ) / latent_heat
    vpd = calculate_vpd(tair, vapor_pressure[i])
    air_dens_term = (
        (AIR_C * surface_pressure[i] * PA_PER_KPA) /
            (T_FREEZE + tair) *
            (C_P_AIR * vpd * DAY_SEC)
    )

    for t in cell_tiles[i]:(cell_tiles[i + 1] - 1)
        ra = aerodynamic_resistance[t]
        if veg[t] < nveg
            # Vegetation: canopy resistance without water stress (rmin / LAI)
            resistance_ratio = (
                (minimum_resistance[t] / max(lai[t], EPS)) +
                    architectural_resistance[t]
            ) / ra
        else
            # Bare soil (last vegetation class): no canopy resistance
            resistance_ratio = architectural_resistance[t] / ra
        end

        potential_evaporation[t] = max(
            (
                (slope * (net_radiation[t] * DAY_SEC) + (air_dens_term / ra)) /
                    (latent_heat * (slope + gamma * (1.0f0 + resistance_ratio)))
            ), 0.0f0
        )
    end
end

function calculate_potential_evaporation!(
        potential_evaporation,
        grid_parameters::GridParameters,
        forcing_variables::ForcingVariables,
        surface_energy_variables::SurfaceEnergyVariables,
        architectural_resistance,
        minimum_resistance,
        lai,
        tiles::Tiles
    )
    (; elevation) = grid_parameters
    (; net_radiation, aerodynamic_resistance) = surface_energy_variables
    (; air_temperature, surface_pressure, vapor_pressure) = forcing_variables

    potential_evaporation_kernel!(get_backend(potential_evaporation))(
        potential_evaporation,
        air_temperature,
        surface_pressure,
        vapor_pressure,
        elevation,
        net_radiation,
        aerodynamic_resistance,
        architectural_resistance,
        minimum_resistance,
        lai,
        tiles.cell_tiles,
        tiles.veg,
        tiles.nveg,
        ndrange = size(air_temperature)
    )
    return nothing
end

"""
Perform the initial energy balance and atmospheric calculations
"""
function update_energy_balance!(model::Model)
    calculate_band_forcings!(
        model.grid_parameters, model.forcing_variables, model.snow_variables, model.tiles
    )

    update_aerodynamic_resistance!(model)

    # Step 1: compute WITHOUT snow for PE
    calculate_net_radiation!(
        model.forcing_variables,
        model.surface_energy_variables,
        model.vegetation_parameters.albedo,
        model.tiles
    )

    calculate_potential_evaporation!(
        model.surface_energy_variables.potential_evaporation,
        model.grid_parameters,
        model.forcing_variables,
        model.surface_energy_variables,
        model.vegetation_parameters.architectural_resistance,
        model.vegetation_parameters.minimum_resistance,
        model.vegetation_parameters.lai,
        model.tiles,
    )

    # Step 2: recompute WITH snow for the full energy balance
    calculate_net_radiation!(
        model.forcing_variables,
        model.surface_energy_variables,
        model.snow_variables,
        model.vegetation_parameters.albedo,
        model.tiles
    )

    calculate_potential_evaporation!(
        model.surface_energy_variables.soil_potential_evaporation,
        model.grid_parameters,
        model.forcing_variables,
        model.surface_energy_variables,
        model.vegetation_parameters.architectural_resistance,
        model.vegetation_parameters.minimum_resistance,
        model.vegetation_parameters.lai,
        model.tiles,
    )
    return nothing
end

function update_net_radiation_post_closure!(model)
    calculate_net_radiation!(
        model.forcing_variables,
        model.surface_energy_variables,
        model.snow_variables,
        model.vegetation_parameters.albedo,
        model.tiles
    )
    return nothing
end
