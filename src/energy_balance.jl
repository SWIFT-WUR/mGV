using .Constants: T_FREEZE, PA_PER_KPA, P_STD, C_P_AIR, DAY_SEC

"""
Calculate per-snow-band forcings.
"""
function calculate_band_forcings!(
    grid_parameters::GridParameters,
    forcing_variables::ForcingVariables,
    snow_variables::SnowVariables
)
    (; elevation, snow_band_elevation, snow_band_area_fraction, snow_band_precipitation_factor) = grid_parameters
    (; air_temperature, precipitation) = forcing_variables
    (; band_air_temperature, band_precipitation) = snow_variables

    @. band_air_temperature = air_temperature - 0.0065f0 * (snow_band_elevation - elevation)

    # can't the result of this if-else statement be pre-calculated?
    @. band_precipitation = precipitation * ifelse(
                    snow_band_area_fraction > 1f-6,
                    snow_band_precipitation_factor / snow_band_area_fraction,
                    0.0f0
    )
    return nothing
end

"""
Function containing the aerodynamic resistance computation.

Note: it's not a kernel. Copied from old code, could likely be made into
KernelAbstractions GPU kernel.
"""
function aerodynamic_kernel(z0, d0, tsurf, tair, wind, Z2, Kt, gt, Tf, Ric, z_floor, d_floor, w_floor, L2_min, ra_min, ra_max)
    # 1. Roughness & Effective Height
    rough = max(z0, z_floor)
    d_eff = max(Z2 - d0, d_floor)

    # 2. Log-law terms
    ratio = clamp(d_eff / rough, 1.0f-6, 1.0f6)
    L     = log(ratio)
    L2    = max(L^2, L2_min)
    a_sq  = (Kt^2) / L2
    ccoef = 49.82f0 * a_sq * sqrt(ratio)

    # 3. Stability (Richardson Number)
    w_spd = max(wind, w_floor)
    Tmean = max(((tair + Tf) + (tsurf + Tf)) * 0.5f0, 100f0)
    
    Ri_B  = gt * (tair - tsurf) * d_eff / (Tmean * w_spd^2)
    Ri_B  = clamp(Ri_B, -0.5f0, Ric)

    # 4. Friction Factor (Fw)
    Fw_neg = 1f0 - (9.4f0 * Ri_B) / (1f0 + ccoef * sqrt(abs(Ri_B)))
    Fw_pos = 1f0 / (1f0 + 4.7f0 * Ri_B)^2
    Fw     = ifelse(Ri_B < 0.0f0, Fw_neg, Fw_pos)
    Fw     = clamp(Fw, 1.0f-3, 10f0)

    # 5. Final Resistance
    C_H = max(1f0 * a_sq * Fw, 1.0f-6)
    ra_val = 1f0 / (C_H * w_spd)
    
    return clamp(ra_val, ra_min, ra_max)
end

function update_aerodynamic_resistance!(model)
    displacement_height = model.vegetation_parameters.displacement_height
    roughness_length = model.vegetation_parameters.roughness_length

    (; surface_temperature, aerodynamic_resistance) = model.surface_energy_variables
    (; air_temperature, wind_speed) = model.forcing_variables
    (; bare_roughness) = model.soil_parameters
    (; Z2, VON_KARMAN, G, T_FREEZE, RI_CR) = Constants

    # Local constants
    z_floor = 1f-3
    d_floor = 1f-2
    w_floor = 0.1f0
    ra_min  = 1.0f0
    ra_max  = 1f5

    # Pre-calculated log expression
    L2_min  = 9.901f-5 # = log(1.01)^2

    # Grid dimensions
    n_all = size(aerodynamic_resistance, 4)
    veg_dim = max(n_all - 1, 0)

    # soil_tiles (last index)
    @views @. aerodynamic_resistance[:, :, :, n_all:n_all] = aerodynamic_kernel(
        bare_roughness,                
        displacement_height[:,:,:,n_all:n_all], 
        surface_temperature,                     
        air_temperature,                  
        wind_speed,                  
        Z2, VON_KARMAN, G, T_FREEZE, RI_CR, 
        z_floor, d_floor, w_floor, L2_min, ra_min, ra_max
    )

    if veg_dim > 0
        @views @. aerodynamic_resistance[:, :, :, 1:veg_dim] = aerodynamic_kernel(
            roughness_length[:,:,:,1:veg_dim],   
            displacement_height[:,:,:,1:veg_dim],   
            surface_temperature,
            air_temperature,
            wind_speed,
            Z2, VON_KARMAN, G, T_FREEZE, RI_CR, 
            z_floor, d_floor, w_floor, L2_min, ra_min, ra_max
        )
    end
    return nothing
end

"""
Compute the net radiation (ignoring snow).
"""
function calculate_net_radiation!(
    forcing_variables::ForcingVariables,
    surface_energy_variables::SurfaceEnergyVariables,
    albedo
)
    (; shortwave_down, longwave_down) = forcing_variables 
    (; surface_temperature, net_radiation) = surface_energy_variables
    (; EMISSIVITY, SIGMA) = Constants

    @. net_radiation = (
        (1.0f0 - albedo) * shortwave_down
        + longwave_down
        - EMISSIVITY * SIGMA * (surface_temperature + 273.15f0) ^ 4
    )
    return nothing
end

@kernel function net_radiation_snow_kernel!(
    net_radiation,
    @Const(albedo), @Const(surface_temperature),
    @Const(shortwave_down), @Const(longwave_down),
    @Const(coverage), @Const(snow_albedo), @Const(snow_surface_temperature)
)
    (; EMISSIVITY, SIGMA) = Constants
    i, j, b, v = @index(Global, NTuple)

    alb = albedo[i, j, 1, v]
    ts = surface_temperature[i, j]
    sc = coverage[i, j, b, v]

    # Blend albedo and surface temperature with the snow values where snow is present
    snowy = !(isnan(sc) || sc <= 0f0)
    eff_alb = snowy ? sc * snow_albedo[i, j, b, v] + (1f0 - sc) * alb : alb
    eff_t = snowy ? sc * snow_surface_temperature[i, j, b, v] + (1f0 - sc) * ts : ts

    net_radiation[i, j, b, v] = (
        (1f0 - eff_alb) * shortwave_down[i, j] +
        longwave_down[i, j] -
        EMISSIVITY * SIGMA * (eff_t + 273.15f0)^4
    )
end

"""
Compute the net radiation, including the effect of snow cover
"""
function calculate_net_radiation!(
    forcing_variables::ForcingVariables,
    surface_energy_variables::SurfaceEnergyVariables,
    snow_variables::SnowVariables,
    albedo
)
    (; shortwave_down, longwave_down) = forcing_variables 
    (; surface_temperature, net_radiation) = surface_energy_variables
    (; coverage) = snow_variables

    net_radiation_snow_kernel!(device_backend)(
        net_radiation,
        albedo, surface_temperature,
        shortwave_down, longwave_down,
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
    nbands,
    nveg
)
    i, j = @index(Global, NTuple)

    # Local coefficients
    G_COEFF = 1628.6f0
    AIR_C = 0.003486f0
    EPS = 1.0f-6

    # Meteo terms of this grid cell, the same for all its tiles
    tair = air_temperature[i, j]
    slope = calculate_svp_slope(tair)
    latent_heat = calculate_latent_heat(tair)
    scale_height = calculate_scale_height(tair, elevation[i, j])
    gamma = G_COEFF * (
        P_STD * exp(-elevation[i, j] / scale_height)
        ) / latent_heat
    vpd = calculate_vpd(tair, vapor_pressure[i, j])
    air_dens_term = (
        (AIR_C * surface_pressure[i, j] * PA_PER_KPA) /
        (T_FREEZE + tair) *
        (C_P_AIR * vpd * DAY_SEC)
    )

    for v in 1:nveg, b in 1:nbands
        ra = aerodynamic_resistance[i, j, b, v]
        if v < nveg
            # Vegetation: canopy resistance without water stress (rmin / LAI)
            resistance_ratio = (
                (minimum_resistance[i, j, 1, v] / max(lai[i, j, 1, v], EPS)) +
                architectural_resistance[i, j, 1, v]
            ) / ra
        else
            # Bare soil (last vegetation class): no canopy resistance
            resistance_ratio = architectural_resistance[i, j, 1, v] / ra
        end

        potential_evaporation[i, j, b, v] = max(
            (
                (slope * (net_radiation[i, j, b, v] * DAY_SEC) + (air_dens_term / ra)) /
                (latent_heat * (slope + gamma * (1f0 + resistance_ratio)))
            ), 0f0
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
    lai
)
    (; elevation) = grid_parameters
    (; net_radiation, aerodynamic_resistance) = surface_energy_variables
    (; air_temperature, surface_pressure, vapor_pressure) = forcing_variables

    potential_evaporation_kernel!(device_backend)(
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
        size(potential_evaporation, 3),
        size(potential_evaporation, 4),
        ndrange = size(air_temperature)
    )
    return nothing
end

"""
Perform the initial energy balance and atmospheric calculations
"""
function update_energy_balance!(model::Model)
    calculate_band_forcings!(
        model.grid_parameters, model.forcing_variables, model.snow_variables
    )

    update_aerodynamic_resistance!(model)

    # Step 1: compute WITHOUT snow for PE
    calculate_net_radiation!(
        model.forcing_variables,
        model.surface_energy_variables,
        model.vegetation_parameters.albedo
    )

    calculate_potential_evaporation!(
        model.surface_energy_variables.potential_evaporation,
        model.grid_parameters,
        model.forcing_variables,
        model.surface_energy_variables,
        model.vegetation_parameters.architectural_resistance,
        model.vegetation_parameters.minimum_resistance,
        model.vegetation_parameters.lai,
    )

    # Step 2: recompute WITH snow for the full energy balance
    calculate_net_radiation!(
        model.forcing_variables,
        model.surface_energy_variables,
        model.snow_variables,
        model.vegetation_parameters.albedo
    )

    calculate_potential_evaporation!(
        model.surface_energy_variables.soil_potential_evaporation,
        model.grid_parameters,
        model.forcing_variables,
        model.surface_energy_variables,
        model.vegetation_parameters.architectural_resistance,
        model.vegetation_parameters.minimum_resistance,
        model.vegetation_parameters.lai,
    )
    return nothing
end

function update_net_radiation_post_closure!(model)
    calculate_net_radiation!(
        model.forcing_variables,
        model.surface_energy_variables,
        model.snow_variables,
        model.vegetation_parameters.albedo
    )
end
