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
    (; EMISSIVITY, SIGMA) = Constants

    eff_alb(alb, sc, s_alb) = (isnan(sc) || sc <= 0f0) ? alb : (sc * s_alb + (1f0 - sc) * alb)
    eff_t(ts, sc, s_ts) = (isnan(sc) || sc <= 0f0) ? ts : (sc * s_ts + (1f0 - sc) * ts)
    
    @. net_radiation = (
        (1f0 - eff_alb(albedo, coverage, snow_variables.albedo)) * shortwave_down + 
        longwave_down -
        EMISSIVITY * SIGMA * (
            eff_t(surface_temperature, coverage, snow_variables.surface_temperature) + 273.15f0
        )^4
    )
    return nothing
end


@kernel function potential_evaporation_precompute_kernel!(
    air_temperature,
    surface_pressure,
    vapor_pressure,
    latent_heat,
    elevation,
    slope,
    scale_height,
    gamma,
    vpd,
    air_dens_term
)
    I = @index(Global)

    # Local coefficients
    G_COEFF = 1628.6f0
    AIR_C = 0.003486f0

    slope[I] = calculate_svp_slope(air_temperature[I])
    latent_heat[I] = calculate_latent_heat(air_temperature[I])

    scale_height[I] = calculate_scale_height(air_temperature[I], elevation[I])
    gamma[I] = G_COEFF * (
        P_STD * exp(-elevation[I] / scale_height[I])
        ) / latent_heat[I]

    vpd[I] = calculate_vpd(air_temperature[I], vapor_pressure[I])

    air_dens_term[I] = (
        (AIR_C * surface_pressure[I] * PA_PER_KPA) / 
        (T_FREEZE + air_temperature[I]) * 
        (C_P_AIR * vpd[I] * DAY_SEC)
    )
end

function potential_evaporation_precompute!(
    air_temperature,
    surface_pressure,
    vapor_pressure,
    latent_heat,
    elevation,
    slope,
    scale_height,
    gamma,
    vpd,
    air_dens_term
)
    potential_evaporation_precompute_kernel!(device_backend)(
        air_temperature,
        surface_pressure,
        vapor_pressure,
        latent_heat,
        elevation,
        slope,
        scale_height,
        gamma,
        vpd,
        air_dens_term,    
        ndrange = length(air_temperature)
    )
    return nothing
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

    # Grid dimensions
    nveg = size(aerodynamic_resistance, 4)
    veg_dim = 1:(nveg - 1)

    EPS = 1.0f-6

    # Scratch buffers live on the model, not allocated per timestep
    slope = surface_energy_variables.pe_slope
    latent_heat = surface_energy_variables.pe_latent_heat
    scale_height = surface_energy_variables.pe_scale_height
    gamma = surface_energy_variables.pe_gamma
    vpd = surface_energy_variables.pe_vpd
    air_dens_term = surface_energy_variables.pe_air_dens_term

    # 2. Pre-calculate 2D Meteorological Terms
    potential_evaporation_precompute!(
        air_temperature,
        surface_pressure,
        vapor_pressure,
        latent_heat,
        elevation,
        slope,
        scale_height,
        gamma,
        vpd,
        air_dens_term
    )

    ## ToDo; fold the following lines into the kernel as well if possible
    # Vegetation tiles: PE at minimum canopy resistance (gsm_inv=1)
    @views @. potential_evaporation[:, :, :, veg_dim] = max(
        (
            (slope * (net_radiation[:, :, :, veg_dim] * DAY_SEC) + 
            (air_dens_term / aerodynamic_resistance[:, :, :, veg_dim])) / 
            (latent_heat * (slope + gamma * (1f0 + 
            ((minimum_resistance[:, :, :, veg_dim] / max(lai[:, :, :, veg_dim], EPS)) + 
            architectural_resistance[:, :, :, veg_dim]) / aerodynamic_resistance[:, :, :, veg_dim])))
        ), 0f0
    )

    # Bare Soil Tile (rc=0, compute_pot_evap bare soil PE)
    @views @. potential_evaporation[:, :, :, nveg] = max(
        (
            (slope * (net_radiation[:, :, :, nveg] * DAY_SEC) + (air_dens_term / aerodynamic_resistance[:, :, :, nveg])) / 
            (latent_heat * (slope + gamma * (1f0 + architectural_resistance[:, :, :, nveg] / aerodynamic_resistance[:, :, :, nveg])))
        ), 0f0
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
