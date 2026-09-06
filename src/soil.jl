include("runoff.jl")

@kernel function soil_properties_kernel!(
    # Outputs
    bulk_dens_min, soil_dens_min, porosity,
    soil_moisture_max, soil_moisture_critical, 
    field_capacity, wilting_point, residual_moisture,
    # Inputs
    @Const(bulk_dens), @Const(soil_dens), @Const(depth),
    @Const(Wcr), @Const(Wfc), @Const(Wpwp), @Const(residmoist),
    # Constants
    ORGANIC_FRAC, BULK_DENS_ORG, SOIL_DENS_ORG
)
    i, j, k = @index(Global, NTuple)

    # 1. Mineral Densities
    bd_val = bulk_dens[i, j, k]
    sd_val = soil_dens[i, j, k]
    
    # We use the types passed in (FloatType) automatically
    bd_min = (bd_val - ORGANIC_FRAC * BULK_DENS_ORG) / (1 - ORGANIC_FRAC)
    sd_min = (sd_val - ORGANIC_FRAC * SOIL_DENS_ORG) / (1 - ORGANIC_FRAC)
    
    bulk_dens_min[i, j, k] = bd_min
    soil_dens_min[i, j, k] = sd_min

    # 2. Porosity
    p = 1 - (bd_val / sd_val)
    # Ensure 0 matches the array type to avoid type promotion issues
    p = max(p, zero(eltype(porosity)))
    porosity[i, j, k] = p

    # 3. Hydraulic Properties
    d = depth[i, j, k]
    
    w_max = d * p * 1000
    soil_moisture_max[i, j, k] = w_max
    
    # Fractions
    soil_moisture_critical[i, j, k] = Wcr[i, j, k] * w_max
    field_capacity[i, j, k]         = Wfc[i, j, k] * w_max
    wilting_point[i, j, k]          = Wpwp[i, j, k] * w_max
    
    residual_moisture[i, j, k]      = residmoist[i, j, k] * d * 1000
end

"""
Some soil parameters are not precalculated in the parameter input data.
These need to be precalculated to model-compatible forms.
Most notably some parameters are denoted as fractions (e.g. [mm/mm]),
but are needed in absolute terms ([mm]).
"""
function derive_soil_parameters!(soil_parameters)
    (;
        # Outputs
        minimum_bulk_density, minimum_particle_density, porosity,
        maximum_moisture, critical_moisture,
        field_capacity, wilting_point,
        residual_moisture, 
        # Inputs
        bulk_density, particle_density, depth,
        critical_moisture_fraction, # Wcr
        field_capacity_fraction, # Wfc
        wilting_point_fraction,  # Wpwp
        residual_moisture_fraction,  # resid_moist
    ) = soil_parameters
    (; ORGANIC_FRAC, BULK_DENS_ORG, SOIL_DENS_ORG) = Constants

    kernel_launcher! = soil_properties_kernel!(device_backend)
    
    kernel_launcher!(
        minimum_bulk_density, minimum_particle_density, porosity,
        maximum_moisture, critical_moisture, 
        field_capacity, wilting_point, residual_moisture,
        bulk_density, particle_density, depth,
        critical_moisture_fraction, field_capacity_fraction, wilting_point_fraction, residual_moisture_fraction,
        ORGANIC_FRAC, BULK_DENS_ORG, SOIL_DENS_ORG;
        ndrange=size(bulk_density)
    )
    
end

"""
NIJSSEN2001 BASEFLOW CONVERSION KERNEL
"""
@kernel function convert_nijssen2001_kernel!(Dsmax, Ds, Ws, @Const(c), @Const(max_moist))
    i, j = @index(Global, NTuple)
    
    d1 = Ds[i, j]
    d2 = Dsmax[i, j]
    d3 = Ws[i, j]
    d4 = c[i, j]
    
    # VIC extracts ARNO limits strictly across the Layer 3 bound natively `options.Nlayer - 1`
    m_max = max_moist[i, j, 3]
    
    T = eltype(Dsmax)
    EPS = T(1e-9)
    
    if m_max > T(0) && d3 < m_max
        new_Dsmax = d2 * ((m_max - d3) ^ d4) + d1 * m_max
        new_Ds = (d1 * d3) / max(new_Dsmax, EPS)
        new_Ws = d3 / m_max
        
        Dsmax[i, j] = new_Dsmax
        Ds[i, j] = new_Ds
        Ws[i, j] = new_Ws
    end
end

function convert_nijssen2001_to_arno!(soil_parameters)
    (; 
      nijssen_nonlin_reservoir, nijssen_lin_reservoir, 
      moisture_depth_baseflow_transition, baseflow_curve_exp, maximum_moisture
    ) = soil_parameters

    kernel_launcher! = convert_nijssen2001_kernel!(device_backend)
    kernel_launcher!(
        nijssen_nonlin_reservoir, nijssen_lin_reservoir, moisture_depth_baseflow_transition, baseflow_curve_exp, maximum_moisture;
        ndrange=size(nijssen_nonlin_reservoir)
    )
end

"""
Scalar Physics Kernel (Inner Function)
"""
function soil_evap_kernel(sm_top, sm_max_top, resid_top, pe, b_i, cv, cov)
    # 1. Calculate Max Infiltration
    max_infil = (1f0 + b_i) * sm_max_top
    
    # 2. Moisture Ratio
    # This fraction determines how full the theoretical soil column is.
    ratio = clamp(1f0 - sm_top / sm_max_top, 0f0, 1f0)
    
    # 3. Handle b_i == -1.0 case without explicit branching
    # In VIC, if the shape parameter b_i is exactly -1.0, it denotes a linear storage limit.
    ratio_adj = b_i == -1f0 ? ratio : ratio ^ (1f0 / (b_i + 1f0))
    tmp = b_i == -1f0 ? max_infil : max_infil * (1f0 - ratio_adj)

    # 4. Saturation Check
    is_saturated = tmp >= max_infil
    
    # 5. ARNO Evaporation Logic
    # We replace the recursive if/else thresholds with branchless ternary expressions to compute the beta scaler.
    ratio_unsat = clamp(1f0 - (tmp / max_infil), 0f0, 1f0)
    
    ratio_powered = ratio_unsat > 0f0 ? ratio_unsat ^ b_i : 0f0
    as_val = 1f0 - ratio_powered
    ratio_beta = ratio_powered > 0f0 ? ratio_powered ^ (1f0 / b_i) : 0f0
    
    # 6. Series Expansion for exact curve integration
    dummy = 1f0
    ratio_pow_term = ratio_beta
    for k in 1:40
        dummy += (b_i * ratio_pow_term) / (b_i + Float32(k))
        ratio_pow_term *= ratio_beta
    end

    beta_asp = as_val + (1f0 - as_val) * (1f0 - ratio_beta) * dummy
    
    # 7. Final Calculation 
    # Apply the mathematically selected multiplier (beta_asp) directly 
    # unless full saturation naturally demands maximum potential evaporation.
    esoil = is_saturated ? pe : pe * beta_asp
    esoil = esoil * (1f0 - cov) * cv
    
    # 8. Cap at Available Moisture
    avail = max(sm_top - resid_top, 0f0)
    esoil = clamp(esoil, 0f0, avail)
    
    return esoil
end


function calculate_soil_evaporation!(
    soil_evap,
    soil_moisture, soil_moisture_max, potential_evaporation, 
    b_infilt_gpu, cv_gpu, coverage_gpu, residual_moisture, AreaFract_gpu
)
    # Clear the output array first (since we accumulate into it)
    fill!(soil_evap, 0f0)

    # --- 2. Apply Logic (Accumulate over Veg Types and Bands) ---
    N_veg = size(cv_gpu, 4)
    N_bands = size(AreaFract_gpu, 3)
    for i in 1:N_veg
        for b in 1:N_bands
            # Esoil = Esoil_pot * beta(sm) * (1-fcanopy) * Cv per tile.
            # pe passed in is Step 2 (snow-blended) PE which captures snow energy effects.
            @views @. soil_evap += soil_evap_kernel(
                soil_moisture[:,:,1],           
                soil_moisture_max[:,:,1],       
                residual_moisture[:,:,1],       
                potential_evaporation[:,:,b,i],
                b_infilt_gpu,                   
                cv_gpu[:,:,1,i] * AreaFract_gpu[:,:,b],
                coverage_gpu[:,:,1,i]           
            )
        end
    end
    
    return nothing
end


function update_soil!(model)
    (;
        moisture, evaporation, surface_runoff, subsurface_runoff, 
        saturated_fraction, infiltration, interlayer_drainage
    ) = model.soil_variables
    (;
        nijssen_infilt_b, residual_moisture, maximum_moisture,
        hydraulic_conductivity, campbell_n, 
        nijssen_lin_reservoir, nijssen_nonlin_reservoir,
        moisture_depth_baseflow_transition, baseflow_curve_exp
    ) = model.soil_parameters
    (; soil_potential_evaporation) = model.surface_energy_variables
    (; vegetation_fraction, canopy_coverage) = model.vegetation_parameters
    (; snow_band_area_fraction) = model.grid_parameters
    (; throughfall, transpiration_layers) = model.canopy_variables
    coverage_this_month = canopy_coverage

    calculate_soil_evaporation!(
        evaporation, moisture, maximum_moisture, soil_potential_evaporation,  # Step 2 (snow-blended) PE
        nijssen_infilt_b, vegetation_fraction, coverage_this_month, residual_moisture, snow_band_area_fraction
    )

    calculate_surface_runoff!(
        surface_runoff, saturated_fraction,
        throughfall, moisture,
        maximum_moisture, nijssen_infilt_b, vegetation_fraction, snow_band_area_fraction
    )

    calculate_infiltration!(
        infiltration, throughfall, surface_runoff,
        vegetation_fraction, snow_band_area_fraction
    )

    # Soil moisture update
    transpiration_grid = sum(transpiration_layers .* coverage_this_month, dims=4)
    solve_runoff_and_drainage!(
        moisture, subsurface_runoff, surface_runoff, interlayer_drainage,
        infiltration, evaporation, transpiration_grid,
        maximum_moisture, hydraulic_conductivity, residual_moisture, campbell_n,
        nijssen_nonlin_reservoir, nijssen_lin_reservoir, 
        moisture_depth_baseflow_transition, baseflow_curve_exp
    )
    return nothing
end

@kernel function soil_conductivity_kernel!(
    moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, ORGANIC_FRAC, porosity, kappa
)
    (; KW, KI, KS_ORG, KDRY_ORG) = Constants

    I = @index(Global)

    # 1. Unfrozen water content
    Wu = moist[I] - ice_frac[I]

    # 2. Dry conductivity (Kdry)
    # Formula: (0.135*bulk + 64.7) / (soil_dens - 0.947*bulk)
    Kdry_min = (0.135f0 * bulk_dens_min[I] + 64.7f0) / (soil_dens_min[I] - 0.947f0 * bulk_dens_min[I])
    Kdry     = (1f0 - ORGANIC_FRAC) * Kdry_min + ORGANIC_FRAC * KDRY_ORG

    # 3. Fractional degree of saturation (Sr)
    Sr = ifelse(porosity[I] > 0f0, moist[I] / porosity[I], 0f0)

    # 4. Mineral soil conductivity (Ks_min)
    Ks_min = ifelse(
        quartz[I] < 0.2f0,
        7.7f0 ^ quartz[I] * 3f0 ^ (1f0 - quartz[I]),
        ifelse(
            quartz[I] <= 1f0,
            7.7f0 ^ quartz[I] * 2.2f0 ^ (1f0 - quartz[I]),
            0f0
        )
    )
    
    Ks = (1f0 - ORGANIC_FRAC) * Ks_min + ORGANIC_FRAC * KS_ORG

    # 5. Saturated conductivity (Ksat)
    Ksat = ifelse(Wu == moist[I],
                  Ks ^ (1f0 - porosity[I]) * KW ^ porosity[I],
                  Ks ^ (1f0 - porosity[I]) * KI ^ (porosity[I] - Wu) * KW ^ Wu)

    # 6. Effective saturation parameter (Ke)
    Ke = ifelse(Wu == moist[I],
                0.7f0 * log10(max(Sr, 1.0f-10)) + 1f0,
                Sr)

    # 7. Final Kappa Calculation
    # If moist > 0, interpolate. Else Kdry.
    term_moist = (Ksat - Kdry) * Ke + Kdry
    kappa[I] = ifelse(
        moist[I] > 0f0,
        max(term_moist, Kdry),
        Kdry
    )
end

function update_soil_conductivity!(model)
    (; ORGANIC_FRAC) = Constants
    (; minimum_particle_density, minimum_bulk_density, quartz_content, porosity) = model.soil_parameters
    (; thermal_conductivity, moisture, ice_fraction) = model.soil_variables

    kernel = soil_conductivity_kernel!(device_backend)

    kernel(
        moisture, 
        ice_fraction, 
        minimum_particle_density, 
        minimum_bulk_density, 
        quartz_content, 
        ORGANIC_FRAC, 
        porosity,
        thermal_conductivity,
        ndrange=size(moisture)
    )
    return nothing


end


"""
Update the soil's volumetric heat capacity based on new soil moisture and ice fraction values.
"""
function update_soil_volumetric_heat_capacity!(model)
    (; bulk_density, particle_density) = model.soil_parameters
    (; heat_capacity, moisture, ice_fraction) = model.soil_variables
    (; RHO_W, ORGANIC_FRAC) = Constants

    @. begin
        # Calculate Cs
        # (1.0 - ORGANIC_FRAC) splits the soil_fract into mineral/organic components
        # Constant values are volumetric heat capacities in J/m^3/K

        heat_capacity = (
            2.0f6 * (bulk_density / particle_density) * (1f0 - ORGANIC_FRAC) +
            2.7f6 * (bulk_density / particle_density) * ORGANIC_FRAC +
            4.2f6 * (moisture / RHO_W) +
            1.9f6 * ice_fraction +
            1.3f3 * (1f0 - ((bulk_density / particle_density) + (moisture / RHO_W) + ice_fraction))
        )
    end

    return nothing
end


function estimate_soil_layer_temperature!(model)
    (; temperature) = model.soil_variables
    (; depth, column_depth) = model.soil_parameters
    (; average_temperature) = model.grid_parameters
    (; surface_temperature) = model.surface_energy_variables

    # Define views for clarity
    T_L1 = @view temperature[:, :, 1]
    T_L2 = @view temperature[:, :, 2]
    T_L3 = @view temperature[:, :, 3]

    D_L2 = @view depth[:, :, 2]
    D_L3 = @view depth[:, :, 3]

    # --- 1. Update Layer 3 ---
    # Must be done FIRST because it depends on the OLD values of L1 and L2
    # We inline the calculation of top_avg = (L1 + L2) * 0.5
    # Layer 3 relaxes toward the annual-mean/deep soil temperature (average_temperature),
    # not today's air temperature -- the deep soil boundary condition is ~constant.
    @. T_L3 = average_temperature - (column_depth / D_L3) * (((T_L1 + T_L2) * 0.5f0) - average_temperature) * (exp(-(D_L2 + D_L3) / column_depth) - exp(-D_L2 / column_depth))

    # --- 2. Update Layer 1 ---
    # We inline top_avg again 
    @. T_L1 = 0.5f0 * (surface_temperature + ((T_L1 + T_L2) * 0.5f0))

    # --- 3. Update Layer 2 ---
    # L2 is modeled identically to L1, so we just copy the new L1 values
    @. T_L2 = T_L1
    
    return nothing
end
