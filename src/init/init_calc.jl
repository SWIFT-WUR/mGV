# ============================================================================
# KERNEL DEFINITION
# ============================================================================
@kernel function soil_properties_kernel!(
    # Outputs
    bulk_dens_min, soil_dens_min, porosity,
    soil_moisture_max, soil_moisture_critical, 
    field_capacity, wilting_point, residual_moisture,
    # Inputs
    @Const(bulk_dens), @Const(soil_dens), @Const(depth),
    @Const(Wcr), @Const(Wfc), @Const(Wpwp), @Const(residmoist),
    # Constants
    organic_frac, bulk_dens_org, soil_dens_org
)
    i, j, k = @index(Global, NTuple)

    # 1. Mineral Densities
    bd_val = bulk_dens[i, j, k]
    sd_val = soil_dens[i, j, k]
    
    # We use the types passed in (FloatType) automatically
    bd_min = (bd_val - organic_frac * bulk_dens_org) / (1 - organic_frac)
    sd_min = (sd_val - organic_frac * soil_dens_org) / (1 - organic_frac)
    
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

# ============================================================================
# WRAPPER FUNCTION
# ============================================================================
function calculate_soil_properties!(
    # Outputs
    bulk_dens_min, soil_dens_min, porosity,
    soil_moisture_max, soil_moisture_critical, 
    field_capacity, wilting_point, residual_moisture,
    # Inputs
    bulk_dens_gpu, soil_dens_gpu, depth_gpu,
    Wcr_gpu, Wfc_gpu, Wpwp_gpu, residmoist_gpu,
    # Constants
    organic_frac, bulk_dens_org, soil_dens_org
)

    kernel_launcher! = soil_properties_kernel!(device_backend)
    
    kernel_launcher!(
        bulk_dens_min, soil_dens_min, porosity,
        soil_moisture_max, soil_moisture_critical, 
        field_capacity, wilting_point, residual_moisture,
        bulk_dens_gpu, soil_dens_gpu, depth_gpu,
        Wcr_gpu, Wfc_gpu, Wpwp_gpu, residmoist_gpu,
        FloatType(organic_frac), FloatType(bulk_dens_org), FloatType(soil_dens_org);
        ndrange=size(bulk_dens_gpu)
    )
    
end