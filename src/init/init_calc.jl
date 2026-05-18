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
    ORGANIC_FRAC, BULK_DENS_ORG, SOIL_DENS_ORG
)

    kernel_launcher! = soil_properties_kernel!(device_backend)
    
    kernel_launcher!(
        bulk_dens_min, soil_dens_min, porosity,
        soil_moisture_max, soil_moisture_critical, 
        field_capacity, wilting_point, residual_moisture,
        bulk_dens_gpu, soil_dens_gpu, depth_gpu,
        Wcr_gpu, Wfc_gpu, Wpwp_gpu, residmoist_gpu,
        FloatType(ORGANIC_FRAC), FloatType(BULK_DENS_ORG), FloatType(SOIL_DENS_ORG);
        ndrange=size(bulk_dens_gpu)
    )
    
end

# ============================================================================
# NIJSSEN2001 BASEFLOW CONVERSION KERNEL
# ============================================================================
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

function convert_nijssen2001_to_arno!(Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu, soil_moisture_max)
    kernel_launcher! = convert_nijssen2001_kernel!(device_backend)
    kernel_launcher!(
        Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu, soil_moisture_max;
        ndrange=size(Dsmax_gpu)
    )
end