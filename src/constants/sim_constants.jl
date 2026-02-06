module SimConstants
    export K_L, K, z2, Ri_cr, emissivity,
           Ki, Kw, Kdry_org, Ks_org, 
           organic_frac, bulk_dens_org, soil_dens_org

    # Append 'f0' to make Float32 literals
    const K_L = 0.1f0 
    const K   = 0.4f0
    const z2  = 10.0f0    # Even for whole numbers, use .0f0 or just f0
    const Ri_cr = 0.2f0
    const emissivity = 1.0f0 

    # Thermal conductivity constants
    const Ki       = 2.2f0   # Thermal conductivity of ice (W/mK)
    const Kw       = 0.57f0  # Thermal conductivity of water (W/mK)
    const Kdry_org = 0.05f0  # Dry thermal conductivity of organic fraction (W/mK)
    const Ks_org   = 0.25f0  # Thermal conductivity of organic solid (W/mK)

    # Ground composition constants
    const organic_frac  = 0.0f0
    const bulk_dens_org = 0.0f0
    const soil_dens_org = 0.0f0

end