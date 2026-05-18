function soil_conductivity_kernel(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, ORGANIC_FRAC, porosity)

    # 1. Unfrozen water content
    Wu = moist - ice_frac

    # 2. Dry conductivity (Kdry)
    # Formula: (0.135*bulk + 64.7) / (soil_dens - 0.947*bulk)
    Kdry_min = (ft(0.135) * bulk_dens_min + ft(64.7)) / (soil_dens_min - ft(0.947) * bulk_dens_min)
    Kdry     = (ft(1.0) - ORGANIC_FRAC) * Kdry_min + ORGANIC_FRAC * KDRY_ORG

    # 3. Fractional degree of saturation (Sr)
    Sr = ifelse(porosity > ft(0.0), moist / porosity, ft(0.0))

    # 4. Mineral soil conductivity (Ks_min)
    Ks_min = ifelse(quartz < ft(0.2),
                    ft(7.7) ^ quartz * ft(3.0) ^ (ft(1.0) - quartz),
                    ifelse(quartz <= ft(1.0),
                           ft(7.7) ^ quartz * ft(2.2) ^ (ft(1.0) - quartz),
                           ft(0.0)))
    
    Ks = (ft(1.0) - ORGANIC_FRAC) * Ks_min + ORGANIC_FRAC * KS_ORG

    # 5. Saturated conductivity (Ksat)
    Ksat = ifelse(Wu == moist,
                  Ks ^ (ft(1.0) - porosity) * KW ^ porosity,
                  Ks ^ (ft(1.0) - porosity) * KI ^ (porosity - Wu) * KW ^ Wu)

    # 6. Effective saturation parameter (Ke)
    Ke = ifelse(Wu == moist,
                ft(0.7) * log10(max(Sr, ft(1.0e-10))) + ft(1.0),
                Sr)

    # 7. Final Kappa Calculation
    # If moist > 0, interpolate. Else Kdry.
    term_moist = (Ksat - Kdry) * Ke + Kdry
    kappa = ifelse(moist > ft(0.0),
                   max(term_moist, Kdry),
                   Kdry)
                   
    return kappa
end

function soil_conductivity!(kappa_array, moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, ORGANIC_FRAC, porosity)

    @. kappa_array = soil_conductivity_kernel(
        moist, 
        ice_frac, 
        soil_dens_min, 
        bulk_dens_min, 
        quartz, 
        ORGANIC_FRAC, 
        porosity
    )
    return nothing
end


function volumetric_heat_capacity!(cs_array, bulk_dens, soil_dens, soil_moisture, RHO_W, ice_frac, ORGANIC_FRAC)

    @. begin
        # Calculate Cs
        # (1.0 - ORGANIC_FRAC) splits the soil_fract into mineral/organic components
        # Constant values are volumetric heat capacities in J/m^3/K

        cs_array = ft(2.0e6) * (bulk_dens / soil_dens) * (ft(1.0) - ORGANIC_FRAC) +
                   ft(2.7e6) * (bulk_dens / soil_dens) * ORGANIC_FRAC +
                   ft(4.2e6) * (soil_moisture / RHO_W) +
                   ft(1.9e6) * ice_frac +
                   ft(1.3e3) * (ft(1.0) - ((bulk_dens / soil_dens) + (soil_moisture / RHO_W) + ice_frac))
    end

    return nothing
end

function calculate_interlayer_drainage(Ksat, current_moist, max_moist, resid_moist, expt)
    # Cast entirely to Float64 strictly to mirror VIC's double-precision root solving 
    # preventing catastrophic cancellation against the exponent 19 limits natively.
    Z64 = 0.0
    EPS64 = 1e-9
    ONE64 = 1.0

    m = max(Float64(expt), 3.001)
    
    W_m = max(Float64(max_moist) - Float64(resid_moist), EPS64)
    W_a = max(Float64(current_moist) - Float64(resid_moist), Z64)

    F = clamp((W_a / W_m), Z64, ONE64)
    
    tiny_mask = F < 0.01

    term1 = F ^ (ONE64 - m)
    term2 = (Float64(Ksat) / W_m) * (ONE64 - m)
    
    inner = max(term1 - term2, EPS64)
    W_new = W_m * (inner ^ (ONE64 / (ONE64 - m)))
    
    Q12 = W_a - W_new
    Q12 = tiny_mask ? Z64 : Q12

    return Float32(clamp(Q12, Z64, W_a))
end



# Eq. 21a–21b (Liang 1994)
function calculate_baseflow(W, Wres, Wmax, Dsmax, Ds, Ws, cexp)
    EPS = ft(1e-9)
    eff_max = max.(Wmax .- Wres, EPS)
    rel_moist = clamp.((W .- Wres) ./ eff_max, ft(0), ft(1))
    
    Ws_safe = max.(Ws, EPS)
    Ws_compl = max.(ft(1) .- Ws, EPS)
    
    linear_coeff = (Dsmax .* Ds) ./ Ws_safe
    Qb_lin = linear_coeff .* rel_moist
    
    nonlin_amp = Dsmax .* (ft(1) .- Ds ./ Ws_safe)
    nonlin_frac = max.(rel_moist .- Ws, ft(0)) ./ Ws_compl
    Qb_nonlin = Qb_lin .+ nonlin_amp .* (nonlin_frac .^ cexp)
    
    Qb = ifelse.(rel_moist .<= Ws, Qb_lin, Qb_nonlin)
    
    avail = max.(W .- Wres, ft(0))
    return clamp.(Qb, ft(0), avail)
end


@kernel function infiltration_kernel!(infiltration, throughfall, surface_runoff, cv_grid)
    i, j = @index(Global, NTuple)

    if i <= size(infiltration, 1) && j <= size(infiltration, 2)
        
        # 1. Initialize accumulator locally
        acc = -surface_runoff[i, j]

        # 2. Accumulate throughfall
        # We loop over the 4th dimension (vegetation tiles) 
        n_tiles = size(throughfall, 4)
        for k in 1:n_tiles
            # throughfall is (nx, ny, 1, nveg), so we index [i, j, 1, k]
            # Multiply by fractional area cv_grid to preserve mass conservation!
            acc += throughfall[i, j, 1, k] * cv_grid[i, j, 1, k]
        end

        # 3. Write result
        infiltration[i, j] = acc
    end
end

function calculate_infiltration!(infiltration, throughfall, surface_runoff, cv_grid)

    kernel! = infiltration_kernel!(device_backend)
    nx, ny  = size(infiltration)

    # Launch kernel
    kernel!(infiltration, throughfall, surface_runoff, cv_grid; ndrange=(nx, ny))

    return nothing
end


@kernel function runoff_drainage_kernel!(
    soil_moisture,          # (nx, ny, 3)
    subsurface_runoff,      # (nx, ny)
    surface_runoff,         # (nx, ny)
    interlayer_drainage,    # (nx, ny, 2)
    surface_inflow,         # (nx, ny)
    soil_evap,              # (nx, ny)
    transpiration,          # (nx, ny, nlayer)
    moisture_max,           # (nx, ny, 3)
    ksat,                   # (nx, ny, 3)
    resid_moisture,         # (nx, ny, 3)
    expt,                   # (nx, ny, 3)
    Dsmax, Ds, Ws, c_expt   # (nx, ny) Arrays
)
    i, j = @index(Global, NTuple)

    # Boundary check
    if i <= size(soil_moisture, 1) && j <= size(soil_moisture, 2)
        
        tiny = ft(1e-9)
        zero = ft(0.0)
        one  = ft(1.0)

        # Load values for this  pixel (i,j)
        max1, max2, max3 = moisture_max[i,j,1], moisture_max[i,j,2], moisture_max[i,j,3]
        res1, res2, res3 = resid_moisture[i,j,1], resid_moisture[i,j,2], resid_moisture[i,j,3]
        exp1, exp2       = expt[i,j,1], expt[i,j,2]
        k1, k2           = ksat[i,j,1], ksat[i,j,2]
        
        # Baseflow params 
        _Dsmax  = Dsmax[i,j]
        _Ds     = Ds[i,j]
        _Ws     = Ws[i,j]
        _c_expt = c_expt[i,j]

        # Load state
        sm1 = soil_moisture[i,j,1]
        sm2 = soil_moisture[i,j,2]
        sm3 = soil_moisture[i,j,3]

        # Transpiration handling
        n_trans_layers = size(transpiration, 3)
        t1 = transpiration[i,j,1] 
        t2 = (n_trans_layers >= 2) ? transpiration[i,j,2] : zero
        t3 = (n_trans_layers >= 3) ? transpiration[i,j,3] : zero

        inflow = surface_inflow[i,j]
        evap   = soil_evap[i,j]

        # ==========================================================
        # Fractional Sub-Daily State Discretization
        # Config: RUNOFF_STEPS_PER_DAY = 24 (hourly runoff sub-steps)
        # MODEL_STEPS_PER_DAY = 1 (daily meteo) -> runoff_steps_per_dt = 24/1 = 24
        # ==========================================================
        N_STEPS = 24
        INV_STEPS = ft(1.0 / 24.0)
        
        inflow_sub = inflow * INV_STEPS
        evap_sub   = evap * INV_STEPS
        t1_sub     = t1 * INV_STEPS
        t2_sub     = t2 * INV_STEPS
        t3_sub     = t3 * INV_STEPS
        
        tot_drain_1 = zero
        tot_drain_2 = zero
        tot_baseflow = zero
        tot_spill_1 = zero
        
        for step in 1:N_STEPS
            # ==================== LAYER 1 ====================
            eff_sm1 = max(sm1 + inflow_sub - evap_sub - t1_sub, zero)
            dpot_1 = calculate_interlayer_drainage(k1 * INV_STEPS, eff_sm1, max1, res1, exp1)
            # Bound drainage dynamically across the fractional scalar
            d_1 = min(dpot_1, max(eff_sm1 - res1, zero))
            
            sm1 = sm1 + inflow_sub - (evap_sub + t1_sub) - d_1

            # ==================== LAYER 2 ====================
            eff_sm2 = max(sm2 + d_1 - t2_sub, zero)
            dpot_2 = calculate_interlayer_drainage(k2 * INV_STEPS, eff_sm2, max2, res2, exp2)
            d_2 = min(dpot_2, max(eff_sm2 - res2, zero))
            
            sm2 = sm2 + d_1 - t2_sub - d_2
            
            # Q12[lindex] += (liq[lindex] + ice[lindex]) - resid_moist[lindex]  <- makes Q12 negative
            # liq[lindex] = resid_moist  <- clamps to resid
            # This negative Q12 flows into L3 as negative inflow (upward redistribution).
            if sm2 < res2
                deficit_2 = res2 - sm2
                d_2 = d_2 - deficit_2      # d_2 goes negative = upward flow from L3
                sm2 = res2
            end

            # ==================== LAYER 3 ====================
            sm3_avail = max(sm3 + d_2 - t3_sub, zero)
            base_pot = calculate_baseflow(sm3_avail, res3, max3, _Dsmax, _Ds, _Ws, _c_expt)
            b = min(base_pot * INV_STEPS, max(sm3_avail - res3, zero))
            
            sm3 = sm3 + d_2 - t3_sub - b
            
            # Residual floor for L3 
            # If baseflow caused L3 to go below resid, reduce baseflow to compensate.
            if sm3 < res3
                deficit_3 = res3 - sm3
                b = b - deficit_3         # reduce baseflow (can go negative)
                sm3 = res3
            end
            # VIC: negative baseflow -> reduce evap (layer[lindex].evap += baseflow[fidx])
            # then set baseflow = 0. We clamp b to 0 here.
            b = max(b, zero)
            
            # Upward Spill (Cascading vertically upwards)
            sp_3 = max(sm3 - max3, zero)
            sm3 -= sp_3
            sm2 += sp_3
            
            sp_2 = max(sm2 - max2, zero)
            sm2 -= sp_2
            sm1 += sp_2
            
            sp_1 = max(sm1 - max1, zero)
            sm1 -= sp_1
            
            tot_spill_1 += sp_1
            tot_drain_1 += d_1
            tot_drain_2 += d_2
            tot_baseflow += b
        end
        
        surface_runoff[i,j] += tot_spill_1
        
        sm1_new = sm1
        sm2_new = sm2
        sm3_new = sm3
        drain_1 = tot_drain_1
        drain_2 = tot_drain_2
        runoff_val = tot_baseflow
        
        if sm3_new < res3
            shortage = res3 - sm3_new
            runoff_val -= shortage
            sm3_new = res3
        end
        
        runoff_val = max(runoff_val, zero)

        # ==================== WRITE BACK ====================
        soil_moisture[i,j,1] = sm1_new
        soil_moisture[i,j,2] = sm2_new
        soil_moisture[i,j,3] = sm3_new
        
        interlayer_drainage[i,j,1] = drain_1
        interlayer_drainage[i,j,2] = drain_2
        
        subsurface_runoff[i,j] = runoff_val
    end
end


function solve_runoff_and_drainage!(
    soil_moisture, subsurface_runoff, surface_runoff, interlayer_drainage,
    surface_inflow, soil_evaporation, transpiration,
    soil_moisture_max, ksat, residual_moisture, expt,
    Dsmax, Ds, Ws, c_expt
)
    kernel_launcher! = runoff_drainage_kernel!(device_backend)
    nx, ny = size(surface_inflow)
    
    kernel_launcher!(
        soil_moisture, subsurface_runoff, surface_runoff, interlayer_drainage,
        surface_inflow, soil_evaporation, transpiration,
        soil_moisture_max, ksat, residual_moisture, expt,
        Dsmax, Ds, Ws, c_expt;
        ndrange = (nx, ny)
    )

    return nothing
end