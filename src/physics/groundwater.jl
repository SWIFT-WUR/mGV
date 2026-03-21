# Pure scalar function (compiles to a GPU device function)
function soil_conductivity_kernel(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, organic_frac, porosity)

    # 1. Unfrozen water content
    Wu = moist - ice_frac

    # 2. Dry conductivity (Kdry)
    # Formula: (0.135*bulk + 64.7) / (soil_dens - 0.947*bulk)
    Kdry_min = (ft(0.135) * bulk_dens_min + ft(64.7)) / (soil_dens_min - ft(0.947) * bulk_dens_min)
    Kdry     = (ft(1.0) - organic_frac) * Kdry_min + organic_frac * Kdry_org

    # 3. Fractional degree of saturation (Sr)
    Sr = ifelse(porosity > ft(0.0), moist / porosity, ft(0.0))

    # 4. Mineral soil conductivity (Ks_min)
    Ks_min = ifelse(quartz < ft(0.2),
                    ft(7.7) ^ quartz * ft(3.0) ^ (ft(1.0) - quartz),
                    ifelse(quartz <= ft(1.0),
                           ft(7.7) ^ quartz * ft(2.2) ^ (ft(1.0) - quartz),
                           ft(0.0)))
    
    Ks = (ft(1.0) - organic_frac) * Ks_min + organic_frac * Ks_org

    # 5. Saturated conductivity (Ksat)
    Ksat = ifelse(Wu == moist,
                  Ks ^ (ft(1.0) - porosity) * Kw ^ porosity,
                  Ks ^ (ft(1.0) - porosity) * Ki ^ (porosity - Wu) * Kw ^ Wu)

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

function soil_conductivity!(kappa_array, moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, organic_frac, porosity)
    # Broadcast the kernel function over the arrays.
    # Julia will fuse this into a single GPU kernel.
    @. kappa_array = soil_conductivity_kernel(
        moist, 
        ice_frac, 
        soil_dens_min, 
        bulk_dens_min, 
        quartz, 
        organic_frac, 
        porosity
    )
    return nothing
end


function volumetric_heat_capacity!(cs_array, bulk_dens, soil_dens, soil_moisture, rho_w, ice_frac, organic_frac)

    @. begin
        # Calculate Cs
        # (1.0 - organic_frac) splits the soil_fract into mineral/organic components
        # Constant values are volumetric heat capacities in J/m^3/K

        cs_array = ft(2.0e6) * (bulk_dens / soil_dens) * (ft(1.0) - organic_frac) +
                   ft(2.7e6) * (bulk_dens / soil_dens) * organic_frac +
                   ft(4.2e6) * (soil_moisture / rho_w) +
                   ft(1.9e6) * ice_frac +
                   ft(1.3e3) * (ft(1.0) - ((bulk_dens / soil_dens) + (soil_moisture / rho_w) + ice_frac))
    end

    return nothing
end

function calculate_interlayer_drainage(Ksat, current_moist, max_moist, resid_moist, expt)
    T = eltype(current_moist)
    Z = T(0)
    EPS = T(1e-9)

    # Maintain VIC constraint expt > 3
    expt = max.(expt, T(3.001))

    denom = max.(max_moist .- resid_moist, EPS)
    init_moist = max.(current_moist, resid_moist .+ EPS)

    term1 = (init_moist .- resid_moist) .^ (1 .- expt)
    term2 = Ksat ./ (denom .^ expt) .* (1 .- expt)
    inner = max.(term1 .- term2, Z)
    Q12 = init_moist .- (inner .^ (1 ./ (1 .- expt))) .- resid_moist

    avail = max.(init_moist .- resid_moist, Z)
    Q12 = clamp.(Q12, Z, avail)

    return Q12
end



# VIC Eq. 21a–21b (Liang 1994)
#function calculate_baseflow(W, Wres, Wc, Dsmax, Ds, Ws, c_exp)
#    # Work with absolute storages; Ws is a fraction in (0,1)
#    WsWc  = Ws .* Wc                  # threshold storage
#    term1 = (Ds .* Dsmax) .* (W ./ max.(WsWc, eps(eltype(W))))  # linear part
#
#    # Below threshold: purely linear
#    Qb_lin = term1
#
#    # Above threshold: add nonlinear part
#    num   = max.(W .- WsWc, 0)
#    den   = max.(Wc .- WsWc, eps(eltype(W)))
#    nonlin = (Dsmax .- (Ds .* Dsmax) ./ Ws) .* (num ./ den) .^ c_exp
#
#    Qb = ifelse.(W .<= WsWc, Qb_lin, term1 .+ nonlin)
#    # Do not withdraw more than available above residual
#    avail = max.(W .- Wres, 0)
#    return clamp.(Qb, 0, avail)
#end

function calculate_baseflow(W, Wres, Wmax, Dsmax, Ds, Ws, cexp)
    EPS = ft(1e-9)
    frac = clamp.((W .- Wres) ./ max.(Wmax .- Wres, EPS), ft(0), ft(1))
    Ws_eff = Ws  # Ws is fraction of effective capacity, but VIC uses it directly on effective frac

    bf = ifelse.(frac .<= Ws_eff,
        Dsmax .* (Ds ./ Ws_eff) .* frac,  # Linear: add / Ws for correct slope
        Dsmax .* Ds + Dsmax .* (ft(1) .- Ds) .* ((frac .- Ws_eff) ./ (ft(1) .- Ws_eff)) .^ cexp  # Nonlinear: remove * Ws, as it's Dsmax * Ds start point
    )
    return max.(bf, ft(0))
end


@kernel function infiltration_kernel!(infiltration, throughfall, surface_runoff)
    i, j = @index(Global, NTuple)

    if i <= size(infiltration, 1) && j <= size(infiltration, 2)
        
        # 1. Initialize accumulator locally
        acc = -surface_runoff[i, j]

        # 2. Accumulate throughfall
        # We loop over the 4th dimension (vegetation tiles) 
        n_tiles = size(throughfall, 4)
        for k in 1:n_tiles
            # throughfall is (nx, ny, 1, nveg), so we index [i, j, 1, k]
            acc += throughfall[i, j, 1, k]
        end

        # 3. Write result
        infiltration[i, j] = acc
    end
end

function calculate_infiltration!(infiltration, throughfall, surface_runoff)

    kernel! = infiltration_kernel!(device_backend)
    nx, ny  = size(infiltration)

    # Launch kernel
    kernel!(infiltration, throughfall, surface_runoff; ndrange=(nx, ny))

    return nothing
end


@kernel function runoff_drainage_kernel!(
    soil_moisture,          # (nx, ny, 3)
    subsurface_runoff,      # (nx, ny)
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

        # --- LOAD SCALAR PARAMETERS ---
        # We load values for this specific pixel (i,j)
        max1, max2, max3 = moisture_max[i,j,1], moisture_max[i,j,2], moisture_max[i,j,3]
        res1, res2, res3 = resid_moisture[i,j,1], resid_moisture[i,j,2], resid_moisture[i,j,3]
        exp1, exp2       = expt[i,j,1], expt[i,j,2]
        k1, k2           = ksat[i,j,1], ksat[i,j,2]
        
        # Baseflow params (Scalars)
        _Dsmax  = Dsmax[i,j]
        _Ds     = Ds[i,j]
        _Ws     = Ws[i,j]
        _c_expt = c_expt[i,j]

        # Load State
        sm1 = soil_moisture[i,j,1]
        sm2 = soil_moisture[i,j,2]
        sm3 = soil_moisture[i,j,3]

        # Transpiration Handling
        n_trans_layers = size(transpiration, 3)
        t1 = transpiration[i,j,1] 
        t2 = (n_trans_layers >= 2) ? transpiration[i,j,2] : zero
        t3 = (n_trans_layers >= 3) ? transpiration[i,j,3] : zero

        inflow = surface_inflow[i,j]
        evap   = soil_evap[i,j]

        # ==================== LAYER 1 ====================
        sm1_new = sm1 + inflow

        # ET Removal
        eff_sm1 = max(sm1_new - res1, zero)
        denom_1 = max(evap + t1, tiny)
        ratio_1 = min(one, eff_sm1 / denom_1)
        sm1_new -= (evap + t1) * ratio_1

        # Drainage L1 -> L2 (CALLING YOUR FUNCTION)
        drain_pot_1 = calculate_interlayer_drainage(k1, sm1_new, max1, res1, exp1)
        
        drain_1 = min(drain_pot_1, max(sm1_new - res1, zero))
        sm1_new -= drain_1

        # Spillover L1
        spill_1 = max(sm1_new - max1, zero)
        sm1_new -= spill_1
        drain_1 += spill_1

        # ==================== LAYER 2 ====================
        sm2_new = sm2 + drain_1

        # ET Removal
        eff_sm2 = max(sm2_new - res2, zero)
        denom_2 = max(t2, tiny)
        ratio_2 = min(one, eff_sm2 / denom_2)
        sm2_new -= t2 * ratio_2

        # Drainage L2 -> L3 (CALLING YOUR FUNCTION)
        drain_pot_2 = calculate_interlayer_drainage(k2, sm2_new, max2, res2, exp2)
        
        drain_2 = min(drain_pot_2, max(sm2_new - res2, zero))
        sm2_new -= drain_2

        # Spillover L2
        spill_2 = max(sm2_new - max2, zero)
        sm2_new -= spill_2
        drain_2 += spill_2

        # ==================== LAYER 3 ====================
        sm3_new = sm3 + drain_2

        # ET Removal
        eff_sm3 = max(sm3_new - res3, zero)
        denom_3 = max(t3, tiny)
        ratio_3 = min(one, eff_sm3 / denom_3)
        sm3_new -= t3 * ratio_3

        # Baseflow (CALLING YOUR FUNCTION)
        # We pass the scalar values we loaded above
        baseflow_pot = calculate_baseflow(sm3_new, res3, max3, _Dsmax, _Ds, _Ws, _c_expt)
        
        runoff = min(baseflow_pot, max(sm3_new - res3, zero))
        sm3_new -= runoff

        # Deep Drainage
        deep_drain = max(sm3_new - max3, zero)
        sm3_new -= deep_drain
        runoff += deep_drain

        # ==================== WRITE BACK ====================
        soil_moisture[i,j,1] = sm1_new
        soil_moisture[i,j,2] = sm2_new
        soil_moisture[i,j,3] = sm3_new
        
        interlayer_drainage[i,j,1] = drain_1
        interlayer_drainage[i,j,2] = drain_2
        
        subsurface_runoff[i,j] = runoff
    end
end


function solve_runoff_and_drainage!(
    soil_moisture, subsurface_runoff, interlayer_drainage,
    surface_inflow, soil_evaporation, transpiration,
    soil_moisture_max, ksat, residual_moisture, expt,
    Dsmax, Ds, Ws, c_expt
)
    kernel_launcher! = runoff_drainage_kernel!(device_backend)
    nx, ny = size(surface_inflow)
    
    kernel_launcher!(
        soil_moisture, subsurface_runoff, interlayer_drainage,
        surface_inflow, soil_evaporation, transpiration,
        soil_moisture_max, ksat, residual_moisture, expt,
        Dsmax, Ds, Ws, c_expt;
        ndrange = (nx, ny)
    )

    return nothing
end