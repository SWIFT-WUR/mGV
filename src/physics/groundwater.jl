# Pure scalar function (compiles to a GPU device function)
function soil_conductivity_kernel(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, organic_frac, porosity)

    # 1. Unfrozen water content
    Wu = moist - ice_frac

    # 2. Dry conductivity (Kdry)
    # Formula: (0.135*bulk + 64.7) / (soil_dens - 0.947*bulk)
    Kdry_min = (0.135f0 * bulk_dens_min + 64.7f0) / (soil_dens_min - 0.947f0 * bulk_dens_min)
    Kdry     = (1.0f0 - organic_frac) * Kdry_min + organic_frac * Kdry_org

    # 3. Fractional degree of saturation (Sr)
    Sr = ifelse(porosity > 0.0f0, moist / porosity, 0.0f0)

    # 4. Mineral soil conductivity (Ks_min)
    Ks_min = ifelse(quartz < 0.2f0,
                    7.7f0 ^ quartz * 3.0f0 ^ (1.0f0 - quartz),
                    ifelse(quartz <= 1.0f0,
                           7.7f0 ^ quartz * 2.2f0 ^ (1.0f0 - quartz),
                           0.0f0))
    
    Ks = (1.0f0 - organic_frac) * Ks_min + organic_frac * Ks_org

    # 5. Saturated conductivity (Ksat)
    Ksat = ifelse(Wu == moist,
                  Ks ^ (1.0f0 - porosity) * Kw ^ porosity,
                  Ks ^ (1.0f0 - porosity) * Ki ^ (porosity - Wu) * Kw ^ Wu)

    # 6. Effective saturation parameter (Ke)
    Ke = ifelse(Wu == moist,
                0.7f0 * log10(max(Sr, 1f-10)) + 1.0f0,
                Sr)

    # 7. Final Kappa Calculation
    # If moist > 0, interpolate. Else Kdry.
    term_moist = (Ksat - Kdry) * Ke + Kdry
    kappa = ifelse(moist > 0.0f0,
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

        cs_array = 2.0f6 * (bulk_dens / soil_dens) * (1.0f0 - organic_frac) +
                   2.7f6 * (bulk_dens / soil_dens) * organic_frac +
                   4.2f6 * (soil_moisture / rho_w) +
                   1.9f6 * ice_frac +
                   1.3f3 * (1.0f0 - ((bulk_dens / soil_dens) + (soil_moisture / rho_w) + ice_frac))
    end

    return nothing
end

function calculate_gsm_inv(soil_moisture, soil_moisture_critical, wilting_point)
    ## Initialize gsm_inv to zeros (handles full stress case: soil_moisture < wilting_point)
    # println("soil_moisture shape: ", size(soil_moisture))

    gsm_inv = CUDA.zeros(eltype(soil_moisture), size(soil_moisture,1), size(soil_moisture,2), size(soil_moisture,3) )
    # println("gsm_inv shape: ", size(gsm_inv))
    
    # Calculate the partial stress term for all elements
    partial_stress = (soil_moisture .- wilting_point) ./ (soil_moisture_critical .- wilting_point)

    # Use ifelse to handle the two remaining cases:
    # - Case 1: No stress (soil_moisture >= soil_moisture_critical) -> 1
    # - Case 2: Partial stress (wilting_point <= soil_moisture < soil_moisture_critical) -> partial_stress
    # - Case 3: Anything still zero is implicitly soil_moisture < wilting_point

    gsm_inv .= ifelse.(soil_moisture .>= soil_moisture_critical,
                      1.0,
                      partial_stress)

    return gsm_inv
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
    term2 = Ksat ./ (denom .^ expt) .* (1 .- expt)   # <-- keep this as-is
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
    T = eltype(W)
    EPS = T(1e-9)
    frac = clamp.((W .- Wres) ./ max.(Wmax .- Wres, EPS), T(0), T(1))
    Ws_eff = Ws  # Ws is fraction of effective capacity, but VIC uses it directly on effective frac

    bf = ifelse.(frac .<= Ws_eff,
        Dsmax .* (Ds ./ Ws_eff) .* frac,  # Linear: add / Ws for correct slope
        Dsmax .* Ds + Dsmax .* (T(1) .- Ds) .* ((frac .- Ws_eff) ./ (T(1) .- Ws_eff)) .^ cexp  # Nonlinear: remove * Ws, as it's Dsmax * Ds start point
    )
    return max.(bf, T(0))
end


function calculate_infiltration!(infiltration, throughfall, surface_runoff)
    # Logic: infiltration = max(sum(throughfall) - runoff, 0)
    
    # 1. Initialize with negative runoff
    # This sets the baseline: we subtract runoff from the inputs.
    @. infiltration = -surface_runoff
    
    # 2. Accumulate Throughfall (Sum over tiles)
    # We iterate tiles to fuse the "sum_with_nan_handling" logic without allocations.
    n_tiles = size(throughfall, 4)
    
    for i in 1:n_tiles
        thr_i = @view throughfall[:, :, :, i]
        
        # Accumulate valid values only (NaN -> 0.0f0)
        # This acts as the "Total Input" summation
        @. infiltration += ifelse(isnan(thr_i), 0.0f0, thr_i)
    end

    # 3. Clamp to zero
    # If runoff > input (conceptually impossible but good for safety), clamp to 0.
    @. infiltration = max(infiltration, 0.0f0)

    return nothing
end

function solve_runoff_and_drainage!(
    soil_moisture_new,   # (ny,nx,nlayer) [Mutated Output]
    subsurface_runoff,   # (ny,nx)        [Mutated Output]
    interlayer_drainage, # (ny,nx,2)      [Mutated Output]
    surface_inflow,      # (ny,nx)
    soil_evaporation,    # (ny,nx)
    transpiration,       # (ny,nx,nlayer, ...) OR (ny,nx,1, ...)
    soil_moisture_old,   # (ny,nx,nlayer)
    soil_moisture_max,   
    ksat,           
    residual_moisture,
    expt,           
    Dsmax, Ds, Ws, c_expt
)
    # 1. Initialize New State
    copyto!(soil_moisture_new, soil_moisture_old)

    # Constants
    zero_val = 0.0f0
    one_val  = 1.0f0
    epsilon  = 1.0f-9

    # --- HANDLE TRANSPIRATION INPUT DIMS ---
    # Check if we have 3 layers of transpiration or just 1 (total)
    has_layers = size(transpiration, 3) >= 3

    # Define views for T (Transpiration)
    # If 1 layer provided, all T is removed from Layer 1. L2 and L3 get 0.
    trans_L1 = @view(transpiration[:, :, 1])
    
    # We use a scalar 0.0f0 for L2/L3 if data is missing; broadcasting handles this efficiently.
    trans_L2 = has_layers ? @view(transpiration[:, :, 2]) : zero_val
    trans_L3 = has_layers ? @view(transpiration[:, :, 3]) : zero_val

    # ========================================================================
    # LAYER 1: Inflow + ET + Drainage to Layer 2
    # ========================================================================
    
    sm_L1    = @view(soil_moisture_new[:, :, 1])
    resid_L1 = @view(residual_moisture[:, :, 1])
    max_L1   = @view(soil_moisture_max[:, :, 1])
    drain_L1 = @view(interlayer_drainage[:, :, 1]) 

    # 1.1 Inflow
    @. sm_L1 += surface_inflow

    # 1.2 Evapotranspiration Removal
    # We use trans_L1 here
    @. sm_L1 -= (soil_evaporation + trans_L1) * min(one_val, max(sm_L1 - resid_L1, zero_val) / max(soil_evaporation + trans_L1, epsilon))

    # 1.3 Drainage (L1 -> L2)
    drain_pot = calculate_interlayer_drainage(
        @view(ksat[:,:,1]), sm_L1, max_L1, resid_L1, @view(expt[:,:,1])
    )
    
    @. drain_L1 = min(drain_pot, max(sm_L1 - resid_L1, zero_val))
    @. sm_L1 -= drain_L1
    
    spill = max.(sm_L1 .- max_L1, zero_val)
    @. sm_L1 -= spill
    @. drain_L1 += spill 

    # ========================================================================
    # LAYER 2: Inflow from L1 + ET + Drainage to Layer 3
    # ========================================================================
    
    sm_L2    = @view(soil_moisture_new[:, :, 2])
    resid_L2 = @view(residual_moisture[:, :, 2])
    max_L2   = @view(soil_moisture_max[:, :, 2])
    drain_L2 = @view(interlayer_drainage[:, :, 2]) 

    # 2.1 Inflow from Layer 1
    @. sm_L2 += drain_L1

    # 2.2 Transpiration Removal
    # Uses trans_L2 (which is either a view or 0.0f0)
    @. sm_L2 -= trans_L2 * min(one_val, max(sm_L2 - resid_L2, zero_val) / max(trans_L2, epsilon))

    # 2.3 Drainage (L2 -> L3)
    drain_pot_2 = calculate_interlayer_drainage(
        @view(ksat[:,:,2]), sm_L2, max_L2, resid_L2, @view(expt[:,:,2])
    )
    
    @. drain_L2 = min(drain_pot_2, max(sm_L2 - resid_L2, zero_val))
    @. sm_L2 -= drain_L2
    
    spill_2 = max.(sm_L2 .- max_L2, zero_val)
    @. sm_L2 -= spill_2
    @. drain_L2 += spill_2

    # ========================================================================
    # LAYER 3: Inflow from L2 + ET + Baseflow
    # ========================================================================
    
    sm_L3    = @view(soil_moisture_new[:, :, 3])
    resid_L3 = @view(residual_moisture[:, :, 3])
    max_L3   = @view(soil_moisture_max[:, :, 3])

    # 3.1 Inflow from Layer 2
    @. sm_L3 += drain_L2

    # 3.2 Transpiration Removal
    # Uses trans_L3 (which is either a view or 0.0f0)
    @. sm_L3 -= trans_L3 * min(one_val, max(sm_L3 - resid_L3, zero_val) / max(trans_L3, epsilon))

    # 3.3 Baseflow
    baseflow_pot = calculate_baseflow(
        sm_L3, resid_L3, max_L3, Dsmax, Ds, Ws, c_expt
    )
    
    @. subsurface_runoff = min(baseflow_pot, max(sm_L3 - resid_L3, zero_val))
    @. sm_L3 -= subsurface_runoff

    # 3.4 Deep Drainage
    deep_drain = max.(sm_L3 .- max_L3, zero_val)
    @. sm_L3 -= deep_drain
    @. subsurface_runoff += deep_drain

    return nothing
end