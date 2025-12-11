function soil_conductivity(moist, ice_frac, soil_dens_min, bulk_dens_min, quartz, organic_frac, porosity)
    # Get the element type from the input arrays
    T = eltype(moist)
    
    # Unfrozen water content
    Wu = moist .- ice_frac

    # Calculate dry conductivity as a weighted average of mineral and organic fractions, constants Reference: Farouki, O.T., "Thermal Properties of Soils" 1986
    Kdry_min = (T(0.135) .* bulk_dens_min .+ T(64.7)) ./ (soil_dens_min .- T(0.947) .* bulk_dens_min)
    Kdry = (T(1.0) .- organic_frac) .* Kdry_min .+ organic_frac .* T(Kdry_org)

    # Fractional degree of saturation
    Sr = ifelse.(porosity .> T(0.0), moist ./ porosity, T(0.0))

    # Compute Ks of mineral soil based on quartz content
    Ks_min = ifelse.((quartz .< T(0.2)) .& (quartz .<= T(1.0)),
                    T(7.7) .^ quartz .* T(3.0) .^ (T(1.0) .- quartz),
                    ifelse.(quartz .<= T(1.0),
                            T(7.7) .^ quartz .* T(2.2) .^ (T(1.0) .- quartz),
                            T(0.0)))

    Ks = (T(1.0) .- organic_frac) .* Ks_min .+ organic_frac .* T(Ks_org)

    # Calculate Ksat depending on whether the soil is unfrozen (Wu == moist) or partially frozen
    Ksat = ifelse.(Wu .== moist,
                  Ks .^ (T(1.0) .- porosity) .* T(Kw) .^ porosity,
                  Ks .^ (T(1.0) .- porosity) .* T(Ki) .^ (porosity .- Wu) .* T(Kw) .^ Wu)

    # Compute the effective saturation parameter, Ke
    Ke = ifelse.(Wu .== moist,
                T(0.7) .* log10.(max.(Sr, T(1e-10))) .+ T(1.0),
                Sr)

    # Final Kappa calculation using ifelse to handle moist > 0 condition
    Kappa = ifelse.(moist .> T(0.0),
                   max.((Ksat .- Kdry) .* Ke .+ Kdry, Kdry),
                   Kdry)

    return Kappa
end

function volumetric_heat_capacity(soil_fract, water_fract, ice_fract, organic_frac)
    # Constant values are volumetric heat capacities in J/m^3/K
    Cs = 2.0e6 .* soil_fract .* (1 .- organic_frac) .+
         2.7e6 .* soil_fract .* organic_frac .+
         4.2e6 .* water_fract .+
         1.9e6 .* ice_fract .+
         1.3e3 .* (1.0 .- (soil_fract .+ water_fract .+ ice_fract))  # Air component

    return Cs
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