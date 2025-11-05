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



#function calculate_interlayer_drainage(Ksat, current_moisture, max_moisture, residual_moisture, expt)
#    # VIC's exact calc_Q12 formula - keep same parameter names!
#    eff_init = current_moisture .- residual_moisture
#    eff_max = max_moisture .- residual_moisture
#    
#    # Avoid division by zero
#    if any(eff_max .<= 0) || any(expt .== 1.0)
#        return CUDA.zeros(eltype(current_moisture), size(current_moisture))
#    end
#    
#    term1 = eff_init .^ (1.0 .- expt)
#    term2 = Ksat ./ (eff_max .^ expt) .* (1.0 .- expt)
#    inner = max.(term1 .- term2, 1e-10)  # Prevent negative
#    
#    final_eff = inner .^ (1.0 ./ (1.0 .- expt))
#    Q12 = eff_init .- final_eff
#    
#    return max.(Q12, 0.0)
#end

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
function calculate_baseflow(W, Wres, Wc, Dsmax, Ds, Ws, c_exp)
    # Work with absolute storages; Ws is a fraction in (0,1)
    WsWc  = Ws .* Wc                  # threshold storage
    term1 = (Ds .* Dsmax) .* (W ./ max.(WsWc, eps(eltype(W))))  # linear part

    # Below threshold: purely linear
    Qb_lin = term1

    # Above threshold: add nonlinear part
    num   = max.(W .- WsWc, 0)
    den   = max.(Wc .- WsWc, eps(eltype(W)))
    nonlin = (Dsmax .- (Ds .* Dsmax) ./ Ws) .* (num ./ den) .^ c_exp

    Qb = ifelse.(W .<= WsWc, Qb_lin, term1 .+ nonlin)
    # Do not withdraw more than available above residual
    avail = max.(W .- Wres, 0)
    return clamp.(Qb, 0, avail)
end



"""
Solves the runoff and drainage for a multi-layered soil column.

This function calculates the movement of water between soil layers (drainage),
the runoff from the surface, and the baseflow from the bottom layer. It then
updates the soil moisture for each layer based on a water balance that now
correctly includes inflows (surface water, drainage from above) and outflows
(bare soil evaporation, plant transpiration, drainage to below, baseflow).

Args:
    surface_inflow: Water reaching the soil surface (2D array).
    soil_evaporation: Evaporation from bare soil in each layer (3D array).
                      NOTE: This is assumed to only have non-zero values in the first layer.
    transpiration: Water uptake by plants from each layer (3D array).
                   NOTE: This array dictates which layers lose water to transpiration.
    soil_moisture_old: Moisture from the previous time step (3D array).
    Wfc_gpu: Field capacity of each layer (3D array).
    soil_moisture_max: Maximum moisture (porosity) of each layer (3D array).
    ksat_gpu: Saturated hydraulic conductivity of each layer (3D array).
    residual_moisture: Residual moisture of each layer (3D array).
    expt_gpu: Brooks-Corey exponent for each layer (3D array).
    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu: ARNO baseflow parameters (2D arrays).

Returns:
    A tuple containing:
    - soil_moisture_new: The updated soil moisture for each layer (3D array).
    - baseflow: The calculated baseflow from the bottom layer (2D array).
    - Q12: The drainage flux between layers (3D array).
"""
function solve_runoff_and_drainage(
    surface_inflow,      # (ny,nx)  [mm]
    soil_evaporation,    # (ny,nx,nlayer,nveg)  [mm]
    transpiration,       # (ny,nx,nlayer,nveg)  [mm]
    soil_moisture_old,   # (ny,nx,nlayer)       [mm]
    soil_moisture_max,   # (ny,nx,nlayer)       [mm]
    ksat_gpu,            # (ny,nx,nlayer)
    residual_moisture,   # (ny,nx,nlayer)
    expt_gpu,            # (ny,nx,nlayer)
    cv_gpu,
    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu       # (ny,nx)
)

    T  = eltype(soil_moisture_old)
    Z  = T(0); O = T(1); EPS = T(1e-9)

    W_old  = T.(soil_moisture_old)
    Wmax   = T.(soil_moisture_max)
    Wres   = T.(residual_moisture)
    Ksat   = T.(ksat_gpu)
    exptT  = T.(expt_gpu)
    DsmaxT = T.(Dsmax_gpu); DsT = T.(Ds_gpu)
    WsT    = T.(Ws_gpu);    cexpT = T.(c_expt_gpu)

    inflow = T.(surface_inflow)
    L      = size(W_old, 3)
    saturation_runoff = CUDA.zeros(T, size(inflow))

    # Aggregate E and T per layer (sum over veg)
    EvL = CUDA.zeros(T, size(W_old))
    TrL = CUDA.zeros(T, size(W_old))
    nTL = min(L, size(transpiration,3))
    println("nTL shape !!!!!!!!!!!!!!!!!: ", nTL)

    TrL[:, :, 1:nTL] .= sum(T.(transpiration), dims=4)[:, :, 1:nTL]
    #TrL[:, :, 1:nTL] .= sum(T.(cv_gpu[:, :, :, 1:end-1]) .* T.(transpiration[:, :, :, 1:end-1]), dims=4)[:, :, 1:nTL, 1]

    EvL[:, :, 1]     .= sum(T.(soil_evaporation), dims=(3,4)) 

    Ev1_eff = CUDA.zeros(T, size(inflow))
    TrL_eff = CUDA.zeros(T, size(W_old))

    W = similar(W_old)
    W .= W_old
    Q12 = CUDA.zeros(T, size(W,1), size(W,2), max(L-1,0))

    # -------- LAYER 1 --------
    W[:, :, 1] .= W[:, :, 1] .+ inflow

    avail1     = max.(W[:, :, 1] .- Wres[:, :, 1], Z)
    loss1_pot  = EvL[:, :, 1] .+ TrL[:, :, 1]
    scale1     = min.(O, avail1 ./ max.(loss1_pot, EPS))
    Ev1        = EvL[:, :, 1] .* scale1
    Tr1        = TrL[:, :, 1] .* scale1
    W[:, :, 1] .= W[:, :, 1] .-(Ev1 .+ Tr1)

    Ev1_eff          .= Ev1
    TrL_eff[:, :, 1] .= Tr1

    if L >= 2
        q12_1 = calculate_interlayer_drainage(
                    Ksat[:, :, 1], W[:, :, 1], Wmax[:, :, 1], Wres[:, :, 1], exptT[:, :, 1]
                )
        avail1 = max.(W[:, :, 1] .- Wres[:, :, 1], Z)
        q12_1  = min.(q12_1, avail1)
        W[:, :, 1] .= W[:, :, 1] .- q12_1

        spill1 = max.(W[:, :, 1] .- Wmax[:, :, 1], Z)
        W[:, :, 1] .= W[:, :, 1] .- spill1
        #saturation_runoff .= saturation_runoff .+ spill1

        Q12[:, :, 1] .= q12_1 .+ spill1
    end

    # -------- INTERIOR LAYERS (2..L-1) --------
    for l in 2:max(L-1, 1)
        in_l = (l == 2 ? Q12[:, :, 1] : Q12[:, :, l-1])
        W[:, :, l] .= W[:, :, l] .+ in_l

        avail   = max.(W[:, :, l] .- Wres[:, :, l], Z)
        loss_p  = TrL[:, :, l]
        scale   = min.(O, avail ./ max.(loss_p, EPS))
        Trl     = loss_p .* scale
        W[:, :, l] .= W[:, :, l] .- Trl
        TrL_eff[:, :, l] .= Trl

        q12_l = calculate_interlayer_drainage(
                    Ksat[:, :, l], W[:, :, l], Wmax[:, :, l], Wres[:, :, l], exptT[:, :, l]
                )
        avail  = max.(W[:, :, l] .- Wres[:, :, l], Z)
        q12_l  = min.(q12_l, avail)
        W[:, :, l] .= W[:, :, l] .- q12_l

        spill  = max.(W[:, :, l] .- Wmax[:, :, l], Z)
        W[:, :, l] .= W[:, :, l] .- spill
        q12_l  .= q12_l .+ spill

        Q12[:, :, l] .= q12_l
    end

    # -------- BOTTOM LAYER --------
    if L == 1
        Wpre   = W[:, :, 1]
        bf_pot = calculate_baseflow(Wpre, Wres[:, :, 1], Wmax[:, :, 1], DsmaxT, DsT, WsT, cexpT)
        avail  = max.(Wpre .- Wres[:, :, 1], Z)
        bf     = min.(bf_pot, avail)
        W[:, :, 1] .= Wpre .- bf
        baseflow = bf
    else
        inN = Q12[:, :, L-1]
        W[:, :, L] .= W[:, :, L] .+ inN

        avail    = max.(W[:, :, L] .- Wres[:, :, L], Z)
        loss_p   = TrL[:, :, L]
        scale    = min.(O, avail ./ max.(loss_p, EPS))
        TrL_bot  = loss_p .* scale
        W[:, :, L] .= W[:, :, L] .- TrL_bot
        TrL_eff[:, :, L] .= TrL_bot

        bf_pot = calculate_baseflow(W[:, :, L], Wres[:, :, L], Wmax[:, :, L], DsmaxT, DsT, WsT, cexpT)
        avail  = max.(W[:, :, L] .- Wres[:, :, L], Z)
        bf     = min.(bf_pot, avail)
        W[:, :, L] .= W[:, :, L] .- bf

        deep    = max.(W[:, :, L] .- Wmax[:, :, L], Z)
        W[:, :, L] .= W[:, :, L] .- deep
        baseflow = bf .+ deep
    end


    return W, baseflow, Q12
end