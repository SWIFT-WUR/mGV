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

function solve_runoff_and_drainage(
    surface_inflow,      # (ny,nx)  [mm]
    soil_evaporation,    # (ny,nx,nlayer,nveg)  [mm]
    transpiration,       # (ny,nx,nlayer,nveg)  [mm]
    soil_moisture_old,   # (ny,nx,nlayer)       [mm]
    soil_moisture_max,   # (ny,nx,nlayer)       [mm]
    ksat_gpu,            # (ny,nx,nlayer)
    residual_moisture,   # (ny,nx,nlayer)
    expt_gpu,            # (ny,nx,nlayer)
    Dsmax_gpu, Ds_gpu, Ws_gpu, c_expt_gpu       # (ny,nx)
)

    # Constants (match Float32 if needed)
    zero_val = zero(eltype(soil_moisture_old))
    one_val = one(eltype(soil_moisture_old))
    epsilon = convert(eltype(soil_moisture_old), 1e-9)

    # Input arrays (GPU-safe copy)
    soil_moisture = copy(soil_moisture_old)
    soil_moisture_max = soil_moisture_max
    residual_moisture = residual_moisture
    ksat = ksat_gpu
    expt = expt_gpu
    Dsmax = Dsmax_gpu
    Ds = Ds_gpu
    Ws = Ws_gpu
    c_expt = c_expt_gpu

    inflow = surface_inflow

    # Aggregate ET per layer (sum over veg; soil evap only on layer 1; GPU-safe with dropdims)
    layer_soil_evap = dropdims(sum(soil_evaporation[:, :, 1, :], dims=3), dims=3)  # (ny,nx)
    layer_transp = CUDA.zeros(eltype(soil_moisture), size(soil_moisture))  # (ny,nx,nlayer)
    for l in 1:1#1:3
        layer_transp[:, :, l] .= dropdims(sum(transpiration[:, :, l, :], dims=3), dims=3)
    end

    # Effective ET outputs (unused in return but kept for consistency; GPU-safe)
    effective_soil_evap = CUDA.zeros(eltype(inflow), size(inflow))
    effective_transp = CUDA.zeros(eltype(layer_transp), size(layer_transp))

    # Interlayer drainage (2 fluxes for 3 layers; GPU-safe)
    interlayer_drainage = CUDA.zeros(eltype(soil_moisture), size(soil_moisture)[1:2]..., 2)

    # -------- Layer 1: Inflow + ET + Drainage to Layer 2 --------
    soil_moisture[:, :, 1] .+= inflow

    available = max.(soil_moisture[:, :, 1] .- residual_moisture[:, :, 1], zero_val)
    potential_loss = layer_soil_evap .+ layer_transp[:, :, 1]
    scale_factor = min.(one_val, available ./ max.(potential_loss, epsilon))
    actual_soil_evap = layer_soil_evap .* scale_factor
    actual_transp = layer_transp[:, :, 1] .* scale_factor
    soil_moisture[:, :, 1] .-= (actual_soil_evap .+ actual_transp)

    effective_soil_evap .= actual_soil_evap
    effective_transp[:, :, 1] .= actual_transp

    # Drainage from layer 1
    drainage_1 = calculate_interlayer_drainage(
        ksat[:, :, 1], soil_moisture[:, :, 1], soil_moisture_max[:, :, 1],
        residual_moisture[:, :, 1], expt[:, :, 1]
    )
    available = max.(soil_moisture[:, :, 1] .- residual_moisture[:, :, 1], zero_val)
    drainage_1 = min.(drainage_1, available)
    soil_moisture[:, :, 1] .-= drainage_1

    # Spillover (saturation excess, added to drainage)
    spillover_1 = max.(soil_moisture[:, :, 1] .- soil_moisture_max[:, :, 1], zero_val)
    soil_moisture[:, :, 1] .-= spillover_1
    interlayer_drainage[:, :, 1] .= drainage_1 .+ spillover_1

    # -------- Layer 2: Drainage In + ET + Drainage to Layer 3 --------
    inflow_to_2 = interlayer_drainage[:, :, 1]
    soil_moisture[:, :, 2] .+= inflow_to_2

    available = max.(soil_moisture[:, :, 2] .- residual_moisture[:, :, 2], zero_val)
    potential_loss = layer_transp[:, :, 2]
    scale_factor = min.(one_val, available ./ max.(potential_loss, epsilon))
    actual_transp = layer_transp[:, :, 2] .* scale_factor
    soil_moisture[:, :, 2] .-= actual_transp

    effective_transp[:, :, 2] .= actual_transp

    # Drainage from layer 2
    drainage_2 = calculate_interlayer_drainage(
        ksat[:, :, 2], soil_moisture[:, :, 2], soil_moisture_max[:, :, 2],
        residual_moisture[:, :, 2], expt[:, :, 2]
    )
    available = max.(soil_moisture[:, :, 2] .- residual_moisture[:, :, 2], zero_val)
    drainage_2 = min.(drainage_2, available)
    soil_moisture[:, :, 2] .-= drainage_2

    # Spillover (added to drainage)
    spillover_2 = max.(soil_moisture[:, :, 2] .- soil_moisture_max[:, :, 2], zero_val)
    soil_moisture[:, :, 2] .-= spillover_2
    interlayer_drainage[:, :, 2] .= drainage_2 .+ spillover_2

    # -------- Layer 3: Drainage In + ET + Baseflow + Deep Drainage --------
    inflow_to_3 = interlayer_drainage[:, :, 2]
    soil_moisture[:, :, 3] .+= inflow_to_3

    available = max.(soil_moisture[:, :, 3] .- residual_moisture[:, :, 3], zero_val)
    potential_loss = layer_transp[:, :, 3]
    scale_factor = min.(one_val, available ./ max.(potential_loss, epsilon))
    actual_transp = layer_transp[:, :, 3] .* scale_factor
    soil_moisture[:, :, 3] .-= actual_transp

    effective_transp[:, :, 3] .= actual_transp

    # Baseflow
    potential_baseflow = calculate_baseflow(
        soil_moisture[:, :, 3], residual_moisture[:, :, 3], soil_moisture_max[:, :, 3],
        Dsmax, Ds, Ws, c_expt
    )
    available = max.(soil_moisture[:, :, 3] .- residual_moisture[:, :, 3], zero_val)
    actual_baseflow = min.(potential_baseflow, available)
    soil_moisture[:, :, 3] .-= actual_baseflow

    # Deep drainage (spillover to groundwater)
    deep_drainage = max.(soil_moisture[:, :, 3] .- soil_moisture_max[:, :, 3], zero_val)
    soil_moisture[:, :, 3] .-= deep_drainage

    # Total baseflow (includes deep)
    total_baseflow = actual_baseflow .+ deep_drainage

    return soil_moisture, total_baseflow, interlayer_drainage
end