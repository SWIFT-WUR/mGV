@kernel function surface_runoff_kernel!(
    surface_runoff,
    A_sat,
    soil_moisture,
    soil_moisture_max,
    b_i_grid,
    throughfall,
    cv_grid,
    AreaFract
)
    i, j = @index(Global, NTuple)

    # Boundary check
    if i <= size(surface_runoff, 1) && j <= size(surface_runoff, 2)
        
        # Constants
        eps  = 1f-9
        one  = 1f0
        zero = 0f0

        # --- 1. Topsoil Moisture (Sum Layers 1 + 2) ---
        # Load directly from global memory to register
        sm1 = soil_moisture[i,j,1]
        sm2 = soil_moisture[i,j,2]
        
        max1 = soil_moisture_max[i,j,1]
        max2 = soil_moisture_max[i,j,2]
        
        top_max = max1 + max2
        top_sm  = min(sm1 + sm2, top_max) # Clamp inplace logic
        
        # --- 2. Infiltration Shape Parameter ---
        b = b_i_grid[i,j]
        
        # --- 3. A_sat (Saturated Area Fraction) ---
        # VIC/ARNO curve logic
        ratio = top_sm / max(top_max, eps)
        ratio = clamp(ratio, zero, one)
        
        # ex = b / (1 + b)
        ex_param = b / (one + b)
        
        # A_sat = 1 - (1 - ratio)^ex
        term_ratio = max(one - ratio, zero)
        asat_val = one - (term_ratio ^ ex_param)
        
        # Store A_sat
        A_sat[i,j] = asat_val
        
        # --- 4. Inflow Summation (Reduction) ---
        # Sum 4D throughfall (nx, ny, nbands, n_veg) -> Scalar Inflow
        # We loop over dim 4 manually to avoid allocating a reduction array
        inflow_sum = zero
        n_bands = size(AreaFract, 3)
        n_veg = size(throughfall, 4)
        
        for k in 1:n_veg
            for b in 1:n_bands
                val = throughfall[i,j,b,k] * cv_grid[i,j,1,k] * AreaFract[i,j,b]
                # Branchless NaN guard
                inflow_sum += ifelse(isnan(val), zero, val)
            end
        end

        # --- 5. Runoff Calculation ---
        # Max Infiltration: (1 + b) * W_max
        max_infil = (one + b) * top_max
        
        # i_0 = max_infil * (1 - (1 - A_sat)^(1/b))
        term_asat = max(one - asat_val, zero)
        pow_b = one / max(b, eps)
        i_0 = max_infil * (one - (term_asat ^ pow_b))
        
        # --- 6. VIC Runoff Logic (Branchless Formulation) ---
        # 
        # The VIC model partitions inflow into:
        #   1. Direct runoff from saturated areas
        #   2. Infiltration into unsaturated areas
        #   3. Additional runoff if infiltration exceeds soil capacity
        #
        # Variables:
        #   inflow_sum:     Total water entering the soil
        #   i_0:            Current maximum available infiltration capacity 
        #   max_infil:      Absolute maximum infiltration capacity before saturation
        max_infil_safe = max(max_infil, eps)
        
        # Calculate runoff if the soil becomes FULLY saturated during this timestep
        runoff_full_sat = (inflow_sum - top_max) + top_sm
        
        # Calculate runoff if the soil is PARTIALLY saturated (follows non-linear ARNO curve)
        basis = max(one - (i_0 + inflow_sum) / max_infil_safe, zero)
        runoff_partial_sat = runoff_full_sat + (top_max * (basis ^ (one + b)))
        
        # Determine the physical state of the grid cell
        state_no_inflow       = inflow_sum <= eps
        state_impervious      = (!state_no_inflow) & (max_infil <= eps)
        state_fully_saturated = (!state_no_inflow) & (!state_impervious) & ((i_0 + inflow_sum) > max_infil)
        
        # Branchless selection of the correct runoff calculation
        # This replaces a slow 4-way if/elseif block, preventing warp divergence
        runoff = ifelse(state_no_inflow, zero,
                   ifelse(state_impervious, inflow_sum,
                    ifelse(state_fully_saturated, runoff_full_sat, runoff_partial_sat)))
        
        # Final Clamp
        runoff = clamp(runoff, zero, inflow_sum)
        
        # Store Result
        surface_runoff[i,j] = runoff
    end
end

function calculate_surface_runoff!(
    surface_runoff, A_sat, 
    throughfall, 
    soil_moisture, soil_moisture_max, 
    b_i, cv_grid, AreaFract
)

    kernel_launcher! = surface_runoff_kernel!(device_backend)    
    nx, ny = size(surface_runoff)
    
    kernel_launcher!(
        surface_runoff, A_sat, 
        soil_moisture, soil_moisture_max, 
        b_i, throughfall, cv_grid, AreaFract;
        ndrange = (nx, ny)
    )

    return nothing
end

function calculate_total_runoff!(total_runoff, surface_runoff, subsurface_runoff)
    
    @. total_runoff = surface_runoff + subsurface_runoff

    return nothing
end

function calculate_interlayer_drainage(Ksat, current_moist, max_moist, resid_moist, expt)
    EPS = 1f-6  # limit eps to 1e-6 as 

    m = max(expt, 3.001f0)

    W_m = max(max_moist - resid_moist, EPS)
    W_a = max(current_moist - resid_moist, 0f0)

    F = clamp((W_a / W_m), 0f0, 1f0)

    # F^(1-m) computed as exp((1-m)*log F) so the exponent can be clamped
    #  before overflow, not after.
    exponent = (1f0 - m) * log(F)
    if exponent > 80f0  # term 1 will be too large to compute; interlayer drainage is 0
        return 0f0
    end

    term1 = exp(exponent)
    term2 = (Ksat / W_m) * (1f0 - m)

    inner = max(term1 - term2, EPS)
    W_new = W_m * (inner ^ (1f0 / (1f0 - m)))

    Q12 = W_a - W_new
    return clamp(Q12, 0f0, W_a)
end

# Eq. 21a–21b (Liang 1994)
function calculate_baseflow(W, Wres, Wmax, Dsmax, Ds, Ws, cexp)
    EPS = 1f-9
    eff_max = max.(Wmax .- Wres, EPS)
    rel_moist = clamp.((W .- Wres) ./ eff_max, 0f0, 1f0)
    
    Ws_safe = max.(Ws, EPS)
    Ws_compl = max.(1f0 .- Ws, EPS)
    
    linear_coeff = (Dsmax .* Ds) ./ Ws_safe
    Qb_lin = linear_coeff .* rel_moist
    
    nonlin_amp = Dsmax .* (1f0 .- Ds ./ Ws_safe)
    nonlin_frac = max.(rel_moist .- Ws, 0f0) ./ Ws_compl
    Qb_nonlin = Qb_lin .+ nonlin_amp .* (nonlin_frac .^ cexp)
    
    Qb = ifelse.(rel_moist .<= Ws, Qb_lin, Qb_nonlin)
    
    avail = max.(W .- Wres, 0f0)
    return clamp.(Qb, 0f0, avail)
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
        
        tiny = 1f-9
        zero = 0f0
        one  = 1f0

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
        INV_STEPS = 1f0 / 24f0

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

@kernel function infiltration_kernel!(
    infiltration, throughfall, surface_runoff, cv_grid, area_fraction
)
    i, j = @index(Global, NTuple)

    if i <= size(infiltration, 1) && j <= size(infiltration, 2)
        
        # 1. Initialize accumulator locally
        acc = -surface_runoff[i, j]

        # 2. Integrate all vegetation tiles and snow bands.
        n_bands = size(throughfall, 3)
        n_tiles = size(throughfall, 4)
        for k in 1:n_tiles
            for b in 1:n_bands
                val = (
                    throughfall[i, j, b, k] *
                    cv_grid[i, j, 1, k] *
                    area_fraction[i, j, b]
                )
                acc += ifelse(isnan(val), 0f0, val)
            end
        end

        # 3. Write result
        infiltration[i, j] = acc
    end
end

function calculate_infiltration!(
    infiltration, throughfall, surface_runoff, cv_grid, area_fraction
)

    kernel! = infiltration_kernel!(device_backend)
    nx, ny  = size(infiltration)

    # Launch kernel
    kernel!(
        infiltration, throughfall, surface_runoff, cv_grid, area_fraction;
        ndrange=(nx, ny)
    )

    return nothing
end

function update_total_runoff!(model)
    (; surface_runoff, subsurface_runoff, total_runoff) = model.soil_variables

    @. total_runoff = surface_runoff + subsurface_runoff

    return nothing
end