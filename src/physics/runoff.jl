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
        eps  = ft(1e-9)
        one  = ft(1.0)
        zero = ft(0.0)

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