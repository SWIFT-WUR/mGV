@kernel function surface_runoff_kernel!(
    surface_runoff,
    A_sat,
    soil_moisture,
    soil_moisture_max,
    b_i_grid,
    throughfall,
    cv_grid
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
        # Sum 4D throughfall (nx, ny, 1, n_veg) -> Scalar Inflow
        # We loop over dim 4 manually to avoid allocating a reduction array
        inflow_sum = zero
        n_veg = size(throughfall, 4)
        
        for k in 1:n_veg
            val = throughfall[i,j,1,k] * cv_grid[i,j,1,k]
            # Handle NaN check inline
            if !isnan(val)
                inflow_sum += val
            end
        end

        # --- 5. Runoff Calculation ---
        # Max Infiltration: (1 + b) * W_max
        max_infil = (one + b) * top_max
        
        # i_0 = max_infil * (1 - (1 - A_sat)^(1/b))
        term_asat = max(one - asat_val, zero)
        pow_b = one / max(b, eps)
        i_0 = max_infil * (one - (term_asat ^ pow_b))
        
        # --- 6. VIC Runoff Logic ---
        runoff = zero
        
        if inflow_sum <= eps
            runoff = zero
        elseif max_infil <= eps
            runoff = inflow_sum
        elseif (i_0 + inflow_sum) > max_infil
            # Eq 3a: Saturation Excess (Soil fills up)
            runoff = (inflow_sum - top_max) + top_sm
        else
            # Eq 3b: Infiltration Excess (Rain > Infil Rate)
            max_infil_safe = max(max_infil, eps)
            basis = one - (i_0 + inflow_sum) / max_infil_safe
            basis = max(basis, zero) 
            
            # Runoff = Inflow - Delta_Storage + Excess_Term
            runoff = (inflow_sum - top_max + top_sm) + (top_max * (basis ^ (one + b)))
        end
        
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
    b_i, cv_grid
)

    kernel_launcher! = surface_runoff_kernel!(device_backend)    
    nx, ny = size(surface_runoff)
    
    kernel_launcher!(
        surface_runoff, A_sat, 
        soil_moisture, soil_moisture_max, 
        b_i, throughfall, cv_grid;
        ndrange = (nx, ny)
    )

    return nothing
end

function calculate_total_runoff!(total_runoff, surface_runoff, subsurface_runoff)
    
    @. total_runoff = surface_runoff + subsurface_runoff

    return nothing
end