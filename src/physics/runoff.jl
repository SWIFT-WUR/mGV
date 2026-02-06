function calculate_surface_runoff!(surface_runoff, A_sat, prec_gpu, throughfall, soil_moisture, soil_moisture_max, b_i, cv_gpu)
    # Define epsilon as Float32
    EPS = 1f-9
    
    # --- 1. Calculate Topsoil Moisture ---
    # Optimization: Use views to sum layers 1 & 2 without creating a slice copy.
    topsoil_moisture = @view(soil_moisture[:, :, 1]) .+ @view(soil_moisture[:, :, 2])
    topsoil_moisture_max = @view(soil_moisture_max[:, :, 1]) .+ @view(soil_moisture_max[:, :, 2])
    
    # Clamp topsoil_moisture (in-place update of the temporary array)
    topsoil_moisture .= min.(topsoil_moisture, topsoil_moisture_max)
    
    # --- 2. Calculate Saturation Area (A_sat) ---
    # Calculate ratio
    ratio = topsoil_moisture ./ (topsoil_moisture_max .+ EPS)
    clamp!(ratio, 0.0f0, 1.0f0) 
    
    # Calculate ex parameter
    ex = b_i ./ (1.0f0 .+ b_i)
    
    # Update A_sat IN-PLACE
    @. A_sat = 1.0f0 - ((1.0f0 - ratio) ^ ex)
    
    # --- 3. Calculate Infiltration Parameters ---
    # Maximum infiltration capacity
    max_infil = (1.0f0 .+ b_i) .* topsoil_moisture_max
    
    # Initial infiltration
    i_0 = max_infil .* (1.0f0 .- ((1.0f0 .- A_sat) .^ (1.0f0 ./ b_i)))
    
    # Total water input (inflow)
    # Assumes sum_with_nan_handling returns a compatible type (likely Float32 if inputs are GPU)
    inflow = sum_with_nan_handling(throughfall, 4)
    
    # --- 4. Calculate Runoff ---
    
    max_infil_safe = max.(max_infil, EPS)
    basis = 1.0f0 .- (i_0 .+ inflow) ./ max_infil_safe
    
    # [cite_start]Update surface_runoff IN-PLACE using the conditional logic [cite: 61, 65, 66]
    @. surface_runoff = ifelse(inflow <= EPS,
            0.0f0,                                    # inflow == 0
            ifelse(max_infil <= EPS,
                inflow,                               # max_infil == 0
                ifelse((i_0 + inflow) > max_infil,
                        inflow - topsoil_moisture_max + topsoil_moisture, # Eq 3a
                        inflow - topsoil_moisture_max + topsoil_moisture + 
                        topsoil_moisture_max * (basis ^ (1.0f0 + b_i))    # Eq 3b
                )
            )
    )
    
    # Final Clamp IN-PLACE
    clamp!(surface_runoff, 0.0f0, Inf32)  # Ensure non-negative
    @. surface_runoff = min(surface_runoff, inflow) # Ensure <= inflow

    return nothing
end


# Eq. (24): Total runoff
function calculate_total_runoff!(total_runoff, surface_runoff, subsurface_runoff, fillvalue_threshold)
    
    # 1. Clean Surface Runoff (Mutates input directly, matching original logic)
    # We use Float32 literals (0.0f0)
    @. surface_runoff = ifelse(isnan(surface_runoff) | (abs(surface_runoff) > fillvalue_threshold), 0.0f0, surface_runoff)

    # 2. Clean Subsurface Runoff (Mutates input directly)
    @. subsurface_runoff = ifelse(isnan(subsurface_runoff) | (abs(subsurface_runoff) > fillvalue_threshold), 0.0f0, subsurface_runoff)

    # 3. Compute Total Runoff (Writes to total_runoff)
    # Simple addition: total = surface + subsurface
    @. total_runoff = surface_runoff + subsurface_runoff

    return nothing
end