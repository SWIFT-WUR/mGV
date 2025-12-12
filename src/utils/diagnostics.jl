# Diagnostic utilities for debugging and analyzing simulation outputs.

function run_external_debug(day, g_sw_1, g_sw_2, root_gpu, transpiration)
    println("\n=== EXTERNAL DEBUG - Day $day ===")
    
    i, j, veg = 7, 19, 1
    
    g1_val = Array(g_sw_1)[i, j]
    g2_val = Array(g_sw_2)[i, j]
    f1_val = Array(root_gpu)[i, j, 1, veg]
    f2_val = Array(root_gpu)[i, j, 2, veg]
    transp_val = Array(transpiration)[i, j, 1, veg]
    
    layer2_should_dominate = (g2_val >= 0.99) && (f2_val >= 0.5)
    layer1_should_dominate = (g1_val >= 0.99) && (f1_val >= 0.5) && !layer2_should_dominate
    
    expected_gsw = if layer2_should_dominate
        1.0
    elseif layer1_should_dominate
        1.0
    else
        (f1_val * g1_val + f2_val * g2_val) / (f1_val + f2_val)
    end
    
    println("Soil stress: g1=$g1_val, g2=$g2_val")
    println("Root fractions: f1=$f1_val, f2=$f2_val")
    println("Layer 2 should dominate: $layer2_should_dominate")
    println("Layer 1 should dominate: $layer1_should_dominate")
    println("Expected g_sw: $expected_gsw")
    println("Final transpiration: $transp_val mm/day")
    
    println("=== END EXTERNAL DEBUG ===\n")
end