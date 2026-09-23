using NCDatasets

# 1. Configuration
input_file  = "vic_global_5min_params_fix2.nc"
output_file = "vic_global_5min_params_fix2_f32.nc"

# 2. Conversion Function
function convert_to_float32(in_path, out_path)
    println("Opening $in_path...")
    ds_in = NCDataset(in_path, "r")
    
    if isfile(out_path)
        rm(out_path)
    end
    ds_out = NCDataset(out_path, "c")

    try
        # --- Copy Dimensions ---
        for (dname, dlen) in ds_in.dim
            defDim(ds_out, dname, dlen)
        end

        # --- Copy Global Attributes ---
        for (k, v) in ds_in.attrib
            ds_out.attrib[k] = v
        end

        # --- Copy Variables ---
        for (vname, var) in ds_in
            # FIX: Strip 'Missing' from the type using nonmissingtype()
            orig_type = nonmissingtype(eltype(var))
            
            # DECISION LOGIC: Only change Float64 -> Float32
            if orig_type == Float64
                new_type = Float32
                println("  Converting $vname (Float64 -> Float32)")
            else
                new_type = orig_type
                println("  Keeping    $vname ($orig_type)")
            end

            # Handle Attributes (specifically _FillValue)
            att_dict = Dict(var.attrib)
            
            # If we are converting to Float32, we must also convert the _FillValue
            if haskey(att_dict, "_FillValue") && new_type == Float32
                val = att_dict["_FillValue"]
                # Convert the fill value if it's a number
                if val isa Number
                    att_dict["_FillValue"] = Float32(val)
                end
            end

            # Define Variable in new file (using the concrete new_type)
            v_out = defVar(ds_out, vname, new_type, dimnames(var); attrib=att_dict)

            # Copy Data
            # We load data into memory
            data = var[:]
            
            if new_type != orig_type
                # Convert data elements to Float32, preserving Missing
                # This map handles: Missing -> Missing, Float64 -> Float32
                v_out[:] = map(x -> ismissing(x) ? missing : Float32(x), data)
            else
                # Direct copy for Integers / Strings
                v_out[:] = data
            end
        end
        println("\nSuccess! Output saved to: $out_path")
        
    finally
        close(ds_in)
        close(ds_out)
    end
end

# 3. Run it
convert_to_float32(input_file, output_file)
