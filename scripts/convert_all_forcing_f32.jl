using NCDatasets
using Base.Threads

# --- 1. The Conversion Core ---
function convert_to_float32(in_path, out_path)
    # Skip if output already exists
    if isfile(out_path)
        println("Skipping existing: $out_path")
        return
    end

    println("Processing: $in_path -> $out_path")
    
    ds_in = NCDataset(in_path, "r")
    
    # Create temp file to avoid partial writes on failure
    temp_out = out_path * ".tmp"
    if isfile(temp_out); rm(temp_out); end
    ds_out = NCDataset(temp_out, "c")

    try
        # Copy Dimensions
        for (dname, dlen) in ds_in.dim
            defDim(ds_out, dname, dlen)
        end

        # Copy Global Attributes
        for (k, v) in ds_in.attrib
            ds_out.attrib[k] = v
        end

        # Copy Variables
        for (vname, var) in ds_in
            # --- FIX: Access raw type to avoid DateTime error ---
            # var.var gives access to the raw NetCDF variable (numbers), 
            # ignoring the CF decoding that turns time into DateTime.
            raw_var = var.var 
            orig_type = nonmissingtype(eltype(raw_var))

            # DECISION LOGIC
            if orig_type == Float64
                new_type = Float32
            else
                new_type = orig_type
            end

            # Attributes & FillValue
            att_dict = Dict(var.attrib)
            if haskey(att_dict, "_FillValue") && new_type == Float32
                val = att_dict["_FillValue"]
                if val isa Number
                    att_dict["_FillValue"] = Float32(val)
                end
            end

            # Define Variable
            v_out = defVar(ds_out, vname, new_type, dimnames(var); attrib=att_dict)

            # --- FIX: Copy Data Safely ---
            # If we are just copying (no type change), copy raw to keep it fast.
            # If we are converting, we load the raw data to avoid DateTime conversion issues.
            data = raw_var[:] 
            
            if new_type != orig_type
                # Convert Float64 -> Float32
                v_out[:] = map(x -> ismissing(x) ? missing : Float32(x), data)
            else
                # Copy directly (keeps dates as raw numbers, which is what we want!)
                v_out[:] = data
            end
        end
        
    catch e
        close(ds_out)
        rm(temp_out)
        println("Error processing $in_path: $e")
        # We don't rethrow so other threads can continue
    finally
        close(ds_in)
        close(ds_out)
    end

    # Rename temp file to final name upon success
    if isfile(temp_out)
        mv(temp_out, out_path)
        println("Completed: $out_path")
    end
end

# --- 2. Main Processing Loop ---
function process_forcing_directories()
    # List of directories
    dirs = ["vp", "tair", "psurf", "wind", "lwdown", "swdown", "prec"]
    
    # Collect all tasks
    tasks = []
    
    for d in dirs
        if !isdir(d)
            println("Warning: Directory '$d' not found. Skipping.")
            continue
        end

        files = readdir(d)
        for f in files
            # Filter: Must be .nc and NOT already contain "_f32"
            if endswith(f, ".nc") && !occursin("_f32", f)
                # Parse filename: name_YEAR.nc -> name_f32_YEAR.nc
                m = match(r"^(.*)_(\d{4})\.nc$", f)
                
                if m !== nothing
                    base_name = m.captures[1]
                    year = m.captures[2]
                    new_name = "$(base_name)_f32_$(year).nc"
                    
                    in_path = joinpath(d, f)
                    out_path = joinpath(d, new_name)
                    
                    push!(tasks, (in_path, out_path))
                end
            end
        end
    end

    println("Found $(length(tasks)) files to convert.")
    
    # Process in parallel
    Threads.@threads for (in_p, out_p) in tasks
        convert_to_float32(in_p, out_p)
    end
end

# --- 3. Execute ---
process_forcing_directories()
