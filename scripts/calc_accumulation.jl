using NCDatasets

function calculate_and_add_accumulation(filename)
    println("Processing: $filename")
    
    # 1. Open file in "append" mode ("a")
    ds = NCDataset(filename, "a")
    
    # 2. Load Topology and Area
    # These arrays will be of type Union{T, Missing}
    ids_raw    = ds["downstream_id"][:,:]
    target_raw = ds["downstream"][:,:]
    area_raw   = ds["cell_area"][:,:]
    
    nx, ny = size(ids_raw)
    n_total = nx * ny
    
    # Flatten arrays
    ids_flat    = vec(ids_raw)
    target_flat = vec(target_raw)
    area_flat   = vec(area_raw)
    
    # 3. Build Graph (Map ID -> Index)
    println("  -> Building network graph...")
    id_to_idx = Dict{Int32, Int}()
    
    for i in 1:n_total
        # FIX: Check for missing before conversion
        raw_val = ids_flat[i]
        if !ismissing(raw_val)
            val = Int32(raw_val)
            if val > 0 # Valid ID check
                id_to_idx[val] = i
            end
        end
    end
    
    # Build Adjacency
    downstream_idx = fill(0, n_total)
    in_degree      = fill(0, n_total)
    
    for i in 1:n_total
        # FIX: Check if current cell is missing
        if ismissing(ids_flat[i]); continue; end
        
        # FIX: Check if target is missing
        t_val = target_flat[i]
        if ismissing(t_val); continue; end
        
        target_id = Int32(t_val)
        
        # If target exists in our grid, record the connection
        if haskey(id_to_idx, target_id)
            dest = id_to_idx[target_id]
            
            # Prevent self-loops
            if dest != i
                downstream_idx[i] = dest
                in_degree[dest] += 1
            end
        end
    end

    # 4. Calculate Accumulation (Topological Sum)
    println("  -> Summing upstream areas...")
    
    # Initialize accumulation with local area (coalesce missing areas to 0.0)
    accumulation = map(x -> ismissing(x) ? 0.0 : Float64(x), area_flat)
    
    # Queue all headwaters (cells with 0 inflows)
    queue = Int[]
    for i in 1:n_total
        # Only valid cells (not missing IDs) can be headwaters
        if in_degree[i] == 0 && !ismissing(ids_flat[i])
            push!(queue, i)
        end
    end
    
    # Process queue (Kahn's Algorithm)
    processed_count = 0
    while !isempty(queue)
        u = popfirst!(queue) 
        processed_count += 1
        
        v = downstream_idx[u]
        
        if v > 0
            accumulation[v] += accumulation[u]
            
            in_degree[v] -= 1
            if in_degree[v] == 0
                push!(queue, v)
            end
        end
    end
    
    println("  -> Processed $processed_count cells.")
    
    # 5. Write to NetCDF
    acc_grid = reshape(accumulation, nx, ny)
    fill_val = 1.0e20
    
    if haskey(ds, "accumulation")
        println("  -> Overwriting existing 'accumulation' variable.")
        v = ds["accumulation"]
        v[:,:] = acc_grid
    else
        println("  -> Creating new 'accumulation' variable.")
        defVar(ds, "accumulation", acc_grid, ("lon", "lat"), 
               attrib = Dict(
                   "units" => "m2", 
                   "long_name" => "flow accumulation from upstream area",
                   "_FillValue" => fill_val
               ))
    end
    
    close(ds)
    println("Done! Saved to $filename")
end

calculate_and_add_accumulation("./vic_global_5min_routing_param_wbt_f32.nc")
