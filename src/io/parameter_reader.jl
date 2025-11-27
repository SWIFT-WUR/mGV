function read_and_allocate_parameter(varname::String)
    println("Loading $varname parameter input...")

    # 1) Open netCDF file, read variable into a CPU array and copy array into preload
    dataset       = NetCDF.open(input_param_file)
    # cpu_arr access removed as it wasn't used/needed if we read immediately below
    var_dims      = size(dataset[varname]) 

    # Handle slicing based on dimensionality
    cpu_preload = if length(var_dims) == 1
        dataset[varname][:]
    elseif length(var_dims) == 2
        dataset[varname][:, :]
    elseif length(var_dims) == 3
        dataset[varname][:, :, :]
    elseif length(var_dims) == 4
        dataset[varname][:, :, :, :]
    else
        error("Unsupported variable dimensionality: ", length(var_dims))
    end
    
    # Print info
    if length(var_dims) <= 4
        println("Element type for $(length(var_dims))D: ", eltype(cpu_preload))
    end
    println("Full size of $varname: ", size(cpu_preload))

    # 2) Conditionally allocate a GPU array AND PIN CPU MEMORY
    if GPU_USE
        # --- PINNING OPTIMIZATION ---
        # "Pin" the CPU memory. This prevents the OS from swapping it out 
        # and enables high-speed DMA transfers to the GPU.
        try
            CUDA.Mem.pin(cpu_preload)
            println("  -> CPU memory pinned successfully.")
        catch e
            println("  -> WARNING: Failed to pin CPU memory. Transfer will be slower. Error: $e")
        end
        # ----------------------------

        # Adjust dimensions for the GPU array
        adjusted_dims = if length(var_dims) == 4
            (var_dims[1], var_dims[2], (var_dims[3] == 12 ? 1 : var_dims[3]), var_dims[4])
        else
            var_dims
        end

        gpu_arr = CUDA.zeros(float_type, adjusted_dims...)
        println("Allocated GPU array of size: ", size(gpu_arr))
        return cpu_preload, gpu_arr
    else
        return cpu_preload, nothing
    end
end

# This macro automates the creation of the CPU and GPU variables
macro load_params(vars...)
    quote
        # The `esc(...)` is what allows the macro to create variables
        # in the scope where you call it.
        $(map(vars) do var
            cpu_name = Symbol(String(var), "_cpu")
            gpu_name = Symbol(String(var), "_gpu")
            source_var = Symbol(String(var), "_var")
            
            # This generates the line: (var_cpu, var_gpu) = read_and_allocate_parameter(var_var)
            :(($(esc(cpu_name)), $(esc(gpu_name))) = read_and_allocate_parameter($(esc(source_var))))
        end...)
    end
end


"""
    @vars(names...)

Takes a list of base variable names and expands them into two lists:
one with a `_cpu` suffix and one with a `_gpu` suffix.
Returns a tuple containing the two lists.
"""
macro vars(names...)
    # Create a list of symbols with the `_cpu` suffix
    cpu_vars = [Symbol(String(name), "_cpu") for name in names]
    # Create a list of symbols with the `_gpu` suffix
    gpu_vars = [Symbol(String(name), "_gpu") for name in names]
    
    # The `esc()` is crucial for the macro to access the variables
    # from the scope where it is called.
    # We construct two array expressions, e.g., `[var1_cpu, var2_cpu]`
    # and `[var1_gpu, var2_gpu]`.
    quote
        ( [$(esc.(cpu_vars)...)], [$(esc.(gpu_vars)...)] )
    end
end


function read_and_allocate_forcing(prefix::String, year::Int, varname::String)
    println("Loading $varname forcing input...")

    # 1) Open netCDF file, read variable into a CPU array and copy array into preload
    file_path     = "$(prefix)$(year).nc"
    dataset       = NetCDF.open(file_path)
    cpu_arr       = dataset[varname]
    cpu_preload   = dataset[varname][:, :, :]
    
    # 2) Conditionally allocate a GPU array
    if GPU_USE
        gpu_arr = CUDA.zeros(float_type, size(cpu_arr, 1), size(cpu_arr, 2))
        println("Allocated GPU array of size: ", size(gpu_arr))
        return cpu_preload, gpu_arr
    else
        return cpu_preload, nothing
    end
end


"""
    @load_forcing(year_var, names...)

Takes a year variable and a list of base variable names. For each name,
it generates a call to `read_and_allocate_forcing`, creating the
corresponding `_cpu` and `_gpu` variables.
Parallelized version: Spawns a thread for each variable to read from disk concurrently.
"""

macro load_forcing(year_var, names...)
    # Generate unique temporary names for the tasks
    task_vars = [gensym() for _ in names]

    # 1. Create a block of code to SPAWN all tasks immediately
    spawns = map(zip(names, task_vars)) do (name, task_var)
        prefix_sym = esc(Symbol("input_", String(name), "_prefix"))
        source_sym = esc(Symbol(String(name), "_var"))
        year_esc   = esc(year_var)

        # Generate: task_1 = Threads.@spawn read_and_allocate_forcing(...)
        quote
            $task_var = Threads.@spawn read_and_allocate_forcing($prefix_sym, $year_esc, $source_sym)
        end
    end

    # 2. Create a block of code to FETCH results into your variables
    fetches = map(zip(names, task_vars)) do (name, task_var)
        cpu_var = esc(Symbol(String(name), "_cpu"))
        gpu_var = esc(Symbol(String(name), "_gpu"))

        # Generate: (prec_cpu, prec_gpu) = fetch(task_1)
        quote
            ($cpu_var, $gpu_var) = fetch($task_var)
        end
    end

    # 3. Return the combined block
    quote
        println("Starting parallel forcing load for year ", $(esc(year_var)), " with ", Threads.nthreads(), " threads...")
        $(spawns...)  # Launch all reads
        $(fetches...) # Wait for all reads
        println("Parallel load complete.")
    end
end

function gpu_load_static_inputs(cpu_vars, gpu_vars)
    for (cpu, gpu) in zip(cpu_vars, gpu_vars)
        CUDA.copyto!(gpu, cpu)
    end
end

function gpu_load_monthly_inputs(month, month_prev, cpu_vars, gpu_vars)
    if month != month_prev
        for (cpu, gpu) in zip(cpu_vars, gpu_vars)
            # 1. Get dimensions
            # cpu is typically (nx, ny, 12, nveg)
            # gpu is typically (nx, ny, 1, nveg)
            nx, ny = size(cpu, 1), size(cpu, 2)
            n_months = size(cpu, 3) # Should be 12
            n_tiles = size(cpu, 4)  # Vegetation tiles (1 if variable is 3D)

            block_size = nx * ny

            # 2. Iterate over vegetation tiles to copy each contiguous chunk
            for k in 1:n_tiles
                # Calculate Source Offset (CPU)
                # Skip previous tiles (k-1) * (months * block)
                # Skip previous months in current tile (month-1) * block
                offset_cpu = (k - 1) * (block_size * n_months) + (month - 1) * block_size + 1

                # Calculate Dest Offset (GPU)
                # GPU only has 1 month, so we just skip previous tiles
                offset_gpu = (k - 1) * block_size + 1

                # 3. Direct Copy
                copyto!(gpu, offset_gpu, cpu, offset_cpu, block_size)
            end
        end
    end
end

function gpu_load_daily_inputs(day, day_prev, cpu_vars, gpu_vars)
    if day != day_prev
        for (cpu, gpu) in zip(cpu_vars, gpu_vars)
        # Calculate the starting linear index for the current day
            # cpu is (nx, ny, days), gpu is (nx, ny)
            len = length(gpu)
            offset = (day - 1) * len + 1
            
            # Direct copy using linear offsets: copyto!(dest, dest_offset, src, src_offset, count)
            # This avoids allocating a View and guarantees a fast Memcpy
            copyto!(gpu, 1, cpu, offset, len)
        end
    end
end
