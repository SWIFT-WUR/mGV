function read_and_allocate_parameter(varname::String)
    println("Loading $varname parameter input...")

    # 1) Open netCDF file 
    dataset = NetCDF.open(input_param_file)
    var_dims = size(dataset[varname])

    # 2) Read (landuse) parameter into CPU memory (RAM) for 1D/2D/3D/4D arrays
    slicing_indices = repeat([:], length(var_dims))
    cpu_preload = dataset[varname][slicing_indices...]

    # 3) Print array sizes for diagnostics
    if length(var_dims) <= 4
        println("Element type for $(length(var_dims))D: ", eltype(cpu_preload))
    end
    println("Full size of $varname: ", size(cpu_preload))

    # 4) Optimizations for data transfer
    # Locks memory pages if using NVIDIA/AMD; does nothing on CPU/Metal.
    try
        pin_memory!(cpu_preload)
    catch e
        println("  -> WARNING: Failed to pin CPU memory. Transfer will be slower. Error: $e")
    end

    # 5) Allocate device memory
    # Handle 4D reshaping logic to only pre-allocate memory daily for monthly (vegetation) tiles 
    adjusted_dims = if length(var_dims) == 4
        (var_dims[1], var_dims[2], (var_dims[3] == 12 ? 1 : var_dims[3]), var_dims[4])
    else
        var_dims
    end

    # 6) Pre-allocating memory on the active device (VRAM/RAM)
    device_arr = alloc(FloatType, adjusted_dims...)
    println("Allocated $backend_name array of size: ", size(device_arr))

    return cpu_preload, device_arr
end

# Macro to make the creation of the CPU and GPU variables easier and more compact
macro load_params(vars...)
    quote
        $(map(vars) do var
            cpu_name = Symbol(String(var), "_cpu")
            gpu_name = Symbol(String(var), "_gpu")
            source_var = Symbol(String(var), "_var")

            # This generates the line: (var_cpu, var_gpu) = read_and_allocate_parameter(var_var)
            :(($(esc(cpu_name)), $(esc(gpu_name))) = read_and_allocate_parameter($(esc(source_var))))
        end...)
    end
end

macro vars(names...)
    # Create list of symbols with the "_cpu" and "_gpu" suffixes
    cpu_vars = [Symbol(String(name), "_cpu") for name in names]
    gpu_vars = [Symbol(String(name), "_gpu") for name in names]

    # We construct two array expressions, e.g., `[var1_cpu, var2_cpu]`
    # and `[var1_gpu, var2_gpu]`.
    quote
        ([$(esc.(cpu_vars)...)], [$(esc.(gpu_vars)...)])
    end
end

function read_and_allocate_forcing(prefix::String, year::Int, varname::String)
    println("Loading $varname forcing input...")

    # 1) Open netCDF file, read yearly forcing variable into CPU memory (RAM)
    file_path = "$(prefix)$(year).nc"
    dataset = NetCDF.open(file_path)
    cpu_preload = dataset[varname][:, :, :] # 3D array: [x, y, time]

    # 2) Optimizations for data transfer
    # Locks memory pages if using NVIDIA/AMD; does nothing on CPU/Metal.
    try
        pin_memory!(cpu_preload)
    catch e
        println("  -> WARNING: Failed to pin forcing memory for $varname. Error: $e")
    end

    # 3) Allocate device buffer (2D)
    # For forcing, we only keep the current day (therefore a 2D buffer) on the device to save memory
    nx, ny = size(cpu_preload)[1:2]
    device_arr = alloc(FloatType, nx, ny)

    println("Allocated $backend_name buffer size: ", size(device_arr))

    return cpu_preload, device_arr
end


# Macro that takes a year variable and a list of base variable names. For each name,
# it generates a call to `read_and_allocate_forcing`, to create the corresponding `_cpu` and `_gpu` variables.
macro load_forcing(year_var, names...)
    # The `quote ... end` block collects all the generated lines of code.
    quote
        # `map` iterates through each variable name provided (e.g., :prec, :tair)
        $(map(names) do name
            # Construct all the necessary variable names from the base name
            cpu_var = esc(Symbol(String(name), "_cpu"))
            gpu_var = esc(Symbol(String(name), "_gpu"))
            prefix_var = esc(Symbol("input_", String(name), "_prefix"))
            source_var = esc(Symbol(String(name), "_var"))
            year_esc = esc(year_var)

            # This is the line of code that will be generated for each name:
            # e.g., (prec_cpu, prec_gpu) = read_and_allocate_forcing(input_prec_prefix, year, prec_var)
            :(($cpu_var, $gpu_var) = read_and_allocate_forcing($prefix_var, $year_esc, $source_var))
        end...)
    end
end

function gpu_load_static_inputs(cpu_vars, gpu_vars)
    for (cpu, gpu) in zip(cpu_vars, gpu_vars)
        copyto!(gpu, cpu)
    end
end


function gpu_load_monthly_inputs(month, month_prev, cpu_vars, gpu_vars)
    month == month_prev && return

    for (cpu, gpu) in zip(cpu_vars, gpu_vars)
        # Dimensions: cpu is (nx, ny, 12, nveg), gpu is (nx, ny, 1, nveg)
        nx, ny   = size(cpu, 1), size(cpu, 2)
        n_months = size(cpu, 3)
        n_tiles  = size(cpu, 4)
        block_size = nx * ny

        for k in 1:n_tiles
            # Calculate linear memory addresses to avoid View allocations
            # CPU: Skip previous tiles (k-1) + previous months (month-1)
            offset_cpu = (k - 1) * (n_months * block_size) + (month - 1) * block_size + 1
            
            # GPU: Skip previous tiles only (buffer holds current month)
            offset_gpu = (k - 1) * block_size + 1

            copyto!(gpu, offset_gpu, cpu, offset_cpu, block_size)
        end
    end
end


function gpu_load_daily_inputs(day, day_prev, cpu_vars, gpu_vars)
    day == day_prev && return

    for (cpu, gpu) in zip(cpu_vars, gpu_vars)
        # cpu is (nx, ny, days), gpu is (nx, ny) buffer
        len    = length(gpu)
        offset = (day - 1) * len + 1
        
        # Direct linear copy: fast, safe, and alloc-free
        copyto!(gpu, 1, cpu, offset, len)
    end
end