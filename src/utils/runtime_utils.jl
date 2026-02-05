function parse_case_args()
    # Use the first argument in ARGS as the CASE, defaulting to "global" if not provided
    local_case = length(ARGS) > 0 ? ARGS[1] : "global"
    
    # Notify the user if defaulting to "global"
    if length(ARGS) == 0
        println("No CASE provided. Defaulting to 'global'.")
    end
    
    # Parse optional start and end year from ARGS
    local_start_year_arg = length(ARGS) > 1 ? parse(Int, ARGS[2]) : nothing
    local_end_year_arg   = length(ARGS) > 2 ? parse(Int, ARGS[3]) : nothing
    
    return local_case, local_start_year_arg, local_end_year_arg
end

function ensure_output_directory(output_dir::String)
    # Check if the output directory exists, otherwise create it
    if !isdir(output_dir)
        println("Output directory '$output_dir' does not exist. Creating it...")
        mkpath(output_dir)
    end
end

function has_input_files(year)
    # List of input file prefixes
    input_prefixes = [
        input_prec_prefix,
        input_tair_prefix,
        input_wind_prefix,
        input_vp_prefix,
        input_swdown_prefix,
        input_lwdown_prefix
    ]

    # Check if all required files exist
    for prefix in input_prefixes
        file_path = "$(prefix)$(year).nc"
        if !isfile(file_path)
            println("⚠️ WARNING: Input file for year $year not found: $file_path")
            return false
        end
    end
    return true
end

function day_to_month(day::Int, year::Int)
    # Construct the date from the year and day of the year
    date = Date(year, 1, 1) + Day(day - 1)
    return month(date)  # Extract the month from the date
end

# Asynchronously compresses a NetCDF file using `nccopy`, offloading compression to the shell for efficiency.
# Accepts `output_file` and `compression_level` (1 = fast/light, 9 = slow/max).
function compress_file_async(output_file::String, compression_level::Int)
    # Check if `nccopy` is available
    try
        nccopy_path = read(`which nccopy`, String) |> strip
        println("Using nccopy for NetCDF file compression: $nccopy_path")
    catch 
        println("""
        WARNING: Output file compression will not occur because the `nccopy` command is not available on your system.
        Please load the appropriate module or install it using your package manager.
        """)
        return
    end

    println("Attempting compression of file $output_file asynchronously with compression level $compression_level...")

    # Construct the shell command as a string
    command = """
    nohup bash -c 'nccopy -d $compression_level $output_file $output_file.tmp && mv $output_file.tmp $output_file' > /dev/null 2>&1 &
    """

    # Run the detached shell command
    try
        run(`sh -c $command`)
        println("Compression process for $output_file started in the background.\n")
    catch err
        println("Failed to start compression for $output_file: $err")
    end
end