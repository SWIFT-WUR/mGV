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