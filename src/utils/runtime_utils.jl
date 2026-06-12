const USAGE = "Usage: julia run.jl <config_file> [<start_year> [<end_year>]] [--start-year=YYYY] [--end-year=YYYY] [--nc|--netcdf|--output=nc]"

function get_output_format()
    for arg in ARGS
        if startswith(arg, "--output=")
            val = split(arg, "=")[2]
            if val in ["netcdf", "nc"]
                return :netcdf
            end
        elseif arg in ["--netcdf", "--nc"]
            return :netcdf
        end
    end
    return :zarr
end

function parse_args()
    config_file    = nothing
    start_year_arg = nothing
    end_year_arg   = nothing

    # Collect bare integer args (e.g. julia run.jl config.jl 1987 1989)
    # Named flags --start-year= / --end-year= take priority over these if both are given.
    year_positionals = Int[]

    for arg in ARGS
        if startswith(arg, "--start-year=")
            start_year_arg = parse(Int, split(arg, "=")[2])

        elseif startswith(arg, "--end-year=")
            end_year_arg = parse(Int, split(arg, "=")[2])

        elseif arg in ["--nc", "--netcdf"] || startswith(arg, "--output=")
            continue  # handled separately by get_output_format()

        elseif startswith(arg, "--")
            error("Unknown argument: '$arg'\n$USAGE")

        elseif isnothing(config_file)
            config_file = arg

        elseif !isnothing(tryparse(Int, arg))
            push!(year_positionals, parse(Int, arg))

        else
            error("Unexpected argument: '$arg'\n$USAGE")
        end
    end

    if isnothing(config_file)
        error(USAGE)
    end

    # Map positional years -> start/end (named flags win if both specified)
    if length(year_positionals) > 2
        error("Too many positional year arguments (got $(length(year_positionals)), expected at most 2).")
    end
    if length(year_positionals) >= 1
        start_year_arg = something(start_year_arg, year_positionals[1])
    end
    if length(year_positionals) >= 2
        end_year_arg = something(end_year_arg, year_positionals[2])
    end

    if !isabspath(config_file)
        config_file = abspath(config_file)
    end

    if !isfile(config_file)
        error("Config file '$config_file' does not exist or is not reachable from this path!")
    end

    return config_file, start_year_arg, end_year_arg, get_output_format()
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
            println("WARNING: Input file for year $year not found: $file_path")
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
