# Convert a config's per-year NetCDF forcing files into per-year Zarr stores
# next to them, so a run can point its forcing paths at `..._{year}.zarr`.
#
#   julia --project=. scripts/convert_forcing_to_zarr.jl configs/mekong_config.toml
#
# Values are passed through the same CF decoding the model uses and fill values
# become NaN, so a Zarr run reads bit-identical forcing to a NetCDF one.

using NCDatasets
using Zarr
using Dates

include(joinpath(@__DIR__, "..", "src", "config.jl"))
using .Config: load_config

const FORCING_VARS = [
    "precipitation",
    "air_temperature",
    "wind_speed",
    "vapor_pressure",
    "shortwave_down",
    "longwave_down",
    "surface_pressure"
]

# Must match ZARR_TIME_EPOCH in src/reader.jl.
const TIME_EPOCH = DateTime(1970, 1, 1)

compressor() = Zarr.BloscCompressor(cname="lz4", clevel=1, shuffle=1)

function write_coordinate(group, name, values, dim_name)
    array = zcreate(
        Float32, group, name, length(values);
        chunks=(length(values),),
        compressor=compressor(),
        attrs=Dict{String, Any}("_ARRAY_DIMENSIONS" => [dim_name]),
    )
    array[:] = Float32.(coalesce.(values, NaN32))
    return nothing
end

function convert_year(nc_path, zarr_path, names)
    if isdir(zarr_path)
        println("  Skipping existing: $zarr_path")
        return
    end

    # Build under a temporary name so an interrupted conversion cannot leave a
    # half-written store that later looks complete.
    tmp_path = zarr_path * ".tmp"
    isdir(tmp_path) && rm(tmp_path, recursive=true)

    NCDataset(nc_path) do ds
        var = ds[names.variable]
        nx, ny, nt = size(var)
        println("  $nc_path -> $zarr_path  ($(nx)x$(ny)x$(nt))")

        group = zgroup(tmp_path)

        # One chunk per timestep matches how the model reads: a single step is
        # then one chunk, which Zarr decompresses straight into the destination.
        zarr_var = zcreate(
            Float32, group, names.variable, nx, ny, nt;
            chunks=(nx, ny, 1),
            compressor=compressor(),
            fill_value=NaN32,
            attrs=Dict{String, Any}("_ARRAY_DIMENSIONS" => ["time", "lat", "lon"]),
        )

        # Stream a timestep at a time -- a full year of a global grid does not
        # fit in memory.
        buffer = Array{Float32, 3}(undef, nx, ny, 1)
        for t in 1:nt
            buffer .= coalesce.(var[:, :, t:t], NaN32)
            zarr_var[:, :, t:t] = buffer
        end

        times = collect(DateTime, ds[names.time][:])
        z_time = zcreate(
            Int64, group, "time", nt;
            chunks=(nt,),
            compressor=compressor(),
            attrs=Dict{String, Any}(
                "_ARRAY_DIMENSIONS" => ["time"],
                "units" => "milliseconds since 1970-01-01",
            ),
        )
        z_time[:] = [Dates.value(t - TIME_EPOCH) for t in times]

        write_coordinate(group, "lat", ds[names.latitude][:], "lat")
        write_coordinate(group, "lon", ds[names.longitude][:], "lon")
    end

    mv(tmp_path, zarr_path)
    return nothing
end

function convert_forcing(config_file)
    cfg = load_config(config_file)

    for var in FORCING_VARS
        path = getfield(cfg.input.paths, Symbol("$(var)_file"))
        endswith(path, ".nc") || error("Expected a NetCDF path ending in .nc for $var, got '$path'")
        names = (
            variable = getfield(cfg.input.names, Symbol(var)),
            time = cfg.input.names.time,
            latitude = cfg.input.names.latitude,
            longitude = cfg.input.names.longitude,
        )
        println("$var ($(names.variable)):")

        for year in cfg.start_year:cfg.end_year
            nc_path = replace(path, "{year}" => string(year))
            zarr_path = nc_path[1:end-3] * ".zarr"
            convert_year(nc_path, zarr_path, names)
        end
    end
end

if isempty(ARGS)
    error("Usage: julia --project=. scripts/convert_forcing_to_zarr.jl <config.toml>")
end
convert_forcing(ARGS[1])
