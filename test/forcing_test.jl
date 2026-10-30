using Test

using mGV
using Dates
using TOML
using Zarr: zgroup, zcreate, zopen, BloscCompressor

const MEKONG_CONFIG = joinpath(@__DIR__, "../configs/mekong_config.toml")

"""
Write a copy of the Mekong config to `dir` without output. With `forcing`, all forcing
variables are read from that path instead.
"""
function forcing_test_config(dir; forcing = nothing)
    cfg = TOML.parsefile(MEKONG_CONFIG)
    for (key, path) in cfg["input"]["paths"]
        cfg["input"]["paths"][key] = abspath(joinpath(dirname(MEKONG_CONFIG), path))
    end
    if !isnothing(forcing)
        for var in mGV.FORCING_VARS
            delete!(cfg["input"]["paths"], "$(var)_file")
        end
        cfg["input"]["paths"]["forcing_file"] = forcing
    end
    cfg["output"]["dir"] = dir
    cfg["output"]["variables"] = []
    config_file = joinpath(dir, "config_$(isnothing(forcing) ? "netcdf" : "land").toml")
    open(io -> TOML.print(io, cfg), config_file, "w")
    return config_file
end

# Air temperature is stored as int16 with this scale factor, the other variables as Float32
const TAIR_SCALE = 0.01f0
quantize(x) = isnan(x) ? Int16(-32768) : round(Int16, x / TAIR_SCALE)
unpack(q) = q == -32768 ? NaN32 : Float32(q) * TAIR_SCALE + 0.0f0

"""
Write the NetCDF forcing of `year` as a land-only Zarr store at `path`. The cells are stored
in reverse order, followed by a cell outside the domain, in small chunks of 100 cells and 10
days, so the chunks at the edges are partly filled.
"""
function write_land_store(path, cfg, domain, netcdf, year)
    names = cfg.input.names
    n = mGV.ncells(domain)
    outside = first(setdiff(1:prod(domain.grid_size), domain.cells))
    cells = [reverse(domain.cells); outside]
    nx = first(domain.grid_size)
    times = filter(t -> Dates.year(t) == year, netcdf.times)

    ds = mGV.open_param_source(cfg.input.paths.input_param_file)
    lon, lat = mGV.readfull(ds, names.longitude, 0.0), mGV.readfull(ds, names.latitude, 0.0)
    mGV.close_param_source(ds)

    group = zgroup(path)
    function vector(name, values, dim; attrs = Dict{String, Any}())
        z = zcreate(eltype(values), group, name, length(values);
            attrs = merge(attrs, Dict{String, Any}("_ARRAY_DIMENSIONS" => [dim])))
        z[:] = values
    end
    vector(names.longitude, lon, names.longitude)
    vector(names.latitude, lat, names.latitude)
    vector("lon_index", Int32.((cells .- 1) .% nx), "cell")
    vector("lat_index", Int32.((cells .- 1) .÷ nx), "cell")
    vector(names.time, Int32[Dates.value(Date(t) - Date(1970)) for t in times], names.time;
        attrs = Dict{String, Any}("units" => "days since 1970-01-01", "calendar" => "standard"))

    for var in mGV.FORCING_VARS
        data = reduce(hcat, (mGV.read_var(t, netcdf, var) for t in times))[reverse(1:n), :]
        attrs = Dict{String, Any}("_ARRAY_DIMENSIONS" => [names.time, "cell"])
        if var == "air_temperature"
            data = quantize.(data)
            attrs["scale_factor"] = TAIR_SCALE
            attrs["add_offset"] = 0.0f0
        end
        # The cell outside the domain gets values that should never be read
        data = vcat(data, fill(eltype(data)(12345), 1, length(times)))
        z = zcreate(eltype(data), group, mGV.getval(names, var), size(data)...;
            chunks = (100, 10), compressor = BloscCompressor(cname = "zstd", clevel = 5, shuffle = 1),
            fill_value = eltype(data) == Int16 ? Int16(-32768) : NaN32, attrs)
        z[:, :] = data
    end
    return nothing
end

@testset "Land-only Zarr forcing" begin
    dir = mktempdir()
    netcdf_cfg = mGV.load_config(forcing_test_config(dir))
    domain = mGV.Domain(netcdf_cfg)
    n = mGV.ncells(domain)
    netcdf = mGV.open_forcing(netcdf_cfg, domain)
    for year in 1979:1980
        write_land_store(joinpath(dir, "forcing_$year.zarr"), netcdf_cfg, domain, netcdf, year)
    end
    land_config = forcing_test_config(dir; forcing = joinpath(dir, "forcing_{year}.zarr"))
    land_cfg = mGV.load_config(land_config)

    @testset "Reads the same values as the NetCDF forcing" begin
        # A 20-day cache, so the two years are read in many blocks, which are prefetched
        land = mGV.open_forcing(land_cfg, domain; cache_bytes = 2 * 20 * n * 4 * length(mGV.FORCING_VARS))
        @test land.sources isa Dict{String, mGV.LandZarrForcingVar}
        @test land.capacity == 20
        @test land.times == netcdf.times
        @test mGV.plain_blosc(mGV.Zarr.get_pipeline(first(land.sources["precipitation"].arrays).metadata))

        mismatches = 0
        prefetches = 0
        for t in netcdf.times
            for var in mGV.FORCING_VARS
                expected = mGV.read_var(t, netcdf, var)
                var == "air_temperature" && (expected = unpack.(quantize.(expected)))
                mismatches += !isequal(mGV.read_var(t, land, var), expected)
            end
            prefetches += !isnothing(land.prefetch)
        end
        @test mismatches == 0
        @test prefetches > 0
        @test land.cache_start > 700  # all blocks were read
    end

    @testset "Rejects stores without all active cells" begin
        # Replace the last active cell (the first row, as the rows are reversed) by the cell
        # outside the domain
        path = joinpath(dir, "partial.zarr")
        cp(joinpath(dir, "forcing_1979.zarr"), path)
        group = zopen(path, "w")
        index = group["lon_index"][:]
        index[1] = index[end]
        group["lon_index"][:] = index
        lat_index = group["lat_index"][:]
        lat_index[1] = lat_index[end]
        group["lat_index"][:] = lat_index
        partial_cfg = mGV.load_config(forcing_test_config(mktempdir(); forcing = path))
        @test_throws ErrorException(
            "Land-only forcing '$path' is missing 1 of the $n active cells, or holds some of them more than once"
        ) mGV.open_forcing(partial_cfg, domain)
    end

    @testset "Runs the model" begin
        model = mGV.Model(land_config; backend = "cpu")
        for _ in 1:3
            mGV.update!(model)
        end
        # The forcing of the 4th day, read after the 3rd update
        @test isequal(model.forcing_variables.precipitation, mGV.read_var(DateTime(1979, 1, 4), netcdf, "precipitation"))
        @test all(isfinite, model.soil_variables.moisture)
        mGV.finalize!(model)
    end
end
