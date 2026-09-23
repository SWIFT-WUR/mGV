# Convert the domain NetCDF file (`coverage_file`) into a Zarr store, so
# `coverage_file` can point at the resulting `<name>.zarr` directory.
#
#   julia --project=. scripts/convert_domain_to_zarr.jl <domain_f32.nc> [output.zarr]
#
# Note: nothing in src/ currently reads `coverage_file`. This conversion is
# for future use.

using NCDatasets
using Zarr

compressor() = Zarr.BloscCompressor(cname="lz4", clevel=1, shuffle=1)

function convert_domain(nc_path, zarr_path)
    if isdir(zarr_path)
        println("Skipping existing: $zarr_path")
        return
    end

    tmp_path = zarr_path * ".tmp"
    isdir(tmp_path) && rm(tmp_path, recursive=true)

    NCDataset(nc_path) do ds
        group = zgroup(tmp_path)

        for (vname, var) in ds
            dims = size(var)
            eltyp = nonmissingtype(eltype(var))
            out_type = eltyp <: AbstractFloat ? Float32 : eltyp
            fallback = eltyp <: AbstractFloat ? Float32(-9999) : zero(eltyp)

            println("  $vname  dims=$(dimnames(var))  size=$dims  ($eltyp to $out_type)")

            data = out_type.(nomissing(Array(var), fallback))

            zarr_var = zcreate(
                out_type, group, vname, dims...;
                chunks=dims,
                compressor=compressor(),
                attrs=Dict{String, Any}("_ARRAY_DIMENSIONS" => reverse(collect(dimnames(var)))),
            )
            zarr_var[ntuple(_ -> Colon(), length(dims))...] = data
        end
    end

    mv(tmp_path, zarr_path)
    println("Wrote $zarr_path")
    return nothing
end

if isempty(ARGS)
    error("Usage: julia --project=. scripts/convert_domain_to_zarr.jl <domain_f32.nc> [output.zarr]")
end
nc_path = ARGS[1]
zarr_path = length(ARGS) >= 2 ? ARGS[2] : replace(nc_path, r"\.nc$" => ".zarr")
convert_domain(nc_path, zarr_path)
