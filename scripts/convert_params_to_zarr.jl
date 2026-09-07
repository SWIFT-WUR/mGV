# Convert the Float32 landsurface parameter NetCDF file into a Zarr store,
# so a run can point `input_param_file` at the resulting `<name>.zarr`
# directory instead.
#
#   julia --project=. scripts/convert_params_to_zarr.jl <params_f32.nc> [output.zarr]
#
# Every variable becomes one Zarr array. Missing values are filled with 0.0
# at conversion time, matching what `read_parameters` in src/parameters.jl
# already did for every field, so the runtime Zarr read needs no extra
# missing-value handling.

using NCDatasets
using Zarr

compressor() = Zarr.BloscCompressor(cname="lz4", clevel=1, shuffle=1)

# Blosc caps a compressed chunk at about 2GiB, and a full 4D field (e.g. LAI
# at 4320x1680x12x14 Float32 is about 4.9GiB) is too big as one chunk. Keep
# only the first two dimensions (lon, lat) full size.
chunk_shape(dims::NTuple{N, Int}) where {N} =
    N <= 2 ? dims : (dims[1], dims[2], ntuple(_ -> 1, N - 2)...)

function convert_params(nc_path, zarr_path)
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

            println("  $vname  dims=$(dimnames(var))  size=$dims  ($eltyp to $out_type)")

            data = if eltyp <: AbstractFloat
                Float32.(nomissing(Array(var), 0.0))
            else
                nomissing(Array(var), zero(eltyp))
            end

            zarr_var = zcreate(
                out_type, group, vname, dims...;
                chunks=chunk_shape(dims),
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
    error("Usage: julia --project=. scripts/convert_params_to_zarr.jl <params_f32.nc> [output.zarr]")
end
nc_path = ARGS[1]
zarr_path = length(ARGS) >= 2 ? ARGS[2] : replace(nc_path, r"\.nc$" => ".zarr")
convert_params(nc_path, zarr_path)
