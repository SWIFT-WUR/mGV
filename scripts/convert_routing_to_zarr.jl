# Convert the routing parameter NetCDF file into a Zarr store, so a run can
# point `routing_param_file` at the resulting `<name>.zarr` directory instead.
#
#   julia --project=. scripts/convert_routing_to_zarr.jl <routing_param_wbt_f32.nc> [output.zarr]
#
# Only converts the 5 variables `RoutingState` in src/routing.jl actually
# reads. The file also has unit-hydrograph fields nothing in the model uses.
# Missing values are filled at conversion time with the same fallbacks
# `load_safe` in src/routing.jl uses (Int32 ids fall back to -1, Float32
# physics fields fall back to NaN32), so the Zarr store is already clean.

using NCDatasets
using Zarr

compressor() = Zarr.BloscCompressor(cname="lz4", clevel=1, shuffle=1)

const ROUTING_VARS = [
    ("downstream_id", Int32, Int32(-1)),
    ("downstream",    Int32, Int32(-1)),
    ("cell_dist",     Float32, NaN32),
    ("cell_area",     Float32, NaN32),
    ("accumulation",  Float32, NaN32),
]

function convert_routing(nc_path, zarr_path)
    if isdir(zarr_path)
        println("Skipping existing: $zarr_path")
        return
    end

    tmp_path = zarr_path * ".tmp"
    isdir(tmp_path) && rm(tmp_path, recursive=true)

    NCDataset(nc_path) do ds
        group = zgroup(tmp_path)

        for (vname, out_type, fallback) in ROUTING_VARS
            var = ds[vname]
            dims = size(var)
            println("  $vname  dims=$(dimnames(var))  size=$dims  to $out_type (fallback=$fallback)")

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
    error("Usage: julia --project=. scripts/convert_routing_to_zarr.jl <routing_param_wbt_f32.nc> [output.zarr]")
end
nc_path = ARGS[1]
zarr_path = length(ARGS) >= 2 ? ARGS[2] : replace(nc_path, r"\.nc$" => ".zarr")
convert_routing(nc_path, zarr_path)
