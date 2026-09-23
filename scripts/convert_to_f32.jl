#!/usr/bin/env julia
# Convert a NetCDF file's Float64 variables to Float32.
#
#   julia --project=. scripts/convert_to_f32.jl <input.nc> <output.nc>
#
# Also repairs a bug seen in some VIC parameter files, where a variable's
# fill value attribute is spelled backwards (e.g. "eulaVlliF_" instead of
# "_FillValue"), so NCDatasets never masks it and raw NaN leaks through.

using NCDatasets

const REVERSED_FILLVALUE_KEY = reverse("_FillValue")  # "eulaVlliF_"

function convert_to_float32(in_path, out_path)
    println("Opening $in_path...")
    ds_in = NCDataset(in_path, "r")

    isfile(out_path) && rm(out_path)
    ds_out = NCDataset(out_path, "c")

    try
        for (dname, dlen) in ds_in.dim
            defDim(ds_out, dname, dlen)
        end

        for (k, v) in ds_in.attrib
            ds_out.attrib[k] = v
        end

        for (vname, var) in ds_in
            orig_type = nonmissingtype(eltype(var))

            new_type = if orig_type == Float64
                println("  Converting $vname (Float64 to Float32)")
                Float32
            else
                println("  Keeping    $vname ($orig_type)")
                orig_type
            end

            att_dict = Dict(var.attrib)

            has_reversed_fill = haskey(att_dict, REVERSED_FILLVALUE_KEY)
            reversed_fill_value = nothing
            if has_reversed_fill
                reversed_fill_value = att_dict[REVERSED_FILLVALUE_KEY]
                println("  WARNING: $vname has reversed fill-value attribute " *
                        "'$REVERSED_FILLVALUE_KEY' = $reversed_fill_value (not auto-masked by NCDatasets). " *
                        "Repairing to a proper _FillValue.")
                delete!(att_dict, REVERSED_FILLVALUE_KEY)
            end

            fill_value = if haskey(att_dict, "_FillValue")
                att_dict["_FillValue"]
            elseif has_reversed_fill
                reversed_fill_value
            else
                nothing
            end

            # The reversed attribute's own value is NaN, so it can't be used
            # as the real fill value either. Fall back to 0.0.
            if has_reversed_fill && (fill_value === nothing || (fill_value isa AbstractFloat && isnan(fill_value)))
                fill_value = 0.0
            end

            if new_type == Float32 && fill_value isa Number
                fill_value = Float32(fill_value)
            end
            if fill_value !== nothing
                att_dict["_FillValue"] = fill_value
            end

            v_out = defVar(ds_out, vname, new_type, dimnames(var); attrib=att_dict)

            data = Array(var)

            if has_reversed_fill && eltype(data) <: AbstractFloat
                n_nan = count(isnan, data)
                if n_nan > 0
                    println("  Scrubbing $n_nan raw NaN cells in $vname to fill value $fill_value")
                    data = map(x -> isnan(x) ? fill_value : x, data)
                end
            end

            v_out[:] = if new_type != orig_type
                map(x -> ismissing(x) ? missing : Float32(x), data)
            else
                data
            end
        end
        println("\nSuccess! Output saved to: $out_path")

    finally
        close(ds_in)
        close(ds_out)
    end
end

if length(ARGS) != 2
    error("Usage: julia --project=. scripts/convert_to_f32.jl <input.nc> <output.nc>")
end
convert_to_float32(ARGS[1], ARGS[2])
