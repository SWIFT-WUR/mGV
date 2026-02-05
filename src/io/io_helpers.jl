function reshape_static_inputs!()
    global rmin_gpu, rarc_gpu, cv_gpu

    # Check and reshape rmin_gpu if it has 3 dimensions.
    if ndims(rmin_gpu) == 3
        rmin_gpu = CUDA.reshape(rmin_gpu,
                                size(rmin_gpu, 1),
                                size(rmin_gpu, 2),
                                1,
                                size(rmin_gpu, 3))
        println("rmin_gpu reshaped to: ", size(rmin_gpu))
    else
        println("rmin_gpu already has ", ndims(rmin_gpu), " dimensions; no reshape needed.")
    end

    # Check and reshape rarc_gpu if it has 3 dimensions.
    if ndims(rarc_gpu) == 3
        rarc_gpu = CUDA.reshape(rarc_gpu,
                                size(rarc_gpu, 1),
                                size(rarc_gpu, 2),
                                1,
                                size(rarc_gpu, 3))
        println("rarc_gpu reshaped to: ", size(rarc_gpu))
    else
        println("rarc_gpu already has ", ndims(rarc_gpu), " dimensions; no reshape needed.")
    end

    # Check and reshape cv_gpu if it has 3 dimensions.
    if ndims(cv_gpu) == 3
        cv_gpu = CUDA.reshape(cv_gpu,
                              size(cv_gpu, 1),
                              size(cv_gpu, 2),
                              1,
                              size(cv_gpu, 3))
        println("cv_gpu reshaped to: ", size(cv_gpu))
    else
        println("cv_gpu already has ", ndims(cv_gpu), " dimensions; no reshape needed.")
    end

end

# Helper function to sum over a dimension with NaN handling
function sum_with_nan_handling(arr::CuArray{T}, dim::Int) where T
    # 1. Fused Map & Reduce
    # Map: x -> (value_or_zero, count)
    #      If NaN, we contribute 0.0 to sum and 0 to count.
    #      If Valid, we contribute x to sum and 1 to count.
    # Reduce: Element-wise sum of the tuples.
    
    results = mapreduce(
        x -> isnan(x) ? (zero(T), Int32(0)) : (x, Int32(1)), 
        (a, b) -> (a[1] + b[1], a[2] + b[2]), 
        arr; 
        dims = dim, 
        init = (zero(T), Int32(0))
    )

    # 2. Post-Process (Lightweight)
    # Extract the sum. If count is 0 (all were NaN), return NaN.
    # We map over the much smaller 'results' array here.
    output = map(r -> r[2] == 0 ? T(NaN) : r[1], results)

    # 3. Match original API (drop the singleton dimension)
    return dropdims(output, dims=dim)
end


san_nan(A) = begin
    T   = eltype(A)
    thr = T(fillvalue_threshold)
    rep = T(NaN)
    ifelse.(isnan.(A) .| (abs.(A) .> thr), rep, A)
end

san_zero(A) = begin
    T   = eltype(A)
    thr = T(fillvalue_threshold)
    rep = T(0.0)
    ifelse.(isnan.(A) .| (abs.(A) .> thr), rep, A)
end

convcv(A) = convert.(eltype(A), cv_gpu)  # cast cv to A's eltype


function get_output_format()
    # 1. Check for explicit assignment: --output=netcdf
    for arg in ARGS
        if startswith(arg, "--output=")
            val = split(arg, "=")[2]
            if val in ["netcdf", "nc"]
                return :netcdf
            end
        # 2. Check for simple flags: --netcdf or --nc
        elseif arg in ["--netcdf", "--nc"]
            return :netcdf
        end
    end
    
    # 3. Default
    return :zarr 
end