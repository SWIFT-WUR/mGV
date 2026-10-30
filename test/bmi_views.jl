# Checks for the BMI getters and setters of the per-layer and per-band variables, which BMI
# exposes as views into 3D or 4D model arrays. For 4D arrays these views are non-contiguous.
# Shared between opencl_test.jl and gpu_test.jl, so the arrays live on a device: the expected
# values are always computed from a host copy of the full model array.

is_layered(name::String) = occursin(r"_(layer|band)_\d", name)

# Host copy of the full model array behind the layered variable `name`, and the layer/band index
function host_field(model::mGV.Model, name::String)
    name_2d, ind = if occursin(r"_layer_\d", name)
        mGV.soil_layer_standard_name(name)
    else
        mGV.snow_band_standard_name(name)
    end
    model_vals, _ = mGV.get_field_in_model(model, name_2d)
    return Array(model_vals), ind
end

# Flattened (writable) slice of a host array, in the order BMI uses for the variable
host_slice(field::Array, ind::Int) = vec(selectdim(field, 3, ind))

function var_size_type(model::mGV.Model, name::String)
    n = BMI.get_grid_size(model, BMI.get_var_grid(model, name))
    T = mGV.TYPES[BMI.get_var_type(model, name)]
    return n, T
end

# Deterministic test data: distinct (repeating) values and unsorted indices
test_values(T, n) = T.(mod1.(1:n, 100))
test_indices(n) = collect(n:-7:1)[1:min(end, 50)]

function check_get_value(model::mGV.Model, name::String)
    n, T = var_size_type(model, name)
    field, ind = host_field(model, name)
    dest = zeros(T, n)
    BMI.get_value(model, name, dest)
    @test isequal(dest, host_slice(field, ind))
    return nothing
end

function check_get_value_at_indices(model::mGV.Model, name::String)
    n, T = var_size_type(model, name)
    field, ind = host_field(model, name)
    idx = test_indices(n)
    dest = zeros(T, length(idx))
    BMI.get_value_at_indices(model, name, dest, idx)
    @test isequal(dest, host_slice(field, ind)[idx])
    return nothing
end

# The full model array is compared, so writes outside of the layer/band are detected as well
function check_set_value(model::mGV.Model, name::String)
    n, T = var_size_type(model, name)
    expected, ind = host_field(model, name)
    src = test_values(T, n)
    host_slice(expected, ind) .= src
    BMI.set_value(model, name, src)
    @test isequal(first(host_field(model, name)), expected)
    return nothing
end

function check_set_value_at_indices(model::mGV.Model, name::String)
    n, T = var_size_type(model, name)
    expected, ind = host_field(model, name)
    idx = test_indices(n)
    src = reverse(test_values(T, length(idx)))
    host_slice(expected, ind)[idx] .= src
    BMI.set_value_at_indices(model, name, idx, src)
    @test isequal(first(host_field(model, name)), expected)
    return nothing
end

# Run all checks on the layered variables of an initialized model. The setters overwrite
# model state, so the model should not be used for anything else afterwards.
function test_layered_views(model::mGV.Model)
    BMI.update(model)  # make sure the variables hold model values, not just initial values

    out_vars = filter(is_layered, BMI.get_output_var_names(model))
    in_vars = filter(is_layered, BMI.get_input_var_names(model))
    @test !isempty(out_vars)
    @test !isempty(in_vars)

    @testset "get $name" for name in out_vars
        check_get_value(model, name)
        check_get_value_at_indices(model, name)
    end
    @testset "set $name" for name in in_vars
        check_set_value(model, name)
        check_set_value_at_indices(model, name)
    end
    return nothing
end
