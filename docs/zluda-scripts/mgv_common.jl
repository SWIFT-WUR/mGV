# Helpers shared by the mgv_*.jl scripts. Needs MGV_COPY (the copy of mGV, with configs/ and
# input_data/) and MGV_OUT (a directory for run outputs); env.sh sets both.
using mGV, TOML
using Zarr: zopen
const REPO = ENV["MGV_COPY"]
const MEKONG_CFG = joinpath(REPO, "configs/mekong_config.toml")
"Mekong config with absolute input paths and output under `dir`"
function make_config(dir; output...)
    cfg = TOML.parsefile(MEKONG_CFG)
    for (key, path) in cfg["input"]["paths"]
        cfg["input"]["paths"][key] = abspath(joinpath(REPO, "configs", path))
    end
    merge!(cfg["output"], Dict{String,Any}("dir" => dir), Dict{String,Any}(string(k) => v for (k, v) in output))
    file = joinpath(dir, "config.toml")
    open(io -> TOML.print(io, cfg), file, "w")
    return file
end
function run_landonly(backend, dir; ndays = 5, write_interval = 8, packing = false)
    mkpath(dir)
    file = make_config(dir; format = "zarr_landonly", write_interval, packing)
    model = mGV.Model(file; backend)
    t = @elapsed for d in 1:ndays
        mGV.update!(model)
    end
    mGV.finalize!(model)
    store = zopen(joinpath(dir, only(filter(endswith(".zarr"), readdir(dir)))))
    return Dict(name => collect(selectdim(a[ntuple(_ -> Colon(), ndims(a))...], 2, 1:ndays)) for (name, a) in store.arrays if ndims(a) >= 2), t
end
