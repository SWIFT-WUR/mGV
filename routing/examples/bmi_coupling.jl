# Coupling mGV to SubbasinRouting through mGV's BMI.
#
# mGV runs the land surface with its own routing switched off (`enable_routing = false`).
# After every BMI update, the total runoff is read through BMI and routed with
# SubbasinRouting. A second mGV instance, with its built-in routing switched on, serves
# as a check: both give bitwise identical discharge.
#
# usage (from the mGV root):
#   julia --project=routing/examples routing/examples/bmi_coupling.jl [ndays]
#   MGV_BACKEND=rocm julia --project=routing/examples routing/examples/bmi_coupling.jl [ndays]
#   MGV_BACKEND=opencl julia --project=<env with OpenCL.jl> routing/examples/bmi_coupling.jl [ndays]
#
# BMI.initialize selects mGV's backend from the MGV_BACKEND environment variable; the
# router runs on the same backend, so the runoff never leaves the device.
const BACKEND = get(ENV, "MGV_BACKEND", "cpu")
BACKEND == "rocm" && using AMDGPU  # loading the GPU package registers mGV's backend
BACKEND == "opencl" && using OpenCL
import BasicModelInterface as BMI
using mGV, SubbasinRouting, KernelAbstractions, Printf, TOML

ndays = isempty(ARGS) ? 30 : parse(Int, ARGS[1])
ENV["MGV_BACKEND"] = BACKEND

# --- Configs: the Mekong example, with and without mGV's built-in routing ------------------
const CONFIG = joinpath(@__DIR__, "..", "..", "configs", "mekong_config.toml")
function write_config(enable_routing)
    cfg = TOML.parsefile(CONFIG)
    cfg["enable_routing"] = enable_routing
    cfg["output"]["dir"] = mktempdir()
    for (key, path) in cfg["input"]["paths"]  # make input paths absolute
        cfg["input"]["paths"][key] = abspath(joinpath(dirname(CONFIG), path))
    end
    file = joinpath(mktempdir(), "config.toml")
    open(io -> TOML.print(io, cfg), file, "w")
    return file
end
config_file = write_config(false)
check_file = write_config(true)

# --- Land surface model, without routing ----------------------------------------------------
model = BMI.initialize(mGV.Model, config_file)
check = BMI.initialize(mGV.Model, check_file)

# The runoff variable feeding the routing [mm/day]; BMI returns it per land cell, in the
# same cell order as the routing network
runoff_name = "land__runoff~total"
grid = BMI.get_var_grid(model, runoff_name)
n = BMI.get_grid_size(model, grid)
@assert BMI.get_var_units(model, runoff_name) == "mm day-1" "the router expects runoff in mm/day"
backend = KernelAbstractions.get_backend(BMI.get_value_ptr(model, runoff_name))

# --- River network: mGV's own preprocessing of the routing parameter file ---------------------
# (downstream index, upstream lists, channel length, Manning coefficient, cell area) over
# the model's land cells, so the network equals that of the built-in routing.
domain = mGV.Domain(model.config)
network = mGV.RoutingState(model.config, domain)
@assert length(network.downstream_idx) == n

# Large workgroups compile very slowly on KernelAbstractions' CPU backend, so use small
# units there; on a GPU, S = 2048 with two cells per thread. OpenCL devices often allow
# at most 512 work items per workgroup (e.g. Intel Iris Xe), so use S = 1024 there.
S, cpt = backend isa CPU ? (64, 1) : BACKEND == "opencl" ? (1024, 2) : (2048, 2)
layout = SubbasinLayout(network.downstream_idx, network.upstream_ptr, network.upstream_idx; S)
router = SubbasinRouter(backend, layout, network.length, network.alpha, network.cell_area; cells_per_thread = cpt)
println(layout)

# Routing state lives with the coupler (vectors over the land cells, on the model's backend)
area = KernelAbstractions.zeros(backend, Float32, n)       # channel cross-section [m2]
discharge = KernelAbstractions.zeros(backend, Float32, n)  # [m3/s]
runoff = KernelAbstractions.zeros(backend, Float32, n)

# Sub-steps as in mGV.update_routing!
dt = Int32(BMI.get_time_step(model))
nsub = Int(ceil(dt / mGV.ROUTING_DT))
dt_step = dt / Float32(nsub)

# --- Time loop ---------------------------------------------------------------------------
outlet = argmax(network.accumulation)
q_check = KernelAbstractions.zeros(backend, Float32, n)
println(" day   outlet Q coupled [m3/s]   outlet Q built-in [m3/s]")
for day in 1:ndays
    BMI.update(model)
    BMI.get_value(model, runoff_name, runoff)
    route!(router, area, discharge, runoff, nsub, dt_step)

    BMI.update(check)
    BMI.get_value(check, "channel_water__volume_flow_rate", q_check)
    if day % 5 == 0 || day == ndays
        @printf("%4d   %22.3f   %24.3f\n", day, Array(discharge)[outlet], Array(q_check)[outlet])
    end
end

q1, q2 = Array(discharge), Array(q_check)
ndiff = count(i -> q1[i] !== q2[i], eachindex(q1))
println(ndiff == 0 ? "Discharge is bitwise identical to mGV's built-in routing in all $n cells." :
    "Discharge differs from mGV's built-in routing in $ndiff cells.")

BMI.finalize(model)
BMI.finalize(check)
