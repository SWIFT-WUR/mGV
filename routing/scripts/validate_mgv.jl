# Validate SubbasinRouting against mGV.update_routing! (bitwise) and time both.
#
# usage: julia --project=routing/scripts routing/scripts/validate_mgv.jl <backend> <network> [S] [cells_per_thread] [ndays]
#   backend: cpu | rocm
#   network: mekong | indus | path to a routing file with an `elevation` variable
#            (e.g. the global network from make_routing.py)
# On the CPU use S = 64: large workgroups compile very slowly on KernelAbstractions' CPU backend.
const BACKEND = ARGS[1]
BACKEND == "rocm" && using AMDGPU
using mGV, SubbasinRouting, KernelAbstractions, NCDatasets, Printf, Random, Statistics
const M = mGV
BACKEND == "rocm" && (AMDGPU.HARD_MEMORY_LIMIT[] = UInt64(12) * 1024^3)
const backend = BACKEND == "rocm" ? ROCBackend() : CPU()
sync() = KernelAbstractions.synchronize(backend)
dev(x) = (y = KernelAbstractions.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)

const ROOT = joinpath(@__DIR__, "..", "..")
net = ARGS[2]
S = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2048
cpt = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2
ndays = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3
pf, ef, ev = net == "mekong" ? ("$ROOT/input_data/mekong/routing/VIC_rout_params_Mekong.nc", "$ROOT/input_data/mekong/vic_mekong_5min_params.nc", "elev") :
    net == "indus" ? ("$ROOT/input_data/indus/routing/VIC_rout_params_Indus_accfix.nc", "$ROOT/input_data/indus/VIC_params_Mirca_calibrated_Indus.nc", "elev") :
    (net, net, "elevation")

elevation = NCDataset(ds -> replace(ds[ev][:, :], missing => 0.0f0), ef)
nx, ny = size(elevation)
cfg = (input = (paths = (routing_param_file = pf,),),)
rh = redirect_stdout(() -> M.RoutingState(cfg, elevation), devnull)
active = isfinite.(rh.cell_area)

t0 = time()
layout = SubbasinLayout(rh.downstream_idx, rh.upstream_mask, nx, active; S)
router = SubbasinRouter(backend, layout, rh.length, rh.alpha, rh.cell_area; cells_per_thread = cpt)
@printf("# %s %s: %d x %d grid, %d active, %s, setup %.1f s\n", BACKEND, basename(net), nx, ny, count(active), layout, time() - t0)

function runoff_field(day)
    rng = Xoshiro(day); ph = rand(rng, 2) .* 2pi
    Float32[max(0, 8 + 6 * sin(i / 37 + ph[1]) * cos(j / 23 + ph[2]) + 4 * randn(rng)) for i in 1:nx, j in 1:ny]
end
runoffs = [dev(runoff_field(d)) for d in 1:ndays]
mock(r, ro, dt) = (; routing = r, soil_variables = (; total_runoff = ro), clock = (; dt = (; value = dt)))
to_dev(r) = M.RoutingState((fn == :violation_counter ? dev(copy(getfield(r, fn))) : dev(copy(getfield(r, fn))) for fn in fieldnames(typeof(r)))...)
nsub_of(dt) = Int(ceil(Int32(dt) / M.ROUTING_DT))
diag!(r) = M.routing_diagnostics_kernel!(backend)(r.cfl, r.travel_time, r.area, r.discharge, r.length,
    600.0f0, Int32(length(r.area)); ndrange = length(r.area))

for dt in (86400, 600 * 143)
    nsub = nsub_of(dt)
    rref = to_dev(rh)
    rnew = to_dev(rh)
    for d in 1:ndays
        M.update_routing!(mock(rref, runoffs[d], dt))
        route!(router, rnew.area, rnew.discharge, runoffs[d], nsub, Int32(dt) / Float32(nsub))
    end
    sync()
    a1, q1, a2, q2 = Array(rref.area), Array(rref.discharge), Array(rnew.area), Array(rnew.discharge)
    nbad = count(i -> active[i] && (a1[i] !== a2[i] || q1[i] !== q2[i]), eachindex(a1))
    @printf("VALIDATE nsub=%d days=%d differing cells: %d %s\n", nsub, ndays, nbad, nbad == 0 ? "(bitwise identical)" : "FAILED")
end

if BACKEND != "cpu"
    btime(f; n = 10) = (f(); f(); sync(); median([(sync(); t = time_ns(); f(); sync(); (time_ns() - t) / 1.0e6) for _ in 1:n]))
    rref = to_dev(rh); rnew = to_dev(rh); ro = runoffs[1]
    t_ref = btime(() -> M.update_routing!(mock(rref, ro, 86400)))
    t_new = btime(() -> (route!(router, rnew.area, rnew.discharge, ro, 144, 600.0f0); diag!(rnew)))
    t_new_nl = btime(() -> (route!(router, rnew.area, rnew.discharge, ro, 144, 600.0f0; load_state = false); diag!(rnew)))
    @printf("TIME update_routing! %.2f ms | route! + diagnostics %.2f ms (%.2f ms without load_state)\n", t_ref, t_new, t_new_nl)
end
