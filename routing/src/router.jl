# Device-side state and the routing step.

"""
    SubbasinRouter(backend, layout, length, alpha, cell_area; cells_per_thread = 2, max_substeps = 144)

Device arrays for routing with `layout`. `length`, `alpha` and `cell_area` are per grid
cell (host or device vectors); they are permuted into the renumbered layout once.

`cells_per_thread` sets the workgroup size `S ÷ cells_per_thread` (at most 1024). Two
cells per thread at S = 2048, or four at S = 4096, were fastest on an RX 7800 XT.
"""
struct SubbasinRouter{B, V, I}
    backend::B
    layout::SubbasinLayout
    cells_per_thread::Int
    max_substeps::Int
    perm::I
    ustart::I
    uptr::I
    uidx::I
    cell_rec::I
    length::V
    alpha::V
    cell_area::V
    lateral::V
    area::V
    discharge::V
    outq::V       # outlet discharge of each record at sub-steps 0..max_substeps
end

function SubbasinRouter(backend, L::SubbasinLayout, len, alpha, cell_area; cells_per_thread = 2, max_substeps = 144)
    cells_per_thread in CELLS_PER_THREAD || error("cells_per_thread must be one of $CELLS_PER_THREAD")
    L.S ÷ cells_per_thread <= 1024 || error("workgroup size S ÷ cells_per_thread must be ≤ 1024")
    dev(x) = (y = KernelAbstractions.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)
    perm = L.perm
    host(x) = Float32.(Array(x))[perm]
    m = ncells(L)
    z(n) = KernelAbstractions.zeros(backend, Float32, n)
    return SubbasinRouter(backend, L, cells_per_thread, max_substeps,
        dev(perm), dev(L.ustart), dev(L.uptr), dev(L.uidx), dev(L.cell_rec),
        dev(host(len)), dev(host(alpha)), dev(host(cell_area)),
        z(m), z(m), z(m), z(max(1, L.nrec) * (max_substeps + 1)))
end

"""
    route!(router, area, discharge, runoff, nsub, dt_step; max_velocity = 6f0, load_state = true)

Route one model time step of `nsub` sub-steps of `dt_step` seconds. `area` and `discharge`
are the model's grid vectors (channel cross-section [m2] and discharge [m3/s]); `runoff`
is the grid runoff [mm/day]. With `load_state = true` the state is read from `area` and
`discharge` first, so changes made to them outside the router (e.g. through BMI) are
respected; it costs one gather over the active cells. The new state is written back to
`area` and `discharge`. Inactive cells are left untouched.
"""
function route!(r::SubbasinRouter, area, discharge, runoff, nsub, dt_step; max_velocity = 6.0f0, load_state = true)
    nsub <= r.max_substeps || error("nsub = $nsub exceeds max_substeps = $(r.max_substeps)")
    L = r.layout
    backend = r.backend
    m = Int32(ncells(L))
    m == 0 && return nothing
    load_state && gather_kernel!(backend)(r.area, r.discharge, area, discharge, r.perm; ndrange = m)
    lateral_kernel!(backend)(r.lateral, vec(runoff), r.cell_area, r.perm; ndrange = m)
    T = L.S ÷ r.cells_per_thread
    k! = subbasin_kernel(L.S, r.cells_per_thread)(backend, T)
    for l in 1:nlevels(L)
        nu = L.lunit[l + 1] - L.lunit[l]
        nu == 0 && continue
        k!(r.area, r.discharge, r.lateral, r.length, r.alpha, r.ustart, r.uptr, r.uidx, r.outq, r.cell_rec,
            Int32(nsub), Float32(dt_step), Int32(L.lunit[l]), Float32(max_velocity); ndrange = nu * T)
    end
    scatter_kernel!(backend)(area, discharge, r.area, r.discharge, r.perm; ndrange = m)
    return nothing
end
