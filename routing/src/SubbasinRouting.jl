"""
Kinematic-wave river routing on a D8 network, decomposed into sub-basins.

Each sub-basin unit runs all routing sub-steps of a model time step in one GPU workgroup,
with per-cell data in registers and the discharge exchange in local memory. Units are
grouped into levels: a unit only depends on units in earlier levels, through their
recorded outlet discharge per sub-step. There is one kernel launch per level.

The arithmetic and summation order equal mGV's `kinematic_wave_kernel!`, so results are
bitwise identical to `mGV.update_routing!`.

Usage:

    layout = SubbasinLayout(downstream_idx, upstream_mask, nx, active)  # once, on the host
    router = SubbasinRouter(backend, layout, length, alpha, cell_area)  # once
    route!(router, area, discharge, runoff, nsub, dt_step)              # every time step
"""
module SubbasinRouting

using KernelAbstractions
using KernelAbstractions.Extras: @unroll

export SubbasinLayout, SubbasinRouter, route!

include("layout.jl")
include("kernels.jl")
include("router.jl")

end
