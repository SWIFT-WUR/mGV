# SubbasinRouting

An alternative implementation of mGV's kinematic-wave river routing for GPUs. It gives
the same results as `mGV.update_routing!`, bitwise, and is much faster on large domains.
It is a separate package and is not used by mGV yet.

## How it works

The D8 network is split into sub-basins of at most `S` cells (default 2048).

1. **One workgroup per unit.** A unit (one or more sub-basins) runs all sub-steps of a
   time step in one GPU workgroup.
   - Each thread keeps the data of its cells in registers.
   - Only the discharge, which neighbouring cells read, is in local memory.
   - A barrier separates the sub-steps.
2. **Levels.** Units are grouped into levels. A unit reads the outflow of upstream units
   from their *outlet records*: the outlet discharge at every sub-step, written by an
   earlier level. There is one kernel launch per level.
3. **Rescheduling.** A sub-basin can run in any level between its earliest one (after
   all its upstream sub-basins) and its latest one (still in time for its downstream
   sub-basins). Sub-basins are moved within that range so that every level holds about
   one wave of `concurrency` workgroups, the number the device runs at once.

The cell update and the order of summing upstream inflow are the same as in mGV's
`kinematic_wave_kernel!`.

## Usage

```julia
using SubbasinRouting

# Once, on the host: decomposition and renumbering
layout = SubbasinLayout(routing.downstream_idx, routing.upstream_mask, nx, active;
    S = 2048, concurrency = 60)

# Once: device arrays (length, alpha, cell_area per grid cell)
router = SubbasinRouter(backend, layout, routing.length, routing.alpha, routing.cell_area;
    cells_per_thread = 2, max_substeps = 144)

# Every model time step: reads and writes the model's grid vectors
route!(router, routing.area, routing.discharge, total_runoff, n_substeps, dt_step)
```

- **Inputs:**
  - `active`: the cells that take part in routing, e.g. `isfinite.(routing.cell_area)`.
  - `routing.upstream_mask`: mGV's D8 bitmask of upstream neighbours.
- **State and output:**
  - `route!` reads the state from `area`/`discharge` first, so changes made through BMI
    are respected. Pass `load_state = false` to skip that gather.
  - It leaves inactive cells untouched.
  - It doesn't compute the CFL and travel-time diagnostics; run mGV's
    `routing_diagnostics_kernel!` afterwards.
- **`S` and `cells_per_thread`:** `S` must be one of `KERNEL_SIZES`. The workgroup size is
  `S ÷ cells_per_thread`, at most 1024.
  - Fastest on an RX 7800 XT: `S = 2048, cells_per_thread = 2`, or
    `S = 4096, cells_per_thread = 4` (fewer levels).
  - On KernelAbstractions' CPU backend, use `S = 64`: large workgroups compile very
    slowly there. That size is meant for testing.
- **`concurrency`:** about the number of compute units (60 on an RX 7800 XT).

## Performance

On an RX 7800 XT, per daily time step of 144 sub-steps, including lateral inflow, the
gather and scatter, and mGV's diagnostics kernel:

| Domain | `update_routing!` | `route!` |
|---|---|---|
| Global 5′ (4320×1680, 2.28M land cells), S=2048 | 58.4 ms | 7.35 ms |
| Global 5′, S=4096, 4 cells per thread | 59.0 ms | 7.04 ms |
| Indus (204×180) | 1.18 ms | 0.96 ms |

`update_routing!` runs over every grid cell, ocean included. Against a per-sub-step gather
over land cells only (21.2 ms globally), the gain is about 2.7×. See
`docs/subbasin-routing-results.md` for the measurements and the variants that were tried.

## Tests and scripts

```sh
# Unit tests on the CPU backend (Pkg.test runs with bounds checking): synthetic D8
# networks, compared bitwise with a plain per-sub-step gather
julia --project=routing -e 'using Pkg; Pkg.test()'

# Bitwise check against mGV.update_routing! for 144 and 143 sub-steps, and timing on GPU
julia --project=routing/scripts routing/scripts/validate_mgv.jl cpu indus 64 2 2
julia --project=routing/scripts routing/scripts/validate_mgv.jl rocm <global_5min.nc> 2048 2 3

# Global 5' D8 network from the HydroSHEDS 30" DEM (Python with pyflwdir, rasterio, netCDF4)
PRE=2 python -I routing/scripts/make_routing.py hyd_glo_dem_30s.tif global_5min.nc
```

The DEM is at
`https://data.hydrosheds.org/file/hydrosheds-v1-dem/hyd_glo_dem_30s.zip`. `PRE=2`
aggregates it to 1′ before deriving flow directions; the full 30″ run needs more than
36 GB of RAM.

**Untested so far:** CUDA, Metal and oneAPI. The kernels use only portable
KernelAbstractions features (`@localmem`, `@private`, `@synchronize`).
