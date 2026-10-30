# Device kernels.

# mm/day -> m/s, the same constant as mGV's lateral_inflow_kernel!
const MM_DAY_TO_M_S = 1.15740741f-8

# Unit sizes (S) and cells per thread for which a sub-basin kernel is generated. The
# workgroup size is S ÷ cells_per_thread (at most 1024).
const KERNEL_SIZES = (64, 256, 1024, 2048, 4096)
const CELLS_PER_THREAD = (1, 2, 4)

# One cell update: identical arithmetic to mGV's kinematic_wave_kernel!
@inline function kw_update(A_old, Q_old, Q_in, lat, len, alpha, dt, vmax)
    Q_total_in = Q_in + lat
    dAdt = (Q_total_in - Q_old) / len
    A_new = max(A_old + dAdt * dt, 0.0f0)
    Q_new = alpha * (A_new * cbrt(A_new)^2)
    if A_new > 1.0f-6
        v = Q_new / A_new
        if v > vmax
            v = vmax
            Q_new = v * A_new
        end
    end
    return A_new, Q_new
end

@inline unit_range(ustart, u) = (ustart[u] - Int32(1), ustart[u + 1] - ustart[u])

# Sub-basin kernel: one workgroup per unit, all sub-steps in one launch.
# Each thread owns CPT cells (j = tid + (c - 1) * TT); their data is held in @private
# arrays (registers). Only the double-buffered discharge is in local memory, because
# neighbours read it. Written for KernelAbstractions' CPU semantics too: nothing per item
# is carried across a @synchronize outside @private arrays, so the unit range is recomputed
# and the buffer parity follows from the sub-step number.
for CAP in KERNEL_SIZES, CPT in CELLS_PER_THREAD
    TT = CAP ÷ CPT
    TT > 1024 && continue
    name = Symbol(:subbasin_kernel_, CAP, :_, CPT)
    @eval @kernel inbounds = true function $name(area, discharge, @Const(lateral), @Const(len), @Const(alpha),
            @Const(ustart), @Const(uptr), @Const(uidx), outq, @Const(cell_rec), nsub, dt, unit0, vmax)
        tid = @index(Local, Linear)
        g = @index(Group, Linear)
        stride = @uniform nsub + Int32(1)
        Q = @localmem Float32 ($(2CAP),)
        A = @private Float32 ($CPT,)
        Ls = @private Float32 ($CPT,)
        Ln = @private Float32 ($CPT,)
        La = @private Float32 ($CPT,)
        rc = @private Int32 ($CPT,)
        p0 = @private Int32 ($CPT,)
        nup = @private Int32 ($CPT,)
        e1 = @private Int32 ($CPT,)
        e2 = @private Int32 ($CPT,)
        e3 = @private Int32 ($CPT,)

        # Load per-cell data into registers and discharge into local memory; record the
        # initial outlet discharge (sub-step 0)
        c0, nc = unit_range(ustart, unit0 + g)
        @unroll for c in 1:$CPT
            j = tid + (c - 1) * $TT
            nup[c] = Int32(0); rc[c] = Int32(0); p0[c] = Int32(1)
            e1[c] = Int32(0); e2[c] = Int32(0); e3[c] = Int32(0)
            if j <= nc
                i = c0 + j
                A[c] = area[i]; Ls[c] = lateral[i]; Ln[c] = len[i]; La[c] = alpha[i]
                rc[c] = cell_rec[i]; p0[c] = uptr[i]; nup[c] = uptr[i + 1] - uptr[i]
                nup[c] >= 1 && (e1[c] = uidx[p0[c]])
                nup[c] >= 2 && (e2[c] = uidx[p0[c] + Int32(1)])
                nup[c] >= 3 && (e3[c] = uidx[p0[c] + Int32(2)])
                q = discharge[i]
                Q[j] = q
                rc[c] > 0 && (outq[(rc[c] - 1) * stride + 1] = q)
            end
        end
        @synchronize
        for t in Int32(1):nsub
            c0, nc = unit_range(ustart, unit0 + g)
            src = isodd(t) ? 0 : $CAP
            dst = $CAP - src
            @unroll for c in 1:$CPT
                j = tid + (c - 1) * $TT
                if j <= nc
                    # Inflow from upstream cells in D8 order: local cells from the previous
                    # sub-step, upstream units from their outlet record of sub-step t - 1
                    Q_in = 0.0f0
                    for q in Int32(1):nup[c]
                        e = q == 1 ? e1[c] : q == 2 ? e2[c] : q == 3 ? e3[c] : uidx[p0[c] + q - Int32(1)]
                        Q_in += e > 0 ? Q[src + e] : outq[(-e - 1) * stride + t]
                    end
                    a, qn = kw_update(A[c], Q[src + j], Q_in, Ls[c], Ln[c], La[c], dt, vmax)
                    A[c] = a
                    Q[dst + j] = qn
                    rc[c] > 0 && (outq[(rc[c] - 1) * stride + t + 1] = qn)
                end
            end
            @synchronize
        end
        c0, nc = unit_range(ustart, unit0 + g)
        last = isodd(nsub) ? $CAP : 0  # buffer written by the last sub-step
        @unroll for c in 1:$CPT
            j = tid + (c - 1) * $TT
            if j <= nc
                area[c0 + j] = A[c]
                discharge[c0 + j] = Q[last + j]
            end
        end
    end
end
subbasin_kernel(cap, cpt) = getfield(@__MODULE__, Symbol(:subbasin_kernel_, cap, :_, cpt))

# Lateral inflow [m3/s] from runoff [mm/day], straight into the renumbered layout
@kernel inbounds = true function lateral_kernel!(lateral, @Const(runoff), @Const(cell_area), @Const(perm))
    j = @index(Global, Linear)
    if j <= length(lateral)
        lateral[j] = (runoff[perm[j]] * cell_area[j]) * MM_DAY_TO_M_S
    end
end

# Grid arrays -> renumbered layout
@kernel inbounds = true function gather_kernel!(area, discharge, @Const(area_grid), @Const(discharge_grid), @Const(perm))
    j = @index(Global, Linear)
    if j <= length(area)
        area[j] = area_grid[perm[j]]
        discharge[j] = discharge_grid[perm[j]]
    end
end

# Renumbered layout -> grid arrays
@kernel inbounds = true function scatter_kernel!(area_grid, discharge_grid, @Const(area), @Const(discharge), @Const(perm))
    j = @index(Global, Linear)
    if j <= length(area)
        area_grid[perm[j]] = area[j]
        discharge_grid[perm[j]] = discharge[j]
    end
end
