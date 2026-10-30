# Sub-basin decomposition, level scheduling and the renumbered cell layout (host side).

# Linear index offset of D8 neighbour `b` on a grid with `nx` cells along x. Same bit
# numbering as mGV's `upstream_mask` (see `d8_bit` in mGV's routing.jl).
@inline d8_offset(b, nx) = b < 3 ? -nx + b - 1 : b == 3 ? -1 : b == 4 ? 1 : nx + b - 6

"""
    partition(downstream_idx, active, S)

Greedy split into sub-basins of at most `S` cells. Cells are visited from the headwaters
down; when the uncut cells draining into a cell exceed `S`, its largest upstream branches
are cut off as separate sub-basins. Returns the sub-basin of each cell (0 = inactive), the
sub-basin outlet cells, the earliest level of each sub-basin (1 = headwaters), the sizes,
and the cells in topological order.
"""
function partition(ds::AbstractVector{<:Integer}, active::AbstractVector{Bool}, S::Integer)
    n = length(ds)
    indeg = zeros(Int32, n)
    for i in 1:n
        active[i] && ds[i] > 0 && (indeg[ds[i]] += 1)
    end
    # Topological order, headwaters first
    order = Int32[i for i in 1:n if active[i] && indeg[i] == 0]
    sizehint!(order, count(active))
    k = 1
    while k <= length(order)
        d = ds[order[k]]
        if d > 0
            indeg[d] -= 1
            indeg[d] == 0 && push!(order, d)
        end
        k += 1
    end
    length(order) == count(active) || error("routing network has a cycle")

    # Upstream lists (CSR)
    ptr = zeros(Int32, n + 1)
    for i in 1:n
        active[i] && ds[i] > 0 && (ptr[ds[i] + 1] += 1)
    end
    ptr[1] = 1
    for i in 1:n
        ptr[i + 1] += ptr[i]
    end
    ups = Vector{Int32}(undef, ptr[end] - 1)
    pos = copy(ptr)
    for i in 1:n
        if active[i] && ds[i] > 0
            ups[pos[ds[i]]] = i
            pos[ds[i]] += 1
        end
    end

    pending = zeros(Int32, n)
    is_outlet = falses(n)
    buf = Int32[]
    for c in order
        empty!(buf)
        tot = Int32(1)
        for p in ptr[c]:(ptr[c + 1] - 1)
            push!(buf, ups[p])
            tot += pending[ups[p]]
        end
        if tot > S
            sort!(buf; by = u -> -pending[u])
            for u in buf
                tot <= S && break
                is_outlet[u] = true
                tot -= pending[u]
            end
        end
        pending[c] = tot
        ds[c] <= 0 && (is_outlet[c] = true)
    end

    sb = zeros(Int32, n)
    outlets = Int32[]
    for c in Iterators.reverse(order)
        if is_outlet[c]
            push!(outlets, c)
            sb[c] = length(outlets)
        else
            sb[c] = sb[ds[c]]
        end
    end
    nsb = length(outlets)
    sizes = zeros(Int32, nsb)
    for c in order
        sizes[sb[c]] += 1
    end
    level = ones(Int32, nsb)
    for c in order
        if is_outlet[c] && ds[c] > 0
            d = sb[ds[c]]
            level[d] = max(level[d], level[sb[c]] + 1)
        end
    end
    return (; sb, outlets, level, sizes, order, nsb)
end

# First-fit-decreasing packing of sub-basin sizes into bins of `S` cells; returns bin count
function npacked(sizes, S)
    loads = Int[]
    for z in sort(sizes; rev = true)
        j = findfirst(x -> x + z <= S, loads)
        j === nothing ? push!(loads, z) : (loads[j] += z)
    end
    return length(loads)
end

"""
    reschedule(p, downstream_idx, S, concurrency; fill_fraction = 0.9)

Move sub-basins within their slack (earliest to latest level) so that each level holds
about one wave of `concurrency` units, the number of workgroups the device runs at once.
Levels are filled in order: first the sub-basins that are due (latest level reached), then
others by least slack, until the level fits in whole waves.
"""
function reschedule(p, ds, S, concurrency; fill_fraction = 0.9)
    nlev = maximum(p.level)
    down = zeros(Int32, p.nsb)
    for (s, o) in enumerate(p.outlets)
        ds[o] > 0 && (down[s] = p.sb[ds[o]])
    end
    # Latest level: a downstream sub-basin always has a higher earliest level, so visiting
    # sub-basins by decreasing earliest level finalises the downstream one first.
    latest = fill_latest(p, down, nlev)
    nup = zeros(Int32, p.nsb)
    for s in 1:p.nsb
        down[s] > 0 && (nup[down[s]] += 1)
    end
    level = zeros(Int32, p.nsb)
    ready = Int32[s for s in 1:p.nsb if nup[s] == 0]
    for L in 1:nlev
        sort!(ready; by = s -> (latest[s], -p.sizes[s]))
        must = count(s -> latest[s] == L, ready)
        cells_must = sum(s -> p.sizes[s], view(ready, 1:must); init = 0)
        nw = max(1, cld(cells_must, concurrency * S))
        cap = nw * concurrency * S * fill_fraction
        tot = 0
        k = 0
        for (i, s) in enumerate(ready)
            (i > must && tot + p.sizes[s] > cap) && break
            tot += p.sizes[s]
            k = i
        end
        # back off optional sub-basins until the packed level fits in whole waves
        while k > must && npacked(p.sizes[view(ready, 1:k)], S) > nw * concurrency
            k = max(must, k - max(1, (k - must) ÷ 20))
        end
        for s in view(ready, 1:k)
            level[s] = L
            d = down[s]
            if d > 0
                nup[d] -= 1
                nup[d] == 0 && push!(ready, d)
            end
        end
        ready = ready[(k + 1):end]
    end
    (isempty(ready) && all(>(0), level)) || error("rescheduling failed")
    return (; p..., level)
end

function fill_latest(p, down, nlev)
    latest = fill(Int32(nlev), p.nsb)
    for s in sortperm(p.level; rev = true)
        down[s] > 0 && (latest[s] = latest[down[s]] - 1)
    end
    all(latest .>= p.level) || error("inconsistent sub-basin levels")
    return latest
end

# Pack the sub-basins of each level into units of at most S cells (first fit decreasing over
# the last 64 units, which bounds the cost for levels with many small sub-basins).
function pack_units(p, S)
    nlev = maximum(p.level)
    units = [Vector{Vector{Int32}}() for _ in 1:nlev]
    for l in 1:nlev
        ids = findall(==(l), p.level)
        sort!(ids; by = i -> -p.sizes[i])
        loads = Int[]
        for i in ids
            j = findnext(x -> x + p.sizes[i] <= S, loads, max(1, length(loads) - 63))
            if j === nothing
                push!(loads, p.sizes[i])
                push!(units[l], Int32[i])
            else
                loads[j] += p.sizes[i]
                push!(units[l][j], Int32(i))
            end
        end
    end
    return units
end

"""
    SubbasinLayout(downstream_idx, upstream_mask, nx, active; S = 2048, concurrency = 60)

Host-side decomposition of a D8 routing network on a grid with `nx` cells along x.

- `downstream_idx`: linear index of the downstream cell, or ≤ 0 for an outlet.
- `upstream_mask`: bit `b` set when D8 neighbour `b` drains into the cell (mGV's layout).
- `active`: cells that take part in routing.
- `S`: maximum cells per unit (workgroup); must be one of the kernel sizes in
  `KERNEL_SIZES`.
- `concurrency`: the number of workgroups the device runs at once, used to balance levels
  (about the number of compute units; 60 on an RX 7800 XT).

Cells are renumbered by level, unit, sub-basin and upstream-first. Upstream lists keep D8
bit order, so the summation order equals the gather kernel. An entry `k > 0` is a local
index in the unit, `k < 0` refers to outlet record `-k` of an upstream unit.
"""
struct SubbasinLayout
    S::Int
    perm::Vector{Int32}       # renumbered cell -> grid cell
    ustart::Vector{Int32}     # first renumbered cell of each unit (+ end marker)
    lunit::Vector{Int32}      # units before each level (+ end marker)
    uptr::Vector{Int32}       # upstream list per renumbered cell (CSR)
    uidx::Vector{Int32}
    cell_rec::Vector{Int32}   # outlet record written by the cell, 0 if none
    nrec::Int
end

function SubbasinLayout(ds, upstream_mask, nx, active; S = 2048, concurrency = 60, fill_fraction = 0.9)
    S in KERNEL_SIZES || error("S = $S: choose one of $KERNEL_SIZES")
    ds = Int32.(ds)
    p = partition(ds, active, S)
    p = reschedule(p, ds, S, concurrency; fill_fraction)
    units = pack_units(p, S)

    # cells of each sub-basin, upstream-first
    cnt = zeros(Int32, p.nsb + 1)
    for c in p.order
        cnt[p.sb[c] + 1] += 1
    end
    cnt[1] = 1
    for s in 1:p.nsb
        cnt[s + 1] += cnt[s]
    end
    sbcells = Vector{Int32}(undef, length(p.order))
    pos = copy(cnt)
    for c in p.order
        sbcells[pos[p.sb[c]]] = c
        pos[p.sb[c]] += 1
    end
    perm = Int32[]
    ustart = Int32[1]
    lunit = Int32[0]
    for l in eachindex(units)
        for u in units[l], s in u
            append!(perm, view(sbcells, cnt[s]:(cnt[s + 1] - 1)))
        end
        for u in units[l]
            push!(ustart, ustart[end] + sum(s -> p.sizes[s], u))
        end
        push!(lunit, length(ustart) - 1)
    end
    m = length(perm)
    inv = zeros(Int32, length(ds))
    inv[perm] .= 1:m
    unit_of = zeros(Int32, m)
    for u in 1:(length(ustart) - 1)
        unit_of[ustart[u]:(ustart[u + 1] - 1)] .= u
    end
    level_of_unit = zeros(Int32, length(ustart) - 1)
    for l in 1:(length(lunit) - 1)
        level_of_unit[(lunit[l] + 1):lunit[l + 1]] .= l
    end
    rec_of = zeros(Int32, length(ds))
    nrec = 0
    for o in p.outlets
        if ds[o] > 0
            nrec += 1
            rec_of[o] = nrec
        end
    end
    uptr = Int32[1]
    uidx = Int32[]
    cell_rec = zeros(Int32, m)
    for j in 1:m
        c = perm[j]
        mask = upstream_mask[c]
        for b in 0:7
            if (mask >> b) & 0x01 != 0x00
                up = c + d8_offset(b, nx)
                if p.sb[up] == p.sb[c]
                    push!(uidx, inv[up] - ustart[unit_of[j]] + 1)
                else
                    # an upstream sub-basin must run in an earlier level
                    (rec_of[up] > 0 && level_of_unit[unit_of[inv[up]]] < level_of_unit[unit_of[j]]) ||
                        error("invalid schedule at cell $c")
                    push!(uidx, -rec_of[up])
                end
            end
        end
        push!(uptr, length(uidx) + 1)
        cell_rec[j] = rec_of[c]
    end
    maximum(diff(ustart); init = 0) <= S || error("unit larger than S")
    return SubbasinLayout(S, perm, ustart, lunit, uptr, uidx, cell_rec, nrec)
end

nlevels(L::SubbasinLayout) = length(L.lunit) - 1
nunits(L::SubbasinLayout) = length(L.ustart) - 1
ncells(L::SubbasinLayout) = length(L.perm)

function Base.show(io::IO, L::SubbasinLayout)
    print(io, "SubbasinLayout(S = $(L.S): $(ncells(L)) cells, $(nunits(L)) units, $(nlevels(L)) levels)")
end
