using SubbasinRouting, KernelAbstractions, Random, Test
const SR = SubbasinRouting

# mGV's D8 bit for an upstream cell at offset (dx, dy) from the cell it drains into
d8_bit(dx, dy) = dy == -1 ? dx + 1 : dy == 0 ? (dx == -1 ? 3 : 4) : dx + 6

# Random D8 network: steepest descent on a tilted random surface (long rivers, several
# basins), with an inactive region. Returns downstream_idx, upstream_mask, active.
function random_network(nx, ny; seed = 1, tilt = 1.0)
    rng = Xoshiro(seed)
    elev = [tilt * (i + 0.5j) + rand(rng) for i in 1:nx, j in 1:ny]
    active = vec([!(i > 0.7nx && j > 0.6ny) for i in 1:nx, j in 1:ny])
    cart = CartesianIndices((nx, ny)); lin = LinearIndices((nx, ny))
    ds = fill(Int32(-1), nx * ny)
    mask = zeros(UInt8, nx * ny)
    for c in 1:(nx * ny)
        active[c] || continue
        i, j = Tuple(cart[c])
        best, bz = 0, elev[i, j]
        for dj in -1:1, di in -1:1
            (di == 0 && dj == 0) && continue
            ii, jj = i + di, j + dj
            (1 <= ii <= nx && 1 <= jj <= ny) || continue
            active[lin[ii, jj]] || continue
            elev[ii, jj] < bz && ((best, bz) = (lin[ii, jj], elev[ii, jj]))
        end
        if best > 0
            ds[c] = best
            dx, dy = Tuple(cart[c]) .- Tuple(cart[best])
            mask[best] |= UInt8(1) << d8_bit(dx, dy)
        end
    end
    return ds, mask, active
end

# Reference: the gather scheme of mGV's kinematic_wave_kernel!, one sweep per sub-step
function reference!(area, discharge, runoff, mask, active, nx, len, alpha, cell_area, nsub, dt)
    n = length(area)
    lat = [(runoff[i] * cell_area[i]) * SR.MM_DAY_TO_M_S for i in 1:n]
    q = copy(discharge); qn = similar(q)
    for _ in 1:nsub
        for i in 1:n
            active[i] || continue
            Q_in = 0.0f0
            for b in 0:7
                (mask[i] >> b) & 0x01 != 0x00 && (Q_in += q[i + SR.d8_offset(b, nx)])
            end
            area[i], qn[i] = SR.kw_update(area[i], q[i], Q_in, lat[i], len[i], alpha[i], dt, 6.0f0)
        end
        q, qn = qn, q
    end
    for i in 1:n
        active[i] && (discharge[i] = q[i])
    end
end

@testset "SubbasinRouting" begin
    nx, ny = 48, 40
    ds, mask, active = random_network(nx, ny)
    deep = random_network(nx, ny; tilt = 3.0)  # long rivers: about 19 levels at S = 64
    n = nx * ny
    rng = Xoshiro(2)
    len = Float32.(5000 .+ 5000 .* rand(rng, n))
    alpha = Float32.(0.2 .+ 3 .* rand(rng, n))
    cell_area = fill(7.0f7, n)
    # very steep cells hit the velocity cap
    alpha[1:7:end] .= 50.0f0

    @testset "layout S=$S concurrency=$C" for S in (64,), C in (2, 8, 60)
        L = SubbasinLayout(ds, mask, nx, active; S, concurrency = C)
        @test sort(L.perm) == findall(active)
        @test maximum(diff(L.ustart)) <= S
        @test SR.nlevels(L) >= 5  # long rivers force several levels
    end

    @testset "bitwise vs gather, $name network, cells_per_thread=$cpt nsub=$nsub" for (name, (ds, mask, active)) in
            (("default", (ds, mask, active)), ("deep", deep)), cpt in (1, 2, 4), nsub in (12, 13, 144)
        L = SubbasinLayout(ds, mask, nx, active; S = 64, concurrency = 4)
        router = SubbasinRouter(CPU(), L, len, alpha, cell_area; cells_per_thread = cpt, max_substeps = 144)
        a_ref = zeros(Float32, n); q_ref = zeros(Float32, n)
        a = fill(NaN32, n); q = fill(NaN32, n)  # inactive cells must stay untouched
        a[active] .= 0; q[active] .= 0
        for day in 1:3
            runoff = Float32.(20 .* rand(Xoshiro(10 + day), n))
            reference!(a_ref, q_ref, runoff, mask, active, nx, len, alpha, cell_area, nsub, 600.0f0)
            route!(router, a, q, runoff, nsub, 600.0f0)
        end
        @test a[active] == a_ref[active]
        @test q[active] == q_ref[active]
        @test all(isnan, a[.!active]) && all(isnan, q[.!active])
        @test maximum(q[active]) > 0
    end

    @testset "state changed outside the router is picked up" begin
        L = SubbasinLayout(ds, mask, nx, active; S = 64, concurrency = 4)
        router = SubbasinRouter(CPU(), L, len, alpha, cell_area)
        runoff = fill(5.0f0, n)
        a = zeros(Float32, n); q = zeros(Float32, n)
        route!(router, a, q, runoff, 12, 600.0f0)
        a .*= 2  # e.g. set through BMI
        a_ref, q_ref = copy(a), copy(q)
        route!(router, a, q, runoff, 12, 600.0f0)
        reference!(a_ref, q_ref, runoff, mask, active, nx, len, alpha, cell_area, 12, 600.0f0)
        @test a[active] == a_ref[active] && q[active] == q_ref[active]
    end

    @testset "argument checks" begin
        @test_throws ErrorException SubbasinLayout(ds, mask, nx, active; S = 100)
        L = SubbasinLayout(ds, mask, nx, active; S = 64)
        r = SubbasinRouter(CPU(), L, len, alpha, cell_area; max_substeps = 10)
        @test_throws ErrorException route!(r, zeros(Float32, n), zeros(Float32, n), zeros(Float32, n), 11, 600.0f0)
    end
end
