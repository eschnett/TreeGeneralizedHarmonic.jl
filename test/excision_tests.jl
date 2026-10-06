# Excision: the `:excised` interior variant (added in step X2b).
#
# `CODE.md`, "Excision", and `PLAN.md`, step X2b. The claims, on the step-5
# fixture at `q = 2`, `N = 8` (`h = 5/64` at the hole) with the excision
# surface at `r_E = 3/4`: the per-point classes are the geometry's; the
# closures are exact where they claim to be; where no tap is excised the
# operator is today's, bit for bit; no kernel reads an excised value; the two
# geometries are one operator; the monitors read no excised value; the
# refusals fire, each by name; and a run goes, and restarts, as the run.
#
# The run file of the excision round, so it prices its claims: the fixture's
# problem is built once and shared by the testsets that only evaluate on it.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using Random: Xoshiro
using StaticArrays: SVector

const TGHx = TreeGeneralizedHarmonic

@testset verbose = true "Excision (step X2b)" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    N = 8
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    r_E = T(3 // 4)
    case = hole_fixture(T; q=q, variant=:excised, r_1=r_E)
    forest = hole_fixture_forest(T, case; N=N)

    # A problem on the fixture's mesh with the case's initial data, and its
    # state vector.
    function setup(c; interior=c.interior, perturb=nothing)
        U = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        fill_exact!(U, c, zero(T); interior=interior)
        perturb === nothing || perturb(U)
        p = GHProblem(U, GhostSchedule(U, ops), c; q=q, interior=interior)
        u = statevector(U)
        gather!(u, U)
        return p, u
    end
    p, u = setup(case)
    ex = p.excision
    U = p.U
    du = similar(u)
    gh_rhs!(du, u, p, zero(T))
    A = statearray(du, U)
    owned = CartesianIndices((N, N, N))
    # The class of owned point `I` of block `b`.
    cls(ex, I, b) = ex.classes[I[1] + G, I[2] + G, I[3] + G, b]
    mask = interior_mask(case.interior, zero(T))

    @testset "the classes are the geometry's, the ghosts their owners', the zone the enumeration's" begin
        # Guards the single source of truth. A class array that disagreed with
        # the masks' predicate anywhere — an owned point, a ghost the exchange
        # filled, an outer face the hook filled — would make the kernels and
        # the norms exclude different sets; a zone narrower than the
        # enumeration would leave a centered stencil reading an excised value,
        # one wider than it would put a closure where today's operator runs.
        nbad = 0
        for b in 1:nblocks(U), S in CartesianIndices(size(ex.classes)[1:3])
            x = coordinates(U, b, Tuple(S))
            want = !is_evolved(mask, x)
            nbad += (ex.classes[S, b] == TGHx.CLASS_EXCISED) != want
        end
        @test nbad == 0
        h = minimum_spacing(T, forest)
        tapped(x) = any(!is_evolved(mask, x .+ off) for off in taps)
        taps = NTuple{3,Float64}[]
        for d in 1:3, a in (-G):G
            a == 0 || push!(taps, ntuple(k -> k == d ? a * h : 0.0, 3))
        end
        for (i, j) in ((1, 2), (1, 3), (2, 3)), a in -(q ÷ 2):(q ÷ 2),
            e in -(q ÷ 2):(q ÷ 2)
            push!(taps, ntuple(k -> k == i ? a * h : k == j ? e * h : 0.0, 3))
        end
        nzone = 0
        nexc = 0
        wrong = 0
        for b in 1:nblocks(U), I in owned
            x = coordinates(U, b, Tuple(I) .+ G)
            c = cls(ex, I, b)
            if !is_evolved(mask, x)
                nexc += 1
            else
                z = tapped(x)
                nzone += z
                wrong += z != (c == TGHx.CLASS_ZONE)
            end
        end
        @test wrong == 0
        @test (ex.nzone, ex.nexcised) == (nzone, nexc)
        @test ex.nzone + ex.nexcised + ex.ncentered == nblocks(U) * N^3
        # The numbers on the record: a ball of radius 3/4 at h = 5/64, and
        # its band; the zone lives in 32 of the 64 fine blocks.
        @test (ex.nexcised, ex.nzone, ex.nzoneblocks) == (3743, 2192, 32)
        @test ex.W == 2 * h && ex.h == h
        @test count(Array(ex.zoneblocks)) == ex.nzoneblocks
    end

    @testset "the closures are exact on polynomials across faces, edges and corners" begin
        # Guards the closure provider's indexing — a table read at the wrong
        # `[slot, k⁻ + 1, k⁺ + 1]`, a node range off by one, a mixed derivative
        # whose inner closure is the outer point's rather than the outer
        # node's. On one block of a host working array with an excised
        # half-space — `i ≤ c` (faces), `i + j ≤ c` (edges: two axis neighbours
        # excised), `i + j + k ≤ c` (corners: three) — every evolved point
        # within reach of it evaluates, on a polynomial of the degree every
        # closure is exact to (`q/2 + 1`), `d1`, `d2`, the nested `dmix` and the
        # lopsided `adv` exactly, and `ko` annihilates the degree below `G`
        # that `:msn` keeps near the surface.
        for qq in (2, 4)
            GG = qq ÷ 2 + 1
            n = 4GG + 5
            deg = qq ÷ 2 + 1
            rng = Xoshiro(qq)
            mons = [(a, b, c) for a in 0:deg for b in 0:deg for c in 0:deg
                    if a + b + c ≤ deg]
            coef = [T(rand(rng, -9:9)) / 8 for _ in mons]
            # `∂^e P` exactly, `e` the derivative count per axis.
            ff(a, e) = e == 0 ? 1 : e == 1 ? a : a * (a - 1)
            dpoly(x, e) = sum(coef[m] * prod(ff(p[k], e[k]) == 0 ? 0.0 :
                                             ff(p[k], e[k]) * x[k]^(p[k] - e[k])
                                             for k in 1:3)
                              for (m, p) in enumerate(mons))
            P(x) = dpoly(x, (0, 0, 0))
            unit(d, k=1) = ntuple(l -> l == d ? k : 0, 3)
            # A polynomial of degree below `G`, which `:msn` annihilates.
            linp(x) = 1.5 - 0.25 * x[1] + 0.75 * x[2] - 0.5 * x[3] +
                     (GG > 2 ? 0.125 * x[1] * x[2] - 0.375 * x[3]^2 : 0.0)
            tab = TGHx.closure_arrays(T, Val(qq), :msn, TreeGeneralizedHarmonic.CPU())
            for shape in (:face, :edge, :corner)
                c0 = 2GG + 3 + (shape === :face ? 0 : shape === :edge ? n ÷ 2 : n)
                excised(I) = shape === :face ? I[1] ≤ c0 :
                             shape === :edge ? I[1] + I[2] ≤ c0 : I[1] + I[2] + I[3] ≤ c0
                work = Array{T}(undef, n, n, n, 20, 1)
                cl = Array{UInt8}(undef, n, n, n, 1)
                for I in CartesianIndices((n, n, n))
                    x = T.(Tuple(I))
                    for v in 1:20
                        work[I, v, 1] = v == 1 ? P(x) : v == 2 ? linp(x) : T(NaN)
                    end
                    cl[I, 1] = excised(Tuple(I)) ? TGHx.CLASS_EXCISED : TGHx.CLASS_ZONE
                    excised(Tuple(I)) && (work[I, 1, 1] = work[I, 2, 1] = T(NaN))
                end
                wst, wsv, _ = TGHx.work_strides(work)
                npts = 0
                worst = 0.0
                kos = 0.0
                for I in CartesianIndices((n, n, n))
                    t = Tuple(I)
                    (all(d -> GG < t[d] ≤ n - GG, 1:3) && !excised(t)) || continue
                    # Within reach of the excised set along some axis.
                    any(excised(Base.setindex(t, t[d] + s * a, d))
                        for d in 1:3, s in (-1, 1), a in 1:GG) || continue
                    base = 1 + (t[1] - 1) * wst[1] + (t[2] - 1) * wst[2] +
                           (t[3] - 1) * wst[3]
                    S = TGHx.closure_provider(T, Val(GG), wst, cl, base, tab, one(T),
                                              one(T))
                    x = T.(t)
                    for d in 1:3
                        worst = max(worst, abs(TGHx.d1(S, work, base, d) -
                                               dpoly(x, unit(d))))
                        worst = max(worst, abs(TGHx.d2(S, work, base, d) -
                                               dpoly(x, unit(d, 2))))
                        for β_d in (-one(T), one(T))
                            worst = max(worst, abs(TGHx.adv(S, β_d, zero(T), work,
                                                            base, d) -
                                                   dpoly(x, unit(d))))
                        end
                        kos = max(kos, abs(TGHx.ko(S, work, base + wsv, d)))
                    end
                    for (i, j) in ((1, 2), (1, 3), (2, 3))
                        worst = max(worst, abs(TGHx.dmix(S, work, base, i, j) -
                                               dpoly(x, unit(i) .+ unit(j))))
                    end
                    npts += 1
                end
                scale = maximum(abs, filter(isfinite, work[:, :, :, 1:2, 1]))
                @test npts > 0
                @test worst ≤ 1e-12 * scale
                @test kos ≤ 1e-12 * scale
            end
        end
    end

    # The working array of `p` with the ghosts of `u` filled, and a point's
    # linear indices into it and into the class array.
    TGHx.scatter!(U, u)
    fill_ghosts!(U, p.schedule; boundary=dirichlet(case, zero(T)))
    st, sv, sb = TGHx.work_strides(U.work)
    lin(I, b) = (1 + (b - 1) * sb + (I[1] + G - 1) * st[1] + (I[2] + G - 1) * st[2] +
                 (I[3] + G - 1) * st[3],
                 1 + (b - 1) * sv + (I[1] + G - 1) * st[1] + (I[2] + G - 1) * st[2] +
                 (I[3] + G - 1) * st[3])

    @testset "with no excised tap the closure provider is the centered one" begin
        # Guards "the exterior is unchanged bit for bit" at the provider: the
        # closure table's centered rows contracted over their nodes in
        # ascending order must be `axis_stencil`'s and `mixed_stencil`'s
        # arithmetic, so that a centered point reached by the closure provider
        # — the inner closures of a nested mixed derivative, a zone point's
        # far side — is today's. Each contraction is claimed `isequal`. The
        # whole `F` is one body compiled for two providers (`CLAUDE.md`, "Two
        # spellings of one expression"): it was `isequal` on Apple silicon
        # until step X4, and since `main`'s head (merged in step X4) the two
        # heads — one around the closure provider's loops over its table, one
        # around the centered provider's unrolled stencils — fuse
        # `metric_quantities`' StaticArrays `muladd`s differently (`β` by 2.8
        # eps, `A^{ij}` by 6.8). In a stationary background `F` is the small
        # difference of `O(1)` terms, so that is up to 238 eps of `F`'s own
        # largest value at these points: the whole `F` is held to roundoff as
        # the suite's other comparisons of two specialisations are, 512 eps of
        # each variable's largest `|du|` on the mesh (measured: 103, at 1682
        # of the 4508 points; amended in step X4).
        C = TGHx.Centered(T, Val(q), st)
        nc = 0
        each = true
        bitwise = true
        nbitwise = 0
        dmax = zeros(20)
        for b in 1:nblocks(U), I in owned
            cls(ex, I, b) == TGHx.CLASS_CENTERED || continue
            # Centered points next to the band: their taps are evolved points
            # with codes below `G` on one side, which the closure provider
            # reads through its own tables.
            x = coordinates(U, b, Tuple(I) .+ G)
            sqrt(sum(abs2, x)) < r_E + 4 * ex.h || continue
            var, cb = lin(I, b)
            S = TGHx.closure_provider(T, Val(G), st, ex.classes, cb, ex.table,
                                      inv(spacing(T, forest, forest.leaves[b])),
                                      zero(T))
            for v in (1, 6, 11, 18), d in 1:3
                bv = var + (v - 1) * sv
                each &= isequal(TGHx.d1(S, U.work, bv, d), TGHx.d1(C, U.work, bv, d))
                each &= isequal(TGHx.d2(S, U.work, bv, d), TGHx.d2(C, U.work, bv, d))
                each &= isequal(TGHx.ko(S, U.work, bv, d), TGHx.ko(C, U.work, bv, d))
            end
            for v in (1, 6), (i, j) in ((1, 2), (1, 3), (2, 3))
                bv = var + (v - 1) * sv
                each &= isequal(TGHx.dmix(S, U.work, bv, i, j),
                                TGHx.dmix(C, U.work, bv, i, j))
            end
            inv_h = inv(spacing(T, forest, forest.leaves[b]))
            γ0 = damping_rate(case.γ0, zero(T), x)
            εh = dissipation_rate(case.ε_KO, zero(T), x) * inv_h
            Hw = TGHx.gauge_work(p.Hsrc)
            F1 = TGHx.gh_rhs_at_point(S, T, U.work, Hw, Tuple(I), b, var, sv, inv_h,
                                      γ0, case.γ2, εh, Val(true), Val(true))
            F0 = TGHx.gh_rhs_at_point(T, U.work, Hw, Tuple(I), b, var, st, sv, inv_h,
                                      γ0, case.γ2, εh, Val(q), Val(true), Val(true))
            bitwise &= isequal(F1, F0)
            nbitwise += isequal(F1, F0)
            dmax .= max.(dmax, abs.(vcat(F1[1], F1[2]) .- vcat(F0[1], F0[2])))
            nc += 1
        end
        vscale = [maximum(abs, A[:, :, :, v, :]) for v in 1:20]
        worst = maximum(dmax ./ vscale)
        @info "the closure provider's F at $nc centered points next to the band: " *
              (bitwise ? "bit for bit" :
               "$nbitwise bit for bit, the rest within $(worst / eps(T)) eps of " *
               "each variable's largest |du|") * " the centered one's"
        @test nc > 1000
        @test each
        @test bitwise || worst ≤ 512 * eps(T)
    end

    @testset "no kernel reads an excised value" begin
        # Guards `0 · NaN = NaN` and every way of reading the excised set: a
        # stencil with a zero weight on it, a closure that reaches past its
        # codes, a ghost an exchange copied from an excised point. A
        # degenerate metric (`h = −η`, so `g = 0`, and `Π = NaN`) planted on
        # every excised owned point must leave every non-excised `du` bit for
        # bit the clean state's, and every excised `du` exactly zero.
        u2 = copy(u)
        S2 = statearray(u2, U)
        η = (1, 0, 0, 0, -1, 0, 0, -1, 0, -1)
        for b in 1:nblocks(U), I in owned
            cls(ex, I, b) == TGHx.CLASS_EXCISED || continue
            for v in 1:10
                S2[I, v, b] = η[v]
                S2[I, 10 + v, b] = T(NaN)
            end
        end
        du2 = similar(u)
        gh_rhs!(du2, u2, p, zero(T))
        B = statearray(du2, U)
        same = 0
        zero_ = 0
        for b in 1:nblocks(U), I in owned
            if cls(ex, I, b) == TGHx.CLASS_EXCISED
                zero_ += all(v -> A[I, v, b] === zero(T) && B[I, v, b] === zero(T), 1:20)
            else
                same += all(v -> isequal(A[I, v, b], B[I, v, b]), 1:20)
            end
        end
        @test same == ex.nzone + ex.ncentered
        @test zero_ == ex.nexcised
    end

    @testset "the centered points' du is the unexcised operator's, and the zone points' is not" begin
        # Guards the main kernel's `:excised` branch: at a centered point it
        # must be today's `F` — the `:none` kernel's, on the same state — and
        # at a zone point the zone kernel's closures, which differ from the
        # centered operator reading the core rule's data by far more than
        # roundoff. To `512 eps` (two kernel specialisations, `CLAUDE.md`).
        none = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10),
                                interior=nothing)
        Un = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        pn = GHProblem(Un, GhostSchedule(Un, ops), none; q=q)
        dun = similar(u)
        gh_rhs!(dun, u, pn, zero(T))
        C = statearray(dun, Un)
        worst = 0.0
        scale = 0.0
        nbit = 0
        zmin = Inf
        for b in 1:nblocks(U), I in owned
            c = cls(ex, I, b)
            d = maximum(v -> abs(A[I, v, b] - C[I, v, b]), 1:20)
            if c == TGHx.CLASS_CENTERED
                worst = max(worst, d)
                scale = max(scale, maximum(v -> abs(A[I, v, b]), 1:20))
                nbit += all(v -> isequal(A[I, v, b], C[I, v, b]), 1:20)
            elseif c == TGHx.CLASS_ZONE
                zmin = min(zmin, d)
            end
        end
        @info "centered points: $nbit of $(ex.ncentered) bit for bit the :none " *
              "kernel's, worst $(worst / (eps(T) * scale)) eps; zone points differ " *
              "by at least $zmin"
        @test worst ≤ 512 * eps(T) * scale
        @test zmin > 1e6 * 512 * eps(T) * scale
    end

    @testset "an excised sphere and a FittedInterior holding it are one operator" begin
        # Guards the two geometries' protocol: the tracked geometry's excised
        # set is `d > 0` below its offset surface, the sphere's `r < r_1`, and
        # a `FittedInterior` whose surface *is* that sphere (a degree-0 shape
        # with `r_in = r_out = 2`, offset `2 − 3/4`) must give the same classes
        # and the same `du` — `isequal` where the platform gives it.
        fint = FittedInterior(T; center=(0, 0, 0),
                              shape=SVector(2 * sqrt(4 * T(π))), lmax=0,
                              offset=2 - r_E, thickness=r_E - T(2 // 5),
                              variant=:excised, margin=8,
                              h=minimum_spacing(T, forest), r_in=2, r_out=2)
        pf, uf = setup(case; interior=fint)
        @test isequal(uf, u)
        @test pf.excision.classes == ex.classes
        @test (pf.excision.nzone, pf.excision.nexcised) == (ex.nzone, ex.nexcised)
        duf = similar(u)
        gh_rhs!(duf, u, pf, zero(T))
        bitwise = isequal(duf, du)
        @info "the tracked geometry's du is " *
              (bitwise ? "bit for bit" : "$(maximum(abs, duf - du) / (eps(T) * maximum(abs, du))) eps of") *
              " the sphere's"
        @test bitwise || maximum(abs, duf - du) ≤ 64 * eps(T) * maximum(abs, du)
    end

    @testset "the lopsided blend is off above its shell, and one operator in both kernels" begin
        # Guards the blend's two promises (`CODE.md`, "Excision"): beyond the
        # depth where it starts it is not there at all — `adv` returns its
        # argument through a branch, so the exterior is today's operator bit
        # for bit — and the main kernel's `Lopsided` and the zone kernel's
        # closure provider are one lopsided derivative, the table's centered
        # row being `lopsided_centered_weights` bit for bit. With the blend
        # from one cell below the horizon, full at five: every point above
        # `r_h − h` keeps the blend-free `du`, and the points below it where
        # the blend's weight is at least a hundredth do not.
        up = hole_fixture(T; q=q, variant=:excised, r_1=r_E,
                          excision=Excision(T; upwind=(1, 4)))
        pu, _ = setup(up)
        @test pu.excision.classes == ex.classes
        duu = similar(u)
        gh_rhs!(duu, u, pu, zero(T))
        B = statearray(duu, U)
        bl = pu.excision.blend
        above = 0
        same = 0
        below = 0
        moved = 0
        zmoved = 0
        for b in 1:nblocks(U), I in owned
            c = cls(ex, I, b)
            c == TGHx.CLASS_EXCISED && continue
            x = coordinates(U, b, Tuple(I) .+ G)
            λ = blend_weight(bl, x)
            if λ == 0
                above += 1
                same += all(v -> isequal(A[I, v, b], B[I, v, b]), 1:20)
            elseif c == TGHx.CLASS_ZONE
                zmoved += any(v -> A[I, v, b] != B[I, v, b], 1:20)
            elseif λ ≥ 1 // 100
                # (Just inside the start the blend's weight is `O(s³)` and can
                # round away; a hundredth of the lopsided derivative cannot.)
                below += 1
                moved += any(v -> A[I, v, b] != B[I, v, b], 1:20)
            end
        end
        @test same == above && above > 0
        @test moved == below && below > ex.nzone
        # At `q = 2` the lopsided derivative of a first evolved point *is* its
        # one-sided closure (`lopsided_weights` starts at `−k_down = 0`), so a
        # zone point whose advected axes are all such faces does not move:
        # 282 of the 2192 here. The rest do — the zone kernel blends too.
        @test zmoved == ex.nzone - 282
        # The two providers' lopsided derivative at a centered point, at full
        # weight, and at zero weight the argument itself.
        I, b = first((I, b) for b in 1:nblocks(U), I in owned
                     if cls(ex, I, b) == TGHx.CLASS_CENTERED &&
                        sqrt(sum(abs2, coordinates(U, b, Tuple(I) .+ G))) < 1)
        var, cb = lin(I, b)
        ih = inv(spacing(T, forest, forest.leaves[b]))
        Lp = TGHx.Lopsided(T, Val(q), st, ih, one(T))
        Cp = TGHx.closure_provider(T, Val(G), st, ex.classes, cb, ex.table, ih, one(T))
        agree = all(isequal(TGHx.adv(Lp, β, zero(T), U.work, var + (v - 1) * sv, d),
                            TGHx.adv(Cp, β, zero(T), U.work, var + (v - 1) * sv, d))
                    for v in (1, 7, 13, 20), d in 1:3, β in (-one(T), one(T)))
        @test agree
        L0 = TGHx.Lopsided(T, Val(q), st, ih, zero(T))
        @test isequal(TGHx.adv(L0, one(T), T(NaN), U.work, var, 1), T(NaN))
    end

    @testset "the tracked geometry is frozen, and the found horizon stays m h outside it" begin
        # Guards the driver's tracked `:excised` path: the geometry built once
        # from the seed (no rebuild per chunk, none on a find), the finder run
        # every chunk for the record, and the assertion's row — the found
        # horizon's least distance from the frozen surface, `m` cells for the
        # seed. `m = 16` puts the surface at `3/4`, inside the fixture's fine
        # cube with its `(G + q + 2) h`; `n_L = 4` is the core rule's depth.
        tracked = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10),
                                   interior=FittedSpec(T; variant=:excised,
                                                       margin=16, n_L=4),
                                   horizon=Horizon(T; every=1, N=12, spin=false))
        out = evolve!(T, tracked; forest=deepcopy(forest), q=q, ops=ops,
                      t_end=T(1 // 10))
        @test out.geometry isa FittedInterior{T,:excised}
        @test out.problem.excision.interior === out.geometry
        @test all(r -> r.finite && r.horizon_success === true, out.records)
        @test all(r -> 15.9 < r.excision_horizon_margin < 16.1, out.records)
        @test all(r -> r.excision_into == 0 && r.excision_normal_min > 1, out.records)
        @test abs(out.problem.excision.nzone - ex.nzone) ≤ ex.nzone ÷ 50
    end

    @testset "the monitors read no excised value, and the horizon guard refuses one" begin
        # Guards `monitor_mask` and the masks the pointwise monitors keep:
        # with `NaN` in every excised owned point the gauge and ADM
        # constraints (which take stencils, masked to the excised set widened
        # by `W`), the error, the speed, the validity monitor, the outflow
        # monitor and the non-finite count (which read points, masked by the
        # interior's own mask, the band included) must all be finite — and
        # TreeAMR's interpolation, through the horizon finder's guard, must
        # refuse a footprint that reaches an excised point.
        u3 = copy(u)
        S3 = statearray(u3, U)
        for b in 1:nblocks(U), I in owned
            cls(ex, I, b) == TGHx.CLASS_EXCISED || continue
            for v in 1:20
                S3[I, v, b] = T(NaN)
            end
        end
        gh_constraint!(p, u3, zero(T))
        c = constraint_norms(p)
        @test all(isfinite, c.gauge_l2) && all(isfinite, c.gauge_linf)
        adm_constraint!(p, u3, zero(T))
        a = constraint_norms(p)
        @test isfinite(a.ham_l2) && isfinite(a.ham_linf) && all(isfinite, a.mom_linf)
        gh_error!(p, u3, zero(T); shell=horizon_shell(case, p.interior))
        e = error_norms(p)
        @test isfinite(e.err_l2) && isfinite(e.err_linf) && e.residual == 0
        @test evolved_nonfinite(p, u3, zero(T)) == 0
        @test isfinite(TGHx.max_speed_of(p, u3, zero(T)))
        val = validity_rows(p, u3, zero(T))
        @test all(x -> x === nothing || isfinite(x), values(val))
        r = excision_rows(p, u3, zero(T))
        @test r.excision_band == ex.nzone && r.excision_band_nonfinite == 0
        @test r.excision_normal_min > 0 && r.excision_into == 0
        # The guard: a query half a cell outside the surface reads 1.5 cells
        # inside it (`q + 2` points), one well outside reads nothing excised.
        h = minimum_spacing(T, forest)
        TGHx.scatter!(U, u3)
        fill_ghosts!(U, p.schedule; boundary=dirichlet(case, zero(T)))
        @test_throws ArgumentError gh_interpolate(U, [SVector(r_E + h / 2, h / 3, h / 5)];
                                                  q=q, mask=mask)
        @test all(isfinite, first(gh_interpolate(U, [SVector(r_E + 3h, h / 3, h / 5)];
                                                 q=q, mask=mask)))
    end

    @testset "an interior without excision parameters prints as it did" begin
        # Guards every checkpoint written before step X2b: the recipe holds
        # `repr(case.interior)`, and the `excision` field the variant added to
        # `Interior` and `FittedSpec` would otherwise change it for every
        # layer variant and refuse their restarts. The strings are the base
        # branch's (8fa6658), printed in the same session; an `:excised`
        # interior prints its parameters, which the recipe then carries.
        @test repr(Interior(T; center=(0, 0, 0), r_0=0.4, r_1=1.15)) ==
              "Interior{Float64, :damped, Nothing}(HoleCenter{Float64}([0.0, 0.0, 0.0], " *
              "[0.0, 0.0, 0.0]), 0.4, 1.15, 0.0, 0.5, 0.5, 8, Val{:damped}(), nothing)"
        @test repr(Interior(Float32; center=(0, 0, 0), r_0=0.4, r_1=1.15)) ==
              "Interior{Float32, :damped, Nothing}(HoleCenter{Float32}(Float32[0.0, " *
              "0.0, 0.0], Float32[0.0, 0.0, 0.0]), 0.4f0, 1.15f0, 0.0f0, 0.5f0, " *
              "0.5f0, 8, Val{:damped}(), nothing)"
        @test repr(FittedSpec(T; variant=:fitted,
                              target_bounds=default_bounds(T; M=1, r_gate=0.9))) ==
              "FittedSpec{Float64, :fitted, Nothing, StateBounds{Float64}}(8, 0, 2, 4, " *
              "8, 0.0, 0.5, 1.0, 3, 0.1, Val{:fitted}(), nothing, " *
              "StateBounds{Float64}(0.02, 50.0, 0.01, 1000.0, 10.0, 100.0, 0.9), " *
              "true, 1)"
        @test occursin("Excision{Float64, :msn}(1.0, 4.0, Val{:msn}())",
                       repr(Interior(T; center=(0, 0, 0), r_0=0.4, r_1=0.75,
                                     variant=:excised,
                                     excision=Excision(T; upwind=(1, 4)))))
        @test_throws ArgumentError Interior(T; center=(0, 0, 0), r_0=0.4, r_1=1.15,
                                            excision=Excision(T))
    end

    @testset "the refusals, each by name" begin
        # Guards the checks the design rests on, each where it is asserted:
        # the surface on one level, the singular set inside the core rule's
        # sphere, the margin, `ε_KO > 0`, `m ≥ ⌈√3 G⌉` with a horizon finder,
        # the shift into the excised set (a spinning hole), the range
        # projection, and the driver's keywords and the Π post-pass.
        msg(f) = try
            f()
            ""
        catch err
            sprint(showerror, err)
        end
        mk(c) = () -> setup(c)
        @test occursin("one level",
                       msg(mk(hole_fixture(T; q=q, variant=:excised, r_1=T(11 // 10)))))
        @test occursin("singular set",
                       msg(mk(hole_fixture(T; q=q, a=T(3 // 5), variant=:excised,
                                           r_0=T(1 // 2), r_1=r_E))))
        @test occursin("not far enough inside the horizon",
                       msg(mk(hole_fixture(T; q=q, variant=:excised, r_1=T(3 // 2)))))
        @test occursin("needs Kreiss–Oliger dissipation",
                       msg(mk(hole_fixture(T; q=q, variant=:excised, r_1=r_E,
                                           ε_KO=zero(T)))))
        @test occursin("⌈√3 G⌉",
                       msg(mk(hole_fixture(T; q=q, variant=:excised, r_1=r_E, margin=3,
                                           horizon=Horizon(T; every=1, N=12)))))
        @test occursin("shift points into the excised set",
                       msg(mk(hole_fixture(T; q=q, a=T(3 // 5), variant=:excised,
                                           r_0=T(31 // 50), r_1=T(19 // 25)))))
        @test occursin("no range projection",
                       msg(() -> hole_fixture(T; q=q, variant=:excised, r_1=r_E,
                                              bounds=default_bounds(T; M=1,
                                                                    r_gate=T(1 // 2)))))
        ev(; kw...) = () -> evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                                    t_end=T(1 // 10), kw...)
        @test occursin("regrid = true is refused", msg(ev(; regrid=true)))
        @test occursin("adapt = true is refused", msg(ev(; adapt=true)))
        @test occursin("no relaxation rate", msg(ev(; ρ_max_fixed=T(4))))
        @test occursin("refused for an :excised hole",
                       msg(() -> discrete_gradient_momentum!(U, case, zero(T), q,
                                                             p.schedule)))
    end

    @testset "a run to M/5 is finite with a positive normal margin, and restarts as the run" begin
        # Guards the driver's `:excised` path end to end — the frozen
        # geometry, the problem rebuilt per chunk carrying its classes, the
        # outflow rows — and its checkpoint: an `:excised` interior's
        # parameters are in its `repr`, which the recipe holds, and a restart
        # rebuilds the classes from the file's state. Two chunks of `M/10`;
        # the chain of two one-chunk jobs must be the run bit for bit.
        run = evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                      t_end=T(1 // 5))
        rows = run.records
        @test length(rows) == 3
        @test all(r -> r.finite && r.variant === :excised, rows)
        @test all(r -> r.excision_band == ex.nzone && r.excision_band_nonfinite == 0,
                  rows)
        @test all(r -> r.excision_normal_min > 1 && r.excision_into == 0, rows)
        @test all(r -> r.residual == 0, rows)
        # Outside the step-5 fixture's own layer, `r ≥ 23/20`, the error is
        # the `:damped` fixture's: `4.390e−3` against `4.395e−3` at `M/5`
        # (measured in step X2b, `CODE.md`, "Excision"); a closure that
        # leaked more than the layer would show here first.
        far = ShellMask(SVector{3,T}(0, 0, 0), T(23 // 20), T(100))
        gh_error!(run.problem, run.u, T(1 // 5); mask=far)
        @test error_norms(run.problem).err_l2 < 4.5e-3
        @info "the excised fixture to M/5" err_l2 = [r.err_l2 for r in rows] gauge_l2 =
            [r.gauge_l2 for r in rows] normal = [r.excision_normal_min for r in rows]
        pre = joinpath(mktempdir(), "run")
        common = (q=q, ops=ops, t_end=T(1 // 5), checkpoint_path_prefix=pre,
                  max_walltime_seconds=1e-9, checkpoint_sync_to_disk=false)
        first_ = evolve!(T, case; forest=deepcopy(forest), common...)
        @test !first_.finished
        second = evolve!(T, case; common..., restart_file=latest_checkpoint(pre))
        @test second.finished
        @test isequal(second.u, run.u)
        @test isequal(second.records, run.records)
    end
end
