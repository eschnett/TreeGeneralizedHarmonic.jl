# Several holes in one case (step B3): `CODE.md`, "Binary black holes" — "A case
# without a background", "One interior per hole", "Refinement for two holes",
# "The record" and "Initial data" (item 1). A case takes `holes`, a tuple of
# `HoleSpec`s; one hole is the case it was, and two are a binary: the data a
# superposition, each hole's interior step B2's tracked target, the composite
# gauge source of the holes fed by their tracks (pairing (α), decided by Erik
# 2026-10-26), one track and one find per hole, and the record's rows per hole.
#
# Priced on what the suite compiles already where it can: the single-hole claims
# rerun step B2's own runs (its kernels), and the pair is the cheapest that holds
# a tracked layer at `q = 2` — two holes of mass one at `d = 8`, `h = 1/8` at the
# holes, `m = 4`, 288 blocks. The head-on pair of mass one in total, at `1/48`,
# to `2 M`, is `test/binary_runs.jl`'s, on an H200.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using LinearAlgebra: dot, norm
using StaticArrays: SMatrix, SVector
import SpacetimeMetrics as SM

const TGHh = TreeGeneralizedHarmonic

@testset verbose = true "Several holes in one case (step B3)" begin
    T = Float64
    q = 2
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    unit = TrackedSpec(T; r_0=T(3 // 5), r_1=T(7 // 5), margin=4)

    # Guards the pieces the binary is made of, host-side: a hole that would put
    # the target where the source is not (a tilted spin, a :fitted interior) is
    # refused; the pair is where its center of mass says, falling toward itself;
    # the superposition is the sum it says it is; and the companion attenuation
    # is what makes the data a 3+1 metric inside the horizons (found in step B3:
    # the plain sum has α² < 0 on the side facing the companion).
    @testset "a pair's holes, its data and its profiles" begin
        @test_throws "along z" HoleSpec(T; mass=1, spin=(0.1, 0, 0), center=(0, 0, 0),
                                        interior=unit)
        @test_throws "TrackedSpec" HoleSpec(T; mass=1, center=(0, 0, 0),
                                            interior=FittedSpec(T))
        hs = head_on_holes(T; d=8, interior=unit)
        @test hs[1].center == SVector(-4, 0, 0) && hs[2].center == SVector(4, 0, 0)
        @test hs[1].acceleration ≈ SVector(1 / 128, 0, 0)
        @test hs[1].interior.r_1 == T(7 // 10)            # scaled by M_A = 1/2
        @test center_at(holes_center(hs), zero(T)) == SVector(0, 0, 0)

        plain = superposed_kerr_schild(T, hs)
        x = SVector{4,T}(0, 1 // 3, 1 // 2, -1 // 5)
        η = SM.metric(SM.Minkowski(), x)
        sum_ = η + (SM.metric(hole_metric(hs[1]), x) - η) +
               (SM.metric(hole_metric(hs[2]), x) - η)
        @test isapprox(SM.metric(plain, x), sum_; rtol=4eps(T))
        α²(g) = (γ = g[2:4, 2:4]; βl = g[1, 2:4]; -g[1, 1] + dot(βl, γ \ βl))
        attn = superposed_kerr_schild(T, hs; companion=4)
        # On hole 1's core sphere, facing its companion: Kerr-Schild's
        # `α² = r/(r + 2M)`, and the plain sum's below zero.
        y = SVector{4,T}(0, -4 + 5 // 16, 0, 0)
        @test α²(SM.metric(plain, y)) < 0
        @test α²(SM.metric(attn, y)) > α²(SM.metric(hole_metric(hs[1]), y)) - 1 // 100

        case = binary_case(T; holes=hs, halfwidth=16, chunk=T(1 // 4),
                           horizon=Horizon(T; every=1, N=12, spin=false))
        @test case.interior isa HolesSpec{2} && case.interior.superposed
        @test case.background isa SuperposedMetric
        @test_throws "several holes" GHProblem(FieldSet{T}(gh_forest(T, case; N=8,
                                                                      roots=4), 20;
                                                           G=2,
                                                           centering=vertexcentered(3)),
                                               nothing, case; q=q)
        γ0 = case.γ0
        @test damping_rate(γ0, zero(T), (-4.0, 0.0, 0.0)) ≈
              1 / 10 + (2 - 1 / 10) + (1 - 1 / 10) * exp(-16 / 128) rtol = 1e-3
        # A source whose holes are not the case's at t = 0 — here without the
        # seeds' accelerations — is refused by name.
        @test_throws "not the case's hole 1" check_holes_source(
            hs, CompositeKerrSchildSource(TrackedHole(T; M=1 // 2, center=(-4, 0, 0)),
                                          TrackedHole(T; M=1 // 2, center=(4, 0, 0))), T)
        @test check_holes_source(hs, holes_source(T, hs), T) === nothing
    end

    # Guards the composite's protocol at a point: the hole deepest in its own
    # layer acts — so a point in hole 2's layer sees hole 2's interior and a
    # point outside both sees an interior that leaves it alone — and the
    # composite mask evolves a point only outside every layer.
    @testset "the composite selects the hole that acts, and masks every layer" begin
        hs = head_on_holes(T; d=8, interior=unit)
        case = binary_case(T; holes=hs, halfwidth=16, chunk=T(1 // 4),
                           horizon=Horizon(T; every=1, N=12, spin=false))
        c = holes_interior(case, seed_tracks(case, 0), zero(T))
        @test c isa HoleInteriors{2}
        @test acting_interior(c, 0.0, (4.2, 0.0, 0.0)).center == c.holes[2].center
        @test acting_interior(c, 0.0, (-4.2, 0.1, 0.0)).center == c.holes[1].center
        @test outside_all(c, 0.0, (0.0, 0.0, 0.0))
        @test !outside_all(c, 0.0, (4.5, 0.0, 0.0))
        m = interior_mask(c, 0.0)
        @test !is_evolved(m, (4.5, 0.0, 0.0)) && !is_evolved(m, (-3.6, 0.0, 0.0))
        @test is_evolved(m, (0.0, 0.0, 0.0)) && is_evolved(m, (4.0, 0.0, 0.9))
        @test is_evolved(layer_mask(c, 0.0), (4.5, 0.0, 0.0))
        @test !is_evolved(layer_mask(c, 0.0), (0.0, 0.0, 0.0))
        # The finder's guard: a footprint hits the union of the layers exactly
        # when TreeAMR's enumeration of its points says so (the region's own
        # method takes each hole's exact one).
        reg = UnevolvedRegion(m)
        agree = true
        for base in ((-23, -3, -3), (-27, -2, 1), (9, -1, -4), (17, -3, -3), (-2, 0, 0),
                     (12, 3, -2), (23, 2, 2)), h in (1 // 8, 1 // 16)
            o, off = (0.0, 0.0, 0.0), (0, 0, 0)
            hh = T(h) * 2
            fast = TreeAMR.stencil_hits(reg, o, hh, base, off, Val(4))
            slow = any(J -> TreeAMR.inside(reg, ntuple(dd -> TreeAMR.stencil_position(
                    o, hh, base, off, dd, J[dd] - 1), 3)), CartesianIndices((4, 4, 4)))
            agree &= fast == slow
        end
        @test agree
        # The level floors: one per hole, and a block's the larger of the two.
        rc = with_refinement(case, Refinement(T; refine_tol=T(2 // 5),
                                              coarsen_tol=T(1 // 10), maxlevel_cap=8,
                                              floor_margin=zero(T)))
        fo = gh_forest(T, case; N=8, roots=4)
        lb = level_bounds(rc, fo, zero(T), 2; interior=c)
        @test lb isa TGHh.UnionLevelBounds{T,2}
        floors = map(k -> first(block_level_bounds(lb, k)), fo.leaves)
        near(k, x) = all(d -> block_extent(T, fo, k)[d][1] ≤ x[d] ≤ block_extent(T, fo, k)[d][2], 1:3)
        @test all(floors[i] > 0 for i in eachindex(fo.leaves) if near(fo.leaves[i], (-4, 0, 0)))
        @test all(floors[i] > 0 for i in eachindex(fo.leaves) if near(fo.leaves[i], (4, 0, 0)))
        @test any(==(0), floors)
        # ε_KO per hole: each hole's profile near it, the exterior's elsewhere.
        εs = HolesDissipation{T,2}(ntuple(A -> HorizonDissipation(T; ε_out=1 // 2, ε_in=2,
                                                                 r_1=hs[A].interior.r_1,
                                                                 r_h=1, center=initial_line(hs[A])),
                                         2))
        @test dissipation_rate(εs, 0.0, (4.5, 0.0, 0.0)) == 2
        @test dissipation_rate(εs, 0.0, (-3.5, 0.0, 0.0)) == 2
        @test dissipation_rate(εs, 0.0, (0.0, 0.0, 0.0)) == 1 // 2
        @test binary_case(T; holes=hs, halfwidth=16, chunk=T(1 // 4), ε_KO=εs,
                          horizon=Horizon(T; every=1, N=12)).ε_KO === εs
        # The data at t = 0 are the targets' superposition there, bit for bit:
        # the layer starts at its target.
        x = (-4.5, 0.1, 0.2)
        @test isequal(background_state(case.background, 0.0, x),
                      background_state(c.holes[1].target, 0.0, x))
        # The gauge-consistent momentum keeps the metric and changes `Π`.
        src = holes_source(T, hs)
        h, Π = gauge_consistent_state(src, case.background, 0.0, x)
        g = SM.metric(case.background, SVector{4,T}(0, x...))
        @test isequal(h, pack_g(g))
        @test norm(Π - background_state(case.background, 0.0, x)[2]) > 0
    end

    # The suite's pair: two holes of mass one at `d = 8`, `h = 1/8` at the
    # holes — the step-5 fixture's resolution of a unit hole at `q = 2` — with a
    # margin of 4, 288 blocks.
    pair_holes = head_on_holes(T; d=8, masses=(1, 1), interior=unit)
    pair = binary_case(T; holes=pair_holes, halfwidth=8, chunk=T(1 // 10),
                       horizon=Horizon(T; every=1, N=12, spin=false))
    pair_forest = holes_forest(T, pair; N=8, roots=2, radii=(T(4), T(3), T(8 // 5)))

    # Guards the one claim the composition is held to point by point: wherever
    # one hole's layer acts, the composite right-hand side is that hole's
    # single-hole kernel, and outside every layer it is `F` — compared against
    # the single-hole kernel of the nearer hole, the same case, the same state,
    # on every owned point of the two-hole mesh, on SIMD lanes and without.
    # Measured 2026-10-26: bit for bit.
    @testset "the composite right-hand side is the nearer hole's single-hole kernel" begin
        U = FieldSet{T}(pair_forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
        geom = with_ρ_max(holes_interior(pair, seed_tracks(pair, 0), zero(T)), T(4))
        fill_exact!(U, pair, zero(T); interior=geom)
        u = statevector(U)
        gather!(u, U)
        sched = GhostSchedule(U, ops)
        for W in (nothing, 1)
            du = [similar(u) for _ in 1:3]
            for (k, int) in enumerate((geom, geom.holes[1], geom.holes[2]))
                p = GHProblem(U, sched, pair; q=q, interior=int, margin_check=false,
                              simd_width=W)
                gh_rhs!(du[k], u, p, zero(T))
            end
            DU = [statearray(d, U) for d in du]
            nfar = nlayer = ndiff = 0
            same = true
            for b in 1:nblocks(U), I in CartesianIndices(interiorview(U, b, 1))
                x = coordinates(U, b, Tuple(I) .+ U.G)
                A = norm(SVector(x) - pair_holes[1].center) ≤
                    norm(SVector(x) - pair_holes[2].center) ? 2 : 3
                same &= isequal(DU[1][I, :, b], DU[A][I, :, b])
                inl = !outside_all(geom, zero(T), x)
                nlayer += inl
                nfar += !inl
                ndiff += inl && !isequal(DU[1][I, :, b], DU[5 - A][I, :, b])
            end
            @test same
            @test nlayer > 0 && nfar > 0
            @test ndiff > 0     # the other hole's kernel does not hold this layer
        end
        # And `momentum = :gauge` starts the pair with `C_a` at truncation: the
        # superposition's own `Π` leaves the overlap's `H + Γ` (measured
        # 2026-10-26 on this mesh, outside the layers: see the @info).
        pg = binary_case(T; holes=pair_holes, halfwidth=8, chunk=T(1 // 10),
                         horizon=Horizon(T; every=1, N=12, spin=false), momentum=:gauge)
        cs = map((pair, pg)) do cs_case
            fill_exact!(U, cs_case, zero(T); interior=geom)
            gather!(u, U)
            p = GHProblem(U, sched, cs_case; q=q, interior=geom, margin_check=false)
            gh_constraint!(p, u, zero(T))
            maximum(constraint_norms(p).gauge_l2)
        end
        @info "the pair's C_a at t = 0 (step B3)" superposition = cs[1] gauge = cs[2]
        @test cs[2] < cs[1] / 4
    end

    # Guards the decision that the single hole is a binary with one hole (Erik,
    # 2026-10-26): `holes = (spec,)` built from the same background is the case
    # built with `interior = TrackedSpec(…)`, field for field, and `evolve!` runs
    # it bit for bit — the state after two chunks and every record row — on the
    # suite's static fixture with its sampled source and on step B2's moving
    # quadrant hole with the algebraic one.
    @testset "one hole is the case it was, bit for bit" begin
        function as_holes(c)
            spec = HoleSpec(T; mass=1, spin=(0, 0, 0), center=c.center.c0,
                            velocity=c.center.v, interior=c.interior)
            return GHCase(T, c.background; box=c.box, periodic=c.periodic,
                          reflecting=c.reflecting, ε_KO=c.ε_KO, γ0=c.γ0, γ2=c.γ2,
                          horizon=c.horizon, chunk=c.chunk, gauge_source=c.gauge,
                          holes=(spec,))
        end
        static = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10),
                                  interior=TrackedSpec(T; r_0=T(2 // 5), r_1=T(23 // 20)),
                                  horizon=Horizon(T; every=1, N=12, spin=false))
        quad = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10), x0=-1 // 4,
                                v=1 // 5, gauge_source=:algebraic, octant=:quadrant,
                                interior=TrackedSpec(T; r_0=T(2 // 5), r_1=T(11 // 10),
                                                     max_interval=T(1 // 20)),
                                horizon=Horizon(T; every=1, N=12, spin=false))
        forests = (hole_forest(T, static; N=8, roots=1, radii=(T(3), T(3), one(T))),
                   gh_forest(T, quad; N=12, roots=(4, 2, 2)))
        for (c, f) in zip((static, quad), forests)
            ch = as_holes(c)
            @test ch.holes isa Tuple{HoleSpec} && ch.interior === c.interior
            @test ch.center === c.center && ch.background === c.background
            @test ch.γ0 === c.γ0 && ch.gauge === c.gauge
            a = evolve!(T, c; forest=deepcopy(f), q=q, ops=ops, t_end=T(2 // 10))
            b = evolve!(T, ch; forest=deepcopy(f), q=q, ops=ops, t_end=T(2 // 10))
            @test isequal(a.u, b.u)
            @test isequal(a.records, b.records)
            @test length(b.records) == 3 && all(r -> r.horizon_success === true, b.records)
        end
        @test_throws "background's" GHCase(T, static.background; box=static.box,
                                            periodic=static.periodic, ε_KO=static.ε_KO,
                                            γ0=static.γ0, γ2=static.γ2,
                                            holes=(HoleSpec(T; mass=2, center=(0, 0, 0),
                                                            interior=static.interior),))
    end

    # Guards the loop of several holes and its run state: both holes found at
    # every row from their own tracks, the record's rows per hole and for the
    # pair, and a chain of restarts — one chunk a job — that is the
    # uninterrupted run bit for bit, the two tracks, their horizon offsets and
    # the finder's seeds included.
    @testset "a pair evolves, finds both holes and restarts bit for bit" begin
        pre = joinpath(mktempdir(), "pair")
        common = (q=q, ops=ops, t_end=T(2 // 10), checkpoint_sync_to_disk=false)
        a = evolve!(T, pair; forest=deepcopy(pair_forest), common...)
        @test length(a.records) == 3
        @test all(r -> r.horizon_success === true && r.finite, a.records)
        @test all(r -> all(h -> h.success === true, r.holes), a.records)
        @test all(r -> r.err_l2 === nothing && r.drift === nothing, a.records)
        @test a.track isa Vector{HorizonTrack{T}} && length(a.track) == 2
        @test a.interior isa HoleInteriors{2}
        r0, r2 = a.records[1], a.records[end]
        @test r0.separation == 8 && r0.residual == 0
        # The pair falls together; over 0.2 M free fall is 1.6·10⁻³ of a cell,
        # and the tracks follow the horizons' centroids, which the data's junk
        # moves more than that (recorded, not asserted beyond a cell).
        @test abs(r2.separation - 8) < 1 // 8
        @test r2.holes[1].track_center[1] < 0 < r2.holes[2].track_center[1]
        @test all(r -> maximum(h -> h.track_prediction_max, r.holes) < 1, a.records)
        @test all(h -> abs(h.horizon_offset[1]) < 1 // 8, r0.holes)
        @info "the suite's pair (step B3)" separation = [r.separation for r in a.records] gauge_l2 = [r.gauge_l2 for r in a.records] residual = [r.residual for r in a.records] areas = [map(h -> h.area, r.holes) for r in a.records] offsets = [h.horizon_offset[1] for h in r0.holes]

        b = evolve!(T, pair; forest=deepcopy(pair_forest), common...,
                    checkpoint_path_prefix=pre, max_walltime_seconds=1e-9)
        jobs = 1
        while !b.finished
            b = evolve!(T, pair; common..., checkpoint_path_prefix=pre,
                        max_walltime_seconds=1e-9, restart_file=latest_checkpoint(pre))
            jobs += 1
        end
        @test jobs == 2
        @test isequal(a.u, b.u)
        @test isequal(a.records, b.records)
        @test isequal(repr(a.track), repr(b.track))
        @test a.nsteps == b.nsteps
    end
end
