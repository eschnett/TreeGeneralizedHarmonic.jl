# Two harmonic holes with no source (step H4): `PLAN.md`'s step H4 and `CODE.md`'s
# "Harmonic gauge for the inspiral". A hole takes a chart; holes in the harmonic
# chart are superposed in it and run in harmonic gauge, `H ≡ 0`, with `Π` from
# `C_a = 0` in the data and in every layer target, or with the damped harmonic
# gauge rolled on from zero at a `t_0` that is a time or an event of the tracks'
# separation.
#
# Priced: the host-side claims cost nothing; one harmonic hole of mass one needs
# `h = 1/16` at `q = 2` (its horizon is at `r = M`, half Kerr-Schild's), which is
# 288 blocks on `[−2, 2]³`; the pair at `d = 8` is 1296 blocks, its two chunks the
# suite's largest harmonic run (about four minutes at four threads, most of it the
# set-up). The production rows, `1/96` and `1/128` at the holes, are
# `test/binary_runs.jl chart=harmonic`'s, on H200s.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using LinearAlgebra: norm
using StaticArrays: SMatrix, SVector
import SpacetimeMetrics as SM

const TGHH = TreeGeneralizedHarmonic

isdefined(@__MODULE__, :TEST_TIER) || include("tiers.jl")  # `@full_tier`

# `C_a = H_a + Γ_a` at `H = 0` from a state's `h`, `Π` and the analytic `∂_i h`:
# `∂_t h` from the first evolution equation, `Γ_a = g^{bc} ∂_b g_ac − ½ g^{bc}
# ∂_a g_bc`.
function pointwise_gauge_constraint(h, Π, ∂h)
    _, gu, α, β, _, sqrtγ = TGHH.metric_quantities(TGHH._sym4(h))
    ∂ₜh = β[1] * ∂h[1] + β[2] * ∂h[2] + β[3] * ∂h[3] + (α / sqrtγ) * Π
    D = (TGHH._sym4(∂ₜh), TGHH._sym4(∂h[1]), TGHH._sym4(∂h[2]), TGHH._sym4(∂h[3]))
    Γ = SVector{4}(sum(gu[b, c] * D[b][a, c] for b in 1:4, c in 1:4) -
                   sum(gu[b, c] * D[a][b, c] for b in 1:4, c in 1:4) / 2 for a in 1:4)
    return maximum(abs, Γ)
end

@testset verbose = true "Two harmonic holes with no source (step H4)" begin
    T = Float64
    q = 2
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    # The harmonic chart's layer for a hole of mass one at `q = 2`, `h = 1/16`:
    # `r_1` four cells inside the horizon at `r = 1`, the ramp six cells thick.
    unit = TrackedSpec(T; r_0=T(7 // 20), r_1=T(37 // 50), margin=4)

    # Guards the pieces the harmonic pair is made of, host-side: a hole's chart,
    # one chart per case, the data and the layer targets in harmonic gauge with
    # `C_a = 0` pointwise at `t = 0` (the superposition's own `Π` violates it
    # where the fields overlap), and no source of the holes.
    @testset "a harmonic pair's holes, data and gauge" begin
        @test_throws "chart" HoleSpec(T; mass=1, center=(0, 0, 0), interior=unit,
                                      chart=:isotropic)
        hs = head_on_holes(T; d=8, masses=(1, 1), interior=unit, chart=:harmonic)
        @test map(hole_chart, hs) == (:harmonic, :harmonic)
        @test isharmonic(hole_metric(hs[1]))
        ks = head_on_holes(T; d=8, masses=(1, 1), interior=unit)
        @test_throws "one chart" superposed_holes(T, (hs[1], ks[2]))
        @test_throws "Kerr-Schild holes" superposed_kerr_schild(T, hs)
        @test_throws "no source of their own" holes_source(T, hs)

        pair = binary_case(T; holes=hs, halfwidth=8, chunk=T(1 // 20),
                           horizon=Horizon(T; every=1, N=12, spin=false))
        @test pair.gauge === nothing
        @test isharmonic(pair.background) && TGHH.has_gauge_momentum(pair.background)
        @test pair.interior.gauge_momentum
        own = binary_case(T; holes=hs, halfwidth=8, chunk=T(1 // 20),
                          momentum=:superposition,
                          horizon=Horizon(T; every=1, N=12, spin=false))
        @test !TGHH.has_gauge_momentum(own.background)
        # `C_a` at `t = 0` at roundoff with the gauge's `Π`, and of the order of
        # the companion's field without it.
        worst_gauge = worst_own = 0.0
        for x in ((0.0, 0.3, 0.1), (-2.0, 1.0, 0.5), (3.0, -0.4, 0.2), (-5.5, 0.0, 1.0),
                  (-4.0, 1.5, -0.5))
            worst_gauge = max(worst_gauge,
                              pointwise_gauge_constraint(background_state(pair.background,
                                                                          0.0, x)...))
            worst_own = max(worst_own,
                            pointwise_gauge_constraint(background_state(own.background,
                                                                        0.0, x)...))
        end
        @test worst_gauge < 1e-14
        @test worst_own > 1e-4
        # The data are the gauge-consistent state's, and each layer's target at
        # `t = 0` is the data, bit for bit: the layer starts at its target.
        c = holes_interior(pair, seed_tracks(pair, 0), zero(T))
        x = (-4.5, 0.1, 0.2)
        h, Π = gauge_consistent_state(zero(SVector{4,T}), pair.background, 0.0, x)
        @test isequal((h, Π), background_state(pair.background, 0.0, x)[1:2])
        @test isequal(background_state(pair.background, 0.0, x),
                      background_state(c.holes[1].target, 0.0, x))
        @test c.own[1] isa SM.TranslatedMetric && isharmonic(c.own[1])
    end

    # Guards the roll-on from zero, host-side: up to `t_0` — `Inf` included, the
    # start an event leaves open — a launch is handed no source, and a launch
    # handed none takes the harmonic kind's zeros; after it the blend, then the
    # damped harmonic gauge alone. A harmonic case takes no source or such a
    # roll-on and refuses any other.
    @testset "the roll-on from zero is no source until t_0" begin
        dh = DampedHarmonicSource(T; σ_r=32)
        for t_0 in (zero(T), T(2), T(Inf))
            r = RolledGaugeSource(ZeroGaugeSource(T), dh; σ_g=1, t_0=t_0)
            @test effective_source(r, T(0)) === nothing
            isfinite(t_0) || continue
            @test effective_source(r, t_0 + 1) === r
            @test effective_source(r, t_0 + 10) === dh
        end
        r = RolledGaugeSource(ZeroGaugeSource(T), dh; σ_g=1, t_0=T(Inf))
        r3 = with_rollon_time(r, 3)
        @test typeof(r3) === typeof(r) && r3.t_0 == 3 && r3.to === dh
        @test with_rollon_time(dh, 3) === dh
        z = (zero(SVector{4,T}), zero(SMatrix{4,4,T}))
        @test isequal(TGHH.gauge_source(T, nothing, (1, 1, 1), 1, Val(:algebraic),
                                        (0.0, 0.0, 0.0), 0.0, nothing, nothing, nothing), z)
        @test isequal(TGHH.gauge_at(T, nothing, (1, 1, 1), 1, Val(false)), z)
        # The check of a single hole's source: harmonic Kerr is exact under the
        # roll-on before it starts, Kerr-Schild is not.
        @test check_gauge_source(SM.Harmonic(one(T)), r, T; center=(0, 0, 0), scale=1) < 1e-12
        @test_throws "does not make the background exact" check_gauge_source(
            SM.KerrSchild(one(T)), r, T; center=(0, 0, 0), scale=1)
        hs = head_on_holes(T; d=8, masses=(1, 1), interior=unit, chart=:harmonic)
        @test check_holes_source(hs, nothing, T) === nothing
        @test check_holes_source(hs, r, T) === nothing
        @test_throws "harmonic gauge" check_holes_source(hs, dh, T)
        @test_throws "harmonic gauge" binary_case(T; holes=hs, halfwidth=8,
                                                   chunk=T(1 // 20), gauge_source=dh,
                                                   horizon=Horizon(T; every=1, N=12))
        single = harmonic_kerr_case(T; halfwidth=T(2), chunk=T(1 // 20), interior=:damped,
                                    r_0=T(7 // 20), r_1=T(37 // 50), margin=4,
                                    gauge_source=r)
        @test_throws "rollon_separation" evolve!(T, single; q=q, ops=ops, t_end=T(1 // 20),
                                                  forest=hole_forest(T, single; N=8,
                                                                     roots=1,
                                                                     radii=(one(T),)),
                                                  rollon_separation=4)
    end

    # Guards the decision that one hole is the case it was, in the harmonic chart
    # too: `holes = (spec,)` with `chart = :harmonic` is the tracked harmonic case,
    # bit for bit, the state and every record row; and the roll-on from zero is
    # the run without a source until its `t_0`, bit for bit (the kernel is handed
    # no source and takes the harmonic kind's zeros), while one that starts at
    # `t = 0` is another run.
    @full_tier @testset "one harmonic hole, and the roll-on from zero, bit for bit" begin
        tracked = harmonic_kerr_case(T; halfwidth=T(2), chunk=T(1 // 20), interior=unit,
                                     horizon=Horizon(T; every=1, N=12, spin=false))
        spec = HoleSpec(T; mass=1, center=tracked.center.c0, velocity=tracked.center.v,
                        interior=unit, chart=:harmonic)
        as_hole = GHCase(T, tracked.background; box=tracked.box, periodic=tracked.periodic,
                         reflecting=tracked.reflecting, ε_KO=tracked.ε_KO, γ0=tracked.γ0,
                         γ2=tracked.γ2, horizon=tracked.horizon, chunk=tracked.chunk,
                         gauge_source=tracked.gauge, holes=(spec,))
        @test_throws "background's" GHCase(T, tracked.background; box=tracked.box,
                                            periodic=tracked.periodic, ε_KO=tracked.ε_KO,
                                            γ0=tracked.γ0, γ2=tracked.γ2,
                                            holes=(HoleSpec(T; mass=1, center=(0, 0, 0),
                                                            interior=unit),))
        f = hole_forest(T, tracked; N=8, roots=1, radii=(T(3 // 2), T(6 // 5), one(T)))
        @test nleaves(f) == 288 && minimum_spacing(T, f) == 1 // 16
        run(c) = evolve!(T, c; forest=deepcopy(f), q=q, ops=ops, t_end=T(1 // 10))
        a = run(tracked)
        b = run(as_hole)
        @test isequal(a.u, b.u)
        @test isequal(a.records, b.records)
        @test all(r -> r.horizon_success === true, b.records)

        sphere(src) = harmonic_kerr_case(T; halfwidth=T(2), chunk=T(1 // 20),
                                         interior=:damped, r_0=T(7 // 20),
                                         r_1=T(37 // 50), margin=4, gauge_source=src)
        dh = DampedHarmonicSource(T; σ_r=32)
        none = run(sphere(nothing))
        later = run(sphere(RolledGaugeSource(ZeroGaugeSource(T), dh; σ_g=1, t_0=Inf)))
        started = run(sphere(RolledGaugeSource(ZeroGaugeSource(T), dh; σ_g=1, t_0=0)))
        @test isequal(none.u, later.u)
        @test isequal(none.records, later.records)
        @test !isequal(none.u, started.u)
    end

    # Guards the harmonic pair's loop and its run state: both holes found at every
    # row with no source, the roll-on started by the tracks' separation at the
    # first row that has fallen to `d_g` — not at `t = 0`, where the tracks are
    # the seeds, exactly `d` apart — and recorded from that row on; and a chain of
    # restarts from the run's `t = 0` checkpoint, one chunk a job, that is the
    # uninterrupted run bit for bit, the roll-on's start included (`PLAN.md`:
    # `t_0` is run state).
    @full_tier @testset "a harmonic pair rolls on at its separation and restarts bit for bit" begin
        hs = head_on_holes(T; d=8, masses=(1, 1), interior=unit, chart=:harmonic)
        pair = binary_case(T; holes=hs, halfwidth=8, chunk=T(1 // 20),
                           gauge_source=RolledGaugeSource(ZeroGaugeSource(T),
                                                          DampedHarmonicSource(T; σ_r=32);
                                                          σ_g=1, t_0=Inf),
                           horizon=Horizon(T; every=1, N=12, spin=false))
        f = holes_forest(T, pair; N=8, roots=2, radii=(T(4), T(3), T(2), T(6 // 5)))
        @test minimum_spacing(T, f) == 1 // 16
        common = (q=q, ops=ops, t_end=T(1 // 10), checkpoint_sync_to_disk=false,
                  rollon_separation=8 - 1e-9)
        a = evolve!(T, pair; forest=deepcopy(f), common...)
        @test length(a.records) == 3
        @test all(r -> r.finite && all(h -> h.success === true, r.holes), a.records)
        @test a.records[1].separation == 8 && a.records[1].residual == 0
        @test [r.rollon_t0 for r in a.records] == [Inf, 1 / 20, 1 / 20]
        @test a.records[1].gauge_l2 < 1e-3
        @info "the suite's harmonic pair (step H4)" separation = [r.separation for r in a.records] gauge_l2 = [r.gauge_l2 for r in a.records] residual = [r.residual for r in a.records] areas = [map(h -> h.area, r.holes) for r in a.records]

        # The chain starts from the run's `t = 0` checkpoint (step S1's start,
        # chunk 0), which is how a pair from solved data is run: what the start
        # saves of the pair — each hole's track and offset, and the roll-on's
        # `t_0`, `Inf` until it starts — must come back as the run.
        pre = joinpath(mktempdir(), "hpair")
        b = evolve!(T, pair; forest=deepcopy(f), common..., t_end=zero(T),
                    checkpoint_path_prefix=pre, checkpoint_every_chunks=1)
        @test b.finished && b.nchunks == 0 && length(b.checkpoints_written) == 1
        jobs = 0
        b = (; finished=false)
        while !b.finished
            b = evolve!(T, pair; common..., checkpoint_path_prefix=pre,
                        max_walltime_seconds=1e-9, restart_file=latest_checkpoint(pre))
            jobs += 1
        end
        @test jobs == 2
        @test isequal(a.u, b.u)
        @test isequal(a.records, b.records)
        @test a.nsteps == b.nsteps
    end

    # Guards a fitted hole's rule, host-side: a hole takes a `:fitted` spec and
    # refuses a `:damped` one (that is `TrackedSpec`'s), scaled by its mass as a
    # tracked target is; and its composite is built on the mesh, which a
    # composite of tracked targets does not need.
    @testset "a fitted hole's rule" begin
        fs = FittedSpec(T; variant=:fitted, margin=4, fit_cont=2, ρ_max=8)
        @test_throws "variant :damped" HoleSpec(T; mass=1, center=(0, 0, 0),
                                                 interior=FittedSpec(T))
        hs = head_on_holes(T; d=8, masses=(1 // 2, 1 // 2), interior=fs, chart=:harmonic)
        @test hs[1].interior.ρ_max == 16 && hs[1].interior.margin == 4
        pair = binary_case(T; holes=hs, halfwidth=8, chunk=T(1 // 20),
                           horizon=Horizon(T; every=1, N=12, spin=false))
        @test pair.interior isa HolesSpec{2,<:FittedSpec}
        @test_throws "forest, G and n_L" holes_interior(pair, seed_tracks(pair, 0), 0.0)
        @test all(tr -> tr.quadratic, seed_tracks(pair, 0))
    end

    # Guards the fitted pair's composition and run state (step H4): the cache of
    # two fitted holes is each hole's own cache — the single hole's fill, bit for
    # bit — inside that hole's fill sphere and zero elsewhere; the pair evolves
    # with both holes found and both fits valid at every row; and a chain of
    # restarts from the run's `t = 0` checkpoint, one chunk a job, is the
    # uninterrupted run bit for bit, each hole's fits, ranges and initial fit
    # included.
    @full_tier @testset "a fitted harmonic pair: its cache, its rows and its restarts" begin
        fs = FittedSpec(T; variant=:fitted, margin=4, fit_cont=2)
        hs = head_on_holes(T; d=8, masses=(1, 1), interior=fs, chart=:harmonic)
        pair = binary_case(T; holes=hs, halfwidth=8, chunk=T(1 // 20),
                           horizon=Horizon(T; every=1, N=12, spin=false))
        f = holes_forest(T, pair; N=8, roots=2, radii=(T(4), T(3), T(2), T(6 // 5)))
        common = (q=q, ops=ops, t_end=T(1 // 10), checkpoint_sync_to_disk=false)
        a = evolve!(T, pair; forest=deepcopy(f), common...)
        @test length(a.records) == 3
        @test all(r -> r.finite && all(h -> h.success === true && h.fit_valid, r.holes),
                  a.records)
        @test a.interior isa HoleInteriors{2} && interior_variant(a.interior) === :fitted
        @test a.fits isa Tuple && length(a.fits) == 2 && all(f -> f[2] !== nothing, a.fits)
        @test all(r -> all(h -> h.layer_offset == 4 // 16, r.holes), a.records)
        @info "the suite's fitted harmonic pair (step H4)" separation = [r.separation for r in a.records] gauge_l2 = [r.gauge_l2 for r in a.records] fit_residual = [map(h -> h.fit_residual, r.holes) for r in a.records] areas = [map(h -> h.area, r.holes) for r in a.records] fit_cost = a.fit_cost

        # The composite cache against each hole's own, on the run's last mesh.
        U = FieldSet{T}(a.forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
        origins = block_origins(U.forest, T)
        spacings = block_spacings(U.forest, T)
        tf = T(1 // 10)
        both = TGHH.fill_target!(TGHH.target_cache(U), origins, spacings, a.interior, a.fits, tf)
        same = true
        nin = 0
        for A in 1:2
            own = TGHH.fill_target!(TGHH.target_cache(U), origins, spacings,
                               a.interior.holes[A], a.fits[A], tf)
            g = a.interior.holes[A]
            r_fill = (g.r_out - g.offset) + g.h
            for b in 1:nblocks(U), I in CartesianIndices(interiorview(U, b, 1))
                x = coordinates(U, b, Tuple(I) .+ U.G)
                TGHH.interior_radius(g, tf, x) < r_fill || continue
                nin += 1
                same &= isequal(own.work[I, :, b], both.work[I, :, b])
            end
        end
        @test same && nin > 0

        # The chain starts from the run's `t = 0` checkpoint (step S1's start,
        # chunk 0), which is how a pair from solved data is run: what the start
        # saves of the pair — each hole's track, offset, fits and initial fit —
        # must come back as the run.
        pre = joinpath(mktempdir(), "fpair")
        b = evolve!(T, pair; forest=deepcopy(f), common..., t_end=zero(T),
                    checkpoint_path_prefix=pre, checkpoint_every_chunks=1)
        @test b.finished && b.nchunks == 0 && length(b.checkpoints_written) == 1
        jobs = 0
        b = (; finished=false)
        while !b.finished
            b = evolve!(T, pair; common..., checkpoint_path_prefix=pre,
                        max_walltime_seconds=1e-9, restart_file=latest_checkpoint(pre))
            jobs += 1
        end
        @test jobs == 2
        @test isequal(a.u, b.u)
        @test isequal(a.records, b.records)
    end
end
