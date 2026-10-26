# The tracked analytic target (step B2): `CODE.md`, "One interior per hole" —
# "Tracked `:damped`" — and "The tracks". A `TrackedSpec` case runs step 5's
# `:damped` sphere about the horizon track, relaxing toward the boosted
# Kerr-Schild hole on the track's line, rebuilt every piece of a chunk; the
# track predicts along the quadratic of its last four finds, and finds as often
# as the cadence asks, at the pieces' boundaries. The host-side claims — the
# prediction on synthetic orbits, the blend, the target a track gives — are
# `tracking_tests.jl`'s; the chain of restarts is `checkpoint_tests.jl`'s; the
# `40 M` runs, the sensitivity to the target's parameters and the recovery are
# `test/octant_runs.jl interior=tracked`'s (`SINGULARITY_HANDLING.md`, "The tracked
# analytic target").
#
# Priced on what the suite already compiles: the moving quadrant hole of
# `boosted_tests.jl`'s steady-state testset — `q = 2`, `h = 5/48`, the `:damped`
# `Val(:algebraic)` kernel — with the finder of `horizon_tests.jl`; the one new
# kernel is the `:damped` sphere carrying a target metric.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using StaticArrays: SVector

@testset verbose = true "The tracked analytic target (step B2)" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    v = 1 // 5
    kw = (; halfwidth=T(5 // 2), chunk=T(1 // 10), x0=-1 // 4, v=v,
          gauge_source=:algebraic, octant=:quadrant,
          horizon=Horizon(T; every=1, N=12, spin=false))
    radii = (; r_0=T(2 // 5), r_1=T(11 // 10))
    ana = kerr_schild_case(T; kw..., radii...)
    # Two finds a chunk, so that one of them is at a piece's boundary.
    spec = TrackedSpec(T; radii..., max_interval=T(1 // 20))
    trk = kerr_schild_case(T; kw..., interior=spec)
    forest = gh_forest(T, ana; N=12, roots=(4, 2, 2))

    # Guards the claim the sensitivity study is measured against: with the
    # track at the analytic answer the tracked target *is* the analytic layer
    # — its sphere, its target and its core — so that whatever the runs below
    # and on Symmetry measure is the track's and the target's parameters and
    # not a second spelling. Measured 2026-10-26: bit for bit.
    @testset "the target built from the analytic seed is the analytic layer, bit for bit" begin
        int0 = tracked_interior(spec, seed_track(trk, 0), zero(T); mass=1, spin=0,
                                ρ_max=T(4))
        U = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                        parity=state_parity(forest))
        fill_exact!(U, ana, zero(T))
        u = statevector(U)
        gather!(u, U)
        sched = GhostSchedule(U, ops)
        du_a, du_t = similar(u), similar(u)
        gh_rhs!(du_a, u, GHProblem(U, sched, ana; q=q,
                                   interior=with_ρ_max(ana.interior, T(4))), zero(T))
        gh_rhs!(du_t, u, GHProblem(U, sched, trk; q=q, interior=int0), zero(T))
        @test isequal(du_a, du_t)
        @test_throws "TrackedSpec" GHProblem(U, sched, trk; q=q)
    end

    # Guards the loop: the target rebuilt every piece on the track's line (a
    # target left at the chunk's start lags the hole by `v · chunk`, a target
    # on the wrong line drags the layer), the finds at the pieces' boundaries
    # (two a chunk here, the row's the second), and the row the track writes.
    # A tracked target is the analytic layer to the tracking: measured
    # 2026-10-26 at `t = 0.3 M`, the masked error `5.534e−3` against `5.547e−3`,
    # the residual `3.04` against `2.98`, the track `3·10⁻⁴` cells off the
    # analytic center.
    @testset "the tracked target holds the moving hole as the analytic layer does" begin
        ra = evolve!(T, ana; forest=deepcopy(forest), q=q, ops=ops, t_end=T(3 // 10))
        rt = evolve!(T, trk; forest=deepcopy(forest), q=q, ops=ops, t_end=T(3 // 10))
        @test all(r -> r.horizon_success === true, rt.records)
        @test all(r -> r.track_source === :found, rt.records)
        @test [r.track_finds for r in rt.records] == [1, 2, 2, 2]
        @test all(r -> r.track_interval ≈ 1 / 20, rt.records)
        @test all(r -> r.track_offset < 1e-2, rt.records)
        @test all(r -> r.track_prediction_max < 1e-2, rt.records)
        @test all(r -> r.variant === :damped, rt.records)
        @test rt.interior isa Interior && rt.interior.target !== nothing
        @test rt.track.nhist == TRACK_FIT_FINDS
        @test all(r -> r.track_finds === nothing, ra.records)
        for (a, b) in zip(ra.records[2:end], rt.records[2:end])
            @test b.err_l2 ≈ a.err_l2 rtol = 1e-2
            @test b.gauge_l2 ≈ a.gauge_l2 rtol = 1e-2
            @test b.residual ≈ a.residual rtol = 5e-2
        end
        @info "the tracked target against the analytic layer on the q = 2 quadrant (step B2)" err_l2 = [(a.err_l2, b.err_l2) for (a, b) in zip(ra.records, rt.records)] residual = [(a.residual, b.residual) for (a, b) in zip(ra.records, rt.records)] track_offset = [r.track_offset for r in rt.records] prediction = [r.track_prediction_max for r in rt.records]
    end

    # Guards the recovery the track must make on its own: a seed whose velocity
    # is off predicts the first finds off by `δv Δt`, the chord of the second
    # and third finds is the hole's, and the run goes on with no refusal.
    @testset "a track seeded 0.05 too fast recovers within a few finds" begin
        off = kerr_schild_case(T; kw...,
                               interior=TrackedSpec(T; radii..., max_interval=T(1 // 20),
                                                    seed_velocity_offset=(1 // 20, 0, 0)))
        r = evolve!(T, off; forest=deepcopy(forest), q=q, ops=ops, t_end=T(3 // 10))
        @test all(x -> x.horizon_success === true, r.records)
        vx = [x.track_velocity[1] for x in r.records]
        @test vx[1] ≈ 1 // 4                          # the seed's, before a chord
        @test abs(vx[end] - 1 // 5) < 1e-2            # the hole's, after it
        @test r.records[end].track_offset < 0.1
        @info "the recovery of a seed 0.05 too fast (step B2)" track_velocity = vx track_offset = [x.track_offset for x in r.records] prediction = [x.track_prediction_max for x in r.records]
    end
end
