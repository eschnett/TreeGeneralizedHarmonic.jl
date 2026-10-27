# The controlled source line (step B3; Erik's decision of 2026-10-26 on `CODE.md`'s
# item 7, "What feeds the composite source"): a slow control law for the
# track-fed gauge source, tested on one moving hole fed by its own track first.
# Each hole's source rides a line `c̈ = a_N + ω² (c_tr − c) + 2ω (ċ_tr − ċ)`, a
# critically damped loop of time `τ = 1/ω` onto its track, with the Newtonian
# pull of the other holes as its model (`CODE.md`, "The controlled source line").
#
# Host-side, the loop's arithmetic: its response on analytic inputs against the
# filter's closed form, on synthetic finds through a real track, the model's free
# fall, and the source's `∂_t H` against the time derivative of the source the line
# carries. Then one moving hole on the quadrant fixture of
# `tracked_target_tests.jl` — `q = 2`, `h = 5/48`, the kernels that file and
# `holes_tests.jl` already compile — fed on its seed's line and on a line never
# corrected (bit for bit the analytic run), and on a controlled line through a
# chain of restarts. The `20 M` screens on one H200 are `test/octant_runs.jl
# feed=…`'s (`CODE.md`, "The controlled source line").

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using StaticArrays: SVector
import AbstractSphericalHarmonics as ASH

@testset verbose = true "The controlled source line (step B3)" begin
    T = Float64
    X(x) = SVector{3,T}(x, 0, 0)
    # A line carried to `t1` in pieces of `Δ`, as the driver carries it.
    function carry(lines, input, masses, τs, t1; Δ=T(1 // 20), newton=length(lines) > 1)
        t = lines[1].t
        out = Tuple{T,Vector{SourceLine{T}}}[]
        while t < t1 - sqrt(eps(T))
            t = min(t + Δ, T(t1))
            lines = advance_source_lines(lines, input, masses, τs, t; newton=newton)
            push!(out, (t, lines))
        end
        return lines, out
    end

    # Guards the law and its integration: a sign in the feedback, a factor in
    # the damping, or an integrator that is not the ODE shows here as a response
    # that is not the critically damped filter's. On an analytic input `x(t)`
    # the line's error `e = c − x` obeys `ë + 2ω ė + ω² e = −ẍ`: a velocity seeded
    # `δv` off decays as `δv t e^{−ωt}`, a constant acceleration is followed with
    # the lag `a τ²` and no velocity error, and a sinusoid of frequency `Ω`
    # passes with the gain `|T(iΩ)| = √(ω⁴ + 4ω²Ω²)/(Ω² + ω²)`.
    @testset "the loop is the critically damped filter on an analytic input" begin
        τ = T(10)
        ω = 1 / τ
        v, δv = T(1 // 5), T(1 // 20)
        uniform = (A, s) -> (X(v * s), X(v))
        l0 = [SourceLine{T}(zero(T), X(0), X(v + δv))]
        _, path = carry(l0, uniform, T[1], T[τ], 6τ; newton=false)
        worst = maximum(path) do (t, ls)
            abs(ls[1].c[1] - v * t - δv * t * exp(-ω * t)) / (δv * τ)
        end
        @test worst < 1e-9
        a = T(1 // 128)
        parabola = (A, s) -> (X(a * s^2 / 2), X(a * s))
        l1, _ = carry([SourceLine{T}(zero(T), X(0), X(0))], parabola, T[1], T[τ], 25τ;
                      newton=false)
        t1 = l1[1].t
        @test (a * t1^2 / 2 - l1[1].c[1]) / (a * τ^2) ≈ 1 rtol = 1e-6
        @test abs(a * t1 - l1[1].v[1]) < 1e-6 * a * τ
        # The junk's period, measured on the head-on pair (`7 M_A`), at its
        # amplitude there.
        Ω, A = T(2π / 7), T(1 // 50)
        wave = (B, s) -> (X(A * sin(Ω * s)), X(A * Ω * cos(Ω * s)))
        _, path = carry([SourceLine{T}(zero(T), X(0), X(A * Ω))], wave, T[1], T[τ],
                        20τ + 7 * 2π / Ω; newton=false)
        # The amplitude over the last seven periods, by least squares in
        # `sin Ωt` and `cos Ωt` (exact for the steady response).
        tail = filter(p -> p[1] > 20τ, path)
        Sm = [sin(Ω * p[1]) for p in tail]
        Cm = [cos(Ω * p[1]) for p in tail]
        y = [p[2][1].c[1] for p in tail]
        M2 = [sum(Sm .* Sm) sum(Sm .* Cm); sum(Sm .* Cm) sum(Cm .* Cm)]
        ab = M2 \ [sum(Sm .* y), sum(Cm .* y)]
        gain = sqrt(ω^4 + 4 * ω^2 * Ω^2) / (Ω^2 + ω^2)
        @test sqrt(sum(abs2, ab)) / A ≈ gain rtol = 1e-6
        @info "the loop's gain at the junk's period, τ = 10 (step B3)" gain measured = sqrt(sum(abs2, ab)) / A
    end

    # Guards the loop's input: the line reads the track's position and velocity
    # through `track_state`, which the finds move — a line wired to the find's
    # origin rather than the track, or to the track's acceleration, would pass
    # the junk at the finds' cadence. A hole moving at `v` with the junk's
    # oscillation on top, found every `M/4` as the pair's holes are: the line
    # carries `v` and a fraction of the junk's velocity near the filter's gain.
    @testset "on synthetic finds the line follows the track and filters the junk" begin
        G, h = 3, T(1 // 24)
        L = 4
        probe = seeded_track(T; t=0, center=(0, 0, 0), velocity=(0, 0, 0), r_min=1 // 2)
        hlm = complex_from_real(probe.shape, L)
        grid = ASH.EquiangularGrid(L)
        find_at(x) = (success=true, origin=Tuple(x), origin_r_min=1 / 2,
                      origin_r_max=1 / 2, hlm=hlm, grid=grid)
        v, Ω, A = T(1 // 5), T(2π / 7), T(1 // 50)
        x(t) = X(v * t + A * sin(Ω * t))
        ẋ(t) = X(v + A * Ω * cos(Ω * t))
        τ = T(10)
        tr = seeded_track(T; t=0, center=x(0), velocity=ẋ(0), r_min=1 // 2)
        tr = update_track(tr, find_at(x(0)), 0; G=G, h=h, max_interval=1 // 4)
        lines = seed_source_lines([tr], 0)
        t = zero(T)
        vdev = T[]
        cdev = T[]
        while t < 30τ
            for _ in 1:5
                lines = advance_source_lines(lines, track_input([tr]), T[1], T[τ],
                                             lines[1].t + T(1 // 20); newton=false)
                if lines[1].t > 20τ
                    push!(vdev, lines[1].v[1] - v)
                    push!(cdev, lines[1].c[1] - v * lines[1].t)
                end
            end
            t = lines[1].t
            tr = update_track(tr, find_at(x(t)), t; G=G, h=h, max_interval=1 // 4)
        end
        gain = sqrt(1 / τ^4 + 4 * Ω^2 / τ^2) / (Ω^2 + 1 / τ^2)
        # The velocity's swing, against the junk's `A Ω` through the filter.
        swing = (maximum(vdev) - minimum(vdev)) / 2
        @test swing < 1.2 * gain * A * Ω
        @test swing > 0.8 * gain * A * Ω
        # No drift: the line's mean is the hole's uniform motion.
        @test abs(sum(cdev) / length(cdev)) < A / 20
        @test abs(sum(vdev) / length(vdev)) < A * Ω / 20
        @info "the line on synthetic finds with the junk, τ = 10 (step B3)" swing junk = A * Ω gain
    end

    # Guards the model: two lines that are never corrected (`τ = Inf`) fall
    # together as Newtonian point masses from rest, which is what the seeds'
    # quadratic is the first term of; a sign or a factor in the pull, or a
    # model applied to one hole, is a separation that is not free fall's.
    @testset "the model alone is Newtonian free fall" begin
        still = (A, s) -> (X(A == 1 ? -4 : 4), X(0))
        l0 = [SourceLine{T}(zero(T), X(-4), X(0)), SourceLine{T}(zero(T), X(4), X(0))]
        l1, _ = carry(l0, still, T[1 // 2, 1 // 2], T[Inf, Inf], 10)
        y = (T(8), zero(T))
        n = 100_000
        dt = T(10) / n
        f((d, w)) = (w, -1 / d^2)
        for _ in 1:n
            k1 = f(y)
            k2 = f(y .+ (dt / 2) .* k1)
            k3 = f(y .+ (dt / 2) .* k2)
            k4 = f(y .+ dt .* k3)
            y = y .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        end
        @test l1[2].c[1] - l1[1].c[1] ≈ y[1] rtol = 1e-10
        @test l1[2].v[1] - l1[1].v[1] ≈ y[2] rtol = 1e-8
        @test l1[1].c[1] == -l1[2].c[1]
        # And a line whose track is the free fall stays on it: the loop pulls
        # toward nothing.
        fall = (A, s) -> (s ≈ 10 ? l1[A].c : X(NaN), X(NaN))
        @test_throws "carried back" advance_source_lines(l1, fall, T[1 // 2, 1 // 2],
                                                         T[10, 10], 9)
    end

    # Guards `∂_t H` with the control on: the piece's source is a
    # `TrackedHole` about `t_p` with the line's center, velocity and
    # acceleration, and its time row must be the derivative of the source the
    # line carries — the source of the piece about `t_p ± δ` at `t_p ± δ` — or
    # the system is the generalized harmonic system of no `H`, and `C_a` shows
    # it (`src/gauge.jl`). Measured on a pair whose tracks are a cell off the
    # lines, so that the loop and the model both act.
    @testset "the source's ∂_t H is the time derivative of the source the line carries" begin
        unit = TrackedSpec(T; r_0=T(3 // 5), r_1=T(7 // 5), margin=4, control_time=4)
        holes = head_on_holes(T; d=8, masses=(1, 1), interior=unit)
        pair = binary_case(T; holes=holes, halfwidth=T(8), chunk=T(1 // 10),
                           horizon=Horizon(T; every=1, N=12, spin=false),
                           source_feed=:controlled)
        @test pair.interior.feed == TreeGeneralizedHarmonic.FEED_CONTROLLED
        @test_throws "control_time" binary_case(T; holes=head_on_holes(T; d=8,
                                                                       masses=(1, 1),
                                                                       interior=TrackedSpec(T; r_0=T(3 // 5), r_1=T(7 // 5), margin=4)),
                                                halfwidth=T(8), chunk=T(1 // 10),
                                                horizon=Horizon(T; every=1, N=12,
                                                                spin=false),
                                                source_feed=:controlled)
        tracks = seed_tracks(pair, 0)
        G, h = 2, T(1 // 16)
        L = 4
        probe = seeded_track(T; t=0, center=(0, 0, 0), velocity=(0, 0, 0), r_min=2)
        hz(x) = (success=true, origin=Tuple(x), origin_r_min=2, origin_r_max=2,
                 hlm=complex_from_real(probe.shape, L), grid=ASH.EquiangularGrid(L))
        tracks = [update_track(tracks[A], hz(tracks[A].c_find + X(A == 1 ? h : -h)), 0;
                               G=G, h=h) for A in 1:2]
        lines = seed_source_lines(tracks, 0)
        tp, δ = T(1 // 2), T(1 // 1000)
        inp = track_input(tracks)
        masses = T[1, 1]
        τs = T[4, 4]
        lm = advance_source_lines(lines, inp, masses, τs, tp - δ)
        l0 = advance_source_lines(lm, inp, masses, τs, tp)
        lp = advance_source_lines(l0, inp, masses, τs, tp + δ)
        lh = advance_source_lines(l0, inp, masses, τs, tp + δ / 2)
        Sm, S0, Sp, Sh = (lines_source_at(pair, l, tracks, tt)
                          for (l, tt) in ((lm, tp - δ), (l0, tp), (lp, tp + δ),
                                          (lh, tp + δ / 2)))
        @test any(h -> !iszero(h.acc), S0.holes)
        for x in (SVector{3,T}(-3, 1 // 2, 0), SVector{3,T}(-1, 1, 1 // 2),
                  SVector{3,T}(0, 2, 0))
            g = state_tuple(pair.background, zero(T), x)
            hm = SVector{10,T}(g[1:10]...)
            z4 = zero(SVector{10,T})
            ∂h = ntuple(_ -> z4, 3)
            H(S, t) = algebraic_gauge_source(S, x, t, hm, z4, ∂h)
            Hm, _ = H(Sm, tp - δ)
            Hp, _ = H(Sp, tp + δ)
            H0, dH0 = H(S0, tp)
            rate = (Hp - Hm) / (2δ)
            row = SVector{4,T}(dH0[1, 1], dH0[1, 2], dH0[1, 3], dH0[1, 4])
            @test maximum(abs, rate - row) < 1e-5 * maximum(abs, H0) + 1e-10
            # And the piece about `t_p` meets the next piece's source at `t_p + δ`
            # to the line's jerk, `O(δ²)`: the line is `C²` and each piece its
            # Taylor quadratic.
            jump(δ′, S′) = maximum(abs, first(H(S0, tp + δ′)) - first(H(S′, tp + δ′)))
            @test jump(δ, Sp) < 1e-6 * maximum(abs, H0)
            @test 3 < jump(δ, Sp) / jump(δ / 2, Sh) < 5
        end
    end

    # The moving hole of `tracked_target_tests.jl` with the one-hole composite
    # source.
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    kw = (; halfwidth=T(5 // 2), chunk=T(1 // 10), x0=-1 // 4, v=1 // 5,
          gauge_source=:composite, octant=:quadrant,
          horizon=Horizon(T; every=1, N=12, spin=false))
    radii = (; r_0=T(2 // 5), r_1=T(11 // 10))
    fixture(feed; τ=zero(T), seed_dv=0) =
        kerr_schild_case(T; kw...,
                         interior=TrackedSpec(T; radii..., max_interval=T(1 // 20),
                                              source_feed=feed, control_time=τ,
                                              seed_velocity_offset=(seed_dv, 0, 0)))
    forest = gh_forest(T, fixture(:own); N=12, roots=(4, 2, 2))
    run(c; kw2...) = evolve!(T, c; forest=deepcopy(forest), q=q, ops=ops,
                             t_end=T(2 // 10), kw2...)
    common(r) = Base.structdiff(r, NamedTuple{(:source_center, :source_velocity,
                                                :source_acceleration)})

    # Guards the plumbing of a fed source: rebuilding the one-hole composite
    # every piece from the seed's line, or from a controlled line that is never
    # corrected, is the case's own source — one hole's source does not read its
    # center, and a line without acceleration is `KerrSchildSource`'s arithmetic —
    # so any difference is the driver's: a piece handed the wrong case, a source
    # rebuilt at the wrong time, the lines advanced out of order.
    @testset "one hole fed on its seed's line, or on a line never corrected, is the analytic run bit for bit" begin
        r_own = run(fixture(:own))
        r_seed = run(fixture(:seed))
        r_inf = run(fixture(:controlled; τ=T(Inf)))
        @test isequal(r_own.u, r_seed.u)
        @test isequal(r_own.u, r_inf.u)
        @test isequal(r_own.records, map(common, r_seed.records))
        @test isequal(r_own.records, map(common, r_inf.records))
        @test all(r -> r.source_velocity == (1 / 5, 0.0, 0.0), r_inf.records)
        @test all(r -> r.source_acceleration == (0.0, 0.0, 0.0), r_inf.records)
        @test !haskey(r_own.records[end], :source_velocity)
        @test_throws "gauge_source = :composite" kerr_schild_case(T; kw...,
                gauge_source=:algebraic,
                interior=TrackedSpec(T; radii..., source_feed=:seed)) |>
            c -> evolve!(T, c; forest=deepcopy(forest), q=q, ops=ops, t_end=T(1 // 10))
    end

    # Guards the run state and the line's start: the controlled lines are carried
    # by the checkpoint, so a chain of restarts — one chunk a job — is the
    # uninterrupted run bit for bit, the source rows included. And the line starts
    # on the case's own source — the data's gauge — and not on the track's seed:
    # with step B2's recovery seed, `0.05` too fast, the track recovers on its
    # finds as it does on the analytic source (`tracked_target_tests.jl`), and the
    # line, pulled toward it while it does, stays on the hole. A line started on
    # the seed instead carried the hole with it — measured 2026-10-26 on this
    # fixture at `τ = 1`: the source at `0.255`, the track at `0.252` after
    # `0.3 M` and `0.18` cells off the analytic line, `C_a` `10⁴×` at `t = 0`:
    # the hole's coordinate velocity is the gauge source's.
    @testset "a controlled line restarts bit for bit, and starts on the data's gauge" begin
        c = fixture(:controlled; τ=T(1), seed_dv=1 // 20)
        a = run(c)
        @test all(r -> r.horizon_success === true && r.finite, a.records)
        vs = [r.source_velocity[1] for r in a.records]
        vt = [r.track_velocity[1] for r in a.records]
        @test vs[1] == 1 / 5 && vt[1] ≈ 1 / 4
        @test all(v -> abs(v - 1 / 5) < 1 / 100, vs)
        # Recovering on its finds: measured `0.211` at `0.2 M`, from `0.25`.
        @test abs(vt[end] - 1 / 5) < abs(vt[1] - 1 / 5) / 3
        @test a.records[end].track_offset < 1 // 10
        pre = joinpath(mktempdir(), "line")
        b = run(c; checkpoint_path_prefix=pre, max_walltime_seconds=1e-9,
                checkpoint_sync_to_disk=false)
        jobs = 1
        while !b.finished
            b = evolve!(T, c; q=q, ops=ops, t_end=T(2 // 10), checkpoint_path_prefix=pre,
                        max_walltime_seconds=1e-9, checkpoint_sync_to_disk=false,
                        restart_file=latest_checkpoint(pre))
            jobs += 1
        end
        @test jobs == 2
        @test isequal(a.u, b.u)
        @test isequal(a.records, b.records)
        @info "a controlled line against a track seeded 0.05 too fast, τ = 1 (step B3)" source_velocity = vs track_velocity = vt track_offset = [r.track_offset for r in a.records]
    end
end
