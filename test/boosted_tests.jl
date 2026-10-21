# A boosted Kerr-Schild hole on the quadrant (added 2026-10-20): the pieces a
# non-spinning hole crossing the mesh along `x` at binary speed needs — the
# case (`kerr_schild_case(; v, x0)` with the algebraic source, which moves with
# the hole), the quadrant `[−L, L] × [0, L]²` it is its own mirror image on, the
# `2 × 1 × 1` root brick that domain takes, and the frozen capsule of shells
# swept along its path. The run itself is a study (`test/octant_runs.jl`);
# what is claimed here is that each piece is what it says, and that a few steps
# of the moving hole on the quadrant are the same steps on the whole box.
#
# Priced on what the suite already compiles: the run is the `q = 2`, `:damped`,
# `Val(:algebraic)` kernel `gauge_source_tests.jl` builds, on the reflection
# tests' fixture radii and `h = 5/48`.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using StaticArrays: SVector, SMatrix
using LinearAlgebra: det
import ForwardDiff
import SpacetimeMetrics as SM

const TGHb = TreeGeneralizedHarmonic

@testset verbose = true "A boosted Kerr-Schild hole on the quadrant" begin
    T = Float64
    v, x0 = 1 // 5, -4
    kw = (; halfwidth=T(8), r_0=T(5 // 8), r_1=T(23 // 16), chunk=T(1 // 2),
          gauge_source=:algebraic)
    quad = kerr_schild_case(T; v=v, x0=x0, kw..., octant=:quadrant)

    @testset "the boosted, translated hole is a case, and its source is its −Γ" begin
        # Guards the sign of the boost (`boost(m, u)` moves the hole at `−u`),
        # the order of the translation and the boost (the hole at `x0` at
        # `t = 0`), and the source's velocity — a source built with `+u`, or
        # one that does not move with the hole, makes the exact solution a
        # gauge transient, which only a comparison away from `t = 0` shows:
        # `check_gauge_source` itself looks at `t = 0`.
        full = kerr_schild_case(T; v=v, x0=x0, kw...)
        for case in (full, quad)
            bg = case.background
            @test hole_velocity(bg) == SVector{3,T}(1 // 5, 0, 0)
            @test case.center.v == SVector{3,T}(1 // 5, 0, 0)
            @test center_at(case.center, zero(T)) == SVector{3,T}(-4, 0, 0)
            @test center_at(case.center, T(10)) ≈ SVector{3,T}(-2, 0, 0) atol = 8eps(T)
            γ = 1 / sqrt(1 - T(1 // 25))
            @test case.gauge.M == 1 && case.gauge.u ≈ SVector{4,T}(γ, γ / 5, 0, 0)
            @test iszero(case.gauge.S)
            @test check_gauge_source(bg, case.gauge, T; center=(T(x0), 0, 0),
                                     scale=1) ≤ 1e-13
        end
        # Directly, at `t = 10`, two to five M from where the hole is then:
        # the closed form on the state against SpacetimeMetrics' `−Γ_a`.
        bg = quad.background
        worst = zero(T)
        for x in (SVector{3,T}(0, 2, 1), SVector{3,T}(-6, -3, 2), SVector{3,T}(-2, 1, -4))
            p = SVector{4,T}(10, x...)
            Hs, dHs = SM.gauge_source_grad(bg, p)
            g, dg = SM.dmetric(bg, p)
            h = TGHb.pack_g(g)
            ∂ₜh = TGHb.pack_sym(SMatrix{4,4,T}(dg[a, b, 1] for a in 1:4, b in 1:4))
            ∂h = ntuple(i -> TGHb.pack_sym(SMatrix{4,4,T}(dg[a, b, i + 1]
                                                         for a in 1:4, b in 1:4)), 3)
            Ha, dHa = algebraic_gauge_source(quad.gauge, h, ∂ₜh, ∂h)
            mag = max(maximum(abs, Hs), maximum(abs, dHs))
            worst = max(worst, maximum(abs, Ha - Hs) / mag, maximum(abs, dHa - dHs) / mag)
        end
        @test worst ≤ 1e-13                     # measured 2e−16
        # The metric is singular where the center says the hole is, and regular
        # where the opposite sign would have put it — read beside the singular
        # point, never on it.
        δ = T(1 // 1000)
        gtt(t, x) = SM.metric(bg, SVector{4,T}(t, x...))[1, 1]
        @test abs(gtt(10, center_at(quad.center, T(10)) + SVector{3,T}(0, δ, 0))) > 1 / δ
        @test abs(gtt(10, SVector{3,T}(-6, δ, 0))) < 10
        # With no motion and no translation the case is today's static hole,
        # and a center beside `x0` is refused rather than silently overridden.
        @test kerr_schild_case(T; kw...).background === SM.KerrSchild(one(T), zero(T))
        @test_throws "Leave center unset" kerr_schild_case(T; v=v, kw...,
                                                             center=(T(1), 0, 0))
        # The sampled source still refuses the moving hole, and names the way out.
        @test_throws "gauge_source = KerrSchildSource" kerr_schild_case(
            T; v=v, x0=x0, kw..., gauge_source=nothing)
    end

    @testset "the quadrant is refused for a hole that is not its own mirror image" begin
        # Guards the three ways a hole breaks the `y = 0` and `z = 0` mirrors —
        # a spin, a center off the axis, a velocity across it — each of which
        # would run on the quadrant as a different, wrong problem.
        msg = "its own mirror image in both planes"
        @test quad.box == ((T(-8), T(8)), (zero(T), T(8)), (zero(T), T(8)))
        @test quad.reflecting == ((false, false), (true, false), (true, false))
        @test quad.rotating == (0, 0) && has_outer_face(quad)
        @test_throws msg kerr_schild_case(T; a=T(1 // 2), kw..., octant=:quadrant)
        @test_throws msg kerr_schild_case(T; kw..., center=(T(1), T(1 // 4), 0),
                                          octant=:quadrant)
        @test_throws msg hole_case(T, SM.boost(SM.KerrSchild(one(T), zero(T)),
                                               SVector{3,T}(0, 1 // 5, 0));
                                   kw..., octant=:quadrant)
        # A spin hidden under the boost is read from the bare hole.
        @test_throws msg hole_case(T, SM.boost(SM.KerrSchild(one(T), T(1 // 2)),
                                               SVector{3,T}(-1 // 5, 0, 0));
                                   kw..., octant=:quadrant)
        @test_throws ":quadrant, got :half" kerr_schild_case(T; kw..., octant=:half)
        # A hole at rest anywhere on the axis is one.
        @test kerr_schild_case(T; kw..., center=(T(1), 0, 0),
                               octant=:quadrant).center.v == zero(SVector{3,T})
    end

    @testset "the (2, 1, 1) brick has cubic cells and covers the quadrant" begin
        # Guards the brick `gh_forest` now takes: blocks of two sizes, or a
        # forest that misses part of the box, are both a mesh that is wrong
        # everywhere it is used; and a box whose cells would not be cubes is
        # refused, since TreeAMR's blocks are.
        f = gh_forest(T, quad; N=8, roots=(2, 1, 1))
        @test f.roots == (2, 1, 1) && nleaves(f) == 2
        exts = [block_extent(T, f, k) for k in f.leaves]
        @test all(e -> all(d -> e[d][2] - e[d][1] == 8, 1:3), exts)
        @test sum(e -> prod(d -> e[d][2] - e[d][1], 1:3), exts) == 16 * 8 * 8
        @test all(d -> minimum(e -> e[d][1], exts) == quad.box[d][1] &&
                       maximum(e -> e[d][2], exts) == quad.box[d][2], 1:3)
        @test gh_forest(T, quad; N=8, roots=[4, 2, 2]).roots == (4, 2, 2)
        @test_throws "root blocks must be cubes" gh_forest(T, quad; N=8, roots=2)
        @test_throws "root blocks must be cubes" gh_forest(T, quad; N=8,
                                                           roots=(2, 2, 1))
        @test_throws "brick of three" gh_forest(T, quad; N=8, roots=(2, 1))
        # The cubic box takes one count, as before.
        cube = kerr_schild_case(T; kw..., gauge_source=nothing)
        @test gh_forest(T, cube; N=8, roots=2).roots == (2, 2, 2)
    end

    @testset "the swept shells hold the path, and doubling N keeps the layout" begin
        # Guards the frozen capsule a crossing runs on: a point of the path
        # outside the finest level is a hole that leaves its resolution; a
        # layout that changes with `N` is a convergence study on different
        # meshes; and the two overlap tests against a brute-force sweep of the
        # segment — a diagonal one, where the swept cube is not a box.
        path = (T(8), zero(T), zero(T))
        for shape in (:cube, :ball)
            f = hole_forest(T, quad; N=8, roots=(4, 2, 2), radii=[3, 3 // 2],
                            shape=shape, path=path)
            @test forest_levels(f)[end] > 0 && length(forest_levels(f)) == 3
            for s in range(zero(T), one(T); length=33)
                x = (T(x0) + 8s, zero(T), zero(T))
                owners = filter(f.leaves) do k
                    e = block_extent(T, f, k)
                    all(d -> e[d][1] ≤ x[d] ≤ e[d][2], 1:3)
                end
                @test !isempty(owners) && all(k -> level(k) == 2, owners)
            end
            g = hole_forest(T, quad; N=16, roots=(4, 2, 2), radii=[3, 3 // 2],
                            shape=shape, path=path)
            @test g.leaves == f.leaves
            # A path of length zero is the shell it starts from, and no path is
            # the shell of before, bit for bit.
            c = (T(3 // 2), T(1 // 3), T(1 // 4))
            zero_path = hole_forest(T, quad; N=8, roots=(4, 2, 2), center=c,
                                    radii=[3, 3 // 2], shape=shape, path=(0, 0, 0))
            plain = hole_forest(T, quad; N=8, roots=(4, 2, 2), center=c,
                                radii=[3, 3 // 2], shape=shape)
            @test zero_path.leaves == plain.leaves
        end
        # The overlap tests on random boxes against the segment sampled finely,
        # where the sample decides clearly.
        c, Δc, R = (T(-1), T(1 // 2), T(0)), (T(3), T(2), T(-1)), T(1)
        ss = range(zero(T), one(T); length=2001)
        agree_ball = agree_cube = decided = 0
        for i in 1:250
            lo = ntuple(d -> T(-4) + 8 * T(mod(i * (0.618 + 0.13d), 1)), 3)
            w = ntuple(d -> T(1 // 4) + 2 * T(mod(i * (0.414 + 0.07d), 1)), 3)
            ext = ntuple(d -> (lo[d], lo[d] + w[d]), 3)
            gap(s, d) = max(ext[d][1] - c[d] - s * Δc[d], c[d] + s * Δc[d] - ext[d][2], 0)
            d2 = minimum(s -> sqrt(sum(d -> gap(s, d)^2, 1:3)), ss)
            d∞ = minimum(s -> maximum(d -> gap(s, d), 1:3), ss)
            abs(d2 - R) > 1e-2 && abs(d∞ - R) > 1e-2 || continue
            decided += 1
            agree_ball += TGHb._box_meets_capsule(ext, c, Δc, R) == (d2 ≤ R)
            agree_cube += TGHb._box_meets_swept_cube(ext, c, Δc, R) == (d∞ < R)
        end
        @test decided > 180 && agree_ball == decided && agree_cube == decided
    end

    @testset "the moving sphere's core and mask follow c(t)" begin
        # Guards the `:damped` layer of a moving hole: a core rule or a mask
        # left at `c(0)` freezes evolved points and evolves frozen ones once
        # the hole has moved by a cell.
        int = quad.interior
        for t in (zero(T), T(5))
            c = center_at(quad.center, t)
            @test c ≈ SVector{3,T}(-4 + t / 5, 0, 0)
            # A point inside the core maps onto its sphere about `c(t)`.
            x = c + SVector{3,T}(1 // 4, 1 // 8, 0)
            y = SVector{3,T}(core_position(int, t, x)...)
            @test sqrt(sum(abs2, y - c)) ≈ 5 // 8
            @test (y - c) ≈ (5 // 8) * (x - c) / sqrt(sum(abs2, x - c))
            m = interior_mask(int, t)
            @test !is_evolved(m, c + SVector{3,T}(7 // 5, 0, 0))
            @test is_evolved(m, c + SVector{3,T}(3 // 2, 0, 0))
            @test !is_evolved(m, c - SVector{3,T}(7 // 5, 0, 0))
        end
    end

    @testset "a moving :damped core relaxes toward the analytic solution at its current core position, and a resting one does not move" begin
        # Guards the core of a moving analytic sphere (amended 2026-10-20). A
        # core with `du = 0` keeps the solution about `c(0)` and releases it
        # into the layer on the trailing side — the boosted hole's residual grew
        # linearly and the run died at `r_0/v`; a core relaxing toward the
        # solution at `x` rather than at the core rule's point `c(t) + r_0 n̂`
        # evaluates it at the singularity; a core relaxing linearly in `g`
        # crosses Euclidean metrics where the center's path flips `n̂` — the
        # first version, which ended the run in its first chunk; and a hole at
        # rest whose core moved at all would no longer be the static run bit for
        # bit.
        #
        # First the rate on its own: zero at the target, the ADM variables
        # relaxing at `ρ`, and the path from Kerr-Schild at `+r_0 x̂` to `−r_0 x̂`
        # a metric throughout, where the straight line in `g` is not.
        bgs = SM.KerrSchild(one(T), zero(T))
        r0 = T(5 // 8)
        hp, _, _ = background_state(bgs, zero(T), SVector{3,T}(r0, 0, 0))
        hm, _, _ = background_state(bgs, zero(T), SVector{3,T}(-r0, 0, 0))
        ρ = T(4)
        @test iszero(TGHb.core_relaxation_rate(hm, hm, ρ))
        lorentzian(h) = (g = TGHb._sym4(h) + SMatrix{4,4,T}(-1, 0, 0, 0, 0, 1, 0, 0,
                                                            0, 0, 1, 0, 0, 0, 0, 1);
                         det(g) < 0 && det(g[2:4, 2:4]) > 0 && g[2, 2] > 0 &&
                             det(g[2:3, 2:3]) > 0)
        @test lorentzian(hp) && lorentzian(hm) && !lorentzian((hp + hm) / 2)
        let h = hp, dτ = T(1 // 400), ok = true, A0 = TGHb._adm_split(hp)
            for _ in 1:800                  # to `ρ τ = 8`, RK2
                k1 = TGHb.core_relaxation_rate(h, hm, ρ)
                k2 = TGHb.core_relaxation_rate(h + dτ / 2 * k1, hm, ρ)
                h += dτ * k2
                ok &= lorentzian(h)
            end
            @test ok && maximum(abs, h - hm) ≤ 1e-3 * maximum(abs, hp - hm)
            # Along the rate, `α²` and `β^i` move at `−ρ (· − ·_t)`.
            ε = T(1 // 2^20)
            d = TGHb.core_relaxation_rate(hp, hm, ρ)
            Ap, βp, _, _ = TGHb._adm_split(hp + ε * d)
            Am, βm, _, _ = TGHb._adm_split(hp - ε * d)
            At, βt, _, _ = TGHb._adm_split(hm)
            # (`α²` is the same on both sides of the sphere, so its rate is
            # zero, to the difference's roundoff.)
            @test abs((Ap - Am) / 2ε + ρ * (A0[1] - At)) ≤ 1e-8 * ρ * abs(A0[1])
            @test (βp - βm) / 2ε ≈ -ρ * (A0[2] - βt) rtol = 1e-8
        end

        # Then one right-hand side on the next testset's quadrant at `t = 1` —
        # the hole two cells from where the data were made — on a state
        # perturbed in the core, for the moving sphere and for the same sphere
        # at rest (the same `Interior` type, so the same kernel); then sixteen
        # RK4 steps of the moving one.
        q = 2
        G = q ÷ 2 + 1
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        case = kerr_schild_case(T; halfwidth=T(5 // 2), r_0=T(2 // 5),
                                r_1=T(11 // 10), chunk=T(1 // 10), v=v, x0=-1 // 4,
                                gauge_source=:algebraic, octant=:quadrant)
        f = gh_forest(T, case; N=12, roots=(4, 2, 2))
        U = FieldSet{T}(f, 20; G=G, centering=vertexcentered(3),
                        parity=state_parity(f))
        fill_exact!(U, case, zero(T))
        u0 = statevector(U)
        gather!(u0, U)
        moving = with_ρ_max(case.interior, ρ)
        resting = Interior(T; center=HoleCenter(T, moving.center.c0), r_0=moving.r_0,
                           r_1=moving.r_1, ρ_max=ρ, w_ramp=moving.w_ramp,
                           ρ_ramp=moving.ρ_ramp, margin=moving.margin)
        @test typeof(resting) === typeof(moving)
        @test TGHb.core_follows(moving) && !TGHb.core_follows(resting)
        @test !TGHb.core_follows(Interior(T; center=moving.center, r_0=moving.r_0,
                                          r_1=moving.r_1, variant=:frozen))
        sched = GhostSchedule(U, ops)
        p_mov = GHProblem(U, sched, case; q=q, interior=moving)
        p_rest = GHProblem(U, sched, case; q=q, interior=resting)
        # The core's points at `t`, as (block, stored index, position).
        function core_points(int, t)
            pts = Tuple{Int,NTuple{3,Int},SVector{3,T}}[]
            c = center_at(int.center, t)
            for k in 1:nleaves(f), I in CartesianIndices(interiorview(U, k, 1))
                J = Tuple(I) .+ U.G
                x = SVector{3,T}(coordinates(U, k, J)...)
                sum(abs2, x - c) < int.r_0^2 && push!(pts, (k, J, x))
            end
            return pts
        end
        t = one(T)
        # The state perturbed in both cores, by a pattern that is not the
        # solution anywhere.
        scatter!(U, u0)
        for int in (moving, resting), (k, J, x) in core_points(int, t)
            for var in 1:20
                U.work[J..., var, k] += T(1 // 100) * sin(7 * x[1] + 3 * x[2] + var)
            end
        end
        u = similar(u0)
        gather!(u, U)
        D = FieldSet{T}(f, 20; G=G, centering=vertexcentered(3),
                        parity=state_parity(f))
        du = similar(u)
        gh_rhs!(du, u, p_mov, t)
        scatter!(D, du)
        pts = core_points(moving, t)
        worst, off, spatial, ratescale = zero(T), zero(T), true, zero(T)
        for (k, J, x) in pts
            y = SVector{3,T}(core_position(moving, t, x)...)
            ht, Πt, _ = background_state(case.background, t, y)
            hu = SVector{10,T}(U.work[J..., 1:10, k])
            relax = vcat(TGHb.core_relaxation_rate(hu, ht, ρ),
                         -ρ * (SVector{10,T}(U.work[J..., 11:20, k]) - Πt))
            # The spatial metric's rate is `−ρ (γ − γ_t)` itself.
            spatial &= relax[5:10] == -ρ * (hu[5:10] - ht[5:10])
            # Plus the target's own rate in the core's outer half (amended
            # 2026-10-20): the next testset's claim, here the same spelling.
            ∂ₜt = vcat(TGHb.core_target_rate(moving, case.background, t, x,
                                             TGHb.layer_rate_step(moving))...)
            worst = max(worst, maximum(abs, D.work[J..., :, k] - relax - ∂ₜt))
            off = max(off, maximum(abs, relax))
            ratescale = max(ratescale, maximum(abs, ∂ₜt))
        end
        @info "moving :damped core: $(length(pts)) points, max|du − want| = " *
              "$worst, max|want| = $off, max|∂ₜ target| = $ratescale"
        @test length(pts) > 50 && spatial
        @test off > 1e-2                        # the core is off its target …
        @test worst ≤ 16 * eps(T) * (off + ratescale)   # … and relaxes toward it
        gh_rhs!(du, u, p_rest, t)
        scatter!(D, du)
        rest_pts = core_points(resting, t)
        @test length(rest_pts) > 50
        @test all(((k, J, _),) -> all(iszero, D.work[J..., :, k]), rest_pts)
        # Sixteen steps of the moving one from the exact data: the core is
        # closer to its target at the end than the data it started from — what
        # a frozen core keeps — are.
        dt = T(5 // 48) / 8
        t1 = 16 * dt
        u1 = gh_solve(p_mov, u0, (zero(T), t1); dt=dt)
        scatter!(D, u1)
        scatter!(U, u0)
        now, stale = zero(T), zero(T)
        for (k, J, x) in core_points(moving, t1)
            ut = case_state_tuple(case.background, moving, t1, x)
            for var in 1:20
                now = max(now, abs(D.work[J..., var, k] - ut[var]))
                stale = max(stale, abs(U.work[J..., var, k] - ut[var]))
            end
        end
        @info "moving :damped core after 16 steps: max|u − u_target| = $now, " *
              "the frozen core's $stale"
        @test isfinite(now) && stale > 1e-3 && now ≤ 4 // 5 * stale
    end

    @testset "the moving analytic solution is a steady state of the moving :damped layer" begin
        # Guards the target's rate of a moving analytic sphere (amended
        # 2026-10-20). Without `(1 − w) ∂_t u_exact` in the layer it trails
        # the moving solution by `|∂_t u_exact|/ρ_max`, a residual of `6.5`
        # at `4/M` that no `h` reduces and an error just outside the horizon
        # `28–44×` the resting hole's at `h = 1/16`
        # (`SINGULARITY_HANDLING.md`, "The boosted Kerr-Schild hole at binary
        # speed"); without the core's rate in its outer half the points the
        # core releases on its trailing side carry the core's own lag into the
        # layer. Wrong ways this goes: a rate of the wrong sign or index order
        # (`dmetric`'s derivative axis is last), a difference step that is all
        # roundoff, `w` in place of `1 − w`, a rate added to a hole at rest.
        #
        # The rates against an independent spelling: a boosted, translated
        # stationary hole moves rigidly, `u(t, x) = f(x − v t)`, so `∂_t u =
        # −v · ∇u`, and ForwardDiff takes that through `state_tuple`'s
        # generic position — no difference, no second pass in time. The core
        # rule's target `u(t, c(t) + r_0 n̂)` moves with the center and turns
        # with `n̂`, so its rate is `∇u · (ẏ − v)` at `y`, `ẏ − v = −(r_0/r)(I
        # − n̂ n̂) v`.
        bg = quad.background
        int = quad.interior
        vv = hole_velocity(bg)
        ∇along(t, y, e; bg=bg) = ForwardDiff.derivative(
            s -> SVector{20}(state_tuple(bg, t, y + s * e)), zero(T))
        δ = TGHb.layer_rate_step(int)
        @test 0 < δ < 1e-5 * int.r_0
        t = T(5)
        c = center_at(quad.center, t)
        worsth, worstΠ = zero(T), zero(T)
        for d in (SVector{3,T}(7 // 10, 1 // 5, 1 // 10), SVector{3,T}(-6 // 5, 3 // 10, 0),
                  SVector{3,T}(0, 13 // 10, 2 // 5), SVector{3,T}(-1, 0, 0))
            x = c + d
            h, Π, ∂ₜh, ∂ₜΠ = TGHb.background_state_rate(bg, t, x, δ)
            h0, Π0, _ = background_state(bg, t, x)
            @test h == h0 && Π == Π0
            want = -∇along(t, x, vv)
            worsth = max(worsth, maximum(abs, ∂ₜh - want[1:10]) / maximum(abs, want[1:10]))
            worstΠ = max(worstΠ, maximum(abs, ∂ₜΠ - want[11:20]) / maximum(abs, want[11:20]))
        end
        @test worsth ≤ 1e-13 && worstΠ ≤ 1e-8           # measured 0, 6e−10
        let r0 = int.r_0, worst = zero(T), scale = zero(T)
            for d in (SVector{3,T}(-1 // 2, 1 // 4, 0), SVector{3,T}(1 // 5, 0, 1 // 4),
                      SVector{3,T}(-1 // 4, 1 // 3, 1 // 10))
                x = c + d * (r0 / sqrt(sum(abs2, d))) * T(9 // 10)
                n = d / sqrt(sum(abs2, d))
                r = sqrt(sum(abs2, x - c))
                y = SVector{3,T}(core_position(int, t, x)...)
                want = ∇along(t, y, -(r0 / r) * (vv - n * (n' * vv)))
                @test TGHb.core_rate_weight(int, r) == 1
                ∂ₜh, ∂ₜΠ = TGHb.core_target_rate(int, bg, t, x, δ)
                worst = max(worst, maximum(abs, vcat(∂ₜh, ∂ₜΠ) - want))
                scale = max(scale, maximum(abs, want))
            end
            @test scale > 1e-2 && worst ≤ 1e-8 * scale
            # No rate near the center, and none without a pass.
            @test TGHb.core_rate_weight(int, r0 / 4) == 0
            @test all(iszero, TGHb.core_target_rate(int, bg, t, c + SVector{3,T}(0, 0, r0 / 5),
                                                    δ)[2])
        end

        # Then one right-hand side on the exact data at `t = 1` on the previous
        # testset's quadrant (`q = 2`, `h = 5/48`): in the layer `du = w F +
        # (1 − w) ∂_t u_exact`, `F` from the same mesh with no hole, and so
        # `du − ∂_t u_exact = w (F − ∂_t u_exact)`, truncation error, where it
        # was `(1 − w) ∂_t u_exact` before; and the run: the layer's residual
        # after 32 RK4 steps at `4/M` is within a factor two of the resting
        # hole's, where it was `13×` it (measured 2026-10-20: `3.33` against
        # `2.61`; without the rates `18.2`, `27.3` and `34.6` at 8, 16 and 32
        # steps against `1.09`, `1.74` and `2.61`, and with the layer's rate
        # but not the core's `12.3` at 32).
        q = 2
        G = q ÷ 2 + 1
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        ρ = T(4)
        hk = (; halfwidth=T(5 // 2), r_0=T(2 // 5), r_1=T(11 // 10), chunk=T(1 // 10),
              x0=-1 // 4, gauge_source=:algebraic, octant=:quadrant)
        mov = kerr_schild_case(T; hk..., v=v)
        function setup(case, t)
            f = gh_forest(T, case; N=12, roots=(4, 2, 2))
            U = FieldSet{T}(f, 20; G=G, centering=vertexcentered(3),
                            parity=state_parity(f))
            fill_exact!(U, case, t)
            u = statevector(U)
            gather!(u, U)
            return f, U, u
        end
        f, U, u = setup(mov, one(T))
        mint = with_ρ_max(mov.interior, ρ)
        sched = GhostSchedule(U, ops)
        du, dF = similar(u), similar(u)
        gh_rhs!(du, u, GHProblem(U, sched, mov; q=q, interior=mint), one(T))
        gh_rhs!(dF, u, GHProblem(U, sched, mov; q=q, interior=nothing), one(T))
        D = FieldSet{T}(f, 20; G=G, centering=vertexcentered(3), parity=state_parity(f))
        DF = FieldSet{T}(f, 20; G=G, centering=vertexcentered(3), parity=state_parity(f))
        scatter!(D, du)
        scatter!(DF, dF)
        c1 = center_at(mint.center, one(T))
        ident, scale, steady, lag, npts = zero(T), zero(T), zero(T), zero(T), 0
        for k in 1:nleaves(f), I in CartesianIndices(interiorview(U, k, 1))
            J = Tuple(I) .+ U.G
            x = SVector{3,T}(coordinates(U, k, J)...)
            r = sqrt(sum(abs2, x - c1))
            mint.r_0 ≤ r < mint.r_1 || continue
            w, _ = TGHb.interior_profiles(mint, r)
            ∂ₜe = -∇along(one(T), x, vv; bg=mov.background)
            ident = max(ident, maximum(abs, D.work[J..., :, k] -
                                            (w * DF.work[J..., :, k] + (1 - w) * ∂ₜe)))
            scale = max(scale, maximum(abs, ∂ₜe), maximum(abs, DF.work[J..., :, k]))
            steady = max(steady, maximum(abs, D.work[J..., :, k] - ∂ₜe))
            lag = max(lag, maximum(abs, (1 - w) * ∂ₜe))
            npts += 1
        end
        @info "moving :damped layer: $npts points, max|du − (w F + (1 − w) ∂ₜu)| = " *
              "$ident of $scale; max|du − ∂ₜu| = $steady, without the rate $lag"
        @test npts > 1000
        @test ident ≤ 1e-7 * scale                      # measured 4.7e−9
        @test steady ≤ lag / 8                          # measured 0.081
        dt = T(5 // 48) / 8
        res = map((mov, kerr_schild_case(T; hk...))) do case
            _, U0, u0 = setup(case, zero(T))
            p = GHProblem(U0, GhostSchedule(U0, ops), case; q=q,
                          interior=with_ρ_max(case.interior, ρ))
            u1 = gh_solve(p, u0, (zero(T), 32 * dt); dt=dt)
            gh_error!(p, u1, 32 * dt)
            error_norms(p).residual
        end
        @info "layer residual after 32 steps at 4/M: moving $(res[1]), at rest $(res[2])"
        @test isfinite(res[1]) && res[1] ≤ 2 * res[2]
    end

    @testset "the moving hole on the quadrant is the hole in the box, to roundoff" begin
        # Guards the whole chain under the mirrors with a hole that moves: the
        # parities, the moving core and layer straddling `y = 0` and `z = 0`,
        # the algebraic source at a velocity, the Gaussian `γ0` following
        # `c(t)`, the Dirichlet data of a moving hole, the non-cubic brick. The
        # reflection tests' fixture (`q = 2`, half-width `5/2`, uniform
        # `h = 5/48`), its layer narrowed to fit the contracted horizon
        # (`r_h,min = 2√(1 − v²) = 1.96`, so `r_1 ≤ 1.13` at `m = 8`) and
        # relaxing at `4/M`, the hole starting at `x = −1/4` and moving at
        # `1/5`, as the quadrant and as the full box, compared point by point
        # outside the core after sixteen RK4 steps. Through `gh_solve` rather
        # than `evolve!`: a translated, boosted background is a new type, so
        # everything a run touches compiles anew, and the driver's analysis is
        # most of that and is not what is claimed. Uniform meshes only: a
        # refined quadrant is a different discretization (`CLAUDE.md`, "A
        # refined octant is not the refined full box").
        q = 2
        G = q ÷ 2 + 1
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        hk = (; halfwidth=T(5 // 2), r_0=T(2 // 5), r_1=T(11 // 10), chunk=T(1 // 10),
              v=v, x0=-1 // 4, gauge_source=:algebraic)
        h = T(5 // 48)
        dt = h / 8
        nsteps = 16
        runs = map((:quadrant, false)) do octant
            case = kerr_schild_case(T; hk..., octant=octant)
            f = gh_forest(T, case; N=12, roots=octant === false ? 4 : (4, 2, 2))
            U = FieldSet{T}(f, 20; G=G, centering=vertexcentered(3),
                            parity=state_parity(f))
            fill_exact!(U, case, zero(T))
            u = statevector(U)
            gather!(u, U)
            p = GHProblem(U, GhostSchedule(U, ops), case; q=q,
                          interior=with_ρ_max(case.interior, 4))
            u1 = gh_solve(p, u, (zero(T), nsteps * dt); dt=dt)
            change = maximum(abs, u1 - u)
            scatter!(U, u1)
            (; case, U, change)
        end
        o, b = runs
        @test o.U.forest.roots == (4, 2, 2) && minimum_spacing(T, o.U.forest) == h
        @test o.case.gauge === b.case.gauge
        # The box's owned point at position `x`.
        function point(U, x)
            f = U.forest
            for k in 1:nleaves(f)
                org = block_origin(T, f, f.leaves[k])
                i = ntuple(d -> round(Int, (x[d] - org[d]) / spacing(T, f, f.leaves[k])), 3)
                all(d -> 0 ≤ i[d] < f.N, 1:3) && return k, i .+ (1 .+ U.G)
            end
            return nothing
        end
        c1 = center_at(o.case.center, nsteps * dt)
        diff, scale, npts = zero(T), zero(T), 0
        for k in 1:nleaves(o.U.forest), I in CartesianIndices(interiorview(o.U, k, 1))
            J = Tuple(I) .+ o.U.G
            x = SVector{3,T}(coordinates(o.U, k, J)...)
            sum(abs2, x - c1) < (2 // 5)^2 && continue      # the core: stale by design
            kb, Jb = point(b.U, x)
            diff = max(diff, maximum(abs, o.U.work[J..., :, k] .- b.U.work[Jb..., :, kb]))
            scale = max(scale, maximum(abs, o.U.work[J..., :, k]))
            npts += 1
        end
        @info "moving Kerr-Schild hole, quadrant against the box after $nsteps steps: " *
              "max|Δ| = $diff, max|u| = $scale over $npts points, " *
              "max|u(t) − u(0)| = $(o.change)"
        @test npts > 27_000
        @test o.change > 1e-6                   # the state moved
        # Roundoff of the target's rate, not of the arithmetic (amended
        # 2026-10-20): the moving layer's `∂_t Π` is a centered difference in
        # time, whose roundoff `ε |Π|/δ ≈ ε^{2/3} |Π|` differs between a point
        # and its mirror image, and the quadrant's ghosts are the mirror image.
        # `64 eps` before the rate; measured `3.8e−11` (`0.01 ε^{2/3}`) with it.
        @test diff ≤ eps(T)^(2 // 3) * scale
    end
end
