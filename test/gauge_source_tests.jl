# The algebraic Kerr-Schild gauge source (added 2026-10-02): `CODE.md`, "Open
# questions", "Gauge sources that know less about the hole". A closed form in
# `g` and three constants of the run — `M`, the spin 4-vector, the 4-velocity
# — that makes every boosted, spinning Kerr-Schild hole exact; evaluated in
# the kernels instead of the sampled `Hsrc` field set.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using StaticArrays: SVector
import SpacetimeMetrics as SM

@testset verbose = true "The algebraic Kerr-Schild gauge source" begin
    T = Float64

    @testset "the closed form is the background's own −Γ_a, and its gradient" begin
        # Guards the formula, its sign, the boost of the spin 4-vector, the
        # `boost(m, v)`-moves-at-`−v` convention and the layout `∂_c H_a` of
        # the gradient — each of which a sampled-source run never sees — against
        # SpacetimeMetrics' `gauge_source_grad`, at twelve points per hole
        # between 2.5 M and 6 M from it.
        for (bg, src) in [
            (SM.KerrSchild(1.0, 0.0), KerrSchildSource(T; M=1)),
            (SM.KerrSchild(2.0, 0.0), KerrSchildSource(T; M=2)),
            (SM.KerrSchild(1.0, 0.9), KerrSchildSource(T; M=1, spin=(0, 0, 0.9))),
            (SM.translate(SM.KerrSchild(1.0, 0.0), SVector(0.0, 1.0, -2.0, 0.5)),
             KerrSchildSource(T; M=1)),
            (SM.boost(SM.KerrSchild(1.0, 0.5), SVector(0.3, 0.0, 0.0)),
             KerrSchildSource(T; M=1, spin=(0, 0, 0.5), velocity=(-0.3, 0, 0))),
            (SM.boost(SM.KerrSchild(1.0, 0.7), SVector(0.2, 0.3, -0.1)),
             KerrSchildSource(T; M=1, spin=(0, 0, 0.7), velocity=(-0.2, -0.3, 0.1)))]
            @test check_gauge_source(bg, src, T; center=(0.0, 0.0, 0.0),
                                     scale=src.M) ≤ 1e-13   # measured 2–4e−15
        end
    end

    @testset "a source with the wrong constants is refused" begin
        # Guards the check a case runs at construction: the wrong mass, the
        # boost's velocity with the wrong sign, a spin the hole does not have,
        # and a background that is not Kerr-Schild.
        ks = SM.KerrSchild(1.0, 0.0)
        msg = "does not make the background exact"
        @test_throws msg check_gauge_source(ks, KerrSchildSource(T; M=1.1), T;
                                            center=(0.0, 0.0, 0.0), scale=1.0)
        @test_throws msg check_gauge_source(
            SM.boost(ks, SVector(0.3, 0.0, 0.0)),
            KerrSchildSource(T; M=1, velocity=(0.3, 0, 0)), T;
            center=(0.0, 0.0, 0.0), scale=1.0)
        @test_throws msg check_gauge_source(
            ks, KerrSchildSource(T; M=1, spin=(0, 0, 0.5)), T;
            center=(0.0, 0.0, 0.0), scale=1.0)
        @test_throws msg check_gauge_source(
            SM.Harmonic(1.0, 0.0), KerrSchildSource(T; M=1), T;
            center=(0.0, 0.0, 0.0), scale=1.0)
        @test_throws msg kerr_schild_case(T; M=2, halfwidth=5, r_0=3 // 2, r_1=3,
                                          chunk=1, gauge_source=KerrSchildSource(T; M=1))
        @test_throws "Kerr-Schild hole at rest" harmonic_kerr_case(
            T; halfwidth=5, r_0=1 // 4, r_1=1 // 2, chunk=1, gauge_source=:algebraic)
        @test_throws "speed of light" KerrSchildSource(T; M=1, velocity=(1, 0, 0))
    end

    @testset "a static hole runs on it as on the sampled source" begin
        # Guards the kernel path — `Val(:algebraic)` with the source itself
        # where the `Hsrc` array was, in the right-hand side, the gauge
        # constraint and the ADM monitor — on the suite's octant fixture
        # (`q = 2`, `h = 5/48`) to `1/5 M`: the same hole and the same exact
        # initial data, so the two runs differ only in how the source is
        # linearised, at the truncation error's size.
        q = 2
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        kw = (; halfwidth=T(5 // 2), r_0=T(2 // 5), r_1=T(23 // 20), chunk=T(1 // 10),
              octant=true)
        runs = map((nothing, :algebraic)) do gs
            case = kerr_schild_case(T; kw..., gauge_source=gs)
            evolve!(T, case; forest=gh_forest(T, case; N=12, roots=2), q=q, ops=ops,
                    t_end=T(1 // 5), adm_every=1)
        end
        s, a = runs
        @test s.problem.valH === Val(true) && s.problem.Hsrc isa FieldSet
        @test a.problem.valH === Val(:algebraic) && a.problem.Hsrc isa KerrSchildSource
        # The initial data are exact for both, so their monitors agree to
        # roundoff (the ADM monitor reaches `∂H` through each source's own
        # arithmetic).
        @test a.records[1].gauge_l2 ≈ s.records[1].gauge_l2 rtol = 1e-10
        @test a.records[1].ham_l2 ≈ s.records[1].ham_l2 rtol = 1e-10
        ra, rs = a.records[end], s.records[end]
        @info "sampled against algebraic at 1/5 M: err $(rs.err_l2) / $(ra.err_l2), " *
              "C $(rs.gauge_l2) / $(ra.gauge_l2), ℋ $(rs.ham_l2) / $(ra.ham_l2)"
        @test ra.finite && rs.finite
        @test 1 / 2 < ra.err_l2 / rs.err_l2 < 2
        @test 1 / 2 < ra.gauge_l2 / rs.gauge_l2 < 2
        @test 1 / 2 < ra.ham_l2 / rs.ham_l2 < 2
    end
end

# --- step B1: the composite, the damped harmonic gauge, the roll-on, and the
# momentum that satisfies `C_a = 0` (`CODE.md`, "Binary black holes", "The
# gauge"). The closed forms below are written out again, plainly and apart from
# `src/gauge.jl`'s kernel spelling, and differentiated by ForwardDiff: what a
# source's `∂_a H_b` must be is the derivative of the `H` it returns, every
# explicit dependence on `x` and `t` included.

using ForwardDiff
using LinearAlgebra: det, dot
using Random: Xoshiro
using StaticArrays: SArray, SMatrix

const ηB1 = SMatrix{4,4,Float64}(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)

# Kerr-Schild's closed form for one hole at its 4-velocity `u` and spin `S`.
function ks_closed(g, M, u, S)
    k = g - ηB1
    w = k * u
    K = dot(u, w)
    s = dot(S, w)
    return -K * w / (M + sqrt(max(M^2 - s^2, zero(s))))
end

# A hole's line as the test spells it: `u = γ(1, v)` and the boosted spin, with
# `v = v₀ + a τ`, to first order in `τ` about `t_p` (the piece's line), the
# rates by ForwardDiff — taken once, outside any other derivative, so that no
# dual of the outer Jacobian is ever nested in them; the center to second order.
u_of_v(v) = (γ = 1 / sqrt(1 - sum(abs2, v)); SVector(γ, γ * v[1], γ * v[2], γ * v[3]))
function S_of_v(v, s)
    γ = 1 / sqrt(1 - sum(abs2, v))
    vs = dot(v, s)
    Sv = s + (γ^2 / (γ + 1) * vs) * v
    return SVector(γ * vs, Sv[1], Sv[2], Sv[3])
end
function hole_line(hole)
    v0, a, s = SVector{3,Float64}(hole.v), SVector{3,Float64}(hole.a),
               SVector{3,Float64}(hole.s)
    return (; hole.M, t_p=Float64(hole.t_p), c0=SVector{3,Float64}(hole.c), v0, a,
            u0=u_of_v(v0), u̇=ForwardDiff.derivative(σ -> u_of_v(v0 + σ * a), 0.0),
            S0=S_of_v(v0, s), Ṡ=ForwardDiff.derivative(σ -> S_of_v(v0 + σ * a, s), 0.0))
end
function hole_closed(l, t)
    τ = t - l.t_p
    return l.u0 + τ * l.u̇, l.S0 + τ * l.Ṡ, l.c0 + τ * l.v0 + (τ^2 / 2) * l.a
end
function composite_closed(g, x, t, lines)
    K = length(lines)
    at = [hole_closed(l, t) for l in lines]
    Hs = [ks_closed(g, l.M, a[1], a[2]) for (l, a) in zip(lines, at)]
    q = [sum(abs2, x - a[3])^2 for a in at]
    P = [prod((q[B] for B in 1:K if B != A); init=one(eltype(q))) for A in 1:K]
    w = P ./ sum(P)
    return sum(w[A] * Hs[A] for A in 1:K)
end
function dh_closed(g, x, p)
    α, β, γ = adm_from_metric(g)
    L = log(sqrt(det(γ)) / α)
    ℓ = log(1 / α)
    W = exp(-sum(abs2, x - SVector{3}(p.center)) / p.σ_r^2)
    μ1 = p.A_L1 * W * L^p.e_L1
    μ2 = p.A_L2 * W * ℓ^p.e_L2
    μS = p.A_S * W * L^p.e_S
    n = SVector(-α, zero(α), zero(α), zero(α))
    gβ = SVector(ntuple(a -> sum(g[a, i + 1] * β[i] for i in 1:3), 4))
    return (μ1 * L + μ2 * ℓ) * n - μS * gβ / α
end
rollon_closed(t, t_0, σ_g) = t ≤ t_0 ? one(t) : exp(-((t - t_0) / σ_g)^4)

TrackedHoleB1(T, h) = TrackedHole(T; M=h.M, spin=h.s, center=h.c, velocity=h.v,
                                  acceleration=h.a, t_p=h.t_p)
dh_source(T, p) = DampedHarmonicSource(T; σ_r=p.σ_r, A_L1=p.A_L1, A_L2=p.A_L2,
                                       A_S=p.A_S, e_L1=p.e_L1, e_L2=p.e_L2,
                                       e_S=p.e_S, center=p.center)

# A random smooth metric near the given ones: `g(p) = Σ g_bg(p) − (n − 1) η + ε P(p)`,
# `P` a symmetric field of sines and cosines.
function near_metric(bgs, rng; ε=1 // 20)
    S = [(A = randn(rng, 4, 4); SMatrix{4,4}(A + A')) for _ in 1:3]
    k = [SVector{4}(randn(rng, 4)) for _ in 1:2]
    return p -> sum(SM.metric(bg, p) for bg in bgs) - (length(bgs) - 1) * ηB1 +
                ε * (S[1] + sin(dot(k[1], p)) * S[2] + cos(dot(k[2], p)) * S[3])
end

# `H` and `∂_c H_a` of `src` at `p0` — the source a launch at that time
# evaluates — against the closed form's value and its ForwardDiff Jacobian, as
# the worst error over their size.
function source_against_closed(src, closed, gfun, p0)
    hfun(p) = pack_g(gfun(p))
    J = ForwardDiff.jacobian(hfun, p0)                  # J[v, c] = ∂_c h_v
    h = hfun(p0)
    ∂ₜh = SVector{10}(J[:, 1])
    ∂h = ntuple(i -> SVector{10}(J[:, i + 1]), 3)
    H, dH = algebraic_gauge_source(effective_source(src, p0[1]), (p0[2], p0[3], p0[4]),
                                   p0[1], h, ∂ₜh, ∂h)
    Hfun(p) = closed(gfun(p), SVector(p[2], p[3], p[4]), p[1])
    Href = Hfun(p0)
    dHref = ForwardDiff.jacobian(Hfun, p0)'             # dHref[c, a] = ∂_c H_a
    scale = max(maximum(abs, Href), maximum(abs, dHref))
    return max(maximum(abs, H - Href), maximum(abs, dH - dHref)) / scale, scale
end

# Points between `rmin` and `rmax` from `center`, at times in `[0, 3]`.
function sample_points(rng, center, n; rmin=2.5, rmax=6.0)
    return [SVector{4}(3 * rand(rng),
                       (SVector{3}(center) + (rmin + (rmax - rmin) * rand(rng)) *
                        normalize3(SVector{3}(randn(rng, 3))))...) for _ in 1:n]
end
normalize3(v) = v / sqrt(sum(abs2, v))

# A background's Cauchy data at `t = 0`: `γ`, `K` (the package's sign), `α`, `β`,
# their spatial derivatives, and the analytic `Π`.
function cauchy_data(bg, x)
    g, dg = SM.dmetric(bg, SVector(0.0, x...))
    α, β, γ = adm_from_metric(g)
    lapse_shift(y) = ((a, b, _) = adm_from_metric(SM.metric(bg, SVector(zero(eltype(y)), y...)));
                      SVector(a, b[1], b[2], b[3]))
    J = ForwardDiff.jacobian(lapse_shift, x)       # J[1, k] = ∂_k α, J[1 + i, k] = ∂_k β^i
    ∂α = SVector{3}(J[1, 1], J[1, 2], J[1, 3])
    ∂β = ntuple(k -> SVector{3}(J[2, k], J[3, k], J[4, k]), 3)
    ∂γ = ntuple(k -> SMatrix{3,3}(dg[i + 1, j + 1, k + 1] for i in 1:3, j in 1:3), 3)
    h, Π, ∂h = background_state(bg, 0.0, x)
    _, _, K = adm_vars_from_state(h, Π, ∂h...)
    return (; γ, K, α, β, ∂γ, ∂α, ∂β, Π)
end

@testset verbose = true "Step B1: the composite, the damped harmonic gauge, the roll-on" begin
    T = Float64
    rng = Xoshiro(20261025)
    # Two holes of a binary, a piece of each one's track: spins, velocities and
    # accelerations in general directions, the piece starting at `t_p = 1/2`.
    holeA = (M=1 // 2, s=(0, 0.1, 0.4), c=(4, 0.5, 0), v=(0.05, 0.18, 0.02),
             a=(-0.012, 0.003, 0.001), t_p=0.5)
    holeB = (M=0.45, s=(0.05, 0, -0.2), c=(-4, -0.5, 0.2), v=(-0.04, -0.2, 0.01),
             a=(0.013, -0.002, 0), t_p=0.5)
    dhp = (σ_r=8.0, A_L1=1.0, A_L2=0.5, A_S=0.8, e_L1=2, e_L2=1, e_S=2,
           center=(1.0, -0.5, 0.3))
    dhp0 = (dhp..., A_L2=0.0, e_L1=0, e_L2=0, e_S=0)

    @testset "a tracked hole's rates are the derivatives of its 4-velocity and spin" begin
        # Guards `TrackedHole`'s `u̇` and `Ṡ` — the boost of the spin is the
        # piece of the composite no closed form below recomputes the same way —
        # and that a hole without acceleration is `KerrSchildSource`'s own.
        for h in (holeA, holeB)
            th = TrackedHoleB1(T, h)
            v0, a, s = SVector{3}(h.v), SVector{3}(h.a), SVector{3}(h.s)
            @test th.ks.u ≈ u_of_v(v0) rtol = 1e-15
            @test th.ks.S ≈ S_of_v(v0, s) rtol = 1e-14
            @test th.du ≈ ForwardDiff.derivative(σ -> u_of_v(v0 + σ * a), 0.0) rtol = 1e-13
            @test th.dS ≈ ForwardDiff.derivative(σ -> S_of_v(v0 + σ * a, s), 0.0) rtol = 1e-13
        end
        still = TrackedHole(T; M=1, spin=(0, 0, 1 // 2), center=(1, 2, 3),
                            velocity=(1 // 5, 0, 0))
        @test still.ks === KerrSchildSource(T; M=1, spin=(0, 0, 1 // 2),
                                            velocity=(1 // 5, 0, 0))
        @test iszero(still.du) && iszero(still.dS)
        @test isbits(CompositeKerrSchildSource(TrackedHoleB1(T, holeA),
                                               TrackedHoleB1(T, holeB)))
        roll = RolledGaugeSource(KerrSchildSource(T; M=1), dh_source(T, dhp); σ_g=15,
                                 t_0=2)
        @test isbits(roll)
        # A launch is given `from` until the roll starts and `to` once it is
        # done, which is what makes those two stretches their sources' bits.
        @test effective_source(roll, 2.0) === roll.from
        @test effective_source(roll, 2.0 + 5.3 * 15) === roll.to
        @test effective_source(roll, 10.0) === roll
        @test rollon_weight(roll, 2.0) == (1.0, -0.0) &&
              rollon_weight(roll, 17.0)[1] ≈ exp(-1)
    end

    @testset "H and ∂H of each source are the forward-mode derivatives of its closed form" begin
        # Guards every chain rule the kernels evaluate: the composite's weights
        # and their gradients in space and in time (through each center's
        # velocity), each hole's rate through its acceleration, the damped
        # harmonic gauge's `α`, `β`, `√γ` and Gaussian, the roll-on's `∂_t f`.
        # A missing explicit term would make `∂_t H` the derivative of another
        # `H`, which is a gauge transient in `C_a` and nothing else. Random
        # smooth metrics near the holes, eight points each.
        bgA = SM.translate(SM.boost(SM.KerrSchild(0.5, 0.0), SVector(-0.05, -0.18, -0.02)),
                           SVector(0.0, 4.0, 0.5, 0.0))
        bgB = SM.translate(SM.boost(SM.KerrSchild(0.45, 0.0), SVector(0.04, 0.2, -0.01)),
                           SVector(0.0, -4.0, -0.5, 0.2))
        ksbg = SM.boost(SM.KerrSchild(1.0, 0.6), SVector(0.1, -0.2, 0.05))
        one_hole = (M=1.0, s=(0, 0, 0.6), c=(0, 0, 0), v=(-0.1, 0.2, -0.05),
                    a=(0.01, 0.02, -0.005), t_p=1.0)
        comp1 = CompositeKerrSchildSource(TrackedHoleB1(T, one_hole))
        comp2 = CompositeKerrSchildSource(TrackedHoleB1(T, holeA), TrackedHoleB1(T, holeB))
        lines1 = [hole_line(one_hole)]
        linesAB = [hole_line(holeA), hole_line(holeB)]
        ks = KerrSchildSource(T; M=1, spin=(0, 0, 0.6), velocity=(-0.1, 0.2, -0.05))
        rows = [
            ("one accelerating hole", comp1, (g, x, t) -> composite_closed(g, x, t, lines1),
             near_metric((ksbg,), rng), (0.0, 0.0, 0.0)),
            ("two holes", comp2, (g, x, t) -> composite_closed(g, x, t, linesAB),
             near_metric((bgA, bgB), rng), (2.0, 0.0, 0.0)),
            ("damped harmonic", dh_source(T, dhp), (g, x, t) -> dh_closed(g, x, dhp),
             near_metric((ksbg,), rng), (0.0, 0.0, 0.0)),
            ("damped harmonic, e = 0", dh_source(T, dhp0), (g, x, t) -> dh_closed(g, x, dhp0),
             near_metric((ksbg,), rng), (0.0, 0.0, 0.0)),
            ("roll-on, Kerr-Schild to damped harmonic",
             RolledGaugeSource(ks, dh_source(T, dhp); σ_g=1.5, t_0=0.25),
             (g, x, t) -> (f = rollon_closed(t, 0.25, 1.5);
                           f * ks_closed(g, 1.0, u_of_v(SVector(-0.1, 0.2, -0.05)),
                                         S_of_v(SVector(-0.1, 0.2, -0.05),
                                                SVector(0.0, 0.0, 0.6))) +
                           (1 - f) * dh_closed(g, x, dhp)),
             near_metric((ksbg,), rng), (0.0, 0.0, 0.0)),
            ("roll-on, two holes to damped harmonic",
             RolledGaugeSource(comp2, dh_source(T, dhp); σ_g=1.5),
             (g, x, t) -> (f = rollon_closed(t, 0.0, 1.5);
                           f * composite_closed(g, x, t, linesAB) +
                           (1 - f) * dh_closed(g, x, dhp)),
             near_metric((bgA, bgB), rng), (2.0, 0.0, 0.0))]
        for (label, src, closed, gfun, center) in rows
            worst = 0.0
            # Two holes: between them, more than a mass from either center.
            two = center != (0.0, 0.0, 0.0)
            for p0 in sample_points(rng, two ? (0.0, 0.0, 0.0) : center, 8;
                                    rmin=two ? 0.5 : 2.5, rmax=two ? 2.5 : 6.0)
                err, _ = source_against_closed(src, closed, gfun, p0)
                worst = max(worst, err)
            end
            @info "the source against its closed form" label worst
            @test worst ≤ 1e-12
        end
    end

    @testset "the composite's weights are a partition of unity, 1 − O((r/d)⁴) at a hole" begin
        # Guards `p = 4` and the product form: a weight that is not one at its
        # own hole puts the companion's source on the horizon.
        src = CompositeKerrSchildSource(TrackedHoleB1(T, holeA), TrackedHoleB1(T, holeB))
        cA = hole_closed(hole_line(holeA), 0.5)[3]
        for x in (cA, cA + SVector(1.0, 0, 0), SVector(0.3, 2.0, -1.0))
            w = composite_weights(src, x, 0.5)
            @test sum(w) ≈ 1 rtol = 1e-15
            @test all(≥(0), w)
        end
        @test composite_weights(src, cA, 0.5)[1] == 1
        # One horizon radius from hole A at a separation of eight: (1/8)⁴.
        cB = hole_closed(hole_line(holeB), 0.5)[3]
        d = sqrt(sum(abs2, cA - cB))
        w = composite_weights(src, cA + SVector(1.0, 0, 0) * (d / 8), 0.5)
        @test 0.5 * (1 / 8)^4 < 1 - w[1] < 2 * (1 / 8)^4
    end

    @testset "gauge_consistent_momentum is the boosted hole's own Π under its own source" begin
        # Guards the four linear equations: their `Γ⁰`, the `g^{tt}` on the
        # diagonal, the shift's and the lapse's rates, and `Π` from `∂_t g`. A
        # boosted, spinning, translated Kerr-Schild hole under its own source is
        # an exact solution with `C_a = 0`, so its `∂_t α` and `∂_t β` are what
        # the equations must return, and its `Π` what the data then have.
        bg = SM.translate(SM.boost(SM.KerrSchild(1.0, 0.7), SVector(0.2, 0.3, -0.1)),
                          SVector(0.0, 1.0, -2.0, 0.5))
        src = KerrSchildSource(T; M=1, spin=(0, 0, 0.7), velocity=(-0.2, -0.3, 0.1))
        comp = CompositeKerrSchildSource(TrackedHole(T; M=1, spin=(0, 0, 0.7),
                                                     center=(1, -2, 0.5),
                                                     velocity=(-0.2, -0.3, 0.1)))
        worst = 0.0
        for p in sample_points(rng, (1.0, -2.0, 0.5), 10)
            x = SVector(p[2], p[3], p[4])
            d = cauchy_data(bg, x)
            for s in (src, comp)
                r = gauge_consistent_momentum(s, x, 0.0, d.γ, d.K, d.α, d.β, d.∂γ,
                                              d.∂α, d.∂β)
                worst = max(worst, maximum(abs, r.Π - d.Π) / maximum(abs, d.Π))
            end
        end
        @info "gauge_consistent_momentum against the boosted hole's Π" worst
        @test worst ≤ 1e-12
    end

    @testset "gauge_consistent_momentum gives C_a = 0 for a perturbed lapse, under every source" begin
        # Guards the claim B8 rests on: whatever the source, the data's `Π` makes
        # the gauge constraint vanish at `t = 0` — checked by the package's own
        # `gauge_constraint_at_node`, not by the solve's arithmetic.
        bg = SM.boost(SM.KerrSchild(1.0, 0.5), SVector(0.15, 0.0, 0.1))
        srcs = (KerrSchildSource(T; M=1, spin=(0, 0, 0.5), velocity=(-0.15, 0, -0.1)),
                CompositeKerrSchildSource(TrackedHoleB1(T, holeA), TrackedHoleB1(T, holeB)),
                dh_source(T, dhp),
                RolledGaugeSource(KerrSchildSource(T; M=1, spin=(0, 0, 0.5),
                                                   velocity=(-0.15, 0, -0.1)),
                                  dh_source(T, dhp); σ_g=1))
        ψ(x) = exp(-sum(abs2, x - SVector(2.0, 1.0, 0.0)) / 4)
        worst = 0.0
        for p in sample_points(rng, (0.0, 0.0, 0.0), 6), (k, s) in enumerate(srcs)
            x = SVector(p[2], p[3], p[4])
            t = k == 4 ? 1.0 : 0.0                  # the roll-on half way
            d = cauchy_data(bg, x)
            ε = 1 // 10
            α = d.α * (1 + ε * ψ(x))
            ∂α = d.∂α * (1 + ε * ψ(x)) + d.α * ε * ForwardDiff.gradient(ψ, x)
            r = gauge_consistent_momentum(s, x, t, d.γ, d.K, α, d.β, d.∂γ, ∂α, d.∂β)
            g = TreeGeneralizedHarmonic._sym4(r.h) + ηB1
            dgs = (r.∂ₜh, r.∂h...)
            dg = SArray{Tuple{4,4,4}}(TreeGeneralizedHarmonic._sym4(dgs[c])[a, b]
                                      for a in 1:4, b in 1:4, c in 1:4)
            C = g * gauge_constraint_at_node(g, dg, inv(g) * r.Hl)
            scale = maximum(abs, r.Hl) + maximum(abs, r.∂ₜh)
            worst = max(worst, maximum(abs, C) / scale)
        end
        @info "C_a of gauge-consistent data" worst
        @test worst ≤ 1e-13
    end

    @testset "the damped harmonic gauge drives log(√γ/α), the lapse and the shift back" begin
        # Guards the sign, the one thing a transcription from SpECTRE's form can
        # get wrong and no derivative test notices: on flat space, constant data
        # with `K = 0` have `∂_t γ = 0`, and `C_a = 0` gives `∂_t α = α² μ_L1 L`
        # and `∂_t β^i = −α μ_S β^i` with `L = log(√γ/α)` — `L` decreasing
        # wherever it is not zero, the shift decaying (`CODE.md`, "The gauge").
        src = DampedHarmonicSource(T; σ_r=Inf)
        z3 = zero(SMatrix{3,3,T})
        z = zero(SVector{3,T})
        I3 = SMatrix{3,3,T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        for (α, γs, β) in ((1.1, 1.0, z), (0.9, 1.0, z), (1.0, 1.2, z), (1.0, 0.8, z),
                           (1.1, 1.0, SVector(0.1, -0.05, 0.0)))
            γ = γs * I3
            r = gauge_consistent_momentum(src, z, 0.0, γ, z3, α, β, (z3, z3, z3), z,
                                          (z, z, z))
            L = log(sqrt(det(γ)) / α)
            # `∂_t L = −∂_t α/α` here, so `L ∂_t L < 0` is the claim.
            @test L * (-r.∂ₜα / α) < 0
            @test r.∂ₜα ≈ α^2 * L^3 rtol = 1e-13          # A_L1 = 1, e_L1 = 2
            @test r.∂ₜβ ≈ -α * L^2 * β rtol = 1e-13 atol = 1e-17
            iszero(β) || @test all(r.∂ₜβ .* β .≤ 0)
        end
    end

    @testset "a lapse perturbation of flat space decays under the damped harmonic gauge, at C_a's truncation" begin
        # Guards the kernels' damped harmonic gauge on a mesh: its sign in
        # `∂_t Π` (a wrong sign grows the perturbation), its `∂_a H_b` (a wrong
        # one is a source of `C_a`), and gauge-consistent data filled point by
        # point. Periodic flat space, `α = 1 + ε sin(2πx/L)` with `β = 0`,
        # `γ = δ`, `K = 0` and the `Π` of `C_a = 0`, `q = 4`, `h = 1/4`, to `t = 2`:
        # with `A_L1 = A_S = 1` and `e = 0` (a linear rate, so that the decay
        # shows at this amplitude) against the same run with no damping, which
        # is the harmonic gauge (`H = 0`).
        q = 4
        Lbox = 4
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        ε = 1 // 10
        function run(A)
            src = DampedHarmonicSource(T; σ_r=Inf, A_L1=A, A_S=A, e_L1=0, e_L2=0, e_S=0)
            case = GHCase(T, SM.Minkowski(); box=ntuple(_ -> (zero(T), T(Lbox)), Val(3)),
                          periodic=(true, true, true), ε_KO=T(1 // 2), γ0=one(T),
                          γ2=zero(T), gauge_source=src)
            f = gh_forest(T, case; N=8, roots=2)
            U = FieldSet{T}(f, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                            parity=state_parity(f))
            z3 = zero(SMatrix{3,3,T})
            z = zero(SVector{3,T})
            I3 = SMatrix{3,3,T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
            k = 2π / Lbox
            fill_by_coordinates!(AllVariables(x -> begin
                    α = 1 + ε * sin(k * x[1])
                    r = gauge_consistent_momentum(src, x, zero(T), I3, z3, T(α), z,
                                                  (z3, z3, z3),
                                                  SVector{3,T}(ε * k * cos(k * x[1]), 0, 0),
                                                  (z, z, z))
                    (Tuple(r.h)..., Tuple(r.Π)...)
                end), U)
            u = statevector(U)
            gather!(u, U)
            p = GHProblem(U, GhostSchedule(U, ops), case; q=q)
            gh_constraint!(p, u, zero(T))
            C0 = maximum(constraint_norms(p).gauge_linf)
            dt = T(1 // 4) / 4
            u1 = gh_solve(p, u, (zero(T), T(2)); dt=dt)
            gh_constraint!(p, u1, T(2))
            C1 = maximum(constraint_norms(p).gauge_linf)
            # `h_tt = −α² + 1` with `β = 0`: the lapse's perturbation, twice.
            scatter!(U, u1)
            htt = maximum(b -> maximum(abs, interiorview(U, b, 1)), 1:nblocks(U))
            return (; C0, C1, htt)
        end
        damped, harmonic = run(1), run(0)
        @info "flat space, a lapse perturbation, to t = 2" damped harmonic
        # `C_a` at `t = 0` is the data's discretization (`1.65·10⁻⁴`, the same in
        # both: `H = 0` on flat space at rest), and it falls in both by `t = 2`
        # (measured `3.4·10⁻⁵` damped, `5.8·10⁻⁵` harmonic).
        @test damped.C0 ≤ 2e-4 && damped.C0 == harmonic.C0
        @test damped.C1 < damped.C0 / 2 && harmonic.C1 < harmonic.C0 / 2
        # The harmonic gauge keeps the perturbation (`|h_tt| ≈ 2ε`, measured
        # `0.210`); the damped one halves it (measured `0.107`).
        @test harmonic.htt > 2ε * 0.95
        @test damped.htt < 0.6 * harmonic.htt
    end
    @testset "one tracked hole is KerrSchildSource bit for bit, and a roll-on is its source before it starts" begin
        # Guards the claim step B3 builds on (`CODE.md`, "The gauge"): a case on
        # the composite of one hole is the case it was. The right-hand side and
        # the gauge constraint of the moving quadrant hole (`boosted_tests.jl`'s
        # fixture: `q = 2`, `h = 5/48`, the layer at `4/M`), on a state perturbed
        # everywhere, at both kernel widths, `isequal`. A launch is handed the
        # one hole's `KerrSchildSource` itself (`effective_source`): the
        # composite's own kernel is another specialisation, contracted into
        # fused multiply-adds differently, and differed in a third of the
        # entries by up to `6·10⁻¹³` against terms of `800` (measured in step
        # B1; its arithmetic is the closed-form and lane tests' above). The
        # roll-on before `t_0` is handed `from` itself, and half way it is not
        # the hole's source, which shows the comparison can fail.
        q = 2
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        kw = (; halfwidth=T(5 // 2), r_0=T(2 // 5), r_1=T(11 // 10), chunk=T(1 // 10),
              v=1 // 5, x0=-1 // 4, octant=:quadrant)
        ca = kerr_schild_case(T; kw..., gauge_source=:algebraic)
        cc = kerr_schild_case(T; kw..., gauge_source=:composite)
        roll = RolledGaugeSource(ca.gauge, DampedHarmonicSource(T; σ_r=32); σ_g=1, t_0=1)
        cr = kerr_schild_case(T; kw..., gauge_source=roll)
        @test cc.gauge isa CompositeKerrSchildSource{T,1}
        @test cc.gauge.holes[1].ks === ca.gauge
        f = gh_forest(T, ca; N=12, roots=(4, 2, 2))
        U = FieldSet{T}(f, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                        parity=state_parity(f))
        fill_exact!(U, ca, zero(T))
        u = statevector(U)
        gather!(u, U)
        u .+= T(1 // 10000) .* sin.(T.(1:length(u)))
        sched = GhostSchedule(U, ops)
        problem(c, W) = GHProblem(U, sched, c; q=q, interior=with_ρ_max(c.interior, 4),
                                  simd_width=W)
        for W in (1, nothing), t in (zero(T), T(1 // 2))
            dus = map((ca, cc, cr)) do c
                du = similar(u)
                gh_rhs!(du, u, problem(c, W), t)
                du
            end
            @test isequal(dus[1], dus[2])
            @test isequal(dus[1], dus[3])
        end
        diags = map((ca, cc)) do c
            p = problem(c, 1)
            gh_constraint!(p, u, T(1 // 2))
            copy(p.diag.work)
        end
        @test isequal(diags[1], diags[2])
        du_ks, du_roll = similar(u), similar(u)
        gh_rhs!(du_ks, u, problem(ca, nothing), T(2))
        gh_rhs!(du_roll, u, problem(cr, nothing), T(2))
        @test all(isfinite, du_roll) && !isequal(du_ks, du_roll)
    end
end
