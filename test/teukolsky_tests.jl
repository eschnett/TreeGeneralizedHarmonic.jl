# The Teukolsky wave (step B5): the linearised even-parity `ℓ = 2`, `m = 0`
# quadrupole wave, `src/teukolsky.jl`.
#
# It is a solution of the vacuum equations to first order in its amplitude
# only, so every claim here is a scaling: the Ricci tensor and the harmonic
# gauge source, both through `SpacetimeMetrics`' automatic differentiation of
# the metric, fall as `a²`, while the metric's deviation from Minkowski falls
# as `a`. A wrong sign or factor in Teukolsky's `A`, `B`, `C` or in their
# Cartesian components leaves a Ricci tensor of order `a`, and a wrong
# ingoing half leaves one that is singular at the origin. The series about
# the origin is the closed form's cancellation done exactly, and the two must
# agree where the metric switches between them.

using ForwardDiff: ForwardDiff
using SpacetimeMetrics: RicciTensor, ddmetric, dmetric, gauge_source, metric
using StaticArrays: SMatrix, SVector
using TreeGeneralizedHarmonic: teukolsky_abc_closed, teukolsky_abc_series

@testset verbose = true "The Teukolsky wave" begin
    T = Float64
    η = SMatrix{4,4,T}(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    # Points at both sides of the switch (`r = 1/2` at `λ = 1`), near the
    # origin and far out, at the wave's peak and later.
    points = (SVector{4,T}(0.3, 0.7, -0.4, 0.9), SVector{4,T}(1.1, 0.2, 0.3, 0.1),
              SVector{4,T}(0, 0.05, 0.1, -0.2), SVector{4,T}(0.2, 2.0, 1.0, 1.5),
              SVector{4,T}(-0.7, -0.3, 0.25, 0.2))

    # Guards a wrong coefficient in `A`, `B` or `C`, a wrong angular function,
    # or a wrong Cartesian component. The metric is `η + a h̃` with `h̃` exactly
    # linear in `a`, so the first-order statement is exact: `h̃` is transverse,
    # traceless and solves the flat wave equation, `□h̃ = 0`, to roundoff of its
    # second derivatives — through `SpacetimeMetrics`' `ddmetric`.
    @testset "h is a linearised vacuum solution: □h = 0, ∂_i h_ij = 0" begin
        m = TeukolskyWave(T; amplitude=1, width=1)
        for x in points
            _, dg, ddg = ddmetric(m, x)
            scale = maximum(abs, ddg)
            for i in 2:4, j in 2:4
                box = -ddg[i, j, 1, 1] + ddg[i, j, 2, 2] + ddg[i, j, 3, 3] + ddg[i, j, 4, 4]
                @test abs(box) ≤ 1e-12 * scale
            end
            for j in 2:4
                @test abs(dg[2, j, 2] + dg[3, j, 3] + dg[4, j, 4]) ≤ 1e-13 * maximum(abs, dg)
            end
        end
    end

    # The nonlinear statement behind it: the full Ricci tensor, by three nested
    # dual passes, is `O(a²)` — a ratio of a hundred between the amplitudes
    # `10⁻³` and `10⁻⁴`, where an error of order `a` would give ten. It costs
    # about 70 s of compilation, and the linearised testset above makes the
    # same claim exactly, so it is the full tier's.
    @full_tier @testset "it is a vacuum solution to first order: Ricci is O(a²)" begin
        for x in points
            R(a) = maximum(abs, RicciTensor(TeukolskyWave(T; amplitude=a, width=1), x))
            h(a) = maximum(abs, metric(TeukolskyWave(T; amplitude=a, width=1), x) - η)
            @test h(1e-3) / h(1e-4) ≈ 10 rtol = 1e-9
            @test 70 < R(1e-3) / R(1e-4) < 130
        end
    end

    # Guards `isharmonic`'s row: a source of order `a` would be a gauge
    # constraint violation the scheme reports from the first step.
    @testset "it is harmonic to first order: Γ_a is O(a²)" begin
        for x in points
            H(a) = maximum(abs, gauge_source(TeukolskyWave(T; amplitude=a, width=1), x))
            @test 70 < H(1e-3) / H(1e-4) < 130
        end
        @test isharmonic(TeukolskyWave(T; amplitude=1e-3, width=1))
    end

    # Guards the switch between the two forms: a series with too few terms,
    # or a coefficient `κ(m)` off, is a jump in the metric at `r = λ/2`.
    @testset "the series about the origin is the closed form at the switch" begin
        m = TeukolskyWave(T; amplitude=1, width=1)
        for t in (0.0, 0.4, -1.1, 2.0), r in (0.45, 0.5, 0.55)
            c = teukolsky_abc_closed(m, t, r)
            s = teukolsky_abc_series(m, t, r^2)
            @test maximum(abs.(c .- s)) ≤ 1e-12 * maximum(abs.(c))
        end
        # The origin: `A = B = C = −a` at `t = 0`, and `h = a diag(1, 1, −2)`.
        @test all(teukolsky_abc_series(m, 0.0, 0.0) .≈ -1)
        g0 = metric(m, SVector{4,T}(0, 0, 0, 0))
        @test g0 - η ≈ SMatrix{4,4,T}(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -2)
    end

    # Guards the components that are not Teukolsky's: a time component, a
    # trace, or an asymmetry in the Cartesian form.
    @testset "it is transverse-traceless: h_tμ = 0, trace zero, symmetric" begin
        m = TeukolskyWave(T; amplitude=1e-2, width=1)
        for x in points
            h = metric(m, x) - η
            @test all(iszero, h[1, :])
            @test abs(h[2, 2] + h[3, 3] + h[4, 4]) ≤ 1e-14
            @test h == h'
        end
    end

    # Guards the origin, where the closed form is `0/0` and `dmetric`'s dual
    # pass differentiates `√r²`: the limit there must be the limit of the
    # metric along every ray, and the first derivatives finite, the spatial
    # ones zero (the metric is even in `x`).
    @testset "the origin is the metric's limit, and dmetric is finite there" begin
        m = TeukolskyWave(T; amplitude=1e-2, width=1)
        for t in (0.0, 0.3)
            p0 = SVector{4,T}(t, 0, 0, 0)
            g0, dg0 = dmetric(m, p0)
            @test all(isfinite, dg0)
            @test all(iszero, dg0[:, :, 2:4])
            for n in ((1, 0, 0), (0, 0, 1), (0.6, -0.48, 0.64)), δ in (1e-3, 1e-4)
                g = metric(m, SVector{4,T}(t, δ * n[1], δ * n[2], δ * n[3]))
                @test maximum(abs, g - g0) ≤ 1e-2 * 10 * δ^2
            end
            # `∂_t g` at the origin, against a difference in time.
            τ = 1e-5
            fd = (metric(m, SVector{4,T}(t + τ, 0, 0, 0)) -
                  metric(m, SVector{4,T}(t - τ, 0, 0, 0))) / 2τ
            @test maximum(abs, dg0[:, :, 1] - fd) ≤ 1e-8
        end
    end
end
