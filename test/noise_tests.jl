# Robust stability: white noise on flat space, a thousand steps.
#
# `CODE.md`, milestone G2, and the second of GHSO2's findings under
# "sonic-surface instability" (`notes/methods-ghso2.md`): the
# generalized-harmonic system on a finite-difference mesh grows a
# grid-scale mode that Kreiss–Oliger dissipation at `ε_KO ≈ 0.5` removes.
# This is the standard Apples-with-Apples robust-stability test — random
# unconstrained data of an amplitude far below any nonlinearity, evolved
# for many crossing times — and what it measures is the *discrete*
# system's spectrum, which no pointwise test can see.
#
# The claim is boundedness, not decay: the continuum system has zero
# modes (a gauge wave of amplitude `1e−8` is a solution), so the norm may
# not fall and must not climb. What is asserted is therefore a ceiling,
# and the number without dissipation is **recorded rather than asserted**
# — `PLAN.md` asks for exactly that, because it is a measurement of the
# scheme and not a requirement on it.

using SpacetimeMetrics: Minkowski

@testset "White noise on flat space stays bounded with dissipation" begin
    # Guards the configuration every black-hole run in this package will
    # use: `ε_KO = 0.5` with constraint damping on. A thousand steps at
    # `cfl = 1/4` on a box of side 1 is 18 crossing times, which is long
    # enough for an exponential to be unmistakable — the run without
    # dissipation below grows by an order of magnitude over the same
    # interval.
    T = Float64
    case = minkowski_case(T; L=one(T), ε_KO=T(1 // 2), γ0=one(T),
                          γ2=zero(T))
    r = gh_noise_growth(T, case; N=8, roots=1, q=4, nsteps=1000,
                        amplitude=T(1 // 10)^8)
    @info "robust stability ε_KO=0.5: nsteps=$(r.nsteps) t_end=$(r.t_end) " *
          "l2 $(r.l2_0) -> $(r.l2_1) (×$(r.l2_ratio)) linf ×$(r.linf_ratio)"
    @test r.finite
    @test r.t_end > 15                     # ~18 crossings of a unit box
    @test r.l2_ratio < 2
    @test r.linf_ratio < 5
    # The L∞ ratio is a transient, not a rate: measured at 250, 500, 1000
    # and 2000 steps it is 3.10, 2.53, 2.10, 1.45 (and the L2 ratio 1.01,
    # 0.92, 0.66, 0.64), so the norm turns over and falls. The ceiling
    # above is what a test can assert cheaply; the turnover is why
    # "bounded" is the right word.
end

@testset "Without dissipation the same noise grows, and by how much" begin
    # The negative control, and the number `CODE.md` records. Nothing here
    # asserts that the growth is *bad* — it asserts that it is there, so
    # that "the dissipation is what holds the grid-scale mode down" is a
    # measurement and not a story. If this run ever stopped growing, the
    # test above would have stopped testing anything.
    T = Float64
    case = minkowski_case(T; L=one(T), ε_KO=zero(T), γ0=one(T), γ2=zero(T))
    r = gh_noise_growth(T, case; N=8, roots=1, q=4, nsteps=1000,
                        amplitude=T(1 // 10)^8)
    @info "robust stability ε_KO=0: nsteps=$(r.nsteps) t_end=$(r.t_end) " *
          "l2 $(r.l2_0) -> $(r.l2_1) (×$(r.l2_ratio)) linf ×$(r.linf_ratio)"
    @test r.finite
    @test r.l2_ratio > 2
    # Still far below any nonlinearity: the metric is flat to within 1e-6,
    # so what grew is the discrete system's mode and not the physics.
    @test r.linf_1 < 1e-5
end
