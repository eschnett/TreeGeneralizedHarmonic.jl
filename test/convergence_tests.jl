# The order of the scheme, measured by evolving solutions that are known
# exactly.
#
# `CODE.md`, milestone G2. This is the file the whole of steps 1–3 exists
# to make possible: the pointwise algebra, the stencils and the streaming
# kernel are all correct in the only sense that matters here, which is
# that the error against an analytic solution falls like `h^q`.
#
# Two cases, and between them they cover what step 3 built:
#
#   * the **gauge wave**, periodic in every dimension, harmonic (`H ≡ 0`),
#     and genuinely time-dependent — the case that measures `q`;
#   * **shifted Minkowski**, static, with a nonzero shift, a *sampled*
#     gauge source and a **Dirichlet** boundary in `x` — the case that
#     measures the `Hsrc` path and the boundary hook.
#
# **The resolutions and the times are a budget** (proposed in step 3).
# `PLAN.md` asked for `N = 8`, roots `2 … 8` and one crossing; that is
# minutes per order at three dimensions, and two of its numbers cannot be
# had at all: `N = 8` is below TreeAMR's vertex invariant `N ≥ 2G + 2` at
# `q = 6`, where `G = 4` forces `N ≥ 10`. What is run instead is roots
# `1, 2, 3` and an eighth of a crossing, which is enough time for the
# error to be a hundred times the roundoff floor at the finest resolution
# and short enough that the file stays near `PLAN.md`'s 30 s rule of
# thumb. The rates below are what that measures; the numbers are recorded
# in `CODE.md`.

using SpacetimeMetrics: Minkowski

# The rate a row must reach. The measured values sit within 0.09 of `q`
# (recorded in `CODE.md`), so the margin below is generous by design: what
# it has to catch is an order *lost* — a stencil at the wrong width, a
# ghost filled at the wrong order, a coefficient derivative that is not
# the chain rule — and those cost a whole order, not a tenth of one.
rate_floor(q) = q - 1 // 4

@testset "The gauge wave converges at order q: q=$q" for q in (2, 4, 6)
    # Guards the scheme's order, which is the one claim that would survive
    # every unit test in this package being wrong in the same direction at
    # once. The gauge wave is flat spacetime in an oscillating chart, so
    # every term of the right-hand side is exercised and none of them is
    # zero, and the exact solution is known at every time.
    T = Float64
    N = q == 6 ? 10 : 8
    case = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=zero(T), γ0=one(T),
                           γ2=zero(T))
    t_end = T(1 // 8)
    results = map((1, 2, 3)) do roots
        gh_errors(T, case; N=N, roots=roots, q=q, t_end=t_end)
    end
    hs = [r.h for r in results]
    l2 = [r.l2 for r in results]
    linf = [r.linf for r in results]
    rate_l2 = convergence_rate(hs, l2)
    rate_linf = convergence_rate(hs, linf)
    @info "gauge wave q=$q N=$N: h=$hs l2=$l2 linf=$linf " *
          "rate_l2=$rate_l2 rate_linf=$rate_linf"
    @test rate_l2 ≥ rate_floor(q)
    @test rate_linf ≥ rate_floor(q)
    @test issorted(l2; rev=true)           # and it falls at every step
    @test issorted(linf; rev=true)
    # Far above the roundoff floor at the finest resolution, which is what
    # makes the rate a measurement of truncation error and not of `eps/h²`
    # (step 2's warning about the window).
    @test l2[end] > 1e-12
end

@testset "Dissipation does not cost the gauge wave its order" begin
    # Guards `CODE.md`'s claim for the Kreiss–Oliger term: it is
    # `O(h^{2r−1}) = O(h^{q+1})` on smooth data, one order better than the
    # scheme, so a run at the recipe's `ε_KO = 0.5` must converge at `q`
    # just as one without it does. If the term were scaled by the wrong
    # power of `h` — the split between the weights and the caller is a
    # single `1/h_d`, not `2r − 1` of them — this is where it would show,
    # as an order lost rather than as an instability.
    T = Float64
    q = 4
    case = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=T(1 // 2),
                           γ0=one(T), γ2=T(-1 // 2))
    results = map((1, 2, 3)) do roots
        gh_errors(T, case; N=8, roots=roots, q=q, t_end=T(1 // 8))
    end
    hs = [r.h for r in results]
    l2 = [r.l2 for r in results]
    rate_l2 = convergence_rate(hs, l2)
    @info "gauge wave with dissipation q=$q ε_KO=$(case.ε_KO): h=$hs " *
          "l2=$l2 rate_l2=$rate_l2"
    @test rate_l2 ≥ rate_floor(q)
end

@testset "Shifted Minkowski converges at order q through a Dirichlet face" begin
    # Guards the two paths the gauge wave does not touch: the sampled
    # gauge source `Hsrc` (this background is not harmonic, so `H ≠ 0` and
    # the source terms of `S0` are all live) and the time-dependent
    # Dirichlet hook at the `x` faces. A boundary that filled its ghosts
    # from a stale time, or a gauge source read at the wrong slot, costs
    # the run its order here while leaving the periodic gauge wave
    # untouched.
    T = Float64
    q = 4
    case = shifted_minkowski_case(T; A=T(1 // 2), w=T(2), halfwidth=T(2),
                                  ε_KO=zero(T), γ0=one(T), γ2=zero(T))
    @test !isharmonic(case.background)
    @test case.periodic == (false, true, true)
    results = map((2, 3, 4)) do roots
        gh_errors(T, case; N=8, roots=roots, q=q, t_end=T(1 // 4))
    end
    hs = [r.h for r in results]
    l2 = [r.l2 for r in results]
    linf = [r.linf for r in results]
    rate_l2 = convergence_rate(hs, l2)
    rate_linf = convergence_rate(hs, linf)
    @info "shifted Minkowski q=$q: h=$hs l2=$l2 linf=$linf " *
          "rate_l2=$rate_l2 rate_linf=$rate_linf"
    @test rate_l2 ≥ rate_floor(q)
    @test rate_linf ≥ rate_floor(q)
    @test issorted(l2; rev=true)
end

@testset "Minkowski does not move under the integrator either" begin
    # Guards the end-to-end path the convergence studies run through —
    # `gh_dt`, `ODEProblem`, four RK4 stages per step, the ghost fill
    # between them — on the one solution whose right-hand side is *exactly*
    # zero. Anything that leaked a term anywhere in that loop shows here as
    # a state vector that is no longer identically zero, with no tolerance
    # to argue about.
    T = Float64
    for q in (2, 4)
        case = minkowski_case(T; L=one(T), ε_KO=T(1 // 2), γ0=one(T),
                              γ2=T(-1 // 2))
        r = gh_errors(T, case; N=8, roots=1, q=q, t_end=T(1 // 2))
        @test r.nsteps ≥ 4
        @test r.l2 == 0
        @test r.linf == 0
    end
end
