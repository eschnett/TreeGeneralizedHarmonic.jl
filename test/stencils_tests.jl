# What `src/stencils.jl` claims: the weights are the textbook ones, exactly;
# each operator is exact to the degree it is built for and no further; the
# dissipation damps rather than drives; and the rounding into `T` happens
# once.
#
# The failure these tests exist to catch is a *plausible* stencil. A weight
# vector with a sign flipped, a denominator off by one, or an order built in
# `Float64` and converted down still differentiates, still converges, and
# still produces pictures — at a rate one lower than the scheme claims,
# which is the number three later milestones are measured by. So almost
# everything below is asserted in `Rational`, where "exact" is a statement a
# test can make without a tolerance: the tables entry for entry, the
# polynomial degrees each operator does and does not annihilate, the
# dissipation's Nyquist eigenvalue. The floating-point testsets then measure
# the two things exact arithmetic cannot say — the observed order on smooth
# data, and that the conversion into `T` rounds once.
#
# Nothing here evaluates a background. Step 1's files pay this suite's whole
# compilation cost in `SpacetimeMetrics`' nested dual passes; the stencils
# are weights and polynomials, and this file stays cheap.

using KernelAbstractions: CPU, @Const, @index, @kernel, synchronize
using LinearAlgebra: Symmetric, eigen
using MultiFloats: Float32x2
using Random: MersenneTwister
using StaticArrays: SVector
using Test
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: lagrange_derivative_weights,
                               rational_derivative_weights,
                               rational_dissipation_weights

# `CODE.md`, "Finite-difference stencils": the orders the kernel is ever
# instantiated at, and the dissipation rank `r = q/2 + 1` that goes with
# each.
const STENCIL_ORDERS = (2, 4, 6, 8)
const STENCIL_RANKS = (2, 3, 4, 5)

# The textbook central-difference tables, written out as the exact rationals
# they are. These are the numbers `src/stencils.jl` must reproduce; they are
# spelled here rather than derived so that the test is a second source and
# not the same construction run twice.
const FD1_TABLE = (
    2 => [-1//2, 0//1, 1//2],
    4 => [1//12, -2//3, 0//1, 2//3, -1//12],
    6 => [-1//60, 3//20, -3//4, 0//1, 3//4, -3//20, 1//60],
    8 => [1//280, -4//105, 1//5, -4//5, 0//1, 4//5, -1//5, 4//105, -1//280])
const FD2_TABLE = (
    2 => [1//1, -2//1, 1//1],
    4 => [-1//12, 4//3, -5//2, 4//3, -1//12],
    6 => [1//90, -3//20, 3//2, -49//18, 3//2, -3//20, 1//90],
    8 => [-1//560, 8//315, -1//5, 8//5, -205//72, 8//5, -1//5, 8//315,
          -1//560])
# `(−1)^{r+1} (Δ_+Δ_−)^r / 2^{2r}`: the binomial row of `2r`, alternating,
# with the outer sign that puts the negative at the center.
const KO_TABLE = (
    2 => [-1, 4, -6, 4, -1] .// 16,
    3 => [1, -6, 15, -20, 15, -6, 1] .// 64,
    4 => [-1, 8, -28, 56, -70, 56, -28, 8, -1] .// 256,
    5 => [1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1] .// 1024)

lookup(table, key) = table[findfirst(p -> first(p) == key, table)].second

# The exactness claims are made in `Rational{BigInt}`, not `Rational{Int}`:
# a monomial of degree `q + 2` at an off-grid rational point overflows a
# 64-bit denominator at `q = 6` already, and `Rational` arithmetic is
# checked, so the symptom is an `OverflowError` in the test rather than a
# wrong answer. Same reasoning as `src/stencils.jl`'s `StencilRational`.
const RQ = Rational{BigInt}

# A monomial and its `m`-th derivative, in whatever arithmetic they are
# handed — `Rational` for the exactness claims, `Float64` for the rates.
monomial(d) = x -> x^d
monomial_derivative(d, m) =
    x -> d < m ? zero(x) : prod((d - m + 1):d) * x^(d - m)

# `(−1)^j` for a possibly negative offset `j`, which `Base` refuses to raise
# an `Int` to.
alternating(j) = isodd(j) ? -1 : 1

# The message an `ArgumentError` carries, for the testset that asserts the
# errors say *why* (`CLAUDE.md`, "Conventions").
errmsg(f) = try
    f()
    ""
catch err
    sprint(showerror, err)
end

# The observed order of a stencil on smooth data: the ratio of the errors of
# the finest pair of spacings at which the truncation error is still a
# thousand times the roundoff floor `~eps/h^m`. Choosing the window by that
# rule rather than by hand is what makes the measured rates below reproduce
# on a machine whose `sin` differs in the last place — at `q = 8` the
# floating-point error of the contraction overtakes the truncation error
# before `h = 1/32`, and a fixed window would measure the floor instead.
function stencil_rate(w, f, exact, x, m; scale=1.0)
    hs = [1 / 2.0^k for k in 1:7]
    errs = [abs(apply_stencil(w, f, x, h) / h^m - exact) for h in hs]
    floors = [4 * eps(Float64) * scale / h^m for h in hs]
    j = findlast(k -> errs[k] >= 1000 * floors[k], eachindex(hs))
    (j isa Integer && j >= 2) ||
        error("no resolution pair is free of roundoff: errs=$errs")
    return log2(errs[j - 1] / errs[j])
end

# The same, for the tensor-product mixed derivative, whose roundoff floor is
# `~eps/(hx·hy)` because both sums divide by a spacing.
function mixed_rate(w, f, exact, x, y; scale=1.0)
    hs = [1 / 2.0^k for k in 1:6]
    errs = [abs(apply_mixed_stencil(w, f, x, y, h, h) / h^2 - exact)
            for h in hs]
    floors = [4 * eps(Float64) * scale / h^2 for h in hs]
    j = findlast(k -> errs[k] >= 1000 * floors[k], eachindex(hs))
    (j isa Integer && j >= 2) ||
        error("no resolution pair is free of roundoff: errs=$errs")
    return log2(errs[j - 1] / errs[j])
end

# ---------------------------------------------------------------------------

@testset "The weights are the textbook tables, entry for entry: q=$q" for q in
                                                                         STENCIL_ORDERS
    # Guards the quiet failure: a stencil with one wrong entry is still a
    # stencil. Compared as exact rationals against a table written out
    # independently of the construction, because at `Float64` a wrong entry
    # in the last place and a correctly rounded one are the same assertion.
    @test rational_derivative_weights(q, 1) == lookup(FD1_TABLE, q)
    @test rational_derivative_weights(q, 2) == lookup(FD2_TABLE, q)

    # Centered means symmetric: the first derivative is odd, the second
    # even, and a stencil that lost that property would advect.
    w1 = rational_derivative_weights(q, 1)
    w2 = rational_derivative_weights(q, 2)
    @test w1 == -reverse(w1)
    @test w2 == reverse(w2)
    @test w1[q ÷ 2 + 1] == 0                    # the center weight of ∂
    @test sum(w1) == 0                          # both annihilate constants
    @test sum(w2) == 0
    @test length(w1) == q + 1 && length(w2) == q + 1
end

@testset "The dissipation weights are the binomial row, signed: r=$r" for r in
                                                                         STENCIL_RANKS
    # Guards `CODE.md`'s formula entry for entry, including the outer sign
    # `(−1)^{r+1}` — the one factor whose only visible effect is whether the
    # term damps or drives.
    @test rational_dissipation_weights(r) == lookup(KO_TABLE, r)
    @test length(rational_dissipation_weights(r)) == 2r + 1
end

@testset "Each derivative is exact to its degree and not one further: q=$q" for q in
                                                                               STENCIL_ORDERS
    # `CODE.md`, "Finite-difference stencils": the first derivative is exact
    # on polynomials of degree ≤ q, the second — one better than it was
    # built for, by the symmetry of an even q — on degree ≤ q + 1. The
    # "and not one further" half is the one that catches a stencil that is
    # accidentally *wider* than its order, which is how a weight vector
    # padded with zeros or an off-by-one in the node list shows up.
    #
    # In `Rational`, at an off-grid center and a spacing that is not a
    # power of two, so that nothing passes by cancellation of round numbers.
    x0, h = RQ(3//7), RQ(2//5)
    for m in (1, 2)
        w = SVector{q + 1}(rational_derivative_weights(q, m)...)
        exact_to = m == 1 ? q : q + 1
        for d in 0:exact_to
            @test apply_stencil(w, monomial(d), x0, h) // h^m ==
                  monomial_derivative(d, m)(x0)
        end
        d = exact_to + 1
        @test apply_stencil(w, monomial(d), x0, h) // h^m !=
              monomial_derivative(d, m)(x0)
    end
end

@testset "The dissipation annihilates smooth data and damps Nyquist: r=$r" for r in
                                                                              STENCIL_RANKS
    # The two halves of `CODE.md`'s claim, both exact.
    #
    # It annihilates polynomials of degree < 2r and not 2r: that is what
    # makes the term `O(h^{2r−1}) = O(h^{q+1})` and therefore invisible to a
    # q-th order scheme, and a dissipation that touched smooth data would
    # show up as a convergence rate of 2r − 1 instead of q.
    w = SVector{2r + 1}(rational_dissipation_weights(r)...)
    x0, h = RQ(3//7), RQ(2//5)
    for d in 0:(2r - 1)
        @test apply_stencil(w, monomial(d), x0, h) == 0
    end
    @test apply_stencil(w, monomial(2r), x0, h) != 0

    # And the grid-scale mode `u_j = (−1)^j` is an eigenvector with
    # eigenvalue exactly `−1`, so that `Q = (ε/h)·(this)` damps Nyquist at
    # exactly `ε/h`. Both halves matter: the sign is the difference between
    # dissipation and an amplifier of exactly the noise it is there to
    # remove, and the magnitude is the `2^{−2r}` normalisation that keeps
    # `ε ∈ (0, 1)` neutral to the CFL condition.
    @test sum(w) == 0
    @test sum(w[k] * alternating(k - 1 - r) for k in 1:(2r + 1)) == -1
    @test w[r + 1] < 0                                    # the center weight
    @test w == reverse(w)
end

@testset "The dissipation damps every mode, not only Nyquist: r=$r" for r in
                                                                       STENCIL_RANKS
    # The statement "damping" in full: on a periodic grid the operator is
    # negative semidefinite, `⟨u, Qu⟩ ≤ 0` for every `u`. The Nyquist
    # eigenvalue above fixes the worst mode; this fixes the sign of all of
    # them at once, and it is the property an energy estimate for the
    # evolved system would use.
    w = dissipation_weights(Float64, Val(r))
    n = 32
    rng = MersenneTwister(1234 + r)
    for trial in 1:8
        u = randn(rng, n)
        qu = [sum(w[k] * u[mod1(i + k - 1 - r, n)] for k in 1:(2r + 1))
              for i in 1:n]
        @test sum(u .* qu) < 0
    end

    # And the Fourier symbol is the one the docstring states, mode by mode:
    # `−sin^{2r}(kh/2)`, real, negative, and ≤ 1 in magnitude. Absolute
    # tolerance, not relative: at `kh ≪ 1` the contraction is a difference
    # of `O(1)` samples giving `O((kh)^{2r})`, so what a correct stencil
    # delivers there is roundoff-limited by construction.
    x0, h = 0.123, 0.01
    for kh in (π, π / 2, π / 4, 2π / 8)
        k = kh / h
        got = apply_stencil(w, x -> sin(k * x), x0, h)
        want = -sin(kh / 2)^(2r) * sin(k * x0)
        @test isapprox(got, want; atol=32 * eps(Float64))
        @test abs(want) <= 1
    end
end

@testset "The Nyquist mode moves outward under the advection stencil: q=$q" for q in
                                                                              STENCIL_ORDERS
    # Guards the premise of `CODE.md`'s leakage margin (added in step 8a,
    # `test/dispersion.jl`): inside a horizon the shift advection makes every
    # continuum mode ingoing, but every centered first derivative annihilates
    # the grid-scale mode and has the opposite slope there, so the Nyquist
    # mode is carried *outward*. A change to the first-derivative weights
    # moves the number below, and with it the margin the interior needs.
    #
    # The sign convention is the code's: `∂_t h = +β^i ∂_i h + …` (`CODE.md`,
    # "The equations"). On `u_j = e^{i(jθ − ωt)}`, `θ = kh`, the stencil's
    # symbol is `i s(θ)/h` with `s(θ) = Σ_j w_j sin(jθ)`, so the advection
    # term alone gives `ω = −β s(θ)/h` and the group velocity
    # `v_g = dω/dk = −β s′(θ)`, `s′(θ) = Σ_j j w_j cos(jθ)`. In the continuum
    # `s′ = 1` and `v_g = −β`: inward, for Kerr-Schild's outward-pointing
    # shift. At Nyquist `v_g = −β s′(π)`, and `s′(π) = Σ_j j w_j (−1)^j` is
    # an exact rational number.
    w = rational_derivative_weights(q, 1)
    r = q ÷ 2
    js = (-r):r
    sgn = [alternating(j) for j in js]
    # The stencil annihilates the grid-scale mode `u_j = (−1)^j`: advection
    # does not act on it at all.
    @test sum(w .* sgn) == 0
    # The continuum slope, `s′(0) = 1`: first-derivative consistency.
    @test sum(j * w[j + r + 1] for j in js) == 1
    # The Nyquist slope, exactly, and the group velocity it gives, `−β s′(π)`,
    # in units of `β`: `+1` at `q = 2`, `+5/3` at `q = 4` — outward, and
    # faster the higher the order.
    slope = sum(j * w[j + r + 1] * alternating(j) for j in js)
    @test slope == lookup((2 => -1 // 1, 4 => -5 // 3, 6 => -11 // 5,
                           8 => -93 // 35), q)
    @test -slope > 0
    q > 2 && @test -slope >
                   -sum(j * rational_derivative_weights(q - 2, 1)[j + r] *
                        alternating(j) for j in (1 - r):(r - 1))
end

@testset "The mixed derivative is the product of two first-derivative vectors: q=$q" for q in
                                                                                        STENCIL_ORDERS
    # `CODE.md`, "Finite-difference stencils": `∂_i∂_j` for `i ≠ j` has no
    # weights of its own. Exactness is the tensor-product statement — degree
    # ≤ q in *each* variable — and it is what reads the edge and corner
    # ghosts TreeAMR fills unconditionally. A kernel that instead formed the
    # mixed derivative from a compact cross stencil would pass a
    # one-dimensional test and fail this one.
    w = SVector{q + 1}(rational_derivative_weights(q, 1)...)
    x0, y0, hx, hy = RQ(3//7), RQ(-2//9), RQ(2//5), RQ(3//4)
    for a in 0:q, b in 0:q
        f = (x, y) -> x^a * y^b
        got = apply_mixed_stencil(w, f, x0, y0, hx, hy) // (hx * hy)
        want = monomial_derivative(a, 1)(x0) * monomial_derivative(b, 1)(y0)
        @test got == want
    end
    # One degree further in either variable and it is no longer exact.
    @test apply_mixed_stencil(w, (x, y) -> x^(q + 1) * y, x0, y0, hx, hy) //
          (hx * hy) != monomial_derivative(q + 1, 1)(x0)
    @test apply_mixed_stencil(w, (x, y) -> x * y^(q + 1), x0, y0, hx, hy) //
          (hx * hy) != monomial_derivative(q + 1, 1)(y0)

    # The two ends give the same operator in exact arithmetic, which is what
    # lets the docstring name one of them without changing the scheme.
    f = (x, y) -> (2x - 3y)^q + x^2 * y^3
    @test apply_mixed_stencil(w, f, x0, y0, hx, hy) ==
          apply_mixed_stencil(w, (y, x) -> f(x, y), y0, x0, hy, hx)
end

@testset "The rounding into T happens once: q=$q" for q in STENCIL_ORDERS
    # `CODE.md`: the weights are built in exact rational arithmetic and
    # rounded once, so that a stencil is the same object at every precision.
    # The claim is testable because every entry is a ratio of small integers,
    # each exactly representable: the correctly rounded value of `p//d` and
    # the quotient `p/d` computed in an IEEE `T` are the same bits, so `===`
    # is exactly the assertion that nothing was accumulated in floating
    # point on the way. Weights built in `Float64` and converted down — the
    # obvious implementation — differ from these in the last places, and
    # would put a floor under every convergence rate measured at a type
    # wider than the one they were built in.
    #
    # `Float32x2` is here for a second reason, and it is the one that found
    # a bug: the conversion into `T` must happen at the *call site*, because
    # a `@generated` method's generator may only call methods that existed
    # when it was defined, and this package is precompiled long before a
    # driver loads MultiFloats. The `Float32x2` assertions below are what a
    # generator that converted on its own side fails on, with `MethodError:
    # ... The applicable method may be too new` — and they fail only when
    # MultiFloats is loaded *after* `TreeGeneralizedHarmonic`, which is what
    # `runtests.jl` does and what a driver would do.
    for m in (1, 2)
        rs = rational_derivative_weights(q, m)
        for T in (Float64, Float32, Float32x2)
            wT = derivative_weights(T, Val(q), Val(m))
            @test eltype(wT) === T
            @test all(wT[k] == T(numerator(rs[k])) / T(denominator(rs[k]))
                      for k in eachindex(rs))
        end

        # At an IEEE type the division of two exactly representable small
        # integers is correctly rounded, so "rounded once" is the *same
        # bits* as converting the exact rational — `===`, which is the
        # assertion that nothing was computed in floating point on the way.
        for T in (Float64, Float32)
            wT = derivative_weights(T, Val(q), Val(m))
            @test all(wT[k] === T(rs[k]) for k in eachindex(rs))
        end

        # At `Float32x2` the division is the type's own, not necessarily
        # correctly rounded to the last place, so the claim is one ulp
        # rather than identity — measured, at the type the suite keeps in
        # order to prove that nothing depends on a hardware float.
        w2 = derivative_weights(Float32x2, Val(q), Val(m))
        @test all(isapprox(w2[k], Float32x2(rs[k]); rtol=8 * eps(Float32x2))
                  for k in eachindex(rs))
    end

    # The same for the dissipation, whose entries are dyadic and therefore
    # exact in every one of these types — identity at all three.
    r = q ÷ 2 + 1
    rs = rational_dissipation_weights(r)
    for T in (Float64, Float32, Float32x2)
        wT = dissipation_weights(T, Val(r))
        @test all(wT[k] == T(rs[k]) for k in eachindex(rs))
        @test all(wT[k] == T(numerator(rs[k])) / T(denominator(rs[k]))
                  for k in eachindex(rs))
    end
end

@testset "The weights are isbits, inferred, and free at run time" begin
    # Guards what makes them usable from the fused kernel of step 3: they
    # are `@generated`, so the `Rational{BigInt}` construction happens while
    # the method compiles and the emitted code holds only `Int` literals and
    # one division. A weight vector that was a `Vector`, or that carried a
    # `BigInt`, would work on the host, allocate per point, and fail to
    # compile for a device.
    for q in STENCIL_ORDERS, m in (1, 2)
        w = @inferred derivative_weights(Float64, Val(q), Val(m))
        @test w isa SVector{q + 1,Float64}
        @test isbits(w)
    end
    for r in STENCIL_RANKS
        w = @inferred dissipation_weights(Float64, Val(r))
        @test w isa SVector{2r + 1,Float64}
        @test isbits(w)
    end

    # No allocation, at the call site a kernel would use.
    call_d() = derivative_weights(Float64, Val(4), Val(2))
    call_k() = dissipation_weights(Float64, Val(3))
    for f in (call_d, call_k)
        f()                                      # compile before measuring
        @test @allocated(f()) == 0
    end

    # `T` defaults to `Float64`, as every driver in this package does.
    @test derivative_weights(Val(4), Val(1)) ===
          derivative_weights(Float64, Val(4), Val(1))
    @test dissipation_weights(Val(3)) === dissipation_weights(Float64, Val(3))

    # And the rank relation `2r = q + 2` has one spelling. It is also the
    # ghost width `G` of `CODE.md`, "Field sets and layout", because the
    # dissipation is the widest stencil an evaluation takes.
    for (q, r) in zip(STENCIL_ORDERS, STENCIL_RANKS)
        @test dissipation_rank(Val(q)) === Val(r)
        @test length(dissipation_weights(dissipation_rank(Val(q)))) == 2r + 1
        @test r == q ÷ 2 + 1
    end
end

@testset "The operators converge at their order on smooth data: q=$q" for q in
                                                                         STENCIL_ORDERS
    # The end-to-end claim the exact tests cannot make: on a function that
    # is not a polynomial, the error falls as `h^q`. The numbers are
    # recorded in `CODE.md`, "Measured results", G1b.
    f, x0 = x -> exp(sin(x)), 0.37
    df = cos(x0) * exp(sin(x0))
    d2f = (cos(x0)^2 - sin(x0)) * exp(sin(x0))
    for (m, exact) in ((1, df), (2, d2f))
        w = derivative_weights(Float64, Val(q), Val(m))
        @test stencil_rate(w, f, exact, x0, m) ≈ q atol = 0.3
    end

    # The dissipation is `O(h^{2r−1}) = O(h^{q+1})` on smooth data — one
    # order better than the scheme, which is why `CODE.md` can add it
    # without tightening the interface-order rule. Asserted as a lower
    # bound: it is exactly `q + 1` where the window is clean (`q = 2, 4`)
    # and drifts above it at `q = 6, 8`, where the roundoff floor leaves
    # only a coarse, pre-asymptotic pair.
    r = q ÷ 2 + 1
    wk = dissipation_weights(Float64, Val(r))
    @test stencil_rate(wk, f, 0.0, x0, 1) >= q + 0.5

    # Mixed, in two variables, at the same order.
    g = (x, y) -> exp(sin(x)) * cos(y / 2)
    y0 = 0.5
    w1 = derivative_weights(Float64, Val(q), Val(1))
    exact_xy = cos(x0) * exp(sin(x0)) * (-sin(y0 / 2) / 2)
    @test mixed_rate(w1, g, exact_xy, x0, y0) ≈ q atol = 0.3
end

@testset "A shifted stencil keeps its order" begin
    # `lagrange_derivative_weights` is exact for *any* distinct nodes, which
    # is the property a one-sided or interface-shifted stencil would rest
    # on. Nothing in the proof of concept uses one — every stencil here is
    # centered and every coarse-fine face is TreeAMR's business — so this
    # guards the docstring rather than a caller.
    nodes = [Rational{BigInt}(j) for j in -1:4]           # six nodes, shifted
    for m in (1, 2)
        w = SVector{6}(lagrange_derivative_weights(nodes, m)...)
        for d in 0:5
            got = sum(w[k] * (nodes[k] * (3//5) + 2//7)^d for k in 1:6)
            # Samples at `x0 + node·h` with `h = 3//5`, `x0 = 2//7`.
            @test got // (3//5)^m == monomial_derivative(d, m)(2//7)
        end
    end
end

# The weights are asked for *inside* the kernel, from `Val` parameters, which
# is how step 3's fused right-hand side will reach them.
@kernel function stencil_kernel!(out, @Const(u), vq::Val, vm::Val, off)
    i = @index(Global)
    T = eltype(out)
    w = derivative_weights(T, vq, vm)
    out[i] = apply_stencil(w, u, i + off)
end

@kernel function dissipation_kernel!(out, @Const(u), vr::Val, off)
    i = @index(Global)
    T = eltype(out)
    w = dissipation_weights(T, vr)
    out[i] = apply_stencil(w, u, i + off)
end

@testset "The stencils run as a kernel on CPU(): T=$T" for T in (Float64,
                                                                Float32)
    # Guards the requirement the weights exist under: step 3's right-hand
    # side is one fused KernelAbstractions kernel per owned point, and it
    # asks for the weights *inside* it with `q` a `Val` parameter. A
    # construction that allocated, or that left a `Rational` in the emitted
    # code, would still pass every testset above and fail here — or, worse,
    # compile on the host and not for a device. Bit-for-bit against a host
    # loop, not a tolerance: the same weights and the same summation order
    # are the same arithmetic, and anything else would put a floor under
    # every error this package measures.
    n = 64
    rng = MersenneTwister(20260917)
    u = T.(randn(rng, n))
    for q in STENCIL_ORDERS, m in (1, 2)
        g = q ÷ 2
        out = fill(T(NaN), n - 2g)
        stencil_kernel!(CPU(), 8)(out, u, Val(q), Val(m), g; ndrange=n - 2g)
        synchronize(CPU())
        w = derivative_weights(T, Val(q), Val(m))
        ref = [apply_stencil(w, u, i + g) for i in 1:(n - 2g)]
        @test isequal(out, ref)
        @test eltype(out) === T
    end
    for r in STENCIL_RANKS
        out = fill(T(NaN), n - 2r)
        dissipation_kernel!(CPU(), 8)(out, u, Val(r), r; ndrange=n - 2r)
        synchronize(CPU())
        w = dissipation_weights(T, Val(r))
        ref = [apply_stencil(w, u, i + r) for i in 1:(n - 2r)]
        @test isequal(out, ref)
    end
end

@testset "A stencil that does not exist is refused, with the reason" begin
    # `ArgumentError`s say why, not just what (`CLAUDE.md`, "Conventions").
    # An odd `q` is the one a caller reaches by arithmetic — `q = p − 2`
    # with an odd prolongation order, say — and a centered stencil has no
    # meaning there; `m = 3` is the one a caller reaches by generalising.
    # Both are refused where the weights are built *and* through the
    # `@generated` front door, which is a separate code path.
    @test_throws ArgumentError rational_derivative_weights(3, 1)
    @test_throws ArgumentError rational_derivative_weights(0, 1)
    @test_throws ArgumentError rational_derivative_weights(4, 3)
    @test_throws ArgumentError rational_derivative_weights(4, 0)
    @test_throws ArgumentError rational_dissipation_weights(0)
    @test_throws ArgumentError derivative_weights(Float64, Val(3), Val(1))
    @test_throws ArgumentError derivative_weights(Float64, Val(4), Val(3))
    @test_throws ArgumentError dissipation_weights(Float64, Val(0))

    @test occursin("even", errmsg(() -> rational_derivative_weights(3, 1)))
    @test occursin("twice", errmsg(() -> rational_derivative_weights(4, 3)))
    @test occursin("2r = q + 2", errmsg(() -> rational_dissipation_weights(0)))
end

# ---------------------------------------------------------------------------
# The closures at an excision surface (added in step X1)
# ---------------------------------------------------------------------------
#
# `CODE.md`, "Excision": at an evolved point with `k⁻, k⁺ ∈ 0…G` consecutive
# non-excised points on each side of an axis, every stencil that would read
# past them is replaced by a closure on the points it may read. The claims
# are exact, in `Rational`, as above: where each closure reads, what it is
# exact on, that the full-width code is the centered one, that the
# dissipation's closure keeps its sign, and that the table a kernel will
# carry rounds once. No kernel uses them yet (step X2b's zone kernel will).
# Each claim is one assertion per `(q, k⁻, k⁺)` — the cases are many and
# cheap, and a failure names the case in its testset's loop variables.

using TreeGeneralizedHarmonic: DISSIPATION_CLOSURES, closure_admissible,
                               closure_derivative_weights,
                               closure_dissipation_weights,
                               closure_exact_degree, closure_nodes,
                               closure_table, lopsided_weights

# The ghost width, which is the cap on `k⁻, k⁺` and on every closure's reach.
ghost(q) = q ÷ 2 + 1

# Every `(k⁻, k⁺)` a class can hold, and whether it has a `∂²` closure at all.
closure_cases(q) = [(km, kp) for km in 0:ghost(q) for kp in 0:ghost(q)]
has_d2(q, km, kp) = length(closure_nodes(q, km, kp)) >= 3

# A closure's contraction against a function sampled at `x0 + j h`.
apply_closure(nodes, w, f, x0, h) = sum(w[k] * f(x0 + j * h)
                                        for (k, j) in enumerate(nodes))

# Whether `nodes, w` is exact for `∂^m` on every monomial of degree `≤ deg`
# and not on degree `deg + 1`, in `Rational` at an off-grid center and a
# spacing that is not a power of two, so that nothing passes by the
# cancellation of round numbers.
function exact_to(nodes, w, m, deg)
    x0, h = RQ(3//7), RQ(2//5)
    ok = all(apply_closure(nodes, w, monomial(d), x0, h) // h^m ==
             monomial_derivative(d, m)(x0) for d in 0:deg)
    return ok && apply_closure(nodes, w, monomial(deg + 1), x0, h) // h^m !=
                 monomial_derivative(deg + 1, m)(x0)
end

@testset "A closure reads only what it may, and is centered where that fits: q=$q" for q in
                                                                                     STENCIL_ORDERS
    # Guards the two halves of `CODE.md`'s definition. A closure that read
    # one point past `k⁻` would read an excised value — `0 · NaN = NaN` at
    # best, a stale core at worst — and one past `G` would need a halo the
    # mesh does not have; and a closure that differed from the centered
    # stencil where the centered one fits would change the exterior's
    # operator, which `PLAN.md`'s sharp edges require to be unchanged bit for
    # bit. Not centered, it takes *every* point it may read — the most
    # accurate stencil there, and the one an extrapolation of the excised taps
    # followed by the centered stencil gives (`CODE.md`, "Excision").
    G = ghost(q)
    r = q ÷ 2
    inside(ns, km, kp) = first(ns) >= -min(km, G) && last(ns) <= min(kp, G)
    @test all(inside(closure_nodes(q, km, kp), km, kp)
              for (km, kp) in closure_cases(q))
    for (km, kp) in closure_cases(q), m in (1, 2)
        has_d2(q, km, kp) || continue
        ns, w = closure_derivative_weights(q, m, km, kp)
        want = min(km, kp) >= r ? ((-r):r) : ((-min(km, G)):min(kp, G))
        @test ns == closure_nodes(q, km, kp) == want && length(w) == length(ns)
        min(km, kp) >= r && @test w == rational_derivative_weights(q, m)
        # A mirrored point has the mirrored closure: `∂` odd, `∂²` even.
        ms, wm = closure_derivative_weights(q, m, kp, km)
        @test ms == (-last(ns)):(-first(ns)) &&
              wm == (isodd(m) ? -1 : 1) .* reverse(w)
    end
    # The starting family's orders at the first evolved point, `CODE.md`:
    # `q/2 + 1` for `∂` and `q/2` for `∂²` — `(2, 1)` at `q = 2`, `(3, 2)` at
    # `q = 4` — and the centered `q` from `k = q/2` on.
    @test closure_exact_degree(q, 1, 0, G) == q ÷ 2 + 1          # order q/2 + 1
    @test closure_exact_degree(q, 2, 0, G) - 1 == q ÷ 2          # order q/2
    @test closure_exact_degree(q, 1, r, G) == q
    @test closure_exact_degree(q, 2, r, G) - 1 == q
end

@testset "Every closure is exact to its degree and not one further: q=$q" for q in
                                                                             STENCIL_ORDERS
    # Guards the claim `CODE.md` makes of every closure, and the one a
    # convergence measurement near the surface would be read against: a
    # closure is the Lagrange derivative on its nodes, exact to
    # `closure_exact_degree` and *not* exact one degree further — which is
    # what catches a node list off by one or a weight vector padded with a
    # point it does not use.
    for (km, kp) in closure_cases(q), m in (1, 2)
        has_d2(q, km, kp) || continue
        @test exact_to(closure_derivative_weights(q, m, km, kp)..., m,
                       closure_exact_degree(q, m, km, kp))
    end
    # A wider reach than `G` — only step X1's models ask for one — is still
    # the Lagrange derivative on every node it may read.
    G = ghost(q)
    nodes, w = closure_derivative_weights(q, 2, 0, G + 2; reach=G + 2)
    @test nodes == 0:(G + 2)
    @test exact_to(nodes, w, 2, G + 2)
end

# The Mattsson–Svärd–Nordström operator built *independently* of
# `closure_dissipation_weights`: on a line of `n` points with the excised
# ones marked, `−2^{−2r} Dᵀ B D` with `D` the `r`-th forward difference over
# every window `[k, k + r]` and `B` its indicator of a window of evolved
# points. And the same operator assembled row by row from the closures,
# with `k⁻, k⁺` counted on the line and capped at `G`.
function msn_global(r, excised)
    n = length(excised)
    Q = zeros(RQ, n, n)
    d = [RQ((iseven(r - j) ? 1 : -1) * binomial(r, j)) for j in 0:r]
    for k in 1:(n - r)
        all(.!excised[k:(k + r)]) || continue
        Q[k:(k + r), k:(k + r)] .-= (d * d') ./ RQ(2)^(2r)
    end
    return Q
end

function closure_line_operator(q, kind, excised)
    n = length(excised)
    G = ghost(q)
    Q = zeros(RQ, n, n)
    for i in 1:n
        excised[i] && continue
        km = 0
        while km < G && i - km - 1 >= 1 && !excised[i - km - 1]
            km += 1
        end
        kp = 0
        while kp < G && i + kp + 1 <= n && !excised[i + kp + 1]
            kp += 1
        end
        nodes, w = closure_dissipation_weights(q, kind, km, kp)
        for (j, wj) in zip(nodes, w)
            Q[i, i + j] += wj
        end
    end
    return Q
end

@testset "The dissipation's closures are the centered operator where it fits: q=$q" for q in
                                                                                       STENCIL_ORDERS
    # Guards the dissipation's half of "the exterior's operator is unchanged
    # bit for bit", and its reach: every kind is the centered operator of
    # rank `G` where `min(k⁻, k⁺) ≥ G`, reads only `[−k⁻, k⁺] ∩ [−G, G]`
    # elsewhere, annihilates constants (a dissipation that touched them
    # would be a source), and does not drive the Nyquist mode at the point.
    # A closure switches off only where it has no room: the reduced rank at
    # the first evolved point of either side, the other two only where both
    # sides are short — which no admissible point is.
    G = ghost(q)
    for kind in DISSIPATION_CLOSURES, (km, kp) in closure_cases(q)
        nodes, w = closure_dissipation_weights(q, kind, km, kp)
        ny = sum(w[k] * alternating(j) for (k, j) in enumerate(nodes))
        @test first(nodes) >= -min(km, G) && last(nodes) <= min(kp, G) &&
              sum(w) == 0 && ny <= 0 &&
              (!iszero(ny) || (kind === :reduced && min(km, kp) == 0) ||
               !closure_admissible(q, km, kp))
        min(km, kp) >= G &&
            @test nodes == (-G):G && w == rational_dissipation_weights(G)
    end
    # The Mattsson–Svärd–Nordström closure annihilates polynomials of degree
    # below `G` — the forward difference's own — so it is `O(h^{G−1})` near
    # the surface and the centered `O(h^{2G−1})` away from it.
    x0, h = RQ(3//7), RQ(2//5)
    for (km, kp) in closure_cases(q)
        nodes, w = closure_dissipation_weights(q, :msn, km, kp)
        all(iszero, w) && continue
        @test all(apply_closure(nodes, w, monomial(d), x0, h) == 0
                  for d in 0:(G - 1))
    end
end

@testset "The MSN closure damps in l², and the other two do not: q=$q" for q in
                                                                         STENCIL_ORDERS
    # Guards the choice `CODE.md` records for step X2b (proposed in step X1):
    # the dissipation's closure has to keep the sign that makes it
    # dissipation, and the norm that sign is stated in is the discrete
    # `l²`, the one in which the centered operator is negative
    # semidefinite. Mattsson, Svärd and Nordström's form is `−2^{−2r} DᵀBD`,
    # so it is symmetric and negative semidefinite on *any* pattern of
    # excised points — here a line with a gap, a one-point sliver, a
    # three-point one and both ends, so that every `(k⁻, k⁺)` occurs — and
    # the closures assembled row by row are exactly that matrix.
    G = ghost(q)
    excised = falses(48)
    excised[1:2] .= true
    excised[14:17] .= true
    excised[22] = true
    excised[27:29] .= true
    excised[47:48] .= true
    keep = .!excised
    Q = closure_line_operator(q, :msn, excised)
    @test Q == msn_global(G, excised)
    @test Q[keep, keep] == transpose(Q[keep, keep])
    rng = MersenneTwister(20261005 + q)
    xs = [[RQ(rand(rng, -64:64), rand(rng, 1:16)) for _ in 1:48] .* keep
          for _ in 1:16]
    @test all(x' * Q * x <= 0 for x in xs)

    # The reduced rank and the one-sided closure do not drive the Nyquist
    # mode at any point (above), but neither is negative semidefinite in
    # `l²`: a witness, exact, on the same line. That is why step X1 chose
    # the third — a closure that is not damping in some norm has no energy
    # estimate behind it.
    for kind in (:reduced, :onesided)
        Qk = closure_line_operator(q, kind, excised)
        S = Float64.(Qk[keep, keep] + transpose(Qk[keep, keep])) ./ 2
        λ, V = eigen(Symmetric(S))
        x = zeros(RQ, 48)
        x[keep] .= [rationalize(BigInt, v; tol=1e-9) for v in V[:, end]]
        @test λ[end] > 0 && x' * Qk * x > 0
    end
end

@testset "The lopsided advection is order q, reads upwind, and damps Nyquist: q=$q" for q in
                                                                                      STENCIL_ORDERS
    # Guards the candidate cure `CODE.md` names for the grid-scale leakage:
    # the lopsided first derivative is the order-`q` stencil on `q + 1`
    # nodes shifted one point to the upwind side — exact to degree `q`, not
    # `q + 1`, reaching `G` upwind and `q/2 − 1` downwind — and, unlike every
    # centered one, it acts on the Nyquist mode, with the sign that *damps*
    # it under the code's advection `∂_t u = +β ∂u` with `β` pointing
    # upwind.
    G = ghost(q)
    for up in (-1, 1)
        nodes, w = lopsided_weights(q, up, G, G)
        @test nodes == (up > 0 ? ((1 - q ÷ 2):G) : ((-G):(q ÷ 2 - 1)))
        @test exact_to(nodes, w, 1, q)
        # `β` has the sign of `up`: `β · Σ_j w_j (−1)^j < 0` is a decaying
        # Nyquist mode. And the whole symbol's real part has that sign, `β Re
        # D(θ) ≤ 0` at every phase: dissipative everywhere, not only at
        # Nyquist.
        @test up * sum(w[k] * alternating(j) for (k, j) in enumerate(nodes)) < 0
        wf = Float64.(w)
        @test all(up * sum(wf[k] * cos(j * θ) for (k, j) in enumerate(nodes)) <=
                  8 * eps(Float64) for θ in range(0, π; length=65))
        # Near the surface on the downwind side it reads only what it may,
        # and where the upwind side is short it is the closure, unlopsided.
        for k in 0:G
            km, kp = up > 0 ? (k, G) : (G, k)
            ns, _ = lopsided_weights(q, up, km, kp)
            @test first(ns) >= -min(km, G) && last(ns) <= min(kp, G)
            km2, kp2 = up > 0 ? (G, k) : (k, G)
            (k < G && has_d2(q, km2, kp2)) || continue
            @test lopsided_weights(q, up, km2, kp2) ==
                  closure_derivative_weights(q, 1, km2, kp2)
        end
    end
end

@testset "The closure table rounds once, and its centered rows are the kernel's: q=$q" for q in
                                                                                         STENCIL_ORDERS
    # Guards what step X2b's zone kernel will carry: one table per order,
    # `isbits`, every entry the exact rational rounded once — so that its
    # centered rows are the very weights the main kernel uses, `===` at
    # `Float64` and `Float32`, and a zone point with no excised tap computes
    # what the centered code computes — with the nodes each closure reads, so
    # that a contraction never touches an excised value. (It is a plain
    # function, not `@generated`, so it converts in the caller's world at any
    # type: `Float32x2` builds too — checked by hand in step X1 and left out
    # here, where its compilation would cost eighteen seconds.)
    G = ghost(q)
    once(T, w) = T(Int(numerator(w))) / T(Int(denominator(w)))
    for T in (Float64, Float32)
        tab = closure_table(T, Val(q))
        @test isbits(tab)
        w1 = derivative_weights(T, Val(q), Val(1))
        w2 = derivative_weights(T, Val(q), Val(2))
        wk = dissipation_weights(T, Val(G))
        for (km, kp) in closure_cases(q)
            i, j = km + 1, kp + 1
            @test tab.admissible[i, j] == closure_admissible(q, km, kp)
            if min(km, kp) >= q ÷ 2
                @test SVector{q + 1}(tab.d1[2:(end - 1), i, j]) === w1 &&
                      SVector{q + 1}(tab.d2[2:(end - 1), i, j]) === w2 &&
                      iszero(tab.d1[1, i, j]) && iszero(tab.d1[end, i, j])
            end
            min(km, kp) >= G && @test SVector{2G + 1}(tab.ko[:, i, j]) === wk
            if has_d2(q, km, kp)
                for (arr, m) in ((tab.d1, 1), (tab.d2, 2))
                    nodes, w = closure_derivative_weights(q, m, km, kp)
                    @test tab.d_lo[i, j] == first(nodes) &&
                          tab.d_hi[i, j] == last(nodes) &&
                          all(arr[n + G + 1, i, j] === once(T, w[k])
                              for (k, n) in enumerate(nodes)) &&
                          all(iszero(arr[s, i, j]) for s in 1:(2G + 1)
                              if !(s - G - 1 in nodes))
                end
            else
                @test tab.d_lo[i, j] > tab.d_hi[i, j]
            end
            nodes, w = closure_dissipation_weights(q, :msn, km, kp)
            @test tab.ko_lo[i, j] == first(nodes) && tab.ko_hi[i, j] == last(nodes) &&
                  all(tab.ko[n + G + 1, i, j] === once(T, w[k])
                      for (k, n) in enumerate(nodes))
        end
        # The other dissipation closures are a keyword away.
        @test all(iszero, closure_table(T, Val(q); dissipation=:reduced).ko[:, 1, G + 1])
    end
end

@testset "A closure that does not exist is refused, with the reason" begin
    # `ArgumentError`s say why (`CLAUDE.md`, "Conventions"): a point excised
    # on both sides of an axis within a point has no `∂²` at all, an odd
    # order has no centered scheme to close, and an unknown dissipation
    # closure is a typo the table would otherwise silently build.
    @test_throws ArgumentError closure_derivative_weights(4, 2, 0, 1)
    @test occursin("both sides", errmsg(() -> closure_derivative_weights(4, 2, 0, 1)))
    @test_throws ArgumentError closure_derivative_weights(4, 3, 0, 3)
    @test_throws ArgumentError closure_nodes(3, 0, 2)
    @test_throws ArgumentError closure_nodes(4, -1, 2)
    @test_throws ArgumentError closure_dissipation_weights(4, :upwind, 0, 3)
    @test occursin("msn", errmsg(() -> closure_dissipation_weights(4, :upwind, 0, 3)))
    @test_throws ArgumentError lopsided_weights(4, 0, 0, 3)
    @test !closure_admissible(4, 2, 2) && closure_admissible(4, 0, 3)
end

# ---------------------------------------------------------------------------
# The frame-dragged faces' extrapolation (added in step X5)
# ---------------------------------------------------------------------------
#
# `CODE.md`, "Excision", "The frame-dragged faces (step X5)": where a closure
# axis has the shift pointing into the excised set, the advective derivative
# is the centered stencil with each excised tap extrapolated along the
# lattice direction nearest the surface's normal, from the consecutive
# non-excised points `k₀ … k₀ + n − 1` beyond it inside the point's `G`-box.
# The claims are exact, in `Rational`, as above; step X6's zone kernel is what
# will use the table.

using TreeGeneralizedHarmonic: extrapolation_table, extrapolation_weights

@testset "An extrapolation is exact to degree n − 1 and not n: k₀=$k0" for k0 in 1:5
    # Guards the order the frame-dragged faces' advection is built on: the
    # filled tap is the Lagrange extrapolation, so it reproduces every
    # polynomial of degree below `n` along its direction and *not* degree
    # `n` — which catches a node list off by one (an extrapolation from the
    # wrong side of the gap reproduces nothing) and a weight vector built for
    # `n − 1` nodes. Its weights sum to one: a constant state is filled with
    # itself, so a static hole's frozen-in data stay a fixed point.
    for n in 1:4
        nodes, w = extrapolation_weights(k0, n)
        @test nodes == k0:(k0 + n - 1) && length(w) == n && sum(w) == 1
        x0, h = RQ(3//7), RQ(2//5)
        fill_at(f) = sum(w[i] * f(x0 + k * h) for (i, k) in enumerate(nodes))
        @test all(fill_at(monomial(d)) == monomial(d)(x0) for d in 0:(n - 1))
        @test fill_at(monomial(n)) != monomial(n)(x0)
    end
    # The quadratic from the first three points beyond the gap, written out:
    # `3u₁ − 3u₂ + u₃`, the textbook one.
    @test extrapolation_weights(1, 3)[2] == RQ[3, -3, 1]
end

@testset "The extrapolation table rounds once and stays in the G-box: q=$q" for q in
                                                                              STENCIL_ORDERS
    # Guards what the zone kernel will carry: every entry the exact rational
    # rounded once into `T` (`===` at `Float64` and `Float32`), zero beyond
    # its `n` nodes, and nothing past the point's `G`-box, `k₀ + n − 1 ≤ G` —
    # a source further out is outside the halo the mesh has.
    G = ghost(q)
    once(T, w) = T(Int(numerator(w))) / T(Int(denominator(w)))
    for T in (Float64, Float32), degree in (1, 2)
        tab = extrapolation_table(T, Val(q); degree=degree)
        @test isbits(tab) && size(tab) == (degree + 1, G, degree + 1)
        for k0 in 1:G, n in 1:(degree + 1)
            if k0 + n - 1 <= G
                _, w = extrapolation_weights(k0, n)
                @test all(tab[i, k0, n] === once(T, w[i]) for i in 1:n) &&
                      all(iszero(tab[i, k0, n]) for i in (n + 1):(degree + 1))
            else
                @test all(iszero, tab[:, k0, n])
            end
        end
    end
end

@testset "An extrapolation that does not exist is refused, with the reason" begin
    # `ArgumentError`s say why (`CLAUDE.md`, "Conventions"): node 0 is the
    # excised point itself, and an extrapolation from it is a read of an
    # excised value.
    @test_throws ArgumentError extrapolation_weights(0, 3)
    @test occursin("excised point itself", errmsg(() -> extrapolation_weights(0, 3)))
    @test_throws ArgumentError extrapolation_weights(1, 0)
    @test_throws ArgumentError extrapolation_table(Float64, Val(4); degree=-1)
end
