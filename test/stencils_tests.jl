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
