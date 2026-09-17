# The finite-difference and Kreiss–Oliger weights, built in exact rational
# arithmetic and rounded once into `T`.
#
# `CODE.md`, "Finite-difference stencils" and "Kreiss–Oliger dissipation".
# Centered stencils of even order `q ∈ {2, 4, 6, 8}` for `∂_i` and `∂_i∂_i`,
# the mixed `∂_i∂_j` as the tensor product of two first derivatives, and the
# dissipation operator of order `2r = q + 2`. All of them are a *property of
# the scheme*, not of the run's precision: the weights are exact rational
# numbers, and the only rounding anywhere in the construction is the single
# division of two exact integers that produces each one in `T`. This is TreeAMR's rule for its interpolation weights
# (its `operators.jl`, `lagrange_weights`) and it is here for the same
# reason -- building them in floating point would fix an accuracy ceiling at
# whatever type they were built in, and would need hardware `Float64` to
# reach it.
#
# Four things the rest of the package relies on:
#
#   * **The weights are for unit spacing.** `derivative_weights` returns the
#     weights of `∂^m` on samples one apart; the physical operator is
#     `apply_stencil(w, …) / h^m`. `dissipation_weights` returns the weights
#     of `(−1)^{r+1} 2^{−2r} (Δ_+Δ_−)^r` on the same samples, so the physical
#     operator is `(ε/h) apply_stencil(w, …)` -- see its docstring for where
#     each factor of the `CODE.md` formula went.
#   * **They are `isbits` and free at run time.** Both are `@generated`, so
#     the `Rational{BigInt}` construction happens once, when the method is
#     compiled for a given `(q, m)`, and the emitted code holds the exact
#     numerator and denominator of each weight as `Int` literals with one
#     division between them. A kernel of step 3 may call them per point with
#     `q` a `Val` parameter; nothing rational, nothing heap-allocated and
#     nothing of `BigInt` ever reaches a device, and at `Float64` and
#     `Float32` the division folds into a constant before the kernel runs.
#   * **The conversion into `T` happens at the call site, not in the
#     generator.** This is not a style choice (measured in step 2): a
#     generator may only call methods that existed when the generated
#     function was *defined*, and this package is precompiled long before a
#     driver loads MultiFloats. A generator that wrote `T(w)` for an exact
#     rational `w` therefore worked at `Float64` and `Float32` and threw
#     `MethodError: ... The applicable method may be too new` at
#     `Float32x2` — at exactly the type `CODE.md` keeps in the suite to
#     prove nothing depends on a hardware float. Emitting `T(num)/T(den)`
#     moves the conversion into the caller's world, where every type's
#     methods exist.
#   * **`q` is even.** Everything here is centered, and the half-width
#     `q ÷ 2` is what `CODE.md`'s ghost width `G = q/2 + 1` is built from --
#     one more than the derivatives need, because the dissipation reaches
#     one point further.
#
# `apply_stencil` and `apply_mixed_stencil` are the host-side reference
# contractions. They are what the tests measure the weights with, and what
# the streaming kernel of step 3 is written *against* rather than with: the
# kernel forms and consumes one stencil at a time and never builds a vector
# of all of them (`CLAUDE.md`, "The RHS kernel is written in streaming
# order").

# `BigInt` for the same reason TreeAMR uses it: `Rational` arithmetic is
# checked, so an overflowing intermediate would be a hard error rather than
# a wrong answer, and a bignum removes the ceiling instead of moving it.
# This runs at compile time, never per point.
const StencilRational = Rational{BigInt}

# Multiply the polynomial `c` (coefficients low to high) by `(x + a)`.
function _polymul_linear(c::Vector{StencilRational}, a::StencilRational)
    out = Vector{StencilRational}(undef, length(c) + 1)
    out[1] = a * c[1]
    for i in 2:length(c)
        out[i] = c[i - 1] + a * c[i]
    end
    out[end] = c[end]
    return out
end

"""
    lagrange_derivative_weights(nodes, m) -> Vector{Rational{BigInt}}

Weights `w` with `sum(w[i] * u(nodes[i])) == u^(m)(0)` for every polynomial
`u` of degree less than `length(nodes)`.

The `m`-th derivative at the origin of the Lagrange basis on `nodes`,
computed by building each basis polynomial's numerator
`∏_{k≠i} (x − x_k)` coefficient by coefficient and reading off `m! · c_m`.
Nodes are exact rationals and so is the result; that exactness is the whole
point, and it is why this is written out here rather than taken from a
Vandermonde solve.

Exactness holds for *any* distinct nodes, so this is also what a one-sided
or shifted stencil would be built from if one were ever wanted. The
centered stencils this package uses are [`derivative_weights`](@ref).
"""
function lagrange_derivative_weights(nodes::AbstractVector{<:Rational},
                                     m::Integer)
    n = length(nodes)
    m >= 0 || throw(ArgumentError("a derivative order must be non-negative, " *
                                  "but m=$m"))
    ns = StencilRational.(nodes)
    ws = Vector{StencilRational}(undef, n)
    for i in 1:n
        c = [one(StencilRational)]
        den = one(StencilRational)
        for k in 1:n
            k == i && continue
            c = _polymul_linear(c, -ns[k])
            den *= (ns[i] - ns[k])
        end
        # `u^(m)(0) = m! · c_m`, and a polynomial of degree below `m` has no
        # such coefficient at all.
        ws[i] = m < n ? factorial(big(m)) * c[m + 1] / den :
                zero(StencilRational)
    end
    return ws
end

"""
    rational_derivative_weights(q, m) -> Vector{Rational{BigInt}}

The exact weights behind [`derivative_weights`](@ref): the centered `q`-th
order stencil for `∂^m` on the `q + 1` integer nodes `−q/2 … q/2`.

Exposed because exactness is a claim a test should be able to make in
`Rational` rather than through a tolerance — the textbook tables, the
polynomial degrees each operator is and is not exact on, and the
bit-identity of the conversion into `Float64`, all of which
`test/stencils_tests.jl` asserts here rather than on the rounded weights.
"""
function rational_derivative_weights(q::Integer, m::Integer)
    q >= 2 && iseven(q) || throw(ArgumentError(
        "a centered stencil needs an even order q ≥ 2 so that its half-width " *
        "q/2 is an integer and CODE.md's ghost width G = q/2 + 1 covers it, " *
        "but q=$q"))
    m in (1, 2) || throw(ArgumentError(
        "this package differentiates at most twice — CODE.md's second-order " *
        "reduction takes ∂_i and ∂_i∂_j and nothing else — but m=$m"))
    r = q ÷ 2
    return lagrange_derivative_weights([StencilRational(j) for j in (-r):r], m)
end

"""
    rational_dissipation_weights(r) -> Vector{Rational{BigInt}}

The exact weights behind [`dissipation_weights`](@ref): the `2r + 1` entries
of `(−1)^{r+1} 2^{−2r} (Δ_+Δ_−)^r`, the undivided Kreiss–Oliger operator on
unit-spaced samples.

`(Δ_+Δ_−)^r` is the `2r`-th undivided central difference, whose coefficient
at offset `j` is `(−1)^{r−j} binom(2r, r−j)`; the alternating sign
`(−1)^{r+1}` of `CODE.md`'s formula makes the center coefficient negative at
every `r`, which is what makes the operator damping rather than driving.
"""
function rational_dissipation_weights(r::Integer)
    r >= 1 || throw(ArgumentError(
        "the Kreiss–Oliger operator of order 2r = q + 2 needs a rank r ≥ 1 " *
        "to have a stencil at all — and r ≥ 2 at every order q ≥ 2 this " *
        "package uses — but r=$r"))
    sgn = iseven(r + 1) ? 1 : -1
    den = StencilRational(big(2)^(2r))
    return [sgn * (-1)^(r - j) * StencilRational(binomial(2r, r - j)) / den
            for j in (-r):r]
end

# The body every weight method returns: one `T(num)/T(den)` per entry, with
# the exact numerator and denominator spliced as `Int` literals.
#
# The division is the *only* rounding in the construction, and it happens in
# the caller's world at the caller's type. For an IEEE type it is correctly
# rounded — bit-identical to converting the exact rational, since `num` and
# `den` are small integers and therefore exact in `T` — and for a software
# type it is correct to that type's own division accuracy, which is one ulp
# of something far below anything this package measures. Doing it the other
# way, converting in the generator, is what the header comment records as
# failing at `Float32x2`.
function _weight_expr(ws::Vector{StencilRational})
    entries = [:(T($(Int(numerator(w)))) / T($(Int(denominator(w)))))
               for w in ws]
    return :(SVector{$(length(ws)),T}($(entries...)))
end

"""
    derivative_weights([T = Float64], ::Val{q}, ::Val{m}) -> SVector{q+1,T}

The centered finite-difference weights of order `q` for the `m`-th
derivative, `m ∈ {1, 2}`, on `q + 1` samples **one apart**, in the offset
order `−q/2 … q/2`.

The physical operator divides by the spacing: `∂^m u ≈ apply_stencil(w, u,
i) / h^m`, with `h` the block's own spacing. Keeping `h` out of the weights
is what lets one weight vector serve every refinement level, and it is the
form the streaming kernel wants — the `1/h^m` are per-block coefficients
read once per point, the weights are compile-time constants.

`q` is even, and `Val`-wrapped because the kernel specialises on it
(`CLAUDE.md`: the `Val`s are built once per chunk, never per evaluation).
The first derivative's center weight is exactly zero and is returned anyway,
so that both operators have the same layout; a kernel may skip it.

Order, in the sense a convergence test measures: the first derivative is
exact on polynomials of degree `≤ q` and not `q + 1`; the second, on degree
`≤ q + 1` — one better than it was built for, by the symmetry of an even
`q` — and not `q + 2`. Both therefore carry a truncation error `O(h^q)`.

The **mixed** derivative `∂_i∂_j`, `i ≠ j`, has no weights of its own: it is
the tensor product of two of these first-derivative vectors, one per axis,
which is the form that reads the edge and corner ghosts TreeAMR fills
unconditionally (`CODE.md`, "Finite-difference stencils"). The reference
contraction is [`apply_mixed_stencil`](@ref), and its docstring says which
end the sum is formed at.

Built in exact rational arithmetic and rounded once into `T`; see
[`rational_derivative_weights`](@ref). The method is `@generated`, so the
rationals exist only while it compiles: what it emits is each weight's exact
numerator and denominator as `Int` literals with one division between them,
and the result is an `isbits` `SVector` a device kernel holds in registers.
At `Float64` and `Float32` that division is correctly rounded — the same
bits as converting the exact rational — and folds to a constant before the
kernel runs.
"""
@generated function derivative_weights(::Type{T}, ::Val{q},
                                       ::Val{m}) where {T,q,m}
    ws = try
        rational_derivative_weights(q, m)
    catch err
        err isa ArgumentError || rethrow()
        return :(throw($err))
    end
    return _weight_expr(ws)
end

@inline derivative_weights(q::Val, m::Val) = derivative_weights(Float64, q, m)

"""
    dissipation_weights([T = Float64], ::Val{r}) -> SVector{2r+1,T}

The Kreiss–Oliger dissipation weights of order `2r = q + 2` on `2r + 1`
samples **one apart**, in the offset order `−r … r`.

`CODE.md`, "Kreiss–Oliger dissipation", fixes the operator as

    Q_d u = ε (−1)^{r+1} (h_d^{2r−1} / 2^{2r}) (D_+ D_−)^r u,   r = q/2 + 1

with `h_d` the block's own spacing. What is in the weights and what is not:

| factor | where it is applied |
|---|---|
| `(−1)^{r+1}` | **in the weights** |
| `2^{−2r}` | **in the weights** |
| `(D_+D_−)^r`, the undivided part | **in the weights** |
| `h_d^{2r−1}` against `(D_+D_−)^r`'s own `h_d^{−2r}` | by the caller, as a single `1/h_d` |
| `ε` | by the caller |

so that the whole operator is `Q_d u = (ε / h_d) * apply_stencil(w, u, i)`.
One factor of the spacing, not `2r − 1` of them: that is the point of the
`h_d^{2r−1}` in the formula, and it is why a refinement level's dissipation
scales with its own resolution and `ε ∈ (0, 1)` is neutral to the CFL
condition.

**The sign is damping**, and it is carried by the weights: the center weight
is `−binom(2r, r)/2^{2r} < 0` at every `r`, and on a Fourier mode
`u_j = exp(i k x_j)` the contraction is

    apply_stencil(w, u, i) = −sin^{2r}(k h_d / 2) · u_i    ≤ 0 · u_i

so `Q_d u = −(ε/h_d) sin^{2r}(k h_d/2) u`. The Nyquist mode `k h_d = π` is
damped at exactly `ε/h_d` and the smooth modes are barely touched, which is
the normalisation the formula's `2^{−2r}` exists for. `Q_d` is *added* to
the right-hand side; flipping the sign turns the term into an amplifier of
exactly the grid-scale noise it is there to remove, and the symptom is a run
that blows up faster with larger `ε`.

`Q_d` annihilates polynomials of degree `< 2r`, so it contributes
`O(h^{2r−1}) = O(h^{q+1})` to the truncation error and does not degrade a
`q`-th order scheme. Its stencil reaches `r = q/2 + 1` points, one further
than the derivatives, which is where `CODE.md`'s ghost width `G = q/2 + 1`
comes from.

Its role is GHSO2's second finding under "sonic-surface instability"
(`notes/methods-ghso2.md`): with a horizon in the evolved domain the
grid-scale instability is cured at `ε ≈ 0.5`, and `ε` is a case parameter
because a smooth run without a hole needs little or none.

Built in exact rational arithmetic and rounded once into `T`; see
[`rational_dissipation_weights`](@ref) and [`derivative_weights`](@ref) for
what the `@generated` method emits. Every entry here is dyadic, so the
conversion is exact in any binary floating-point type.
"""
@generated function dissipation_weights(::Type{T}, ::Val{r}) where {T,r}
    ws = try
        rational_dissipation_weights(r)
    catch err
        err isa ArgumentError || rethrow()
        return :(throw($err))
    end
    return _weight_expr(ws)
end

@inline dissipation_weights(r::Val) = dissipation_weights(Float64, r)

"""
    dissipation_rank(::Val{q}) -> Val{r}

The rank `r = q/2 + 1` of the dissipation operator that goes with a `q`-th
order scheme, so that no call site has to spell `CODE.md`'s relation
`2r = q + 2` again.

The same number is the ghost width `G`, and for the same reason: the
dissipation's stencil is the widest one an evaluation takes, reaching `r`
points where the derivatives reach `q/2` **(proposed in step 2** — the
relation is `CODE.md`'s, only the spelling is new**)**.
"""
@inline dissipation_rank(::Val{q}) where {q} = Val(q ÷ 2 + 1)

"""
    apply_stencil(w, u::AbstractVector, i) -> scalar
    apply_stencil(w, f, x, h) -> scalar

The centered contraction of a weight vector: `sum(w[k] * u[i + k − 1 − r])`
over the `2r + 1` entries of `w`, either against samples stored in `u` around
index `i`, or against a callable `f` sampled at `x + j·h` for `j = −r … r`.

Host-side, and for the tests: this is the reference the weights are measured
with, and the definition the streaming kernel of step 3 has to agree with
while never materialising `u` or `w` as anything but registers
(`CLAUDE.md`, "The RHS kernel is written in streaming order"). Nothing in an
evaluation calls it.

It returns the raw contraction. The scale factor is the caller's, and which
one it is depends on the weights: `/h^m` for
[`derivative_weights`](@ref), `ε/h` for [`dissipation_weights`](@ref).

Summation runs from the lowest offset to the highest, which is a choice and
not a law — a different order differs in the last place, and `CODE.md`'s
"Measured results" records what that does and does not mean for the
bit-identity the threading test asserts.
"""
@inline function apply_stencil(w::SVector{n}, u::AbstractVector,
                               i::Integer) where {n}
    r = (n - 1) ÷ 2
    s = w[1] * u[i - r]
    for k in 2:n
        s += w[k] * u[i - r + k - 1]
    end
    return s
end

@inline function apply_stencil(w::SVector{n}, f, x, h) where {n}
    r = (n - 1) ÷ 2
    s = w[1] * f(x - r * h)
    for k in 2:n
        s += w[k] * f(x + (k - 1 - r) * h)
    end
    return s
end

"""
    apply_mixed_stencil(w, f, x, y, hx, hy) -> scalar

The mixed derivative as the tensor product of two first-derivative weight
vectors: `sum_a w[a] * sum_b w[b] * f(x + a·hx, y + b·hy)`, the reference
for `∂_i∂_j` with `i ≠ j`.

`CODE.md`, "Finite-difference stencils": the mixed derivative has no weights
of its own. The same vector serves both axes, and the `(q+1)²` product is
never built — it is a sum of sums, which is what lets a kernel form it one
line at a time.

**Which end it is formed at**: the *inner* sum runs along the second axis
`y`, the *outer* along the first axis `x`. The two orders are equal in exact
arithmetic and differ in the last place in floating point, so the order is
part of the operator, not an implementation detail; step 3's kernel picks
one and keeps it.

As with [`apply_stencil`](@ref) the result is the raw contraction: the
physical `∂_x∂_y f` divides it by `hx·hy`.
"""
@inline function apply_mixed_stencil(w::SVector, f, x, y, hx, hy)
    return apply_stencil(w, ξ -> apply_stencil(w, η -> f(ξ, η), y, hy), x, hx)
end
