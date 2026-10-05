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

# ---------------------------------------------------------------------------
# The closures at an excision surface (added in step X1)
# ---------------------------------------------------------------------------
#
# `CODE.md`, "Excision": at an evolved point next to the excised set the
# centered stencils of an axis would read excised points, so per axis and
# side the point counts `k⁻, k⁺ ∈ 0…G`, the consecutive non-excised points
# on each side (capped at `G`, so `G` means "at least `G`"), and every
# stencil that would reach past them is replaced by a **closure** on the
# points it may read. Everything below is a function of `(q, k⁻, k⁺)` and
# nothing else — no position, no block, no level — built in `Rational` on
# `lagrange_derivative_weights`, exactly as the centered weights are, and
# rounded once into `T` by `closure_table`. No kernel uses it yet: step
# X2b's zone kernel is what will, and step X1's models
# (`test/excision_model.jl`) are what chose among the options.
#
# Three families, each with the full-width code as the case where it fits:
#
#   * the derivatives `∂` and `∂²` (`closure_derivative_weights`): centered
#     when `min(k⁻, k⁺) ≥ q/2`, otherwise the Lagrange derivative on *every*
#     node in `[−k⁻, k⁺]` capped at the reach (`G` by default) — the most
#     accurate stencil the point may read, and the one an extrapolation of
#     the excised taps followed by the centered stencil would give;
#   * the dissipation (`closure_dissipation_weights`): centered when
#     `min(k⁻, k⁺) ≥ G`, otherwise one of three closures — reduced rank,
#     one-sided, or Mattsson–Svärd–Nordström's boundary-modified form, the
#     one that keeps the damping sign in the discrete `l²` norm;
#   * the lopsided advection (`lopsided_weights`): the order-`q` first
#     derivative on `q + 1` nodes shifted one point to the upwind side,
#     which reaches exactly `G` there and acts on the Nyquist mode that
#     every centered first derivative annihilates.

# The ghost width `G = q/2 + 1`: the farthest any stencil of the scheme
# reaches, and therefore the cap on `k⁻, k⁺` and on a closure's reach.
_reach(q::Integer) = q ÷ 2 + 1

function _check_closure_args(q, kminus, kplus)
    q >= 2 && iseven(q) || throw(ArgumentError(
        "a closure belongs to a centered scheme of even order q ≥ 2, whose " *
        "half-width q/2 and ghost width G = q/2 + 1 define it, but q=$q"))
    kminus >= 0 && kplus >= 0 || throw(ArgumentError(
        "k⁻ and k⁺ count the consecutive non-excised points on each side of " *
        "the point and cannot be negative, but k⁻=$kminus, k⁺=$kplus"))
    return nothing
end

"""
    closure_nodes(q, k⁻, k⁺; reach = q/2 + 1) -> UnitRange{Int}

The offsets a closure of the order-`q` scheme reads at a point with `k⁻`
and `k⁺` consecutive non-excised points to its left and right (`CODE.md`,
"Excision"): the centered `−q/2 … q/2` when `min(k⁻, k⁺) ≥ q/2`, and
otherwise **every** offset in `[−min(k⁻, reach), min(k⁺, reach)]`.

`reach` is the cap, the scheme's ghost width `G = q/2 + 1` by default —
the halo the mesh already has — so a closure never reads further than a
centered dissipation stencil does. Other values exist for step X1's
models, which ask whether a narrower or a wider closure is more stable.

The range is the same for `∂` and `∂²`; whether it holds enough nodes for
either is [`closure_derivative_weights`](@ref)'s question.
"""
function closure_nodes(q::Integer, kminus::Integer, kplus::Integer;
                       reach::Integer=_reach(q))
    _check_closure_args(q, kminus, kplus)
    reach >= 1 || throw(ArgumentError(
        "a closure's reach is how far it may read and must be at least one " *
        "point, but reach=$reach"))
    r = q ÷ 2
    min(kminus, kplus) >= r && return (-r):r
    return (-min(kminus, reach)):min(kplus, reach)
end

"""
    closure_derivative_weights(q, m, k⁻, k⁺; reach = q/2 + 1)
        -> (nodes::UnitRange{Int}, weights::Vector{Rational{BigInt}})

The closure of `∂^m`, `m ∈ {1, 2}`, at a point with `k⁻, k⁺` consecutive
non-excised points on each side: the Lagrange derivative on
[`closure_nodes`](@ref), in `Rational`, for unit spacing (the caller divides
by `h^m`, as for [`derivative_weights`](@ref)).

Where `min(k⁻, k⁺) ≥ q/2` this **is** the centered stencil,
[`rational_derivative_weights`](@ref)`(q, m)`, entry for entry. Elsewhere it
is exact to the degree [`closure_exact_degree`](@ref) says — at the first
evolved point (`k⁻ = 0`, reach `G`) of orders `q/2 + 1` for `∂` and `q/2`
for `∂²`: `(2, 1)` at `q = 2`, `(3, 2)` at `q = 4`.

A closure needs `m + 1` nodes; fewer is an `ArgumentError`, which is what a
point excised on both sides of an axis within a point or two meets.
"""
function closure_derivative_weights(q::Integer, m::Integer, kminus::Integer,
                                    kplus::Integer; reach::Integer=_reach(q))
    m in (1, 2) || throw(ArgumentError(
        "this package differentiates at most twice — CODE.md's second-order " *
        "reduction takes ∂_i and ∂_i∂_j and nothing else — but m=$m"))
    nodes = closure_nodes(q, kminus, kplus; reach=reach)
    length(nodes) >= m + 1 || throw(ArgumentError(
        "∂^$m needs at least $(m + 1) nodes, and a point with k⁻=$kminus, " *
        "k⁺=$kplus non-excised neighbours (reach $reach) has only the " *
        "offsets $nodes: it is excised on both sides of this axis too close " *
        "to have a closure"))
    if nodes == (-(q ÷ 2)):(q ÷ 2)
        return nodes, rational_derivative_weights(q, m)
    end
    return nodes,
           lagrange_derivative_weights([StencilRational(j) for j in nodes], m)
end

"""
    closure_exact_degree(q, m, k⁻, k⁺; reach = q/2 + 1) -> Int

The highest polynomial degree on which [`closure_derivative_weights`](@ref)
is exact: `n − 1` on `n` nodes, and one more when the nodes are symmetric
about the point and `n − m` is odd — the centered `∂²`'s extra degree. The
order of the truncation error is this plus `1 − m`. `test/stencils_tests.jl`
asserts it in `Rational`, and that the closure is *not* exact one degree
further.
"""
function closure_exact_degree(q::Integer, m::Integer, kminus::Integer,
                              kplus::Integer; reach::Integer=_reach(q))
    nodes = closure_nodes(q, kminus, kplus; reach=reach)
    n = length(nodes)
    sym = first(nodes) == -last(nodes)
    return n - 1 + (sym && isodd(n - m) ? 1 : 0)
end

"""
    DISSIPATION_CLOSURES

The three closures of the Kreiss–Oliger operator step X1 compares
(`CODE.md`, "Excision"):

- `:reduced` — the centered operator of the largest rank `r′ ≤ r` that fits,
  `r′ = min(k⁻, k⁺)`, and none at all at the first evolved point;
- `:onesided` — the `2r′`-th difference on the most nodes that fit,
  shifted toward the evolved side and signed so that it damps the
  Nyquist mode at the point;
- `:msn` — Mattsson, Svärd and Nordström's boundary-modified form
  (J. Sci. Comput. 21, 57, 2004), `−2^{−2r} D_rᵀ B D_r`, with `D_r` the
  `r`-th undivided forward difference and `B` the indicator of the rows
  whose `r + 1` points are all evolved: the interior rows are the centered
  operator, and the assembled operator is symmetric and negative
  semidefinite in the discrete `l²` norm **by construction**, on any
  pattern of excised points.
"""
const DISSIPATION_CLOSURES = (:reduced, :onesided, :msn)

# The `r`-th undivided forward difference's coefficient at offset `j` of a
# window starting at `0`: `(−1)^{r−j} binom(r, j)`.
_forward_difference(r, j) =
    StencilRational((iseven(r - j) ? 1 : -1) * binomial(big(r), big(j)))

"""
    closure_dissipation_weights(q, kind, k⁻, k⁺)
        -> (nodes::UnitRange{Int}, weights::Vector{Rational{BigInt}})

The Kreiss–Oliger operator of rank `r = G = q/2 + 1` at a point with `k⁻,
k⁺` consecutive non-excised points on each side, closed by `kind ∈`
[`DISSIPATION_CLOSURES`](@ref). The weights carry the operator's sign and
`2^{−2r}` exactly as [`rational_dissipation_weights`](@ref) does, so the
caller applies `ε/h` as for the centered one.

Where `min(k⁻, k⁺) ≥ G` every kind **is** the centered operator,
[`rational_dissipation_weights`](@ref)`(G)`. Elsewhere the nodes lie in
`[−min(k⁻, G), min(k⁺, G)]`, and a closure that has no room returns the
single node `0` with weight `0`: no dissipation at that point.
"""
function closure_dissipation_weights(q::Integer, kind::Symbol, kminus::Integer,
                                     kplus::Integer)
    _check_closure_args(q, kminus, kplus)
    kind in DISSIPATION_CLOSURES || throw(ArgumentError(
        "the dissipation closures are $(DISSIPATION_CLOSURES) — reduced rank, " *
        "one-sided, and Mattsson–Svärd–Nordström's — but kind=:$kind"))
    G = _reach(q)
    none = (0:0, [zero(StencilRational)])
    min(kminus, kplus) >= G && return (-G):G, rational_dissipation_weights(G)
    lo, hi = -min(kminus, G), min(kplus, G)
    if kind === :reduced
        rr = min(kminus, kplus)
        rr == 0 && return none
        return (-rr):rr, rational_dissipation_weights(rr)
    elseif kind === :onesided
        rr = min(G, (hi - lo) ÷ 2)
        rr == 0 && return none
        s = clamp(0, lo + rr, hi - rr)
        sgn = iseven(s) ? 1 : -1
        return (s - rr):(s + rr), sgn .* rational_dissipation_weights(rr)
    else                                   # :msn
        # The windows `[k, k + G]` that contain the point and lie in
        # `[lo, hi]`; the row is `−2^{−2G} Σ_k D_{k,0} D_{k,·}`.
        ks = max(lo, -G):min(0, hi - G)
        isempty(ks) && return none
        nodes = first(ks):(last(ks) + G)
        w = zeros(StencilRational, length(nodes))
        scale = -inv(StencilRational(big(2)^(2G)))
        for k in ks
            d0 = _forward_difference(G, -k)       # the point is at `−k` in it
            for j in 0:G
                w[k + j - first(nodes) + 1] += scale * d0 *
                                               _forward_difference(G, j)
            end
        end
        return nodes, w
    end
end

"""
    lopsided_weights(q, up, k⁻, k⁺) -> (nodes::UnitRange{Int},
                                       weights::Vector{Rational{BigInt}})

The lopsided (upwind-biased) first derivative of order `q` for the shift
advection `β^d ∂_d`, at a point with `k⁻, k⁺` consecutive non-excised
points on each side, `up = ±1` the upwind side — the side the shift points
to, since `∂_t u = +β^d ∂_d u` carries information against `β`.

On the open line it is the Lagrange derivative on the `q + 1` nodes
`1 − q/2 … q/2 + 1` (mirrored for `up = −1`): order `q`, reaching exactly
`G` on the upwind side and `q/2 − 1` on the other, and **not** annihilating
the Nyquist mode, on which every centered `D₁` vanishes — inside a horizon
it damps the grid-scale content that the centered advection carries outward
(`CODE.md`, "Kreiss–Oliger dissipation", step 8a). Where the downwind side
is shorter than `q/2 − 1` the nodes start at `−k_down`; where the upwind
side has fewer than `G` points it is not lopsided at all, and the result is
[`closure_derivative_weights`](@ref)`(q, 1, k⁻, k⁺)`.
"""
function lopsided_weights(q::Integer, up::Integer, kminus::Integer,
                          kplus::Integer)
    _check_closure_args(q, kminus, kplus)
    up in (-1, 1) || throw(ArgumentError(
        "the upwind side is +1 or −1 along the axis, the sign of the shift " *
        "component, but up=$up"))
    G = _reach(q)
    kup, kdn = up > 0 ? (kplus, kminus) : (kminus, kplus)
    kup >= G || return closure_derivative_weights(q, 1, kminus, kplus)
    # In the upwind frame (the upwind side positive).
    lo = max(1 - q ÷ 2, -kdn)
    w = lagrange_derivative_weights([StencilRational(j) for j in lo:G], 1)
    up > 0 && return lo:G, w
    # Mirrored: `∂` is odd, so the weights change sign and order.
    return (-G):(-lo), -reverse(w)
end

"""
    lopsided_centered_weights([T = Float64], ::Val{q}, ::Val{up}) -> SVector{q+1,T}

The lopsided first derivative of [`lopsided_weights`](@ref) on the open
line — `k⁻, k⁺ ≥ G` — as an `isbits` vector for the main kernel (added in
step X2b): the order-`q` weights on the `q + 1` offsets `1 − q/2 … q/2 + 1`
for `up = +1`, and on `−(q/2 + 1) … q/2 − 1` for `up = −1`, in ascending
offset order ([`lopsided_first`](@ref) is the first), for unit spacing.
`@generated` as [`derivative_weights`](@ref) is, emitting each weight's
exact numerator and denominator, so that its entries are
[`closure_table`](@ref)'s `lop[·, G + 1, G + 1, ·]` bit for bit at every IEEE
type: the main kernel's blend and the zone kernel's are one operator.
"""
@generated function lopsided_centered_weights(::Type{T}, ::Val{q},
                                              ::Val{up}) where {T,q,up}
    ws = try
        lopsided_weights(q, up, _reach(q), _reach(q))[2]
    catch err
        err isa ArgumentError || rethrow()
        return :(throw($err))
    end
    return _weight_expr(ws)
end

"""
    lopsided_first(::Val{q}, ::Val{up}) -> Int

The offset [`lopsided_centered_weights`](@ref)`(T, Val(q), Val(up))` starts
at: `1 − q/2` upwind to the right, `−(q/2 + 1)` to the left.
"""
@inline lopsided_first(::Val{q}, ::Val{up}) where {q,up} =
    up > 0 ? 1 - q ÷ 2 : -(q ÷ 2 + 1)

"""
    closure_admissible(q, k⁻, k⁺) -> Bool

Whether a point with `k⁻, k⁺` non-excised points on each side of an axis
(capped at `G`) has a closure step X2b's build accepts: at least one side
clear to the full reach `G` (`PLAN.md`, step X2b: "refusing a zone point
with no admissible closure — excised on both sides of one axis within
reach"). The weights exist for more than this — a gap of three points has
a `∂²` closure — but a convex excised set never makes one, and a point that
meets it is a statement about the geometry, not about the stencil
**(proposed in step X1)**.
"""
function closure_admissible(q::Integer, kminus::Integer, kplus::Integer)
    _check_closure_args(q, kminus, kplus)
    return max(kminus, kplus) >= _reach(q)
end

"""
    ClosureTable

What [`closure_table`](@ref) returns: every closure of one order `q`,
rounded once into `T`, as `isbits` arrays a kernel argument can carry.
Weights are stored on the `2G + 1` offsets `−G … G` (slot `j + G + 1`), zero
outside a closure's nodes, and indexed `[slot, k⁻ + 1, k⁺ + 1]`; the
lopsided advection has a fourth index, `1` for `up = −1` and `2` for
`up = +1`. `*_lo` and `*_hi` are the nodes each closure reads, so that a
contraction runs over those and **never** touches an excised value, not
even with weight zero (`0 · NaN = NaN`). `admissible` is
[`closure_admissible`](@ref); where the derivative closures do not exist
at all (fewer than three nodes) their weights are zero and `d_lo > d_hi`.
"""
struct ClosureTable{T,q,A3,A4,I2,I3,B2}
    d1::A3
    d2::A3
    ko::A3
    lop::A4
    d_lo::I2
    d_hi::I2
    ko_lo::I2
    ko_hi::I2
    lop_lo::I3
    lop_hi::I3
    admissible::B2
end

"""
    closure_table(T, ::Val{q}; dissipation = :msn) -> ClosureTable

Every closure of the order-`q` scheme — `∂`, `∂²`, the dissipation closed
by `dissipation ∈` [`DISSIPATION_CLOSURES`](@ref) and the lopsided
advection, for every `k⁻, k⁺ ∈ 0…G` — built in `Rational` and rounded
**once** into `T` as `T(num)/T(den)`, the conversion
[`derivative_weights`](@ref) emits, so that the table's centered entries
are the centered weights bit for bit at every IEEE type
(`test/stencils_tests.jl`). A host function and not `@generated`: the
table is a kernel *argument* for step X2b, built once per problem, and a
plain function converts in the caller's world at every type.

`:msn` is the default because it is the closure that keeps the damping
sign in the `l²` norm on any excised pattern (`CODE.md`, "Excision", and
"Excision: the analysis (step X1)" under "Measured results").
"""
function closure_table(::Type{T}, ::Val{q};
                       dissipation::Symbol=:msn) where {T,q}
    G = _reach(q)
    W = 2G + 1
    K = G + 1
    conv(w) = T(Int(numerator(w))) / T(Int(denominator(w)))
    d1 = zeros(T, W, K, K)
    d2 = zeros(T, W, K, K)
    ko = zeros(T, W, K, K)
    lop = zeros(T, W, K, K, 2)
    d_lo = zeros(Int8, K, K)
    d_hi = fill(Int8(-1), K, K)
    ko_lo = zeros(Int8, K, K)
    ko_hi = zeros(Int8, K, K)
    lop_lo = zeros(Int8, K, K, 2)
    lop_hi = fill(Int8(-1), K, K, 2)
    adm = falses(K, K)
    for km in 0:G, kp in 0:G
        adm[km + 1, kp + 1] = closure_admissible(q, km, kp)
        if length(closure_nodes(q, km, kp)) >= 3
            for (arr, m) in ((d1, 1), (d2, 2))
                nodes, w = closure_derivative_weights(q, m, km, kp)
                for (j, wj) in zip(nodes, w)
                    arr[j + G + 1, km + 1, kp + 1] = conv(wj)
                end
                d_lo[km + 1, kp + 1] = first(nodes)
                d_hi[km + 1, kp + 1] = last(nodes)
            end
            for (iu, up) in enumerate((-1, 1))
                nodes, w = lopsided_weights(q, up, km, kp)
                for (j, wj) in zip(nodes, w)
                    lop[j + G + 1, km + 1, kp + 1, iu] = conv(wj)
                end
                lop_lo[km + 1, kp + 1, iu] = first(nodes)
                lop_hi[km + 1, kp + 1, iu] = last(nodes)
            end
        end
        nodes, w = closure_dissipation_weights(q, dissipation, km, kp)
        for (j, wj) in zip(nodes, w)
            ko[j + G + 1, km + 1, kp + 1] = conv(wj)
        end
        ko_lo[km + 1, kp + 1] = first(nodes)
        ko_hi[km + 1, kp + 1] = last(nodes)
    end
    A3 = SArray{Tuple{W,K,K}}
    I2 = SArray{Tuple{K,K}}
    I3 = SArray{Tuple{K,K,2}}
    t = (A3(d1), A3(d2), A3(ko), SArray{Tuple{W,K,K,2}}(lop), I2(d_lo),
         I2(d_hi), I2(ko_lo), I2(ko_hi), I3(lop_lo), I3(lop_hi), I2(adm))
    return ClosureTable{T,q,typeof(t[1]),typeof(t[4]),typeof(t[5]),
                        typeof(t[9]),typeof(t[11])}(t...)
end
