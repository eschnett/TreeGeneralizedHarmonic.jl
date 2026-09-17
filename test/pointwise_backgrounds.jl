# The backgrounds of `CODE.md`'s "Initial data and backgrounds" table, and
# the analytic data the pointwise tests compare against, in one place
# because two test files need them.
#
# Every claim about `src/pointwise.jl` is a claim against the *exact*
# solution: the algebra is validated by evaluating it on data that solves
# the Einstein equations and checking that what should vanish vanishes and
# what should be an identity is one. `SpacetimeMetrics` supplies both the
# solution and, through forward-mode automatic differentiation, its
# derivatives, so nothing here differentiates anything by hand.
#
# Nothing in this file is general relativity that this package owns; it is
# the reference the package is measured against.

import SpacetimeMetrics
using Random: MersenneTwister
using SpacetimeMetrics: AbstractMetric, GaugeWave, Harmonic, KerrSchild,
                        Minkowski, ShiftedMinkowski, boost, ddmetric, dmetric,
                        gauge_source, gauge_source_grad, metric
using StaticArrays: SArray, SMatrix, SVector
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: NC, _sym4

# ForwardDiff is a test-only dependency: it is how `metric_derivatives` and
# the expanded form are checked against a derivative this package did not
# compute. `SpacetimeMetrics` uses it internally for `dmetric`; here it is
# used on top of that, which is one extra dual layer and no more.
using ForwardDiff: ForwardDiff

# The six rows of `CODE.md`'s table, in its order, with the flag that says
# whether the background is harmonic (`H ≡ 0`) — the two hole cases differ
# in exactly that, and the gauge-source path is only exercised by the
# non-harmonic ones. `M = 1` and `a = 9/10` are the proof-of-concept
# parameters; `|v| = 3/10` is the boost the milestone names.
#
# Parameters are built from `Rational`s, never decimal literals, so that
# the `Float32` and `Float64` tables are the same numbers rounded once.
function gh_backgrounds(::Type{T}) where {T}
    return (
        (name="Minkowski", bg=Minkowski(), harmonic=true),
        (name="gauge wave", bg=GaugeWave(T(1//20), T(1)), harmonic=true),
        (name="shifted Minkowski",
         bg=ShiftedMinkowski(T(1//2), T(1)), harmonic=false),
        (name="Kerr-Schild", bg=KerrSchild{T}(1, 0), harmonic=false),
        (name="harmonic Kerr", bg=Harmonic{T}(1, 9//10), harmonic=true),
        (name="boosted harmonic Kerr",
         bg=boost(Harmonic{T}(1, 9//10), SVector{3,T}(3//10, 0, 0)),
         harmonic=true),
    )
end

# Points in a shell that contains no horizon and no ring singularity: the
# smallest coordinate radius of the `a = 9/10` horizon in harmonic
# coordinates is about `0.44 M` (`CLAUDE.md`), and the boost moves the
# centre by `|v| t`, which is a twentieth of a `M` at the time used here.
# Drawn from a seeded generator so that "random points" is reproducible and
# a failure can be replayed.
function gh_points(::Type{T}, n::Int; seed::Int=20260916) where {T}
    rng = MersenneTwister(seed)
    return ntuple(n) do _
        u = SVector{3,T}(2 * rand(rng, T) - 1, 2 * rand(rng, T) - 1,
                         2 * rand(rng, T) - 1)
        r = sqrt(sum(abs2, u))
        # Rescale onto a radius in [5/2, 5]; a draw at the origin is
        # impossible in practice and would be caught by the shell bound.
        return (T(5//2) * (1 + rand(rng, T)) / r) * u
    end
end

const GH_TIME = 1 // 4                  # a time at which nothing is stationary

# The damping parameters used wherever a test wants the Gundlach–Pretorius
# term switched on. `γ0 = 1` is `CODE.md`'s `γ0 ≈ 1/M` at `M = 1`;
# `γ2 = −1/2` is inside the continuum requirement `γ2 > −1` and is not zero,
# so a term that dropped the trace part would show.
gh_γ0(::Type{T}) where {T} = one(T)
gh_γ2(::Type{T}) where {T} = T(-1//2)

"""
The analytic state `(h, Π, ∂h)` of a background at `(t, x)`.

`Π` is the **densitised, Lie-advected** momentum
`Π_ab = (√γ/α)(∂_t − β^i ∂_i) g_ab`, built from the analytic `∂_t g` and
the analytic spatial gradients — not `∂_t h`, which is the first evolution
equation. `dmetric` returns `dg[a, b, c] = ∂_c g_ab`, the derivative axis
*last*, so the conversion into the packed, derivative-axis-first form this
package's algebra speaks happens here; on the mesh it happens in
`initialdata.jl` and nowhere else (`CLAUDE.md`).

Generic in the element type of `x` so that it can be called with dual
numbers: the checks below differentiate it.
"""
function gh_state(bg, t, x::SVector{3,D}) where {D}
    p = SVector{4,D}(D(t), x[1], x[2], x[3])
    g, dg = dmetric(bg, p)
    h = pack_g(g)
    ∂ₜg = pack_sym(SMatrix{4,4,D}(dg[a, b, 1] for a in 1:4, b in 1:4))
    ∂h = ntuple(Val(3)) do i
        pack_sym(SMatrix{4,4,D}(dg[a, b, i + 1] for a in 1:4, b in 1:4))
    end
    _, _, α, β, _, sqrtγ = metric_quantities(_sym4(h))
    Π = (sqrtγ / α) * (∂ₜg - β[1] * ∂h[1] - β[2] * ∂h[2] - β[3] * ∂h[3])
    return h, Π, ∂h
end

"""
The prescribed gauge source of a background: the lowered `H_b = g_bc H^c`
with `H^c = −Γ^c`, and its gradient `dHl[a, b] = ∂_a H_b`.

`H ≡ 0` for a harmonic background, and `SpacetimeMetrics` returns exactly
that; the call is made anyway so that the harmonic cases run the same code
path, which is what makes "the gauge-source term vanishes" a claim rather
than an assumption.
"""
gh_gauge(bg, t, x::SVector{3,D}) where {D} =
    gauge_source_grad(bg, SVector{4,D}(D(t), x[1], x[2], x[3]))

"""
The analytic fluxes `F^i_ab = β^i Π_ab + α√γ γ^{ij} ∂_j h_ab` at `(t, x)`,
spelled as `gh_node_rhs` spells them and generic in the element type of `x`
so that the divergence can be taken by automatic differentiation rather
than by a difference.

Written out rather than read off `gh_node_rhs`'s return value on purpose.
The flux involves neither the gauge source nor the reduced source, but
`gh_node_rhs` computes both — four rank-three contractions over a
four-dimensional index — and under the dual numbers the divergence needs,
compiling that is minutes rather than seconds.
`pointwise_identity_tests.jl` asserts that this shorter route gives
`gh_node_rhs`'s fluxes to roundoff, so the identity tests still measure
the ported function's flux and not a second opinion about it.
"""
function gh_fluxes(bg, t, x::SVector{3,D}) where {D}
    return gh_fluxes_of(gh_state(bg, t, x)...)
end

# The same, from a state already in hand — `gh_derivatives` has one and
# would otherwise evaluate the background twice under dual numbers.
@inline function gh_fluxes_of(h::SVector{NC,D}, Π::SVector{NC,D},
                              ∂h::NTuple{3,SVector{NC,D}}) where {D}
    _, _, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    a_mul = α * sqrtγ
    return ntuple(Val(3)) do i
        β[i] * Π + a_mul * (γu[i, 1] * ∂h[1] + γu[i, 2] * ∂h[2] +
                            γu[i, 3] * ∂h[3])
    end
end

"""
Everything the expanded right-hand side needs that is a derivative *of* the
analytic data, in one forward-mode pass: `∂Π[i] = ∂_i Π`, the second
derivatives `∂∂h` packed in `gh_node_rhs_expanded`'s column-major
lower-triangular order `(xx, xy, xz, yy, yz, zz)`, and the exact flux
divergence `∂_i F^i` the flux form is assembled from.

One pass and not three. Each of these is a derivative of a quantity that
`gh_state` already computes with a dual pass of its own, so taking them
costs a second dual layer — the same depth as `ddmetric` — and the compiler
pays for that depth once per background and per element type. Asking for
them separately compiled the same nested pass three times and was two
thirds of this file's cost **(measured in step 1)**.

The second derivative is symmetrised over its index pair, as `ddmetric`
symmetrises its own; `pointwise_tests.jl` checks this route against
`ddmetric` on one background, which is the independent confirmation that
the pair is packed the way `gh_node_rhs_expanded` reads it. (`ddmetric`
is also what `prerequisite_tests.jl` lists as a name the design calls, so
the check keeps it exercised rather than merely exported.)
"""
function gh_derivatives(bg, t, x::SVector{3,T}) where {T}
    function f(y::SVector{3,D}) where {D}
        h, Π, ∂h = gh_state(bg, t, y)
        F = gh_fluxes_of(h, Π, ∂h)
        return vcat(Π, ∂h[1], ∂h[2], ∂h[3], F[1], F[2], F[3])
    end
    J = ForwardDiff.jacobian(f, x)          # J[v, i] = ∂_i f[v]
    ∂Π = ntuple(i -> SVector{NC,T}(J[v, i] for v in 1:NC), Val(3))
    # Row block `NC + (j−1)NC` is `∂_j h`, so its column `i` is `∂_i ∂_j h`.
    ∂∂ = (j, i) -> SVector{NC,T}(J[NC + (j - 1) * NC + v, i] for v in 1:NC)
    pairs = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    ∂∂h = ntuple(Val(6)) do n
        j, k = pairs[n]
        (∂∂(j, k) + ∂∂(k, j)) / 2
    end
    divF = SVector{NC,T}(J[4NC + v, 1] + J[5NC + v, 2] + J[6NC + v, 3]
                         for v in 1:NC)
    return ∂Π, ∂∂h, divF
end

"""
The analytic second spatial derivatives `∂∂h` by the other route, through
`SpacetimeMetrics`' own second-derivative pass.

`ddmetric` returns `ddg[a, b, c, d] = ∂_d ∂_c g_ab`, with both derivative
axes trailing and the pair symmetrised for us. It is the independent
opinion about the packing that [`gh_derivatives`](@ref) is checked against,
on one background, in `pointwise_tests.jl`; running it on all of them would
compile a second nested pass per background for no further claim.
"""
function gh_second_derivatives(bg, t, x::SVector{3,T}) where {T}
    p = SVector{4,T}(T(t), x[1], x[2], x[3])
    _, _, ddg = ddmetric(bg, p)
    pairs = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    return ntuple(Val(6)) do n
        j, k = pairs[n]
        pack_sym(SMatrix{4,4,T}(ddg[a, b, j + 1, k + 1] for a in 1:4, b in 1:4))
    end
end

"""
`‖a − b‖∞`, the difference two tests compare against a scale they name.

A *relative* difference is the wrong instrument almost everywhere in these
tests, and silently so. Half of what is compared here is zero on the
background it is evaluated on — `∂_iβ^j` on the gauge wave, `∂_tΠ` on a
static hole — while the terms that produce it are O(1), so dividing by the
result turns two roundoff-sized numbers into a ratio of order one and calls
a correct answer a failure. Each assertion therefore states the magnitude
its tolerance is measured against: the size of the terms that cancelled,
not the size of what is left.
"""
absdiff(a, b) = maximum(abs, a .- b)

"‖a − b‖∞ / ‖b‖∞, for the few places where the reference really is O(1)."
function reldiff(a, b)
    d = absdiff(a, b)
    s = maximum(abs, b)
    return s == 0 ? d : d / s
end
