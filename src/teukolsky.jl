# The Teukolsky wave (step B5): the linearised even-parity `ℓ = 2`, `m = 0`
# quadrupole wave of Teukolsky (1982), `g = η + h`, regular at the origin —
# the wave step B5 sends through the cube and the shells, and the one step
# B6's `Ψ₄` is checked on. See `CODE.md`, "The Teukolsky wave".
#
# Teukolsky's metric in spherical coordinates, for a wave of `F(t − r)`:
#
#     ds² = −dt² + (1 + A f_rr) dr² + 2B f_rθ r dr dθ
#           + (1 + C f¹_θθ + A f²_θθ) r² dθ² + (1 + C f¹_φφ + A f²_φφ) r² sin²θ dφ²
#
# with, for `m = 0`, `f_rr = 2 − 3 sin²θ`, `f_rθ = −3 sinθ cosθ`,
# `f¹_θθ = 3 sin²θ`, `f²_θθ = −1`, `f¹_φφ = −3 sin²θ`, `f²_φφ = 3 sin²θ − 1`,
# and `A`, `B`, `C` the combinations of `F⁽ⁿ⁾/rᵖ` below. The ingoing wave is
# the time reverse, `F(t − r) → F(−t − r)`, which flips the terms with an odd
# derivative; the solution regular at `r = 0` is the outgoing wave of `F` plus
# the ingoing wave of `−F`, whose singular terms cancel. Every term has
# `n + p = 5`, so the Taylor series in `r` about the origin is
# `A = Σ_m κ_A(m) F⁽ᵐ⁺⁵⁾(t) rᵐ`, even in `r`, which is how the metric is
# evaluated near the origin, where the closed form is a cancellation of
# `1/r⁵` terms.

"""
    TeukolskyWave(T = Float64; amplitude, width)

The linearised even-parity `ℓ = 2`, `m = 0` quadrupole wave of Teukolsky
(1982), regular at the origin and time symmetric about `t = 0`: the metric
`g = η + h`, `h_tμ = 0`, with `F(x) = (a λ⁵/24) (x/λ) e^{−x²/λ²}`, `a` the
`amplitude` and `λ` the `width`. At the origin `h = A(t, 0) diag(−1, −1, 2)`
with `A(0, 0) = −(2/5) F⁽⁵⁾(0) = −a`, so `a` is the size of `h`, which is
`a diag(1, 1, −2)` there at `t = 0`.

**It solves the vacuum equations to first order in `a` only.** Its Ricci
tensor and its harmonic gauge source `Γ_a` are `O(a²)`: the metric is in the
transverse-traceless gauge, which is de Donder's at first order. So it is a
test of a scheme at an amplitude where `a²` is below the truncation error,
and [`isharmonic`](@ref) is `true` for it, which is right to that order and
gives the case no gauge source; `test/teukolsky_tests.jl` measures both
`O(a²)` claims.

An `AbstractMetric` of `SpacetimeMetrics`, `isbits`, generic in the number
type of its argument, so that `dmetric`'s dual pass differentiates it. Away
from the origin it is the closed form; within `width/2` of it the series in
`r²` (`TEUKOLSKY_SERIES` terms, which agree with the closed form there to
roundoff, a test). At the origin itself it is the limit, whose spatial
gradient is zero because `h` is even in `x`.
"""
struct TeukolskyWave{T} <: AbstractMetric
    amplitude::T
    width::T
end

function TeukolskyWave(::Type{T}=Float64; amplitude, width) where {T}
    width > 0 || throw(ArgumentError(
        "the Teukolsky wave's width λ is the scale of F(x) = (a λ⁵/24) (x/λ) " *
        "e^{−x²/λ²} and must be positive, got $width"))
    return TeukolskyWave{T}(T(amplitude), T(width))
end

Base.nameof(m::TeukolskyWave) =
    "Teukolsky wave (ℓ = 2, m = 0, a = $(m.amplitude), λ = $(m.width))"

# The terms of `A`, `B` and `C`: `c [F⁽ⁿ⁾(t − r) + s F⁽ⁿ⁾(t + r)] / rᵖ`, as
# `(c, n, p, s)`, for the regular superposition.
const TEUKOLSKY_A = ((3 // 1, 2, 3, -1), (9 // 1, 1, 4, 1), (9 // 1, 0, 5, -1))
const TEUKOLSKY_B = ((-1 // 1, 3, 2, 1), (-3 // 1, 2, 3, -1), (-6 // 1, 1, 4, 1),
                     (-6 // 1, 0, 5, -1))
const TEUKOLSKY_C = ((1 // 4, 4, 1, -1), (1 // 2, 3, 2, 1), (9 // 4, 2, 3, -1),
                     (21 // 4, 1, 4, 1), (21 // 4, 0, 5, -1))

# The number of even powers `r^{2j}`, `j = 0 … J − 1`, of the series about the
# origin. At `r = λ/2` the last term is about `10⁻¹⁵` of the first.
const TEUKOLSKY_SERIES = 15

# The coefficient of `rᵐ` in the series of one of the term lists, which
# multiplies `F⁽ᵐ⁺⁵⁾(t)`: `Σ c [(−1)^{m+p} + s] / (m + p)!`, exactly.
function teukolsky_kappa(terms, m::Integer)
    κ = 0 // 1
    for (c, n, p, s) in terms
        n + p == 5 || error("internal error: a Teukolsky term with n + p ≠ 5")
        κ += c * ((-1)^(m + p) + s) // factorial(big(m + p))
    end
    return κ
end

# The series' coefficients as `Float64`s, `κ(2j)` for `j = 0 … J − 1`, per list.
const TEUKOLSKY_KAPPA = ntuple(Val(3)) do k
    terms = (TEUKOLSKY_A, TEUKOLSKY_B, TEUKOLSKY_C)[k]
    ntuple(j -> Float64(teukolsky_kappa(terms, 2(j - 1))), Val(TEUKOLSKY_SERIES))
end

# `F⁽ᵏ⁾(x)` for `k = 0 … K − 1` at one `x`: with `u = x/λ`,
# `F⁽ᵏ⁾ = (a/24) λ^{5−k} (−1)ᵏ ½ H_{k+1}(u) e^{−u²}`, `H` the physicists' Hermite
# polynomials by their recurrence `H_{n+1} = 2u H_n − 2n H_{n−1}`.
@inline function teukolsky_derivatives(m::TeukolskyWave, x, ::Val{K}) where {K}
    λ = oftype(x, m.width)
    u = x / λ
    scale = oftype(x, m.amplitude) * λ^5 * exp(-u * u) / 48
    return _teukolsky_chain(Val(K), u, one(u), 2u, scale, λ, 0)
end

# `F⁽ᵏ⁾, F⁽ᵏ⁺¹⁾, …` from `H_k`, `H_{k+1}` and the scale of `F⁽ᵏ⁾`, by recursion
# rather than a loop into a tuple, so that nothing captured is reassigned.
@inline _teukolsky_chain(::Val{0}, u, hp, hc, s, λ, k) = ()
@inline function _teukolsky_chain(::Val{K}, u, hp, hc, s, λ, k) where {K}
    val = isodd(k) ? -s * hc : s * hc
    return (val, _teukolsky_chain(Val(K - 1), u, hc, 2u * hc - 2(k + 1) * hp, s / λ, λ,
                                  k + 1)...)
end

# `A`, `B`, `C` at `(t, r)` from the closed form, `r > 0`.
@inline function teukolsky_abc_closed(m::TeukolskyWave, t, r)
    Fm = teukolsky_derivatives(m, t - r, Val(5))
    Fp = teukolsky_derivatives(m, t + r, Val(5))
    rp = ntuple(p -> r^p, Val(5))
    term(c, n, p, s) = oftype(r, c) * (Fm[n + 1] + s * Fp[n + 1]) / rp[p]
    A = term(TEUKOLSKY_A[1]...) + term(TEUKOLSKY_A[2]...) + term(TEUKOLSKY_A[3]...)
    B = term(TEUKOLSKY_B[1]...) + term(TEUKOLSKY_B[2]...) + term(TEUKOLSKY_B[3]...) +
        term(TEUKOLSKY_B[4]...)
    C = term(TEUKOLSKY_C[1]...) + term(TEUKOLSKY_C[2]...) + term(TEUKOLSKY_C[3]...) +
        term(TEUKOLSKY_C[4]...) + term(TEUKOLSKY_C[5]...)
    return A, B, C
end

# `A`, `B`, `C` at `(t, r)` from the series in `r²` about the origin.
@inline function teukolsky_abc_series(m::TeukolskyWave, t, r2)
    F = teukolsky_derivatives(m, t, Val(2TEUKOLSKY_SERIES + 4))
    A, B, C = zero(r2), zero(r2), zero(r2)
    ρ = one(r2)
    for j in 1:TEUKOLSKY_SERIES
        f = F[2(j - 1) + 6] * ρ         # F⁽²⁽ʲ⁻¹⁾⁺⁵⁾ r^{2(j−1)}
        A += oftype(r2, TEUKOLSKY_KAPPA[1][j]) * f
        B += oftype(r2, TEUKOLSKY_KAPPA[2][j]) * f
        C += oftype(r2, TEUKOLSKY_KAPPA[3][j]) * f
        ρ *= r2
    end
    return A, B, C
end

"""
    teukolsky_abc(m::TeukolskyWave, t, r) -> (A, B, C)

Teukolsky's radial functions of the regular wave at `(t, r)`: the closed form
for `r ≥ width/2`, the series in `r²` inside.
"""
@inline function teukolsky_abc(m::TeukolskyWave, t, r)
    rs = oftype(r, m.width) / 2
    return r < rs ? teukolsky_abc_series(m, t, r * r) : teukolsky_abc_closed(m, t, r)
end

function SpacetimeMetrics.metric(m::TeukolskyWave, p::AbstractVector)
    p = SVector{4}(p)
    t, x, y, z = p
    R = typeof(x)
    r2 = x * x + y * y + z * z
    o, ze = one(R), zero(R)
    if iszero(r2)
        # The limit at the origin, `A(t, 0) (3 ẑẑ − δ)`: `h` is even in `x`, so its
        # gradient there is zero, which a constant in `x` gives the dual pass.
        A0, _, _ = teukolsky_abc_series(m, t + ze, r2)
        return SMatrix{4,4,R}(-o, ze, ze, ze,
                              ze, o - A0, ze, ze,
                              ze, ze, o - A0, ze,
                              ze, ze, ze, o + 2A0)
    end
    r = sqrt(r2)
    A, B, C = r < oftype(r, m.width) / 2 ? teukolsky_abc_series(m, t + ze, r2) :
              teukolsky_abc_closed(m, t + ze, r)
    X = (x, y, z)
    c2 = z * z / r2                                 # cos²θ
    v = (z * x / r2, z * y / r2, (z * z - r2) / r2)  # sinθ θ̂
    arr = A * (3c2 - 1) / r2                        # rr, on x_i x_j
    arθ = -3B * z / r2                              # rθ, on x_i v_j + v_i x_j
    aφ = 3(A - C) * (1 - c2) - A                    # on δ_ij − x_i x_j / r²
    avv = 3(2C - A)                                 # on v_i v_j
    # Each component once, so that `h` is symmetric to the bit.
    hs(i, j) = (arr - aφ / r2) * X[i] * X[j] + arθ * (X[i] * v[j] + v[i] * X[j]) +
               (i == j ? aφ : ze) + avv * v[i] * v[j]
    h11, h12, h13 = hs(1, 1), hs(1, 2), hs(1, 3)
    h22, h23, h33 = hs(2, 2), hs(2, 3), hs(3, 3)
    return SMatrix{4,4,R}(-o, ze, ze, ze,
                          ze, o + h11, h12, h13,
                          ze, h12, o + h22, h23,
                          ze, h13, h23, o + h33)
end

# Harmonic to first order in the amplitude: the transverse-traceless gauge is de
# Donder's at that order, and `Γ_a = O(a²)` (measured in
# `test/teukolsky_tests.jl`). See [`TeukolskyWave`](@ref).
isharmonic(::TeukolskyWave) = true
