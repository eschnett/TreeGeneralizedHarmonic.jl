# The fitted target: a regular fit of the state on the offset surface, which
# the layer of step 8e's `:fitted` variant relaxes toward.
#
# `CODE.md`, "The interior" — "The fitted target (added in step 8e)" — and
# `PLAN.md`'s step 8e. Step 5's layer relaxes toward the analytic solution,
# which needs an analytic interior whose singular set a core can contain;
# harmonic Kerr at `a = 9/10` has none that fits. The fitted target needs no
# interior at all: it is a **polynomial in `x`** — regular everywhere,
# including the center — fitted by least squares to the state (or to the
# analytic solution, for the initial data) on the tracked geometry's offset
# surface `r_1(n̂) = r_h(n̂) − m h`, where the state is still the solution.
# This file is the host half (8e-i): the samplers, the fit, its validity
# sweep, and the pointwise evaluator a kernel will call (8e-ii).
#
# Four things are easy to get wrong, and each is written out where it
# happens:
#
#   * **The fit is made in ADM variables, never in `g_ab`** (`PLAN.md`'s
#     finding 2). The Lorentzian metrics are not convex in `g_ab`: the
#     angular mean of Kerr-Schild `g_ab` on the sphere `r = 1.15 M` has
#     Euclidean signature, and a least-squares fit is a weighted mean. The
#     twenty variables are `(log α, β^i, γ_ij, Π_ab)`: `α = exp(log α)` is
#     positive by construction, `γ` is convex (a mean of positive-definite
#     matrices is one), `β` is unconstrained, and `g_ab` is reassembled.
#   * **The shift keeps its constant term (decided in review, step 8e).**
#     `PLAN.md` asked for `l ≥ 1` only, so that `β` vanishes at the center;
#     a *boosted* hole's shift has an `l = 0` part on the offset surface —
#     its angular mean is `0.17` against a mean `|β|` of `0.63` for
#     `boost(KerrSchild(1, 0), 0.3 x̂)` — which neither `l ≥ 1` nor a slope
#     `2B₀` of `ρ² ỹ_00` can match (measured in step 8e: the state on the
#     surface `1.3e−2` off without the constant, `1.9e−5` with it). Validity
#     does not need `β(0) = 0` — any shift with `α > 0` and `γ ≻ 0` is a
#     Lorentzian metric — and the static hole is the same fit to roundoff
#     either way. `shift_constant = false` is the brief's ansatz, kept as a
#     switch: no `ρ⁰ ỹ_00` term, the `ρ²`, `ρ⁴` ones kept.
#   * **One QR serves both designs.** Without its constant the shift's
#     columns are the scalars' minus the constant, which is ordered *last*:
#     Householder QR without pivoting factors the leading columns first, so
#     the leading block of `R` is the shift's own factorization and its
#     solution is the least-squares one, not a truncation of the scalars'.
#   * **The evaluator is the kernel's.** [`fit_variables_at`](@ref) and
#     [`fit_state`](@ref) are `@inline` functions on scalars and `SVector`s
#     over a coefficient array — no allocation, no `return` in the body,
#     generic in `T` — so step 8e-ii's kernel calls the same arithmetic the
#     tests hold against the least-squares model here.

# --- the fit's variables -------------------------------------------------------

# The twenty fitted variables, in slots: `log α`, the contravariant shift
# `β^i` (three), the spatial metric `γ_ij` **as its offset** `γ_ij − δ_ij`
# in `(xx, xy, xz, yy, yz, zz)` (six — `h`'s own spatial slots), and `Π_ab`
# in the packed order (ten).
const NFIT = 2NC
const FIT_LAPSE = 1
const FIT_SHIFT = 2:4
const FIT_METRIC = 5:10
const FIT_MOMENTUM = 11:20
const FIT_GROUPS = (FIT_LAPSE:FIT_LAPSE, FIT_SHIFT, FIT_METRIC, FIT_MOMENTUM)

# The spatial block of a packed offset metric as a full `γ_ij`, or — without
# the `δ` — of a packed derivative.
@inline _spatial(u, ::Val{true}) =
    SMatrix{3,3}(one(eltype(u)) + u[5], u[6], u[7], u[6], one(eltype(u)) + u[8],
                 u[9], u[7], u[9], one(eltype(u)) + u[10])
@inline _spatial(u, ::Val{false}) =
    SMatrix{3,3}(u[5], u[6], u[7], u[6], u[8], u[9], u[7], u[9], u[10])
@inline _lowered_shift(u) = SVector{3}(u[2], u[3], u[4])

# `(log α, β^i, γ_ij − δ_ij, Π_ab)` from the split's pieces.
@inline _fit_vector(lα, β, u) =
    SVector{NFIT}(lα, β[1], β[2], β[3], u[5], u[6], u[7], u[8], u[9], u[10],
                  u[11], u[12], u[13], u[14], u[15], u[16], u[17], u[18],
                  u[19], u[20])

"""
    fit_variables(u; tilde = false) -> v
    fit_variables(u, u′; tilde = false) -> (v, v′)
    fit_variables(u, u′, u″; tilde = false) -> (v, v′, v″)

The fit's twenty variables of the packed state `u = (h_ab, Π_ab)`
(`SVector{20}`), and — given its first and second derivatives along a
direction, `u′` and `u″` — theirs, **by the chain rule** (proposed in step
8e): with `γ_ij = δ_ij + h_ij`, the lowered shift `b_i = h_ti`, `β^i =
γ^{ij} b_j`, `β·b = b_iβ^i` and `α² = (1 − h_tt) + β·b`,

    β′ = γ⁻¹ (b′ − γ′β),           β″ = γ⁻¹ (b″ − γ″β − 2γ′β′),
    (β·b)′ = b′·β + b·β′,          (β·b)″ = b″·β + 2 b′·β′ + b·β″,
    (log α)′ = (α²)′/(2α²),        (log α)″ = (α²)″/(2α²) − ((α²)′)²/(2(α²)²),

and `γ_ij`, `Π_ab` linear. The variables are `(log α, β^i, γ_ij − δ_ij,
Π_ab)` — `γ` held as its offset so that near-flat data keeps its digits
(the fit is the same function either way: the ansatz has a constant).

**`β^i` and not `β_i` (proposed in step 8e).** Either is a valid fit
variable — any shift with `α > 0` and `γ ≻ 0` is a Lorentzian metric — but
the contravariant one is the 3+1 split's own shift and reassembles without
an inverse: `β_i = γ_ij β^j` and `g_tt = −α² + β^iβ_i` are products, where
`β_i` would need `γ^{ij}` at every layer point of every evaluation.

**`tilde = true` fits `Π̃_ab = (α/√γ) Π_ab` in slots 11–20 instead of
`Π_ab` (added in step 8f)** — the momentum as the first evolution equation
adds it to `∂_t h`, and the quantity the range projection's `K_max` bounds.
The densitised `Π` carries `√γ/α`, which on harmonic Kerr's equator grows
by three orders of magnitude between the axis and the ring, and a polynomial
of degree `L + 2` cannot follow it (`CODE.md`, "The fitted target", piece
12). With `s = α/√γ`, `(log s)′ = (log α)′ − tr(γ⁻¹γ′)/2`,
`(log s)″ = (log α)″ − (tr(γ⁻¹γ″) − tr(γ⁻¹γ′γ⁻¹γ′))/2`, `s′ = s (log s)′`,
`s″ = s ((log s)″ + (log s)′²)`, and `Π̃′ = s′Π + sΠ′`,
`Π̃″ = s″Π + 2s′Π′ + sΠ″`.

Host-side and at fit cadence. The sample must be a metric (`α² > 0`,
`γ ≻ 0`); [`build_fit`](@ref) checks that before it calls this.
"""
function fit_variables(u::SVector{NFIT,T}; tilde::Bool=false) where {T}
    v, _, _ = _fit_variables(u, nothing, nothing, tilde)
    return v
end

function fit_variables(u::SVector{NFIT,T}, u1::SVector{NFIT,T};
                       tilde::Bool=false) where {T}
    v, v1, _ = _fit_variables(u, u1, nothing, tilde)
    return v, v1
end

fit_variables(u::SVector{NFIT,T}, u1::SVector{NFIT,T}, u2::SVector{NFIT,T};
              tilde::Bool=false) where {T} = _fit_variables(u, u1, u2, tilde)

# The momentum's ten components of a packed state or of its derivative.
@inline _momentum(u) = SVector{NC}(ntuple(k -> u[NC + k], Val(NC)))

# Replace the momentum slots of a fit vector.
@inline _with_momentum(v::SVector{NFIT,T}, Π) where {T} =
    SVector{NFIT,T}(ntuple(k -> k ≤ NC ? v[k] : Π[k - NC], Val(NFIT)))

function _fit_variables(u::SVector{NFIT,T}, u1, u2, tilde::Bool) where {T}
    γ = _spatial(u, Val(true))
    γu = inv(γ)
    b = _lowered_shift(u)
    β = γu * b
    α² = (one(T) - u[1]) + dot(b, β)
    v = _fit_vector(log(α²) / 2, β, u)
    s = tilde ? sqrt(α² / det(γ)) : one(T)
    tilde && (v = _with_momentum(v, s * _momentum(u)))
    u1 === nothing && return v, nothing, nothing
    γ1 = _spatial(u1, Val(false))
    b1 = _lowered_shift(u1)
    β1 = γu * (b1 - γ1 * β)
    α²1 = -u1[1] + (dot(b1, β) + dot(b, β1))
    lα1 = α²1 / (2α²)
    v1 = _fit_vector(lα1, β1, u1)
    A1 = γu * γ1
    ls1 = lα1 - tr(A1) / 2
    s1 = s * ls1
    tilde && (v1 = _with_momentum(v1, s1 * _momentum(u) + s * _momentum(u1)))
    u2 === nothing && return v, v1, nothing
    γ2 = _spatial(u2, Val(false))
    b2 = _lowered_shift(u2)
    β2 = γu * (b2 - γ2 * β - 2 * (γ1 * β1))
    α²2 = -u2[1] + (dot(b2, β) + 2 * dot(b1, β1) + dot(b, β2))
    lα2 = α²2 / (2α²) - α²1 * α²1 / (2 * α² * α²)
    v2 = _fit_vector(lα2, β2, u2)
    if tilde
        ls2 = lα2 - (tr(γu * γ2) - tr(A1 * A1)) / 2
        s2 = s * (ls2 + ls1 * ls1)
        v2 = _with_momentum(v2, s2 * _momentum(u) + 2 * s1 * _momentum(u1) +
                                s * _momentum(u2))
    end
    return v, v1, v2
end

"""
    state_from_fit(v, tilde = false) -> (h, Π)

The packed state of the fit's variables: `α² = exp(2 log α)`,
`γ_ij = δ_ij + v_ij`, `β_i = γ_ij β^j`, `h_tt = (1 − α²) + β_iβ^i`,
`h_ti = β_i`, `h_ij = v_ij`, and `Π_ab` as it is — or, where the fit holds
`Π̃ = (α/√γ)Π` (`tilde`, added in step 8f), `Π = (√det γ/α) Π̃`. The inverse
of [`fit_variables`](@ref), inverse-free, explicit scalar arithmetic,
`@inline`, allocation-free and generic in `T` — the reassembly step 8e-ii's
kernel runs at every layer point.
"""
@inline function state_from_fit(v::SVector{NFIT,T}, tilde::Bool=false) where {T}
    o = one(T)
    α² = exp(2 * v[1])
    β1 = v[2]
    β2 = v[3]
    β3 = v[4]
    g11 = o + v[5]
    g12 = v[6]
    g13 = v[7]
    g22 = o + v[8]
    g23 = v[9]
    g33 = o + v[10]
    b1 = g11 * β1 + g12 * β2 + g13 * β3
    b2 = g12 * β1 + g22 * β2 + g23 * β3
    b3 = g13 * β1 + g23 * β2 + g33 * β3
    bb = b1 * β1 + b2 * β2 + b3 * β3
    h = SVector{NC,T}((o - α²) + bb, b1, b2, b3, v[5], v[6], v[7], v[8], v[9],
                      v[10])
    detγ = g11 * (g22 * g33 - g23 * g23) - g12 * (g12 * g33 - g23 * g13) +
           g13 * (g12 * g23 - g22 * g13)
    # A fit that is not a metric has no `√det γ`: its momentum comes back
    # non-finite, which the validity sweep counts and `bounds_project` repairs.
    f = !tilde ? o : detγ > zero(T) ? sqrt(detγ / α²) : zero(T) / zero(T)
    Π = SVector{NC,T}(ntuple(k -> f * v[NC + k], Val(NC)))
    return h, Π
end

# --- the solid real harmonics --------------------------------------------------

# One `(l, m)` of the recurrence handed to the fold's function: the `l0`
# slot, or the cosine and the sine slot of `|m|`, with 8d's real harmonics
# `ỹ^c = √2 Re Y`, `ỹ^s = −√2 Im Y` (`real_harmonic_index`, `shape_series`).
@inline function _fold_term(f::F, acc, l::Int, m::Int, q, C, S, s2) where {F}
    if m == 0
        acc = f(acc, l * l + l + 1, l, q)
    else
        acc = f(acc, l * l + l + m + 1, l, s2 * q * C)
        acc = f(acc, l * l + l - m + 1, l, -(s2 * q * S))
    end
    return acc
end

"""
    _solid_harmonic_fold(f, acc, lmax, ξx, ξy, ξz, ρ²) -> acc

`acc = f(acc, slot, l, S)` for every real **solid** harmonic
`S_lm(ξ) = ρ^l ỹ_lm(ξ/ρ)` to degree `lmax` at `ξ` (`ρ² = |ξ|²`), in
[`real_harmonic_index`](@ref)'s slots and 8d's normalisation — the fit's
basis, and at `|ξ| = 1` exactly the harmonics [`shape_series`](@ref) sums.

It is [`shape_series`](@ref)'s recurrence with `ξ` in place of the unit
vector and the Legendre step's `q_{l−2}` term multiplied by `ρ²`:
`Q_lm = ρ^{l−m} q_lm(ξ_z/ρ)` obeys `Q_{m+1,m} = √(2m+3) ξ_z Q_mm` and
`Q_lm = α (ξ_z Q_{l−1,m} − β ρ² Q_{l−2,m})`, and `(ξ_x + i ξ_y)^m` carries
the rest. Every term is a polynomial in `ξ`, so there is no division, no
angle and no special case at the center — where only `S_00 = 1/√(4π)`
survives, which is what makes a fit without a constant term vanish there.

`@inline`, runtime loop bound, no allocation; a kernel may call it with an
`isbits` `f` (step 8e-ii).
"""
@inline function _solid_harmonic_fold(f::F, acc, lmax::Int, x::T, y::T, z::T,
                                      ρ²::T) where {F,T}
    s2 = sqrt(T(2))
    qmm = inv(sqrt(4 * T(π)))
    C = one(T)
    S = zero(T)
    m = 0
    while m ≤ lmax
        if m > 0
            qmm = -sqrt(T(2m + 1) / T(2m)) * qmm
            C, S = C * x - S * y, C * y + S * x
        end
        acc = _fold_term(f, acc, m, m, qmm, C, S, s2)
        if m + 1 ≤ lmax
            qa = qmm
            qb = sqrt(T(2m + 3)) * z * qmm
            acc = _fold_term(f, acc, m + 1, m, qb, C, S, s2)
            l = m + 2
            while l ≤ lmax
                α = sqrt(T(4 * l * l - 1) / T(l * l - m * m))
                β = sqrt(T((l - 1) * (l - 1) - m * m) /
                         T(4 * (l - 1) * (l - 1) - 1))
                qc = α * (z * qb - β * ρ² * qa)
                acc = _fold_term(f, acc, l, m, qc, C, S, s2)
                qa = qb
                qb = qc
                l += 1
            end
        end
        m += 1
    end
    return acc
end

"""
    real_solid_harmonics(lmax, ξ) -> Vector

Every real solid harmonic `S_lm(ξ) = |ξ|^l ỹ_lm(ξ/|ξ|)` to degree `lmax` at
the point `ξ`, in [`real_harmonic_index`](@ref)'s slots — one row of the
fit's design matrix, through the fold its evaluator uses. Host-side.
"""
function real_solid_harmonics(lmax::Integer, ξ)
    T = float(eltype(ξ))
    out = zeros(T, (lmax + 1)^2)
    x, y, z = T(ξ[1]), T(ξ[2]), T(ξ[3])
    _solid_harmonic_fold((acc, slot, l, s) -> (out[slot] = s; acc), nothing,
                         Int(lmax), x, y, z, x * x + y * y + z * z)
    return out
end

"""
    fit_directions(L) -> Vector{SVector{3,Float64}}

The collocation directions: the points of `EquiangularGrid(L)` —
`L + 1` midpoint colatitudes by `2L + 1` longitudes, no pole — as unit
vectors, in the grid's `CartesianIndices` order (`ash_point_coord`).
"""
function fit_directions(L::Integer)
    grid = EquiangularGrid(Int(L))
    dirs = SVector{3,Float64}[]
    for ij in CartesianIndices(ash_grid_size(grid))
        θ, φ = ash_point_coord(grid, ij)
        sθ, cθ = sincos(θ)
        sφ, cφ = sincos(φ)
        push!(dirs, SVector{3,Float64}(sθ * cφ, sθ * sφ, cθ))
    end
    return dirs
end

# --- the samplers ------------------------------------------------------------------

"""
    state_sampler(fs, q; t) -> StateSampler

The state on the mesh as the fit reads it: called as `sampler(xs, ns)` with
points `xs` and the unit radial vectors `ns` there, it returns `(u, ∂_r u)`
— two vectors of `SVector{20}`, the packed `(h, Π)` and its derivative
`n̂·∇u` along the ray — by [`interpolate_grad`](@ref) at order `q + 2`. `t`
is the time the field set holds; [`build_fit`](@ref) reads it.

**The footprint guard is off for this call, and that is safe (proposed in
step 8e).** The collocation points are *on* the offset surface, the
evolved region's boundary, so their `(q + 2)`-point windows reach `G h`
inside it, into the layer's outer part — which the horizon finder's guard
exists to refuse. The fit may read it because step 8c's rule puts the
relaxation there below `1/M`: the ramp `n_L ≥ 4G` cells means depth `G h` is
at most a quarter of it, where `ρ = ρ_max · smoothstep(1/4) = 0.10 ρ_max`,
`0.41/M` at the default `4/M` — the outer quarter of the layer is the
unmodified equations plus a weak relaxation toward a target that is itself a
fit of this state, and the data there is the solution continued smoothly
across `r_1`. `mask = AllPoints()` says so at the call.

**The ghosts must be filled first**, with the hook of the time `t` — the
windows of points near a block face reach into them (`CLAUDE.md`, "Ghosts
must be filled before anything is interpolated"). The radial derivative is
the interpolant's own gradient, one order behind its value: `O(h^{q+1})`.
The sampler provides no second derivative, so it serves `cont = 1` only.
"""
struct StateSampler{F,T}
    fs::F
    q::Int
    t::T
end

function state_sampler(fs::FieldSet{T,3}, q::Integer; t) where {T}
    fs.nvars == NFIT || throw(ArgumentError(
        "the fit samples the packed state (h, Π), $NFIT variables, but this " *
        "field set holds $(fs.nvars)."))
    return StateSampler{typeof(fs),T}(fs, Int(q), T(t))
end

function (s::StateSampler{F,T})(xs::AbstractVector, ns::AbstractVector) where {F,T}
    vals, grads = interpolate_grad(s.fs, xs; q=s.q, mask=AllPoints())
    u = [SVector{NFIT,T}(vals[i]) for i in eachindex(xs)]
    u1 = map(eachindex(xs)) do i
        n = ns[i]
        g = grads[i]
        SVector{NFIT,T}(T(n[1]) * g[1] + T(n[2]) * g[2] + T(n[3]) * g[3])
    end
    return (u, u1)
end

"""
    analytic_sampler(background, t; δ) -> AnalyticSampler

The analytic solution as the fit reads it — for the initial data and for
the tests: `sampler(xs, ns)` returns `(u, ∂_r u, ∂_r² u)`, the packed
[`state_tuple`](@ref) at each point and its first two derivatives along the
ray by **fourth-order central differences** with step `δ` (callers use
`δ = h/8`): `(u₋₂ − 8u₋₁ + 8u₁ − u₂)/(12δ)` and
`(−u₋₂ + 16u₋₁ − 30u₀ + 16u₁ − u₂)/(12δ²)` **(proposed in step 8e**: the
brief's central differences, at the order that puts the truncation at
`δ⁴ ≈ 10⁻⁸ M⁴` rather than `δ² ≈ 10⁻⁴`; the roundoff of the second
difference is `5 eps |u|/δ²`, `10⁻¹¹` at `Float64` and a percent at
`Float32` — measured, `1.7e−2` in the curvature rows of the fixture's
`cont = 2` fit — so a `Float32` run's initial-data fit wants its samples in
`Float64`**)**.

No core rule is applied: the offset surface lies `n_L h` outside the core
surface, and the analytic solution is regular there by construction of the
geometry — a singular point on it is a configuration error, which
[`build_fit`](@ref) refuses by checking that every sample is finite.
"""
struct AnalyticSampler{B,T}
    background::B
    t::T
    δ::T
end

function analytic_sampler(background, t; δ)
    T = float(typeof(δ))
    T(δ) > 0 || throw(ArgumentError(
        "the difference step must be positive, got δ = $δ; callers use h/8."))
    return AnalyticSampler{typeof(background),T}(background, T(t), T(δ))
end

function (s::AnalyticSampler{B,T})(xs::AbstractVector, ns::AbstractVector) where {B,T}
    δ = s.δ
    at(x, n, k) = SVector{NFIT,T}(state_tuple(s.background, s.t,
                                              ntuple(d -> T(x[d]) + (k * δ) * T(n[d]),
                                                     Val(3))))
    u0 = similar(xs, SVector{NFIT,T})
    u1 = similar(xs, SVector{NFIT,T})
    u2 = similar(xs, SVector{NFIT,T})
    for i in eachindex(xs)
        x, n = xs[i], ns[i]
        m2, m1, c0, p1, p2 = at(x, n, -2), at(x, n, -1), at(x, n, 0),
                             at(x, n, 1), at(x, n, 2)
        u0[i] = c0
        u1[i] = ((m2 - p2) + 8 * (p1 - m1)) / (12 * δ)
        u2[i] = (16 * (m1 + p1) - (m2 + p2) - 30 * c0) / (12 * δ * δ)
    end
    return (u0, u1, u2)
end

# --- the fit ---------------------------------------------------------------------

"""
    FitParams{T}(lmax, cont, rbar, center, bounds[, tilde = false])

The `isbits` half of an [`InteriorFit`](@ref): what an evaluation of the fit
needs besides its coefficient array — the degree `lmax`, the radial order
`cont`, the radius `r̄` the ansatz is scaled by, the [`HoleCenter`](@ref) it
is about (the track's, so that the fit moves with the hole), the
[`StateBounds`](@ref) [`fit_state`](@ref) projects its result into, and
whether the momentum slots hold `Π̃ = (α/√γ)Π` (`tilde`, added in step 8f). A kernel
argument in step 8e-ii, beside the coefficients.
"""
struct FitParams{T}
    lmax::Int
    cont::Int
    rbar::T
    center::HoleCenter{T}
    bounds::StateBounds{T}
    tilde::Bool
end

FitParams{T}(lmax, cont, rbar, center, bounds) where {T} =
    FitParams{T}(lmax, cont, rbar, center, bounds, false)

"""
    InteriorFit

A fitted target: the coefficients `coeffs[slot, k + 1, v]` of

    f_v(x) = Σ_{l,m} S_lm(ξ) Σ_{k=0}^{cont} C_{lm,k,v} ρ^{2k},
    ξ = (x − c(t))/r̄,  ρ = |ξ|,  S_lm(ξ) = ρ^l ỹ_lm(ξ/ρ),

for the twenty variables `v` of [`fit_variables`](@ref), as an array of
shape `((L+1)², cont+1, 20)` in `T` **on the backend** (`coeffs`, through
[`to_backend`](@ref) — `Hsrc`'s precedent) and on the host (`host`, the same
array on the CPU); the `isbits` [`FitParams`](@ref) (`params`: `L`, `cont`,
`r̄`, the center, the bounds); and what the fit was built from and what it
measured — `t` and `c` (the time and the center it was built at),
`points` (the collocation points) and `model` (the least-squares model's
values there, in the fit's variables), `residual`
([`fit_residual`](@ref)), `conditioning` (the
1-norm condition numbers of the scalars' and the shift's triangular
factors), `valid` ([`fit_valid`](@ref)) and `sweep` (the validity sweep's
worst numbers, [`fit_sweep`](@ref)).

Built by [`build_fit`](@ref); evaluated by [`fit_state`](@ref).
"""
struct InteriorFit{T,A,H}
    params::FitParams{T}
    coeffs::A
    host::H
    t::T
    c::SVector{3,T}
    points::Vector{SVector{3,T}}
    model::Vector{SVector{NFIT,T}}
    residual::NamedTuple
    conditioning::NamedTuple
    valid::Bool
    sweep::NamedTuple
end

"""
    fit_residual(fit) -> (; per_variable, value, overall)

How well the fit reproduces what it was fitted to, **relative, block by
block**: for each of the twenty variables the largest residual of each
block of rows — the values, the slopes `r̄ ∂_r`, the curvatures `r̄² ∂_r²` —
over the largest datum of *that block* in the variable's *group* (`log α`;
the shift; the spatial metric; the momentum), so that a component whose data
vanishes is measured against its siblings' scale and a steep curvature does
not hide a poor value; `per_variable` is the largest over the blocks,
`value` the value rows' worst over the variables, and `overall` the worst
of `per_variable`. The analysis row step 8e-ii writes is `overall`
(proposed in step 8e).
"""
fit_residual(fit::InteriorFit) = fit.residual

"""
    fit_valid(fit) -> Bool

Whether every point of the fit's validity sweep ([`fit_sweep`](@ref)) was a
Lorentzian metric: finite, `γ_ij` positive definite, `α > 0`.
"""
fit_valid(fit::InteriorFit) = fit.valid

"""
    fit_row_weights(L, cont) -> NTuple{cont + 1}

The weight of each block of the fit's rows: `P^{−b}` for the `b`-th radial
derivative, `P = L + 2 cont` the ansatz's degree in `ρ` — the factor by
which differentiating a degree-`P` polynomial along the ray can grow it, so
that every block of the design matrix is `O(1)` and the slopes and
curvatures do not outvote the values **(proposed in step 8e)**. Measured on
the analytic holes it cuts the condition number 4–20× and the value
residual 2–7× against unweighted rows, and changes neither validity nor a
consistent system's solution (`CODE.md`, "The fitted target").
"""
fit_row_weights(L::Integer, cont::Integer) =
    ntuple(b -> 1 // (Int(L) + 2 * Int(cont))^(b - 1), Int(cont) + 1)

# Where the coefficient of `(slot, k)` sits among the columns: natural order,
# `k` fastest, except that the constant `(1, 0)` is **last** — so that the
# shift, which has no constant, is fitted by the leading block of the same
# factorization.
@inline _fit_column(slot, k, nb, ncol) =
    (slot == 1 && k == 0) ? ncol : (slot - 1) * nb + k

"""
    solve_fit(ξs, samples, L, cont, rbar; weights = fit_row_weights(L, cont),
              shift_constant = true)
        -> (coeffs, residual, conditioning, model)

The least-squares system of [`build_fit`](@ref), given the collocation
points in the ansatz's coordinates `ξ_p = (x_p − c)/r̄` and the samples in
the fit's variables — `samples = (v, v′[, v″])`, vectors of `SVector{20}`
with the derivatives along the ray, unscaled.

Rows: the values `f(ξ_p) = v_p`, the radial derivatives
`∂_ρ f(ξ_p) = r̄ v′_p` and, for `cont = 2`, `∂_ρ² f(ξ_p) = r̄² v″_p` — with
`∂_ρ (S_lm ρ^{2k}) = (l + 2k) S_lm ρ^{2k}/ρ` along the ray, so that the
basis is differentiated exactly. Columns: `(L + 1)² (cont + 1)`, the
constant last. **One Householder QR** (`LinearAlgebra.qr`, no pivoting) for
all twenty right-hand sides: the seventeen scalar and tensor variables are
solved with the whole triangular factor, and the shift either so or with its leading
block, which *is* the factorization of the shift's columns, when
`shift_constant = false`; by default the shift keeps its constant and is
solved like the others (see `fit.jl`'s header: what a moving hole needs). Each block of
rows is weighted by [`fit_row_weights`](@ref), and within a block the
points are weighted alike, so the residual the solution minimises is the
one at the collocation points, which is where [`fit_residual`](@ref)
reports it, block by block.

`model` is the least-squares model's own values at the collocation points
— the value rows of the solved system, in the fit's variables — which is
what [`fit_variables_at`](@ref) must reproduce there to roundoff.

BLAS's threads are not `julia -t`'s: the factorization is the same at every
Julia thread count, and nothing here sets either.
"""
function solve_fit(ξs::AbstractVector{SVector{3,T}}, samples, L::Integer,
                   cont::Integer, rbar; weights=fit_row_weights(L, cont),
                   shift_constant::Bool=true) where {T}
    np = length(ξs)
    NS = (L + 1)^2
    nb = cont + 1
    ncol = NS * nb
    nrow = np * nb
    length(samples) ≥ nb || throw(ArgumentError(
        "a fit with cont = $cont needs the values and $cont radial " *
        "derivative(s) at each point, got $(length(samples)) sample sets: " *
        "the state sampler provides one derivative and serves cont = 1; " *
        "cont = 2 is the analytic sampler's."))
    nrow ≥ ncol || throw(ArgumentError(
        "$nrow rows cannot determine $ncol coefficients: EquiangularGrid(L) " *
        "has (L+1)(2L+1) points and the ansatz (L+1)² (cont+1) terms."))
    rb = T(rbar)
    ls = [isqrt(slot - 1) for slot in 1:NS]
    X = zeros(T, nrow, ncol)
    Y = zeros(T, nrow, NFIT)
    for p in 1:np
        ξ = ξs[p]
        ρ = sqrt(ξ[1] * ξ[1] + ξ[2] * ξ[2] + ξ[3] * ξ[3])
        S = real_solid_harmonics(L, ξ)
        for slot in 1:NS, k in 0:cont
            j = _fit_column(slot, k, nb, ncol)
            n = ls[slot] + 2k
            base = S[slot] * ρ^(2k)
            X[p, j] = T(weights[1]) * base
            X[np + p, j] = T(weights[2]) * (n * base / ρ)
            cont ≥ 2 && (X[2np + p, j] = T(weights[3]) * (n * (n - 1) * base / (ρ * ρ)))
        end
        for b in 0:cont, v in 1:NFIT
            Y[b * np + p, v] = T(weights[b + 1]) * (rb^b * samples[b + 1][p][v])
        end
    end
    F = qr(X)
    QtY = F.Q' * Y
    R = F.R
    C = UpperTriangular(R) \ QtY[1:ncol, :]
    Rs = R[1:(ncol - 1), 1:(ncol - 1)]
    if !shift_constant
        Cs = UpperTriangular(Rs) \ QtY[1:(ncol - 1), FIT_SHIFT]
        for (i, v) in enumerate(FIT_SHIFT)
            C[1:(ncol - 1), v] .= Cs[:, i]
            C[ncol, v] = zero(T)
        end
    end
    coeffs = zeros(T, NS, nb, NFIT)
    for slot in 1:NS, k in 0:cont, v in 1:NFIT
        coeffs[slot, k + 1, v] = C[_fit_column(slot, k, nb, ncol), v]
    end
    # The residual of each block of rows against that block's own scale in
    # the variable's group — a row weight scales both and cancels.
    res = X * C - Y
    per = zeros(Float64, NFIT)
    val = zeros(Float64, NFIT)
    for g in FIT_GROUPS, b in 0:cont
        rows = (b * np + 1):((b + 1) * np)
        scale = tofloat64(maximum(abs, view(Y, rows, g)))
        scale = scale > 0 ? scale : 1.0
        for v in g
            r = tofloat64(maximum(abs, view(res, rows, v))) / scale
            per[v] = max(per[v], r)
            b == 0 && (val[v] = r)
        end
    end
    residual = (per_variable=SVector{NFIT,Float64}(per), value=maximum(val),
                overall=maximum(per))
    conditioning = (scalars=tofloat64(cond(UpperTriangular(R), 1)),
                    shift=tofloat64(cond(UpperTriangular(Rs), 1)),
                    rows=nrow, columns=ncol)
    XC = X * C
    w0 = T(weights[1])
    model = [SVector{NFIT,T}(ntuple(v -> XC[p, v] / w0, Val(NFIT))) for p in 1:np]
    return coeffs, residual, conditioning, model
end

# The radial factors of one slot for all twenty variables:
# `Σ_k C[slot, k+1, v] ρ^{2k}`.
@inline function _radial_sum(coeffs, slot::Int, cont::Int, P2::T,
                             P4::T) where {T}
    return SVector{NFIT,T}(ntuple(Val(NFIT)) do v
        a = coeffs[slot, 1, v]
        if cont ≥ 1
            a += P2 * coeffs[slot, 2, v]
        end
        if cont ≥ 2
            a += P4 * coeffs[slot, 3, v]
        end
        a
    end)
end

"""
    fit_variables_at(params, coeffs, x, t) -> SVector{20}

The fit's twenty variables at the point `x` at time `t`: the ansatz of
[`InteriorFit`](@ref) about `c(t)` of `params.center`, summed by the solid
harmonics' fold (`_solid_harmonic_fold`). `@inline`, allocation-free,
generic in `T`, one `return`: the evaluator a kernel calls (step 8e-ii).
"""
@inline function fit_variables_at(p::FitParams{T}, coeffs, x, t) where {T}
    c = center_at(p.center, t)
    ir = inv(p.rbar)
    ξ1 = (T(x[1]) - c[1]) * ir
    ξ2 = (T(x[2]) - c[2]) * ir
    ξ3 = (T(x[3]) - c[3]) * ir
    ρ² = ξ1 * ξ1 + ξ2 * ξ2 + ξ3 * ξ3
    P4 = ρ² * ρ²
    cont = p.cont
    f = (acc, slot, l, s) -> acc + s * _radial_sum(coeffs, slot, cont, ρ², P4)
    return _solid_harmonic_fold(f, zero(SVector{NFIT,T}), p.lmax, ξ1, ξ2, ξ3,
                                ρ²)
end

"""
    fit_state(fit, x, t) -> (h, Π)
    fit_state(params, coeffs, x, t) -> (h, Π)

The fitted target's packed state at `x`, `t`: [`fit_variables_at`](@ref),
[`state_from_fit`](@ref) — `g_ab` reassembled from `(α, β^i, γ_ij)` — and
[`bounds_project`](@ref) into the fit's bounds as the **guarantee**: the
validity sweep checks the fit at its sample points, and the projection is
what makes every other point a metric too. On a fit the sweep passed and
whose values lie inside the bounds the projection is the identity, bit for
bit ([`bounds_project`](@ref) returns an unfired state with its bits).
`@inline`, allocation-free, generic in `T`; the host form reads the host
coefficients.
"""
@inline function fit_state(p::FitParams{T}, coeffs, x, t) where {T}
    h, Π = state_from_fit(fit_variables_at(p, coeffs, x, t), p.tilde)
    h′, Π′, _, _ = bounds_project(h, Π, p.bounds)
    return h′, Π′
end

fit_state(fit::InteriorFit, x, t) = fit_state(fit.params, fit.host, x, t)

# The number of radii of the validity sweep along each collocation ray.
const FIT_SWEEP_RADII = 8

"""
    fit_sweep(params, coeffs, ns, r1s, t) -> (; valid, npoints, ninvalid,
        min_detγ, min_α, min_λ, max_β_center, hits, worst)

The fit evaluated **raw** — reassembled, not projected — at the collocation
directions `ns` times the radii `s r_1(n̂)`, `s ∈ {1/8, 2/8, …, 1}`, and at
the center: `8 N + 1` points covering the inside of the offset surface,
where the fit is the layer's and the core's target. A point is valid when
its state is finite, `γ_ij` is positive definite (the smallest eigenvalue by
[`sym_eigen3`](@ref)) and the signed lapse of [`state_validity`](@ref) is
positive; `min_detγ`, `min_α` and `min_λ` are the worst over the sweep,
`max_β_center` the shift's magnitude at the center (zero by the ansatz,
unless the fit was asked for the shift's constant),
`hits` the number of points [`bounds_project`](@ref) would move, and
`worst` the first invalid point's position.

**The sweep's radii are fractions of each ray's own `r_1(n̂)` (proposed in
step 8e)**, not of the ansatz's `r̄`: `s = 1` is the collocation surface
itself on every ray, so the sweep covers exactly the region the fit is used
in, oblate or not.
"""
function fit_sweep(p::FitParams{T}, coeffs, ns, r1s, t) where {T}
    c = center_at(p.center, T(t))
    pts = SVector{3,T}[]
    for (n, r1) in zip(ns, r1s), s in 1:FIT_SWEEP_RADII
        push!(pts, c + ((T(s) / FIT_SWEEP_RADII) * r1) * SVector{3,T}(n))
    end
    push!(pts, c)
    min_detγ = floatmax(T)
    min_α = floatmax(T)
    min_λ = floatmax(T)
    ninvalid = 0
    hits = 0
    worst = nothing
    for x in pts
        h, Π = state_from_fit(fit_variables_at(p, coeffs, x, t), p.tilde)
        finite = all(isfinite, h) & all(isfinite, Π)
        detγ, α, _, _ = state_validity(h, Π)
        λ, _ = sym_eigen3(SMatrix{3,3,T}(1 + h[5], h[6], h[7], h[6], 1 + h[8],
                                         h[9], h[7], h[9], 1 + h[10]))
        λmin = minimum(λ)
        min_detγ = min(min_detγ, detγ)
        min_α = min(min_α, α)
        min_λ = min(min_λ, λmin)
        hits += bounds_project(h, Π, p.bounds)[3]
        if !(finite & (λmin > 0) & (α > 0))
            ninvalid += 1
            worst === nothing && (worst = Tuple(x))
        end
    end
    vc = fit_variables_at(p, coeffs, c, t)
    βc = sqrt(vc[2] * vc[2] + vc[3] * vc[3] + vc[4] * vc[4])
    return (valid=ninvalid == 0, npoints=length(pts), ninvalid=ninvalid,
            min_detγ=tofloat64(min_detγ), min_α=tofloat64(min_α),
            min_λ=tofloat64(min_λ), max_β_center=tofloat64(βc), hits=hits,
            worst=worst)
end

# A sample that is not finite, or not a Lorentzian metric, has no `log α`
# and no `β^i` — and on the offset surface it is a configuration error, not
# something to fit.
function _check_finite(u, x, t, what)
    all(isfinite, u) || throw(ArgumentError(
        "the fit's sample$what at x = $(Tuple(x)), t = $t on the offset " *
        "surface is not finite: the solution the layer's target is fitted to is singular " *
        "there. The offset surface lies m cells inside the horizon and n_L " *
        "cells outside the core, where the analytic solution is regular by " *
        "construction of the geometry — a singular point on it is a " *
        "configuration error (a chart's singular set outside the tracked " *
        "core, or a tracked center far from the hole), not something to fit."))
    return nothing
end

function _check_sample(u, x, t)
    _check_finite(u, x, t, "")
    h = SVector{NC}(ntuple(k -> u[k], Val(NC)))
    detγ, α, _, _ = state_validity(h, SVector{NC}(ntuple(k -> u[NC + k], Val(NC))))
    (detγ > 0 && α > 0) || throw(ArgumentError(
        "the fit's sample at x = $(Tuple(x)), t = $t is not a Lorentzian " *
        "metric (det γ = $detγ, signed α = $α): the fit's variables log α and " *
        "β^i do not exist there. The offset surface is evolved by the " *
        "unmodified equations, so a degenerate state on it is a failed run, " *
        "and the validity monitor's shell rows say where it began."))
    return nothing
end

"""
    build_fit(sampler, int::FittedInterior, spec::FittedSpec; cont = 1,
              bounds, L = spec.lmax_fit, backend = CPU(), check = true,
              weights = fit_row_weights(L, cont), shift_constant = true,
              tilde = spec.fit_tilde)
        -> InteriorFit

The fitted target of step 8e, on the tracked geometry `int` at the
sampler's time `t` — `CODE.md`, "The fitted target":

 1. **Collocation** at `x_p = c + r_1(n̂_p) n̂_p` on the offset surface
    `r_1(n̂) = r_h(n̂) − offset` ([`shape_radius`](@ref)), with `n̂_p` the
    points of `EquiangularGrid(L)` ([`fit_directions`](@ref)) and
    `c = c(t)` the geometry's (tracked) center.
 2. **Samples** from `sampler(xs, ns)` — [`state_sampler`](@ref) (values and
    one radial derivative) or [`analytic_sampler`](@ref) (two) — each
    checked finite and a metric, and converted to the fit's variables by
    the chain rule ([`fit_variables`](@ref)).
 3. **The ansatz** `f_v = Σ_{lm} S_lm(ξ) Σ_k C_{lm,k,v} ρ^{2k}`, `ξ = (x −
    c)/r̄`, `r̄` the mean of `r_1(n̂_p)`: a polynomial in `x` of degree `L +
    2 cont`, regular at the center; `k ≤ cont`, and the shift without its
    constant term only if `shift_constant = false` — the default keeps it,
    which a moving hole needs (`fit.jl`'s header; **decided in review, step
    8e**). `cont = 1` matches
    values and slopes (the evolved state's fit); `cont = 2` also curvatures
    (the initial data's, from the analytic sampler).
 4. **One least-squares solve** ([`solve_fit`](@ref)), the residual
    ([`fit_residual`](@ref)) and the conditioning.
 5. **The validity sweep** ([`fit_sweep`](@ref)), which **throws** an
    `ArgumentError` with its worst numbers and the remedies when a swept
    point is not a metric — unless `check = false`, when the fit is
    returned with `valid = false` for the caller to record.

`bounds` has no default: it is the [`StateBounds`](@ref) [`fit_state`](@ref)
projects into — the case's own where it has them, [`default_bounds`](@ref)
otherwise; the gate radius is not read. The coefficients go to `backend`
through [`to_backend`](@ref). Host-side, milliseconds at `L = 8` (measured
under "The fitted target" in `CODE.md`).
"""
function build_fit(sampler, int::FittedInterior{T}, spec::FittedSpec;
                   cont::Integer=1, bounds::StateBounds, L::Integer=spec.lmax_fit,
                   backend=CPU(), check::Bool=true,
                   weights=fit_row_weights(L, cont),
                   shift_constant::Bool=true,
                   tilde::Bool=spec.fit_tilde) where {T}
    cont in (1, 2) || throw(ArgumentError(
        "cont is the fit's radial order, 1 (values and slopes, the evolved " *
        "state's fit) or 2 (and curvatures, the initial data's), got $cont."))
    L ≥ 1 || throw(ArgumentError("the fit's degree must be at least 1, got L = $L."))
    bd = _bounds_in(T, bounds)
    t = T(sampler.t)
    c = center_at(int.center, t)
    ns = [SVector{3,T}(T(n[1]), T(n[2]), T(n[3])) for n in fit_directions(L)]
    r1s = [shape_radius(int, n) - int.offset for n in ns]
    rbar = sum(r1s) / length(r1s)
    xs = [c + r1 * n for (r1, n) in zip(r1s, ns)]
    samples = sampler(xs, ns)
    for i in eachindex(xs)
        _check_sample(samples[1][i], xs[i], t)
        for b in 2:length(samples)
            _check_finite(samples[b][i], xs[i], t,
                          " ($(b - 1)$(b == 2 ? "st" : "nd") radial derivative)")
        end
    end
    nb = cont + 1
    length(samples) ≥ nb || throw(ArgumentError(
        "a fit with cont = $cont needs $cont radial derivative(s) of the " *
        "state, and this sampler provides $(length(samples) - 1): the state " *
        "sampler has only the interpolant's gradient and serves cont = 1; " *
        "cont = 2 (the initial data's fit) is the analytic sampler's."))
    vs = map(eachindex(xs)) do i
        cont == 1 ? fit_variables(samples[1][i], samples[2][i]; tilde=tilde) :
        fit_variables(samples[1][i], samples[2][i], samples[3][i]; tilde=tilde)
    end
    fitted = ntuple(b -> [vs[i][b] for i in eachindex(xs)], nb)
    ξs = [(r1 / rbar) * n for (r1, n) in zip(r1s, ns)]
    coeffs, residual, conditioning, model = solve_fit(ξs, fitted, L, cont,
                                                      rbar; weights=weights,
                                                      shift_constant=shift_constant)
    params = FitParams{T}(Int(L), Int(cont), rbar, int.center, bd, tilde)
    sweep = fit_sweep(params, coeffs, ns, r1s, t)
    sweep.valid || !check || throw(ArgumentError(
        "the fitted target (L = $L, cont = $cont) is not a valid metric at " *
        "$(sweep.ninvalid) of the $(sweep.npoints) points of its validity " *
        "sweep inside the offset surface (r̄ = $rbar about $(Tuple(c)), " *
        "t = $t): min λ(γ) = $(sweep.min_λ), min det γ = $(sweep.min_detγ), " *
        "min α = $(sweep.min_α), first at x = $(sweep.worst). The fit is a " *
        "polynomial of degree L + 2cont in x, fitted on the surface and " *
        "extrapolated inward, and a high degree oscillates there: lower " *
        "lmax_fit, use cont = 1, or widen the margin so that the offset " *
        "surface sits where the solution is smoother (CODE.md, \"The fitted " *
        "target\")."))
    return InteriorFit{T,typeof(to_backend(backend, coeffs)),typeof(coeffs)}(
        params, to_backend(backend, coeffs), coeffs, t, c, xs, model,
        residual, conditioning, sweep.valid, sweep)
end

# --- the target's ranges ---------------------------------------------------------

"""
    derive_target_bounds(T, background, int::FittedInterior; t = 0, L = 8)
        -> StateBounds{T}

The ranges the `:fitted` target is projected into ([`fit_state`](@ref)),
from the **analytic** data on the geometry's offset surface at `t` — the
seed's, at the start of a run — **(decided in review, step 8e)**: each upper
range four times the largest value found there (`α`, the eigenvalues of `γ`,
`|β|`, `max_ab |(α/√γ)Π_ab|`), and each lower one a quarter of the smallest
(`α`, `λ(γ)`) **(the factors proposed in step 8e)**, widened where needed so
that flat space stays inside (`StateBounds` requires it: `α_min, λ_min ≤ 1/2`
and `α_max, λ_max ≥ 2`).

Why not [`default_bounds`](@ref): those are step 8b's proposal for the
Kerr-Schild fixture, and harmonic Kerr at `a = 9/10` has `|(α/√γ)Π| = 366/M`
on its offset surface at `h = 5/256`, `m = 4` against their `K_max =
100/M` (measured in step 8e) — a projection into them would clamp the
target where the chart's own solution lives. The samples are `Float64` and
the collocation grid is the fit's own (`fit_directions(L)`); the gate radius
is not read by `fit_state` and is set to the offset surface's smallest.
"""
function derive_target_bounds(::Type{T}, background, int::FittedInterior;
                              t=0, L::Integer=8) where {T}
    c = center_at(int.center, typeof(int.r_in)(t))
    αlo, αhi = Inf, 0.0
    λlo, λhi = Inf, 0.0
    βhi, Khi = 0.0, 0.0
    for n in fit_directions(L)
        nn = SVector{3,typeof(int.r_in)}(n)
        r1 = tofloat64(shape_radius(int, nn) - int.offset)
        x = ntuple(d -> tofloat64(c[d]) + r1 * n[d], 3)
        u = SVector{NFIT,Float64}(state_tuple(background, Float64(t), x))
        all(isfinite, u) || throw(ArgumentError(
            "the analytic data on the offset surface is not finite at " *
            "$x: the geometry's surface meets the chart's singular set, and " *
            "no range can be derived there (nor any fit built)."))
        h = SVector{NC,Float64}(ntuple(k -> u[k], Val(NC)))
        s = _adm_split(_sym4(h))
        α = sqrt(s.α²)
        λ, _ = sym_eigen3(SMatrix{3,3,Float64}(1 + h[5], h[6], h[7], h[6], 1 + h[8],
                                               h[9], h[7], h[9], 1 + h[10]))
        K = maximum(k -> abs(u[NC + k]), 1:NC) * α / sqrt(s.detγ)
        αlo, αhi = min(αlo, α), max(αhi, α)
        λlo, λhi = min(λlo, minimum(λ)), max(λhi, maximum(λ))
        βhi = max(βhi, sqrt(max(s.bb, 0.0)))
        Khi = max(Khi, K)
    end
    rg = tofloat64(int.r_in - int.offset)
    return StateBounds{T}(min(αlo / 4, 0.5), max(4αhi, 2.0), min(λlo / 4, 0.5),
                          max(4λhi, 2.0), βhi > 0 ? 4βhi : 1.0,
                          Khi > 0 ? 4Khi : 1.0, rg)
end

# --- the target on the grid (the kernel half, step 8e-ii) --------------------------
#
# `CODE.md`, "The fitted target" — the cache (decided in review, step 8e): the
# right-hand side does not evaluate the fit, at `2.2 µs` a point at `L = 8`;
# it reads it from a field set filled once per fit and per refill. The field
# set is `Hsrc`'s shape — `G = 0`, read at the owned point, never differenced
# — with **40 variables: the target `A` (the packed `(h, Π)`, 1–20) at the
# fill time `t_f` and its slope in time `S` (21–40)**, so that the kernel's
# target at `t` is `A + (t − t_f) S`, two loads per variable (**proposed in
# step 8e**: one field set rather than two, and the slope rather than the
# previous fit's values, since the slope is what the time interpolation
# multiplies). `A` and `S` are the latest fit `F_a` (built at `t_a`) and the
# one before it `F_b` (at `t_b`), both evaluated at the fill time about their
# own tracked centers:
#
#     S = (F_a − F_b)/(t_a − t_b),    A = F_a + (t_f − t_a) S,
#
# the linear continuation of the last two fits — `S = 0` when there is no
# previous fit or the two were built at the same time.

# The target's value at the owned point `inner` of block `b`, variable `v`
# (1–20), at time `t`: the kernel side of the cache.
@inline _cached_target(tw, inner, b::Int, v::Int, t, t_f) =
    @inbounds tw[inner..., v, b] + (t - t_f) * tw[inner..., 2NC + v, b]

# The target's rate `∂_t u_fit` at the owned point — the cache's slope — for
# the feed-forward term of step 8.
@inline _cached_rate(tw, inner, b::Int, v::Int) = @inbounds tw[inner..., 2NC + v, b]

"""
    fit_target_kernel!(out, origins, spacings, interior, pa, ca, pb, cb, t_a, κ,
                       t_f, r_fill, ::Val{HASB})

Fill the target cache at every owned point with `r < r_fill` about the
geometry's center — the offset surface's bounding sphere plus one cell,
so that a center that moves by the refill rule's `h/4` never exposes a layer
point that was not filled — and zero elsewhere: `A` and `S` of the latest
fit `(pa, ca)` built at `t_a` and, where `HASB`, the previous one `(pb, cb)`
with `κ = 1/(t_a − t_b)`, both through [`fit_state`](@ref) with their
coefficient arrays as device arguments. One evaluator pass per fit per
filled point, at the refill cadence.
"""
@kernel function fit_target_kernel!(out, @Const(origins), @Const(spacings),
                                    interior, pa, @Const(ca), pb, @Const(cb),
                                    t_a, κ, t_f, r_fill, δt,
                                    ::Val{HASB}, ::Val{RATE}) where {HASB,RATE}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    T = eltype(out)
    x = point_position(origins, spacings, b, I)
    if interior_radius(interior, t_f, x) < r_fill
        ha, Πa = fit_state(pa, ca, x, t_f)
        # The translation of the latest fit with the track at this point
        # (step 8): the fit is a polynomial about a center that moves at the
        # track's velocity, so its rate at a fixed point is a centered
        # difference in the time it is evaluated at — exactly zero for a fit
        # whose center does not move.
        hp, Πp = RATE ? fit_state(pa, ca, x, t_f + δt) : (ha, Πa)
        hm, Πm = RATE ? fit_state(pa, ca, x, t_f - δt) : (ha, Πa)
        iδ = RATE ? inv(2 * δt) : zero(T)
        if HASB
            hb, Πb = fit_state(pb, cb, x, t_f)
            ntuple(Val(NC)) do v
                sh = κ * (ha[v] - hb[v])
                sΠ = κ * (Πa[v] - Πb[v])
                out[inner..., v, b] = ha[v] + (t_f - t_a) * sh
                out[inner..., NC + v, b] = Πa[v] + (t_f - t_a) * sΠ
                out[inner..., 2NC + v, b] = RATE ? sh + iδ * (hp[v] - hm[v]) : sh
                out[inner..., 3NC + v, b] = RATE ? sΠ + iδ * (Πp[v] - Πm[v]) : sΠ
                nothing
            end
        else
            ntuple(Val(NC)) do v
                out[inner..., v, b] = ha[v]
                out[inner..., NC + v, b] = Πa[v]
                out[inner..., 2NC + v, b] = RATE ? iδ * (hp[v] - hm[v]) : zero(T)
                out[inner..., 3NC + v, b] = RATE ? iδ * (Πp[v] - Πm[v]) : zero(T)
                nothing
            end
        end
    else
        ntuple(Val(4NC)) do v
            out[inner..., v, b] = zero(T)
            nothing
        end
    end
end

"""
    target_cache(U::FieldSet) -> FieldSet

A fresh, zeroed target cache on `U`'s forest and backend: `4 NC = 40`
variables (the target and its slope), `G = 0`, `U`'s centering.
"""
target_cache(U::FieldSet{T,3}) where {T} =
    FieldSet{T}(U.forest, 4NC; G=0, centering=U.centering,
                backend=get_backend(U.work))

"""
    fill_target!(target, origins, spacings, interior, fits, t) -> target

Fill the cache `target` at time `t` from `fits = (latest, previous)`
([`InteriorFit`](@ref)s, the second `nothing` before there are two) on the
geometry `interior` ([`fit_target_kernel!`](@ref)).
"""
function fill_target!(target::FieldSet{T,3}, origins, spacings,
                      interior::FittedInterior, fits, t; rate::Bool=false) where {T}
    fa, fb = fits
    hasb = fb !== nothing && fa.t > fb.t
    κ = hasb ? inv(fa.t - fb.t) : zero(T)
    r_fill = (interior.r_out - interior.offset) + interior.h
    pb, cb = hasb ? (fb.params, fb.coeffs) : (fa.params, fa.coeffs)
    # The time step of the translation's difference: the fit's center moves
    # an eighth of a cell over it (any positive number for a fit at rest,
    # whose difference is then exactly zero).
    speed = sqrt(sum(abs2, fa.params.center.v))
    δt = speed > 0 ? interior.h / (8 * speed) : one(T)
    map_blocks!(fit_target_kernel!, target, target.work, origins, spacings,
                interior, fa.params, fa.coeffs, pb, cb, T(fa.t), κ, T(t),
                T(r_fill), T(δt), Val(hasb), Val(rate))
    return target
end

"""
    snapshot_target_kernel!(out, ua, origins, spacings, interior, t_f, r_fill)

The **snapshot target** (added in step 8f, `PLAN.md`'s idea 5, the control
row of step 8f's matrix): the cache holds the *state itself* at the fill
time — `A = u(t_f)`, `S = 0` — at every owned point with `r < r_fill`, and
zero elsewhere, so that the `:fitted` branch relaxes the layer and the core
toward the state as it was at the chunk's start. No fit, no regularisation:
the question it answers is what the fit's regularity buys over holding the
evolved state still. It is **not advected** with the track's velocity
**(proposed in step 8f)**: the row that runs it is the static hole, where
the velocity is the finder's noise.
"""
@kernel function snapshot_target_kernel!(out, @Const(ua), @Const(origins),
                                         @Const(spacings), interior, t_f,
                                         r_fill)
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    T = eltype(out)
    x = point_position(origins, spacings, b, I)
    if interior_radius(interior, t_f, x) < r_fill
        ntuple(Val(2NC)) do v
            out[inner..., v, b] = ua[inner..., v, b]
            out[inner..., 2NC + v, b] = zero(T)
            nothing
        end
    else
        ntuple(Val(4NC)) do v
            out[inner..., v, b] = zero(T)
            nothing
        end
    end
end

"""
    fill_snapshot!(target, ua, origins, spacings, interior, t) -> target

Fill the cache `target` with the state array `ua` inside the offset
surface's bounding sphere plus one cell ([`snapshot_target_kernel!`](@ref)).
"""
function fill_snapshot!(target::FieldSet{T,3}, ua, origins, spacings,
                        interior::FittedInterior, t) where {T}
    r_fill = (interior.r_out - interior.offset) + interior.h
    map_blocks!(snapshot_target_kernel!, target, target.work, ua, origins,
                spacings, interior, T(t), T(r_fill))
    return target
end

"""
    fitted_state_kernel!(u, tw, origins, spacings, bg, interior, t, t_f, d_init)

The `:fitted` variant's initial data (decided in review, step 8e): the
analytic state [`state_tuple`](@ref) outside the offset surface and the
cached target — the `cont = 1` fit of the analytic solution — below it,
written into the state array's owned points. No core rule and no analytic
evaluation inside the offset surface, so a chart whose interior is singular
— harmonic Kerr's disk — gets regular data. `d_init` moves the switch to the
depth `d_init` below the offset surface (at most the ramp's thickness): `0`
is the decision; a positive depth is the study knob of `evolve!`'s
`fit_initial_depth` (proposed in step 8e).
"""
@kernel function fitted_state_kernel!(u, @Const(tw), @Const(origins),
                                      @Const(spacings), bg, interior, t, t_f,
                                      d_init)
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    x = point_position(origins, spacings, b, I)
    g = interior_point(interior, t, x)
    if g.r ≥ g.r_1 - d_init
        vals = state_tuple(bg, t, x)
        ntuple(Val(2NC)) do v
            u[inner..., v, b] = vals[v]
            nothing
        end
    else
        ntuple(Val(2NC)) do v
            u[inner..., v, b] = _cached_target(tw, inner, b, v, t, t_f)
            nothing
        end
    end
end
