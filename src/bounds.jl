# The range projection — the third and last writer of the state — and the
# validity monitor that says where it is needed.
#
# `CODE.md`, "The interior" (the range projection, added in step 8b), and
# `PLAN.md`'s step 8b. TreeHydro's atmosphere reset is the pattern: a
# **pointwise map** installed as the integrator's stage limiter, **written
# back only where it fired**, idempotent on the state, counted, and
# checked by a bitwise control — a run on which nothing fires is bit for
# bit the run without the mechanism.
#
# What it is *not* is as important as what it is, because the two obvious
# alternatives are both wrong here:
#
#   * **Not a reset to a fixed state.** TreeHydro's atmosphere replaces a
#     vacuum cell by `(ρ_atm, 0, p_atm)` because below `ρ_atm` the cell *is*
#     vacuum and what it held is dust. A metric that has left the range of
#     metrics is not dust: most of it is still right, and the map moves the
#     one quantity that is wrong and leaves the others alone.
#   * **Not a projection onto flat space, and not a clamp per component of
#     `h_ab`.** The Lorentzian metrics are not convex in `g_ab` — the angular
#     mean of Kerr-Schild `g_ab` on the sphere `r = 1.15 M` has `g_tt = +0.74`
#     and `g_ti = 0`, a Euclidean signature (`PLAN.md`, finding 2) — so a box
#     in `h_ab` contains non-metrics and excludes metrics. The ranges are
#     stated in **ADM variables**, where `α > 0` and `γ ≻ 0` are convex
#     conditions: the eigenvalues of `γ_ij`, the norm of the shift, the
#     lapse, and the scale of `Π`.
#
# Three things here are deliberate and easy to undo by accident, and each
# is written out where it is made:
#
#   * **A healthy quantity keeps its bits.** Nothing is reassembled unless
#     it fired: `h_tt` is written only when `α` is clamped, `h_ti` only when
#     the shift is, the spatial block only when `γ`'s spectrum is. A map that
#     wrote `g′ − η` unconditionally would move every component by an ulp
#     — `(−1 + h_tt) + 1` is not `h_tt` — and the control below would be a
#     tolerance instead of an equality.
#   * **The tests have a slack and the targets do not.** A clamped value
#     recomputed from the stored state comes back an ulp either side of its
#     bound, and a test without slack fires again on it: the state would
#     still be a fixed point, but its *flag* would not be, which is what
#     TreeHydro measured of its pressure floor. So every test is relaxed by
#     `8 eps` of the scale of the terms it is made of, and every clamp moves
#     to the bound exactly.
#   * **The ADM split is explicit scalar arithmetic.** It is evaluated at
#     several points of one call and again, on the stored result, by the
#     next call; Julia does not contract `a*b + c` into a fused multiply-add
#     unless asked (StaticArrays' matrix product asks, through `muladd`), so
#     written out it is the same bits at every call site — which is what
#     lets a second pass see exactly the lapse the first pass wrote.

"""
    StateBounds{T}(α_min, α_max, λ_min, λ_max, β_max, K_max, r_gate)
    StateBounds(T = Float64; α_min, α_max, λ_min, λ_max, β_max, K_max, r_gate)

The ranges [`bounds_project`](@ref) holds the state inside, and the radius
inside which it is applied.

  * `α_min ≤ α ≤ α_max` — the lapse, `α² = β_iβ^i − g_tt`;
  * `λ_min ≤ λ ≤ λ_max` — every eigenvalue of the spatial metric `γ_ij`;
  * `|β| ≤ β_max` — the shift's norm `√(β_i γ^{ij} β_j)`;
  * `max_ab |(α/√γ) Π_ab| ≤ K_max` — the scale of the momentum as the first
    evolution equation `∂_t h = β^i∂_i h + (α/√γ)Π` sees it: an inverse time,
    and the analogue of a bound on the extrinsic curvature;
  * `r_gate` — the projection runs only at `r < r_gate` from the hole's
    center, so that the discontinuity a clamp makes sits deeper than any
    point an evolved stencil reads (see [`default_gate`](@ref)).

**There is no default for any of them** — a range is a statement about the
case, and `CODE.md`'s rule is that a caller thinks about every number of
that kind. [`default_bounds`](@ref) is the named set this package proposes,
which a caller asks for by name.

**Flat space must lie inside every range** (`α_min < 1 < α_max`,
`λ_min < 1 < λ_max`), and the constructor says so when it does not: a
component that is not finite is replaced by its Minkowski value, and a
repair that the next check would move again is not idempotent.

`isbits`, so that it travels into the kernel as an argument; a field of
[`GHCase`](@ref) beside `horizon`, with `nothing` meaning "no projection".
"""
struct StateBounds{T}
    α_min::T
    α_max::T
    λ_min::T
    λ_max::T
    β_max::T
    K_max::T
    r_gate::T

    function StateBounds{T}(α_min, α_max, λ_min, λ_max, β_max, K_max,
                            r_gate) where {T}
        vals = map(x -> convert(T, x),
                   (α_min, α_max, λ_min, λ_max, β_max, K_max, r_gate))
        all(isfinite, vals) || throw(ArgumentError(
            "every bound must be finite, got $vals: a bound at infinity is " *
            "no bound, and it is spelled by leaving the case's bounds at " *
            "`nothing` instead."))
        a0, a1, l0, l1, b, K, rg = vals
        0 < a0 < 1 < a1 || throw(ArgumentError(
            "the lapse range needs 0 < α_min < 1 < α_max, got α_min = $a0 and " *
            "α_max = $a1: α > 0 is what makes the state a Lorentzian metric at " *
            "all, and flat space's α = 1 has to be inside the range because " *
            "a non-finite component is repaired to its Minkowski value — a " *
            "repair the next check would move again is not idempotent."))
        0 < l0 < 1 < l1 || throw(ArgumentError(
            "the spatial metric's eigenvalue range needs 0 < λ_min < 1 < " *
            "λ_max, got λ_min = $l0 and λ_max = $l1: γ ≻ 0 is what makes the " *
            "slice spacelike, and flat space's λ = 1 has to be inside the " *
            "range for the non-finite repair to be idempotent."))
        b > 0 || throw(ArgumentError(
            "the shift's norm bound must be positive, got β_max = $b: a zero " *
            "bound clamps every shift to zero, which is a reset and not a " *
            "range."))
        K > 0 || throw(ArgumentError(
            "the momentum's scale bound must be positive, got K_max = $K: a " *
            "zero bound sets Π = 0 wherever the projection runs, which is a " *
            "reset and not a range."))
        rg > 0 || throw(ArgumentError(
            "the gate radius must be positive, got r_gate = $rg: the " *
            "projection runs at r < r_gate, so a non-positive gate is a " *
            "projection that never runs — which is `bounds = nothing`, and " *
            "should be spelled so."))
        return new{T}(a0, a1, l0, l1, b, K, rg)
    end
end

StateBounds(::Type{T}=Float64; α_min, α_max, λ_min, λ_max, β_max, K_max,
            r_gate) where {T} =
    StateBounds{T}(α_min, α_max, λ_min, λ_max, β_max, K_max, r_gate)

# The same ranges in another working type — how a `FittedSpec`'s target
# bounds and `build_fit`'s `bounds` keyword meet a run's `T` (step 8e).
_bounds_in(::Type{T}, b::StateBounds{T}) where {T} = b
_bounds_in(::Type{T}, b::StateBounds) where {T} =
    StateBounds{T}(b.α_min, b.α_max, b.λ_min, b.λ_max, b.β_max, b.K_max,
                   b.r_gate)

"""
    default_bounds(T = Float64; M, r_gate) -> StateBounds{T}

The ranges this package proposes **(proposed in step 8b)**, chosen far
outside any state a healthy interior holds, so that on a hole that is
being evolved well the projection never fires:

| bound | value | Kerr-Schild `a = 0` at `r_0 = 2/5 M` |
|---|---|---|
| `α_min`, `α_max` | `1/50`, `50` | `α = 1/√6 = 0.41` |
| `λ_min`, `λ_max` | `1/100`, `1000` | `λ = 1, 1, 6` |
| `β_max` | `10` | `|β| = 2.04` |
| `K_max` | `100/M` | `max |(α/√γ)Π| ≈ 11/M` |

— a factor of 20 below the smallest lapse and above the largest `|β|`, and
of 9 above the largest momentum scale, that the step-5 fixture's core holds
(the core is filled from the sphere `r_0`, so nothing deeper is ever
evaluated). `r_gate` has no default here: it is a statement about the mesh,
and [`default_gate`](@ref) is the proposal for it.
"""
default_bounds(::Type{T}=Float64; M, r_gate) where {T} =
    StateBounds{T}(T(1 // 50), T(50), T(1 // 100), T(1000), T(10),
                   T(100) / T(M), T(r_gate))

"""
    default_gate(interior, forest, q; t = 0) -> r_gate

`r_1 − 2 G h`, with `G = q/2 + 1` and `h` the spacing of the blocks that
contain `r_1` ([`layer_spacing`](@ref)) — **(proposed in step 8b)** as the
gate of the range projection until step 8a's leakage margin exists.

The projection is a pointwise map with a discontinuous derivative: where it
fires, its output is a clamp of its input, and a stencil straddling the
surface between fired and unfired points reads a kink. The innermost point
an evolved stencil reads is one reach inside `r_1` (`G h` for `q ≤ 4`,
[`check_bounds_gate`](@ref) asserts it), and a second `G h` is the margin
that keeps the kink's own stencils off the evolved ones too. Step 8a's
measured penetration length is the number that should replace the factor 2.
"""
function default_gate(int::Interior{T}, forest, q::Integer; t=zero(T)) where {T}
    h, _ = layer_spacing(forest, int, T(t))
    return int.r_1 - 2 * (q ÷ 2 + 1) * h
end

# The tracked geometry's gate (added in step 8d): the same `2 G h` below the
# offset surface's *smallest* radius, `r_in − offset`, since the gate stays a
# sphere about the (tracked) center and has to lie inside the surface in every
# direction.
function default_gate(int::FittedInterior{T}, forest, q::Integer;
                      t=zero(T)) where {T}
    h, _ = geometry_spacing(forest, int, T(t))
    return (int.r_in - int.offset) - 2 * (q ÷ 2 + 1) * h
end

# The Euclidean reach of the right-hand side's stencils, in spacings: the
# Kreiss–Oliger operator's `G = q/2 + 1` along an axis, and the mixed
# derivative's `√2 · q/2` along a diagonal — which is the larger of the two
# from `q = 6` on (`4.24` against `4`).
_stencil_reach(q::Integer, ::Type{T}) where {T} =
    max(T(q ÷ 2 + 1), sqrt(T(2)) * T(q ÷ 2))

"""
    check_bounds_gate(forest, interior, bounds, q; t = 0)

Assert that the range projection's gate lies deeper than every point an
evolved stencil reads: `r_gate ≤ r_1 − R h`, with `R` the stencils'
Euclidean reach in spacings (`G = q/2 + 1` for `q ≤ 4`, `√2 · q/2` beyond)
and `h` the spacing of the blocks containing `r_1`.

It is checked where [`check_interior_radii`](@ref) is — in the
[`GHProblem`](@ref) constructor, and therefore after every regrid — for the
same reason: the gate is a radius fixed with the case, and the spacing it
is measured against belongs to the mesh as it now is. It throws, and it is
meant to; the remedy is a deeper gate or a finer layer, never a shallower
check.
"""
function check_bounds_gate(forest::Forest{3}, int::Interior{T},
                           bounds::StateBounds{T}, q::Integer;
                           t=zero(T)) where {T}
    h, _ = layer_spacing(forest, int, T(t))
    R = _stencil_reach(q, T)
    allowed = int.r_1 - R * h
    bounds.r_gate ≤ allowed || throw(ArgumentError(
        "the range projection's gate is too shallow: r_gate = " *
        "$(bounds.r_gate) but the layer's outer radius is r_1 = $(int.r_1) " *
        "at a spacing h = $h, and the right-hand side's stencils reach " *
        "$(R) spacings, so an evolved point (r ≥ r_1) reads points down to " *
        "r = $allowed. The projection's output is a clamp, which is a kink " *
        "wherever it fires, and a kink an evolved stencil reads is an O(1) " *
        "right-hand-side error outside the layer. Move r_gate inward — " *
        "default_gate proposes r_1 − 2Gh — or refine around r_1."))
    return (h=h, reach=R, allowed=allowed)
end

check_bounds_gate(forest, ::Nothing, bounds, q; t=0) = nothing
check_bounds_gate(forest, ::Nothing, ::Nothing, q; t=0) = nothing

# On the tracked geometry (step 8d): the innermost point an evolved stencil
# reads is one reach inside the offset surface's smallest radius.
function check_bounds_gate(forest::Forest{3}, int::FittedInterior{T},
                           bounds::StateBounds{T}, q::Integer;
                           t=zero(T)) where {T}
    h, _ = geometry_spacing(forest, int, T(t))
    R = _stencil_reach(q, T)
    r_1 = int.r_in - int.offset
    allowed = r_1 - R * h
    bounds.r_gate ≤ allowed || throw(ArgumentError(
        "the range projection's gate is too shallow for the tracked layer: " *
        "r_gate = $(bounds.r_gate), but the offset surface's smallest radius " *
        "is r_in − offset = $r_1 at a spacing h = $h, and the stencils reach " *
        "$R spacings, so an evolved point reads points down to r = $allowed. " *
        "Move r_gate inward — default_gate proposes (r_in − offset) − 2Gh."))
    return (h=h, reach=R, allowed=allowed)
end
check_bounds_gate(forest, int, ::Nothing, q; t=0) = nothing

# --- the pointwise map ------------------------------------------------------

# The 3+1 split of an offset metric `hm = _sym4(h)` in ADM variables and
# nothing else: the spatial metric `γ_ij = δ_ij + h_ij` (six entries), the
# **lowered** shift `β_i = g_ti = h_ti` (η has no `ti` part), `det γ`, the
# shift's square `β_iβ^i` with `β^i = γ^{ij}β_j`, and the lapse's square
# `α² = β_iβ^i − g_tt = (1 − h_tt) + β_iβ^i`.
#
# This is `adm_from_metric`'s split, written so that it *survives* the
# states it exists to repair: that function inverts the 4-metric and takes
# `α = 1/√(−g^{tt})`, which has no answer for a state whose `g^{tt}` has the
# wrong sign — and that state is one of the six this projection is for.
# Here the inverse is `γ`'s own, by the adjugate, and `α²` is a number that
# may be negative and says so. Explicit scalar arithmetic, for the reason
# the file header gives.
@inline function _adm_split(hm::SMatrix{4,4,T}) where {T}
    o = one(T)
    γ11 = o + hm[2, 2]
    γ12 = hm[3, 2]
    γ13 = hm[4, 2]
    γ22 = o + hm[3, 3]
    γ23 = hm[4, 3]
    γ33 = o + hm[4, 4]
    β1 = hm[2, 1]
    β2 = hm[3, 1]
    β3 = hm[4, 1]
    # The adjugate of the symmetric γ; `γ^{ij} = c_ij / det γ`.
    c11 = γ22 * γ33 - γ23 * γ23
    c12 = γ13 * γ23 - γ12 * γ33
    c13 = γ12 * γ23 - γ13 * γ22
    c22 = γ11 * γ33 - γ13 * γ13
    c23 = γ12 * γ13 - γ11 * γ23
    c33 = γ11 * γ22 - γ12 * γ12
    detγ = γ11 * c11 + γ12 * c12 + γ13 * c13
    qf = β1 * (c11 * β1 + c12 * β2 + c13 * β3) +
         β2 * (c12 * β1 + c22 * β2 + c23 * β3) +
         β3 * (c13 * β1 + c23 * β2 + c33 * β3)
    bb = qf / detγ
    α² = (o - hm[1, 1]) + bb
    return (γ=(γ11, γ12, γ13, γ22, γ23, γ33), β=(β1, β2, β3), detγ=detγ,
            bb=bb, α²=α²)
end

# Whether the symmetric 3×3 matrix `(a11, a12, a13, a22, a23, a33)` is
# *certainly* positive definite: Sylvester's criterion — the three leading
# principal minors positive — with each minor required to exceed `16 eps` of
# the sum of the magnitudes of its terms, which bounds its rounding error.
# A `true` is therefore a fact; a `false` is "not certain", and the caller
# then does the eigendecomposition. It is the fast path of the spectrum test:
# a healthy `γ` — every eigenvalue far inside `[λ_min, λ_max]` — never pays
# for a Jacobi sweep.
@inline function _surely_posdef(a11::T, a12::T, a13::T, a22::T, a23::T,
                                a33::T) where {T}
    tol = 16 * eps(T)
    m2 = a11 * a22 - a12 * a12
    e2 = abs(a11 * a22) + a12 * a12
    c1 = a22 * a33 - a23 * a23
    c2 = a12 * a33 - a23 * a13
    c3 = a12 * a23 - a22 * a13
    m3 = a11 * c1 - a12 * c2 + a13 * c3
    e3 = abs(a11) * (abs(a22 * a33) + a23 * a23) +
         abs(a12) * (abs(a12 * a33) + abs(a23 * a13)) +
         abs(a13) * (abs(a12 * a23) + abs(a22 * a13))
    return (a11 > zero(T)) & (m2 > tol * e2) & (m3 > tol * e3)
end

# One Jacobi rotation of the symmetric `A` in the `(p, q)` plane, the one
# that annihilates `A[p, q]`, accumulated into `V` (Golub & Van Loan's
# `sym.schur2`, with Numerical Recipes' update of the two diagonal entries,
# which is the accurate one). `t = tan θ` is the smaller root of
# `t² + 2τt − 1 = 0`, taken as `1/(2τ)` once `τ²` would overflow.
@inline function _jacobi_rotation(A::SMatrix{3,3,T}, V::SMatrix{3,3,T},
                                  p::Int, q::Int) where {T}
    apq = A[p, q]
    iszero(apq) && return A, V
    app = A[p, p]
    aqq = A[q, q]
    τ = (aqq - app) / (2 * apq)
    t = abs(τ) > inv(sqrt(eps(T))) ? inv(2 * τ) :
        copysign(one(T), τ) / (abs(τ) + sqrt(one(T) + τ * τ))
    c = inv(sqrt(one(T) + t * t))
    s = t * c
    r = 6 - p - q                               # the third index
    arp = A[r, p]
    arq = A[r, q]
    newp = app - t * apq
    newq = aqq + t * apq
    newrp = c * arp - s * arq
    newrq = s * arp + c * arq
    A′ = SMatrix{3,3,T}(ntuple(Val(9)) do k
        i = (k - 1) % 3 + 1
        j = (k - 1) ÷ 3 + 1
        (i == p && j == p) ? newp :
        (i == q && j == q) ? newq :
        ((i == p && j == q) || (i == q && j == p)) ? zero(T) :
        ((i == r && j == p) || (i == p && j == r)) ? newrp :
        ((i == r && j == q) || (i == q && j == r)) ? newrq : A[i, j]
    end)
    V′ = SMatrix{3,3,T}(ntuple(Val(9)) do k
        i = (k - 1) % 3 + 1
        j = (k - 1) ÷ 3 + 1
        j == p ? c * V[i, p] - s * V[i, q] :
        j == q ? s * V[i, p] + c * V[i, q] : V[i, j]
    end)
    return A′, V′
end

# At most this many cyclic sweeps. A 3×3 converges quadratically once the
# off-diagonal is small and takes four to six from a generic start; the cap
# is a guard against a state the stopping test never satisfies, not a
# budget.
const JACOBI_SWEEPS = 16

"""
    sym_eigen3(A::SMatrix{3,3}) -> (λ::SVector{3}, V::SMatrix{3,3})

The eigenvalues and orthonormal eigenvectors (the columns of `V`) of a
symmetric 3×3 matrix, by cyclic Jacobi rotations.

Written here rather than taken from `LinearAlgebra` because it runs inside
a kernel and because of *which* matrices it is asked about. StaticArrays'
3×3 closed form goes through `acos` of a quantity that is `±1` at a double
eigenvalue, where its error is `√eps` rather than `eps` — and a double
eigenvalue is exactly what Kerr-Schild's `γ = δ + (2M/r) l l` has, and what
two eigenvalues clamped to the same `λ_min` produce. Jacobi's error is a few
`eps` of `‖A‖` whatever the spectrum, which is what the spectrum test's
`8 eps` slack is measured against.
"""
@inline function sym_eigen3(A::SMatrix{3,3,T}) where {T}
    V = one(SMatrix{3,3,T})
    for _ in 1:JACOBI_SWEEPS
        off = A[1, 2] * A[1, 2] + A[1, 3] * A[1, 3] + A[2, 3] * A[2, 3]
        on = A[1, 1] * A[1, 1] + A[2, 2] * A[2, 2] + A[3, 3] * A[3, 3]
        off > eps(T) * eps(T) * on || break
        A, V = _jacobi_rotation(A, V, 1, 2)
        A, V = _jacobi_rotation(A, V, 1, 3)
        A, V = _jacobi_rotation(A, V, 2, 3)
    end
    return SVector{3,T}(A[1, 1], A[2, 2], A[3, 3]), V
end

# `Σ_k V_ik λ_k V_jk`, one entry of `V diag(λ) Vᵀ`.
@inline _recompose(V, λ, i, j) =
    V[i, 1] * λ[1] * V[j, 1] + V[i, 2] * λ[2] * V[j, 2] +
    V[i, 3] * λ[3] * V[j, 3]

# The slack of every test, in units of `eps` of the scale of the terms the
# tested number is made of (see the file header).
const BOUNDS_SLACK = 8

# The spectrum of `γ` clamped into `[λ_min, λ_max]`: `(γ′, fired)`. `γ` is
# returned *as it came* — the same six numbers — unless an eigenvalue is
# outside the range by more than the slack, and then the whole of `γ` is
# recomposed from its eigenvectors and the clamped eigenvalues. Recomposing
# rather than adding `Σ (λ′ − λ) q qᵀ` to `γ` keeps the rounding of the
# result at `eps` of the *new* spectrum, which is what lets the next call's
# test find the clamped eigenvalue within its slack of the bound however far
# it was moved.
@inline function _spectrum_clamp(γ, λ_min::T, λ_max::T) where {T}
    a11, a12, a13, a22, a23, a33 = γ
    lo = _surely_posdef(a11 - λ_min, a12, a13, a22 - λ_min, a23, a33 - λ_min)
    hi = _surely_posdef(λ_max - a11, -a12, -a13, λ_max - a22, -a23,
                        λ_max - a33)
    (lo & hi) && return γ, false
    λ, V = sym_eigen3(SMatrix{3,3,T}(a11, a12, a13, a12, a22, a23, a13, a23,
                                     a33))
    S = max(abs(λ[1]), abs(λ[2]), abs(λ[3]))
    s = BOUNDS_SLACK * eps(T) * S
    inside = ntuple(k -> (λ_min - s ≤ λ[k]) & (λ[k] ≤ λ_max + s), Val(3))
    (inside[1] & inside[2] & inside[3]) && return γ, false
    λ′ = SVector{3,T}(clamp(λ[1], λ_min, λ_max), clamp(λ[2], λ_min, λ_max),
                      clamp(λ[3], λ_min, λ_max))
    return (_recompose(V, λ′, 1, 1), _recompose(V, λ′, 1, 2),
            _recompose(V, λ′, 1, 3), _recompose(V, λ′, 2, 2),
            _recompose(V, λ′, 2, 3), _recompose(V, λ′, 3, 3)), true
end

# The offset metric with one of its ADM blocks replaced — the spatial block
# from `γ′` (as `γ′ − δ`), the shift from `β′`, or `h_tt` — and every other
# entry carried over **bit for bit** from `hm`.
@inline function _with_spatial(hm::SMatrix{4,4,T}, γ) where {T}
    o = one(T)
    γ11, γ12, γ13, γ22, γ23, γ33 = γ
    return SMatrix{4,4,T}(hm[1, 1], hm[2, 1], hm[3, 1], hm[4, 1],
                          hm[2, 1], γ11 - o, γ12, γ13,
                          hm[3, 1], γ12, γ22 - o, γ23,
                          hm[4, 1], γ13, γ23, γ33 - o)
end

@inline function _with_shift(hm::SMatrix{4,4,T}, β) where {T}
    β1, β2, β3 = β
    return SMatrix{4,4,T}(hm[1, 1], β1, β2, β3,
                          β1, hm[2, 2], hm[3, 2], hm[4, 2],
                          β2, hm[3, 2], hm[3, 3], hm[4, 3],
                          β3, hm[4, 2], hm[4, 3], hm[4, 4])
end

@inline function _with_lapse(hm::SMatrix{4,4,T}, htt) where {T}
    return SMatrix{4,4,T}(htt, hm[2, 1], hm[3, 1], hm[4, 1],
                          hm[2, 1], hm[2, 2], hm[3, 2], hm[4, 2],
                          hm[3, 1], hm[3, 2], hm[3, 3], hm[4, 3],
                          hm[4, 1], hm[4, 2], hm[4, 3], hm[4, 4])
end

"""
    bounds_project(h, Π, bounds::StateBounds) -> (h′, Π′, hit, nonfinite)

The range projection of one point's state: the packed offset metric `h` and
momentum `Π` (`SVector{10}` each), returned with every ADM quantity inside
the ranges of `bounds`, whether anything moved, and whether that was
because a component was not finite.

In order, each step reading the previous step's result:

 1. **a non-finite component** of `h` or `Π` takes its Minkowski value, `0`,
    and `nonfinite` says so — it is a different failure from a state that
    has left the range, and it is counted separately;
 2. **the spectrum of `γ_ij`** is clamped into `[λ_min, λ_max]` by a
    symmetric eigendecomposition ([`sym_eigen3`](@ref)). A rescale to
    `det γ ≥ δ` would not do: two negative eigenvalues have a positive
    determinant;
 3. **the shift** `|β| = √(β_i γ^{ij} β_j)`, with the projected `γ`, is capped
    at `β_max` by scaling `β_i`;
 4. **the lapse** `α² = β_iβ^i − g_tt`, with the projected `γ` and `β`, is
    clamped into `[α_min², α_max²]` by moving `g_tt` alone —
    `g′_tt = −α′² + β_iβ^i`;
 5. **the momentum** is rescaled by `(α/α′)(√γ′/√γ)` when the lapse was
    *raised* from a positive value — so that `(α/√γ)Π`, the term the first
    evolution equation adds to `∂_t h`, is what it was — and its scale is
    then capped: `Π` is scaled so that `max_ab |(α′/√γ′)Π_ab| ≤ K_max`.

`g′ = (−α′² + β′·β′, β′_i, γ′_ij)` is reassembled only where a step fired,
and `h′ = g′ − η` only in the components that step moved; **a healthy
quantity is returned with the bits it came in with**, so the map is the
identity — exactly — on a state inside every range.

**Idempotent on the state, bit for bit, and on the flag too**, by two
provisions. Every test is relaxed by `8 eps` of the scale of the terms it is
made of: a clamped eigenvalue, recomputed from the recomposed `γ′`, comes
back within a few `eps` of `‖γ′‖` of its bound; a clamped `α²` is a
difference of `1 − h_tt` and `β_iβ^i` and comes back within a few `eps` of
*their* size. Without the slack the second pass would fire again on the
ulps — TreeHydro measured exactly that of its pressure floor, whose state
was a fixed point and whose flag was not. And where a step fired, **the
result is tested again, by the same arithmetic the next call will apply to
it**, and projected again until it passes (at most `BOUNDS_PASSES` times):
the slack alone is not enough where `γ` is ill-conditioned, because
`β_iγ^{ij}β_j` then carries `cond(γ)·eps` of rounding — measured, 18 in
20 000 random states of scale `|h| ≤ 5`, `|Π| ≤ 100` re-fired on the shift
without it (step 8b). Since that last test *is* the next call's test, a
state it passes is one the next call leaves alone.

Pure, pointwise, `isbits` in and out, generic in `T`: callable from a
kernel, identical on every backend and at every thread count. The kernel
that applies it is `gh_bounds_kernel!`; this function is only the ranges.
"""
@inline function bounds_project(h::SVector{NC,T}, Π::SVector{NC,T},
                                bd::StateBounds{T}) where {T}
    h1, Π1, hit, nonfinite = _project_once(h, Π, bd)
    if hit
        # Verify the result with the next call's own arithmetic, and repair
        # what rounding left outside a range. A healthy point never gets
        # here, so this costs nothing where nothing fired.
        for _ in 1:BOUNDS_PASSES
            h2, Π2, again, _ = _project_once(h1, Π1, bd)
            again || break
            h1, Π1 = h2, Π2
        end
    end
    return h1, Π1, hit, nonfinite
end

# How many times a fired projection is re-applied to its own result before
# it is accepted: the guard of a loop that measures one or two, not a
# budget.
const BOUNDS_PASSES = 4

# One application of the five steps of `bounds_project`.
@inline function _project_once(h::SVector{NC,T}, Π::SVector{NC,T},
                               bd::StateBounds{T}) where {T}
    o = one(T)
    z = zero(T)
    slack = BOUNDS_SLACK * eps(T)

    # (1) Non-finite components take Minkowski's value. `map` rebuilds the
    #     vector, but a finite component is passed through as it came.
    nonfinite = !(all(isfinite, h) & all(isfinite, Π))
    h1 = nonfinite ? map(x -> isfinite(x) ? x : zero(x), h) : h
    Π1 = nonfinite ? map(x -> isfinite(x) ? x : zero(x), Π) : Π
    hm = _sym4(h1)
    s0 = _adm_split(hm)

    # (2) The spectrum of γ.
    γ′, γfire = _spectrum_clamp(s0.γ, bd.λ_min, bd.λ_max)
    hmγ = γfire ? _with_spatial(hm, γ′) : hm
    s1 = γfire ? _adm_split(hmγ) : s0

    # (3) The shift, with the projected γ. `!(x ≤ bound)` rather than
    #     `x > bound`: a `NaN` must take the clamping branch, not escape it.
    βmax² = bd.β_max * bd.β_max
    βfire = !(s1.bb ≤ βmax² + slack * βmax²)
    f = βfire ? bd.β_max / sqrt(s1.bb) : o
    hmβ = βfire ? _with_shift(hmγ, (f * s1.β[1], f * s1.β[2], f * s1.β[3])) :
          hmγ
    s2 = βfire ? _adm_split(hmβ) : s1

    # (4) The lapse, by moving g_tt alone. The slack is `8 eps` of the two
    #     terms α² is the difference of, not of α²: at a lapse near α_min
    #     beside a large shift the difference is far smaller than either.
    lo² = bd.α_min * bd.α_min
    hi² = bd.α_max * bd.α_max
    sα = slack * (o + abs(hmβ[1, 1]) + abs(s2.bb))
    αlow = !(s2.α² ≥ lo² - sα)
    αhigh = !(s2.α² ≤ hi² + sα)
    αfire = αlow | αhigh
    αt² = αlow ? lo² : hi²
    hm′ = αfire ? _with_lapse(hmβ, (o - αt²) + s2.bb) : hmβ
    s3 = αfire ? _adm_split(hm′) : s2
    hfire = γfire | βfire | αfire

    # (5) The momentum. `s3` is computed from `hm′` by the same arithmetic
    #     the next call will apply to the stored `h′`, so the lapse the cap
    #     is taken against is the lapse the next test will see, bit for bit.
    raise = αlow & (s0.α² > z) & (s0.detγ > z) & (s3.α² > s0.α²)
    Πr = raise ? Π1 * ((sqrt(s0.α²) / sqrt(s3.α²)) *
                       (sqrt(s3.detγ) / sqrt(s0.detγ))) : Π1
    a = sqrt(s3.α²) / sqrt(s3.detγ)
    m = maximum(x -> abs(a * x), Πr)
    Kfire = !(m ≤ bd.K_max + slack * bd.K_max)
    Π′ = Kfire ? Πr * (bd.K_max / m) : Πr

    h′ = hfire ? _pack10(hm′) : h1
    hit = nonfinite | hfire | raise | Kfire
    return h′, Π′, hit, nonfinite
end

"""
    state_validity(h, Π) -> (detγ, α, hmax, Πmax)

The four numbers of the validity monitor at one point: `det γ_ij`, the
**signed** lapse — `√α²` where `α² = β_iβ^i − g_tt ≥ 0` and `−√(−α²)` where
it is negative, so that a state whose `g^{tt}` has the wrong sign reports a
negative lapse rather than a `NaN` — and the largest component magnitudes
`max |h_ab|` and `max |Π_ab|`.

Where `det γ ≤ 0` the lapse is not defined (the split inverts `γ`), and it
is returned as `floatmax`, which is the neutral value of the minimum the
monitor takes: that point's failure is reported by `det γ`, which is the
row that is about it.
"""
@inline function state_validity(h::SVector{NC,T}, Π::SVector{NC,T}) where {T}
    s = _adm_split(_sym4(h))
    α = !(s.detγ > zero(T)) ? floatmax(T) :
        s.α² ≥ zero(T) ? sqrt(s.α²) : -sqrt(-s.α²)
    return s.detγ, α, maximum(abs, h), maximum(abs, Π)
end

# --- on the mesh -------------------------------------------------------------

"""
    gh_bounds_kernel!(state, diag, origins, spacings, bounds, interior, t)

[`bounds_project`](@ref) at every owned point with `r < r_gate`, **written
back only where it fired**, and three `diag` slots per point: `1`/`0` in
`DIAG_BOUNDS` (it fired), `1`/`0` in `DIAG_BOUNDS_NF` (a component was not
finite), and the point's radius in `DIAG_BOUNDS_R` where it fired (`0`
elsewhere) — whose maximum is the outermost radius the projection reached.

`state` is `statearray(u, U)`: owned points only, no ghosts, so the launch's
index is the point's index there and nothing is added to it — the split the
paste kernel makes, and TreeHydro's reset kernel. No neighbour is read, so
no ghost fill and no scatter is needed: the stage limiter runs on the stage
vector as the integrator formed it.

Only a point where the projection fired is written: a point it left alone
keeps its bits, so a run on which nothing fires is **bit for bit** the run
without the mechanism, which is the control `test/bounds_tests.jl` asserts
and the claim every later experiment rests on. The slots are written
through a branch, never as `flag * r`: a value in the masked region may be
a `NaN` (`CLAUDE.md`).
"""
@kernel function gh_bounds_kernel!(state, diag, @Const(origins),
                                   @Const(spacings), bounds, interior, t)
    I = @index(Global, NTuple)
    b = I[4]
    c = ntuple(d -> I[d], Val(3))
    T = eltype(state)

    x = point_position(origins, spacings, b, I)
    r = interior_radius(interior, t, x)
    if r < bounds.r_gate
        hv = SVector{NC,T}(ntuple(v -> state[c..., v, b], Val(NC)))
        Πv = SVector{NC,T}(ntuple(v -> state[c..., NC + v, b], Val(NC)))
        h′, Π′, hit, nonfinite = bounds_project(hv, Πv, bounds)
        if hit
            ntuple(Val(NC)) do v
                state[c..., v, b] = h′[v]
                state[c..., NC + v, b] = Π′[v]
                nothing
            end
        end
        diag[c..., DIAG_BOUNDS, b] = hit ? one(T) : zero(T)
        diag[c..., DIAG_BOUNDS_NF, b] = nonfinite ? one(T) : zero(T)
        diag[c..., DIAG_BOUNDS_R, b] = hit ? r : zero(T)
    else
        diag[c..., DIAG_BOUNDS, b] = zero(T)
        diag[c..., DIAG_BOUNDS_NF, b] = zero(T)
        diag[c..., DIAG_BOUNDS_R, b] = zero(T)
    end
end

"""
    BoundsAccounting()

The host-side record of what the range projection has done over a run:
how many times it was applied, how many points it moved, how many of those
carried a non-finite component, the outermost radius it fired at, and when
it first fired — for the whole run, and for the chunk being integrated.

**Mutable, host-side, and one per [`evolve!`](@ref)**, handed to every
[`GHProblem`](@ref) the run builds — TreeHydro's `ResetAccounting` and for
its reason: a run rebuilds its problem after every regrid and once per
chunk ([`with_interior`](@ref)), and the totals have to survive that. No
kernel ever receives it.

The counts are *per call* and summed, not read back off the flag slot once
per chunk: a point may fire in one stage and not the next, and a stage's
flags are overwritten by the next stage's. A point that fires in all four
stages of a step is four hits. The chunk fields are what the record's
`bounds_hits`, `bounds_nonfinite` and `bounds_r_max` rows are made of, and
[`take_chunk!`](@ref) resets them.
"""
mutable struct BoundsAccounting
    calls::Int
    hits::Int
    nonfinite::Int
    r_max::Float64               # −1: never fired
    first_t::Float64             # NaN: never fired
    first_r::Float64             # the outermost radius of that first call
    chunk_hits::Int
    chunk_nonfinite::Int
    chunk_r_max::Float64
end

BoundsAccounting() = BoundsAccounting(0, 0, 0, -1.0, NaN, NaN, 0, 0, -1.0)

"""
    take_chunk!(acc::BoundsAccounting) -> (hits, nonfinite, r_max)

The projection's counts since the previous call, and a reset of them — what
one row of the analysis record holds. `r_max = −1` means it did not fire,
which is outside every radius this package compares against (the same
convention as a case with no hole's `interior_radius`).
"""
function take_chunk!(acc::BoundsAccounting)
    out = (hits=acc.chunk_hits, nonfinite=acc.chunk_nonfinite,
           r_max=acc.chunk_r_max)
    acc.chunk_hits = 0
    acc.chunk_nonfinite = 0
    acc.chunk_r_max = -1.0
    return out
end

"""
    apply_bounds!(p::GHProblem, u, t) -> Int

The range projection on the state vector `u` at time `t`, gated on the
problem's interior at `t`: one launch of `gh_bounds_kernel!` over
`statearray(u, p.U)`, then the hit count (and, where it is not zero, the
non-finite count and the outermost radius) reduced out of `p.diag` in block
order and added to `p.accounting`. Returns the number of points moved.

A no-op returning `0` for a case whose `bounds` is `nothing`, and a refusal
for one with bounds and no interior — the gate is a radius about the hole's
center, and a problem built with `interior = nothing` has none.

This is the whole of [`gh_stage_limiter!`](@ref), and the driver calls it
directly on the initial data and on every freshly regridded state, neither
of which went through a stage.
"""
apply_bounds!(p, u, t) = _apply_bounds!(p.case.bounds, p.interior, p, u, t)

_apply_bounds!(::Nothing, interior, p, u, t) = 0

_apply_bounds!(bd::StateBounds, ::Nothing, p, u, t) = throw(ArgumentError(
    "this case carries range bounds but the problem has no interior: the " *
    "projection is gated on r < r_gate from the hole's center, and a " *
    "problem built with `interior = nothing` has no center to measure r " *
    "from. Build the problem with the case's own interior."))

function _apply_bounds!(bd::StateBounds, int::Union{Interior,FittedInterior}, p,
                        u, t)
    T = eltype(p.U.work)
    map_blocks!(gh_bounds_kernel!, p.U, statearray(u, p.U), p.diag.work,
                p.origins, p.spacings, bd, int, T(t))
    # Exact integers in `T`: a block holds at most `N³` points.
    n = round(Int, tofloat64(sum(block_mapreduce(identity, +, zero(T),
                                                 p.diag; vars=DIAG_BOUNDS))))
    acc = p.accounting
    if acc !== nothing
        acc.calls += 1
        if n > 0
            nf = round(Int, tofloat64(sum(block_mapreduce(
                identity, +, zero(T), p.diag; vars=DIAG_BOUNDS_NF))))
            rmax = tofloat64(maximum(block_mapreduce(
                identity, max, zero(T), p.diag; vars=DIAG_BOUNDS_R)))
            acc.hits += n
            acc.nonfinite += nf
            acc.r_max = max(acc.r_max, rmax)
            acc.chunk_hits += n
            acc.chunk_nonfinite += nf
            acc.chunk_r_max = max(acc.chunk_r_max, rmax)
            if isnan(acc.first_t)
                acc.first_t = tofloat64(t)
                acc.first_r = rmax
            end
        end
    end
    return n
end

"""
    gh_stage_limiter!(u, integrator, p::GHProblem, t)

RK4's `stage_limiter!` hook: the range projection ([`apply_bounds!`](@ref))
on every stage vector as the integrator forms it — the three intermediate
stages and the step's result, four calls per step — and nothing at all for
a case whose `bounds` is `nothing`.

**A stage limiter and not a step limiter**, because what it guards against
is a right-hand side evaluated on a state that is not a metric: a `NaN` or a
degenerate `γ` in a stage vector is an `F` of `NaN` at every point whose
stencil reads it, one stage later. It is passed to `solve` as the
`stage_limiter` keyword beside `step_limiter = gh_step_limiter!` — a `solve`
keyword and not an `RK4(; …)` argument, the constructor form being
deprecated and, once the deprecation completes, silently unread
(TreeHydro's "Floors and the atmosphere", amended in its step 8). Read
against the installed `OrdinaryDiffEqLowOrderRK` 2.2.5: RK4 calls it on the
three stage vectors and then on `u`, *before* the FSAL evaluation and
before the step limiter, so the `:pasted` overwrite still has the last word
on the ball it owns.

`integrator` is not read and may be `nothing`. This is the second of the
package's two limiters and the third and last writer of the state
(`CODE.md`, "The interior"): the right-hand side never mutates `u`, the
`:pasted` paste writes the ball `r < r_1` from the step limiter, and this
writes the points inside `r_gate` where a range was violated — and nothing
else writes the state.
"""
function gh_stage_limiter!(u, integrator, p, t)
    apply_bounds!(p, u, t)
    return nothing
end

# --- the validity monitor ----------------------------------------------------

"""
    gh_validity_kernel!(diag, state, origins, spacings, region)

The validity monitor's four slots at every owned point inside `region` (a
[`ShellMask`](@ref) — the layer or the shell outside `r_1`):
`DIAG_DETG = det γ`, `DIAG_LAPSE =` the signed lapse, `DIAG_HMAX = max|h_ab|`
and `DIAG_PIMAX = max|Π_ab|` ([`state_validity`](@ref)). Outside the region
each slot holds the neutral value of the reduction it is read with —
`floatmax` for the two minima, `0` for the two maxima — written through a
branch.

It reads the state array, not the working array: no stencil, no ghosts, no
scatter.
"""
@kernel function gh_validity_kernel!(diag, @Const(state), @Const(origins),
                                     @Const(spacings), region)
    I = @index(Global, NTuple)
    b = I[4]
    c = ntuple(d -> I[d], Val(3))
    T = eltype(diag)
    if is_evolved(region, point_position(origins, spacings, b, I))
        hv = SVector{NC,T}(ntuple(v -> state[c..., v, b], Val(NC)))
        Πv = SVector{NC,T}(ntuple(v -> state[c..., NC + v, b], Val(NC)))
        dγ, α, hm, Πm = state_validity(hv, Πv)
        diag[c..., DIAG_DETG, b] = dγ
        diag[c..., DIAG_LAPSE, b] = α
        diag[c..., DIAG_HMAX, b] = hm
        diag[c..., DIAG_PIMAX, b] = Πm
    else
        diag[c..., DIAG_DETG, b] = floatmax(T)
        diag[c..., DIAG_LAPSE, b] = floatmax(T)
        diag[c..., DIAG_HMAX, b] = zero(T)
        diag[c..., DIAG_PIMAX, b] = zero(T)
    end
end

# The four extremes over one region, `nothing` where the region holds no
# point (a minimum still at its neutral `floatmax`).
function _validity(p, u, region)
    T = eltype(p.U.work)
    map_blocks!(gh_validity_kernel!, p.U, p.diag.work, statearray(u, p.U),
                p.origins, p.spacings, region)
    mn(v) = minimum(block_mapreduce(identity, min, floatmax(T), p.diag;
                                    vars=v))
    mx(v) = maximum(block_mapreduce(identity, max, zero(T), p.diag; vars=v))
    report(x) = x == floatmax(T) ? nothing : tofloat64(x)
    return (detγ=report(mn(DIAG_DETG)), α=report(mn(DIAG_LAPSE)),
            h=tofloat64(mx(DIAG_HMAX)), Π=tofloat64(mx(DIAG_PIMAX)))
end

"""
    validity_rows(p::GHProblem, u, t) -> NamedTuple

The validity monitor's eight rows of the analysis record: the minimum of
`det γ`, the minimum signed lapse, and the largest `|h_ab|` and `|Π_ab|`,
over **the layer** `r_0 ≤ r < r_1` and over **the shell** `r_1 ≤ r < r_1 +
G h` just outside it (`h` the finest spacing present — the `G` points
`CODE.md` compares the interior variants over): `min_detγ_layer`,
`min_α_layer`, `max_h_layer`, `max_Π_layer`, and the same four ending in
`_shell`. All `nothing` for a case with no interior. **From step 8d** the two
bands are the interior's own ([`layer_mask`](@ref), [`shell_mask`](@ref)) —
the same `ShellMask`s for the sphere, the bands about the offset surface for
a tracked geometry — and two more rows reduce over the whole evolved region,
`min_detγ_evolved` and `min_α_evolved`, the lapse-collapse trigger's input.

What it is for: the range projection says *that* a state left the range of
metrics and where; these say how close the layer, and the evolved points
next to it, are to leaving it, chunk by chunk — whether or not the
projection is on — so a run that ends in a degenerate metric has its
approach on the record. `Float64`, as every row is.
"""
function validity_rows(p, u, t)
    int = p.interior
    int === nothing && return (min_detγ_layer=nothing, min_α_layer=nothing,
                               max_h_layer=nothing, max_Π_layer=nothing,
                               min_detγ_shell=nothing, min_α_shell=nothing,
                               max_h_shell=nothing, max_Π_shell=nothing,
                               min_detγ_evolved=nothing, min_α_evolved=nothing)
    T = eltype(p.U.work)
    h = minimum_spacing(T, p.U.forest)
    G = first(p.U.G)
    # The bands through the interior's own masks (amended in step 8d), so
    # that the tracked geometry's layer and shell are its surface's bands and
    # the sphere's are step 8b's `ShellMask`s, value for value.
    layer = _validity(p, u, layer_mask(int, T(t)))
    shell = _validity(p, u, shell_mask(int, T(t), G * h))
    # The whole evolved region (added in step 8d): the minimum lapse over it
    # is the lapse-collapse trigger's input, and `det γ` comes with the pass.
    evolved = _validity(p, u, interior_mask(int, T(t)))
    return (min_detγ_layer=layer.detγ, min_α_layer=layer.α,
            max_h_layer=layer.h, max_Π_layer=layer.Π,
            min_detγ_shell=shell.detγ, min_α_shell=shell.α,
            max_h_shell=shell.h, max_Π_shell=shell.Π,
            min_detγ_evolved=evolved.detγ, min_α_evolved=evolved.α)
end

# --- the evolved region's finiteness -----------------------------------------

@kernel function gh_nonfinite_kernel!(diag, @Const(state), @Const(origins),
                                      @Const(spacings), mask)
    I = @index(Global, NTuple)
    b = I[4]
    c = ntuple(d -> I[d], Val(3))
    T = eltype(diag)
    if is_evolved(mask, point_position(origins, spacings, b, I))
        diag[c..., DIAG_NONFINITE, b] = _fold(ntuple(Val(2 * NC)) do v
            isfinite(state[c..., v, b]) ? zero(T) : one(T)
        end)
    else
        diag[c..., DIAG_NONFINITE, b] = zero(T)
    end
end

"""
    evolved_nonfinite(p::GHProblem, u, t; mask = interior_mask(p.interior, t)) -> Int

How many values of the state vector `u` are `NaN` or `Inf` **at evolved
points** — `r ≥ r_1` from the hole's center at `t`, or everywhere for a
case with no hole.

This is what the driver's finiteness checks read (amended in step 8b):
`max_speed_of`'s refusal and the record's `finite` row used to be
`all(isfinite, u)` over the whole state, which ended a run at the first
`NaN` in the frozen core — a region nothing evolved reads until the
layer's stencils do, and the region the range projection exists to repair.
A `NaN` in the core is a hit, not the end of the run; a `NaN` where the
equations are evolved still is.
"""
function evolved_nonfinite(p, u, t; mask=interior_mask(p.interior,
                                                       eltype(p.U.work)(t)))
    T = eltype(p.U.work)
    map_blocks!(gh_nonfinite_kernel!, p.U, p.diag.work, statearray(u, p.U),
                p.origins, p.spacings, mask)
    return round(Int, tofloat64(sum(block_mapreduce(identity, +, zero(T),
                                                    p.diag;
                                                    vars=DIAG_NONFINITE))))
end
