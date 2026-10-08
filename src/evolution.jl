# One right-hand-side evaluation: the fused kernel, the problem it reads,
# and the time step.
#
# `CODE.md`, "One right-hand-side evaluation", fixes TreeAMR's contract —
# three steps and nothing between them,
#
#     scatter!(U, u)                                          # state → working array
#     fill_ghosts!(U, schedule; boundary = dirichlet(case, t))
#     map_blocks!(gh_rhs_kernel!, U, statearray(du, U), …)
#
# — and, more importantly, the *internal order* of the third step, which
# is a design commitment rather than an implementation detail: it is what
# decides whether the kernel fits in a GPU thread's registers. Written
# naively — load the block, form all of `∂_i h` (30), `∂_i Π` (30),
# `∂_i∂_j h` (60) and the dissipation, then run the algebra — a thread
# holds about 140 `Float64` values before the algebra starts, which is
# already over the 255 32-bit registers it has, and it spills. So the
# kernel is written in **streaming order**:
#
#   1. load `h` and `Π` at the point, form the 30 first derivatives
#      `∂_i h` (which are kept) and, from them and `h`, the coefficient
#      set `g^{ab}, α, β^i, √γ, γ^{ij}` and its two contracted
#      derivatives `∂_iβ^i`, `∂_i(α√γγ^{ij})` — about 30 more values;
#   2. loop over the 10 components: form `∂_iΠ_ab` (3 stencils),
#      `∂_i∂_j h_ab` (3 compact and 3 tensor products) and the
#      dissipation of `h_ab` and `Π_ab` **on the fly**, contract them
#      immediately into two accumulators, and drop them;
#   3. add the source `S0 + Z` and write 20 values of `du`.
#
# Nothing here ever builds an `SVector` of all the derivatives, and
# nothing that can be rebuilt from `h` and `∂_i h` is stored
# (`CLAUDE.md`, "The RHS kernel is written in streaming order"). The
# per-component step is a loop with a small live set because the
# principal part is the *same scalar wave operator for every component*;
# all the coupling is in the coefficients and in the source.
#
# The kernel is **block-local**: it reads its own block's stored points
# and nothing else, so it runs on every backend unchanged. Per-block
# spacings and origins travel to the backend once per chunk, as TreeWave's
# spacings do.
#
# Every stencil it takes comes from a **stencil provider** (step X2a): the
# centered one, [`Centered`](@ref), is the arithmetic above bit for bit, and
# step X2b's closures at an excision surface are another provider of the
# same five methods, so the physics stays one copy.
#
# An `:excised` problem (step X2b) adds a second launch: the zone kernel of
# `excision.jl`, at the evolved points next to the excision surface, with the
# closures; the main kernel skips those points.
#
# Two things this file does *not* do, and must not start doing: it never
# mutates `u` (the right-hand side is a pure function of `(u, t)`,
# TreeAMR's contract), and it never consults the tree — the schedule is
# built once per chunk and replayed.

"""
    convergence_rate(hs, errs)

Least-squares slope of `log(err)` against `log(h)` — the observed order of
a scheme over a sequence of resolutions.

TreeWave's, character for character, because the four packages' tables are
read side by side and a different fit would make them incomparable.
"""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

# The variable slots of the `diag` field set: the characteristic speed the
# time step is taken from, the two constraint monitors of step 4, and the
# indicator that says which points a masked norm counts. Step 5 adds the
# masked error and the interior residual, step 6 the refinement indicator,
# step 8b the range projection's three and the validity monitor's four —
# **appended**, never inserted, for the reason the next paragraph gives.
#
# `DIAG_CGH` and `DIAG_MOM` are the *first* of a contiguous run — four and
# three slots — because `block_mapreduce` reduces a contiguous range of
# variables and nothing else: a device cannot be handed an arbitrary index
# vector cell by cell.
const DIAG_SPEED = 1          # λ = α√(tr γ^{ij}) + |β|
const DIAG_CGH = 2            # C_a = Γ_a + H_a, a = t, x, y, z   (2:5)
const DIAG_HAM = 6            # the ADM Hamiltonian constraint ℋ
const DIAG_MOM = 7            # the ADM momentum constraint ℳ_i   (7:9)
const DIAG_MASK = 10          # 1 where the point is evolved, 0 inside r_1
const DIAG_ERR = 11           # ‖u − u_exact‖, masked to the evolved region
const DIAG_RES = 12           # the same, inside the layer r_0 ≤ r < r_1
const DIAG_DRIFT = 13         # |h_tt − h_tt,exact| in a shell at the horizon
const DIAG_TAU = 14           # the Löhner indicator τ, masked inside r_1
const DIAG_BOUNDS = 15        # 1 where the range projection fired, else 0
const DIAG_BOUNDS_NF = 16     # 1 where it found a non-finite component
const DIAG_BOUNDS_R = 17      # the radius where it fired, 0 elsewhere
const DIAG_DETG = 18          # det γ in the monitor's region (floatmax outside)
const DIAG_LAPSE = 19         # the signed lapse there (floatmax outside)
const DIAG_HMAX = 20          # max_ab |h_ab| there (0 outside)
const DIAG_PIMAX = 21         # max_ab |Π_ab| there (0 outside)
const DIAG_NONFINITE = 22     # non-finite values at an evolved point
const NDIAG = 22

# **The error slots are magnitudes, not components (proposed in step 5.)**
# `CODE.md`'s analysis table says "`|u − u_exact|` per component into
# `diag`", which would be twenty more slots — more than tripling a field
# set that is `nvars × (N+1)³ × nblocks` — for a number the record reads
# as one norm. What is stored instead is the pointwise Euclidean magnitude
# over the twenty components, whose volume-weighted L2 *is* the L2 norm of
# the whole state error; the per-component split, if a component is ever
# in question, is a targeted kernel and not a permanent cost on every run.

# The position of an **owned** point, formed exactly as TreeAMR forms it —
# the same origin, the same spacing, the same expression in the same order
# — so that a mask, an interior profile or a Dirichlet value computed here
# lands on the value `coordinates(fs, b, idx)` gives and not merely near
# it. TreeAMR's own `coordinates_kernel!` writes
# `origin[d] + (I[d] - off[d]) * h` with `off` a whole cell along a
# vertex-like dimension, and every field set in this package is
# vertex-centered (`CODE.md`, "Field sets and layout"), which
# [`GHProblem`](@ref) checks so that this line may assume it.
@inline function point_position(origins, spacings, b::Int, I)
    h = spacings[b]
    origin = origins[b]
    off = oftype(h, 1 // 1)
    # Spelled out rather than `ntuple(d -> …, Val(3))`: no closure for a device
    # to compile as a call (amended 2026-10-05).
    return (origin[1] + (I[1] - off) * h, origin[2] + (I[2] - off) * h,
            origin[3] + (I[3] - off) * h)
end

# --- the kernel-side stencil contractions -----------------------------------
#
# `stencils.jl`'s `apply_stencil` and `apply_mixed_stencil` are the
# host-side reference *definitions*; these are what the kernel evaluates,
# and the two have to agree. Both sum from the lowest offset to the
# highest, and the mixed one runs its inner sum along the second axis —
# which is a property of the operator and not of the implementation, since
# the other order differs in the last place (`CODE.md`, "Finite-difference
# stencils"). Nothing is materialised: each contraction is formed and
# consumed where it is written.
#
# They address the working array by a **linear index**: a base index for
# the point and one stride per axis, rather than a `(i, j, k, v, b)` tuple
# per load. The working array is dense and column-major — TreeAMR
# allocates it — so the two are the same array element, and the tests
# compare the kernel against `apply_stencil`'s cartesian spelling. The
# reason is measured (step 3): at `q = 4` the stencil half of the kernel
# costs **1376 ns** per point with the cartesian index and **573 ns** with
# the linear one, for bit-identical output. Five-dimensional index
# arithmetic at every one of the ~1600 loads a point takes is not a cost
# the compiler removes, and it is not arithmetic this scheme is about.

# A left fold over a tuple, in order. `sum` would do, but its association
# is not part of its interface, and the order of a stencil's summation is
# part of this operator's.
@inline _fold(t::Tuple) = _fold(t[1], Base.tail(t))
@inline _fold(s, t::Tuple) = _fold(s + t[1], Base.tail(t))
@inline _fold(s, ::Tuple{}) = s

# The working array's strides: `(per axis), per variable, per block`. One
# call per point, from `size` alone, so it holds for any dense array on
# any backend.
@inline function work_strides(work)
    n1, n2, n3 = size(work, 1), size(work, 2), size(work, 3)
    sv = n1 * n2 * n3
    return (1, n1, n1 * n2), sv, sv * size(work, 4)
end

# The linear index of point `I = (i1, i2, i3, block)`'s first variable in a
# **state-layout** array `(N, N, N, nvars, nblocks)` — `du` — and the stride
# between its variables (added 2026-10-05, for the kernel that stores `F` as it
# goes). From `size` alone, as `work_strides` is.
@inline function state_offset(du, I)
    n1, n2, n3 = size(du, 1), size(du, 2), size(du, 3)
    sd = n1 * n2 * n3
    return I[1] + n1 * (I[2] - 1) + n1 * n2 * (I[3] - 1) + sd * size(du, 4) * (I[4] - 1), sd
end

# `∑_k w[k] u[base + (k − 1 − r)·stride]`: the contraction every
# one-dimensional operator here is, whatever the weights mean. It is
# **generated**, so the sum is written out term by term from the lowest offset
# to the highest — the left fold `_fold` would form, the same arithmetic — and
# the offsets are constants. **(Amended 2026-10-05:** it was an
# `ntuple(Val(n)) do … end`, whose closure a device compiles as a real call
# unless inlining is forced, with the weights passed through the stack —
# `CODE.md`, "The right-hand side on an H200".**)** The generator builds an
# expression and nothing else, so it calls no method of the caller's type. The
# body carries an explicit `:inline` meta: a generated method is not inlined
# because its generator is marked `@inline`, and without the meta a device
# compiles each stencil as a call (measured: 2.2 against 1.1 ns a point).
@generated function axis_stencil(w::SVector{n,T}, work, base::Int,
                                 stride::Int) where {n,T}
    r = (n - 1) ÷ 2
    ex = :(w[1] * work[base + $(-r) * stride])
    for k in 2:n
        ex = :($ex + w[$k] * work[base + $(k - 1 - r) * stride])
    end
    return Expr(:block, Expr(:meta, :inline), :(@inbounds $ex))
end

# The mixed derivative: the tensor product of two first-derivative vectors,
# outer sum along the first axis, inner along the second. It reads the edge
# ghosts TreeAMR fills unconditionally, which is why `∂_i∂_j` needs no
# wider halo than `∂_i∂_i`. Generated for the same reason as `axis_stencil`,
# with the same two left folds the `ntuple` version formed.
@generated function mixed_stencil(w::SVector{n,T}, work, base::Int, s1::Int,
                                  s2::Int) where {n,T}
    r = (n - 1) ÷ 2
    outer = :(nothing)
    for a in 1:n
        inner = :(w[1] * work[base + $(a - 1 - r) * s1 + $(-r) * s2])
        for e in 2:n
            inner = :($inner + w[$e] * work[base + $(a - 1 - r) * s1 + $(e - 1 - r) * s2])
        end
        outer = a == 1 ? :(w[1] * $inner) : :($outer + w[$a] * $inner)
    end
    return Expr(:block, Expr(:meta, :inline), :(@inbounds $outer))
end

# --- the stencil provider (added in step X2a) --------------------------------
#
# `CODE.md`, "Excision": the physics is one copy. The right-hand side asks a
# *provider* for every stencil it takes, and the provider decides which
# weights and which taps: the centered one below is the kernel's arithmetic,
# and step X2b's closure provider reads the per-point codes `k±` and the
# closure table at the points next to an excision surface. The right-hand
# side itself — the coefficients, the streaming order, the source — is not
# repeated. **(Amended in step X4:** `main`'s rewrite of the kernel for a
# device split the body into [`gh_rhs_head`](@ref) and [`gh_rhs_pi`](@ref);
# the provider is now their argument, so the one copy is those two functions,
# and every caller — the store, the two-vector form, the zone kernel — reaches
# the stencils through them.**)**

"""
    StencilProvider

What the right-hand side takes its stencils from (added in step X2a): the
argument of [`gh_rhs_head`](@ref), [`gh_rhs_pi`](@ref) and therefore of
[`gh_rhs_store!`](@ref) and [`gh_rhs_at_point`](@ref). A provider is an
`isbits` value built per point, and it answers five questions, each about
**one component at one point**, addressed as the stencils address the
working array — by `base`, the linear index of that component at the point
(`var + (v − 1)·sv` for `h_v`, plus `NC·sv` for `Π_v`):

| method | returns | the caller scales by |
|---|---|---|
| `d1(S, work, base, d)` | the first derivative along `d`, unit spacing | `1/h` |
| `d2(S, work, base, d)` | the second derivative along `d` | `1/h²` |
| `dmix(S, work, base, i, j)` | `∂_i∂_j`, `i < j`: outer sum along `i`, inner along `j` | `1/h²` |
| `ko(S, work, base, d)` | the Kreiss–Oliger contraction along `d` | `ε_KO/h` |
| `adv(S, β_d, ∂f_d, work, base, d)` | the derivative that multiplies `β^d` | — |

The first four are raw contractions on unit spacing, as
[`derivative_weights`](@ref) and [`dissipation_weights`](@ref) are, so the
caller's `1/h`, `1/h²` and `ε_KO/h` are applied where they always were.

`adv` is the one that differs: it is asked for the derivative in the two
advective terms `β^k ∂_k h_ab` and `β^k ∂_k Π_ab` — and **only** there —
and it is handed the shift's component `β_d` and the *scaled* derivative
`∂f_d` the right-hand side has already formed along `d` (for `h`, the head's
`∂_d h`, which also feeds the coefficients; for `Π`, `d1/h`). It returns a
scaled derivative. The centered provider returns `∂f_d` itself; step X2b's
lopsided blend returns a mix of `∂f_d` and an upwinded derivative, the side
read from the sign of `β_d` (`CODE.md`, "Excision", and
[`lopsided_weights`](@ref)), and so needs `1/h` of its own.

**`d1` of `h` is asked twice per component and axis** (amended in step X4):
once by the head, for the coefficients and the advection of `h`, and once by
[`gh_rhs_pi`](@ref), for `∂_i(α√γγ^{ij}) ∂_j h` — `main`'s choice of three
stencils of cached loads over thirty values kept live across the source. A
provider must answer both alike, which any provider whose methods are
functions of `(work, base, d)` does.
"""
abstract type StencilProvider end

"""
    Centered(T, ::Val{q}, st) -> Centered{T,q}

The centered stencils of order `q` — the provider of every kernel but the
excision's zone kernel, and the one the original signatures of
[`gh_rhs_head`](@ref), [`gh_rhs_pi`](@ref), [`gh_rhs_store!`](@ref) and
[`gh_rhs_at_point`](@ref) build (added in step X2a). It holds the working
array's per-axis strides `st` ([`work_strides`](@ref)) and nothing else.

Its methods are the [`axis_stencil`](@ref) and [`mixed_stencil`](@ref)
calls of the kernel, with the weights the kernel used —
[`derivative_weights`](@ref) for `m = 1, 2` and [`dissipation_weights`](@ref)
at rank `q/2 + 1`, `@generated` constants formed **inside** each method — the
same strides and summation order; its `adv` returns the `∂f_d` it is handed,
so the kernel forms no new stencil. Inlined, the code is the kernel's without
a provider (`CODE.md`, "One right-hand-side evaluation", measured in steps
X2a and X4). **(Amended in step X4:** until then it carried the three weight
vectors as fields, 160 bytes a point that a device which does not inline the
right-hand side passes through the stack — the `+5 %` step X3 measured on the
H200. Formed in the method they are constants wherever the method is
compiled.**)**
"""
struct Centered{T,q} <: StencilProvider
    st::NTuple{3,Int}
end

@inline Centered(::Type{T}, ::Val{q}, st::NTuple{3,Int}) where {T,q} =
    Centered{T,q}(st)

# The five methods. Their names are short because the right-hand side reads
# as the equation with them; a function that calls them must not have a
# local of the same name (`d1`, `d2` and `ko` are common ones elsewhere).
@inline d1(S::Centered{T,q}, work, base::Int, d::Int) where {T,q} =
    axis_stencil(derivative_weights(T, Val(q), Val(1)), work, base, S.st[d])
@inline d2(S::Centered{T,q}, work, base::Int, d::Int) where {T,q} =
    axis_stencil(derivative_weights(T, Val(q), Val(2)), work, base, S.st[d])
@inline dmix(S::Centered{T,q}, work, base::Int, i::Int, j::Int) where {T,q} =
    mixed_stencil(derivative_weights(T, Val(q), Val(1)), work, base, S.st[i],
                  S.st[j])
@inline ko(S::Centered{T,q}, work, base::Int, d::Int) where {T,q} =
    axis_stencil(dissipation_weights(T, dissipation_rank(Val(q))), work, base,
                 S.st[d])
@inline adv(S::Centered, β_d, ∂f_d, work, base::Int, d::Int) = ∂f_d

"""
    gh_rhs_head(T, work, Hwork, inner, b, var, st, sv, inv_h, γ0, γ2, εh,
                ::Val{q}, ::Val{HASH}, ::Val{DISS}) -> (; ∂ₜh, msrc, β, divβ, divA, A)
    gh_rhs_head(S::StencilProvider, T, work, Hwork, inner, b, var, sv, inv_h,
                γ0, γ2, εh, ::Val{HASH}, ::Val{DISS})

What every component of `F(u)` at one owned point needs. In `CODE.md`'s streaming
order, as amended 2026-10-05:

1. the state at the point, the 30 first derivatives `∂_i h` and the coefficient
   set built from them once;
2. `∂ₜh_ab = β^i ∂_i h_ab + (α/√γ) Π_ab + Q_d h_ab`, complete;
3. the source `−α√γ (S0 + Z)` ([`gh_node_source_lean`](@ref)), from `∂ₜh`;
4. the coefficients the Π components read: `β^i`, `∂_i β^i`, `∂_i(α√γ γ^{ij})`
   ([`metric_divergences`](@ref)) and `A^{ij} = α√γ γ^{ij}`.

The Π components themselves are [`gh_rhs_pi`](@ref), one at a time, after it.

**The source comes before the Π components, not after them** (amended
2026-10-05, `CODE.md`, "The right-hand side on an H200"). The source is the
register-heaviest part of the evaluation: computed last, it was live together with
the ten accumulated `∂ₜΠ` and spilled. Computed here, with `∂ₜh` final, it leaves
ten numbers behind.

**The `∂_t g` the source is given is `∂ₜh`, dissipation included** — the time
derivative of the numerical solution, which is what the reduced source's
`−Γ^ν ∂_ν g_ab` means. The difference is `O(h^{q+1})`, the dissipation's own order
**(recorded in step 3**, where `CODE.md` had said only "from `h`, `∂_i h`, `∂_t h`
and the coefficients"**)**.

**Every stencil comes from a provider** (step X2a; amended in step X4). The
second form takes one, `S` ([`StencilProvider`](@ref)), in place of `st` and
`q`, and asks it for each `d1`, `ko` and `adv`; the first builds
[`Centered`](@ref) from `st` and `q` and calls the second, which inlined is
the head without a provider.

`var` is the point's linear index in `work`, `st` the per-axis strides and `sv` the
per-variable one ([`work_strides`](@ref)); `γ0` is this point's constraint-damping
rate, a **profile** the caller evaluates (`CODE.md`, "Gauge and constraint
damping"). No closure is formed here or below: every loop is `@ntuple`, so nothing
is a call on a device whether or not inlining is forced.
"""
@inline function gh_rhs_head(::Type{T}, work, Hwork, inner, b::Int, var::Int, st,
                             sv::Int, inv_h, γ0, γ2, εh, ::Val{q}, ::Val{HASH},
                             ::Val{DISS}) where {T,q,HASH,DISS}
    return gh_rhs_head(Centered(T, Val(q), st), T, work, Hwork, inner, b, var, sv,
                       inv_h, γ0, γ2, εh, Val(HASH), Val(DISS))
end

@inline function gh_rhs_head(S::StencilProvider, ::Type{T}, work, Hwork, inner,
                             b::Int, var::Int, sv::Int, inv_h, γ0, γ2, εh,
                             ::Val{HASH}, ::Val{DISS}) where {T,HASH,DISS}
    # (1) the state at the point, the 30 first derivatives of `h`, and the
    #     coefficients built from them once.
    hv = SVector{NC,T}(@ntuple 10 v -> (@inbounds work[var + (v - 1) * sv]))
    Πv = SVector{NC,T}(@ntuple 10 v -> (@inbounds work[var + (NC + v - 1) * sv]))
    ∂h = @ntuple 3 d -> inv_h * SVector{NC,T}(@ntuple 10 v ->
        d1(S, work, var + (v - 1) * sv, d))
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ

    # (2) `∂ₜh`, complete: the advection, the momentum and the dissipation.
    ∂ₜh = SVector{NC,T}(@ntuple 10 v ->
        _dth(S, work, var + (v - 1) * sv, εh, β, a_div, ∂h[1][v], ∂h[2][v], ∂h[3][v],
             Πv[v], Val(DISS)))

    # (3) the source, from the state, its gradients and `∂ₜh` — the gauge source
    #     read at the owned point (`Hsrc` has no ghosts to read) or, for the
    #     algebraic source, evaluated from the state and the `∂g` just formed
    #     (added 2026-10-02).
    Hl, dHl = gauge_source(T, Hwork, inner, b, Val(HASH), hv, ∂ₜh, ∂h)
    msrc = gh_node_source_lean(g4, gu4, α, sqrtγ, ∂ₜh, ∂h, Hl, dHl, γ0, γ2)

    # (4) the coefficients of the Π components.
    A = _scale(α * sqrtγ, γu)                     # A^{jk} = α√γ γ^{jk}
    divβ, divA = metric_divergences(gu4, α, β, γu, sqrtγ, ∂h)
    return (; ∂ₜh, msrc, β, divβ, divA, A)
end

# One component of `∂ₜh`: `β^i ∂_i h + (α/√γ) Π`, then the dissipation —
# the package's summation order since step 3. The advective derivatives go
# through the provider's `adv` (step X2a), which the centered provider answers
# with the derivative it is handed.
@inline function _dth(S, work, bh, εh, β, a_div, ∂h1, ∂h2, ∂h3, Π_v,
                      ::Val{DISS}) where {DISS}
    s = β[1] * adv(S, β[1], ∂h1, work, bh, 1) + β[2] * adv(S, β[2], ∂h2, work, bh, 2) +
        β[3] * adv(S, β[3], ∂h3, work, bh, 3) + a_div * Π_v
    DISS || return s
    return s + εh * (ko(S, work, bh, 1) + ko(S, work, bh, 2) + ko(S, work, bh, 3))
end

"""
    gh_rhs_pi(T, work, var, st, sv, v, inv_h, εh, head, ::Val{q}, ::Val{DISS})
    gh_rhs_pi(S::StencilProvider, T, work, var, sv, v, inv_h, εh, head, ::Val{DISS})

Component `v` of `∂ₜΠ` without its source: nine stencils of `h_v` (`∂_i h`, the
compact `∂_i∂_i`, the tensor-product `∂_i∂_j`) and seven of `Π_v` (the value, `∂_i Π`,
the dissipation), formed, contracted with [`gh_rhs_head`](@ref)'s coefficients and
dropped:

    β^i ∂_i Π + (∂_iβ^i) Π + ∂_i(α√γ γ^{ij}) ∂_j h + α√γ γ^{ij} ∂_i∂_j h + Q_d Π

in the package's summation order since step 3. `∂_i h_v` is formed again here
rather than kept from the head (amended 2026-10-05): three stencils of cached
loads cost less than thirty values live across the source. `v` may be a run-time
index — the kernel loops over the components — or a constant. The advective
derivatives of `Π` go through the provider's `adv`; `∂_i(α√γγ^{ij}) ∂_j h` takes
the plain `d1` (step X2a). The second form takes the provider in place of `st`
and `q` (amended in step X4).
"""
@inline function gh_rhs_pi(::Type{T}, work, var::Int, st, sv::Int, v::Int, inv_h, εh,
                           head, ::Val{q}, ::Val{DISS}) where {T,q,DISS}
    return gh_rhs_pi(Centered(T, Val(q), st), T, work, var, sv, v, inv_h, εh, head,
                     Val(DISS))
end

@inline function gh_rhs_pi(S::StencilProvider, ::Type{T}, work, var::Int, sv::Int,
                           v::Int, inv_h, εh, head, ::Val{DISS}) where {T,DISS}
    inv_h² = inv_h * inv_h
    β, divβ, divA, A = head.β, head.divβ, head.divA, head.A
    bh = var + (v - 1) * sv                       # this component of `h`
    bΠ = bh + NC * sv                             # and of `Π`
    Π_v = @inbounds work[bΠ]
    ∂h1 = inv_h * d1(S, work, bh, 1)
    ∂h2 = inv_h * d1(S, work, bh, 2)
    ∂h3 = inv_h * d1(S, work, bh, 3)
    ∂Π1 = inv_h * d1(S, work, bΠ, 1)
    ∂Π2 = inv_h * d1(S, work, bΠ, 2)
    ∂Π3 = inv_h * d1(S, work, bΠ, 3)
    s = β[1] * adv(S, β[1], ∂Π1, work, bΠ, 1) + β[2] * adv(S, β[2], ∂Π2, work, bΠ, 2) +
        β[3] * adv(S, β[3], ∂Π3, work, bΠ, 3) + divβ * Π_v +
        divA[1] * ∂h1 + divA[2] * ∂h2 + divA[3] * ∂h3
    s += A[1, 1] * (inv_h² * d2(S, work, bh, 1)) +
         A[2, 2] * (inv_h² * d2(S, work, bh, 2)) +
         A[3, 3] * (inv_h² * d2(S, work, bh, 3))
    ∂xy = inv_h² * dmix(S, work, bh, 1, 2)
    ∂xz = inv_h² * dmix(S, work, bh, 1, 3)
    ∂yz = inv_h² * dmix(S, work, bh, 2, 3)
    s += 2 * (A[1, 2] * ∂xy + A[1, 3] * ∂xz + A[2, 3] * ∂yz)
    if DISS
        s += εh * (ko(S, work, bΠ, 1) + ko(S, work, bΠ, 2) + ko(S, work, bΠ, 3))
    end
    return s
end

"""
    gh_rhs_store!(du, o, sd, T, work, Hwork, inner, b, var, st, sv, inv_h, γ0, γ2, εh,
                  ::Val{q}, ::Val{HASH}, ::Val{DISS})
    gh_rhs_store!(du, o, sd, S::StencilProvider, T, work, Hwork, inner, b, var, sv,
                  inv_h, γ0, γ2, εh, ::Val{HASH}, ::Val{DISS})

`F(u)` at one owned point written straight into `du`, component `v` at
`du[o + (v − 1) sd]`, as each is finished (added 2026-10-05). The ten `∂ₜh` are
stored after [`gh_rhs_head`](@ref), and the ten `∂ₜΠ` one at a time by a **run-time
loop** over [`gh_rhs_pi`](@ref): nothing of `F` stays live longer than it takes to
store it.

This is the kernel where nothing modifies `F`: no hole, and the `:excised`
variant's evolved points — the centered ones in the main kernel, through
[`Centered`](@ref) or the lopsided blend's provider, and the zone points in the
zone kernel, through the closures (amended in step X4). The layer variants
combine `F` with the layer's terms and take it as two vectors
([`gh_rhs_at_point`](@ref)), from the same two functions. Measured on an H200, with
the source before the components and the loop at run time, it is 1.2 ns a point
against 8.5 for step 3's kernel (`CODE.md`, "The right-hand side on an H200").
"""
@inline function gh_rhs_store!(du, o::Int, sd::Int, ::Type{T}, work, Hwork, inner,
                               b::Int, var::Int, st, sv::Int, inv_h, γ0, γ2, εh,
                               ::Val{q}, ::Val{HASH}, ::Val{DISS}) where {T,q,HASH,DISS}
    return gh_rhs_store!(du, o, sd, Centered(T, Val(q), st), T, work, Hwork, inner, b,
                         var, sv, inv_h, γ0, γ2, εh, Val(HASH), Val(DISS))
end

@inline function gh_rhs_store!(du, o::Int, sd::Int, S::StencilProvider, ::Type{T},
                               work, Hwork, inner, b::Int, var::Int, sv::Int, inv_h,
                               γ0, γ2, εh, ::Val{HASH},
                               ::Val{DISS}) where {T,HASH,DISS}
    head = gh_rhs_head(S, T, work, Hwork, inner, b, var, sv, inv_h, γ0, γ2, εh,
                       Val(HASH), Val(DISS))
    ∂ₜh = head.∂ₜh
    @nexprs 10 v -> (@inbounds du[o + (v - 1) * sd] = ∂ₜh[v])
    msrc = head.msrc
    for v in 1:NC
        @inbounds du[o + (NC + v - 1) * sd] =
            gh_rhs_pi(S, T, work, var, sv, v, inv_h, εh, head, Val(DISS)) + msrc[v]
    end
    return nothing
end

"""
    gh_rhs_at_point(T, work, Hwork, inner, b, var, st, sv, inv_h, γ0, γ2, εh,
                    ::Val{q}, ::Val{HASH}, ::Val{DISS}) -> (∂ₜh, ∂ₜΠ)
    gh_rhs_at_point(S::StencilProvider, T, work, Hwork, inner, b, var, sv,
                    inv_h, γ0, γ2, εh, ::Val{HASH}, ::Val{DISS}) -> (∂ₜh, ∂ₜΠ)

`F(u)` at one owned point, as two `SVector{10}`s: the fused right-hand side of
`(EXPANDED)`, with the Kreiss–Oliger term and the source already in it.

    ∂ₜh_ab = β^i ∂_i h_ab + (α/√γ) Π_ab                    + Q_d h_ab
    ∂ₜΠ_ab = β^i ∂_i Π_ab + (∂_iβ^i) Π_ab
           + α√γ γ^{ij} ∂_i∂_j h_ab + ∂_i(α√γ γ^{ij}) ∂_j h_ab
           − α√γ (S0_ab + Z_ab)                            + Q_d Π_ab

It is [`gh_rhs_head`](@ref) and the ten [`gh_rhs_pi`](@ref) unrolled, the same
arithmetic as [`gh_rhs_store!`](@ref) in the same order, so the two agree bit for
bit wherever they are compiled alike. The interior variants call it, because they
combine `F` with the layer's terms before storing (amended 2026-10-05; until then
it was the whole kernel's body).

**Every stencil comes from a provider** (added in step X2a). The second form
takes one, `S` ([`StencilProvider`](@ref)), in place of `st` and `q`; the first
builds [`Centered`](@ref) from `st` and `q` and calls the second. The centered
provider's `adv` is the identity on the derivative it is handed, so the two forms
are the same arithmetic in the same order; step X2b's closure provider is the
second form at the points next to an excision surface (`CODE.md`, "Excision").

It is a **plain function called from the kernel** rather than the kernel's own body
(restructured in step 5). The reason is `CODE.md`'s rule that `F` is never
evaluated where `w = 0`: the frozen core holds finite but stale data on which `F`
may be `NaN`, and `0 · NaN = NaN`, so the kernel has to branch *around* this whole
computation. KernelAbstractions refuses a `return` statement anywhere in a kernel
body, closures included, so the branch cannot be an early exit.

The source is [`gh_node_source_lean`](@ref) and the coefficient derivatives are
[`metric_divergences`](@ref) — the kernel's spellings of [`gh_node_source`](@ref)
and [`metric_derivatives`](@ref), the functions [`gh_node_rhs_expanded`](@ref)
calls, which is what makes that function the reference this kernel is checked
against on analytic data. The two are not bit-identical and are not expected to
be: the spellings sum in other orders, and one body reached from two call sites is
contracted into fused multiply-adds differently (`CODE.md`, "Measured results").
"""
@inline function gh_rhs_at_point(::Type{T}, work, Hwork, inner, b::Int,
                                 var::Int, st, sv::Int, inv_h, γ0, γ2, εh,
                                 ::Val{q}, ::Val{HASH},
                                 ::Val{DISS}) where {T,q,HASH,DISS}
    return gh_rhs_at_point(Centered(T, Val(q), st), T, work, Hwork, inner, b, var,
                           sv, inv_h, γ0, γ2, εh, Val(HASH), Val(DISS))
end

@inline function gh_rhs_at_point(S::StencilProvider, ::Type{T}, work, Hwork,
                                 inner, b::Int, var::Int, sv::Int, inv_h, γ0,
                                 γ2, εh, ::Val{HASH},
                                 ::Val{DISS}) where {T,HASH,DISS}
    head = gh_rhs_head(S, T, work, Hwork, inner, b, var, sv, inv_h, γ0, γ2, εh,
                       Val(HASH), Val(DISS))
    msrc = head.msrc
    ∂ₜΠ = SVector{NC,T}(@ntuple 10 v ->
        gh_rhs_pi(S, T, work, var, sv, v, inv_h, εh, head, Val(DISS)) + msrc[v])
    return head.∂ₜh, ∂ₜΠ
end

"""
    gh_rhs_point!(du, work, Hwork, origins, spacings, bg, damping, γ2, ε_KO, interior,
                  t, tw, t_f, rate, trail, fitp, cls, blend, I, ::Val{G}, ::Val{q},
                  ::Val{HASH}, ::Val{DISS}, ::Val{INT})

The right-hand side at the one owned point `I = (i1, i2, i3, block)`: the body
[`gh_rhs_kernel!`](@ref) had until 2026-10-05, moved here unchanged so that the
kernel can run it for one point (`W = 1`: every device, and the types SIMD.jl has
no lanes for) or for each point of a group of `W` that straddles the hole's layer
([`gh_rhs_lanes!`](@ref)) — and, for `:excised`, at every point whatever `W` is
(merged 2026-10-08). The kernel's docstring describes it.
"""
@inline function gh_rhs_point!(du, work, Hwork, origins, spacings, bg, damping, γ2,
                               ε_KO, interior, t, tw, t_f, rate, trail, fitp, cls,
                               blend, I, ::Val{G}, ::Val{q}, ::Val{HASH}, ::Val{DISS},
                               ::Val{INT}) where {G,q,HASH,DISS,INT}
    b = I[4]
    inner = (I[1], I[2], I[3])                    # state-layout index
    T = eltype(du)

    inv_h = inv(spacings[b])

    # The point's linear index in the working array — the owned index plus
    # the ghost width along each axis — and the strides the stencils step
    # by. `var` is the first variable's base; variable `v` is `var + (v−1)·sv`.
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb +
          (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]

    # The position, for the damping profile and for the interior. Three
    # fused multiply-adds per point, and the compiler drops them where
    # neither asks (a constant `γ0` and `INT === :none`).
    x = point_position(origins, spacings, b, I)
    γ0 = damping_rate(damping, t, x)
    # The Kreiss–Oliger amplitude over the cell, per point (step 8c): the
    # identity on a number, which is what every case but a calibration
    # passes, and a profile of the distance to the hole otherwise
    # ([`dissipation_rate`](@ref), the `γ0` pattern).
    εh = dissipation_rate(ε_KO, t, x) * inv_h

    # **Every name a closure below captures is assigned once in this body**
    # (amended 2026-09-26). Lowering boxes a captured variable that is
    # assigned twice — in two branches, even when a `Val` compiles one of
    # them away — and a `Core.Box` is untyped: the `:none` branch's `∂ₜh`,
    # captured by its store, shared its name with the layer's, and every
    # right-hand side on the CPU allocated ~570 bytes a point through the
    # box while no device would compile it at all ("unsupported dynamic
    # function invocation"). Hence `ρk`/`tk` in the levered core and the
    # `w`/`ρ` renaming below.
    if INT === :none
        # No hole: `F` itself, each component stored as it is finished
        # ([`gh_rhs_store!`](@ref), amended 2026-10-05). This branch forms no
        # closure, so a device compiles it without calls whether or not the
        # backend forces inlining (`CODE.md`, "The right-hand side on an H200").
        o, sd = state_offset(du, I)
        gh_rhs_store!(du, o, sd, T, work, Hwork, inner, b, var, st, sv, inv_h, γ0,
                      γ2, εh, Val(q), Val(HASH), Val(DISS))
    elseif INT === :excised
        # **Excision (added in step X2b)**: the class of the point decides,
        # and nothing else does — the classes are the single source of truth
        # for what is excised, built once per problem (`build_excision`). A
        # centered point is the `:none` branch's call — the same function with
        # the same arguments — or, inside the lopsided advection's shell, the
        # same through the `Lopsided` provider. An excised point's `du` is
        # zero and `F` is not evaluated. A zone point is the zone kernel's,
        # launched next, and nothing is written here. The class array has the
        # working array's spatial strides and one variable. **(Amended in step
        # X4:** stored as it is finished, `main`'s spill-free store, and
        # closure-free like the `:none` branch, so a device compiles this
        # branch without calls too. And where the blend's weight is zero the
        # point takes the `:none` call itself, not `Lopsided` at `λ = 0`: the
        # arithmetic is the same, but `main`'s head compiled around a provider
        # with a branch in its `adv` fuses `metric_quantities`' `muladd`s
        # differently, and the exterior beyond the shell differed from the
        # blend-free run in the last place at 496 of 25 165 points of the
        # suite's fixture.**)**
        cb = 1 + (b - 1) * sv + (I[1] + G[1] - 1) * st[1] +
             (I[2] + G[2] - 1) * st[2] + (I[3] + G[3] - 1) * st[3]
        cl = cls[cb]
        oe, sde = state_offset(du, I)
        λe = blend_weight(blend, x)
        if cl == CLASS_CENTERED
            if blend === nothing || iszero(λe)
                gh_rhs_store!(du, oe, sde, T, work, Hwork, inner, b, var, st, sv,
                              inv_h, γ0, γ2, εh, Val(q), Val(HASH), Val(DISS))
            else
                gh_rhs_store!(du, oe, sde, Lopsided(T, Val(q), st, inv_h, λe), T,
                              work, Hwork, inner, b, var, sv, inv_h, γ0, γ2, εh,
                              Val(HASH), Val(DISS))
            end
        elseif cl == CLASS_EXCISED
            for v in 1:(2 * NC)
                @inbounds du[oe + (v - 1) * sde] = zero(T)
            end
        end
    else
        # The interior's view of the point (step 8d): the radius for step
        # 5's sphere, and for the tracked geometry the radius with the two
        # surfaces' radii along the ray — `interior_point`, which evaluates
        # the shape's series only between its bounding spheres. The three
        # branches below are the same for both.
        g = interior_point(interior, t, x)
        if is_frozen(interior, g)
            # `F` is not evaluated: this is the branch `CLAUDE.md` says must
            # come before the stencils. `du = 0` for the analytic variants;
            # the fitted core (step 8e) relaxes toward the cached target at
            # the full rate, `du = −ρ_max (u − u_fit)`.
            if INT === :fitted && (fitp !== nothing || !iszero(trail))
                # Step 8′'s levers (the side-dependent ramp does not touch
                # the core, where `ρ = ρ_max` already; the exact target does).
                ρk = interior.ρ_max
                tk = _lever_target(tw, inner, b, t, t_f, fitp, x)
                ntuple(Val(2 * NC)) do v
                    du[inner..., v, b] = (rate ? _cached_rate(tw, inner, b, v) :
                                          zero(T)) -
                                         ρk * (work[var + (v - 1) * sv] - tk[v])
                    nothing
                end
            elseif INT === :fitted
                ρc = interior.ρ_max
                # With the target's rate (step 8) the core follows the moving
                # target instead of lagging it by `|∂_t u_fit|/ρ_max`: `du =
                # ∂_t u_fit − ρ_max (u − u_fit)`, `∂_t u_fit` the cache's slope.
                if rate
                    ntuple(Val(2 * NC)) do v
                        du[inner..., v, b] = _cached_rate(tw, inner, b, v) -
                                             ρc * (work[var + (v - 1) * sv] -
                                                   _cached_target(tw, inner, b, v,
                                                                  t, t_f))
                        nothing
                    end
                else
                    ntuple(Val(2 * NC)) do v
                        du[inner..., v, b] = -ρc * (work[var + (v - 1) * sv] -
                                                    _cached_target(tw, inner, b, v,
                                                                   t, t_f))
                        nothing
                    end
                end
            else
                ntuple(Val(2 * NC)) do v
                    du[inner..., v, b] = zero(T)
                    nothing
                end
            end
        else
            ∂ₜh, ∂ₜΠ = gh_rhs_at_point(T, work, Hwork, inner, b, var, st, sv,
                                       inv_h, γ0, γ2, εh, Val(q), Val(HASH),
                                       Val(DISS))
            if is_outside(interior, g)
                ntuple(Val(NC)) do v
                    du[inner..., v, b] = ∂ₜh[v]
                    du[inner..., NC + v, b] = ∂ₜΠ[v]
                    nothing
                end
            elseif INT === :fitted && (fitp !== nothing || !iszero(trail))
                # Step 8′'s levers in the layer: the ramp narrowed on the
                # trailing side (`trail`), the target the latest fit carried
                # by its center to `t` exactly (`fitp`), or both.
                wl, ρl = iszero(trail) ? interior_profiles(interior, g) :
                         _trail_profiles(interior, g, t, x, trail)
                tg = _lever_target(tw, inner, b, t, t_f, fitp, x)
                ol = one(T) - wl
                ntuple(Val(NC)) do v
                    du[inner..., v, b] =
                        wl * ∂ₜh[v] +
                        (rate ? ol * _cached_rate(tw, inner, b, v) : zero(T)) -
                        ρl * (work[var + (v - 1) * sv] - tg[v])
                    du[inner..., NC + v, b] =
                        wl * ∂ₜΠ[v] +
                        (rate ? ol * _cached_rate(tw, inner, b, NC + v) : zero(T)) -
                        ρl * (work[var + (NC + v - 1) * sv] - tg[NC + v])
                    nothing
                end
            elseif INT === :fitted
                # The fitted layer (step 8e): the same profiles, relaxing
                # toward the cached target, `A + (t − t_f) S`, two loads a
                # variable and no evaluation of the fit here.
                # (Its own names: `w` and `ρ` are captured below, and a
                # captured variable assigned twice in one function is boxed.)
                wf, ρf = interior_profiles(interior, g)
                if rate
                    # The target's rate where `F` is switched off (step 8):
                    # `(1 − w) ∂_t u_fit`, so that a target that is the moving
                    # solution is a steady state of the layer too.
                    of = one(T) - wf
                    ntuple(Val(NC)) do v
                        du[inner..., v, b] =
                            wf * ∂ₜh[v] + of * _cached_rate(tw, inner, b, v) -
                            ρf * (work[var + (v - 1) * sv] -
                                  _cached_target(tw, inner, b, v, t, t_f))
                        du[inner..., NC + v, b] =
                            wf * ∂ₜΠ[v] + of * _cached_rate(tw, inner, b, NC + v) -
                            ρf * (work[var + (NC + v - 1) * sv] -
                                  _cached_target(tw, inner, b, NC + v, t, t_f))
                        nothing
                    end
                else
                    ntuple(Val(NC)) do v
                        du[inner..., v, b] =
                            wf * ∂ₜh[v] - ρf * (work[var + (v - 1) * sv] -
                                                _cached_target(tw, inner, b, v, t, t_f))
                        du[inner..., NC + v, b] =
                            wf * ∂ₜΠ[v] - ρf * (work[var + (NC + v - 1) * sv] -
                                                _cached_target(tw, inner, b, NC + v,
                                                               t, t_f))
                        nothing
                    end
                end
            else
                w, ρ = interior_profiles(interior, g)
                # `u_exact` is the layer's *target* (step 8c): the case's
                # background unless the interior names another metric.
                he, Πe, _ = background_state(layer_target(interior, bg), t, x)
                ntuple(Val(NC)) do v
                    du[inner..., v, b] =
                        w * ∂ₜh[v] - ρ * (work[var + (v - 1) * sv] - he[v])
                    du[inner..., NC + v, b] =
                        w * ∂ₜΠ[v] - ρ * (work[var + (NC + v - 1) * sv] - Πe[v])
                    nothing
                end
            end
        end
    end
    return nothing
end

"""
    gh_rhs_lanes!(du, work, Hwork, origins, spacings, bg, damping, γ2, ε_KO, interior,
                  t, tw, t_f, rate, trail, fitp, I, ::Val{W}, ::Val{G}, ::Val{q},
                  ::Val{HASH}, ::Val{DISS}, ::Val{INT})

The right-hand side at the `W` owned points `I[1] … I[1] + W − 1` of one row, as
one evaluation on SIMD.jl's `Vec{W,T}` lanes (added 2026-10-05; `CODE.md`, "The
right-hand side on a CPU"). The package's own [`gh_rhs_store!`](@ref) and
[`gh_rhs_at_point`](@ref) run with `Vec{W,T}` for `T`, reading the working array
through [`Lanes`](@ref): every stencil load is a load of `W` neighbouring values
and every store a store of `W`. Each lane does its point's scalar operations in the
scalar order, so `du` is the scalar kernel's to roundoff: bit for bit on the gauge
wave, and about an eps of the terms on a hole, where StaticArrays' `muladd`s are
fused into FMAs differently in the two contexts (`test/simd_tests.jl`).

- The damping rate and the Kreiss–Oliger amplitude are evaluated per lane, at each
  lane's position: a profile varies along the row.
- **Without a hole** the `W` points take [`gh_rhs_store!`](@ref), as one point does.
- **With a hole**, the `W` points take the lanes when every one of them is outside
  the layer — most of the mesh — and [`gh_rhs_point!`](@ref) one at a time
  otherwise, so that the frozen core, the layer and its target are the scalar
  code's, branch for branch.
- **`:excised`** never comes here: [`gh_rhs_kernel!`](@ref) gives it the scalar
  path (merged 2026-10-08).

A lane's `sqrt` is the instruction, so a degenerate metric gives a `NaN` here where
the scalar code throws a `DomainError`; the record's `finite` and the next chunk's
speed check read it.
"""
@inline function gh_rhs_lanes!(du, work, Hwork, origins, spacings, bg, damping, γ2,
                               ε_KO, interior, t, tw, t_f, rate, trail, fitp, I,
                               ::Val{W}, ::Val{G}, ::Val{q}, ::Val{HASH}, ::Val{DISS},
                               ::Val{INT}) where {W,G,q,HASH,DISS,INT}
    b = I[4]
    inner = (I[1], I[2], I[3])                    # the first lane's index
    T = eltype(du)
    V = Vec{W,T}
    inv_h = inv(spacings[b])
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb +
          (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]
    # Each lane's position, and the two profiles there.
    xs = ntuple(l -> point_position(origins, spacings, b, (I[1] + l - 1, I[2], I[3])),
                Val(W))
    γ0 = Vec(ntuple(l -> damping_rate(damping, t, xs[l]), Val(W)))
    εh = Vec(ntuple(l -> dissipation_rate(ε_KO, t, xs[l]), Val(W))) * inv_h
    wl = lanes(Val(W), work)
    Hl = lanes(Val(W), Hwork)
    dl = lanes(Val(W), du)
    o, sd = state_offset(du, I)
    if INT === :none
        gh_rhs_store!(dl, o, sd, V, wl, Hl, inner, b, var, st, sv, inv_h, γ0, V(γ2),
                      εh, Val(q), Val(HASH), Val(DISS))
    elseif all(ntuple(l -> is_outside(interior, interior_point(interior, t, xs[l])),
                      Val(W)))
        ∂ₜh, ∂ₜΠ = gh_rhs_at_point(V, wl, Hl, inner, b, var, st, sv, inv_h, γ0, V(γ2),
                                   εh, Val(q), Val(HASH), Val(DISS))
        @nexprs 10 v -> (dl[o + (v - 1) * sd] = ∂ₜh[v])
        @nexprs 10 v -> (dl[o + (NC + v - 1) * sd] = ∂ₜΠ[v])
    else
        for l in 1:W
            gh_rhs_point!(du, work, Hwork, origins, spacings, bg, damping, γ2, ε_KO,
                          interior, t, tw, t_f, rate, trail, fitp, nothing, nothing,
                          (I[1] + l - 1, I[2], I[3], b), Val(G), Val(q), Val(HASH),
                          Val(DISS), Val(INT))
        end
    end
    return nothing
end

"""
    gh_rhs_kernel!(du, work, Hwork, origins, spacings, bg, damping, γ2, ε_KO,
                   interior, t, tw, t_f, rate, trail, fitp, cls, blend, ::Val{G},
                   ::Val{q}, ::Val{HASH}, ::Val{DISS}, ::Val{INT}, ::Val{W})

The right-hand side at one owned point: `F(u)`, modified inside the hole by
`CODE.md`'s `(INTERIOR)`,

    ∂_t u = w(r) · F(u)  −  ρ(r) · (u − u_exact(x, t)) .

Without a hole (`INT === :none`), `F` is stored component by component as it
is finished ([`gh_rhs_store!`](@ref), from 2026-10-05). The interior variants
take it as two vectors ([`gh_rhs_at_point`](@ref)) to combine with the layer's
terms.

`du` is in **state layout** (no ghosts, so the global index is used as it
comes); `work` is the ghosted working array (so the same index plus `G`).
`Hwork` is the gauge source's working array or `nothing`.

The **six** `Val`s are built once per chunk in [`GHProblem`](@ref) and
resolved when the kernel compiles: the ghost width, the difference order,
whether there is a gauge source, whether there is dissipation, — added
in step 5 — which of `CODE.md`'s interior variants is running, `:none`
meaning there is no hole, and — added 2026-10-05 — the SIMD width `W`. Building
them per evaluation would recompile or dispatch dynamically at every RK stage
(`CLAUDE.md`).

**`W` points at a time on the CPU** (added 2026-10-05; `CODE.md`, "The right-hand
side on a CPU"). With `W = 1` — every device, and the types SIMD.jl has no lanes
for — each work item is its point ([`gh_rhs_point!`](@ref)). With `W > 1` the launch
is the same, `(N, N, N, nblocks)`, and only a *leader* works
([`is_lane_leader`](@ref)): the item at the start of each group of `W` along the
first axis, which evaluates the group on SIMD lanes ([`gh_rhs_lanes!`](@ref)). A
row whose length `W` does not divide ends in an **overlapping** group, the last `W`
points of the row: the points it shares with the group before are computed again
and stored again with the same bits, so no lane is ever outside the row and nothing
is masked. The other items do nothing, and cost less than one percent.

**The interior's fifth `Val` is the variant and not a `Bool`
(proposed in step 5.)** `CODE.md` and `PLAN.md` call it "has interior";
`:none`, `:damped`, `:pasted` and `:frozen` say that and *which*, in one
parameter, and the three variants differ in the kernel — `:frozen` has
`ρ ≡ 0` and `:pasted` freezes the whole ball `r < r_1` — so a `Bool` would
have needed a second parameter beside it.

`ε_KO` is the case's Kreiss–Oliger amplitude, a number or — from step 8c —
a [`HorizonDissipation`](@ref) evaluated per point by
[`dissipation_rate`](@ref), which is the identity on a number; and the
layer's `u_exact` is the interior's [`layer_target`](@ref), the background
itself unless the interior names another metric (step 8c).

**`:excised` (added in step X2b)** has a branch of its own, before the
interior's: `cls` is the problem's class array and `blend` its lopsided
blend (both `nothing` for every other variant). A centered point is the
`:none` branch's `F` — through the [`Lopsided`](@ref) provider where the
blend is on — an excised point's `du` is zero with `F` not evaluated, and a
zone point is left to [`gh_zone_kernel!`](@ref), which [`gh_rhs!`](@ref)
launches next (`CODE.md`, "Excision"). **It runs scalar on the CPU too**
(proposed in the main merge, 2026-10-08): whatever the problem's `W`, every work
item of an `:excised` problem is its own point, [`gh_rhs_point!`](@ref) — the path
`W = 1` takes on every device — so no lane ever holds an excised value, and the
zone and frame-dragged kernels are launches of their own, one point an item. SIMD
lanes for excision are future work (`CODE.md`, "Possible extensions").

**The three branches, in the order they must be in.** The core predicate
is asked *before* any stencil is touched, because the frozen core holds
finite but stale data on which `F` may be `NaN` and `0 · NaN = NaN`
(`CLAUDE.md`). Outside `r_1` the answer is `F` itself and not `1·F − 0·(…)`,
which also saves the analytic solution's dual pass at every point of the
evolved region — `u_exact` is evaluated in the layer and nowhere else.
"""
@kernel function gh_rhs_kernel!(du, @Const(work), Hwork, @Const(origins),
                                @Const(spacings), bg, damping, γ2, ε_KO,
                                interior, t, tw, t_f, rate, trail, fitp,
                                cls, blend, ::Val{G}, ::Val{q},
                                ::Val{HASH}, ::Val{DISS},
                                ::Val{INT}, ::Val{W}) where {G,q,HASH,DISS,INT,W}
    I = @index(Global, NTuple)                    # (i1, i2, i3, block)
    # `:excised` takes the scalar path whatever `W` is (proposed in the main
    # merge, 2026-10-08): every item its own point, as on a device.
    if W == 1 || INT === :excised
        gh_rhs_point!(du, work, Hwork, origins, spacings, bg, damping, γ2, ε_KO,
                      interior, t, tw, t_f, rate, trail, fitp, cls, blend, I, Val(G),
                      Val(q), Val(HASH), Val(DISS), Val(INT))
    elseif is_lane_leader(I[1], size(du, 1), W)
        gh_rhs_lanes!(du, work, Hwork, origins, spacings, bg, damping, γ2, ε_KO,
                      interior, t, tw, t_f, rate, trail, fitp, I, Val(W), Val(G),
                      Val(q), Val(HASH), Val(DISS), Val(INT))
    end
end

"""
    gh_paste_kernel!(u, origins, spacings, bg, interior, t, ::Val{G})

The `:pasted` variant's overwrite: the analytic state written into every
owned point with `r < r_1`, straight into the **state** array.

`CODE.md`, "Why a smooth layer and not a hard paste": overwriting a ball
with the analytic solution is `(INTERIOR)` in the limit `ρ → ∞` on a step
profile, and it is implemented exactly through RK4's `step_limiter!`, as
TreeHydro implements its atmosphere reset. It is one of the **two** places
in the package where the state is written outside the integrator's own
arithmetic, and [`gh_step_limiter!`](@ref) is its only caller; the other
is step 8b's range projection, from the stage limiter
([`gh_stage_limiter!`](@ref)). `CLAUDE.md`, "The RHS never mutates `u`":
three writers, and do not add a fourth.

The core rule applies here as everywhere the analytic solution is written
into a grid: inside `r_0` the query goes to the sphere `r_0` along the ray
([`core_position`](@ref)), because the solution is singular at the center.
What is written is the interior's *target* ([`layer_target`](@ref), added in
step 8c) — the case's background unless the interior names another metric,
which is how step 8c's E0 puts a hard step of a wrong solution at `r_1`.
"""
@kernel function gh_paste_kernel!(u, @Const(origins), @Const(spacings), bg,
                                  interior, t, ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    T = eltype(u)

    x = point_position(origins, spacings, b, I)
    # Inside the layer's outer surface: `r < r_1` for the sphere, and below
    # the offset surface for the tracked geometry (step 8d).
    if !is_outside(interior, interior_point(interior, t, x))
        vals = case_state_tuple(layer_target(interior, bg), interior, t, x)
        ntuple(Val(2 * NC)) do v
            u[inner..., v, b] = vals[v]
            nothing
        end
    end
end

"""
    gh_speed_kernel!(speed, work, origins, spacings, mask, ::Val{G})

GHSO2's conservative bound on the coordinate characteristic speed at one
owned point, `λ = α √(tr γ^{ij}) + |β|`, written into the `diag` field
set's speed slot — and **zero where the mask says the point is not
evolved**.

It reads the point and nothing else — no stencil, no ghosts — so it is the
cheapest kernel in the package and can be run at every chunk boundary
without thinking about it. `mesh_mapreduce(max)` over its output is
[`max_speed`](@ref); see `CODE.md`, "The time step".

**The mask is not optional (added in step 5.)** `CODE.md` lists the speed
kernel among the ones that write zero for `r < r_1`, and the reason is the
same as everywhere else: the frozen core holds data that is not a
numerical solution, and a *degenerate* metric there — which is what a
stale core looks like once it has been interpolated by a regrid — gives a
`NaN` or an enormous `λ`, which would then set the time step for the whole
hierarchy. A branch and not a multiplication, because `0 · NaN = NaN`.
The layer's own speeds go with it; they are bounded by the evolved
region's, since `w ≤ 1` scales the characteristics down and the driver
bounds the relaxation separately (`ρ_max · dt ≤ 1` at every rate it runs:
the default `4/M` is checked against it, and the grid rate is it).
"""
@kernel function gh_speed_kernel!(speed, @Const(work), @Const(origins),
                                  @Const(spacings), mask, ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    c = ntuple(d -> I[d] + G[d], Val(3))
    T = eltype(speed)

    if is_evolved(mask, point_position(origins, spacings, b, I))
        hv = SVector{NC,T}(ntuple(v -> work[c..., v, b], Val(NC)))
        _, _, α, β, γu, _ = metric_quantities(_sym4(hv))
        speed[inner..., DIAG_SPEED, b] =
            α * sqrt(γu[1, 1] + γu[2, 2] + γu[3, 3]) +
            sqrt(β[1] * β[1] + β[2] * β[2] + β[3] * β[3])
    else
        speed[inner..., DIAG_SPEED, b] = zero(T)
    end
end

"""
    GHProblem(U::FieldSet, schedule, case::GHCase; q, t = 0)

Everything one right-hand-side evaluation needs, built once per chunk: the
state's field set and its ghost schedule, the gauge source sampled into
its own field set (or `nothing` where the background is harmonic), the
`diag` field set the analysis kernels write into, the per-block geometry
on whatever backend the state lives on, the case, and the four `Val`s the
kernel specialises on.

It is the `p` of SciML's `f!(du, u, p, t)`, and the application writes
`f!` itself — [`gh_rhs!`](@ref) — rather than reaching for a
`semidiscretize`-style wrapper, which is TreeAMR's contract and TreeWave's
pattern.

`q` is the finite-difference order and has no default: it fixes the ghost
width `G = q/2 + 1`, which the field set was already built with, and the
prolongation order `p = q + 2` that the schedule's `Operators` must carry
(`CODE.md`, "The interface-order rule"). The constructor checks the first
of those; TreeAMR's `check_operators` and step 4's table are what check
the second.

The gauge source is **sampled here**, at `t`, because that is what "after
every regrid" means when every chunk builds a fresh problem — and because
a background that reaches this point is static, so the time it is sampled
at cannot matter. `CODE.md`, "Gauge and constraint damping", and the
refusal in [`GHCase`](@ref) are the two halves of that sentence.

`accounting` is the run's [`BoundsAccounting`](@ref), or `nothing` (added in
step 8b): the host-side record the range projection adds its hits to. It is
*handed in* rather than made here, because a run builds a fresh problem
after every regrid and the totals have to survive that — [`evolve!`](@ref)
makes one per run. For a case with bounds the constructor also asserts that
the projection's gate lies deeper than every point an evolved stencil reads
([`check_bounds_gate`](@ref)), beside the interior's own radius checks and
for the same reason.

`simd_width` is how many neighbouring points the kernel evaluates at once on SIMD
lanes (added 2026-10-05; `CODE.md`, "The right-hand side on a CPU"): `nothing`, the
default, is the host's ([`default_simd_width`](@ref) — four `Float64` on AVX2 and
aarch64, eight with AVX-512, one on a device), and `1` is the scalar kernel. The
lanes compute the scalar kernel's numbers to roundoff, and the same numbers at any
thread count.
"""
struct GHProblem{T,G,q,HASH,DISS,INT,W,F,S,H,D,O,V,C,I,A,X,Y,Z}
    U::F
    schedule::S
    Hsrc::H                      # the sampled gauge source, or `nothing`
    diag::D                      # the speed, the monitors, the errors
    # Per block, on the backend the field set lives on. The origins are
    # what the interior profiles, the masks and the damping profile turn a
    # cell index into a position with, and they are uploaded here because
    # that is where the per-chunk metadata belongs.
    origins::O
    spacings::V
    case::C
    interior::I                  # an `Interior` or a `FittedInterior` at this chunk's ρ_max, or `nothing`
    accounting::A                # the run's `BoundsAccounting`, or `nothing`
    # The `:fitted` target (step 8e): the cache the kernel reads (a 40-variable
    # `G = 0` field set, or `nothing`), the fits it was filled from (host-side,
    # `(latest, previous)`, or `nothing`) and the time it was filled at.
    target::X
    fits::Y
    t_target::T
    # Whether the `:fitted` target's rate is fed forward (step 8): the cache's
    # slope then includes the fit's translation with the track, and the
    # kernel adds `(1 − w) ∂_t u_fit` in the layer and the core.
    target_rate::Bool
    # Step 8′'s levers: the trailing side's ramp narrowing `trail` (0 = off)
    # and the exact target (the latest fit evaluated in the kernel at `t`).
    trail::T
    target_exact::Bool
    # The `:excised` variant's classes, closures and monitor (step X2b): an
    # `ExcisionData` built with the problem, or `nothing`.
    excision::Z
    hasdirichlet::Bool
    valG::Val{G}
    valq::Val{q}
    valH::Val{HASH}
    valdiss::Val{DISS}
    valint::Val{INT}
    # The SIMD width of the kernel (added 2026-10-05): `W` points at a time on the
    # CPU, `1` on a device (`lanes.jl`, `default_simd_width`).
    valsimd::Val{W}
end

function GHProblem(U::FieldSet{T,3}, schedule, case::GHCase{T}; q::Integer,
                   t=zero(T), interior=case.interior, margin_check=true,
                   accounting=nothing, target=nothing, fits=nothing,
                   t_target=zero(T), target_rate::Bool=false, trail=zero(T),
                   target_exact::Bool=false, simd_width=nothing) where {T}
    q ≥ 2 && iseven(q) || throw(ArgumentError(
        "the finite-difference order must be even and at least 2, so that " *
        "the centered stencils have an integer half-width q/2 and CODE.md's " *
        "ghost width G = q/2 + 1 covers them, but q=$q"))
    U.nvars == 2NC || throw(ArgumentError(
        "the evolved state is h (1:10) and Π (11:20), $(2NC) variables " *
        "(CODE.md, \"Field sets and layout\"), but this field set has " *
        "$(U.nvars)"))
    all(g -> g == q ÷ 2 + 1, U.G) || throw(ArgumentError(
        "CODE.md fixes the ghost width at G = q/2 + 1 = $(q ÷ 2 + 1) for " *
        "q=$q — one more than the derivatives need, because the " *
        "Kreiss–Oliger operator of order q + 2 reaches one point further — " *
        "but this field set has G = $(U.G). A narrower halo makes the " *
        "dissipation read a ghost that was never filled; a wider one is " *
        "memory the scheme does not use."))
    all(c -> c === :vertex, U.centering) || throw(ArgumentError(
        "CODE.md fixes this package's field sets as vertex-centered — a " *
        "finite-difference scheme wants restriction along a stagger, which " *
        "is injection and exact for any data — but this field set is " *
        "$(U.centering). The masks and the interior profiles turn an owned " *
        "index into a position assuming it (`point_position`), so a " *
        "staggered set would be evaluated half a cell from where its " *
        "values sit."))
    interior isa FittedSpec && throw(ArgumentError(
        "this case's interior is a FittedSpec — the rule a tracked geometry " *
        "is built by, once per chunk, from the horizon that was found — and a " *
        "problem needs the geometry itself: pass `interior = " *
        "fitted_interior(spec, track, forest, G; t, n_L)`, which is what " *
        "evolve! does at every chunk (CODE.md, \"The tracked geometry\")."))
    backend = get_backend(U.work)
    # The mirrors are the forest's and the hook's faces the case's (added
    # 2026-10-02): a face the case calls reflecting and the mesh does not
    # would take Dirichlet data the hook was never asked for, and the reverse
    # a mirror of a solution with no symmetry.
    U.forest.reflecting == case.reflecting || throw(ArgumentError(
        "the mesh reflects at $(U.forest.reflecting) and the case at " *
        "$(case.reflecting): build the forest from the case (gh_forest, " *
        "hole_forest), which passes the case's faces."))
    # And so is the seam (added 2026-10-04): a seam the case does not know of
    # turns a solution with no such symmetry, and a seam the mesh lacks would
    # hand the hook two faces the case has no data for.
    seam_dims(U.forest) == seam_dims(case) || throw(ArgumentError(
        "the mesh has the rotating seam $(something(seam_dims(U.forest), "none")) " *
        "and the case $(something(seam_dims(case), "none")): build the forest " *
        "from the case (gh_forest, hole_forest), which passes the case's seam."))

    # The gauge source's kind: none on a harmonic background, the closed form
    # where the case carries one (added 2026-10-02) — it is then the kernel
    # argument itself, in place of the sampled `Hsrc` array — and the sample
    # otherwise.
    HASH = case.gauge !== nothing ? :algebraic : !isharmonic(case.background)
    Hsrc = if HASH === :algebraic
        case.gauge
    elseif HASH
        fs = FieldSet{T}(U.forest, 2NC; G=0, centering=U.centering,
                         parity=even_parity(U.forest, 2NC),
                         rotation=identity_rotation(U.forest, 2NC), backend=backend)
        sample_gauge_source!(fs, case.background, t; interior=interior)
        fs
    else
        nothing
    end
    diag = FieldSet{T}(U.forest, NDIAG; G=0, centering=U.centering,
                       parity=even_parity(U.forest, NDIAG),
                       rotation=identity_rotation(U.forest, NDIAG), backend=backend)

    origins = to_backend(backend, block_origins(U.forest, T))
    spacings = to_backend(backend, block_spacings(U.forest, T))
    DISS = has_dissipation(case.ε_KO)
    hasdirichlet = has_outer_face(case)

    # CODE.md asks for the two radius requirements "at every regrid", and a
    # fresh problem is built after every one — so this is where they are
    # checked, on the mesh as it now is. `margin_check = false` exists for
    # the one caller that has a reason not to: a test that builds a problem
    # in order to watch the check fire elsewhere.
    INT = interior_variant(interior)
    if interior !== nothing && margin_check
        check_interior_radii(U.forest, interior, case.background, q ÷ 2 + 1;
                             t=t, center=case.center)
        check_bounds_gate(U.forest, interior, case.bounds, q; t=t)
    end

    INT === :fitted && target === nothing && throw(ArgumentError(
        "a :fitted interior relaxes toward a target the kernel reads from a " *
        "cache field set, and this problem has none: pass `target = " *
        "target_cache(U)` and fill it (fill_target!), which is what evolve! " *
        "does (CODE.md, \"The fitted target\")."))
    # The `:excised` variant's classes (step X2b), from the geometry and the
    # state in `U`'s working array — built here, once, with the problem.
    excision = INT === :excised ?
               build_excision(U, schedule, case, interior; q=q, t=T(t)) : nothing
    # The kernel's SIMD width (added 2026-10-05): the host's, unless the caller
    # asks for another — `simd_width = 1` is the scalar kernel.
    W = simd_width === nothing ? default_simd_width(T, backend, U.forest.N) :
        check_simd_width(simd_width, T, backend, U.forest.N)
    return GHProblem{T,U.G,Int(q),HASH,DISS,INT,W,typeof(U),typeof(schedule),
                     typeof(Hsrc),typeof(diag),typeof(origins),
                     typeof(spacings),typeof(case),typeof(interior),
                     typeof(accounting),typeof(target),typeof(fits),
                     typeof(excision)}(
        U, schedule, Hsrc, diag, origins, spacings, case, interior,
        accounting, target, fits, T(t_target), target_rate, T(trail),
        target_exact, excision, hasdirichlet, Val(U.G),
        Val(Int(q)), Val(HASH), Val(DISS), Val(INT), Val(W))
end

"""
    with_interior(p::GHProblem, interior) -> GHProblem

The same problem carrying a different [`Interior`](@ref) — what the driver
builds at the start of every chunk once it knows that chunk's `dt`, since
the rate is chosen per chunk — the default `4/M` checked against `1/dt`, or
the grid rate `ρ_max_factor/dt` itself (`CODE.md`, "The profiles and their
parameters"; amended in step 8c′).

It shares the field sets, the schedule, the sampled gauge source and the
uploaded geometry: rebuilding a whole [`GHProblem`](@ref) would re-sample
the gauge source, which is the most expensive setup phase there is and
which nothing about a new `ρ_max` invalidates. It shares the run's
[`BoundsAccounting`](@ref) too, which is what that record is for.
"""
function with_interior(p::GHProblem{T,G,q,HASH,DISS,INT0,W}, interior;
                       fits=p.fits, target=p.target,
                       t_target=p.t_target,
                       target_rate::Bool=p.target_rate, trail=p.trail,
                       target_exact::Bool=p.target_exact) where {T,G,q,HASH,DISS,
                                                                 INT0,W}
    INT = interior_variant(interior)
    INT === :fitted && target === nothing && throw(ArgumentError(
        "a :fitted interior needs the problem's target cache; this problem " *
        "has none (see GHProblem's `target`)."))
    # The excision is carried while the geometry is the one it was built for
    # — which on a run is always, the geometry being frozen — rebuilt for
    # another, and dropped for another variant (step X2b).
    excision = INT !== :excised ? nothing :
               p.excision !== nothing && p.excision.interior === interior ?
               p.excision :
               build_excision(p.U, p.schedule, p.case, interior; q=q, t=zero(T))
    return GHProblem{T,G,q,HASH,DISS,INT,W,typeof(p.U),typeof(p.schedule),
                     typeof(p.Hsrc),typeof(p.diag),typeof(p.origins),
                     typeof(p.spacings),typeof(p.case),typeof(interior),
                     typeof(p.accounting),typeof(target),typeof(fits),
                     typeof(excision)}(
        p.U, p.schedule, p.Hsrc, p.diag, p.origins, p.spacings, p.case,
        interior, p.accounting, target, fits, T(t_target), target_rate,
        T(trail), target_exact, excision, p.hasdirichlet,
        p.valG, p.valq, p.valH, p.valdiss, Val(INT), p.valsimd)
end

"""
    refill_target(p::GHProblem, t; fits = p.fits) -> GHProblem

Fill `p`'s target cache at time `t` from `fits = (latest, previous)` on
`p`'s geometry ([`fill_target!`](@ref)), and return the problem that reads
it: the same field sets, with the fits and the fill time recorded. What the
driver calls after every fit and whenever the refill rule says the tracked
center has moved far enough (step 8e).
"""
function refill_target(p::GHProblem{T}, t; fits=p.fits) where {T}
    p.target === nothing && throw(ArgumentError(
        "this problem has no target cache to fill."))
    fill_target!(p.target, p.origins, p.spacings, p.interior, fits, T(t);
                 rate=p.target_rate)
    return with_interior(p, p.interior; fits=fits, t_target=T(t))
end

# The exact target's kernel argument (step 8′): the latest fit's parameters
# and coefficients, or `nothing` — which compiles the lever away.
_exact_fit(p::GHProblem) =
    p.target_exact && p.fits !== nothing ? (p.fits[1].params, p.fits[1].coeffs) :
    nothing

# Step 8′'s target at one point: the cache's linear continuation, or with
# `fitp` the latest fit evaluated at `(x, t)` — carried by its tracked center
# exactly rather than linearly between refills.
@inline function _lever_target(tw, inner, b, t, t_f, fitp, x)
    T = eltype(tw)
    if fitp === nothing
        return SVector{2NC,T}(ntuple(v -> _cached_target(tw, inner, b, v, t, t_f),
                                     Val(2NC)))
    else
        h, Π = fit_state(fitp[1], fitp[2], x, t)
        return SVector{2NC,T}(ntuple(v -> v ≤ NC ? T(h[v]) : T(Π[v - NC]),
                                     Val(2NC)))
    end
end

# Step 8′'s side-dependent ramp: `ρ`'s ramp fraction narrowed by the factor
# `1 − trail · ζ` with `ζ = max(0, −n̂ · v̂)` about the tracked center, so that
# on the trailing side, where grid points leave the layer, `ρ` reaches
# `ρ_max` nearer the offset surface; `w` is unchanged.
@inline function _trail_profiles(int::FittedInterior{T}, g, t, x, trail) where {T}
    c = center_at(int.center, t)
    v = int.center.v
    s = sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
    ζ = if iszero(s) || iszero(g.r)
        zero(T)
    else
        max(zero(T), -((x[1] - c[1]) * v[1] + (x[2] - c[2]) * v[2] +
                       (x[3] - c[3]) * v[3]) / (g.r * s))
    end
    return _layer_profiles(g.r, g.r_0, g.r_1, int.ρ_max, int.w_ramp,
                           int.ρ_ramp * (one(T) - trail * ζ))
end

# The cache's working array, or `nothing`, for the kernels.
target_work(::Nothing) = nothing
target_work(fs::FieldSet) = fs.work

# The gauge source's working array, or `nothing` where there is none. The
# kernel never asks whether it has one — its `Val` already said.
gauge_work(::Nothing) = nothing
gauge_work(fs::FieldSet) = fs.work
gauge_work(src::KerrSchildSource) = src

"""
    gh_rhs!(du, u, p::GHProblem, t)

One right-hand-side evaluation, in SciML's `f!(du, u, p, t)` signature:
scatter the state into the working array, fill the ghosts (with the
Dirichlet hook of this `t`, where the case has a physical boundary), and
launch the fused kernel.

Three steps and nothing between them — TreeAMR's contract, `CODE.md`'s
"One right-hand-side evaluation". **`u` is never mutated**: the working
array is scratch, and the interior treatment of step 5 is a *term* in the
right-hand side rather than a write to the state.

The hook is rebuilt here, at every evaluation, with that evaluation's `t`.
That is not a cost worth avoiding — it is a closure over two `isbits`
values — and the alternative is a boundary that lags the solution by up to
a chunk (`CLAUDE.md`, "Hooks depend on time").
"""
function gh_rhs!(du, u, p::GHProblem, t)
    scatter!(p.U, u)
    if p.hasdirichlet
        fill_ghosts!(p.U, p.schedule; boundary=dirichlet(p.case, t))
    else
        fill_ghosts!(p.U, p.schedule)
    end
    map_blocks!(gh_rhs_kernel!, p.U, gh_rhs_kernel_args(p, du, t)...)
    # The zone points of an excised hole (step X2b): the closures, on the
    # working array the main kernel just read — after it, since the main
    # kernel writes nothing there.
    p.excision === nothing || gh_zone!(du, p, t)
    return nothing
end

# The kernel's arguments, in one place (added 2026-10-05): `gh_rhs!` and the
# benchmarks that launch the kernel by itself (`bench/`) build them alike.
gh_rhs_kernel_args(p::GHProblem, du, t) =
    (statearray(du, p.U), p.U.work, gauge_work(p.Hsrc), p.origins, p.spacings,
     p.case.background, p.case.γ0, p.case.γ2, p.case.ε_KO, p.interior,
     eltype(p.U.work)(t), target_work(p.target), p.t_target, p.target_rate, p.trail,
     _exact_fit(p), excision_classes(p.excision), excision_blend(p.excision),
     p.valG, p.valq, p.valH, p.valdiss, p.valint, p.valsimd)

"""
    gh_step_limiter!(u, integrator, p::GHProblem, t)

RK4's `step_limiter!` hook: for the `:pasted` variant, the analytic
solution written over the ball `r < r_1` at the end of every step; for
every other variant, nothing at all.

`CODE.md`, "Three variants, one switch": the hard paste is `(INTERIOR)` in
the limit `ρ → ∞` on a step profile, and RK4's limiter hook is where it can
be implemented *exactly*. It is one of the two limiters that write the
state outside the integrator — the other is the range projection of step
8b, [`gh_stage_limiter!`](@ref) — and `CLAUDE.md`'s "The RHS never mutates
`u`" names exactly these three writers. The dispatch below keeps this one
to the variant that needs it: the `:none`, `:damped` and `:frozen` methods
are empty and compile away, so the same limiter serves every run
([`gh_limiter!`](@ref), which projects and then calls this on every stage
value and every step's result; amended 2026-09-26).

`u` arrives in state layout and `statearray(u, p.U)` is the block view, as
`PLAN.md`'s "Sharp edges" says.
"""
gh_step_limiter!(u, integrator, p::GHProblem{T,G,q,HASH,DISS,:none},
                 t) where {T,G,q,HASH,DISS} = nothing
gh_step_limiter!(u, integrator, p::GHProblem{T,G,q,HASH,DISS,:damped},
                 t) where {T,G,q,HASH,DISS} = nothing
gh_step_limiter!(u, integrator, p::GHProblem{T,G,q,HASH,DISS,:frozen},
                 t) where {T,G,q,HASH,DISS} = nothing
# The fitted layer writes nothing (step 8e): its target is a term of `du`.
gh_step_limiter!(u, integrator, p::GHProblem{T,G,q,HASH,DISS,:fitted},
                 t) where {T,G,q,HASH,DISS} = nothing
# Nor does an excised hole (step X2b): its excised points have `du = 0`, and
# there is no fourth writer of the state.
gh_step_limiter!(u, integrator, p::GHProblem{T,G,q,HASH,DISS,:excised},
                 t) where {T,G,q,HASH,DISS} = nothing

function gh_step_limiter!(u, integrator,
                          p::GHProblem{T,G,q,HASH,DISS,:pasted},
                          t) where {T,G,q,HASH,DISS}
    map_blocks!(gh_paste_kernel!, p.U, statearray(u, p.U), p.origins,
                p.spacings, p.case.background, p.interior, T(t), p.valG)
    return nothing
end

# A limiter is also wanted on the *initial* state of a `:pasted` run and
# after every regrid, since neither went through a step. Same kernel, no
# integrator.
"""
    paste_interior!(p::GHProblem, u, t)

Apply the `:pasted` variant's overwrite to `u` without an integrator —
what the driver does to the initial data and to a freshly regridded state,
neither of which went through a step. A no-op for the other variants.
"""
paste_interior!(p::GHProblem, u, t) = gh_step_limiter!(u, nothing, p, t)

"""
    max_speed(p::GHProblem) -> T

The largest characteristic speed over every owned point, from the state
currently in the working array.

One launch of [`gh_speed_kernel!`](@ref) into `diag`, then TreeAMR's
`mesh_mapreduce(max)`, whose per-block maxima are combined on the host, so
the answer does not depend on the thread count (`CODE.md`, "Analysis
quantities"; a host fold of this package's own until 2026-10-01).

It reads the working array, not a state vector: call it after a
`scatter!`, which is what [`gh_dt`](@ref) does.
"""
function max_speed(p::GHProblem{T}; t=zero(T),
                   mask=interior_mask(p.interior, t)) where {T}
    map_blocks!(gh_speed_kernel!, p.U, p.diag.work, p.U.work, p.origins,
                p.spacings, mask, p.valG)
    # `zero(T)` as the identity rather than `typemin`: a characteristic
    # speed is `α√(tr γ^{ij}) + |β| ≥ 0`, and `typemin` is not defined for
    # every type this package runs in. A `NaN` still propagates, which is
    # what makes a blown-up state visible here rather than as a `dt` of
    # zero two lines later.
    return mesh_mapreduce(identity, max, zero(T), p.diag; vars=DIAG_SPEED)
end

"""
    gh_dt(p::GHProblem, u; cfl) -> T

The time step, `cfl · minimum_spacing(forest) / λ_max`, with `λ_max` the
largest characteristic speed of the state `u` (`CODE.md`, "The time
step"). `cfl = 1/4` is GHSO2's default and this package's.

There is no subcycling anywhere in TreeAMR, so this one number advances
the whole hierarchy and the finest spacing present is what sets it.

It scatters `u` into the working array on the way — the speed kernel reads
the working array — which is scratch and about to be overwritten by the
first evaluation anyway. `u` itself is untouched.
"""
function gh_dt(p::GHProblem{T}, u; cfl, t=zero(T)) where {T}
    scatter!(p.U, u)
    λ = max_speed(p; t=t)
    isfinite(λ) && λ > 0 || throw(ArgumentError(
        "the maximum characteristic speed α√(tr γ^{ij}) + |β| came out as " *
        "$λ, so there is no CFL-limited time step: the state is either not " *
        "a metric any more (a blown-up run) or identically zero, which no " *
        "initial data of this package produces — h = 0 is Minkowski, whose " *
        "speed is √3."))
    return T(cfl) * minimum_spacing(T, p.U.forest) / λ
end
