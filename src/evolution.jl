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
    return ntuple(d -> origin[d] + (I[d] - off) * h, Val(3))
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

# `∑_k w[k] u[base + (k − 1 − r)·stride]`: the contraction every
# one-dimensional operator here is, whatever the weights mean.
# `ntuple(…, Val(n))` unrolls it, so the offsets and the weights are
# compile-time constants and the loop leaves no index arithmetic behind.
@inline function axis_stencil(w::SVector{n,T}, work, base::Int,
                              stride::Int) where {n,T}
    r = (n - 1) ÷ 2
    return _fold(ntuple(Val(n)) do k
        @inbounds w[k] * work[base + (k - 1 - r) * stride]
    end)
end

# The mixed derivative: the tensor product of two first-derivative vectors,
# outer sum along the first axis, inner along the second. It reads the edge
# ghosts TreeAMR fills unconditionally, which is why `∂_i∂_j` needs no
# wider halo than `∂_i∂_i`.
@inline function mixed_stencil(w::SVector{n,T}, work, base::Int, s1::Int,
                               s2::Int) where {n,T}
    r = (n - 1) ÷ 2
    return _fold(ntuple(Val(n)) do a
        w[a] * _fold(ntuple(Val(n)) do e
            @inbounds w[e] * work[base + (a - 1 - r) * s1 + (e - 1 - r) * s2]
        end)
    end)
end

"""
    gh_rhs_at_point(T, work, Hwork, inner, b, var, st, sv, inv_h, γ0, γ2, εh,
                    ::Val{q}, ::Val{HASH}, ::Val{DISS}) -> (∂ₜh, ∂ₜΠ)

`F(u)` at one owned point: the fused right-hand side of `(EXPANDED)`, in
`CODE.md`'s streaming order, with the Kreiss–Oliger term and the source
already in it.

It is a **plain function called from the kernel** rather than the kernel's
own body (restructured in step 5). The reason is `CODE.md`'s rule that `F`
is never evaluated where `w = 0`: the frozen core holds finite but stale
data on which `F` may be `NaN`, and `0 · NaN = NaN`, so the kernel has to
branch *around* this whole computation — and KernelAbstractions refuses a
`return` statement anywhere in a kernel body, closures included, so the
branch cannot be an early exit. Inlined, the generated code and the
streaming order are what they were; `test/evolution_tests.jl`'s comparison
against [`gh_node_rhs_expanded`](@ref) is unchanged and still passes at the
same tolerance.

`var` is the point's linear index in `work`, `st` the per-axis strides and
`sv` the per-variable one ([`work_strides`](@ref)); `γ0` is this point's
constraint-damping rate, which is now a **profile** evaluated by the
caller (`CODE.md`, "Gauge and constraint damping").

What it computes:

    ∂ₜh_ab = β^i ∂_i h_ab + (α/√γ) Π_ab                    + Q_d h_ab
    ∂ₜΠ_ab = β^i ∂_i Π_ab + (∂_iβ^i) Π_ab
           + α√γ γ^{ij} ∂_i∂_j h_ab + ∂_i(α√γ γ^{ij}) ∂_j h_ab
           − α√γ (S0_ab + Z_ab)                            + Q_d Π_ab

The source `S0 + Z` is [`gh_node_source`](@ref) and the coefficient
derivatives are [`metric_derivatives`](@ref) — the same functions
[`gh_node_rhs_expanded`](@ref) calls, which is what makes that function the
reference this kernel is checked against on analytic data. The two are not
bit-identical and are not expected to be: one body reached from two call
sites is contracted into fused multiply-adds differently (`CODE.md`,
"Measured results").

**The `∂_t g` the source is given is the accumulated `∂ₜh`, dissipation
included** — the two accumulators per component that the streaming order
budgets, and the honest answer besides: it is the time derivative of the
numerical solution, which is what the reduced source's `−Γ^ν ∂_ν g_ab`
means. The difference is `O(h^{q+1})`, the dissipation's own order
**(recorded in step 3**, where `CODE.md` had said only "from `h`, `∂_i h`,
`∂_t h` and the coefficients"**)**.
"""
@inline function gh_rhs_at_point(::Type{T}, work, Hwork, inner, b::Int,
                                 var::Int, st, sv::Int, inv_h, γ0, γ2, εh,
                                 ::Val{q}, ::Val{HASH},
                                 ::Val{DISS}) where {T,q,HASH,DISS}
    inv_h² = inv_h * inv_h
    w1 = derivative_weights(T, Val(q), Val(1))
    w2 = derivative_weights(T, Val(q), Val(2))
    wD = dissipation_weights(T, dissipation_rank(Val(q)))

    # (1) the state at the point, the 30 first derivatives of `h`, and the
    #     coefficients built from them once.
    hv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (v - 1) * sv]),
                              Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (NC + v - 1) * sv]),
                              Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, work, var + (v - 1) * sv, st[d])
        end)
    end

    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    A = (α * sqrtγ) * γu                          # A^{jk} = α√γ γ^{jk}
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]                     # ∂_i β^i
    divA = SVector{3,T}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)

    # (2) one component at a time: nine stencils formed, contracted and
    #     dropped, leaving two accumulators.
    acc = ntuple(Val(NC)) do v
        ∂h1 = ∂h[1][v]
        ∂h2 = ∂h[2][v]
        ∂h3 = ∂h[3][v]
        Π_v = Πv[v]
        bh = var + (v - 1) * sv                   # this component of `h`
        bΠ = bh + NC * sv                         # and of `Π`

        ∂ₜh_v = β[1] * ∂h1 + β[2] * ∂h2 + β[3] * ∂h3 + a_div * Π_v

        ∂Π1 = inv_h * axis_stencil(w1, work, bΠ, st[1])
        ∂Π2 = inv_h * axis_stencil(w1, work, bΠ, st[2])
        ∂Π3 = inv_h * axis_stencil(w1, work, bΠ, st[3])
        ∂ₜΠ_v = β[1] * ∂Π1 + β[2] * ∂Π2 + β[3] * ∂Π3 + divβ * Π_v +
                divA[1] * ∂h1 + divA[2] * ∂h2 + divA[3] * ∂h3

        ∂ₜΠ_v += A[1, 1] * (inv_h² * axis_stencil(w2, work, bh, st[1])) +
                 A[2, 2] * (inv_h² * axis_stencil(w2, work, bh, st[2])) +
                 A[3, 3] * (inv_h² * axis_stencil(w2, work, bh, st[3]))
        ∂xy = inv_h² * mixed_stencil(w1, work, bh, st[1], st[2])
        ∂xz = inv_h² * mixed_stencil(w1, work, bh, st[1], st[3])
        ∂yz = inv_h² * mixed_stencil(w1, work, bh, st[2], st[3])
        ∂ₜΠ_v += 2 * (A[1, 2] * ∂xy + A[1, 3] * ∂xz + A[2, 3] * ∂yz)

        if DISS
            ∂ₜh_v += εh * (axis_stencil(wD, work, bh, st[1]) +
                           axis_stencil(wD, work, bh, st[2]) +
                           axis_stencil(wD, work, bh, st[3]))
            ∂ₜΠ_v += εh * (axis_stencil(wD, work, bΠ, st[1]) +
                           axis_stencil(wD, work, bΠ, st[2]) +
                           axis_stencil(wD, work, bΠ, st[3]))
        end
        (∂ₜh_v, ∂ₜΠ_v)
    end
    ∂ₜh = SVector{NC,T}(ntuple(v -> acc[v][1], Val(NC)))
    ∂ₜΠ = SVector{NC,T}(ntuple(v -> acc[v][2], Val(NC)))

    # (3) the source, from the state, its gradients and the coefficients —
    #     the gauge source is read at the owned point, `Hsrc` having no
    #     ghosts to read.
    Hl, dHl = gauge_at(T, Hwork, inner, b, Val(HASH))
    msrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)
    return ∂ₜh, ∂ₜΠ + msrc
end

"""
    gh_rhs_kernel!(du, work, Hwork, origins, spacings, bg, damping, γ2, ε_KO,
                   interior, t, ::Val{G}, ::Val{q}, ::Val{HASH}, ::Val{DISS},
                   ::Val{INT})

The right-hand side at one owned point: `F(u)` from
[`gh_rhs_at_point`](@ref), modified inside the hole by `CODE.md`'s
`(INTERIOR)`,

    ∂_t u = w(r) · F(u)  −  ρ(r) · (u − u_exact(x, t)) .

`du` is in **state layout** (no ghosts, so the global index is used as it
comes); `work` is the ghosted working array (so the same index plus `G`).
`Hwork` is the gauge source's working array or `nothing`.

The **five** `Val`s are built once per chunk in [`GHProblem`](@ref) and
resolved when the kernel compiles: the ghost width, the difference order,
whether there is a gauge source, whether there is dissipation, and — added
in step 5 — which of `CODE.md`'s interior variants is running, `:none`
meaning there is no hole. Building them per evaluation would recompile or
dispatch dynamically at every RK stage (`CLAUDE.md`).

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

**The three branches, in the order they must be in.** The core predicate
is asked *before* any stencil is touched, because the frozen core holds
finite but stale data on which `F` may be `NaN` and `0 · NaN = NaN`
(`CLAUDE.md`). Outside `r_1` the answer is `F` itself and not `1·F − 0·(…)`,
which also saves the analytic solution's dual pass at every point of the
evolved region — `u_exact` is evaluated in the layer and nowhere else.
"""
@kernel function gh_rhs_kernel!(du, @Const(work), Hwork, @Const(origins),
                                @Const(spacings), bg, damping, γ2, ε_KO,
                                interior, t, ::Val{G}, ::Val{q}, ::Val{HASH},
                                ::Val{DISS}, ::Val{INT}) where {G,q,HASH,DISS,
                                                                INT}
    I = @index(Global, NTuple)                    # (i1, i2, i3, block)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))             # state-layout index
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

    if INT === :none
        ∂ₜh, ∂ₜΠ = gh_rhs_at_point(T, work, Hwork, inner, b, var, st, sv,
                                   inv_h, γ0, γ2, εh, Val(q), Val(HASH),
                                   Val(DISS))
        ntuple(Val(NC)) do v
            du[inner..., v, b] = ∂ₜh[v]
            du[inner..., NC + v, b] = ∂ₜΠ[v]
            nothing
        end
    else
        # The interior's view of the point (step 8d): the radius for step
        # 5's sphere, and for the tracked geometry the radius with the two
        # surfaces' radii along the ray — `interior_point`, which evaluates
        # the shape's series only between its bounding spheres. The three
        # branches below are the same for both.
        g = interior_point(interior, t, x)
        if is_frozen(interior, g)
            # `du = 0`, and `F` is not evaluated: this is the branch
            # `CLAUDE.md` says must come before the stencils.
            ntuple(Val(2 * NC)) do v
                du[inner..., v, b] = zero(T)
                nothing
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
without thinking about it. `block_mapreduce(max)` over its output is
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
"""
struct GHProblem{T,G,q,HASH,DISS,INT,F,S,H,D,O,V,C,I,A}
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
    hasdirichlet::Bool
    valG::Val{G}
    valq::Val{q}
    valH::Val{HASH}
    valdiss::Val{DISS}
    valint::Val{INT}
end

function GHProblem(U::FieldSet{T,3}, schedule, case::GHCase{T}; q::Integer,
                   t=zero(T), interior=case.interior, margin_check=true,
                   accounting=nothing) where {T}
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

    HASH = !isharmonic(case.background)
    Hsrc = if HASH
        fs = FieldSet{T}(U.forest, 2NC; G=0, centering=U.centering,
                         backend=backend)
        sample_gauge_source!(fs, case.background, t; interior=interior)
        fs
    else
        nothing
    end
    diag = FieldSet{T}(U.forest, NDIAG; G=0, centering=U.centering,
                       backend=backend)

    origins = to_backend(backend, block_origins(U.forest, T))
    spacings = to_backend(backend, block_spacings(U.forest, T))
    DISS = has_dissipation(case.ε_KO)
    hasdirichlet = !all(case.periodic)

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

    return GHProblem{T,U.G,Int(q),HASH,DISS,INT,typeof(U),typeof(schedule),
                     typeof(Hsrc),typeof(diag),typeof(origins),
                     typeof(spacings),typeof(case),typeof(interior),
                     typeof(accounting)}(
        U, schedule, Hsrc, diag, origins, spacings, case, interior,
        accounting, hasdirichlet, Val(U.G), Val(Int(q)), Val(HASH), Val(DISS),
        Val(INT))
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
function with_interior(p::GHProblem{T,G,q,HASH,DISS}, interior) where {T,G,q,
                                                                       HASH,
                                                                       DISS}
    INT = interior_variant(interior)
    return GHProblem{T,G,q,HASH,DISS,INT,typeof(p.U),typeof(p.schedule),
                     typeof(p.Hsrc),typeof(p.diag),typeof(p.origins),
                     typeof(p.spacings),typeof(p.case),typeof(interior),
                     typeof(p.accounting)}(
        p.U, p.schedule, p.Hsrc, p.diag, p.origins, p.spacings, p.case,
        interior, p.accounting, p.hasdirichlet, p.valG, p.valq, p.valH,
        p.valdiss, Val(INT))
end

# The gauge source's working array, or `nothing` where there is none. The
# kernel never asks whether it has one — its `Val` already said.
gauge_work(::Nothing) = nothing
gauge_work(fs::FieldSet) = fs.work

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
    map_blocks!(gh_rhs_kernel!, p.U, statearray(du, p.U), p.U.work,
                gauge_work(p.Hsrc), p.origins, p.spacings, p.case.background,
                p.case.γ0, p.case.γ2, p.case.ε_KO, p.interior, eltype(p.U.work)(t),
                p.valG, p.valq, p.valH, p.valdiss, p.valint)
    return nothing
end

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
are empty and compile away, so the same `solve(…; step_limiter =
gh_step_limiter!)` serves every run.

`u` arrives in state layout and `statearray(u, p.U)` is the block view, as
`PLAN.md`'s "Sharp edges" says.
"""
gh_step_limiter!(u, integrator, p::GHProblem{T,G,q,HASH,DISS,:none},
                 t) where {T,G,q,HASH,DISS} = nothing
gh_step_limiter!(u, integrator, p::GHProblem{T,G,q,HASH,DISS,:damped},
                 t) where {T,G,q,HASH,DISS} = nothing
gh_step_limiter!(u, integrator, p::GHProblem{T,G,q,HASH,DISS,:frozen},
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

One launch of [`gh_speed_kernel!`](@ref) into `diag`, then
`block_mapreduce(max)` and a fold over the per-block values **in block
order**, so the answer does not depend on the thread count
(`CODE.md`, "Analysis quantities").

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
    return maximum(block_mapreduce(identity, max, zero(T), p.diag;
                                   vars=DIAG_SPEED))
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
