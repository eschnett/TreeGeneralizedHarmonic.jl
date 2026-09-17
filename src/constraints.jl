# The two constraint monitors, and the masked norms a run is judged by.
#
# `CODE.md`, "Analysis quantities": a run is judged by what it records,
# and the first two rows of that table are the constraints.
#
#   * the **gauge constraint** `C_a = Γ_a + H_a`, from the state and its
#     first derivatives — one stencil pass, a small fraction of a
#     right-hand side;
#   * the **ADM Hamiltonian and momentum constraints** `ℋ`, `ℳ_i`, from
#     the covariant Einstein tensor projected on the foliation normal,
#     with *every* second derivative of `g_ab` present: the spatial ones
#     from the same compact and tensor-product stencils the right-hand
#     side uses, and the `∂_t∂_i` and `∂_t∂_t` blocks **reconstructed from
#     the reduced evolution equations**, so that the monitor measures the
#     discrete dynamics and not a second, independent scheme (GHSO2's
#     construction, `notes/methods-ghso2.md`, "Diagnostics").
#
# The two are not independent: modulo the evolution equations the ADM
# constraints are combinations of `C_a` and its derivatives
# (`notes/methods-ghso2.md`, "Constraint damping"), which is why one
# damping term services both sectors. They fail differently all the same —
# `C_a` sees first derivatives, `ℋ` and `ℳ_i` see second ones — and both
# are in the record.
#
# Three things about this file are decisions rather than transcription,
# and each is stated where it is made:
#
#   * **which `∂_t g`**. The evolution relation `∂_t g = β^i ∂_i g +
#     (α/√γ)Π` without the Kreiss–Oliger term, and `∂_tΠ` from
#     `gh_node_rhs_expanded` — the reduced equation with its source and
#     its constraint damping, also without the dissipation. The
#     dissipation is `O(h^{q+1})`, one order below what these monitors
#     measure, and carrying it would mean differencing it as well.
#   * **the mask**. Every kernel here takes a mask and writes **zero**
#     where it says the point is not evolved, and the norms below divide
#     by the evolved volume rather than by the domain's. Step 4 has only
#     the trivial mask [`AllPoints`](@ref); step 5's interior is what it
#     exists for (`CLAUDE.md`, "The interior is masked in every norm").
#   * **the cost**. The ADM kernel holds a hundred second derivatives and
#     the full four-dimensional Ricci tensor at once, which is the
#     opposite of the right-hand side's streaming order and deliberately
#     so: it runs once per chunk, not once per stage. On a device it will
#     spill, and that is the right trade for a diagnostic; `CODE.md`'s
#     register budget is a statement about `gh_rhs_kernel!`.

"""
    AllPoints()

The trivial mask: every point is an evolved point.

Every norm and monitor in this package takes a mask, because from step 5
on there is a region — the damping layer and the frozen core inside the
horizon — that is *not* a numerical solution and must not be reported as
one (`CODE.md`, "The interior: a pointwise damping layer"). Until that
region exists the mask is this, and it is a type rather than a `nothing`
so that [`is_evolved`](@ref) is a plain dispatch and the kernels have no
branch on a missing argument.

It is `isbits` and empty, so it costs a kernel argument and nothing else.
"""
struct AllPoints end

"""
    is_evolved(mask, x) -> Bool

Whether the point at position `x` is evolved by the unmodified equations
— the predicate every masked norm, monitor and (from step 6) refinement
indicator asks.

`AllPoints` says yes everywhere. Step 5's interior says `r ≥ r_1`, with
`r` the distance to the hole's analytic center: a function of position
and time and of nothing about blocks, levels or ghost widths
(`CODE.md`).
"""
@inline is_evolved(::AllPoints, x) = true

# `dg[a, b, c] = ∂_c g_ab`, SpacetimeMetrics' derivative index order —
# the derivative axis *last*. `_dg4` builds GHSO2's, axis first; this is
# the transpose, and it exists because `gauge_constraint_at_node` is the
# one ported function that speaks SpacetimeMetrics' order, being fed
# straight from `dmetric` in its original home (`pointwise.jl`'s header).
# Written once, next to the conversion it mirrors.
@inline function _dg4_last(∂ₜg::SVector{NC,T},
                           ∂h::NTuple{3,SVector{NC,T}}) where {T}
    dt4 = _sym4(∂ₜg)
    dx4 = _sym4(∂h[1]); dy4 = _sym4(∂h[2]); dz4 = _sym4(∂h[3])
    return SArray{Tuple{4,4,4},T}(
        (c == 1 ? dt4[a,b] : c == 2 ? dx4[a,b] : c == 3 ? dy4[a,b] : dz4[a,b])
        for a in 1:4, b in 1:4, c in 1:4)
end

# `ddg[μ, ν, a, b] = ∂_μ∂_ν g_ab` from the ten packed second derivatives,
# which are indexed by the symmetric **pair** `(μν)` in exactly the order
# `_pack10` indexes `(ab)`: `(tt, tx, ty, tz, xx, xy, xz, yy, yz, zz)`.
# One packing convention, used on both index pairs.
@inline function _ddg4(dd::NTuple{NC,SVector{NC,T}}) where {T}
    m = ntuple(n -> _sym4(dd[n]), Val(NC))
    return SArray{Tuple{4,4,4,4},T}(
        m[_pairindex(μ, ν)][a, b]
        for μ in 1:4, ν in 1:4, a in 1:4, b in 1:4)
end

"""
    adm_constraints_at_node(g4, gu4, α, β, dg, ddg) -> (ℋ, ℳ::SVector{3})

The ADM Hamiltonian and momentum constraints at one point, as the
projections of the four-dimensional Einstein tensor on the foliation
normal `n^μ = (1, −β^i)/α`:

    ℋ   = 2 G_μν n^μ n^ν            = R³ + K² − K_ij K^ij
    ℳ_i = −G_μν n^μ γ^ν_i           = D_j(K^j_i − δ^j_i K)

Both vanish in vacuum. `dg[a, b, c] = ∂_a g_bc` and
`ddg[a, b, c, d] = ∂_a∂_b g_cd` are **GHSO2's** derivative index order,
derivative axes first, and their `t` slices are the ones the caller
reconstructed from the evolution equations — this function does not know
where they came from, which is what lets `test/constraints_tests.jl`
check it against a background's analytic derivatives with no mesh in
sight.

The Ricci tensor is the textbook one,

    R_ab = ∂_c Γ^c_ab − ∂_b Γ^c_ca + Γ^c_cd Γ^d_ab − Γ^c_bd Γ^d_ca ,

written out rather than reduced through the generalized-harmonic
identity `R_ab = −½ g^cd ∂_c∂_d g_ab + ∇_(a Γ_b) + …`. The reduced form
would be cheaper and is exactly what the evolution equations already
encode, so a monitor built on it would be checking the right-hand side
against itself; this one is an independent assembly, and the price is the
four-dimensional contraction. `CODE.md` prices the whole kernel at about
one right-hand-side evaluation, GHSO2 measured two, and the cadence is
per chunk.

`α`, `β` and `g4` come from [`metric_quantities`](@ref); nothing here is
recomputed from `h`.
"""
@inline function adm_constraints_at_node(g4::SMatrix{4,4,T},
                                         gu4::SMatrix{4,4,T}, α::T,
                                         β::SVector{3,T},
                                         dg::SArray{Tuple{4,4,4},T},
                                         ddg::SArray{Tuple{4,4,4,4},T}) where {T}
    # ∂_e g^{ab} = −g^{ac}(∂_e g_cd) g^{db}, the exact identity.
    dgu = ntuple(Val(4)) do e
        de = SMatrix{4,4,T}(dg[e, a, b] for a in 1:4, b in 1:4)
        -(gu4 * de) * gu4
    end

    # Γ_{a,bc} = ½(∂_b g_ac + ∂_c g_ab − ∂_a g_bc) and Γ^a_{bc}, exactly as
    # `gh_node_source` writes them.
    Γlll = SArray{Tuple{4,4,4},T}(
        (dg[b,a,c] + dg[c,a,b] - dg[a,b,c]) / 2 for a in 1:4, b in 1:4, c in 1:4)
    Γ4 = SArray{Tuple{4,4,4},T}(
        sum(gu4[a,x] * Γlll[x,b,c] for x in 1:4) for a in 1:4, b in 1:4, c in 1:4)
    # Γ^c_{cd}, the trace that is ∂_d ln√|g|.
    Γtr = SVector{4,T}(sum(Γ4[c,c,d] for c in 1:4) for d in 1:4)

    # ∂_e Γ_{a,bc} = ½(∂_e∂_b g_ac + ∂_e∂_c g_ab − ∂_e∂_a g_bc), contracted
    # immediately into the only two combinations the Ricci tensor needs:
    #   A_ab = ∂_c Γ^c_{ab},   B_ab = ∂_b Γ^c_{ca}.
    A = SMatrix{4,4,T}(
        sum(dgu[c][c,d] * Γlll[d,a,b] for c in 1:4, d in 1:4) +
        sum(gu4[c,d] * (ddg[c,a,d,b] + ddg[c,b,d,a] - ddg[c,d,a,b]) / 2
            for c in 1:4, d in 1:4)
        for a in 1:4, b in 1:4)
    B = SMatrix{4,4,T}(
        sum(dgu[b][c,d] * Γlll[d,c,a] for c in 1:4, d in 1:4) +
        sum(gu4[c,d] * (ddg[b,c,d,a] + ddg[b,a,d,c] - ddg[b,d,c,a]) / 2
            for c in 1:4, d in 1:4)
        for a in 1:4, b in 1:4)

    R = SMatrix{4,4,T}(
        A[a,b] - B[a,b] + sum(Γtr[d] * Γ4[d,a,b] for d in 1:4) -
        sum(Γ4[c,b,d] * Γ4[d,c,a] for c in 1:4, d in 1:4)
        for a in 1:4, b in 1:4)
    Rs = sum(gu4[a,b] * R[a,b] for a in 1:4, b in 1:4)

    # n^μ = (1, −β^i)/α, the future-pointing unit normal; n·n = −1, so
    # 2 G_μν n^μ n^ν = 2 R_μν n^μ n^ν + R.
    n = SVector{4,T}(one(T), -β[1], -β[2], -β[3]) / α
    Rnn = sum(R[a,b] * n[a] * n[b] for a in 1:4, b in 1:4)
    ℋ = 2 * Rnn + Rs
    # n_i = 0 for a spatial index, so γ^ν_i = δ^ν_i there; the `g` term is
    # written out anyway rather than dropped, because it is the definition
    # and costs four multiplies.
    ℳ = SVector{3,T}(
        -sum(n[a] * (R[a, i+1] - g4[a, i+1] * Rs / 2) for a in 1:4)
        for i in 1:3)
    return ℋ, ℳ
end

"""
    gh_constraint_kernel!(diag, work, Hwork, origins, spacings, mask,
                          ::Val{G}, ::Val{q}, ::Val{HASH})

The gauge constraint `C_a = Γ_a + H_a` at one owned point, written into
the `diag` field set's four `DIAG_CGH` slots, together with the mask
indicator in `DIAG_MASK`.

It needs the state and its **first** derivatives only, so it is one
stencil pass and a pointwise contraction — the cheap monitor, and the one
that can run at every chunk without thinking about it. `∂_t g` comes from
the first evolution equation `β^i ∂_i g + (α/√γ)Π`, which is what makes
this a statement about the evolved state rather than about a slice of it.

`C_a` is stored **lowered**, as `CODE.md`'s table names it and as the
Gundlach–Pretorius damping term uses it;
[`gauge_constraint_at_node`](@ref) returns the contravariant `C^μ` and is
fed `dg[a, b, c] = ∂_c g_ab`, SpacetimeMetrics' index order, because it is
the one ported function written against `dmetric`.
"""
@kernel function gh_constraint_kernel!(diag, @Const(work), Hwork,
                                       @Const(origins), @Const(spacings), mask,
                                       ::Val{G}, ::Val{q},
                                       ::Val{HASH}) where {G,q,HASH}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    T = eltype(diag)

    inv_h = inv(spacings[b])
    w1 = derivative_weights(T, Val(q), Val(1))
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb +
          (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]

    hv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (v - 1) * sv]),
                              Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (NC + v - 1) * sv]),
                              Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, work, var + (v - 1) * sv, st[d])
        end)
    end

    g4, gu4, α, β, _, sqrtγ = metric_quantities(_sym4(hv))
    ∂ₜh = β[1]*∂h[1] + β[2]*∂h[2] + β[3]*∂h[3] + (α / sqrtγ) * Πv
    Hl, _ = gauge_at(T, Hwork, inner, b, Val(HASH))
    Cl = g4 * gauge_constraint_at_node(g4, _dg4_last(∂ₜh, ∂h), gu4 * Hl)

    # **A branch, not a multiplication by zero (fixed in step 5).** Step 4
    # wrote `keep * Cl[a]`, which is the same number for every mask that
    # exists when nothing is masked. It is not the same number once the
    # interior is: the frozen core holds stale data on which `C_a` may be
    # `NaN`, and `0 · NaN = NaN` — `CLAUDE.md`'s trap, met here rather than
    # in the right-hand side. Every masked slot in this file is written
    # through a branch for that reason.
    keep = is_evolved(mask, point_position(origins, spacings, b, I))
    ntuple(Val(4)) do a
        diag[inner..., DIAG_CGH + a - 1, b] = keep ? Cl[a] : zero(T)
        nothing
    end
    diag[inner..., DIAG_MASK, b] = keep ? one(T) : zero(T)
end

"""
    gh_error_kernel!(diag, work, origins, spacings, bg, interior, mask, t,
                     r_shell_lo, r_shell_hi, ::Val{G})

The three error rows of `CODE.md`'s analysis table, in one launch over the
owned points:

  * `DIAG_ERR` — `‖u − u_exact‖` at the point, zero where the mask says the
    point is not evolved, with the mask indicator beside it in `DIAG_MASK`
    so that [`masked_norms`](@ref) divides by the *evolved* volume;
  * `DIAG_RES` — the same magnitude, nonzero only in the damping layer
    `r_0 ≤ r < r_1`: `CODE.md`'s "interior residual, the layer's own
    health", read in L∞;
  * `DIAG_DRIFT` — `|h_tt − h_tt,exact|` in the shell
    `r_shell_lo ≤ r ≤ r_shell_hi`, which the driver sets to a band around
    the horizon. Its L∞ against time is the *gauge drift* GHSO2 measured at
    `≈ 0.14/M` on the excised hole (`notes/methods-ghso2.md`), and `h_tt`
    is the component that carries it: the lapse is read off `g_tt`.

It reads the point and nothing else — no stencil, no ghosts — so it is as
cheap as the speed kernel and can run at every chunk. The analytic solution
is one forward-mode dual pass through the background per point, which is
the expensive part and the reason this is a per-chunk quantity rather than
a per-step one.

`interior === nothing` is no hole: the residual and the drift slots are
written zero, which is what a case without an interior should report.
"""
@kernel function gh_error_kernel!(diag, @Const(work), @Const(origins),
                                  @Const(spacings), bg, interior, mask, t,
                                  r_shell_lo, r_shell_hi, ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    c = ntuple(d -> I[d] + G[d], Val(3))
    T = eltype(diag)

    x = point_position(origins, spacings, b, I)
    vals = case_state_tuple(bg, interior, t, x)
    # A fold rather than an accumulator: a kernel body may not close over a
    # mutated local, and the summation order of a norm is part of it.
    e² = _fold(ntuple(Val(2 * NC)) do v
        d = work[c..., v, b] - vals[v]
        d * d
    end)
    e = sqrt(e²)

    keep = is_evolved(mask, x)
    diag[inner..., DIAG_MASK, b] = keep ? one(T) : zero(T)
    diag[inner..., DIAG_ERR, b] = keep ? e : zero(T)
    r = interior_radius(interior, t, x)
    diag[inner..., DIAG_RES, b] = in_layer(interior, r) ? e : zero(T)
    inshell = (r_shell_lo ≤ r) & (r ≤ r_shell_hi)
    diag[inner..., DIAG_DRIFT, b] =
        inshell ? abs(work[c..., 1, b] - vals[1]) : zero(T)
end

"""
    adm_constraint_kernel!(diag, work, Hwork, origins, spacings, γ0, γ2, mask,
                           ::Val{G}, ::Val{q}, ::Val{HASH})

The ADM Hamiltonian and momentum constraints at one owned point, written
into `DIAG_HAM` and the three `DIAG_MOM` slots, with the mask indicator in
`DIAG_MASK`.

All hundred second derivatives of `g_ab` are formed here, which is what
makes this the expensive monitor:

  * `∂_i∂_j g` from the right-hand side's own stencils — compact on the
    diagonal, the tensor product of two first derivatives off it;
  * `∂_i∂_t g` by differentiating the first evolution equation,
    `∂_i(β^j ∂_j g + φ Π)` with `φ = α/√γ`, whose coefficient derivatives
    are [`metric_derivatives_along`](@ref)'s closed forms;
  * `∂_t∂_t g` by differentiating it again in time, with `∂_tΠ` from
    [`gh_node_rhs_expanded`](@ref) — **the reduced equation**, source and
    constraint damping included.

That last substitution is `CODE.md`'s "`∂_tt g` from the reduced equation
(GHSO2's construction, consistent with the discrete dynamics)": the
monitor is a statement about the scheme that is running, not about a
different one. What it leaves out is the Kreiss–Oliger term, in both time
derivatives (recorded in step 4): it is `O(h^{q+1})`, one order below the
truncation error these monitors converge at, and carrying it would mean
differencing the dissipation operator as well.
"""
@kernel function adm_constraint_kernel!(diag, @Const(work), Hwork,
                                        @Const(origins), @Const(spacings),
                                        damping, γ2, mask, t, ::Val{G},
                                        ::Val{q}, ::Val{HASH}) where {G,q,HASH}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    T = eltype(diag)
    x = point_position(origins, spacings, b, I)
    γ0 = damping_rate(damping, t, x)

    inv_h = inv(spacings[b])
    inv_h² = inv_h * inv_h
    w1 = derivative_weights(T, Val(q), Val(1))
    w2 = derivative_weights(T, Val(q), Val(2))
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb +
          (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]

    hv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (v - 1) * sv]),
                              Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (NC + v - 1) * sv]),
                              Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, work, var + (v - 1) * sv, st[d])
        end)
    end
    ∂Π = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, work, var + (NC + v - 1) * sv, st[d])
        end)
    end
    # The six spatial second derivatives, packed `(xx, xy, xz, yy, yz, zz)`.
    ∂∂h = ntuple(Val(6)) do n
        d1 = n == 1 ? 1 : n == 2 ? 1 : n == 3 ? 1 : n == 4 ? 2 : n == 5 ? 2 : 3
        d2 = n == 1 ? 1 : n == 2 ? 2 : n == 3 ? 3 : n == 4 ? 2 : n == 5 ? 3 : 3
        inv_h² * SVector{NC,T}(ntuple(Val(NC)) do v
            d1 == d2 ? axis_stencil(w2, work, var + (v - 1) * sv, st[d1]) :
            mixed_stencil(w1, work, var + (v - 1) * sv, st[d1], st[d2])
        end)
    end

    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    Hl, dHl = gauge_at(T, Hwork, inner, b, Val(HASH))
    ∂ₜh, ∂ₜΠ = gh_node_rhs_expanded(hv, Πv, ∂h, ∂Π, ∂∂h, Hl, dHl, γ0, γ2)

    φ = α / sqrtγ
    # ∂_i∂_t g: the first evolution equation differentiated along x^i.
    ∂ᵢ∂ₜh = ntuple(Val(3)) do i
        dα, dβ, dsqrtγ, _ = metric_derivatives_along(gu4, α, β, γu, sqrtγ, ∂h[i])
        dφ = (dα - φ * dsqrtγ) / sqrtγ
        acc = dφ * Πv + φ * ∂Π[i]
        acc + (dβ[1] * ∂h[1] + dβ[2] * ∂h[2] + dβ[3] * ∂h[3]) +
        (β[1] * ∂∂h[_pairindex3(i, 1)] + β[2] * ∂∂h[_pairindex3(i, 2)] +
         β[3] * ∂∂h[_pairindex3(i, 3)])
    end
    # ∂_t∂_t g: the same equation differentiated along t, with ∂_tΠ from
    # the reduced equation.
    dαt, dβt, dsqrtγt, _ = metric_derivatives_along(gu4, α, β, γu, sqrtγ, ∂ₜh)
    dφt = (dαt - φ * dsqrtγt) / sqrtγ
    ∂ₜ∂ₜh = dφt * Πv + φ * ∂ₜΠ +
            (dβt[1] * ∂h[1] + dβt[2] * ∂h[2] + dβt[3] * ∂h[3]) +
            (β[1] * ∂ᵢ∂ₜh[1] + β[2] * ∂ᵢ∂ₜh[2] + β[3] * ∂ᵢ∂ₜh[3])

    dd = (∂ₜ∂ₜh, ∂ᵢ∂ₜh[1], ∂ᵢ∂ₜh[2], ∂ᵢ∂ₜh[3],
          ∂∂h[1], ∂∂h[2], ∂∂h[3], ∂∂h[4], ∂∂h[5], ∂∂h[6])
    ℋ, ℳ = adm_constraints_at_node(g4, gu4, α, β, _dg4(∂ₜh, ∂h), _ddg4(dd))

    keep = is_evolved(mask, x)
    diag[inner..., DIAG_HAM, b] = keep ? ℋ : zero(T)
    ntuple(Val(3)) do i
        diag[inner..., DIAG_MOM + i - 1, b] = keep ? ℳ[i] : zero(T)
        nothing
    end
    diag[inner..., DIAG_MASK, b] = keep ? one(T) : zero(T)
end

# The state → working array → ghosts preamble both monitors share. It is
# the right-hand side's first two steps, with that call's `t` in the hook
# (`CLAUDE.md`, "Hooks depend on time"): a monitor evaluated against stale
# ghosts reports the boundary, not the constraint.
function _prepare_monitor!(p::GHProblem, u, t)
    scatter!(p.U, u)
    if p.hasdirichlet
        fill_ghosts!(p.U, p.schedule; boundary=dirichlet(p.case, t))
    else
        fill_ghosts!(p.U, p.schedule)
    end
    return nothing
end

"""
    gh_constraint!(p::GHProblem, u, t; mask = interior_mask(p.interior, t))

Evaluate the gauge constraint `C_a` of the state `u` at time `t` into
`p.diag`, and return `p`.

Scatter, fill the ghosts with this `t`'s hook, launch
[`gh_constraint_kernel!`](@ref) — the right-hand side's own preamble, for
the same reason: the stencils reach `q/2` points past the owned range and
a monitor run on unfilled ghosts measures whatever was left there.

[`constraint_norms`](@ref) is what turns the field into the numbers the
record holds.
"""
function gh_constraint!(p::GHProblem{T}, u, t;
                        mask=interior_mask(p.interior, T(t))) where {T}
    _prepare_monitor!(p, u, t)
    map_blocks!(gh_constraint_kernel!, p.U, p.diag.work, p.U.work,
                gauge_work(p.Hsrc), p.origins, p.spacings, mask,
                p.valG, p.valq, p.valH)
    return p
end

"""
    adm_constraint!(p::GHProblem, u, t; mask = interior_mask(p.interior, t))

Evaluate the ADM Hamiltonian and momentum constraints of the state `u` at
time `t` into `p.diag`, and return `p`.

The expensive monitor — every second derivative of the metric, and the
four-dimensional Ricci tensor — so `CODE.md` runs it every `k`-th chunk
where the gauge constraint runs at every one.
"""
function adm_constraint!(p::GHProblem{T}, u, t;
                         mask=interior_mask(p.interior, T(t))) where {T}
    _prepare_monitor!(p, u, t)
    map_blocks!(adm_constraint_kernel!, p.U, p.diag.work, p.U.work,
                gauge_work(p.Hsrc), p.origins, p.spacings, p.case.γ0,
                p.case.γ2, mask, eltype(p.U.work)(t), p.valG, p.valq, p.valH)
    return p
end

"""
    gh_error!(p::GHProblem, u, t; mask = interior_mask(p.interior, t),
              shell = (0, -1))

Evaluate the error against the analytic solution, the interior residual
and the gauge drift of the state `u` at time `t` into `p.diag`, and return
`p`.

No stencil and no ghosts — [`gh_error_kernel!`](@ref) reads the point —
so this needs the scatter and nothing else. `shell` is the band of radii
the drift is measured over, `(lo, hi)`; the default is empty, and the
driver passes a band around the horizon.

The mask defaults to the problem's own interior at this `t`, which is
`CODE.md`'s rule that the modified region is never reported as a numerical
solution; [`AllPoints`](@ref) is how a test asks for the unmasked number
and sees how much larger it is.
"""
function gh_error!(p::GHProblem{T}, u, t; mask=interior_mask(p.interior, T(t)),
                   shell=(zero(T), -one(T))) where {T}
    scatter!(p.U, u)
    map_blocks!(gh_error_kernel!, p.U, p.diag.work, p.U.work, p.origins,
                p.spacings, p.case.background, p.interior, mask, T(t),
                T(shell[1]), T(shell[2]), p.valG)
    return p
end

"""
    error_norms(p::GHProblem) -> NamedTuple

The three error rows of the analysis record, from whatever
[`gh_error!`](@ref) last wrote into `p.diag`: `err_l2` and `err_linf`
(masked, over the evolved region), `residual` (the layer's L∞) and `drift`
(the horizon shell's L∞ of `|h_tt − h_tt,exact|`).

The residual and the drift are L∞ only, as `CODE.md`'s table asks: an L2
over a region the mask excludes would have to be divided by that region's
own volume, and the number those rows are read for is the worst point.
"""
function error_norms(p::GHProblem{T}) where {T}
    counts = masked_counts(p)
    err = masked_norms(p, DIAG_ERR; counts=counts)
    res = masked_norms(p, DIAG_RES; counts=counts)
    drift = masked_norms(p, DIAG_DRIFT; counts=counts)
    return (err_l2=err.l2, err_linf=err.linf, residual=res.linf,
            drift=drift.linf)
end

"""
    masked_counts(p::GHProblem) -> Vector

How many points of each block the mask counted, from the indicator the
constraint kernels wrote into `DIAG_MASK`.

It is the denominator of every masked norm, it is the same for all of
them, and it is a full sweep of the `diag` array — so
[`constraint_norms`](@ref) takes it once and hands it to the eight norms
it assembles rather than letting each recompute it.
"""
masked_counts(p::GHProblem{T}) where {T} =
    block_mapreduce(identity, +, zero(T), p.diag; vars=DIAG_MASK)

"""
    masked_norms(p::GHProblem, v::Integer; counts) -> (l2, linf)

The volume-weighted L2 and L∞ norms of `diag` variable `v` over the
**evolved** points — the mask's own definition of which those are.

`CODE.md`, "Analysis quantities": the norms are `block_mapreduce`
partials weighted by each block's `h³` and combined **in block order**, so
they are bit-identical across thread counts. The L2 is normalized by the
evolved volume rather than by the domain's, so that masking a region out
does not make the number look smaller than it is; where nothing is masked
that is TreeAMR's `volume_weighted_norm` exactly, which
`test/constraints_tests.jl` asserts rather than assumes.

The masked points hold exactly zero — the kernels wrote it there — so the
maximum is over the evolved points too, and `NaN` from a blown-up run
still propagates.
"""
function masked_norms(p::GHProblem{T}, v::Integer;
                      counts=masked_counts(p)) where {T}
    sq = block_mapreduce(x -> x * x, +, zero(T), p.diag; vars=v)
    mx = block_mapreduce(abs, max, zero(T), p.diag; vars=v)
    forest = p.U.forest
    num = zero(T)
    den = zero(T)
    for b in 1:nblocks(p.diag)
        cellvolume = spacing(T, forest, blockkey(p.diag, b))^3
        num += cellvolume * sq[b]
        den += cellvolume * counts[b]
    end
    l2 = iszero(den) ? zero(T) : sqrt(num / den)
    return (l2=l2, linf=isempty(mx) ? zero(T) : maximum(mx))
end

"""
    constraint_norms(p::GHProblem) -> NamedTuple

The constraint norms of whatever is currently in `p.diag`, as the analysis
record holds them: `gauge_l2` and `gauge_linf` per component of `C_a`
(four each), `ham_l2`, `ham_linf` for `ℋ`, and `mom_l2`, `mom_linf` per
component of `ℳ_i` (three each).

It reads the `diag` field set and launches nothing, so the caller decides
which monitors ran: [`gh_constraint!`](@ref) alone leaves the ADM slots
holding whatever the last ADM pass wrote, which is why the record names
them separately and `CODE.md` gives them different cadences.
"""
function constraint_norms(p::GHProblem{T}) where {T}
    counts = masked_counts(p)
    gauge = ntuple(a -> masked_norms(p, DIAG_CGH + a - 1; counts=counts), Val(4))
    ham = masked_norms(p, DIAG_HAM; counts=counts)
    mom = ntuple(i -> masked_norms(p, DIAG_MOM + i - 1; counts=counts), Val(3))
    return (gauge_l2=SVector{4,T}(ntuple(a -> gauge[a].l2, Val(4))),
            gauge_linf=SVector{4,T}(ntuple(a -> gauge[a].linf, Val(4))),
            ham_l2=ham.l2, ham_linf=ham.linf,
            mom_l2=SVector{3,T}(ntuple(i -> mom[i].l2, Val(3))),
            mom_linf=SVector{3,T}(ntuple(i -> mom[i].linf, Val(3))))
end
