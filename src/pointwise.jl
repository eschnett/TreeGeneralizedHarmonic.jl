# The pointwise generalized-harmonic algebra: everything this package does at
# one point, with no mesh in sight.
#
# Most of this file is GHSO2's `pointwise.jl`, copied into `notes/` on
# 2026-09-16 and ported here **as it is** -- `NC`, `_sym4`, `_pack10`,
# `pack_g`, `pack_sym`, `metric_quantities`, `adm_from_metric`,
# `gh_node_rhs`, `adm_vars_from_state`, `gauge_constraint_at_node`. Those
# expressions were validated against `SpacetimeMetrics` automatic
# differentiation on SBP-SAT spectral elements; changing one is changing a
# validated result, and `CLAUDE.md` says what that costs. What is new here
# is the mesh's half of the decision recorded in `CODE.md`, "The equations":
# this package discretises the **expanded** form of the momentum equation
# rather than GHSO2's flux form, so it needs the closed-form derivatives of
# the coefficients (`metric_derivatives`) and the assembled right-hand side
# that uses them (`gh_node_rhs_expanded`). GHSO2's flux form stays, for the
# tests only -- it is the identity `∂_tΠ − ∂_iF^i = msrc` that validates the
# source term, and it is never called on the mesh.
#
# Three conventions run through the file, and mixing them is the bug that
# looks right in Minkowski and wrong everywhere else (`CLAUDE.md`):
#
#   * **The packed component order** is `(tt, tx, ty, tz, xx, xy, xz, yy,
#     yz, zz)` -- the column-major lower triangle of a symmetric 4x4. `h` is
#     variables 1:10 of the state and `Π` is 11:20. Only `_sym4` and
#     `_pack10` know this; nothing outside this file indexes a component by
#     a literal.
#   * **GHSO2's derivative index order** is `dg[a, b, c] = ∂_a g_bc`, the
#     derivative axis *first*. `gh_node_rhs`, `gh_node_source` and
#     `adm_vars_from_state` speak it.
#   * **SpacetimeMetrics' derivative index order** is `dg[a, b, c] = ∂_c
#     g_ab`, the derivative axis *last*. `gauge_constraint_at_node` speaks
#     it, because it is fed straight from `dmetric`. Each function's
#     docstring says which one it takes; the mesh-side conversion happens in
#     `initialdata.jl` and nowhere else.
#
# Floating-point hygiene, GHSO2's, and the reason the state stores the offset
# `h = g − η` rather than `g`: every derived quantity that would otherwise be
# a difference of O(1) numbers is computed as an offset instead,
#
#     g^{ab} − η^{ab} = −g^{ac} h_{cd} η^{db}        (exact identity)
#     det(g) + 1      = −(e1 + e2 + e3 + e4)(η h)    (elem. sym. polys)
#     det(γ) − 1      = −d1 − q1 + d1 q1
#
# with `d1 = det(g) + 1` and `q1 = g^{tt} + 1`, so that `α`, `β^i` and `√γ`
# keep full relative precision at `‖h‖ ~ 1e−13` where the naive route keeps
# none. `test/pointwise_tests.jl` measures both.
#
# Everything here is pure algebra on StaticArrays: no arrays, no allocation,
# no mutation, type-generic (`Float32`, `Float64`, MultiFloats) and callable
# from a KernelAbstractions kernel on any backend.

"Number of independent components of a symmetric 4×4 tensor."
const NC = 10

@inline _η4(::Type{T}) where {T} =
    SMatrix{4,4,T}(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)

# Symmetric 4×4 from the 10 packed components in pack_g order
# (column-major lower-triangular: tt, tx, ty, tz, xx, xy, xz, yy, yz, zz).
@inline function _sym4(v::SVector{NC,T}) where {T}
    return SMatrix{4,4,T}(v[1], v[2], v[3], v[4],
                          v[2], v[5], v[6], v[7],
                          v[3], v[6], v[8], v[9],
                          v[4], v[7], v[9], v[10])
end

@inline function _pack10(M::SMatrix{4,4,T}) where {T}
    return SVector{NC,T}(M[1,1], M[2,1], M[3,1], M[4,1],
                         M[2,2], M[3,2], M[4,2],
                         M[3,3], M[4,3], M[4,4])
end

"""
    pack_g(g::SMatrix{4,4,T}) -> SVector{10,T}

Pack the unique entries of a symmetric 4×4 metric minus η in the
column-major lower-triangular component order used throughout:
`(g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz) − η`.
"""
@inline pack_g(g::SMatrix{4,4,T}) where {T} = _pack10(g - _η4(T))

"""
    pack_sym(M::SMatrix{4,4}) -> SVector{10}

Pack a symmetric 4×4 tensor (no η offset — e.g. a metric derivative).
"""
@inline pack_sym(M::SMatrix{4,4,T}) where {T} = _pack10(M)

"""
    metric_quantities(h::SMatrix{4,4,T}) -> (g4, gu4, α, β, γu, sqrtγ, guo)

Derived metric quantities from the offset `h = g − η`: the full metric
`g4`, the inverse `gu4 = g^{ab}` (computed via the cancellation-free
offset identity `g^{ab} − η^{ab} = −g^{ac} h_{cd} η^{db}`), the ADM
lapse `α = 1/√(−g^{tt})`, contravariant shift `β^i = −g^{ti}/g^{tt}`,
inverse spatial metric `γ^{ij} = g^{ij} − g^{ti}g^{tj}/g^{tt}`,
`√γ = √(det γ_ij)` via the offset `det(γ) − 1`, and the inverse-metric
offset `guo = g^{ab} − η^{ab}` itself (full relative precision even for
‖h‖ ≪ 1; note `gu4 = η + guo` rounds the offset to absolute eps, so use
`guo` where the offset matters). All cancellation-sensitive quantities
(`α`, `β`, `√γ`) are computed from the offsets, never by subtracting
O(1) terms.
"""
@inline function metric_quantities(h::SMatrix{4,4,T}) where {T}
    η = _η4(T)
    g4 = η + h
    # Inverse-metric offset: inv(A) − inv(B) = −inv(A)(A−B)inv(B) with
    # A = η+h, B = η (and inv(η) = η) gives gu − η = −gu·h·η. Evaluating
    # the right side with the directly-computed inverse keeps the offset
    # accurate to a relative eps even when ‖h‖ ≪ 1.
    guo = -(inv(g4) * h) * η
    gu4 = η + guo
    gutt = gu4[1, 1]
    q1 = guo[1, 1]                       # g^{tt} + 1
    α = inv(sqrt(one(T) - q1))           # 1/√(−g^{tt})
    β = SVector{3,T}(-gu4[1,2] / gutt, -gu4[1,3] / gutt, -gu4[1,4] / gutt)
    γu = SMatrix{3,3,T}(
        gu4[1+1,1+1] - gu4[1,1+1]*gu4[1,1+1]/gutt,
        gu4[2+1,1+1] - gu4[1,2+1]*gu4[1,1+1]/gutt,
        gu4[3+1,1+1] - gu4[1,3+1]*gu4[1,1+1]/gutt,
        gu4[1+1,2+1] - gu4[1,1+1]*gu4[1,2+1]/gutt,
        gu4[2+1,2+1] - gu4[1,2+1]*gu4[1,2+1]/gutt,
        gu4[3+1,2+1] - gu4[1,3+1]*gu4[1,2+1]/gutt,
        gu4[1+1,3+1] - gu4[1,1+1]*gu4[1,3+1]/gutt,
        gu4[2+1,3+1] - gu4[1,2+1]*gu4[1,3+1]/gutt,
        gu4[3+1,3+1] - gu4[1,3+1]*gu4[1,3+1]/gutt)
    # det(g)+1 from the elementary symmetric polynomials of B = η·h:
    # det(η+h) = det(η)·det(I+B) = −(1 + e1 + e2 + e3 + e4).
    B = η * h
    trB = tr(B)
    B2 = B * B
    trB2 = tr(B2)
    trB3 = tr(B2 * B)
    e1 = trB
    e2 = (trB*trB - trB2) / 2
    e3 = (trB*trB*trB - 3*trB*trB2 + 2*trB3) / 6
    e4 = det(B)
    d1 = -(e1 + e2 + e3 + e4)            # det(g4) + 1
    # det(γ) = det(g4)·g^{tt} ⇒ det(γ) − 1 = −d1 − q1 + d1·q1.
    detγm1 = -d1 - q1 + d1*q1
    sqrtγ = sqrt(one(T) + detγm1)
    return g4, gu4, α, β, γu, sqrtγ, guo
end

"""
    adm_from_metric(g::SMatrix{4,4,T}) -> (α, β::SVector{3}, γ::SMatrix{3,3})

ADM 3+1 split of a covariant 4-metric: lapse `α = 1/√(−g^{tt})`,
contravariant shift `β^i = −g^{ti}/g^{tt}`, spatial metric `γ_ij = g_ij`.
"""
@inline function adm_from_metric(g::SMatrix{4,4,T}) where {T}
    gu = inv(g)
    gutt = gu[1, 1]
    α = one(T) / sqrt(-gutt)
    β = SVector{3,T}(-gu[1,2]/gutt, -gu[1,3]/gutt, -gu[1,4]/gutt)
    γ = SMatrix{3,3,T}(g[2,2], g[3,2], g[4,2],
                       g[2,3], g[3,3], g[4,3],
                       g[2,4], g[3,4], g[4,4])
    return α, β, γ
end

"""
    gh_node_rhs(h, Pi, dxg, dyg, dzg, Hl, dHl, γ0, γ2)
        -> (dtg, Fx, Fy, Fz, msrc)

The complete pointwise GH right-hand side at one node. Inputs are the
packed 10-component state and spatial gradients (`SVector{10}`), the
lowered gauge source `Hl_b` (`SVector{4}`) with gradient
`dHl[a,b] = ∂_a Hl_b` (`SMatrix{4,4}`), and the Gundlach–Pretorius
damping parameters `γ0 ≥ 0`, `γ2`. Outputs (each `SVector{10}`):

  * `dtg`  = ∂_t g_ab = β^i ∂_i g_ab + (α/√γ) Π_ab,
  * `Fx, Fy, Fz` = F^i_ab = β^i Π_ab + α√γ γ^{ij} ∂_j g_ab,
  * `msrc` = −α√γ (S0_ab + damping)  — the non-divergence part of ∂_t Π.

The mesh driver completes ∂_t Π_ab = ∂_i F^i_ab + msrc_ab.

**This is GHSO2's flux form, and this package does not discretise it.**
It is ported verbatim from `notes/pointwise-ghso2.jl` and kept for the
tests: the identity `∂_tΠ − ∂_iF^i = msrc`, with `∂_iF^i` a finite
difference of the analytic `F^i`, is what says the source algebra is the
right reduced Einstein equation, and `gh_node_rhs_expanded` — what the
mesh does evaluate — is checked against the divergence of these same
fluxes. `CODE.md`, "The equations", says why the flux form buys nothing
on a mesh whose ghosts are interpolated. Its internal derivative index
order is GHSO2's, `dg[a, b, c] = ∂_a g_bc`.
"""
@inline function gh_node_rhs(h::SVector{NC,T}, Pi::SVector{NC,T},
                             dxg::SVector{NC,T}, dyg::SVector{NC,T},
                             dzg::SVector{NC,T},
                             Hl::SVector{4,T}, dHl::SMatrix{4,4,T},
                             γ0::T, γ2::T) where {T}
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    a_div = α / sqrtγ
    a_mul = α * sqrtγ

    # ∂_t g_ab (packed components are scalars ⇒ elementwise SVector ops).
    dtg = β[1]*dxg + β[2]*dyg + β[3]*dzg + a_div*Pi

    # dg[a, b, c] = ∂_a g_bc with derivative axis a ∈ (t, x, y, z).
    dt4 = _sym4(dtg); dx4 = _sym4(dxg); dy4 = _sym4(dyg); dz4 = _sym4(dzg)
    dg = SArray{Tuple{4,4,4},T}(
        (a == 1 ? dt4[b,c] : a == 2 ? dx4[b,c] : a == 3 ? dy4[b,c] : dz4[b,c])
        for a in 1:4, b in 1:4, c in 1:4)

    # C_a^b_c = g^{bx} ∂_a g_xc ;  C_a^{bc} = g^{cy} C_a^b_y.
    Clul = SArray{Tuple{4,4,4},T}(
        sum(gu4[b,x] * dg[a,x,c] for x in 1:4) for a in 1:4, b in 1:4, c in 1:4)
    Cluu = SArray{Tuple{4,4,4},T}(
        sum(gu4[c,y] * Clul[a,b,y] for y in 1:4) for a in 1:4, b in 1:4, c in 1:4)
    # C2_ab = C_a^{μν} C_μνb ; symmetrised C2sym = C2 + C2ᵀ.
    C2 = SMatrix{4,4,T}(
        sum(Cluu[a,x,y] * dg[x,y,b] for x in 1:4, y in 1:4) for a in 1:4, b in 1:4)
    C2sym = C2 + C2'

    # Christoffels Γ_abc = ½(∂_b g_ac + ∂_c g_ab − ∂_a g_bc); Γ^a_bc; the
    # contraction Γ2_ab = Γ^γ_{να}Γ^ν_{γβ}; contracted Γ^ν = g^{cd}Γ^ν_{cd}.
    Γlll = SArray{Tuple{4,4,4},T}(
        (dg[b,a,c] + dg[c,a,b] - dg[a,b,c]) / 2 for a in 1:4, b in 1:4, c in 1:4)
    Γ4 = SArray{Tuple{4,4,4},T}(
        sum(gu4[a,x] * Γlll[x,b,c] for x in 1:4) for a in 1:4, b in 1:4, c in 1:4)
    Γ2 = SMatrix{4,4,T}(
        sum(Γ4[x,y,a] * Γ4[y,x,b] for x in 1:4, y in 1:4) for a in 1:4, b in 1:4)
    Γup = SVector{4,T}(
        sum(gu4[d,e] * Γ4[c,d,e] for d in 1:4, e in 1:4) for c in 1:4)

    # Reduced source S0 = C2sym − 2Γ2 − (∂_aH_b + ∂_bH_a) + 2Γ^c_{ab}H_c
    #                     − Γ^ν ∂_ν g_ab.
    S0 = SMatrix{4,4,T}(
        C2sym[a,b] - 2*Γ2[a,b]
        - dHl[a,b] - dHl[b,a] + 2*sum(Γ4[c,a,b] * Hl[c] for c in 1:4)
        - sum(Γup[c] * dg[c,a,b] for c in 1:4)
        for a in 1:4, b in 1:4)

    # Gundlach–Pretorius constraint damping: with the gauge constraint
    # C^μ = Γ^μ + H^μ (lowered C_a = Γ_a + H_a) and the foliation normal
    # one-form t_a = −α δ_a^t,
    #   Δ_ab = γ0 [ t_a C_b + t_b C_a − (1+γ2) g_ab t^c C_c ] ,
    # added to S0 (which enters ∂_tΠ with −α√γ, so γ0 > 0 damps).
    # Vanishes on the constraint surface ⇒ exact solutions are untouched.
    if γ0 != 0
        Cup = Γup + gu4 * Hl
        Cl = g4 * Cup
        tl = SVector{4,T}(-α, zero(T), zero(T), zero(T))
        tu = gu4 * tl
        tC = dot(tu, Cl)
        Δ = SMatrix{4,4,T}(
            γ0 * (tl[a]*Cl[b] + tl[b]*Cl[a] - (1+γ2)*g4[a,b]*tC)
            for a in 1:4, b in 1:4)
        S0 = S0 + Δ
    end

    # Fluxes F^i_ab = β^i Π_ab + α√γ γ^{ij} ∂_j g_ab.
    Fx = β[1]*Pi + a_mul*(γu[1,1]*dxg + γu[1,2]*dyg + γu[1,3]*dzg)
    Fy = β[2]*Pi + a_mul*(γu[2,1]*dxg + γu[2,2]*dyg + γu[2,3]*dzg)
    Fz = β[3]*Pi + a_mul*(γu[3,1]*dxg + γu[3,2]*dyg + γu[3,3]*dzg)

    msrc = -a_mul * _pack10(S0)
    return dtg, Fx, Fy, Fz, msrc
end

# dg[a, b, c] = ∂_a g_bc from the time derivative and the three spatial
# gradients, all packed. GHSO2's derivative index order, derivative axis
# first. Written once, because it is the one place the two conventions
# could be transposed silently.
@inline function _dg4(dtg::SVector{NC,T}, ∂h::NTuple{3,SVector{NC,T}}) where {T}
    dt4 = _sym4(dtg)
    dx4 = _sym4(∂h[1]); dy4 = _sym4(∂h[2]); dz4 = _sym4(∂h[3])
    return SArray{Tuple{4,4,4},T}(
        (a == 1 ? dt4[b,c] : a == 2 ? dx4[b,c] : a == 3 ? dy4[b,c] : dz4[b,c])
        for a in 1:4, b in 1:4, c in 1:4)
end

"""
    gh_node_source(g4, gu4, α, sqrtγ, dg, Hl, dHl, γ0, γ2) -> SVector{10}

`msrc_ab = −α√γ (S0_ab + Z_ab)`, the part of `∂_t Π_ab` that is not a
derivative of the state: GHSO2's reduced source together with the
Gundlach–Pretorius constraint damping. `dg[a, b, c] = ∂_a g_bc` is
**GHSO2's** derivative index order, derivative axis first, and its `a = 1`
slice is `∂_t g` from the first evolution equation — not a stored field.

This is a **second copy** of [`gh_node_rhs`](@ref)'s source block, written
out of it character for character rather than factored out of it:
`gh_node_rhs` keeps its own, exactly as `notes/pointwise-ghso2.jl` has it,
because staying diffable against that file is what makes it the validated
reference the tests' flux identity runs through. So the two copies coexist,
and `test/pointwise_identity_tests.jl` asserts they agree **to roundoff**
on every background. If that assertion ever fires, one of them has
drifted; fix the copy, not the port. (To roundoff and not bit for bit,
although the two are the same characters: the compiler fuses a multiply
and an add in one inlining context and not in the other, and the observed
disagreement is one unit in the last place — see `CLAUDE.md`, "Two
spellings of one expression are not bit-identical".)

It takes the coefficient set rather than `h`, as
[`metric_derivatives`](@ref) does, because the streaming kernel of
`CODE.md`'s "One right-hand-side evaluation" has already built it: the
source is step 3 there, after the coefficients of step 1 and the
per-component stencils of step 2.
"""
@inline function gh_node_source(g4::SMatrix{4,4,T}, gu4::SMatrix{4,4,T},
                                α::T, sqrtγ::T,
                                dg::SArray{Tuple{4,4,4},T},
                                Hl::SVector{4,T}, dHl::SMatrix{4,4,T},
                                γ0::T, γ2::T) where {T}
    a_mul = α * sqrtγ

    # C_a^b_c = g^{bx} ∂_a g_xc ;  C_a^{bc} = g^{cy} C_a^b_y.
    Clul = SArray{Tuple{4,4,4},T}(
        sum(gu4[b,x] * dg[a,x,c] for x in 1:4) for a in 1:4, b in 1:4, c in 1:4)
    Cluu = SArray{Tuple{4,4,4},T}(
        sum(gu4[c,y] * Clul[a,b,y] for y in 1:4) for a in 1:4, b in 1:4, c in 1:4)
    # C2_ab = C_a^{μν} C_μνb ; symmetrised C2sym = C2 + C2ᵀ.
    C2 = SMatrix{4,4,T}(
        sum(Cluu[a,x,y] * dg[x,y,b] for x in 1:4, y in 1:4) for a in 1:4, b in 1:4)
    C2sym = C2 + C2'

    # Christoffels Γ_abc = ½(∂_b g_ac + ∂_c g_ab − ∂_a g_bc); Γ^a_bc; the
    # contraction Γ2_ab = Γ^γ_{να}Γ^ν_{γβ}; contracted Γ^ν = g^{cd}Γ^ν_{cd}.
    Γlll = SArray{Tuple{4,4,4},T}(
        (dg[b,a,c] + dg[c,a,b] - dg[a,b,c]) / 2 for a in 1:4, b in 1:4, c in 1:4)
    Γ4 = SArray{Tuple{4,4,4},T}(
        sum(gu4[a,x] * Γlll[x,b,c] for x in 1:4) for a in 1:4, b in 1:4, c in 1:4)
    Γ2 = SMatrix{4,4,T}(
        sum(Γ4[x,y,a] * Γ4[y,x,b] for x in 1:4, y in 1:4) for a in 1:4, b in 1:4)
    Γup = SVector{4,T}(
        sum(gu4[d,e] * Γ4[c,d,e] for d in 1:4, e in 1:4) for c in 1:4)

    # Reduced source S0 = C2sym − 2Γ2 − (∂_aH_b + ∂_bH_a) + 2Γ^c_{ab}H_c
    #                     − Γ^ν ∂_ν g_ab.
    S0 = SMatrix{4,4,T}(
        C2sym[a,b] - 2*Γ2[a,b]
        - dHl[a,b] - dHl[b,a] + 2*sum(Γ4[c,a,b] * Hl[c] for c in 1:4)
        - sum(Γup[c] * dg[c,a,b] for c in 1:4)
        for a in 1:4, b in 1:4)

    # Gundlach–Pretorius constraint damping: with the gauge constraint
    # C^μ = Γ^μ + H^μ (lowered C_a = Γ_a + H_a) and the foliation normal
    # one-form t_a = −α δ_a^t,
    #   Δ_ab = γ0 [ t_a C_b + t_b C_a − (1+γ2) g_ab t^c C_c ] ,
    # added to S0 (which enters ∂_tΠ with −α√γ, so γ0 > 0 damps).
    # Vanishes on the constraint surface ⇒ exact solutions are untouched.
    if γ0 != 0
        Cup = Γup + gu4 * Hl
        Cl = g4 * Cup
        tl = SVector{4,T}(-α, zero(T), zero(T), zero(T))
        tu = gu4 * tl
        tC = dot(tu, Cl)
        Δ = SMatrix{4,4,T}(
            γ0 * (tl[a]*Cl[b] + tl[b]*Cl[a] - (1+γ2)*g4[a,b]*tC)
            for a in 1:4, b in 1:4)
        S0 = S0 + Δ
    end

    return -a_mul * _pack10(S0)
end

"""
    metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h) -> (dα, dβ, dA)
    metric_derivatives(h, ∂h)                    -> (dα, dβ, dA)

The spatial derivatives of the coefficients the expanded momentum equation
needs, in closed form from the coefficient set and the state's spatial
gradients:

  * `dα[i]     = ∂_i α`                    (`SVector{3}`),
  * `dβ[i, j]  = ∂_i β^j`                  (`SMatrix{3,3}`),
  * `dA[i,j,k] = ∂_i (α √γ γ^{jk})`        (`SArray{3,3,3}`).

`∂h[i]` is the packed derivative of the offset metric along `x^i`; because
`η` is constant, `∂_i h = ∂_i g`. The first four arguments are
[`metric_quantities`](@ref)'s `gu4`, `α`, `β`, `γu` and `sqrtγ` — the
coefficient set, taken rather than rebuilt, exactly as
[`gh_node_source`](@ref) takes it and for the same reason: the streaming
kernel of `CODE.md`'s "One right-hand-side evaluation" forms it *once* per
point, in step 1 of that order, and rebuilding it here would buy a second
`inv`, a second determinant and a second square root per point. The
`(h, ∂h)` method is the convenience wrapper — it calls `metric_quantities`
and forwards — and is what the tests and the host-side diagnostics use;
`test/pointwise_tests.jl` asserts the two agree **to roundoff**, which is
all that is true of them: the same body inlined at two call sites is
contracted into fused multiply-adds differently, and the answers differ in
the last place (`CLAUDE.md`, "Two spellings of one expression are not
bit-identical" — one spelling at two call sites is already enough).

This is the other half of `CODE.md`'s decision to discretise the expanded
form `(EXPANDED)` rather than the flux form: the divergence of the flux is
a derivative of a *computed* quantity, which on a finite-difference mesh
costs either a second ghost exchange or ghosts twice as wide, and the
product rule removes both — at the price of these coefficients. Deriving
them is a chain rule through the algebraic map `h ↦ (α, β, √γ, γ^{jk})`:

    ∂_i g^{ab} = −g^{ac} (∂_i g_cd) g^{db}
    ∂_i α      = ½ α³ ∂_i g^{tt}
    ∂_i β^j    = −(∂_i g^{tj} + β^j ∂_i g^{tt}) / g^{tt}
    ∂_i γ^{jk} = ∂_i g^{jk} + β^k ∂_i g^{tj} + β^j ∂_i g^{tk}
                 + β^j β^k ∂_i g^{tt}
    ∂_i √γ     = ½ √γ γ^{jk} ∂_i γ_jk

and the product rule for `A^{jk} = α √γ γ^{jk}`. A forward-mode dual pass
through [`metric_quantities`](@ref) is the independent check, and
`test/pointwise_tests.jl` runs it on every background.

The right-hand side contracts these to four numbers — `∂_i β^i` and
`∂_i A^{ij}` — and the unused components fall out of the inlined code, so
the streaming kernel contracts rather than asking for a leaner shape.
"""
@inline function metric_derivatives(h::SVector{NC,T},
                                    ∂h::NTuple{3,SVector{NC,T}}) where {T}
    _, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    return metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
end

@inline function metric_derivatives(gu4::SMatrix{4,4,T}, α::T,
                                    β::SVector{3,T}, γu::SMatrix{3,3,T},
                                    sqrtγ::T,
                                    ∂h::NTuple{3,SVector{NC,T}}) where {T}
    gutt = gu4[1, 1]

    # ∂_i g_ab, and the inverse metric's derivative by the exact identity
    # ∂_i g^{ab} = −g^{ac} (∂_i g_cd) g^{db}.
    dgl = ntuple(i -> _sym4(∂h[i]), Val(3))
    dgu = ntuple(i -> -(gu4 * dgl[i]) * gu4, Val(3))

    # ∂_i α = ½ α³ ∂_i g^{tt}.
    dα = SVector{3,T}(α*α*α * dgu[i][1,1] / 2 for i in 1:3)

    # ∂_i β^j = −(∂_i g^{tj} + β^j ∂_i g^{tt}) / g^{tt}.
    dβ = SMatrix{3,3,T}(
        -(dgu[i][1, j+1] + β[j] * dgu[i][1,1]) / gutt for i in 1:3, j in 1:3)

    # ∂_i √γ = ½ √γ γ^{jk} ∂_i γ_jk, with γ_jk the spatial block of g_ab.
    dsqrtγ = SVector{3,T}(
        sqrtγ * sum(γu[j,k] * dgl[i][j+1, k+1] for j in 1:3, k in 1:3) / 2
        for i in 1:3)

    # ∂_i γ^{jk}, from γ^{jk} = g^{jk} − g^{tj}g^{tk}/g^{tt} and
    # β^j = −g^{tj}/g^{tt}.
    dγu = ntuple(Val(3)) do i
        SMatrix{3,3,T}(
            dgu[i][j+1, k+1] + β[k]*dgu[i][1, j+1] + β[j]*dgu[i][1, k+1]
            + β[j]*β[k]*dgu[i][1,1]
            for j in 1:3, k in 1:3)
    end

    # ∂_i (α √γ γ^{jk}) by the product rule.
    dA = SArray{Tuple{3,3,3},T}(
        dα[i]*sqrtγ*γu[j,k] + α*dsqrtγ[i]*γu[j,k] + α*sqrtγ*dγu[i][j,k]
        for i in 1:3, j in 1:3, k in 1:3)

    return dα, dβ, dA
end

"""
    gh_node_rhs_expanded(h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl, γ0, γ2) -> (∂ₜh, ∂ₜΠ)

The pointwise right-hand side this package evolves, `(EXPANDED)` of
`CODE.md`'s "The equations":

    ∂ₜh_ab = β^i ∂_i h_ab + (α/√γ) Π_ab
    ∂ₜΠ_ab = β^i ∂_i Π_ab + (∂_i β^i) Π_ab
           + α√γ γ^{ij} ∂_i ∂_j h_ab + ∂_i(α√γ γ^{ij}) ∂_j h_ab
           − α√γ (S0_ab + Z_ab)

Arguments, all packed in the `(tt, tx, …, zz)` order and all `isbits`:
the state `h`, `Π :: SVector{10}`; the gradients `∂h[i] = ∂_i h`,
`∂Π[i] = ∂_i Π`, each an `NTuple{3,SVector{10}}`; the second derivatives
`∂∂h`, an `NTuple{6,SVector{10}}` in the column-major lower-triangular
order `(xx, xy, xz, yy, yz, zz)` — the same packing as a symmetric tensor,
one index range shorter; the lowered gauge source `Hl_b` with
`dHl[a, b] = ∂_a Hl_b`; and the damping parameters `γ0 ≥ 0`, `γ2 > −1`.

It is the same equation as [`gh_node_rhs`](@ref)'s flux form and differs
only in what is discretised: here every derivative is a derivative of the
*state*, taken with a compact centered stencil of one half-width, so the
ghost width is set by the dissipation operator and not by a derivative of
a derivative. `test/pointwise_identity_tests.jl` asserts the two agree to
roundoff on analytic data, with the flux divergence differentiated
exactly rather than differenced.

The body is straight-line: the elementwise `SVector{10}` combinations
below unroll to scalar arithmetic with no loop and no allocation, which is
the "scalarised" of `CODE.md`'s "One right-hand-side evaluation"
**(decided in step 1**: written by hand in this form rather than
generated, and the register pressure it produces is G6's measurement, not
G1's**)**. Step 3's kernel does not call this function as a whole — it
streams the second derivatives component by component, exactly the three
groups of terms above — but it calls the same
[`metric_derivatives`](@ref) and [`gh_node_source`](@ref), and this is
what it is checked against.
"""
@inline function gh_node_rhs_expanded(h::SVector{NC,T}, Π::SVector{NC,T},
                                      ∂h::NTuple{3,SVector{NC,T}},
                                      ∂Π::NTuple{3,SVector{NC,T}},
                                      ∂∂h::NTuple{6,SVector{NC,T}},
                                      Hl::SVector{4,T}, dHl::SMatrix{4,4,T},
                                      γ0::T, γ2::T) where {T}
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    a_div = α / sqrtγ
    a_mul = α * sqrtγ
    A = a_mul * γu                              # A^{jk} = α√γ γ^{jk}

    # The coefficient set is already in hand, so it is passed rather than
    # rebuilt — the same call the streaming kernel of step 3 makes.
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1,1] + dβ[2,2] + dβ[3,3]          # ∂_i β^i
    divA = SVector{3,T}(dA[1,1,j] + dA[2,2,j] + dA[3,3,j] for j in 1:3)

    # ∂ₜh_ab: the first evolution equation, and the ∂_t g_ab the source
    # needs.
    ∂ₜh = β[1]*∂h[1] + β[2]*∂h[2] + β[3]*∂h[3] + a_div*Π

    # The advective and principal parts of ∂ₜΠ_ab. ∂∂h is packed lower
    # triangular, so the off-diagonal terms carry the factor 2.
    ∂ₜΠ = β[1]*∂Π[1] + β[2]*∂Π[2] + β[3]*∂Π[3] + divβ*Π +
          A[1,1]*∂∂h[1] + 2*A[1,2]*∂∂h[2] + 2*A[1,3]*∂∂h[3] +
          A[2,2]*∂∂h[4] + 2*A[2,3]*∂∂h[5] + A[3,3]*∂∂h[6] +
          divA[1]*∂h[1] + divA[2]*∂h[2] + divA[3]*∂h[3]

    msrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)

    return ∂ₜh, ∂ₜΠ + msrc
end

"""
    adm_vars_from_state(h, Pi, dxh, dyh, dzh)
        -> (γ::SMatrix{3,3}, ∂γ::SArray{3,3,3}, K::SMatrix{3,3})

ADM Cauchy data at one node from the packed GH state and its spatial
gradients: the spatial metric `γ_ij`, its Cartesian derivatives
`∂γ[i,j,k] = ∂_k γ_ij`, and the extrinsic curvature

    K_ij = −(∂_t γ_ij − D_i β_j − D_j β_i)/(2α),

with `∂_t g` from the evolution relation `β^i ∂_i g + (α/√γ) Π` and
`D_i β_j = ∂_i β_j − Γ^k_{ij} β_k` (3-Christoffels of γ). This is the
quantity set the apparent-horizon finder consumes.
"""
@inline function adm_vars_from_state(h::SVector{NC,T}, Pi::SVector{NC,T},
                                     dxh::SVector{NC,T}, dyh::SVector{NC,T},
                                     dzh::SVector{NC,T}) where {T}
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    dtg = β[1]*dxh + β[2]*dyh + β[3]*dzh + (α / sqrtγ) * Pi
    dt4 = _sym4(dtg)
    dx4 = _sym4(dxh); dy4 = _sym4(dyh); dz4 = _sym4(dzh)
    γ = SMatrix{3,3,T}(g4[i+1, j+1] for i in 1:3, j in 1:3)
    ∂γ = SArray{Tuple{3,3,3},T}(
        (k == 1 ? dx4[i+1, j+1] : k == 2 ? dy4[i+1, j+1] : dz4[i+1, j+1])
        for i in 1:3, j in 1:3, k in 1:3)
    # Lowered shift and its derivatives: β_j = g_{tj}.
    βl = SVector{3,T}(g4[1, 2], g4[1, 3], g4[1, 4])
    dβl = SMatrix{3,3,T}(
        (i == 1 ? dx4[1, j+1] : i == 2 ? dy4[1, j+1] : dz4[1, j+1])
        for i in 1:3, j in 1:3)                     # dβl[i,j] = ∂_i β_j
    # 3-Christoffels Γ^k_{ij} = ½ γ^{kl}(∂_i γ_lj + ∂_j γ_li − ∂_l γ_ij).
    Γ3 = SArray{Tuple{3,3,3},T}(
        sum(γu[k, l] * (∂γ[l, j, i] + ∂γ[l, i, j] - ∂γ[i, j, l]) / 2
            for l in 1:3)
        for k in 1:3, i in 1:3, j in 1:3)
    K = SMatrix{3,3,T}(
        -(dt4[i+1, j+1] - dβl[i, j] - dβl[j, i] +
          2 * sum(Γ3[k, i, j] * βl[k] for k in 1:3)) / (2α)
        for i in 1:3, j in 1:3)
    return γ, ∂γ, (K + K') / 2
end

"""
    gauge_constraint_at_node(g, dg, Hup = 0) -> SVector{4,T}

The generalized-harmonic gauge constraint `C^μ = Γ^μ + H^μ` with
`Γ^μ = g^{αβ} Γ^μ_{αβ}` the contracted Christoffel. `dg` uses the
**SpacetimeMetrics** convention `dg[a, b, c] = ∂_c g_ab` — the derivative
axis *last*, the opposite of every other function in this file, because
this one is fed straight from `dmetric`. `Hup` is the **contravariant**
gauge source (raise the stored lowered `H_a` with `inv(g) * H_a`).
Vanishes for an exact solution of the GH system.
"""
@inline function gauge_constraint_at_node(g::SMatrix{4,4,T},
                                          dg::SArray{Tuple{4,4,4},T},
                                          Hup::SVector{4,T} = zero(SVector{4,T})) where {T}
    gu = inv(g)
    Γlll = SArray{Tuple{4,4,4},T}(
        (dg[a,c,b] + dg[a,b,c] - dg[b,c,a]) / 2
        for a in 1:4, b in 1:4, c in 1:4)
    Γ = SArray{Tuple{4,4,4},T}(
        sum(gu[μ,γ] * Γlll[γ,α,β] for γ in 1:4)
        for μ in 1:4, α in 1:4, β in 1:4)
    Γup = SVector{4,T}(
        sum(gu[α,β] * Γ[μ,α,β] for α in 1:4, β in 1:4) for μ in 1:4)
    return Γup + Hup
end
