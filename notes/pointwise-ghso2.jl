# Copied verbatim from `/Users/eschnett/src/jl/GeneralizedHarmonicSecondOrder2/src/pointwise.jl` on 2026-09-16, repository at commit `8fd820a`,
# clean at that commit. The original repository is unpublished; this copy is the
# citable reference for TreeGeneralizedHarmonic's `CODE.md`. Do not edit it here;
# amend `CODE.md` instead.

# Pointwise generalized-harmonic algebra.
#
# Conservative first-order-in-time GH system (see METHODS.md and
# Garfinkle, PRD 65, 044029 (2002), eqn. (9)): the state is
# (g_ab − η_ab, Π_ab) with the densitised Lie-advected momentum
#
#     Π_ab = (√γ/α) (∂_t − β^i ∂_i) g_ab = √|g| n^μ ∂_μ g_ab ,
#
# and the per-component evolution is the flux-conservative scalar wave
# with a source,
#
#     ∂_t g_ab = β^i ∂_i g_ab + (α/√γ) Π_ab
#     ∂_t Π_ab = ∂_i F^i_ab − α√γ S0_ab ,
#     F^i_ab   = β^i Π_ab + α√γ γ^{ij} ∂_j g_ab ,
#     S0_ab    = C2sym_ab − 2 Γ2_ab − 2 ∇_(a H_b) − Γ^ν ∂_ν g_ab  (+ damping)
#
# where C_abc = ∂_a g_bc, C2sym_ab = C_a^{μν}C_μνb + C_b^{μν}C_μνa,
# Γ2_ab = Γ^γ_{να}Γ^ν_{γβ}, and the −Γ^ν ∂_ν g_ab term converts the
# densitised operator realised by the conservative flux,
# □_dens = (1/√|g|)∂_μ(√|g| g^{μν} ∂_ν ·), back to the bare reduced
# operator g^{cd}∂_c∂_d. The source enters ∂_t Π with a minus sign
# because Π = √|g| n^μ∂_μ g ⇒ √|g| g^{tν}∂_ν g = −Π.
#
# This file is pure node-local algebra on StaticArrays — no mesh, no
# arrays — shared by the KernelAbstractions RHS kernel (kernels.jl), the
# boundary SAT kernel, the host-side diagnostics, and the tests. All
# functions are type-generic (Float32 / Float64 / MultiFloats) and
# GPU-safe (no heap allocation, no mutation).
#
# Floating-point accuracy: the state stores g − η, and the derived
# quantities are computed as offsets too, without catastrophic
# cancellation:
#     g^{ab} − η^{ab} = −g^{ac} h_{cd} η^{db}        (exact identity)
#     det(g) + 1      = −(e1 + e2 + e3 + e4)(η h)    (elem. sym. polys)
#     det(γ) − 1      = −d1 − q1 + d1 q1
# with h = g − η, d1 = det(g)+1, q1 = g^{tt}+1.

using LinearAlgebra: det, dot, tr
using StaticArrays

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
SpacetimeMetrics convention `dg[a, b, c] = ∂_c g_ab`. `Hup` is the
**contravariant** gauge source (raise the stored lowered `H_a` with
`inv(g) * H_a`). Vanishes for an exact solution of the GH system.
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
