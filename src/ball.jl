# The cube and the shells (step B5): the state on TreeMultiPatch's
# thirteen-patch ball. The cube's blocks run the kernel of `evolution.jl`
# unchanged, with their physical origins and spacings; the shells' blocks run
# the same stencils in their logical coordinates with the map's chain rule
# contracted into the coefficients. The state stays in Cartesian components on
# every patch, so a glue interpolates each component as a scalar (TreeAMR's
# `glue = :interpolate`) and only the derivatives see the map. See `CODE.md`,
# "The cube and the shells".

"""
    BallGeometry

What a [`GHProblem`](@ref) on a forest of patches needs beyond the brick's: the
atlas `layout` (a TreeMultiPatch layout, `isbits`, a kernel argument), each local
block's patch, the cube's blocks' **physical** origins and spacings (the cube's
map is affine, `x = origin + scale ⋅ ξ`), and `hmin`, the smallest of the cube's
physical spacings and the shells' logical spacings over their largest logical
speed of light, which is what the time step reads. `frozen` makes the outer
sphere's hook the data at `t = 0` (`CODE.md`, "The outer boundary and the wave
zone", option 1). Built by [`ball_geometry`](@ref).
"""
struct BallGeometry{L,P,O,V,T}
    layout::L
    patches::P
    corigins::O
    cspacings::V
    hmin::T
    frozen::Bool
end

"""
    ball_geometry(U::FieldSet, layout; frozen = false) -> BallGeometry

The [`BallGeometry`](@ref) of `U`'s forest under the atlas `layout`, which must
be the forest's (TreeMultiPatch's `check_layout`). `hmin` is a minimum over the
ranks.
"""
function ball_geometry(U::FieldSet{T,3}, layout; frozen::Bool=false) where {T}
    forest = U.forest
    TreeMultiPatch.check_layout(layout, forest)
    TreeMultiPatch.affine_patch(layout, 1) || throw(ArgumentError(
        "the cube and the shells keep the holes and the refinement in patch 1, " *
        "whose map must be affine, and this layout's patch 1 is not"))
    backend = get_backend(U.work)
    patches = block_patches(forest)
    origins = block_origins(forest, T)
    spacings = block_spacings(forest, T)
    o1, s1 = TreeMultiPatch.affine_map(layout, 1)
    scale = T(s1)
    corigins = map(1:nblocks(U)) do b
        o = origins[b]
        patches[b] == 1 ?
            ntuple(d -> T(o1[d]) + scale * o[d], Val(3)) : o
    end
    cspacings = map(b -> patches[b] == 1 ? scale * spacings[b] : spacings[b],
                    1:nblocks(U))
    hmin = typemax(T)
    for b in 1:nblocks(U)
        if patches[b] == 1
            hmin = min(hmin, cspacings[b])
            continue
        end
        # The largest logical speed over a 3³ lattice of the closed block.
        h, o = spacings[b], origins[b]
        c = zero(T)
        for I in CartesianIndices((3, 3, 3))
            ξ = ntuple(d -> o[d] + (I[d] - 1) * (forest.N * h / 2), Val(3))
            c = max(c, T(TreeMultiPatch.logical_speed(layout, ξ, patches[b])))
        end
        hmin = min(hmin, h / c)
    end
    hmin = rank_reduce(min, forest, hmin; present=nblocks(U) > 0)
    return BallGeometry(layout, to_backend(backend, patches),
                        to_backend(backend, corigins), to_backend(backend, cspacings),
                        hmin, frozen)
end

ball_layout(::Nothing) = nothing
ball_layout(g::BallGeometry) = g.layout

"""
    ball_case(T = Float64, background; layout, ε_KO, γ0, γ2)

A [`GHCase`](@ref) of `background` on the ball `layout`: its box is the ball's
bounding cube, no face periodic or reflecting, so the outer sphere takes the
hook ([`ball_dirichlet`](@ref)). No hole (step B5).
"""
function ball_case(::Type{T}, background; layout, ε_KO, γ0, γ2) where {T}
    R = T(TreeMultiPatch.outer_radius(layout))
    return GHCase(T, background; box=ntuple(_ -> (-R, R), Val(3)),
                  periodic=(false, false, false), ε_KO=ε_KO, γ0=γ0, γ2=γ2)
end

"""
    ball_problem(T = Float64, case, layout; N, q, frozen = false, simd_width = nothing,
                 comm = nothing) -> GHProblem

The problem of `case` on TreeMultiPatch's `layout`, every patch at its roots, `N`
points a block: the forest (`layout_forest`), the state's field set interpolated
across every glue (`glue = :interpolate`) with `G = q/2 + 1`, and its schedule
with the atlas, prolongation and restriction at `q + 2` and the interpolation
across a glue at `q + 2` — exact to degree `q + 1`, as a second derivative of order
`q` needs (`CODE.md`, "The cube and the shells").
"""
function ball_problem(::Type{T}, case::GHCase{T}, layout; N, q, frozen::Bool=false,
                      simd_width=nothing, comm=nothing) where {T}
    forest = TreeMultiPatch.layout_forest(T, layout; N=N, comm=comm)
    U = FieldSet{T}(forest, 2NC; G=q ÷ 2 + 1, centering=vertexcentered(3),
                    glue=:interpolate)
    ops = Operators(prolongation=q + 2, restriction=q + 2, glue=q + 2)
    schedule = GhostSchedule(U, ops; atlas=layout)
    return GHProblem(U, schedule, case; q=q, layout=layout, frozen=frozen,
                     simd_width=simd_width)
end

# --- the map's chain rule ------------------------------------------------------
#
# With `K^a_i = ∂ξ^a/∂x^i` and the map's Hessian `H^k_bc = ∂²x^k/∂ξ^b∂ξ^c`,
#
#     ∂_i u = K^a_i ∂_a u,
#     ∂_i∂_j u = K^a_i K^b_j ∂_a∂_b u + (∂_i K^a_j) ∂_a u,
#     ∂_i K^a_j = −K^a_k H^k_bc K^b_i K^c_j .
#
# The Π components contract `∂_i∂_j h` with `A^{ij} = α√γ γ^{ij}`, `∂_i Π` with
# `β^i` and `∂_j h` with `∂_i A^{ij}`, so the map goes into the coefficients once
# per point, not into the twenty variables' derivatives:
#
#     Ã^{ab} = K^a_i A^{ij} K^b_j,   β̃^a = K^a_i β^i,
#     Ã^a = K^a_j ∂_i A^{ij} − K^a_k H^k_bc Ã^{bc},
#
# and a component is `β̃^a ∂_aΠ + (∂_iβ^i) Π + Ã^a ∂_a h + Ã^{ab} ∂_a∂_b h`, the
# cube's expression in logical derivatives. Everything here is generic in the
# number type: on the CPU's lanes `T` is a `Vec`.

# `K[a][i] = ∂ξ^a/∂x^i` from `J[i][a] = ∂x^i/∂ξ^a`, the adjugate over the
# determinant.
@inline function _inv3(J)
    j11, j12, j13 = J[1]
    j21, j22, j23 = J[2]
    j31, j32, j33 = J[3]
    c11 = j22 * j33 - j23 * j32
    c12 = j23 * j31 - j21 * j33
    c13 = j21 * j32 - j22 * j31
    c21 = j13 * j32 - j12 * j33
    c22 = j11 * j33 - j13 * j31
    c23 = j12 * j31 - j11 * j32
    c31 = j12 * j23 - j13 * j22
    c32 = j13 * j21 - j11 * j23
    c33 = j11 * j22 - j12 * j21
    id = one(j11) / (j11 * c11 + j12 * c12 + j13 * c13)
    return ((c11 * id, c21 * id, c31 * id), (c12 * id, c22 * id, c32 * id),
            (c13 * id, c23 * id, c33 * id))
end

"""
    shell_geometry(layout, ξ, p) -> (K, H)

The inverse Jacobian `K[a][i] = ∂ξ^a/∂x^i` and the Hessian `H[k][b][c] =
∂²x^k/∂ξ^b∂ξ^c` of patch `p`'s map at logical `ξ`, from TreeAMR's
`chart_jacobian` and `chart_hessian`.
"""
@inline function shell_geometry(layout, ξ, p)
    J = TreeAMR.chart_jacobian(layout, ξ, p)
    return _inv3(J), TreeAMR.chart_hessian(layout, ξ, p)
end

@inline _norm3(k) = sqrt(k[1] * k[1] + k[2] * k[2] + k[3] * k[3])

# `Ã^{ac} = (KA)^a_j K^c_j`.
@inline _kak(KA, K, a, c) = KA[a][1] * K[c][1] + KA[a][2] * K[c][2] + KA[a][3] * K[c][3]

"""
    gh_shell_head(T, work, Hwork, inner, b, x, t, var, st, sv, inv_h, γ0, γ2, εh, K,
                  H, ::Val{q}, ::Val{HASH}, ::Val{DISS}) -> (; ∂ₜh, msrc, β̃, divβ, Ã1, Ã)

[`gh_rhs_head`](@ref) on a shell: the same streaming order, with the stencils
along the logical axes, `inv_h` the logical spacing's inverse, `εh` the
dissipation's amplitude per logical axis, and the chain rule of `K` and `H` (see
the comment above). Returns the Π components' coefficients in logical
form: `β̃`, `∂_iβ^i`, `Ã^a` and `Ã^{ab}` (as `(11, 12, 13, 22, 23, 33)`).
"""
@inline function gh_shell_head(::Type{T}, work, Hwork, inner, b::Int, x, t, var::Int,
                               st, sv::Int, inv_h, γ0, γ2, εh, K, H, ::Val{q},
                               ::Val{HASH}, ::Val{DISS}, mat=nothing,
                               ::Val{HASMATTER}=Val(false)) where {T,q,HASH,DISS,
                                                                  HASMATTER}
    w1 = derivative_weights(T, Val(q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(q)))

    hv = SVector{NC,T}(@ntuple 10 v -> (@inbounds work[var + (v - 1) * sv]))
    Πv = SVector{NC,T}(@ntuple 10 v -> (@inbounds work[var + (NC + v - 1) * sv]))
    # The logical first derivatives of `h`, and the physical ones from them.
    Dh = @ntuple 3 a -> inv_h * SVector{NC,T}(@ntuple 10 v ->
        axis_stencil(w1, work, var + (v - 1) * sv, st[a]))
    ∂h = @ntuple 3 i -> (_scale(K[1][i], Dh[1]) + _scale(K[2][i], Dh[2])) +
                        _scale(K[3][i], Dh[3])
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    β̃ = @ntuple 3 a -> K[a][1] * β[1] + K[a][2] * β[2] + K[a][3] * β[3]

    ∂ₜh = SVector{NC,T}(@ntuple 10 v ->
        _shell_dth(work, var + (v - 1) * sv, st, εh, β̃, a_div, Dh[1][v], Dh[2][v],
                   Dh[3][v], Πv[v], wD, Val(DISS)))

    Hl, dHl = gauge_source(T, Hwork, inner, b, Val(HASH), x, t, hv, ∂ₜh, ∂h)
    msrc = gh_node_source_lean(g4, gu4, α, sqrtγ, ∂ₜh, ∂h, Hl, dHl, γ0, γ2)
    if HASMATTER
        m = matter_at(T, mat, inner, b)
        msrc = msrc + matter_source(g4, α, β, γu, sqrtγ, m.ρ, m.S, m.Sij)
    end

    A = _scale(α * sqrtγ, γu)
    divβ, divA = metric_divergences(gu4, α, β, γu, sqrtγ, ∂h)
    # `Ã^{ab} = K^a_i A^{ij} K^b_j`, through `KA[a][j] = K^a_i A^{ij}`.
    KA = @ntuple 3 a -> (@ntuple 3 j -> K[a][1] * A[1, j] + K[a][2] * A[2, j] +
                                        K[a][3] * A[3, j])
    Ã = (_kak(KA, K, 1, 1), _kak(KA, K, 1, 2), _kak(KA, K, 1, 3), _kak(KA, K, 2, 2),
         _kak(KA, K, 2, 3), _kak(KA, K, 3, 3))
    # `H^k_bc Ã^{bc}` for each `k`, then `Ã^a = K^a_j ∂_i A^{ij} − K^a_k (H Ã)^k`.
    HÃ = @ntuple 3 k -> (H[k][1][1] * Ã[1] + H[k][2][2] * Ã[4] + H[k][3][3] * Ã[6]) +
                        2 * ((H[k][1][2] * Ã[2] + H[k][1][3] * Ã[3]) + H[k][2][3] * Ã[5])
    Ã1 = @ntuple 3 a -> (K[a][1] * divA[1] + K[a][2] * divA[2] + K[a][3] * divA[3]) -
                        (K[a][1] * HÃ[1] + K[a][2] * HÃ[2] + K[a][3] * HÃ[3])
    return (; ∂ₜh, msrc, β̃, divβ, Ã1, Ã)
end

# One component of `∂ₜh` on a shell: `β̃^a ∂_a h + (α/√γ) Π`, then the
# dissipation along each logical axis with that axis's amplitude.
@inline function _shell_dth(work, bh, st, εh, β̃, a_div, D1, D2, D3, Π_v, wD,
                            ::Val{DISS}) where {DISS}
    s = β̃[1] * D1 + β̃[2] * D2 + β̃[3] * D3 + a_div * Π_v
    DISS || return s
    return s + (εh[1] * axis_stencil(wD, work, bh, st[1]) +
                εh[2] * axis_stencil(wD, work, bh, st[2]) +
                εh[3] * axis_stencil(wD, work, bh, st[3]))
end

"""
    gh_shell_pi(T, work, var, st, sv, v, inv_h, εh, head, ::Val{q}, ::Val{DISS})

Component `v` of `∂ₜΠ` on a shell without its source: [`gh_rhs_pi`](@ref)'s
nine stencils of `h_v` and seven of `Π_v` along the logical axes, contracted with
[`gh_shell_head`](@ref)'s logical coefficients, and the dissipation per axis.
"""
@inline function gh_shell_pi(::Type{T}, work, var::Int, st, sv::Int, v::Int, inv_h, εh,
                             head, ::Val{q}, ::Val{DISS}) where {T,q,DISS}
    inv_h² = inv_h * inv_h
    w1 = derivative_weights(T, Val(q), Val(1))
    w2 = derivative_weights(T, Val(q), Val(2))
    wD = dissipation_weights(T, dissipation_rank(Val(q)))
    β̃, divβ, Ã1, Ã = head.β̃, head.divβ, head.Ã1, head.Ã
    bh = var + (v - 1) * sv
    bΠ = bh + NC * sv
    Π_v = @inbounds work[bΠ]
    D1 = inv_h * axis_stencil(w1, work, bh, st[1])
    D2 = inv_h * axis_stencil(w1, work, bh, st[2])
    D3 = inv_h * axis_stencil(w1, work, bh, st[3])
    E1 = inv_h * axis_stencil(w1, work, bΠ, st[1])
    E2 = inv_h * axis_stencil(w1, work, bΠ, st[2])
    E3 = inv_h * axis_stencil(w1, work, bΠ, st[3])
    s = β̃[1] * E1 + β̃[2] * E2 + β̃[3] * E3 + divβ * Π_v +
        Ã1[1] * D1 + Ã1[2] * D2 + Ã1[3] * D3
    s += Ã[1] * (inv_h² * axis_stencil(w2, work, bh, st[1])) +
         Ã[4] * (inv_h² * axis_stencil(w2, work, bh, st[2])) +
         Ã[6] * (inv_h² * axis_stencil(w2, work, bh, st[3]))
    D12 = inv_h² * mixed_stencil(w1, work, bh, st[1], st[2])
    D13 = inv_h² * mixed_stencil(w1, work, bh, st[1], st[3])
    D23 = inv_h² * mixed_stencil(w1, work, bh, st[2], st[3])
    s += 2 * (Ã[2] * D12 + Ã[3] * D13 + Ã[5] * D23)
    if DISS
        s += εh[1] * axis_stencil(wD, work, bΠ, st[1]) +
             εh[2] * axis_stencil(wD, work, bΠ, st[2]) +
             εh[3] * axis_stencil(wD, work, bΠ, st[3])
    end
    return s
end

# `F(u)` at one point of a shell, stored as each component is finished, as
# [`gh_rhs_store!`](@ref) does on the cube.
@inline function gh_shell_store!(du, o::Int, sd::Int, ::Type{T}, work, Hwork, inner,
                                 b::Int, x, t, var::Int, st, sv::Int, inv_h, γ0, γ2, εh,
                                 K, H, ::Val{q}, ::Val{HASH}, ::Val{DISS}, mat=nothing,
                                 ::Val{HASMATTER}=Val(false)) where {T,q,HASH,DISS,
                                                                    HASMATTER}
    head = gh_shell_head(T, work, Hwork, inner, b, x, t, var, st, sv, inv_h, γ0, γ2,
                         εh, K, H, Val(q), Val(HASH), Val(DISS), mat, Val(HASMATTER))
    ∂ₜh = head.∂ₜh
    @nexprs 10 v -> (@inbounds du[o + (v - 1) * sd] = ∂ₜh[v])
    msrc = head.msrc
    for v in 1:NC
        @inbounds du[o + (NC + v - 1) * sd] =
            gh_shell_pi(T, work, var, st, sv, v, inv_h, εh, head, Val(q), Val(DISS)) +
            msrc[v]
    end
    return nothing
end

"""
    gh_shell_point!(du, work, Hwork, origins, spacings, patches, layout, damping, γ2,
                    ε_KO, t, I, ::Val{G}, ::Val{q}, ::Val{HASH}, ::Val{DISS})

The right-hand side at one point `I = (i1, i2, i3, block)` of a shell's block:
its logical position from TreeAMR's origins and spacings, its physical one
through the atlas, the map's `K` and `H` there, and [`gh_shell_store!`](@ref).
The dissipation's amplitude along logical axis `a` is `ε_KO |∇ξ^a| / h`, the
cube's `ε_KO/h` with the physical spacing along that axis, `h/|∇ξ^a|`.
"""
@inline function gh_shell_point!(du, work, Hwork, origins, spacings, patches, layout,
                                 damping, γ2, ε_KO, t, I, ::Val{G}, ::Val{q},
                                 ::Val{HASH}, ::Val{DISS}, mat=nothing,
                                 ::Val{HASMATTER}=Val(false)) where {G,q,HASH,DISS,
                                                                    HASMATTER}
    b = I[4]
    inner = (I[1], I[2], I[3])
    T = eltype(du)
    inv_h = inv(spacings[b])
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb +
          (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]
    ξ = point_position(origins, spacings, b, I)
    p = Int(patches[b])
    x = TreeAMR.chart_map(layout, ξ, p)
    K, H = shell_geometry(layout, ξ, p)
    γ0 = damping_rate(damping, t, x)
    εi = dissipation_rate(ε_KO, t, x) * inv_h
    εh = (εi * _norm3(K[1]), εi * _norm3(K[2]), εi * _norm3(K[3]))
    o, sd = state_offset(du, I)
    gh_shell_store!(du, o, sd, T, work, Hwork, inner, b, x, t, var, st, sv, inv_h, γ0,
                    γ2, εh, K, H, Val(q), Val(HASH), Val(DISS), mat, Val(HASMATTER))
    return nothing
end

# `W` points of a shell's row on SIMD lanes: the map evaluated per lane on the
# host's scalars and packed, then the chain rule and the stencils on the lanes.
@inline function gh_shell_lanes!(du, work, Hwork, origins, spacings, patches, layout,
                                 damping, γ2, ε_KO, t, I, ::Val{W}, ::Val{G}, ::Val{q},
                                 ::Val{HASH}, ::Val{DISS}, mat=nothing,
                                 ::Val{HASMATTER}=Val(false)) where {W,G,q,HASH,DISS,
                                                                    HASMATTER}
    b = I[4]
    inner = (I[1], I[2], I[3])
    T = eltype(du)
    V = Vec{W,T}
    inv_h = inv(spacings[b])
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb +
          (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]
    p = Int(patches[b])
    ξs = ntuple(l -> point_position(origins, spacings, b, (I[1] + l - 1, I[2], I[3])),
                Val(W))
    xs = ntuple(l -> TreeAMR.chart_map(layout, ξs[l], p), Val(W))
    geo = ntuple(l -> shell_geometry(layout, ξs[l], p), Val(W))
    K = ntuple(a -> ntuple(i -> V(ntuple(l -> T(geo[l][1][a][i]), Val(W))), Val(3)),
               Val(3))
    H = ntuple(k -> ntuple(c -> ntuple(e -> V(ntuple(l -> T(geo[l][2][k][c][e]), Val(W))),
                                       Val(3)), Val(3)), Val(3))
    γ0 = V(ntuple(l -> damping_rate(damping, t, xs[l]), Val(W)))
    εi = V(ntuple(l -> dissipation_rate(ε_KO, t, xs[l]), Val(W))) * inv_h
    εh = (εi * _norm3(K[1]), εi * _norm3(K[2]), εi * _norm3(K[3]))
    xv = (V(ntuple(l -> xs[l][1], Val(W))), V(ntuple(l -> xs[l][2], Val(W))),
          V(ntuple(l -> xs[l][3], Val(W))))
    wl = lanes(Val(W), work)
    Hl = lanes(Val(W), Hwork)
    dl = lanes(Val(W), du)
    o, sd = state_offset(du, I)
    gh_shell_store!(dl, o, sd, V, wl, Hl, inner, b, xv, t, var, st, sv, inv_h, γ0,
                    V(γ2), εh, K, H, Val(q), Val(HASH), Val(DISS), mat, Val(HASMATTER))
    return nothing
end

# --- the appendix ---------------------------------------------------------------

# The state's `AppendixArray` addressed by the linear index of an array of its
# size, which is how the cube's and the shells' point functions store
# (`state_offset`): the closed range's `(N + 1)³ × nvars × nblocks`. A write to a
# point the block does not own is discarded, as the `AppendixArray`'s is.
struct AppendixLinear{A}
    a::A
end

Adapt.adapt_structure(to, w::AppendixLinear) = AppendixLinear(Adapt.adapt(to, w.a))
@inline Base.size(w::AppendixLinear, d::Integer) = size(w.a)[d]
@inline Base.eltype(w::AppendixLinear) = eltype(w.a)
@inline function Base.setindex!(w::AppendixLinear, x, k::Int)
    n1, n2, n3, nv = size(w.a)
    k0 = k - 1
    i1 = k0 % n1
    k0 = k0 ÷ n1
    i2 = k0 % n2
    k0 = k0 ÷ n2
    i3 = k0 % n3
    k0 = k0 ÷ n3
    v = k0 % nv
    b = k0 ÷ nv
    @inbounds w.a[i1 + 1, i2 + 1, i3 + 1, v + 1, b + 1] = x
    return w
end

# --- the kernel ---------------------------------------------------------------------

"""
    gh_ball_rhs_kernel!(du, da, work, Hwork, origins, spacings, corigins, cspacings,
                        patches, layout, bg, damping, γ2, ε_KO, interior, t, tw, t_f,
                        rate, trail, fitp, ::Val{G}, ::Val{q}, ::Val{HASH}, ::Val{DISS},
                        ::Val{INT}, ::Val{W}, mat, ::Val{HASMATTER})

The right-hand side on the ball, one launch over the closed range `(N + 1)³` of
every block (`map_blocks!(…; appendix = true)`). The branch is per block, on its
patch:

- **the cube** runs [`gh_rhs_kernel!`](@ref)'s body — [`gh_rhs_point!`](@ref), or
  [`gh_rhs_lanes!`](@ref) on the CPU's lanes — with its physical origins and
  spacings `corigins`, `cspacings`, so that its cores are the brick kernel's bits
  (`test/ball_tests.jl` compares them);
- **a shell** runs [`gh_shell_point!`](@ref) or [`gh_shell_lanes!`](@ref).

`du` is the cores' array (`statearray(du, U)`), which the point functions store
into by a linear index; `da` is the whole state's `AppendixArray`, through which
an owned point of the appendix is stored, one point at a time.
"""
@kernel function gh_ball_rhs_kernel!(du, da, @Const(work), Hwork, @Const(origins),
                                     @Const(spacings), @Const(corigins),
                                     @Const(cspacings), @Const(patches), layout, bg,
                                     damping, γ2, ε_KO, interior, t, tw, t_f, rate,
                                     trail, fitp, ::Val{G}, ::Val{q}, ::Val{HASH},
                                     ::Val{DISS}, ::Val{INT}, ::Val{W}, mat,
                                     ::Val{HASMATTER}) where {G,q,HASH,DISS,INT,W,
                                                              HASMATTER}
    I = @index(Global, NTuple)                    # (i1, i2, i3, block), closed
    N = size(du, 1)
    b = I[4]
    cube = patches[b] == 1
    if I[1] ≤ N && I[2] ≤ N && I[3] ≤ N
        if W == 1
            if cube
                gh_rhs_point!(du, work, Hwork, corigins, cspacings, bg, damping, γ2,
                              ε_KO, interior, t, tw, t_f, rate, trail, fitp, I, Val(G),
                              Val(q), Val(HASH), Val(DISS), Val(INT), mat,
                              Val(HASMATTER))
            else
                gh_shell_point!(du, work, Hwork, origins, spacings, patches, layout,
                                damping, γ2, ε_KO, t, I, Val(G), Val(q), Val(HASH),
                                Val(DISS), mat, Val(HASMATTER))
            end
        elseif is_lane_leader(I[1], N, W)
            if cube
                gh_rhs_lanes!(du, work, Hwork, corigins, cspacings, bg, damping, γ2,
                              ε_KO, interior, t, tw, t_f, rate, trail, fitp, I, Val(W),
                              Val(G), Val(q), Val(HASH), Val(DISS), Val(INT), mat,
                              Val(HASMATTER))
            else
                gh_shell_lanes!(du, work, Hwork, origins, spacings, patches, layout,
                                damping, γ2, ε_KO, t, I, Val(W), Val(G), Val(q),
                                Val(HASH), Val(DISS), mat, Val(HASMATTER))
            end
        end
    elseif isowned(da, I[1], I[2], I[3], 1, b)
        dl = AppendixLinear(da)
        if cube
            gh_rhs_point!(dl, work, Hwork, corigins, cspacings, bg, damping, γ2, ε_KO,
                          interior, t, tw, t_f, rate, trail, fitp, I, Val(G), Val(q),
                          Val(HASH), Val(DISS), Val(INT), mat, Val(HASMATTER))
        else
            gh_shell_point!(dl, work, Hwork, origins, spacings, patches, layout,
                            damping, γ2, ε_KO, t, I, Val(G), Val(q), Val(HASH),
                            Val(DISS), mat, Val(HASMATTER))
        end
    end
end

# The ball's launch of the right-hand side: `rhs!` on a ball.
function ball_rhs!(du, p, t, at)
    g = p.ball
    args = gh_rhs_kernel_args(p, du, t, at)
    # `gh_rhs_kernel_args` begins with the cores' array, the working array, the
    # gauge source, the origins and the spacings; the ball's kernel takes the
    # appendix after the cores and its own geometry after TreeAMR's.
    rest = args[6:end]
    map_blocks!(gh_ball_rhs_kernel!, p.U, args[1],
                statearray(du, p.U; appendix=true), args[2], args[3], args[4],
                args[5], g.corigins, g.cspacings, g.patches, g.layout, rest...;
                appendix=true)
    return nothing
end

# --- the hook, the data and the norms -------------------------------------------

"""
    ball_dirichlet(g::BallGeometry, case, t) -> CellBoundary

The outer sphere's hook: the case's background at each ghost's physical
position, through the atlas (TreeWave's `PlaneWaveHook`), at time `t`, or at
`t = 0` for a frozen sphere (`g.frozen`; `CODE.md`, "The outer boundary and the
wave zone", option 1).
"""
function ball_dirichlet(g::BallGeometry, case::GHCase{T}, t) where {T}
    bg = case.background
    layout = g.layout
    tt = g.frozen ? zero(T) : T(t)
    return CellBoundary(WithPatch(AllVariables((ξ, q, δ) ->
        state_tuple(bg, tt, TreeAMR.chart_map(layout, ξ, q)))))
end

# The physical-boundary hook of a problem at `t`: the brick's or the ball's.
gh_boundary(p, t) = _gh_boundary(p.ball, p.case, t)
_gh_boundary(::Nothing, case, t) = dirichlet(case, t)
_gh_boundary(g::BallGeometry, case, t) = ball_dirichlet(g, case, t)

"""
    gh_ball_error_kernel!(diag, work, origins, spacings, patches, layout, bg, t, ::Val{G})

`|u − u_exact|` over the twenty components at every point of the closed range,
the exact state at the point's physical position through the atlas, into
`DIAG_ERR`, and `1` into `DIAG_MASK`. Launched with `appendix = true`; a point a
block does not own is written and never read, since the fold counts owned
points only.
"""
@kernel function gh_ball_error_kernel!(diag, @Const(work), @Const(origins),
                                       @Const(spacings), @Const(patches), layout, bg, t,
                                       ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    T = eltype(diag)
    c = (I[1] + G[1], I[2] + G[2], I[3] + G[3])
    ξ = point_position(origins, spacings, b, I)
    x = TreeAMR.chart_map(layout, ξ, Int(patches[b]))
    ex = state_tuple(bg, t, x)
    s = zero(T)
    for v in 1:(2NC)
        d = work[c[1], c[2], c[3], v, b] - ex[v]
        s += d * d
    end
    diag[I[1], I[2], I[3], DIAG_ERR, b] = sqrt(s)
    diag[I[1], I[2], I[3], DIAG_MASK, b] = one(T)
end

"""
    ball_error!(p::GHProblem, u, t) -> p

The error of the state `u` at time `t` against the case's background on the
ball, into `p.diag` ([`gh_ball_error_kernel!`](@ref)). [`ball_norms`](@ref) reads
it.
"""
function ball_error!(p::GHProblem{T}, u, t) where {T}
    g = p.ball
    g === nothing && throw(ArgumentError(
        "ball_error! is the error on a forest of patches, and this problem's is a " *
        "brick: use gh_error!"))
    scatter!(p.U, u)
    map_blocks!(gh_ball_error_kernel!, p.U, p.diag.work, p.U.work, p.origins,
                p.spacings, g.patches, g.layout, p.case.background, T(t), p.valG;
                appendix=true)
    return p
end

# The `√g`-weighted fold of one `diag` slot over the ball: `p.x` is the point's
# logical position and `p.tag` its patch, so the weight is the cell's physical
# volume `√g h³`.
struct BallSlotNorm{L}
    layout::L
    lo::Int
    hi::Int
end

@inline function (f::BallSlotNorm)(acc, p)
    q = Int(p.tag)
    keep = f.lo ≤ q ≤ f.hi
    z = zero(p.h)
    w = keep ? TreeMultiPatch.volume_element(f.layout, p.x, q) * p.h^3 : z
    u = keep ? p.u[1] : z
    return TreeAMR.combine(acc, (vol=Sum(w), s2=Sum(u * u * w), mx=Max(abs(u))))
end

"""
    ball_norms(p::GHProblem, slot; patches = 1:13) -> (l2, linf, volume)

The L2 norm of `diag` slot `slot` over the ball, each point weighted by its
physical volume `√g h³` (TreeMultiPatch's `volume_element`) and counted once
(TreeAMR's fold over a set with an appendix), divided by the ball's volume; and
its L∞. `patches`, a range, restricts both to those patches: `1:1` the cube,
`2:7` the inner shells, `8:13` the outer ones.
"""
function ball_norms(p::GHProblem{T}, slot::Integer; patches::UnitRange=1:13) where {T}
    g = p.ball
    forest = p.diag.forest
    init = (vol=Sum(zero(T)), s2=Sum(zero(T)), mx=Max(zero(T)))
    r = mesh_fold(BallSlotNorm(g.layout, first(patches), last(patches)),
                  TreeAMR.combine, init, p.diag;
                  vars=Int(slot):Int(slot), tag=k -> Int32(patch(forest, k)))
    return (l2=sqrt(r.s2.value / r.vol.value), linf=r.mx.value, volume=r.vol.value)
end

"""
    ball_state_callback(case::GHCase, layout, t) -> WithPatch

The case's exact state at time `t` on the patches of `layout`, as the
`WithPatch` callback `fill_by_coordinates!` takes: each point's logical `ξ` in
patch `q` mapped through the atlas. No interior: a hole on the ball needs
TreeAMR's M19.
"""
function ball_state_callback(case::GHCase{T}, layout, t) where {T}
    bg = case.background
    tt = T(t)
    return WithPatch(AllVariables((ξ, q) -> state_tuple(bg, tt,
                                                         TreeAMR.chart_map(layout, ξ, q))))
end

# --- refinement ---------------------------------------------------------------------

"""
    glue_frozen(forest, k) -> Bool

Whether `refine_flags` must leave leaf `k` of a forest of patches at its level:
every block of a shell, and every block of the cube within one of its own widths
of the cube's faces, which are glued. TreeAMR's `regrid!` keeps every block
within reach of a glue at its level and refuses a change 2:1 balance would carry
into one ("No refinement boundary at a glue"); this keeps the flags from asking.
`false` for every leaf of a brick.
"""
function glue_frozen(forest, k)
    patch(forest, k) == 1 || return true
    roots = forest.roots
    ext = block_extent(Float64, forest, k)
    return any(d -> ext[d][1] - (ext[d][2] - ext[d][1]) ≤ 0 ||
                    ext[d][2] + (ext[d][2] - ext[d][1]) ≥ roots[d], 1:3)
end

# The flags of `refine_flags` with every glue-frozen leaf of a forest of patches
# kept, bare; a brick's flags as they are. `flags` is this rank's, one per local
# block.
function freeze_glue_flags(forest, flags)
    any(k -> patch(forest, k) != 1, forest.leaves) || return flags
    range = blockrange(forest)
    return [glue_frozen(forest, forest.leaves[range[b]]) ? Keep : flags[b]
            for b in eachindex(flags)]
end
