# Public access to the vertex metric in ADM variables (added 2026-11-05).
#
# A downstream code that couples to this package's metric — TreeGRRMHD's
# general-relativistic MHD, its GH-1 — needs `α`, `β^i`, `γ_ij` and `K_ij` at the
# vertices of the GH field set. `adm_vars` decodes them from the packed `(h, Π)`
# where they are stored, so that the caller needs neither the packing nor the
# variable layout of `U.work` nor the stencils (`CODE.md`, "Public access to the
# vertex metric").

"""
    adm_vars(work, I, b) -> (; α, β, γ, sqrtγ, γu)
    adm_vars(work, I, b, inv_h, Val(q)) -> (; α, β, γ, sqrtγ, γu, K)

The metric at one stored vertex of a GH field set in ADM variables: the lapse
`α`, the contravariant shift `β^i` (`SVector{3}`), the spatial metric `γ_ij`, its
determinant's root `sqrtγ = √det γ` and its inverse `γu = γ^{ij}` (`SMatrix{3,3}`),
and with the second method the extrinsic curvature `K_ij` (`SMatrix{3,3}`, the
NR sign, `K_ij = −(∂ₜγ_ij − D_iβ_j − D_jβ_i)/(2α)`, as `SpacetimeMetrics`'
`ExtrinsicCurvature`).

- `work` is the field set's working array `(n, n, n, nvars, nblocks)` — `U.work`,
  scattered and ghost-filled by the caller. Only variables `1:20` are read.
- `I` is the **stored** index `(i, j, k)`, 1-based, ghosts included — what
  `coordinates(fs, b, I)` takes — and `b` the block.
- `inv_h` is the block's `1/h`, and `q` the order the right-hand side
  differentiates at.

`α`, `β`, `sqrtγ` and `γu` are [`metric_quantities`](@ref)'s, the offset-accurate
path the right-hand side takes, and `γ` is `η + h`'s spatial block. `K` is
[`adm_vars_from_state`](@ref)'s, from `Π` and from `∂_i h` formed with the
right-hand side's own first-derivative stencil, in its summation order.

**Where it is valid.** The first method reads one point and is valid at every
stored vertex. The second reads `q/2` points along each axis on each side, so it
is valid where `q/2 < I[d] ≤ n − q/2`: the owned vertices and, since `G = q/2 + 1`,
one layer of ghosts. A `@boundscheck` refuses any other `I` with the
`BoundsError` of the farthest point the stencil would read; `@inbounds` at the call
site elides it. Inside a black hole's core the stored data are stale by design
(`CODE.md`, "The interior") and the result means nothing there.

Both methods are `@inline`, allocate nothing, form no closure and return `isbits`
named tuples, so they can be called from a KernelAbstractions kernel on any
backend. Destructure by name, `(; α, β, γ) = adm_vars(…)`: fields may be added.
"""
@inline function adm_vars(work::AbstractArray{T,5}, I::NTuple{3,Int},
                          b::Int) where {T}
    @boundscheck checkbounds(work, I[1], I[2], I[3], NC, b)
    var, sv = _adm_base(work, I, b)
    hv = SVector{NC,T}(@ntuple 10 v -> (@inbounds work[var + (v - 1) * sv]))
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    return (; α, β, γ=_spatial3(g4), sqrtγ, γu)
end

@inline function adm_vars(work::AbstractArray{T,5}, I::NTuple{3,Int}, b::Int,
                          inv_h, ::Val{q}) where {T,q}
    r = q ÷ 2
    @boundscheck begin
        checkbounds(work, I[1], I[2], I[3], 2NC, b)
        ok = (r < I[1] <= size(work, 1) - r) & (r < I[2] <= size(work, 2) - r) &
             (r < I[3] <= size(work, 3) - r)
        ok || _stencil_out_of_range(work, I, b, r)
    end
    st, sv, _ = work_strides(work)
    var, _ = _adm_base(work, I, b)
    w1 = derivative_weights(T, Val(q), Val(1))
    # The state and its gradients, spelled as `gh_rhs_head` spells them.
    hv = SVector{NC,T}(@ntuple 10 v -> (@inbounds work[var + (v - 1) * sv]))
    Πv = SVector{NC,T}(@ntuple 10 v -> (@inbounds work[var + (NC + v - 1) * sv]))
    ∂h = @ntuple 3 d -> inv_h * SVector{NC,T}(@ntuple 10 v ->
        axis_stencil(w1, work, var + (v - 1) * sv, st[d]))
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    K = _extrinsic_curvature(g4, α, β, γu, sqrtγ, Πv, ∂h[1], ∂h[2], ∂h[3])
    return (; α, β, γ=_spatial3(g4), sqrtγ, γu, K)
end

# The linear index of `(I, b)`'s first variable and the stride between
# variables: `gh_rhs_kernel!`'s, for a stored index (no `+ G`).
@inline function _adm_base(work, I, b)
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb + (I[1] - 1) * st[1] + (I[2] - 1) * st[2] +
          (I[3] - 1) * st[3]
    return var, sv
end

# The point the stencil would read outside the stored range, reported as
# the `BoundsError` an unguarded read would throw — integers only, so that a
# device compiles it.
@noinline function _stencil_out_of_range(work, I, b, r)
    Base.throw_boundserror(work, (_reach(I[1], r, size(work, 1)),
                                  _reach(I[2], r, size(work, 2)),
                                  _reach(I[3], r, size(work, 3)), 1, b))
end
@inline _reach(i, r, n) = i - r < 1 ? i - r : i + r > n ? i + r : i
