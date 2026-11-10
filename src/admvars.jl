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
- `inv_h` is the block's `1/h`, of the array's element type `T` (a `Float64`
  spacing with a `Float32` array would make `∂h` and `K` mixed-precision, so it
  is refused), and `q` the order the right-hand side differentiates at.

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
                          inv_h::T, ::Val{q}) where {T,q}
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

# The metric adapter of the coupling contract (added 2026-11-08): `adm_vars`
# behind an object that holds the working array, the block spacings and `q`, so
# that the consumer needs none of them (`CODE.md`, "The coupling contract with
# TreeGRRMHD").

"""
    ADMMetric(p::GHProblem, u, t)
    ADMMetric(p::GHProblem)

The metric adapter of the coupling contract with TreeGRRMHD (GH-1): this
package's metric at its own vertices, as `α`, `β^i`, `γ_ij` and `K_ij`
(`CODE.md`, "The coupling contract with TreeGRRMHD").

The first method scatters `u` into the problem's field set and fills its ghosts
at `t`, as [`gh_rhs!`](@ref) does, and then wraps the result. The second wraps
the field set as it is, for a caller that has just filled it at the time it
wants. Both are a **snapshot**: the adapter reads the field set's working array
in place, so it is valid until this package's state changes. Rebuild it after a
regrid, which replaces the field set.

`ADMMetric` is the host side and is not passed to a kernel. Binding it to a
field set of the consumer's,

```julia
at = ADMMetric(p, u, t)(M)
(; α, β, γ, K) = at(n, b)
```

checks that `M` is vertex-centered and on this problem's forest (`===`), and
returns the kernel-side object. That object is `isbits` once adapted to the
backend (`Adapt.adapt`, which a KernelAbstractions launch applies). `n` is the
**owned** index, `1 ≤ n[d] ≤ N`, the index `map_blocks!` hands a kernel over
the default range, and `b` the local block. The call returns
[`adm_vars`](@ref)'s second method at the stored index `n + G`, with this
package's `1/h` and `q`. An owned index means the same point in every
vertex-centered set on the forest, so `M`'s own `G` does not enter.

The contract promises the owned vertices only. `K`'s stencil reaches `q/2`
points, so with `G = q/2 + 1` it would be defined one ghost layer further, but
the consumer fills its own ghosts. A `@boundscheck` refuses any other `n` with
the `BoundsError` of its stored index; `@inbounds` at the call site elides it.
"""
struct ADMMetric{F,A,S,q}
    forest::F
    work::A
    spacings::S
    valq::Val{q}
end

ADMMetric(p::GHProblem{T,G,q}) where {T,G,q} =
    ADMMetric(p.U.forest, p.U.work, p.spacings, Val(q))

ADMMetric(p::GHProblem, u, t) = ADMMetric(prepare!(p, u, t))

# The kernel side: everything a call reads, and nothing that is host-only.
struct BoundADMMetric{A,S,G,q}
    work::A
    spacings::S
    valG::Val{G}
    valq::Val{q}
end

Adapt.@adapt_structure BoundADMMetric

function (adm::ADMMetric{F,A,S,q})(M::FieldSet) where {F,A,S,q}
    M.forest === adm.forest || throw(ArgumentError(
        "the metric adapter is bound to a field set on another forest: the " *
        "owned index it is called with names a point only on this problem's " *
        "forest, so build the consumer's field set on it (`===`), and rebind " *
        "after every regrid (CODE.md, \"The coupling contract with " *
        "TreeGRRMHD\")"))
    all(c -> c === :vertex, M.centering) || throw(ArgumentError(
        "the metric adapter answers at this package's vertices, and this " *
        "field set is $(M.centering): the contract leaves every conversion " *
        "between centerings to the consumer, which interpolates from a " *
        "vertex-centered set of its own (CODE.md, \"The coupling contract " *
        "with TreeGRRMHD\")"))
    return BoundADMMetric(adm.work, adm.spacings, Val(q ÷ 2 + 1), adm.valq)
end

@inline function (at::BoundADMMetric{A,S,G,q})(n::NTuple{3,Int},
                                               b::Int) where {A,S,G,q}
    I = (n[1] + G, n[2] + G, n[3] + G)
    @boundscheck begin
        N = size(at.work, 1) - 2G - 1
        ok = (1 <= n[1] <= N) & (1 <= n[2] <= N) & (1 <= n[3] <= N)
        ok || _owned_out_of_range(at.work, I, b)
        checkbounds(at.spacings, b)
    end
    inv_h = inv(@inbounds at.spacings[b])
    return @inbounds adm_vars(at.work, I, b, inv_h, Val(q))
end

# An index outside the owned range, reported as the `BoundsError` of its stored
# index — integers only, so that a device compiles it.
@noinline _owned_out_of_range(work, I, b) =
    Base.throw_boundserror(work, (I[1], I[2], I[3], 1, b))
