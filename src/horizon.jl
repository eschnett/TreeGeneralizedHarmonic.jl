# The apparent horizon: where it is, how big it is, and how fast it spins.
#
# `CODE.md`, "Analysis quantities" (the horizon rows) and "Upstream
# prerequisites" (point interpolation). Three things live here, in the
# order the numbers are produced:
#
#   1. **Point interpolation from a field set** — `find_leaf` to locate
#      the containing block, then tensor-product Lagrange interpolation of
#      order `q + 2` over that block's *stored* points, batched over a host
#      array of query points. This is the one piece of mesh machinery this
#      package writes, and it is a **stopgap**: TreeAMR's `TODO.md` lists
#      "generic interpolation", and when it grows it this file loses
#      [`interpolate`](@ref) and keeps everything else (`CLAUDE.md`, "No
#      mesh machinery"; `CODE.md`, "Upstream prerequisites", item 1).
#   2. **The ADM provider** `ApparentHorizonFinder` consumes, in its
#      *batched* form — all surface points at once, so the interpolation
#      runs threaded — built out of `pointwise.jl`'s
#      [`adm_vars_from_state`](@ref): `γ_ij` and `∂_kγ_ij` from the
#      interpolated `h` and its interpolated gradient, `K_ij` from `Π`
#      through the evolution relation.
#   3. **The find itself**, [`find_gh_horizon`](@ref): GHSO2's composition
#      of the two libraries (`notes/methods-ghso2.md`, "Apparent horizons
#      and spin") — the fast flow for the location, shape and proper area,
#      `KorzynskiSpin.horizon_spin` on the same collocation grid for `J`
#      and its axis, and `M_irr = √(A/16π)`,
#      `M_ch = √(M_irr² + J²/(4M_irr²))` from those.
#
# Four things are easy to get wrong here, and each is written out where it
# happens:
#
#   * **The interpolation footprint must not reach `r_1`.** Inside the
#     layer the state is not a numerical solution and inside the core it is
#     stale by design, so an interpolant that reaches either would report a
#     horizon of data the equations never produced. The provider *throws*
#     (`CODE.md`: "the provider throws if a query point's interpolation
#     footprint reaches `r_1`"), and the check is exact rather than
#     conservative: the footprint is a tensor-product lattice, so the
#     nearest of its points to the hole's center is the per-axis nearest in
#     each direction.
#   * **Ghosts must be filled first.** The window of `q + 2` points around
#     a query near a block face reaches `G` points into the neighbour, and
#     `G = q/2 + 1 = (q + 2)/2` is exactly half the window — which is why a
#     point anywhere in the block can be interpolated *without crossing
#     into another block's array*, and why the ghosts have to be current.
#     [`find_gh_horizon`](@ref) fills them with this `t`'s hook.
#   * **The horizon finder is `Float64`.** `ApparentHorizonFinder`'s origin
#     and grid are `Float64` whatever the run computes in, as the analysis
#     record is (`driver.jl`). The interpolation runs in the field set's
#     own `T` and the result is converted once, at the provider's exit.
#   * **The finder is a diagnostic, not a tracker — in this file.** Nothing
#     here feeds back into the layer, the mesh or the gauge; `CODE.md` says a
#     run in which the found horizon and the analytic center disagree by
#     more than a few finest spacings has found a bug. From step 8d a case
#     may ask for its layer to follow the found horizon, and then the answer
#     is fed back — by `tracking.jl`, from the named tuple this file returns,
#     and nowhere here.

"""
    Horizon(T = Float64; every = 1, N = 16, r_seed = 0, spin = true,
            unif_tol = 1e-8, atol = 0, maxiters = 1000, verbosity = 0)

The horizon analysis a case asks for: how often to look, at what angular
resolution, and from which sphere to start.

`CODE.md`'s analysis table puts the horizon rows "every `k`-th chunk, `k` a
case parameter", so this is a parameter *of the case* — like
[`Refinement`](@ref) and for the same reason: a driver keyword would make
the cadence a property of the run rather than of the study, and two runs of
one case would not be comparable.

- `every` is that `k`: the horizon is found at chunk `0, k, 2k, …` of the
  record, and `every = 0` is "never", which is what a run with no hole
  wants.
- `N` is the finder's angular resolution — `EquiangularGrid(N − 1)`, so
  `N` collocation points in `θ` and `2N − 1` in `φ`, and `l ≤ N − 1`
  multipoles of the shape.
- `r_seed` is the radius of the *first* seed sphere; `0` means "take it
  from the background's analytic horizon radii", which is the mean of
  [`horizon_min_radius`](@ref) and [`horizon_max_radius`](@ref) and is the
  only sensible guess a case can make by itself. Later finds are seeded
  with the previous shape.
- `spin = false` skips the Korzyński spin, which is the second half of the
  cost of a find; it is computed on the finder's own collocation grid,
  which is what `CODE.md` asks for and what makes it free of a second
  interpolation.
- `unif_tol` is the tolerance of the spin's conformal uniformization.
  **The default is `1e-8` and not the library's `1e-13` (proposed in step
  7).** `1e-13` is the round-off floor of *analytic* Cauchy data; data
  interpolated off a finite-difference mesh has a floor set by its own
  error — measured at `2.2e−5` on the suite's hole at `h = 5/64` — so a
  `1e-13` demand stalls there and reports `success = false` on every
  find while returning **the same `J` to eleven digits**. The looser
  default makes the flag mean something; the number it guards does not
  move.
- `atol`, `maxiters` and `verbosity` go straight to `find_horizon`. Its
  default `atol = 0` iterates to the round-off floor, detected by stalled
  progress; a positive `atol` stops early and reports `success = false`
  when the floor arrives first.
"""
struct Horizon{T}
    every::Int
    N::Int
    r_seed::T
    spin::Bool
    unif_tol::Float64
    atol::Float64
    maxiters::Int
    verbosity::Int
end

function Horizon(::Type{T}=Float64; every::Integer=1, N::Integer=16,
                 r_seed=zero(T), spin::Bool=true, unif_tol=1.0e-8,
                 atol=0.0, maxiters::Integer=1000,
                 verbosity::Integer=0) where {T}
    every ≥ 0 || throw(ArgumentError(
        "the horizon cadence is a number of chunks and cannot be negative, " *
        "got every = $every; 0 means the horizon is never looked for, " *
        "which is what a case with no hole wants."))
    N > 1 || throw(ArgumentError(
        "the finder's angular resolution must be at least 2 (it becomes " *
        "EquiangularGrid(N − 1), and the fast flow needs lmax ≥ 1), got " *
        "N = $N."))
    T(r_seed) ≥ 0 || throw(ArgumentError(
        "the seed radius must be non-negative, got $r_seed; zero means " *
        "\"derive it from the background's analytic horizon radii\", which " *
        "is the only guess a case can make without being told one."))
    maxiters ≥ 0 || throw(ArgumentError(
        "maxiters must be non-negative, got $maxiters."))
    unif_tol > 0 || throw(ArgumentError(
        "the uniformization tolerance must be positive, got $unif_tol."))
    0 ≤ verbosity ≤ 2 || throw(ArgumentError(
        "ApparentHorizonFinder's verbosity is 0, 1 or 2, got $verbosity."))
    return Horizon{T}(Int(every), Int(N), T(r_seed), spin,
                      Float64(unif_tol), Float64(atol), Int(maxiters),
                      Int(verbosity))
end

# --- point location ---------------------------------------------------------

"""
    locate_block(forest, x) -> Int or nothing

The index into `forest.leaves` of the leaf whose interior contains `x`, or
`nothing` when `x` is outside the domain.

The descent TreeAMR does not (yet) export: the root brick from the domain
extents, the cell coordinates at the finest level present, then
[`find_leaf`](@ref) on that node and on each of its ancestors in turn —
the first that is a leaf is the one that covers `x`. `O(maxlevel · log
nleaves)`, host-side, and called once per query point.

A periodic dimension is wrapped, a non-periodic one refuses: a point one
cell outside a Dirichlet boundary is not in the mesh, and interpolating it
from the nearest block would quietly extrapolate. Points exactly on the
upper face belong to the last cell, which is where the shared plane's data
lives.
"""
function locate_block(forest::Forest{3,R}, x) where {R}
    root = 0
    stride = 1
    cell = ntuple(_ -> 0, Val(3))
    L = maxlevel(forest)
    n = 1 << L
    for d in 1:3
        lo, hi = forest.extents[d]
        s = (R(x[d]) - lo) / (hi - lo)
        if forest.periodic[d]
            s = s - floor(s)
        elseif s < 0 || s > 1
            return nothing
        end
        # The cell index across the whole brick at the finest level, then
        # split into (root brick, coordinate within it).
        u = min(floorint(s * forest.roots[d] * n), forest.roots[d] * n - 1)
        rp = u ÷ n
        root += rp * stride
        stride *= forest.roots[d]
        cell = Base.setindex(cell, u - rp * n, d)
    end
    for l in L:-1:0
        b = find_leaf(forest, root, l, cell)
        b === nothing || return b
        cell = ntuple(d -> cell[d] >> 1, Val(3))
    end
    return nothing
end

# --- the interpolation weights ----------------------------------------------

# The Lagrange basis of `n` equispaced nodes `0, 1, …, n−1`, evaluated at
# `ξ` together with its derivative — value weights and derivative weights
# in *node units*, so the physical gradient is `d / h`.
#
# Written as the plain products rather than in barycentric form on purpose:
# the barycentric weight `∏(ξ − j)/(ξ − k)` is `0/0` when the query sits
# exactly on a node, which is the *first* thing a test of "exact on
# polynomials" does. These products are `O(n³)` per axis per point and the
# find is dominated by the `n³ · nvars` loads of the contraction below.
@inline function lagrange_point_weights(::Val{n}, ξ::T) where {n,T}
    dx = ntuple(j -> ξ - T(j - 1), Val(n))
    den = ntuple(Val(n)) do k
        p = one(T)
        for j in 1:n
            j == k && continue
            p *= T(k - j)
        end
        p
    end
    w = ntuple(Val(n)) do k
        p = one(T)
        for j in 1:n
            j == k && continue
            p *= dx[j]
        end
        p / den[k]
    end
    d = ntuple(Val(n)) do k
        s = zero(T)
        for m in 1:n
            m == k && continue
            p = one(T)
            for j in 1:n
                (j == k || j == m) && continue
                p *= dx[j]
            end
            s += p
        end
        s / den[k]
    end
    return SVector{n,T}(w), SVector{n,T}(d)
end

# The stored-index window of `n` points along one axis, and the query's
# position in it. `s` is the query in cell units from the block's lower
# corner; the containing cell is clamped into `0 … N−1` so that a point on
# the upper face is interpolated from the last cell's window rather than
# from one that runs past the array.
#
# The window's first *stored* index is `c + G − n/2 + 2`, which for this
# package's `G = q/2 + 1` and `n = q + 2` is `c + 2`: it never reaches
# index 1 and never reaches `N + 2G + 1`, the shared upper plane that no
# kernel writes (`CLAUDE.md`).
@inline function interpolation_span(::Val{n}, s::T, N::Int, G::Int) where {n,T}
    c = clamp(floorint(s), 0, N - 1)
    i0 = c + G - (n ÷ 2) + 2
    ξ = s - T(c - (n ÷ 2) + 1)
    return i0, ξ
end

# The lower corner of the footprint along one axis, as a position.
@inline footprint_origin(origin::T, h::T, i0::Int, G::Int) where {T} =
    origin + T(i0 - G - 1) * h

# Whether every point of the footprint lattice is evolved. Exact rather
# than conservative: the lattice is a tensor product, so the nearest of its
# points to the center is the per-axis nearest, and the distance is the
# root of the sum of those three squares.
@inline footprint_evolved(::AllPoints, x0, h, ::Val{n}) where {n} = true

@inline function footprint_evolved(m::InteriorMask{T}, x0, h::T,
                                   ::Val{n}) where {T,n}
    r² = zero(T)
    for d in 1:3
        # The nearest node index, as `floor(s + 1/2)` rather than `round`:
        # `round` closes through a conversion MultiFloats does not have,
        # and this package spells that `floorint` (`precision.jl`).
        k = clamp(floorint((m.center[d] - x0[d]) / h + T(1 // 2)), 0, n - 1)
        δ = x0[d] + T(k) * h - m.center[d]
        r² += δ * δ
    end
    return r² ≥ m.r_1 * m.r_1
end

# On the tracked geometry (step 8d) the evolved region is `r ≥ r_1(n̂)`, and
# the per-axis nearest lattice point is no longer the one that decides: the
# surface is not a sphere. So the guard is exact **by enumeration** where the
# shape can matter — the nearest point outside the offset surface's bounding
# sphere `r_out − offset` passes the whole footprint, one inside `r_in −
# offset` refuses it, and in between every one of the `n³` lattice points is
# classified by the same `is_evolved` the norms use **(proposed in step 8d**,
# over `PLAN.md`'s conservative "use the bounding sphere `r_in − offset`",
# which would let a footprint read the layer's outer part wherever the
# horizon is farther out than its smallest radius — on harmonic Kerr's
# equator, by more than the whole margin**)**.
@inline function footprint_evolved(m::ShapeMask{T}, x0, h::T,
                                   ::Val{n}) where {T,n}
    r² = zero(T)
    for d in 1:3
        k = clamp(floorint((m.center[d] - x0[d]) / h + T(1 // 2)), 0, n - 1)
        δ = x0[d] + T(k) * h - m.center[d]
        r² += δ * δ
    end
    lo = m.r_in - m.offset
    hi = m.r_out - m.offset
    r² ≥ hi * hi && return true
    r² < lo * lo && return false
    for k3 in 0:(n - 1), k2 in 0:(n - 1), k1 in 0:(n - 1)
        p = (x0[1] + T(k1) * h, x0[2] + T(k2) * h, x0[3] + T(k3) * h)
        is_evolved(m, p) || return false
    end
    return true
end

# The refusal, dispatched on the mask so that the trivial one carries no
# message about a radius it does not have.
footprint_error(::AllPoints, x, n) = ErrorException("unreachable")

footprint_error(m::ShapeMask, x, n) = ArgumentError(
    "the interpolation footprint of $(Tuple(x)) reaches below the tracked " *
    "layer's offset surface r_h(n̂) − $(m.offset) around $(Tuple(m.center)) " *
    "(its radius lies between $(m.r_in - m.offset) and " *
    "$(m.r_out - m.offset)): the $(n)³ points this query would read are not " *
    "all in the evolved region, and the layer and the frozen core are not a " *
    "numerical solution (CODE.md, \"The interior\"). The horizon lies outside " *
    "the layer by the margin m, and so must everything interpolated from " *
    "the state.")

footprint_error(m::InteriorMask, x, n) = ArgumentError(
    "the interpolation footprint of $(Tuple(x)) reaches inside r_1 = " *
    "$(m.r_1) around $(Tuple(m.center)): the $(n)³ points this query would " *
    "read are not all in the evolved region, and the layer and the frozen " *
    "core are not a numerical solution (CODE.md, \"The interior\"). The " *
    "horizon lies outside the layer by the margin m, and so must " *
    "everything that is interpolated from the state — raise the margin, " *
    "shrink r_1, or stop asking for a surface inside the hole.")

# The tensor-product contraction, formed and consumed in place: the value
# of every variable at the query point and, when `DG` says so, its three
# spatial derivatives. Nothing of size `n³` is materialised.
@inline function interpolate_window(work, base::Int, st, sv::Int,
                                    ::Val{NV}, wx::SVector{n,T},
                                    wy::SVector{n,T}, wz::SVector{n,T},
                                    dx::SVector{n,T}, dy::SVector{n,T},
                                    dz::SVector{n,T}, inv_h::T,
                                    ::Val{DG}) where {NV,n,T,DG}
    val = zero(SVector{NV,T})
    gx = zero(SVector{NV,T})
    gy = zero(SVector{NV,T})
    gz = zero(SVector{NV,T})
    for k3 in 1:n, k2 in 1:n
        base23 = base + (k2 - 1) * st[2] + (k3 - 1) * st[3]
        for k1 in 1:n
            idx = base23 + (k1 - 1) * st[1]
            u = SVector{NV,T}(ntuple(v -> (@inbounds work[idx + (v - 1) * sv]),
                                     Val(NV)))
            val += (wx[k1] * wy[k2] * wz[k3]) * u
            if DG
                gx += (dx[k1] * wy[k2] * wz[k3]) * u
                gy += (wx[k1] * dy[k2] * wz[k3]) * u
                gz += (wx[k1] * wy[k2] * dz[k3]) * u
            end
        end
    end
    return val, (inv_h * gx, inv_h * gy, inv_h * gz)
end

# One query: locate, check the footprint, contract. The whole of the
# stopgap interpolator, and the only function in this file that indexes a
# working array.
function interpolate_point(fs::FieldSet{T,3}, x, ::Val{NV}, ::Val{n},
                           mask, ::Val{DG}) where {T,NV,n,DG}
    b = locate_block(fs.forest, x)
    b === nothing && throw(ArgumentError(
        "the point $(Tuple(x)) is outside the mesh: no leaf of the forest " *
        "covers it, so there is nothing to interpolate from. The horizon " *
        "finder queries the surface it is iterating toward, so a seed " *
        "sphere larger than the box — or a flow that has run away — " *
        "arrives here."))
    k = blockkey(fs, b)
    origin = block_origin(T, fs.forest, k)
    h = spacing(T, fs.forest, k)
    N = fs.forest.N
    spans = ntuple(Val(3)) do d
        interpolation_span(Val(n), (T(x[d]) - origin[d]) / h, N, fs.G[d])
    end
    x0 = ntuple(d -> footprint_origin(origin[d], h, spans[d][1], fs.G[d]),
                Val(3))
    footprint_evolved(mask, x0, h, Val(n)) ||
        throw(footprint_error(mask, x, n))
    st, sv, sb = work_strides(fs.work)
    base = 1 + (b - 1) * sb + sum(ntuple(d -> (spans[d][1] - 1) * st[d],
                                         Val(3)))
    wx, dx = lagrange_point_weights(Val(n), spans[1][2])
    wy, dy = lagrange_point_weights(Val(n), spans[2][2])
    wz, dz = lagrange_point_weights(Val(n), spans[3][2])
    return interpolate_window(fs.work, base, st, sv, Val(NV), wx, wy, wz,
                              dx, dy, dz, inv(h), Val(DG))
end

# The window is `n = q + 2` wide and half of it sits on each side of the
# query's cell, so the ghosts have to be at least `n/2 = G` deep — which
# they are, by `CODE.md`'s `G = q/2 + 1`. Checked once per batch rather
# than assumed, because this is the invariant that makes "interpolated
# without crossing into another block's array" true.
function check_interpolation_width(fs::FieldSet{T,3}, n::Int) where {T}
    all(g -> 2g ≥ n, fs.G) || throw(ArgumentError(
        "interpolation of order $n needs $(n ÷ 2) points on each side of " *
        "the query's cell, so the field set's ghosts must be at least " *
        "that deep, but G = $(fs.G). CODE.md's G = q/2 + 1 and order " *
        "q + 2 are exactly matched; a narrower halo would have to read a " *
        "neighbouring block's array, which is what this interpolator " *
        "exists not to do."))
    all(c -> c === :vertex, fs.centering) || throw(ArgumentError(
        "this interpolator assumes vertex-centered storage — the stored " *
        "point `j` of a block sits at `origin + (j − G − 1)·h`, which is " *
        "where `coordinates` puts it — but this field set is " *
        "$(fs.centering)."))
    # Nothing else is needed: with `G ≥ n/2` the window of the *clamped*
    # containing cell runs from stored index `c + G − n/2 + 2 ≥ 2` to
    # `c + G + n/2 + 1 ≤ N + 2G`, for every `c` in `0 … N−1` and every
    # `N ≥ 1` — so it never reaches index 1's ghost corner and never
    # reaches `N + 2G + 1`, the shared upper plane no kernel writes.
    return nothing
end

"""
    interpolate(fs::FieldSet{T,3}, xs; q, mask = AllPoints())
    interpolate_grad(fs::FieldSet{T,3}, xs; q, mask = AllPoints())

Every variable of `fs` at each point of `xs`, by tensor-product Lagrange
interpolation of order `q + 2` over the containing block's stored points —
and, for `interpolate_grad`, the three spatial gradients as well.

`xs` is a host array of points of any shape; the result is an array of the
same shape holding `SVector{nvars,T}` (and, for the gradient form, a second
array of `NTuple{3,SVector{nvars,T}}`). The batch is what makes the
interpolation *threaded*: each point is located, checked and contracted
independently and written to its own slot, so the answer does not depend on
the thread count.

**This is a stopgap** (`CODE.md`, "Upstream prerequisites", item 1):
point interpolation from a field set belongs in TreeAMR, whose `TODO.md`
lists it, and this package carries it only because the horizon finder
cannot wait for it. It is written as the horizon finder needs it — on the
host, at analysis cadence, over a whole batch — and not as a general
facility.

Order `q + 2` and not `q`: the interpolant is exact on polynomials of
degree `≤ q + 1` and converges at `O(h^{q+2})`, one order better than the
scheme, so the horizon's location is the *solution's* error and not the
interpolation's. Its gradient is one order behind, `O(h^{q+1})`, which is
what makes `K_ij` one order behind `γ_ij` in the ADM data below
(`notes/methods-ghso2.md` measures exactly that).

`mask` is the guard: an [`InteriorMask`](@ref) makes every query whose
footprint reaches inside `r_1` **throw** rather than interpolate data the
equations never produced. Ghosts must be filled before either function is
called; [`find_gh_horizon`](@ref) does that.
"""
function interpolate(fs::FieldSet{T,3}, xs::AbstractArray; q::Integer,
                     mask=AllPoints()) where {T}
    return first(_interpolate(fs, xs, q, mask, Val(false)))
end

function interpolate_grad(fs::FieldSet{T,3}, xs::AbstractArray; q::Integer,
                          mask=AllPoints()) where {T}
    return _interpolate(fs, xs, q, mask, Val(true))
end

function _interpolate(fs::FieldSet{T,3}, xs::AbstractArray, q::Integer, mask,
                      ::Val{DG}) where {T,DG}
    q ≥ 2 && iseven(q) || throw(ArgumentError(
        "the interpolation order follows the scheme's q, which is even and " *
        "at least 2 (CODE.md, \"The interface-order rule\"), but q=$q"))
    check_interpolation_width(fs, Int(q) + 2)
    return _interpolate(fs, xs, Val(fs.nvars), Val(Int(q) + 2), mask,
                        Val(DG))
end

function _interpolate(fs::FieldSet{T,3}, xs::AbstractArray, ::Val{NV},
                      ::Val{n}, mask, ::Val{DG}) where {T,NV,n,DG}
    vals = similar(xs, SVector{NV,T})
    grads = similar(xs, NTuple{3,SVector{NV,T}})
    # One output slot per input point and no accumulation anywhere, which
    # is what makes this safe to thread and its result independent of the
    # thread count (`CLAUDE.md`: bit-identity across thread counts is the
    # invariant). `CODE.md` asks for the batched form for exactly this.
    #
    # TreeAMR's `threaded_foreach` rather than `Threads.@threads` (amended
    # with TreeAMR's owner-based threading, 2026-09-25): chunk `c` of the
    # query points runs on thread `c` every call, it nests inside a caller's
    # own parallel loop where `@threads` would not, and it rethrows the
    # exception the body threw rather than a `TaskFailedException` — the
    # refusal of a query that reaches the layer *must* reach the caller
    # readable, since the driver records its message.
    TreeAMR.threaded_foreach(length(xs)) do j
        i = eachindex(xs)[j]
        v, g = interpolate_point(fs, xs[i], Val(NV), Val(n), mask, Val(DG))
        @inbounds vals[i] = v
        @inbounds grads[i] = g
    end
    return vals, grads
end

# The first real exception inside a `TaskFailedException` or a
# `CompositeException`, or the thing itself when it is neither. What a
# kernel throws on the CPU backend arrives wrapped (the `DomainError` of a
# degenerate metric, `hole_runs.jl`), which is what this is still for.
unwrap_task_failure(e) = e
unwrap_task_failure(e::TaskFailedException) =
    e.task.exception isa Exception ? unwrap_task_failure(e.task.exception) : e
unwrap_task_failure(e::CompositeException) =
    isempty(e.exceptions) ? e : unwrap_task_failure(first(e.exceptions))

# --- the ADM provider -------------------------------------------------------

"""
    GHADMProvider(fs, q, mask)

The batched ADM-variable provider `ApparentHorizonFinder` and
`KorzynskiSpin` consume: called with an array of Cartesian points, it
returns an array of `ADMVars(γ, ∂γ, K)` of the same shape.

Each point is [`interpolate_grad`](@ref)ed out of the state — `h` and `Π`
and the three `∂_i h` — and handed to [`adm_vars_from_state`](@ref), which
is GHSO2's pointwise extraction: `γ_ij = g_ij`, `∂_kγ_ij` from the
interpolated gradient, and `K_ij` from `∂_t g = β^i ∂_i g + (α/√γ)Π` and
the 3-Christoffels. The result is `Float64` whatever the run computes in,
because the finder's grid, origin and flow are.

**It holds a one-entry cache keyed on the identity of the query array.**
`KorzynskiSpin.surface_geometry` asks for `γ_ij` and `K_ij` in two separate
calls with the *same* points, and each call would otherwise repeat the
whole interpolation; the cache makes the second free. It is keyed on `===`
and not on the contents, so a different array — the next iteration's
surface — misses it and is recomputed, and nothing stale can be returned.
"""
mutable struct GHADMProvider{T,F,M}
    const fs::F
    const q::Int
    const mask::M
    lastxs::Any
    lastvals::Any
end

function GHADMProvider(fs::FieldSet{T,3}, q::Integer, mask) where {T}
    check_interpolation_width(fs, Int(q) + 2)
    return GHADMProvider{T,typeof(fs),typeof(mask)}(fs, Int(q), mask,
                                                    nothing, nothing)
end

function (p::GHADMProvider{T})(xs::AbstractArray) where {T}
    p.lastxs === xs && return p.lastvals
    vals, grads = _interpolate(p.fs, xs, Val(2NC), Val(p.q + 2), p.mask,
                               Val(true))
    out = similar(xs, ADMVars{Float64})
    for i in eachindex(xs)
        u = vals[i]
        g = grads[i]
        hv = SVector{NC,T}(ntuple(v -> u[v], Val(NC)))
        Πv = SVector{NC,T}(ntuple(v -> u[NC + v], Val(NC)))
        dh = ntuple(d -> SVector{NC,T}(ntuple(v -> g[d][v], Val(NC))), Val(3))
        γ, ∂γ, K = adm_vars_from_state(hv, Πv, dh[1], dh[2], dh[3])
        out[i] = ADMVars(SMatrix{3,3,Float64}(tofloat64.(γ)),
                         SArray{Tuple{3,3,3},Float64}(tofloat64.(∂γ)),
                         SMatrix{3,3,Float64}(tofloat64.(K)))
    end
    p.lastxs = xs
    p.lastvals = out
    return out
end

# The `q + 2` of the provider's window is a *number* and not a `Val`, so
# `p.q + 2` above would be a dynamic dispatch once per batch. It is: one
# per horizon find, against `n³ · 20` loads per point, and making it a type
# parameter would put the interpolation order in the record's type.

"""
    gh_adm_provider(p::GHProblem, t) -> GHADMProvider

The provider for *this* problem at *this* time: the state field set brought
to the host ([`hostcopy`](@ref)), the scheme's `q`, and the interior's mask
at `t` as the guard.

The mask is the one every norm takes, [`interior_mask`](@ref), so "the
horizon finder reads only the evolved region" and "the norms count only the
evolved region" are the same statement with the same radius — and the guard
moves with the hole, because it is built at this call's `t` like every
other hook (`CLAUDE.md`, "Hooks depend on time").
"""
function gh_adm_provider(p::GHProblem{T,G,q}, t) where {T,G,q}
    return GHADMProvider(hostcopy(p.U), q, interior_mask(p.interior, T(t)))
end

# --- the find ---------------------------------------------------------------

"""
    horizon_radii(points, center) -> (r_min, r_mean, r_max)

The coordinate radii of a found surface, measured from `center`.

`r_min` and `r_max` are the extremes over the finder's collocation points —
the two numbers the analytic Kerr values are compared against, since the
horizon is oblate in both charts and its smallest and largest coordinate
radii are [`horizon_min_radius`](@ref) and [`horizon_max_radius`](@ref) —
and `r_mean` is their `sin θ`-weighted average, which is the round-sphere
mean `∮ r dΩ/4π` to the accuracy of the grid's own quadrature.

**Measured from the analytic center rather than from the finder's `origin`
(proposed in step 7).** `CODE.md` asks for "the coordinate radii of the
surface points" without saying from where, and the claims made on them are
about the hole: that the horizon encloses the layer by the margin `m`, and
that `r_min` and `r_max` are Kerr's. Both are statements about the center
the layer is built around. The offset between the two centers is a row of
its own, `center_offset`.
"""
function horizon_radii(points::AbstractMatrix{<:SVector{3}}, θs,
                       center::SVector{3,Float64})
    rs = [sqrt(sum(abs2, p - center)) for p in points]
    wsum = 0.0
    rsum = 0.0
    for ij in CartesianIndices(rs)
        w = sin(θs[ij[1]])
        wsum += w
        rsum += w * rs[ij]
    end
    return minimum(rs), rsum / wsum, maximum(rs)
end

"""
    find_gh_horizon(p::GHProblem, u, t; every keyword of `Horizon`,
                    center = the analytic center, origin = center,
                    hlm = nothing)

Find the apparent horizon of the state `u` at time `t` and return
everything `CODE.md`'s analysis table asks of it:

    (; success, iters, origin, center, center_offset, r_min, r_mean, r_max,
       origin_r_min, origin_r_mean, origin_r_max, area, M_irr, J, spin_axis,
       M_ch, hlm, grid, H_norm, spin_success)

`center` is the point `r_min`, `r_mean`, `r_max` and `center_offset` are
measured from — the analytic center `c(t)` unless given, which is step 7's
choice for a diagnostic of the analytic hole; a tracked run passes its
tracked center predicted to `t`, so that `center_offset` is the track's
prediction error (added in step 8d). `origin_r_min`, `origin_r_mean` and
`origin_r_max` are the same radii about the surface's own recentred
`origin`, which is what the shape `hlm` describes and what a tracked
geometry is built from.

GHSO2's `find_gh_horizon` (`notes/methods-ghso2.md`, "Apparent horizons and
spin") composed out of this package's mesh: `ApparentHorizonFinder`'s fast
flow over the interpolating [`GHADMProvider`](@ref) for the location, the
shape `hlm` and the proper area; `KorzynskiSpin.horizon_spin` on the same
collocation grid for `J` and its coordinate-space axis; and

    M_irr = √(A/16π),   M_ch = √(M_irr² + J²/(4 M_irr²))

from those. For Kerr the reference values are `A = 4π(r₊² + a²)`,
`M_irr = √(A/16π)`, `J = M a` and `M_ch = M`, and a boost changes none of
them.

**It scatters the state and fills the ghosts first**, with this `t`'s
Dirichlet hook: the interpolation window of a query near a block face
reaches `G` points into the neighbour, and those points are ghosts. The
state itself is not modified — the scatter writes `p.U` from `u`, which is
what every monitor in this package does before it reads a stencil.

The seed is `CODE.md`'s: the sphere `(origin, r_seed)` on the first find,
and on every later one the previous `hlm` **recentred on `origin`** — by
default the analytic center, not the previous origin, because the layer and
the mesh follow the analytic center and a shape that drifted with the
surface would seed the next find from a worse place than the case's own
answer; a tracked run's layer follows the track, and it passes the tracked
center instead (step 8d).

A failed *spin* leaves `J = NaN` and `spin_success = false` without failing
the find, which is GHSO2's behaviour: the area and the location are still
the run's numbers. A failed *find* is reported in `success`; the flow's
own diagnostics are `iters` and `H_norm`.
"""
function find_gh_horizon(p::GHProblem{T,G,q}, u, t; N::Integer=16,
                         r_seed=nothing, origin=nothing, hlm=nothing,
                         center=nothing, spin::Bool=true, unif_tol=1.0e-8,
                         atol=0.0, maxiters::Integer=1000,
                         verbosity::Integer=0) where {T,G,q}
    case = p.case
    scatter!(p.U, u)
    boundary = dirichlet(case, T(t))
    if boundary === nothing
        fill_ghosts!(p.U, p.schedule)
    else
        fill_ghosts!(p.U, p.schedule; boundary=boundary)
    end

    # The point the radii are measured from (step 8d): the analytic center
    # unless the caller names another — the driver of a tracked run passes
    # the tracked center, predicted to `t`.
    c_at = center === nothing ? center_at(case.center, T(t)) : center
    c64 = SVector{3,Float64}(tofloat64(c_at[1]), tofloat64(c_at[2]),
                             tofloat64(c_at[3]))
    x0 = origin === nothing ? c64 : SVector{3,Float64}(Float64(origin[1]),
                                                       Float64(origin[2]),
                                                       Float64(origin[3]))
    provider = gh_adm_provider(p, t)
    result = if hlm !== nothing
        # Seeded from the previous find: the *shape* is that one's, the
        # origin is `c(t)`, and the seed radius plays no part.
        find_horizon(provider, x0, Vector{ComplexF64}(hlm), Float64(atol),
                     Int(maxiters); verbosity=Int(verbosity))
    else
        r0 = r_seed !== nothing ? Float64(r_seed) :
             (tofloat64(horizon_min_radius(case.background)) +
              tofloat64(horizon_max_radius(case.background))) / 2
        r0 > 0 || throw(ArgumentError(
            "the seed sphere needs a positive radius, got $r0: with no " *
            "radius given, one is derived from the background's analytic " *
            "horizon radii, and a background with no horizon has none — " *
            "pass r_seed."))
        find_horizon(provider, x0, Int(N), r0, Float64(atol), Int(maxiters);
                     verbosity=Int(verbosity))
    end

    points = horizon_points(result)
    θs, _ = horizon_grid(result.grid)
    size(points, 1) == length(θs) || throw(ErrorException(
        "the finder's collocation points are $(size(points)) and its θ " *
        "values $(length(θs)): horizon_radii weights the first axis by " *
        "sin θ, which assumes the (θ, φ) layout EquiangularGrid has."))
    r_min, r_mean, r_max = horizon_radii(points, θs, c64)
    # And about the surface's own recentred origin (added in step 8d): the
    # radii of the shape `hlm` describes, which is what a tracked geometry is
    # an offset of.
    o_min, o_mean, o_max = horizon_radii(points, θs, result.origin)

    area = result.area
    M_irr = sqrt(area / (16π))
    J, axis, spin_success = if spin
        gh_horizon_spin(result, provider, Float64(unif_tol))
    else
        (NaN, SVector{3,Float64}(NaN, NaN, NaN), false)
    end
    M_ch = isnan(J) ? NaN : sqrt(M_irr^2 + J^2 / (4 * M_irr^2))

    return (success=result.success, iters=result.iters, origin=result.origin,
            center=c64, center_offset=sqrt(sum(abs2, result.origin - c64)),
            r_min=r_min, r_mean=r_mean, r_max=r_max, origin_r_min=o_min,
            origin_r_mean=o_mean, origin_r_max=o_max, area=area, M_irr=M_irr,
            J=J, spin_axis=axis, M_ch=M_ch, hlm=result.hlm, grid=result.grid,
            H_norm=result.H_norm, spin_success=spin_success)
end

# The Korzyński spin on the surface the flow converged to, with the failure
# contained. `notes/methods-ghso2.md`: "a failed spin computation leaves
# `J = NaN` without failing the find" — the area and the location are the
# run's numbers either way, and an analysis quantity that ends a run is
# worse than one that reports a `NaN` beside the chunk it failed at. The
# providers are the *same* `GHADMProvider`, so the second of the two calls
# `surface_geometry` makes is answered from its cache.
function gh_horizon_spin(result, provider, unif_tol::Float64)
    metric3(xs::AbstractArray) = map(v -> v.γ, provider(xs))
    excurv3(xs::AbstractArray) = map(v -> v.K, provider(xs))
    try
        s = horizon_spin(result, metric3, excurv3; unif_tol=unif_tol)
        return s.J, s.axis_embedding, s.success
    catch e
        e isa InterruptException && rethrow()
        return NaN, SVector{3,Float64}(NaN, NaN, NaN), false
    end
end
