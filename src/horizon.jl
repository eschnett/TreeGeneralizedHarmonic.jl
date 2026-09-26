# The apparent horizon: where it is, how big it is, and how fast it spins.
#
# `CODE.md`, "Analysis quantities" (the horizon rows) and "Upstream
# prerequisites" (point interpolation). Three things live here, in the
# order the numbers are produced:
#
#   1. **Point interpolation from a field set** — TreeAMR's `interpolate`
#      (its M11) with a `Lagrange(q + 2)` basis, batched over an array of
#      query points and run on the field set's backend. This package adds
#      only the order and the guard below ([`gh_interpolate`](@ref)). Until
#      2026-09-26 this was a stopgap of this package's own — `locate_block`
#      and a host-side window contraction — carried until TreeAMR grew it
#      (`CLAUDE.md`, "No mesh machinery"; `CODE.md`, "Upstream
#      prerequisites", item 1, amended then).
#   2. **The ADM provider** `ApparentHorizonFinder` consumes, in its
#      *batched* form — all surface points at once, so the interpolation
#      is one launch — built out of `pointwise.jl`'s
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
#     footprint reaches `r_1`"). TreeAMR *flags* a query whose stencil
#     reaches an excluded region; the region is the mask's complement,
#     [`UnevolvedRegion`](@ref), and the check is exact: TreeAMR places a
#     stencil point where `coordinates` does, and the region asks the same
#     `is_evolved` the norms ask.
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

# --- the guard, as a TreeAMR region -----------------------------------------

"""
    UnevolvedRegion(mask)

The region `mask` does **not** evolve — the damping layer and the frozen
core — as a TreeAMR `Region`, which is how the footprint guard reaches
TreeAMR's `interpolate`: its `exclude` flags every query whose stencil has a
point inside the region, and [`gh_interpolate`](@ref) turns a flag into a
refusal (added 2026-09-26, with the port to TreeAMR's M11).

A point is inside exactly when `is_evolved(mask, x)` is false — the same
predicate every masked norm, the indicator and the speed kernel ask — and
TreeAMR evaluates a stencil point at the position `coordinates` gives, which
is [`point_position`](@ref)'s expression in the same order. So the guard
refuses **exactly** what the norms mask, bit for bit: a footprint is refused
iff one of the `(q + 2)³` points it reads is a point the norms do not count.

It is not TreeAMR's `Ellipsoid` for the round mask, although that is a ball
and its test is as cheap: `Σ((x − c)/r₁)² < 1` rounds differently from
`is_evolved`'s `Σ(x − c)² ≥ r₁²`, so the two disagree on points within
roundoff of `r₁`, and a separable test of our own costs the same (measured
2026-09-26, `CODE.md`, "Analysis quantities").
"""
struct UnevolvedRegion{M} <: Region
    mask::M
end

@inline TreeAMR.inside(r::UnevolvedRegion, x) = !is_evolved(r.mask, x)

# The guard's region for a mask: none where every point is evolved.
exclude_region(::AllPoints) = nothing
exclude_region(mask) = UnevolvedRegion(mask)

# The squared distance from `c` to the nearest point of a stencil lattice,
# with each stencil point where TreeAMR puts it (`stencil_position`, the
# expression `coordinates` evaluates). The lattice is a tensor product, so
# the nearest point is the per-axis nearest, and its squared distance is
# formed from the same differences, squares and left-to-right sum as
# `is_evolved`'s: this *is* that point's `r²`, bit for bit, and every other
# stencil point's is at least as large, because floating-point addition is
# monotone.
@inline function nearest_r2(c, origin, h, base, off, ::Val{n}) where {n}
    r² = zero(h)
    for d in 1:3
        best = zero(h)
        for k in 0:(n - 1)
            y = TreeAMR.stencil_position(origin, h, base, off, d, k) - c[d]
            best = k == 0 ? y * y : min(best, y * y)
        end
        r² += best
    end
    return r²
end

# TreeAMR's default `stencil_hits` enumerates the `n³` stencil points; these
# are the cheaper methods its `Region` docstring invites, and each agrees
# with the enumeration exactly. The round mask: the nearest point decides.
@inline function TreeAMR.stencil_hits(r::UnevolvedRegion{<:InteriorMask},
                                      origin::NTuple{3}, h, base, off,
                                      ::Val{n}) where {n}
    m = r.mask
    return nearest_r2(m.center, origin, h, base, off, Val(n)) < m.r_1 * m.r_1
end

# The tracked geometry (step 8d): the evolved region is `r ≥ r_1(n̂)`, which
# is not a sphere, so the nearest point decides only outside the offset
# surface's two bounding spheres — compared as `sqrt(r²)` against the radii,
# which is `is_evolved`'s own comparison, so both fast paths are exact by
# the monotonicity of `sqrt` — and in between every stencil point is
# classified by `is_evolved` **(proposed in step 8d** over `PLAN.md`'s
# conservative "use the bounding sphere `r_in − offset`", which would refuse
# a footprint wherever the horizon is farther out than its smallest radius —
# on harmonic Kerr's equator, by more than the whole margin**)**.
@inline function TreeAMR.stencil_hits(r::UnevolvedRegion{<:ShapeMask},
                                      origin::NTuple{3}, h, base, off,
                                      ::Val{n}) where {n}
    m = r.mask
    ρ = sqrt(nearest_r2(m.center, origin, h, base, off, Val(n)))
    ρ ≥ m.r_out - m.offset && return false
    ρ < m.r_in - m.offset && return true
    for J in CartesianIndices(ntuple(_ -> n, Val(3)))
        x = ntuple(d -> TreeAMR.stencil_position(origin, h, base, off, d,
                                                 J[d] - 1), Val(3))
        is_evolved(m, x) || return true
    end
    return false
end

# The refusal, dispatched on the mask so that the trivial one carries no
# message about a radius it does not have.
footprint_error(::AllPoints, x, n) = ErrorException("unreachable")

footprint_error(m, x, n) = ArgumentError(
    "the interpolation footprint of $(Tuple(x)) reaches a point the mask " *
    "$(nameof(typeof(m))) does not evolve: the $(n)³ points this query " *
    "would read are not all in the evolved region.")

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

# --- the interpolation -------------------------------------------------------

# The multi-indices TreeAMR's `interpolate` takes: the value alone, and the
# value with the three first derivatives, in that order.
const INTERP_VALUE = ((0, 0, 0),)
const INTERP_VALUE_GRAD = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))

function check_interpolation_order(q::Integer)
    q ≥ 2 && iseven(q) || throw(ArgumentError(
        "the interpolation order follows the scheme's q, which is even and " *
        "at least 2 (CODE.md, \"The interface-order rule\"), but q=$q"))
    return nothing
end

# The one call into TreeAMR: `Lagrange(q + 2)` over the containing block's
# stored points, on the field set's backend, with the mask's region as
# `exclude`; the flags turned into the refusal on the host. Returns the host
# array `vals[v, k, j]` — variable `v`, derivative `derivs[k]`, point `j`
# (linear index of `xs`).
#
# The points are converted to `SVector{3,T}` and moved to the field set's
# backend first: TreeAMR interpolates where the data is and wants the points
# there too, and a device without `Float64` could not hold the finder's
# `Float64` points. `T(x[d])` is the conversion TreeAMR would make itself.
function interpolate_state(fs::FieldSet{T,3}, xs::AbstractArray, q::Integer,
                           mask, derivs) where {T}
    check_interpolation_order(q)
    n = Int(q) + 2
    xv = vec(xs)
    pts = to_backend(get_backend(fs.work),
                     [SVector{3,T}(T(x[1]), T(x[2]), T(x[3])) for x in xv])
    res = TreeAMR.interpolate(fs, pts, Lagrange(n); derivs=derivs,
                              exclude=exclude_region(mask))
    vals = res.values isa Array ? res.values : Array(res.values)
    excluded = res.excluded isa Array ? res.excluded : Array(res.excluded)
    j = findfirst(excluded)
    j === nothing || throw(footprint_error(mask, xv[j], n))
    return vals
end

# `vals[:, k, j]` as one `SVector` per point, in the shape of the query.
function unpack_values(vals::AbstractArray{T,3}, k::Int, ::Val{NV},
                       sz) where {T,NV}
    return reshape([SVector{NV,T}(ntuple(v -> vals[v, k, j], Val(NV)))
                    for j in axes(vals, 3)], sz)
end

"""
    gh_interpolate(fs::FieldSet{T,3}, xs; q, mask = AllPoints())
    gh_interpolate_grad(fs::FieldSet{T,3}, xs; q, mask = AllPoints())

Every variable of `fs` at each point of `xs`, by TreeAMR's tensor-product
[`Lagrange`](@ref)`(q + 2)` interpolation over the containing block's stored
points — and, for `gh_interpolate_grad`, the three spatial gradients as
well. What this package adds to TreeAMR's `interpolate` is the order and the
guard; the location, the weights, the contraction and the batch are
TreeAMR's M11 (amended 2026-09-26: these were `interpolate` and
`interpolate_grad`, a stopgap of this package's own, and the name collided
with TreeAMR's once it grew one).

`xs` is an array of points of any shape; the result is a host array of the
same shape holding `SVector{nvars,T}` (and, for the gradient form, a second
array of `NTuple{3,SVector{nvars,T}}`). The field set may live on a device:
the interpolation runs there, one launch per call, and only the points and
the answers cross. Every query writes its own slot, so the result does not
depend on the thread count.

Order `q + 2` and not `q`: the interpolant is exact on polynomials of
degree `≤ q + 1` and converges at `O(h^{q+2})`, one order better than the
scheme, so the horizon's location is the *solution's* error and not the
interpolation's. Its gradient is one order behind, `O(h^{q+1})`, which is
what makes `K_ij` one order behind `γ_ij` in the ADM data below
(`notes/methods-ghso2.md` measures exactly that). At this package's
`G = q/2 + 1` the window is centred everywhere but on the domain's upper
face.

`mask` is the guard: an [`InteriorMask`](@ref) or a `ShapeMask` makes every
query whose footprint reaches a point the mask does not evolve **throw**
rather than interpolate data the equations never produced
([`UnevolvedRegion`](@ref); TreeAMR only flags, and the refusal is this
package's). A point outside the domain is TreeAMR's `ArgumentError`. Ghosts
must be filled before either function is called; [`find_gh_horizon`](@ref)
does that.
"""
function gh_interpolate(fs::FieldSet{T,3}, xs::AbstractArray; q::Integer,
                        mask=AllPoints()) where {T}
    vals = interpolate_state(fs, xs, q, mask, INTERP_VALUE)
    return unpack_values(vals, 1, Val(fs.nvars), size(xs))
end

function gh_interpolate_grad(fs::FieldSet{T,3}, xs::AbstractArray; q::Integer,
                             mask=AllPoints()) where {T}
    vals = interpolate_state(fs, xs, q, mask, INTERP_VALUE_GRAD)
    return _unpack_grad(vals, Val(fs.nvars), size(xs))
end

function _unpack_grad(vals::AbstractArray{T,3}, ::Val{NV}, sz) where {T,NV}
    grads = reshape([ntuple(d -> SVector{NV,T}(ntuple(v -> vals[v, d + 1, j],
                                                      Val(NV))), Val(3))
                     for j in axes(vals, 3)], sz)
    return unpack_values(vals, 1, Val(NV), sz), grads
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

Each point is [`gh_interpolate_grad`](@ref)ed out of the state — `h` and `Π`
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
    check_interpolation_order(q)
    fs.nvars == 2NC || throw(ArgumentError(
        "the ADM provider reads the packed state (h, Π), $(2NC) variables, " *
        "but this field set holds $(fs.nvars)."))
    return GHADMProvider{T,typeof(fs),typeof(mask)}(fs, Int(q), mask,
                                                    nothing, nothing)
end

function (p::GHADMProvider{T})(xs::AbstractArray) where {T}
    p.lastxs === xs && return p.lastvals
    # `vals[v, k, j]`: variable `v`, the value (`k = 1`) or `∂_{k−1}`, point
    # `j` — read straight into the pointwise algebra's arguments.
    vals = interpolate_state(p.fs, xs, p.q, p.mask, INTERP_VALUE_GRAD)
    out = similar(xs, ADMVars{Float64})
    for (j, i) in enumerate(eachindex(xs))
        hv = SVector{NC,T}(ntuple(v -> vals[v, 1, j], Val(NC)))
        Πv = SVector{NC,T}(ntuple(v -> vals[NC + v, 1, j], Val(NC)))
        dh = ntuple(d -> SVector{NC,T}(ntuple(v -> vals[v, d + 1, j], Val(NC))),
                    Val(3))
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
# `Lagrange(q + 2)` in `interpolate_state` is a dynamic dispatch once per
# batch. It is: one per batch, against `n³ · 20` loads per point, and making
# it a type parameter would put the interpolation order in the record's type.

"""
    gh_adm_provider(p::GHProblem, t) -> GHADMProvider

The provider for *this* problem at *this* time: the state field set where
it lives, the scheme's `q`, and the interior's mask at `t` as the guard.
TreeAMR interpolates on the field set's backend, so a device-resident state
is read where it is and only the surface points and their answers cross —
not the whole state, which `hostcopy` copied once per find until the port
to TreeAMR's M11 (amended 2026-09-26).

The mask is the one every norm takes, [`interior_mask`](@ref), so "the
horizon finder reads only the evolved region" and "the norms count only the
evolved region" are the same statement with the same radius — and the guard
moves with the hole, because it is built at this call's `t` like every
other hook (`CLAUDE.md`, "Hooks depend on time").
"""
function gh_adm_provider(p::GHProblem{T,G,q}, t) where {T,G,q}
    return GHADMProvider(p.U, q, interior_mask(p.interior, T(t)))
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
