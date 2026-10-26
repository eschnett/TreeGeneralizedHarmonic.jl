# The tracked horizon: where the layer is put once it follows the horizon
# that was found rather than the one that was assumed.
#
# `CODE.md`, "The interior" — "The tracked geometry" (added in step 8d) — and
# `PLAN.md`'s step 8d. Step 7's finder was a *diagnostic*: it read the state
# and fed nothing back. From step 8d a case may ask for its layer to follow
# the found horizon, and then the finder's answer is an input, which is why
# it lives in a file of its own, on the host, between the finder
# (`horizon.jl`) and the loop that calls it (`driver.jl`):
#
#   1. **The conversions** — the finder's complex `hlm` to the real
#      coefficients the kernel evaluates ([`real_shape`](@ref)), and the
#      analytic horizon sampled into the same form for the seed
#      ([`analytic_shape`](@ref)).
#   2. **The track** — [`HorizonTrack`](@ref), a plain host-side record of
#      the last find and a velocity estimate; [`seed_track`](@ref) from the
#      case's analytic answer, [`update_track`](@ref) from each find, and
#      [`track_center`](@ref), which turns it into the one thing every
#      kernel already consumes, a [`HoleCenter`](@ref).
#   3. **The geometry** — [`fitted_interior`](@ref), the kernel argument
#      built from a track and a mesh once per chunk, and
#      [`margin_efolds`](@ref), what its margin buys against step 8a's
#      leakage.
#
# Three things are easy to get wrong, and each is written out where it
# happens:
#
#   * **A track is data, never a mutated field.** `update_track` returns a
#     new track, and the kernel sees the trajectory `c(t) = c_find + v_est
#     (t − t_find)` through a `HoleCenter` — a function of `t`, exactly as
#     step 5's analytic center is (`CLAUDE.md`, "The center is a function of
#     `t`, never a mutated field").
#   * **A failed find is not a lost track.** The track *coasts* — the last
#     geometry, carried along the velocity estimate — and only
#     `max_misses` consecutive misses end the run; a find that moves the
#     horizon's smallest radius by more than half a stencil is refused
#     outright, because the geometry would then expose points no layer ever
#     treated.
#   * **The radii of a found surface are about its own origin.** Step 7
#     measured them from the analytic center, which is what a diagnostic of
#     the analytic hole wants; the track is the surface's own geometry, and
#     `find_gh_horizon` now reports both.
#
# **From step B2 a track may predict along a quadratic** (`quadratic = true`,
# a [`TrackedSpec`](@ref)'s default; a [`FittedSpec`](@ref)'s track keeps the
# line, bit for bit): the least-squares quadratic in time through the last four
# finds, blended `C¹` into the previous prediction over the next find interval,
# with the found origin refused beyond `G h/2` of the prediction and the
# interval chosen so that `|a| Δt² ≤ G h/4` ([`track_state`](@ref),
# [`find_interval`](@ref); `CODE.md`, "The tracks"). The same file builds the
# **tracked analytic target** from it, [`tracked_interior`](@ref): step 5's
# sphere about the track, relaxing toward a boosted Kerr-Schild hole on the
# track's line.

# --- the conversions ---------------------------------------------------------

"""
    real_shape(hlm, grid, lmax) -> Vector{Float64}

The finder's shape — complex spin-0 coefficients `hlm` of `r_h(θ, φ)` about
its origin, on its collocation grid — as the real coefficients of
[`real_harmonic_index`](@ref)'s layout to degree `lmax`:
`AbstractSphericalHarmonics.ash_resample` onto `EquiangularGrid(lmax)`
(spectral truncation, or zero-padding), then [`real_from_complex`](@ref).
"""
function real_shape(hlm::AbstractVector{<:Complex}, grid::SphereGrid,
                    lmax::Integer)
    c = ash_resample(EquiangularGrid(Int(lmax)), hlm, grid, 0)
    return real_from_complex(c, lmax)
end

"""
    analytic_shape(background, lmax) -> Vector{Float64}

The background's analytic horizon ([`analytic_horizon_radius`](@ref)) as
real coefficients to degree `lmax`, about the hole's center: sampled on
`EquiangularGrid(L)` with `L = max(4 lmax + 4, 32)`, transformed with
`ash_transform`, truncated to `lmax` and made real.

**Sampled finer than the degree it is truncated to (proposed in step 8d).**
`PLAN.md` says "sample on `EquiangularGrid(lmax_shape)`", which is
*interpolation* at the degree's own collocation points and aliases the
spheroid's higher multipoles into the kept ones; sampling at `L ≥ 4 lmax`
and truncating is the spectral projection, whose error is the neglected
multipoles alone, which is what "the truncation the `l ≤ lmax` fit has"
means. The cost is a few thousand evaluations, once per run.
"""
function analytic_shape(background, lmax::Integer)
    L = max(4 * Int(lmax) + 4, 32)
    grid = EquiangularGrid(L)
    r = Matrix{ComplexF64}(undef, ash_grid_size(grid)...)
    for ij in CartesianIndices(r)
        θ, φ = ash_point_coord(grid, ij)
        sθ, cθ = sincos(θ)
        sφ, cφ = sincos(φ)
        n = SVector{3,Float64}(sθ * cφ, sθ * sφ, cθ)
        r[ij] = tofloat64(analytic_horizon_radius(background, n))
    end
    return real_shape(ash_transform(grid, r, 0), grid, lmax)
end

# A real shape vector at another degree: the layout is shared across
# degrees, so truncation and zero-padding are a copy of the leading entries.
function _resize_shape(a::AbstractVector, lmax::Integer)
    out = zeros(Float64, (lmax + 1)^2)
    n = min(length(a), length(out))
    out[1:n] .= a[1:n]
    return out
end

# --- the track ---------------------------------------------------------------

"""
    TrackLine{T}(t0, c, v, a)

One prediction of a track's center (added in step B2): the quadratic in time
about `t0`, `c(t) = c + v τ + ½ a τ²` with `τ = t − t0`. `a = 0` is a line.
"""
struct TrackLine{T}
    t0::T
    c::SVector{3,T}
    v::SVector{3,T}
    a::SVector{3,T}
end

# The prediction and its first two derivatives at `t`.
@inline function line_at(l::TrackLine{T}, t) where {T}
    τ = T(t) - l.t0
    return (l.c + τ * l.v + (τ * τ / 2) * l.a, l.v + τ * l.a, l.a)
end

"""
    TRACK_FIT_FINDS = 4

How many of the latest finds the quadratic prediction is fitted to, by least
squares in time (`CODE.md`, "The tracks", proposed 2026-10-25 and built in step
B2). With fewer the track predicts along the chord of the last two.
"""
const TRACK_FIT_FINDS = 4

"""
    HorizonTrack{T}

The tracked horizon, host-side: what the last successful find said and how
the track has fared since.

- `t_find`, `c_find` — the time of the last find and the finder's recentred
  origin then; `v_est` — the velocity the track carries the center along;
- `r_min`, `r_max` — the found surface's smallest and largest coordinate
  radius **about `c_find`**, over the finder's collocation points;
- `shape`, `lmax` — its radius `r_h(n̂)` about `c_find` as real coefficients
  ([`real_harmonic_index`](@ref)) to the degree the geometry uses;
- `hlm`, `grid` — the finder's own complex coefficients and grid, which the
  next find is seeded with (`nothing` before the first);
- `source` — `:analytic` (the seed), `:found` (the last find succeeded) or
  `:coasting` (it did not, and the geometry is the last one carried along
  `v_est`); `misses` — consecutive failed finds; `nfinds` — successful ones.

**The quadratic prediction (added in step B2)**, used where `quadratic` is
`true` and ignored where it is `false` — a [`FittedSpec`](@ref)'s track, which
carries the center along the line `c_find + v_est (t − t_find)` as it did:

- `history`, `nhist` — the times and origins `(t, x, y, z)` of the last
  `nhist ≤ TRACK_FIT_FINDS` finds, oldest first, packed four reals a find;
- `pred` — the prediction made at the last find ([`TrackLine`](@ref));
- `prev`, `t_blend`, `blend` — the prediction in use before it, as its
  quadratic Taylor polynomial about `t_blend`, and the interval `blend` over
  which `pred` takes over from it (`0`: none).

A plain immutable struct, and [`update_track`](@ref) returns a new one: the
track is data the driver passes along, never a field that changes under a
kernel. [`track_center`](@ref) and [`track_state`](@ref) are what the driver
reads of it.
"""
struct HorizonTrack{T}
    t_find::T
    c_find::SVector{3,T}
    v_est::SVector{3,T}
    r_min::T
    r_max::T
    shape::Vector{Float64}
    lmax::Int
    hlm::Union{Nothing,Vector{ComplexF64}}
    grid::Union{Nothing,SphereGrid}
    source::Symbol
    misses::Int
    nfinds::Int
    quadratic::Bool
    history::SVector{16,T}
    nhist::Int
    pred::TrackLine{T}
    prev::TrackLine{T}
    t_blend::T
    blend::T
end

# The track of step 8d: the line, with no quadratic state (`quadratic =
# false`), from the twelve fields it had.
function HorizonTrack{T}(t_find, c_find, v_est, r_min, r_max, shape, lmax, hlm,
                         grid, source, misses, nfinds) where {T}
    l = TrackLine{T}(T(t_find), SVector{3,T}(c_find), SVector{3,T}(v_est),
                     zero(SVector{3,T}))
    return HorizonTrack{T}(T(t_find), c_find, v_est, r_min, r_max, shape, lmax, hlm,
                           grid, source, misses, nfinds, false,
                           zero(SVector{16,T}), 0, l, l, T(t_find), zero(T))
end

# The same track with the quadratic fields replaced.
function _with_prediction(tr::HorizonTrack{T}; quadratic=tr.quadratic,
                          history=tr.history, nhist=tr.nhist, pred=tr.pred,
                          prev=tr.prev, t_blend=tr.t_blend,
                          blend=tr.blend) where {T}
    return HorizonTrack{T}(tr.t_find, tr.c_find, tr.v_est, tr.r_min, tr.r_max,
                           tr.shape, tr.lmax, tr.hlm, tr.grid, tr.source,
                           tr.misses, tr.nfinds, quadratic, history, nhist, pred,
                           prev, t_blend, blend)
end

# The `C²` weight the blend turns over with, and its first two derivatives in
# `s`: the quintic smoothstep, `0` at `s ≤ 0` and `1` at `s ≥ 1`, with both
# derivatives zero at either end — so the blended prediction and its velocity
# are continuous where the blend starts and where it ends.
@inline function _blend_weight(s::T) where {T}
    s ≤ 0 && return (zero(T), zero(T), zero(T))
    s ≥ 1 && return (one(T), zero(T), zero(T))
    σ = s * s * s * (10 + s * (6 * s - 15))
    σ′ = 30 * s * s * (1 - s) * (1 - s)
    σ″ = 60 * s * (1 - s) * (1 - 2 * s)
    return (σ, σ′, σ″)
end

"""
    track_state(tr::HorizonTrack, t) -> (c, ċ, c̈)

Where the track puts the hole's center at `t`, and its velocity and
acceleration there (added in step B2).

For a track that predicts along a line (`quadratic = false`, every
[`FittedSpec`](@ref)'s) this is `c_find + v_est (t − t_find)` exactly as
[`track_center`](@ref) evaluates it, `v_est` and zero. For a quadratic track
it is the prediction `pred` of the last find, blended into the prediction in
use before it, `prev`, over `[t_blend, t_blend + blend]`:

    c(t) = c_prev(t) + σ(s) (c_pred(t) − c_prev(t)),    s = (t − t_blend)/blend,

`σ` the quintic smoothstep, with the velocity and acceleration of that
expression — so that `c` and `ċ` are continuous at every find, which the
target and the gauge source read (`CODE.md`, "The tracks": continuity in time).
"""
function track_state(tr::HorizonTrack{T}, t) where {T}
    tt = T(t)
    if !tr.quadratic
        return (center_at(track_center(tr), tt), tr.v_est, zero(SVector{3,T}))
    end
    cn, vn, an = line_at(tr.pred, tt)
    tr.blend > 0 || return (cn, vn, an)
    s = (tt - tr.t_blend) / tr.blend
    s ≥ 1 && return (cn, vn, an)
    co, vo, ao = line_at(tr.prev, tt)
    σ, σ′, σ″ = _blend_weight(s)
    Δ = tr.blend
    dc, dv, da = cn - co, vn - vo, an - ao
    c = co + σ * dc
    ċ = vo + σ * dv + (σ′ / Δ) * dc
    c̈ = ao + σ * da + (2 * σ′ / Δ) * dv + (σ″ / (Δ * Δ)) * dc
    return (c, ċ, c̈)
end

"""
    track_position(tr::HorizonTrack, t) -> c

The center the track predicts at `t`: [`track_state`](@ref)'s first value, and
for a track along a line `center_at(track_center(tr), t)`, bit for bit.
"""
track_position(tr::HorizonTrack, t) = first(track_state(tr, t))

# The packed history's `k`-th find, `(t, x⃗)`.
@inline _hist_find(h::SVector{16,T}, k) where {T} =
    (h[4k - 3], SVector{3,T}(h[4k - 2], h[4k - 1], h[4k]))

# The history with the find `(t, x⃗)` appended and the oldest dropped beyond
# `TRACK_FIT_FINDS`.
function _push_find(h::SVector{16,T}, n::Int, t, x) where {T}
    rows = [_hist_find(h, k) for k in 1:n]
    push!(rows, (T(t), SVector{3,T}(x)))
    length(rows) > TRACK_FIT_FINDS && popfirst!(rows)
    m = length(rows)
    v = zeros(T, 16)
    for (k, (tk, xk)) in enumerate(rows)
        v[4k - 3] = tk
        v[(4k - 2):(4k)] .= xk
    end
    return SVector{16,T}(v), m
end

"""
    fit_track_quadratic(ts, xs, t0) -> TrackLine

The least-squares quadratic `c + v τ + ½ a τ²`, `τ = t − t0`, through the
finds `(ts[k], xs[k])`, one fit per axis (added in step B2): the normal
equations of the `n × 3` design matrix in `τ` scaled by the finds' span, so
that their condition does not depend on the cadence. Needs three distinct
times; `TRACK_FIT_FINDS = 4` is what the track gives it.
"""
function fit_track_quadratic(ts::AbstractVector{T}, xs::AbstractVector{SVector{3,T}},
                             t0) where {T}
    n = length(ts)
    n ≥ 3 || throw(ArgumentError(
        "a quadratic in time needs at least three finds, got $n."))
    span = maximum(ts) - minimum(ts)
    span > 0 || throw(ArgumentError(
        "the finds a quadratic is fitted to are all at t = $(ts[1])."))
    A = zeros(T, n, 3)
    for k in 1:n
        s = (ts[k] - T(t0)) / span
        A[k, 1] = one(T)
        A[k, 2] = s
        A[k, 3] = s * s / 2
    end
    N = SMatrix{3,3,T}(A' * A)
    coef = ntuple(3) do d
        b = SVector{3,T}(A' * [xs[k][d] for k in 1:n])
        N \ b
    end
    c = SVector{3,T}(coef[1][1], coef[2][1], coef[3][1])
    v = SVector{3,T}(coef[1][2], coef[2][2], coef[3][2]) / span
    a = SVector{3,T}(coef[1][3], coef[2][3], coef[3][3]) / (span * span)
    return TrackLine{T}(T(t0), c, v, a)
end

"""
    find_count(a, G, h, interval) -> n

How many finds an interval of length `interval` is divided into so that the
prediction's error at each, `|a| Δt²` for the acceleration `a` it predicts with,
stays at or below a quarter of a stencil's reach, `G h/4` (`CODE.md`, "The
tracks": a find cadence set by the predicted error; added in step B2): `1` for
`a = 0`, and otherwise `⌈interval / √(G h/(4|a|))⌉`.
"""
function find_count(a, G::Integer, h, interval)
    na = sqrt(sum(abs2, a))
    iszero(na) && return 1
    Δ = sqrt(G * h / (4 * na))
    return max(1, ceil(Int, interval / Δ - sqrt(eps(float(one(Δ))))))
end

"""
    find_interval(a, G, h; max) -> Δt

The find interval of [`find_count`](@ref): `max` divided into that many equal
intervals, so that the finds a chunk makes fall on its pieces and its last is
the record's (added in step B2).
"""
find_interval(a, G::Integer, h; max) = max / find_count(a, G, h, max)

"""
    TrackLostError(msg, track, records = [])

The run's horizon track has missed `max_misses` consecutive finds: the
geometry the layer is placed by is a prediction from a find that is too old
to trust, and the run ends. It carries the coasting `track` and — when
[`evolve!`](@ref) throws it — the analysis record up to and including the
row of the last miss, so that the run's account of *why* survives it (a
lost track ends the run; losing the record with it would be the failure
step 7 avoided by recording a failed find rather than throwing it).
"""
struct TrackLostError <: Exception
    msg::String
    track::Any
    records::Vector{Any}
end

TrackLostError(msg, track) = TrackLostError(msg, track, Any[])

Base.showerror(io::IO, e::TrackLostError) = print(io, "TrackLostError: ", e.msg)

# The lmax a case's track is kept at: the spec's, for a tracked case.
_track_lmax(case) = case.interior isa Union{FittedSpec,TrackedSpec} ?
                    case.interior.lmax_shape : 4

"""
    seed_track(case::GHCase, t; lmax = the case's lmax_shape) -> HorizonTrack

The track before anything has been found: the case's **analytic** center
`center_at(case.center, t)` and velocity `case.center.v`, the background's
analytic radii ([`horizon_min_radius`](@ref), [`horizon_max_radius`](@ref))
and its analytic shape ([`analytic_shape`](@ref)), `source = :analytic`,
no `hlm`.

**The seed's smallest radius is the analytic surface's, not the
`√(1 − v²)` bound (amended in step 8).** [`horizon_min_radius`](@ref) of a
boosted hole is the rest frame's smallest radius times the contraction
factor — a lower bound, exact when the smallest radius lies along the boost
(a boosted Schwarzschild hole) and not otherwise: harmonic Kerr at
`a = 7/10` boosted along `x̂`, G5's case, has its smallest radius `0.714`
on the spin axis, which the boost does not contract, and the bound says
`0.681`. The first find then reads `0.716` and is refused as a jump of
`0.034`, over half a stencil at `h = 5/256` (measured in step 8). So the seed
takes the larger of the bound and the least of
[`analytic_horizon_radius`](@ref) over [`shape_sample_directions`](@ref),
which includes both poles, for a hole that moves; a hole at rest keeps the
bound, which is then exact, bit for bit.

It is what the first chunk's geometry — and the initial data's core rule —
is built from, and what the first find is judged against: a first find
whose smallest radius is more than half a stencil from the analytic one is
refused like any other jump.

**A [`TrackedSpec`](@ref)'s seed predicts along a quadratic (added in step
B2)**, from the analytic center, the analytic velocity plus the spec's
`seed_velocity_offset` (zero but for the recovery study) and `acceleration`
(zero for one hole; what a binary's seed knows from its orbit), and a
[`FittedSpec`](@ref)'s along the line it always did. `quadratic` overrides it.
"""
function seed_track(case::GHCase{T}, t; lmax::Integer=_track_lmax(case),
                    quadratic::Bool=case.interior isa TrackedSpec && case.interior.quadratic,
                    acceleration=zero(SVector{3,T})) where {T}
    bg = case.background
    c = center_at(case.center, T(t))
    v = case.interior isa TrackedSpec ?
        case.center.v + case.interior.seed_velocity_offset : case.center.v
    tr = HorizonTrack{T}(T(t), c, v, _seed_r_min(bg, lmax),
                         T(horizon_max_radius(bg)), analytic_shape(bg, lmax),
                         Int(lmax), nothing, nothing, :analytic, 0, 0)
    quadratic || return tr
    l = TrackLine{T}(T(t), c, v, SVector{3,T}(acceleration))
    return _with_prediction(tr; quadratic=true, pred=l, prev=l, t_blend=T(t),
                            blend=zero(T))
end

"""
    seeded_track(T; t, center, velocity, acceleration = 0, r_min, r_max = r_min,
                 shape = sphere, lmax = 4, quadratic = true) -> HorizonTrack

A track seeded from numbers rather than a case (added in step B2): a hole at
`center` at time `t`, moving at `velocity` with `acceleration`, whose horizon is
the sphere of radius `r_min` unless a `shape` is given. What a test of the
prediction on synthetic finds needs, and a binary's seed will.
"""
function seeded_track(::Type{T}; t, center, velocity, acceleration=(0, 0, 0),
                      r_min, r_max=r_min, shape=nothing, lmax::Integer=4,
                      quadratic::Bool=true) where {T}
    c, v, a = SVector{3,T}(center...), SVector{3,T}(velocity...),
              SVector{3,T}(acceleration...)
    sh = shape === nothing ?
         [i == 1 ? Float64(r_min) * sqrt(4π) : 0.0 for i in 1:((lmax + 1)^2)] :
         collect(Float64, shape)
    tr = HorizonTrack{T}(T(t), c, v, T(r_min), T(r_max), sh, Int(lmax), nothing,
                         nothing, :analytic, 0, 0)
    quadratic || return tr
    l = TrackLine{T}(T(t), c, v, a)
    return _with_prediction(tr; quadratic=true, pred=l, prev=l, t_blend=T(t),
                            blend=zero(T))
end

function _seed_r_min(bg, lmax)
    bound = horizon_min_radius(bg)
    iszero(sum(abs2, hole_velocity(bg))) && return bound
    sampled = minimum(n -> analytic_horizon_radius(bg, n),
                      shape_sample_directions(lmax))
    return max(bound, sampled)
end

"""
    track_center(tr::HorizonTrack) -> HoleCenter

The tracked trajectory as a [`HoleCenter`](@ref):
`HoleCenter(c_find − v_est t_find, v_est)`, so that
`center_at(track_center(tr), t) = c_find + v_est (t − t_find)` and every
consumer of a center — the masks, `interior_radius`, the range projection's
gate, `core_position` — works on the tracked one without knowing it is.
"""
track_center(tr::HorizonTrack{T}) where {T} =
    HoleCenter{T}(tr.c_find - tr.v_est * tr.t_find, tr.v_est)

"""
    track_center(tr::HorizonTrack, t) -> HoleCenter

The line the track holds the center on near `t` (added in step B2): for a
track along a line, [`track_center`](@ref)`(tr)` itself; for a quadratic one,
the tangent at `t` of [`track_state`](@ref)'s curve, `HoleCenter(c(t) − ċ(t) t,
ċ(t))`, which is what a piece of a chunk carries the target and its layer along.
"""
function track_center(tr::HorizonTrack{T}, t) where {T}
    tr.quadratic || return track_center(tr)
    c, ċ, _ = track_state(tr, t)
    return HoleCenter{T}(c - ċ * T(t), ċ)
end

"""
    update_track(tr, hz, t; max_misses = 3, G, h, lmax = tr.lmax) -> HorizonTrack

The track after the find `hz` at time `t` — a named tuple with at least
`success`, and on success `origin`, `origin_r_min`, `origin_r_max`, `hlm`
and `grid` ([`find_gh_horizon`](@ref)'s, or the driver's horizon row):

- **`success === nothing`** — no find was attempted (the cadence skipped
  this chunk): the track unchanged.
- **`success === true`** — the found surface: `c_find` its origin, `r_min`,
  `r_max` its radii about that origin, `shape` its `hlm` made real at
  `lmax` ([`real_shape`](@ref)), `source = :found`, `misses = 0`. The
  velocity is `(origin − c_find)/(t − t_find)` — two finds' difference —
  once there is a previous find to difference against, i.e. when
  `nfinds ≥ 1` and the previous source was not `:analytic`; until then it is
  the seed's analytic velocity, which a boosted case knows and a first find
  cannot.
- **`success === false`** — a failed find: everything kept, `misses + 1`,
  `source = :coasting`. At `max_misses` consecutive misses it **throws a
  [`TrackLostError`](@ref)** saying how old the geometry is.

**A jump is refused** (an `ArgumentError`, naming both radii and `G h`):
a find whose `r_min` differs from the track's by more than `G h / 2`, half
a stencil reach at the geometry's spacing `h`. The layer's surfaces are
offsets of this one, so a jump of the smallest radius moves the core and
the layer by as much — exposing, on the side it moves away from, points
that no layer ever relaxed and that the frozen core held stale — and a real
horizon does not do that between two finds (proposed in step 8d).

**A quadratic track (added in step B2)** also refuses a find whose **origin**
lies more than `G h/2` from where the track predicted it ([`track_position`](@ref)
at `t`), with an `ArgumentError` naming both — the center jump `CODE.md`'s
"The tracks" adds beside the radius one, since the layer and the gauge source
move with the center — and then predicts:

- after the first find, from the found origin along the velocity and
  acceleration the last prediction gives there (the seed's);
- after the second and third, along the chord of the last two finds, its
  velocity carried to the newer one by the last prediction's acceleration,
  which it keeps (today's line where that is zero);
- from the fourth on, along the least-squares quadratic in time through the
  last [`TRACK_FIT_FINDS`](@ref) ([`fit_track_quadratic`](@ref)).

The new prediction takes over from the one in use over `interval` — by default
[`find_interval`](@ref) of its acceleration capped at `max_interval`, the next
find's distance — so that the center and its velocity are continuous at the
find. `v_est` is the new prediction's velocity at `t`.
"""
function update_track(tr::HorizonTrack{T}, hz, t; max_misses::Integer=3,
                      G::Integer, h, lmax::Integer=tr.lmax, interval=nothing,
                      max_interval=one(T)) where {T}
    success = hz.success
    success === nothing && return tr
    tt = T(t)
    if success === true
        origin = SVector{3,T}(Tuple(hz.origin))
        rmin = T(hz.origin_r_min)
        rmax = T(hz.origin_r_max)
        tol = T(G) * T(h) / 2
        abs(rmin - tr.r_min) ≤ tol || throw(ArgumentError(
            "the horizon found at t = $tt has its smallest coordinate radius " *
            "at r_min = $rmin about its origin, and the track's was " *
            "$(tr.r_min) (from the $(tr.source === :analytic ? "analytic seed" : "find at t = $(tr.t_find)")): " *
            "a jump of $(abs(rmin - tr.r_min)), more than half a stencil " *
            "reach G h/2 = $tol at G = $G, h = $h. The layer is an offset of " *
            "this surface, so a jump moves the core and the layer by as much " *
            "and exposes points no layer has treated. A horizon does not jump " *
            "between two finds; a finder that converged to another surface " *
            "does — check horizon_note and the seed before loosening this."))
        tr.quadratic && return _update_quadratic(tr, hz, tt, origin, rmin, rmax,
                                                 tol; G=G, h=h, lmax=lmax,
                                                 interval=interval,
                                                 max_interval=max_interval)
        v = if tr.nfinds ≥ 1 && tr.source !== :analytic && tt > tr.t_find
            (origin - tr.c_find) / (tt - tr.t_find)
        else
            tr.v_est
        end
        return HorizonTrack{T}(tt, origin, v, rmin, rmax,
                               real_shape(hz.hlm, hz.grid, lmax), Int(lmax),
                               Vector{ComplexF64}(hz.hlm), hz.grid, :found, 0,
                               tr.nfinds + 1)
    end
    new = _with_prediction(HorizonTrack{T}(tr.t_find, tr.c_find, tr.v_est, tr.r_min,
                                           tr.r_max, tr.shape, tr.lmax, tr.hlm,
                                           tr.grid, :coasting, tr.misses + 1,
                                           tr.nfinds);
                           quadratic=tr.quadratic, history=tr.history,
                           nhist=tr.nhist, pred=tr.pred, prev=tr.prev,
                           t_blend=tr.t_blend, blend=tr.blend)
    if new.misses ≥ max_misses
        note = hasproperty(hz, :note) && hz.note !== nothing ?
               " The last find said: $(hz.note)" : ""
        throw(TrackLostError(
            "the horizon track has missed $(new.misses) consecutive finds " *
            "(max_misses = $max_misses) and is lost: its geometry is from the " *
            "$(tr.source === :analytic || tr.nfinds == 0 ? "analytic seed" : "find") " *
            "at t = $(tr.t_find), $(tt - tr.t_find) M before t = $tt, " *
            "carried since along v_est = $(Tuple(tr.v_est)). The layer would be " *
            "placed by a prediction that stale, so the run ends here." * note,
            new))
    end
    return new
end

# A quadratic track's successful find (step B2): the center jump refused, the
# find appended to the history, the new prediction, and the blend from the
# prediction in use — its Taylor polynomial at `t`, which matches it in value,
# velocity and acceleration there whatever it was (a blend itself, if a find
# came before the last blend ended).
function _update_quadratic(tr::HorizonTrack{T}, hz, tt, origin, rmin, rmax, tol;
                           G, h, lmax, interval, max_interval) where {T}
    c_now, v_now, a_now = track_state(tr, tt)
    δ = sqrt(sum(abs2, origin - c_now))
    δ ≤ tol || throw(ArgumentError(
        "the horizon found at t = $tt has its origin at $(Tuple(origin)), and the " *
        "track predicted the center at $(Tuple(c_now)) (from the " *
        "$(tr.source === :analytic ? "analytic seed" : "find at t = $(tr.t_find)")): " *
        "a jump of $δ, more than half a stencil reach G h/2 = $tol at G = $G, " *
        "h = $h. The layer, its target and the gauge source all move with the " *
        "tracked center, so a jump moves them by as much (CODE.md, \"The " *
        "tracks\": the center-jump check). A hole does not jump between two " *
        "finds; a prediction that is off is a find cadence too long for the " *
        "acceleration, or a seed velocity that is wrong — shorten the interval " *
        "(max_interval) before loosening this."))
    hist, n = _push_find(tr.history, tr.nhist, tt, origin)
    # Before the fourth find the acceleration is the last prediction's own —
    # the seed's, or zero — and not the blend's, whose `c̈` carries the
    # turnover from one prediction to the next.
    _, v_pred, a_pred = line_at(tr.pred, tt)
    pred = if n ≥ TRACK_FIT_FINDS
        finds = [_hist_find(hist, k) for k in 1:n]
        fit_track_quadratic(first.(finds), last.(finds), tt)
    elseif n ≥ 2
        t1, x1 = _hist_find(hist, n - 1)
        Δt = tt - t1
        v = (origin - x1) / Δt + (Δt / 2) * a_pred
        TrackLine{T}(tt, origin, v, a_pred)
    else
        TrackLine{T}(tt, origin, v_pred, a_pred)
    end
    prev = TrackLine{T}(tt, c_now, v_now, a_now)
    Δ = interval === nothing ?
        T(find_interval(pred.a, G, T(h); max=T(max_interval))) : T(interval)
    base = HorizonTrack{T}(tt, origin, pred.v, rmin, rmax,
                           real_shape(hz.hlm, hz.grid, lmax), Int(lmax),
                           Vector{ComplexF64}(hz.hlm), hz.grid, :found, 0,
                           tr.nfinds + 1)
    return _with_prediction(base; quadratic=true, history=hist, nhist=n, pred=pred,
                            prev=prev, t_blend=tt, blend=Δ)
end

"""
    tracked_interior(spec::TrackedSpec, tr::HorizonTrack, t; mass, spin,
                     ρ_max = 0) -> Interior

The **tracked analytic target** at time `t` (added in step B2): step 5's
`:damped` sphere of `spec`'s radii about the track's line at `t`
([`track_center`](@ref)`(tr, t)`), relaxing toward the boosted, translated
Kerr-Schild hole of mass `mass · spec.mass_factor` and spin `spin` along `z` on
that line — the hole the track says is there, as an `isbits` target metric
([`layer_target`](@ref)). The perturbations of step B2's sensitivity study move
the target and its sphere together: `center_offset` is added to the line's
center and `velocity_offset` to its velocity, the line then passing through
the offset center at `t`.

The driver builds one per piece of a chunk at the piece's midpoint, so that
the target follows the track as a moving `:fitted` target is refilled
([`evolve!`](@ref)); it is an `Interior` like any other, so its core follows
its target with the target's rate ([`core_follows`](@ref)) and its layer
carries `(1 − w) ∂_t u_target` ([`background_state_rate`](@ref)), the target's
own rate along its line. A track seeded from the analytic answer, at the
seed's time and unperturbed, gives the case's own sphere and background, value
for value.
"""
function tracked_interior(spec::TrackedSpec{T}, tr::HorizonTrack, t; mass, spin,
                          ρ_max=zero(T)) where {T}
    tt = T(t)
    line = track_center(tr, tt)
    c = center_at(line, tt) + spec.center_offset
    v = line.v + spec.velocity_offset
    center = HoleCenter{T}(c - v * tt, v)
    ks = KerrSchild(T(mass) * spec.mass_factor, T(spin))
    # `0 − v`, not `−v`: a zero component stays `+0`, as the case's own
    # `boost(…, (−v, 0, 0))` has it, so that the seed's target is the
    # background's bits (a `−0` turns the sign of zeros in the boost's matrix).
    target = SpacetimeMetrics.translate(SpacetimeMetrics.boost(ks, zero(v) - v),
                                        SVector{4,T}(zero(T), (c - v * tt)...))
    return Interior(T; center=center, r_0=spec.r_0, r_1=spec.r_1, ρ_max=ρ_max,
                    variant=:damped, w_ramp=spec.w_ramp, ρ_ramp=spec.ρ_ramp,
                    margin=spec.margin, target=target)
end

# --- the geometry from a track ------------------------------------------------

# The coarsest spacing among the leaves meeting the annulus `lo ≤ r ≤ hi`
# about `c`, and how many there were.
function _annulus_spacing(forest::Forest{3}, ::Type{T}, c, lo, hi) where {T}
    h = zero(T)
    n = 0
    for k in forest.leaves
        _box_meets_annulus(block_extent(T, forest, k), c, lo, hi) || continue
        h = max(h, spacing(T, forest, k))
        n += 1
    end
    return h, n
end

"""
    fitted_interior(spec::FittedSpec, tr::HorizonTrack, forest, G; t, n_L,
                    ρ_max = 0) -> FittedInterior

The layer on the tracked geometry for the mesh `forest` at time `t`: the
track's shape about [`track_center`](@ref)`(tr)`, its bounding radii
([`shape_bounds`](@ref)), and the two lengths the rule states in spacings —
`offset = m h` and `thickness = n_L h` — with `h` **the coarsest spacing
among the blocks the layer lives in**, the annulus
`[r_in − (m + n_L) h, r_out − m h]` about `c(t)`, found by one iteration
from `minimum_spacing(forest)`.

**The spacing is the layer's and not the margin's (proposed in step 8d).**
`PLAN.md` asks for the annulus out to `r_max`, the horizon itself; on the
suite's fixture the horizon at `2 M` lies in blocks twice as coarse as the
layer's, and a spacing read there asks for a layer that does not fit inside
it at all. So `h` is measured where [`layer_spacing`](@ref) measures step
5's sphere — in the blocks the profiles are resolved on — and what the
margin buys along its actual path through coarser blocks is reported in
e-folds, at each block's own spacing, by [`margin_efolds`](@ref).

`n_L` has no default here: step 8c's rule ([`layer_cells`](@ref)) needs the
scheme's `G` and the run's rate, which [`evolve!`](@ref) knows. `ρ_max` is a
placeholder, as it is in a case's [`Interior`](@ref); the driver sets it
per chunk ([`with_ρ_max`](@ref)).

It **refuses** `r_in − (m + n_L + core_min) h ≤ 0` by name: the core would
be smaller than `core_min` spacings, or the offset surface would reach the
center. [`check_interior_radii`](@ref) is still what asserts the result
against the mesh, as it asserts the sphere.
"""
function fitted_interior(spec::FittedSpec{T,V,X}, tr::HorizonTrack, forest,
                         G::Integer; t, n_L::Integer=spec.n_L,
                         ρ_max=zero(T)) where {T,V,X}
    n_L > 0 || throw(ArgumentError(
        "the ramp n_L must be resolved before a geometry is built: 0 in a " *
        "FittedSpec means step 8c's rule max(4G, ⌈G (10 ρ_max M)^{1/3}⌉), " *
        "which needs the scheme's G and the run's rate — evolve! resolves " *
        "it with layer_cells(G, ρ_max, M)."))
    L = spec.lmax_shape
    NM = (L + 1)^2
    center = track_center(tr)
    sv = SVector{NM,T}(ntuple(i -> T(_resize_shape(tr.shape, L)[i]), NM))
    r_in, r_out = shape_bounds(sv, L)
    c = center_at(center, T(t))
    m = spec.margin
    h0 = T(minimum_spacing(T, forest))
    h, nb = _annulus_spacing(forest, T, c, r_in - (m + n_L) * h0,
                             r_out - m * h0)
    nb > 0 || throw(ArgumentError(
        "no block of this forest meets the tracked layer's annulus " *
        "[$(r_in - (m + n_L) * h0), $(r_out - m * h0)] around $(Tuple(c)): " *
        "the tracked horizon is outside the domain."))
    need = (m + n_L + spec.core_min) * h
    r_in - need > 0 || throw(ArgumentError(
        "the tracked horizon's smallest radius r_in = $r_in cannot hold the " *
        "layer at this mesh's spacing h = $h: the margin m = $m, the ramp " *
        "n_L = $n_L and the smallest core core_min = $(spec.core_min) need " *
        "(m + n_L + core_min) h = $need of it (r_in − (m + n_L + core_min) h " *
        "≤ 0). Refine around the hole — every one of the three is stated in " *
        "spacings — or lower the margin toward its floor G + 1."))
    return FittedInterior(T; center=center, shape=sv, lmax=L, offset=m * h,
                          thickness=n_L * h, ρ_max=ρ_max, variant=V,
                          w_ramp=spec.w_ramp, ρ_ramp=spec.ρ_ramp, margin=m,
                          n_L=n_L, h=h, target=spec.target, r_in=r_in,
                          r_out=r_out)
end

"""
    surface_shift(a::FittedInterior, b::FittedInterior, t) -> length

How far the core surface of `b` is from that of `a` at time `t`: the largest
difference of their core radii along [`shape_sample_directions`](@ref), plus
the distance between their centers — what the driver compares with half a
spacing to decide whether the gauge source, which is sampled with the core
rule of the geometry its problem was built with, must be re-sampled
(proposed in step 8d).
"""
function surface_shift(a::FittedInterior{T}, b::FittedInterior{T}, t) where {T}
    ca = center_at(a.center, T(t))
    cb = center_at(b.center, T(t))
    δc = sqrt(sum(abs2, ca - cb))
    worst = zero(T)
    for n in shape_sample_directions(max(a.lmax, b.lmax))
        nn = SVector{3,T}(n)
        ra = (shape_radius(a, nn) - a.offset) - a.thickness
        rb = (shape_radius(b, nn) - b.offset) - b.thickness
        worst = max(worst, abs(ra - rb))
    end
    return worst + δc
end

# Two spheres (step B2): their core radii's difference and their centers'
# distance at `t` — the tracked analytic target's spheres move with the track
# and keep their radii.
function surface_shift(a::Interior{T}, b::Interior{T}, t) where {T}
    δc = sqrt(sum(abs2, center_at(a.center, T(t)) - center_at(b.center, T(t))))
    return abs(a.r_0 - b.r_0) + δc
end

# --- what the margin buys: step 8a's leakage, along the geometry --------------
#
# `CODE.md`, "The interior" (the leakage margin, step 8a) and "Kreiss–Oliger
# dissipation": grid-scale content made at the offset surface leaks outward
# attenuated by `e^{−n_e}`, `n_e = ∫_{r_1}^{r_h} ε dr/(h ℓ_max,1(r))`, with
# `ℓ_max,1` the frozen-coefficient penetration length at `ε_KO = 1` along a
# grid axis. `test/dispersion.jl` computes it for the analytic holes (its
# section 1c); this is its closed form along an axis, moved into `src/` so
# that a tracked run reports what its margin buys, at the spacing of each
# block the path crosses (`PLAN.md`, step 8d's hand-over from step 8c).

"""
    axis_dispersion(q) -> (; θ, sp, sq, κ)

The package's own order-`q` weights as Fourier symbols along a grid axis, on
a grid of phases `θ = kh`: `s′(θ)` of the first derivative, `(√c)′(θ)` of
the compact second one, and the dissipation's `κ(θ) = sin^{2r}(θ/2)` — what
`test/dispersion.jl`'s `stencil_symbols` and `ellmax_axis` are built from,
on 512 uniform phases and 64 logarithmic ones down to `10⁻³`, where the
sonic point's supremum moves.
"""
function axis_dispersion(q::Integer)
    w1 = derivative_weights(Float64, Val(Int(q)), Val(1))
    w2 = derivative_weights(Float64, Val(Int(q)), Val(2))
    r1 = Int(q) ÷ 2
    rk = Int(q) ÷ 2 + 1
    NT = 512
    θs = sort(vcat([π * i / NT for i in 1:NT],
                   exp.(range(log(1e-3), log(π / NT); length=64))))
    sp = [sum(j * w1[j + r1 + 1] * cos(j * θ) for j in (-r1):r1) for θ in θs]
    sq = map(θs) do θ
        c = -sum(w2[j + r1 + 1] * cos(j * θ) for j in (-r1):r1)
        c′ = sum(j * w2[j + r1 + 1] * sin(j * θ) for j in (-r1):r1)
        c′ / (2 * sqrt(c))
    end
    κ = [sin(θ / 2)^(2rk) for θ in θs]
    return (θ=θs, sp=sp, sq=sq, κ=κ)
end

# `ℓ_max` at `ε_KO = 1` for advection `b` and wave speed `a` along an axis:
# the larger branch's outgoing group velocity over the dissipation,
# `max_θ (−b s′ + a |(√c)′|)⁺ / κ`.
function _ellmax_axis(D, b, a)
    best = 0.0
    for i in eachindex(D.θ)
        vg = -b * D.sp[i] + a * abs(D.sq[i])
        vg > 0 || continue
        best = max(best, vg / D.κ[i])
    end
    return best
end

"""
    margin_efolds(background, int::FittedInterior, q; t, ε_KO, spacing,
                  per = 8) -> (; min, per_direction)

The e-folds step 8a's frozen-coefficient model gives grid-scale content
crossing the tracked geometry's margin, from the offset surface `r_1(n̂)`
out to the horizon `r_h(n̂)`, along the six grid-axis directions `±x̂, ±ŷ,
±ẑ` from the tracked center — the directions in which the one-dimensional
symbol is exact — with the coefficients `b = β·n̂`, `a = α√(γ^{nn})` of the
`background` at each point, `ε_KO` there (a number or a profile,
[`dissipation_rate`](@ref)), and the spacing `spacing(x)` of the block each
point is in (a function, or a number). `per` samples per cell, midpoint
rule. A direction whose path meets the chart's singular set is `NaN`.

It is a **lower bound** on what any packet gets, as the model's is
(`CODE.md`, "Kreiss–Oliger dissipation": the fixture's `1.81` e-folds across
eight cells against the `2.9–4.6` measured); `min` is the smallest over the
six directions, which is what a row of step 8f's matrix carries.
"""
function margin_efolds(background, int::FittedInterior{T}, q::Integer; t,
                       ε_KO, spacing, per::Integer=8) where {T}
    D = axis_dispersion(q)
    c = center_at(int.center, T(t))
    hof(x) = spacing isa Number ? Float64(spacing) : Float64(spacing(x))
    out = Float64[]
    for d in 1:3, s in (1, -1)
        n = SVector{3,Float64}(ntuple(k -> k == d ? Float64(s) : 0.0, 3))
        rh = tofloat64(shape_radius(int, SVector{3,T}(n)))
        r = rh - tofloat64(int.offset)
        acc = 0.0
        while r < rh
            x0 = SVector{3,Float64}(ntuple(k -> tofloat64(c[k]) + r * n[k], 3))
            hl = hof(x0)
            dr = min(hl / per, rh - r)
            rm = r + dr / 2
            x = ntuple(k -> tofloat64(c[k]) + rm * n[k], 3)
            h4, _, _ = background_state(background, tofloat64(T(t)), x)
            if !all(isfinite, h4)
                acc = NaN
                break
            end
            _, _, α, β, γu, _ = metric_quantities(_sym4(h4))
            b = s * β[d]
            a = α * sqrt(γu[d, d])
            ℓ = _ellmax_axis(D, b, a)
            ε = Float64(dissipation_rate(ε_KO, T(t), x))
            ℓ > 0 && (acc += ε * dr / (hl * ℓ))
            r += dr
        end
        push!(out, acc)
    end
    finite = filter(isfinite, out)
    return (min=isempty(finite) ? NaN : minimum(finite), per_direction=out)
end
