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

A plain immutable struct, and [`update_track`](@ref) returns a new one: the
track is data the driver passes along, never a field that changes under a
kernel. [`track_center`](@ref) is what a kernel sees of it.
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
end

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

# The lmax a case's track is kept at: the spec's, for a fitted case.
_track_lmax(case) = case.interior isa FittedSpec ? case.interior.lmax_shape : 4

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
"""
function seed_track(case::GHCase{T}, t; lmax::Integer=_track_lmax(case)) where {T}
    bg = case.background
    c = center_at(case.center, T(t))
    return HorizonTrack{T}(T(t), c, case.center.v, _seed_r_min(bg, lmax),
                           T(horizon_max_radius(bg)), analytic_shape(bg, lmax),
                           Int(lmax), nothing, nothing, :analytic, 0, 0)
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
"""
function update_track(tr::HorizonTrack{T}, hz, t; max_misses::Integer=3,
                      G::Integer, h, lmax::Integer=tr.lmax) where {T}
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
    new = HorizonTrack{T}(tr.t_find, tr.c_find, tr.v_est, tr.r_min, tr.r_max,
                          tr.shape, tr.lmax, tr.hlm, tr.grid, :coasting,
                          tr.misses + 1, tr.nfinds)
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
