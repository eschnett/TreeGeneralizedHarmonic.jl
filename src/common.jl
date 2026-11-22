# The common horizon of several holes (step B7; `CODE.md`, "The common horizon
# and the handover"): where to look, which surface to accept, and what to record.
# It observes the run and changes nothing in it until the handover exists.

"""
    CommonHorizon(T = Float64; d_c = 4, N = 25, atol = 0, atol_factor = 10,
                  atol_floor = 1e-9, enclose = 1/20, maxiters = 1000, spin = true,
                  streak = 3, unif_tol = 1e-8, handover = false,
                  handover_from = -Inf, remnant_depth = 1, handover_aspect = 7/4)

What [`evolve!`](@ref)'s `common_horizon` keyword asks of a case of several holes:
at every record row whose tracks are at most `d_c` apart (in the case's units),
one attempt to find a horizon that encloses every hole, at the finder's
resolution `N`. It is not part of the case and not part of the recipe a
checkpoint holds: it observes, and a restart may switch it on, off or change it.

The attempt starts from the previous accepted common surface if the row before
found one, and otherwise from a sphere about the holes' mass-weighted center that
encloses every found individual surface, `enclose` (a fraction) beyond the
farthest of them. **It is accepted only on four checks of this package's own**,
because the finder cannot tell a missing horizon from a slow one (measured in
step B7; `CODE.md`, "The common horizon and the handover"):

- the finder's convergence at a positive tolerance, `atol` if it is positive and
  otherwise `atol_factor` times the worse residual `|H|` of the holes' own finds
  but at least `atol_floor` — at `atol = 0`, which the holes' own finds pass, a
  stall is reported as a success. The floor is there because a common surface's
  residual bottoms out higher than a hole's: on step H4's pair the holes' finds
  reached `2–6·10⁻¹⁴` and the common ones `1–5·10⁻¹³`, within 2 % of ten times
  the holes', while every stalled attempt stopped at `0.03` or above;
- a finite residual;
- every hole's found surface strictly inside it, at every collocation point;
- its origin between the holes, on the segment between the first two found
  origins' projections.

A failure is a row (`success = false` and a `note`), never an exception.
The Korzyński spin (`spin`) is computed only on an accepted surface. The common
horizon has *formed* at the first of `streak` successive accepted finds whose
area does not decrease; the row carries that time from then on.

**With `handover = true`** the run then hands the pair over to one remnant (step
B7; `CODE.md`, "The common horizon and the handover"): at the first row after
formation whose common surface holds both holes' cores (a positive core gap),
is round enough — its largest radius about its origin at most `handover_aspect`
times its smallest — and whose fit is a metric, the two holes' layers, tracks
and fits are replaced by one `:fitted` hole about the common horizon. Until then
it observes, as without it. The bound `7/4` is step B7's measurement on step
H4's pair: a remnant handed a `1.88 : 1` surface failed within a `M`, and every
one handed a `1.74 : 1` or rounder surface held (`CODE.md`, "The common horizon
and the handover").

Two diagnostics (step B7's handover at formation, `CODE.md`, "The common horizon
and the handover"): `handover_from` holds the handover back until that time,
and `remnant_depth` makes the remnant's margin and ramp that many times the
holes', so that its layer is as deep at a finer spacing.
"""
struct CommonHorizon{T}
    d_c::T
    N::Int
    atol::Float64
    atol_factor::Float64
    atol_floor::Float64
    enclose::Float64
    maxiters::Int
    spin::Bool
    streak::Int
    unif_tol::Float64
    handover::Bool
    handover_from::Float64
    remnant_depth::Int
    handover_aspect::Float64
end

function CommonHorizon(::Type{T}=Float64; d_c=4, N::Integer=25, atol=0,
                       atol_factor=10, atol_floor=1.0e-9, enclose=1 // 20,
                       maxiters::Integer=1000,
                       spin::Bool=true, streak::Integer=3,
                       unif_tol=1.0e-8, handover::Bool=false, handover_from=-Inf,
                       remnant_depth::Integer=1, handover_aspect=7 // 4) where {T}
    T(d_c) > 0 || throw(ArgumentError(
        "d_c is the tracks' separation below which a common horizon is looked " *
        "for, and must be positive, got $d_c."))
    N ≥ 4 || throw(ArgumentError(
        "N is the finder's resolution and must be at least 4, got $N; step B7 " *
        "measured N = 16 and N = 25 to agree to 10⁻⁴ in the area at formation."))
    (atol ≥ 0 && atol_factor > 0 && atol_floor ≥ 0) || throw(ArgumentError(
        "atol and atol_floor must not be negative and atol_factor must be " *
        "positive, got $atol, $atol_floor and $atol_factor: a common find needs a " *
        "positive tolerance, since the finder reports a stall at atol = 0 as a " *
        "success."))
    enclose ≥ 0 || throw(ArgumentError(
        "enclose widens the seed sphere beyond the farthest hole's surface, as a " *
        "fraction, and must not be negative, got $enclose."))
    (maxiters ≥ 1 && streak ≥ 1) || throw(ArgumentError(
        "maxiters and streak must be positive, got $maxiters and $streak."))
    remnant_depth ≥ 1 || throw(ArgumentError(
        "remnant_depth multiplies the remnant's margin and ramp, and must be at " *
        "least 1, got $remnant_depth."))
    handover_aspect ≥ 1 || throw(ArgumentError(
        "handover_aspect bounds the common surface's largest radius over its " *
        "smallest, which is at least 1, got $handover_aspect."))
    return CommonHorizon{T}(T(d_c), Int(N), Float64(atol), Float64(atol_factor),
                            Float64(atol_floor), Float64(enclose), Int(maxiters),
                            spin, Int(streak), Float64(unif_tol), handover,
                            Float64(handover_from), Int(remnant_depth),
                            Float64(handover_aspect))
end

# The run state the common finder carries from one row to the next, and in a
# checkpoint: the shape the next attempt is seeded with, the current streak of
# accepted finds of nondecreasing area, its first time and its last area, the
# formation time, and the last accepted surface's least radius and time — what
# the remnant's growth rate is first measured from. Plain data (`Float64`, `Int`,
# `nothing`, the shape's vector).
common_state0() = (hlm=nothing, streak=0, area=nothing, t_first=nothing,
                   t_formed=nothing, r_min=nothing, t_last=nothing)

# The state after a row whose attempt was `row`, at `t`.
function common_state_after(st, row, t, ch::CommonHorizon)
    if row.success === true
        grows = st.area === nothing || row.area ≥ st.area
        streak = grows ? st.streak + 1 : 1
        t_first = grows && st.t_first !== nothing ? st.t_first : Float64(t)
        t_formed = st.t_formed !== nothing ? st.t_formed :
                   streak ≥ ch.streak ? t_first : nothing
        return (hlm=row.hlm, streak=streak, area=row.area, t_first=t_first,
                t_formed=t_formed, r_min=row.r_min, t_last=Float64(t))
    end
    return (hlm=nothing, streak=0, area=nothing, t_first=nothing,
            t_formed=st.t_formed, r_min=nothing, t_last=nothing)
end

# The row of a run that asks for no attempt at this row, or of one that made it.
const NO_COMMON = (attempted=false, success=nothing, note=nothing, iters=nothing,
                   H_norm=nothing, atol=nothing, from=nothing, origin=nothing,
                   r_min=nothing, r_mean=nothing, r_max=nothing, area=nothing,
                   M_irr=nothing, J=nothing, M_ch=nothing, depth=nothing,
                   core_gap=nothing, time=nothing, hlm=nothing, grid=nothing)

# A surface's radius `h(θ, φ)` about its origin, resampled onto a fine grid and
# read bilinearly: a test of which side of the surface a point lies on, accurate
# far below the margins it decides. (Summing the modes at every point cost a
# minute an attempt in step B7's probe.)
function surface_table(grid, hlm)
    fine = EquiangularGrid(4 * ash_lmax(grid) + 3)
    h = real.(ash_evaluate(fine, ash_resample(fine, Vector{ComplexF64}(hlm), grid, 0),
                           0))
    return (; h, θs=ash_thetas(fine), φs=ash_phis(fine))
end

function surface_radius_at(tab, θ, φ)
    θs, φs, h = tab.θs, tab.φs, tab.h
    i = clamp(searchsortedlast(θs, θ), 1, length(θs) - 1)
    a = clamp((θ - θs[i]) / (θs[i+1] - θs[i]), 0, 1)
    Δφ = φs[2] - φs[1]
    x = mod(φ - φs[1], 2π) / Δφ
    j = floor(Int, x)
    b = x - j
    j1 = mod(j, length(φs)) + 1
    j2 = mod(j + 1, length(φs)) + 1
    return (1 - a) * ((1 - b) * h[i, j1] + b * h[i, j2]) +
           a * ((1 - b) * h[i+1, j1] + b * h[i+1, j2])
end

# The depth of the point `x` below the surface `(origin, tab)`: positive inside.
function surface_depth(tab, origin::SVector{3,Float64}, x::SVector{3,Float64})
    y = x - origin
    r = sqrt(sum(abs2, y))
    r == 0 && return Inf
    return surface_radius_at(tab, acos(clamp(y[3] / r, -1, 1)), atan(y[2], y[1])) - r
end

# The least depth of a set of points below the surface.
least_depth(tab, origin, xs) =
    minimum(x -> surface_depth(tab, origin, SVector{3,Float64}(x)), xs)

# A hole's core surface, sampled along the directions of `grid`: the tracked
# geometry's `r_0(n̂)` about its center at `t`, and its depth below the horizon it
# was built from (`offset + thickness`), or `nothing` for a geometry that is not a
# tracked offset of a horizon.
function core_surface(int::FittedInterior, t, grid)
    c = center_at(int.center, t)
    pts = map(CartesianIndices(ash_grid_size(grid))) do ij
        θ, φ = ash_point_coord(grid, ij)
        n = SVector{3,Float64}(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
        r_0 = tofloat64((shape_radius(int, n) - int.offset) - int.thickness)
        SVector{3,Float64}(tofloat64(c[1]), tofloat64(c[2]), tofloat64(c[3])) + r_0 * n
    end
    return pts, tofloat64(int.offset + int.thickness)
end
core_surface(int, t, grid) = nothing

"""
    common_horizon_row(p, u, t, inds, masses, geom, seed, ch::CommonHorizon)

One attempt at the common horizon of a case of several holes at `t` (step B7),
after the holes' own finds `inds` (`horizon_row`'s rows, one per hole, with their
surfaces), and its record row. `masses` weight the seed sphere's center, `geom`
is the holes' composite geometry for the core gap, and `seed` is the previous
accepted common shape or `nothing`. On rank 0 inside the row's analysis: it
fills and interpolates through `find_gh_horizon`'s two served steps only.

The row: whether it was `attempted` and whether it `success`fully passed every
check of [`CommonHorizon`](@ref) (a `note` says which failed); the finder's
`iters`, `H_norm` and the `atol` it was given; where it started (`from`,
`:sphere` or `:last`); the surface's `origin`, its radii about it, `area`,
`M_irr`, and `J` and `M_ch` on an accepted one; per hole the least `depth` of the
hole's own surface below it and, for fitted holes, the `core_gap` — the least
depth of the hole's core surface below the common surface moved inward by the
holes' own core depth, which is positive where a common layer of the same depth
would hold the hole's core inside its own; the attempt's wall `time`; and `hlm`,
the shape, which the run carries as the next seed and the record drops.
"""
function common_horizon_row(p::GHProblem{T}, u, t, inds, masses, geom, seed,
                            ch::CommonHorizon) where {T}
    K = length(inds)
    for A in 1:K
        inds[A].success === true || return merge(NO_COMMON, (
            note="hole $A's own find failed this row, so there is no surface to " *
                 "enclose",))
    end
    os = [SVector{3,Float64}(Tuple(inds[A].origin)) for A in 1:K]
    ms = [Float64(m) for m in masses]
    c = sum(ms .* os) / sum(ms)
    r_enc = (1 + ch.enclose) * maximum(A -> sqrt(sum(abs2, os[A] - c)) +
                                            Float64(inds[A].origin_r_max), 1:K)
    H_ind = maximum(A -> Float64(inds[A].H_norm), 1:K)
    atol = ch.atol > 0 ? ch.atol : max(ch.atol_factor * H_ind, ch.atol_floor)
    (isfinite(atol) && atol > 0) || return merge(NO_COMMON, (
        attempted=true, success=false,
        note="the holes' own residuals give no positive tolerance (|H| = $H_ind)",))
    from = seed === nothing ? :sphere : :last
    t0 = time_ns()
    found = try
        locate_gh_horizon(p, u, t; N=ch.N, r_seed=r_enc, origin=c, center=c,
                          hlm=seed, atol=atol, maxiters=ch.maxiters)
    catch e
        e isa InterruptException && rethrow()
        return merge(NO_COMMON, (attempted=true, success=false, atol=atol, from=from,
                                 note=first(split(sprint(showerror, e), '\n')),
                                 time=(time_ns() - t0) / 1e9))
    end
    out, result, provider = found
    o = SVector{3,Float64}(out.origin)
    # The checks, in the order of `CommonHorizon`'s list; the depths are computed
    # whenever the surface is finite, so that a failed row still says how close
    # it came.
    tab = isfinite(out.H_norm) ? surface_table(out.grid, out.hlm) : nothing
    depth = tab === nothing ? nothing :
            ntuple(A -> least_depth(tab, o, horizon_points(SVector{3,Float64}(os[A]),
                                                          inds[A].grid, inds[A].hlm)),
                   K)
    e12 = os[2] - os[1]
    s = sum((o - os[1]) .* e12) / sum(abs2, e12)
    note = if !out.success
        "not converged: |H| = $(out.H_norm) above atol = $atol after $(out.iters) " *
        "iterations"
    elseif !isfinite(out.H_norm)
        "the residual is not finite"
    elseif !all(>(0), depth)
        "it does not enclose hole $(findfirst(≤(0), depth)) (least depth " *
        "$(minimum(depth)))"
    elseif !(0 ≤ s ≤ 1)
        "its origin is not between the holes (at $(s) of the way from hole 1 to 2)"
    else
        nothing
    end
    ok = note === nothing
    ok && ch.spin && (out = with_horizon_spin(out, result, provider, ch.unif_tol))
    # The core gap of fitted holes: each hole's core surface against the common
    # surface moved inward by the deepest of the holes' core depths.
    gap = nothing
    if tab !== nothing && geom isa HoleInteriors
        cores = [core_surface(geom.holes[A], T(t), out.grid) for A in 1:K]
        if all(!isnothing, cores)
            D = maximum(last, cores)
            gap = ntuple(A -> least_depth(tab, o, first(cores[A])) - D, K)
        end
    end
    return (attempted=true, success=ok, note=note, iters=out.iters,
            H_norm=out.H_norm, atol=atol, from=from, origin=Tuple(o),
            r_min=out.origin_r_min, r_mean=out.origin_r_mean, r_max=out.origin_r_max,
            area=out.area, M_irr=out.M_irr, J=ok && ch.spin ? out.J : nothing,
            M_ch=ok && ch.spin ? out.M_ch : nothing, depth=depth, core_gap=gap,
            time=(time_ns() - t0) / 1e9, hlm=ok ? out.hlm : nothing,
            grid=ok ? out.grid : nothing)
end

# The record's row: the attempt without its shape, and the run's streak and
# formation after it.
common_record(row, st) =
    (; Base.structdiff(row, NamedTuple{(:hlm, :grid)})..., streak=st.streak,
     t_formed=st.t_formed)

# The state as a checkpoint gives it back.
_plain_float(s, k) = !haskey(s, k) || s[k] === nothing ? nothing : Float64(s[k])
plain_common_state(s) =
    (hlm=s.hlm === nothing ? nothing : Vector{ComplexF64}(s.hlm), streak=Int(s.streak),
     area=_plain_float(s, :area), t_first=_plain_float(s, :t_first),
     t_formed=_plain_float(s, :t_formed), r_min=_plain_float(s, :r_min),
     t_last=_plain_float(s, :t_last))

# --- the handover ---------------------------------------------------------------------

"""
    remnant_track(t, row, tracks, masses; lmax) -> HorizonTrack

The remnant's track at the handover (step B7): seeded from the accepted common
surface `row` at `t` — its origin, its radii about it, its shape at the holes'
degree `lmax` — moving with the holes' mass-weighted velocity and acceleration
at `t` (`track_state` of each hole's track), and predicting along the quadratic
with that acceleration from this one find, as a hole's track does after its
first.
"""
function remnant_track(t, row, tracks, masses; lmax::Integer)
    T = eltype(first(tracks).c_find)
    tt = T(t)
    ms = [T(m) for m in masses]
    M = sum(ms)
    states = [track_state(tr, tt) for tr in tracks]
    v = sum(ms[A] * states[A][2] for A in eachindex(ms)) / M
    a = sum(ms[A] * states[A][3] for A in eachindex(ms)) / M
    c = SVector{3,T}(row.origin)
    hist, n = _push_find(zero(SVector{16,T}), 0, tt, c)
    base = HorizonTrack{T}(tt, c, v, T(row.r_min), T(row.r_max),
                           real_shape(row.hlm, row.grid, lmax), Int(lmax),
                           Vector{ComplexF64}(row.hlm), row.grid, :found, 0, 1)
    l = TrackLine{T}(tt, c, v, a)
    return _with_prediction(base; quadratic=true, history=hist, nhist=n, pred=l,
                            prev=l, t_blend=tt, blend=zero(T))
end

"""
    remnant_bounds(bounds, int::FittedInterior) -> StateBounds

The ranges the remnant's fit is projected into (step B7): the union of the holes'
own ([`derive_target_bounds`](@ref)'s, from their analytic data on their offset
surfaces), with the gate radius at the remnant's offset surface. There is no
analytic remnant to derive them from, and the remnant's offset surface lies
outside the holes' horizons, where the values are milder than on theirs.
"""
function remnant_bounds(bounds, int::FittedInterior{T}) where {T}
    bs = collect(bounds)
    return StateBounds{T}(minimum(b -> b.α_min, bs), maximum(b -> b.α_max, bs),
                          minimum(b -> b.λ_min, bs), maximum(b -> b.λ_max, bs),
                          maximum(b -> b.β_max, bs), maximum(b -> b.K_max, bs),
                          int.r_in - int.offset)
end

# The finds a remnant's growing horizon asks for over `span` (step B7): its
# least radius may move at most `G h/4` between two finds — half of what the
# track's jump guard refuses — at the growth rate `rate` measured from its last
# two finds.
growth_finds(rate, G, h, span) =
    rate > 0 ? ceil(Int, span * rate / (G * h / 4) - sqrt(eps(typeof(span)))) : 1
