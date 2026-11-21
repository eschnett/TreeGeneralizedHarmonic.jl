# The common horizon of several holes (step B7; `CODE.md`, "The common horizon
# and the handover"): where to look, which surface to accept, and what to record.
# It observes the run and changes nothing in it until the handover exists.

"""
    CommonHorizon(T = Float64; d_c = 4, N = 25, atol = 0, atol_factor = 10,
                  atol_floor = 1e-9, enclose = 1/20, maxiters = 1000, spin = true,
                  streak = 3, unif_tol = 1e-8)

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
end

function CommonHorizon(::Type{T}=Float64; d_c=4, N::Integer=25, atol=0,
                       atol_factor=10, atol_floor=1.0e-9, enclose=1 // 20,
                       maxiters::Integer=1000,
                       spin::Bool=true, streak::Integer=3,
                       unif_tol=1.0e-8) where {T}
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
    return CommonHorizon{T}(T(d_c), Int(N), Float64(atol), Float64(atol_factor),
                            Float64(atol_floor), Float64(enclose), Int(maxiters), spin, Int(streak),
                            Float64(unif_tol))
end

# The run state the common finder carries from one row to the next, and in a
# checkpoint: the shape the next attempt is seeded with, the current streak of
# accepted finds of nondecreasing area, its first time and its last area, and the
# formation time. Plain data (`Float64`, `Int`, `nothing`, the shape's vector).
common_state0() = (hlm=nothing, streak=0, area=nothing, t_first=nothing,
                   t_formed=nothing)

# The state after a row whose attempt was `row`, at `t`.
function common_state_after(st, row, t, ch::CommonHorizon)
    if row.success === true
        grows = st.area === nothing || row.area ≥ st.area
        streak = grows ? st.streak + 1 : 1
        t_first = grows && st.t_first !== nothing ? st.t_first : Float64(t)
        t_formed = st.t_formed !== nothing ? st.t_formed :
                   streak ≥ ch.streak ? t_first : nothing
        return (hlm=row.hlm, streak=streak, area=row.area, t_first=t_first,
                t_formed=t_formed)
    end
    return (hlm=nothing, streak=0, area=nothing, t_first=nothing,
            t_formed=st.t_formed)
end

# The row of a run that asks for no attempt at this row, or of one that made it.
const NO_COMMON = (attempted=false, success=nothing, note=nothing, iters=nothing,
                   H_norm=nothing, atol=nothing, from=nothing, origin=nothing,
                   r_min=nothing, r_mean=nothing, r_max=nothing, area=nothing,
                   M_irr=nothing, J=nothing, M_ch=nothing, depth=nothing,
                   core_gap=nothing, time=nothing, hlm=nothing)

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
            time=(time_ns() - t0) / 1e9, hlm=ok ? out.hlm : nothing)
end

# The record's row: the attempt without its shape, and the run's streak and
# formation after it.
common_record(row, st) =
    (; Base.structdiff(row, NamedTuple{(:hlm,)})..., streak=st.streak,
     t_formed=st.t_formed)

# The state as a checkpoint gives it back.
plain_common_state(s) =
    (hlm=s.hlm === nothing ? nothing : Vector{ComplexF64}(s.hlm), streak=Int(s.streak),
     area=s.area === nothing ? nothing : Float64(s.area),
     t_first=s.t_first === nothing ? nothing : Float64(s.t_first),
     t_formed=s.t_formed === nothing ? nothing : Float64(s.t_formed))
