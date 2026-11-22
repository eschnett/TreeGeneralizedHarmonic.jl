# How the horizon finder behaves on a pair's grid data where there is no common
# horizon yet, and what an attempt costs (step B7). `test/binary_runs.jl
# common_probe=<t>` includes this file and passes `evolve!` the finder wrapped
# by `CommonProbe`. From `t` on, after the two holes' own finds at a time, it
# makes common attempts from a sphere about the holes' midpoint that encloses
# both found surfaces: `N = 16` at `atol = 0` (how the run's finds are made),
# `N = 16` and `N = 25` at `atol` ten times the worse of the two individual
# residuals, and, once a common surface has been accepted, `N = 25` from that
# surface. It prints one line per attempt (`common:`) and one per pair of
# individual finds (`indiv:`). The attempts read the state the individual finds
# read and change nothing the run carries, so the run is the run without them.
#
# An attempt's outcome:
#   common       the finder's success, |H| ≤ the attempt's atol (or, at atol = 0,
#                ≤ 10× the individual finds' worse |H|), and both individual
#                surfaces at positive depth below it
#   one_hole     the same, but it encloses one individual surface only
#   stalled      `success` at atol = 0 with |H| above that bound: AHF reports a
#                stall at atol = 0 as a success
#   failed       `success = false` (atol not reached, maxiters, or the mean radius
#                below 1e-6)
#   refused      the finder threw: the footprint guard (a point's stencil reaches a
#                hole's layer) or another error, named in the note
#
# The equal masses of step H4's pair are assumed: the sphere's center is the
# midpoint of the two found origins.

using AbstractSphericalHarmonics: EquiangularGrid, ash_evaluate, ash_lmax, ash_phis,
                                  ash_resample, ash_thetas
using LinearAlgebra: norm
using Printf: @printf
using StaticArrays: SVector

mutable struct CommonProbe{F}
    find::F
    t_from::Float64
    pending::Any                 # (t, out) of the first hole's find at this time
    accepted::Any                # the last accepted common result, or nothing
end

CommonProbe(find; t_from) = CommonProbe(find, Float64(t_from), nothing, nothing)

# The surface of `res` resampled onto a fine grid, read bilinearly: a test of
# which side a point lies on, accurate to far below the margins it is used for.
# (Summing `sYlm` over the modes at every point cost a minute an attempt.)
function _shape_table(res)
    fine = EquiangularGrid(4 * ash_lmax(res.grid) + 3)
    h = real.(ash_evaluate(fine, ash_resample(fine, res.hlm, res.grid, 0), 0))
    return (; h, θs=ash_thetas(fine), φs=ash_phis(fine))
end

function _shape_at(tab, θ, φ)
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

# The least depth of the surface `ind` below the surface `res` (positive: inside).
function _min_depth(res, ind)
    tab = _shape_table(res)
    pts = TreeGeneralizedHarmonic.horizon_points(ind.origin, ind.grid, ind.hlm)
    return minimum(pts) do x
        y = x - res.origin
        r = norm(y)
        _shape_at(tab, acos(clamp(y[3] / r, -1, 1)), atan(y[2], y[1])) - r
    end
end

function (cp::CommonProbe)(p, u, t; kwargs...)
    t0 = time_ns()
    out = cp.find(p, u, t; kwargs...)
    dt = (time_ns() - t0) / 1e9
    tt = Float64(t)
    if cp.pending !== nothing && cp.pending[1] == tt
        first_out = cp.pending[2]
        cp.pending = nothing
        try
            _probe!(cp, p, u, t, (first_out, (; out..., time=dt)))
        catch e
            e isa InterruptException && rethrow()
            println("common: t = $tt  probe error: ", first(split(sprint(showerror, e), '\n')))
        end
        flush(stdout)
    else
        cp.pending = (tt, (; out..., time=dt))
    end
    return out
end

function _probe!(cp, p, u, t, inds)
    tt = Float64(t)
    tt < cp.t_from && return nothing
    o1, o2 = SVector{3,Float64}(inds[1].origin), SVector{3,Float64}(inds[2].origin)
    c = (o1 + o2) / 2
    sep = norm(o1 - o2)
    @printf("indiv: t = %.5f  sep %.5f  r_max %.4f %.4f  |H| %.2e %.2e  success %s %s  time %.2f %.2f s\n",
            tt, sep, inds[1].r_max, inds[2].r_max, inds[1].H_norm, inds[2].H_norm,
            inds[1].success, inds[2].success, inds[1].time, inds[2].time)
    all(o -> o.success, inds) || return nothing
    # A sphere about the midpoint enclosing both found surfaces, 5 % out.
    r_enc = 1.05 * maximum(o -> norm(SVector{3,Float64}(o.origin) - c) + o.origin_r_max, inds)
    H_ind = maximum(o -> o.H_norm, inds)
    atol = 10 * H_ind
    attempts = Any[(16, 0.0, nothing), (16, atol, nothing), (25, atol, nothing)]
    cp.accepted === nothing || push!(attempts, (25, atol, cp.accepted))
    for (N, a, seed) in attempts
        t0 = time_ns()
        res = nothing
        note = ""
        try
            res = find_gh_horizon(p, u, t; N=N, r_seed=r_enc, origin=c, center=c,
                                  hlm=seed === nothing ? nothing : seed.hlm,
                                  spin=false, atol=a, maxiters=1000)
        catch e
            e isa InterruptException && rethrow()
            note = first(split(sprint(showerror, e), '\n'))
        end
        dt = (time_ns() - t0) / 1e9
        from = seed === nothing ? "sphere" : "last"
        if res === nothing
            @printf("common: t = %.5f  N %2d  atol %.1e  from %-6s refused       %7.3f s  %s\n",
                    tt, N, a, from, dt, first(note, 160))
            continue
        end
        bound = a > 0 ? a : 10 * H_ind
        depths = (_min_depth(res, inds[1]), _min_depth(res, inds[2]))
        kind = if res.success && isfinite(res.H_norm) && res.H_norm ≤ bound
            all(>(0), depths) ? :common : any(>(0), depths) ? :one_hole : :other
        elseif res.success
            :stalled
        else
            :failed
        end
        kind === :common && N == 25 && (cp.accepted = res)
        @printf("common: t = %.5f  N %2d  atol %.1e  from %-6s %-9s iters %4d  |H| %.2e  area %.5f  M_irr %.6f  r %.4f–%.4f  depth %+.4f %+.4f  origin (%+.4f, %+.4f, %+.4f)  %7.3f s\n",
                tt, N, a, from, kind, res.iters, res.H_norm, res.area, res.M_irr,
                res.origin_r_min, res.origin_r_max, depths..., res.origin..., dt)
    end
    return nothing
end
