# How ApparentHorizonFinder behaves where there is no common horizon yet, and what
# a failed attempt costs (step B7). The data are an equal-mass Brill–Lindquist
# pair of total mass one, `γ = ψ⁴ δ` with `ψ = 1 + Σ m_A / 2|x − c_A|` and `K = 0`,
# the punctures at `±d/2` on the `z` axis. It has a common horizon below a
# critical separation: about `1.53` in units of one hole's mass in the
# literature, `0.766 M` of the total; `threshold` bisects it.
#
# The provider refuses a point within `r_ex` of a puncture, as the grid's
# footprint guard refuses a point whose stencil reaches a layer, and logs each
# call: one call per fast-flow iteration plus one for the area.
#
#     julia --project=. bench/common_horizon_probe.jl [trace] [sweep] [threshold]
#
# Options `key=value`: `r_ex=` (default 0.15, half an individual horizon's
# coordinate radius 1/4; 0 turns the guard off), `maxiters=` (1000).

using ApparentHorizonFinder
using AbstractSphericalHarmonics
using LinearAlgebra
using Printf
using StaticArrays

const SV = SVector{3,Float64}

struct BLPair
    m::NTuple{2,Float64}
    c::NTuple{2,SV}
    r_ex::Float64
    calls::Vector{NamedTuple}
end

BLPair(d; r_ex=0.15) =
    BLPair((0.5, 0.5), (SV(0, 0, d / 2), SV(0, 0, -d / 2)), r_ex, NamedTuple[])

function admvars(p::BLPair, x::SV)
    ψ = 1.0
    dψ = zero(SV)
    for A in 1:2
        y = x - p.c[A]
        r = norm(y)
        r < p.r_ex && throw(ArgumentError(@sprintf("a point at %.3f from puncture %d is inside r_ex = %.3f", r, A, p.r_ex)))
        ψ += p.m[A] / (2r)
        dψ -= p.m[A] / (2r^3) * y
    end
    ψ4 = ψ^4
    γ = ψ4 * SMatrix{3,3,Float64}(I)
    ∂γ = SArray{Tuple{3,3,3},Float64}((i == j ? 4ψ^3 * dψ[k] : 0.0) for i in 1:3, j in 1:3, k in 1:3)
    return ADMVars(γ, ∂γ, zero(SMatrix{3,3,Float64}))
end

function (p::BLPair)(Xs::AbstractArray)
    c = sum(Xs) / length(Xs)
    push!(p.calls, (; npts=length(Xs), centroid=c,
        rmean=sum(x -> norm(x - c), Xs) / length(Xs),
        dmin=ntuple(A -> minimum(x -> norm(x - p.c[A]), Xs), 2)))
    return map(x -> admvars(p, x), Xs)
end

# h(θ, φ) of a result's shape, from its modes.
function shape_at(res, θ, φ)
    grid = res.grid
    l_max = ash_lmax(grid)
    s = zero(ComplexF64)
    for l in 0:l_max, m in (-l):l
        s += res.hlm[ash_mode_index(grid, 0, l, m)] * sYlm(0, l, m, θ, φ)
    end
    return real(s)
end

# Depth of `x` below the surface of `res`: positive inside.
function depth(res, x::SV)
    y = x - res.origin
    r = norm(y)
    θ = acos(clamp(y[3] / r, -1, 1))
    φ = atan(y[2], y[1])
    return shape_at(res, θ, φ) - r
end

# The individual horizons, found about their punctures with no guard (they do not
# reach it) and a tight tolerance.
function individual(d, N)
    out = map(1:2) do A
        p = BLPair(d; r_ex=0.0)
        find_horizon(p, p.c[A], N, 0.25, 1e-10, 1000; verbosity=0)
    end
    return out
end

function classify(res, p::BLPair, inds; tol_floor=1e-8)
    !isfinite(res.H_norm) && return :nan
    enc = map(inds) do ind
        minimum(x -> depth(res, x), horizon_points(ind))
    end
    if res.success && res.H_norm ≤ tol_floor
        # (−1e-8: at a small `d` the "individual" find is the common surface)
        all(>(-1e-8), enc) && return :common
        any(>(-1e-8), enc) && return :one_hole
        return :other_surface
    end
    res.success && return :stall_as_success
    return :not_converged
end

function attempt(d, N, r_seed; atol=0.0, maxiters=1000, r_ex=0.15, inds=individual(d, N),
    verbosity=0)
    p = BLPair(d; r_ex)
    t0 = time_ns()
    res = nothing
    err = nothing
    try
        res = find_horizon(p, SV(0, 0, 0), N, r_seed, atol, maxiters; verbosity)
    catch e
        e isa InterruptException && rethrow()
        err = e
    end
    t = (time_ns() - t0) / 1e9
    if err !== nothing
        kind = err isa ArgumentError ? :refused : Symbol(nameof(typeof(err)))
        return (; d, N, r_seed, atol, kind, iters=length(p.calls), calls=length(p.calls),
            H=NaN, area=NaN, origin=NaN, t, p, res, note=first(split(sprint(showerror, err), '\n')))
    end
    kind = classify(res, p, inds)
    return (; d, N, r_seed, atol, kind, iters=res.iters, calls=length(p.calls), H=res.H_norm,
        area=res.area, origin=res.origin[3], t, p, res, note="")
end

function row(a)
    @printf("%5.2f %3d %6.3f %7.0e  %-17s %5d %5d  %9.2e %8.4f %+7.3f %8.3f s\n",
        a.d, a.N, a.r_seed, a.atol, a.kind, a.iters, a.calls, a.H, a.area, a.origin, a.t)
end
header() = println("    d   N r_seed    atol  outcome           iters calls      |H|     area  origin_z      time")

function trace(; r_ex, maxiters)
    for d in (0.7, 0.8, 1.6, 4.0), atol in (0.0, 1e-9)
        println("\n# trace: d = $d, N = 16, seed d/2 + 0.35, atol = $atol, r_ex = $r_ex")
        a = attempt(d, 16, d / 2 + 0.35; atol, maxiters, r_ex)
        header(); row(a)
        a.note == "" || println("  note: ", a.note)
        println("  call  centroid_z  ⟨r⟩      dmin1    dmin2")
        n = length(a.p.calls)
        for (i, c) in enumerate(a.p.calls)
            (i ≤ 12 || i > n - 6 || i % 25 == 0) || continue
            @printf("  %4d  %+8.4f  %7.4f  %7.4f  %7.4f\n", i, c.centroid[3], c.rmean, c.dmin...)
        end
    end
end

function sweep(; r_ex, maxiters)
    println("\n# sweep (r_ex = $r_ex)")
    header()
    for N in (12, 16, 25)
        # reference: a successful individual find from a nearby sphere
        p = BLPair(8.0; r_ex=0.0)
        t0 = time_ns()
        r = find_horizon(p, p.c[1], N, 0.3, 0.0, maxiters; verbosity=0)
        @printf("# reference: individual hole at d = 8, N = %d: %d iterations, |H| %.1e, %.3f s\n",
            N, r.iters, r.H_norm, (time_ns() - t0) / 1e9)
        for d in (0.4, 0.6, 0.7, 0.75, 0.76, 0.77, 0.8, 0.9, 1.0, 1.5, 2.0, 4.0, 8.0)
            inds = individual(d, N)
            for r_seed in (d / 2 + 0.35, max(1.0, d / 2 + 0.6), 2.0 + d / 2), atol in (0.0, 1e-9)
                row(attempt(d, N, r_seed; atol, maxiters, r_ex, inds))
            end
        end
    end
end

function threshold(; r_ex, maxiters)
    println("\n# threshold (N = 16, atol = 1e-9, seed d/2 + 0.35)")
    found(d) = attempt(d, 16, d / 2 + 0.35; atol=1e-9, maxiters, r_ex).kind === :common
    lo, hi = 0.5, 1.0
    @assert found(lo) && !found(hi)
    for _ in 1:20
        mid = (lo + hi) / 2
        found(mid) ? (lo = mid) : (hi = mid)
    end
    @printf("common horizon up to d = %.4f M (bisection between %.6f and %.6f)\n", lo, lo, hi)
end

function main(args)
    opts = Dict{String,String}()
    sections = String[]
    for a in args
        if occursin('=', a)
            k, v = split(a, '='; limit=2)
            opts[k] = v
        else
            push!(sections, a)
        end
    end
    isempty(sections) && (sections = ["trace", "sweep", "threshold"])
    r_ex = parse(Float64, get(opts, "r_ex", "0.15"))
    maxiters = parse(Int, get(opts, "maxiters", "1000"))
    # A self-check of `shape_at` against the grid's own evaluation.
    let res = find_horizon(BLPair(1.0; r_ex=0.0), SV(0, 0, 0), 12, 0.8, 1e-10, 1000; verbosity=0)
        h = real.(ash_evaluate(res.grid, res.hlm, 0))
        err = maximum(CartesianIndices(h)) do ij
            θ, φ = ash_point_coord(res.grid, ij)
            abs(shape_at(res, θ, φ) - h[ij])
        end
        @assert err < 1e-10 "shape_at disagrees with ash_evaluate by $err"
    end
    "trace" in sections && trace(; r_ex, maxiters)
    "sweep" in sections && sweep(; r_ex, maxiters)
    "threshold" in sections && threshold(; r_ex, maxiters)
    return nothing
end

main(ARGS)
