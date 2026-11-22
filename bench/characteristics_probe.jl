# Where the characteristics point inside a just-formed common horizon (step B7).
# `test/binary_runs.jl char_probe=1` includes this file and wraps the run's
# common finder and its finder: after every accepted common find, and after
# every find of the remnant once the pair has been handed over, it reads the
# state on surfaces at fixed depths below the found surface and prints, per
# depth, how fast the outgoing light front moves through that surface.
#
# The question (Erik, 2026-11-21): the remnant's layer failed when it was
# handed a surface whose radii were `1.43–2.49` (`CODE.md`, "The common horizon
# and the handover", "When."); whether the characteristics just inside such a
# surface point outward somewhere, so that the layer's relaxation reaches the
# exterior, and how soon after formation they no longer do.
#
# At a point `x` of the surface `r = R(θ, φ) − d` about the found origin, with
# `ŝ` its outward Euclidean unit normal, the characteristic speeds of the
# second-order system along `ŝ` are `λ± = −β^i ŝ_i ± α |ŝ|_γ` and
# `λ₀ = −β^i ŝ_i`, as coordinate speeds along `ŝ`. The surface itself moves at
# `v_n = ∂_t R / |∇f|` along `ŝ`, `f = |x − o| − R(θ, φ)`, from the previous
# accepted surface (`NaN` at the first). `u₊ = λ₊ − v_n` is the outgoing front
# relative to the surface: `u₊ < 0` everywhere on a layer's outer surface means
# nothing inside it reaches outside it. Every point is tagged with the region
# of the run's interior it lies in: evolved, a hole's layer, or a hole's core.
#
# One line per depth (`chars:`) and one CSV row per depth in `out/chars.csv`.
# The probe reads the state the find has just filled and changes nothing the
# run carries.

using AbstractSphericalHarmonics: EquiangularGrid, ash_evaluate, ash_lmax, ash_phis,
                                  ash_resample, ash_thetas
using LinearAlgebra: dot, inv, norm
using Printf: @printf
using StaticArrays: SMatrix, SVector

const CHAR_DEPTHS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5)

mutable struct CharProbe{F,C}
    find::F
    common::C
    csv::String
    nθ::Int
    nφ::Int
    last::Any                 # (t, table, origin) of the last surface probed
end

CharProbe(find, common; csv, nθ=33, nφ=64) =
    CharProbe(find, common, String(csv), Int(nθ), Int(nφ), nothing)

# The surface's radius on a fine table, read bilinearly (`common_find_probe.jl`'s).
function _char_table(origin, grid, hlm)
    fine = EquiangularGrid(4 * ash_lmax(grid) + 3)
    h = real.(ash_evaluate(fine, ash_resample(fine, hlm, grid, 0), 0))
    return (; h, θs=ash_thetas(fine), φs=ash_phis(fine),
            o=SVector{3,Float64}(Tuple(origin)))
end

function _char_R(tab, θ, φ)
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

_char_angles(y) = (r = norm(y); (acos(clamp(y[3] / r, -1, 1)), atan(y[2], y[1])))

# f = |x − o| − R(θ, φ) about the table's origin.
_char_f(tab, x) = (y = x - tab.o; (θ, φ) = _char_angles(y); norm(y) - _char_R(tab, θ, φ))

# Which region of the run's interior `x` lies in: 0 evolved, A a hole's layer,
# −A a hole's core.
function _char_region(int, t, x)
    holes = int isa TreeGeneralizedHarmonic.HoleInteriors ? int.holes :
            int isa TreeGeneralizedHarmonic.FittedInterior ? (int,) : ()
    for (A, h) in enumerate(holes)
        g = TreeGeneralizedHarmonic.interior_point(h, t, x)
        g.r < g.r_0 && return -A
        g.r < g.r_1 && return A
    end
    return 0
end

function _char_probe!(cp::CharProbe, p::GHProblem{T,G,q}, t, src, origin, grid,
                      hlm) where {T,G,q}
    tab = _char_table(origin, grid, hlm)
    prev = cp.last
    cp.last = (t=Float64(t), tab=tab)
    θs = [π * (i - 1 / 2) / cp.nθ for i in 1:cp.nθ]
    φs = [2π * (j - 1) / cp.nφ for j in 1:cp.nφ]
    δ = 0.02
    newfile = !isfile(cp.csv)
    io = open(cp.csv, "a")
    newfile && println(io, "t,src,depth,npts,u_max,u_min,frac_out,lam_max,vn_max," *
                           "vn_min,alpha_min,n_evolved,n_layer,n_core,u_max_evolved," *
                           "theta_worst,phi_worst,r_worst")
    for d in CHAR_DEPTHS
        xs = SVector{3,Float64}[]
        for θ in θs, φ in φs
            n̂ = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
            r = _char_R(tab, θ, φ) - d
            r > 0.05 && push!(xs, tab.o + r * n̂)
        end
        isempty(xs) && continue
        vals = gh_interpolate(p.U, xs; q=q)
        u_max = -Inf; u_min = Inf; nout = 0; lam_max = -Inf
        vn_max = -Inf; vn_min = Inf; α_min = Inf; u_ev = -Inf
        nreg = [0, 0, 0]; worst = (NaN, NaN, NaN)
        for (k, x) in enumerate(xs)
            g = vals[k]
            # The state's first ten slots are `h = g_ab − η_ab` (`pointwise.jl`).
            gtt = g[1] - 1
            βl = SVector(g[2], g[3], g[4])
            γ = SMatrix{3,3}(1 + g[5], g[6], g[7], g[6], 1 + g[8], g[9], g[7], g[9],
                             1 + g[10])
            γi = inv(γ)
            βu = γi * βl
            α2 = -gtt + dot(βl, βu)
            α = α2 > 0 ? sqrt(α2) : NaN
            ∇f = SVector(ntuple(i -> (_char_f(tab, x + δ * SVector(ntuple(==(i), 3))) -
                                      _char_f(tab, x - δ * SVector(ntuple(==(i), 3)))) /
                                     (2δ), 3))
            s = ∇f / norm(∇f)
            λp = -dot(βu, s) + α * sqrt(dot(s, γi * s))
            vn = NaN
            if prev !== nothing && t > prev.t
                (θ, φ) = _char_angles(x - tab.o)
                y = x - prev.tab.o
                (θp, φp) = _char_angles(y)
                vn = (_char_R(tab, θ, φ) - _char_R(prev.tab, θp, φp)) /
                     (Float64(t) - prev.t) / norm(∇f)
            end
            u = isnan(vn) ? λp : λp - vn
            reg = _char_region(p.interior, T(t), SVector{3,T}(x))
            nreg[reg == 0 ? 1 : reg > 0 ? 2 : 3] += 1
            reg == 0 && (u_ev = max(u_ev, u))
            if u > u_max
                u_max = u
                (θw, φw) = _char_angles(x - tab.o)
                worst = (θw, φw, norm(x - tab.o))
            end
            u_min = min(u_min, u)
            u > 0 && (nout += 1)
            lam_max = max(lam_max, λp)
            isnan(vn) || (vn_max = max(vn_max, vn); vn_min = min(vn_min, vn))
            α_min = min(α_min, α)
        end
        n = length(xs)
        vn_max == -Inf && (vn_max = vn_min = NaN)
        @printf("  chars: t %.4f %-7s d %.3f  u₊ %+.4f … %+.4f  out %5.1f%%  λ₊max %+.4f  v_n %+.4f … %+.4f  α_min %.4f  ev/lay/core %d/%d/%d  u₊max(ev) %+.4f  worst θ %.3f φ %.3f r %.3f\n",
                Float64(t), src, d, u_min, u_max, 100 * nout / n, lam_max, vn_min, vn_max,
                α_min, nreg..., u_ev, worst...)
        println(io, join((Float64(t), src, d, n, u_max, u_min, nout / n, lam_max, vn_max,
                          vn_min, α_min, nreg..., u_ev, worst...), ","))
    end
    close(io)
    return nothing
end

# The common finder, wrapped: probe after every accepted common find.
function (cp::CharProbe)(p, u, t, inds, masses, geom, seed, ch)
    row = cp.common(p, u, t, inds, masses, geom, seed, ch)
    if row.success === true && row.hlm !== nothing && row.grid !== nothing
        _char_probe!(cp, p, t, "common", row.origin, row.grid, row.hlm)
    end
    return row
end

# The finder, wrapped: after the handover the composite has one piece, and
# every successful find is the remnant's.
struct CharFind{P}
    cp::P
end

function (cf::CharFind)(p, u, t; kwargs...)
    out = cf.cp.find(p, u, t; kwargs...)
    int = p.interior
    single = int isa TreeGeneralizedHarmonic.HoleInteriors && length(int.holes) == 1
    if single && out.success === true
        _char_probe!(cf.cp, p, t, "remnant", out.origin, out.grid, out.hlm)
    end
    return out
end
