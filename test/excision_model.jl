# The host-side analysis of steps X1 and X5: whether a static hole can be
# excised on this mesh — one-sided closures per stencil at a lego surface
# inside the horizon — before a kernel is written (X1, Kerr-Schild `a = 0`),
# and which closure holds the spinning hole's frame-dragged faces, where the
# shift points into the excised set (X5, Kerr-Schild `a = 3/5`). Run by hand,
# with its numbers recorded in `CODE.md` under "Excision" and, in "Measured
# results", "Excision: the analysis (step X1)" and "Excision: the
# frame-dragged faces (step X5)".
#
# It is a **script and not a test**, in the manner of `test/dispersion.jl`,
# whose model it builds on: it prints Markdown tables, loads no `Test`, and
# its checks on itself — the frozen coefficients against Kerr-Schild's closed
# form, the radial line's layer against `dispersion.jl`'s recorded numbers to
# their printed digits, the parity sectors (X5: the turns' sectors) against
# the full operator, X5's turned coefficients against the metric and its
# general operator against X1's at `a = 0` — throw when they fail. The exact
# claims about the closures and the extrapolation (where they read, what they
# are exact on, the dissipation's sign, the tables' rounding) are
# `test/stencils_tests.jl`'s. It builds no mesh: every model is a sparse
# matrix assembled on the host from the package's own closure and
# extrapolation weights (`src/stencils.jl`) and the background's
# coefficients, read through `background_state`, `metric_quantities` and
# `metric_derivatives` as the kernel reads them.
#
# Sections, selected by name (all three when none is named); `section=part,
# part` selects parts of one, and `key=value` overrides a part's defaults
# (`q=2,4`, `n=24,32`, `eps=1/2,1`, `rE=…`, `ratios=…`, `h=…`, `m=…`,
# `t_end=…`, `n_fine=…`, `fam=axis,axislop,extrap,extraplop`, and for X5's
# parts `a=3/5` and `fam=…,hybrid,hybridlop,hybridadv,hybridadvlop`). Each
# table in `CODE.md` is one of these commands, with the time it took on the
# development machine (Apple silicon, 12 threads) at a load of 10–40, or on
# one of Symmetry's 64-core EPYC nodes (`amdq`) where it says so — there a
# command was one process per `(q, ε, r_E)` or per family, each a few
# threads, all at once:
#
#     julia --project=. --threads=4 test/excision_model.jl margins=margins       # 20 s
#     julia --project=. --threads=4 test/excision_model.jl model1d=frozen        # 20 s
#     julia --project=. --threads=4 test/excision_model.jl model1d=reflect       # 30 s
#     julia --project=. --threads=4 test/excision_model.jl model1d=radial        # 20 s
#     julia --project=. --threads=4 test/excision_model.jl model2d=eig           # 32 min
#     julia --project=. --threads=4 test/excision_model.jl model2d=controls      # 13 min
#     julia --project=. --threads=4 test/excision_model.jl model2d=noise         # 4 min
#     julia --project=. --threads=4 test/excision_model.jl model2d=fine          # 36 min
#
#     julia --project=. --threads=4 test/excision_model.jl margins=window        # 20 s
#     julia --project=. --threads=2 test/excision_model.jl model2d=spinfaces     # 10 s
#     julia --project=. --threads=4 test/excision_model.jl model2d=spineig n=24  # 22 min
#     julia --project=. --threads=2 test/excision_model.jl model2d=spineig n=32 q=… eps=… rE=…
#                                                       # Symmetry, 28 processes: 8 min
#     julia --project=. --threads=3 test/excision_model.jl model2d=spineig n=48 q=4 eps=1/2 rE=… fam=…
#                                                       # Symmetry, 21 processes: 14 min
#     julia --project=. --threads=4 test/excision_model.jl model2d=spincontrols q=… n=… rE=…
#                                                       # Symmetry, 8 processes: 10 min
#     julia --project=. --threads=64 test/excision_model.jl model2d=spinnoise    # Symmetry: 5 min
#     julia --project=. --threads=64 test/excision_model.jl model2d=spinfine     # Symmetry: 23 min
#
#   * `margins` — `margins`: the outflow condition on the seed's offset
#     surfaces of Kerr-Schild `a = 0, 3/5, 9/10` and harmonic Kerr `a = 7/10,
#     9/10`, at depths of `m` cells for the octant runs' spacings `h = 1/16 …
#     1/48`: the margin along the true normal and its least value, the
#     per-axis ratios `b/a` over the lego surface's closure faces (their
#     distribution, the inflow-like fraction, predicted `r_E/(2M)` for
#     Kerr-Schild `a = 0`, and the fraction whose shift points into the
#     excised set), the clearance of the chart's singular set, and the answer
#     per chart. `window` (step X5): Kerr-Schild `a = 3/5`'s window for the
#     sphere and the tracked offset surface at `h = 1/24, 1/32, 1/48` — the
#     normal outflow, the depth in cells at the poles and on the equator, the
#     room for the core rule's sphere outside the ring, X1's faces and X2b's
#     closure axes with the fractions whose shift points into the excised set,
#     and whether X5's rule finds its extrapolation's sources in 3D.
#   * `model1d` — `frozen`: the constant-coefficient system of
#     `dispersion.jl` on a half-line closed at its left end, for `b/a` from
#     `−5/4` (both characteristics entering) through `0 … 1` (one, inflow
#     along the axis) to `2` (outflow), the closure's reach, the dissipation's
#     three closures and centered against lopsided advection: the dense
#     semi-discrete spectrum's largest `Re λ`, its zero modes, where its mode
#     lives, and the RK4 step the closures allow. `reflect`: a packet into the
#     closure and what comes back (amplitude, phase, group velocity).
#     `radial`: `dispersion.jl`'s variable-coefficient radial line of
#     Kerr-Schild `a = 0`, its layer replaced by a closure at `r_E`: the
#     leakage ripple and a pulse into the surface, what crosses the horizon,
#     against the layer.
#   * `model2d` — the go/no-go: one component's principal part, with the
#     coefficient-gradient terms that make it a conservative wave equation,
#     on Kerr-Schild's equatorial plane, with a lego circle of radius `r_E`:
#     per-axis closures against per-stencil extrapolation along the lattice
#     direction nearest the normal, each with and without lopsided advection,
#     at `ε_KO = 1/2, 1`, `r_E = M/2 … 7M/4`, `q = 2, 4`, against the
#     `:damped` layer. `eig`: the dense spectrum at `49²` and `65²` points by
#     parity sector. `controls`: the same for the variations the go/no-go is
#     read against (no dissipation, the other dissipation closures, other
#     extrapolation degrees, a bare frozen core). `noise`: noise evolutions to
#     `100 M` at `129²`; `fine`: the same at `257²`. Step X5's parts are the
#     same on Kerr-Schild `a = 3/5`'s plane, where frame dragging puts faces
#     with the shift pointing into the excised set on the lego circle, with
#     two more families — the hybrids, per-axis where the shift points out
#     and the extrapolation where it points in, for every operator along that
#     axis or for the advection alone — over `r_E = 0.65 … 1.7`: `spinfaces`
#     (the faces at every resolution), `spineig` (`n = 24, 32`, and `48` for
#     three families), `spincontrols`, `spinnoise` (`129²`) and `spinfine`
#     (`257²`).
#
# What the models leave out is what `dispersion.jl`'s leave out — the source
# terms, the coupling between components, a third dimension — and what they
# keep is the question X1 and X5 ask: whether the closures, including the
# lego staircase's inflow-like and frame-dragged ones, are stable.

import Printf
using LinearAlgebra: BLAS, Diagonal, eigen, eigvals, mul!, norm, svdvals
using Random: MersenneTwister
using SparseArrays: sparse
using StaticArrays: SVector
import SpacetimeMetrics as SM
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: _sym4, closure_derivative_weights,
                               closure_dissipation_weights, lopsided_weights,
                               lagrange_derivative_weights, extrapolation_weights

# Dense eigenvalues use BLAS; the noise runs use the Julia threads instead.
BLAS.set_num_threads(max(1, Threads.nthreads()))

say(fmt, args...) = (println(Printf.format(Printf.Format(fmt), args...)); flush(stdout))
fmt(fmtstr, args...) = Printf.format(Printf.Format(fmtstr), args...)

# --- options ------------------------------------------------------------------

const SECTIONS = ("margins", "model1d", "model2d")
const OPTS = Dict{String,String}()
const RUN = String[]
for a in ARGS
    if occursin('=', a)
        k, v = split(a, '='; limit=2)
        OPTS[String(k)] = String(v)
        k in SECTIONS && push!(RUN, String(k))
    else
        a in SECTIONS || error("unknown section $a; the sections are $SECTIONS")
        push!(RUN, a)
    end
end
isempty(RUN) && append!(RUN, SECTIONS)

# A part of a section runs if the section was named without a subset, or if
# its subset names the part.
runs(section, part) = section in RUN &&
                      (!haskey(OPTS, section) || part in split(OPTS[section], ','))
parse_list(T, key, default) =
    haskey(OPTS, key) ? [T(eval(Meta.parse(s))) for s in split(OPTS[key], ',')] :
    collect(default)
opt(T, key, default) = haskey(OPTS, key) ? T(eval(Meta.parse(OPTS[key]))) : default

# --- the coefficients, from the metric ------------------------------------------

"""
The coefficients of one component's evolution at `x` on `bg`, read the way
the kernel reads them: `background_state` for `h` and its analytic spatial
gradient, `metric_quantities` for `α`, `β^i`, `γ^{ij}`, `√γ`, and
`metric_derivatives` for `∂_iβ^i` and `∂_iA^{ij}`, `A = α√γ γ^{ij}`. `nothing`
on the chart's singular set.
"""
function coefficients(bg, x)
    h, _, ∂h = background_state(bg, 0.0, SVector{3,Float64}(x...))
    all(isfinite, h) || return nothing
    _, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    (isfinite(α) && all(isfinite, β) && all(isfinite, γu)) || return nothing
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    divA = SVector{3}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)
    # The same over `x` and `y` only, for the plane: see `model2d`.
    divβ2 = dβ[1, 1] + dβ[2, 2]
    divA2 = SVector{2}(dA[1, 1, j] + dA[2, 2, j] for j in 1:2)
    return (α=α, β=β, γu=γu, sqrtγ=sqrtγ, A=(α * sqrtγ) * γu, cu=α / sqrtγ,
            divβ=divβ, divA=divA, divβ2=divβ2, divA2=divA2,
            speed=α * sqrt(γu[1, 1] + γu[2, 2] + γu[3, 3]) + norm(β))
end

const KS0 = SM.KerrSchild(1.0, 0.0)

# The check on the route: Kerr-Schild `a = 0` against its closed form,
# `β = H/(1+H) n̂`, `α = 1/√(1+H)`, `γ^{ij} = δ^{ij} − H/(1+H) n̂ⁱn̂ʲ`,
# `√γ = √(1+H)`, `H = 2M/r`, at points in every octant.
let rng = MersenneTwister(1)
    for _ in 1:32
        x = SVector{3}(randn(rng, 3)...)
        x *= (0.3 + 3 * rand(rng)) / norm(x)
        r = norm(x)
        n̂ = x / r
        H = 2 / r
        c = coefficients(KS0, x)
        want = (α=1 / sqrt(1 + H), β=H / (1 + H) * n̂, sqrtγ=sqrt(1 + H),
                γu=[(i == j) - H / (1 + H) * n̂[i] * n̂[j] for i in 1:3, j in 1:3])
        err = max(abs(c.α - want.α), maximum(abs.(c.β - want.β)),
                  abs(c.sqrtγ - want.sqrtγ), maximum(abs.(c.γu - want.γu)))
        err ≤ 1e-12 || error("KerrSchild(1, 0) at $x: coefficients off by $err")
    end
end

# The two speeds along a unit covector `n`: `b = β^i n_i` and
# `a = α √(γ^{ij} n_i n_j)`; the characteristics move at `−b ± a` along `n`,
# and a closure whose boundary lies on the `−n` side is **outflow-like** when
# both point to that side, `b > a`, and inflow-like when `b < a`.
speeds(c, n) = (sum(c.β[i] * n[i] for i in 1:3),
                c.α * sqrt(sum(c.γu[i, j] * n[i] * n[j] for i in 1:3, j in 1:3)))

# The quantiles of a sorted vector.
quantile_sorted(v, p) = v[clamp(round(Int, p * (length(v) - 1)) + 1, 1, length(v))]

# ==============================================================================
# margins
# ==============================================================================

const CHARTS = (("KerrSchild(1, 0)", SM.KerrSchild(1.0, 0.0)),
                ("KerrSchild(1, 3/5)", SM.KerrSchild(1.0, 0.6)),
                ("KerrSchild(1, 9/10)", SM.KerrSchild(1.0, 0.9)),
                ("Harmonic(1, 7/10)", SM.Harmonic(1.0, 0.7)),
                ("Harmonic(1, 9/10)", SM.Harmonic(1.0, 0.9)))

# The octant runs' spacings at the hole (`CODE.md`, "Robust stability on the
# octant": `h = 1/16, 1/24, 1/32`, and the spinning hole's proposed `1/48`).
const MARGIN_H = (1 / 16, 1 / 24, 1 / 32, 1 / 48)
const MARGIN_M = (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48)

# The offset surface `r_E(n̂) = r_h(n̂) − m h` (the seed's analytic horizon,
# `analytic_horizon_radius`, untruncated) and its outward normal covector,
# the gradient of `|x| − r_E(x/|x|)`, by central differences.
surface_radius(bg, n̂, off) = analytic_horizon_radius(bg, n̂) - off
function surface_normal(bg, x, off)
    F(y) = norm(y) - surface_radius(bg, y / norm(y), off)
    δ = 1e-6 * norm(x)
    g = SVector{3}((F(x + δ * e) - F(x - δ * e)) / (2δ)
                   for e in (SVector(1.0, 0, 0), SVector(0, 1.0, 0),
                             SVector(0, 0, 1.0)))
    return g / norm(g)
end

# The distance from `x` to the chart's singular disk `z = 0`, `ρ ≤ a`.
function disk_distance(x, a)
    ρ = hypot(x[1], x[2])
    return ρ ≤ a ? abs(x[3]) : hypot(ρ - a, x[3])
end

"""
The offset surface `m` cells below the seed's horizon on `bg` at spacing
`h`: the outflow margin `b_n/a_n − 1` along the true normal (least over a
`θ × φ` grid of the surface), the clearance of the singular disk, and the
lego surface — every lattice point `x = h (i, j, k)` outside the surface with
an excised neighbour along an axis, each such (point, axis, side) one closure
face — with its per-axis ratios `b/a`, `b = −s β^d` toward the excised side
`s`, `a = α√γ^{dd}`.
"""
function surface_stats(bg, h, m)
    off = m * h
    a_spin = singular_radius(bg)
    rmin = horizon_min_radius(bg) - off
    rmin > 0 || return nothing
    # The continuous surface: the normal margin and the clearance.
    nmin = Inf
    clear = Inf
    for it in 1:181, φ in (0.0, π / 5)
        θ = π * (it - 1) / 180
        n̂ = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
        x = surface_radius(bg, n̂, off) * n̂
        a_spin > 0 && (clear = min(clear, disk_distance(x, a_spin)))
        c = coefficients(bg, x)
        c === nothing && return (normal=-Inf, clear=clear, faces=0)
        b, a = speeds(c, surface_normal(bg, x, off))
        nmin = min(nmin, b / a - 1)
    end
    clear ≤ 0 && return (normal=nmin, clear=clear, faces=0)
    # The lego surface.
    excised(i, j, k) = (x = h * SVector(i, j, k); r = norm(x);
                        r < surface_radius(bg, x / r, off))
    rmax = horizon_max_radius(bg) - off
    K = ceil(Int, (rmax + 2h) / h)
    ratios = Float64[]
    for i in (-K):K, j in (-K):K
        # The k range whose radius can be within two cells of the surface.
        ρ2 = (i^2 + j^2) * h^2
        ρ2 > (rmax + 2h)^2 && continue
        for k in (-K):K
            r = h * sqrt(i^2 + j^2 + k^2)
            (rmin - 2h ≤ r ≤ rmax + 2h) || continue
            excised(i, j, k) && continue
            c = nothing
            for d in 1:3, s in (-1, 1)
                e = ntuple(l -> l == d ? s : 0, 3)
                excised(i + e[1], j + e[2], k + e[3]) || continue
                c === nothing && (c = coefficients(bg, h * SVector(i, j, k)))
                c === nothing && continue
                push!(ratios, -s * c.β[d] / (c.α * sqrt(c.γu[d, d])))
            end
        end
    end
    sort!(ratios)
    return (normal=nmin, clear=clear, faces=length(ratios), ratios=ratios,
            inflow=count(<(1), ratios) / length(ratios),
            into=count(<(0), ratios) / length(ratios))
end

if runs("margins", "margins")
    println("\n=== margins: the outflow condition on the seed's offset surfaces " *
            "r_E(n̂) = r_h(n̂) − m h ===")
    println("normal: least b_n/a_n − 1 over the surface along its true normal " *
            "(> 0 is outflow); faces: the lego surface's closure faces; b/a: " *
            "the per-axis ratio at a face, its least value and quantiles; " *
            "inflow: the fraction of faces with b/a < 1 (predicted r_E/(2M) " *
            "for Kerr-Schild a = 0, r_E/(2M) printed beside it); b/a < 0: the " *
            "fraction whose shift points into the excised set; clearance: " *
            "the least distance from the surface to the singular disk, in M " *
            "and in cells")
    hs = parse_list(Float64, "h", MARGIN_H)
    ms = parse_list(Int, "m", MARGIN_M)
    answers = String[]
    for (label, bg) in CHARTS
        # Where outflow must end: Kerr-Schild's inner horizon, the spheroid
        # `R = r₋ = M − √(M² − a²)` (equatorial radius `√(r₋² + a²)`), below
        # which the surfaces of constant `R` are timelike again; the harmonic
        # chart ends at its disk `R = 0` (`r = M`) before it reaches `r₋`.
        a = singular_radius(bg)
        inner = bg isa SM.KerrSchild ?
                (r₋ = 1 - sqrt(1 - a^2); (sqrt(r₋^2 + a^2), r₋)) : (a, 0.0)
        println("\n-- $label: r_h ∈ [$(round(horizon_min_radius(bg); digits=4)), " *
                "$(round(horizon_max_radius(bg); digits=4))], singular disk " *
                "radius $a, " *
                (bg isa SM.KerrSchild ? "inner horizon" : "the chart's end (the disk)") *
                " at the equator $(round(inner[1]; digits=4)): depth " *
                "$(round(horizon_max_radius(bg) - inner[1]; digits=4)) M there --")
        println("| h | m | r_E range | normal | faces | b/a min | 1 % | 10 % | 50 % " *
                "| inflow | b/a < 0 | r_E/(2M) | clearance M | cells |")
        println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for h in hs
            shallow = nothing
            deep = nothing
            for m in ms
                st = surface_stats(bg, h, m)
                st === nothing && continue
                r1 = horizon_min_radius(bg) - m * h
                r2 = horizon_max_radius(bg) - m * h
                clr = isfinite(st.clear) ?
                      (fmt("%.3f", st.clear), fmt("%.1f", st.clear / h)) :
                      ("—", "—")
                if st.faces == 0
                    say("| 1/%d | %d | %.3f–%.3f | %s | — | the surface meets the " *
                        "singular set | | | | | | | %s | %s |", round(Int, 1 / h), m, r1, r2,
                        isfinite(st.normal) ? fmt("%+.3f", st.normal) : "—",
                        clr...)
                    continue
                end
                v = st.ratios
                pred = iszero(singular_radius(bg)) && bg isa SM.KerrSchild ?
                       fmt("%.3f", r1 / 2) : "—"
                say("| 1/%d | %d | %.3f–%.3f | %+.3f | %d | %.3f | %.3f | %.3f | %.3f " *
                    "| %.3f | %.3f | %s | %s | %s |", round(Int, 1 / h), m, r1, r2,
                    st.normal, st.faces, v[1], quantile_sorted(v, 0.01),
                    quantile_sorted(v, 0.1), quantile_sorted(v, 0.5), st.inflow,
                    st.into, pred, clr...)
                st.normal > 0 && shallow === nothing && (shallow = (m, st))
                st.normal > 0 && st.clear > 3 * h && (deep = (m, st))
            end
            push!(answers,
                  fmt("| %s | 1/%d | %s | %s |", label, round(Int, 1 / h),
                      shallow === nothing ? "none" :
                      fmt("m = %d (%.3f M), inflow %.3f", shallow[1],
                          shallow[1] * h, shallow[2].inflow),
                      deep === nothing ? "none" :
                      fmt("m = %d (%.3f M), %sinflow %.3f", deep[1], deep[1] * h,
                          isfinite(deep[2].clear) ?
                          fmt("clearance %.1f cells, ", deep[2].clear / h) : "",
                          deep[2].inflow)))
        end
    end
    println("\n-- the answer per chart: the shallowest surface with normal outflow " *
            "everywhere, and the deepest of the depths scanned with normal " *
            "outflow everywhere that clears the singular disk by more than " *
            "three cells --")
    println("| chart | h | shallowest normal outflow | deepest normal outflow clearing the disk |")
    println("|---|---|---|---|")
    foreach(println, answers)
end

# --- margins, window: the spinning hole's window (step X5) ---------------------
#
# Kerr-Schild at spin `a` (`a=…`, default `3/5`), the two geometries step X2b
# built — the sphere `r < r_E` about the center and the tracked offset surface
# `r < r_h(n̂) − m h` — at the spinning round's spacings `h = 1/24, 1/32, 1/48`
# (`PLAN.md`, step X7): the window (normal outflow along the surface's own
# normal, which ends near Kerr-Schild's inner horizon — `√(r₋² + a²)` on the
# equator, `0.632` at `a = 3/5` — and the room for the core rule's surface
# `r_0` between the ring `ρ = a` and `r_E`), the depth in cells below the
# horizon at the poles (`r₊`) and on the equator (`√(r₊² + a²)`), and the lego
# census at `q = 4`: X1's faces (an immediate excised neighbour along an
# axis) and X2b's closure axes (`k_s < G` on side `s`), with the per-axis
# ratio `b/a`, `b = −s β^d`, `a = α√γ^{dd}`, the fraction whose shift points
# into the excised set (`b/a < 0`, what X2b refuses) and the fraction that
# has both characteristics entering from it (`b/a < −1`).

const WINDOW_H = (1 / 24, 1 / 32, 1 / 48)
const WINDOW_RE = (0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10, 1.133, 1.20, 1.30,
                   1.40, 1.50, 1.60, 1.70)
const WINDOW_M = (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56)

# The lattice directions of 3D, and the one nearest a unit vector (the first
# in this order on a tie, which an integer point on a sphere never makes).
const DIRS26 = [(a, b, c) for a in -1:1 for b in -1:1 for c in -1:1
                if (a, b, c) != (0, 0, 0)]
function nearest_direction26(n̂)
    best = DIRS26[1]
    bd = -Inf
    for e in DIRS26
        d = (e[1] * n̂[1] + e[2] * n̂[2] + e[3] * n̂[3]) / sqrt(e[1]^2 + e[2]^2 + e[3]^2)
        d > bd && (bd = d; best = e)
    end
    return best
end

# The census of a lego surface: every lattice point `h (i, j, k)` not excised
# within `G + 1` cells of the radial shell `[r_lo, r_hi]` the surface lies
# in, and at it every axis and side whose run of non-excised points `k_s` is
# shorter than `G` — a closure axis — with its ratio `b/a`; `k_s = 0` is a
# face. With `normal` (the surface's outward normal at a point), the
# frame-dragged faces' rule is tried on every closure axis whose shift points
# into the excised set: each excised tap `Q` of the advective stencil
# (`k_s < j ≤ q/2`) is filled along the lattice direction (of 26) nearest the
# normal at `Q`, from the first consecutive non-excised points `Q + k e`
# inside the point's `G`-box, at most three; `fills[n + 1]` counts the taps
# with `n` sources, `k0max` is the farthest first source and `kmax` the
# farthest source (`extrapolation_table` holds `k₀ + n − 1 ≤ G`). Returns the
# two sorted lists of ratios and the fill census.
function lego_census(bg, h, excised, rlo, rhi; q=4, normal=nothing)
    G = q ÷ 2 + 1
    w = (G + 1) * h
    K = ceil(Int, (rhi + w) / h)
    faces = Float64[]
    axes = Float64[]
    fills = zeros(Int, 4)
    k0max = 0
    kmax = 0
    for i in (-K):K, j in (-K):K
        (i^2 + j^2) * h^2 > (rhi + w)^2 && continue
        for k in (-K):K
            r = h * sqrt(i^2 + j^2 + k^2)
            (rlo - w ≤ r ≤ rhi + w) || continue
            excised(i, j, k) && continue
            c = nothing
            for d in 1:3, s in (-1, 1)
                e = ntuple(l -> l == d ? s : 0, 3)
                ks = 0
                while ks < G && !excised(i + (ks + 1) * e[1], j + (ks + 1) * e[2],
                                         k + (ks + 1) * e[3])
                    ks += 1
                end
                ks < G || continue
                c === nothing && (c = coefficients(bg, h * SVector(i, j, k)))
                c === nothing && error("a closure axis at $(h .* (i, j, k)) is " *
                                       "on the chart's singular set")
                ratio = -s * c.β[d] / (c.α * sqrt(c.γu[d, d]))
                push!(axes, ratio)
                ks == 0 && push!(faces, ratio)
                (normal === nothing || ratio >= 0) && continue
                for jt in (ks + 1):(q ÷ 2)
                    Q = (i + jt * e[1], j + jt * e[2], k + jt * e[3])
                    en = nearest_direction26(normal(h * SVector(Q...)))
                    n = 0
                    k0 = 0
                    for kk in 1:(4G)
                        S = Q .+ kk .* en
                        maximum(abs.(S .- (i, j, k))) ≤ G || break
                        if excised(S...)
                            n == 0 || break
                            continue
                        end
                        n == 0 && (k0 = kk)
                        n += 1
                        n == 3 && break
                    end
                    fills[n + 1] += 1
                    k0max = max(k0max, k0)
                    n > 0 && (kmax = max(kmax, k0 + n - 1))
                end
            end
        end
    end
    return sort!(faces), sort!(axes), (fills=fills, k0max=k0max, kmax=kmax)
end

# The least normal margin `b_n/a_n − 1` of a surface `r = R(n̂)` along its own
# normal, over `θ` (Kerr-Schild is axisymmetric about the spin axis).
function normal_margin(bg, R, normal)
    nmin = Inf
    for it in 1:721
        θ = π * (it - 1) / 720
        n̂ = SVector(sin(θ), 0.0, cos(θ))
        x = R(n̂) * n̂
        c = coefficients(bg, x)
        c === nothing && return -Inf
        b, a = speeds(c, normal(x))
        nmin = min(nmin, b / a - 1)
    end
    return nmin
end

frac(v, p) = isempty(v) ? 0.0 : count(p, v) / length(v)

function census_cells(faces, axes, fl)
    return fmt("%d | %.3f | %.3f | %.3f | %.3f | %.3f | %d | %d | %.3f | %.3f | " *
               "%d | %d, %d, %d, %d | %d, %d",
               length(faces), faces[1], quantile_sorted(faces, 0.01),
               frac(faces, <(1)), frac(faces, <(0)), frac(faces, <(-1)),
               length(axes), count(<(0), axes), frac(axes, <(0)), axes[1],
               sum(fl.fills), reverse(fl.fills)..., fl.k0max, fl.kmax)
end

if runs("margins", "window")
    a_spin = opt(Float64, "a", 0.6)
    bg = SM.KerrSchild(1.0, a_spin)
    rp = horizon_min_radius(bg)
    req = horizon_max_radius(bg)
    rm = 1 - sqrt(1 - a_spin^2)
    inner_eq = sqrt(rm^2 + a_spin^2)
    hs = parse_list(Float64, "h", WINDOW_H)
    println("\n=== margins, window: Kerr-Schild a = $a_spin — horizon r₊ = " *
            "$(round(rp; digits=4)) at the poles, $(round(req; digits=4)) on the " *
            "equator; the ring at ρ = $a_spin; the inner horizon r₋ = " *
            "$(round(rm; digits=4)) at the poles, $(round(inner_eq; digits=4)) on " *
            "the equator ===")
    println("normal: least b_n/a_n − 1 along the surface's own normal (> 0 is " *
            "outflow); poles, equator: the depth below the horizon in cells; r_0 " *
            "room: (r_E − ρ_ring)/h, the cells between the ring and the surface for " *
            "the core rule's sphere; faces (X1): the lego surface's immediate " *
            "closure faces, b/a min and 1 %, the inflow-like (b/a < 1), into " *
            "(b/a < 0) and both-entering (b/a < −1) fractions; closure axes (X2b's " *
            "census, q = 4): the (point, axis, side) with k_s < G, how many have " *
            "b/a < 0 (X2b's refusal), their fraction, and the least b/a; filled " *
            "taps: the excised taps of those axes' advective stencils (k_s < j ≤ " *
            "q/2) that X5's rule fills along the lattice direction (of 26) " *
            "nearest the normal, by the number of sources it finds in the G-box " *
            "(0 would be a refusal), the farthest first source k₀ and the " *
            "farthest source k₀ + n − 1")
    hdr = "| faces | b/a min | 1 % | inflow | b/a < 0 | b/a < −1 | closure axes " *
          "| b/a < 0 | fraction | least | filled taps | 3, 2, 1, 0 sources | k₀, k₀ + n − 1 max |"
    for h in hs
        println("\n-- the sphere r < r_E, h = 1/$(round(Int, 1/h)) --")
        println("| r_E | normal | poles | equator | r_0 room " * hdr)
        println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r_E in parse_list(Float64, "rE", WINDOW_RE)
            r_E ≤ a_spin && continue
            nm = normal_margin(bg, _ -> r_E, x -> x / norm(x))
            excised(i, j, k) = h^2 * (i^2 + j^2 + k^2) < r_E^2
            faces, axes, fl = lego_census(bg, h, excised, r_E, r_E;
                                          normal=x -> x / norm(x))
            say("| %.3f | %+.3f | %.1f | %.1f | %.1f | %s |", r_E, nm, (rp - r_E) / h,
                (req - r_E) / h, (r_E - a_spin) / h, census_cells(faces, axes, fl))
        end
        println("\n-- the tracked offset surface r < r_h(n̂) − m h, " *
                "h = 1/$(round(Int, 1/h)) (the depth is m cells along every ray) --")
        println("| m | r_E range | normal | r_0 room at the equator " * hdr)
        println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in parse_list(Int, "m", WINDOW_M)
            off = m * h
            r1, r2 = rp - off, req - off
            r2 ≤ a_spin && continue
            nm = normal_margin(bg, n̂ -> surface_radius(bg, n̂, off),
                               x -> surface_normal(bg, x, off))
            excised(i, j, k) = (x = h * SVector(i, j, k); r = norm(x);
                                r < surface_radius(bg, x / r, off))
            faces, axes, fl = lego_census(bg, h, excised, r1, r2;
                                          normal=x -> surface_normal(bg, x, off))
            say("| %d | %.3f–%.3f | %+.3f | %.1f | %s |", m, r1, r2, nm,
                (r2 - a_spin) / h, census_cells(faces, axes, fl))
        end
    end
end

# ==============================================================================
# shared: closure weights in Float64, the RK4 region, sparse evolution
# ==============================================================================

const G_OF = q -> q ÷ 2 + 1

# The package's weights, rounded once into `Float64` and cached per case —
# behind a lock, since `model2d`'s noise runs assemble on several threads.
const WCACHE = Dict{Any,Tuple{UnitRange{Int},Vector{Float64}}}()
const WLOCK = ReentrantLock()
wfloat(nw) = (nw[1], Float64.(nw[2]))
cached(f, key) = lock(() -> get!(f, WCACHE, key), WLOCK)
wd(q, m, km, kp, R) = cached(() -> wfloat(closure_derivative_weights(q, m, km, kp;
                                                                       reach=R)),
                             (:d, q, m, km, kp, R))
wk(q, kind, km, kp) = cached(() -> wfloat(closure_dissipation_weights(
                                              q, kind, min(km, G_OF(q)),
                                              min(kp, G_OF(q)))),
                             (:k, q, kind, min(km, G_OF(q)), min(kp, G_OF(q))))
wl(q, up, km, kp) = cached(() -> wfloat(lopsided_weights(q, up, min(km, G_OF(q)),
                                                         min(kp, G_OF(q)))),
                           (:l, q, up, min(km, G_OF(q)), min(kp, G_OF(q))))
# The centered ones, through the same cache.
wc(q, m) = wd(q, m, 1000, 1000, G_OF(q))
wkc(q) = wk(q, :msn, 1000, 1000)
wlc(q, up) = wl(q, up, 1000, 1000)

# RK4's amplification factor and the largest `ν = dt` that keeps every
# eigenvalue `λ` with `Re λ ≤ tol` inside its stability region (along the
# ray `λ ν`), by a scan and a bisection per eigenvalue.
rk4(z) = 1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24
function rk4_step_limit(λs; tol=1e-9)
    νmax = Inf
    scale = maximum(abs, λs)
    for λ in λs
        # A zero mode does not limit the step, and neither does a growing
        # one, which no step makes stable; both are reported on their own.
        # Growing is relative to the eigenvalue's size, so that a cluster at
        # zero split by roundoff is not taken for an imaginary-axis one.
        (real(λ) > tol * abs(λ) || abs(λ) < 1e-6 * scale) && continue
        lo, hi = 0.0, 4.0 / abs(λ)
        # The first exit along the ray: scan, then bisect.
        prev = 0.0
        found = false
        for k in 1:400
            ν = hi * k / 400
            if abs(rk4(λ * ν)) > 1 + 1e-12
                lo, hi = prev, ν
                found = true
                break
            end
            prev = ν
        end
        found || continue
        for _ in 1:60
            mid = (lo + hi) / 2
            abs(rk4(λ * mid)) > 1 + 1e-12 ? (hi = mid) : (lo = mid)
        end
        νmax = min(νmax, lo)
    end
    return νmax
end

# Classical RK4 on `du/dt = A u`, in place, with three scratch vectors.
struct RK4Work
    k1::Vector{Float64}
    k2::Vector{Float64}
    k3::Vector{Float64}
    k4::Vector{Float64}
    y::Vector{Float64}
end
RK4Work(n) = RK4Work(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))
function rk4_step!(u, A, dt, w::RK4Work)
    mul!(w.k1, A, u)
    @. w.y = u + dt / 2 * w.k1
    mul!(w.k2, A, w.y)
    @. w.y = u + dt / 2 * w.k2
    mul!(w.k3, A, w.y)
    @. w.y = u + dt * w.k3
    mul!(w.k4, A, w.y)
    @. u += dt / 6 * (w.k1 + 2 * w.k2 + 2 * w.k3 + w.k4)
    return u
end

# A sparse matrix from coordinate lists, summing duplicates.
struct Coo
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
end
Coo() = Coo(Int[], Int[], Float64[])
@inline function put!(c::Coo, i, j, v)
    iszero(v) && return nothing
    push!(c.I, i)
    push!(c.J, j)
    push!(c.V, v)
    return nothing
end
tomatrix(c::Coo, n) = sparse(c.I, c.J, c.V, n, n)

# ==============================================================================
# model1d — a line closed at its left end
# ==============================================================================
#
# The unknowns are `u_i, v_i`, `i = 0 … N−1` at `x_i = x_0 + i h`; the right
# end's ghosts hold zero (the perturbation's Dirichlet data, as at the outer
# face); the left end is either a **closure** — the points `i < 0` excised,
# each operator closed by `(k⁻, k⁺) = (min(i, R), R)` — or **Dirichlet**,
# ghosts of zero, which is what a frozen core holding the unperturbed state
# is to a perturbation. Per point:
#
#     ∂_t u = w (b Adv u + c_u v + (ε/h) K u) − ρ u
#     ∂_t v = w (b Adv v + c_A D₂ u + (ε/h) K v [+ divβ v + divA D₁u]) − ρ v
#
# with `Adv = (1 − λ) D₁ + λ L` the advection blended toward the lopsided
# `L`, `w, ρ` a layer's profiles (`1, 0` without one), the bracket the
# coefficient-gradient terms that make it the covariant wave equation.

"""
The line operator: `coef[i] = (b, cu, cA, divβ, divA)`, `prof[i] = (w, ρ)`,
`εs[i]`, `λs[i]` per point; `left ∈ (:closure, :dirichlet)`; `R` the
derivative closures' reach; `diss` the dissipation's closure.
"""
function line_operator(q, h, coef, prof, εs, λs; left=:closure, R=G_OF(q),
                       diss=:msn)
    N = length(coef)
    c = Coo()
    big = 1000
    G = G_OF(q)
    for i in 0:(N - 1)
        w, ρ = prof[i + 1]
        b, cu, cA, divβ, divA = coef[i + 1]
        ru, rv = i + 1, N + i + 1
        if w == 0
            continue                 # a frozen point: du = 0
        end
        km = left === :closure ? min(i, max(R, G)) : big
        kp = big
        D1 = wd(q, 1, km, kp, R)
        D2 = wd(q, 2, km, kp, R)
        K = left === :closure ? wk(q, diss, km, kp) : wkc(q)
        L = wl(q, b >= 0 ? 1 : -1, km, kp)
        λ = λs[i + 1]
        # A tap at offset `j` of field `f` (0: u, 1: v) with weight `wt`:
        # zero outside `0 … N−1` — the ghosts — and refused if excised.
        tap(row, j, f, wt) = begin
            k = i + j
            if k < 0
                left === :closure && error("line_operator read an excised point")
            elseif k < N
                put!(c, row, f * N + k + 1, wt)
            end
        end
        for f in (0, 1)
            row = f == 0 ? ru : rv
            for (j, wj) in zip(D1...)
                tap(row, j, f, w * b * (1 - λ) * wj / h)
            end
            if λ > 0
                for (j, wj) in zip(L...)
                    tap(row, j, f, w * b * λ * wj / h)
                end
            end
            for (j, wj) in zip(K...)
                tap(row, j, f, w * εs[i + 1] * wj / h)
            end
            put!(c, row, row, -ρ)
        end
        put!(c, ru, rv, w * cu)
        for (j, wj) in zip(D2...)
            tap(rv, j, 0, w * cA * wj / h^2)
        end
        if divβ != 0 || divA != 0
            put!(c, rv, rv, w * divβ)
            for (j, wj) in zip(D1...)
                tap(rv, j, 0, w * divA * wj / h)
            end
        end
    end
    return tomatrix(c, 2N)
end

# The share of an eigenvector's squared norm on the first `k` points of
# the line (its `u` and `v` parts together).
function boundary_share(vec, N, k)
    tot = sum(abs2, vec)
    near = sum(abs2, vec[1:k]) + sum(abs2, vec[(N + 1):(N + k)])
    return near / tot
end

# The spectrum's rightmost eigenvalue away from zero, its mode's share on
# the first ten points, the zero modes — the eigenvalues within `1e-4/h` of
# zero, how many, and whether they are fewer independent null vectors than
# eigenvalues (a Jordan block: a mode that grows linearly in time) — and the
# RK4 step the whole spectrum allows.
function spectrum_summary(A, N)
    M = Matrix(A)
    F = eigen(M)
    zero_ = abs.(F.values) .< 1e-4
    nz = count(zero_)
    vals = [v for (v, z) in zip(F.values, zero_) if !z]
    k = argmax(real.(F.values) .- 1e9 .* zero_)
    nullity = nz == 0 ? 0 : count(<(1e-10 * maximum(abs, F.values)), svdvals(M))
    return (maxre=real(F.values[k]), share=boundary_share(F.vectors[:, k], N, 10),
            ν=rk4_step_limit(F.values), λs=F.values, nzero=nz,
            jordan=nz > nullity)
end

# The frozen-coefficient symbol on the open line, both branches, for the
# periodic reference: its largest `Re λ` and the RK4 step it allows.
function periodic_summary(q, b, a, ε; lop=false)
    w1 = wc(q, 1)
    w2 = wc(q, 2)
    wkk = wkc(q)
    wlo = wlc(q, b >= 0 ? 1 : -1)
    sym(w, θ) = sum(wj * cis(j * θ) for (j, wj) in zip(w...))
    λs = ComplexF64[]
    for θ in range(0, π; length=721)
        μ = b * (lop ? sym(wlo, θ) : sym(w1, θ)) + ε * sym(wkk, θ)
        P = a^2 * sym(w2, θ)
        r = sqrt(complex(P))
        push!(λs, μ + r, μ - r)
    end
    return (maxre=maximum(real, λs), ν=rk4_step_limit(λs))
end

const FROZEN_A = 0.5                # the wave speed; b = (b/a) a
const FROZEN_N = 200
# From both characteristics entering through the closure (`b/a < −1`, which
# the spinning charts' lego faces reach: `margins`) through one (`|b/a| < 1`)
# to none (`b/a > 1`, outflow).
const FROZEN_RATIOS = (-1.25, -0.5, 0.0, 0.5, 0.8, 0.95, 1.05, 1.25, 1.5, 2.0)

function frozen_line(q, ratio, ε; left=:closure, R=G_OF(q), diss=:msn, lop=false,
                     N=FROZEN_N)
    a = FROZEN_A
    b = ratio * a
    coef = fill((b, 1.0, a^2, 0.0, 0.0), N)
    A = line_operator(q, 1.0, coef, fill((1.0, 0.0), N), fill(ε, N),
                      fill(lop ? 1.0 : 0.0, N); left=left, R=R, diss=diss)
    return A
end

# `+1.2e−3 (b)`: the rightmost `Re λ` away from zero in units of `1/h`,
# `< 0` to roundoff, with `b` when its mode has half its norm on the first ten
# points; then `0` for a zero mode, `0²` for a Jordan block at zero.
function fmt_growth(s)
    g = s.maxre ≤ 1e-10 ? "< 0" :
        fmt("%+.1e%s", s.maxre, s.share > 0.5 ? " (b)" : "")
    s.nzero == 0 && return g
    return g * (s.jordan ? ", 0²" : ", 0")
end

if runs("model1d", "frozen")
    println("\n=== model1d, frozen: the constant-coefficient line (a = $FROZEN_A, " *
            "b = (b/a) a, h = 1, N = $FROZEN_N) closed at its left end ===")
    println("each entry: the semi-discrete spectrum's largest Re λ away from " *
            "zero in units of 1/h ('< 0' to 1e-10; '(b)' when its mode has " *
            "more than half its norm on the first ten points), then '0' for a " *
            "zero mode (|λ| < 1e-4/h) and '0²' for a Jordan block at zero, a " *
            "mode growing linearly in time; cfl: the largest RK4 step the whole " *
            "spectrum allows, as dt (a + b)/h, against the open line's")
    for q in parse_list(Int, "q", (2, 4))
        G = G_OF(q)
        println("\n-- q = $q: the default closure (reach G = $G, MSN dissipation, " *
                "centered advection) against ε_KO, and the Dirichlet-ghost left " *
                "end and the open line --")
        println("| b/a | ε = 0 | 1/4 | 1/2 | 1 | cfl at 1/2 | Dirichlet, 1/2 | its cfl " *
                "| open line cfl at 1/2 |")
        println("|---|---|---|---|---|---|---|---|---|")
        for ratio in parse_list(Float64, "ratios", FROZEN_RATIOS)
            ss = [spectrum_summary(frozen_line(q, ratio, ε), FROZEN_N)
                  for ε in (0.0, 0.25, 0.5, 1.0)]
            sd = spectrum_summary(frozen_line(q, ratio, 0.5; left=:dirichlet),
                                  FROZEN_N)
            sp = periodic_summary(q, ratio * FROZEN_A, FROZEN_A, 0.5)
            spd = (1 + abs(ratio)) * FROZEN_A
            say("| %.2f | %s | %s | %s | %s | %.3f | %s | %.3f | %.3f |", ratio,
                fmt_growth.(ss)..., ss[3].ν * spd, fmt_growth(sd), sd.ν * spd,
                sp.ν * spd)
        end
        println("\n-- q = $q, ε_KO = 1/2 (and 1): the closure's reach, the " *
                "dissipation's closure, and the lopsided advection --")
        reaches = [R for R in (G - 1, G, G + 1, G + 2) if R >= 2]
        println("| b/a | " * join(["reach $R" for R in reaches], " | ") *
                " | reduced | one-sided | MSN | reduced, ε = 1 | one-sided, ε = 1 " *
                "| lopsided | lopsided cfl | open lopsided cfl |")
        println("|---|" * repeat("---|", length(reaches) + 9))
        for ratio in parse_list(Float64, "ratios", FROZEN_RATIOS)
            sr = [spectrum_summary(frozen_line(q, ratio, 0.5; R=R), FROZEN_N)
                  for R in reaches]
            sk = [spectrum_summary(frozen_line(q, ratio, ε; diss=k), FROZEN_N)
                  for (k, ε) in ((:reduced, 0.5), (:onesided, 0.5), (:msn, 0.5),
                                 (:reduced, 1.0), (:onesided, 1.0))]
            sl = spectrum_summary(frozen_line(q, ratio, 0.5; lop=true), FROZEN_N)
            spl = periodic_summary(q, ratio * FROZEN_A, FROZEN_A, 0.5; lop=true)
            spd = (1 + abs(ratio)) * FROZEN_A
            say("| %.2f | %s | %s | %s | %.3f | %.3f |", ratio,
                join(fmt_growth.(sr), " | "), join(fmt_growth.(sk), " | "),
                fmt_growth(sl), sl.ν * spd, spl.ν * spd)
        end
    end
end

# --- model1d, reflect: a packet into the closure ------------------------------
#
# On the frozen line (`N = 600`), a packet on branch 1 — the fast ingoing one,
# `v_g → −b − a` — with carrier `θ₀` and Gaussian envelope `σ` cells, made a
# pure discrete branch-1 superposition by projecting in Fourier space
# (`v̂ = i sign(θ) a √c(θ) û`, the symbol's eigenvector), launched at `x₀ =
# 300` toward the closure. Whatever is in the domain once it has arrived is
# what the closure sent back: its largest amplitude off the closure's own
# points (`j > G`) against the packet's, the phase `θ` it is carried at (the
# peak of its spectrum), and the group velocity of its centroid, against the
# open line's `v_g` of that `θ` on each branch.

# The symbols of the centered weights on unit spacing, as functions of θ.
function line_symbols(q)
    w1 = wc(q, 1)
    w2 = wc(q, 2)
    s(θ) = sum(wj * sin(j * θ) for (j, wj) in zip(w1...))
    s′(θ) = sum(j * wj * cos(j * θ) for (j, wj) in zip(w1...))
    c(θ) = -sum(wj * cos(j * θ) for (j, wj) in zip(w2...))
    c′(θ) = sum(j * wj * sin(j * θ) for (j, wj) in zip(w2...))
    sq′(θ) = c′(θ) / (2 * sqrt(c(θ)))
    vg(θ, b, a, br) = br == 1 ? -b * s′(θ) - a * sq′(θ) : -b * s′(θ) + a * sq′(θ)
    return (s=s, c=c, vg=vg)
end

function dft(u)
    N = length(u)
    return [sum(u[j + 1] * cis(-2π * k * j / N) for j in 0:(N - 1)) for k in 0:(N - 1)]
end
idft(û) = (N = length(û); [sum(û[k + 1] * cis(2π * k * j / N) for k in 0:(N - 1)) / N
                           for j in 0:(N - 1)])

function reflect_run(q, ratio, ε; θ0, σ, lop=false, N=600, x0=300, T_obs=150.0)
    a = FROZEN_A
    b = ratio * a
    S = line_symbols(q)
    G = G_OF(q)
    A = frozen_line(q, ratio, ε; lop=lop, N=N)
    u = [exp(-((j - x0) / σ)^2 / 2) * cos(θ0 * (j - x0)) for j in 0:(N - 1)]
    û = dft(u)
    θk = [2π * (k ≤ N ÷ 2 ? k : k - N) / N for k in 0:(N - 1)]
    v̂ = [im * sign(θ) * a * sqrt(max(S.c(abs(θ)), 0.0)) * û[k]
          for (k, θ) in enumerate(θk)]
    v = real.(idft(v̂))
    U = vcat(u, v)
    A0 = maximum(abs, u)
    v1 = S.vg(θ0, b, a, 1)
    v1 < 0 || error("the packet's branch-1 group velocity $v1 is not ingoing")
    dt = 0.25 / (a + b)
    t_arr = (x0 + 4σ) / abs(v1)
    n_arr = ceil(Int, t_arr / dt)
    n_obs = ceil(Int, T_obs / dt)
    w = RK4Work(2N)
    for _ in 1:n_arr
        rk4_step!(U, A, dt, w)
    end
    R = 0.0
    cent(U) = (ws = [U[j]^2 for j in (G + 2):N];
               sum(((G + 1):(N - 1)) .* ws) / max(sum(ws), 1e-300))
    x1 = 0.0
    θdom = NaN
    for n in 1:n_obs
        rk4_step!(U, A, dt, w)
        all(isfinite, U) || return (R=Inf, θ=NaN, vg=NaN, vb=(NaN, NaN))
        R = max(R, maximum(abs, @view U[(G + 2):N]) / A0)
        if n == n_obs ÷ 2
            x1 = cent(U)
            ûr = dft(U[1:N])
            k = argmax([abs(ûr[k + 1]) for k in 1:(N ÷ 2)])
            θdom = 2π * k / N
        end
    end
    x2 = cent(U)
    vmeas = (x2 - x1) / ((n_obs - n_obs ÷ 2) * dt)
    return (R=R, θ=θdom, vg=vmeas, vb=(S.vg(θdom, b, a, 1), S.vg(θdom, b, a, 2)))
end

if runs("model1d", "reflect")
    println("\n=== model1d, reflect: a branch-1 packet (carrier θ₀, envelope σ " *
            "cells) into the closure on the frozen line (a = $FROZEN_A) ===")
    println("R: the largest |u| off the closure's points after the packet has " *
            "arrived, over 150/h of time, against the packet's; θ/π: the phase " *
            "it is carried at; v_g: its centroid's velocity (cells per unit " *
            "time), against the open line's at that θ on branch 1 and 2")
    for q in parse_list(Int, "q", (2, 4))
        println("\n-- q = $q, the default closure (reach G, MSN) --")
        println("| b/a | θ₀/π, σ | ε | advection | R | θ/π | v_g | v_g branch 1, 2 |")
        println("|---|---|---|---|---|---|---|---|")
        for ratio in parse_list(Float64, "ratios", (0.8, 1.25, 2.0)),
            (θ0, σ) in ((π / 8, 16.0), (π / 4, 8.0)), ε in (0.0, 0.5),
            lop in (false, true)

            r = reflect_run(q, ratio, ε; θ0=θ0, σ=σ, lop=lop)
            say("| %.2f | %.3f, %d | %.1f | %s | %.2e | %.3f | %+.3f | %+.3f, %+.3f |",
                ratio, θ0 / π, σ, ε, lop ? "lopsided" : "centered", r.R,
                r.θ / π, r.vg, r.vb...)
        end
    end
end

# --- model1d, radial: dispersion.jl's line, closed at r_E ----------------------
#
# `test/dispersion.jl`'s section (4): the principal part along the `x` axis of
# Kerr-Schild `a = 0` with the coefficients varying, `h = 5/64` from the
# center to the Dirichlet face at `5/2`, RK4 at `cfl = 1/4` on the fixture's
# `λ_max`, sampled every `1/20 M`. Its layer (`r_0 = 0.4`, `r_1 = 1.15`,
# `ρ_max = 1/dt`) is the control and is checked against the numbers `CODE.md`
# records for it; the excised line starts at the first point `x ≥ r_E` and
# closes there. The perturbations: `dispersion.jl`'s ripple (`A = 1e−3` in
# `h`, `Π = 0`, centered `d` cells below the horizon, wavelength `λ`), and a
# smooth pulse — a Gaussian of half-width two cells halfway between `r_E` and
# the horizon — that runs into the surface.

const H_FIX = 5 / 64
const R_H = 2.0
const CFL_FIX = 0.25
const LAMBDA_FIX = 1.6709529240269554
const CHUNK = 1 / 20

radial_dt(h) = (per = ceil(Int, CHUNK / (CFL_FIX * h / LAMBDA_FIX)); (CHUNK / per, per))

ripple(x; d, λ, h, A=1e-3, W=2) =
    (r_c = R_H - d * h; s = (x - r_c) / (W * h);
     abs(s) < 1 ? A * (1 - s^2)^3 * cos(2π * (x - r_c) / λ) : 0.0)

# `C²` blend of the lopsided advection: zero at and outside `start` cells
# below the horizon, one from `start + width` cells on.
blend(x, h; start, width) = width ≤ 0 ? 0.0 : smoothstep(((R_H - x) / h - start) / width)

"""
The radial line at spacing `h`: `mode = :layer` (`dispersion.jl`'s, `r_0`,
`r_1`, `ρ_max = 1/dt`) or `:excise` (closed at the first point `x ≥ r_E`),
`lop = (start, width)` in cells or `nothing`. Returns the operator, the
positions, `dt`, the steps per chunk and the indices of the evolved points.
"""
function radial_line(q, ε, h; mode=:layer, r_E=1.15, r_0=0.4, r_1=1.15,
                     lop=nothing, diss=:msn, R=G_OF(q), div=false)
    n = round(Int, 2.5 / h)
    dt, per = radial_dt(h)
    i0 = mode === :excise ? ceil(Int, r_E / h - 1e-12) : 0
    xs = [i * h for i in i0:(n - 1)]
    int = Interior(Float64; center=(0.0, 0.0, 0.0), r_0=r_0, r_1=r_1, ρ_max=1 / dt)
    prof = mode === :excise ? fill((1.0, 0.0), length(xs)) :
           [interior_profiles(int, x) for x in xs]
    coef = map(xs) do x
        c = (mode === :layer && x < r_0) ? nothing :
            coefficients(KS0, SVector(x, 0.0, 0.0))
        c === nothing && return (0.0, 0.0, 0.0, 0.0, 0.0)
        (c.β[1], c.cu, c.A[1, 1], div ? c.divβ : 0.0, div ? c.divA[1] : 0.0)
    end
    λs = lop === nothing ? zeros(length(xs)) :
         [blend(x, h; start=lop[1], width=lop[2]) for x in xs]
    A = line_operator(q, h, coef, prof, fill(ε, length(xs)), λs;
                      left=mode === :excise ? :closure : :dirichlet, R=R, diss=diss)
    active = [i for i in eachindex(xs) if prof[i][1] > 0]
    return (A=A, xs=xs, dt=dt, per=per, active=active)
end

# One run: `u0` on the line's points, `Π = 0`, to `t_end`, the largest `|u|`
# per shell `[r_h + k h, r_h + (k+1) h)` over the chunk boundaries.
function radial_run(L, u0, t_end; ks=0:5, h)
    N = length(L.xs)
    U = vcat(u0, zeros(N))
    w = RK4Work(2N)
    shells = [findall(x -> R_H + k * h ≤ x < R_H + (k + 1) * h, L.xs) for k in ks]
    amax = zeros(length(ks))
    nsteps = round(Int, t_end / CHUNK) * L.per
    for step in 1:nsteps
        rk4_step!(U, L.A, L.dt, w)
        step % L.per == 0 || continue
        for (m, idx) in enumerate(shells)
            isempty(idx) || (amax[m] = max(amax[m], maximum(abs, U[idx])))
        end
    end
    return amax
end

# `CODE.md`'s record of `dispersion.jl`'s model at `ε_KO = 1/2`: the largest
# `A_0/A` over `λ = 2h, 4h, 8h` at `2 M` and `10 M`, per `q` and `d`.
# Kept as printed, so that the check is to the digits recorded.
const DISPERSION_1D = Dict((2, 2) => ("0.166", "0.410"), (2, 4) => ("0.092", "0.136"),
                           (2, 8) => ("8.9e-3", "0.058"), (4, 2) => ("0.117", "0.348"),
                           (4, 4) => ("0.044", "0.177"), (4, 8) => ("0.011", "0.037"))

# Whether `x` rounds to the recorded `s` at the digits `s` has.
function rounds_to(x, s)
    m = match(r"^([0-9.]+)(?:e(-?[0-9]+))?$", s)
    mant, ex = m.captures[1], m.captures[2] === nothing ? 0 : parse(Int, m.captures[2])
    dec = occursin('.', mant) ? length(split(mant, '.')[2]) : 0
    return abs(x - parse(Float64, s)) ≤ 0.5 * 10.0^(ex - dec) * (1 + 1e-9)
end

if runs("model1d", "radial") let
    println("\n=== model1d, radial: dispersion.jl's line (Kerr-Schild a = 0 along " *
            "x, h = 5/64, face at 5/2, RK4 cfl = 1/4), its layer (r_0 = 0.4, " *
            "r_1 = 1.15, ρ_max = 1/dt) against a closure at r_E ===")
    h = H_FIX
    println("A_0/A: the largest |δh| in the first shell outside the horizon over " *
            "the chunk boundaries, the ripple's largest over λ = 2h, 4h, 8h; " *
            "pulse: a Gaussian (half-width 2 cells) halfway between the surface " *
            "and the horizon; lopsided: blended in from 1 cell below the horizon " *
            "over 4; spectrum: the line's largest Re λ in 1/M")
    for q in parse_list(Int, "q", (2, 4))
        rows = (("layer (control)", (; mode=:layer)),
                ("excised at 1.15", (; mode=:excise, r_E=1.15)),
                ("excised at 1.15, lopsided", (; mode=:excise, r_E=1.15, lop=(1, 4))),
                ("excised at 0.75", (; mode=:excise, r_E=0.75)),
                ("excised at 1.5", (; mode=:excise, r_E=1.5)),
                ("excised at 1.5, lopsided", (; mode=:excise, r_E=1.5, lop=(1, 4))))
        println("\n-- q = $q, ε_KO = 1/2 --")
        println("| interior | ripple d = 2: 2 M, 10 M | d = 4: 2 M, 10 M | d = 8: 2 M, " *
                "10 M | pulse: 2 M, 10 M | spectrum, h = 5/64, 5/128, 5/256 |")
        println("|---|---|---|---|---|---|")
        for (label, kw) in rows
            L = radial_line(q, 0.5, h; kw...)
            cells = String[]
            for d in (2, 4, 8)
                a2 = maximum(radial_run(L, [ripple(x; d=d, λ=λc * h, h=h)
                                            for x in L.xs], 2.0; ks=0:0, h=h)[1]
                             for λc in (2, 4, 8)) / 1e-3
                a10 = maximum(radial_run(L, [ripple(x; d=d, λ=λc * h, h=h)
                                             for x in L.xs], 10.0; ks=0:0, h=h)[1]
                              for λc in (2, 4, 8)) / 1e-3
                if kw.mode === :layer
                    want = DISPERSION_1D[(q, d)]
                    for (got, w) in zip((a2, a10), want)
                        rounds_to(got, w) ||
                            error("the radial line's layer at q = $q, d = $d: " *
                                  "$got, dispersion.jl recorded $w")
                    end
                end
                push!(cells, fmt("%.2e, %.2e", a2, a10))
            end
            r_s = kw.mode === :layer ? 1.15 : kw.r_E
            xc = (r_s + R_H) / 2
            pulse = [1e-3 * exp(-((x - xc) / (2h))^2) for x in L.xs]
            p2 = radial_run(L, pulse, 2.0; ks=0:0, h=h)[1] / 1e-3
            p10 = radial_run(L, pulse, 10.0; ks=0:0, h=h)[1] / 1e-3
            spec = String[]
            for hh in (h, h / 2, h / 4)
                Lh = radial_line(q, 0.5, hh; kw...)
                M = Matrix(Lh.A)
                act = vcat(Lh.active, length(Lh.xs) .+ Lh.active)
                λs = eigvals(M[act, act])
                push!(spec, fmt("%+.2e", maximum(real, λs)))
            end
            say("| %s | %s | %s | %s | %.2e, %.2e | %s |", label, cells..., p2, p10,
                join(spec, ", "))
        end
    end
end end

# ==============================================================================
# model2d — the go/no-go: a lego circle on Kerr-Schild's equatorial plane
# ==============================================================================
#
# The square `[−L, L]²`, `L = 5/2 M` (the suite's fixture box), vertex-centered
# at `h = L/n` with the hole at the origin on a vertex, as on the octant; the
# ghosts beyond the square hold zero (Dirichlet data for a perturbation). One
# component's evolution on the plane `z = 0` of Kerr-Schild `a = 0`, with the
# coefficients varying:
#
#     ∂_t u = β^a Adv_a u + (α/√γ) v + (ε/h) Σ_a K_a u
#     ∂_t v = β^a Adv_a v + (∂_aβ^a) v + (∂_aA^{ab}) D_b u
#             + A^{xx} D_xx u + A^{yy} D_yy u + 2 A^{xy} D_xy u + (ε/h) Σ_a K_a v
#
# `a, b ∈ {x, y}`, `A = α√γ γ^{ij}` — the anisotropic `γ^{ij}`, so that an axis
# at angle `θ` to the radius has `b/a = H cos θ/√(1 + H sin²θ)` — and the two
# divergences over the plane. That is the kernel's discretization of
# `∂_t v = ∂_a(β^a v + A^{ab}∂_b u)`, `∂_t u = β^a∂_a u + (α/√γ) v`: the wave
# equation of a stationary 2+1 metric whose radial speeds are Kerr-Schild's,
# `−b ± a`, with its horizon at `r = 2M`, and a conserved Killing energy — a
# continuum without growing or static modes, so that what grows here is the
# discretization's **(decided in step X1**: the 3D divergences at `z = 0`, the
# first version, leave `(∂_zβ^z) v` and `(∂_zA^{zb}) ∂_b u`, lower-order terms
# with no energy behind them, and the `:damped` control grew at `+0.04/M`
# under them**)**. The excised set is the lego circle `r < r_E`.
#
# The two families:
#
#   * `:axis` — per-axis closures (`src/stencils.jl`): every operator along an
#     axis closed by the point's `(k⁻, k⁺)` on that axis, reach `G`, the
#     dissipation by `diss` (MSN by default), and the mixed derivative nested
#     — the outer sum along `x` with the point's `x`-closure, the inner along
#     `y` with the `y`-closure of the point `P + a e_x` it is taken at.
#   * `:extrap` — per-stencil extrapolation: every operator centered, and each
#     excised tap `Q` of the stencil of `P` replaced by the Lagrange
#     extrapolation of degree `≤ p` along the lattice direction (of the eight)
#     nearest the normal at `Q`, from the non-excised points `Q + k e` that lie
#     inside `P`'s `G`-box — what Cartesian excision codes have done, here
#     inside the existing reach.
#
# Each with the advection centered or blended toward the lopsided stencil,
# `C²` from `start` cells below the horizon over `width` cells. The control
# is the `:damped` layer of the octant runs (`r_0 = 3/4`, `r_1 = 3/2`,
# `ρ_ramp = 1`, `ρ_max = 4/M`).

const UNK = Int8(1)
const ZRO = Int8(0)
const EXC = Int8(-1)
const L_SQ = 2.5

struct Plane
    q::Int
    n::Int
    h::Float64
    st::Matrix{Int8}          # [i + n + 1, j + n + 1]
    idx::Matrix{Int}
    pts::Vector{NTuple{2,Int}}
    prof::Vector{NTuple{2,Float64}}
end

pstate(pl::Plane, i, j) = (abs(i) > pl.n || abs(j) > pl.n) ? ZRO :
                          pl.st[i + pl.n + 1, j + pl.n + 1]
pindex(pl::Plane, i, j) = pl.idx[i + pl.n + 1, j + pl.n + 1]

"""
The plane at `n` points per half-width: `interior = (:excise, r_E)` (excised
where `r < r_E`) or `(:layer, r_0, r_1, ρ_max)` (frozen where `w = 0`).
"""
function make_plane(q, n, interior)
    h = L_SQ / n
    st = fill(UNK, 2n + 1, 2n + 1)
    idx = zeros(Int, 2n + 1, 2n + 1)
    pts = NTuple{2,Int}[]
    prof = NTuple{2,Float64}[]
    int = interior[1] === :layer ?
          Interior(Float64; center=(0.0, 0.0, 0.0), r_0=interior[2],
                   r_1=interior[3], ρ_max=interior[4], ρ_ramp=1) : nothing
    for j in (-n):n, i in (-n):n
        r = h * sqrt(i^2 + j^2)
        if interior[1] === :excise
            r < interior[2] && (st[i + n + 1, j + n + 1] = EXC; continue)
            wρ = (1.0, 0.0)
        else
            wρ = interior_profiles(int, r)
            wρ[1] == 0 && (st[i + n + 1, j + n + 1] = ZRO; continue)
        end
        push!(pts, (i, j))
        push!(prof, (Float64(wρ[1]), Float64(wρ[2])))
        idx[i + n + 1, j + n + 1] = length(pts)
    end
    return Plane(q, n, h, st, idx, pts, prof)
end

# The coefficients on the plane, computed on the quadrant `i, j ≥ 0` and
# mirrored with their exact parities, so that the operator commutes with the
# two reflections bit for bit and its spectrum splits into four sectors.
function plane_coefficients(pl::Plane; bg=KS0)
    cache = Dict{NTuple{2,Int},Any}()
    get_c(i, j) = get!(() -> coefficients(bg, SVector(pl.h * i, pl.h * j, 0.0)),
                       cache, (i, j))
    return map(pl.pts) do (i, j)
        c = get_c(abs(i), abs(j))
        sx = i < 0 ? -1.0 : 1.0
        sy = j < 0 ? -1.0 : 1.0
        (βx=sx * c.β[1], βy=sy * c.β[2], cu=c.cu, Axx=c.A[1, 1], Ayy=c.A[2, 2],
         Axy=sx * sy * c.A[1, 2], divβ=c.divβ2, divAx=sx * c.divA2[1],
         divAy=sy * c.divA2[2], speed=c.speed)
    end
end

const DIRS8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

# The lattice direction (of the eight) nearest the outward normal at `Q`.
function nearest_direction(Q)
    r = hypot(Q[1], Q[2])
    r > 0 || error("an extrapolation was asked for at the center")
    best = DIRS8[1]
    bd = -Inf
    for e in DIRS8
        d = (e[1] * Q[1] + e[2] * Q[2]) / (r * hypot(e[1], e[2]))
        d > bd && (bd = d; best = e)
    end
    return best
end

# The extrapolation of the excised tap `Q` of `P`'s stencil: sources
# `Q + k e` inside `P`'s `G`-box, the first `p + 1` consecutive non-excised
# ones, Lagrange-extrapolated to `k = 0`.
function extrapolation(pl::Plane, P, Q, p)
    G = G_OF(pl.q)
    e = nearest_direction(Q)
    ks = Int[]
    for k in 1:(4G)
        S = (Q[1] + k * e[1], Q[2] + k * e[2])
        max(abs(S[1] - P[1]), abs(S[2] - P[2])) ≤ G || break
        if pstate(pl, S...) == EXC
            isempty(ks) || break
            continue
        end
        push!(ks, k)
        length(ks) == p + 1 && break
    end
    isempty(ks) && error("no extrapolation source for $Q in the G-box of $P")
    # The sources are consecutive, `k₀ … k₀ + n − 1`: the package's weights
    # (`extrapolation_weights`, added in step X5; until then the same
    # rationals from `lagrange_derivative_weights` here).
    ks == first(ks):last(ks) || error("the extrapolation's sources $ks are not consecutive")
    c = Float64.(extrapolation_weights(first(ks), length(ks))[2])
    return [((Q[1] + k * e[1], Q[2] + k * e[2]), ck) for (k, ck) in zip(ks, c)]
end

# The consecutive non-excised points from `P` along `(dx, dy)`, capped.
function kcount(pl::Plane, P, dx, dy, cap)
    k = 0
    while k < cap && pstate(pl, P[1] + (k + 1) * dx, P[2] + (k + 1) * dy) != EXC
        k += 1
    end
    return k
end

"""
The plane's operator for `fam ∈ (:centered, :axis, :extrap, :dirichlet)` —
the layer is `:centered`; `:dirichlet` reads every excised point as zero, the
frozen core of step 5 with no layer, which over-specifies an outflow
boundary — `ε`, the lopsided blend `lop = (start, width)` cells below the
horizon or `nothing`, the dissipation closure `diss`, the extrapolation
degree `p`.
"""
function plane_operator(pl::Plane, coef; fam=:axis, ε=0.5, lop=nothing, diss=:msn,
                        p=2, R=G_OF(pl.q))
    q, h = pl.q, pl.h
    N = length(pl.pts)
    G = G_OF(q)
    cap = max(R, G)
    c = Coo()
    big = 1000
    function emit!(row, P, Q, wt, f)
        s = pstate(pl, Q...)
        if s == UNK
            put!(c, row, f * N + pindex(pl, Q...), wt)
        elseif s == EXC && fam !== :dirichlet
            fam === :extrap || error("$fam read the excised point $Q from $P")
            for (S, ck) in extrapolation(pl, P, Q, p)
                emit!(row, P, S, wt * ck, f)
            end
        end
        return nothing
    end
    # One operator along axis `d` from `P`: offsets and weights.
    function axisop(kind, P, d; up=1)
        dx, dy = d == 1 ? (1, 0) : (0, 1)
        if fam === :axis
            km = kcount(pl, P, -dx, -dy, cap)
            kp = kcount(pl, P, dx, dy, cap)
        else
            km = kp = big
        end
        kind === :d1 && return wd(q, 1, km, kp, R)
        kind === :d2 && return wd(q, 2, km, kp, R)
        kind === :ko && return fam === :axis ? wk(q, diss, km, kp) : wkc(q)
        return fam === :axis ? wl(q, up, km, kp) : wlc(q, up)
    end
    for (k, P) in enumerate(pl.pts)
        w, ρ = pl.prof[k]
        cf = coef[k]
        ru, rv = k, N + k
        r = h * hypot(P...)
        λ = lop === nothing ? 0.0 : blend(r, h; start=lop[1], width=lop[2])
        for (d, β, divA, Add) in ((1, cf.βx, cf.divAx, cf.Axx),
                                  (2, cf.βy, cf.divAy, cf.Ayy))
            e = d == 1 ? (1, 0) : (0, 1)
            at(j) = (P[1] + j * e[1], P[2] + j * e[2])
            for (j, wj) in zip(axisop(:d1, P, d)...)
                for (row, f) in ((ru, 0), (rv, 1))
                    emit!(row, P, at(j), w * β * (1 - λ) * wj / h, f)
                end
                emit!(rv, P, at(j), w * divA * wj / h, 0)
            end
            if λ > 0
                for (j, wj) in zip(axisop(:lop, P, d; up=β >= 0 ? 1 : -1)...)
                    for (row, f) in ((ru, 0), (rv, 1))
                        emit!(row, P, at(j), w * β * λ * wj / h, f)
                    end
                end
            end
            for (j, wj) in zip(axisop(:d2, P, d)...)
                emit!(rv, P, at(j), w * Add * wj / h^2, 0)
            end
            for (j, wj) in zip(axisop(:ko, P, d)...)
                for (row, f) in ((ru, 0), (rv, 1))
                    emit!(row, P, at(j), w * ε * wj / h, f)
                end
            end
        end
        # The mixed derivative, nested: outer along x, inner along y.
        for (a, wa) in zip(axisop(:d1, P, 1)...)
            Pa = (P[1] + a, P[2])
            pstate(pl, Pa...) == EXC && fam === :axis &&
                error("the mixed derivative's outer sum read an excised point")
            inner = fam === :axis ? axisop(:d1, Pa, 2) : wc(q, 1)
            for (b, wb) in zip(inner...)
                emit!(rv, P, (Pa[1], Pa[2] + b), w * 2 * cf.Axy * wa * wb / h^2, 0)
            end
        end
        put!(c, ru, rv, w * cf.cu)
        put!(c, rv, rv, w * cf.divβ)
        put!(c, ru, ru, -ρ)
        put!(c, rv, rv, -ρ)
    end
    return tomatrix(c, 2N)
end

# The four parity sectors of the reflections `x → −x`, `y → −y`: the operator
# restricted to each, `(PᵀP)⁻¹ Pᵀ A P` on the quadrant's orbit
# representatives, checked against `A P = P A_s`.
function parity_sectors(pl::Plane, A)
    N = length(pl.pts)
    reps = [k for (k, P) in enumerate(pl.pts) if P[1] >= 0 && P[2] >= 0]
    out = []
    for (sx, sy) in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        cols = [k for k in reps
                if !((sx == -1 && pl.pts[k][1] == 0) || (sy == -1 && pl.pts[k][2] == 0))]
        m = length(cols)
        I = Int[]
        J = Int[]
        V = Float64[]
        for (cidx, k) in enumerate(cols)
            i, j = pl.pts[k]
            for img in unique([(a * i, b * j) for a in (1, -1), b in (1, -1)])
                s = (img[1] < 0 ? sx : 1) * (img[2] < 0 ? sy : 1)
                gi = pindex(pl, img...)
                gi > 0 || error("the plane is not mirror-symmetric at $img")
                append!(I, (gi, N + gi))
                append!(J, (cidx, m + cidx))
                append!(V, (s, s))
            end
        end
        Pm = sparse(I, J, Float64.(V), 2N, 2m)
        AP = A * Pm
        d = vec(sum(abs2, Pm; dims=1))
        As = sparse(Diagonal(1 ./ d)) * (transpose(Pm) * AP)
        res = norm(AP - Pm * As) / norm(AP)
        res ≤ 1e-12 || error("the operator does not commute with the reflections " *
                             "($sx, $sy): residual $res")
        push!(out, (s=(sx, sy), P=Pm, As=As))
    end
    return out
end


"""
The plane's spectrum by sector: the rightmost `Re λ` (in `1/M`), where its
mode lives — the share of its norm within three cells of the surface (or of
the layer's outer radius), and within three cells of the outer face — and the
RK4 step the spectrum allows as `cfl = dt λ_max / h`, `λ_max` the code's
largest speed.
"""
function plane_spectrum(pl::Plane, A, coef, r_surf)
    secs = parity_sectors(pl, A)
    N = length(pl.pts)
    best = (re=-Inf, sec=0, k=0)
    all_λ = ComplexF64[]
    vals = Vector{Vector{ComplexF64}}()
    for (si, s) in enumerate(secs)
        λs = eigvals(Matrix(s.As))
        push!(vals, λs)
        append!(all_λ, λs)
        k = argmax(real.(λs))
        real(λs[k]) > best.re && (best = (re=real(λs[k]), sec=si, k=k))
    end
    # Where the rightmost mode lives, when it is not plainly decaying (the
    # eigenvectors cost twice the eigenvalues).
    near = outer = NaN
    if best.re > -1e-2
        s = secs[best.sec]
        F = eigen(Matrix(s.As))
        k = argmin(abs.(F.values .- vals[best.sec][best.k]))
        v = s.P * F.vectors[:, k]
        wts = [abs2(v[i]) + abs2(v[N + i]) for i in 1:N]
        tot = sum(wts)
        near = sum(wts[i] for i in 1:N
                   if pl.h * hypot(pl.pts[i]...) < r_surf + 3pl.h; init=0.0) / tot
        outer = sum(wts[i] for i in 1:N
                    if max(abs.(pl.pts[i])...) > pl.n - 3; init=0.0) / tot
    end
    λmax = maximum(c.speed for c in coef)
    ν = rk4_step_limit(all_λ)
    return (maxre=best.re, near=near, outer=outer, cfl=ν * λmax / pl.h,
            nzero=count(<(1e-6), abs.(all_λ)))
end

# The lego circle's closure faces and their per-axis ratios, on the plane.
function plane_faces(pl::Plane, coef)
    ratios = Float64[]
    for (k, P) in enumerate(pl.pts), (d, s) in ((1, 1), (1, -1), (2, 1), (2, -1))
        e = d == 1 ? (s, 0) : (0, s)
        pstate(pl, P[1] + e[1], P[2] + e[2]) == EXC || continue
        c = coef[k]
        β = d == 1 ? c.βx : c.βy
        x = SVector(pl.h * P[1], pl.h * P[2], 0.0)
        cc = coefficients(KS0, x)
        push!(ratios, -s * β / (cc.α * sqrt(cc.γu[d, d])))
    end
    return ratios
end

fmt_re(s) = s.maxre ≤ 1e-9 ? fmt("%+.1e", s.maxre) :
            fmt("**%+.2e**%s", s.maxre,
                s.near > 0.5 ? " (s)" : s.outer > 0.5 ? " (o)" : "")

const R_E_LIST = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
const LAYER = (:layer, 0.75, 1.5, 4.0)
const LOP = (1, 4)

if runs("model2d", "eig")
    println("\n=== model2d, eig: the dense spectrum on Kerr-Schild's equatorial " *
            "plane, [−5/2, 5/2]², by parity sector ===")
    println("each entry: the largest Re λ in 1/M (bold when above 1e-9; '(s)' " *
            "when its mode has more than half its norm within three cells of " *
            "the surface, '(o)' near the outer face) / the RK4 cfl = dt λ_max/h " *
            "the spectrum allows. axis: per-axis closures (MSN); extrap: " *
            "per-stencil quadratic extrapolation along the lattice direction " *
            "nearest the normal, inside the G-box; lop: the advection lopsided " *
            "from 1 cell below the horizon over 4")
    for q in parse_list(Int, "q", (2, 4)), n in parse_list(Int, "n", (24, 32))
        h = L_SQ / n
        println("\n-- q = $q, n = $n (h = $(round(h; digits=4)) = 5/$(round(Int, 5/h))) --")
        for ε in parse_list(Float64, "eps", (0.5, 1.0))
            pl = make_plane(q, n, LAYER)
            coef = plane_coefficients(pl)
            sref = plane_spectrum(pl, plane_operator(pl, coef; fam=:centered, ε=ε),
                                  coef, LAYER[3])
            say("\nε_KO = %.2f; the :damped layer (r_0 = 3/4, r_1 = 3/2, 4/M): %s / %.2f",
                ε, fmt_re(sref), sref.cfl)
            println("| r_E/M | r_E/h | faces | inflow | axis | axis, lop | extrap | " *
                    "extrap, lop |")
            println("|---|---|---|---|---|---|---|---|")
            for r_E in parse_list(Float64, "rE", R_E_LIST)
                pl = make_plane(q, n, (:excise, r_E))
                coef = plane_coefficients(pl)
                fr = plane_faces(pl, coef)
                cells = String[]
                for (fam, lop) in ((:axis, nothing), (:axis, LOP), (:extrap, nothing),
                                   (:extrap, LOP))
                    A = plane_operator(pl, coef; fam=fam, ε=ε, lop=lop)
                    s = plane_spectrum(pl, A, coef, r_E)
                    push!(cells, fmt("%s / %.2f", fmt_re(s), s.cfl))
                end
                say("| %.2f | %.1f | %d | %.3f | %s |", r_E, r_E / h, length(fr),
                    count(<(1), fr) / length(fr), join(cells, " | "))
            end
        end
    end
end

# --- model2d, controls: what the plane does to a closure that should fail ------
#
# The same spectrum for the variations the go/no-go is to be read against:
# no dissipation (`ε_KO = 0`: closures without SBP, whose stability CODE.md
# says rests on it), the other two dissipation closures, the extrapolation at
# the largest degree the `G`-box holds, and the frozen core without a layer
# (`:dirichlet`), at `r_E = M` and `3M/2`.

if runs("model2d", "controls")
    println("\n=== model2d, controls: the plane's spectrum (largest Re λ in 1/M " *
            "/ RK4 cfl) for variations the go/no-go is read against ===")
    rows = (("the :damped layer, ε = 0 (independent of r_E)", (; fam=:layer, ε=0.0)),
            ("axis, ε = 0", (; fam=:axis, ε=0.0)),
            ("extrap, ε = 0", (; fam=:extrap, ε=0.0)),
            ("axis, reduced rank", (; fam=:axis, diss=:reduced)),
            ("axis, one-sided", (; fam=:axis, diss=:onesided)),
            ("extrap, degree ≤ q", (; fam=:extrap, p=8)),
            ("extrap, degree 1", (; fam=:extrap, p=1)),
            ("Dirichlet core", (; fam=:dirichlet)),
            ("Dirichlet core, ε = 0", (; fam=:dirichlet, ε=0.0)))
    for q in parse_list(Int, "q", (2, 4)), n in parse_list(Int, "n", (24, 32))
        h = L_SQ / n
        println("\n-- q = $q, n = $n (h = 5/$(round(Int, 5/h))), ε_KO = 1/2 unless " *
                "stated --")
        rEs = parse_list(Float64, "rE", (1.0, 1.5))
        println("| variation | " * join(["r_E = $r" for r in rEs], " | ") * " |")
        println("|---|" * repeat("---|", length(rEs)))
        for (label, kw) in rows
            cells = String[]
            for r_E in rEs
                if kw.fam === :layer
                    pl = make_plane(q, n, LAYER)
                    coef = plane_coefficients(pl)
                    A = plane_operator(pl, coef; fam=:centered, ε=kw.ε)
                    s = plane_spectrum(pl, A, coef, LAYER[3])
                else
                    pl = make_plane(q, n, (:excise, r_E))
                    coef = plane_coefficients(pl)
                    A = plane_operator(pl, coef; kw...)
                    s = plane_spectrum(pl, A, coef, r_E)
                end
                push!(cells, fmt("%s / %.2f", fmt_re(s), s.cfl))
            end
            say("| %s | %s |", label, join(cells, " | "))
        end
    end
end

# --- model2d, noise: long evolutions on finer grids ----------------------------
#
# Uniform noise in `[−1, 1]` on every unknown, RK4 at `cfl = dt λ_max/h = 1/2`
# (the octant runs'), the `l²` norm every `M`. A run whose norm passes `1e8`
# of its start, or goes non-finite, has blown up; otherwise the late rate is
# the least-squares slope of `log ‖u‖` over the last 40 % of the run.

function plane_noise(pl::Plane, A, coef; cfl=0.5, t_end=100.0, seed=20261005)
    N2 = size(A, 1)
    rng = MersenneTwister(seed)
    u = 2 .* rand(rng, N2) .- 1
    n0 = norm(u)
    λmax = maximum(c.speed for c in coef)
    nper = ceil(Int, 1 / (cfl * pl.h / λmax))
    dt = 1 / nper
    w = RK4Work(N2)
    ts = Float64[]
    ns = Float64[]
    for m in 1:round(Int, t_end)
        for _ in 1:nper
            rk4_step!(u, A, dt, w)
        end
        nr = norm(u) / n0
        push!(ts, m)
        push!(ns, nr)
        (isfinite(nr) && nr < 1e8) || return (ns=ns, rate=NaN, blowup=m)
    end
    sel = ts .>= 0.6 * t_end
    x = ts[sel]
    y = log.(ns[sel])
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    rate = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    return (ns=ns, rate=rate, blowup=NaN)
end

function fmt_noise(r, t_end)
    isnan(r.blowup) || return fmt("**blows up at %d M**", r.blowup)
    at(t) = r.ns[min(round(Int, t), length(r.ns))]
    return fmt("%.1e, %.1e, %.1e; %s/M", at(10), at(t_end / 2), at(t_end),
               r.rate > 1e-4 ? fmt("**%+.1e**", r.rate) : fmt("%+.1e", r.rate))
end

# Every configuration of a noise part, run on the threads there are, each
# building its own plane and operator.
function noise_table(q, n, rEs, famlops; ε=0.5, t_end=100.0, layer=true)
    jobs = Any[]
    layer && push!(jobs, (:layer, nothing, nothing))
    for r_E in rEs, (fam, lop) in famlops
        push!(jobs, (r_E, fam, lop))
    end
    res = Vector{Any}(undef, length(jobs))
    Threads.@threads :dynamic for k in eachindex(jobs)
        r_E, fam, lop = jobs[k]
        if r_E === :layer
            pl = make_plane(q, n, LAYER)
            coef = plane_coefficients(pl)
            A = plane_operator(pl, coef; fam=:centered, ε=ε)
        else
            pl = make_plane(q, n, (:excise, r_E))
            coef = plane_coefficients(pl)
            A = plane_operator(pl, coef; fam=fam, ε=ε, lop=lop)
        end
        res[k] = plane_noise(pl, A, coef; t_end=t_end)
    end
    return jobs, res
end

if runs("model2d", "noise") || runs("model2d", "fine")
    println("\n=== model2d, noise: uniform noise on every unknown, RK4 at cfl = 1/2, " *
            "ε_KO = 1/2; ‖u‖/‖u₀‖ at 10 M, half-way and the end, and the late " *
            "rate (bold above 1e-4/M) ===")
end

if runs("model2d", "noise")
    t_end = opt(Float64, "t_end", 100.0)
    n = opt(Int, "n", 64)
    famlops = ((:axis, nothing), (:axis, LOP), (:extrap, nothing), (:extrap, LOP))
    for q in parse_list(Int, "q", (2, 4))
        jobs, res = noise_table(q, n, parse_list(Float64, "rE", R_E_LIST), famlops;
                                t_end=t_end)
        h = L_SQ / n
        println("\n-- q = $q, n = $n (h = 5/$(round(Int, 5/h))), to $(t_end) M --")
        say("the :damped layer: %s", fmt_noise(res[1], t_end))
        println("| r_E/M | axis | axis, lop | extrap | extrap, lop |")
        println("|---|---|---|---|---|")
        for (i, r_E) in enumerate(parse_list(Float64, "rE", R_E_LIST))
            cells = [fmt_noise(res[1 + 4(i - 1) + k], t_end) for k in 1:4]
            say("| %.2f | %s |", r_E, join(cells, " | "))
        end
    end
end

if runs("model2d", "fine")
    t_end = opt(Float64, "t_end", 100.0)
    n = opt(Int, "n_fine", 128)
    fams = Dict("axis" => (:axis, nothing), "axislop" => (:axis, LOP),
                "extrap" => (:extrap, nothing), "extraplop" => (:extrap, LOP))
    names = haskey(OPTS, "fam") ? split(OPTS["fam"], ',') :
            ["axis", "axislop", "extrap", "extraplop"]
    famlops = [fams[k] for k in names]
    for q in parse_list(Int, "q", (2, 4))
        rEs = parse_list(Float64, "rE", R_E_LIST)
        jobs, res = noise_table(q, n, rEs, famlops; t_end=t_end)
        h = L_SQ / n
        println("\n-- q = $q, n = $n (h = 5/$(round(Int, 5/h))), to $(t_end) M --")
        say("the :damped layer: %s", fmt_noise(res[1], t_end))
        println("| r_E/M | " * join(names, " | ") * " |")
        println("|---|" * repeat("---|", length(names)))
        for (i, r_E) in enumerate(rEs)
            cells = [fmt_noise(res[1 + length(names) * (i - 1) + k], t_end)
                     for k in 1:length(names)]
            say("| %.2f | %s |", r_E, join(cells, " | "))
        end
    end
end

# ==============================================================================
# model2d, spin — the frame-dragged faces (step X5)
# ==============================================================================
#
# The same plane on Kerr-Schild at spin `a` (`a=…`, default `3/5`), whose
# equatorial plane is where frame dragging lies in the plane: `β` has an
# azimuthal part, so that along an axis nearly tangent to the lego circle the
# shift can point *into* the excised set (`b/a < 0`), the faces X1's frozen
# line found unstable under the per-axis closure. The model is X1's, in its
# conservation form; what changes:
#
#   * the coefficients are no longer mirror-symmetric (the spin picks a sense
#     of rotation), so they are computed on the orbit representatives
#     `i > 0, j ≥ 0` of the quarter turn `(i, j) → (−j, i)` and turned with
#     their exact tensor rules — `β` and `∂_aA^{ab}` as vectors, `A^{ab}` as a
#     tensor — and the spectrum splits into the turn's four sectors (the
#     eigenvalues `1, i, −1, −i`; the last is the second's conjugate);
#   * the horizon the lopsided blend is measured from is the sphere
#     geometry's, `r₊` (`horizon_min_radius`), as X2b's `ExcisionBlend` reads
#     it for the sphere — on the equator that starts the blend about one
#     cell of `5/64` deeper than the plane's own horizon;
#   * the families: `:axis` and `:extrap` are X1's; **`:hybrid`** uses, at a
#     point and an axis, the per-axis closure unless that axis is a closure
#     axis (`k_s < G` on side `s`) whose shift points into the excised set,
#     `−s β^d < 0`, and there every operator along the axis — `D₁`, `D₂`, the
#     dissipation, the lopsided advection and the mixed derivative's sum
#     along it — is centered with each excised tap extrapolated as `:extrap`
#     does; **`:hybridadv`** does that for the advection's derivative only,
#     the one the provider's `adv` method forms, and keeps the closures for
#     everything else. The nested mixed derivative takes, at each outer node,
#     that node's own rule along the inner axis; an outer node that is
#     excised (only under an extrapolated outer sum) has its inner sum
#     extrapolated tap by tap.

const SPIN_RE = (0.65, 0.75, 0.90, 1.10, 1.30, 1.50, 1.70)
const SPIN_FAMS = ((:axis, nothing), (:axis, LOP), (:extrap, nothing), (:extrap, LOP),
                   (:hybrid, nothing), (:hybrid, LOP), (:hybridadv, nothing),
                   (:hybridadv, LOP))
const SPIN_FAMNAMES = Dict(:axis => "axis", :extrap => "extrap", :hybrid => "hybrid",
                           :hybridadv => "hybrid-adv", :normal => "normal",
                           :hybridnormal => "hybrid-normal")
famname(fam, lop) = SPIN_FAMNAMES[fam] * (lop === nothing ? "" : ", lop")

# A quarter turn of a lattice point, and the turn that carries the orbit
# representative (`i > 0, j ≥ 0`) to a point: `P = R^k(P₀)`.
qturn(P) = (-P[2], P[1])
function orbit_rep(P)
    Q = P
    for k in 0:3
        Q[1] > 0 && Q[2] >= 0 && return Q, k
        Q = qturn(qturn(qturn(Q)))         # R⁻¹
    end
    error("the center has no orbit representative")
end

"""
The plane's coefficients on Kerr-Schild `bg`, exactly covariant under the
quarter turn: computed at the orbit representatives and turned `k` times —
`(vx, vy) → (−vy, vx)` for `β` and `∂_aA^{ab}`, `(A^{xx}, A^{yy}, A^{xy}) →
(A^{yy}, A^{xx}, −A^{xy})`, and the axis light speeds `a_x = α√γ^{xx}`,
`a_y` exchanged. Checked against the metric at the turned points.
"""
function plane_coefficients_turn(pl::Plane, bg)
    cache = Dict{NTuple{2,Int},Any}()
    function rep_coef(P0)
        get!(cache, P0) do
            c = coefficients(bg, SVector(pl.h * P0[1], pl.h * P0[2], 0.0))
            c === nothing && error("the plane's evolved point $P0 is on the " *
                                   "singular set")
            (βx=c.β[1], βy=c.β[2], cu=c.cu, Axx=c.A[1, 1], Ayy=c.A[2, 2],
             Axy=c.A[1, 2], divβ=c.divβ2, divAx=c.divA2[1], divAy=c.divA2[2],
             speed=c.speed, ax=c.α * sqrt(c.γu[1, 1]), ay=c.α * sqrt(c.γu[2, 2]))
        end
    end
    turn(c) = (βx=-c.βy, βy=c.βx, cu=c.cu, Axx=c.Ayy, Ayy=c.Axx, Axy=-c.Axy,
               divβ=c.divβ, divAx=-c.divAy, divAy=c.divAx, speed=c.speed,
               ax=c.ay, ay=c.ax)
    coef = map(pl.pts) do P
        P0, k = orbit_rep(P)
        c = rep_coef(P0)
        for _ in 1:k
            c = turn(c)
        end
        c
    end
    # The turn rules against the metric, at a few points of every quadrant.
    for (k, P) in enumerate(pl.pts)
        k % 97 == 0 || continue
        c = coefficients(bg, SVector(pl.h * P[1], pl.h * P[2], 0.0))
        t = coef[k]
        err = maximum(abs, (t.βx - c.β[1], t.βy - c.β[2], t.Axx - c.A[1, 1],
                            t.Ayy - c.A[2, 2], t.Axy - c.A[1, 2],
                            t.divAx - c.divA2[1], t.divAy - c.divA2[2]))
        err ≤ 1e-11 * (1 + abs(c.A[1, 1])) ||
            error("the turned coefficients at $P are off the metric's by $err")
    end
    return coef
end

"""
The plane's operator for every family (`fam ∈ (:axis, :extrap, :hybrid,
:hybridadv, :centered, :dirichlet)`), with the coefficients of
`plane_coefficients_turn`: X1's `plane_operator` with the rule chosen per
point, axis and operator rather than per family. At `a = 0`, `:axis`,
`:extrap`, `:centered` and `:dirichlet` are X1's operators (checked below to
roundoff; the summation order of the assembly differs).
"""
function spin_operator(pl::Plane, coef, rh; fam=:axis, ε=0.5, lop=nothing,
                       diss=:msn, p=2, thresh=0.0)
    q, h = pl.q, pl.h
    N = length(pl.pts)
    G = G_OF(q)
    c = Coo()
    big = 1000
    kc(P, d, s) = kcount(pl, P, d == 1 ? s : 0, d == 2 ? s : 0, G)
    # A closure axis whose shift points into the excised set, `b/a < 0` — or,
    # for the controls, `b/a < thresh`.
    function into(P, d)
        k = (abs(P[1]) > pl.n || abs(P[2]) > pl.n) ? 0 : pindex(pl, P...)
        k == 0 && return false
        β, a = d == 1 ? (coef[k].βx, coef[k].ax) : (coef[k].βy, coef[k].ay)
        return any(s -> kc(P, d, s) < G && -s * β / a < thresh, (-1, 1))
    end
    # The rule at `P` along `d` for an operator `kind`.
    function mode(P, d, kind)
        fam === :axis && return :closure
        fam === :extrap && return :extrap
        fam in (:centered, :dirichlet) && return :open
        fam === :hybrid && return into(P, d) ? :extrap : :closure
        fam === :hybridadv &&
            return (kind in (:adv, :lop) && into(P, d)) ? :extrap : :closure
        error("unknown family $fam")
    end
    function weights(kind, P, d, md; up=1)
        km, kp = md === :closure ? (kc(P, d, -1), kc(P, d, 1)) : (big, big)
        kind in (:adv, :d1, :mix) && return wd(q, 1, km, kp, G)
        kind === :d2 && return wd(q, 2, km, kp, G)
        kind === :ko && return md === :closure ? wk(q, diss, km, kp) : wkc(q)
        return md === :closure ? wl(q, up, km, kp) : wlc(q, up)
    end
    function emit!(row, P, Q, wt, f, md)
        s = pstate(pl, Q...)
        if s == UNK
            put!(c, row, f * N + pindex(pl, Q...), wt)
        elseif s == EXC && fam !== :dirichlet
            md === :extrap || error("$fam ($md) read the excised point $Q from $P")
            for (S, ck) in extrapolation(pl, P, Q, p)
                emit!(row, P, S, wt * ck, f, md)
            end
        end
        return nothing
    end
    for (k, P) in enumerate(pl.pts)
        w, ρ = pl.prof[k]
        cf = coef[k]
        ru, rv = k, N + k
        r = h * hypot(P...)
        λ = lop === nothing ? 0.0 : blend_below(r, h, rh; start=lop[1], width=lop[2])
        for (d, β, divA, Add) in ((1, cf.βx, cf.divAx, cf.Axx),
                                  (2, cf.βy, cf.divAy, cf.Ayy))
            e = d == 1 ? (1, 0) : (0, 1)
            at(j) = (P[1] + j * e[1], P[2] + j * e[2])
            ma = mode(P, d, :adv)
            for (j, wj) in zip(weights(:adv, P, d, ma)...)
                for (row, f) in ((ru, 0), (rv, 1))
                    emit!(row, P, at(j), w * β * (1 - λ) * wj / h, f, ma)
                end
            end
            md = mode(P, d, :d1)
            for (j, wj) in zip(weights(:d1, P, d, md)...)
                emit!(rv, P, at(j), w * divA * wj / h, 0, md)
            end
            if λ > 0
                ml = mode(P, d, :lop)
                for (j, wj) in zip(weights(:lop, P, d, ml; up=β >= 0 ? 1 : -1)...)
                    for (row, f) in ((ru, 0), (rv, 1))
                        emit!(row, P, at(j), w * β * λ * wj / h, f, ml)
                    end
                end
            end
            m2 = mode(P, d, :d2)
            for (j, wj) in zip(weights(:d2, P, d, m2)...)
                emit!(rv, P, at(j), w * Add * wj / h^2, 0, m2)
            end
            mk = mode(P, d, :ko)
            for (j, wj) in zip(weights(:ko, P, d, mk)...)
                for (row, f) in ((ru, 0), (rv, 1))
                    emit!(row, P, at(j), w * ε * wj / h, f, mk)
                end
            end
        end
        # The mixed derivative, nested: outer along x by `P`'s rule, inner
        # along y by the outer node's own.
        mx = mode(P, 1, :mix)
        for (a, wa) in zip(weights(:mix, P, 1, mx)...)
            Pa = (P[1] + a, P[2])
            if pstate(pl, Pa...) == EXC
                (mx === :extrap || fam === :dirichlet) ||
                    error("the mixed derivative's outer sum read an excised point")
                my = mx
                inner = wc(q, 1)
            else
                my = mode(Pa, 2, :mix)
                inner = weights(:mix, Pa, 2, my)
            end
            for (b, wb) in zip(inner...)
                emit!(rv, P, (Pa[1], Pa[2] + b), w * 2 * cf.Axy * wa * wb / h^2, 0,
                      my)
            end
        end
        put!(c, ru, rv, w * cf.cu)
        put!(c, rv, rv, w * cf.divβ)
        put!(c, ru, ru, -ρ)
        put!(c, rv, rv, -ρ)
    end
    return tomatrix(c, 2N)
end

# The lopsided blend `C²` in the depth below `rh`: zero at and above `start`
# cells below it, one from `start + width` on.
blend_below(x, h, rh; start, width) =
    width ≤ 0 ? 0.0 : smoothstep(((rh - x) / h - start) / width)

# The sectors of the turns: the operator restricted to the span of
# `Σ_k μ^{−k} δ_{R^k P₀}` over the orbit representatives `P₀` of the turn
# `R` by `2π/order`, `μ = e^{2πi m/order}`, checked against `A P = P A_m`.
# `order = 4` is the quarter turn, whose sector 3 is sector 1's conjugate and
# is not computed; it commutes with the operator only where the mixed
# derivative is symmetric in its two axes — the centered stencil, and
# `:extrap`'s tap by tap — since the nested closures take the outer sum along
# `x` at every point (as the kernel's `dmix` does). `order = 2`, the half
# turn `P → −P`, keeps that order and commutes with every family.
function turn_sectors(pl::Plane, A; order=4)
    N = length(pl.pts)
    rep(P) = order == 4 ? (P[1] > 0 && P[2] >= 0) : (P[1] > 0 || (P[1] == 0 && P[2] > 0))
    step(P) = order == 4 ? qturn(P) : (-P[1], -P[2])
    reps = [k for (k, P) in enumerate(pl.pts) if rep(P)]
    nr = length(reps)
    order * nr == N || error("the plane's evolved points are not whole orbits of " *
                             "the turn ($N points, $nr representatives)")
    out = []
    for m in (order == 4 ? (0, 1, 2) : (0, 1))
        μ = cispi(2m / order)
        I = Int[]
        J = Int[]
        V = ComplexF64[]
        for (cidx, k) in enumerate(reps)
            P = pl.pts[k]
            for kk in 0:(order - 1)
                gi = pindex(pl, P...)
                gi > 0 || error("the plane is not invariant under the turn at $P")
                cf = conj(μ)^kk
                append!(I, (gi, N + gi))
                append!(J, (cidx, nr + cidx))
                append!(V, (cf, cf))
                P = step(P)
            end
        end
        real_sector = order == 2 || m != 1
        Pm = real_sector ? sparse(I, J, real.(V), 2N, 2nr) : sparse(I, J, V, 2N, 2nr)
        AP = A * Pm
        As = (Pm' * AP) ./ order
        res = norm(AP - Pm * As) / norm(AP)
        res ≤ 1e-12 || error("the operator does not commute with the turn of order " *
                             "$order (sector $m): residual $res")
        push!(out, (m=m, P=Pm, As=As, conjugate=(order == 4 && m == 1)))
    end
    return out
end

# The turn a family's operator commutes with.
turn_order(fam) = fam in (:centered, :extrap, :dirichlet) ? 4 : 2

"""
The spinning plane's spectrum by sector, as `plane_spectrum`: the rightmost
`Re λ` (in `1/M`), the share of its mode within three cells of the surface
and of the outer face, the RK4 `cfl` the spectrum allows, and its
eigenvalue's imaginary part.
"""
function spin_spectrum(pl::Plane, A, coef, r_surf; order=2)
    secs = turn_sectors(pl, A; order=order)
    N = length(pl.pts)
    best = (re=-Inf, sec=0, k=0)
    all_λ = ComplexF64[]
    vals = Vector{Vector{ComplexF64}}()
    for (si, s) in enumerate(secs)
        λs = ComplexF64.(eigvals(Matrix(s.As)))
        push!(vals, λs)
        append!(all_λ, λs)
        s.conjugate && append!(all_λ, conj.(λs))
        k = argmax(real.(λs))
        real(λs[k]) > best.re && (best = (re=real(λs[k]), sec=si, k=k))
    end
    near = outer = NaN
    if best.re > -1e-2
        s = secs[best.sec]
        F = eigen(Matrix(s.As))
        k = argmin(abs.(F.values .- vals[best.sec][best.k]))
        v = s.P * F.vectors[:, k]
        wts = [abs2(v[i]) + abs2(v[N + i]) for i in 1:N]
        tot = sum(wts)
        near = sum(wts[i] for i in 1:N
                   if pl.h * hypot(pl.pts[i]...) < r_surf + 3pl.h; init=0.0) / tot
        outer = sum(wts[i] for i in 1:N
                    if max(abs.(pl.pts[i])...) > pl.n - 3; init=0.0) / tot
    end
    λmax = maximum(c.speed for c in coef)
    ν = rk4_step_limit(all_λ)
    return (maxre=best.re, im=imag(vals[best.sec][best.k]), near=near, outer=outer,
            cfl=ν * λmax / pl.h)
end

# The lego circle's faces and closure axes on the spinning plane: X1's face
# ratios, and the closure axes (`k_s < G`) whose shift points into the excised
# set — the axes `:hybrid` extrapolates.
function spin_faces(pl::Plane, coef)
    G = G_OF(pl.q)
    faces = Float64[]
    nin = 0
    for (k, P) in enumerate(pl.pts), d in 1:2, s in (-1, 1)
        β, a = d == 1 ? (coef[k].βx, coef[k].ax) : (coef[k].βy, coef[k].ay)
        ratio = -s * β / a
        kcount(pl, P, d == 1 ? s : 0, d == 2 ? s : 0, G) < G && ratio < 0 &&
            (nin += 1)
        e = d == 1 ? (s, 0) : (0, s)
        pstate(pl, P[1] + e[1], P[2] + e[2]) == EXC && push!(faces, ratio)
    end
    return sort!(faces), nin
end

spin_background() = SM.KerrSchild(1.0, opt(Float64, "a", 0.6))

# The self-check of the general operator: at `a = 0` its `:axis`, `:extrap`
# and `:centered` operators are X1's, to roundoff.
if any(p -> runs("model2d", p), ("spineig", "spincontrols", "spinnoise", "spinfine"))
    for (fam, lop, interior) in ((:axis, LOP, (:excise, 1.0)),
                                 (:extrap, LOP, (:excise, 1.0)),
                                 (:centered, nothing, LAYER))
        pl = make_plane(4, 16, interior)
        A0 = plane_operator(pl, plane_coefficients(pl); fam=fam, lop=lop)
        A1 = spin_operator(pl, plane_coefficients_turn(pl, KS0), R_H; fam=fam,
                           lop=lop)
        err = norm(A0 - A1) / norm(A0)
        err ≤ 1e-13 || error("spin_operator($fam) at a = 0 is off X1's " *
                             "plane_operator by $err")
    end
end

fmt_spin(s) = s.maxre ≤ 1e-9 ? fmt("%+.3f", s.maxre) :
              fmt("**%+.2e**%s", s.maxre,
                  s.near > 0.5 ? " (s)" : s.outer > 0.5 ? " (o)" : "")

if runs("model2d", "spineig")
    bg = spin_background()
    rh = horizon_min_radius(bg)
    fams = haskey(OPTS, "fam") ?
           [f for f in SPIN_FAMS if replace(famname(f...), ", " => "", "-" => "") in
                                    split(OPTS["fam"], ',')] : collect(SPIN_FAMS)
    println("\n=== model2d, spineig: the dense spectrum on Kerr-Schild a = " *
            "$(bg.spin)'s equatorial plane, [−5/2, 5/2]², by the sectors of the half " *
            "turn (the quarter turn where the family allows it) ===")
    println("each entry: the largest Re λ in 1/M (bold above 1e-9; '(s)' when " *
            "its mode has more than half its norm within three cells of the " *
            "surface, '(o)' near the outer face) / the RK4 cfl = dt λ_max/h. " *
            "faces: the lego circle's immediate closure faces, their least b/a, " *
            "and the fractions with b/a < 1 and b/a < 0; into: the closure axes " *
            "(k_s < G) whose shift points into the excised set. lop: the advection " *
            "lopsided from 1 cell below r₊ = $rh over 4")
    for q in parse_list(Int, "q", (4, 2)), n in parse_list(Int, "n", (24, 32))
        h = L_SQ / n
        println("\n-- q = $q, n = $n (h = $(round(h; digits=4)) = 5/$(round(Int, 5/h))) --")
        for ε in parse_list(Float64, "eps", (0.5, 1.0))
            pl = make_plane(q, n, LAYER)
            coef = plane_coefficients_turn(pl, bg)
            sref = spin_spectrum(pl, spin_operator(pl, coef, rh; fam=:centered, ε=ε),
                                 coef, LAYER[3]; order=4)
            say("\nε_KO = %.2f; the :damped layer (r_0 = 3/4, r_1 = 3/2, 4/M): %s / %.2f",
                ε, fmt_spin(sref), sref.cfl)
            println("| r_E/M | r_E/h | faces | b/a min | inflow | b/a < 0 | into | " *
                    join([famname(f...) for f in fams], " | ") * " |")
            println("|---|---|---|---|---|---|---|" * repeat("---|", length(fams)))
            for r_E in parse_list(Float64, "rE", SPIN_RE)
                pl = make_plane(q, n, (:excise, r_E))
                coef = plane_coefficients_turn(pl, bg)
                fr, nin = spin_faces(pl, coef)
                cells = String[]
                for (fam, lop) in fams
                    A = spin_operator(pl, coef, rh; fam=fam, ε=ε, lop=lop)
                    s = spin_spectrum(pl, A, coef, r_E; order=turn_order(fam))
                    push!(cells, fmt("%s / %.2f", fmt_spin(s), s.cfl))
                end
                say("| %.2f | %.1f | %d | %.2f | %.3f | %.3f | %d | %s |", r_E, r_E / h,
                    length(fr), fr[1], count(<(1), fr) / length(fr),
                    count(<(0), fr) / length(fr), nin, join(cells, " | "))
            end
        end
    end
end

# --- model2d, spinfaces: the lego circle's faces at every resolution ---------
#
# What the spinning plane's tables are read against: at each `n` of the
# spectra and the noise runs, the lego circle's faces, the least `b/a`, how
# many have the shift pointing into the excised set (`b/a < 0`) and how many
# have both characteristics entering from it (`b/a < −1`), and the closure
# axes `:hybrid` extrapolates.

if runs("model2d", "spinfaces")
    bg = spin_background()
    println("\n=== model2d, spinfaces: the lego circle's faces on Kerr-Schild a = " *
            "$(bg.spin)'s plane, q = 4 (into: closure axes with b/a < 0) ===")
    ns = parse_list(Int, "n", (24, 32, 48, 64, 128))
    println("| r_E/M | " * join(["n = $n: faces, b/a min, < 0, < −1, into" for n in ns],
                                 " | ") * " |")
    println("|---|" * repeat("---|", length(ns)))
    for r_E in parse_list(Float64, "rE", SPIN_RE)
        cells = String[]
        for n in ns
            pl = make_plane(parse_list(Int, "q", (4,))[1], n, (:excise, r_E))
            fr, nin = spin_faces(pl, plane_coefficients_turn(pl, bg))
            push!(cells, fmt("%d, %.2f, %d, %d, %d", length(fr), fr[1], count(<(0), fr),
                             count(<(-1), fr), nin))
        end
        say("| %.2f | %s |", r_E, join(cells, " | "))
    end
end

# --- model2d, spincontrols: what the spinning plane is read against ----------
#
# X1's controls on the spinning plane: no dissipation, the other dissipation
# closures, the extrapolation's degree, the bare frozen core.

if runs("model2d", "spincontrols")
    bg = spin_background()
    rh = horizon_min_radius(bg)
    println("\n=== model2d, spincontrols: the spinning plane's spectrum (largest Re λ " *
            "in 1/M / RK4 cfl) for the variations the go/no-go is read against, " *
            "Kerr-Schild a = $(bg.spin) ===")
    rows = (("the :damped layer, ε = 0 (independent of r_E)", (; fam=:layer, ε=0.0)),
            ("axis, ε = 0", (; fam=:axis, ε=0.0)),
            ("extrap, ε = 0", (; fam=:extrap, ε=0.0)),
            ("hybrid, ε = 0", (; fam=:hybrid, ε=0.0)),
            ("hybrid-adv, ε = 0", (; fam=:hybridadv, ε=0.0)),
            ("axis, reduced rank", (; fam=:axis, diss=:reduced)),
            ("axis, one-sided", (; fam=:axis, diss=:onesided)),
            ("hybrid, reduced rank", (; fam=:hybrid, diss=:reduced)),
            ("hybrid-adv, one-sided", (; fam=:hybridadv, diss=:onesided)),
            ("hybrid-adv, extrapolation degree 1", (; fam=:hybridadv, p=1)),
            ("hybrid-adv, extrapolation degree ≤ q", (; fam=:hybridadv, p=8)),
            ("hybrid-adv where b/a < 1/2", (; fam=:hybridadv, thresh=0.5)),
            ("hybrid-adv at every closure axis", (; fam=:hybridadv, thresh=Inf)),
            ("extrap, degree 1", (; fam=:extrap, p=1)),
            ("Dirichlet core", (; fam=:dirichlet)),
            ("Dirichlet core, ε = 0", (; fam=:dirichlet, ε=0.0)))
    for q in parse_list(Int, "q", (4, 2)), n in parse_list(Int, "n", (24,))
        h = L_SQ / n
        println("\n-- q = $q, n = $n (h = 5/$(round(Int, 5/h))), ε_KO = 1/2 unless " *
                "stated --")
        rEs = parse_list(Float64, "rE", (0.75, 1.1))
        println("| variation | " * join(["r_E = $r" for r in rEs], " | ") * " |")
        println("|---|" * repeat("---|", length(rEs)))
        for (label, kw) in rows
            cells = String[]
            for r_E in rEs
                if kw.fam === :layer
                    pl = make_plane(q, n, LAYER)
                    coef = plane_coefficients_turn(pl, bg)
                    A = spin_operator(pl, coef, rh; fam=:centered, ε=kw.ε)
                    s = spin_spectrum(pl, A, coef, LAYER[3]; order=4)
                else
                    pl = make_plane(q, n, (:excise, r_E))
                    coef = plane_coefficients_turn(pl, bg)
                    A = spin_operator(pl, coef, rh; kw...)
                    s = spin_spectrum(pl, A, coef, r_E; order=turn_order(kw.fam))
                end
                push!(cells, fmt("%s / %.2f", fmt_spin(s), s.cfl))
            end
            say("| %s | %s |", label, join(cells, " | "))
        end
    end
end

# --- model2d, spinnoise and spinfine: long evolutions on the spinning plane ----
#
# X1's noise runs, `plane_noise` unchanged, on the spinning plane: uniform
# noise on every unknown, RK4 at `cfl = 1/2`, to `t_end` (`100 M`), each
# configuration on a thread of its own. `spinnoise` at `n = 64` (`h = 5/128`),
# `spinfine` at `n_fine = 128` (`5/256`).

function spin_noise_table(bg, q, n, rEs, famlops; ε=0.5, t_end=100.0)
    rh = horizon_min_radius(bg)
    jobs = Any[(:layer, nothing, nothing)]
    for r_E in rEs, (fam, lop) in famlops
        push!(jobs, (r_E, fam, lop))
    end
    res = Vector{Any}(undef, length(jobs))
    Threads.@threads :dynamic for k in eachindex(jobs)
        r_E, fam, lop = jobs[k]
        if r_E === :layer
            pl = make_plane(q, n, LAYER)
            coef = plane_coefficients_turn(pl, bg)
            A = spin_operator(pl, coef, rh; fam=:centered, ε=ε)
        else
            pl = make_plane(q, n, (:excise, r_E))
            coef = plane_coefficients_turn(pl, bg)
            A = spin_operator(pl, coef, rh; fam=fam, ε=ε, lop=lop)
        end
        res[k] = plane_noise(pl, A, coef; t_end=t_end)
    end
    return jobs, res
end

for (part, nkey, ndefault) in (("spinnoise", "n", 64), ("spinfine", "n_fine", 128))
    runs("model2d", part) || continue
    local bg = spin_background()
    local t_end = opt(Float64, "t_end", 100.0)
    local n = opt(Int, nkey, ndefault)
    local fams = haskey(OPTS, "fam") ?
           [f for f in SPIN_FAMS if replace(famname(f...), ", " => "", "-" => "") in
                                    split(OPTS["fam"], ',')] : collect(SPIN_FAMS)
    println("\n=== model2d, $part: noise on Kerr-Schild a = $(bg.spin)'s plane, " *
            "RK4 at cfl = 1/2; ‖u‖/‖u₀‖ at 10 M, half-way and the end, and the " *
            "late rate (bold above 1e-4/M) ===")
    for q in parse_list(Int, "q", (4, 2)), ε in parse_list(Float64, "eps", (0.5,))
        rEs = parse_list(Float64, "rE", SPIN_RE)
        jobs, res = spin_noise_table(bg, q, n, rEs, fams; ε=ε, t_end=t_end)
        h = L_SQ / n
        println("\n-- q = $q, n = $n (h = 5/$(round(Int, 5/h))), ε_KO = $ε, to " *
                "$(t_end) M --")
        say("the :damped layer: %s", fmt_noise(res[1], t_end))
        println("| r_E/M | " * join([famname(f...) for f in fams], " | ") * " |")
        println("|---|" * repeat("---|", length(fams)))
        for (i, r_E) in enumerate(rEs)
            cells = [fmt_noise(res[1 + length(fams) * (i - 1) + k], t_end)
                     for k in 1:length(fams)]
            say("| %.2f | %s |", r_E, join(cells, " | "))
        end
    end
end

println("\ndone")
