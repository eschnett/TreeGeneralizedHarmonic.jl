# The frozen-coefficient dispersion analysis of step 8a — how far grid-scale
# content made inside the horizon travels outward before Kreiss–Oliger
# dissipation removes it. Run by hand, once, with its numbers recorded in
# `CODE.md` under "Kreiss–Oliger dissipation".
#
#     julia --project=. test/dispersion.jl
#
# It is a **script and not a test**, in the manner of `test/hole_runs.jl`:
# it prints tables and loads no `Test`, and it takes about a minute. It
# builds no mesh; its only evolutions are the one-dimensional model runs of
# sections (4) and (5). The one claim it rests on that is exact — the
# Nyquist mode's group velocity under the order-`q` advection stencil — is
# asserted in `Rational` by `test/stencils_tests.jl`; the checks below are
# the script's checks on *itself* (the eigenvalues against the closed form,
# the numerical group velocity against the analytic one, the frozen
# coefficients against the metric, the dissipation's symbol against
# `−sin^{2r}(θ/2)`), and a failed one throws.
#
# ## The model, and the sign convention
#
# `CODE.md`'s evolution equations are `∂_t h = β^i ∂_i h + (α/√γ) Π` and
# `∂_t Π = β^i ∂_i Π + α√γ γ^{ij} ∂_i ∂_j h + (lower order)`, both with the
# dissipation `Q = (ε/h) Σ_d (weights)` added (`CODE.md`, "Kreiss–Oliger
# dissipation"). Frozen at a point, with the lower-order terms dropped, the
# principal part of *every* component is the same scalar operator, which
# along one direction `x` is
#
#     (∂_t − b ∂_x)² u = a² ∂_x² u,      b = β^x,   a = α √γ^{xx}.
#
# **The advection enters with a plus sign in the code** — `∂_t h = +β ∂h` —
# so the operator is `(∂_t − b ∂_x)`, and the characteristic speeds are
# `−b ± a`. `PLAN.md` writes `(∂_t + b ∂_x)²` with `b = β^r` and the speeds
# `−b ± a`; those three statements are consistent only with the code's sign,
# which is the one used here. For `KerrSchild(1, 0)` along a radius,
# `β^r = H/(1 + H)`, `α = 1/√(1 + H)`, `γ^{rr} = 1/(1 + H)` with `H = 2M/r`,
# so `b = H/(1 + H)` and `a = 1/(1 + H)`: at `r = 1.5 M`, `b = 4/7 = 0.571`
# and `a = 3/7 = 0.429`, both speeds `−b ± a` negative, as everywhere inside
# `r = 2M`. The script reads `α`, `β` and `γ^{ij}` from `SpacetimeMetrics`
# through the package's own `metric_quantities` and checks them against
# these closed forms.
#
# Semi-discretely, with the package's weights on unit spacing — the order-`q`
# centered first derivative `D₁`, the compact second derivative `D₂`, and the
# order-`q + 2` dissipation `K` of rank `r = q/2 + 1` — a Fourier mode
# `e^{i(kx − ωt)}`, `θ = kh`, sees the symbols
#
#     D₁ → i s(θ)/h,    s(θ) = Σ_j w₁_j sin(jθ),
#     D₂ → −c(θ)/h²,    c(θ) = −Σ_j w₂_j cos(jθ),
#     K  → −κ(θ),       κ(θ) = −Σ_j w_K_j cos(jθ) = sin^{2r}(θ/2),
#
# and the first-order-in-time system `(u, v)` with `∂_t u = b D₁u + v + εKu/h`,
# `∂_t v = b D₁v + a² D₂u + εKv/h` — the code's `(h, Π)` with the factors
# `α/√γ` and `α√γ` absorbed into `v`, which leaves only their product `a²` —
# has the two branches
#
#     ω = (−b s(θ) ∓ a √c(θ))/h − i ε κ(θ)/h.
#
# The script computes them as the eigenvalues of the 2×2 symbol and checks
# the closed form against those. Branch 1 (`−`) is the fast ingoing one,
# `v_g → −b − a`; branch 2 (`+`) is the one the horizon is about, `v_g → −b
# + a`, which is zero at `r = 2M`. The damping `σ = −Im ω = ε κ(θ)/h` is the
# same on both. The group velocity `v_g = Re dω/dk` is formed numerically
# by differentiating in `θ`, and checked against `−b s′(θ) ∓ a (√c)′(θ)`;
# the penetration length is `ℓ(θ) = max(v_g, 0)/σ`, in **cells per
# e-fold**. It is infinite wherever `σ = 0` and `v_g > 0` — at `ε_KO = 0`
# for every outgoing mode, and printed as `∞`.
#
# Two things the model says that the continuum does not. Every centered
# `D₁` annihilates the Nyquist mode (`s(π) = 0`) while `s′(π) < 0`, and
# `(√c)′(π) = 0`, so at `θ = π` *both* branches move at `−b s′(π)` —
# **outward**, at `b` for `q = 2`, `5b/3` for `q = 4`, `11b/5` for `q = 6`.
# And branch 2 turns outgoing well before Nyquist, where `a (√c)′ > b s′`
# (for `q = 2`, `a cos(θ/2) > b cos θ`), which is where `κ` is small: those
# intermediate wavelengths, not Nyquist, set `ℓ_max`.
#
# ## What is printed
#
#   1. The table `PLAN.md` asks for: `ℓ_max = max_θ ℓ` over both branches,
#      its `θ`, and `e^{−8/ℓ_max}` — the attenuation across the default
#      margin `m = 8` *if* the whole margin had the coefficients of that
#      radius — against `q ∈ {2, 4, 6}`, `ε_KO ∈ {0, 1/4, 1/2, 1}` and
#      Kerr-Schild `r = 1.0, 1.2, 1.5, 1.8, 2.0 M`, along a grid axis.
#      Beside it, `ℓ_max` of the fully discrete scheme (RK4 at `cfl = 1/4`
#      on the fixture's `λ_max`), which is what `ε_KO = 0` actually gets.
#      Then (1b) the same along the spin axis and the equator of
#      `Harmonic(1, 9/10)` and `KerrSchild(1, 9/10)`, at depths `d = 2, 4, 8`
#      cells of `h = 5/256` below each horizon, with `b/a` against depth and
#      the sonic point checked against the horizon, and (1c) the e-folds a
#      margin of `m = 1 … 8` cells buys there at `ε_KO = 1/2` — step 8d's
#      question whether `m h < 0.1 M` on the harmonic equator is enough.
#   2. The same along the grid **diagonal**, where the principal part reads
#      the mixed `D₁ ⊗ D₁` and the dissipation acts on three axes at a
#      third of the phase each: the 3D measurement of `test/hole_runs.jl`'s
#      `leakage` section is an L∞ over every direction, and the axis is not
#      obviously the worst one.
#   3. The **path** prediction for that section's runs: `ℓ` varies along
#      the way out, from `ℓ(r_h − d h)` to `∞` at the horizon, so a
#      single-radius `e^{−(d+k)/ℓ_max}` is only a bound of one kind or the
#      other. For each mode `θ` the number of e-folds from depth `d` inside
#      the horizon to `k` cells outside it is `∫ σ/v_g dr/h` with the
#      coefficients of each radius (zero transmission if `v_g ≤ 0` anywhere
#      on the way), and the least-attenuated mode is the leak. Its arrival
#      time `∫ dr/v_g` is printed with it, because a run of finite length
#      only sees the modes that have arrived.
#   4. The **one-dimensional model run**: the code's principal part along a
#      grid axis with the coefficients varying, the fixture's layer, RK4,
#      and the `leakage` section's own ripple — the bridge from one mode at
#      one radius to a packet on the way out, to `2 M` (what the 3D runs
#      reach) and `10 M` (what they cannot afford).
#   5. The same model with `ε_KO` rising **inside the layer** only, and
#      rising **across the margin** from the horizon to `r_1` — `PLAN.md`'s
#      question whether a dissipation profile makes `m = 8` enough.
#
# The frozen-coefficient model ignores refraction (a packet on a stationary
# background conserves `ω`, not `θ`), the lower-order terms, and every
# coupling between components; the `leakage` runs measure what those do.

import Printf
using LinearAlgebra: eigvals
using StaticArrays: SMatrix, SVector
import SpacetimeMetrics as SM
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: _sym4

say(fmt, args...) = println(Printf.format(Printf.Format(fmt), args...))

const ORDERS = (2, 4, 6)
const EPSILONS = (0.0, 0.25, 0.5, 1.0)
const RADII = (1.0, 1.2, 1.5, 1.8, 2.0)
const MARGIN = 8                    # CODE.md's default m
const NTHETA = 4096                 # θ grid on (0, θ_max]

# The mesh the `leakage` runs use (`hole_fixture` on a uniform level-3
# forest, `test/hole_runs.jl`): `h = 5/64`, `r_h = 2 M`, and the step
# `cfl · h / λ_max` with the `λ_max` that fixture records, `1.67095` at the
# box's outer corner (`CODE.md`, "The time step").
const H_FIXTURE = 5 / 64
const R_H = 2.0
const CFL = 0.25
const LAMBDA_MAX = 1.6709529240269554
# The length of those runs, which is what "has arrived" is measured against,
# and the cadence they are sampled at (`test/hole_runs.jl`, `LEAK_CHUNK`).
const T_RUN = 2.0
const CHUNK = 1 / 20

# --- the frozen coefficients, from the metric ------------------------------

const KERR_SCHILD_0 = SM.KerrSchild(1.0, 0.0)

"""
The frozen coefficients at `x = r n̂` on `background` (default
`KerrSchild(1, 0)`): `α`, `β^i`, `γ^{ij}`, read through `background_state`
and `metric_quantities` — the package's own route from a background to the
coefficients the kernel uses — and `b = β^i n̂_i`, `a = α √(γ^{ij} n̂_i n̂_j)`,
the advection and wave speeds of a mode whose wave vector is along `n̂`.

On `KerrSchild(1, 0)` they are checked against the closed forms of the
header, in every direction; on any other background there is nothing
closed-form to check against here, and the sonic point is checked against
the horizon instead (section 1b). A point on the chart's singular set —
harmonic Kerr's equatorial disk — returns `nothing`.
"""
function frozen_coefficients(r, n̂; background=KERR_SCHILD_0)
    bg = background
    x = (r * n̂[1], r * n̂[2], r * n̂[3])
    h, _, _ = background_state(bg, 0.0, x)
    all(isfinite, h) || return nothing
    _, _, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    br = sum(β[i] * n̂[i] for i in 1:3)
    γrr = sum(γu[i, j] * n̂[i] * n̂[j] for i in 1:3, j in 1:3)
    (isfinite(α) && isfinite(br) && isfinite(γrr) && γrr > 0) || return nothing
    if bg isa SM.KerrSchild && iszero(bg.spin) && isone(bg.mass)
        H = 2 / r
        for (got, want, what) in ((α, 1 / sqrt(1 + H), "α"),
                                  (br, H / (1 + H), "β^r"),
                                  (γrr, 1 / (1 + H), "γ^rr"))
            abs(got - want) ≤ 1e-12 ||
                error("KerrSchild(1, 0) at r = $r: $what = $got, closed form $want")
        end
    end
    return (α=α, β=β, γu=γu, sqrtγ=sqrtγ, b=br, a=α * sqrt(γrr))
end

# --- the symbols of the package's weights ----------------------------------

"""
The Fourier symbols of the weights the kernel uses, on unit spacing, as
functions of the phase `θ` per grid step: `D₁(θ) = i s(θ)`, `D₂(θ) = −c(θ)`,
`K(θ) = −κ(θ)`. Summed from the weight vectors themselves, so that nothing
here assumes the textbook form of any of them.
"""
function stencil_symbols(q)
    w1 = derivative_weights(Float64, Val(q), Val(1))
    w2 = derivative_weights(Float64, Val(q), Val(2))
    wk = dissipation_weights(Float64, dissipation_rank(Val(q)))
    r1 = q ÷ 2
    rk = length(wk) ÷ 2
    D1(θ) = sum(w1[j + r1 + 1] * cis(j * θ) for j in (-r1):r1)
    D2(θ) = sum(w2[j + r1 + 1] * cis(j * θ) for j in (-r1):r1)
    # The dissipation's symbol summed from its weights is `−sin^{2r}(θ/2)`
    # to roundoff — checked here, and asserted exactly in `Rational` at
    # Nyquist and to roundoff at four phases by `test/stencils_tests.jl` —
    # but the sum is a cancellation of `O(1)` terms down to `O(θ^{2r})`,
    # which at `θ ≲ 0.01` is roundoff and would make `σ` meaningless where
    # the sonic point's `ℓ` is formed. So the closed form is what is used.
    Kw(θ) = sum(wk[j + rk + 1] * cis(j * θ) for j in (-rk):rk)
    K(θ) = complex(-sin(θ / 2)^(2rk))
    for θ in range(0, π; length=257)
        abs(Kw(θ) - K(θ)) ≤ 1e-14 ||
            error("q = $q: the dissipation symbol at θ = $θ is $(Kw(θ)), " *
                  "not −sin^{2r}(θ/2) = $(K(θ))")
    end
    # The analytic derivatives, for the check on the numerical `v_g`.
    s′(θ) = sum(j * w1[j + r1 + 1] * cos(j * θ) for j in (-r1):r1)
    c(θ) = -real(D2(θ))
    c′(θ) = sum(j * w2[j + r1 + 1] * sin(j * θ) for j in (-r1):r1)
    return (D1=D1, D2=D2, K=K, s′=s′, c=c, c′=c′, rank=rk)
end

"""
The semi-discrete symbol of `(u, v)` for a wave vector `k n̂`, `θ = kh`, at
frozen coefficients `co`: `[[μ, 1], [P, μ]]` with `μ = A + εK`, the
advection `A = Σ_d β^d D₁(θ n̂_d)`, the principal part `P = α²(Σ_d γ^{dd}
D₂(θ n̂_d) + Σ_{d≠e} γ^{de} D₁(θ n̂_d) D₁(θ n̂_e))` — compact on the diagonal,
the tensor product of two first derivatives off it, as `CODE.md`'s kernel
forms them — and the dissipation summed over the three axes. Along an axis
it is the one-dimensional model of the header.
"""
function symbol_parts(S, co, n̂, θ, ε)
    θd = ntuple(d -> θ * n̂[d], 3)
    A = sum(co.β[d] * S.D1(θd[d]) for d in 1:3)
    P = zero(ComplexF64)
    for d in 1:3, e in 1:3
        P += co.γu[d, e] * (d == e ? S.D2(θd[d]) : S.D1(θd[d]) * S.D1(θd[e]))
    end
    P *= co.α^2
    μ = A + ε * sum(S.K(θd[d]) for d in 1:3)
    return μ, P
end

# `ω = iλ` for `e^{λt} = e^{−iωt}`; with `λ = μ ± √P` and `P < 0` real,
# `ω = −Im μ ∓ √(−P) + i Re μ`. Sorted by `Re ω`, which separates the two
# branches everywhere on `(0, θ_max]` because `√(−P) > 0` there.
function branches_closed(S, co, n̂, θ, ε)
    μ, P = symbol_parts(S, co, n̂, θ, ε)
    root = sqrt(-real(P))
    return (complex(-imag(μ) - root, real(μ)), complex(-imag(μ) + root, real(μ)))
end

# The same from the eigenvalues of the 2×2 symbol, checked against the
# closed form every time it is called.
function branches(S, co, n̂, θ, ε)
    μ, P = symbol_parts(S, co, n̂, θ, ε)
    abs(imag(P)) ≤ 1e-12 * (1 + abs(P)) && real(P) < 0 ||
        error("the principal symbol is not real and negative: P = $P")
    Mθ = [μ one(μ); P μ]
    ωs = sort([im * λ for λ in eigvals(Mθ)]; by=real)
    closed = branches_closed(S, co, n̂, θ, ε)
    maximum(abs.(ωs .- collect(closed))) ≤ 1e-10 * (1 + maximum(abs.(closed))) ||
        error("eigenvalues $ωs disagree with the closed form $closed")
    return (ωs[1], ωs[2])
end

# The fully discrete RK4 multiplier per step, and the equivalent `ω`:
# `e^{−iω dt} = G(λ dt)`, so `ω = i log G / dt`, with `ν = dt/h` and `λ = −iω`.
rk4(z) = 1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24
function branches_rk4(S, co, n̂, θ, ε, ν)
    ωs = branches(S, co, n̂, θ, ε)
    return (im * log(rk4(-im * ωs[1] * ν)) / ν, im * log(rk4(-im * ωs[2] * ν)) / ν)
end

"""
`ω(θ)` on both branches over the θ grid, `σ = −Im ω`, the group velocity by
central differences in `θ` (one-sided at the end point), and `ℓ = max(v_g,
0)/σ` in cells. `fd = ν` makes it the fully discrete RK4 scheme instead.
"""
function dispersion(S, co, n̂, ε; θmax=π, fd=nothing)
    θs = [θmax * i / NTHETA for i in 1:NTHETA]
    f(θ) = fd === nothing ? branches(S, co, n̂, θ, ε) :
           branches_rk4(S, co, n̂, θ, ε, fd)
    δ = 1e-5
    return map(θs) do θ
        ω = f(θ)
        dω = θ + δ ≤ θmax ? (f(θ + δ) .- f(θ - δ)) ./ (2δ) :
             (3 .* ω .- 4 .* f(θ - δ) .+ f(θ - 2δ)) ./ (2δ)
        vg = real.(dω)
        σ = .-imag.(ω)
        ℓ = ntuple(k -> vg[k] ≤ 0 ? 0.0 : σ[k] ≤ 0 ? Inf : vg[k] / σ[k], 2)
        (θ=θ, ω=ω, vg=vg, σ=σ, ℓ=ℓ)
    end
end

"""
The largest `ℓ` over both branches and the grid, where it is, and at what
group velocity. `tozero` marks a supremum at the smallest `θ` of the grid
with `ℓ` still growing — the sonic point, where branch 2's continuum speed
`a − b` vanishes and the discrete one is `O(θ^q)` against a dissipation of
`O(θ^{q+2})`.
"""
function ellmax(rows; σmin=0.0)
    best = (ℓ=0.0, θ=NaN, vg=NaN, branch=0, tozero=false)
    for row in rows, k in 1:2
        # `σmin` drops modes whose damping is below what the arithmetic
        # resolves — RK4's own damping at `ε_KO = 0` is `O((ω dt)⁶)`, and
        # at small `θ` that is below roundoff.
        row.σ[k] < σmin && continue
        if row.ℓ[k] > best.ℓ
            best = (ℓ=row.ℓ[k], θ=row.θ, vg=row.vg[k], branch=k,
                    tozero=row.θ < rows[end].θ / 50)
        end
    end
    return best
end

fmtℓ(b) = !isfinite(b.ℓ) ? "∞" :
          b.tozero ? "∞ (θ→0)" : Printf.format(Printf.Format("%.2f"), b.ℓ)
fmtatt(b, m) = !isfinite(b.ℓ) || b.tozero ? "1" :
               Printf.format(Printf.Format("%.2e"), exp(-m / b.ℓ))
fmtθ(b) = !isfinite(b.ℓ) || b.tozero ? "—" :
          Printf.format(Printf.Format("%.3f"), b.θ / π)
fmtvg(b) = !isfinite(b.ℓ) || b.tozero ? "—" :
           Printf.format(Printf.Format("%.3f"), b.vg)

# --- the self-check on the group velocity ----------------------------------

# The numerical group velocity against the analytic one along the axis,
# `−b s′ ∓ a (√c)′`, at every θ of the grid and both branches, for every
# order and radius — the check that `dispersion` differentiates the branch
# it names.
let axis = SVector(1.0, 0.0, 0.0)
    for q in ORDERS, r in RADII
        S = stencil_symbols(q)
        co = frozen_coefficients(r, axis)
        rows = dispersion(S, co, axis, 0.5)
        err = 0.0
        for row in rows
            θ = row.θ
            sq′ = S.c′(θ) / (2 * sqrt(S.c(θ)))
            want = (-co.b * S.s′(θ) - co.a * sq′, -co.b * S.s′(θ) + co.a * sq′)
            err = max(err, maximum(abs.(row.vg .- want)))
        end
        err ≤ 1e-6 || error("q = $q, r = $r: numerical v_g off by $err")
    end
    println("self-check: numerical v_g agrees with −b s′ ∓ a (√c)′ to 1e-6 " *
            "on every row")
end

# --- (1) the table, along a grid axis --------------------------------------

println("\n=== (1) frozen-coefficient penetration length ℓ (cells per " *
        "e-fold), Kerr-Schild M = 1, along a grid axis ===")
println("b = β^r, a = α√γ^rr; ℓ_max over both branches and θ ∈ (0, π]; " *
        "e^{-8/ℓ} is the attenuation over m = 8 cells at that radius; " *
        "ℓ_RK4 is the fully discrete scheme (RK4, cfl = 1/4, λ_max = " *
        "$(round(LAMBDA_MAX; digits=5))); θ_c is where branch 2 turns " *
        "outgoing")
let axis = SVector(1.0, 0.0, 0.0)
    ν = CFL / LAMBDA_MAX
    for q in ORDERS
        S = stencil_symbols(q)
        say("\n-- q = %d: Nyquist group velocity −b s′(π) = %.4f b; " *
            "KO rank r = %d --", q, -S.s′(π), S.rank)
        println("| r/M | b/a | ε_KO | ℓ_max | θ/π | branch | v_g | e^{-8/ℓ} | " *
                "ℓ_RK4 | θ_c/π |")
        for r in RADII
            co = frozen_coefficients(r, axis)
            for ε in EPSILONS
                rows = dispersion(S, co, axis, ε)
                b = ellmax(rows)
                bf = ellmax(dispersion(S, co, axis, ε; fd=ν); σmin=1e-12)
                ic = findfirst(row -> row.vg[2] > 0, rows)
                θc = ic === nothing ? NaN : rows[ic].θ
                # RK4's own damping is `O((ω dt)⁶)`, so at `ε_KO = 0` the fully
                # discrete `ℓ` is finite but set by the roundoff cutoff; and
                # at the sonic point the supremum is the semi-discrete one's.
                fd = b.tozero ? "∞ (θ→0)" :
                     iszero(ε) ? (bf.ℓ > 1e6 ? "> 1e6" : fmtℓ(bf)) : fmtℓ(bf)
                say("| %.1f | %.3f | %.2f | %s | %s | %d | %s | %s | %s | %.3f |",
                    r, co.b / co.a, ε, fmtℓ(b), fmtθ(b), b.branch, fmtvg(b),
                    fmtatt(b, MARGIN), fd, θc / π)
            end
        end
    end
end

# --- (1b) the spinning holes, a few cells inside their horizons ------------
#
# Step 8d keys the layer on the found horizon's offset surface
# `r_1(n̂) = r_h(n̂) − m h`, and `PLAN.md`'s finding 3 puts the proof-of-concept
# case, harmonic Kerr at `a = 9/10`, at `h = 5/256`, where its equator leaves
# `0.1 M` — 5.1 cells — between the singular disk and the horizon. So the
# same table, on the package's two spinning charts along the spin axis and
# along the equator, both grid axes and both directions in which the
# horizon's normal is radial by symmetry: at depths `d = 2, 4, 8` cells of
# `h = 5/256` below the horizon, with the raw `b/a` against depth rather
# than an assumed slope — on the harmonic equator the metric varies over the
# `0.1 M` between the disk and the horizon.
#
# Along a grid axis the symbol is the one-dimensional model's exactly, with
# `b = β^x` and `a = α√γ^{xx}`, so `ℓ` is closed-form in `θ` and is
# evaluated so, on a grid that reaches `θ = 10⁻³` — the peak moves toward
# `θ → 0` as the horizon is approached — and checked against the
# eigenvalue route on the Kerr-Schild table above.

const H_SPIN = 5 / 256
const THETA_AXIS = sort(vcat([π * i / NTHETA for i in 1:NTHETA],
                             exp.(range(log(1e-3), log(π / NTHETA); length=512))))

"""
`ℓ_max` at `ε_KO = 1` along a grid axis, closed-form: `max_θ max(−b s′(θ) +
a (√c)′(θ), 0) / sin^{2r}(θ/2)` — branch 2, which is the larger of the two
wherever `(√c)′ ≥ 0` — with the `θ` it is attained at and its `v_g`.
"""
function ellmax_axis(S, b, a; θs=THETA_AXIS)
    best = (ℓ=0.0, θ=NaN, vg=NaN)
    for θ in θs
        sq′ = S.c′(θ) / (2 * sqrt(S.c(θ)))
        vg = max(-b * S.s′(θ) + a * sq′, -b * S.s′(θ) - a * sq′)
        vg > 0 || continue
        ℓ = vg / -real(S.K(θ))
        ℓ > best.ℓ && (best = (ℓ=ℓ, θ=θ, vg=vg))
    end
    return best
end

# The check on the closed form: along the axis it is the eigenvalue route's
# number on the Kerr-Schild table, on the same uniform grid.
let axis = SVector(1.0, 0.0, 0.0)
    θu = [π * i / NTHETA for i in 1:NTHETA]
    for q in ORDERS, r in RADII[1:(end - 1)]
        S = stencil_symbols(q)
        co = frozen_coefficients(r, axis)
        want = ellmax(dispersion(S, co, axis, 1.0)).ℓ
        got = ellmax_axis(S, co.b, co.a; θs=θu).ℓ
        abs(got - want) ≤ 1e-6 * want ||
            error("q = $q, r = $r: closed-form ℓ_max $got, eigenvalue route $want")
    end
end

# The holes: label, background, direction, horizon radius along it.
const SPIN_CASES = let ax = SVector(0.0, 0.0, 1.0), eq = SVector(1.0, 0.0, 0.0)
    ha = SM.Harmonic(1.0, 0.9)
    ks = SM.KerrSchild(1.0, 0.9)
    (("Harmonic(1, 9/10), axis", ha, ax, horizon_min_radius(ha)),
     ("Harmonic(1, 9/10), equator", ha, eq, horizon_max_radius(ha)),
     ("KerrSchild(1, 9/10), axis", ks, ax, horizon_min_radius(ks)),
     ("KerrSchild(1, 9/10), equator", ks, eq, horizon_max_radius(ks)),
     ("KerrSchild(1, 0), reference", KERR_SCHILD_0, SVector(1.0, 0.0, 0.0),
      2.0))
end

δ_at(bg, n̂, r) = (co = frozen_coefficients(r, n̂; background=bg);
                  co === nothing ? NaN : co.b / co.a - 1)

println("\n=== (1b) the spinning holes, d cells of h = 5/256 inside the " *
        "horizon, along the spin axis and the equator ===")
println("r_s: where b/a = 1 by bisection, against the analytic r_h; g_h: " *
        "d(b/a)/d(depth) at the horizon, per M; ℓ_max at ε_KO = 1 (÷ ε for " *
        "any other), with θ/π; e^{-8/ℓ}: at ε_KO = 1/2")
for (label, bg, n̂, r_h) in SPIN_CASES
    # The sonic point for a mode along `n̂` is where `b = a`, and along a
    # direction in which the horizon's normal is radial that is the horizon
    # itself (`g^{nn} = γ^{nn} − (β^n)²/α² = 0`, the null condition).
    lo, hi = r_h - H_SPIN, r_h + H_SPIN
    for _ in 1:200
        mid = (lo + hi) / 2
        δ_at(bg, n̂, mid) > 0 ? (lo = mid) : (hi = mid)
    end
    r_s = (lo + hi) / 2
    abs(r_s - r_h) ≤ 1e-9 ||
        error("$label: the sonic point is at $r_s, the horizon at $r_h")
    g_h = δ_at(bg, n̂, r_h - 1e-6) / 1e-6
    sing = bg isa SM.KerrSchild && iszero(bg.spin) ? 0.0 : singular_radius(bg)
    say("\n-- %s: r_h = %.6f, r_s − r_h = %.1e, g_h = %.3f/M, singular " *
        "disk to r = %.2f along this direction --", label, r_h, r_s - r_h, g_h,
        n̂[3] == 1 ? 0.0 : sing)
    println("| d | r | b | a | b/a | δ/(d h) | ℓ_max, q = 2 | q = 4 | q = 6 | " *
            "e^{-8/ℓ}, q = 2, 4, 6 |")
    for d in (2, 4, 8)
        r = r_h - d * H_SPIN
        co = frozen_coefficients(r, n̂; background=bg)
        if co === nothing
            say("| %d | %.4f | on the chart's singular disk |", d, r)
            continue
        end
        bs = [ellmax_axis(stencil_symbols(q), co.b, co.a) for q in ORDERS]
        say("| %d | %.4f | %.4f | %.4f | %.4f | %.3f | %.2f (%.3f) | %.2f " *
            "(%.3f) | %.2f (%.3f) | %s |", d, r, co.b, co.a, co.b / co.a,
            (co.b / co.a - 1) / (d * H_SPIN), bs[1].ℓ, bs[1].θ / π, bs[2].ℓ,
            bs[2].θ / π, bs[3].ℓ, bs[3].θ / π,
            join((Printf.format(Printf.Format("%.2e"), exp(-8 * 0.5 / b.ℓ))
                  for b in bs), ", "))
    end
end

# --- (1c) what a margin of m cells buys there ------------------------------
#
# The leakage margin under `CODE.md`'s "The interior" is the path integral
# `n_e = ∫_{r_h − m h}^{r_h} dr / (h ℓ_max(r))` — e-folds of the
# least-attenuated mode at each radius, a lower bound on what any packet
# gets — at `ε_KO = 1/2`, `h = 5/256`, for `m = 1 … 8`, midpoint rule at 32
# points per cell. A margin that reaches the singular disk has no number.

"""
The e-folds `ε ∫ dr/(h ℓ_max,1(r))` across margins of `1 … mmax` cells
below `r_h` along `n̂`, cumulative; `NaN` from the first margin whose path
meets the singular set.
"""
function margin_efolds(S, bg, n̂, r_h; h, ε, mmax=8, per=32)
    out = fill(NaN, mmax)
    acc = 0.0
    for m in 1:mmax, j in 1:per
        r = r_h - (m - 1 + (j - 1 // 2) / per) * h
        co = frozen_coefficients(r, n̂; background=bg)
        co === nothing && return out
        ℓ = ellmax_axis(S, co.b, co.a).ℓ / ε
        acc += 1 / (per * ℓ)
        j == per && (out[m] = acc)
    end
    return out
end

println("\n=== (1c) e-folds across a margin of m cells below the horizon, " *
        "ε_KO = 1/2, h = 5/256 (and the step-5 fixture's 5/64 for the " *
        "reference) ===")
println("n_e = ∫ dr/(h ℓ_max(r)) from r_h − m h to r_h, cumulative in m; " *
        "'—' where the margin reaches the singular disk")
for q in (2, 4)
    S = stencil_symbols(q)
    println("\n-- q = $q --")
    println("| case | h | n_e at m = 1 … 8 | e^{-n_e} at m = 4, 8 |")
    for (label, bg, n̂, r_h) in SPIN_CASES, h in (H_SPIN, H_FIXTURE)
        h == H_FIXTURE && bg !== KERR_SCHILD_0 && continue
        n = margin_efolds(S, bg, n̂, r_h; h=h, ε=0.5)
        say("| %s | %s | %s | %s, %s |", label, h == H_SPIN ? "5/256" : "5/64",
            join((isnan(x) ? "—" : Printf.format(Printf.Format("%.2f"), x)
                  for x in n), " "),
            isnan(n[4]) ? "—" : Printf.format(Printf.Format("%.2e"), exp(-n[4])),
            isnan(n[8]) ? "—" : Printf.format(Printf.Format("%.2e"), exp(-n[8])))
    end
end

# --- (2) the same along the grid diagonal ----------------------------------

println("\n=== (2) the same along the grid diagonal (1,1,1)/√3: θ is the " *
        "radial phase kh, θ_max = √3 π ===")
let diag = SVector(1.0, 1.0, 1.0) / sqrt(3.0)
    for q in ORDERS
        S = stencil_symbols(q)
        println("\n-- q = $q --")
        println("| r/M | ε_KO | ℓ_max | θ/π | branch | v_g | e^{-8/ℓ} |")
        for r in RADII, ε in EPSILONS
            co = frozen_coefficients(r, diag)
            b = ellmax(dispersion(S, co, diag, ε; θmax=sqrt(3.0) * π))
            say("| %.1f | %.2f | %s | %s | %d | %s | %s |", r, ε, fmtℓ(b),
                fmtθ(b), b.branch, fmtvg(b), fmtatt(b, MARGIN))
        end
    end
end

# --- (3) the path prediction for the `leakage` runs ------------------------

"""
For each mode `θ` and branch, the e-folds `∫ σ/v_g dr/h` and the arrival time
`∫ dr/v_g` from `r_h − d h` to `r_h + k h` at spacing `h`, with the
coefficients of each radius (midpoint rule, closed-form branches); a mode
with `v_g ≤ 0` anywhere on the way does not arrive. Returns the
least-attenuated mode overall and among those arriving by `t_run`.
"""
function path_leak(S, ε, d, k; h=H_FIXTURE, t_run=T_RUN, n=128,
                   n̂=SVector(1.0, 0.0, 0.0), θmax=π, nθ=1024)
    r0 = R_H - d * h
    r1 = R_H + k * h
    dr = (r1 - r0) / n
    cos_ = [frozen_coefficients(r0 + (i - 1 // 2) * dr, n̂) for i in 1:n]
    none = (e=Inf, θ=NaN, t=NaN, branch=0)
    best = none
    bestT = none
    δ = 1e-5
    for i in 1:nθ, br in 1:2
        θ = θmax * i / nθ
        θp = min(θ + δ, θmax)
        efolds = 0.0
        t = 0.0
        ok = true
        for co in cos_
            ωp = branches_closed(S, co, n̂, θp, ε)[br]
            ωm = branches_closed(S, co, n̂, θ - δ, ε)[br]
            vg = real(ωp - ωm) / (θp - (θ - δ))
            vg > 0 || (ok = false; break)
            σ = -imag(branches_closed(S, co, n̂, θ, ε)[br])
            efolds += σ / vg * dr / h
            t += dr / vg
        end
        ok || continue
        efolds < best.e && (best = (e=efolds, θ=θ, t=t, branch=br))
        t ≤ t_run && efolds < bestT.e && (bestT = (e=efolds, θ=θ, t=t, branch=br))
    end
    return best, bestT
end

fmtpath(b) = !isfinite(b.e) ? "none arrives" :
             Printf.format(Printf.Format("%.2e (θ/π = %.3f, branch %d, t = %.2f M)"),
                           exp(-b.e), b.θ / π, b.branch, b.t)

println("\n=== (3) the path prediction for the leakage runs: transmission " *
        "from depth d cells inside r_h to k cells outside, h = 5/64, along " *
        "an axis ===")
println("PLAN.md's e^{-(d+k)/ℓ_max} with ℓ_max at the source's radius " *
        "r_h − d h; then the path integral: the least-attenuated mode " *
        "overall, and among those arriving by t = $(T_RUN) M")
for q in (2, 4)
    S = stencil_symbols(q)
    println("\n-- q = $q --")
    println("| ε_KO | d | k | ℓ_max(r_h − d h) | e^{-(d+k)/ℓ_max} | path, " *
            "overall | path, by t = $(T_RUN) M |")
    for ε in EPSILONS[2:end], d in (2, 4, 8), k in (0, 4, 8)
        co = frozen_coefficients(R_H - d * H_FIXTURE, SVector(1.0, 0.0, 0.0))
        bl = ellmax(dispersion(S, co, SVector(1.0, 0.0, 0.0), ε))
        b, bT = path_leak(S, ε, d, k)
        say("| %.2f | %d | %d | %s | %s | %s | %s |", ε, d, k, fmtℓ(bl),
            fmtatt(bl, d + k), fmtpath(b), fmtpath(bT))
    end
end

# --- (4) the one-dimensional model run -------------------------------------
#
# The frozen-coefficient numbers above are for one mode at one radius. What
# the `leakage` section of `test/hole_runs.jl` measures is a *packet* — a
# ripple in a window four cells wide, so broadband whatever its nominal
# wavelength — launched at a depth inside the horizon and carried through
# coefficients that vary by a factor of two on the way out. The bridge is
# the same principal part with its coefficients *varying* along the axis:
# `∂_t h = w (β^x D₁h + (α/√γ) Π + (ε/h) K h) − ρ h` and `∂_t Π = w (β^x D₁Π
# + α√γ γ^{xx} D₂h + (ε/h) K Π) − ρ Π`, on the fixture's axis at `h = 5/64`
# from the center to the Dirichlet face at `x = 5/2`, with the fixture's own
# interior profiles `w` and `ρ` (`interior_profiles`, `ρ_max = 1/dt`), the
# same RK4 step, and the same ripple in `h` with `Π = 0`. It is the code's
# principal part along an axis exactly; what it leaves out is the
# lower-order terms (the source, the damping, `∂β` and `∂(α√γγ)`), the
# coupling between components, and the other directions — which is what the
# 3D runs then add.

"""
The ripple of `test/hole_runs.jl`'s `leakage` section as a function of the
radius: `A (1 − s²)³ cos(2π r/λ)` for `|s| < 1`, `s = (r − r_c)/(W h)`,
centered at depth `d` cells below the horizon, `r_c = r_h − d h`, with a
window half-width of `W = 2` cells.
"""
ripple(r; d, λ, h=H_FIXTURE, A=1e-3, W=2) =
    (r_c = R_H - d * h; s = (r - r_c) / (W * h);
     abs(s) < 1 ? A * (1 - s^2)^3 * cos(2π * (r - r_c) / λ) : 0.0)

"""
One run of the model: the ripple `(d, λ)` to `t_end`, with the step the
driver takes — `cfl · h / λ_max` rounded down to a whole number of steps
per chunk of `1/20 M` — and `ρ_max = 1/dt`, sampled at every chunk
boundary as the 3D runs are. `εprofile(x)` replaces the constant `ε` when
it is given. Returns the largest `|δh|` per shell `k`.
"""
function model_run(q, ε; d, λcells, t_end, h=H_FIXTURE, halfwidth=2.5,
                   r_0=0.4, r_1=1.15, ks=0:8, εprofile=nothing)
    S = derivative_weights(Float64, Val(q), Val(1))
    S2 = derivative_weights(Float64, Val(q), Val(2))
    SK = dissipation_weights(Float64, dissipation_rank(Val(q)))
    G = q ÷ 2 + 1
    n = round(Int, halfwidth / h)          # owned 0 … n−1, x = n is the face
    xs = [i * h for i in 0:(n - 1)]
    per = ceil(Int, CHUNK / (CFL * h / LAMBDA_MAX))
    dt = CHUNK / per
    nsteps = round(Int, t_end / CHUNK) * per
    int = Interior(Float64; center=(0.0, 0.0, 0.0), r_0=r_0, r_1=r_1,
                   ρ_max=1 / dt)
    εs = [εprofile === nothing ? ε : εprofile(x) for x in xs]
    prof = [interior_profiles(int, x) for x in xs]
    co = [x < r_0 ? nothing : frozen_coefficients(x, SVector(1.0, 0.0, 0.0))
          for x in xs]
    coef = map(eachindex(xs)) do i
        c = co[i]
        c === nothing && return (0.0, 0.0, 0.0)
        (c.β[1], c.α / c.sqrtγ, c.α * c.sqrtγ * c.γu[1, 1])
    end
    # Padded storage: `G` zeros below the center (the frozen core holds the
    # unperturbed state, so the perturbation is zero there) and above the
    # face (the Dirichlet data is the exact solution).
    pad(u) = vcat(zeros(G), u, zeros(G + 1))
    function rhs(u, v)
        up, vp = pad(u), pad(v)
        du = similar(u)
        dv = similar(v)
        for i in eachindex(xs)
            w, ρ = prof[i]
            if w == 0
                du[i] = 0.0
                dv[i] = 0.0
                continue
            end
            j = i + G
            D1(f) = sum(S[m + q ÷ 2 + 1] * f[j + m] for m in (-q ÷ 2):(q ÷ 2)) / h
            D2(f) = sum(S2[m + q ÷ 2 + 1] * f[j + m] for m in (-q ÷ 2):(q ÷ 2)) / h^2
            KO(f) = sum(SK[m + G + 1] * f[j + m] for m in (-G):G) * εs[i] / h
            b, cu, cv = coef[i]
            du[i] = w * (b * D1(up) + cu * v[i] + KO(up)) - ρ * u[i]
            dv[i] = w * (b * D1(vp) + cv * D2(up) + KO(vp)) - ρ * v[i]
        end
        return du, dv
    end
    u = [ripple(x; d=d, λ=λcells * h, h=h) for x in xs]
    v = zeros(length(xs))
    # The shells `[r_h + k h, r_h + (k+1) h)` and the running maximum of `|h|`
    # in each, recorded at the times asked for.
    shell = [findall(x -> R_H + k * h ≤ x < R_H + (k + 1) * h, xs) for k in ks]
    amax = zeros(length(ks))
    for step in 1:nsteps
        k1u, k1v = rhs(u, v)
        k2u, k2v = rhs(u .+ dt / 2 .* k1u, v .+ dt / 2 .* k1v)
        k3u, k3v = rhs(u .+ dt / 2 .* k2u, v .+ dt / 2 .* k2v)
        k4u, k4v = rhs(u .+ dt .* k3u, v .+ dt .* k3v)
        u = u .+ dt / 6 .* (k1u .+ 2 .* k2u .+ 2 .* k3u .+ k4u)
        v = v .+ dt / 6 .* (k1v .+ 2 .* k2v .+ 2 .* k3v .+ k4v)
        step % per == 0 || continue
        for (m, idx) in enumerate(shell)
            isempty(idx) || (amax[m] = max(amax[m], maximum(abs, u[idx])))
        end
    end
    return amax
end

"""
The e-folds per cell of a shell profile `A_k`, by least squares of `log A_k`
against `k` over the shells that hold a number above `floor`.
"""
function fit_per_cell(ks, A; floor=1e-30)
    sel = [i for i in eachindex(A) if A[i] > floor]
    length(sel) ≥ 2 || return NaN
    x = [float(ks[i]) for i in sel]
    y = [log(A[i]) for i in sel]
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    slope = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    return -slope
end

println("\n=== (4) the one-dimensional model run: the leakage section's " *
        "ripple (A = 1e-3 in h, Π = 0) along the axis, variable " *
        "coefficients, the fixture's layer, RK4 ===")
println("A_k/A: the largest |δh| at the chunk boundaries in shell k outside " *
        "r_h, by t; the axis meets the Dirichlet face at 2.5 = r_h + 6.4 h, " *
        "which reflects, so the e-folds per cell are fitted over k = 0 … 3")
for q in (2, 4)
    println("\n-- q = $q --")
    println("| ε_KO | λ/h | d | A_0/A by $(T_RUN) M | A_4/A by $(T_RUN) M | " *
            "A_0/A by 10 M | A_4/A by 10 M | e-folds/cell outside, 10 M |")
    for ε in EPSILONS, λc in (2, 4, 8), d in (2, 4, 8)
        A2 = model_run(q, ε; d=d, λcells=λc, t_end=T_RUN, ks=0:5) ./ 1e-3
        A10 = model_run(q, ε; d=d, λcells=λc, t_end=10.0, ks=0:5) ./ 1e-3
        say("| %.2f | %d | %d | %.2e | %.2e | %.2e | %.2e | %.3f |", ε, λc, d,
            A2[1], A2[5], A10[1], A10[5], fit_per_cell(0:3, A10[1:4]))
    end
end

# --- (5) what a dissipation profile inside the horizon buys ----------------
#
# `PLAN.md` asks whether `ε_KO` rising inside the layer is needed to make
# the default margin `m = 8` enough, and step 8c plans the profile as "the
# exterior's value at and outside `r_1`, rising to `ε_in` inside the layer".
# The leakage the margin is there for is made *at or outside* `r_1` — a
# layer's kink at `r_1` is a source at depth `m` — and crosses the margin
# `r_1 ≤ r < r_h`, where that profile is still the exterior's. So the model
# is run with two `C²` profiles from `ε_out = 1/2`: rising inside the layer
# only, `r_1 → (r_0 + r_1)/2`, and rising across the margin, from `ε_out` at
# the horizon to `ε_in` at `r_1` and held there inside. RK4 at `cfl = 1/4`
# is stable to `ε ≈ 6` on the 3D corner mode (`3 ε dt/h < 2.8`), so every
# `ε_in` below is admissible without a smaller step.

smooth(s) = (c = clamp(s, 0.0, 1.0); c^3 * (10 - 15c + 6c^2))
profile_layer(ε_in; ε_out=0.5, r_0=0.4, r_1=1.15) =
    x -> ε_out + (ε_in - ε_out) * smooth((r_1 - x) / ((r_1 - r_0) / 2))
profile_margin(ε_in; ε_out=0.5, r_1=1.15) =
    x -> ε_out + (ε_in - ε_out) * smooth((R_H - x) / (R_H - r_1))

println("\n=== (5) ε_KO rising inside the horizon, the one-dimensional " *
        "model to 10 M from ε_out = 1/2: A_0/A at d = 2, 4, 8 ===")
for q in (2, 4)
    println("\n-- q = $q --")
    println("| profile | ε_in | λ/h | A_0/A, d = 2 | d = 4 | d = 8 |")
    for (name, prof) in (("constant", ε -> nothing), ("layer only", profile_layer),
                         ("margin", profile_margin)),
        ε_in in (name == "constant" ? (0.5,) : (1.0, 2.0, 4.0)), λc in (2, 4)
        A = [model_run(q, 0.5; d=d, λcells=λc, t_end=10.0, ks=0:0,
                       εprofile=prof(ε_in))[1] / 1e-3 for d in (2, 4, 8)]
        say("| %s | %.1f | %d | %.2e | %.2e | %.2e |", name, ε_in, λc, A...)
    end
end

println("\ndone")
