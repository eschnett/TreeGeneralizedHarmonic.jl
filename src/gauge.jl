# The prescribed gauge source, and the two questions about a background
# that decide whether there is one at all.
#
# `CODE.md`, "Gauge and constraint damping": `H_a(x)` is sampled from the
# background, `H^a = −Γ^a[g_exact]`, together with `∂_a H_b`, so that the
# background is a *stationary point of the discrete system* rather than
# only of the continuum one. The 20 values live in the `Hsrc` field set
# with `G = 0` — the kernel reads them at the owned point and nowhere else
# — and they are re-sampled after every regrid rather than transferred,
# because they are a function of position and re-evaluating is exact where
# interpolation would not be.
#
# Two consequences shape this file, and both are `CODE.md`'s:
#
#   * **A harmonic background has `H ≡ 0` exactly**, and then there is no
#     `Hsrc` field set at all and the kernel is compiled without the
#     gauge-source terms. That is [`isharmonic`](@ref), and it is a
#     property of the *background*, not a number to be measured at run
#     time: `Val(false)` has to be exactly right, not nearly right.
#   * **A time-dependent gauge source is not supported.** A per-chunk
#     sample cannot represent `H_a(x − vt)`, and evaluating it in the
#     kernel would cost about one right-hand side. So a background that is
#     neither harmonic nor static is *refused*, with a message that names
#     the case that works instead. That is [`isstatic`](@ref) and the
#     check in `GHCase`.

"""
    isharmonic(background) -> Bool

Whether `H^a ≡ −Γ^a` vanishes identically for this background — whether
its coordinates are harmonic, `□x^a = 0`.

This decides a `Val` parameter of the right-hand-side kernel and whether
an `Hsrc` field set exists at all (`CODE.md`, "Gauge and constraint
damping"), so it has to be *exact*. It is therefore a table over the
background types and not a measurement: the gauge source of a harmonic
background evaluates to roundoff and not to zero — `1.4e−16` for
`boost(Harmonic(1, 9/10), 0.3 x̂)`, measured in step 3 — and no threshold
on that number is a fact about a spacetime.

The rows are `CODE.md`'s table under "Initial data and backgrounds":
Minkowski, the AwA gauge wave and Kerr in harmonic coordinates are
harmonic; Kerr-Schild and shifted Minkowski are not. The three affine
wrappers — `translate`, `rotate`, `boost` — take the inner metric's
answer, because `□x^a = 0` is preserved by any affine change of
coordinates and a Lorentz boost is one. **That is what makes the
proof-of-concept case possible at all**: a boosted spinning hole moves,
so a sampled source could not represent it, and it is harmonic, so there
is nothing to sample.

The fallback is `false`: an unknown background is treated as having a
source, which costs a sampling pass and is never wrong about the physics.
A harmonic background this table does not know would be refused if it
also moved — with a message saying so — rather than evolved with a
silently wrong `H`.

`test/gauge_tests.jl` checks every row against a *measurement* of
`gauge_source`, which is what keeps the table honest and what would
notice if one of the unexported wrapper types below were renamed on
`SpacetimeMetrics`' `main`.
"""
isharmonic(::AbstractMetric) = false
isharmonic(::Minkowski) = true
isharmonic(::Harmonic) = true
isharmonic(::KerrSchild) = false

# The wrappers. `SpacetimeMetrics` does not export these types — only the
# functions that build them — so they are named through the module. They
# are listed in `test/prerequisite_tests.jl` for that reason: a rename on
# `main` would otherwise turn "is this harmonic?" into the fallback
# `false` and be found as a mysteriously slow run, or as a refusal, a
# step later.
isharmonic(m::SpacetimeMetrics.TranslatedMetric) = isharmonic(m.metric)
isharmonic(m::SpacetimeMetrics.RotatedMetric) = isharmonic(m.metric)
isharmonic(m::SpacetimeMetrics.BoostedMetric) = isharmonic(m.metric)

# The gauge wave is the one row that is not a rule about a wrapper. The
# AwA testbed `ds² = −H dt² + H dx² + dy² + dz²`, `H = 1 − A sin(2π(x−t)/d)`,
# is harmonic — `√(−g) g^{tt} = −1` and `√(−g) g^{xx} = 1` are constants,
# and the transverse block is `√(−g) = H` differentiated along `y` and `z`
# — but that is a statement about this transformation applied to
# *Minkowski*, not about the transformation. Applied to anything else it
# is a coordinate change with no reason to preserve `□x^a = 0`, so the
# method asks what it was applied to.
isharmonic(m::SpacetimeMetrics.GaugeWaveMetric) = m.metric isa Minkowski

"""
    isstatic(background) -> Bool

Whether `∂_t g_ab` vanishes identically for this background.

Unlike [`isharmonic`](@ref) this one is *measured*, and exactly: a static
background's metric expression does not mention `t`, so the `t` partial
of `dmetric`'s forward-mode pass is an identical zero and not a small
number (measured in step 3 on every row of `CODE.md`'s table —
Kerr-Schild, harmonic Kerr and shifted Minkowski give `0.0` in every
component at every sample point, the gauge wave and the boosted hole give
`0.28` and `0.033`). So the test is `iszero`, with no tolerance to
choose, which is what makes a measurement admissible here where it is not
admissible for harmonicity.

The samples are a fixed, deliberately unsymmetric set of points at two
different times, chosen away from the origin because the hole cases are
singular there. A background that happened to be momentarily stationary
at one of them would still have to be so at all of them.

Its consumer is the refusal in [`GHCase`](@ref): a background that is
neither harmonic nor static has a *time-dependent* gauge source, which
`CODE.md` does not support and this package will not pretend to.
"""
function isstatic(bg::AbstractMetric)
    for t in (0.0, 0.37), x in ((2.3, 1.7, -3.1), (-1.9, 2.7, 1.3),
                                (3.7, -2.1, 0.9))
        _, dg = dmetric(bg, SVector{4,Float64}(t, x[1], x[2], x[3]))
        for a in 1:4, b in 1:4
            iszero(dg[a, b, 1]) || return false
        end
    end
    return true
end

"""
    sample_gauge_source!(Hsrc::FieldSet, background, t)

Fill `Hsrc` with the lowered gauge source `H_b` and its gradient
`∂_a H_b` of `background` at time `t`, as one `fill_by_coordinates!` over
the field set's own backend.

The 20 variables are `H_b` in `1:4` and `∂_a H_b` in `5:20`, the latter
in `SMatrix`' column-major order, so that the kernel can rebuild the
`SMatrix{4,4}` from a contiguous run — see [`gauge_at`](@ref), which is
the only other place that knows this order.

`Hsrc` has `G = 0` (`CODE.md`, "Field sets and layout"): the source is
read at the owned point and never differenced, so there is nothing for a
ghost to hold. This is the most expensive setup phase there is — nested
forward-mode duals through the metric, GHSO2's measurement — and it runs
once per chunk, after every regrid, never per evaluation.

`t` is an argument although every background that reaches here is static:
the refusal in [`GHCase`](@ref) is what makes that true, and passing the
time anyway keeps the call sites honest about which time's data they
asked for.

**`interior` is not optional where there is a hole (fixed in step 5).**
`H^a = −Γ^a[g_exact]` is evaluated at every owned point of the domain,
*including the frozen core*, where the analytic solution is singular — at
the center itself `KerrSchild`'s `k^i = (…, z/r)` divides by zero and the
sample is `NaN`. Nothing in the right-hand side reads it there, because
the kernel branches on the core before it reaches [`gauge_at`](@ref); the
constraint monitors do read it, and `0 · NaN = NaN` is what a masked norm
then reports (`CLAUDE.md`). So this takes the interior and applies
[`core_position`](@ref), exactly as every other path that evaluates the
background on a grid does.
"""
function sample_gauge_source!(Hsrc::FieldSet{T,3}, bg, t;
                              interior=nothing) where {T}
    Hsrc.nvars == 20 || throw(ArgumentError(
        "the gauge-source field set holds H_b (4) and ∂_a H_b (16) = 20 " *
        "variables, but this one has $(Hsrc.nvars)"))
    all(iszero, Hsrc.G) || throw(ArgumentError(
        "the gauge source is read at the owned point only and is never " *
        "differenced, so its field set has G = 0 (CODE.md, \"Field sets and " *
        "layout\"), but this one has G = $(Hsrc.G)"))
    tt = T(t)
    fill_by_coordinates!(
        AllVariables(x -> gauge_tuple(bg, tt, core_position(interior, tt, x))),
        Hsrc)
    return Hsrc
end

"""
    gauge_tuple(background, t, x) -> NTuple{20}

`(H_b..., ∂_a H_b...)` at `(t, x)`, the 20 values `Hsrc` stores.

A free function rather than a closure body so that the packing order is
written once; [`gauge_at`](@ref) is its inverse.
"""
@inline function gauge_tuple(bg, t, x)
    D = typeof(x[1])
    Hl, dHl = gauge_source_grad(bg, SVector{4,D}(D(t), x[1], x[2], x[3]))
    return (Tuple(Hl)..., Tuple(dHl)...)
end

"""
    gauge_at(T, Hwork, idx, b, ::Val{has_source}) -> (Hl, dHl)

The gauge source at one stored point of the `Hsrc` working array, or a
pair of zeros when the background is harmonic and there is no array at
all.

The `Val` is `CODE.md`'s "has gauge source" switch, built once per chunk
in [`GHProblem`](@ref) and resolved when the kernel compiles: on a
harmonic background the whole source term folds away rather than being
multiplied by zero at run time.

`idx` is the **owned** index, which is also the stored index here,
because `Hsrc` has `G = 0`.
"""
@inline function gauge_at(::Type{T}, Hwork, idx::NTuple{3,Int}, b::Int,
                          ::Val{true}) where {T}
    # `@ntuple`, not `ntuple(k -> …)`: no closure for a device to compile as a
    # call (amended 2026-10-05).
    Hl = SVector{4,T}(@ntuple 4 k -> Hwork[idx..., k, b])
    dHl = SMatrix{4,4,T}(@ntuple 16 k -> Hwork[idx..., 4 + k, b])
    return Hl, dHl
end

@inline gauge_at(::Type{T}, Hwork, idx::NTuple{3,Int}, b::Int,
                 ::Val{false}) where {T} =
    (zero(SVector{4,T}), zero(SMatrix{4,4,T}))

# --- the algebraic Kerr-Schild source (added 2026-10-02) ---------------------
#
# `CODE.md`, "Open questions", "Gauge sources that know less about the hole":
# a gauge source that is a closed-form function of the metric and of three
# constants of the run — the hole's mass `M`, its spin 4-vector `S^a` and its
# 4-velocity `u^a` — and makes every boosted, spinning Kerr-Schild hole an
# exact stationary solution wherever it is:
#
#     Γ_a = k(u,u) k_ab u^b / (M + √(M² − (S^a k_ab u^b)²)),    H_a = −Γ_a ,
#
# with `k_ab = g_ab − η_ab`, which is the state's `h` itself. It mentions no
# position, so it moves with the hole, and it needs no `Hsrc` field set: the
# kernel evaluates it from `h` at the point, and `∂_a H_b` by the chain rule
# through the `∂_a g` it already forms.

"""
    AlgebraicGaugeSource{T}

The supertype of every gauge source the kernels evaluate in closed form at the
point, behind `Val(:algebraic)` (added in step B1): [`KerrSchildSource`](@ref),
[`CompositeKerrSchildSource`](@ref), [`DampedHarmonicSource`](@ref) and
[`RolledGaugeSource`](@ref). Each is `isbits` and is the kernel argument where a
sampled source passes its `Hsrc` array; each has a method of
[`algebraic_gauge_source`](@ref)`(src, x, t, h, ∂ₜh, ∂h)`, which is all the
kernels call. A source may read the point's position `x` and the time `t` as
well as the state: the composite's weights and the damped harmonic gauge's
Gaussian read `x`, the tracks and the roll-on read `t`, and `∂_a H_b` includes
every such explicit dependence (`CODE.md`, "The gauge").
"""
abstract type AlgebraicGaugeSource{T} end

"""
    KerrSchildSource(T = Float64; M, spin = (0, 0, 0), velocity = (0, 0, 0))

The constants of [`algebraic_gauge_source`](@ref): the mass `M`, the spin
4-vector `S^a` — the rest frame's `(0, a⃗)`, `a⃗ = J⃗/M`, boosted into the lab
— and the 4-velocity `u^a = γ(1, v⃗)`, from the rest-frame `spin` and the
hole's lab `velocity` (its motion, [`hole_velocity`](@ref); note that
`boost(m, v)` moves the hole at `−v`). `isbits`, so it is the kernel argument
where a sampled source passes its `Hsrc` array. A case takes it as
`GHCase(…; gauge_source)`, which checks that it reproduces the background's
own `H_a = −Γ_a` before anything is built ([`check_gauge_source`](@ref)).
"""
struct KerrSchildSource{T} <: AlgebraicGaugeSource{T}
    M::T
    S::SVector{4,T}
    u::SVector{4,T}
end

function KerrSchildSource(::Type{T}=Float64; M, spin=(0, 0, 0),
                          velocity=(0, 0, 0)) where {T}
    T(M) > 0 || throw(ArgumentError("the hole's mass must be positive, got $M"))
    v = SVector{3,T}(velocity...)
    v2 = sum(abs2, v)
    v2 < 1 || throw(ArgumentError(
        "the hole's velocity must be below the speed of light, got |v| = $(sqrt(v2))"))
    a = SVector{3,T}(spin...)
    γ = 1 / sqrt(1 - v2)
    va = sum(v .* a)
    # The boost of the rest frame's `(0, a⃗)`: `S⁰ = γ v·a`,
    # `S⃗ = a⃗ + (γ − 1)(v·a)/v² v⃗`.
    Ss = iszero(v2) ? a : a + ((γ - 1) * va / v2) * v
    return KerrSchildSource{T}(T(M), SVector{4,T}(γ * va, Ss...),
                               SVector{4,T}(γ, (γ * v)...))
end

"""
    algebraic_gauge_source(src::KerrSchildSource, h, ∂ₜh, ∂h) -> (Hl, dHl)

The gauge source `H_a` of `CODE.md`'s closed form at one point, and its
gradient `dHl[c, a] = ∂_c H_a` — the layout of a sampled source's
[`gauge_at`](@ref) — from the packed `h = g − η` and its packed derivatives
`∂ₜh` and `∂h[i]`. Algebraic in `g`, so the principal part is unchanged.

With `w_a = k_ab u^b`, `K = u^a w_a`, `s = S^a w_a` and
`D = M + √(M² − s²)`, `H_a = −K w_a / D`; the root's argument is clamped at
zero (on a solution it vanishes only at `r = a|cos θ|`, inside the horizon),
and where it is clamped the root's derivative is taken as zero.
"""
@inline function algebraic_gauge_source(src::KerrSchildSource, h::SVector{NC,T},
                                        ∂ₜh::SVector{NC,T},
                                        ∂h::NTuple{3,SVector{NC,T}}) where {T}
    # Converted entry by entry with `T(x)` and scaled by `_scale`, and the two
    # clamps chosen by `_select` rather than branched on (amended 2026-10-05): the
    # kernel evaluates this on SIMD lanes too (`lanes.jl`), where a condition has a
    # value per lane. The chosen values are the branches' own; the root of the
    # clamped side is taken of `max(rad, 0)`, so that it is never a `DomainError`.
    M = T(src.M)
    u = SVector{4,T}(T(src.u[1]), T(src.u[2]), T(src.u[3]), T(src.u[4]))
    S = SVector{4,T}(T(src.S[1]), T(src.S[2]), T(src.S[3]), T(src.S[4]))
    w = _sym4(h) * u
    K = sum(u .* w)
    sw = sum(S .* w)
    rad = M * M - sw * sw
    root = _select(rad > 0, sqrt(max(rad, zero(T))), zero(T))
    D = M + root
    Hl = _scale(-K / D, w)
    dw = (_sym4(∂ₜh) * u, _sym4(∂h[1]) * u, _sym4(∂h[2]) * u, _sym4(∂h[3]) * u)
    # Unrolled by `@ntuple` rather than an `ntuple` do-block: no closure for a
    # device to compile as a call (amended 2026-10-05).
    dHl = SMatrix{4,4,T}(@ntuple 16 n -> _algebraic_dH(u, S, w, dw, K, sw, root, D, Val(n)))
    return Hl, dHl
end

# Entry `n` of `dHl` in column-major order: `c = (n − 1) % 4 + 1` the derivative's
# direction, `a = (n − 1) ÷ 4 + 1` the source's index.
@inline function _algebraic_dH(u, S, w, dw, K, sw, root, D, ::Val{n}) where {n}
    c = (n - 1) % 4 + 1
    a = (n - 1) ÷ 4 + 1
    dK = sum(u .* dw[c])
    dD = _select(root > 0, -sw * sum(S .* dw[c]) / root, zero(root))
    return -(dK * w[a] + K * dw[c][a]) / D + K * w[a] * dD / (D * D)
end

"""
    gauge_source(T, Hwork, idx, b, ::Val{kind}, x, t, h, ∂ₜh, ∂h) -> (Hl, dHl)

The gauge source at one point, whichever kind the problem has: a pair of
zeros (`Val(false)`, harmonic), the sampled `Hsrc` value at the owned index
(`Val(true)`), or the closed form of the [`AlgebraicGaugeSource`](@ref) that
`Hwork` then is (`Val(:algebraic)`, added 2026-10-02), from the point's state
and derivatives, its position `x` and the time `t` (added in step B1: the
composite's weights, the damped harmonic gauge's Gaussian and the roll-on read
them; the sampled and the harmonic kinds and [`KerrSchildSource`](@ref) ignore
them, and their arithmetic is what it was, bit for bit). The kernels call this
and nothing else. The method without `x` and `t` is the one before step B1, for
the sources that need neither.
"""
@inline gauge_source(::Type{T}, Hwork, idx, b, kind::Union{Val{true},Val{false}},
                     x, t, h, ∂ₜh, ∂h) where {T} = gauge_at(T, Hwork, idx, b, kind)
@inline gauge_source(::Type{T}, src::AlgebraicGaugeSource, idx, b, ::Val{:algebraic},
                     x, t, h, ∂ₜh, ∂h) where {T} =
    algebraic_gauge_source(src, x, t, h, ∂ₜh, ∂h)
@inline gauge_source(::Type{T}, Hwork, idx, b, kind::Union{Val{true},Val{false}},
                     h, ∂ₜh, ∂h) where {T} = gauge_at(T, Hwork, idx, b, kind)
@inline gauge_source(::Type{T}, src::KerrSchildSource, idx, b, ::Val{:algebraic},
                     h, ∂ₜh, ∂h) where {T} = algebraic_gauge_source(src, h, ∂ₜh, ∂h)

# The Kerr-Schild source mentions no position and no time.
@inline algebraic_gauge_source(src::KerrSchildSource, x, t, h, ∂ₜh, ∂h) =
    algebraic_gauge_source(src, h, ∂ₜh, ∂h)

"""
    check_gauge_source(background, src, T; center, scale, t = 0)

Refuse an [`AlgebraicGaugeSource`](@ref) that does not make `background`
exact: at twelve fixed points between `2.5` and `6` mass scales from `center`,
at time `t`, the closed form's `H_a` and `∂_a H_b` on the background's own
metric must be its `−Γ_a` and that gradient, as `gauge_source_grad` gives them,
to `10⁻⁹` of their size. A source with the wrong constants — or a background
that is not a Kerr-Schild hole — would otherwise start a run with a gauge
transient that looks like a physical one.

Generic from step B1: the source is evaluated through
`algebraic_gauge_source(src, x, t, h, ∂ₜh, ∂h)`, in `Float64`
([`convert_source`](@ref)), which for a [`KerrSchildSource`](@ref) is the
arithmetic this check always did. A [`RolledGaugeSource`](@ref) is its `from`
source up to `t_0`, so a roll-on that starts at `t_0 ≥ 0` from a source that
makes the background exact passes at `t = 0`; a damped harmonic source on its
own does not on a hole, by design: Kerr-Schild data are not stationary in it
(`CODE.md`, "The gauge").
"""
function check_gauge_source(bg, src::AlgebraicGaugeSource, ::Type{T}; center,
                            scale, t=0) where {T}
    dirs = ((3, 1, -2), (-2, 2, 1), (1, -3, 2), (-1, -1, -3), (2, 3, 1),
            (-3, 1, 1), (1, 1, 3), (3, -2, -1), (-2, -3, 2), (2, -1, 3),
            (-1, 3, -2), (1, 2, -3))
    worst = 0.0
    where_ = nothing
    s64 = convert_source(Float64, src)
    t64 = Float64(t)
    for (k, d) in enumerate(dirs)
        n = SVector{3,Float64}(d...) / sqrt(sum(abs2, d))
        x = SVector{3,Float64}(center...) +
            Float64(scale) * (2.5 + 3.5 * (k - 1) / (length(dirs) - 1)) * n
        p = SVector{4,Float64}(t64, x...)
        Hs, dHs = gauge_source_grad(bg, p)
        g, dg = dmetric(bg, p)
        h = pack_g(g)
        ∂ₜh = pack_sym(SMatrix{4,4,Float64}(dg[a, b, 1] for a in 1:4, b in 1:4))
        ∂h = ntuple(i -> pack_sym(SMatrix{4,4,Float64}(dg[a, b, i + 1]
                                                       for a in 1:4, b in 1:4)), Val(3))
        Ha, dHa = algebraic_gauge_source(effective_source(s64, t64), (x[1], x[2], x[3]),
                                         t64, h, ∂ₜh, ∂h)
        mag = max(maximum(abs, Hs), maximum(abs, dHs), eps())
        err = max(maximum(abs, Ha - Hs), maximum(abs, dHa - dHs)) / mag
        err > worst && (worst = err; where_ = x)
    end
    worst ≤ 1e-9 || throw(ArgumentError(
        "this gauge source does not make the background exact: its H_a or ∂_a H_b " *
        "differs from the background's own −Γ_a by $(worst) of their size at " *
        "x = $(where_). The constants are the hole's mass, rest-frame spin and " *
        "lab velocity (KerrSchildSource(; M, spin, velocity); boost(m, v) moves " *
        "the hole at −v), and the background must be a Kerr-Schild hole. A " *
        "damped harmonic source is not exact on a hole: roll it on from the " *
        "hole's own source (RolledGaugeSource)."))
    return worst
end

"""
    convert_source(S, src) -> the same source with `S` numbers

What [`check_gauge_source`](@ref) evaluates in `Float64`, whatever the case's
type (added in step B1).
"""
convert_source(::Type{S}, src::KerrSchildSource) where {S} =
    KerrSchildSource{S}(S(src.M), SVector{4,S}(src.S), SVector{4,S}(src.u))

"""
    gauge_scale(src) -> the length the source's check is measured in

The mass of a [`KerrSchildSource`](@ref), the lightest hole's of a
[`CompositeKerrSchildSource`](@ref), `from`'s of a [`RolledGaugeSource`](@ref)
and `1` for a [`DampedHarmonicSource`](@ref), whose only length is its Gaussian
(added in step B1, for `GHCase`'s check).
"""
gauge_scale(src::KerrSchildSource) = src.M

# --- step B1: the composite, the damped harmonic gauge, the roll-on ----------
#
# `CODE.md`, "Binary black holes", "The gauge", and "Open questions", "Gauge
# sources that know less about the hole". Three more closed forms in `g`, each
# evaluated in the kernels behind `Val(:algebraic)` as `KerrSchildSource` is,
# and the initial momentum that makes data satisfy `C_a = 0` under any of them:
#
#   * the **tracked Kerr-Schild source** of `K` holes, `H_a = Σ_A w_A H^A_a(g)`,
#     each `H^A` the closed form above with hole `A`'s mass, spin and
#     4-velocity on the piece's line, and `w_A` a partition of unity with
#     `p = 4`;
#   * the **damped harmonic gauge** in SpECTRE's form;
#   * the **roll-on** `f(t) H_from + (1 − f(t)) H_to`,
#     `f = e^{−((t − t_0)/σ_g)⁴}`.
#
# Everything a kernel reaches here is generic in its number type — the CPU
# kernel evaluates it on SIMD lanes (`lanes.jl`), so no branch is taken on a
# value that has one per lane, only on the run's constants and on `t` — and
# closure-free, so that a device compiles no calls (`CLAUDE.md`). `∂_a H_b`
# includes every explicit dependence on `x` and `t`: if `∂_t H` is not the time
# derivative of the `H` in use, the system is the generalized harmonic system for
# no `H` at all, and the difference shows in `C_a`.

# `dHl` with `δ` added to its time row, `dHl[1, a] + δ[a]` (column-major: the
# time row is entries 1, 5, 9, 13).
@inline _add_time_row(dHl::SMatrix{4,4,T}, δ::SVector{4,T}) where {T} =
    SMatrix{4,4,T}(dHl[1] + δ[1], dHl[2], dHl[3], dHl[4],
                   dHl[5] + δ[2], dHl[6], dHl[7], dHl[8],
                   dHl[9] + δ[3], dHl[10], dHl[11], dHl[12],
                   dHl[13] + δ[4], dHl[14], dHl[15], dHl[16])

# The outer product `o[c, a] = d[c] H[a]`: a weight's gradient times a source.
@inline _outer4(d::SVector{4,T}, H::SVector{4,T}) where {T} =
    SMatrix{4,4,T}(d[1] * H[1], d[2] * H[1], d[3] * H[1], d[4] * H[1],
                   d[1] * H[2], d[2] * H[2], d[3] * H[2], d[4] * H[2],
                   d[1] * H[3], d[2] * H[3], d[3] * H[3], d[4] * H[3],
                   d[1] * H[4], d[2] * H[4], d[3] * H[4], d[4] * H[4])

# `dHl[c, a] = r_c[a]` from the four directional rows `t, x, y, z`.
@inline _rows4(r1::SVector{4,T}, r2::SVector{4,T}, r3::SVector{4,T},
               r4::SVector{4,T}) where {T} =
    SMatrix{4,4,T}(r1[1], r2[1], r3[1], r4[1], r1[2], r2[2], r3[2], r4[2],
                   r1[3], r2[3], r3[3], r4[3], r1[4], r2[4], r3[4], r4[4])

# `x^e` for an exponent `e ≥ 0` that is a constant of the run, by repeated
# multiplication (no `^` with a lane base), and its derivative `e x^{e−1}`,
# zero for `e = 0` rather than `0 · x^{−1}`.
@inline function _ipow(x, e::Int)
    p = one(x)
    for _ in 1:e
        p *= x
    end
    return p
end
@inline _ipow_deriv(x, e::Int) = e == 0 ? zero(x) : e * _ipow(x, e - 1)

"""
    TrackedHole(T = Float64; M, spin = (0, 0, 0), center, velocity = (0, 0, 0),
                acceleration = (0, 0, 0), t_p = 0)

One hole of a [`CompositeKerrSchildSource`](@ref) on one piece of its track
(added in step B1): its mass `M` and rest-frame spin `a⃗ = J⃗/M` (the
[`KerrSchildSource`](@ref)'s), and the piece's line about `t_p` — the center
`c(t) = c₀ + v₀ τ + ½ a τ²`, `τ = t − t_p`, its velocity `ċ = v₀ + a τ`, and the
4-velocity and spin 4-vector to first order, `u(t) = u₀ + u̇ τ`,
`S(t) = S₀ + Ṡ τ`, with `u₀ = γ(1, v₀)`, `S₀` the rest-frame spin boosted by
`v₀`, and `u̇`, `Ṡ` their rates under the acceleration `a`.

The source reads `u` and `S` at `t`, and its time derivative carries
`(∂H/∂u) u̇ + (∂H/∂S) Ṡ`, so that `∂_t H` is the derivative of the `H` in use
whatever the line is. A hole with no acceleration is
[`KerrSchildSource`](@ref)'s arithmetic exactly: `u₀` and `S₀` are that
constructor's. Step B2's tracks will build one per piece; the spin is held fixed
in the rest frame (no Thomas precession) **(proposed in step B1)**.
"""
struct TrackedHole{T}
    ks::KerrSchildSource{T}      # M, S₀, u₀
    c0::SVector{3,T}
    v0::SVector{3,T}
    acc::SVector{3,T}
    du::SVector{4,T}             # u̇
    dS::SVector{4,T}             # Ṡ
    t_p::T
end

function TrackedHole(::Type{T}=Float64; M, spin=(0, 0, 0), center,
                     velocity=(0, 0, 0), acceleration=(0, 0, 0), t_p=0) where {T}
    ks = KerrSchildSource(T; M=M, spin=spin, velocity=velocity)
    length(center) == 3 && length(acceleration) == 3 || throw(ArgumentError(
        "a hole's center and acceleration are 3-vectors, got center = $center and " *
        "acceleration = $acceleration"))
    v = SVector{3,T}(velocity...)
    a = SVector{3,T}(acceleration...)
    s = SVector{3,T}(spin...)
    γ = 1 / sqrt(1 - sum(abs2, v))
    # `u = γ (1, v⃗)`: `γ̇ = γ³ (v⃗·a⃗)`.
    γ̇ = γ * γ * γ * sum(v .* a)
    du = SVector{4,T}(γ̇, (γ̇ * v + γ * a)...)
    # `S⁰ = γ v⃗·s⃗`, `S⃗ = s⃗ + f (v⃗·s⃗) v⃗` with `f = (γ − 1)/v² = γ²/(γ + 1)`.
    vs, as = sum(v .* s), sum(a .* s)
    f = γ * γ / (γ + 1)
    ḟ = γ * (γ + 2) / ((γ + 1) * (γ + 1)) * γ̇
    dSv = (ḟ * vs + f * as) * v + (f * vs) * a
    dS = SVector{4,T}(γ̇ * vs + γ * as, dSv...)
    return TrackedHole{T}(ks, SVector{3,T}(center...), v, a, du, dS, T(t_p))
end

# The center and its velocity at `t`: the piece's line.
@inline function _hole_line(hole::TrackedHole, t)
    τ = t - hole.t_p
    c = hole.c0 + τ * hole.v0 + (τ * τ / 2) * hole.acc
    ċ = hole.v0 + τ * hole.acc
    return c, ċ
end

# `H^A` and `∂H^A` of one hole at `t`: `KerrSchildSource`'s arithmetic at the
# piece's `u(t)` and `S(t)`, and the rate `(∂H/∂u) u̇ + (∂H/∂S) Ṡ` in the time
# row. Without an acceleration both are constants, and this is that source's
# own arithmetic, bit for bit (a branch on a constant of the run, not on a lane).
#
# **One call of the closed form, and the branch on its inputs** (found in step
# B1). Written as an early `return algebraic_gauge_source(hole.ks, …)` before a
# second call at `u(t)`, Julia 1.13.2 on aarch64 compiled this, inlined into a
# caller that also took ForwardDiff Jacobians, into code whose `∂_x H` was off by
# `10⁻³` of the source for an accelerating hole — at `-O2` and not at `-O1`, not
# with the call kept out of line, and not on Julia 1.14.0-alpha1: a miscompile.
# `test/gauge_source_tests.jl`'s closed-form testset calls the source in that
# shape and failed with the early return (`CLAUDE.md`, "Things that will bite").
# Do not merge two inlined copies of a source through an early return.
@inline function _tracked_source(hole::TrackedHole, t, h::SVector{NC,T}, ∂ₜh,
                                 ∂h) where {T}
    steady = iszero(hole.acc)
    τ = t - hole.t_p
    u = steady ? hole.ks.u : hole.ks.u + τ * hole.du
    S = steady ? hole.ks.S : hole.ks.S + τ * hole.dS
    ks = KerrSchildSource{eltype(u)}(hole.ks.M, S, u)
    Hl, dHl = algebraic_gauge_source(ks, h, ∂ₜh, ∂h)
    steady && return Hl, dHl
    return Hl, _add_time_row(dHl, _kerr_schild_rate(ks, hole.du, hole.dS, h))
end

# `∂H_a/∂u^b u̇^b + ∂H_a/∂S^b Ṡ^b` of `H_a = −K w_a / D`, `w_a = k_ab u^b`,
# `K = u·w`, `s = S·w`, `D = M + √(M² − s²)`: `δw = k u̇`, `δK = 2 w·u̇`,
# `δs = S·δw + w·Ṡ`, `δD = −s δs/√(M² − s²)` (zero where the root is clamped).
@inline function _kerr_schild_rate(src::KerrSchildSource, du, dS,
                                   h::SVector{NC,T}) where {T}
    M = T(src.M)
    u = SVector{4,T}(T(src.u[1]), T(src.u[2]), T(src.u[3]), T(src.u[4]))
    S = SVector{4,T}(T(src.S[1]), T(src.S[2]), T(src.S[3]), T(src.S[4]))
    δu = SVector{4,T}(T(du[1]), T(du[2]), T(du[3]), T(du[4]))
    δS = SVector{4,T}(T(dS[1]), T(dS[2]), T(dS[3]), T(dS[4]))
    k = _sym4(h)
    w = k * u
    K = sum(u .* w)
    sw = sum(S .* w)
    rad = M * M - sw * sw
    root = _select(rad > 0, sqrt(max(rad, zero(T))), zero(T))
    D = M + root
    δw = k * δu
    δK = 2 * sum(w .* δu)
    δs = sum(S .* δw) + sum(w .* δS)
    δD = _select(root > 0, -sw * δs / root, zero(T))
    c1 = -δK / D + K * δD / (D * D)
    c2 = -K / D
    return SVector{4,T}(c1 * w[1] + c2 * δw[1], c1 * w[2] + c2 * δw[2],
                        c1 * w[3] + c2 * δw[3], c1 * w[4] + c2 * δw[4])
end

# The partition of unity's building block for one hole: `q = r⁴`, `r = |x − c(t)|`,
# and its gradient `(∂_t q, ∂_i q) = 4 r² (−d·ċ, d)`, `d = x − c(t)`.
@inline function _hole_weight(hole::TrackedHole, x, t, ::Type{T}) where {T}
    c, ċ = _hole_line(hole, t)
    d1 = x[1] - T(c[1])
    d2 = x[2] - T(c[2])
    d3 = x[3] - T(c[3])
    ρ = d1 * d1 + d2 * d2 + d3 * d3
    f = 4 * ρ
    q = ρ * ρ
    dq = SVector{4,T}(-f * (d1 * T(ċ[1]) + d2 * T(ċ[2]) + d3 * T(ċ[3])), f * d1,
                      f * d2, f * d3)
    return q, dq
end

"""
    CompositeKerrSchildSource(holes::TrackedHole...)

`CODE.md`'s tracked Kerr-Schild source of `K` holes (added in step B1):
`H_a = Σ_A w_A(x, t) H^A_a(g)`, with `H^A` the closed form of
[`KerrSchildSource`](@ref) at hole `A`'s mass, spin and 4-velocity on its
piece's line ([`TrackedHole`](@ref)), and the partition of unity

    w_A = Π_{B≠A} r_B^p / Σ_C Π_{D≠C} r_D^p,    r_A = |x − c_A(t)|,  p = 4,

`1 − O((r_A/d)^p)` on hole `A`'s horizon at separation `d` **(proposed:
`p = 4`)**. `∂_a H_b` adds to each hole's chain rule the weights' gradient — in
space, and in time through each center's velocity — and each hole's own rate
through its acceleration.

`K` is a type parameter, so the sum is unrolled where the source is compiled.
**With one hole the source is `KerrSchildSource` bit for bit** (the weight is
`1` and is not formed) on a line without acceleration. `isbits`, and the kernel
argument where a sampled source passes its `Hsrc` array.
"""
struct CompositeKerrSchildSource{T,K} <: AlgebraicGaugeSource{T}
    holes::NTuple{K,TrackedHole{T}}
end

CompositeKerrSchildSource(holes::TrackedHole{T}...) where {T} =
    CompositeKerrSchildSource{T,length(holes)}(holes)

# The sum over the holes, unrolled: the weights, their gradients, and the
# weighted sources. Generated so that `K` holes are `K` straight-line blocks
# with no closure; the generator calls `Base` alone, and the body returns the
# one hole's own source for `K = 1`.
@generated function algebraic_gauge_source(src::CompositeKerrSchildSource{S,K}, x,
                                           t, h::SVector{NC,T}, ∂ₜh,
                                           ∂h) where {S,K,T}
    body = Expr(:block, Expr(:meta, :inline))
    if K == 1
        push!(body.args, :(return _tracked_source(src.holes[1], t, h, ∂ₜh, ∂h)))
        return body
    end
    sym(s, A) = Symbol(s, "_", A)
    prodex(list) = foldl((a, b) -> :($a * $b), list)
    sumex(list) = foldl((a, b) -> :($a + $b), list)
    for A in 1:K
        push!(body.args, :(($(sym(:H, A)), $(sym(:dH, A))) =
                               _tracked_source(src.holes[$A], t, h, ∂ₜh, ∂h)))
        push!(body.args, :(($(sym(:q, A)), $(sym(:dq, A))) =
                               _hole_weight(src.holes[$A], x, t, $T)))
    end
    for A in 1:K
        others = [B for B in 1:K if B != A]
        push!(body.args, :($(sym(:P, A)) = $(prodex([sym(:q, B) for B in others]))))
        terms = Any[]
        for B in others
            rest = [sym(:q, D) for D in others if D != B]
            push!(terms, isempty(rest) ? sym(:dq, B) :
                         :(_scale($(prodex(rest)), $(sym(:dq, B)))))
        end
        push!(body.args, :($(sym(:dP, A)) = $(sumex(terms))))
    end
    push!(body.args, :(Σ = $(sumex([sym(:P, A) for A in 1:K]))))
    push!(body.args, :(dΣ = $(sumex([sym(:dP, A) for A in 1:K]))))
    push!(body.args, :(iΣ = one($T) / Σ))
    for A in 1:K
        push!(body.args, :($(sym(:w, A)) = $(sym(:P, A)) * iΣ))
        push!(body.args, :($(sym(:dw, A)) =
                               _scale(iΣ, $(sym(:dP, A)) - _scale($(sym(:w, A)), dΣ))))
    end
    push!(body.args, :(Hl = $(sumex([:(_scale($(sym(:w, A)), $(sym(:H, A))))
                                     for A in 1:K]))))
    push!(body.args, :(dHl = $(sumex([:(_scale($(sym(:w, A)), $(sym(:dH, A))) +
                                        _outer4($(sym(:dw, A)), $(sym(:H, A))))
                                      for A in 1:K]))))
    push!(body.args, :(return Hl, dHl))
    return body
end

"""
    composite_weights(src::CompositeKerrSchildSource, x, t) -> NTuple{K}

The partition of unity `w_A` at `(t, x)`, for tests and diagnostics.
"""
function composite_weights(src::CompositeKerrSchildSource{S,K}, x, t) where {S,K}
    qs = ntuple(A -> first(_hole_weight(src.holes[A], x, t, S)), Val(K))
    Ps = ntuple(A -> prod((qs[B] for B in 1:K if B != A); init=one(S)), Val(K))
    return Ps ./ sum(Ps)
end

"""
    DampedHarmonicSource(T = Float64; σ_r, A_L1 = 1, A_L2 = 0, A_S = 1,
                         e_L1 = 2, e_L2 = 2, e_S = 2, center = (0, 0, 0))

The damped harmonic gauge of Szilágyi, Lindblom and Scheel (2009) in
SpECTRE's form (added in step B1; `CODE.md`, "The gauge"):

    H_a = [μ_L1 L + μ_L2 log(1/α)] n_a − μ_S g_ai β^i / α ,
    L = log(√γ/α),   μ_L1 = A_L1 W L^{e_L1},   μ_L2 = A_L2 W log(1/α)^{e_L2},
    μ_S = A_S W L^{e_S},   W = e^{−(r/σ_r)²},  r = |x − center| ,

with `n_a = (−α, 0, 0, 0)` the future unit normal. **The sign is the package's
`H_a = −Γ_a`**, which is SpECTRE's (its gauge constraint is `C_a = H_a + Γ_a`):
on flat space with `β = 0` and a fixed `γ`, `C_t = 0` reads
`∂_t L = −α (μ_L1 L + μ_L2 log(1/α))`, so `L` and a perturbed lapse are driven
back, and `C_i = 0` reads `∂_t β^i = −α μ_S β^i` (checked in
`test/gauge_source_tests.jl` through [`gauge_consistent_momentum`](@ref)).

The amplitudes are rates in the case's unit of length, `1` being `1/M` for a
hole of mass `1`; the published binary runs take `A_L1 = A_S = 1`, `A_L2 = 0`,
`e = 2`. The exponents are integers `≥ 0`. `σ_r` has no default: it is where the
gauge turns harmonic, and `Inf` is no Gaussian at all. The Gaussian is about a
fixed `center` (the grid's origin in SpECTRE), so `∂_t W = 0`.

It knows one rate and nothing about the hole; Kerr-Schild data are not
stationary in it, so on a hole it is rolled on from the hole's own source
([`RolledGaugeSource`](@ref)). `isbits`.
"""
struct DampedHarmonicSource{T} <: AlgebraicGaugeSource{T}
    A_L1::T
    A_L2::T
    A_S::T
    e_L1::Int
    e_L2::Int
    e_S::Int
    σ_r::T
    center::SVector{3,T}
end

function DampedHarmonicSource(::Type{T}=Float64; σ_r, A_L1=1, A_L2=0, A_S=1,
                              e_L1=2, e_L2=2, e_S=2, center=(0, 0, 0)) where {T}
    all(A -> T(A) ≥ 0, (A_L1, A_L2, A_S)) || throw(ArgumentError(
        "the damped harmonic gauge's amplitudes are damping rates and must be " *
        "non-negative, got A_L1 = $A_L1, A_L2 = $A_L2, A_S = $A_S: a negative one " *
        "drives log(√γ/α), the lapse or the shift away from the values it damps."))
    all(e -> e isa Integer && e ≥ 0, (e_L1, e_L2, e_S)) || throw(ArgumentError(
        "the damped harmonic gauge's exponents are integers ≥ 0, got e_L1 = $e_L1, " *
        "e_L2 = $e_L2, e_S = $e_S."))
    T(σ_r) > 0 || throw(ArgumentError(
        "σ_r is the width of the Gaussian that turns the gauge harmonic far away, " *
        "and must be positive (Inf for no Gaussian), got $σ_r."))
    return DampedHarmonicSource{T}(T(A_L1), T(A_L2), T(A_S), Int(e_L1), Int(e_L2),
                                   Int(e_S), T(σ_r), SVector{3,T}(center...))
end

# `∂α`, `∂β^i` and `∂√γ` along one direction from that direction's packed `∂g`:
# `metric_derivatives_along`'s chain rule without its generators (a closure is a
# call on a device) and without the inverse spatial metric's derivative.
@inline function _adm_rates(gu4::SMatrix{4,4,T}, α, β::SVector{3,T},
                            γu::SMatrix{3,3,T}, sqrtγ, dg::SVector{NC,T}) where {T}
    dgl = _sym4(dg)
    dgu = -(gu4 * dgl) * gu4                     # ∂ g^{ab} = −g^{ac}(∂g_cd)g^{db}
    gutt = gu4[1, 1]
    dα = α * α * α * dgu[1, 1] / 2               # ∂α = ½ α³ ∂g^{tt}
    dβ = SVector{3,T}(-(dgu[1, 2] + β[1] * dgu[1, 1]) / gutt,
                      -(dgu[1, 3] + β[2] * dgu[1, 1]) / gutt,
                      -(dgu[1, 4] + β[3] * dgu[1, 1]) / gutt)
    trγ = γu[1, 1] * dgl[2, 2] + γu[2, 2] * dgl[3, 3] + γu[3, 3] * dgl[4, 4] +
          2 * (γu[1, 2] * dgl[2, 3] + γu[1, 3] * dgl[2, 4] + γu[2, 3] * dgl[3, 4])
    return dα, dβ, sqrtγ * trγ / 2
end

# One row `∂_c H_a` of the damped harmonic gauge, from that direction's `∂g` and
# `∂_c W`. `Φ = W Ψ` is the bracket multiplying `n_a`, `μ_S = W Λ`.
@inline function _damped_harmonic_row(gu4, α, β, γu, sqrtγ, βl, W, dW, Ψ, Ψ_L, Ψ_ℓ,
                                      Λ, Λ_L, B2, Φ, μS, dg::SVector{NC,T}) where {T}
    dα, dβ, dsqrtγ = _adm_rates(gu4, α, β, γu, sqrtγ, dg)
    iα = one(T) / α
    dL = dsqrtγ / sqrtγ - dα * iα                # L = log(√γ/α)
    dℓ = -dα * iα                                # ℓ = log(1/α)
    dΦ = dW * Ψ + W * (Ψ_L * dL + Ψ_ℓ * dℓ)
    dμS = dW * Λ + W * Λ_L * dL
    dB2 = dg[2] * β[1] + dg[3] * β[2] + dg[4] * β[3] +
          βl[1] * dβ[1] + βl[2] * dβ[2] + βl[3] * dβ[3]
    r = μS * dα * iα * iα                         # from ∂(1/α)
    dHt = -dα * Φ - α * dΦ - (dμS * B2 + μS * dB2) * iα + r * B2
    dH1 = -(dμS * βl[1] + μS * dg[2]) * iα + r * βl[1]
    dH2 = -(dμS * βl[2] + μS * dg[3]) * iα + r * βl[2]
    dH3 = -(dμS * βl[3] + μS * dg[4]) * iα + r * βl[3]
    return SVector{4,T}(dHt, dH1, dH2, dH3)
end

@inline function algebraic_gauge_source(src::DampedHarmonicSource, x, t,
                                        h::SVector{NC,T}, ∂ₜh::SVector{NC,T},
                                        ∂h::NTuple{3,SVector{NC,T}}) where {T}
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    # The Gaussian about the fixed center, and its gradient (`∂_t W = 0`).
    σ2 = T(src.σ_r) * T(src.σ_r)
    d1 = x[1] - T(src.center[1])
    d2 = x[2] - T(src.center[2])
    d3 = x[3] - T(src.center[3])
    W = exp(-(d1 * d1 + d2 * d2 + d3 * d3) / σ2)
    cW = -2 * W / σ2
    L = log(sqrtγ / α)
    ℓ = -log(α)
    A1, A2, AS = T(src.A_L1), T(src.A_L2), T(src.A_S)
    e1, e2, eS = src.e_L1, src.e_L2, src.e_S
    # `Φ = μ_L1 L + μ_L2 ℓ = W Ψ` and `μ_S = W Λ`, with their partials in `L`, `ℓ`.
    Ψ = A1 * _ipow(L, e1 + 1) + A2 * _ipow(ℓ, e2 + 1)
    Ψ_L = A1 * (e1 + 1) * _ipow(L, e1)
    Ψ_ℓ = A2 * (e2 + 1) * _ipow(ℓ, e2)
    Λ = AS * _ipow(L, eS)
    Λ_L = AS * _ipow_deriv(L, eS)
    Φ = W * Ψ
    μS = W * Λ
    # The lowered shift `β_i = g_ti = h_ti` and `β_i β^i`.
    βl = SVector{3,T}(h[2], h[3], h[4])
    B2 = βl[1] * β[1] + βl[2] * β[2] + βl[3] * β[3]
    iα = one(T) / α
    Hl = SVector{4,T}(-α * Φ - μS * B2 * iα, -μS * βl[1] * iα, -μS * βl[2] * iα,
                      -μS * βl[3] * iα)
    z = zero(T)
    rt = _damped_harmonic_row(gu4, α, β, γu, sqrtγ, βl, W, z, Ψ, Ψ_L, Ψ_ℓ, Λ, Λ_L,
                              B2, Φ, μS, ∂ₜh)
    rx = _damped_harmonic_row(gu4, α, β, γu, sqrtγ, βl, W, cW * d1, Ψ, Ψ_L, Ψ_ℓ, Λ,
                              Λ_L, B2, Φ, μS, ∂h[1])
    ry = _damped_harmonic_row(gu4, α, β, γu, sqrtγ, βl, W, cW * d2, Ψ, Ψ_L, Ψ_ℓ, Λ,
                              Λ_L, B2, Φ, μS, ∂h[2])
    rz = _damped_harmonic_row(gu4, α, β, γu, sqrtγ, βl, W, cW * d3, Ψ, Ψ_L, Ψ_ℓ, Λ,
                              Λ_L, B2, Φ, μS, ∂h[3])
    return Hl, _rows4(rt, rx, ry, rz)
end

"""
    RolledGaugeSource(from, to; σ_g, t_0 = 0)

The roll-on `H = f(t) H_from + (1 − f(t)) H_to` (added in step B1; `CODE.md`,
"The gauge"), with `f = 1` up to `t_0` and `f = e^{−((t − t_0)/σ_g)⁴}` after,
which is `C³` at `t_0`; `∂_t H` carries `∂_t f (H_from − H_to)`. SpEC rolls the
damped harmonic gauge on over `σ_g = 15–25 M`. `from` is the data's own source
(a [`KerrSchildSource`](@ref) or a [`CompositeKerrSchildSource`](@ref)) and `to`
a [`DampedHarmonicSource`](@ref); any two algebraic sources may be rolled.
`isbits`.

**The kernels never see the two ends** ([`effective_source`](@ref)): up to
`t_0` a launch is given `from` itself, and once `f` underflows to zero
(`t − t_0 > 5.2 σ_g` in `Float64`) `to` itself, so the roll-on is `from` and then
`to` bit for bit, and costs both sources only while it rolls. The choice is made
on the host at each launch, from that launch's `t`, which is the same at every
point; this method itself always forms the blend, with no branch (an early
return between inlined sources is what Julia 1.13.2 miscompiled, `CLAUDE.md`).
"""
struct RolledGaugeSource{T,F,G} <: AlgebraicGaugeSource{T}
    from::F
    to::G
    t_0::T
    σ_g::T
end

function RolledGaugeSource(from::AlgebraicGaugeSource{T}, to::AlgebraicGaugeSource{T};
                           σ_g, t_0=zero(T)) where {T}
    T(σ_g) > 0 || throw(ArgumentError(
        "σ_g is the roll-on's time scale and must be positive, got $σ_g."))
    return RolledGaugeSource{T,typeof(from),typeof(to)}(from, to, T(t_0), T(σ_g))
end

"""
    rollon_weight(src::RolledGaugeSource, t) -> (f, ∂_t f)

The weight of `from` at `t`: `1` up to `t_0`, `e^{−s⁴}` with
`s = (t − t_0)/σ_g` after, and its derivative `−4 s³ f/σ_g`.
"""
@inline function rollon_weight(src::RolledGaugeSource, t)
    s = max(t - src.t_0, zero(t)) / src.σ_g
    s3 = s * s * s
    f = exp(-s3 * s)
    return f, -4 * s3 * f / src.σ_g
end

@inline function algebraic_gauge_source(src::RolledGaugeSource, x, t,
                                        h::SVector{NC,T}, ∂ₜh, ∂h) where {T}
    f, ḟ = rollon_weight(src, t)
    H1, dH1 = algebraic_gauge_source(src.from, x, t, h, ∂ₜh, ∂h)
    H2, dH2 = algebraic_gauge_source(src.to, x, t, h, ∂ₜh, ∂h)
    a = T(f)
    b = T(one(f) - f)
    Hl = _scale(a, H1) + _scale(b, H2)
    dHl = _add_time_row(_scale(a, dH1) + _scale(b, dH2), _scale(T(ḟ), H1 - H2))
    return Hl, dHl
end

"""
    effective_source(src, t) -> the source a launch at `t` evaluates

`src` itself, except (added in step B1):

  * a [`RolledGaugeSource`](@ref) is its `from` up to `t_0`, its `to` once the
    weight has underflowed to zero, and the blend between;
  * a [`CompositeKerrSchildSource`](@ref) of one hole without acceleration is
    that hole's [`KerrSchildSource`](@ref).

[`gauge_work`](@ref) passes this to every kernel launch, the right-hand side's and
the monitors', so a kernel compiled for one of them carries nothing it does not
evaluate, and the two reductions are their sources' kernels **bit for bit**. The
second is not only a saving. The composite's own kernel forms the same
arithmetic, but it is another specialisation, which LLVM contracts into fused
multiply-adds differently: on the moving quadrant hole its right-hand side
differed from the Kerr-Schild kernel's in a third of the entries, by at most
`6·10⁻¹³` against terms of `800` (measured in step B1; `CLAUDE.md`, "Two
spellings of one expression are not bit-identical").
"""
effective_source(src, t) = src
effective_source(src::CompositeKerrSchildSource{T,1}, t) where {T} =
    iszero(src.holes[1].acc) ? src.holes[1].ks : src
function effective_source(src::RolledGaugeSource, t)
    f, _ = rollon_weight(src, t)
    f == 1 && return effective_source(src.from, t)
    iszero(f) && return effective_source(src.to, t)
    return src
end

convert_source(::Type{S}, h::TrackedHole) where {S} =
    TrackedHole{S}(convert_source(S, h.ks), SVector{3,S}(h.c0), SVector{3,S}(h.v0),
                   SVector{3,S}(h.acc), SVector{4,S}(h.du), SVector{4,S}(h.dS), S(h.t_p))
convert_source(::Type{S}, src::CompositeKerrSchildSource{T,K}) where {S,T,K} =
    CompositeKerrSchildSource{S,K}(ntuple(A -> convert_source(S, src.holes[A]), Val(K)))
convert_source(::Type{S}, src::DampedHarmonicSource) where {S} =
    DampedHarmonicSource{S}(S(src.A_L1), S(src.A_L2), S(src.A_S), src.e_L1, src.e_L2,
                            src.e_S, S(src.σ_r), SVector{3,S}(src.center))
function convert_source(::Type{S}, src::RolledGaugeSource) where {S}
    from, to = convert_source(S, src.from), convert_source(S, src.to)
    return RolledGaugeSource{S,typeof(from),typeof(to)}(from, to, S(src.t_0),
                                                        S(src.σ_g))
end

gauge_scale(src::CompositeKerrSchildSource) = minimum(h -> h.ks.M, src.holes)
gauge_scale(src::DampedHarmonicSource{T}) where {T} = one(T)
gauge_scale(src::RolledGaugeSource) = gauge_scale(src.from)

"""
    gauge_consistent_momentum(src, x, t, γ, K, α, β, ∂γ, ∂α, ∂β)
        -> (; Π, ∂ₜα, ∂ₜβ, h, ∂ₜh, ∂h, Hl)

The momentum `Π_ab` that makes Cauchy data satisfy the gauge constraint
`C_a = H_a + Γ_a = 0` at one point (added in step B1; `CODE.md`, "The gauge",
"The initial momentum follows from the gauge"). The data are the spatial metric
`γ_ij`, the extrinsic curvature `K_ij` (the package's sign,
`∂_t γ_ij = −2α K_ij + D_i β_j + D_j β_i`), the lapse `α` and the shift `β^i`,
with their spatial derivatives `∂γ[k] = ∂_k γ_ij`, `∂α[k] = ∂_k α` and
`∂β[k] = ∂_k β^i`; `src` is an [`AlgebraicGaugeSource`](@ref), evaluated at
`(t, x)` on the data's own metric, or the lowered `H_a` itself as an
`SVector{4}`.

`∂_t α` and `∂_t β^i` are free in the data. `Γ_a` is linear in `∂_t g_tb`:
with `X_b = ∂_t g_tb` and `Γ⁰_a` the trace with `X = 0`, `Γ_t = Γ⁰_t + ½ g^{tt}
X_t` and `Γ_i = Γ⁰_i + g^{tt} X_i`, so `C_a = 0` is four decoupled equations
with `g^{tt} = −1/α²` on the diagonal. The result is `∂_t g_ab` complete, and
`Π = (√γ/α)(∂_t g − β^i ∂_i g)`, the first evolution equation's inverse. Every
source here is algebraic in `g` (it reads no derivative of it), so `H_a` is known
before `∂_t g` is.

For one boosted Kerr-Schild hole under its own source the result is the
analytic `Π` to roundoff (`test/gauge_source_tests.jl`), and step B8 builds the
binary's initial momentum with it. A host function, not a kernel's.
"""
function gauge_consistent_momentum(src, x, t, γ::SMatrix{3,3,T}, K::SMatrix{3,3,T},
                                   α::T, β::SVector{3,T},
                                   ∂γ::NTuple{3,SMatrix{3,3,T}}, ∂α::SVector{3,T},
                                   ∂β::NTuple{3,SVector{3,T}}) where {T}
    βl = γ * β
    g4 = SMatrix{4,4,T}((a == 1 && b == 1) ? -α * α + sum(βl .* β) :
                        a == 1 ? βl[b - 1] : b == 1 ? βl[a - 1] : γ[a - 1, b - 1]
                        for a in 1:4, b in 1:4)
    # `∂_k g_ab`: `∂_k β_i = ∂_k γ_ij β^j + γ_ij ∂_k β^j`,
    # `∂_k g_tt = −2α ∂_k α + ∂_k β_i β^i + β_i ∂_k β^i`.
    dg = ntuple(Val(3)) do k
        dβl = ∂γ[k] * β + γ * ∂β[k]
        dtt = -2 * α * ∂α[k] + sum(dβl .* β) + sum(βl .* ∂β[k])
        SMatrix{4,4,T}((a == 1 && b == 1) ? dtt : a == 1 ? dβl[b - 1] :
                       b == 1 ? dβl[a - 1] : ∂γ[k][a - 1, b - 1] for a in 1:4, b in 1:4)
    end
    # `∂_t γ_ij = −2α K_ij + β^k ∂_k γ_ij + γ_kj ∂_i β^k + γ_ik ∂_j β^k`.
    J = SMatrix{3,3,T}(∂β[i][k] for k in 1:3, i in 1:3)        # J[k, i] = ∂_i β^k
    Y = -2 * α * K + (β[1] * ∂γ[1] + β[2] * ∂γ[2] + β[3] * ∂γ[3]) + J' * γ + γ * J
    h = pack_g(g4)
    Hl = _source_value(effective_source(src, t), x, t, h)
    gu = inv(g4)
    dt0 = SMatrix{4,4,T}((a == 1 || b == 1) ? zero(T) : Y[a - 1, b - 1]
                         for a in 1:4, b in 1:4)
    D = (dt0, dg[1], dg[2], dg[3])                               # D[c][a, b] = ∂_c g_ab
    # `Γ⁰_a = g^{bc} ∂_b g_ac − ½ g^{bc} ∂_a g_bc` with `∂_t g_tb = 0`.
    Γ0 = SVector{4,T}(sum(gu[b, c] * D[b][a, c] for b in 1:4, c in 1:4) -
                      sum(gu[b, c] * D[a][b, c] for b in 1:4, c in 1:4) / 2
                      for a in 1:4)
    gutt = gu[1, 1]
    Xt = -2 * (Hl[1] + Γ0[1]) / gutt
    Xi = SVector{3,T}(-(Hl[i + 1] + Γ0[i + 1]) / gutt for i in 1:3)
    ∂ₜg = SMatrix{4,4,T}((a == 1 && b == 1) ? Xt : a == 1 ? Xi[b - 1] :
                         b == 1 ? Xi[a - 1] : Y[a - 1, b - 1] for a in 1:4, b in 1:4)
    ∂ₜh = pack_sym(∂ₜg)
    ∂h = (pack_sym(dg[1]), pack_sym(dg[2]), pack_sym(dg[3]))
    # `∂_t β_i = X_i = ∂_t γ_ij β^j + γ_ij ∂_t β^j`, and
    # `X_t = −2α ∂_t α + X_i β^i + β_i ∂_t β^i`.
    ∂ₜβ = inv(γ) * (Xi - Y * β)
    ∂ₜα = -(Xt - sum(Xi .* β) - sum(βl .* ∂ₜβ)) / (2 * α)
    sqrtγ = sqrt(det(γ))
    Π = (sqrtγ / α) * (∂ₜh - β[1] * ∂h[1] - β[2] * ∂h[2] - β[3] * ∂h[3])
    return (; Π, ∂ₜα, ∂ₜβ, h, ∂ₜh, ∂h, Hl)
end

# The source's `H_a` at a point: every algebraic source here reads `g` and not
# its derivatives, so the value needs none.
function _source_value(src::AlgebraicGaugeSource, x, t, h::SVector{NC,T}) where {T}
    z = zero(SVector{NC,T})
    return first(algebraic_gauge_source(src, x, t, h, z, (z, z, z)))
end
_source_value(Hl::SVector{4}, x, t, h) = Hl

# --- the constraint-damping rate, as a function of position -----------------
#
# `CODE.md`, "Gauge and constraint damping": the Gundlach–Pretorius term
# `Z_ab` carries `γ0(x)`, a **function of position** — a Gaussian of width
# a few `M` around the hole's analytic center, tapered to a small value in
# the wave zone — and a constant `γ2`. GHSO2 measured `γ0 ≈ 1/M` near the
# hole as the requirement for a stable evolution with the horizon in the
# domain (`notes/methods-ghso2.md`: `γ0 = 0` blows up at the surface-gravity
# rate `κ`, `γ0 = 1/M` is stable with constraints flat at truncation), and
# it is *near the hole* that the measurement was made: the same rate out in
# the wave zone damps nothing that is there and costs a term everywhere.
#
# Both profiles are `isbits` and are evaluated per point inside the kernel
# (added in step 5, replacing the constant `γ0` field of `GHCase` that
# steps 3 and 4 carried). A flat-space case takes [`ConstantDamping`](@ref)
# and the arithmetic is what it was.

"""
    ConstantDamping(γ0)

One constraint-damping rate everywhere — what every case without a hole
uses, and what `GHCase` wraps a bare number in.

It ignores the position and the time, so the kernel's `damping_rate` call
folds away to a field load and the flat-space runs of steps 3 and 4 are the
arithmetic they were.
"""
struct ConstantDamping{T}
    γ0::T
end

ConstantDamping(::Type{T}, γ0) where {T} = ConstantDamping{T}(T(γ0))

@inline damping_rate(d::ConstantDamping, t, x) = d.γ0

"""
    GaussianDamping(T = Float64; near, far, width, center)

`CODE.md`'s position-dependent rate: `γ0(x) = far + (near − far)
exp(−r²/2w²)` with `r = |x − c(t)|` the distance to the hole's analytic
center — `near` at the hole, falling to `far` in the wave zone over a width
`w` of a few `M`.

A Gaussian rather than a compactly supported bump because it is `C^∞` and
because nothing depends on it vanishing exactly: `far` is the wave zone's
rate, not zero. `center` is a [`HoleCenter`](@ref) and is evaluated at the
call's `t`, so the profile follows a moving hole without anything being
updated.

`isbits`, and a kernel argument at every right-hand-side evaluation.
"""
struct GaussianDamping{T}
    near::T
    far::T
    width::T
    center::HoleCenter{T}
end

function GaussianDamping(::Type{T}=Float64; near, far, width, center) where {T}
    T(width) > 0 || throw(ArgumentError(
        "the damping profile's width must be positive, got $width: it is the " *
        "Gaussian's width, a few M by CODE.md's recipe, and a zero width is " *
        "a rate that is `far` everywhere except at one point."))
    (T(near) ≥ 0 && T(far) ≥ 0) || throw(ArgumentError(
        "both constraint-damping rates must satisfy γ0 ≥ 0, got near = " *
        "$near and far = $far: the term enters ∂_tΠ with a factor −α√γ, so " *
        "a negative rate drives the violation it is there to damp."))
    c = center isa HoleCenter ? HoleCenter{T}(SVector{3,T}(center.c0),
                                              SVector{3,T}(center.v)) :
        HoleCenter(T, center)
    return GaussianDamping{T}(T(near), T(far), T(width), c)
end

@inline function damping_rate(d::GaussianDamping{T}, t, x) where {T}
    c = center_at(d.center, t)
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    r² = d1 * d1 + d2 * d2 + d3 * d3
    return d.far + (d.near - d.far) * exp(-r² / (2 * d.width * d.width))
end

"""
    damping_bounds(profile) -> (lo, hi)

The smallest and largest value the profile takes anywhere — what `GHCase`
checks `γ0 ≥ 0` on, since a profile has no single number to check.
"""
damping_bounds(d::ConstantDamping) = (d.γ0, d.γ0)
damping_bounds(d::GaussianDamping) = (min(d.near, d.far), max(d.near, d.far))

# --- the Kreiss–Oliger amplitude, as a function of position (step 8c) --------
#
# `CODE.md`, "The interior" (the `ε_KO(r)` profile, added in step 8c) and
# "Kreiss–Oliger dissipation". Step 8a measured that dissipation raised only
# *inside* the layer buys nothing — the grid-scale content the margin is for
# is made at or outside `r_1` and crosses `r_1 ≤ r < r_h`, where a profile
# that rises only inside the layer still has the exterior's value — and that
# dissipation raised *across the margin*, from the horizon inward, cuts what
# reaches the horizon 4–15× at `ε_in = 4`. So the profile below rises from
# the horizon, not from `r_1`. It is the `γ0` pattern: a number is the
# identity ([`dissipation_rate`](@ref) returns it), a profile is `isbits`
# and evaluated per point in the kernel, and the kernel's `Val{DISS}` asks
# [`has_dissipation`](@ref), which is `!iszero` on a number.

"""
    dissipation_rate(ε_KO, t, x) -> ε

The Kreiss–Oliger amplitude at `(t, x)`: `ε_KO` itself when it is a number
— every case but step 8c's calibration, whose arithmetic is therefore what
it was — and the profile's value when it is a [`HorizonDissipation`](@ref).
The kernel multiplies the result by `1/h` once per point.
"""
@inline dissipation_rate(ε::Real, t, x) = ε

"""
    has_dissipation(ε_KO) -> Bool

Whether the Kreiss–Oliger term is present at all — the right-hand-side
kernel's `Val{DISS}`, built once per chunk in [`GHProblem`](@ref).
`!iszero` on a number; for a profile, whether it is nonzero anywhere.
"""
has_dissipation(ε::Real) = !iszero(ε)

"""
    HorizonDissipation(T = Float64; ε_out, ε_in, r_1, r_h, center)

A `C²` Kreiss–Oliger amplitude that is `ε_out` at and outside the
horizon's smallest coordinate radius `r_h` and rises to `ε_in` at the
layer's outer radius `r_1`, held at `ε_in` inside it — a quintic
[`smoothstep`](@ref) of `r = |x − c(t)|` between the two radii, `c(t)` the
hole's analytic trajectory (**added in step 8c**, `CODE.md`, "The
interior").

**It rises from the horizon, not from `r_1` (amended after step 8a).** The
grid-scale content the interior makes is made at `r_1` and has to cross
`r_1 ≤ r < r_h` to get out; in the continuum that region is causally
disconnected from the exterior, and the dissipation is `O(h^{q+1})`
whatever `ε` is, so raising it there costs the exterior nothing. `r_h` is
[`horizon_min_radius`](@ref) of the case's background — boost-contracted,
as that function is — and [`horizon_dissipation`](@ref) builds the profile
from a case.

Exactly `ε_out` for `r ≥ r_h` and exactly `ε_in` for `r ≤ r_1`, both
through a branch rather than the blend, so that a profile whose two values
agree is that number at every point, bit for bit.

**`ε_in ≤ 4`, and `ε_out` too.** The Kreiss–Oliger operator damps the
corner mode of the lattice at `3 ε/h`, and RK4 is stable on the negative
real axis to `2.79`: at `cfl = 1/4` and the fixture's `λ ≈ 1.67` that is
`ε ≲ 6`, and `4` is the margin step 8a recommended.

`isbits`, and a kernel argument at every right-hand-side evaluation.
"""
struct HorizonDissipation{T}
    ε_out::T
    ε_in::T
    r_1::T
    r_h::T
    center::HoleCenter{T}
end

function HorizonDissipation(::Type{T}=Float64; ε_out, ε_in, r_1, r_h,
                            center) where {T}
    eo, ei, r1, rh = T(ε_out), T(ε_in), T(r_1), T(r_h)
    (eo ≥ 0 && ei ≥ 0) || throw(ArgumentError(
        "both Kreiss–Oliger amplitudes must be non-negative, got ε_out = $eo " *
        "and ε_in = $ei: the operator carries the damping sign in its " *
        "weights, so a negative amplitude amplifies the grid-scale modes it " *
        "is there to remove."))
    (eo ≤ 4 && ei ≤ 4) || throw(ArgumentError(
        "the Kreiss–Oliger amplitude is capped at 4, got ε_out = $eo and " *
        "ε_in = $ei: the operator damps the lattice's corner mode at 3ε/h, " *
        "and RK4 is stable on the negative real axis only to 2.79, so " *
        "3ε·dt/h ≤ 2.79 is ε ≲ 6 at cfl = 1/4 on the fixture — and 4 is the " *
        "margin step 8a recommended (CODE.md, \"The interior\")."))
    0 < r1 < rh || throw(ArgumentError(
        "the profile rises from the horizon inward, so it needs " *
        "0 < r_1 < r_h, got r_1 = $r1 and r_h = $rh: r_h is the horizon's " *
        "smallest coordinate radius and r_1 the layer's outer radius, which " *
        "CODE.md's placement bound puts m spacings inside it."))
    c = center isa HoleCenter ? HoleCenter{T}(SVector{3,T}(center.c0),
                                              SVector{3,T}(center.v)) :
        HoleCenter(T, center)
    return HorizonDissipation{T}(eo, ei, r1, rh, c)
end

@inline function dissipation_rate(d::HorizonDissipation{T}, t, x) where {T}
    c = center_at(d.center, t)
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    r = sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    r ≥ d.r_h && return d.ε_out
    r ≤ d.r_1 && return d.ε_in
    s = (d.r_h - r) / (d.r_h - d.r_1)
    return d.ε_out + (d.ε_in - d.ε_out) * smoothstep(s)
end

has_dissipation(d::HorizonDissipation) = !(iszero(d.ε_out) && iszero(d.ε_in))

"""
    dissipation_bounds(ε_KO) -> (lo, hi)

The smallest and largest amplitude the profile takes anywhere, as
[`damping_bounds`](@ref) is for `γ0`.
"""
dissipation_bounds(ε::Real) = (ε, ε)
dissipation_bounds(d::HorizonDissipation) =
    (min(d.ε_out, d.ε_in), max(d.ε_out, d.ε_in))
