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
    Hl = SVector{4,T}(ntuple(k -> Hwork[idx..., k, b], Val(4)))
    dHl = SMatrix{4,4,T}(ntuple(k -> Hwork[idx..., 4 + k, b], Val(16)))
    return Hl, dHl
end

@inline gauge_at(::Type{T}, Hwork, idx::NTuple{3,Int}, b::Int,
                 ::Val{false}) where {T} =
    (zero(SVector{4,T}), zero(SMatrix{4,4,T}))

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
