# The interior: a pointwise damping layer, and nothing about the mesh.
#
# `CODE.md`, "The interior: a pointwise damping layer". There is no
# excision. Inside the horizon the right-hand side is
#
#     ∂_t u = w(r) · F(u)  −  ρ(r) · (u − u_exact(x, t))          (INTERIOR)
#
# with `r = |x − c(t)|` the distance to the hole's **analytic** center and
# two `C²` profiles: `w` switches the equations off around the
# singularity, `ρ` relaxes the solution toward the analytic one in a layer
# well inside the horizon. Three regions, in the table `CODE.md` draws:
#
#   * `r ≥ r_1`      — evolved, `w = 1`, `ρ = 0`: the equations, untouched;
#   * `r_0 ≤ r < r_1` — the layer, `w` ramping to 0 at its inner edge and
#                       `ρ` rising from 0 at `r_1` to `ρ_max`;
#   * `r < r_0`      — the frozen core, `du = 0` and `F` **not evaluated**.
#
# Everything in this file is a function of position and time. Nothing here
# knows about a block, a level or a ghost width, and a change that makes
# one of these depend on such a thing is wrong (`CLAUDE.md`, "No
# singularity handling, and the interior is pointwise"). The one place the
# mesh appears is [`check_interior_radii`](@ref), which is an *assertion*
# about where the layer was put and not an input to it.
#
# Three things are easy to get wrong here, and each is written out where
# it happens:
#
#   * **`F` is never evaluated where `w = 0`.** The frozen core holds
#     finite but stale data by design — the analytic solution is singular
#     inside it — and `F` of that data may be `NaN`; `0 · NaN = NaN`. The
#     kernel branches on [`is_frozen`](@ref) *before* it touches a
#     stencil.
#   * **The right-hand side never mutates `u`.** `(INTERIOR)` is a term of
#     `du`. Only the `:pasted` variant writes the state, and only from
#     RK4's `step_limiter!` ([`gh_step_limiter!`](@ref)).
#   * **The center is a function of `t`**, never a mutated field:
#     [`HoleCenter`](@ref) is two vectors and `center_at` is a line.

"""
    HoleCenter(c0, v = (0, 0, 0))

The hole's analytic trajectory, `c(t) = c₀ + v t` — the one thing the
interior, the damping profile and (from step 6) the refinement centroid
all measure a distance from.

It is a *function of time* held as two vectors and evaluated by
[`center_at`](@ref), and deliberately not a field somebody updates: a
mutated center would be state that the right-hand side reads, which is
exactly what makes a kernel argument non-`isbits` and a run
thread-dependent (`CLAUDE.md`, "A callback must capture no `Type` and no
host array").

`v` is the coordinate velocity of `boost(background, v)` and is zero for
every static case; G5 is where it is not.
"""
struct HoleCenter{T}
    c0::SVector{3,T}
    v::SVector{3,T}
end

function HoleCenter(::Type{T}, c0, v=(zero(T), zero(T), zero(T))) where {T}
    return HoleCenter{T}(SVector{3,T}(T(c0[1]), T(c0[2]), T(c0[3])),
                         SVector{3,T}(T(v[1]), T(v[2]), T(v[3])))
end

HoleCenter(c0, v=(0, 0, 0)) = HoleCenter(Float64, c0, v)

"""
    center_at(c::HoleCenter, t) -> SVector{3}

`c(t) = c₀ + v t`. One line, inlined into every kernel that asks where the
hole is.
"""
@inline center_at(c::HoleCenter{T}, t) where {T} = c.c0 + oftype(c.c0[1], t) * c.v

"""
    smoothstep(s)

The quintic smoothstep `10s³ − 15s⁴ + 6s⁵`, clamped to `[0, 1]`.

`C²` at both ends — value, first *and* second derivative match the
constants it joins — which is what `CODE.md` asks of the interior's
profiles: a `C¹` ramp would leave a jump in `∂²w` that the compact second
derivative sees as a delta, and the layer would advertise itself in every
convergence study.

Exactly `0` at `s ≤ 0` and exactly `1` at `s ≥ 1`, both as floating-point
values (`10 − 15 + 6 = 1`), so the evolved region outside `r_1` is
multiplied by a hard `1` and not by something a bit below it.

**The result is clamped as well as the argument (measured in step 5.)**
The polynomial has a triple root at `s = 1`, so its evaluation just below
that point rounds either way: at `s = 1 − 2⁻⁵³` the Horner form returns
`1 + 1.3e−15`, and a `w` above one would amplify `F` where `CODE.md` says
the equations are untouched. Clamping the value costs two comparisons and
makes "`w ∈ [0, 1]` everywhere" a property rather than a near-property.
"""
@inline function smoothstep(s::T) where {T}
    u = clamp(s, zero(T), one(T))
    return clamp(u * u * u * (10 + u * (6 * u - 15)), zero(T), one(T))
end

"""
    Interior(T = Float64; center, r_0, r_1, ρ_max = 0, variant = :damped,
             w_ramp = 1//2, ρ_ramp = 1//2, margin = 8, target = nothing)

The damping layer: where it is, how strong it is, and which of `CODE.md`'s
three variants is running.

`center` is a [`HoleCenter`](@ref) (or the `c₀` of one); `r_0` and `r_1`
are the frozen core's radius and the layer's outer radius, both in the
case's coordinates; `ρ_max` is the relaxation rate at the core's edge,
which **the driver overrides once per chunk** with `1/dt` — `CODE.md`
bounds it by RK4's stability on the negative real axis (about `2.8/dt`)
and `ρ_max · dt = 1` relaxes by a factor `e` per step, which is as strong
as it needs to be. A value carried in a case is therefore a placeholder,
and [`with_ρ_max`](@ref) is how the driver replaces it.

`w_ramp` and `ρ_ramp` are the fractions of the layer's width over which
the two profiles turn over, measured from the *inner* edge for `w` and
from the *outer* edge for `ρ`. At the default `1//2` each, the outer half
of the layer is fully evolved (`w = 1`) with the relaxation fading to zero
at `r_1`, and the inner half relaxes at full strength (`ρ = ρ_max`) with
the evolution fading to zero at `r_0` — complementary, so that no point
is both frozen and undamped.

`variant` is `CODE.md`'s switch:

| variant | what the right-hand side is | `F` skipped where |
|---|---|---|
| `:damped` | `(INTERIOR)` in full — the default | `r < r_0` |
| `:frozen` | the pure mask, `ρ ≡ 0`, `du = w F(u)` | `r < r_0` |
| `:pasted` | `du = 0` inside `r_1`, the state overwritten by RK4's `step_limiter!` | `r < r_1` |

`margin` is the `m` of `r_1 ≤ r_h,min − m·h` and belongs here because it
is a statement about where this layer was put; no kernel reads it, and
[`check_interior_radii`](@ref) is what does.

`target` is what the layer relaxes *to* **(added in step 8c)**: `nothing`,
the default, meaning the case's own background — the analytic solution,
step 5's design — or another `SpacetimeMetrics` metric, `isbits`, that the
kernel evaluates in its place wherever `(INTERIOR)` reads `u_exact`: the
layer branch of the right-hand side and the `:pasted` overwrite. Nothing
else reads it. The initial data, the Dirichlet hook, the gauge source and
the error reference stay on the case's *true* background, so the record's
`residual` measures the layer's distance from the truth and not from its
target. It exists so that step 8c can calibrate the layer against a target
that is *wrong* — a different mass, a displaced center, a curvature error —
which is what a target fitted to evolved data will be (`PLAN.md`, steps
8c and 8e); [`layer_target`](@ref) is how the kernel resolves it.

The struct is `isbits` — it is a kernel argument at every right-hand-side
evaluation — and `variant` lives in a `Val` for that reason, a `Symbol`
field not being `isbits`.
"""
struct Interior{T,V,X}
    center::HoleCenter{T}
    r_0::T
    r_1::T
    ρ_max::T
    w_ramp::T
    ρ_ramp::T
    margin::Int
    valvariant::Val{V}
    target::X                    # the layer's target metric, or `nothing`
end

const INTERIOR_VARIANTS = (:damped, :pasted, :frozen)

function Interior(::Type{T}=Float64; center, r_0, r_1, ρ_max=zero(T),
                  variant::Symbol=:damped, w_ramp=T(1 // 2), ρ_ramp=T(1 // 2),
                  margin::Integer=8, target=nothing) where {T}
    variant in INTERIOR_VARIANTS || throw(ArgumentError(
        "the interior variant must be one of $(INTERIOR_VARIANTS), got " *
        ":$variant. CODE.md names exactly three and measures all three on " *
        "the static hole: :damped is (INTERIOR) itself and the default, " *
        ":frozen is the pure mask ρ = 0 that pile-up at the freezing radius " *
        "is expected to defeat, and :pasted is the hard overwrite through " *
        "RK4's step_limiter!."))
    c = center isa HoleCenter ? HoleCenter{T}(SVector{3,T}(center.c0),
                                              SVector{3,T}(center.v)) :
        HoleCenter(T, center)
    r0, r1 = T(r_0), T(r_1)
    r0 > 0 || throw(ArgumentError(
        "the frozen core's radius must be positive, got r_0 = $r0: r_0 is " *
        "chosen where the analytic solution is still moderate (|h| ≲ 10), " *
        "and inside it the solution is singular and is never evaluated."))
    r1 > r0 || throw(ArgumentError(
        "the layer needs r_1 > r_0, got r_0 = $r0 and r_1 = $r1: r_1 is the " *
        "layer's outer radius and r_0 the frozen core's, and CODE.md asks " *
        "for at least 2(G+1) spacings between them so that no stencil of an " *
        "evolved point reaches the core (check_interior_radii is what " *
        "measures that against a mesh)."))
    wr, ρr = T(w_ramp), T(ρ_ramp)
    (0 < wr ≤ 1 && 0 < ρr ≤ 1) || throw(ArgumentError(
        "the ramp fractions are fractions of the layer's width and must lie " *
        "in (0, 1], but w_ramp = $wr and ρ_ramp = $ρr. A ramp of zero width " *
        "is a step, which is the :pasted variant and is selected by name."))
    T(ρ_max) ≥ 0 || throw(ArgumentError(
        "the relaxation rate must satisfy ρ_max ≥ 0, got $ρ_max: a negative " *
        "rate drives the solution away from the analytic one it is there to " *
        "hold it at."))
    margin ≥ 1 || throw(ArgumentError(
        "the margin m is a number of grid points and must be at least 1, " *
        "got $margin; CODE.md's default is 8 and its floor is G + 1."))
    check_layer_target(target)
    return Interior{T,variant,typeof(target)}(c, r0, r1, T(ρ_max), wr, ρr,
                                              Int(margin), Val(variant), target)
end

# The one refusal a target needs (added in step 8c): it is evaluated inside
# the kernel exactly as the background is — `dmetric` at a point — so it has
# to be a `SpacetimeMetrics` metric, and it travels as a kernel argument, so
# it has to be `isbits`.
check_layer_target(::Nothing) = nothing

function check_layer_target(target)
    target isa AbstractMetric || throw(ArgumentError(
        "the layer's target must be a SpacetimeMetrics metric or `nothing` " *
        "(the case's own background), got a $(typeof(target)): the kernel " *
        "evaluates it where (INTERIOR) reads u_exact, through `dmetric` at " *
        "the point, exactly as it evaluates the background."))
    isbits(target) || throw(ArgumentError(
        "the layer's target $(typeof(target)) is not isbits, so it cannot be " *
        "a kernel argument: it is evaluated at every layer point of every " *
        "right-hand-side evaluation, on whatever backend the state lives on " *
        "(CLAUDE.md, \"A callback must capture no Type and no host array\")."))
    return nothing
end

"""
    layer_target(interior, background) -> metric

The metric the layer relaxes toward: the interior's `target` where it has
one, and the case's `background` where it is `nothing` — resolved from the
type, so that a case without a target compiles to exactly the kernel it
was before step 8c.
"""
@inline layer_target(::Interior{T,V,Nothing}, bg) where {T,V} = bg
@inline layer_target(int::Interior, bg) = int.target

"""
    with_ρ_max(int::Interior, ρ_max) -> Interior

The same layer with a different relaxation rate — what the driver builds
once per chunk from that chunk's `dt`, since `ρ_max · dt = 1` is a
statement about the integrator and not about the hole (`CODE.md`, "The
profiles and their parameters").

It is a reconstruction rather than a mutation because the interior is a
kernel argument: an `isbits` value that a kernel closed over must not
change underneath it.
"""
with_ρ_max(int::Interior{T,V,X}, ρ_max) where {T,V,X} =
    Interior{T,V,X}(int.center, int.r_0, int.r_1, T(ρ_max), int.w_ramp,
                    int.ρ_ramp, int.margin, int.valvariant, int.target)

"""
    interior_variant(int) -> Symbol

Which of `CODE.md`'s three variants this interior runs, or `:none` for
`nothing` — the value the right-hand-side kernel's fifth `Val` carries.
"""
interior_variant(::Nothing) = :none
interior_variant(::Interior{T,V}) where {T,V} = V

# The distance from the hole's analytic center at time `t`. Written out
# rather than through `norm`, so that it needs nothing of `LinearAlgebra`
# inside a kernel and is the same three multiplies on every backend.
#
# `nothing` — no hole — answers `−1`, which is outside every radius this
# package compares against and is what makes the analysis kernels read the
# same on a case with a hole and one without, without a branch of their own.
@inline interior_radius(::Nothing, t, x) = -one(typeof(x[1]))

# Whether the point is inside the damping layer proper: the region the
# interior residual is measured over, and the only region where `u_exact`
# is evaluated during an evaluation.
@inline in_layer(::Nothing, r) = false
@inline in_layer(int::Interior, r) = (int.r_0 ≤ r) & (r < int.r_1)

@inline function interior_radius(int::Interior{T}, t, x) where {T}
    c = center_at(int.center, t)
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    return sqrt(d1 * d1 + d2 * d2 + d3 * d3)
end

"""
    is_frozen(int::Interior, r) -> Bool

Whether the point at distance `r` from the center has `du = 0` and — this
is the point of the predicate — whether `F` must **not** be evaluated
there.

`r < r_0` for `:damped` and `:frozen`, `r < r_1` for `:pasted`, which
freezes the whole interior and lets the limiter write it. `CLAUDE.md`:
the frozen core holds finite, stale data by design, the analytic solution
is singular inside it, `F` of that data may be `NaN`, and `0 · NaN = NaN`
— so the kernel asks this before it touches a stencil, not after.
"""
@inline is_frozen(int::Interior{T,:damped}, r) where {T} = r < int.r_0
@inline is_frozen(int::Interior{T,:frozen}, r) where {T} = r < int.r_0
@inline is_frozen(int::Interior{T,:pasted}, r) where {T} = r < int.r_1

"""
    interior_profiles(int::Interior, r) -> (w, ρ)

The two `C²` profiles of `(INTERIOR)` at distance `r` from the center:
`w` multiplying the right-hand side and `ρ` the relaxation toward the
analytic solution.

Exactly `(1, 0)` for `r ≥ r_1` — the evolved region is untouched, not
nearly untouched — and `(0, ρ_max)` for `r ≤ r_0`. In between they turn
over on the fractions `w_ramp` (from the inner edge) and `ρ_ramp` (from
the outer one) of the layer's width.

`:frozen` is the same thing with `ρ ≡ 0`; the kernel reads that off the
variant rather than from a zero `ρ_max`, so that `:frozen` and a badly
configured `:damped` are not the same run.
"""
@inline function interior_profiles(int::Interior{T}, r) where {T}
    s = (r - int.r_0) / (int.r_1 - int.r_0)
    w = smoothstep(s / int.w_ramp)
    ρ = int.ρ_max * smoothstep((1 - s) / int.ρ_ramp)
    return w, ρ
end

@inline function interior_profiles(int::Interior{T,:frozen}, r) where {T}
    s = (r - int.r_0) / (int.r_1 - int.r_0)
    return smoothstep(s / int.w_ramp), zero(T)
end

"""
    core_position(int, t, x) -> x

Where the analytic solution is to be **evaluated** for the point `x`: `x`
itself outside the frozen core, and the core's own radial projection
`c(t) + r_0 (x − c(t))/|x − c(t)|` inside it.

`CODE.md`, "The frozen core": the core holds finite data and is never
read, and the initial-data callback fills it with the analytic solution
on the sphere `r_0` along the ray — continuous at `r_0`, finite
everywhere, and a legitimate metric at every point (it is the analytic
metric *somewhere*), which is what keeps `det g` away from zero where a
stencil of a layer point reaches in.

At the center itself the ray is undefined and the `+ẑ` direction is
taken: for both hole backgrounds the axis is the regular direction — the
harmonic chart is singular on the disk `z = 0, x² + y² ≤ a²` and nowhere
on the axis — so a tie broken any other way would be the worse one.

`nothing` as the interior is no hole at all and the position is returned
unchanged, which is what makes this safe to call from the one initial-data
path every case shares.
"""
@inline core_position(::Nothing, t, x) = x

@inline function core_position(int::Interior{T}, t, x) where {T}
    c = center_at(int.center, t)
    d = SVector{3}(x[1] - c[1], x[2] - c[2], x[3] - c[3])
    r = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    r ≥ int.r_0 && return (x[1], x[2], x[3])
    ẑ = SVector{3}(zero(r), zero(r), one(r))
    n = iszero(r) ? ẑ : d / r
    return (c[1] + int.r_0 * n[1], c[2] + int.r_0 * n[2], c[3] + int.r_0 * n[3])
end

"""
    InteriorMask(center, r_1)

The mask every norm, monitor and (from step 6) refinement indicator takes
when there is a hole: [`is_evolved`](@ref) is `r ≥ r_1`.

`CODE.md`, "The frozen core": the modified region is not a numerical
solution and must not be reported as one or refined for its own sake. It
is a *snapshot* — the center already evaluated at the call's `t` — because
`is_evolved(mask, x)` takes a position and no time, and a mask that
carried a trajectory would have to be told which time it meant at every
point. [`interior_mask`](@ref) is what builds it.
"""
struct InteriorMask{T}
    center::SVector{3,T}
    r_1::T
end

@inline function is_evolved(m::InteriorMask, x)
    d1 = x[1] - m.center[1]
    d2 = x[2] - m.center[2]
    d3 = x[3] - m.center[3]
    return d1 * d1 + d2 * d2 + d3 * d3 ≥ m.r_1 * m.r_1
end

"""
    ShellMask(center, lo, hi)

A mask that counts only the spherical shell `lo ≤ r < hi` — how a norm is
taken over *part* of the evolved region rather than over all of it.

`CODE.md`'s G4 asks for the three interior variants' "constraint norms in
the `G` points outside `r_1`", which is a shell one stencil wide just
outside the layer: the points whose stencils reach into it, and therefore
the only ones the variants can differ at until the difference propagates.
The gauge drift is read over a shell at the horizon in the same way.

It is a mask and not a norm of its own, so every masked norm in the package
takes it unchanged; a snapshot of the center, for the reason
[`InteriorMask`](@ref) is.
"""
struct ShellMask{T}
    center::SVector{3,T}
    lo::T
    hi::T
end

@inline function is_evolved(m::ShellMask, x)
    d1 = x[1] - m.center[1]
    d2 = x[2] - m.center[2]
    d3 = x[3] - m.center[3]
    r² = d1 * d1 + d2 * d2 + d3 * d3
    return (m.lo * m.lo ≤ r²) & (r² < m.hi * m.hi)
end

"""
    interior_mask(interior, t) -> AllPoints or InteriorMask

The mask of `interior` at time `t`: [`AllPoints`](@ref) where there is no
hole, and an [`InteriorMask`](@ref) at `c(t)` where there is.

Built at each call with that call's `t`, like every other hook in this
package (`CLAUDE.md`, "Hooks depend on time"). A mask built one chunk ago
would exclude the wrong ball once the hole moves.
"""
interior_mask(::Nothing, t) = AllPoints()
interior_mask(int::Interior{T}, t) where {T} =
    InteriorMask{T}(center_at(int.center, t), int.r_1)

# --- where the horizon is, analytically -------------------------------------
#
# `CODE.md`: `r_h,min` is the smallest **coordinate** distance from the
# center to the horizon over all directions, and it is analytic for every
# background here. The Kerr horizon is oblate in these charts, so the
# minimum is on the spin axis and the maximum on the equator; a boost
# contracts along `v`, and taking `√(1 − v²)` on the minimum whatever the
# boost's direction is conservative — which is the side an assertion
# wants to be wrong on.

"""
    horizon_min_radius(background) -> T
    horizon_max_radius(background) -> T

The smallest and largest **coordinate** radius of the event horizon,
measured from the hole's center, for the backgrounds that have one.

In Kerr-Schild Cartesian coordinates the spheroidal radius of the horizon
is `r₊ = M + √(M² − a²)` and the coordinate radius is `r₊` on the axis and
`√(r₊² + a²)` on the equator; the fully harmonic chart is the same surface
with `R = r − M`, so the axis value is `√(M² − a²)` and the equatorial one
is exactly `M`. `translate` and `rotate` move and turn the surface without
changing either radius; `boost` contracts along `v` by `√(1 − v²)`, which
lowers the minimum and leaves the maximum (the transverse extent)
unchanged.

This is what [`check_interior_radii`](@ref) puts the layer inside of, and
what the horizon finder of step 7 will be checked against. For `a = 0.9`
in harmonic coordinates the minimum is about `0.44 M`, which is what sets
the finest spacing a run of this package needs (`CODE.md`, "The interior").
"""
function horizon_min_radius end

function horizon_max_radius end

horizon_min_radius(ks::KerrSchild) = ks.mass + sqrt(ks.mass^2 - ks.spin^2)
horizon_max_radius(ks::KerrSchild) =
    sqrt(horizon_min_radius(ks)^2 + ks.spin^2)
horizon_min_radius(ha::Harmonic) = sqrt(ha.mass^2 - ha.spin^2)
horizon_max_radius(ha::Harmonic) =
    sqrt(horizon_min_radius(ha)^2 + ha.spin^2)

horizon_min_radius(m::SpacetimeMetrics.TranslatedMetric) =
    horizon_min_radius(m.metric)
horizon_max_radius(m::SpacetimeMetrics.TranslatedMetric) =
    horizon_max_radius(m.metric)
horizon_min_radius(m::SpacetimeMetrics.RotatedMetric) =
    horizon_min_radius(m.metric)
horizon_max_radius(m::SpacetimeMetrics.RotatedMetric) =
    horizon_max_radius(m.metric)
horizon_min_radius(m::SpacetimeMetrics.BoostedMetric) =
    horizon_min_radius(m.metric) *
    sqrt(1 - (m.velocity[1]^2 + m.velocity[2]^2 + m.velocity[3]^2))
horizon_max_radius(m::SpacetimeMetrics.BoostedMetric) =
    horizon_max_radius(m.metric)

"""
    singular_radius(background) -> T

The largest **coordinate** radius, measured from the hole's center, at
which the background's chart is singular — what the frozen core has to
contain.

For Kerr in either chart that is **`|a|`**, and it is not the origin: both
`KerrSchild` and `Harmonic` solve `R⁴ − R²(x²+y²+z²−a²) − a²z² = 0` for
their radial coordinate, and on the equatorial disk `z = 0`,
`x² + y² ≤ a²` that gives `R = 0`, which every expression in the metric
divides by. The ring singularity is the disk's edge; the disk itself is
the chart's inner boundary, and a grid point on it is `NaN`.

**This is what makes a rapidly spinning hole hard, and it is not the
resolution (found in step 5.)** The frozen core is a *ball* of radius
`r_0`, so it contains the disk only if `r_0 > |a|`, while
[`check_interior_radii`](@ref) needs `r_0 < r_1 ≤ r_h,min − m·h`. For
`KerrSchild` at `a = 9/10` those are compatible — `r_h,min = r₊ = 1.436`
and the disk is at `0.9` — but for `Harmonic` they are **not**:
`r_h,min = √(M² − a²) = 0.436` is *smaller* than the disk's `0.9`. The
Kerr horizon is oblate and the singular disk is flat, so in the harmonic
chart the disk pokes out of any sphere that fits inside the horizon along
the axis. See `CODE.md`, "The interior", for what that means for the
proof-of-concept case and for the two ways out.
"""
function singular_radius end

singular_radius(ks::KerrSchild) = abs(ks.spin)
singular_radius(ha::Harmonic) = abs(ha.spin)
singular_radius(m::SpacetimeMetrics.TranslatedMetric) =
    singular_radius(m.metric)
singular_radius(m::SpacetimeMetrics.RotatedMetric) = singular_radius(m.metric)
# A boost contracts along `v` and leaves the transverse extent, so the
# unboosted radius is an upper bound — which is the side to be wrong on.
singular_radius(m::SpacetimeMetrics.BoostedMetric) = singular_radius(m.metric)
# A background with no horizon never reaches here (the layer needs one),
# and one with a horizon and no listed singular set is a point at the
# center, which every positive `r_0` contains.
singular_radius(::AbstractMetric) = 0

horizon_min_radius(bg::AbstractMetric) = throw(ArgumentError(
    "$(typeof(bg)) has no horizon this package knows the coordinate radius " *
    "of, so the interior layer cannot be placed inside one. CODE.md gives " *
    "r_h,min analytically for KerrSchild and Harmonic (and for translate, " *
    "rotate and boost of either); a background outside that list needs the " *
    "radius stated there before a layer is put in it."))
horizon_max_radius(bg::AbstractMetric) = horizon_min_radius(bg)

# The distance from `c` to the nearest and the farthest point of an
# axis-aligned box, which is what decides whether a block's extent meets
# the sphere `r = R`. Host-side, per block, at regrid cadence.
function _box_radii(ext, c)
    near = zero(float(c[1]))
    far = zero(float(c[1]))
    for d in 1:3
        lo, hi = ext[d][1], ext[d][2]
        below = lo - c[d]
        above = c[d] - hi
        gap = max(below, above, zero(below))
        near += gap * gap
        far += max((c[d] - lo)^2, (hi - c[d])^2)
    end
    return sqrt(near), sqrt(far)
end

_box_meets_sphere(ext, c, R) =
    ((near, far) = _box_radii(ext, c); near ≤ R ≤ far)

_box_meets_ball(ext, c, R) = first(_box_radii(ext, c)) ≤ R

"""
    layer_spacing(forest, interior, t) -> (h, nblocks)

The coarsest spacing among the blocks whose extent meets the sphere
`r = r_1` — the `h` both of `CODE.md`'s radius requirements are stated at,
and the number of blocks that contributed it.

The *coarsest* rather than the finest: the requirements say how many grid
points fit between two radii, and the block with the largest cells is the
one that fits the fewest. On a mesh whose level floor around the horizon
is in place (step 6) they are all the same block size and the distinction
is moot; on a hand-built hierarchy it is not, which is the case this
package tests first.
"""
function layer_spacing(forest::Forest{3}, int::Interior{T}, t) where {T}
    c = center_at(int.center, t)
    h = zero(T)
    n = 0
    for k in forest.leaves
        _box_meets_sphere(block_extent(T, forest, k), c, int.r_1) || continue
        h = max(h, spacing(T, forest, k))
        n += 1
    end
    return h, n
end

"""
    check_interior_radii(forest, interior, background, G; t = 0)

Assert `CODE.md`'s two placement requirements for the damping layer, at
the mesh `forest` currently is:

    r_1 ≤ r_h,min − m · h        (m ≥ G + 1, default m = 8)
    r_1 − r_0 ≥ 2(G + 1) · h

with `h` the coarsest spacing among the blocks containing `r_1`
([`layer_spacing`](@ref)) and `r_h,min` the horizon's smallest coordinate
radius, boost-contracted ([`horizon_min_radius`](@ref)).

The first says the horizon and at least `m` grid points inside it are
evolved by the *unmodified* equations, so that no stencil of a point
outside the horizon reaches the layer. The second says the layer is thick
enough for the profiles to be resolved and for no stencil of an evolved
point to reach the frozen core. Together they are what makes it legitimate
to touch the interior at all, and `CODE.md` asks for them **at every
regrid** — which is here, since a fresh [`GHProblem`](@ref) is built after
every one.

**It throws, and it is meant to.** `CLAUDE.md`: if one fires, raise the
refinement's level floor, shrink the layer, or widen the floor's shell —
do not remove the assertion, and do not lower `m`. The message names which
of the two failed, by how much, and the three remedies.
"""
function check_interior_radii(forest::Forest{3}, int::Interior{T}, background,
                              G::Integer; t=zero(T)) where {T}
    int.margin ≥ G + 1 || throw(ArgumentError(
        "the interior's margin is m = $(int.margin) but the ghost width is " *
        "G = $G, and CODE.md's floor is m ≥ G + 1 = $(G + 1): the margin has " *
        "to cover a whole stencil, or a point outside the horizon reaches " *
        "into the layer and the claim that the horizon is evolved by the " *
        "unmodified equations is false."))
    h, nb = layer_spacing(forest, int, t)
    nb > 0 || throw(ArgumentError(
        "no block of this forest contains the sphere r_1 = $(int.r_1) around " *
        "the center $(center_at(int.center, t)) at t = $t: the layer is " *
        "outside the domain, and the two radius requirements are statements " *
        "about the spacing of the blocks that hold it."))
    r_sing = T(singular_radius(background))
    int.r_0 > r_sing || throw(ArgumentError(
        "the frozen core does not contain the chart's singular set: " *
        "r_0 = $(int.r_0) but $(typeof(background)) is singular out to a " *
        "coordinate radius of $r_sing — for Kerr that is the equatorial " *
        "disk |x| ≤ |a|, z = 0, on which the chart's radial coordinate " *
        "vanishes and every expression in the metric divides by it. A grid " *
        "point there is NaN, and the core rule is what is supposed to keep " *
        "it off the mesh. Raise r_0 above $r_sing — and note that the " *
        "requirement below then has to hold as well, which for the " *
        "harmonic chart at a = 9/10 it cannot (CODE.md, \"The interior\")."))
    r_h = T(horizon_min_radius(background))
    allowed = r_h - int.margin * h
    int.r_1 ≤ allowed || throw(ArgumentError(
        "the damping layer is not far enough inside the horizon: " *
        "r_1 = $(int.r_1) but the horizon's smallest coordinate radius is " *
        "r_h,min = $r_h and the blocks containing r_1 have h = $h, so " *
        "CODE.md's r_1 ≤ r_h,min − m·h with m = $(int.margin) allows at most " *
        "$allowed. The horizon and m grid points inside it must be evolved " *
        "by the unmodified equations. Refine around the hole (a smaller h), " *
        "move r_1 inward, or — never — lower m below G + 1."))
    thickness = int.r_1 - int.r_0
    needed = 2 * (G + 1) * h
    thickness ≥ needed || throw(ArgumentError(
        "the damping layer is too thin: r_1 − r_0 = $thickness at a spacing " *
        "of h = $h, and CODE.md asks for at least 2(G+1)h = $needed so that " *
        "the profiles are resolved by the stencils and no stencil of an " *
        "evolved point reaches the frozen core. Refine around the hole, " *
        "shrink r_0, or raise r_1 — subject to the horizon bound above, " *
        "which is what makes this a resolution requirement and not a free " *
        "choice."))
    return (h=h, nblocks=nb, r_h_min=r_h, allowed=allowed, thickness=thickness,
            needed=needed)
end
