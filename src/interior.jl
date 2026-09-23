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
# **The second half of the file is the tracked geometry (step 8d)**: the same
# three regions below the found horizon's offset surface `r_h(n̂) − m h`
# about the *tracked* center, keyed on the depth under it — `FittedInterior`,
# its real-harmonic shape, its masks and its checks. The two geometries share
# the profiles and speak one protocol (`interior_point`, `is_frozen`,
# `is_outside`, `interior_profiles`, `in_layer(int, t, x)`, `core_position`,
# `interior_mask`, `layer_mask`, `shell_mask`, `geometry_radii`), so that the
# kernels do not know which one they hold.
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

`v` is the hole's coordinate velocity in the lab frame, zero for every
static case; G5 is where it is not. **For `boost(background, u)` it is
`−u`, not `u` (corrected in step 8e):** `SpacetimeMetrics` evaluates the
boosted metric at `Λᵀx`, so the rest frame's origin moves at `−u` in the
lab. [`hole_velocity`](@ref) is the one place that is written, and
[`GHCase`](@ref) derives `v` from it — and refuses a `velocity` keyword that
disagrees.
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
which **the driver sets once per chunk** — by default to the physical rate
`4/M` of [`default_relaxation_rate`](@ref), or to the grid rate
`ρ_max_factor/dt` or a fixed `ρ_max_fixed` when [`evolve!`](@ref) is given
one (`CODE.md`, "The profiles and their parameters": `ρ_max = 4/M`,
decided 2026-09-23 in step 8c′; `1/dt` was the default until then). RK4
bounds any of them at about `2.8/dt` on the negative real axis. A value
carried in a case is therefore a placeholder, and [`with_ρ_max`](@ref) is
how the driver replaces it.

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

# `:fitted` (added in step 8e) relaxes toward the fitted target on the
# tracked geometry and is a `FittedInterior`'s only: `Interior` refuses it.
const INTERIOR_VARIANTS = (:damped, :pasted, :frozen, :fitted)

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
    variant === :fitted && throw(ArgumentError(
        "the :fitted variant relaxes toward a fit of the state on the tracked " *
        "geometry's offset surface (CODE.md, \"The fitted target\"), and a " *
        "sphere about the analytic center has none: give the case interior = " *
        "FittedSpec(T; variant = :fitted, …) and a Horizon to track it with."))
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
once per chunk: at the default `4/M` the rate is the same in every chunk,
and at the grid rate `ρ_max_factor/dt` it follows that chunk's `dt`
(`CODE.md`, "The profiles and their parameters"; amended in step 8c′).

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

@inline function interior_radius(int::Interior{T}, t, x) where {T}
    c = center_at(int.center, t)
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    return sqrt(d1 * d1 + d2 * d2 + d3 * d3)
end

"""
    in_layer(interior, t, x) -> Bool

Whether the point `x` at time `t` is inside the damping layer proper: the
region the interior residual is measured over, and the only region where
`u_exact` is evaluated during an evaluation — `r_0 ≤ r < r_1` for the
sphere of an [`Interior`](@ref), and `r_0(n̂) ≤ r < r_1(n̂)` below the
offset surface of a [`FittedInterior`](@ref).

**A position and a time, not a radius (amended in step 8d).** It took `r`
until the layer stopped being a sphere: once the layer's radii depend on the
direction, a radius alone does not say whether a point is in it, and the
protocol is the same for both types so that no caller has to know which one
it holds. `nothing` — no hole — has no layer.
"""
@inline in_layer(::Nothing, t, x) = false

@inline function in_layer(int::Interior, t, x)
    r = interior_radius(int, t, x)
    return (int.r_0 ≤ r) & (r < int.r_1)
end

# The kernel's view of an interior at one point (added in step 8d): what the
# three predicates below and the profiles are evaluated on. For the sphere it
# is the radius `r` itself, so the kernel's arithmetic on an `Interior` is
# exactly what it was before a second geometry existed; for the fitted
# geometry it is the radius together with the two surfaces' radii along the
# ray ([`interior_point`](@ref) of a `FittedInterior`).
@inline interior_point(int::Interior, t, x) = interior_radius(int, t, x)

# Whether the point is evolved by the unmodified equations — `r ≥ r_1`, the
# boundary included, as it always was.
@inline is_outside(int::Interior, r) = r ≥ int.r_1

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
@inline interior_profiles(int::Interior{T}, r) where {T} =
    _layer_profiles(r, int.r_0, int.r_1, int.ρ_max, int.w_ramp, int.ρ_ramp)

@inline interior_profiles(int::Interior{T,:frozen}, r) where {T} =
    (_layer_w(r, int.r_0, int.r_1, int.w_ramp), zero(T))

# The two profiles between the core's radius `r_0` and the layer's `r_1`,
# written **once** for both geometries (step 8d): the sphere passes its two
# radii and the fitted geometry the two radii of its surfaces along the ray,
# and the arithmetic is the same expression in the same order, which is what
# makes a `FittedInterior` holding a sphere bit for bit an `Interior` —
# `test/tracking_tests.jl` asserts it on a right-hand side. It is the
# expression step 5 wrote, unchanged.
@inline function _layer_profiles(r, r_0, r_1, ρ_max, w_ramp, ρ_ramp)
    s = (r - r_0) / (r_1 - r_0)
    w = smoothstep(s / w_ramp)
    ρ = ρ_max * smoothstep((1 - s) / ρ_ramp)
    return w, ρ
end

@inline function _layer_w(r, r_0, r_1, w_ramp)
    s = (r - r_0) / (r_1 - r_0)
    return smoothstep(s / w_ramp)
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

"""
    layer_mask(interior, t) -> mask
    shell_mask(interior, t, width) -> mask

The two bands the validity monitor and the variants' comparison read, as
masks: **the layer** itself, `r_0 ≤ r < r_1`, and **the shell** of `width`
just outside it, `r_1 ≤ r < r_1 + width` — a [`ShellMask`](@ref) about
`c(t)` for the sphere, and the same two bands below and above the offset
surface for a [`FittedInterior`](@ref) (a [`ShapeBand`](@ref)).

Added in step 8d so that no caller builds a `ShellMask` from `r_0` and `r_1`
itself: once the layer follows a surface, "the `G` points outside `r_1`" is
a band about that surface, and the caller should not have to know which
geometry it holds.
"""
layer_mask(int::Interior{T}, t) where {T} =
    ShellMask{T}(center_at(int.center, t), int.r_0, int.r_1)

shell_mask(int::Interior{T}, t, width) where {T} =
    ShellMask{T}(center_at(int.center, t), int.r_1, int.r_1 + T(width))

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
    geometry_radii(interior, background) -> (r_min, r_max)

The smallest and largest coordinate radius of the **horizon the layer is
placed inside**, about the interior's own center: the background's analytic
radii ([`horizon_min_radius`](@ref), [`horizon_max_radius`](@ref)) for step
5's sphere, and the tracked shape's bounding radii `r_in`, `r_out` for a
[`FittedInterior`](@ref), whose background is not consulted.

**One accessor for both (added in step 8d)**, so that the refinement's level
floor ([`horizon_floor_level`](@ref), [`level_bounds`](@ref)) reads the
horizon the layer actually follows: a floor derived from the analytic radii
around a tracked layer would be a statement about a surface nobody measured.
"""
geometry_radii(int::Interior{T}, background) where {T} =
    (T(horizon_min_radius(background)), T(horizon_max_radius(background)))

"""
    layer_radii(interior) -> (r_0, r_1)

The layer's innermost radii: the sphere's `r_0` and `r_1`, and for a
[`FittedInterior`](@ref) the smallest radius of its core surface and of its
offset surface, `r_in − offset − thickness` and `r_in − offset` — what the
level floor's two requirements are stated in (added in step 8d).
"""
layer_radii(int::Interior) = (int.r_0, int.r_1)

"""
    hole_mass(background) -> T

The hole's **mass parameter** `M`, for the backgrounds that have one — the
unit every physical rate of the interior is stated in.

`KerrSchild` and `Harmonic` carry it as `.mass`; `translate`, `rotate` and
`boost` pass it through unchanged, exactly as they pass the horizon radii
above. For the boost that is a statement about what `M` *is* rather than a
convenience: the mass parameter is the hole's rest mass, which a boost does
not change (the boosted hole's energy is `M/√(1 − v²)`, and nothing here
asks for it).

It is what the driver's default relaxation rate is read from
([`default_relaxation_rate`](@ref), added in step 8c′), so that the default
is a statement about the hole and not a number in the driver: `CODE.md`,
"The profiles and their parameters".
"""
function hole_mass end

hole_mass(ks::KerrSchild) = ks.mass
hole_mass(ha::Harmonic) = ha.mass
hole_mass(m::SpacetimeMetrics.TranslatedMetric) = hole_mass(m.metric)
hole_mass(m::SpacetimeMetrics.RotatedMetric) = hole_mass(m.metric)
# The rest mass: a boost changes the hole's energy and its coordinate shape,
# not the parameter the solution is written in.
hole_mass(m::SpacetimeMetrics.BoostedMetric) = hole_mass(m.metric)

"""
    hole_velocity(background) -> SVector{3}

The hole's **coordinate velocity in the lab frame**, for the backgrounds that
have a hole: the `v` of the analytic trajectory `c(t) = c₀ + v t` that
[`GHCase`](@ref) builds its [`HoleCenter`](@ref) from.

`KerrSchild` and `Harmonic` are static, so zero. `translate` moves the hole
and does not change its velocity; `rotate` turns it, `R v` (the metric at `x`
is the unrotated one at `Rᵀx`). **`boost(m, v)` moves the hole at `−v`
(found in step 8d, fixed in step 8e):** `SpacetimeMetrics` evaluates the
boosted metric as `Λ g(Λᵀx) Λᵀ`, with `Λ`'s `+γv` entries, so the rest
frame's origin `Λᵀx = 0` is the lab's `x = −v t` —
`metric(boost(KerrSchild(1, 0), (0.3, 0, 0)), (1, x, 0, 0))` is singular at
`x = −0.3`, not at `+0.3`. A boost of a hole that already moves composes the
two velocities relativistically (`Λ(−v)` applied to the inner hole's
4-velocity), which is `−v` exactly when the inner hole is static.

A background this package cannot classify — flat space in any chart, or a
metric outside the list — is refused by name, as [`hole_mass`](@ref) refuses
it: a velocity that is a guess would put the layer, the damping profile and
the tracked seed on the wrong trajectory without saying so.
"""
function hole_velocity end

hole_velocity(ks::KerrSchild) = zero(SVector{3,typeof(ks.mass)})
hole_velocity(ha::Harmonic) = zero(SVector{3,typeof(ha.mass)})
hole_velocity(m::SpacetimeMetrics.TranslatedMetric) = hole_velocity(m.metric)

function hole_velocity(m::SpacetimeMetrics.RotatedMetric)
    u = hole_velocity(m.metric)
    R = m.R
    # `x = R x_old` on the spatial block, so a trajectory `x_old = u t` is
    # `x = (R u) t`.
    return SVector(R[2, 2] * u[1] + R[2, 3] * u[2] + R[2, 4] * u[3],
                   R[3, 2] * u[1] + R[3, 3] * u[2] + R[3, 4] * u[3],
                   R[4, 2] * u[1] + R[4, 3] * u[2] + R[4, 4] * u[3])
end

function hole_velocity(m::SpacetimeMetrics.BoostedMetric)
    u = hole_velocity(m.metric)
    v = m.velocity
    # The rest frame's origin `Λᵀx = 0` is `x = −v t`: exactly `−v`, not a
    # composition that rounds to it, for the static hole every case has.
    iszero(u) && return -v
    # `x = Λ(−v) x_old` for the symmetric `Λ`, so the lab 4-velocity is
    # `Λ(−v)` of `γ_u (1, u)` and the lab velocity is its ratio.
    β² = v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    γ = 1 / sqrt(1 - β²)
    f = (γ - 1) / β²
    vu = v[1] * u[1] + v[2] * u[2] + v[3] * u[3]
    den = γ * (1 - vu)
    return SVector(ntuple(i -> (u[i] + f * v[i] * vu - γ * v[i]) / den, 3))
end

hole_velocity(bg::AbstractMetric) = throw(ArgumentError(
    "$(typeof(bg)) has no hole whose velocity this package knows how to " *
    "read: hole_velocity classifies KerrSchild and Harmonic (static) and " *
    "translate, rotate and boost of either — and SpacetimeMetrics' " *
    "boost(m, v) moves the hole at −v, since it evaluates m at Λᵀx. A " *
    "background outside that list needs a hole_velocity method before a " *
    "case derives its trajectory from it; a case with no hole passes its " *
    "`velocity` explicitly or takes zero."))

# Whether `hole_velocity` classifies the background: the one question
# `GHCase` asks before deriving a trajectory from it, so that a flat-space
# case keeps its zero velocity without a `try` (added in step 8e).
has_hole_velocity(::Union{KerrSchild,Harmonic}) = true
has_hole_velocity(m::Union{SpacetimeMetrics.TranslatedMetric,
                           SpacetimeMetrics.RotatedMetric,
                           SpacetimeMetrics.BoostedMetric}) =
    has_hole_velocity(m.metric)
has_hole_velocity(::AbstractMetric) = false

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

hole_mass(bg::AbstractMetric) = throw(ArgumentError(
    "$(typeof(bg)) has no hole mass this package knows how to read, so " *
    "there is no physical time scale to state the interior's relaxation " *
    "rate in: the default rate is 4/M (CODE.md, \"The profiles and their " *
    "parameters\"), and M is read as .mass of KerrSchild and Harmonic, " *
    "through translate, rotate and boost. A background outside that list " *
    "needs a hole_mass method, or the run an explicit ρ_max_fixed or " *
    "ρ_max_factor."))

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
                              G::Integer; t=zero(T), center=nothing) where {T}
    # `center` is the tracked geometry's (step 8d): the sphere's own center
    # is the analytic one, so there is no distance between the two to add.
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

# --- the tracked geometry (step 8d) -----------------------------------------
#
# `CODE.md`, "The interior" — "The tracked geometry" — and `PLAN.md`'s
# finding 3. Step 5's layer is a sphere about the hole's *analytic* center;
# from step 8d it can instead follow the **found** horizon: a surface
# `r_h(n̂)` about the *tracked* center, held as real spherical-harmonic
# coefficients, and the layer keyed on the **depth**
#
#     d = r_h(n̂) − offset − |x − c(t)|,        offset = m h,
#
# below the offset surface `r_1(n̂) = r_h(n̂) − m h`. The sphere is the `l = 0`
# case of this geometry, and a `FittedInterior` holding a sphere is bit for
# bit an `Interior` — the profiles are one function, `_layer_profiles`, and
# the surfaces' radii along the ray are what it is handed.
#
# Everything here is still a function of position and time and of the
# *track*; nothing knows about a block (the geometry's spacing `h` is a
# number it was built with, and `check_interior_radii` is still the only
# thing that looks at a mesh).

# --- real spherical harmonics ------------------------------------------------

"""
    real_harmonic_index(l, m) -> Int

Where the real coefficient of degree `l` and order `m` lives in a shape
vector: **`l² + l + m + 1`, the same slot as the complex canonical layout of
`AbstractSphericalHarmonics` (`ash_mode_index`)**, with `m ≥ 0` holding the
`l0` and cosine coefficients and `m < 0` the sine coefficient of order `|m|`
(proposed in step 8d). The real harmonics are

    ỹ_l0 = Y_l0,    ỹ_lm^c = √2 Re Y_lm,    ỹ_lm^s = −√2 Im Y_lm   (m ≥ 1)

with `Y_lm` the orthonormal, Condon–Shortley-phased harmonics of
`AbstractSphericalHarmonics` (`sYlm(0, l, m, θ, φ)`), and a real function
`f = Σ_lm c_lm Y_lm` — whose coefficients obey `c_{l,−m} = (−1)^m c̄_lm` —
is `Σ a_l0 ỹ_l0 + Σ_{m ≥ 1} (a_lm^c ỹ_lm^c + a_lm^s ỹ_lm^s)` with

    a_l0 = Re c_l0,    a_lm^c = √2 Re c_lm,    a_lm^s = √2 Im c_lm .

[`real_from_complex`](@ref) and [`complex_from_real`](@ref) are the two
directions, and `test/tracking_tests.jl` holds the conversion against
`sYlm` and `ash_evaluate` to roundoff. **Step 8e's fit shares this ordering
and this conversion**, so that a fitted target and the shape it is fitted on
are read by the same recurrence ([`shape_series`](@ref)).
"""
@inline real_harmonic_index(l::Integer, m::Integer) = l * l + l + m + 1

"""
    real_from_complex(c, lmax) -> Vector{Float64}
    complex_from_real(a, lmax) -> Vector{ComplexF64}

The real coefficients of the real function whose complex coefficients (in
the canonical layout) are `c`, and back — [`real_harmonic_index`](@ref)'s
convention, truncated or zero-padded to `lmax` (the resampling of
`AbstractSphericalHarmonics.ash_resample`, whose layout is the same).

`real_from_complex` reads only `m ≥ 0`: the reality condition makes the
negative orders a copy, and a vector that violated it would describe a
complex function, which a horizon's radius is not.
"""
function real_from_complex(c::AbstractVector{<:Complex}, lmax::Integer)
    L = isqrt(length(c)) - 1
    (L + 1)^2 == length(c) || throw(ArgumentError(
        "a complex coefficient vector in the canonical layout has (L+1)² " *
        "entries, got $(length(c))"))
    a = zeros(Float64, (lmax + 1)^2)
    s2 = sqrt(2.0)
    for l in 0:min(L, lmax), m in 0:l
        z = c[real_harmonic_index(l, m)]
        if m == 0
            a[real_harmonic_index(l, 0)] = real(z)
        else
            a[real_harmonic_index(l, m)] = s2 * real(z)
            a[real_harmonic_index(l, -m)] = s2 * imag(z)
        end
    end
    return a
end

function complex_from_real(a::AbstractVector{<:Real}, lmax::Integer)
    L = isqrt(length(a)) - 1
    (L + 1)^2 == length(a) || throw(ArgumentError(
        "a real coefficient vector has (L+1)² entries, got $(length(a))"))
    c = zeros(ComplexF64, (lmax + 1)^2)
    is2 = 1 / sqrt(2.0)
    for l in 0:min(L, lmax)
        c[real_harmonic_index(l, 0)] = a[real_harmonic_index(l, 0)]
        for m in 1:l
            z = is2 * complex(a[real_harmonic_index(l, m)],
                              a[real_harmonic_index(l, -m)])
            c[real_harmonic_index(l, m)] = z
            c[real_harmonic_index(l, -m)] = (isodd(m) ? -1 : 1) * conj(z)
        end
    end
    return c
end

# One term of the series, `a ỹ` for `m = 0` and `a^c ỹ^c + a^s ỹ^s` above it,
# with `q` the reduced Legendre value and `C + iS = (n_x + i n_y)^m`.
@inline function _shape_term(shape, l::Int, m::Int, q, C, S, s2)
    if m == 0
        return (@inbounds shape[l * l + l + 1]) * q
    else
        ac = @inbounds shape[l * l + l + m + 1]
        as = @inbounds shape[l * l + l - m + 1]
        return s2 * q * (ac * C - as * S)
    end
end

"""
    shape_series(shape::SVector, lmax, n̂) -> T

`Σ_lm a_lm ỹ_lm(n̂)` for the real coefficients `shape` of
[`real_harmonic_index`](@ref)'s layout, at the unit vector `n̂` — the
kernel-side evaluation of a tracked horizon's radius.

**No angle is formed.** `Y_lm = q_lm(cos θ) sin^m θ e^{imφ}` with `q_lm` the
normalized associated Legendre function divided by `sin^m θ`, and
`sin^m θ e^{imφ} = (n_x + i n_y)^m` for a unit vector — so the azimuthal
factor is the Chebyshev recurrence for `cos mφ, sin mφ` *multiplied through
by `sin^m θ`*, a complex power of `(n_x, n_y)`, and the polar one the fully
normalized recurrences in `z = n_z`,

    q_00 = 1/√(4π),   q_mm = −√((2m+1)/2m) q_{m−1,m−1},
    q_{m+1,m} = √(2m+3) z q_mm,
    q_lm = √((4l²−1)/(l²−m²)) (z q_{l−1,m} − √(((l−1)²−m²)/(4(l−1)²−1)) q_{l−2,m}),

which is Condon–Shortley's phase. Both are polynomials in `n̂`, so there is
no `atan`, no division by `sin θ`, and no special case on the axis — the
guard the Chebyshev recurrence would need there is what multiplying it
through by `sin^m θ` removes.

The loop bound is the runtime `lmax`, and the coefficients are read by a
runtime index, so a kernel is compiled per length of `shape` and not per
degree of the terms it happens to hold. No allocation, no `return` inside a
kernel body (it is an `@inline` function *called* from one), generic in `T`:
the recurrence coefficients are square roots of exact integers in `T`.
"""
@inline function shape_series(shape::SVector{NM,T}, lmax::Int, n) where {NM,T}
    x = T(n[1])
    y = T(n[2])
    z = T(n[3])
    s2 = sqrt(T(2))
    acc = zero(T)
    qmm = inv(sqrt(4 * T(π)))
    C = one(T)
    S = zero(T)
    m = 0
    while m ≤ lmax
        if m > 0
            qmm = -sqrt(T(2m + 1) / T(2m)) * qmm
            C, S = C * x - S * y, C * y + S * x
        end
        acc += _shape_term(shape, m, m, qmm, C, S, s2)
        if m + 1 ≤ lmax
            qa = qmm
            qb = sqrt(T(2m + 3)) * z * qmm
            acc += _shape_term(shape, m + 1, m, qb, C, S, s2)
            l = m + 2
            while l ≤ lmax
                α = sqrt(T(4 * l * l - 1) / T(l * l - m * m))
                β = sqrt(T((l - 1) * (l - 1) - m * m) /
                         T(4 * (l - 1) * (l - 1) - 1))
                qc = α * (z * qb - β * qa)
                acc += _shape_term(shape, l, m, qc, C, S, s2)
                qa = qb
                qb = qc
                l += 1
            end
        end
        m += 1
    end
    return acc
end

# The directions a shape's bounding radii are sampled over: both poles, and
# `n_θ` interior colatitudes `π t/(n_θ + 1)` (odd `n_θ`, so the equator is
# one) by `2 n_θ` longitudes — four times the grid a degree-`lmax` series
# needs, with the two directions an axisymmetric horizon has its extremes in
# sampled exactly.
function shape_sample_directions(lmax::Integer)
    nθ = 4 * Int(lmax) + 3
    nφ = 2 * nθ
    dirs = SVector{3,Float64}[SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, -1.0)]
    for t in 1:nθ, p in 0:(nφ - 1)
        θ = π * t / (nθ + 1)
        φ = 2π * p / nφ
        push!(dirs, SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)))
    end
    return dirs
end

"""
    shape_bounds(shape::SVector, lmax) -> (r_in, r_out)

The smallest and largest value of the series over
[`shape_sample_directions`](@ref) — the bounding spheres of the surface, in
the shape's own type.

They are not a rigorous bound on the series between the samples, and they
do not have to be: the geometry is **defined** as the series clamped into
`[r_in, r_out]` ([`shape_radius`](@ref)), so the fast paths that skip the
series outside `r_out − offset` and inside `r_in − offset − thickness` are
exact shortcuts of the evaluation rather than approximations of it
(proposed in step 8d). Where the true surface pokes past a sampled extreme
it does so by the series' own variation between samples, a small fraction
of the shape's `l ≥ 1` amplitude; the clamp flattens it there.
"""
function shape_bounds(shape::SVector{NM,T}, lmax::Integer) where {NM,T}
    lo = floatmax(T)
    hi = -floatmax(T)
    for n in shape_sample_directions(lmax)
        v = shape_series(shape, Int(lmax), SVector{3,T}(n))
        lo = min(lo, v)
        hi = max(hi, v)
    end
    return lo, hi
end

# --- the analytic horizon, for the seed ---------------------------------------

"""
    analytic_horizon_radius(background, n̂) -> T

The coordinate distance from the hole's center to its horizon along the
unit vector `n̂`, for the backgrounds this package knows — the surface the
tracked geometry is **seeded** with before the first find (step 8d).

Both Kerr charts put the horizon on the oblate spheroid
`(x² + y²)/(R² + a²) + z²/R² = 1` of their own radial coordinate — `R = r₊ =
M + √(M² − a²)` for `KerrSchild`, `R = r₊ − M = √(M² − a²)` for `Harmonic` —
so along `n̂`

    r_h(θ) = R √((R² + a²)/(R² + a² cos²θ)),    cos θ = n̂_z,

which is `R` on the axis and `√(R² + a²)` on the equator:
[`horizon_min_radius`](@ref) and [`horizon_max_radius`](@ref).
`SpacetimeMetrics` exposes no horizon of its own, so `test/tracking_tests.jl`
checks this against the charts' quartic for the radial coordinate instead.
`translate` leaves the shape alone (it moves the center, which is the case's
[`HoleCenter`](@ref)); `rotate` turns it, `r_h(Rᵀ n̂)`; `boost` contracts it
along `v` by `√(1 − v²)` — the lab-frame point `ρ n̂` sits at `ρ n̂′`,
`n̂′ = n̂ + (γ − 1)(v̂·n̂)v̂`, in the hole's rest frame, where the horizon is
static, so `ρ = r_h(n̂′/|n̂′|)/|n̂′|`.
"""
function analytic_horizon_radius end

@inline function _spheroid_radius(R, a, cz)
    R² = R * R
    a² = a * a
    return R * sqrt((R² + a²) / (R² + a² * cz * cz))
end

analytic_horizon_radius(ks::KerrSchild, n) =
    _spheroid_radius(horizon_min_radius(ks), ks.spin, n[3])
analytic_horizon_radius(ha::Harmonic, n) =
    _spheroid_radius(horizon_min_radius(ha), ha.spin, n[3])
analytic_horizon_radius(m::SpacetimeMetrics.TranslatedMetric, n) =
    analytic_horizon_radius(m.metric, n)

function analytic_horizon_radius(m::SpacetimeMetrics.RotatedMetric, n)
    R = m.R
    # `x_old = Rᵀ x`, spatial block only.
    n_old = SVector(R[2, 2] * n[1] + R[3, 2] * n[2] + R[4, 2] * n[3],
                    R[2, 3] * n[1] + R[3, 3] * n[2] + R[4, 3] * n[3],
                    R[2, 4] * n[1] + R[3, 4] * n[2] + R[4, 4] * n[3])
    return analytic_horizon_radius(m.metric, n_old)
end

function analytic_horizon_radius(m::SpacetimeMetrics.BoostedMetric, n)
    v = m.velocity
    β² = v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    iszero(β²) && return analytic_horizon_radius(m.metric, n)
    γ = 1 / sqrt(1 - β²)
    vn = (v[1] * n[1] + v[2] * n[2] + v[3] * n[3]) / β²
    n′ = SVector(n[1] + (γ - 1) * vn * v[1], n[2] + (γ - 1) * vn * v[2],
                 n[3] + (γ - 1) * vn * v[3])
    s = sqrt(n′[1] * n′[1] + n′[2] * n′[2] + n′[3] * n′[3])
    return analytic_horizon_radius(m.metric, n′ / s) / s
end

analytic_horizon_radius(bg::AbstractMetric, n) = horizon_min_radius(bg)

# --- what a case holds for the tracked geometry -------------------------------

"""
    FittedSpec(T = Float64; variant = :damped, margin = 8, n_L = 0,
               core_min = 2, lmax_shape = 4, lmax_fit = 8, ρ_max = 0,
               w_ramp = 1//2, ρ_ramp = 1, max_misses = 3, α_trigger = 1//10,
               target = nothing, target_bounds = nothing)

What a [`GHCase`](@ref) holds as its `interior` for the **tracked**
geometry (step 8d): not a layer, but the rule a layer is built by, once per
chunk, from the tracked horizon ([`fitted_interior`](@ref)). The case is
the hole; the geometry is a function of the run.

- `variant` is `CODE.md`'s switch, as for [`Interior`](@ref); all three run
  on the tracked geometry.
- `margin` is `m`, the offset surface's depth below the found horizon in
  spacings — `8`, step 8a's default (`PLAN.md`'s finding 3 uses `4` on
  harmonic Kerr's equator and says the margin will want to depend on the
  direction; one `m` for now).
- `n_L` is the ramp in spacings; `0` means step 8c's rule,
  `max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)` — `8` at `q = 2`, `12` at `q = 4` at
  the default rate — which needs the scheme's `G` and is therefore resolved
  by [`evolve!`](@ref), where `q` is known ([`layer_cells`](@ref)).
- `core_min` is the smallest core the geometry may leave, in spacings:
  `fitted_interior` refuses `r_in − (m + n_L + core_min) h ≤ 0` by name.
- `lmax_shape` is the degree the found shape is truncated to, and
  `lmax_fit` the degree of step 8e's fitted target ([`build_fit`](@ref)):
  `L = 8` by default, the collocation grid `EquiangularGrid(L)` on the offset
  surface (added in step 8e).
- `ρ_max = 0` means the driver's default rate `4/M` (step 8c′); a positive
  number is this case's default rate instead. `w_ramp` and `ρ_ramp` are the
  profiles' ramps; **`ρ_ramp = 1` and `w_ramp = 1/2` are step 8c's rule** —
  `n_L` is the width over which `ρ` rises from `0` at the offset surface to
  `ρ_max`, and `w` turns over in the inner half **(proposed in step 8d**:
  step 5's `1/2`, `1/2` is the fixture's, and `n_L`'s calibration is for
  the full ramp**)**.
- `max_misses` is how many consecutive failed finds the track coasts
  through before the run ends ([`update_track`](@ref)); `α_trigger` is the
  lapse below which, over the evolved region, a find is forced at the next
  chunk boundary whatever the cadence (the lapse-collapse trigger).
- `target` is what the layer relaxes toward, as for [`Interior`](@ref) —
  for the analytic variants; `:fitted` (step 8e) relaxes toward the fit of
  the state and takes none.
- `target_bounds` is the [`StateBounds`](@ref) the `:fitted` target is
  projected into ([`fit_state`](@ref)); `nothing`, the default, derives them
  at the start of a run from the analytic data on the seed's offset surface
  ([`derive_target_bounds`](@ref)), **decided in review, step 8e**, because
  `default_bounds` is Kerr-Schild's and harmonic Kerr's data exceeds it.

`isbits`: the numbers that have a "use the rule" value spell it `0`, since a
`Union{Nothing, T}` field would not be; the two optional objects are type
parameters.
"""
struct FittedSpec{T,V,X,B}
    margin::Int
    n_L::Int
    core_min::Int
    lmax_shape::Int
    lmax_fit::Int
    ρ_max::T
    w_ramp::T
    ρ_ramp::T
    max_misses::Int
    α_trigger::T
    valvariant::Val{V}
    target::X
    target_bounds::B
end

function FittedSpec(::Type{T}=Float64; variant::Symbol=:damped,
                    margin::Integer=8, n_L::Integer=0, core_min::Integer=2,
                    lmax_shape::Integer=4, lmax_fit::Integer=8, ρ_max=zero(T),
                    w_ramp=T(1 // 2),
                    ρ_ramp=one(T), max_misses::Integer=3,
                    α_trigger=T(1 // 10), target=nothing,
                    target_bounds=nothing) where {T}
    variant in INTERIOR_VARIANTS || throw(ArgumentError(
        "the interior variant must be one of $(INTERIOR_VARIANTS), got " *
        ":$variant; the tracked geometry runs CODE.md's three analytic " *
        "variants as the sphere does, and step 8e's :fitted."))
    variant === :fitted && target !== nothing && throw(ArgumentError(
        "a :fitted layer relaxes toward the fit of the evolved state, so it " *
        "takes no analytic target metric; `target` is for :damped, :frozen " *
        "and :pasted."))
    target_bounds === nothing || target_bounds isa StateBounds || throw(ArgumentError(
        "target_bounds is a StateBounds (the ranges the :fitted target is " *
        "projected into) or nothing, to derive them from the seed's data; " *
        "got a $(typeof(target_bounds))."))
    margin ≥ 1 || throw(ArgumentError(
        "the margin m is a number of spacings and must be at least 1, got " *
        "$margin; CODE.md's default is 8 and its floor G + 1, which " *
        "check_interior_radii asserts once the scheme's G is known."))
    n_L ≥ 0 || throw(ArgumentError(
        "the ramp n_L is a number of spacings, or 0 for step 8c's rule, got " *
        "$n_L."))
    core_min ≥ 1 || throw(ArgumentError(
        "core_min is the smallest core the geometry may leave, in spacings, " *
        "and must be at least 1, got $core_min: a core surface that reaches " *
        "the center has no inside for the core rule to project onto."))
    lmax_shape ≥ 0 || throw(ArgumentError(
        "lmax_shape is a spherical-harmonic degree, got $lmax_shape."))
    lmax_fit ≥ 1 || throw(ArgumentError(
        "lmax_fit is the fitted target's spherical-harmonic degree and must " *
        "be at least 1, got $lmax_fit: the shift has no constant term, so a " *
        "degree-0 fit could not hold a radial shift at all."))
    T(ρ_max) ≥ 0 || throw(ArgumentError(
        "ρ_max is a relaxation rate, or 0 for the driver's default 4/M, got " *
        "$ρ_max."))
    wr, ρr = T(w_ramp), T(ρ_ramp)
    (0 < wr ≤ 1 && 0 < ρr ≤ 1) || throw(ArgumentError(
        "the ramp fractions must lie in (0, 1], got w_ramp = $wr and " *
        "ρ_ramp = $ρr."))
    max_misses ≥ 1 || throw(ArgumentError(
        "max_misses counts failed finds the track may coast through and must " *
        "be at least 1, got $max_misses."))
    T(α_trigger) ≥ 0 || throw(ArgumentError(
        "α_trigger is a lapse and must be non-negative (0 switches the " *
        "trigger off), got $α_trigger."))
    check_layer_target(target)
    tb = target_bounds === nothing ? nothing : _bounds_in(T, target_bounds)
    return FittedSpec{T,variant,typeof(target),typeof(tb)}(
        Int(margin), Int(n_L), Int(core_min), Int(lmax_shape), Int(lmax_fit),
        T(ρ_max), wr,
        ρr, Int(max_misses), T(α_trigger), Val(variant), target, tb)
end

interior_variant(::FittedSpec{T,V}) where {T,V} = V

"""
    layer_cells(G, ρ_max, M) -> Int

Step 8c's rule for the ramp of a layer relaxing at `ρ_max` around a hole of
mass `M`, in spacings: `n_L = max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)` — `8` at
`q = 2` and `12` at `q = 4` at the default `4/M` (`CODE.md`, "The layer for
an inexact target", measured at `q = 2` and **(proposed in step 8c)** for
other orders).
"""
layer_cells(G::Integer, ρ_max, M) =
    max(4 * Int(G), ceil(Int, G * cbrt(10 * Float64(ρ_max) * Float64(M))))

# --- the kernel argument ------------------------------------------------------

"""
    FittedInterior(T = Float64; center, shape, lmax = √length − 1, offset,
                   thickness, ρ_max = 0, variant = :damped, w_ramp = 1//2,
                   ρ_ramp = 1//2, margin = 8, n_L = 0, h = 0,
                   target = nothing, r_in = nothing, r_out = nothing)

The damping layer on the **tracked** geometry (step 8d): the `isbits`
kernel argument beside [`Interior`](@ref), sharing its variant `Val` — the
right-hand-side kernel's `Val{INT}` is `:damped`, `:frozen` or `:pasted`
for both — and its profiles.

- `center` is a [`HoleCenter`](@ref): the **tracked** trajectory
  ([`track_center`](@ref)), `c(t) = c_find + v_est (t − t_find)`, so every
  consumer of a center — the masks, `interior_radius`, the range
  projection's gate — works on it unchanged.
- `shape` holds the real spherical-harmonic coefficients to `lmax` of the
  horizon's radius `r_h(n̂)` about `center` ([`real_harmonic_index`](@ref)),
  and `r_in`, `r_out` its bounding radii ([`shape_bounds`](@ref)): the
  surface **is** the series clamped into `[r_in, r_out]`
  ([`shape_radius`](@ref)).
- `offset = m h` puts the offset surface `r_1(n̂) = r_h(n̂) − offset` the
  margin inside the horizon, and `thickness = n_L h` the core surface
  `r_0(n̂) = r_1(n̂) − thickness` the ramp inside that — the **depth**
  `d = r_1(n̂) − r` is what the layer is keyed on.
- `ρ_max`, `w_ramp`, `ρ_ramp` and `target` are [`Interior`](@ref)'s, with
  the same meaning in the depth that they have in the radius there.
- `margin`, `n_L` and `h` record the rule the geometry was built by — `m`,
  the ramp in spacings, and the spacing — for the checks and the record;
  no kernel reads them.

The regions keep the sphere's conventions at their boundaries: a point on
the offset surface is evolved, a point on the core surface is in the layer.
"""
struct FittedInterior{T,V,NM,X}
    center::HoleCenter{T}
    shape::SVector{NM,T}
    lmax::Int
    r_in::T
    r_out::T
    offset::T
    thickness::T
    ρ_max::T
    w_ramp::T
    ρ_ramp::T
    margin::Int
    n_L::Int
    h::T
    valvariant::Val{V}
    target::X
end

function FittedInterior(::Type{T}=Float64; center, shape, lmax=nothing,
                        offset, thickness, ρ_max=zero(T),
                        variant::Symbol=:damped, w_ramp=T(1 // 2),
                        ρ_ramp=T(1 // 2), margin::Integer=8, n_L::Integer=0,
                        h=zero(T), target=nothing, r_in=nothing,
                        r_out=nothing) where {T}
    variant in INTERIOR_VARIANTS || throw(ArgumentError(
        "the interior variant must be one of $(INTERIOR_VARIANTS), got " *
        ":$variant."))
    L = lmax === nothing ? isqrt(length(shape)) - 1 : Int(lmax)
    (L + 1)^2 == length(shape) || throw(ArgumentError(
        "a shape of degree lmax = $L has (lmax + 1)² = $((L + 1)^2) real " *
        "coefficients, got $(length(shape)) (the layout is " *
        "real_harmonic_index's)."))
    c = center isa HoleCenter ? HoleCenter{T}(SVector{3,T}(center.c0),
                                              SVector{3,T}(center.v)) :
        HoleCenter(T, center)
    sv = SVector{(L + 1)^2,T}(ntuple(i -> T(shape[i]), (L + 1)^2))
    lo, hi = r_in === nothing || r_out === nothing ? shape_bounds(sv, L) :
             (T(r_in), T(r_out))
    off, th = T(offset), T(thickness)
    wr, ρr = T(w_ramp), T(ρ_ramp)
    0 < lo ≤ hi || throw(ArgumentError(
        "the horizon's bounding radii must satisfy 0 < r_in ≤ r_out, got " *
        "r_in = $lo, r_out = $hi: a shape that reaches the center is not " *
        "the surface of a hole."))
    (off > 0 && th > 0) || throw(ArgumentError(
        "the offset m·h and the thickness n_L·h are lengths and must be " *
        "positive, got offset = $off and thickness = $th."))
    (lo - off) - th > 0 || throw(ArgumentError(
        "the core surface r_h(n̂) − offset − thickness reaches the center: " *
        "r_in = $lo, offset = $off, thickness = $th. The core rule projects " *
        "the frozen core onto that surface along the ray, and a surface " *
        "through the center has no inside — refine the mesh, so that the " *
        "offset and the ramp, which are stated in spacings, shrink."))
    (0 < wr ≤ 1 && 0 < ρr ≤ 1) || throw(ArgumentError(
        "the ramp fractions must lie in (0, 1], got w_ramp = $wr and " *
        "ρ_ramp = $ρr."))
    T(ρ_max) ≥ 0 || throw(ArgumentError(
        "the relaxation rate must satisfy ρ_max ≥ 0, got $ρ_max."))
    margin ≥ 1 || throw(ArgumentError(
        "the margin m is a number of grid points and must be at least 1, " *
        "got $margin."))
    check_layer_target(target)
    return FittedInterior{T,variant,(L + 1)^2,typeof(target)}(
        c, sv, L, lo, hi, off, th, T(ρ_max), wr, ρr, Int(margin), Int(n_L),
        T(h), Val(variant), target)
end

with_ρ_max(int::FittedInterior{T,V,NM,X}, ρ_max) where {T,V,NM,X} =
    FittedInterior{T,V,NM,X}(int.center, int.shape, int.lmax, int.r_in,
                             int.r_out, int.offset, int.thickness, T(ρ_max),
                             int.w_ramp, int.ρ_ramp, int.margin, int.n_L,
                             int.h, int.valvariant, int.target)

interior_variant(::FittedInterior{T,V}) where {T,V} = V

@inline layer_target(::FittedInterior{T,V,NM,Nothing}, bg) where {T,V,NM} = bg
@inline layer_target(int::FittedInterior, bg) = int.target

geometry_radii(int::FittedInterior, background) = (int.r_in, int.r_out)

layer_radii(int::FittedInterior) =
    ((int.r_in - int.offset) - int.thickness, int.r_in - int.offset)

# The surface: the series clamped into its bounding shell.
@inline _surface_radius(shape, lmax, r_in, r_out, n) =
    clamp(shape_series(shape, lmax, n), r_in, r_out)

"""
    shape_radius(int::FittedInterior, n̂) -> r_h

The tracked horizon's coordinate radius along the unit vector `n̂` about the
interior's center: [`shape_series`](@ref) clamped into `[r_in, r_out]`,
which is what makes the geometry's fast paths exact.
"""
@inline shape_radius(int::FittedInterior, n) =
    _surface_radius(int.shape, int.lmax, int.r_in, int.r_out, n)

@inline function interior_radius(int::FittedInterior{T}, t, x) where {T}
    c = center_at(int.center, t)
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    return sqrt(d1 * d1 + d2 * d2 + d3 * d3)
end

"""
    interior_point(int::FittedInterior, t, x) -> (; r, r_1, r_0)

The kernel's view of the tracked geometry at one point: the distance `r`
from the tracked center and the radii of the offset surface and the core
surface **along the ray through `x`**, `r_1 = r_h(n̂) − offset` and
`r_0 = r_1 − thickness` — so that the depth is `d = r_1 − r` and the
predicates and the profiles are the sphere's, in the ray's radii.

**Two fast paths, both exact** (proposed in step 8d). Outside the offset
surface's bounding sphere, `r ≥ r_out − offset`, the point is evolved
whatever the direction; inside the core surface's, `r < r_in − offset −
thickness`, it is frozen. Neither evaluates the series, and because the
surface is *defined* as the series clamped into `[r_in, r_out]`, both give
the classification the full evaluation would. On a fast path `r_1` and `r_0`
are those bounding spheres' and not the ray's, which no caller of the
classification reads; [`fitted_geometry`](@ref) is the full evaluation for a
caller that wants the depth itself.
"""
@inline function interior_point(int::FittedInterior{T}, t, x) where {T}
    c = center_at(int.center, t)
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    r = sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    r1_out = int.r_out - int.offset
    if r ≥ r1_out
        return (r=r, r_1=r1_out, r_0=r1_out - int.thickness)
    end
    r1_in = int.r_in - int.offset
    r0_in = r1_in - int.thickness
    if r < r0_in
        return (r=r, r_1=r1_in, r_0=r0_in)
    end
    n = iszero(r) ? SVector{3,T}(zero(T), zero(T), one(T)) :
        SVector{3,T}(d1 / r, d2 / r, d3 / r)
    r_1 = shape_radius(int, n) - int.offset
    return (r=r, r_1=r_1, r_0=r_1 - int.thickness)
end

"""
    fitted_geometry(int::FittedInterior, t, x) -> (; r, n̂, d, r_1, r_0)

The tracked geometry at one point, **evaluated in full**: the distance `r`
from the tracked center `c(t)`, the unit vector `n̂` from it (`+ẑ` at the
center, the core rule's tie-break), the depth `d = r_h(n̂) − offset − r`
below the offset surface, and the two surfaces' radii along the ray. `d ≤ 0`
is evolved, `0 < d ≤ thickness` is the layer and beyond it the core.
"""
@inline function fitted_geometry(int::FittedInterior{T}, t, x) where {T}
    c = center_at(int.center, t)
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    r = sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    n = iszero(r) ? SVector{3,T}(zero(T), zero(T), one(T)) :
        SVector{3,T}(d1 / r, d2 / r, d3 / r)
    r_1 = shape_radius(int, n) - int.offset
    return (r=r, n=n, d=r_1 - r, r_1=r_1, r_0=r_1 - int.thickness)
end

# The predicates and the profiles in the ray's radii — the sphere's, in the
# sphere's order (`is_frozen` docstring above).
@inline is_frozen(int::FittedInterior{T,:damped}, g) where {T} = g.r < g.r_0
@inline is_frozen(int::FittedInterior{T,:frozen}, g) where {T} = g.r < g.r_0
@inline is_frozen(int::FittedInterior{T,:pasted}, g) where {T} = g.r < g.r_1
# The fitted core (step 8e): `F` is not evaluated below the core surface, and
# the kernel relaxes it toward the target at `ρ_max` instead of freezing it.
@inline is_frozen(int::FittedInterior{T,:fitted}, g) where {T} = g.r < g.r_0

@inline is_outside(int::FittedInterior, g) = g.r ≥ g.r_1

@inline interior_profiles(int::FittedInterior, g) =
    _layer_profiles(g.r, g.r_0, g.r_1, int.ρ_max, int.w_ramp, int.ρ_ramp)

@inline interior_profiles(int::FittedInterior{T,:frozen}, g) where {T} =
    (_layer_w(g.r, g.r_0, g.r_1, int.w_ramp), zero(T))

@inline function in_layer(int::FittedInterior, t, x)
    g = interior_point(int, t, x)
    return (g.r_0 ≤ g.r) & (g.r < g.r_1)
end

"""
    core_position(int::FittedInterior, t, x) -> x

The core rule on the tracked geometry: `x` itself outside the core surface,
and inside it the core surface's own point on the ray, `c(t) + r_0(n̂) n̂`,
with `+ẑ` at the center — the sphere's rule, with the ray's radius in place
of `r_0`. Outside the core surface's bounding sphere the series is not
evaluated.
"""
@inline function core_position(int::FittedInterior{T}, t, x) where {T}
    c = center_at(int.center, t)
    d = SVector{3}(x[1] - c[1], x[2] - c[2], x[3] - c[3])
    r = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    r ≥ (int.r_out - int.offset) - int.thickness && return (x[1], x[2], x[3])
    ẑ = SVector{3}(zero(r), zero(r), one(r))
    n = iszero(r) ? ẑ : d / r
    r_0 = (shape_radius(int, n) - int.offset) - int.thickness
    r ≥ r_0 && return (x[1], x[2], x[3])
    return (c[1] + r_0 * n[1], c[2] + r_0 * n[2], c[3] + r_0 * n[3])
end

# --- the masks of the tracked geometry ----------------------------------------

"""
    ShapeMask(center, shape, lmax, r_in, r_out, offset)

The mask of a [`FittedInterior`](@ref) at one time: [`is_evolved`](@ref)
is `r ≥ r_1(n̂)`, depth `d ≤ 0`, with the classification's two fast paths —
a snapshot of the tracked center, as [`InteriorMask`](@ref) is of the
analytic one. Every masked norm, the indicator, the speed kernel and the
horizon finder's footprint guard take it through
[`interior_mask`](@ref), so they all exclude the same region the kernel
modifies.
"""
struct ShapeMask{T,NM}
    center::SVector{3,T}
    shape::SVector{NM,T}
    lmax::Int
    r_in::T
    r_out::T
    offset::T
end

@inline function is_evolved(m::ShapeMask{T}, x) where {T}
    d1 = x[1] - m.center[1]
    d2 = x[2] - m.center[2]
    d3 = x[3] - m.center[3]
    r = sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    r ≥ m.r_out - m.offset && return true
    r < m.r_in - m.offset && return false
    n = iszero(r) ? SVector{3,T}(zero(T), zero(T), one(T)) :
        SVector{3,T}(d1 / r, d2 / r, d3 / r)
    return r ≥ _surface_radius(m.shape, m.lmax, m.r_in, m.r_out, n) - m.offset
end

"""
    ShapeBand(center, shape, lmax, r_in, r_out, offset, lo, hi)

The band `r_1(n̂) + lo ≤ r < r_1(n̂) + hi` about the offset surface of a
tracked geometry — [`ShellMask`](@ref)'s job for a surface that is not a
sphere: `lo = −thickness, hi = 0` is the layer ([`layer_mask`](@ref)) and
`lo = 0, hi = width` the shell just outside it ([`shell_mask`](@ref)).
"""
struct ShapeBand{T,NM}
    center::SVector{3,T}
    shape::SVector{NM,T}
    lmax::Int
    r_in::T
    r_out::T
    offset::T
    lo::T
    hi::T
end

@inline function is_evolved(m::ShapeBand{T}, x) where {T}
    d1 = x[1] - m.center[1]
    d2 = x[2] - m.center[2]
    d3 = x[3] - m.center[3]
    r = sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    r ≥ (m.r_out - m.offset) + m.hi && return false
    r < (m.r_in - m.offset) + m.lo && return false
    n = iszero(r) ? SVector{3,T}(zero(T), zero(T), one(T)) :
        SVector{3,T}(d1 / r, d2 / r, d3 / r)
    r_1 = _surface_radius(m.shape, m.lmax, m.r_in, m.r_out, n) - m.offset
    return (r_1 + m.lo ≤ r) & (r < r_1 + m.hi)
end

interior_mask(int::FittedInterior{T,V,NM}, t) where {T,V,NM} =
    ShapeMask{T,NM}(center_at(int.center, t), int.shape, int.lmax, int.r_in,
                    int.r_out, int.offset)

layer_mask(int::FittedInterior{T,V,NM}, t) where {T,V,NM} =
    ShapeBand{T,NM}(center_at(int.center, t), int.shape, int.lmax, int.r_in,
                    int.r_out, int.offset, -int.thickness, zero(T))

shell_mask(int::FittedInterior{T,V,NM}, t, width) where {T,V,NM} =
    ShapeBand{T,NM}(center_at(int.center, t), int.shape, int.lmax, int.r_in,
                    int.r_out, int.offset, zero(T), T(width))

# --- where the tracked layer was put, against the mesh ------------------------

_box_meets_annulus(ext, c, lo, hi) =
    ((near, far) = _box_radii(ext, c); near ≤ hi && far ≥ lo)

"""
    geometry_spacing(forest, int::FittedInterior, t) -> (h, nblocks)

The coarsest spacing among the blocks whose extent meets the layer's
annulus `[r_in − offset − thickness, r_out − offset]` about the tracked
center — the blocks the profiles live in, which is [`layer_spacing`](@ref)'s
"the blocks containing `r_1`" for a surface that is not a sphere (added in
step 8d).
"""
function geometry_spacing(forest::Forest{3}, int::FittedInterior{T}, t) where {T}
    c = center_at(int.center, t)
    lo = (int.r_in - int.offset) - int.thickness
    hi = int.r_out - int.offset
    h = zero(T)
    n = 0
    for k in forest.leaves
        _box_meets_annulus(block_extent(T, forest, k), c, lo, hi) || continue
        h = max(h, spacing(T, forest, k))
        n += 1
    end
    return h, n
end

layer_spacing(forest::Forest{3}, int::FittedInterior, t) =
    geometry_spacing(forest, int, t)

"""
    check_interior_radii(forest, int::FittedInterior, background, G;
                         t = 0, center = nothing)

`CODE.md`'s placement requirements on the **tracked** geometry, at the mesh
`forest` currently is (step 8d): with `h` the coarsest spacing of the
blocks the layer lives in ([`geometry_spacing`](@ref)),

    offset ≥ m h            (r_1(n̂) ≤ r_h(n̂) − m h in every direction)
    thickness ≥ 2(G + 1) h
    r_in − offset − thickness > singular_radius(background) + |c(t) − c_analytic(t)|

(The third is skipped for step 8e's `:fitted`, which evaluates no analytic
interior; its `singular` field is then `−1`.)

The first two are step 5's two requirements with the **track's** radii in
place of `horizon_min_radius(background)`: the horizon the layer is put
inside is the one that was found, in every direction, and the margin is
counted in the direction's own radius rather than the smallest one. The
third keeps step 5's `singular_radius` check for the analytic-target
variants, which is every variant this geometry runs until step 8e's
`:fitted`: the core rule still evaluates the analytic solution on the core
surface, so that surface must still contain the chart's singular set — a
ball of `singular_radius` about the **analytic** center, which is why the
distance between the two centers (`center`, the case's analytic
trajectory, where it is given) is added to it. For a tracked geometry it
is the smallest core radius that has to clear the set, since the core
surface is not a sphere.

It throws, and it is meant to, with the remedies the sphere's check names.
"""
function check_interior_radii(forest::Forest{3}, int::FittedInterior{T},
                              background, G::Integer; t=zero(T),
                              center=nothing) where {T}
    int.margin ≥ G + 1 || throw(ArgumentError(
        "the interior's margin is m = $(int.margin) but the ghost width is " *
        "G = $G, and CODE.md's floor is m ≥ G + 1 = $(G + 1): the margin has " *
        "to cover a whole stencil, or a point outside the horizon reaches " *
        "into the layer."))
    h, nb = geometry_spacing(forest, int, t)
    nb > 0 || throw(ArgumentError(
        "no block of this forest meets the tracked layer's annulus around " *
        "$(center_at(int.center, t)) at t = $t: the layer is outside the " *
        "domain."))
    r_core = (int.r_in - int.offset) - int.thickness
    δ = if center === nothing
        zero(T)
    else
        a = center_at(center, T(t))
        b = center_at(int.center, T(t))
        sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2 + (a[3] - b[3])^2)
    end
    # The `:fitted` variant evaluates no analytic interior — its initial data
    # and its target are the fit's inside the offset surface (step 8e) — so
    # the chart's singular set may lie anywhere inside that surface, which is
    # the whole point of it; the check is the analytic variants'.
    r_sing = interior_variant(int) === :fitted ? -one(T) :
             T(singular_radius(background)) + δ
    r_core > r_sing || throw(ArgumentError(
        "the tracked core surface does not contain the chart's singular set: " *
        "its smallest radius is r_in − offset − thickness = $r_core about the " *
        "tracked center, and $(typeof(background)) is singular out to a " *
        "coordinate radius of $(r_sing - δ) about the analytic one, $δ away " *
        "— for Kerr the equatorial disk |x| ≤ |a|, z = 0. The core rule still " *
        "evaluates the analytic solution on that surface, so a grid point on " *
        "the disk would be NaN. Refine (the offset and the ramp are stated in " *
        "spacings), or wait for step 8e's :fitted target, which needs no " *
        "analytic interior (CODE.md, \"The interior\")."))
    allowed = int.margin * h
    int.offset ≥ allowed || throw(ArgumentError(
        "the tracked layer is not far enough inside the found horizon: its " *
        "offset is $(int.offset), but the blocks it lives in have h = $h and " *
        "the margin m = $(int.margin) asks for m·h = $allowed — the geometry " *
        "was built on a finer mesh than this one. The horizon and m grid " *
        "points inside it must be evolved by the unmodified equations: " *
        "rebuild the geometry on this mesh (fitted_interior), or keep the " *
        "refinement's level floor around the horizon."))
    needed = 2 * (G + 1) * h
    int.thickness ≥ needed || throw(ArgumentError(
        "the tracked layer is too thin: its thickness is $(int.thickness) at " *
        "a spacing of h = $h, and CODE.md asks for at least 2(G+1)h = " *
        "$needed so that the profiles are resolved and no stencil of an " *
        "evolved point reaches the core."))
    return (h=h, nblocks=nb, r_h_min=int.r_in, allowed=int.r_in - allowed,
            thickness=int.thickness, needed=needed, r_core=r_core,
            singular=r_sing)
end
