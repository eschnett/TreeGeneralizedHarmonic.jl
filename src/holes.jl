# Several holes in one case (step B3): the hole as a value of its own, the data
# a binary is started from, and the tuple of interiors the kernel composes.
#
# `CODE.md`, "Binary black holes": "A case without a background", "One interior
# per hole" and "Initial data" (item 1). A single-hole case is built around one
# analytic background, which supplies the initial data, the Dirichlet data, the
# gauge source, the layer's target, the error reference, the center and the
# radii; a binary has no analytic solution, so the background is demoted to
# **data** — read at `t = 0` and by the boundary hook, never as the truth — and
# each of the others comes from the holes (decided by Erik 2026-10-26: one
# `GHCase` with `holes`, no `BinaryCase`; the single hole is a binary with one
# hole). This file holds what needs no track:
#
#   1. **[`HoleSpec`](@ref)** — a hole's mass, spin, initial center, velocity
#      and acceleration, and its interior's rule (step B2's `TrackedSpec`);
#   2. **[`SuperposedMetric`](@ref)** — `g = η + Σ_A w_A (g_A − η)`, `isbits`,
#      with XCTS's optional attenuation `w_A = exp(−|x − c_A|²/ℓ²)`: the data,
#      written here (decided by Erik 2026-10-26, not in SpacetimeMetrics);
#   3. **[`HolesSpec`](@ref)** — what a case of `K ≥ 2` holes holds as its
#      `interior`: the rule the composite is built by, per piece, from the
#      tracks (`tracking.jl`'s [`holes_interior`](@ref));
#   4. **[`HoleInteriors`](@ref)** — the kernel argument: the `K` holes'
#      interiors, composed point by point by [`acting_interior`](@ref), and the
#      composite masks every norm, the indicator, the projection's gate and the
#      finder's guard take;
#   5. the per-hole profiles: `γ0` as three Gaussians ([`HolesDamping`](@ref))
#      and `ε_KO` per hole ([`HolesDissipation`](@ref)).
#
# Two things are easy to get wrong, and each is written where it happens:
#
#   * **The composition is a selection, and it is the rule.** `CODE.md` composes
#     `w = min_A w_A`, `ρ = max_A ρ_A`, the union of the cores and the target of
#     the hole whose `ρ_A` is largest. Each profile is a monotone function of the
#     hole's own layer coordinate `s_A = (r_A − r_0,A)/(r_1,A − r_0,A)`, so with
#     the holes' ramps alike the hole of the smallest `s_A` has the smallest `w`,
#     the largest `ρ` and, below `s = 0`, the core: the rule is "run the hole of
#     the smallest `s_A`" — one hole's arithmetic at every point, which is the
#     single-hole kernel of that hole bit for bit wherever only one layer acts
#     **(proposed in step B3)**. Before merger the layers are disjoint, and the
#     rule decides nothing.
#   * **No closure, and no runtime index, in the kernel's path.** `K` is a type
#     parameter and every sum or selection over the holes is unrolled by a
#     generated body (`CLAUDE.md`, "On a device a closure is a call"), with
#     `ifelse` choosing between whole interiors rather than an index into the
#     tuple, which a device would put in local memory.

# --- the hole -------------------------------------------------------------------

"""
    HoleSpec(T = Float64; mass, spin = (0, 0, 0), center, velocity = (0, 0, 0),
             acceleration = (0, 0, 0), interior)

One hole of a [`GHCase`](@ref)'s `holes` (added in step B3; `CODE.md`, "A case
without a background"): its mass `M_A`, its spin `a⃗ = J⃗/M` (a length, as
[`KerrSchildSource`](@ref) takes it), its center, velocity and acceleration at
`t = 0`, and the rule its interior is built by — step B2's
[`TrackedSpec`](@ref), whose radii, rate and interval are stated in the case's
units (a hole of mass `M_A` takes the single hole's numbers times `M_A`,
[`scaled_spec`](@ref)).

- **The acceleration is the seed's** (step B2's hand-over): a binary's track
  must know its orbit's acceleration from the start, or the first find of a
  tight orbit is refused as a center jump. A head-on pair from rest has
  `M_B/d²` toward the companion ([`head_on_holes`](@ref) puts it there).
- **The spin is along `z` (proposed in step B3).** The tracked target is an
  unrotated Kerr-Schild hole boosted along the track, which spins about `z`, and
  the gauge source holds the spin fixed in the rest frame; a tilted spin is
  refused by name. The binaries of steps B3–B9 are non-spinning or aligned.
- **The interior is a `TrackedSpec` (decided by Erik 2026-10-26, pairing (α)).**
  A `FittedSpec` per hole is the alternative the composition is written to admit
  ([`HoleInteriors`](@ref)) and is refused here: it is not built.

`isbits`, as everything a case holds is.
"""
struct HoleSpec{T,I}
    mass::T
    spin::SVector{3,T}
    center::SVector{3,T}
    velocity::SVector{3,T}
    acceleration::SVector{3,T}
    interior::I
end

function HoleSpec(::Type{T}=Float64; mass, spin=(0, 0, 0), center, velocity=(0, 0, 0),
                  acceleration=(0, 0, 0), interior) where {T}
    vec3(x, what) = (length(x) == 3 || throw(ArgumentError(
        "a hole's $what is a 3-vector, got $(repr(x)).")); SVector{3,T}(T(x[1]), T(x[2]),
                                                                         T(x[3])))
    M = T(mass)
    M > 0 || throw(ArgumentError("a hole's mass must be positive, got $mass."))
    s = vec3(spin, "spin")
    (iszero(s[1]) && iszero(s[2])) || throw(ArgumentError(
        "a hole's spin is along z in step B3, got $(Tuple(s)): the tracked target is " *
        "an unrotated Kerr-Schild hole boosted along the track, which spins about z, " *
        "and the binaries of steps B3–B9 are non-spinning or aligned with z."))
    abs(s[3]) < M || throw(ArgumentError(
        "a hole's spin a = J/M must satisfy |a| < M, got a = $(s[3]) at M = $M."))
    v = vec3(velocity, "velocity")
    sum(abs2, v) < 1 || throw(ArgumentError(
        "a hole moves slower than light, got |v| = $(sqrt(sum(abs2, v)))."))
    interior isa TrackedSpec{T} || throw(ArgumentError(
        "a hole's interior is a TrackedSpec{$T} — step B2's tracked :damped target, " *
        "pairing (α) (decided by Erik 2026-10-26) — got a $(typeof(interior)). A " *
        ":fitted hole is what the composition is written to admit, and is not built."))
    return HoleSpec{T,typeof(interior)}(M, s, vec3(center, "center"), v,
                                        vec3(acceleration, "acceleration"), interior)
end

"""
    scaled_spec(spec::TrackedSpec, M) -> TrackedSpec

The tracked target's rule for a hole of mass `M` from the rule stated for a hole
of mass one (added in step B3; `CODE.md`, "Binary black holes", units): the radii,
the find interval, the target's center offset and the source line's control time
times `M`, a stated rate over `M`; the margin, the ramps, the shape's degree, the
velocities and the feed unchanged.
"""
function scaled_spec(spec::TrackedSpec{T}, M) where {T}
    m = T(M)
    return TrackedSpec(T; r_0=spec.r_0 * m, r_1=spec.r_1 * m, w_ramp=spec.w_ramp,
                       ρ_ramp=spec.ρ_ramp, margin=spec.margin, ρ_max=spec.ρ_max / m,
                       max_misses=spec.max_misses, lmax_shape=spec.lmax_shape,
                       quadratic=spec.quadratic, max_interval=spec.max_interval * m,
                       mass_factor=spec.mass_factor,
                       velocity_offset=spec.velocity_offset,
                       center_offset=spec.center_offset * m,
                       seed_velocity_offset=spec.seed_velocity_offset,
                       source_feed=feed_name(spec.source_feed),
                       control_time=spec.control_time * m,
                       control_integral=spec.control_integral,
                       control_model=spec.control_model)
end

"""
    initial_line(spec::HoleSpec) -> HoleCenter

The hole's straight line from its initial center along its initial velocity,
`c₀ + v t` — what the data and the case's profiles are built on before any track
exists (added in step B3).
"""
initial_line(h::HoleSpec{T}) where {T} = HoleCenter{T}(h.center, h.velocity)

"""
    kerr_schild_on_line(T, M, a, line::HoleCenter) -> metric

The boosted, translated Kerr-Schild hole of mass `M` and spin `a` about `z` that
moves along `line`, `c(t) = c₀ + v t`: `translate(boost(KerrSchild(M, a), 0 − v),
(0, c₀))` — `boost(m, u)` moves the hole at `−u` ([`hole_velocity`](@ref)), and
`0 − v` keeps a zero component `+0` (added in step B3; step B2's
[`tracked_interior`](@ref) builds its target the same way).
"""
function kerr_schild_on_line(::Type{T}, M, a, line::HoleCenter{T}) where {T}
    v = line.v
    return SpacetimeMetrics.translate(SpacetimeMetrics.boost(KerrSchild(T(M), T(a)),
                                                             zero(v) - v),
                                      SVector{4,T}(zero(T), line.c0...))
end

"""
    hole_metric(spec::HoleSpec) -> metric

The hole's own boosted, translated Kerr-Schild metric on its initial line
([`kerr_schild_on_line`](@ref)): what its radius checks, its seed and its
single-hole gauge-source check read (added in step B3).
"""
hole_metric(h::HoleSpec{T}) where {T} =
    kerr_schild_on_line(T, h.mass, h.spin[3], initial_line(h))

# --- the data -------------------------------------------------------------------

"""
    SuperposedMetric{T,K,M}(holes, lines, ℓ, λ)
    superposed_kerr_schild(T, holes; attenuation = nothing, companion = nothing)
        -> SuperposedMetric

`CODE.md`'s initial data item 1 (added in step B3): the superposition

    g = η + Σ_A w_A (g_A − η),
    w_A = e^{−|x − c_A(t)|²/ℓ_A²} · Π_{B≠A} (1 − e^{−|x − c_B(t)|²/λ_B²}),

of `K` metrics `g_A` of one type — boosted, translated Kerr-Schild holes on their
lines `c_A(t)` — with two optional attenuations, each off by default:

- **XCTS's own**, `e^{−r_A²/ℓ_A²}` (`ℓ_A = Inf`: none), which turns each hole's
  field off far from it;
- **the companions'**, `1 − e^{−r_B²/λ_B²}` (`λ_B = 0`: none) **(proposed in step
  B3)**, which turns every other hole's field off within about `λ_B` of hole `B`,
  so that each hole's neighbourhood is its own Kerr-Schild hole to
  `O((r_B/λ_B)²)` and the far field is still the plain sum.

**Why the companions' attenuation (found in step B3).** The plain superposition
is not a 3+1 metric inside the horizons: `α² = −g_tt + β_i β^i` turns negative on
the side of each hole that faces its companion, where the two Kerr-Schild shifts
`2H l_i` cancel. For an equal-mass pair of total mass one, over the directions at
each radius from hole 1, the least `α²` is negative out to `r_A = 0.52` at
`d = 8` (`−0.154` on the sphere `r_A = 5/16`, the single hole's core radius
times `M_A`, where Kerr-Schild has `0.238`), to `0.40` at `d = 10` and `0.35` at
`d = 12`, and through the whole horizon at `d = 4`. A layer and a core cannot be
put there. With `λ_B = 4 M_B` the least `α²` is within `0.01` of the single
hole's from the core radius out at `d = 8`, and the companion keeps `98 %` of its
field at the midpoint.

A `SpacetimeMetrics` metric, so that `dmetric` differentiates it through the same
forward-mode pass every background takes, and `isbits`, so that it is a kernel
argument: the initial data, the Dirichlet data and the binary's layer targets are
evaluated from it. The sums and products are unrolled over `K` (no closure on a
device). **Here, not in SpacetimeMetrics (decided by Erik 2026-10-26).** It is not
a solution where the holes' fields overlap or where an attenuation varies: the
constraint damping removes the violation, which radiates as junk — good for the
head-on collision and every development test, not for a waveform (`CODE.md`,
"Initial data").

`superposed_kerr_schild` builds it from a case's [`HoleSpec`](@ref)s, each hole on
its initial line; `attenuation` is `nothing`, one length for every hole or one per
hole, and `companion` is `nothing` or `λ_B / M_B`, one number or one per hole.
"""
struct SuperposedMetric{T,K,M} <: AbstractMetric
    holes::NTuple{K,M}
    lines::NTuple{K,HoleCenter{T}}
    ℓ::NTuple{K,T}
    λ::NTuple{K,T}
end

Base.nameof(m::SuperposedMetric{T,K}) where {T,K} =
    "a superposition of $K metrics (" * join((nameof(h) for h in m.holes), "; ") * ")"

@inline _minkowski4(::Type{E}) where {E} =
    SMatrix{4,4,E}(-one(E), zero(E), zero(E), zero(E), zero(E), one(E), zero(E), zero(E),
                   zero(E), zero(E), one(E), zero(E), zero(E), zero(E), zero(E), one(E))

# `|x − c(t)|²` about a line, at the 4-position `p`: the line moves, so `∂_t` of
# an attenuation is part of `∂_t g`.
@inline function _line_r2(line::HoleCenter, p)
    d1 = p[2] - (line.c0[1] + p[1] * line.v[1])
    d2 = p[3] - (line.c0[2] + p[1] * line.v[2])
    d3 = p[4] - (line.c0[3] + p[1] * line.v[3])
    return d1 * d1 + d2 * d2 + d3 * d3
end

# The own attenuation `e^{−r²/ℓ²}`, exactly one at `ℓ = Inf`, and the companion
# factor `1 − e^{−r²/λ²}`, exactly one at `λ = 0` — each a branch on a number of
# the metric, not on the point.
@inline _own_weight(line, ℓ, p) =
    isinf(ℓ) ? one(eltype(p)) : exp(-_line_r2(line, p) / (ℓ * ℓ))
@inline _companion_weight(line, λ, p) =
    iszero(λ) ? one(eltype(p)) : one(eltype(p)) - exp(-_line_r2(line, p) / (λ * λ))

@generated function SpacetimeMetrics.metric(m::SuperposedMetric{T,K},
                                            x::AbstractVector) where {T,K}
    body = Expr(:block, Expr(:meta, :inline), :(p = SVector{4}(x)),
                :(η4 = _minkowski4(eltype(p))))
    sym(s, A) = Symbol(s, "_", A)
    for A in 1:K
        push!(body.args, :($(sym(:own, A)) = _own_weight(m.lines[$A], m.ℓ[$A], p)))
        push!(body.args, :($(sym(:cmp, A)) = _companion_weight(m.lines[$A], m.λ[$A], p)))
    end
    for A in 1:K
        w = foldl((a, b) -> :($a * $b), [sym(:cmp, B) for B in 1:K if B != A];
                  init=sym(:own, A))
        push!(body.args, :($(sym(:g, A)) =
                               $w * (SpacetimeMetrics.metric(m.holes[$A], p) - η4)))
    end
    sumex = foldl((a, b) -> :($a + $b), [sym(:g, A) for A in 1:K]; init=:η4)
    push!(body.args, :(return $sumex))
    return body
end

function _per_hole(::Type{T}, x, K, default, what) where {T}
    x === nothing && return ntuple(_ -> default, K)
    x isa Real && return ntuple(_ -> T(x), K)
    length(x) == K && return ntuple(A -> T(x[A]), K)
    throw(ArgumentError(
        "$what is nothing, one number for every hole or one per hole ($K), got " *
        "$(repr(x))."))
end

function superposed_kerr_schild(::Type{T}, holes::NTuple{K,HoleSpec{T}};
                                attenuation=nothing, companion=nothing) where {T,K}
    ℓ = _per_hole(T, attenuation, K, T(Inf), "attenuation")
    all(x -> x > 0, ℓ) || throw(ArgumentError(
        "an attenuation length is positive (Inf: none), got $ℓ."))
    c = _per_hole(T, companion, K, zero(T), "companion")
    all(x -> x ≥ 0, c) || throw(ArgumentError(
        "a companion attenuation is λ_B/M_B ≥ 0 (0: none), got $c."))
    λ = ntuple(A -> c[A] * holes[A].mass, K)
    lines = ntuple(A -> initial_line(holes[A]), K)
    ms = ntuple(A -> hole_metric(holes[A]), K)
    return SuperposedMetric{T,K,eltype(ms)}(ms, lines, ℓ, λ)
end

# What the single-hole code asks of a background, answered for a superposition:
# no hole of its own (each hole's numbers are its `HoleSpec`'s), not harmonic.
isharmonic(::SuperposedMetric) = false
has_hole_velocity(::SuperposedMetric) = false
horizon_min_radius(m::SuperposedMetric) = throw(ArgumentError(
    "a superposition of holes has no horizon of its own: each hole's radii are " *
    "its own metric's (hole_metric), and a binary's checks are made per hole."))
hole_mass(m::SuperposedMetric) = throw(ArgumentError(
    "a superposition of holes has no single mass: read each hole's own " *
    "(HoleSpec.mass) — hole_mass(case.background) is the wrong mass for a binary " *
    "(PLAN.md, \"Sharp edges for steps B1–B9\")."))

# --- the case's rule -------------------------------------------------------------

"""
    HolesSpec{K,S}(specs, superposed, feed, gauge_momentum)

What a [`GHCase`](@ref) of `K ≥ 2` holes holds as its `interior` (added in step
B3): not a layer but the rule the composite is built by, once per piece of a
chunk, from the holes' tracks ([`holes_interior`](@ref)) — each hole's step-B2
[`TrackedSpec`](@ref), and where its target comes from:

- `superposed = true` **(the default, proposed in step B3)**: each layer
  relaxes toward the superposition of all `K` holes on their tracks — the data's
  own form, which near hole `A` is `A`'s Kerr-Schild hole plus its companions'
  fields;
- `superposed = false`: toward hole `A`'s own boosted Kerr-Schild hole on its
  track, step B2's target as written.

**Why the superposition by default.** The superposed data put the companion's
field into hole `A`'s layer — `2 M_B/d`, `0.125` in `h` at `d = 8 M` — and a
target without it is wrong there by that much, six times what a target `1 %`
off in mass is (`2M · 1 %/r` near the horizon), which step B2 measured at
`7.5×` the exact target's `ℋ` just outside the horizon. `CODE.md`'s "it misses
the companion's tidal field, `M_B r²/d³`" holds for the companion's *gradient*
across the layer, not for its potential, which the plain superposition carries
into every hole's layer (the companion attenuation of [`SuperposedMetric`](@ref)
removes most of it).

Two more switches, each a choice of the step-B3 screen:

- `feed`, the code of the case's `source_feed`: `FEED_TRACKS` (Erik's pairing
  (α), decided 2026-10-26), the composite gauge source's holes on the tracks
  every piece ([`holes_source_at`](@ref)); `FEED_SEED`, on their seeds'
  quadratics — the `HoleSpec`'s initial center, velocity and acceleration,
  Newtonian free fall from rest for a head-on pair — with the tracks feeding the
  targets alone, as step B2's single hole's analytic source did; or (added in
  step B3, Erik's decision of 2026-10-26 on `CODE.md`'s item 7)
  `FEED_CONTROLLED`, on the controlled source lines ([`SourceLine`](@ref)),
  each with its own spec's `control_time`.
- `gauge_momentum = false`: the initial `Π` is the superposition's own, from its
  `∂_t g`; `true`: from `C_a = 0` at `t = 0` under the case's source, `∂_t g_tb`
  solved for as [`gauge_consistent_momentum`](@ref) does
  ([`gauge_consistent_state`](@ref); `CODE.md`, "Initial data", item 1 names
  both).

`isbits`; `K` is a type parameter.
"""
struct HolesSpec{K,S}
    specs::NTuple{K,S}
    superposed::Bool
    feed::Int
    gauge_momentum::Bool
end

interior_variant(::HolesSpec) = :damped

# --- the kernel argument ------------------------------------------------------------

"""
    HoleInteriors(holes, own)

The interior of a case of `K ≥ 2` holes at one piece (added in step B3; `CODE.md`,
"One interior per hole"): `holes`, the `K` holes' interiors — step B2's tracked
`:damped` spheres, one type — and `own`, each hole's own boosted Kerr-Schild
metric on its track's line, which the radius checks and the level floor read and
no kernel does.

The kernels compose it point by point through [`acting_interior`](@ref): the hole
deepest in its own layer runs its single-hole arithmetic, which is `CODE.md`'s
`w = min_A w_A`, `ρ = max_A ρ_A`, the union of the cores and the target of the
largest `ρ_A` (see the file's header). Everything that asks whether a point is
evolved takes the composite mask ([`interior_mask`](@ref)): evolved outside
every hole's layer.

**What admits a `:fitted` hole later.** The protocol a hole is asked through —
its layer coordinate, its masks, its checks — is the one `Interior` and
`FittedInterior` already share; a tuple of different types would need the
selection generated over the types as well, and nothing else.

`isbits`, and a kernel argument at every evaluation; rebuilt per piece.
"""
struct HoleInteriors{K,I,M}
    holes::NTuple{K,I}
    own::NTuple{K,M}
end

interior_variant(c::HoleInteriors) = interior_variant(c.holes[1])

"""
    layer_coordinate(interior, t, x) -> s

Where the point is in the hole's own layer: `s = (r − r_0)/(r_1 − r_0)`, `0` at the
core surface and `1` at the layer's outer surface — the radii along the ray for a
tracked geometry. Every profile of the layer is a monotone function of `s`, which
is what makes [`acting_interior`](@ref)'s selection the composition rule (added in
step B3).
"""
@inline layer_coordinate(int::Interior, t, x) =
    (interior_radius(int, t, x) - int.r_0) / (int.r_1 - int.r_0)

@inline function layer_coordinate(int::FittedInterior, t, x)
    g = interior_point(int, t, x)
    return (g.r - g.r_0) / (g.r_1 - g.r_0)
end

"""
    acting_interior(interior, t, x) -> interior

The one hole's interior that acts at `(t, x)` (added in step B3): for a
[`HoleInteriors`](@ref), the hole of the smallest [`layer_coordinate`](@ref) — the
first of them on a tie — chosen by `ifelse` over whole interiors, unrolled over
`K`; any other interior is its own. The right-hand side, the core rule, the
residual and the paste all run on what it returns, so that a point in one hole's
layer sees that hole's single-hole arithmetic and a point outside every layer sees
`F`.
"""
@inline acting_interior(int, t, x) = int

@generated function acting_interior(c::HoleInteriors{K}, t, x) where {K}
    body = Expr(:block, Expr(:meta, :inline), :(best_1 = c.holes[1]),
                :(s_1 = layer_coordinate(best_1, t, x)))
    for A in 2:K
        b0, s0 = Symbol(:best_, A - 1), Symbol(:s_, A - 1)
        b1, s1, sA, take = Symbol(:best_, A), Symbol(:s_, A), Symbol(:sA_, A),
                           Symbol(:take_, A)
        push!(body.args, :($sA = layer_coordinate(c.holes[$A], t, x)))
        push!(body.args, :($take = $sA < $s0))
        push!(body.args, :($b1 = ifelse($take, c.holes[$A], $b0)))
        push!(body.args, :($s1 = ifelse($take, $sA, $s0)))
    end
    push!(body.args, Symbol(:best_, K))
    return body
end

"""
    outside_all(interior, t, x) -> Bool

Whether the point is evolved by the unmodified equations: outside the layer of the
interior, or of every hole of a [`HoleInteriors`](@ref) (added in step B3). The
SIMD lanes take a group on lanes when every one of its points says so.
"""
@inline outside_all(int, t, x) = is_outside(int, interior_point(int, t, x))

@generated function outside_all(c::HoleInteriors{K}, t, x) where {K}
    ex = :(outside_all(c.holes[1], t, x))
    for A in 2:K
        ex = :($ex & outside_all(c.holes[$A], t, x))
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end

# The distance to the nearest hole's center: what the range projection's gate and
# the error kernel's band ask of a composite.
@generated function interior_radius(c::HoleInteriors{K}, t, x) where {K}
    ex = :(interior_radius(c.holes[1], t, x))
    for A in 2:K
        ex = :(min($ex, interior_radius(c.holes[$A], t, x)))
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end

@inline in_layer(c::HoleInteriors, t, x) = in_layer(acting_interior(c, t, x), t, x)
@inline core_position(c::HoleInteriors, t, x) =
    core_position(acting_interior(c, t, x), t, x)

with_ρ_max(c::HoleInteriors{K}, ρ_max) where {K} =
    HoleInteriors(ntuple(A -> with_ρ_max(c.holes[A], ρ_max), Val(K)), c.own)

core_follows(c::HoleInteriors) = any(core_follows, c.holes)

# --- the composite masks ---------------------------------------------------------------

"""
    HolesMask(masks)
    AnyMask(masks)

The masks of a [`HoleInteriors`](@ref) at one time (added in step B3): `HolesMask`
evolves a point that every hole's mask evolves — the composite mask, outside every
layer — and `AnyMask` counts a point that any hole's band counts, which is how the
layer and the shell of `K` holes are read. `isbits`, unrolled over the holes.
"""
struct HolesMask{K,M}
    masks::NTuple{K,M}
end

struct AnyMask{K,M}
    masks::NTuple{K,M}
end

@inline _all_evolved(::Tuple{}, x) = true
@inline _all_evolved(ms::Tuple, x) = is_evolved(first(ms), x) & _all_evolved(Base.tail(ms), x)
@inline _any_evolved(::Tuple{}, x) = false
@inline _any_evolved(ms::Tuple, x) = is_evolved(first(ms), x) | _any_evolved(Base.tail(ms), x)

@inline is_evolved(m::HolesMask, x) = _all_evolved(m.masks, x)
@inline is_evolved(m::AnyMask, x) = _any_evolved(m.masks, x)

interior_mask(c::HoleInteriors{K}, t) where {K} =
    HolesMask(ntuple(A -> interior_mask(c.holes[A], t), Val(K)))
layer_mask(c::HoleInteriors{K}, t) where {K} =
    AnyMask(ntuple(A -> layer_mask(c.holes[A], t), Val(K)))
shell_mask(c::HoleInteriors{K}, t, width) where {K} =
    AnyMask(ntuple(A -> shell_mask(c.holes[A], t, width), Val(K)))

# --- the checks, per hole ------------------------------------------------------------

"""
    check_interior_radii(forest, c::HoleInteriors, background, G; t = 0, center = nothing)

`CODE.md`'s placement requirements per hole (added in step B3): each hole's layer
against its own horizon at its current speed — its own boosted Kerr-Schild metric
on its track's line (`c.own`), the radii [`horizon_min_radius`](@ref) gives, exact
across the spin — and its own singular set. The data `background` is not read: a
superposition has no horizon of its own. Returns the per-hole results.
"""
function check_interior_radii(forest::Forest{3}, c::HoleInteriors{K}, background,
                              G::Integer; t=0, center=nothing) where {K}
    return ntuple(A -> check_interior_radii(forest, c.holes[A], c.own[A], G; t=t),
                  Val(K))
end

# The range projection (step 8b) of several holes: gated at `r_gate` about the
# nearest hole's center (`interior_radius`), and the gate checked against every
# hole's layer.
function check_bounds_gate(forest::Forest{3}, c::HoleInteriors{K},
                           bounds::StateBounds, q::Integer; t=0) where {K}
    return ntuple(A -> check_bounds_gate(forest, c.holes[A], bounds, q; t=t), Val(K))
end

default_gate(c::HoleInteriors, forest, q::Integer; t=0) =
    minimum(h -> default_gate(h, forest, q; t=t), c.holes)

_apply_bounds!(bd::StateBounds, int::HoleInteriors, p, u, t) =
    _apply_bounds_kernel!(bd, int, p, u, t)

geometry_spacing(forest::Forest{3}, c::HoleInteriors{K}, t) where {K} =
    _spacing_union(ntuple(A -> layer_spacing(forest, c.holes[A], t), Val(K)))

layer_spacing(forest::Forest{3}, c::HoleInteriors, t) = geometry_spacing(forest, c, t)

_spacing_union(r) = (maximum(first, r), sum(last, r))

# --- the profiles -------------------------------------------------------------------

"""
    HolesDamping(T = Float64; holes, remnant, far)

`γ0` for several holes: SpEC's three Gaussians **(proposed in `CODE.md`,
built in step B3)** — one on each hole's track, at the single hole's `1/M_A` with
width `3 M_A`, one on the center of mass of width `d` for the remnant, and `far`
everywhere —

    γ0 = far + Σ_A (near_A − far) e^{−r_A²/2w_A²} + (near_r − far) e^{−r_c²/2w_r²} ,

a sum so that the profile is `C^∞`. `holes` and `remnant` are
[`GaussianDamping`](@ref)s whose own `far` is this one's; [`holes_damping`](@ref)
builds the default, and the driver moves the holes' Gaussians onto their tracks
every piece. `isbits`.
"""
struct HolesDamping{T,K}
    holes::NTuple{K,GaussianDamping{T}}
    remnant::GaussianDamping{T}
    far::T
end

@inline function _gauss_excess(d::GaussianDamping{T}, t, x) where {T}
    c = center_at(d.center, t)
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    r² = d1 * d1 + d2 * d2 + d3 * d3
    return (d.near - d.far) * exp(-r² / (2 * d.width * d.width))
end

@generated function damping_rate(d::HolesDamping{T,K}, t, x) where {T,K}
    ex = :(d.far + _gauss_excess(d.holes[1], t, x))
    for A in 2:K
        ex = :($ex + _gauss_excess(d.holes[$A], t, x))
    end
    ex = :($ex + _gauss_excess(d.remnant, t, x))
    return Expr(:block, Expr(:meta, :inline), ex)
end

function damping_bounds(d::HolesDamping)
    ex = [h.near - d.far for h in (d.holes..., d.remnant)]
    return (d.far + sum(min.(ex, 0)), d.far + sum(max.(ex, 0)))
end

"""
    holes_damping(T, holes; far = 1/(10 M), remnant = 1/M) -> HolesDamping

The default `γ0` of a case of several holes (proposed in step B3): each hole's
Gaussian at `1/M_A` of width `3 M_A` on its initial line, and the remnant's at
`remnant` on the center of mass, of width the holes' largest initial distance from
it times two — the separation `d` for a pair — with `M = Σ M_A`. At `d = 8 M` the
remnant's Gaussian adds `0.79/M` at each hole of an equal-mass pair, to the hole's
own `2/M`.
"""
function holes_damping(::Type{T}, holes::NTuple{K,HoleSpec{T}}; far=nothing,
                       remnant=nothing) where {T,K}
    M = sum(h -> h.mass, holes)
    f = far === nothing ? one(T) / (10 * M) : T(far)
    nr = remnant === nothing ? one(T) / M : T(remnant)
    com = holes_center(holes)
    d = 2 * maximum(h -> sqrt(sum(abs2, h.center - center_at(com, zero(T)))), holes)
    gs = ntuple(A -> GaussianDamping(T; near=1 / holes[A].mass, far=f,
                                     width=3 * holes[A].mass, center=initial_line(holes[A])),
                Val(K))
    rem = GaussianDamping(T; near=nr, far=f, width=d, center=com)
    return HolesDamping{T,K}(gs, rem, f)
end

"""
    holes_center(holes) -> HoleCenter

The holes' center of mass and its velocity, mass-weighted: what a case of several
holes carries as its `center` (proposed in step B3) — the remnant's Gaussian, the
refinement centroid's reference and a hierarchy's default center.
"""
function holes_center(holes::NTuple{K,HoleSpec{T}}) where {K,T}
    M = sum(h -> h.mass, holes)
    c = sum(h -> h.mass * h.center, holes) / M
    v = sum(h -> h.mass * h.velocity, holes) / M
    return HoleCenter{T}(c, v)
end

"""
    HolesDissipation(profiles)

`ε_KO` per hole (added in step B3): the largest of the holes'
[`HorizonDissipation`](@ref) profiles, each `ε_out` outside its own horizon, so
that where the holes' horizons are apart the profile is each hole's own exactly.
`isbits`.
"""
struct HolesDissipation{T,K}
    profiles::NTuple{K,HorizonDissipation{T}}
end

@generated function dissipation_rate(d::HolesDissipation{T,K}, t, x) where {T,K}
    ex = :(dissipation_rate(d.profiles[1], t, x))
    for A in 2:K
        ex = :(max($ex, dissipation_rate(d.profiles[$A], t, x)))
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end

has_dissipation(d::HolesDissipation) = any(has_dissipation, d.profiles)
dissipation_bounds(d::HolesDissipation) =
    (minimum(p -> first(dissipation_bounds(p)), d.profiles),
     maximum(p -> last(dissipation_bounds(p)), d.profiles))
