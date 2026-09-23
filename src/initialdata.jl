# The backgrounds, the case they are wrapped in, and the one place where
# `SpacetimeMetrics`' derivative index convention is converted into
# GHSO2's.
#
# `CODE.md`, "Initial data and backgrounds": every case is a **background
# plus parameters**, and the same object supplies the initial data, the
# Dirichlet data, the error reference, the gauge source where there is
# one, and — from step 5 — the interior's `u_exact` and the hole's
# analytic trajectory. There is no separate "initial data" object and
# never a stored copy of the exact solution: it is a function of `(t, x)`
# and is evaluated where it is needed.
#
# **The conversion.** `SpacetimeMetrics.dmetric` returns
# `dg[a, b, c] = ∂_c g_ab`, the derivative axis *last*; GHSO2's pointwise
# algebra, which `pointwise.jl` ports, uses `dg[a, b, c] = ∂_a g_bc`, the
# derivative axis *first*. `CLAUDE.md` says what mixing them costs: a
# solution that looks right in Minkowski and is wrong everywhere else.
# [`background_state`](@ref) is the one function that converts, and
# `test/initialdata_tests.jl` checks it against an independent spelling
# rather than against itself.
#
# **`Π` is not `∂_t h`.** The evolved momentum is the densitised,
# Lie-advected derivative along the normal,
# `Π_ab = (√γ/α)(∂_t − β^i ∂_i) g_ab`. The first evolution equation,
# `∂_t h_ab = β^i ∂_i h_ab + (α/√γ) Π_ab`, is what a state built from
# `∂_t g` alone would silently be; the two agree only where the shift
# vanishes and `α = √γ`.

"""
    GHCase(T = Float64, background; box, periodic, ε_KO, γ0, γ2,
           center = (0, 0, 0), velocity = the background's,
           interior = nothing,
           r_0 = 0, r_1 = 0, margin = 8, w_ramp = 1//2, ρ_ramp = 1//2,
           target = nothing, refinement = nothing, horizon = nothing,
           bounds = nothing, chunk = 0)

A case: the background, the box it is evolved in, the hole's analytic
trajectory and its damping layer, and the parameters of the scheme that
are properties of the physics rather than of the mesh.

`CODE.md`'s table under "Initial data and backgrounds" is the list of
backgrounds; `box` is the domain's `(lo, hi)` per dimension and `periodic`
says which dimensions close on themselves (the others take the Dirichlet
hook of [`dirichlet`](@ref)). `ε_KO` is the Kreiss–Oliger amplitude and
`γ2 > −1` the Gundlach–Pretorius trace parameter. Neither has a default:
each is a number a run is judged by, and `CODE.md` records `ε_KO ≈ 0.5`
and `γ0 ≈ 1/M` as GHSO2's *recipe near a hole*, not as something a
flat-space test should inherit silently.

**`γ0` is a profile, not a number (added in step 5).** `CODE.md`, "Gauge
and constraint damping", makes the constraint-damping rate a function of
position — a Gaussian around the hole's center, tapered in the wave zone
— so this keyword takes a [`ConstantDamping`](@ref) or a
[`GaussianDamping`](@ref); a bare number is wrapped in the first, which is
what every case without a hole passes and what steps 3 and 4 already do.

**The interior (added in step 5).** `interior` is `nothing` — no hole,
which is every case up to step 4 — or one of `CODE.md`'s three variants
`:damped`, `:pasted`, `:frozen`, in which case `r_0` and `r_1` are the
frozen core's and the layer's radii and an [`Interior`](@ref) is built
around this case's `center` and `velocity`. The layer's `ρ_max` is left at
zero here and set by the driver once per chunk — by default to `4/M`, read
from the background's mass (`CODE.md`, "The profiles and their
parameters"; decided 2026-09-23, step 8c′), or to the grid rate `1/dt` or a
fixed rate when [`evolve!`](@ref) is asked for one — because which rate a
run relaxes at is a choice about the run, and the case is the hole.

**Or a tracked layer (added in step 8d).** `interior` may instead be a
[`FittedSpec`](@ref): the rule a layer is built by, once per chunk, from the
horizon that was found — `evolve!` seeds a [`HorizonTrack`](@ref) from this
case's analytic answer, updates it from each find, and runs each chunk on
the [`FittedInterior`](@ref) [`fitted_interior`](@ref) builds from it. Such a
case takes no `r_0`, `r_1` or `target` of its own (the spec carries its
target) and needs a [`Horizon`](@ref) to be tracked with.

`center` and `velocity` are the hole's analytic trajectory `c(t) = c₀ + v
t` — the thing the interior, the damping profile and the refinement
centroid all measure a distance from. **The velocity is the background's
(amended in step 8e):** left unset it is [`hole_velocity`](@ref)`(background)`
— zero for a static hole, `−u` for `boost(m, u)`, which `SpacetimeMetrics`
evaluates at `Λᵀx` so that its hole moves the other way — and zero for a
background with no hole; given, it must agree with the background's, and an
`ArgumentError` stating both and the convention refuses one that does not.
Step 8d found the sign; before step 8e a boosted case carried `v = +u` unless
told otherwise, and nothing in G4 moves. `chunk` is the regrid
cadence [`evolve!`](@ref) runs at; zero means "not a case that is
evolved in chunks", and the driver says so rather than assuming one.

**The refinement (added in step 6).** `refinement` is `nothing` — a case
that is run on a mesh somebody else chose, which is every case up to step 5
and every frozen-hierarchy convergence study — or a [`Refinement`](@ref)
carrying the two thresholds, the level cap and the two geometric
corrections. `CODE.md` lists those as fields of the case; they are one
struct, so that a case without refinement carries one `nothing` rather than
five placeholders. [`evolve!`](@ref) refuses `regrid = true` and
`adapt = true` on a case without one, because this package has exactly one
refinement mechanism.

**The horizon (added in step 7).** `horizon` is `nothing` — a case whose
record carries no horizon rows, which is every case with no hole — or a
[`Horizon`](@ref) carrying the cadence `k`, the finder's resolution and its
seed. `CODE.md`'s analysis table puts the horizon rows "every `k`-th chunk,
`k` a case parameter", so it is a field of the case for the same reason
`refinement` is: a driver keyword would make the cadence a property of the
run rather than of the study.

**The range bounds (added in step 8b).** `bounds` is `nothing` — no range
projection, which is every case before step 8b and the default — or a
[`StateBounds`](@ref): the ranges of `α`, of `γ`'s spectrum, of `|β|` and of
`Π`'s scale that the stage limiter holds the state inside at `r < r_gate`.
A field of the case like `horizon`, for the same reason. It needs an
interior, because its gate is a radius about the hole's center, and the gate
must lie inside `r_1`; both are refused here by name, and the mesh-dependent
half of the gate's placement is asserted where the interior's radii are.

**The layer's target and the dissipation profile (added in step 8c).**
`target` is passed to the [`Interior`](@ref): `nothing`, the case's own
background, or another metric the layer relaxes toward instead — and only
the layer; every other path that evaluates the analytic solution stays on
`background`. `ε_KO` is a number, as it always was, or a
[`HorizonDissipation`](@ref) about this case's center; a number stays a
number, and the kernel's arithmetic on it is what it was
([`dissipation_rate`](@ref) is the identity on it).

**A moving non-harmonic background is refused here**, with the message
`CODE.md` asks for under "Gauge and constraint damping": such a background
has a gauge source `H_a(x − vt)` that a per-chunk sample cannot represent,
and evaluating it in the kernel would cost about one right-hand side. The
proof-of-concept case, `boost(Harmonic(M, a), v)`, is harmonic — a boost
preserves `□x^a = 0` — which is why it is the case.

The struct is `isbits` whenever the background is, because the whole case
travels into kernels: the Dirichlet hook closes over it at every ghost
fill, and the interior's `u_exact` does at every evaluation. That is why
the absent interior is a field of concrete type `Nothing` behind a type
parameter rather than a `Union`, and why the variant is a `Val`.

**(Amended in step 3.)** `CODE.md`'s file table put `GHCase` in
`driver.jl`. It is here instead, with the backgrounds it is made of: the
right-hand side needs a case two steps before there is a driver, and a
struct cannot be defined twice. `driver.jl` adds `evolve!` and the
refinement fields step 6 needs.
"""
struct GHCase{T,B,D,I,R,H,X,E}
    background::B
    box::NTuple{3,Tuple{T,T}}
    periodic::NTuple{3,Bool}
    ε_KO::E                      # a number, or a `HorizonDissipation` (step 8c)
    γ0::D                        # a damping profile, not a number
    γ2::T
    center::HoleCenter{T}
    interior::I                  # an `Interior`, or `nothing`
    refinement::R                # a `Refinement`, or `nothing`
    horizon::H                   # a `Horizon`, or `nothing`
    bounds::X                    # a `StateBounds`, or `nothing`
    chunk::T
end

function GHCase(::Type{T}, background; box, periodic, ε_KO, γ0, γ2,
                center=(zero(T), zero(T), zero(T)), velocity=nothing,
                interior=nothing, r_0=zero(T), r_1=zero(T), margin::Integer=8,
                w_ramp=T(1 // 2), ρ_ramp=T(1 // 2), target=nothing,
                refinement=nothing, horizon=nothing, bounds=nothing,
                chunk=zero(T)) where {T}
    isharmonic(background) || isstatic(background) || throw(ArgumentError(
        "this background is neither harmonic nor static, so its prescribed " *
        "gauge source H_a(x − vt) depends on time, and CODE.md's Hsrc field " *
        "set — sampled once per chunk as a function of position — cannot " *
        "represent it; evaluating H_a in the kernel would cost about one " *
        "right-hand side and is listed as an extension, not built. A boosted " *
        "Kerr-Schild is exactly this case. Use boost(Harmonic(M, a), v) " *
        "instead: a boost preserves the harmonic condition □x^a = 0, so the " *
        "boosted hole in harmonic coordinates has H ≡ 0 and needs no source " *
        "at all. That is why it is the proof-of-concept case."))
    damping = γ0 isa Real ? ConstantDamping(T, γ0) : γ0
    first(damping_bounds(damping)) ≥ 0 || throw(ArgumentError(
        "the constraint-damping rate must satisfy γ0 ≥ 0 everywhere — the " *
        "term enters ∂_tΠ with a factor −α√γ, so a negative γ0 drives the " *
        "constraint violation it is there to damp — but this profile falls " *
        "to $(first(damping_bounds(damping)))"))
    γ2 > -1 || throw(ArgumentError(
        "the Gundlach–Pretorius trace parameter must satisfy γ2 > −1 at the " *
        "continuum level for the damped system to stay well posed, but " *
        "γ2 = $γ2"))
    all(d -> box[d][2] > box[d][1], 1:3) || throw(ArgumentError(
        "every dimension of the box needs hi > lo, but box = $box"))
    T(chunk) ≥ 0 || throw(ArgumentError(
        "the chunk length is a regrid cadence and cannot be negative, got " *
        "$chunk; zero means the case states none and evolve! must be told."))
    c = HoleCenter(T, center, case_velocity(T, background, velocity))
    interior === nothing && target !== nothing && throw(ArgumentError(
        "this case has a layer target but no interior: the target is what " *
        "the damping layer relaxes toward, and a case with no hole has no " *
        "layer. Give the case an interior, or leave `target = nothing`."))
    int = if interior === nothing
        nothing
    elseif interior isa FittedSpec
        # The tracked geometry (step 8d): the case holds the rule, and the
        # layer is built from the found horizon once per chunk. Its target
        # travels in the spec, and it has no radii of its own.
        interior isa FittedSpec{T} || throw(ArgumentError(
            "a case's FittedSpec is stated in the case's own type $T, got a " *
            "$(typeof(interior))."))
        target === nothing || throw(ArgumentError(
            "a tracked case's layer target is the FittedSpec's own `target`; " *
            "the case's `target` keyword is for step 5's sphere."))
        interior
    else
        Interior(T; center=c, r_0=r_0, r_1=r_1, variant=Symbol(interior),
                 margin=margin, w_ramp=w_ramp, ρ_ramp=ρ_ramp, target=target)
    end
    check_case_bounds(bounds, int, T)
    ε = case_dissipation(ε_KO, c, T)
    return GHCase{T,typeof(background),typeof(damping),typeof(int),
                  typeof(refinement),typeof(horizon),typeof(bounds),
                  typeof(ε)}(
        background, ntuple(d -> (T(box[d][1]), T(box[d][2])), Val(3)),
        ntuple(d -> Bool(periodic[d]), Val(3)), ε, damping, T(γ2), c,
        int, refinement, horizon, bounds, T(chunk))
end

"""
    case_velocity(T, background, velocity) -> SVector{3,T}

The velocity a case's [`HoleCenter`](@ref) carries (added in step 8e): the
background's own, [`hole_velocity`](@ref), when `velocity` is `nothing`;
zero when it is `nothing` and the background has no hole this package can
classify (flat space in any chart); and `velocity` itself when it is given —
after checking it against the background's where there is one, to `8 eps`,
and refusing it with both vectors and the convention when it disagrees. A
keyword that disagrees with the background is a case whose layer, damping
profile and tracked seed move one way while its hole moves the other, which
is what step 8d found `boost`'s sign would do to a case built with `+u`.
"""
function case_velocity(::Type{T}, background, velocity) where {T}
    known = has_hole_velocity(background)
    if velocity === nothing
        known || return zero(SVector{3,T})
        vb = hole_velocity(background)
        return SVector{3,T}(T(vb[1]), T(vb[2]), T(vb[3]))
    end
    v = SVector{3,T}(T(velocity[1]), T(velocity[2]), T(velocity[3]))
    known || return v
    vb = hole_velocity(background)
    vbt = SVector{3,T}(T(vb[1]), T(vb[2]), T(vb[3]))
    δ = sqrt(sum(abs2, v - vbt))
    δ ≤ 8 * eps(T) || throw(ArgumentError(
        "the case's velocity = $(Tuple(v)) disagrees with its background's " *
        "hole, which moves at hole_velocity(background) = $(Tuple(vbt)) " *
        "(a difference of $δ): SpacetimeMetrics' boost(m, u) evaluates m at " *
        "Λᵀx, so the boosted hole's rest-frame origin is the lab's x = −u t " *
        "and its velocity is −u, not u (found in step 8d, fixed in step 8e). " *
        "A case whose trajectory disagrees with its hole puts the layer, the " *
        "damping profile and the tracked seed where the hole is not. Leave " *
        "`velocity` unset to take the background's."))
    return vbt
end

# The Kreiss–Oliger amplitude a case carries (added in step 8c): a number in
# the case's type, as before, or a profile in it about the case's own hole.
case_dissipation(ε::Real, c, ::Type{T}) where {T} = T(ε)

function case_dissipation(ε, c, ::Type{T}) where {T}
    ε isa HorizonDissipation{T} || throw(ArgumentError(
        "a case's ε_KO is a number or a HorizonDissipation{$T}, got a " *
        "$(typeof(ε)): it is a kernel argument in the case's own working " *
        "type, like every other number the case carries."))
    ε.center == c || throw(ArgumentError(
        "the dissipation profile is centered on $(ε.center) but the case's " *
        "hole is at $c: the profile rises from the horizon of *this* hole " *
        "inward, and a profile about another point would raise the " *
        "dissipation outside the horizon. horizon_dissipation(case; ε_in) " *
        "builds it from the case."))
    return ε
end

# The two refusals of a case's bounds that need no mesh (added in step 8b):
# the working type, and an interior for the gate to be a radius about, with
# the gate inside the layer's outer radius. The rest of the gate's placement
# — deeper than every point an evolved stencil reads — depends on the
# spacing, and is `check_bounds_gate`'s, at every regrid.
check_case_bounds(::Nothing, int, ::Type) = nothing

# A tracked case has no radius to hold the gate against until its geometry is
# built; `check_bounds_gate` asserts it against each one (step 8d).
check_case_bounds(::Nothing, ::FittedSpec, ::Type) = nothing

function check_case_bounds(bd, ::FittedSpec, ::Type{T}) where {T}
    bd isa StateBounds{T} || throw(ArgumentError(
        "a case's bounds are a StateBounds{$T} or `nothing`, got a " *
        "$(typeof(bd))."))
    return nothing
end

function check_case_bounds(bd, int, ::Type{T}) where {T}
    bd isa StateBounds{T} || throw(ArgumentError(
        "a case's bounds are a StateBounds{$T} or `nothing`, got a " *
        "$(typeof(bd)): the ranges are kernel arguments in the case's own " *
        "working type, like every other number the case carries."))
    int === nothing && throw(ArgumentError(
        "this case has range bounds but no interior: the projection is gated " *
        "on r < r_gate from the hole's analytic center, and a case with no " *
        "hole has no center — and nothing that is not a numerical solution " *
        "for the projection to guard. Give the case an interior, or leave " *
        "`bounds = nothing`."))
    bd.r_gate ≤ int.r_1 || throw(ArgumentError(
        "the range projection's gate r_gate = $(bd.r_gate) is outside the " *
        "layer's outer radius r_1 = $(int.r_1): the projection would clamp " *
        "evolved points, which are the Einstein equations and nothing else. " *
        "The gate has to lie deeper than every point an evolved stencil " *
        "reads; default_gate proposes r_1 − 2Gh."))
    return nothing
end

GHCase(background; kwargs...) = GHCase(Float64, background; kwargs...)

"""
    with_interior(case::GHCase, interior) -> GHCase

The same case carrying a different [`Interior`](@ref) — what the driver
builds once per chunk when it replaces the layer's `ρ_max` with that
chunk's rate, and what a test that compares `CODE.md`'s three variants
changes between runs.

A reconstruction and not a mutation, for the reason
[`with_ρ_max`](@ref) is: the case is a kernel argument at every ghost fill
and every evaluation.
"""
with_interior(case::GHCase{T}, interior) where {T} =
    GHCase{T,typeof(case.background),typeof(case.γ0),typeof(interior),
           typeof(case.refinement),typeof(case.horizon),typeof(case.bounds),
           typeof(case.ε_KO)}(
        case.background, case.box, case.periodic, case.ε_KO, case.γ0, case.γ2,
        case.center, interior, case.refinement, case.horizon, case.bounds,
        case.chunk)

"""
    with_refinement(case::GHCase, refinement) -> GHCase

The same case carrying different refinement parameters — what a test that
loosens `refine_tol` to watch the level floor bind changes between two
flagging passes, and what a study that sweeps the thresholds varies.

A reconstruction and not a mutation, for the reason
[`with_interior`](@ref) is.
"""
with_refinement(case::GHCase{T}, refinement) where {T} =
    GHCase{T,typeof(case.background),typeof(case.γ0),typeof(case.interior),
           typeof(refinement),typeof(case.horizon),typeof(case.bounds),
           typeof(case.ε_KO)}(
        case.background, case.box, case.periodic, case.ε_KO, case.γ0, case.γ2,
        case.center, case.interior, refinement, case.horizon, case.bounds,
        case.chunk)

"""
    with_horizon(case::GHCase, horizon) -> GHCase

The same case carrying different horizon-analysis parameters — what a test
that wants the horizon rows of an otherwise ordinary fixture changes, and
what a run that wants them at a different cadence or resolution varies.

A reconstruction and not a mutation, for the reason
[`with_interior`](@ref) is (added in step 7).
"""
with_horizon(case::GHCase{T}, horizon) where {T} =
    GHCase{T,typeof(case.background),typeof(case.γ0),typeof(case.interior),
           typeof(case.refinement),typeof(horizon),typeof(case.bounds),
           typeof(case.ε_KO)}(
        case.background, case.box, case.periodic, case.ε_KO, case.γ0, case.γ2,
        case.center, case.interior, case.refinement, horizon, case.bounds,
        case.chunk)

"""
    with_bounds(case::GHCase, bounds) -> GHCase

The same case carrying different range bounds — `nothing` to switch the
projection off, or a [`StateBounds`](@ref) — which is how a study compares a
run with the projection against the same run without it, and how the gate
is set once the mesh, and with it [`default_gate`](@ref), is known.

A reconstruction and not a mutation, for the reason
[`with_interior`](@ref) is (added in step 8b); the refusals of the
constructor apply.
"""
function with_bounds(case::GHCase{T}, bounds) where {T}
    check_case_bounds(bounds, case.interior, T)
    return GHCase{T,typeof(case.background),typeof(case.γ0),
                  typeof(case.interior),typeof(case.refinement),
                  typeof(case.horizon),typeof(bounds),typeof(case.ε_KO)}(
        case.background, case.box, case.periodic, case.ε_KO, case.γ0, case.γ2,
        case.center, case.interior, case.refinement, case.horizon, bounds,
        case.chunk)
end

"""
    with_dissipation(case::GHCase, ε_KO) -> GHCase

The same case with a different Kreiss–Oliger amplitude — a number, or a
[`HorizonDissipation`](@ref) about the case's hole — which is how step 8c's
calibration puts the `ε_KO(r)` profile on a case built with the exterior's
number. A reconstruction and not a mutation, for the reason
[`with_interior`](@ref) is (added in step 8c); the constructor's refusals
apply.
"""
function with_dissipation(case::GHCase{T}, ε_KO) where {T}
    ε = case_dissipation(ε_KO, case.center, T)
    return GHCase{T,typeof(case.background),typeof(case.γ0),
                  typeof(case.interior),typeof(case.refinement),
                  typeof(case.horizon),typeof(case.bounds),typeof(ε)}(
        case.background, case.box, case.periodic, ε, case.γ0, case.γ2,
        case.center, case.interior, case.refinement, case.horizon,
        case.bounds, case.chunk)
end

"""
    horizon_dissipation(case::GHCase; ε_in, ε_out = case.ε_KO) -> HorizonDissipation

`CODE.md`'s `ε_KO(r)` profile for this case's hole (added in step 8c):
`ε_out` — by default the case's own number — at and outside the horizon's
smallest coordinate radius ([`horizon_min_radius`](@ref) of the background,
boost-contracted), rising to `ε_in` at the layer's outer radius `r_1` and
held inside it, about the case's own center.
"""
function horizon_dissipation(case::GHCase{T}; ε_in, ε_out=case.ε_KO) where {T}
    case.interior === nothing && throw(ArgumentError(
        "the dissipation profile rises from the horizon to the layer's outer " *
        "radius r_1, and this case has no interior, so no r_1."))
    case.interior isa FittedSpec && throw(ArgumentError(
        "the dissipation profile is a function of the radius about the " *
        "analytic center, and a tracked case's layer follows a surface: step " *
        "8c measured the profile off for a smooth target, and a tracked one " *
        "would have to be stated in the depth (not built in step 8d)."))
    ε_out isa Real || throw(ArgumentError(
        "the profile's exterior amplitude is a number, got a " *
        "$(typeof(ε_out)): pass `ε_out` explicitly for a case whose ε_KO is " *
        "already a profile."))
    return HorizonDissipation(T; ε_out=ε_out, ε_in=ε_in,
                              r_1=case.interior.r_1,
                              r_h=horizon_min_radius(case.background),
                              center=case.center)
end

"""
    minkowski_case(T = Float64; L, ε_KO, γ0, γ2)

Flat space on a periodic box of side `L` — `CODE.md`'s first row, the
robust-stability case. Its exact solution is `h = Π = 0`, so the
right-hand side is *exactly* zero on it, which is the sharpest statement
any test in this package makes about the kernel.
"""
minkowski_case(::Type{T}=Float64; L, ε_KO, γ0, γ2) where {T} =
    GHCase(T, Minkowski(); box=ntuple(_ -> (zero(T), T(L)), Val(3)),
           periodic=(true, true, true), ε_KO=ε_KO, γ0=γ0, γ2=γ2)

"""
    gauge_wave_case(T = Float64; A = 1//20, d = 1, ε_KO, γ0, γ2)

The Apples-with-Apples gauge wave, `ds² = −H dt² + H dx² + dy² + dz²` with
`H = 1 − A sin(2π(x − t)/d)` — flat spacetime in an oscillating chart,
travelling along `x` at coordinate speed 1, `CODE.md`'s second row and the
case that measures the scheme's order.

The box is one wavelength cubed and periodic in every dimension, so the
solution is exactly periodic and the mesh needs no boundary at all; one
crossing is `t = d`. `A = 1//20` is `PLAN.md`'s amplitude, written as a
rational so that the `Float32` and `Float64` runs are the same number
rounded once.
"""
gauge_wave_case(::Type{T}=Float64; A=T(1//20), d=one(T), ε_KO, γ0,
                γ2) where {T} =
    GHCase(T, GaugeWave(T(A), T(d)); box=ntuple(_ -> (zero(T), T(d)), Val(3)),
           periodic=(true, true, true), ε_KO=ε_KO, γ0=γ0, γ2=γ2)

"""
    shifted_minkowski_case(T = Float64; A = 1//2, w = 2, halfwidth = 2,
                           ε_KO, γ0, γ2)

Flat space in time-skewed coordinates `t̂ = t + A w tanh(x/w)` — static,
with a space-varying subluminal shift and, being non-harmonic, a
**sampled gauge source**. `CODE.md`'s third row, and the case that
exercises the `Hsrc` path and a nonzero `β^i`.

The profile `ψ′(x) = A sech²(x/w)` is a function of `x` alone and is not
periodic, so the box is Dirichlet in `x` and periodic in `y` and `z`
**(amended in step 3**; `CODE.md`'s "Boundaries: Periodic" had listed this
case as periodic in every dimension, which the profile is not: it is
*even* in `x`, so closing the box would join two equal values with
opposite gradients — a kink, which no stencil resolves and which every
error norm would then report as truncation error**)**. Being exactly
independent of `y` and `z`, the other two dimensions are periodic without
any mismatch at all. The Dirichlet data is the analytic solution, so the
boundary is exact; this is the case that puts [`dirichlet`](@ref) under
test.

`w = 2` on a box of half-width `2` **(proposed in step 3)**: what a
convergence study needs is the profile *resolved* at the coarsest
resolution it uses, and `w = 1` is not, at the three the suite can
afford. Measured at `q = 4`, `N = 8`, `roots = 2 … 4`: `w = 1` converges
at **3.6** and `w = 2` at **3.93**, with the same code — the shortfall is
the sech² profile against `h = L/16`, not the scheme. The Dirichlet
boundary is exact, so a profile that is still `0.42 A` at the face costs
nothing.
"""
shifted_minkowski_case(::Type{T}=Float64; A=T(1//2), w=T(2), halfwidth=T(2),
                       ε_KO, γ0, γ2) where {T} =
    GHCase(T, ShiftedMinkowski(T(A), T(w));
           box=ntuple(_ -> (-T(halfwidth), T(halfwidth)), Val(3)),
           periodic=(false, true, true), ε_KO=ε_KO, γ0=γ0, γ2=γ2)

"""
    hole_case(T = Float64, background; halfwidth, r_0, r_1, chunk,
              M = 1, center = (0,0,0), velocity = the background's,
              interior = :damped, margin = 8, ε_KO = 1//2,
              γ0 = GHSO2's recipe, γ2 = 0, w_ramp, ρ_ramp, target = nothing,
              refinement = nothing, horizon = nothing, bounds = nothing)

A black hole in a **Dirichlet** box, with the damping layer of
`CODE.md`'s "The interior" and GHSO2's recipe near a hole — the shape both
[`kerr_schild_case`](@ref) and [`harmonic_kerr_case`](@ref) take, written
once because the only thing that differs between them is the chart.

The box is `[-halfwidth, halfwidth]³` and is Dirichlet in **every**
dimension: the exact solution is known everywhere at every time, so the
boundary data is the solution (`CODE.md`, "Boundaries"). `CODE.md`'s G4
asks for `halfwidth ≥ 20 M`; the suite's runs are far smaller and say so
where they are written, because the boundary is exact and a small box
costs accuracy rather than validity.

`ε_KO = 1//2` and `γ0 ≈ 1/M` are **GHSO2's measured requirements** with the
horizon in the domain (`notes/methods-ghso2.md`: `γ0 = 0` blows up at the
surface-gravity rate `κ`, and the grid-scale layer is cured by
`ε_KO ≈ 0.5`), so they are the defaults *here* and nowhere else — a
flat-space case has no business inheriting them silently. The default
`γ0` is a [`GaussianDamping`](@ref) of width `3 M` around the hole,
`1/M` at the center and `1/(10 M)` in the wave zone
**(proposed in step 5**: `CODE.md` asks for "a Gaussian of width a few `M`
… tapered to a small value in the wave zone" and leaves the three numbers
open**)**.

`r_0`, `r_1` and `chunk` have no defaults: the two radii are what
[`check_interior_radii`](@ref) measures against the mesh and the horizon,
and the chunk is the cadence the analysis record is written at. A
`:damped`, `:frozen` or `:pasted` layer without both radii is refused by
name, and a tracked one (`interior = FittedSpec(…)`, step 8d) with either —
its radii come from the horizon that was found.

`bounds = nothing` is the one default of step 8b's range projection, and it
means "off" (added in step 8b): a [`StateBounds`](@ref) has no default for
any of its ranges, and [`default_bounds`](@ref) is the named proposal a
caller asks for explicitly.
"""
function hole_case(::Type{T}, background; halfwidth, r_0=nothing, r_1=nothing,
                   chunk,
                   M=one(T), center=(zero(T), zero(T), zero(T)),
                   velocity=nothing, interior=:damped,
                   margin::Integer=8, ε_KO=T(1 // 2), γ0=nothing,
                   γ2=zero(T), w_ramp=T(1 // 2), ρ_ramp=T(1 // 2),
                   target=nothing, refinement=nothing, horizon=nothing,
                   bounds=nothing) where {T}
    # `r_0` and `r_1` have no default for step 5's sphere — they are what
    # `check_interior_radii` measures — and no meaning for the tracked
    # geometry, whose radii come from the horizon that was found (step 8d).
    if interior isa FittedSpec
        (r_0 === nothing && r_1 === nothing) || throw(ArgumentError(
            "a tracked case derives its layer's radii from the found horizon " *
            "(offset = m h and thickness = n_L h below it), so it takes no " *
            "r_0 or r_1; got r_0 = $r_0, r_1 = $r_1."))
    elseif interior !== nothing
        (r_0 === nothing || r_1 === nothing) && throw(ArgumentError(
            "a :$interior layer needs its two radii, r_0 (the frozen core) and " *
            "r_1 (the layer's outer radius), and they have no default: they " *
            "are what check_interior_radii measures against the mesh and the " *
            "horizon (CODE.md, \"The interior\")."))
    end
    # The damping profile moves with the hole, so it is built on the case's
    # own trajectory — the background's velocity unless one is given, and
    # then that one, checked by `GHCase` (amended in step 8e: it was built on
    # the keyword's default zero, which a boosted hole does not have).
    v = case_velocity(T, background, velocity)
    damping = γ0 !== nothing ? γ0 :
              GaussianDamping(T; near=1 / T(M), far=1 / (10 * T(M)),
                              width=3 * T(M), center=HoleCenter(T, center, v))
    return GHCase(T, background;
                  box=ntuple(_ -> (-T(halfwidth), T(halfwidth)), Val(3)),
                  periodic=(false, false, false), ε_KO=ε_KO, γ0=damping,
                  γ2=γ2, center=center, velocity=v, interior=interior,
                  r_0=r_0 === nothing ? zero(T) : r_0,
                  r_1=r_1 === nothing ? zero(T) : r_1, margin=margin,
                  w_ramp=w_ramp,
                  ρ_ramp=ρ_ramp, target=target, refinement=refinement,
                  horizon=horizon, bounds=bounds, chunk=chunk)
end

hole_case(background; kwargs...) = hole_case(Float64, background; kwargs...)

"""
    kerr_schild_case(T = Float64; M = 1, a = 0, halfwidth, r_0, r_1, chunk, …)

`CODE.md`'s fourth row: a hole in **Kerr-Schild** Cartesian coordinates,
which is horizon-penetrating and *not* harmonic — so it carries a
**sampled gauge source**, and it is the case that puts the `Hsrc` path
under a hole rather than under a shift.

Its horizon's smallest coordinate radius is `r₊ = M + √(M² − a²)`, twice
the harmonic chart's at `a = 0`, which makes it the cheaper of the two
holes to resolve and the one a first test should use.

Every other keyword is [`hole_case`](@ref)'s.
"""
kerr_schild_case(::Type{T}=Float64; M=one(T), a=zero(T), kwargs...) where {T} =
    hole_case(T, KerrSchild(T(M), T(a)); M=T(M), kwargs...)

"""
    harmonic_kerr_case(T = Float64; M = 1, a = 0, halfwidth, r_0, r_1, chunk, …)

`CODE.md`'s fifth row: a hole in **fully harmonic** Cartesian coordinates,
horizon penetrating and harmonic — `H ≡ 0` exactly, so no `Hsrc` field set
exists and the kernel is compiled without the gauge-source terms — and the
chart the proof-of-concept case is a boost of.

Its horizon's smallest coordinate radius is `√(M² − a²)`, which is `M` at
`a = 0` and about `0.44 M` at `a = 9/10`: the spinning hole is the
expensive one, and `CODE.md` names those bounds, not the exterior, as what
sets the finest spacing a run needs.

Every other keyword is [`hole_case`](@ref)'s.
"""
harmonic_kerr_case(::Type{T}=Float64; M=one(T), a=zero(T), kwargs...) where {T} =
    hole_case(T, Harmonic(T(M), T(a)); M=T(M), kwargs...)

"""
    gh_forest(T = Float64, case::GHCase; N, roots, refined = false)

A forest of `roots³` root blocks of `N³` points each, over the case's box
and with the case's periodicity — the mesh every test in step 3 runs on,
and, with `refined = true`, the **two-level** mesh that step 4 measures the
interface-order rule on.

`N` and `roots` have no defaults: they are the two numbers a convergence
study varies. TreeAMR's blocks are cubes with one isotropic spacing, so
the box has to be one too; `h = (box width)/(roots·N)`.

`refined = true` refines the one root block that contains the point three
eighths of the way along each axis, and balances — TreeWave's
`wave_forest` pattern (added in step 4), with two differences that matter
here. It refines a **single** block rather than a middle sub-box, because
the interface study runs at `roots = 2`, where every root block *is* a
half of the domain; and the refined region is a fixed *block*, so that a
convergence study which doubles `N` leaves the block layout alone and
every spacing halves — `CODE.md`'s frozen-hierarchy protocol under
"Refinement and regridding", and the only way a rate measured on a refined
mesh means anything. One refined block in a periodic `2³` root grid has a
coarse-fine face on each of its six sides, so both directions of the
interface — prolongation into the fine ghosts, injection into the coarse
ones — are exercised.

The leading `T` is the type the whole run is computed in and reaches the
field set, the schedule and the state vector through the forest, which is
the only place it has to be said (TreeWave's `wave_forest`, which this
follows).
"""
function gh_forest(::Type{T}, case::GHCase; N, roots, refined=false) where {T}
    widths = ntuple(d -> case.box[d][2] - case.box[d][1], Val(3))
    all(w -> w ≈ widths[1], widths) || throw(ArgumentError(
        "TreeAMR's blocks are cubes with a single spacing per level, so a " *
        "uniform forest over this box would have anisotropic cells: the " *
        "widths are $widths"))
    forest = Forest{T}(ntuple(_ -> roots, Val(3)); N=N, periodic=case.periodic,
                       extents=case.box)
    refined || return forest
    xref = ntuple(d -> case.box[d][1] + 3 * widths[d] / 8, Val(3))
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> ext[d][1] ≤ xref[d] < ext[d][2], 1:3)
    end
    length(targets) == 1 || throw(ArgumentError(
        "the two-level mesh refines exactly the one root block holding " *
        "$xref, but $(length(targets)) blocks claim it — which means the " *
        "reference point landed on a block face, and a mesh that depends on " *
        "which side it lands is not a mesh a convergence study can be " *
        "frozen on"))
    refine!(forest, targets)
    balance!(forest)
    return forest
end

gh_forest(case::GHCase; kwargs...) = gh_forest(Float64, case; kwargs...)

"""
    hole_forest(T = Float64, case::GHCase; N, roots, center = case's, radii,
                levels = length(radii))

A **fixed** nested hierarchy of `levels` shells around `center`: the
uniform forest of [`gh_forest`](@ref), then, for each level `ℓ`, every
leaf of level `ℓ − 1` whose extent reaches within `radii[ℓ]` of the center
refined and the forest balanced.

This is **not** the refinement mechanism. The indicator that chooses a
mesh from the solution is step 6's `refinement.jl`; this is the *frozen
hierarchy* `CODE.md`'s convergence protocol is stated on — "the block
layout is unchanged and every spacing halves" — and the first mesh the
interior layer is tested on. Which blocks are refined depends on `radii`,
`roots` and `levels` and on nothing else, so **doubling `N` leaves the
layout alone and halves every spacing**, which is the only way a rate
measured around a hole means anything. TreeAMR's `wave_forest` pattern,
as [`gh_forest`](@ref)`(; refined = true)` is.

`radii` is one radius per level and must decrease: the shells nest, and a
shell that grew outward would refine a region its parent level does not
cover, which 2:1 balance would then have to repair by refining half the
domain. The default `center` is the case's own `c(0)`, because that is
where its hole is at the time the hierarchy is built.

`levels` is `length(radii)` and is accepted explicitly so that a caller
that says both is told when they disagree rather than silently getting one
of them.
"""
function hole_forest(::Type{T}, case::GHCase; N, roots,
                     center=center_at(case.center, zero(T)), radii,
                     levels::Integer=length(radii)) where {T}
    length(radii) == levels || throw(ArgumentError(
        "hole_forest takes one radius per refinement level, but levels = " *
        "$levels and radii has $(length(radii)) entries: the shells are the " *
        "hierarchy, and a level without a radius has nothing to be built " *
        "around."))
    all(ℓ -> radii[ℓ] > 0, 1:levels) || throw(ArgumentError(
        "every shell radius must be positive, got radii = $radii"))
    all(ℓ -> radii[ℓ] ≤ radii[ℓ - 1], 2:levels) || throw(ArgumentError(
        "the shells must nest, so the radii must not increase with the " *
        "level, but radii = $radii. A shell wider than its parent's would " *
        "ask for a fine block outside the coarser refined region, and 2:1 " *
        "balance would answer by refining everything between them."))
    forest = gh_forest(T, case; N=N, roots=roots)
    c = ntuple(d -> T(center[d]), Val(3))
    for ℓ in 1:levels
        R = T(radii[ℓ])
        targets = filter(forest.leaves) do k
            level(k) == ℓ - 1 &&
                _box_meets_ball(block_extent(T, forest, k), c, R)
        end
        isempty(targets) && throw(ArgumentError(
            "no leaf of level $(ℓ - 1) comes within radii[$ℓ] = $R of " *
            "$c, so shell $ℓ would refine nothing: either the center is " *
            "outside the box, or the radii fall faster than the block size " *
            "does and the hierarchy stops before the level it was asked for."))
        refine!(forest, targets)
        balance!(forest)
    end
    return forest
end

hole_forest(case::GHCase; kwargs...) = hole_forest(Float64, case; kwargs...)

"""
    background_state(background, t, x) -> (h, Π, ∂h)

The analytic state of a background at `(t, x)`: the offset metric
`h_ab = g_ab − η_ab`, the densitised momentum
`Π_ab = (√γ/α)(∂_t − β^i ∂_i) g_ab`, and the spatial gradients
`∂h[i] = ∂_i h_ab`, all packed in the `(tt, tx, …, zz)` order.

**This is the one place the two derivative index conventions meet.**
`dmetric` returns `dg[a, b, c] = ∂_c g_ab`, so `dg[:, :, 1]` is `∂_t g`
and `dg[:, :, i+1]` is `∂_i g`; everything downstream of here speaks
GHSO2's `∂_a g_bc` (`CLAUDE.md`, "Two derivative index conventions"). The
gradients are the *analytic* ones, not differences of the sampled state:
`CODE.md`, "Initial data and backgrounds", keeps GHSO2's discrete-gradient
`Π` as a post-pass option for G4 and measures there whether it changes a
hole's stationarity.

Generic in the element type of `x`, including `ForwardDiff` duals, so that
a test may differentiate it — which is how `test/initialdata_tests.jl`
checks the conversion against a spelling that does not use `dmetric` at
all. It is also a kernel argument's body: [`state_callback`](@ref) and
[`dirichlet`](@ref) are closures around it, evaluated per point inside
`fill_by_coordinates!` and the boundary hook.
"""
@inline function background_state(bg, t, x)
    D = typeof(x[1])
    p = SVector{4,D}(D(t), x[1], x[2], x[3])
    g, dg = dmetric(bg, p)
    h = pack_g(g)
    ∂ₜg = pack_sym(SMatrix{4,4,D}(dg[a, b, 1] for a in 1:4, b in 1:4))
    ∂h = ntuple(Val(3)) do i
        pack_sym(SMatrix{4,4,D}(dg[a, b, i + 1] for a in 1:4, b in 1:4))
    end
    _, _, α, β, _, sqrtγ = metric_quantities(_sym4(h))
    Π = (sqrtγ / α) * (∂ₜg - β[1] * ∂h[1] - β[2] * ∂h[2] - β[3] * ∂h[3])
    return h, Π, ∂h
end

"""
    state_tuple(background, t, x) -> NTuple{20}

`(h..., Π...)`, the 20 evolved values at one point — [`background_state`](@ref)
in the shape TreeAMR's `AllVariables` callbacks return.

The spatial gradients are dropped here and *not* stored: the evolution
takes them from the mesh, and keeping the analytic ones would be a second
opinion about a quantity the scheme has to produce for itself.
"""
@inline function state_tuple(bg, t, x)
    h, Π, _ = background_state(bg, t, x)
    return (Tuple(h)..., Tuple(Π)...)
end

"""
    case_state_tuple(background, interior, t, x) -> NTuple{20}

The 20 evolved values the *case* puts at `x`: the background's, except
inside the frozen core, where [`core_position`](@ref) sends the query to
the sphere `r_0` along the ray.

`CODE.md`, "The frozen core": the core holds finite data and is never
read, and it is filled with the analytic solution on the sphere `r_0`
along the ray — continuous at `r_0`, finite everywhere. This is the one
function that applies that rule, and every path that writes the analytic
solution into a field set goes through it: the initial data, the error
reference, and the Dirichlet hook (where the core is nowhere near the
boundary and the rule is the identity, which is why it costs nothing to
be consistent).

`interior === nothing` is no hole and the rule is the identity, so the
whole branch folds away for every case up to step 4.
"""
@inline case_state_tuple(bg, interior, t, x) =
    state_tuple(bg, t, core_position(interior, t, x))

"""
    state_callback(case::GHCase, t) -> AllVariables

The case's exact state at time `t` as the coordinate callback
`fill_by_coordinates!` and `adapt_to_initial_data!` take: `x ↦ (h…, Π…)`,
once per point, all 20 variables at once — with the frozen core's rule
applied (added in step 5; see [`case_state_tuple`](@ref)).

It closes over the background, the interior and `t` — all `isbits`, none a
`Type` — so it is a kernel argument like any other and runs wherever the
field set lives (`CODE.md`, "Initial data and backgrounds"; the dependency
risk it names is what `test/prerequisite_tests.jl` settles on the CPU).

Built fresh at each call with that call's time, which is the rule for
every hook in this package: the initial data at `t = 0`, the error
reference at the end of a chunk, the Dirichlet data at every ghost fill
(`CLAUDE.md`, "Hooks depend on time").
"""
function state_callback(case::GHCase{T}, t; interior=case.interior) where {T}
    bg = case.background
    int = interior
    int isa FittedSpec && throw(ArgumentError(
        "this case's interior is a FittedSpec, the rule a tracked geometry is " *
        "built by: the core rule needs the geometry itself — pass `interior " *
        "= fitted_interior(…)`, which is what evolve! does for the initial " *
        "data (step 8d)."))
    tt = T(t)
    return AllVariables(x -> case_state_tuple(bg, int, tt, x))
end

"""
    fill_exact!(fs::FieldSet, case::GHCase, t)

Fill `fs` with the case's exact state at time `t` — the initial data, and
the reference an error norm is taken against.

One line over [`state_callback`](@ref), written out because it is the
call every study makes twice, and because a field set filled from a
*different* layout than the state's would sample the solution at
different points (TreeWave's note on the error field set).
"""
fill_exact!(fs::FieldSet, case::GHCase, t; interior=case.interior) =
    fill_by_coordinates!(state_callback(case, t; interior=interior), fs)
