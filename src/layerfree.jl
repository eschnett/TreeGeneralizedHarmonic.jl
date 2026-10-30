# A layer-free interior (step I1, `CODE.md`, "A layer-free interior").
#
# The moving-puncture way of holding a hole, in this package's formulation: no
# damping layer and no analytic target. The Einstein equations run everywhere;
# the interior is made harmless by its **data** — the smoothed Kerr-Schild hole
# (`SmoothedKerrSchild`, `initialdata.jl`), which takes the singularity off the
# initial slice — and by its **gauge**, a source steered toward a collapsed lapse
# inside the horizon (`SteeredSource`); the range projection (`bounds.jl`) is the
# guard. What this file holds:
#
#   * `LayerFreeInterior`, the `isbits` object a case carries in place of a
#     layer: the steering's radii, the data's pre-collapse, an optional anchor,
#     and the rules its radii obey on a mesh (`check_interior_radii`,
#     `check_bounds_gate`). Its kernel variant is `:none`, or `:anchored` with the
#     anchor; there is no frozen core and no `r_1`, so every mask is `AllPoints`.
#   * `SteeredSource`, the algebraic Kerr-Schild source plus the steering, and
#     `CorrectedSource`, the same with the frozen correction `ΔH`, sampled once
#     and read behind the gauge-source kind `Val(:corrected)`.
#   * the initial data with a pre-collapsed lapse (`layerfree_state`), the
#     validity rows by region, the hits by kind, and the per-shell diagnostics the
#     screen's record reads (`layerfree_diagnostics!`).

# --- the interior ---------------------------------------------------------------

"""
    LayerFreeInterior(T = Float64; r_s0, r_s1, r_core = 0, ρ_a = 0, κ = 1,
                      margin = 8, ε_cells = 8, gate_rule = true)

The layer-free interior of step I1 (`CODE.md`, "A layer-free interior"): what a
case carries where a hole's layer would be, `GHCase(…; interior)` or
[`hole_case`](@ref)`(…; interior)`, which take no `r_0` and no `r_1` beside it.

- `r_s0 < r_s1` are the **steering's** radii: its weight `ψ` is `1` inside
  `r_s0` and `0` outside `r_s1` ([`steering_profile`](@ref)). The source itself is
  the case's [`SteeredSource`](@ref), which carries the same two radii; the
  interior holds them for the data, the rules and the record.
- `κ` is the **data's** lapse factor: the initial lapse is the smoothed hole's
  times `1 − (1 − κ) ψ`, with `γ` and `K` kept and `Π` from
  [`gauge_consistent_momentum`](@ref) (`κ = 1`: the smoothed data as they are).
- `ρ_a > 0` is the **anchor**, `−ρ_a χ(r) (u − u_s)` toward the smoothed state
  inside the smoothing radius `ε` ([`anchor_weight`](@ref)), with `F` evaluated
  everywhere; its kernel variant is `:anchored`, and `:none` without it.
- `r_core > 0` fills the data inside `r_core` from the sphere `r_core` along the
  ray, step 5's core rule — for the analytic Kerr-Schild hole, whose center is
  singular (the screen's R1); the smoothed hole needs none.
- `margin` and `ε_cells` are the rules' cells: `r_s1 ≤ r_h,min − margin · h`,
  the range projection's gate at `r_gate ≤ r_h,min − margin · h`, and
  `ε ≥ ε_cells · h` ([`check_interior_radii`](@ref)). `gate_rule = false` lifts
  the gate's rule, for the screen's row that projects everywhere.

The center, the smoothing radius `ε` and the horizon's least radius are the
case's: [`GHCase`](@ref) builds the interior about its own center and reads `ε`
([`smoothing_radius`](@ref)) and `r_h,min` ([`horizon_min_radius`](@ref)) from the
background. `isbits`: a kernel argument.
"""
struct LayerFreeInterior{T,V}
    center::HoleCenter{T}
    ε::T             # the data's smoothing radius, the background's (0: none)
    r_core::T        # the core rule's radius (0: none)
    r_s0::T
    r_s1::T
    ρ_a::T           # the anchor's rate (0: none)
    κ::T             # the data's lapse factor at the center (1: none)
    margin::Int
    ε_cells::Int
    gate_rule::Bool
    r_h::T           # the horizon's least radius, the background's (0: not a case's)
end

function LayerFreeInterior(::Type{T}=Float64; r_s0, r_s1, r_core=zero(T), ρ_a=zero(T),
                           κ=one(T), margin::Integer=8, ε_cells::Integer=8,
                           gate_rule::Bool=true, center=(zero(T), zero(T), zero(T)),
                           ε=zero(T)) where {T}
    0 < T(r_s0) < T(r_s1) || throw(ArgumentError(
        "the steering's weight is 1 inside r_s0 and 0 outside r_s1, so it needs " *
        "0 < r_s0 < r_s1, got r_s0 = $r_s0 and r_s1 = $r_s1."))
    T(ρ_a) ≥ 0 || throw(ArgumentError(
        "the anchor's rate is a relaxation rate and must be non-negative (0: no " *
        "anchor), got ρ_a = $ρ_a."))
    0 < T(κ) ≤ 1 || throw(ArgumentError(
        "the data's lapse factor κ scales the lapse at the center down, " *
        "0 < κ ≤ 1 (1: the smoothed data as they are), got κ = $κ."))
    T(r_core) ≥ 0 || throw(ArgumentError("the core rule's radius is ≥ 0, got $r_core."))
    margin ≥ 1 && ε_cells ≥ 1 || throw(ArgumentError(
        "the rules' cells are positive, got margin = $margin, ε_cells = $ε_cells."))
    c = center isa HoleCenter ? HoleCenter{T}(SVector{3,T}(center.c0),
                                              SVector{3,T}(center.v)) :
        HoleCenter(T, center)
    V = T(ρ_a) > 0 ? :anchored : :none
    return LayerFreeInterior{T,V}(c, T(ε), T(r_core), T(r_s0), T(r_s1), T(ρ_a), T(κ),
                                  Int(margin), Int(ε_cells), gate_rule, zero(T))
end

# What `GHCase` makes of the interior it is given: the same rule about the case's
# center, with the background's smoothing radius, and the refusals that need the
# case.
function layerfree_case_interior(int::LayerFreeInterior, c::HoleCenter,
                                 ::Type{T}; target, refinement,
                                 background) where {T}
    int isa LayerFreeInterior{T} || throw(ArgumentError(
        "a case's LayerFreeInterior is stated in the case's own type $T, got a " *
        "$(typeof(int))."))
    target === nothing || throw(ArgumentError(
        "a layer-free interior has no layer, so no layer target: leave `target = " *
        "nothing`. Its anchor relaxes toward the background itself."))
    refinement === nothing || throw(ArgumentError(
        "a layer-free interior has no level floor for a refinement to keep, and none " *
        "is built (step I1 runs on frozen nested cubes): leave `refinement = " *
        "nothing`."))
    ε = T(smoothing_radius(background))
    V = interior_variant(int)
    V === :anchored && !(ε > 0) && throw(ArgumentError(
        "the anchor relaxes toward the smoothed state inside the smoothing radius ε, " *
        "and this background is not smoothed: use a SmoothedKerrSchild, or ρ_a = 0."))
    (ε > 0 || int.r_core > 0) || throw(ArgumentError(
        "a layer-free interior evaluates the equations at every point, the hole's " *
        "center included, and this background is singular there: smooth it " *
        "(SmoothedKerrSchild) or fill a core by the core rule (r_core > 0)."))
    return LayerFreeInterior{T,V}(c, ε, int.r_core, int.r_s0, int.r_s1, int.ρ_a, int.κ,
                                  int.margin, int.ε_cells, int.gate_rule,
                                  T(horizon_min_radius(background)))
end

interior_variant(::LayerFreeInterior{T,V}) where {T,V} = V

@inline function interior_radius(int::LayerFreeInterior, t, x)
    c = center_at(int.center, t)
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    return sqrt(d1 * d1 + d2 * d2 + d3 * d3)
end

# No layer: no point is in one, every point is evolved, and nothing is pasted.
@inline in_layer(::LayerFreeInterior, t, x) = false
interior_mask(::LayerFreeInterior, t) = AllPoints()

# The core rule, where the data have a core (R1's analytic hole), and the
# identity otherwise.
@inline function core_position(int::LayerFreeInterior{T}, t, x) where {T}
    iszero(int.r_core) && return x
    c = center_at(int.center, t)
    d = SVector{3}(x[1] - c[1], x[2] - c[2], x[3] - c[3])
    r = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    r ≥ int.r_core && return (x[1], x[2], x[3])
    ẑ = SVector{3}(zero(r), zero(r), one(r))
    n = iszero(r) ? ẑ : d / r
    return (c[1] + int.r_core * n[1], c[2] + int.r_core * n[2],
            c[3] + int.r_core * n[3])
end

# The anchor's support, which is where the lanes fall back to the scalar code.
@inline outside_all(int::LayerFreeInterior, t, x) = interior_radius(int, t, x) ≥ int.ε

# The smoothed ball (or the core), the steering's ramp and the band between the
# ramp and the horizon: the three regions the validity rows read (step I1).
"""
    layerfree_ball(interior) -> the radius of the smoothed ball (or the core)
"""
layerfree_ball(int::LayerFreeInterior) = max(int.ε, int.r_core)
layer_mask(int::LayerFreeInterior{T}, t) where {T} =
    ShellMask{T}(center_at(int.center, t), zero(T), layerfree_ball(int))
shell_mask(int::LayerFreeInterior{T}, t, width) where {T} =
    ShellMask{T}(center_at(int.center, t), int.r_s0, int.r_s1)
band_mask(int::LayerFreeInterior{T}, t, r_h) where {T} =
    ShellMask{T}(center_at(int.center, t), int.r_s1, T(r_h))

# There is no layer rate: the anchor's is its own, and `chunk_interior` leaves
# the interior as it is.
with_ρ_max(int::LayerFreeInterior, ρ) = int
interior_ρ_max(int::LayerFreeInterior) = int.ρ_a

# The drift shell: from the horizon's least radius to its largest plus the
# steering ramp's width, as step 5's adds the layer's.
function horizon_shell(case::GHCase{T}, int::LayerFreeInterior) where {T}
    lo = T(horizon_min_radius(case.background))
    hi = T(horizon_max_radius(case.background)) + (int.r_s1 - int.r_s0)
    return (lo, hi)
end

"""
    anchor_weight(interior, t, x) -> χ

The anchor's profile at `x` (proposed in step I1): `1` inside `ε/2`, `0` from
`ε` out, the quintic [`smoothstep`](@ref) between — `C²`, so the anchor is a
smooth term of `F` wherever it acts. `0` without an anchor.
"""
@inline function anchor_weight(int::LayerFreeInterior{T}, t, x) where {T}
    iszero(int.ρ_a) && return zero(T)
    r = interior_radius(int, t, x)
    return one(T) - smoothstep(2 * r / int.ε - one(T))
end

# The coarsest spacing of the blocks meeting the sphere `r = R` about `c`.
function _sphere_spacing(forest::Forest{3}, c, R, ::Type{T}) where {T}
    h = zero(T)
    n = 0
    for k in forest.leaves
        _box_meets_sphere(block_extent(T, forest, k), c, R) || continue
        h = max(h, spacing(T, forest, k))
        n += 1
    end
    return h, n
end

"""
    check_interior_radii(forest, interior::LayerFreeInterior, background, G; t = 0)

Step I1's two rules, in place of the layer's (`CODE.md`, "A layer-free
interior"), with `h` the coarsest spacing of the blocks meeting each sphere:

    r_s1 ≤ r_h,min − margin · h      (margin = 8)
    ε    ≥ ε_cells · h               (ε_cells = 8, where the data are smoothed)

The first keeps the steering's ramp, and everything it does to the slice, eight
cells inside the horizon — `e^{−3.6}` at step 8a's leak per cell. The second is
the ring rule's eight cells across the data's own scale: a smoothed ball the
stencils do not resolve is a kink. It throws, as the layer's checks do.
"""
function check_interior_radii(forest::Forest{3}, int::LayerFreeInterior{T}, background,
                              G::Integer; t=zero(T), center=nothing) where {T}
    int.margin ≥ G + 1 || throw(ArgumentError(
        "the layer-free interior's margin is $(int.margin) cells but the ghost width " *
        "is G = $G: the steering's ramp has to be a whole stencil inside the " *
        "horizon, margin ≥ G + 1 = $(G + 1)."))
    c = center_at(int.center, t)
    h, nb = _sphere_spacing(forest, c, int.r_s1, T)
    nb > 0 || throw(ArgumentError(
        "no block of this forest contains the sphere r_s1 = $(int.r_s1) about $c."))
    r_h = T(horizon_min_radius(background))
    allowed = r_h - int.margin * h
    int.r_s1 ≤ allowed || throw(ArgumentError(
        "the steering's ramp is not far enough inside the horizon: r_s1 = " *
        "$(int.r_s1), but the horizon's least radius is r_h,min = $r_h and the " *
        "blocks there have h = $h, so step I1's r_s1 ≤ r_h,min − $(int.margin) h " *
        "allows at most $allowed. Refine around the hole or move r_s1 inward."))
    hε = zero(T)
    if int.ε > 0
        hε, nε = _sphere_spacing(forest, c, int.ε, T)
        nε > 0 || throw(ArgumentError(
            "no block of this forest contains the smoothing sphere ε = $(int.ε)."))
        int.ε ≥ int.ε_cells * hε || throw(ArgumentError(
            "the smoothed ball is not resolved: ε = $(int.ε) at h = $hε, and step " *
            "I1's rule is ε ≥ $(int.ε_cells) h = $(int.ε_cells * hε) — eight cells " *
            "across the data's own scale, as the ring rule asks of a spinning hole's. " *
            "Refine around the hole or smooth over a larger ε."))
    end
    return (h=h, nblocks=nb, r_h_min=r_h, allowed=allowed, h_ε=hε)
end

"""
    check_bounds_gate(forest, interior::LayerFreeInterior, bounds, q; t = 0)

The layer-free gate rule (step I1): `r_gate ≤ r_h,min − margin · h`, with `h` the
coarsest spacing of the blocks meeting the sphere `r_gate`. There is no `r_1`,
and the projection's kink is read by evolved stencils here on purpose; the
margin keeps it, at step 8a's leak, about `e^{−3.6}` from the horizon. Lifted
with `gate_rule = false` (the screen's row with the gate everywhere).
"""
function check_bounds_gate(forest::Forest{3}, int::LayerFreeInterior{T},
                           bounds::StateBounds{T}, q::Integer; t=zero(T)) where {T}
    int.gate_rule || return nothing
    int.r_h > 0 || throw(ArgumentError(
        "this layer-free interior was not built by a case, so it has no horizon to " *
        "hold the gate against: pass it to GHCase or hole_case, which read r_h,min " *
        "from the background."))
    h, nb = _sphere_spacing(forest, center_at(int.center, t), bounds.r_gate, T)
    nb > 0 || throw(ArgumentError(
        "no block of this forest contains the gate's sphere r_gate = $(bounds.r_gate)."))
    allowed = int.r_h - int.margin * h
    bounds.r_gate ≤ allowed || throw(ArgumentError(
        "the range projection's gate is too close to the horizon: r_gate = " *
        "$(bounds.r_gate), the horizon's least radius r_h,min = $(int.r_h) at h = $h, " *
        "and step I1's rule is r_gate ≤ r_h,min − $(int.margin) h = $allowed: the " *
        "projection's kink is read by evolved stencils here on purpose, and the " *
        "margin is what keeps it inside the horizon. Move r_gate inward, or lift the " *
        "rule by name (gate_rule = false) for a run that projects everywhere."))
    return (h=h, allowed=allowed)
end

check_case_bounds(::Nothing, ::LayerFreeInterior, ::Type) = nothing
function check_case_bounds(bd::StateBounds, ::LayerFreeInterior, ::Type{T}) where {T}
    bd isa StateBounds{T} || throw(ArgumentError(
        "a case's bounds are a StateBounds{$T} or `nothing`, got a $(typeof(bd))."))
    return nothing
end

_apply_bounds!(bd::StateBounds, int::LayerFreeInterior, p, u, t) =
    _apply_bounds_kernel!(bd, int, p, u, t)

"""
    layerfree_bounds(T = Float64; r_gate, M = 1, α_min = 1//1000, α_max = 2,
                     λ_min = 1//1000, λ_max = 1000, β_max = 3, K_max = 100)

The ranges step I1 proposes for the layer-free interior's guard (`CODE.md`, "A
layer-free interior"): `α ∈ [10⁻³, 2]` (TreeBSSN's working floor), `λ(γ) ∈ [10⁻³,
10³]`, `|β| ≤ 3` (the CFL condition's coordinate speed), `max |(α/√γ)Π| ≤
100/M` (`default_bounds`' value) — `K_max` in units of `1/M` — gated at `r_gate`.
Each is a keyword so that the screen's rows can move one (R15, R16).
"""
layerfree_bounds(::Type{T}=Float64; r_gate, M=1, α_min=1 // 1000, α_max=2,
                 λ_min=1 // 1000, λ_max=1000, β_max=3, K_max=100) where {T} =
    StateBounds{T}(T(α_min), T(α_max), T(λ_min), T(λ_max), T(β_max), T(K_max) / T(M),
                   T(r_gate))

# --- the hits by kind and by radius -------------------------------------------------

struct BoundsDetailFold{T}
    inv_h::T
end

@inline function (f::BoundsDetailFold)(acc, p)
    T = typeof(p.h)
    hit = !iszero(p.u[1])
    k = hit ? Base.unsafe_trunc(Int, p.u[4]) : 0
    s = one(T)
    z = zero(T)
    kinds = (Sum(hit & ((k & BOUNDS_KIND_NONFINITE) != 0) ? s : z),
             Sum(hit & ((k & BOUNDS_KIND_SPECTRUM) != 0) ? s : z),
             Sum(hit & ((k & BOUNDS_KIND_SHIFT) != 0) ? s : z),
             Sum(hit & ((k & BOUNDS_KIND_LAPSE) != 0) ? s : z),
             Sum(hit & ((k & BOUNDS_KIND_MOMENTUM) != 0) ? s : z))
    j = hit ? min(Base.unsafe_trunc(Int, p.u[3] * f.inv_h), BOUNDS_HIST_BINS - 1) + 1 : 0
    # Two halves of the bins: `combine` maps over a tuple, and `map` over more than
    # 32 elements allocates, which a device refuses (found on an H200 after step I1).
    H = BOUNDS_HIST_BINS ÷ 2
    return (kinds=TreeAMR.combine(acc.kinds, kinds),
            hist_lo=addto(acc.hist_lo, j, Sum(s)),
            hist_hi=addto(acc.hist_hi, j - H, Sum(s)))
end

function _bounds_detail!(acc, int::LayerFreeInterior, p)
    acc isa BoundsAccounting || return nothing
    T = eltype(p.U.work)
    z = Sum(zero(T))
    H = BOUNDS_HIST_BINS ÷ 2
    init = (kinds=ntuple(_ -> z, Val(5)), hist_lo=ntuple(_ -> z, Val(H)),
            hist_hi=ntuple(_ -> z, Val(H)))
    f = BoundsDetailFold{T}(inv(minimum_spacing(T, p.U.forest)))
    r = mesh_fold(f, TreeAMR.combine, init, p.diag; vars=DIAG_BOUNDS:DIAG_BOUNDS_KIND)
    for i in 1:5
        acc.chunk_kinds[i] += round(Int, tofloat64(r.kinds[i].value))
    end
    for i in 1:H
        acc.chunk_hist[i] += round(Int, tofloat64(r.hist_lo[i].value))
        acc.chunk_hist[H + i] += round(Int, tofloat64(r.hist_hi[i].value))
    end
    return nothing
end

# The accounting's checkpoint holds its run totals, as it did before step I1: the
# hits by kind and radius are the chunk's, and taken before a checkpoint is written.
const _ACCOUNTING_SAVED = (:calls, :hits, :nonfinite, :r_max, :first_t, :first_r,
                           :chunk_hits, :chunk_nonfinite, :chunk_r_max)

function to_plain(acc::BoundsAccounting; path="run")
    values = ntuple(i -> to_plain(getfield(acc, _ACCOUNTING_SAVED[i]);
                                  path="$path.$(_ACCOUNTING_SAVED[i])"),
                    length(_ACCOUNTING_SAVED))
    return (; kind="BoundsAccounting", names=collect(String.(_ACCOUNTING_SAVED)),
            values=values)
end

function from_plain(::Type{BoundsAccounting}, v; path="run")
    (v isa NamedTuple && get(v, :kind, "") == "BoundsAccounting" &&
     v.names == collect(String.(_ACCOUNTING_SAVED))) || throw(ArgumentError(
        "the checkpoint's run state at $path is not a BoundsAccounting with the " *
        "fields $(collect(_ACCOUNTING_SAVED)): the file was written by another " *
        "version of this package, or is damaged."))
    types = (Int, Int, Int, Float64, Float64, Float64, Int, Int, Float64)
    args = ntuple(i -> from_plain(types[i], v.values[i];
                                  path="$path.$(_ACCOUNTING_SAVED[i])"), 9)
    return BoundsAccounting(args...)
end

# --- the validity rows by region ---------------------------------------------------

struct LayerFreeValidityFold{ML,MS,MB}
    ball::ML
    ramp::MS
    band::MB
end

@inline function (f::LayerFreeValidityFold)(acc, p)
    T = typeof(p.h)
    inl = is_evolved(f.ball, p.x)
    ins = is_evolved(f.ramp, p.x)
    inb = is_evolved(f.band, p.x)
    dγ, α, hm, Πm = _state_validity(p.u)
    q = (layer=_validity_bin(inl, dγ, α, hm, Πm), shell=_validity_bin(ins, dγ, α, hm, Πm),
         band=_validity_bin(inb, dγ, α, hm, Πm),
         evolved=_validity_bin(true, dγ, α, hm, Πm),
         nonfinite=Sum(count(x -> !isfinite(x), p.u)))
    return TreeAMR.combine(acc, q)
end

"""
    layerfree_validity(p, u, t) -> (rows, nonfinite)

The validity monitor of a layer-free interior (step I1): the record's rows by
region — `_layer` is the smoothed ball (or the core), `_shell` the steering's
ramp `r_s0 ≤ r < r_s1`, `_band` the band `r_s1 ≤ r < r_h,min` inside the
horizon, `_evolved` every point — and the non-finite count over every point,
in one fold.
"""
function layerfree_validity(p, u, t)
    int = p.interior
    T = eltype(p.U.work)
    r_h = horizon_min_radius(p.case.background)
    f = LayerFreeValidityFold(layer_mask(int, T(t)), shell_mask(int, T(t), zero(T)),
                              band_mask(int, T(t), r_h))
    init = (layer=_validity_bin(T), shell=_validity_bin(T), band=_validity_bin(T),
            evolved=_validity_bin(T), nonfinite=Sum(0))
    r = mesh_fold(f, TreeAMR.combine, init, p.U, u)
    l = _validity_report(r.layer)
    s = _validity_report(r.shell)
    b = _validity_report(r.band)
    e = _validity_report(r.evolved)
    rows = (min_detγ_layer=l.detγ, min_α_layer=l.α, max_h_layer=l.h, max_Π_layer=l.Π,
            min_detγ_shell=s.detγ, min_α_shell=s.α, max_h_shell=s.h, max_Π_shell=s.Π,
            min_detγ_evolved=e.detγ, min_α_evolved=e.α,
            min_detγ_band=b.detγ, min_α_band=b.α, max_h_band=b.h, max_Π_band=b.Π)
    return (rows, r.nonfinite.value)
end

# --- the smoothed hole's lapse and volume, in closed form ------------------------------

# Whether any lane holds: a `Bool` is its own answer, a lane mask any of its
# lanes (`lanes.jl`).
@inline _any_lane(c::Bool) = c
@inline _any_lane(c::Vec{W,Bool}) where {W} = any(c)

"""
    smoothed_logs(M, ε, r²) -> (log α_s, log √γ_s, ∂ log α_s/∂(r²), ∂ log √γ_s/∂(r²))

The smoothed Kerr-Schild hole's lapse and volume element at `r² = |x − c|²`, in
closed form (added in step I1), for the steering's volume-coupled target: with
`r_s` [`SmoothedKerrSchild`](@ref)'s radius and `H = M/r_s`,
`α_s = (1 + 2H)^{−1/2}` and `γ_s = 1 + 2M r²/r_s³`. Generic in the number type —
the inside and the outside of `ε` are chosen lane by lane (`_select`), never
branched on — and closure-free, for the kernels. `ε = 0` is the bare Kerr-Schild
hole, singular at `r = 0`.
"""
@inline function smoothed_logs(M, ε, r2::T) where {T}
    ε2 = ε * ε
    inside = r2 < ε2
    s = r2 / ε2
    P = T(35) / T(128) + s * (T(35) / T(32) + s * (-T(35) / T(64) +
                                                   s * (T(7) / T(32) - s * (T(5) / T(128)))))
    dP = T(35) / T(32) + s * (-T(35) / T(32) + s * (T(21) / T(32) - s * (T(5) / T(32))))
    rq = sqrt(r2)
    rs = _select(inside, ε * P, rq)
    drs = _select(inside, dP / ε, one(T) / (2 * rq))      # ∂ r_s/∂(r²)
    A = one(T) + 2 * M / rs
    lα = -log(A) / 2
    dlα = M * drs / (rs * rs * A)
    rs3 = rs * rs * rs
    D = one(T) + 2 * M * r2 / rs3
    lv = log(D) / 2
    dD = 2 * M / rs3 - 6 * M * r2 * drs / (rs3 * rs)
    dlv = dD / (2 * D)
    return lα, lv, dlα, dlv
end

"""
    steering_profile(r_s0, r_s1, r) -> (ψ, ψ′/max(r, r_s0))

The steering's weight (step I1): `1` for `r ≤ r_s0`, `0` for `r ≥ r_s1`, and
`1 − smoothstep` between, `C²`; with its radial derivative divided by
`max(r, r_s0)`, so that `∂_i ψ = (x_i − c_i)` times it is finite at the center,
where it is zero. Generic in the number type, closure-free.
"""
@inline function steering_profile(r_s0, r_s1, r::T) where {T}
    w = r_s1 - r_s0
    u = min(max((r - r_s0) / w, zero(T)), one(T))
    S = u * u * u * (10 + u * (6 * u - 15))
    dS = 30 * u * u * (one(T) - u) * (one(T) - u) / w
    return one(T) - S, -dS / max(r, r_s0)
end

# --- the steered source ---------------------------------------------------------

"""
    SteeredSource(T = Float64; M = 1, ε, r_s0, r_s1, steering = :volume, p = 1,
                  κ = 1, μ_α = 4, μ_β = 0, correction = true, center = (0, 0, 0))

The layer-free interior's gauge source (step I1; `CODE.md`, "A layer-free
interior"): the algebraic Kerr-Schild source of the hole of mass `M`
([`KerrSchildSource`](@ref)), plus — with `correction = true` — the frozen
correction `ΔH = −Γ[g_s] − H^{KS}(h_s)` of the smoothed hole of smoothing radius
`ε`, sampled once ([`CorrectedSource`](@ref)), plus the **steering**

    ψ(r) [μ_α log(α_*/α) n_a − μ_β g_ai β^i/α],    n_a = (−α, 0, 0, 0),

with `ψ` the [`steering_profile`](@ref) between `r_s0` and `r_s1`. The target
lapse is

    α_* = κ α_s(x) (√γ/√γ_s(x))^p,

`α_s` and `√γ_s` the smoothed hole's, in closed form ([`smoothed_logs`](@ref)):
`steering = :volume` is **V**, `κ = 1` and `p ≥ 1` (`p = 1` like harmonic
slicing, `α ∝ √γ`), which vanishes on the smoothed data; `:fixed` is **F**,
`p = 0` and the pre-collapsed `α_* = κ α_s`; `:none` is no steering (`μ_α = μ_β
= 0`). The shift is steered toward `β_* = 0`.

**The signs** are B1's damped harmonic gauge's (`CODE.md`, "The gauge"): on a
slice with `γ` fixed, `C_t = 0` reads `∂_t log α = α μ_α ψ log(α_*/α)`, so the
lapse is driven to `α_*` at the rate `α μ_α`, and `C_i = 0` reads `∂_t β^i =
−α μ_β ψ β^i`. `test/layerfree_tests.jl` checks both through
[`gauge_consistent_momentum`](@ref).

`ψ = 0` from `r_s1` out, through a branch: the exterior's gauge is the
Kerr-Schild source's (the screen compares it with the baseline's). `∂_c H_a`
carries `ψ`'s and the target's explicit gradients and the state's through
[`_adm_rates`](@ref); the code is generic in the number type, for the SIMD
lanes, and closure-free, for a device. `isbits`.
"""
struct SteeredSource{T} <: AlgebraicGaugeSource{T}
    ks::KerrSchildSource{T}
    center::SVector{3,T}
    M::T
    ε::T
    μ_α::T
    μ_β::T
    logκ::T
    p::Int
    r_s0::T
    r_s1::T
    correction::Bool
end

function SteeredSource(::Type{T}=Float64; M=1, ε, r_s0, r_s1, steering::Symbol=:volume,
                       p::Integer=1, κ=1, μ_α=4, μ_β=0, correction::Bool=true,
                       center=(0, 0, 0)) where {T}
    steering in (:volume, :fixed, :none) || throw(ArgumentError(
        "steering is :volume (V, α_* = α_s (√γ/√γ_s)^p), :fixed (F, α_* = κ α_s) " *
        "or :none, got $(repr(steering))."))
    T(ε) > 0 || throw(ArgumentError(
        "the steering's target is the smoothed hole's lapse and volume, which need " *
        "a smoothing radius ε > 0 (the bare hole's are singular at its center), " *
        "got ε = $ε."))
    0 < T(r_s0) < T(r_s1) || throw(ArgumentError(
        "the steering needs 0 < r_s0 < r_s1, got r_s0 = $r_s0, r_s1 = $r_s1."))
    T(μ_α) ≥ 0 && T(μ_β) ≥ 0 || throw(ArgumentError(
        "the steering's rates are non-negative, got μ_α = $μ_α, μ_β = $μ_β: a " *
        "negative one drives the lapse or the shift away from its target."))
    steering === :volume && (T(κ) == 1 && p ≥ 1 || throw(ArgumentError(
        "the volume-coupled target is α_s (√γ/√γ_s)^p with p ≥ 1 and no factor, " *
        "got p = $p, κ = $κ; a factor is the fixed target's (steering = :fixed).")))
    steering === :fixed && (0 < T(κ) ≤ 1 || throw(ArgumentError(
        "the fixed target is κ α_s with 0 < κ ≤ 1, got κ = $κ.")))
    on = steering !== :none
    return SteeredSource{T}(KerrSchildSource(T; M=M), SVector{3,T}(center...), T(M),
                            T(ε), on ? T(μ_α) : zero(T), on ? T(μ_β) : zero(T),
                            steering === :fixed ? log(T(κ)) : zero(T),
                            steering === :volume ? Int(p) : 0, T(r_s0), T(r_s1),
                            correction)
end

convert_source(::Type{S}, src::SteeredSource) where {S} =
    SteeredSource{S}(convert_source(S, src.ks), SVector{3,S}(src.center), S(src.M),
                     S(src.ε), S(src.μ_α), S(src.μ_β), S(src.logκ), src.p, S(src.r_s0),
                     S(src.r_s1), src.correction)
gauge_scale(src::SteeredSource) = src.M

@inline function algebraic_gauge_source(src::SteeredSource, x, t, h::SVector{NC,T},
                                        ∂ₜh::SVector{NC,T},
                                        ∂h::NTuple{3,SVector{NC,T}}) where {T}
    Hk, dHk = algebraic_gauge_source(src.ks, h, ∂ₜh, ∂h)
    d1 = x[1] - T(src.center[1])
    d2 = x[2] - T(src.center[2])
    d3 = x[3] - T(src.center[3])
    r2 = d1 * d1 + d2 * d2 + d3 * d3
    r1 = T(src.r_s1)
    # A branch on the run's constants and on whether any lane is inside the ramp's
    # outer sphere — not an early return between two inlined sources (`CLAUDE.md`,
    # the miscompile of step B1): both sides are this one closed form.
    steers = !(iszero(src.μ_α) & iszero(src.μ_β)) && _any_lane(r2 < r1 * r1)
    Hs, dHs = steers ? _steering(src, d1, d2, d3, r2, h, ∂ₜh, ∂h) :
              (zero(SVector{4,T}), zero(SMatrix{4,4,T}))
    return Hk + Hs, dHk + dHs
end

# The steering term and its gradient at one point (or one group of lanes).
@inline function _steering(src::SteeredSource, d1, d2, d3, r2, h::SVector{NC,T}, ∂ₜh,
                           ∂h) where {T}
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    r = sqrt(r2)
    ψ, gψ = steering_profile(T(src.r_s0), T(src.r_s1), r)
    lα, lv, dlα, dlv = smoothed_logs(T(src.M), T(src.ε), r2)
    p = src.p
    Λ = T(src.logκ) + lα + p * (log(sqrtγ) - lv) - log(α)
    # The explicit gradient of `log α_s − p log √γ_s`, a function of `r²`.
    gx = 2 * (dlα - p * dlv)
    μα = T(src.μ_α)
    μβ = T(src.μ_β)
    βl = SVector{3,T}(h[2], h[3], h[4])
    B2 = βl[1] * β[1] + βl[2] * β[2] + βl[3] * β[3]
    iα = one(T) / α
    μS = μβ * ψ
    Hl = SVector{4,T}(-α * μα * ψ * Λ - μS * B2 * iα, -μS * βl[1] * iα,
                      -μS * βl[2] * iα, -μS * βl[3] * iα)
    z = zero(T)
    rt = _steer_row(gu4, α, β, γu, sqrtγ, βl, B2, ψ, z, Λ, z, μα, μβ, p, ∂ₜh)
    rx = _steer_row(gu4, α, β, γu, sqrtγ, βl, B2, ψ, gψ * d1, Λ, gx * d1, μα, μβ, p, ∂h[1])
    ry = _steer_row(gu4, α, β, γu, sqrtγ, βl, B2, ψ, gψ * d2, Λ, gx * d2, μα, μβ, p, ∂h[2])
    rz = _steer_row(gu4, α, β, γu, sqrtγ, βl, B2, ψ, gψ * d3, Λ, gx * d3, μα, μβ, p, ∂h[3])
    return Hl, _rows4(rt, rx, ry, rz)
end

# One row `∂_c H_a` of the steering, from that direction's `∂g`, the explicit
# gradients `∂_c ψ` and `∂_c (log α_s − p log √γ_s)`.
@inline function _steer_row(gu4, α, β, γu, sqrtγ, βl, B2, ψ, dψ, Λ, dΛx, μα, μβ, p,
                            dg::SVector{NC,T}) where {T}
    dα, dβ, dsqrtγ = _adm_rates(gu4, α, β, γu, sqrtγ, dg)
    iα = one(T) / α
    dΛ = dΛx + p * dsqrtγ / sqrtγ - dα * iα
    dHα = -μα * (dα * ψ * Λ + α * dψ * Λ + α * ψ * dΛ)
    μS = μβ * ψ
    dμS = μβ * dψ
    dB2 = dg[2] * β[1] + dg[3] * β[2] + dg[4] * β[3] +
          βl[1] * dβ[1] + βl[2] * dβ[2] + βl[3] * dβ[3]
    rr = μS * dα * iα * iα                              # from ∂(1/α)
    dHt = dHα - (dμS * B2 + μS * dB2) * iα + rr * B2
    dH1 = -(dμS * βl[1] + μS * dg[2]) * iα + rr * βl[1]
    dH2 = -(dμS * βl[2] + μS * dg[3]) * iα + rr * βl[2]
    dH3 = -(dμS * βl[3] + μS * dg[4]) * iα + rr * βl[3]
    return SVector{4,T}(dHt, dH1, dH2, dH3)
end

# --- the frozen correction ------------------------------------------------------

"""
    has_frozen_correction(src) -> Bool

Whether the case's gauge source carries step I1's frozen correction `ΔH`, which
[`GHProblem`](@ref) then samples and the kernels read behind `Val(:corrected)`:
a [`SteeredSource`](@ref) with `correction = true`, or a roll-on between two
such (whose ends must agree on it). `false` for every other source.
"""
has_frozen_correction(src) = false
has_frozen_correction(src::SteeredSource) = src.correction
function has_frozen_correction(src::RolledGaugeSource)
    a, b = has_frozen_correction(src.from), has_frozen_correction(src.to)
    a == b || throw(ArgumentError(
        "a roll-on between a source with the frozen correction and one without would " *
        "switch ΔH off or on inside the smoothed ball as it rolls; give both ends the " *
        "same `correction`."))
    a && correction_hole(src.from) != correction_hole(src.to) && throw(ArgumentError(
        "a roll-on's two ends share one sampled frozen correction, so they need the " *
        "same hole: M, ε and center."))
    return a
end

# The hole a source's correction is of: its Kerr-Schild part, its center and the
# smoothing radius.
correction_hole(src::SteeredSource) = (src.ks, src.center, src.ε)
correction_hole(src::RolledGaugeSource) = correction_hole(src.from)

"""
    frozen_correction_tuple(background, ks, t, x, center, ε) -> NTuple{20}

`ΔH_b` and `∂_a ΔH_b` at `x` (step I1), in the sampled source's packing
([`gauge_tuple`](@ref)): `ΔH = −Γ[g_s] − H^{KS}(h_s)`, the smoothed hole's own
`−Γ_a` and its gradient, by the nested dual pass the sampled source uses, less
the algebraic Kerr-Schild source `ks` evaluated on the smoothed hole's `h`, `∂ₜh`
and `∂h`. **Exactly zero for `r ≥ ε`, through a branch**, where the smoothed
hole is Kerr-Schild and the algebraic source its `−Γ`: the exterior's source is
the baseline's, bit for bit.
"""
@inline function frozen_correction_tuple(bg, ks, t, x, c, ε)
    D = typeof(x[1])
    d1 = x[1] - c[1]
    d2 = x[2] - c[2]
    d3 = x[3] - c[3]
    r2 = d1 * d1 + d2 * d2 + d3 * d3
    r2 < ε * ε || return @ntuple 20 k -> zero(D)
    p = SVector{4,D}(D(t), x[1], x[2], x[3])
    Hs, dHs = gauge_source_grad(bg, p)
    g, dg = dmetric(bg, p)
    h = pack_g(g)
    ∂ₜh = pack_sym(SMatrix{4,4,D}(dg[a, b, 1] for a in 1:4, b in 1:4))
    ∂h = (pack_sym(SMatrix{4,4,D}(dg[a, b, 2] for a in 1:4, b in 1:4)),
          pack_sym(SMatrix{4,4,D}(dg[a, b, 3] for a in 1:4, b in 1:4)),
          pack_sym(SMatrix{4,4,D}(dg[a, b, 4] for a in 1:4, b in 1:4)))
    Hk, dHk = algebraic_gauge_source(ks, h, ∂ₜh, ∂h)
    return (Tuple(Hs - Hk)..., Tuple(dHs - dHk)...)
end

"""
    CorrectedSource(src, fs, support)

A gauge source with step I1's frozen correction: the algebraic source `src` (a
[`SteeredSource`](@ref), or a roll-on between two) beside the sample `fs` of `ΔH`
and its gradient — a `G = 0` field set of 20 variables, the sampled source's
layout — and `support = (c₁, c₂, c₃, ε)`, the ball outside which `ΔH` is zero.
What a problem's `Hsrc` holds for the gauge-source kind `Val(:corrected)`; a
launch is handed `(effective_source(src, t), fs.work, support)`
([`gauge_work`](@ref)), and the kernels add the sample to the closed form only
where a point (or a lane) is inside the ball.

**Why a sample and not a closed form** (decided in step I1, `CODE.md`): `ΔH`'s
gradient is the smoothed metric's second derivatives, which in the kernel would be
a nested dual pass — a call with its stack on a device, larger than the rest of
the right-hand side, in every launch — and no `Vec` lane can carry a `Dual`. The
sample costs one nested-dual pass at each problem's construction (the static hole
never regrids in the screen) and twenty loads per point inside `ε` only. A moving
hole would need it in closed form.
"""
struct CorrectedSource{S,F,T}
    src::S
    fs::F
    support::SVector{4,T}
end

function corrected_source(case::GHCase{T}, U::FieldSet, t) where {T}
    bg = case.background
    bg isa SmoothedKerrSchild || throw(ArgumentError(
        "the frozen correction ΔH = −Γ[g_s] − H^KS(h_s) is the smoothed hole's, and " *
        "this case's background is a $(typeof(bg)): use a SmoothedKerrSchild, or a " *
        "SteeredSource with correction = false."))
    ks, c, ε = correction_hole(case.gauge)
    ε == bg.ε && ks.M == bg.mass || throw(ArgumentError(
        "the source's frozen correction is of the hole M = $(ks.M), ε = $ε, and the " *
        "background is M = $(bg.mass), ε = $(bg.ε)."))
    fs = FieldSet{T}(U.forest, 2NC; G=0, centering=U.centering,
                     parity=even_parity(U.forest, 2NC),
                     rotation=identity_rotation(U.forest, 2NC),
                     backend=get_backend(U.work))
    sample_frozen_correction!(fs, bg, ks, c, ε, t)
    return CorrectedSource(case.gauge, fs, SVector{4,T}(c[1], c[2], c[3], ε))
end

"""
    sample_frozen_correction!(fs, background, ks, center, ε, t)

Fill `fs` with [`frozen_correction_tuple`](@ref) at every owned point, as one
`fill_by_coordinates!` on the field set's backend — [`sample_gauge_source!`](@ref)
for `ΔH`.
"""
function sample_frozen_correction!(fs::FieldSet{T,3}, bg, ks, c, ε, t) where {T}
    tt = T(t)
    cc = SVector{3,T}(c)
    εε = T(ε)
    fill_by_coordinates!(AllVariables(x -> frozen_correction_tuple(bg, ks, tt, x, cc, εε)),
                         fs)
    return fs
end

gauge_work(c::CorrectedSource, t) = (effective_source(c.src, t), c.fs.work, c.support)

# The lanes read the sample `W` points at a time and the closed form as it is.
@inline lanes(::Val{W}, c::Tuple{AlgebraicGaugeSource,Any,SVector{4}}) where {W} =
    (c[1], lanes(Val(W), c[2]), c[3])

@inline function gauge_source(::Type{T}, Hwork::Tuple, idx, b, ::Val{:corrected}, x, t, h,
                              ∂ₜh, ∂h) where {T}
    H, dH = algebraic_gauge_source(Hwork[1], x, t, h, ∂ₜh, ∂h)
    s = Hwork[3]
    d1 = x[1] - T(s[1])
    d2 = x[2] - T(s[2])
    d3 = x[3] - T(s[3])
    inside = _any_lane(d1 * d1 + d2 * d2 + d3 * d3 < T(s[4] * s[4]))
    # The sample only inside the ball, through a branch; outside it is zero.
    ΔH, dΔH = inside ? gauge_at(T, Hwork[2], idx, b, Val(true)) :
              (zero(SVector{4,T}), zero(SMatrix{4,4,T}))
    return H + ΔH, dH + dΔH
end

# --- the initial data -------------------------------------------------------------

"""
    layerfree_needs_gauge_momentum(case, interior) -> Bool

Whether a layer-free case's data take `Π` from [`gauge_consistent_momentum`](@ref)
(step I1): on the smoothed hole, under an algebraic source, when the lapse is
pre-collapsed (`κ < 1`) or the source is not the smoothed hole's own `−Γ_a` — no
frozen correction (the screen's R9). Otherwise the smoothed metric's own `(h, Π)`
are the data, and `C_a = 0` at `t = 0` because the source is `−Γ[g_s]`.
"""
layerfree_needs_gauge_momentum(case, int::LayerFreeInterior) =
    case.background isa SmoothedKerrSchild && case.gauge !== nothing &&
    (int.κ != 1 || !has_frozen_correction(case.gauge))

"""
    layerfree_state(background, src, interior, t, x) -> (h, Π)

The layer-free interior's data at `x` (step I1): the smoothed hole's `γ_ij`, `K_ij`
— from its static metric, `K = L_β γ/(2α)` — and shift, with the lapse scaled by
`1 − (1 − κ) ψ(r)` (`ψ` the [`steering_profile`](@ref)), and `Π` from
[`gauge_consistent_momentum`](@ref) under the case's source at `t`: its closed form
at the collapsed metric plus, with the correction, the sampled `ΔH`'s value.
`∂_t α` and `∂_t β^i` are then the ones that make `C_a = 0`.
"""
function layerfree_state(bg::SmoothedKerrSchild, src, int::LayerFreeInterior, t, x)
    res = layerfree_momentum(bg, src, int, t, x)
    return res.h, res.Π
end

"""
    layerfree_momentum(background, src, interior, t, x) -> NamedTuple

[`layerfree_state`](@ref)'s whole solve: [`gauge_consistent_momentum`](@ref)'s
result — `Π`, `∂ₜα`, `∂ₜβ`, `h`, `∂ₜh`, `∂h` and the source `Hl` it was solved
for — at the collapsed lapse.
"""
function layerfree_momentum(bg::SmoothedKerrSchild, src, int::LayerFreeInterior, t, x)
    D = typeof(x[1])
    xs = SVector{3,D}(x[1], x[2], x[3])
    p = SVector{4,D}(D(t), xs[1], xs[2], xs[3])
    g, dg = dmetric(bg, p)
    γ = SMatrix{3,3,D}(g[i + 1, j + 1] for i in 1:3, j in 1:3)
    βl = SVector{3,D}(g[1, 2], g[1, 3], g[1, 4])
    γi = inv(γ)
    β = γi * βl
    α = sqrt(sum(βl .* β) - g[1, 1])
    ∂γ = ntuple(k -> SMatrix{3,3,D}(dg[i + 1, j + 1, k + 1] for i in 1:3, j in 1:3),
                Val(3))
    ∂βl = ntuple(k -> SVector{3,D}(dg[1, 2, k + 1], dg[1, 3, k + 1], dg[1, 4, k + 1]),
                 Val(3))
    ∂β = ntuple(k -> γi * (∂βl[k] - ∂γ[k] * β), Val(3))
    ∂α = SVector{3,D}(ntuple(k -> (sum(∂βl[k] .* β) + sum(βl .* ∂β[k]) -
                                   dg[1, 1, k + 1]) / (2 * α), Val(3)))
    J = SMatrix{3,3,D}(∂β[i][k] for k in 1:3, i in 1:3)   # J[k, i] = ∂_i β^k
    K = (β[1] * ∂γ[1] + β[2] * ∂γ[2] + β[3] * ∂γ[3] + J' * γ + γ * J) / (2 * α)
    c = center_at(int.center, t)
    d = xs - SVector{3,D}(c[1], c[2], c[3])
    r = sqrt(sum(abs2, d))
    ψ, gψ = steering_profile(D(int.r_s0), D(int.r_s1), r)
    a = one(D) - D(int.κ)
    f = one(D) - a * ψ
    α′ = f * α
    ∂α′ = f * ∂α - (α * a * gψ) * d
    g′ = SMatrix{4,4,D}((i == 1 && j == 1) ? -α′ * α′ + sum(βl .* β) :
                        i == 1 ? βl[j - 1] : j == 1 ? βl[i - 1] : γ[i - 1, j - 1]
                        for i in 1:4, j in 1:4)
    h′ = pack_g(g′)
    se = effective_source(src, t)
    Hl = _source_value(se, xs, t, h′)
    if has_frozen_correction(src)
        ks, cc, ε = correction_hole(src)
        Δ = frozen_correction_tuple(bg, ks, t, xs, cc, ε)
        Hl = Hl + SVector{4,D}(Δ[1], Δ[2], Δ[3], Δ[4])
    end
    return gauge_consistent_momentum(Hl, xs, t, γ, K, α′, β, ∂γ, ∂α′, ∂β)
end

@inline function layerfree_state_tuple(bg, src, int, t, x)
    h, Π = layerfree_state(bg, src, int, t, x)
    return (Tuple(h)..., Tuple(Π)...)
end

# --- the record's per-shell diagnostics ------------------------------------------------

# The slots of the diagnostics' field set (step I1).
const LF_ROUGH = 1        # the largest undivided sixth difference, any variable, any axis
const LF_STAT = 2         # |∂_t h|, the `h` part of the right-hand side
const LF_ALPHA = 3        # the signed lapse (`state_validity`'s)
const LF_VOLUME = 4       # √γ/√γ_s, the volume against the smoothed hole's
const LF_LAMBDA = 5       # the largest eigenvalue of γ_ij
const LF_NAUX = 5

"""
    layerfree_aux(p) -> FieldSet

The `G = 0` field set of `LF_NAUX` variables [`layerfree_diagnostics!`](@ref)
writes into, on `p`'s forest and backend (step I1). A caller makes one per mesh.
"""
layerfree_aux(p) = FieldSet{eltype(p.U.work)}(p.U.forest, LF_NAUX; G=0,
                                              centering=p.U.centering,
                                              parity=even_parity(p.U.forest, LF_NAUX),
                                              rotation=identity_rotation(p.U.forest,
                                                                         LF_NAUX),
                                              backend=get_backend(p.U.work))

@kernel function layerfree_diag_kernel!(aux, @Const(work), @Const(du), @Const(origins),
                                        @Const(spacings), center, M, ε,
                                        ::Val{G}, ::Val{q}) where {G,q}
    I = @index(Global, NTuple)
    b = I[4]
    inner = (I[1], I[2], I[3])
    T = eltype(aux)
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb +
          (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]
    wD = dissipation_weights(T, dissipation_rank(Val(q)))
    rough = zero(T)
    for v in 1:(2 * NC)
        bv = var + (v - 1) * sv
        rough = max(rough, abs(axis_stencil(wD, work, bv, st[1])),
                    abs(axis_stencil(wD, work, bv, st[2])),
                    abs(axis_stencil(wD, work, bv, st[3])))
    end
    stat2 = zero(T)
    for v in 1:NC
        stat2 += du[inner..., v, b] * du[inner..., v, b]
    end
    hv = SVector{NC,T}(@ntuple 10 v -> (@inbounds work[var + (v - 1) * sv]))
    s = _adm_split(_sym4(hv))
    α = !(s.detγ > zero(T)) ? floatmax(T) :
        s.α² ≥ zero(T) ? sqrt(s.α²) : -sqrt(-s.α²)
    x = point_position(origins, spacings, b, I)
    d1 = x[1] - center[1]
    d2 = x[2] - center[2]
    d3 = x[3] - center[3]
    r2 = d1 * d1 + d2 * d2 + d3 * d3
    vol = if r2 > zero(T) || ε > zero(T)
        _, lv, _, _ = smoothed_logs(M, ε, r2)
        sqrt(max(s.detγ, zero(T))) / exp(lv)
    else
        one(T)
    end
    γ11, γ12, γ13, γ22, γ23, γ33 = s.γ
    λ, _ = sym_eigen3(SMatrix{3,3,T}(γ11, γ12, γ13, γ12, γ22, γ23, γ13, γ23, γ33))
    aux[inner..., LF_ROUGH, b] = rough
    aux[inner..., LF_STAT, b] = sqrt(stat2)
    aux[inner..., LF_ALPHA, b] = α
    aux[inner..., LF_VOLUME, b] = vol
    aux[inner..., LF_LAMBDA, b] = max(λ[1], λ[2], λ[3])
end

struct LayerFreeShellFold{T,NS}
    center::SVector{3,T}
    edges::NTuple{NS,Tuple{T,T}}
end

@inline _lf_bin(::Type{T}) where {T} =
    (n=Sum(zero(T)), stat2=Sum(zero(T)), stat=Max(zero(T)), α=Min(floatmax(T)),
     vol=Min(floatmax(T)), λ=Max(zero(T)), rough=Max(zero(T)))

@inline function (f::LayerFreeShellFold{T,NS})(acc, p) where {T,NS}
    u = p.u
    q = (n=Sum(one(T)), stat2=Sum(u[LF_STAT] * u[LF_STAT]), stat=Max(u[LF_STAT]),
         α=Min(u[LF_ALPHA]), vol=Min(u[LF_VOLUME]), λ=Max(u[LF_LAMBDA]),
         rough=Max(u[LF_ROUGH]))
    k = sum(ntuple(i -> is_evolved(ShellMask(f.center, f.edges[i]...), p.x) ? i : 0,
                   Val(NS)); init=0)
    return addto(acc, k, q)
end

"""
    layerfree_diagnostics!(aux, du, p, u, t; edges) -> Vector{NamedTuple}

The layer-free screen's per-shell rows (step I1; `CODE.md`, "A layer-free
interior"): the right-hand side of `u` into `du` (which fills the ghosts), one
kernel into `aux` and one fold, binned into the shells `edges[k] ≤ r <
edges[k + 1]` about the case's center. Each shell's row holds its point count,
the L2 and L∞ of `|∂_t h|` (the `h` part of the right-hand side: stationarity),
the least lapse, the least `√γ/√γ_s` (how far the volume has collapsed against
the smoothed hole's), the largest eigenvalue of `γ_ij` and the largest undivided
sixth difference of any variable along any axis — TreeBSSN's roughness, the
Kreiss–Oliger operator of `q = 4` without its `ε/h`: `O(h⁶)` on smooth data, so
the grid-scale part.
"""
function layerfree_diagnostics!(aux, du, p, u, t; edges)
    T = eltype(p.U.work)
    int = p.interior
    gh_rhs!(du, u, p, T(t))
    c = center_at(p.case.center, T(t))
    bg = p.case.background
    map_blocks!(layerfree_diag_kernel!, aux, aux.work, p.U.work, statearray(du, p.U),
                p.origins, p.spacings, SVector{3,T}(c), T(hole_mass(bg)),
                T(smoothing_radius(bg)), p.valG, p.valq)
    e = [T(x) for x in edges]
    NS = length(e) - 1
    bands = ntuple(i -> (e[i], e[i + 1]), NS)
    f = LayerFreeShellFold{T,NS}(SVector{3,T}(c), bands)
    r = mesh_fold(f, TreeAMR.combine, ntuple(_ -> _lf_bin(T), NS), aux;
                  vars=1:LF_NAUX)
    return [let b = r[i], n = b.n.value
                (points=n, stat_l2=n > 0 ? sqrt(b.stat2.value / n) : zero(T),
                 stat_linf=b.stat.value, α_min=b.α.value, vol_min=b.vol.value,
                 λ_max=b.λ.value, rough=b.rough.value)
            end for i in 1:NS]
end
