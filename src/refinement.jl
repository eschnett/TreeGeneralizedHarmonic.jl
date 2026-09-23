# The refinement criterion: a per-cell Löhner indicator on `h`, masked
# inside the interior, with a level floor around the horizon and a level
# ceiling at the outer boundary — reduced to one flag per block.
#
# `CODE.md`, "Refinement and regridding". The split is TreeAMR's: the mesh
# takes a flag per block and says that "per-cell criteria are reduced to a
# block verdict inside the application's flag function", so the indicator
# lives here and so does the reduction. TreeWave's `src/refinement.jl` is
# the template and most of the reasoning below is its, transplanted. What
# differs, and why:
#
#   * **The fields are the ten components of `h`, and their noise floor is
#     one global amplitude.** TreeWave's lesson — a floor that shrinks with
#     the data refines the numerical dust — with one change this package
#     needs and which is written out at [`field_scales`](@ref): the floor
#     is the largest `|h|` over *all ten* components rather than one per
#     component, because a chart in which a component vanishes identically
#     (the gauge wave's `h_xy`) is exactly the case a per-component floor
#     of zero fails on.
#   * **The indicator is masked inside `r_1`.** The damping layer and the
#     frozen core are not a numerical solution — the core holds stale data
#     with steep, meaningless differences — so `τ` is zero there and blocks
#     wholly inside the interior are free to coarsen. The layer is resolved
#     from *outside*: the evolved points just beyond `r_1` sit in the
#     steepest part of the metric and the indicator refines them.
#   * **A level floor and a level ceiling**, both pure functions of
#     position, applied as `clamp(request, floor, ceiling)`. The floor is
#     what makes the interior's two radius requirements hold by
#     construction rather than by the indicator's mood; the ceiling keeps
#     the Dirichlet boundary — whose truncation-order mismatch with the
#     numerical solution is a kink the indicator scores — at the coarsest
#     level. Because the bounds must have the last word, the **travelling
#     margin is dilated here** and `regrid!` is handed `buffer = 0`: see
#     [`refine_flags`](@ref) for what happens when it is not.
#   * **One evaluation of `τ`, into `diag`.** TreeWave keeps a host loop
#     beside its `firing_boxes` form; TreeHydro keeps only the second. This
#     package writes `τ` into the `diag` field set with a kernel of its own
#     and then runs `firing_boxes` over *that* — see [`gh_tau!`](@ref) for
#     why (the analysis record asks for `τ_max`, which a firing count
#     cannot give).
#
# What refinement is for is **resolution**, not amplitude. The differences
# are undivided, so the spacing enters implicitly: `τ` measures how well the
# *mesh* represents the data it holds. Around a hole `h` falls off as `M/r`
# and the ratio itself falls like `h/r`, so a fixed threshold produces
# nested shells of refinement around the hole — coarser with distance —
# without being told to, and refinement terminates on its own.

"""
    Refinement(T = Float64; refine_tol, coarsen_tol, maxlevel_cap,
               floor_margin, ceiling_cells = 1, ceiling_level = 0,
               ε = 1//100)

The refinement's parameters, as a case carries them: the two thresholds,
the cap the indicator may reach, and the two geometric corrections.

`CODE.md`, "Refinement and regridding" lists these as fields of the case;
they are one struct rather than five fields **(proposed in step 6)** so
that a case without refinement carries one `nothing` rather than five
placeholders, and so that a test may build a second set of thresholds for
one call without rebuilding the case.

**No default for the thresholds, the cap or the floor's margin**, because
each is a number the caller must think about (`PLAN.md`'s rule for a driver
signature):

- `refine_tol` is "under-resolved here" and `coarsen_tol` is "there is
  something here at all"; the gap between them is the hysteresis dead band.
  Both are calibrated against the mesh — `τ_max` on uniform meshes at
  successive `h`, tabulated under "Measured results" in `CODE.md` — and
  Löhner's canonical `0.8` is a shock detector, not a threshold for smooth
  data.
- `maxlevel_cap` is named a cap rather than a `maxlevel` because it is one:
  at calibrated tolerances the indicator terminates refinement on its own
  and the cap never binds. The name also keeps it from shadowing TreeAMR's
  exported `maxlevel` query inside a body that wants both
  (`CLAUDE.md`).
- `floor_margin` is how far beyond `r_h,max` the level floor's shell
  reaches — `CODE.md`'s "a few `M`". It has no default because "a few `M`"
  is a fraction of a production box and the whole of a test box, and the
  floor's cost is the volume of that shell.

`ceiling_cells` is the ceiling's margin in **coarse** cells (the spacing at
level 0), and `ceiling_level` the level blocks inside it are capped at.
`ε` is [`lohner`](@ref)'s noise-floor coefficient.

The struct is `isbits`, because the case is a kernel argument at every
ghost fill; nothing in it reaches a kernel today, and that is not a reason
to make the case stop being `isbits`.
"""
struct Refinement{T}
    refine_tol::T
    coarsen_tol::T
    maxlevel_cap::Int
    floor_margin::T
    ceiling_cells::Int
    ceiling_level::Int
    ε::T
end

function Refinement(::Type{T}=Float64; refine_tol, coarsen_tol, maxlevel_cap,
                    floor_margin, ceiling_cells::Integer=1,
                    ceiling_level::Integer=0, ε=T(1 // 100)) where {T}
    rtol, ctol = T(refine_tol), T(coarsen_tol)
    0 < ctol < rtol || throw(ArgumentError(
        "the thresholds must satisfy 0 < coarsen_tol < refine_tol, but " *
        "coarsen_tol = $ctol and refine_tol = $rtol. The gap between them is " *
        "the hysteresis dead band: a block refines above refine_tol and " *
        "coarsens only once every cell has dropped below coarsen_tol, so with " *
        "the two equal a block sitting at the threshold refines and coarsens " *
        "on alternate regrids and the hierarchy never settles."))
    rtol < 1 || throw(ArgumentError(
        "refine_tol = $rtol, but τ ∈ [0, 1] by construction — the second " *
        "difference is bounded by the two first differences — so a threshold " *
        "at or above 1 can never fire. Löhner's canonical 0.8 is a shock " *
        "detector; smooth data scores far lower, and the thresholds for this " *
        "package are calibrated (CODE.md, \"Measured results\")."))
    maxlevel_cap ≥ 0 || throw(ArgumentError(
        "maxlevel_cap must be non-negative, got $maxlevel_cap; zero is the " *
        "uniform control run, in which the criterion can refine nothing."))
    T(floor_margin) ≥ 0 || throw(ArgumentError(
        "the level floor's shell reaches from r_1 out to r_h,max + " *
        "floor_margin, so the margin cannot be negative, got $floor_margin."))
    ceiling_cells ≥ 0 || throw(ArgumentError(
        "the ceiling's margin is a number of coarse cells and cannot be " *
        "negative, got $ceiling_cells; zero is a ceiling that binds only on " *
        "blocks touching the boundary."))
    0 ≤ ceiling_level ≤ maxlevel_cap || throw(ArgumentError(
        "the ceiling's level must lie between 0 and maxlevel_cap = " *
        "$maxlevel_cap, got $ceiling_level: CODE.md caps the blocks within a " *
        "few coarse cells of the outer boundary at the *coarsest* level, and " *
        "a ceiling above the cap is not a ceiling."))
    T(ε) > 0 || throw(ArgumentError(
        "the Löhner noise floor's coefficient must be positive, got $ε: with " *
        "ε = 0 the indicator is scale-free and scores τ ≈ 1 on numerical " *
        "dust, which is the failure TreeWave measured and the global floor " *
        "exists to prevent."))
    return Refinement{T}(rtol, ctol, Int(maxlevel_cap), T(floor_margin),
                         Int(ceiling_cells), Int(ceiling_level), T(ε))
end

"""
    lohner(um, u0, up, scale; ε = 1//100)

The Löhner error indicator for three consecutive values of one variable
along one dimension: the second difference normalized by the first
differences and a noise floor,

    τ = |u₊ − 2u₀ + u₋| / (|u₊ − u₀| + |u₀ − u₋| + 4 ε · scale)

Undivided differences, so the spacing enters implicitly and `τ ∈ [0, 1]`:
the numerator is bounded by the two first differences. For smooth data `τ`
falls as `h` shrinks, which is what makes refinement terminate on its own.
A zero denominator gives zero rather than a `NaN`.

`scale` is a **global** reference amplitude, and it is what makes the
indicator usable. Löhner's classic form floors the denominator with the
local `ε(|u₊| + 2|u₀| + |u₋|)`, which is scale-free and therefore cannot
tell a real feature from a ripple in the numerical dust: TreeWave measured
three consecutive tail values ten orders of magnitude below its peak
scoring `τ = 0.986`, because the floor shrinks along with them, and a
criterion that refined the whole domain at every threshold tried. The
fields here — the ten components of `h` — cross zero and decay as `M/r`,
so that failure mode is available and the global floor is not optional.
See [`field_scales`](@ref).

`ε` defaults to `1//100` converted to the value's own type rather than to
the literal `0.01`: a `Float64` literal in the denominator would drag every
`τ` into `Float64` however the field is stored (`CODE.md`, "Precision,
threads, devices"). `CODE.md` writes the floor as `ε_g · U_ref`; this is
TreeWave's spelling of the same number, `ε_g = 4ε`.

!!! note "Not the canonical threshold"
    Löhner's usual `τ > 0.8` is a shock detector. Smooth data never comes
    close, so the thresholds for a problem like this one are much smaller
    and are calibrated against the mesh — the table is under "Measured
    results" in `CODE.md`.
"""
@inline function lohner(um, u0, up, scale; ε=oftype(float(u0), 1 // 100))
    num = abs(up - 2 * u0 + um)
    den = abs(up - u0) + abs(u0 - um) + 4 * ε * abs(scale)
    return iszero(den) ? zero(num) : num / den
end

"""
    cell_tau(work, base, st, sv, scale, ε) -> τ

The worst [`lohner`](@ref) indicator over the ten components of `h` and the
three dimensions at one point: `work` the ghost-inclusive working array,
`base` the point's **linear** index in it, `st` the per-axis strides and
`sv` the per-variable one ([`work_strides`](@ref)).

The whole of the criterion's arithmetic, in one function, because it is
the one place `τ` is defined. It reads variables `1:NC` — the offset metric
`h_ab` — and not `Π`: **(proposed in step 6)** `CODE.md` leaves `Π` as a
switch, and it is left off, because `Π` is a *time* derivative whose
structure around a static hole is that of `h` and whose amplitude on a
noisy state is the state's error rather than its features. A case where the
two disagree would be one where the momentum carries a feature the metric
does not, which nothing in the proof of concept produces.

The linear-index form is the package's, for the reason under "the
kernel-side stencil contractions" in `evolution.jl`: five-dimensional index
arithmetic at every load is not arithmetic this scheme is about. Everything
it takes is `isbits` — two integers, a tuple of three, and two scalars — so
it is legal inside a kernel.
"""
@inline function cell_tau(work, base::Int, st::NTuple{3,Int}, sv::Int, scale, ε)
    τ = zero(scale)
    for v in 1:NC
        i0 = base + (v - 1) * sv
        u0 = @inbounds work[i0]
        for d in 1:3
            up = @inbounds work[i0+st[d]]
            um = @inbounds work[i0-st[d]]
            τ = max(τ, lohner(um, u0, up, scale; ε=ε))
        end
    end
    return τ
end

"""
    gh_scale_kernel!(out, work, origins, spacings, mask, ::Val{G})

`|h_v|` at every owned point, masked to the evolved region — the field
[`field_scales`](@ref) reduces.

A branch and not a multiplication by the mask, for the reason every masked
slot in this package is written through a branch: the frozen core holds
stale data on which `h` may be `NaN`, and `0 · NaN = NaN` would poison the
maximum and with it every `τ` on the mesh.
"""
@kernel function gh_scale_kernel!(out, @Const(work), @Const(origins),
                                  @Const(spacings), mask, ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    c = ntuple(d -> I[d] + G[d], Val(3))
    T = eltype(out)

    keep = is_evolved(mask, point_position(origins, spacings, b, I))
    ntuple(Val(NC)) do v
        out[inner..., v, b] = keep ? abs(work[c..., v, b]) : zero(T)
        nothing
    end
end

"""
    field_scales(U::FieldSet, mask, origins, spacings) -> NTuple{NC}
    field_scale(U::FieldSet, mask, origins, spacings) -> T

The largest `|h_v|` over the **evolved** points, per component and (the
second form) over all ten — the global reference amplitude
[`lohner`](@ref) refers its noise floor to.

`CODE.md`: "`U_ref` is the largest `|h|` in the *evolved* region at
`t = 0`". Masked, because the frozen core holds the analytic solution on
the sphere `r_0`, where `|h|` is several times its largest evolved value:
a reference taken from the core would inflate the floor by a factor that
depends on `r_0`, which is a parameter of the layer and not a property of
the solution.

**The indicator uses one amplitude for all ten components
(proposed in step 6)**, where `CODE.md` says "a per-field reference
amplitude". The ten are components of one tensor in one chart, not
independent fields: in a chart where a component vanishes identically — the
gauge wave's `h_xy`, which is exactly zero — a per-component reference is
**exactly zero**, the floor with it, and the component's numerical dust
then scores `τ ≈ 1` over the whole domain. That is TreeWave's blast-wave
trap (`∂ₜu ≡ 0` at `t = 0`) met here in a chart rather than in initial
data, and the fix is the same one: refer the floor to an amplitude the data
actually has. The per-component maxima are still measured and returned,
because they are what the calibration table reports.

Computed **once per flagging pass** and never from inside a per-block
callback: `firing_boxes` evaluates its predicate concurrently over blocks,
and a reduction nested inside one would be a reduction per cell. It is
refreshed at every pass rather than frozen at `t = 0`, because the
hierarchy changes under the initial-data cycle and the largest `|h|` over
*grid points* outside `r_1` moves with the resolution.

One scratch field set is allocated per call, which is the pattern
TreeHydro's criterion uses for its primitives: during
`adapt_to_initial_data!` the forest is still changing, so a field set built
before the pass would have the wrong number of blocks by the second one.
"""
function field_scales(U::FieldSet{T,3}, mask, origins, spacings) where {T}
    backend = get_backend(U.work)
    scratch = FieldSet{T}(U.forest, NC; G=0, centering=U.centering,
                          backend=backend)
    map_blocks!(gh_scale_kernel!, U, scratch.work, U.work, origins, spacings,
                mask, Val(U.G))
    # One reduction per component: `block_mapreduce` takes a contiguous
    # variable range and reduces it to one value per block, so a single
    # launch over `1:NC` would give the maximum over all ten and lose the
    # per-component numbers the calibration table reports.
    return ntuple(v -> maximum(block_mapreduce(identity, max, zero(T), scratch;
                                               vars=v)), Val(NC))
end

field_scale(U::FieldSet{T,3}, mask, origins, spacings) where {T} =
    maximum(field_scales(U, mask, origins, spacings))

"""
    gh_tau_kernel!(diag, work, origins, spacings, mask, scale, ε, slot, ::Val{G})

The Löhner indicator at every owned point, written into variable `slot` of
the `diag` field set — and **zero where the mask says the point is not
evolved**.

`CODE.md`, "Three additions the black hole and the boundary need": for
`r < r_1` the damping layer and the frozen core are not a numerical
solution, so `τ` is set to zero there and a block wholly inside the
interior reports nothing and is free to coarsen as far as 2:1 balance
lets it.

The mask is asked **before** [`cell_tau`](@ref) is evaluated, not
afterwards: the core's stale data may be `NaN` and `0 · NaN = NaN` — the
same trap as the frozen core's in the right-hand side, met here in the
indicator.

The stencil reaches one point past the owned range at each face, so the
**ghosts must be filled** before this runs; [`gh_indicator!`](@ref) is
what does it, and `regrid!` cannot, because it fills ghosts only *after*
the flags are computed, for its own prolongation.
"""
@kernel function gh_tau_kernel!(diag, @Const(work), @Const(origins),
                                @Const(spacings), mask, scale, ε, slot,
                                ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    T = eltype(diag)

    st, sv, sb = work_strides(work)
    base = 1 + (b - 1) * sb +
           (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
           (I[3] + G[3] - 1) * st[3]

    keep = is_evolved(mask, point_position(origins, spacings, b, I))
    diag[inner..., slot, b] =
        keep ? cell_tau(work, base, st, sv, scale, ε) : zero(T)
end

"""
    gh_tau!(τfs::FieldSet, U::FieldSet, origins, spacings, mask; scale, ε,
            slot = DIAG_TAU)

Evaluate the indicator of the state currently in `U`'s working array into
variable `slot` of `τfs`, and return `τfs`.

**`U`'s ghosts must already hold this time's data**: the Löhner stencil
reaches one point past the block face, and a stale ghost corrupts the
verdict silently rather than raising anything.

`τfs` is a ghost-free field set over the same forest — the problem's `diag`
during an evolution, a scratch set during the initial-data cycle, where
there is no problem because the forest is still changing under the cycle.

**Why `τ` is materialised at all**, where TreeWave and TreeHydro evaluate
their per-cell indicator inside `firing_boxes`' predicate and keep nothing:
`CODE.md`'s analysis record asks for "the indicator's `τ_max`" among the
mesh statistics, and a firing *count* cannot give it. Writing `τ` once and
reducing that field — a maximum for the record, two box sweeps for the
marks — is also one evaluation of the criterion instead of three, and it
makes the number the record holds and the number the marks were taken from
the same number by construction.
"""
function gh_tau!(τfs::FieldSet{T,3}, U::FieldSet{T,3}, origins, spacings, mask;
                 scale, ε, slot::Integer=DIAG_TAU) where {T}
    τfs.nvars ≥ slot || throw(ArgumentError(
        "the indicator writes τ into variable $slot of a field set with " *
        "$(τfs.nvars) variables: `diag` carries NDIAG = $NDIAG slots and " *
        "τ is DIAG_TAU = $DIAG_TAU among them (CODE.md, \"Analysis " *
        "quantities\"), so a scratch set for the initial-data cycle needs " *
        "at least that many."))
    map_blocks!(gh_tau_kernel!, U, τfs.work, U.work, origins, spacings, mask,
                T(scale), T(ε), Int(slot), Val(U.G))
    return τfs
end

"""
    tau_max(τfs::FieldSet; slot = DIAG_TAU) -> T

The largest indicator value over the whole mesh — the `τ_max` of
`CODE.md`'s mesh statistics, and the number the calibration table is a
column of.

`block_mapreduce` partials combined in block order, so it is bit-identical
whatever the thread count.
"""
tau_max(τfs::FieldSet{T,3}; slot::Integer=DIAG_TAU) where {T} =
    maximum(block_mapreduce(identity, max, zero(T), τfs; vars=slot))

# --- the level floor and the level ceiling ----------------------------------
#
# Both are pure functions of position evaluated per block, and both are
# applied to the indicator's *request* as `clamp(request, floor, ceiling)`.
# Being functions of position is what makes them consistent under
# refinement: a block's floor is the largest floor over its extent and its
# ceiling the smallest ceiling over it, so a block and its parent cannot
# disagree about whether the region may be refined — which is what keeps a
# capped block from coarsening and refining on alternate regrids.

"""
    LevelBounds(...)

Everything the marks need about *where* a block is: the hole's center at
this time, the shell the level floor applies over, the floor's level, the
margin at the outer boundary and the level blocks inside it are capped at,
and the cap the indicator may reach elsewhere.

Host-side only — it is read in the loop over blocks that turns firing
counts into flags, at regrid cadence, and never by a kernel.
[`level_bounds`](@ref) is what builds it from a case and a forest.
"""
struct LevelBounds{T,F}
    forest::F
    center::SVector{3,T}
    floor_lo::T
    floor_hi::T                   # `floor_hi < floor_lo` means "no floor"
    floor_level::Int
    box::NTuple{3,Tuple{T,T}}
    periodic::NTuple{3,Bool}
    ceiling_margin::T
    ceiling_level::Int
    maxlevel_cap::Int
end

"""
    horizon_floor_level(forest, interior, background, G) -> Int

The level `L_h` the floor asks for around the horizon: the coarsest level
whose spacing satisfies **both** of `CODE.md`'s interior radius
requirements,

    h ≤ (r_h,min − r_1) / m        and        h ≤ (r_1 − r_0) / (2(G+1))

so that [`check_interior_radii`](@ref) holds by construction on any mesh
the floor covers. `CODE.md` says the floor level is "chosen from `r_h,min`
and `m` so that the requirements hold"; this is that choice, derived rather
than stated **(proposed in step 6)** — a floor a caller had to compute by
hand is a number that goes stale the moment the box, the margin or the
radii move.

Returns `0` when the root spacing already satisfies both, which is the
statement that no floor is needed at all.
"""
function horizon_floor_level(forest::Forest{3}, int::Interior{T}, background,
                             G::Integer) where {T}
    r_h = T(horizon_min_radius(background))
    r_h > int.r_1 || throw(ArgumentError(
        "the layer's outer radius r_1 = $(int.r_1) is not inside the " *
        "horizon's smallest coordinate radius r_h,min = $r_h, so no " *
        "refinement level can satisfy CODE.md's r_1 ≤ r_h,min − m·h: the " *
        "layer has to be moved inward before a mesh can be built around it."))
    h_horizon = (r_h - int.r_1) / int.margin
    h_layer = (int.r_1 - int.r_0) / (2 * (G + 1))
    h_needed = min(h_horizon, h_layer)
    ℓ = 0
    h = spacing(T, forest, 0)
    while h > h_needed && ℓ < MAX_LEVEL
        ℓ += 1
        h /= 2
    end
    return ℓ
end

"""
    level_bounds(case::GHCase, forest, t, G) -> LevelBounds

The floor and the ceiling of the refinement, as functions of position, at
time `t`.

The floor's shell is `r_1 ≤ r ≤ r_h,max + floor_margin` around the hole's
analytic center `c(t)`, and its level is [`horizon_floor_level`](@ref)'s.
A case with no interior has no horizon and therefore no floor, which the
empty shell `(0, −1)` says.

The ceiling's margin is `ceiling_cells` **coarse** cells — the spacing at
level 0, not the current finest one — measured inward from every
non-periodic face of the box. A block whose extent comes within it is
capped at `ceiling_level`, which is 0: `CODE.md`'s "blocks within a few
coarse cells of the domain boundary are capped at the coarsest level".
"""
function level_bounds(case::GHCase{T}, forest::Forest{3}, t,
                      G::Integer) where {T}
    ref = case.refinement
    ref === nothing && throw(ArgumentError(
        "this case carries no refinement parameters, so it has no thresholds " *
        "and no level bounds to flag with: build the case with " *
        "`refinement = Refinement(T; refine_tol, coarsen_tol, maxlevel_cap, " *
        "…)`. CODE.md's driver flags with the masked Löhner verdict and with " *
        "nothing else."))
    int = case.interior
    c = int === nothing ? SVector{3,T}(zero(T), zero(T), zero(T)) :
        center_at(int.center, T(t))
    lo, hi, L = if int === nothing
        (zero(T), -one(T), 0)
    else
        (int.r_1, T(horizon_max_radius(case.background)) + ref.floor_margin,
         horizon_floor_level(forest, int, case.background, G))
    end
    L ≤ ref.maxlevel_cap || throw(ArgumentError(
        "the interior's radius requirements need level $L around this hole — " *
        "h ≤ min((r_h,min − r_1)/m, (r_1 − r_0)/(2(G+1))) at the root spacing " *
        "$(spacing(T, forest, 0)) — but maxlevel_cap is $(ref.maxlevel_cap). " *
        "The refinement would then be asked for a level it may not reach, and " *
        "check_interior_radii would fire on the mesh it settled at. Raise the " *
        "cap to at least $L, start from a finer root brick, or move the " *
        "layer's radii outward."))
    margin = ref.ceiling_cells * spacing(T, forest, 0)
    return LevelBounds{T,typeof(forest)}(forest, c, lo, hi, L, case.box,
                                         case.periodic, margin,
                                         ref.ceiling_level, ref.maxlevel_cap)
end

# Whether a block's extent meets the spherical shell `lo ≤ r ≤ hi` around
# `c`: its nearest point is no farther than `hi` and its farthest no nearer
# than `lo`. `_box_radii` is `interior.jl`'s, so the floor and the layer's
# own assertions measure a block the same way.
function _box_meets_shell(ext, c, lo, hi)
    hi ≥ lo || return false
    near, far = _box_radii(ext, c)
    return near ≤ hi && far ≥ lo
end

# Whether a block's extent comes within `margin` of a non-periodic face of
# the box. A periodic dimension has no outer boundary and is skipped.
function _box_near_boundary(ext, box, periodic, margin)
    for d in 1:3
        periodic[d] && continue
        (ext[d][1] - box[d][1] ≤ margin || box[d][2] - ext[d][2] ≤ margin) &&
            return true
    end
    return false
end

"""
    block_level_bounds(lb::LevelBounds, k::MortonKey) -> (floor, ceiling)

The two bounds at one block: the floor it must reach because its extent
meets the horizon shell, and the ceiling it may not pass because its extent
reaches the outer boundary.

It **throws** when the floor exceeds the ceiling, naming the configuration
rather than letting `clamp` pick one: a block that is both in the horizon
shell and within the boundary margin is a box too small for a hole this
size, and the remedies — a larger box, a narrower ceiling margin, a smaller
`floor_margin` — are choices the caller has to make.
(The other way the two can disagree, a `maxlevel_cap` below the level the
interior needs, is caught once by [`level_bounds`](@ref) rather than per
block.)
"""
function block_level_bounds(lb::LevelBounds{T}, k::MortonKey{3}) where {T}
    ext = block_extent(T, lb.forest, k)
    flo = lb.floor_level > 0 &&
          _box_meets_shell(ext, lb.center, lb.floor_lo, lb.floor_hi) ?
          lb.floor_level : 0
    capped = _box_near_boundary(ext, lb.box, lb.periodic, lb.ceiling_margin)
    cap = capped ? lb.ceiling_level : lb.maxlevel_cap
    flo ≤ cap || throw(ArgumentError(
        "the refinement's level floor and its ceiling disagree about the " *
        "block $ext: the horizon shell [$(lb.floor_lo), $(lb.floor_hi)] " *
        "around $(lb.center) asks for level $flo there, and the block reaches " *
        "within the ceiling's $(lb.ceiling_margin)-wide margin of the outer " *
        "boundary, which caps it at $cap. The box is too small for a hole " *
        "this size — the region that has to be resolved around the horizon " *
        "touches the region that has to stay coarse at the boundary: widen " *
        "the box, shrink the floor's margin beyond r_h,max, or lower the " *
        "ceiling's margin. Do not resolve it by dropping either bound: the " *
        "floor is what makes the interior's radius requirements hold and the " *
        "ceiling is what keeps the Dirichlet mismatch from being refined."))
    return flo, cap
end

"""
    refine_flags(τfs::FieldSet, lb::LevelBounds, ref::Refinement;
                 slot = DIAG_TAU) -> (flags, τ_max, centroid, nfiring)

The flag vector [`regrid!`](@ref) takes, one entry per leaf, from the
indicator already written into variable `slot` of `τfs` — together with the
two mesh statistics `CODE.md`'s record asks for and which fall out of the
same two sweeps.

**Two thresholds, two meanings, four marks** (TreeWave's, verbatim):

    τ > refine_tol, level < cap  ->  (Refine, box)
    box !== nothing              ->  (Keep, box)    the travelling margin
    nothing fired, level > 0     ->  Coarsen        (bare)
    otherwise                    ->  Keep           (bare)

- `refine_tol` is **"under-resolved here"** — the mesh is not representing
  what it holds, so go finer.
- `coarsen_tol` is **"there is something here at all"** — the feature is
  present even where it is adequately resolved. The gap between the two is
  the hysteresis dead band, which is why `coarsen_tol ≥ refine_tol` is
  refused rather than accepted as a degenerate case.

**The box is the `coarsen_tol` sweep's**, and that is load-bearing. A block
refined to the cap has by construction stopped being under-resolved — its
`τ` fell below `refine_tol`, which is precisely why refinement stopped
there — so keying the box on `refine_tol` would make a feature-holding
block at the cap report nothing, and the `(Keep, box)` margin would be
unreachable in the one case it exists for. Reporting a box is what makes a
block a dilation source, and a source asks for `level + 1` when it says
`Refine` and its own `level` when it says `Keep`: a block holding the
feature at the finest level it may reach still asks for an **equal-level
margin that travels with it**. A quiet `Keep` stays bare for the mirror
reason — a box on a block whose criterion did not fire would recruit its
neighbours, and since most blocks are quiet most of the time, coarsening
would die everywhere at once.

**The floor and the ceiling enter as `clamp`.** The four marks above give a
*requested level* — `level + 1`, `level`, or `level − 1` — and the mark
actually issued is that request clamped between
[`block_level_bounds`](@ref)'s two numbers. A floor that forces a refine on
a quiet block issues a bare `Refine`, whose box is the whole interior: it
is a prescribed region and not a feature, so there is nothing narrower to
report. A floor that forces a `Keep` on a block that asked to coarsen
issues a **bare** `Keep`, for the same reason a quiet `Keep` is bare.

**The travelling margin is dilated here, not by `regrid!`
(proposed in step 6).** TreeAMR's `buffered_flags` widens every reported
box by `buffer` cells and promotes the leaves it reaches — and it promotes
them whatever this file said about them, so a block the ceiling capped at
the coarsest level is refined anyway as soon as the refined region comes
within a margin of it. On a root brick small enough that the boundary
blocks *are* the refined region's neighbours, which is every mesh a test of
this package can afford, that defeats the ceiling completely (measured in
step 6: 12 of 64 boundary root blocks refined through the margin). So the
dilation is done here, through the same `buffered_flags`, and the two
bounds are applied **again** to its result; the caller then passes
`buffer = 0` to `regrid!`, which is what makes the marks final. The floor
survives either order — the buffer never asks for less — and the boxes are
consumed by the dilation, which is all they were for.

`centroid` is the volume-weighted centroid of the cells above
`coarsen_tol` — `CODE.md`'s "refinement centroid", the measurement its G5
acceptance compares with the hole's analytic center — taken over the
bounding boxes the sweep already reports, at cell resolution, weighted by
each block's firing count and cell volume. It is `nothing` when nothing
fired anywhere.
"""
function refine_flags(τfs::FieldSet{T,3}, lb::LevelBounds{T},
                      ref::Refinement{T}; slot::Integer=DIAG_TAU,
                      buffer::Integer=0) where {T}
    forest = τfs.forest
    rtol, ctol = ref.refine_tol, ref.coarsen_tol
    s = Int(slot)

    # Two sweeps, because there are two thresholds: a firing count answers
    # one yes-or-no question per block and the criterion asks two — is any
    # cell above `refine_tol`, and *where* are the cells above
    # `coarsen_tol`. Both predicates close over two scalars only.
    refires = firing_boxes(τfs) do work, idx, b, x
        work[idx..., s, b] > rtol
    end
    boxfires = firing_boxes(τfs) do work, idx, b, x
        work[idx..., s, b] > ctol
    end

    flags = map(1:nblocks(τfs)) do b
        k = blockkey(τfs, b)
        ℓ = level(k)
        flo, cap = block_level_bounds(lb, k)
        nrefine, _ = refires[b]
        nbox, box = boxfires[b]
        request = if nrefine > 0 && ℓ < cap
            ℓ + 1
        elseif nbox > 0
            ℓ
        else
            max(ℓ - 1, 0)
        end
        target = clamp(request, flo, cap)
        if target > ℓ
            nbox > 0 ? (Refine, box) : Refine
        elseif target < ℓ
            Coarsen
        else
            nbox > 0 ? (Keep, box) : Keep
        end
    end

    marks = buffer > 0 ? clamp_marks(lb, buffered_flags(forest, flags, buffer)) :
            flags
    return (flags=marks, τ_max=tau_max(τfs; slot=slot),
            centroid=refinement_centroid(forest, boxfires),
            nfiring=sum(first, boxfires))
end

"""
    clamp_marks(lb::LevelBounds, marks) -> Vector{RegridFlag}

The level floor and the ceiling applied to already-dilated marks — the
second half of [`refine_flags`](@ref)'s travelling margin.

`buffered_flags` answers in bare flags, so each is read back as the level it
asks for (`level + 1` for `Refine`, `level` for `Keep`, `level − 1` for
`Coarsen`), clamped between the block's two bounds, and written out as a
flag again. A capped block the margin promoted becomes `Keep` rather than
`Coarsen`: the margin wanted it held, and the ceiling only says it may not
go finer.
"""
function clamp_marks(lb::LevelBounds{T}, marks::AbstractVector) where {T}
    forest = lb.forest
    return map(1:length(marks)) do b
        k = forest.leaves[b]
        ℓ = level(k)
        flo, cap = block_level_bounds(lb, k)
        f = marks[b]
        request = f === Refine ? ℓ + 1 : f === Coarsen ? max(ℓ - 1, 0) : ℓ
        target = clamp(request, flo, cap)
        target > ℓ ? Refine : target < ℓ ? Coarsen : Keep
    end
end

"""
    refinement_centroid(forest, fires) -> SVector{3} or nothing

The volume-weighted centroid of the region the indicator flagged, from the
`(count, box)` pairs of a [`firing_boxes`](@ref) sweep: each block
contributes its bounding box's center, weighted by the number of cells that
fired in it and by its cell volume.

`CODE.md`'s mesh statistics ask for "the refinement centroid against the
analytic center", and its G5 acceptance for that centroid to stay "within a
few finest spacings of the analytic center at every chunk". This is that
number, and its definition is **(proposed in step 6)** — the box centers
rather than the cells themselves, because the boxes are what the flagging
sweep already computed and a centroid over cells would be a third sweep for
a diagnostic.

**A point is weighted at the point, not half a cell above it, because the
data is vertex-centered**: the volume a vertex represents is the dual cell
*around* it, and shifting each contribution by `h/2` — the cell-centered
convention — biases the centroid of a symmetric region by `√3 h/2`
(measured in step 6: it doubled the offset below).

**The centroid has a bias of order the *coarsest* firing spacing, and the
reason is worth knowing before G5 reads it (measured in step 6).** On the
static hole, where the mesh and the solution are symmetric about the
center, the centroid comes out `1.7` to `2.4` finest spacings away from it
rather than at it. The cause is TreeAMR's half-open ownership: a block owns its
lower plane and not its upper one, so at a coarse-fine interface the coarse
block on the `+` side owns the interface plane while the coarse block on
the `−` side starts one coarse cell further out. Those interface points are
the coarse points nearest the hole, they carry the largest `τ` of their
level, and they fire on one side only. It is a property of the mesh's
indexing rather than of the indicator, it scales with the coarse spacing,
and it is what "within a few finest spacings" in G5 has to be read against.

Host-side, over the per-block results in block order, so it does not move
with the thread count.
"""
function refinement_centroid(forest::Forest{3,T}, fires) where {T}
    num = SVector{3,T}(zero(T), zero(T), zero(T))
    den = zero(T)
    for b in 1:length(fires)
        n, box = fires[b]
        n == 0 && continue
        k = forest.leaves[b]
        h = spacing(T, forest, k)
        o = block_origin(T, forest, k)
        # A vertex-centered owned index `i` sits at `origin + (i − 1)h`,
        # which is `point_position`'s expression and TreeAMR's own.
        x = SVector{3,T}(ntuple(Val(3)) do d
            o[d] + (T(first(box[d]) + last(box[d])) / 2 - 1) * h
        end)
        w = T(n) * h^3
        num += w * x
        den += w
    end
    return iszero(den) ? nothing : num / den
end

"""
    refinement_buffer(forest, maxlevel_cap, travel) -> Int

The buffer width in cells that covers a feature moving `travel` in physical
units between one regrid and the next, measured at the spacing of level
`maxlevel_cap`:

    buffer = ceil(travel / spacing(forest, maxlevel_cap)) + 1

The width is the application's to choose because it is physics — the hole's
speed times the regrid cadence, `|v| · chunk`, which the driver supplies —
and the mesh cannot know it. What TreeAMR measured is that the margin must
*exceed* the motion it covers: a margin narrower than the travel per
interval came out slightly worse than no margin at all, since the feature
leaves the refined region either way and the narrow buffer only adds cells.
Hence the `+ 1` rather than a bare `ceil`, and hence a static hole still
gets a one-cell margin.

The spacing is taken at `maxlevel_cap` rather than from `minimum_spacing`,
which reports the *current* finest spacing — coarse while the hierarchy is
still being built, and so would derive a uselessly narrow margin on the one
pass that builds it.

Recruitment reaches exactly one ring of neighbours, so TreeAMR caps the
buffer at `N`. That cap is a statement about cadence — the hole may not
cross a whole finest-level block between regrids — and this throws naming
it rather than letting the caller meet an opaque rejection inside
`regrid!`.
"""
function refinement_buffer(forest::Forest, maxlevel_cap::Integer, travel::Real)
    h = spacing(forest, maxlevel_cap)
    cells = ceilint(travel / h) + 1
    cells ≤ forest.N || throw(ArgumentError(
        "a hole travelling $travel between regrids needs a $cells-cell margin " *
        "at level $maxlevel_cap (h = $h), which exceeds the block width N = " *
        "$(forest.N). Recruitment reaches one ring of neighbours, so the " *
        "travel must stay under one finest-level block width, " *
        "$(forest.N * h): shorten the chunk, or lower the cap."))
    return cells
end

"""
    indicator_flags(U::FieldSet, case::GHCase, t; scratch = nothing, G = U.G,
                    buffer = 0) -> (flags, τ_max, centroid, nfiring, scale)

`CODE.md`'s masked Löhner verdict for every leaf, with the level floor and
the ceiling: the whole criterion in one call, as the driver and the
initial-data cycle both need it.

The steps are the ones this file is made of — the masked reference
amplitude ([`field_scales`](@ref)), the indicator into a `diag` slot
([`gh_tau!`](@ref)), and the four marks with the box keyed on
`coarsen_tol` ([`refine_flags`](@ref)) — and the reason they are one
function is that they must be done in this order with the same mask and the
same `t`.

**`U`'s ghosts must already be filled with this `t`'s hook.** The driver
fills them (the analysis monitors do it anyway) and
`adapt_to_initial_data!` fills them before it calls its criterion;
`regrid!` cannot, because it fills ghosts only after the flags exist.

`scratch` is the field set `τ` is written into: the problem's `diag` during
an evolution, so that the record and the marks read one number, and a fresh
set per pass during the cycle, where the forest is still changing and a set
built before the pass would have the wrong number of blocks.

`buffer` is the travelling margin in cells, and it is applied **here**
rather than by `regrid!` — see [`refine_flags`](@ref) for why. A caller that
passes it here passes `buffer = 0` to `regrid!` and to
`adapt_to_initial_data!`.
"""
function indicator_flags(U::FieldSet{T,3}, case::GHCase{T}, t;
                         scratch=nothing, G::Integer=first(U.G),
                         buffer::Integer=0) where {T}
    ref = case.refinement
    ref === nothing && throw(ArgumentError(
        "this case carries no refinement parameters: build it with " *
        "`refinement = Refinement(T; refine_tol, coarsen_tol, maxlevel_cap, " *
        "floor_margin, ceiling_cells)`. There is one refinement mechanism in " *
        "this package and it is CODE.md's masked Löhner indicator."))
    backend = get_backend(U.work)
    origins = to_backend(backend, block_origins(U.forest, T))
    spacings = to_backend(backend, block_spacings(U.forest, T))
    mask = interior_mask(case.interior, T(t))
    scale = field_scale(U, mask, origins, spacings)
    τfs = scratch === nothing ?
          FieldSet{T}(U.forest, NDIAG; G=0, centering=U.centering,
                      backend=backend) : scratch
    gh_tau!(τfs, U, origins, spacings, mask; scale=scale, ε=ref.ε)
    out = refine_flags(τfs, level_bounds(case, U.forest, t, G), ref;
                       buffer=buffer)
    return (flags=out.flags, τ_max=out.τ_max, centroid=out.centroid,
            nfiring=out.nfiring, scale=scale)
end

"""
    gh_indicator!(p::GHProblem, u, t; buffer = 0) -> NamedTuple

The indicator of the state `u` at time `t`, on the problem's own mesh: the
right-hand side's preamble — scatter, then fill the ghosts with **this**
`t`'s Dirichlet hook — and then [`indicator_flags`](@ref) into `p.diag`'s
`DIAG_TAU` slot.

This is what the driver calls at a chunk boundary, and the ghost fill is
the part that cannot be skipped: the Löhner stencil reaches one point past
the block face, `regrid!` fills ghosts only after the flags are computed,
and a stale ghost corrupts the verdict silently.
"""
function gh_indicator!(p::GHProblem{T}, u, t; buffer::Integer=0) where {T}
    _prepare_monitor!(p, u, t)
    return indicator_flags(p.U, p.case, T(t); scratch=p.diag, G=first(p.U.G),
                           buffer=buffer)
end
