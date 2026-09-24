# The driver: **one** chunked evolve-and-regrid loop, and the analysis
# record a run is judged by.
#
# `CODE.md`, "Refinement and regridding" (the loop) and "Analysis
# quantities" (what it writes down). Regridding changes both the length and
# the meaning of the state vector, so a run cannot be one `solve` call:
# every chunk is a fresh `solve` on a fresh [`GHProblem`](@ref), and between
# chunks the mesh is rebuilt. That is TreeAMR's prescription and TreeWave's,
# TreeHydro's and Burgers' practice, and this file is TreeHydro's one loop
# with cases as data.
#
# Three things in the loop are easy to get wrong and are each written out
# where they happen:
#
#   * **`ρ_max` is per chunk.** The interior the kernel sees is rebuilt at
#     every chunk ([`with_interior`](@ref)) — not taken from the case, which
#     carries a placeholder. By default it relaxes at the physical rate
#     `4/M` of [`default_relaxation_rate`](@ref), the same in every chunk
#     (decided 2026-09-23, step 8c′); `ρ_max_factor` selects the grid rate
#     `factor/dt` instead, which follows *this* chunk's `dt` and was the
#     default until then. RK4 bounds either at about `2.8/dt` on the
#     negative real axis, and a run that blows up in the layer after
#     raising `ρ_max` has found the integrator, not the physics.
#   * **The boundary hook goes to three places** — `fill_ghosts!` (inside
#     the right-hand side), `regrid!`, and `adapt_to_initial_data!` — each
#     with that call's `t`. All three are here from step 6; forgetting the
#     second is the bug that arrives one chunk late, and passing a stale
#     `t` is the bug that arrives as a boundary reflection.
#   * **Chunks are counted, not accumulated.** `nchunks = ceilint(t_end /
#     chunk)` and `stop = min(c · chunk, t_end)`, never a
#     `while t < t_end − tiny` guard: an absolute slack is meaningless at a
#     type whose ulp is larger than it (TreeHydro's note on precision).
#
# **The mesh (added in step 6).** `adapt = true` runs `CODE.md`'s
# initial-data cycle with the masked Löhner indicator before the first
# chunk, and `regrid = true` flags with the same indicator at every chunk
# boundary but the last. Both refuse a case with no [`Refinement`](@ref) by
# name, because this package has exactly one refinement mechanism: a driver
# that regridded on some other criterion would be a second one to delete
# later. The regrid is skipped after the final chunk, so the forest that
# comes back and the state that comes back describe the same mesh.

"""
    check_cfl(dt, h_min, cfl, λ_end; chunk = nothing, λ = nothing)

The end-of-chunk CFL recheck: throw an `ArgumentError` unless the step
actually taken satisfies `dt · λ_end / h_min ≤ cfl` against the fastest
signal now present. Returns the CFL number the step achieved.

**It throws on purpose** (`CODE.md`, "The time step": *throw, do not warn,
if the step used violated the bound*). `λ_max` is measured once per chunk
from the state at its start, which is a bound within the chunk only in
practice; this turns "in practice" into a fact. It is a *detector* rather
than a guard — it fires after the chunk that violated the condition has
been integrated — and the remedy is a shorter chunk or a smaller `cfl`,
never deleting the check.

The bound is compared with a few ulp of slack, because the step actually
taken is `(stop − t)/steps` with an integer `steps` and is therefore *at
most* the step asked for: the equality case is real and must not fail on
rounding.

A pure function of its four numbers, so the tests can exercise it on
synthetic ones; the keywords are context and enter only the message.
"""
function check_cfl(dt, h_min, cfl, λ_end; chunk=nothing, λ=nothing)
    ν = dt * λ_end / h_min
    ν ≤ cfl * (1 + 8 * eps(float(one(ν)))) && return ν
    where = chunk === nothing ? "" : " in chunk $chunk"
    sized = λ === nothing ? "" : " The step was sized from λ = $λ."
    throw(ArgumentError(
        "the CFL condition was violated$where: the step actually taken, " *
        "dt = $dt at h_min = $h_min, is the step for a fastest signal of " *
        "$(cfl * h_min / dt), and the fastest signal at the end of the " *
        "chunk is λ_end = $λ_end — a CFL number of $ν against the requested " *
        "$cfl." * sized *
        " λ_max is measured once per chunk and this recheck is a detector " *
        "rather than a guard, so the chunk has already been integrated at " *
        "the wrong step: shorten the chunk so that the speed has less room " *
        "to grow between measurements, or lower cfl. Do not delete the " *
        "check."))
end

"""
    forest_levels(forest) -> Vector{Int}

How many leaves sit at each level, `0` first — `CODE.md`'s "leaf count per
level" row of the mesh statistics.
"""
function forest_levels(forest::Forest)
    counts = zeros(Int, maxlevel(forest) + 1)
    for k in forest.leaves
        counts[level(k) + 1] += 1
    end
    return counts
end

"""
    horizon_shell(case::GHCase, T) -> (lo, hi)

The band of radii the gauge drift is measured over: from the horizon's
smallest coordinate radius to its largest plus one `M`-ish margin — the
shell `CODE.md` calls "at the horizon".

The margin is the layer's own width, `r_1 − r_0`, rather than a number
picked out of the air: it is the only length in the case that is set by the
hole and known to be resolved. A case with no interior has no shell, and the
empty band `(0, −1)` is what the error kernel then writes zeros for.

**(Proposed in step 5.)** `CODE.md` asks for "the gauge drift rate of
`h_tt` at the horizon" and does not say over what set; this is the set.
"""
horizon_shell(case::GHCase) = horizon_shell(case, case.interior)

horizon_shell(case::GHCase{T}, ::Nothing) where {T} = (zero(T), -one(T))

function horizon_shell(case::GHCase{T}, int::Interior) where {T}
    lo = T(horizon_min_radius(case.background))
    hi = T(horizon_max_radius(case.background)) + (int.r_1 - int.r_0)
    return (lo, hi)
end

# The tracked geometry's shell (added in step 8d): the same band about its
# own center, from the tracked horizon's smallest radius to its largest plus
# the ramp — the layer's width, as for the sphere.
horizon_shell(case::GHCase, int::FittedInterior) =
    (int.r_in, int.r_out + int.thickness)

horizon_shell(case::GHCase, ::FittedSpec) = throw(ArgumentError(
    "a tracked case's drift shell is stated about its geometry, which is " *
    "built from the track once per chunk: pass the geometry, " *
    "horizon_shell(case, fitted_interior(…))."))

"""
    evolve!([T], case::GHCase; forest, q, ops, t_end, chunk = case's,
            cfl = 1//4, regrid = false, adapt = false, buffer = nothing,
            maxpasses = 8, adm_every = 0, backend = CPU(),
            observer = nothing, ρ_max_factor = nothing,
            ρ_max_fixed = nothing, find = find_gh_horizon,
            fit_initial_cont = 1, fit_initial_depth = 0, handover = 0,
            target_source = :fit)

**The** time-stepping loop: fill the initial data on `forest`, then evolve
in chunks of `chunk`, writing `CODE.md`'s analysis record at every chunk
boundary. Returns a named tuple of everything the claims are made of — the
record per chunk, the mesh statistics, the final state, state vector and
forest.

There is **one** of these for every case, which is the decision recorded in
`CODE.md`'s "The driver"; what differs between cases is data
([`GHCase`](@ref)), and nothing here knows what a black hole is beyond
asking the case for its interior.

## Why a chunk is a restart

Regridding changes both the *length* and the *meaning* of the state
vector, so each chunk is a fresh fixed-step `solve`. It is a restart even
when nothing regrids, because `ρ_max` changes from chunk to chunk and that
is the same rebuild.

## The step, the relaxation rate, and the recheck

One global step for the whole hierarchy, `dt = cfl · h_min / λ` with `λ`
from [`max_speed`](@ref) at the chunk's start. At the chunk's end the speed
is measured again and [`check_cfl`](@ref) **throws** if the step actually
taken violated the bound.

**The layer relaxes at `4/M` by default (decided 2026-09-23, step 8c′).**
With neither rate keyword given, `ρ_max` is
[`default_relaxation_rate`](@ref) of the case — `4/M`, `M` the hole's mass
parameter — in every chunk. Two keywords choose otherwise, and are refused
together, because they are two answers to one question:

  * `ρ_max_factor` selects the **grid rate** `ρ_max_factor/dt`, per chunk —
    `CODE.md`'s former `ρ_max · dt = 1`, which relaxes by a factor `e` per
    step and was the default until step 8c′. It is about `107/M` on the
    suite's fixture at `cfl = 1/5`, a paste two cells deep, and on the exact target it
    ends a `50 M` run with six times the error `4/M` does (step 8c); it
    stays as the option a study of the rate, or a replay of step 5's
    numbers, asks for.
  * `ρ_max_fixed` selects a fixed rate in the case's units (added in step
    8c).

A fixed rate — the default's included — above `1/dt` is refused at the
chunk that would take it: it would relax harder than the grid rate, and
RK4's real-axis limit is `2.79/dt` **(proposed in step 8c)**. The default
is far below it on every mesh that resolves a hole (`0.04/dt` to `0.05/dt`
on the suite's fixture). The record's `ρ_max` row is the rate the chunk ran at,
whichever rule chose it. A case with no interior has no rate, and none of
this touches it.

## What is recorded, and when

At every chunk boundary, and once at `t = 0`: the masked gauge-constraint
norms, the masked error against the analytic solution, the interior
residual, the gauge drift in the horizon shell, the fastest speed, the step
and the mesh statistics — including, for a case that carries a
[`Refinement`](@ref), the indicator's `τ_max` and the refinement centroid
against the analytic center. The ADM monitor runs every `adm_every`-th
chunk (`0` means never, which is what a short run wants — `CODE.md` prices
it at about one right-hand-side evaluation and compiles in 18 s). **Every
norm is masked** by the interior at that chunk's `t`: the layer and the core
are not a numerical solution and are not reported as one.

Every row has the **validity monitor** (added in step 8b): the minimum of
`det γ`, the minimum signed lapse and the largest `|h_ab|` and `|Π_ab|` over
the layer and over the `G` points outside `r_1` ([`validity_rows`](@ref)) —
and the **range projection's** counts for the chunk, `bounds_hits`,
`bounds_nonfinite` and `bounds_r_max` (the outermost radius it fired at, `−1`
where it did not), which are `nothing` for a case whose `bounds` is
`nothing`. `finite` is the evolved region's: a `NaN` in the frozen core is
the projection's business, not the end of the run ([`evolved_nonfinite`](@ref)).

A case that carries a [`Horizon`](@ref) adds the horizon rows every `k`-th
chunk (added in step 7): the found `origin` and its distance from the
analytic center, the coordinate radii `r_min`, `r_mean`, `r_max`, the
proper `area`, `M_irr`, the Korzyński `J` with its `spin_axis`, `M_ch`, and
the shape `hlm` — which is also the next find's seed. They are `nothing` on
a chunk the cadence skips, as the ADM rows are, and a find that fails
leaves `horizon_success = false` and its message in `horizon_note` rather
than ending the run.

## Keywords

`forest` has no default: the mesh is the study's, built by
[`gh_forest`](@ref) or [`hole_forest`](@ref) — or, with `adapt = true`, by
the indicator's own cycle starting from whichever of those was passed.
`q`, `ops`, `t_end` and `chunk` have none either, for `PLAN.md`'s reason —
the operator order in particular, since this system takes second
derivatives and a prolongation below `q + 2` costs the scheme an order.

`adapt` runs `CODE.md`'s initial-data cycle before the first chunk: fill,
flag with [`indicator_flags`](@ref), regrid *without transferring*, and
re-evaluate the initial data on the new mesh, until the hierarchy stops
changing — re-evaluating rather than interpolating, since interpolating
would bake the coarse mesh's resolution into the refined blocks. It throws
if the cycle has not converged within `maxpasses`.

`regrid` flags with the same indicator at the end of every chunk but the
last, and rebuilds the schedule, the problem (which re-samples the gauge
source and **re-asserts the interior's radius requirements**) and the state
vector whenever the mesh moved. It is skipped after the final chunk so that
the forest and the state that come back describe the same mesh. Both
keywords refuse a case with no `Refinement` by name.

`buffer` is the travelling margin in cells; left at `nothing` it is derived
by [`refinement_buffer`](@ref) from the hole's own speed and the chunk,
`|v| · chunk`, which is one cell for a static hole and grows with `|v|` in
G5.

`observer(p, t, u)` is called with the state scattered into `p.U` and the
analysis record for that chunk already written — once at `t = 0` and once
per chunk — which is what keeps a viewer free of any time stepping of its
own.

## The range projection (added in step 8b)

A case with [`StateBounds`](@ref) gets `CODE.md`'s third state writer:
[`gh_stage_limiter!`](@ref) is passed to `solve` as its `stage_limiter`,
beside the `:pasted` variant's `step_limiter`, and runs on every stage
vector; the same projection is applied once to the initial data and once to
every freshly regridded state, neither of which went through a stage —
TreeHydro's atmosphere reset, applied where TreeHydro applies it. One
[`BoundsAccounting`](@ref) is made per run, handed to every problem the run
builds, and returned as `bounds`; for a case without bounds the limiter is
a no-op and `bounds` is `nothing`. Where the projection never fires, the
run is **bit for bit** the run without it.

## The fitted target's switches (added in steps 8e and 8f)

`fit_initial_cont` and `fit_initial_depth` are step 8e's study knobs for a
`:fitted` case's initial data: the radial order of the analytic fit, and how
far below the offset surface the analytic solution gives way to it (`0`, the
decision, or down to the core surface `n_L h`, which needs the analytic
solution regular on the whole layer).

`handover > 0` runs a `:fitted` case on the analytic `:damped` layer of the
*same* tracked geometry until the first chunk that starts at or after
`handover`, fitting the evolved state at every row all the while, and on the
`:fitted` target from then on: a newly found horizon's first fit made from
evolved data (step 8f's hand-over row). Its initial data is the analytic
layer's, through the core rule.

`target_source = :snapshot` fills a `:fitted` case's target cache with the
state itself at every refill instead of the fit — the snapshot control of
step 8f's matrix ([`snapshot_target_kernel!`](@ref)).

`target_rate = true` (added in step 8, the default) feeds the target's rate
forward: the cache's slope includes the fit's translation with the track,
and the kernel adds `(1 − w) ∂_t u_fit` to the layer and the core, so that a
point relaxing toward a *moving* target follows it instead of lagging it by
`|∂_t u_fit|/ρ` — which on harmonic Kerr at `a = 7/10` moving at `0.3` is
thousands, released on the trailing side into the evolved region
(`CODE.md`, "The fitted target"). `false` is step 8e's layer.

`adapt = true` on a `:fitted` case chooses the mesh on the case's own
initial data — the analytic solution outside the offset surface, the fit of
it inside — rebuilt with the geometry on every pass
([`adapt_fitted_initial_data!`](@ref); amended in step 8, which replaced
step 8f's cycle on the analytic `:damped` data and so lifted its refusal of
a chart whose analytic core meets the singular set, G5's). A hand-over case
still chooses it on the analytic layer, which is its initial data.

**A moving hole (added in step 8).** `regrid = true` flags the evolved state
at every chunk boundary with the geometry the next chunk runs on, and the
level floor is widened by `|v| · chunk` inward and outward (`travel`, `v`
the case's analytic velocity), so that the layer stays in blocks the floor
held while it moves; the travelling margin `buffer` is derived from the same
distance.
"""
evolve!(case::GHCase{T}; kwargs...) where {T} = evolve!(T, case; kwargs...)

evolve!(::Type{S}, case::GHCase{T}; kwargs...) where {S,T} = throw(ArgumentError(
    "evolve! was asked for $S but the case is stated in $T: a case carries " *
    "its own working type — its box, its radii and its damping rates are " *
    "all in it — so the type is chosen when the case is built and not when " *
    "it is run. Build the case at $S instead."))

function evolve!(::Type{T}, case::GHCase{T}; forest, q::Integer, ops, t_end,
                 chunk=case.chunk, cfl=T(1 // 4), regrid::Bool=false,
                 adapt::Bool=false, buffer=nothing, maxpasses::Integer=8,
                 adm_every::Integer=0, backend=CPU(), observer=nothing,
                 ρ_max_factor=nothing, ρ_max_fixed=nothing,
                 find=find_gh_horizon, fit_initial_cont::Integer=1,
                 fit_initial_depth=0, handover=0,
                 target_source::Symbol=:fit, target_rate::Bool=true) where {T}
    # The tracked geometry (step 8d) is built from the horizon that was found,
    # so a case that asks for it must carry the finder's parameters.
    fitted = case.interior isa FittedSpec
    fitted && (case.horizon === nothing || case.horizon.every < 1) &&
        throw(ArgumentError(
            "this case's layer follows the tracked horizon (its interior is a " *
            "FittedSpec) but it carries no horizon finder to track it with" *
            (case.horizon === nothing ? "" : " (its Horizon has every = 0, " *
                                             "which is never)") *
            ": the geometry is rebuilt from each find, and a track that is " *
            "never updated is the analytic seed carried along forever — step " *
            "5's layer with extra steps. Give the case `horizon = Horizon(T; " *
            "every ≥ 1, N, …)`, or use a :damped/:frozen/:pasted sphere."))
    (regrid || adapt) && case.refinement === nothing && throw(ArgumentError(
        "evolve! was asked to $(regrid ? "regrid" : "adapt the initial data") " *
        "but this case carries no refinement parameters, so there are no " *
        "thresholds to flag with. Build the case with `refinement = " *
        "Refinement(T; refine_tol, coarsen_tol, maxlevel_cap, floor_margin, " *
        "…)`: CODE.md's loop flags with the masked Löhner indicator — its " *
        "interior mask, its level floor around the horizon and its ceiling at " *
        "the outer boundary — and with nothing else, and a driver that " *
        "regridded on some other criterion would be a second refinement " *
        "mechanism to delete later."))
    t_end, chunk, cfl = T(t_end), T(chunk), T(cfl)
    t_end > 0 || throw(ArgumentError("t_end must be positive, got $t_end."))
    chunk > 0 || throw(ArgumentError(
        "chunk must be positive, got $chunk: it is the cadence the analysis " *
        "record is written at and the regrid cadence from step 6, and the " *
        "number of chunks is counted as ceilint(t_end / chunk). The case " *
        "carries one; pass it here if the case does not."))
    (ρ_max_factor === nothing || ρ_max_fixed === nothing) ||
        throw(ArgumentError(
            "evolve! was given both ρ_max_factor = $ρ_max_factor and " *
            "ρ_max_fixed = $ρ_max_fixed, but they are two answers to one " *
            "question: the layer's relaxation rate is either the grid rate " *
            "factor/dt (CODE.md's former ρ_max · dt = 1) or a fixed physical " *
            "rate (step 8c), and with neither it is the default 4/M " *
            "(step 8c′). Pass one of them, or none."))
    ρ_max_factor === nothing || T(ρ_max_factor) > 0 || throw(ArgumentError(
        "ρ_max_factor scales the grid rate 1/dt and must be positive, got " *
        "$ρ_max_factor; it exists so that a study may measure what the grid " *
        "rate is worth against the default 4/M, not so that a run may " *
        "switch the layer off."))
    ρ_max_fixed === nothing || T(ρ_max_fixed) > 0 || throw(ArgumentError(
        "ρ_max_fixed is the layer's relaxation rate and must be positive, " *
        "got $ρ_max_fixed; a rate of zero is the :frozen variant, which is " *
        "selected by name."))
    # The rule, resolved once for the whole run: the grid rate if a factor
    # was given, a fixed rate if one was, and otherwise the default — read
    # from the hole's mass, and only where there is a hole, since a case
    # without an interior has neither a mass nor a layer to relax.
    ρ_max_default = ρ_max_factor === nothing && ρ_max_fixed === nothing &&
                    case.interior !== nothing
    ρ_max_factor = ρ_max_factor === nothing ? nothing : T(ρ_max_factor)
    ρ_max_fixed = ρ_max_fixed !== nothing ? T(ρ_max_fixed) :
                  ρ_max_default ? default_relaxation_rate(case) : nothing

    G = q ÷ 2 + 1
    U = FieldSet{T}(forest, 2NC; G=G, centering=vertexcentered(3),
                    backend=backend)

    # The travelling margin, in cells at the finest level the indicator may
    # reach: `CODE.md`'s `|v| · chunk`, the hole's own speed times the
    # regrid cadence, plus one — TreeAMR measured that a margin narrower
    # than the motion it covers is worse than none at all. A static hole
    # still gets the one cell.
    speed = sqrt(sum(abs2, case.center.v))
    # The distance the hole moves between two regrids, which the level floor
    # is widened by (added in step 8; zero for a static hole).
    travel = speed * chunk
    bufferwidth = buffer !== nothing ? Int(buffer) :
                  case.refinement === nothing ? 0 :
                  refinement_buffer(forest, case.refinement.maxlevel_cap,
                                    speed * chunk)

    # The geometry (step 8d). For step 5's sphere it is the case's own
    # interior, the same in every chunk; for a tracked case it is built from
    # the track — the analytic seed first, each find's answer after it — on
    # the mesh as it is, once per chunk and after every regrid. `n_L` is step
    # 8c's rule at this scheme's `G` and this run's rate (the default `4/M`
    # where the rate is the grid's, whose value changes per chunk; proposed
    # in step 8d).
    spec = fitted ? case.interior : nothing
    tr = fitted ? seed_track(case, zero(T)) : nothing
    n_L = !fitted ? 0 :
          spec.n_L > 0 ? spec.n_L :
          layer_cells(G, ρ_max_fixed === nothing ? default_relaxation_rate(case) :
                         ρ_max_fixed, hole_mass(case.background))
    geometry(f, t, track) = fitted_interior(spec, track, f, G; t=T(t), n_L=n_L)
    geom = fitted ? geometry(forest, zero(T), tr) : case.interior

    # The fitted target (step 8e). Its ranges come from the seed's analytic
    # data on the offset surface unless the spec states them, and the first
    # fit is of the *analytic* solution, `cont = 1`, from `Float64` samples
    # whatever `T` is (both decided in review, step 8e): it is the initial
    # data inside the offset surface and the first chunk's target. After
    # that every record row fits the state it records (`refit!`, below).
    # Both are built once the mesh is settled, after the initial-data cycle.
    fitmode = fitted && interior_variant(spec) === :fitted
    # The hand-over (added in step 8f): a `:fitted` case that runs the
    # analytic `:damped` layer on the same tracked geometry until `handover`,
    # fitting the evolved state at every row all the while, and relaxes
    # toward the fit from the first chunk that starts at or after it — a
    # newly found horizon's first fit made from evolved data.
    handover_t = T(handover)
    handover_t ≥ 0 || throw(ArgumentError(
        "handover is the time the :fitted target takes over, got $handover."))
    handover_t > 0 && !fitmode && throw(ArgumentError(
        "handover = $handover asks a :fitted case to start on the analytic " *
        ":damped layer and hand over to its fit; this case's interior is " *
        "not :fitted."))
    target_source in (:fit, :snapshot) || throw(ArgumentError(
        "target_source is :fit (the fitted target, step 8e) or :snapshot (the " *
        "state at the chunk's start, step 8f's control), got :$target_source."))
    target_source === :snapshot && !fitmode && throw(ArgumentError(
        "target_source = :snapshot is the :fitted variant's cache filled with " *
        "the state; this case's interior is not :fitted."))
    # The target's rate (added in step 8): fed forward on the `:fitted`
    # target unless asked not to; the snapshot control's slope is zero.
    rate_on = fitmode && target_rate && target_source === :fit
    d_init = T(fit_initial_depth)
    fitmode && handover_t == 0 && !(zero(T) ≤ d_init ≤ geom.thickness) &&
        throw(ArgumentError(
            "fit_initial_depth = $d_init must lie between 0 (the offset " *
            "surface, the decision) and the ramp's thickness $(geom.thickness)."))
    # The layer the kernel runs at time `t`: the geometry, or before the
    # hand-over the same geometry's analytic `:damped` layer.
    kgeom(g, t) = handover_t > 0 && t < handover_t ? with_variant(g, :damped) : g
    refilling(t) = fitmode && !(handover_t > 0 && t < handover_t)
    tbounds = nothing
    fit_initial = nothing
    fits = nothing
    # What the fits and the cache cost, and how often the cache is refilled
    # (the numbers step 8e records).
    fitcost = (build_ns=Ref(0), nbuild=Ref(0), fill_ns=Ref(0), nfill=Ref(0),
               nrefills=Ref(0), chunk_refills=Ref(0), nfailed=Ref(0))

    # The initial-data cycle, or the plain fill. The cycle fills the data
    # itself — that is what it is for: it re-evaluates it on each new mesh
    # rather than interpolating, since interpolating would bake the coarse
    # mesh's resolution into the blocks the refinement just bought.
    passes = 0
    converged = true
    schedule = GhostSchedule(U, ops)
    if adapt && fitmode && handover_t == 0
        # **A `:fitted` case's mesh is chosen on its own initial data (added
        # in step 8)** — the analytic solution outside the offset surface and
        # the fit of it inside, rebuilt with the geometry on every pass's
        # mesh — so that a chart whose analytic interior is singular, G5's,
        # has a cycle at all. See `adapt_fitted_initial_data!`.
        schedule, passes, converged, geom = adapt_fitted_initial_data!(
            U, ops, case, g -> geometry(g, zero(T), tr); G=G,
            buffer=bufferwidth, travel=travel, maxpasses=maxpasses,
            cont=fit_initial_cont, depth=d_init, backend=backend)
        converged || throw(ErrorException(
            "the initial-data cycle of this :fitted case had not converged " *
            "after $passes passes: the hierarchy was still changing when " *
            "maxpasses ran out. Raise maxpasses only if the passes were still " *
            "making progress; otherwise the criterion never stops asking, " *
            "which is a maxlevel_cap too high for the data or a refine_tol " *
            "below what the mesh can reach."))
    elseif adapt
        # The margin is dilated inside the indicator, so that the level
        # ceiling is applied *after* it — `buffer = 0` here and there is
        # what `refine_flags` means by "the caller passes zero". A tracked
        # geometry is rebuilt on each pass's mesh, since its offset and ramp
        # are stated in that mesh's spacings.
        #
        # **A hand-over case's mesh is chosen on the analytic `:damped` data
        # of the same geometry (added in step 8f** for every `:fitted` case;
        # **amended in step 8**, which gave a `:fitted` case without a
        # hand-over a cycle on its own data, the branch above**)**: its
        # initial data *is* that analytic layer's. That needs the analytic
        # solution regular on the core surface, which `check_interior_radii`
        # asserts for the `:damped` geometry — it refuses harmonic Kerr's
        # spinning charts by name.
        acycle(g) = fitmode ? with_variant(g, :damped) : g
        fitmode && check_interior_radii(forest, acycle(geom), case.background,
                                        G; t=zero(T), center=case.center)
        criterion(fs) = indicator_flags(fs, case, zero(T); G=G,
                                        buffer=bufferwidth, travel=travel,
                                        interior=fitted ?
                                                 acycle(geometry(fs.forest,
                                                                 zero(T), tr)) :
                                                 case.interior).flags
        schedule, passes, converged = adapt_to_initial_data!(
            U, ops; initial=state_callback(case, zero(T); interior=acycle(geom)),
            flags=criterion, buffer=0, maxpasses=maxpasses,
            boundary=dirichlet(case, zero(T)))
        converged || throw(ErrorException(
            "the initial-data cycle had not converged after $passes passes: " *
            "the hierarchy was still changing when maxpasses ran out. The " *
            "cycle re-evaluates the initial data on each new mesh rather " *
            "than interpolating it, so it terminates when the criterion " *
            "stops asking for anything new — and a criterion that never " *
            "stops is either a maxlevel_cap too high for the data or a " *
            "refine_tol below what the mesh can reach. Raise maxpasses only " *
            "if the passes were still making progress."))
        # The cycle filled with the geometry of the mesh it started from;
        # a tracked geometry is rebuilt on the mesh it settled at, and the
        # data re-evaluated with its core rule — re-evaluated, not
        # interpolated, for the cycle's own reason.
        if fitted
            geom = geometry(forest, zero(T), tr)
            (fitmode && handover_t == 0) ||
                fill_exact!(U, case, zero(T); interior=kgeom(geom, zero(T)))
        end
    elseif !fitmode || handover_t > 0
        fill_exact!(U, case, zero(T); interior=kgeom(geom, zero(T)))
    end
    # The fitted variant's initial data (decided in review, step 8e): the
    # cache is filled from the analytic solution's fit first, and the state
    # is the analytic solution outside the offset surface and the cache
    # inside it — no analytic evaluation below the surface, so a chart whose
    # interior is singular gets regular data (`fill_fitted_initial!`, which
    # the `:fitted` cycle flags on; factored out in step 8). Before a
    # hand-over the data is the analytic layer's, and only the fit is made.
    target0 = nothing
    if fitmode && handover_t == 0
        ff = fill_fitted_initial!(U, case, geom; cont=fit_initial_cont,
                                  depth=d_init, backend=backend, rate=rate_on)
        tbounds, fit_initial, target0 = ff.bounds, ff.fit, ff.target
        fits = (fit_initial, nothing)
    elseif fitmode
        tbounds = spec.target_bounds !== nothing ?
                  _bounds_in(T, spec.target_bounds) :
                  derive_target_bounds(T, case.background, geom; t=0,
                                       L=spec.lmax_fit)
        fit_initial = build_fit(analytic_sampler(case.background, 0.0;
                                                 δ=tofloat64(geom.h) / 8),
                                geom, spec; cont=fit_initial_cont,
                                bounds=tbounds, backend=backend)
        fits = (fit_initial, nothing)
        target0 = target_cache(U)
    end
    u = statevector(U)
    gather!(u, U)
    # The initial data must be finite *everywhere*, core and layer included:
    # the core rule makes it so, and a non-finite value here is a case whose
    # analytic solution is singular somewhere the mesh reaches — a
    # configuration error, not something for the range projection to repair
    # (kept unmasked in step 8b, where the per-chunk check became the
    # evolved region's).
    all(isfinite, u) || throw(ArgumentError(
        "the initial data is not finite: $(count(!isfinite, u)) of " *
        "$(length(u)) values are NaN or Inf. The core rule fills every point " *
        "inside r_0 from the sphere r_0, so this is a case whose analytic " *
        "solution is singular somewhere the mesh reaches — check r_0 against " *
        "the chart's singular set and against where |h| is still moderate."))

    # The problem is built once here — the gauge source is sampled in it,
    # which is the expensive setup phase — and only its interior is
    # replaced per chunk. Its constructor is also where `CODE.md`'s two
    # interior radius requirements are asserted, so the mesh the cycle just
    # chose is checked before anything is integrated on it. The range
    # projection's record is made once, here, and every problem the run
    # builds shares it (step 8b).
    acc = case.bounds === nothing ? nothing : BoundsAccounting()
    p0 = GHProblem(U, schedule, case; q=q, t=zero(T),
                   interior=kgeom(geom, zero(T)), accounting=acc,
                   target=target0, fits=fits, target_rate=rate_on)
    # The geometry the gauge source was sampled with (step 8d): the sample
    # applies the core rule, so a tracked core that moves far from it asks
    # for a fresh sample — see the chunk loop.
    geom_sampled = geom
    nresamples = 0
    # **The record is `Float64` whatever the run computes in.** That is
    # what `precision.jl`'s `tofloat64` exists for: the analysis time
    # series, the numbers a test compares against `CODE.md`, and the I/O of
    # step 9 are `Float64` at every element type, so a `Float32` run's
    # table is comparable with a `Float64` one without a conversion at
    # every call site. It is host-side and at chunk frequency, which is the
    # only place that function is allowed.
    R = tofloat64

    records = NamedTuple[]
    nsteps = 0
    nregrids = 0
    λ_initial = zero(T)
    # The previous find's shape, which the next one is seeded with
    # (`CODE.md`: "Each find is seeded with the previous result, recentred
    # on `c(t)`"). It is the *shape* that is carried and not the origin:
    # `find_gh_horizon` recentres on the analytic center at every call.
    #
    # A tracked run's *first* find is seeded with the seed track's own shape
    # — the analytic horizon, zero-padded onto the finder's grid — rather
    # than with a sphere (proposed in step 8e): on harmonic Kerr at
    # `a = 9/10` the sphere of the mean radius, `0.72`, lies inside the
    # offset surface's equator at `0.92`, so its first iterates read the
    # layer and the footprint guard refuses them, and a track whose first
    # three finds fail is lost. On a spherical horizon the two seeds are the
    # same surface.
    hlm_seed = fitted && case.horizon !== nothing ?
               complex_from_real(tr.shape, case.horizon.N - 1) : nothing
    # The lapse-collapse trigger (step 8d): set by a row whose evolved region
    # has `min α` below the spec's `α_trigger`, it forces a find at the next
    # chunk boundary whatever the cadence, and that row says so.
    trigger_pending = false

    # The track's rows for one record entry, and the next chunk's geometry:
    # after this row's find, `update_track`, then `fitted_interior` from the
    # updated track on this mesh — asserted against it like the sphere, and
    # priced in e-folds (step 8d). A lost track is *returned*, so that the row
    # of its last miss is recorded before the run ends with it.
    no_track = (track_source=nothing, track_center=nothing,
                track_velocity=nothing, track_r_min=nothing,
                track_r_max=nothing, track_offset=nothing,
                track_misses=nothing, track_prediction=nothing,
                track_trigger=nothing, margin_efolds=nothing, layer_h=nothing,
                layer_offset=nothing, layer_thickness=nothing,
                layer_r_in=nothing, layer_r_out=nothing)
    no_fit = (fit_valid=nothing, fit_residual=nothing, fit_min_detγ=nothing,
              fit_min_α=nothing, fit_min_λ=nothing, fit_hits=nothing,
              fit_refills=nothing)

    # The fit of the state at a record row (step 8e): on the geometry the
    # next chunk runs on (`track!` has just rebuilt it), from the state
    # sampler on freshly filled ghosts — the find filled them, but so may
    # every monitor since, and a fill is a fifth of a right-hand side —
    # `cont = 1`, into the target ranges. A fit whose sweep is not a metric
    # does not become the target: the previous fits are kept, the row says
    # `fit_valid = false`, and the run goes on (proposed in step 8e — the
    # find's coasting, applied to the fit).
    function refit!(p, u, t)
        scatter!(p.U, u)
        bnd = dirichlet(case, T(t))
        if bnd === nothing
            fill_ghosts!(p.U, p.schedule)
        else
            fill_ghosts!(p.U, p.schedule; boundary=bnd)
        end
        t0 = time_ns()
        f = build_fit(state_sampler(hostcopy(p.U), q; t=T(t)), geom, spec;
                      cont=1, bounds=tbounds, backend=backend, check=false)
        fitcost.build_ns[] += time_ns() - t0
        fitcost.nbuild[] += 1
        if f.valid
            fits = (f, fits[1])
        else
            fitcost.nfailed[] += 1
        end
        rows = (fit_valid=f.valid, fit_residual=f.residual.overall,
                fit_min_detγ=f.sweep.min_detγ, fit_min_α=f.sweep.min_α,
                fit_min_λ=f.sweep.min_λ, fit_hits=f.sweep.hits,
                fit_refills=fitcost.chunk_refills[])
        fitcost.chunk_refills[] = 0
        return rows
    end

    # The cache, refilled from the current fits at `t` (step 8e), timed — or,
    # for step 8f's snapshot control, from the state `u` itself.
    function refill(p, t, u)
        t0 = time_ns()
        p′ = if target_source === :snapshot
            fill_snapshot!(p.target, statearray(u, p.U), p.origins, p.spacings,
                           p.interior, T(t))
            with_interior(p, p.interior; fits=fits, t_target=T(t))
        else
            refill_target(p, t; fits=fits)
        end
        fitcost.fill_ns[] += time_ns() - t0
        fitcost.nfill[] += 1
        return p′
    end

    function track!(t, hz, c_pred, forced)
        h_fine = minimum_spacing(T, forest)
        prediction = hz.success === true ?
                     R(sqrt(sum(abs2, SVector{3,T}(Tuple(hz.origin)) - c_pred)) /
                       h_fine) : nothing
        lost = nothing
        try
            tr = update_track(tr, hz, t; max_misses=spec.max_misses, G=G,
                              h=geom.h)
        catch e
            e isa TrackLostError || rethrow()
            lost = e
            tr = e.track
        end
        if lost === nothing
            geom = geometry(forest, t, tr)
            check_interior_radii(forest, kgeom(geom, T(t)), case.background, G;
                                 t=T(t), center=case.center)
            check_bounds_gate(forest, geom, case.bounds, q; t=T(t))
        end
        ct = center_at(track_center(tr), T(t))
        ca = center_at(case.center, T(t))
        leaf_h(x) = (b = locate_block(forest, x);
                     b === nothing ? h_fine : spacing(T, forest, forest.leaves[b]))
        efolds = margin_efolds(case.background, geom, q; t=T(t),
                               ε_KO=case.ε_KO, spacing=leaf_h).min
        rows = (track_source=tr.source, track_center=ntuple(d -> R(ct[d]), 3),
                track_velocity=ntuple(d -> R(tr.v_est[d]), 3),
                track_r_min=R(tr.r_min), track_r_max=R(tr.r_max),
                track_offset=R(sqrt(sum(abs2, ct - ca)) / h_fine),
                track_misses=tr.misses, track_prediction=prediction,
                track_trigger=forced, margin_efolds=efolds,
                layer_h=R(geom.h), layer_offset=R(geom.offset),
                layer_thickness=R(geom.thickness), layer_r_in=R(geom.r_in),
                layer_r_out=R(geom.r_out))
        return rows, lost
    end

    function record!(p, t, u, dt, steps, λ, λ_end, cflnum)
        shell = horizon_shell(case, p.interior)
        gh_constraint!(p, u, t)
        c = constraint_norms(p)
        adm = if adm_every > 0 && iszero(mod(length(records), adm_every))
            adm_constraint!(p, u, t)
            a = constraint_norms(p)
            (ham_l2=a.ham_l2, ham_linf=a.ham_linf,
             mom_l2=maximum(a.mom_l2), mom_linf=maximum(a.mom_linf))
        else
            (ham_l2=nothing, ham_linf=nothing, mom_l2=nothing,
             mom_linf=nothing)
        end
        gh_error!(p, u, t; shell=shell)
        e = error_norms(p)
        # The horizon, every `k`-th chunk of the record — `CODE.md`'s
        # cadence, counted on the same `length(records)` the ADM monitor
        # is. It runs *before* the indicator and after the norms because
        # it scatters and fills ghosts of its own; nothing it does
        # survives into the flags.
        # A tracked run looks from where its track predicts the hole to be,
        # and measures the radii from there — so `center_offset` is the
        # track's prediction error — and a pending lapse-collapse trigger
        # forces the find whatever the cadence (step 8d).
        forced = fitted && trigger_pending
        trigger_pending = false
        c_pred = fitted ? center_at(track_center(tr), T(t)) : nothing
        hz = horizon_row(p, u, t, hlm_seed, length(records); force=forced,
                         origin=c_pred, center=c_pred, find=find)
        hz.hlm === nothing || (hlm_seed = hz.hlm)
        trk, lost = fitted ? track!(t, hz, c_pred, forced) : (no_track, nothing)
        fr = fitmode && lost === nothing ? refit!(p, u, t) : no_fit
        # The indicator last, and only where the case refines: it writes
        # `DIAG_TAU` and reads nothing the norms above left behind, and the
        # flags it produces are what the regrid below uses — one evaluation
        # of the criterion per chunk, so that the number the record holds
        # and the number the mesh was chosen by are the same number.
        ind = case.refinement === nothing ? nothing :
              gh_indicator!(p, u, t; buffer=bufferwidth, travel=travel)
        centroid = ind === nothing || ind.centroid === nothing ? nothing :
                   ntuple(d -> R(ind.centroid[d]), 3)
        # The range projection's counts since the previous row, and the
        # validity monitor (step 8b). Both read the state array and nothing
        # the rows above left in `diag`.
        bh = acc === nothing ? (hits=nothing, nonfinite=nothing,
                                r_max=nothing) : take_chunk!(acc)
        val = validity_rows(p, u, t)
        if fitted && spec.α_trigger > 0 && val.min_α_evolved !== nothing &&
           val.min_α_evolved < spec.α_trigger
            trigger_pending = true
        end
        rec = (t=R(t), dt=R(dt), steps=steps, λ=R(λ), λ_end=R(λ_end),
               cfl=R(cflnum),
               gauge_l2=R(maximum(c.gauge_l2)),
               gauge_linf=R(maximum(c.gauge_linf)),
               ham_l2=adm.ham_l2, ham_linf=adm.ham_linf,
               mom_l2=adm.mom_l2, mom_linf=adm.mom_linf,
               err_l2=R(e.err_l2), err_linf=R(e.err_linf),
               residual=R(e.residual), drift=R(e.drift),
               horizon_success=hz.success, origin=hz.origin,
               center_offset=hz.center_offset, r_min=hz.r_min,
               r_mean=hz.r_mean, r_max=hz.r_max, area=hz.area,
               M_irr=hz.M_irr, J=hz.J, spin_axis=hz.spin_axis,
               M_ch=hz.M_ch, hlm=hz.hlm, horizon_note=hz.note,
               ρ_max=R(interior_ρ_max(p.interior)),
               variant=interior_variant(p.interior),
               τ_max=ind === nothing ? nothing : R(ind.τ_max),
               centroid=centroid,
               centroid_offset=centroid === nothing ? nothing :
                               R(centroid_offset(case, t, ind.centroid)),
               bounds_hits=bh.hits, bounds_nonfinite=bh.nonfinite,
               bounds_r_max=bh.r_max, val..., trk..., fr...,
               nblocks=nleaves(forest), levels=forest_levels(forest),
               h=R(minimum_spacing(T, forest)),
               finite=evolved_nonfinite(p, u, t) == 0)
        push!(records, rec)
        observer === nothing || observer(p, t, u)
        lost === nothing ||
            throw(TrackLostError(lost.msg, lost.track, Any[records...]))
        return ind
    end

    # `t = 0`, before anything has been integrated: the record's first row
    # is the initial data's own, which is what makes "the error grew from
    # zero" a statement a test can check rather than assume.
    λ_initial = max_speed_of(p0, u, zero(T))
    dt0 = cfl * minimum_spacing(T, forest) / λ_initial
    p0 = with_interior(p0, chunk_interior(case, dt0, ρ_max_factor,
                                          ρ_max_fixed; default=ρ_max_default,
                                          interior=kgeom(geom, zero(T))))
    # The initial data has not been through a stage, so neither limiter
    # has seen it: the range projection first, in the order RK4 applies
    # the two (stage, then step), and the paste after it **(proposed in
    # step 8b** — on analytic data neither fires**)**.
    apply_bounds!(p0, u, zero(T))
    paste_interior!(p0, u, zero(T))
    record!(p0, zero(T), u, zero(T), 0, λ_initial, λ_initial, zero(T))

    nchunks = ceilint(t_end / chunk)
    p = p0
    # **A moving hole's step is sized for the speed it will have (proposed in
    # step 8f).** `λ_max` is measured at the chunk's start and the recheck
    # throws if the speed at its end asks for a smaller step; a hole crossing
    # the box raises the fastest speed monotonically, by 0.1–0.3 % a chunk of
    # `M/4` on step 8f's boosted rows, and a step that rounds to within that
    # of the requested one then fails the recheck at any `cfl` (measured:
    # chunk 7 at `cfl = 1/4`, chunk 5 at `1/5`). So for a case whose hole
    # moves, the step is sized from `λ` times the square of the growth the
    # previous chunk measured, `(λ_end/λ)²` when that is above one — a
    # static hole's step is unchanged, bit for bit. The recheck stays.
    # **And never by less than 1 % (amended in step 8)**: the growth is not
    # monotonic once the hole crosses a mesh finer than step 8f's capsule —
    # on step 8's uniform `5/128` control the speed grew 0.27 % in chunk 4
    # after a quieter chunk 3, and the recheck stopped the run at a CFL
    # number of `0.20003` against `0.2`. A 1 % margin costs 1 % of the steps.
    moving = sum(abs2, case.center.v) > 0
    growth = one(T)
    for c in 1:nchunks
        tstart = min((c - 1) * chunk, t_end)
        stop = min(c * chunk, t_end)
        stop > tstart || break

        # (0) a tracked run's geometry for this chunk (step 8d), built from
        # the track at the previous row. The gauge source was sampled with
        # the core rule of an earlier geometry: a point that core released
        # into the layer reads the source of its projection, weighted by a
        # `w` that vanishes to second order at the core surface — harmless
        # while the surface has moved by a small fraction of a cell, and
        # re-sampled, by rebuilding the problem, once it has moved half of
        # one (proposed in step 8d).
        if fitted
            if p.Hsrc !== nothing &&
               surface_shift(geom_sampled, geom, tstart) > geom.h / 2
                p = GHProblem(U, schedule, case; q=q, t=tstart,
                              interior=kgeom(geom, tstart), accounting=acc,
                              target=p.target, fits=p.fits,
                              t_target=p.t_target, target_rate=rate_on)
                geom_sampled = geom
                nresamples += 1
            end
            p = with_interior(p, with_ρ_max(kgeom(geom, tstart),
                                            interior_ρ_max(p.interior)))
        end

        # (1) the step, and with it this chunk's relaxation rate.
        λ = max_speed_of(p, u, tstart)
        h_min = minimum_spacing(T, forest)
        dt = cfl * h_min / (moving ? λ * max(growth * growth, T(101 // 100)) : λ)
        steps = max(1, ceilint((stop - tstart) / dt))
        dt_used = (stop - tstart) / steps
        p = with_interior(p, chunk_interior(case, dt_used, ρ_max_factor,
                                            ρ_max_fixed;
                                            default=ρ_max_default,
                                            interior=fitted ? kgeom(geom, tstart) :
                                                     geom))
        # The fitted target for this chunk (step 8e): the cache refilled from
        # the fits the last row built, at the chunk's start — and, for a
        # geometry that moves, again whenever the tracked center would move
        # by more than `h/4` since the last fill: the chunk's steps are split
        # into `npieces` solves of at most `h/(4|v|)` each, with a refill
        # between (proposed in step 8e, over lowering the chunk: the record's
        # cadence stays the case's). A static hole's `v_est` is the finder's
        # noise, a ten-thousandth of a cell per `M`, and is one piece.
        npieces = 1
        if refilling(tstart)
            p = refill(p, tstart, u)
            speed = sqrt(sum(abs2, tr.v_est))
            npieces = speed > 0 ?
                      clamp(ceilint(speed * (stop - tstart) / (geom.h / 4)), 1,
                            steps) : 1
        end

        # `step_limiter` on `solve` and not `RK4(; step_limiter! = …)`:
        # the constructor form is deprecated in the resolved
        # `OrdinaryDiffEqCore` and warns once per solve, which is once per
        # chunk (noted in step 5; `PLAN.md`'s sharp edge named the older
        # spelling). The hook and its signature are unchanged. The
        # `stage_limiter` beside it is step 8b's range projection, a no-op
        # for a case without bounds; both are `solve` keywords for the same
        # reason.
        tcur = tstart
        for piece in 1:npieces
            nk = (steps * piece) ÷ npieces - (steps * (piece - 1)) ÷ npieces
            tnext = piece == npieces ? stop : tcur + nk * dt_used
            if piece > 1
                p = refill(p, tcur, u)
                fitcost.nrefills[] += 1
                fitcost.chunk_refills[] += 1
            end
            sol = solve(ODEProblem(gh_rhs!, u, (tcur, tnext), p), RK4();
                        dt=dt_used, adaptive=false, save_everystep=false,
                        stage_limiter=gh_stage_limiter!,
                        step_limiter=gh_step_limiter!)
            u = sol.u[end]
            tcur = tnext
        end
        nsteps += steps

        # (2) the recheck. It throws, and it is meant to.
        λ_end = max_speed_of(p, u, stop)
        growth = max(one(T), λ_end / λ)
        cflnum = check_cfl(dt_used, h_min, cfl, λ_end; chunk=c, λ=λ)

        # (3) the record, and whatever is watching — before the regrid that
        # would invalidate the mesh the record describes.
        ind = record!(p, stop, u, dt_used, steps, λ, λ_end, cflnum)

        # (4) the regrid, on the flags the record has already computed from
        # this state, with this `t`'s hook. **Not after the last chunk**:
        # the forest that comes back is then the one the returned state was
        # computed on. `regrid!` is handed the state field set alone — the
        # fresh `GHProblem` below allocates a new `diag` and re-samples the
        # gauge source on the new mesh, so resizing either of them through
        # the transfer would be work thrown away (amended in step 6;
        # `CODE.md`'s loop lists all three).
        if regrid && c < nchunks && ind !== nothing
            moved = regrid!(forest, U => schedule; flags=ind.flags, buffer=0,
                            boundary=dirichlet(case, stop))
            if moved
                nregrids += 1
                schedule = GhostSchedule(U, ops)
                # A tracked geometry is rebuilt on the new mesh from the same
                # track — its offset and ramp are stated in the mesh's
                # spacings — and the problem's constructor asserts it, as it
                # asserts the sphere (step 8d).
                if fitted
                    geom = geometry(forest, stop, tr)
                    geom_sampled = geom
                end
                # A fitted run's cache is made anew on the new mesh and filled
                # at the next chunk's start from the same fits — a fit is a
                # polynomial in `x` and does not know the mesh (proposed in
                # step 8e).
                p = GHProblem(U, schedule, case; q=q, t=stop,
                              interior=fitted ? kgeom(geom, stop) : geom,
                              accounting=acc,
                              target=fitmode ? target_cache(U) : nothing,
                              fits=fits, t_target=stop, target_rate=rate_on)
                u = statevector(U)
                gather!(u, U)
                # The transferred state has not been through a step, so
                # neither limiter has run on it: the range projection (a
                # no-op without bounds) — the prolongation into a fresh fine
                # block is unlimited and can leave an owned point outside
                # every range, TreeHydro's reason for the same call — and
                # the `:pasted` variant's paste, which every other variant
                # compiles away.
                apply_bounds!(p, u, stop)
                paste_interior!(p, u, stop)
            end
        end
    end

    return (records=records, nsteps=nsteps, nchunks=length(records) - 1,
            nregrids=nregrids, passes=passes, converged=converged,
            buffer=bufferwidth, bounds=acc,
            λ_initial=R(λ_initial), h=R(minimum_spacing(T, forest)),
            nblocks=nleaves(forest), levels=forest_levels(forest),
            interior=p.interior, problem=p, U=U, u=u, forest=forest,
            track=tr, geometry=fitted ? geom : nothing, n_L=n_L,
            nresamples=nresamples, fits=fits, fit_initial=fit_initial,
            target_bounds=tbounds, handover=R(handover_t),
            target_source=target_source,
            fit_cost=fitmode ?
                     (build_ms=fitcost.build_ns[] / 1e6, nbuild=fitcost.nbuild[],
                      fill_ms=fitcost.fill_ns[] / 1e6, nfill=fitcost.nfill[],
                      nrefills=fitcost.nrefills[], nfailed=fitcost.nfailed[]) :
                     nothing)
end

"""
    fill_fitted_initial!(U, case, geom; cont = 1, depth = 0, backend = CPU())
        -> (; bounds, fit, target)

A `:fitted` case's initial data on `U`'s mesh for the geometry `geom`
(decided in review, step 8e): the target ranges (the spec's, or derived from
the analytic data on the offset surface), the `cont` fit of the **analytic**
solution on it from `Float64` samples, the target cache filled from that fit
at `t = 0`, and the state — the analytic solution outside the depth `depth`
below the offset surface and the cache inside it ([`fitted_state_kernel!`](@ref))
— written into `U`'s owned points. Ghosts are not filled.

What [`evolve!`](@ref) fills a `:fitted` run with, and what
[`adapt_fitted_initial_data!`](@ref) flags on at every pass (factored out in
step 8, so that the two are one computation).
"""
function fill_fitted_initial!(U::FieldSet{T,3}, case::GHCase{T},
                              geom::FittedInterior; cont::Integer=1, depth=0,
                              backend=get_backend(U.work),
                              rate::Bool=false) where {T}
    spec = case.interior
    spec isa FittedSpec || throw(ArgumentError(
        "fill_fitted_initial! fills a tracked :fitted case, whose interior is " *
        "a FittedSpec; this case's is $(typeof(spec))."))
    bounds = spec.target_bounds !== nothing ? _bounds_in(T, spec.target_bounds) :
             derive_target_bounds(T, case.background, geom; t=0,
                                  L=spec.lmax_fit)
    fit = build_fit(analytic_sampler(case.background, 0.0;
                                     δ=tofloat64(geom.h) / 8),
                    geom, spec; cont=cont, bounds=bounds, backend=backend)
    target = target_cache(U)
    origins = to_backend(backend, block_origins(U.forest, T))
    spacings = to_backend(backend, block_spacings(U.forest, T))
    fill_target!(target, origins, spacings, geom, (fit, nothing), zero(T);
                 rate=rate)
    u = statevector(U)
    map_blocks!(fitted_state_kernel!, U, statearray(u, U), target.work,
                origins, spacings, case.background, geom, zero(T), zero(T),
                T(depth))
    scatter!(U, u)
    return (bounds=bounds, fit=fit, target=target)
end

"""
    adapt_fitted_initial_data!(U, ops, case, geometry; G, buffer, travel = 0,
                               maxpasses = 8, cont = 1, depth = 0,
                               backend = CPU())
        -> (schedule, passes, converged, geom)

`CODE.md`'s initial-data cycle for a tracked **`:fitted`** case, flagging on
the data the run will start from: on every pass the geometry is rebuilt on
the current mesh (`geometry(forest)`, the driver's `fitted_interior` of the
seed track), the case's initial data is filled on it by
[`fill_fitted_initial!`](@ref) — the analytic solution outside the offset
surface, the fit of it inside — the ghosts are filled with the `t = 0` hook,
the masked indicator ([`indicator_flags`](@ref), with the geometry's mask and
floor) flags, and `regrid!` rebuilds the mesh **without transferring**, until
the hierarchy stops changing. TreeAMR's `adapt_to_initial_data!` is this
loop for a coordinate callback; a `:fitted` case's data lives partly in a
cache that a callback cannot read, so the loop is written out here with the
same four calls in the same order.

**Why (added in step 8).** Step 8f's cycle chose a `:fitted` case's mesh on
the analytic `:damped` data of the same geometry and refused a chart whose
analytic core surface meets its singular set — G5's own, harmonic Kerr at
`a = 7/10`, where there is then nothing to flag on. The indicator is masked
inside the offset surface, so outside it the two data are the same analytic
solution and flag the same blocks; what this cycle adds is that the one
point of Löhner's stencil that reaches inside reads the fit, which is finite
on every chart, rather than the core rule's analytic data, which is not.
The returned `geom` is the geometry of the mesh the cycle settled at; the
state in `U` is that mesh's initial data.

`buffer` and `travel` are the driver's: the travelling margin, applied
inside the indicator (so `regrid!` gets `buffer = 0`), and the distance the
level floor is widened by.
"""
function adapt_fitted_initial_data!(U::FieldSet{T,3}, ops, case::GHCase{T},
                                    geometry; G::Integer, buffer::Integer,
                                    travel=zero(T), maxpasses::Integer=8,
                                    cont::Integer=1, depth=0,
                                    backend=get_backend(U.work)) where {T}
    forest = U.forest
    boundary = dirichlet(case, zero(T))
    fill!(fs, sched) = boundary === nothing ? fill_ghosts!(fs, sched) :
                       fill_ghosts!(fs, sched; boundary=boundary)
    schedule = GhostSchedule(U, ops)
    geom = geometry(forest)
    fill_fitted_initial!(U, case, geom; cont=cont, depth=depth,
                         backend=backend)
    for pass in 1:maxpasses
        fill!(U, schedule)
        flags = indicator_flags(U, case, zero(T); G=G, buffer=buffer,
                                interior=geom, travel=travel).flags
        changed = boundary === nothing ?
                  regrid!(forest, U => schedule; flags=flags, buffer=0,
                          transfer=false) :
                  regrid!(forest, U => schedule; flags=flags, buffer=0,
                          boundary=boundary, transfer=false)
        schedule = GhostSchedule(U, ops)
        geom = geometry(forest)
        fill_fitted_initial!(U, case, geom; cont=cont, depth=depth,
                             backend=backend)
        changed || return (schedule, pass, true, geom)
    end
    return (schedule, Int(maxpasses), false, geom)
end

# The distance from the indicator's centroid to the hole's analytic center
# at this time — `CODE.md`'s "the refinement centroid against the analytic
# center", which G5 asks to stay within a few finest spacings. A case with
# no hole measures it against the origin, which is where its `HoleCenter`
# already is.
function centroid_offset(case::GHCase{T}, t, centroid) where {T}
    c = center_at(case.center, T(t))
    return sqrt(sum(abs2, SVector{3,T}(centroid) - c))
end

# `max_speed` reads the working array, so the state has to be there first.
# Named, because the driver does it three times a chunk and forgetting the
# scatter is a speed measured on the previous chunk — and because the
# answer has to be **checked** here rather than three lines later: `dt` is
# `cfl·h/λ`, and a `λ` of `NaN` or zero reaches `ceilint` as an
# `InexactError: Int64(NaN)`, which says nothing about what went wrong.
# `gh_dt` makes the same check for the same reason (added in step 5, after
# a harmonic-Kerr configuration whose initial data was not finite in the
# evolved region produced exactly that error).
#
# **The check is the evolved region's (amended in step 8b).** It was
# `all(isfinite, u)`, which ended a run at the first `NaN` anywhere — the
# frozen core included, where nothing evolved reads it until the layer's
# stencils do, and where the range projection exists to repair it. A `NaN`
# in the core is a hit; a `NaN` at `r ≥ r_1` is still the end.
function max_speed_of(p::GHProblem{T}, u, t) where {T}
    nbad = evolved_nonfinite(p, u, t)
    nbad == 0 || throw(ArgumentError(
        "the state at t = $t is not finite where it is evolved, so there is " *
        "no time step: $nbad values at points outside the interior (r ≥ r_1, " *
        "or everywhere for a case with no hole) are NaN or Inf. On the " *
        "initial data this is a case whose analytic solution is singular " *
        "somewhere the mesh reaches — check r_0 against where |h| is still " *
        "moderate — and after a chunk it is a run that blew up, whose last " *
        "analysis record says where."))
    scatter!(p.U, u)
    λ = max_speed(p; t=t)
    isfinite(λ) && λ > 0 || throw(ArgumentError(
        "the maximum characteristic speed α√(tr γ^{ij}) + |β| over the " *
        "evolved points came out as $λ at t = $t, so there is no " *
        "CFL-limited time step. The interior is masked out of this " *
        "reduction (CODE.md, \"The time step\"), so a non-finite answer is " *
        "a state outside r_1 that is no longer a metric, and a zero one is " *
        "a state that is identically zero — which no initial data of this " *
        "package produces, h = 0 being Minkowski, whose speed is √3."))
    return λ
end

# The horizon rows of one record entry: the find at `t` when this is a
# `k`-th chunk and the case asks for one, and a row of `nothing` otherwise.
# `CODE.md`'s analysis table, "every `k`-th chunk, `k` a case parameter".
#
# **The find is caught (proposed in step 7).** The horizon is a
# *diagnostic* — nothing in the evolution reads it — and a diagnostic that
# ends a run is worse than one that says it failed: the interpolating
# provider throws by design where a query reaches the layer, and a flow
# that wanders inward for one chunk would otherwise take the whole run and
# its record with it. What the record holds instead is
# `horizon_success = false` and the message in `horizon_note`, beside the
# chunk it happened at, and a test asserts on `horizon_success` rather than
# on the absence of an exception.
#
# **A tracked run passes three more things (step 8d)**: `force`, the
# lapse-collapse trigger, which runs the find whatever the cadence; `origin`
# and `center`, the tracked center predicted to `t`, which the find starts
# from and measures its radii from; and `find`, the function called — the
# finder, or a test's wrapper around it. The row also carries the radii about
# the found origin and the finder's grid, which the track is updated from and
# the record does not keep.
function horizon_row(p::GHProblem{T}, u, t, seed, index; force::Bool=false,
                     origin=nothing, center=nothing,
                     find=find_gh_horizon) where {T}
    empty = (success=nothing, origin=nothing, center_offset=nothing,
             r_min=nothing, r_mean=nothing, r_max=nothing, area=nothing,
             M_irr=nothing, J=nothing, spin_axis=nothing, M_ch=nothing,
             hlm=nothing, note=nothing, origin_r_min=nothing,
             origin_r_max=nothing, grid=nothing)
    hz = p.case.horizon
    hz === nothing && return empty
    force || (hz.every > 0 && iszero(mod(index, hz.every))) || return empty
    out = try
        find(p, u, t; N=hz.N,
             r_seed=iszero(hz.r_seed) ? nothing : tofloat64(hz.r_seed),
             origin=origin, center=center, hlm=seed, spin=hz.spin,
             unif_tol=hz.unif_tol, atol=hz.atol, maxiters=hz.maxiters,
             verbosity=hz.verbosity)
    catch e
        e isa InterruptException && rethrow()
        return merge(empty, (success=false,
                             note=first(split(sprint(showerror, e), '\n'))))
    end
    return (success=out.success, origin=Tuple(out.origin),
            center_offset=out.center_offset, r_min=out.r_min,
            r_mean=out.r_mean, r_max=out.r_max, area=out.area,
            M_irr=out.M_irr, J=out.J, spin_axis=Tuple(out.spin_axis),
            M_ch=out.M_ch, hlm=out.hlm, note=nothing,
            origin_r_min=out.origin_r_min, origin_r_max=out.origin_r_max,
            grid=out.grid)
end

"""
    default_relaxation_rate(case::GHCase) -> T

The layer's relaxation rate when [`evolve!`](@ref) is given no rate keyword:
**`ρ_max = 4/M`**, `M` the hole's mass parameter ([`hole_mass`](@ref) of the
case's background), in the case's own units and type. This is the one place
the `4` is written.

**Why a physical rate and not the grid rate `1/dt` (decided 2026-09-23, step
8c′, on step 8c's measurement).** `ρ_max · dt = 1` — step 5's default —
relaxes by a factor `e` per step, which on the suite's fixture at
`cfl = 1/5` is about `107/M` against the hole's surface gravity
`κ = 1/(4M)`: with the quintic ramp it is `37/M` two cells inside `r_1`, so
the layer is a **paste two cells deep** with a two-cell transition. A
target that is not an exact solution, pinned that hard that close to the
evolved stencils, is a kink the compact second derivative turns into an
`O(1)` right-hand-side error, and every inexact target step 8c tried ended
its run at the grid rate on every ramp up to 8 cells; at `4/M` on a ramp of
at least `4G` cells the same targets leave the exterior where the exact one
leaves it. Even on the **exact** target, where the paste survives, it ends
a `50 M` run with six times the error: the shell's `C_a` L2 `0.181`
against `0.029` on the fixture's own layer (`CODE.md`, "The layer for an
inexact target"). `4/M` is `16 κ`, and it is fast: the layer forgets a
perturbation in `M/4`. The grid rate stays available as `ρ_max_factor`.

It asks the background for its mass and nothing else, so a case whose
background has none is refused by `hole_mass`, by name.
"""
default_relaxation_rate(case::GHCase{T}) where {T} =
    _spec_rate(case.interior, T) > 0 ? _spec_rate(case.interior, T) :
    T(4) / T(hole_mass(case.background))

# A tracked case may state its own default rate (step 8d); `0` is the rule.
_spec_rate(int, ::Type{T}) where {T} = zero(T)
_spec_rate(spec::FittedSpec, ::Type{T}) where {T} = T(spec.ρ_max)

# The interior the kernel sees this chunk: the case's radii and variant at
# `ρ_max = factor/dt`, or at the fixed rate where one is given — the
# default `4/M` is one (step 8c′). `nothing` stays `nothing`.
#
# **The guard `fixed · dt ≤ 1` at the default (checked in step 8c′).** On
# every hole the suite evolves the default is a twenty-eighth to an
# eleventh of the grid rate — `4/M · dt`, over the chunks and the `t = 0`
# row, is `0.040–0.047` on the fixture at `h = 5/64`, `cfl = 1/4`
# (`dt ≈ 0.01 M`), `0.036–0.037` at `N = 10`, `0.057–0.062` at its coarsest
# `N = 6`, and `0.067–0.092` on the refinement's fixture at `h = 5/32`, both
# of its runs — so the guard never fires on a mesh that resolves a hole, and
# when it does, the step is longer than `M/4` and the refusal says the mesh,
# not the rate, is what is wrong.
chunk_interior(case::GHCase, dt, factor) = chunk_interior(case, dt, factor,
                                                          nothing)

function chunk_interior(case::GHCase, dt, factor, fixed; default::Bool=false,
                        interior=case.interior)
    # `interior` is the geometry this chunk runs on: the case's own sphere, or
    # a tracked case's geometry built from the track (step 8d).
    interior === nothing && return nothing
    fixed === nothing && return with_ρ_max(interior, factor / dt)
    fixed * dt ≤ 1 || throw(ArgumentError(default ?
        "the default relaxation rate ρ_max = 4/M = $fixed is above this " *
        "chunk's grid rate 1/dt = $(1 / dt): the step is longer than a " *
        "quarter of the hole's mass, so the mesh does not resolve the time " *
        "scale the default is stated in. Refine the mesh around the hole, or " *
        "pass ρ_max_factor for a grid rate or a lower ρ_max_fixed." :
        "the fixed relaxation rate ρ_max = $fixed is above this chunk's grid " *
        "rate 1/dt = $(1 / dt): a fixed rate is the physical alternative to " *
        "the grid rate ρ_max · dt = 1, and one above it relaxes harder than " *
        "a paste, up to RK4's real-axis limit of 2.79/dt — which a run then " *
        "finds as a blow-up in the layer. Lower ρ_max_fixed, or pass " *
        "ρ_max_factor for a grid rate."))
    return with_ρ_max(interior, fixed)
end

# What the record reports as this chunk's relaxation rate.
interior_ρ_max(::Nothing) = 0
interior_ρ_max(int::Interior) = int.ρ_max
interior_ρ_max(int::FittedInterior) = int.ρ_max

"""
    discrete_gradient_momentum!(U::FieldSet, case::GHCase, t, q, schedule)

GHSO2's **discrete-gradient `Π` post-pass**: recompute `Π` from the
*differenced* spatial gradients of `h` rather than from the background's
analytic ones, so that a static solution has `∂_t g = 0` to roundoff rather
than to truncation order.

`CODE.md`, "Initial data and backgrounds", keeps this as an option after
the initial-data cycle converges and asks G4 to measure whether it changes
a hole's stationarity visibly **(predicted: not beyond the first chunk)**.
The relation inverted is the first evolution equation,

    ∂_t g_ab = β^i ∂_i g_ab + (α/√γ) Π_ab ,

with `∂_t g` the background's analytic value: `Π = (√γ/α)(∂_t g − β^i D_i
h)` with `D_i` the scheme's own first-derivative operator. Where the
background is static the first term is zero and this is exactly "the
momentum that makes the discrete right-hand side vanish".

It fills ghosts first, because `D_i` reaches `q/2` points past the owned
range, and it writes the state — which is legitimate here and only here:
this runs on the initial data, before the integrator exists.
"""
function discrete_gradient_momentum!(U::FieldSet{T,3}, case::GHCase{T}, t,
                                     q::Integer, schedule;
                                     interior=case.interior) where {T}
    interior isa FittedSpec && throw(ArgumentError(
        "the Π post-pass applies the core rule, which on a tracked case needs " *
        "the geometry: pass `interior = fitted_interior(…)` (step 8d)."))
    boundary = dirichlet(case, t)
    if boundary === nothing
        fill_ghosts!(U, schedule)
    else
        fill_ghosts!(U, schedule; boundary=boundary)
    end
    origins = to_backend(get_backend(U.work), block_origins(U.forest, T))
    spacings = to_backend(get_backend(U.work), block_spacings(U.forest, T))
    map_blocks!(discrete_momentum_kernel!, U, U.work, origins, spacings,
                case.background, interior, T(t), Val(U.G), Val(Int(q)))
    return U
end

@kernel function discrete_momentum_kernel!(work, @Const(origins),
                                           @Const(spacings), bg, interior, t,
                                           ::Val{G}, ::Val{q}) where {G,q}
    I = @index(Global, NTuple)
    b = I[4]
    T = eltype(work)
    inv_h = inv(spacings[b])
    w1 = derivative_weights(T, Val(q), Val(1))
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb +
          (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]

    hv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (v - 1) * sv]),
                              Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, work, var + (v - 1) * sv, st[d])
        end)
    end
    _, _, α, β, _, sqrtγ = metric_quantities(_sym4(hv))

    x = point_position(origins, spacings, b, I)
    xe = core_position(interior, t, x)
    D = T
    g, dg = dmetric(bg, SVector{4,D}(D(t), xe[1], xe[2], xe[3]))
    ∂ₜg = pack_sym(SMatrix{4,4,D}(dg[a, c, 1] for a in 1:4, c in 1:4))
    Π = (sqrtγ / α) * (∂ₜg - β[1] * ∂h[1] - β[2] * ∂h[2] - β[3] * ∂h[3])

    ntuple(Val(NC)) do v
        work[var + (NC + v - 1) * sv] = Π[v]
        nothing
    end
end
