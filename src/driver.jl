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
#   * **`ρ_max` is per chunk.** `CODE.md` bounds the relaxation rate by
#     RK4's stability on the negative real axis and sets `ρ_max · dt = 1`,
#     so the interior the kernel sees is rebuilt from *this* chunk's `dt`
#     ([`with_interior`](@ref)) — not from the case, which carries a
#     placeholder. A run that blows up in the layer after raising `ρ_max`
#     has found the integrator, not the physics.
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
function horizon_shell(case::GHCase{T}) where {T}
    case.interior === nothing && return (zero(T), -one(T))
    lo = T(horizon_min_radius(case.background))
    hi = T(horizon_max_radius(case.background)) +
         (case.interior.r_1 - case.interior.r_0)
    return (lo, hi)
end

"""
    evolve!([T], case::GHCase; forest, q, ops, t_end, chunk = case's,
            cfl = 1//4, regrid = false, adapt = false, buffer = nothing,
            maxpasses = 8, adm_every = 0, backend = CPU(),
            observer = nothing, ρ_max_factor = 1, ρ_max_fixed = nothing)

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
from [`max_speed`](@ref) at the chunk's start, and `ρ_max = ρ_max_factor /
dt` for the interior — `CODE.md`'s `ρ_max · dt = 1`, which relaxes by a
factor `e` per step and stays well inside RK4's stability limit of about
`2.8/dt` on the negative real axis. At the chunk's end the speed is
measured again and [`check_cfl`](@ref) **throws** if the step actually
taken violated the bound.

**A fixed rate instead (added in step 8c).** `ρ_max_fixed` is a relaxation
rate in the case's units — `4/M`, say — used as `ρ_max` in every chunk
instead of `ρ_max_factor/dt`. `CODE.md`'s `1/dt` is a *grid* rate (about
`107/M` on the suite's fixture), which is a paste two cells deep; a target
that is not an exact solution needs a physical one, and step 8c's
calibration is where the two are compared. The two keywords are exclusive
and refused together, and a fixed rate above `1/dt` is refused at the
chunk that would take it — it would be stronger than the grid rate the
default already is, and RK4's real-axis limit is `2.79/dt`
**(proposed in step 8c)**. The record's `ρ_max` row is the rate the chunk
ran at either way.

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
                 ρ_max_factor=nothing, ρ_max_fixed=nothing) where {T}
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
            "factor/dt (CODE.md's ρ_max · dt = 1, the default) or a fixed " *
            "physical rate (step 8c). Pass one of them."))
    ρ_max_factor = ρ_max_factor === nothing ? one(T) : T(ρ_max_factor)
    ρ_max_factor > 0 || throw(ArgumentError(
        "ρ_max_factor scales CODE.md's ρ_max · dt = 1 and must be positive, " *
        "got $ρ_max_factor; it exists so that a test may measure what the " *
        "bound is worth, not so that a run may switch the layer off."))
    ρ_max_fixed === nothing || T(ρ_max_fixed) > 0 || throw(ArgumentError(
        "ρ_max_fixed is the layer's relaxation rate and must be positive, " *
        "got $ρ_max_fixed; a rate of zero is the :frozen variant, which is " *
        "selected by name."))
    ρ_max_fixed = ρ_max_fixed === nothing ? nothing : T(ρ_max_fixed)

    G = q ÷ 2 + 1
    U = FieldSet{T}(forest, 2NC; G=G, centering=vertexcentered(3),
                    backend=backend)

    # The travelling margin, in cells at the finest level the indicator may
    # reach: `CODE.md`'s `|v| · chunk`, the hole's own speed times the
    # regrid cadence, plus one — TreeAMR measured that a margin narrower
    # than the motion it covers is worse than none at all. A static hole
    # still gets the one cell.
    speed = sqrt(sum(abs2, case.center.v))
    bufferwidth = buffer !== nothing ? Int(buffer) :
                  case.refinement === nothing ? 0 :
                  refinement_buffer(forest, case.refinement.maxlevel_cap,
                                    speed * chunk)

    # The initial-data cycle, or the plain fill. The cycle fills the data
    # itself — that is what it is for: it re-evaluates it on each new mesh
    # rather than interpolating, since interpolating would bake the coarse
    # mesh's resolution into the blocks the refinement just bought.
    passes = 0
    converged = true
    schedule = GhostSchedule(U, ops)
    if adapt
        # The margin is dilated inside the indicator, so that the level
        # ceiling is applied *after* it — `buffer = 0` here and there is
        # what `refine_flags` means by "the caller passes zero".
        criterion(fs) = indicator_flags(fs, case, zero(T); G=G,
                                        buffer=bufferwidth).flags
        schedule, passes, converged = adapt_to_initial_data!(
            U, ops; initial=state_callback(case, zero(T)), flags=criterion,
            buffer=0, maxpasses=maxpasses,
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
    else
        fill_exact!(U, case, zero(T))
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
    p0 = GHProblem(U, schedule, case; q=q, t=zero(T), accounting=acc)
    shell = horizon_shell(case)
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
    hlm_seed = nothing

    function record!(p, t, u, dt, steps, λ, λ_end, cflnum)
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
        hz = horizon_row(p, u, t, hlm_seed, length(records))
        hz.hlm === nothing || (hlm_seed = hz.hlm)
        # The indicator last, and only where the case refines: it writes
        # `DIAG_TAU` and reads nothing the norms above left behind, and the
        # flags it produces are what the regrid below uses — one evaluation
        # of the criterion per chunk, so that the number the record holds
        # and the number the mesh was chosen by are the same number.
        ind = case.refinement === nothing ? nothing :
              gh_indicator!(p, u, t; buffer=bufferwidth)
        centroid = ind === nothing || ind.centroid === nothing ? nothing :
                   ntuple(d -> R(ind.centroid[d]), 3)
        # The range projection's counts since the previous row, and the
        # validity monitor (step 8b). Both read the state array and nothing
        # the rows above left in `diag`.
        bh = acc === nothing ? (hits=nothing, nonfinite=nothing,
                                r_max=nothing) : take_chunk!(acc)
        val = validity_rows(p, u, t)
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
               τ_max=ind === nothing ? nothing : R(ind.τ_max),
               centroid=centroid,
               centroid_offset=centroid === nothing ? nothing :
                               R(centroid_offset(case, t, ind.centroid)),
               bounds_hits=bh.hits, bounds_nonfinite=bh.nonfinite,
               bounds_r_max=bh.r_max, val...,
               nblocks=nleaves(forest), levels=forest_levels(forest),
               h=R(minimum_spacing(T, forest)),
               finite=evolved_nonfinite(p, u, t) == 0)
        push!(records, rec)
        observer === nothing || observer(p, t, u)
        return ind
    end

    # `t = 0`, before anything has been integrated: the record's first row
    # is the initial data's own, which is what makes "the error grew from
    # zero" a statement a test can check rather than assume.
    λ_initial = max_speed_of(p0, u, zero(T))
    dt0 = cfl * minimum_spacing(T, forest) / λ_initial
    p0 = with_interior(p0, chunk_interior(case, dt0, ρ_max_factor,
                                          ρ_max_fixed))
    # The initial data has not been through a stage, so neither limiter
    # has seen it: the range projection first, in the order RK4 applies
    # the two (stage, then step), and the paste after it **(proposed in
    # step 8b** — on analytic data neither fires**)**.
    apply_bounds!(p0, u, zero(T))
    paste_interior!(p0, u, zero(T))
    record!(p0, zero(T), u, zero(T), 0, λ_initial, λ_initial, zero(T))

    nchunks = ceilint(t_end / chunk)
    p = p0
    for c in 1:nchunks
        tstart = min((c - 1) * chunk, t_end)
        stop = min(c * chunk, t_end)
        stop > tstart || break

        # (1) the step, and with it this chunk's relaxation rate.
        λ = max_speed_of(p, u, tstart)
        h_min = minimum_spacing(T, forest)
        dt = cfl * h_min / λ
        steps = max(1, ceilint((stop - tstart) / dt))
        dt_used = (stop - tstart) / steps
        p = with_interior(p, chunk_interior(case, dt_used, ρ_max_factor,
                                            ρ_max_fixed))

        # `step_limiter` on `solve` and not `RK4(; step_limiter! = …)`:
        # the constructor form is deprecated in the resolved
        # `OrdinaryDiffEqCore` and warns once per solve, which is once per
        # chunk (noted in step 5; `PLAN.md`'s sharp edge named the older
        # spelling). The hook and its signature are unchanged. The
        # `stage_limiter` beside it is step 8b's range projection, a no-op
        # for a case without bounds; both are `solve` keywords for the same
        # reason.
        sol = solve(ODEProblem(gh_rhs!, u, (tstart, stop), p), RK4();
                    dt=dt_used, adaptive=false, save_everystep=false,
                    stage_limiter=gh_stage_limiter!,
                    step_limiter=gh_step_limiter!)
        u = sol.u[end]
        nsteps += steps

        # (2) the recheck. It throws, and it is meant to.
        λ_end = max_speed_of(p, u, stop)
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
                p = GHProblem(U, schedule, case; q=q, t=stop, accounting=acc)
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
            interior=p.interior, problem=p, U=U, u=u, forest=forest)
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
function horizon_row(p::GHProblem{T}, u, t, seed, index) where {T}
    empty = (success=nothing, origin=nothing, center_offset=nothing,
             r_min=nothing, r_mean=nothing, r_max=nothing, area=nothing,
             M_irr=nothing, J=nothing, spin_axis=nothing, M_ch=nothing,
             hlm=nothing, note=nothing)
    hz = p.case.horizon
    hz === nothing && return empty
    hz.every > 0 && iszero(mod(index, hz.every)) || return empty
    out = try
        find_gh_horizon(p, u, t; N=hz.N,
                        r_seed=iszero(hz.r_seed) ? nothing :
                               tofloat64(hz.r_seed),
                        hlm=seed, spin=hz.spin, unif_tol=hz.unif_tol,
                        atol=hz.atol, maxiters=hz.maxiters,
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
            M_ch=out.M_ch, hlm=out.hlm, note=nothing)
end

# The interior the kernel sees this chunk: the case's radii and variant at
# `ρ_max = factor/dt`, or at the fixed rate where one is given (step 8c).
# `nothing` stays `nothing`.
chunk_interior(case::GHCase, dt, factor) = chunk_interior(case, dt, factor,
                                                          nothing)

function chunk_interior(case::GHCase, dt, factor, fixed)
    case.interior === nothing && return nothing
    fixed === nothing && return with_ρ_max(case.interior, factor / dt)
    fixed * dt ≤ 1 || throw(ArgumentError(
        "the fixed relaxation rate ρ_max = $fixed is above this chunk's grid " *
        "rate 1/dt = $(1 / dt): a fixed rate is the physical alternative to " *
        "CODE.md's ρ_max · dt = 1, and one above it relaxes harder than the " *
        "paste the default already is, up to RK4's real-axis limit of " *
        "2.79/dt — which a run then finds as a blow-up in the layer. Lower " *
        "ρ_max_fixed, or pass ρ_max_factor for a grid rate."))
    return with_ρ_max(case.interior, fixed)
end

# What the record reports as this chunk's relaxation rate.
interior_ρ_max(::Nothing) = 0
interior_ρ_max(int::Interior) = int.ρ_max

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
                                     q::Integer, schedule) where {T}
    boundary = dirichlet(case, t)
    if boundary === nothing
        fill_ghosts!(U, schedule)
    else
        fill_ghosts!(U, schedule; boundary=boundary)
    end
    origins = to_backend(get_backend(U.work), block_origins(U.forest, T))
    spacings = to_backend(get_backend(U.work), block_spacings(U.forest, T))
    map_blocks!(discrete_momentum_kernel!, U, U.work, origins, spacings,
                case.background, case.interior, T(t), Val(U.G), Val(Int(q)))
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
