# Working notes for Claude in TreeGeneralizedHarmonic.jl

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. The runs behind the black hole's interior — step 5's
layer, steps 8a–8′, the single holes on the octant — are in
`SINGULARITY_HANDLING.md` (moved out of `CODE.md` on 2026-10-08), and the
settings they recommend for a single hole are `CODE.md`'s "Single black
holes: recommended settings". This file is only about mechanics.

**Contents**

- [What this package is](#what-this-package-is)
- [Current state](#current-state)
- [Commands](#commands)
- [Things that will bite](#things-that-will-bite)
- [Conventions](#conventions)
- [Repository facts](#repository-facts)

## What this package is

The third downstream application of
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl), and the first step
toward a production code: the vacuum Einstein equations in the
generalized harmonic formulation, second order in space and first order
in time, on TreeAMR's octree of uniform blocks. What `CODE.md`
specifies is a **proof of concept** — a single boosted, spinning black
hole with its interior driven to the analytic solution — and its
milestones stop there. TreeAMR is the mesh and contains no physics;
[TreeWave](https://github.com/eschnett/TreeWave.jl) is the scalar wave
equation and [TreeHydro](../TreeHydro) is Newtonian hydrodynamics; this
package is general relativity. The formulation and the pointwise
algebra come from `GeneralizedHarmonicSecondOrder2` ("GHSO2"), where
they were validated on SBP-SAT spectral elements. That repository is
unpublished, so the documents this package rests on are copied verbatim
into `notes/` — read `notes/methods-ghso2.md` for the physics and the
numbers this package compares itself against, and cite `notes/`, not
the sibling checkout.

Three rules follow from `CODE.md` and govern every change here:

- **No mesh machinery.** If a change is about trees, ghost cells,
  interpolation or reductions, it belongs upstream in TreeAMR. The one
  stopgap this package carried — point interpolation for the horizon
  finder — went upstream on 2026-09-26 (TreeAMR 0.1.3's M11,
  `interpolate` with `Lagrange(q + 2)`); what `src/horizon.jl` keeps is the
  order, the footprint guard as a TreeAMR `Region` and the refusal
  (amended 2026-09-26).
- **The interior is pointwise and generic, and there is no excision.**
  Inside the horizon the right-hand side is modified by smooth profiles
  of the *depth* below a surface `m` cells inside the horizon — the
  analytic center's sphere for step 5's layer, the tracked apparent
  horizon's offset surface for the generic one (PLAN.md steps 8a–8g,
  added 2026-09-23) — relaxing toward a target that is the analytic
  solution (`:damped`) or a regular fit of the evolved state
  (`:fitted`), with a range projection deep inside as the insurance
  that reports where either fails. Those profiles depend on position,
  time and the tracked horizon, and on nothing about blocks, levels or
  ghost widths. Do not add an excision mask or a one-sided stencil:
  excision is `CODE.md`'s fallback, priced there and decided by step
  8g's host-side test, not built.
- **Inherit GHSO2's algebra, do not re-derive it.** `pointwise.jl` is a
  port of `notes/pointwise-ghso2.jl`; a change to the equations there is
  a change to a validated result and needs the corresponding test
  against `SpacetimeMetrics` automatic differentiation to say so.

## Current state

**G0–G4 are done — there is a black hole on a mesh the code chose, and the
code knows where its horizon is: the interior damping layer, the driver
with its per-chunk analysis record and its regrid branch, the masked
Löhner indicator with its interior mask, its derived level floor and its
boundary ceiling, the masked error converging at order `q` on a frozen
hierarchy, and the apparent horizon with its area, `M_irr`, Korzyński `J`
and `M_ch` at Kerr's values in both charts and at `a = 9/10`. Before the
moving hole (G5) come PLAN.md's steps 8a–8g (added 2026-09-23), the
generic interior: step 5's layer needs an analytic center and an analytic
interior, and its spherical core cannot hold harmonic Kerr's singular
disk at `a = 9/10`, which is G5's case. Steps 8a (the leakage margin),
8b (the range projection), 8c (the calibration of the layer for an
inexact target), 8c′ (`ρ_max = 4/M` the default, decided 2026-09-23) and
8d (the tracked horizon geometry), 8e (the fitted target and the
`:fitted` variant, with the boost-sign fix) and 8f (the measurement matrix)
are done, and the generic interior is marked *(Done.)* under G5: `:fitted`
reaches `50 M` on the static Kerr-Schild hole at twice `:damped`'s error, so
step 8g (excision) is not needed; the analytic layer stays the default where
a chart admits it and `:fitted` is for G5's chart. **G5 runs at `a = 7/10`
(decided 2026-09-23)**, which runs at `h = 5/256` (2472 blocks, `10 M` in
1¾ node-hours) and not at `5/128`; harmonic `a = 9/10` waits with its price
written down in `SINGULARITY_HANDLING.md`'s "The interior's questions"
(`h ≲ 5/1024` on the equator,
23 000 blocks, `38 h` a `50 M` run, and a fit that holds 45° first). Step 8,
the moving hole, has run (2026-09-24) and G5 is **not** done: the mesh
follows the hole (the `:fitted` cycle on the case's own data, the floor from
the core surface widened by the travel), the Dirichlet data are exact, the
frozen hierarchy converges at order 2, the adaptive run is the uniform one
at the hole with 6–9× fewer points, the horizon is found along the
trajectory with Kerr's numbers and the boost's contraction, `:frozen` fails
— but on G5's chart the moving layer's error is `3.3×` the static hole's
at `7 M` and growing, carried out through the layer's trailing side
(`CODE.md`'s G5 and `SINGULARITY_HANDLING.md`'s "The moving hole (step 8)").
Step 8′ (2026-09-25) found
the lever for the trailing side — `evolve!(…; trail_ramp = 9/10)`, `ρ`'s ramp
narrowed where grid points leave the layer — which takes the crossing's
excess over the resting hole from `5.0×` to `2.05×` at `13 M` and makes the
two sides of the layer equal; G5 is still open on the uniform growth that is
left and on a drift of the horizon's `J` (`0.846` at `13 M`) that the moving
spinning hole shows with and without it (`SINGULARITY_HANDLING.md`, "The
trailing side
(step 8′)" and "The interior's questions").**
`CODE.md` is complete and reviewed three times (2026-09-16): the expanded
form of the momentum equation, three dimensions only, a pointwise damping
layer instead of excision, a single boosted spinning black hole as the
proof-of-concept target, RK4 from OrdinaryDiffEq (IMEXRungeKutta's RK4 by
block owner since 2026-09-26), the analysis quantities
as part of the deliverable, an error indicator for refinement, `Float64`
on Symmetry's H200 as the device requirement, no checkpointing, GPU
kernel efficiency deferred to a research project. `PLAN.md` breaks the
milestones G0–G6 into steps 0–10, each a brief for one agent with a fresh
context (see its "Running a step as an agent"); step 8 (a hole that
moves) is next. `notes/` holds the inherited documents.

What exists in `src/` is the module shell, `precision.jl` (the `Base`
bridges for software floating-point types), `device.jl` (`to_backend`,
`hostcopy`, `hostcopy!`), `pointwise.jl` — GHSO2's node-local algebra
ported from `notes/pointwise-ghso2.jl`, plus the expanded momentum
equation this package discretises (`metric_derivatives`,
`gh_node_rhs_expanded`) and `gh_node_source` — `stencils.jl` — the
centered finite-difference and Kreiss–Oliger weights at order `q`, built
in `Rational` and rounded once into `T` by a `@generated` method, plus the
host-side `apply_stencil` the tests measure them with — and, from step 3,
the mesh-side physics: `evolution.jl` (the fused right-hand-side kernel in
`CODE.md`'s streaming order, `GHProblem`, `gh_rhs!`, the speed kernel and
`gh_dt`, `convergence_rate`), `initialdata.jl` (`GHCase`, the three
cases, the uniform forest builder, and the one conversion of
`SpacetimeMetrics`' derivative index order), `gauge.jl` (`isharmonic`,
`isstatic`, the sampled `Hsrc` and the refusal of a moving non-harmonic
background) and `boundaries.jl` (the time-dependent Dirichlet hook) —
and, from step 4, `constraints.jl`: the gauge-constraint kernel, the ADM
one (every second derivative of `g_ab`, with the `∂_t` blocks
reconstructed from the evolution equations and the four-dimensional Ricci
tensor assembled rather than reduced), the `AllPoints` mask and the
`is_evolved` predicate step 5 extends, `masked_norms` and
`constraint_norms`. Step 4 also added `metric_derivatives_along` to
`pointwise.jl` (the same chain rule along **one** direction, which is how
the monitors reach `∂_t α`, `∂_t β^j` and `∂_t √γ`) and a
`refined = true` forest to `initialdata.jl`.

From step 5 there is a **black hole**: `interior.jl` (the `C²` profiles
`w` and `ρ`, the core rule, `HoleCenter`, the horizon's analytic
coordinate radii, the `InteriorMask` and `ShellMask` every norm takes,
and `check_interior_radii` — the two placement bounds, asserted wherever
a problem is built and therefore after every regrid) and `driver.jl`
(`evolve!`, the chunked loop with the CFL recheck and the per-chunk
analysis record, `observer`, and GHSO2's discrete-gradient `Π`
post-pass). `evolution.jl` grew the `(INTERIOR)` term, a fifth `Val`
carrying the interior *variant*, and the `:pasted` variant's
`step_limiter!`; `constraints.jl` grew the error kernel and
`error_norms`; `gauge.jl` grew the position-dependent `γ0` profile;
`initialdata.jl` grew `hole_forest` and the two hole cases.

From step 6 there is a **mesh the code chose**: `refinement.jl` (the
`Refinement` parameters a case carries, `lohner` with the global
amplitude, the masked `field_scales`, the `τ` kernel into `diag`'s
fourteenth slot, `tau_max`, the four marks with the box keyed on
`coarsen_tol`, the derived `horizon_floor_level`, the `LevelBounds` the
floor and the ceiling are applied through, the travelling margin,
`refinement_centroid`, and `indicator_flags`/`gh_indicator!`, the two
entry points the cycle and the driver call). `driver.jl` grew `adapt` (the
initial-data cycle) and `regrid` (the branch at every chunk boundary but
the last), and the record grew `τ_max`, the centroid and its distance from
the analytic center.

From step 7 the code **knows where the horizon is**: `horizon.jl` —
`gh_interpolate`/`gh_interpolate_grad` (TreeAMR's `interpolate` with a
`Lagrange(q + 2)` basis, one launch on the field set's backend, unpacked
into `SVector`s, with the footprint guard `UnevolvedRegion` — a TreeAMR
`Region` — that makes a query reading inside `r_1` throw; until
2026-09-26 these were the *stopgap* `locate_block`/`interpolate`/
`interpolate_grad`, whose exported `interpolate` collided with TreeAMR's),
`GHADMProvider` (the batched `ADMVars` provider, `Float64` out whatever
the run computes in, with a one-entry cache because `KorzynskiSpin` asks
for `γ` and `K` in two calls with the same points), `find_gh_horizon`
(the fast flow, the proper area, `M_irr`, the Korzyński `J` with its
axis, `M_ch`) and `Horizon`, the cadence and resolution the case carries.
`initialdata.jl`'s `GHCase` grew a sixth type parameter for it and
`with_horizon`; `driver.jl`'s record grew the horizon rows, seeded from
the previous find. There is no I/O and no CLI: those are step 9.

From step 8b there is an **instrument**: `bounds.jl` — `StateBounds` (the
ranges of `α`, `γ`'s spectrum, `|β|` and `Π`'s scale, and the gate radius;
`default_bounds` and `default_gate` are the named proposals, and a
`GHCase` carries one as `bounds`, default `nothing`), the pointwise
`bounds_project` in ADM variables, `gh_bounds_kernel!` behind
`gh_stage_limiter!` — the state's third writer, the integrator's stage
limiter beside the paste — `BoundsAccounting`, the validity monitor
(`validity_rows`) and `evolved_nonfinite`, which is what `max_speed_of`
and the record's `finite` now read. `diag` has 22 slots (17 from
2026-10-23, when the monitor became a fold); the record grew
`bounds_hits`, `bounds_nonfinite`, `bounds_r_max` and the eight
`min_detγ_layer` … `max_Π_shell` rows. `test/bounds_tests.jl` is its
file, and `hole_runs.jl`'s `bounds` section its long runs.

From step 8c the layer can be **calibrated against a wrong target**: an
`Interior` carries an optional `target` metric (`isbits`, default `nothing`
= the case's background, resolved by `layer_target`) that the kernel reads
where `(INTERIOR)` reads `u_exact` — the layer branch and the `:pasted`
paste — and nowhere else, so the record's `residual` is the layer's
distance from the *truth*; `evolve!` takes `ρ_max_fixed`, a rate in the
case's units, instead of `ρ_max_factor/dt` (the two are refused together,
and a fixed rate above `1/dt` is refused); and `gauge.jl` has
`HorizonDissipation`, the `ε_KO(r)` profile (`ε_out` at and outside
`r_h,min`, rising `C²` to `ε_in ≤ 4` at `r_1`), which `GHCase` takes where
it took a number — `ε_KO` is a type parameter now, `dissipation_rate` the
identity on a number and `has_dissipation` the kernel's `Val{DISS}`;
`with_dissipation` and `horizon_dissipation` build it from a case. The
measured rule is in `CODE.md`, "The interior": **`ρ_max = 4/M` and a
ramp of at least `4G = 8` cells, `ε_KO` constant** — a grid rate on an
inexact target ends the run in 2–18 M, and on the exact one costs a factor
six in the 50 M error. `test/evolution_cases.jl` has the E3 target
(`CurvatureTarget`) and a `gh_outside_shell_norms(p, u, t)` method, and
`hole_runs.jl` a `calibration` section.

From step 8c′ **`ρ_max = 4/M` is the default** (Erik's decision of
2026-09-23, for every variant, the analytic `:damped` layer included):
`interior.jl` has `hole_mass(background)` beside the horizon radii (`.mass`
through `translate`, `rotate` and `boost`, an `ArgumentError` for a
background without one), `driver.jl` has `default_relaxation_rate(case)`,
the one place the `4` is written, and `evolve!` with no rate keyword relaxes
at it in every chunk. The grid rate `ρ_max · dt = 1` is the option
`ρ_max_factor = 1` — it is what reproduces step 5's tables, which `CODE.md`
keeps as history — and `hole_runs.jl`'s `calibration` (`:grid`) and `bounds`
sections ask for it by name.

From step 8d the layer can **follow the horizon that was found**: a case
whose `interior` is a `FittedSpec` (the rule: `margin`, `n_L` — `0` for step
8c's `max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)`, resolved by `evolve!` — `core_min`,
`lmax_shape`, the ramps, `max_misses`, `α_trigger`, a `target`) has its
layer rebuilt every chunk as a `FittedInterior` (`interior.jl`: the `isbits`
kernel argument, the same variant `Val`s as `Interior`) from a
`HorizonTrack` (`tracking.jl`: `seed_track` from the analytic answer,
`update_track` from each find — coasting on a miss, a `TrackLostError`
after `max_misses`, an `ArgumentError` for a jump of `r_min` over `G h/2` —
and `track_center`, a `HoleCenter`). The layer is keyed on the **depth**
`d = r_h(n̂) − m h − |x − c(t)|` below the tracked shape's offset surface, the
shape being real spherical-harmonic coefficients in `ash_mode_index`'s
slots (`real_harmonic_index`; `m < 0` is the sine), evaluated by
`shape_series` with no angle. A `FittedInterior` holding step 5's sphere is
step 5's layer **bit for bit**. `interior.jl` also has `ShapeMask`,
`ShapeBand`, `geometry_radii`, `layer_radii`, `layer_mask`, `shell_mask`
and `in_layer(int, t, x)`; `horizon.jl`'s `find_gh_horizon` takes
`center =` and returns the radii about its own origin; `bounds.jl`'s monitor
reduces over the evolved region too (`min_α_evolved`, the lapse-collapse
trigger's input); `tracking.jl` has `margin_efolds`, step 8a's leakage moved
in from `test/dispersion.jl`; the record has the `track_*` and `layer_*`
rows and `margin_efolds`; `evolve!` takes `find`. `AbstractSphericalHarmonics`
is a direct dependency. `test/tracking_tests.jl` is its file.

From step 8e-i there is a **fitted target** on the host: `fit.jl` —
`fit_variables` (the packed state to `(log α, β^i, γ_ij − δ_ij, Π_ab)` and
the radial derivatives by the chain rule) and `state_from_fit` (back,
inverse-free), the real **solid** harmonics (`_solid_harmonic_fold`,
`shape_series`'s recurrence with `ρ²`), `state_sampler` and
`analytic_sampler` (called as `sampler(xs, ns)`), `solve_fit` (one QR for
twenty right-hand sides, rows weighted by `fit_row_weights`), `build_fit`
with its validity sweep `fit_sweep`, `FitParams` (`isbits`) and
`InteriorFit`, and the kernel-callable `fit_variables_at`/`fit_state`.
`FittedSpec` has `lmax_fit = 8`. `interior.jl` has `hole_velocity` beside
`hole_mass`, and `GHCase` derives its velocity from it. `test/fit_tests.jl`
is its file.

From step 8e-ii there is a **`:fitted` variant**: `INTERIOR_VARIANTS` has
`:fitted` (a `FittedInterior`'s only), `FittedSpec` has `target_bounds`
(derived by `derive_target_bounds` when `nothing`), `fit.jl` has the
40-variable target cache (`target_cache`, `fit_target_kernel!`,
`fill_target!`) and `fitted_state_kernel!` for the initial data,
`evolution.jl`'s `GHProblem` carries `target`, `fits`, `t_target` and has
`refill_target`, the kernel's `:fitted` core and layer read the cache,
`constraints.jl`'s error kernel measures the `:fitted` residual against it,
and `driver.jl` builds the analytic `cont = 1` fit for the initial data,
`refit!`s the state at every row, refills at every chunk start and splits a
moving hole's chunk into pieces of `h/(4|v|)`. `evolve!` has the study knobs
`fit_initial_cont` and `fit_initial_depth`. `hole_runs.jl` has a `fitted`
section.

From step 8f the fit holds **`Π̃ = (α/√γ)Π`** by default
(`fit_variables(…; tilde)`, `state_from_fit(v, tilde)`, `FitParams.tilde`,
`FittedSpec`'s `fit_tilde = true`); `interior.jl` has `with_variant` for a
spec and a geometry; `evolve!` has `handover` (a `:fitted` case on the
analytic `:damped` layer of the same geometry until then) and `target_source
= :snapshot` (the cache filled with the state, `fill_snapshot!`), and
`adapt = true` works for a `:fitted` case — the mesh chosen on the analytic
data of the same geometry, refused by name where the analytic core meets the
chart's singular set; the record has a `variant` row; the drift reads the
evolved points of its band only. `hole_runs.jl` has the `generic` section,
the matrix. `driver.jl` sizes a moving hole's step from `λ` times the
square of the previous chunk's growth `λ_end/λ` (a static hole's step is
unchanged bit for bit): the fastest speed of a hole crossing the box grows
by 0.1–0.3 % a chunk, and the CFL recheck stopped every boosted row at
`cfl = 1/4` and at `1/5` without it.

From step 8 the hole **moves across the mesh**: `driver.jl` has
`adapt_fitted_initial_data!` and `fill_fitted_initial!` — a `:fitted` case's
initial-data cycle flags on its own data (analytic outside the offset
surface, the fit inside), rebuilt with the geometry on every pass, so G5's
chart has a cycle (step 8f's flagged on the analytic layer and refused it);
`evolve!` has `target_rate` (default on: the cache's slope carries the fit's
translation with the track and the kernel adds `(1 − w) ∂_t u_fit` in the
layer and the core; `GHProblem.target_rate`, `fill_target!(…; rate)`) and
`fit_initial_blend` (the initial data blended `C²` in the fit's variables
from the analytic solution at the offset surface to the fit at
`fit_initial_depth`, G5's rows' choice); the level floor of a tracked
geometry starts at its **core** surface and is widened by the travel
`|v| · chunk` (`level_bounds`, `indicator_flags`, `gh_indicator!` take
`travel`); a moving seed's `r_min` is its analytic surface's least radius
(not `horizon_min_radius`'s `√(1 − v²)` bound); a moving hole's step keeps
a 1 % margin. `test/moving_tests.jl` is its file and `hole_runs.jl`'s
`moving` section its runs.

From 2026-09-26 the **integrator is IMEXRungeKutta's RK4 by block owner**:
`stepping.jl` — `state_partition` (TreeAMR's `threadchunks` as the state
vector's partition, `nothing` on a device), `gh_limiter!` (the
projection, then the paste, passed as both the stage and the step limiter), `ProblemRef`, `gh_integrator` and `gh_solve` —
and `evolve!` builds one integrator per chunk, stepping its own state in
place. OrdinaryDiffEq and SciMLBase are gone from both environments. The
same change removed a `Core.Box` from `gh_rhs_kernel!` (see "Things that
will bite"), which made the right-hand side 3–4× faster at 64 threads and
let it compile for a **device for the first time**: `evolve!` runs end to
end on Metal at `Float32` and on an H200 at both precisions, the `Float64`
record equal to the CPU's. `bench/stepping.jl` measures a step by
integrator and backend; `CODE.md`, "Time integration", has its numbers.
There are still no device *tests* (step 9).

From 2026-10-01 the package is on **TreeAMR 0.1.4**, and two of its
features are in use. **Every reduction that crosses blocks is
`mesh_mapreduce`** — the masked norms (whose denominator is
`evolved_volume`), the speed, the indicator's scales and `τ_max`, the
projection's counts and the validity monitor's extremes — rather than a host
fold of this package's own over `block_mapreduce`'s per-block vector; the L2
norms moved in the last place. And **a run checkpoints and restarts**
(Erik's decision of 2026-10-01, reversing `CODE.md`'s "no checkpointing"):
`checkpoint.jl` — TreeHydro's file names, rotation, `latest_checkpoint`,
exact reals and recipe refusal, plus `to_plain`/`from_plain` for the run
state's own structs, `fit_from_plain`, `run_criterion` and `save_run`/
`load_run` over TreeAMR's M9a — and `evolve!`'s keywords
`checkpoint_path_prefix`, `checkpoint_every_chunks`,
`checkpoint_interval_seconds`, `max_walltime_seconds`,
`num_checkpoints_keep`, `checkpoint_hdf5_filters`,
`checkpoint_sync_to_disk` and `restart_file`, TreeHydro's names. Two
differences from TreeHydro, both Erik's: HDF5 (TreeIOHDF5 from 2026-10-09)
is a **hard dependency**, and
the checkpoint is written **before the regrid**, so that a restart may
change the regridding criterion (the case's `Refinement`, `regrid`,
`buffer`) and regrids with it first; everything else must match the
recipe. `evolve!`'s regrid is a local `regrid_mesh` that the loop and a
restart share. `test/checkpoint_tests.jl` is its file, and `hole_runs.jl`'s
`generic` and `moving` workers take `checkpoint=<dir>` and `walltime=<s>`.

From 2026-10-02/03 a single hole runs **on an octant**, and the static
Kerr-Schild hole has a gauge source that needs no sampling. **Reflecting
faces** (TreeAMR's M10): `GHCase(; reflecting)`, `state_parity`/`even_parity`
at every `FieldSet` site, `has_outer_face`, `minkowski_octant_case`,
`hole_case(; octant = true)` (a hole at the origin, at rest, `a = 0` only),
`hole_forest(; shape = :cube)` for nested boxes. **The algebraic Kerr-Schild
source** (`CODE.md`, "Gauge and constraint damping"): `KerrSchildSource(T; M,
spin, velocity)` behind `Val(:algebraic)`, evaluated in the kernels from `h`,
checked against the background at construction (`check_gauge_source`), which
lifts the refusal of a moving Kerr-Schild hole; `hole_case(; gauge_source =
:algebraic)`. **Run tools**: `add_noise!` (with `exclude` for a frozen core),
`evolve!(; perturb)`, point-weighted and per-level norms (`weighting =
:points`, `at_level`, `level_constraint_norms`), an observer that takes a
fourth argument gets the chunk's record row, and `src/simwatch.jl` writes
SimWatch status files (`SimWatchWriter`; https://github.com/eschnett/simwatch;
from 2026-10-19 with a `summary` and the `[history]` window SimWatch draws as
sparklines, `simwatch_history!` refilling it after a restart; from 2026-10-21
at SimWatch 1.3.0, with `group`, `fraction` and a chain's `previous_job_ids`).
**`FittedSpec(; fit_cont = 2)`** fits curvatures too (the state sampler's
`order = 2` differences the interpolated gradient: TreeAMR 0.1.4 has no
second-derivative interpolation; TreeAMR's `main` has it, unreleased).
`test/reflection_tests.jl`, `gauge_source_tests.jl` and `simwatch_tests.jl`
are its files; `test/octant_runs.jl` is the run script (`octant_rates.jl`,
`octant_study.jl`, `octant_diff.jl` analyse its output), and
`SINGULARITY_HANDLING.md`'s
"Single holes on the octant" holds the numbers — among them the `:fitted`
setup at `a = 0`: `h = 1/24` at the hole, `m = 16`, `n_L = 20`,
`fit_cont = 2`; `CODE.md`'s "Single black holes: recommended settings" has
the table for every spin to `9/10`.

From 2026-10-04 a **spinning** hole has an octant too, on **TreeAMR 0.1.7**'s
rotating seam (M12): `GHCase(; rotating = (d1, d2))` (stored as `(0, 0)` for
none, so the case stays `isbits`), `gh_forest` passing it, `state_rotation`
for the state and `identity_rotation` for every `G = 0` set at every
`FieldSet` site (beside the parities; `hostcopy` keeps it), `GHProblem`
refusing a forest whose seam is not the case's, the ceiling skipping the
seam's faces, `add_noise!` leaving both seam planes alone, and
`hole_case(; octant = :rotating)` — the seam `(1, 2)` with the mirror at
`z = 0`, for a hole at the origin, at rest, any spin along `z`
(`octant = true` is `:reflecting`). The horizon finder works through the
seam unchanged. `test/rotation_tests.jl` is its file (29 claims, `9.4 s` at
four threads inside the suite, which then measured **4824 assertions in
13m21**); `CODE.md`, "The rotating octant", has the numbers.

From 2026-10-05 the right-hand side is **fast on a device**. `CODE.md`, "The
right-hand side on an H200", found the kernel uninlined and spilling at 8.5 ns a
point on an H200, and it now runs at 1.1, and 1.7× faster on the CPU. What changed:

- **`pointwise.jl`** has `gh_node_source_lean`: the source held by unique
  components, phase-ordered and unrolled by `@ntuple`, a third spelling tested
  against `gh_node_source`. Beside it, `metric_divergences` forms the two
  contractions of `metric_derivatives` the kernel reads.
- **`evolution.jl`** splits the body:
  - `gh_rhs_head` — the state, `∂h`, the coefficients, `∂ₜh`, the source and the
    divergences;
  - `gh_rhs_pi` — one Π component.

  The kernel without an interior stores as it goes (`gh_rhs_store!`); the
  interior variants collect through `gh_rhs_at_point`. `axis_stencil` and
  `mixed_stencil` are `@generated`.
- **The investigation's prototypes** are `bench/rhs_lab.jl` and its companions
  (see "Commands").

From 2026-10-05 the right-hand side runs on **SIMD lanes on the CPU** (`CODE.md`,
"The right-hand side on a CPU"): the kernel evaluates `W` neighbouring points along
the first axis at once as SIMD.jl's `Vec{W,T}` (SIMD.jl is a dependency), through the
package's own algebra, on every branch — no hole, a hole's evolved region (a group
with a point in the layer or the core falls back to the scalar code point by point),
the sampled and the algebraic gauge sources. `W` is the sixth kernel `Val`, chosen by
`GHProblem` (`default_simd_width`: 4 `Float64` on AVX2 and aarch64, 8 with AVX-512,
1 on a device or for a software type; `simd_width = 1` asks for the scalar kernel,
also through `evolve!`). A row that `W` does not divide ends in an **overlapping**
group. `src/lanes.jl` has the width, `Lanes` (an array read `W` elements at a time)
and the leader rule; the kernel's body is `gh_rhs_point!` (one point) and
`gh_rhs_lanes!` (a group). `pointwise.jl`'s algebra and the algebraic source asked
three things of their number type that a `Vec` lacks, now spelled `_scale`,
`_anynonzero` and `_select`. `test/simd_tests.jl` is its file, and
`bench/rhs_cpu_lab.jl` the lab that measured it. Measured: the kernel 2.2–2.4× on
Zen 3 (four lanes) and 2.2–2.8× on Skylake-AVX512 (eight), the gauge wave's RK4 step
1.42× at 64 threads; the hole fixture's step only 1.08×, because on a refined mesh
TreeAMR's prolongation is 145 of a 200 ms `gh_rhs!` — that, not the kernel, is a
hole's next lever. Devices run the `W = 1` kernel at PR #4's speed.

From 2026-10-09 the package is on **TreeAMR 0.2.0**, whose one break is that
the checkpoint functions and TreeAMR's HDF5 extension `TreeAMRHDF5Ext` moved
into the companion package **TreeIOHDF5** (unregistered, pinned to GitHub
`main` by `[sources]`, `[compat]` `0.1.0`), with the same names, signatures
and file format. TreeIOHDF5 replaces HDF5 as the hard dependency (`src/` used
HDF5 for nothing else), and `prerequisite_tests.jl` asserts that
`save_checkpoint` and `load_checkpoint` are TreeIOHDF5's and not TreeAMR's.
No internal this package reaches for changed.

From 2026-10-19 a run is **distributed over MPI**, and so runs on several
GPUs, one per rank (TreeAMR's M7, released since 0.1.5, which already did
the exchange, the regrid, the reductions, the interpolation and, in
TreeIOHDF5, the checkpoints). `gh_forest`/`hole_forest` take `comm`,
`evolve!` takes it for a restart (and refuses a forest over another one),
`load_run` passes it, and `src/distributed.jl` has TreeHydro's agreements
(`rank_reduce`, `agree_any`, `agree_refusal`, `isroot`) plus `rank_bcast`
and `on_root`: **the find, the track and the fits run on rank 0 alone**, the
other ranks serving its ghost fills (`fill_state!`) and interpolations
(`interpolate_state`) by broadcast request, and every rank adopts rank 0's
result. Fixed on the way: the travelling margin's flags are gathered as
`RegridMark`s before `buffered_flags`, the refinement centroid read leaves by
a local index, `add_noise!` draws from a `Xoshiro` per block (its values
changed), checkpoint rotation removes part files on rank 0, the wall-clock
decisions and the fits' timings are agreed, and `SimWatchWriter(; comm)` is
collective. `evolve!` grew `checkpoint_io`. MPI is a test dependency only;
`test/mpi_workload.jl` under `mpiexec -n 3` is the claim (`CODE.md`,
"Running distributed"), and `test/octant_runs.jl mpi=true` the multi-GPU run.

From 2026-10-20 a hole **moves on a quadrant**: a boosted Kerr-Schild hole
at `a = 0` and `v = 1/5`, the speed of a binary member a few orbits before
merger, the first boosted Kerr-Schild hole the package has evolved.
`hole_case(; octant = :quadrant)` is `[−L, L] × [0, L]²` with mirrors at
`y = 0` and `z = 0`, for a hole on the `x` axis moving along it without spin;
`gh_forest` takes a root brick `(2r, r, r)`; `hole_forest(; path)` sweeps the
shells along the hole's path (the frozen capsule); `kerr_schild_case(; v, x0)`
and `gauge_source = :algebraic` build the moving hole with its source; and a
moving `:damped` core relaxes toward its target in ADM variables
(`core_follows`, `core_relaxation_rate`), where it had `du = 0` and died at
`r_0/v`. `test/boosted_tests.jl` is its file. On one H200 a run to `40 M`
took `0.4`, `1.4` and `2.7 h` at `h = 1/16`, `1/24`, `1/32`: with `ρ_max =
8/M` the moving hole is at the resting hole's level and converges at order
four, the default `4/M` lags the moving target, and `:fitted` holds it at
`4/M`'s level and not `8/M`'s (`SINGULARITY_HANDLING.md`, "The boosted
Kerr-Schild hole at binary speed"; the settings in `CODE.md`'s table). This
says nothing about G5's chart, which is still open. From 2026-10-21 the moving
`:damped` layer adds its target's rate, `(1 − w) ∂_t u_exact`
(`background_state_rate`, a centered difference in time for `∂_t Π`), and its
core's outer half its own (`core_target_rate`): at the default `4/M` the moving
hole is then at the resting hole's level from `1/24` on and converges at order
four, better than `8/M` without the rates (`SINGULARITY_HANDLING.md`, "The
moving `:damped` layer's target rate").

From 2026-10-21 a **spinning hole moves along its spin axis** on the rotating
quadrant: `hole_case(; octant = :rotating_quadrant)` is `[0, L]² × [−L, L]`
with the seam `(1, 2)` and no mirror, for a hole on the `z` axis moving along
it and spinning about it; `kerr_schild_case(; axis = 3)` puts the hole on `z`;
`gh_forest` takes the brick `(r, r, 2r)` (`CODE.md`, "The rotating quadrant").
Kerr-Schild `a = 1/2` at `v = 1/5` from `z = −4` to `4`, on one H200 a run to
`40 M` at `h = 1/24`, `1/32`, `1/40` (`1.45`, `2.8`, `6.0 h`): `:damped` at
`4/M` with the target's rate is the resting hole's level in `ℋ` and converges
at order four, and the horizon's `J − a` drifts linearly at the resting
hole's rate and converges at order four — so G5's `J` drift is not the motion
of a spinning hole as such (an elimination, not a test of G5's case).
`:fitted` (`m = 20`, `n_L = 18`, from `1/32`) holds it but does not converge
to `1/40` (`SINGULARITY_HANDLING.md`, "A spinning hole along its axis"; the
moving `a = 1/2` rows of `CODE.md`'s table).

From 2026-10-22 the package requires **TreeAMR 0.2.4** (`78e60ae`), whose
nested-mesh ghost fill took a moving-hole row on one H200 from `254` to
`239 s/M`. A benchmark of the analysis cadence found that most of those
`239 s/M` are the analysis row, not the evolution: the trailing/leading split costs `109 s/M`
at `chunk = M/2`, the rest of the row `33 s` a chunk, the finder under a
second, and the evolution alone `64 s/M`, `20 ns` per point and step.
Production runs now use `chunk = M` and `split=0` (`CODE.md`, "The analysis
cadence on an H200"). The **aligned campaign** then moved Kerr-Schild
`a = 0`, `1/2` and `9/10` along their spin at `v = 0`, `1/5` and `1/2`, to
`20 M`, each at three spacings on one H200 a run: 28 runs, `29.7` GPU-hours.
It forced two fixes to the moving `:damped` core. `281a639` maps the core
target's rate in ADM variables (`a = 0`, `v = 1/2` died in its first chunk
with `α² < 0`), and `031ab30` relaxes the core at `max(ρ_max, 12 |v|/r_0)`
(a residual of `4.7` at every `h` at `v = 1/2`). Every cell that fits one
H200 works and converges at order four outside the horizon. At `a = 9/10`
no single layer serves every speed: `r_1 = 9/8`, which the contracted pole
at `v = 1/2` needs, grows at rest, and `5/4` holds only `v ≤ 1/5`
(`SINGULARITY_HANDLING.md`, "Aligned spin and velocity"; the *along `z`*
rows of `CODE.md`'s table).

From 2026-10-23 the package requires **TreeAMR 0.2.6**, and **every norm is
a fold over points**, its M15 `mesh_fold`, which this package asked for: the
callback sees several `diag` slots of one point, its position (bit for bit
`point_position`'s), level and a block tag, and folds into an `isbits`
accumulator of `Sum`/`Max`/`Min` fields — one sweep and one collective for
any number of results. `constraints.jl` has `diag_norms` (the one fold behind
`masked_norms`, `evolved_volume`, `constraint_norms`, `error_norms`),
`slot_norms`, `record_norms` (constraints and errors together: what
`record!` reads once after its three monitors, which write disjoint slots)
and `level_constraint_norms` as one fold binned by level; `bounds.jl`'s
validity monitor and `evolved_nonfinite` are folds over the state vector
(`validity_and_nonfinite` both at once; their kernels and `diag`'s slots
18–22 are gone, `NDIAG = 17`); the projection's counts and `field_scales`
are one fold each. A record row went from 37 reductions (54 on an ADM chunk,
65 with the indicator) to 2 (4). `test/octant_runs.jl`'s `observe` runs each
monitor once and folds three times — the whole mesh, the levels, the
boundary band, the shells and the half-shells in one accumulator — where it
re-ran the ADM monitor and the error kernel once per shell and half-shell to
select points and made about 280–350 reductions; its CSV is the old one to
the ten digits it writes on the quadrant, the rotating octant and a
`:fitted` hole at `a = 0` (`CODE.md`, "Analysis quantities").

From 2026-10-24 a **spinning hole moves across its spin** on the **bitant**:
`hole_case(; octant = :bitant)` is `[−L, L]² × [0, L]` with the mirror at
`z = 0` alone, for a hole in that plane, spinning about `z` and moving in the
plane (`kerr_schild_case(; a, v, x0)`, `axis = 1` or `2`). `gh_forest` takes
the brick `(2r, 2r, r)`. It is half the box and twice the quadrants' points,
because no rotation maps that hole onto itself (`CODE.md`, "The bitant").
`horizon_min_radius` of a boosted spinning hole is now exact across the spin,
the contracted spheroid's shortest semi-axis (`horizon_ellipsoid`), and it is
unchanged bit for bit where the old bound was exact. `test/aligned_study.jl
campaign=transverse` reads the campaign's data and sets each row against the
aligned row of the same `a`, `v` and `h`. The **transverse campaign**
(2026-10-25, Symmetry `transverse-1`) moved Kerr-Schild `a = 0`, `1/2` and
`9/10` along `x`, spinning about `z`, at `v = 0`, `1/5` and `1/2`, to `20 M`:
23 rows on one H200 each, `29.1` GPU-hours. Every row is stable, and every
cell converges at order four outside the horizon except `a = 1/2` at
`v = 1/2`, not yet in that regime at `1/32`. The exact least radius lets
`r_1 = 5/4` serve `a = 9/10` at every speed. Moving across the spin adds an
excess near the horizon that falls at order `7–8` (at `a = 1/2` `4–8×` the
aligned hole's at `1/24`, `1.6–2.5×` at `1/32`). At `v = 1/2` the layer
residual does not fall, and at `a = 9/10` the band inside the horizon does
not converge. The finder's `J` offset is the resting hole's
(`SINGULARITY_HANDLING.md`, "Spin across the motion"). `1/40` at `a = 1/2`
and `1/96` at `a = 9/10`, `v = 1/2` need several GPUs.

From 2026-10-25 (step B1) there are **more algebraic gauge sources**, all
behind `Val(:algebraic)` as `AlgebraicGaugeSource`s evaluated as
`algebraic_gauge_source(src, x, t, h, ∂ₜh, ∂h)` — the kernels pass the point's
position and the time now: `CompositeKerrSchildSource` of `K` `TrackedHole`s
(each hole's line on a piece, the partition of unity at `p = 4`), the damped
harmonic gauge `DampedHarmonicSource` in SpECTRE's form, and the roll-on
`RolledGaugeSource(from, to; σ_g, t_0)`. Every launch is handed
`effective_source(src, t)`: a roll-on's `from` before `t_0` and its `to` once
it is done, and a steady one-hole composite's `KerrSchildSource`, so those are
their sources' kernels bit for bit. `gauge_consistent_momentum` gives `Π` from
`C_a = 0` (step B8's). `hole_case(; gauge_source = :composite)`;
`test/octant_runs.jl` takes `source=composite` and `gauge=dh`;
`bench/gauge_source_cost.jl` times the kernel per source (`CODE.md`, "What step
B1 built and measured"). Its runs, the damped harmonic gauge rolled on over
`10 M` on the boosted and the resting `a = 0` hole on the quadrant with
`:fitted` to `40 M` (`SINGULARITY_HANDLING.md`, "The damped harmonic gauge
rolled on"): the resting hole settles at the Kerr-Schild level outside `r = 3`
(`1.2–1.6×`, order four); the moving hole falls `0.49 M` behind `c(t)` and is
`6–31×` the Kerr-Schild hole's at `1/32` (`70–200×` at `1/24`); the analytic
`:damped` layer does not survive the gauge moving.

From 2026-10-26 (step B2) a `:damped` layer can take its **target and its
center from the horizon track**: `hole_case(; interior = TrackedSpec(T; r_0,
r_1, …))` runs step 5's sphere about the track, relaxing toward the boosted
Kerr-Schild hole of the background's mass and spin on the track's line,
rebuilt at every piece's midpoint (`tracked_interior`; pieces of `h/(4|v|)` as
for a moving `:fitted` target), its core following that target with the
target's rate — an `Interior` with a `target`, so nothing in the kernel is new.
Its track predicts along the **quadratic** of its last four finds
(`TRACK_FIT_FINDS`, `fit_track_quadratic`), the chord before that, blended `C¹`
into the previous prediction over the next find interval (`track_state`,
`track_position`, `track_center(tr, t)`), refuses a found origin more than
`G h/2` from the prediction, and finds `find_count(a, G, h, chunk)` times a
chunk (`|a| Δt² ≤ G h/4`, capped by `max_interval`) at the pieces' boundaries,
the last at the row, which always finds. A `FittedSpec`'s track keeps step 8d's
line bit for bit (`quadratic = false`). The record has `track_finds`,
`track_prediction_max`, `track_acceleration` and `track_interval`; the spec's
`mass_factor`, `velocity_offset`, `center_offset` and `seed_velocity_offset`
are the sensitivity study's perturbations (`test/octant_runs.jl
interior=tracked`). `test/tracked_target_tests.jl` is its file; the synthetic
orbits are `tracking_tests.jl`'s and the chain `checkpoint_tests.jl`'s
(`CODE.md`, "What step B2 built and measured"; `SINGULARITY_HANDLING.md`, "The
tracked analytic target"). Its runs, the boosted `a = 0` hole on the quadrant at
`1/24` and `1/32` to `40 M` on one H200 a row (`step-B2-tracked`): the tracked
target is the analytic `:damped` layer to `10⁻³` in every shell, at order four
and at its cost (`38` and `88 s/M`); a target off by `1 %` in mass, `0.02` in
velocity or a cell in center holds `ℋ` just outside the horizon at a steady
`7.5×`, `18×` and `16–26×` and leaves `r ≥ 3` as it was once the start's
transient has passed; a track seeded `0.05` too fast recovers in two finds with
no refusal. The numbers favour pairing (α).

From 2026-10-26 (step B3) a case may hold **several holes** — Erik's three
decisions of that day: pairing (α), `GHCase(…; holes)` with the background
demoted to data, and the superposition kept here. `src/holes.jl` has
`HoleSpec` (mass, spin along `z`, center, velocity, the seed's acceleration,
a `TrackedSpec`), `SuperposedMetric` (`η + Σ w_A (g_A − η)`, `isbits`, XCTS's
own attenuation and the companions' `1 − e^{−r_B²/λ_B²}`), `HolesSpec` (the
case's rule for `K ≥ 2`, with `hole_target`, `source_feed` and `momentum`),
`HoleInteriors` (the kernel's tuple, composed by `acting_interior`: the hole of
the smallest layer coordinate runs its own single-hole arithmetic), the
composite masks `HolesMask`/`AnyMask`, `HolesDamping` (three Gaussians) and
`HolesDissipation`. `initialdata.jl` has the `holes` keyword (`case_holes`),
`binary_case`, `head_on_holes`, `holes_forest`, `holes_source`,
`check_holes_source` (the `t = 0` check of several holes: the source's holes
are the specs', each hole's own source exact on its own hole) and
`gauge_consistent_state`. `tracking.jl` has `seed_tracks`, `holes_interior`,
`holes_source_at`, `holes_seed_source_at` and `holes_case_at` — the composite,
the source and the profiles rebuilt from the tracks every piece — and
`driver.jl` the loop's branch (`binary`): the tracks a vector, `reseed_holes!`
(each hole found once on the data; its **horizon offset** corrects every later
find), `holes_track!`, the holes finding together, the per-hole record rows
(`holes`, `separation`, `phase`) and the run state; `evolution.jl` has
`with_case`, `refinement.jl` `UnionLevelBounds`, `horizon.jl` the union guard.
One hole is the case it was: `holes = (spec,)` runs the single-hole path bit
for bit. `test/holes_tests.jl` is its file and `test/binary_runs.jl` its script;
`mpi_workload.jl` has a pair (`CODE.md`, "What step B3 built and measured").
Its runs, the head-on pair from rest at `d = 8 M` on `[−32, 32]³` to `2 M`, one
H200 a row (`step-B3-holes`): `1/48` at the holes (`1/24` in `M_A`, the validated
spacing; `63.6 M` points, `104 GB`, `362 s/M`; `1/24` is `7.9 M`, `29 GB`, `61
s/M`, and `1/64` does not fit one H200). **The source fed by the tracks — (α) as
decided — ended every row within `1 M`** on the track's own center or radius
refusals (a feedback: the source carries the holes where the tracks say); **fed by
the holes' Newtonian seeds (`source_feed = :seed`) both momenta ran to `2 M`**,
every find successful, the constraints bounded and, with `momentum = :gauge`,
falling (`C_a` from `1.6·10⁻⁸`, the areas `+0.7 %`; the superposition's `Π` grows
them `18 %`). **The found separation's fall is a gauge transient that decelerates**
(corrected in the review): its drop per `M/4` is `0.0068, 0.0098, 0.0102, 0.0094,
0.0078, 0.0059, 0.0038, 0.0017` against free fall's `0.0005` to `0.0073`
(`b3sg48`), so by `2 M` it falls at `0.23×` free fall's rate (`0.66×` with the
superposition's `Π`) — the same at `1/24` and `1/48` to six digits, the
continuum solution's coordinate motion under the junk, not the finder's accuracy;
read no ratio over a window as a rate. **The layers' residual grows** from `0.41`
to `4.49` over `2 M`, still rising, `15–18×` step B2's single hole scaled to
`M_A = 1/2` (`Π` carries `1/M`: B2's `0.12–0.15` is at most `0.25–0.30` there);
each track is `0.3–0.8` cells off the seed the source carries its hole on, but
the residual follows neither that nor the velocity mismatch alone (`CODE.md`,
"What step B3 built and measured", and item 7 of "What this asks Erik to
decide").

From 2026-10-26/30 (step B3's review, two rounds) a tracked hole's gauge source
may ride a **controlled source line** — Erik's decision on `CODE.md`'s item 7: a
slow control law for the track-fed source, tested on one moving hole fed by its
own track first (`CODE.md`, "The controlled source line"). `src/tracking.jl` has
`SourceLine` (a hole's line: its center, velocity and the integral of its
distance from the track), `SourceLaw(; integral, model)` and `source_law(case)` —
with `e = c_tr − c`, `ω = 1/τ`, the PD loop `c̈ = a_m + ω² e + 2ω ė` or the PID
loop `c̈ = a_m + 3ω² e + 3ω ė + ω³ i`, `i̇ = e`, `a_m` the Newtonian pull of the
other holes or zero — `seed_source_lines` (on the case's own source, the data's
gauge, the integral at zero), `advance_source_lines` (RK4 in steps of `τ/64`),
`source_line_states`, `lines_source_at` and `fed_case_at` (one hole's one-hole
composite rebuilt every piece). A `TrackedSpec` takes `source_feed = :own`
(default: every earlier run, bit for bit) `| :seed | :tracks | :controlled`,
`control_time` (in the hole's mass; `Inf` is the model alone) and the law,
`control_integral` (default `true`) and `control_model` (default `false`);
`binary_case(; source_feed = :controlled)` reads each hole's spec, which must
agree. The driver carries the lines (`case_at`), the checkpoint stores them, and
the record has each piece's `source_center`, `source_velocity`,
`source_acceleration` (per hole for a pair). **Measured:** on step B2's
production hole at `1/24` to `20 M` every feed — raw, PD at `τ = 10 M` and `4 M`,
PID at `10 M` — is `b2t24` to `0.15 %` in `ℋ`, and from B2's recovery seed every
controlled line reproduces `b2r24`'s `ℋ` just outside the horizon and settles back
onto the hole's motion, `0.0006–0.015` cells off the analytic line at `20 M` (the
loop's neutral mode) — but raises `ℋ` and `C_a` on the coarsest level and in the
boundary band about `10³×` (see "Things that will bite").
On the head-on pair at `1/48` every controlled line runs to `6 M` where the raw
feed died by `1 M`: every find, `C_a` bounded. **The model term is what put the
line off the holes**: the PD loop's steady error is `(c̈_tr − a_m) τ²`, and while
the gauge decelerated the holes (`1–3.5 M`, `c̈_tr` from `−0.007` to `−0.020`
against `a_N = 0.008`) the model added `40–100 %` to the forcing — the line `4.24`
cells off at `τ = 10 M_A`
and `1.86` at `4 M_A` at `6 M`, against `1.29` and `0.60` without it, and `1.10`
for the PID loop without it (the line's equation from the measured track
reproduces every row to `0.05` cells). The tracks' acceleration has not settled
by `6 M`: it crosses `a_N` between `5` and `5.75 M` and is `1.1–1.7 a_N` and still rising
at `6 M` on the controlled rows, so the PD loop's `c̈ τ²` lag behind a steady
acceleration, which the PID loop does not have, is a property of the laws and not
yet a measurement; running PID at `10 M_A` and PD at `4 M_A` to about `12 M`
would decide between them (Erik's call, not run). The residual's rise from `3 M` follows the line's distance (`5.6` at `0.6`
cells, `8.45` at `6.6`) over a floor of about `5.6` that every feed shares.
**Proposed for B4: the PID loop without the model, integral from zero, `τ = 10
M_A`** (now the defaults). The holes' coordinate motion is the gauge's and barely
the line's: in every row they fall faster than free fall for `2 M`, stop near
`2.2–2.4 M` and recede, `6.6–8.6` cells off the Newtonian seed by `6 M`.
`test/source_line_tests.jl` is its file; `test/octant_runs.jl feed=` and
`test/binary_runs.jl feed=controlled` its runs, `test/fed_study.jl` and
`test/binary_study.jl` their tables.

From 2026-10-30 (step I1, branch `claude/step-I1-layer-free-interior`) a hole
may be held **without a layer**: `src/layerfree.jl` has `LayerFreeInterior`
(kernel variant `:none`, or `:anchored` with its anchor; every mask
`AllPoints`; its own radius and gate rules), `SteeredSource` (the algebraic
Kerr-Schild source plus a lapse steering toward a collapsed target), the frozen
correction `ΔH` sampled behind the new gauge-source kind `Val(:corrected)`
(`CorrectedSource`), the pre-collapsed data (`layerfree_momentum`), the
validity rows by region, the hits by kind (`DIAG_BOUNDS_KIND`, `NDIAG = 18`) and
`layerfree_diagnostics!`; `initialdata.jl` has `SmoothedKerrSchild` and
`kerr_schild_case(; smoothing)`. `test/layerfree_tests.jl` is its file,
`test/octant_runs.jl interior=layerfree` its runs and `test/layerfree_study.jl`
their reader; `CODE.md`, "A layer-free interior (proposed in step I1)", is the
design, and `SINGULARITY_HANDLING.md`, "A layer-free interior (step I1)", the
screen's command lines, not yet run.

What exists in `test/` is `precision_tests.jl`, `prerequisite_tests.jl`
(the pinned TreeAMR still exports the names the design calls, a
`SpacetimeMetrics` background compiles and runs as a kernel argument on
`CPU()` filling a field set bit-for-bit as a host loop does, and the
*unexported* metric wrappers `gauge.jl` dispatches on still exist), the
pointwise pair — `pointwise_tests.jl` and `pointwise_identity_tests.jl`
over a shared `pointwise_backgrounds.jl`, which is where the six
backgrounds of `CODE.md`'s table and the analytic data are built —
`stencils_tests.jl`, which evaluates no background at all and costs
seconds, and step 3's five: `gauge_tests.jl`, `initialdata_tests.jl`,
`evolution_tests.jl` (the kernel against `gh_node_rhs_expanded`, and the
properties nothing else can state), `convergence_tests.jl` and
`noise_tests.jl`, over the helper `evolution_cases.jl` — the runs
themselves, which live in `test/` because what they wrap is the
integrator loop and `driver.jl` is step 5's — and step 4's four:
`constraints_tests.jl` (both monitors, against `ddmetric` on analytic
data and against the mesh), `interface_tests.jl` (the interface-order
table on the two-level mesh), `type_tests.jl` (`Float32` end to end,
`Float32x2` for the algebra) and `threading_tests.jl` over the standalone
`thread_workload.jl`; and step 5's two — `interior_tests.jl` (the
profiles, the core rule, the masks, the radius assertions firing, and one
right-hand-side evaluation with the layer on a mesh) and
`driver_tests.jl` (the runs: the record, the order on the frozen
hierarchy, the three variants, the drift, the `Π` post-pass) — over the
hole fixture in `evolution_cases.jl`; and step 6's `refinement_tests.jl`
(the Löhner algebra and its global floor, the four marks on a `τ` field
written by hand, the mask, the level floor and the ceiling, the
initial-data cycle, and two short adaptive runs — one whose mesh does not
move and one whose does), over that file's second hole fixture and the
three new helpers in `evolution_cases.jl`; and step 7's
`horizon_tests.jl` (the interpolator's exactness and its rate, the
footprint guard, Kerr's horizon from a displaced guess on both the step-5
fixture and the mesh the indicator chose, and the record's horizon rows at
the case's cadence); and step 8e's `fit_tests.jl` (the fit's harmonics, its
least squares, the static and spinning holes' fits and their validity, the
`g_ab` control, the evaluator at two precisions, and the fit of the one
tracked run's state, which `evolution_cases.jl`'s `tracked_fixture_run`
shares with `tracking_tests.jl`). **`test/hole_runs.jl` is a standalone script, not
part of the suite**: the `t = 50 M` runs, `q = 4`, the two harmonic
charts, the indicator's calibration and — from step 7 — the horizon
section (Kerr's numbers at `a = 9/10` and in the harmonic chart, on meshes
of a thousand blocks) are minutes rather than seconds, and its numbers are
in `CODE.md` with the command that produced them. Step 8a added a second
script, **`test/dispersion.jl`** — the frozen-coefficient dispersion
analysis of the package's own weights inside a horizon, the penetration
length `ℓ` of grid-scale content, and a one-dimensional model run that
predicts what the 3D runs measure — and a `leakage` section to
`hole_runs.jl`, which is *not* in its default list (it is a batch job).
Step 8c's `calibration` section is not in it either: about 150 runs of
`5 M` and `50 M`, a dozen node-hours on Symmetry.
`Project.toml` carries the `[sources]` pins and CI is in place. **Julia
1.13 is the floor and the only version CI runs** (decided by Erik on
2026-10-16: supporting earlier versions slows development and is not
useful at the moment); the 1.11 cells, dropped on 2026-09-19 for
`pointwise_tests.jl:480`'s 176 bytes, are not owed back. Code coverage
runs on **the macOS cell only** (2026-09-26): instrumented runs compute
*wrong numbers* on GitHub's Linux runners (207 failures on the threaded
Linux cell at Julia 1.13.1, which a removed gate had left instrumented),
while the instrumented macOS cell and the instrumented suite on Symmetry
pass. Probes of 146 GitHub runners found the trigger: `--code-coverage`
together with `--check-bounds=yes` (what `julia-runtest` passes; either
alone is fine) on AMD Zen 4/Zen 5 **with AVX-512 exposed** (EPYC 9V45
always, EPYC 9V74 only on the VMs that expose it), on Julia **1.13.0 and
1.13.1**, deterministically — a miscompilation, not a bad machine. Zen 3,
Intel with AVX-512 and nightly (1.14.0-DEV.3386) pass. Keep coverage off
the Linux cells until a 1.13 release is shown to carry the fix. There is
no `Manifest.toml` (deliberately, and permanently: it is what makes the
clean-checkout check below mean something), no `bin/`, and there is now a
remote — `git@github.com:eschnett/TreeGeneralizedHarmonic.jl.git`.

The suite is **7605 assertions in 36m38** at four threads (EPYC 7532, job
586055) and **7597 in 33m14** at one (EPYC 7543, job 586054) after step I1
(2026-10-30, branch `claude/step-I1-layer-free-interior`; run on Symmetry's
`amdq`, the development machine being full — read the times as a cluster node's,
about twice the development machine's): its 1589 new claims are
`layerfree_tests.jl`'s, `3m04` on its own at four threads — 1507 of them the
smoothed data's lattice points, and most of the time the fixture's two short
runs and the `:anchored` kernel's compilation. It was **6016 assertions in 17m26** at four threads and **6008 in 20m59**
at one after the review's second round (2026-10-30, the PID law): its 24 new claims
are `source_line_tests.jl`'s, `16.3 s` and `20.7 s` inside the suite — the PID
loop against its closed form and the junk, `∂_t H` under the PID law, a PID line
never corrected bit for bit the analytic run, and the restart chain on the PID
loop. It was **5992 assertions in 15m17** at four threads and **5984 in 18m58**
at one after step B3's review (2026-10-26/27, the controlled source line): its 40
new claims are `source_line_tests.jl`'s, `12.9 s` at four threads and `15.7 s` at
one inside the suite (the loop against the filter's closed form, on synthetic
finds, the model's free fall and the source's `∂_t H`, host-side; one moving hole
fed on its seed's line and on a line never corrected against the analytic run bit
for bit, and a chain of a controlled run, on `tracked_target_tests.jl`'s
kernels). It was **5952 assertions in 17m19** at four threads and **5944 in 19m58**
at one after step B3 (2026-10-26, a development machine loaded 4–9 by other
work): its 103 new claims are `holes_tests.jl` (72, `49.0 s` at four threads and
`1m07` at one — the pair's pieces host-side, the composite against the nearer
hole's kernel point by point on a 288-block mesh (`17.6 s`), the one-hole case
against the case it was on two of step B2's runs (`12.9 s`), and the pair's run
and its chain (`16.7 s`)) and the MPI comparison's 31 lines of the pair (294, the
workload `5m29` on its own). It was **5849 assertions in 24m40** at four threads and **5841 in 21m02**
at one after step B2 (2026-10-26, the two run side by side on a development
machine loaded 5–28 by them, two smoke runs and other work — read the times
against that): its 112 new claims are `tracking_tests.jl`'s quadratic track (77,
`1.6 s` at one thread: the Keplerian orbits, the blend, the line kept for a
fitted case, the target a track gives), `tracked_target_tests.jl` (25, `6.5 s`:
the seed's target bit for bit, the tracked target against the analytic layer
on the `q = 2` quadrant, the recovery) and `checkpoint_tests.jl`'s tracked chain
(10, `16.4 s`). It was **5737 assertions in 17m01** at four threads after step B1, rebased
on the bitant (2026-10-26, TreeAMR 0.2.6, on a development machine loaded 7–9
by other work): its 73 new claims are
`gauge_source_tests.jl`'s step-B1 testset (64, `29.5 s` at one thread — the
closed forms against ForwardDiff, `gauge_consistent_momentum`, the flat-space
damped gauge on a mesh, the composite and the roll-on bit for bit on the
moving quadrant) and `simd_tests.jl`'s roll-on kernel row and lane-by-lane
sources (9, `19.9 s`). Before the rebase, on the tree without the bitant, it
was **5690 in 16m11** at four threads and **5682 in 26m47** at one (2026-10-25,
on a development machine loaded 12–18 by the step's own smoke runs and MPI job;
that tree before step B1 measured 5617 in 16m37 and 5609 in 20m04 the same
afternoon). It was **5664 assertions in 14m12** at four threads after the bitant
(2026-10-24, TreeAMR 0.2.6): its 47 new claims are `boosted_tests.jl`'s four
bitant testsets (the domain and its refusals, the exact least radius against
the sampled surface, the source of a hole moving across its spin, and the
bitant against the box, `3.9 s`) and two rewritten radius claims in
`interior_tests.jl` and `moving_tests.jl`. It was **5617 assertions in 14m38** at four threads after the folds
(2026-10-23, TreeAMR 0.2.6): its 13 new claims are `bounds_tests.jl`'s
validity rows and finiteness count against a host loop (8), the per-level
bins against `at_level` in `reflection_tests.jl` (4) and `record_norms`
against its two halves in `constraints_tests.jl` (1); the MPI comparison's
263 pass unchanged. It was **5604 assertions in 14m59** at four threads after the two
moving-core fixes (2026-10-23, `281a639`, `031ab30`, on TreeAMR 0.2.4): its 8
new claims are `boosted_tests.jl`'s, which a moving core staying a metric near
its path and releasing its trailing cells within a jump/200 of their targets
make `29.0 s` inside the suite. It was **5596 assertions in 15m42** at four
threads after the SimWatch 1.3.0 change (2026-10-21, `1921d46`): its 66 new claims are the rotating
quadrant's in `boosted_tests.jl` (`8efc5fe`: the refusals, the spinning moving
hole's source, the quadrant against the box) and SimWatch 1.3.0's writer and
SVG thumbnails in `simwatch_tests.jl`. It was **5530 assertions in 14m00** at
four threads after the moving `:damped` layer's target rate (2026-10-20): its
16 new claims are `boosted_tests.jl`'s steady-state testset, `9.3 s` inside
the suite. It was **5514 assertions in 15m49** at four threads after the moving
hole on the quadrant (2026-10-20; 5500 in 20m31 before the moving core, on a
machine loaded 15–22): 142 of its 150 new claims are `boosted_tests.jl`'s,
`14.6 s` inside the suite, most of it the quadrant-against-box run, and the
other 8 the SimWatch history's. It was **5364
assertions in 16m35** at four threads after the MPI port
(2026-10-19; HEAD before it, on TreeAMR 0.2.3 from a clean checkout, was 5101
in 14m55): its 263 new claims are `mpi_tests.jl`'s comparison of the
workload's lines at one, two and three ranks with the serial run's, whose
`mpiexec` job compiles and runs beside the suite (4m37 on its own), so the
file itself is `0.3 s` at the end. It was **4907 assertions in 15m32** at four threads after the lean
right-hand side (2026-10-05, development machine loaded 9–12; the one-thread
suite was not rerun). It also passes on an EPYC node, x86-64, where the
contraction differences live (Symmetry cn079, 26m21 at four threads). Its 112 new claims are the two kernel spellings:

- their identity with the port on every background
  (`pointwise_identity_tests.jl`);
- their allocation and their kernel launch (`pointwise_tests.jl`);
- their types, `Float32x2` included (`type_tests.jl`).

One claim was relaxed: `evolution_tests.jl`'s dissipation switch, from `1e−290`
everywhere to roundoff of each component. Its two specialisations now round a
product of two roundoff-sized numbers differently (`1.9e−68` on components that
are zero).

The suite was **4690 assertions in 18m43** at one thread and **4698 in
12m59** at four after TreeAMR 0.1.4's two features (2026-10-01, on a machine
loaded 6–10 by other work — read the times against that): `mesh_mapreduce`
changed no count, and `checkpoint_tests.jl` is 78 new claims in `2m02` /
`1m02`, most of it its four chains — a static, an adaptive, a tracked
`:fitted` with a regrid and a moving `:fitted` run, each run twice — and the
compilation of the save and the load. It is over `PLAN.md`'s 30 s rule
because a chain is the claim; the cheapest chains that still reach each
carried piece of state are the ones there. It was **4610 assertions in 14m03** at one thread and **4618 in
10m04** at four after the port to TreeAMR's interpolation and the
16-variable provider (2026-09-26, load 5–7): 78 fewer than before it — the
retired `locate_block` testset's 93 per-point claims, which are TreeAMR's
own now, against the port's and the provider's new ones. It was **4684 assertions in 13m34** at one thread and **4692 in
10m04** at four after the IMEXRungeKutta driver (2026-09-26; the eight extra
at four threads are `stepping_tests.jl`'s per-block owner checks, which have
nothing to say at one thread): `stepping_tests.jl` is 13 or 21 new claims in
about 37 s, and the three minutes the one-thread run lost are OrdinaryDiffEq's
compilation and the right-hand side's allocation, both gone. It was **4670
assertions in 16m33** at one thread and **10m57** at
four after step 8′ (load 3–9): its 11 new claims are `moving_tests.jl`'s
levers, the file `20.8 s` / `18.6 s`. After step 8 it was **4659 assertions
in 17m16** at one thread and **11m36** at four (load 7–13): its 28 new claims are `moving_tests.jl`,
`19.9 s` / `16.9 s`, most of it the compilation of the target's rate in the
cache fill, and one of step 8f's claims — the refusal of G5's chart by the
`:fitted` cycle — is gone with the refusal. After step 8f it was **4632
assertions in 20m59** at one thread and **12m22** at
four, on a machine loaded 8–15 by other work (11m22 at four
threads the evening before at a load of 5): its sixteen new claims are
`fit_tests.jl`'s `Π̃` chain rule and "the snapshot, the hand-over and the
mesh cycle", `25.9 s` / `16.6 s`. After step 8e it was **4616 assertions in
14m21** at one thread and **10m41** at four on the development machine
(load 6–13): the 38 new
claims are the `:fitted` variant's, `47.7 s` / `39.5 s`, most of it one
`0.15 M` run of the fixture and one `Float32` chunk. After step 8e-i it was
**4578 in 15m09** and **10m52** (load 6–8; an earlier
four-thread run under a load of 13 took 37m25, its excess all in
`constraints_tests.jl`'s and `interface_tests.jl`'s compilation — a number
about the machine). Its 168 new claims are `fit_tests.jl`'s 146,
`12.6 s` / `12.9 s`, which fits the state
of the tracked run `tracking_tests.jl` already makes (`tracked_fixture_run`
in `evolution_cases.jl` runs it once for both), and 22 in
`interior_tests.jl` for the boost sign. Step 8d measured **4410 in 15m28**
and **11m21** on a machine shared with sibling agents (load 6–7): its 553
new claims were `tracking_tests.jl`, `54.7 s` / `32.4 s`, of which the two
tracked runs are `40 s` / `18 s`.
Step 8c′ measured **3857 in 14m33** and **10m30** (the tree before it
measured 3798 in 13m45 at one thread the
same afternoon): the default rate's price is `driver_tests.jl`'s variants
testset, `26 s` → `82 s` at one thread, whose `:damped` and `:frozen` runs
go to `1/2 M` because a residual relaxing at `4/M` saturates only there.
Step 8c measured 3798 in 13m25 / 10m12 (its 56 new claims are
11.8 s / 7.2 s: one right-hand side each for the target and the profile,
and one `1/20 M` run for `ρ_max_fixed`); step 8b measured 3723 in 13m00 /
9m33 on a machine shared with a
sibling agent (step 7 measured 3444 in 12m31 / 8m38; step 6 measured 2968
in 13m35 / 10m45 here and 29m32 / 20m29 on Symmetry; the wall clock has
gone *down* while the count went up before, so read each step's numbers as
that step's rather than as a regression). `bounds_tests.jl` is 22.6 s /
10.1 s of it, most of which is its control: two short runs of the step-5
fixture. Most of it is **compilation**, and the things that pay for it
are, in order: `SpacetimeMetrics`' nested forward-mode passes for six
backgrounds at two precisions (step 1's cost, unchanged); a
right-hand-side kernel per `(q, has gauge source, has dissipation,
interior variant, T)`; the ADM constraint kernel, whose first
specialisation is **18 s** and each further one about 4 s; and the four
files whose cost is *arithmetic* rather than compilation —
`driver_tests.jl` (the suite's black hole, 1m59 / 44.8 s),
`interface_tests.jl`, where the ghost fill at `p = 6` is 79 % of every
evaluation, `refinement_tests.jl` (22.1 / 12.8 s) and `horizon_tests.jl`
(28.7 / 12.8 s, of which the two evolutions are most: the interpolator's
own claims are under two seconds and a *find* is a tenth of one).
Several files are over `PLAN.md`'s 30 s rule of thumb and `CODE.md`
records each with what it buys. Before adding a row anywhere, price it: a new `q` or a new element
type is a new kernel; a new background is a new dual pass; a resolution
added to an interface sweep is `N⁴` of ghost filling. Before adding a test
that differentiates a background, look at what `pointwise_backgrounds.jl`
already computes in one pass.

## Commands

The full suite, from the package root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

and the same at four threads — `Pkg.test` does not inherit `-t`, so it
has to be passed explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

The clean-checkout check, which is what the `[sources]` pins exist for: a
tree with no `Manifest.toml` resolves the five pinned packages from
GitHub and TreeAMR from the General registry (from 2026-09-26, at `0.1.3`,
from 2026-10-01 at `0.1.4`, from 2026-10-04 at `0.1.7` and from
2026-10-09 at `0.2.0`; also between
2026-09-21 and 2026-09-23),
and HDF5 with its binary library from General too (2026-10-01; from
2026-10-09 TreeAMR at `0.2.0`, with TreeIOHDF5 from GitHub and HDF5 through
it), and passes. From 2026-09-21 it is a real check —
every source is public, so it works anonymously, which is what CI does:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

It runs from a git worktree as happily as from the checkout, which is how
the per-step agents work; the `[sources]` pins mean every worktree
resolves the same branches.

The MPI workload runs on its own too (added 2026-10-19), from an
environment that has MPI and the package — `test/mpi_tests.jl` starts
exactly this from the suite's sandbox, beside the suite where the machine
has room and to its end *before* the suite where it has not
(`TREEGH_TEST_MPI_CONCURRENT=0` or `1` decides it by hand). Before the
suite is from 2026-10-30: run after it, beside a test process that had
compiled everything, the ranks put the threaded Linux cell's 16 GB runner
down in five of its thirteen attempts from 2026-10-23 on. It is
skipped on a machine with less than 24 GB — three ranks of about 3 GB
beside the suite, which a macOS runner (7 GB) and the serial Linux runner
could not hold (2026-10-19) — unless `TREEGH_TEST_MPI=1`, which CI sets for
its threaded Linux cell only. It
writes `OUTDIR/n0.txt` … `n3.txt`, the serial run and the three rank counts:

```bash
mpiexec -n 3 julia --threads=1 --project=<env> test/mpi_workload.jl OUTDIR
```

Where a step goes over MPI, rank by rank — the scatter, the fill and each of
its stages' local groups, packs and unpacks, the kernel, a right-hand side, an
RK4 step, and each rank's load (added 2026-10-19; `CODE.md`, "Running
distributed"). On the CPU from the MPI environment, on H200s through
`bench/symmetry_mpi_breakdown.sh`:

```bash
mpiexec -n 2 julia --threads=1 --project=<env> bench/mpi_breakdown.jl mpi=true L=16 N=16 radii=8,4,2 reps=2
```

The thread workload runs on its own, which is how a digest mismatch is
bisected — `test/threading_tests.jl` starts exactly this in a subprocess:

```bash
julia --project=. -t 4 test/thread_workload.jl
```

The black-hole runs that are too long for the suite — the default margin
`m = 8` at `q = 4`, the `t = 50 M` run of all three interior variants,
the two harmonic charts, and from step 6 the indicator's calibration and
its adaptive run — are a **script**, run by hand, with its numbers
recorded under "Measured results" (added in step 5) — the indicator's and
the horizon's in `CODE.md`, the interior's in `SINGULARITY_HANDLING.md` (moved there
2026-10-08). It takes
an optional list of sections (`order`, `long`, `charts`, `indicator`,
`horizon`, `bounds`; and `leakage`, `calibration` and `tracked`, which are
not in the default list). An option `key=value` whose key is a section's name selects
that section's subset and so names the section — `bounds=damped6` runs
the one row and nothing by default (amended in step 8c):

```bash
julia --project=. --threads=4 test/hole_runs.jl
julia --project=. --threads=4 test/hole_runs.jl indicator
julia --project=. --threads=4 test/hole_runs.jl horizon
julia --project=. --threads=4 test/hole_runs.jl bounds=cost,damped6,pasted8
```

The `horizon` section (added in step 7) is Kerr's `A`, `M_irr`, `J` and
`M_ch` from sampled data in both charts and at `a = 9/10`, plus the
horizon rows of a `t = 10 M` run; its two spinning-hole meshes are about a
thousand blocks each, which is why they are here and not in the suite. It
takes about eight minutes at four threads.

The `leakage` section (added in step 8a) is eighty-eight `2 M` evolutions
on a 512-block mesh — a ripple inside the horizon against the same run
without it — and is **not** run by default: it is run as **one Symmetry
job**, `julia --project=. --threads=64 test/hole_runs.jl leakage` on a
64-core node, where it fans its groups out into sixteen subprocesses
(45m47 on one `amddebugq` node, inside that queue's hour with a quarter to
spare — a longer `t_end` or a third order would not fit). The
`symmetry-hpc` skill has the cluster mechanics, as for the suite below. A
subset of it runs locally through `key=value` options, one group after the
other:

```bash
julia --project=. --threads=4 test/hole_runs.jl leakage q=2 d=4 eps=0,1/2 lambda=2 t_end=1/4
```

Its predictions are a script of their own, a minute at one thread:

```bash
julia --project=. test/dispersion.jl
```

The `bounds` section (added in step 8b) has three rows, selectable as
`bounds=<row>,…` so that they split across batch jobs: `cost` (the stage
limiter against a right-hand side, seconds), and `damped6` and `pasted8` —
step 5's two failing runs, each three times (no projection, the proposed
gate, the widest gate), with radial bins of the state's validity and a
step-by-step replay of the fatal chunk. Those two took 14 and 26
minutes on one Symmetry node; `TREEGH_BOUNDS_TEND` shortens them for a
smoke test and `TREEGH_BOUNDS_AUTOPSY=1` forces the replay.

The `calibration` section (added in step 8c) is step 8c's experiments on
the step-5 fixture — the E3 sweep to `3/20 M`, the `n_L × ρ_max` scan with
an inexact target, the `ε_KO(r)` profile, the E1 and E2 targets, E0's hard
step on 8a's uniform mesh, the controls, and the survivors to `50 M` — in
groups, `calibration=<group>+<group>` (`+` or `,`): `sweep`, `scan`,
`profile`, `targets`, `controls`, `e0`, `e0p`, `exact`, and `long1` …
`long8` for the lists of `50 M` runs written into the script. With sixteen
or more threads it fans its runs out to subprocess workers (four threads a
screen, eight a `50 M` run, sixteen an E0 pair), whose logs carry one row
per chunk; each group was one `amddebugq` job — a screen group about 20
minutes, a `50 M` group of eight about 30 (`symmetry-run.sh …
"hole:calibration=long3"`). A single run, shortened, is how it is
validated locally:

```bash
julia --project=. --threads=4 test/hole_runs.jl calibration runs=e3-n8-r4-c t_end=1/2
```

The `tracked` section (added in step 8d) is not in the default list either:
the suite's tracked hole against step 5's sphere with the same layer, to
`5 M` by default or to `tracked=<t_end>`, four minutes at four threads:

```bash
julia --project=. --threads=4 test/hole_runs.jl tracked
julia --project=. --threads=4 test/hole_runs.jl tracked=1/2
```

The `generic` section (added in step 8f) is the measurement matrix and is
not in the default list: groups `ks0` (Kerr-Schild `a = 0` to `50 M`, seven
rows), `ks9` (Kerr-Schild `a = 9/10` to `20 M` on 1632 blocks), `harm`
(harmonic `a = 0`, and `a = 7/10` at `5/128`), `h7` (G5's chart at `5/256`,
a node on its own), `boost` (a boosted hole crossing a fixed fine region,
the rates, the coasting track) and `probe` (harmonic `a = 9/10`, host-side,
three minutes at four threads). A row alone is `generic=<label>`;
`t_end=<t>` shortens every row, `budget=<s>` stops them at a deadline and
records how far they got, and `tag=<name>` names the workers' directory.
With sixteen or more threads each row is a subprocess worker with one BLAS
thread (seven OpenBLAS pools spinning put a node at twice its cores). The
long groups ran on `amdq` (`ks0` about two hours, `ks9` about 2.3) and the
rest in `amddebugq` hours with `budget=3300`. Step 8f ran them through a
scratch copy of `.claude/orchestration/symmetry-run.sh` with three changes
the orchestration's own script does not have: `PART` and `TLIM` for the
partition and the limit, `~` for a space in a job
(`'hole:generic=boost~budget=3300~tag=boost2'`), and **no precompile on the
login node** — its Cascade Lake image made two jobs starting together on
EPYC nodes race to rebuild the package and fail ("Unable to find compatible
target in cached code image"):

```bash
julia --project=. --threads=4 test/hole_runs.jl generic=ks0-fitted t_end=1/2
julia --project=. --threads=4 test/hole_runs.jl generic=probe
```

The `fitted` section (added in step 8e) is not in the default list either:
three rows, `fitted=fixture` (the suite's tracked hole to `1 M` under four
choices of initial data against `:damped`, two minutes at four threads),
`fitted=boosted` (a moving seed on 848 blocks, `:fitted` and `:damped`,
ninety seconds) and `fitted=harmonic` (harmonic Kerr at `a = 9/10` on 2472
blocks at `h = 5/256` — the initial data, one right-hand side, and the run,
which ends in its first chunk; three minutes and about 3 GB):

```bash
julia --project=. --threads=4 test/hole_runs.jl fitted=fixture,boosted,harmonic
```

The `moving` section (added in step 8) is G5's runs and is not in the default
list: groups `g5` (G5's chart crossing a box of `5 M` from `x = 2` to
`x = −1.9` in `13 M`, adaptive, and the same hole at rest — two workers on
one `amdq` node, about seven hours), `g5u` (the crossing in a box of
`5/2 M` against a uniform mesh at `5/256`, 256³ points, about 90 GB and five
hours), `conv` (the frozen hierarchy, `N = 8, 12, 16`, on G5's chart and on
the boosted `a = 0` hole with the analytic sphere; the `N = 16` G5 row is
about four hours at 36 threads), `ctl` (the boosted `a = 0` hole across the
box: `:fitted`, `:damped` at `20/M`, `:frozen`, and the adaptive-against-
uniform pair at `5/128`) and the screens `g5v` … `g5z` (the moving layer's
parameters to `M/2`, one `amddebugq` hour each). `moving=mesh` runs only the
initial-data cycle of every adaptive row and prints its mesh (three minutes
at four threads); `moving=<label>` one row; `t_end`, `budget`, `tag` and
`threads` as in `generic`. Each worker prints one line per chunk: the masked
error, the error outside a fixed sphere enclosing the horizon (what the
convergence rows compare, since the masked region moves with the offset
surface), the shell's `C_a`, the layer's error off the truth by depth and by
side (trailing or leading), the finder's numbers and the horizon's extent
along the boost against the contraction. Stagger job starts, and do not
rsync the worktree while a job's workers are starting:

```bash
julia --project=. --threads=4 test/hole_runs.jl moving=mesh
julia --project=. --threads=4 test/hole_runs.jl moving=ctl-a0-fitted t_end=1/2
```

Step 8′ added the groups `l0` (the trailing-side levers on the boosted
`a = 0` hole in the `5/2 M` box to `M/2`, ten rows of about four minutes at
four threads here — run them three at a time), `l7` (the same on G5's chart,
one `amddebugq` hour) and `g5t` (G5's crossing with `trail_ramp = 9/10`,
eleven hours alone on an `amdq` node). **Give every long Symmetry study a
remote directory of its own** (`symmetry-run.sh <worktree> <unique name>`):
the package image of a project path is one cache slot, and rsyncing changed
sources into a directory whose jobs are running, then precompiling, rewrites
the `.so` they have mapped — the likely cause of step 8's `SIGBUS`es, which
did not recur from `step-8-sigbus` and `step-8-t`.

**A long row is a chain of jobs** (added 2026-10-01): `checkpoint=<dir>`
makes every `generic` or `moving` row checkpoint each chunk to
`<dir>/<label>.it….h5` and continue from the newest file there, and
`walltime=<s>` stops it with a checkpoint before that many seconds of its
`evolve!` (leave the job's startup and compilation as margin below the
queue's limit). The same command resubmitted continues the row; a row that
is finished refuses to restart ("nothing left to run"). Give each study its
own `<dir>`, as its own remote directory:

```bash
julia --project=. --threads=4 test/hole_runs.jl moving=l0-trail-9 t_end=1/2 checkpoint=out/ck walltime=3000
```

The step benchmark (added 2026-09-26) is `bench/stepping.jl`, driven by
environment variables its header lists; on Symmetry it ran pinned
(`JULIA_EXCLUSIVE=1`, `srun --cpu-bind=none`, the fastest of the four
placements) in a scratch copy with `OrdinaryDiffEqLowOrderRK` and
`SciMLBase` added for its comparison rows, and on an H200 in a copy with
`CUDA` added — never in the package's own environment:

```bash
BENCH_MODE=step BENCH_CASE=wave,hole julia --project=. -t 4 bench/stepping.jl
```

The GPU right-hand-side prototypes (added 2026-10-05) are `bench/rhs_lab.jl`:
`key=value` options, one mode per round of `CODE.md`'s "The right-hand side on an
H200", listed in its header. Like `bench/stepping.jl` on a device, it runs from a
copy with `CUDA` added — on Symmetry `rhs-gpu-lab`, one H200 a job in `h200debugq`.
`bench/rhs_lab_source.jl` is the lean source it measures. `bench/rhs_lab_cpu.jl`
checks and times that source on the CPU in the package's own environment, and
`bench/sass_stats.jl` counts the instructions of a SASS dump it writes:

```bash
julia --project=. bench/rhs_lab.jl mode=round9 N=32 roots=8
```

```bash
julia --project=. bench/rhs_lab_cpu.jl
```

The CPU right-hand-side lab (added 2026-10-05) is `bench/rhs_cpu_lab.jl`, the
measurements of `CODE.md`'s "The right-hand side on a CPU": where `gh_rhs!` goes,
the kernel at each SIMD width (`simd_width`), checked against the scalar one, the
stencils without their zero weights (`LAB_ZW=1`), the native code, the head's pieces
and a profile of the ghost fill — modes and variables in its header; `LAB_CASE=hole`
for the hole fixture. It runs in the package's own environment, pinned on Symmetry as
`bench/stepping.jl` is; `BENCH_SIMD=1` gives `bench/stepping.jl` the scalar kernel:

```bash
LAB_MODE=breakdown,simd julia --project=. -t 4 bench/rhs_cpu_lab.jl
```

**On Symmetry** (added in step 6, and step 9 writes the batch job for
real): the suite and the long studies run there as one SLURM job each on a
64-core EPYC node, which is what makes them parallel — a node *core* is
about **twice as slow** as the development machine's, so the cluster buys
throughput and not wall clock (measured in step 6: the same 2968
assertions in 29m32 / 20m29 at one and four threads there against
13m35 / 10m45 here). One quirk that
will bite a batch script: **do not export `JULIA_EXCLUSIVE=1` for the
one-thread suite.** A one-thread parent then pins itself to a single CPU,
and `test/threading_tests.jl`'s four-thread subprocess inherits the
affinity mask and aborts with `Too many threads requested for
JULIA_EXCLUSIVE option` — the environment, not the code. Two more (found 2026-10-23, with
the folds): **the suite no longer fits `amddebugq`'s or `h200debugq`'s hour**
on these nodes — three jobs stopped at `type_tests.jl` at 60 min, with two
compiling against each other's pidfiles — so submit it to `amdq` or `h200q`;
and **`mpi_tests.jl` inside a SLURM job needs `HYDRA_LAUNCHER=fork`**
(with `MPIR_CVAR_CH4_SHM_POSIX_RELEASE_GATHER_ENABLE=0`): MPICH's `mpiexec`
otherwise starts its ranks with `srun`, which waits for a step the suite
already holds ("Requested nodes are busy") until the workload's 3600 s
timeout. With both, the suite passed on an EPYC 7532 (cn084), an EPYC 8534P
(cn112) and an H200 host (cn111, Xeon Gold 6442Y), the MPI comparison in its
own job (cn079, cn113). The `symmetry-hpc`
skill has the rest of the cluster's mechanics.

**SimWatch in a batch job** (2026-10-21): `bench/simwatch.sh` is a verbatim
copy of SimWatch 1.3.0's `writers/simwatch.sh` (`simwatch_queued` after
`sbatch`, `simwatch_mark DIR failed "…"` on a non-zero exit; neither changed
since SimWatch 1.0.0). A new sbatch script sources the copy inside its own
rsynced directory, `. "$R/bench/simwatch.sh"` with `R` that directory — not
`spin-a06/out/simwatch.sh` or `ks-study2/out/simwatch.sh`, the older copies
the running studies' scripts name and source again at their exit, which stay
as they are.

The octant runs (added 2026-10-02) are `test/octant_runs.jl`, one run per
call, `key=value` options listed in its header — the case (`case=minkowski`
or `ks`, `interior=damped|fitted`, margins, `fit_cont`; from 2026-10-04
`octant=reflecting|rotating` and the spin `a=`, which needs `rotating`), the mesh (`L`, `N`,
`roots`, `radii`), the run (`t_end`, `cfl`, `chunk`), noise, `backend=cuda`,
`out=<dir>` (CSV, `records.csv`, `simwatch.toml` — from 2026-10-19 with
sparklines of the constraints, the error and the horizon, and thumbnails in
`out/plots/` drawn by `test/simwatch_plots.jl`, plain Julia: from 2026-10-21
SVG on a transparent background with `currentColor` axes, which SimWatch 1.3.0
shows as it is or flips for a dark terminal by its user's `image_theme`, in
place of the dark-on-light PNGs; `group=` names the SimWatch group) and
`checkpoint=<dir>`.
On Symmetry they ran one H200 each from a copy with `CUDA` added to its
`Project.toml` (`ks-octant`, `ks-study2`); `cfl = 1/2` is as accurate as
`1/4` there and half the cost:

```bash
julia --project=. --threads=4 test/octant_runs.jl case=ks interior=fitted L=8 N=16 roots=2 radii=4,2 t_end=1 chunk=1/2 cfl=1/2 amplitude=0 out=out/smoke
julia test/octant_study.jl out/study t_from=8 series=dA64,dA96,dA128:16,24,32
julia --project=. --threads=4 test/octant_runs.jl case=ks octant=rotating a=1/2 L=8 N=16 roots=1 radii=4,2,1 r_0=3/4 r_1=5/4 t_end=1/2 chunk=1/4 out=out/spin
```

From 2026-10-20 the same script runs a hole moving along `x` on the quadrant
(`octant=quadrant v= x0= path=`; `variant=` and `trail_ramp=` for a tracked
layer, `rho=` for the rate), and `test/boosted_study.jl` reads a directory of
such runs (`dir=`, default `.claude/data/boosted-ks`, one subdirectory per
batch) and prints the Markdown tables of `SINGULARITY_HANDLING.md`'s "The
boosted Kerr-Schild hole at binary speed" — moving against resting, the
orders, trailing against leading, the horizon, the layer residual. The first
line is a local smoke run; the second is the production `:damped` row at
`1/32` (on Symmetry, from a copy with `CUDA` added, one H200, about
`2.7 h`):

```bash
julia --project=. --threads=4 test/octant_runs.jl case=ks octant=quadrant v=1/5 x0=-1 L=8 N=16 roots=1 radii=4,3,2 r_0=5/8 r_1=23/16 t_end=1 chunk=1/2 cfl=1/2 amplitude=0 out=out/quad
julia --project=. test/octant_runs.jl backend=cuda case=ks octant=quadrant L=64 roots=4 radii=32,16,8,4 N=32 t_end=40 chunk=1/2 cfl=1/2 amplitude=0 r_0=5/8 r_1=23/16 v=1/5 x0=-4 path=8 out=out/bk32 checkpoint=out/bk32/ck walltime=40000
julia --project=. test/boosted_study.jl times=1,4,8,16,24,32,40
```

From 2026-10-21 the production row is the default `4/M` (the moving layer's
target rate; `rho=8` is no longer needed), and `baseline=<dir>` compares each
moving row of `dir` with the same kind's row in another data directory — how
the rate was measured against the runs without it (`.claude/data/boosted-ks-rate`
holds the new batch `ks3/` and links to the resting controls in `rest/`):

```bash
julia --project=. test/boosted_study.jl dir=.claude/data/boosted-ks-rate baseline=.claude/data/boosted-ks rate=4,8
```

From 2026-10-21 `octant=rotating_quadrant` (with `a=`) moves a spinning hole
along its spin, `z`, on the root brick `(r, r, 2r)` and a capsule along `z`;
`x0` and `path` are along `z` there. The production `:damped` row at `1/32`
(job 580105 in `boosted-ks-4`, about `2.8 h` on one H200), and `prefix=s`,
which reads the spinning family's rows (`sk<N>`, `sk<N>s`, `sf<N>`, `sf<N>s`;
`a=` sets the expected horizon, default `1/2`) and prints
`SINGULARITY_HANDLING.md`'s "A spinning hole along its axis" from
`.claude/data/boosted-ks/ks4` (the shell `r1.93185` is the equatorial horizon
radius `√(r₊² + a²)`):

```bash
julia --project=. test/octant_runs.jl backend=cuda case=ks octant=rotating_quadrant a=1/2 L=64 roots=4 radii=32,16,8,4 N=32 t_end=40 chunk=1/2 cfl=1/2 amplitude=0 r_0=3/4 r_1=23/16 v=1/5 x0=-4 path=8 out=out/sk32 checkpoint=out/sk32/ck walltime=40000
julia --project=. test/boosted_study.jl prefix=s shells=r1.93185,r3,r5 times=1,4,8,16,24,32,40
```

From 2026-10-22 a production run takes **`chunk=1 split=0`**: the analysis
row every `M` and no trailing/leading split, which is `2.5×` cheaper than
`chunk=1/2` with the split (`CODE.md`, "The analysis cadence on an H200";
`finder=0` turns the finder off, which saves under a second a find). The
aligned campaign's rows are one such command each. The line below is its
`a = 9/10`, `v = 1/2`, `1/96` row (job 580900 in `boosted-ks-7`, `3.5 h` on
one H200). `test/aligned_study.jl` reads the campaign's data
(`dir=`, default `.claude/data/aligned`, one subdirectory per batch; a batch
named `*obsolete*` is skipped unless `batches=` names it; `runs=`, `times=`)
and prints `SINGULARITY_HANDLING.md`'s "Aligned spin and velocity": the runs,
moving against resting, the orders, the horizon, the layer residual, the cost
and the verdict matrix. It needs no package, so it runs without `--project`:

```bash
julia --project=. test/octant_runs.jl backend=cuda case=ks octant=rotating_quadrant a=9/10 v=1/2 x0=-2 path=10 L=64 roots=4 radii=32,16,8,4,3,2 N=24 t_end=20 chunk=1 split=0 cfl=1/2 amplitude=0 r_0=19/20 r_1=9/8 group=aligned out=out/a9v5h96 checkpoint=out/a9v5h96/ck checkpoint_every=4 walltime=40000
julia test/aligned_study.jl
```

From 2026-10-25 (step B1) `gauge=dh` rolls the damped harmonic gauge on from
the hole's own source (`sigma_g=10 t0=0 sigma_r=32 A_L1=1 A_L2=0 A_S=1 e=2`
are the defaults), and `source=composite` is the one-hole composite. The
first line is a local smoke run, the second step B1's moving `1/32` row with
its shells about the tracked horizon (`b1dm32t`, jobs 582624, 582634 and 582663
in `step-B1-gauge`, one H200), the third the kernel's cost per point with each
source (`backend=cuda` on an H200, from a copy with `CUDA` added):

```bash
julia --project=. --threads=6 test/octant_runs.jl case=ks octant=quadrant L=16 N=16 roots=1 radii=8,6,4,3 r_0=5/8 r_1=23/16 t_end=4 chunk=1/2 cfl=1/2 amplitude=0 x0=-1 path=4/5 v=1/5 gauge=dh sigma_g=1 out=out/s16dm
julia --project=. test/octant_runs.jl backend=cuda case=ks octant=quadrant L=64 roots=4 radii=32,16,8,4 N=32 t_end=40 chunk=1 split=1 cfl=1/2 amplitude=0 interior=fitted margin=16 n_L=20 fit_cont=2 lmax_fit=8 v=1/5 x0=-4 path=8 gauge=dh sigma_g=10 sigma_r=32 shell_center=track group=step-B1 out=out/b1dm32t checkpoint=out/b1dm32t/ck checkpoint_every=4 walltime=40000
julia --project=. -t 1 bench/gauge_source_cost.jl N=16 roots=4
```

From 2026-10-26 (step B2) `interior=tracked` is the tracked analytic target
(`r_0=`, `r_1=` as for `damped`), with `prediction=quadratic|line`,
`find_interval=` (the cap, in `M`) and the sensitivity study's perturbations
along the motion axis: `target_mass=` (the target's mass factor), `target_dv=`
(added to its velocity), `target_dx=` (added to its center, in finest cells)
and `seed_dv=` (added to the seed track's velocity); `shell_center=track` reads
the track's center there. `records.csv` adds the track's prediction, cadence,
center and velocity. The first line is a local smoke run, the second the
production tracked row at `1/24` (`b2t24`, job 582988 in `step-B2-tracked`, one
H200, `25 min`), the third prints `SINGULARITY_HANDLING.md`'s "The tracked
analytic target" from the campaign's data (no package needed):

```bash
julia --project=. --threads=3 test/octant_runs.jl case=ks octant=quadrant L=16 N=16 roots=1 radii=8,6,4,3 r_0=5/8 r_1=23/16 t_end=4 chunk=1/2 cfl=1/2 amplitude=0 x0=-1 path=4/5 v=1/5 interior=tracked out=out/s16t
julia --project=. test/octant_runs.jl backend=cuda case=ks octant=quadrant L=64 roots=4 radii=32,16,8,4 N=24 t_end=40 chunk=1 split=0 cfl=1/2 amplitude=0 r_0=5/8 r_1=23/16 v=1/5 x0=-4 path=8 interior=tracked group=step-B2 out=out/b2t24 checkpoint=out/b2t24/ck checkpoint_every=4 walltime=40000
julia test/tracked_study.jl dir=.claude/data/step-B2 times=1,2,4,8,16,24,32,40
```

From 2026-10-26 (step B3) `test/binary_runs.jl` runs a pair of holes
(`head_on_holes`, `binary_case`, `holes_forest`), one run per call, `key=value`
options in its header: the separation and the box, each hole's layer for a hole of
mass one (scaled by `M_A`), `target=superposed|own`, `feed=tracks|seed`,
`momentum=superposition|gauge`, `companion=`, `adm=`, `backend=cuda`, `mpi=true`,
`checkpoint=` and `walltime=`. Its CSV has the constraints over the evolved
region and in bands about the nearer hole's track, each hole's track and horizon,
the separations (tracks, found origins, Newtonian free fall) and the cost. The
first line is a local smoke run (`q = 2`, `h = 1/16`, six minutes at four
threads), the second step B3's head-on row (`b3sg48`, job 583289 in
`step-B3-holes`, `27 min` on one H200, from a copy with `CUDA` added):

```bash
julia --project=. --threads=4 test/binary_runs.jl d=8 L=8 N=8 roots=4 radii=2,3/2,5/4 q=2 r_0=3/5 r_1=7/5 margin=4 t_end=1 chunk=1/10 finder_N=12 spin=0 adm=0 feed=seed momentum=gauge out=out/pair
julia --project=. test/binary_runs.jl backend=cuda d=8 L=32 roots=8 radii=16,8,4,2 N=24 t_end=2 chunk=1/4 cfl=1/2 feed=seed momentum=gauge out=out/b3sg48 checkpoint=out/b3sg48/ck walltime=12000
julia test/binary_study.jl dir=.claude/data/step-B3 runs=b3s48,b3sg48
```

The third prints the rows' tables from their CSVs (no package needed).

From 2026-10-26/27 (step B3's review) the gauge source may be fed: on one tracked
hole `test/octant_runs.jl interior=tracked source=composite feed=own|seed|tracks|controlled
control_time=<τ in M>` (the CSV then ends with the track's and the line's center,
velocity and acceleration, the error against the hole on the track's line and the
residual against the layer's own target; use `shell_center=track`), and on a pair
`test/binary_runs.jl feed=controlled control_time=<τ for a hole of mass one>`;
`control_law=pid|pd` (default `pid`) chooses the loop, and on a pair
`control_model=0|1` (default `0`) the Newtonian model. The first line is one of the
review's screens (`b3fp24`, job 585660, `15 min` on one H200), the second the
proposed pair row (`b3iz48`, jobs 585985 and 586001, two `h200debugq` links of
under an hour, `walltime=3000`), the third and fourth their tables (no package
needed; `base=` is step B2's data):

```bash
julia --project=. test/octant_runs.jl backend=cuda case=ks octant=quadrant L=64 roots=4 radii=32,16,8,4 N=24 t_end=20 chunk=1 split=0 cfl=1/2 amplitude=0 r_0=5/8 r_1=23/16 v=1/5 x0=-4 path=8 interior=tracked source=composite feed=controlled control_time=10 control_law=pid shell_center=track group=step-B3 out=out/b3fp24 checkpoint=out/b3fp24/ck checkpoint_every=4 walltime=3000
julia --project=. test/binary_runs.jl backend=cuda d=8 L=32 roots=8 radii=16,8,4,2 N=24 t_end=6 chunk=1/4 cfl=1/2 momentum=gauge feed=controlled control_time=10 control_law=pid control_model=0 out=out/b3iz48 checkpoint=out/b3iz48/ck checkpoint_every=4 walltime=3000
julia test/fed_study.jl dir=.claude/data/step-B3 base=<B2's .claude/data/step-B2>/b2t24 runs=b3fa24,b3fb24,b3fd24,b3fp24,b3fc24,b3fe24,b3fpr24
julia test/binary_study.jl dir=.claude/data/step-B3 runs=b3sl48,b3cg48,b3cd48,b3ng48,b3nd48,b3ig48,b3iz48
```

Later: the CLI (`julia --project bin/gh.jl --case=boosted_kerr …`) and
the viewers (`julia --project=bin bin/visualize.jl`) arrive in step 9,
and device tests behind
`TREEGH_TEST_BACKEND` (`cuda` on Symmetry's H200 is the requirement,
`metal` on the development machine is desirable) in an environment of
your own that has the device package, in step 9. Neither this package
nor TreeAMR depends on a device package. The `symmetry-hpc` skill has
the cluster mechanics (modules, SLURM, NUMA, precompilation).

## Things that will bite

Carried over from TreeAMR, TreeWave and TreeHydro where they apply, plus
what is specific to a GR code. Each is in `CODE.md` with its reason.

- **Five dependencies are pinned to GitHub `main`, and TreeAMR comes
  from the registry** (amended 2026-09-26, when Erik registered TreeAMR
  0.1.3 for its M11 interpolation; from 2026-09-23 to then TreeAMR was
  pinned too). `SpacetimeMetrics`, `ApparentHorizonFinder`,
  `KorzynskiSpin`, `IMEXRungeKutta` and `TreeIOHDF5` (from 2026-10-09) are
  what `Project.toml`'s `[sources]` entries resolve, and TreeAMR is
  General's release under `[compat]` `0.2.6` (from 2026-10-23, for M15's
  folds over points; `0.2.4` from 2026-10-22, for its
  nested-mesh ghost fill; `0.2` from 2026-10-09, when the
  checkpoints moved to TreeIOHDF5; `0.1.7` from 2026-10-04, for M12's
  rotating seam; `0.1.4` from 2026-10-01, for M9a's checkpoints; `0.1.3`
  before) — so `~/src/jl/…` is *not*
  what the tests see; an unpushed change there is invisible here, a pushed
  change to TreeAMR's `main` is invisible too until it is *released*, and
  the local SpacetimeMetrics checkout has been behind `main` before. Read
  what Pkg installed under `~/.julia/packages/` when in doubt about an API.
  Say so rather than editing a checkout and assuming the tests see it. A
  local `Manifest.toml` resolved while TreeAMR was pinned needs
  `Pkg.update("TreeAMR")` (a plain `resolve` refuses the old version);
  `test/prerequisite_tests.jl` is what notices when a moving branch or a
  release drops a name.
  **`KorzynskiSpin`'s repository became public on 2026-09-21**, so its URL
  is plain `https`, the read-only deploy key and the `ssh-agent` step and
  `JULIA_PKG_USE_CLI_GIT` are all gone from `CI.yml`, and an anonymous
  clean checkout and a fork's pull request both resolve it — the
  clean-checkout check below finally proves what it claims. Do not vendor
  a package and do not add a local-path source.
- **Most `[sources]` pins are necessary, not chosen.** (The section needs
  Julia 1.11, which no longer constrains anything: the floor is 1.13.) A
  pin goes only when its package is in General at the bound asked for:
  `KorzynskiSpin` and `TreeIOHDF5` are not registered at all, and
  `ApparentHorizonFinder` **2.1** is not released (General carries
  `2.0.0`, and `[compat]` here asks for `2.1`). `SpacetimeMetrics` is
  registered at the bound this package asks for, so its pin is a choice
  rather than a necessity.
- **TreeIOHDF5 `main` can be ahead of the TreeAMR it admits** (found
  2026-10-22). TreeIOHDF5 0.1.4 reads TreeAMR 0.2.5's `FieldSet.halfturn`,
  but its `[compat]` lists TreeAMR `=0.2.2, …, =0.2.5`. So the `[sources]`
  pin to `main` resolves it beside TreeAMR 0.2.4, and the first checkpoint
  throws in `set_layout` ("type FieldSet has no field `halfturn`"). The
  first `boosted-ks-7` batch lost ten jobs to it, each at `4 M`. That is
  an upstream compat bug: TreeIOHDF5 0.1.4 should require `=0.2.5`. Until
  it does, a copy that stays on TreeAMR 0.2.4 pins TreeIOHDF5 0.1.3, as
  `boosted-ks-7` did. A run without `checkpoint=` does not notice.
  **(Resolved 2026-10-23** for this package: it requires TreeAMR 0.2.6,
  which TreeIOHDF5 `main` (0.1.5) admits. A copy of an older commit on
  TreeAMR 0.2.4 still needs the 0.1.3 pin.**)**
- **`notes/` is read-only.** The copies carry their provenance; when
  `CODE.md` departs from them, `CODE.md` says so. Do not "fix" a copy.
- **Two derivative index conventions.** `SpacetimeMetrics.dmetric`
  returns `dg[a, b, c] = ∂_c g_ab` (derivative axis *last*); GHSO2's
  pointwise algebra uses `dg[a, b, c] = ∂_a g_bc` (derivative axis
  *first*). Convert in one place, `initialdata.jl`, and test the
  conversion; a wrong convention produces a solution that looks right
  in Minkowski and wrong everywhere else.
- **The packed component order is** `(tt, tx, ty, tz, xx, xy, xz, yy,
  yz, zz)`, column-major lower triangle, GHSO2's. `h` is variables
  `1:10`, `Π` is `11:20`. Nothing indexes a component by a literal
  outside `pointwise.jl`'s pack/unpack helpers.
- **Off by `G`.** Owned point `i` is stored at `i + G[d]`;
  `coordinates(fs, b, idx)` takes **stored** indices; a kernel launched
  by `map_blocks!` (default range) gets the owned index and adds `G`.
  Every stencil reaches `±(q/2 + 1)` and `G = q/2 + 1` is exactly that.
  Getting this wrong produces plots that look almost right. The
  right-hand-side kernel does the addition **once**, into a linear index,
  and the stencils step by a stride per axis — same element, a quarter of
  the time (measured in step 3), and one place to get it wrong instead of
  1600.
- **`N = 8` does not exist at every order.** TreeAMR's vertex invariant
  is `N ≥ 2G + 2`, so `q = 4, 6, 8` need `N ≥ 8, 10, 12`. The tests use
  `N = 10` at `q = 6`; `PLAN.md`'s "`N = 8`, `q = 2, 4, 6`" was one row
  wider than the mesh allows, and `FieldSet` says so rather than running.
- **A closure in a kernel must not capture a name assigned twice in that
  body** (found 2026-09-26). Lowering boxes such a variable — even when the
  two assignments are in branches a `Val` compiles away — and a `Core.Box`
  is untyped: `gh_rhs_kernel!`'s `:none` branch and its layer branch both
  assigned `∂ₜh, ∂ₜΠ`, which the `ntuple(Val(NC)) do v … end` store
  captured, so every right-hand side on the CPU allocated ~570 bytes a
  point (7–15 % of its time at four threads, and 3–4× at 64, where the
  threads contend for the garbage collector) and no device would compile
  the kernel at all ("unsupported dynamic function invocation (call to
  getindex)"). Give each branch its own names. The symptom on the CPU is
  `@allocated gh_rhs!(…)` growing with the block count; the ghost fill's
  13 kB is fixed. `Profile.Allocs` names the line.
- **KernelAbstractions refuses a `return` in a kernel** — anywhere in the
  body, closures included, which is what `ntuple(Val(10)) do v … end` is.
  End the block with the value instead. The error names the kernel and
  arrives at precompilation, so it is cheap; it is here because the
  package's own convention asks for an explicit `return` everywhere else.
- **The time integrator is IMEXRungeKutta's `RK4()`, by block owner, and
  every run goes through `gh_integrator`/`gh_solve`** (`src/stepping.jl`,
  from 2026-09-26; OrdinaryDiffEq and SciMLBase are no longer dependencies
  of the package or its tests). `state_partition` hands IMEXRungeKutta the
  entries of TreeAMR's `threadchunks(nblocks)` per thread, so each block's
  stage arithmetic runs on the thread `map_blocks!` runs the block on; on a
  device it is `nothing`, the broadcast path. `evolve!` passes `alias_u0 =
  true`: the integrator steps the run's own state vector in place, so an
  observer that keeps `u` keeps a vector that changes (none does). The
  integrator's `p` and `dt` are constants and each `init` allocates and
  first-touches four state-sized scratch vectors — 0.13–0.36 s at 64
  threads on 320 MB, measured 2026-09-26 — so `evolve!` builds **one per
  chunk** and swaps a moving hole's refilled problem in between pieces
  (`swappable = true`: the integrator's `p` is a `ProblemRef`, `integ.p.p =
  p′`), rather than one per piece — and **each chunk's integrator takes
  over the previous chunk's scratch** (`reuse = integ_prev`, IMEXRungeKutta
  1.2), so the allocation is once per mesh. A regrid that moves the mesh
  sets `integ_prev = nothing`: its state vector is a new one, usually of
  another length, whose scratch IMEXRungeKutta would refuse by name (it
  checks the length, the array type and the partition). Its
  `RK4` clashes with OrdinaryDiffEq's by name: the package `import`s it
  `as IRK`.
- **The RHS never mutates `u`, and the state has exactly three writers**
  (amended in step 8b). The interior layer is a term of the right-hand
  side, `du = w F(u) − ρ (u − u_exact)`, so the integrator's arithmetic is
  the first writer; the `:pasted` paste of the ball `r < r_1`, from RK4's
  `step_limiter!`, is the second; step 8b's range projection
  (`src/bounds.jl`), from the `stage_limiter!` on every stage vector and
  once after the initial fill and every regrid, is the third. Both limiters
  are **integrator keywords**, passed by `gh_integrator` (`src/stepping.jl`)
  and nowhere else. **Under IMEXRungeKutta (from 2026-09-26) the two are
  one limiter on every state vector** (Erik's decision): `gh_limiter!`, the
  projection and then the paste, is passed as both the stage limiter (the
  stage values `f_exp!` reads) and the step limiter (the result, which the
  next step's first stage reads), because IMEXRungeKutta's stage limiter
  never sees the result. Do not split them again, and do not add a fourth
  writer.
- **A limiter writes back only where it fired, and a run on which nothing
  fires is bit for bit the run without it.** The range projection keeps a
  healthy quantity's bits — it reassembles `h_tt`, `h_ti` or the spatial
  block only when that block's range was violated, because `(−1 + h_tt) +
  1` is not `h_tt` — and the kernel writes a point only where `hit` came
  back true. `test/bounds_tests.jl`'s control asserts `isequal(u, u′)` for
  the fixture with and without bounds; it is the claim every experiment
  that compares the two rests on. A change that makes it a tolerance has
  broken the projection, whatever else it fixed. The tests carry an
  `8 eps` slack and a fired result is re-tested by the next call's own
  arithmetic, which is what makes the *flag* idempotent too (without the
  re-test, 18 in 20 000 random states re-fired on an ill-conditioned `γ`).
- **A `NaN` in the core is a hit, not the end of the run** (step 8b):
  `max_speed_of` and the record's `finite` count non-finite values at
  evolved points only (`evolved_nonfinite`). The projection is gated at
  `r < r_gate ≤ r_1 − (stencil reach)·h`, asserted at every regrid; the
  outer part of the layer, `r_gate ≤ r < r_1`, is unguarded by design, and
  a degenerate metric there still ends a run through the `DomainError` of
  `metric_quantities`' `sqrt`. **Step 5's two failures are not deep**
  (measured in step 8b): `N = 6` `:damped` and `N = 8` `:pasted` both
  degenerate in the evolved shell just outside `r_1`, the projection never
  fires on either at any legal gate, and the runs with it end bit for bit
  where the runs without it do. Read the validity rows (`min_α_shell`,
  `max_Π_shell`) and the shell error, not `bounds_hits`, for a failure at
  `r_1`.
- **`ρ_max = 1/dt` is a grid rate, and a wrong target cannot survive it**
  (measured in step 8c). It is about `107/M` on the fixture — a paste two
  cells deep — and every inexact target at it ends its run in 2–15 M on a
  ramp of up to 8 cells, degenerating in the shell outside `r_1`; the
  calibrated layer is `4/M` — the default from step 8c′ — with a ramp of at
  least 8 cells (`ρ_ramp = 1`, `r_0 = r_1 − n_L h`). A rate of `1/M` is too
  slow the other way: a deep layer is not held and fails from the inside.
- **A layer target is evaluated wherever the layer is**, `r_0 ≤ r < r_1`,
  and at the core's sphere for `:pasted` — so its own singular set has to
  be inside `r_0` exactly as the background's does, and nothing checks it:
  step 8c's E2 target translated by `4h` puts its singular point on a grid
  point of a 12-cell layer and throws at `t = 0`. The `residual` row is the
  layer's distance from the *background*, not from the target, and is
  large by construction for a wrong one.
- **A `HoleCenter` is sometimes the *tracked* trajectory** (step 8d). A
  `FittedInterior`'s center is `track_center(track)`, `c_find + v_est (t −
  t_find)`, so its masks, its `interior_radius`, the range projection's gate
  and the core rule are about the track, while `case.center` — the damping
  profile, the refinement centroid's offset, `track_offset` — stays the
  analytic one. Before comparing a radius with an analytic value, say which
  center it is about: `find_gh_horizon`'s `r_min` is about its `center =`
  (the analytic one unless given; a tracked run passes the prediction, so its
  `center_offset` is the track's prediction error), and `origin_r_min` is
  about the found surface's own origin, which is what the track and its
  shape are.
- **The fit is made in `(log α, β^i, γ_ij, Π_ab)`, never in `g_ab`**
  (step 8e). The Lorentzian metrics are not convex in `g_ab` — the angular
  mean of Kerr-Schild `g_ab` on `r = 1.15 M` has Euclidean signature, which
  `fit_tests.jl` asserts — and a least-squares fit is a weighted mean.
  `γ` is held as its offset `γ_ij − δ_ij` in slots 5–10, the contravariant
  shift in 2–4. Convert only through `fit_variables`/`state_from_fit`.
- **The fit's state sampler reads the layer, on purpose** (step 8e): the
  collocation points are on the offset surface, so the interpolation window
  reaches `G h` inside it, and `state_sampler` passes `mask = AllPoints()`
  — the horizon finder's guard is exactly what it switches off. It is safe
  because step 8c's `n_L ≥ 4G` holds `ρ ≤ 0.10 ρ_max` there; a thinner ramp
  would make it read a relaxed state. Ghosts must be filled first, as for
  every interpolation.
- **The shift has no constant term by default, and a moving hole needs
  one** (measured in step 8e). `PLAN.md`'s `β(0) = 0` fits a static hole to
  roundoff and leaves a boosted hole's shift slope `2`–`22` times its own
  scale off, because the boost gives `β^i` an `l = 0` part on the surface;
  `build_fit(…; shift_constant = true)` fits it (`1.7e−5` against
  `1.1e−3`). G5 wants it on.
- **A fit to curvature (`cont = 2`) is not a metric on harmonic Kerr at
  `a = 9/10`** at `h = 5/256`, `m = 4` — `min λ(γ) = −73` at `L = 8`, and
  invalid at every `L` to 16 — while `cont = 1` is from `L = 8`. The sweep
  catches it (`build_fit` throws with the numbers); 8e-ii's initial data for
  that chart cannot be the `cont = 2` fit as planned. And that chart's data
  on the offset surface exceeds `default_bounds`' `K_max = 100/M` (`366/M`),
  so its fit needs bounds of its own before `fit_state` projects into them.
- **The fit's evaluator costs `2.2 µs` a point at `L = 8`** (`cont = 1`;
  `2.9 µs` at `cont = 2`), twenty-three analytic `u_exact`s: the
  `20 × 81 × 2` contraction, not the recurrence (`85 ns`). That is why the
  right-hand side reads a **cache** and never calls it (decided in review,
  step 8e): a fill is `0.4`–`0.7` of a right-hand side, once a chunk.
- **The cache is refilled when the center moves** (step 8e). It holds the
  target on the grid at the fill time plus a slope in time; a moving
  geometry would leave it behind, so a chunk whose tracked center would move
  more than `h/4` is solved in pieces with a refill between, and the fill
  covers the offset surface's bounding sphere plus one cell. A new mesh
  (regrid) gets a new cache, filled at the next chunk's start. Change the
  geometry of a `:fitted` problem and you must refill (`refill_target`).
- **`:fitted` initial data are `C¹` at `r_1`, and it shows** (step 8e): the
  `cont = 1` fit of the analytic solution inside the offset surface
  (decided in review — `cont = 2` is not a metric on harmonic `a = 9/10`)
  has the right value and slope and the wrong curvature, which the compact
  second difference reads as an `O(1)` error at the first evolved points:
  the fixture's masked error is `4.3×` `:damped`'s at `0.15 M`, `1.24×` at
  `1 M`. `fit_initial_depth = n_L h` (the switch at the core surface) removes
  it wherever the analytic solution is regular on the layer. Do not read a
  `:fitted` run's early error as the target's.
- **On harmonic Kerr at `a = 9/10` the fit is not good enough** (measured in
  step 8e): with `m = 4` at `h = 5/256` the ring is `0.02 M` inside `r_1` at
  the equator, the data there are a thousand times the axis's, and a
  degree-`L + 2` polynomial is off between and below its collocation points
  by `10³`–`10⁵` times the analytic second difference off the equator; the
  run ends in its first chunk. `Π̃ = (α/√γ)Π` as the fitted momentum and
  `L = 12` shrink that 20–650× each (host-side probe); 8f decides.
- **A moving layer exports what it holds through its trailing side**
  (measured in step 8). Grid points move outward in depth at the hole's
  speed on the side it leaves, so whatever the layer's inner part holds —
  above all a step in the initial data — reaches the evolved part of the
  layer, where `F` of a jump is `O(jump/h²)`. On G5's chart the step at the
  core surface (`fit_initial_depth = n_L h`) is `25×` the solution, and the
  first chunk's masked L∞ was `334` against the static hole's `2.4`;
  `fit_initial_blend = true` makes it `18`. The target's rate
  (`target_rate`, the lag of a point relaxing toward a moving target) is
  not what matters there: `0.2 %`. Read the `moving` rows' trailing-side
  layer error before suspecting the target.
- **Behind a moving layer, narrow the ramp** (measured in step 8′):
  `trail_ramp = 9/10` makes `ρ` rise where points leave the layer and is
  what removed the trailing side's excess; refilling the target more often
  and evaluating the fit exactly in the kernel changed the fourth digit.
  It is off by default (a static run is unchanged bit for bit); pass it for
  every moving `:fitted` run. The horizon's `J` of a moving spinning hole
  drifts with or without it — read it before trusting a long crossing.
- **`horizon_min_radius` of a boosted hole is a bound, not the radius**
  (found in step 8): the rest frame's smallest radius times `√(1 − v²)`,
  exact only when that radius lies along the boost. A moving seed takes
  its surface's least radius instead, or the first find of a boosted
  spinning hole is refused as a jump.
- **A tracked floor starts at the core surface** (amended in step 8): the
  geometry's spacing is read over the whole layer, and a floor from the
  offset surface let the blocks inside it coarsen and the next geometry
  double its spacing, one level per regrid.
- **A tracked run's first find starts from the seed's shape** (step 8e), not
  a sphere of the mean radius, which on an oblate horizon can lie inside the
  offset surface and be refused by the footprint guard.
- **The depth replaces the radius** (step 8d). On the tracked geometry the
  layer is `0 < d ≤ n_L h` below the offset surface `r_1(n̂) = r_h(n̂) − m h`,
  so no radius alone says whether a point is in it: `in_layer(int, t, x)`
  takes a position and a time (the old `in_layer(int, r)` is gone), the
  kernel's predicates are asked of `interior_point(int, t, x)`, and a band
  about the layer is `layer_mask`/`shell_mask`, never a `ShellMask` built
  from `r_0` and `r_1` — a `FittedInterior` has neither field. A tracked case
  holds a `FittedSpec`, not a layer: `GHProblem`, `state_callback`,
  `level_bounds` and the `Π` post-pass refuse it and ask for `interior =
  fitted_interior(…)`, which `evolve!` builds every chunk.
- **The shape is a series at every layer point** (step 8d): `21 ns` at
  `lmax = 4` and `79 ns` at `8` per point, against `96 ns` for the analytic
  `u_exact` the layer already pays, evaluated only between the bounding
  spheres `r_in − offset − thickness ≤ r < r_out − offset` — the fast paths
  outside them are exact because the surface *is* the series clamped into
  `[r_in, r_out]`. A right-hand side on the fixture is `1.1 %` dearer. The
  real harmonics' slots are `ash_mode_index`'s, `m < 0` the sine, `ỹ^s =
  −√2 Im Y_lm`; step 8e's fit must use the same convention, and
  `real_from_complex` is the only conversion. Harmonic Kerr at `a = 9/10`
  needs `lmax = 12` for a tenth of a cell at `h = 5/256` (Kerr-Schild: `4`).
- **A tracked geometry's `h` is its layer's, and its margin is stated in
  it.** `fitted_interior` reads the coarsest spacing of the blocks the layer
  lives in; on the step-5 fixture the finest level is the cube `|x|_∞ ≤ 5/4`,
  so `m = 8` puts the offset surface outside it and the geometry is refused
  (`18 h = 2.8 M` needed of a horizon of `2 M`) — the suite's tracked runs
  use `m = 10`. `margin_efolds` is what the margin buys along its real path,
  at each block's own spacing.
- **`F` is never evaluated where `w = 0`.** The frozen core holds
  finite, stale data by design — the analytic solution is singular
  inside it — and `F` of that data may be `NaN`; `0 · NaN = NaN`. The
  kernel branches on the core predicate before it touches a stencil.
- **The interior is masked in every norm and in the indicator.**
  Constraint, error and speed kernels and the Löhner indicator write
  zero for `r < r_1`. If a number looks wrong near the hole, check the
  mask before the physics. **Write the masked slot through a branch, not
  as `keep * value`** (step 5): the masked region may hold a `NaN`, and
  `0 · NaN = NaN` — the same trap as the frozen core's, met in the
  monitors.
- **Every path that puts the analytic solution on the grid goes through
  the core rule**, `core_position` — the initial data, the error
  reference, the `:pasted` limiter, the Dirichlet hook *and the
  gauge-source sample*. The last is the one that was forgotten once
  (step 5): `H^a = −Γ^a[g_exact]` is sampled at every owned point
  including the center, where `KerrSchild`'s `k^i` divides by zero, and
  the `NaN` surfaces in the constraint monitors and nowhere else.
- **The mask plumbing.** A kernel takes a `mask`, asks
  `is_evolved(mask, x)`, writes zero where it says no and a `1`/`0`
  indicator into `DIAG_MASK`; `masked_norms` divides by the **evolved**
  volume, not the domain's, so masking a region out does not make the
  number smaller by diluting it with zeros. `AllPoints` is the trivial
  mask, `InteriorMask` is `r ≥ r_1` at that call's `t`, and `ShellMask`
  is a band — which is how the three interior variants are compared over
  "the `G` points outside `r_1`" and how the gauge drift is read at the
  horizon.
- **A reduction that crosses blocks is TreeAMR's `mesh_fold` or
  `mesh_mapreduce`, never a host fold of ours** (from 2026-10-01; the fold
  from 2026-10-23). TreeAMR combines the per-block partials on the host,
  and over MPI (its M7) gathers one partial per rank; a `sum`, `maximum` or
  weighted loop over `block_mapreduce`'s vector here would be a per-rank
  answer under MPI. Several numbers, or one number over a region smaller
  than a block, are **one fold** with a field each, never several
  reductions or a monitor re-run with a mask to select points: a point's
  values do not depend on the mask, and each reduction is a sweep and a
  collective. The fold's `rf` must return `init`'s exact type (a `Sum(1)`
  beside a `Sum(0.0)` is refused), everything it closes over must be
  `isbits`, and a region is asked of the point as `is_evolved(mask, p.x)`
  with the masks the kernels take. The blocks combine pairwise, so an L2
  compared with a stored number is compared to roundoff;
  `threading_tests.jl`'s exact claim names the combination.
- **A checkpoint is written after the analysis row and before the regrid**
  (decided 2026-10-01; TreeHydro writes after it). A restart recomputes the
  flags with the criterion it is given and runs the loop's own
  `regrid_mesh` before its first step, which is what makes a chain of jobs
  the uninterrupted run bit for bit — keep the regrid one code path. Save
  `(U, u)`, never `U` alone: the row's find, fit and indicator scatter into
  `U.work`. The last chunk is written only when `t_end` is a whole number
  of chunks in `T` (`3 · 1/20 ≠ 3/20`). A restart takes the forest from the
  file and refuses one passed beside it, and refuses any changed recipe
  field by name — only `t_end` and the criterion (`Refinement`, `regrid`,
  `buffer`) may change. **Anything added to the loop's carried state goes
  into the run state too** (`driver.jl`'s `state = (; …)` and the restore
  branch), or restarts silently stop being the run;
  `test/checkpoint_tests.jl`'s chains are what notice. TreeIOHDF5, which
  holds the checkpoint functions from TreeAMR 0.2.0, is a hard dependency
  here (TreeHydro left HDF5 to the caller), so they are always there.
- **The `diag` slots are a contiguous-range interface.** `block_mapreduce`
  and `mesh_fold` read an integer or a *contiguous* range of variables and refuse
  anything else — a device cannot be handed an arbitrary index vector cell
  by cell — so `DIAG_CGH` (four) and `DIAG_MOM` (three) are the *first* of
  a run and are indexed as `DIAG_CGH + a - 1`. Adding a slot in the middle
  of either run is how that breaks.
- **A vertex-centered field set stores the shared upper plane, and no
  kernel writes it.** The stored size is `N + 2G + 1` per axis, and the
  point at `N + 2G + 1` belongs to no owned range: `map_blocks!` never
  reaches it and the Dirichlet hook or a neighbour's ghost exchange fills
  it. So a test that reads `fs.work[:, :, :, v, :]` sees a plane of zeros
  that means nothing; use `interiorview(fs, b, v)`. `GHProblem` refuses a
  field set that is not vertex-centered, because `point_position` — the
  one place an owned index becomes a position for the masks and the
  interior profiles — assumes it.
- **The ADM monitor costs 18 s to compile**, and about 4 s for every
  further specialisation of `(G, q, has gauge source, T)`. It is a
  four-dimensional Ricci tensor out of a hundred second derivatives, which
  is the opposite of the right-hand side's streaming order and
  deliberately so: it runs once per chunk, not once per stage. Before
  putting it in a new test row or a new loop, count the specialisations.
  It is *not* in `test/thread_workload.jl` for exactly this reason.
- **A coarse-fine face is expensive in time as well as in order.** On the
  two-level mesh at `p = 6` the ghost fill is **79 %** of a
  right-hand-side evaluation (22 % on a uniform mesh), because a
  tensor-product prolongation reads `6³ = 216` coarse points per fine
  ghost point. A study on a refined mesh costs roughly four times what the
  same study costs uniform, which is why `interface_tests.jl` runs an
  eighth of a crossing on `N = 8, 10, 12` and not more.
- **A spherical frozen core cannot hide Kerr's singularity in the
  harmonic chart at `a = 0.9`** (found in step 5, and it is the
  proof-of-concept case). Both Kerr charts are singular on the equatorial
  **disk** of coordinate radius `|a|`, not at a point, so the core needs
  `r_0 > |a|`; the placement bound needs `r_0 < r_h,min`, and harmonic
  Kerr's `r_h,min = √(M²−a²) = 0.436` is smaller than `0.9`. A ball fits
  only where `a < M/√2`. `check_interior_radii` refuses it by name
  (`singular_radius`); `SINGULARITY_HANDLING.md`'s "The interior's questions"
  has the two
  ways out.
  Kerr-Schild at `a = 0.9` runs, because `r₊ = 1.436 > 0.9`.
- **The three interior radii are asserted at every regrid**: `r_1`
  inside the horizon by `m` spacings of the blocks containing it
  (`m = 8` by default, never below `G + 1`); `r_1 − r_0` at least
  `2(G + 1)` such spacings; the refinement's level floor covering the
  horizon. If one fires, raise the floor, shrink the layer or widen the
  floor's shell — do not remove the assertion. For `a = 0.9` in harmonic
  coordinates the horizon's smallest coordinate radius is about
  `0.44 M`, and these bounds are what set the finest spacing.
- **The refinement is an error indicator with three geometric
  corrections, and all four matter.** TreeWave's Löhner indicator with
  a *global* noise floor (the local floor refines `1e−16` tails), the
  box keyed on `coarsen_tol` (keying it on `refine_tol` silently
  disables the travelling margin), the mask inside `r_1` (stale core
  data has steep meaningless differences), the level *floor* around the
  horizon (it is what guarantees the interior's resolution) and the
  level *ceiling* at the outer boundary (the Dirichlet mismatch is a
  kink the indicator scores). Ghosts must be filled before flagging.
  Convergence claims are made on a hierarchy frozen at `t = 0`, not on
  the adapting mesh.
- **The reference amplitude is one number for all ten components, and
  that is not an approximation** (step 6). In the gauge wave's chart six
  components are *identically zero*, so a per-component reference is
  exactly zero, the floor with it, and that component's dust scores
  `τ ≈ 1` over the whole domain — TreeWave's blast-wave trap in a chart
  instead of in initial data.
- **The travelling margin is dilated inside `refine_flags`, and
  `regrid!` is then given `buffer = 0`** (step 6). TreeAMR's
  `buffered_flags` promotes every leaf a dilated box reaches whatever the
  application said about it, so a margin applied afterwards refines the
  blocks the ceiling just capped — measured, 12 of 64 boundary root
  blocks. Dilate, *then* clamp.
- **The floor and the ceiling together are a statement about the box.**
  `block_level_bounds` throws when a block is both in the horizon shell
  and within the boundary margin, because the region that must be
  resolved has met the region that must stay coarse. That is what a box
  of half-width `5/2 M` around a hole whose horizon is at `2 M` does:
  the refinement's reference configuration uses `5 M`, and step 5's
  fixture — which has no refinement — keeps `5/2`. The floor's *level* is
  derived from the interior's own radii, so a `maxlevel_cap` below it is
  refused rather than discovered later as a `check_interior_radii`
  failure.
- **2:1 balance overrides the ceiling, and that is TreeAMR's invariant.**
  A deep enough hierarchy in a small enough box pushes refinement out to
  the boundary through balance alone. If the boundary blocks are not at
  the coarsest level, count the levels before suspecting the ceiling.
- **`ρ_max` is `4/M` by default, read from the hole, and bounded by RK4's
  stability** (amended in step 8c′). `evolve!` with no rate keyword relaxes
  at `default_relaxation_rate(case) = 4/hole_mass(case.background)` in every
  chunk; `ρ_max_factor = 1` is the grid rate `ρ_max · dt = 1`, the default
  until 2026-09-23 and what step 5's numbers need; `ρ_max_fixed` is any
  other rate, and the two are refused together. RK4 is stable to about
  `2.8/dt` on the negative real axis and a fixed rate — the default's
  included — above `1/dt` is refused, which on the suite's holes is eleven
  to twenty-eight times away (`4/M · dt = 0.036–0.092`). A run that blows up in
  the layer after raising `ρ_max` has found the integrator, not the
  physics. **At `4/M` the layer is held to `τ/ρ_max`, not `τ · dt`**: the
  `:damped` residual is twenty times the grid rate's (`1.42` against
  `0.062` on the fixture) and saturates only after about `2/ρ_max = M/2`,
  so a claim about the residual's saturation needs a run that long — which
  is why `driver_tests.jl`'s variants read it at `1/2 M` and the shell at
  `1/10 M`.
- **The horizon finder reads only the evolved region, and it says so by
  throwing.** The interpolation window is `q + 2` points per axis, `G` of
  them on each side of the query's cell, so a query *outside* `r_1` can
  still read *inside* it. The guard is on the footprint and not on the
  query point, and it is exact — bit for bit against the norms' mask,
  since 2026-09-26: the guard is `UnevolvedRegion(mask)`, TreeAMR's
  `interpolate` flags a query whose stencil reaches it at the positions
  `coordinates` gives, the region asks the norms' own `is_evolved`, and
  `gh_interpolate` throws for the first flagged query. Its `stencil_hits`
  overrides (the per-axis nearest point, then enumeration on a tracked
  surface) are exact by construction and `tracking_tests.jl` checks them
  against enumeration; do not swap in TreeAMR's `Ellipsoid`, which rounds
  differently from `is_evolved` and measured no cheaper. A find that fails
  because of it is recorded in the run's record as
  `horizon_success = false` with the message in `horizon_note`, never
  thrown out of `evolve!`: the horizon is a diagnostic. If a find fails,
  read `r_1 + (q+2)h/2` against the horizon's smallest coordinate radius
  before suspecting the finder — and remember that the fast flow's
  *transient* dips below its seed, which is how the harmonic chart at
  `h = 5/64` fails and the same chart at `5/128` does not. On a tracked
  geometry (step 8d) the guard is a `ShapeMask`'s and exact by enumeration:
  a footprint between the offset surface's two bounding spheres has every
  one of its `(q+2)³` points classified, so it refuses exactly what the
  norms mask.
- **Ghosts must be filled before anything is interpolated**, with that
  call's hook. `find_gh_horizon` scatters and fills; `gh_adm_provider` and
  `gh_interpolate` do not, and an unfilled halo makes the metric garbage
  exactly at the block faces — which the fast flow then walks into the
  layer to escape.
- **The Korzyński spin is what a find costs**, not the interpolation:
  `16×`, `45×` and `105×` a right-hand-side evaluation at
  `N_ah = 12, 16, 20`, against `1.7×`, `2.7×` and `3.6×` for the find
  itself — one batch of 496 interpolated `ADMVars` is 0.14 ms at four
  threads through TreeAMR's `interpolate` of the 16 variables the ADM data
  reads (`ADM_VARS`; `Π_tt` and `Π_ti` are `NaN`, and `horizon_tests.jl`
  says why that is safe) — 0.16 ms with all 20, 0.23 ms through the
  stopgap on the day of the port, 0.26 ms in step 7. Lower the
  cadence or pass `spin = false` before lowering `N_ah`. Its `unif_tol` is
  `1e-8` here and not the library's `1e-13`, because interpolated data has
  a residual floor of its own (`2.2e−5` at `h = 5/64`): the tighter
  tolerance reports `success = false` on every find and returns the same
  `J`.
- **The analysis quantities are part of the deliverable**: masked
  constraint norms, the interior residual, horizon location, area,
  `M_irr`, `J`, `M_ch`, the refinement centroid, mesh statistics, at
  every chunk boundary, appended to the run's time series. A run that
  finishes without them is not a result, and a test that asserts only
  "it ran" is not a test.
- **`Float64` on the H200 is the requirement; `Float32` is a
  nice-to-have.** Record a `Float32` failure, do not fix it at
  `Float64`'s expense. The type-generic discipline stays anyway: a
  decimal literal in a `T` expression is a leak — `T(1//2)`, not `0.5`;
  `oftype(x, 2)` inside closures.
- **The RHS kernel is written in streaming order, and stays that way**
  (amended 2026-10-05).
  - **The head, once per point:** the state, `∂_i h`, the coefficient set,
    `∂ₜh` (stored at once), the source and the two divergences.
  - **Then a run-time loop over the ten Π components,** each formed from its own
    stencils, combined and stored. Nothing of `F` stays live longer than it
    takes to store it.
  - **The source comes before the Π components.** Computed last, as until
    2026-10-05, it was live with the ten accumulators and spilled.

  Never build an `SVector` of all derivatives: that is about 140 `Float64`
  values, over a GPU thread's 255 registers, and it spills. GHAccel's ten-field
  kernel fit only as fully scalarised generated code. Spills show as `STL`/`LDL`
  in the SASS (`bench/sass_stats.jl`).
- **On a device a closure is a call** (found 2026-10-05).
  - **Why:** KernelAbstractions' default `CUDABackend()` does not force inlining,
    and TreeAMR launches with it whatever backend a field set was built with.
  - **What becomes a call** in a kernel's path: an `ntuple(Val(n)) do … end`, a
    StaticArrays generator `SVector(f(i) for i in 1:3)`, or a `@generated`
    method whose body lacks an `:inline` meta. Its `SVector` arguments then
    travel through the stack: 8 KB of local memory a thread, and most of the
    right-hand side's 8.5 ns a point before 2026-10-05.
  - **How to write kernel-path code instead:** `Base.Cartesian.@ntuple`/`@nexprs`
    with literal counts, small tuples spelled out, and a generated body returning
    `Expr(:block, Expr(:meta, :inline), …)`.
  - **The symptom** is a `CALL` to `julia_…` in the SASS
    (`bench/rhs_lab.jl mode=baseline` writes it, `bench/sass_stats.jl` counts it).
    On the CPU the same closures cost 7× (`bench/rhs_lab_cpu.jl`).
- **Over MPI the host analysis is rank 0's, and the other ranks serve it**
  (added 2026-10-19). Inside `on_root(…; serve = (p, u))` — the find, the
  track's update, the fits — rank 0 announces each collective step by a
  broadcast request and the other ranks take it with no points of their own.
  The only two steps are `fill_state!` and `interpolate_state`. **A new
  collective inside the analysis must go through one of them, or a new
  request kind in `serve_requests`**: a `fill_ghosts!`, a `mesh_mapreduce`
  or an `interpolate` called directly there runs on rank 0 alone and hangs
  the job. And never assign a captured variable inside the analysis: what
  it returns is adopted on every rank after it, which is how the track, the
  geometry and the fits stay the same bits everywhere.
- **A block index is this rank's, a leaf index is the forest's** (TreeAMR's
  M7). `nblocks(fs)`, `blockkey(fs, b)`, `block_origins`, `block_mapreduce`
  and `firing_boxes` are per local block; `forest.leaves`, `nleaves` and
  `locate_point` are global. `forest.leaves[b]` with a local `b` is right on
  one rank and wrong on every other — the refinement centroid had it until
  2026-10-19. A flag vector `buffered_flags` takes is the whole forest's.
- **Multi-GPU runs go through HPC-X with `--bind-to none`, not MPICH_jll**
  (measured 2026-10-19, `CODE.md`, "Running distributed"). Open MPI's
  `mpiexec` binds each rank to one core by default, and the rank's Julia
  threads — rank 0's find, spin and fit above all — share it: 3.8× slower
  analysis, invisible in the step. MPICH_jll's shared-memory transport moves
  a fill's 35–38 MB prolongation messages so slowly that 4 H200s step slower
  than one (0.86×, against HPC-X's 1.89×). `bench/symmetry_mpi_gpu_env.sh`
  builds the HPC-X environment (with `OpenMPI_jll` pointed at its libraries)
  and the `bench/symmetry_mpi_*.sh` jobs pass the flag. What is left of the
  fill's poor scaling is TreeAMR's: each prolongation level is one rank's
  work, and the levels run in sequence.
- **A decision from a rank's own clock or file system must be agreed**
  (`agree_any`, `agree_refusal`), and a number combined on the host from
  per-block values must be reduced over the ranks (`rank_reduce`), or the
  ranks part ways at the next collective and the job hangs. What the
  checkpoint's `data` holds must be the same on every rank, or TreeIOHDF5
  refuses the save.
- **Hooks depend on time.** `dirichlet(case, t)` is built at each call.
  It goes to `fill_ghosts!` inside the RHS, to `regrid!`, and to
  `adapt_to_initial_data!`, each with that call's `t`. Forgetting the
  second is the bug that arrives one chunk late; passing a stale `t` is
  the bug that arrives as a boundary reflection.
- **A moving non-harmonic background with a sampled source is refused.**
  A boosted Kerr-Schild has a time-dependent gauge source that a per-chunk
  sample cannot represent. Give it the algebraic source instead,
  `gauge_source = :algebraic` — `KerrSchildSource` with the hole's lab
  velocity `hole_velocity(bg)`, which moves with the hole, and which
  `kerr_schild_case(; v, x0)` builds (amended 2026-10-20) — or use
  `boost(Harmonic(M, a), v)`, which is harmonic (`H ≡ 0`) — a boost
  preserves the harmonic condition. The refusal's message says this; do
  not weaken it.
- **`boost(m, v)` moves the hole at `−v`** (found in step 8d, fixed in step
  8e). `SpacetimeMetrics` evaluates the boosted metric at `Λᵀx`, so
  `boost(KerrSchild(1, 0), (0.3, 0, 0))` is singular at `x = −0.3 t`.
  `hole_velocity(background)` is the one place the sign is written; `GHCase`
  derives the case's velocity from it when the keyword is left unset and
  refuses a keyword that disagrees. Do not pass `velocity = v` for a boost
  by `v` — leave it unset. **Never test a metric *on* its singular point**
  (amended 2026-09-26): whether rounding leaves `r` at `2e−17` or at zero —
  a huge `g_tt` on Apple silicon, `NaN` on the x86 CI runners — is the
  platform's, not the physics'. `interior_tests.jl` asks for
  `g_tt = −1 + 2γ²M/δ` at a transverse offset `δ` instead.
- **A moving `:damped` core relaxes in ADM variables, never linearly in `g`**
  (2026-10-20). With `du = 0` the core keeps the solution about `c(0)` and
  releases it on its trailing side; the moving Kerr-Schild hole died at
  `r_0/v = 3.1 M`. So it relaxes toward `u_exact` at the core rule's point
  (`core_follows`), `(α², β, γ)` through `core_relaxation_rate` and `Π`
  linearly. The straight line in `g_ab` between the targets on the two sides
  of the sphere `r_0` passes through a Euclidean metric (`g_tt = 2.2` at
  `r_0 = 5/8`), and that version ended in its first chunk. A hole at rest
  keeps `du = 0` bit for bit.
- **A tracked `:damped` core is still frozen** (2026-10-20): `core_follows`
  is false for a `FittedInterior`, so `variant = :damped` on the tracked
  geometry — the screen before a new `:fitted` depth — dies at
  `r_core/v` on a moving hole, as step 8's `:frozen` did. Screen a moving
  depth for less than that, or with `:fitted`.
- **A relaxation toward a moving target needs the target's rate** (measured
  2026-10-20, built 2026-10-21). Without `(1 − w) ∂_t u_exact` the analytic
  layer trailed the moving hole by `|∂_t u|/ρ_max` — a residual of `6.5` at
  `4/M` that no `h` reduced, the error just outside the horizon `28–44×` the
  resting hole's at `1/16` — and the core, relaxing without a rate, released
  its own lag into the layer on its trailing side. Both carry their rate now,
  behind `core_follows`; `4/M` is the moving hole's setting again. Anything
  else that relaxes toward something that moves — a new variant, a new
  target — needs the same term in every region it relaxes in, the core
  included. The rate is a centered difference (`δ = ε^{1/3} r_0`), so a moving
  hole's quadrant and box agree to `ε^{2/3}`, not to roundoff:
  `boosted_tests.jl` says so, and so must any new mirror test of a moving
  hole.
- **Check `r_1` against the contracted horizon** (2026-10-20). A hole moving
  at `v` has its horizon at `2√(1 − v²)` along the boost (`1.960` at `1/5`),
  and `check_interior_radii` measures the margin from that
  (`horizon_min_radius`), so the octant's `r_1 = 3/2`, `7.4` cells inside it
  at `1/16`, is refused at `m = 8`; `23/16` is `8.4` cells inside. Choose the
  radii for the contracted horizon before submitting a run.
- **The quadrant needs a hole on the `x` axis, moving along it, with
  `a = 0`** (2026-10-20). It mirrors `y = 0` and `z = 0`, and `hole_case`
  refuses any other hole — a spin along `z` keeps only the `z` mirror. Its
  root brick is `(2r, r, r)` and its cells must be cubes. As with the octant,
  a refined quadrant is not the refined box's restriction; compare the two on
  uniform meshes only.
- **The rotating quadrant needs a hole on the `z` axis, moving along it and
  spinning about it** (2026-10-21). Its symmetry is the quarter turn about
  `z`, and a spin or a velocity across `z` would turn with it; `hole_case`
  refuses any other hole, including a rotated spinning one. Its root brick is
  `(r, r, 2r)`, and `kerr_schild_case(; axis = 3)` puts the hole on `z`.
- **A moving hole's symmetric domain is the box only to the target rate's
  amplified roundoff** (2026-10-21). The moving layer's and core's rate is a
  centered difference over `δ = ε^{1/3} r_0` of the analytic background, which
  is four-fold symmetric (or mirror-symmetric) only to its own roundoff, `56 ε`
  under a quarter turn — so the box's rate is asymmetric by about that over
  `δ`, while the quadrant's ghosts are the exact image. The difference grows
  linearly with the steps (`2.5·10⁻¹²` after one, `1.3·10⁻¹⁰` after sixteen in
  `boosted_tests.jl`), and the claim bounds it by `nsteps · dt` times the
  rate's measured asymmetry. A hole at rest takes the old branches and agrees
  to `0.4 ε`. Bound any new symmetry test of a moving hole the same way, and do
  not read a growing difference there as a seam bug before checking the box's
  own asymmetry.
- **The ring rule limits `:fitted` to `h ≤ 1/32` at `a = 1/2`** (2026-10-21).
  The `:fitted` core surface is `m + n_L` cells below the horizon, and on the
  equator (`√(r₊² + a²) = 1.932`) it must clear the ring `r = a`: at `m = 20`,
  `n_L = 18` it is at `0.74` at `1/32` and `0.35 < a` at `1/24`; the rules'
  least `16 + 18` cells would put it at `0.52`, on the ring. Spin raises the
  coarsest spacing `:fitted` can run before it raises `:damped`'s, whose `r_0`
  is in `M`.
- **`hz_J` is a component on the rotating quadrant and a magnitude elsewhere**
  (2026-10-21). `test/octant_runs.jl` writes `J n̂_z` there (`n̂` the finder's
  spin axis) and the Korzyński `|J|` on every other domain, which cannot go
  negative or show a flipped axis; `records.csv`'s `J` is the magnitude
  everywhere. On the `a = 1/2` runs the two agree to the CSV's ten digits.
  Say which one a comparison reads.
- **`1/48` on the moving hole's capsule does not fit one H200** (2026-10-21).
  The capsule of radii `32, 16, 8, 4` along `8 M` of travel is 856 blocks,
  `94.7 M` points at `N = 48`; linear in the points from `1/32` and `1/40`
  that is about `141 GB` for `:damped` and `171 GB` for `:fitted`, against
  `144 GB`. A finer or longer run needs several GPUs (`mpi=true`) or a mesh
  that follows the hole.
- **On a GPU the analysis row, not the evolution, is the expensive part**
  (measured 2026-10-22). On one H200 at `28 M` points the evolution is
  `64 s/M`. The trailing/leading split (`test/octant_runs.jl split=1`, the
  default on the quadrants) adds `109 s/M` at `chunk=1/2`, and the rest of
  the row `33 s` a chunk: a pair of the ADM monitor and the error kernel is
  about `4.5 s`, some thirty right-hand sides, and the row runs seven of
  them, the split twelve more. The finder is under a second. Use `chunk=1
  split=0` for production, and the split only to diagnose a moving layer's
  sides. Every number before 2026-10-22 that reads `71–175 ns` per point and
  step was at `chunk=1/2` with the split. Before adding a monitor to the
  row, price it in ADM-monitor evaluations.
- **At `a = 9/10` the layer depends on the speed** (2026-10-22). The
  contracted pole at `v = 1/2` is `1.244`, so `r_1` must be `9/8` there at
  `1/72` (`5/4` is outside it). With `r_0 = 19/20` that layer grows inside
  the horizon at rest from `10 M`, by `×382`, `×36` and `×3.4` over
  `[10, 20] M` at `1/72`, `1/96` and `1/128`, while at `v = 1/2` it holds.
  The resting `r_1 = 5/4` holds `v = 0` and `1/5`. A run that changes speed,
  or a resting control for a `v = 1/2` row, has no layer that is known to
  work; compare the moving `9/8` rows with the resting `5/4` ones outside
  the horizon, where the two resting layers agree to three digits by `20 M`.
- **A half turn cannot hold a hole whose spin and velocity are not parallel**
  (2026-10-24). A rotation keeps only the vectors along its axis, and a mirror
  keeps the polar vectors in its plane and the axial ones along its normal. So
  a hole spinning about `z` and moving along `x` has the `z` mirror and no
  turn: TreeAMR's half-turn seam (M14) about `x` reverses its spin, and one
  about `z` reverses its velocity. Its domain is the bitant, at twice the
  quadrants' points. The half turn is for a pair of holes, each the other's
  image.
- **Do not merge two inlined copies of a source through an early return**
  (found in step B1). Julia 1.13.2 on aarch64 compiled the tracked source —
  `if steady; return closed_form(u₀); end; …; return closed_form(u(t)) + rate`
  — inlined into a function that also took ForwardDiff Jacobians, into code
  whose `∂_x H` was off by `10⁻³` of the source: at `-O2` and `-O3`, not at
  `-O1`, not out of line, not on 1.14.0-alpha1, unchanged by
  `--check-bounds=yes`. Evaluate the closed form once and branch on its
  inputs, and choose between whole sources on the host
  (`effective_source(src, t)`, which is how the roll-on's two ends reach the
  kernels). The symptom is an error of the size of a term in one row of
  `∂_a H_b` that comes and goes with unrelated code around the call;
  `test/gauge_source_tests.jl`'s closed-form testset calls the sources in that
  shape.
- **A gauge source is checked at `t = 0`, and must be exact there**
  (step B1). `GHCase` evaluates `effective_source(src, 0)` against the
  background's `−Γ_a`, so a `RolledGaugeSource` from the hole's own source with
  `t_0 ≥ 0` passes and a bare `DampedHarmonicSource` on a hole is refused, as is
  a composite with an acceleration (its `∂_t H` is not the background's). A
  test that needs such a source in a kernel builds a case without the check
  (`bench/gauge_source_cost.jl`'s `carrying`) or evaluates it pointwise.
- **The damped harmonic gauge is harmonic far away, and the Dirichlet data are
  Kerr-Schild** (step B1). `μ ∝ L²` with `L ≈ 2M/r`, so after the roll-on the
  faces' data violate `C_a` by Kerr-Schild's `2M/r²` (`5·10⁻⁴` at `64 M`),
  whatever `σ_r` is. Read the shells about the hole, not the global `C_a`, of a
  `gauge=dh` run, and see `CODE.md`, "What step B1 built and measured".
- **Under a gauge that moves the hole, measure about the hole** (step B1).
  `test/octant_runs.jl`'s shells and split are about the analytic `c(t)` by
  default; under the damped harmonic gauge the moving hole fell `0.49 M` behind
  it, and a shell about `c(t)` read the horizon's inside (`ℋ` `5×` too large).
  Pass `shell_center=track` (a `:fitted` hole), and remember that the gauge also
  grows the horizon's coordinate radius (`2.00` → `2.06`), so `[2, 2.25)`
  straddles it.
- **The lanes' `log` and `exp` are LLVM's, the scalar ones Julia's** (step
  B1). The damped harmonic gauge's kernel on lanes is the scalar one to `19 ε`
  of the terms, not the `1–2 ε` of the other sources.
- **Two kinds of track, and only one predicts along a quadratic** (step B2).
  `HorizonTrack.quadratic` is a `TrackedSpec`'s; a `FittedSpec`'s track is step
  8d's line, and every `:fitted` run is bit for bit what it was because its
  position is still `center_at(track_center(tr), t)`. Read a track's center
  through `track_position(tr, t)` or `track_center(tr, t)`, never through
  `c_find + v_est (t − t_find)` written out: on a quadratic track that is the
  new prediction alone, without the blend that keeps the center `C¹` at a find.
- **A binary's seed must know its acceleration** (step B2). The center-jump
  check refuses a find more than `G h/2` off the prediction, and a seed with a
  position and a velocity only predicts a `d = 3` orbit `2.4 G h/4` off at its
  first find a chunk later — refused. Pass `seed_track(…; acceleration)` from
  the orbit, or a short `max_interval` until four finds exist.
- **The tracked target's row runs on the sphere at the chunk's end** (step
  B2), built from the track before the row's find, and the checkpoint's
  `track_chunk` holds that track for it; a fitted run's holds the chunk's
  start. A change to when the row's sphere is built has to change both, or a
  restart is no longer the run (`checkpoint_tests.jl`'s tracked chain notices).
- **Negate a velocity as `zero(v) − v`, not `−v`, where bits matter** (step
  B2). `−v` turns a zero component into `−0.0`, which `boost` carries into the
  signs of its matrix's zero entries; the tracked target built from the seed is
  the case's background bit for bit only with `0 − v`, as the case's own
  `(−v, 0, 0)` is.
- **`Val`s once per chunk.** `G`, `q`, "has gauge source", "has
  dissipation", — from step 5 — the interior *variant* (`:none`,
  `:damped`, `:pasted`, `:frozen`, which is "has interior" and *which* in
  one parameter) and — from 2026-10-05 — the SIMD width `W` are `Val`
  parameters built in `GHProblem`'s constructor.
  Building them per evaluation recompiles or dispatches dynamically on
  every RK stage. The price is paid at compile time instead: a test row
  at a new `q` is a new kernel, which is most of what
  `evolution_tests.jl`'s 56 s are, and each interior variant is another
  one. The interior itself is rebuilt per chunk anyway — the grid-rate
  option `ρ_max_factor` follows `1/dt`, and the default `4/M` is checked
  against it — and `with_interior` shares the field sets and the sampled
  gauge source rather than rebuilding the problem, which would re-sample
  `H_a`.
- **On the CPU the kernel's algebra runs on SIMD lanes, so it must stay generic
  in its number type** (added 2026-10-05). Everything `gh_rhs_head` and
  `gh_rhs_pi` reach — `metric_quantities`, the lean source, `metric_divergences`,
  the gauge sources, the damping and dissipation profiles' *results* — is evaluated
  with `T = Vec{W,T}`, which is not a `Number`. So: **no branch on a computed value**
  (`if x > 0`, `?:` — a lane condition has a value per lane; use `_select`, and
  `_anynonzero` for "is it zero"); **no scalar times a static array** (`a * M`;
  use `_scale`, which is also closure-free for a device); **no literal converted to
  `T`** (`SMatrix{4,4,T}(-1, 0, …)`; use `one(T)`, `zero(T)`, `T(x)`). A violation
  is a `MethodError` or a `TypeError` at the first CPU evaluation, which
  `test/simd_tests.jl` makes on every branch. Two consequences to know: a lane's
  `sqrt` is the instruction, so a degenerate metric in the evolved region gives a
  `NaN` (read by the record's `finite` and the next chunk's speed check) where the
  scalar code throws a `DomainError` — `simd_width = 1` restores the throw; and the
  lanes are the scalar kernel to **roundoff, not bits** (StaticArrays' `muladd`s are
  fused into FMAs by context): bitwise on the gauge wave, 1.0–2.0 eps of the terms
  on a hole. Bit-identity across thread counts is unaffected — a block is one thread's.
- **KernelAbstractions refuses a `return` statement anywhere in a kernel
  body**, closures included. That is why the streaming right-hand side
  lives in `gh_rhs_at_point`, a plain `@inline` function: the frozen
  core's branch has to be an `if` *around* the whole computation, and it
  cannot be an early exit.
- **A `@generated` method must not convert to the caller's type.** Its
  generator may only call methods that existed when the generated function
  was *defined*, and this package is precompiled long before a driver loads
  MultiFloats: a generator that wrote `T(w)` for a rational `w` worked at
  `Float64` and `Float32` and threw "the applicable method may be too new"
  at `Float32x2`. Emit the exact integers and let the call site divide.
  The symptom only appears when this package is loaded *before* the type's
  package, which is what `runtests.jl` does — so a test file that loads
  MultiFloats above `TreeGeneralizedHarmonic` hides it.
- **The stencil weights are for unit spacing, and the dissipation's
  factor of `h` is a single one.** `derivative_weights` is divided by
  `h^m` at the call site; `dissipation_weights` is multiplied by `ε/h_d`,
  because the `h_d^{2r−1}` of `CODE.md`'s formula cancels all but one
  power against `(D₊D₋)^r`'s own `h_d^{−2r}`. The weights already carry
  `(−1)^{r+1}`, the `2^{−2r}` and therefore the *damping* sign: Nyquist is
  damped at exactly `ε/h_d`. A run that blows up faster the larger `ε_KO`
  is has that sign backwards, and `test/stencils_tests.jl` asserts it as
  an exact rational eigenvalue.
- **The recipe near a hole is `ε_KO ≈ 0.5`, `γ0 ≈ 1/M`.** GHSO2
  measured both as requirements with the horizon in the domain
  (`notes/methods-ghso2.md`); a run that blows up at the sonic surface
  with `ε_KO = 0` is not a bug in the stencils.
- **The interface-order rule.** This system takes second derivatives,
  so prolongation order `p = q + 2` (order 6 at `q = 4`), or the global
  rate drops by one. Never give `ops` a default that hides it; TreeAMR
  refuses `G` below `p/2 − 1` and this package's `G = q/2 + 1` clears it.
- **Don't name a keyword `maxlevel`.** It shadows TreeAMR's exported
  `maxlevel(forest)` inside the function body. Use `maxlevel_cap`.
- **A callback must capture no `Type` and no host array.** Backgrounds,
  the interior profiles, the damping profile, the indicator's thresholds
  and floors become kernel arguments; they are `isbits` structs and
  tuples. The center is a function of `t`, never a mutated field.
- **Two spellings of one expression are not bit-identical — and neither
  are two call sites of one function.** The same arithmetic written twice
  — `gh_node_source` and the block it copies, `gh_fluxes` and
  `gh_node_rhs`'s fluxes — disagrees in the last place on this machine,
  because the compiler fuses a multiply and an add in one inlining context
  and not in the other. So does one body reached two ways:
  `metric_derivatives`'s wrapper and its coefficient-set method disagree
  on 3 of 12 points at `Float64` (measured in step 1). Compare any of
  these to roundoff against the scale that produced the number, never with
  `isequal`. The bit-identity that *is* an invariant is the same compiled
  code, at the same call site, at a different thread count, which is what
  `test/threading_tests.jl` will assert. Two tests claimed more and were
  relaxed to roundoff on 2026-09-27, when x86-64 stopped giving it:
  `interior_tests.jl`'s `:damped` against `:none` outside `r_1` (two kernel
  specialisations, 120 eps on `znver3`) and `moving_tests.jl`'s fill
  against a host `state_tuple` (under coverage on Linux).
- **Never thread anything a TreeAMR callback can reach**, and never
  accumulate into shared state in a loop of your own: bit-identity
  across thread counts is the invariant, and `test/threading_tests.jl`
  is the only thing that will report a violation.
- **A parallel loop of this package's own is TreeAMR's
  `threaded_foreach`/`threaded_chunks`, never `Threads.@threads` or
  `@spawn`, and a kernel is a `map_blocks!`** (from TreeAMR's owner-based
  threading, 2026-09-25). TreeAMR runs every per-block pass on the
  block's owner thread; a launch on whichever thread is free moves the
  blocks between cores and cost TreeAMR's RHS `2.4×` on 64 cores. The two
  helpers are unexported, so `prerequisite_tests.jl` names the one this
  package calls (`threadchunks`, for the integrator's partition;
  `threaded_foreach` went with the stopgap interpolator's batch on
  2026-09-26, and the batch is TreeAMR's `interpolate`, one launch over
  points), beside the `Region` extension points the footprint guard
  reaches (`inside`, `stencil_hits`, `stencil_position`). RK4's stage
  arithmetic is owner-based too from 2026-09-26 (IMEXRungeKutta by
  `state_partition`; it was OrdinaryDiffEq's serial broadcast, 12–14 % of a
  step at 64 threads — `CODE.md`, "Time integration").
- **`Base` is not generic even though the mesh is.** MultiFloats defines
  no `rem`, no conversion to `Integer`, no `Float64(::Float32x2)`. Use
  `wrap` / `ceilint` / `floorint` / `tofloat64` from `precision.jl`.
- **Measured numbers go into `CODE.md`**, beside the prediction they
  confirm or correct, so a regression shows up as a changed number and
  not as a test that merely still passes — those of the black hole's
  interior into `SINGULARITY_HANDLING.md`, and a recommendation that changes into
  `CODE.md`'s table (2026-10-08).
- **A refined octant is not the refined full box** (found 2026-10-02):
  vertex centering puts a refinement cube's plane `x = −R` on the fine level
  and `x = +R` on the coarse one, so the box's discretization is not
  mirror-symmetric there. Compare an octant with its box on uniform meshes
  only (`test/reflection_tests.jl` does).
- **An octant's corner is in the core, and its stale data is not of definite
  parity** (2026-10-02): the core rule maps the origin to a point of the
  sphere `r_0` off the walls, so it holds `h_tx ≠ 0` on a wall. Never project
  or perturb the core; `add_noise!(; exclude)` skips it.
- **A Symmetry node failure requeues a job from scratch** (2026-10-03): Slurm
  restarts it with no checkpoint unless the run has one, over its own log and
  appending to its CSV. Give every run longer than an hour `checkpoint=` and a
  `walltime=`. And inside a `while read` loop `ssh` eats the loop's stdin —
  use `ssh -n`. **A requeue can hit a run that has finished** (2026-10-22, a
  Slurm controller outage). `a9v2h72r` (job 580825) had reached `20 M`. The
  requeued job restarted from its checkpoint at `t = 20`, `evolve!` refused
  ("there is nothing left to run"), the batch script marked the SimWatch
  status "failed", and the job's log was overwritten. The CSV and
  `records.csv` were complete. After an outage, read the CSV's last `t`
  before trusting a "failed" status. Correct the status by hand
  (`status = "finished"` and a `message` saying what happened), and keep the
  peak memory and wall time elsewhere, since the log is gone.
- **The plain superposition of two Kerr-Schild holes is not a 3+1 metric
  inside the horizons** (found in step B3). On the side of each hole that faces
  its companion the two shifts `2H l_i` cancel and `α² = −g_tt + β_i β^i < 0`: for
  an equal pair of total mass one at `d = 8` out to `0.52 M` from the hole's
  center, through the core sphere and half the layer. `binary_case` turns each
  hole's companions off within `4 M_B` of it (`companion = 4`, the default); with
  `companion = 0` the initial data's `sqrt` throws a `DomainError`. Check `α²` on
  the core sphere before trusting another superposition.
- **A hole's horizon is not centred on the hole in superposed data** (step B3).
  The companion displaces it toward itself (`0.022 M` at `d = 8`), and a track
  that reads the found origin as the hole's center moves its target off the
  data's hole, or turns the offset into a velocity through the blend
  (`σ′ δ/Δt`). `evolve!` measures each hole's offset on the initial data
  (`reseed_holes!`) and corrects every find by it; the record's `holes[A]` has
  `origin` (the find's) and `track_center` (the hole's). Compare separations of
  the tracks, not of the found origins.
- **A case of one hole is not run through the path of several** (step B3).
  `holes = (spec,)` is the single-hole case, bit for bit — its analytic source,
  its record — and the vector of tracks, the per-hole rows and the track-fed
  source start at `K = 2` (`case.interior isa HolesSpec`). A test of the K-hole
  path needs two holes.
- **`hole_mass(case.background)` throws on a superposition** (step B3, by
  design: `PLAN.md`'s units edge). Read each hole's `HoleSpec.mass`; the default
  rate of several holes is `4/M_A` of the lightest (`_case_mass`).
- **The composite kernel is the nearer hole's kernel bit for bit, and must
  stay so** (step B3): `acting_interior` returns one hole's whole interior by
  `ifelse`, and every branch of `gh_rhs_point!` then runs on it (`ia`). A
  composition that mixes two holes' numbers at a point — `min`, `max` of
  profiles — would break `holes_tests.jl`'s point-by-point claim and the
  single-hole arithmetic every hole was validated with; before merger the
  layers are disjoint and nothing needs mixing.
- **The composite source fed by the tracks runs away on a pair** (found in step
  B3). The source carries each hole at its track's velocity, the coordinates
  follow, and the track follows a horizon centroid that the data's junk moves: on
  the head-on pair every `source_feed = :tracks` row ended within `1 M` on the
  track's own refusals (`1/48` and `1/24`, either momentum), the predictions
  growing to `0.4–0.9` cells first. `source_feed = :seed` (the holes' Newtonian
  quadratics) holds it, and so does `:controlled` (step B3's review, to `6 M`).
  It is the default `:tracks` because it is what Erik decided; a run that must
  reach its end passes `:controlled` (with each spec's `control_time`) or `:seed`.
  The raw feed's harm is its lead — the track's quadratic carries the last finds'
  acceleration into the source — which amplified the junk's swing until a
  prediction failed the center-jump check; do not feed a source the track's
  acceleration.
- **A controlled source line starts on the data's gauge, not on the track's
  seed** (step B3's review). A source on another line than the data's violates
  `C_a` at `t = 0` by the difference (`10⁴×` for a seed `0.05` off), and a line and
  target started on step B2's recovery seed carried the hole at the seed's speed.
  `seed_source_lines(case)` reads the case's own source; keep it so.
- **The track–line loop has no reference but the track** (step B3's review): its
  gain is one at zero frequency, so a displacement the line once caused stays —
  `0.014` cells on step B2's hole after a recovery at `τ = 10 M`. And on the pair
  the holes' coordinate motion is the gauge's, the same within `10–25 %` with the
  line on the Newtonian seeds or on the tracks: read no fall off a coordinate
  separation in this gauge, and no physics off the line's position.
- **No model in the source line, and no integral started at the data's
  acceleration** (step B3's review, second round). The PD loop's steady error is
  `(c̈_tr − a_m) τ²`; in this gauge the holes' coordinate acceleration is `−0.007`
  to `−0.020` over `1–3.5 M`, then crosses `a_N` and is still rising at `6 M` — the
  gauge's transient is not over by then — so a Newtonian model added `40–100 %` to
  the forcing and put the line `4.2` cells off the holes at `6 M`
  (`τ = 10 M_A`), and a PID integral started at `a₀/ω³` held the same model for
  its first `τ` (`3.1` cells). The defaults are the PID loop without the model,
  its integral from zero; a PD line without a model lags a steady acceleration by
  `c̈ τ²`. Before reading a line's offset as the loop's, integrate the law on the
  row's own track (it reproduces every row to `0.05` cells): the offset is the
  response to the forcing's history, not its steady value.
- **A controlled source line that leaves the data's line costs at the outer
  boundary** (step B3's review, third look, from the screens' CSVs). The source
  rides the line everywhere while the Dirichlet data are the analytic hole's, so
  after a recovery that moved the line (`b3fc24`, `b3fe24`, `b3fpr24`) `ℋ` and
  `C_a` on the coarsest level and in the boundary band are about `10³×` the
  analytic source's (`ℋ` on `L0` `800–4100×` from `2 M`), scaling with the line's
  excursion, while the shell just outside the horizon looks fine. Read `L0` and
  `_bnd`, not just the shells. B4's boundary has to follow the lines — Dirichlet
  data from the holes on their lines, the source blended back to the data's gauge
  far out, or a constraint-preserving boundary (options, not decisions).
- **A pre-empted `h200preq` job can end `FAILED`, not requeued** (step B2,
  2026-10-26). Two rows (jobs 583003, 583004) got `SIGTERM` after `8–9 min`,
  their `srun` exited `143`, and the jobs ended `FAILED 15:0` with nothing
  requeued. Step B1's batch scripts resubmit themselves on `143` for this
  reason; a script without that branch has to be resubmitted by hand, and the
  row then continues from its last checkpoint. Read `sacct` for a job that
  vanished from `squeue` before its `40 M`.

## Conventions

Match TreeAMR's, since the four packages are read together:

- 4-space indent, wrap at about 80–90 columns.
- `return` on the last line of any non-trivial function.
- `ntuple(d -> f(d), Val(3))` rather than comprehensions in
  kernel-adjacent code; states are `SVector{10}`, tensors `SMatrix{4,4}`.
- Unicode in mathematical contexts (`α`, `β`, `γ`, `Π`, `∂ₜ`, `Σ`).
- Keyword-heavy driver signatures with no default for anything the caller
  must think about; `T` as a leading positional argument defaulting to
  `Float64`, `backend` a keyword defaulting to `CPU()`.
- `ArgumentError`s say *why*, not just what.
- Docstrings are prose-first: what it is, then why, pointing at `CODE.md`.
- **Testset names are claims**, each opening with a comment naming the
  failure mode it guards.
- Spec-first: when the implementation shows `CODE.md` was wrong or
  incomplete, amend it and say so in it — "(amended in step N)",
  "(measured in step N)" — rather than diverging silently.
- `CLAUDE.md` and `CODE.md` open with a **table of contents** (added
  2026-10-05) of every heading below the title. A heading added or renamed
  goes into it, linked by GitHub's anchor: lowercase, backticks and
  punctuation dropped (an en dash too), spaces to hyphens.

## Repository facts

- **The remote is `git@github.com:eschnett/TreeGeneralizedHarmonic.jl.git`**,
  and the rule from the siblings applies: work on a branch, and do not
  push, open a pull request, or merge to `main` without being asked. Each
  step lands on `main` only after review.
- `TODO.md`, when it appears, is Erik's personal to-do list. **Do not
  modify it.** It is gitignored.
- `CODE.md`, `SINGULARITY_HANDLING.md`, `PLAN.md`, `README.md`, `notes/`, `src/`,
  `test/`, `.github/` and this file are committed. `Manifest.toml` files,
  `bin/output/` and `docs/build/` are gitignored — no `Manifest.toml` is
  tracked, which is what makes the clean-checkout check above mean
  something.
- Sibling checkouts: `~/src/jl/TreeAMR` (the mesh; read its `CLAUDE.md`
  and `CODE.md` for the API and its sharp edges), `~/src/jl/TreeWave`
  and `~/src/jl/TreeHydro` (the other applications; copy the *patterns*
  of `precision.jl`, `device.jl`, `refinement.jl`, `bin/backend.jl`, the
  viewers and the thread workload — do not depend on them),
  `~/src/jl/SpacetimeMetrics` (the backgrounds),
  `~/src/jl/ApparentHorizonFinder` and `KorzynskiSpin` (the horizon
  diagnostics, from step 7). `~/src/jl/GeneralizedHarmonicSecondOrder2`
  is the *origin* of `notes/`; the text to cite is in `notes/`.
