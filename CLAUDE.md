# Working notes for Claude in TreeGeneralizedHarmonic.jl

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. This file is only about mechanics.

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
  stopgap this package carries — point interpolation for the horizon
  finder — is marked as such in `CODE.md` and goes upstream when
  TreeAMR grows it.
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
(decided 2026-09-23)**, which runs at `h = 5/256` (2472 blocks, a node-hour
per six `M`) and not at `5/128`; harmonic `a = 9/10` waits with its price
written down in `CODE.md`'s "Open questions" (`h ≲ 5/1024` on the equator,
23 000 blocks, `38 h` a `50 M` run, and a fit that holds 45° first). Step 8,
the moving hole, is next — and it inherits step 8f's findings that
`:fitted` holds a boosted hole at `4/M` while the analytic `:damped` layer
needs `ρ_max ≳ 20/M` (its frozen core is released on the trailing side
after about `M` at `v = 0.3`), and that a moving hole's step is sized for
the speed it will have.**
`CODE.md` is complete and reviewed three times (2026-09-16): the expanded
form of the momentum equation, three dimensions only, a pointwise damping
layer instead of excision, a single boosted spinning black hole as the
proof-of-concept target, RK4 from OrdinaryDiffEq, the analysis quantities
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
`locate_block` and `interpolate`/`interpolate_grad` (the *stopgap* point
interpolator, `find_leaf` then a tensor-product Lagrange window of
`q + 2` points, batched and threaded over a host array, with the
footprint guard that refuses a query reading inside `r_1`),
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
`gh_stage_limiter!` — the state's third writer, a `solve` keyword beside
`step_limiter` — `BoundsAccounting`, the validity monitor
(`validity_rows`) and `evolved_nonfinite`, which is what `max_speed_of`
and the record's `finite` now read. `diag` has 22 slots; the record grew
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
`Project.toml` carries the `[sources]` pins and CI is in place — but
**the CI matrix is temporarily reduced** (2026-09-19): Julia 1.11 and code
coverage are both dropped, each with the removed lines and the reason in a
comment at its site in `CI.yml`. 1.11 fails only
`pointwise_tests.jl:480` (176 bytes where zero is claimed); coverage
computes *wrong numbers* on GitHub's runners while the identical
instrumented suite passes on Symmetry. Both are owed back. There is
no `Manifest.toml` (deliberately, and permanently: it is what makes the
clean-checkout check below mean something), no `bin/`, and there is now a
remote — `git@github.com:eschnett/TreeGeneralizedHarmonic.jl.git`.

The suite is **4616 assertions in 14m21** at one thread and **10m41** at
four on the development machine after step 8e (load 6–13): the 38 new
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
tree with no `Manifest.toml` resolves all four pinned packages from
GitHub (TreeAMR from the registry between 2026-09-21 and 2026-09-23), and
passes. From 2026-09-21 it is a real check —
every source is public, so it works anonymously, which is what CI does:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

It runs from a git worktree as happily as from the checkout, which is how
the per-step agents work; the `[sources]` pins mean every worktree
resolves the same branches.

The thread workload runs on its own, which is how a digest mismatch is
bisected — `test/threading_tests.jl` starts exactly this in a subprocess:

```bash
julia --project=. -t 4 test/thread_workload.jl
```

The black-hole runs that are too long for the suite — the default margin
`m = 8` at `q = 4`, the `t = 50 M` run of all three interior variants,
the two harmonic charts, and from step 6 the indicator's calibration and
its adaptive run — are a **script**, run by hand, with its numbers
recorded in `CODE.md` under "Measured results" (added in step 5). It takes
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
JULIA_EXCLUSIVE option` — the environment, not the code. The `symmetry-hpc`
skill has the rest of the cluster's mechanics.

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

- **All four dependencies are pinned to GitHub `main`, not to the local
  checkouts — TreeAMR included again** (Erik's `Project.toml` of
  2026-09-23, which dropped the explanatory comment block; amended in step
  8f). `TreeAMR`, `SpacetimeMetrics`, `ApparentHorizonFinder` and
  `KorzynskiSpin` are what `Project.toml`'s `[sources]` entries resolve, so
  `~/src/jl/…` is *not* what the tests see; an unpushed change there is
  invisible here, and the local SpacetimeMetrics checkout has been behind
  `main` before. Read what Pkg installed under `~/.julia/packages/` when in
  doubt about an API. Say so rather than editing a checkout and assuming
  the tests see it. A pushed change to TreeAMR's `main` is visible at the
  next resolve without a release (from 2026-09-21 to 2026-09-23 it came from
  the General registry at `0.1.1`, where only a release published it);
  `test/prerequisite_tests.jl` is what notices when a moving branch drops a
  name.
  **`KorzynskiSpin`'s repository became public on 2026-09-21**, so its URL
  is plain `https`, the read-only deploy key and the `ssh-agent` step and
  `JULIA_PKG_USE_CLI_GIT` are all gone from `CI.yml`, and an anonymous
  clean checkout and a fork's pull request both resolve it — the
  clean-checkout check below finally proves what it claims. Do not vendor
  a package and do not add a local-path source.
- **`[sources]` is why the Julia floor is 1.11**, and it cannot be lowered
  yet. The section was introduced in 1.11, so the floor drops only when
  every entry goes: that needs `KorzynskiSpin` registered in General (it
  is not registered at all) and `ApparentHorizonFinder` **2.1** released
  (General carries `2.0.0`, and `[compat]` here asks for `2.1`).
  `SpacetimeMetrics` is registered at the bound this package asks for, so
  its pin is a choice rather than a necessity.
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
- **KernelAbstractions refuses a `return` in a kernel** — anywhere in the
  body, closures included, which is what `ntuple(Val(10)) do v … end` is.
  End the block with the value instead. The error names the kernel and
  arrives at precompilation, so it is cheap; it is here because the
  package's own convention asks for an explicit `return` everywhere else.
- **The RHS never mutates `u`, and the state has exactly three writers**
  (amended in step 8b). The interior layer is a term of the right-hand
  side, `du = w F(u) − ρ (u − u_exact)`, so the integrator's arithmetic is
  the first writer; the `:pasted` paste of the ball `r < r_1`, from RK4's
  `step_limiter!`, is the second; step 8b's range projection
  (`src/bounds.jl`), from the `stage_limiter!` on every stage vector and
  once after the initial fill and every regrid, is the third. Both limiters
  are **`solve` keywords**, not `RK4(; …)` arguments — the constructor form
  is deprecated and will be silently unread. Do not add a fourth writer.
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
- **The `diag` slots are a contiguous-range interface.** `block_mapreduce`
  reduces an integer or a *contiguous* range of variables and refuses
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
  (`singular_radius`); `CODE.md`'s "Open questions" has the two ways out.
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
  query point, and it is exact (the footprint is a lattice, so the nearest
  of its points to the center is the per-axis nearest). A find that fails
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
  `interpolate` do not, and an unfilled halo makes the metric garbage
  exactly at the block faces — which the fast flow then walks into the
  layer to escape.
- **The Korzyński spin is what a find costs**, not the interpolation:
  `16×`, `45×` and `105×` a right-hand-side evaluation at
  `N_ah = 12, 16, 20`, against `1.7×`, `2.7×` and `3.6×` for the find
  itself — one batch of 496 interpolated `ADMVars` is 0.26 ms. Lower the
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
- **The RHS kernel is written in streaming order, and stays that way.**
  Coefficients and `∂_i h` once per point; then per component, stencils
  formed and consumed immediately; then the source. Never build an
  `SVector` of all derivatives — that is about 140 `Float64` values,
  over a GPU thread's 255 registers, and it spills. GHAccel's ten-field
  kernel fit only as fully scalarised generated code; spills show as
  `ld.local` in the PTX. GPU kernel *efficiency* beyond this order is a
  research project, not a milestone; G6 measures, it does not tune.
- **Hooks depend on time.** `dirichlet(case, t)` is built at each call.
  It goes to `fill_ghosts!` inside the RHS, to `regrid!`, and to
  `adapt_to_initial_data!`, each with that call's `t`. Forgetting the
  second is the bug that arrives one chunk late; passing a stale `t` is
  the bug that arrives as a boundary reflection.
- **A moving non-harmonic background is refused.** A boosted
  Kerr-Schild has a time-dependent gauge source that a per-chunk sample
  cannot represent. Use `boost(Harmonic(M, a), v)`, which is harmonic
  (`H ≡ 0`) — a boost preserves the harmonic condition. The refusal's
  message says this; do not weaken it.
- **`boost(m, v)` moves the hole at `−v`** (found in step 8d, fixed in step
  8e). `SpacetimeMetrics` evaluates the boosted metric at `Λᵀx`, so
  `boost(KerrSchild(1, 0), (0.3, 0, 0))` is singular at `x = −0.3 t`.
  `hole_velocity(background)` is the one place the sign is written; `GHCase`
  derives the case's velocity from it when the keyword is left unset and
  refuses a keyword that disagrees. Do not pass `velocity = v` for a boost
  by `v` — leave it unset.
- **`Val`s once per chunk.** `G`, `q`, "has gauge source", "has
  dissipation" and — from step 5 — the interior *variant* (`:none`,
  `:damped`, `:pasted`, `:frozen`, which is "has interior" and *which* in
  one parameter) are `Val` parameters built in `GHProblem`'s constructor.
  Building them per evaluation recompiles or dispatches dynamically on
  every RK stage. The price is paid at compile time instead: a test row
  at a new `q` is a new kernel, which is most of what
  `evolution_tests.jl`'s 56 s are, and each interior variant is another
  one. The interior itself is rebuilt per chunk anyway — the grid-rate
  option `ρ_max_factor` follows `1/dt`, and the default `4/M` is checked
  against it — and `with_interior` shares the field sets and the sampled
  gauge source rather than rebuilding the problem, which would re-sample
  `H_a`.
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
  `test/threading_tests.jl` will assert.
- **Never thread anything a TreeAMR callback can reach**, and never
  accumulate into shared state in a loop of your own: bit-identity
  across thread counts is the invariant, and `test/threading_tests.jl`
  is the only thing that will report a violation.
- **`Base` is not generic even though the mesh is.** MultiFloats defines
  no `rem`, no conversion to `Integer`, no `Float64(::Float32x2)`. Use
  `wrap` / `ceilint` / `floorint` / `tofloat64` from `precision.jl`.
- **Measured numbers go into `CODE.md`**, beside the prediction they
  confirm or correct, so a regression shows up as a changed number and
  not as a test that merely still passes.

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

## Repository facts

- **The remote is `git@github.com:eschnett/TreeGeneralizedHarmonic.jl.git`**,
  and the rule from the siblings applies: work on a branch, and do not
  push, open a pull request, or merge to `main` without being asked. Each
  step lands on `main` only after review.
- `TODO.md`, when it appears, is Erik's personal to-do list. **Do not
  modify it.** It is gitignored.
- `CODE.md`, `PLAN.md`, `README.md`, `notes/`, `src/`, `test/`,
  `.github/` and this file are committed. `Manifest.toml` files,
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
