# TreeGeneralizedHarmonic implementation plan

For the sessions that implement TreeGeneralizedHarmonic, one step at a
time. The **design** lives in `CODE.md` — read it first, in full; it is
authoritative, and this file is only the work breakdown: what each step
changes, what it must not change, and what it must measure and record.
`CLAUDE.md` has the mechanics and the traps. Delete this file when the
last milestone is marked *(Done.)* in `CODE.md`.

**Steps 0–7, 8a–8f and 8c′ are done; the generic interior is *(Done.)*
and step 8g is not needed (`:fitted` reaches `50 M` on 8f's first row).
Step 8, the moving hole at `a = 7/10` (decided 2026-09-23), ran on
2026-09-24 and is merged with G5 *not* marked done: seven of its ten items
hold, and the moving layer exports error through its trailing side on G5's
chart (`CODE.md`, "Open questions"). Step 8′, the bounded round on the
trailing side, ran 2026-09-24/25 and is merged: the side-dependent ramp
(`trail_ramp = 9/10`) removes the trailing side's export, and what is left
— a uniform growth and a drift of the moving spinning hole's `J` — keeps
G5 open, with the recommendation "more interior work, the spin drift
first, before excision" (proposed in step 8′) awaiting Erik's decision.
Step 9 (G6), which does not depend on it, is running (started
2026-09-25).**

**Excision is reopened (2026-10-05, Erik's decision): steps X1, X2a, X2b
and X3 below, static holes first, as per-step agents on the integration
branch `claude/excision-singularity-handling-30feec`.** They supersede step
8g's brief. **X1 is done and merged (2026-10-05): go for the static
Kerr-Schild `a = 0` hole with per-axis closures and `:msn` dissipation,
`ε_KO > 0` required, spinning holes not covered. X2a is done and merged
(the stencil provider, no bit changed). X2b is done and merged (the
`:excised` variant on both geometries, every other run bit for bit). X3 is
done and merged (2026-10-06): feasible — the excised static Kerr-Schild
`a = 0` hole at `r_E = M/2`, without the blend, matches `:damped` outside
the horizon to three digits at `h = 1/32`, beats `:fitted`, and shows no
gauge drift to `50 M`. The static round is complete. Erik's decision of
2026-10-06: the static spinning hole at `a = 3/5`, with `main` merged
first — steps X4–X7 below. X5 is done and merged (2026-10-06): go for
`a = 3/5` with the hybrid rule (`hybrid-adv`). X4 is done and merged
(2026-10-06): the branch is on `main`'s spill-free kernel and the rotating
octant, and the merged suite passes (6952 at one thread, 6960 at four). X6
is done and merged (2026-10-06): the frame-dragged rule and the symmetric
zone mixed derivative; the `a = 3/5` smoke runs on the CPU and an H200. X7
is done and merged (2026-10-08): the excised static `a = 3/5` hole at
`r_E = 4/5` is the `:damped` layer outside the horizon from `h = 1/32` on (to
four digits at `1/48`), its `J` drifts as the layer's does (`1.03×`), and a
shallow surface (`r_E = 17/15`) fails at a lego corner at the pole — cure
found on a scratch copy, not built. The spinning round is complete; the next
(proposed in step X7: the polar corner, then `a = 9/10` on the tracked
surface, then moving holes) is Erik's decision.**

The steps map onto `CODE.md`'s milestones G0–G6, split so that every
step ends in a green test suite and a `CODE.md` update, and so that each
is a brief a single agent with a fresh context can carry. The order is
the dependency order.

## Ground rules, every step

- Read `CLAUDE.md` first, then the `CODE.md` sections the step names.
  The inherited documents are in `notes/`; cite them, do not go looking
  in the sibling checkouts for text.
- **Work on a branch** named `claude/step-N-<slug>`, off `main`. Commit
  there in the style of TreeAMR's history (implement → measure →
  record, measured numbers in the commit body). **Do not merge into
  `main` and do not push**; report the branch and its commits. There is
  no remote yet; when there is one, the rule stands.
- **TreeAMR and SpacetimeMetrics are pinned to their GitHub `main`**
  through `[sources]`. The checkouts at `~/src/jl/TreeAMR` and
  `~/src/jl/SpacetimeMetrics` are *not* what the tests see. If a step
  turns out to need something from either, stop, describe exactly what
  and why, and report — do not edit those checkouts and carry on.
- **Generic in `T` and in the backend from the first line.** Every
  driver takes `T` as a leading positional argument (default `Float64`)
  and `backend` as a keyword (default `CPU()`); no floating-point literal
  in an expression where `T` is in play (`T(1//2)`, `oftype(x, 2)`);
  every callback captures `isbits` only; per-block metadata a kernel
  reads goes through `to_backend`. Steps 4 and 9 add the *tests and
  measurements* of these properties, not the properties — TreeWave
  records that retrofitting them was a rewrite. `Float64` on the H200 is
  the device requirement; `Float32` anywhere is desirable and recorded
  either way.
- **Every test is 3D and therefore small**: `N = 8`, two to eight
  roots, one refinement level, short times; GHSO2's rule of thumb is
  under 30 seconds per test file.
- Before and after each step: the full suite, once at the default thread
  count and once with `julia_args = ["--threads=4"]`.
- Spec-first: when the implementation shows `CODE.md` wrong or
  incomplete, amend it and say so in it ("amended in step N"), and
  replace each **(predicted)** the step measures with the measured
  number, marked **(measured in step N)**. Never loosen an existing
  assertion to get green; report the failure instead.
- Testset names are claims; each opens with a comment naming the
  failure mode it guards. Convergence rates, constraint norms and mesh
  statistics are asserted as numbers with tolerances.
- One step at a time; the next starts from a green suite on `main`.

## Running a step as an agent

Each step below is written as one brief for one agent with a fresh
context. Nothing in it assumes the agent saw the conversation that
produced the design; everything it needs is in `CODE.md`, `CLAUDE.md`,
`notes/` and the step's own text. The pattern that keeps Erik's rule —
each step lands on `main` only after review — is:

- An orchestrating session (or Erik by hand) starts **one
  implementation agent per step**, in its own git worktree, with the
  brief below. Steps depend on each other, so at most one runs at a
  time. Opus is sufficient for implementation against a spec this
  explicit; the orchestrator's job is review, not repair.
- The agent works on its branch, does not merge or push, and ends with
  the report below. It does not ask questions mid-step; where it would
  have asked, it decides, says so in the report, and marks the decision
  in `CODE.md` as **(proposed in step N)**.
- The reviewer checks out the branch, runs the suite at one and four
  threads, reads the `CODE.md` diff for the recorded numbers and
  amendments, walks the step's acceptance list item by item against the
  report, and only then merges to `main`. A failed item goes back to the
  same agent with the reviewer's finding, not to a new one, and not to
  the reviewer's own hands.

**The brief** (paste as the agent's prompt, replacing `N`):

    You are implementing step N of PLAN.md in the Julia package at
    <path to this repository>. Read CLAUDE.md first, then PLAN.md's
    "Ground rules", "Sharp edges" and step N, then the CODE.md sections
    step N names, then the notes/ files it cites. Work on the branch
    claude/step-N-<slug> in this worktree; commit in TreeAMR's style
    with measured numbers in the commit body; do not merge, push, or
    touch TODO.md or notes/. Follow every ground rule. Do all of step N.
    If part of it is blocked — something needed from TreeAMR or
    SpacetimeMetrics, a device you do not have — finish everything
    else and say exactly what is blocked and why. Never loosen an
    assertion to get green; report the failure instead. When you would
    ask a question, decide, mark the decision "(proposed in step N)" in
    CODE.md, and list it in the report. End with the report PLAN.md
    specifies under "Running a step as an agent".

**The report** the agent ends with, in this order:

1. the branch name and the commits on it, one line each;
2. the step's acceptance list, every item marked *done*, *measured*
   (with the number), or *blocked* (with the reason and what would
   unblock it);
3. the test suite's summary lines at one and at four threads, verbatim;
4. every change made to `CODE.md`, quoted, with the marker it carries;
5. decisions taken where the step was silent, each marked (proposed);
6. anything the step revealed that the next step should know.

**What the reviewer does not do:** rewrite the plan mid-step; accept
"the tests pass" without the numbers; start a step before the previous
one is on `main`; fix the agent's branch directly.

**Where an agent cannot go.** The H200 measurements of step 9 need
Symmetry: the agent writes the scripts and the batch job, runs the
CPU rows locally (and Metal if the machine has it), and leaves the H200
rows to be run there; the step is done when the local rows are in
`CODE.md` and the scripts are in place, and the H200 rows are recorded
when they exist.

## Sharp edges to know before starting

The full list is in `CLAUDE.md`; these are the ones that bite while
writing kernels.

- **Two derivative index conventions.** `SpacetimeMetrics.dmetric`
  returns `dg[a, b, c] = ∂_c g_ab`; GHSO2's `pointwise.jl`, ported here
  from `notes/pointwise-ghso2.jl`, uses `dg[a, b, c] = ∂_a g_bc`.
  Convert once, in `initialdata.jl`, and nowhere else.
- **Off by `G`.** A kernel launched by `map_blocks!` gets the *owned*
  index and adds `G` to reach the working array; `coordinates` takes
  *stored* indices. Every stencil reaches `±(q/2 + 1)` from the stored
  index and never past `1 … N + 2G + 1`.
- **The RHS never mutates `u`.** The interior layer is a *term* of the
  right-hand side, `du = w F(u) − ρ (u − u_exact)`; only the `:pasted`
  variant writes the state, and only from RK4's
  `step_limiter!(u, integrator, p, t)`. Nothing else writes the state
  outside the integrator.
- **`F` is not evaluated where `w = 0`.** The frozen core holds finite
  but arbitrary data on which `F` may be `NaN`, and `0 · NaN = NaN`. The
  kernel branches on the core predicate before touching a stencil.
- **The interior is masked in every norm and in the indicator.**
  Constraint, error and speed kernels and the Löhner indicator write
  zero for `r < r_1`. A number that looks wrong near the hole is a
  missing mask before it is a bug in the physics.
- **The interior knows nothing about blocks.** `w` and `ρ` are
  functions of `r = |x − c(t)|` and of nothing else; if a change makes
  them depend on a block index, a level or a ghost width, it is wrong.
  The refinement's level floor around the horizon is what makes the
  interior's resolution requirements hold; the interior does not ask
  for it.
- **The indicator needs ghosts.** Löhner's stencil reaches one point
  past the block face; the driver fills ghosts with the current hook
  before flagging. The box is keyed on `coarsen_tol`, the floor is a
  *global* amplitude, the marks are TreeWave's four.
- **Hooks depend on time.** `dirichlet(case, t)` is a `CellBoundary`
  closure built at each call with the current `t`, and it goes to
  `fill_ghosts!` (inside the RHS), to `regrid!` and to
  `adapt_to_initial_data!`, each with that call's time. Forgetting the
  second is the bug that arrives one chunk late.
- **`Val`s are built once per chunk** in `GHProblem`: `G`, `q`, whether
  there is a gauge source, whether there is an interior. Building them
  per evaluation recompiles or dispatches dynamically on every stage.
- **`ρ_max` is a physical rate, `4/M` by default** (decided 2026-09-23,
  step 8c′; `M` from `hole_mass`), read once per run; `ρ_max_factor`
  selects the former grid rate `factor/dt`, which RK4's stability bounds
  at about `2.8/dt` on the negative real axis, and a fixed rate above
  `1/dt` is refused.
- **Don't name a keyword `maxlevel`** (it shadows TreeAMR's
  `maxlevel(forest)`); use `maxlevel_cap`.
- **`RK4(; step_limiter!)`** exists in `OrdinaryDiffEqLowOrderRK` with
  the signature `limiter!(u, integrator, p, t)`; `u` is in state layout,
  `statearray(u, U)` gives the block view. Use `adaptive = false`, a
  `dt` from `gh_dt`, and `save_everystep = false`.

## Step 0 — Scaffolding (G0)

`CODE.md`: "File layout", milestone G0.

Changes:

- `Project.toml`: deps `TreeAMR`, `SpacetimeMetrics`,
  `KernelAbstractions`, `StaticArrays`, `OrdinaryDiffEqLowOrderRK`,
  `SciMLBase`; compat bounds after TreeWave's (`KernelAbstractions =
  "0.9.42, 1"`, `SciMLBase = "3.50.1"`, `StaticArrays = "1"`,
  `TreeAMR = "0.1.0"`, `SpacetimeMetrics = "1.6"`); `julia = "1.11"`;
  `[sources]` entries pinning TreeAMR and SpacetimeMetrics to their
  GitHub `main`, with TreeWave's comment on why they exist.
- `.gitignore` in TreeWave's image: `*~`, `*.swp`, `.DS_Store`,
  `/docs/build/`, `Manifest.toml`, `/bin/output/`, `TODO.md`.
- `src/TreeGeneralizedHarmonic.jl`: the module shell with its docstring
  and the includes that exist, replacing the template's `hello` and
  `domath`; `src/precision.jl` and `src/device.jl` ported from TreeWave
  (`wrap`, `ceilint`, `floorint`, `tofloat64`; `to_backend`, `hostcopy`
  with TreeHydro's `hostcopy!` split, written against the M8 `FieldSet`
  signature).
- `test/Project.toml` (`Test`, `TreeAMR`, `SpacetimeMetrics`,
  `KernelAbstractions`, `StaticArrays`, `MultiFloats`,
  `OrdinaryDiffEqLowOrderRK`, `SciMLBase`), `test/runtests.jl`,
  `test/precision_tests.jl` (the Base-bridge claims at `Float64`,
  `Float32`, `Float32x2`), `test/prerequisite_tests.jl`: the pinned
  TreeAMR exports every name this package calls (`FieldSet`,
  `GhostSchedule`, `map_blocks!`, `block_mapreduce`, `CellBoundary`,
  `AllVariables`, `firing_boxes`, `regrid!`, `adapt_to_initial_data!`,
  `coordinates`, `find_leaf`, …); and **a `SpacetimeMetrics` metric
  runs as a kernel argument**: `fill_by_coordinates!(AllVariables(x ->
  packed dmetric(KerrSchild(1, 0), (0, x...))), fs)` on `CPU()` fills a
  field set with bit-for-bit the values of a host loop.
- `.github/workflows/CI.yml` after TreeWave's (Julia 1.11 and release,
  Linux and macOS, one 4-thread entry), `.github/dependabot.yml`.
- `README.md`: a short blurb with the status; `notes/` is already in
  place and is committed with the documents.

Accept: `Pkg.test()` green; a clean archive (`git archive HEAD | tar -x
-C /tmp/clean`) instantiates from the pins and passes; `CLAUDE.md`'s
"Current state" and "Commands" describe what now exists. Mark G0
*(Done.)* in `CODE.md`.

## Step 1 — Pointwise algebra (G1a)

`CODE.md`: "The equations" (all of it), "Gauge and constraint damping";
`notes/pointwise-ghso2.jl` is the source of the port.

Changes: `src/pointwise.jl` — GHSO2's `NC`, `_sym4`, `_pack10`,
`pack_g`, `pack_sym`, `metric_quantities`, `adm_from_metric`,
`gauge_constraint_at_node`, `adm_vars_from_state` ported as they are;
GHSO2's flux-form `gh_node_rhs` kept **for the tests only**; new:
`metric_derivatives(h, ∂h) -> (∂_iα, ∂_iβ^j, ∂_i(α√γγ^{jk}))` in closed
form, and `gh_node_rhs_expanded(h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl, γ0, γ2) ->
(∂_t h, ∂_t Π)` implementing `(EXPANDED)`, written as scalarised
straight-line code (by hand or generated) since it is what the
streaming kernel of step 3 calls. States and gradients are
`SVector{10}`; everything `isbits`, no literals.

Accept: on every background in the table, at random points, in
`Float64` and `Float32`: ADM extraction against `adm_decompose`; the
offset identities at `‖h‖ ~ 1e−13` to full relative precision; the flux
identity `∂_tΠ − ∂_iF^i = msrc` with `∂_iF^i` by central differences of
the analytic `F^i`; `metric_derivatives` against a ForwardDiff dual pass
through `metric_quantities`; `gh_node_rhs_expanded` equal to the flux
form's `∂_tΠ` assembled from analytic derivatives, to roundoff; `C_a`
and `Z_ab` zero on exact data (with sampled `H` for non-harmonic
backgrounds); all of it callable from a trivial kernel on `CPU()`.

## Step 2 — Stencils (G1b)

`CODE.md`: "Finite-difference stencils", "Kreiss–Oliger dissipation".

Changes: `src/stencils.jl` — `derivative_weights(::Val{q}, ::Val{m})`
for `m = 1, 2` at even `q`, `dissipation_weights(::Val{r})`, all built
in `Rational` and converted once to `T`; the mixed derivative as the
product of two first-derivative weight vectors; a host-side
`apply_stencil` for the tests.

Accept: the weights reproduce the textbook tables at `q = 2, 4, 6, 8`;
each operator is exact on polynomials of degree `≤ q` (first
derivative) and `≤ q + 1` (second, even `q`) and *not* one degree
higher; the dissipation operator applied to a sine mode has the
damping sign and vanishes on polynomials of degree `< 2r`; the weights
are bit-identical at `Float64` between a direct construction and the
`Rational` route. Mark G1 *(Done.)*.

## Step 3 — The right-hand side and the gauge wave (G2)

`CODE.md`: "Field sets and layout", "One right-hand-side evaluation",
"The time step", "Gauge and constraint damping", "Boundaries:
Periodic", "Initial data and backgrounds", "Time integration".
TreeAMR's `test/wave.jl` and TreeWave's `evolution.jl` are the worked
examples for the RHS pattern.

Changes: `src/evolution.jl` — `GHProblem` (the field sets, the
schedule, per-block origins and spacings on the backend, the case, the
`Val`s), the fused `gh_rhs_kernel!` in **streaming order** — the
coefficient set and `∂_i h` once per point, then a loop over the ten
components forming `∂_i Π_ab`, `∂_i∂_j h_ab` and the dissipation on the
fly and contracting them immediately into two accumulators, then the
source from `h`, `∂_i h` and the coefficients; scalarised straight-line
algebra, no `SVector` of all derivatives ever formed (`du` in state
layout; no interior yet), `gh_rhs!` (scatter → ghost fill with the
Dirichlet hook or `nothing` → kernel), the speed kernel into `diag`,
`max_speed`, `gh_dt`; `src/initialdata.jl` — the background table,
`state_callback(case, t)` as an `AllVariables` closure (no hole yet),
the index conversion from `dmetric`; `src/gauge.jl` — `isharmonic`,
`isstatic`, `sample_gauge_source!` as a `fill_by_coordinates!` of
`gauge_source_grad`, the refusal of a moving non-harmonic background;
`src/boundaries.jl` — `dirichlet(case, t)`; the periodic uniform forest
builder; `convergence_rate` after TreeWave. Cases: Minkowski with
noise, the gauge wave, shifted Minkowski.

Accept: Minkowski is stationary to roundoff (`du` exactly zero) and
shifted Minkowski to truncation, converging at order `q`; the gauge
wave (`A = 0.05`) at `q = 2, 4, 6` on `N = 8`, roots `2 … 8`, one
crossing, converges at order `q` in the volume-weighted L2 and L∞
norms; white noise of amplitude `1e−8` on flat space stays bounded over
a thousand steps at `ε_KO = 0.5`, and the growth at `ε_KO = 0` is
recorded; two evaluations at the same `u` give identical `du` and `u`
is untouched; the RHS throughput per owned point on the CPU recorded in
`CODE.md`. Mark G2 *(Done.)*.

## Step 4 — Coarse-fine faces, constraints, threads (G3)

`CODE.md`: "The interface-order rule", "Analysis quantities" (the two
constraint monitors), "Precision, threads, devices".

Changes: the two-level forest (TreeAMR's `wave_forest`, ported);
`src/constraints.jl` — the GH constraint kernel, the ADM constraint
kernel (`∂_tt g` from the reduced equation), norms through
`block_mapreduce` in block order (the mask argument present, all-true
for now); `test/thread_workload.jl` (a gauge wave with one regrid,
digests per chunk, self-contained) and `test/threading_tests.jl` (a
subprocess at the other count, compared character for character);
`test/type_tests.jl` (steps 3 and 4 at `Float32` on `CPU()`, the
pointwise algebra at `Float32x2`).

Accept: the interface-order table on the gauge wave at `q = 4`:
predicted L2 rates 3 and 4 at prolongation orders 4 and 6, the same at
restriction orders 2 and 4 and at `ε_KO = 0` and `0.5`, the unrefined
control at 4 — recorded, replacing the prediction; both constraint
monitors converge at order `q` on the gauge wave across the interface
and vanish on exact data to roundoff; digests identical at 1 and 4
threads; `Float32` reproduces the gauge-wave rate at coarse resolution.
Mark G3 *(Done.)*.

## Step 5 — A black hole, static: the interior and the driver (G4a)

`CODE.md`: "The interior: a pointwise damping layer", "Boundaries:
Dirichlet", "Gauge and constraint damping" (the damping profile),
"Analysis quantities" (masked norms, the record), "Refinement and
regridding" (the driver loop and the frozen-hierarchy protocol only;
the indicator is step 6).

Changes: `src/interior.jl` — `Interior(center(t), r_0, r_1, ρ_max,
ramps)`, the `C²` smoothstep profiles `w(r)` and `ρ(r)`, the core
predicate, the `(INTERIOR)` term inside `gh_rhs_kernel!` with the
`w = 0` branch that skips `F`, `u_exact(x, t)` from the background in
the kernel, the core rule in the initial-data callback, the `:pasted`
variant's `step_limiter!` and the `:frozen` variant (`ρ = 0`), the
checks `r_1 ≤ r_h,min − m·h` (the horizon's smallest coordinate radius,
boost-contracted; `m` defaults to 8 and is never below `G + 1`; `h` the
spacing of the blocks containing `r_1`) and `r_1 − r_0 ≥ 2(G + 1) h`,
and `ρ_max = 1/dt` per chunk; a **test fixture** `hole_forest(center;
radii, levels)` that builds a fixed nested hierarchy around a point with
`refine!` and `balance!` (TreeAMR's `wave_forest` pattern; this is *not*
the refinement mechanism, which step 6 adds — it is the frozen hierarchy
the convergence protocol needs and the mesh the interior is first
tested on); `src/driver.jl` — `GHCase` (background, box, periodicity,
`r_0`, `r_1`, `interior`, `ε_KO`, `γ0` profile, `γ2`, `chunk`, and the
refinement fields step 6 fills in), `evolve!` as `CODE.md`'s loop with
the CFL recheck, the per-chunk analysis record, `observer`, and a
`regrid = false` switch (the only mode this step uses); the mask
`r < r_1` wired into every norm and monitor, the interior residual
added to the record; cases: Kerr-Schild (`a = 0`, sampled `H`) and
harmonic Kerr (`a = 0`, `0.9`).

Accept: the radius assertions asserted and tested to fire; the
per-chunk analysis record holding the masked constraint norms, the
interior residual and the mesh statistics; the masked error against the
exact solution converges at order `q` on the fixed hierarchy as `N`
doubles, to `t = 50 M` at the coarser resolutions and to a few `M` in
the suite; constraints flat at truncation, masked; the interior residual
at truncation for `:damped`; the three variants run on the static hole
and their constraint norms in the `G` points outside `r_1` recorded, the
default confirmed or changed in `CODE.md`; the gauge drift rate of
`h_tt` at the horizon recorded beside GHSO2's `≈ 0.14/M`; the
discrete-gradient `Π` post-pass measured against the analytic one; the
same run in `Float32` on `CPU()` reaching the same result to `Float32`
accuracy; the layer's share of an RHS evaluation recorded.

## Step 6 — The refinement indicator (G4b)

`CODE.md`: "Refinement and regridding" (all of it). TreeWave's
`refinement.jl` and its `CODE.md` section "The refinement criterion"
are the source of the port.

Changes: `src/refinement.jl` — `lohner` with the global floor,
`field_scales`, `cell_indicator` over the ten `h` components through
`firing_boxes` with the interior mask, the four marks with the box
keyed on `coarsen_tol`, the level floor around the horizon and the
ceiling near the boundary as `clamp(request, floor(x), ceiling(x))`,
`refine_flags`, `refinement_buffer`, the refinement centroid; the
refinement fields of `GHCase` (`refine_tol`, `coarsen_tol`,
`maxlevel_cap`, the floor and ceiling parameters); `evolve!`'s regrid
branch with the ghost fill before flagging; `adapt_to_initial_data!`
driven by the indicator; the centroid added to the record.

Accept: the calibration table of `τ_max` against `h` on uniform meshes
for the static hole, the thresholds chosen from it and recorded; the
initial-data cycle converges to nested shells around the hole, coarse
at the boundary, and regrids during the static run change nothing; the
floor not binding at the calibrated thresholds and binding when
`refine_tol` is loosened (both tested); the ceiling holding the boundary
blocks at the coarsest level; the adaptive static run agreeing with the
step-5 fixed-hierarchy run at the same finest spacing to the level the
coarser outer shells allow (recorded); the interior assertions holding
on the indicator's mesh.

## Step 7 — Horizons (G4c)

`CODE.md`: "Analysis quantities" (the horizon rows), "Upstream
prerequisites" (point interpolation); `notes/methods-ghso2.md`,
"Apparent horizons and spin".

Changes: `src/horizon.jl` — `interpolate(fs, xs)` (host, `find_leaf`
then Lagrange interpolation of order `q + 2` within the containing
block's stored points, batched), the `ADMVars` provider from
`adm_vars_from_state` in the finder's batched form, `find_gh_horizon`
as GHSO2's — location (`origin`, `r_min`, `r_mean`, `r_max`), shape
`hlm`, `area`, `M_irr`, `J` with its axis, `M_ch` — seeded from the
previous find and recentred on the analytic center; the horizon
quantities added to the analysis record; deps `ApparentHorizonFinder`
and `KorzynskiSpin` added with `[sources]` pins.

Accept: interpolation exact on polynomials of degree `≤ q + 1` and
converging at order `q + 2` on the analytic metric; the horizon of the
step-5 and step-6 runs found from a displaced initial guess, enclosing
the layer by the margin `m`; area `4π(r_+² + a²)`, `M_irr`, `J = M a`
and `M_ch = M` recovered to interpolation accuracy for `a = 0` and
`0.9`, with the spin axis along `±ẑ`; the provider throws when a
query's interpolation footprint reaches `r_1`. Mark G4 *(Done.)*.

## Steps 8a–8g — The generic interior (added 2026-09-23)

These come before step 8 because step 5's layer needs two things step 8's
case does not have: an analytic center, and an analytic interior whose
singular set a *ball* can contain. Harmonic Kerr at `a = 9/10` is singular
on the equatorial disk of coordinate radius `0.9` while its horizon's
smallest coordinate radius is `0.436`, so `check_interior_radii` refuses
the proof-of-concept case. The seven steps below replace both analytic
inputs by quantities from the tracked apparent horizon and the evolved
state, add the instrument that reports where an interior treatment fails,
and measure the result on every hole this package has. The design and the
review that shaped it are in `CODE.md`, "The interior" (as these steps
amend it) and "Open questions". Steps 8a and 8b are independent and may
run as two agents at once; everything after is sequential, each from a
green suite on the merged branch.

Four findings of the design review that every step below rests on. They
are stated here because they contradict what a reader of step 5's code
would assume, and because each is a *prediction* the steps measure:

1. **`ρ_max = 1/dt` is a grid rate, not a physical one.** On the suite's
   fixture (`h = 5/64`, `cfl = 1/5`) it is about `107/M` against the
   surface gravity `κ = 1/(4M)` and the transport rate `λ/h ≈ 21/M`; with
   the quintic ramp `ρ ≈ 37/M` two cells inside `r_1`. Step 5's `:damped`
   is therefore a paste two cells deep with a two-cell transition, and it
   survives to `50 M` because its target is exact. An *inexact* target
   pinned that hard, that close to the evolved stencils, is a kink: the
   compact `∂²` stencil divides by `h²`, so a curvature mismatch
   `[u''] = O(1)` at `r_1` is an `O(1)` right-hand-side error at the
   innermost evolved points, independent of `h` and `q`. A generic target
   needs a thick ramp at a physical `ρ_max` (a few `/M`) so that `ρ` at
   depth `G` is below `1/M`; the prediction is `n_L ≳ G (10 ρ_max M)^{1/3}`
   cells, and step 8c measures it.
2. **The Lorentzian metrics are not convex in `g_ab`.** The angular mean of
   Kerr-Schild `g_ab` on the sphere `r = 1.15 M` has `g_tt = +0.74`,
   `g_ti = 0`: Euclidean signature. Every blend, mean, fit or clamp is
   made in ADM variables `(α, β^i, γ_ij)`, where `α > 0` and `γ` positive
   definite are convex, and reassembled into `g_ab`. A regular vector
   field vanishes at the center, so `β → 0` there and the center is a
   valid metric whenever `α` and `γ` are.
3. **A sphere keyed on the *tracked* `r_h,min` still does not unblock
   harmonic `a = 9/10`; the offset surface does.** The singular disk must
   lie inside the surface on which the analytic solution is last used,
   and `0.436 − m h` is far inside the disk. The layer is keyed on the
   *depth* `d = r_h(n̂) − m h − |x − c(t)|` below the found horizon's
   offset surface: on the equator `r_h = 1.0`, so the disk is inside when
   `m h < 0.1 M` (`m = 4`, `h = 5/256` gives `r_1(eq) = 0.92`,
   `r_1(axis) = 0.36`). The sphere is the `l = 0` case of this geometry.
4. **The discrete scheme is not causal at the grid scale.** In the
   continuum every GH mode lies inside the light cone, so inside the
   horizon nothing escapes, discontinuities included. Discretely, every
   centered first-derivative stencil annihilates the Nyquist mode, so the
   shift advection that makes everything ingoing does not act on it: in
   the frozen-coefficient model `(∂_t − b ∂_x)² u = a² ∂_x² u` (the code's
   sign: `∂_t h = +β ∂h`, speeds `−b ± a`) the Nyquist mode's group
   velocity is `−b s′(π)` — outward at the shift speed for
   `q = 2`, `5b/3` for `q = 4` — and intermediate wavelengths turn outgoing
   once `a cos(θ/2) > b cos θ` (at `q = 2`), where Kreiss–Oliger damping
   is weak. Grid-scale content generated inside the horizon *can* cross it,
   attenuated by `e^{−d/ℓ(θ)}` with `ℓ = v_g/σ_KO` cells per e-fold. This
   is GHSO2's "grid-scale layer cured by `ε_KO ≈ 0.5`" and the reason the
   margin `m` exists; every interior treatment is a *source* of these
   modes, and `ℓ_max` for this package's stencils is computed in step 8a
   before anything else is built.

### Sharp edges for steps 8a–8g

- **`diag` slots are appended, never inserted.** `DIAG_CGH` and `DIAG_MOM`
  are contiguous ranges reduced as ranges; a slot in the middle of either
  breaks `block_mapreduce`. New slots start at `NDIAG + 1 = 15`.
- **Three state writers and no more**: the RHS never mutates `u`; the
  `:pasted` limiter and step 8b's range projection write it, from RK4's
  `step_limiter!`/`stage_limiter!` only, and write back **only where they
  fired** — a run on which nothing fires must be bitwise the run without
  them, and a test asserts it. Both limiters are `solve` keywords, not
  `RK4(; …)` arguments (the constructor form is deprecated and silently
  unread in newer OrdinaryDiffEq).
- **Masked slots are written through a branch**, never as `keep * value`:
  the masked region may hold a `NaN`.
- **Kernels capture no `Type` and no host array**; a device array passed
  as an argument is fine (`Hsrc` is the precedent, and step 8e's fit
  coefficients follow it). No `return` anywhere in a kernel body.
- **Everything generic in `T`**: `T(1//2)`, `oftype`, no decimal literal in
  a `T` expression; `Float32` rows are recorded, pass or fail.
- **Price every test**: the suite is 12m31 at one thread; a new `q` or
  interior variant is a new kernel (about 20 s); a `50 M` run is 19 min at
  four threads and belongs in `test/hole_runs.jl`, never in the suite.
- **Long runs go to Symmetry** through `.claude/orchestration/
  symmetry-run.sh` (the `symmetry-hpc` skill has the mechanics; do not
  export `JULIA_EXCLUSIVE=1` for the one-thread suite). **Wait for a job
  with a blocking command or a polling loop; never "arm a monitor" and
  end the turn** — an agent that does so stalls.
- **The fixtures**: `hole_fixture` in `test/evolution_cases.jl` (Kerr-Schild
  `a = 0`, `q = 2`, `N = 8`, `h = 5/64`, `r_0 = 2/5`, `r_1 = 23/20`,
  `m = 8`, box `5/2`, 120 leaves); `adaptive_hole_fixture` (box `5`,
  `m = 4`); `gh_outside_shell_norms` reads the `G` points outside `r_1`
  through a `ShellMask`. `hole_runs.jl` takes section names as arguments;
  add a section, do not lengthen an existing one.
- **Work in the worktree you were given**, on your step's branch, commit
  in TreeAMR's style with measured numbers in the body, do not merge or
  push, and do not touch `TODO.md` or `notes/`.

## Step 8a — Expectations: anomalous group velocity and leakage

`CODE.md`: "Finite-difference stencils", "Kreiss–Oliger dissipation", "The
interior" (the margin `m`), `notes/methods-ghso2.md` lines 210–233 and
`notes/sonic-surface.md` (GHSO2's grid-scale layer). **No `src` change.**

Changes: `test/dispersion.jl`, a standalone analysis script in the manner
of `hole_runs.jl` — for the frozen-coefficient model
`(∂_t − b ∂_x)² u = a² ∂_x² u` (`b = β^r`, `a = α√γ^{rr}`; the code's
sign is `∂_t h = +β ∂h`, so both characteristic speeds `−b ± a` are
negative inside the horizon),
semi-discretised with the package's own weights (`derivative_weights` of
order `q` for the advection, the compact second derivative,
`dissipation_weights` of order `q + 2` scaled by `ε_KO`), the two branches
`ω(θ)` for `θ = kh ∈ (0, π]`, the damping `σ(θ) = −Im ω`, the group
velocity `v_g(θ) = Re dω/dk`, and the penetration length
`ℓ(θ) = max(v_g, 0)/σ` in cells per e-fold; the table of `ℓ_max = max_θ ℓ`
and its `θ` against `q ∈ {2, 4, 6}`, `ε_KO ∈ {0, 1/4, 1/2, 1}` and `b/a`
at Kerr-Schild `r = 1.0, 1.2, 1.5, 1.8, 2.0 M`, with the attenuation across
the default margin `e^{−8/ℓ_max}`; a fully discrete column (RK4 at
`cfl = 1/4`) if the semi-discrete numbers are marginal. One testset in
`stencils_tests.jl`: the Nyquist mode's group velocity under the order-`q`
advection stencil is `−b s′(π)` (`+b` at `q = 2`, `+5b/3` at `q = 4`), a
`Rational` claim about the weights beside the existing damping-sign claim.
One section `leakage` in `hole_runs.jl`: on `hole_fixture` at `q = 2` and
`q = 4`, add to the initial data a radial ripple of wavelength `2h`, `4h`,
`8h` and amplitude `1e−3` confined to a shell at depth `d = 2, 4, 8` cells
inside the horizon and *outside* `r_1`; evolve to `1 M` at
`ε_KO ∈ {0, 1/4, 1/2, 1}`; record the L∞ of the difference to the
unperturbed run in shells `[r_h + k h, r_h + (k+1) h]`, `k = 0 … 8`,
against time (a `ShellMask` per shell), and fit the attenuation per cell.
Each run is about two minutes; the section is a Symmetry job.

Accept: the table in `CODE.md` beside the stencil section, marked
**(measured in step 8a)**; the measured attenuation against the predicted
`e^{−(d+k)/ℓ_max}`, with the discrepancy recorded; two rules stated under
"The interior": the stencil margin `m ≥ G + 1` and the *leakage* margin
`m ≥ n_e ℓ_max` for a wanted attenuation `e^{−n_e}`, both marked
**(proposed in step 8a)** for the reviewer to confirm; and a recommendation
on whether `ε_KO` rising inside the layer is needed to make `m = 8` enough.
The stencil testset in the suite; nothing else added to it.

## Step 8b — The instrument: range projection and validity monitor

`CODE.md`: "The interior" (the variants and the state writers), "Analysis
quantities" (the record), "Time integration" (`step_limiter!`); TreeHydro's
`src/floors.jl` and its `CODE.md` "Floors and the atmosphere" are the
pattern — a pointwise map installed as a stage limiter, writing back only
where it fired, idempotent on the state, counted, with a bitwise control.
**Not a reset to a fixed state and not a projection onto flat space**: a
minimal clamp of each ADM quantity into a range with a floor and a
ceiling (finding 2 says why the ranges are stated in ADM variables and not
per component of `h_ab`).

Changes: `src/bounds.jl` — `StateBounds{T}` (`isbits`: `α_min, α_max,
λ_min, λ_max, β_max, K_max` and the gating depth), a `GHCase` field like
`horizon` with default `nothing` (`initialdata.jl`, `hole_case`);
`bounds_project(h, Π, bounds) -> (h′, Π′, hit, nonfinite)` in
`pointwise.jl`'s style over `metric_quantities`/`adm_from_metric` and the
pack helpers — a non-finite component takes its Minkowski value and is
flagged separately; the eigenvalues of `γ_ij` are clamped into
`[λ_min, λ_max]`; `|β| ≤ β_max`; `α²` into its range; `Π` rescaled by `α`
and `√γ` and capped at `K_max`; `g′ = (−α² + β·β, β_i, γ′)`; a healthy
quantity is not touched, and the map is idempotent on the state with an
`8 eps` relative slack on the eigenvalue test (TreeHydro measured that the
*flag* is not idempotent without one); `gh_bounds_kernel!` over
`statearray(u, U)`, gated on the interior's depth (deeper than step 8a's
leakage margin, which the bounds carry as a radius), writing back only
where `hit` and `1/0` into the appended `DIAG_BOUNDS = 15`; a validity
monitor writing the extremes of `det γ`, `α`, `|h|` and `|Π|` over the
layer and over `ShellMask(r_1, r_1 + G h)`; `gh_stage_limiter!` dispatching
on `case.bounds` (a no-op for `nothing`), passed as `stage_limiter=` beside
`step_limiter=gh_step_limiter!` in `evolve!`'s `solve`, and applied once
after every regrid transfer as TreeHydro does; `BoundsAccounting` (host,
mutable, one per `evolve!`, shared by every rebuilt `GHProblem`), and the
record rows `bounds_hits`, `bounds_nonfinite`, `bounds_r_max` (the
outermost radius that fired), `min_detγ`, `min_α`, `max_h`, `max_Π` for
the layer and the shell. `max_speed_of`'s `all(isfinite, u)` and the
record's `finite` become masked to the evolved region: a `NaN` in the core
is a hit, not the end of the run. `test/bounds_tests.jl`. A section
`bounds` in `hole_runs.jl`.

Accept: on six synthetic states (one negative eigenvalue; two negative with
`det γ > 0`; `−g^{tt} < 0`; a `NaN` in one component; an `Inf`; healthy)
the projection returns a state `metric_quantities` accepts, moves only the
offending quantity, and is bitwise idempotent; it is the identity on every
background of `pointwise_backgrounds.jl` off its singular set;
`hole_fixture` `:damped` to `0.15 M` has zero hits and a `u` bitwise
identical with and without bounds (the control); the `bounds` section runs
`N = 6` `:damped` (dies at `21 M` in step 5's table) and `:pasted` (`17 M`)
with bounds on and records when and at what radius hits start — the
prediction to confirm or correct is "deep, several `M` before the crash" —
and whether the run then survives, with 8a's shells outside the horizon
saying whether the clamp's discontinuity reached it. The kernel's cost
recorded (prediction: `0.4 %` of a step). One short run added to the suite.

## Step 8c — Calibrate the layer for an inexact target

`CODE.md`: "The interior" (the profiles, `ρ_max`, the three variants and
their measured table); finding 1 above is the hypothesis under test.

**What steps 8a and 8b hand over** (their reports' section 6, and
`CODE.md`'s step 8a and 8b entries under "Measured results"):

- **Step 5's two failing runs die at the transition `r_1`, not deep
  inside** (measured in step 8b): `N = 6` `:damped` and `N = 8` `:pasted`
  both degenerate in the *evolved* shell just outside `r_1` (`r = 1.18` to
  `1.31` against `r_1 = 1.15`), through the lapse's square root on a stage
  vector, after a shell error that grows from the first chunk; the range
  projection never fires on either. That is finding 1 measured from the
  other side, and it is what this step calibrates. The numbers to watch
  are the validity monitor's shell rows — `min_α_shell`, `min_detγ_shell`,
  `max_Π_shell` — and the shell error, not `bounds_hits`.
- **Every run of this step carries the range projection** (`default_bounds`,
  `default_gate = r_1 − 2 G h`) as a passive instrument; its hit count is
  expected to stay zero, and a nonzero count is a finding to record with
  the radius and time of the first hit (`bounds_r_max`, the accounting's
  `first_t`, `first_r`). `diag` has 22 slots; do not insert one.
- **The leakage margin** (measured in step 8a): on this fixture `m = 8`
  attenuates grid-scale content made at `r_1` by `e^{−2.9}` to `e^{−4.6}`
  (about 1 % by `2 M`, 4–6 % long-time); the frozen-coefficient prediction
  is right on the depth dependence and overstates the level 4–7×. The
  shells `[r_h + k h, r_h + (k+1) h]` and their `ShellMask`s are built in
  `hole_runs.jl`'s `leakage` section (`leak_geometry`); reuse them for E0.
- **Dissipation raised only inside the layer buys nothing; raised across
  the margin it cuts the transmission 4–15×** (step 8a's one-dimensional
  model, `ε_in = 4`). So the profile below rises from the *horizon*, not
  from `r_1`, and `ε_in ≤ 4` keeps RK4 inside its real-axis limit at
  `cfl = 1/4`.

Changes, the smallest that make the experiments possible: `Interior` gets
an optional `target` background (`isbits`, default `nothing` meaning the
case's own), read where the kernel evaluates `u_exact`
(`background_state(bg, t, x)` in the layer branch of `gh_rhs_kernel!`) and
in `gh_paste_kernel!`; the initial data, the Dirichlet hook, the gauge
source and the error reference stay on the *true* background, so
`DIAG_RES` measures the layer's distance from the truth. `chunk_interior`
gains `ρ_max_fixed` (a rate) as the alternative to `factor/dt`, threaded
through `evolve!`. A `C²` dissipation profile `ε_KO(r)` in `gauge.jl`
beside `GaussianDamping` — the exterior's value at and outside the
*horizon* (`r_h,min` of the case), rising to `ε_in` at `r_1` and held
inside (amended after step 8a; see the handover above) — accepted by
`GHCase` where a number is today. Test-side target wrappers in `test/evolution_cases.jl` implementing
`SpacetimeMetrics`' `metric`/`dmetric` interface. A section `calibration`
in `hole_runs.jl` with the experiments below, on `hole_fixture` (`q = 2`,
`N = 8`, `h = 5/64`), screened at `5 M` and run to `50 M` where they
survive, reading `gh_outside_shell_norms`, `residual`, `drift`, the
finder's `M_irr`, step 8a's shells outside the horizon and step 8b's rows:

- **E0** — `:pasted` with `KerrSchild(1.2, 0)` as target: a 20 % hard
  step at `r_1`, the discontinuity of the question "can it get out"; the
  shells outside the horizon to `5 M` at `ε_KO ∈ {1/4, 1/2, 1}` against
  8a's predicted attenuation.
- **E3** — `u_fit = u_exact + A (r − r_1)² χ(r)` in `h_tt`, `A ≈ 2/M²`,
  `χ → 0` by `r_0`: value and slope right, curvature wrong. The
  `N = 6, 8, 10` sweep to `0.15 M`: does the `G`-point-shell `C_a` still
  converge at order `q`? Then the scan `n_L ∈ {2G, 3G, 4G, 6G}` cells
  (through `r_0`) × `ρ_max M ∈ {M/dt, 10, 4, 1}`, and the same scan with
  `ε_in ∈ {1, 2, 4}`.
- **E1** — `KerrSchild(1.2, 0)` as the `:damped` target: a valid metric
  that is not a solution; survival, the shell `C_a`, the drift of `M_irr`.
- **E2** — `translate(KerrSchild(1, 0), (δ, 0, 0))`, `δ = h` and `4h`: the
  proxy for a tracking error; the shell `C_a` against `δ`,
  `center_offset`.
- Controls: `:frozen` and `:pasted`, and the step-5 `:damped` at `N = 6`
  whose end step 8b replayed, all with the range projection on; the
  shell validity rows of each beside the scan's.

Accept: the scan's table in `CODE.md` "Measured results" and, under "The
interior", the layer rule for a generic target — ramp thickness in cells,
`ρ_max` as a rate, `ε_KO(r)` — marked **(measured in step 8c)** where the
scan decides and **(proposed in step 8c)** where it interpolates; the
prediction `n_L ≳ G (10 ρ_max M)^{1/3}` confirmed or replaced; a stated
recommendation — proceed to steps 8d–8f, or to 8g — with the numbers it
rests on. Nothing long in the suite; the `target` keyword and the profile
get one short claim each.

## Step 8c′ — `ρ_max = 4/M` is the default (decided 2026-09-23)

Erik's decision on step 8c's proposal: the fixed physical rate `4/M` is
the default relaxation rate of the layer for **every** variant, the
analytic `:damped` layer included. `ρ_max · dt = 1` — a grid rate, about
`107/M` on the suite's fixture, a paste two cells deep, and on the exact
target six times the `50 M` error (step 8c) — stays available as the option
`ρ_max_factor` and is no longer what a run gets by default.

Changes: `evolve!` with neither `ρ_max_factor` nor `ρ_max_fixed` given
relaxes at `4/M`, `M` the case's hole mass — a `hole_mass(background)`
dispatch beside `horizon_min_radius` in `interior.jl` (`.mass` of
`KerrSchild` and `Harmonic`, through `translate`, `rotate` and `boost`), so
the default is a statement about the hole and not a number in the driver;
`ρ_max_factor` given selects the grid rate as before; a case without a hole
is untouched; the record's `ρ_max` row says what ran. The suite's claims
that encoded the grid rate are **amended, not loosened** — this is a decided
change of the spec: `driver_tests.jl`'s `r.ρ_max * r.dt ≈ 1` becomes
`r.ρ_max ≈ 4/M`; the three-variants claim (`pasted.residual == 0`,
`frozen.residual > 2 · damped.residual`, the shell `C_a` within
`rtol = 1/4`) is re-measured at `4/M` — the sink now relaxes in `1/4 M`
rather than in one step, so at `t = 1/10 M` the `:damped` residual may not
yet be the saturated one; if the claim needs a longer `t_end` to be true,
change the time and say why in `CODE.md`, and if it is false at any time,
report it; the order sweep, the drift, the `Π` post-pass, the `Float32`
row, the horizon and refinement runs re-measured, with no other assertion
changed unless it encodes the rate. `hole_runs.jl`'s sections run at the new
default; step 8c's `calibration` keeps its `:grid` rows through
`ρ_max_factor`.

`CODE.md`: "The profiles and their parameters" — `ρ_max · dt = 1`
**(proposed)** becomes `ρ_max = 4/M` **(decided 2026-09-23)**, the grid rate
kept as the option with the reason above; step 5's measured tables stay as
history, annotated "at the grid rate `1/dt`, the default until 2026-09-23",
and the re-measured suite numbers go beside them marked **(measured in step
8c′)**; "Open questions" item 5 and every sentence that says `ρ_max·dt = 1`
is the default; `CLAUDE.md`'s "Things that will bite" (the `ρ_max` entry)
and "Current state". No new long run: step 8c already measured the exact
target at `4/M` on the fixture's own layer to `50 M` (shell `C_a` `0.029`,
masked L2 `0.027`), and that row is the default's `50 M` number.

Accept: the suite green at one and four threads with the re-measured
numbers in `CODE.md`; a `:damped` run of the fixture with no rate keyword
has `ρ_max == 4/M` in every record row; `ρ_max_factor = 1` reproduces
step 5's `t = 1/10 M` variants row to the digits `CODE.md` records; the
default stated in one place and read from the hole's mass.

## Step 8d — The tracked horizon geometry

`CODE.md`: "The interior" (placement, the radius assertions), "Analysis
quantities" (the horizon rows), "Refinement and regridding" (the level
floor); finding 3 above.

**What step 8c hands over** (`CODE.md`, "The layer for an inexact target"
and the step 8c entry under "Measured results"):

- **The layer rule, measured on the fixture**: `ρ_max = 4/M` as a fixed
  physical rate (`evolve!(…; ρ_max_fixed)` exists; the code's default is
  still `1/dt` for the analytic `:damped` layer until the reviewer takes
  the proposal), a ramp `n_L = max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)` cells over
  which `ρ` rises from `0` at `r_1` to `ρ_max` (`8` at `q = 2`, `12` at
  `q = 4`), `w` turning over in the inner half, and a *constant* `ε_KO`
  (the `HorizonDissipation` profile makes a smooth transition worse and
  stays off). `FittedInterior`'s `thickness` is `n_L h` by this rule.
- **The tracked center must be good to about one cell.** A displaced
  target at `δ = h` costs the exterior nothing; at `4h` it costs 3–4× or
  ends the run. So `update_track` records the tracked center's distance
  from the analytic one in cells wherever a case has one, the velocity
  estimate's error times the chunk must stay below `h` for a moving hole
  (a refusal or a recorded warning, decide and mark it), and the finder's
  own accuracy (`center_offset` at `N_ah = 12–16`, `h = 5/64`: about `1e−3`
  on the static hole) is what makes that achievable.
- **Every run carries the range projection as a passive instrument**; its
  hit count stayed zero on every surviving run of 8c, and the shell
  validity rows are the signal of a failing transition.
- **The leakage margin depends on the background** (step 8a's table for the
  spinning holes): Kerr-Schild `a = 9/10` gets under one e-fold from
  `m = 8` at `h = 5/256`, the harmonic equator 7–10 from `m = 4`. Keep
  `m = 8` as the default and let `fitted_interior` report the path
  integral `∫ dr/(h ℓ_max)` across its margin, computed by the functions
  of `test/dispersion.jl` moved into `src/` if that is what it takes, so
  that step 8f's rows carry their e-folds.

Changes: `src/tracking.jl` — `HorizonTrack{T}` (host: `t_find, c_find,
v_est, r_min, r_max, hlm, source ∈ {:analytic, :found, :coasting}, misses,
nfinds`), `seed_track(case, t)` from the analytic center and radii,
`update_track(tr, hz, t)` (velocity from the last two finds, coasting on a
failed find, a throw naming the staleness after `max_misses`), and
`track_center(tr) -> HoleCenter` so that every `center_at` consumer — the
damping profile, the masks, `interior_radius` — is untouched. In
`interior.jl`: `FittedInterior{T,NM}` (`isbits` kernel argument:
`center::HoleCenter`, the shape as an `SVector` of *real* spherical-harmonic
coefficients to `lmax_shape`, converted from the finder's `hlm` through
`ash_resample` with the conversion tested against `ash_evaluate`; the
analytic shape `r_h(θ) = R √((R² + a²)/(R² + a² cos²θ))`, boost-contracted,
for the seed; bounding spheres `r_in, r_out` for the fast paths;
`offset = m h`, `thickness = n_L h`; `ρ_max`, the ramps, `margin`),
`fitted_geometry(int, t, x) -> (r, n̂, d)` with the depth `d` of finding 3,
and `is_frozen`, `interior_profiles`, `in_layer(int, t, x)` (a protocol
change for both interior types), `core_position` onto the core surface,
`ShapeMask` for the norms and `footprint_evolved` on the depth;
`fitted_interior(spec, tr, forest, G; t)` deriving `h` as the coarsest
spacing among the blocks meeting the annulus (`_box_radii`) and refusing
`r_min − (m + n_L + core_min) h ≤ 0` by name; `check_interior_radii` and
`horizon_floor_level`/`level_bounds` on the track's radii, with the
`singular_radius` check kept for the analytic-target variants only;
`find_gh_horizon` gains `center=`; the lapse-collapse trigger — `min α`
over the evolved region (8b's monitor) below a threshold forces a find at
the next chunk boundary whatever the cadence. Record rows for the track's
source, center, velocity and radii.

Accept: the depth of an oblate spheroid recovers its axis and equator; the
tracked geometry of the static Kerr-Schild hole agrees with the analytic
one to interpolation accuracy after one find; a run whose finder is
disabled after `0.1 M` coasts and records `:coasting`; a find whose `r_min`
is perturbed by `G h` is refused; the kernels are bit-identical between an
`Interior` and a `FittedInterior` holding the same sphere with the same
profiles; the cost of the shape evaluation per layer point recorded.

## Step 8e — The fitted target and the `:fitted` variant

`CODE.md`: "The interior" (the target, as 8c and 8d amended it), "Initial
data and backgrounds" (the core rule), "One right-hand-side evaluation";
findings 1 and 2 above. **What step 8c hands over**: the layer rule above
(`ρ_max = 4/M`, `n_L = max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)`, constant `ε_KO`),
under which a 20 % mass error, a curvature error of `4/M²` and a
displacement of `h` each leave the exterior where the exact target leaves
it (shell `C_a` `0.029–0.039`, masked error `0.027–0.032`, `M_irr` `0.9972`,
flat from `10 M` to `50 M`, no projection hit); and the four things the
fitted target must therefore be: **smooth** (a hard step at `r_1` gets
out at about a percent and ends the run at `ε_KO ≤ 1/2`), **a valid
metric at every point of the layer** (E3's other sign drove `α²` through
zero; the validity sweep below is not optional), **regular on the layer**
(a singular point of the target on a grid point ends the run at `t = 0`,
and nothing checked it — the fit is a polynomial, so this is a check that
the sampled data was finite), and **centered to about a cell** (8d's job).
Continuity in *time* is the fifth: the coefficients are interpolated
linearly between the last two fits. `FittedSpec`'s defaults are the rule's
numbers, and the driver applies its rate through `chunk_interior`'s fixed
path.

**What step 8d hands over** (its report's section 6 and `CODE.md`, "The
tracked geometry"):

- **The real-harmonic convention the fit must share**: slot `l² + l + m + 1`
  (as `ash_mode_index`), `m ≥ 0` cosine (and `l0`), `m < 0` sine;
  `ỹᶜ = √2 Re Y_lm`, `ỹˢ = −√2 Im Y_lm`, `a_l0 = Re c_l0`, `aᶜ = √2 Re c`,
  `aˢ = √2 Im c`; convert only with `real_from_complex`/`complex_from_real`;
  `shape_series` (a Legendre recurrence in `n_z` times `(n_x + i n_y)^m`, no
  angles) is the kernel-side evaluator to reuse for `fitted_state`. It costs
  21 ns per point at `lmax = 4` and 79 ns at `8`, against 96 ns for one
  analytic `u_exact`; harmonic Kerr at `a = 9/10` needs `lmax = 12` for a
  tenth of a cell at `h = 5/256`, Kerr-Schild is fine at `4`.
- **The protocol 8d changed**: `in_layer(int, t, x)`; the kernel asks
  `interior_point`/`is_frozen`/`is_outside`/`interior_profiles`;
  `geometry_radii(int, bg)`, `layer_radii(int)`, `layer_mask`, `shell_mask`;
  `find_gh_horizon(center=)` returning `origin_r_min/mean/max`;
  `FittedInterior` carries `h`, `n_L`, `margin` and `target`; a raw
  `FittedSpec` is refused by `GHProblem`, `state_callback`, `level_bounds`
  and the `Π` post-pass — the driver builds the geometry per chunk with
  `fitted_interior(spec, track, forest, G; t, n_L)`.
- **The track's accuracy**: `track_offset` `1.8e−4` cells after one find,
  `3.3e−4` at `0.15 M`, `3.0e−3` at `5 M` on the static hole; the growth is
  the two-find velocity integrating finder noise, and smoothing the velocity
  over several finds is the first thing to try if a moving hole needs it.
- **Harmonic `a = 9/10` is still refused** by the analytic-target core rule
  (a ball cannot hold the disk); it is this step's fitted target and its
  initial-data fill that lift the refusal — that row of 8f is the proof.
- **The boost sign (found in step 8d, verified in review)**:
  `SpacetimeMetrics.boost(m, v)` moves the hole at `−v`, and `HoleCenter`'s
  docstring says the opposite. Fix it here: a `hole_velocity(background)`
  dispatch beside `hole_mass` (zero for a static hole, `−v` for a
  `BoostedMetric`, through translate and rotate), `GHCase` deriving
  `velocity` from it when the keyword is not given and refusing a keyword
  that disagrees, the docstring corrected, and a test that the seed
  track's center at `t = 1` sits where the boosted metric is singular.
 **Two halves, reviewed between them**: 8e-i is
host-side and has no kernel; 8e-ii is the kernel, the driver and the
variant. An agent does 8e-i, reports, and continues to 8e-ii only when the
reviewer has read 8e-i's numbers.

8e-i — `src/fit.jl`: `build_fit(sampler, int, spec; cont)` — collocation
points `x_p = c + r_1(n̂_p) n̂_p` on `EquiangularGrid(L)`; `state_sampler(fs,
q)` through `interpolate_grad` with the footprint guard off *for this call*
(its window reads `G h` inside `r_1`, where `ρ` is below `1/M` by 8c's
rule); `analytic_sampler(bg, t)` by central differences along the ray; the
samples converted to `(log α, β^i, γ_ij, Π_ab)` and their radial
derivatives; the ansatz `Σ_{l ≥ 1} ỹ_lm(n̂) ρ^l (A_lm + B_lm ρ²)` plus
`A₀ + B₀ ρ²` for scalars and tensors and `l ≥ 1` only for `β` (a polynomial
in `x`, regular at the center, a valid metric there by finding 2); one
`qr` for the 20 right-hand sides, `cont = 2` adding `C ρ⁴` and the
second-derivative rows for the initial data; `fit_residual`; a validity
sweep over eight radii × the collocation directions plus the center through
`bounds_project`, recorded as `fit_valid` and thrown with `min det γ`,
`min(−g^{tt})` and the remedies if it fails; `InteriorFit` holding the
coefficients `((L+1)², cont+1, 20)` in a device array through `to_backend`
(the `Hsrc` precedent). `test/fit_tests.jl`.

Accept 8e-i: the complex→real harmonic conversion agrees with
`ash_evaluate` to roundoff; the `cont = 2` fit of analytic Kerr-Schild data
on `r_1` reproduces it there to `L` truncation and to interpolation order,
and is a valid metric at every swept point for `a = 0` and `a = 9/10`; the
`g_ab` angular-mean control is asserted *invalid*; the fit of the
*evolved* state at the end of a `:damped` run agrees with the fit of the
analytic solution to the run's masked error; the host cost per fit
recorded (prediction: milliseconds).

8e-ii — `GHProblem` gains `fit` and `with_interior(p, int; fit)`; the
kernel gains a `fitwork` argument and the branch `INT === :fitted`: core
`du = −ρ_max (u − u_fit)` with `F` not evaluated, layer
`du = w F − ρ (u − u_fit)` with the fit evaluated only where `ρ > 0`,
evolved `du = F`; `fitted_state(int, fitwork, r, n̂)` with the Legendre and
Chebyshev recurrences shared with the shape evaluation, ADM reassembly, no
allocation, no `return`; `gh_step_limiter!` and `paste_interior!` no-ops
for `:fitted`. The driver per chunk: `with_interior` at this chunk's
`ρ_max` (a rate from the spec); `solve`; `record!` (constraints, error,
monitor, find, indicator, in that order); `update_track`; the regrid branch
with `interior=` and `fit=`; `fitted_interior` at `stop`; `build_fit` on
the ghosts the find just filled; the coefficients of the last two fits
interpolated linearly in time inside the chunk so the target never jumps.
Initial data: `case_state_tuple` for `:fitted` evaluates the `cont = 2`
fit of the *analytic* solution inside `r_1(n̂)`, so a chart whose interior
is singular gets regular data; `core_position` stays for `:damped`.
`INTERIOR_VARIANTS` grows by `:fitted`; `FittedSpec{T}` (`margin` from
8a's rule, `n_L`, `core_min`, `lmax_fit = 8`, `lmax_shape`, `ρ_max`, the
ramps, the `ε_KO(r)` profile from 8c, `max_misses`) is what
`case.interior` holds for it.

Accept 8e-ii: one right-hand side with the `:fitted` layer equals
`:damped`'s outside `r_1` bit-for-bit; `hole_fixture` `:fitted` to `0.15 M`
has the masked error at `:damped`'s level, `fit_valid = true` and zero
bounds hits; the same at `Float32`; the kernel cost of the fit recorded
(prediction: `+5–10 %` per right-hand side at `L = 8`, evaluated only where
`ρ > 0`).

## Step 8f — The measurement matrix

`CODE.md`: "Measured results", "The interior", milestone G5's acceptance.
A section `generic` in `hole_runs.jl`; the long rows are Symmetry jobs.

**What step 8e hands over** (its two reports' section 6 and `CODE.md`, "The
fitted target", pieces 8–12):

- **`:fitted` works on the static hole**: bit-identical to `:damped` outside
  the offset surface; the masked error is `4.3×` `:damped`'s at `0.15 M`
  because the decided `cont = 1` initial data has a curvature kink at `r_1`,
  decaying to `1.24×` by `1 M`; `evolve!(…; fit_initial_depth = n_L·h)`
  (analytic data down to the core surface, the fit only inside it) removes
  it and matches `:damped` to 2 % — use it on every chart whose layer lies
  outside the singular set (Kerr-Schild at both spins, harmonic `a = 7/10`).
  The cache read costs nothing measurable; a refill is `0.4–0.7` of one RHS,
  once per chunk (twice on the moving seed).
- **The moving seed works**: `boost(Harmonic(1, 0), 0.3 x̂)` on 848 blocks at
  `h = 5/128` to `0.1 M`, every find succeeds, `track_offset ≤ 3.9e−3` cells;
  runs past about `1 M` need `regrid = true` (refused for `:fitted` with
  `adapt = true` today — 8f decides whether to build the regrid path for it
  or to widen the fine region) or a wider fine region.
- **Harmonic `a = 9/10` is blocked by resolution, not by the interior**: its
  initial data now exists (1.27 million points, all a valid metric,
  `min det γ = 0.648`, `min α = 0.208`), but at `m = 4`, `h = 5/256` the ring
  is `0.02 M` inside the offset surface on the equator, the data there are
  a thousand times the axis values, the fit is wrong away from its
  collocation points, and the solution's own length scale at the first
  evolved point is `0.008 M`, below half a cell; the run ends at `2.5e−3 M`.
  It wants `h ≲ 5/1024` on the equator (a node-sized mesh), `Π̃ = (α/√γ)Π`
  fitted in place of `Π`, and `L ≥ 12` at the least — or a
  direction-dependent margin. **Decided 2026-09-23 (the orchestrating
  session, with the decision delegated by Erik): G5 runs at `a = 7/10`, the
  plan's fallback; no node is spent on `a = 9/10`.** The reasons, to be
  recorded in `CODE.md`'s "Open questions": at the equator the *solution's
  own* length scale at the first evolved point is below half a cell at
  `h = 5/256`, so the failure is resolution before it is the fit; the best
  host-side fit (`Π̃`, `L = 12`) is still `~100×` off the analytic second
  difference at 45°; and what would unblock the chart is three unbuilt
  ingredients (`Π̃` as the fitted momentum, `L ≥ 12`, an equator four times
  finer or a direction-dependent margin) followed by a run whose time step
  is a quarter of today's on a mesh with tens of thousands of blocks and no
  checkpointing — a multi-node-day research item, not a row. **(Amended
  2026-10-01:** `evolve!` checkpoints and restarts now, so such a run is a
  chain of jobs; the other ingredients stand.**)** The matrix
  below therefore runs the `a = 7/10` rows as G5's chart. Its `a = 9/10`
  row is **reduced to a host-side probe** that costs minutes and no node:
  the initial data and one right-hand side at `h = 5/256` as 8e measured
  them, the `Π̃` and `L = 12` kink numbers at the three latitudes
  re-measured with whatever 8f changes in the fit, and a written estimate
  of the node run (blocks, `dt`, right-hand sides, wall clock at 64
  threads) at `h = 5/1024` on the equator — so that `CODE.md` closes the
  question with numbers and a price rather than by abandoning it. If 8f
  finds a cheap change that makes the `5/256` run survive its first chunk,
  it may run that case to whatever `t_end` fits four local threads in
  fifteen minutes, and report; it does not submit it to Symmetry.
- **Rows and costs** (from 8e's estimates): Kerr-Schild `a = 0`, `50 M`,
  120 blocks, `m = 10`, `n_L = 8`, `L = 8`: about 40 min per variant at four
  threads; Kerr-Schild `a = 9/10`, `h = 5/128`, `m = 8`, about 1000 blocks:
  about 1.5 h per variant at four threads for `50 M`, a Symmetry job;
  harmonic `a = 7/10`: `m = 4` at `h = 5/128` or `5/256` with
  `fit_initial_depth = n_L·h`; the target off the equator is still coarse
  (at 45° its second difference is `18×` the analytic one, `5×` with `Π̃`) —
  fit `Π̃` before running the spinning rows. The hand-over row (`:damped`
  switching to `:fitted` mid-run) is not built; build it in 8f or drop it
  with a reason. Coasting is testable through `evolve!(…; find)`.

Changes: the section, and `CODE.md`. Every row records survival time,
masked L2 and L∞, the shell `C_a` (8a's shells outside the horizon
included), the residual against the truth where one exists, the drift,
bounds hits and `bounds_r_max`, `fit_valid` and `fit_residual`, the
horizon's `A`, `M_irr`, `J`, `M_ch` and the tracked `center_offset`:

| case | variants | what it decides |
|---|---|---|
| Kerr-Schild `a = 0`, `50 M` | `:damped` (control), `:fitted`, the snapshot target, the fitted-Kerr target | the generic layer costs nothing on the case that needs nothing |
| Kerr-Schild `a = 9/10`, `h = 5/128` | `:damped`, `:fitted` | the offset-surface layer on an oblate horizon |
| harmonic `a = 0` and `a = 7/10` | `:fitted` | the small-horizon chart, and the first spin a sphere cannot hold |
| **harmonic `a = 9/10`** | `:fitted` only, **host-side probe** (decided 2026-09-23) | the blocked case: initial data, one right-hand side, the fit's kink at three latitudes, and the *price* of the node run at `h ≲ 5/1024` written down; no node is spent |
| boosted harmonic `a = 0` or `7/10`, `v = 0.3` | `:fitted` tracked against `:damped` analytic | G5's stand-in: tracking, release, staleness |
| hand-over | `:damped` to `5 M`, then `:fitted` | a newly found horizon's first fit from evolved data |
| coasting | the finder disabled for five chunks | the geometry without a find |

Accept: the table in `CODE.md`; "The interior" rewritten around geometry,
target, the ramp rule and `ρ_max` as a rate, with step 5's design kept as
the analytic control; the `a = 9/10` open question closed with the
measured `h`, the node run's estimated price, and G5 at `a = 7/10` recorded
as decided (2026-09-23);
G5's acceptance naming the tracked geometry; `CLAUDE.md`'s "Current state"
and "Commands" updated. Mark the generic interior *(Done.)* under G5's
entry.

## Step 8g — Excision by a host reference (only if 8f says so)

**(Superseded 2026-10-05 by steps X1–X3 below**: excision by per-stencil
closures in a zone kernel rather than a fill by host extrapolation. The
brief is kept as history.**)**

`CODE.md`: "Possible extensions" (excision), "The interior"; finding 4
above; `notes/methods-ghso2.md` lines 210–233 for the sonic-surface
recipe. Run only if the `:fitted` variant does not reach `50 M` on the
first row of 8f's table.

Changes, test-only, about 150 lines: a `stage_limiter!` that fills every
owned point within `G` cells inside the sphere `r_1` by degree-`(q+1)`
Lego extrapolation along the outward axis nearest the normal, reading its
sources across blocks from `statearray(u, U)` (a host loop, allowed in
`test/` only); the kernel runs the `:pasted` branch with the paste
disabled. The `N = 6, 8, 10` sweep to `0.15 M` and the `50 M` run at
extrapolation degrees `q − 1, q, q + 1`.

Accept: the shell `C_a` at order `q` or not, survival or not, at each
degree, recorded; the halo table (a block-local fill needs `3G` ghosts:
stored volume ×4–7 at `N = 8` where `N ≥ 6G + 2` refuses it, ×1.8–2.7 at
`N = 32`) recorded under "Possible extensions" as excision's price either
way; if the test says build, the TreeAMR request (a restricted ghost
exchange over the mask's blocks) written under "Upstream prerequisites".

## Step 8 — A hole that moves (G5)

`CODE.md`: "The interior" (the moving hole), "Refinement and
regridding" (what follows for the hole), "Boundaries: Dirichlet" (time
dependence), milestone G5.

**Starts from step 8f** (added 2026-09-23): the interior is the
`:fitted` variant on the tracked geometry, with the analytic `:damped`
layer as the control, and **`a = 7/10`** (decided 2026-09-23 before step
8f started; `CODE.md` records why `a = 9/10` waits).

**What step 8f hands over** (its report's section 6 and `CODE.md`, "The
generic interior: the measurement matrix (step 8f)" and the summary at the
head of "The interior"):

- **G5's chart runs only as `:fitted`, and only at `h = 5/256`.** Harmonic
  `a = 7/10` with `m = 4`, `lmax_shape = lmax_fit = 12`, `Π̃ = (α/√γ)Π` as
  the fitted momentum (the default from 8f) and the state bounds derived
  from the whole layer reaches `10 M` on 2472 blocks — `630 s` of a node
  per `M` when the hole sits still, `10 M` in 1¾ node-hours on `amdq` —
  with every fit valid, the track within `1.5e−3` cells, `M_irr` `0.3 %`
  low, `J` `1.0 %` high, `M_ch` `6e−4` low, and a masked error that grows
  and slows (`3.60` at `10 M`). At `5/128` it ends at `0.5–3.5 M`. There is
  no analytic control on this chart (the analytic core cuts the disk), so
  the control is the boosted **`a = 0`** hole, where `:damped` exists.
- **Regridding a `:fitted` case needs something new, and step 8 builds
  it.** `adapt = true` on a `:fitted` case chooses its mesh on the analytic
  `:damped` data of the same geometry and is refused by name where the
  analytic core meets the singular set — G5's own chart. Step 8 needs the
  indicator to flag on the fitted data (a callback reading the cache) or on
  the evolved state after the first chunk, and `regrid = true` along the
  trajectory; 8f's moving rows crossed a fixed capsule of fine blocks
  instead, which measures the layer and not the regrid. Decide, build,
  record.
- **The moving hole, measured on `boost(Harmonic(1, 0), 0.3 x̂)`** from
  `x = 0.75` on 1128 blocks at `h = 5/128`, `cfl = 1/5`: `:fitted` holds
  it at `4/M` to `5 M` with the track within `0.022` cells of the analytic
  center and no projection hit; the analytic `:damped` layer at `4/M` ends
  at `1.0–1.5 M` because its frozen core is released on the trailing side
  after about `M` and needs **`ρ_max ≳ 20/M`** (proposed in 8f) — use that
  rate for the analytic control, not the default. Every survivor's masked
  error grows by about `0.08/M`, harmonic truncation at `5/128`. The step
  is now sized from `λ (λ_end/λ)²` of the previous chunk (`driver.jl`),
  because a crossing hole's fastest speed grows `0.1–0.3 %` a chunk and the
  CFL recheck otherwise stops the run at any `cfl`; the recheck stays.
- **Coasting** (the finder off for five chunks) costs `1 %`; the sixth miss
  ends the run by `TrackLostError` at `max_misses`, as designed.
- **The record has a `variant` row, the drift reads evolved points only**
  (an oblate horizon's band reaches into the layer), and `evolve!` has
  `handover` (the analytic layer until then, the fit after) and
  `target_source = :snapshot` (a control that fails at `8 M`).
- **Costs to plan with.** A spinning `:fitted` run's error is the fit's
  (`L = 12` residual `2.5 %` on Kerr-Schild `a = 9/10`, `41×` the analytic
  control's masked error); a cache fill at `L = 12` is `420 ms` on 1632
  blocks; the evaluator doubles from `L = 8` to `12`.
- **Symmetry mechanics.** `.claude/orchestration/symmetry-run.sh` now takes
  `PART=amdq TLIM=24:00:00` in the environment, `~` for a space in a
  `hole:` job, and does not precompile on the login node (its image made
  jobs starting together race to rebuild). Give subprocess workers
  `OPENBLAS_NUM_THREADS=1` and stagger job starts; `hole_runs.jl generic`'s
  `budget=`, `tag=`, `t_end=` options and its one-row-per-chunk logs are
  the model for step 8's section.

Changes: the boosted harmonic Kerr case (`boost(Harmonic(M, a), v)`,
`|v| ≈ 0.3`, `a = 7/10`), run as `:fitted` on the tracked geometry;
`refinement_buffer` from `|v| · chunk`; the regrid path for a `:fitted`
case along the trajectory (above); the refinement centroid against the
analytic center in the record; the horizon finder along the trajectory;
a uniform-mesh control run at the finest spacing; a section `moving` in
`hole_runs.jl` for the rows that are Symmetry jobs.

Accept: as `CODE.md` G5 — the indicator's refinement follows the hole
with its centroid within a few finest spacings of the analytic center
at every chunk; the layer follows the center with the radius assertions
holding at every regrid; the Dirichlet data exact at the boundary (the
analytic solution there, checked); the masked error at the static run's
level over the crossing and converging at order `q` on the frozen
hierarchy; the adaptive run matching the uniform control at fewer
points; the interior residual at truncation, with points the core
releases relaxed within `1/ρ_max`; `:frozen` measured and its failure
recorded; the horizon found along the trajectory with `J` and the
boost's contraction recovered. Mark G5 *(Done.)*.

## Step 8′ — The trailing side (added 2026-09-24, after step 8's report)

`CODE.md`: "Open questions", "The moving layer's trailing side"; "Measured
results", "The moving hole (step 8)" (the screens and G5's crossing);
"The interior", piece 13. One bounded round by step 8's agent on its own
branch, with the reviewer's finding: G5's items 5 and 8 are not met, and
the three levers `CODE.md` names are untried.

Changes: the three levers, each behind a switch that leaves a run without
it bit for bit — a ramp whose width depends on the side (`ρ` rising faster
where the depth *decreases* along `v`, the trailing side), a target refill
on a cadence finer than `h/(4|v|)`, and relaxing released points toward
the target's evolved continuation (the cache advected with `F` off) — and
`hole_runs.jl moving`'s screens extended by them. Screen on the boosted
`a = 0` hole in the `5/2 M` box to `M/2` first, where a row is minutes at
four threads and where the analytic `:damped` control at `20/M` gives the
truth of what the layer should export; then the surviving lever(s) on G5's
chart to `M/2` and, if one halves the trailing side's excess there, the
crossing to whatever `t_end` a node-day allows. Read and record the three
step-8 jobs that were unread when the tunnel closed (`moving/g5-2`,
`moving/g5a-solo`, `conv-h7-N16`) as soon as Symmetry answers. Diagnose the
`SIGBUS` only as far as one attempt with `--heap-size-hint` and one with
`OPENBLAS_NUM_THREADS=1`/`JULIA_NUM_GC_THREADS` on the uniform control;
record the outcome either way.

Accept: G5's items 5 and 8 re-measured with each lever, and either met —
then G5 *(Done.)* — or not, with the trailing side's excess per lever in
`CODE.md`'s open question and a recommendation between the interior and
excision's price (step 8g); the suite green at one and four threads; the
step-8 record completed from the unread jobs.

## Steps X1–X3 — Excision for static holes (added 2026-10-05)

Erik reopened excision on 2026-10-05: as an interior variant, `:excised`,
beside the layer and not instead of it, for **static holes first**.
- The design is `CODE.md`'s "Excision (added 2026-10-05)". Read it, and
  "The tracked geometry", step 5's layer, "Kreiss–Oliger dissipation" (step
  8a's analysis), "The margin", and "Robust stability on the octant".
- Steps X1 → X2a → X2b → X3 run in that order, one agent each. Each works
  in its own worktree on `claude/step-xN-<slug>`, branched from the
  integration branch `claude/excision-singularity-handling-30feec` (not
  `main`). Each is reviewed and fast-forwarded into the integration branch
  before the next starts.
- Nothing goes to `main`.
- The ground rules above hold, with "off `main`" read as "off the
  integration branch".

### Sharp edges for steps X1–X3

- **Every simulation writes SimWatch status** (Erik's request):
  `test/octant_runs.jl` writes `simwatch.toml` in its `out` directory, and
  any new run script does the same through `SimWatchWriter`
  (`src/simwatch.jl`).
- **The exterior's operator is unchanged bit for bit.** A point none of
  whose stencil taps is excised is evolved by today's centered code.
  `isequal`, not a tolerance, says so. A change that makes it a tolerance
  has broken the design.
- **No kernel reads an excised value**, not even with weight zero:
  `0 · NaN = NaN`. Tests plant a degenerate metric (`h = −η`) on every
  excised point and require every non-excised `du` to be `isequal` to the
  clean run's.
- **The per-point classes are the single source of truth** for what is
  excised, in every kernel, for the life of a problem. No kernel re-asks
  the geometry at a stage's time.
- **No fourth state writer.** The `:excised` step limiter is a no-op; the
  RHS kernels write no `diag`.
- **`diag` slots are appended** after `NDIAG = 22`, contiguous, never
  inserted.
- **`Val`s per chunk, not per stage**, and the `Core.Box` rule — each
  captured name assigned once — in every kernel body. No `return` in a
  kernel.
- **Excision's parameters are fields of the interior or its spec**, so that
  `run_recipe` carries them through `repr(case.interior)`. A new `evolve!`
  keyword makes every old checkpoint refuse to restart unless
  `check_recipe` reads its absence as the default.
- **Symmetry**: `.claude/orchestration/symmetry-run.sh` and
  `symmetry-status.sh`, and the `symmetry-hpc` skill.
  - Give each study its own remote directory.
  - Stagger job starts.
  - Every run longer than an hour gets `checkpoint=` and `walltime=`.
  - Use `ssh -n` inside loops.
  - Wait with a polling loop; never end the turn to wait for a job.
  - H200 runs use a copy of the worktree with `CUDA` added to its
    `Project.toml` (CLAUDE.md, "The octant runs").

## Step X1 — The host-side analysis: closures, outflow, and the go/no-go

`CODE.md`: "Excision", "Kreiss–Oliger dissipation", "The margin";
`notes/methods-ghso2.md` (boundary classes, the sonic-surface recipe) and
`notes/sonic-surface.md`. **No kernel change.**

**Changes.**

1. **The closure weights** in `src/stencils.jl`, all built on
   `lagrange_derivative_weights`, in `Rational`:
   - the closure nodes and weights for `∂` and `∂²` at a point with
     `k⁻, k⁺ ∈ 0…G` non-excised points on each side, capped at reach `G`;
   - the dissipation's closure options: reduced rank, one-sided, or
     Mattsson–Svärd–Nordström's boundary-modified form;
   - the lopsided advection stencils;
   - a host table rounded once into `T` (`T(num)/T(den)`) for the kernel
     step to use.

   The starting family: centered when `min(k⁻, k⁺) ≥ q/2` (`≥ G` for the
   dissipation), otherwise the most nodes inside `[−k⁻, min(k⁺, G)]`, which
   gives orders `(2, 1)` at `q = 2` and `(3, 2)` at `q = 4` for
   `(∂, ∂²)`.

   Exact claims in `test/stencils_tests.jl`:
   - every closure is exact to its degree and not one further;
   - its nodes lie inside `[−k⁻, k⁺] ∩ [−G, G]`;
   - the full-width code is `rational_derivative_weights`/
     `rational_dissipation_weights`;
   - the table's centered rows are `derivative_weights(T, …)` bit for bit
     at `Float64` and `Float32`;
   - the dissipation closures have the damping sign in the norm the step
     chooses.
2. **`test/excision_model.jl`**, a standalone script in `dispersion.jl`'s
   manner (no `Test`, Markdown tables, `key=value` options selecting
   sections), in three sections:
   1. **`margins`**: Kerr-Schild `a = 0, 3/5, 9/10` and harmonic `7/10,
      9/10`, on the seed's offset surfaces at depths of `m` cells for the
      resolutions the octant runs use. Read the coefficients through
      `background_state` and `metric_quantities`, as `dispersion.jl`'s
      `frozen_coefficients` does.
      - The outflow margin along the true normal, and its least value over
        the surface.
      - The per-axis ratios over the lego surface's closure faces: their
        distribution and the inflow-like fraction. The prediction for
        Kerr-Schild `a = 0` is `r_E/(2M)`.
      - The clearance of the chart's singular set.
      - The answer per chart: the shallowest surface with normal outflow,
        and for Kerr-Schild `a = 9/10` whether there is room between the
        ring and the horizon at all.
   2. **`model1d`**: the frozen-coefficient system of `dispersion.jl` and
      the variable-coefficient radial line of its `model_run` (Kerr-Schild
      `a = 0`), with the `w = 0` core replaced by a closure at `r_E`. Run
      at `q = 2, 4`, `b/a` from below 1 (inflow along an axis) to 2,
      closure widths, the dissipation's closures, and centered against
      lopsided advection. Report:
      - the dense semi-discrete spectrum's largest `Re λ`;
      - the RK4 step the closures allow;
      - what the closure reflects (amplitude, group velocity);
      - a pulse into the surface and what crosses the horizon, against
        `model_run`'s layer.
   3. **`model2d`**, **the go/no-go**: the principal part of one component
      on Kerr-Schild's equatorial plane — the anisotropic `γ^{ij}`, so that
      an axis at angle `θ` to the normal has
      `b/a = H cos θ / √(1 + H sin²θ)` — with a lego circle of radius `r_E`
      and nested mixed derivatives.
      - Dense eigenvalues at `48²`–`64²`, and long noise evolutions on
        finer grids.
      - It compares per-axis closures with per-stencil extrapolation along
        the lattice direction nearest the normal (sources inside the
        point's `G`-box), each with and without lopsided advection, at
        `ε_KO = 1/2` and `1`, over `r_E` from `M/2` to `3M/2`.

**Records**, in `CODE.md`'s "Excision" and a new Measured results entry,
"Excision: the analysis (step X1)":
- every table;
- the closure family and the dissipation's closure;
- whether to upwind, and with what depth profile;
- the least depth in `M` and in cells, per chart;
- the step's cost.

**Accept:**
- the weights exact;
- the three sections' tables recorded;
- a **go/no-go with the two-dimensional numbers**: which family is stable
  on a lego circle, and from which `r_E`. If none is, the report says so,
  and X2a and X2b do not start until Erik has read it;
- the suite green at one and four threads.

**What step X1 hands over** (its report's section 6, 2026-10-05; the
numbers are `CODE.md`'s "Excision" and "Excision: the analysis (step X1)").
X1 found a **go** for the static Kerr-Schild `a = 0` hole with per-axis
closures and `:msn` dissipation, at every `r_E` from `M/2` to `7M/4`, with
`ε_KO > 0` required. Spinning holes are not covered.

- **For X2a — the advection must be a provider method of its own.** X2b's
  optional lopsided advection replaces the centered `D₁` in the two
  advective terms `β^k ∂_k h` and `β^k ∂_k Π` — and only there. So the
  provider takes the shift's component:
  - `adv(S, β_d, ∂f_d, work, base, d)` for both fields;
  - the centered provider returns the `∂f_d` it is handed, so the main
    kernel forms no new stencil and stays bit for bit.
- **The closure family is the per-axis one, as built**: `closure_table(T,
  Val(q); dissipation = :msn)`, with the mixed derivative nested.
  - The table layout: `d1`, `d2` and `ko` are `[slot, k⁻ + 1, k⁺ + 1]` on
    the slots `−G … G` (slot `j + G + 1`, zeros outside each closure's
    nodes).
  - `lop` has a fourth index for `up = −1, +1`.
  - `d_lo/d_hi`, `ko_lo/ko_hi` and `lop_lo/lop_hi` (`Int8`) are the nodes
    each closure reads, and `admissible` is the refusal.
  - **Contract over `d_lo:d_hi` in ascending order**, as `axis_stencil`
    does: the centered rows are `derivative_weights` bit for bit, so "no
    excised tap ⇒ the centered `F`" can be `isequal`.
- **The table is 4624 bytes at `q = 4`, `Float64`.** That is above CUDA's
  classic 4 kB kernel-parameter limit, so pass it as a **device array**
  (`to_backend`), not as an `isbits` argument.
- **The dissipation's closure is `:msn`**, the only one negative
  semidefinite in `l²`. It reads `[−k⁻, G]`.
- **`ε_KO > 0` at the surface is part of the variant.** Refuse `:excised`
  with `ε_KO = 0`, or with a profile that vanishes at the surface, saying
  why: without dissipation the extrapolation family grows `+1–4/M`, and the
  per-axis one as the interior does.
- **Spinning holes are refused in this round.** On their lego faces frame
  dragging can turn the shift *into* the excised set along an axis
  (`b/a < 0`, 2–18 % of the faces), and there the closure is unstable on
  the frozen line, `0.03–0.19/h`.
  - The refusal is the physics, not the spin: `build_excision` computes, at
    every band point and closure axis, the shift's component toward the
    excised side from the state it is built on, and refuses any negative
    one by name.
  - The record carries the count every chunk.
- **The lopsided advection is an option, off by default.** It is a `C²`
  blend (`smoothstep`) in the depth below the horizon: zero at `1` cell
  below, full from `5` cells (`start = 1`, `width = 4`), the profile X1
  measured.
  - With it on, every point from the surface to one cell below the horizon
    is non-centered. That is a thick shell (about 23 cells at `h = 1/16`,
    `r_E = M/2`), not the zone of a few cells the brief sized.
  - So **the blend lives in the main kernel's `:excised` specialisation**,
    through the provider's `adv`, gated by a `Val`, so that no other
    variant's kernel changes. The zone kernel keeps the band points only,
    with the blend applied there too.
  - It costs RK4's step: the plane's stable `cfl` falls from `1.86` to
    `1.52–1.79` at `q = 4`, which `cfl = 1/2` absorbs.
- **The least depth.** `check_interior_radii`'s `:excised` method asserts
  `m ≥ G + 1`. A case with a `Horizon` asserts `m ≥ ⌈√3 G⌉`, 6 cells at
  `q = 4`: the finder's footprint reaches `√3 G h` on a diagonal and must
  not touch the excised set.
- **For X3:**
  - the depth window at `h = 1/16` is `r_E = M/2 … 13M/8` (6 to 24 cells);
  - run every depth with and without the lopsided advection;
  - `ε_KO = 1/2`;
  - X1's plane reached an inflow-like fraction of `0.73` against `0.875` on
    the 3D sphere at the same `r_E`. The go rests on the kind of face
    (`0 ≤ b/a < 1`), which both share, and X3 is the 3D test.
- **Still open:** the gauge drift, which X3 measures.

## Step X2a — The stencil provider (one copy of the physics)

`CODE.md`: "One right-hand-side evaluation", "Finite-difference stencils",
"Excision". **Only `src/evolution.jl`** (and tests).

**Changes.** `gh_rhs_at_point` takes a stencil *provider* `S` with methods
`d1(S, work, base, d)`, `d2`, `dmix(S, work, base, i, j)` (outer sum along
`i`, inner along `j`, as now), `ko`, and — amended after X1 —
`adv(S, β_d, ∂f_d, work, base, d)`. `adv` is the derivative that multiplies
`β^d` in the two advective terms, `β^k ∂_k h` and `β^k ∂_k Π`, and only
there.
- `Centered{T,q}` holds the strides and the three `@generated` weight
  vectors.
  - Its methods are today's `axis_stencil`/`mixed_stencil` calls, in
    today's order.
  - Its `adv` returns the `∂f_d` it is handed, so the main kernel forms no
    new stencil.
- X2b's lopsided blend is another `adv`, which reads the sign of `β_d`.
- The existing `gh_rhs_at_point` signature builds `Centered` itself, so
  `gh_rhs_kernel!`'s call site is unchanged.
- No other provider yet: X2b adds the closure one.

**Accept** — the refactor is invisible:
- `test/thread_workload.jl`'s output and a one-chunk `test/octant_runs.jl`
  CSV are identical, character for character, to the integration
  branch's, at one and four threads;
- every existing `isequal` claim holds;
- `bench/stepping.jl` (`wave`, `hole`) is within noise of the base, run
  back to back on the same machine;
- the suite green at one and four threads.

The H200's register count and `ld.local` spills for the `:damped` `q = 4`
kernel are measured in X3.

**What step X2a hands over** (its report's section 6, 2026-10-05; the
interface is `CODE.md`'s "One right-hand-side evaluation", "The stencils come
from a provider"). The refactor changed no bit: the thread digest, one-chunk
octant CSVs, eleven CPU and three Metal kernel variants are identical to the
base.

- **A closure provider subtypes `TreeGeneralizedHarmonic.StencilProvider`**
  and implements all five methods with exactly these signatures:
  - `d1(S, work, base::Int, d::Int)`, `d2(…)` and `ko(…)` — raw
    contractions on unit spacing;
  - `dmix(S, work, base::Int, i::Int, j::Int)` — the same, outer sum along
    `i`, inner along `j`;
  - `adv(S, β_d, ∂f_d, work, base::Int, d::Int)` — handed and returning a
    **scaled** derivative, so a lopsided `adv` carries its own `1/h`.

  The caller applies `1/h`, `1/h²` and `ε_KO/h` to the first four as before.
- **Calling it.** The zone kernel calls `gh_rhs_at_point(S, T, work, Hwork,
  inner, b, var, sv, inv_h, γ0, γ2, εh, Val(HASH), Val(DISS))`, with `var`,
  `sv`, `inv_h`, `γ0` and `εh` computed exactly as `gh_rhs_kernel!` does
  them. `q` enters only through the provider.
- **What it is asked for, per point:**
  - `d1` of all 20 fields along each axis (`h`'s feed the coefficients too);
  - `adv` of all 20 along each axis;
  - `d2` and `dmix` (pairs `(1,2)`, `(1,3)`, `(2,3)`) of `h` only;
  - `ko` of all 20.

  `base` is the component's linear index in `work`, so **the per-point
  codes live in the provider**: `base` cannot be decoded into them. The
  provider holds:
  - the point's `k±` along the three axes;
  - the class array, with the working array's spatial strides, plus the
    point's index in it and `st`, for `dmix`'s per-outer-node `j`-codes;
  - the table (a device array);
  - `1/h`;
  - the blend weight.
- **The lopsided blend** is a provider that wraps `Centered` and overrides
  only `adv`. It is used in the main kernel's `:excised` specialisation,
  behind a `Val`, and in the closure provider.
  - Where the weight is zero it must return `∂f_d` through a **branch**,
    not as `(1 − λ) ∂f + λ L`, to keep the exterior bit for bit.
  - Price the centered `d1/h` of `Π`, which is still formed where the blend
    is full, across the main kernel's thick blend shell.
- **Traps:**
  - The module-level names `d1`, `d2` and `ko` collide with common locals
    (`constraints.jl`, `interior.jl`, `ClosureTable`'s fields). A function
    that calls them must not have a local of the same name.
  - Providers are `isbits`, with `@inline` methods.
  - `test/evolution_tests.jl`'s host-only `ProbeProvider` is a template for
    the closure provider's host tests.
  - "No excised tap ⇒ the centered `F`" was bit for bit on Apple silicon
    for a non-inlined provider. Claim `isequal` where it holds and keep a
    `64 eps` fallback for x86-64.
- **Benchmarks:** at this machine's load the base alone varies 6–8 %, and
  the refactor measured ±3 % with opposite signs in the two cases (read as
  code layout, proposed). Interleave base and branch runs when the zone
  launch is added.
- **Environment:**
  - a fresh resolve takes TreeAMR **0.1.7** (`[compat]` `0.1.4` allows it),
    and the suite passes with it;
  - running a single test file needs a scratch environment combining
    `test/Project.toml`, the four `[sources]` and a path source for the
    package;
  - a Metal check needs a fresh resolve: the current Manifest pins LLVM 10,
    which Metal 1.11.1 cannot use.

## Step X2b — The `:excised` variant

`CODE.md`: "Excision" (and its "On the mesh, and on a device"), "The
tracked geometry", "The range projection", "Analysis quantities",
"Checkpoint and restart". Starts from X1's recommendation and X2a's
provider. **Read "What step X1 hands over" above first.** Where it and the
list below differ, it wins:
- the per-axis family with `:msn`, from `closure_table`, as a device array;
- the refusals of `ε_KO = 0` and of a shift pointing into the excised set;
- the lopsided blend in the main kernel's `:excised` specialisation;
- `m ≥ ⌈√3 G⌉` with a `Horizon`.

**What a review of the code found** (2026-10-05). Each item is a thing the
naive design gets wrong:
- `FittedInterior` refuses `thickness = 0` and `fitted_interior` refuses
  `n_L = 0`; `check_interior_radii` asserts a least thickness, and
  `horizon_floor_level` divides by it. For `:excised` the thickness is
  only the core rule's depth, or zero where the code is taught it.
- `chunk_interior` calls `with_ρ_max(…, factor/dt)`, which fails with no
  factor. `:excised` has no rate, so `chunk_interior` returns its interior
  unchanged.
- `discrete_gradient_momentum!` (the `Π` post-pass) takes centered
  stencils everywhere: refuse it for `:excised`.
- `in_layer` must be false everywhere for `:excised`. `layer_mask` and
  `shell_mask` become the band `[r_E, r_E + W)` and the `G h` beyond it.
- `InteriorMask` compares squared radii and `is_outside` compares a
  `sqrt`. Build the excised bit from **one** predicate, the masks'
  `is_evolved(interior_mask(int, t), x)`, so that the classes, the norms,
  the speed and the horizon guard exclude the same set.
- `check_recipe` compares every recipe field (see the sharp edges).

**Changes.**
1. **`src/interior.jl`**:
   - `:excised` in `INTERIOR_VARIANTS`.
   - On `Interior`, excised where `r < r_1`; `r_0` is the core rule's
     radius only.
   - On `FittedSpec`/`FittedInterior`, excised where `d > 0`.
   - `is_frozen = !is_outside`, and `in_layer = false`.
   - `check_interior_radii`'s `:excised` method: `m ≥ G + 1`,
     `r_1 ≤ r_h,min − m h`, the singular set inside the core surface, and
     no thickness check.
   - The excision parameters — the upwind blend's start and width below
     the horizon in cells (0 off), and the dissipation's closure kind — as
     fields, per the sharp edges.
2. **The geometry is frozen for the run** (proposed in review): built once
   from the case's sphere or the seed's shape, with the center's velocity
   zero.
   - A tracked case still finds and tracks every chunk, for the record and
     for an assertion that the found horizon stays `m h` outside the
     surface.
   - A restart takes the same frozen geometry, so there is no new carried
     state.
   - `regrid`, `adapt`, `bounds`, `handover`, `target_source`, the rate
     keywords and the `Π` post-pass are refused for `:excised`, each
     saying why.
3. **New `src/excision.jl`**:
   - `ExcisionData`: the classes (`UInt8`, stored points), a device `Bool`
     vector of blocks that touch the zone, the weight table, `W`, `h`, the
     geometry it was built for, and the counts.
   - `build_excision` makes three passes:
     1. the excised bit on owned points;
     2. one `fill_ghosts!` of a one-variable `FieldSet{T}` with even
        parity, so that every ghost equals its owner and the octant's
        walls mirror it;
     3. a `stored = true` pass writing the class (centered, excised, zone)
        and refusing a zone point with no admissible closure — excised on
        both sides of one axis within reach.
   - `check_excision_mesh`: every leaf meeting `r_E ± (G + q + 2) h` is on
     one level.
   - The zone kernel with the closure provider. Its contractions run over
     the closure's own nodes only; neighbour codes are read for the nested
     mixed derivatives; the lopsided advection blend comes in where the
     interior asks for it.
   - The record-time outflow monitor and its rows: the least normal
     margin, the least per-axis ratio, the inflow-like count, and the
     band's point and non-finite counts.
4. **`src/evolution.jl`**:
   - `GHProblem` gets an `excision` field. The constructor builds it for
     `:excised`; `with_interior` rebuilds it only when the geometry
     differs, carries it otherwise, and drops it for other variants.
   - `gh_rhs_kernel!` gets the classes as an argument (`nothing`
     otherwise). Its `:excised` branch computes `F` at centered points,
     writes `0` at excised ones and nothing at zone points.
   - `gh_rhs!` launches the zone kernel right after the main one.
   - `gh_step_limiter!` gets a no-op `:excised` method.
   - `monitor_mask(p, t)` — the excised set widened by `W =
     max(G, ⌈√2 q/2⌉) h`, plus slack for the shape's slope — is the
     default mask of `gh_constraint!`, `adm_constraint!` and the indicator
     (a `mask` keyword on `indicator_flags`). The error, `max_speed`,
     `evolved_nonfinite`, `validity_rows` and the horizon guard keep
     `interior_mask`, which reads the band.
5. **`test/octant_runs.jl`**:
   - `interior=excised` with `geometry=sphere|tracked`, `r_E=` or
     `margin=`, `upwind=<start>,<width>` and the dissipation closure;
   - the `in` shell starts at `r_E + W`;
   - the noise excludes the excised set;
   - the CSV, `records.csv` and SimWatch's `setup` and `extra` get the
     excision rows.
6. **`bench/stepping.jl`**: `BENCH_CASE=excised`.

**Tests**, a new `test/excision_tests.jl` on the `q = 2`, `N = 8` hole
fixture with `r_E = 3/4`, each testset a claim:
- the classes are the geometry's, the ghosts their owners', the zone the
  enumeration's;
- the closures are exact on polynomials across faces, edges and corners
  (host calls of the provider);
- with no excised tap the closure provider's `F` is the centered one —
  `isequal` if it is, otherwise recorded and held to `64 eps`;
- with a degenerate metric planted on every excised point, every
  non-excised `du` is `isequal` to the clean run's and every excised `du`
  is exactly zero;
- the centered points' `du` is `:none`'s, to `512 eps` (two kernel
  specialisations), and the zone points' is not;
- an excised sphere and a `FittedInterior` holding it give identical
  classes and `du`;
- the monitors are finite with `NaN` in the excised set, the non-finite
  count is zero, and the horizon guard refuses a footprint that reaches an
  excised point;
- the refusals: mixed levels at the surface, the singular set, the margin,
  `regrid`/`adapt`/`bounds`;
- a two-chunk run to `M/5` with finite rows and a positive normal margin,
  and a restart after the first chunk `isequal` to the run.

Elsewhere:
- one `:excised` right-hand side in `test/thread_workload.jl`'s digest;
- one `Float32` `:excised` right-hand side in `type_tests.jl`.

About 60 s at one thread. Price it in the report, as `driver_tests.jl`'s
is.

**Accept:**
- all of the above;
- the suite green at one and four threads, with its times;
- the zone kernel's cost per point and as a share of a right-hand side on
  the CPU;
- a local smoke run that writes `simwatch.toml`:
  `test/octant_runs.jl case=ks interior=excised L=8 N=16 roots=2 radii=4,2
  t_end=1 out=…`;
- `CODE.md`'s "Excision" amended with what was built, marked
  **(amended in step X2b)**.

**What step X2b hands over** (its report's section 6, 2026-10-05; the
numbers are `CODE.md`'s "Excision", "What step X2b built", and Measured
results, "Excision: the variant (step X2b)").

`:excised` runs on both geometries:
- every other run is unchanged bit for bit;
- the exterior's `du` is `:none`'s exactly at all 55 505 centered points of
  the fixture;
- a degenerate metric planted in the excised set changes no non-excised
  `du`.

Its run to `M/5` matches `:damped`'s error outside `r = 23/20` to 0.1 %. The
zone kernel costs 620 ns a zone point at four threads, 0.64 % of a
right-hand side; the blend adds 0.3 %.

- **The depth scan** (`h = 1/16`, about `10 M` a row), with and without
  `upwind=1,4`:

  ```
  julia --project=. --threads=<n> test/octant_runs.jl case=ks interior=excised geometry=sphere L=64 N=64 roots=2 radii=32,16,8 r_E=<1/2|3/4|1|5/4|3/2|13/8> t_end=10 chunk=1 cfl=1/2 [upwind=1,4] backend=cuda out=<dir>/rE<r>[-up] checkpoint=<dir>/ck-rE<r>[-up] walltime=<s>
  ```

  - The defaults give `q = 4`, `eps = 1/2`, the algebraic source, the finder
    every chunk with the spin, noise `1e-8` and `:msn`.
  - `r_0 = r_E/2` and the margin `m = ⌊(2 − r_E)·16⌋` (24 down to 6 cells)
    are derived.
  - `r_E = 13/8` is the shallowest the finder allows at `h = 1/16`
    (`m = ⌈√3 G⌉ = 6`).
  - **Pass `cfl=1/2`**: the script's default is `1/4`.
- **The production rows:** the same with `N = 64, 96, 128` (`h = 1/16, 1/24,
  1/32`), the chosen `r_E`, `t_end=24`, and one row to `50`.
  - The tracked alternative is `geometry=tracked margin=<cells>`.
  - The surface's `(G + q + 2) h` neighbourhood lies inside the finest cube
    `[0, 8]³` for every `r_E` in the window.
- **Compare on the shells outside the horizon, not on `err_l2`.** The
  record's masked error now counts the band, where the one-sided closures'
  truncation error is largest: on the `q = 2` fixture `err_linf` is `0.5`
  there against `0.04` outside `r = 1.15`. The CSV's shells `[2, 2.25)`,
  `[2.25, 3)`, … are what `CODE.md`'s `:damped`/`:fitted` tables use.
- **CUDA at `Float64` is untested.** The kernels compiled and ran on Metal
  at `Float32`, within `6.5e−5` of the CPU's scale. Run a short CUDA smoke
  row first:
  - the classes are `UInt8` device arrays;
  - the closure table is a `NamedTuple` of `CuArray`s;
  - the kernels to inspect for registers and spills are `gh_rhs_kernel!`
    specialised for `:excised` and `gh_zone_kernel!`, beside the `:damped`
    `q = 4` kernel (X2a's check).
- **On the H200 the zone kernel launches over all blocks.** In the bench,
  32 of 512 blocks do work. A subset launch stays a TreeAMR wish.
- A spinning Kerr-Schild hole is refused at build time by the shift's sign
  (checked at `a = 3/5`). Spinning rows are not part of X3.
- **The monitor rows in the CSV and `simwatch.toml`'s `[extra.excision]`:**
  - `excision_band` and `_band_nonfinite`;
  - `_normal_min`, the outflow margin along the true normal;
  - `_faces`, `_axis_min` and `_inflow`, X1's per-axis faces;
  - `_into` (shift into the excised set, must stay 0);
  - `_horizon_margin` when tracked; it ends a run below `m − G/2`.

## Step X3 — The static hole on the octant, on Symmetry's H200

`CODE.md`: "Excision", "Robust stability on the octant" (the `:damped` and
`:fitted` tables this step is compared with). Runs through
`test/octant_runs.jl`; each row has its own `out=` (the CSV, `records.csv`
and `simwatch.toml`) and its own `checkpoint=` with a `walltime=`.

**Rows**: Kerr-Schild `a = 0` on the octant, with the algebraic source,
`q = 4`, `cfl = 1/2`, `ε_KO = 1/2`.
1. **The depth scan** at `h = 1/16`, about `10 M` a row: `r_E` across X1's
   window, with and without lopsided advection. This gives the stability
   boundary.
2. **At the chosen depth**: `h = 1/16`, `1/24` and `1/32` to `24 M`, and one
   row to `50 M`, on the mesh of the exterior study (the octant `[0, 64]³`,
   root brick `2³`, cubes `32, 16, 8`, `N = 64, 96, 128`). Report:
   - the shells' `ℋ` and error and their orders;
   - `M_irr`, `dM_irr/dt` and the drift of `h_tt`;
   - the band inside the horizon;
   - the outflow rows;

   against the recorded `:damped` and `:fitted` rows.
3. **Cost**: the zone kernel's share of a right-hand side on the H200, and
   the PTX register and spill counts of both kernels (X2a's check).

**Records**: a Measured results entry, "Excision on the static hole (step
X3)", and the recommendation:
- whether excision is feasible here, and at what depth;
- whether its exterior matches `:damped`'s and beats `:fitted`'s;
- whether the gauge drift returned;
- what the moving round and the spinning holes need.

**Accept**: every row run or its failure diagnosed; the tables in
`CODE.md`; the suite green at one and four threads.

## Steps X4–X7 — Excision of a static spinning hole, `a = 3/5` (added 2026-10-06)

Erik's decision of 2026-10-06, after X3: go on to the spinning hole, **at
`a = 3/5` as the stepping stone** (not `9/10`), **with sufficient
resolution**, and **bring `main` into the integration branch first**.
`main` carries the GPU kernel rewrite: PR #4, 92c7f0f, "Evaluate the
right-hand side without spills or calls on a device".

Two things stand between the static round and this one:
- **X1 found the per-axis closures unstable** on the lego faces where frame
  dragging turns the shift *into* the excised set along the face's axis
  (`b/a < 0`, 2–14 % of the faces at Kerr-Schild `a = 3/5`). X2b refuses
  such a case at build time.
- **The rotating octant**, the only octant a spinning hole fits, is on the
  branch `claude/octant-mode-spinning-bh-75bd22` (0a96f27), not on the
  integration branch. That branch's `CODE.md` holds the `a = 3/5` reference
  rows: `:damped` and `:fitted` at `h = 1/32` to `64 M`, and the
  convergence pair `1/16`, `1/24`. They are what X7 is compared with.

The steps:
- **X4** (sync) and **X5** (host analysis) are independent and run as two
  agents at once, like 8a and 8b.
- **X6** starts from both, merged into the integration branch.
- **X7** runs on Symmetry's H200s, with SimWatch status for every run.

The sharp edges of steps X1–X3 hold throughout.

## Step X4 — The integration branch on `main` and on the rotating octant

`CODE.md`: on `main`, the GPU kernel work's sections (read `git show
origin/main:CODE.md`). On the integration branch: "One right-hand-side
evaluation" (the stencil provider) and "Excision". On the rotating-octant
branch: its "rotating octant" section and its `a = 3/5` and `a = 9/10`
records.

**Changes.**
1. **Merge `origin/main` (cbe3662) into the step branch.** The conflict that
   matters is `src/evolution.jl`. `main`'s spill-free right-hand side and
   the excision round's stencil provider (X2a), main-kernel `:excised`
   branch and zone kernel (X2b) must become one design:
   - `main`'s device performance is kept for every existing variant;
   - the excision invariants hold: the exterior bit for bit against
     `main`'s own centered kernel, no excised value read, no fourth writer;
   - the zone kernel may stay generic, since it is a thin shell.

   Where the provider cannot be `main`'s arithmetic bit for bit, say why
   and hold it to roundoff. Resolve `CODE.md` and `CLAUDE.md` by keeping
   both sides' content.
2. **Then merge `claude/octant-mode-spinning-bh-75bd22` (0a96f27)**, its
   committed head only.
   - **Do not touch its worktree** (`.claude/worktrees/
     optimistic-ramanujan-852fdf`), whose uncommitted changes belong to
     another session.
   - Reconcile `hole_case`/`GHCase` (`octant = :rotating`, `a`, beside
     `excision`), `test/octant_runs.jl` (`octant=rotating a=` beside
     `interior=excised …`), `CODE.md` and `CLAUDE.md`.
   - Make `build_excision`'s one-variable class field set correct on the
     rotating octant: its parity and its seam rotation, for a scalar. The
     classes must be consistent across the seam, and a test must say so.
3. **TreeAMR 0.1.7** (the seam) becomes the `[compat]` floor if the rotating
   branch needs it.

**Accept:**
- the suite green at one and four threads;
- the thread digest against `main`'s for the lines both have;
- the excision tests unchanged in strength;
- a static `a = 0` excised row on the rotating octant with the same
  classes and the same record as on the mirror octant, where the two are
  comparable;
- on an H200: the right-hand side's ns per point and the register and
  spill counts of `:damped`, the `:excised` main kernel and the zone kernel,
  against `main`'s recorded numbers, and whether X2a's `+5 %` is gone;
- `CODE.md` amended **(amended in step X4)**.

## Step X5 — The frame-dragged faces: a closure, and the go/no-go for `a = 3/5`

`CODE.md`: "Excision", "Excision: the analysis (step X1)", and the
rotating-octant branch's `a = 3/5` record (`git show
claude/octant-mode-spinning-bh-75bd22:CODE.md`). **No kernel change.**

**Changes** — `test/excision_model.jl` grows, and so does `src/stencils.jl`
if a new family needs weights (with exact tests).
1. **`margins` at Kerr-Schild `a = 3/5`**, for the sphere `r_E` and the
   tracked offset surface, at `h = 1/24, 1/32, 1/48`:
   - the window — normal outflow to the inner horizon, which is at `0.632`
     on the equator, with the ring at `0.6` and the core rule's surface
     outside the ring;
   - the `b/a < 0` face fraction and its least value against `r_E`;
   - the depth in cells at the poles (`r₊ = 1.8`) and on the equator
     (`1.897`) for each `h`.
2. **`model2d` on Kerr-Schild `a = 3/5`'s equatorial plane**, where frame
   dragging is in the plane, keeping X1's conservation form. Compare
   (a) per-axis closures (X1's), (b) per-stencil extrapolation along the
   lattice direction nearest the normal, with sources inside the point's
   `G`-box (X1's `extrap`), and (c) a **hybrid**: per-axis where the
   face's shift points out of the excised set, the extrapolation where it
   points in.
   - Each with and without lopsided advection, at `ε_KO = 1/2, 1` and
     `q = 4` (and 2), over the window's `r_E`, against the `:damped` layer
     on the same plane.
   - Dense spectra and noise evolutions, as X1's `eig` and `noise`.
   - Add another family if one of these fails and a better one is
     visible.

**Records**: `CODE.md`'s "Excision" and a Measured results entry,
"Excision: the frame-dragged faces (step X5)". The step decides:
- the family X6 builds, as a per-face rule X6 can implement from the
  per-point classes;
- the `r_E` window and the resolutions X7 should run;
- whether lopsided advection is needed here.

**Accept:**
- a **go/no-go with the plane's numbers**: which family is stable at
  `a = 3/5`, and from which `r_E`. If none, say so; X6 and X7 then wait for
  Erik;
- the suite green at one and four threads.

**What step X4 hands over** (its report's section 6, 2026-10-06; the
numbers are `CODE.md`'s "One right-hand-side evaluation", "One design after
`main`'s rewrite", "Excision", "What step X4 changed", and Measured results,
"The merge with `main` and the rotating octant (step X4)").

The integration branch is on `main`'s spill-free right-hand side and on the
rotating octant (TreeAMR `0.1.7`).
- `:damped` on the H200 is `main`'s kernel to `ptxas`'s report.
- The excised right-hand side is `2.88 ns` a point on the octant, `3.9×`
  faster than before.
- The zone kernel is `238 ns` a zone point, `1.5 %` of a right-hand side.
- The lopsided blend costs `+2.7 %`.
- X2a's `+5 %` is gone.

- **Where a closure provider and a per-face rule plug in.**
  `gh_zone_kernel!` (`src/excision.jl`) builds `closure_provider(T, Val(G),
  st, cls, cb, tab, inv_h, λ)` per zone point and calls `gh_rhs_store!(du,
  o, sd, S, T, work, Hwork, inner, b, var, sv, inv_h, γ0, γ2, εh,
  Val(HASH), Val(DISS))`.
  - `main`'s head asks for `d1` of all 20 variables, plus `adv` and `ko` of
    `h`. Each `Π` component then asks for `d1` of `h_v` again, `d1` of
    `Π_v`, `d2` and the three `dmix` of `h_v`, and `adv` and `ko` of `Π_v`.
  - A per-face rule is another row family chosen per (point, axis) inside
    `closure_provider`. It is decided at build time and stored in extra
    bits of the `UInt8` classes or in a second per-point array; the
    classes stay the single source of truth.
  - New table rows go in as extra fields of `closure_arrays`' `NamedTuple`.
  - The zone set is `_reads_excised` (`±G` along axes, `q/2` boxes for the
    mixed derivatives). A family that reads farther must widen it and
    `closure_admissible`.
  - The refusal to narrow is `build_excision`'s `ninto == 0` (census slot
    `EXM_INTO`). The outflow rows come from `_outflow_kernel!`.
- **What to claim bit for bit.** Any provider other than `Centered` changes
  how LLVM fuses `muladd`s into FMAs around the head.
  - Keep every non-zone point on the `:none` call, which also keeps a
    blend weight of zero bit for bit.
  - Claim only *contractions* as `isequal`.
  - Claim `F` to roundoff on each variable's largest `|du|`. X4 held the
    closure provider at points with no excised tap to `512 eps` (measured
    `103`) for that reason.
- **The nested mixed derivative is not symmetric** (measured in step X4).
  Outer `i`, inner `j`, with `i < j`, is not invariant under `x ↔ y` or the
  seam's quarter turn. At zone points the two nestings differ by `3·10⁻³`
  (`5·10⁻¹⁴` under `:damped`), so the rotating octant reproduces the mirror
  octant's excised `a = 0` record only to `1.4·10⁻³` in L∞ inside the
  horizon at `t = 1`. Averaging both nestings made the two agree to every
  printed digit on a scratch copy.
  - **X6 builds the symmetrized zone-point mixed derivative,
    `½(D_i D_j + D_j D_i)`** (proposed in step X4; taken into X6 by the
    orchestrating session on 2026-10-06, for Erik to confirm in review). It changes X3's validated zone operator at
    truncation level.
  - X6 re-measures what it changes: the fixture's excised run against X2b's
    record, the static `a = 0` excised row on the rotating octant against
    the mirror octant's, and one short `a = 0` octant row against X3's.
- **The rotating octant's mechanics:**
  - the seam is `(1,2)`: `R` takes `e_x → e_y` and `e_y → −e_x`, and
    `u(Rp) = Q u(p)` with `state_rotation`. There is a mirror at `z = 0`
    only;
  - every `FieldSet` over the forest needs `rotation=`; the classes' bit
    field set is a scalar, `identity_rotation(forest, 1)`;
  - the seam planes `x = 0` and `y = 0` are the same physical points;
  - ghost classes across the seam are the rotated owner's;
  - the census reads each owned point's own `β`, so it needs no rotation;
  - `hole_case(…; octant = :rotating, …)`;
  - `test/octant_runs.jl octant=rotating a=… interior=excised …`. With a
    spin it measures margins against `r₊`, and the core rule's default
    radius is `max(r_E/2, (r_E + a)/2)`.
- **The `a = 3/5` smoke** (`h = 1/16`, `r_E = 1`, `r_0 = 0.8`, `m = 12`) is
  refused at the build today: 72 pairs with the shift into the excised set,
  least `b/a = −0.499`. SimWatch writes status `failed` with that message.
  X6's rule lifts the refusal for what it covers.

**What step X5 hands over** (its report's section 6, 2026-10-06; the
numbers are `CODE.md`'s "Excision", "The frame-dragged faces (step X5)", and
Measured results, "Excision: the frame-dragged faces (step X5)").

**Go for Kerr-Schild `a = 3/5`**, with the rule `hybrid-adv`.
- On the equatorial plane it is stable at every `r_E` from `0.65`, just
  above the inner horizon, to the horizon, at `q = 2, 4`, `ε_KO = 1/2, 1`,
  `h = 5/48 … 5/96`, with and without the blend.
- Its rightmost eigenvalue is within `0.002/M` of the `:damped` layer's, and
  its noise decays at the layer's rate to `100 M`.
- The per-axis closures alone are unstable near the inner horizon, at
  `+1.2` to `+4.8/M` from `r_E = 0.65`. Extrapolating the advection on
  every closure axis is unstable too, at `+2.6` to `+18/M`, so the rule
  must be selective.
- `ε_KO > 0` is still required.

- **The rule (`hybrid-adv`).** At a zone point and axis `d`, it applies when
  a side `s` with `k_s < G` has `−s β^d < 0` in the build state: the shift
  points into the excised set along that axis. `d1`, `d2`, `dmix` and `ko`
  stay per-axis everywhere. **Only `adv`** along such an axis becomes
  `inv_h · Σ_j w_j ũ(x + j e_d)`:
  - `w` is the centered `D₁` weights;
  - `ũ = u` at non-excised taps;
  - at an excised tap `Q`, `ũ = Σ_i tab[i, k₀, n] · u(Q + (k₀ + i − 1) e)`;
  - `e` is the nearest of the 26 lattice directions to the outward normal at
    `Q`;
  - `k₀` is the first non-excised step along `e`, and `n ≤ 3` the
    consecutive non-excised sources inside the point's `G`-box;
  - `tab = extrapolation_table(T, Val(q))` (`src/stencils.jl`, `isbits`,
    `3 × G × 3`, 216 bytes at `q = 4`), which covers every
    `k₀ + n − 1 ≤ G`. That is all that occurs at `a = 3/5`: X5 counted every
    tap in 3D.

  Kerr-Schild `a = 0` has no such axis, so its runs stay X2b's bit for bit.
- **The per-point information**, built once from the frozen geometry and
  the build state:
  - three rule bits per zone point, set from the state's shift in the
    census pass. They go beside the three-valued `UInt8` class or in a
    second `UInt8` array;
  - a direction code (0–25) per excised point within `q/2` of a zone point
    along an axis, from the geometry's normal (the sphere: `x − c`; tracked:
    `excision_normal`).

  The kernel reads codes, never geometry. The sources' classes come from the
  class array.
- **Refusals and the record:**
  - X2b's refusal of a shift into the excised set becomes a refusal only of
    an excised tap with no source;
  - `excision_into` keeps counting every chunk;
  - add a count of rule axes whose current shift sign disagrees with the
    build's, which must stay 0;
  - report the faces per rule;
  - no `z` axis is frame-dragged at `a = 3/5`.
- **With the blend on**, the lopsided row at such an axis is the open row
  with its excised taps extrapolated. That is what X5 measured, but the
  blend is not needed.
- **For X7:**
  - sphere geometry, the algebraic source, `q = 4`, `cfl = 1/2`,
    `ε_KO = 1/2`, blend off;
  - **always pass `r_0=`** inside `(0.6, r_E)`: the default `r_E/2` lies
    inside the ring below `r_E = 1.2` and is refused by name;
  - the depth scan at `h = 1/24` over `r_E ∈ {0.70, 0.80, 0.90, 1.00,
    1.133}` (26.4 down to 16 cells at the poles);
  - production at **`r_E = 0.80`, `r_0 = 0.70`**: 24, 32 and 48 cells below
    the poles at `h = 1/24, 1/32, 1/48`, `1.0 M` under `r₊`;
  - normal outflow ends at `r_E = 0.641`.
- **CLAUDE.md** has no X5 paragraph yet: X5 left it alone for X4's merge. X6
  adds it beside its own. The `excision_model.jl` commands are in the
  script's header. X1's `margins` is now `margins=margins`; bare `margins`
  runs both parts.

## Step X6 — The frame-dragged faces in the zone kernel

Starts from the integration branch with X4 and X5 merged. `CODE.md`:
"Excision", X5's records, "One right-hand-side evaluation" as X4 left it.

**Changes:**
- the family X5 chose, in the zone kernel's closure provider, selected per
  face from the classes. The build records which rule each closure axis
  uses;
- the zone points' mixed derivative symmetrized, `½(D_i D_j + D_j D_i)`
  (X4's hand-over). The interior or excision parameters may carry a switch
  for it, so the old nesting can still be compared; if so, it is on by
  default;
- X2b's build-time refusal of a shift into the excised set becomes a
  refusal only of what the chosen family does not cover;
- the record's outflow rows report the faces per rule;
- `test/octant_runs.jl octant=rotating a=3/5 interior=excised …` runs.

**Tests** (claims, priced):
- every existing excision claim still holds;
- the frame-dragged rule changes nothing where no axis is frame-dragged: a
  static `a = 0` excised right-hand side with the rule built in is bit for
  bit the same build's without it (X1: `a = 0` has no such faces);
- the symmetrized mixed derivative:
  - is exact on polynomials to its degree;
  - makes the rotating octant's excised `a = 0` right-hand side the mirror
    octant's to roundoff;
  - its change to the fixture's run against X2b's record, and to one short
    `a = 0` octant row against X3's, is measured and recorded;
- the new rule is exact on polynomials to its degree;
- a planted degenerate metric in the excised set leaves every
  non-excised `du` `isequal`;
- one `a = 3/5` excised right-hand side and a short run on the rotating
  octant, finite, with the new rows.

**Accept:**
- the above;
- the suite green at one and four threads;
- a local smoke run at `a = 3/5` that writes `simwatch.toml`;
- `CODE.md` amended **(amended in step X6)**.

**What step X6 hands over** (its report's section 6, 2026-10-06; the
numbers are `CODE.md`'s "Excision", "What step X6 built", and Measured
results, step X6).

The frame-dragged rule is built: `DraggedProvider`, launched by
`gh_dragged_kernel!` after the zone kernel. The zone points' mixed
derivative is symmetric by default (`Excision(T; mixed = :symmetric)`;
`:nested` is the X3-era operator, for comparisons only).
- **The `a = 3/5` smoke** (`h = 1/16`, `r_E = 1`, `r_0 = 0.8`) runs end to
  end on the CPU and on an H200, the CSV within `3.2·10⁻⁹`: 72
  frame-dragged axes, all along `y`, and `excision_flips = 0` at every row.
- **The rotating and mirror octants agree to `8.5·10⁻¹⁰`** at `a = 0`.
- **The symmetric derivative** moves X2b's fixture error outside `r = 23/20`
  by `7·10⁻⁸` relative, and X3's shells by at most `3.5·10⁻⁵`.
- **Cost:** the rule is about `0.5 ms` a right-hand side on the H200,
  latency-bound, predicted `0.6 %` of a right-hand side at `h = 1/24` and
  `0.3 %` at `1/32`. The new kernel has 255 registers and a 1872-byte frame.

- **Every row** is Kerr-Schild `a = 3/5` on the rotating octant with the
  sphere geometry, the algebraic source, `q = 4`, `ε_KO = 1/2`, `cfl = 1/2`,
  no blend, on an H200 from a copy with `CUDA` added, one row a job. The
  scan has the default noise; production has `amplitude=0`. **Check the
  reference rows' own noise and options** in `CODE.md`'s `a = 3/5` record
  and match them, or run both.
- **The depth scan at `h = 1/24`, `10 M` a row, five rows:**

  ```
  julia --project=. --threads=8 test/octant_runs.jl backend=cuda case=ks octant=rotating a=3/5 interior=excised geometry=sphere L=64 N=96 roots=2 radii=32,16,8 r_E=<rE> r_0=<r0> t_end=10 chunk=1 cfl=1/2 out=<dir>/scan/rE<rE> checkpoint=<dir>/scan/ck-rE<rE> walltime=<s>
  ```

  | `r_E` | `r_0` | cells below the poles | below the equator |
  |---|---|---|---|
  | 7/10 | 13/20 | 26 | 28.7 |
  | 4/5 | 7/10 | 24 | 26.3 |
  | 9/10 | 3/4 | 21 | 23.9 |
  | 1 | 4/5 | 19 | 21.5 |
  | 17/15 | 13/15 | 16 | 18.3 |

  `r_0 = (0.6 + r_E)/2` in every row: the script's default with a spin,
  passed anyway so it is on record.
- **Production at `r_E = 4/5`, `r_0 = 7/10`** (unless the scan prefers
  another), `amplitude=0 t_end=24`, and the `1/32` row also to `t_end=64`:

  | `h` | mesh options | blocks | points | cells below the poles |
  |---|---|---|---|---|
  | 1/24 | `L=64 N=96 roots=2 radii=32,16,8` | 29 | 25.7 M | 24 |
  | 1/32 | `L=64 N=128 roots=2 radii=32,16,8` | 29 | 60.8 M | 32 |
  | 1/48 | `L=64 N=96 roots=2 radii=32,16,8,4` | 36 | 31.9 M | 48 |

  - `1/32` is the reference rows' mesh.
  - At `1/48` only `[0, 4]³` is at `1/48`, while `r = 4 … 8` stays at
    `1/24`. Only the shells inside `r < 4` form a clean
    `1/24 – 1/32 – 1/48` series, and the `[3, 5)` shell straddles the
    boundary. If an order is wanted there, use radii that keep the shells
    on one level, or report only the shells inside `r = 4`.
  - The excision's one-level check passed for all fifteen
    `(h, r_E)` combinations: the surface's `9h` neighbourhood lies inside
    the finest cube.
- **Predicted cost** (X3's H200 rates, plus about 3 % for `a = 3/5`):

  | `h` | per `M` | a `10 M` scan row | `24 M` | `64 M` |
  |---|---|---|---|---|
  | 1/24 | ~140 s | ~25 min | ~1 h | — |
  | 1/32 | ~380 s | — | ~2.5 h | ~7 h |
  | 1/48 | ~350 s | — | ~2.3 h | — |
- **Expected frame-dragged axes on the octant:** about 190, 355 and 850 at
  `1/24`, `1/32` and `1/48`.
- **Read `excision_flips` at every row before trusting a checkpoint
  chain.** A restart rebuilds the rule bits from its own state, so a chain
  is the uninterrupted run only while flips are 0.
- **Report the faces per rule** as `excision_faces_dragged` against
  `excision_faces`.
- **Symmetry leftovers:** X6's `excision-x6` directory holds a CUDA copy
  and logs. X7 uses a directory of its own, and deletes its checkpoints once
  the data are copied back.

## Step X7 — The static `a = 3/5` hole on the rotating octant, on the H200s

`CODE.md`: "Excision" with X5's window, the rotating octant's `a = 3/5`
reference rows, and X3's entry (the method).

**Resolution** (Erik: "ensure you have sufficient resolution"):
- the reference `a = 3/5` convergence pair found `h = 1/16` outside the
  asymptotic regime, so production runs at **`h = 1/24, 1/32, 1/48`** (an
  extra cube for `1/48`);
- the excision surface sits **at least about 16 cells** below the horizon
  at the poles at the coarsest production `h`, and the depth scan decides
  how much more.

**Rows**: Kerr-Schild `a = 3/5`, static, on the rotating octant, with the
algebraic source, `q = 4`, `cfl = 1/2`, `ε_KO = 1/2`, and `amplitude = 0` for
the convergence rows. Each row has its own `out=` with `simwatch.toml` and
its own `checkpoint=`/`walltime=`.
1. **The depth scan** at `h = 1/24` across X5's window, about `10 M` a row,
   with and without the blend if X5 asks.
2. **Production** at the chosen `r_E`: `h = 1/24, 1/32, 1/48` to `24 M`,
   and the `1/32` row to `64 M`, against the reference rows. Report:
   - the shells' `ℋ` and error with their orders;
   - `J` and `M_irr` and their drifts. The reference `:damped` has `J`
     rising at `5.2·10⁻⁸/M` at `h = 1/32`, as truncation error at order
     four;
   - the band;
   - the outflow rows, with the faces per rule.
3. **Cost** on the H200.

**Records**: a Measured results entry, "Excision of the static spinning
hole (step X7)", and the recommendation: `a = 9/10` (which X1's window puts
at about `h ≲ 1/40` there), moving holes, or stop. Delete the run's
checkpoints on Symmetry once the CSVs, records and `simwatch.toml` are copied
back (Erik's instruction of 2026-10-06 for X3's).

**Accept**: every row run or its failure diagnosed; the tables in `CODE.md`;
the suite green at one and four threads.

## Step 9 — Infrastructure and the H200 (G6)

`CODE.md`: "I/O and viewers", "Precision, threads, devices", milestone
G6; `notes/ghaccel-bench.jl` for the roofline format.

**What steps 8 and 8′ hand over** (their reports' section 6; `CODE.md`,
"The moving hole (step 8)", "The trailing side (step 8′)", milestone G5's
status paragraphs, and the open question "The moving layer's trailing
side"):

- **G5 is open, and its case exists.** The G5 run this step's device rows
  and CLI name is harmonic Kerr `a = 7/10` boosted at `0.3`, `:fitted` on
  the tracked geometry, `m = 4`, `n_L = 8`, `lmax_shape = lmax_fit = 12`,
  finest `h = 5/256`, `fit_initial_blend = true`, `trail_ramp = 9/10`
  (`hole_runs.jl moving=g5t` is the row); the boosted `a = 0` hole
  (`moving=ctl`) is the cheap stand-in with an analytic control at `20/M`.
  Do not wait for G5 to be *(Done.)*: this step's acceptance is the
  infrastructure and the measurements, and the device row asks for "the
  same mesh, horizon and analysis record as the host", not for G5's
  numbers to be good.
- **Costs to plan the benchmark and the device rows with.** G5's chart is
  3900–5600 blocks over a crossing and `630 s` of a node per `M` at rest;
  the boosted `a = 0` hole 1128–1912 blocks at `5/128`, `13 M` in `5947 s`
  on a node. A uniform `1.7e7`-point run needs about 80 GB, and long
  Symmetry studies must each run from a **remote directory of their own**:
  rsyncing changed sources into a directory with running jobs and
  precompiling there rewrites the package image those jobs have mapped
  (the likeliest cause of step 8's `SIGBUS`, proposed in step 8′) — bake
  that into the batch job this step writes. Give workers
  `OPENBLAS_NUM_THREADS=1`; do not precompile on the login node.
- **The record's rows this step's I/O must carry**: everything step 8f's
  matrix and step 8's crossing tables read — the masked error L2/L∞, the
  shell `C_a` and 8a's horizon shells, the residual against the truth or
  the target, the drift (evolved points only), the projection hits and
  `bounds_r_max`, `fit_valid`/`fit_residual`, the horizon's `A`, `M_irr`,
  `J`, `M_ch`, its extent ratio along the boost, the track's center and
  offset, the refinement centroid and its offset, block counts, `variant`,
  `ρ_max`, the wall clock per chunk. `hole_runs.jl`'s sections and
  `out/readjls.jl` on Symmetry read them from `.jls` today; the time series
  replaces that.
- **The viewers' slices** should show the tracked offset surface and the
  core surface (the shape series, not a sphere) and the found horizon's
  cross-section, since G5's layer is not spherical.

**Checkpointing exists (added 2026-10-01)**: `evolve!`'s
`checkpoint_path_prefix`, `max_walltime_seconds` and `restart_file` with
`latest_checkpoint` (`CODE.md`, "Checkpoint and restart"), so the Symmetry
batch job this step writes is a job chain — resubmitted until `finished` —
and `bin/gh.jl` passes the checkpoint keywords through. The time series is
not the checkpoint's record: the record a restart brings back is in the
checkpoint, and the time series is appended from the restart on.

Changes: `src/io.jl` — the analysis time series (one dataset per
recorded quantity, appended and flushed at every chunk boundary), slice
output; `src/benchmark.jl` and `bin/benchmark.jl` after TreeWave's, and
a batch job for Symmetry after TreeWave's `benchmark.sbatch`;
`bin/gh.jl` (the CLI after GHSO2's `gh3d.jl`: case, `N`, thresholds,
`q`, `interior`, `T`, backend, output); `bin/backend.jl`,
`bin/visualize.jl` (slices with block outlines, the layer's radii, the
horizon cross-section, `τ`; norms against time), `bin/Project.toml`
with both pins; `test/device_tests.jl` behind `TREEGH_TEST_BACKEND`
(`cuda`, `metal`), running on `CPU()` by default; the CI figure job.

Accept: every figure written in CI; the per-phase tables on threads
(locally, and on Symmetry when run there, with and without page
interleaving), `q = 4, 6, 8`, `N = 16` against `32` — recorded in
`CODE.md` and the defaults chosen from them; the device rows — the G5
run in `Float64` reaching the same mesh, horizon and analysis record as
the host, and for the RHS kernel in `Float64` and, if it compiles,
`Float32`, fused and split, at two workgroup shapes: registers per
thread and spill bytes from `ptxas` (`CUDA.@device_code_ptx`;
`ld.local` is the tell), achieved occupancy from `ncu`, picoseconds per
point against the roofline in the format of `notes/ghaccel-bench.jl`,
beside GHAccel's 25.8 % — run on the H200 by whoever has Symmetry, with
the scripts this step provides, and recorded when they exist; `Float32`
on Metal recorded as it comes out, pass or fail. Mark G6 *(Done.)* when
the H200 rows are in.

## Step 10 — Review pass

Read `CODE.md` against the code once more. Mark G0–G6 *(Done.)*; update
`README.md`'s status and `CLAUDE.md`'s "Current state" and "Commands";
move anything still marked **(predicted)** to measured or to "Possible
extensions"; delete this file.
