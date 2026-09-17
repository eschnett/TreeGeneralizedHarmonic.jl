# TreeGeneralizedHarmonic implementation plan

For the sessions that implement TreeGeneralizedHarmonic, one step at a
time. The **design** lives in `CODE.md` — read it first, in full; it is
authoritative, and this file is only the work breakdown: what each step
changes, what it must not change, and what it must measure and record.
`CLAUDE.md` has the mechanics and the traps. Delete this file when the
last milestone is marked *(Done.)* in `CODE.md`.

**Nothing is done yet; step 0 is next.**

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
- **`ρ_max` is bounded by RK4's stability**, about `2.8/dt` on the
  negative real axis; the driver sets `ρ_max · dt = 1` per chunk.
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

## Step 8 — A hole that moves (G5)

`CODE.md`: "The interior" (the moving hole), "Refinement and
regridding" (what follows for the hole), "Boundaries: Dirichlet" (time
dependence), milestone G5.

Changes: the boosted harmonic Kerr case (`boost(Harmonic(M, a), v)`,
`|v| ≈ 0.3`, `a = 0.9`); `refinement_buffer` from `|v| · chunk`; the
layer with `c(t)`; the refinement centroid against the analytic center
in the record; the horizon finder along the trajectory; a uniform-mesh
control run at the finest spacing.

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

## Step 9 — Infrastructure and the H200 (G6)

`CODE.md`: "I/O and viewers", "Precision, threads, devices", milestone
G6; `notes/ghaccel-bench.jl` for the roofline format.

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
