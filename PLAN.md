# TreeGeneralizedHarmonic implementation plan

For the sessions that implement TreeGeneralizedHarmonic, one step at a
time. The **design** lives in `CODE.md` — read it first, in full; it is
authoritative, and this file is only the work breakdown: what each step
changes, what it must not change, and what it must measure and record.
`CLAUDE.md` has the mechanics and the traps. Delete this file when the
last milestone is marked *(Done.)* in `CODE.md`.

**Steps 0–7 are done. Steps 8a and 8b are next** — the generic interior
(steps 8a–8g, added 2026-09-23), which step 8 needs before it can run
its case.

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

Changes, the smallest that make the experiments possible: `Interior` gets
an optional `target` background (`isbits`, default `nothing` meaning the
case's own), read where the kernel evaluates `u_exact`
(`background_state(bg, t, x)` in the layer branch of `gh_rhs_kernel!`) and
in `gh_paste_kernel!`; the initial data, the Dirichlet hook, the gauge
source and the error reference stay on the *true* background, so
`DIAG_RES` measures the layer's distance from the truth. `chunk_interior`
gains `ρ_max_fixed` (a rate) as the alternative to `factor/dt`, threaded
through `evolve!`. A `C²` dissipation profile `ε_KO(r)` in `gauge.jl`
beside `GaussianDamping` — the exterior's value at and outside `r_1`,
rising to `ε_in` inside the layer — accepted by `GHCase` where a number is
today. Test-side target wrappers in `test/evolution_cases.jl` implementing
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
- Controls: `:frozen` and `:pasted` with the range projection on.

Accept: the scan's table in `CODE.md` "Measured results" and, under "The
interior", the layer rule for a generic target — ramp thickness in cells,
`ρ_max` as a rate, `ε_KO(r)` — marked **(measured in step 8c)** where the
scan decides and **(proposed in step 8c)** where it interpolates; the
prediction `n_L ≳ G (10 ρ_max M)^{1/3}` confirmed or replaced; a stated
recommendation — proceed to steps 8d–8f, or to 8g — with the numbers it
rests on. Nothing long in the suite; the `target` keyword and the profile
get one short claim each.

## Step 8d — The tracked horizon geometry

`CODE.md`: "The interior" (placement, the radius assertions), "Analysis
quantities" (the horizon rows), "Refinement and regridding" (the level
floor); finding 3 above.

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
findings 1 and 2 above. **Two halves, reviewed between them**: 8e-i is
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
| **harmonic `a = 9/10`** | `:fitted` only | the blocked case; needs `m h < 0.1 M` on the equator (`h ≈ 5/256` at `m = 4`); the last row, and it may need a node |
| boosted harmonic `a = 0` or `7/10`, `v = 0.3` | `:fitted` tracked against `:damped` analytic | G5's stand-in: tracking, release, staleness |
| hand-over | `:damped` to `5 M`, then `:fitted` | a newly found horizon's first fit from evolved data |
| coasting | the finder disabled for five chunks | the geometry without a find |

Accept: the table in `CODE.md`; "The interior" rewritten around geometry,
target, the ramp rule and `ρ_max` as a rate, with step 5's design kept as
the analytic control; the `a = 9/10` open question closed, or restated
with the measured `h` and the recommendation to run G5 at `a = 7/10`;
G5's acceptance naming the tracked geometry; `CLAUDE.md`'s "Current state"
and "Commands" updated. Mark the generic interior *(Done.)* under G5's
entry.

## Step 8g — Excision by a host reference (only if 8f says so)

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
layer as the control, and `a` is `9/10` if 8f's last row ran and
`7/10` otherwise — Erik's call, recorded in `CODE.md`.

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
