# TreeGeneralizedHarmonic implementation plan

For the sessions that implement TreeGeneralizedHarmonic, one step at a
time. The **design** lives in `CODE.md` — read it first, in full; it is
authoritative, and this file is only the work breakdown: what each step
changes, what it must not change, and what it must measure and record.
`CLAUDE.md` has the mechanics and the traps. Delete this file when the
last milestone is marked *(Done.)* in `CODE.md`. **(Amended 2026-10-08:**
the measured results of the interior — step 5's layer, steps 8a–8′, and the
open questions they answered — moved from `CODE.md` to
`SINGULARITY_HANDLING.md`; where
a brief below says "`CODE.md`, Measured results" or "Open questions" for
those, read `SINGULARITY_HANDLING.md`.**)**

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

**Steps B1–B9 (added 2026-10-25) are proposed; B1 ran on 2026-10-25, B2 and B3
on 2026-10-26** (B1 and B2 on PR #18, B3 on `claude/step-B3-several-holes`, each
awaiting review; B4–B9 not started): binary black holes, toward `CODE.md`'s
milestones G7 and G8, which go beyond the proof of concept. The design is
`CODE.md`, "Binary black holes". **Erik decided the pairing on 2026-10-26: (α)**,
the composite tracked Kerr-Schild source fed by the per-hole tracks with B2's
tracked `:damped` target per hole — and `GHCase(…; holes)`, and the superposed
metric kept here. **And on what feeds that source, the same day (`CODE.md`'s item
7):** a slow control law for the track-fed source, tested on one moving hole fed
by its own track first — built in B3's review, which in its second round, on
Erik's request, turned the model off and added an integral: the PID loop without
the model is proposed for B4 (`CODE.md`, "The controlled source line"). This file
stays until G8 is done.

**Step I1 (added 2026-10-30) is implemented, screened and shelved**
(branch `claude/step-I1-layer-free-interior`, 2026-10-30): a layer-free
interior in the spirit of the moving puncture — smoothed data, a lapse
steered toward collapse inside the horizon, and the range projection as a
guard — screened on the static `a = 0` hole against `:damped`. It builds on
B1's gauge sources and is independent of B2–B9. A CPU diagnosis on Symmetry
(2026-10-30) found that no start survives `M/4` without a strong anchor toward
the smoothed state, so the screen was rebuilt around the anchor at
`ρ_a = 10/M`, which held the ball to `4 M`. That screen ran on H200s the same
day: the base passes at `8 M` with `ℋ` just outside the horizon at `0.34×`
`:damped`'s and the error at `0.71×`, and the steering, `ΔH` and the guard
are inert; the promotion (`1/24`, `1/32`, `24 M`, and `128 M`) is not run.
**Shelved by Erik on 2026-10-30**: this exact design is not pursued further,
though similar ones will be; the code stays on its branch as a record. The
screen's command lines and results are in `SINGULARITY_HANDLING.md`, "A
layer-free interior (step I1)".

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
extensions"; delete this file. **(Amended 2026-10-25:** keep it, and the
binary steps below, until G8 is *(Done.)*.**)**

## Steps B1–B9 — Binary black holes (added 2026-10-25)

These take the package past the proof of concept, to two holes that merge
and the waves they emit: `CODE.md`'s milestones G7 (a head-on collision)
and G8 (an equal-mass inspiral). **The design is `CODE.md`, "Binary black
holes"; read it in full first.** Nothing here is started. The steps are a
proposal for Erik's review, and the section's "What this asks Erik to
decide" lists what he settles first.

**Order:**

- **B1 and B2** are single-hole screens and may run as two agents at once.
- **Erik decides** the gauge–interior pairing, (α) or (β), from their
  reports before **B3** starts.
- **B5 and B6** need no hole. They may run beside B1–B4, B5 first, since B6
  uses B5's wave.
- **B3, B4, B7** then follow in order.
- **B8** follows B3, and uses B1's `gauge_consistent_momentum`.
- **B9** follows B7 and B8.

Each runs from a green suite on the merged branch and ends with the report
under "Running a step as an agent". Branches are named
`claude/step-B<n>-<slug>`.

**What the single holes hand over.** Read `CODE.md`, "What the single holes
hand over" and "Single black holes: recommended settings". Three of those
findings set the steps below:

- **The moving `:damped` layer is the accurate one.** With its target's
  rate and a core that follows, it holds a boosted Kerr-Schild hole at the
  resting hole's level.
- **`:fitted` holds every moving hole but is the weaker interior.**
  `13–176×` the constraint violation just outside the horizon, and no
  convergence at a margin fixed in cells.
- **`HorizonTrack` predicts along a line.** That is enough at `d = 10 M`
  and four cells off at `d = 3 M` at `h = 1/64`.

### Sharp edges for steps B1–B9

- **Units.** The total mass is `M = 1` and each hole of an equal-mass pair
  is `M_A = 1/2`. Every single-hole setting scales by `M_A`: the moving
  `:damped` pair `1/24`, `1/32` is `1/48`, `1/64` here, and
  `ρ_max = 4/M_A = 8/M`. `hole_mass(case.background)` is the wrong mass
  for a binary: read each hole's own.
- **The background becomes data.** Everything that reads
  `case.background` or `case.center` as the truth needs a per-hole or a
  data replacement: the hook, the error norms, the target, the radius
  checks, the seed, the damping profile, the refinement centroid.
  `grep -n 'background\|case.center' src/` before designing anything.
- **Everything that relaxes toward something that moves needs that
  thing's rate**, in every region it relaxes in, the core included
  (`CLAUDE.md`). A tracked target and a tracked source are such things.
- **A gauge source's `∂_t H` must be its own time derivative**, including
  the parts that come from the tracks. If it is not, `C_a` grows and no
  `H` explains it.
- **The tuple of interiors is a kernel argument.**
  - `K` is a type parameter, unrolled by `@ntuple`/`@nexprs`, with no
    closures (on a device a closure is a call).
  - It is `isbits`.
  - It is built once per chunk or piece, with its `Val`s.
  - On the CPU it runs on SIMD lanes, so the composition uses `_select`,
    not `if`, on computed values.
- **Over MPI the finds, the fits and the extraction are rank 0's.** A new
  collective inside them goes through `fill_state!`/`interpolate_state`,
  or a new request kind in `serve_requests`.
- **Anything the loop carries goes into the run state**: the tracks, the
  common track, the handover, the predictions' history and the
  extraction's integrals. The checkpoint chains are what notice when
  something is missing.
- **`diag` slots are appended, never inserted.**
- **A moving pair's symmetric domain matches the box only to the target
  rate's amplified roundoff.** Bound it as `boosted_tests.jl` does.
- **Price every test.** Two holes are two layers, two finds and a larger
  mesh. Runs longer than a minute go into a script (`test/octant_runs.jl`'s
  options, or a `test/binary_runs.jl` beside it), and long runs to
  Symmetry as job chains with their own remote directories.
- **No downloads without asking.** SXS waveforms and any data from outside
  the machine are Erik's to supply.

## Step B1 — Two gauge sources and the initial momentum

`CODE.md`: "The gauge" (under "Binary black holes"); "Gauge and constraint
damping" (the algebraic source); "Open questions", "Gauge sources that know
less about the hole". **One hole only**: the binary case is B3's.

**Changes**, all in `src/gauge.jl`, evaluated in the kernels as
`KerrSchildSource` is, generic in the number type for the SIMD lanes and
closure-free for a device:

- **The composite tracked Kerr-Schild source**, `K` holes: each hole's
  `M_A` and `S_A`, the piece's line `c_A(t)`, `u_A(t)` and `du_A/dt`, and
  the partition of unity with `p = 4`. `∂_a H_b` includes `∂w_A` and the
  `du_A/dt` term.
- **The damped harmonic source** in SpECTRE's form (`A_L1`, `A_L2`, `A_S`,
  `e`, `σ_r`).
- **The roll-on** `f(t) H_from + (1 − f(t)) H_to`, with
  `f = e^{−((t − t_0)/σ_g)⁴}` and its `∂_t f`.
- **`gauge_consistent_momentum`**: `∂_t α` and `∂_t β^i` from
  `C_a = 0`, given `(γ, K, α, β)`, their spatial derivatives and the
  source.
- **`test/octant_runs.jl` options** for the source and the roll-on.

**Accept:**

- The composite with one hole is `KerrSchildSource` bit for bit: the
  right-hand side of the moving quadrant hole.
- `H` and `∂H` of each new source agree with forward-mode differentiation
  of its closed form, at random states near Kerr-Schild, to roundoff.
- `gauge_consistent_momentum` returns a boosted Kerr-Schild hole's
  analytic `Π` under its own source to roundoff, and gives `C_a = 0` at
  `t = 0` for a perturbed lapse.
- **The damped harmonic gauge rolled on**, on the boosted `a = 0` hole on
  the quadrant (`v = 1/5`, `:fitted`, `1/24` and `1/32` in `M_A`) to
  `40 M`, against the same hole in its own Kerr-Schild gauge. Record:
  - `ℋ` and `C_a` just outside the horizon;
  - the horizon's `M_irr`;
  - the hole's coordinate trajectory;
  - the static hole too, which Lindblom and Szilágyi saw settle.
- The kernel's cost per point with each source.
- The runs in `SINGULARITY_HANDLING.md` under a section of their own, and
  summarised under `CODE.md`'s "The gauge".

## Step B2 — The tracked analytic target

`CODE.md`: "One interior per hole", "The tracks"; "The interior" (step 8c's
rule, the moving `:damped` layer, `core_follows`). **One hole only.**

**Changes:**

- **`src/tracking.jl`:**
  - the quadratic prediction, a least-squares fit in time over the last
    four finds (today's line until there are four);
  - its blend into the previous prediction over the next interval;
  - the center-jump check, the found origin within `G h/2` of the
    prediction;
  - the find interval from the predicted acceleration, `a Δt² ≤ G h/4`;
  - the prediction's error in the record.
- **`src/interior.jl` and `src/driver.jl`:** a `:damped` interior whose
  target and center come from a track. The target is the boosted,
  translated Kerr-Schild hole with the hole's `M_A` and `S_A` on the
  piece's line, rebuilt every piece, with the core following its target
  and the target's rate along the line.
- **`test/octant_runs.jl`:** the tracked target, and knobs that perturb its
  mass, velocity and center.

**Accept:**

- **The prediction on synthetic finds** on a circle at `d = 10` and
  `d = 3`, a host test: its error under `G h/4` at the proposed cadence.
- **On the moving `a = 0` hole** on the quadrant (`v = 1/5`, `1/24` and
  `1/32` in `M_A`) to `40 M`, the tracked target:
  - `ℋ` just outside the horizon within `1.5×` the analytic `:damped`
    layer's;
  - order four between the two spacings.
- **The sensitivity**, recorded for each perturbation: `ℋ` and the error
  outside the horizon, and the layer residual.
  - the target's mass off by `±1 %`;
  - its velocity off by `±0.02`;
  - its center off by `±1` cell.
- **Recovery:** a track started with a velocity off by `0.05` recovers
  within a few finds, with no refusal.
- **The report** says, beside B1's, which pairing its numbers favour.

## Step B3 — Several holes in one case

`CODE.md`: "A case without a background", "One interior per hole",
"Refinement for two holes", "The record", "Initial data" (item 1). **Starts
after Erik's decision on the pairing** (made 2026-10-26: (α); B3 ran that day,
`CODE.md`, "What step B3 built and measured"; its review added the controlled
source line on Erik's decision of the same day, "The controlled source line").

**Changes:**

- **The case:** `HoleSpec`, and `GHCase(…; holes)` with the background as
  data. A `binary_case` constructor on the full box.
- **The data:** an `isbits` superposition with the optional attenuation,
  here, unless Erik takes it to `SpacetimeMetrics` first.
- **The kernel:** the interior a tuple with the composition rule, built
  from B2's tracked target or `:fitted` per hole. The composite mask, used
  by every norm, the indicator, the projection's gate and the finder's
  guard. `γ0` as three Gaussians, and `ε_KO` per hole.
- **The driver:** per-hole tracks and finds in `evolve!`, and the gauge
  source of the chosen pairing fed by the tracks. Per-hole level floors,
  their union taken. Per-hole record rows, and the separation and phase.
- **The run state:** the `K` tracks in the checkpoint.
- **MPI:** the `K` finds in rank 0's analysis.

**Accept:**

- **One hole is the old case:** `holes = (spec,)` bit for bit, `u` after
  two chunks and the record. Run on the suite's fixture and on the moving
  quadrant hole, at one thread and at four.
- **The composition is local:** the composite right-hand side equals the
  single-hole kernel of the nearer hole at every point where only one
  layer acts. Compared point by point on a two-hole mesh.
- **A head-on pair** from rest at `d = 8 M` on the box, `h = 1/24` at the
  holes, to `2 M`:
  - every find succeeds and no assertion fires;
  - the separation falls as free fall predicts, to the finder's accuracy;
  - the constraints stay bounded;
  - the cost is recorded.
- **Restarts:** that run's checkpoint chain is the uninterrupted run, bit
  for bit.

## Step B4 — The symmetric domains and the half turn

`CODE.md`: "The symmetric domains"; "Reflecting faces", "The rotating
octant"; TreeAMR's `CODE.md` on the half turn (M14).

**Changes:**

- **The half turn:** `GHCase(; halfturn)` beside `rotating`, and
  `state_halfturn` with the `G = 0` sets' identity map at every `FieldSet`
  site. `gh_forest` takes the half turn and the brick `(r, 2r, r)`.
- **`binary_case(; symmetry)`**, one of `:octant`, `:rotating_octant`,
  `:halfturn`, `:bitant` or `:none`. Each refuses a pair its symmetry does
  not hold, and says why.
- **The image geometry** under the half turn, and the single track.
- **`add_noise!`** leaves the seam plane alone.
- **The checkpoint's recipe** carries the half turn.

**Accept:**

- **On uniform meshes:**
  - the head-on octant is the box;
  - the half-turn quarter is the box for a pair in circular motion over
    16 steps, to the moving holes' roundoff bound.
- **Across the seam:** a find of hole 1 at `x < 0` agrees with the box's
  find to the finder's tolerance.
- **Every refusal** fires with its reason.
- **A hole crosses the seam:** tracked through the crossing, with no
  refusal and no failed find.

## Step B5 — The far boundary, the wave zone and a Teukolsky wave

`CODE.md`: "The outer boundary and the wave zone", "Refinement for two
holes". **No hole.**

**Changes:**

- **The Teukolsky wave:** the linearised even-parity `ℓ = 2`, `m = 0`
  quadrupole wave (Teukolsky 1982) as an `isbits` `AbstractMetric`,
  `η + ε h`.
- **The boundary choice:** the frozen hook (the data at `t = 0`) and the
  relative hook beside the analytic one.
- **The wave-zone floor** in `Refinement`.
- **`test/octant_runs.jl` options** for all three.

**Accept:**

- **The wave converges** at order `q` with the analytic hook, on the
  octant its symmetry allows, at an amplitude where `ε²` is below
  truncation.
- **Each hook's reflection** is recorded, for two box sizes: the error
  against the analytic wave inside the box after the wave meets the
  boundary.
- **The floor holds the wave:** the indicator does not coarsen the wave
  zone while the wave crosses it.
- **The outer levels' cost** is recorded.

## Step B6 — `Ψ₄` and the extraction

`CODE.md`: "The waveform". **No hole.** Uses B5's wave.

**Changes:**

- **`src/weyl.jl`:** the pointwise `γ`, `K`, `∂K`, the spatial Ricci
  tensor, `E` and `B`, the radial tetrad and `Ψ₄`, from interpolated
  values.
- **`src/extraction.jl`:**
  - the spheres on `AbstractSphericalHarmonics` grids;
  - the interpolation with second derivatives through `interpolate_state`
    (a new request kind if it does not carry them);
  - the spin `−2` transform;
  - the ADM surface integrals;
  - the radiated energy, momentum and angular momentum;
  - the CSV;
  - the cadence at piece boundaries.
- **`evolve!(; extract)`.**
- **`test/weyl_tests.jl` and `test/extraction_tests.jl`.**

**Accept:**

- **`Ψ₄` pointwise** against `SpacetimeMetrics`' `RiemannTensor`,
  contracted with the same tetrad, on boosted Kerr and on the Teukolsky
  metric, to roundoff of the scale.
- **The convention:** the sign and the normalisation, from the Teukolsky
  wave's analytic strain, written into `CODE.md`.
- **The Teukolsky wave's `(2, 0)` mode** converges at order `q` in
  amplitude and phase on a small box. The other modes sit at truncation.
- **The ADM mass** of a static and a boosted Kerr-Schild hole at radius
  `R`, with its `1/R` approach recorded.
- **The extraction's cost** against a right-hand side.
- **MPI:** the same modes at one and three ranks (the MPI workload).

## Step B7 — Merger: the common horizon and the handover (G7)

`CODE.md`: "The common horizon and the handover", milestone G7.

**Changes:**

- **The common finder:** its seed, its `lmax`, and its cadence below
  `d_c`. Failures are recorded, not thrown.
- **Formation** by the proposed criterion.
- **The handover:** the common `:fitted` layer added, the individual
  layers dropped and their tracks retired.
- **The remnant's** source and target.
- **The record's common rows**, and the handover state in the run state.
- **The head-on runs** as a script, with their batch jobs.

**Accept:** G7 as `CODE.md` states it. Also record:

- the gap between the individual and the common core surfaces at
  formation;
- how long the handover waited;
- whether the remnant's `:damped` switch helped.

## Step B8 — Constraint-satisfying data from XCTS

`CODE.md`: "Initial data" (item 2). Uses B1's `gauge_consistent_momentum`
and B3's case.

**Changes:**

- **A host tool**, a script run from an environment with `XCTS` added
  beside the package (never in the package's own `Project.toml`). It
  solves or loads XCTS's solution, evaluates it at TreeAMR's points on the
  mesh chosen from the superposed data, and writes the forest and the
  state through TreeIOHDF5.
- **`evolve!` starts from such a file.**
- **The excision:** XCTS's default inner condition (`horizon_bc = false`,
  Dirichlet to the free data) on spheres inside each core. If XCTS does
  not converge with spheres that deep, describe exactly what fails and
  report it, and use the isolated-horizon data extended inward by a fit
  meanwhile.
- **`Π`** through `gauge_consistent_momentum`.

**Accept:**

- **A single boosted Kerr-Schild hole through XCTS:** the state on the
  grid is the analytic one to XCTS's accuracy, and the run is
  indistinguishable from the analytic run.
- **The head-on pair from XCTS data** against the same pair from
  superposed data: the junk and the constraints, at `t = 0` and after.
- **The quasi-circular pair at `d = 10`:** the data's constraints, the
  ADM quantities, and the horizons' masses and spins at `t = 0`.

## Step B9 — The inspiral (G8)

`CODE.md`: milestone G8, "What it costs".

**Changes:**

- **The run scripts and batch jobs:** job chains with SimWatch, at
  `1/48`, `1/64` and `1/80` on the half-turn quarter.
- **The analysis scripts:** extrapolation in radius, the strain by
  fixed-frequency integration, and the comparison with an SXS waveform
  that Erik supplies.

**Accept:** G8 as `CODE.md` states it. Record the cost per `M` and per run
beside the prediction under "What it costs".

## Step I1 — A layer-free interior on the static hole (added 2026-10-30)

`CODE.md`: "The interior" (the range projection, its gate and its ranges),
"Gauge and constraint damping" (the algebraic and the sampled source),
"Single black holes: recommended settings" (the `a = 0` `:damped` row), and
"The gauge" under "Binary black holes" (B1's sources).
`SINGULARITY_HANDLING.md`: "Step 8a" (the leak per cell), "The range
projection (step 8b)" and "The layer against an inexact target (step 8c)"
(how an abrupt change of the state ends a run), and "Single holes on the
octant" (the baseline). TreeBSSN's `CODE.md`, "The smoothed hole", "The
puncture", "The lapse's floor" and "The comparison with TreeGH", for what
the moving-puncture side measured.

**Starts from PR #18's branch** (`eschnett-claude/binary-black-hole-setup-f5b05f`,
tip `b97e72b`), or from `main` once it is merged: it uses B1's
`AlgebraicGaugeSource`, `RolledGaugeSource`, `gauge_consistent_momentum` and
`_adm_rates`. Branch `claude/step-I1-layer-free-interior`. One hole, at rest,
`a = 0`; spin, motion and binaries are out of scope.

**The question.** Can this package hold a black hole the way moving-puncture
codes do, and what do the constraints outside the horizon look like then?
That means no damping layer and no analytic target. The Einstein equations
run everywhere, the interior is made harmless by its data and its gauge, and
a pointwise limit on the state is the backstop. The conjecture is Erik's
(2026-10-30). The reasons to expect it and the reasons not to:

- **For.** Every GH characteristic moves at zero or light speed relative to
  the normal observers. Inside the horizon every one of them points inward,
  so in the continuum nothing made deep inside reaches the exterior. That is
  stronger than BSSN's case, whose 1+log gauge speed reaches `√2`. The
  discrete leak (step 8a) is `0.40–0.51` e-folds per cell at `ε_KO = 1/2`
  over depths of 2–8 cells (`0.25–0.37` in the 1D model at long times).
  This interior puts 8–16 cells between what it does and the horizon at
  `1/16`, and twice that at `1/32`.
- **Against.** The moving puncture works because its gauge takes the
  curvature singularity off the slice (a trumpet), and collapses the lapse so
  that the floored points hardly evolve. TreeBSSN's floors act at a few
  thousand points per chunk, and a floor above the lapse the gauge wants was
  "much worse" there (`α_min = 10⁻²` ended its runs in 5–16 M). This
  package's Kerr-Schild source keeps the singularity on the slice, with
  `β^r → 1` and `Π ~ 1/r²`. Steps 8b and 8c saw abrupt changes of the state
  end runs in 2–18 M, by degenerating the evolved points next to them: a
  failure inside the horizon, with nothing leaking.

So the step takes the singularity off the initial slice, gives the interior
a gauge that collapses the lapse where the volume collapses, and keeps the
projection only as a guard.

**The design**, to go into `CODE.md` under "The interior" as "A layer-free
interior (proposed in step I1)":

1. **Smoothed data.** TreeBSSN's `SmoothedKerrSchild`, which is the Exact
   thorn's `parabolic` smoothing: inside `ε` the Kerr-Schild radius becomes a
   `C⁴` polynomial in `r²`. Port it as an `isbits` background, `a = 0` only
   (at `a ≠ 0` it has a kink on the ring; TreeBSSN's `CODE.md` says why). It
   is Kerr-Schild exactly for `r ≥ ε` and not vacuum inside: a turducken whose
   constraint violation is causally hidden. The data's extremes over
   `r ≤ 3/2` (computed 2026-10-30 on the host by finite differences; confirm
   them on the grid), with `|β| = √(β_i γ^{ij} β_j)` and the momentum scale
   as `bounds.jl` reads them:

   | data | min `α` | `λ(γ)` | max `\|β\|` | max `\|(α/√γ)Π\|` |
   |---|---|---|---|---|
   | `ε = 3/4` | `0.305` | `[1, 6.18]` | `1.98` | `15.8` |
   | `ε = 1` | `0.347` | `[1, 4.88]` | `1.68` | `8.44` |
   | `ε = 5/4` | `0.382` | `[1, 4.11]` | `1.47` | `5.16` |
   | Kerr-Schild, `r ≥ 1/4` | `0.333` | `[1, 9]` | `2.67` | `28.4` |

   (Kerr-Schild's momentum scale is `10.4` at `r = 2/5`, step 8b's number.)
2. **A frozen gauge correction, for a gentle start.** The algebraic
   Kerr-Schild source, evaluated on the smoothed state, is not `−Γ_a` of the
   smoothed metric. The mismatch reaches `|C_a| = 52` at `r = 0.21` for
   `ε = 1` (host-side, as above). `gauge_consistent_momentum` would turn that
   into `∂_t α` up to `3.4/M` and `|∂_t β|` up to `2.3/M` at `t = 0`. So the
   source is `H_a = H_a^{KS}(h) + ΔH_a(x)`, where
   `ΔH = −Γ[g_s] − H^{KS}(h_s)` is sampled once, with its gradient, by the
   nested-dual pass the sampled source already uses.
   - `ΔH` is identically zero for `r ≥ ε`, where the smoothed metric is
     Kerr-Schild and the algebraic source is its `−Γ`. So the exterior gauge
     is the baseline's.
   - The smoothed static data start with `C_a = 0` and `∂_t α = ∂_t β = 0`.
   - This is TreeBSSN's explicit gauge form, the frozen sources, in GH.
3. **Interior lapse steering.** Add
   `ψ(r) [μ_α log(α_*/α) n_a − μ_β g_ai (β^i − β_*^i)/α]` to `H_a`, with `ψ` a
   `C²` smoothstep: `1` for `r ≤ r_s0`, `0` for `r ≥ r_s1`.
   - **Why it steers.** B1's damped harmonic source documents that
     `μ log(1/α) n_a` drives `α → 1` at the rate `αμ` (`C_t = 0` on flat space,
     with `γ` fixed and `β = 0`), and that `−μ_S g_ai β^i/α` drives `β → 0` at
     `αμ_S`. So `log(α_*/α)` drives the lapse to `α_*`. **Test both signs
     before anything runs.**
   - **Why there is no 1+log.** A gauge source may read `g` but not `∂g`
     without changing the principal part, and 1+log reads `K`. With zero
     shift, harmonic slicing integrates to `α ∝ √γ`; the algebraic stand-in
     is a target tied to the volume.
   - **V, volume-coupled:** `α_* = α_s(x) (√γ/√γ_s(x))^p`, with `α_s` and
     `√γ_s` the smoothed background's, in closed form. It vanishes on the
     initial data (to roundoff) and collapses the lapse where the volume
     collapses: `p = 1` like harmonic slicing, `p = 2` more strongly.
   - **F, fixed:** `α_* = κ α_s(x)`, a pre-collapsed lapse. The data either
     start at it (lapse scaled by `1 − (1 − κ) ψ`, with `γ` and `K` kept and
     `Π` from `gauge_consistent_momentum`) or have it rolled on
     (`RolledGaugeSource` from the unsteered source, over `σ_g`).
4. **The range projection as a guard** (`bounds.jl`, unchanged in
   substance). Its ranges catch a state that is about to stop being a
   metric or to break the step, and never shape a healthy one:
   - **`α ∈ [10⁻³, 2]`.** TreeBSSN's working floor: `10⁻⁸` died there of a
     term that divides by `α`, and `10⁻²` fought the gauge. Here only the
     shift steering divides by `α` (off by default); the floor protects the
     steering's logarithm and `metric_quantities`, and V is meant to approach
     it. The lapse inside the horizon never exceeds `0.71`; `α_max = 2` only
     satisfies the constructor's `1 < α_max`.
   - **`λ(γ) ∈ [10⁻³, 10³]`.** Under V with `p = 1` the lapse reaches `10⁻³`
     where the volume has fallen about 300-fold, an isotropic spectrum of
     `0.02`. The spectrum's floor sits below that, so the lapse floor is met
     first and the spectrum's catches only a non-metric. The ceiling is 200×
     the data's.
   - **`|β| ≤ 3`.** That is `1.8×` the smoothed data's `1.68` (`1.5×` at
     `ε = 3/4`). The shift is a coordinate speed, and `gh_dt` sizes the step
     for the far field's `√3`; a shift past 3 is a run about to break its CFL
     condition.
   - **`max |(α/√γ)Π| ≤ 100/M`**, `default_bounds`' value: 12× the data's.
     It shrinks with the lapse.
   - **The gate** at `r_g = r_s1`, under a rule that replaces the layer's:
     `r_g ≤ r_h,min − 8h`, since there is no `r_1`. The projection's kink is
     read by evolved stencils here on purpose. Eight cells keep it, at step
     8a's rate, about `e^{−3.6}` from the horizon, before the clamp's own
     region is counted.
   - **It must not fire on the initial data** (asserted at `t = 0`). A run on
     which it never fires is then the run without it, bit for bit (the
     existing control).
5. **Optional drivers:**
   - the Kreiss–Oliger amplitude inside (`HorizonDissipation`'s `ε_in`,
     rising from `r_h` to `r_s0`);
   - `γ0` near the hole. Its direction is unknown: damping pushes the
     smoothed region toward vacuum, and a vacuum interior re-forms the
     singularity;
   - an **anchor** `−ρ_a χ_ε(r) (u − u_s)` toward the smoothed state inside
     `ε`, with `F` evaluated everywhere (no `w`). It is the bridge to
     `:damped`, with a regular target.

**Changes:**

- **The smoothed background** in `src/initialdata.jl`. It is also the error
  reference; only its exterior, `r ≥ ε`, is Kerr-Schild.
- **The layer-free interior**, an `isbits` object (`ε`, `r_s0`, `r_s1`,
  `r_g`, the anchor) that `GHCase`/`hole_case` accept. Its kernel variant is
  `:none` (or the anchor's), with no frozen core and no `r_1`. It brings:
  - masks for the norms and the speed, which cover the interior now;
  - the gate rule above, in place of `check_bounds_gate`'s;
  - radius checks `r_s1 ≤ r_h,min − 8h` and `ε ≥ 8h` (the ring rule's eight
    cells across the data's scale), in place of `check_interior_radii`'s;
  - validity rows by region (the smoothed ball, the steering's ramp, the band
    inside the horizon) in place of the layer's.
- **The steered source**, `SteeredSource <: AlgebraicGaugeSource`: the
  algebraic Kerr-Schild source, plus `ΔH`, plus `ψ` times the steering. `H`
  and `∂H` go through `_adm_rates`; the code is generic in the number type
  for the SIMD lanes and closure-free for a device. `ΔH` is zero for
  `r ≥ ε` through a branch, not a product.
- **The initial data**: the smoothed metric's own `(h, Π)`; with `κ ≠ 1` the
  lapse is scaled and `Π` comes from `gauge_consistent_momentum`.
- **The record**, every chunk:
  - shells `[0, ε)`, `[ε, r_s1)`, `[r_s1, 2)`, `[2, 9/4)`, `[9/4, 3)`,
    `[3, 5)`, `[5, 8)`, `r ≥ 8`, each with `ℋ`, `ℳ`, `C_a`, the error, the
    `h` part of the right-hand side (stationarity), `min α`, `min √γ/√γ_s`,
    `max λ(γ)` and the 6th-difference roughness (TreeBSSN's
    `variables.csv`);
  - the projection's hits by kind (lapse, spectrum, shift, momentum,
    non-finite), the radius of the outermost hit, and a histogram in cells;
  - the lapse along `x` and along the diagonal (step 8a found the leak
    strongest along the axes), and the radii where `α = 1/2` and `α = 1/10`;
  - `C_a` across the steering's ramp;
  - `M_irr` and `r_min` from the finder.

  One fold per chunk, as `observe` does now.
- **Run tools**: a `test/octant_runs.jl` key for every knob, and a reader
  (`test/layerfree_study.jl`, or `octant_study.jl` extended) that prints the
  tables below.
- **Tests, priced as the ground rules ask:**
  - the steering's two signs;
  - `H` and `∂H` of the steered source against forward-mode
    differentiation at random states near the smoothed data;
  - `ΔH` exactly zero for `r ≥ ε`;
  - `C_a = 0` at `t = 0` on a small mesh, to roundoff;
  - V's steering zero on the data, to roundoff;
  - the gate and radius refusals;
  - one short run of the fixture on which the projection never fires,
    `isequal` to the run without it.

**The screen.** The baseline's case and mesh:
- Kerr-Schild `a = 0` on the reflecting octant `[0, 64]³`, root brick `2³`,
  cubes `R = 32, 16, 8`, so `h = 4/N` on `[0, 8]³`;
- `q = 4`, `cfl = 1/2`, `ε_KO = 1/2`, the Gaussian `γ0`, `chunk = 1`, the
  finder every chunk;
- one H200 a row on Symmetry.

The **base point** is:
- data: `ε = 1`, with `ΔH` on;
- steering: V with `p = 1` and `μ_α = 4/M`, `μ_β = 0`, `r_s0 = 1`,
  `r_s1 = 3/2` (8 cells inside the horizon at `1/16`, as `:damped`'s `r_1`);
- the guard: the ranges above, gated at `3/2`;
- no drivers.

Each row changes one thing:

| row | what it is |
|---|---|
| R0 | `:damped`, `r_0 = 3/4`, `r_1 = 3/2`, the baseline, on this commit, to `24 M` |
| R1 | analytic Kerr-Schild through the core rule at `r_0 = 1/4`, the algebraic source, the guard only: the bare conjecture (its data do not fire the guard) |
| R2 | smoothed data and `ΔH`, no steering: how fast the singularity re-forms |
| R3 | **the base point** |
| R4, R5 | `p = 2`; `μ_α = 1/M` |
| R6 | F, `κ = 1/3`, the data pre-collapsed |
| R7 | F, `κ = 1/3`, the data not collapsed, rolled on over `σ_g = 4 M` |
| R8 | the base with the shift steering, `μ_β = 1/M`, `β_* = 0` |
| R9 | the base without `ΔH` (`gauge_consistent_momentum`'s `∂_t α` up to `3.4/M`): is the gentle start needed? |
| R10, R11 | `ε = 3/4`; `ε = 5/4` |
| R12, R13 | `ε_in = 2`; `γ0` near the hole `1/(10M)` |
| R14 | the anchor, `ρ_a = 1/(10M)` |
| R15 | `α_min = 10⁻⁶`: does the floor act? |
| R16 | tight ranges at `2×` the data's extremes (`α ≥ 0.17`, `λ(γ) ∈ [1/2, 10]`, the momentum scale `≤ 17/M`, the shift as in the base): does limiting the values help? |
| R17 | the gate everywhere |

**The screen rebuilt around the anchor (amended 2026-10-30, after the
diagnosis).** The table above is the first screen. Its start, run on
Symmetry's CPUs before any GPU row (`SINGULARITY_HANDLING.md`, "Why the
smoothed ball dies within `M/4`, and what holds it"), ended every row without
a strong anchor within `M/4`: the ball's own constraint violation (`ℋ = 1182`
at `ε = 1`) grows at `15–25/M` whatever the gauge, `ΔH` or the resolution
does. Only the anchor at `ρ_a ≥ 10/M` held it — to `4 M`, with no hit and a
stationary ball, and outside the horizon at `:damped`'s level on that CPU mesh.
So R1–R9 and R14 are answered, and the screen is now:

- the **base point** A10: the first base point with the anchor at
  `ρ_a = 10/M`;
- R0 as before;
- A6 and A30 (the anchor's rate; `4/M` ends within `M/2`);
- A10n (no steering), A10c0 (no `ΔH`);
- A10e34, A10e54 (`ε = 3/4`, `5/4`) and A30e54;
- A10t (R16's tight ranges on the anchored base).

An anchored interior is a damping layer at `r < ε` toward a regular state, so
the question the screen answers is now whether that beats `:damped`'s layer
toward the singular analytic state, outside the horizon and in cost. The
command lines are in `SINGULARITY_HANDLING.md`, "A layer-free interior (step
I1)".

**Cost.** The screen is `8 M` at `1/16`, about five minutes a row, scaled from
the `9.4 M`-point octant run's `26 s/M`; with R0 at `24 M`, about two
GPU-hours (about one for the rebuilt screen's ten rows). A row **passes** if it:
- reaches `8 M`;
- has hits per chunk that level off, with the outermost hit at least 8 cells
  inside the horizon and not moving outward;
- has `ℋ` and the error in `[2, 9/4)` within `3×` R0's, flat or falling from
  `4 M`.

The failure signatures to name are step 8b's and 8c's — the shell error next
to the smoothed ball or the ramp growing from the first chunk — an outermost
hit creeping outward, and `M_irr` drifting.

**Promotion.** At most three passing rows go to `1/24` and `1/32` at `24 M`
(about 45 minutes and 2.2 hours a row). The best of them goes to `128 M` at
`1/16` on `ks-octant`'s `[0, 128]³` mesh, against job 568189 (`ℋ =
7.24·10⁻⁶`, the error `5.56·10⁻⁶`, both masked to `r ≥ 3/2`). Long rows run
with `checkpoint=` and `walltime=`, each study in its own remote directory.

**Accept:**

- The design in `CODE.md` marked (proposed in step I1), and the data's
  ranges and the start's `C_a` confirmed on the grid, **(measured in step
  I1)**.
- R0 reproduces the baseline at `24 M` and `1/16` (`ℋ [2, 9/4) = 2.42·10⁻⁵`,
  the error there `1.08·10⁻⁵`) to two digits.
- Every row with its outcome, its records and its failure signature where it
  failed, in `SINGULARITY_HANDLING.md` under "A layer-free interior (step
  I1)".
- For each promoted row:
  - the order outside the horizon;
  - `ℋ` and the error per shell against `:damped` at each `h` (`ℋ [2, 9/4)`
    `2.42·10⁻⁵`, `1.46·10⁻⁶`, `4.30·10⁻⁷` at `1/16`, `1/24`, `1/32`);
  - `dM_irr/dt` against `:damped`'s `2.09·10⁻⁸`, `4.05·10⁻⁹`, `1.27·10⁻⁹`;
  - the `128 M` row's stationarity.
- A recommendation:
  - **if a row holds:** whether to take it to spin (which needs a smoothing
    without the ring's kink), to a moving hole (which needs `ΔH` and the
    reference `α_s`, `√γ_s` to move, so an algebraic form) and to B3's pair;
  - **if none holds:** what failed, where and when.
- The suite green at one and four threads.

**Sharp edges for step I1:**

- **The exterior's gauge must stay the baseline's.** Anything that changes
  `H` for `r ≥ r_s1` changes the comparison; so does a source that is the
  sampled `−Γ` instead of the algebraic one outside `ε` (the two run alike on
  the static hole, but they are not the same gauge off the solution).
- **The error against Kerr-Schild means something only for `r ≥ ε`**, and
  for `r ≥ r_s1` once the steering has moved the slice inside. Inside, read
  the error against nothing; read the constraints and the record's ranges.
- **With no `r_1`, the speed and every norm cover the interior.** The step
  should not change, since the far field's `√3` is the largest speed and the
  smoothed interior's is `0.6–1.6`, but check it.
- **The kernel takes a sampled source or an algebraic one, not both.** `ΔH`
  sampled beside the algebraic source needs a new gauge-source `Val` (the
  sampled `Hsrc` holds `H_b` and `∂_a H_b`, 20 variables, `G = 0`), or `ΔH`
  in closed form from the smoothed metric's derivatives. Decide, and say
  which.
- **A projection that fires at `t = 0` means the ranges or the data are
  wrong.** Fix that before reading a run.
- **The sampled `ΔH` is static.** That is fine here. A moving hole would be
  refused, as a moving non-harmonic background with a sampled source is, and
  needs `ΔH` in closed form: a later step's.
