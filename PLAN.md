# TreeGeneralizedHarmonic implementation plan

For the sessions that implement TreeGeneralizedHarmonic, one step at a
time. The **design** lives in `CODE.md` — read it first, in full; it is
authoritative, and this file is only the work breakdown: what each step
changes, what it must not change, and what it must measure and record.
`CLAUDE.md` has the mechanics and the traps. The measured results of the
interior — step 5's layer, steps 8a–8′, and the open questions they
answered — are in `SINGULARITY_HANDLING.md`; where a brief says
"`CODE.md`, Measured results" or "Open questions" for those, read
`SINGULARITY_HANDLING.md`.

**Steps 0–7, 8a–8f and 8c′ are done; the generic interior is *(Done.)*
and step 8g is not needed (`:fitted` reaches `50 M` on 8f's first row).
Step 8, the moving hole at `a = 7/10`, ran on
2026-09-24 and is merged with G5 *not* marked done: seven of its ten items
hold, and the moving layer exports error through its trailing side on G5's
chart (`SINGULARITY_HANDLING.md`, "The interior's questions, opened in step 5
and closed through step 8′"). Step 8′, the bounded round on the
trailing side, ran 2026-09-24/25 and is merged: the side-dependent ramp
(`trail_ramp = 9/10`) removes the trailing side's export, and what is left
— a uniform growth and a drift of the moving spinning hole's `J` — keeps
G5 open, with the recommendation "more interior work, the spin drift
first, before excision" (proposed) awaiting Erik's decision.
Step 9 (G6), which does not depend on it, is running (started
2026-09-25).**

**Steps B1–B9 are proposed and not started**: binary
black holes, toward the milestones G7 and G8 ("Milestones" below), which go beyond
the proof of concept. The design is `CODE.md`, "Binary black holes", and Erik
decides the gauge and interior pairing after B1 and B2. This file stays
until G8 is done.

**Steps C1–C5 are proposed and not started**: a second route to G7 and G8
for holes that orbit — the holes at rest on a grid that rotates with them,
the damped harmonic gauge and the `:fitted` interior. The design is
`CODE.md`, "A comoving grid frame for orbiting holes". They start from steps
B1–B3's code, and Erik decides the route of B4–B9 from step C2's report.

**Steps H1–H5**: a third route to G7 and G8 —
harmonic gauge with no source for the inspiral, the damped harmonic gauge
rolled on before the merger, and the `:fitted` interior. The design is
`CODE.md`, "Harmonic gauge for the inspiral". **Steps H1 and H2 are done**
(they ran 2026-11-11 and 2026-11-12); steps H3–H5 are proposed and not
started, and the grid is Erik's decision from H2's report. Steps H1–H3 are
single holes and need only `main`.

**Steps B1–B3 will not land** (Erik, 2026-11-12):
[eschnett/TreeGeneralizedHarmonic.jl#18](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/18)
stays unmerged. What the C steps and steps H4 and H5 took from it — the damped
harmonic source and the roll-on, `gauge_consistent_momentum`, `HoleSpec` and
the composition, the tracks of several holes — is not on `main`, and where
they start instead is **(open)**.

The steps map onto the milestones G0–G6 ("Milestones" below), split so that every
step ends in a green test suite and a `CODE.md` update, and so that each
is a brief a single agent with a fresh context can carry. The order is
the dependency order.

**Contents**

- [Milestones](#milestones)
  - [Milestones G7 and G8](#milestones-g7-and-g8)
- [Ground rules, every step](#ground-rules-every-step)
- [Running a step as an agent](#running-a-step-as-an-agent)
- [Sharp edges to know before starting](#sharp-edges-to-know-before-starting)
- [Step 0 — Scaffolding (G0)](#step-0--scaffolding-g0)
- [Step 1 — Pointwise algebra (G1a)](#step-1--pointwise-algebra-g1a)
- [Step 2 — Stencils (G1b)](#step-2--stencils-g1b)
- [Step 3 — The right-hand side and the gauge wave (G2)](#step-3--the-right-hand-side-and-the-gauge-wave-g2)
- [Step 4 — Coarse-fine faces, constraints, threads (G3)](#step-4--coarse-fine-faces-constraints-threads-g3)
- [Step 5 — A black hole, static: the interior and the driver (G4a)](#step-5--a-black-hole-static-the-interior-and-the-driver-g4a)
- [Step 6 — The refinement indicator (G4b)](#step-6--the-refinement-indicator-g4b)
- [Step 7 — Horizons (G4c)](#step-7--horizons-g4c)
- [Steps 8a–8g — The generic interior](#steps-8a8g--the-generic-interior)
- [Step 8a — Expectations: anomalous group velocity and leakage](#step-8a--expectations-anomalous-group-velocity-and-leakage)
- [Step 8b — The instrument: range projection and validity monitor](#step-8b--the-instrument-range-projection-and-validity-monitor)
- [Step 8c — Calibrate the layer for an inexact target](#step-8c--calibrate-the-layer-for-an-inexact-target)
- [Step 8c′ — `ρ_max = 4/M` is the default](#step-8c--ρ_max--4m-is-the-default)
- [Step 8d — The tracked horizon geometry](#step-8d--the-tracked-horizon-geometry)
- [Step 8e — The fitted target and the `:fitted` variant](#step-8e--the-fitted-target-and-the-fitted-variant)
- [Step 8f — The measurement matrix](#step-8f--the-measurement-matrix)
- [Step 8g — Excision by a host reference (only if 8f says so)](#step-8g--excision-by-a-host-reference-only-if-8f-says-so)
- [Step 8 — A hole that moves (G5)](#step-8--a-hole-that-moves-g5)
- [Step 8′ — The trailing side](#step-8--the-trailing-side)
- [Step 9 — Infrastructure and the H200 (G6)](#step-9--infrastructure-and-the-h200-g6)
- [Step 10 — Review pass](#step-10--review-pass)
- [Steps B1–B9 — Binary black holes](#steps-b1b9--binary-black-holes)
  - [Sharp edges for steps B1–B9](#sharp-edges-for-steps-b1b9)
- [Step B1 — Two gauge sources and the initial momentum](#step-b1--two-gauge-sources-and-the-initial-momentum)
- [Step B2 — The tracked analytic target](#step-b2--the-tracked-analytic-target)
- [Step B3 — Several holes in one case](#step-b3--several-holes-in-one-case)
- [Step B4 — The symmetric domains and the half turn](#step-b4--the-symmetric-domains-and-the-half-turn)
- [Step B5 — The far boundary, the wave zone and a Teukolsky wave](#step-b5--the-far-boundary-the-wave-zone-and-a-teukolsky-wave)
- [Step B6 — `Ψ₄` and the extraction](#step-b6--ψ₄-and-the-extraction)
- [Step B7 — Merger: the common horizon and the handover (G7)](#step-b7--merger-the-common-horizon-and-the-handover-g7)
- [Step B8 — Constraint-satisfying data from XCTS](#step-b8--constraint-satisfying-data-from-xcts)
- [Step B9 — The inspiral (G8)](#step-b9--the-inspiral-g8)
- [Steps C1–C5 — The comoving grid frame](#steps-c1c5--the-comoving-grid-frame)
  - [Sharp edges for steps C1–C5](#sharp-edges-for-steps-c1c5)
- [Step C1 — The frame's terms](#step-c1--the-frames-terms)
- [Step C2 — A moving hole at rest on its grid](#step-c2--a-moving-hole-at-rest-on-its-grid)
- [Step C3 — The far zone on a rotating grid](#step-c3--the-far-zone-on-a-rotating-grid)
- [Step C4 — Two holes at rest on the grid](#step-c4--two-holes-at-rest-on-the-grid)
- [Step C5 — The orbit](#step-c5--the-orbit)
- [Steps H1–H5 — Harmonic gauge for the inspiral](#steps-h1h5--harmonic-gauge-for-the-inspiral)
  - [Sharp edges for steps H1–H5](#sharp-edges-for-steps-h1h5)
- [Step H1 — A harmonic hole at rest](#step-h1--a-harmonic-hole-at-rest)
- [Step H2 — A moving harmonic hole](#step-h2--a-moving-harmonic-hole)
- [Step H3 — Spin in the harmonic chart](#step-h3--spin-in-the-harmonic-chart)
- [Step H4 — Two holes with no source](#step-h4--two-holes-with-no-source)
- [Step H5 — The orbit in harmonic gauge](#step-h5--the-orbit-in-harmonic-gauge)

## Milestones

Each has an acceptance test; serial `Float64` correctness first. Every
test is three-dimensional and therefore small: `N = 8`, a few roots, one
refinement level, short times.

- **G0 — Scaffolding.** *(Done,* step 0.*)* Its acceptance and what it found are in [`HISTORY.md`](HISTORY.md#milestone-g0); its numbers under `CODE.md`'s [Measured results](CODE.md#measured-results).
- **G1 — Pointwise algebra and stencils.** *(Done,* steps 1 and 2, G1a and G1b.*)* Its acceptance and what it found are in [`HISTORY.md`](HISTORY.md#milestone-g1); its numbers under `CODE.md`'s [Measured results](CODE.md#measured-results).
- **G2 — The RHS on a uniform periodic mesh.** *(Done,* step 3.*)* Its acceptance and what it found are in [`HISTORY.md`](HISTORY.md#milestone-g2); its numbers under `CODE.md`'s [Measured results](CODE.md#measured-results).
- **G3 — Coarse-fine faces, static mesh.** *(Done,* step 4.*)* Its acceptance and what it found are in [`HISTORY.md`](HISTORY.md#milestone-g3); its numbers under `CODE.md`'s [Measured results](CODE.md#measured-results).
- **G4 — A black hole.** *(Done,* steps 5, 6 and 7: G4a, G4b, G4c.*)* Its acceptance and what it found are in [`HISTORY.md`](HISTORY.md#milestone-g4); G4a's numbers are under [step 5](SINGULARITY_HANDLING.md#step-5-the-static-holes-interior-and-the-driver-g4a), G4b's and G4c's under `CODE.md`'s [Measured results](CODE.md#measured-results).
- **G5 — A hole that moves.** Boosted (`|v| ≈ 0.3`), spinning
  Kerr in harmonic coordinates crossing the box — **at `a = 7/10`**
  (`a = 9/10` waits with its price written down,
  [the interior's
  questions](SINGULARITY_HANDLING.md#the-interiors-questions-opened-in-step-5-and-closed-through-step-8)),
  on **the tracked geometry with
  the `:fitted` target** (the analytic core cuts
  harmonic Kerr's disk at `a = 7/10`, so the analytic layer is not
  available on G5's own chart, and runs as the control on the boosted
  `a = 0` hole).
  *Accept:* the indicator's refinement follows the hole, its centroid
  within a few finest spacings of the analytic center at every chunk;
  the layer follows the **tracked** center and shape — the found
  horizon's offset surface, rebuilt every chunk from the track and
  within a cell of the analytic center — with the radius assertions
  holding at every regrid, the fit valid at every row, and the
  initial-data cycle choosing the mesh on the analytic data of the same
  geometry where the chart allows it; the time-dependent Dirichlet data exact at
  the boundary; the masked error stays at the static run's level over
  the crossing and converges at order `q` on the frozen hierarchy; the
  adaptive run matches the uniform-fine reference at fewer points; the
  interior residual at truncation, points released by the core relaxed
  within `1/ρ_max`; `:frozen` measured to fail as predicted; the
  horizon found along the trajectory with its area, mass, spin and the
  boost's contraction recovered.
  **The generic interior (steps 8a–8f) is *(Done.)*** — the leakage
  margin (8a), the range projection and validity monitor (8b), the layer
  rule for an inexact target and `ρ_max = 4/M` (8c, 8c′), the tracked
  geometry (8d), the fitted target and the `:fitted` variant (8e), and the
  measurement matrix (8f) under [steps 8a–8f](SINGULARITY_HANDLING.md);
  excision (8g) is not needed, since `:fitted` reaches `50 M` on the
  matrix's first row. What G5 inherits from 8f: G5's chart at `h = 5/256`
  is 2472 blocks and `630 s` of a node per `M` when the hole sits still; at
  `5/128` it does not survive; `:fitted` holds a boosted hole at `4/M`
  across `1.5 M` of the box, while the analytic `:damped` control on the
  boosted `a = 0` hole needs `ρ_max ≳ 20/M`, its frozen core released on the
  trailing side after about `M` at `v = 0.3`; and a moving hole's step is
  sized for the speed it will have, since the fastest speed grows by
  0.1–0.3 % a chunk and the CFL recheck otherwise stops the run.
  **Step 8 (2026-09-24): G5 is not done; seven of its ten items hold.**
  Item by item (the numbers under [The moving hole (step
  8)](SINGULARITY_HANDLING.md#the-moving-hole-step-8)): *the refinement follows
  the hole* — yes, the
  mesh is re-chosen at every chunk boundary around the tracked layer (the
  `:fitted` cycle on the case's own data, the floor from the core surface
  widened by the travel), with the centroid within `1.4` finest spacings of
  the analytic center over the boosted `a = 0` hole's crossing on the
  analytic layer and within `7.3` on the fitted one; on G5's chart the
  centroid is `6.7` finest spacings off *before anything moves*, the
  boosted spinning chart's own asymmetry, and its per-chunk record is on
  Symmetry, unread (**partly measured**); *the layer follows the tracked
  center and shape, the radius assertions at every regrid, the fit valid at
  every row* — the runs regridded tens of times without an assertion firing
  and without a failed find; the fit's per-row validity on G5's chart is in
  the unread record (**measured on `a = 0`: every fit valid, the track
  within `0.022` cells**); *the initial-data cycle* — done, on the fitted
  data where the chart has no analytic interior;
  *the Dirichlet data exact* — done, bit for bit at `t = 7/10`; *the masked
  error at the static run's level over the crossing* — **not met**: `3.3×`
  in L2 at `7 M` and growing four times as fast; *order `q` on the frozen
  hierarchy* — **measured, `2.3`** in the masked norm over the first
  `0.15 M` of travel; *the adaptive run matching the uniform one at fewer
  points* — **measured to `1 M`**: to three digits at the hole with
  6.5–9.3× fewer points, 2–4× the far-field error, the uniform run lost to a
  `SIGBUS` twice after `1 M`; *the interior residual at truncation, released
  points relaxed within `1/ρ_max`* — **not met**: the trailing side's outer
  layer carries twice the leading side's error; *`:frozen` measured to fail*
  — done, `1.0 M` on the boosted `a = 0` hole (on G5's chart it cannot be
  built: the analytic core cuts the disk); *the horizon along the
  trajectory* — done, `M_irr` `0.04 %`, `J` `0.3 %`, `M_ch` `7e−4` from
  Kerr's and the contraction `0.9534` against `0.95394`. What remains is
  the moving layer's trailing side on G5's chart.
  **Step 8′ (2026-09-25): G5 is still not done.** The side-dependent ramp
  (`trail_ramp = 9/10`) removes the trailing side's excess — the layer's
  outer quarter off the truth `268` behind the hole against `277` ahead at
  `13 M` — so *points released by the core relaxed* holds in the sense of
  "as well behind the hole as ahead of it", though the layer's residual is
  `1.7×` the resting hole's (**item 8 partly met**); *the masked error at
  the static run's level* is **still not met**: `2.05×` at `13 M` (without
  the ramp `5.0×`), growing on both sides; the centroid, with the ramp,
  `5.3`–`14.5` finest spacings off (without it `27`–`32`; **item 1 not
  met** on G5's chart); the track, the fits and the finds hold at every one
  of 53 rows (**item 2 met**); and **item 10 degrades over the crossing**:
  `J` drifts to `0.846` and `M_ch` to `1.030` by `13 M` with and without the
  ramp, the extent ratio to `0.948`, where the resting hole keeps `0.710`,
  `0.9985`, `1.000`. The adaptive run matches the uniform one to `2 M`
  (**item 7 met**); the frozen hierarchy converges at order 2 to `M/4` and
  not at `M/2` in the masked norm (**item 6 partly met**). The open question
  records what is left and the recommendation (more interior work — the
  spin drift first — before excision's price).
  **2026-10-20, outside G5's chart:** a boosted *Kerr-Schild* hole at
  `a = 0`, `v = 1/5`, crosses `8 M` of a frozen capsule mesh on the quadrant
  at the resting hole's level on the analytic layer at `8/M` — the error just
  outside the horizon `1.1–1.3×` the resting hole's to `16 M` at `1/32`,
  `ℋ` `0.8–1.2×` to `40 M`, order four from `r = 3` out, every find with
  Schwarzschild's `M_irr`, `J` and contraction ([the boosted Kerr-Schild hole
  at binary
  speed](SINGULARITY_HANDLING.md#the-boosted-kerr-schild-hole-at-binary-speed-2026-10-20)).
  It says the analytic layer can follow a moving hole; it says nothing about
  G5's chart, spin or `:fitted` layer, and G5 is not done. **2026-10-21:**
  with the target's rate in the moving layer and core the same holds at the
  default `4/M` — `ℋ` just outside the horizon the resting hole's to `3 %`
  from `1/24` on, order four in every shell ([the moving `:damped` layer's
  target
  rate](SINGULARITY_HANDLING.md#the-moving-damped-layers-target-rate-2026-10-21)).
  **2026-10-21, a spinning hole along its axis:** Kerr-Schild at `a = 1/2`
  moving along its spin at `v = 1/5` on the rotating quadrant, `:damped` at
  `4/M`, is at the resting hole's level in `ℋ` (`0.81–1.08×`) to `40 M` at
  `1/24`, `1/32` and `1/40`, and its `J − a` drifts linearly at the resting hole's rate to `0.4–3 %`
  (`4.0·10⁻⁸/M` at `1/32`), converging at order four; `:fitted` keeps `J`
  within `4·10⁻⁶` too ([a spinning hole along its
  axis](SINGULARITY_HANDLING.md#a-spinning-hole-along-its-axis-2026-10-21)).
  So a moving spinning hole as such does not drift, and G5's `0.013` a `M`
  belongs to something this run does not have — the boost across the spin,
  the harmonic chart, or G5's `q = 2`, `v = 0.3`, `a = 7/10` and `m = 4`.
  That is an elimination from one run that differs from G5 in all of these at
  once, not a measurement of any of them: no run has yet moved a spinning hole
  across its spin in Kerr-Schild, or along it in the harmonic chart.
- **G6 — Infrastructure and the H200.** `io.jl`, slice output and
  viewers, `bin/gh.jl`, the per-phase benchmark on threads (TreeWave's
  table, on Symmetry) and **on the H200 in `Float64`**, in-kernel metric
  evaluation on the device, `q = 4, 6, 8` cost, `N = 16` against `32`;
  the three black-box attempts at the RHS kernel — the stencil/algebra
  split, `Float32`, the workgroup shape — each with the kernel's
  registers per thread, spill bytes, achieved occupancy and picoseconds
  per point against the roofline. *Accept:* the G5 run on the H200 in
  `Float64` reaching the same mesh, horizon and analysis record as the
  host; the tables recorded in `CODE.md`, the kernel numbers beside GHAccel's,
  and the defaults chosen from them; the same in `Float32` on the H200
  or on Metal *if it runs*, recorded either way. What G6 does *not*
  promise is a fast GPU kernel: it measures, and hands the measurement
  to the research project under
  [Possible extensions](CODE.md#possible-extensions).
- **G7 — Two holes; G8 — An inspiral** (proposed 2026-10-25). Beyond the
  proof of concept: a head-on collision through merger and ringdown, then
  an equal-mass binary from XCTS data, each with its waveform. The design
  and the acceptance are under [Binary black
  holes](CODE.md#binary-black-holes-proposed), and the steps are
  B1–B9 below.

### Milestones G7 and G8

The design is `CODE.md`'s [Binary black
holes](CODE.md#binary-black-holes-proposed).

- **G7 — Two holes.** Equal-mass, non-spinning holes from rest at
  `d = 10 M` on the reflecting octant, from superposed Kerr-Schild data,
  through merger and ringdown to the wave's passage of `R_ext`, at three
  resolutions. *Accept:*
  - the run stable, with the constraints at truncation level under the
    composite mask and converging;
  - each individual horizon found at every find until the handover, and
    the common horizon found with a nondecreasing area, `J` at zero to
    truncation;
  - the handover completed with no projection hit outside the common core;
  - `Ψ₄`'s `(2, 0)` mode converging, and its ringdown frequency within
    `1 %` of the Schwarzschild mode for the measured `M_f`;
  - `M_ADM − E_rad − M_f` recorded;
  - the extraction validated on the Teukolsky wave and the boundary's
    reflection measured first (steps B5 and B6).
- **G8 — An inspiral.** An equal-mass, non-spinning, quasi-circular binary
  from XCTS data at `d ≈ 10 M`, on the half-turn quarter domain, through
  merger and ringdown, at three resolutions. *Accept:*
  - `Ψ₄`'s `(2, 2)` phase and amplitude converging, with the measured
    order and the phase difference between the two finest at merger
    recorded;
  - `M_f` and `χ_f` within their self-convergence error of Scheel et al.'s;
  - the energy and angular-momentum balances closing to the junk's
    uncertainty;
  - the waveform compared with an SXS waveform of the same mass ratio,
    with the mismatch recorded;
  - the cost recorded.

  Aligned spins, unequal masses and eccentricity reduction come after G8.

## Ground rules, every step

- Read `CLAUDE.md` first, then the `CODE.md` sections the step names.
  The inherited documents are in `notes/`; cite them, do not go looking
  in the sibling checkouts for text.
- **Work on a branch** named `claude/step-N-<slug>`, off `main`. Commit
  there in the style of TreeAMR's history (implement → measure →
  record, measured numbers in the commit body). **Do not merge into
  `main` and do not push**; report the branch and its commits. There is
  no remote yet; when there is one, the rule stands.

  **Inconsistent:** `CLAUDE.md`, "Repository facts", says the remote
  is `git@github.com:eschnett/TreeGeneralizedHarmonic.jl.git`.
- **TreeAMR and SpacetimeMetrics are pinned to their GitHub `main`**
  through `[sources]`. The checkouts at `~/src/jl/TreeAMR` and
  `~/src/jl/SpacetimeMetrics` are *not* what the tests see. If a step
  turns out to need something from either, stop, describe exactly what
  and why, and report — do not edit those checkouts and carry on.

  **Inconsistent:** `CLAUDE.md`, "Things that will bite" ("Five
  dependencies are pinned to GitHub `main`, and TreeAMR comes from the
  registry"), says TreeAMR is General's release under `[compat]`, not a
  `[sources]` pin; SpacetimeMetrics is still pinned there.
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
  incomplete, fix `CODE.md` to describe the current state, record in
  `HISTORY.md` what changed, when and why, and replace each
  **(predicted)** the step measures by the measured number in `CODE.md`. Never loosen an existing
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
  in `CODE.md` as **(proposed)**, recording in `HISTORY.md` that
  step N proposed it.
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
    ask a question, decide, mark the decision "(proposed)" in
    CODE.md, record in HISTORY.md that step N proposed it, and list it
    in the report. End with the report PLAN.md
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

  **Inconsistent:** `CLAUDE.md`, "Things that will bite" ("The RHS
  never mutates `u`, and the state has exactly three writers"), names a
  third writer, the range projection, and says that under
  IMEXRungeKutta one limiter, `gh_limiter!`, is passed as both the stage
  and the step limiter.
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

  **Inconsistent:** `CLAUDE.md`, "Things that will bite" ("The depth
  replaces the radius"), says that on the tracked geometry the layer is
  keyed on the depth below the tracked shape's offset surface, so that
  no radius alone says whether a point is in it.
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
- **`ρ_max` is a physical rate, `4/M` by default** (`M` from `hole_mass`), read once per run; `ρ_max_factor`
  selects the former grid rate `factor/dt`, which RK4's stability bounds
  at about `2.8/dt` on the negative real axis, and a fixed rate above
  `1/dt` is refused.
- **Don't name a keyword `maxlevel`** (it shadows TreeAMR's
  `maxlevel(forest)`); use `maxlevel_cap`.
- **`RK4(; step_limiter!)`** exists in `OrdinaryDiffEqLowOrderRK` with
  the signature `limiter!(u, integrator, p, t)`; `u` is in state layout,
  `statearray(u, U)` gives the block view. Use `adaptive = false`, a
  `dt` from `gh_dt`, and `save_everystep = false`.

  **Inconsistent:** `CLAUDE.md`, "Things that will bite" ("The time
  integrator is IMEXRungeKutta's `RK4()`, by block owner"), says every
  run goes through IMEXRungeKutta's `RK4()` and that OrdinaryDiffEq and
  SciMLBase are no longer dependencies of the package or its tests.
- **Long runs go to Symmetry** through `.claude/orchestration/
  symmetry-run.sh` (the `symmetry-hpc` skill has the mechanics; do not
  export `JULIA_EXCLUSIVE=1` for the one-thread suite). **Wait for a job
  with a blocking command or a polling loop; never "arm a monitor" and
  end the turn** — an agent that does so stalls.

## Step 0 — Scaffolding (G0)

Done (G0). The layout it set up is `CODE.md`'s "File layout", and its
numbers are under `CODE.md`'s "Measured results" (G0); the brief is in
`HISTORY.md`, "Step 0 — Scaffolding (G0)".

## Step 1 — Pointwise algebra (G1a)

Done (G1a). The algebra is `CODE.md`'s "The equations", and its numbers
are under "Measured results" (G1a); the brief is in `HISTORY.md`, "Step 1 —
Pointwise algebra (G1a)".

## Step 2 — Stencils (G1b)

Done (G1b). The weights are `CODE.md`'s "Finite-difference stencils" and
"Kreiss–Oliger dissipation", and its numbers are under "Measured results"
(G1b); the brief is in `HISTORY.md`, "Step 2 — Stencils (G1b)".

## Step 3 — The right-hand side and the gauge wave (G2)

Done (G2). The right-hand side is `CODE.md`'s "One right-hand-side
evaluation" and "Time integration", and its numbers are under "Measured
results" (G2); the brief is in `HISTORY.md`, "Step 3 — The right-hand side and
the gauge wave (G2)".

## Step 4 — Coarse-fine faces, constraints, threads (G3)

Done (G3). What it built is in `CODE.md`, "The interface-order rule, and
what it costs a second-order system", "Analysis quantities" and
"Precision, threads, devices", and its numbers under "Measured results"
(G3); the brief is in `HISTORY.md`, "Step 4 — Coarse-fine faces, constraints,
threads (G3)".

## Step 5 — A black hole, static: the interior and the driver (G4a)

Done (G4a). The design is `CODE.md`'s "The interior: a pointwise damping
layer", and the runs are in `SINGULARITY_HANDLING.md`, "Step 5: the static
hole's interior and the driver (G4a)"; the brief is in `HISTORY.md`, "Step 5 —
A black hole, static: the interior and the driver (G4a)".

## Step 6 — The refinement indicator (G4b)

Done (G4b). The indicator is `CODE.md`'s "Refinement and regridding", and
its numbers are under "Measured results" (G4b); the brief is in
`HISTORY.md`, "Step 6 — The refinement indicator (G4b)".

## Step 7 — Horizons (G4c)

Done (G4c). The horizon rows are in `CODE.md`'s "Analysis quantities", and
its numbers under "Measured results" (G4c); the brief is in `HISTORY.md`,
"Step 7 — Horizons (G4c)".

## Steps 8a–8g — The generic interior

Done: the generic interior is *(Done.)* under G5 ("Milestones" above), and
step 8g was not needed. The design is `CODE.md`'s "The interior: a
pointwise damping layer", and the runs are in `SINGULARITY_HANDLING.md`,
"Step 8a: what crosses the horizon from inside it" and "Steps 8b–8′: the
generic interior and the moving hole". The group's introduction, the four
findings of the design review and the group's sharp edges are in
`HISTORY.md`, "Steps 8a–8g — The generic interior".

## Step 8a — Expectations: anomalous group velocity and leakage

Done. Its table and the leakage runs are in `SINGULARITY_HANDLING.md`,
"Step 8a: what crosses the horizon from inside it"; the brief is in
`HISTORY.md`, "Step 8a — Expectations: anomalous group velocity and leakage".

## Step 8b — The instrument: range projection and validity monitor

Done. Its runs are in `SINGULARITY_HANDLING.md`, "The range projection
(step 8b)"; the brief is in `HISTORY.md`, "Step 8b — The instrument: range
projection and validity monitor".

## Step 8c — Calibrate the layer for an inexact target

Done. The layer rule is in `CODE.md`'s "The interior: a pointwise damping
layer", and the scan in `SINGULARITY_HANDLING.md`, "The layer against an
inexact target (step 8c)"; the brief is in `HISTORY.md`, "Step 8c —
Calibrate the layer for an inexact target".

## Step 8c′ — `ρ_max = 4/M` is the default

Done. Its re-measured numbers are in `SINGULARITY_HANDLING.md`, "The
default rate (step 8c′)"; the brief is in `HISTORY.md`, "Step 8c′ —
`ρ_max = 4/M` is the default".

## Step 8d — The tracked horizon geometry

Done. Its numbers are in `SINGULARITY_HANDLING.md`, "The tracked geometry
(step 8d)"; the brief is in `HISTORY.md`, "Step 8d — The tracked horizon
geometry".

## Step 8e — The fitted target and the `:fitted` variant

Done, both halves. Their numbers are in `SINGULARITY_HANDLING.md`, "The
fitted target, host half (step 8e-i)" and "The fitted variant, kernel half
(step 8e-ii)"; the brief is in `HISTORY.md`, "Step 8e — The fitted target and
the `:fitted` variant".

## Step 8f — The measurement matrix

Done. The matrix is in `SINGULARITY_HANDLING.md`, "The generic interior:
the measurement matrix (step 8f)"; the brief is in `HISTORY.md`, "Step 8f —
The measurement matrix".

## Step 8g — Excision by a host reference (only if 8f says so)

Not needed, and so finished without running: `:fitted` reaches `50 M` on
the first row of step 8f's table. The brief is in `HISTORY.md`, "Step 8g —
Excision by a host reference (only if 8f says so)".

## Step 8 — A hole that moves (G5)

Done as a step (it ran 2026-09-24); G5 is not done ("Milestones" above).
Its runs are in `SINGULARITY_HANDLING.md`, "The moving hole (step 8)"; the
brief is in `HISTORY.md`, "Step 8 — A hole that moves (G5)".

## Step 8′ — The trailing side

Done as a step (it ran 2026-09-24/25); G5 is still open. Its runs are in
`SINGULARITY_HANDLING.md`, "The trailing side (step 8′)", and what is left
in "The interior's questions, opened in step 5 and closed through step
8′"; the brief is in `HISTORY.md`, "Step 8′ — The trailing side".

## Step 9 — Infrastructure and the H200 (G6)

`CODE.md`: "I/O and viewers", "Precision, threads, devices"; milestone
G6 ("Milestones" above); `notes/ghaccel-bench.jl` for the roofline format.

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
  (the likeliest cause of step 8's `SIGBUS`) — bake
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

**Checkpointing exists**: `evolve!`'s
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
`README.md`'s status and `CLAUDE.md`'s "Commands";
move anything still marked **(predicted)** to measured or to "Possible
extensions"; keep this file, and the
binary steps below, until G8 is *(Done.)*.

## Steps B1–B9 — Binary black holes

These take the package past the proof of concept, to two holes that merge
and the waves they emit: the milestones G7 (a head-on collision)
and G8 (an equal-mass inspiral), under "Milestones" above. **The design is `CODE.md`, "Binary black
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
after Erik's decision on the pairing.**

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

`CODE.md`: "The common horizon and the handover"; milestone G7 ("Milestones"
above).

**Changes:**

- **The common finder:** its seed, its `lmax`, and its cadence below
  `d_c`. Failures are recorded, not thrown.
- **Formation** by the proposed criterion.
- **The handover:** the common `:fitted` layer added, the individual
  layers dropped and their tracks retired.
- **The remnant's** source and target.
- **The record's common rows**, and the handover state in the run state.
- **The head-on runs** as a script, with their batch jobs.

**Accept:** G7 as "Milestones" above states it. Also record:

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

`CODE.md`: "What it costs"; milestone G8 ("Milestones" above).

**Changes:**

- **The run scripts and batch jobs:** job chains with SimWatch, at
  `1/48`, `1/64` and `1/80` on the half-turn quarter.
- **The analysis scripts:** extrapolation in radius, the strain by
  fixed-frequency integration, and the comparison with an SXS waveform
  that Erik supplies.

**Accept:** G8 as "Milestones" above states it. Record the cost per `M` and per run
beside the prediction under "What it costs".

## Steps C1–C5 — The comoving grid frame

A second route to G7 and G8 for holes that orbit. **The design is
`CODE.md`, "A comoving grid frame for orbiting holes"; read it in full
first**, then "Binary black holes". Nothing here is started. The steps are a
proposal for Erik's review, and the section's "What the frame asks Erik to
decide" lists what is to be settled.

**What they start from.** Steps B1–B3's code: the damped harmonic source and
the roll-on, `gauge_consistent_momentum`, the position and the time in the
kernels, `HoleSpec` and the composition, the tracks of several holes, and
the PID law of the source line. They are on
[eschnett/TreeGeneralizedHarmonic.jl#18](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/18)
and not on `main`: **the C steps start once that pull request has landed.**

**Order:**

- **C1** first.
- **C2 and C3** may run as two agents at once after C1; C3 needs no hole.
- **Erik decides the route** of steps B4–B9 from C2's report: this design,
  or pairing (α) with the controlled source line.
- **C4**, then **C5**.
- B5 and B6 (the far boundary, the wave, `Ψ₄`) do not depend on the route.
  B4's half turn is simpler on this one: no hole crosses the seam. B7's
  handover is the same with no source to blend.

Each runs from a green suite on the merged branch and ends with the report
under "Running a step as an agent". Branches are named
`claude/step-C<n>-<slug>`.

### Sharp edges for steps C1–C5

- **The rule the design exists for.** The equations never read anything
  derived from a found horizon to better than a few cells, and never an
  analytic hole once the roll-on has ended. A change that feeds a track's
  center, velocity or shape into a target or a source breaks it.
- **The frame's terms change a validated equation.** `(FRAME)` needs its
  own pointwise identity against automatic differentiation
  (`CLAUDE.md`, "Inherit GHSO2's algebra"). Without a frame the kernel is
  today's bit for bit, and the thread digest's lines stand.
- **The basis and the coordinates are two things.** The time index is the
  inertial one; only spatial indices turn. `g_tt` far away is `−1`, not
  `−1 + Ω² r²`: if a far-field component grows with `r`, the basis has
  become a rotating chart.
- **The frame derivative rule.** Everything prescribed on the grid — a
  source's weights and parameters, Dirichlet data, a roll-on's `from` — has
  the inertial time derivative `∂_t Q̄ − V̄·∂̄Q̄ + Ω × Q̄`. A source whose
  `∂_t H` is not that one grows `C_a` with no `H` to explain it.
- **The gauge does not move with the grid.** `Γ_a` is a covector under
  affine maps only, and the grid's map is not one. A closed form evaluated
  on the stored components imposes the *inertial* chart's condition (it is
  invariant under a rotation at one instant, and under nothing that depends
  on time). A source meant to be comoving has to say so: the comoving shift
  is `β + V`.
- **`S0`'s `∂_t g` stays inertial.** Only the two evolution equations gain
  the frame's terms.
- **The map is `C²` in time and constant in kind within a piece**, handed
  over about the piece's time as a `TrackedHole` is, and it goes into the
  run state and the checkpoint.
- **The frame's `Val` is built with the problem**, once per chunk. `V̄`
  depends on the position, which is a `Vec` per axis on SIMD lanes; no
  branch on it.
- **The speed kernel must see `V̄`**, or the step is wrong in the far zone
  and nothing reports it until the run ends there.
- **A comparison between frames is to truncation, not to roundoff.** The
  same run on a rotating grid has other stencils at every point. Compare
  convergence orders and norms; assert bit identity only without a frame.
- **Measure about the hole on the grid.** Shells and masks are about the
  hole's grid point; the analytic center of a boosted hole is `R x̄ + b`.
- **Price every test**, as for the B steps; runs longer than a minute go
  into `test/octant_runs.jl` or `test/binary_runs.jl` options and to
  Symmetry as job chains with their own remote directories.

## Step C1 — The frame's terms

`CODE.md`: "The frame"; "The equations"; "One right-hand-side evaluation";
"The time step".

**Changes:**

- **A map type**, `isbits`: `Φ`, `Ω`, `Ω̇`, `b`, `ḃ`, `b̈` about a piece's
  time, with `x(x̄, t)`, `V̄(x̄, t)` and the rotation of a state between the
  bases. The identity map is no frame.
- **The right-hand-side kernel**: `(FRAME)`'s advection and index terms for
  `h` and `Π`, behind a `Val`, on both kernel widths and on a device.
- **The speed kernel**: the speeds shifted by `−V̄`.
- **Initial data, the error reference and the Dirichlet hook** through the
  map: sampled at `x = R x̄ + b`, rotated into the grid's basis.
- **The sampled and the algebraic gauge sources** under the frame derivative
  rule.
- **The record**: `Φ`, `Ω`, `b` per row; the horizon's `J` rotated back.
- **`test/octant_runs.jl` options**: a constant `Ω` and a constant velocity
  of the grid.

**Accept:**

- **Without a frame** the right-hand side, the monitors and the thread
  digest are bit for bit what they were.
- **The pointwise identity**: `(FRAME)`'s `∂_t ū` against forward-mode
  differentiation of analytic data in the grid's basis — a boosted
  Kerr-Schild hole seen from a rotating and translating grid — to roundoff,
  at both element types.
- **The gauge wave on a rotating Dirichlet box** converges at order `q` to
  the analytic solution, with the error within `2×` the same box at rest.
- **Minkowski with noise** on a rotating box is robustly stable, as the
  octant's rows are.
- **Kerr-Schild `a = 1/2` at the origin of a rotating grid**, `Ω = 1/10`,
  `:damped`, to `8 M` at two spacings: the error and `ℋ` outside the horizon
  within `1.5×` the resting fixture's, order four between the two.
- **The boosted `a = 0` hole on a grid translating with it**, in its own
  Kerr-Schild gauge with `:damped`: the resting hole's row to `1.5×`, with
  no target rate.
- The kernel's cost per point with the frame, on an H200 and on a CPU.

## Step C2 — A moving hole at rest on its grid

`CODE.md`: "The three parts", "The gauge in the frame", "The interior at
rest", "The control acts on the map". **One hole. This is the run that
decides the design.**

**Changes:**

- **The map's control**: the source line's PID law steering `b` (and, for
  step C5, `Φ`) from the found center on the grid, `τ` an option.
- **The far-field weight** of the roll-on: its `from` source kept beyond
  `r_far`.
- **The comoving shift near the hole**, the second candidate of "Which
  shift is damped": `β^i + V̄_A^i` under a weight about the hole, with the
  covector's time component, its `∂_a H_b` by the frame derivative rule,
  and both checked against automatic differentiation of the closed form.
- **`test/octant_runs.jl` options** for all three.

**Accept**, on the boosted `a = 0` hole at `v = 1/5` on the quadrant, on a
grid translating with it, the damped harmonic gauge rolled on over
`σ_g = 10 M` with `:fitted` (`m = 16`, `n_L = 20`, `fit_cont = 2`), at
`1/24` and `1/32`, to `40 M` — step B1's rows, whose data are the
comparison — **under each of the two shifts**, the inertial one and the
comoving one near the hole:

- **The claim:** `ℋ` and `C_a` in `[3, 5)` and `[5, 8)` about the hole
  within `2×` step B1's **resting** row at both spacings, flat from `20 M`,
  and `M_irr` the resting row's. Step B1's moving row is `6–31×` at `1/32`.
- **What separates the two shifts:** the hole's lag behind the boosted
  hole's line (step B1's moving row: `0.49 M` by `40 M`, the gauge's) and
  the horizon's contraction along the boost. Under the inertial shift the
  lag is expected to stay and the constraints to fall; under the comoving
  shift both are expected at the resting row's.
- **The exact image, as a check of the terms:** the resting hole in the
  Galilean chart `x = y + V t` under the resting gauge's full image is the
  resting run to truncation, with `g_ti → −V_i` far away.
- **The hole's drift on the grid** with the map open-loop, and with the
  control: the offset stays under the tracked layer's reach, and `C_a` does
  not read the control's `τ` after the roll-on (two values of `τ`).
- **The faces**: `C_a` in the blocks at the outer faces with and without
  the far-field weight, against step B1's `2.4·10⁻⁴` at `40 M`.
- **The margin in `M`**: `:fitted` at a fixed physical depth at `1/16`,
  `1/24` and `1/32` on the resting hole; record the order outside the
  horizon beside the margin fixed in cells.
- **If the claim fails under both shifts**, do not tune: report the rows
  and what distinguishes them from step B1's resting row.
- The runs in `SINGULARITY_HANDLING.md` under a section of their own, and
  summarised under `CODE.md`'s "The three parts". The report says which
  route its numbers favour.

## Step C3 — The far zone on a rotating grid

`CODE.md`: "The frame" ("The far zone"); "The outer boundary and the wave
zone". **No hole.** Uses step B5's wave if B5 has run, and a gauge wave
packet otherwise.

**Accept:**

- **The step's bound**: the measured largest speed per level on the
  binary's mesh at `Ω = 1/20, 1/10, 1/5`, against `Ω R_ℓ / h_ℓ ≲ λ_f / h_f`;
  the step is the resting mesh's at each.
- **A wave through the far zone** on a grid whose outer faces move faster
  than light: its error against the same wave on the grid at rest, at two
  spacings, and what the coarse-fine faces add.
- **The faces**: frozen Dirichlet data on a rotating cube, flat space with
  noise and the wave, to `10` turns of the grid: stable, with the reflection
  from the outflow half measured.
- `CODE.md`'s **(predicted)** and **(open)** under "The far zone" replaced
  by what was measured.

## Step C4 — Two holes at rest on the grid

`CODE.md`: "The interior at rest", "The control acts on the map"; "One
interior per hole"; "Initial data" (item 1).

**Changes:**

- **A `FittedSpec` in a `HoleSpec`**: `:fitted` per hole in the
  composition, each fit one hole's own, the cache refilled per hole.
- **The roll-on from the composite source on grid-fixed lines** to the
  damped harmonic gauge, for several holes.
- **The map's control from several tracks**: the centers' mean steers `b`.
- **`test/binary_runs.jl` options.**

**Accept**, on step B3's head-on pair (`d = 8 M`, `1/24` and `1/48` at the
holes), with the grid at rest but for `b`:

- one hole in this path is step C2's hole bit for bit;
- every find of both holes at every row to `12 M`, no refusal;
- `C_a` and `ℋ` under the composite mask bounded and falling between the
  two spacings, against step B3's seed-fed and controlled rows at `6 M`;
- each layer's residual from its own fit at truncation level, where step
  B3's tracked targets hold `5.6`;
- the areas constant to the junk's absorption;
- the cost per `M` beside step B3's.

## Step C5 — The orbit

`CODE.md`: "The control acts on the map", "The plunge and the merger";
milestone G8.

**Changes:**

- **The control of `Φ`** from the angle of `c̄_2 − c̄_1`.
- **The quasi-circular superposed Kerr-Schild pair** at `d = 10 M` as a
  case, with `Ω(0)` the data's.
- **The level floors along `x̄`** for the creep.

**Accept**, on the bitant (and on the half-turn quarter if B4 has run), at
`1/48` at the holes, to three orbits:

- both holes found at every row, each within the fine region's margin of
  its grid point, the control's error and `Ω(t)` recorded;
- `C_a` and `ℋ` bounded, and at `1/64` to one orbit lower by the order;
- each horizon's `M_irr` and `J` constant to the junk's absorption;
- the separation's proper length along the axis falling;
- the cost per `M` and per orbit.

What follows — the plunge, the handover and the waveform — is B7's and
B9's on this route.

## Steps H1–H5 — Harmonic gauge for the inspiral

A third route to G7 and G8. **The design is `CODE.md`, "Harmonic gauge for
the inspiral"; read it in full first**, then "Binary black holes" and "A
comoving grid frame for orbiting holes". Steps H1 and H2 are done; nothing
else here is started. The steps are a proposal for Erik's review, and the section's "What this route asks
Erik to decide" lists what is to be settled.

**What they start from.**

- **H1–H3** are single holes and start from `main`: the `Harmonic`
  backgrounds and `harmonic_kerr_case`, both interiors, the octant, the
  quadrant, the rotating quadrant and the bitant, and
  `test/octant_runs.jl`.
- **H4 and H5** start from steps B1–B3's code —
  `gauge_consistent_momentum`, the damped harmonic source and the roll-on,
  `HoleSpec` and the composition, the tracks of several holes,
  `test/binary_runs.jl` — which is on
  [eschnett/TreeGeneralizedHarmonic.jl#18](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/18)
  and not on `main`. That pull request will not land (2026-11-12), so
  where H4 and H5 start is **(open)**.

**Order:**

- **H1** first: it sets the spacing every later step uses.
- **H2 and H3** may run as two agents at once after H1.
- **Erik decides the grid** from H2's report, the moving hole against the
  resting one: the inertial grid with the fine level following the holes,
  or the comoving grid, which needs step C1 first.
- **H4**, then **H5**.

**How the C steps change on this route.** C1 and C3 do not. C2 becomes the
boosted harmonic hole on a grid translating with it, against H1's resting
row: no shift to choose, no roll-on and no far-field weight. C4 and C5 are
H4 and H5 on the grid.

Each runs from a green suite and ends with the report under "Running a step
as an agent". Branches are named `claude/step-H<n>-<slug>`.

### Sharp edges for steps H1–H5

- **Units.** In H1–H3 `M = 1` is the hole's mass. In H4 and H5 it is the
  total mass, each hole has `M_A = 1/2`, and every single-hole spacing and
  radius halves.
- **The horizon is at `M`, not `2 M`.** Its smallest radius is
  `√(M² − a²)` and its equator is at `M`. Every radius copied from a
  Kerr-Schild row — the layer's, the shells', the finest level's cube, the
  capsule's — is wrong by about a factor of two.
- **A harmonic radius is the areal radius less `M`.** A shell `[3, 5)`
  here is not the Kerr-Schild rows' `[3, 5)`. Compare harmonic rows with
  harmonic rows; between charts compare orders and stationarity.
- **The chart's singular set must be inside `r_0`, or inside the core
  surface**: the origin at `a = 0`, where `|h|` reaches `10` at `0.4 M`,
  and the equatorial disk of radius `a` with spin. Nothing checks it for a
  target (`CLAUDE.md`, "A layer target is evaluated wherever the layer
  is").
- **`:fitted` is the route's interior and `:damped` its control.** Run
  both wherever the analytic target fits. A binary has no analytic target.
- **A convergence series keeps the layer fixed in `M`**, the `:fitted`
  margin included; record the margin fixed in cells beside it.
- **A moving layer takes its target's rate.** `trail_ramp = 9/10` helped
  this chart's early rows at `q = 2` and hurt the Kerr-Schild hole at
  `q = 4`: run with and without.
- **`boost(m, v)` moves the hole at `−v`**, and a boosted harmonic
  background is harmonic: it needs no source and is allowed to move.
- **A level more along a capsule has twice the points of the level below
  it**, and the step halves. Price the mesh before submitting:
  Kerr-Schild's `1/48` capsule did not fit one H200.
- **Before `t_0` a run with the roll-on is the run without a source, bit
  for bit.** The source enters the problem at the chunk that begins at
  `t_0`; until then the kernel is the one compiled without a source.
- **`t_0` is run state.** It goes into the loop's carried state and the
  checkpoint, or a restart rolls on at another time.
- **Price every test**, as for the B steps. Runs longer than a minute go
  into `test/octant_runs.jl` or `test/binary_runs.jl` options and to
  Symmetry as job chains with a checkpoint and a remote directory of their
  own.
- **The numbers** go into `SINGULARITY_HANDLING.md` under a section per
  step, and the settings that hold into `CODE.md`'s "Single black holes:
  recommended settings" as harmonic rows.

## Step H1 — A harmonic hole at rest

Done (it ran 2026-11-11). Its runs are in `SINGULARITY_HANDLING.md`, "A
harmonic hole at rest (step H1, 2026-11-11)", its settings in `CODE.md`'s
"Single black holes: recommended settings" (the *harmonic* rows) and "What
harmonic gauge costs"; the brief is in `HISTORY.md`, "Step H1 — A harmonic
hole at rest". The coarsest convergent spacings it hands to step H2 are
`1/24` (`:damped`, `r_0 = 2/5`, `r_1 = 3/4`) and `1/40` (`:fitted`,
`m = 16`, `n_L = 20`).

## Step H2 — A moving harmonic hole

Done (it ran 2026-11-12). Its runs are in `SINGULARITY_HANDLING.md`, "A
moving harmonic hole (step H2, 2026-11-12)", its settings in `CODE.md`'s
"Single black holes: recommended settings" (the *harmonic, moving* rows), its
verdict in "On which grid" and the ladder's item 2; the brief is in
`HISTORY.md`, "Step H2 — A moving harmonic hole". At `v = 1/5` the moving
hole is the resting one from `5/4 M` out with no lag, and a fine level that
follows it costs `0.6–0.8×` the capsule; just outside the horizon the trailing
side carries `1.6–3×` (`:damped`) and `12–18×` (`:fitted`) in `ℋ`; at
`v = 1/2` `:fitted` does not hold. The grid is Erik's decision.

## Step H3 — Spin in the harmonic chart

`CODE.md`: "What harmonic gauge costs" (the spin's room), "The spinning
harmonic chart", "The fitted target". **`a = 1/2` and `a = 7/10`,
`:fitted`.**

**Accept:**

- **The settings** per spin, screened at rest on the rotating octant: the
  margin and the depth that fit between the horizon's equator and the
  disk, and the spacing they need. If `a = 7/10` does not fit one H200,
  report its price and run `a = 1/2`.
- **At rest** to `24 M`: stable, with `M_irr` and `J` constant at the
  Kerr-Schild rows' level and falling with the spacing.
- **Along the spin** at `v = 1/5` on the rotating quadrant, and **across
  it** on the bitant, to `20 M`: within `2×` the resting row outside the
  horizon.
- **The early drift**: `a = 7/10` at `v = 3/10` across the spin drifted in
  `J` by `0.013` a `M` at `q = 2`. The same configuration at fourth order:
  the drift's size and order, at rest and moving.
- Harmonic rows in "Single black holes: recommended settings".

## Step H4 — Two holes with no source

`CODE.md`: "The data", "The merger"; "One interior per hole"; "The common
horizon and the handover". **Starts once the binary steps B1–B3 are on
`main`.**

**Changes:**

- **Superposed boosted harmonic holes** as a case, with `Π` from
  `gauge_consistent_momentum` at `H = 0`.
- **A `FittedSpec` in a `HoleSpec`**, as in step C4.
- **The roll-on from zero**: `H = (1 − f) H_DH`, its `t_0` an event of the
  tracks' separation, in the run state and the checkpoint.
- **`test/binary_runs.jl` options** for the chart, `d_g` and `σ_g`.

**Accept**, on the head-on pair from rest at `d = 8 M`, at `1/96` and
`1/128` at the holes:

- one hole in this path is step H1's hole bit for bit;
- `C_a` at `t = 0` at roundoff, and the data's `ℋ` beside the superposed
  Kerr-Schild pair's;
- **in pure harmonic gauge**: every find of both holes to the end of the
  run, and how it ends — the smallest lapse and the largest `√γ/α` between
  the holes against time. This is a measurement, not a pass;
- **with the roll-on at `t_0 = 0`**, `σ_g = 5 M` and `10 M`: every find to
  the first common horizon, `C_a` and `ℋ` under the composite mask bounded
  and falling between the two spacings;
- each layer's residual from its own fit at truncation level;
- the handover, if step B7 has run;
- the cost per `M` beside step B3's.

## Step H5 — The orbit in harmonic gauge

`CODE.md`: "On which grid", "The merger"; milestone G8.

**Changes:**

- **The quasi-circular superposed harmonic pair** at `d = 10 M` as a case.
- On the inertial grid: the tracked floors for two holes at this spacing.
  On the comoving grid: step C5's control of the map.

**Accept**, on the bitant (and on the half-turn quarter if B4 has run), at
`1/96` at the holes, to three orbits:

- both holes found at every row;
- `C_a` and `ℋ` bounded, and at `1/128` to one orbit lower by the order;
- each horizon's `M_irr` and `J` constant to the junk's absorption;
- the tracks' separation and phase beside the post-Newtonian orbit in
  harmonic coordinates, and the data's eccentricity;
- the cost per `M` and per orbit.

What follows — the roll-on at `d_g`, the plunge, the handover and the
waveform — is B7's and B9's on this route.
