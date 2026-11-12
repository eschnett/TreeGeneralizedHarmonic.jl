# TreeGeneralizedHarmonic.jl — History

How the package got where it is: who decided what and when, the
alternatives that were tried or rejected and why, measurements of code that
no longer exists or that later measurements replaced, upstream bugs since
fixed, and the release history. It is ordered by topic, following
[`CODE.md`](CODE.md)'s sections; [`CODE.md`](CODE.md) describes the current
state and links here where a reason lives here.

The runs behind the black hole's interior are in
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md), not here.

**Step numbers.** Much of what follows is dated by the steps of
[`PLAN.md`](PLAN.md), each a brief for one agent:

- steps 0–7: scaffolding (G0), the pointwise algebra (G1a), the stencils
  (G1b), the right-hand side and the gauge wave (G2), coarse-fine faces,
  constraints and threads (G3), a static black hole with the interior and
  the driver (G4a), the refinement indicator (G4b) and horizons (G4c);
- steps 8a–8g: the generic interior — 8a the expected leakage, 8b the range
  projection and validity monitor, 8c the calibration of the layer for an
  inexact target, 8c′ `ρ_max = 4/M` as the default, 8d the tracked horizon
  geometry, 8e the fitted target and the `:fitted` variant, 8f the
  measurement matrix, 8g excision (not needed);
- step 8: a hole that moves (G5); step 8′: the trailing side;
- step 9: infrastructure and the H200 (G6); step 10: the review pass;
- steps B1–B9: binary black holes (G7, G8);
- steps C1–C5: the comoving grid frame, a second route to G7 and G8;
- steps H1–H5: harmonic gauge for the inspiral, a third route to G7 and G8.

The briefs of the finished steps are under
[The implementation plan](#the-implementation-plan).

**Contents**

- [Goals](CODE.md#goals)
- [Scope and non-goals](CODE.md#scope-and-non-goals)
- [Lineage: what is inherited and what changes](CODE.md#lineage-what-is-inherited-and-what-changes)
- [The equations](CODE.md#the-equations)
- [Discretization](CODE.md#discretization)
  - [Field sets and layout](CODE.md#field-sets-and-layout)
  - [Finite-difference stencils](CODE.md#finite-difference-stencils)
  - [Kreiss–Oliger dissipation](CODE.md#kreissoliger-dissipation)
  - [The interface-order rule, and what it costs a second-order system](CODE.md#the-interface-order-rule-and-what-it-costs-a-second-order-system)
  - [One right-hand-side evaluation](CODE.md#one-right-hand-side-evaluation)
  - [The time step](CODE.md#the-time-step)
- [Gauge and constraint damping](CODE.md#gauge-and-constraint-damping)
  - [The prescribed and algebraic gauge sources](#the-prescribed-and-algebraic-gauge-sources)
  - [The `γ0` profile](#the-γ0-profile)
- [Boundaries](CODE.md#boundaries)
  - [Periodic](CODE.md#periodic)
  - [The outer boundary](#the-outer-boundary)
  - [Reflecting faces](#reflecting-faces)
  - [The rotating octant](#the-rotating-octant)
  - [The quadrant's mirror test](#the-quadrants-mirror-test)
  - [The rotating quadrant](#the-rotating-quadrant)
  - [The bitant](#the-bitant)
  - [The interior: a pointwise damping layer](CODE.md#the-interior-a-pointwise-damping-layer)
  - [The interior: the layer for an inexact target](#the-interior-the-layer-for-an-inexact-target)
  - [The interior: the frozen core and the core rule](#the-interior-the-frozen-core-and-the-core-rule)
  - [The interior: the range projection](#the-interior-the-range-projection)
  - [The interior: the tracked geometry](#the-interior-the-tracked-geometry)
  - [The interior: the fitted target](#the-interior-the-fitted-target)
  - [The quadrant](#the-quadrant)
- [Initial data and backgrounds](CODE.md#initial-data-and-backgrounds)
  - [Where `GHCase` lives, and the boosted Kerr-Schild case (step 3; 2026-10-20)](#where-ghcase-lives-and-the-boosted-kerr-schild-case-step-3-2026-10-20)
  - [The discrete-gradient post-pass (step 5)](#the-discrete-gradient-post-pass-step-5)
  - [The hole cases (step 5)](#the-hole-cases-step-5)
  - [Backgrounds inside kernels (step 0)](#backgrounds-inside-kernels-step-0)
- [Refinement and regridding](CODE.md#refinement-and-regridding)
  - [The error indicator (design review; step 6)](#the-error-indicator-design-review-step-6)
  - [The frozen hierarchy (step 5)](#the-frozen-hierarchy-step-5)
  - [The driver (steps 5, 6 and 8c′)](#the-driver-steps-5-6-and-8c)
  - [The tracked floor and the moving mesh (steps 8d, 8 and 8f)](#the-tracked-floor-and-the-moving-mesh-steps-8d-8-and-8f)
- [Time integration](CODE.md#time-integration)
  - [From OrdinaryDiffEq to IMEXRungeKutta (2026-09-26)](#from-ordinarydiffeq-to-imexrungekutta-2026-09-26)
  - [The first device runs (2026-09-26)](#the-first-device-runs-2026-09-26)
  - [IMEXRungeKutta by block owner (2026-09-26)](#imexrungekutta-by-block-owner-2026-09-26)
  - [The step benchmark](#the-step-benchmark)
  - [IMEXRungeKutta](#imexrungekutta)
- [Analysis quantities](CODE.md#analysis-quantities)
  - [The record's rows (steps 4–8e)](#the-records-rows-steps-48e)
  - [Reductions: `block_mapreduce`, `mesh_mapreduce`, `mesh_fold` (step 4; 2026-10-01; 2026-10-23)](#reductions-block_mapreduce-mesh_mapreduce-mesh_fold-step-4-2026-10-01-2026-10-23)
  - [The `diag` slots (steps 4–8b; 2026-10-23)](#the-diag-slots-steps-48b-2026-10-23)
  - [What steps 4–8d settled in the monitors and the finder](#what-steps-48d-settled-in-the-monitors-and-the-finder)
  - [The horizon provider's interpolation (step 7; 2026-09-26)](#the-horizon-providers-interpolation-step-7-2026-09-26)
  - [Reductions across blocks](#reductions-across-blocks)
- [The coupling contract with TreeGRRMHD](CODE.md#the-coupling-contract-with-treegrrmhd)
  - [The redesign of 2026-11-08](#the-redesign-of-2026-11-08)
  - [Building GH-1 (2026-11-08)](#building-gh-1-2026-11-08)
  - [Building GH-3 (2026-11-09)](#building-gh-3-2026-11-09)
  - [Building GH-4 (2026-11-09)](#building-gh-4-2026-11-09)
- [Public access to the vertex metric](CODE.md#public-access-to-the-vertex-metric)
- [Precision, threads, devices](CODE.md#precision-threads-devices)
  - [The thread workload](#the-thread-workload)
  - [Owner threads (2026-09-25 and 2026-09-26)](#owner-threads-2026-09-25-and-2026-09-26)
  - [`Float32` and the H200](#float32-and-the-h200)
  - [GPU kernel efficiency](#gpu-kernel-efficiency)
  - [The boxed right-hand-side kernel](#the-boxed-right-hand-side-kernel)
- [Running distributed](CODE.md#running-distributed)
  - [The MPI port (2026-10-19)](#the-mpi-port-2026-10-19)
  - [The MPI job before the suite (2026-10-30)](#the-mpi-job-before-the-suite-2026-10-30)
  - [On Symmetry's H200s (2026-10-19)](#on-symmetrys-h200s-2026-10-19)
  - [The MPI commands](#the-mpi-commands)
- [I/O and viewers](CODE.md#io-and-viewers)
  - [No checkpoint and restart](#no-checkpoint-and-restart)
  - [The SimWatch status file](#the-simwatch-status-file)
  - [SimWatch in batch jobs and in the octant runs](#simwatch-in-batch-jobs-and-in-the-octant-runs)
- [Checkpoint and restart](CODE.md#checkpoint-and-restart)
  - [Chains of jobs](#chains-of-jobs)
  - [Where the checkpoint is written](#where-the-checkpoint-is-written)
  - [Requeued jobs on Symmetry](#requeued-jobs-on-symmetry)
- [Upstream prerequisites](CODE.md#upstream-prerequisites)
  - [Item 1: point interpolation](#item-1-point-interpolation)
  - [Items 2 and 3 before they were done](#items-2-and-3-before-they-were-done)
  - [Items 4 to 7](#items-4-to-7)
- [File layout](CODE.md#file-layout)
- [Measured results](CODE.md#measured-results)
  - [Where the measurements came from](#where-the-measurements-came-from)
  - [Single black holes: recommended settings](CODE.md#single-black-holes-recommended-settings)
  - [The analysis cadence on an H200](CODE.md#the-analysis-cadence-on-an-h200)
  - [The right-hand side on an H200](CODE.md#the-right-hand-side-on-an-h200)
  - [The right-hand side on a CPU](CODE.md#the-right-hand-side-on-a-cpu)
  - [The run scripts](#the-run-scripts)
- [Binary black holes](#binary-black-holes)
  - [The design proposed on 2026-10-25](#the-design-proposed-on-2026-10-25)
  - [The comoving grid frame proposed on 2026-11-09](#the-comoving-grid-frame-proposed-on-2026-11-09)
  - [Harmonic gauge for the inspiral proposed on 2026-11-11](#harmonic-gauge-for-the-inspiral-proposed-on-2026-11-11)
  - [Step H1 measured on 2026-11-11](#step-h1-measured-on-2026-11-11)
- [Possible extensions](CODE.md#possible-extensions)
  - [Extensions since built](#extensions-since-built)
  - [Dates of the remaining extensions' notes](#dates-of-the-remaining-extensions-notes)
- [Open questions](CODE.md#open-questions)
  - [Settled in review on 2026-09-16](#settled-in-review-on-2026-09-16)
  - [The interior's questions](#the-interiors-questions)
  - [The streaming-order fused kernel (answered 2026-10-05)](#the-streaming-order-fused-kernel-answered-2026-10-05)
  - [The defaults `m` and `ρ_max` (2026-09-23)](#the-defaults-m-and-ρ_max-2026-09-23)
  - [Gauge sources that know less about the hole (2026-10-02 to 2026-10-20)](#gauge-sources-that-know-less-about-the-hole-2026-10-02-to-2026-10-20)
- [Releases and dependencies](#releases-and-dependencies)
  - [TreeAMR](#treeamr)
  - [Dependencies](#dependencies)
  - [The pins of step 7](#the-pins-of-step-7)
  - [The clean-checkout check](#the-clean-checkout-check)
  - [The `[sources]` pins and TreeAMR's `[compat]`](#the-sources-pins-and-treeamrs-compat)
  - [TreeIOHDF5 against TreeAMR 0.2.4 (2026-10-22)](#treeiohdf5-against-treeamr-024-2026-10-22)
- [The test suite](#the-test-suite)
  - [With the MPI comparison](#with-the-mpi-comparison)
  - [The suite step by step, steps 0–7](#the-suite-step-by-step-steps-07)
  - [The first green CI build (2026-09-21)](#the-first-green-ci-build-2026-09-21)
  - [What the suite costs, and where (measured 2026-09-19 on Symmetry)](CODE.md#what-the-suite-costs-and-where)
  - [The tiers (2026-11-06)](#the-tiers-2026-11-06)
  - [After the SIMD lanes (2026-10-05)](#after-the-simd-lanes-2026-10-05)
  - [The test files](CODE.md#the-test-files)
  - [CI](#ci)
  - [Counts and times](#counts-and-times)
  - [Commands and runs on Symmetry](#commands-and-runs-on-symmetry)
- [The documents](#the-documents)
  - [`CODE.md`'s status line (2026-09-17)](#codemds-status-line-2026-09-17)
  - [The marker (decided), retired 2026-11-09](#the-marker-decided-retired-2026-11-09)
  - [The review of `CODE.md` (2026-09-16)](#the-review-of-codemd-2026-09-16)
  - [`SINGULARITY_HANDLING.md`](#singularity_handlingmd)
  - [The interior's measurements leave `CODE.md` (2026-10-08)](#the-interiors-measurements-leave-codemd-2026-10-08)
  - [PLAN.md](#planmd)
  - [`CLAUDE.md`'s "Current state" (until 2026-11-09)](#claudemds-current-state-until-2026-11-09)
  - [The rules for amending the documents](#the-rules-for-amending-the-documents)
  - [When the sharp edges of `CLAUDE.md` were added](#when-the-sharp-edges-of-claudemd-were-added)
  - [The README's status at milestone G3](#the-readmes-status-at-milestone-g3)
- [The implementation plan](#the-implementation-plan)
  - [Milestone G0](#milestone-g0)
  - [Milestone G1](#milestone-g1)
  - [Milestone G2](#milestone-g2)
  - [Milestone G3](#milestone-g3)
  - [Milestone G4](#milestone-g4)
  - [The plan's status](#the-plans-status)
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
  - [Step H1 — A harmonic hole at rest](#step-h1--a-harmonic-hole-at-rest)

## Goals

- **2026-09-16.** Kernel efficiency is a later research project: decided in
  review.
- **2026-10-25.** The goals were amended: binaries, with the gauge, the
  boundary and the waveform they need, are designed under Binary black
  holes, with milestones G7 and G8 beyond the proof.

## Scope and non-goals

- **2026-09-16 and before.** Second order in space, three spatial
  dimensions only, and no singularity handling were marked (decided):
  inherited from GHSO2 or settled in the review of 2026-09-16.
- **2026-09-23.** The non-goal "No singularity handling" was amended by
  `PLAN.md` steps 8a–8g: the generic interior lifts the restriction to
  spacetimes with a known analytic solution.
- **2026-09-26.** Point interpolation for the horizon finder went upstream.
  Until then the non-goal "No mesh machinery" read: "The one thing this
  package writes that arguably belongs upstream — point interpolation from a
  field set, for the horizon finder — is written as a stopgap and listed
  under [Upstream prerequisites](CODE.md#upstream-prerequisites)." Amended
  2026-09-26: it went upstream — TreeAMR 0.1.3's M11 — and the stopgap is
  gone.
- **2026-10-01.** Checkpoint and restart stopped being a non-goal. Until
  then it read: "**No checkpoint and restart** (decided in review). A proof
  of concept runs to completion; checkpointing arrives with TreeAMR's M9 or
  as a production extension." Amended 2026-10-01, Erik's decision: it
  arrived with TreeAMR 0.1.4's M9a, and G5's runs outlast a queue's day.
- **2026-10-25.** "Analytic initial data, boundary data and gauge sources
  only" was amended: binary data are proposed under Initial data
  (superposed Kerr-Schild, then XCTS through a file).
- **2026-11-08.** "Vacuum only" was amended: TreeGRRMHD's GH-3 will fill
  the stress-energy slot from `(ρ, S_i, S_ij)`.
- **2026-11-09.** GH-3 was built: the non-goal became "Vacuum, unless a
  coupled code supplies the matter", through `gh_rhs!`'s `matter`.

## Lineage: what is inherited and what changes

- **2026-09-26.** The fixed-step RK4 inherited from TreeWave and TreeHydro
  came from OrdinaryDiffEq until 2026-09-26: "RK4 (from OrdinaryDiffEq, and from IMEXRungeKutta since 2026-09-26, as in
TreeAMR's own examples)".
  The table's "RK4, fixed step" read "OrdinaryDiffEq's until 2026-09-26,
  IMEXRungeKutta's by block owner since". Both it and the expanded form
  were marked (decided).

## The equations

- **2026-09-16.** Second order in space, first order in time, and the
  expanded form were marked (decided); that the flux form would cost ghosts
  and buy nothing this mesh can use was decided in review.
- **2026-09-17 (step 1).** The flux identity, the expanded form's agreement
  with it, the closed-form coefficient derivatives and the constraint
  residuals on exact data were measured in step 1, and the argument shapes
  proposed there: "The shapes follow, and step 3's kernel is written against them
**(proposed in step 1**, since `PLAN.md` fixed the argument list but not
its spelling**)**."
- **2026-10-09.** The weight on the normal derivative in `Π_ab`'s
  definition was amended to `√γ`, not `√|g|`.

## Discretization

### Field sets and layout

- **2026-09-16.** Vertex-centered field sets were marked (decided) for a
  finite-difference scheme.
- **2026-09-17 (step 3).** The block size at `q = 6` was amended in step 3,
  where the invariant bit: `FieldSet` refuses `N = 8` at `q = 6`, so
  `PLAN.md`'s "the gauge wave at `q = 2, 4, 6` on `N = 8`" was one row wider
  than the mesh permits.
- **2026-09-17 (steps 4–6).** `diag` holds **ten** slots from step 4
  (proposed in step 4, which is where the set first had to be written
  down). Steps 5 and 6 add the masked error, the interior residual and the
  indicator beside them. That the field sets are vertex-centered and
  `GHProblem` refuses anything else was added in step 4.

### Finite-difference stencils

- **2026-09-17 (step 2).** That the derivative weights are for unit spacing,
  with the `1/h^m` the caller's, was decided in step 2, which is where the
  formula had to be split between the weights and the kernel; `CODE.md` had
  said only that the weights exist. The reason the conversion is emitted
  rather than performed in the generator cost step 2 a debugging session.
  The order of the mixed derivative's sums was decided in step 2, and its
  convergence rates were measured in step 2.

### Kreiss–Oliger dissipation

- **2026-09-17 (steps 2–3).** The dissipation weights' split (the caller
  applies `ε` and a single `1/h_d`) was decided in step 2, the same split as
  the derivative weights, and its damping sign measured there in `Rational`;
  the rates with and without dissipation and the noise runs were measured
  in step 3.
- **2026-09-23 (steps 8a, 8c).** The profile `HorizonDissipation` arrived
  in step 8c. The dispersion analysis inside a horizon and on the spinning
  holes (`test/dispersion.jl`) was measured in step 8a, which also amended
  the sign of the advection: `PLAN.md`'s finding 4 writes the operator as
`(∂_t + b ∂_x)²` with `b = β^r` and the speeds `−b ± a`, and the three are
consistent only with the minus sign.

### The interface-order rule, and what it costs a second-order system

- **2026-09-17 (step 4).** The rates on the two-level mesh were measured in
  step 4, and the mesh and the resolutions proposed there as a budget.
- **2026-11-07.** The suite prints the
fraction on every run and asserts nothing about it: it
is the ratio of two timings on a shared machine, and on GitHub's runners it
crossed one at `p = 6` (macOS, 2026-10-19) and reached 1.19 at `p = 4`
(Linux, 2026-11-07).
  Until 2026-11-07 the suite asserted a bound on the fraction.

### One right-hand-side evaluation

- **2026-09-16.** TreeAMR's three-step contract was marked (decided), and
  the one fused kernel (proposed).
- **2026-09-17 (step 1).** The design said the pointwise algebra is
  scalarised "by hand or by generation, and the choice is an implementation
  detail of G1"; it was decided in step 1: by hand.
- **2026-09-17 (step 3).** That the `∂_t g` the source is given is the
  accumulated `∂ₜh`, dissipation included, was recorded in step 3.
- **2026-09-17 (step 5).** The interior's `Val` was amended in step 5 to
  carry the variant rather than a `Bool`.
- **2026-10-05.** The kernel's evaluation of `W` points at a time on the
  CPU was added.

The kernel as first implemented and measured in step 3 (2026-09-17):

**(Implemented and measured in step 3.)** The kernel is the order above,
written by hand: `h`, `Π` and the thirty `∂_i h` at the point; the
coefficient set and its two contractions `∂_iβ^i`, `∂_i(α√γγ^{ij})`;
then `ntuple(Val(10))` over the components, each forming three `∂_iΠ`,
three compact and three tensor-product second derivatives and, where
there is dissipation, six more contractions — all consumed into two
accumulators before the next component starts; then `gh_node_source`.

Until 2026-10-05 the streaming order was the one below. On 2026-10-05 the
source moved before the Π components, to the order `CODE.md` describes
(**(Implemented 2026-10-05.)** The kernel's order is now: …). The order
before:

1. load `h` and `Π` at the point; form the coefficient set once —
   `g^{ab}`, `α`, `β^i`, `√γ`, `γ^{ij}`, and `∂_iβ^i`, `∂_i(α√γγ^{ij})`
   from the closed-form chain rule — about 30 values, after forming
   the 30 first derivatives `∂_i h`, which are kept;
2. loop over the 10 components: form `∂_i Π_ab` (3 stencils) and
   `∂_i∂_j h_ab` (6 stencils) and the dissipation of `h_ab` and `Π_ab`
   *on the fly*, contract them immediately into `∂_t h_ab` and the
   principal and advective part of `∂_t Π_ab`, and drop them; only two
   accumulators per component survive;
3. add the source `S0 + Z` from `h`, `∂_i h`, `∂_t h` and the
   coefficients — GHSO2's algebra, which GHAccel's generated code of
   the same shape ran without spilling — and the interior profiles of
   the next section; write 20 values of `du`.

The H200 measurements of 2026-10-05 (`CODE.md`, "The right-hand side on an
H200") said:

- The order above changes once the source is lean: the source moves before the Π
  components (step 3 becomes step 2), and the per-component streaming stays.

From 2026-10-05 the right-hand side is **fast on a device**. `CODE.md`, "The
right-hand side on an H200", found the kernel uninlined and spilling at 8.5 ns a
point on an H200, and it now runs at 1.1, and 1.7× faster on the CPU. What changed is in `CODE.md`, "One right-hand-side evaluation"
and "The right-hand side on an H200".

From 2026-10-05 the right-hand side runs on **SIMD lanes on the CPU** (`CODE.md`,
"The right-hand side on a CPU"), described in `CODE.md`'s "One right-hand-side evaluation", with its
measurements under "The right-hand side on a CPU".

`CLAUDE.md`'s "Off by `G`" summarised the linear index of the right-hand-side kernel as: same element, a quarter of
  the time (measured in step 3).

- The streaming-order rule in `CLAUDE.md` was amended on 2026-10-05: the
  source now comes before the Π components. Computed last, as until
  2026-10-05, it was live with the ten accumulators and spilled.
- "On a device a closure is a call" was found on 2026-10-05; the
  uninlined closures' `SVector` arguments, through the stack, were most of
  the right-hand side's 8.5 ns a point before then.
- The interior *variant* became a `Val` in step 5, and the SIMD width `W`
  on 2026-10-05.

### The time step

- **2026-09-17 (step 3).** `λ_max` on flat space and the runs at
  `cfl = 1/4` were measured in step 3, when `CODE.md` said: "The
re-check at the chunk's end arrives with the driver in step 5; the speed
kernel, `max_speed` and `gh_dt` are in `evolution.jl` now."
- **2026-09-17 (step 5).** The mask in the speed kernel and the recheck
  firing on the static Kerr-Schild hole were measured in step 5, when
  `CODE.md` noted on the mask — this section
had not said so and the one above it had.
- **2026-09-24 (steps 8f and 8).** A moving hole's step was sized from `λ`
  times the square of the previous chunk's growth in step 8f, and amended
  in step 8 to never by less than 1 %, after step 8's uniform `5/128`
  control of the boosted `a = 0` hole (measured in step 8); that the recheck
  stays was proposed in step 8.
- **2026-10-01.** The speed is reduced by `mesh_mapreduce(max)`; until then
  by `block_mapreduce(max)`.

## Gauge and constraint damping

- The refusal of a moving non-harmonic background with a sampled source
  was amended on 2026-10-20: the algebraic source `KerrSchildSource`,
  which `kerr_schild_case(; v, x0)` builds, runs it.

### The prescribed and algebraic gauge sources

The prescribed gauge sources, sampled from the background, and the
Gundlach–Pretorius constraint damping with a position-dependent `γ0` were
marked *(decided)* in `CODE.md`'s design. Step 3 built the answers to the two
questions (harmonicity a table, staticity measured) in `src/gauge.jl`.

2026-10-02: the algebraic Kerr-Schild source was proposed under `CODE.md`'s
[Open questions](CODE.md#open-questions) and built the same day; its closed
form against `gauge_source_grad` and its run against the sampled source on
the static `a = 0` octant were measured that day.

2026-10-20: the refusal of a non-harmonic background that is not static was
amended: a boosted Kerr-Schild hole runs with the algebraic source,
`KerrSchildSource` with the hole's lab velocity.

### The `γ0` profile

Step 5 implemented the `γ0` profile in `src/gauge.jl` and proposed its three
numbers, `near = 1/M`, `far = 1/(10 M)`, `width = 3 M`: `CODE.md` asked for
"a Gaussian of width a few `M` … tapered to a small value in the wave zone"
and left them open. The flat-space cases of steps 3 and 4 kept their
arithmetic, a bare number wrapped in `ConstantDamping`.

2026-11-11: TreeGRRMHD's GH-8 note (its branch `claude/step-11-dynamical-tov`,
commit `ff4e8f1`) reported that `γ0 = 1/M` grows a mode on its TOV star, and it
left open which scale sets the threshold. Erik asked for the measurement on this
package's Plummer star, run on Symmetry. Twenty-five runs went to `500 M`: three
stars (`R/M = 5, 10, 20`), `γ0 M` from 0 to 10, `γ2 = −1/2` and `+1`, and
`h = R/12` and `R/24`. They ran as jobs 595935 and 595936 on `amdpreq`, several
runs to a node. The first job was pre-empted six times and continued from its
checkpoints each time. The results are in `CODE.md`, "Gauge and constraint
damping": within the Plummer family the onset is fixed in `γ0 R`, and the TOV
star's is five times lower. The data are in `.claude/data/star-damping` (not
committed), and on Symmetry under
`/mnt/beegfs/eschnetter/claude/TreeGeneralizedHarmonic/star-damping/out`.
`star_case`'s `γ0 = 1` is `1/(10M)`, not the `10/M` that GH-8 and a comment
in `test/matter_tests.jl` said, and it is below the onset, so it was left as it is. The
star and its matter moved from `test/matter_tests.jl` to `test/plummer_star.jl`,
which the study script shares.

## Boundaries

### Periodic

Step 3 amended the boundaries: shifted Minkowski is not periodic in `x`, and
is Dirichlet in `x` and periodic in `y` and `z`.

### The outer boundary

Step 3 measured the Dirichlet hook bit for bit against `state_tuple`, and
shifted Minkowski's order through it. Step 8 (2026-09-24) measured the hook
on G5's own moving background in `test/moving_tests.jl`. Every moving run of step
8 fills its boundary this way at every stage.

### Reflecting faces

2026-10-02: reflecting faces (TreeAMR's M10) and the hole on the octant
(`hole_case(; octant = true)`) were added, and the octant was measured
against the box that day, in `test/reflection_tests.jl` and on the static
Kerr-Schild `a = 0` hole.

From 2026-10-02/03 a single hole runs **on an octant**, and the static
Kerr-Schild hole has a gauge source that needs no sampling. The reflecting faces are TreeAMR's M10; the source, the run tools,
`src/simwatch.jl` and `FittedSpec(; fit_cont = 2)` came with them, and
`CODE.md` describes each. SimWatch's `summary` and `[history]` window
arrived on 2026-10-19, and SimWatch 1.3.0's `group`, `fraction` and a
chain's `previous_job_ids` on 2026-10-21. `fit_cont = 2` differences the
interpolated gradient because TreeAMR 0.1.4 had no second-derivative
interpolation (TreeAMR's `main` had it, unreleased). The octant's numbers
are in `SINGULARITY_HANDLING.md`'s "Single holes on the octant".

### The rotating octant

2026-10-04: the rotating octant (TreeAMR 0.1.7's M12 seam) was added and
measured against the box in `test/rotation_tests.jl`.

From 2026-10-04 a **spinning** hole has an octant too, on **TreeAMR 0.1.7**'s
rotating seam (M12): the case's seam and its maps are in `CODE.md`, "The rotating octant". `test/rotation_tests.jl` is its file (29 claims, `9.4 s` at
four threads inside the suite, which then measured **4824 assertions in
13m21**); `CODE.md`, "The rotating octant", has the numbers.

### The quadrant's mirror test

2026-10-20: the quadrant was added. Its mirror test in
`test/boosted_tests.jl` measured, against the box `[−5/2, 5/2]³`: after 16 RK4 steps the two agree to
`5.8·10⁻¹⁶` relative at every point outside the core.
From 2026-10-21, with the moving layer's target rate, the `x` quadrant's
mirror test has met the same mechanism as the rotating quadrant's:
`3.8·10⁻¹¹`, where it was `5.8·10⁻¹⁶`.

### The rotating quadrant

2026-10-21: the rotating quadrant was added and measured against the box in
`test/boosted_tests.jl`.

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

(The analysis cadence that set the production runs' `chunk = M` and
`split=0` is `CODE.md`'s "The analysis cadence on an H200".) The **aligned campaign** then moved Kerr-Schild
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

### The bitant

2026-10-24: the bitant was added and measured against the box in
`test/boosted_tests.jl`. The same day the least horizon radius across the
spin was amended: `horizon_min_radius` of a boosted hole became the shortest
semi-axis of the boosted ellipsoid (`horizon_ellipsoid`). 2026-10-25: the
transverse campaign on the bitant was measured ([spin across the
motion](SINGULARITY_HANDLING.md#spin-across-the-motion-the-bitant-2026-10-24)).

From 2026-10-24 a **spinning hole moves across its spin** on the **bitant**:
`hole_case(; octant = :bitant)` is `[−L, L]² × [0, L]` with the mirror at
`z = 0` alone, for a hole in that plane, spinning about `z` and moving in the
plane (`kerr_schild_case(; a, v, x0)`, `axis = 1` or `2`). `gh_forest` takes
the brick `(2r, 2r, r)`. It is half the box and twice the quadrants' points,
because no rotation maps that hole onto itself (`CODE.md`, "The bitant").
`horizon_min_radius` of a boosted spinning hole is now exact across the spin,
the contracted spheroid's shortest semi-axis (`horizon_ellipsoid`), and it is
unchanged bit for bit where the old bound was exact. The **transverse campaign**
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

### The interior: a pointwise damping layer

#### The design

Step 8f rewrote `CODE.md`'s interior around the four choices (geometry,
target, ramp, rate), with step 5's design as the first row of each. The
tracked offset surface is step 8d's, the fitted target step 8e's, the ramp
rule step 8c's and the leakage margin's path integral step 8a's (steps
8a–8e: 2026-09-23). Step 8f found that the analytic core cuts G5's chart's
disk at every `h` the proof of concept can afford, and proposed the fit in
`Π̃ = (α/√γ)Π`. `ρ_max = 4/M` was decided on 2026-09-23 (step 8c′); step 5
had started with the grid rate `1/dt`. Step 8f measured that a moving
analytic layer wants more and proposed `ρ_max ≳ 20/M` for it. Step 8f's
matrix proposed that excision (step 8g) is not needed and that the analytic
`:damped` layer stays the default wherever the chart's singular set fits
inside the core.

#### The analytic layer

*No excision* was marked *(decided)* in `CODE.md`'s design; the pointwise
profiles, the relaxation rather than a slowed evolution alone, and the
smooth layer rather than a hard paste were decided in review (the pure mask
`ρ = 0` was the first proposal). Step 8d amended the layer: for a case that
asks for it, the profiles are functions of the depth below the tracked
horizon's offset surface. 2026-10-20: for a moving hole the `:damped` layer
adds its target's rate `(1 − w) ∂_t u_exact`, and the moving `:damped` core
relaxes toward `u_exact` (the table's rows); the same date amended the claim
that a boosted hole is an exact solution of `(INTERIOR)` in the layer.

#### The margin

Step 8a (2026-09-23) proposed the two rules, the stencil margin and the
leakage margin, for the reviewer to confirm; measured the frozen-coefficient
margins on harmonic Kerr's equator and axis; and found that `ε_KO` rising
inside the layer does not buy margin, recommending to step 8c that its
`ε_KO(r)` profile rise from the horizon. Step 8c measured the hard step
(E0) against step 8a's rate and proposed that the default `m = 8` stay.

#### The spinning harmonic chart

That a ball cannot hide Kerr's singularity in the harmonic chart at
`a = 9/10` was found in step 5, when that was the proof-of-concept case
("found in step 5, and this is the proof-of-concept case"), and step 5 raised the two ways out as an open question.

#### The interior's profiles and rate

Step 5 implemented the profiles in `src/interior.jl`, proposed the ramp
widths as halves of the layer, and measured that the smoothstep must clamp
its result. The grid rate `ρ_max · dt = 1` was the
default from step 5, **(proposed)** there because it relaxes by a factor
`e` per step and looked as strong as a layer needs to be; step 8c measured
why it is not ([the profiles and the rate](CODE.md#the-profiles-and-the-rate) has the
numbers). Step 8c′ (2026-09-23) amended it: `4/M` became the default, and
step 8c′ measured the default at `0.036/dt` to `0.092/dt` on the suite's
holes and what it costs the layer.

Dates of `CLAUDE.md`'s sharp edges about the interior:

- The state's third writer, the range projection, came with step 8b; "a `NaN` in the core is a hit" is step 8b's, and "Step 5's two failures are not deep" was measured in step 8b.
- "`ρ_max = 1/dt` is a grid rate, and a wrong target cannot survive it" was measured in step 8c; `4/M` has been the default from step 8c′.
- From step 8d: a `HoleCenter` that is the tracked trajectory, the depth in place of the radius (the old `in_layer(int, r)` is gone), and the shape as a series.
- From step 8e: the fit in ADM variables, the state sampler reading the layer, the shift's constant term (measured; "G5 wants it on."), the `cont = 2` fit on harmonic Kerr at `a = 9/10` ("8e-ii's initial data for
  that chart cannot be the `cont = 2` fit as planned"), the cache (decided in review, step 8e) and its refill, the `C¹` initial data, the first find from the seed's shape, and the harmonic `a = 9/10` fit measured not good enough, with the remedy left to step 8f ("8f decides").
- From step 8: the moving layer's export through its trailing side (measured), `horizon_min_radius` found to be a bound for a boosted hole, and the tracked floor from the core surface (amended in step 8). From step 8′: the narrowed trailing ramp (measured).
- From step 5: the masked slot written through a branch, the gauge-source sample that once skipped the core rule, and the spherical core that cannot hide harmonic Kerr's singularity at `a = 0.9` (found in step 5, and it is the proof-of-concept case).

- The grid rate `ρ_max · dt = 1` (the option `ρ_max_factor = 1`, what
  step 5's numbers need) was the layer's default rate until 2026-09-23,
  when step 8c′ made `4/M` the default (`CLAUDE.md`'s sharp edge on
  `ρ_max` was amended then).
- `boost(m, v)` moving the hole at `−v` was found in step 8d and fixed in
  step 8e. On 2026-09-26 the rule "never test a metric on its singular
  point" was added to that sharp edge: whether rounding leaves `r` at
  `2e−17` or at zero is the platform's (a huge `g_tt` on Apple silicon,
  `NaN` on the x86 CI runners).
- 2026-10-20: the moving `:damped` core was made to relax in ADM
  variables, and it was noted that a tracked `:damped` core stays frozen.
  The need for the target's rate in a relaxation toward a moving target
  was measured on 2026-10-20 and the rate built on 2026-10-21.

### The interior: the layer for an inexact target

Step 8c (2026-09-23) added the layer for an inexact target: step 8e's target would be a fit of the evolved state, not a solution, so step 8c calibrated the layer against targets that are wrong on purpose (E1–E3), with the optional `target`, `ρ_max_fixed` and `HorizonDissipation` as its three knobs. Proposed in step 8c: the refusal of a fixed rate above `1/dt`, the sign of E3's error, the ramp's width as the width over which `ρ` rises, the ramp rule `n_L = max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)` for other orders and spacings, `ε_KO` built but off by default, and that the tracked center of step 8d has to be good to a cell. Measured in step 8c: the findings that `ρ_max` is a physical rate, `4/M`; that the ramp is at least `4G = 8` cells; that finding 1's rule holds for `ρ_max M ≥ 4`; that `ε_KO` stays constant for a smooth target; and that the layer absorbs a tracking error of about `h`. `4/M` became the default in step 8c′, Erik's decision of 2026-09-23: from then a run with neither `ρ_max_fixed` nor `ρ_max_factor` relaxes at `4/M`, and the grid rate is asked for as `ρ_max_factor = 1`.

### The interior: the frozen core and the core rule

The core rule was implemented in step 5 (`src/interior.jl`), where `+ẑ` at the center was proposed and two bugs were fixed: the gauge-source sample, taken without the core rule, made the constraint monitors report `NaN`; and step 4's monitor kernels wrote a masked slot as `keep * value`.

2026-10-20: a moving hole's `:damped` core follows its target (amended), relaxing in ADM variables, after the plain linear relaxation ended its run in its first chunk; the same day, the core's outer half was given its target's rate (`core_target_rate`). 2026-10-22, during the aligned campaign: the metric's part of that rate became an ADM rate (`core_relaxation_rate(h, h_t, ρ, ∂ₜh_t)`), and a moving core's rate became `max(ρ_max, κ |v|/r_0)`, `κ = 12` (`core_rate`).

### The interior: the range projection

The range projection was added in step 8b (`src/bounds.jl`). Proposed in step 8b: its call on the initial data, the gate `r_gate = r_1 − 2 G h` and the ranges of `default_bounds`. Measured in step 8b: that it is idempotent on the state and on the flag, and that it never fires on step 5's two runs that end.

Under OrdinaryDiffEq, before 2026-09-26, the projection was the stage limiter alone and the paste the step limiter. Read against `OrdinaryDiffEqLowOrderRK` 2.2.5, RK4 calls the
stage limiter on its three intermediate stages and then on `u`, before
the FSAL evaluation and before the step limiter: four calls per step.
From 2026-09-26, under IMEXRungeKutta, both writers are one limiter, `gh_limiter!`, passed as the stage and the step limiter alike (decided 2026-09-26 by Erik); the description of the four calls was amended the same day.

### The interior: the tracked geometry

The tracked geometry was added in step 8d (`src/tracking.jl` and the second half of `src/interior.jl`); its protocol for both geometries (`in_layer` of a position and a time, `interior_point`, …) was amended then. Proposed in step 8d: `n_L = 0` resolved by `evolve!` at the default `4/M` under the grid rate; the ramps `ρ_ramp = 1`, `w_ramp = 1/2` (the step's brief said `1/2` for both); the refusal of a find that moves `r_min` by more than `G h/2`; the layer's `h` read from the blocks it lives in; the real harmonics' slots and conversion, which step 8e's fit then shared; the clamped surface and its exact fast paths; the seed sampled and truncated rather than interpolated; the footprint guard exact by enumeration, over the brief's "use the bounding sphere `r_in − offset`"; the loop's four decisions; `evolve!`'s `find`; and the damping profile and `HorizonDissipation` on the analytic trajectory. Measured in step 8d: the sphere as the tracked geometry bit for bit, the conversion's roundoff, the truncation's cost, the margin's e-folds, the cost, and the static hole. `margin_efolds` moved into `src/` from `test/dispersion.jl` in step 8d, as `PLAN.md`'s hand-over from step 8c asked. In step 8 (2026-09-24) a moving seed's `r_min` became the least radius of its analytic surface (amended), measured on G5's hole.

**A boost moves the hole the other way** was found in step 8d and fixed in step 8e. Before the fix, `HoleCenter`'s docstring called `v` "the
coordinate velocity of `boost(background, v)`", which is the wrong sign, and
a case built without a `velocity` keyword carried zero. At the fix: No existing test, fixture or `hole_runs.jl` section passes a
`velocity`, and the one boosted case the suite builds (`gauge_tests.jl`'s
harmonic neighbour of the refused Kerr-Schild) asserts only that it is
accepted; nothing in G4 moves, so nothing measured changes.

### The interior: the fitted target

The fitted target was added in step 8e (`src/fit.jl`, with its kernel half in `src/evolution.jl`, `src/constraints.jl` and `src/driver.jl`): the fit and the seven pieces `build_fit` meets in step 8e-i, the `:fitted` variant and its five pieces in step 8e-ii. Decided in review in step 8e: `shift_constant = true` as the default, validity as Lorentzian-ness and not a range, the target's own ranges, the cached target, the refill rule, the variant, the initial data, and harmonic Kerr `a = 9/10` at `m = 4` as the proof-of-concept chart. Proposed in step 8e: the contravariant shift, the chain rule for both samplers, the shift's `ρ²` and `ρ⁴` terms, `sampler(xs, ns)` (over the brief's `sampler(xs)`), the state sampler's unguarded footprint, the analytic sampler's fourth-order differences (where the brief asked for central differences), the row weights, the block-by-block residual, the sweep's fractions of `r_1`, the factors of `derive_target_bounds`, the cache's form and its extra cell, the refill rule's pieces, the cache refilled from the same fits after a regrid, the fit's coasting, the `:fitted` residual and the error kernel's evaluation, and the first find from the seed's shape. Every measurement of items 1–12 is step 8e's.

In step 8e the shift first had no `ρ⁰ ỹ_00` term by default (so `β(0) = 0`, as `PLAN.md` asks); step 8e measured that default wrong for a moving hole, and the review made `shift_constant = true` the default. Before `derive_target_bounds`, step 8e's measurement of harmonic Kerr `a = 9/10` (item 7) ended with the clause — the proposed bounds are Kerr-Schild's, and harmonic `a = 9/10` needs
     its own before 8e-ii projects into them.

Item 12's chart was decided in review in step 8e as the proof-of-concept chart. Its status at the end of step 8e: **So the proof-of-concept chart does not run on this target
   at `h = 5/256`**; step 8f's row for it needs `Π̃` and `L ≥ 12` at the
   least, and the design's own fallbacks are `a = 7/10` and 8g's excision.

Item 13, the moving target, was added in step 8 (2026-09-24): the target's rate fed forward (`target_rate`) and the initial data blended across the layer (`fit_initial_blend`) were proposed and measured there.

### The quadrant

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

## Initial data and backgrounds

### Where `GHCase` lives, and the boosted Kerr-Schild case (step 3; 2026-10-20)

The analytic backgrounds from `SpacetimeMetrics` were marked "(decided)" in
the original design. `GHCase`'s place in `initialdata.jl` rather than in
`driver.jl` was amended in step 3: the right-hand side
needs a case two steps before there is a driver, and a struct cannot be
defined twice. On 2026-10-20 the case's constructor was amended so that,
with the algebraic Kerr-Schild source, the boosted Kerr-Schild hole is a case.

### The discrete-gradient post-pass (step 5)

The post-pass option was measured in step 5 (`discrete_gradient_momentum!`),
against G4's prediction "not beyond the first chunk"; the measurement, which
confirmed it, is in `CODE.md`, "Initial data and backgrounds".

### The hole cases (step 5)

`kerr_schild_case` and `harmonic_kerr_case` over a shared `hole_case` were
added in step 5.

### Backgrounds inside kernels (step 0)

Evaluating the background inside kernels was marked "(decided, and a
dependency risk)" in the original design. The CPU measurement that both
`KerrSchild(1, 0)` and `boost(Harmonic(1, 9/10), 0.3 x̂)` compile into
`fill_by_coordinates!`'s kernel and fill a field set bit for bit as a host
loop does was made in step 0.

## Refinement and regridding

### The error indicator (design review; step 6)

The error indicator was decided in review of the design (by 2026-09-16, when
`CODE.md` had been reviewed three times), replacing prescribed spheres. Its
fields, the ten components of `h`, were first marked **proposed**; step 6
proposed and implemented one amplitude `U_ref` for all ten components rather
than one per component. The derived floor level `L_h` was proposed in step
6, and the prediction that the floor never binds on a calibrated run was
confirmed in step 6. The two things the ceiling needed — dilating the
travelling margin before the bounds are applied, with `buffer = 0` passed to
`regrid!` (proposed in step 6), and refusing a block that is both in the
horizon shell and within the boundary margin — were found in step 6.

### The frozen hierarchy (step 5)

The frozen hierarchy was implemented in step 5 as `hole_forest` in
`initialdata.jl`, before the indicator existed (which was step 6's):
"Before the indicator exists there is nothing for it to have chosen, so
the frozen hierarchy is built by hand".

### The driver (steps 5, 6 and 8c′)

The driver, TreeHydro's one loop with cases as data, was marked "(decided)"
in the original design. Step 5 implemented it in `src/driver.jl` without the
regrid:

**(Implemented in step 5**, `src/driver.jl`, without the regrid.**)** The
loop exists and runs the static hole; `evolve!` takes `regrid = false`
and **refuses `true` by name** until step 6 supplies the indicator, for
the reason the refusal says: this section's loop flags with the masked
Löhner verdict and with nothing else, and a driver that regridded on some
other criterion now would be a second refinement mechanism to delete
later. For the same reason `evolve!` takes the **forest** as a keyword
rather than building one: step 5's mesh is the frozen hierarchy of
`hole_forest`, and step 6's is what `adapt_to_initial_data!` produces
from the indicator.

In step 5 `ρ_max` was `1/dt`, rebuilt with the interior at the top of every
chunk. Step 8c′ (2026-09-23) amended it: the default is now `4/M`, the same
in every chunk, and the rebuild stays.

Step 6 implemented the indicator in `src/refinement.jl` and the loop's two
branches (`adapt` and `regrid`), and amended three of its points in step 6:
`τ` materialised into `diag` with the marks taken from that field, `regrid!`
handed the state field set alone, and the regrid skipped after the last
chunk.

### The tracked floor and the moving mesh (steps 8d, 8 and 8f)

Step 8d amended the floor to read the horizon the layer actually follows,
and proposed that the next geometry is rebuilt on whatever the indicator
chose. Step 8 (2026-09-24) amended the section with "the mesh follows the
hole", its four pieces each proposed in step 8, and measured the boosted
`a = 0` hole (`hole_runs.jl moving=ctl`). The travelling margin `buffer` of
`|v| · chunk` cells dates from step 6. Before step 8 the initial-data cycle
of a `:fitted` case was step 8f's: Step 8f's cycle flagged on the analytic `:damped` layer and refused G5's chart, where the analytic core cuts the disk. The first regrids of a tracked geometry along a trajectory are step 8's `moving` rows.

## Time integration

### From OrdinaryDiffEq to IMEXRungeKutta (2026-09-26)

The integrator was amended on 2026-09-26, at Erik's direction, from
OrdinaryDiffEq's RK4 to IMEXRungeKutta's classical RK4 with its stage
arithmetic by block owner, as TreeAMR's own examples now use it;
OrdinaryDiffEq until then, whose decision and measurements follow as
history. IMEXRungeKutta 1.2's `reuse` dates from the same day. The original
decision, as `CODE.md` stated it:

**Fixed-step RK4 from `OrdinaryDiffEqLowOrderRK`** (decided), as
TreeWave uses it: four stages for a fourth-order spatial scheme, the
RHS in the SciML signature `f!(du, u, p, t)`, `adaptive = false`, `dt`
from the CFL bound above. TreeWave measured what this costs on many
cores — the integrator's serial stage broadcasts cap a threaded step at
about 3.6× however well the mesh scales, and `thread = True()`
oversubscribes the cores KernelAbstractions already uses — and this
package accepts the same cap for the same reason: correctness and a
shared pattern first. GHSO2's native tableau-generic stepper with fused
stage updates is the extension that lifts it, listed below with what it
needs.

Measured in step 3, under OrdinaryDiffEq: The integrator is `RK4()` with
`adaptive = false`, `save_everystep = false` and `dt` from `gh_dt`, and it
is what every convergence rate above was measured through.

The relaxation rate's bound was amended in step 8c′, and the limiters, under
OrdinaryDiffEq, were: the `:pasted` variant uses RK4's `step_limiter!(u, integrator, p, t)`
hook, and step 8b's range projection its `stage_limiter!` — the two places
the state may be written outside the RHS, both passed as `solve` keywords
**(amended in step 8b**; see [The
interior](CODE.md#the-interior-a-pointwise-damping-layer)**)**.

What changed with IMEXRungeKutta (2026-09-26):

- **The limiters: one, on every state vector** (decided 2026-09-26 by
  Erik). Under OrdinaryDiffEq only the projection reached
  the stage values, the paste was a step limiter alone, and the FSAL
  tendency was evaluated on the result before the paste.
- **The result agrees to roundoff, not bitwise.** The stages are summed in
  a different order: the gauge wave's `L2` error at `q = 4`, `N = 16` is
  `2.2860177409713e−5` against OrdinaryDiffEq's `…409697e−5`, and the
  suite's hole fixture agrees to twelve digits at `t = 1/5`; a moving
  `:fitted` hole (`hole_runs.jl moving=l0-base t_end=1/4`, seven refills
  swapped into its chunks' integrators) prints the same row as `main` to
  every digit. By owner it is
  bitwise its own broadcast and bitwise across thread counts, which
  `test/thread_workload.jl` now digests (its hand-written RK4 is gone).
- OrdinaryDiffEq allocated about fifteen state-sized vectors per
  `solve`, serially. IMEXRungeKutta 1.2's `reuse` (amended 2026-09-26)
  takes the previous chunk's four scratch arrays over.

The step benchmark was measured on 2026-09-26 on Symmetry (`bench/stepping.jl`,
cn096, jobs 563975 and 563982; `CODE.md`, "Time integration", has the
current rows). The rows of the code before the change:

| per call, pinned, first touch | gauge wave | hole |
|---|---|---|
| RHS, `main` (the boxed kernel, below) | 482–530 | 755–786 |
| OrdinaryDiffEq RK4 step, `main` | 3146–3180 | 4354–4504 |
| OrdinaryDiffEq RK4 step, fixed kernel | 920–947 | 1605–1665 |
| `solve` per 4 steps, per step: OrdinaryDiffEq on `main` (the old driver) | 3825–3924 | 5118–5160 |

So at 64 threads a step of the old driver was **5.6×** (wave) and **4.3×**
(hole) today's. About 3–4× of that was the right-hand-side kernel itself
(the `Core.Box` of "Precision, threads, devices": ~570 bytes allocated per
point, 1.2 GB per evaluation, which 64 threads turn into garbage-collector
contention — the same bug cost 7–15 % at four threads on the development
machine), and 1.5–1.6× the integrator (its serial stage arithmetic and its
per-`solve` buffers). A step by owner is now `4.2–4.9` right-hand sides.
Job 563749's anomaly — the owner-mapped
update 3–5× slower under first touch, suspected NUMA balancing — did not
reproduce: after a process's OrdinaryDiffEq rows (every one a serial pass)
the owner-mapped step was no slower than before them. The driver end to
end: per step, `main` against today, **4111 →
544 ms** and **5297 → 1140 ms** pinned, **3460 → 843** and **4547 → 1516**
unpinned, with the records equal to every printed digit.

The block-size scan of `CODE.md`, "Time integration", was measured on
2026-09-26 too.

### The first device runs (2026-09-26)

**On a device (measured 2026-09-26**, the first device runs of this
package**)**. With the box gone the right-hand-side kernel compiles for a
GPU (Metal at `Float32`, one H200 at `Float64` and `Float32`, job 563978,
cn113; `CODE.md`, "Time integration", has the rest). The timings, before the
kernel was tuned on 2026-10-05:

| H200, ms | wave `16³` | hole `16³` | wave `32³` | hole `32³` |
|---|---|---|---|---|
| RHS, `Float64` | 20.7 | 28.8 | 151.2 | 170.5 |
| RK4 step, `Float64` | 84.3 | 116.2 | 613.8 | 691.7 |
| `evolve!` per step, `Float64` | 85.2 | 120.2 | | |
| RHS, `Float32` | 13.0 | 20.6 | | |
| `evolve!` per step, `Float32` | 53.3 | 86.9 | | |

At the same 2.1 M points one H200 runs the right-hand side **6.4×** (wave)
and **8.9×** (hole) as fast as a pinned 64-core node, and the hole's
`evolve!` step **9.5×** as fast as the 1140 ms it takes on cn112; `32³`
blocks, 16.8 M points, cost about 10 ns a point. `Float32` is 1.4–1.6×
faster than `Float64` (not investigated), and its error is
roundoff-dominated (`1.9e−5` against
`1.7e−6` on the hole). None of this is tuned: G6's kernel efficiency is
still a research project, and these are the numbers it starts from.
**(Taken up 2026-10-05**, in [The right-hand side on an
H200](CODE.md#the-right-hand-side-on-an-h200): on the device the
kernel is not inlined and spills 8 KB a thread; inlined, with a register-lean
source, it is 7× faster.**)**

### IMEXRungeKutta by block owner (2026-09-26)

From 2026-09-26 the **integrator is IMEXRungeKutta's RK4 by block owner**
(`src/stepping.jl`); OrdinaryDiffEq and SciMLBase are gone from both
environments, and the same change removed a `Core.Box` from
`gh_rhs_kernel!`, which let the right-hand side compile for a **device
for the first time** (`CLAUDE.md`'s "Current state").

### The step benchmark

`bench/stepping.jl`, the step benchmark of `CLAUDE.md`, "Commands", was
added on 2026-09-26.

### IMEXRungeKutta

`src/stepping.jl` and IMEXRungeKutta's `RK4()`,
  from 2026-09-26; OrdinaryDiffEq and SciMLBase are no longer dependencies
  of the package or its tests. The two limiters became one under IMEXRungeKutta (from 2026-09-26) (Erik's decision).

## Analysis quantities

From 2026-10-23 the package requires **TreeAMR 0.2.6**, and **every norm is
a fold over points**, its M15 `mesh_fold`, which this package asked for, described in `CODE.md`, "Analysis quantities". `test/octant_runs.jl`'s `observe` runs each
monitor once and folds three times — the whole mesh, the levels, the
boundary band, the shells and the half-shells in one accumulator — where it
re-ran the ADM monitor and the error kernel once per shell and half-shell to
select points and made about 280–350 reductions; its CSV is the old one to
the ten digits it writes on the quadrant, the rotating octant and a
`:fitted` hole at `a = 0` (`CODE.md`, "Analysis quantities").

### The record's rows (steps 4–8e)

The table's rows were added or amended step by step: the interior residual's
`:fitted` reading (`|u − u_fit|`) was amended in step 8e; the range
projection's rows and the validity monitor's layer and shell bands were added
in step 8b; the validity monitor's evolved-region rows, the horizon track
and the tracked layer rows were added in step 8d; the fitted-target rows
were added, and proposed, in step 8e.

### Reductions: `block_mapreduce`, `mesh_mapreduce`, `mesh_fold` (step 4; 2026-10-01; 2026-10-23)

As first implemented (step 4), norms were `block_mapreduce` partials weighted by each block's `h³`, combined in
block order, so they are bit-identical across thread counts. **(Amended
2026-10-01:** they are TreeAMR's `mesh_mapreduce` — the same partials, the
weight `h³` applied by TreeAMR, the combination on the host, which is the
one place TreeAMR's M7 combines the ranks' partials, by an `allgather`
folded in rank order — and so is every other
reduction that crosses blocks: the speed, the indicator's scales and
`τ_max`, the projection's counts and the validity monitor's extremes. They
are still bit-identical across thread counts. `mesh_mapreduce` combines
pairwise where this package's loop summed left to right, so the L2 norms
moved in the last place and nothing printed to the digits this document
records did; `max`, `min` and the counts are unchanged bit for bit. The
denominator is `evolved_volume`, one weighted reduction of the mask.**)**

On 2026-10-23 the norms became TreeAMR 0.2.6's folds over points (M15),
with `record_norms` new. The kernel that
wrote them into four `diag` slots, and the non-finite kernel's slot, are
gone: `diag` has 17 slots. A record row made 37 reductions (54 on an ADM chunk, 65 with the
indicator) and makes 2 (4 with the indicator); `test/octant_runs.jl`'s row,
which re-ran the ADM monitor and the error kernel once per shell and
half-shell only to select their points, runs each monitor once and folds
three times. The weight is now per point, `p.h³`, where `mesh_mapreduce`
scaled a block's sum: a sum may move in the last place (`threading_tests.jl`'s
exact claim held unchanged, its spacings being powers of two), maxima, minima and counts
are unchanged bit for bit, and each block is still one sequential fold on
its owner's thread, so the norms are bit-identical across thread counts.

### The `diag` slots (steps 4–8b; 2026-10-23)

`diag` had ten slots after step 4 (the speed, `C_a`, `ℋ`, `ℳ_i`, the mask
indicator). **(Thirteen from step 5**: the masked error, the interior
residual and the gauge drift, appended rather than inserted, because
the two contiguous runs above must not move. **Fourteen from step 6**:
the refinement indicator `τ`, appended for the same reason.**)**
`diag` has **22** slots from step 8b: the projection's three, the
monitor's four and the non-finite count, appended. **(Amended
2026-10-23:** 17 — the monitor and the count are folds over the state
vector, see [Analysis quantities](CODE.md#analysis-quantities)'s constraints
paragraph.**)**

### What steps 4–8d settled in the monitors and the finder

The constraint monitors were implemented and measured in step 4
(`src/constraints.jl`); leaving the Kreiss–Oliger term out, assembling the
Ricci tensor, `metric_derivatives_along` and dividing by the evolved volume
were proposed there. The error rows were implemented and measured in step 5,
which proposed the error magnitude and the gauge drift's shell. The range
projection's and validity monitor's rows were implemented in step 8b
(`src/bounds.jl`), which proposed the per-call hit count and amended
`finite` to the evolved region's. The mesh-statistics row's `τ_max` and
centroid were implemented and measured in step 6, which proposed the
centroid and measured its bias. The horizon finder was implemented and
measured in step 7 (`src/horizon.jl`), which proposed the footprint guard,
the radii from the analytic center, the recorded failed find and the
uniformization tolerance; step 8d amended the finder to take `center =`
and return the radii about the found surface's own origin, and amended "the
finder is a diagnostic" for a case whose interior is a `FittedSpec`. On
2026-10-22 the analysis-cadence measurement on an H200 amended the table's
"about one RHS" for the ADM monitor.

### The horizon provider's interpolation (step 7; 2026-09-26)

Until 2026-09-26 the provider interpolated with a stopgap of its own: The
provider is the one place this package interpolates: point location by
`find_leaf`, then Lagrange interpolation of order `q + 2` over the
containing block's stored points (which reach `G` into the neighbors,
so a point near a block face is interpolated without crossing it), on
the host at analysis cadence — a stopgap for TreeAMR's "generic
interpolation" to-do, and the first item under
[Upstream prerequisites](CODE.md#upstream-prerequisites). On 2026-09-26 it was
amended to TreeAMR's `interpolate` (M11) with the basis `Lagrange(q + 2)` —
the same stencil — and the footprint guard was amended with the port to
TreeAMR's M11 (`UnevolvedRegion`).
One batch of 496 interpolated `ADMVars` took 0.26 ms in step 7 and 0.23 ms
through the stopgap on the day of the port (`CLAUDE.md`, until 2026-11-09).

### Reductions across blocks

Every reduction that crosses blocks has been TreeAMR's `mesh_mapreduce` (from 2026-10-01; the fold from 2026-10-23).

## The coupling contract with TreeGRRMHD

From 2026-11-08 the **coupling contract with TreeGRRMHD is 3+1 variables
at this package's vertices, in both directions** (Erik's redesign; `CODE.md`, "The coupling contract with
TreeGRRMHD").

### The redesign of 2026-11-08

The coupling contract — neither package depends on the other, and a short script owns the loop and calls each package's verbs in the order the data flows — was decided by Erik 2026-11-05 (TreeGRRMHD's `CODE.md`, "The coupling contract"). On 2026-11-08 Erik redesigned what crosses between the two packages, and where: that both directions cross at this package's vertices, in 3+1 variables, was decided by Erik 2026-11-08, and that every conversion between centerings is TreeGRRMHD's was Erik's decision of 2026-11-08, from TreeGRRMHD's proposal.

This reverses TreeGRRMHD's amendment of 2026-11-06. That amendment made GH-1 a `FourMetric` adapter here, answering `g_ab` and `∂_c g_ab` at every point of every consumer centering, 4th order in the outermost ghost layer of a `G = 3` cell set, and it made GH-3 a covariant `T_ab`. The redesign also reverses the metric half of 2026-11-05's "each interpolating its own data".

### Building GH-1 (2026-11-08)

GH-1's adapter, `ADMMetric`, was built 2026-11-08 in `src/admvars.jl`, and the CPU and Metal measurements under "What this package provides (GH-1)" are of that day. That the consumer's set needs `N ≥ 10` at `G = 4` was found 2026-11-08, building the test.

### Building GH-3 (2026-11-09)

GH-3, the matter term, was built 2026-11-09. Its parts are `matter_source` in
`src/pointwise.jl`, the head's call in `src/evolution.jl`, the lanes' adapter
call in `src/lanes.jl`, the ADM monitor's half in `src/constraints.jl`, and
`matter` on the integrator in `src/stepping.jl`. The measurements under "What
this package takes (GH-3)" are of that day.

- **The entry point is a keyword on `gh_rhs!`**, `gh_rhs!(du, u, p, t; matter
  = nothing)`, decided by Erik 2026-11-09. The alternative was a field of
  `GHProblem`, set by a `with_matter`. It was rejected for two reasons.
  - The adapter is a snapshot that a coupled run replaces at every stage.
  - A field would have had to pass through `with_interior`, `refill_target`
    and the checkpoint's carried state.

  GH-4's `rhs!` is to forward to the keyword.
- **The seventh `Val`.** `W`, the SIMD width, became the right-hand-side
  kernel's sixth `Val` on 2026-10-05, and GH-3's `HASMATTER` became the
  seventh on 2026-11-09.
  - It is the one `Val` not built in `GHProblem`'s constructor: `gh_rhs!`
    builds it from the bound adapter's type.
  - It goes last in the kernel's arguments with the adapter, so that
    `bench/rhs_cpu_lab.jl`, which slices the argument list by position, kept
    working.
- **The physical test** was added at Erik's request, 2026-11-09: "anti-Cowling",
  the matter held fixed and the spacetime evolved, on an analytic static star
  in the quick tier (Erik's choice over a TOV star solved in the test).
  - The acceptance first used a separate bump metric, a lapse, conformal and
    shift bump on `(−2, 2)³`.
  - It was replaced by the same star in a tilted slicing, because
    `EinsteinTensor` costs 75 s to compile per metric type.
  - The star's first box, `(−4, 4)³`, was too coarse for `N = 8 … 16`: `ℳ_i`
    converged at 3.29 there.

### Building GH-4 (2026-11-09)

GH-4, the verbs a script calls, was built 2026-11-09. The parts are:

- `prepare!` and `rhs!` in `src/evolution.jl`, as `gh_rhs!`'s two halves;
- `max_speed(p, u, t)` there, which until then was the driver's unexported
  `max_speed_of`;
- `stage_limit!`, `step_limit!`, `initial_state!` and `evolved_sets` in
  `src/stepping.jl`;
- `LaneArray` in `src/lanes.jl`.

The measurements under "The verbs (GH-4)" are of that day.

- **`prepare!` returns the problem**, decided by Erik 2026-11-09, as
  TreeGRRMHD's `prepare!` (built in its step 10) returns its own. The
  alternative was to return GH-1's adapter, which was the open item
  "whether GH-4's `prepare!` returns the adapter".
- **`initial_state!` and `evolved_sets` were added** at Erik's choice,
  2026-11-09. GH-4's request lists only `prepare!`, `rhs!`, the limiters and
  the speed. TreeGRRMHD's step-11 sketch calls these two as well, and
  without them a script would reach into `p.U`.
- **A segment broke the SIMD lanes.** The first run of the script test
  failed in the kernel's store. `Lanes` admitted only a `DenseArray`, so the
  reshape of a segment view fell through to the scalar `setindex!` with a
  `Vec`. It now admits a contiguous view (`LaneArray`).
- **`prepare!` converts `t` to the problem's type** for the Dirichlet hook,
  as `ADMMetric(p, u, t)` did. `gh_rhs!` passed it unconverted. The
  integrator converts its span to `T`, so a run is unchanged, and so are the
  `du` hashes.

## Public access to the vertex metric

`adm_vars` (`src/admvars.jl`) was added 2026-11-05, for TreeGRRMHD's GH-1. The refusal of a spacing `inv_h` of another precision than the array's was added in review, 2026-11-05. The measurements under "Measured" there (the CPU, the convergence, the allocation, Metal, and the H200 in Symmetry jobs 590175 and 590181) are of 2026-11-05.

From 2026-11-05 a downstream code can **read the metric at a vertex in ADM
variables**: `adm_vars`, TreeGRRMHD's GH-1 (`CODE.md`, "Public access to the
vertex metric").

## Precision, threads, devices

- The rule that a parallel loop of this package's own is TreeAMR's helper
  dates from TreeAMR's owner-based threading (2026-09-25). `threaded_foreach`
  went with the stopgap interpolator's batch on 2026-09-26, and RK4's stage
  arithmetic is owner-based from 2026-09-26.
- The SIMD-lane rule (the algebra must stay generic in its number type)
  was added on 2026-10-05.
- The two spellings that disagree, `metric_derivatives`'s wrapper and its
  coefficient-set method, were measured in step 1.

### The thread workload

The bit-identity across thread counts was measured in step 4: it holds. That the workload carries the gauge monitor and not the ADM one was proposed in step 4. The bullet first said that `test/thread_workload.jl` runs "a gauge wave with a regrid and a chunk of a hole with its layer"; it was amended in step 5: the workload stays the gauge wave and does not grow a chunk of a hole with its layer. That sentence was written before the interior existed and promised one.

### Owner threads (2026-09-25 and 2026-09-26)

The rule that every per-block pass runs on the block's owner thread was added 2026-09-25, with TreeAMR's owner-based threading of 2026-09-23, `4be726e`, when `[compat]` asked for TreeAMR `0.1.2`. The one parallel loop this package wrote itself, the horizon interpolator's batch over query points, was `Threads.@threads` and is TreeAMR's `threaded_foreach` now: the same chunk on the same thread every call, nestable, and it rethrows the body's own exception, so the footprint guard's refusal still reaches the record readable. (Amended 2026-09-26: that loop is gone with the stopgap.)

**What is not owner-based is RK4's stage arithmetic.** OrdinaryDiffEq's `RK4()` forms `uprev + (dt/2) k` and the final combination with a serial broadcast (`thread = Serial()`), on the calling thread, over the whole state. **(Measured 2026-09-25** on the development machine, the gauge wave at `q = 4`, 64 blocks of `16³`:**)** the four broadcasts of a step are `5.7 ms` against `4 × 519 ms` of right-hand side at one thread (0.3 %) and `4 × 143 ms` at four (1.0 %). On a 64-core Symmetry node it is not measured, and is expected to matter more for two reasons: a single core's bandwidth is a small share of the node's, and the broadcast reads every block from one core between launches, which is exactly the migration TreeAMR removed. It also first-touches the vectors the integrator allocates (its copy of `u`, the stage vector `tmp`, the saved end state) from that one core, so TreeAMR's advice — pin the threads, then let first touch place the pages — makes the field sets domain-local and leaves the state vectors on one domain. Today's Symmetry jobs neither pin nor interleave; pinned, with and without `numactl --interleave=all`, is the comparison to make before choosing. The ways out are `RK4(; thread = True())` (Polyester, whose chunk-to-thread map is its own) or an integrator whose stage update is a `map_blocks!` — a change to "RK4 from OrdinaryDiffEq", and G6's to measure. **(Resolved 2026-09-26.)** Measured on Symmetry (job 563749), the serial updates were 12–14 % of a step at 64 threads, and every `solve` added the serial allocation of about fifteen state-sized vectors; the integrator is IMEXRungeKutta's RK4 by block owner since, which removes both — [Time integration](CODE.md#time-integration) has the numbers. The placement question is answered there too: pinned with first touch is the fastest configuration, and the 3–5× first-touch anomaly of job 563749 did not reproduce once nothing serial touched the state.

### `Float32` and the H200

That `Float64` on Symmetry's H200 is the requirement was decided in review. The `Float32` claims were measured in step 4, on `CPU()`; what ran at `Float32` was everything steps 3 and 4 built, and the two-resolution rule was step 2's. `metric_derivatives_along` is step 4's.

### GPU kernel efficiency

The predicted budget was measured 2026-10-05, in "The right-hand side on an H200".

### The boxed right-hand-side kernel

The rule that a closure in a kernel must not capture a name assigned twice in that body was found 2026-09-26.

## Running distributed

From 2026-10-19 a run is **distributed over MPI**, and so runs on several
GPUs, one per rank (TreeAMR's M7, released since 0.1.5, which already did
the exchange, the regrid, the reductions, the interpolation and, in
TreeIOHDF5, the checkpoints). The interface and the rank-0 analysis are in `CODE.md`, "Running
distributed". Fixed on the way: the travelling margin's flags are gathered as
`RegridMark`s before `buffered_flags`, the refinement centroid read leaves by
a local index, `add_noise!` draws from a `Xoshiro` per block (its values
changed), checkpoint rotation removes part files on rank 0, the wall-clock
decisions and the fits' timings are agreed.

- The rule that the host analysis is rank 0's, served by the other ranks,
  was added on 2026-10-19, and so was the HPC-X measurement (2026-10-19).
- `forest.leaves[b]` with a local `b` is right on one rank and wrong on
  every other — the refinement centroid had it until
  2026-10-19.

### The MPI port (2026-10-19)

Running distributed was added 2026-10-19, on TreeAMR's M7, released since TreeAMR 0.1.5. That the host analysis runs on rank 0 alone was decided 2026-10-19. With the per-block draw, `add_noise!`'s values changed with the scheme. The MPI comparison (`test/mpi_tests.jl`) was measured 2026-10-19 on the development machine.

### The MPI job before the suite (2026-10-30)

Running the three ranks to their end before the suite, where the machine has no room for them beside it, is an amendment of 2026-10-30. Until then they ran after it, beside a test process that had compiled everything, and that cell's runner went down in five of its thirteen attempts from 2026-10-23 to 2026-10-30 ("lost communication", "received a shutdown signal"), each time while the ranks ran. The memory measurement that explains it is of 2026-11-06.

### On Symmetry's H200s (2026-10-19)

The measurements on Symmetry's H200s (jobs 578514, 578572 and 578648) are of 2026-10-19, and the choice of HPC-X's Open MPI with `--bind-to none` was decided by the measurement that day. Before the core binding was found, the bullet on HPC-X's 2.5–3× in job 578514 went on to explain it: the module also puts nvhpc's CUDA libraries on the library path, which CUDA.jl warns of, and its UCX progress may compete with the ranks' threads. Not traced; MPICH_jll is the one to use until it is.

Corrected 2026-10-19: the step-alone result had said that the ghost fill and prolongation were most of a step, as on the CPU; on a GPU the kernel is ("Where a step goes, rank by rank"). HPC-X's 2.5–3× in job 578514, in the analysis and not the step, was found 2026-10-19 to be Open MPI's core binding.

### The MPI commands

The standalone MPI workload command (`test/mpi_workload.jl`) and the
rank-by-rank breakdown (`bench/mpi_breakdown.jl`) were added to `CLAUDE.md`,
"Commands", on 2026-10-19. The suite's MPI test has been skipped below
24 GB since 2026-10-19, and has run its ranks before the suite where the
machine has no room since 2026-10-30. CI set `TREEGH_TEST_MPI=1` for its
threaded Linux cell until 2026-11-06, and from then for its `mpi` job.

## I/O and viewers

### No checkpoint and restart

The design first had no checkpoint and restart:

- **No checkpoint and restart** (decided in review): a proof of concept
  runs to completion. When it is needed it is a few dozen lines over
  TreeAMR's leaf keys and the state vector, or TreeAMR's M9; it is under
  [Possible extensions](CODE.md#possible-extensions). **(Amended 2026-10-01:** it
  is TreeAMR's M9a, and [Checkpoint and restart](CODE.md#checkpoint-and-restart)
  is its section.**)**

### The SimWatch status file

The SimWatch status file was added 2026-10-02, and the observer's fourth argument (the chunk's record row) the same day. It was amended 2026-10-21 for SimWatch 1.3.0, when the thumbnails became SVG, instead of dark-on-light PNGs.

### SimWatch in batch jobs and in the octant runs

`bench/simwatch.sh`, the batch jobs' copy of SimWatch's shell helper, dates
from 2026-10-21. `test/octant_runs.jl`'s `simwatch.toml` has carried
sparklines of the constraints, the error and the horizon, and thumbnails,
from 2026-10-19; from 2026-10-21 the thumbnails are SVG, in place of the
dark-on-light PNGs.

## Checkpoint and restart

Checkpoint and restart was added 2026-10-01, on TreeAMR 0.1.4, whose M9a adds `save_checkpoint` and `load_checkpoint` through a package extension on HDF5. Erik's decision of that day reversed "No checkpoint and restart" above; the mechanism is TreeHydro's — its `CODE.md`, "Checkpoint and restart", 2026-09-29 — on purpose, so that the applications of TreeAMR checkpoint alike, with the two differences marked. **Amended 2026-10-09:** TreeAMR 0.2.0 moved the checkpoint functions, with their names, signatures and file format, out of its HDF5 extension into the companion package TreeIOHDF5, which this package now calls.

The "No checkpoint and restart" it reversed is under "I/O and viewers" above. That the checkpoint code is a hard dependency, that the file is written after the analysis row and before the regrid, and that the recipe and the criterion are kept apart were decided 2026-10-01; the chains and the costs were measured 2026-10-01. Until 2026-10-09 it was HDF5 that was the dependency here, loading TreeAMR's `TreeAMRHDF5Ext`.

Since step 8d the run state is a good deal more than TreeHydro's accumulators, and the old note under [Possible extensions](CODE.md#possible-extensions) — "bit-identical, since the layer and the hook depend on `(x, t)` only" — was no longer true. The gauge source's half-cell re-sample rule is step 8d's, the speed growth step 8f's, and the moving harmonic hole of the chain test runs with step 8′'s trailing ramp.

From 2026-10-01 **a run checkpoints and restarts**
(Erik's decision of 2026-10-01, reversing `CODE.md`'s "no checkpointing"):
`checkpoint.jl` over TreeAMR's M9a, and `evolve!`'s checkpoint keywords
(`CODE.md`, "Checkpoint and restart"). Two
differences from TreeHydro, both Erik's: HDF5 (TreeIOHDF5 from 2026-10-09)
is a **hard dependency**, and
the checkpoint is written **before the regrid**; `CODE.md`, "Checkpoint and restart", has why.

### Chains of jobs

`test/hole_runs.jl`'s `checkpoint=<dir>` and `walltime=<s>` for the
`generic` and `moving` rows, which make a long row a chain of jobs, were
added on 2026-10-01.

### Where the checkpoint is written

After the analysis row and before the regrid (decided 2026-10-01; TreeHydro writes after it).

### Requeued jobs on Symmetry

- A Symmetry node failure requeuing a job from scratch was found on
  2026-10-03.
- **A requeue hit a run that had finished** (2026-10-22, a Slurm
  controller outage). `a9v2h72r` (job 580825) had reached `20 M`. The
  requeued job restarted from its checkpoint at `t = 20`, `evolve!` refused
  ("there is nothing left to run"), the batch script marked the SimWatch
  status "failed", and the job's log was overwritten. The CSV and
  `records.csv` were complete.

## Upstream prerequisites

### Item 1: point interpolation

Point interpolation from a field set was written here first as a stopgap; TreeAMR's `TODO.md` listed "generic interpolation". It was written in step 7, in `src/horizon.jl`: `locate_block` — the descent TreeAMR does not export, root brick to finest cell to the first ancestor that is a leaf — and `interpolate`/`interpolate_grad`, the tensor-product Lagrange window of `q + 2` points with its value and its gradient formed in one pass. The two properties now under item 1 were the two things the port upstream would have to keep.

Done upstream, 2026-09-26: TreeAMR 0.1.3's M11 kept both properties and adds the ones the stopgap lacked: one binary search per point instead of `maxlevel` `find_leaf`s, device execution, periodic and reflecting faces. TreeAMR 0.1.3 is registered, and `Project.toml` takes it from General (`[compat]` `0.1.3`, no `[sources]` entry). The port removed `locate_block` and the exported `interpolate`/`interpolate_grad`, whose name collided with TreeAMR's.

### Items 2 and 3 before they were done

2. **A device reduce-to-scalar** (TreeAMR `TODO.md`) would let the
   per-chunk norms stay on the device; today `block_mapreduce` copies
   one value per block back, which is fine at chunk frequency.
   **(Done upstream:** TreeAMR 0.1.2 reduces on a device in two launches,
   and `mesh_mapreduce` is the one-number form; this package reduces
   through it from 2026-10-01.**)**
3. **MPI (M7) and I/O (M9)** are on TreeAMR's roadmap and this package
   is written so they arrive transparently: no host loop over blocks
   assumes all blocks are local, and every reduction is a
   `block_mapreduce` or a `firing_boxes`. **(Amended 2026-10-01:** every
   reduction that crosses blocks is a `mesh_mapreduce` — the host folds
   over `block_mapreduce`'s per-block vector, which would have been
   per-rank answers under M7, are gone — and M9's checkpoint half, M9a,
   is in use: [Checkpoint and restart](CODE.md#checkpoint-and-restart).**)**
   **(Amended 2026-10-23:** or a `mesh_fold`, TreeAMR 0.2.6's fold over
   points, which carries many results through one collective.**)**
   **(Done upstream and in use, 2026-10-19:** M7 has been released since
   TreeAMR 0.1.5, and [Running distributed](CODE.md#running-distributed) is what
   this package had left to do — four host paths that were not, after all,
   free of the assumption.**)**

### Items 4 to 7

The measurements under items 4 and 5 are of 2026-10-05, when item 5 was added. Items 6 and 7 were noted 2026-10-19, each Erik's suggestion; the removal of an index's part files came with the MPI port.

## File layout

Where the files of the layout came from: `src/precision.jl` and `src/device.jl` were amended in step 0 (TreeHydro's `hostcopy!` split). In `src/pointwise.jl`, `metric_derivatives`, `gh_node_rhs_expanded` and `gh_node_source` were added in step 1 and `metric_derivatives_along` in step 4. `src/stencils.jl` was added in step 2, where `dissipation_rank` was proposed; the streaming kernel it has to agree with is step 3's. `src/evolution.jl` was added in step 3, the SIMD width became its sixth `Val` on 2026-10-05, and step 5 split the streaming body out of the kernel into `gh_rhs_at_point` and added `gh_paste_kernel!`. `src/lanes.jl` was added 2026-10-05. `src/gauge.jl` was added in step 3, its `γ0` profiles in step 5 and `HorizonDissipation` in step 8c. `src/boundaries.jl`'s parities were added 2026-10-02 and its rotations 2026-10-04. `src/bounds.jl` was added in step 8b. `src/interior.jl` was added in step 5, `hole_mass` in step 8c′, and step 8d added the tracked geometry's kernel side; putting the `:pasted` limiter in `evolution.jl` was an amendment of step 5. In `src/initialdata.jl`, `GHCase` and the case constructors were placed there rather than in `driver.jl` in step 3, `refined = true` was added in step 4 and `hole_forest` in step 5. `src/constraints.jl` was added in step 4; step 5's interior added a method to `is_evolved`. `src/horizon.jl` was added in step 7: `locate_block` and `interpolate`/`interpolate_grad` (the stopgap of [Upstream prerequisites](CODE.md#upstream-prerequisites), item 1, with the footprint guard that refuses a query reaching inside `r_1`; **amended 2026-09-26**: `gh_interpolate`/`gh_interpolate_grad` over TreeAMR's `interpolate`, and the guard as the `Region` `UnevolvedRegion`). `src/tracking.jl` was added in step 8d, with step 8a's leakage e-folds. `src/fit.jl` was added in step 8e, its kernel half in 8e-ii. `src/checkpoint.jl` was added 2026-10-01. In `src/driver.jl`, `default_relaxation_rate` was added in step 8c′ and `discrete_gradient_momentum!` in step 5.

In `test/`, `prerequisite_tests.jl` was added in step 0. `evolution_cases.jl` was added in step 3, and lives in `test/` because `driver.jl` is step 5's. The pointwise tests were split into two files in step 1, and the interface-order table's own file was proposed in step 4. Step 5 added `interior_tests.jl`, `driver_tests.jl`, the hole fixture and the standalone `test/hole_runs.jl` (proposed in step 5). Step 8b added `bounds_tests.jl` and the `bounds` section; step 8c the E3 target and the `calibration` section; step 8d `tracking_tests.jl` and the shell norms' `shell_mask`; step 8e `fit_tests.jl`, and for 8e-ii the `:fitted` claims, the move of `fitted_fixture` into `evolution_cases.jl` and the `fitted` section.

## Measured results

### Where the measurements came from

The paragraphs of `CODE.md`'s Measured results labelled G0 to G4c were measured in the steps of those milestones: G0 in step 0, G1a in step 1, G1b in step 2, G2 in step 3, G3 in step 4, G4a in step 5, G4b in step 6 and G4c in step 7 (the harmonic chart's finer mesh among them). Until 2026-09-23 the layer's default rate was the grid rate `1/dt`, and the adaptive run against the frozen hierarchy and the horizon of a `10 M` run were measured at it; the regrid that moves the mesh was measured again at the default `4/M` in step 8c′. The port to TreeAMR's interpolation was measured on 2026-09-26, and the provider's 16 variables were added the same day, in review. The GC-thread table was measured on 2026-09-19, and the flat-space robust-stability run on the octant on 2026-10-02.

### Single black holes: recommended settings

`CODE.md`'s table of recommended settings for a single hole was added on 2026-10-08, from the octant runs. It gained the `a = 0` hole moving at `v = 1/5` on the quadrant on 2026-10-20, the `a = 1/2` hole moving along its spin on the rotating quadrant on 2026-10-21, the aligned rows (*along `z`*) on 2026-10-22 and the rows *across* the spin, on the bitant, on 2026-10-25. The moving `:damped` rows carry the target's rate from 2026-10-21 (`core_follows` was built on 2026-10-20), and the aligned `a = 0` row's core relaxes at `max(ρ_max, 12 |v|/r_0)` from 2026-10-22.

The rules below the table changed as follows. The ring rule was amended on 2026-10-16: necessary, not sufficient. The moving-hole item was added on 2026-10-20 (measured at `a = 0`, `v = 1/5`), and its `:damped` rule was amended on 2026-10-21 for the target's rate. "Along its spin" was added on 2026-10-21; "Choose `r_1` for the fastest speed the run will reach" on 2026-10-22, from the aligned campaign, and amended for motion across the spin on 2026-10-25, when "Across its spin" (the bitant) was added too. The production cadence dates from 2026-10-22. The rule on thin outer levels was measured on 2026-10-19, and the rule on the finest cube found on 2026-10-14.

What a single hole cost on one H200 before the aligned campaign (whose costs were measured on 2026-10-22 and stay in `CODE.md`), with the pre-rewrite kernel: `26 s/M` at
  `a = 0`, `1/16` (9.4 M points); `6.5 min/M` at `3/5`, `1/32` (60.8 M);
  `5 min/M` at `9/10`, `1/48` (31.9 M); `15 min/M` at `9/10`, `1/96`
  (52.8 M). On a hole, `main`'s rewritten kernel is `1.6–2.2×` faster per
  point, not the gauge wave's `8×` (measured 2026-10-14): `220 s/M` at `1/48`
  (38.0 M points on `L = 128`) and `460 s/M` at `1/96` (58.9 M). The layer's
  kernel and the refined mesh's prolongation are what is left. The moving
  `a = 0` hole on the quadrant's capsule (measured 2026-10-20, `56–112` steps
  per `M`): `34 s/M` at `1/16` (`3.51 M` points, `21 GB`), `125 s/M` at `1/24`
  (`11.8 M`, `33 GB`), `234–250 s/M` at `1/32` (`28.0 M`, `54 GB`; `:fitted`
  `64 GB`) — `40 M` in `0.4`, `1.4` and `2.6–2.8 h`, and `75–175 ns` per point
  and step, `2–4×` the `a = 9/10` family's. The moving `a = 1/2` hole on the
  rotating quadrant's capsule (measured 2026-10-21, `84–140` steps per `M`):
  `130 s/M` at `1/24` (`11.8 M` points, `33 GB`), `254 s/M` at `1/32`
  (`28.0 M`, `54 GB`; `:fitted` `243 s/M`, `64 GB`), `543 s/M` at `1/40`
  (`54.8 M`, `89 GB`; `:fitted` `525 s/M`, `107 GB`) — `40 M` in `1.45`,
  `2.7–2.8` and `5.8–6.0 h`, `71–131 ns` per point and step. Every number so far in this
  item ran the analysis every `M/2`, the moving ones with the
  trailing/leading split, and most of the moving runs' `71–175 ns` was that
  analysis.

### The analysis cadence on an H200

The benchmark was made on 2026-10-22, after moving-hole runs on one H200 on 2026-10-20 and 2026-10-21 had cost `71–175 ns` per point and step; the rule for a production run (`chunk = M`, `split=0`) dates from then. On 2026-10-23 the analysis row became one fold (`mesh_fold`), and the section was amended with the prediction that follows the table in `CODE.md`. One of the benchmark's findings:

- **TreeAMR 0.2.4's nested-mesh ghost fill** is `b1` against the same row on
  0.2.3 (`254 s/M`, 2026-10-21): about `6 %`.

### The right-hand side on an H200

The study was made on 2026-10-05 as an analysis, changing nothing in `src/`, in a scope Erik set, and items 1–3 and half of item 5 of its order of work were implemented the same day. What follows is the study's account of the kernel before it, and of the prototypes (the rounds of `bench/rhs_lab.jl`) that the implementation did not keep.

- **Correctness:** every prototype was compared with `gh_rhs!` on the same filled
  working array. `h` agrees bit for bit without inlining and to `2·10⁻³²` relative
  with it. `Π` agrees to `7.1·10⁻¹³` of `max |∂ₜΠ|`. That is also what the
  package's *own* kernel shows when it is merely compiled with inlining forced:
  contraction differences of one body at two call sites ("Two spellings of one
  expression are not bit-identical"), not a different scheme.

**Where the time goes.** The kernel is nine tenths of an evaluation, and the mesh
pattern around it is small today:

| per owned point, ns | 512 × `16³` | 512 × `32³` | 8 × `128³` |
|---|---|---|---|
| `gh_rhs!` | 9.94 | 9.21 | 9.31 |
| — the kernel (`map_blocks!`) | 8.55 | 8.48 | 8.90 |
| — `scatter!` | 0.36 | 0.34 | 0.33 |
| — `fill_ghosts!` | **1.03** | 0.40 | 0.08 |
| RK4 step (IMEXRungeKutta, broadcast) | 40.4 | 37.4 | 37.8 |

A step is 4.06 right-hand sides, and the stage arithmetic is 0.54 ns a point a step.
An evaluation launches 28 kernels (26 transfer groups, the scatter, the kernel), each
4 µs of host time, and the GPU is busy 97 % of it: launch overhead is not a cost here.
The workgroup shape moves the kernel by less than 10 % (`(32, 8, 1)` and `(32, 4, 1)`
best, `(8, 8, 2)` worst).

Before 2026-10-05 the kernel was not inlined on a device. Julia's inlining heuristics kept as real functions:

- the `ntuple(Val(n)) do … end` blocks of `gh_rhs_at_point`;
- `mixed_stencil`'s closures;
- the StaticArrays generator closures of `gh_node_source` and `metric_derivatives`.

The SASS:

| `gpu_gh_rhs_kernel_` | as compiled | with `always_inline = true` |
|---|---|---|
| instructions (callees included) | 11 224 | 10 992, straight line |
| call sites | 162, about 140 into Julia closures | 0 to Julia code |
| registers, local memory per thread | 255, 8.1 KB | 255, 3.8 KB |
| spill stores / loads (`STL` / `LDL`) | 2417 / 2020 | 914 / 990 |
| FP64 instructions (`DFMA` + `DMUL` + `DADD`) | — | 3867 (2656 + 930 + 281) |
| global loads (`LDG`) | — | 871 |
| kernel | 8.48 ns/pt | 5.0–5.4 ns/pt |

At 255 registers a 256-thread block fills an SM, so there are eight warps. The local
memory per SM is 1–2 MB, beyond L1 (256 KB) and beyond an SM's share of L2 (50 MB
for 132). So the spills go to HBM — inferred, not counted: no `ncu`.

In the inlined build the ~1900 spill instructions move about 15 KB a point. At HBM
bandwidth that alone is 70 ms of the kernel's 72. The FP64 pipes are about 5 %
busy. So neither the arithmetic nor the traffic of [Precision, threads,
devices](CODE.md#precision-threads-devices)' budget is what the kernel costs, and **the
stencils are not the problem.** Ablations of the inlined kernel (raw CUDA, static
strides):

| | ns/pt |
|---|---|
| state, `∂_i h`, the dissipation of `h`, the coefficients and `∂ₜh` only | 0.32 |
| everything but the source | 1.20 |
| everything but the Π stencils | 2.54 |
| everything | 4.30 |

On the order of the fused kernel once the source was lean: The package's kernel
with only `gh_node_source` replaced runs at 7.56 ns/pt inlined and 49 ns/pt without.
`CODE.md`'s streaming order puts the source last, as step 3.

`CODE.md` keeps the verdict: not worth doing are shared-memory staging of the principal part, `maxregs`, one component per thread and further workgroup tuning.

**The design's other ideas, measured.**

| idea (where it was proposed) | measured | verdict |
|---|---|---|
| Stencil/algebra split ("One right-hand-side evaluation", black-box attempt) | package source: 3.2 + 0.29 against 4.3 fused (inlined); lean source: 1.14 against 1.21 | 6 % once the source fits; the Π principal part alone is a spill-free 0.29 ns/pt kernel at ~80 % of L1 load bandwidth |
| Workgroup shape (black-box attempt; GHAccel's 10 %) | within 10 % for a spilling kernel; static against dynamic is ~20 % for the split | set a static `(32, 4, 1)`; no further tuning |
| `Float32` (black-box attempt) | not measured: `Float64` is the requirement (1.4–1.6× in September) | — |
| Shared-memory staging of one variable's ghosted block ([Possible extensions](CODE.md#possible-extensions)) | Π principal part, one component's `h` and `Π` tiles at a time: 0.95–1.59 ns/pt against 0.28 through L1 | **no**: L1 already holds the reuse |
| Fusing the scatter and the stage update into the kernel (GHSO2's "Tier 5") | scatter plus stage arithmetic ≈ 0.48 ns/pt per right-hand side | second order now; ~10–15 % of a step after the kernel (below) |
| Generated code for the source's register schedule | the hand-written lean source, 11× the source and 7× the kernel | **yes**, and no generator is needed |
| Kernel or scatter-and-ghost pattern first ([Possible extensions](CODE.md#possible-extensions)) | the kernel is 92 % of an evaluation | the kernel, then the pattern |
| `Int32`, arrays, `div` (GHAccel) | static against dynamic strides: no difference while spilling, 4 % in the lean kernel | minor |
| Skipping the stencils' zero weights (new; IEEE forbids dropping `0·x`, so a fifth of the first derivative's loads and over a third of the mixed derivative's are zeros) | Π principal part 0.283 against 0.309 ns/pt; nothing elsewhere | minor |
| `maxregs` (new) | always slower: the fused kernel 5.95 at 128 and 5.72 at 168 against 4.30 | no |
| One Π component per thread (new) | 0.31–0.43 against 0.28–0.31 ns/pt | no |

The algebraic Kerr-Schild source has `gh_node_source`'s structure — an `SMatrix`
generator over closures. It adds 0.47 ns/pt to the fused lean kernel and costs 0.54
ns/pt in the source's own kernel against 0.18 without it, so it wants the same
treatment. So does `metric_derivatives`: the stencils-and-coefficients kernel still
spills 1.26 KB.

**The projection (predicted)**, per owned point at `N = 32`, from the measured
pieces:

| ns/pt | today | kernel fixed | and TreeAMR's copies at bandwidth | and a ghosted state with a fused stepper |
|---|---|---|---|---|
| kernel | 8.48 | 1.21 | 1.21 | ~1.3 with the epilogue |
| scatter | 0.34 | 0.34 | 0.14 | 0 |
| ghost fill | 0.40 | 0.40 | ~0.1 | ~0.1 |
| right-hand side | 9.21 | 1.95 | 1.45 | ~1.4 |
| RK4 step | 37.4 | ~8.3 | ~6.3 | ~5.6 |

That is 4.5× to 7× per step. At `128³` the first column's change alone gives about
8 ns/pt.

**The order of the work (proposed 2026-10-05)**, items 1–3, all done the same day:

1. **The lean, closure-free source in `pointwise.jl`.** It is a third spelling,
   tested to roundoff against `gh_node_source` on every background of
   `pointwise_backgrounds.jl` and against the forward-mode pass. It helps the CPU
   too. **(Done 2026-10-05:** `gh_node_source_lean`.**)**
2. **The kernel in the order above**, with `Val(N)` for static strides and linear
   `@inbounds` stores. This amends "One right-hand-side evaluation": the source
   moves from step 3 to step 2, and the per-component streaming stays. **(Done
   2026-10-05, without `Val(N)`:** the reason is in `CODE.md`, "The right-hand side on an H200".**)**
3. **Forced inlining and a static workgroup** on the device, through TreeAMR keeping
   the backend a field set was built with, or the kernel's remaining closures
   rewritten.
   - Steps 1–3 together are measured: **8.5 → 1.2 ns/pt**.
   - Steps 1–2 alone are 3.1 ns/pt.

   **(Done 2026-10-05 by the second route:** nothing in the kernel is a closure
   any more, and forcing inlining changes nothing — below.**)**

### The right-hand side on a CPU

The study was made on 2026-10-05 as an analysis, changing nothing in `src/`, in a scope Erik set, and implemented the same day. Its list "What SIMD in `src/` would take (proposed; done 2026-10-05 as below, with an
overlapping last group instead of a scalar tail, and on every branch)" planned, where `CODE.md` now describes what was built:

- item 1: The `γ0 != 0` branch would be spelled through a helper that a `Vec` answers with
   `any`.
- item 2: the launch over `(N/W, N, N, nblocks)` on the CPU; `N` must be a multiple of
   `W`, or the row needs a scalar tail. The overlapping last group was Erik's proposal in place of a scalar tail or masks.
- item 3 came out as predicted.
- item 4: **The interior and the gauge sources are not measured.** A layer branches per
   point; the natural split is the vector path where all `W` lanes are outside the
   layer, which is most of the mesh, and the scalar one elsewhere. The sampled gauge
   source reads `Hsrc` along the first axis as the state is read; the algebraic
   Kerr-Schild source would have to accept a `Vec`.

The prototype's claim that the lanes are bit for bit the scalar kernel was amended in the implementation (2026-10-05): it holds for the gauge wave, and on a hole the claim is roundoff. `default_simd_width`'s 64 bytes was a prediction until the Skylake-AVX512 row of the implementation's measurements.

### The run scripts

`test/hole_runs.jl` and the "Measured results" its numbers are recorded
under were added in step 5. Step 6 added the indicator's calibration and its
adaptive run; step 7 the `horizon` section; step 8a `leakage`; step 8b
`bounds`; step 8c `calibration`, and amended the options so that a
`key=value` option whose key is a section's name selects that section's
subset; step 8d `tracked`; step 8e `fitted`; step 8f `generic`; step 8
`moving`; step 8′ the `moving` groups `l0`, `l7` and `g5t`. The interior's
numbers moved to `SINGULARITY_HANDLING.md` on 2026-10-08.

Step 8f ran the `generic` groups on Symmetry through a scratch copy of
`.claude/orchestration/symmetry-run.sh`. A remote directory of its own for
every long Symmetry study: rsyncing into a running study's directory was
the likely cause of step 8's `SIGBUS`es, which did not recur from
`step-8-sigbus` and `step-8-t`.

`bench/rhs_lab.jl` (the GPU right-hand-side prototypes) and
`bench/rhs_cpu_lab.jl` (the CPU right-hand-side lab) were added on
2026-10-05.

`test/octant_runs.jl` was added on 2026-10-02, and `octant=reflecting|rotating`
with the spin `a=` on 2026-10-04. From 2026-10-20 it runs a hole moving
along `x` on the quadrant (and `test/boosted_study.jl` reads such runs), from
2026-10-21 `octant=rotating_quadrant`. From 2026-10-21 the quadrant's
production row is the default `4/M` (the moving layer's target rate), and
`rho=8` is no longer needed. From 2026-10-22 a production run takes
`chunk=1 split=0`.

## Binary black holes

### The design proposed on 2026-10-25

The binary black hole design in `CODE.md` ("Binary black holes") was
proposed on 2026-10-25, for Erik's review, with `PLAN.md`'s steps B1–B9
and milestones G7 and G8. Its "What the single holes hand over" dated
what it built on: the moving `:damped` layer with its target's rate and a
core that follows (2026-10-20/21); the aligned campaign to `v = 1/2` and
`a ≤ 9/10` (2026-10-22); the bitant campaign, which moved holes across
their spin (2026-10-24/25, `5f391ee` and `6558e06`, not yet on `main` when
the design was written); the tracked geometry, the finder along a
trajectory, coasting and the jump refusal (steps 8d and 8). It read: "(The
bitant campaign's branch makes `horizon_min_radius` exact across the spin,
which a binary's spinning holes need.)" The calibration rule a smooth
layer at `4/M` with a ramp of at least `4G` cells follows is step 8c's.

### The comoving grid frame proposed on 2026-11-09

On 2026-11-09 Erik asked for a mechanism for the singularity that is likely
to work for orbiting holes, calling progress slow and the current best
method, blending to a known solution, delicate, and noting that BSSN makes
this much easier. Erik fixed three goals — the generalized harmonic system, a
Cartesian grid (not a SpEC-like tracking multi-block system) and a
finite-differencing-like discretization — and left everything else free,
coordinates and gauge conditions included. The design in `CODE.md` ("A
comoving grid frame for orbiting holes") and `PLAN.md`'s steps C1–C5 were
proposed the same day, from the documents alone: nothing was run.

**Decided by Erik on 2026-11-09:** a rigidly rotating Cartesian grid is
within the Cartesian-grid goal, and the design is written up as a proposal.
Its other choices are open, under `CODE.md`'s "What the frame asks Erik to
decide".

**Corrected the same day, on Erik's question.** The first write-up called
the damped harmonic source "a covariant function" and treated the comoving
shift as a fallback. Erik pointed out that the gauge conditions are not
covariant. The section now says what the closed form is invariant under (a
rotation at one instant), that the gauge stays the inertial chart's while
the grid moves, and that which shift is damped — the inertial one, or the
comoving one near each hole — is open, with step C2 running both.

**Where its evidence stood that day.** None of it was on `main`. Steps
B1–B3 (the damped harmonic gauge on one hole at rest and moving, the
track-fed source's loop, the controlled source line) were on
[eschnett/TreeGeneralizedHarmonic.jl#18](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/18),
open. Step I1, the layer-free interior, was the closed draft
[eschnett/TreeGeneralizedHarmonic.jl#19](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/19),
shelved by Erik on 2026-10-30. Step I2, the held interior, was the
local branch `claude/step-I2-held-interior`. The excision steps X1–X8 were
the closed draft
[eschnett/TreeGeneralizedHarmonic.jl#7](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/7),
shelved on 2026-10-19, with principled
excision restarted as a separate project, TreeExcision. A comoving or
rotating grid had not been considered in any of the documents before.

**Considered and not proposed:**

- **A puncture-like interior.** The system evolves `g_ab`, which has no
  finite state on a trumpet slice, and step I1 measured the regular
  substitute collapsing within `M/4` unless anchored at `10/M`, which is a
  damping layer again.
- **The analytic target on a comoving grid.** The target would rest on the
  grid, but the hole would still have to be held on it to a quarter of a
  cell, and the superposition is still not a solution.
- **Excision as this package's interior.** It is the method with the
  record in generalized harmonic codes, and it fits the design as a
  replacement of `:fitted`. It was left to TreeExcision, where it is being
  rebuilt.
- **A diffusive fill**, the right-hand side switched off deep inside and
  replaced by a heat equation in the fit's variables, so that the interior
  is the evolved state's smooth continuation with no fit. It is local and
  indifferent to the merger's topology, and it is unproven: `:frozen`,
  its limit without the diffusion, ends at `13 M`. Not written into the
  design.
- **The damped harmonic gauge with `:fitted` on the inertial grid at a
  finer spacing**, pairing (β) as step B1 left it: the fallback if a
  rotating grid had been refused, priced there at several GPUs.

### Harmonic gauge for the inspiral proposed on 2026-11-11

The comoving grid frame's proposal was merged on 2026-11-10
([eschnett/TreeGeneralizedHarmonic.jl#28](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/28)).
On 2026-11-11 Erik asked whether the fallback named beside it — the damped
harmonic gauge with `:fitted` on the inertial grid at a finer spacing — was
expected to hold holes that move across the grid. It was not: step B1 had
priced that run and not made it, its moving hole had not settled at `1/24`
or `1/32`, and after the correction of 2026-11-09 the comoving grid with
the inertial shift damped is the same problem in the continuum.

Erik then proposed going back to a black hole in harmonic coordinates,
which simplifies the gauge conditions, at the price of more resolution — a
price that was being paid anyway. The design in `CODE.md` ("Harmonic gauge
for the inspiral") and `PLAN.md`'s steps H1–H5 were written the same day on
Erik's request, from the documents alone: nothing was run.

**What the proposal adds to Erik's suggestion:** the damped harmonic gauge
rolled on from zero before the merger, where pure harmonic gauge is
reported to fail; and that the route makes the comoving grid's open
question, which shift the gauge damps, without an object.

**Where its evidence stood that day.** The package's harmonic runs were
those of steps 8f, 8 and 8′, of late September 2026: on CPUs, to `13 M`
at most, the moving ones at `q = 2`. The binary design of 2026-10-25 had
considered two gauges, the tracked Kerr-Schild source and the damped
harmonic gauge, and not `H ≡ 0`. Steps B1–B3 were still on
[eschnett/TreeGeneralizedHarmonic.jl#18](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/18),
open.

**Considered and not proposed:**

- **The roll-on at `t_0 = 0`** on harmonic data, close to how SpEC evolves
  superposed harmonic data: it keeps the gauge's rest frame, and with it
  step B1's moving hole, for the whole inspiral. It is the alternative the
  section lists.
- **A chart that is harmonic in time only** (Ma, Giesler, Scheel and Varma
  2021), which reaches spins of `0.8–0.9`: it has a spatial gauge source,
  and so a line again.
- **A damped wave equation for `H_t` alone**, Pretorius's first gauge: it
  evolves the source as a field, which this system's algebraic sources
  avoid.

**Corrected the same day.** The first estimate of the cost on a fixed
capsule, eight times a Kerr-Schild run's, counted the new level as covering
the old one's volume. It covers half the radius: the section's
`2 (1 + 2s)` replaces it.

### Step H1 measured on 2026-11-11

Erik asked for step H1 the day the route was proposed, and approved the
Symmetry campaign (ten H200 rows, then the `128 M` extensions). The runs
and their numbers are in `SINGULARITY_HANDLING.md`, "A harmonic hole at rest
(step H1, 2026-11-11)"; what they replaced in `CODE.md`'s "What harmonic
gauge costs":

- **The settings** were predicted by scaling the Kerr-Schild table by the
  horizon's radius: `:damped` from `1/32` and `:fitted` from `1/48`, both
  marked as guesses to screen. Measured: `1/24` and `1/40`, each a step
  coarser than predicted, with the predicted layers.
- **"The finest level is twice as fine"** was the prediction for the price.
  Measured: at the spacing that runs, it is `1.5–1.7×` (`1/24` against
  `1/16`, `1/40` against `1/24`); for the Kerr-Schild hole's accuracy at
  `1/24`, shell against shell
  at the same areal radius, it is `2.3–3.3×`. In a binary's units the
  predicted `1/96` to `1/128` at the holes became `1/48` and `1/80` to run,
  and `1/112–1/156` for the accuracy.
- **"Outside the horizon the spacing buys nothing (predicted)"** was
  refuted: at the same `h` the harmonic hole's error in `[2, 4)` is
  `90–120×` the Kerr-Schild hole's at the same areal radii, `50–65×` relative
  to the fields.
- **The binary's cost** `2 (1 + s)` for one level more is kept as the
  prediction it was; H1 adds that the same accuracy is `1.2–1.7` levels
  more, and `5.2×` on the octant at rest.
- Two of "What is not known on this route" were answered for the harmonic
  hole at rest: it is stationary for `128 M` where the hole is, and `:fitted`
  converges outside the horizon at a margin fixed in `M`.

Decided in the step without asking (each a choice of what to measure, not of
the design): the mesh, `L = 64` with cubes `32, 16, 8, 4` (the Kerr-Schild
study's with one level more); `:fitted`'s margin fixed in `M` as the offset
`M/4` and the ramp `5M/16`, the one series that fits integer cells at
`1/32`, `1/48` and `1/64`; the coarsest rows of each variant as the ones run
to `128 M`; and a Kerr-Schild `1/24` control on the same day's code for the
price's cost.

## Possible extensions

### Extensions since built

These items of `CODE.md`'s list were built, and left it:

- **Checkpoint and restart**: the forest's leaf keys, the time, the
  case and the state vector to HDF5, read back into a fresh forest
  (`Forest`, `refine!` to the keys, `balance!`, `scatter!`); bit-identical
  to an uninterrupted run, since the layer and the hook depend on
  `(x, t)` only. A few dozen lines, or TreeAMR's M9. **(Done 2026-10-01**
  through TreeAMR 0.1.4's M9a, whose load builds the forest from its leaf
  list directly (`Forest(roots; leaves)`) — see [Checkpoint and
  restart](CODE.md#checkpoint-and-restart); and since step 8d the layer depends on
  the track and the fits as well, which the run state carries.**)**
- **A hole tracker** for the *interior layer* when the center is not
  known in advance — the volume-weighted centroid of `α < α_thr`
  through `block_mapreduce`, or the horizon finder itself; the
  refinement already follows the indicator. **(Built in step 8d** from the
  finder; a binary's orbit needs a quadratic prediction and a center-jump
  check, proposed 2026-10-25 under [The tracks](CODE.md#the-tracks).**)**
- **Smoothing the interior** for a spacetime without an analytic
  solution — the interior layer relaxing toward something other than
  the exact solution. **(Became PLAN.md step 8e on 2026-09-23**: the
  fitted target; see [the interior's
  questions](SINGULARITY_HANDLING.md#the-interiors-questions-opened-in-step-5-and-closed-through-step-8).**)**

### Dates of the remaining extensions' notes

- Excision was priced on 2026-09-23, in `PLAN.md` step 8g.
- The damped harmonic gauge driver, the binaries and the
  gravitational-wave extraction were designed on 2026-10-25, under
  "Binary black holes".
- The note that TreeAMR's relative hook is the radiative boundary's hook
  was added on 2026-10-25.
- GPU kernel efficiency was investigated on 2026-10-05, and SIMD across
  grid points on the CPU measured and implemented on 2026-10-05.

## Open questions

### Settled in review on 2026-09-16

Settled in review on 2026-09-16: the expanded form (the flux form is not
implemented on the mesh); three dimensions only; no excision, the
interior treated pointwise by profiles of the distance to the center;
the proof-of-concept target a single boosted, spinning hole;
OrdinaryDiffEq's RK4 for time integration (amended 2026-09-26:
IMEXRungeKutta's RK4, by block owner); the analysis quantities as
part of the deliverable; an error indicator for refinement; `Float64`
on the H200 as the device requirement; no checkpointing (reversed
2026-10-01); the inherited documents copied into `notes/`.

### The interior's questions

The interior's questions were opened in step 5 and closed or measured
through step 8′; they moved from `CODE.md`'s "Open questions" to
`SINGULARITY_HANDLING.md` on 2026-10-08.

### The streaming-order fused kernel (answered 2026-10-05)

Open question 3, with G2's half and the H200's answer:

3. **The streaming-order fused kernel** against the stencil/algebra
   split, decided by measured spills and time (G2 on the CPU, G6 on the
   H200); anything beyond the black-box attempts is research, not a
   milestone. *G2's half is in*: the fused kernel runs at 1244 ns per
   owned point at `q = 4` on one CPU thread, of which the pointwise
   algebra is 505 ns and the stencils the rest, and the mesh pattern
   around it adds another 616. A host CPU says nothing about spills, so
   the question stays open for the H200; what G2 adds to it is that the
   split it would be measured against is 3:1 and not 1:1. **(Answered on the
   H200 2026-10-05**, in [The right-hand side on an
   H200](CODE.md#the-right-hand-side-on-an-h200).**)**
   - The split is worth 1.25× with the package's source and 6 % with a
     register-lean one, so the fused kernel stays.
   - The spills come from the source and from closures left uninlined on the
     device, not from the stencils.

### The defaults `m` and `ρ_max` (2026-09-23)

Open question 5's list of defaults gave `ρ_max = 4/M` as
"(`ρ_max · dt = 1` until 2026-09-23)", and told the story of `m` and
`ρ_max` this way:

> Step 8a split
> `m` into a stencil margin and a leakage margin, measured that `m = 8`
> attenuates grid-scale content from `r_1` by `e^{−2.9}` to `e^{−4.6}` on
> the fixture, and left the default where it is until step 8c says what
> amplitude it has to hold back **(proposed in step 8a**; see [The
> interior](CODE.md#the-interior-a-pointwise-damping-layer)**)**. Step 8c kept
> `m = 8` — a smooth layer's wrong target leaves the exterior what the exact
> one leaves — and replaced `ρ_max · dt = 1` by `ρ_max = 4/M` with a ramp of
> `4G` cells, measured for an inexact target and **(proposed in step 8c)**
> for the analytic one, where it cuts the `50 M` error sixfold. **Erik took
> it on 2026-09-23** for every variant, the analytic `:damped` layer
> included, and step 8c′ made `4/M` the code's default; the grid rate is
> the option `ρ_max_factor` **(decided 2026-09-23)**. The ramp is not part
> of that decision: it is the case's `r_0`, `r_1` and `ρ_ramp`, and the
> suite's fixture keeps step 5's layer, a ramp of `4.8` cells, on which
> step 8c measured the exact target at `4/M` to `50 M`.

### Gauge sources that know less about the hole (2026-10-02 to 2026-10-20)

The question was opened on 2026-10-02 in a design discussion with Erik,
when nothing was built: "Today `H_a(x)` is sampled from the background".
The algebraic source was built on 2026-10-02.

**Proposed (2026-10-02):** build the source above as an alternative to the
sampled `Hsrc` behind the kernel's existing gauge-source `Val`, measure (1)
on the static Kerr-Schild hole, and then run a boosted Kerr-Schild hole
across the mesh with the analytic layer, against G5's harmonic `a = 7/10`
rows; the damped harmonic gauge stays the extension for binaries, where no
constants of the run make the data stationary.

**Done (2026-10-20) for `a = 0`**: the boosted Kerr-Schild hole at `v = 1/5`,
the speed of one member of a binary a few orbits before merger, crossed `8 M`
of a frozen capsule mesh on [the quadrant](CODE.md#the-quadrant-a-hole-moving-along-x)
to `40 M` with the analytic layer, at the resting hole's level once the layer
relaxes at `8/M` and its core follows the target — under [the boosted
Kerr-Schild hole at binary
speed](SINGULARITY_HANDLING.md#the-boosted-kerr-schild-hole-at-binary-speed-2026-10-20)
— and, from 2026-10-21, at the default `4/M` once the layer and the core carry
their target's rate ([the moving `:damped` layer's target
rate](SINGULARITY_HANDLING.md#the-moving-damped-layers-target-rate-2026-10-21)).

## Releases and dependencies

On 2026-10-01 the package moved to **TreeAMR 0.1.4**, and two of its
features came into use: `mesh_mapreduce` for every reduction that crosses
blocks (`CODE.md`, "Analysis quantities"; the norms became `mesh_fold`
folds on 2026-10-23) and M9a's checkpoints. On 2026-10-04 it moved to
**TreeAMR 0.1.7** for the rotating seam (M12).

From 2026-10-09 the package is on **TreeAMR 0.2.0**, whose one break is that
the checkpoint functions and TreeAMR's HDF5 extension `TreeAMRHDF5Ext` moved
into the companion package **TreeIOHDF5** (registered, `[compat]` `0.1.0`), with the same names, signatures
and file format. TreeIOHDF5 replaces HDF5 as the hard dependency (`src/` used
HDF5 for nothing else), and `prerequisite_tests.jl` asserts that
`save_checkpoint` and `load_checkpoint` are TreeIOHDF5's and not TreeAMR's.
No internal this package reaches for changed.

From 2026-10-22 the package requires **TreeAMR 0.2.4** (`78e60ae`), whose
nested-mesh ghost fill took a moving-hole row on one H200 from `254` to
`239 s/M`.

- `ApparentHorizonFinder` and `KorzynskiSpin`, the horizon diagnostics,
  are used from step 7.

### TreeAMR

- **2026-10-19.** Threads, GPUs and MPI through the mesh: TreeAMR's M7 is
  in use from 2026-10-19 (`CODE.md`'s opening paragraph said so).

### Dependencies

`IMEXRungeKutta` replaced `OrdinaryDiffEqLowOrderRK` and `SciMLBase` on 2026-09-26. `LinearAlgebra` was added in step 1 and not listed when `PLAN.md` enumerated step 0's `Project.toml` (proposed in step 1). `TreeIOHDF5` is a dependency from 2026-10-09; it replaced `HDF5`, a hard dependency from 2026-10-01 to load TreeAMR's HDF5 extension, when TreeAMR 0.2.0 moved the checkpoint functions there. `ApparentHorizonFinder` and `KorzynskiSpin` were added in step 7.

**`KorzynskiSpin` is pinned over `ssh`, because its GitHub repository is private** — an anonymous `https` clone gets a 404, so `git@github.com:eschnett/KorzynskiSpin.jl.git` is the only URL that resolves it: **(resolved 2026-09-21 — the repository is public. The pin is the plain `https` URL of the others, the deploy key, the `ssh-agent` step and `JULIA_PKG_USE_CLI_GIT` are gone from `CI.yml`, and an anonymous clean checkout and a fork's pull request both resolve it, so the clean-checkout check proves what it claims for the first time.)** `TreeAMR` left `[sources]` the same day: it is registered, and `[compat]` selected **0.1.1**. The `[sources]` entries were three until TreeIOHDF5 joined them on 2026-10-09.

The Julia floor of 1.13 (CI 1.13 only) was decided by Erik on 2026-10-16, because supporting earlier versions slows development and is not useful at the moment. Until then the floor was 1.11, set by `[sources]`, and CI had run 1.11 beside the release until 2026-09-19 (the failure that ended it is under [Measured results](CODE.md#measured-results)).

`AbstractSphericalHarmonics` has been a direct dependency from step 8d; it was already in the manifest through `ApparentHorizonFinder`. `ForwardDiff` in the tests was added in step 1.

The compat bounds followed TreeWave's, including `OrdinaryDiffEqLowOrderRK = "2.2.5"` (proposed in step 0, which is the one bound `PLAN.md` left unstated). The two pins resolved in step 0 to TreeAMR v0.1.0 and SpacetimeMetrics v1.6.0. The two added in step 7 resolved to `ApparentHorizonFinder` v2.1.0 and `KorzynskiSpin` v1.1.0.

### The pins of step 7

Step 7 (G4c): Two further pins are resolved, `ApparentHorizonFinder` v2.1.0 and
`KorzynskiSpin` v1.1.0.

### The clean-checkout check

The clean-checkout check resolved TreeAMR from the General registry from
2026-09-26, at `0.1.3`, from 2026-10-01 at `0.1.4`, from 2026-10-04 at
`0.1.7` and from 2026-10-09 at `0.2.0`; also between 2026-09-21 and
2026-09-23. HDF5 with its binary library came from General too from
2026-10-01; from 2026-10-09 TreeAMR at `0.2.0`, with TreeIOHDF5 from GitHub
and HDF5 through it. From 2026-09-21 it is a real check: every source is
public.

### The `[sources]` pins and TreeAMR's `[compat]`

TreeAMR comes from the registry (amended 2026-09-26, when Erik registered TreeAMR
  0.1.3 for its M11 interpolation; from 2026-09-23 to then TreeAMR was
  pinned too). `TreeIOHDF5` is one of the `[sources]` pins (from 2026-10-09). TreeAMR is General's release under `[compat]` `0.2.6` (from 2026-10-23, for M15's
  folds over points; `0.2.4` from 2026-10-22, for its
  nested-mesh ghost fill; `0.2` from 2026-10-09, when the
  checkpoints moved to TreeIOHDF5; `0.1.7` from 2026-10-04, for M12's
  rotating seam; `0.1.4` from 2026-10-01, for M9a's checkpoints; `0.1.3`
  before).

### TreeIOHDF5 against TreeAMR 0.2.4 (2026-10-22)

`TreeIOHDF5` `main` being ahead of the TreeAMR it admits was found 2026-10-22. The
  first `boosted-ks-7` batch lost ten jobs to it, each at `4 M`. Until TreeIOHDF5 does (see `CODE.md`, "Checkpoint and restart"), a copy that stays on TreeAMR 0.2.4 pins TreeIOHDF5 0.1.3, as
  `boosted-ks-7` did.
  **(Resolved 2026-10-23** for this package: it requires TreeAMR 0.2.6,
  which TreeIOHDF5 `main` (0.1.5) admits.**)**

## The test suite

### With the MPI comparison

After the MPI port (2026-10-19), the suite with the MPI test was 5364 assertions in 16m35 at four threads.

Recorded beside the memory measurement of 2026-11-06: with the job first, as that cell runs it, the suite is 5664 assertions in 15m09 after the job's 230 s, the MPI comparison's 263 among them (load 9–11).

### The suite step by step, steps 0–7

**G0 (step 0).** The suite is 82 assertions in 7.7 s at one thread and
7.6 s at four, on the development machine (Apple silicon, 12 CPU threads,
Julia 1.13.0): 0.7 s of it `precision_tests.jl` and 7.4 s
`prerequisite_tests.jl`, which is the compilation and evaluation of the
two `dmetric` passes and nothing else. Both are inside GHSO2's rule of
thumb of 30 s per test file, and the evaluated metric is the cost to
watch as the suite grows. A clean archive of the tree, with no
`Manifest.toml`,
resolves TreeAMR v0.1.0 and SpacetimeMetrics v1.6.0 from GitHub `main`
through the `[sources]` pins, instantiates and passes with the same 82.
The prerequisite kernel fill is under [Initial data and
backgrounds](CODE.md#initial-data-and-backgrounds). No physics is measured yet:
`src/` holds the module shell, `precision.jl` and `device.jl`.

**G1a (step 1), the pointwise algebra.** The suite is 1007 assertions in
124 s at one thread and 116 s at four, on the development machine
(Apple silicon, 12 CPU threads, Julia 1.13.0), up from step 0's 82 in
7.7 s.

**G1b (step 2), the stencils.** The suite is 1682 assertions in 128 s at
one thread and 122 s at four, on the development machine (Apple silicon,
12 CPU threads, Julia 1.13.0), up from step 1's 1079 in 124 s. Of that,
`stencils_tests.jl` is **603** assertions in **5 s** of testset time — 7 s
standalone, including its own compilation: the step's tests evaluate no
background at all, weights and polynomials only, so the suite's cost is
still step 1's compilation of `SpacetimeMetrics`' nested dual passes and
the stencils did not move it. Three of those five seconds are the two
KernelAbstractions launches, at `Float64` and `Float32`. This is the file
to copy when a later step needs a cheap test.

**G2 (step 3), the right-hand side on a uniform mesh.** The suite is 1877
assertions in **3m48** at one thread and **3m21** at four, on the
development machine (Apple silicon, 12 CPU threads, Julia 1.13.0), up from
step 2's 1682 in 128 s. A clean archive with no `Manifest.toml` resolves
the two pins from GitHub `main`, instantiates and passes with the same
1877. The growth is the physics: `evolution_tests.jl` is **56 s** and
`convergence_tests.jl` **40 s** of it, both over `PLAN.md`'s 30 s rule of
thumb and recorded as such — a row of either costs a kernel specialisation
to compile, and the evolutions themselves are seconds.

**G3 (step 4), coarse-fine faces, the constraints and the threads.** The
suite is **2144 assertions in 8m40** at one thread and **6m55** at
four, on the development machine (Apple silicon, 12 CPU threads, Julia
1.13.0), up from step 3's 1877 in 3m48. A clean archive with no
`Manifest.toml` resolves the two pins from GitHub `main`, instantiates and
passes with the same count. Where the growth went, and why each piece is
what it is:

| file | 1 thread | what it pays for |
|---|---|---|
| `type_tests.jl` | 99 s | the fused kernel, both monitors and the pointwise algebra compiled again at `Float32` (45 s) and `Float32x2` (33 s) |
| `interface_tests.jl` | 93 s | fifteen evolutions on the two-level mesh, where the ghost fill is 79 % of an evaluation |
| `constraints_tests.jl` | 74 s | the ADM kernel's first specialisation (18 s) and its second (4 s), and `ddmetric` on a second background (40 s) |
| `threading_tests.jl` | 25 s | the workload twice — once in process, once in a subprocess at the other thread count |

Three of the four are over `PLAN.md`'s 30 s rule of thumb, as
`pointwise_tests.jl`, `evolution_tests.jl` and `convergence_tests.jl`
already were, which makes six files in the suite over it; each is recorded
here with what it buys. Three of step 4's four are **compilation**, and
the fourth — the interface table — is the only file in the suite whose
cost is arithmetic. Before adding to any of them, price it: a new `q` or a
new element type is a new kernel; a new background is a new dual pass; and
a resolution added to an interface sweep is `N⁴` of ghost filling at 216
coarse points per fine ghost point.

The cheapest thing step 4 could have done and did not is worth recording
too: the thread workload carries the gauge monitor and not the ADM one,
which keeps `threading_tests.jl` at 25 s instead of about 60.

**G4b (step 6), the refinement indicator.** The suite is **2968 assertions
in 13m35 at one thread and 10m45 at four** on the development machine
(Apple silicon, 12 CPU threads, Julia 1.13.0), up from step 5's 2486 in
11m51 and 8m33. Step 6 also ran it on **Symmetry**, one node of the
64-core AMD EPYC `amddebugq` partition at the same Julia, where the same
2968 take **29m32** and **20m29**: a cluster core is about **twice** as
slow as this laptop's, so the cluster buys studies in parallel rather than
wall clock, and the laptop column stays the one comparable with steps 0–5.
That run is also the clean-checkout check by construction — the tree is
copied there without any `Manifest.toml` and the `[sources]` pins resolve
from GitHub before the job starts.

| file | 1 thread | 4 threads | what it pays for |
|---|---|---|---|
| `driver_tests.jl` | 2m19 | 56 s | step 5's five static-hole runs, unchanged |
| `refinement_tests.jl` | **26 s** | **16 s** | the algebra and the marks (about 1 s), the initial-data cycle (1.2 s), and the two adaptive runs (10.3 s and 8.4 s) |

**G4c (step 7), the horizon.** The suite is **3444 assertions in 12m31**
at one thread and **8m38** at four on the development
machine (Apple silicon, 12 CPU threads, Julia 1.13.0), up from step 6's
2968. `test/horizon_tests.jl` is **28.7 s / 12.8 s** of that (one thread /
four) and 51 s run on its own — the difference is the right-hand-side
kernel and the hole fixture, which `driver_tests.jl` has already compiled
by the time it runs. Its two evolutions are most of it; the interpolator's
own claims, which evaluate no background at all, are under two seconds.
(The same runs put `driver_tests.jl` at `1m59 / 44.8 s` and
`refinement_tests.jl` at `22.1 / 12.8 s`; step 6 recorded `2m19` and `26 s`
for those at four threads, so the machine or the depot is faster than it
was, and the step-6 numbers should be read as that step's and not as a
regression here.)

### The first green CI build (2026-09-21)

**That acceptance item is no longer blocked** (2026-09-21), and is
recorded under [File layout](CODE.md#file-layout): `KorzynskiSpin`'s repository
is public, so every dependency resolves anonymously. The clean-checkout
check — a `git archive` of the tree with no `Manifest.toml`, instantiated
and tested — resolves `TreeAMR` from the registry at `0.1.1` and the other
three from GitHub, and passes here, **3444 assertions in 12m26**. It now
means what it says in an anonymous clone and in a fork's pull request as
well.

That first green build also measured the suite in CI for the first time —
no job had reached `julia-runtest` before, every run having died in
`julia-buildpkg` at the clone — and it found **four pre-existing failures
that the clone failure had been masking**. They are not regressions and
are unfixed as of this writing:

- On the **floor version, 1.11**, on both operating systems, the
  zero-allocation claims fail: `@allocated` is **176** bytes for all four
  of `gh_node_rhs_expanded`, `gh_node_rhs`, `metric_derivatives` and
  `adm_vars_from_state` at both `Float64` and `Float32`
  (`pointwise_tests.jl:480`), and **48** bytes for the weights
  (`stencils_tests.jl:347`). Both are zero on 1.13. So the claim holds on
  the version the suite usually runs and not on the version `[sources]`
  makes the floor; one of the two is wrong. **(Moot from 2026-10-16:
  the floor is 1.13 and nothing runs 1.11 any more; see
  [File layout](CODE.md#file-layout).)**

  **It does not reproduce in isolation.** On 1.11.9 here, `@allocated` of
  `derivative_weights` and `dissipation_weights` is **0** — measured
  through the tests' own loop-over-closures spelling *and* through a named
  function taking the arguments, at `--check-bounds=yes` as
  `julia-actions/julia-runtest` runs it as well as at `auto`, and under
  `--code-coverage=user`, which every cell was getting by default and
  which was the most promising guess of the three — coverage instrumentation is a standard way to break an `@allocated`
  claim. It is not this
  one: the measurement is 0 with instrumentation demonstrably active,
  `.cov` files being written for `stencils.jl` and for 47 of StaticArrays'
  files and 22 of TreeAMR's in the same process. So neither the spelling,
  the bounds-checking flag nor coverage is the mechanism, and whatever
  is left needs the full suite's context — `runtests.jl` loads this
  package before `MultiFloats`, and `derivative_weights` is `@generated`,
  which is the one interaction this package already knows is order-
  dependent. TreeHydro is the closest comparison and is **green**: its
  `riemann_tests.jl` asserts the same `@allocated(…) == 0` of
  `face_states` and `riemann_flux`, its matrix pins `1.11` explicitly, and
  it differs in two ways worth trying here — the measurement goes through
  a top-level named function that takes the arguments, and the functions
  return plain `NTuple{M,T}` rather than `SVector`/`SMatrix`.
- On **Linux at 1.13 under code coverage**, four exact-equality
  assertions fail: `first_r.err_l2 == 0`, `err_linf == 0` and
  `residual == 0` evaluate to `3.2e−17`, `1.4e−15` and `1.2e−13`
  (`driver_tests.jl:93`–`95`), and `inside_exact` — a `===` comparison of
  the `:pasted` state against the exact reference — fails
  (`interior_tests.jl:552`). These are this document's own rule about
  bit-identity across call sites, met in the tests rather than in the
  code: the initial data and the error reference both reach the analytic
  solution through `case_state_tuple`, by different call sites, so exact
  equality was never an invariant. **Fixed** by comparing to roundoff —
  `100 eps(T)`, and `10⁴ eps(T)` for the residual, which is the only one of
  the three read inside `r_1` where the solution is steepest.

  **It is the machine, not the flags** (measured 2026-09-18 on three).
  The first reading — that code coverage flips it — was wrong, and is
  recorded here because the CI evidence for it looked clean: the same
  Linux cell failed instrumented and passed uninstrumented. Symmetry
  settles it. On an EPYC 7543 the three values are

      err_l2 = 3.241779465395761e-17, err_linf = 1.3653937842860002e-15,
      residual = 1.2253080414080616e-13

  **with coverage and without it alike, and bit-identical to what CI
  reports**. So three machines give three behaviours: aarch64 returns an
  exact `0.0` however it is compiled (no combination of `--code-coverage`
  scope and `--check-bounds=yes` perturbs it there), GitHub's Linux runner
  returns `0.0` uninstrumented and these values instrumented, and EPYC
  returns these values always. The quantity is not reproducible across
  microarchitectures, which is the same lesson TreeHydro records for
  Base's `@simd` reductions: bit-identity across *thread counts* is an
  invariant and is asserted; bit-identity across *microarchitectures* was
  never on offer. That the instrumented Intel runner and the EPYC agree to
  the last bit says the alternative is a *determinate* second value — one
  fused multiply-add taken or not — and not noise.

  The fix is verified where the failure lives: on Symmetry at 1.13 the
  patched `driver_tests.jl` passes 243 of 243 and `interior_tests.jl` with
  it, against values 2 to 4 orders of magnitude inside the new bounds.

  *Note for `Pkg.test`*: its coverage scope is `@<pkgdir>`, not the `user`
  one might assume. It also passes `--check-bounds=yes` — **but only up to
  Julia 1.12**; 1.13's `Pkg` dropped it from `gen_subprocess_flags`
  (checked in all three). So the 1.11 cells test with bounds checking
  forced on and the 1.13 cells do not, which is part of why 1.11 is slower
  and is a difference in the harness rather than in the compiler.

### What the suite costs, and where (measured 2026-09-19 on Symmetry)

What is still current of this profile — how threads and coverage separate compilation from arithmetic, and the GC threads — is under [What the suite costs, and where](CODE.md#what-the-suite-costs-and-where); what the Core.Box fix did to `out_identical` is under [Measured results](CODE.md#measured-results), G2.

Measured on one `amddebugq` node, Julia 1.13.0, `Float64`, no coverage —
the configuration the suite is normally run in. **3444 assertions pass in
`31m12` at one thread and `16m41` at four**, a speed-up of **1.87×**. Read
these against the development machine's `12m31 / 8m38`: a Symmetry core is
about 2.5× slower, so the cluster buys throughput and not wall clock, as
step 6 found.

**97 % of the wall clock is inside testsets** — 0.8 min of 31.2 sits
outside them, so there is nothing to win in loading or in the harness. The
distribution is a long tail with one fat head: the top 5 testsets are
35 % of the run, the top 10 are 52 %, the top 20 are 74 %, over 124
top-level testsets.

| testset | 1 thread | 4 threads | speed-up |
|---|---|---|---|
| The driver and the static hole | 5.50 | 1.68 | 3.28× |
| The dissipation does not change what the interface costs | 1.41 | 0.33 | 4.22× |
| A ghost filled at order `p` … an order | 1.40 | 0.34 | 4.18× |
| The horizon | 1.30 | 0.45 | 2.90× |
| The pointwise algebra … `T=Float32x2` | 1.28 | 0.88 | 1.46× |
| A run computes in the type it is given: `T=Float32` | 1.13 | 0.77 | 1.46× |
| The ADM constraints vanish on an exact vacuum solution | 1.08 | 0.77 | 1.40× |

(minutes). `driver_tests.jl` alone is **17.6 %** of the serial run, and
inside it the convergence sweep *the masked error converges at order `q` on
the frozen hierarchy* is **2.22 min — 7 % of the whole suite**.

**The physics tests are per-step bound, not overhead bound** — which was
worth measuring, because the opposite was the plausible guess: the ADM
monitor is a hundred second derivatives and runs once per chunk, and a
short run has few steps per chunk. It is not what they cost. Running each
resolution of the sweep to `t_end` and to `2 t_end` and taking the slope
and the intercept:

| `N` | points | per step | fixed per run | fixed share |
|---|---|---|---|---|
| 6 | 25 920 | 1.06 s | 1.43 s | 10.9 % |
| 8 | 61 440 | 2.01 s | 1.32 s | 4.5 % |
| 10 | 120 000 | 3.40 s | 1.41 s | 2.4 % |

Setup, the per-chunk analysis and the horizon find together are **1.3–1.4 s
per run whatever the resolution**, and the rest is the integrator. Per
point per RK4 stage that is **11.5, 8.6 and 7.2 µs** at `N = 6, 8, 10` —
falling with `N` because the per-block costs amortise — and the sweep's
total cost grows as `N^2.95`, not `N⁴`, for the same reason. The constant
is the right-hand side's own, and [Possible
extensions](CODE.md#possible-extensions) already books kernel efficiency as a
research project rather than a milestone; until that is taken up, a
physics test costs what its points and its steps cost.

So the levers, in order of what they return and with what they cost:

1. **Four threads.** Already 1.87× on the whole suite and 3.28× on
   `driver_tests.jl`, for nothing. CI runs one of its two pull-request
   cells serial, and two of its three cells on `main`.
2. **The sweep's resolutions.** `Ns = (6, 8, 10)` costs `13.1 + 29.5 +
   59.3 s`, and `N = 10` alone is 58 % of it. At `N^2.95`, `(6, 7, 8)`
   costs 62 % of `(6, 8, 10)` and saves about **38 s**; `N ≥ 2G + 2 = 6` is
   TreeAMR's vertex invariant, so 6 is the floor and there is nothing
   below it. The price is lever arm: `h` spans 1.33× instead of 1.67×, so
   the fitted rate is noisier and the slack in `rate_l2 > q − 1/4` has to
   absorb it.
3. **`t_end`.** 97 % of a run is per-step, so halving `t_end = 3/20`
   nearly halves the sweep — but at 11–17 steps it is already short, and
   the error has to stay clear of roundoff for a rate to mean anything.
4. **Fewer specialisations**, for the 13 % that threads do not touch.
   `T=Float32x2` in the `pointwise` rows is the single largest at 0.88 min
   even at four threads.

None of 2–4 drops a physics claim; they change how finely it is measured.

**Where the time actually goes** (profiled 2026-09-19, one thread on
Symmetry: the whole suite under `@time`, and one `N = 8`, `q = 2` hole run
sampled at 1 ms and by allocation). Three findings, and none of them is
the physics.

**Compilation is 52.4 % of the suite**: `1877 s, 7.04 G allocations:
242.670 GiB, 6.00 % gc time, 52.43 % compilation time`. That bounds every
other optimisation — making the numerics twice as fast buys 24 % of the
suite, not 50 % — and it is the same statement as the threading and
coverage split above, arrived at independently.

**About a fifth of the run is ghost-exchange bookkeeping, and it is
upstream.** Of 28 889 samples, the hottest leaves are `rem` (11.5 %), `==`
(9.6 %) and `div` (1.3 %), and resolving their callers puts almost all of
them in one place: **3263 of 3310 `rem` samples and 358 of 365 `div`
samples are `TreeAMR/src/ghosts.jl:27`**,

    off = ntuple(d -> (r ÷ boxstride[d]) % boxlen[d], Val(D))

— an integer division and a remainder per dimension, per ghost point, per
variable, in `cpu_transfer_kernel!`. A further 1930 `==` samples are
`CartesianIndices` iteration in the same kernel's tensor-product stencil
loop. Actual floating-point work — `+`, `*`, `muladd`, `-` — is about
22 %. So the transfer kernel's *addressing* costs nearly as much as the
arithmetic of the whole run, and by this package's rules that is TreeAMR's
to fix, not ours.

**One hole run allocates 3.66 GiB**, which it should not:

| est. per run | what | where |
|---|---|---|
| 2390 MiB | boxed `Float64` | `ntuple.jl:68` (caller not resolved) |
| 779 MiB | `SVector{10,Float64}` | **`evolution.jl:378`** |
| 132 MiB | `Core.Box` | a closure capturing a mutated local |
| 25 MiB | `BigInt` / `MPQ._MPQ` | `Rational` arithmetic at run time |

`evolution.jl:378` is `∂ₜh, ∂ₜΠ = gh_rhs_at_point(…)`, whose two
`SVector{10}` returns are supposed to stay in registers — "Never build an
`SVector` of all derivatives" under [Precision, threads,
devices](CODE.md#precision-threads-devices) is about exactly this, and
`pointwise_tests.jl` asserts the node-local algebra allocates nothing. The
algebra keeps that promise; the *kernel around it* does not, so the return
is escaping rather than being inlined away. `Core.Box` is the classic
closure-capture instability, and `BigInt` at run time should not exist at
all — the stencil weights are built in `Rational` and rounded once into `T`
at compile time by design.

None of this is measured on a device, and none of it is the deferred GPU
question: these are host costs, in host code, at `q = 2`.

**The right-hand-side kernel boxes its own results, and unboxing them
collides with an invariant.** `∂ₜh, ∂ₜΠ = gh_rhs_at_point(…)` is assigned
*inside a conditional branch* and then captured by the
`ntuple(Val(NC)) do v … end` write-back closures — the idiom
KernelAbstractions forces, since it refuses a `return` in a kernel body.
Julia lowers a branch-assigned captured variable into a `Core.Box`, after
which `∂ₜh[v]` infers as `Any`, and every component write becomes a
dynamic dispatch allocating a boxed `Float64`. One cause, all three of the
allocation rows above. Wrapping each closure in `let ∂ₜh = ∂ₜh, … end`
(and the layer branch's `w`, `ρ`, `he`, `Πe` with it) measures, on the
same node:

| | allocations | time |
|---|---|---|
| as written | 3662.8 MiB | 29.54 s |
| with `let` | **365.2 MiB** | **25.51 s** |

Ten times fewer allocations and 14 % faster — and it is not only a speed
question: `pointwise_tests.jl` proves the *node-local algebra* allocates
nothing, which is true and stayed true, while the *kernel around it*
allocated ~65 MiB per evaluation. Nothing asserts that, and on a device it
is not expressible at all, so this blocks G6 rather than merely slowing
G0–G5.

**The fix is not applied**, because it fails one assertion:
`interior_tests.jl`'s `out_identical`, which requires the `:damped`
kernel outside `r_1` to be **bit-identical** to `:none` — "the same
numbers … not merely close ones", as [The interior: a pointwise damping
layer](CODE.md#the-interior-a-pointwise-damping-layer) puts it. Its two siblings
(`core_zero`, `worst ≤ 1e-12·scale`) still pass, so the disagreement is in
the last bits, and the mechanism is this document's own: the two are
separate `Val`-specialised kernels, and once the write-back is statically
typed each fuses its multiply-adds in its own inlining context. **The
invariant was being met by the bug** — both paths were equally dynamic
before. Three ways out, and the choice is a design decision: keep the
identity and the allocations; keep the fix and weaken `out_identical` to
roundoff (amending the claim here); or find a spelling that gives both.

**(Decided 2026-09-27: the fix, and roundoff.** IMEXRungeKutta's
integration of 2026-09-26 removed the box by giving each branch its own
names, and the collision arrived as predicted.**)**

**What allocation remains is TreeAMR's, and so is the `BigInt`.** With the
`let` patch in place `evolution.jl` leaves the allocation profile
entirely, and the named sites are all upstream:

| est. | site |
|---|---|
| ~22 MiB | `lagrange_weights@TreeAMR/src/operators.jl:377`–`380` — `BigInt`/`MPQ` rationals, thousands of allocations, **at run time** |
| ~10 MiB | `run_group!@TreeAMR/src/ghosts.jl:70` — KernelAbstractions launch-argument tuples, per launch |
| 1.5 MiB | `prolong_stencil@schedule.jl:480` — `BigFloat` |

So the run-time `BigInt` is not this package's stencil weights, which are
rounded into `T` once at compile time as designed: it is TreeAMR
recomputing exact-rational Lagrange interpolation weights per ghost fill.
That is the same file region as the `rem`/`div` hot spot, so TreeAMR's
ghost machinery now owns both the time and the allocations. (Two caveats
on those estimates: the profiler's own buffer appears as a single sample
scaled to 468 MiB and is discarded, and the attributed rows sum to about
30 MiB of the 365, so a fragmented tail is unaccounted for.)

The same table (the GC threads, under [What the suite costs, and
where](CODE.md#what-the-suite-costs-and-where)) calibrates GitHub. Symmetry at eight CPUs and four threads
is `16m41`, at four CPUs `21m30`, so the narrow allocation costs 29 %; and
GitHub's four-thread cell measures `22.7` and `23.2` min, which is the
four-CPU figure. Its runners behave exactly as four-core machines should.
One run of three took `61.6` min against a `43.6` min serial sibling —
that is a bad runner, not a property of the matrix, and three samples with
a 2.7× spread are not enough to act on.

### The tiers (2026-11-06)

The suite's tiers (`CODE.md`, "The suite's tiers, and where each runs") are Erik's decision of 2026-11-06; the GitHub times of the quick tier were measured on PR #23 (2026-11-07).

### After the SIMD lanes (2026-10-05)

**The suite** was 5037 assertions, green on the development machine (12m20 at four
  threads), on an EPYC 8534P (Zen 4c, AVX-512, so eight lanes; 23m59 at four threads)
  and on an EPYC 7543 (27m14 at one thread, 5029). `test/simd_tests.jl` is about 20 s
  of it, most of it the kernels at `W = 1` it compiles to compare against.

### The test files

`CLAUDE.md`'s inventory of `test/` named the files by the plan step that
added them: step 3's five (`gauge_tests.jl`, `initialdata_tests.jl`,
`evolution_tests.jl`, `convergence_tests.jl`, `noise_tests.jl`, over the
helper `evolution_cases.jl`, which lives in `test/` because `driver.jl` is
step 5's), step 4's four (`constraints_tests.jl`, `interface_tests.jl`,
`type_tests.jl`, `threading_tests.jl`), step 5's two (`interior_tests.jl`,
`driver_tests.jl`), step 6's `refinement_tests.jl`, step 7's
`horizon_tests.jl` and `hole_runs.jl`'s horizon section, step 8a's
`test/dispersion.jl` and `leakage` section, step 8c's `calibration`
section and step 8e's `fit_tests.jl`. The suite's compilation of
`SpacetimeMetrics`' nested passes is step 1's cost, unchanged since.

### CI

Code coverage has run on the macOS cell only since 2026-09-26. Erik
made Julia 1.13 the floor and the only version CI runs on 2026-10-16:
supporting earlier versions slows development and is not useful at the
moment. The two CI tiers are Erik's decision of 2026-11-05, the same in
every Tree* repository, and GitHub has run the quick tier since
2026-11-06 (Erik: a free runner is not where black holes are evolved).
With it the MPI comparison became a CI job of its own
(`TREEGH_TEST_TIER=mpi`), no longer 17.5 minutes in front of the threaded
cell's suite.

### Counts and times

The suite is **5687 assertions in 42m17** at four threads after `adm_vars`
(2026-11-05, a machine loaded 8–9 by other work, and a first try killed
halfway through. Read the wall clock as the machine's). Its 23 new claims are
`admvars_tests.jl`'s, `2.4 s` inside the suite. It was **5664 assertions in
14m12** at four threads after the bitant
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
compilation of the save and the load. It was **4610 assertions in 14m03** at one thread and **4618 in
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
fixture. 

### Commands and runs on Symmetry

The quick tier (`TREEGH_TEST_TIER=quick`) and the full suite as a commit
status (`ci/full-suite.sh`, `ci/full-suite-symmetry.sh`) were added on
2026-11-06. The notes on running the suite on Symmetry were added in step 6
(its numbers are under "Measured results", G4b). Two quirks were found on
2026-10-23, with the folds: the suite no longer fits `amddebugq`'s or
`h200debugq`'s hour, and `mpi_tests.jl` inside a SLURM job needs
`HYDRA_LAUNCHER=fork`. With both, the suite passed on an EPYC 7532 (cn084),
an EPYC 8534P (cn112) and an H200 host (cn111, Xeon Gold 6442Y), the MPI
comparison in its own job (cn079, cn113).

## The documents

### `CODE.md`'s status line (2026-09-17)

Until the reorganization of 2026-11-09, `CODE.md` opened with a status line
last written in step 4 (2026-09-17) and not updated after G3:

*Status: milestones G0–G3 are done and their numbers are in this file —
the pointwise algebra, the stencils, the fused right-hand side, coarse-fine
faces, the constraint monitors and the thread and precision invariants. A
black hole arrives with G4.* (The line here said "nothing implemented,
nothing measured" through step 3, which stopped being true at step 1;
corrected in step 4.)

### The marker (decided), retired 2026-11-09

`CODE.md`'s marker legend had four markers. The reorganization of 2026-11-09
kept **(proposed)**, **(predicted)** and **(open)**, and retired the fourth:
**(decided)** is inherited from TreeAMR,
TreeWave/TreeHydro or GHSO2, or was settled in review. The decisions it marked are unmarked in
`CODE.md` now; where their origin was recorded, it is below, under each
section.

### The review of `CODE.md` (2026-09-16)

**Settled in review** (2026-09-16, three rounds): the list of settled
decisions at the top of `CODE.md` was settled in that review. Two of its
items changed later: time integration through IMEXRungeKutta's RK4 by block owner
(through OrdinaryDiffEq until 2026-09-26); and no
checkpointing **(amended 2026-10-01**, Erik's decision: checkpoint and
restart through TreeAMR 0.1.4's M9a, see [Checkpoint and
restart](CODE.md#checkpoint-and-restart)**)**.

`CLAUDE.md`'s "Current state" said, until 2026-11-09:
`CODE.md` is complete and reviewed three times (2026-09-16): the expanded
form of the momentum equation, three dimensions only, a pointwise damping
layer instead of excision, a single boosted spinning black hole as the
proof-of-concept target, RK4 from OrdinaryDiffEq (IMEXRungeKutta's RK4 by
block owner since 2026-09-26), the analysis quantities
as part of the deliverable, an error indicator for refinement, `Float64`
on Symmetry's H200 as the device requirement, no checkpointing, GPU
kernel efficiency deferred to a research project.

### `SINGULARITY_HANDLING.md`

2026-10-08: the runs behind the interior's choices moved from `CODE.md` to
`SINGULARITY_HANDLING.md`.

### The interior's measurements leave `CODE.md` (2026-10-08)

On 2026-10-08 the measured results of the black hole's interior — step 5's layer (G4a), step 8a, steps 8b–8′ and the single holes on the octant — moved from `CODE.md`'s Measured results to `SINGULARITY_HANDLING.md`, leaving pointers behind (a subsection "Steps 8b–8′: the generic interior and the moving hole" among them).

### PLAN.md

Until 2026-10-25 PLAN.md's introduction said: "Delete this file when the last milestone is marked *(Done.)* in `CODE.md`."
On 2026-10-25 step 10 was amended to keep the file, and the binary steps,
until G8 is *(Done.)*.

On 2026-10-08 the measured results of the interior — step 5's layer,
steps 8a–8′, and the open questions they answered — moved from `CODE.md`
to `SINGULARITY_HANDLING.md`, and PLAN.md's introduction gained the note
**(Amended 2026-10-08:** … **)** saying so.

Until 2026-11-09 PLAN.md's ground rules said:

> - Spec-first: when the implementation shows `CODE.md` wrong or
>   incomplete, amend it and say so in it ("amended in step N"), and
>   replace each **(predicted)** the step measures with the measured
>   number, marked **(measured in step N)**. Never loosen an existing
>   assertion to get green; report the failure instead.

and an agent that decided where its step was silent marked the decision in
`CODE.md` as **(proposed in step N)**. On 2026-11-09 the rule became: fix
`CODE.md` to describe the current state, record in `HISTORY.md` what
changed, when and why, and replace each **(predicted)** by the measured
number in `CODE.md`; a decision an agent takes is marked **(proposed)** in
`CODE.md`, and `HISTORY.md` records which step proposed it.

PLAN.md's remaining provenance markers, dropped on 2026-11-09: G5's
spin `a = 7/10` was decided on 2026-09-23; G5's acceptance was amended in
step 8f ("the analytic core cuts harmonic Kerr's disk at `a = 7/10`, so the
analytic layer is not available on G5's own chart, and runs as the control
on the boosted `a = 0` hole") and in step 8 ("on the analytic data of the
same geometry where the chart allows it" became the cycle on the case's own
data, which agrees with it where it allows it); step 9's `ρ_max = 4/M`
default was decided on 2026-09-23 in step 8c′; the remote directory per
study as the likeliest cause of step 8's `SIGBUS` was proposed in step 8′;
the paragraph "Checkpointing exists" was added on 2026-10-01; and step 10's
"delete this file" was amended on 2026-10-25 to keep it until G8 is done.

### `CLAUDE.md`'s "Current state" (until 2026-11-09)

Until 2026-11-09 `CLAUDE.md` had a section "Current state": the bold
status paragraph (now `README.md`'s "Status"), an inventory of `src/`,
and one paragraph per step or date saying what it added. The API and
file facts of those paragraphs are in `CODE.md` ("File layout" and the
section of each); what each step added is under "The implementation
plan" below, and the dated changes under the section they belong to.
Its opening also noted that the runs behind the interior were moved out
of `CODE.md` into `SINGULARITY_HANDLING.md` on 2026-10-08, and its rule
"No mesh machinery" that the horizon finder's stopgap interpolation went
upstream on 2026-09-26 (TreeAMR 0.1.3's M11); its rule "The interior is
pointwise and generic" was amended for PLAN.md's steps 8a–8g, added
2026-09-23.

### The rules for amending the documents

- Until 2026-11-09 `CLAUDE.md`'s conventions said:

  > Spec-first: when the implementation shows `CODE.md` was wrong or
  > incomplete, amend it and say so in it — "(amended in step N)",
  > "(measured in step N)" — rather than diverging silently.

  It was replaced by: fix `CODE.md` to describe the current state, and
  record in `HISTORY.md` what changed, when and why.
- Until 2026-11-09 the measured-numbers rule said that measured numbers go
  into `CODE.md` beside the prediction they confirm or correct, those of
  the black hole's interior into `SINGULARITY_HANDLING.md` and a
  recommendation that changes into `CODE.md`'s table (the last two from
  2026-10-08). It now sends numbers that support a current decision to
  `CODE.md` and replaced ones to `HISTORY.md`.
- The table-of-contents convention was added on 2026-10-05, for
  `CLAUDE.md` and `CODE.md` only ("of every heading below the title");
  on 2026-11-09 it was extended to `README.md`, `PLAN.md` and
  `HISTORY.md`, with `##` and `###` headings for `CODE.md`, `PLAN.md` and
  `HISTORY.md` and `##` only for `README.md` and `CLAUDE.md`.

### When the sharp edges of `CLAUDE.md` were added

The second half of "Things that will bite" carried these markers:

- the reference amplitude and the travelling margin: step 6;
- the horizon finder's guard: exact bit for bit against the norms' mask
  since 2026-09-26, a `ShapeMask`'s on a tracked geometry from step 8d;
- the moving `:damped` core, the tracked `:damped` core, the contracted
  horizon, the quadrant: 2026-10-20;
- the rotating quadrant, a moving hole's symmetric domain, the ring rule,
  `hz_J`, `1/48` on one H200: 2026-10-21;
- the analysis row on a GPU and the `a = 9/10` layer's speed: 2026-10-22;
- the half turn: 2026-10-24;
- a refined octant, an octant's corner: 2026-10-02;
- a test that evolves a hole is `@full_tier`: 2026-11-06.

### The README's status at milestone G3

Until the reorganization of 2026-11-09 the README carried the status as
of milestone G3 (step 4):

**Status: milestones G0–G3 are done — the equations are solved on an
adaptively refined mesh at the order the interface rule predicts, both
constraint monitors converge, and a run is bit-identical across thread
counts; a black hole is next.** What exists is the module shell, the
`Base` bridges for software floating-point types, the host-copy helpers,
GHSO2's node-local generalized-harmonic algebra together with the expanded
momentum equation this package discretises, the centered finite-difference
and Kreiss–Oliger weights at `q ∈ {2, 4, 6, 8}` built in exact rational
arithmetic and rounded once into the run's type, and the mesh-side
physics: the fused right-hand-side kernel, the cases and their initial
data, the prescribed gauge source, the time-dependent Dirichlet boundary,
the CFL time step, and the gauge and ADM constraint monitors with their
masked norms.

The tests say the pinned TreeAMR still provides what the scheme is written
against, that a `SpacetimeMetrics` background — including the boosted,
spinning one the proof of concept runs — compiles and runs as a kernel
argument, that the algebra and the weights agree with an independent
source (automatic differentiation for the equations, exact rational
arithmetic for the stencils), and that the kernel evaluates the same
equation the validated pointwise reference does. The gauge wave converges
at **1.99, 3.95 and 5.92** for `q = 2, 4, 6`, flat space in a Dirichlet
box with a shift and a sampled gauge source at **3.93**, Minkowski's
right-hand side is *exactly* zero, and white noise stays bounded over a
thousand steps with Kreiss–Oliger dissipation at `ε = 0.5` while growing
tenfold without it.

Across a coarse-fine face the same wave converges at **3.18** with an
order-4 prolongation and **3.98** with an order-6 one, which is what makes
`p = q + 2` a requirement rather than a taste; both constraint monitors
converge there too, and vanish to roundoff on exact data; the whole cycle
— initial-data adaptation, evolution, a regrid that moves data, the
monitors and their norms — prints digests that are identical character for
character at one and four threads; and everything runs at `Float32`,
reproducing the gauge wave's rate and its error. There is no refinement
indicator, no black-hole interior and no driver yet; those are the next
three steps.

## The implementation plan

PLAN.md breaks the milestones into steps, each a brief for one agent with
a fresh context: steps 0–10 for G0–G6, with steps 8a–8g (the generic
interior) and 8′ (the trailing side) added before and after step 8, and
steps B1–B9 for the binary black holes, G7 and G8. The briefs of the
finished steps follow, as PLAN.md had them.

### Milestone G0

- **G0 — Scaffolding.** *(Done.)* `Project.toml` with the pins, CI on
  1.11 and release at one and four threads (1.13 only from 2026-10-16),
  `CLAUDE.md`, this document, `notes/`; `precision.jl`, `device.jl`; a prerequisite test that the
  pinned TreeAMR exports what this package calls, and that a
  `SpacetimeMetrics` metric with its `dmetric` pass runs as a kernel
  argument on `CPU()`. *Accept:* a clean archive instantiates and passes.
  The prerequisite test runs **two** backgrounds rather than the one
  named above **(proposed in step 0)**: `KerrSchild(1, 0)` and
  `boost(Harmonic(1, 9/10), 0.3 x̂)`, the proof-of-concept case itself.
  `KerrSchild` is static, so the `∂_t g` its `dmetric` pass returns
  vanishes identically and a callback that filled the momentum half from
  the wrong slice of `dg` — the two derivative index conventions above
  are exactly that mistake — would pass. (That test fills those ten slots
  with `∂_t g` as a *stand-in*: the evolved `Π` above is densitised and
  Lie-advected, and equals `∂_t g` only where the shift vanishes and
  `α = √γ`. Nothing in step 0 evolves anything, and the claim under test
  is about the kernel argument; the real `Π` arrives with
  `initialdata.jl`.) The boosted case is also the expensive one to
  compile, nested duals through a coordinate pullback, and it compiling
  is what the
  dependency risk under [Initial data and
  backgrounds](CODE.md#initial-data-and-backgrounds) is about.

### Milestone G1

- **G1 — Pointwise algebra and stencils.** *(Done.)* `pointwise.jl`,
  `stencils.jl`. *Accept:* against `SpacetimeMetrics` automatic
  differentiation on every background — ADM extraction, the offset
  identities at `‖h‖ ~ 1e−13`, GHSO2's identity `∂_tΠ − ∂_iF^i = msrc`
  by finite differences of the analytic solution, the expanded form's
  coefficient derivatives against a dual pass, the vanishing of `C_a`
  and `Z_ab` on exact data; rational stencil weights reproducing the
  textbook tables and exact to degree `q`; every function callable from
  a trivial kernel on `CPU()`. `pointwise.jl` and its half of the
  acceptance are done (step 1); `stencils.jl` and the weights are step 2,
  which marks the milestone. The numbers are under [Measured
  results](CODE.md#measured-results). Step 2 found that the acceptance was one
  claim short of the design: "exact to degree `q`" is the *first*
  derivative's statement, and the compact second derivative is exact to
  degree `q + 1` by the symmetry of an even `q`; both are asserted, with
  their "and not one degree higher" halves, in `Rational` (amended in step
  2). And the one thing the step found that the design did not say: see [Measured results](CODE.md#measured-results), G1a.

### Milestone G2

- **G2 — The RHS on a uniform periodic mesh.** *(Done.)* `evolution.jl`,
  `initialdata.jl`, `gauge.jl`, `boundaries.jl`; the gauge wave and
  shifted Minkowski cases; RK4. *Accept:* Minkowski and shifted Minkowski
  are stationary to roundoff (the RHS is exactly zero on data the stencils
  reproduce); the gauge wave converges at order `q` for `q = 2, 4, 6` on
  `N = 8`, roots `2 … 8`; white noise on flat space stays bounded for
  a thousand steps with `ε_KO > 0` and its growth without is recorded;
  the RHS is pure and never mutates `u`; the fused kernel's throughput
  per point recorded. Three of those sentences needed amending, and step 3
  amended them where they are stated rather than only here:
  **Minkowski is exactly stationary and shifted Minkowski is not** — its
  `Π` is built from the *analytic* spatial gradients, which the stencils
  reproduce only to truncation order, so it is stationary to `O(h^q)` and
  what converges at order `q` is its error (GHSO2 built `Π` from the
  discrete gradients to make the stronger statement true, and `CODE.md`
  keeps that as a G4 post-pass; `PLAN.md`'s step 3 had it right where this
  line did not). **`N = 8` does not exist at `q = 6`**, where TreeAMR's
  vertex invariant forces `N ≥ 10`. And **roots `2 … 8` over one crossing**
  is minutes per order in three dimensions: what is run is roots `1, 2, 3`
  over an eighth of a crossing, which measures the same rate — the
  numbers are under [Measured results](CODE.md#measured-results).

### Milestone G3

- **G3 — Coarse-fine faces, static mesh.** *(Done.)* TreeAMR's two-level
  mesh; `constraints.jl`. *Accept:* the interface-order table (predicted
  rates 3 and 4 at `p = 4, 6` for `q = 4`, control at 4), independent of
  the restriction order and of `ε_KO`; both constraint monitors converge on
  the gauge wave across the interface; the thread-workload digests
  identical at one and four threads; G2 and G3 on `CPU()` in `Float32`.
  All four hold; the numbers are under [The interface-order
  rule](CODE.md#the-interface-order-rule-and-what-it-costs-a-second-order-system),
  [Analysis quantities](CODE.md#analysis-quantities) and [Precision, threads,
  devices](CODE.md#precision-threads-devices), and the suite's cost is under
  [Measured results](CODE.md#measured-results). Two sentences of the acceptance
  needed amending where they are stated rather than only here.
  **"Both constraint monitors converge on the gauge wave across the
  interface" is a claim about the interface and nothing else**: the same
  case on a *uniform* mesh has no constraint violation above roundoff, so
  the refined rows measure the interpolated ghosts alone, and they
  converge at `q + 1/2` rather than at `q` because that error lives on a
  set of measure `~h`. The statement that the monitors converge at `q` is
  made on harmonic Kerr, where the violation is the bulk truncation error
  — and both rows are in the suite, because between them they say the
  monitors see both. And **the restriction order is not a row of the
  table**: on a vertex-centered mesh restriction is injection, so the two
  orders are the same computation and the test asserts bit-identity rather
  than two equal rates.

### Milestone G4

- **G4 — A black hole.** *(Done.)* Kerr-Schild (`a = 0`, sampled `H`) and harmonic
  Kerr (`a = 0` and `a = 0.9`, `H = 0`) in a Dirichlet box of half-width
  `≥ 20 M`, the indicator with its mask, floor and ceiling, the interior
  layer, the recipe (`ε_KO = 0.5`, `γ0 = 1/M`), `driver.jl`,
  `interior.jl`, `refinement.jl`, `horizon.jl`. *Accept:* the
  calibration table of `τ_max` against `h` and the thresholds chosen
  from it; the initial-data cycle converges to a hierarchy of nested
  shells around the hole, coarse at the boundary, with the floor not
  binding at calibrated thresholds and binding when they are loosened;
  the radius assertions asserted and tested to fire; the masked error
  against the exact solution converging at order `q` on the frozen
  hierarchy as `N` doubles, to `t = 50 M`; constraints flat at
  truncation and *masked*; the interior residual at truncation; the
  horizon found, enclosing the layer by the margin, with its area,
  `M_irr`, `J` and `M_ch` at the Kerr values to interpolation accuracy;
  the analysis record complete at every chunk; the three interior
  variants measured and the default confirmed or changed; the gauge
  drift rate recorded beside GHSO2's `≈ 0.14/M`; the discrete-gradient
  `Π` post-pass measured; regrids on the static hole changing nothing
  after the cycle; the same run in `Float32` on `CPU()` reaching the
  same mesh.
  **G4a (step 5) is done**: `interior.jl`, `driver.jl`, the layer with
  its three variants, the radius assertions, the per-chunk analysis
  record, the masked error at order `q` on a frozen hierarchy, the
  `Float32` row, the drift and the `Π` post-pass — the numbers are under
  [step
  5](SINGULARITY_HANDLING.md#step-5-the-static-holes-interior-and-the-driver-g4a).
  **G4b (step 6) is done with it**:
  `refinement.jl`, the masked Löhner indicator with its global amplitude,
  the four marks, the derived level floor and the boundary ceiling, the
  travelling margin, the initial-data cycle and the regrid branch of the
  driver, the calibration table and the thresholds chosen from it, and the
  refinement centroid in the record. **G4c (step 7) closes it**:
  `horizon.jl`, the stopgap interpolator with its footprint guard, the
  batched interpolating `ADMVars` provider, `find_gh_horizon` over
  `ApparentHorizonFinder` and `KorzynskiSpin`, and the horizon rows of the
  analysis record at the case's own cadence — with Kerr's `A`, `M_irr`,
  `J` and `M_ch` recovered from sampled data in both charts and at
  `a = 9/10`, from a displaced guess, enclosing the layer by the margin.
  Two sentences of the acceptance needed amending where *they* are stated.
  **"harmonic Kerr at `a = 0.9`" is not among the rows**, for step 5's
  reason: the chart is refused by `check_interior_radii`, so the spinning
  hole's horizon is measured in Kerr-Schild, where `r₊ = 1.436 > |a|`, and
  the second chart is harmonic Kerr at `a = 0`. And **the horizon rows are
  not measured in the suite at `a = 9/10`**: the layer of a spinning hole
  needs `h ≈ 0.04 M` and that is a thousand blocks, so the suite runs
  `a = 0` and `test/hole_runs.jl` runs the rest. Two further sentences needed
  amending where they are stated rather than only here. The calibration's portability is now under [Measured
  results](CODE.md#measured-results), G4b. And **the
  indicator's mesh and the hand-built hierarchy of step 5 are not the same
  mesh at the same cost**: see the comparison under [Measured
  results](CODE.md#measured-results). Two sentences of the acceptance needed amending where they are
  stated rather than only here. The production box is under the same G4b.
  And **the three variants are separated by the interior residual and by
  what survives to `t = 50 M`, not by the constraints outside `r_1`** —
  see the numbers; the prediction that `:damped` is the smaller of the two
  violations holds against `:pasted` and is a 3 % effect, while the
  residual separates `:frozen` from the other two by more than an order of
  magnitude and the long run separates all three: only `:damped` reaches
  `50 M`. A third sentence needed amending, in
  [the interior](CODE.md#the-interior-a-pointwise-damping-layer) where it is
  stated: **`a = 9/10` in the harmonic chart is refused**, because the
  chart's singular disk has coordinate radius `a` and the horizon's
  smallest coordinate radius is `√(M² − a²)`, so no ball contains the one
  and fits inside the other. That is G5's case, and it is the open
  question step 5 leaves.

### The plan's status

G5's case, the moving hole at `a = 7/10`, was decided on 2026-09-23. The
recommendation "more interior work, the spin drift first, before
excision" was proposed in step 8′ (2026-09-25). Steps B1–B9 were added
to the plan on 2026-10-25, steps C1–C5 on 2026-11-09 and steps H1–H5 on
2026-11-11. Step H1 ran on 2026-11-11.

### Step 0 — Scaffolding (G0)

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

### Step 1 — Pointwise algebra (G1a)

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

### Step 2 — Stencils (G1b)

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

### Step 3 — The right-hand side and the gauge wave (G2)

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

From step 3 `src/` had the mesh-side physics: `evolution.jl`,
`initialdata.jl`, `gauge.jl` and `boundaries.jl`.

### Step 4 — Coarse-fine faces, constraints, threads (G3)

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

Step 4 added `constraints.jl`, `metric_derivatives_along` to
`pointwise.jl` and a `refined = true` forest to `initialdata.jl`.

### Step 5 — A black hole, static: the interior and the driver (G4a)

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

From step 5 there is a **black hole**: `interior.jl` and `driver.jl`.
`evolution.jl` grew the `(INTERIOR)` term, a fifth `Val` carrying the
interior *variant*, and the `:pasted` variant's `step_limiter!`;
`constraints.jl` grew the error kernel and `error_norms`; `gauge.jl` grew
the position-dependent `γ0` profile; `initialdata.jl` grew `hole_forest`
and the two hole cases.

### Step 6 — The refinement indicator (G4b)

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

From step 6 there is a **mesh the code chose**: `refinement.jl`.
`driver.jl` grew `adapt` and `regrid`, and the record grew `τ_max`, the
centroid and its distance from the analytic center.

### Step 7 — Horizons (G4c)

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

From step 7 the code **knows where the horizon is**: `horizon.jl`.
`GHCase` grew a sixth type parameter (the `Horizon`) and `with_horizon`,
and the record the horizon rows. `CLAUDE.md` then said: "There is no
I/O and no CLI: those are step 9."

### Steps 8a–8g — The generic interior

Added to the plan on 2026-09-23.

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

#### Sharp edges for steps 8a–8g

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
- **Work in the worktree you were given**, on your step's branch, commit
  in TreeAMR's style with measured numbers in the body, do not merge or
  push, and do not touch `TODO.md` or `notes/`.

Two more items of this list moved on 2026-11-09: the one on long runs
on Symmetry to PLAN.md's "Sharp edges to know before starting", and the
one on the fixtures to `CODE.md`'s "File layout".

### Step 8a — Expectations: anomalous group velocity and leakage

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

### Step 8b — The instrument: range projection and validity monitor

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

From step 8b there is an **instrument**: `bounds.jl`. `diag` has 22 slots (17 from
2026-10-23, when the monitor became a fold); the record grew
`bounds_hits`, `bounds_nonfinite`, `bounds_r_max` and the eight
`min_detγ_layer` … `max_Π_shell` rows; `test/bounds_tests.jl` and
`hole_runs.jl`'s `bounds` section were added.

### Step 8c — Calibrate the layer for an inexact target

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

From step 8c the layer can be **calibrated against a wrong target**:
the `Interior`'s `target`, `evolve!`'s `ρ_max_fixed` and `gauge.jl`'s
`HorizonDissipation`, which `GHCase` takes where it took a number before:
`ε_KO` became a type parameter; `test/evolution_cases.jl` gained
`CurvatureTarget` and `gh_outside_shell_norms(p, u, t)`, `hole_runs.jl`
a `calibration` section.

### Step 8c′ — `ρ_max = 4/M` is the default

Decided 2026-09-23.

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

From step 8c′ **`ρ_max = 4/M` is the default** (Erik's decision of
2026-09-23, for every variant, the analytic `:damped` layer included):
`hole_mass` and `default_relaxation_rate` were added, and the grid rate
became the option `ρ_max_factor = 1`.

### Step 8d — The tracked horizon geometry

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

From step 8d the layer can **follow the horizon that was found**:
`FittedSpec`, `FittedInterior`, `tracking.jl`, the depth, the shape
masks, the record's `track_*` and `layer_*` rows and `margin_efolds`;
`AbstractSphericalHarmonics` became a direct dependency, and
`test/tracking_tests.jl` was added.

### Step 8e — The fitted target and the `:fitted` variant

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

From step 8e-i there is a **fitted target** on the host (`fit.jl`,
`hole_velocity`, `test/fit_tests.jl`); from step 8e-ii a **`:fitted`
variant** (the 40-variable target cache, `refit!`, the pieces of a
moving hole's chunk, `hole_runs.jl`'s `fitted` section).

### Step 8f — The measurement matrix

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

From step 8f the fit holds **`Π̃ = (α/√γ)Π`** by default; step 8f
added `with_variant`, `handover`, `target_source = :snapshot`, the
`:fitted` initial-data cycle on the analytic layer, the `variant` row,
`hole_runs.jl`'s `generic` section, and a moving hole's step sized from
the previous chunk's growth.

### Step 8g — Excision by a host reference (only if 8f says so)

Not run: `:fitted` reached `50 M` on the first row of step 8f's table.

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

### Step 8 — A hole that moves (G5)

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

From step 8 the hole **moves across the mesh**: step 8 added
`adapt_fitted_initial_data!` and `fill_fitted_initial!` (step 8f's cycle
flagged on the analytic layer and refused G5's chart), `target_rate`,
`fit_initial_blend`, the floor from the core surface widened by the
travel, a moving seed's `r_min` from its analytic surface, the 1 % step
margin, `test/moving_tests.jl` and `hole_runs.jl`'s `moving` section.

### Step 8′ — The trailing side

Added to the plan on 2026-09-24, after step 8's report.

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

### Step H1 — A harmonic hole at rest

Added to the plan on 2026-11-11 with steps H2–H5; ran the same day.

`CODE.md`: "What the package has measured in this chart", "What harmonic
gauge costs"; "Single black holes: recommended settings". **`a = 0`, on the
mirrored octant.**

**Changes:**

- **`case=harmonic` in `test/octant_runs.jl`**, through
  `harmonic_kerr_case`, with `a=`, both interiors, the error reference and
  the shells about the hole.
- Nothing in `src/` unless a run shows a defect; then the defect first.

**Accept**, at `q = 4`, `cfl = 1/2`, `ε_KO = 1/2`, `ρ_max = 4/M`:

- **The settings**, screened to `8 M`: `:damped` about `r_0 = 2/5`,
  `r_1 = 3/4` at `1/32`; `:fitted` about `m = 16`, `n_L = 20` at `1/48`.
  Report the coarsest spacing at which each holds.
- **The order**: `:damped` at three spacings from that one, to `24 M`:
  order four in the error and in `ℋ` outside the horizon, in the shells
  `[2, 4)` and `[4, 7)` (the Kerr-Schild rows' `[3, 5)` and `[5, 8)` in
  areal radius).
- **`:fitted`** at the same spacings, with the margin fixed in `M` and
  fixed in cells: its `ℋ` and error against `:damped`'s, and its order.
- **Stationary**: both variants to `128 M` at the coarsest convergent
  spacing: the shells' error, `ℋ` and `C_a` flat from `24 M`, `M_irr`
  constant to the Kerr-Schild row's level, the horizon's radius `M`.
- **The price**: the spacing at which this chart's shell error equals the
  Kerr-Schild `a = 0` hole's at `1/24`, and the cost per `M` of both on an
  H200.
- **If the hole is not stationary**, report the growth's rate, where it
  sits and its order, and stop: the route rests on this row.
- `CODE.md`'s **(predicted)** settings and costs replaced by what was
  measured.
