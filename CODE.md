# TreeGeneralizedHarmonic.jl — Design

TreeGeneralizedHarmonic.jl evolves the vacuum Einstein equations in the
generalized harmonic (GH) formulation on the adaptively refined block
mesh of [TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl). It is the
third downstream application of TreeAMR, after
[TreeWave](https://github.com/eschnett/TreeWave.jl) (a second-order
finite-difference scheme, no conservation) and
[TreeHydro](../TreeHydro) (a finite-volume scheme, conservation at
coarse-fine faces). Its end goal is a production code; what this
document specifies is the **proof of concept** that comes first, and
the milestones below stop where the proof is made.

The physics is not new to this author's packages: the formulation, the
pointwise algebra, the constraint damping, the gauge sources and the
dissipation-and-damping recipe for a black hole are those of
`GeneralizedHarmonicSecondOrder2` (GHSO2 below), validated there on
SBP-SAT spectral elements. Those repositories are unpublished and not
citable, so the documents this design rests on are **copied verbatim
into [`notes/`](notes/README.md)** and cited from there. What is new is
the mesh: finite differences with ghost zones on an octree of uniform
blocks, a global time step, regridding driven by an error indicator, and
— through the mesh — threads, GPUs and (when TreeAMR's M7 lands) MPI,
without this package writing any of that itself.

*Status: milestones G0–G3 are done and their numbers are in this file —
the pointwise algebra, the stencils, the fused right-hand side, coarse-fine
faces, the constraint monitors and the thread and precision invariants. A
black hole arrives with G4.* (The line here said "nothing implemented,
nothing measured" through step 3, which stopped being true at step 1;
corrected in step 4.) Markers: **(decided)** is inherited from TreeAMR,
TreeWave/TreeHydro or GHSO2, or was settled in review; **(proposed)** is
a decision this document makes and still wants confirmed;
**(predicted)** is a number a milestone will measure; **(open)** points
at [Open questions](#open-questions). What this package needs from
TreeAMR is under [Upstream prerequisites](#upstream-prerequisites).

**Settled in review** (2026-09-16, three rounds): the expanded form of
the momentum equation, not the flux form; three spatial dimensions only;
no excision — inside the horizon the solution is *driven to the analytic
one* by a pointwise damping layer and frozen around the singularity,
with nothing in that decision knowing about blocks or levels **(amended
2026-10-05**, Erik's decision: excision is reopened as an interior variant
under study, `:excised`, beside the layer — see
[Excision](#excision-added-2026-10-05)**)**; the
proof-of-concept target is a **single black hole with nonzero boost and
spin**; time integration through IMEXRungeKutta's RK4 by block owner
(through OrdinaryDiffEq until 2026-09-26); the analysis quantities
(constraints, horizon location, area, mass and spin) are part of the
deliverable; refinement is driven by an **error indicator**, not by
prescribed spheres; `Float64` on Symmetry's H200 is the device
requirement and `Float32` on a device is desirable, not required; no
checkpointing **(amended 2026-10-01**, Erik's decision: checkpoint and
restart through TreeAMR 0.1.4's M9a, see [Checkpoint and
restart](#checkpoint-and-restart)**)**; GPU kernel efficiency is a later
research project; the inherited documents live in `notes/`.

**Contents**

- [Goals](#goals)
- [Scope and non-goals](#scope-and-non-goals)
- [Lineage: what is inherited and what changes](#lineage-what-is-inherited-and-what-changes)
- [The equations](#the-equations)
- [Discretization](#discretization)
  - [Field sets and layout](#field-sets-and-layout)
  - [Finite-difference stencils](#finite-difference-stencils)
  - [Kreiss–Oliger dissipation](#kreissoliger-dissipation)
  - [The interface-order rule, and what it costs a second-order system](#the-interface-order-rule-and-what-it-costs-a-second-order-system)
  - [One right-hand-side evaluation](#one-right-hand-side-evaluation)
  - [The time step](#the-time-step)
- [Gauge and constraint damping](#gauge-and-constraint-damping)
- [Boundaries](#boundaries)
  - [Periodic](#periodic)
  - [Outer boundary: Dirichlet from the background, at the current time](#outer-boundary-dirichlet-from-the-background-at-the-current-time)
  - [Reflecting faces: symmetry planes (added 2026-10-02)](#reflecting-faces-symmetry-planes-added-2026-10-02)
  - [The rotating octant: a quarter turn about `z` (added 2026-10-04)](#the-rotating-octant-a-quarter-turn-about-z-added-2026-10-04)
  - [The interior: a pointwise damping layer](#the-interior-a-pointwise-damping-layer)
    - [The design, as steps 8a–8f leave it (rewritten in step 8f)](#the-design-as-steps-8a8f-leave-it-rewritten-in-step-8f)
    - [Step 5's layer: the analytic control](#step-5s-layer-the-analytic-control)
    - [The margin](#the-margin)
    - [The spinning harmonic chart](#the-spinning-harmonic-chart)
    - [Why touch the interior, and why relax](#why-touch-the-interior-and-why-relax)
    - [The profiles and the rate](#the-profiles-and-the-rate)
    - [The layer for an inexact target](#the-layer-for-an-inexact-target)
    - [The range projection](#the-range-projection)
    - [The tracked geometry](#the-tracked-geometry)
    - [The fitted target](#the-fitted-target)
    - [What the layer costs](#what-the-layer-costs)
    - [Excision (added 2026-10-05)](#excision-added-2026-10-05)
- [Initial data and backgrounds](#initial-data-and-backgrounds)
- [Refinement and regridding](#refinement-and-regridding)
- [Time integration](#time-integration)
- [Analysis quantities](#analysis-quantities)
- [Precision, threads, devices](#precision-threads-devices)
- [I/O and viewers](#io-and-viewers)
- [Checkpoint and restart](#checkpoint-and-restart)
- [Upstream prerequisites](#upstream-prerequisites)
- [File layout](#file-layout)
- [Milestones](#milestones)
- [Measured results](#measured-results)
  - [What the suite costs, and where (measured 2026-09-19 on Symmetry)](#what-the-suite-costs-and-where-measured-2026-09-19-on-symmetry)
  - [Steps 8b–8′: the generic interior and the moving hole](#steps-8b8-the-generic-interior-and-the-moving-hole) (moved to [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md))
  - [Robust stability on the octant (measured 2026-10-02)](#robust-stability-on-the-octant-measured-2026-10-02)
  - [Single black holes: recommended settings (added 2026-10-08)](#single-black-holes-recommended-settings-added-2026-10-08)
  - [The right-hand side on an H200 (measured 2026-10-05)](#the-right-hand-side-on-an-h200-measured-2026-10-05)
  - [The right-hand side on a CPU (measured 2026-10-05)](#the-right-hand-side-on-a-cpu-measured-2026-10-05)
  - [Excision, steps X1–X7](#excision-steps-x1x7) (moved to [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-steps-x1x7-2026-10-05-to-2026-10-08))
- [Possible extensions](#possible-extensions)
- [Open questions](#open-questions)

## Goals

- **A proof of concept of a black-hole GH code on TreeAMR.** A boosted,
  spinning Kerr black hole crossing the domain, its interior driven to
  the analytic solution, refinement following it because an error
  indicator says so, the constraints and the horizon's location, area,
  mass and spin recorded as time series; correct in `Float64`, running
  in `Float64` on Symmetry's H200, bit-identical across thread counts.
  Milestones G4–G6 define the proof. What a production code adds on top
  — binaries, excision, a dynamical gauge, radiative boundaries,
  an efficient GPU kernel — is under
  [Possible extensions](#possible-extensions), each with the design note
  that would start it.
- **GPU-efficient, meaning memory-bandwidth-frugal.** GHSO2's goal
  statement carries over verbatim: reduce storage and memory bandwidth
  as far as the formulation allows. That is why the second-order-in-space
  form with 20 evolved fields is kept, not the 50-field first-order
  reduction, and why the momentum equation is discretised in the form
  that needs the fewest ghost layers. What the design does *not*
  promise is an efficient GPU *kernel*: a GPU has few registers,
  especially for `Float64`, and a straightforward kernel of this size
  spills. Kernel efficiency is a **later research project** (decided in
  review); the design commits only to a kernel structure that does not
  foreclose it, to a short list of black-box attempts, and to measuring
  the result — see [One right-hand-side
  evaluation](#one-right-hand-side-evaluation) and
  [Precision, threads, devices](#precision-threads-devices).
- **Exercise TreeAMR as a finite-difference AMR code would.** Wide
  ghost zones, high-order interpolation at coarse-fine faces, an
  error-driven refinement hierarchy that moves, and boundary data that
  depends on time.
- **Inherit, do not re-derive — and keep the source here.** The
  formulation and its well-posedness argument, the cancellation-free
  offset algebra, the sonic-surface investigation and the pointwise
  code are in `notes/` as verbatim copies with their provenance; this
  document cites them and writes out only what the mesh changes.

## Scope and non-goals

- **Vacuum only.** No matter, no scalar field. The reduced source keeps
  the algebraic slot where a stress-energy term would enter.
- **Second order in space, first order in time** (decided; see
  [The equations](#the-equations)). The fully first-order
  Lindblom–Scheel–Kidder–Owen–Rinne system is not implemented: it was
  the candidate cure for an excision instability that GHSO2 resolved
  inside the second-order form, and this package does not excise.
  **(Amended 2026-10-05:** excision is under study as the `:excised`
  variant, in the second-order form, with GHSO2's resolution of that
  instability as its recipe — [Excision](#excision-added-2026-10-05).**)**
- **Three spatial dimensions only** (decided). TreeAMR is `D`-generic
  and its siblings test in `D = 1, 2`; this package fixes `D = 3`, the
  spacetime metric has 10 components, and GHSO2's four-dimensional
  pointwise algebra is used as it stands. The cost is that every test is
  a 3D test and must be small; the gain is no second copy of the tensor
  algebra.
- **Finite differences with ghost zones, not SBP-SAT.** TreeAMR's model
  is overlapping stencils through exchanged ghosts, and the energy
  estimate that motivated SBP-SAT in GHSO2 does not survive interpolation
  at coarse-fine faces in any case. Stability rests, as in every
  finite-difference GH code since Pretorius (2005), on centered stencils
  plus Kreiss–Oliger dissipation, and is *measured* (robust-stability
  tests) rather than proven.
- **Singularity handling: the interior layer, and excision under study
  (amended 2026-10-05**, Erik's decision; this bullet read "**No
  singularity handling** (decided). Nothing is excised." until then**)**.
  Excision is reopened as an interior variant, `:excised`, beside the
  layer: points beyond a surface well inside the horizon are not evolved,
  and evolved points whose stencils reach them use one-sided closures
  instead — no target, no fit. Whether it is stable on a Cartesian lego
  surface, and whether it holds the gauge as well as the layer does, is
  what `PLAN.md`'s steps X1–X3 measure; the design is under
  [Excision](#excision-added-2026-10-05). **(Measured in step X3:** stable,
  and as accurate outside the horizon as the layer, on the static
  Kerr-Schild `a = 0` hole, with no gauge drift to `50 M`.**)** The rest of
  this bullet is the layer's, as it stood. Where a run has a black hole,
  the right-hand side inside the horizon is modified point by point:
  driven toward the analytic solution in a layer well inside the horizon,
  switched off around the singularity (see [The
  interior](#the-interior-a-pointwise-damping-layer)). This
  restricts black-hole runs to spacetimes with a known analytic
  solution — which is the proof-of-concept target, and exactly what an
  excision milestone would be measured against. **(Amended 2026-09-23,
  PLAN.md steps 8a–8g.)** The restriction is lifted by the *generic*
  interior: the tracked apparent horizon supplies the geometry, a regular
  fit of the evolved state the relaxation target, and a range projection
  deep inside the insurance that reports where either fails. Nothing is
  excised still; excision is priced under [Possible
  extensions](#possible-extensions) and decided by step 8g's host-side
  test. The design as those steps amend it is under [The
  interior](#the-interior-a-pointwise-damping-layer) and the decision
  under [the interior's
  questions](SINGULARITY_HANDLING.md#the-interiors-questions-opened-in-step-5-and-closed-through-step-8).
- **Analytic initial data, boundary data and gauge sources only.**
  Everything `SpacetimeMetrics` provides: Minkowski, the gauge wave,
  shifted Minkowski, Kerr-Schild, Kerr in harmonic coordinates, and
  translations, rotations and boosts of these. Constraint-satisfying
  binary data from an elliptic solver is an extension.
- **No checkpoint and restart** (decided in review). A proof of concept
  runs to completion; checkpointing arrives with TreeAMR's M9 or as a
  production extension. **(Amended 2026-10-01**, Erik's decision: it
  arrived with TreeAMR 0.1.4's M9a, and G5's runs — `38`–`149 h` for
  harmonic `a = 9/10`, eleven hours for step 8′'s crossing — outlast a
  queue's day, so `evolve!` checkpoints and restarts; see [Checkpoint and
  restart](#checkpoint-and-restart).**)**
- **No mesh machinery.** Trees, ghosts, interpolation, regrid transfer
  and reductions are TreeAMR's. The one thing this package writes that
  arguably belongs upstream — point interpolation from a field set, for
  the horizon finder — is written as a stopgap and listed under
  [Upstream prerequisites](#upstream-prerequisites). **(Amended
  2026-09-26:** it went upstream — TreeAMR 0.1.3's M11 — and the stopgap
  is gone; what this package keeps is the order `q + 2` and the footprint
  guard, as a TreeAMR `Region`.**)**
- **No subcycling, one global `dt`**, which suits a black-hole run
  badly in principle (the coarse outer levels are advanced at the
  horizon's time step) and is accepted here for the reasons TreeAMR
  gives: no time interpolation at interfaces, one state vector for the
  integrator, simpler everything.

## Lineage: what is inherited and what changes

From **GHSO2**, verbatim, with the text in `notes/`:

- the state `(h_ab = g_ab − η_ab, Π_ab)` with the densitised
  Lie-advected momentum (`notes/formulation.md`, `notes/methods-ghso2.md`);
- the reduced source `S0`, the Gundlach–Pretorius damping term, the
  gauge-source term, and the cancellation-free offset identities
  (`notes/pointwise-ghso2.jl`);
- prescribed gauge sources `H_a(x)` sampled from the background, with
  their gradients, for backgrounds that are not harmonic;
- the black-hole recipe: `ε_KO ≈ 0.5`, `γ0 ≈ 1/M` near the hole
  (`notes/methods-ghso2.md`, "sonic-surface instability, resolved";
  `notes/sonic-surface.md` for the investigation);
- the horizon analysis: the interpolating `ADMVars` provider for
  `ApparentHorizonFinder`, area, `M_irr`, Korzyński's `J`, `M_ch`;
- the validation ladder: pointwise algebra against `SpacetimeMetrics`
  automatic differentiation, free-stream preservation, convergence on
  the gauge wave, robust stability on noise, constraint monitors,
  stationarity of a black hole;
- the GPU roofline measurements the RHS kernel is compared against
  (`notes/ghaccel-bench.jl`).

From **TreeWave and TreeHydro**: the right-hand-side contract, fixed-step
RK4 (from OrdinaryDiffEq, and from IMEXRungeKutta since 2026-09-26, as in
TreeAMR's own examples), the chunked regrid-and-restart driver with
cases as data, the `observer` hook, `precision.jl` and `device.jl`, the
Löhner refinement indicator with its global noise floor and two
thresholds, the thread-workload digest test, the `[sources]` pin to
TreeAMR's `main`, and the spec-first workflow with measured numbers
recorded in this file.

What **changes** because the mesh changes:

| GHSO2 (SBP-SAT elements) | here (TreeAMR blocks) |
|---|---|
| gradient / divergence as two-pass kernels with face traces | centered stencils reading ghost zones; one fused kernel per RHS |
| flux-conservative `∂_i F^i` for the energy estimate | the expanded form with compact second-derivative stencils (decided) |
| SAT penalties for boundaries; excision faces | ghost-filling boundary hooks; a pointwise damping layer inside the horizon instead of excision (from 2026-10-05 also `:excised`: one-sided closures at a lego excision surface, under study) |
| `ε μ⁻⁵ D⁶` dissipation normalised by the SBP spectral radius | the standard Kreiss–Oliger operator of order `q + 2` on a uniform block |
| one mesh for the whole run | error-driven regridding that follows the hole; a fresh problem per chunk |
| Tsit5 / Vern6–9 matched to the element order; a native stepper | RK4, fixed step (decided): OrdinaryDiffEq's until 2026-09-26, IMEXRungeKutta's by block owner since |
| `Float32`/`Float64`/`Float64x2`, CPU/Metal/CUDA | `Float64` on CUDA as the requirement; the rest inherited from TreeAMR |

## The equations

The vacuum Einstein equations in generalized harmonic form, **second
order in space, first order in time** (decided; `notes/methods-ghso2.md`,
"Formulation", and `notes/formulation.md` for the symmetrizer and the
well-posedness argument). The evolved state per point is 20 fields:

    h_ab = g_ab − η_ab                                    (10)
    Π_ab = (√γ/α)(∂_t − β^i ∂_i) g_ab = √|g| n^μ ∂_μ g_ab (10)

in GHSO2's packed order `(tt, tx, ty, tz, xx, xy, xz, yy, yz, zz)`.
Storing the offset `h` rather than `g` is GHSO2's floating-point hygiene:
every derived quantity (`g^{ab} − η^{ab}`, `det g + 1`, `det γ − 1`,
`α`, `β^i`, `√γ`) is computed as an offset by the identities in
`notes/pointwise-ghso2.jl`, never by subtracting O(1) numbers. TreeAMR
interpolates `h`, which is linear in `g`, so the offset does not
interact with the mesh.

With the gauge source `H_a` and the GH constraint `C_a = Γ_a + H_a`,
GHSO2's system per component is

    ∂_t h_ab = β^i ∂_i h_ab + (α/√γ) Π_ab
    ∂_t Π_ab = ∂_i F^i_ab − α√γ (S0_ab + Z_ab)                    (FLUX)
    F^i_ab   = β^i Π_ab + α√γ γ^{ij} ∂_j h_ab
    S0_ab    = C2sym_ab − 2 Γ2_ab − 2 ∇_(a H_b) − Γ^ν ∂_ν g_ab
    Z_ab     = γ0 [ t_a C_b + t_b C_a − (1 + γ2) g_ab t^c C_c ]

with `C_abc = ∂_a g_bc`, `C2sym_ab = C_a^{μν}C_{μνb} + C_b^{μν}C_{μνa}`,
`Γ2_ab = Γ^γ_{να}Γ^ν_{γβ}`, `2∇_(aH_b) = ∂_aH_b + ∂_bH_a − 2Γ^c_{ab}H_c`,
`t_a = −α δ_a^t`, and the densitisation correction `−Γ^ν ∂_ν g_ab` that
GHSO2's predecessor found by measurement to be O(1) off harmonic gauge
(`notes/formulation.md`, §2). The time derivative `∂_t g_ab` inside `S0`
is the first equation.

**The expanded form** (decided). On a finite-difference mesh the
divergence in `(FLUX)` is a derivative of a computed quantity: it costs
either a second ghost exchange of `F^i` or ghost zones twice as wide, so
that `F^i` can be formed `q/2` points into the ghosts before the
divergence reads it. The product rule removes both:

    ∂_t Π_ab = β^i ∂_i Π_ab + (∂_i β^i) Π_ab
             + α√γ γ^{ij} ∂_i ∂_j h_ab + ∂_i(α√γ γ^{ij}) ∂_j h_ab
             − α√γ (S0_ab + Z_ab)                                  (EXPANDED)

where `∂_i β^i` and `∂_i(α√γ γ^{ij})` follow from `∂_i h` by the chain
rule through the algebraic map `h ↦ (α, β, √γ, γ^{ij})`, in closed form:
`∂_i g^{ab} = −g^{ac} ∂_i g_cd g^{db}`, then `∂_i α = ½ α³ ∂_i g^{tt}`,
`∂_i β^j = −∂_i(g^{tj}/g^{tt})`, `∂_i γ^{jk}` from
`γ^{jk} = g^{jk} − g^{tj}g^{tk}/g^{tt}`, and `∂_i √γ = ½ √γ γ^{jk} ∂_i γ_jk`.
A forward-mode dual pass through `metric_quantities` is the independent
check the tests use. `(EXPANDED)` reads `∂_i h`, `∂_i Π` and `∂_i∂_j h`
at the point, all with compact centered stencils of one half-width, so
the ghost width is set by the dissipation operator and not by a
derivative of a derivative.

`(FLUX)` and `(EXPANDED)` are the same equation and differ only in what
is discretised; the flux form is not implemented on the mesh. It stays
in the pointwise tests: GHSO2's identity `∂_tΠ − ∂_iF^i = msrc`, evaluated
on analytic data, is what validates the pointwise algebra, and the
expanded form's coefficients are checked against a finite difference of
the analytic `F^i`. The energy-estimate argument for the flux form was
the reason GHSO2 chose it, and it does not carry over: with overlapping
stencils through interpolated ghosts there is no discrete
summation-by-parts identity across block or level boundaries to
preserve, so the flux form would cost ghosts and buy nothing this mesh
can use (decided in review).

**(Measured in step 1.)** Both hold. On the six backgrounds of the table
at two points each, in `Float64`, the residual of `∂_tΠ − ∂_iF^i − msrc`
on analytic data is between `2e−11` and `4e−8` relative to the size of
the terms that make it up, and it falls by a factor of **16** when the
fourth-order difference's step is halved: the residual *is* that
difference's truncation error, and the ported source is the vacuum
Einstein equation in this gauge. (Minkowski is exactly zero and the gauge
wave sits at roundoff, `7e−15`, at every step: it depends on `x − t`
alone, so the temporal and the spatial truncation errors cancel.)
`(EXPANDED)`'s `∂_tΠ` and `(FLUX)`'s agree to **under one `eps`** of the
size of the terms when the flux's divergence is taken exactly by
automatic differentiation rather than by a stencil, on every background
and at both `Float64` and `Float32` — the expansion is the same equation
to the last bit. The closed-form coefficient derivatives agree with a
forward-mode dual pass through `metric_quantities` to **0.6 `eps`** of
`max_i ‖∂_i h‖`, and `C_a` and `Z_ab` vanish on exact data to **1.5 `eps`**
and **0.3 `eps`** of the terms that build them, with `H_a` and `∂_aH_b`
under **2.4 `eps`** on each of the four harmonic rows.

The shapes follow, and step 3's kernel is written against them
**(proposed in step 1**, since `PLAN.md` fixed the argument list but not
its spelling**)**. `metric_derivatives` returns the *full*
`(∂_iα, ∂_iβ^j, ∂_i(α√γγ^{jk}))` rather than the four contractions
`(EXPANDED)` needs, because that is what a dual pass can be compared
against; the caller contracts, and the components it does not ask for
leave the inlined code. `∂h` and `∂Π` are `NTuple{3,SVector{10}}`, and
`∂∂h` is an `NTuple{6,SVector{10}}` in the column-major lower-triangular
order `(xx, xy, xz, yy, yz, zz)` — the packing of a symmetric tensor, one
index range shorter, so that the file has one packing convention and not
two. Two functions take the **coefficient set** rather than `h`, because
the kernel forms it once per point in step 1 of the order above and
rebuilding it would buy a second `inv`, determinant and square root:
`metric_derivatives(gu4, α, β, γu, √γ, ∂h)`, whose `(h, ∂h)` method is a
wrapper for the tests and the host-side diagnostics — the two agree to
roundoff and, as it turns out, not bit for bit, for the reason under
[Measured results](#measured-results) — and
`gh_node_source(g4, gu4, α, √γ, dg, H, ∂H, γ0, γ2)`, which the kernel
reaches with the coefficients two stages old.

`gh_node_source` is a second **copy** of the reduced source and the
damping, not a factoring-out: `gh_node_rhs` keeps its own, so that the
ported function stays diffable against `notes/pointwise-ghso2.jl` and goes
on being the validated reference the flux identity runs through. The price
is that the two can drift, and `test/pointwise_identity_tests.jl` is what
notices — to roundoff rather than bit for bit, for the reason under
[Measured results](#measured-results).

## Discretization

### Field sets and layout

One forest, vertex-centered field sets **(decided** for a
finite-difference scheme: TreeWave's argument — restriction along a
stagger is injection, exact for any data — applies unchanged**)**:

| set | `nvars` | `G` | state vector | at a regrid |
|---|---|---|---|---|
| `U`, the state `(h, Π)` | `20` | `q/2 + 1` | yes | `U => schedule`, `PointValue`, prolongation `p = q + 2` |
| `Hsrc`, gauge source `H_b` and `∂_a H_b`, non-harmonic backgrounds only | `20` | `0` | no | `Hsrc => nothing`, then re-sampled: a function of position |
| `diag`, constraints, speeds, masked errors, the indicator, on demand | small | `0` | no | `diag => nothing` |

`diag` holds **ten** slots from step 4 **(proposed in step 4**, which is
where the set first had to be written down**)**: the characteristic speed,
the four components of `C_a`, the ADM Hamiltonian, the three components of
`ℳ_i`, and the `1`/`0` mask indicator the masked norms divide by. `C_a`
and `ℳ_i` occupy contiguous runs on purpose — `block_mapreduce` reduces a
contiguous range of variables and a device cannot be handed an arbitrary
index vector cell by cell. Steps 5 and 6 add the masked error, the
interior residual and the indicator beside them.

The field sets are **vertex-centered and `GHProblem` refuses anything
else** (added in step 4). The table above already decided it; what made it
an assertion is that the masks and the interior profiles turn an owned
index into a position — `point_position`, written to reproduce TreeAMR's
`coordinates` expression bit for bit — and a staggered set would be
evaluated half a cell from where its values sit.

`q` is the finite-difference order (below), and `G = q/2 + 1` because
the Kreiss–Oliger operator of order `q + 2` reaches one point further
than the derivatives do. TreeAMR's vertex invariant `N ≥ 2G + 2` then
puts `N ≥ 8, 10, 12` at `q = 4, 6, 8`; the tests use `N = 8` where that
allows it and `N = 10` at `q = 6` **(amended in step 3**, where the
invariant bit: `FieldSet` refuses `N = 8` at `q = 6`, so `PLAN.md`'s "the
gauge wave at `q = 2, 4, 6` on `N = 8`" was one row wider than the mesh
permits**)**, the proof-of-concept runs `N = 16` or `32`. The working array is
`((N + 2G + 1)/N)^3` times the state: 2.6 at `N = 16, G = 3`, 1.7 at
`N = 32`, which is the price of wide ghosts on small blocks and the
reason `N = 32` is expected to be the default **(predicted; G6 measures
it against the RHS throughput)**.

### Finite-difference stencils

Centered stencils of even order `q ∈ {2, 4, 6, 8}` as a `Val` parameter
of the kernel, for `∂_i`, `∂_i∂_i` (compact) and `∂_i∂_j` (the tensor
product of two first derivatives, which reads the edge and corner ghosts
TreeAMR fills unconditionally). Weights are built in exact rational
arithmetic and rounded once into `T`, as TreeAMR builds its interpolation
weights, so a stencil is the same object at every precision. `q = 4` is
the development and test default; `6` and `8` exist because black-hole
production codes use them, and G6 measures what they cost on this mesh
(the interface rule below makes high `q` expensive in a way it is not on
a unigrid).

`derivative_weights(T, Val(q), Val(m))` returns the weights for **unit
spacing**, in the offset order `−q/2 … q/2`; the `1/h^m` is the caller's,
read once per point as a per-block coefficient **(decided in step 2**,
which is where the formula had to be split between the weights and the
kernel; `CODE.md` had said only that the weights exist**)**. One weight
vector therefore serves every refinement level. Both are `@generated`, so
the `Rational{BigInt}` construction happens while the method compiles and
what it emits is each weight's exact numerator and denominator as `Int`
literals with one division between them — an `SVector` a device kernel
holds in registers, with no rational and no `BigInt` anywhere in it, and at
`Float64` and `Float32` no division either once LLVM has folded it. The
conversion is emitted rather than performed in the generator for a reason
that cost step 2 a debugging session; see [Measured
results](#measured-results).

The mixed derivative has no weights of its own — it is the product of two
first-derivative vectors, and the sum over the *second* axis is the inner
one **(decided in step 2**: the two orders are equal in exact arithmetic
and differ in the last place in floating point, so which one it is belongs
to the operator and not to the implementation**)**. Measured in step 2 on
`exp(sin x)·cos(y/2)`: the tensor product converges at the same rate as
the one-dimensional first derivative, 2.0, 3.98, 5.95, 7.79 at `q = 2, 4,
6, 8`.

### Kreiss–Oliger dissipation

The standard operator of order `2r = q + 2`, per dimension, on all 20
fields:

    Q_d u = ε (−1)^{r+1} (h_d^{2r−1} / 2^{2r}) (D_+ D_−)^r u,   r = q/2 + 1

with `h_d` the block's own spacing, so that a refinement level's
dissipation scales with its resolution and `ε ∈ (0, 1)` is neutral to
the CFL condition. It is added inside the RHS kernel, read from the
same ghosted working array, and has no separate launch.

`dissipation_weights(T, Val(r))` carries `(−1)^{r+1}`, the `2^{−2r}` and
the undivided `(Δ_+Δ_−)^r`; the caller applies `ε` and a **single**
`1/h_d`, which is what `h_d^{2r−1}` against `(Δ_+Δ_−)^r`'s own `h_d^{−2r}`
leaves **(decided in step 2**, the same split as the derivative weights
above**)**. So `Q_d u = (ε/h_d) · (the contraction)`, and two consequences
are measured rather than asserted (step 2, in `Rational`, exactly): the
center weight is `−binom(2r, r)/2^{2r} < 0` and the grid-scale mode
`u_j = (−1)^j` is an eigenvector with eigenvalue exactly `−1`, so
**the sign is damping** and Nyquist is damped at exactly `ε/h_d`. On a
periodic grid the operator is negative semidefinite. `Q_d` annihilates
polynomials of degree `< 2r` and is `O(h^{2r−1}) = O(h^{q+1})` on smooth
data, one order better than the scheme, which is why it does not tighten
the interface-order rule below. Its role is
GHSO2's second finding under "sonic-surface instability"
(`notes/methods-ghso2.md`): the grid-scale layer of that instability on
a black hole whose horizon lies in the evolved domain is cured by
dissipation at `ε ≈ 0.5`; smooth runs without a hole need little or
none. `ε` is a case parameter — a number, or from step 8c a profile of the
distance to the hole that rises from the horizon inward
(`HorizonDissipation`; measured under [The
interior](#the-interior-a-pointwise-damping-layer), where it is off by
default).

**(Measured in step 3.)** Both halves hold on the mesh. The gauge wave at
`q = 4` converges at **4.12** with `ε_KO = 0.5` against **3.95** without
it, so the term is `O(h^{q+1})` as claimed and costs the scheme nothing;
and white noise of amplitude `1e−8` on flat space, over a thousand steps
— eighteen crossings of the box — falls to **0.66** of its initial L2
norm with `ε_KO = 0.5` and **grows by 9.2** without it, in L∞ by **30.5**.
The kernel compiles the dissipation away entirely when `ε_KO = 0`
(`Val(false)`), which is worth about a fifth of an evaluation
(1106 against 1409 ns per point at `q = 4`), and a run with
`ε_KO = 1e−300` gives the same numbers as one with the term compiled out.

**What the dissipation has to do inside a horizon (measured in step 8a,
`test/dispersion.jl`).** The discrete scheme is not causal at the grid
scale, and the dissipation is the only thing that stands in for causality
there. Frozen at a point, with the lower-order terms dropped, the principal
part of every component along a direction `x` is

    (∂_t − b ∂_x)² u = a² ∂_x² u,      b = β^x,   a = α √γ^{xx},

**with the advection's sign the code's**, `∂_t h = +β^i ∂_i h`, so that the
characteristic speeds are `−b ± a` and both are negative inside the horizon
— for `KerrSchild(1, 0)` along a radius `b = H/(1+H)` and `a = 1/(1+H)`
with `H = 2M/r`, so `b = 0.571`, `a = 0.429` at `r = 1.5 M`, which the
script reads from the metric through `metric_quantities` and checks
**(amended in step 8a**: `PLAN.md`'s finding 4 writes the operator as
`(∂_t + b ∂_x)²` with `b = β^r` and the speeds `−b ± a`, and the three are
consistent only with the minus sign**)**. With the package's own weights on
unit spacing, a mode `e^{i(kx − ωt)}`, `θ = kh`, has the two branches

    ω = (−b s(θ) ∓ a √c(θ))/h − i ε sin^{2r}(θ/2)/h,
    s(θ) = Σ_j w₁_j sin jθ,   c(θ) = −Σ_j w₂_j cos jθ,

the symbols of `D₁` and of the compact `D₂`; the script forms them as the
eigenvalues of the 2×2 symbol, differentiates in `θ` for the group velocity
`v_g`, and checks both against the closed forms. Branch 1 is the fast
ingoing one (`v_g → −b − a`); branch 2 (`v_g → a − b`) is the one the
horizon is about. Two things the continuum does not have:

- **The Nyquist mode moves outward.** Every centered `D₁` annihilates it
  (`s(π) = 0`) with the slope `s′(π) = −1, −5/3, −11/5, −93/35` at
  `q = 2, 4, 6, 8`, and `(√c)′(π) = 0`, so at `θ = π` *both* branches move
  at `−b s′(π)`: `+b`, `+5b/3`, `+11b/5` — outward, faster the higher the
  order. `test/stencils_tests.jl` asserts the slopes in `Rational`, beside
  the damping-sign claim above.
- **The intermediate wavelengths are the worst.** Branch 2 turns outgoing
  at `θ_c`, where `a (√c)′ = b s′` (at `q = 2`, `a cos(θ/2) = b cos θ`),
  far below Nyquist, where `sin^{2r}(θ/2)` is small; the penetration length
  `ℓ(θ) = max(v_g, 0)/σ`, in cells per e-fold, peaks just above `θ_c`.

`v_g` does not depend on `ε` and `σ` is proportional to it, so **`ℓ ∝ 1/ε`
exactly** and the table is at `ε_KO = 1` — divide by `ε` for any other; at
`ε_KO = 0` every outgoing mode has `ℓ = ∞`. `ℓ_max` in cells along a grid
axis, with the `θ/π` and the `v_g` it is attained at, and `θ_c/π`:

| `r/M` | `b/a` | `q = 2` | `q = 4` | `q = 6` | `θ_c/π` at `q = 2, 4, 6` |
|---|---|---|---|---|---|
| 1.0 | 2.000 | 0.96 (0.540, 0.304) | 1.32 (0.679, 0.594) | 1.64 (0.749, 0.864) | 0.362, 0.471, 0.534 |
| 1.2 | 1.667 | 1.07 (0.478, 0.231) | 1.35 (0.628, 0.454) | 1.63 (0.704, 0.664) | 0.325, 0.443, 0.510 |
| 1.5 | 1.333 | 1.45 (0.374, 0.136) | 1.51 (0.539, 0.267) | 1.70 (0.626, 0.394) | 0.258, 0.389, 0.464 |
| 1.8 | 1.111 | 3.10 (0.234, 0.052) | 2.13 (0.411, 0.102) | 2.10 (0.511, 0.151) | 0.164, 0.304, 0.389 |
| 2.0 | 1.000 | ∞ (`θ → 0`) | ∞ (`θ → 0`) | ∞ (`θ → 0`) | 0 |

and the attenuation over the default margin at the default `ε_KO = 1/2`,
`e^{−8/ℓ_max} = e^{−4/ℓ_max(1)}`, **if the whole margin had the
coefficients of that radius**:

| `r/M` | 1.0 | 1.2 | 1.5 | 1.8 | 2.0 |
|---|---|---|---|---|---|
| `q = 2` | 1.5e−2 | 2.4e−2 | 6.3e−2 | 0.28 | 1 |
| `q = 4` | 4.8e−2 | 5.2e−2 | 7.1e−2 | 0.15 | 1 |
| `q = 6` | 8.7e−2 | 8.6e−2 | 9.6e−2 | 0.15 | 1 |

Four things read off it. **At the horizon the supremum is at `θ → 0`**:
branch 2's continuum speed `a − b` vanishes there, the discrete one's
outward error is `O(θ^q)` against a dissipation of `O(θ^{q+2})`, and near
the sonic surface the smooth outgoing modes are barely damped — the
continuum's marginal trapping, which no dissipation of this form can
touch. Close to it, with `δ = b/a − 1`, the script's numbers follow
`ℓ_max ≈ C_q / (ε δ^{2/q})` with `C_2 = 0.28`, `C_4 = 0.64`, `C_6 = 0.93`
(fitted at `r = 1.9 … 1.99 M`, `δ = 0.053 … 0.005`; `C_2 = 72/256` is also
what the leading-order expansion of `v_g/σ` gives), so `ℓ` is finite only
some depth below the horizon, and **a margin measured in cells buys less at
finer `h`**, where it lies closer to the horizon. **Higher order helps near
the horizon and not deep inside**: `δ^{−1/2}` and `δ^{−1/3}` diverge more
slowly than `δ^{−1}`, so at `r = 1.8 M` `q = 4` and `6` have `2.13` and
`2.10` against `q = 2`'s `3.10`; deep inside (`r = M`) `ℓ_max` grows with
`q`, because the peak moves toward Nyquist, where the dissipation is
stronger, but `v_g` rises as much. **The axis is the direction to design
for**: along the grid diagonal, where the principal part reads `D₁ ⊗ D₁`
and the dissipation acts on all three axes at a third of the phase each,
`ℓ_max` is shorter at every entry (`0.63, 0.75, 1.14, 2.69` at `q = 2`,
`0.74, 0.80, 0.97, 1.47` at `q = 4`, `0.88, 0.91, 1.02, 1.34` at `q = 6`, for
`r = 1.0 … 1.8` at `ε_KO = 1`). And **the time integrator adds nothing**:
the fully discrete scheme, RK4 at `cfl = 1/4` on the fixture's
`λ_max = 1.671`, agrees with every semi-discrete entry off the horizon to
the digits printed (at `r = 2 M`, where both are unbounded as `θ → 0`, its
supremum over the `θ` grid is over a thousand cells), and at `ε_KO = 0`
RK4's own damping, `O((ω dt)⁶)`, leaves `ℓ > 10⁶` cells. What this means
for the margin `m` is under [The
interior](#the-interior-a-pointwise-damping-layer), and what a 3D run does
with it is under [step
8a](SINGULARITY_HANDLING.md#step-8a-what-crosses-the-horizon-from-inside-it).

**The spinning holes (measured in step 8a, `test/dispersion.jl` sections 1b
and 1c).** Step 8d keys the layer on the found horizon's offset surface
`r_1(n̂) = r_h(n̂) − m h`, and `PLAN.md`'s finding 3 puts harmonic Kerr at
`a = 9/10` at `h = 5/256`, where its equator leaves `0.1 M` — 5.1 cells —
between the singular disk and the horizon. The same numbers along the spin
axis and along the equator of both spinning charts — grid axes both, and
the two directions in which the horizon's normal is radial by symmetry, so
the sonic point `b = a` *is* the horizon (`g^{nn} = γ^{nn} − (β^n)²/α² = 0`;
the script finds it by bisection within `5e−16` of the analytic radius in
all four), at depths `d = 2, 4, 8` cells of `h = 5/256`. `g_h` is
`d(b/a)/d(depth)` at the horizon, which Kerr-Schild at `a = 0` has at
`1/(2M)`; `ℓ_max` is at `ε_KO = 1` with its `θ/π`:

| case, `r_h`, `g_h` | `d` | `r/M` | `b` | `a` | `b/a` | `q = 2` | `q = 4` | `q = 6` |
|---|---|---|---|---|---|---|---|---|
| `Harmonic(1, 9/10)`, axis, `0.4359`, `1.00/M` | 2 | 0.3968 | 0.1499 | 0.1441 | 1.0401 | 2.23 (0.145) | 0.97 (0.316) | 0.82 (0.422) |
| | 4 | 0.3578 | 0.1477 | 0.1364 | 1.0823 | 1.13 (0.204) | 0.68 (0.381) | 0.64 (0.483) |
| | 8 | 0.2796 | 0.1423 | 0.1212 | 1.1734 | 0.58 (0.285) | 0.48 (0.460) | 0.50 (0.556) |
| `Harmonic(1, 9/10)`, equator, `1.0000`, `2.20/M` | 2 | 0.9609 | 0.0535 | 0.0486 | 1.1008 | 0.34 (0.224) | 0.23 (0.401) | 0.22 (0.502) |
| | 4 | 0.9219 | 0.0304 | 0.0241 | 1.2618 | 0.09 (0.339) | 0.09 (0.509) | 0.10 (0.600) |
| | 8 | 0.8438 | — | — | — | on the singular disk | | |
| `KerrSchild(1, 9/10)`, axis, `1.4359`, `0.30/M` | 2 | 1.3968 | 0.5029 | 0.4971 | 1.0118 | 24.42 (0.079) | 5.92 (0.230) | 4.10 (0.337) |
| | 4 | 1.3578 | 0.5058 | 0.4942 | 1.0234 | 12.59 (0.111) | 4.25 (0.275) | 3.29 (0.382) |
| | 8 | 1.2796 | 0.5112 | 0.4888 | 1.0457 | 6.71 (0.154) | 3.11 (0.327) | 2.68 (0.433) |
| `KerrSchild(1, 9/10)`, equator, `1.6946`, `0.31/M` | 2 | 1.6556 | 0.4952 | 0.4894 | 1.0119 | 23.88 (0.080) | 5.81 (0.231) | 4.03 (0.337) |
| | 4 | 1.6165 | 0.4970 | 0.4857 | 1.0234 | 12.38 (0.111) | 4.18 (0.275) | 3.23 (0.382) |
| | 8 | 1.5384 | 0.4994 | 0.4781 | 1.0447 | 6.69 (0.152) | 3.07 (0.325) | 2.63 (0.431) |
| `KerrSchild(1, 0)`, reference, `2.0000`, `0.50/M` | 2 | 1.9609 | 0.5049 | 0.4951 | 1.0199 | 14.67 (0.103) | 4.59 (0.264) | 3.46 (0.371) |
| | 4 | 1.9219 | 0.5100 | 0.4900 | 1.0407 | 7.47 (0.146) | 3.28 (0.317) | 2.77 (0.423) |
| | 8 | 1.8438 | 0.5203 | 0.4797 | 1.0847 | 3.88 (0.206) | 2.38 (0.384) | 2.25 (0.486) |

and the e-folds `n_e = ∫_{r_h − m h}^{r_h} dr/(h ℓ_max(r))` that a margin of
`m` cells buys at `ε_KO = 1/2` — the path form of the leakage margin under
[The interior](#the-interior-a-pointwise-damping-layer), with the
least-attenuated mode at each radius, so a lower bound on what any packet
gets:

| case, `h` | `q = 2`, `m = 4` | `m = 5` | `m = 8` | `q = 4`, `m = 4` | `m = 5` | `m = 8` |
|---|---|---|---|---|---|---|
| `Harmonic(1, 9/10)`, axis, `5/256` | 0.89 | 1.39 | 3.51 | 1.94 | 2.72 | 5.52 |
| `Harmonic(1, 9/10)`, equator, `5/256` | 7.29 | 17.9 | — (disk at 5.12 cells) | 9.74 | 19.8 | — |
| `KerrSchild(1, 9/10)`, axis, `5/256` | 0.08 | 0.13 | 0.31 | 0.32 | 0.44 | 0.88 |
| `KerrSchild(1, 9/10)`, equator, `5/256` | 0.08 | 0.13 | 0.32 | 0.32 | 0.45 | 0.89 |
| `KerrSchild(1, 0)`, `5/256` | 0.14 | 0.21 | 0.53 | 0.41 | 0.57 | 1.14 |
| `KerrSchild(1, 0)`, `5/64` (the fixture's) | 0.50 | 0.77 | 1.81 | 0.78 | 1.08 | 2.07 |

Four things read off them. **`b/a` is not Kerr-Schild's anywhere but in
Kerr-Schild**: on the harmonic equator it leaves the horizon at `2.2/M`
and averages `3.4/M` over the first four cells — nearly seven times
Kerr-Schild's `1/(2M)` — while both speeds fall toward zero at the disk
(`a = 0.069` at the horizon, `0.024` four cells in, `0.006` at five).
**`ℓ` scales with `a`** at fixed `b/a`, so the slow harmonic chart is
strongly damped per cell:
`ℓ_max` is below a cell four cells inside the equatorial horizon, and even
on the harmonic axis, where `a ≈ 0.15`, it is a third of Kerr-Schild's at
the same `b/a`. **The frozen-coefficient model is at the edge of its
validity on that equator**: `a` falls by a factor three across the first
four cells below the horizon and by four across the fifth, on the scale of
the grid-scale wavelengths themselves, so the `m = 4` number is an
indication and the `m = 5` one is not to be trusted. And **Kerr-Schild at `a = 9/10`
is the hard case**, not the harmonic chart: with `g_h = 0.30/M`, `b/a`
takes three times as many cells as on the harmonic axis to move off the
horizon's value, at `a ≈ 1/2` rather than `0.15` each of those cells damps
a third as much, and a margin of eight cells at `h = 5/256` buys a third of
an e-fold at `q = 2` and `0.9` at `q = 4`. Against the fixture's
measurement the bound is conservative: at
`h = 5/64` it gives `1.81` e-folds across eight cells at `q = 2` where the
3D runs measured `4.6` at `2 M` and the one-dimensional model `2.9` in the
long run.

### The interface-order rule, and what it costs a second-order system

TreeAMR's measured rule (its `CODE.md`, "Operators"): a ghost filled by
an order-`p` operator carries an `O(h^p)` error, an `m`-th derivative
divides it by `h^m`, and the global rate is `min(q, p − m + 1)`. This
system takes **second** derivatives, so `m = 2` and `p = q + 2` is the
first order that does not degrade the scheme: `p = 6` at `q = 4`.
Restriction along a vertex-like dimension is injection and has no
order. The dissipation term is `h^{2r−1} ∂^{2r}` and contributes
`O(h^{p−1})`, like a first derivative, so it does not tighten this.
Prolongation needs `G ≥ p/2 − 1 = q/2`, which the dissipation width
already exceeds.

The cost is the stencil: a tensor-product prolongation at `p = 6` reads
`6³ = 216` coarse points per fine ghost point, at `p = 10` (for
`q = 8`) a thousand. It applies only at coarse-fine faces, and TreeAMR
batches it into one kernel per stencil kind, but it is a real
difference from a unigrid code and one reason to expect `q = 4` or `6`
to be the order of choice rather than `8` **(predicted; G6)**.

**(Measured in step 4.)** The prediction holds, on the gauge wave
(`A = 1/20`, one wavelength cubed, periodic) at `q = 4` on the two-level
mesh — a `2³` root grid with **one root block refined**, so that each of
its six faces is a coarse-fine face — held **frozen** while `N` runs over
`8, 10, 12` (15 leaves at every resolution, every spacing shrinking with
`N` and the block layout unchanged), a sixteenth of a crossing at
`cfl = 1/4`. Both the mesh and the
resolutions are a budget **(proposed in step 4)**: TreeWave's `wave_forest`
refines a middle sub-box, which at `roots = 2` is every block or none, and
a fourth resolution or a longer run is `N⁴` of ghost filling at 216 coarse
points per fine ghost point. What is run measures the rate cleanly in both
norms and costs about a minute and a half at one thread.

| prolongation | restriction | `ε_KO` | L2 rate | L∞ rate |
|---|---|---|---|---|
| 4 | 2 or 4 | 0 | **3.18** | **3.25** |
| 6 | 2 or 4 | 0 | **3.98** | **3.94** |
| 4 | 2 or 4 | 0.5 | **3.26** | **3.07** |
| 6 | 2 or 4 | 0.5 | **4.08** | **4.09** |
| unrefined control, 4 or 6 | any | 0 | **3.98** | **3.97** |

So `p = q + 2` is a requirement and not a choice: an order-4 ghost costs
this system a whole order, and the dissipation does not change that —
`Q_d` contributes `O(h^{p−1})` like a first derivative, exactly as the
section above says. The **restriction order does not appear** because on
a vertex-centered mesh restriction is injection: the `p = 2` and `p = 4`
rows are not merely equal, they are the same computation, and
`test/interface_tests.jl` asserts `l2 === l2` rather than a tolerance
(TreeWave's finding, confirmed here). The control's two rows are the same
computation for the stronger reason that an unrefined mesh never
prolongates at all.

What the interface costs in *time* is larger than what it costs in order,
and it is TreeAMR's cost rather than this package's. At `N = 12`, `q = 4`,
`Float64`, one thread, on the two-level mesh: a right-hand-side evaluation
is **5211 ns** per owned point at `p = 6` and **2521 ns** at `p = 4`,
against **1399 ns** on the unrefined mesh — and the ghost fill alone is
**79 %** of the evaluation at `p = 6`, **56 %** at `p = 4` and **22 %**
unrefined. The 216 coarse points per fine ghost point are real. This is
why `test/interface_tests.jl` is the most expensive file in the suite and
why its runs are as short as a clean rate allows.

### One right-hand-side evaluation

TreeAMR's contract, three steps and nothing between them (decided):

    scatter!(U, u)                                             # (0) state → working array
    fill_ghosts!(U, schedule; boundary = dirichlet(case, t))   # (1) copies, restrictions, hook, prolongations
    map_blocks!(gh_rhs_kernel!, U, statearray(du, U), U.work, Hsrc.work, …)   # (2)

Step (2) is **one fused kernel** per owned point **(proposed)**, and
its internal order is a design commitment, because it is what decides
whether the kernel fits in a GPU's registers. Written naively — load
the 20 fields on the footprint, form all of `∂_i h` (30), `∂_i Π` (30),
`∂_i∂_j h` (60) and the dissipation, then run the algebra — the kernel
holds about 140 `Float64` values before the algebra starts, which is
already over the 255 32-bit registers a thread has, and it spills.
GHSO2's two-pass alternative — derivatives into scratch, then a
pointwise kernel — would cost a scratch field set of 120 values per
point, six times the state's memory traffic. The kernel is therefore
written in **streaming order**:

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

The principal part is the same scalar wave operator for every
component, which is what makes step 2 a loop with a small live set;
the coupling between components is entirely in the coefficients and
the source. Recompute rather than store is the rule throughout: the
effective stencil weights (metric coefficient times finite-difference
weight) are formed where they are used, and nothing that can be
rebuilt from `h` and `∂_i h` is kept. The pointwise algebra is
scalarised straight-line code — GHAccel found that the compiler does
not register-allocate a large `SVector` expression well unless every
element is spelled out, and generated its kernel for that reason; this
package does the same, by hand or by generation, and the choice is an
implementation detail of G1. **(Decided in step 1: by hand.)** The
expanded form is written as elementwise `SVector{10}` combinations of the
stencils and the coefficients, which `StaticArrays` unrolls into scalar
arithmetic with no loop and no branch; `test/pointwise_tests.jl` asserts
that every pointwise function allocates **zero bytes**, which is the part
of "kernel-safe" a host test can settle. Whether that unrolling is also
the register schedule GHAccel needed is a question about `ptxas` output
and belongs to G6, which measures it; generating the code is the fallback
the milestone leaves open, not a thing to do before there is a
measurement.

This is the whole of the design's commitment to GPU kernel efficiency.
Beyond it, the **black-box attempts** G6 makes and records are: the
split into a stencil kernel (steps 1–2, `du` written) and an algebra
kernel (step 3, `du` accumulated), costing one extra read and write of
`du` and a recomputation of `∂_i h`; `Float32` where a device has it,
which halves the register footprint; and the launch configuration
(workgroup shape), which GHAccel measured as worth 10 % of peak.
Everything further — shared-memory staging of one variable's ghosted
block at a time, fusing the scatter, the ghost fill or the integrator's
stage update into the kernel, code generation for the register schedule
— is the research project under
[Possible extensions](#possible-extensions), and nothing here is built
in a way that would prevent it. `G`, `q` and the switches (dissipation,
gauge source present or not, an interior present or not) are `Val`
parameters, resolved once per chunk in the problem constructor, never
per evaluation. **(Amended in step 5:** the interior's `Val` carries the
*variant* rather than a `Bool` — `:none`, `:damped`, `:pasted`,
`:frozen`. `PLAN.md` calls it "has interior"; it has to say *which* as
well, because the three variants differ inside the kernel, so a `Bool`
would have needed a second parameter beside it.**)**

**(Measured on the H200 2026-10-05**, in [The right-hand side on an
H200](#the-right-hand-side-on-an-h200-measured-2026-10-05)**.)**

- The black-box attempts are worth little: the workgroup shape under 10 %; the
  split 6 % once the source fits in registers.
- What the kernel needed was inlining forced on the device and a register-lean
  spelling of the source: 8.5 → 1.2 ns a point.
- The order above changes once the source is lean: the source moves before the Π
  components (step 3 becomes step 2), and the per-component streaming stays.

**(Implemented 2026-10-05.)** The kernel's order is now:

1. **The head.** Load `h` and `Π` at the point, form the thirty `∂_i h` and the
   coefficient set.
2. **`∂ₜh`, complete with its dissipation,** stored at once.
3. **The source**, [`gh_node_source_lean`](#the-right-hand-side-on-an-h200-measured-2026-10-05),
   from `∂ₜh`.
4. **`∂_i β^i` and `∂_i(α√γγ^{ij})`** from `metric_divergences`, which forms only
   those four numbers.
5. **A run-time loop over the ten Π components.** Each forms its nine stencils of
   `h_v` (`∂_i h_v` again, from cached loads, rather than thirty values kept live
   across the source) and its seven of `Π_v`, adds its source component and is
   stored.

In the code, steps 1–4 are `gh_rhs_head`, step 5's component is `gh_rhs_pi` and
the store is `gh_rhs_store!`, all in `src/evolution.jl`. The interior variants
combine `F` with the layer before storing, so they take it as two vectors from
`gh_rhs_at_point`: the same two functions, with the components unrolled.

Nothing in the kernel without an interior forms a closure:

- its loops are `Base.Cartesian.@ntuple` and `@nexprs`;
- the stencils are `@generated` with an explicit `:inline` meta;
- the gauge sources are spelled out.

So a device compiles it without a single call into Julia code whether or not the
backend forces inlining. Measured: 8.48 → 1.08 ns a point on the H200; the
numbers are under [The right-hand side on an
H200](#the-right-hand-side-on-an-h200-measured-2026-10-05).

The kernel is *block-local*: it reads its own block's stored points and
nothing else, so it runs on every backend unchanged. Per-block spacings
and origins travel to the backend once per chunk, as TreeWave's spacings
do; the origins are new here, because the interior profiles, the masks
and the damping profile need positions inside the kernel.

The RHS never mutates `u`, and it is a pure function of `(u, t)`
(TreeAMR's contract): the interior treatment below is a term *in* the
right-hand side, not a write to the state.

**(Implemented and measured in step 3.)** The kernel is the order above,
written by hand: `h`, `Π` and the thirty `∂_i h` at the point; the
coefficient set and its two contractions `∂_iβ^i`, `∂_i(α√γγ^{ij})`;
then `ntuple(Val(10))` over the components, each forming three `∂_iΠ`,
three compact and three tensor-product second derivatives and, where
there is dissipation, six more contractions — all consumed into two
accumulators before the next component starts; then `gh_node_source`. It agrees with
`gh_node_rhs_expanded` on analytic data, on three backgrounds and at
`q = 2, 4, 6` with and without dissipation, to **under 1e−12** of the size
of the terms, and Minkowski's `du` is **exactly** zero at every order.

Two things the writing settled that the design had left open:

- **The `∂_t g` the source is given is the accumulated `∂ₜh`, dissipation
  included** (recorded in step 3). Two accumulators per component is what
  the order above budgets, and keeping an undissipated copy would be ten
  more live values; it is also the honest answer, since the reduced
  source's `−Γ^ν ∂_ν g_ab` means the time derivative of the solution
  being evolved, which is the one with `Q_d` in it. The difference is
  `O(h^{q+1})`, the dissipation's own order, and the convergence rates
  above are measured with it.
- **The stencils address the working array by a linear index** — a base
  index per point and one stride per axis — rather than by an
  `(i, j, k, v, b)` tuple per load. The array is dense and column-major,
  so the two name the same element and the output is bit-identical; the
  reason is cost. Measured at `q = 4` in `Float64` on one thread: the
  stencil half of the kernel is **1376 ns** per point with the cartesian
  index and **573 ns** with the linear one, and the whole right-hand side
  went from 2770 to 2117 ns. Five-dimensional index arithmetic at every
  one of the ~1600 loads a point takes is not a cost the compiler removes.

What an evaluation costs, per owned point, on the development machine
(Apple silicon, 12 CPU threads, `Float64`, `N` at the vertex invariant,
`ε_KO = 0.5`, one gauge-source-free case) — the kernel alone, and the
whole of `gh_rhs!` with the scatter and the ghost fill around it:

| `q` | kernel, 1 thread | `gh_rhs!`, 1 thread | kernel, 4 threads | `gh_rhs!`, 4 threads |
|---|---|---|---|---|
| 2 | 617 ns | 1032 ns | 173 ns | 290 ns |
| 4 | 1244 ns | 1860 ns | 332 ns | 508 ns |
| 6 | 1611 ns | 2208 ns | 409 ns | 595 ns |
| 8 | 2330 ns | 3002 ns | 603 ns | 763 ns |

The scatter and the ghost fill are **a third of an evaluation** at `q = 4`
and do not grow with the order, which is the traffic estimate under
[Precision, threads, devices](#precision-threads-devices) seen from the
host: the kernel's arithmetic and the mesh pattern around it are of the
same order, and neither dominates. Threading is worth 3.6–3.9× on four
threads, and the numbers above are what G6 measures the H200 against.

**The stencils come from a provider (amended in step X2a).** The kernel no
longer calls its contractions directly: `gh_rhs_at_point(S, …)` asks a
*stencil provider* `S` for each one, so that the closures at an excision
surface are this physics with other stencils and not a second copy of it
([Excision](#excision-added-2026-10-05), "On the mesh, and on a device").
A provider is an `isbits` value built per point. It answers five methods,
each about one component at the point, addressed by that component's linear
index `base` in the working array:
- `d1(S, work, base, d)`, `d2(S, work, base, d)`, `dmix(S, work, base, i,
  j)` (outer sum along `i`, inner along `j`, as before) and `ko(S, work,
  base, d)` are raw contractions on unit spacing. The `1/h`, `1/h²` and
  `ε_KO/h` stay in the right-hand side where they were **(proposed in step
  X2a**: the brief named the methods and not where the spacing goes;
  unscaled is what keeps today's arithmetic, since the dissipation scales a
  sum of three contractions by `ε_KO/h` once**)**.
- `adv(S, β_d, ∂f_d, work, base, d)` is the derivative that multiplies
  `β^d` in the two advective terms, `β^k ∂_k h_ab` and `β^k ∂_k Π_ab`, and
  nowhere else. It is handed the shift's component and the scaled
  derivative the kernel already formed — `∂_d h`, which the coefficients and
  `∂_i(α√γγ^{ij}) ∂_j h` keep, and `d1/h` of `Π` — and returns a scaled
  one.

`Centered{T,q}` holds the working array's strides and the three
`@generated` weight vectors. Its methods are the `axis_stencil` and
`mixed_stencil` calls of before, and its `adv` returns `∂f_d`, so the kernel
forms no new stencil. The original signature, `gh_rhs_at_point(T, …, st,
…, Val(q), …)`, builds it and calls the provider form, so `gh_rhs_kernel!`'s
call site did not change. The other kernels that take stencils — the two
constraint monitors, Löhner's `τ` and the `Π` post-pass — keep their own
centered contractions: step X2b masks the first three around the excised
set and refuses the last.

**The refactor is invisible (measured in step X2a)**, against the
integration branch's `5dcddab` in a second checkout with the same manifest
(TreeAMR 0.1.7, Julia 1.13.1, Apple silicon):
- `test/thread_workload.jl`'s digest is identical, character for character,
  at one and at four threads.
- A one-chunk `test/octant_runs.jl` (`case=ks L=8 N=16 roots=2 radii=4,2
  t_end=1/2 chunk=1/2 cfl=1/2`, 28 steps at `q = 4` with the algebraic
  source), for `:damped` with the default noise and for `:fitted` without:
  `octant.csv` is identical in every column but `wall`, and `records.csv`
  is identical, at one and at four threads.
- One `gh_rhs!` in eleven kernel specialisations — the gauge wave at
  `q = 2, 4, 6`, with and without dissipation and at `Float32`; shifted
  Minkowski and harmonic Kerr with the sampled source; step 5's fixture
  `:damped`, `:pasted` and `:frozen`, and with the algebraic source — gives
  an identical `du`. `@allocated gh_rhs!` is unchanged: 12.5 kB on the
  gauge wave and 114 kB on the fixture, the ghost fill's, none per point.
- On Metal at `Float32` (a scratch environment with `Metal` 1.11.1, as
  CLAUDE.md asks), the kernel compiles, and one `gh_rhs!` of the gauge wave
  at `q = 2, 4` and of the fixture's `:damped` layer gives a `du` identical
  to the base's.
- `test/evolution_tests.jl` claims the interface: the centered provider's
  four contractions are `isequal` to the host's cartesian `apply_stencil`
  and `apply_mixed_stencil` (`q = 2, 4`, `Float64` and `Float32`), and a
  host-side `ProbeProvider`, which wraps `Centered` and logs every request,
  sees each of a point's 240 stencils asked for exactly once, `adv` handed
  `β^d` and `d1/h` of the same field, and an offset added to `adv`'s answer
  move `∂ₜh` or `∂ₜΠ` by `Σ β^d δ_d` to roundoff and nothing else. The
  probe's `F` — the same body compiled for a provider whose methods are not
  inlined — is bit for bit the built-in's on Apple silicon; the test claims
  `64 eps` only, because two specialisations of one body have differed in
  the last place on x86-64 before (CLAUDE.md, "Two spellings of one
  expression").
- `bench/stepping.jl` (`BENCH_CASE=wave,hole`, four threads, `N = 16`, 512
  blocks), base and branch interleaved base–branch–branch–base, four runs
  each, on a machine loaded 5–17 by other work. Minimum time in ms, the
  range over the four runs and their median:

  | row | base | branch | change of the median |
  |---|---|---|---|
  | wave, `rhs` | 961–981 (969) | 930–961 (941) | −2.9 % |
  | wave, `imex_owner_step` | 3922–4240 (3937) | 3771–3849 (3824) | −2.9 % |
  | hole, `rhs` | 1397–1486 (1407) | 1435–1489 (1453) | +3.3 % |
  | hole, `imex_owner_step` | 5633–5841 (5666) | 5799–5869 (5861) | +3.5 % |

  The differences have opposite signs in the two cases and are of the size
  of the base's own spread (6 % on the hole's `rhs`, 8 % on the wave's
  step). A second check, `gh_rhs!` alone 25 times in each of three
  processes per tree, interleaved, gave the same picture (wave 945–962
  against 973–990, hole 1451–1452 against 1409–1436, leaving out one
  branch process that ran during a load spike to 88). The arithmetic is the
  same bit for bit; what differs in the compiled code is the layout of the
  closures' captured environments — `S` where `w1`, `w2`, `wD` and `st`
  were, the same 160 bytes — and a ±3 % that changes sign with the
  specialisation is what code layout does **(proposed in step X2a**, not
  measured further; X3's H200 measurement of the `:damped` `q = 4` kernel's
  registers and spills is the check on a device**)**. **(Measured in step
  X3:** on the H200 the values are identical before and after the refactor,
  but the kernel's stack frame grew from 10 528 to 10 640 bytes and its
  spills by 16 bytes each way, at 255 registers both, and its right-hand
  side is `+5.2 %` slower — 28.8 against 30.3 ms on `bench/stepping.jl`'s
  hole, reproducibly; recorded, not fixed (proposed in step X3), under
  [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-on-the-static-hole-step-x3), "Excision on the static hole (step
  X3)".**)**

**Two more providers, and a second launch (amended in step X2b).** The
`:excised` variant ([Excision](#excision-added-2026-10-05), "What step X2b
built") adds `ClosureProvider`, the closures at the evolved points next to
an excision surface, and `Lopsided`, `Centered` with the lopsided advection
blended into `adv`. `gh_rhs_kernel!` takes two more arguments, the class
array and the blend — `nothing` for every other variant, which compiles to
the code it was — and for an `:excised` problem `gh_rhs!` launches
`gh_zone_kernel!` after it. **Every other run is the run it was (measured in
step X2b)**, against `8fa6658`: the thread digest's six lines identical, the
one-chunk octant runs identical in every column but the wall clock, and the right-hand side's cost within the
machine's noise — under [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-the-variant-step-x2b), "Excision:
the variant (step X2b)".

**One design after `main`'s rewrite (amended in step X4).** Step X4 merged
`main`'s spill-free kernel (above, "Implemented 2026-10-05") into the
excision round. The provider is now the argument of `main`'s two functions:
`gh_rhs_head(S, …)` asks it for the thirty `d1` of `h`, the advection of
`h` (`adv`) and the dissipation of `h` (`ko`), and `gh_rhs_pi(S, …)` for one
Π component's `d1` of `h` and `Π`, `d2`, the three `dmix`, the advection of
`Π` and its `ko`. `gh_rhs_store!(du, o, sd, S, …)` (the head, then a run-time
loop over the components, each stored as it is finished) and
`gh_rhs_at_point(S, …)` (the same, unrolled, as two vectors) are built on
them, and `main`'s signatures with `st` and `Val(q)` build `Centered` and call
the provider form. So:
- **`d1` of `h` is asked twice** per component and axis, by the head and by
  the Π component (`main` re-forms it rather than keep thirty values live
  across the source); a provider must answer both alike, which one whose
  methods are functions of `(work, base, d)` does.
- **`Centered{T,q}` holds the strides and nothing else** **(proposed in step
  X4)**: its methods form `derivative_weights` and `dissipation_weights` —
  `@generated` constants — inside, so that inlined they *are* `main`'s
  `axis_stencil` and `mixed_stencil` calls. X2a's `Centered` carried the three
  weight vectors as fields, 160 bytes a point that a device which does not
  inline passes through the stack.
- **The main kernel's branches**: `:none` is `main`'s
  `gh_rhs_store!(du, o, sd, T, …, Val(q), …)`; `:excised` calls the same
  function with the same arguments at a centered point, writes zeros by a
  run-time loop at an excised one, and nothing at a zone point; inside the
  lopsided blend's shell it calls the store with `Lopsided`. The layer
  variants are `main`'s, through `gh_rhs_at_point`. No branch forms a closure.
- **The zone kernel** stores through `gh_rhs_store!(…, ClosureProvider, …)`;
  the closure provider's own contractions stay generic loops over the table,
  since the zone is a shell.
- **Where the blend's weight is zero the main kernel takes the `:none` call,
  not `Lopsided` at `λ = 0`** **(proposed in step X4)**. The arithmetic of the
  two is the same — `adv` returns its argument through a branch — but
  `metric_quantities` forms `g^{ab}` with StaticArrays' products, which are
  `muladd`s, and LLVM fuses them into FMAs by the code around them: the head
  compiled around a provider with a branch in `adv` differed from the
  blend-free one in the last place at 496 of the fixture's 25 165 points
  beyond the blend's shell. The same is why **the closure provider's `F` at a
  point with no excised tap is no longer bit for bit `Centered`'s** (it was
  on Apple silicon through X3): every contraction still is, `β` differs by
  2.8 eps and `A^{ij}` by 6.8, and in a stationary background, where `F` is
  the small difference of `O(1)` terms, that is up to 103 eps of each
  variable's largest `|du|` on the fixture (1682 of 4508 points; the test
  holds it to 512, as the suite's other comparisons of two specialisations).
  No production path evaluates the closure provider at such a point.

**Measured (step X4)**, against `main` (`cbe3662`) and the integration branch
before the merge (`b80f4ed`), the same manifest (TreeAMR 0.1.7, Julia
1.13.1):
- **Every other run is `main`'s.** `test/thread_workload.jl` at four threads
  prints `main`'s six lines character for character (its seventh, the excised
  right-hand side, has X2b's counts and classes); the one-chunk octant runs
  (`case=ks L=8 N=16 roots=2 radii=4,2 t_end=1/2 chunk=1/2 cfl=1/2`), `:damped`
  with the default noise and `:fitted` without, give `main`'s `octant.csv` in
  every cell but the wall clock and `main`'s `records.csv` in every column
  `main` has.
- **On the H200 the `:damped` kernel is `main`'s**, PTX and all: 255 registers,
  a 6576-byte frame, 5700/8480 bytes of spill stores/loads, 50 `ld.local`, 119
  `st.local` and 47 call sites in 8015 lines of PTX, in both; its right-hand
  side on `bench/stepping.jl`'s hole (512 blocks of `16³`) 17.71 and 17.72 ms
  against `main`'s 17.66 and 17.66 and the integration branch's 30.35, and on
  X3's scan octant (`N = 64`, 7.6 M points) 5.28 ns a point against `main`'s
  5.29 and 11.40. **X2a's `+5 %` is gone** with the rest of the integration
  branch's kernel: the provider costs nothing `ptxas` or the clock can see.
- **The excised kernels are `main`'s kind now.** The `:excised` main kernel: a
  1544-byte frame (8800 before), 2692/3700 bytes of spills (192/192), 5 call
  sites (131); 1.50 ns a point on the scan octant (9.73) and the right-hand
  side 2.88 (11.14). The zone kernel: 1512 bytes (9984), 2140/3740 (172/172), 6
  call sites (112); 238 ns a zone point on the octant (382) and 70.7 on the
  bench's 512 small blocks (140). The blend costs `+2.7 %` of a right-hand
  side on the octant (it cost `+20 %`) and `+10.5 %` on the bench (`+10.8 %`).
  Under [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#the-merge-with-main-and-the-rotating-octant-step-x4), "The merge with `main` and the
  rotating octant (step X4)".

**A third launch for a spinning hole (amended in step X6).** An `:excised`
problem whose build found a frame-dragged axis launches `gh_dragged_kernel!`
after the zone kernel: the same `gh_rhs_store!` with a fourth provider,
`DraggedProvider` — the zone point's `ClosureProvider` with `adv` replaced
along those axes by step X5's rule — at the zone points that have one, and
nowhere else ([Excision](#excision-added-2026-10-05), "What step X6 built").
The main kernel and the zone kernel are unchanged by it, the zone kernel's
mixed derivative aside (symmetric in its two axes from X6). On the H200 the
new kernel sits at 255 registers with a 1872-byte frame, the zone kernel's
1504.

**On the CPU the kernel evaluates `W` points at a time (added 2026-10-05).**
The algebra above is generic in its number type, so on the CPU step (2) runs it on
SIMD.jl's `Vec{W,T}`: `W` neighbouring points along the first axis, every stencil a
contiguous vector load and every store a vector store. The launch is unchanged —
`map_blocks!` over every owned point — and the item at the start of each group of
`W` does the group's work; a row that `W` does not divide ends in a group that
overlaps the one before it and stores the shared points again. In a hole's evolved
region a group with a point in the layer or the core runs the scalar code point by
point. `W` is chosen by `GHProblem` — four `Float64` on AVX2 and aarch64, eight with
AVX-512, one on a device. Why and what it buys: [The right-hand side on a
CPU](#the-right-hand-side-on-a-cpu-measured-2026-10-05).

**Excision runs scalar on the CPU (proposed in the main merge, 2026-10-08**, on
Erik's guidance that the singularity handling need not take the lanes yet**).**
`main`'s scalar path is the kernel's `W = 1` branch, `gh_rhs_point!` — what every
device and every software type runs — and an `:excised` problem takes it whatever
its `W` (`gh_rhs_kernel!`'s `W == 1 || INT === :excised`): every work item is its
own point, as on a device, and the `:excised` branch inside it is X4's and X6's,
unchanged. The zone and frame-dragged kernels were always launches of their own,
one point an item. Nothing else changed: `:none` and the layer variants are
`main`'s lanes, and `gh_rhs_lanes!` is `main`'s. So no lane ever holds an excised
value; every centered point is the `:none` kernel's *scalar* call — the `:none`
kernel at `simd_width = 1` — bit for bit, and the default lanes' to roundoff,
which is what `excision_tests.jl` has claimed since X4 (512 eps; 24 605 of the
fixture's 55 505 centered points are bit for bit, the rest within 210 eps, and
17 237 of the spinning hole's 31 840 within 110);
and `excision_tests.jl` holds the excised right-hand side at the host's `W` to the
`W = 1` problem's (bit for bit here). **SIMD lanes for excision are future work**,
priced under [Possible extensions](#possible-extensions).

**The weights are inlined (amended 2026-10-08**, in the same merge**).** Through
X4's stencil provider, `Centered` forms its weights inside each method, and
`derivative_weights` and `dissipation_weights` were generated methods without an
`:inline` meta. At an IEEE type inference folds them to constants, but not at
SIMD.jl's `Vec{W,T}`, whose division it cannot evaluate: each stencil on the lanes
then called the method and read the vector back through memory, where `main`'s
kernel forms the weights once per `gh_rhs_head` and `gh_rhs_pi`. Measured on
`bench/stepping.jl`'s hole at one thread, before the meta: the merged kernel on lanes
1.47–1.52 s, `main`'s 1.03, the scalar kernel 1.43 in both — the lanes bought the
merged tree nothing. With the meta in the generated body the weights are constants
again at every type, and the kernel on lanes is 0.85–0.86 s, against `main`'s
0.97–0.98 and the scalar kernel's 1.33–1.36 in both (two alternations, one
thread). The values are the same: the merged `du` of that hole, on lanes and
scalar, and of the gauge wave, is `main`'s bit for bit.

**Measured (main merge, 2026-10-08)**, development machine (Apple silicon), Julia
1.13.1, TreeAMR 0.1.7, the same manifest for every tree:
- **Every other run is `main`'s.** `test/thread_workload.jl` at four threads prints
  `main`'s six lines character for character, and its two excised lines are X7's
  (`bd0d584`) character for character: the scalar excised kernel is X7's
  arithmetic. The merged `du` of `bench/stepping.jl`'s hole (`:damped`, 2.1 M
  points) and of the gauge wave is `main`'s bit for bit, on lanes and at `W = 1`.
- **The two excised smokes on the rotating octant** (CLAUDE.md's commands, `a =
  3/5` with `r_E = 1`, `r_0 = 0.8`, and `a = 0` without noise, to `t = 1`) finish
  with `excision_flips = 0` at every row and write `simwatch.toml`; every cell of
  their CSVs is X7's within `7.7·10⁻¹⁰` (identical at `t = 0`), the roundoff of
  `main`'s respelled algebra (`_scale`, `_select`) in the scalar kernel.
- **On the H200** (job 571644, `h200debugq`, from a copy with `CUDA` added, the
  same remote manifest for both trees: TreeAMR 0.1.8, KernelAbstractions 0.9.44,
  CUDA.jl 6.4.2) the device path is the one X4 and X6 built — a device runs
  `W = 1`, `gh_rhs_point!` — and measures as `bd0d584`'s: one right-hand side of
  the smoke's octant within `5.7·10⁻¹²` (`a = 0`) and `1.1·10⁻¹²` (`a = 3/5`) of the
  CPU's, the classes identical; on X7's `1/24` mesh the excised right-hand side
  `2.13 ns` a point against `2.19`, the zone kernel and the rule `5.2 %` of it,
  `:damped` `4.84` against `4.98`; `ptxas`: the excised main kernel 255 registers
  and a 1592-byte frame (X7's 1544), the zone kernel 1544 (1504), the rule's 1936
  (1872), `:damped` 6584 (6576). The `a = 3/5` smoke on CUDA is the CPU's to
  `3.2·10⁻⁹` in every cell of its CSV, `excision_flips = 0`. (Measured on the
  merge's masked-lanes prototype before the weights' inline meta. Neither
  difference reaches a device: it runs `W = 1` either way, and at `Float64`
  inference folds the weights with or without the meta.)
- **What scalar excision costs on the CPU** (`bench/stepping.jl`, four threads,
  2.1 M points in 512 blocks of `16³`, the bench's excised ball `r < 3/4`;
  interleaved with `main` and `bd0d584`, three rounds; the machine's own spread is
  6–8 %), the least time of each of three rounds in ms:

  | tree, case | right-hand side | RK4 step by owner |
  |---|---|---|
  | `main`, `:damped` on lanes | 977, 853, 856 | 3933, 3464, 3519 |
  | merged, `:damped` on lanes | 900, 830, 827 | 3705, 3395, 3376 |
  | `bd0d584`, `:damped` scalar | 1067, 1008, 960 | 4360, 4146, 3889 |
  | merged, `:excised` scalar | 1056, 953, 953 | 4346, 3904, 3876 |
  | `bd0d584`, `:excised` | 1025, 1279, 950 | 4266, 4260, 3849 |
  | the masked-lanes prototype, `:excised` | 886, 820, 820 | 3822, 3364, 3317 |

  The merged `:damped` is `main`'s (3 % faster at the best, inside the spread); the
  scalar `:excised` is X7's; and on the merged tree it costs **15 %** more than
  `:damped` on lanes, a right-hand side and a step alike (953 against 827 ms,
  3876 against 3376) — the price of the future-work item below. The zone kernel is
  0.93–0.95 % of the excised right-hand side, `628–690 ns` a zone point. On one
  thread, the kernel alone (two alternations with `main`): `:damped` on lanes
  0.85–0.86 s (`main`'s 0.97–0.98), `:excised` scalar 1.29–1.30 s at the host's
  `W = 4` and at `W = 1` alike, the prototype's masked lanes 0.78 s; the scalar
  route costs the kernel 1.5× and, the ghost fill being two thirds of a right-hand
  side here, the right-hand side 15 %.

### The time step

    dt = cfl · minimum_spacing(forest) / λ_max,
    λ_max = max over owned points of  α √(tr γ^{ij}) + |β|

GHSO2's conservative bound on the coordinate characteristic speed, taken
once per chunk from a speed slot in `diag` (a kernel writes it,
`block_mapreduce(max)` reduces it — `mesh_mapreduce(max)` from 2026-10-01)
and re-checked at the chunk's end as
TreeHydro does — throw, do not warn, if the step used violated the bound.
`cfl = 1/4` **(proposed** default, GHSO2's**)**.

**(Measured in step 3.)** `λ_max` is exactly `√3` on flat space — the
conservative bound, not the physical speed 1 — and larger wherever there
is a shift. Every run of step 3 took its step from it at `cfl = 1/4` and
none needed a smaller one: the gauge wave at `q = 2, 4, 6`, shifted
Minkowski through a Dirichlet face, and a thousand steps of noise. The
re-check at the chunk's end arrives with the driver in step 5; the speed
kernel, `max_speed` and `gh_dt` are in `evolution.jl` now.

**(Measured in step 5.)** Two things a hole adds. First, **the speed
kernel takes the mask like every other analysis kernel** — this section
had not said so and the one above it had; the frozen core holds data that
is not a numerical solution, and a *degenerate* metric there, which is
what a stale core looks like after a regrid has interpolated it, gives a
`NaN` speed that would then set the step for the whole hierarchy. It is
a branch and not a multiplication, for `0 · NaN = NaN`. Second, **the
recheck fires, and the first time it did it was right.** On the static
Kerr-Schild hole the maximum speed is *not* at the hole: it is at the
outer corner of the box, where the metric is nearly flat and `λ` is just
below flat space's `√3 = 1.73205` — `1.67095` at half-width `5/2 M`. As
the discrete solution settles, that corner drifts and `λ` climbs *past*
`√3`, so a step sized at exactly `cfl = 1/4` from a chunk's opening value
has no room: at `q = 2`, `h = 5/48`, `chunk = 1 M`, the recheck threw at
`t = 25 M` with `λ_end = 1.74281` against the `1.67095` the step was
sized from, a CFL number of `0.2574`. The remedies the message names are
a shorter chunk or a smaller `cfl`; `cfl = 1/5` runs the same
configuration to `t = 50 M`. This is the one measurement in the package
that would not exist without the check, which is the argument for having
it throw.

**(Amended in step 8.)** A moving hole's step is sized from `λ` times the
square of the growth the previous chunk measured (step 8f) **and never by
less than 1 %**: once the hole crosses a mesh finer than step 8f's capsule
the growth is not monotonic, and on step 8's uniform `5/128` control of the
boosted `a = 0` hole the speed grew `0.27 %` in chunk 4 after a quieter
chunk 3 — the recheck stopped the run at `0.75 M` with a CFL number of
`0.20003` against the requested `0.2` (measured in step 8). The margin costs
1 % of the steps of a moving run and nothing on a static one, whose step is
unchanged bit for bit; the recheck stays **(proposed in step 8)**.

## Gauge and constraint damping

**Prescribed gauge sources** (decided). `H_a(x)` is sampled from the
background, `H^a = −Γ^a[g_exact]`, together with `∂_a H_b`, so that the
background is a stationary point of the discrete system. The 20 values
live in the `Hsrc` field set with `G = 0`, are read by the RHS kernel at
the owned point only, and are **re-sampled after every regrid** rather
than transferred: they are a function of position, and re-evaluating is
exact where interpolation would not be. Sampling is the most expensive
setup phase in GHSO2 (nested forward-mode duals through the metric) and
runs as a `fill_by_coordinates!`.

Two consequences shape the case list:

- **Harmonic backgrounds have `H ≡ 0` exactly** — Minkowski, the gauge
  wave, Kerr in harmonic coordinates — and the kernel is compiled
  without the gauge-source terms (`Val(false)`), with no `Hsrc` set at
  all. A **boost preserves the harmonic condition** (`□x^a = 0` is
  Lorentz covariant), so a *boosted* Kerr in harmonic coordinates is
  harmonic too. That is what makes it the proof-of-concept case.
- **A time-dependent gauge source is not supported.** Boosted
  Kerr-Schild has `H_a(x − vt)`, which a per-chunk sample cannot
  represent and an in-kernel evaluation (nested duals per point per
  evaluation) would price at about one RHS. The driver refuses a
  non-harmonic background that is not static, and says why; in-kernel
  sources are an extension.

**How the two questions are answered** (step 3, `src/gauge.jl`), because
they are not the same kind of question. **Harmonicity is a table** over
the background types, not a measurement: it decides a `Val` parameter and
whether an `Hsrc` field set exists at all, so it has to be exactly right,
and a harmonic background's sampled `H^a` is **1.4e−16** rather than zero
— eight orders below a non-harmonic one's `0.02 … 0.07`, but no threshold
on that number is a fact about a spacetime. `translate`, `rotate` and
`boost` take the inner metric's answer, since an affine change of
coordinates preserves `□x^a = 0`; the gauge-wave transformation is
harmonic over Minkowski and is not claimed to be over anything else; the
fallback is `false`, which costs a sampling pass and is never wrong about
the physics. **Staticity is measured**, and exactly: a static background's
metric expression does not mention `t`, so the `t` partial of `dmetric`'s
pass is an identical zero and the test is `iszero` with no tolerance to
choose. `test/gauge_tests.jl` checks the table against the measurement on
every row, which is also what would notice if one of `SpacetimeMetrics`'
unexported wrapper types were renamed on `main` — they are listed in
`prerequisite_tests.jl` for that reason.

**The algebraic Kerr-Schild source (built 2026-10-02**, proposed under
[Open questions](#open-questions) the same day**)**. A case may carry
`gauge_source = KerrSchildSource(T; M, spin, velocity)` (or `:algebraic`
through `hole_case` for an unboosted, unrotated `KerrSchild`): `H_a = −K w_a
/ (M + √(M² − s²))` with `w_a = k_ab u^b`, `K = u·w`, `s = S·w`, evaluated in
the kernels from the state itself (`k = h`) — the right-hand side, the gauge
constraint and the ADM monitor — and `∂_a H_b` by the chain rule through the
`∂g` each kernel already forms (`∂_t g` from the first evolution equation
where the source is needed before it). It travels behind the existing
gauge-source `Val` as `Val(:algebraic)`, the `isbits` source itself in the
slot where the sampled `Hsrc` array goes, so there is no field set, no
nested-dual sampling and no re-sampling after a regrid. `GHCase` refuses a
source that does not reproduce the background's own `−Γ_a` and its gradient
at twelve points (`check_gauge_source`), and with one the refusal of a moving
non-harmonic background is lifted, because the source moves with the hole.
**(Measured 2026-10-02.)** The closed form is `gauge_source_grad`'s `H_a`
and `∂_a H_b` to `2–4·10⁻¹⁵` for Kerr-Schild at rest (`a = 0`, `0.9`),
translated, and boosted along `x` and in a general direction with spin
(`test/gauge_source_tests.jl`). On the static `a = 0` hole (octant `[0, 3]³`,
uniform `h = 1/16`, `q = 4`, the `:damped` layer, `t = 2 M`) it runs as the
sampled source does — masked error `1.44·10⁻⁵` against `1.11·10⁻⁵`, `ℋ`
`1.81·10⁻⁵` against `1.93·10⁻⁵`, `C_a` `3.87·10⁻⁶` against `3.96·10⁻⁶` — in
`62 s` against `90 s`; `cfl = 1/2` reproduces `cfl = 1/4` to six digits at
every row to `4 M`.

**Constraint damping** (decided): the Gundlach–Pretorius term `Z_ab`,
with `γ0(x)` a *function of position* — a Gaussian of width a few `M`
around the hole's analytic center, tapered to a small value in the wave
zone, an `isbits` closure over `(center(t), M)` in the problem — and a
constant `γ2 > −1`. `γ0 ≈ 1/M` near the hole is GHSO2's measured
requirement for a stable evolution with the horizon in the domain.

**(Implemented in step 5**, `src/gauge.jl`.**)** The profile is
`γ0(x) = far + (near − far) exp(−r²/2w²)` with `r = |x − c(t)|`, and the
three numbers are `near = 1/M`, `far = 1/(10 M)`, `width = 3 M`
**(proposed in step 5**: `CODE.md` asked for "a Gaussian of width a few
`M` … tapered to a small value in the wave zone" and left them open**)**.
A Gaussian rather than a compactly supported bump because nothing depends
on it vanishing exactly — `far` is the wave zone's rate, not zero — and
because `C^∞` costs one `exp` per point either way. Two consequences for
the code around it: `GHCase`'s `γ0` field now holds a *profile* and not a
number (a bare number is wrapped in `ConstantDamping`, so every
flat-space case of steps 3 and 4 is the arithmetic it was), and the
right-hand-side kernel therefore forms the point's position
unconditionally — three fused multiply-adds, which the compiler drops
where the profile is constant and there is no interior.

## Boundaries

### Periodic

Free, through the tree. The gauge wave and the robust-stability tests are
periodic in every dimension. **Shifted Minkowski is not** (amended in step
3): its profile `ψ′(x) = A sech²(x/w)` is a function of `x` alone and is
*even* in it, so closing the box in `x` would join two equal values with
opposite gradients — a kink, which no stencil resolves and which every
error norm would then report as truncation error. That case is therefore
Dirichlet in `x` and periodic in `y` and `z`, which its solution does not
depend on at all, and it is what puts the hook below under test at G2
rather than at G4.

### Outer boundary: Dirichlet from the background, at the current time

    boundary = CellBoundary(AllVariables(x -> state(background, t, x)))

built *inside the RHS* at each evaluation, closing over `t` and the
background (both `isbits`, so it is a kernel argument like any other and
the closure's type does not change between evaluations). It is exact for
every background this package evolves, static or moving: the analytic
solution is known everywhere at every time, so the boundary data is the
solution and a wave leaving through the boundary is absorbed to
truncation order. The same hook goes to `regrid!` and to
`adapt_to_initial_data!` (TreeHydro's "the hook goes to three places"),
each with the time of that call. For a vertex-centered field set the
domain's upper boundary plane belongs to nobody and the hook fills it;
the lower boundary points are owned and evolved with ghosts the hook
filled below them.

This is the only outer boundary the package has. A radiative condition
for a solution that is *not* known at the boundary needs a hook that
reads the interior, which TreeAMR's device form cannot do — see
[Possible extensions](#possible-extensions).

**(Measured in step 3.)** The hook writes the analytic state into the
outer ghosts and into the shared upper plane **bit for bit** — the same
numbers `state_tuple` gives on the host at `coordinates(fs, b, idx)`,
because TreeAMR's boundary kernel builds its position the same way
`coordinates` does — and a run of shifted Minkowski through it converges
at order `q` (3.93 in L2 at `q = 4`), which is the statement that a
Dirichlet face costs the scheme nothing. `dirichlet(case, t)` returns
`nothing` where the case is periodic in every dimension, so the
right-hand side branches once on a `Bool` the problem stores rather than
handing `fill_ghosts!` an argument whose type depends on the case.

**(Measured in step 8.)** For G5's own background, `boost(Harmonic(1,
7/10), 0.3 x̂)`, the hook built at `t = 7/10` writes the analytic state
into the outer ghost planes, the shared upper plane and a corner **bit for
bit** (`test/moving_tests.jl`), where the same points at `t = 0` differ by
more than `10⁻³`: the time dependence of a hole that moves reaches the
boundary through the hook's `t` and nothing else. Every moving run of step
8 fills its boundary this way at every stage.

### Reflecting faces: symmetry planes (added 2026-10-02)

A case's `reflecting` is TreeAMR's M10 (`(lo, hi)` per dimension, refused
together with `periodic` in one dimension): a face across which the
solution is its own mirror image. It is not the hook's — the ghost schedule
turns a ghost region crossing it into a copy, prolongation or restriction
from the mirrored source, times each variable's parity — so `dirichlet`
returns `nothing` only when *no* face is outer (`has_outer_face`), and the
refinement ceiling skips reflecting faces (they have no Dirichlet mismatch,
and on an octant they are where the hole is). `GHProblem` refuses a forest
whose faces are not the case's.

**The parities are the tensor's.** `h_ab` and `Π_ab` are symmetric
tensors, so component `ab` is odd across `x^d = 0` exactly when `x^d`
occurs an odd number of times in `ab`: `h_tx` is odd in `x`, `h_xy` in `x`
and `y`, `h_tt` and the diagonal even everywhere (`state_parity`). Every
other field set of the package has `G = 0` and is never ghost-filled or
transferred, and is declared even (`even_parity`), which TreeAMR requires
and nothing reads. Over a forest without a reflecting face both are
`nothing`, so such a run and its checkpoints are what they were.

**The low wall plane is evolved, and an odd variable is not forced to zero
on it** (TreeAMR's rule): it stays zero to roundoff if it starts zero.
`add_noise!` therefore projects its noise onto the parity on the wall
planes. **(Measured 2026-10-02**, `test/reflection_tests.jl`**.)** A
uniform octant `[0, 4]³` is the box `[−4, 4]³` with the noise extended by
parity, to `1.1·10⁻¹⁵` relative after 32 steps; on a three-level octant
the odd components stay on their walls at `7·10⁻¹⁸` relative after 20
steps (not exactly zero: a fused multiply-add keeps a mirrored pair from
cancelling bit for bit), and with every variable declared even they leave
at the data's size. A *refined* octant is not the refined box: vertex
centering puts the plane `x = −R` of a cube `[−R, R]³` on the fine level
and `x = +R` on the coarse one, so the box's discretization has no mirror
symmetry at a refinement boundary and the octant's is a different (and
symmetric) discretization of the same problem.

**A hole on the octant** (`hole_case(; octant = true)`, added 2026-10-02)
needs one that is its own mirror image in all three planes — at the origin,
at rest, `a = 0` — and refuses any other: a spin along `z` keeps only the
`z` mirror and a boost breaks its own axis. **(Measured 2026-10-02.)**
Kerr-Schild `a = 0` with the `:damped` layer (`r_0 = 3/4`, `r_1 = 3/2`),
`q = 4`, uniform at `h = 1/16` on `[0, 3]³` against `[−3, 3]³`: after 224
steps (`t = 2 M`, past the layer's saturation at `M/2`) the two agree to
`6·10⁻¹⁴` at every point outside the core — in the layer, at the horizon and
next to the outer face alike — where the solution's own error is `6·10⁻²` in
the layer and `10⁻⁶` at the boundary; the layer residual agrees to every
printed digit at every row; the octant costs `75 s` against `491 s`. The
frozen core, the layer, the sampled gauge source and the `γ0` profile need
nothing of their own at the walls. The suite's version is the `q = 2`
fixture to `1/5 M` (`3.6·10⁻¹⁴`).

### The rotating octant: a quarter turn about `z` (added 2026-10-04)

A spinning hole has no mirror in `x` or `y`, so the octant above refuses it.
It does have every rotation about its axis, and TreeAMR 0.1.7's M12 makes
one of them a seam: `rotating = (d1, d2)` evolves only the quadrant of the
`(d1, d2)` plane, glues the low face of `d1` to the low face of `d2`, and
fills the ghosts across it from the data a quarter turn away, `u(Rp) = Q
u(p)`, with `R` taking `e_{d1}` to `e_{d2}` and `e_{d2}` to `−e_{d1}`.
Together with the mirror at `z = 0` (equatorial symmetry, which a spin
along `z` keeps) it is an octant again, a quarter of the bitant proposed
under [single holes on the
octant](SINGULARITY_HANDLING.md#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07).
A case's `rotating` is that pair,
`(0, 0)` for none so that the case stays `isbits`; it is refused on a
periodic or reflecting dimension and between unequal widths, and `GHProblem`
refuses a forest whose seam is not the case's. `hole_case(; octant =
:rotating)` builds the seam `(1, 2)` with the mirror at `z = 0`, and needs
the axisymmetric hole — at the origin, at rest, any spin along `z` — and
refuses any other; `octant = true` is `:reflecting` and still refuses a
spin, naming `:rotating`.

**The map is the tensor's.** A quarter turn sends every Cartesian
component to plus or minus one other: with `(d1, d2) = (1, 2)`, index `x`
of the turned tensor is `−y` of the original and `y` is `+x`, so `h_tx →
−h_ty`, `h_ty → h_tx`, `h_xx ↔ h_yy`, `h_xy → −h_xy`, `h_xz → −h_yz`, `h_yz
→ h_xz`, and `h_tt`, `h_tz`, `h_zz` stay (`state_rotation`; the same for
`Π`). The `G = 0` field sets are declared with the identity
(`identity_rotation`), as they are declared even. Over a forest without a
seam both are `nothing`, and a checkpoint's recipe carries `rotating` only
where there is a seam, so older checkpoints still restart.

**The two seam planes are the same points.** Vertex centering owns the low
plane of `x` and of `y` (TreeAMR's decision), both evolved, from ghosts
that are each other's images. `add_noise!` leaves both unperturbed, as it
projects odd components to zero on a mirror's wall: independent draws there
would be a solution that disagrees with itself. The refinement ceiling
skips the seam's faces as it skips reflecting ones.

**(Measured 2026-10-04**, `test/rotation_tests.jl`**.)** Kerr-Schild at
`a = 3/10` (the ring inside `r_0 = 7/20`, `r_1 = 11/10` eight cells inside
`r₊ = 1.954`), `q = 2`, uniform at `h = 5/48`: the analytic state is its own
image under the map to `2·10⁻¹⁶`; after one ghost fill every stored point
— owned, mirrored, turned or the hook's — holds the exact state to
`1.3·10⁻¹⁴` relative (the worst next to the ring, where the solution is
`350`), and `1.3` with every variable declared to turn into itself; after
two chunks (`t = 1/5`) the rotating octant `[0, 5/2]³` is the box `[−5/2,
5/2]³` to `2.6 eps` relative at every point outside the core, the layer
residual agreeing to `10⁻¹⁴`. The horizon finder needs nothing of its own:
TreeAMR's `interpolate` turns the finder's sphere into the quadrant, and
`M_irr` and `J` on the octant are the box's to `10⁻¹⁰` (`J = 0.29994`,
Kerr's `0.3`). A restart chain on the rotating octant is the uninterrupted
run bit for bit (checked by hand, not in the suite).

### The interior: a pointwise damping layer

#### The design, as steps 8a–8f leave it (rewritten in step 8f)

Inside the horizon the right-hand side is modified point by point, and the
modification is four independent choices; everything below this summary is
the history that made each of them, kept because its numbers are the
evidence (**rewritten in step 8f** around these four; step 5's design is
the first row of each and stays as the analytic control):

1. **The geometry — where the layer is.** Either step 5's **sphere** about
   the analytic center, `r_1 ≤ r_h,min − m h` (`Interior`), or step 8d's
   **tracked offset surface**, the depth `d = r_h(n̂) − m h − |x − c(t)|`
   below the horizon the finder found about the center it found
   (`FittedSpec` in the case, a `FittedInterior` rebuilt every chunk). The
   sphere is the offset surface's `l = 0` case bit for bit. The tracked
   surface is the only geometry for a hole whose center or shape is not
   known in advance, and the only one whose offset surface can hold harmonic
   Kerr's flat singular disk inside an oblate horizon.
2. **The target — what the layer relaxes toward.** Either the **analytic**
   solution (`:damped`, `:pasted`; an optional `target` metric for studies)
   or step 8e's **fit** of the evolved state on the offset surface,
   continued inward as a polynomial in `x` and cached on the grid
   (`:fitted`). The analytic target needs the chart's singular set inside
   the core surface; the fitted one needs nothing inside, and is the only
   target for harmonic Kerr at `a = 7/10`, G5's chart (step 8f: the analytic
   core there cuts the disk at every `h` the proof of concept can afford).
   The fit is made in `(log α, β^i, γ_ij, Π̃_ab)`, `Π̃ = (α/√γ)Π` from step
   8f (**proposed in step 8f**).
3. **The ramp — how wide.** `ρ` rises from `0` at the offset surface to
   `ρ_max` over `n_L = max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)` cells, `w` turning
   over in the inner half (step 8c's rule, measured at `q = 2`, **proposed**
   for other orders); a margin of `m ≥ G + 1` cells (the stencil margin)
   and, for grid-scale leakage, as many more as step 8a's path integral asks.
4. **The rate — how hard.** `ρ_max` is a physical rate, **`4/M` (decided
   2026-09-23)**, not the grid rate `1/dt` step 5 started with, which only
   an exact target survives. **A moving analytic layer wants more (measured
   in step 8f)**: its core is frozen, a point crosses an eight-cell layer at
   `v = 0.3` in about `M`, and at `4/M` what the core held is relaxed by
   `e^{−2}` before the layer releases it on the trailing side — the boosted
   `:damped` rows end at `1.0–1.5 M` at `4/M` and reach `5 M` at `20/M`
   **(proposed in step 8f: `ρ_max ≳ 20/M` for a moving analytic layer)**;
   the `:fitted` core relaxes toward a target that moves with the track and
   reaches `5 M` at `4/M` ([the measurement
   matrix](SINGULARITY_HANDLING.md#the-generic-interior-the-measurement-matrix-step-8f)).

**What step 8f's matrix decides (proposed in step 8f).** The `:fitted`
target on the tracked geometry holds every hole the package has that the
mesh resolves — Kerr-Schild `a = 0` to `50 M`, `a = 9/10` and harmonic
`a = 0` to the ends of their rows (`20 M`, `10 M`), harmonic `a = 7/10` at
`h = 5/256` to `10 M`, the boosted hole to `5 M` — so **excision (step 8g) is not needed**. It is not
free: where the analytic target exists it is the better one — `2×` the
masked error and `2.8×` the shell's `C_a` on the static Kerr-Schild hole,
`40×` on the spinning one — so **the analytic `:damped` layer stays the
default wherever the chart's singular set fits inside the core, and
`:fitted` is for the charts where it does not**, G5's included.
**(Amended 2026-10-05:** the octant runs since have found where `:fitted`
does not hold — Kerr-Schild `a = 9/10` at `h = 1/48` on the rotating
octant, unstable at `cont = 1` and `2`, with `:damped` under-resolved next
to the ring (2026-10-04, recorded on the branch
`claude/octant-mode-spinning-bh-75bd22`, not yet on `main`) — and Erik
reopened excision, as a variant beside the layer
rather than step 8g's fallback; see [Excision](#excision-added-2026-10-05).**)**

**The runs behind these choices are in
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md) (moved 2026-10-08)**,
and the settings they recommend for a single hole at `a ≤ 9/10` are under
[Single black holes: recommended
settings](#single-black-holes-recommended-settings-added-2026-10-08).

#### Step 5's layer: the analytic control

**No excision** (decided; **amended 2026-10-05**: this is the *layer's*
rule — the `:excised` variant under [Excision](#excision-added-2026-10-05)
is the exception, and is under study). Inside the horizon the solution is not left
to the Einstein equations alone: in a layer well inside the horizon it
is *driven to the analytic solution*, and around the singularity the
evolution is *switched off*. Both are decisions made **point by point**,
as functions of the distance `r = |x − c(t)|` to the hole's analytic
center `c(t) = c_0 + v t`, and neither knows anything about blocks,
ghost zones or refinement levels (decided in review). **(Amended in step
8d:** or, for a case that asks for it, as functions of the *depth* below
the tracked horizon's offset surface about the *tracked* center — "The
tracked geometry" below, of which this section's sphere is the `l = 0`
case.**)** The right-hand side at every point is

    ∂_t u = w(r) · F(u)  −  ρ(r) · (u − u_exact(x, t))                (INTERIOR)

with `F` the GH right-hand side `(EXPANDED)` including the dissipation,
`u_exact` the background's `(h, Π)` at `(x, t)`, and two smooth
profiles:

| region | radii | `w` | `ρ` | what happens |
|---|---|---|---|---|
| evolved | `r ≥ r_1` | 1 | 0 | the Einstein equations, untouched |
| damping layer | `r_0 ≤ r < r_1` | 1, ramping to 0 at its inner edge | 0 at `r_1`, rising smoothly to `ρ_max` | the equations plus a relaxation toward the analytic solution |
| frozen core | `r < r_0` | 0 | 0 | nothing: `du = 0`, `F` not evaluated |

The layer's outer radius sits inside the horizon with a margin: with
`r_h,min(t)` the smallest coordinate distance from `c(t)` to the
horizon over all directions — the Kerr horizon is oblate in these
coordinates, and a boost contracts it along `v` — and `h_L` the finest
spacing present around the hole,

    r_1 ≤ r_h,min − m · h_L,        m ≥ G + 1, default m = 8,

so that the horizon and at least `m` grid points inside it are evolved
by the unmodified equations and no stencil of a point outside the
horizon reaches the layer. The layer is at least `2(G + 1)` spacings
thick at the resolution of the blocks that contain its outer part, so
that the profiles are resolved by the stencils and no stencil of an
evolved point reaches the frozen core. `r_h,min` is analytic for every
background here (`r_+` on the axis for Kerr-Schild, `r_+ − M` for
harmonic Kerr, both times `√(1 − v²)` under a boost); the driver asserts
both bounds at every regrid, and the horizon finder confirms at `t = 0`
that the found surface encloses the layer by the margin. For a rapidly
spinning hole in harmonic coordinates `r_h,min` is small — about
`0.44 M` at `a = 0.9` — and these bounds, not the exterior, set the
finest spacing the refinement must reach (about `0.02 M` there); the
level floor under [Refinement](#refinement-and-regridding) is what
guarantees it.

#### The margin

**The margin `m` does two jobs, and they have different sizes (proposed in
step 8a).** Inside the horizon the continuum lets nothing out, but the
discrete scheme lets grid-scale content out, held back by the dissipation
alone ([Kreiss–Oliger dissipation](#kreissoliger-dissipation), step 8a's
table), and every interior treatment is a source of such content at `r_1`.
So there are two rules, both **(proposed in step 8a)** for the reviewer to
confirm:

1. **The stencil margin, `m ≥ G + 1`** — the bound above, unchanged and now
   named: no stencil of a point on or outside the horizon reaches a point
   where `w < 1` or `ρ > 0`, so the horizon is evolved by the unmodified
   equations. It is a statement about the scheme's reach and it is exact.
2. **The leakage margin, `m ≥ n_e ℓ_max`**, for a wanted attenuation
   `e^{−n_e}` of what the interior makes at `r_1`, with `ℓ_max` the
   frozen-coefficient penetration length at `r_1` along a grid axis at the
   run's `q` and `ε_KO`. That is *necessary* and not sufficient: `ℓ` grows
   toward the horizon, so what the margin actually buys is the path integral
   `n_e ≤ ∫_{r_1}^{r_h} dr / (h ℓ_max(r))`. Near the horizon, with
   `δ = b/a − 1 ≈ g (r_h − r)` along the normal (`g = 1/(2M)` for
   Kerr-Schild at `a = 0`) and `ℓ_max ≈ C_q/(ε δ^{2/q})`, the integral is
   `ε (g h)^{2/q} m^{1+2/q} / ((1 + 2/q) C_q)` e-folds — at `q = 2`,
   `m ≥ √(2 C_2 n_e / (ε g h))`, which is `m ≥ √(1.1 n_e M/(ε h))` in
   Kerr-Schild. (`C_q` is Kerr-Schild's, where `a = 1/2` at the horizon;
   `ℓ` scales with `a` at fixed `b/a`, so elsewhere `ℓ_max ≈ 2a C_q/(ε
   δ^{2/q})`, and `g` is the background's own — `0.30/M` for Kerr-Schild at
   `a = 9/10`, `1.0/M` and `2.2/M` on harmonic Kerr's axis and equator,
   [Kreiss–Oliger dissipation](#kreissoliger-dissipation).) **The leakage
   margin in cells grows as the mesh is refined**, as `h^{−1/2}` at
   `q = 2` and `h^{−1/3}` at `q = 4`, because a margin of a fixed number of
   cells lies ever closer to the sonic surface.

**The `0.1 M` on harmonic Kerr's equator is not too thin; the spin axis is
what binds (measured in step 8a, frozen-coefficient).** Toward the disk
every characteristic speed falls to zero, so `ℓ_max` drops below a cell and
a margin of `m = 4` at `h = 5/256` buys `7.3` e-folds at `q = 2` and `9.7`
at `q = 4` at `ε_KO = 1/2`, while the same hole's axis gets `0.9` and `1.9`
at `m = 4` and `3.5` and `5.5` at `m = 8` — so the offset surface wants a
margin that depends on direction, not a thicker equator.

**What the fixture measures against it** (`test/hole_runs.jl leakage`,
under [step
8a](SINGULARITY_HANDLING.md#step-8a-what-crosses-the-horizon-from-inside-it)).
On the step-5 hole at
`h = 5/64` and `ε_KO = 1/2`, content made `d` cells inside the horizon
reaches the first shell outside it attenuated by **`e^{−0.40…0.51}` per
cell of depth** at `t = 2 M` in 3D (`ℓ ≈ 2.0–2.5` cells, against
`ℓ_max(r_1) = 2.1` at `q = 2` and `2.7` at `q = 4`), and the
one-dimensional model of `test/dispersion.jl` — which reproduces the 3D
numbers to within a factor 2.3 at `λ ≥ 4h` — says the slower modes that
arrive later bring that down to **`e^{−0.16…0.37}` per cell by `10 M`**. From
depth 8, the default margin, the transmitted amplitude is **1 %** of the
source at `2 M` in 3D and **4–6 %** by `10 M` in 1D: `m = 8` is `e^{−2.9}`
to `e^{−4.6}` at this resolution, not the `e^{−8/ℓ}` a constant `ℓ` would
suggest, and `e^{−5}` needs `m ≈ 10–15` here (the `q = 2` formula says
`12`) and more at finer `h`. What `n_e` has to be is set by the amplitude
of what the interior makes, which step 8c's inexact targets measure; so the
rules are stated and **the default `m = 8` is not changed**.

**`ε_KO` rising inside the layer does not buy margin (proposed in step
8a).** The leakage the margin is for is made at or outside `r_1` and crosses
`r_1 ≤ r < r_h`, where a profile that rises only inside the layer still has
the exterior's value: in the one-dimensional model to `10 M`, `ε_in = 1, 2,
4` inside the layer changes the transmission from every depth by at most
1.2 %. What does act is dissipation raised *across the margin* — `C²` from
`ε_out = 1/2` at the horizon to `ε_in` at `r_1`, and held inside: at
`ε_in = 4` it cuts the transmission from depth 8 by **4–15×** (`q = 2, 4`,
`λ = 2h, 4h`), five to nine cells' worth at the long-time rate, and does not
help a source within two cells of the horizon; doubling `ε_KO` everywhere,
measured in 3D, cuts it by 1.7–2.5×. The recommendation to step 8c is
therefore to start its `ε_KO(r)` profile's rise **at the horizon rather
than at `r_1`** — the region is causally disconnected from the exterior in
the continuum, and the dissipation is `O(h^{q+1})` whatever `ε` is — with
`ε_in ≤ 4`, inside RK4's real-axis limit of about `6` on the 3D corner mode
at `cfl = 1/4` (`3 ε dt/h ≤ 2.8`); and, if `n_e` must exceed about 5 at this
resolution, to widen the margin as well, since neither lever alone is
enough.

**A hard step inside the horizon does get out, at the rate 8a measured
(measured in step 8c).** `:pasted` onto `KerrSchild(6/5, 0)` (E0) holds a 20 %
step — `|δh| = 0.348` — at `r_1`, `10.9` cells below the horizon, on 8a's
uniform 512-block mesh, and the first shell outside the horizon carries,
against the same run with the exact target, **`4.2e−3` of it at `2 M` and
`2.2e−2` at `5 M`, still rising** at `ε_KO = 1` everywhere; `0.35–0.39`
e-folds per cell outside. That is 8a's *measured* attenuation — its ripple
from depth 8 continued at its own per-cell rate puts `2.6–3.1e−3` at
`10.9` cells at `2 M` and `ε_KO = 1/2`, where E0 has `2.5e−3` at `1.5 M` — and
140× the frozen-coefficient `e^{−d/ℓ_max(r_1)} = 3e−5`. At `ε_KO ≤ 1/2`
everywhere the step ends the run first (`1.0 M`, `1.75 M`), in the evolved
shell next to it; with the `ε_KO(r)` profile rising from `1/2` at the horizon
to `ε_in = 2` or `4` at `r_1` it survives, with a masked error L∞ of `8.1` and
`6.3` against `244` at `ε_KO = 1` everywhere, and transmits **no less at
`2 M`** (`2.6e−3`, `3.5e−3`) and 1.4–2.3× less by `5 M` (`1.6e−2`,
`9.5e−3`) — not the 4–15× of 8a's one-dimensional model. So `m = 8` holds a
discontinuity back to a percent or two over a few `M`, and nothing holds the
exterior together against one for long: the answer to a discontinuous
interior is a smooth one, rule 2 of [the layer for an inexact
target](#the-interior-a-pointwise-damping-layer), whose smooth wrong targets
leave the exterior indistinguishable from the exact one — and **the default
`m = 8` stays (proposed in step 8c)**.

#### The spinning harmonic chart

**A ball cannot hide Kerr's singularity in the harmonic chart at
`a = 9/10` (found in step 5, and this is the proof-of-concept case).**
The frozen core is a *ball* of radius `r_0`, and what it has to contain is
not a point: both `KerrSchild` and `Harmonic` solve
`R⁴ − R²(x²+y²+z²−a²) − a²z² = 0` for their radial coordinate, so on the
equatorial **disk** `z = 0`, `x² + y² ≤ a²` that coordinate is zero and
every expression in the metric divides by it. The disk's coordinate
radius is `|a|`. So the core needs `r_0 > |a|`, while the placement bound
needs `r_0 < r_1 ≤ r_h,min − m·h`. Those are compatible in Kerr-Schild at
`a = 9/10` — the disk is at `0.9` and `r₊ = 1.436` — and **incompatible
in the harmonic chart**, where `r_h,min = √(M² − a²) = 0.436` is *smaller*
than `0.9`. The Kerr horizon is oblate and the singular disk is flat; in
the harmonic chart the disk pokes out of every sphere that fits inside the
horizon along the axis. `check_interior_radii` refuses the configuration
by name rather than letting a grid point on the disk become `NaN`
(`singular_radius`), and the suite tests the refusal.

This does not touch the static holes of G4 at `a = 0`, and it does not
touch `a = 9/10` in Kerr-Schild, which runs. It **does** stand between
here and the proof-of-concept case, which is `boost(Harmonic(M, 9/10), v)`
— harmonic because a sampled gauge source cannot be time-dependent. Two
ways out, neither built and neither chosen here **(open question, raised
in step 5)**:

1. **Key the interior on the chart's own radial coordinate.** `w` and `ρ`
   become functions of `R`, the spheroidal radius the metric already
   solves for, instead of `r = |x − c(t)|`. Then the core `R < R_0` is an
   oblate spheroid that contains the disk exactly, the horizon is
   `R = √(M² − a²)`, and "`m` grid points inside the horizon" is a
   statement about `R` — the geometry becomes as clean as it is at
   `a = 0`. The cost is that `R(x)` is a background-specific function
   that the kernel must evaluate, so the interior stops being a function
   of position *alone* and becomes a function of position *and the
   background*; `CODE.md`'s "neither knows anything about blocks, ghost
   zones or refinement levels" survives, but "a function of `r`" does
   not. This is the smaller change and the one to try first.
2. **Lower the spin.** `√(M² − a²) > a` needs `a < M/√2 ≈ 0.707`, so a
   harmonic hole at `a = 0.7` admits a spherical core with room to spare
   and `a = 0.9` does not. G5 at `a = 0.7` is a weaker proof of concept
   and a true one.

#### Why touch the interior, and why relax

**Why it is correct to touch the interior at all.** Inside the horizon
every characteristic points inward, so in the continuum nothing outside
the layer depends on what happens inside it; the modified equation
`(INTERIOR)` agrees with the true one wherever `w = 1` and `ρ = 0`, which
includes the horizon and a margin inside it. Discretely, the stencils of
evolved points reach into the outer part of the layer, where `ρ` is
still small and the equations are still (nearly) the true ones, and
find a solution that is the analytic one to truncation order: the best
data an outflow boundary can have, and smooth across the transition.
The analytic solution is an **exact solution of `(INTERIOR)`** wherever
it is regular: `F(u_exact) = ∂_t u_exact` and the relaxation term
vanishes on it, so a static hole is an equilibrium and a boosted hole is
an exact time-dependent solution, in the layer as much as outside.

**Why a relaxation and not a slowed evolution alone** (decided in
review; the pure mask `ρ = 0` was the first proposal and stays as a
measured variant). With `ρ = 0`, `∂_t u = w F(u)` slows the evolution
where `w < 1` and stops it where `w = 0`: the characteristic speeds
scale with `w`, and perturbations that enter the ramp from outside —
truncation error, noise, anything the horizon lets through — slow
down, compress, and pile up against the freezing radius. For an
ingoing wave the energy density grows like the inverse speed, the
gradients grow exponentially at the rate `|∂w|`, and only dissipation
at the grid scale stands against it; in a nonlinear system with a
source quadratic in `∂g` a large pile-up can push the metric toward
degeneracy, which the pointwise algebra cannot survive. The relaxation
term turns the layer from a sticky wall into a **sink**: a perturbation
entering it decays at rate `ρ` toward the analytic solution instead of
accumulating, and nothing reaches the frozen core with an amplitude
worth speaking of. It also makes the moving hole work: `u_exact(x, t)`
moves with `c(t)`, points that the frozen core releases into the layer
are relaxed to the solution within a time `1/ρ_max`, and points the
layer releases into the evolved region already hold it. With `ρ = 0`
neither is true — the frozen values are stale where the core has moved
away, and the exact boosted solution is *not* a solution of `w F` in
the ramp.

**Why a smooth layer and not a hard paste** (decided in review; the
hard paste stays as the second measured variant). Overwriting a ball
with the analytic solution is `(INTERIOR)` in the limit `ρ → ∞` on a
step profile, and can be implemented exactly through RK4's
`step_limiter!`, as TreeHydro implements its atmosphere reset. It
introduces a surface at which the evolved solution meets the exact one
with a truncation-order mismatch, and the stencils that straddle that
surface see it as a kink; the smooth layer spreads the same mismatch
over the ramp. Both are pointwise; the smooth one is what a
finite-difference code should prefer, and `(INTERIOR)` keeps it inside
the right-hand side, where the integrator sees a pure function of
`(u, t)` and no limiter is needed.

#### The profiles and the rate

**The profiles and their parameters.** `w` and `ρ` are `C²` smoothstep
polynomials of `r`, `isbits` closures over `(c(t), r_0, r_1, ρ_max)` and
the ramp widths, evaluated per point in the kernel. **`ρ_max = 4/M`
(decided 2026-09-23)**, `M` the hole's mass parameter: the layer's
relaxation rate for every run and every target, the analytic `:damped`
layer included (`:pasted` and `:frozen` do not read it). `evolve!` given no
rate keyword relaxes at `default_relaxation_rate(case) =
4/hole_mass(background)` in every chunk — `hole_mass` reads `.mass` of
`KerrSchild` and `Harmonic` through `translate`, `rotate` and `boost`, the
rest mass a boost does not change, and refuses a background with none — so
the default is a statement about the hole and not a number in the driver.
It is `16 κ`, and the layer forgets a perturbation in `M/4`. `ρ_max` is
bounded by the explicit integrator — RK4 is stable on the negative real
axis to about `2.8/dt` — and the driver refuses a fixed rate, the
default's included, above `1/dt` at the chunk that would take it; on every
hole the suite evolves the default is `0.036/dt` to `0.092/dt` **(measured
in step 8c′)**, so the refusal is a statement that the mesh does not
resolve `M/4`.

**The grid rate `ρ_max · dt = 1` stays as the option `ρ_max_factor`**, and
is no longer what a run gets by default (amended in step 8c′). It was the
default from step 5, **(proposed)** there because it relaxes by a factor
`e` per step and looked as strong as a layer needs to be; step 8c measured
why it is not: `1/dt` is a *grid* rate, about `107/M` on the suite's
fixture at `cfl = 1/5`, and a paste two cells deep, which every inexact
target ends its run against on every ramp up to eight cells, and even on
the exact solution it ends a `50 M` run with six times the error of `4/M` —
the `G`-point shell's `C_a` L2 `0.181` against `0.029`, the masked error
`0.160` against `0.027` ([the layer for an inexact
target](#the-interior-a-pointwise-damping-layer) below). `ρ_max_factor = 1`
reproduces step 5's numbers, which is what it is kept for; `ρ_max_fixed`
is any other rate, and the two are refused together.

**What the default costs the layer (measured in step 8c′).** A relaxation
at a fixed rate holds the layer to the analytic solution to `τ/ρ_max`, `τ`
the truncation error of `F` there, where the grid rate held it to `τ · dt`:
on the suite's fixture the `:damped` residual — the layer's L∞ distance
from the truth — saturates at `1.42` by `1/2 M` against the grid rate's
`6.2e−2`, a few percent of the solution itself at the layer's inner edge,
where `|Π|` is `60`, and it converges at order `q` (`2.03` over `N = 6, 8,
10`) where the grid rate's extra `1/h` made it `2.81`. The exterior does not
see it: the masked error outside `r_1` is `2.26e−3` against `2.27e−3` at
`1/10 M` and `1.51e−2` against `1.61e−2` at `1 M`, and at `50 M` it is step
8c's row, `0.027` against `0.160`. The residual is a statement about the
layer's own health and not about the solution, and at `4/M` it is the
larger one.

**(Implemented in step 5**, `src/interior.jl`.**)** The smoothstep is the
quintic `10s³ − 15s⁴ + 6s⁵`, whose value, first *and* second derivatives
match the constants it joins. The two ramp widths are **halves of the
layer (proposed in step 5)**, measured from opposite ends: `w` rises from
`0` at `r_0` to `1` at the layer's midpoint and stays there, `ρ` falls
from `ρ_max` at the midpoint to `0` at `r_1`. They are complementary, so
no point is both frozen and undamped, and the outer half of the layer is
the unmodified equations *plus* a relaxation — which is what makes the
data the evolved stencils reach into the analytic solution to truncation
order. The smoothstep **clamps its result as well as its argument
(measured in step 5)**: the polynomial has a triple root at `s = 1`, and
its Horner form at `s = 1 − 2⁻⁵³` returns `1 + 1.3e−15`, so without the
clamp `w` would exceed one just inside `r_1` and amplify `F` where this
section says the equations are untouched.

#### The layer for an inexact target

**The layer for an inexact target (added in step 8c).** Step 8e's target is
a fit of the evolved state, not a solution, so step 8c calibrated the layer
against targets that are wrong on purpose, with three knobs and no new
`Val`: an `Interior` carries an optional `target` metric (`isbits`,
default `nothing` = the case's background, resolved by `layer_target`)
that the kernel reads where `(INTERIOR)` reads `u_exact` — the layer branch
and the `:pasted` overwrite — and nowhere else, so the initial data, the
Dirichlet hook, the gauge source and the error reference stay on the truth
and the record's `residual` is the layer's distance *from the truth*;
`evolve!(…; ρ_max_fixed)` relaxes at a rate in the case's units instead of
`ρ_max_factor/dt` (the two are refused together, and a fixed rate above
`1/dt` is refused at the chunk that would take it, **(proposed in step
8c)**; from step 8c′ a run with neither relaxes at the default `4/M`, and
the grid rate is asked for as `ρ_max_factor = 1`); and `HorizonDissipation`,
the `ε_KO(r)` profile below. The targets:
**E1** `KerrSchild(6/5, 0)`, a valid metric that is not a solution; **E2**
`translate(KerrSchild(1, 0), (0, δ, 0, 0))`, `δ = h, 4h`, a tracking error;
**E3** the solution with `h_tt` off by `−2(r − r_1)²χ(r)/M²`, value and slope
right at `r_1` and curvature wrong by `4/M²` (the sign is **(proposed in
step 8c)**: `+2/M²` drives the target's `α²` through zero in any layer of
eight or more cells, which is finding 2's non-metric and not a curvature
error). The ramp's width `n_L` is the width over which `ρ` rises from `0`
at `r_1` to `ρ_max` — `ρ_ramp = 1`, `r_0 = r_1 − n_L h`, `w` turning over
in the inner half **(proposed in step 8c**, since finding 1's prediction is
a statement about that width**)**. On the suite's fixture (Kerr-Schild
`a = 0`, `q = 2`, `h = 5/64`, `cfl = 1/5`, `ε_KO = 1/2`, the range projection
on) to `50 M`, the numbers under [Measured
results](SINGULARITY_HANDLING.md#the-layer-against-an-inexact-target-step-8c)
decide:

1. **`ρ_max` is a physical rate, `4/M` (measured in step 8c**, and the
   default from step 8c′, decided 2026-09-23**).** At the grid
   rate every inexact target ends its run on every ramp up to 8 cells — E1
   in `2–4 M`, E2 at `δ = h` in `3–5 M`, E3 in `6–15 M` — degenerating in the
   shell outside `r_1` as step 8b saw step 5's failures do; on 12 cells E1
   still ends (`18 M`) and E2 and E3 live at five times the error; and the
   shell's `C_a` loses the scheme's order (`0.65` over `N = 6, 8, 10` against
   `2.38` at `4/M`). `10/M` needs a thicker ramp than `4/M`,
   and `1/M` is too slow the other way: at `n_L = 12`, whose layer reaches
   down to `r_0 = 0.21 M`, E1 ends at `15 M` and E2 at `6 M`, *from the
   inside* — the shell healthy to the end, the projection firing at
   `r = 0.55` — a deep layer that a weak relaxation does not hold against a
   target that is not a solution.
2. **The ramp is at least `4G = 8` cells at that rate (measured in step
   8c).** At `n_L = 8` and `12`, `ρ_max = 4/M`, every target the layer can
   contain — E1, E2 at `δ = h`, E3 — holds the hole to `50 M` with the
   `G`-point shell's `C_a` L2 at `0.029–0.039` and the masked error L2 at
   `0.027–0.032`, both flat from `10 M` on, `M_irr` within `0.3 %` of `M`,
   and no projection hit: **the same numbers as the exact target on the same
   layers** (`0.029`, `0.027`), so the exterior does not see what the layer
   relaxes toward. At `n_L = 6` it costs 1.3–4× (`0.039–0.115`), at `4`
   more. Step 5's layer on the exact solution at the grid rate ends at
   `0.181` and `0.160`.
3. **Finding 1's `n_L ≳ G (10 ρ_max M)^{1/3}` is confirmed for `ρ_max M ≥ 4`
   and is not sufficient below it (measured in step 8c).** It asks for `6.8`
   and `9.3` cells at `4/M` and `10/M`; the scan's thinnest ramp that holds
   is `8` and `12`, and one row thinner costs 2–4× in the shell (`n_L = 6` at
   `4/M`, `8` at `10/M`). At `1/M` it would allow `4.3` cells, and the
   ramps of 4–6 cells are 2.5× the best while the 12-cell one fails from
   the inside: the rule is the prediction **and** `ρ_max ≳ 4/M`. Stated for
   other orders and spacings — `n_L = max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)` cells
   at `ρ_max = 4/M`, `4/M` being `16 κ` — it is **(proposed in step 8c)**:
   one fixture, one order, one spacing measured it.
4. **`ε_KO` stays constant for a smooth target (measured in step 8c).** The
   profile below, raised across the margin as step 8a recommended, makes the
   shell *worse* in proportion to `ε_in` — at `n_L = 12`, `4/M`: `0.029`,
   `0.032`, `0.045`, `0.065` for `ε_in = 1/2` (constant), `1, 2, 4` —
   and changes no survival where the ramp is right; where it is wrong (the
   grid rate at `n_L = 8`) it delays the end from `15 M` to `22 M`. It is
   what keeps a *hard step* alive (E0, above), so it stays built, as the
   remedy for a target that can jump, and off by default **(proposed in step
   8c)**.
5. **The tracking error the layer absorbs is about `h` (measured in step
   8c).** E2 at `δ = h` is the exact target's `0.029` at both good ramps;
   at `δ = 4h = 0.31 M` only `n_L = 8`, `4/M` holds near the rule's error
   (`0.041`), `n_L = 6` at `4/M` and `8` at `10/M` reach `50 M` at 3–4× it
   (`0.160`, `0.123`), the others end in `1–6 M`, and at `n_L = 12` the
   displaced singular point
   lies *on a grid point of the layer* and the run throws at `t = 0` — a
   target's singular set must be inside `r_0` exactly as the background's
   must, and nothing checks it. Step 8d's tracked center has to be good to
   a cell for this layer to be the rule's **(proposed in step 8c)**.

`r_0` is chosen where the analytic
solution is still moderate — `|h| ≲ 10`, a fraction of the horizon
radius — so that `u_exact` and `F(u_exact)` are well within range
throughout the layer; inside `r_0` the analytic solution may be
singular and is never evaluated. `F` is **not evaluated where `w = 0`**:
the kernel branches on the core predicate before touching the stencils,
because the core holds finite but arbitrary data on which `F` may be
`NaN`, and `0 · NaN = NaN`.

**The frozen core** holds finite data and is never read: the
initial-data callback fills it with the analytic solution evaluated on
the sphere `r_0` along the ray (continuous at `r_0`, finite everywhere),
`du = 0` there, nothing else writes it, and the regrid transfer
interpolates it like any other data. Every norm, monitor and error
kernel — and the refinement indicator — masks the whole interior
`r < r_1`; the modified region is not a numerical solution and must not
be reported as one or refined for its own sake. The apparent horizon
lies outside `r_1` by the margin, and so must the interpolation
footprint of the horizon finder (checked).

**(Implemented in step 5.)** Three things the writing of the core rule
settled, each stated where it is made in `src/interior.jl`:

- **At the center the ray is undefined and `+ẑ` is taken
  (proposed in step 5)**. It is not an arbitrary tie-break: the harmonic
  chart is singular on the disk `z = 0, x² + y² ≤ a²` and regular on the
  axis, so the axis is the direction whose value is safest to smear over
  a ball.
- **The rule is applied by *every* path that writes the analytic solution
  onto the grid**, not only by the initial data: the error reference, the
  `:pasted` limiter, the Dirichlet hook (where it is the identity, the
  core being nowhere near the boundary) — and the **gauge-source
  sampling**, which is where leaving it out bit. `H^a = −Γ^a[g_exact]` is
  sampled at every owned point including the core, and at the center
  `KerrSchild`'s `k^i = (…, z/r)` divides by zero. The right-hand side
  never reads it there, because the kernel branches on the core first;
  the constraint monitors do, and what they then report is `NaN`
  (fixed in step 5).
- **A masked slot is written through a branch, not multiplied by zero
  (fixed in step 5).** Step 4's kernels wrote `keep * value` with `keep`
  a `1`/`0`; that is the same number for every mask that exists when
  nothing is masked, and it is not the same number once the interior is,
  because `0 · NaN = NaN`. This is `CLAUDE.md`'s trap met in the
  monitors rather than in the right-hand side, and it is the one place
  where a correct-looking multiplication had to become an `if`.

**Three variants, one switch** (G4 measures all three on the static
hole, the first two on the moving one): `:damped` — `(INTERIOR)` as
above, the default **(proposed)**; `:pasted` — the hard paste through
`step_limiter!`, `w = 0` and `du = 0` inside `r_1`; `:frozen` — the pure
mask, `ρ = 0`. **(predicted)** `:damped` and `:pasted` both hold the
static hole to `t = 50 M` with constraints at truncation outside `r_1`,
`:damped` with the smaller violation in the `G` points outside `r_1`;
`:frozen` holds the static hole only with `ε_KO ≈ 0.5` and a wide ramp,
with a growing layer of compressed features at the freezing radius
whose amplitude the dissipation may or may not saturate; on the moving
hole `:frozen` fails and the other two agree.

#### The range projection

**The range projection: the third and last writer of the state (added in
step 8b**, `src/bounds.jl`**).** Every interior treatment above is a
*source* of states the equations cannot continue from — step 5 measured
two of the three ending in "`√(det γ)` of a state that is no longer a
metric" — and the generic interior of steps 8c–8e will be more so, since
its target is no longer an exact solution. So there is an instrument that
catches such a state where it first appears, repairs it minimally, and
says when and where it did: a pointwise map of `(h, Π)` at every owned
point with `r < r_gate`, installed as RK4's **stage limiter**,

 1. a **non-finite** component takes its Minkowski value, `0`, and is
    counted separately — it is a different failure;
 2. the **spectrum of `γ_ij`** is clamped into `[λ_min, λ_max]` by a
    symmetric eigendecomposition (Jacobi; a rescale to `det γ ≥ δ` would
    not do, since two negative eigenvalues have a positive determinant);
 3. the **shift** `|β| = √(β_iγ^{ij}β_j)`, with the projected `γ`, is capped
    at `β_max` by scaling `β_i`;
 4. the **lapse** `α² = β_iβ^i − g_tt` is clamped into `[α_min², α_max²]` by
    moving `g_tt` alone;
 5. the **momentum** is rescaled by `(α/α′)(√γ′/√γ)` where the lapse was
    raised from a positive value — so that `(α/√γ)Π`, the term the first
    evolution equation adds to `∂_t h`, is unchanged — and its scale is
    capped, `max_ab |(α′/√γ′)Π_ab| ≤ K_max`;

and `g′ = (−α′² + β′·β′, β′_i, γ′_ij)` reassembled **only in the blocks
that moved**. It is TreeHydro's atmosphere reset in shape (a pointwise map
in the integrator's limiter hook, written back only where it fired,
idempotent, counted, with a bitwise control) and deliberately *not* in
substance, for two reasons that are both findings of the design review:

- **Not a reset to a fixed state.** A metric that has left the range of
  metrics is not vacuum dust: typically one quantity is wrong — a
  spectrum through zero, a lapse through zero — and the rest is the
  solution. Replacing the point would put an `O(1)` discontinuity into
  the data a layer stencil reads (Kerr-Schild's `|h|` is about 5 at the
  core's edge, against `0` for any fixed state), and the map is built to
  move the offending quantity and leave the others their bits.
- **Not a clamp per component of `h_ab`, and not a projection onto flat
  space**, because the Lorentzian metrics are not convex in `g_ab`: the
  angular mean of Kerr-Schild `g_ab` on the sphere `r = 1.15 M` has
  `g_tt = +0.74`, a Euclidean signature (`PLAN.md`, finding 2), so a box
  in `h_ab` contains non-metrics and excludes metrics. The ranges are
  stated in ADM variables, where `α > 0` and `γ ≻ 0` are convex, and each
  clamp is the nearest point of its own range.

**Three writers, and no fourth**: the right-hand side never mutates `u`
(the integrator's arithmetic is the first writer); the `:pasted` paste of
the ball `r < r_1`, from the step limiter, is the second; the range
projection, from the stage limiter, is the third. **(Amended in step X2b:**
an `:excised` hole has neither — its step limiter is a no-op and a case
with it refuses `bounds`, since a clamp would guard a set nothing reads.**)** Both limiters are
integrator keywords (`stage_limiter`, `step_limiter`); under
IMEXRungeKutta (from 2026-09-26) **both writers are one limiter,
`gh_limiter!` — the projection, then the paste — passed as the stage and
the step limiter alike** (decided 2026-09-26 by Erik: the limiter applies to
every state vector), so every stage value the right-hand side reads and
every step's result is projected and pasted — see [Time
integration](#time-integration). The projection also runs
once on the initial data and once after every regrid transfer, before the
paste, which is the order RK4 applies the two — neither state went
through a stage, and a prolongation into a fresh fine block is unlimited
(TreeHydro's reason for the same call) **(proposed in step 8b**, for the
initial data**)**.

**Why a stage limiter and not a step limiter**: what it guards against is
`F` evaluated on a stage vector that is not a metric, and a `NaN` in a
stage is a `NaN` in the next stage's `F` at every point whose stencil
reads it. Read against `OrdinaryDiffEqLowOrderRK` 2.2.5, RK4 calls the
stage limiter on its three intermediate stages and then on `u`, before
the FSAL evaluation and before the step limiter: four calls per step.
**(Amended 2026-09-26.)** Under IMEXRungeKutta's RK4 the same four
projections happen, each followed by the paste: three as the stage limiter,
on the stage values the right-hand side reads, and the fourth as the step
limiter, on the result.

**Where: the gate.** The projection's output is a clamp, and a clamp is a
kink wherever it fires; a kink an evolved stencil reads is an `O(1)`
right-hand-side error outside the layer. So it runs only at `r < r_gate`,
and `check_bounds_gate` asserts at every regrid, beside the interior's
radius checks, that `r_gate ≤ r_1 − R h` with `R` the stencils' Euclidean
reach in spacings (`G` for `q ≤ 4`, the mixed derivative's `√2 · q/2` from
`q = 6`). The proposed gate is **`r_gate = r_1 − 2 G h`** — twice the reach
— **(proposed in step 8b**, until step 8a's leakage margin exists to
replace the factor 2**)**; on the suite's fixture that is `0.8375` at
`h = 5/64`, with `8.4 %` of the owned points inside it. The outer part of
the layer, `r_gate ≤ r < r_1`, is unguarded by construction.

**The proposed ranges** (`default_bounds`; a `StateBounds` has no default
for any of them and `hole_case` has none beyond `bounds = nothing`)
**(proposed in step 8b)**: `α ∈ [1/50, 50]`, `λ(γ) ∈ [1/100, 1000]`,
`|β| ≤ 10`, `K_max = 100/M`, against the step-5 fixture's deepest data —
Kerr-Schild at `r_0 = 2/5 M`, where `α = 0.41`, `λ = (1, 1, 6)`, `|β| = 2.04`
and `max |(α/√γ)Π| = 10.4/M` — so that a healthy interior never fires.
Flat space must be inside every range, and the constructor refuses one
that excludes it: the non-finite repair writes Minkowski's components,
and a repair the next check moves again is not idempotent.

**Idempotent on the state and on the flag (measured in step 8b).** Every
test carries a slack of `8 eps` of the scale of the terms it is made of —
`‖γ‖` for the spectrum, `1 + |h_tt| + β_iβ^i` for `α²`, of which it is a
difference — and every clamp moves to its bound exactly. TreeHydro found
its floor's *state* a fixed point and its *flag* not; here the slack alone
was not enough either: `β_iγ^{ij}β_j` carries `cond(γ)·eps` of rounding, and
on 20 000 random states with `|h_ab| ≤ 5`, `|Π_ab| ≤ 100`, **18 re-fired**
on the shift (402 at `|h_ab| ≤ 50`), moving the state by ulps. So a fired
result is re-tested by exactly the arithmetic the next call will apply
to the stored state — the ADM split is explicit scalar code, which Julia
does not contract into fused multiply-adds, so it is the same bits at
every call site — and re-projected until it passes (at most four times).
With that, **zero** of 80 000 random states re-fire or move on a second
application, at every scale.

**Where nothing fires, the run is bit for bit the run without it.** That
is the control the suite asserts, and it is what the "written back only
where it fired" rule buys: the fixture's `:damped` run to `3/20 M` with
the projection on makes `4 · nsteps + 1` limiter calls, fires on none, and
ends in a state `isequal` to the run without it — so every comparison of
a run with the projection against one without compares the projection and
not roundoff.

**What it does on the two runs that end** is measured under [Measured
results](SINGULARITY_HANDLING.md#the-range-projection-step-8b): step 5's
`N = 6` `:damped` and
`N = 8` `:pasted`. **It never fires on either (measured in step 8b)**: both
degenerate in the evolved shell just outside `r_1`, where no projection
gated below `r_1` reaches and none should, and the runs with it end bit for
bit where the runs without it do. The prediction that hits would start
"deep, several `M` before the crash" is wrong; the failures are surface
failures at `r_1`, which is step 8c's hypothesis about the layer's
transition, measured from the other side.

#### The tracked geometry

**The tracked geometry (added in step 8d**, `src/tracking.jl` and the
second half of `src/interior.jl`**).** Everything above is keyed on
`r = |x − c(t)|` about the *analytic* center. From step 8d a case may
instead key its layer on the **found** horizon, `PLAN.md`'s finding 3: with
`r_h(n̂)` the tracked apparent horizon's coordinate radius along the unit
vector `n̂` from the *tracked* center `c(t)`, the layer is a function of the
**depth**

    d = r_h(n̂) − m h − |x − c(t)|,        offset surface r_1(n̂) = r_h(n̂) − m h,

below the offset surface — `d ≤ 0` evolved, `0 < d ≤ n_L h` the layer
(core surface `r_0(n̂) = r_1(n̂) − n_L h`), and beyond it the frozen core.
The sphere is the `l = 0` case, and **a tracked geometry holding step 5's
sphere is step 5's layer bit for bit (measured in step 8d)**: one
right-hand side on the fixture is `isequal` for `:damped`, `:frozen` and
`:pasted`, and so is the paste, because the profiles are one function
(`_layer_profiles`) handed the two surfaces' radii along the ray, and the
predicates are the sphere's in those radii — a point on the offset surface
is evolved and one on the core surface is in the layer, as for the sphere.
Six pieces, in the order a run meets them:

1. **What the case holds is a rule, `FittedSpec`**, not a layer: the
   variant, `m` (`margin = 8`, step 8a's default — one `m` for now, though
   finding 3 already says the harmonic equator wants `4` and the axis more),
   the ramp `n_L` (`0` meaning step 8c's rule `max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)`,
   resolved by `evolve!` at the scheme's `G` and the run's rate — `8` at
   `q = 2`, `12` at `q = 4` — and at the default `4/M` when the rate is the
   grid's, whose value changes per chunk **(proposed in step 8d)**),
   `core_min = 2`, `lmax_shape = 4`, `ρ_max` (`0` = the driver's default),
   `max_misses = 3`, `α_trigger = 1/10` and a `target`. **Its ramps are
   step 8c's rule, `ρ_ramp = 1`, `w_ramp = 1/2`**, since `n_L` was
   calibrated as the width over which `ρ` rises; step 5's `1/2, 1/2` is the
   fixture's **(proposed in step 8d**; the brief said `1/2` for both**)**. A
   `FittedSpec` case needs a `Horizon` with `every ≥ 1`, and `evolve!`
   refuses one without, saying why: a track that is never updated is the
   analytic seed carried along forever.
2. **The track, `HorizonTrack`** — host-side and immutable: the last find's
   time `t_find` and recentred origin `c_find`, a velocity `v_est`, the found
   surface's radii `r_min`, `r_max` **about its own origin**, its shape as
   real coefficients, the finder's `hlm` and grid for the next seed, and
   `source ∈ {:analytic, :found, :coasting}`, `misses`, `nfinds`.
   `seed_track` starts it from the case's analytic center, velocity, radii
   and shape. `update_track` replaces it on a successful find (the velocity
   becomes the two last finds' difference once there is a previous find, and
   stays the analytic one after the first); on a failed find it **coasts** —
   everything kept, `misses + 1`, `source = :coasting` — and at `max_misses`
   consecutive misses it throws a `TrackLostError` saying how old the
   geometry is. **A find that moves `r_min` by more than half a stencil
   reach `G h/2` is refused** (an `ArgumentError` naming both radii and
   `G h`): the layer is an offset of that surface, and a jump exposes, on
   the side it moves away from, points no layer ever treated **(proposed in
   step 8d)**. `track_center(tr) = HoleCenter(c_find − v_est t_find, v_est)`
   is all a kernel sees of it — so the masks, `interior_radius`, the range
   projection's gate and the core rule work on the tracked trajectory
   unchanged, and the center is still a function of `t` and never a mutated
   field.
3. **The kernel argument, `FittedInterior`** — built by `fitted_interior`
   once per chunk and after every regrid from the track and the mesh:
   the center, the shape as an `SVector` of real coefficients to
   `lmax_shape`, its bounding radii `r_in`, `r_out`, `offset = m h`,
   `thickness = n_L h`, the rate, the ramps and the target, and the `Val`
   of the variant — so the right-hand-side kernel's `Val{INT}` is
   `:damped`, `:frozen` or `:pasted` for both geometries. **`h` is the
   coarsest spacing of the blocks the layer lives in**, the annulus
   `[r_in − (m + n_L) h, r_out − m h]`, by one iteration from the finest
   spacing — `PLAN.md` asked for the annulus out to the horizon itself, and
   on the suite's fixture the horizon at `2 M` is in blocks twice as coarse
   as its layer's, where a spacing read there asks for a layer that does not
   fit (`18 h = 2.8 M`); so `h` is measured where step 5's `layer_spacing`
   measures the sphere, and what the margin buys along its real path is
   reported in e-folds, below **(proposed in step 8d)**. `fitted_interior`
   refuses `r_in − (m + n_L + core_min) h ≤ 0` by name, and
   `check_interior_radii` asserts the result — `offset ≥ m h` and
   `thickness ≥ 2(G+1) h` at the layer blocks' spacing, which is step 5's
   `r_1 ≤ r_h,min − m h` with **the track's radii** in every direction, and,
   for the analytic-target variants (all three), `r_in − offset − thickness
   > singular_radius + |c(t) − c_analytic(t)|`: the core rule still
   evaluates the analytic solution on the core surface, so that surface
   must still contain the chart's singular set, and that set is a ball about
   the *analytic* center. The check is spherical — the core surface's
   *smallest* radius against the set's largest — and so conservative for an
   oblate core around a flat disk: on harmonic Kerr at `a = 9/10` it is
   `0.436 − (m + n_L) h` against `0.9` and fails at every `h`, and even the
   exact statement, the equatorial core radius `1 − (m + n_L) h` against
   `0.9`, would need `(m + n_L) h < 0.1 M`, sixteen cells at `h ≤ 1/160` at
   `q = 2`. **The tracked geometry does not unblock the proof-of-concept
   case with an analytic target**; step 8e's `:fitted` target, which needs
   no analytic interior and so no singular set inside the core, is what
   does.
4. **The shape: real spherical harmonics in `ash_mode_index`'s slots.**
   `ỹ_l0 = Y_l0`, `ỹ_lm^c = √2 Re Y_lm`, `ỹ_lm^s = −√2 Im Y_lm` for `m ≥ 1`,
   `Y_lm` the orthonormal Condon–Shortley harmonics of
   `AbstractSphericalHarmonics`; the real coefficient of `(l, m)` is in slot
   `l² + l + m + 1` with `m ≥ 0` the cosine (and `l0`) and `m < 0` the sine
   of `|m|`, and a real function's complex coefficients convert as
   `a_l0 = Re c_l0`, `a^c = √2 Re c_lm`, `a^s = √2 Im c_lm`
   (`real_from_complex`, `complex_from_real`, `real_harmonic_index`)
   **(proposed in step 8d; step 8e's fit shares the ordering and the
   conversion)**. The finder's `hlm` is truncated to `lmax_shape` by
   `ash_resample` first. The kernel evaluates the series
   (`shape_series`) with **no angle**: `Y_lm = q_lm(cos θ) sin^m θ e^{imφ}`
   with `q_lm` the fully normalized associated Legendre function over
   `sin^m θ`, and `sin^m θ e^{imφ} = (n_x + i n_y)^m` — the Chebyshev
   recurrence for `cos mφ, sin mφ` multiplied through by `sin^m θ`, which
   removes both the `atan` and the division by `sin θ` that would need a
   guard on the axis. Runtime loop bound `lmax`, no allocation, generic in
   `T`. **The conversion is `sYlm`'s to roundoff (measured in step 8d)**:
   at 200 random directions the largest error is `3.7` and `6.4` eps of the
   function at `lmax = 4, 8` at `Float64`, `1.6` and `5.3` eps at `Float32`;
   on `EquiangularGrid(6)` it is `ash_evaluate`'s to the same.
   **The surface is the series clamped into `[r_in, r_out]`** — the extremes
   over both poles and a `(4 lmax + 3) × (8 lmax + 6)` grid of directions —
   which makes the two fast paths **exact shortcuts rather than
   approximations** (proposed in step 8d): outside `r_out − offset` a point
   is evolved and inside `r_in − offset − thickness` it is frozen in every
   direction, and neither evaluates the series.
5. **The seed is the analytic horizon**, `r_h(θ) = R √((R² + a²)/(R² +
   a² cos²θ))` — `R = r₊` for Kerr-Schild, `√(M² − a²)` for the harmonic
   chart; both charts' spheroid of constant radial coordinate, checked
   against their quartic at random directions, since `SpacetimeMetrics`
   exposes no horizon — turned for `rotate`, contracted along `v` by
   `√(1 − v²)` for `boost` (`analytic_horizon_radius`), and **sampled at
   `L = max(4 lmax + 4, 32)` and truncated**, the spectral projection,
   rather than interpolated at `lmax`'s own points, which would alias the
   spheroid's higher multipoles into the kept ones **(proposed in step
   8d)**. **What the truncation costs (measured in step 8d)**, the largest
   radius error over directions: **Kerr-Schild `a = 9/10`: `1.2e−3` at
   `lmax = 4`, `8.1e−6` at `8`** (`5.5e−8`, `3.8e−10` at `12`, `16`) — the
   axis and equator and `r_in`, `r_out` against `horizon_min_radius`,
   `horizon_max_radius` to the same; **harmonic Kerr `a = 9/10`: `4.6e−2`,
   `7.2e−3`, `1.1e−3` and `1.7e−4` at `lmax = 4, 8, 12, 16`** — `2.4`,
   `0.37`, `0.057` and `0.009` cells at `h = 5/256` — since its spheroid is
   oblate in the ratio that sets the multipoles' decay (`a²/R² = 4.3` against
   `0.39`). The default `lmax_shape = 4` is Kerr-Schild's (`0.06` cells at
   `h = 5/256`); harmonic Kerr at `a = 9/10` needs `lmax = 12` for a tenth of
   a cell, which step 8f's rows should carry. **(Amended in step 8:)** the
   seed's `r_min` of a **moving** hole is the least of the analytic surface
   over `shape_sample_directions` (both poles included), not
   `horizon_min_radius`'s `√(1 − v²)` bound, which is exact only when the
   smallest radius lies along the boost: G5's hole has it on the spin axis,
   `0.714`, which a boost along `x̂` does not contract, where the bound says
   `0.681`, and the first find (`0.716`) was refused as a jump of `0.034`,
   over `G h/2` at `h = 5/256` (measured in step 8). A hole at rest keeps
   the bound, which is then exact, bit for bit.
6. **The masks and the guard.** `ShapeMask` (`d ≤ 0` evolved, the same fast
   paths) is what `interior_mask` returns, so every masked norm, the speed
   kernel and the indicator exclude exactly the region the kernel modifies;
   `ShapeBand` is the band `r_1(n̂) + lo ≤ r < r_1(n̂) + hi`, which
   `layer_mask` and `shell_mask` return for the validity monitor and the
   variants' shell — the sphere returns step 8b's `ShellMask`s, value for
   value. **The footprint guard is exact by enumeration** where the shape
   can matter: a footprint whose nearest lattice point is outside
   `r_out − offset` passes, one inside `r_in − offset` is refused, and in
   between each of the `(q+2)³` points is classified by the mask itself
   **(proposed in step 8d**, over the brief's "use the bounding sphere
   `r_in − offset`", which would let a footprint read the layer's outer part
   wherever the horizon is farther than its smallest radius — on harmonic
   Kerr's equator by more than the whole margin**)**; 2000 random
   footprints about an `a = 9/10` shape agree with the definition.

**The protocol both geometries speak (amended in step 8d).** `in_layer(int,
t, x)` takes a position and a time and not a radius, `interior_point(int, t,
x)` is what the kernel's three predicates (`is_frozen`, `is_outside`,
`interior_profiles`) are evaluated on — the radius for the sphere, the
radius with the two surfaces' radii along the ray for the tracked geometry —
`geometry_radii(int, background)` is the horizon's smallest and largest
radius the layer is placed inside (the analytic ones, or `r_in`, `r_out`),
`layer_radii(int)` the layer's innermost radii, and `layer_mask`,
`shell_mask` the two bands; `find_gh_horizon` takes `center =`, the point
its radii are measured from, and returns the radii about its own origin as
`origin_r_min`, `origin_r_mean`, `origin_r_max`.

**An excised tracked geometry is frozen (amended in step X2b).** For
`FittedSpec(…; variant = :excised)` the geometry is built once, from the
seed's shape at `t = 0` on the run's mesh, and never rebuilt: the finds
still update the track every chunk, for the record and for the row
`excision_horizon_margin`, which ends the run below `m − G/2` cells
([Excision](#excision-added-2026-10-05), "What step X2b built").

**Per chunk, in the driver.** The initial data goes through the core rule
of the seed's geometry (`fill_exact!(…; interior)`); at every row the find
starts from the track's prediction `c_find + v_est (t − t_find)` and
measures its radii from there, so the row's `center_offset` is **the
track's prediction error**; `update_track`; the next chunk's geometry from
the updated track on this mesh, asserted like the sphere; after a regrid,
the same track on the new mesh. Four decisions the loop needed, each
**(proposed in step 8d)**:

- **The lapse-collapse trigger** (the design review's idea 10): the
  validity monitor now also reduces over the whole evolved region
  (`min_detγ_evolved`, `min_α_evolved`), and a row whose `min_α_evolved` is
  below `α_trigger` forces a find at the next chunk boundary whatever the
  cadence; that row records `track_trigger = true`.
- **A lost track ends the run with its record.** `evolve!` records the row
  of the last miss, calls the observer, and throws the `TrackLostError`
  carrying the record — step 7's reason for recording a failed find rather
  than throwing it, applied to the one failure that has to end the run.
- **The velocity estimate's error is recorded, not refused.** The row's
  `track_prediction` is the found origin's distance from the prediction in
  cells; the jump refusal is what guards the geometry, and step 8c measured
  a displacement of `h` costing the exterior nothing. A moving hole whose
  prediction error exceeds a cell per chunk is G5's to decide on.
- **The gauge source is re-sampled only when the core surface has moved
  half a cell** (`surface_shift`) since the sample it was taken with: the
  sample applies the core rule, so a point the moving core releases into
  the layer reads the source of its projection on the old surface — within
  the surface's movement of the true point, and weighted by a `w` that
  vanishes to second order there. On the static hole the surface moves by
  `10⁻⁴` a chunk and nothing is re-sampled.

**A boost moves the hole the other way (found in step 8d, fixed in step
8e).** `SpacetimeMetrics.boost(m, v)` evaluates `m` at `Λᵀx` with `Λ`'s
`+γv` entries, so the rest frame's origin is at `x = −v t` in the lab: the
metric of `boost(KerrSchild(1, 0), (0.3, 0, 0))` at `t = 1` is singular at
`x = −0.3`, not `+0.3` (measured). `HoleCenter`'s docstring called `v` "the
coordinate velocity of `boost(background, v)`", which is the wrong sign, and
a case built without a `velocity` keyword carried zero. **Fixed in step 8e**
by one dispatch beside `hole_mass`: `hole_velocity(background)` is zero for
`KerrSchild` and `Harmonic`, passed through `translate`, turned by `rotate`
(`R u`, since the rotated metric at `x` is the original at `Rᵀx`), `−v` for
`boost(m, v)` of a static hole and the relativistic composition `Λ(−v)` of
the inner hole's 4-velocity otherwise, and an `ArgumentError` naming the type
for a background it cannot classify. `GHCase` (and `hole_case`, whose default
damping profile moves with the hole) derives `velocity` from it when the
keyword is not given — zero for a background with no hole — and refuses a
keyword that disagrees with it by more than `8 eps`, stating both vectors and
the convention; `HoleCenter`'s docstring says `−u`. `test/interior_tests.jl`
asserts that the seed track of a `boost(Harmonic(1, 0), (0.3, 0, 0))` case is
at `(−0.3, 0, 0)` at `t = 1`, where the boosted Kerr-Schild metric is
singular (`|g_tt| = 1.0e17` there, `4.9` at `+0.3`), and that the former sign
is refused. No existing test, fixture or `hole_runs.jl` section passes a
`velocity`, and the one boosted case the suite builds (`gauge_tests.jl`'s
harmonic neighbour of the refused Kerr-Schild) asserts only that it is
accepted; nothing in G4 moves, so nothing measured changes. The analytic
shape's contraction does not depend on the sign.

`evolve!` also takes `find`, the function the horizon rows call —
`find_gh_horizon`, or a test's wrapper that disables it — because the
observer is called after the find and cannot reach it **(proposed in step
8d)**. The damping profile `γ0` and a `HorizonDissipation` stay on the
case's analytic trajectory: both are smooth on the scale of `M`, not of a
cell **(proposed in step 8d)**.

**What the margin buys, per row** (`margin_efolds`, moved into `src/` from
`test/dispersion.jl` as `PLAN.md`'s hand-over from step 8c asks): step 8a's
path integral `n_e = ∫_{r_1(n̂)}^{r_h(n̂)} ε dr/(h ℓ_max,1)`, along the six
grid axes from the tracked center — where the one-dimensional symbol is
exact — with the background's coefficients, `ε_KO` at each point and the
spacing of the block each point is in; the row carries the least of the
six. It reproduces the script's `1.81` and `2.07` e-folds for `m = 8` at
`h = 5/64` (`q = 2`, `4`) to the digits `CODE.md` records **(measured in
step 8d)**. On the suite's tracked fixture — `m = 10` cells of `5/64` from
`r_1 = 1.22`, of which the outer `0.75 M` lie in blocks of `5/32` — it is
**`1.45`**, where a uniform `5/64` would give `2.68`.

**What it costs (measured in step 8d**, Apple silicon, Julia 1.13, one
thread**).** The series is **`21 ns` per point at `lmax = 4` and `79 ns`
at `lmax = 8`**, against **`96 ns`** for one analytic `u_exact` of
Kerr-Schild `a = 9/10` (`background_state`, the dual pass the layer already
pays), and it is evaluated only between the bounding spheres. A
right-hand side on the fixture with the tracked geometry at `lmax = 4` is
**`1.1 %`** dearer than with the sphere, a `0.15 M` run **`5 %`** (four
threads: `3.16 s` against `3.00 s`), of which the host side per chunk is
`fitted_interior` at `0.04 ms` and `margin_efolds` at `0.6 ms`.

**What it measures on the static hole (measured in step 8d).** One find of
the fixture's initial data (`N_ah = 12`) puts the origin `1.4e−5 M` from
the analytic center — **`track_offset = 1.8e−4` cells** — with the found
surface's `r_min = 2.00043`, `r_max = 2.00068` about it against `r₊ = 2`.
A `:damped` run of the fixture on the tracked geometry to `0.15 M`, the
finder every chunk, against step 5's sphere with the same layer (`r_1 =
2 − 10h`, `n_L = 8`, step 8c's ramps): masked error **`3.110274e−3`
against `3.110271e−3`**, the shell's `C_a` L2 **`1.0745462e−2` against
`1.0745448e−2`**, the layer residual `0.203` against `0.206`, no
projection hit on either, the track at most **`3.3e−4` cells** from the
analytic center and its prediction error `1.2–1.8e−4` cells per find —
the tracked hole is the analytic one to six digits.

#### The fitted target

**The fitted target (added in step 8e**, `src/fit.jl`, with its kernel half
in `src/evolution.jl`, `src/constraints.jl` and `src/driver.jl`**).** The
tracked geometry still relaxes toward the
analytic solution, so its core surface must still contain the chart's
singular set, and harmonic Kerr at `a = 9/10` refuses that at every `h`.
The fitted target needs no interior at all: a **polynomial in `x`** —
regular everywhere, the center included — fitted by least squares to the
state (for 8e-ii's layer) or to the analytic solution (for its initial data)
on the offset surface `r_1(n̂) = r_h(n̂) − m h`, where the state is still the
solution, and continued inward. Twelve pieces: seven in the order
`build_fit` meets them (step 8e-i), and five for the variant that relaxes
toward it (step 8e-ii):

1. **The variables are ADM variables, never `g_ab`** (`PLAN.md`'s finding
   2): `(log α, β^i, γ_ij, Π_ab)`, 1 + 3 + 6 + 10 = 20 (`fit_variables`).
   `α = exp(log α)` is positive by construction and `γ` is convex, where a
   least-squares fit — a weighted mean — of `g_ab` is not a metric: **the
   angular mean of Kerr-Schild `a = 0`'s `g_ab` on the sphere `r = 1.15 M`
   has `g_tt = +0.739` and the eigenvalues `(0.739, 1.580, 1.580, 1.580)`,
   Euclidean signature, a signed lapse of `−0.86`, while the same mean of
   the fit's variables reassembles into a metric with `α = 0.60`,
   `det γ = 3.94` (measured in step 8e**, the control the suite asserts).
   **The shift is the contravariant `β^i` (proposed in step 8e)**: either is
   valid, but `β^i` is the split's own shift and reassembles without an
   inverse — `β_i = γ_ij β^j`, `h_tt = (1 − α²) + β^iβ_i` (`state_from_fit`)
   — where `β_i` would need `γ^{ij}` at every layer point. `γ_ij` is held as
   its offset `γ_ij − δ_ij` (the same fit, with near-flat digits kept) and
   `Π_ab` as it is. **The samples' radial derivatives go through the chain
   rule** (`β′ = γ⁻¹(b′ − γ′β)`, `(log α)′ = (α²)′/2α²` and their second
   derivatives, `b_i = h_ti`) **for both samplers (proposed in step 8e)**,
   checked against differencing the converted samples along the same rays:
   they agree to `5.7e−10` in the slopes and `7.3e−11` in the curvatures on
   the fixture's surface (measured in step 8e).
2. **The ansatz** is `f_v(x) = Σ_{lm} S_lm(ξ) Σ_{k=0}^{cont} C_{lm,k,v}
   ρ^{2k}`, `ξ = (x − c(t))/r̄`, `ρ = |ξ|`, `r̄` the mean of `r_1(n̂)` over the
   collocation points, `S_lm = ρ^l ỹ_lm` the real **solid** harmonics in
   8d's slots and normalisation — a polynomial of degree `L + 2 cont` in
   `x`, `L = lmax_fit = 8` by default (a new `FittedSpec` field), `cont = 1`
   (values and slopes, the evolved state's fit) or `2` (and curvatures, the
   initial data's). The harmonics are `shape_series`'s recurrence with `ξ`
   for the unit vector and `ρ²` multiplying the Legendre step's `q_{l−2}`
   (`_solid_harmonic_fold`): no angle, no division, and at the center only
   `S_00` survives; at unit vectors they are `shape_series`'s harmonics to
   `16 eps`. **The shift has no `ρ⁰ ỹ_00` term by default**, so `β(0) = 0`,
   as `PLAN.md` asks; its `ρ² ỹ_00` and `ρ⁴ ỹ_00` terms, which also vanish at
   the center, stay, since `l ≥ 1` alone cannot hold a shift's `l = 0` part
   on the surface at all **(proposed in step 8e)**. **That default is wrong
   for a moving hole (measured in step 8e):** a boosted hole's shift has an
   `l = 0` part on its offset surface — the angular mean of `β^x` is `0.17`
   against a mean `|β|` of `0.63` for `boost(KerrSchild(1, 0), 0.3 x̂)` on
   `r = 1.22`, `0.28` against `0.36` for the harmonic chart on `0.69` — which
   `ρ² ỹ_00` matches in value and cannot in slope: the shift's slope rows
   are left `2.0` (Kerr-Schild) and `22` (harmonic) times their own scale,
   and the state on the surface `1.3e−2` and `1.1e−3` off. `shift_constant =
   true` fits the shift's constant like every other variable's: `1.9e−5`
   and `1.7e−5`, `|β(0)| = 0.19`, `0.29`, where the boost puts it — and on
   the static hole the same fit to roundoff. Validity never needed
   `β(0) = 0` (any shift with `α > 0`, `γ ≻ 0` is Lorentzian), so
   **`shift_constant = true` is the default (decided in review, step 8e)**:
   the shift keeps its `l = 0` constant, and `false` is the brief's ansatz,
   kept as a switch; on the static hole the constant comes out `6.2e−14` of
   `|β|` (measured in step 8e).
3. **The samplers** are called as `sampler(xs, ns)` — the points *and* the
   unit rays, since the radial derivative is along the ray from the tracked
   center, which a sampler does not know **(proposed in step 8e**, over the
   brief's `sampler(xs)`**)** — and return `(u, ∂_r u[, ∂_r² u])` as vectors
   of packed `(h, Π)`. `state_sampler(fs, q; t)` interpolates through
   `interpolate_grad` with **the footprint guard off, `mask = AllPoints()`
   at the call**: the collocation points are on the evolved region's
   boundary, so their windows read `G h` into the layer, where step 8c's
   `n_L ≥ 4G` holds the relaxation to `ρ_max · smoothstep(1/4) = 0.10
   ρ_max`, `0.41/M` at the default — the layer's outer quarter is the
   equations plus a weak pull toward a fit of this same state **(proposed in
   step 8e)**. It gives no second derivative, so it serves `cont = 1`, and
   `cont = 2` with it is refused. `analytic_sampler(bg, t; δ)` evaluates
   `state_tuple` at five points along the ray and differences them at
   **fourth order, `δ = h/8` (proposed in step 8e**, the brief's central
   differences at the order that puts the truncation at `δ⁴`**)**: its `∂_r
   h` is the analytic gradient's to `1.6e−8` on the fixture (measured in
   step 8e); the second difference's roundoff `5 eps |u|/δ²` is `10⁻¹¹` at
   `Float64` and `1.7e−2` at `Float32`, so a `Float32` run's initial-data
   fit wants its samples in `Float64`. Every sample is checked finite and a
   metric: a singular point on the offset surface is a configuration error
   and refused by name — which is what harmonic `a = 9/10` at `m = 8`,
   `h = 5/256` is, its equatorial offset radius `0.843` inside the disk.
4. **One least-squares solve** (`solve_fit`): rows for the values, the
   slopes `r̄ ∂_r` and the curvatures `r̄² ∂_r²` at the `(L+1)(2L+1)` points of
   `EquiangularGrid(L)`, columns `(L+1)²(cont+1)`, and **one Householder QR
   for all twenty right-hand sides**: the constant column is ordered last,
   so the shift's design — the others' without it — is the leading block of
   the same factorization, and its solution is the least-squares one.
   **Each block of rows is weighted `P^{−b}`, `P = L + 2 cont` (proposed in
   step 8e)**, the growth of a degree-`P` polynomial under `∂_ρ`: it cuts the
   condition number 4–20× (`85.6` and `1.4e3` at `L = 8`, `cont = 1, 2` on
   the fixture, against `377` and `2.9e4` unweighted) and the value residual
   2.5–8× on the spinning holes, and changes no consistent system's solution.
   **The residual is block by block (proposed in step 8e)** — each block's
   worst over the largest datum of *that block* in the variable's group
   (`log α`, the shift, `γ`, `Π`), so a steep curvature cannot hide a poor
   value; `fit_residual(fit).overall` is the worst. BLAS's threads are not
   `julia -t`'s, so a fit is the same at every Julia thread count; nothing
   sets either.
5. **The validity sweep** (`fit_sweep`) evaluates the fit **raw** —
   reassembled, not projected — along every collocation ray at `s r_1(n̂)`,
   `s ∈ {1/8, …, 1}`, and at the center, `8N + 1` points (`1225` at
   `L = 8`): **fractions of each ray's own `r_1` (proposed in step 8e)**, so
   `s = 1` is the collocation surface and the sweep covers exactly the
   region the fit is used in, oblate or not. A point is valid when finite,
   `γ` positive definite (`sym_eigen3`) and the signed lapse positive —
   **Lorentzian-ness and not a range (decided in review, step 8e)**; the
   sweep records `fit_valid`, the worst `det γ`, `α` and `λ(γ)`, `|β(0)|`,
   and how many points `bounds_project` would move into the target's
   ranges (a count, not a verdict), and `build_fit`
   **throws** with those numbers and the remedies — lower `lmax_fit`,
   `cont = 1`, a wider margin — unless `check = false` asks for the fit back
   with `valid = false`.
6. **The fit and its evaluator.** `InteriorFit` holds the coefficients
   `((L+1)², cont+1, 20)` in `T` through `to_backend` (`Hsrc`'s precedent)
   and on the host, the `isbits` `FitParams` (`L`, `cont`, `r̄`, the track's
   `HoleCenter`, the `StateBounds`), the time and center it was built at,
   the collocation points and the model's own values there, the residual,
   the conditioning and the sweep. `fit_variables_at(params, coeffs, x, t)`
   and `fit_state` — the first, `state_from_fit` and `bounds_project` into
   the fit's bounds as the guarantee — are `@inline`, allocation-free
   (asserted), generic in `T`, one `return`: 8e-ii's kernel calls them. At
   the collocation points they are the least-squares model to `28 eps`
   (`Float64`) and `25 eps` (`Float32`) of the largest variable, the
   projection is the identity on them bit for bit, and at the center `β = 0`
   and the metric is valid (measured in step 8e). **The ranges it projects
   into are the target's own (decided in review, step 8e)**: `FittedSpec`'s
   `target_bounds`, or — `nothing`, the default — `derive_target_bounds` of
   the analytic data on the seed's offset surface at `t = 0`: each upper
   range four times the largest value found there (`α`, `λ(γ)`, `|β|`,
   `|(α/√γ)Π|`), each lower one a quarter of the smallest, widened to keep
   flat space inside **(the factors proposed in step 8e)** — because step
   8b's `default_bounds` are Kerr-Schild's, and harmonic Kerr at `a = 9/10`
   has `|(α/√γ)Π| = 366/M` on its offset surface against their `100/M`. On
   the fixture that is `α ∈ [0.154, 2.46]`, `λ ∈ [1/4, 10.6]`, `|β| ≤ 4.04`,
   `K ≤ 3.35`; on harmonic `a = 9/10`, `α ≥ 0.066`, `λ ≤ 774`, `K ≤ 1464`.
7. **What it measures (measured in step 8e**, on analytic data unless
   said**).**
   - **Kerr-Schild `a = 0`** on the fixture's tracked geometry (`m = 10`,
     `n_L = 8`, `h = 5/64`, the offset surface at `1.22`): the fit's
     variables have `l ≤ 2` structure on a sphere, so **`L = 2` is what it
     takes** — the state on the surface is `1.10` off at `L = 1` and within
     `5.1e−15`–`3.1e−13` of the analytic one at `L = 2, 4, 8`, both orders.
     From **the state sampler** on a field set holding the exact solution
     it is off by the interpolation: **`2.2e−4` at `N = 8` and `8.3e−6` at
     `N = 16`, a rate of `4.65`** (`3.9` from `16` to `32`, measured once),
     below the samples' own `2.6e−4` and `9.2e−6` — the fit adds nothing to
     the interpolant's error. Every sweep is valid: `min λ(γ) = 1`, `min α =
     0.527` (`cont = 1`) and `0.480` (`cont = 2`), at the center, no
     projection hit.
   - **Kerr-Schild `a = 9/10`** on its own oblate offset surface (`h =
     5/128`, `m = 8`, `n_L = 8`, `lmax_shape = 4`; `r_1` from `1.12` on the
     axis to `1.38` on the equator): valid at every `L` and both orders
     (`min λ(γ) ≥ 0.945`, no hit), the truncation of the state on the
     surface `0.26`, `0.051`, `9.1e−3` at `L = 4, 8, 12` for `cont = 1`
     (`0.32`, `0.075`, `1.4e−2` for `cont = 2`) of its largest component,
     `Π_tx`, `Π_ty` the worst, and the value rows' residual `2.7e−4` at
     `L = 16`: **at the default `L = 8` the target is 3–5 % off the state on
     a spinning hole's surface** — inside what step 8c measured the layer to
     absorb (a 20 % mass error), and the first knob if 8f says otherwise.
   - **Harmonic Kerr `a = 9/10`** — finding 3's configuration, `h = 5/256`,
     `m = 4`, `lmax_shape = 12`, the offset surface from `0.36` (axis) to
     `0.92` (equator, `0.02` outside the ring): **`cont = 1` is valid for
     `L ≥ 8`** (`min λ(γ) = 2.27`, value residual `1.9e−3` at `L = 8`,
     `3.0e−5` at `12`) and not at `L = 4` (`min λ = −10`); **`cont = 2` is
     invalid at every `L` from 4 to 16** (`min λ(γ) = −73` at `L = 8`, 953 of
     1225 points), its curvature rows forcing the polynomial through
     negative eigenvalues inside the surface. At `h = 5/512` (equator at
     `0.96`) `cont = 1` is valid from `L = 4` and `cont = 2` first at
     `L = 16`. **The data itself is outside `default_bounds`**: the samples'
     `max |(α/√γ)Π| = 366/M` against step 8b's `K_max = 100/M` at 17 of 153
     points, so `fit_state`'s projection would clamp the target's `Π` there
     — the proposed bounds are Kerr-Schild's, and harmonic `a = 9/10` needs
     its own before 8e-ii projects into them.
   - **The evolved state's fit**, at the end of the suite's tracked `:damped`
     run to `0.15 M`, against the analytic solution's fit on the same
     geometry: **`0.031` apart on the surface, falling to `0.016` at the
     center** (Euclidean over the twenty components), where the samples are
     `0.034` from the truth and the run's masked error is `0.035` in L∞
     (`3.1e−3` in L2): the least-squares projection passes the state's error
     through at `0.92` and damps it inward; both fits valid, no hit.
   - **What it costs** (Apple silicon, Julia 1.13, one thread, a shared
     machine): **`build_fit` at `L = 8` is `3.6 ms` (analytic, `cont = 1`),
     `5.2 ms` (`cont = 2`) and `7.0 ms` (state sampler)** — `0.4–0.6 ms` at
     `L = 4`, `20–25 ms` at `12`; the prediction was milliseconds. **The
     evaluator is not cheap: `fit_state` is `2.2 µs` a point at `L = 8`,
     `cont = 1` (`2.9 µs` at `cont = 2`; `0.64` and `4.3 µs` at `L = 4`,
     `12`), against `91–96 ns` for one `u_exact` on the same machine, 23
     of them** — the recurrence is `85 ns` (8d's shape series, `79 ns`) and
     the `20 × (L+1)² × (cont+1)` contraction the rest. `PLAN.md`'s
     `+5–10 %` per right-hand side assumed an evaluator near `u_exact`'s
     cost; at `2.2 µs` on the fixture's layer (21 % of its points) it would
     have been `+40–75 %`, which is why the kernel reads a cache instead
     (piece 8).
8. **The target is cached on the grid (decided in review, step 8e)**, not
   evaluated in the right-hand side: a field set of `Hsrc`'s shape (`G = 0`,
   read at the owned point, never differenced) holding **40 variables — the
   target `A` (the packed `(h, Π)`) at the fill time `t_f` and its slope in
   time `S`** — so that the kernel's target at `t` is `A + (t − t_f) S`, two
   loads a variable **(proposed in step 8e**: one field set, and the slope
   rather than the previous fit's values, since the slope is what the time
   interpolation multiplies**)**. From the latest fit `F_a` (built at `t_a`)
   and the one before it `F_b` (at `t_b`), both evaluated at `t_f` about
   their own tracked centers, `S = (F_a − F_b)/(t_a − t_b)` and `A = F_a +
   (t_f − t_a) S` — the linear continuation of the last two fits, `S = 0`
   while there is one or the two were built at the same time. A
   `map_blocks!` kernel (`fit_target_kernel!`) fills it by calling
   `fit_state` with the coefficient arrays as device arguments, at every
   owned point inside the offset surface's bounding sphere **plus one cell**
   — so that a center moving by the refill rule's `h/4` never exposes an
   unfilled layer point **(proposed in step 8e)** — and zeros elsewhere; the
   cache is the host evaluator bit for bit. **What it costs (measured in step
   8e**, one thread, on the fixture's 120 blocks**)**: a fill is `43–44 ms`
   from one fit and `75 ms` from two, `0.4` and `0.7` of a right-hand side,
   once a chunk of about forty evaluations; a right-hand side with the
   fitted layer is within `±4 %` of the `:damped` layer's either way — `111`
   against `107 ms`, `106.7` against `107.6`, and in the suite's run `114`
   against `118` — the cache's two loads a variable where `:damped` takes the
   analytic solution's dual pass, which the prediction ("a few percent")
   had as a cost and is at worst a wash.
9. **The refill rule (decided in review, step 8e).** The cache is refilled
   from the current fits at every chunk's start — after every fit — and,
   for a geometry that moves, whenever the tracked center would move by
   more than `h/4` since the last fill: the chunk's steps are split into
   `⌈|v_est| · chunk / (h/4)⌉` solves with a refill between **(proposed in
   step 8e**, over lowering the chunk, which would change the record's
   cadence**)**. On the static fixture `v_est` is the finder's noise and a
   chunk is one piece; on the boosted seed below (`|v| = 0.3`, `h = 5/128`,
   chunks of `M/20`) it is `1.54` → two pieces and one mid-chunk refill a
   chunk (measured in step 8e). **The moving seed (measured in step 8e,
   `hole_runs.jl fitted=boosted`)**: `boost(Harmonic(1, 0), 0.3 x̂)` on 848
   blocks at `h = 5/128`, `m = 8`, to `0.1 M` — the finder succeeds on every
   row from the boosted hole's initial data, the track is at `x = −0.01492`
   and `−0.02985` at `0.05` and `0.1 M` against the analytic `−0.015` and
   `−0.03` (`v_est = −0.2983`, `−0.2986`), `track_offset` `2.1e−3` and
   `3.9e−3` cells, every fit valid; the masked error is `3.26e−2` against
   `:damped`'s `2.03e−2` on the same mesh and track (`1.6×`, the initial
   data's kink again), in 14 steps and `39 s` at four threads.
10. **The variant (decided in review, step 8e).** `INTERIOR_VARIANTS` has
   `:fitted`, a `FittedInterior`'s only (`Interior` refuses it, a sphere
   about the analytic center having no offset surface to fit on), and the
   right-hand-side kernel's branch for it reads the cache: **the core
   `du = −ρ_max (u − u_fit)` with `F` not evaluated, the layer `du = w F −
   ρ (u − u_fit)`, the evolved region `du = F`**; the step limiter and the
   paste are no-ops. `GHProblem` carries the cache, the fits that filled it
   and `t_f` (`target`, `fits`, `t_target`; `refill_target`), and refuses a
   `:fitted` interior without a cache; `with_interior(p, int; fits,
   target, t_target)` keeps them. **One right-hand side with the `:fitted`
   layer is `:damped`'s bit for bit at every point outside the offset
   surface** (45 545 of the fixture's 61 440), its core is `−ρ_max (u −
   u_fit)` exactly, and its layer differs from `:damped`'s by `ρ (u_fit −
   u_exact)` to `1.4e−14` of the right-hand side (measured in step 8e). The
   driver, per chunk: the geometry and the rate (`with_interior`) → the
   refill → the solve (in pieces, piece 9) → the record (constraints, error,
   monitor, find, indicator) → `update_track` and the next chunk's geometry
   → **`refit!`: the state sampler on freshly filled ghosts** (the find
   filled them, but so may every monitor since — a fill is a fifth of a
   right-hand side) → a `cont = 1` fit into the target's ranges → the regrid
   branch, whose new mesh gets a new cache filled at the next chunk's start
   from the same fits (a fit is a polynomial in `x` and does not know the
   mesh; **proposed in step 8e**). **A fit whose sweep is not a metric does
   not become the target**: the previous fits are kept, the row says
   `fit_valid = false`, and the run goes on — the find's coasting applied to
   the fit **(proposed in step 8e)**. The record's `residual` is, for
   `:fitted`, the layer's distance from its *target*, and the error kernel
   evaluates the analytic solution only at the points its error and drift
   rows read **(both proposed in step 8e)**: the fitted interior has no
   analytic one, and harmonic Kerr's is singular inside the offset surface.
   `check_interior_radii` skips its singular-set check for `:fitted`. **A
   tracked run's first find starts from the seed's shape** on the finder's
   grid, not a sphere of the mean radius **(proposed in step 8e)**: on
   harmonic `a = 9/10` that sphere, `0.72`, lies inside the offset surface's
   equator at `0.92`, and the footprint guard refuses its first iterates.
11. **The initial data (decided in review, step 8e)**: the analytic solution
   outside the offset surface and **the `cont = 1` fit of the analytic
   solution inside it, for every chart**, from **`Float64` samples whatever
   `T` is** — the cache filled from the analytic fit first, then the state
   (`fitted_state_kernel!`); no analytic evaluation below the surface, so a
   chart whose interior is singular gets regular data, and `state_callback`
   refuses a `:fitted` geometry. The reason for `cont = 1` is harmonic `a =
   9/10`, whose curvature-matched fit is not a metric at any `L` to 16
   (piece 7). **Its price on the fixture (measured in step 8e)**: the data
   are `C¹` at `r_1`, value and slope right and the curvature off by
   `O(1/M²)`, which the compact second difference turns into an `O(1)`
   right-hand-side error at the innermost evolved points — finding 1's kink,
   in the *data* where step 8c's E3 had it in the *target* only. The masked
   error at `0.15 M` is **`1.33e−2` against `:damped`'s `3.11e−3`** (`4.3×`;
   L∞ `0.177` against `0.035`) and decays to `1.24×` by `1 M`; at `Float32`
   the same run agrees with `Float64` to `7e−6` (measured in step 8e,
   `hole_runs.jl fitted=fixture` for the table):

   | initial data | masked L2 at `0.1 M` | `0.5 M` | `1 M` | shell `C_a` L2 at `1 M` |
   |---|---|---|---|---|
   | `:damped` (the analytic solution everywhere, the core rule) | `2.09e−3` | `9.81e−3` | `1.40e−2` | `1.24e−2` |
   | `:fitted`, the fit below `r_1` (decided) | `9.77e−3` | `1.90e−2` | `1.74e−2` | `4.02e−2` |
   | `:fitted`, the `cont = 2` fit below `r_1` | `3.05e−3` | `1.01e−2` | `1.66e−2` | `3.22e−2` |
   | `:fitted`, the fit below the core surface (`fit_initial_depth = n_L h`) | `2.05e−3` | `9.72e−3` | `1.56e−2` | `2.57e−2` |
   | `:fitted`, the fit below `3h` | `4.18e−3` | `3.30e−2` | `3.92e−2` | `7.98e−2` |

   Moving the switch to the core surface — where no evolved stencil reads
   it — makes the fitted run `:damped`'s to 2 % for the first `0.5 M`, and
   the shell's `C_a` stays 2–3× `:damped`'s at `1 M` on every row: the
   target is a fit of the evolved state, continued inward, and not the
   solution. The core-surface switch needs the analytic solution regular on
   the whole layer, which Kerr-Schild at any spin and harmonic Kerr at
   `a = 7/10` have and harmonic `a = 9/10` does not (its disk cuts the
   equatorial layer); `evolve!(…; fit_initial_depth, fit_initial_cont)` are
   the study knobs, defaults the decision.
12. **The proof-of-concept chart: harmonic Kerr `a = 9/10`, `m = 4`
   (decided in review, step 8e)**, the case's own `FittedSpec(margin = 4,
   lmax_shape = 12)` and not a change of the default `8`, at `h = 5/256` on
   2472 blocks (2240 of them at `5/256`) in a box of half-width `5/4 M`
   (`hole_runs.jl fitted=harmonic`, three minutes at four threads). **The
   first `:fitted` initial data of this chart exists (measured in step
   8e)**: `r_1` from `0.359` (axis) to `0.921` (equator), the core from
   `0.203`; the analytic fit valid (value residual `1.9e−3`, sweep `min λ(γ)
   = 2.27`, `min α = 0.208`); the target's ranges `α ≥ 0.066`, `λ ≤ 774`,
   `K ≤ 1464`; **every one of the 1 265 664 points finite and a metric**
   (`min det γ = 0.648`, `min α = 0.208`, `min λ(γ) = 0.5`); one right-hand
   side `0.71 s`, finite. **And the run ends in its first chunk
   (`2.5e−3 M`) in a degenerate metric**: the right-hand side is `1.1e9` at
   the offset surface just above and below the disk (`|z| = 0.12`,
   cylindrical radius `0.88`). The reason is the data on the surface, not
   the variant: at the equator the ring is `0.02 M` inside `r_1`, and the
   analytic solution one cell outside it has `|u| = 3.9e4` and a second
   difference of `6.4e8`, where the axis has `|u| = 31` and `416`. A
   polynomial of degree `L + 2` fitted to a surface whose equatorial data
   are a thousand times its axis's is right at the collocation points and
   wrong between and below them, so the composite data's second difference
   at the first evolved point is off by **`9.5e5` on the axis (against the
   analytic `416`), `1.8e7` at 45° (against `111`) and `9.4e7` on the
   equator (against `6.4e8`)** at `L = 8`. Three levers measured (host-side,
   the same probe): fitting `Π̃ = (α/√γ)Π` in place of `Π` shrinks the
   off-equator kinks 20–40× (`2.6e4`, `7.5e5`), `L = 12` another 20–650×
   (`39`, `1.3e4` with `Π̃`), and weighting the collocation points by their
   data's inverse scale buys nothing (the fit is already right at the
   points); halving `h` (the equator's `r_1` at `0.96`) changes the ratios
   little. For comparison, Kerr-Schild `a = 9/10`'s equator is `46` against
   `61` and harmonic `a = 7/10`'s 45° `6000` against `334` (`1720` with `Π̃`
   at `L = 12`). **So the proof-of-concept chart does not run on this target
   at `h = 5/256`**; step 8f's row for it needs `Π̃` and `L ≥ 12` at the
   least, and the design's own fallbacks are `a = 7/10` and 8g's excision.
   **And the evolved region on the equator is itself under-resolved there
   (measured in step 8e)**: the analytic solution's own length scale at the
   first evolved point on the equator, `√(|u|/|Δ²u|)`, is `0.008 M` at
   `h = 5/256` — under half a cell — `0.019 M` (two cells) at `5/512` and
   `0.024 M` (five cells) at `5/1024`, with `m = 4`: the ring sits `m h +
   0.02 M` inside it, and the solution varies on the scale of that distance.
   Whatever the interior does, the chart at `a = 9/10` wants `h ≲ 5/1024` on
   its equator, or a margin that depends on the direction.
13. **A moving target (added in step 8).** Two things change when the
   geometry moves through the grid, and step 8 built one switch for each:
   - **The target's rate is fed forward** (`evolve!(…; target_rate = true)`,
     the default, **proposed in step 8**). A point relaxing toward a target
     that moves lags it by `|∂_t u_fit|/ρ`, and where `w < 1` the layer also
     advects the hole's structure at `w v` instead of `v`; so the cache's
     slope now includes the latest fit's translation with the track — a
     centered difference in the time the fit is evaluated at, an eighth of a
     cell of travel each way, exactly zero for a fit at rest — and the
     kernel adds `(1 − w) ∂_t u_fit` in the layer and the core, which makes a
     target that is the moving solution a steady state of the modified
     equation. The slope agrees with the cache's own difference to
     `1.2e−4` relative, and a problem whose slope is zero is the problem
     without it, `==` (`test/moving_tests.jl`). **It changes almost nothing
     measurable (measured in step 8)**: `4.56e−2` against `4.67e−2` masked
     at `0.5 M` on the boosted `a = 0` hole, `0.132` against `0.172` at
     `M/4` on G5's chart with `w_ramp = 1` and `0.492` against `0.494` on the
     default ramp —
     the lag is not what the moving layer suffers from (next). It stays on,
     since it is the consistent form and costs two evaluations of the fit
     per filled point per refill.
   - **The initial data are blended into the fit across the layer**
     (`evolve!(…; fit_initial_blend = true)`, G5's rows, **proposed in step
     8**). With `fit_initial_depth = n_L h` the data step from the analytic
     solution to the fit at the core surface, and on G5's chart that step is
     twenty-five times the solution (`Π_xx` from `1.5e4` to `600` on the
     trailing equator). A static hole relaxes it away where `w = 0`; a moving
     one carries it outward in depth on its trailing side, at the hole's
     speed, into the evolved part of the layer, where `F` of a jump is
     `O(jump/h²)`: at `t = M/4` the worst error of step 8's first G5 run was
     `Π_xx = +2650` four cells deep on the trailing equator against a
     solution of `1100`, where the fit was `−580` off — neither the truth nor
     the target, and the masked L∞ was `334` against the static hole's `2.4`
     (`hole_runs.jl moving`, the screens). The blend replaces the step by a
     `C²` ramp in the fit's variables `(log α, β^i, γ_ij, Π̃_ab)` — never in
     `g_ab` — from the analytic solution at the offset surface to the fit
     at `fit_initial_depth`: masked L2 / L∞ at `M/4` **`0.115` / `18`**
     against `0.49` / `334` with the step, the trailing layer's outer
     quarter `53` against `1425` off the truth. It is not free on a static
     hole (`0.075` / `9.5` against `0.032` / `2.4` at `M/4`: the blended
     layer is not a solution where `w = 1`), which is why it is G5's row's
     choice and not the default.
   The table of the screens is in [The moving hole (step
   8)](SINGULARITY_HANDLING.md#the-moving-hole-step-8).

#### What the layer costs

**What the layer costs.** `u_exact` is evaluated at every point of the
layer at every RHS evaluation — one forward-mode dual pass through the
background's metric per point, about a microsecond on a CPU by GHSO2's
measurement, on a region of a few hundred thousand points at
proof-of-concept resolution: a few percent of an RHS, measured in G4.

**What GHSO2's findings say about this configuration.** The horizon —
the sonic surface where a characteristic speed vanishes — lies in the
evolved region between the layer and the outer boundary, exactly as in
GHSO2's excised runs. Its recipe applies: `ε_KO ≈ 0.5` for the grid-scale
layer, `γ0 ≈ 1/M` for the constraints (`notes/methods-ghso2.md`,
`notes/sonic-surface.md`). GHSO2 also measured a *slow,
constraint-preserving gauge drift* (≈ 0.14/M) of the excised hole under
prescribed sources; whether exact interior and boundary data anchor the
gauge better is **(predicted: partly — the drift rate falls but does not
vanish)** and is the main risk to a long run. G4 records the rate; the
damped harmonic gauge driver is the extension that would remove it.

#### Excision (added 2026-10-05)

**Excision is reopened as an interior variant under study (Erik's
decision, 2026-10-05), for static holes first.** The layer needs a target —
the analytic solution, or a fit of the evolved state — and fails where
neither is good enough. Excision needs none. `PLAN.md`'s steps X1–X3 decide
whether it works on this mesh: X1 by models on the host, X2 by building it,
X3 by measuring it on the octant against `:damped` and `:fitted`. Nothing
below is built yet; every number is **(predicted)** until a step measures
it. **(Amended in step X1:** the closure weights are built, in
`src/stencils.jl`, and X1's models have measured what is marked so; no
kernel uses either yet.**)** **(Amended in step X2b:** the variant is built,
on both geometries — what and how is under "What step X2b built" at the end
of this section, and its numbers under [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-the-variant-step-x2b), "Excision: the variant (step X2b)"; X3 measures
it on the octant.**)** **(Amended in step X3:** measured on the octant on
Symmetry's H200s — feasible for the static Kerr-Schild `a = 0` hole, as
accurate outside the horizon as the `:damped` layer at `r_E = M/2`, and with
no gauge drift to `50 M`; the recommendation is under "What step X3
measured" at the end of this section.**)** **(Amended in step X4:** on
`main`'s spill-free right-hand side, 3.9× faster on the H200, and on the
rotating octant too — "What step X4 changed", before X3's.**)** **(Amended in step X5:** the spinning
hole's frame-dragged faces — closure axes whose shift points into the
excised set — have a rule: the per-axis closures everywhere else, and on
such an axis the advective derivative formed from the centered stencil
with its excised taps extrapolated along the lattice direction nearest the
surface's normal; on Kerr-Schild `a = 3/5`'s equatorial plane it is stable
at every depth of the window, where the per-axis closures are not. The
rule, the window and what X6 and X7 need are under "The frame-dragged faces
(step X5)" at the end of this section.**)** **(Amended in step X6:** the rule
is built — in the zone kernel's second launch, from rule bits and direction
codes built once with the problem — the zone points' mixed derivative is
symmetric in its two axes, and the static `a = 3/5` hole runs on the rotating
octant; "What step X6 built", at the end of this section.**)** **(Amended in
step X7:** measured on the rotating octant on Symmetry's H200s — the excised
static `a = 3/5` hole at `r_E = 4M/5` is the `:damped` layer outside the
horizon from `h = 1/32` on, its spin drifts as the layer's does, and the
depth scan found one failure, a lego corner at the pole of a shallow surface;
"What step X7 measured", at the end of this section, has the numbers and the
recommendation.**)** **(Amended in step X8:** the excised set is shaved —
a lattice point the geometry leaves outside whose three neighbours one step
toward the center are inside it is excised too, one pass, decided once with
the classes from a predicate every mask evaluates; "What step X8 built" and
"What step X8 measured", at the end of this section.**)** Moving holes —
points that leave the excised set on the trailing side and need values — are
a later round.

**The variant, `:excised`.** Points beyond the **excision surface** are
not evolved: `du = 0`, `F` never evaluated, their data finite and never
read. The surface is the layer's own outer surface: the sphere `r = r_1`
of [step 5's layer](#step-5s-layer-the-analytic-control) about the
analytic center, or the tracked offset surface `d = 0` of [the tracked
geometry](#the-tracked-geometry) — the set `:pasted` already freezes,
without the paste and without a layer inside it. `r_0` (the core surface)
is only where the core rule puts the initial data inside. There is no
target, no fit, no relaxation rate. For this round the geometry is
**frozen for the whole run** **(proposed in review, 2026-10-05)**: built
once from the case's sphere or the seed's shape, with the center's
velocity zero. The track still updates, for the record and for an
assertion that the found horizon stays `m h` outside the surface. So the
excised set changes only when a problem is built, and no point leaves it.
`regrid` and `adapt` are refused for `:excised` in this round. A level
change near the surface would prolong stale excised data into evolved
points, and the moving round has to solve that anyway.

**Why step 8g's price does not apply (proposed 2026-10-05).** Step 8g
priced excision as an *extrapolation fill* of the excised points within
reach of evolved stencils. A fill that keeps the order needs a `3G` halo or
a second ghost exchange, and depends on how the blocks are cut ([Possible
extensions](#possible-extensions)). Extrapolating into the excised set
and then applying the centered stencil *is* a one-sided stencil. Done **per
stencil** rather than per excised point, it reads only the stencil's own
side, inside the existing reach `G`:
- no wider halo;
- no second exchange;
- the same operator wherever the block boundaries fall.

**The closures.** At an evolved point, per axis and side, `k± ∈ 0…G`
counts the consecutive non-excised points. A stencil whose taps reach past
them uses the **closure** on the nodes `−k⁻ … k⁺`, capped at reach `G`.
Its weights come from `lagrange_derivative_weights` (`src/stencils.jl`),
built in `Rational` and rounded once into `T`.
- At the first evolved point (`k⁻ = 0`) the reach-capped closure's order is
  `q/2 + 1` for `∂` and `q/2` for `∂²`. It is centered again from
  `k = q/2` on.
- At an outflow boundary that order loss should be invisible outside the
  horizon in the continuum, because the boundary's truncation error is
  carried into the hole. What matters is stability, and what the closure
  reflects into the outgoing grid-scale mode of [step 8a's
  analysis](#kreissoliger-dissipation).
- Mixed derivatives are nested. The outer sum along `i` uses the point's
  `i`-closure; the inner sum along `j` uses the `j`-closure of the point
  `x + a e_i` it is taken at.
- Kreiss–Oliger dissipation near the surface needs a one-sided or
  reduced-rank form that keeps the damping sign. Step X1 chooses it.
  **(Chosen in step X1: Mattsson–Svärd–Nordström's, `:msn`, below.)**
- **Lopsided (upwinded) shift advection** inside the horizon is the
  candidate cure for the grid-scale leakage. Inside the horizon `β`
  points away from the hole, so the upwind side of `β^k ∂_k` *is* the
  evolved side. A lopsided `D₁` also acts on the Nyquist mode, which every
  centered `D₁` annihilates. It is a blend, `C²` in the depth below the
  horizon, off outside it, so the exterior's operator is unchanged bit for
  bit. X1 says whether it is needed; X2 builds it as an option, off by
  default. **(Answered in step X1: not for stability; it is the lever on
  leakage. Below.)**

**The closures as built (amended in step X1).** `src/stencils.jl` holds
every closure as a function of `(q, k⁻, k⁺)` alone, in `Rational` on
`lagrange_derivative_weights`, with `k⁻, k⁺` capped at `G` (`G` meaning
"at least `G`"):
- `closure_nodes`, `closure_derivative_weights` and `closure_exact_degree`
  are the starting family above, unchanged: centered from `min(k⁻, k⁺) ≥
  q/2`, otherwise every node in `[−min(k⁻, G), min(k⁺, G)]`. At the first
  evolved point the orders are `(2, 1)` at `q = 2`, `(3, 2)` at `q = 4` and
  `(q/2 + 1, q/2)` at every `q`. A `reach` keyword moves the cap; only the
  models use it.
- `closure_dissipation_weights(q, kind, k⁻, k⁺)` has the three closures,
  `DISSIPATION_CLOSURES = (:reduced, :onesided, :msn)`:
  - the reduced rank `r′ = min(k⁻, k⁺)`, none at the first evolved point;
  - the `2r′`-th difference shifted to the evolved side and signed to damp
    Nyquist at the point;
  - Mattsson, Svärd and Nordström's `−2^{−2r} D_rᵀ B D_r`, `D_r` the
    `r`-th forward difference and `B` the indicator of its windows of
    evolved points. Its interior rows are the centered operator, it reads
    `[−k⁻, G]`, and it annihilates degree `< G` near the surface.

  Every kind is the centered operator where `min(k⁻, k⁺) ≥ G`, and none
  drives Nyquist at any point. **Only `:msn` is negative semidefinite in
  the discrete `l²` norm, on any excised pattern; the reduced rank and the
  one-sided closure are not** (an exact witness each in
  `test/stencils_tests.jl`). The symmetric part's largest eigenvalue, in
  units of `ε/h`, is `+0.033–0.035` (reduced) and `+0.022–0.096`
  (one-sided) on a half-line at `q = 2 … 8`, and `+0.052`, `+0.059–0.099`
  on the test's line with gaps, against `10⁻¹⁶` for `:msn` (measured in
  step X1). So **the dissipation's closure is `:msn` (proposed in step
  X1)**, in the norm in which the centered operator is damping. The models
  cannot tell the three apart — on the frozen line all three are stable at
  the faces Kerr-Schild `a = 0` has and unstable where the shift points
  into the excised set at `ε_KO = 1/2` (the reduced rank and the one-sided
  closure escape some of those rows at `ε_KO = 1`), and on the plane all
  three decay at the same rate —
  so the choice rests on the estimate only `:msn` has.
- `lopsided_weights(q, up, k⁻, k⁺)` is the order-`q` first derivative on
  `1 − q/2 … q/2 + 1`, mirrored for `up = −1`. It reaches `G` upwind and
  `q/2 − 1` downwind; nearer the surface on the downwind side it starts at
  `−k_down`, and where the upwind side is short it is the closure,
  unlopsided. Its symbol's real part has the damping sign at every phase,
  not only at Nyquist (asserted at 65 phases).
- `closure_admissible(q, k⁻, k⁺)` is one side clear to `G`. That is X2b's
  build-time refusal "excised on both sides of one axis within reach"
  **(proposed in step X1)**. The weights exist for more (a three-point gap
  has a `∂²` closure), but a convex excised set never makes such a point.
- `closure_table(T, Val(q); dissipation = :msn)` returns a `ClosureTable`,
  the kernel argument X2b will carry. `d1`, `d2`, `ko` are `[slot, k⁻ + 1,
  k⁺ + 1]` on the `2G + 1` slots `−G … G` (slot `j + G + 1`, zero outside
  the closure's nodes), and `lop` has a fourth index for `up = −1, +1`.
  `d_lo/d_hi`, `ko_lo/ko_hi` and `lop_lo/lop_hi` are the nodes each closure
  reads, so that a contraction never touches an excised value, and
  `admissible` is the refusal. Every entry is `T(num)/T(den)` once, so its
  centered rows are `derivative_weights`/`dissipation_weights` bit for bit
  at `Float64` and `Float32`. It is **4624 bytes at `q = 4`, `Float64`**
  (1888 at `q = 2`, 2384 at `q = 4`, `Float32`): above CUDA's classic 4 kB
  kernel-parameter limit and inside the 32 kB that CUDA 12.1 allows on an
  H200. X2b decides between an argument and a device array.

**The outflow condition, and the lego staircase (predicted 2026-10-05; X1
measures).** In the continuum, a surface inside the horizon needs no
boundary condition where every characteristic speed along its normal
points into the hole. That is `GHSO2`'s outflow class
(`notes/methods-ghso2.md`), and near the horizon it holds for surfaces
parallel to it. A closure along a *grid axis* sees that axis's speeds, not
the normal's. For Kerr-Schild `a = 0`, along an axis at angle `θ` to the
radial normal, the ratio `b/a` of shift to light speed is
`H cos θ / √(1 + H sin²θ)` with `H = 2M/r`. Outflow along the axis needs
that to exceed one, which is `cos²θ > r/(2M)`:
- `r < M` at `45°`;
- `r < 2M/3` on the cube diagonal;
- at no depth for an axis nearly tangent to the surface.

Weighting the closure faces by their projected area, **the fraction of
inflow-like closures on a lego sphere of radius `r_E` is `r_E/(2M)`**: half
of them at `r_E = M`. At such a face one characteristic enters from the
excised side, and the closure there is an extrapolation at an inflow
boundary. So whether lego excision is stable is the feasibility question,
and step X1 answers it with a two-dimensional model before a kernel is
written. It compares:
- per-axis closures;
- per-stencil extrapolation along the lattice direction nearest the normal,
  with its sources inside the point's `G`-box — which is what Cartesian
  excision codes have used — with and without lopsided advection.
The normal-direction fill priced in [Possible
extensions](#possible-extensions) stays the last resort.

**What X1 measured: go, for Kerr-Schild `a = 0` (measured in step X1).**
The tables are under [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-the-analysis-step-x1), "Excision: the
analysis (step X1)":
- **The prediction holds.** On the lego sphere the inflow-like fraction is
  `r_E/(2M)` to `0.03` at every `h` from `1/16` to `1/48`.
- **Every inflow-like face of Kerr-Schild `a = 0` is of the benign kind.**
  `β` is radial, so at every face the shift still points out of the excised
  set, `0 ≤ b/a < 1`. On the frozen line such a face is marginal: it has a
  zero mode, the static `u = c(x − L)` that an inflow boundary without data
  admits (a Jordan block at `q = 4`), and nothing to the right of zero. An
  outflow face (`b/a > 1`) is strictly stable.
- **On the lego circle the per-axis closures are stable at every `r_E` from
  `M/2` to `7M/4`.** That is inflow-like fractions `0` to `0.73`, at
  `q = 2, 4`, `ε_KO = 1/2` and `1`, with and without lopsided advection.
  The rightmost eigenvalue is `−0.31` to `−0.37/M` in every
  configuration, against the `:damped` layer's `−0.26` to `−0.34/M`. Noise
  decays at the box's own `−0.32` to `−0.37/M` to roundoff by `100 M` at
  `h = 5/128` and `5/256`. The per-stencil extrapolation is stable too, at
  `ε_KO > 0`.
- **`ε_KO > 0` is required, and it is what separates the two families.**
  Without dissipation the per-axis closures grow as the interior itself
  does: `+0.08` to `+0.14/M` against the layer's `+0.11` to `+0.14/M`,
  which has no surface, and nearly independent of `h` (GHSO2's grid-scale
  layer, `notes/sonic-surface.md`). The extrapolation grows at `+1.1` to
  `+3.8/M`, on the surface, and faster at the finer `h`.
- **The closures do what a bare frozen core does not.** Centered stencils
  reading the unperturbed core's data — Dirichlet at an outflow surface —
  barely decay at `r_E = 3M/2`, `−0.07` to `+0.04/M` (growing at `q = 2`,
  `h = 5/48`), against the closures' `−0.31` to `−0.37/M`.
- **The closure leaks what the layer leaks.** At the layer's outer radius it
  transmits the layer's amount of grid-scale content through the horizon,
  so step 8a's leakage margin carries over unchanged. The lopsided
  advection is the lever on that leakage, not on stability: from eight
  cells deep it cuts what crosses the horizon `7.6–72×`. It costs RK4's
  step — on the plane `cfl` from `2.05` to `1.21–1.50` at `q = 2` and from
  `1.86` to `1.52–1.79` at `q = 4` (`ε_KO = 1/2`) — which `cfl = 1/2`
  absorbs.
- **Spinning holes are not covered.** Their lego surfaces have faces where
  frame dragging turns the shift *into* the excised set along the axis
  (`b/a < 0`, on the surfaces with normal outflow: 2–14 % of faces at
  Kerr-Schild `a = 3/5`, 5–10 % at `9/10`, 12–17 % on harmonic `a = 7/10`
  and 17–18 % at `9/10`, down to `b/a = −1.2` to `−2.9`). On the frozen
  line the closure is **unstable** there, `0.03–0.19/h`, at every `ε_KO`
  under `:msn` and under every dissipation closure at `ε_KO = 1/2`,
  lopsided or not. And Kerr-Schild's normal outflow ends at the inner
  horizon — `0.63 M` below the outer one at the equator at `a = 9/10`, not
  the ring — and the harmonic chart's at its disk, `0.3 M` and `0.1 M`
  below at `a = 7/10` and `9/10`. **(Answered for Kerr-Schild `a = 3/5` in
  step X5:** on the equatorial plane the per-axis closures fail where faces
  reach `b/a ≈ −1.3` and below, near the inner horizon, and a rule that
  extrapolates the advection on those axes holds every depth — "The
  frame-dragged faces (step X5)", below.**)**

**What X2a and X2b take from X1 (proposed in step X1).**
1. **The closure family is the per-axis one, as built** (the starting
   family, reach `G`, `closure_table`), with the mixed derivative nested.
   The per-stencil extrapolation is as stable at `ε_KO > 0`. It needs no
   `k±` tables, but it is ten times worse without dissipation and grows
   with resolution there, and it extrapolates per tap.
2. **The dissipation's closure is `:msn`.** It is the only one with the
   damping sign in `l²`, and the plane cannot tell the three apart at
   `ε_KO = 1/2`.
3. **`ε_KO > 0` near the surface is part of the variant.** X2b refuses
   `:excised` with `ε_KO = 0` (or a dissipation profile that vanishes at
   the surface), saying why.
4. **The lopsided advection is built as the option the plan says, off by
   default**, and X3's depth scan runs every depth with and without it. The
   blend is `C²` (`smoothstep`) in the depth below the horizon, from
   `1` cell to full at `5` cells (`start = 1`, `width = 4`), the profile the
   models measured.
5. **The least depth, for Kerr-Schild `a = 0`:**
   - The closures set none: `r_E = 7M/4` is stable, `0.25 M` and four cells
     of `5/64` below the horizon.
   - The scheme's reach sets it in cells. X2b's assertion is `m ≥ G + 1`
     (3 cells at `q = 2`, 4 at `q = 4`).
   - The horizon finder's footprint must not reach the excised set. Its
     `q + 2` points per axis reach less than `G h` from the query, `√3 G h`
     on a diagonal, so `m ≥ ⌈√3 G⌉`: **6 cells at `q = 4`** (`0.375 M` at
     `h = 1/16`, `0.19 M` at `1/32`), 4 at `q = 2`. The octant study's
     `:damped` run at `m = 4`, `q = 4`, whose finder reached the layer, is
     the same bound met.
   - Step 8a's leakage margin and the octant study's 8–12 cells are what
     buys accuracy beyond that.
   - **X3's depth window at `h = 1/16` is therefore `r_E = M/2 … 13M/8`,
     6 to 24 cells.**

   For the spinning charts the window exists in normal outflow (above), but
   the per-axis closures do not hold its `b/a < 0` faces. That is the
   question a spinning round has to answer first, with the normal-direction
   fill or a closure for those faces.

**On the mesh, and on a device (proposed 2026-10-05; step X2 decides the
details).** These are the pieces:
- **A class per stored point** — centered, excised or zone (a band point,
  or an upwinded one) — in a small integer array.
  - It is built when a problem is built: the excised bit on owned points,
    from the masks' own predicate; then one ghost exchange through TreeAMR,
    so that every ghost agrees with its owner and the octant's walls
    mirror it; then the counts `k±`.
  - **Within a run the classes are the single source of truth** for what
    is excised, in every kernel.
- **The main kernel** is unchanged except that it skips excised and zone
  points (`du = 0` for the former).
- **A zone kernel**, launched right after it, computes `F` at zone points
  with the closures. It runs over all blocks with a block-uniform early
  exit, since TreeAMR has no launch over a subset.
  - The physics is one copy. `gh_rhs_at_point` takes a stencil *provider*:
    the centered one is today's code, bit for bit; the closure one reads
    `k±` and a small table. **(Built in step X2a**, with the centered
    provider only: the interface and its measurements are under [One
    right-hand-side evaluation](#one-right-hand-side-evaluation). What the
    closure provider has to implement is below.**)**
  - **How the closure provider plugs in (proposed in step X2a).** It is a
    `StencilProvider` subtype built per zone point, `isbits` like
    `Centered`, holding what the point needs: its codes `k±` along the three
    axes, the class array (or the codes) and the point's own linear index in
    it — stored per point as the working array is, `(n₁, n₂, n₃, block)`, it
    has the same spatial strides, so a neighbour along `i` at offset `a` is
    that index plus `a·st[i]` — the weight table (a device array, X1's
    4.6 kB), `1/h` and the lopsided blend's weight at the point. Its `d1`, `d2` and `ko` contract the row
    `[·, k⁻ + 1, k⁺ + 1]` of the axis over `d_lo:d_hi` (`ko_lo:ko_hi`) in
    ascending order, unscaled; its `dmix` runs the outer sum along `i` over
    the point's `i`-closure and, at each outer node, the inner sum along `j`
    over **that node's** `j`-closure, read from the class array; its `adv`
    returns `∂f_d` where the blend is zero — so that the exterior's operator
    stays bit for bit — and otherwise mixes it with `1/h` times the
    lopsided row for `up = sign(β_d)` (`lopsided_weights`: the upwind side is
    the side the shift points to). Nothing else in `gh_rhs_at_point`
    changes.
  - No kernel reads an excised value: zero-weight taps are branched out or
    clamped to the point itself, because `0 · NaN = NaN`.
  - The zone is a shell a few cells thick, around `10⁴`–`10⁵` points
    against `10⁷` or more in the mesh. So it may be generic code without
    costing the main kernel's register budget.
- **The monitors that take stencils** — the gauge and ADM constraints and
  Löhner `τ` — mask the excised set widened by their reach,
  `max(G, ⌈√2 q/2⌉)` cells. This is a separate `monitor_mask`.
- **The error, the speed, the non-finite count and the horizon finder's
  guard** read every evolved point, the band included.
- **No new state writer:** the step limiter's `:excised` method is a no-op.

**Checks**, asserted wherever a problem is built:
- the margin `m ≥ G + 1` between the surface and the horizon;
- the singular set inside the core surface;
- every block within `G + (q + 2)/2` cells of the surface, on either side,
  on one level, so that no prolongation reads excised data into an evolved
  stencil. **(Amended in step X2b:** the build asserts `PLAN.md`'s wider
  `(G + q + 2) h` — `6h` at `q = 2`, `9h` at `q = 4` — which also covers a
  coarse neighbour's injected ghosts, `2G` fine cells deep.**)**

**Record rows:**
- the least margin of the outflow condition along the true normal, which
  must stay positive;
- the per-axis distribution: its minimum and the inflow-like count;
- the band's point and non-finite counts.

**The risks the steps are built to expose:**
- the inflow-like closures above;
- the **gauge drift**. GHSO2's excised hole drifted off the stationary
  background at `≈ 0.14/M` under prescribed sources and failed near `45 M`
  (`notes/sonic-surface.md`). This package's analytic layer holds the same
  drift at `8·10⁻⁵/M` because it holds the gauge inside. Excision gives that
  up **(predicted: the drift returns)**. Step X3's drift rows to `50 M`
  measure it, and the damped harmonic gauge driver is the fix [Possible
  extensions](#possible-extensions) names. **(Measured in step X3: it did
  not return** — the drift of `h_tt` at the horizon peaks near `40 M` and
  turns down as the layer's does, at `1.0–1.4×` it, to `50 M`.**)**;
- **closures without SBP**, whose stability rests on dissipation.
  **(Measured in step X3:** at `ε_KO = 1/2` nothing grows at any depth of the
  window to `24 M`.**)**

GHSO2's recipe applies unchanged: `ε_KO ≈ 0.5` and `γ0 ≳ 1/M`.

**What step X2b built (amended in step X2b**, `src/excision.jl` and the
variant through the package**).** The design above, with these choices:

- **The variant and its parameters.** `:excised` is in `INTERIOR_VARIANTS`
  and runs on both geometries: `Interior(…; variant = :excised)` excises
  `r < r_1`, `FittedSpec(…; variant = :excised)` the depth `d > 0` below the
  seed's offset surface. `r_0` — for the tracked geometry `thickness = n_L
  h` — is the core rule's depth and nothing else. The parameters are
  `Excision(T; upwind = (start, width) | nothing, dissipation = :msn)`, a
  field `excision` of `Interior`, `FittedSpec` and `FittedInterior`
  (refused for every other variant): the lopsided blend's start and width
  in cells below the horizon, off by default (`width = 0`), and the
  dissipation's closure. `GHCase` and `hole_case` take `excision` for the
  sphere. **A legacy `show` keeps the `repr` of every interior without
  excision parameters byte for byte the base's** — the recipe holds
  `repr(case.interior)`, and the new field would otherwise have refused the
  restart of every checkpoint written before it **(proposed in step X2b)**.
- **The classes**, built once with the problem (`build_excision`): the
  excised bit on owned points from the masks' predicate; one `fill_ghosts!`
  of a one-variable `FieldSet{T}` with even parity, replaying the problem's
  own schedule — a schedule belongs to a layout, not to a variable count —
  with the outer faces' ghosts given the predicate at their own positions;
  then a `stored = true` pass writing a `UInt8` per stored point: excised;
  **zone**, an owned point some stencil of the right-hand side reads an
  excised point through (the dissipation's `±G` along an axis, the mixed
  derivatives' boxes of half-width `q/2`); centered. A ghost that is not
  excised is marked centered, a placeholder nothing reads. The zone is the
  closure points only; the lopsided blend's thick shell is the main
  kernel's (X1's hand-over).
- **The main kernel's `:excised` branch** reads the class at the point:
  centered — the `:none` branch's call, `Centered`, or with the blend on
  the `Lopsided` provider, `Centered` with `adv` overridden (the argument
  itself through a branch where the weight is zero); excised — `du = 0`, no
  `F`; zone — nothing written. **`gh_zone_kernel!`**, launched right after
  it, runs over every owned point with a block-uniform early exit through a
  device `Bool` per block, and calls `gh_rhs_at_point` with a
  `ClosureProvider` built per zone point (`closure_provider`): its codes
  `k±` scanned from the class array, at most `G` reads a side; the table
  as a `NamedTuple` of device arrays (`closure_arrays`), which
  KernelAbstractions adapts field by field where a struct would need an
  Adapt rule this package does not depend on **(proposed in step X2b)**;
  `d1`, `d2`, `ko` contracted over the row's own nodes from the first
  product in ascending order; `dmix` nested, the inner closure's codes read
  from the class array at each outer node rather than stored per point;
  `adv` the table's lopsided row for the side `β_d` points to, blended.
  `lopsided_centered_weights` (`@generated`) is the open-line lopsided row,
  bit for bit the table's, so the two kernels' blends are one operator.
- **The blend's weight** is `λ = smoothstep((d/h − start)/width)` with `d`
  the depth below the horizon — the background's `horizon_min_radius` for
  the sphere (a degree-0 shape with `r_in = r_out`, which the clamp makes
  exact), the seed's shape for the tracked geometry — and `h` the surface's
  spacing, one number, so that `λ` is a function of position the two
  kernels evaluate alike (`ExcisionBlend`) **(proposed in step X2b)**.
- **The monitors.** `monitor_mask(p, t)` — the excised set widened by
  `W = max(G, ⌈√2 q/2⌉) h` (`2h` at `q = 2`, `3h` at `q = 4`), and on the
  tracked shape by `r_out − r_in` more, which bounds how much farther along
  the ray than `W` an excised point within `W` lies — is the default mask
  of `gh_constraint!`, `adm_constraint!` and `gh_indicator!`
  (`indicator_flags` takes a `mask`); the error, the speed, the non-finite
  count, the validity monitor and the horizon guard keep `interior_mask`,
  which counts the band. `W` strictly exceeds every monitor stencil's reach
  (`√2 q/2 < ⌈√2 q/2⌉`), so the boundary's rounding cannot let a tap in. The
  validity monitor's two bands are the band `[r_E, r_E + W)` and the `G h`
  beyond it (`layer_mask(int, t; band)`, `shell_mask(…; band)`).
- **The record's outflow rows** (`excision_rows`), every chunk, from the
  state at the band's points: `excision_band` and
  `excision_band_nonfinite`; `excision_normal_min`, the least `b_n/a_n − 1`
  along the surface's normal (radial for the sphere, central differences of
  the shape for the tracked geometry); X1's faces — an excised immediate
  neighbour along an axis — `excision_faces`, their least ratio
  `excision_axis_min` and the inflow-like ones `excision_inflow`
  (`b/a < 1`); and `excision_into`, the (band point, closure axis `k_s < G`)
  pairs whose shift points into the excised set, the build's refusal counted
  every chunk. A tracked case adds `excision_horizon_margin`. Every other
  variant's rows are `nothing`. They are written into a seven-variable
  field set of the excision's own, not into new `diag` slots, so that no
  other run's `diag` grows **(proposed in step X2b)**.
- **The checks**, each refusing by name: the margin `m ≥ G + 1` and
  `r_1 ≤ r_h,min − m h` (`offset ≥ m h` tracked) and the singular set inside
  the core rule's surface (`check_interior_radii`'s `:excised` methods, with
  no thickness requirement); every leaf within `(G + q + 2) h` of the surface
  on one level (`check_excision_mesh`); `ε_KO > 0`, a static hole, no range
  projection, and `m ≥ ⌈√3 G⌉` with a `Horizon` (`check_excision_case`);
  and from the census, no inadmissible zone point, `ε_KO > 0` at every zone
  point, and **no closure axis whose shift points into the excised set** —
  the spinning holes' refusal, by the physics.
- **The driver.** The geometry is built once — the case's sphere, or the
  seed's shape at `t = 0` on the run's mesh — and a restart builds the same,
  so there is no new carried state; a restart scatters the file's state
  before the classes are built, since the shift's refusal reads it.
  `regrid`, `adapt`, the rate keywords, `handover`, `target_source` and the
  `Π` post-pass are refused, each saying why; `chunk_interior` returns the
  interior unchanged; the level floor asks only for the margin. A tracked
  case finds and tracks every chunk, and `excision_horizon_margin` — the
  found horizon's least distance from the frozen surface, in cells — ends
  the run, with its record written, below `m − G/2`, the jump test's half a
  stencil **(proposed in step X2b)**. The step limiter's `:excised` method
  is a no-op: no fourth writer.
- `test/octant_runs.jl` takes `interior=excised` with `geometry=sphere|
  tracked`, `r_E=` or `margin=`, `r_0=`, `upwind=<start>,<width>` and
  `closure=`; `bench/stepping.jl` takes `BENCH_CASE=excised`.

**What step X4 changed (amended in step X4).** Step X4 merged `main`'s
spill-free right-hand side and the rotating octant into the integration
branch; the excision's operator is X2b's, its arithmetic `main`'s.
- **The kernels** are those of [One right-hand-side
  evaluation](#one-right-hand-side-evaluation), "One design after `main`'s
  rewrite": the closure provider plugs into `gh_rhs_head(S, …)` and
  `gh_rhs_pi(S, …)` through `gh_rhs_store!(du, o, sd, S, …)`, and the main
  kernel's `:excised` branch is the `:none` call at a centered point — also
  inside the blend's start, where the weight is zero. Every excision claim of
  X2b holds as it did, but one: the closure provider at a point with no
  excised tap is `Centered`'s to roundoff, not bit for bit (`main`'s head is
  compiled around it with other FMA fusions). The planted degenerate metric
  still changes no non-excised `du`, and the centered points' `du` is the
  `:none` kernel's at all 55 505 points of the fixture.
- **The rotating octant** ([The rotating
  octant](#the-rotating-octant-a-quarter-turn-about-z-added-2026-10-04)).
  The classes' bit is exchanged as a scalar — even parity, the identity
  rotation — so a ghost across the seam holds the class of the owned point
  it is the image of, and the two owned seam planes have one class at each
  point (`test/excision_tests.jl` says both). `hole_case(; octant =
  :rotating, interior = :excised)` and `test/octant_runs.jl octant=rotating
  a=… interior=excised …` build; at `a = 3/5` the build refuses the case by
  the physics, as X2b's refusal says — on the smoke's octant (`h = 1/16`,
  `r_E = 1`) 72 (zone point, closure axis) pairs with the shift into the
  excised set, the least `b/a = −0.499`.
- **The nested mixed derivative is not symmetric under `x ↔ y` (measured in
  step X4).** At a zone point `dmix(S, …, i, j)` takes the outer sum along `i`
  with the point's `i`-closure and the inner sum along `j` with each outer
  node's `j`-closure, always `i < j`. Near the surface the closures of the two
  axes differ, so `D_x(D_y f) ≠ D_y(D_x f)` at the truncation level, and the
  discrete operator is not equivariant under the diagonal reflection — nor,
  since a quarter turn is that reflection composed with a mirror, under the
  rotating seam's quarter turn. Measured on the smoke's octant (`a = 0`,
  `r_E = 1`, `h = 1/16`, `q = 4`) at `t = 1`: the evolved state violates the
  diagonal reflection by `3.0·10⁻³` at zone points and `1.9·10⁻⁴` elsewhere,
  on the mirror and on the rotating octant alike, where `:damped`'s does by
  `5·10⁻¹⁴`. The mirror octant is the full box's solution (the operator is
  mirror-equivariant); the rotating octant imposes a symmetry the operator
  lacks, so the two differ at that level: the record's `L∞` norms inside the
  horizon by `1.4·10⁻³` relative, the shells outside it by `9.5·10⁻⁵`, the
  `L2` norms by `10⁻⁷`; the classes, the census and the first row are the
  same. Not an instability, and not a seam defect: the two owned seam planes
  stay one state to `1.6·10⁻¹³`. **Proposed in step X4, for X6: the zone
  points' mixed derivative as the mean of both nestings, `½(D_i D_j + D_j
  D_i)`**, which on a scratch copy took the diagonal violation to `5·10⁻¹⁴`
  at zone points and made the two octants' records agree to the CSV's ten
  printed digits (`records.csv` to `10⁻¹²`). It costs a second nested sum at
  zone points only (the zone kernel is 1.5 % of a right-hand side on the
  octant), keeps the exactness of the closures, and changes X3's operator at
  zone points, which is why X4 records it rather than building it. **(Built
  in step X6**, the default, `Excision(T; mixed = :symmetric)`: the two
  octants, each stepped four times on its own, now hold one state to `0.8`
  eps where the single nesting left `1.7·10⁻⁶`, and X3's smoke row changes
  by at most `3.5·10⁻⁵` in the shells' `L2` norms outside the horizon —
  "What step X6 built"**)**.

**What step X3 measured, and the recommendation (amended in step X3).** The
static Kerr-Schild `a = 0` hole on the octant, on Symmetry's H200s at
`Float64`, against the exterior study's `:damped` and `:fitted` rows; the
numbers are under [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-on-the-static-hole-step-x3), "Excision on the
static hole (step X3)".
- **Excision is feasible here.** The variant ran on CUDA as built, its
  right-hand side within `5·10⁻¹²` of the CPU's. There is **no stability
  boundary** in X1's window: every depth from `r_E = M/2` to `13M/8` (24 to
  6 cells at `h = 1/16`), with and without the blend, runs to `24 M` and is
  stationary near the hole from `3–10 M` on — the 3D lego sphere, whose
  per-axis closures are up to 81 % inflow-like, behaves as X1's circle did.
- **The depth is set by accuracy: excise at a fixed radius well inside, `r_E
  = M/2`, without the blend (proposed in step X3).** What limits a shallow
  surface is leakage, not stability: without the blend the error just
  outside the horizon grows by a factor `e` every 3–5 cells the surface
  moves up, and it is 3–19× the `:damped` layer's at the same radius. At
  `r_E = M/2` — 24, 36 and 48 cells at `h = 1/16, 1/24, 1/32` — the
  closures' leakage falls faster than the truncation error and the exterior
  *is* `:damped`'s: at `h = 1/32` the two agree to three digits in every
  shell from `r = 2` out, in `M_irr`'s drift and in the drift of `h_tt`, and
  the orders are four. A surface at a fixed number of cells is not enough:
  at `r_E = M`, 24 cells at `h = 1/24` give `1.5×` `:damped`'s ℋ just
  outside, where 24 cells at `1/16` gave `0.34×`.
- **The lopsided blend is the lever for a shallow surface, not the
  default.** It makes the exterior independent of the depth from `M/2` to
  `5M/4` at `h = 1/16`, but its own lopsided stencils one cell below the
  horizon leak too: at `r_E = M` with X1's profile (`upwind=1,4`) ℋ just
  outside is `1.6×` `:damped`'s at `h = 1/24` and `2.2×` at `1/32`
  (`1.2×` at `1/24` when it starts four cells down, `upwind=4,4`), and on the
  H200 it costs 10–20 % of every right-hand side. Keep it off by default, as
  X2b built it.
- **The exterior matches `:damped`'s and beats `:fitted`'s.** At `r_E =
  M/2` ℋ in `[2, 2.25)` is `0.34×`, `0.92×` and `1.00×` the layer's at
  `h = 1/16, 1/24, 1/32`, the error there `0.87×`, `1.03×` and `1.00×`,
  every shell from `r = 2.25` out the same to two or three digits, and the
  mass drift `dM_irr/dt` the same to 2 %; against the best `:fitted`
  setups of the exterior study it is `9×`, `1.7×` and `2.6×` lower in ℋ
  just outside the horizon. Inside the horizon the band carries the
  closures' lower order (`7·10⁻³` at `h = 1/16`, converging at order 2),
  which nothing outside sees.
- **The gauge drift did not return.** GHSO2's excised hole drifted at
  `0.14/M` and failed near `45 M`; here, to `50 M`, the drift of `h_tt` at
  the horizon peaks near `40 M` and turns down exactly as the layer's does,
  at `1.4×` the layer's at `h = 1/16` and `1.0×` at `1/24` and `1/32`;
  `M_irr` peaks at `26 M` and then falls at `−3.7·10⁻⁹/M` (B; `−5.0·10⁻⁹/M`
  for A), below the truncation drift `:damped` has at this `h` (`2·10⁻⁸/M`)
  **(corrected in review**: this sentence said "flat to `10⁻⁸`"; the
  numbers are X3's entry's in `SINGULARITY_HANDLING.md`**)**, and ℋ near the hole changes by less than 0.4 % between
  `24` and `50 M`. The static gauge source holds the gauge without a layer
  inside; the damped harmonic gauge driver is not needed for this.
- **Cost**: an excised run at `r_E = M/2` costs what a `:damped` one costs
  on the H200 (`0–8 %` more per `M`), the zone kernel `0.6 %` of a
  right-hand side on the octant's 29 blocks; the blend `+16–24 %`.
- **What the moving round needs.** The frozen geometry and its refusals
  (`regrid`, `adapt`) were this round's simplification: a moving hole needs
  (1) values at the points its trailing side uncovers — an extrapolation
  fill along the motion, or a thin `:damped`-like layer behind the surface
  that hands them over — and the classes rebuilt when the surface moves; (2)
  the one-level constraint carried by a refinement floor that travels with
  the surface; (3) the depth measured here: the uncovered points start
  their life at the surface, `r_h − r_E` below the horizon, and the `3M/2`
  that `r_E = M/2` leaves is what keeps their transient inside. A boosted
  Kerr-Schild hole is possible (`KerrSchildSource` takes a velocity); G5's
  harmonic `a = 7/10` chart has only `0.3 M` between the horizon and its
  disk (X1), 7 cells at `h = 1/24`, which this round's leakage numbers say
  is far too shallow.
- **What the spinning holes need.** X1's frozen line found the per-axis
  closures unstable where frame dragging turns the shift into the excised
  set (`b/a < 0`, 2–18 % of the faces), and X2b refuses such a build by the
  physics. A closure for those faces — the normal-direction fill of
  [Possible extensions](#possible-extensions), or an extrapolation along
  `β` — has to be found stable on X1's models first. And the window ends at
  the inner horizon (`0.63 M` below the outer one at the equator at
  `a = 9/10`, `1.26 M` at `a = 3/5`): the depth that kept the leakage below
  the truncation error here, about `3M/2` and at least 24 cells, does not
  fit at `a = 9/10`, where 24 cells in `0.63 M` need `h ≈ 1/40` and the
  surface sits right above the inner horizon.
- **The next round (proposed in step X3): the static spinning hole**, on
  the rotating octant (on `claude/octant-mode-spinning-bh-75bd22`, not yet
  on this branch) against its `:damped` and `:fitted` rows at `a = 3/5` and
  `9/10` — a host-side step for the `b/a < 0` faces first, then the
  runs. It is the smaller step (no trailing side, the classes frozen), it is
  where the layer is weakest (the `:fitted` target failed at `a = 9/10`), and
  spin is the one thing the proof of concept needs that excision cannot do
  yet. Moving holes after it.

**The frame-dragged faces (step X5): a rule, the window, and go for
Kerr-Schild `a = 3/5` (measured in step X5).** Host-side, no kernel change:
`test/excision_model.jl`'s `margins=window` and `model2d=spin…` parts, and
the extrapolation's weights in `src/stencils.jl` (`extrapolation_weights`,
`extrapolation_table`). The tables are under [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-the-frame-dragged-faces-step-x5), "Excision: the frame-dragged faces (step X5)".

- **What the spinning hole adds.** Frame dragging gives `β` an azimuthal
  part, so along an axis nearly tangent to the lego surface the shift can
  point *into* the excised set: `b/a = −s β^d/(α√γ^{dd}) < 0` at a closure
  axis (`k_s < G` on the excised side `s`), the faces X2b refuses. On the
  sphere at `a = 3/5` they are about `1450`, `2800` and `6800` closure axes
  at `h = 1/24, 1/32, 1/48`, at every depth — 2 % of the closure axes near
  the horizon, 12 % near the inner horizon — only along `x` and `y` (a
  `z`-face's shift is radial), with the least `b/a` falling from `−0.25`
  near the horizon to `−2.6 … −3.9` at `r_E = 0.65`; both characteristics
  enter (`b/a < −1`) on 0.3–2 % of the faces below `r_E ≈ 0.9`.
- **The per-axis closures fail there, near the inner horizon.** On the
  equatorial plane, where frame dragging is in the plane, the per-axis
  family has a surface mode growing at `+1.2` to `+4.8/M` at `r_E = 0.65`
  (`b/a` down to `−2.2 … −2.6`) at `h = 5/48`, `5/64` and `5/96`, at `q = 4`
  and 2 and `ε_KO = 1/2` and `1` (all but `q = 2`, `ε_KO = 1`, `5/64`), and at
  `r_E = 0.75` on the coarser plane (`b/a = −1.31`, `+0.02` to `+1.75/M` at
  three of the four `(q, ε_KO)`); noise blows up at `5 M` (`q = 4`) and
  `26 M` (`q = 2`) at `r_E = 0.65`, `h = 5/128`, and at `5–6 M` (`q = 4`) at
  `5/256`. Where `−1 < b/a < 0` the plane is stable — the frozen
  line's weak growth there (`10⁻⁴` to `5·10⁻²/h` for `b/a` from `−0.02` to
  `−0.5` at `q = 4`, `ε_KO = 1/2`) does not survive the
  neighbouring faces — but whether a face with `b/a ≲ −1` breaks it depends
  on the staircase (`r_E = 0.75` holds at `5/64` and `5/96` with the same
  `−1.31` that breaks `5/48`): the per-axis family is not a rule for the
  deep part of the window.
- **The extrapolation everywhere is not either**: X1's `extrap` grows at
  `+2.0` to `+3.2/M` on the surface at `r_E = 0.65`, `h = 5/48`, unless the
  lopsided blend is on, and ten to twenty-five times the layer's rate
  without dissipation, as at `a = 0`. And extrapolating the advection at *every*
  closure axis — the faces where the shift points out included — is
  unstable at every `r_E`, `+2.6` to `+18/M`: the extrapolation belongs only
  where the axis cannot reach its upwind side.
- **The hybrid holds every face.** Per-axis closures where the shift points
  out of the excised set and, where it points in, the centered stencil with
  its excised taps extrapolated along the lattice direction nearest the
  normal — for every operator along that axis (`hybrid`) or for the
  advection alone (`hybrid-adv`) — has its rightmost eigenvalue within
  `0.002/M` of the `:damped` layer's on the same plane, or below it, at every
  `r_E` from `0.65` to `1.7`,
  at `q = 4` and 2, `ε_KO = 1/2` and `1`, `h = 5/48`, `5/64` and (`q = 4`,
  `ε_KO = 1/2`) `5/96`, with and without the lopsided blend: `−0.075` to
  `−0.19/M` against the layer's `−0.077` to `−0.11/M`, the box's own slowest
  decay (equal to it at `r_E = 0.65`, below it where the hole is larger).
  RK4's step is the layer's (`cfl = 1.86–1.89` at `q = 4`, `2.06–2.08`
  at `q = 2`, `ε_KO = 1/2`). Noise falls at the layer's rate to `100 M`
  (`−0.083` to `−0.099/M` against `−0.088`, `−0.091/M`) at `h = 5/128` and
  `−0.073` to `−0.088/M` against `−0.074`, `−0.086/M` at `5/256`, where the
  per-axis closures blow up at `r_E = 0.65` in `5–6 M` (`q = 4`). The two
  hybrids agree to three digits everywhere, so
  **extrapolating the advection alone is enough**. The rule is robust: a
  threshold of `b/a < 1/2` instead of `0`, the extrapolation's degree (1,
  2, `≤ q`) and the dissipation closure change nothing at `ε_KO = 1/2`.
  Without dissipation it grows as the layer does (`+0.12` to `+0.23/M`
  against `+0.11` to `+0.14`): **`ε_KO > 0` stays required.**
- **In 3D the rule always finds its sources.** On the lego sphere and the
  tracked offset surface at `h = 1/24, 1/32, 1/48` and every depth, every
  excised tap of a frame-dragged axis's advective stencil has at least two
  non-excised points beyond it along the nearest of the 26 lattice
  directions inside the point's `G`-box — three for all but 8–124 of them
  (at most 1.7 %, all at `r_E ≤ 0.8`) — the first at most two steps out and
  the last at most `G = 3`.

**The rule X6 builds (proposed in step X5): the frame-dragged faces' rule,
in the advection only.** At a zone point and an axis `d`, if a side `s`
with `k_s < G` has the shift pointing into the excised set, `−s β^d < 0`
in the state the problem is built on — X2b's census condition, its refusal
turned into a rule — then the provider's `adv` for that axis (the
derivative beside `β^d` in `β^k ∂_k h` and `β^k ∂_k Π`, and nowhere else)
returns the centered `D₁` with every excised tap `Q = x + j e_d` replaced by
`Σ_i w_i u(Q + (k₀ + i − 1) e)`: `e` the lattice direction (of 26) nearest
the excision surface's outward normal at `Q`, `k₀` the first step out of
the excised set, at most three consecutive non-excised sources inside the
point's `G`-box, `w = extrapolation_table(T, Val(q))[:, k₀, n]`. Every other
stencil at the point — `d1` for the other terms, `d2`, `dmix`, `ko` — stays
the per-axis closure. With the lopsided blend on, the lopsided row at such
an axis is the open one with its excised taps filled the same way (what the
plane measured; it is not needed). Kerr-Schild `a = 0` has no such axis, so
its runs are X2b's bit for bit. The criterion is `b/a < 0` and not the
`b/a ≲ −1` where the plane fails: it is the frozen line's, it is the
census X2b already computes, and the plane is as stable with it as with
`b/a < 1/2`. **(Built in step X6** as written, with the sources capped at `G`
steps — the table's bound — and the rule a second launch of the zone
kernel's; "What step X6 built", below.**)**

**What X6 needs from the classes** (proposed in step X5; **built in step
X6**, the bits and the codes in one second `UInt8` array, the direction
coded at every excised stored point, the weights a kernel argument beside
the closure table — "What step X6 built", below):
- **A rule bit per zone point and axis** (three bits), set at build from
  the state's shift where X2b's census now counts `excision_into`. The class
  is a `UInt8` with three values, so the bits fit beside it; or a second
  `UInt8` array of the same layout.
- **The extrapolation's direction per excised point** within `q/2` of a
  zone point along an axis: a code `0…25` for the nearest lattice direction
  to the surface's normal there — `x − c` for the sphere, `excision_normal`
  for the tracked shape — built in the classes' pass from the frozen
  geometry, so that the zone kernel reads a code and not the geometry (the
  direction is not "is it excised"; the classes stay the only answer to
  that). The sources' classes are read from the class array, as the codes
  `k±` are.
- **The weights**: `extrapolation_table(T, Val(q); degree = 2)`, an `isbits`
  `3 × G × 3` array (216 bytes at `q = 4`, `Float64`), rounded once; small
  enough for a kernel argument beside `closure_arrays`. It holds the
  sources up to `G` steps out, `k₀ + n − 1 ≤ G`, which is as far as any
  frame-dragged tap's sources go on the sphere and the tracked surface.
- **The refusals and the record**: an excised tap with no source in the
  `G`-box is refused at build by name (none occurs on the sphere or the
  tracked surface at `a = 3/5`); `excision_into` keeps counting, every
  chunk, the axes whose shift points in, and a second count — those whose
  sign disagrees with the rule bit — must stay zero. The faces per rule go
  into the outflow rows.

**The window and the resolutions for X7 (proposed in step X5).** The sphere
geometry, as X3 ran:
- **Normal outflow** holds on the sphere from `r_E = 0.641` (`+0.052` at
  `0.65`, the inner horizon being `0.632` on the equator) to the horizon,
  largest at `r_E ≈ 0.9–1.0` (`+0.43`); the tracked offset surface has it
  from `m = 1` (X1) down to the inner horizon.
- **The core rule's sphere must lie between the ring and the surface**,
  `0.6 < r_0 < r_E`: `check_interior_radii` refuses `r_0 ≤ 0.6` by name, so
  `test/octant_runs.jl`'s default `r_0 = r_E/2` is refused below
  `r_E = 1.2` and X7 passes `r_0=` itself — midway, `(0.6 + r_E)/2`.
- **At least 16 cells at the poles at `h = 1/24`** (Erik's floor) is
  `r_E ≤ 1.133`. X3's lesson is that the depth in `M`, not in cells, sets
  the leakage, so the deepest healthy surface is the target: **the depth
  scan at `h = 1/24` over `r_E = 0.70, 0.80, 0.90, 1.00, 1.133`** — 26.4,
  24, 21.6, 19.2 and 16 cells at the poles, 28.7 … 18.3 on the equator —
  without the blend; **production at `r_E = 0.80`, `r_0 = 0.70`** until the
  scan says otherwise: normal margin `+0.38`, `1.0 M` below the poles'
  horizon (24, 32 and 48 cells at `h = 1/24, 1/32, 1/48`), `1.1 M` below the
  equator's, the frame-dragged faces at `b/a ≥ −1.4`, and `4.8` cells of
  `1/24` between the ring and the surface.
- **The lopsided blend is not needed**: no family's stability depends on it
  except X1's `extrap`, which the rule does not use; it costs RK4's step on
  the plane (`1.88 → 1.53–1.58` at `q = 4`) and the H200 10–20 % (X3). Off,
  as X3 recommended.
- **`ε_KO = 1/2`**, as X3 ran: `1` is as stable and shortens the step.

**What step X6 built (amended in step X6**, `src/excision.jl`; its numbers
under [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-the-frame-dragged-faces-in-the-zone-kernel-step-x6), "Excision: the frame-dragged
faces in the zone kernel (step X6)"**).** X5's rule in the zone kernel, the
zone points' mixed derivative symmetric in its two axes, and the spinning
hole on the rotating octant end to end:

- **The codes.** `ExcisionData` carries `codes`, a `UInt8` array of the
  classes' layout — a second array beside the three-valued classes rather
  than bits in them, so that every `== CLASS_…` test of the classes stays as
  it was **(proposed in step X6)**:
  - at a **zone point**, three **rule bits** (bit `d − 1` for axis `d`), set
    by the census where X2b refused: a side `s` with `k_s < G` whose `b/a =
    −s β^d/(α√γ^{dd})` is negative in the state the problem is built on;
  - at an **excised point**, the **direction code** `1…26` of the lattice
    direction nearest the surface's outward normal there
    (`lattice_direction`, `direction_code`: the largest `e·n/|e|`, the first
    in step X5's order on a tie), written for **every** excised stored point,
    ghosts included, by a fourth pass over the stored points from the frozen
    geometry (`excision_normal`: radial for the sphere, the shape's gradient
    for the tracked surface) **(proposed in step X6**: X5 asked for those
    within `q/2` of a zone point along an axis; all of them is one pass with
    no test, and a ghost's code is computed at its own position, which on
    an octant is its owner's image, so that the direction turns with the
    data across the seam and the mirror**)**.

  The two sets are disjoint. The kernels read the codes and never the
  geometry; the classes stay the only answer to "is it excised", and the
  sources' classes are read from them.
- **The provider.** `DraggedProvider` wraps a zone point's
  `ClosureProvider` with the codes, `extrapolation_table(T, Val(q))` and the
  point's rule bits. `d1`, `d2`, `dmix`, `ko` — and `adv` along an axis whose
  bit is clear — are the closure provider's. Along an axis whose bit is
  set, `adv` is `1/h` times the centered `D₁` contracted in `axis_stencil`'s
  order with every excised tap `Q` replaced by `Σ_i w[i, k₀, n] u(Q + (k₀ + i
  − 1) e)` (`_tap_sources`: `e` from `Q`'s code, `k₀` the first non-excised
  step, `n ≤ 3` consecutive sources, every one inside the point's `G`-box
  **and at most `G` steps out**, which is what `extrapolation_table` holds —
  X5's census never needed more, and the cap makes the table's bound a
  property of the scan **(proposed in step X6)**). Where the axis's centered
  `D₁` reaches no excised point (`k_s = q/2`) it is the `∂f_d` it is handed.
  With the lopsided blend on it is `(1 − λ)` that plus `λ` times the open
  lopsided row filled the same way, as X5 measured. **At `q = 2` the sources
  stop at `G = 2` steps, so there are two at most and the rule's
  extrapolation is linear** (at `q = 4` it is quadratic wherever three fit:
  nine taps in ten on `test/excision_tests.jl`'s ball).
- **A second launch, not a branch** **(proposed in step X6)**.
  `gh_dragged_kernel!` runs after `gh_zone_kernel!`, over the blocks holding a
  frame-dragged axis (`drag.blocks`, a device `Bool` per block), and
  overwrites the `du` of the zone points with a bit set — the zone kernel's
  work there, 5–12 % of the zone at `a = 3/5`, is thrown away. So the zone kernel
  is the same compiled code with and without the rule: "the rule changes
  nothing where no axis is frame-dragged" is a statement about which points
  the second launch writes, and needs nothing from the compiler ([One
  right-hand-side evaluation](#one-right-hand-side-evaluation), "One design
  after `main`'s rewrite": a second provider compiled into one body changes
  how the head is fused). Where the build finds no frame-dragged axis
  — Kerr-Schild `a = 0` — `drag` is `nothing` and the launch does not
  happen, so those runs are the zone kernel's alone. On a device the second
  kernel also keeps the rule's registers out of the zone kernel.
- **The refusal is narrowed** to what the rule does not cover: an excised
  tap the rule's stencils read — its centered `D₁`'s, to `q/2`, and with the
  lopsided blend the lopsided row's, which reaches `G` into the excised side
  of a frame-dragged axis — with no source (the census runs the kernel's own
  `_tap_sources`). X5 met none at `a = 3/5`, nor does any surface the tests
  or X7 build without the blend; on a coarse octant (`h = 1/4`, `r_E =
  17/25`, `q = 4`) the blend's row has two, which `excision_tests.jl` uses
  as its witness.
- **The record** adds three outflow rows, every chunk, from the band:
  `excision_dragged` (the (zone point, axis) pairs with a rule bit — a
  constant of the problem), `excision_faces_dragged` (X1's faces on those
  axes: the faces per rule are it and `excision_faces` less it) and
  `excision_flips` (the axes whose shift's sign now disagrees with the bit,
  pointing in without the rule or out with it), computed with the census's
  own function. `excision_into` keeps counting the axes whose shift points
  in. **The rule bits are the build state's, and a restart rebuilds them
  from the state it restarts from** — the checkpoint holds no bits — **so a
  chain of jobs is the uninterrupted run exactly while `excision_flips` is 0
  at the checkpoint's row** (the checkpoint is written after the row); the
  run is not stopped when it is not, and the record says so **(proposed in
  step X6)**. `test/excision_tests.jl`'s spinning run and its chain of two
  jobs are `isequal`.
- **The mixed derivative** (X4's proposal, taken into X6 by the
  orchestrating session, for Erik to confirm in review): `Excision(T; mixed
  = :symmetric)` by default, `ClosureProvider{…,SYM}`'s `dmix` is `½(D_i D_j
  + D_j D_i)` — the nesting along `i` and the one along `j`, each with its
  inner closures at its own outer nodes — **wherever the point's `(q + 1)²`
  box in the `(i, j)` plane meets the excised set; where it does not, both
  nestings are the centered tensor product and the one along `i` is taken**,
  `mixed_stencil`'s arithmetic, so that the closure provider's contractions
  at a point with no excised tap stay `isequal` to `Centered`'s **(proposed
  in step X6)**. `a + b == b + a` in floating point, so a point and its
  image across the diagonal take the same value bit for bit; the operator is
  equivariant under `x ↔ y` and the rotating seam's quarter turn, and the
  rotating octant evolves as the mirror octant does. `mixed = :nested` is
  steps X2b–X5's operator, bit for bit (measured: X3's smoke row identical
  in every column), and **prints as `Excision` printed before the field**, so
  that a checkpoint written before step X6 restarts when that is asked for by
  name and its recipe refuses the new default **(proposed in step X6)**.
  `test/octant_runs.jl` takes `mixed=`.
- **`test/octant_runs.jl octant=rotating a=3/5 interior=excised …` runs**,
  with SimWatch: the build no longer refuses, and the CSV, `records.csv` and
  `[extra.excision]` carry the three rows.

**What step X7 measured, and the recommendation (amended in step X7).** The
excised static Kerr-Schild `a = 3/5` hole on the rotating octant, on
Symmetry's H200s at `Float64`, with X5's rule as X6 built it, against the
`:damped` and `:fitted` reference rows of [Robust stability on the
octant](#robust-stability-on-the-octant-measured-2026-10-02) ("A spinning hole
on the rotating octant, `a = 3/5`"); the tables are under [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-of-the-static-spinning-hole-step-x7), "Excision of the static spinning hole (step
X7)".
- **The production depth runs; the shallow end of the window does not.** The
  depth scan at `h = 1/24` (X5's window, `r_0` midway between the ring and
  the surface) ran `r_E = 0.70, 0.80, 0.90, 1.00` to `10 M` with and without
  noise: the band stationary from `2 M`, no non-finite value, `excision_flips
  = 0`. `r_E = 17/15` — 16 cells below the poles, Erik's floor — **blows up**
  at `4.5 M` (a `DomainError` in the kernel), with and without noise. The
  production rows at `r_E = 4/5` (24, 32 and 48 cells below the poles at
  `h = 1/24, 1/32, 1/48`) ran to `24 M`, and the `1/32` row to `64 M`.
- **The failure is a lego corner at the pole, not the frame-dragged rule
  (diagnosed in step X7).** It reproduces bit for bit on the CPU on a small
  rotating octant with the same spacing at the hole (`L = 8`), where the
  points of largest error were printed every quarter `M`. The growth, about
  `2/M` from `2 M`, sits at the rim of the excised ball's topmost lattice
  layer — at `r_E = 27.2 h` that layer, `z = 27 h`, is the disk `x² + y² ≤
  10.8 h²` — at the zone point `(3h, 2h, 27h)` and its image across the
  diagonal, `0.04` cells outside the sphere, whose `−x`, `−y` and `−z`
  neighbours are all excised (`k⁻ = 0` on all three axes). There frame
  dragging nearly cancels the radial shift's component along the nearly
  tangent `y` axis: `b/a = +0.011` (`+0.078` at the same point at `a = 0`).
  So the per-axis closure is X1's marginal inflow-like kind at almost zero
  speed, and in that corner it is not marginal. The point has no rule bit.
  - What does not change it: the nested mixed derivative (grows too), the
    noise, the lopsided blend. `ε_KO = 1` halves the rate. Extending the rule
    to `b/a < 0.02` or `0.05` makes it grow faster, and to `b/a < 1/2` the
    whole surface blows up within `1 M` — X5's "at every closure axis"
    control, in 3D.
  - What removes it: the same lattice at `a = 0`; the same radius at `h =
    1/32`; `r_E = 1.10` or `1.15` at `1/24` (other caps). `r_E = 21/20` at
    `1/24`, whose topmost layer is the same disk, fails at `5 M`.
  - **What cures it, on a scratch copy: shaving the lego corners.** Excising
    every point whose three inward axis neighbours are excised — about 200
    points of the octant's band at `1/24`, among them the two growing ones —
    makes `r_E = 17/15` and `21/20` stationary to `8 M` (the largest error
    `0.117` and `0.154`, the equator's) and leaves `r_E = 4/5` as it was
    (`2.07`). Not built: the scratch copy changed only the classes' bit, so
    its masks and monitors still saw the sphere.

  It is the staircase's lottery X5 met on the plane, in three dimensions:
  whether a run meets it depends on which lattice points the sphere leaves at
  its poles, not on the depth alone.
- **At `r_E = 4M/5` the exterior is the `:damped` layer's from `h = 1/32`
  on.** At `24 M`, ℋ just outside the horizon, `[1.90, 2.25)`, is `2.09×`,
  `1.15×` and `1.00×` the layer's at `h = 1/24, 1/32, 1/48` (at `1/48`
  against a `:damped` row run for the comparison), converging at order
  `7.0/4.4` where the layer's converges at `4.9/4.0`; the error there is
  `0.87×`, `1.06×`, `1.00×`; every shell from `r = 2.25` out is within 5 % at
  `1/32` and to `r = 5` at `1/24` (where the start-up pulse still makes the
  far shells' ℋ `1.2–1.5×`), and the same to three digits at `1/48`. To
  `64 M` at `1/32` it stays so: ℋ just outside the horizon `1.15×` at `24`,
  `40` and `64 M`, the error there `1.04–1.06×` as both grow with the drift
  of `J`. So the closures' leakage falls faster than the truncation error,
  as at `a = 0` (X3's `r_E = M/2`), and by `1/32` it is below it. The depth
  scan says the same at `1/24`: `r_E = 0.70` and
  `0.80` give the same exterior, and `0.90` and `1.00` ℋ just outside the
  horizon `1.8×` and `2.4×` theirs.
- **The spin drifts as the layer's does: truncation error at order four.**
  At `24 M`, `J − a` and `dJ/dt` are `0.82×` and `0.90×` `:damped`'s at
  `h = 1/24`, `1.03×` and `1.03×` at `1/32`, and the layer's to three digits
  at `1/48` (`dJ/dt = 1.07·10⁻⁸/M` in both); the orders of `dJ/dt` are
  `3.6/4.1` (the layer's `4.1/4.0`). **To `64 M` at `h = 1/32`, `J` rises
  at `5.56`, `5.25` and `5.30·10⁻⁸/M`** over `8–24`, `24–40` and `40–64 M`,
  against the layer's `5.42`, `5.11` and `5.16·10⁻⁸/M` — `1.03×` in every
  interval — to `J − a = 3.69·10⁻⁶` (the layer: `3.58·10⁻⁶`), and `M_irr`
  falls with it as the layer's does (`−2.71·10⁻⁷` below Kerr's at `64 M`,
  the layer `−2.67·10⁻⁷`). The drift of `h_tt` at the horizon is the
  layer's to 1 %. Excision neither causes nor cures the drift of `J`: it is
  the scheme's, outside the hole.
- **The band inside the horizon carries the closures' low order, and it
  stays there.** The largest error of a run, `2.1`, `1.1` and `0.60` in `L∞`
  at `h = 1/24, 1/32, 1/48`, sits in the closure band on the equator, in
  `Π_tt` at frame-dragged zone points next to the surface, where the data are
  steepest (Kerr-Schild's radius `0.53` there, `H ≈ 3.8`); it is stationary
  from `1 M`. In `L2` ℋ in `[r_E + 3h, 3/2)` is `4.5·10⁻²`, `2.8·10⁻²`,
  `1.4·10⁻²` (order `1.7/1.6`) and the error `8.9·10⁻³`, `4.8·10⁻³`,
  `1.8·10⁻³` (order `2.2/2.4`); in `[3/2, 1.90)`, which the layer evolves
  too, ℋ is `4.9×` and `11×` the layer's at `1/24` and `1/32` and converges at
  order `6.7/8.9`. A deeper surface is rougher: at `r_E = 0.70`, `0.1 M`
  outside the ring on the equator, the band's ℋ is `4×` that at `0.80` and
  its `L∞` error `14`, with the same exterior — which is why `4/5` and not
  `7/10`.
- **The outflow rows** are constant for the whole run: the normal margin
  `+0.381`; the least per-axis `b/a` `−1.38`, `−1.17`, `−1.39` at `h = 1/24,
  1/32, 1/48` (X5's census to its digits); the frame-dragged axes 194, 365
  and 870 (X6 predicted about 190, 355, 850) and their faces 77, 140 and 317
  of 924, 1623 and 3594 (`excision_faces_dragged` against
  `excision_faces`); half the faces inflow-like; and **`excision_flips = 0` at
  every row of every run**, so the chained rows are the uninterrupted runs.
- **Cost.** A right-hand side of the excised hole is `2.2–2.6 ns` a point on
  the H200, `0.47–0.48×` the `:damped` layer's on the same mesh (`main`'s
  `:damped` branch still spills, X4). The zone kernel is 1.6–3.1 % of it and
  the rule's launch 2.5–3.2 % — **X6 predicted 0.6 % and 0.3 %**: the launch
  costs `1.6–3.9 ms`, growing with the frame-dragged points and the block
  holding them, not a fixed latency. A run costs `62–67`, `148` and `112 s`
  per `M` at `h = 1/24, 1/32, 1/48`, the analysis every chunk included — a
  third of what X6 predicted from X3's rates, the spill-free kernel's gain —
  and the `:damped` row at `1/48` `165 s`: an excised run is `0.68×` a
  layer's on this kernel.
- **The recommendation (proposed in step X7): fix the polar corner first,
  then the static `a = 9/10` hole; moving holes after; do not stop.**
  Excision is now what the layer is for a static spinning hole with an
  analytic interior, at the same cost or less. It is worth carrying on where
  the layer has no good target — `a = 9/10`, where `:fitted` is unstable and
  `:damped` needs `h = 1/96` — and to the moving hole. What each needs:
  - **The polar corner** (one agent step, a day): build the corner shave
    that cured it on the scratch copy — a point whose three inward axis
    neighbours are excised is excised — as a property of the excised set
    itself, so that the classes, the masks of the monitors and the horizon
    guard, and the census agree (the classes stay the one answer to "is it
    excised"); on both geometries, decided once at the build. Its tests: the
    exterior bit for bit where no corner is shaved, the planted degenerate
    metric, and the lottery itself — every `r_E` of X5's window at `h =
    1/24, 1/32, 1/48` built and run to `10 M` on the small rotating octant
    (`L = 8`), which reproduces the H200's rows bit for bit at minutes a row
    on the CPU — plus one H200 row at `r_E = 17/15`, `1/24`. A
    frozen-coefficient spectrum of a patch around the cap is the analysis
    that would say why, if the shave does not hold at another lattice.
  - **`a = 9/10`**, after the corner, **on the tracked offset surface, not
    the sphere.** Normal outflow ends at the inner horizon, at `ρ = 1.06` on
    the equator, and the horizon is at `1.436` at the poles, so a sphere is at
    most `0.37 M` below the poles — shallower than `r_E = 1` here, whose
    exterior was `2.4×` the layer's at `1/24`. The tracked offset surface
    (X2b, frozen) has normal outflow to `0.63 M` below the horizon, but only
    just: X1 measured its margin peaking at `+0.079` half an `M` down, against
    `+0.38` at this round's `r_E = 4/5`. That is the near-null inward speed
    the `a = 9/10` study on the rotating-octant branch suspects in its oblate
    *layer*, unstable 24 cells (`0.25 M`) below the horizon at `h = 1/96` and
    stable at 12 (`e1bb8eb`, not on this branch). So the surface's depth is a
    stability question again, and the scan has to bracket it, from `0.125`
    to `0.5 M` below the horizon: `m = 6, 9, 12, 18, 24` at `h = 1/48` (36
    blocks, 31.9 M points, `≈ 112 s/M`; five `10 M` rows, 1.6 H200-hours),
    then production at `1/48`, `1/64` (75.5 M points,
    `≈ 350 s/M`) and on that study's `1/96` mesh (52.8 M points, `≈ 370 s/M`)
    to `24 M`, against its `:damped` row at `1/96` (ℋ just outside the
    horizon `2.9·10⁻⁸`, `J` drifting at `2.5·10⁻⁹/M`): about 7 H200-hours.
    The rule's census at `9/10` (X1: `b/a` down to `−1.2` within half an `M`
    of the horizon) is the other thing to watch.
  - **Moving holes**, after both: X3's list — values at the points the
    trailing side uncovers, the classes rebuilt as the surface moves, the
    one-level floor travelling with it — and two things this round adds: the
    rule bits are the build state's, so a moving surface rebuilds them; and a
    moving surface passes through every lattice configuration, so the polar
    corner must be solved, not avoided by the choice of `r_E`.
  - **Stopping** is not recommended: nothing measured here is worse than the
    layer outside the horizon, and the two open problems — the corner and the
    oblate surface — are bounded.

**What step X8 built (amended in step X8**, `src/interior.jl`'s
`ShavedMask` and `src/excision.jl`; its numbers under
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#the-polar-corner-step-x8),
"The polar corner (step X8)"**).** X7's cure, built as a property of the
excised set:

- **The rule.** A lattice point of the surface's level that the geometry
  leaves outside is excised too when its three neighbours **one step toward
  the excision center**, one along each axis, are inside the geometry's
  excised set. "Toward the center" is per axis, from the sign of `x_d − c_d`,
  so it holds for a hole anywhere (X7's scratch copy used the sign of `x_d`,
  right only at the origin); a coordinate on the center's plane has no inward
  neighbour along its axis, and such a point is not shaved — on the rotating
  octant the seam planes and the mirror plane are center planes. A shaved
  point lies within `h/√3` of the geometric surface along the ray (`0.573 h`
  at most over every lattice sphere of 1 to 90 cells).
- **One pass, not a fixed point (proposed in step X8).** The rule iterated
  does not stop at the corners: a shaved rim point makes its tangent
  neighbour a corner, and the closure of a lattice sphere under the rule
  grows along the rims of its layers toward square cross-sections. On X7's
  surfaces it takes 11 to 51 passes and reaches 3.9 to 17.7 cells outside the
  sphere (`r_E = 4/5 … 17/15` at `h = 1/24`, `17/15` and `1.7` at `1/48`),
  which would eat the margin to the horizon. One pass excises the corners
  the geometry made — 217 on the open octant at `r_E = 17/15`, `h = 1/24`, X7's
  scratch copy's count — and leaves the corners of the shaved set (192 there),
  points with `k⁻ = 0` along all three axes again: the shave changes **which**
  corners there are, not that there are some. Whether that is enough is what
  "What step X8 measured", below, answers: not at every depth.
- **A predicate every mask evaluates, not a lookup of the classes (proposed
  in step X8).** `ShavedMask(base, h)` is the geometry's mask (`InteriorMask`
  for the sphere, `ShapeMask` for the frozen tracked surface) with the rule
  applied on the lattice of spacing `h`, the surface's (the one level
  `check_excision_mesh` asserts there); `excised_mask(interior, t, h)` builds
  it where `Excision(T; shave)` asks. Everything that asks "is it excised"
  takes a position — the error's, the speed's, the non-finite count's and the
  validity monitor's kernels, and above all the horizon finder's footprint
  guard, which TreeAMR asks at stencil positions it computes itself and which
  no class array can answer — so the shave is expressed where they all are:
  `build_excision`'s first pass evaluates `ShavedMask` at every owned point,
  as X2b's evaluated the masks' own predicate, and the masks, the guard and
  `test/octant_runs.jl`'s noise exclusion evaluate the same function at the
  same positions. The classes stay the single source of truth: the zone, the
  census, the rule bits and the direction codes read them as before, and a
  shaved point's direction code is its own position's.
- **The problem's masks** are `evolved_mask(p, t)`: the interior's own
  `interior_mask` for every problem but an `:excised` one whose build shaved
  a point (`ExcisionData.nshaved > 0`), and the shaved set's mask for that one
  — the default of `gh_error!`, `max_speed`, `evolved_nonfinite`,
  `validity_rows`' evolved region and the horizon finder's provider. Where the
  build shaved nothing the two masks agree everywhere and the interior's own
  is taken, so such a problem's analysis is steps X2b–X7's arithmetic. The
  monitors that take stencils widen their mask by `W + h` where a point was
  shaved (`monitor_mask`), the excised set reaching `h/√3` beyond the
  surface; the validity monitor's band `[r_E, r_E + W)` leaves the shaved
  points out (`BothMask`). The guard's `stencil_hits` for a `ShavedMask`
  enumerates the footprint unless its nearest point is `2h` beyond the
  geometric set — the predicate's own fast path, so the shortcut is exact.
- **The core rule** needs nothing: the shave adds points only outside the
  geometric surface, where the initial data are the analytic solution at the
  point itself as at every evolved point, and the core rule's sphere stays
  inside the excised set.
- **The checks.** With the shave a `Horizon` asks `m ≥ ⌈√3 G + 1/√3⌉` (still
  6 cells at `q = 4`; 5 at `q = 2`, where it was 4) **(proposed in step
  X8)**. `check_excision_mesh`'s `(G + q + 2) h` is not widened: a
  prolongation needs `G + (q + 2)/2` cells, and the neighbourhood exceeds
  that plus `h/√3` by `(q + 2)/2 − 1/√3` cells.
- **The switch and the recipe.** `Excision(T; shave = true)` is the default;
  `shave = false` is steps X2b–X7's excised set bit for bit (the fixture's
  classes are X2b's 3743 excised and 2192 zone points, and the small rotating
  octant at `r_E = 17/15` reproduces X7's diagnostic to its four printed
  digits). Without the shave the struct prints as steps X6–X7 printed it, and
  with `mixed = :nested` as well as steps X2b–X5 did, so that every earlier
  checkpoint restarts under its own excised set when that is asked for by
  name and its recipe refuses the shaved one **(proposed in step X8)**.
- **The record** has `excision_shaved`, the owned points the shave excised —
  a constant of the problem, counted by the census (`EXM_SHAVED`, the
  excision's eleventh monitor slot); `test/octant_runs.jl` writes it into the
  CSV, `records.csv` and `[extra.excision]`, takes `shave=on|off`, and keeps
  its noise off the shaved set and its `in` shell's constraint norms one cell
  farther out with it (the shells take the problem's masks too).

**What step X8 measured, and the recommendation (amended in step X8).** The
tables are under
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#the-polar-corner-step-x8),
"The polar corner (step X8)". The tunnel to Symmetry went down two hours into
the step, with the lottery scan at `h = 1/24`, `1/32` and `1/48` and the two
H200 rows running there; the first five items are what was read back before
that and what ran on the development machine, and the last, **"The scan read
back"**, is the scan itself, finished that night — it qualifies the first.
- **The shave cures the polar corner.** On the small rotating octant at
  `h = 1/24` (which reproduces the H200's rows to four digits and X7's
  diagnostic to its printed digits) `r_E = 17/15` and `21/20` are stationary
  to `10 M` with the shave and blow up without it, at `4.5` and `9 M`; the
  H200 row at `17/15` (`L = 64`) was stationary to `10 M`, the last row read
  back. Every other depth read back from `0.65` to `1.25` is stationary with
  and without it, and `1.35`, `1.45` and `1.55` with it.
- **It does not hold at every depth: `r_E = 3/2` fails with it**, at
  `0.64/M` from `2 M` to a blow-up at `9 M` (about `2/M` without), 7 cells
  below the polar horizon — and at `h = 1/32` too, 9 cells deep, at `1.8/M`;
  `r_E = 1.6`, 4 cells, fails at about `5/M` at the same kind of corner. The
  growing points are corners the shave left: one pass removes the corners the
  geometry made and leaves the corners of the shaved set — points with `k⁻ = 0`
  along all three axes again, a tenth fewer — and at `3/2` three of them sit
  on the lattice sphere's cap along `x`, where the `y` axis is frame-dragged
  and the `z` axis nearly tangent (`b/a = +0.17`): X7's polar corner on its
  side.
- **The analysis says why** (`test/excision_patch.jl`, proposed in step X8:
  the package's right-hand side linearized about the analytic state and
  restricted to a box of lattice points — a frozen-coefficient spectrum of
  the discrete operator itself). The rightmost eigenvalue whose eigenvector
  lives off the box's faces is positive exactly at the growing corners, at
  the runs' rates: `+2.00/M` at `17/15` without the shave (X7 measured
  `1.9/M`), `+0.66/M` and `+1.79/M` at `3/2` with it at `h = 1/24` and `1/32`
  (the runs `0.64/M` and `1.8/M`), `+2.23/M` and
  `+0.92/M` at `3/2` and `21/20` without; and negative where the runs are
  stationary — `17/15` and `21/20` with the shave (`−0.77`, `−1.69/M`), and
  both caps at `r_E = 4/5` (`−6.8`, `−9.9/M`). **A growing lego corner is a
  local mode of the operator at a triple corner on a cap, and the shave moves
  it rather than removing it.**
- **Why not the fixed point.** A set with no triple corner is closed under the
  rule, and the closure of a lattice sphere reaches 3.9 to 17.7 cells outside
  it at X7's depths. So the cure for the corners that remain is in the
  operator, not the set **(proposed in step X8)**: at a triple corner on a cap,
  X5's extrapolated advection on its nearly tangent axes (not only where the
  shift points in — the patch spectrum, not the sign, should choose them), or
  X1's per-stencil extrapolation along the normal, or more dissipation at the
  caps (`ε_KO = 1` halved X7's rate). The patch spectrum tests each in
  minutes.
- **The recommendation (proposed in step X8): stop here for Erik**, as the
  brief asks where the shave does not hold. For production nothing changes:
  `r_E = 4M/5` with the shave on — the default — is stationary at `1/24`, with
  no triple corner on a cap and its caps' spectrum far in the left half-plane,
  and the shave cures `17/15` and `21/20`; a surface shallower than about
  `1.3 M` at `a = 3/5` is not safe, with the shave or without it, until the
  corner's operator is fixed **(amended below: `1.0 M`, by the scan read
  back)**. `a = 9/10` on the tracked offset surface, which
  X7 proposed next, would put the surface where this failure lives — shallow
  (normal outflow ends `0.63 M` below the horizon, with a margin of at most
  `+0.08`) and under stronger frame dragging — so it should wait for the
  corner's operator, and its `m` scan should take the patch spectrum at each
  surface's caps before any H200 hour. **Two checks stand in its way as built
  (found in step X8,** building the `a = 9/10` tracked hole on a small octant
  at `h = 1/48`**):** the driver's `excision_horizon_margin` compares the found
  horizon's *least* radius with the surface's *largest* (`r_out − offset`),
  exact for a sphere and `−6.2` cells at `m = 6` on the oblate `a = 9/10`
  horizon, which ends the run at `t = 0`; and the singular-set check compares
  the core surface's least radius, at the pole, with the ring's, in the
  equator, which refuses `m + n_L ≳ 26` at `1/48` (the recommended settings'
  `:fitted` note met it too). Both need the direction: the margin per
  collocation direction of the found surface against the shape's radius
  there, the ring against the core surface on the equator. The build itself
  is fine — at `m = 6` it shaves 1614 corners and finds 1783 frame-dragged
  axes, twice `a = 3/5`'s at the same `h` (870, X7).
- **The scan read back (amended the same night, the tunnel back): the shave
  relabels the lottery; it does not end it.** 464 rows at `h = 1/24`, `1/32`
  (on `amddebugq`) and `1/48` (on an H200), with and without the shave, sorted
  by the polar cap's top disk `R² − Z²`:
  - at `1/24` the shave cures the two classes that fail without it (`R² − Z²
    ∈ (10, 13)`, X7's, from `r_E ≈ 1.05`; `(5, 8)` from `1.13`) and makes the
    class `(9, 10)` fail at the same radii and the same times: one pass excises
    `(3, 1, Z) h` from a `(9, 10)` cap, which is then the unshaved `(10, 13)`
    cap with X7's rim corner `(3, 2, Z) h` — its patch spectrum is
    `+2.0021 ± 0.8014i`, X7's corner's to every digit;
  - at `1/32` and `1/48` the shave moves the onset deeper — the class
    `(10, 13)` fails from `r_E ≈ 0.97` and `0.85` without it, from `1.16` and
    `1.06` with it — but `17/15` fails with it at `1/32` and `21/20` at `1/48`,
    each at a rim corner of its shaved polar cap that the patch spectrum finds
    (`+1.87/M` with `b/a = +0.017`, `+1.44/M` with `b/a = +0.008` along `y`), and
    from `r_E ≈ 1.25` every `(10, 13)` row at `1/32` and `1/48` blows up within
    one to three `M`, shaved or not: the corner's mode grows faster with
    resolution; at `1/24` every class fails from about `1.2` in one column or
    the other, the shave only reordering them;
  - **with the shave every scanned configuration from `r_E = 0.65` to `1.00`
    holds at all three spacings, without it to `0.82`**; `r_E = 4/5` holds
    everywhere with and without; the H200 row at `17/15`, `1/24`, is stationary
    to `24 M` (X7's blew up at `4.5 M`), and `4/5` at `1/32` is X7's production
    row outside the horizon to `3 %` in every shell, the drift of `J` to
    `0.3 %`, with the band's ℋ `0.78×`.

  So the recommendation stands, with its number corrected **(proposed in
  step X8)**: keep the shave on — it widens the safe window by `0.18 M` and
  costs nothing outside the horizon — and production at `r_E = 4/5`; treat
  `r_E > 1.0` at `a = 3/5` (less than `0.8 M` below the polar horizon) as
  unsafe at any `h` until a cap's triple corner has an operator the patch
  spectrum finds stable, which is Erik's call.

**Scalar on the CPU (proposed in the main merge, 2026-10-08).** `main`'s SIMD
lanes evaluate `W` points at a time on the CPU; an `:excised` problem does not take
them: its main kernel gives every point the scalar path, as on a device, and the
zone and frame-dragged kernels are scalar launches as before. Lanes for excision
are future work. Under [One right-hand-side
evaluation](#one-right-hand-side-evaluation), "Excision runs scalar on the CPU",
with what it costs.

## Initial data and backgrounds

**Analytic, from `SpacetimeMetrics`** (decided): `Minkowski`,
`GaugeWave(A, d)`, `ShiftedMinkowski(A, w)`, `KerrSchild(M, a)`,
`Harmonic(M, a)` (Kerr in fully harmonic coordinates, horizon
penetrating), and `translate`, `rotate`, `boost` of any of them. Every
case is a **background plus parameters**: the same object supplies the
initial data, the Dirichlet data, the interior's `u_exact`, the gauge
source where there is one, the error reference, and the hole's analytic
trajectory.

| case | background | `H` | static | what it measures |
|---|---|---|---|---|
| Minkowski + noise | `Minkowski()` | 0 | yes | robust stability; dissipation |
| gauge wave | `GaugeWave(A, d)` | 0 | no | the scheme's order; the interface-order rule |
| shifted Minkowski | `ShiftedMinkowski(A, w)` | sampled | yes | a nonzero shift; the gauge-source path; the Dirichlet hook |
| Kerr-Schild | `KerrSchild(M, a)` | sampled | yes | a hole with a non-harmonic gauge source |
| harmonic Kerr | `Harmonic(M, a)` | 0 | yes | a hole in harmonic gauge; spin |
| **boosted harmonic Kerr** | `boost(Harmonic(M, a), v)` | 0 | no | **the proof-of-concept case**: a spinning hole crossing the mesh |

Each case is also its box and its periodicity, and the parameters that are
properties of the physics rather than of the mesh (`ε_KO`, `γ0`, `γ2`);
that is `GHCase`, which lives here with the backgrounds it is made of
rather than in `driver.jl` **(amended in step 3**: the right-hand side
needs a case two steps before there is a driver, and a struct cannot be
defined twice**)**. Its constructor is where a moving non-harmonic
background is refused.

A background gives the initial state through one callback,
`x ↦ (h_ab, Π_ab)` at time `t`, with `Π` from `∂_t g` (the `dmetric`
forward-mode pass, which returns `∂_c g_ab` — note the index order
differs from GHSO2's `dg[a, b, c] = ∂_a g_bc`) and the *analytic*
spatial gradients, and with the core rule above applied inside a hole.
This is the `AllVariables` form TreeAMR's `fill_by_coordinates!` and
`adapt_to_initial_data!` take, so the initial-data cycle re-evaluates the
data on every mesh it produces, as TreeAMR prescribes. GHSO2 instead
builds `Π` from the *discrete* gradients so that a static solution has
`∂_t g = 0` to roundoff; that is kept as a **post-pass option** after the
cycle converges, and G4 measures whether it changes the stationarity
error of a hole visibly **(predicted: not beyond the first chunk)**.

**(Measured in step 5**, `discrete_gradient_momentum!`.**)** The
prediction is right, and the reason is worth stating because it is not
that the post-pass fails. Inverting `∂_t g = β^i ∂_i g + (α/√γ)Π` for
`Π` with the *scheme's own* `D_i` makes the **first** evolution
equation's residual roundoff on a static background: on Kerr-Schild
`a = 0` at `q = 2` the worst `|∂_t h|` over the evolved region falls from
**1.25e−2** to **2.2e−16** with `ε_KO = 0`, which is exactly GHSO2's
claim and is an identity, not a coincidence — the post-pass subtracts the
same operator the kernel adds back. The **second** equation is untouched:
`|∂_tΠ|` is `1.50e−1` before and `9.14e−2` after, a change of the same
order as itself, and it is `∂_tΠ` that dominates. With the dissipation on
(`ε_KO = 1/2`) even the first equation keeps a residual, because the
Kreiss–Oliger term is `O(h^{q+1})` and is not part of the inversion:
`2.10e−2` against `8.47e−3`. So the post-pass buys the momentum
constraint's own equation exactly and the run nothing measurable, which
is what "not beyond the first chunk" meant.

**The hole cases** (added in step 5) are `kerr_schild_case` and
`harmonic_kerr_case` over a shared `hole_case`: a Dirichlet box, the
layer's two radii and the variant, GHSO2's recipe (`ε_KO = 1/2`,
`γ0 ≈ 1/M`) as **their** defaults and nobody else's, and the chunk. The
resolution a hole needs follows from the two radius requirements
together, `r_h,min ≥ (m + 2G + 2)·h + r_0`, and therefore from the
horizon's *smallest coordinate radius* — which is `2 M` for Kerr-Schild
at `a = 0`, `M` for the harmonic chart at `a = 0` and `0.44 M` at
`a = 9/10`. Kerr-Schild is thus the cheapest hole to put on a mesh by a
factor of eight in points, which is why it and not the harmonic chart is
the one the suite runs; it is also the one with a sampled gauge source,
so the cheap case is the one that exercises the `Hsrc` path under a hole.
`r_0` is where `|h| ≲ 10`: **`0.2 M`** for Kerr-Schild `a = 0`,
**`0.4 M`** for harmonic `a = 0`, and — the surprise — anywhere at all
for harmonic `a = 9/10`, whose `|h|` is between 10 and 12 from `r = 2 M`
all the way in to `0.05 M`, the chart being singular on the ring and not
at the origin. The spinning hole is the gentle one in amplitude and the
expensive one in resolution.

**The background is evaluated inside kernels** (decided, and a
dependency risk). The interior's `u_exact` is needed at every RHS
evaluation and the Dirichlet hook at every ghost fill; both must be
kernel arguments, so `SpacetimeMetrics` — StaticArrays plus ForwardDiff
duals, both `isbits` — has to compile as one. On the CPU backend that is
certain; on CUDA it is expected and untested, and G0 tests it on the CPU
while G6 tests it on the H200. **(Measured in step 0** on the CPU
backend: `KerrSchild(1, 0)` and `boost(Harmonic(1, 9/10), 0.3 x̂)` both
compile into `fill_by_coordinates!`'s `AllVariables` kernel and fill a
20-variable, `G = 3`, vertex-centered field set over a two-level forest
with **bit-for-bit** the values of a host loop through `coordinates`, at
`Float64` and at `Float32`; both backgrounds, and `boost`'s wrapper
around one, are `isbits`.**)** If the device refuses, the fix is a
kernel-safe evaluation path *in `SpacetimeMetrics`*, not a redesign
here: nothing in this package's structure depends on where the metric
is evaluated, only its cost does.

## Refinement and regridding

**An error indicator drives the refinement** (decided in review,
replacing prescribed spheres). It is TreeWave's machinery ported: a
per-cell **Löhner indicator** on the state, reduced to a per-block
verdict through TreeAMR's `firing_boxes`, with two thresholds and a
travelling margin. Whether that is straightforward was the question
asked in review; the answer is yes for the mechanics, which exist and
are measured in TreeWave, plus three additions that are each a pure
function of position, and one protocol change for the convergence
studies. In order:

**The indicator.** Löhner's second-difference ratio per cell,

    τ = max over fields of  |Δ²u| / ( |Δ⁺u| + |Δ⁻u| + ε_g · U_ref ),

with the classic *local* floor `ε(|u₊| + 2|u₀| + |u₋|)` replaced by a
**global** one, `ε_g · U_ref` with `U_ref` a per-field reference
amplitude — TreeWave's lesson, recorded in its `CODE.md`: fields that
cross zero or decay to `1e−16` tails score `τ ≈ 1` against a floor that
shrinks with them, and the whole domain refines. Here the fields are the
10 components of `h` (**proposed**; `Π` is a switch), which fall off as
`M/r` around a hole, and `U_ref` is the largest `|h|` in the *evolved*
region at `t = 0`.

**`U_ref` is one amplitude for all ten components, not one per component
(proposed in step 6)**, and `Π` is left off. Per-component is the trap in
this package's own charts rather than a refinement of the rule: in the
gauge wave's chart six of the ten components are **identically zero**, so a
per-component reference is exactly zero, the floor with it, and that
component's numerical dust scores `τ ≈ 1` over the whole domain — TreeWave's
blast-wave failure (`∂ₜu ≡ 0` at `t = 0`) met in a chart instead of in
initial data. The ten are components of one tensor in one chart, not
independent fields, so they share the amplitude they actually have. The
per-component maxima are still measured, because they are what the
calibration table reports. `ε_g = 4ε` with `ε = 1/100`, which is TreeWave's
spelling of the same floor. For a `1/r` field the ratio itself decreases like
`h/r`, so the indicator produces nested shells of refinement around the
hole without being told to — coarser with distance, exactly the
hierarchy the prescribed spheres would have been. The stencil reaches one
point past the block face, so ghosts are filled before flagging (the
driver does it, at the chunk boundary, with the current hook).

**Two thresholds, two meanings, four marks** (TreeWave's, verbatim):
`refine_tol` means "under-resolved here", `coarsen_tol` means "something
is here at all", their gap is the hysteresis; a block below the level
cap with `τ_max > refine_tol` reports `(Refine, box)`, a block at the cap
with `τ_max > coarsen_tol` reports `(Keep, box)` — the equal-level margin
that travels with the hole — a block with `τ_max < coarsen_tol` reports
bare `Coarsen`, otherwise bare `Keep`. The box is keyed on `coarsen_tol`
(TreeWave: keying it on `refine_tol` silently disables the travelling
margin). The buffer width is TreeAMR's `buffer` from the hole's speed
and the chunk, `|v| · chunk` in cells at the finest level, plus one.
Thresholds are calibrated as TreeWave calibrates them: `τ_max` on
uniform meshes at successive `h` on the static hole, tabulated,
thresholds chosen mid-plateau (G4).

**Three additions the black hole and the boundary need**, all pure
functions of position, all cheap, all block-level:

1. **The indicator is masked inside the interior.** For `r < r_1` the
   damping layer and the frozen core are not a numerical solution — the
   core holds stale data with steep, meaningless differences — so `τ` is
   set to zero there, and blocks entirely inside the interior are free
   to coarsen as far as 2:1 balance lets them. The layer is resolved
   from outside: the evolved points just beyond `r_1` sit in the
   steepest part of the metric and are refined by the indicator, and it
   is the resolution of the blocks *containing `r_1`* that the layer's
   thickness requirement refers to.
2. **A level floor around the horizon.** The interior's two radius
   requirements are stated in grid points at the resolution of the
   blocks around `r_1`, and the indicator alone does not *guarantee*
   any resolution — it produces one. So a block whose extent intersects
   the shell `r_1 ≤ r ≤ r_h,max + a few M` is refined to at least a
   floor level `L_h`, chosen from `r_h,min` and `m` so that the
   requirements hold by construction; the mark is
   `max(indicator's request, floor)`. The assertions stay, and with the
   floor in place they are a check on the arithmetic, not on the
   indicator's mood. **(predicted)** the indicator asks for the floor
   level or more everywhere the floor applies, so the floor never binds
   on a calibrated run; the test that it *can* bind uses a deliberately
   loose `refine_tol`. **(Confirmed in step 6**, on the reference
   configuration below: the marks with the floor and the marks without it
   are identical at the calibrated thresholds, and with `refine_tol` raised
   to `99/100` — above anything the data reaches — the floor is the only
   thing that refines, on exactly the blocks whose extent meets the shell.**)**
   **`L_h` is derived rather than stated (proposed in step 6)**: it is the
   coarsest level whose spacing satisfies *both* requirements,
   `h ≤ min((r_h,min − r_1)/m, (r_1 − r_0)/2(G+1))`, so a floor a caller
   had to compute by hand cannot go stale when the box, the margin or the
   radii move. A `maxlevel_cap` below it is refused by name, since the mesh
   would then settle where `check_interior_radii` fires.
3. **A level ceiling near the outer boundary.** Two things happen at
   the outer boundary that an indicator misreads: the Dirichlet data
   meets the numerical solution with a truncation-order mismatch, a kink
   whose second difference the indicator scores; and the `1/r` tails
   are where a scale-free floor would refine forever. The global floor
   handles the second; for the first, blocks within a few coarse cells
   of the domain boundary are capped at the coarsest level, again as
   `min(indicator's request, ceiling)`. The ceiling also keeps
   coarse-fine faces off the outer boundary, which is a legal but
   pointless configuration to exercise.

   **Two things the ceiling turned out to need (step 6).** First, the
   **travelling margin has to be dilated before the bounds are applied, not
   after**: TreeAMR's `buffered_flags` promotes every leaf a dilated box
   reaches, whatever this package said about it, so on any root brick small
   enough that the boundary blocks are the refined region's neighbours the
   margin refines them anyway — measured, 12 of 64 boundary root blocks.
   So `refine_flags` calls `buffered_flags` itself, applies the floor and
   the ceiling to its result, and the driver passes `buffer = 0` to
   `regrid!` **(proposed in step 6)**. Second, **the ceiling and the floor
   together are a statement about the box**: a block that is both in the
   horizon shell and within the boundary margin is refused by name, because
   the region that must be resolved and the region that must stay coarse
   have met. That is what a box of half-width `5/2 M` around a hole whose
   horizon is at `r = 2 M` does, which is why the refinement's reference
   configuration below uses `5 M` where step 5's fixture used `5/2`. What
   the ceiling does *not* override is 2:1 balance: a deep enough hierarchy
   in a small enough box pushes refinement outward through the balance
   condition, and that is TreeAMR's invariant rather than this package's
   choice.

**Convergence studies on a frozen hierarchy.** An error-driven mesh
changes with the resolution, which muddles a convergence study. The
claims of order `q` in G4 and G5 are therefore made on the hierarchy the
indicator chose at `t = 0` for the coarsest run, *held fixed in space*
and refined uniformly by doubling `N` — the block layout is unchanged
and every spacing halves — with regridding off; the adaptive runs are
then compared against the uniform-fine reference in TreeWave's manner
(the tracked pulse against `uniform_pulse`). Both are needed: the frozen
hierarchy measures the scheme, the adaptive run measures the indicator.

**(Implemented in step 5**, as `hole_forest` in `initialdata.jl`.**)**
Before the indicator exists there is nothing for it to have chosen, so
the frozen hierarchy is built by hand: one shell radius per refinement
level, each level refining the blocks of the level below whose extent
reaches within that radius of the hole's center. Which blocks are
refined therefore depends on the radii, the root brick and the number of
levels **and on nothing else** — so raising `N` halves every spacing and
leaves the layout exactly where it was, which is the whole content of the
protocol above. Two things it has to get right and one it must not
pretend to be: the shells must *nest* (a shell wider than its parent's
would ask for a fine block outside the coarser refined region, and 2:1
balance would answer by refining everything between them, which the
constructor refuses); the innermost shell must contain the sphere `r_1`
whole, or the coarsest blocks touching it are a level up and it is
*their* spacing the two radius requirements are stated at; and it is not
the refinement mechanism, which is step 6's.

**What follows for the hole.** The indicator follows the moving hole on
its own, and its center of refinement is a measurement to be compared
with the analytic center — a disagreement of more than a few finest
spacings is a bug in the indicator or the thresholds. The interior layer
still uses the analytic center, because it must. The initial-data cycle
converges to a fixed hierarchy on a static hole (the indicator is
time-independent up to truncation noise), so regrids during a static run
should change nothing **(predicted)**; on the moving hole they move the
shells with it, one block ring per chunk at the finest level.

**The driver** is TreeHydro's one loop with cases as data (decided):

    adapt_to_initial_data!(U, ops; initial = state(case, t0), flags = indicator_flags,
                           buffer, boundary = dirichlet(case, t0))
    sample_gauge_source!(Hsrc, case)                    # non-harmonic backgrounds only
    p = GHProblem(U, Hsrc, ops, case; q, ε_KO, γ0, γ2, interior, …)
    while t < t_end
        λ = max_speed(p); dt = cfl · minimum_spacing(forest) / λ
        gh_solve(p, u, (t, stop); dt, alias_u0 = true)  # IMEXRungeKutta's RK4 by owner
        assert the CFL bound held; record the analysis quantities
        observer(p, t, u)                               # before the regrid
        scatter!(U, u); fill_ghosts!(U, p.schedule; boundary = dirichlet(case, t))
        if regrid!(forest, (U => p.schedule, Hsrc => nothing, diag => nothing);
                   flags = indicator_flags(U, case, t), buffer, boundary = dirichlet(case, t))
            sample_gauge_source!(Hsrc, case)
            p = GHProblem(…)                            # new schedule, origins, spacings, ρ_max
            u = statevector(U); gather!(u, U)
        end
    end

with `indicator_flags` the masked Löhner verdict with floor and
ceiling. Every chunk is a fresh solve, for TreeAMR's reason (the state
vector changes length and meaning). What the loop returns is what the
tests assert: the analysis record per chunk, and through `observer`
whatever the viewer wants.

**(Implemented in step 5**, `src/driver.jl`, without the regrid.**)** The
loop exists and runs the static hole; `evolve!` takes `regrid = false`
and **refuses `true` by name** until step 6 supplies the indicator, for
the reason the refusal says: this section's loop flags with the masked
Löhner verdict and with nothing else, and a driver that regridded on some
other criterion now would be a second refinement mechanism to delete
later. For the same reason `evolve!` takes the **forest** as a keyword
rather than building one: step 5's mesh is the frozen hierarchy of
`hole_forest`, and step 6's is what `adapt_to_initial_data!` produces
from the indicator. Three things the writing settled:

- **`ρ_max` is what makes a chunk a restart even without a regrid.** It
  was `1/dt` and `dt` is measured per chunk, so the interior the kernel
  closes over is rebuilt at the top of every chunk (`with_interior`),
  which shares the field sets, the schedule and the **sampled gauge
  source** rather than rebuilding the problem — re-sampling `H_a` is the
  most expensive setup phase there is and nothing about a new `ρ_max`
  invalidates it. **(Amended in step 8c′:** the default is now `4/M`, the
  same in every chunk, and the rebuild stays — it is what the grid-rate
  option `ρ_max_factor` needs, it is where the default is checked against
  that chunk's `1/dt`, and it costs nothing.**)**
- **The record's first row is `t = 0`**, before anything has been
  integrated, so that "the error grew from zero" is a statement a test
  can check rather than assume; `nchunks` is the number of rows after it.
- **The CFL recheck is `check_cfl`, and it throws** (`CODE.md`, "The time
  step": *throw, do not warn*). It is a detector and not a guard — the
  chunk has already been integrated — and the remedy is a shorter chunk
  or a smaller `cfl`. On the static hole `λ_max` is `1.671` for
  Kerr-Schild `a = 0` and does not move between chunks, so the recheck
  has never fired; it exists for G5, where the hole crosses the mesh.

**(Implemented in step 6**, `src/refinement.jl` and the loop's two
branches.**)** `evolve!` now takes `adapt` and `regrid`, both of which
refuse a case with no `Refinement` by name; the parameters are one
`isbits` struct on the case (`refine_tol`, `coarsen_tol`, `maxlevel_cap`,
the floor's margin beyond `r_h,max`, the ceiling's margin in coarse cells
and its level, and `ε`) rather than five fields, so that a case that is run
on somebody else's mesh carries one `nothing`. Five things the writing
settled, each stated where it happens:

- **`τ` is materialised into `diag`, and the marks are taken from that
  field (amended in step 6).** TreeWave and TreeHydro evaluate their
  per-cell indicator inside `firing_boxes`' predicate and keep nothing;
  this package's analysis record asks for `τ_max` among the mesh
  statistics, and a firing *count* cannot give it. So one kernel writes `τ`
  into `DIAG_TAU` — the fourteenth `diag` slot, appended — and the two box
  sweeps read it back. That is one evaluation of the criterion instead of
  three, and it makes the number the record holds and the number the mesh
  was chosen by the same number by construction.
- **The ghost fill before flagging is the monitors' own preamble.**
  `gh_indicator!` scatters and fills with *this* `t`'s Dirichlet hook, as
  the constraint monitors do, because `regrid!` fills ghosts only after the
  flags exist and a stale ghost corrupts the verdict silently.
- **`regrid!` is handed the state field set alone (amended in step 6).**
  The loop above lists `Hsrc => nothing, diag => nothing`; both are rebuilt
  by the fresh `GHProblem` — which is also where the gauge source is
  re-sampled and where `check_interior_radii` re-asserts the interior's two
  radius requirements on the new mesh — so resizing them through the
  transfer would be work thrown away.
- **The regrid is skipped after the last chunk (amended in step 6)**, so
  that the forest that comes back and the state that comes back describe
  the same mesh. TreeHydro's rule, for TreeHydro's reason.
- **The record gains `τ_max`, the refinement centroid and its distance from
  the analytic center**, which is the mesh-statistics row of the analysis
  table, and the run's `passes`, `converged`, `nregrids` and the buffer
  width it used.

**(Amended in step 8d:** the floor reads the horizon the layer actually
follows.**)** `horizon_floor_level` and `level_bounds` take the layer's
radii through two accessors, `geometry_radii(int, background)` — the
horizon's smallest and largest radius, the analytic ones for step 5's
sphere and the tracked shape's `r_in`, `r_out` for a tracked geometry — and
`layer_radii(int)`, so the sphere's floor is value for value what it was and
a tracked one's shell runs from its offset surface's smallest radius to its
horizon's largest plus `floor_margin`. The tracked geometry's **level** is
the level of the spacing its offset and ramp were built at: they are stated
in spacings, `offset = m h`, so `offset/m` *is* the `h` the sphere's formula
derives, and the floor keeps the layer's blocks from coarsening past it —
which is what would make `check_interior_radii` fire at the next chunk —
without asking for a finer one; the next geometry is rebuilt on whatever the
indicator chose **(proposed in step 8d)**. The indicator takes the
geometry the run holds (`indicator_flags(…; interior)`), and the
initial-data cycle rebuilds a tracked geometry on every pass's mesh and
re-evaluates the data with the final one's core rule. The suite exercises
none of this on a mesh that moves — its tracked runs are on the step-5
fixture's frozen hierarchy — and step 8f's matrix is where a tracked
geometry is first regridded.

**(Amended in step 8:** the mesh follows the hole.**)** Four pieces, each
**(proposed in step 8)**:

- **A `:fitted` case's initial-data cycle flags on its own data**
  (`adapt_fitted_initial_data!`): on every pass the geometry is rebuilt on
  the current mesh, the case's initial data are filled on it
  (`fill_fitted_initial!` — the analytic solution outside the offset
  surface, the fit of it inside), the ghosts filled with the `t = 0` hook,
  the masked indicator flags with the geometry's mask and floor, and
  `regrid!` rebuilds without transferring — TreeAMR's
  `adapt_to_initial_data!` written out, because a `:fitted` case's data live
  partly in a cache a coordinate callback cannot read. Step 8f's cycle
  flagged on the analytic `:damped` layer and refused G5's chart, where the
  analytic core cuts the disk; this one converges there (2 passes to 456
  blocks at `5/128` in the suite, 2 passes to 3536–4040 blocks at `5/256`
  in the `moving` rows), and on a chart with analytic data it chooses the
  analytic cycle's mesh exactly (the adaptive fixture: 288 blocks, one pass
  each, `fa.leaves == fb.leaves`). The indicator is masked inside the
  offset surface, so only the one point of Löhner's stencil that reaches
  inside reads the fit. A hand-over case still cycles on its analytic
  layer, which is its initial data.
- **The tracked floor starts at the core surface, not the offset surface.**
  A tracked geometry reads its spacing `h` as the coarsest of every block
  the layer lives in, down to the core surface, and states its offset and
  ramp in it; a floor that started at the offset surface left the inner
  blocks free to coarsen (the indicator is masked there, so it asks to),
  the next geometry would then double its spacing and its floor ask for a
  level less — a mesh that loses a level per regrid around a moving hole.
  Step 5's sphere keeps its floor from `r_1`.
- **The floor is widened by the travel `|v| · chunk`**, inward and outward
  (`level_bounds(…; travel)`): it is evaluated at the regrid's `t` and must
  hold the blocks the layer reaches before the next one. The travelling
  margin `buffer` was already `|v| · chunk` in cells (step 6): 5 cells at
  `5/256` and 3 at `5/128` for `v = 0.3`, chunk `M/4`. Both are zero-width
  for a static hole, whose bounds are unchanged.
- **The regrid flags the evolved state** at every chunk boundary with the
  geometry the next chunk runs on — nothing new: `gh_indicator!` already
  read the state and the layer's mask. The first regrids of a tracked
  geometry along a trajectory are step 8's `moving` rows.

**What it measures on the boosted `a = 0` hole (measured in step 8**,
`hole_runs.jl moving=ctl`, harmonic `a = 0` boosted at `0.3` from `x = 2`
to `x = −1.9` in `13 M`, box `5 M` on a `4³` root brick, finest `5/128`,
`chunk = M/4`**)**: the mesh regrids at 40 of the 52 chunk boundaries,
between 1128 and 1912 blocks, every radius assertion holding at every one;
the refinement centroid stays within **`1.4` finest spacings** of the
analytic center over the crossing on the analytic `:damped` layer (`20/M`)
and within **`7.3`** (typically 4–6) on the `:fitted` one, whose layer
edge leaves more grid-scale content on the trailing side for the indicator
to score; the static configuration's centroid is `0.29` spacings off at
`t = 0`. The travelling hole is tracked within `0.028` cells.

## Time integration

**IMEXRungeKutta's classical RK4, its stage arithmetic by block owner**
(amended 2026-09-26, at Erik's direction, as TreeAMR's own examples now
use it; OrdinaryDiffEq until then, whose decision and measurements follow
as history). `src/stepping.jl` is the whole coupling: `state_partition`
turns TreeAMR's `threadchunks(nblocks)` into IMEXRungeKutta's `partition`,
so that each block's entries are combined on the thread `map_blocks!` runs
the block on (`nothing` on a device: the fused broadcast); `gh_integrator`
builds `init(IMEXProblem(gh_rhs!, nothing, u, tspan, p), RK4(); dt,
stage_limiter = gh_limiter!, step_limiter = gh_limiter!, partition)`, and `gh_solve` runs one to its end. The driver builds **one
integrator per chunk**, stepping the run's own state vector in place
(`alias_u0 = true`); a moving hole's refilled target is swapped into it
between pieces through a `ProblemRef` (`swappable = true`) instead of a new
`init`, and each chunk's integrator takes over the previous chunk's scratch
while the mesh is unchanged (`reuse`, IMEXRungeKutta 1.2, from 2026-09-26),
so the scratch is allocated once per mesh and not once per chunk. What
changed with it:

- **The limiters: one, on every state vector** (decided 2026-09-26 by
  Erik). IMEXRungeKutta calls its stage limiter only on the three stage
  values of a step that `f_exp!` reads and its step limiter once on the
  result, so `gh_limiter!` — the projection, then the paste — is passed as
  both, IMEXRungeKutta's own rule for a correction that must reach every
  right-hand-side input. Under OrdinaryDiffEq only the projection reached
  the stage values, the paste was a step limiter alone, and the FSAL
  tendency was evaluated on the result before the paste. On a static hole
  pasting a stage value changes no bit (inside `r_1` the `:pasted`
  right-hand side is zero and the analytic solution does not depend on the
  stage's time); on a moving one the ball is the analytic solution at every
  stage's time.
- **The result agrees to roundoff, not bitwise.** The stages are summed in
  a different order: the gauge wave's `L2` error at `q = 4`, `N = 16` is
  `2.2860177409713e−5` against OrdinaryDiffEq's `…409697e−5`, and the
  suite's hole fixture agrees to twelve digits at `t = 1/5`; a moving
  `:fitted` hole (`hole_runs.jl moving=l0-base t_end=1/4`, seven refills
  swapped into its chunks' integrators) prints the same row as `main` to
  every digit. By owner it is
  bitwise its own broadcast and bitwise across thread counts, which
  `test/thread_workload.jl` now digests (its hand-written RK4 is gone).
- **One right-hand side fewer per `solve`** (no FSAL start) and no copy of
  the state: OrdinaryDiffEq allocated about fifteen state-sized vectors per
  `solve`, serially; IMEXRungeKutta allocates four scratch vectors per
  `init`, first-touched through the partition — still 0.13–0.36 s at 64
  threads on 320 MB (first touch dearer than interleaved pages), which is
  why a chunk has one integrator and not one per piece. **(Amended
  2026-09-26.)** With IMEXRungeKutta 1.2's `reuse` the next chunk's `init`
  takes those four arrays over and allocates nothing: 0.05 ms against 13–35
  ms at four threads on the development machine (40–75 MB states), so the
  cost is paid once after the initial data and once after every regrid that
  moves the mesh.

**(Measured 2026-09-26 on Symmetry** with `bench/stepping.jl`, one exclusive
64-core EPYC 7543 node, cn096, jobs 563975 and 563982; 512 blocks of `16³` at `q = 4`, a 320 MB
state; the gauge wave and the suite's Kerr-Schild fixture at `halfwidth =
5` with `hole_forest(…; roots = 2, radii = (6, 3, 3/2))`, whose finest
spacing is the fixture's; minimum ms per call, each configuration twice, in
agreement to a few percent**)**:

| per call, pinned, first touch | gauge wave | hole |
|---|---|---|
| RHS, `main` (the boxed kernel, below) | 482–530 | 755–786 |
| RHS, from 2026-09-26 | 120–136 | 252–266 |
| OrdinaryDiffEq RK4 step, `main` | 3146–3180 | 4354–4504 |
| OrdinaryDiffEq RK4 step, fixed kernel | 920–947 | 1605–1665 |
| IMEXRungeKutta RK4 step, broadcast | 770–793 | 1304–1330 |
| IMEXRungeKutta RK4 step, by owner | 553–603 | 1077–1106 |
| `solve` per 4 steps, per step: OrdinaryDiffEq on `main` (the old driver) | 3825–3924 | 5118–5160 |
| `gh_solve` per 4 steps, per step (`init` included) | 643–706 | 1157–1200 |

So at 64 threads a step of the old driver was **5.6×** (wave) and **4.3×**
(hole) today's. About 3–4× of that was the right-hand-side kernel itself
(the `Core.Box` of "Precision, threads, devices": ~570 bytes allocated per
point, 1.2 GB per evaluation, which 64 threads turn into garbage-collector
contention — the same bug cost 7–15 % at four threads on the development
machine), and 1.5–1.6× the integrator (its serial stage arithmetic and its
per-`solve` buffers). A step by owner is now `4.2–4.9` right-hand sides.
**Placement** (the four configurations on cn096): pinned (`JULIA_EXCLUSIVE =
1`, `srun --cpu-bind = none`) with first touch is the fastest for the
right-hand side (133 ms against 151 unpinned on the wave, 257 against 318 on
the hole) and for the step; interleaving (`numactl --interleave = all`)
helps only `init`'s allocation. Job 563749's anomaly — the owner-mapped
update 3–5× slower under first touch, suspected NUMA balancing — did not
reproduce: after a process's OrdinaryDiffEq rows (every one a serial pass)
the owner-mapped step was no slower than before them. **The driver end to
end** (`evolve!`, three chunks, the record included; jobs 563976 on cn112 —
an 8×8-core AVX-512 node with one NUMA domain, pinned — and 563977 on
cn079, an EPYC 7543, unpinned): per step, `main` against today, **4111 →
544 ms** and **5297 → 1140 ms** pinned, **3460 → 843** and **4547 → 1516**
unpinned, with the records equal to every printed digit.

**On a device (measured 2026-09-26**, the first device runs of this
package**)**. With the box gone the right-hand-side kernel compiles for a
GPU: on the development machine's Metal at `Float32`, and on one H200 (job
563978, cn113) at `Float64` and `Float32`, with `evolve!` end to end — the
record's monitors on the device, the broadcast stage arithmetic, `hostcopy`
only for a horizon find (none here). The `Float64` hole's record equals the
CPU run's to every printed digit (`err_l2 = 1.690514e−6`).

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
H200](#the-right-hand-side-on-an-h200-measured-2026-10-05): on the device the
kernel is not inlined and spills 8 KB a thread; inlined, with a register-lean
source, it is 7× faster.**)**

**Block size and block count (measured 2026-09-26**, `bench/stepping.jl`'s
`scan` mode, `q = 4` throughout — fourth-order centred differences, sixth-order
prolongation, `G = 3`; H200 jobs 564111 and 564114, the CPU on cn079, an EPYC
7543, pinned at 64 threads, job 564112; the hole's rows need two roots or more,
since one root at `N ≤ 12` fails the interior's radius checks**)**. The
right-hand side in nanoseconds per point:

| wave, `N` \ blocks | 8 | 64 | 512 | 1728 |
|---|---|---|---|---|
| 8, H200 / CPU | 137 / 834 | 19.7 / 119 | 12.9 / 84 | 12.3 / 99 |
| 16 | 18.9 / 468 | 10.7 / 68 | 9.9 / 71 | 9.7 / 66 |
| 24 | 13.6 / 449 | 10.7 / 62 | 10.4 / 60 | 10.4 / 59 |
| 32 | 10.1 / 433 | 9.1 / 58 | 9.0 / 57 | |

| hole, `N` \ blocks | 512 | 1112 | 2416 |
|---|---|---|---|
| 8, H200 / CPU | 33.8 / 337 | 24.5 / 301 | 19.0 / 290 |
| 12 | 21.0 / 193 | 17.9 / 171 | 16.3 / 172 |
| 16 | 13.7 / 138 | 12.4 / 128 | 11.6 / 128 |
| 24 | 12.4 / 103 | 11.9 / 96 | 11.6 / 96 |
| 32 | 10.2 / 85 | 9.9 / 83 | |

The H200 saturates at about a million points and 9–10 ns a point from
`N = 16` up; `N = 8` costs a third more there (its ghosted block is `6.6×`
its owned points) and the hole at `N = 8` twice as much. The CPU needs a block
per thread before anything else matters (8 blocks on 64 threads is
6–8× the per-point cost), and then gains from larger blocks all the way to
`N = 32` (57 ns against 71 at `N = 16` on the wave). The hole costs 1.5–3.5×
the wave's per point on the CPU and 1.1–2.7× on the H200, the ratio falling as
`N` grows — the layer's analytic target, a forward-mode dual pass per point,
is relatively cheaper on the device — so from 64 blocks up the H200's advantage
is **5.7–8× on the wave and 8–15× on the hole**. A
step is 4.0–4.1 right-hand sides on the H200 at every size and 4.2–4.5 on the
CPU from 512 blocks up (7.6 at 64 blocks of `8³`, where the stage
arithmetic's per-call overhead shows). For G5-sized meshes this says `N ≥ 16`
on either machine, and `N = 24`–`32` where the refinement's granularity
allows it.

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

**(Measured in step 3.)** The integrator is `RK4()` with
`adaptive = false`, `save_everystep = false` and `dt` from `gh_dt`, and it
is what every convergence rate above was measured through. At `cfl = 1/4`
the time error does not contaminate the spatial one even at `q = 6`,
because `dt ∝ h` makes it `O(h⁴)` with a factor `cfl⁴ ≈ 0.004` in front
and the runs are short: the measured rate is 5.92, not 4. That headroom is
a property of the *time* the studies run for — an eighth of a crossing —
and a longer run at `q = 6` would need a smaller `cfl` or a higher-order
tableau to keep it, which is worth knowing before G3 lengthens anything.

The relaxation rate of the interior layer is bounded by RK4's stability
on the negative real axis, and both rates the driver runs keep it well
inside — the default `4/M` is `0.04/dt` to `0.09/dt` on the suite's holes,
and the grid-rate option `ρ_max · dt = 1` is a factor `2.8` inside (amended
in step 8c′);
the `:pasted` variant uses RK4's `step_limiter!(u, integrator, p, t)`
hook, and step 8b's range projection its `stage_limiter!` — the two places
the state may be written outside the RHS, both passed as `solve` keywords
**(amended in step 8b**; see [The
interior](#the-interior-a-pointwise-damping-layer)**)**. Adaptive
stepping is not used: the step is set by the CFL bound and
`volume_weighted_norm` is not wired in as an `internalnorm` (TreeAMR's
open question, not this package's).

## Analysis quantities

A run is judged by what it records, not by finishing. The driver
computes the following at every chunk boundary (the horizon quantities
every `k`-th chunk, `k` a case parameter), appends them to the run's
time series (see [I/O and viewers](#io-and-viewers)), and returns them;
the tests assert on them.

| quantity | how | cadence |
|---|---|---|
| GH constraint `C_a = Γ_a + H_a` | state and first derivatives, one kernel into `diag`; masked L2 and L∞ norms per component | every chunk |
| ADM Hamiltonian `ℋ` and momentum `ℳ_i` | second derivatives from the RHS's compact stencils, `∂_tt g` from the reduced equation (GHSO2's construction, consistent with the discrete dynamics); masked norms | every `k`-th chunk, about one RHS in cost |
| horizon location | the finder's recentred `origin`, and the coordinate radii `r_min`, `r_mean`, `r_max` of the surface points | every `k`-th chunk |
| horizon shape | the spin-0 coefficients `hlm` on the finder's grid, for the viewer and as the next find's seed | with the location |
| horizon area `A` | `ApparentHorizonFinder.horizon_area`, the proper area of the found surface | with the location |
| irreducible mass | `M_irr = √(A/16π)` | with the area |
| spin `J` and its axis | `KorzynskiSpin.horizon_spin` on the finder's collocation grid, from the interpolated `γ_ij` and `K_ij` | with the area |
| Christodoulou mass | `M_ch = √(M_irr² + J²/(4 M_irr²))` | with the spin |
| error against the analytic solution | `|u − u_exact|` per component into `diag`; masked volume-weighted L2 and L∞ | every chunk |
| interior residual | `|u − u_exact|` inside the layer, L∞, the layer's own health — for `:fitted`, `|u − u_fit|`, the distance from its target (amended in step 8e) | every chunk |
| mesh statistics | leaf count per level, finest spacing, the indicator's `τ_max`, the refinement centroid against the analytic center | every chunk |
| range projection | `bounds_hits` (points moved, summed over every stage-limiter call of the chunk), `bounds_nonfinite` (of those, points with a non-finite component), `bounds_r_max` (the outermost radius it fired at, `−1` where it did not); `nothing` for a case without bounds (added in step 8b) | every chunk |
| validity monitor | over the layer `r_0 ≤ r < r_1` and over the `G` points outside it: `min_detγ`, `min_α` (the *signed* lapse, negative where `g^{tt} > 0`), `max_h`, `max_Π` (the largest component magnitudes) — `min_detγ_layer` … `max_Π_shell` (added in step 8b); and over the whole evolved region, `min_detγ_evolved`, `min_α_evolved` — the lapse-collapse trigger's input (added in step 8d). On the tracked geometry the two bands are its own (`layer_mask`, `shell_mask`) | every chunk |
| horizon track | for a case whose interior is a `FittedSpec` (added in step 8d; `nothing` otherwise): `track_source` (`:found`, `:coasting`), `track_center` and `track_velocity` (the tracked trajectory after this row's find), `track_r_min`, `track_r_max` (the found surface's radii about its own origin), **`track_offset`** (the tracked center's distance from the analytic one, in cells of the finest spacing — the number "good to about a cell" is read from), `track_misses`, `track_prediction` (the found origin's distance from where the track predicted it, in cells; `nothing` without a find), `track_trigger` (this row's find was forced by the lapse trigger) | every chunk, the find every `k`-th or when triggered |
| tracked layer | the geometry the next chunk runs on: `layer_h` (the spacing its offset and ramp are stated in), `layer_offset = m h`, `layer_thickness = n_L h`, `layer_r_in`, `layer_r_out` (the shape's bounding radii), and `margin_efolds` — step 8a's leakage e-folds across the margin, the least of the six grid axes (added in step 8d) | every chunk |
| excision | for an `:excised` case (added in step X2b; `nothing` otherwise), from the state at the band's points — the evolved points whose stencils take closures: `excision_band` (their number), `excision_band_nonfinite`, **`excision_normal_min`** (the least `b_n/a_n − 1` along the surface's normal, which must stay positive), `excision_faces`, `excision_axis_min`, `excision_inflow` (step X1's faces, an excised immediate neighbour along an axis: their number, least per-axis `b/a` and the inflow-like ones `b/a < 1`), `excision_into` (band point and closure axis pairs whose shift points into the excised set, the build's refusal); a tracked case adds `excision_horizon_margin`, the found horizon's least distance from the frozen surface in cells | every chunk |
| fitted target | for a `:fitted` case (added in step 8e; `nothing` otherwise), of the fit built from this row's state: **`fit_valid`** (every point of the fit's validity sweep a Lorentzian metric — `fit_valid(fit)`; a fit that is not does not become the target), **`fit_residual`** (`fit_residual(fit).overall`, the fit's worst relative residual block by block against the data it was fitted to), and beside them the sweep's worst **`fit_min_detγ`**, `fit_min_α`, `fit_min_λ`, `fit_hits` (swept points the target's ranges would move) and `fit_refills` (the cache's mid-chunk refills in the chunk this row ends) **(proposed in step 8e)**; for `:fitted` the row `residual` is the layer's distance from its target (the cache), not from the analytic solution | every chunk |

**Constraints.** Both kernels mask the interior `r < r_1` and write zero
inside it; the modified region is not a numerical solution. **(Amended in
step X2b:** for an `:excised` hole their default mask, and the indicator's,
is `monitor_mask` — the excised set widened by the stencils' reach `W` — so
that no stencil reads an excised value.**)** Norms are
`block_mapreduce` partials weighted by each block's `h³`, combined in
block order, so they are bit-identical across thread counts. **(Amended
2026-10-01:** they are TreeAMR's `mesh_mapreduce` — the same partials, the
weight `h³` applied by TreeAMR, the combination on the host, which is the
one place TreeAMR's M7 will put the `Allreduce` — and so is every other
reduction that crosses blocks: the speed, the indicator's scales and
`τ_max`, the projection's counts and the validity monitor's extremes. They
are still bit-identical across thread counts. `mesh_mapreduce` combines
pairwise where this package's loop summed left to right, so the L2 norms
moved in the last place and nothing printed to the digits this document
records did; `max`, `min` and the counts are unchanged bit for bit. The
denominator is `evolved_volume`, one weighted reduction of the mask.**)**

**(Implemented and measured in step 4**, `src/constraints.jl`.**)** Five
things the writing settled, each stated where it is made in that file:

- **Which `∂_t g`.** The gauge constraint takes it from the first
  evolution equation, `β^i ∂_i g + (α/√γ)Π`; the ADM monitor takes
  `∂_i∂_t g` by differentiating that along `x^i` and `∂_t∂_t g` by
  differentiating it along `t`, with `∂_tΠ` from
  `gh_node_rhs_expanded` — the reduced equation with its source and its
  constraint damping, which is what "consistent with the discrete
  dynamics" means. **The Kreiss–Oliger term is left out of both**
  **(proposed in step 4)**: it is `O(h^{q+1})`, one order below what these
  monitors converge at, and carrying it would mean differencing the
  dissipation operator as well. This is the one place this package's
  `∂_t g` and the right-hand side's differ, and the difference is below
  the truncation error either of them measures.
- **The Ricci tensor is assembled, not reduced (proposed in step 4).**
  `R_ab = ∂_cΓ^c_ab −
  ∂_bΓ^c_ca + Γ^c_cd Γ^d_ab − Γ^c_bd Γ^d_ca`, written out, rather than
  through the generalized-harmonic identity `R_ab = −½g^{cd}∂_c∂_d g_ab +
  ∇_(aΓ_b) + …`. The reduced form is what the evolution equations already
  encode, so a monitor built on it would check the right-hand side against
  itself. The price is a four-dimensional contraction and about **18 s**
  of compilation for the first kernel specialisation (4 s for each
  further one) — GHSO2 warned of "minutes" and this is the measured
  figure here.
- **The chain rule needed a one-direction spelling.** `∂_t α`, `∂_t β^j`
  and `∂_t√γ` are not among what `metric_derivatives` returns (three
  spatial directions, with `√γ` folded into `A^{jk}`), so `pointwise.jl`
  gained `metric_derivatives_along`, the same five closed forms for one
  direction **(proposed in step 4)**. The three-direction function is
  untouched, and `test/constraints_tests.jl` asserts the two agree on
  every spatial direction — to roundoff, for the reason under
  [Measured results](#measured-results).
- **The masked norm divides by the evolved volume.** Each kernel writes a
  `1`/`0` indicator into a `diag` slot beside its values, and the L2 norm
  is `√(Σ_b h_b³ Σ|c|² / Σ_b h_b³ Σ 1)` — so masking a region out does not
  make the number smaller merely by diluting it with zeros
  **(proposed in step 4)**. Where nothing is masked that is TreeAMR's
  `volume_weighted_norm` exactly, which the tests assert rather than
  assume.
- **`diag` now has ten slots**: the speed, `C_a` (four), `ℋ`, `ℳ_i`
  (three) and the mask indicator, with `C_a` and `ℳ_i` contiguous because
  `block_mapreduce` reduces a contiguous range of variables and nothing
  else. **(Thirteen from step 5**: the masked error, the interior
  residual and the gauge drift, appended rather than inserted, because
  the two contiguous runs above must not move. **Fourteen from step 6**:
  the refinement indicator `τ`, appended for the same reason.**)**

**(Implemented and measured in step 5.)** The error rows are one more
kernel — the state, the analytic solution, no stencil and no ghosts, so
it costs what the speed kernel costs plus one forward-mode dual pass per
point — and three decisions:

- **The error slots hold a magnitude, not twenty components
  (proposed in step 5).** The table above says "`|u − u_exact|` per
  component into `diag`", which would be twenty more slots, more than
  tripling a field set that is `nvars × (N+1)³ × nblocks`, for a number
  the record reads as one norm. What is stored is the pointwise Euclidean
  magnitude over the twenty components, whose volume-weighted L2 *is* the
  L2 norm of the whole state error; a per-component split, if a component
  is ever in question, is a targeted kernel and not a permanent cost on
  every run.
- **The interior residual and the gauge drift are L∞ over their own
  regions**, and each is written by the same kernel into its own slot with
  its own region test: the residual over the layer `r_0 ≤ r < r_1`, the
  drift over a shell at the horizon. An L2 over a region the *mask*
  excludes would have to be divided by that region's volume, which the
  mask's count does not hold; the number those rows are read for is the
  worst point.
- **The gauge drift's shell is `[r_h,min, r_h,max + (r_1 − r_0)]`
  (proposed in step 5).** `CODE.md` asked for "the drift of `h_tt` at the
  horizon" and did not say over what set. The outer margin is the layer's
  own width rather than a number chosen for the occasion: it is the only
  length in the case that is set by the hole and known to be resolved.
  A general `ShellMask` does the same job for any norm, which is how
  the three interior variants are compared over "the `G` points outside
  `r_1`".

**(Implemented in step 8b**, `src/bounds.jl`**.)** The range projection's
rows and the validity monitor's are written at every chunk, and three
things the writing settled:

- **The hit count is per call, summed over the chunk (proposed in step
  8b).** A stage's flags are overwritten by the next stage's, so a count
  read off the `DIAG_BOUNDS` slot once per chunk would say how many points
  fired in the chunk's last stage, not in the chunk. `BoundsAccounting` —
  host-side, mutable, one per run, shared by every problem the run
  rebuilds, TreeHydro's `ResetAccounting` — adds each call's
  `block_mapreduce` of the slot; a point that fires in every stage of a
  step is four hits. It also keeps the time and radius of the first hit
  for the run, which is the number the "when do hits start" question is
  asked of.
- **The validity monitor runs whether or not the projection is on.** The
  projection says *that* a state left the range and where; the monitor
  says how close the layer and the evolved points next to it are to
  leaving it, so a run that ends in a degenerate metric has its approach
  on the record. The lapse is the signed `sign(α²)√|α²|` from the ADM split
  rather than `metric_quantities`' `1/√(−g^{tt})`, which has no value for
  the states the row exists to report.
- **`finite` is the evolved region's (amended in step 8b).** It was
  `all(isfinite, u)` over the whole state, which with `max_speed_of`'s
  identical check ended a run at the first `NaN` in the frozen core; both
  now count non-finite values at `r ≥ r_1` (`evolved_nonfinite`). A `NaN`
  in the core is the projection's business, not the end of the run. The
  *initial data* is still checked everywhere, before anything else: a
  non-finite value there is a case whose analytic solution is singular
  somewhere the mesh reaches, which is the configuration error step 5's
  check was written for, and not something to repair.
  `diag` has **22** slots from step 8b: the projection's three, the
  monitor's four and the non-finite count, appended.

**(Implemented and measured in step 6.)** The mesh-statistics row gains
`τ_max` and the refinement centroid, and both come out of the flagging pass
the regrid needs anyway:

- **The centroid is the volume-weighted centroid of the cells above
  `coarsen_tol` (proposed in step 6)**, taken over the bounding boxes the
  `coarsen_tol` sweep already reports — each block's box center weighted by
  its firing count and its cell volume. A centroid over cells rather than
  boxes would be a third sweep of the mesh for a diagnostic.
- **It has a bias of order the coarsest *firing* spacing, and the reason is
  the mesh's indexing (measured in step 6).** On the static hole, where the
  mesh and the solution are symmetric about the center, the centroid comes
  out `1.7`–`2.4` finest spacings from it. TreeAMR's blocks own their lower
  plane and not their upper one, so at a coarse-fine interface the coarse
  block on the `+` side owns the interface plane while the one on the `−`
  side starts a coarse cell further out; those interface points carry the
  largest `τ` of their level and fire on one side only. G5's "within a few
  finest spacings of the analytic center" has to be read against that, and
  a tighter measurement would have to weight cells rather than boxes.

The numbers: on flat space both monitors are **exactly zero** at every
order; on the six backgrounds of the table, with *analytic* second
derivatives from `ddmetric`, `ℋ` and `ℳ_i` vanish to **1.2 eps** and
**0.2 eps** of the size of the curvature terms; on the gauge wave across a
coarse-fine face they converge at **5.47** (`C_a`), **4.51** (`ℋ`) and
**5.47** (`ℳ_i`); and on harmonic Kerr (`a = 9/10`, a uniform mesh, no
hole in the box) at **4.26**, **4.21** and **4.27**, which is `q`. The two
rows measure different things and both are needed: the gauge wave on a
*uniform* mesh has no constraint violation at all above roundoff — it
depends on `x − t` alone, so the temporal and spatial truncation errors
cancel — so the refined rows are the interface error and nothing else, and
they converge faster than `q` because that error lives on a set of measure
`~h`, which is worth half an order in an L2 norm. Harmonic Kerr is the
bulk truncation error and nothing else, and it converges at `q`.

**Horizon.** `ApparentHorizonFinder` (Gundlach's fast-flow method)
takes an ADM-variable provider — `ADMVars(γ_ij, ∂_k γ_ij, K_ij)` at a
point, or its batched form over an array of points, which is the form
this package implements so that the interpolation runs threaded — and
a seed sphere `(origin, r0)` at resolution `N_ah`; it returns
`(; success, origin, hlm, area, grid, …)`. `K_ij` comes from `Π`
through the evolution relation, GHSO2's `adm_vars_from_state`. The
provider is the one place this package interpolates: point location by
`find_leaf`, then Lagrange interpolation of order `q + 2` over the
containing block's stored points (which reach `G` into the neighbors,
so a point near a block face is interpolated without crossing it), on
the host at analysis cadence — a stopgap for TreeAMR's "generic
interpolation" to-do, and the first item under
[Upstream prerequisites](#upstream-prerequisites). **(Amended
2026-09-26:** it is TreeAMR's `interpolate` (M11) now, with the basis
`Lagrange(q + 2)` — the same stencil, located by one binary search rather
than an ancestor walk, one launch on the field set's backend rather than a
host loop, so a device-resident state is no longer copied to the host per
find (`hostcopy`); `gh_interpolate`/`gh_interpolate_grad` are the thin
wrapper, and the provider and the fit's `state_sampler` call it.**)** The
horizon lies outside the layer by the margin `m`, and the provider throws
if a query point's interpolation footprint reaches `r_1`. Each find is seeded with
the previous result, recentred on `c(t)`. **Spin and mass** follow
GHSO2's `find_gh_horizon` verbatim (`notes/methods-ghso2.md`, "Apparent
horizons and spin"): `horizon_spin` gives `J` and its coordinate-space
axis from `γ` and `K` on the same grid; `M_irr` and `M_ch` are the two
lines above. For Kerr the reference values are `A = 4π(r_+² + a²)`,
`M_irr = √(A/16π)`, `J = M a`, `M_ch = M`, and a boost changes none of
them — which is what makes the analysis of the boosted spinning hole a
sharp test rather than a picture.

The horizon finder is a *diagnostic*, not a tracker: the layer follows
the analytic center, and the refinement follows the indicator. A run in
which the found horizon and the analytic center disagree by more than a
few finest spacings has found a bug, not a feature to track. **(Amended
in step 8d:** a case whose interior is a `FittedSpec` asks for its layer to
follow the found horizon, and then the finder's answer is an input —
through `src/tracking.jl`'s track, from the row the driver records, under
[The interior](#the-interior-a-pointwise-damping-layer), "The tracked
geometry". Every other case is as this paragraph says, and `track_offset`
is the row that says whether a tracked one has found a bug.**)**

**(Implemented and measured in step 7**, `src/horizon.jl`.**)** Five
things the writing settled, each stated where it is made in that file:

- **The guard is on the footprint and it is exact (proposed in step 7).**
  `CODE.md` asks the provider to throw "if a query point's interpolation
  footprint reaches `r_1`", which is a statement about `(q+2)³` points and
  not about the query. The footprint is a tensor-product lattice, so the
  nearest of its points to the hole's center is the per-axis nearest in
  each direction, and the test is three clamped roundings and a sum of
  squares — exact, rather than the bounding box's conservative
  `√3·(q+2)h/2`. It fires on a query that is *outside* `r_1` and reads
  inside it, which is the whole point of putting it on the footprint.
  **(Amended 2026-09-26**, with the port to TreeAMR's M11:**)** TreeAMR's
  `interpolate` takes an `exclude::Region` and *flags* a query whose
  stencil has a point inside it; the guard is the region
  `UnevolvedRegion(mask)` — inside where `is_evolved(mask, x)` is false —
  and `gh_interpolate` throws the same `ArgumentError` for the first
  flagged query. The test is now exact **bit for bit against the norms'
  mask**, which the stopgap's was not quite: TreeAMR evaluates a stencil
  point at `origin + ((base + k) − off)·h`, the expression `coordinates`
  and `point_position` evaluate in the same order (the stopgap stepped from
  the footprint's corner, `x₀ + k·h`, which rounds differently), and the
  region asks the norms' own predicate. The two masks carry cheaper
  `stencil_hits` methods that agree with enumerating the stencil exactly —
  the round one by the per-axis nearest point, whose `r²` is formed by the
  same arithmetic as `is_evolved`'s and is the lattice minimum because
  floating-point addition is monotone; the tracked `ShapeMask` by the same
  point against its two bounding spheres, compared through `sqrt` as
  `is_evolved` compares, and by enumeration between them. TreeAMR's own
  `Ellipsoid` is not used for the round mask: `Σ((x − c)/r₁)² < 1` rounds
  differently from `Σ(x − c)² ≥ r₁²`, and it buys nothing — on the 496-point
  batch below, `nothing`, an `Ellipsoid` of radius `r₁(1 + 8ε)` and
  `UnevolvedRegion` cost `0.1068`, `0.1090` and `0.1084 ms` at four threads
  and `0.369`, `0.368`, `0.368 ms` at one, all within the noise: the test is
  `3n` subtractions against the `20 n³ · 4` multiply-adds of the
  contraction, and it reads no field data at all. On a `ShapeMask` whose
  offset surface the footprints straddle, the override and plain
  enumeration are `0.160` and `0.162 ms` (the band is enumerated either
  way); outside the band `0.108` against `0.120`.
- **The radii are measured from the analytic center (proposed in step
  7).** `CODE.md` says "the coordinate radii of the surface points" and
  not from where. The two claims made on them — that the horizon encloses
  the layer by the margin `m`, and that `r_min` and `r_max` are
  [`horizon_min_radius`](#the-interior-a-pointwise-damping-layer) and
  `horizon_max_radius` — are both statements about the center the layer is
  built around, so that is the center. The distance between the two
  centers is a row of its own, `center_offset`, and it is the number the
  paragraph above turns into a bug report. `r_mean` is the `sin θ`-weighted
  mean over the collocation points, which is `∮ r dΩ/4π` to the accuracy
  of the grid's own quadrature. **(Amended in step 8d:** `find_gh_horizon`
  takes `center =`, the point the radii and `center_offset` are measured
  from, defaulting to the analytic center, and also returns the radii about
  the surface's own recentred origin, `origin_r_min`, `origin_r_mean`,
  `origin_r_max` — what the shape `hlm` describes, and what a tracked
  geometry is built from. A tracked run passes its predicted center, so its
  rows' `r_min` … `r_max` are about the track and its `center_offset` is the
  track's prediction error.**)**
- **A failed find is recorded, not thrown (proposed in step 7).** The
  horizon is a diagnostic and nothing in the evolution reads it, so a
  find that throws — the guard refusing a flow that wandered inward, a
  seed sphere outside the box — leaves `horizon_success = false` and the
  message in `horizon_note` beside the chunk it happened at, and the run
  goes on. A diagnostic that ends a run loses the record that would have
  said why.
- **The spin's uniformization tolerance is `1e-8` and not the library's
  `1e-13` (proposed in step 7).** `1e-13` is the round-off floor of
  *analytic* Cauchy data; data interpolated off a finite-difference mesh
  has a floor set by its own error — `2.2e−5` on the suite's hole at
  `h = 5/64` — so the library's default stalls there and reports
  `success = false` on every find while returning the same `J` to eleven
  digits. The looser tolerance makes the flag mean something and does not
  move the number.
- **The cadence is a `Horizon` struct on the case**, like
  [`Refinement`](#refinement-and-regridding) and for the same reason: a
  driver keyword would make `k` a property of the run rather than of the
  study. It carries `every`, the finder's `N`, the seed radius (zero
  meaning "derive it from the background's analytic radii"), whether to
  compute the spin, and the three tolerances.

The numbers are under [Measured results](#measured-results): Kerr's
`A`, `M_irr`, `J` and `M_ch` recovered from sampled data in both charts
and at `a = 9/10`, the interpolation's order, and what a find costs.

## Precision, threads, devices

Inherited discipline, restated where this package has more places to
break it:

- **No floating-point literal in an expression where `T` is in play.**
  The pointwise algebra is dense with halves and small integers; they
  are `T(1//2)`, `oftype(x, 2)`. `precision.jl` (`wrap`, `ceilint`,
  `floorint`, `tofloat64`) is copied from TreeWave.
- **Callbacks capture `isbits` only.** Backgrounds (`SpacetimeMetrics`
  structs), the interior profiles, the damping profile, the indicator's
  thresholds and floors are tuples and small structs; the center is a
  function of `t`, never a mutated field.
- **Bit-identical across thread counts.** Every reduction goes through
  `block_mapreduce` or `firing_boxes`; `test/thread_workload.jl` runs a
  gauge wave with a regrid and a chunk of a hole with its layer and
  compares digests. **(Measured in step 4.)** It holds. The workload is
  the gauge wave on a `2³` root grid: the initial-data cycle with its
  flagging pass, two chunks of fixed-step RK4 on a two-level mesh, the
  gauge-constraint monitor and its masked norms, `max_speed`, and a regrid
  that moves data between levels — six printed lines of digests, norms and
  mesh statistics, **character for character identical** at one and at
  four threads. The workload carries the *gauge* monitor and not the ADM
  one **(proposed in step 4)**: the ADM kernel is the same `map_blocks!`
  launch reduced by the same fold, so it adds no parallel structure, and
  compiling it costs 18 s in each of the two processes the test runs.
  What this bit-identity does *not* mean is under
  [Measured results](#measured-results): the same compiled code, at the
  same call site, at a different thread count, and nothing more.
  **(Amended in step 5:** the workload stays the gauge wave and does
  *not* grow a chunk of a hole with its layer. The sentence above was
  written before the interior existed and promised one; what the interior
  adds is a per-point branch inside one `map_blocks!` kernel, two more
  `map_blocks!` launches (the error kernel and the `:pasted` limiter) and
  three more `masked_norms` folds — no new parallel *structure*, and the
  workload already exercises a `map_blocks!` kernel, a masked fold and
  `max_speed`. What it would cost is a second background's dual passes and
  two more kernel specialisations, in each of the two subprocesses the
  test runs, for a claim the existing lines already make. The right time
  to reconsider is G5, where the hole *moves* and the refinement follows
  it — that is a new flagging pass, which is new parallel structure.**)**
- **Every per-block pass runs on the block's owner thread, and this
  package adds none of its own** (added 2026-09-25, with TreeAMR's
  owner-based threading of 2026-09-23, `4be726e`). TreeAMR now gives
  block `b` to the thread whose chunk of `threadchunks(nblocks)` holds it,
  in every `map_blocks!`, `scatter!`/`gather!`, `fill_by_coordinates!`,
  ghost fill, interface fixup, regrid transfer and host loop over blocks,
  because a block that changes core between launches streams at up to
  `1/2.4` the rate on a 64-core EPYC (TreeAMR's `CODE.md`, "What one
  process loses"). Every kernel here — the right-hand side, the limiters,
  the monitors, the indicator, the target cache — already goes through
  `map_blocks!`, and every state vector and field set is allocated by
  TreeAMR, so all of that is inherited with no change to `src/`; `[compat]`
  asks for TreeAMR `0.1.2`. The one parallel loop this package wrote
  itself, the horizon interpolator's batch over query points, was
  `Threads.@threads` and is TreeAMR's `threaded_foreach` now: the same
  chunk on the same thread every call, nestable, and it rethrows the
  body's own exception, so the footprint guard's refusal still reaches
  the record readable. **(Amended 2026-09-26:** that loop is gone with the
  stopgap. The batch is TreeAMR's `interpolate`, one KernelAbstractions
  launch over the points — not a by-owner launch, since points are not
  blocks — which flags rather than throws, so the guard's refusal is raised
  on the host after it; this package has no parallel loop of its own
  now.**)** **What is not owner-based is RK4's stage
  arithmetic.** OrdinaryDiffEq's `RK4()` forms `uprev + (dt/2) k` and the
  final combination with a serial broadcast (`thread = Serial()`), on the
  calling thread, over the whole state. **(Measured 2026-09-25** on the
  development machine, the gauge wave at `q = 4`, 64 blocks of `16³`:**)**
  the four broadcasts of a step are `5.7 ms` against `4 × 519 ms` of
  right-hand side at one thread (0.3 %) and `4 × 143 ms` at four (1.0 %).
  On a 64-core Symmetry node it is not measured, and is expected to
  matter more for two reasons: a single core's bandwidth is a small share
  of the node's, and the broadcast reads every block from one core
  between launches, which is exactly the migration TreeAMR removed. It
  also first-touches the vectors the integrator allocates (its copy of
  `u`, the stage vector `tmp`, the saved end state) from that one core, so
  TreeAMR's advice — pin the threads, then let first touch place the
  pages — makes the field sets domain-local and leaves the state vectors
  on one domain. Today's Symmetry jobs neither pin nor interleave; pinned,
  with and without `numactl --interleave=all`, is the comparison to make
  before choosing. The
  ways out are `RK4(; thread = True())` (Polyester, whose chunk-to-thread
  map is its own) or an integrator whose stage update is a `map_blocks!`
  — a change to "RK4 from OrdinaryDiffEq", and G6's to measure.
  **(Resolved 2026-09-26.)** Measured on Symmetry (job 563749), the serial
  updates were 12–14 % of a step at 64 threads, and every `solve` added
  the serial allocation of about fifteen state-sized vectors; the
  integrator is IMEXRungeKutta's RK4 by block owner since, which removes
  both — [Time integration](#time-integration) has the numbers. The
  placement question is answered there too: pinned with first touch is
  the fastest configuration, and the 3–5× first-touch anomaly of job
  563749 did not reproduce once nothing serial touched the state.
- **`Float64` on Symmetry's H200 is the requirement** (decided in
  review). It is the machine the proof of concept runs on, and the
  precision it runs in; every device claim below is made there first.
  `Float32` on a device — Metal on the development machine, or `Float32`
  on the H200 for the register-footprint experiment — is *desirable*:
  a hole at `Float32` is a real test of the offset identities (`h` near
  the horizon is O(1), and they are what keep `α` and `√γ` accurate),
  GHSO2 ran its whole suite on Metal in `Float32`, and the design
  inherits TreeAMR's type genericity so nothing stands in the way; but
  a `Float32` failure would be recorded, not fixed at the expense of
  `Float64`. The RHS kernel's register pressure and the in-kernel metric
  evaluation are the two device unknowns, and G6 measures both on the
  H200. **(Measured in step 4** on `CPU()`, which is what a host test can
  settle.**)** There is no failure to record. Everything steps 3 and 4
  built runs at `Float32` and returns `Float32`: the fused kernel, the
  time step, RK4, both constraint monitors, the two-level mesh with its
  prolongated ghosts. The gauge wave converges at **3.93** at `Float32`
  against `Float64`'s **3.94** over the two coarsest resolutions, with
  errors `2.00e−4, 1.31e−5` against `2.00e−4, 1.30e−5` — the *same* error,
  not merely the same rate. Two resolutions and no more, for step 2's
  reason: the roundoff floor of a second derivative is `eps/h²`, which at
  `Float32` and `h = 1/24` is already the size of the `Float64` truncation
  error there, so a third point would measure the arithmetic. The
  pointwise algebra — including step 4's `metric_derivatives_along` and
  the four-dimensional curvature assembly — also runs at `Float32x2`, the
  software float that has no hardware path underneath it, and agrees with
  the `Float64` answer on the same rational data to each type's own
  precision.
- **GPU kernel efficiency is deferred, and here is the budget it
  competes against.** Per point and per RHS evaluation, TreeAMR's
  pattern moves roughly: the scatter (read `u`, write the working
  array: 320 bytes at 20 `Float64`), the ghost fill (the ghost points
  of a `N = 32`, `G = 3` block are 70 % of its owned points: about 110
  bytes written), the kernel (its unique reads are the ghosted block,
  about 270 bytes, plus 160 bytes of `du` written), and the RK4 stage
  update (read `u` and `k`, write the stage vector: 480 bytes) — about
  1.3 kB against the 120 bytes of GHAccel's minimal leapfrog. On an
  H200 that traffic alone is about 270 ps per point, while the
  arithmetic — perhaps 5–8 kflop per point at `q = 4` — is 100–250 ps
  at the FP64 peak. So the surrounding data movement and the kernel's
  arithmetic are of the same order, and neither dominates: a kernel at
  a quarter of the arithmetic peak, which is where GHAccel's landed on
  the same hardware without spilling, would already be at parity with
  the traffic around it. That is the argument for deferring: the first
  real gains on a GPU are as likely to come from fusing the pattern
  (scatter, stage update) into the kernel — a TreeAMR-level change — as
  from the kernel's own register schedule, and both are research.
  **(predicted)** and re-derived from the measurements in G6, which
  record for the RHS kernel: registers per thread, spill bytes
  (`ld.local` in the PTX is the tell), achieved occupancy, and
  picoseconds per point against the roofline in the format of
  `notes/ghaccel-bench.jl`, whose numbers (25.8 % of the H200 roofline,
  49.6 % on an A40, for 10 fields) are the reference this kernel is
  compared to. **(Measured 2026-10-05**, in [The right-hand side on an
  H200](#the-right-hand-side-on-an-h200-measured-2026-10-05).**)**
  - **The arithmetic:** 3867 FP64 instructions a point, about 6.5 kflop, inside the
    prediction.
  - **The cost was neither the arithmetic nor the traffic:** the kernel held 255
    registers and spilled 3.8–8 KB a thread, at about 5 % of FP64 peak.
  - **The lean source alone:** 52 % of FP64 peak.
  - **Around the kernel,** TreeAMR's scatter and ghost fill run at 0.65–1 TB/s.
  **(Measured in step X3**, `ptxas`'s report on the H200 at
  `Float64`, `q = 4`**:** every right-hand-side kernel — `:damped`, `:excised`
  with and without the blend, the excision's zone kernel — uses all 255
  registers with a stack frame of 8.8–10.6 kB and 170–320 bytes of spills
  proper; the frame is the 112–143 device calls that are not inlined,
  whose `SVector` arguments pass through local memory. The `:damped` hole's
  right-hand side is `11.4 ns` a point on the octant's `N = 64` mesh.
  Under [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-on-the-static-hole-step-x3), "Excision on the static
  hole (step X3)".**)**

## I/O and viewers

- **The analysis time series**: one HDF5 file per run, one dataset per
  quantity in the table under [Analysis quantities](#analysis-quantities),
  appended at every chunk boundary and flushed, so that a killed run
  keeps its record up to its last chunk; the horizon shape coefficients
  per find beside them. This is what the viewers and the long-run tests
  read.
- **Output for visualization**: slices through the block hierarchy
  written as HDF5 with block extents and levels, and `bin/` viewers in
  CairoMakie after TreeWave's `visualize2d.jl`: a slice of `α` or
  `h_tt` with block outlines colored by level and the layer's two radii
  drawn, the horizon's cross-section when found, the indicator `τ`,
  constraint norms and the error against time. Volume output in a
  VTK-readable form is an extension.
- **No checkpoint and restart** (decided in review): a proof of concept
  runs to completion. When it is needed it is a few dozen lines over
  TreeAMR's leaf keys and the state vector, or TreeAMR's M9; it is under
  [Possible extensions](#possible-extensions). **(Amended 2026-10-01:** it
  is TreeAMR's M9a, and [Checkpoint and restart](#checkpoint-and-restart)
  is its section.**)**
- **A live status file for SimWatch (added 2026-10-02)**: `src/simwatch.jl`
  writes `simwatch.toml` into a run directory for SimWatch
  (`https://github.com/eschnett/simwatch`, whose `FORMAT.md` is the
  reference), a terminal viewer that shows every run below a directory with
  its progress, its Slurm state and its diagnostics. It began as that
  repository's Julia writer and is owned here, layered as a pure document
  builder (`simwatch_document`), an atomic write that never throws
  (`write_simwatch`) and a rate-limited `SimWatchWriter`, which reports
  `update_interval` from the observed spacing of its calls so that a run
  whose chunks take minutes is not shown as stale between them. Its data
  come from `evolve!`'s observer, which is handed the chunk's record row when
  it takes a fourth argument (`observer(p, t, u, row)`, added the same day),
  so the horizon's numbers are not recomputed; `test/octant_runs.jl` writes
  everything its CSV carries — constraints overall, by level, in shells and
  at the boundary, the error, the horizon, the track, the fit and the step —
  and `finished`, `stopped` or `failed` at the end.

## Checkpoint and restart

*(Added 2026-10-01, on TreeAMR 0.1.4, whose M9a adds `save_checkpoint` and
`load_checkpoint` through a package extension on HDF5. Erik's decision of
that day reversed "No checkpoint and restart" above; the mechanism is
TreeHydro's — its `CODE.md`, "Checkpoint and restart", 2026-09-29 — on
purpose, so that the applications of TreeAMR checkpoint alike, with the
two differences marked.)* G5's crossing is eleven hours on a node, the
`a = 9/10` run `38`–`149 h`, and a queue's day is the limit. **Everything
about the file is upstream**: the forest, the field-set layout, the atomic
and durable write, element types as limbs, the provenance and the refusal
of a file a version cannot read — "no mesh machinery" applied to I/O. What
`src/checkpoint.jl` holds is when to write, the run state, the refusal of a
restart with other parameters, and the files' names and rotation.

**HDF5 is a hard dependency** (decided 2026-10-01; *TreeHydro leaves it to
the caller*). `TreeGeneralizedHarmonic.jl` imports it, which loads
TreeAMR's `TreeAMRHDF5Ext`, so a run can always be checkpointed and no job
script has to remember `using HDF5`; `prerequisite_tests.jl` asserts the
extension is loaded.

**Where: at a chunk boundary, after the analysis row and before the regrid**
(decided 2026-10-01; *TreeHydro writes after the regrid*). At a chunk
boundary the fixed-step integrator holds nothing but `(t, u)` —
IMEXRungeKutta's scratch carries no value from one step to the next — so a
restart that restores the state and the run state begins where the
uninterrupted run would. Before the regrid, so that a restart can **regrid
with a changed criterion** as the first thing it does, before its next
step: the flags are computed again from the restored state with the
indicator the restart is given — the record computes them last, from
nothing the other rows leave behind, so with the same criterion they are
the same flags — and the regrid is the loop's own code (`regrid_mesh` in
`driver.jl`), which is what makes the replay exact. The price is one
indicator evaluation per restart. A second consequence: **the last chunk is
a restart point too**, where `t_end` is a whole number of chunks in `T`
(`1/10 = 2 · 1/20` is, `3 · 1/20 ≠ 3/20` is not), since a longer run would
have regridded there; elsewhere it is not written, because a continued run
would move the chunk boundaries.

**Would TreeHydro's work the same way?** Yes, and nothing in TreeAMR is in
the way: its regrid flags read the primitives, which `update_primitives!`
rebuilds from `u`, its buffer is derived from `λ_end` (in its history), and
its post-regrid atmosphere reset is a function of `(u, p, t)` — so a restart
before the regrid replays the flags, the regrid and the reset, as this
package replays the indicator and the regrid. Its recipe would split off the
criterion as below. That change is TreeHydro's to make.

**What is saved.** The forest and the state field set `U` with its state
vector `u`, through TreeAMR, and this package's plain data `(; recipe,
criterion, run)` in its group `TreeGeneralizedHarmonic.jl`, format version
1. `u` and not `U.work`: the row's find, fit and indicator have scattered
into the working array, and without a regrid it holds the integrator's last
stage. The **run state** is what the uninterrupted run would carry into the
next chunk and cannot be recomputed from `(forest, u, t)`; since step 8d
that is a good deal more than TreeHydro's accumulators, and the old note
under [Possible extensions](#possible-extensions) — "bit-identical, since
the layer and the hook depend on `(x, t)` only" — was no longer true:

- the **horizon track**, after the row's update (the next chunk's geometry
  and the next find's jump test), and the track the chunk was built from
  (the chunk's interior, which the restart's indicator masks with);
- the **two fits**, the row's and the previous one (the target and its
  slope), each its parameters and coefficients — the backend's copy is made
  from the host's on load — and the **initial fit** and the **target's
  ranges**, derived once on the initial mesh;
- the **finder's seed**, the previous find's shape, and the pending
  **lapse-collapse trigger**;
- the **gauge source's sample**: the time and the track of the geometry it
  was sampled with, so that the restart samples it with the same core rule
  (a non-harmonic background's tracked core is re-sampled only when it has
  moved half a cell, step 8d);
- the **speed growth** `λ_end/λ` that sizes a moving hole's next step (step
  8f), in `T`, and the chunk's relaxation rate;
- the **range projection's accounting**, the counters, the fits' costs and
  **the record** so far, so that a restarted run's *answer* is the
  uninterrupted one's and not only its state.

**Reals are stored exactly**: TreeAMR's plain data refuse a MultiFloat
scalar, so `plain_reals` stores a native float as itself and any `isbits`
real made of one native float throughout as its limbs (TreeHydro's rule).
**The run state's structs** (`HorizonTrack`, `FitParams`, `StateBounds`,
`BoundsAccounting`, …) are stored field by field and rebuilt from their
declared field types, so that an `SVector{3,T}` comes back as one and not as
the `Vector{Float64}` a plain read gives; only this package's own structs
are taken apart, and anything else is refused with its field's path. The
fits' diagnostics come back as plain `Float64` tuples, which nothing the run
computes reads. The record is already plain data (its vectors are tuples).

**The recipe and the criterion** (decided 2026-10-01; *TreeHydro's recipe
holds its criterion*). A restart is called with the same case and keywords
as the run it continues, and **only `t_end` and the regridding criterion may
change**. The recipe is every parameter that decides a number: the working
type by name, `q` and the operators; the case's background, box,
dissipation, damping, center, interior, horizon finder and bounds (each as
its `repr`, which prints every real in full); and `chunk`, `cfl`, `adapt`,
`adm_every`, the rates and the fitted target's switches, every real through
`T` and then `plain_reals`. A restart whose recipe differs is refused with
**one** `ArgumentError` naming every field that differs and both values. The
criterion is the case's `Refinement` field by field, `regrid` and `buffer`;
a restart whose criterion differs is run, reported field by field, and says
so in `criterion_changed`. The forest is not in either — a restart takes the
file's — and `backend`, `maxpasses`, `find` and the observer decide no number
once the initial data exist.

**The interior's `repr` is unchanged by the `:excised` variant (amended in
step X2b).** Its parameters are a field of `Interior` and `FittedSpec`, so
an excised case's recipe holds them; an interior without them prints byte
for byte as before the field existed (a legacy `show`), so every older
checkpoint still restarts. An excised hole's geometry is frozen and rebuilt
from the case on a restart, so it adds no run state.

**Names, rotation, triggers, the observer — TreeHydro's.** The files are
`"<prefix>.it<iteration>.h5"`, the cumulative step count zero-padded to ten
digits; after a successful write every file of the prefix but the one just
written and the newest `num_checkpoints_keep − 1` others is deleted, earlier
jobs' included; `latest_checkpoint(prefix)`, exported, makes a job chain one
command for every job:

    r = evolve!(case; …, checkpoint_path_prefix = prefix,
                max_walltime_seconds = 23.5 * 3600,
                restart_file = latest_checkpoint(prefix))

with no `forest` once the file exists (a restart refuses one). The triggers
are `checkpoint_every_chunks`, `checkpoint_interval_seconds` and
`max_walltime_seconds`, the last stopping the run — `finished = false`, the
checkpointed state returned, before its regrid — when the elapsed time plus
the longest chunk so far, regrid included, plus the longest write would pass
the limit. Timing decides only *when* a file is written. The observer is not
called for the rows a restart brings back; it sees the chunks it runs.
`test/hole_runs.jl`'s `generic` and `moving` workers take `checkpoint=<dir>`
and `walltime=<s>`, so the same command resubmitted is a job chain.

**What it amounts to** (measured 2026-10-01, `test/checkpoint_tests.jl`): a
chain of restarts, one chunk per job, is the uninterrupted run **bit for
bit** — state, mesh, record, counters, track, accounting and the fits'
coefficients — on the step-5 fixture with the range projection on, on the
adaptive fixture through the regrid that moves its mesh, on a tracked
`:fitted` Kerr-Schild hole through a regrid (its gauge source re-sampled),
and on a moving `:fitted` harmonic hole with step 8′'s trailing ramp; a
finished run continued from its last chunk is the longer run; a restart that
no longer coarsens keeps the corner the uninterrupted run removes. On a real
row — `hole_runs.jl moving=l0-trail-9 t_end=1/2`, the boosted `a = 0` hole
on 820 blocks — the two-job chain's last row is the uninterrupted run's in
every printed digit, at the same 84 steps, from a `68 MB` file.

**What it costs** (measured 2026-10-01 on the development machine, four
threads, its SSD; `save_run`/`load_run` of a 20-variable vertex-centered set
at `N = 8`): on 3536 blocks — G5's mesh at rest is 3900–5600 — the file is
`290 MB`, a save `0.09 s` (`3 GB/s`), `0.11 s` with the flush to stable
storage, and a load `0.37 s`, once compiled; the first save and the first
load of a session compile for `3 s` and `5.5 s`. A G5 chunk is about
`160 s` of a node, so a checkpoint every chunk costs under a tenth of a
percent. The record of 53 rows is noise beside the state. Symmetry's BeeGFS
is not measured.

## Upstream prerequisites

What this package needs from TreeAMR. None blocks G0–G3.

1. **Point interpolation from a field set** (G4, for the horizon
   finder). `find_leaf` plus block-local Lagrange interpolation, batched
   over a host array of points. Written here first as a stopgap;
   TreeAMR's `TODO.md` lists "generic interpolation". **(Written in step
   7**, `src/horizon.jl`: `locate_block` — the descent TreeAMR does not
   export, root brick to finest cell to the first ancestor that is a leaf
   — and `interpolate`/`interpolate_grad`, the tensor-product Lagrange
   window of `q + 2` points with its value and its gradient formed in one
   pass. Two things the port upstream will have to keep: the window is
   *half* the ghost width on each side, which is what makes a query
   anywhere in a block readable without crossing into another block's
   array at `G = q/2 + 1`; and the batch is what makes it threaded, with
   one output slot per input point and no accumulation, so the answer does
   not depend on the thread count.**)**
   **(Done upstream, 2026-09-26:** TreeAMR 0.1.3's M11 — `locate_point`,
   `interpolate`/`interpolate!` with a `Lagrange(n)` basis, `derivs` as
   multi-indices, an `exclude::Region` that flags, one launch on the field
   set's backend — kept both properties above and adds the ones the stopgap
   lacked: one binary search per point instead of `maxlevel` `find_leaf`s,
   device execution, periodic and reflecting faces. TreeAMR 0.1.3 is
   registered, and `Project.toml` takes it from General (`[compat]`
   `0.1.3`, no `[sources]` entry). The port removed `locate_block` and the
   exported `interpolate`/`interpolate_grad`, whose name collided with
   TreeAMR's; what stays in `src/horizon.jl` is the wrapper
   `gh_interpolate`/`gh_interpolate_grad` (the order `q + 2`, the unpacking
   into `SVector`s, the refusal) and the guard as a `Region` — see
   [Analysis quantities](#analysis-quantities), where the guard's exactness
   and the measured costs are.**)**
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
   is in use: [Checkpoint and restart](#checkpoint-and-restart).**)**
4. **A launch-configuration knob on `map_blocks!`** (G6): today it
   launches with KernelAbstractions' default workgroup, and GHAccel
   measured the workgroup shape of a 3D stencil kernel as worth about
   10 % of peak. A `workgroupsize` keyword passed through to the launch
   is the whole request. **(Measured 2026-10-05**, [The right-hand side on an
   H200](#the-right-hand-side-on-an-h200-measured-2026-10-05)**.)**
   - **The right-hand side no longer needs the knob.** Now that its kernel forms
     no closure, KA's default workgroup is within 1 % of the best shape at
     `N = 32` and `128`, and 2.7 % at `16`. Forcing inlining changes nothing
     (1.075 against 1.077 ns a point).
   - **Two halves of the request are still worth having, second to item 5.**
     `map_blocks!` launches with `get_backend(fs.work)`, a default
     `CUDABackend()`, so the backend a field set was built with — its
     `always_inline` and a static workgroup — never reaches the launch.
     Keeping it matters to the kernels that still have closures: the interior
     variants' layers and the monitors. Before this package's kernel was
     rewritten, forcing inlining was worth 1.6×, and a static workgroup 20 %
     to the prototypes.
5. **TreeAMR's copy kernels at bandwidth: the first thing TreeAMR should change
   for a device** (added 2026-10-05).
   - **What is left of an evaluation is TreeAMR's.** After this package's kernel
     went from 8.5 to 1.1 ns a point, `scatter!` and `fill_ghosts!` are 0.73 of
     `gh_rhs!`'s 1.81 ns at `N = 32`, 1.39 of 2.61 at `N = 16`, and 0.41 of 1.70
     at `N = 128`.
   - **Both run far below the H200's 4.8 TB/s.** The scatter moves 320 B a point
     at 0.96 TB/s, and the transfer kernel its ghost data at 0.65 TB/s.
   - **Why:** both launch over a 5-D `CartesianIndices` with run-time sizes. KA
     forms that index with integer divisions for every element, about 1200 SASS
     instructions to copy one value.
   - **What it would buy:** a scatter with the strides known is 2.4× faster
     (0.14 ns a point, measured in `bench/rhs_lab.jl`'s `round5`). A fill at
     copy bandwidth would be about 4× faster, which at `N = 16` saves more than
     the whole kernel costs.
   - **The CPU has the same problem** (measured 2026-10-05, [The right-hand side
     on a CPU](#the-right-hand-side-on-a-cpu-measured-2026-10-05)). On Symmetry's
     EPYC nodes the fill costs about 10 ns a ghost value on one core — a seventh
     to a twentieth of copy speed — and a quarter of memory bandwidth at 64
     threads, where it is a third of `gh_rhs!` at `16³` and, once the kernel is
     vectorized across points, more than half. The profile names the same costs:
     the copy as a weighted tensor-product stencil, KernelAbstractions' 5-D
     iteration, integer index arithmetic, and at 64 threads a fifth of the time
     waiting between phases.
   - **On a refined mesh the prolongation is the cost** (measured 2026-10-05 on the
     hole fixture's mesh, 64 threads): the fill is 145 ms of a 200 ms `gh_rhs!`
     against a 7.8 ms copy floor, a third of its samples threads waiting, so the
     prolongation groups want balancing across threads as well as fewer operations
     a ghost. Once the kernel is vectorized, this is what a hole's evaluation costs.

   In order:
   1. **`scatter!` and `gather!`** with an index that is cheap to form: a
      linear range over owned points, variables and blocks, divided by
      constants, or the block size as a type parameter.
   2. **`transfer_kernel!` the same** for its copy groups (`Ps = (1, 1, 1)`,
      the bulk of a uniform fill): a linear source and target offset per
      transfer, and no `stencil_sum` for a copy.
   3. **Then a state that lives in the working array**, for a native stepper
      that writes the next stage's input straight into it (the stage vectors
      as field sets). That removes the scatter altogether.

Two further items are *not* needed for the proof of concept and are
recorded with the extensions that would need them: an interior-reading
device boundary hook (radiative boundaries, excision) and excised leaves
(excision). **(Amended 2026-10-05:** excision by per-stencil closures —
[Excision](#excision-added-2026-10-05) — needs neither, **(predicted)**. A
`map_blocks!` over a subset of the blocks would let the zone kernel skip
the rest without a launch, and is a wish, not a prerequisite.**)**

## File layout

| file | contents |
|---|---|
| `notes/` | verbatim copies of the inherited documents, with provenance; see `notes/README.md` |
| `src/TreeGeneralizedHarmonic.jl` | module shell: `using`s, exports, includes |
| `src/precision.jl`, `src/device.jl` | copied from TreeWave, with TreeHydro's `hostcopy!` split so that the copying half is exercised host to host (amended in step 0) |
| `src/pointwise.jl` | GHSO2's pointwise algebra (ported from `notes/pointwise-ghso2.jl`), plus the expanded form's coefficient derivatives `metric_derivatives`, the assembled `gh_node_rhs_expanded`, and `gh_node_source` (all added in step 1), and `metric_derivatives_along` — the same chain rule along **one** direction, returning `∂√γ` and `∂γ^{jk}` rather than the assembled `∂(α√γγ^{jk})`, which is what the constraint monitors need along *time* (added in step 4) — and `_pairindex`, the packed slot of a symmetric index pair, so that the file has one packing convention used on two index pairs; `SVector{10}` state, `SMatrix{4,4}` tensors. `gh_node_source` is a **second copy** of the reduced source and the damping, written out of `gh_node_rhs` character for character rather than factored out of it: the port stays diffable against `notes/pointwise-ghso2.jl`, which is what makes it the validated reference, and `test/pointwise_identity_tests.jl` asserts the copy still matches it — to roundoff, because two spellings of one expression are not bit-identical (see "Measured results") |
| `src/stencils.jl` | rational finite-difference and Kreiss–Oliger weights at order `q` (added in step 2): `derivative_weights`, `dissipation_weights`, both `@generated` over `(T, Val(q), Val(m))` and returning `SVector`s of `T` for unit spacing; `lagrange_derivative_weights` and the two `rational_*` constructors behind them, exposed unexported so that the exactness claims can be asserted in `Rational` rather than through a tolerance; `dissipation_rank(Val(q)) = Val(q/2 + 1)`, one spelling of `2r = q + 2` **(proposed in step 2)**; the host-side `apply_stencil` and `apply_mixed_stencil`, which are the reference contractions the tests measure with and the definitions step 3's streaming kernel has to agree with |
| `src/evolution.jl` | the fused RHS kernel in streaming order (added in step 3), the linear-index stencil contractions it evaluates, `GHProblem` with the **six** `Val`s (the SIMD width the sixth, from 2026-10-05) and the per-chunk geometry, `gh_rhs!`, the speed kernel, `max_speed`, `gh_dt`, and `convergence_rate` — TreeWave's, in the file TreeWave keeps it in. Step 5 split the streaming body out of the kernel into `gh_rhs_at_point`, an `@inline` plain function, because `F` must not be evaluated in the frozen core and **KernelAbstractions refuses a `return` statement anywhere in a kernel body** — so the core branch cannot be an early exit and has to be an `if` around the whole computation; and added `gh_paste_kernel!` with `gh_step_limiter!` and `paste_interior!`, the `:pasted` variant's one write to the state |
| `src/lanes.jl` | the right-hand-side kernel's SIMD lanes on the CPU (added 2026-10-05): `default_simd_width` and `check_simd_width`, `Lanes` (an array read and written `W` consecutive elements at a time, with the `Const` wrapper unwrapped), `is_lane_leader` (every `W`-th item, and the overlapping last group of a row), and the `Vec` methods of `pointwise.jl`'s `_anynonzero` and `_select`; the kernel's group is `evolution.jl`'s `gh_rhs_lanes!` |
| `src/gauge.jl` | sampling prescribed sources into `Hsrc` and reading them back at a point (`gauge_at`, the kernel's half of the packing); `isharmonic` as a table over the background types and `isstatic` as an exact measurement, with the reason each is what it is (added in step 3); the two `γ0` profiles (step 5) and the `ε_KO(r)` profile `HorizonDissipation`, with `dissipation_rate` the identity on a number (step 8c) |
| `src/boundaries.jl` | the time-dependent Dirichlet hook, and the declarations TreeAMR's symmetries ask of every field set: the parities of the reflecting faces (`state_parity`, `even_parity`, added 2026-10-02) and the signed maps of the rotating seam (`state_rotation`, `identity_rotation`, added 2026-10-04) |
| `src/bounds.jl` | the range projection (added in step 8b): `StateBounds` and the proposed `default_bounds`/`default_gate`, `check_bounds_gate`, the pointwise `bounds_project` over an explicit-scalar ADM split and a Jacobi `sym_eigen3`, `gh_bounds_kernel!`, `BoundsAccounting`, `apply_bounds!` and `gh_stage_limiter!`; the validity monitor (`state_validity`, `validity_rows`); and `evolved_nonfinite`, the masked finiteness check. Included after `interior.jl` and before `initialdata.jl`, whose `GHCase` carries a `StateBounds` |
| `src/interior.jl` | the profiles `w(r)`, `ρ(r)`, the core rule, the radius checks, the masks; added in step 5. Also `HoleCenter` — `c(t) = c₀ + v t` as two vectors and a line, which is what "the center is a function of `t`, never a mutated field" means as code — the horizon's analytic coordinate radii and the hole's mass (`hole_mass`, added in step 8c′), and `layer_spacing`, the coarsest spacing among the blocks the sphere `r_1` passes through, which is the one number in the file that looks at a mesh (and looks at it only to *assert*). The `:pasted` limiter is in `evolution.jl` instead **(amended in step 5)**, beside the kernel it launches and the state layout it writes. **Step 8d adds the tracked geometry's kernel side**: the real harmonics (`real_harmonic_index`, `real_from_complex`/`complex_from_real`, the recurrence `shape_series`, `shape_bounds`), the analytic horizon of the seed (`analytic_horizon_radius`), `FittedSpec` — what a case holds — and `FittedInterior` — the kernel argument — with `interior_point`, `fitted_geometry`, `core_position`, `ShapeMask`, `ShapeBand`, `geometry_spacing` and its `check_interior_radii`; and the protocol both geometries speak (`in_layer(int, t, x)`, `interior_point`, `is_outside`, `geometry_radii`, `layer_radii`, `layer_mask`, `shell_mask`) |
| `src/initialdata.jl` | backgrounds, `GHCase` and the case constructors (here rather than in `driver.jl`, amended in step 3), the forest builders — uniform, with one root block refined for the frozen two-level hierarchy the interface study needs (`refined = true`, added in step 4), or `hole_forest`'s nested shells around a hole (added in step 5, **here rather than in `interior.jl`**, since a forest builder belongs with the other forest builder) — the `(h, Π)` callback with the core rule, the `SpacetimeMetrics` index conversion and nowhere else |
| `src/refinement.jl` | the Löhner indicator with its global floor, the mask, the level floor and ceiling, the four marks, the buffer; TreeWave's `refinement.jl` ported |
| `src/constraints.jl` | the gauge-constraint kernel (state and first derivatives) and the ADM one (every second derivative of `g_ab`, the `∂_t` blocks from the evolution equations, the four-dimensional Ricci tensor assembled rather than reduced), the masks they take — `AllPoints` and the `is_evolved` predicate step 5's interior adds a method to — `masked_norms` and `constraint_norms`, and `adm_constraints_at_node`, the pointwise curvature assembly the tests check against `ddmetric` (added in step 4) |
| `src/horizon.jl` | the interpolating ADM provider for `ApparentHorizonFinder`; location, shape, area, `M_irr`, `J`, `M_ch`. Added in step 7, in the order the numbers are produced: `locate_block` and `interpolate`/`interpolate_grad` (the stopgap of [Upstream prerequisites](#upstream-prerequisites), item 1, with the footprint guard that refuses a query reaching inside `r_1`; **amended 2026-09-26**: `gh_interpolate`/`gh_interpolate_grad` over TreeAMR's `interpolate`, and the guard as the `Region` `UnevolvedRegion`), `GHADMProvider` (batched, `Float64` out whatever the run computes in, with a one-entry cache keyed on the identity of the query array because `KorzynskiSpin.surface_geometry` asks for `γ` and `K` in two calls with the same points), `find_gh_horizon`, and `Horizon` — the cadence and resolution the case carries |
| `src/tracking.jl` | the tracked horizon, host-side (added in step 8d, after `horizon.jl` and before `driver.jl`): the conversions from the finder's `hlm` (`real_shape`) and of the analytic horizon (`analytic_shape`) into real coefficients, `HorizonTrack` with `seed_track`, `update_track`, `track_center` and `TrackLostError`, `fitted_interior` — the kernel argument from a track and a mesh — `surface_shift` (the gauge source's re-sample rule), and `axis_dispersion`/`margin_efolds`, step 8a's leakage e-folds moved in from `test/dispersion.jl` |
| `src/fit.jl` | the fitted target (added in step 8e, after `tracking.jl` and before `driver.jl`): the fit's variables (`fit_variables`, `state_from_fit`), the real solid harmonics (`_solid_harmonic_fold`, `real_solid_harmonics`, `fit_directions`), the two samplers (`state_sampler`, `analytic_sampler`), the least squares (`solve_fit`, `fit_row_weights`), the validity sweep (`fit_sweep`), `FitParams` and `InteriorFit` with `build_fit`, `fit_residual` and `fit_valid`, the kernel-callable evaluator `fit_variables_at`/`fit_state`, and the kernel half (8e-ii): `derive_target_bounds`, the 40-variable cache (`target_cache`, `fit_target_kernel!`, `fill_target!`) and the initial data's `fitted_state_kernel!`. The variant's branch is in `evolution.jl` (`GHProblem`'s `target`/`fits`/`t_target`, `refill_target`), its residual in `constraints.jl`'s error kernel, its flow in `driver.jl` (`refit!`, the refill and the pieces of a moving chunk) |
| `src/checkpoint.jl` | checkpoint and restart (added 2026-10-01, after `fit.jl` and before `driver.jl`): the file names and their rotation and `latest_checkpoint` (TreeHydro's), `plain_reals` for exact reals, `to_plain`/`from_plain` for the run state's own structs and `fit_from_plain` for an `InteriorFit`, `run_recipe` and `run_criterion`, `check_recipe`, `save_run`/`load_run` over TreeAMR's `save_checkpoint`/`load_checkpoint`, and `check_checkpoint_keywords`. `evolve!` writes and reads through it; `test/checkpoint_tests.jl` is its file |
| `src/excision.jl` | the `:excised` variant's mesh half (planned 2026-10-05, step X2b): the per-point classes and the counts `k±`, the closure tables, the zone kernel and the record-time outflow monitor — see [Excision](#excision-added-2026-10-05) |
| `src/driver.jl` | `evolve!`, the analysis record per chunk, `observer`, `check_cfl`, `horizon_shell`, `forest_levels`, `default_relaxation_rate` — the layer's default `4/M`, the one place the number is written (added in step 8c′) — and `discrete_gradient_momentum!` — GHSO2's `Π` post-pass, which lives here because it runs once on the initial data and is the driver's option, not the initial data's (added in step 5). `GHCase` is in `initialdata.jl`, amended in step 3 |
| `src/io.jl` | the analysis time series, slice output |
| `src/benchmark.jl` | per-phase timings in TreeWave's format |
| `test/` | one `*_tests.jl` per section above, `prerequisite_tests.jl` (what the two pinned dependencies must still provide; added in step 0), `type_tests.jl`, `threading_tests.jl`, `device_tests.jl`, the standalone `thread_workload.jl`, and `evolution_cases.jl` — a *helper*, the runs the convergence and noise studies are made of, which lives in `test/` because what it wraps is the integrator loop and `driver.jl` is step 5's (added in step 3, after TreeAMR's `test/wave.jl`). `pointwise.jl`'s tests are **two** files over a shared `pointwise_backgrounds.jl` — `pointwise_tests.jl` for the algebra as a function of the state, `pointwise_identity_tests.jl` for the two identities that need derivatives of it — because between them they compile the metric library's nested dual passes for six backgrounds at two precisions (amended in step 1). The interface-order table has a file of its own, `interface_tests.jl`, rather than a testset in `convergence_tests.jl` **(proposed in step 4)**: it is fifteen evolutions on a mesh where the ghost fill costs four times what it costs on a uniform one, and separating it keeps the cheap order study cheap. Step 5 adds `interior_tests.jl` (the profiles, the core rule, the masks, the radius assertions, and one right-hand-side evaluation on a mesh), `driver_tests.jl` (the runs), the hole fixture in `evolution_cases.jl`, and the **standalone** `test/hole_runs.jl` — the `t = 50 M` runs, `q = 4`, and the two harmonic charts, which are minutes rather than seconds and are run by hand with their numbers recorded here **(proposed in step 5**, following `PLAN.md`'s instruction to put what cannot fit a test file in a script under `test/`**)**. Step 8b adds `bounds_tests.jl` (the projection on synthetic states and on every background, its `Float32` row, planted failures on the fixture's mesh, and the bitwise control) and `hole_runs.jl`'s `bounds` section. Step 8c adds the E3 target `CurvatureTarget` to `evolution_cases.jl`, its claims to `interior_tests.jl` and `driver_tests.jl`, and `hole_runs.jl`'s `calibration` section. Step 8d adds `tracking_tests.jl` — the real harmonics against `AbstractSphericalHarmonics`, the seed against the charts' quartic, the depth of an oblate spheroid, the footprint guard on a non-spherical surface, the e-folds against `dispersion.jl`, the bit-identity of a fitted sphere with step 5's layer, one find of the fixture, and the tracked runs — and makes `evolution_cases.jl`'s shell norms the interior's own `shell_mask`. Step 8e adds `fit_tests.jl` — the fit's harmonics against `shape_series` and `ash_evaluate`, the whole ansatz recovered by `solve_fit`, the static hole's fit against its truncation and the interpolation order, the validity sweep on three holes and the `g_ab` mean control, the evaluator against the model at `Float64` and `Float32`, and the fit of the tracked run's final state; and, for 8e-ii, the `:fitted` right-hand side against `:damped`'s bit for bit, the fixture's `:fitted` run to `0.15 M` and one chunk of it at `Float32` — and moves `fitted_fixture` into `evolution_cases.jl` beside `tracked_fixture_run`, the one tracked run `tracking_tests.jl` and `fit_tests.jl` share; `hole_runs.jl` gains a `fitted` section (the fixture's initial-data study to `1 M`, the boosted seed, harmonic `a = 9/10`). Steps X1 and X2b (planned 2026-10-05) add the standalone `excision_model.jl` — the outflow margins and the one- and two-dimensional closure models — and `excision_tests.jl` |
| `bin/` | `gh.jl` (the CLI, after GHSO2's `gh3d.jl`), viewers, `benchmark.jl`, `backend.jl`, own `Project.toml` |

Dependencies: `TreeAMR` and `SpacetimeMetrics` (both unregistered, both
pinned to GitHub `main` by `[sources]`, which puts the Julia floor at
1.11 as in the siblings), `KernelAbstractions`, `StaticArrays`
(kernel-safe, and what `SpacetimeMetrics` speaks),
`IMEXRungeKutta` (unregistered, pinned to GitHub `main` by `[sources]`; it
replaced `OrdinaryDiffEqLowOrderRK` and `SciMLBase` on 2026-09-26, and
neither is a dependency of the package or of its tests since), `LinearAlgebra` (`det`, `dot`
and `tr` on `StaticArrays`, which the pointwise algebra uses; a standard
library, added in step 1 and not listed when `PLAN.md` enumerated step 0's
`Project.toml` **(proposed in step 1)**), `HDF5` (from G6; **from
2026-10-01** a hard dependency for the checkpoints, whose writer is
TreeAMR's HDF5 extension),
`ApparentHorizonFinder` and `KorzynskiSpin` (from G4, added in step 7,
both pinned to GitHub `main` by `[sources]` like the first two and both
listed in `test/Project.toml` as well, because `prerequisite_tests.jl`
holds them to the same standard: a moving `main` that dropped a name must
fail at the top of the suite). **`KorzynskiSpin` is pinned over `ssh`,
because its GitHub repository is private** — an anonymous `https` clone
gets a 404, so `git@github.com:eschnett/KorzynskiSpin.jl.git` is the only
URL that resolves it: **(resolved 2026-09-21 — the repository is public.
The pin is the plain `https` URL of the others, the deploy key, the
`ssh-agent` step and `JULIA_PKG_USE_CLI_GIT` are gone from `CI.yml`, and
an anonymous clean checkout and a fork's pull request both resolve it, so
the clean-checkout check proves what it claims for the first time.)**
`TreeAMR` left `[sources]` the same day: it is registered, and `[compat]`
selects **0.1.1**. The remaining three entries are what keeps the Julia
floor at 1.11, and two of them are necessary rather than chosen —
`KorzynskiSpin` is not in General at all, and `ApparentHorizonFinder`
`2.1` is not released there (General has `2.0.0`). Vendoring a package or
adding a local-path source would hide that instead of stating it.
**`AbstractSphericalHarmonics` is a direct dependency from step 8d** (`1.2`,
registered; it was already in the manifest through `ApparentHorizonFinder`):
the tracked geometry converts the finder's coefficients with its
`ash_resample` and samples the seed on its `EquiangularGrid`, and a package
this one calls is one it names, not one it reaches through another's
namespace. The tests list it too, for `sYlm` and `ash_evaluate`. Tests add
`MultiFloats` and `ForwardDiff` — the latter because the checks on the
expanded form differentiate the analytic solution one layer above the one
`SpacetimeMetrics` takes internally (added in step 1). `bin/` adds
`CairoMakie` and `SixelTerm` in its own environment. The compat bounds
follow TreeWave's, including
`OrdinaryDiffEqLowOrderRK = "2.2.5"` **(proposed in step 0**, which is
the one bound `PLAN.md` left unstated**)**. The two pins resolved in step 0
to TreeAMR v0.1.0 and SpacetimeMetrics v1.6.0; a `[compat]` entry on a
`[sources]` dependency is a floor on what `main` may become, not a
selection, which is what `prerequisite_tests.jl` exists to notice. The
two added in step 7 resolved to `ApparentHorizonFinder` v2.1.0 and
`KorzynskiSpin` v1.1.0.

## Milestones

Each has an acceptance test; serial `Float64` correctness first. Every
test is three-dimensional and therefore small: `N = 8`, a few roots, one
refinement level, short times.

- **G0 — Scaffolding.** *(Done.)* `Project.toml` with the pins, CI on
  1.11 and release at one and four threads, `CLAUDE.md`, this document,
  `notes/`; `precision.jl`, `device.jl`; a prerequisite test that the
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
  backgrounds](#initial-data-and-backgrounds) is about.
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
  results](#measured-results). Step 2 found that the acceptance was one
  claim short of the design: "exact to degree `q`" is the *first*
  derivative's statement, and the compact second derivative is exact to
  degree `q + 1` by the symmetry of an even `q`; both are asserted, with
  their "and not one degree higher" halves, in `Rational` (amended in step
  2). And the one thing the step found that the design did not say:
  `‖h‖ ~ 1e−13` is a claim about the *offsets*
  `g^{ab} − η^{ab}`, `det g + 1`, `det γ − 1` — `α`, `β^i` and `√γ` are
  built from them and are accurate to a relative `eps` **as values**,
  which is not the same statement and is the one a test can make about
  what `metric_quantities` returns.
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
  numbers are under [Measured results](#measured-results).
- **G3 — Coarse-fine faces, static mesh.** *(Done.)* TreeAMR's two-level
  mesh; `constraints.jl`. *Accept:* the interface-order table (predicted
  rates 3 and 4 at `p = 4, 6` for `q = 4`, control at 4), independent of
  the restriction order and of `ε_KO`; both constraint monitors converge on
  the gauge wave across the interface; the thread-workload digests
  identical at one and four threads; G2 and G3 on `CPU()` in `Float32`.
  All four hold; the numbers are under [The interface-order
  rule](#the-interface-order-rule-and-what-it-costs-a-second-order-system),
  [Analysis quantities](#analysis-quantities) and [Precision, threads,
  devices](#precision-threads-devices), and the suite's cost is under
  [Measured results](#measured-results). Two sentences of the acceptance
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
  amending where they are stated rather than only here. **The calibration
  is not portable between boxes the way the thresholds are**: the
  thresholds are a property of the solution and the spacing, but *which*
  hierarchy they produce is a property of the box and the root brick as
  well, so the reference configuration is recorded with them. And **the
  indicator's mesh and the hand-built hierarchy of step 5 are not the same
  mesh at the same cost**: see the comparison under [Measured
  results](#measured-results). Two sentences of the acceptance needed amending where they are
  stated rather than only here. **"In a Dirichlet box of half-width
  `≥ 20 M`" is a statement about a production run and not about a test**:
  the two radius requirements need `r_h,min ≥ (m + 2G + 2)·h`, so a hole
  costs about `(m + 2G + 2)³` finest-level points *per `r_h,min` cubed*
  whatever the box is, and a box of `20 M` with a uniform mesh at that
  spacing is `10⁸` points. The boundary is exact, so a small box costs
  accuracy and not validity; the suite runs `5/2 M` and records it.
  And **the three variants are separated by the interior residual and by
  what survives to `t = 50 M`, not by the constraints outside `r_1`** —
  see the numbers; the prediction that `:damped` is the smaller of the two
  violations holds against `:pasted` and is a 3 % effect, while the
  residual separates `:frozen` from the other two by more than an order of
  magnitude and the long run separates all three: only `:damped` reaches
  `50 M`. A third sentence needed amending, in
  [the interior](#the-interior-a-pointwise-damping-layer) where it is
  stated: **`a = 9/10` in the harmonic chart is refused**, because the
  chart's singular disk has coordinate radius `a` and the horizon's
  smallest coordinate radius is `√(M² − a²)`, so no ball contains the one
  and fits inside the other. That is G5's case, and it is the open
  question step 5 leaves.
- **G5 — A hole that moves.** Boosted (`|v| ≈ 0.3`), spinning
  Kerr in harmonic coordinates crossing the box — **at `a = 7/10`
  (decided 2026-09-23**; `a = 9/10` waits with its price written down,
  [the interior's
  questions](SINGULARITY_HANDLING.md#the-interiors-questions-opened-in-step-5-and-closed-through-step-8)**)**,
  on **the tracked geometry with
  the `:fitted` target** (amended in step 8f: the analytic core cuts
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
  data where the chart has no analytic interior (amended in step 8: "on the
  analytic data of the same geometry where the chart allows it" became the
  cycle on the case's own data, which agrees with it where it allows it);
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
- **G6 — Infrastructure and the H200.** `io.jl`, slice output and
  viewers, `bin/gh.jl`, the per-phase benchmark on threads (TreeWave's
  table, on Symmetry) and **on the H200 in `Float64`**, in-kernel metric
  evaluation on the device, `q = 4, 6, 8` cost, `N = 16` against `32`;
  the three black-box attempts at the RHS kernel — the stencil/algebra
  split, `Float32`, the workgroup shape — each with the kernel's
  registers per thread, spill bytes, achieved occupancy and picoseconds
  per point against the roofline. *Accept:* the G5 run on the H200 in
  `Float64` reaching the same mesh, horizon and analysis record as the
  host; the tables recorded here, the kernel numbers beside GHAccel's,
  and the defaults chosen from them; the same in `Float32` on the H200
  or on Metal *if it runs*, recorded either way. What G6 does *not*
  promise is a fast GPU kernel: it measures, and hands the measurement
  to the research project under
  [Possible extensions](#possible-extensions).

## Measured results

This section takes the numbers as the milestones produce them; each also
sits beside the prediction it confirms or corrects, in the section it
belongs to. **The black hole's interior is the exception (moved
2026-10-08)**: step 5's layer, steps 8a–8′ and the single holes on the
octant are in [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md), and what
they recommend for a single hole is [the table
below](#single-black-holes-recommended-settings-added-2026-10-08). So are the
excision round's steps X1–X7 (moved when `main` was merged into it, the same
day; [the stub below](#excision-steps-x1x7)).

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
The two backgrounds of `prerequisite_tests.jl` compile into a
KernelAbstractions kernel on `CPU()` and fill a 20-variable, `G = 3`,
vertex-centered field set over a two-level forest with **bit-for-bit**
the values of a host loop through `coordinates`, at `Float64` and at
`Float32` — see [Initial data and
backgrounds](#initial-data-and-backgrounds). No physics is measured yet:
`src/` holds the module shell, `precision.jl` and `device.jl`.

**G1a (step 1), the pointwise algebra.** The suite is 1007 assertions in
124 s at one thread and 116 s at four, on the development machine
(Apple silicon, 12 CPU threads, Julia 1.13.0), up from step 0's 82 in
7.7 s. Essentially all of the growth is **compilation**: the six
backgrounds of the table, at `Float64` and `Float32`, under
`SpacetimeMetrics`' nested forward-mode passes — `dmetric` once,
`gauge_source_grad` twice, and once more for the pass the tests take on
top of them. Evaluation is microseconds. Three measurements shaped the
test files and are recorded so that a later change does not undo them:
fusing the three separate dual passes the tests needed (`∂_iΠ`,
`∂_i∂_j h`, `∂_iF^i`) into **one** jacobian cut the suite from 156 s to
124 s; writing the analytic flux out rather than reading it off
`gh_node_rhs` — whose reduced source is four rank-three contractions that
the dual layers multiply — cut the two identity testsets from 253 s to
28 s; and asking the proof-of-concept case rather than Kerr-Schild for
the second-derivative packing check cost twenty seconds. What is left is
`pointwise_tests.jl` at about 77 s and `pointwise_identity_tests.jl` at
about 19 s, so the first is over `PLAN.md`'s rule of thumb of 30 s per
file and is recorded as such: splitting it further moves the compilation
rather than removing it, and the remedy, if one is wanted, is fewer
backgrounds or fewer precisions, which is a narrower claim.

The physics numbers are in [The equations](#the-equations) beside the
statements they confirm. Two further results:

- **The offset identities.** At `‖h‖ = 1e−13` in `Float64` the offset
  `g^{ab} − η^{ab}` from GHSO2's identity has a relative error of
  **2.4e−16**, one `eps`; computed as `inv(g) − η` it is **1.1e−3**. At
  `Float32` and `‖h‖ = 1e−5`, **1.4e−7** against **4.2e−3**. The identity
  buys about `eps/‖h‖`, which is the whole accuracy of the wave zone.
- **Bit-identity is not a property of an expression, nor even of a
  function.** Two textually identical copies of the source term, one
  inlined inside `gh_node_rhs` and one in `gh_node_source`, differ on two
  of twenty-four cases at `Float64`: the compiler contracts a multiply and
  an add into a fused multiply-add in one context and not in the other,
  and the answers differ in the last place. Stronger, and measured when
  the two methods of `metric_derivatives` were added: **one** body,
  reached through its own wrapper and directly, differs on 3 of 12 points
  at `Float64` and 6 of 12 at `Float32` — by at most `0.03 eps` and
  `0.19 eps` of `max_i ‖∂_i h‖`. Two call sites are enough; the code need
  not even be written twice. This costs nothing here, because the tests
  compare to roundoff against the scale that produced the number, but it
  says what the bit-identity the threading test of step 4 asserts does and
  does not mean: the *same* compiled code, at the same call site, on a
  different thread count — and nothing more.

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

The weights reproduce the textbook central-difference tables entry for
entry at `q = 2, 4, 6, 8`, for `∂` and for the compact `∂∂`, and the
Kreiss–Oliger binomial rows at `r = 2, 3, 4, 5`, **as exact rationals**.
Rounded into `Float64` and into `Float32` every entry is **bit-identical**
to the correctly rounded conversion of the exact rational — which is what
"built in exact arithmetic and rounded once" means when it is stated as
something a test can fail — and at `Float32x2` it is equal on every weight
this package uses, asserted to within an ulp for the reason below.

Exactness, asserted in `Rational` at an off-grid center and a spacing that
is not a power of two: `∂` is exact on polynomials of degree `≤ q` and not
`q + 1`; `∂∂` on degree `≤ q + 1` and not `q + 2`; the tensor-product
`∂_x∂_y` on degree `≤ q` in each variable separately and not `q + 1` in
either; `Q_d` annihilates degree `< 2r` and not `2r`. The dissipation's
grid-scale eigenvalue is exactly `−1`, its center weight is negative at
every `r`, and on a periodic grid `⟨u, Q_d u⟩ < 0` on random data — the
damping sign, three ways.

Observed order on `exp(sin x)` at `x = 0.37` in `Float64`, measured over
the finest pair of spacings at which the truncation error is still a
thousand times the roundoff floor `~eps/h^m` (choosing the window by that
rule rather than by hand is what makes these reproduce):

| `q` | `∂` | `∂∂` | `∂_x∂_y` | `Q_d`, nominal `2r−1` |
|---|---|---|---|---|
| 2 | 2.00 | 2.00 | 2.00 | 3.00 (3) |
| 4 | 4.00 | 4.00 | 3.98 | 5.00 (5) |
| 6 | 5.99 | 6.21 | 5.95 | 7.38 (7) |
| 8 | 7.95 | 7.87 | 7.79 | 8.75 (9) |

The deviations at `q = 6` and `8` are the window and not the stencil, and
they are worth knowing before G2 and G3 measure convergence rates on this
mesh: at `m = 2` the contraction's own floating-point error reaches the
truncation error by `h = 1/32` at `q = 8`, so a fourth-order operator has
about three clean decades of resolution to converge in and an eighth-order
one has about one. A rate measured on too fine a grid measures `eps/h^m`.

**A `@generated` method may not convert to the caller's type.** The first
implementation built the weights and converted them *inside* the generator,
emitting `T` literals. It passed at `Float64` and `Float32` and threw at
`Float32x2`: `MethodError: MultiFloat{Float32,2}(::BigFloat) … The
applicable method may be too new: running in world age 39155, while current
world is 39162`. A generator may only call methods that existed when the
generated function was defined, and this package is precompiled long before
a driver loads MultiFloats — so the failure appears only when
`TreeGeneralizedHarmonic` is loaded *first*, which is what `runtests.jl` and
every driver do, and not when a test file happens to load MultiFloats above
it. What the method emits instead is each weight's exact numerator and
denominator as `Int` literals with one division between them, so the
conversion happens at the call site in the caller's world. The emitted code
still holds no rational and no `BigInt`, and at `Float64` and `Float32` the
division folds away entirely: the LLVM for `apply_stencil(derivative_weights
(Float64, Val(4), Val(1)), u, i)` contains no `fdiv` and no `sitofp`. The
cost is that "rounded once" is now the type's own division: correctly
rounded, and therefore bit-identical to converting the exact rational, at
every IEEE type; equal on every weight this package uses at `Float32x2`;
and one ulp away on 3 of 11 sampled ratios at `Float64x2`, whose division is
not correctly rounded. This is the trap the type-genericity rule exists to
catch, and it is why `Float32x2` is in the suite.

One thing the step found that is not about the stencils: the *tests* need
`Rational{BigInt}` and not `Rational{Int}`. A monomial of degree `q + 2` at
an off-grid rational point overflows a 64-bit denominator at `q = 6`
already. `Rational` arithmetic is checked, so this arrived as an
`OverflowError` in the test rather than as a wrong answer — TreeAMR's
reason for `BigInt` in its own weight construction, seen from the other
side.

**G2 (step 3), the right-hand side on a uniform mesh.** The suite is 1877
assertions in **3m48** at one thread and **3m21** at four, on the
development machine (Apple silicon, 12 CPU threads, Julia 1.13.0), up from
step 2's 1682 in 128 s. A clean archive with no `Manifest.toml` resolves
the two pins from GitHub `main`, instantiates and passes with the same
1877. The growth is the physics: `evolution_tests.jl` is **56 s** and
`convergence_tests.jl` **40 s** of it, both over `PLAN.md`'s 30 s rule of
thumb and recorded as such — a row of either costs a kernel specialisation
to compile, and the evolutions themselves are seconds.

**The order of the scheme.** The gauge wave (`A = 1/20`, one wavelength
cubed, periodic, `ε_KO = 0`, `γ0 = 1`), evolved an eighth of a crossing at
`cfl = 1/4` on roots `1, 2, 3`, and shifted Minkowski (`A = 1/2`, `w = 2`)
a quarter of a crossing on roots `2, 3, 4` through a Dirichlet face:

| case | `q` | `N` | `h` | L2 rate | L∞ rate |
|---|---|---|---|---|---|
| gauge wave | 2 | 8 | 1/8 … 1/24 | **1.99** | **1.94** |
| gauge wave | 4 | 8 | 1/8 … 1/24 | **3.95** | **3.92** |
| gauge wave | 6 | 10 | 1/10 … 1/30 | **5.92** | **5.92** |
| gauge wave, `ε_KO = 0.5` | 4 | 8 | 1/8 … 1/24 | **4.12** | — |
| shifted Minkowski, Dirichlet in `x` | 4 | 8 | 1/4 … 1/8 | **3.93** | **3.90** |

The errors at `q = 4` are `2.0e−4, 1.3e−5, 2.6e−6` in L2, and at `q = 6`
`5.3e−6, 8.8e−8, 7.9e−9` — four orders above the roundoff floor at the
finest, which is what makes the rate a measurement of truncation error and
not of `eps/h²` (step 2's warning about the window). Shifted Minkowski's
rate is **3.6** at the profile width `w = 1` and **3.93** at `w = 2`: at
`h = L/16` the sech² profile is not resolved, and the shortfall is the
case's parameters and not the scheme — which is worth knowing before a
hole, whose features are sharper still, is put on a mesh this coarse.

**Stationarity.** Minkowski's `du` is **exactly** zero — every component,
at `q = 2, 4, 6`, with and without dissipation, at any `t` — and a run of
it through RK4 returns the initial state with an error of exactly zero.
Shifted Minkowski is stationary only to truncation, for the reason under
G2 above.

**Robust stability** (`N = 8`, one root block, `q = 4`, `γ0 = 1`,
`γ2 = 0`, white noise of amplitude `1e−8` on all 20 variables): over a
thousand steps — eighteen crossings of the box — the L2 norm falls to
**0.66** of its initial value at `ε_KO = 0.5` and **grows by 9.2** at
`ε_KO = 0`; in L∞, **×2.10** against **×30.5**. The L∞ ratio at
`ε_KO = 0.5` is a transient rather than a rate: 3.10, 2.53, 2.10, 1.45 at
250, 500, 1000, 2000 steps, so the norm turns over and falls. This is
GHSO2's `ε_KO ≈ 0.5` confirmed on a finite-difference mesh, at the one
configuration where the answer is not confounded by a hole.

**The kernel against the reference.** `gh_rhs_kernel!` and
`gh_node_rhs_expanded`, evaluated on the same working array at sampled
points of every block, agree to **under 1e−12** of the size of the terms
that build `∂ₜΠ` — on the gauge wave, on shifted Minkowski (with its
sampled gauge source) and on harmonic Kerr in a box off centre, at
`q = 2, 4, 6`, with `ε_KO = 0` and `0.5`. Not bit for bit, and not asked
to be: two call sites of one body are contracted differently (step 1's
finding, now seen from the mesh). Two evaluations at the same `(u, t)`
*are* bit-identical, and `u` is untouched by either.

**What an evaluation costs** is the table under [One right-hand-side
evaluation](#one-right-hand-side-evaluation), together with the linear
index that bought a quarter of it. Two further numbers from the same
measurement: the pointwise algebra — `metric_quantities`,
`metric_derivatives`, `gh_node_source` — is **505 ns** per point of the
`q = 4` kernel's 1409, so the stencils are the larger half at every order
this package uses, and the split into a stencil kernel and an algebra
kernel that G6 will try is a split of roughly 3:1 rather than 1:1.

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

**The interface-order rule**, the table's own numbers and what a
coarse-fine face costs in time, are under [The interface-order
rule](#the-interface-order-rule-and-what-it-costs-a-second-order-system).
Two results there are worth repeating because they are *not* the
prediction:

- **The restriction order is not a degree of freedom.** On a
  vertex-centered mesh restriction is injection, so orders 2 and 4 are the
  same computation: `l2 === l2`, asserted as identity and not as a
  tolerance, at both prolongation orders. The same holds of the unrefined
  control's two operator rows, for the stronger reason that it never
  prolongates.
- **A coarse-fine face costs more in time than in order.** At `p = 6` the
  ghost fill is 79 % of a right-hand-side evaluation against 22 % on a
  uniform mesh, and an evaluation is 3.7 times as expensive per point.
  That is TreeAMR's cost, it is exactly the `6³ = 216` coarse points per
  fine ghost point the section predicts, and it is what makes
  `interface_tests.jl` the suite's most expensive file.

**The constraint monitors** are under [Analysis
quantities](#analysis-quantities), with the five decisions the writing
settled. The sharpest of their numbers is the one with no mesh in it: on
the six backgrounds of the table, fed the *analytic* second derivatives
`ddmetric` returns, `ℋ` and `ℳ_i` vanish to **1.2 eps** and **0.2 eps**
of the size of the curvature terms — which is the only test in the package
that would catch a sign error in `∂_cΓ^c_ab − ∂_bΓ^c_ca`, since on a mesh
such an error leaves a violation that converges at order `q` to something
nonzero.

**Threads and precision** are under [Precision, threads,
devices](#precision-threads-devices): six digest lines identical character
for character at one and four threads, and `Float32` reproducing the gauge
wave's rate and its error at the two coarsest resolutions.

**G4a (step 5), the interior and the driver** — the suite, the layer's order,
the three variants to `50 M`, the other charts, the gauge drift, `Float32` and
the record — moved to
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#step-5-the-static-holes-interior-and-the-driver-g4a)
(2026-10-08).

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

`refinement_tests.jl` stays inside `PLAN.md`'s 30 s rule of thumb, and what
buys that is the fixture's `maxlevel_cap = 1`: at the calibrated cap the
same case is 848 blocks and belongs to `test/hole_runs.jl`, which is where
it is.

**The reference configuration.** Everything below is Kerr-Schild `a = 0`,
`q = 2`, `N = 8`, in a Dirichlet box of half-width **`5 M`** on a `4³` root
brick, with `r_0 = 3/10`, `r_1 = 5/4` and the interior margin `m = 4`. It
is not step 5's fixture and could not be: with the horizon at `r = 2 M` and
a box of half-width `5/2 M`, the shell the level floor must refine and the
margin the level ceiling must keep coarse overlap, and the refinement
refuses that configuration by name. The margin `m = 4` (still above the
floor `G + 1 = 3`) is what makes the interior's radius requirements need
**one** refinement level rather than two, which is what keeps the
suite's adaptive run at the step-5 fixture's 120 blocks and 61 440 points.

**Calibration, table one: `τ_max` against `h`** on uniform meshes over the
static hole's initial data, masked inside `r_1` (`test/hole_runs.jl
indicator`):

| roots | `h` | blocks | points | `τ_max` |
|---|---|---|---|---|
| 2 | 0.625 | 8 | 4 096 | **0.9343** |
| 4 | 0.3125 | 64 | 32 768 | **0.8130** |
| 8 | 0.15625 | 512 | 262 144 | **0.5348** |
| 16 | 0.078125 | 4 096 | 2 097 152 | **0.2257** |

`τ` falls with `h` — which is the property that makes a fixed threshold
terminate refinement — and it falls faster than linearly once the global
floor rather than the first differences dominates the denominator: the
numerator is `O(h²)` and the denominator crosses over from `2|f′|h` to
`4ε·U_ref`. `U_ref` is **1.600** at every resolution (`2M/r₁` at the
layer's edge, which is where the evolved region's largest `|h|` is); the
per-component maxima are `1.600` for the seven components that carry the
`1/r` fall-off and `0.572`–`0.783` for `h_xy`, `h_xz`, `h_yz`, which is
what a per-component floor would have used and why the shared amplitude is
the conservative choice rather than a lossy one.

**Calibration, table two: the depth the initial-data cycle reaches**, with
the cap at 3 so that the *indicator* is what stops it, and
`coarsen_tol = refine_tol/4`:

| `refine_tol` | passes | depth | leaves | points | `h` |
|---|---|---|---|---|---|
| 0.80 | 2 | 1 | 120 | 61 440 | 0.15625 |
| 0.60 | 2 | 1 | 120 | 61 440 | 0.15625 |
| 0.50 | 4 | 2 | 848 | 434 176 | 0.078125 |
| **0.40** | 4 | **2** | **848** | 434 176 | 0.078125 |
| 0.30 | 4 | 2 | 862 | 441 344 | 0.078125 |
| 0.20 | 6 | 3 | 1800 | 921 600 | 0.0390625 |

The plateau is `[0.30, 0.50]`, so the defaults are **`refine_tol = 0.4`,
`coarsen_tol = 0.1`** — mid-plateau, with the cap never binding and the
cycle converging in four passes. The hierarchy it chooses has its finest
spacing at `5/64`, which is the step-5 fixture's, and the
`refine_tol = 0.2` row shows what asking for one more level costs: 2.1×
the points for a hole that is already resolved.

**The level floor does not bind at these thresholds, and does when they are
loosened** (`test/refinement_tests.jl`, both halves on one flagging pass):
the marks computed with the floor and without it are *identical* at
`refine_tol = 0.4`, and with it raised to `0.99` — above anything this data
reaches — the floor is the only thing that refines, on exactly the blocks
whose extent meets the shell `r_1 ≤ r ≤ r_h,max`. That is `CODE.md`'s
prediction confirmed. The derived floor level here is **1**, and the
indicator asks for 2.

**The ceiling holds, and what it took.** On the converged hierarchy every
block within one coarse cell of the outer boundary is at level 0. It was
**not** so before the travelling margin was moved inside the criterion:
with TreeAMR dilating the boxes after the fact, 12 of the 64 boundary root
blocks were refined through the margin, since on a `4³` brick the
boundary blocks *are* the refined region's neighbours.

**The adaptive run against a frozen hierarchy**, both at `refine_tol = 0.4`
to `t = 1/2 M` with `chunk = 1/20`, the same case, the same finest spacing
`5/64`, at the grid rate `1/dt`, the default until 2026-09-23:

| mesh | leaves | points | `err_l2` | `err_linf` | `gauge_l2` |
|---|---|---|---|---|---|
| the indicator's | 848 | 434 176 | **1.088e−3** | **3.261e−2** | **4.403e−4** |
| frozen (`hole_forest`, two shells) | 1128 | 577 536 | 1.060e−3 | 3.261e−2 | 3.963e−4 |

The two agree to **2.7 %** in the masked L2 and to all four digits in L∞,
and the indicator's mesh needs **75 %** of the points — the hand-built
shells are radii applied to whole blocks, and the indicator is what decides
per block. The interior residual is identical (`1.171e−1`), which it should
be: it is measured inside `r_1`, where both meshes are at the floor's level.

**Regrids on a static hole change nothing.** Over ten chunks of the
adaptive run and three of the suite's, `nregrids = 0`: the hierarchy the
cycle converged to is a fixed point of the criterion, which is
`CODE.md`'s prediction and the sharpest statement available about an
indicator whose input is time-independent. `τ_max` moves by less than
`1 %` across the run (`0.2257` at `t = 0`, `0.2241` at `t = 1/2`), which is
the truncation error growing, not the mesh.

**And a regrid that *does* move the mesh rebuilds behind itself** — which a
static hole never exercises, and which is the branch G5 lives in. The suite
starts one run from the indicator's fixed point plus a single refined block
in the far corner of the box: the criterion does not want it, the ceiling
caps it at the coarsest level, and no firing block is a neighbour of it, so
the travelling margin does not hold it either. The first regrid coarsens it
away — **127 leaves to 120** — and the driver rebuilds the schedule, the
problem (re-sampling the gauge source and re-asserting the interior's two
radius requirements on the new mesh) and the state vector; the transferred
state is still a metric and its masked error after two chunks is
`1.36e−3`, against `1.93e−3` after three chunks on the mesh that did not
move — at the grid rate; `1.34e−3` and `1.88e−3` at the default `4/M`
**(measured in step 8c′)**, where the interior residual of the run that did
not move is `6.0` against `0.80` at `t = 3/20 M`: this fixture's layer is
six cells of `h = 5/32` down to `r_0 = 3/10`, where `|Π|` is `131`, and it
saturates at `14.7` by `1 M` while the masked error stays below the grid
rate's (`6.03e−3` against `7.13e−3` at `2 M`, measured once, not in the
suite).

**The refinement centroid is biased by one to two finest spacings**, and
the bias is the mesh's indexing rather than the indicator's: `1.7`–`2.4`
finest spacings on a configuration where the mesh and the solution are both
symmetric about the hole. See [Analysis
quantities](#analysis-quantities) for why, and read G5's "within a few
finest spacings" against it.

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
Two further pins are resolved, `ApparentHorizonFinder` v2.1.0 and
`KorzynskiSpin` v1.1.0, and `prerequisite_tests.jl` holds both to the same
standard as the first two — including a find of Kerr's horizon on
*analytic* Cauchy data at `a = 9/10`, which recovers the area to `1e−8`
and `J = M a` to `1e−6` with the axis along `ẑ`, and is the baseline
everything below is measured against.

**That acceptance item is no longer blocked** (2026-09-21), and is
recorded under [File layout](#file-layout): `KorzynskiSpin`'s repository
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
  makes the floor; one of the two is wrong.

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

**`Float32`.** The same find on a `Float32` field set gives the `Float64`
answer to **seven digits** — `r_mean` `2.0005742` against `2.0005741`, area
`50.294445` against `50.294440`, `M_irr` and `M_ch` to `5e−8` — because the
interpolation is a weighted sum of twenty numbers of order one and the
finder itself is `Float64` whatever the field set is. It converges in 26
iterations rather than 51: the fast flow's stall detector reaches
`Float32`'s floor sooner, and `H_norm` bottoms out at `4.4e−6` rather than
`1.2e−14`.

**The interpolator.** Order `q + 2` over the containing block's stored
points reproduces a polynomial of degree `q + 1` to `1e−12` at every query
point of a two-level mesh — values and gradients, on the fine side of a
coarse-fine face as well, since a prolongation of order `p = q + 2` is
itself exact on that degree — and a polynomial of degree `q + 2` not at
all. On Kerr-Schild's *analytic* metric sampled onto the step-5 fixture at
`N = 6` and `N = 12` (`h = 5/48` and `5/96`), queried on the sphere
`r = 1.9 M`, the worst component of `h` converges at **4.19** and its
gradient at **2.81**, against `q + 2 = 4` and `q + 1 = 3`. That one order
between them is why `K_ij` is one order behind `γ_ij` in the ADM data, as
`notes/methods-ghso2.md` measured on spectral elements, and it is what
sets the accuracy of everything below.

**Kerr's numbers from sampled data** (`test/hole_runs.jl horizon`, a
displaced guess at `(0.1, −0.05, 0.075)` and a seed sphere outside the
horizon; `q = 2`, the layer in place, the guard on):

| background | leaves | `h` | `N_ah` | `r_min` (exact) | `r_max` (exact) | `A` rel. err. | `M_irr` (exact) | `J` (exact) | `M_ch` | axis | offset |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `KerrSchild(1, 0)` | 120 | 5/64 | 16 | 1.999903 (2) | 2.000002 (2) | 4.42e−5 | 0.999978 (1) | 2e−6 (0) | 0.999978 | `−ẑ` | 7.6e−6 |
| `Harmonic(1, 0)` | 960 | 5/128 | 16 | 0.999990 (1) | 1.000014 (1) | 2.66e−6 | 1.000001 (1) | 2e−6 (0) | 1.000001 | `−ẑ` | 7.1e−6 |
| `KerrSchild(1, 9/10)` | 1128 | 5/128 | 20 | 1.436965 (1.435890) | 1.692431 (1.694633) | 1.85e−4 | 0.847238 (0.847316) | **0.899980** (0.9) | 0.999954 | `+ẑ` | 1.2e−6 |

Every entry is at or below the interpolation's own accuracy, which is what
"to interpolation accuracy" in G4's acceptance asks. The spinning row is
the one with content: the horizon is genuinely oblate (`r_min` and `r_max`
differ by 18 %), `J = M a` comes back to `2e−5` relative, the axis is
`ẑ` to four digits, and `M_ch = √(M_irr² + J²/4M_irr²)` recovers `M` to
`5e−5` from an `M_irr` that is `0.847`. **The spin axis is signed and the
sign is not physics**: `±ẑ` are the same axis, and `KorzynskiSpin`
returns whichever the flow's orientation gives — the two `a = 0` rows,
whose `J` is `2e−6`, point along `−ẑ` for no reason at all.

**The harmonic chart needed a finer mesh than the "other charts" table
uses, and it is the mesh `CODE.md` predicted (measured in step 7).** At
`h = 5/64` — step 5's harmonic row — `r_0` has to sit at `0.2 M` and
therefore `r_1` at `0.67 M` against a horizon at `1.0 M`, and the fast
flow's *transient* dips into the layer: the guard refuses the query and
the find fails, which is the guard being right. At `h = 5/128` the same
chart takes `r_0 = 0.3` (where `|h| ≈ 6.7` rather than `29`) and
`r_1 = 0.6`, and the find is the second row above — the best of the three.
That is [the interior](#the-interior-a-pointwise-damping-layer)'s
"`h ≈ M/23` before it has a moderate `r_0`" confirmed from a second
direction.

**The horizon of a run, and what it says about the solution.** Kerr-Schild
`a = 0` on the step-5 fixture (`q = 2`, `N = 8`, `h = 5/64`, `cfl = 1/5`,
the `:damped` layer at the grid rate `1/dt`, the default until 2026-09-23;
step 8c's `4/M` runs end at `M_irr = 0.9972` at `50 M`) to `t = 10 M`, the
horizon found at every chunk:

| `t/M` | 0 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| `r_mean` | 1.999954 | 1.996744 | 1.995522 | 1.994715 | 1.993868 | 1.993114 |
| `A` | 50.263262 | 50.064133 | 50.004753 | 49.968571 | 49.933017 | 49.903602 |
| `M_irr` | 0.999978 | 0.997995 | 0.997403 | 0.997042 | 0.996687 | 0.996394 |
| offset | 7.6e−6 | 7.4e−5 | 1.2e−4 | 2.7e−4 | 4.1e−4 | 5.8e−4 |

The horizon **shrinks by 0.7 % over ten crossings** and its center wanders
by `5.8e−4 M`, which is `0.0075` of a finest spacing — far inside
`CODE.md`'s "a few finest spacings", so the finder agrees with the layer
and there is no bug to report. The shrinkage is the *solution's*
truncation error at `q = 2` and 26 points per `M`, the same drift the
masked error and the gauge drift already show; `J` stays at `1e−6` and
`M_ch` tracks `M_irr` to the last digit, which is what a Schwarzschild
hole should do.

**What a find costs** (`q = 2`, 120 leaves, four threads; one
right-hand-side evaluation on the same mesh is `38.6 ms`):

| `N_ah` | surface points | iterations | find | find + spin |
|---|---|---|---|---|
| 12 | 276 | 51 | 0.067 s (1.7 RHS) | 0.637 s (16.5 RHS) |
| 16 | 496 | 54 | 0.106 s (2.7 RHS) | 1.739 s (45 RHS) |
| 20 | 780 | 51 | 0.140 s (3.6 RHS) | 4.037 s (105 RHS) |

**The interpolation is not what a find costs**: one batch of 496 points
— the location, the guard and `(q+2)³ × 20` loads with the gradient — is
**0.26 ms**, `0.5 µs` per point, and a find is fifty of those. The spin is
the cost, it is inside `KorzynskiSpin`'s uniformization and Möbius
algebra, and it grows steeply with the grid (`16×`, `45×`, `105×` a
right-hand side). Two consequences for G5 and G6: the horizon cadence `k`
is a real parameter and not a formality, and `spin = false` is the knob to
reach for before `N_ah` is lowered. The uniformization's residual floor on
interpolated data is **`2.2e−5`** at `h = 5/64`, which is what the
`unif_tol` decision under [Analysis quantities](#analysis-quantities) is
about.

**The port to TreeAMR's interpolation (measured 2026-09-26).** The same
fixture, `q = 2`, 120 leaves at `h = 5/64`, 496 points of
`EquiangularGrid(15)` at `r = 2 M`, value and gradient of all twenty
variables with the guard on, the best of 200 calls, on the development
machine at a load of 3–5; the stopgap measured the same day against the
same TreeAMR 0.1.3:

| | stopgap, 4 threads | M11, 4 threads | stopgap, 1 thread | M11, 1 thread |
|---|---|---|---|---|
| batch (`interpolate_grad` → `gh_interpolate_grad`) | 0.198 ms | 0.139 ms | 0.739 ms | 0.401 ms |
| provider batch (496 `ADMVars`) | 0.235 ms | 0.162 ms | 0.770 ms | 0.422 ms |
| TreeAMR's `interpolate` alone | | 0.107 ms | | 0.369 ms |
| find, `N_ah = 16`, no spin | 0.083 s | 0.073 s | 0.149 s | 0.121 s |
| find with spin | 1.90 s | 1.78 s | 1.87 s | 1.79 s |

A batch is **1.4× cheaper at four threads and 1.8× at one**; the
difference between the wrapper and TreeAMR's call alone (`0.03 ms`) is the
unpacking into `SVector`s and the host-side flag scan. A find is
10–20 % cheaper without the spin and unchanged with it, which is the
paragraph above again: the interpolation was never what a find costs. The
answers are the stopgap's to roundoff — the same stencil, different weight
arithmetic (truncated Taylor products against plain products) and a
sum-factorised contraction: over 1984 points about `r = 2 M` the values
differ by at most `1.2e−15` (of values of order one) and the gradients by
`8.8e−15`; none is bitwise equal. A find from the displaced guess gives the
same area `50.26326183407128` and `M_irr` to every digit, `r_min` and
`r_mean` differing in the sixteenth, and converges in 48 iterations
against 49 — the fast flow's stall detector, not the surface. The
interpolation's rates are unchanged, `4.19` and `2.81` (`horizon_tests.jl`).
**The provider interpolates 16 variables, not 20 (added 2026-09-26, in
review).** `adm_vars_from_state` reads all of `h`, the gradients of `h_ti`
and `h_ij`, and `Π_ij`: `Π_tt` and `Π_ti` enter only the `tt` and `ti`
components of `∂_t g`, which `K_ij` never uses, and no gradient of `Π` is
read at all — 43 of the 80 numbers per point a 20-variable value-and-gradient
call produces. The provider now asks TreeAMR for `ADM_VARS = [1:10; 15:20]`
and fills `Π_tt` and `Π_ti` with `NaN`, which would surface where a zero
would pass for a number if the extraction ever read them. On the batch above
TreeAMR's call alone goes from `0.35` to `0.30 ms` at one thread and from
`0.105` to `0.091 ms` at four; the provider batch from `0.422` to `0.366 ms`
and from `0.162` to `0.144 ms`. The rest of what is unused is `Π_ij`'s
gradients, which TreeAMR cannot skip — `derivs` applies to every variable of
a call — and splitting it into two calls (nine variables with gradients,
seven without) measured `0.335` and `0.108 ms`, no better than twenty in
one, because the second call repeats the location and the weights (`0.070`
and `0.026 ms` for one variable alone). A per-variable `derivs` in TreeAMR
would be worth about another 10 % of the batch, a percent of a find without
the spin; not asked for. `horizon_tests.jl` holds the 16 variables to the
full call bit for bit and the `NaN` slots to the true values bit for bit at
one call site of the extraction; the provider against the 20-variable path
is `2.8e−16` in `K`, the two call sites of `adm_vars_from_state` fusing
differently.
**Not measured: the device path.** `gh_adm_provider` and the fit's sampler
no longer `hostcopy` the state, and TreeAMR's `interpolate` runs on the
field set's backend, but this package has no device test until step 9.

**Step 8a, what crosses the horizon from inside it** — moved to
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#step-8a-what-crosses-the-horizon-from-inside-it)
(2026-10-08).

### What the suite costs, and where (measured 2026-09-19 on Symmetry)

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

**The speed-up separates the two kinds of cost, and so does code
coverage.** Testsets that evolve something parallelise at 2.9–4.2× and cost
2.2–2.8× more when instrumented; testsets that compile a specialisation
parallelise at 1.26–1.5× and cost 1.00–1.05× more instrumented. Four
testsets over 20 s gain less than 1.3× from threads, 13 % of the
four-thread run, and every one of them is a `Val`-parameter specialisation
— the type-generic `pointwise` rows and the right-hand-side kernel rows.
So the arithmetic half shrinks by doing less work and the compilation half
only by *specialising less*; a new `q`, a new element type or a new
interior variant is priced in the second column and nowhere else.

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
extensions](#possible-extensions) already books kernel efficiency as a
research project rather than a milestone; until that is taken up, a
physics test costs what its points and its steps cost.

So the levers, in order of what they return and with what they cost:

1. **Four threads.** Already 1.87× on the whole suite and 3.28× on
   `driver_tests.jl`, for nothing. CI runs three of its five cells serial.
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
devices](#precision-threads-devices) is about exactly this, and
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
layer](#the-interior-a-pointwise-damping-layer) puts it. Its two siblings
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
names, and the collision arrived as predicted — on x86-64 only: Apple
silicon still fuses the two kernels alike, while on an EPYC 7543
(`znver3`, Symmetry and GitHub's Linux runners) 241 914 of 962 780 values
outside `r_1` differ, by at most **120 eps** of the variable's largest
`|du|`. `interior_tests.jl` now bounds that at 512 eps and reports the
ratio; an interior term leaking outside `r_1` would be of order `ρ_max`
times the test's `10⁻³` perturbation. The same judgement relaxed
`moving_tests.jl`'s "the analytic solution outside the offset surface bit
for bit", a kernel's fill against a host `state_tuple`, which differed
only under coverage on Linux: bounded at 64 eps of each variable's largest
value, measured `0` plain and `1.4` eps under `--code-coverage
--check-bounds=yes` on an EPYC 7532, where the interior's ratio is `120`
and `96` eps.**)**

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

**Four threads on four cores is not oversubscribed in any way that costs**
(measured 2026-09-19, and the obvious hypothesis was wrong). Julia 1.13
gives `-t4` **four** GC threads, so a four-thread suite has eight runnable
threads, and a GitHub Linux runner for a public repository has four vCPUs
— which looks like 2× oversubscription and is not. Under a four-CPU
`cpus-per-task` allocation on Symmetry, at four threads:

| GC threads | suite | real | user |
|---|---|---|---|
| 4 (default) | `21m30` | `21m44` | `39m09` |
| 1 (`JULIA_NUM_GC_THREADS=1`) | `22m31` | `22m49` | `37m55` |

Restricting the collector is **5 % slower**, not faster, so the flag is a
dead end and is recorded here so it is not tried again. The number that
explains it is `user/real = 1.80`: with four cores available the suite
keeps **1.8 of them busy**, because so much of it is single-threaded
compilation. There is no contention to relieve — the cores are idle.

The same table calibrates GitHub. Symmetry at eight CPUs and four threads
is `16m41`, at four CPUs `21m30`, so the narrow allocation costs 29 %; and
GitHub's four-thread cell measures `22.7` and `23.2` min, which is the
four-CPU figure. Its runners behave exactly as four-core machines should.
One run of three took `61.6` min against a `43.6` min serial sibling —
that is a bad runner, not a property of the matrix, and three samples with
a 2.7× spread are not enough to act on.


### Steps 8b–8′: the generic interior and the moving hole

Moved to
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#steps-8b8-the-generic-interior-and-the-moving-hole)
(2026-10-08): the range projection (8b), the layer against an inexact target
(8c), the default rate (8c′), the tracked geometry (8d), the fitted target
(8e-i, 8e-ii), the measurement matrix (8f), the moving hole (8) and its
trailing side (8′).

### Robust stability on the octant (measured 2026-10-02)

`test/octant_runs.jl` on Symmetry's H200 (`octant-run`, jobs 567966 and
567984): flat space on `[0, 128]³`, reflecting at the three faces through the
origin and Minkowski Dirichlet data at the outer ones; six levels, `h = 1` at
the boundary halved inside each cube `[0, R]³`, `R = 64, 32, 16, 8, 4`, to
`h = 1/32` (`N = 32`, 344 blocks, 11.27 M points, about 1.8 M per level);
`q = 4`, `ε_KO = 1/2`, `γ0 = 1`, `γ2 = 0`; uniform noise of `10⁻⁸` on every
point and variable, odd components zero on their walls; to `t = 128`. The
norms give every grid point the same weight (`weighting = :points`), so the
six levels count about equally.

**Nothing grows.** Over `[16, 128]` at `cfl = 1/2` the point-weighted L2
rates are `σ = −0.026` (ℋ), `−0.031` (ℳ_i) and `−0.012` (`C_a`), and over
the last quarter `−0.005`, `−0.006` and `−0.0035`: the norms fall from
`2.7·10⁻⁵`, `7.6·10⁻⁶` and `1.5·10⁻⁷` at `t = 0` (the noise's second
differences, largest on the finest level) to `7.7·10⁻¹¹`, `2.1·10⁻¹¹` and
`2.7·10⁻¹¹`, and the state itself from `5.8·10⁻⁹` to `1.2·10⁻⁹`. By level,
ℳ_i decays everywhere (to `3·10⁻¹⁴` on the finest); ℋ and `C_a` on the two
finest levels fall to a minimum at `t ≈ 48` (`5·10⁻¹²`, `6·10⁻¹³` on level 5)
and come back up to a **plateau** — level 5's ℋ `1.51, 1.70, 1.80, 1.83,
1.84·10⁻¹¹` at `t = 88, 100, 112, 120, 128` — a rise that saturates at the
coarse levels' content (level 0's ℋ is `1.8·10⁻¹⁰` and still falling), not an
exponential. The mesh's L∞ is level 0's and wanders between `3` and
`6·10⁻⁹` without a trend.

**`cfl = 1/2` is the same run.** Every norm of the two runs agrees to three
digits at every row both have, to `t = 128` (the largest relative difference
of any norm at any row is `2.3 %`, an L∞ of `ℳ` in the noise's transient at
`t = 8`; at `t = 128` they agree to eight digits), the step
halves the cost — `55 s` of H200 time per unit of `t` against `107 s`, `2 h 4`
for the whole run — and RK4's margins at `cfl = 1/2` are wide (`ω dt ≈ 1.15`
against `2.83` for the wave part at the finest level, `0.43` against `2.78`
for the dissipation). A 64-core EPYC node is `12.6×` slower than one H200 here
(`1351 s` per unit of `t`).

### Single black holes: recommended settings (added 2026-10-08)

Rules of thumb for a single Kerr-Schild hole, `M = 1`, spin `a ≤ 9/10` along
`z`, at rest on the octant, drawn from the runs in
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07):
`a = 0`, `3/5` and `9/10` are measured, the spins between follow from the
rules below the table, and none of it is a guarantee. Common to every row:
`q = 4`, `cfl = 1/2`, `ε_KO = 1/2`, `ρ_max = 4/M`, the Gaussian `γ0`
(`1/M → 1/(10M)`), the algebraic source, the rotating octant (the mirrored
one at `a = 0`), a finest level whose cube contains the whole horizon, and
`24 M` — the shells about the hole saturate by about `8 M`. `h` is the
spacing at the hole; a convergence pair keeps the layer's radii fixed in `M`,
not in cells. The horizon's coordinate radius is `r₊ = 1 + √(1 − a²)` at the
poles and `√(2 r₊)` on the equator, and the ring has radius `a`.

| `a` | variant | coarsest `h` (convergence pair) | singularity settings | basis |
|---|---|---|---|---|
| 0 | `:damped` | `1/16` (`1/16`, `1/24`) | `r_0 = 3/4`, `r_1 = 3/2` | measured: order 4 from `1/16` |
| 0 | `:fitted` | `1/24` (`1/24`, `1/32`) | `m = 16`, `n_L = 20`, `fit_cont = 2`, `lmax_fit = 8` | measured: `ℋ` outside `1.6×` `:damped`'s, the error `1.1×` |
| 0 | `:excised` | `1/16` (`1/16`, `1/24`) | `r_E = 1/2`, `r_0 = 1/4`, `:msn`, no blend | measured (step X3): `ℋ` outside `0.34×`, `0.92×`, `1.00×` `:damped`'s at `1/16`, `1/24`, `1/32` |
| 0.3 | `:damped` | `1/16` (`1/16`, `1/24`) | `r_0 = 3/4`, `r_1 = 3/2` | the rules |
| 0.3 | `:fitted` | `1/24` | `m = 16`, `n_L = 18`, `fit_cont = 2`, `lmax_fit = 12` | the rules (`fit_cont = 2` not yet run with spin) |
| 0.6 | `:damped` | `1/24` (`1/24`, `1/32`) | `r_0 = 3/4`, `r_1 = 3/2` | measured: order `4.0`; `1/16` is not convergent |
| 0.6 | `:fitted` | `1/32` (`1/32`, `1/48`) | `m = 20`, `n_L = 18`, `fit_cont = 1`, `lmax_fit = 12` | measured at `1/32`: `ℋ` outside `3.2×`, the error equal; the pair predicted |
| 0.6 | `:excised` | `1/32` (`1/32`, `1/48`) | `r_E = 4/5`, `r_0 = 7/10`, `:msn`, no blend, X5's rule, the symmetric mixed derivative and the shave (the defaults) | measured (step X7): `ℋ` outside `2.1×`, `1.15×`, `1.00×` `:damped`'s at `1/24`, `1/32`, `1/48`; `J` drifts as the layer's; (step X8) stationary with the shave at `1/24`, no triple corner on a cap |
| 0.7 | `:damped` | `1/32` (`1/32`, `1/48`) | `r_0 = 0.85`, `r_1 = 1.45` | the rules |
| 0.7 | `:fitted` | `1/48` | `m = 18`, `n_L = 24`, `fit_cont = 1`, `lmax_fit = 12` | a guess: screen the depth first |
| 0.8 | `:damped` | `1/32` (`1/32`, `1/48`) | `r_0 = 0.95`, `r_1 = 1.35` | the rules |
| 0.8 | `:fitted` | `1/64` | `m = 16`, `n_L = 30`, `fit_cont = 1`, `lmax_fit = 12` | a guess: screen the depth first |
| 0.9 | `:damped` | `1/48` (`1/48`, `1/72`) | `r_0 = 1`, `r_1 = 5/4` | predicted: `r_0 = 19/20` is outside the convergent regime at `1/48` and clean at `1/96` |
| 0.9 | `:fitted` | `1/96`, no pair in reach of one H200 | `m = 16–18`, `n_L = 36`, `fit_cont = 1`, `lmax_fit = 12` | measured: stable to `32 M`, but `ℋ` outside `1000–2000×` `:damped`'s |

- **The ring rule, for both variants.** On the equator Kerr-Schild's `H`
  varies on the length `L = (R² − a²)/R` at the core surface's radius `R`;
  resolve it with at least 8 cells. The three runs that failed for this
  reason had `4.0–4.7` cells (`a = 0` `:fitted` at `1/16`, `a = 3/5` and
  `a = 9/10` `:damped` at `1/16` and `1/48`), every run that converged or
  held had at least `6.5`. (The `a = 3/5` record reads its `1/16` row as the
  margin's 4.8 cells; at `a = 0` a margin of 4 cost only `2×`, which favours
  the ring.) For `:damped` the rule sets `r_0` and the coarsest `h`; for
  `:fitted`, whose initial data are analytic down to the core surface, it
  bounds `m + n_L` on the equator.
- **`:damped`**: a ramp `r_1 − r_0` of at least 12 cells (18 lowers `ℋ`
  outside about `3×` at `a = 0`), and `r_1` at least 8 cells inside the
  horizon at the poles, `r₊`; more depth than 8–12 cells did not help.
- **`:fitted`**: a margin of 16–20 cells — the leak halves for every 2–3
  more, and at `a = 0` nothing improved past about 18 — and a ramp of at
  least 18. At high spin there is a depth bound besides: at `a = 9/10` the
  offset surface must stay within `0.19 M` of the polar horizon (`0.21 M` holds
  a bounded ring at the pole, `0.25 M` grows on the axis), while `0.625 M` is
  fine at `3/5`; between the two it is unknown. `fit_cont = 2` won every pair
  at `a = 0`; the spinning runs have used `fit_cont = 1` only.
- **`:excised`** (the rows added 2026-10-08, when `main` was merged into the
  excision round; the runs are in
  [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-steps-x1x7-2026-10-05-to-2026-10-08)):
  no target and no ramp; a surface at least 24 cells below the polar horizon at
  the coarsest `h` and, at `a = 3/5`, clear of the ring by `0.2 M`; `ε_KO > 0`
  is required. Shallower is less accurate outside the horizon at `a = 0` (the
  blend `upwind = 1,4` recovers it) and fails at `a = 3/5`: `r_E = 17/15`,
  16 cells at `1/24`, blows up at a lego corner at the pole. **(Amended in step
  X8:** with the shave, the default from step X8, `17/15` and `21/20` are
  stationary at `1/24`, but `r_E = 3/2`, 7 cells, grows at a corner the shave
  leaves on the equatorial cap; keep the 24 cells and `r_E = 4/5`
  ([`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#the-polar-corner-step-x8)).
  The scan at `h = 1/24`, `1/32`, `1/48` holds everywhere for `r_E ≤ 1.0` with
  the shave and `0.82` without; above that some lattice configurations fail at
  every `h`, and faster at finer `h`.**)**
  Static holes only, and spins above `3/5` are not measured. An excised
  right-hand side costs about half a `:damped` one on the H200.
- **Screen a new `:fitted` depth** before a long run: the same tracked
  geometry with the exact target (`variant = :damped`) for `6 M` at the
  run's `h`, about an hour and a half at `1/96` — a bad depth grows from
  `1 M`. Only the scratch copy on Symmetry (`spin-sync/`) can run it today:
  `test/octant_runs.jl` needs its `variant=` option, and the singular-set
  check, which tests the core's least radius against `a` and so refuses a
  polar radius below the equatorial ring, needs to know the ring's plane.
- **What would firm this up**, in order: `a = 9/10` `:damped` at `1/48` and
  `1/72` with `r_0 = 1` — the ring rule's prediction, about two hours for
  the coarser row on one H200 (the `1/48` row that was outside the convergent
  regime had `r_0 = 19/20`, `L = 4.7` cells; `r_0 = 1` doubles that and still
  leaves a 12-cell ramp and an 8.9-cell margin); depth screens at
  `a = 0.7` and `0.8`; `fit_cont = 2` with spin, at `3/5`.
- **What it costs** on one H200 with this branch's kernel: `26 s/M` at
  `a = 0`, `1/16` (9.4 M points); `6.5 min/M` at `3/5`, `1/32` (60.8 M);
  `5 min/M` at `9/10`, `1/48` (31.9 M); `15 min/M` at `9/10`, `1/96`
  (52.8 M). `main`'s rewritten kernel is about `8×` faster per point.

### The right-hand side on an H200 (measured 2026-10-05)

An investigation of how the right-hand side can be evaluated faster on a GPU, made
as an analysis and changing nothing in `src/`. Scope (Erik's):

- `q = 4`, no interior (`INT = :none`), one H200, `Float64`;
- the state's storage, the ghosts and the time integration are open to change.

It takes up [Possible extensions](#possible-extensions)' research item and answers
[Open questions](#open-questions) 3.

**Setup.**

- **Hardware and versions:** `main` at `acc30ba` on one H200 (Symmetry cn111,
  `h200debugq`): TreeAMR 0.1.7, CUDA.jl 6.4.2, KernelAbstractions 0.9.43,
  Julia 1.13.1.
- **Case:** the gauge wave with `ε_KO = 1/2`, `γ0 = 1` (no gauge source), on a
  uniform periodic mesh. The main mesh is 512 blocks of `32³` (16.8 M points); also
  8 × `128³` (the octant runs' `N`) and 512 × `16³`.
- **Timing:** the minimum of eight calls, in nanoseconds per owned point.
- **Correctness:** every prototype was compared with `gh_rhs!` on the same filled
  working array. `h` agrees bit for bit without inlining and to `2·10⁻³²` relative
  with it. `Π` agrees to `7.1·10⁻¹³` of `max |∂ₜΠ|`. That is also what the
  package's *own* kernel shows when it is merely compiled with inlining forced:
  contraction differences of one body at two call sites ("Two spellings of one
  expression are not bit-identical"), not a different scheme.
- **Where the code is:** the prototypes are raw CUDA.jl kernels and KA kernels in
  `bench/rhs_lab.jl`, whose modes `baseline`, `variants` and `round3` … `round9` are
  the rounds below. It runs from a copy of the package with CUDA added to its
  `Project.toml` (on Symmetry `rhs-gpu-lab`, jobs 570186–570203). Its companions:
  - `bench/rhs_lab_source.jl`, the lean source, shared by the GPU and CPU scripts;
  - `bench/rhs_lab_cpu.jl`, the source's CPU check and timing, in the package's own
    environment;
  - `bench/sass_stats.jl`, which counts a SASS dump's instructions.

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

**Why the kernel is slow: on a device it is not inlined.** KA launches it with a
default `CUDABackend()`, whose `always_inline = false` leaves GPUCompiler to Julia's
inlining heuristics. These keep as real functions:

- the `ntuple(Val(n)) do … end` blocks of `gh_rhs_at_point`;
- `mixed_stencil`'s closures;
- the StaticArrays generator closures of `gh_node_source` and `metric_derivatives`.

A device call passes its `SVector` and `SArray` arguments through the stack, and a
closure that indexes an `SVector` with its argument indexes memory. The SASS:

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
devices](#precision-threads-devices)' budget is what the kernel costs, and **the
stencils are not the problem.** Ablations of the inlined kernel (raw CUDA, static
strides):

| | ns/pt |
|---|---|
| state, `∂_i h`, the dissipation of `h`, the coefficients and `∂ₜh` only | 0.32 |
| everything but the source | 1.20 |
| everything but the Π stencils | 2.54 |
| everything | 4.30 |

The source alone in a kernel of its own, with nothing else live, still holds 255
registers and 1.8 KB of local memory and costs 2.02 ns/pt. `gh_node_source` builds
`Clul`, `Cluu`, `Γlll` and `Γ4` as 64-entry `SArray{Tuple{4,4,4}}` from generators,
and nothing between Julia, LLVM and `ptxas` reorders that DAG to shorten live
ranges.

**The lever: a register-lean spelling of the source.** It keeps `gh_node_source`'s
terms and changes three things about how they are formed:

- **Unique components.** Symmetric tensors are held by their unique components:
  `∂_a g_bc` and `Γ^a_bc` as four `SVector{10}`s in `_pack10`'s slots.
- **Phases ordered so that intermediates die.**
  1. `S = C2 + C2ᵀ`, one first index `a` at a time. `E = G D_a` (4×4) and
     `Cuu = E G` (ten values) give `C2[a, b] = Σ_μν Cuu_μν D_μ[ν, b]`. No 4×4×4
     array is ever formed.
  2. The forty `Γ^a_bc = Σ_x G^{ax} ½(D_b[x,c] + D_c[x,b] − D_x[b,c])` and
     `Γ^c = Σ G^{de} Γ^c_de`, then per packed `(ab)`:
     `−2 Σ_xy Γ^x_ya Γ^y_xb − (∂_a H_b + ∂_b H_a) + 2 Σ_c Γ^c_ab H_c − Σ_c Γ^c ∂_c g_ab`.
  3. The damping.
  4. `−α√γ S`.
- **No closures.** Every loop is unrolled with literal indices by
  `Base.Cartesian.@ntuple`.

It agrees with `gh_node_source` to `8·10⁻¹⁶` relative, the worst of 200 random
Lorentzian states (`bench/rhs_lab_cpu.jl`). It is not a re-derivation: it is a third spelling of the validated algebra,
to be tested against `gh_node_source` and `gh_node_rhs` as the second spelling is
tested against the first. Measured:

| `N = 32` | ns/pt | registers, local |
|---|---|---|
| source alone, `gh_node_source` | 2.02 | 255, 1792 B |
| source alone, lean | **0.178** | 254, 40 B |
| fused kernel, package source, reordered (below) | 3.81 | 255, 3.3 KB |
| fused kernel, lean source, reordered | **1.21** (1.32 at `16³`, 1.49 at `128³`) | 255, 2.0 KB |
| two kernels: pointwise and source, then the Π principal part | 0.85 + 0.29 = 1.14 | |
| three kernels: stencils and coefficients, the source, the Π principal part | 0.67 + 0.18 + 0.29 = 1.14 | |

The lean source does 1879 FP64 instructions a point: 3.1 kflop in 0.178 ns,
17.7 TFLOP/s, **52 % of the H200's FP64 peak**. The package's spelling of the same
terms takes 2.02 ns, eleven times as long.

On the CPU (development machine, one thread, `bench/rhs_lab_cpu.jl`), each
inlined into its caller as in the kernel:

| | ns per call |
|---|---|
| `gh_node_source` | 383–396 |
| the closure-free lean source | 90 |
| the first lean version, each loop an `ntuple` do-block | 2.6–2.7 µs |

So the closure-free spelling is 4.3× faster than `gh_node_source` on the CPU. The
do-block version was also 46 ns/pt on the device without forced inlining: its
closures were calls on both backends.

**The fused kernel's order matters once the source is lean.** The package's kernel
with only `gh_node_source` replaced runs at 7.56 ns/pt inlined and 49 ns/pt without.
`CODE.md`'s streaming order puts the source last, as step 3. The fast kernel keeps
the per-component stencil streaming but moves the source before the Π components:

1. load the state and form `∂_i h` and the coefficient set;
2. form `∂ₜh` with its dissipation and **store it**;
3. form the source;
4. run a *runtime* `for` loop over the ten Π components, each formed, combined with
   its source component and stored.

As a KA kernel (static workgroup, `N = 32`):

| | ns/pt |
|---|---|
| the order above, static strides (`Val(N)`), stencils without the zero weights | **1.21** |
| … with strides from `size(work)` | 1.26 |
| … with the Π components unrolled by `ntuple` | 1.29 |
| … with all stencil weights, zeros included | 1.21 |
| … with the source after the Π components (the package's order) | 1.34 |
| dynamic strides, unrolled, all weights, source last, together | 3.48 |
| the order above without `always_inline` | 51 (3.10 with the closure-free source) |

**KernelAbstractions is not the obstacle, and the launch must be configured.**
Written as KA kernels over `@index(Global, NTuple)`, the three-kernel split matches
raw CUDA exactly: 1.15 ns/pt at `N = 32` and 1.29–1.31 at `N = 128`. That needs two
things:

- **a static workgroup**, `kernel(backend, (32, 4, 1, 1))`. A dynamic one costs
  about 20 % (1.38 ns/pt);
- **forced inlining.** Without it the split runs at 48–50 ns/pt.

The package cannot ask for either today. `launch_by_owner!` launches with
`get_backend(fs.work)`, a default `CUDABackend()`, and a field set does not keep the
backend it was built with. GHSO2 met the same trap on the CPU (`get_backend(::Array)`
always returns the dynamic `CPU()`). There are two ways out:

- **TreeAMR keeps the backend**, its flags and a workgroup, and `map_blocks!`
  launches with it — upstream, by the mesh rule;
- **every helper the kernel reaches is written without closures**, as the lean
  source is. Measured for the source alone: 0.22 ns/pt without the flag, 0.18 with
  it.

**The design's other ideas, measured.**

| idea (where it was proposed) | measured | verdict |
|---|---|---|
| Stencil/algebra split ("One right-hand-side evaluation", black-box attempt) | package source: 3.2 + 0.29 against 4.3 fused (inlined); lean source: 1.14 against 1.21 | 6 % once the source fits; the Π principal part alone is a spill-free 0.29 ns/pt kernel at ~80 % of L1 load bandwidth |
| Workgroup shape (black-box attempt; GHAccel's 10 %) | within 10 % for a spilling kernel; static against dynamic is ~20 % for the split | set a static `(32, 4, 1)`; no further tuning |
| `Float32` (black-box attempt) | not measured: `Float64` is the requirement (1.4–1.6× in September) | — |
| Shared-memory staging of one variable's ghosted block ([Possible extensions](#possible-extensions)) | Π principal part, one component's `h` and `Π` tiles at a time: 0.95–1.59 ns/pt against 0.28 through L1 | **no**: L1 already holds the reuse |
| Fusing the scatter and the stage update into the kernel (GHSO2's "Tier 5") | scatter plus stage arithmetic ≈ 0.48 ns/pt per right-hand side | second order now; ~10–15 % of a step after the kernel (below) |
| Generated code for the source's register schedule | the hand-written lean source, 11× the source and 7× the kernel | **yes**, and no generator is needed |
| Kernel or scatter-and-ghost pattern first ([Possible extensions](#possible-extensions)) | the kernel is 92 % of an evaluation | the kernel, then the pattern |
| `Int32`, arrays, `div` (GHAccel) | static against dynamic strides: no difference while spilling, 4 % in the lean kernel | minor |
| Skipping the stencils' zero weights (new; IEEE forbids dropping `0·x`, so a fifth of the first derivative's loads and over a third of the mixed derivative's are zeros) | Π principal part 0.283 against 0.309 ns/pt; nothing elsewhere | minor |
| `maxregs` (new) | always slower: the fused kernel 5.95 at 128 and 5.72 at 168 against 4.30 | no |
| One Π component per thread (new) | 0.31–0.43 against 0.28–0.31 ns/pt | no |

The algebraic Kerr-Schild source has `gh_node_source`'s structure — an `SMatrix`
generator over closures. It adds 0.47 ns/pt to the fused lean kernel and costs 0.54
ns/pt in the source's own kernel against 0.18 without it, so it wants the same
treatment. So does `metric_derivatives`: the stencils-and-coefficients kernel still
spills 1.26 KB.

**Around the kernel: storage, ghosts and time integration.** Once the kernel is
1.2 ns/pt, the pattern around it is 40 % of an evaluation at `N = 32` and two thirds
at `N = 16`:

| per owned point, ns | `16³` | `32³` | `128³` |
|---|---|---|---|
| TreeAMR `scatter!` | 0.36 | 0.34 | 0.33 |
| the same with static strides | 0.19 | 0.14 | 0.12 |
| TreeAMR `fill_ghosts!` | 1.03 | 0.40 | 0.08 |
| copy floor: 20 values in, 20 out | 0.10 | 0.088 | 0.086 |
| one stage broadcast, `y + c k` | 0.12 | 0.12 | 0.12 |
| stage update and scatter in one kernel | 0.41 | 0.35 | 0.32 |
| the fused lean kernel, for scale | 1.32 | 1.21 | 1.49 |

**TreeAMR's copy kernels run far below bandwidth.**

- The scatter moves 320 B a point at 0.96 TB/s, and the ghost fill's transfer kernel
  0.65–0.68 TB/s, of the H200's 4.8 TB/s.
- Both index through KA's 5-D `CartesianIndices` with run-time sizes. Their SASS is
  about 1200 instructions for one element, with integer division sequences.
- A static-stride scatter is 2.4× faster. A fill at copy bandwidth would be about 4×
  faster, which at `16³` saves more than the whole lean kernel costs.
- That is TreeAMR's: precomputed or static indexing for its copy groups.

**Storage.**

- **Block size.** On the device `N ≥ 32`: at `16³` the fill is as dear as the
  kernel. At `128³` the fill is negligible, but the stencil kernels are 5–25 %
  slower (the fused lean kernel 1.49 against 1.21) because their strides reach
  further. With an efficient fill, `32`–`64` is the range to use.
- **Layout.** The variable-major block layout `(i, j, k, v, b)` stays: warps
  coalesce along `i` and every component is a dense 3-D array. Padding the stored
  `x` extent (39 at `N = 32`) so that owned rows start on 128 B would save L1
  wavefronts on the stencil loads without an `x` offset — about half of them — for
  perhaps 10 % on the stencil kernels **(predicted, not measured)**.
- **A ghosted state.** The integrator's stage vectors are field sets, so there is
  no scatter. The cost is `(N + 2G + 1)³/N³` of memory per stage vector — 1.81 at
  `N = 32`, 1.16 at `N = 128` — unless the stage arithmetic skips the ghosts.

**Time integration.** A native RK4 whose stage update is the right-hand side's
epilogue never stores `k`. The kernel writes `acc += b Δt k`, and writes the next
stage's input `y + a Δt k` straight into a second working array — double-buffered,
since neighbours still read the current one.

- **What it saves:** about 640 B a point a stage against 960, so 0.1–0.2 ns/pt a
  stage and **10–15 % of a step** once the kernel is 1.2 ns/pt. It removes passes,
  not much traffic.
- **What it would change:** the range projection is pointwise and would go into the
  same epilogue. That merges two of [the state's three
  writers](#the-range-projection) into one kernel, which is a
  decision for review.

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

**The order of the work (proposed).**

1. **The lean, closure-free source in `pointwise.jl`.** It is a third spelling,
   tested to roundoff against `gh_node_source` on every background of
   `pointwise_backgrounds.jl` and against the forward-mode pass. It helps the CPU
   too. **(Done 2026-10-05:** `gh_node_source_lean`.**)**
2. **The kernel in the order above**, with `Val(N)` for static strides and linear
   `@inbounds` stores. This amends "One right-hand-side evaluation": the source
   moves from step 3 to step 2, and the per-component streaming stays. **(Done
   2026-10-05, without `Val(N)`:** static strides are worth 4 %, and a kernel
   specialised on the block size would be compiled again at every `N` the suite
   runs.**)**
3. **Forced inlining and a static workgroup** on the device, through TreeAMR keeping
   the backend a field set was built with, or the kernel's remaining closures
   rewritten.
   - Steps 1–3 together are measured: **8.5 → 1.2 ns/pt**.
   - Steps 1–2 alone are 3.1 ns/pt.

   **(Done 2026-10-05 by the second route:** nothing in the kernel is a closure
   any more, and forcing inlining changes nothing — below.**)**
4. **TreeAMR's scatter and transfer kernels at bandwidth:** 0.74 → ~0.25 ns/pt at
   `N = 32`, and more at `N = 16`. Upstream. **(The first change TreeAMR should
   make:** [Upstream prerequisites](#upstream-prerequisites), item 5.**)**
5. **`metric_derivatives` and the algebraic gauge source made lean** like the
   source. **(Half done 2026-10-05:** `metric_divergences` forms the contraction
   alone; the algebraic source has lost its closures but not its spills.**)**
6. **Then, if its 10–15 % is wanted, a native stepper** with a ghosted state and the
   stage update fused into the right-hand side.

Not worth doing, measured: shared-memory staging of the principal part, `maxregs`,
one component per thread, further workgroup tuning.

**Implemented (2026-10-05).** Items 1–3, and half of 5, are in `src/`.

The package's own `gh_rhs!`, through `map_blocks!` and KernelAbstractions'
default backend (H200 job 570217, cn111):

| per owned point, ns | 512 × `16³` | 512 × `32³` | 8 × `128³` |
|---|---|---|---|
| the kernel, before → after | 8.55 → **1.21** | 8.48 → **1.08** | 8.90 → **1.28** |
| `gh_rhs!`, before → after | 9.94 → **2.61** | 9.21 → **1.81** | 9.31 → **1.70** |
| of which TreeAMR's `scatter!` and `fill_ghosts!`, unchanged | 1.39 | 0.73 | 0.41 |

- **Forcing inlining changes nothing now:** 1.075 against 1.077 ns at `32³`, with
  `du` bit for bit the same.
- **The kernel's SASS:** 5992 instructions, 2872 of them FP64, 255 registers, and no
  call into Julia code. Its 32 call sites are CUDA's division and square-root slow
  paths and the exception paths. Some spilling is left (298 `LDL` and 232 `STL`,
  static): the head still spills a little, as the prototypes' K1a did.
- **`bench/stepping.jl` on the H200** (512 × `16³`):
  - the gauge wave's right-hand side 20.7 → 5.47 ms, and its RK4 step 84.3 → 22.7 ms;
  - the hole fixture 28.8 → 17.8 ms and 116 → 72 ms. Its `:damped` layer branch
    still collects `F` and evaluates the analytic solution's dual pass.
- **The driver's records** (`BENCH_MODE=driver`): the hole's `err_l2 =
  1.690514e−06` equals September's to every printed digit, and the wave's is
  `7.683712e−09`.
- **On the CPU** (development machine, four threads, loaded 9–12), the right-hand
  side is faster too: the gauge wave's (64 × `16³`) 140 → 84 ms, the hole
  fixture's (512 × `16³`) 1521 → 1187 ms.

What is left of an evaluation is TreeAMR's: the scatter and the ghost fill are 40 %
of `gh_rhs!` at `32³` and 53 % at `16³`.

**Not measured.**

- A hole: the interior branches, the sampled gauge source and the `:fitted` cache
  reads are outside this scope. The layer kernels run the new head and Π functions
  but keep their own shape.
- `ncu` counters: occupancy and HBM traffic above are inferred from registers,
  local memory and SASS counts.
- `Float32`, more than one GPU, and the monitor kernels.

### The right-hand side on a CPU (measured 2026-10-05)

The H200 study's question, asked of Symmetry's AMD nodes: where does an evaluation go
on a CPU, and what would SIMD **across grid points** buy? Made as an analysis,
changing nothing in `src/`. Scope (Erik's): the H200 study's — `q = 4`, no interior,
`Float64`, the gauge wave with `ε_KO = 1/2`, `γ0 = 1`, no gauge source, on a uniform
periodic mesh — starting from that study's kernel (`c80e52d`).

**Setup.**

- **Hardware:** two kinds of 64-core node answer to `amddebugq`, and both were
  measured: **cn086**, two EPYC 7532 (Zen 2), and **cn096** (and cn095 for the step),
  two EPYC 7543 (Zen 3). Both are AVX2 with two 256-bit FMA pipes a core, sixteen
  `ymm` registers and no AVX-512, so four `Float64` lanes is the natural width; both
  have eight NUMA domains. Zen 3's L3 is 32 MB per eight cores, Zen 2's 16 MB per four.
- **Versions:** Julia 1.13.1, TreeAMR 0.1.7, KernelAbstractions 0.9.43,
  IMEXRungeKutta 1.3.0, SIMD.jl 3.7.2.
- **Placement:** pinned, `JULIA_EXCLUSIVE=1` with `srun --cpu-bind=none` (the fastest
  of September's four placements).
- **Timing:** the minimum of ten calls. A row "per point · thread" is the time per
  owned point times the thread count, so that one and 64 threads read on one scale:
  it is what a point costs one core.
- **Meshes:** 8 blocks of `16³` on one thread, 512 on 64 (2.1 M points); and the
  same points as 1 and 64 blocks of `32³`.
- **Where the code is:** `bench/rhs_cpu_lab.jl`, modes `breakdown`, `simd`, `asm`,
  `parts` and `profile`, with SIMD.jl from an environment stacked behind the
  package's (its header); Symmetry jobs 570286 and 570291–570293, directory
  `rhs-cpu-simd`. The step is `bench/stepping.jl`'s.

**Where an evaluation goes** (`16³`):

| ns per point · thread | Zen 2, 1 thread | Zen 2, 64 threads | Zen 3, 1 thread | Zen 3, 64 threads |
|---|---|---|---|---|
| `gh_rhs!` | 1715 | 2273–2563 | 1423 | 2124 |
| — `scatter!` | 95 | 154 | 93 | 133 |
| — `fill_ghosts!` | **396** | **877–894** | **331** | **767** |
| — the kernel | 1217 | 1229–1242 | 998 | 1152 |
| the copy floor (below) | 57 | 308 | 16 | 238 |

At 64 threads that is 70 ms for `gh_rhs!` on Zen 3 — 4.4 the scatter, 25.1 the fill,
37.7 the kernel — and 74.5–84.0 on Zen 2, whose fill and kernel are 29 and 40.
At `32³` the fill is a smaller part: 8.6 of `gh_rhs!`'s 56.4 ms on Zen 3 (15 %, against
a copy floor of 4.0) and 10.5 of 56–57 on Zen 2 (19 %).

- **The kernel scales with threads, and KernelAbstractions costs it nothing.** A
  point costs a core about the same at 1, 8 and 64 threads (Zen 2: 1189–1247 ns; Zen
  3: 998 on one thread, 1152 at 64, where every core runs at the all-core clock and
  shares its L3). The same body called from a plain loop instead of a KA launch is
  within 4 %.
- **The ghost fill is the other issue.** It costs **8–10 ns a ghost value** on one
  core (at `16³` an owned point has about 39 ghost values behind it). The "copy
  floor" row is each block's whole stored slab — ghosts, owned points and all twenty
  variables, half as much again as the fill moves — copied by its owner thread with
  `unsafe_copyto!`. The fill takes 7× the floor's time on one Zen 2 core, 20× on a
  Zen 3 core (where the eight blocks fit in L3), and 2.9–3.2× at 64 threads, where
  the copy is memory-bound (7.8–10.1 ms, 200–260 GB/s). It is **34–39 % of
  `gh_rhs!`** at `16³` and 64 threads.
- **Why the fill is slow** (`LAB_MODE=profile`, Zen 2, one thread): every same-level
  copy is a tensor-product stencil — weight loads, a weight product and a multiply a
  value. The generated sum is half the samples and the float `*` alone 13 %; then
  KernelAbstractions' 5-D `CartesianIndices` iteration (`==`, `!=` and `__inc`, 11 %),
  integer index arithmetic (11 %) and the bounds checks that are left (3 %). At 64
  threads a further 19 % of the samples are threads in `wait()` between phases. That
  is [Upstream prerequisites](#upstream-prerequisites)' item 5 on the CPU: a copy path
  for the `Ps = (1, 1, 1)` groups.
- `gh_rhs!` allocates 12.5 kB an evaluation on one thread and 0.9 MB at 64: the KA
  launch tuples of TreeAMR's slices, one a thread a phase. Not a cost at this size.

**The kernel's native code** (`LAB_MODE=asm`; Zen 2's and Zen 3's agree to 1 %).
Dynamic counts per point, from the code's one loop, the ten Π components:

| per point | instructions | floating point | stack accesses |
|---|---|---|---|
| scalar | ~9500 | ~4100 | ~3200 |
| SIMD, four lanes | ~2930 | ~1690, all `ymm` | ~1220 |

- **The scalar kernel is already partly vectorized within a point.** LLVM's SLP
  vectorizer packs about 1.8 operations into a floating-point instruction (statically
  288 `ymm`, 1030 packed `xmm` and 1037 scalar ones): the source's sums and the tensor
  algebra pair up. The loop over points is not vectorized and could not be — it is
  thousands of operations with branches and strided stores.
- **It runs at about 2.4 instructions a cycle and one floating-point instruction a
  cycle** (at the 3.3 GHz boost clock): the two FMA pipes are half busy. A third of
  its instructions touch the stack: sixteen registers do not hold the head's live set
  (the thirty `∂_i h`, the metric, its inverse).
- **There are no calls** but the bounds-error and `DomainError` paths, and fourteen
  divisions and two square roots a point.

**SIMD across points: the package's own kernel on SIMD.jl's `Vec{W,Float64}`.**
`gh_rhs_store!` is generic in its number type, so the prototype evaluates `W`
neighbouring points along the first axis by passing `Vec{W,Float64}` as `T`, and
reads the working array through an accessor whose `work[i]` is the `Vec` of elements
`i … i + W − 1`: every stencil load becomes a contiguous vector load and every store a
vector store. Nothing in `src/` is copied. A `Vec` is not a `Number`, so the lab
defines the three things it lacks: conversion from a literal (`_η4`), a `Vec` times a
`StaticArray`, and the lean source's `γ0 != 0`, which becomes "any lane damps". The
launch is over `(N/W, N, N, nblocks)`.

| the kernel, ns per point · thread, `16³` | Zen 2, 1 thread | Zen 2, 64 threads | Zen 3, 1 thread | Zen 3, 64 threads |
|---|---|---|---|---|
| scalar | 1223 | 1222 | 989 | 1164 |
| SIMD, `W = 2` | 769 | 820 | 600 | 729 |
| SIMD, `W = 4` | 519 | 579 | 427 | 576 |
| SIMD, `W = 8` | 541 | 613 | 401 | 579 |
| scalar, zero weights skipped (below) | 1141 | 1168 | 896 | 1010 |
| SIMD, `W = 4`, zero weights skipped | **481** | **544** | **379** | **536** |

- **Four lanes are 2.0–2.4× the scalar kernel.** At 64 threads on Zen 3 the kernel
  goes from 37.7 to 18.9 ms an evaluation, and to **17.5 ms with the zero weights
  skipped** — 2.15×. `32³` is alike: 1165 → 481 on one Zen 2 core and 1218 → 554 at
  64 threads; on Zen 3 at 64 threads, one block a thread, 41–49 → 22.5 ms (the
  noisiest row: one slow core sets the time). Eight lanes are no better than four on
  Zen 2 and up to 8 % better on Zen 3 (401 against 427 on one core, 20.9 against
  22.5 ms at `32³`): two registers a value double the spills (2291 against 957 stack
  loads, statically).
- **The result is bit for bit the scalar kernel's** on the gauge wave, at every
  width, on Zen 2, Zen 3 and the development machine's aarch64. Every lane does the
  scalar code's operations in its order. **(Amended in the implementation,
  2026-10-05:** that is the gauge wave's, not the kernel's. On a hole the lanes
  differ from the scalar kernel in the last bits — up to `1.2e−14` on the step-5
  fixture, 1.0–2.0 eps of the size of the terms at evolved points (`rhs_scale`) —
  because StaticArrays forms `inv(g4) * h` with
  `muladd`, and LLVM fuses each into an FMA or not by the code around it, which the
  lanes change; the gauge wave's zero shift hides it. So the claim is roundoff, as for
  every two call sites of one body; bit-identity across thread counts is unaffected.**)**
- **Why not 4×.** SIMD removes 3.2× of the instructions and gains 2.3–2.5× on one
  core. The vector kernel's instructions are still two fifths stack traffic (its live
  set is four times wider in the same sixteen registers), and the scalar kernel was
  already 1.8-wide in its arithmetic. The **head** — the metric, `∂h`, the source, the
  divergences — is 57 % of the vector kernel's instructions; the ten Π components,
  whose stencils vectorize perfectly, are the rest. At 64 threads the vector kernel
  loses a further 12–35 % a core to the all-core clock and the shared caches, the
  scalar one 0–18 %.
- **The pieces alone** (`LAB_MODE=parts`, one thread, 4096 random states, ns a point):

  | | Zen 2, scalar | Zen 2, `Vec{4}` | Zen 3, scalar | Zen 3, `Vec{4}` | M3, scalar | M3, `Vec{4}` |
  |---|---|---|---|---|---|---|
  | `metric_quantities` | 103.5 | 36.1 | 85.5 | 32.6 | 43.7 | 37.4 |
  | `gh_node_source_lean` | 357.4 | 116.0 | 290.8 | 91.2 | 86.2 | 92.0 |
  | `metric_divergences` | 67.1 | 21.0 | 52.9 | 18.6 | 34.3 | 19.3 |

  On the EPYC nodes four lanes are 2.6–3.2× for every piece. **On the development
  machine's Apple M3 SIMD across points is worth only 1.3× to the kernel** (473 → 369
  ns a point at `W = 4`), and nothing to the source: LLVM's SLP vectorization already
  fills its 128-bit pipes, and a scalar M3 core runs the source four times as fast as
  a Zen 2 core. The lever is an x86 one, and a measurement on the laptop understates
  it.

**The stencils' zero weights.** The first-derivative weights at `q = 4` are `(1/12,
−2/3, 0, 2/3, −1/12)`: a fifth of an `axis_stencil`'s products and 9 of a
`mixed_stencil`'s 25 multiply by zero, and IEEE arithmetic forbids the compiler to
drop `0·x`. With the weights in the type the generated contractions can leave those
terms out (`LAB_ZW=1`). For finite data the sum is the same number — adding `+0.0` is
exact — and measured, the whole `du` is **bitwise equal** to the package's stencils'
(Zen 3 and the M3). It is worth **4–13 %** of the kernel, scalar or SIMD (on the H200
it was "minor"), and it is independent of SIMD.

**What it buys a step** (Zen 3, 64 threads, 512 × `16³`, 2.1 M points; the step is
IMEXRungeKutta's RK4 by owner, `bench/stepping.jl` on cn095, 361.5 ms, of which four
right-hand sides are 279 and the stage arithmetic and limiter the remaining 82):

| ms | `gh_rhs!` | RK4 step |
|---|---|---|
| today | 69.6 | 361.5 |
| the kernel on four lanes, zero weights skipped (measured) | 49.4 | 280 |
| and the ghost fill at copy speed (predicted: two thirds of the slab copy) | 29.5 | 200 |

So **SIMD across points alone is 1.4× an evaluation and 1.3× a step**, though it is
2.15× the kernel: the ghost fill (25 ms) and the stage arithmetic (82 ms a step) do
not move. With the fill at copy speed the two together are 2.4× an evaluation and
1.8× a step, and the stage arithmetic is then 41 % of what is left — the H200 study's
fused stepper, which on the CPU would also remove the scatter.

**What SIMD in `src/` would take (proposed; done 2026-10-05 as below, with an
overlapping last group instead of a scalar tail, and on every branch).**

1. **A lane type.** SIMD.jl's `Vec{W,Float64}` works through the package's algebra
   unchanged but for the three methods above; SIMD.jl has no dependencies of its own.
   The `γ0 != 0` branch would be spelled through a helper that a `Vec` answers with
   `any`. **(Done:** SIMD.jl is a dependency; `pointwise.jl` has `_scale`,
   `_anynonzero` and `_select`, and `_η4` is spelled with `one(T)` and `zero(T)`; no
   method is added to a type the package does not own.**)**
2. **The launch**, over `(N/W, N, N, nblocks)` on the CPU and unchanged on a device
   (`W = 1`), chosen where `GHProblem` builds its `Val`s. `N` must be a multiple of
   `W`, or the row needs a scalar tail: the suite's `N = 10` at `q = 6` is not a
   multiple of four. **(Done otherwise:** the launch stays `map_blocks!`'s, and the
   item at the start of each group of `W` does the group — `is_lane_leader`, the other
   items do nothing. A row that `W` does not divide ends in a group that **overlaps**
   the one before it, Erik's proposal in place of a scalar tail or masks: no lane is
   ever outside the row, and the shared points are stored twice by the same thread.
   `W` is `default_simd_width`'s: 32 bytes of lanes, 64 with AVX-512, at most `N`.**)**
3. **No `DomainError` from a lane.** SIMD.jl's `sqrt` is the instruction: a degenerate
   metric gives a `NaN`, which the record's `finite` reads, where the scalar kernel
   throws (`CLAUDE.md`, the outer part of the layer). **(As predicted; the layer
   and the core keep the scalar code and its `DomainError`, and `simd_width = 1`
   restores it everywhere.)**
4. **The interior and the gauge sources are not measured.** A layer branches per
   point; the natural split is the vector path where all `W` lanes are outside the
   layer, which is most of the mesh, and the scalar one elsewhere. The sampled gauge
   source reads `Hsrc` along the first axis as the state is read; the algebraic
   Kerr-Schild source would have to accept a `Vec`. **(Done, all three:** a group
   takes the lanes when all its points are outside the layer and the scalar code point
   by point otherwise; `Lanes` reads `Hsrc` by its cartesian index; the algebraic
   source's two clamps are `_select`s.**)**
5. **The zero weights** need the weights in the type: `derivative_weights` returning a
   type whose `axis_stencil` and `mixed_stencil` are generated without the zero terms.
   **(Not done:** independent of the lanes, and left for a decision.**)**

**Implemented (2026-10-05).** The kernel above is the package's (`src/lanes.jl`,
`gh_rhs_lanes!`), on by default on the CPU. Measured with `bench/rhs_cpu_lab.jl` and
`bench/stepping.jl`, `simd_width = 1` (the scalar kernel) against the default, on the
same filled state; Symmetry jobs 570353–570361 and 570370, directories
`rhs-simd-*`:

| the package's kernel, ns per point · thread | `W = 1` | `W = 2` | `W = 4` | `W = 8` |
|---|---|---|---|---|
| Zen 3, 1 thread, 8 × `16³` | 975 | 581 | **409** | 430 |
| Zen 3, 1 thread, 8 × `10³` (an overlapping group) | 967 | | **469** | 657 |
| Zen 3, 64 threads, 512 × `16³` | 1238 | 803 | **578–597** | 659 |
| Zen 3, 64 threads, 64 × `32³` | 1324 | 790 | **609** | 732 |
| Skylake-AVX512 (Xeon Gold 6148, cn058), 1 thread | 1558 | | 787 | **551** |
| Skylake-AVX512, 40 threads, 512 × `16³` | 2289 | | 1372 | **922–1034** |

(Zen 2 was not measured again: the job meant for cn086 waited behind another user's
day-long job and ran on cn107, an EPYC 7543, whose rows agree with cn096's to a few
percent. The study's prototype rows above are Zen 2's, and the kernel is the same
code.)

- **The lab's prototype is reproduced:** four lanes are 2.2–2.4× on Zen 3, and the
  default width is the best one on both instruction sets — four on AVX2, eight on
  AVX-512, where the 32 registers hold an eight-wide head without the extra spills
  that cost eight lanes on AVX2 (the native code has the same instruction and spill
  counts at eight lanes as at four). That settles `default_simd_width`'s 64 bytes,
  which was a prediction until this row.
- **The overlapping group costs what it computes twice:** at `N = 10` three groups of
  four cover ten points, 20 % redundant, and the kernel is 469 ns a point where 409 ×
  1.2 = 491 was predicted; eight lanes are a poor fit there (two groups for ten
  points). No scalar tail and no mask.
- **A step** (Zen 3, 64 threads, 512 × `16³`, `bench/stepping.jl` with and without
  `BENCH_SIMD=1`):

  | ms | right-hand side, `W = 1` → `W = 4` | RK4 step, `W = 1` → `W = 4` |
  |---|---|---|
  | gauge wave | 83.9 → 50.6 | 367.3 → **258.4** (1.42×) |
  | the hole fixture (`:damped`, sampled source) | 212.2 → 201.8 | 897.1 → **830.7** (1.08×) |

  The gauge wave's step gains what the study predicted and a little more. The
  hole's kernel gains 1.35× (48.5 → 35.9 ms; Skylake: 155 → 78 ms at eight lanes),
  less than the wave's because a refined mesh around the hole has more groups that
  straddle the layer, which take the scalar code; and its `gh_rhs!` gains only 7 %,
  because about 165 ms of its 200 are spent outside the kernel. That is the ghost
  fill (cn107, another EPYC 7543): `fill_ghosts!` is 161 ms, of which the Dirichlet
  hook is 16 — the fill without it is 145 — against a copy floor of 7.8 ms. On the
  hole's mesh the fill is mostly prolongation onto the fine side of the coarse-fine
  faces, `6³ = 216` coarse points a ghost at `p = q + 2 = 6` ("A coarse-fine face is
  expensive", `CLAUDE.md`), and its profile at 64 threads is a third threads waiting
  (66 % utilisation: the prolongations are not spread evenly over the threads) and
  an eighth the interpolation's own additions and multiplications. **For a hole the
  lever is TreeAMR's prolongation, not the kernel** — item 5's copy path does not
  reach it ([Upstream prerequisites](#upstream-prerequisites)).
- **The lanes are the scalar kernel to roundoff** (`test/simd_tests.jl`, against
  `rhs_scale` at evolved points): bit for bit on the gauge wave at four lanes (Zen 3,
  the M3) and at eight (Zen 4c, Skylake), and 1.0–2.0 eps of the terms on the step-5
  fixture and on the octant hole with the algebraic source, every point written.
- **Devices are unchanged:** the H200 runs the `W = 1` kernel at PR #4's speed — the
  gauge wave's right-hand side 5.45 ms and step 22.7 ms (5.47 and 22.7 before), the
  hole's 17.84 and 72.4 (17.8 and 72) — and Metal compiles and runs it at `Float32`.
- **The suite** is 5037 assertions, green on the development machine (12m20 at four
  threads), on an EPYC 8534P (Zen 4c, AVX-512, so eight lanes; 23m59 at four threads)
  and on an EPYC 7543 (27m14 at one thread, 5029). `test/simd_tests.jl` is about 20 s
  of it, most of it the kernels at `W = 1` it compiles to compare against.

**Not measured.** The monitor kernels, which stay scalar, and hardware counters: `perf` is not open to users on Symmetry
(`perf_event_paranoid = 4`), so instructions a cycle and pipe occupancy above are
inferred from the native code and the clock.

### Excision, steps X1–X7

Moved to
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#excision-steps-x1x7-2026-10-05-to-2026-10-08)
(2026-10-08, when `main` was merged into the excision round): the analysis
([X1](SINGULARITY_HANDLING.md#excision-the-analysis-step-x1)), the variant
([X2b](SINGULARITY_HANDLING.md#excision-the-variant-step-x2b)), the static hole
on the octant ([X3](SINGULARITY_HANDLING.md#excision-on-the-static-hole-step-x3)),
the merge with `main` and the rotating octant
([X4](SINGULARITY_HANDLING.md#the-merge-with-main-and-the-rotating-octant-step-x4)),
the frame-dragged faces
([X5](SINGULARITY_HANDLING.md#excision-the-frame-dragged-faces-step-x5),
[X6](SINGULARITY_HANDLING.md#excision-the-frame-dragged-faces-in-the-zone-kernel-step-x6))
and the static spinning hole
([X7](SINGULARITY_HANDLING.md#excision-of-the-static-spinning-hole-step-x7)). Step
X8's runs went there too, under their own heading,
[The polar corner (step X8)](SINGULARITY_HANDLING.md#the-polar-corner-step-x8). The
design is [Excision](#excision-added-2026-10-05), and why it runs scalar on the
CPU is under [One right-hand-side evaluation](#one-right-hand-side-evaluation),
"Excision runs scalar on the CPU".

## Possible extensions

What separates the proof of concept from a production code, listed with
the design note that would start each:

- **Checkpoint and restart**: the forest's leaf keys, the time, the
  case and the state vector to HDF5, read back into a fresh forest
  (`Forest`, `refine!` to the keys, `balance!`, `scatter!`); bit-identical
  to an uninterrupted run, since the layer and the hook depend on
  `(x, t)` only. A few dozen lines, or TreeAMR's M9. **(Done 2026-10-01**
  through TreeAMR 0.1.4's M9a, whose load builds the forest from its leaf
  list directly (`Forest(roots; leaves)`) — see [Checkpoint and
  restart](#checkpoint-and-restart); and since step 8d the layer depends on
  the track and the fits as well, which the run state carries.**)**
- **Excision**, for spacetimes without an analytic interior. Like the
  interior layer it would be a pointwise decision — a mask that marks
  points as not evolved, with one-sided or extrapolated data supplied
  where evolved stencils reach into the mask — and its one entanglement
  with the mesh is TreeAMR's phase ordering: a prolongation reads a
  coarse block's ghosts before any after-the-fact kernel could repair
  them, so extrapolated data at a mask boundary has to be in place in
  the hook slot between the phases. That wants an interior-reading
  device hook and, cleanly, *excised leaves* in TreeAMR (leaves that
  keep their place in the tree but carry no data). GHSO2's recipe for
  the sonic surface applies unchanged. **(Priced 2026-09-23, PLAN.md
  step 8g.)** A fill that keeps the `∂²` stencil's order must reproduce
  the Taylor polynomial to degree `q + 1`, so a block-local fill needs a
  `3G` halo — stored volume ×4–7 at `N = 8`, where the vertex invariant
  `N ≥ 6G + 2` refuses it, ×1.8–2.7 at `N = 32` — or a second ghost
  exchange per evaluation, and degree `q + 1` amplifies grid-scale noise
  15–320× at depth 2. Excised leaves are block-granular and serve static
  holes only. **(Amended 2026-10-05:** being built and measured as the
  `:excised` variant, `PLAN.md`'s steps X1–X3, under
  [Excision](#excision-added-2026-10-05). The price above is an
  extrapolation fill's *per excised point*; one-sided closures *per stencil*
  read only inside the existing reach `G`, so they need neither the halo,
  the second exchange, the interior-reading hook nor excised leaves.
  Moving holes, whose trailing side releases excised points, are the round
  after the static one.**)**
- **The damped harmonic gauge driver** (Lindblom–Szilágyi 2009;
  Szilágyi–Lindblom–Scheel 2009): `H_a` algebraic in `g` and
  `log(√γ/α)`, its gradient by the chain rule through `∂_a g`, no extra
  variables, no extra exchange; the fix for the gauge drift G4 records,
  and the prerequisite for binaries.
- **A radiative outer boundary** for solutions not known at the
  boundary: the standard `1/r` outgoing extrapolation into the ghosts
  along the ray, needing a device-capable hook that reads the block's
  interior (TreeAMR's `CODE.md` names the gap).
- **Time-dependent prescribed gauge sources** evaluated in the kernel,
  which would admit boosted Kerr-Schild; priced at about one RHS per
  evaluation by GHSO2's `gauge_source_grad` timing.
- **A hole tracker** for the *interior layer* when the center is not
  known in advance — the volume-weighted centroid of `α < α_thr`
  through `block_mapreduce`, or the horizon finder itself; the
  refinement already follows the indicator.
- **Binaries**: superposed Kerr-Schild as the approximate data, then
  constraint-satisfying data from `XCTS` / `SolveConstraints` through
  the same `(h, Π)` callback on the host; needs excision or an interior
  layer per hole, the gauge driver and the radiative boundary.
- **Gravitational-wave extraction**: `Ψ_4` on extraction spheres, modes
  through spin-weighted harmonics, needing the interpolation of
  prerequisite 1.
- **The native RK stepper** with stage updates as kernels (GHSO2's
  `timestepper.jl`, made backend-generic), lifting TreeWave's 3.6× cap.
- **GPU kernel efficiency**, as a research project seeded by G6's
  measurements: shared-memory staging of one variable's ghosted block
  at a time (a `Float64` block of `22³` points is 85 kB, which fits per
  variable and not for twenty); fusing the scatter and the integrator's
  stage update into the RHS kernel so that `u`, the working array and
  the stage vectors stop round-tripping memory (GHSO2's "Tier 5" note,
  and a change to TreeAMR's RHS contract); generated code for the
  register schedule of the source algebra; and the question the traffic
  estimate above raises, whether TreeAMR's scatter-and-ghost pattern or
  the kernel is the first thing to change. **(Investigated 2026-10-05**, in [The
  right-hand side on an H200](#the-right-hand-side-on-an-h200-measured-2026-10-05).**)**
  - Shared-memory staging is slower than L1.
  - The source's register schedule is the lever, and a hand-written spelling
    suffices.
  - The kernel comes first; then TreeAMR's scatter and transfer kernels; the fused
    stage update last, at 10–15 % of a step.
- **SIMD across grid points on the CPU** (measured and implemented 2026-10-05,
  [The right-hand side on a CPU](#the-right-hand-side-on-a-cpu-measured-2026-10-05)):
  the package's own kernel on four SIMD.jl lanes is 2.0–2.4× the scalar one on
  Symmetry's EPYC nodes, the same `du` to roundoff, and the stencils without their
  zero weights a further 4–13 % (not implemented). It is 1.3× a step on its own: the
  ghost fill and the stage arithmetic are what is left, and the fill is TreeAMR's.
- **SIMD lanes for the excised right-hand side** (proposed in the main merge,
  2026-10-08). `:excised` runs scalar on the CPU ([One right-hand-side
  evaluation](#one-right-hand-side-evaluation), "Excision runs scalar on the
  CPU"); its price, measured in the main merge on `bench/stepping.jl`'s 2.1 M points at four
  threads, is 15 % of a right-hand side and of a step against `:damped` on
  lanes (953 against 827 ms, 3876 against 3376), and the main kernel's 1.5× on one thread
  (1.29 against 0.85 s). The design is in hand: a group of `W`
  centered points is the `:none` branch's lane call, and a group that meets the
  zone or the excised set is the same call with the other lanes masked off —
  SIMD.jl's masked `vload`/`vstore`, which touch no memory in a masked lane — before
  its excised and blended points are done one at a time. A scratch prototype of
  the merge built exactly that: every centered point of the suite's fixture and of
  its spinning hole was the `:none` lanes' bit for bit, the suite passed, and the
  `:excised` right-hand side cost what `:damped`'s does (820 against 827 ms; it is
  the local branch `claude/excision-simd-lanes-prototype`, not merged). The zone
  and frame-dragged kernels are a percent of a right-hand side and can stay
  scalar.
- **A truncation-error indicator** in place of Löhner's — Richardson
  between a block and its restriction, which TreeAMR's operators make
  cheap — if the calibration in G4 shows Löhner's thresholds to be
  fragile across resolutions.
- **Smoothing the interior** for a spacetime without an analytic
  solution — the interior layer relaxing toward something other than
  the exact solution. **(Became PLAN.md step 8e on 2026-09-23**: the
  fitted target; see [the interior's
  questions](SINGULARITY_HANDLING.md#the-interiors-questions-opened-in-step-5-and-closed-through-step-8).**)**
- **A hyperboloidal outer layer**, which this author's electrodynamics
  packages rehearse and which would retire the outer boundary question.
- **Matter**, through the source slot: TreeHydro's scheme on this
  spacetime is the GRHD code the two packages together rehearse.

## Open questions

Settled in review on 2026-09-16: the expanded form (the flux form is not
implemented on the mesh); three dimensions only; no excision, the
interior treated pointwise by profiles of the distance to the center
(reopened 2026-10-05: `:excised` is under study, below);
the proof-of-concept target a single boosted, spinning hole;
OrdinaryDiffEq's RK4 for time integration (amended 2026-09-26:
IMEXRungeKutta's RK4, by block owner); the analysis quantities as
part of the deliverable; an error indicator for refinement; `Float64`
on the H200 as the device requirement; no checkpointing (reversed
2026-10-01); the inherited documents copied into `notes/`.

**The interior's questions** — the spherical core against harmonic Kerr at
`a = 9/10`, the generic interior, the price of that chart, the fitted target and the
moving layer's trailing side, all opened in step 5 and closed or measured
through step 8′ — moved to
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#the-interiors-questions-opened-in-step-5-and-closed-through-step-8)
(2026-10-08).

Still proposed, to be confirmed or amended by the milestones that
first touch them:

1. **The damping layer `(INTERIOR)` as the default interior**, against
   the hard paste and the pure mask (G4, G5). The argument for it is
   under [The interior](#the-interior-a-pointwise-damping-layer); the
   numbers are not yet.
2. **The Löhner indicator on `h` with a global floor**, with the mask,
   the level floor around the horizon and the ceiling at the boundary
   (G4): whether calibrated thresholds carry from the static to the
   moving hole, and whether the floor ever binds.
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
   H200](#the-right-hand-side-on-an-h200-measured-2026-10-05).**)**
   - The split is worth 1.25× with the package's source and 6 % with a
     register-lean one, so the fused kernel stays.
   - The spills come from the source and from closures left uninlined on the
     device, not from the stencils.
4. **In-kernel evaluation of `SpacetimeMetrics`** on the H200 (G6); the
   structure does not depend on it, the cost does.
5. **The defaults** — `q = 4`, `N = 32`, `cfl = 1/4`, `ε_KO = 0.5`,
   `γ0 = 1/M` near the hole, `m = 8`, a layer of `2(G + 1)` spacings,
   `ρ_max = 4/M` (`ρ_max · dt = 1` until 2026-09-23), the indicator's
   thresholds — are starting values for G4–G6 to confirm or move. Three
   of them survived G2 on flat space and on a gauge wave: `cfl = 1/4` (no
   run needed less), `ε_KO = 0.5` (the noise test, and no order lost) and
   `q = 4` as the development order (`q = 2, 6, 8` all run, at 0.5×, 1.3×
   and 1.9× the cost of `q = 4`). None of that is yet a statement about a hole. Step 8a split
   `m` into a stencil margin and a leakage margin, measured that `m = 8`
   attenuates grid-scale content from `r_1` by `e^{−2.9}` to `e^{−4.6}` on
   the fixture, and left the default where it is until step 8c says what
   amplitude it has to hold back **(proposed in step 8a**; see [The
   interior](#the-interior-a-pointwise-damping-layer)**)**. Step 8c kept
   `m = 8` — a smooth layer's wrong target leaves the exterior what the exact
   one leaves — and replaced `ρ_max · dt = 1` by `ρ_max = 4/M` with a ramp of
   `4G` cells, measured for an inexact target and **(proposed in step 8c)**
   for the analytic one, where it cuts the `50 M` error sixfold. **Erik took
   it on 2026-09-23** for every variant, the analytic `:damped` layer
   included, and step 8c′ made `4/M` the code's default; the grid rate is
   the option `ρ_max_factor` **(decided 2026-09-23)**. The ramp is not part
   of that decision: it is the case's `r_0`, `r_1` and `ρ_ramp`, and the
   suite's fixture keeps step 5's layer, a ramp of `4.8` cells, on which
   step 8c measured the exact target at `4/M` to `50 M`.

**Gauge sources that know less about the hole (opened 2026-10-02 in a
design discussion with Erik; nothing is built).** Today `H_a(x)` is sampled
from the background ([Gauge and constraint
damping](#gauge-and-constraint-damping)), so it knows the hole's mass, spin,
position and velocity, and a moving Kerr-Schild hole is refused because its
`H_a(x − vt)` is not a per-chunk sample. The question is which gauge
sources need less of that, and which of them still make the initial data an
exact stationary solution, so that a run starts without a gauge transient.
Write `k_ab = g_ab − η_ab`; signs are GHSO2's, `H_a = −Γ_a` on a solution.
The algebra below was checked against the Kerr-Schild metric with Wolfram,
to 20 digits or better.

- **Without a length, only `H = 0`.** `H_a` is an inverse length and
  `g_ab` is dimensionless, so a nonzero gauge source that is an algebraic
  function of the metric carries a constant with the dimension of a
  length: a rate, like the damped harmonic gauge's `μ_0 ≈ 1/M` (on the
  footing of `γ0`), or the hole's own parameters. A gauge source that
  reads `∂g` instead changes the principal part: the equations
  differentiate `H` once (`S0` contains `−2∇_(a H_b)`), so `∂g` in `H` is
  `∂∂g` in the equations — in this package's variables, `∂_i Π` in `Π`'s
  equation — and the limiting case `H_b = −Γ_b[g]` is the Einstein
  equations with no gauge fixed at all. That is not ill-posed by itself
  (1+log slicing and the Gamma driver read `∂g`), but each such choice
  needs a hyperbolicity analysis of its own and gives up the ten decoupled
  shifted wave operators the energy estimate rests on; gauge drivers
  (Lindblom, Matthews, Rinne and Scheel 2008; Lindblom–Szilágyi 2009)
  avoid it by evolving `H_a` as a field.
- **The generic gauge in use elsewhere is damped harmonic**
  (Szilágyi–Lindblom–Scheel 2009), the driver [Possible
  extensions](#possible-extensions) already lists. In SpECTRE's form
  `H_a = [μ_L1 L + μ_L2 log(1/α)] t_a − μ_S g_ai β^i/α` with
  `L = log(√γ/α)` and `μ_X = A_X e^{−(r/σ_r)²} L^{e_X}`; the published
  binary runs take `A_L1 = A_S = 1`, `A_L2 = 0`, `e = 2`, and the Gaussian
  about the grid's origin makes the gauge harmonic at the outer boundary.
  It knows one rate and nothing else about the hole, and single holes
  settle in it: Lindblom and Szilágyi saw strongly perturbed holes reach
  time-independent states, and Varma and Scheel (2018, arXiv:1808.07490)
  construct the equilibrium of a boosted, spinning hole by solving four
  elliptic equations. It does **not** start stationary from Kerr-Schild
  data: Kerr-Schild has `α√γ = 1`, so `L = log(1 + 2H)` — `log 2` at the
  Schwarzschild horizon — and SpEC rolls the gauge on from the data's own
  source over `σ_g = 15–25 M`. The analytic solution then stops being the
  error reference (the constraints and the horizon's invariants remain),
  and a layer relaxing toward the analytic chart relaxes toward the wrong
  gauge; the `:fitted` target is the one that would survive it.
- **The harmonic chart's resolution penalty is forced.** A radial harmonic
  coordinate `f(r) n^i` on Schwarzschild (areal `r`) solves
  `d/dr((r² − 2Mr) f′) = 2f`, whose solutions are `r − M` and
  `(r − M) ln(1 − 2M/r) + 2M`. The second falls off as `−2M³/(3r²)`, so
  asymptotic flatness allows it, and diverges at the horizon, so
  regularity does not: every regular harmonic chart has its horizon at
  `r_H = r_BL − M`. For a spinning hole the price is more than the factor
  two the radius suggests, because the chart's singular disk of radius `a`
  closes in on the horizon's equator (radii in `M`):

  | `a/M` | KS poles | KS equator | KS equator − `a` | harmonic poles | harmonic equator | harmonic equator − `a` |
  |---|---|---|---|---|---|---|
  | `0` | `2` | `2` | `2` | `1` | `1` | `1` |
  | `0.7` | `1.714` | `1.852` | `1.152` | `0.714` | `1` | `0.3` |
  | `0.9` | `1.436` | `1.695` | `0.795` | `0.436` | `1` | `0.1` |

- **Constants of the run buy exactness, and the hole's velocity has to be
  one of them.** With `M` alone, `H_a = −k_tt k_ta/(2M)` makes every
  non-spinning Kerr-Schild hole *at rest* an exact stationary solution,
  wherever it is. For Kerr it is off by `Σ/r² = 1 + a² cos²θ/r²`
  (`1.39` on the axis at the horizon at `a = 9/10`), and `|a|` as a second
  constant does not repair it: a point on the axis at the horizon
  (`H = 1/2`, `r = 1.436`) and a point on the equator at `r = 2 M`
  (`H = 1/2`), the second hole rotated so that the null vectors agree,
  have the same `g_ab` and `Γ_a` differing by `1.39`. No scalar constant
  repairs a boost either. `g − η = 2H l l` is null and of rank one, and a
  boost along `l` only rescales it, so a hole at rest at distance `r` from
  a point and a hole of the same mass moving along the line through that
  point, at rest-frame distance `r D²` (`D = √((1 − v)/(1 + v))`), give the
  point the same `g_ab` and `Γ_a` differing by `D⁻³` — `2.53` at
  `v = 3/10`. With the spin 4-vector `S^a` (`(0, a⃗)` in the hole's rest
  frame, `a⃗ = J⃗/M`) and the 4-velocity `u^a` the source is exact for every
  boosted, spinning Kerr-Schild hole, at any position:

  ```
  Γ_a = k(u,u) k_ab u^b / (M + √(M² − (S^a k_ab u^b)²)),    H_a = −Γ_a
  ```

  This is the rest frame's `Γ_a = 2(M/Σ) l_a` with `l_t = 1`, read off the
  metric — `H = k(u,u)/2`, `l_a = k_ab u^b/(2H)`, `a cos θ = S·l`, and
  `M/Σ = H/r` with `r` the outer root of `H r² − M r + H (S·l)² = 0` —
  rewritten so that nothing divides by `H`. It was checked for boosts in
  general directions, with the spin both along and across them, at
  `v ≤ 7/10` and `a ≤ 9/10`, to 23 digits. On a solution the root's
  argument is `M² (r² − a² cos²θ)²/Σ²`, which vanishes only at
  `r = a |cos θ|`, inside the horizon; off a solution it is clamped at
  zero, and in the core it is masked like everything else.
- **What that source would change here.** It is algebraic in `g`, so the
  principal part is the one GHSO2 analysed, and `∂_a H_b` is the chain rule
  through the `∂_a g` the kernel already forms (`∂_t g` from `Π`): a few
  hundred flops a point instead of the `Hsrc` field set, its nested-dual
  sampling and its re-sampling after every regrid — and instead of the one
  right-hand side [Possible extensions](#possible-extensions) prices an
  in-kernel analytic source at. It mentions no position, so the gauge
  source moves with the hole: a boosted Kerr-Schild hole is stationary in
  its own frame without a time-dependent `H(t, x)`, which would lift the
  refusal of a moving non-harmonic background. Kerr-Schild at `a = 9/10`
  is the chart in which step 5's analytic core already fits
  (`r_+ = 1.436 > 0.9`), with eight times the harmonic chart's room at the
  equator — so this is a route to the proof-of-concept case at the spin
  this document is named for, in a chart that needs neither the fit nor
  `h ≲ 5/1024`.
- **What it does not settle.** (1) **Stability.** The stationary solution
  is the sampled source's, but the linearisation is not — `∂_a H_b` now
  couples to the perturbation through `∂F/∂g` — so the first measurement is
  the static Kerr-Schild hole under `F(g)` against the same hole under the
  sampled `H(x)`, the error and the constraints to `50 M`. (2) **Drift.**
  `M`, `S` and `u` are constants of the run; if the evolved hole's
  parameters move (truncation error, junk), the source is slightly wrong
  and the coordinates drift, without a constraint violation. (3) **One
  hole.** A binary's superposition is not exact, and the constants are per
  hole. (4) **The velocity.** A chart whose `g − η` is not null could read
  the rest frame off the metric: Painlevé–Gullstrand Schwarzschild, with
  its horizon at `2M` too, has `η^{ab} k_ab = −2M/r` and, by a quick
  calculation not checked further, fixes the rest frame from the metric at
  a point outside `r = M/2`. There is no closed form for it here, and
  Kerr's version (Doran's coordinates) is not worked out.

**Built (2026-10-02)**: the source above, `KerrSchildSource` behind
`Val(:algebraic)` — see [Gauge and constraint
damping](#gauge-and-constraint-damping) — and measured first on the static
hole on the octant ([single holes on the
octant](SINGULARITY_HANDLING.md#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07)).

**Proposed (2026-10-02):** build the source above as an alternative to the
sampled `Hsrc` behind the kernel's existing gauge-source `Val`, measure (1)
on the static Kerr-Schild hole, and then run a boosted Kerr-Schild hole
across the mesh with the analytic layer, against G5's harmonic `a = 7/10`
rows; the damped harmonic gauge stays the extension for binaries, where no
constants of the run make the data stationary.

**Excision, reopened (2026-10-05, Erik's decision; nothing is built).** Can
a static hole be excised on this mesh instead of driven by a layer? The
design is under [Excision](#excision-added-2026-10-05): one-sided closures
per stencil at a lego surface well inside the horizon, computed in a zone
kernel beside the unchanged main kernel, with no target. `PLAN.md`'s steps
answer it in order:
- **X1**, the host-side analysis: the closure weights, the outflow margins
  on the offset surfaces of every chart, and a one- and a two-dimensional
  model of the closure. The two-dimensional model is the go/no-go — the
  inflow-like closures of a lego staircase are a fraction `r_E/(2M)` of
  them. **(Answered in step X1: go, for Kerr-Schild `a = 0`.** The
  per-axis closures with `:msn` dissipation are stable on the lego circle
  at every `r_E` from `M/2` to `7M/4` at `ε_KO > 0`, and X3's depth window
  at `h = 1/16` is `r_E = M/2 … 13M/8`. Spinning holes are not covered:
  their faces with the shift pointing into the excised set are unstable on
  the frozen line. See [Excision](#excision-added-2026-10-05).**)**
- **X2a**, the stencil provider: the right-hand side's physics in one copy,
  bit for bit.
- **X2b**, the `:excised` variant.
- **X3**, the static Kerr-Schild hole on the octant at `h = 1/16 … 1/32`
  against the `:damped` and `:fitted` rows of "Robust stability on the
  octant", the gauge drift to `50 M` included. **(Answered in step X3:
  feasible.** No stability boundary between `r_E = M/2` and `13M/8`; at
  `r_E = M/2` without the blend the exterior is `:damped`'s at every `h`
  and better than `:fitted`'s; no gauge drift to `50 M`. The proposed next
  round is the static spinning hole. See
  [Excision](#excision-added-2026-10-05), "What step X3 measured".**)**

Spinning holes wait for the rotating octant on `main`; moving holes wait
for the static answer. **(Answered for Kerr-Schild `a = 3/5` in steps
X4–X7:** the rotating octant is on the integration branch (X4), the faces
where frame dragging turns the shift into the excised set take X5's rule
(X6), and the excised static hole at `r_E = 4M/5` is the `:damped` layer
outside the horizon from `h = 1/32` on, with the layer's drift of `J` (X7).
Open: a lego corner at the pole of a shallow surface is unstable (X7's depth
scan, `r_E = 17/15` at `h = 1/24`), which `a = 9/10` and the moving hole
must not meet; X7's recommendation is to fix it first, then `a = 9/10` on the
tracked surface. See [Excision](#excision-added-2026-10-05), "What step X7
measured".**)**
