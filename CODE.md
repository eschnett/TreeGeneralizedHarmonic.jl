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
the milestones in [`PLAN.md`](PLAN.md) stop where the proof is made.

The physics is not new to this author's packages: the formulation, the
pointwise algebra, the constraint damping, the gauge sources and the
dissipation-and-damping recipe for a black hole are those of
`GeneralizedHarmonicSecondOrder2` (GHSO2 below), validated there on
SBP-SAT spectral elements. Those repositories are unpublished and not
citable, so the documents this design rests on are **copied verbatim
into [`notes/`](notes/README.md)** and cited from there. What is new is
the mesh: finite differences with ghost zones on an octree of uniform
blocks, a global time step, regridding driven by an error indicator, and
— through the mesh — threads, GPUs and MPI (TreeAMR's M7), without this package writing any of that itself.

Markers: **(proposed)** is
a decision this document makes and still wants confirmed;
**(predicted)** is a number a milestone will measure; **(open)** points
at [Open questions](#open-questions). What this package needs from
TreeAMR is under [Upstream prerequisites](#upstream-prerequisites).

**Settled:** the expanded form of
the momentum equation, not the flux form; three spatial dimensions only;
no excision — inside the horizon the solution is *driven to the analytic
one* by a pointwise damping layer and frozen around the singularity,
with nothing in that decision knowing about blocks or levels; the
proof-of-concept target is a **single black hole with nonzero boost and
spin**; time integration through IMEXRungeKutta's RK4 by block owner; the analysis quantities
(constraints, horizon location, area, mass and spin) are part of the
deliverable; refinement is driven by an **error indicator**, not by
prescribed spheres; `Float64` on Symmetry's H200 is the device
requirement and `Float32` on a device is desirable, not required; checkpoint and
restart, see [Checkpoint and
restart](#checkpoint-and-restart); GPU kernel efficiency is a later
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
  - [Reflecting faces: symmetry planes](#reflecting-faces-symmetry-planes)
  - [The rotating octant: a quarter turn about `z`](#the-rotating-octant-a-quarter-turn-about-z)
  - [The quadrant: a hole moving along `x`](#the-quadrant-a-hole-moving-along-x)
  - [The rotating quadrant: a hole moving along its spin axis](#the-rotating-quadrant-a-hole-moving-along-its-spin-axis)
  - [The bitant: a hole moving across its spin](#the-bitant-a-hole-moving-across-its-spin)
  - [The interior: a pointwise damping layer](#the-interior-a-pointwise-damping-layer)
- [Initial data and backgrounds](#initial-data-and-backgrounds)
- [Refinement and regridding](#refinement-and-regridding)
- [Time integration](#time-integration)
- [Analysis quantities](#analysis-quantities)
- [The coupling contract with TreeGRRMHD](#the-coupling-contract-with-treegrrmhd)
  - [What this package provides (GH-1)](#what-this-package-provides-gh-1)
  - [What this package takes (GH-3)](#what-this-package-takes-gh-3)
  - [The verbs (GH-4)](#the-verbs-gh-4)
  - [Why 3+1 variables at the vertices](#why-31-variables-at-the-vertices)
- [Public access to the vertex metric](#public-access-to-the-vertex-metric)
- [Precision, threads, devices](#precision-threads-devices)
- [Running distributed](#running-distributed)
- [I/O and viewers](#io-and-viewers)
- [Checkpoint and restart](#checkpoint-and-restart)
- [Upstream prerequisites](#upstream-prerequisites)
- [File layout](#file-layout)
- [Milestones](#milestones)
- [Measured results](#measured-results)
  - [What the suite costs, and where](#what-the-suite-costs-and-where)
  - [The suite's tiers, and where each runs](#the-suites-tiers-and-where-each-runs)
  - [Robust stability on the octant](#robust-stability-on-the-octant)
  - [Single black holes: recommended settings](#single-black-holes-recommended-settings)
  - [The analysis cadence on an H200](#the-analysis-cadence-on-an-h200)
  - [The right-hand side on an H200](#the-right-hand-side-on-an-h200)
  - [The right-hand side on a CPU](#the-right-hand-side-on-a-cpu)
- [Binary black holes (proposed)](#binary-black-holes-proposed)
  - [What the single holes hand over](#what-the-single-holes-hand-over)
  - [A case without a background](#a-case-without-a-background)
  - [The gauge](#the-gauge)
  - [One interior per hole](#one-interior-per-hole)
  - [The tracks](#the-tracks)
  - [The common horizon and the handover](#the-common-horizon-and-the-handover)
  - [Initial data](#initial-data)
  - [The outer boundary and the wave zone](#the-outer-boundary-and-the-wave-zone)
  - [The symmetric domains](#the-symmetric-domains)
  - [Refinement for two holes](#refinement-for-two-holes)
  - [The waveform](#the-waveform)
  - [The record](#the-record)
  - [What it costs (predicted)](#what-it-costs-predicted)
  - [Validation](#validation)
  - [What this asks Erik to decide](#what-this-asks-erik-to-decide)
- [A comoving grid frame for orbiting holes (proposed)](#a-comoving-grid-frame-for-orbiting-holes-proposed)
  - [Why the tracked target is delicate](#why-the-tracked-target-is-delicate)
  - [The three parts](#the-three-parts)
  - [The frame](#the-frame)
  - [The gauge in the frame](#the-gauge-in-the-frame)
  - [The interior at rest](#the-interior-at-rest)
  - [The control acts on the map](#the-control-acts-on-the-map)
  - [The plunge and the merger](#the-plunge-and-the-merger)
  - [What the frame retires and what it keeps](#what-the-frame-retires-and-what-it-keeps)
  - [What the frame costs (predicted)](#what-the-frame-costs-predicted)
  - [The ladder](#the-ladder)
  - [What is not known](#what-is-not-known)
  - [What the frame asks Erik to decide](#what-the-frame-asks-erik-to-decide)
- [Harmonic gauge for the inspiral (proposed)](#harmonic-gauge-for-the-inspiral-proposed)
  - [What harmonic gauge removes](#what-harmonic-gauge-removes)
  - [What the package has measured in this chart](#what-the-package-has-measured-in-this-chart)
  - [What harmonic gauge costs](#what-harmonic-gauge-costs)
  - [The data](#the-data)
  - [The merger](#the-merger)
  - [On which grid](#on-which-grid)
  - [The ladder of this route](#the-ladder-of-this-route)
  - [What is not known on this route](#what-is-not-known-on-this-route)
  - [What this route asks Erik to decide](#what-this-route-asks-erik-to-decide)
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
  that would start it. Binaries, with the gauge,
  the boundary and the waveform they need, are designed under [Binary black
  holes](#binary-black-holes-proposed), with milestones G7 and
  G8 beyond the proof.
- **GPU-efficient, meaning memory-bandwidth-frugal.** GHSO2's goal
  statement carries over verbatim: reduce storage and memory bandwidth
  as far as the formulation allows. That is why the second-order-in-space
  form with 20 evolved fields is kept, not the 50-field first-order
  reduction, and why the momentum equation is discretised in the form
  that needs the fewest ghost layers. What the design does *not*
  promise is an efficient GPU *kernel*: a GPU has few registers,
  especially for `Float64`, and a straightforward kernel of this size
  spills. Kernel efficiency is a **later research project**; the design commits only to a kernel structure that does not
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

- **Vacuum, unless a coupled code supplies the matter.** This package
  evolves no matter and no scalar field of its own. The reduced source has
  the algebraic slot where a stress-energy term enters, and TreeGRRMHD's
  GH-3 fills it from `(ρ, S_i, S_ij)`, through `gh_rhs!`'s `matter`
  ([The coupling contract with
  TreeGRRMHD](#the-coupling-contract-with-treegrrmhd)).
- **Second order in space, first order in time** (see
  [The equations](#the-equations)). The fully first-order
  Lindblom–Scheel–Kidder–Owen–Rinne system is not implemented: it was
  the candidate cure for an excision instability that GHSO2 resolved
  inside the second-order form, and this package does not excise.
- **Three spatial dimensions only**. TreeAMR is `D`-generic
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
- **No singularity handling**. Nothing is excised. Where a
  run has a black hole, the right-hand side inside the horizon is
  modified point by point: driven toward the analytic solution in a
  layer well inside the horizon, switched off around the singularity
  (see [The interior](#the-interior-a-pointwise-damping-layer)). This
  restricts black-hole runs to spacetimes with a known analytic
  solution — which is the proof-of-concept target, and exactly what an
  excision milestone would be measured against. The restriction is lifted by the *generic*
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
  binary data from an elliptic solver is an extension. Proposed under [Initial data](#initial-data) for binaries:
  superposed Kerr-Schild, then XCTS through a file.
- **Checkpoint and restart are in scope**: G5's runs — `38`–`149 h` for
  harmonic `a = 9/10`, eleven hours for step 8′'s crossing — outlast a
  queue's day, so `evolve!` checkpoints and restarts; see [Checkpoint and
  restart](#checkpoint-and-restart).
- **No mesh machinery.** Trees, ghosts, interpolation, regrid transfer
  and reductions are TreeAMR's. Point interpolation from a field set, for
  the horizon finder, went upstream — TreeAMR's M11 — and what this package keeps is the order `q + 2` and the footprint
  guard, as a TreeAMR `Region`.
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
RK4 (from IMEXRungeKutta, as in
TreeAMR's own examples), the chunked regrid-and-restart driver with
cases as data, the `observer` hook, `precision.jl` and `device.jl`, the
Löhner refinement indicator with its global noise floor and two
thresholds, the thread-workload digest test, the `[sources]` pin to
TreeAMR's `main`, and the spec-first workflow with measured numbers
recorded in this file.

**Inconsistent:** the `[sources]` pin to TreeAMR's `main` listed above contradicts `CLAUDE.md`, "Things that will bite" (its first item), which says TreeAMR comes from the General registry and not from a `[sources]` pin.

What **changes** because the mesh changes:

| GHSO2 (SBP-SAT elements) | here (TreeAMR blocks) |
|---|---|
| gradient / divergence as two-pass kernels with face traces | centered stencils reading ghost zones; one fused kernel per RHS |
| flux-conservative `∂_i F^i` for the energy estimate | the expanded form with compact second-derivative stencils |
| SAT penalties for boundaries; excision faces | ghost-filling boundary hooks; a pointwise damping layer inside the horizon instead of excision |
| `ε μ⁻⁵ D⁶` dissipation normalised by the SBP spectral radius | the standard Kreiss–Oliger operator of order `q + 2` on a uniform block |
| one mesh for the whole run | error-driven regridding that follows the hole; a fresh problem per chunk |
| Tsit5 / Vern6–9 matched to the element order; a native stepper | RK4, fixed step: IMEXRungeKutta's, by block owner |
| `Float32`/`Float64`/`Float64x2`, CPU/Metal/CUDA | `Float64` on CUDA as the requirement; the rest inherited from TreeAMR |

## The equations

The vacuum Einstein equations in generalized harmonic form, **second
order in space, first order in time** (`notes/methods-ghso2.md`,
"Formulation", and `notes/formulation.md` for the symmetrizer and the
well-posedness argument). The evolved state per point is 20 fields:

    h_ab = g_ab − η_ab                                    (10)
    Π_ab = (√γ/α)(∂_t − β^i ∂_i) g_ab = √γ n^μ ∂_μ g_ab   (10)

in GHSO2's packed order `(tt, tx, ty, tz, xx, xy, xz, yy, yz, zz)`.
The weight on the normal derivative is `√γ`, not `√|g|`: with `n^μ = (1/α, −β^i/α)` and `√|g| = α√γ`, `√|g| n^μ ∂_μ
g_ab` is `α Π_ab`. `notes/methods-ghso2.md` (equation (10) and the sign
of the source) and `notes/pointwise-ghso2.jl`'s header write `√|g|`; the
first form, `(√γ/α)(∂_t − β^i ∂_i) g_ab`, is the one the code evolves, and
the source's sign argument holds with `√γ`, since `√|g| g^{tν} ∂_ν g_ab =
−√γ n^ν ∂_ν g_ab = −Π_ab`.
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

**The expanded form**. On a finite-difference mesh the
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
can use.

Both hold. On the six backgrounds of the table
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

The shapes follow, and the kernel is written against them. `metric_derivatives` returns the *full*
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

One forest, vertex-centered field sets (for a
finite-difference scheme, TreeWave's argument — restriction along a
stagger is injection, exact for any data — applies unchanged):

| set | `nvars` | `G` | state vector | at a regrid |
|---|---|---|---|---|
| `U`, the state `(h, Π)` | `20` | `q/2 + 1` | yes | `U => schedule`, `PointValue`, prolongation `p = q + 2` |
| `Hsrc`, gauge source `H_b` and `∂_a H_b`, non-harmonic backgrounds only | `20` | `0` | no | `Hsrc => nothing`, then re-sampled: a function of position |
| `diag`, constraints, speeds, masked errors, the indicator, on demand | small | `0` | no | `diag => nothing` |

`diag` holds the characteristic speed,
the four components of `C_a`, the ADM Hamiltonian, the three components of
`ℳ_i`, and the `1`/`0` mask indicator the masked norms divide by. `C_a`
and `ℳ_i` occupy contiguous runs on purpose — `block_mapreduce` reduces a
contiguous range of variables and a device cannot be handed an arbitrary
index vector cell by cell. The masked error, the
interior residual and the indicator sit beside them.

The field sets are **vertex-centered and `GHProblem` refuses anything
else**. The table above already decided it; what made it
an assertion is that the masks and the interior profiles turn an owned
index into a position — `point_position`, written to reproduce TreeAMR's
`coordinates` expression bit for bit — and a staggered set would be
evaluated half a cell from where its values sit.

`q` is the finite-difference order (below), and `G = q/2 + 1` because
the Kreiss–Oliger operator of order `q + 2` reaches one point further
than the derivatives do. TreeAMR's vertex invariant `N ≥ 2G + 2` then
puts `N ≥ 8, 10, 12` at `q = 4, 6, 8`; the tests use `N = 8` where that
allows it and `N = 10` at `q = 6` (`FieldSet` refuses `N = 8` at `q = 6`), the proof-of-concept runs `N = 16` or `32`. The working array is
`((N + 2G + 1)/N)^3` times the state: 2.6 at `N = 16, G = 3`, 1.7 at
`N = 32`, which is the price of wide ghosts on small blocks and the
reason `N = 32` is expected to be the default **(predicted; G6 measures
it against the RHS throughput)**.

**The `diag` slots are a contiguous-range interface.** `block_mapreduce`
  and `mesh_fold` read an integer or a *contiguous* range of variables and refuse
  anything else — a device cannot be handed an arbitrary index vector cell
  by cell, which is why the runs of `C_a` and `ℳ_i` must not move.

**The shared upper plane.** The stored size is `N + 2G + 1` per axis, and the
  point at `N + 2G + 1` belongs to no owned range: `map_blocks!` never
  reaches it and the Dirichlet hook or a neighbour's ghost exchange fills
  it.

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
read once per point as a per-block coefficient. One weight
vector therefore serves every refinement level. Both are `@generated`, so
the `Rational{BigInt}` construction happens while the method compiles and
what it emits is each weight's exact numerator and denominator as `Int`
literals with one division between them — an `SVector` a device kernel
holds in registers, with no rational and no `BigInt` anywhere in it, and at
`Float64` and `Float32` no division either once LLVM has folded it. The
conversion is emitted rather than performed in the generator for a reason
given under [Measured
results](#measured-results).

The mixed derivative has no weights of its own — it is the product of two
first-derivative vectors, and the sum over the *second* axis is the inner
one: the two orders are equal in exact arithmetic
and differ in the last place in floating point, so which one it is belongs
to the operator and not to the implementation. Measured on
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
leaves (the same split as the derivative weights
above). So `Q_d u = (ε/h_d) · (the contraction)`, and two consequences
are measured rather than asserted (in `Rational`, exactly): the
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
none. `ε` is a case parameter — a number, or a profile of the
distance to the hole that rises from the horizon inward
(`HorizonDissipation`; measured under [The
interior](#the-interior-a-pointwise-damping-layer), where it is off by
default).

Both halves hold on the mesh. The gauge wave at
`q = 4` converges at **4.12** with `ε_KO = 0.5` against **3.95** without
it, so the term is `O(h^{q+1})` as claimed and costs the scheme nothing;
and white noise of amplitude `1e−8` on flat space, over a thousand steps
— eighteen crossings of the box — falls to **0.66** of its initial L2
norm with `ε_KO = 0.5` and **grows by 9.2** without it, in L∞ by **30.5**.
The kernel compiles the dissipation away entirely when `ε_KO = 0`
(`Val(false)`), which is worth about a fifth of an evaluation
(1106 against 1409 ns per point at `q = 4`), and a run with
`ε_KO = 1e−300` gives the same numbers as one with the term compiled out.

**What the dissipation has to do inside a horizon (`test/dispersion.jl`).** The discrete scheme is not causal at the grid
scale, and the dissipation is the only thing that stands in for causality
there. Frozen at a point, with the lower-order terms dropped, the principal
part of every component along a direction `x` is

    (∂_t − b ∂_x)² u = a² ∂_x² u,      b = β^x,   a = α √γ^{xx},

**with the advection's sign the code's**, `∂_t h = +β^i ∂_i h`, so that the
characteristic speeds are `−b ± a` and both are negative inside the horizon
— for `KerrSchild(1, 0)` along a radius `b = H/(1+H)` and `a = 1/(1+H)`
with `H = 2M/r`, so `b = 0.571`, `a = 0.429` at `r = 1.5 M`, which the
script reads from the metric through `metric_quantities` and checks. With the package's own weights on
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

**The spinning holes (`test/dispersion.jl` sections 1b
and 1c).** The tracked geometry keys the layer on the found horizon's offset surface
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

`gauge.jl` has
`HorizonDissipation`, the `ε_KO(r)` profile (`ε_out` at and outside
`r_h,min`, rising `C²` to `ε_in ≤ 4` at `r_1`), which `GHCase` takes in place of a
number — `ε_KO` is a type parameter, `dissipation_rate` the
identity on a number and `has_dissipation` the kernel's `Val{DISS}`;
`with_dissipation` and `horizon_dissipation` build it from a case.

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

The prediction holds, on the gauge wave
(`A = 1/20`, one wavelength cubed, periodic) at `q = 4` on the two-level
mesh — a `2³` root grid with **one root block refined**, so that each of
its six faces is a coarse-fine face — held **frozen** while `N` runs over
`8, 10, 12` (15 leaves at every resolution, every spacing shrinking with
`N` and the block layout unchanged), a sixteenth of a crossing at
`cfl = 1/4`. Both the mesh and the
resolutions are a budget: TreeWave's `wave_forest`
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
why its runs are as short as a clean rate allows. The suite prints the
fraction on every run and asserts nothing about it: it
is the ratio of two timings on a shared machine, and on GitHub's runners it
has crossed one at `p = 6` and reached 1.19 at `p = 4`
([history](HISTORY.md#the-interface-order-rule-and-what-it-costs-a-second-order-system)).

### One right-hand-side evaluation

TreeAMR's contract, three steps and nothing between them:

    scatter!(U, u)                                             # (0) state → working array
    fill_ghosts!(U, schedule; boundary = dirichlet(case, t))   # (1) copies, restrictions, hook, prolongations
    map_blocks!(gh_rhs_kernel!, U, statearray(du, U), U.work, Hsrc.work, …)   # (2)

Step (2) is **one fused kernel** per owned point, and
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

1. **The head.** Load `h` and `Π` at the point, form the thirty `∂_i h` and the
   coefficient set.
2. **`∂ₜh`, complete with its dissipation,** stored at once.
3. **The source**, [`gh_node_source_lean`](#the-right-hand-side-on-an-h200),
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
H200](#the-right-hand-side-on-an-h200).

The principal part is the same scalar wave operator for every
component, which is what makes step 5 a loop with a small live set;
the coupling between components is entirely in the coefficients and
the source. Recompute rather than store is the rule throughout: the
effective stencil weights (metric coefficient times finite-difference
weight) are formed where they are used, and nothing that can be
rebuilt from `h` and `∂_i h` is kept. The pointwise algebra is
scalarised straight-line code — GHAccel found that the compiler does
not register-allocate a large `SVector` expression well unless every
element is spelled out, and generated its kernel for that reason; this
package does the same, by hand. The
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
split into a stencil kernel (the stencils, `du` written) and an algebra
kernel (the source, `du` accumulated), costing one extra read and write of
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
per evaluation. The interior's `Val` carries the
*variant* rather than a `Bool` — `:none`, `:damped`, `:pasted`,
`:frozen`. `PLAN.md` calls it "has interior"; it has to say *which* as
well, because the three variants differ inside the kernel, so a `Bool`
would have needed a second parameter beside it.

**Inconsistent:** the list of variants above (`:none`, `:damped`, `:pasted`, `:frozen`, "the three variants") leaves out `:fitted`, which [The fitted target](#the-fitted-target), item 10, says `INTERIOR_VARIANTS` has.

Measured on the H200, in [The right-hand side on an
H200](#the-right-hand-side-on-an-h200):

- The black-box attempts are worth little: the workgroup shape under 10 %; the
  split 6 % once the source fits in registers.
- What the kernel needed was inlining forced on the device and a register-lean
  spelling of the source: 8.5 → 1.2 ns a point.

The kernel is *block-local*: it reads its own block's stored points and
nothing else, so it runs on every backend unchanged. Per-block spacings
and origins travel to the backend once per chunk, as TreeWave's spacings
do; the origins are new here, because the interior profiles, the masks
and the damping profile need positions inside the kernel.

The RHS never mutates `u`, and it is a pure function of `(u, t)`
(TreeAMR's contract): the interior treatment below is a term *in* the
right-hand side, not a write to the state.

The kernel agrees with
`gh_node_rhs_expanded` on analytic data, on three backgrounds and at
`q = 2, 4, 6` with and without dissipation, to **under 1e−12** of the size
of the terms, and Minkowski's `du` is **exactly** zero at every order.

Two things the writing settled that the design had left open:

- **The `∂_t g` the source is given is the accumulated `∂ₜh`, dissipation
  included**. Two accumulators per component is what
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

**On the CPU the kernel evaluates `W` points at a time.**
The algebra above is generic in its number type, so on the CPU step (2) runs it on
SIMD.jl's `Vec{W,T}`: `W` neighbouring points along the first axis, every stencil a
contiguous vector load and every store a vector store. The launch is unchanged —
`map_blocks!` over every owned point — and the item at the start of each group of
`W` does the group's work; a row that `W` does not divide ends in a group that
overlaps the one before it and stores the shared points again. In a hole's evolved
region a group with a point in the layer or the core runs the scalar code point by
point. `W` is chosen by `GHProblem` — four `Float64` on AVX2 and aarch64, eight with
AVX-512, one on a device. Why and what it buys: [The right-hand side on a
CPU](#the-right-hand-side-on-a-cpu).

The kernel's body is `gh_rhs_point!` (one point) and `gh_rhs_lanes!` (a
group). `W` is the sixth kernel `Val`, `1` on a device or for a software
type, and `simd_width = 1` asks for the scalar kernel, also through
`evolve!`. The seventh, `HASMATTER`, says whether a matter adapter is
passed ([What this package takes (GH-3)](#what-this-package-takes-gh-3)).

### The time step

    dt = cfl · minimum_spacing(forest) / λ_max,
    λ_max = max over owned points of  α √(tr γ^{ij}) + |β|

GHSO2's conservative bound on the coordinate characteristic speed, taken
once per chunk from a speed slot in `diag` (a kernel writes it,
`mesh_mapreduce(max)` reduces it)
and re-checked at the chunk's end as
TreeHydro does — throw, do not warn, if the step used violated the bound.
`cfl = 1/4` **(proposed** default, GHSO2's**)**.

`λ_max` is exactly `√3` on flat space — the
conservative bound, not the physical speed 1 — and larger wherever there
is a shift. Every run measured here took its step from it at `cfl = 1/4` and
none needed a smaller one: the gauge wave at `q = 2, 4, 6`, shifted
Minkowski through a Dirichlet face, and a thousand steps of noise. The speed
kernel, `max_speed` and `gh_dt` are in `evolution.jl`.

Two things a hole adds. First, **the speed
kernel takes the mask like every other analysis kernel**: the frozen core holds data that
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

A moving hole's step is sized from `λ` times the
square of the growth the previous chunk measured **and never by
less than 1 %**: once the hole crosses a mesh finer than step 8f's capsule
the growth is not monotonic, and on the uniform `5/128` control of the
boosted `a = 0` hole the speed grew `0.27 %` in chunk 4 after a quieter
chunk 3 — the recheck stopped the run at `0.75 M` with a CFL number of
`0.20003` against the requested `0.2`. The margin costs
1 % of the steps of a moving run and nothing on a static one, whose step is
unchanged bit for bit; the recheck stays **(proposed)**.

`driver.jl` sizes a moving hole's step from `λ` times the
square of the previous chunk's growth `λ_end/λ` (a static hole's step is
unchanged bit for bit): the fastest speed of a hole crossing the box grows
by 0.1–0.3 % a chunk, and the CFL recheck stopped every boosted row at
`cfl = 1/4` and at `1/5` without it.

## Gauge and constraint damping

**Prescribed gauge sources**. `H_a(x)` is sampled from the
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
- **A time-dependent *sampled* gauge source is not supported.** Boosted
  Kerr-Schild has `H_a(x − vt)`, which a per-chunk sample cannot
  represent and an in-kernel evaluation (nested duals per point per
  evaluation) would price at about one RHS. The driver refuses a
  non-harmonic background that is not static when it is given no source,
  and says why. A boosted Kerr-Schild hole runs
  with the algebraic source below, `KerrSchildSource` with the hole's lab
  velocity, which moves with the hole: `kerr_schild_case(; v, x0,
  gauge_source = :algebraic)` builds it (`test/boosted_tests.jl`).

**How the two questions are answered** (`src/gauge.jl`), because
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

**The algebraic Kerr-Schild source** ([history](HISTORY.md#the-prescribed-and-algebraic-gauge-sources)). A case may carry
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
The closed form is `gauge_source_grad`'s `H_a`
and `∂_a H_b` to `2–4·10⁻¹⁵` for Kerr-Schild at rest (`a = 0`, `0.9`),
translated, and boosted along `x` and in a general direction with spin
(`test/gauge_source_tests.jl`). On the static `a = 0` hole (octant `[0, 3]³`,
uniform `h = 1/16`, `q = 4`, the `:damped` layer, `t = 2 M`) it runs as the
sampled source does — masked error `1.44·10⁻⁵` against `1.11·10⁻⁵`, `ℋ`
`1.81·10⁻⁵` against `1.93·10⁻⁵`, `C_a` `3.87·10⁻⁶` against `3.96·10⁻⁶` — in
`62 s` against `90 s`; `cfl = 1/2` reproduces `cfl = 1/4` to six digits at
every row to `4 M`.

**Constraint damping**: the Gundlach–Pretorius term `Z_ab`,
with `γ0(x)` a *function of position* — a Gaussian of width a few `M`
around the hole's analytic center, tapered to a small value in the wave
zone, an `isbits` closure over `(center(t), M)` in the problem — and a
constant `γ2 > −1`. `γ0 ≈ 1/M` near the hole is GHSO2's measured
requirement for a stable evolution with the horizon in the domain.

The profile (`src/gauge.jl`) is
`γ0(x) = far + (near − far) exp(−r²/2w²)` with `r = |x − c(t)|`, and the
three numbers are `near = 1/M`, `far = 1/(10 M)`, `width = 3 M`.
A Gaussian rather than a compactly supported bump because nothing depends
on it vanishing exactly — `far` is the wave zone's rate, not zero — and
because `C^∞` costs one `exp` per point either way. Two consequences for
the code around it: `GHCase`'s `γ0` field now holds a *profile* and not a
number (a bare number is wrapped in `ConstantDamping`, so every
flat-space case is the arithmetic it was), and the
right-hand-side kernel therefore forms the point's position
unconditionally — three fused multiply-adds, which the compiler drops
where the profile is constant and there is no interior.

**On a star, `1/M` is not the recipe.** The near-hole value grows a mode
inside a star. TreeGRRMHD found it on its TOV star (its `PLAN.md`, GH-8):
`R/M = 6.85`, flat at `γ0 M ≤ 1/10`, growing from `1/4`, doubling every
`25 M` at `1`, at a rate independent of `h`. This package's Plummer star
shows the same mode. The measurement below is of 2026-11-11 (`test/star_runs.jl`,
`test/star_study.jl`), with the matter held fixed (GH-3's anti-Cowling setting), a
sampled `H_a`, `ε_KO = 1/2`, `q = 4`, `h = R/16`, the box `(−2R, 2R)³`, and
constant `γ0` and `γ2`, to `500 M`. The table gives the e-folding time of
the largest change of the state over the last half of the run, in `M`
(`∞`: it does not grow); the change itself at `500 M` is in parentheses.

| `γ0 R` | `R/M = 5` | `R/M = 10` | `R/M = 20` |
|---|---|---|---|
| 0.5–2.5 | ∞ (2e−6) | ∞ (1e−6, 2e−6) | ∞ (7e−7, `γ0 R = 2`) |
| 5 | 4400 (1e−5) | ∞ (6e−6) | ∞ (3e−6) |
| 10 | 68 (0.3) | 434 (6e−5) | 2036 (1e−5) |
| 20 | levels off at 0.4 | 86 (7e−3) | 520 (7e−5) |
| 40 | | 68 (0.15) | 245 (5e−4) |
| 80–100 | | levels off at 0.16 | 160 (3e−3, `γ0 R = 80`) |

`γ0 = 0` stays at `1e−6` on `R/M = 10`.

- **The onset is fixed in `γ0 R`, between 5 and 10, on all three stars.** In
  units of `M` it moves by a factor of four with the star. The box scales with
  `R` here, so `R` and the box's crossing time are not separated. The mode peaks
  at the centre and is thirty times smaller at the box's faces, and GH-8 found
  that a box twice as wide does not change the interior. The growth rate above
  the onset is not a function of `γ0 R` alone: at `γ0 R = 10` the e-folding is
  `14 R`, `43 R` and `100 R` from the most to the least compact star.
- **The threshold does not transfer between stars.** The TOV star's onset is
  between `γ0 R = 1` and `1.7`, five times lower than Plummer's at a similar `R/M`. The TOV
  star has a surface, and its truncation error converges at second order there
  (GH-8: `2.5×` for `h` divided by `1.6`). A rule for a new matter case has to
  be measured on that case, as TreeGRRMHD's `1/(10M)` was.
- **The rate is a property of the equations, and the truncation error seeds
  the mode.** At `γ0 M = 1` on `R/M = 10` the e-folding is 428, 434 and
  `441 M` at `h = R/12`, `R/16` and `R/24`. The amplitude at `250 M` falls by
  3.1 and 5.0 between the three, against `3.16` and `5.06` at order four.
- **What grows is the diagonal spatial metric at the centre.** The change is
  in `h_xx = h_yy = h_zz`, largest within `R/4` of the centre, and falls
  smoothly to the box. `Π` is smaller by about the growth rate, which makes
  this a growth without oscillation, not a wave.
- **`γ2` decides it.** `γ2 = −1/2` holds the change flat at `γ0 R = 10` and at
  `γ0 R = 40` (`4e−6`, `7e−6` at `500 M`), as on the TOV star. `γ2 = +1` at
  `γ0 R = 10` is a different mode: it is in `Π`, at `r ≈ R`, with an e-folding
  time of `10 M`, and the change reaches 1 at `110 M`.

The suite's star (`star_case` in `test/matter_tests.jl`) runs at `γ0 = 1`
with `M = 1/10` and `R = 1`: `γ0 M = 1/10` and `γ0 R = 1`, on the flat side of
the onset. It is the table's `R/M = 10`, `γ0 R = 1` row, which stays at `1e−6`
to `500 M`. (GH-8 calls it `10/M`, which it is not.) On a Plummer star the
hole's `1/M` stays flat only for `R/M ≲ 5`. Any other matter case should use
`γ2 = −1/2` or measure its own threshold.

## Boundaries

### Periodic

Free, through the tree. The gauge wave and the robust-stability tests are
periodic in every dimension. **Shifted Minkowski is not**: its profile `ψ′(x) = A sech²(x/w)` is a function of `x` alone and is
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

The hook writes the analytic state into the
outer ghosts and into the shared upper plane **bit for bit** — the same
numbers `state_tuple` gives on the host at `coordinates(fs, b, idx)`,
because TreeAMR's boundary kernel builds its position the same way
`coordinates` does — and a run of shifted Minkowski through it converges
at order `q` (3.93 in L2 at `q = 4`), which is the statement that a
Dirichlet face costs the scheme nothing. `dirichlet(case, t)` returns
`nothing` where the case is periodic in every dimension, so the
right-hand side branches once on a `Bool` the problem stores rather than
handing `fill_ghosts!` an argument whose type depends on the case.

For G5's own background, `boost(Harmonic(1,
7/10), 0.3 x̂)`, the hook built at `t = 7/10` writes the analytic state
into the outer ghost planes, the shared upper plane and a corner **bit for
bit** (`test/moving_tests.jl`), where the same points at `t = 0` differ by
more than `10⁻³`: the time dependence of a hole that moves reaches the
boundary through the hook's `t` and nothing else.

**Hooks depend on time.** `dirichlet(case, t)` is built at each call.
  It goes to `fill_ghosts!` inside the RHS, to `regrid!`, and to
  `adapt_to_initial_data!`, each with that call's `t`. Forgetting the
  second is the bug that arrives one chunk late; passing a stale `t` is
  the bug that arrives as a boundary reflection.

### Reflecting faces: symmetry planes

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
planes. In `test/reflection_tests.jl` a
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

**A hole on the octant** (`hole_case(; octant = true)`)
needs one that is its own mirror image in all three planes — at the origin,
at rest, `a = 0` — and refuses any other: a spin along `z` keeps only the
`z` mirror and a boost breaks its own axis.
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

Beside `hole_case(; octant = true)` there are `minkowski_octant_case` and
`hole_forest(; shape = :cube)` for nested boxes.

**An octant's corner is in the core, and its stale data is not of definite
  parity**: the core rule maps the origin to a point of the
  sphere `r_0` off the walls, so it holds `h_tx ≠ 0` on a wall. Never project
  or perturb the core; `add_noise!(; exclude)` skips it.

### The rotating octant: a quarter turn about `z`

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

In `test/rotation_tests.jl`, Kerr-Schild at
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

`gh_forest` passes a case's `rotating` on to the forest, and `hostcopy`
keeps a field set's rotation, as it keeps its parities.

### The quadrant: a hole moving along `x`

A non-spinning hole on the `x` axis, moving along it, is its own mirror image
in `y = 0` and in `z = 0` at every time, and in nothing else: the boost breaks
`x → −x`. `hole_case(; octant = :quadrant)` evolves the quarter
`[−L, L] × [0, L]²` of the box, a quarter of its cost, with those two faces
reflecting and the other four Dirichlet (the hook at the current time, so the
data are the moving hole's). It refuses any other hole — off the axis, moving
across it, or spinning (a spin along `z` keeps only the `z` mirror, one along
`x` neither) — and says why. The parities are the octant's `state_parity`,
and nothing else is new at the walls. The box is twice as long as it is
wide, so `gh_forest` takes a root brick `(2r, r, r)`: `roots` is a count or
three counts, and the *cells* must be cubes, `widths[d]/roots[d]` one number.

A hole that crosses the box without a regrid needs its finest level along the
whole path. `hole_forest(; path = Δc)` sweeps each shell along the segment
`[c, c + Δc]` — for `shape = :cube` the box `(min − R, max + R)` along the
path and `(c − R, c + R)` across it, for `:ball` the capsule — built once
with `Δc = v t_end`; with `path = nothing` the shells are the old ones, bit
for bit. It is the frozen hierarchy of the convergence protocol, and it is
not cheap: radii `32, 16, 8, 4` along a travel of `8 M` on `L = 64` are 856
blocks. `test/octant_runs.jl octant=quadrant` measures everything about the
hole — the shells, the noise's excluded core, the finder's guess, the
boundary check — about `c(t)`, and splits each shell into its trailing and
leading half.

In `test/boosted_tests.jl`, the boosted
Kerr-Schild hole (`v = 1/5` from `x = −1/4`, `:damped` at `4/M`, the algebraic
source, `q = 2`, uniform `h = 5/48`) on the quadrant `[−5/2, 5/2] × [0, 5/2]²`
against the box `[−5/2, 5/2]³`: after 16 RK4 steps the two agree to
`3.8·10⁻¹¹` at every point outside the core, the moving layer's target
rate's roundoff ([the rotating quadrant](#the-rotating-quadrant-a-hole-moving-along-its-spin-axis); [history](HISTORY.md#the-quadrants-mirror-test)). As for the octant,
only a uniform quadrant is the box's restriction; a refined one is a
different, symmetric discretization of the same problem. The runs are under
[the boosted Kerr-Schild hole at binary
speed](SINGULARITY_HANDLING.md#the-boosted-kerr-schild-hole-at-binary-speed-2026-10-20).

### The rotating quadrant: a hole moving along its spin axis

A Kerr hole spinning about `z` and moving along `z` has no mirror plane: the
motion breaks `z → −z`, the spin `x → −x` and `y → −y`. It is its own image
under a quarter turn about `z` at every time, though, which is the rotating
octant's seam without its mirror. `hole_case(; octant = :rotating_quadrant)`
evolves `[0, L]² × [−L, L]`, a quarter of the box, with the seam `(1, 2)`
gluing `x = 0` to `y = 0` and Dirichlet data at the four faces `x = L`,
`y = L` and `z = ±L`. It needs that hole — on the `z` axis, at rest or moving
along it, spinning about `z` (Kerr-Schild or harmonic, any `a`, unrotated,
translated or boosted along `z`) — and refuses any other, because a spin or a
velocity across `z` would turn with the quadrant and run there as a different
problem. `kerr_schild_case(; axis = 3)` builds the hole (`axis` is the
coordinate it sits and moves on; `axis = 1` is the quadrant's hole bit for
bit), `gh_forest` takes the root brick `(r, r, 2r)`, and `hole_forest(; path)`
sweeps the capsule along `z`. Nothing at the seam is new: the state turns by
`state_rotation`, both seam planes are evolved, and the finder needs nothing of
its own. `test/octant_runs.jl octant=rotating_quadrant` records `J`'s
component along the spin, `J n̂_z` with `n̂` the finder's spin axis (`hz_J`;
the magnitude on every other domain), and the horizon's extent along `z` over
its extent across, against `√(1 − v²) r₊/√(r₊² + a²)`.

In `test/boosted_tests.jl`, Kerr-Schild at
`a = 1/5` moving along `z` at `v = 1/5` from `z = −1/4` (`:damped` at `4/M`,
`r_0 = 2/5`, `r_1 = 11/10`, the algebraic source, `q = 2`, uniform
`h = 5/48`) on `[0, 5/2]² × [−5/2, 5/2]` against the box `[−5/2, 5/2]³`,
after 16 RK4 steps, at every point outside the core. At rest the two agree to
`3.6·10⁻¹⁴` (`0.4 ε` relative). Moving they differ by `1.3·10⁻¹⁰`, growing
linearly with the steps (`2.5·10⁻¹²` after one), on the seam planes and the
axis in the core and the inner layer, and the same without spin (`1.9·10⁻¹⁰`
at `a = 0`). That is the box's, not the quadrant's: the analytic background is
four-fold symmetric only to `56 ε`, and the moving layer's target rate, a
centered difference over `δ = ε^{1/3} r_0`, divides that asymmetry by `δ` and
injects it every step, while the quadrant's ghosts across the seam are the turn
exactly. The claim bounds the difference by `nsteps · dt` times the rate's
measured asymmetry. The `x` quadrant's mirror test meets the same mechanism
([the quadrant](#the-quadrant-a-hole-moving-along-x)). The runs are under [a spinning hole along its
axis](SINGULARITY_HANDLING.md#a-spinning-hole-along-its-axis-2026-10-21).

**`hz_J` is a component on the rotating quadrant and a magnitude elsewhere**:
  `test/octant_runs.jl` writes `J n̂_z` there (`n̂` the finder's
  spin axis) and the Korzyński `|J|` on every other domain, which cannot go
  negative or show a flipped axis; `records.csv`'s `J` is the magnitude
  everywhere. On the `a = 1/2` runs the two agree to the CSV's ten digits.
  Say which one a comparison reads.

### The bitant: a hole moving across its spin

A Kerr hole spinning about `z` and moving along `x` is its own image under the
mirror `z → −z` and under nothing else. A mirror keeps a velocity in its plane
and a spin (an axial vector) along its normal. A rotation keeps only the vectors
along its axis, so no rotation holds a spin and a velocity that are not
parallel. In particular TreeAMR 0.2.5's half-turn seam (M14) holds a spinning
hole only when the spin and the velocity lie on the turn's axis — the aligned
case, where the rotating quadrant's quarter turn already does better. A turn
about `x` reverses a spin along `z`, and one about `z` reverses the velocity.
`hole_case(; octant = :bitant)` evolves `[−L, L]² × [0, L]`, half the box, with
the face `z = 0` reflecting and the other five Dirichlet. That is twice the
points of the quadrants. It needs a hole in the plane `z = 0`, at rest or moving
in it, spinning about `z` (Kerr-Schild or harmonic, any `a`, unrotated,
translated or boosted in the plane), and refuses any other.
`kerr_schild_case(; a, v, x0)` with `axis = 1` builds the hole, `gh_forest`
takes the root brick `(2r, 2r, r)`, and `hole_forest(; path)` sweeps the
capsule along `x`. The parities are the octant's. `test/octant_runs.jl
octant=bitant` records `J`'s component along `z` as `hz_J`, as the rotating
quadrant does.

**The least horizon radius across the spin.**
`horizon_min_radius` of a boosted hole was the rest frame's least radius times
`√(1 − v²)` in every direction. That is exact along the spin, or without
spin. Across the spin it is far too low: the contraction acts on the
equatorial radius `√(r₊² + a²)` along `v`, and the pole's `r₊` is unchanged.
At `a = 9/10`, `v = 1/2` the bound was `1.244` where the horizon's least radius
is `1.436`, and it refused `r_1 = 5/4`. Both Kerr charts put the horizon on a
spheroid, and a boost maps it to an ellipsoid, `A ↦ Λ A Λ` with
`Λ = I + (γ − 1) v̂ v̂ᵀ` (`horizon_ellipsoid`). Its shortest semi-axis is now the
least radius, and the old bound is kept bit for bit where it was exact. So
across the spin `r_1 = 5/4` is 13 cells inside the `a = 9/10` hole at `1/72` at
every speed. Along the spin the contracted pole at `v = 1/2` forces `9/8`,
which grows at rest ([aligned spin and
velocity](SINGULARITY_HANDLING.md#aligned-spin-and-velocity-which-holes-the-algorithm-holds-2026-10-22)).

In `test/boosted_tests.jl`, Kerr-Schild at
`a = 1/5` moving along `x` at `v = 1/5` from `x = −1/4` (`:damped` at `4/M`,
`r_0 = 2/5`, `r_1 = 11/10`, the algebraic source, `q = 2`, uniform
`h = 5/48`) on `[−5/2, 5/2]² × [0, 5/2]` against the box `[−5/2, 5/2]³`: after
16 RK4 steps they agree to `7.6·10⁻¹¹` of `127` at every point outside the
core, which is the moving layer's target rate's roundoff, as on the quadrants.

From [spin across the
motion](SINGULARITY_HANDLING.md#spin-across-the-motion-the-bitant-2026-10-24):
Kerr-Schild `a ∈ {0, 1/2, 9/10}` × `v ∈ {0, 1/5, 1/2}` along `x`, 23 rows on
one H200 each to `20 M`: every row is stable, and every cell converges at
order four outside the horizon except `a = 1/2` at `v = 1/2`, which is not yet
in that regime at `1/32`. Without spin, and at rest, the bitant agrees with the
rotating quadrant to a few per cent. Moving across the spin adds an excess near
the horizon that falls at order `7–8` from `1/24` to `1/32`.
The source of a hole moving across its spin has the rest frame's spin 4-vector,
`(0, 0, 0, a)`, since `v·ŝ = 0`.

### The interior: a pointwise damping layer

#### The design

Inside the horizon the right-hand side is modified point by point, and the
modification is four independent choices; everything below this summary is
the history that made each of them, kept because its numbers are the
evidence (step 5's design is the first row of each and stays as the
analytic control; [history](HISTORY.md#the-design)):

1. **The geometry — where the layer is.** Either step 5's **sphere** about
   the analytic center, `r_1 ≤ r_h,min − m h` (`Interior`), or the
   **tracked offset surface**, the depth `d = r_h(n̂) − m h − |x − c(t)|`
   below the horizon the finder found about the center it found
   (`FittedSpec` in the case, a `FittedInterior` rebuilt every chunk). The
   sphere is the offset surface's `l = 0` case bit for bit. The tracked
   surface is the only geometry for a hole whose center or shape is not
   known in advance, and the only one whose offset surface can hold harmonic
   Kerr's flat singular disk inside an oblate horizon.
2. **The target — what the layer relaxes toward.** Either the **analytic**
   solution (`:damped`, `:pasted`; an optional `target` metric for studies)
   or a **fit** of the evolved state on the offset surface,
   continued inward as a polynomial in `x` and cached on the grid
   (`:fitted`). The analytic target needs the chart's singular set inside
   the core surface; the fitted one needs nothing inside, and is the only
   target for harmonic Kerr at `a = 7/10`, G5's chart (the analytic
   core there cuts the disk at every `h` the proof of concept can afford).
   The fit is made in `(log α, β^i, γ_ij, Π̃_ab)`, `Π̃ = (α/√γ)Π`.
3. **The ramp — how wide.** `ρ` rises from `0` at the offset surface to
   `ρ_max` over `n_L = max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)` cells, `w` turning
   over in the inner half (measured at `q = 2`, **proposed**
   for other orders); a margin of `m ≥ G + 1` cells (the stencil margin)
   and, for grid-scale leakage, as many more as the leakage margin's path integral asks
   ([the margin](#the-margin)).
4. **The rate — how hard.** `ρ_max` is a physical rate, **`4/M`**, not the grid rate `1/dt`, which only
   an exact target survives. **A moving analytic layer wants more**: its core is frozen, a point crosses an eight-cell layer at
   `v = 0.3` in about `M`, and at `4/M` what the core held is relaxed by
   `e^{−2}` before the layer releases it on the trailing side — the boosted
   `:damped` rows end at `1.0–1.5 M` at `4/M` and reach `5 M` at `20/M`
   **(proposed: `ρ_max ≳ 20/M` for a moving analytic layer)**;
   the `:fitted` core relaxes toward a target that moves with the track and
   reaches `5 M` at `4/M` ([the measurement
   matrix](SINGULARITY_HANDLING.md#the-generic-interior-the-measurement-matrix-step-8f)).

**Inconsistent:** item 4 says a moving analytic layer's core is frozen and
proposes `ρ_max ≳ 20/M` for it, while [the analytic
layer](#step-5s-layer-the-analytic-control) below (its table, and "Why touch
the interior") says a moving `:damped` core relaxes toward `u_exact` and the
moving layer adds its target's rate, and `CLAUDE.md` ("A relaxation toward a
moving target needs the target's rate") says `4/M` is the moving hole's
setting again.

**What the measurement matrix decides.** The `:fitted`
target on the tracked geometry holds every hole the package has that the
mesh resolves — Kerr-Schild `a = 0` to `50 M`, `a = 9/10` and harmonic
`a = 0` to the ends of their rows (`20 M`, `10 M`), harmonic `a = 7/10` at
`h = 5/256` to `10 M`, the boosted hole to `5 M` — so **excision (step 8g) is not needed**. It is not
free: where the analytic target exists it is the better one — `2×` the
masked error and `2.8×` the shell's `C_a` on the static Kerr-Schild hole,
`40×` on the spinning one — so **the analytic `:damped` layer stays the
default wherever the chart's singular set fits inside the core, and
`:fitted` is for the charts where it does not**, G5's included.

**The runs behind these choices are in
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md)**,
and the settings they recommend for a single hole at `a ≤ 9/10` are under
[Single black holes: recommended
settings](#single-black-holes-recommended-settings).

#### Step 5's layer: the analytic control

**No excision**. Inside the horizon the solution is not left
to the Einstein equations alone: in a layer well inside the horizon it
is *driven to the analytic solution*, and around the singularity the
evolution is *switched off*. Both are decisions made **point by point**,
as functions of the distance `r = |x − c(t)|` to the hole's analytic
center `c(t) = c_0 + v t`, and neither knows anything about blocks,
ghost zones or refinement levels. (Or, for a case that asks for it, as functions of the *depth* below
the tracked horizon's offset surface about the *tracked* center — "The
tracked geometry" below, of which this section's sphere is the `l = 0`
case.) The right-hand side at every point is

    ∂_t u = w(r) · F(u)  −  ρ(r) · (u − u_exact(x, t))                (INTERIOR)

with `F` the GH right-hand side `(EXPANDED)` including the dissipation,
`u_exact` the background's `(h, Π)` at `(x, t)`, and two smooth
profiles. (For a moving hole the `:damped` layer
adds the target's rate where `F` is switched off,

    ∂_t u = w F(u)  +  (1 − w) ∂_t u_exact  −  ρ (u − u_exact)

— the `:fitted` layer's `target_rate` for the analytic target —
because without it the moving solution is not a solution of `(INTERIOR)`;
"Why touch the interior" below.)

| region | radii | `w` | `ρ` | what happens |
|---|---|---|---|---|
| evolved | `r ≥ r_1` | 1 | 0 | the Einstein equations, untouched |
| damping layer | `r_0 ≤ r < r_1` | 1, ramping to 0 at its inner edge | 0 at `r_1`, rising smoothly to `ρ_max` | the equations plus a relaxation toward the analytic solution (and, for a moving `:damped` hole, `(1 − w) ∂_t u_exact`) |
| frozen core | `r < r_0` | 0 | 0 (`ρ_max` for a moving `:damped` hole) | nothing: `du = 0`, `F` not evaluated; a moving `:damped` core relaxes toward `u_exact` at the core rule's point, following its rate in the outer half `r > r_0/4` |

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

**Inconsistent:** this paragraph gives `r_h,min` under a boost as the rest
frame's value times `√(1 − v²)`, while [the bitant](#the-bitant-a-hole-moving-across-its-spin) above says
that bound is far too low across the spin and that the least radius is now
the boosted ellipsoid's shortest semi-axis (`horizon_ellipsoid`).

#### The margin

**The margin `m` does two jobs, and they have different sizes
(proposed).** Inside the horizon the continuum lets nothing out, but the
discrete scheme lets grid-scale content out, held back by the dissipation
alone ([Kreiss–Oliger dissipation](#kreissoliger-dissipation)), and every interior treatment is a source of such content at `r_1`.
So there are two rules, both **(proposed)**:

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
what binds (frozen-coefficient).** Toward the disk
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
of what the interior makes, which the inexact targets measure
([the layer for an inexact target](#the-layer-for-an-inexact-target)); so the
rules are stated and **the default `m = 8` is not changed**.

**`ε_KO` rising inside the layer does not buy margin.** The leakage the margin is for is made at or outside `r_1` and crosses
`r_1 ≤ r < r_h`, where a profile that rises only inside the layer still has
the exterior's value: in the one-dimensional model to `10 M`, `ε_in = 1, 2,
4` inside the layer changes the transmission from every depth by at most
1.2 %. What does act is dissipation raised *across the margin* — `C²` from
`ε_out = 1/2` at the horizon to `ε_in` at `r_1`, and held inside: at
`ε_in = 4` it cuts the transmission from depth 8 by **4–15×** (`q = 2, 4`,
`λ = 2h, 4h`), five to nine cells' worth at the long-time rate, and does not
help a source within two cells of the horizon; doubling `ε_KO` everywhere,
measured in 3D, cuts it by 1.7–2.5×. The recommendation is
therefore to start the `ε_KO(r)` profile's rise **at the horizon rather
than at `r_1`** — the region is causally disconnected from the exterior in
the continuum, and the dissipation is `O(h^{q+1})` whatever `ε` is — with
`ε_in ≤ 4`, inside RK4's real-axis limit of about `6` on the 3D corner mode
at `cfl = 1/4` (`3 ε dt/h ≤ 2.8`); and, if `n_e` must exceed about 5 at this
resolution, to widen the margin as well, since neither lever alone is
enough.

**A hard step inside the horizon does get out, at the rate the leakage
study measured.** `:pasted` onto `KerrSchild(6/5, 0)` (E0) holds a 20 %
step — `|δh| = 0.348` — at `r_1`, `10.9` cells below the horizon, on the leakage study's
uniform 512-block mesh, and the first shell outside the horizon carries,
against the same run with the exact target, **`4.2e−3` of it at `2 M` and
`2.2e−2` at `5 M`, still rising** at `ε_KO = 1` everywhere; `0.35–0.39`
e-folds per cell outside. That is the leakage study's *measured* attenuation — its ripple
from depth 8 continued at its own per-cell rate puts `2.6–3.1e−3` at
`10.9` cells at `2 M` and `ε_KO = 1/2`, where E0 has `2.5e−3` at `1.5 M` — and
140× the frozen-coefficient `e^{−d/ℓ_max(r_1)} = 3e−5`. At `ε_KO ≤ 1/2`
everywhere the step ends the run first (`1.0 M`, `1.75 M`), in the evolved
shell next to it; with the `ε_KO(r)` profile rising from `1/2` at the horizon
to `ε_in = 2` or `4` at `r_1` it survives, with a masked error L∞ of `8.1` and
`6.3` against `244` at `ε_KO = 1` everywhere, and transmits **no less at
`2 M`** (`2.6e−3`, `3.5e−3`) and 1.4–2.3× less by `5 M` (`1.6e−2`,
`9.5e−3`) — not the 4–15× of the leakage study's one-dimensional model. So `m = 8` holds a
discontinuity back to a percent or two over a few `M`, and nothing holds the
exterior together against one for long: the answer to a discontinuous
interior is a smooth one, rule 2 of [the layer for an inexact
target](#the-interior-a-pointwise-damping-layer), whose smooth wrong targets
leave the exterior indistinguishable from the exact one — and **the default
`m = 8` stays**.

#### The spinning harmonic chart

**A ball cannot hide Kerr's singularity in the harmonic chart at
`a = 9/10`.**
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
ways out, neither built and neither chosen here **(open question)**:

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

**Inconsistent:** this paragraph calls `boost(Harmonic(M, 9/10), v)` the
proof-of-concept case and its way out an open question, while `README.md`'s
[Status](README.md#status) says G5 runs at `a = 7/10` and
[the design](#the-design) above
names the tracked offset surface as the geometry that can hold harmonic
Kerr's flat singular disk inside an oblate horizon.

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
(Not for the boosted hole: where `w < 1`,
`w F(u_exact) = w ∂_t u_exact` falls short of `∂_t u_exact` by `(1 − w)
∂_t u_exact`, and the layer trails a moving target by about `|∂_t
u_exact|/ρ`: the boosted Kerr-Schild hole at `v = 1/5` had a layer residual
of `6.5` at `4/M` at every `h`, and an error just outside the horizon
`28–44×` the resting hole's at `1/16` (`SINGULARITY_HANDLING.md`, "The
boosted Kerr-Schild hole at binary speed"). The moving `:damped` layer
therefore carries `(1 − w) ∂_t u_exact`, and the static hole, for which it is
zero, takes the branch without it bit for bit. `∂_t h` is `dmetric`'s time
column from the pass the layer already makes; `∂_t Π` is a centered
difference of that pass at `t ± δ`, `δ = ε^{1/3} r_0`
(`background_state_rate`) — `10⁻¹⁰` of `|∂_t Π|`, and generic in the target,
where `ddmetric` would put a second nested dual pass in every interior
kernel. Its roundoff, `ε^{2/3}`, is what a moving hole's quadrant and box
now agree to.)

**Why a relaxation and not a slowed evolution alone** (the pure mask `ρ = 0` was the first proposal and stays as a
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

**Why a smooth layer and not a hard paste** (the
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
the ramp widths, evaluated per point in the kernel. **`ρ_max = 4/M`**, `M` the hole's mass parameter: the layer's
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
hole the suite evolves the default is `0.036/dt` to `0.092/dt`, so the refusal is a statement that the mesh does not
resolve `M/4`.

**The grid rate `ρ_max · dt = 1` stays as the option `ρ_max_factor`**, and
is not what a run gets by default ([history](HISTORY.md#the-interiors-profiles-and-rate)):
`1/dt` is a *grid* rate, about `107/M` on the suite's
fixture at `cfl = 1/5`, and a paste two cells deep, which every inexact
target ends its run against on every ramp up to eight cells, and even on
the exact solution it ends a `50 M` run with six times the error of `4/M` —
the `G`-point shell's `C_a` L2 `0.181` against `0.029`, the masked error
`0.160` against `0.027` ([the layer for an inexact
target](#the-interior-a-pointwise-damping-layer) below). `ρ_max_factor = 1`
reproduces step 5's numbers, which is what it is kept for; `ρ_max_fixed`
is any other rate, and the two are refused together.

**What the default costs the layer.** A relaxation
at a fixed rate holds the layer to the analytic solution to `τ/ρ_max`, `τ`
the truncation error of `F` there, where the grid rate held it to `τ · dt`:
on the suite's fixture the `:damped` residual — the layer's L∞ distance
from the truth — saturates at `1.42` by `1/2 M` against the grid rate's
`6.2e−2`, a few percent of the solution itself at the layer's inner edge,
where `|Π|` is `60`, and it converges at order `q` (`2.03` over `N = 6, 8,
10`) where the grid rate's extra `1/h` made it `2.81`. The exterior does not
see it: the masked error outside `r_1` is `2.26e−3` against `2.27e−3` at
`1/10 M` and `1.51e−2` against `1.61e−2` at `1 M`, and at `50 M` it is the
row above, `0.027` against `0.160`. The residual is a statement about the
layer's own health and not about the solution, and at `4/M` it is the
larger one.

The smoothstep (`src/interior.jl`) is the
quintic `10s³ − 15s⁴ + 6s⁵`, whose value, first *and* second derivatives
match the constants it joins. The two ramp widths are **halves of the
layer**, measured from opposite ends: `w` rises from
`0` at `r_0` to `1` at the layer's midpoint and stays there, `ρ` falls
from `ρ_max` at the midpoint to `0` at `r_1`. They are complementary, so
no point is both frozen and undamped, and the outer half of the layer is
the unmodified equations *plus* a relaxation — which is what makes the
data the evolved stencils reach into the analytic solution to truncation
order. The smoothstep **clamps its result as well as its argument**: the polynomial has a triple root at `s = 1`, and
its Horner form at `s = 1 − 2⁻⁵³` returns `1 + 1.3e−15`, so without the
clamp `w` would exceed one just inside `r_1` and amplify `F` where this
section says the equations are untouched.

The grid rate `ρ_max · dt = 1` is the option
`ρ_max_factor = 1` — it is what reproduces step 5's tables, which `CODE.md`
keeps as history — and `hole_runs.jl`'s `calibration` (`:grid`) and `bounds`
sections ask for it by name.

**Inconsistent:** the sentence above says `CODE.md` keeps step 5's
tables as history; "The design" says the runs
behind the interior are in `SINGULARITY_HANDLING.md`.

#### The layer for an inexact target

**The layer for an inexact target.** The fitted target is
a fit of the evolved state, not a solution, so the layer is calibrated
against targets that are wrong on purpose, with three knobs and no new
`Val`: an `Interior` carries an optional `target` metric (`isbits`,
default `nothing` = the case's background, resolved by `layer_target`)
that the kernel reads where `(INTERIOR)` reads `u_exact` — the layer branch
and the `:pasted` overwrite — and nowhere else, so the initial data, the
Dirichlet hook, the gauge source and the error reference stay on the truth
and the record's `residual` is the layer's distance *from the truth*;
`evolve!(…; ρ_max_fixed)` relaxes at a rate in the case's units instead of
`ρ_max_factor/dt` (the two are refused together, and a fixed rate above
`1/dt` is refused at the chunk that would take it, **(proposed)**; a run
with neither relaxes at the default `4/M`, and
the grid rate is asked for as `ρ_max_factor = 1`); and `HorizonDissipation`,
the `ε_KO(r)` profile below. The targets:
**E1** `KerrSchild(6/5, 0)`, a valid metric that is not a solution; **E2**
`translate(KerrSchild(1, 0), (0, δ, 0, 0))`, `δ = h, 4h`, a tracking error;
**E3** the solution with `h_tt` off by `−2(r − r_1)²χ(r)/M²`, value and slope
right at `r_1` and curvature wrong by `4/M²` (the sign is **(proposed)**: `+2/M²` drives the target's `α²` through zero in any layer of
eight or more cells, which is finding 2's non-metric and not a curvature
error). The ramp's width `n_L` is the width over which `ρ` rises from `0`
at `r_1` to `ρ_max` — `ρ_ramp = 1`, `r_0 = r_1 − n_L h`, `w` turning over
in the inner half **(proposed**, since finding 1's prediction is
a statement about that width**)**. On the suite's fixture (Kerr-Schild
`a = 0`, `q = 2`, `h = 5/64`, `cfl = 1/5`, `ε_KO = 1/2`, the range projection
on) to `50 M`, the numbers under [Measured
results](SINGULARITY_HANDLING.md#the-layer-against-an-inexact-target-step-8c)
decide:

1. **`ρ_max` is a physical rate, `4/M`, and the
   default.** At the grid
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
2. **The ramp is at least `4G = 8` cells at that rate.** At `n_L = 8` and `12`, `ρ_max = 4/M`, every target the layer can
   contain — E1, E2 at `δ = h`, E3 — holds the hole to `50 M` with the
   `G`-point shell's `C_a` L2 at `0.029–0.039` and the masked error L2 at
   `0.027–0.032`, both flat from `10 M` on, `M_irr` within `0.3 %` of `M`,
   and no projection hit: **the same numbers as the exact target on the same
   layers** (`0.029`, `0.027`), so the exterior does not see what the layer
   relaxes toward. At `n_L = 6` it costs 1.3–4× (`0.039–0.115`), at `4`
   more. Step 5's layer on the exact solution at the grid rate ends at
   `0.181` and `0.160`.
3. **Finding 1's `n_L ≳ G (10 ρ_max M)^{1/3}` is confirmed for `ρ_max M ≥ 4`
   and is not sufficient below it.** It asks for `6.8`
   and `9.3` cells at `4/M` and `10/M`; the scan's thinnest ramp that holds
   is `8` and `12`, and one row thinner costs 2–4× in the shell (`n_L = 6` at
   `4/M`, `8` at `10/M`). At `1/M` it would allow `4.3` cells, and the
   ramps of 4–6 cells are 2.5× the best while the 12-cell one fails from
   the inside: the rule is the prediction **and** `ρ_max ≳ 4/M`. Stated for
   other orders and spacings — `n_L = max(4G, ⌈G (10 ρ_max M)^{1/3}⌉)` cells
   at `ρ_max = 4/M`, `4/M` being `16 κ` — it is **(proposed)**:
   one fixture, one order, one spacing measured it.
4. **`ε_KO` stays constant for a smooth target.** The
   profile below, raised across the margin as step 8a recommended, makes the
   shell *worse* in proportion to `ε_in` — at `n_L = 12`, `4/M`: `0.029`,
   `0.032`, `0.045`, `0.065` for `ε_in = 1/2` (constant), `1, 2, 4` —
   and changes no survival where the ramp is right; where it is wrong (the
   grid rate at `n_L = 8`) it delays the end from `15 M` to `22 M`. It is
   what keeps a *hard step* alive (E0, above), so it stays built, as the
   remedy for a target that can jump, and off by default **(proposed)**.
5. **The tracking error the layer absorbs is about `h`.** E2 at `δ = h` is the exact target's `0.029` at both good ramps;
   at `δ = 4h = 0.31 M` only `n_L = 8`, `4/M` holds near the rule's error
   (`0.041`), `n_L = 6` at `4/M` and `8` at `10/M` reach `50 M` at 3–4× it
   (`0.160`, `0.123`), the others end in `1–6 M`, and at `n_L = 12` the
   displaced singular point
   lies *on a grid point of the layer* and the run throws at `t = 0` — a
   target's singular set must be inside `r_0` exactly as the background's
   must, and nothing checks it. The tracked center has to be good to
   a cell for this layer to be the rule's **(proposed)**.

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
interpolates it like any other data. **A moving hole's
`:damped` core follows its target.** A core with `du = 0` keeps its `t = 0`
data, the solution about `c(0)`, and when the hole moves its trailing side
releases them into the layer, off the current solution by an amount that
grows with the distance travelled. A boosted Kerr-Schild hole on the
quadrant (`a = 0`, `v = 1/5`, `r_0 = 5/8`, `r_1 = 23/16`, `h = 1/16`, `4/M`)
had a layer residual growing linearly — 6.3, 10.9, 14.8, 19.1, 30, 38 at
`t = 1/2 … 3`, against `0.1` at rest, the trailing side 20× the leading —
and died at `3.2 M`, when the points that were at the center at `t = 0`
left the core (`r_0/v = 3.1 M`); step 8's moving harmonic sphere died the
same way. So for `:damped` with `v ≠ 0` the core relaxes at the full rate,
toward the layer's own target `u_exact(t, core_position(x, t))` at the core
rule's point `c(t) + r_0 n̂` — the layer's `u_exact` at `r_0`, so the two
agree there — and `F` is still not evaluated. **The metric relaxes in the
ADM variables** `(α², β^i, γ_ij)`, `∂_t(α², β, γ) = −ρ_max ((α², β, γ) −
(…)_exact)` mapped to `∂_t h` by the chain rule (`core_relaxation_rate`),
and `Π` linearly. The plain `−ρ_max (u − u_exact)` was tried first and
ended the run in its first chunk: on the center's path the target jumps from
one side of the sphere `r_0` to the other as the center passes, and the
straight line in `g` between Kerr-Schild at `±r_0 x̂` passes through
`η + (2M/r_0) diag(1, n n)`, which is Euclidean (`g_tt = 2.2` at
`r_0 = 5/8 M`) — the constraint monitor's `sqrt` threw there. Positive `α²`
and positive-definite `γ` are convex, so the ADM path is a metric
throughout, and near the target the two rates agree to first order. In its outer half the core carries its target's rate too,
`s(r) ∂_t u_exact(t, c(t) + r_0 n̂)` with `s` rising `C²` from 0 at `r_0/4`
to 1 at `r_0/2` (`core_target_rate`, `core_rate_weight`), a centered
difference along the core rule's moving point. Without it the core lags its
target by `|∂_t u|/ρ_max`, and the points it releases on its trailing side
carry that lag into the layer: on the suite's quadrant (`q = 2`, `h = 5/48`)
the layer's rate alone took the residual after 32 steps from `34.6` to
`12.3`, with the core's `3.3` against the resting hole's `2.6`. Not near the
center: `n̂` turns at `v/r` and jumps where the center's path crosses a grid
point; a point spends `(r_0/2)/v` in the outer half, many `1/ρ_max`, before
it leaves. The metric's part of that rate is
an ADM rate too — the target's `∂_t h` read as `∂_t (α², β, γ)` at the target
and added to the relaxation's before the one chain rule at the state
(`core_relaxation_rate(h, h_t, ρ, ∂ₜh_t)`). Added in `h`, it is the target's
ADM rate mapped at the target instead of at the state, and near the center's
path, where the target turns at `v r_⊥/r²` and the core lags it by a large
fraction of itself, that drove `α²` through zero: the campaign's `a = 0`,
`v = 1/2` hole (`r_0 = 1/2`, `h = 1/16` and `1/32`, rotating quadrant) ended
in its first chunk in the constraint monitor's `sqrt` at a core point
`(2h, h, −3h)` from the center, `α² = −1.2·10⁻³` against its target's least
`0.088`. With the fix the core's `α²` never falls below that least.
**The core's rate:** a moving core relaxes at
`max(ρ_max, κ |v|/r_0)`, `κ = 12` (`core_rate`); the layer keeps `ρ_max`. The
core rule's target jumps where the center's path crosses a point, from the
leading side of the sphere `r_0` to the trailing side — `Π_tt` from `5.2` to
`97.6` at `a = 0`, `v = 1/2`, `r_0 = 1/2` — and no rate follows a jump, so
the core relaxes it, with `r_0/|v|` left before it releases the point on the
trailing side. At `4/M` and `v = 1/2` that is `e^{−4}`: the campaign's layer
residual of `4.7` (rotating quadrant, `r_0 = 1/2`, `r_1 = 19/16`; `|Π| 4.6`,
`|h| 0.9`) is that point, at depth zero on the trailing axis, flat in time and
the same at `h = 1/16`, `1/24`, `1/32`, and a pointwise model of the core
branch gives it (`3.2–4.2` in `Π` near the path, `0.7` at `v = 0.35`, `0.14`
at `1/5`). What the layer did not absorb crossed `r_1`: on `L = 8`, `h = 1/16`,
moving over resting, the error in `[2, 2.25)` grew from `2.6` to `19` by `6 M`
and in `[3, 5)` from `2.9` to `7.6`, and the evolved region's L∞ error was
`0.36` against `0.0047`. At `12/M` the same run stays at `2.8–3.3` and
`2.7–3.2` — the ratio both runs share to `t = 2`, where nothing from the core
has reached the shells yet, and the campaign's ratio at every `h` at `1–2 M`:
the moving hole's own truncation error — with an L∞ error of `0.036`. `κ = 12`
leaves `e^{−12}` of the jump, about `10⁻³`, and keeps the `v = 1/5` families of
`SINGULARITY_HANDLING.md` (`κ |v|/r_0 = 3.8` and `3.2` at `r_0 = 5/8` and
`3/4`) at `ρ_max`, bit for bit; `chunk_interior` refuses a core rate above
`1/dt` (`12/M · dt = 0.21` at `h = 1/16`, `cfl = 1/2`). **What is left of the
residual at `v = 1/2`** (`2.56`, `1.15` and `0.66` at `cfl = 1/2`, `1/4`,
`1/8`, independent of `ρ_max` from `4` to `16/M`) is first order in `dt`: the
point that has just crossed `r_0` on the trailing side. The core's target is
constant along rays, so its rate there is the tangential part of `−v · ∇u`, and
the layer's is all of it; the rate jumps by `(v · n̂) ∂_r u_exact` across `r_0`,
and RK4 integrates a step that holds the crossing to first order. It converges,
and a core target that matches `∂_r u_exact` at `r_0` would remove it — a
change to the core rule, not made. A hole at rest keeps `du = 0`
bit for bit (the kernel branches on `v` first, `core_follows`); `:frozen`
keeps it too, being the variant that is measured to fail, and `:pasted`
overwrites its core anyway. The tracked geometry is not changed: its
`:fitted` core already relaxes, and its center's `v_est` is never exactly
zero. Every norm, monitor and error
kernel — and the refinement indicator — masks the whole interior
`r < r_1`; the modified region is not a numerical solution and must not
be reported as one or refined for its own sake. The apparent horizon
lies outside `r_1` by the margin, and so must the interpolation
footprint of the horizon finder (checked).

Three things the writing of the core rule
settled, each stated where it is made in `src/interior.jl`:

- **At the center the ray is undefined and `+ẑ` is taken
  (proposed)**. It is not an arbitrary tie-break: the harmonic
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
  the constraint monitors do, and what they then report is `NaN`.
- **A masked slot is written through a branch, not multiplied by zero.** Step 4's kernels wrote `keep * value` with `keep`
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

**A tracked `:damped` core is still frozen**: `core_follows`
  is false for a `FittedInterior`, so `variant = :damped` on the tracked
  geometry — the screen before a new `:fitted` depth — dies at
  `r_core/v` on a moving hole, as step 8's `:frozen` did. Screen a moving
  depth for less than that, or with `:fitted`.

#### The range projection

**The range projection: the third and last writer of the state**
(`src/bounds.jl`). Every interior treatment above is a
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
projection, from the stage limiter, is the third. Both limiters are
integrator keywords (`stage_limiter`, `step_limiter`); under
IMEXRungeKutta **both writers are one limiter,
`gh_limiter!` — the projection, then the paste — passed as the stage and
the step limiter alike** (the limiter applies to
every state vector), so every stage value the right-hand side reads and
every step's result is projected and pasted — see [Time
integration](#time-integration). The projection also runs
once on the initial data and once after every regrid transfer, before the
paste, which is the order RK4 applies the two — neither state went
through a stage, and a prolongation into a fresh fine block is unlimited
(TreeHydro's reason for the same call) **(proposed**, for the
initial data**)**.

**Why a stage limiter and not a step limiter**: what it guards against is
`F` evaluated on a stage vector that is not a metric, and a `NaN` in a
stage is a `NaN` in the next stage's `F` at every point whose stencil
reads it. Under IMEXRungeKutta's RK4 four
projections happen a step, each followed by the paste: three as the stage limiter,
on the stage values the right-hand side reads, and the fourth as the step
limiter, on the result.

**Where: the gate.** The projection's output is a clamp, and a clamp is a
kink wherever it fires; a kink an evolved stencil reads is an `O(1)`
right-hand-side error outside the layer. So it runs only at `r < r_gate`,
and `check_bounds_gate` asserts at every regrid, beside the interior's
radius checks, that `r_gate ≤ r_1 − R h` with `R` the stencils' Euclidean
reach in spacings (`G` for `q ≤ 4`, the mixed derivative's `√2 · q/2` from
`q = 6`). The proposed gate is **`r_gate = r_1 − 2 G h`** — twice the reach
— **(proposed**, until step 8a's leakage margin exists to
replace the factor 2**)**; on the suite's fixture that is `0.8375` at
`h = 5/64`, with `8.4 %` of the owned points inside it. The outer part of
the layer, `r_gate ≤ r < r_1`, is unguarded by construction.

**The proposed ranges** (`default_bounds`; a `StateBounds` has no default
for any of them and `hole_case` has none beyond `bounds = nothing`)
**(proposed)**: `α ∈ [1/50, 50]`, `λ(γ) ∈ [1/100, 1000]`,
`|β| ≤ 10`, `K_max = 100/M`, against the step-5 fixture's deepest data —
Kerr-Schild at `r_0 = 2/5 M`, where `α = 0.41`, `λ = (1, 1, 6)`, `|β| = 2.04`
and `max |(α/√γ)Π| = 10.4/M` — so that a healthy interior never fires.
Flat space must be inside every range, and the constructor refuses one
that excludes it: the non-finite repair writes Minkowski's components,
and a repair the next check moves again is not idempotent.

**Idempotent on the state and on the flag.** Every
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
`N = 8` `:pasted`. **It never fires on either**: both
degenerate in the evolved shell just outside `r_1`, where no projection
gated below `r_1` reaches and none should, and the runs with it end bit for
bit where the runs without it do. The prediction that hits would start
"deep, several `M` before the crash" is wrong; the failures are surface
failures at `r_1`, which is step 8c's hypothesis about the layer's
transition, measured from the other side.

**Written back only where it fired, bit for bit.** The range projection keeps a
  healthy quantity's bits — it reassembles `h_tt`, `h_ti` or the spatial
  block only when that block's range was violated, because `(−1 + h_tt) +
  1` is not `h_tt` — and the kernel writes a point only where `hit` came
  back true.

**The unguarded outer layer.** The outer part of the layer, `r_gate ≤ r < r_1`, is unguarded by design, and
  a degenerate metric there still ends a run through the `DomainError` of
  `metric_quantities`' `sqrt`.

#### The tracked geometry

**The tracked geometry** (`src/tracking.jl` and the
second half of `src/interior.jl`). Everything above is keyed on
`r = |x − c(t)|` about the *analytic* center. A case may
instead key its layer on the **found** horizon, `PLAN.md`'s finding 3: with
`r_h(n̂)` the tracked apparent horizon's coordinate radius along the unit
vector `n̂` from the *tracked* center `c(t)`, the layer is a function of the
**depth**

    d = r_h(n̂) − m h − |x − c(t)|,        offset surface r_1(n̂) = r_h(n̂) − m h,

below the offset surface — `d ≤ 0` evolved, `0 < d ≤ n_L h` the layer
(core surface `r_0(n̂) = r_1(n̂) − n_L h`), and beyond it the frozen core.
The sphere is the `l = 0` case, and **a tracked geometry holding step 5's
sphere is step 5's layer bit for bit**: one
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
   grid's, whose value changes per chunk **(proposed)**),
   `core_min = 2`, `lmax_shape = 4`, `ρ_max` (`0` = the driver's default),
   `max_misses = 3`, `α_trigger = 1/10` and a `target`. **Its ramps are
   step 8c's rule, `ρ_ramp = 1`, `w_ramp = 1/2`**, since `n_L` was
   calibrated as the width over which `ρ` rises; step 5's `1/2, 1/2` is the
   fixture's **(proposed)**. A
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
   the side it moves away from, points no layer ever treated **(proposed)**. `track_center(tr) = HoleCenter(c_find − v_est t_find, v_est)`
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
   reported in e-folds, below **(proposed)**. `fitted_interior`
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
   **(proposed; the fitted target shares the ordering and the
   conversion)**. The finder's `hlm` is truncated to `lmax_shape` by
   `ash_resample` first. The kernel evaluates the series
   (`shape_series`) with **no angle**: `Y_lm = q_lm(cos θ) sin^m θ e^{imφ}`
   with `q_lm` the fully normalized associated Legendre function over
   `sin^m θ`, and `sin^m θ e^{imφ} = (n_x + i n_y)^m` — the Chebyshev
   recurrence for `cos mφ, sin mφ` multiplied through by `sin^m θ`, which
   removes both the `atan` and the division by `sin θ` that would need a
   guard on the axis. Runtime loop bound `lmax`, no allocation, generic in
   `T`. **The conversion is `sYlm`'s to roundoff**:
   at 200 random directions the largest error is `3.7` and `6.4` eps of the
   function at `lmax = 4, 8` at `Float64`, `1.6` and `5.3` eps at `Float32`;
   on `EquiangularGrid(6)` it is `ash_evaluate`'s to the same.
   **The surface is the series clamped into `[r_in, r_out]`** — the extremes
   over both poles and a `(4 lmax + 3) × (8 lmax + 6)` grid of directions —
   which makes the two fast paths **exact shortcuts rather than
   approximations** (proposed): outside `r_out − offset` a point
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
   spheroid's higher multipoles into the kept ones **(proposed)**. **What the truncation costs**, the largest
   radius error over directions: **Kerr-Schild `a = 9/10`: `1.2e−3` at
   `lmax = 4`, `8.1e−6` at `8`** (`5.5e−8`, `3.8e−10` at `12`, `16`) — the
   axis and equator and `r_in`, `r_out` against `horizon_min_radius`,
   `horizon_max_radius` to the same; **harmonic Kerr `a = 9/10`: `4.6e−2`,
   `7.2e−3`, `1.1e−3` and `1.7e−4` at `lmax = 4, 8, 12, 16`** — `2.4`,
   `0.37`, `0.057` and `0.009` cells at `h = 5/256` — since its spheroid is
   oblate in the ratio that sets the multipoles' decay (`a²/R² = 4.3` against
   `0.39`). The default `lmax_shape = 4` is Kerr-Schild's (`0.06` cells at
   `h = 5/256`); harmonic Kerr at `a = 9/10` needs `lmax = 12` for a tenth of
   a cell, which step 8f's rows should carry. The
   seed's `r_min` of a **moving** hole is the least of the analytic surface
   over `shape_sample_directions` (both poles included), not
   `horizon_min_radius`'s `√(1 − v²)` bound, which is exact only when the
   smallest radius lies along the boost: G5's hole has it on the spin axis,
   `0.714`, which a boost along `x̂` does not contract, where the bound says
   `0.681`, and the first find (`0.716`) was refused as a jump of `0.034`,
   over `G h/2` at `h = 5/256`. A hole at rest keeps
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
   **(proposed**, over "use the bounding sphere
   `r_in − offset`", which would let a footprint read the layer's outer part
   wherever the horizon is farther than its smallest radius — on harmonic
   Kerr's equator by more than the whole margin**)**; 2000 random
   footprints about an `a = 9/10` shape agree with the definition.

**The protocol both geometries speak.** `in_layer(int,
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

**Per chunk, in the driver.** The initial data goes through the core rule
of the seed's geometry (`fill_exact!(…; interior)`); at every row the find
starts from the track's prediction `c_find + v_est (t − t_find)` and
measures its radii from there, so the row's `center_offset` is **the
track's prediction error**; `update_track`; the next chunk's geometry from
the updated track on this mesh, asserted like the sphere; after a regrid,
the same track on the new mesh. Four decisions the loop needed, each
**(proposed)**:

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

**A boost moves the hole the other way.** `SpacetimeMetrics.boost(m, v)` evaluates `m` at `Λᵀx` with `Λ`'s
`+γv` entries, so the rest frame's origin is at `x = −v t` in the lab: the
metric of `boost(KerrSchild(1, 0), (0.3, 0, 0))` at `t = 1` is singular at
`x = −0.3`, not `+0.3` (measured). It is handled
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
is refused. The analytic
shape's contraction does not depend on the sign.

`evolve!` also takes `find`, the function the horizon rows call —
`find_gh_horizon`, or a test's wrapper that disables it — because the
observer is called after the find and cannot reach it **(proposed)**. The damping profile `γ0` and a `HorizonDissipation` stay on the
case's analytic trajectory: both are smooth on the scale of `M`, not of a
cell **(proposed)**.

**What the margin buys, per row** (`margin_efolds`): step 8a's
path integral `n_e = ∫_{r_1(n̂)}^{r_h(n̂)} ε dr/(h ℓ_max,1)`, along the six
grid axes from the tracked center — where the one-dimensional symbol is
exact — with the background's coefficients, `ε_KO` at each point and the
spacing of the block each point is in; the row carries the least of the
six. It reproduces the script's `1.81` and `2.07` e-folds for `m = 8` at
`h = 5/64` (`q = 2`, `4`) to the digits `CODE.md` records. On the suite's tracked fixture — `m = 10` cells of `5/64` from
`r_1 = 1.22`, of which the outer `0.75 M` lie in blocks of `5/32` — it is
**`1.45`**, where a uniform `5/64` would give `2.68`.

**What it costs** (Apple silicon, Julia 1.13, one
thread). The series is **`21 ns` per point at `lmax = 4` and `79 ns`
at `lmax = 8`**, against **`96 ns`** for one analytic `u_exact` of
Kerr-Schild `a = 9/10` (`background_state`, the dual pass the layer already
pays), and it is evaluated only between the bounding spheres. A
right-hand side on the fixture with the tracked geometry at `lmax = 4` is
**`1.1 %`** dearer than with the sphere, a `0.15 M` run **`5 %`** (four
threads: `3.16 s` against `3.00 s`), of which the host side per chunk is
`fitted_interior` at `0.04 ms` and `margin_efolds` at `0.6 ms`.

**What it measures on the static hole.** One find of
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

**Which center.** A
  `FittedInterior`'s center is `track_center(track)`, `c_find + v_est (t −
  t_find)`, so its masks, its `interior_radius`, the range projection's gate
  and the core rule are about the track, while `case.center` — the damping
  profile, the refinement centroid's offset, `track_offset` — stays the
  analytic one.

**The fixture's tracked margin.** On the step-5 fixture the finest level is the cube `|x|_∞ ≤ 5/4`,
  so `m = 8` puts the offset surface outside it and the geometry is refused
  (`18 h = 2.8 M` needed of a horizon of `2 M`) — the suite's tracked runs
  use `m = 10`.

**Never test a metric *on* its singular point**: whether rounding leaves `r` at `2e−17` or at zero —
  a huge `g_tt` on Apple silicon, `NaN` on the x86 CI runners — is the
  platform's, not the physics'. `interior_tests.jl` asks for
  `g_tt = −1 + 2γ²M/δ` at a transverse offset `δ` instead.

**Inconsistent:** the paragraph "A boost moves the hole the other way"
above says that `test/interior_tests.jl` asserts the seed track of a
boosted case at `(−0.3, 0, 0)` at `t = 1`, "where the boosted Kerr-Schild
metric is singular (`|g_tt| = 1.0e17` there, `4.9` at `+0.3`)"; the
paragraph just above (moved from `CLAUDE.md`, "Things that will bite")
says the test asks for `g_tt` at a transverse offset `δ` instead of on
the singular point.

#### The fitted target

**The fitted target** (`src/fit.jl`, with its kernel half
in `src/evolution.jl`, `src/constraints.jl` and `src/driver.jl`). The
tracked geometry still relaxes toward the
analytic solution, so its core surface must still contain the chart's
singular set, and harmonic Kerr at `a = 9/10` refuses that at every `h`.
The fitted target needs no interior at all: a **polynomial in `x`** —
regular everywhere, the center included — fitted by least squares to the
state (for the layer) or to the analytic solution (for its initial data)
on the offset surface `r_1(n̂) = r_h(n̂) − m h`, where the state is still the
solution, and continued inward. Twelve pieces: seven in the order
`build_fit` meets them, and five for the variant that relaxes
toward it:

**Inconsistent:** this paragraph counts twelve pieces (seven and five), but the numbered list below has thirteen.

1. **The variables are ADM variables, never `g_ab`** (`PLAN.md`'s finding
   2): `(log α, β^i, γ_ij, Π_ab)`, 1 + 3 + 6 + 10 = 20 (`fit_variables`).
   `α = exp(log α)` is positive by construction and `γ` is convex, where a
   least-squares fit — a weighted mean — of `g_ab` is not a metric: **the
   angular mean of Kerr-Schild `a = 0`'s `g_ab` on the sphere `r = 1.15 M`
   has `g_tt = +0.739` and the eigenvalues `(0.739, 1.580, 1.580, 1.580)`,
   Euclidean signature, a signed lapse of `−0.86`, while the same mean of
   the fit's variables reassembles into a metric with `α = 0.60`,
   `det γ = 3.94`** (the control the suite asserts).
   **The shift is the contravariant `β^i` (proposed)**: either is
   valid, but `β^i` is the split's own shift and reassembles without an
   inverse — `β_i = γ_ij β^j`, `h_tt = (1 − α²) + β^iβ_i` (`state_from_fit`)
   — where `β_i` would need `γ^{ij}` at every layer point. `γ_ij` is held as
   its offset `γ_ij − δ_ij` (the same fit, with near-flat digits kept) and
   `Π_ab` as it is. **The samples' radial derivatives go through the chain
   rule** (`β′ = γ⁻¹(b′ − γ′β)`, `(log α)′ = (α²)′/2α²` and their second
   derivatives, `b_i = h_ti`) **for both samplers (proposed)**,
   checked against differencing the converted samples along the same rays:
   they agree to `5.7e−10` in the slopes and `7.3e−11` in the curvatures on
   the fixture's surface.

   **Inconsistent:** this item says the fit holds `Π_ab` as it is, in `(log α, β^i, γ_ij, Π_ab)`; [The design](#the-design) (item 2) and the `fit_tilde` paragraph at the end of this section say the fit holds `Π̃ = (α/√γ)Π` (by default, `fit_tilde = true`), and item 13 below blends in `(log α, β^i, γ_ij, Π̃_ab)`.

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
   `16 eps`. **With `shift_constant = false` the shift has no `ρ⁰ ỹ_00` term**, so `β(0) = 0`,
   as `PLAN.md` asks; its `ρ² ỹ_00` and `ρ⁴ ỹ_00` terms, which also vanish at
   the center, stay, since `l ≥ 1` alone cannot hold a shift's `l = 0` part
   on the surface at all **(proposed)**. **That is wrong
   for a moving hole:** a boosted hole's shift has an
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
   **`shift_constant = true` is the default**:
   the shift keeps its `l = 0` constant, and `false` is the brief's ansatz,
   kept as a switch; on the static hole the constant comes out `6.2e−14` of
   `|β|`.
3. **The samplers** are called as `sampler(xs, ns)` — the points *and* the
   unit rays, since the radial derivative is along the ray from the tracked
   center, which a sampler does not know **(proposed**, over
   `sampler(xs)`**)** — and return `(u, ∂_r u[, ∂_r² u])` as vectors
   of packed `(h, Π)`. `state_sampler(fs, q; t)` interpolates through
   `interpolate_grad` with **the footprint guard off, `mask = AllPoints()`
   at the call**: the collocation points are on the evolved region's
   boundary, so their windows read `G h` into the layer, where step 8c's
   `n_L ≥ 4G` holds the relaxation to `ρ_max · smoothstep(1/4) = 0.10
   ρ_max`, `0.41/M` at the default — the layer's outer quarter is the
   equations plus a weak pull toward a fit of this same state **(proposed)**. It gives no second derivative, so it serves `cont = 1`, and
   `cont = 2` with it is refused. `analytic_sampler(bg, t; δ)` evaluates
   `state_tuple` at five points along the ray and differences them at
   **fourth order, `δ = h/8` (proposed**, central
   differences at the order that puts the truncation at `δ⁴`**)**: its `∂_r
   h` is the analytic gradient's to `1.6e−8` on the fixture; the second difference's roundoff `5 eps |u|/δ²` is `10⁻¹¹` at
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
   **Each block of rows is weighted `P^{−b}`, `P = L + 2 cont` (proposed)**, the growth of a degree-`P` polynomial under `∂_ρ`: it cuts the
   condition number 4–20× (`85.6` and `1.4e3` at `L = 8`, `cont = 1, 2` on
   the fixture, against `377` and `2.9e4` unweighted) and the value residual
   2.5–8× on the spinning holes, and changes no consistent system's solution.
   **The residual is block by block (proposed)** — each block's
   worst over the largest datum of *that block* in the variable's group
   (`log α`, the shift, `γ`, `Π`), so a steep curvature cannot hide a poor
   value; `fit_residual(fit).overall` is the worst. BLAS's threads are not
   `julia -t`'s, so a fit is the same at every Julia thread count; nothing
   sets either.
5. **The validity sweep** (`fit_sweep`) evaluates the fit **raw** —
   reassembled, not projected — along every collocation ray at `s r_1(n̂)`,
   `s ∈ {1/8, …, 1}`, and at the center, `8N + 1` points (`1225` at
   `L = 8`): **fractions of each ray's own `r_1` (proposed)**, so
   `s = 1` is the collocation surface and the sweep covers exactly the
   region the fit is used in, oblate or not. A point is valid when finite,
   `γ` positive definite (`sym_eigen3`) and the signed lapse positive —
   **Lorentzian-ness and not a range**; the
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
   and the metric is valid. **The ranges it projects
   into are the target's own**: `FittedSpec`'s
   `target_bounds`, or — `nothing`, the default — `derive_target_bounds` of
   the analytic data on the seed's offset surface at `t = 0`: each upper
   range four times the largest value found there (`α`, `λ(γ)`, `|β|`,
   `|(α/√γ)Π|`), each lower one a quarter of the smallest, widened to keep
   flat space inside **(the factors proposed)** — because step
   8b's `default_bounds` are Kerr-Schild's, and harmonic Kerr at `a = 9/10`
   has `|(α/√γ)Π| = 366/M` on its offset surface against their `100/M`. On
   the fixture that is `α ∈ [0.154, 2.46]`, `λ ∈ [1/4, 10.6]`, `|β| ≤ 4.04`,
   `K ≤ 3.35`; on harmonic `a = 9/10`, `α ≥ 0.066`, `λ ≤ 774`, `K ≤ 1464`.
7. **What it measures** (on analytic data unless
   said).
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
     points, so `fit_state`'s projection would clamp the target's `Π` there.
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
8. **The target is cached on the grid**, not
   evaluated in the right-hand side: a field set of `Hsrc`'s shape (`G = 0`,
   read at the owned point, never differenced) holding **40 variables — the
   target `A` (the packed `(h, Π)`) at the fill time `t_f` and its slope in
   time `S`** — so that the kernel's target at `t` is `A + (t − t_f) S`, two
   loads a variable **(proposed**: one field set, and the slope
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
   unfilled layer point **(proposed)** — and zeros elsewhere; the
   cache is the host evaluator bit for bit. **What it costs** (one thread, on the fixture's 120 blocks): a fill is `43–44 ms`
   from one fit and `75 ms` from two, `0.4` and `0.7` of a right-hand side,
   once a chunk of about forty evaluations; a right-hand side with the
   fitted layer is within `±4 %` of the `:damped` layer's either way — `111`
   against `107 ms`, `106.7` against `107.6`, and in the suite's run `114`
   against `118` — the cache's two loads a variable where `:damped` takes the
   analytic solution's dual pass, which the prediction ("a few percent")
   had as a cost and is at worst a wash.
9. **The refill rule.** The cache is refilled
   from the current fits at every chunk's start — after every fit — and,
   for a geometry that moves, whenever the tracked center would move by
   more than `h/4` since the last fill: the chunk's steps are split into
   `⌈|v_est| · chunk / (h/4)⌉` solves with a refill between **(proposed**, over lowering the chunk, which would change the record's
   cadence**)**. On the static fixture `v_est` is the finder's noise and a
   chunk is one piece; on the boosted seed below (`|v| = 0.3`, `h = 5/128`,
   chunks of `M/20`) it is `1.54` → two pieces and one mid-chunk refill a
   chunk. **The moving seed** (`hole_runs.jl fitted=boosted`): `boost(Harmonic(1, 0), 0.3 x̂)` on 848
   blocks at `h = 5/128`, `m = 8`, to `0.1 M` — the finder succeeds on every
   row from the boosted hole's initial data, the track is at `x = −0.01492`
   and `−0.02985` at `0.05` and `0.1 M` against the analytic `−0.015` and
   `−0.03` (`v_est = −0.2983`, `−0.2986`), `track_offset` `2.1e−3` and
   `3.9e−3` cells, every fit valid; the masked error is `3.26e−2` against
   `:damped`'s `2.03e−2` on the same mesh and track (`1.6×`, the initial
   data's kink again), in 14 steps and `39 s` at four threads.
10. **The variant.** `INTERIOR_VARIANTS` has
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
   u_exact)` to `1.4e−14` of the right-hand side. The
   driver, per chunk: the geometry and the rate (`with_interior`) → the
   refill → the solve (in pieces, piece 9) → the record (constraints, error,
   monitor, find, indicator) → `update_track` and the next chunk's geometry
   → **`refit!`: the state sampler on freshly filled ghosts** (the find
   filled them, but so may every monitor since — a fill is a fifth of a
   right-hand side) → a `cont = 1` fit into the target's ranges → the regrid
   branch, whose new mesh gets a new cache filled at the next chunk's start
   from the same fits (a fit is a polynomial in `x` and does not know the
   mesh; **proposed**). **A fit whose sweep is not a metric does
   not become the target**: the previous fits are kept, the row says
   `fit_valid = false`, and the run goes on — the find's coasting applied to
   the fit **(proposed)**. The record's `residual` is, for
   `:fitted`, the layer's distance from its *target*, and the error kernel
   evaluates the analytic solution only at the points its error and drift
   rows read **(both proposed)**: the fitted interior has no
   analytic one, and harmonic Kerr's is singular inside the offset surface.
   `check_interior_radii` skips its singular-set check for `:fitted`. **A
   tracked run's first find starts from the seed's shape** on the finder's
   grid, not a sphere of the mean radius **(proposed)**: on
   harmonic `a = 9/10` that sphere, `0.72`, lies inside the offset surface's
   equator at `0.92`, and the footprint guard refuses its first iterates.
11. **The initial data**: the analytic solution
   outside the offset surface and **the `cont = 1` fit of the analytic
   solution inside it, for every chart**, from **`Float64` samples whatever
   `T` is** — the cache filled from the analytic fit first, then the state
   (`fitted_state_kernel!`); no analytic evaluation below the surface, so a
   chart whose interior is singular gets regular data, and `state_callback`
   refuses a `:fitted` geometry. The reason for `cont = 1` is harmonic `a =
   9/10`, whose curvature-matched fit is not a metric at any `L` to 16
   (piece 7). **Its price on the fixture**: the data
   are `C¹` at `r_1`, value and slope right and the curvature off by
   `O(1/M²)`, which the compact second difference turns into an `O(1)`
   right-hand-side error at the innermost evolved points — finding 1's kink,
   in the *data* where step 8c's E3 had it in the *target* only. The masked
   error at `0.15 M` is **`1.33e−2` against `:damped`'s `3.11e−3`** (`4.3×`;
   L∞ `0.177` against `0.035`) and decays to `1.24×` by `1 M`; at `Float32`
   the same run agrees with `Float64` to `7e−6`
   (`hole_runs.jl fitted=fixture` for the table):

   | initial data | masked L2 at `0.1 M` | `0.5 M` | `1 M` | shell `C_a` L2 at `1 M` |
   |---|---|---|---|---|
   | `:damped` (the analytic solution everywhere, the core rule) | `2.09e−3` | `9.81e−3` | `1.40e−2` | `1.24e−2` |
   | `:fitted`, the fit below `r_1` (the default) | `9.77e−3` | `1.90e−2` | `1.74e−2` | `4.02e−2` |
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
12. **Harmonic Kerr `a = 9/10`, `m = 4`**, the case's own `FittedSpec(margin = 4,
   lmax_shape = 12)` and not a change of the default `8`, at `h = 5/256` on
   2472 blocks (2240 of them at `5/256`) in a box of half-width `5/4 M`
   (`hole_runs.jl fitted=harmonic`, three minutes at four threads). **The
   first `:fitted` initial data of this chart exists**: `r_1` from `0.359` (axis) to `0.921` (equator), the core from
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
   at `L = 12`). **So the chart does not run on this target
   at `h = 5/256`**.
   **And the evolved region on the equator is itself under-resolved there**: the analytic solution's own length scale at the
   first evolved point on the equator, `√(|u|/|Δ²u|)`, is `0.008 M` at
   `h = 5/256` — under half a cell — `0.019 M` (two cells) at `5/512` and
   `0.024 M` (five cells) at `5/1024`, with `m = 4`: the ring sits `m h +
   0.02 M` inside it, and the solution varies on the scale of that distance.
   Whatever the interior does, the chart at `a = 9/10` wants `h ≲ 5/1024` on
   its equator, or a margin that depends on the direction.
13. **A moving target.** Two things change when the
   geometry moves through the grid, and there is one switch for each:
   - **The target's rate is fed forward** (`evolve!(…; target_rate = true)`,
     the default, **proposed**). A point relaxing toward a target
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
     measurable**: `4.56e−2` against `4.67e−2` masked
     at `0.5 M` on the boosted `a = 0` hole, `0.132` against `0.172` at
     `M/4` on G5's chart with `w_ramp = 1` and `0.492` against `0.494` on the
     default ramp —
     the lag is not what the moving layer suffers from (next). It stays on,
     since it is the consistent form and costs two evaluations of the fit
     per filled point per refill.
   - **The initial data are blended into the fit across the layer**
     (`evolve!(…; fit_initial_blend = true)`, G5's rows, **proposed**). With `fit_initial_depth = n_L h` the data step from the analytic
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

The fit holds **`Π̃ = (α/√γ)Π`** by default
(`fit_variables(…; tilde)`, `state_from_fit(v, tilde)`, `FitParams.tilde`,
`FittedSpec`'s `fit_tilde = true`); `interior.jl` has `with_variant` for a
spec and a geometry; `evolve!` has `handover` (a `:fitted` case on the
analytic `:damped` layer of the same geometry until then) and `target_source
= :snapshot` (the cache filled with the state, `fill_snapshot!`), and
`adapt = true` works for a `:fitted` case — the mesh chosen on the analytic
data of the same geometry, refused by name where the analytic core meets the
chart's singular set; the record has a `variant` row; the drift reads the
evolved points of its band only.

`evolve!` has `target_rate` (default on: the cache's slope carries the fit's
translation with the track and the kernel adds `(1 − w) ∂_t u_fit` in the
layer and the core; `GHProblem.target_rate`, `fill_target!(…; rate)`).

**`FittedSpec(; fit_cont = 2)`** fits curvatures too (the state sampler's
`order = 2` differences the interpolated gradient).

**Inconsistent:** item 3 above says the state sampler "gives no second
derivative, so it serves `cont = 1`, and `cont = 2` with it is refused";
the paragraph above says
`FittedSpec(; fit_cont = 2)` fits curvatures through the state sampler's
`order = 2`, and the table under "Single black holes: recommended
settings" uses `fit_cont = 2` for `:fitted` holes.

**Behind a moving layer, narrow the ramp**: `trail_ramp = 9/10` makes `ρ` rise where points leave the layer and is
  what removed the trailing side's excess; refilling the target more often
  and evaluating the fit exactly in the kernel changed the fourth digit. It is off by default (a static run is unchanged bit for bit); [The trailing side (step 8′)](SINGULARITY_HANDLING.md#the-trailing-side-step-8) has the numbers.

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

## Initial data and backgrounds

**Analytic, from `SpacetimeMetrics`**: `Minkowski`,
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
rather than in `driver.jl`. Its constructor is where a moving non-harmonic
background with a sampled source is refused; with the algebraic Kerr-Schild
source the boosted Kerr-Schild hole is a case.

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
error of a hole visibly; the prediction was "not beyond the first chunk".

The post-pass is `discrete_gradient_momentum!`, and the
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

**The hole cases** are `kerr_schild_case` and
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

**The background is evaluated inside kernels** (a
dependency risk). The interior's `u_exact` is needed at every RHS
evaluation and the Dirichlet hook at every ghost fill; both must be
kernel arguments, so `SpacetimeMetrics` — StaticArrays plus ForwardDiff
duals, both `isbits` — has to compile as one. On the CPU backend that is
certain; on CUDA it is expected and untested, and G0 tests it on the CPU
while G6 tests it on the H200. Measured on the CPU
backend: `KerrSchild(1, 0)` and `boost(Harmonic(1, 9/10), 0.3 x̂)` both
compile into `fill_by_coordinates!`'s `AllVariables` kernel and fill a
20-variable, `G = 3`, vertex-centered field set over a two-level forest
with **bit-for-bit** the values of a host loop through `coordinates`, at
`Float64` and at `Float32`; both backgrounds, and `boost`'s wrapper
around one, are `isbits`. If the device refuses, the fix is a
kernel-safe evaluation path *in `SpacetimeMetrics`*, not a redesign
here: nothing in this package's structure depends on where the metric
is evaluated, only its cost does.

**Inconsistent:** this paragraph says that on CUDA a background in a kernel is "expected and untested" and that G6 tests it on the H200; "Time integration" ("On a device") records `evolve!` on a hole running end to end on one H200 at `Float64` and `Float32`, the record equal to the CPU run's.

**Run tools**: `add_noise!` (with `exclude` for a frozen core) and
`evolve!(; perturb)`.

## Refinement and regridding

**An error indicator drives the refinement**. It is TreeWave's machinery ported: a
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
10 components of `h` (`Π` is a switch), which fall off as
`M/r` around a hole, and `U_ref` is the largest `|h|` in the *evolved*
region at `t = 0`.

**`U_ref` is one amplitude for all ten components, not one per component**, and `Π` is left off. Per-component is the trap in
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
   indicator's mood. The indicator asks for the floor
   level or more everywhere the floor applies, so the floor never binds
   on a calibrated run; the test that it *can* bind uses a deliberately
   loose `refine_tol`. On the reference
   configuration below, the marks with the floor and the marks without it
   are identical at the calibrated thresholds, and with `refine_tol` raised
   to `99/100` — above anything the data reaches — the floor is the only
   thing that refines, on exactly the blocks whose extent meets the shell.
   **`L_h` is derived rather than stated**: it is the
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

   **Two things the ceiling turned out to need.** First, the
   **travelling margin has to be dilated before the bounds are applied, not
   after**: TreeAMR's `buffered_flags` promotes every leaf a dilated box
   reaches, whatever this package said about it, so on any root brick small
   enough that the boundary blocks are the refined region's neighbours the
   margin refines them anyway — measured, 12 of 64 boundary root blocks.
   So `refine_flags` calls `buffered_flags` itself, applies the floor and
   the ceiling to its result, and the driver passes `buffer = 0` to
   `regrid!`. Second, **the ceiling and the floor
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

`hole_forest` in `initialdata.jl` builds the frozen hierarchy by hand: one shell radius per refinement
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
the refinement mechanism, which is the indicator's.

**What follows for the hole.** The indicator follows the moving hole on
its own, and its center of refinement is a measurement to be compared
with the analytic center — a disagreement of more than a few finest
spacings is a bug in the indicator or the thresholds. The interior layer
still uses the analytic center, because it must. The initial-data cycle
converges to a fixed hierarchy on a static hole (the indicator is
time-independent up to truncation noise), so regrids during a static run
should change nothing **(predicted)**; on the moving hole they move the
shells with it, one block ring per chunk at the finest level.

**Inconsistent:** "The interior layer still uses the analytic center, because it must" contradicts this section's "The floor reads the horizon the layer actually follows" and "The mesh follows the hole" below, and "The interior" → "The tracked geometry", where a case whose interior is a `FittedSpec` has its layer follow the found horizon (the horizon-finder paragraph under "Analysis quantities" says so too).

**The driver** is TreeHydro's one loop with cases as data:

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

The loop is `src/driver.jl`. `evolve!` takes the **forest** as a keyword
rather than building one: a frozen hierarchy is `hole_forest`'s, and an
adaptive mesh is what `adapt_to_initial_data!` produces
from the indicator. Three things the writing settled:

- **`ρ_max` is what makes a chunk a restart even without a regrid.** Under the grid-rate option
  it is `1/dt` and `dt` is measured per chunk, so the interior the kernel
  closes over is rebuilt at the top of every chunk (`with_interior`),
  which shares the field sets, the schedule and the **sampled gauge
  source** rather than rebuilding the problem — re-sampling `H_a` is the
  most expensive setup phase there is and nothing about a new `ρ_max`
  invalidates it. The default is `4/M`, the
  same in every chunk, and the rebuild stays — it is what the grid-rate
  option `ρ_max_factor` needs, it is where the default is checked against
  that chunk's `1/dt`, and it costs nothing.
- **The record's first row is `t = 0`**, before anything has been
  integrated, so that "the error grew from zero" is a statement a test
  can check rather than assume; `nchunks` is the number of rows after it.
- **The CFL recheck is `check_cfl`, and it throws** (`CODE.md`, "The time
  step": *throw, do not warn*). It is a detector and not a guard — the
  chunk has already been integrated — and the remedy is a shorter chunk
  or a smaller `cfl`. On the static hole `λ_max` is `1.671` for
  Kerr-Schild `a = 0` and does not move between chunks, so the recheck
  has never fired; it exists for G5, where the hole crosses the mesh.

`src/refinement.jl` and the loop's two
branches implement this. `evolve!` takes `adapt` and `regrid`, both of which
refuse a case with no `Refinement` by name; the parameters are one
`isbits` struct on the case (`refine_tol`, `coarsen_tol`, `maxlevel_cap`,
the floor's margin beyond `r_h,max`, the ceiling's margin in coarse cells
and its level, and `ε`) rather than five fields, so that a case that is run
on somebody else's mesh carries one `nothing`. Five things the writing
settled, each stated where it happens:

- **`τ` is materialised into `diag`, and the marks are taken from that
  field.** TreeWave and TreeHydro evaluate their
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
- **`regrid!` is handed the state field set alone.**
  The loop above lists `Hsrc => nothing, diag => nothing`; both are rebuilt
  by the fresh `GHProblem` — which is also where the gauge source is
  re-sampled and where `check_interior_radii` re-asserts the interior's two
  radius requirements on the new mesh — so resizing them through the
  transfer would be work thrown away.
- **The regrid is skipped after the last chunk**, so
  that the forest that comes back and the state that comes back describe
  the same mesh. TreeHydro's rule, for TreeHydro's reason.
- **The record gains `τ_max`, the refinement centroid and its distance from
  the analytic center**, which is the mesh-statistics row of the analysis
  table, and the run's `passes`, `converged`, `nregrids` and the buffer
  width it used.

**The floor reads the horizon the layer actually
follows.** `horizon_floor_level` and `level_bounds` take the layer's
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
indicator chose. The indicator takes the
geometry the run holds (`indicator_flags(…; interior)`), and the
initial-data cycle rebuilds a tracked geometry on every pass's mesh and
re-evaluates the data with the final one's core rule. The suite exercises
none of this on a mesh that moves — its tracked runs are on the step-5
fixture's frozen hierarchy — and step 8f's matrix is where a tracked
geometry is first regridded.

**The mesh follows the hole.** Four pieces:

- **A `:fitted` case's initial-data cycle flags on its own data**
  (`adapt_fitted_initial_data!`): on every pass the geometry is rebuilt on
  the current mesh, the case's initial data are filled on it
  (`fill_fitted_initial!` — the analytic solution outside the offset
  surface, the fit of it inside), the ghosts filled with the `t = 0` hook,
  the masked indicator flags with the geometry's mask and floor, and
  `regrid!` rebuilds without transferring — TreeAMR's
  `adapt_to_initial_data!` written out, because a `:fitted` case's data live
  partly in a cache a coordinate callback cannot read. This cycle converges on G5's chart, where the
  analytic core cuts the disk (2 passes to 456
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
  margin `buffer` is `|v| · chunk` in cells: 5 cells at
  `5/256` and 3 at `5/128` for `v = 0.3`, chunk `M/4`. Both are zero-width
  for a static hole, whose bounds are unchanged.
- **The regrid flags the evolved state** at every chunk boundary with the
  geometry the next chunk runs on — nothing new: `gh_indicator!` already
  read the state and the layer's mask.

**What it measures on the boosted `a = 0` hole** (`hole_runs.jl moving=ctl`, harmonic `a = 0` boosted at `0.3` from `x = 2`
to `x = −1.9` in `13 M`, box `5 M` on a `4³` root brick, finest `5/128`,
`chunk = M/4`): the mesh regrids at 40 of the 52 chunk boundaries,
between 1128 and 1912 blocks, every radius assertion holding at every one;
the refinement centroid stays within **`1.4` finest spacings** of the
analytic center over the crossing on the analytic `:damped` layer (`20/M`)
and within **`7.3`** (typically 4–6) on the `:fitted` one, whose layer
edge leaves more grid-scale content on the trailing side for the indicator
to score; the static configuration's centroid is `0.29` spacings off at
`t = 0`. The travelling hole is tracked within `0.028` cells.

The level floor of a tracked
geometry starts at its **core** surface and is widened by the travel
`|v| · chunk` (`level_bounds`, `indicator_flags`, `gh_indicator!` take
`travel`).

## Time integration

**IMEXRungeKutta's classical RK4, its stage arithmetic by block owner**,
as TreeAMR's own examples use it (OrdinaryDiffEq before it: see
[history](HISTORY.md#from-ordinarydiffeq-to-imexrungekutta-2026-09-26)). `src/stepping.jl` is the whole coupling: `state_partition`
turns TreeAMR's `threadchunks(nblocks)` into IMEXRungeKutta's `partition`,
so that each block's entries are combined on the thread `map_blocks!` runs
the block on (`nothing` on a device: the fused broadcast); `gh_integrator`
builds `init(IMEXProblem(gh_rhs!, nothing, u, tspan, p), RK4(); dt,
stage_limiter = gh_limiter!, step_limiter = gh_limiter!, partition)`, and `gh_solve` runs one to its end. The same pieces are public for a script that owns the loop: `prepare!`, `rhs!`, `stage_limit!`, `step_limit!` and `max_speed(p, u, t)` ([The verbs (GH-4)](#the-verbs-gh-4)). The driver builds **one
integrator per chunk**, stepping the run's own state vector in place
(`alias_u0 = true`); a moving hole's refilled target is swapped into it
between pieces through a `ProblemRef` (`swappable = true`) instead of a new
`init`, and each chunk's integrator takes over the previous chunk's scratch
while the mesh is unchanged (`reuse`, IMEXRungeKutta 1.2),
so the scratch is allocated once per mesh and not once per chunk. A regrid
that moves the mesh drops the scratch *before* the new problem is
allocated: the four state-sized vectors are what a run that fills a device
cannot hold beside the new problem's arrays. The following mesh at `1/48`
on an H200 (`115` of `143` GB) ran out of memory in the new problem's
constructor at its first regrid that moved the mesh; with the scratch
dropped it built the new problem, and then ran out allocating the next
chunk's scratch on a mesh that had grown. That row ran on two H200s over
MPI, at `160 s/M` ([step H2](SINGULARITY_HANDLING.md#a-moving-harmonic-hole-step-h2-2026-11-12)). What follows from it:

- **The limiters: one, on every state vector.** IMEXRungeKutta calls its stage limiter only on the three stage
  values of a step that `f_exp!` reads and its step limiter once on the
  result, so `gh_limiter!` — the projection, then the paste — is passed as
  both, IMEXRungeKutta's own rule for a correction that must reach every
  right-hand-side input. On a static hole
  pasting a stage value changes no bit (inside `r_1` the `:pasted`
  right-hand side is zero and the analytic solution does not depend on the
  stage's time); on a moving one the ball is the analytic solution at every
  stage's time.
- **By owner the result is bitwise its own broadcast and bitwise across
  thread counts**, which `test/thread_workload.jl` digests.
- **One right-hand side fewer per `solve`** (no FSAL start) and no copy of
  the state: IMEXRungeKutta allocates four scratch vectors per
  `init`, first-touched through the partition — still 0.13–0.36 s at 64
  threads on 320 MB (first touch dearer than interleaved pages), which is
  why a chunk has one integrator and not one per piece. With IMEXRungeKutta 1.2's `reuse` the next chunk's `init`
  takes those four arrays over and allocates nothing: 0.05 ms against 13–35
  ms at four threads on the development machine (40–75 MB states), so the
  cost is paid once after the initial data and once after every regrid that
  moves the mesh.

**Measured on Symmetry** with `bench/stepping.jl`, one exclusive
64-core EPYC 7543 node, cn096, jobs 563975 and 563982; 512 blocks of `16³` at `q = 4`, a 320 MB
state; the gauge wave and the suite's Kerr-Schild fixture at `halfwidth =
5` with `hole_forest(…; roots = 2, radii = (6, 3, 3/2))`, whose finest
spacing is the fixture's; minimum ms per call, each configuration twice, in
agreement to a few percent:

| per call, pinned, first touch | gauge wave | hole |
|---|---|---|
| RHS | 120–136 | 252–266 |
| IMEXRungeKutta RK4 step, broadcast | 770–793 | 1304–1330 |
| IMEXRungeKutta RK4 step, by owner | 553–603 | 1077–1106 |
| `gh_solve` per 4 steps, per step (`init` included) | 643–706 | 1157–1200 |

A step by owner is `4.2–4.9` right-hand sides.
**Placement** (the four configurations on cn096): pinned (`JULIA_EXCLUSIVE =
1`, `srun --cpu-bind = none`) with first touch is the fastest for the
right-hand side (133 ms against 151 unpinned on the wave, 257 against 318 on
the hole) and for the step; interleaving (`numactl --interleave = all`)
helps only `init`'s allocation. **The driver end to
end** (`evolve!`, three chunks, the record included; jobs 563976 on cn112 —
an 8×8-core AVX-512 node with one NUMA domain, pinned — and 563977 on
cn079, an EPYC 7543, unpinned): per step, **544 ms** and **1140 ms** pinned, **843** and **1516**
unpinned.

**On a device.** The right-hand-side kernel compiles for a
GPU: on the development machine's Metal at `Float32`, and on one H200 (job
563978, cn113) at `Float64` and `Float32`, with `evolve!` end to end — the
record's monitors on the device, the broadcast stage arithmetic, `hostcopy`
only for a horizon find (none here). The `Float64` hole's record equals the
CPU run's to every printed digit (`err_l2 = 1.690514e−6`).

The first device timings, before the kernel was tuned, are in
[history](HISTORY.md#the-first-device-runs-2026-09-26); the tuned kernel is under [The right-hand side on an
H200](#the-right-hand-side-on-an-h200).

**Block size and block count** (`bench/stepping.jl`'s
`scan` mode, `q = 4` throughout — fourth-order centred differences, sixth-order
prolongation, `G = 3`; H200 jobs 564111 and 564114, the CPU on cn079, an EPYC
7543, pinned at 64 threads, job 564112; the hole's rows need two roots or more,
since one root at `N ≤ 12` fails the interior's radius checks). The
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

**Inconsistent:** the H200 rows above put the right-hand side at 9–10 ns a point from `N = 16` up, while [The right-hand side on an
H200](#the-right-hand-side-on-an-h200) measures the tuned kernel at 1.1 ns a point; the block-size table was taken before that tuning.

At `cfl = 1/4`
the time error does not contaminate the spatial one even at `q = 6`,
because `dt ∝ h` makes it `O(h⁴)` with a factor `cfl⁴ ≈ 0.004` in front
and the runs are short: the measured rate is 5.92, not 4. That headroom is
a property of the *time* the studies run for — an eighth of a crossing —
and a longer run at `q = 6` would need a smaller `cfl` or a higher-order
tableau to keep it, which is worth knowing before G3 lengthens anything.

The relaxation rate of the interior layer is bounded by RK4's stability
on the negative real axis, and both rates the driver runs keep it well
inside — the default `4/M` is `0.04/dt` to `0.09/dt` on the suite's holes,
and the grid-rate option `ρ_max · dt = 1` is a factor `2.8` inside; the
limiters are the one `gh_limiter!` above. Adaptive
stepping is not used: the step is set by the CFL bound and
`volume_weighted_norm` is not wired in as an `internalnorm` (TreeAMR's
open question, not this package's).

**One integrator per chunk.** The
  integrator's `p` and `dt` are constants and each `init` allocates and
  first-touches four state-sized scratch vectors — 0.13–0.36 s at 64
  threads on 320 MB — so `evolve!` builds **one per
  chunk** and swaps a moving hole's refilled problem in between pieces
  (`swappable = true`: the integrator's `p` is a `ProblemRef`, `integ.p.p =
  p′`), rather than one per piece — and **each chunk's integrator takes
  over the previous chunk's scratch** (`reuse = integ_prev`, IMEXRungeKutta
  1.2), so the allocation is once per mesh.

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
| interior residual | `|u − u_exact|` inside the layer, L∞, the layer's own health — for `:fitted`, `|u − u_fit|`, the distance from its target | every chunk |
| mesh statistics | leaf count per level, finest spacing, the indicator's `τ_max`, the refinement centroid against the analytic center | every chunk |
| range projection | `bounds_hits` (points moved, summed over every stage-limiter call of the chunk), `bounds_nonfinite` (of those, points with a non-finite component), `bounds_r_max` (the outermost radius it fired at, `−1` where it did not); `nothing` for a case without bounds | every chunk |
| validity monitor | over the layer `r_0 ≤ r < r_1` and over the `G` points outside it: `min_detγ`, `min_α` (the *signed* lapse, negative where `g^{tt} > 0`), `max_h`, `max_Π` (the largest component magnitudes) — `min_detγ_layer` … `max_Π_shell`; and over the whole evolved region, `min_detγ_evolved`, `min_α_evolved` — the lapse-collapse trigger's input. On the tracked geometry the two bands are its own (`layer_mask`, `shell_mask`) | every chunk |
| horizon track | for a case whose interior is a `FittedSpec` (`nothing` otherwise): `track_source` (`:found`, `:coasting`), `track_center` and `track_velocity` (the tracked trajectory after this row's find), `track_r_min`, `track_r_max` (the found surface's radii about its own origin), **`track_offset`** (the tracked center's distance from the analytic one, in cells of the finest spacing — the number "good to about a cell" is read from), `track_misses`, `track_prediction` (the found origin's distance from where the track predicted it, in cells; `nothing` without a find), `track_trigger` (this row's find was forced by the lapse trigger) | every chunk, the find every `k`-th or when triggered |
| tracked layer | the geometry the next chunk runs on: `layer_h` (the spacing its offset and ramp are stated in), `layer_offset = m h`, `layer_thickness = n_L h`, `layer_r_in`, `layer_r_out` (the shape's bounding radii), and `margin_efolds` — step 8a's leakage e-folds across the margin, the least of the six grid axes | every chunk |
| fitted target | for a `:fitted` case (`nothing` otherwise), of the fit built from this row's state: **`fit_valid`** (every point of the fit's validity sweep a Lorentzian metric — `fit_valid(fit)`; a fit that is not does not become the target), **`fit_residual`** (`fit_residual(fit).overall`, the fit's worst relative residual block by block against the data it was fitted to), and beside them the sweep's worst **`fit_min_detγ`**, `fit_min_α`, `fit_min_λ`, `fit_hits` (swept points the target's ranges would move) and `fit_refills` (the cache's mid-chunk refills in the chunk this row ends); for `:fitted` the row `residual` is the layer's distance from its target (the cache), not from the analytic solution | every chunk |

**Constraints.** Both kernels mask the interior `r < r_1` and write zero
inside it; the modified region is not a numerical solution. The norms are TreeAMR 0.2.6's **folds over
points**, `mesh_fold` (M15, which this package asked for): the callback sees
a whole owned point — several `diag` slots at once, its position bit for bit
`point_position`'s, its spacing, level and a per-block tag — and folds it
into an `isbits` accumulator of `Sum`, `Max` and `Min` fields, so one sweep
and one collective give every norm of every region. `diag_norms` is the one
fold behind `masked_norms`, `evolved_volume`, `constraint_norms`,
`error_norms` and `record_norms` (the last two together, which is
what the driver's row reads after its three monitors: they write disjoint
slots); `level_constraint_norms` bins every level in one fold; the validity
monitor and the evolved region's finiteness count are one fold over the
state vector, with the regions' masks asked at the point (`diag` has 17 slots); the projection's three counts are one fold, and
the indicator's ten reference amplitudes another, without the scratch field
set. A record row makes 2 reductions (4 with the indicator);
`test/octant_runs.jl`'s row runs each monitor once and folds three times. The weight is per point, `p.h³`, and each block is one sequential fold on
its owner's thread, so the norms are bit-identical across thread counts.
The speed and `τ_max`, one number each, are `mesh_mapreduce`s.

`src/constraints.jl` implements both monitors. Five
things the writing settled, each stated where it is made in that file:

- **Which `∂_t g`.** The gauge constraint takes it from the first
  evolution equation, `β^i ∂_i g + (α/√γ)Π`; the ADM monitor takes
  `∂_i∂_t g` by differentiating that along `x^i` and `∂_t∂_t g` by
  differentiating it along `t`, with `∂_tΠ` from
  `gh_node_rhs_expanded` — the reduced equation with its source and its
  constraint damping, which is what "consistent with the discrete
  dynamics" means. **The Kreiss–Oliger term is left out of both**: it is `O(h^{q+1})`, one order below what these
  monitors converge at, and carrying it would mean differencing the
  dissipation operator as well. This is the one place this package's
  `∂_t g` and the right-hand side's differ, and the difference is below
  the truncation error either of them measures.
- **The Ricci tensor is assembled, not reduced.**
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
  direction. The three-direction function is
  untouched, and `test/constraints_tests.jl` asserts the two agree on
  every spatial direction — to roundoff, for the reason under
  [Measured results](#measured-results).
- **The masked norm divides by the evolved volume.** Each kernel writes a
  `1`/`0` indicator into a `diag` slot beside its values, and the L2 norm
  is `√(Σ_b h_b³ Σ|c|² / Σ_b h_b³ Σ 1)` — so masking a region out does not
  make the number smaller merely by diluting it with zeros. Where nothing is masked that is TreeAMR's
  `volume_weighted_norm` exactly, which the tests assert rather than
  assume.
- **`diag` starts with ten slots**: the speed, `C_a` (four), `ℋ`, `ℳ_i`
  (three) and the mask indicator, with `C_a` and `ℳ_i` contiguous because
  `block_mapreduce` reduces a contiguous range of variables and nothing
  else. The masked error, the interior
  residual and the gauge drift follow, appended rather than inserted, because
  the two contiguous runs above must not move, and then
  the refinement indicator `τ`, appended for the same reason.

The error rows are one more
kernel — the state, the analytic solution, no stencil and no ghosts, so
it costs what the speed kernel costs plus one forward-mode dual pass per
point — and three decisions:

- **The error slots hold a magnitude, not twenty components.** The table above says "`|u − u_exact|` per
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
- **The gauge drift's shell is `[r_h,min, r_h,max + (r_1 − r_0)]`.** `CODE.md` asked for "the drift of `h_tt` at the
  horizon" and did not say over what set. The outer margin is the layer's
  own width rather than a number chosen for the occasion: it is the only
  length in the case that is set by the hole and known to be resolved.
  A general `ShellMask` does the same job for any norm, which is how
  the three interior variants are compared over "the `G` points outside
  `r_1`".

In `src/bounds.jl`, the range projection's
rows and the validity monitor's are written at every chunk, and three
things the writing settled:

- **The hit count is per call, summed over the chunk.** A stage's flags are overwritten by the next stage's, so a count
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
- **`finite` is the evolved region's.** It was
  `all(isfinite, u)` over the whole state, which with `max_speed(p, u, t)`'s
  identical check ended a run at the first `NaN` in the frozen core; both
  now count non-finite values at `r ≥ r_1` (`evolved_nonfinite`). A `NaN`
  in the core is the projection's business, not the end of the run. The
  *initial data* is still checked everywhere, before anything else: a
  non-finite value there is a case whose analytic solution is singular
  somewhere the mesh reaches, which is the configuration error step 5's
  check was written for, and not something to repair.
  `diag` has **17** slots, the projection's three appended; the monitor and
  the count are folds over the state
  vector, see [Analysis quantities](#analysis-quantities)'s constraints
  paragraph.

The mesh-statistics row holds
`τ_max` and the refinement centroid, and both come out of the flagging pass
the regrid needs anyway:

- **The centroid is the volume-weighted centroid of the cells above
  `coarsen_tol`**, taken over the bounding boxes the
  `coarsen_tol` sweep already reports — each block's box center weighted by
  its firing count and its cell volume. A centroid over cells rather than
  boxes would be a third sweep of the mesh for a diagnostic.
- **It has a bias of order the coarsest *firing* spacing, and the reason is
  the mesh's indexing.** On the static hole, where the
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
provider interpolates with TreeAMR's `interpolate` (M11), the basis
`Lagrange(q + 2)`, located by one binary search rather
than an ancestor walk, one launch on the field set's backend rather than a
host loop, so a device-resident state is no longer copied to the host per
find (`hostcopy`); `gh_interpolate`/`gh_interpolate_grad` are the thin
wrapper, and the provider and the fit's `state_sampler` call it. The
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
few finest spacings has found a bug, not a feature to track. A case whose interior is a `FittedSpec` asks for its layer to
follow the found horizon, and then the finder's answer is an input —
through `src/tracking.jl`'s track, from the row the driver records, under
[The interior](#the-interior-a-pointwise-damping-layer), "The tracked
geometry". Every other case is as this paragraph says, and `track_offset`
is the row that says whether a tracked one has found a bug.

`src/horizon.jl` implements the finder's side. Five
things the writing settled, each stated where it is made in that file:

- **The guard is on the footprint and it is exact.**
  `CODE.md` asks the provider to throw "if a query point's interpolation
  footprint reaches `r_1`", which is a statement about `(q+2)³` points and
  not about the query. The footprint is a tensor-product lattice, so the
  nearest of its points to the hole's center is the per-axis nearest in
  each direction, and the test is three clamped roundings and a sum of
  squares — exact, rather than the bounding box's conservative
  `√3·(q+2)h/2`. It fires on a query that is *outside* `r_1` and reads
  inside it, which is the whole point of putting it on the footprint.
  TreeAMR's
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
- **The radii are measured from the analytic center.** `CODE.md` says "the coordinate radii of the surface points" and
  not from where. The two claims made on them — that the horizon encloses
  the layer by the margin `m`, and that `r_min` and `r_max` are
  [`horizon_min_radius`](#the-interior-a-pointwise-damping-layer) and
  `horizon_max_radius` — are both statements about the center the layer is
  built around, so that is the center. The distance between the two
  centers is a row of its own, `center_offset`, and it is the number the
  paragraph above turns into a bug report. `r_mean` is the `sin θ`-weighted
  mean over the collocation points, which is `∮ r dΩ/4π` to the accuracy
  of the grid's own quadrature. `find_gh_horizon`
  takes `center =`, the point the radii and `center_offset` are measured
  from, defaulting to the analytic center, and also returns the radii about
  the surface's own recentred origin, `origin_r_min`, `origin_r_mean`,
  `origin_r_max` — what the shape `hlm` describes, and what a tracked
  geometry is built from. A tracked run passes its predicted center, so its
  rows' `r_min` … `r_max` are about the track and its `center_offset` is the
  track's prediction error.
- **A failed find is recorded, not thrown.** The
  horizon is a diagnostic and nothing in the evolution reads it, so a
  find that throws — the guard refusing a flow that wandered inward, a
  seed sphere outside the box — leaves `horizon_success = false` and the
  message in `horizon_note` beside the chunk it happened at, and the run
  goes on. A diagnostic that ends a run loses the record that would have
  said why.
- **The spin's uniformization tolerance is `1e-8` and not the library's
  `1e-13`.** `1e-13` is the round-off floor of
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
The table's "about one RHS" for the ADM monitor is
not what an H200 measures. There a pair of the ADM monitor and the error
kernel costs about `30` right-hand sides, inferred from whole runs, and a
find under a second: [The analysis cadence on an
H200](#the-analysis-cadence-on-an-h200). A production run
takes its analysis row every `M`.

`initialdata.jl`'s `GHCase` has a sixth type parameter for the `Horizon`, and
`with_horizon`; `driver.jl`'s record has the horizon rows, seeded from
the previous find.

Point-weighted and per-level norms: `weighting = :points`, `at_level`,
`level_constraint_norms`. Beside `diag_norms` and `record_norms`,
`constraints.jl` has `slot_norms`; `bounds.jl`'s `validity_and_nonfinite`
is the validity monitor and `evolved_nonfinite` at once (`NDIAG = 17`).

**Reductions across blocks are TreeAMR's.** TreeAMR combines the per-block partials on the host,
  and over MPI (its M7) gathers one partial per rank; a `sum`, `maximum` or
  weighted loop over `block_mapreduce`'s vector here would be a per-rank
  answer under MPI. Several numbers are one fold because a point's
  values do not depend on the mask, and each reduction is a sweep and a
  collective. The fold's `rf` must return `init`'s exact type (a `Sum(1)`
  beside a `Sum(0.0)` is refused), everything it closes over must be
  `isbits`, and a region is asked of the point as `is_evolved(mask, p.x)`
  with the masks the kernels take. The blocks combine pairwise, so an L2
  compared with a stored number is compared to roundoff;
  `threading_tests.jl`'s exact claim names the combination.

**The ADM monitor's shape.** It is a
  four-dimensional Ricci tensor out of a hundred second derivatives, which
  is the opposite of the right-hand side's streaming order and
  deliberately so: it runs once per chunk, not once per stage.

**Ghosts must be filled before anything is interpolated**, with that
call's hook. `find_gh_horizon` scatters and fills; `gh_adm_provider` and
`gh_interpolate` do not, and an unfilled halo makes the metric garbage
  exactly at the block faces — which the fast flow then walks into the
  layer to escape.

## The coupling contract with TreeGRRMHD

TreeGRRMHD evolves general-relativistic MHD on this package's spacetime.
Neither package depends on the other. A short script owns the loop and
calls each package's verbs (`prepare!`, `rhs!`, the limiters) in the order
the data flows (TreeGRRMHD's `CODE.md`, "The coupling contract"). What crosses
between the two packages, and where ([history](HISTORY.md#the-coupling-contract-with-treegrrmhd)):

- **Both directions cross at this package's vertices, in 3+1 variables.**
  - The metric crosses as `(α, β^i, γ_ij, K_ij)`, 16 values a vertex.
  - The matter crosses as `(ρ, S_i, S_ij)`, 10 values a vertex:
    `ρ = n^a n^b T_ab`, `S_i = −γ_i^a n^b T_ab` and
    `S_ij = γ_i^a γ_j^b T_ab`. They are covariant, not densitized, in
    `G = c = 1` and the metric's length unit.
- **Every conversion between centerings is TreeGRRMHD's.** TreeGRRMHD is the only package
  with mixed centerings, so it owns these conversions:
  - it interpolates the vertex metric to its cells, faces and edges;
  - it differentiates its own interpolant for `∂α`, `∂β` and `∂γ`;
  - it averages `(ρ, S_i, S_ij)` from its cells to these vertices.

  This package interpolates nothing and fills none of TreeGRRMHD's sets. It
  knows neither TreeGRRMHD's centerings nor its ghost widths.

GH-5 stays withdrawn: nothing is asked of this package's ghost width.

### What this package provides (GH-1)

`adm_vars(work, I, b, inv_h, Val(q))` already returns those 16 values at a
stored vertex ([below](#public-access-to-the-vertex-metric)),
so GH-1 needed no new algebra, only its form in the contract. It is in
`src/admvars.jl`:

```julia
adm = ADMMetric(p, u, t)        # scatter and ghost fill, as gh_rhs! does
adm = ADMMetric(p)              # or the field set as the caller filled it
at  = adm(M)                    # bound to the consumer's vertex set M
(; α, β, γ, K) = at(n, b)       # at an owned vertex
```

- **Two objects.** `ADMMetric` is the host side: it holds the forest, this
  package's working array, the block spacings and `Val(q)`, and is never
  passed to a kernel. Binding returns the kernel side, which holds the
  arrays, `Val(G)` and `Val(q)`, and is `isbits` once adapted to the backend
  (`Adapt.@adapt_structure`, which a KernelAbstractions launch applies; Adapt
  became a direct dependency for it, at the version KernelAbstractions
  already loads). The consumer needs none of the arrays, `G` or `q`.
- **Binding.** `M` must be vertex-centered and on the same forest (`===`),
  which fixes `N` and the blocks. Otherwise binding throws an
  `ArgumentError`.
- **The index is the owned index** `n`, with `1 ≤ n[d] ≤ N`, as
  `map_blocks!` hands it to a kernel over the default range; `b` is the
  local block. The adapter adds this package's `G` and reads the block's
  own `1/h`. An owned index is the same in every vertex-centered set on the
  forest, so the consumer's `G` never enters. A `@boundscheck` refuses any
  other `n`, or a block out of range, with a `BoundsError`.
- **Accuracy.** `α`, `β` and `γ` are the decode of the stored state, exact
  to roundoff and the same bits as the right-hand side's
  `metric_quantities`. `K` comes from the right-hand side's own
  first-derivative stencil and converges at order `q` (measured: 2.00, 4.00,
  5.99). The contract's 4th order therefore needs `q ≥ 4`.
- **Valid at the owned vertices.** `K`'s stencil reaches `q/2` points.
  With `G = q/2 + 1` it is defined one ghost layer out and no further. The
  contract promises the owned vertices only; the consumer's ghosts are its
  own. The shared upper plane `n = N + 1` is a neighbour's owned plane, or
  at the outer boundary the consumer's hook's.
- **A snapshot.** The adapter reads this package's working array in place
  and is valid until this package's state changes. A regrid replaces the
  field set: rebuild and rebind after one.
- **Extra fields.** It returns `sqrtγ` and `γu` too, as `adm_vars` does.
  Destructure by name.
- **The consumer's set needs `N ≥ 10` at `G = 4`.** TreeAMR's vertex
  invariant is `N ≥ 2G + 2`, so a 16-variable set wide enough for a
  4th-order interpolation to the outermost ghost cell of a `G = 3` cell set
  refuses `N = 8`. TreeGRRMHD's TOV
  runs use `N = 16`.

**Measured.**

- **On the CPU** (`test/admvars_tests.jl`, two testsets, 17 claims, about
  10 s inside the file). On a Kerr-Schild hole at `a = 1/2` on a two-level
  mesh at `N = 10`, the adapter is `adm_vars` bit for bit at every owned
  vertex of both levels, after its own scatter and fill of a working array
  set to `NaN`. In a consumer's `map_blocks!` over a `G = 4` set, the
  metric is the host's bit for bit and `K` agrees to 5.6 eps of its largest
  value: the kernel is another call site ("Two spellings…" in `CLAUDE.md`).
  A call allocates nothing, and `Adapt` reaches both arrays.
- **On Metal at `Float32`** (a scratch environment with Metal added, the
  same mesh and kernel). The metric agrees with the CPU to 0.76 eps of its
  scale and `K` to 36 eps of its largest value, the stencil's roundoff.

**What `K` costs and buys** over the `Π_ij/√γ` that TreeGRRMHD proposed.

- **The cost.** `Π_ij/√γ` needs no stencil here. It could have been
  decoded in every ghost layer, so with `G ≥ 4` the consumer could have read
  this package's own ghosts and skipped a fill. `K` reaches only one ghost
  layer. So the consumer fills its vertex set itself, once each time it reads
  the metric. That fill is about the size of this package's own: 16
  variables at `G = 4` against 20 at `G = 3`, 1.03× at `N = 16` (estimated,
  not measured).
- **It buys a contract that is not GHSO2's.** These are the 3+1 variables
  a BSSN or Z4c code evolves and is sourced by. TreeBSSN, vertex-centered at
  `G = q/2 + 1` on the same mesh, decodes the same 16 values with no stencil
  at all (`γ_ij = e^{4φ} γ̃_ij`, `K_ij = e^{4φ}(Ã_ij + γ̃_ij K/3)`). Its matter
  terms are `(ρ, S_i, S_ij)` exactly. TreeGRRMHD never sees this package's
  `Π_ab = (√γ/α)(∂ₜ − β^k ∂_k) g_ab`, a normalization that is GHSO2's.
- **The matter reads `K` where it needs it.** `K_ij` enters the energy
  equation's source (`S^ij K_ij`). Holding a static star in equilibrium is
  the momentum source's job, which reads `∂α` and `∂γ` from TreeGRRMHD's own
  interpolant either way. For the TOV star `K = 0`.

### What this package takes (GH-3)

```julia
gh_rhs!(du, u, p, t; matter = nothing)     # or GH-4's prepare! and rhs!(du, p, t; matter)
at = matter(p.U)                           # bound to this package's field set, once a call
(; ρ, S, Sij) = at(n, b)                   # in the kernel, at an owned vertex
```

The matter enters through a keyword on `gh_rhs!` and not through `GHProblem`
([history](HISTORY.md#building-gh-3-2026-11-09)). The adapter is a snapshot
of TreeGRRMHD's state, and a coupled run passes a new one at every stage.

- **The adapter.**
  - `matter` is `nothing` (vacuum) or TreeGRRMHD's adapter.
  - `gh_rhs!` binds it to the state's field set on the host, and the
    binding checks that it can answer for that set.
  - The kernel calls the bound object at its owned vertices with the owned
    index and never stores the result. `S` is an `SVector{3,T}` and `Sij` a
    symmetric `SMatrix{3,3,T}`.
  - It is the kernel's seventh `Val`, `HASMATTER` (`W` is the sixth). Unlike
    the other six it is built in `gh_rhs!`, from the bound adapter's *type*,
    so it is just as static. `mat` and its `Val` are the kernel's last two
    arguments, after `W`, so the benchmarks that slice the argument list keep
    their positions.
  - With `false` the kernel is the vacuum one, bit for bit (measured below).
- **The term** is `matter_source` (`src/pointwise.jl`). The head's
  `metric_quantities` give this point's `α`, `β^i` and `γ^ij`. With them the
  covariant `T_ab` is
  ```
  T_tt = α² ρ − 2α β^i S_i + β^i β^j S_ij
  T_ti = −α S_i + β^j S_ij
  T_ij = S_ij
  T    = γ^ij S_ij − ρ
  ```
  and the source gains `16π α√γ (T_ab − g_ab T/2)` in the packed order.
  - It is written as `α√γ (16π T_ab − 8π g_ab T)`, so that nothing is halved.
  - The reasoning comes from `notes/formulation.md` §1. `R_ab = 8π(T_ab −
    g_ab T/2)` moves `S0` by `−16π T̄_ab`, and the source enters as
    `−α√γ S0`. The acceptance test is what pins the sign.
  - `gh_rhs_head` adds the term right after `gh_node_source_lean`, so it is
    part of `F`. Inside an interior layer it is weighted by `w`, like the rest
    of `F`. In a frozen core `F` is not evaluated, so the adapter is never
    asked there. `∂ₜh` is unchanged.
- **The monitors.** `adm_constraint!(p, u, t; matter)` takes the same keyword.
  - It subtracts `16πρ` from `ℋ` and `8πS_i` from `ℳ_i` directly; these are
    the quantities that cross.
  - The ADM monitor reconstructs `∂ₜ∂ₜg` from `∂ₜΠ`, and that `∂ₜΠ` gains the
    same `matter_source`. The term is added outside `gh_node_rhs_expanded`, so
    the three spellings of the vacuum source stay as `pointwise_identity_tests`
    compares them.
  - It needs the matter at the point only, so the owned vertices suffice.
  - `gh_constraint!` is unchanged, and so is the driver's record, which runs
    in vacuum.
- **The integrator.** `gh_integrator` and `gh_solve` take `matter` too.
  - They pass `gh_rhs!` with the adapter **held fixed** at every stage of
    every step. That is a spacetime evolving under a stress-energy that does
    not change: the anti-Cowling test below, and nothing a coupled run uses.
  - A coupled run's script calls `gh_rhs!` stage by stage itself.
- **The kernel price.** The adapter runs inside the right-hand side's
  kernel, the hottest code here.
  - It must be device-clean in the sense of `CLAUDE.md`'s "On a device a
    closure is a call". An uninlined call with `SVector` arguments costs what
    took the kernel from 8.5 to 1.1 ns a point on the H200. On a device the
    head calls it directly, with no closure.
  - On the CPU it is called once per SIMD lane, and the `W` results are
    packed into `Vec`s (`matter_at` in `src/lanes.jl`).
  - Its 10 values add to the head's live set. The head calls the adapter
    where it forms the source, so the 10 values are live only while they are
    added in. On the H200 that costs no time (measured below).

**Measured** (2026-11-09, `test/matter_tests.jl`, four testsets and 20
claims, quick tier).

- **Vacuum is bit for bit the kernel of before.** `du` was hashed before and
  after the change at `simd_width = 1` and at the default width on four
  cases: the gauge wave, the hole fixture (`:damped`, a sampled source), the
  octant hole (the algebraic source) and a static non-harmonic bump. The ADM
  monitor's `diag` was hashed on two of them. Every hash agrees.
- **The fixture.** A static star: Plummer's mass function and potential in
  areal coordinates, `M = 1/10`, `R = 1`, in the box `(−2, 2)³`. Its matter is
  `G_ab/8π` from SpacetimeMetrics' `EinsteinTensor`, sampled at the owned
  vertices by a test adapter. For the acceptance the slicing is tilted,
  `t → t + λ√(r² + R²)` with `λ = 3/10`, which makes the shift, `K_ij` and `S_i`
  nonzero.
- **Acceptance.** On the exact data, with its matter, these vanish at order
  `q = 4`:
  - the right-hand side, with rates 3.76, 3.87 and 3.92 pair by pair from
    `N = 8` to `20` (`roots = 2`);
  - the matter-corrected `ℋ`: 3.75, 3.87, 3.92;
  - `ℳ_i`: 3.71, 3.85, 3.91.

  Without the matter all three stay at the matter's size (rates within
  `0.01` of zero), 870–2450 times larger at `N = 16`.
- **SIMD.** With matter at `W = 4`, the lanes and the scalar kernel agree
  bit for bit on aarch64. On Symmetry's EPYC 7543 (AVX2) they agree to
  0.56 eps of the terms; there the two contract FMAs differently.
- **Anti-Cowling.** The star's exact data, at rest (`λ = 0`) and at the
  recipe's `ε_KO = 1/2`, evolved to `t = 4` (two light-crossing times of
  `2R`) with the matter held fixed:
  - they stay on the star at an error converging at 3.81 and 4.00 pair by
    pair from `N = 8` to `16`;
  - the constraints converge at 3.60–3.87 pair by pair.

  The same data evolved without the matter end `4800×` further from the star
  at `N = 16` (`1.4 × 10⁻²` against `2.9 × 10⁻⁶`): the matter is what holds
  the star static.
- **On an H200** (Symmetry jobs 594179 and 594183, `Float64`, the star at
  `q = 4`, `ε_KO = 1/2`, a sampled gauge source):
  - **The kernel's price.** It runs at `1.47` ns a point with matter and
    `1.52` without, on 64 blocks of `32³` (2.1 M points): no measurable cost.
  - **The SASS.** Both kernels use 253 registers. With matter the kernel has
    6672 instructions against 6464, and its spill traffic is `STL` 339 and
    `LDL` 424 against 328 and 421; the vacuum kernel already spills on this
    case. There is no `CALL` into the adapter, only CUDA's division, `sqrt`
    and reciprocal slow paths and the exception paths, as in vacuum.
  - **Against the CPU.** `du` with matter agrees with the CPU's scalar kernel
    to `2.2e−14`. That is 1.18 times one second-derivative stencil's roundoff,
    `eps · max|u| · Σ|w₂|/h²`, and the right-hand side sums six such
    stencils. The matter-corrected `ℋ` and `ℳ_i` agree to 36 eps of the
    vacuum constraints.
  - **Anti-Cowling on the device.** The run to `t = 4` reproduces the CPU's
    errors to `2.3 × 10⁻¹³` relative, converging at 3.89.
- **The full suite on Symmetry** (job 594178, EPYC 7543, four threads)
  passes 5721 of 5721 in 25m21, with this file in about 2 minutes.
- **Cost.** The file takes 75 s inside the quick tier at four threads (12m22
  for the whole tier), or 2m35 when run alone. Most of it is compiling
  `EinsteinTensor` for the star's type (about 75 s alone, three layers of
  duals). That is why
  one metric type serves every testset: a second type would cost another
  75 s.

### The verbs (GH-4)

```julia
const GH = TreeGeneralizedHarmonic
sets = (GH.evolved_sets(gh)..., MHD.evolved_sets(mhd)...)  # (gh.U,), then the matter's
u = statevector(sets); ug = view(u, staterange(sets, 1))   # this package's segment
GH.initial_state!(ug, gh, t)                 # the exact state, limited
adm = GH.ADMMetric(GH.prepare!(gh, ug, t))   # scatter and fill; GH-1's adapter of it
GH.rhs!(view(du, staterange(sets, 1)), gh, t; matter)      # the kernel alone (GH-3)
GH.stage_limit!(ug, gh, t); GH.step_limit!(ug, gh, t)      # the projection and the paste
λ = max_speed(gh, ug, t)                     # Δt = cfl · h_min / λ
```

A script that owns the time loop, TreeGRRMHD's coupled run, calls these on
this package's segment of one multi-set state vector, between the other
package's verbs of the same names ([history](HISTORY.md#building-gh-4-2026-11-09)).

- **Public, not exported.** `prepare!`, `rhs!`, `stage_limit!`,
  `step_limit!`, `initial_state!` and `evolved_sets` are declared `public`,
  so that loading this package beside TreeGRRMHD, whose verbs have the same
  names, makes nothing ambiguous; a script qualifies them. `max_speed` was
  already exported, and the speed is its method `max_speed(p, u, t)`.
- **`prepare!(p, u, t)` returns `p`**, as TreeGRRMHD's does. It scatters `u`
  into the working array and fills the ghosts with the Dirichlet hook of `t`.
  The metric adapter of the prepared state is `ADMMetric(p)`;
  `ADMMetric(p, u, t)` is `ADMMetric(prepare!(p, u, t))`.
- **`rhs!(du, p, t; matter)` reads the working array as `prepare!` left it.**
  It is `gh_rhs!`'s kernel launch, and `gh_rhs!` is `prepare!` and then
  `rhs!`, bit for bit the function of before. Nothing checks that the
  prepared state is the one meant: `max_speed`, the monitors, `gh_dt`,
  `initial_state!` and another `prepare!` all overwrite the working array,
  and a script calls none of them between the two.
- **`stage_limit!` and `step_limit!` are one function**: the range
  projection, then the `:pasted` paste, which this package passes as both
  IMEXRungeKutta limiters (`gh_limiter!` is `stage_limit!` under
  IMEXRungeKutta's signature). The script names each kind for each package.
  They write `u` where they fire and do not touch the working array, so a
  `prepare!` of the same `u` is stale after one that wrote.
- **`max_speed(p, u, t)`** is the checked speed the driver steps by: it
  refuses a non-finite value at an evolved point and a speed that is not
  finite and positive, each with an `ArgumentError`. It scatters `u`.
- **`initial_state!(u, p, t)`** is `fill_exact!` through the problem's own
  interior, `gather!` and `stage_limit!`: an analytic case's exact state,
  limited. The initial-data cycle, perturbations and a fitted target's data
  are `evolve!`'s.
- **Segments.** Every verb takes TreeAMR's segment view `view(u,
  staterange(sets, i))` in place of a vector of its own. On the CPU the
  kernel's SIMD lanes store `du` through a pointer into it: `Lanes` admits a
  dense array and the reshape of a view with one unit range, which is
  contiguous, and nothing else. On a device a contiguous view of a device
  array is a device array.
- **`evolve!` is a composition of them.** Its integrator's right-hand side
  is `gh_rhs!`, so `prepare!` and `rhs!`. Its limiter is `stage_limit!`, both
  after the initial data and after a regrid, and its speed is
  `max_speed(p, u, t)`. Its record, the regrid and the checkpoints stay its
  own.
- **What the script does not get here.** TreeGRRMHD's step-11 sketch also
  names the following, which are not built:
  - a `GHProblem(forest, case; q)` that builds its own field set and
    schedule;
  - a `GHCase` from a bare metric;
  - `constraint_norms(p, u, t)`;
  - the regrid branch's `flags`, `regrid_pairs` and `rebuild`.

  These wait for step 11 to need them (**open**).

**Measured** (2026-11-09, `test/verbs_tests.jl`, two testsets and 23
claims, quick tier):

- **Vacuum is still bit for bit.** `du` hashes are unchanged for all four
  cases of GH-3's check at both widths.
- **A script is the run.** On the tilted star with its matter (`q = 4`,
  `N = 8`, `roots = 2`), a script runs IMEXRungeKutta's RK4 over a state
  vector of two sets, a three-variable cell-centered set first and this
  package's segment second, at an offset. It calls `prepare!`, `rhs!` with
  the matter, `stage_limit!` and `step_limit!` on the segment. The result
  is `gh_solve`'s run bit for bit at the default SIMD width, and the other
  segment keeps its bits.
- **On a segment.** On the hole fixture (`:pasted`, with bounds) the verbs
  on a segment are those on a vector of its own, bit for bit: the limiters
  on a planted `γ_xx < 0` inside the gate and a planted value in the ball,
  the speed, and `initial_state!`. The speed refuses a planted `NaN` at an
  evolved point.
- **On an H200** (Symmetry job 594413, `Float64`, the same two cases with
  every array on the device, 18 claims):
  - **A script is the run there too.** The script's integrator over a device
    state vector of two sets is `gh_solve`'s on the device bit for bit, and
    the other segment keeps its bits. `initial_state!`, `prepare!` with
    `rhs!`, and the speed on a segment are a vector of its own's. Against
    the CPU's scalar kernel, `du` agrees to `6.1 × 10⁻¹³` relative and the
    state at `t = 1/2` to `3 × 10⁻¹⁵`.
  - **The limiters on a device segment** are those on a vector of its own,
    bit for bit, on the hole fixture's planted values; they agree with the
    CPU to `1.3 × 10⁻¹⁴` relative. The speed refuses the planted `NaN`.
  - **No price for the split.** On 64 blocks of `32³` (2.1 M points),
    `prepare!` and `rhs!` with matter on a segment take `4.556` ms against
    `gh_rhs!`'s `4.550` on a vector of its own; `prepare!` alone (the
    scatter and the ghost fill) is `1.45` ms of it, and `max_speed` `1.31`.
- **Cost.** About 6 s inside the quick tier at four threads, after
  `matter_tests.jl` has compiled the star; the tier passes 5379 with its 26
  hole evolutions skipped, in 12m36.
- **The full suite on Symmetry** (job 594412, EPYC 7543, four threads)
  passes 5744 of 5744 in 27m48, with this file in 12 s.

### Why 3+1 variables at the vertices

- **Positivity survives the averages.** The mean of `ρ` over eight cells
  is a convex combination, so `ρ ≥ 0` holds at the vertex exactly. `T_ab` is
  then assembled with the vertex's own `α` and `β`, the metric this package
  evolves. The alternative averaged eight cells' `T_ab`, each assembled with
  that cell's interpolated lapse and shift.
- **TreeGRRMHD can keep the metric's signature.** Multilinear weights make
  `(α, γ_ij)` a convex combination of the corners, so its fallback keeps
  `α > 0` and `γ` positive definite where the 4th-order interpolant does
  not. An `(α, β, γ)` with those properties always rebuilds a Lorentzian
  metric. Interpolating `g_ab` could not promise that.

**What TreeGRRMHD does with it** (proposed; its `CODE.md` has the details):

- **A vertex set of its own.** It keeps a 16-variable vertex set on the same
  forest, which needs `N ≥ 10`. The owned points come from `ADMMetric`;
  TreeAMR fills the ghosts at `G ≥ 4`, with an outer-boundary hook of
  TreeGRRMHD's own. The prolongation
  must keep the derivatives it takes 4th order next to a coarse-fine face:
  this package's interface rule, `p = q + 2` for second derivatives, gives
  `p ≥ q + 1` for first derivatives.
- **Lagrange interpolation, with a fallback.** It interpolates with 4, 16
  and 64 points to edges, faces and cells. Where the result has `α ≤ 0`, or
  a `γ` that fails a Cholesky factorization, it falls back to multilinear
  weights, and counts the fallbacks.
- **Matter at the cells.** It forms `(ρ, S_i, S_ij)` at the cells, then
  averages them to the vertices.

**Open on this side:**

- the adapter's name (`ADMMetric`; `ADMVars` is ApparentHorizonFinder's);
- the names TreeGRRMHD's step-11 sketch uses that GH-4 did not build (above);
- an H200 run of the metric adapter (Metal and the CPU only so far); the
  matter term and the verbs have run there (above).

## Public access to the vertex metric

A code that couples to this package's metric — TreeGRRMHD, whose GH-1 asks
for it — reads `α`, `β^i`, `γ_ij` and `K_ij` at the vertices of the GH field
set. It does so through one exported function, `src/admvars.jl`'s `adm_vars`,
rather than through the packing helpers, the variable layout of `U.work` and
the stencils, none of which become public:

```julia
adm_vars(work, I, b)              -> (; α, β, γ, sqrtγ, γu)
adm_vars(work, I, b, inv_h, Val(q)) -> (; α, β, γ, sqrtγ, γu, K)
```

- **Arguments.** `work` is the scattered, ghost-filled working array
  `(n, n, n, nvars, nblocks)`. `I` is the **stored** index, ghosts included,
  the index `coordinates` takes. `b` is the block, `inv_h` is the block's
  `1/h`, and `q` is the right-hand side's order. `inv_h` must have the
  array's element type `T`. A spacing of another precision is refused,
  because it would promote `∂h` and make `K`
  mixed-precision.
- **Where each method is valid.** The first reads one point and is valid at
  every stored vertex. The second reads the right-hand side's own
  first-derivative stencil, `q/2` points each way. It is valid where
  `q/2 < I[d] ≤ n − q/2`, which is the owned vertices and one ghost layer,
  since `G = q/2 + 1`. A `@boundscheck` throws the `BoundsError` of the
  farthest point the stencil would read, because `work` is read linearly and
  a stencil past the `i` edge would otherwise read the previous row in
  silence.
- **Definitions.** `α`, `β`, `√γ` and `γ^{ij}` are `metric_quantities`', the
  offset-accurate path the right-hand side takes. `γ` is `η + h`'s spatial
  block. `K` (the NR sign, SpacetimeMetrics' `ExtrinsicCurvature`) comes from
  `Π` and from `∂_i h` formed exactly as `gh_rhs_head` forms it. The `K`
  algebra is `pointwise.jl`'s `_extrinsic_curvature`, which
  `adm_vars_from_state` shares. It is GHSO2's expression term for term,
  written with `@ntuple` instead of generators, and is bit for bit the old
  expression on the same inputs (measured on 20 000 random states at
  `Float64` and `Float32`). The two functions still differ in the last
  places, because `metric_quantities` is inlined into two call sites ("Two
  spellings…" in `CLAUDE.md`).
- **The interior is the caller's.** Inside a hole's core the stored data are
  stale and the result means nothing.

A downstream code decodes once per vertex and interpolates the ADM variables
(16 values, or 17 with `√γ`) to cells, faces and edges itself. With no mesh
filler here, it writes its own `map_blocks!` into an ADM field set of its
own, which it can ghost-fill at any `G` it needs.

**Measured.**

- **On the CPU.** On a boosted spinning Kerr-Schild hole (`a = 1/2`,
  `v = (1/5, −1/10, 0)`), the metric agrees with `adm_decompose` to under
  `64 eps` at every stored vertex.
- **Convergence.** `K` converges to `ExtrinsicCurvature` at order 2.00,
  4.00 and 5.99 at `q = 2`, 4 and 6 (`h = 1/8 … 1/32`, `|K| ≈ 0.13`). The
  error at `h = 1/32` is `8.6e−6`, `6.4e−9` and `2.3e−12`.
- **Allocation.** Neither method allocates.
- **Metal.** On Metal at `Float32`, both methods in one kernel agree with
  the CPU to `2.5 eps`.
- **H200 values** (Symmetry jobs 590175 and 590181). At both precisions and
  at `q = 2, 4, 6`, the metric agrees with the CPU to `7 eps`. `K` agrees to
  `0.03–0.36` of its stencil's roundoff, `eps · max|u| · Σ|w|/h`. Measured
  relative to `|K|`, that is up to `900 eps` at `q = 6`, which is
  cancellation, not error.
- **H200 SASS.** Each kernel is one function. Its `CALL`s are CUDA's
  division, reciprocal and `sqrt` slow paths, `sqrt`'s `DomainError`, the
  bounds checks and the exception reporting. There is no call into Julia
  code. The `K` method stores 36 values to local memory with `@inbounds`
  (43 without), and the metric method 9.

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
  gauge wave with a regrid and
  compares digests. It holds. The workload is
  the gauge wave on a `2³` root grid: the initial-data cycle with its
  flagging pass, two chunks of fixed-step RK4 on a two-level mesh, the
  gauge-constraint monitor and its masked norms, `max_speed`, and a regrid
  that moves data between levels — six printed lines of digests, norms and
  mesh statistics, **character for character identical** at one and at
  four threads. The workload carries the *gauge* monitor and not the ADM
  one: the ADM kernel is the same `map_blocks!`
  launch reduced by the same fold, so it adds no parallel structure, and
  compiling it costs 18 s in each of the two processes the test runs.
  What this bit-identity does *not* mean is under
  [Measured results](#measured-results): the same compiled code, at the
  same call site, at a different thread count, and nothing more.
  The workload stays the gauge wave and does
  *not* grow a chunk of a hole with its layer. What the interior
  adds is a per-point branch inside one `map_blocks!` kernel, two more
  `map_blocks!` launches (the error kernel and the `:pasted` limiter) and
  three more `masked_norms` folds — no new parallel *structure*, and the
  workload already exercises a `map_blocks!` kernel, a masked fold and
  `max_speed`. What it would cost is a second background's dual passes and
  two more kernel specialisations, in each of the two subprocesses the
  test runs, for a claim the existing lines already make. The right time
  to reconsider is G5, where the hole *moves* and the refinement follows
  it — that is a new flagging pass, which is new parallel structure.

  **Inconsistent:** this bullet says every reduction goes through
  `block_mapreduce` or `firing_boxes`; "Upstream prerequisites", item 3,
  and `CLAUDE.md`, "Things that will bite", say every reduction that
  crosses blocks is TreeAMR's `mesh_mapreduce` or `mesh_fold`, never a host
  fold over `block_mapreduce`'s per-block vector.
- **Every per-block pass runs on the block's owner thread, and this
  package adds none of its own.** TreeAMR gives
  block `b` to the thread whose chunk of `threadchunks(nblocks)` holds it,
  in every `map_blocks!`, `scatter!`/`gather!`, `fill_by_coordinates!`,
  ghost fill, interface fixup, regrid transfer and host loop over blocks,
  because a block that changes core between launches streams at up to
  `1/2.4` the rate on a 64-core EPYC (TreeAMR's `CODE.md`, "What one
  process loses"). Every kernel here — the right-hand side, the limiters,
  the monitors, the indicator, the target cache — already goes through
  `map_blocks!`, and every state vector and field set is allocated by
  TreeAMR, so all of that is inherited with no change to `src/`. The horizon interpolator's batch is TreeAMR's `interpolate`, one KernelAbstractions
  launch over the points — not a by-owner launch, since points are not
  blocks — which flags rather than throws, so the guard's refusal is raised
  on the host after it; this package has no parallel loop of its own. RK4's stage arithmetic is owner-based too: the integrator is
  IMEXRungeKutta's RK4 by block owner, and pinned with first touch is the
  fastest configuration — [Time integration](#time-integration) has the
  numbers ([history](HISTORY.md#precision-threads-devices)).
- **`Float64` on Symmetry's H200 is the requirement.** It is the machine the proof of concept runs on, and the
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
  H200. On `CPU()`, which is what a host test can
  settle, there is no failure to record. These
  run at `Float32` and return `Float32`: the fused kernel, the
  time step, RK4, both constraint monitors, the two-level mesh with its
  prolongated ghosts. The gauge wave converges at **3.93** at `Float32`
  against `Float64`'s **3.94** over the two coarsest resolutions, with
  errors `2.00e−4, 1.31e−5` against `2.00e−4, 1.30e−5` — the *same* error,
  not merely the same rate. Two resolutions and no more, for this
  reason: the roundoff floor of a second derivative is `eps/h²`, which at
  `Float32` and `h = 1/24` is already the size of the `Float64` truncation
  error there, so a third point would measure the arithmetic. The
  pointwise algebra — including `metric_derivatives_along` and
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
  compared to. The measurements are in [The right-hand side on an
  H200](#the-right-hand-side-on-an-h200):
  - **The arithmetic:** 3867 FP64 instructions a point, about 6.5 kflop, inside the
    prediction.
  - **The cost was neither the arithmetic nor the traffic:** the kernel held 255
    registers and spilled 3.8–8 KB a thread, at about 5 % of FP64 peak.
  - **The lean source alone:** 52 % of FP64 peak.
  - **Around the kernel,** TreeAMR's scatter and ghost fill run at 0.65–1 TB/s.

**A closure in a kernel must not capture a name assigned twice in that body.** Lowering boxes such a variable — even when the
  two assignments are in branches a `Val` compiles away — and a `Core.Box`
  is untyped: `gh_rhs_kernel!`'s `:none` branch and its layer branch both
  assigned `∂ₜh, ∂ₜΠ`, which the `ntuple(Val(NC)) do v … end` store
  captured, so every right-hand side on the CPU allocated ~570 bytes a
  point (7–15 % of its time at four threads, and 3–4× at 64, where the
  threads contend for the garbage collector) and no device would compile
  the kernel at all ("unsupported dynamic function invocation (call to
  getindex)"). The symptom on the CPU is
  `@allocated gh_rhs!(…)` growing with the block count; the ghost fill's
  13 kB is fixed. `Profile.Allocs` names the line.

## Running distributed

A run
is distributed over MPI, through TreeAMR's M7, by building its forest over a communicator —
`gh_forest(…; comm)`, `hole_forest(…; comm)`, an `MPI.Comm` with `using MPI`
and `MPI.Init()` done — and calling `evolve!` on every rank with the same
arguments; a restart takes `evolve!(…; comm)` and distributes the
checkpoint's forest over it. TreeAMR keeps the forest whole on every rank
and gives each rank a contiguous run of the Morton curve's leaves, and does
everything that crosses ranks itself: the ghost exchange (bit-identical to a
serial fill), the interface restriction, the regrid with its repartition,
`mesh_mapreduce` (an `allgather` of one partial per rank, folded in rank
order), the point interpolation (each point routed to its owner) and, in
TreeIOHDF5, the checkpoint (an index and one part file per I/O group,
readable at any rank count). **This package never loads MPI**: the caller
does, and MPI is a test dependency only — TreeHydro's arrangement.

**Multi-GPU is the same thing.** One rank per GPU, each rank's field sets on
its own device; device buffers are staged through page-locked host memory by
default, or handed to MPI directly with `communicator(comm; deviceaware =
true)` on a CUDA-aware MPI. Choosing a rank's device is the caller's job (the
rank's place on its node, `MPI.Comm_split_type(…, COMM_TYPE_SHARED)`), and
`test/octant_runs.jl mpi=true backend=cuda` does it. TreeAMR measured its
right-hand side at 0.71–0.74 weak-scaling efficiency on four H200s of one
node, bit-identical to serial apart from sums.

What was left for this package, `src/distributed.jl` and the paths it fixed:

- **Agreements** (TreeHydro's `rank_reduce`, `agree_any`, `agree_refusal`,
  `isroot`): the initial data's finiteness count, the refinement centroid and
  its firing count, the wall-clock decisions to checkpoint and to stop, the
  file-system checks of the checkpoint keywords, the level bounds' refusal
  (each rank checks its own blocks), the fits' timings that the checkpoint
  holds. Checkpoint rotation, which also removes part files now, is rank 0's.
- **The travelling margin's flags are the whole forest's.** `buffered_flags`
  recruits neighbours that may be another rank's, so `refine_flags` gathers
  every rank's marks as TreeAMR's `isbits` `RegridMark`s, dilates and clamps
  the whole vector alike on every rank, and hands `regrid!` this rank's slice.
- **`add_noise!` draws per block**: a `Xoshiro` seeded from one draw of the
  caller's generator and the block's Morton key, so the noise is a function
  of the seed and the mesh alone.
- **The host analysis runs on rank 0 alone.** The
  horizon find, the track's update and the fitted target's fits are host
  computations whose iteration counts nobody knows in advance and which read
  the state through the collective `interpolate`. Rank 0 runs them
  (`on_root`); every collective step it takes inside — a scatter and ghost
  fill (`fill_state!`), an interpolation (`interpolate_state`) — is
  announced by a broadcast request first, and the other ranks take the same
  step with no points of their own until rank 0 says stop and broadcasts its
  result, which every rank adopts (a fit's coefficients as its host copy,
  uploaded on each rank). The track, the geometry, the fits and the record's
  horizon rows are therefore rank 0's bits everywhere, whatever each node's
  BLAS would have made of the fit's least squares; an exception in the
  analysis is rank 0's and is rethrown on every rank. The cost is that the
  analysis does not shrink as ranks are added — it did not before either —
  and the Korzyński spin, 16–105 right-hand sides a find, remains the lever.

**The claim, and its test.** A distributed run is the serial run: its state,
mesh, step, chunk and regrid counts, every speed and every maximum, the
horizon's numbers, the track and the fits bit for bit at any rank count, and
its sums — the L2 norms and the refinement centroid — to roundoff, exactly at
one rank. `test/mpi_workload.jl` runs eight cases under `mpiexec -n 3` over
three, two and one ranks and serially in one launch (TreeHydro's pattern):
the gauge wave on one block with noise (ranks with no block), on eight
blocks, the adaptive hole through the initial-data cycle, the tracked
`:fitted` hole that regrids, the static hole checkpointed and
restarted at another rank count from the three-rank checkpoint, a wall-clock
stop, rotation with part files, and a restart file that exists on one rank
only. `test/mpi_tests.jl` compares the lines. Measured on
the development machine, MPICH from `MPI.jl`: every line agrees — 81 a group,
12 of them sums that differ in the last bits at three ranks and agree to
roundoff — in 4m37 for the launch, most of it three ranks compiling; and
`test/octant_runs.jl mpi=true` at two ranks writes the serial run's CSV
character for character on the tracked `:fitted` octant hole. Where the machine has no room for the three ranks beside the
suite — CI's threaded Linux cell, four cores and 16 GB — they run to their end
*before* the suite (`test/mpi_jobs.jl`), and `mpi_tests.jl` compares what they
wrote ([history](HISTORY.md#running-distributed)). Measured on the development machine
(`/usr/bin/time -l`, macOS): the suite's process peaks at 5.0 GB,
and each rank, the workload running alone for 240 s, at 3.4–3.8 GB resident and
5.2–5.4 GB of footprint — the ranks together about 11 GB resident and 16 GB of
footprint, close to what that runner holds on their own.

**On Symmetry's H200s (job 578514, cn113, four of its
eight H200s; `bench/symmetry_mpi_gpu.sh`, environments by
`bench/symmetry_mpi_gpu_env.sh`).** `test/octant_runs.jl`'s tracked `:fitted`
Kerr-Schild hole on the reflecting octant — `L = 32`, `N = 32`, `4³` roots,
`radii = 16, 8, 4`, 232 blocks and 7.6 M points, `q = 4`, `cfl = 1/2`, the
finder with the spin and the fit every chunk of `M/2`, 224 steps to `t = 2` —
on one GPU with no MPI and then one rank a GPU:

| run | chunk 2 | chunks 3, 4 | the whole run |
|---|---|---|---|
| serial, 1 GPU | 27.0 s | 18.9, 18.8 s | 267 s |
| MPICH_jll, 1 rank | 26.4 s | 18.7, 18.8 s | 273 s |
| MPICH_jll, 2 ranks | 23.3 s | 16.0, 16.0 s | 262 s |
| MPICH_jll, 4 ranks | 25.8 s | 18.0, 17.9 s | 274 s |
| HPC-X, 2 ranks, staged / direct | 56.2 / 57.5 s | 48.8, 49.4 / 49.8, 49.9 s | 462 / 462 s |
| HPC-X, 4 ranks, staged / direct | 54.9 / 58.5 s | 45.7, 46.2 / 49.3, 49.4 s | 444 / 471 s |

- **Every run is the serial H200 run**: `octant.csv` and `records.csv` — the
  monitors' maxima, the horizon's area, `M_irr`, `J` and radii, the track, the
  fit — the same text, the L2 sums to `2 · 10⁻¹⁶` (`bench/octant_compare.jl`),
  through MPICH's host staging and through HPC-X's both ways.
- **The wall clock of this run does not scale with the GPUs, and that is
  the run, not the evolution.** A chunk's 224 right-hand sides are a few
  seconds of it; the rest is the analysis at every chunk boundary — the find
  with the Korzyński spin and the fit on rank 0, and the script's observer,
  the ADM monitor and seven shells of masked norms — which no rank count
  shortens. `bench/mpi_steps.jl` times a step alone, which is what the GPUs
  divide.
- **HPC-X is 2.5–3× slower than MPICH_jll here, staged and direct alike**, so
  the messages are not where (the direct path changes nothing).
- **The HPC-X environment needs `OpenMPI_jll` pointed at HPC-X's libraries**:
  HDF5_jll follows MPIPreferences' ABI to the Open MPI build and loads
  `OpenMPI_jll`, whose own `libopen-pal` does not load beside the module's
  (`undefined symbol: pmix_framework_names`). `bench/symmetry_mpi_gpu_env.sh`
  sets the JLL's library paths to the module's.

**The step alone (job 578572, cn113;
`bench/symmetry_mpi_steps.sh`).** `bench/mpi_steps.jl` on the same octant mesh
with the analytic `:damped` layer and the algebraic source — no finder, no
fit, no observer — the wall clock of a step from two single-chunk runs of 7
and 56 steps, in ms (and points a nanosecond):

| ranks (GPUs) | 7.6 M points (`N = 32`) | 25.7 M points (`N = 48`, `L = 48`) |
|---|---|---|
| no MPI | 159.2 (0.048) | 484.1 (0.053) |
| MPICH_jll, 1 | 160.3 (0.047) | 485.7 (0.053) |
| MPICH_jll, 2 | 149.5 (0.051) | 363.8 (0.071) |
| MPICH_jll, 4 | 179.1 (0.042) | 340.4 (0.075) |
| HPC-X staged, 2 | 111.2 (0.068) | — |

- **The step does not strong-scale yet: 1.33× and 1.43× on 2 and 4 GPUs at
  25.7 M points, and none at 7.6 M.** One GPU is throughput-bound — 3.4× the
  points cost 3.0× the time, about 19 ns a point a step. On a GPU the kernel is most of a step, not the ghost fill and
  prolongation as on the CPU — the breakdown below.
- **The messages are a large part of what is left.** HPC-X's UCX moves the
  same staged messages in a step 26 % shorter than MPICH_jll's at 2 ranks
  (111 against 150 ms) — so MPICH_jll's shared-memory path is slow for these
  sizes — and 4 ranks are slower than 2 at 7.6 M points. TreeAMR's own
  weak-scaling runs (0.71–0.74) were on a uniform and a two-level mesh; this
  one has three nested coarse-fine shells, and a split by equal block counts
  may leave the prolongation-heavy blocks on few ranks (a guess, not traced).
- **And HPC-X's 2.5–3× in job 578514 is in the analysis, not the step**:
  rank 0's host work — the find, the spin, the fit. It
  is Open MPI's core binding, below.

**Where a step goes, rank by rank (job 578648 on cn113's
H200s, and the same mesh at `N = 16` on the development machine's CPU, one
thread a rank).** `bench/mpi_breakdown.jl` times, on every rank by itself,
the scatter, a whole ghost fill, each fill stage's local groups, packs and
unpacks (TreeAMR's internals, as its `bench/mpi.jl` does — TreeAMR has no
timers of its own), the fused kernel, one right-hand side and one RK4 step,
and prints each rank's load: its blocks by level and the transfers, target
points and stencil taps its groups compute, by kind. A fill is one stage of
copies and restrictions and then **one stage per prolongation level,
coarsest first, run in sequence**, each with its own messages. On the GPU,
7.6 M points, in ms (the slowest rank):

| run | fill | kernel | rhs | step | ×1 GPU |
|---|---|---|---|---|---|
| 1 GPU, no MPI | 9.5 | 28.5 | 38.0 | 156.8 | 1 |
| MPICH_jll, 2 | 22.4 | 14.3 | 36.7 | 149.0 | 1.05 |
| MPICH_jll, 4 | 42.0 | 7.2 | 49.7 | 181.4 | 0.86 |
| HPC-X staged, 2 | 11.6 | 14.3 | 26.0 | 104.3 | 1.50 |
| HPC-X direct, 2 | 12.8 | 14.2 | 27.2 | 111.1 | 1.41 |
| HPC-X staged, 4 | 15.1 | 7.3 | 22.5 | 82.8 | 1.89 |
| HPC-X direct, 4 | 16.3 | 7.2 | 23.6 | 88.4 | 1.77 |

- **The kernel divides exactly**: 28.5, 14.3, 7.2 ms, every rank within 1 %
  of the others — equal block counts balance it, layer and core included. On
  one GPU it is three quarters of a right-hand side (3.75 ns a point with the
  `:damped` layer and the algebraic source), the fill a quarter: phase 1
  2.3 ms and each of the three prolongation stages 1.8 ms.
- **The fill does not divide, for two reasons.**
  - *Each prolongation level is one rank's work.* A level's coarse-fine
    shell is a contiguous stretch of the Morton curve, so the equal-count
    split puts it on one rank, and the sender — the owner of the coarse
    source blocks — computes the prolonged ghosts it sends. At 4 ranks rank 2
    prolongs level 2 (1.7 ms of packs) and rank 1 level 3 (1.7 ms), in
    sequence, while rank 0, which holds all 58 level-3 blocks, computes no
    prolongation at all and waits. The stages' critical path is about 10 ms
    at every rank count: the serial fill's.
  - *The messages are big.* A prolongation stage sends 35–38 MB of prolonged
    fine ghosts, phase 1 7–27 MB a peer (20 variables of `Float64`). Through
    MPICH_jll's shared memory with host staging that is 7–10 ms a stage —
    the fill's 22 and 42 ms. HPC-X's UCX moves them in about 2–4 ms of a
    fill, and handing it the device buffers directly (`deviceaware`) is no
    faster: the stage chain, not the transport, is what is left.
- **The CPU shows the first reason alone and larger** (`N = 16`, 0.95 M
  points, ms): a right-hand side is 1266 on one rank — fill 850, of which the
  three prolongation stages 600, the kernel 450 — and 998 and 834 on two and
  four, the kernel dividing (231, 121) and the fill not (774, 727), each
  prolongation level's ~200 ms on one rank. A step is about four times
  (fill + kernel/P): 5245, 3942, 3083 ms measured. The prolongation is
  expensive there because TreeAMR's transfer kernel sums the full tensor
  product, `6³ = 216` coarse points per prolonged point and variable: 63 M
  taps a fill against 1.4 M for all the copies, though the prolonged points
  (292 k) are a third of the owned ones.
- **Open MPI binds each rank to one core (two hardware threads) by
  default**, while a rank runs eight Julia threads: `/proc/self/status`'s
  `Cpus_allowed_list` reads 2 under HPC-X's `mpiexec` and 32 under MPICH's.
  The step does not notice — it is the GPU's — but rank 0's host analysis
  does: with `mpiexec --bind-to none`, the chunk of job 578514's octant
  `:fitted` run on 2 GPUs that took 48.8 s bound and 16.0 s under MPICH takes
  **12.8 s**. That was the 2.5–3×.

**What to run with, then:**
HPC-X's Open MPI, through `bench/symmetry_mpi_gpu_env.sh`'s environment
(`OpenMPI_jll` pointed at its libraries), launched with **`--bind-to none`**
— 1.89× on 4 GPUs against MPICH_jll's 0.86×, and the analysis 3.8× faster
than bound — with host staging, which is as fast as the direct path here.
`bench/symmetry_mpi_*.sh` pass the flag. MPICH_jll stays for the CPU and
the test suite.

**A mesh that follows a hole runs the same way.** Step H2's `:fitted` hole at
`1/48` on the following mesh (`test/octant_runs.jl mesh=follow mpi=true
deviceaware=false`, `70–95 M` points over the crossing, a regrid every chunk)
did not fit one H200 and ran `40 M` on two of one node, `mpiexec --bind-to none
-n 2` with host staging, at `160 s/M`: about `1.5×` one H200's rate scaled from
`1/40`. Its first rows equal the one-H200 run's to every printed digit
([step H2](SINGULARITY_HANDLING.md#a-moving-harmonic-hole-step-h2-2026-11-12)).

**What TreeAMR needs for the step to scale further** (the fill is 9.5 of a
38 ms right-hand side on one GPU and 15 of 22.5 on four; on the CPU it is
two thirds of one), in the order of what it buys here:

1. **Spread each prolongation stage over the ranks.** The stages are
   sequential, so the fill's critical path is the sum over levels of each
   level's prolongation on the one rank that owns it — balancing the total
   cost, the cost-weighted contiguous split TreeAMR's `CODE.md` anticipates,
   does not help, because a level's shell would still be one stretch of the
   curve. What would: a partition balanced *per level* (each level's leaves
   split evenly over the ranks — non-contiguous ranges, which touches the
   state vector's order and the checkpoint layout), or prolongation computed
   where its sources' *neighbourhood* is spread, or overlapping a rank's
   kernel on blocks whose ghosts are complete with the stages it only waits
   in (rank 0 at 4 ranks waits through all three).
2. **Sum-factorized prolongation**, as M11's point interpolation already is
   (its `CODE.md`, "Sum factorization along dimension 1"): `6 + 6 + 6`
   rather than `6³` multiplications per prolonged point and variable along a
   row, and half the fine points of a vertex-centered refinement coincide
   with coarse ones per dimension. On the CPU that is most of the 600 ms of a
   1266 ms right-hand side; on the GPU 5.4 of 9.5 ms of a fill.
3. **Smaller prolongation messages**: the receiver prolonging from the coarse
   source data instead of the sender sending prolonged fine ghosts, whose
   35–38 MB a stage are what made MPICH_jll's fill 42 ms.

Nothing in this package's own code is on the fill's path: the kernel and the
integrator (a step is four right-hand sides plus 2–3 %) divide by the ranks.

Not yet measured: `test/hole_runs.jl`, whose workers have no `mpi` option,
and GPUs across nodes.

`evolve!` refuses a forest over another communicator than its `comm`, and
`load_run` passes `comm` on. `src/distributed.jl` has `rank_bcast` beside
`on_root`, `SimWatchWriter(; comm)` is collective, and `evolve!` has
`checkpoint_io`.

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
- **Checkpoint and restart** is TreeAMR's M9a, and
  [Checkpoint and restart](#checkpoint-and-restart) is its section.
- **A live status file for SimWatch**: `src/simwatch.jl`
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
  it takes a fourth argument (`observer(p, t, u, row)`),
  so the horizon's numbers are not recomputed; `test/octant_runs.jl` writes
  everything its CSV carries — constraints overall, by level, in shells and
  at the boundary, the error, the horizon, the track, the fit and the step —
  and `finished`, `stopped` or `failed` at the end. For SimWatch 1.3.0, the writer has the reference writer's `group`,
  `progress.fraction`, `slurm` keys and the chain's `previous_job_ids`, read
  from the status file an earlier job left; the thumbnails
  (`test/simwatch_plots.jl`) are SVG written by hand on a transparent
  background with `currentColor` axes, which SimWatch draws with its own font
  and adapts to a light or dark terminal as its user configures; and `bench/simwatch.sh` is the batch jobs' copy of its
  shell helper.

There is no I/O and no CLI: those are step 9.

**Inconsistent:** "There is no I/O" (from `CLAUDE.md`'s former "Current
state", written at step 7) against this section's SimWatch status file,
`src/simwatch.jl`, and [Checkpoint and restart](#checkpoint-and-restart).

`SimWatchWriter` also writes a `summary` and the `[history]` window
SimWatch draws as sparklines, and `simwatch_history!` refills it after a
restart.

## Checkpoint and restart

The mechanism is
TreeHydro's — its `CODE.md`, "Checkpoint and restart" — on
purpose, so that the applications of TreeAMR checkpoint alike, with the
two differences marked; `save_checkpoint` and `load_checkpoint` are
TreeIOHDF5's. G5's crossing is eleven hours on a node, the
`a = 9/10` run `38`–`149 h`, and a queue's day is the limit. **Everything
about the file is upstream**: the forest, the field-set layout, the atomic
and durable write, element types as limbs, the provenance and the refusal
of a file a version cannot read — "no mesh machinery" applied to I/O. What
`src/checkpoint.jl` holds is when to write, the run state, the refusal of a
restart with other parameters, and the files' names and rotation.

**The checkpoint code is a hard dependency** (*TreeHydro
leaves it to the caller*). `TreeGeneralizedHarmonic.jl` has `using
TreeIOHDF5`, TreeAMR's companion package for checkpoints, so a run can always
be checkpointed and no job script has to load anything; `src/` uses HDF5 for
nothing else, so HDF5 itself is TreeIOHDF5's dependency, not this package's. `prerequisite_tests.jl` asserts that `save_checkpoint` and
`load_checkpoint` are TreeIOHDF5's and that TreeAMR defines neither, which
would make them ambiguous inside the package.

**Where: at a chunk boundary, after the analysis row and before the regrid**
(*TreeHydro writes after the regrid*). At a chunk
boundary the fixed-step integrator holds nothing but `(t, u)` —
IMEXRungeKutta's scratch carries no value from one step to the next — so a
restart that restores the state and the run state begins where the
uninterrupted run would. Before the regrid, so that a restart can **regrid
with a changed criterion** as the first thing it does, before its next
step: the flags are computed again from the restored state with the
indicator the restart is given — the record computes them last, from
nothing the other rows leave behind, so with the same criterion they are
the same flags — and the regrid is the loop's own code (`regrid_forest` and `rebuild` in
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
next chunk and cannot be recomputed from `(forest, u, t)`; that is a good deal more than TreeHydro's accumulators:

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
  moved half a cell);
- the **speed growth** `λ_end/λ` that sizes a moving hole's next step, in `T`, and the chunk's relaxation rate;
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

**The recipe and the criterion** (*TreeHydro's recipe
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

**What it amounts to** (`test/checkpoint_tests.jl`): a
chain of restarts, one chunk per job, is the uninterrupted run **bit for
bit** — state, mesh, record, counters, track, accounting and the fits'
coefficients — on the step-5 fixture with the range projection on, on the
adaptive fixture through the regrid that moves its mesh, on a tracked
`:fitted` Kerr-Schild hole through a regrid (its gauge source re-sampled),
and on a moving `:fitted` harmonic hole with the trailing ramp; a
finished run continued from its last chunk is the longer run; a restart that
no longer coarsens keeps the corner the uninterrupted run removes. On a real
row — `hole_runs.jl moving=l0-trail-9 t_end=1/2`, the boosted `a = 0` hole
on 820 blocks — the two-job chain's last row is the uninterrupted run's in
every printed digit, at the same 84 steps, from a `68 MB` file.

**What it costs** (measured on the development machine, four
threads, its SSD; `save_run`/`load_run` of a 20-variable vertex-centered set
at `N = 8`): on 3536 blocks — G5's mesh at rest is 3900–5600 — the file is
`290 MB`, a save `0.09 s` (`3 GB/s`), `0.11 s` with the flush to stable
storage, and a load `0.37 s`, once compiled; the first save and the first
load of a session compile for `3 s` and `5.5 s`. A G5 chunk is about
`160 s` of a node, so a checkpoint every chunk costs under a tenth of a
percent. The record of 53 rows is noise beside the state. Symmetry's BeeGFS
is not measured.

`evolve!`'s checkpoint keywords are TreeHydro's names:
`checkpoint_path_prefix`, `checkpoint_every_chunks`,
`checkpoint_interval_seconds`, `max_walltime_seconds`,
`num_checkpoints_keep`, `checkpoint_hdf5_filters`,
`checkpoint_sync_to_disk` and `restart_file`.

**TreeIOHDF5 `main` can be ahead of the TreeAMR it admits.** TreeIOHDF5 0.1.4 reads TreeAMR 0.2.5's `FieldSet.halfturn`,
  but its `[compat]` lists TreeAMR `=0.2.2, …, =0.2.5`. So the `[sources]`
  pin to `main` resolves it beside TreeAMR 0.2.4, and the first checkpoint
  throws in `set_layout` ("type FieldSet has no field `halfturn`"). That is
  an upstream compat bug: TreeIOHDF5 0.1.4 should require `=0.2.5`. A run without `checkpoint=` does not notice. This package requires TreeAMR 0.2.6, which TreeIOHDF5 `main` (0.1.5) admits.

## Upstream prerequisites

What this package needs from TreeAMR. None blocks G0–G3.

1. **Point interpolation from a field set** (G4, for the horizon
   finder). `find_leaf` plus block-local Lagrange interpolation, batched
   over a host array of points. Two properties matter: the window is
   *half* the ghost width on each side, which is what makes a query
   anywhere in a block readable without crossing into another block's
   array at `G = q/2 + 1`; and the batch is what makes it threaded, with
   one output slot per input point and no accumulation, so the answer does
   not depend on the thread count.
   **Done upstream:** TreeAMR's M11 — `locate_point`,
   `interpolate`/`interpolate!` with a `Lagrange(n)` basis, `derivs` as
   multi-indices, an `exclude::Region` that flags, one launch on the field
   set's backend — keeps both properties, with one binary search per point,
   device execution, periodic and reflecting faces; what stays in `src/horizon.jl` is the wrapper
   `gh_interpolate`/`gh_interpolate_grad` (the order `q + 2`, the unpacking
   into `SVector`s, the refusal) and the guard as a `Region` — see
   [Analysis quantities](#analysis-quantities), where the guard's exactness
   and the measured costs are.
2. **A device reduce-to-scalar**, which lets the per-chunk norms stay on
   the device. **Done upstream:** TreeAMR reduces on a device in two
   launches, and `mesh_mapreduce` is the one-number form; this package
   reduces through it.
3. **MPI (M7) and I/O (M9)**, which this package is written to take
   transparently: no host loop over blocks assumes all blocks are local,
   and every reduction that crosses blocks is a `mesh_mapreduce` or a
   `mesh_fold`, TreeAMR's fold over points, which carries many results
   through one collective. **Done upstream and in use:** M9's checkpoint
   half, M9a, is [Checkpoint and restart](#checkpoint-and-restart), and
   [Running distributed](#running-distributed) is what this package had
   left to do for M7.
4. **A launch-configuration knob on `map_blocks!`** (G6): today it
   launches with KernelAbstractions' default workgroup, and GHAccel
   measured the workgroup shape of a 3D stencil kernel as worth about
   10 % of peak. A `workgroupsize` keyword passed through to the launch
   is the whole request. Measured in [The right-hand side on an
   H200](#the-right-hand-side-on-an-h200):
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
   for a device.**
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
   - **The CPU has the same problem** ([The right-hand side
     on a CPU](#the-right-hand-side-on-a-cpu)). On Symmetry's
     EPYC nodes the fill costs about 10 ns a ghost value on one core — a seventh
     to a twentieth of copy speed — and a quarter of memory bandwidth at 64
     threads, where it is a third of `gh_rhs!` at `16³` and, once the kernel is
     vectorized across points, more than half. The profile names the same costs:
     the copy as a weighted tensor-product stencil, KernelAbstractions' 5-D
     iteration, integer index arithmetic, and at 64 threads a fifth of the time
     waiting between phases.
   - **On a refined mesh the prolongation is the cost** (measured on the
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
6. **Checkpoint rotation, possibly in TreeIOHDF5** **(open)**; nothing to do now. `rotate_checkpoints!`, with
   `checkpoint_files` and `latest_checkpoint` beside it (`src/checkpoint.jl`),
   is TreeHydro's code restated: the file names, the newest-by-iteration rule,
   keeping the file just written, and the removal of an
   index's part files, which knows TreeIOHDF5's part-file names. Every
   application of TreeAMR that checkpoints needs the same, and TreeIOHDF5
   writes the files and the parts, so it may belong there rather than in each
   application.
7. **The MPI launchers, consolidated in TreeAMR** **(open)**; nothing to do now. Three packages now carry the
   same launching code, copied from one another: the `mpiexec` job of a test
   suite (`test/mpi_jobs.jl` here, in TreeHydro and in TreeAMR — `MPI.mpiexec()`
   with its environment, started beside the suite and collected with a
   deadline), the choice of a rank's GPU by its rank on the node
   (`test/octant_runs.jl`, `bench/mpi_steps.jl`, TreeAMR's
   `test/mpi_device_workload.jl`), the `try … MPI.Abort` around a run, and the
   Symmetry job and environment scripts (`bench/symmetry_mpi_gpu*.sh` here,
   TreeAMR's `bench/symmetry_mpi_gpu.sh`, including the HPC-X `OpenMPI_jll`
   override). They belong in TreeAMR — its MPI extension for the rank and
   device helpers, a test utility for the job — so that the applications call
   them rather than copy them.

Two further items are *not* needed for the proof of concept and are
recorded with the extensions that would need them: an interior-reading
device boundary hook (radiative boundaries, excision) and excised leaves
(excision).

The package requires **TreeAMR 0.2.6**: every norm is a fold over points,
its M15 `mesh_fold`, which this package asked for.

**Inconsistent:** "File layout", in its paragraph on the dependencies,
says that TreeAMR's `[compat]` selects **0.1.1**.

## File layout

| file | contents |
|---|---|
| `notes/` | verbatim copies of the inherited documents, with provenance; see `notes/README.md` |
| `src/TreeGeneralizedHarmonic.jl` | module shell: `using`s, exports, includes |
| `src/precision.jl`, `src/device.jl` | copied from TreeWave, with TreeHydro's `hostcopy!` split so that the copying half is exercised host to host |
| `src/pointwise.jl` | GHSO2's pointwise algebra (ported from `notes/pointwise-ghso2.jl`), plus the expanded form's coefficient derivatives `metric_derivatives`, the assembled `gh_node_rhs_expanded`, and `gh_node_source`, and `metric_derivatives_along` — the same chain rule along **one** direction, returning `∂√γ` and `∂γ^{jk}` rather than the assembled `∂(α√γγ^{jk})`, which is what the constraint monitors need along *time* — and `_pairindex`, the packed slot of a symmetric index pair, so that the file has one packing convention used on two index pairs; `SVector{10}` state, `SMatrix{4,4}` tensors. `gh_node_source` is a **second copy** of the reduced source and the damping, written out of `gh_node_rhs` character for character rather than factored out of it: the port stays diffable against `notes/pointwise-ghso2.jl`, which is what makes it the validated reference, and `test/pointwise_identity_tests.jl` asserts the copy still matches it — to roundoff, because two spellings of one expression are not bit-identical (see "Measured results") |
| `src/stencils.jl` | rational finite-difference and Kreiss–Oliger weights at order `q`: `derivative_weights`, `dissipation_weights`, both `@generated` over `(T, Val(q), Val(m))` and returning `SVector`s of `T` for unit spacing; `lagrange_derivative_weights` and the two `rational_*` constructors behind them, exposed unexported so that the exactness claims can be asserted in `Rational` rather than through a tolerance; `dissipation_rank(Val(q)) = Val(q/2 + 1)`, one spelling of `2r = q + 2`; the host-side `apply_stencil` and `apply_mixed_stencil`, which are the reference contractions the tests measure with and the definitions the streaming kernel has to agree with |
| `src/evolution.jl` | the fused RHS kernel in streaming order, the linear-index stencil contractions it evaluates, `GHProblem` with the **six** `Val`s (the SIMD width the sixth) and the per-chunk geometry, `gh_rhs!` (whose `matter` adds the kernel's seventh, `HASMATTER`, GH-3) and its two halves `prepare!` and `rhs!` (GH-4), the speed kernel, `max_speed` with its checked method `max_speed(p, u, t)`, `gh_dt`, and `convergence_rate` — TreeWave's, in the file TreeWave keeps it in. The streaming body is split out of the kernel into `gh_rhs_at_point`, an `@inline` plain function, because `F` must not be evaluated in the frozen core and **KernelAbstractions refuses a `return` statement anywhere in a kernel body** — so the core branch cannot be an early exit and has to be an `if` around the whole computation; and the file has `gh_paste_kernel!` with `gh_step_limiter!` and `paste_interior!`, the `:pasted` variant's one write to the state |
| `src/lanes.jl` | the right-hand-side kernel's SIMD lanes on the CPU: `default_simd_width` and `check_simd_width`, `Lanes` (an array read and written `W` consecutive elements at a time, with the `Const` wrapper unwrapped; a dense array or a contiguous segment of a state vector, `LaneArray`, GH-4), `is_lane_leader` (every `W`-th item, and the overlapping last group of a row), and the `Vec` methods of `pointwise.jl`'s `_anynonzero` and `_select`; the kernel's group is `evolution.jl`'s `gh_rhs_lanes!` |
| `src/gauge.jl` | sampling prescribed sources into `Hsrc` and reading them back at a point (`gauge_at`, the kernel's half of the packing); `isharmonic` as a table over the background types and `isstatic` as an exact measurement, with the reason each is what it is; the two `γ0` profiles and the `ε_KO(r)` profile `HorizonDissipation`, with `dissipation_rate` the identity on a number |
| `src/boundaries.jl` | the time-dependent Dirichlet hook, and the declarations TreeAMR's symmetries ask of every field set: the parities of the reflecting faces (`state_parity`, `even_parity`) and the signed maps of the rotating seam (`state_rotation`, `identity_rotation`) |
| `src/bounds.jl` | the range projection: `StateBounds` and the proposed `default_bounds`/`default_gate`, `check_bounds_gate`, the pointwise `bounds_project` over an explicit-scalar ADM split and a Jacobi `sym_eigen3`, `gh_bounds_kernel!`, `BoundsAccounting`, `apply_bounds!` and `gh_stage_limiter!`; the validity monitor (`state_validity`, `validity_rows`); and `evolved_nonfinite`, the masked finiteness check. Included after `interior.jl` and before `initialdata.jl`, whose `GHCase` carries a `StateBounds` |
| `src/interior.jl` | the profiles `w(r)`, `ρ(r)`, the core rule, the radius checks, the masks. Also `HoleCenter` — `c(t) = c₀ + v t` as two vectors and a line, which is what "the center is a function of `t`, never a mutated field" means as code — the horizon's analytic coordinate radii and the hole's mass (`hole_mass`), and `layer_spacing`, the coarsest spacing among the blocks the sphere `r_1` passes through, which is the one number in the file that looks at a mesh (and looks at it only to *assert*). The `:pasted` limiter is in `evolution.jl` instead, beside the kernel it launches and the state layout it writes. **The tracked geometry's kernel side**: the real harmonics (`real_harmonic_index`, `real_from_complex`/`complex_from_real`, the recurrence `shape_series`, `shape_bounds`), the analytic horizon of the seed (`analytic_horizon_radius`), `FittedSpec` — what a case holds — and `FittedInterior` — the kernel argument — with `interior_point`, `fitted_geometry`, `core_position`, `ShapeMask`, `ShapeBand`, `geometry_spacing` and its `check_interior_radii`; and the protocol both geometries speak (`in_layer(int, t, x)`, `interior_point`, `is_outside`, `geometry_radii`, `layer_radii`, `layer_mask`, `shell_mask`) |
| `src/initialdata.jl` | backgrounds, `GHCase` and the case constructors (here rather than in `driver.jl`), the forest builders — uniform, with one root block refined for the frozen two-level hierarchy the interface study needs (`refined = true`), or `hole_forest`'s nested shells around a hole (**here rather than in `interior.jl`**, since a forest builder belongs with the other forest builder) — the `(h, Π)` callback with the core rule, the `SpacetimeMetrics` index conversion and nowhere else |
| `src/refinement.jl` | the Löhner indicator with its global floor, the mask, the level floor and ceiling, the four marks, the buffer; TreeWave's `refinement.jl` ported |
| `src/constraints.jl` | the gauge-constraint kernel (state and first derivatives) and the ADM one (every second derivative of `g_ab`, the `∂_t` blocks from the evolution equations, the four-dimensional Ricci tensor assembled rather than reduced), the masks they take — `AllPoints` and the `is_evolved` predicate the interior adds a method to — `masked_norms` and `constraint_norms`, and `adm_constraints_at_node`, the pointwise curvature assembly the tests check against `ddmetric` |
| `src/horizon.jl` | the interpolating ADM provider for `ApparentHorizonFinder`; location, shape, area, `M_irr`, `J`, `M_ch`. In the order the numbers are produced: `gh_interpolate`/`gh_interpolate_grad` over TreeAMR's `interpolate` ([Upstream prerequisites](#upstream-prerequisites), item 1), with the footprint guard that refuses a query reaching inside `r_1` as the `Region` `UnevolvedRegion`, `GHADMProvider` (batched, `Float64` out whatever the run computes in, with a one-entry cache keyed on the identity of the query array because `KorzynskiSpin.surface_geometry` asks for `γ` and `K` in two calls with the same points), `find_gh_horizon`, and `Horizon` — the cadence and resolution the case carries |
| `src/tracking.jl` | the tracked horizon, host-side (after `horizon.jl` and before `driver.jl`): the conversions from the finder's `hlm` (`real_shape`) and of the analytic horizon (`analytic_shape`) into real coefficients, `HorizonTrack` with `seed_track`, `update_track`, `track_center` and `TrackLostError`, `fitted_interior` — the kernel argument from a track and a mesh — `surface_shift` (the gauge source's re-sample rule), and `axis_dispersion`/`margin_efolds`, the leakage e-folds, moved in from `test/dispersion.jl` |
| `src/fit.jl` | the fitted target (after `tracking.jl` and before `driver.jl`): the fit's variables (`fit_variables`, `state_from_fit`), the real solid harmonics (`_solid_harmonic_fold`, `real_solid_harmonics`, `fit_directions`), the two samplers (`state_sampler`, `analytic_sampler`), the least squares (`solve_fit`, `fit_row_weights`), the validity sweep (`fit_sweep`), `FitParams` and `InteriorFit` with `build_fit`, `fit_residual` and `fit_valid`, the kernel-callable evaluator `fit_variables_at`/`fit_state`, and the kernel half: `derive_target_bounds`, the 40-variable cache (`target_cache`, `fit_target_kernel!`, `fill_target!`) and the initial data's `fitted_state_kernel!`. The variant's branch is in `evolution.jl` (`GHProblem`'s `target`/`fits`/`t_target`, `refill_target`), its residual in `constraints.jl`'s error kernel, its flow in `driver.jl` (`refit!`, the refill and the pieces of a moving chunk) |
| `src/checkpoint.jl` | checkpoint and restart (after `fit.jl` and before `driver.jl`): the file names and their rotation and `latest_checkpoint` (TreeHydro's), `plain_reals` for exact reals, `to_plain`/`from_plain` for the run state's own structs and `fit_from_plain` for an `InteriorFit`, `run_recipe` and `run_criterion`, `check_recipe`, `save_run`/`load_run` over TreeIOHDF5's `save_checkpoint`/`load_checkpoint`, and `check_checkpoint_keywords`. `evolve!` writes and reads through it; `test/checkpoint_tests.jl` is its file |
| `src/driver.jl` | `evolve!`, the analysis record per chunk, `observer`, `check_cfl`, `horizon_shell`, `forest_levels`, `default_relaxation_rate` — the layer's default `4/M`, the one place the number is written — and `discrete_gradient_momentum!` — GHSO2's `Π` post-pass, which lives here because it runs once on the initial data and is the driver's option, not the initial data's. `GHCase` is in `initialdata.jl` |
| `src/io.jl` | the analysis time series, slice output |
| `src/benchmark.jl` | per-phase timings in TreeWave's format |
| `test/` | one `*_tests.jl` per section above, `prerequisite_tests.jl` (what the two pinned dependencies must still provide), `type_tests.jl`, `threading_tests.jl`, `device_tests.jl`, the standalone `thread_workload.jl`, and `evolution_cases.jl` — a *helper*, the runs the convergence and noise studies are made of, which lives in `test/` because what it wraps is the integrator loop (after TreeAMR's `test/wave.jl`). `pointwise.jl`'s tests are **two** files over a shared `pointwise_backgrounds.jl` — `pointwise_tests.jl` for the algebra as a function of the state, `pointwise_identity_tests.jl` for the two identities that need derivatives of it — because between them they compile the metric library's nested dual passes for six backgrounds at two precisions. The interface-order table has a file of its own, `interface_tests.jl`, rather than a testset in `convergence_tests.jl`: it is fifteen evolutions on a mesh where the ghost fill costs four times what it costs on a uniform one, and separating it keeps the cheap order study cheap. The hole has `interior_tests.jl` (the profiles, the core rule, the masks, the radius assertions, and one right-hand-side evaluation on a mesh), `driver_tests.jl` (the runs), the hole fixture in `evolution_cases.jl`, and the **standalone** `test/hole_runs.jl` — the `t = 50 M` runs, `q = 4`, and the two harmonic charts, which are minutes rather than seconds and are run by hand with their numbers recorded here (following `PLAN.md`'s instruction to put what cannot fit a test file in a script under `test/`). The range projection has `bounds_tests.jl` (the projection on synthetic states and on every background, its `Float32` row, planted failures on the fixture's mesh, and the bitwise control) and `hole_runs.jl`'s `bounds` section. The layer's calibration has the E3 target `CurvatureTarget` in `evolution_cases.jl`, its claims in `interior_tests.jl` and `driver_tests.jl`, and `hole_runs.jl`'s `calibration` section. The tracked geometry has `tracking_tests.jl` — the real harmonics against `AbstractSphericalHarmonics`, the seed against the charts' quartic, the depth of an oblate spheroid, the footprint guard on a non-spherical surface, the e-folds against `dispersion.jl`, the bit-identity of a fitted sphere with step 5's layer, one find of the fixture, and the tracked runs — and `evolution_cases.jl`'s shell norms are the interior's own `shell_mask`. The fitted target has `fit_tests.jl` — the fit's harmonics against `shape_series` and `ash_evaluate`, the whole ansatz recovered by `solve_fit`, the static hole's fit against its truncation and the interpolation order, the validity sweep on three holes and the `g_ab` mean control, the evaluator against the model at `Float64` and `Float32`, and the fit of the tracked run's final state; and the `:fitted` right-hand side against `:damped`'s bit for bit, the fixture's `:fitted` run to `0.15 M` and one chunk of it at `Float32` — and `fitted_fixture` is in `evolution_cases.jl` beside `tracked_fixture_run`, the one tracked run `tracking_tests.jl` and `fit_tests.jl` share; `hole_runs.jl` has a `fitted` section (the fixture's initial-data study to `1 M`, the boosted seed, harmonic `a = 9/10`); GH-3's matter term has `matter_tests.jl` — the acceptance on a static star with its `EinsteinTensor` matter, the lanes with matter, the anti-Cowling run, and a test adapter (`SampledMatter`) shaped as TreeGRRMHD's will be; the star and its fixed matter are `plummer_star.jl`'s, which the standalone damping study `star_runs.jl` (one run) and `star_study.jl` (its tables) share; GH-4's verbs have `verbs_tests.jl` — a script's integrator over two sets, and the limiters, the speed and the initial state on a segment |
| `bin/` | `gh.jl` (the CLI, after GHSO2's `gh3d.jl`), viewers, `benchmark.jl`, `backend.jl`, own `Project.toml` |

**Inconsistent:** the `test/` row says `prerequisite_tests.jl` checks what
"the two pinned dependencies" must still provide; the dependencies
paragraph below, and `CLAUDE.md`'s "Things that will bite", name more
`[sources]` pins than two.

Dependencies: `TreeAMR` (registered) and `SpacetimeMetrics` (pinned to
GitHub `main` by `[sources]`),
`KernelAbstractions`, `StaticArrays`
(kernel-safe, and what `SpacetimeMetrics` speaks),
`IMEXRungeKutta` (unregistered, pinned to GitHub `main` by `[sources]`; neither
`OrdinaryDiffEqLowOrderRK` nor `SciMLBase` is a dependency of the package or
of its tests), `LinearAlgebra` (`det`, `dot`
and `tr` on `StaticArrays`, which the pointwise algebra uses; a standard
library), `TreeIOHDF5` (TreeAMR's companion package for checkpoints, registered,
`[compat]` `0.1.0`),
`ApparentHorizonFinder` and `KorzynskiSpin` (from G4,
both pinned to GitHub `main` by `[sources]` like `SpacetimeMetrics` and both
listed in `test/Project.toml` as well, because `prerequisite_tests.jl`
holds them to the same standard: a moving `main` that dropped a name must
fail at the top of the suite). The `[sources]` entries would by themselves put the Julia floor at 1.11, and
three of them are necessary rather than chosen — `KorzynskiSpin` and `TreeIOHDF5` are not in
General at all, and `ApparentHorizonFinder` `2.1` is not released there
(General has `2.0.0`). Vendoring a package or
adding a local-path source would hide that instead of stating it.

**Inconsistent:** the paragraph above lists `TreeIOHDF5` as registered
(`[compat]` `0.1.0`, as `HISTORY.md`'s [Releases and dependencies](HISTORY.md#releases-and-dependencies) does) and then as one
of the `[sources]` entries that are necessary because it is "not in General
at all".

**The Julia floor is 1.13** (`julia = "1.13"`), and CI runs 1.13 only,
because supporting earlier versions slows
development and is not useful at the moment.
**`AbstractSphericalHarmonics` is a direct dependency** (`1.2`,
registered):
the tracked geometry converts the finder's coefficients with its
`ash_resample` and samples the seed on its `EquiangularGrid`, and a package
this one calls is one it names, not one it reaches through another's
namespace. The tests list it too, for `sYlm` and `ash_evaluate`. Tests add
`MultiFloats` and `ForwardDiff` — the latter because the checks on the
expanded form differentiate the analytic solution one layer above the one
`SpacetimeMetrics` takes internally. `bin/` adds
`CairoMakie` and `SixelTerm` in its own environment. The compat bounds
follow TreeWave's. A `[compat]` entry on a
`[sources]` dependency is a floor on what `main` may become, not a
selection, which is what `prerequisite_tests.jl` exists to notice.

More of what the files hold:

- `precision.jl` holds the `Base`
  bridges for software floating-point types, and `device.jl`
  `to_backend`, `hostcopy` and `hostcopy!`.
- `interior.jl` has the `InteriorMask` and `ShellMask` every norm takes,
  and `evolution.jl`'s fifth `Val` carries the interior *variant*.
- `refinement.jl` has the
`Refinement` parameters a case carries, `lohner` with the global
amplitude, the masked `field_scales`, the `τ` kernel into `diag`'s
fourteenth slot, `tau_max`, the four marks with the box keyed on
`coarsen_tol`, the derived `horizon_floor_level`, the `LevelBounds` the
floor and the ceiling are applied through, the travelling margin,
`refinement_centroid`, and `indicator_flags`/`gh_indicator!`, the two
entry points the cycle and the driver call.
- `bounds.jl` has `StateBounds` (the
ranges of `α`, `γ`'s spectrum, `|β|` and `Π`'s scale, and the gate radius;
`default_bounds` and `default_gate` are the named proposals, and a
`GHCase` carries one as `bounds`, default `nothing`), the pointwise
`bounds_project` in ADM variables, `gh_bounds_kernel!` behind
`gh_stage_limiter!` — the state's third writer, the integrator's stage
limiter beside the paste — `BoundsAccounting`, the validity monitor
(`validity_rows`) and `evolved_nonfinite`, which is what `max_speed(p, u, t)`
and the record's `finite` read.
- `test/evolution_cases.jl` has the E3 target
  (`CurvatureTarget`) and a `gh_outside_shell_norms(p, u, t)` method, and
  `hole_runs.jl` a `calibration` section; `hole_runs.jl` has the `generic` section,
  the matrix.

**TreeAMR is General's release under `[compat]` `0.2.6`**; the other pinned packages are what `Project.toml`'s `[sources]` entries resolve. `SpacetimeMetrics` is
  registered at the bound this package asks for, so its pin is a choice
  rather than a necessity.

**The fixtures**: `hole_fixture` in `test/evolution_cases.jl` (Kerr-Schild
  `a = 0`, `q = 2`, `N = 8`, `h = 5/64`, `r_0 = 2/5`, `r_1 = 23/20`,
  `m = 8`, box `5/2`, 120 leaves); `adaptive_hole_fixture` (box `5`,
  `m = 4`); `gh_outside_shell_norms` reads the `G` points outside `r_1`
  through a `ShellMask`. `hole_runs.jl` takes section names as arguments;
  add a section, do not lengthen an existing one.

## Milestones

The milestones G0–G8 and their acceptance tests are in [`PLAN.md`](PLAN.md#milestones); what the finished ones found is in [`HISTORY.md`](HISTORY.md#the-implementation-plan) and below.

## Measured results

This section takes the numbers as the milestones produce them; each also
sits beside the prediction it confirms or corrects, in the section it
belongs to. **The black hole's interior is the exception**: step 5's layer, steps 8a–8′ and the single holes on the
octant are in [`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md), and what
they recommend for a single hole is [the table
below](#single-black-holes-recommended-settings).

**G1a, the pointwise algebra.** Essentially all of the pointwise tests' cost is **compilation**: the six
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
- **What `‖h‖ ~ 1e−13` claims.** `‖h‖ ~ 1e−13` is a claim about the *offsets*
  `g^{ab} − η^{ab}`, `det g + 1`, `det γ − 1` — `α`, `β^i` and `√γ` are
  built from them and are accurate to a relative `eps` **as values**,
  which is not the same statement and is the one a test can make about
  what `metric_quantities` returns.
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
  says what the bit-identity the threading test asserts does and
  does not mean: the *same* compiled code, at the same call site, on a
  different thread count — and nothing more.

**G1b, the stencils.** The weights reproduce the textbook central-difference tables entry for
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

One thing that is not about the stencils: the *tests* need
`Rational{BigInt}` and not `Rational{Int}`. A monomial of degree `q + 2` at
an off-grid rational point overflows a 64-bit denominator at `q = 6`
already. `Rational` arithmetic is checked, so this arrived as an
`OverflowError` in the test rather than as a wrong answer — TreeAMR's
reason for `BigInt` in its own weight construction, seen from the other
side.

**G2, the right-hand side on a uniform mesh.**

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
not of `eps/h²` (G1b's warning about the window). Shifted Minkowski's
rate is **3.6** at the profile width `w = 1` and **3.93** at `w = 2`: at
`h = L/16` the sech² profile is not resolved, and the shortfall is the
case's parameters and not the scheme — which is worth knowing before a
hole, whose features are sharper still, is put on a mesh this coarse.

**Stationarity.** Minkowski's `du` is **exactly** zero — every component,
at `q = 2, 4, 6`, with and without dissipation, at any `t` — and a run of
it through RK4 returns the initial state with an error of exactly zero.
Shifted Minkowski is stationary only to truncation: its `Π` is built
from the *analytic* spatial gradients ([Initial data and
backgrounds](#initial-data-and-backgrounds)).

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
to be: two call sites of one body are contracted differently (G1a's
finding, now seen from the mesh). Two evaluations at the same `(u, t)`
*are* bit-identical, and `u` is untouched by either.

**`out_identical` is a roundoff bound, not bit-identity** ([history](HISTORY.md#what-the-suite-costs-and-where-measured-2026-09-19-on-symmetry)):
the `:damped` kernel outside `r_1` and `:none` are separate `Val`-specialised
kernels, each fusing its multiply-adds in its own inlining context — on x86-64 only: Apple
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
and `96` eps.

**What an evaluation costs** is the table under [One right-hand-side
evaluation](#one-right-hand-side-evaluation), together with the linear
index that bought a quarter of it. Two further numbers from the same
measurement: the pointwise algebra — `metric_quantities`,
`metric_derivatives`, `gh_node_source` — is **505 ns** per point of the
`q = 4` kernel's 1409, so the stencils are the larger half at every order
this package uses, and the split into a stencil kernel and an algebra
kernel that G6 will try is a split of roughly 3:1 rather than 1:1.

**G3, coarse-fine faces, the constraints and the threads.**

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

**G4a, the interior and the driver** — the suite, the layer's order,
the three variants to `50 M`, the other charts, the gauge drift, `Float32` and
the record — are in
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#step-5-the-static-holes-interior-and-the-driver-g4a).

**G4b, the refinement indicator.** `refinement_tests.jl` stays inside `PLAN.md`'s 30 s rule of thumb, and what
buys that is the fixture's `maxlevel_cap = 1`: at the calibrated cap the
same case is 848 blocks and belongs to `test/hole_runs.jl`, which is where
it is.

**The calibration
is not portable between boxes the way the thresholds are**: the
thresholds are a property of the solution and the spacing, but *which*
hierarchy they produce is a property of the box and the root brick as
well, so the reference configuration is recorded with them. G4's **"In a Dirichlet box of half-width
`≥ 20 M`" is a statement about a production run and not about a test**:
the two radius requirements need `r_h,min ≥ (m + 2G + 2)·h`, so a hole
costs about `(m + 2G + 2)³` finest-level points *per `r_h,min` cubed*
whatever the box is, and a box of `20 M` with a uniform mesh at that
spacing is `10⁸` points. The boundary is exact, so a small box costs
accuracy and not validity; the suite runs `5/2 M` and records it.

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
`5/64`, at the grid rate `1/dt`:

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
move — at the grid rate; `1.34e−3` and `1.88e−3` at the default `4/M`, where the interior residual of the run that did
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

**G4c, the horizon.** `prerequisite_tests.jl` holds `ApparentHorizonFinder` and
`KorzynskiSpin` to the same standard as the first two pinned dependencies — including a find of Kerr's horizon on
*analytic* Cauchy data at `a = 9/10`, which recovers the area to `1e−8`
and `J = M a` to `1e−6` with the axis along `ẑ`, and is the baseline
everything below is measured against.

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
uses, and it is the mesh `CODE.md` predicted.** At
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
the `:damped` layer at the grid rate `1/dt`; the
`4/M` runs end at `M_irr = 0.9972` at `50 M`) to `t = 10 M`, the
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

**The port to TreeAMR's interpolation.** The same
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
**The provider interpolates 16 variables, not 20.** `adm_vars_from_state` reads all of `h`, the gradients of `h_ti`
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

**Step 8a, what crosses the horizon from inside it** — in
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#step-8a-what-crosses-the-horizon-from-inside-it).

**Steps 8b–8′, the generic interior and the moving hole** — in
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#steps-8b8-the-generic-interior-and-the-moving-hole): the range projection (8b), the layer against an inexact target
(8c), the default rate (8c′), the tracked geometry (8d), the fitted target
(8e-i, 8e-ii), the measurement matrix (8f), the moving hole (8) and its
trailing side (8′).

### What the suite costs, and where

Measured on the suite of 3444 assertions; the rest of that profile is in [`HISTORY.md`](HISTORY.md#what-the-suite-costs-and-where-measured-2026-09-19-on-symmetry).

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

**Four threads on four cores is not oversubscribed in any way that costs**
(the obvious hypothesis was wrong). Julia 1.13
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

#### The test files

The suite's files include `precision_tests.jl`, `prerequisite_tests.jl`
(the pinned TreeAMR still exports the names the design calls, a
`SpacetimeMetrics` background compiles and runs as a kernel argument on
`CPU()` filling a field set bit-for-bit as a host loop does, and the
*unexported* metric wrappers `gauge.jl` dispatches on still exist), the
pointwise pair — `pointwise_tests.jl` and `pointwise_identity_tests.jl`
over a shared `pointwise_backgrounds.jl`, which is where the six
backgrounds of this document's table and the analytic data are built —
`stencils_tests.jl`, which evaluates no background at all and costs
seconds, and `gauge_tests.jl`, `initialdata_tests.jl`,
`evolution_tests.jl` (the kernel against `gh_node_rhs_expanded`, and the
properties nothing else can state), `convergence_tests.jl` and
`noise_tests.jl`, over the helper `evolution_cases.jl`; and
`constraints_tests.jl` (both monitors, against `ddmetric` on analytic
data and against the mesh), `interface_tests.jl` (the interface-order
table on the two-level mesh), `type_tests.jl` (`Float32` end to end,
`Float32x2` for the algebra) and `threading_tests.jl` over the standalone
`thread_workload.jl`; and `interior_tests.jl` (the
profiles, the core rule, the masks, the radius assertions firing, and one
right-hand-side evaluation with the layer on a mesh) and
`driver_tests.jl` (the runs: the record, the order on the frozen
hierarchy, the three variants, the drift, the `Π` post-pass), over the
hole fixture in `evolution_cases.jl`; and `refinement_tests.jl`
(the Löhner algebra and its global floor, the four marks on a `τ` field
written by hand, the mask, the level floor and the ceiling, the
initial-data cycle, and two short adaptive runs — one whose mesh does not
move and one whose does), over that file's second hole fixture and the
three new helpers in `evolution_cases.jl`; and
`horizon_tests.jl` (the interpolator's exactness and its rate, the
footprint guard, Kerr's horizon from a displaced guess on both the step-5
fixture and the mesh the indicator chose, and the record's horizon rows at
the case's cadence); and `fit_tests.jl` (the fit's harmonics, its
least squares, the static and spinning holes' fits and their validity, the
`g_ab` control, the evaluator at two precisions, and the fit of the one
tracked run's state, which `evolution_cases.jl`'s `tracked_fixture_run`
shares with `tracking_tests.jl`). **`test/hole_runs.jl` is a standalone script, not
part of the suite**: the `t = 50 M` runs, `q = 4`, the two harmonic
charts, the indicator's calibration and the horizon
section (Kerr's numbers at `a = 9/10` and in the harmonic chart, on meshes
of a thousand blocks) are minutes rather than seconds, and its numbers are
recorded here with the command that produced them. There is a second
script, **`test/dispersion.jl`** — the frozen-coefficient dispersion
analysis of the package's own weights inside a horizon, the penetration
length `ℓ` of grid-scale content, and a one-dimensional model run that
predicts what the 3D runs measure — and `hole_runs.jl` has a `leakage`
section, which is *not* in its default list (it is a batch job).
The `calibration` section is not in it either: about 150 runs of
`5 M` and `50 M`, a dozen node-hours on Symmetry.
`simwatch_tests.jl` is the SimWatch writer's file (with its SVG
thumbnails). `test/octant_runs.jl`'s output is analysed by
`octant_rates.jl`, `octant_study.jl` and `octant_diff.jl`, and
`test/aligned_study.jl campaign=transverse` reads the transverse
campaign's data and sets each row against the aligned row of the same
`a`, `v` and `h`.

#### What pays for the suite's time

Most of the suite's time is **compilation**, and the things that pay for
it are, in order: `SpacetimeMetrics`' nested forward-mode passes for six
backgrounds at two precisions; a
right-hand-side kernel per `(q, has gauge source, has dissipation,
interior variant, T)`; the ADM constraint kernel, whose first
specialisation is **18 s** and each further one about 4 s; and the four
files whose cost is *arithmetic* rather than compilation —
`driver_tests.jl` (the suite's black hole, 1m59 / 44.8 s),
`interface_tests.jl`, where the ghost fill at `p = 6` is 79 % of every
evaluation, `refinement_tests.jl` (22.1 / 12.8 s) and `horizon_tests.jl`
(28.7 / 12.8 s, of which the two evolutions are most: the interpolator's
own claims are under two seconds and a *find* is a tenth of one).

`checkpoint_tests.jl` is over `PLAN.md`'s 30 s rule because a chain is
the claim; the cheapest chains that still reach each carried piece of
state are the ones there.

### The suite's tiers, and where each runs

Evolving a black hole is what this package is for, and it is work for a
workstation or Symmetry, not for a free GitHub runner.
On GitHub's four-core Linux runners, with `julia-runtest`'s
`--check-bounds=yes`, the suite had grown to **72 minutes serial** and
**51 at four threads**, the threaded cell another 17.5 in front of that for
the MPI workload. Two factors separate that from the development machine's
14–24 minutes. A runner is about **1.7×** slower, and bounds checks cost
another **1.65×**: `driver_tests.jl` and `checkpoint_tests.jl` took 8m38
with them and 5m12 without, at one thread here. Of the 72 minutes, about
**42 were 26 testsets that evolve a hole**: the driver's convergence and
variants, the `:fitted` variant, the tracked runs, the checkpoint chains,
the moving `:damped` core and layer, the moving fit's rate, the regrid
that moves the mesh, the octant hole, the `Float32` hole and the three
mesh evaluations of `(INTERIOR)` with their own kernel specialisations.

So the suite has **tiers** (`test/tiers.jl`), chosen by
`TREEGH_TEST_TIER`:

| Tier | What it runs | Where |
|------|--------------|-------|
| `full` (default) | everything | the workstation before a pull request; Symmetry; GitHub weekly |
| `quick` | all but the 26 `@full_tier` testsets | GitHub, every pull request and push |
| `mpi` | `mpi_tests.jl` alone | GitHub, a job of its own |

The quick tier keeps what only CI can see: x86-64 and its contractions,
bounds checks, a clean environment, the kernels on every branch (the SIMD
lanes on the hole fixture and the octant hole included), the thread
invariance, and one short hole run through `evolve!` (the record's). A
left-out testset is a testset there holding one skipped test, so the summary
counts it as "Broken" and `runtests.jl` lists the names. Measured on the
development machine at four threads: **5035 assertions in 10–11 minutes**,
against the full suite's 16–24. On GitHub the serial
cell's suite went from **72 to 32 minutes** and the threaded cell's from 51
to 24, and the MPI job took 16 on its own runner, so a pull request waits
**34 minutes instead of 75**. What the quick tier still pays most for is
compilation that the left-out testsets used to pay first: the octant hole
against the box (`rotation_tests.jl`) and the moving hole on the quadrant
against the box (`boosted_tests.jl`), 105 and 91 s serial, which stay
because a mirror or a seam that is wrong only on x86-64 is what CI is for.

The full tier's result is put on the commit, so a pull request shows it:
`ci/full-suite.sh --post` tests a clean `git archive` of a commit and sets
its status `full-suite/<host>`, and `ci/full-suite-symmetry.sh` does the
same through an `amdq` job, posting its result from the workstation
(`CLAUDE.md`, "Commands"). A self-hosted GitHub runner would do this
unasked, but on a public repository it runs any fork's pull request on the
machine, and a runner daemon on Symmetry's login node is the
administrators' to allow.

#### CI's cells and coverage

Code coverage
runs on **the macOS cell only**: instrumented runs compute
*wrong numbers* on GitHub's Linux runners (207 failures on the threaded
Linux cell at Julia 1.13.1, which a removed gate had left instrumented),
while the instrumented macOS cell and the instrumented suite on Symmetry
pass. Probes of 146 GitHub runners found the trigger: `--code-coverage`
together with `--check-bounds=yes` (what `julia-runtest` passes; either
alone is fine) on AMD Zen 4/Zen 5 **with AVX-512 exposed** (EPYC 9V45
always, EPYC 9V74 only on the VMs that expose it), on Julia **1.13.0 and
1.13.1**, deterministically — a miscompilation, not a bad machine. Zen 3,
Intel with AVX-512 and nightly (1.14.0-DEV.3386) pass. **CI runs
in two tiers**, the same in every Tree* repository: a
pull request runs the two Linux cells only, uninstrumented; a push to
`main`, a weekly schedule and a manual dispatch add the macOS cell and
its coverage; Markdown-only changes and tags run nothing. The pull-request
cells had taken 65–90 min each, against 17–21 min for the suite on the
laptop, and 61 of the last 80 commits on `main` touched only Markdown.
**GitHub runs the quick tier**; `full` is the default, so `Pkg.test()` is
unchanged, and the weekly schedule and a dispatch with `tier = full` run it
on GitHub too. 

On Symmetry's nodes the full suite no longer fits `amddebugq`'s or
`h200debugq`'s hour — three jobs stopped at `type_tests.jl` at 60 min, with two
compiling against each other's pidfiles — so it is submitted to `amdq` or
`h200q` (`CLAUDE.md`, "Commands"; [history](HISTORY.md#the-test-suite)).

### Robust stability on the octant

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

### Single black holes: recommended settings

Rules of thumb for a single Kerr-Schild hole, `M = 1`, spin `a ≤ 9/10` along
`z`, at rest on the octant, drawn from the runs in
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07)
— and the `a = 0` hole moving at `v = 1/5` on the quadrant
(the rows marked *moving*, from [the boosted Kerr-Schild hole at binary
speed](SINGULARITY_HANDLING.md#the-boosted-kerr-schild-hole-at-binary-speed-2026-10-20)),
and the `a = 1/2` hole moving along its spin at `v = 1/5` on
the rotating quadrant ([a spinning hole along its
axis](SINGULARITY_HANDLING.md#a-spinning-hole-along-its-axis-2026-10-21)),
and `a = 0`, `1/2` and `9/10` moving along their spin at
`v = 1/5` and `1/2` ([aligned spin and
velocity](SINGULARITY_HANDLING.md#aligned-spin-and-velocity-which-holes-the-algorithm-holds-2026-10-22),
the rows marked *along `z`*, `20 M` each, `chunk = M`), and
`a = 1/2` and `9/10` moving along `x` across their spin at `v = 1/5` and
`1/2` on the bitant ([spin across the
motion](SINGULARITY_HANDLING.md#spin-across-the-motion-the-bitant-2026-10-24),
the rows marked *across*, `20 M` each, `chunk = M`; at `a = 0` they are the
*along `z`* rows):
`a = 0`, `3/5` and `9/10` are measured, the spins between follow from the
rules below the table, and none of it is a guarantee. The rows marked
*harmonic* (from step H1, [a harmonic hole at
rest](SINGULARITY_HANDLING.md#a-harmonic-hole-at-rest-step-h1-2026-11-11))
are Kerr at `a = 0` in harmonic coordinates at rest on the mirrored octant,
with no gauge source; its horizon is at `M`, and its radii are the areal
radii less `M`. The rows marked *harmonic, moving* are the same hole boosted
along `x` on the quadrant (step H2, [a moving harmonic
hole](SINGULARITY_HANDLING.md#a-moving-harmonic-hole-step-h2-2026-11-12)),
radii `32, 16, 8, 4` about the hole, on the capsule or on a mesh that follows
the hole (`mesh = follow`, `floor_margin = 3`). The *harmonic* rows at
`a = 1/2` and `7/10` are step H3's ([a spinning harmonic
hole](SINGULARITY_HANDLING.md#a-spinning-harmonic-hole-step-h3-2026-11-12-to-2026-11-13)):
the hole spinning about `z` at rest on the rotating octant, moving along its
spin on the rotating quadrant (*harmonic, along `z`*) and across it on the
bitant (*harmonic, across*), radii `32, 16, 8, 4, 2` about the hole and the
moving rows on a `4 M` capsule; the horizon's pole is at `√(M² − a²)`, its
equator at `M` and the ring at `a`. Common to every row:
`q = 4`, `cfl = 1/2`, `ε_KO = 1/2`, `ρ_max = 4/M`, the Gaussian `γ0`
(`1/M → 1/(10M)`), the algebraic source, the rotating octant (the mirrored
one at `a = 0`), a finest level whose cube contains the whole horizon, and
`24 M` — the shells about the hole saturate by about `8 M`. The moving rows
differ in the domain (`L = 64`: the quadrant for `a = 0` moving along `x`, the
rotating quadrant for `a = 1/2` moving along `z`, its spin),
the mesh (a frozen capsule along the path) and the length (`40 M`, `8 M` of
travel). `h` is the
spacing at the hole; a convergence pair keeps the layer's radii fixed in `M`,
not in cells. The horizon's coordinate radius is `r₊ = 1 + √(1 − a²)` at the
poles and `√(2 r₊)` on the equator, and the ring has radius `a`.

Earlier versions of this table and its rules: [history](HISTORY.md#single-black-holes-recommended-settings).

| `a` | variant | coarsest `h` (convergence pair) | singularity settings | basis |
|---|---|---|---|---|
| 0 | `:damped` | `1/16` (`1/16`, `1/24`) | `r_0 = 3/4`, `r_1 = 3/2` | measured: order 4 from `1/16` |
| 0 | `:fitted` | `1/24` (`1/24`, `1/32`) | `m = 16`, `n_L = 20`, `fit_cont = 2`, `lmax_fit = 8` | measured: `ℋ` outside `1.6×` `:damped`'s, the error `1.1×` |
| 0, *moving* (`v = 1/5` along `x`) | `:damped` (with the target's rate) | `1/16` (`1/24`, `1/32`) | `r_0 = 5/8`, `r_1 = 23/16` (`8.4` cells inside the contracted horizon at `1/16`) | measured to `40 M` at `4/M`: from `1/24` on `ℋ` just outside the horizon `1.00–1.03×` the resting hole's and the error there `1.26–1.36×` to `16 M`; order 4 in every shell from `1/24` to `1/32`; at `1/16` `ℋ` there `1.4–1.6×` (`8/M`: `1.03–1.09×` its own resting hole, which `8/M` lowers too) |
| 0, *moving* (`v = 1/5` along `x`) | `:fitted` | `1/24` | `m = 16`, `n_L = 20`, `fit_cont = 2`, `lmax_fit = 8` | measured to `40 M`: stable, the error just outside the horizon `:damped`'s at `4/M`, `ℋ` there `13–15×` `:damped`'s at `8/M`; `ℋ` there does not converge to `1/32` at `m = 16`, even at rest; `trail_ramp = 9/10` raised the error there `1.5–2.4×` and lowered it over the evolved region to `0.72–0.91×` |
| 0, *along `z`* (`v = 1/5`, `1/2`) | `:damped` (the core at `max(ρ_max, 12 |v|/r_0)`) | `1/16`; `1/24` at `v = 1/2` (`1/16`, `1/24`, `1/32`) | `r_0 = 1/2`, `r_1 = 19/16` (`8.7` cells inside the contracted pole `1.732` at `v = 1/2` and `1/16`; the ramp `11` cells) | measured to `20 M`: order 4 in the error outside the horizon; moving over resting in `[3, 5)` `1.38×` at `1/5` and `4.9×` at `1/2`, the same at every `h` (the moving solution and its position); `ℋ` `0.8–2.0×` at `1/2`, `0.65–1.01×` at `1/5` |
| 0, *harmonic* | `:damped` | `1/24` (`1/24`, `1/32`, `1/48`, `1/64`) | `r_0 = 2/5`, `r_1 = 3/4` (`6` cells inside the horizon at `1/24`, the ramp `8.4`) | measured to `24 M` and at `1/24`, `1/32` to `128 M`: order 4 outside the horizon from `1/24` (`ℋ` just outside it `9.7`, `5.9`, `4.1`, a layer term); stationary near the hole; the mass drift truncation error; the Kerr-Schild row's accuracy at `1/24` needs `1/56–1/65` (`ℋ`) and `1/65–1/78` (the relative error) |
| 0, *harmonic* | `:fitted` | `1/40` (`1/40`, `1/48`, `1/64`) | `m = 16`, `n_L = 20`, `fit_cont = 2`, `lmax_fit = 8` (the core surface at `0.1 M` at `1/40`; the projection fires in the first `M` only) | measured to `24 M`, and at `1/40` to `128 M` (stationary near the hole, the mass drift `:damped`'s): `:damped`'s outside the horizon from `1/48` (`ℋ` just outside `1.03×`, every other shell equal), order 4 there with the margin fixed in cells and in `M`; the band inside does not converge |
| 0, *harmonic, moving* (`v = 1/5`, `1/2` along `x`) | `:damped` (with the target's rate) | `1/24` at `v = 1/5` (`1/24`, `1/32`); `1/32` at `v = 1/2` (`1/32`, `1/40`) | `r_0 = 2/5`, `r_1 = 3/4` at `1/5`; `r_1 = 7/10` at `1/2` (the contracted horizon `0.866`; `5.3` cells inside at `1/32`) | measured to `40 M` (`1/5`) and `20 M` (`1/2`): no lag, order 4 or better outside the horizon; at `1/5` `ℋ` from `5/4` out `1.0–1.5×` the resting hole's and the error `1.2–1.8×`, just outside the horizon `ℋ` `1.6×` (`1/24`) and `3.0×` (`1/32`, the resting row's layer term converging at `9.7`); at `1/2` the trailing side `10–46×` in `ℋ` and `3–4×` in the error, converging at `4–6` |
| 0, *harmonic, moving* (`v = 1/5` along `x`) | `:fitted` | `1/40` (`1/40`, `1/48`; `1/48` needs two H200s) | `m = 16`, `n_L = 20`, `fit_cont = 2`, `lmax_fit = 8`; `trail_ramp = 9/10` optional | measured to `40 M`: from `5/4` out `ℋ` `1.0–1.1×` the resting hole's and the error `1.1–1.6×`; just outside the horizon `ℋ` `14×` (`1/40`) and `18×` (`1/48`) on the trailing side, not converging against rest; a transient at `4 M` (`168×`, `22×` with `trail_ramp`); order `3–4` at `40 M`; **unstable at `v = 1/2`** at `1/48`, with and without `trail_ramp` |
| 0.3 | `:damped` | `1/16` (`1/16`, `1/24`) | `r_0 = 3/4`, `r_1 = 3/2` | the rules |
| 0.5, *moving* (`v = 1/5` along `z`, its spin) | `:damped` (with the target's rate) | `1/24` (`1/24`, `1/32`) | `r_0 = 3/4`, `r_1 = 23/16` (`9.4` cells inside the contracted pole `r₊ √(1 − v²) = 1.828` at `1/24`) | measured to `40 M` at `4/M`, with `1/40`: `ℋ` `0.81–1.08×` the resting hole's in every shell at every `h` and time, the error `1.1–1.6×` (the hole's position, late); order 4 outside the horizon in both pairs; `J − a` drifts at the resting hole's rate, order 4 |
| 0.5, *moving* (`v = 1/5` along `z`, its spin) | `:fitted` | `1/32` (the ring: at `1/24` the core surface is at `0.35 < a` on the equator) | `m = 20`, `n_L = 18`, `fit_cont = 1`, `lmax_fit = 12` | measured to `40 M`: stable, but `ℋ` just outside the horizon `43×` `:damped`'s at `1/32` and `176×` at `1/40`, `10–15×` its own resting hole; it does not converge from `1/32` to `1/40` at `m` fixed in cells, even at rest |
| 0.5, *along `z`* (`v = 1/5`, `1/2`) | `:damped` (the core's transit rate) | `1/24` (`1/24`, `1/32`, `1/40`) | `r_0 = 3/4`, `r_1 = 5/4` (`8.8` cells inside the contracted pole `1.616` at `v = 1/2` and `1/24`) | measured to `20 M`: order 4 in the error outside the horizon; moving over resting in `[3, 5)` `1.13×` and `2.79×`; `J`'s drift `1.74–1.77·10⁻⁸/M` at `1/40` at every speed, order 4; at `v = 1/2` `ℋ` in `[3, 5)` converges at only `2.5–3.0` from `1/32` to `1/40` from `10 M` on, and the layer residual rises again at `1/40` |
| 0.5, *across* (`v = 1/5`, `1/2` along `x`, the spin along `z`) | `:damped` (the core's transit rate) | `1/24` at `v = 1/5`; at `v = 1/2` `1/32` is not yet asymptotic (`1/24`, `1/32`; `1/40` does not fit one H200 on the bitant) | `r_0 = 3/4`, `r_1 = 5/4`, the aligned layer (`10.2` cells inside the contracted equator `1.673` at `v = 1/2` and `1/24`) | measured to `20 M`: stable; near the horizon the error is `4.1×` (`v = 1/5`) and `7.5×` (`1/2`) the aligned hole's at `1/24`, `1.6×` and `2.5×` at `1/32`, falling at order `7–8`; at `v = 1/2` the layer residual stays at `1.0` and `J`'s drift is `2.2×` the aligned hole's at `1/32` |
| 0.5, *harmonic* | `:damped` | `1/48` (`1/48`, `1/64`, `1/96`) | `r_0 = 3/5`, `r_1 = 3/4` (`7.4` cells inside the pole `0.866` at `1/64`); `r_0 = 23/40` at `1/48`, for a ramp of `8` cells | measured to `24 M`: order 4 outside the horizon from `1/48` (`ℋ` just outside it `7–8.7`, the layer term); stationary near the hole; `J`'s drift `8.6·10⁻⁸` and `9.2·10⁻⁹/M` at `1/64` and `1/96`, the Kerr-Schild rows' at `1/24` and below their `1/40`; `M_irr`'s order 4, `8×` the Kerr-Schild rows' at `1/40` at `1/96` |
| 0.5, *harmonic* | `:fitted` | `1/64` (`1/64`, `1/96`, the margin fixed in `M`) | the offset `M/4` and the ramp `3M/16`: `m = 16`, `n_L = 12` at `1/64`, `m = 24`, `n_L = 18` at `1/96` (the core surface `M/16` outside the ring); `fit_cont = 2`, `lmax_fit = 8` | measured to `24 M`: `:damped`'s at `1/96` (`0.97–1.03×` in every shell, and its drifts); at `1/64` `ℋ` just outside the horizon `1.8×` `:damped`'s; with the margin fixed in cells (`m = 16`, `n_L = 12` at `1/96`) `19×` there, not converging, and `J`'s drift `37×` |
| 0.5, *harmonic, along `z`* (`v = 1/5`) | `:damped` (with the target's rate) | `1/96` (`1/64`, `1/96`; two H200s at `1/96`) | the resting layer | measured to `20 M`: the error `1.07–1.38×` the resting hole's at `20 M`; `ℋ` `1.0–1.14×` but for a pulse that leaves the hole at about `4 M` (`6–9.5×` in `[4, 7)`); every excess converging at order `7–10`; no trailing side; `J`'s drift the resting hole's; at `1/64` the error `2.3–3.3×` |
| 0.5, *harmonic, across* (`v = 1/5`, `3/10` along `x`) | `:damped` (with the target's rate) | `1/96` (`1/64`, `1/96`; four H200s at `1/96`) | the resting layer (the least radius across the spin is the pole's) | measured to `20 M`: at `v = 1/5` the error `1.7–1.8×` (rising from `1.1–1.4×` at `1 M`), `ℋ` just outside the horizon `4.3×` on the trailing side and the pulse `3.1×`; at `v = 3/10` `3–16×`; every excess converging at order `8–11`, and `J`'s drift's at `10–14` (`6000×` below G5's `0.013/M` at `1/64`) |
| 0.5, *harmonic, moving* (along `z` and across, `v = 1/5`) | `:fitted` | none holds at `1/64` | the resting margin, `m = 16`, `n_L = 12` | measured to `20 M` at `1/64`: `1.6–85×` the resting hole's along the spin and `60–350×` across it (`43–137×` with `trail_ramp = 9/10`, which along the spin is worse); across it at `v = 3/10` it fails within `2 M` |
| 0.3 | `:fitted` | `1/24` | `m = 16`, `n_L = 18`, `fit_cont = 2`, `lmax_fit = 12` | the rules (`fit_cont = 2` not yet run with spin) |
| 0.6 | `:damped` | `1/24` (`1/24`, `1/32`) | `r_0 = 3/4`, `r_1 = 3/2` | measured: order `4.0`; `1/16` is not convergent |
| 0.6 | `:fitted` | `1/32` (`1/32`, `1/48`) | `m = 20`, `n_L = 18`, `fit_cont = 1`, `lmax_fit = 12` | measured at `1/32`: `ℋ` outside `3.2×`, the error equal; the pair predicted |
| 0.7 | `:damped` | `1/32` (`1/32`, `1/48`) | `r_0 = 0.85`, `r_1 = 1.45` | the rules |
| 0.7 | `:fitted` | `1/48` | `m = 18`, `n_L = 24`, `fit_cont = 1`, `lmax_fit = 12` | a guess: screen the depth first |
| 0.7, *harmonic* | `:fitted` | `1/128`, no pair (two H200s) | `m = 16`, `n_L = 12`, `fit_cont = 2`, `lmax_fit = 8` (the core surface `0.08 M` outside the ring); `:damped` does not fit (the ring at `0.7`, the pole at `0.714`) | measured to `24 M` at rest: `ℋ` just outside the horizon flat at `1.6·10⁻⁴` (`160×` the `a = 1/2` hole's at `1/96`), but the error from `5/4` out grows from `10 M` (`70×` in `[2, 4)` by `24 M`) and `J` drifts at `−1.4·10⁻⁶/M`; `1/80` and `1/96` leak `600–3700×` (`8 M` screens); the moving rows were priced, not run |
| 0.8 | `:damped` | `1/32` (`1/32`, `1/48`) | `r_0 = 0.95`, `r_1 = 1.35` | the rules |
| 0.8 | `:fitted` | `1/64` | `m = 16`, `n_L = 30`, `fit_cont = 1`, `lmax_fit = 12` | a guess: screen the depth first |
| 0.9 | `:damped` | `1/96` (`1/96`, `1/128`) | `r_0 = 19/20`, `r_1 = 5/4` | measured: order `4.0` outside the horizon from `1/96` (from `1/72` except `ℋ` just outside it), to `32 M`; stationary to `128 M`; `1/48` is not convergent at `r_0 = 19/20` or `1` (`1/48 → 1/96` is `1935×` just outside); `r_0 = 1` equals `19/20` outside the horizon at `1/96` and is `13×` worse inside |
| 0.9 | `:fitted` | `1/96`, no pair in reach of one H200 | `m = 16–18`, `n_L = 36`, `fit_cont = 1`, `lmax_fit = 12` | measured: stable to `128 M` at `m = 16`, but `ℋ` outside `1900×` `:damped`'s (the error `23×` at `128 M`) |
| 0.9, *along `z`* (`v = 1/5`) | `:damped` | `1/72` (`1/72`, `1/96`; `1/128` moving does not fit one H200) | `r_0 = 19/20`, `r_1 = 5/4`, the resting layer; radii `32, 16, 8, 4, 3, 2` | measured to `20 M`: order 4 in the error outside the horizon from `1/72`; moving over resting in `[3, 5)` `0.86×`; `J`'s drift order 4; the finder's offset in `|J| − a` `1.1·10⁻⁵` (`6.8·10⁻⁶` at rest) |
| 0.9, *along `z`* (`v = 1/2`) | `:damped` | `1/72` (`1/72`, `1/96`) | `r_0 = 19/20`, `r_1 = 9/8` (`5/4` is outside the contracted pole `1.244`) | measured to `20 M`: stable, order 4 outside the horizon, moving over resting `2.06×`; but the same layer **at rest grows** inside the horizon from `10 M` (`×382`, `×36`, `×3.4` over `[10, 20] M` at `1/72`, `1/96`, `1/128`), so it is not a layer for a hole that is ever at rest; the finder's offset in `|J| − a` `1.6·10⁻⁴` |
| 0.9, *across* (`v = 1/5`, `1/2` along `x`, the spin along `z`) | `:damped` | `1/72` (`1/72`, `1/96` at `1/5`; `1/72`, `1/80` at `1/2`) | `r_0 = 19/20`, `r_1 = 5/4`, the resting layer at every speed (the least radius across the spin is the pole's `1.436`, `13.4` cells at `1/72`); radii `32, 16, 8, 4, 3, 2` | measured to `20 M`: stable, order 4 outside the horizon; moving over resting in `[3, 5)` `0.87×` and `1.96×`; at `v = 1/5` the aligned row outside the horizon; at `v = 1/2` the error in `[r_h, 9/4)` `19×` the resting hole's at `1/80` (order 8), the band inside the horizon and the layer residual not converging; the finder's offset the resting hole's (`6·10⁻⁶`) |

- **The ring rule, for both variants.** On the equator Kerr-Schild's `H`
  varies on the length `L = (R² − a²)/R` at the core surface's radius `R`;
  resolve it with at least 8 cells. The three runs that failed for this
  reason had `4.0–4.7` cells (`a = 0` `:fitted` at `1/16`, `a = 3/5` and
  `a = 9/10` `:damped` at `1/16` and `1/48`), every run that converged or
  held had at least `6.5`. (The `a = 3/5` record reads its `1/16` row as the
  margin's 4.8 cells; at `a = 0` a margin of 4 cost only `2×`, which favours
  the ring.) For `:damped` the rule sets `r_0` and the coarsest `h`; for
  `:fitted`, whose initial data are analytic down to the core surface, it
  bounds `m + n_L` on the equator. **Necessary, not
  sufficient:** at `a = 9/10` and `1/48`, `r_0 = 1` gives the ring 9.1 cells
  and is `3.7×` *worse* than `19/20`'s 4.7, with a ramp of 12 cells against
  14.4. `1/96` is needed whichever `r_0` is used.
- **`:damped`**: a ramp `r_1 − r_0` of at least 12 cells (18 lowers `ℋ`
  outside about `3×` at `a = 0`), and `r_1` at least 8 cells inside the
  horizon at the poles, `r₊`; more depth than 8–12 cells did not help.
- **`:fitted`**: a margin of 16–20 cells — the leak halves for every 2–3
  more, and at `a = 0` nothing improved past about 18 — and a ramp of at
  least 18. At high spin there is a depth bound besides: at `a = 9/10` the
  offset surface must stay within `0.19 M` of the polar horizon (`0.21 M` holds
  a bounded ring at the pole, `0.25 M` grows on the axis), while `0.625 M` is
  fine at `3/5`; between the two it is unknown. `fit_cont = 2` won every pair
  at `a = 0`; the spinning Kerr-Schild runs have used `fit_cont = 1` only.
- **In the harmonic chart with spin** (step H3) a spherical layer lies
  between the ring `a` and the pole `√(M² − a²)`: `:damped` takes `r_0`
  outside the ring and fits only well below `a = M/√2` (`[1/2, 0.866)` at
  `a = 1/2`; `0.014 M` at `7/10`, where only `:fitted` runs). `:fitted`'s
  margin is a depth in `M`: the core surface `(m + n_L) h` below the horizon
  orders the rows, `7/16 M` holds at `a = 1/2` (`M/16` outside the ring on the
  equator), and fixed in cells it stops converging just outside the horizon.
- **A moving hole** (measured at `a = 0`, `v = 1/5`):
  - **`:damped` at the default `4/M`, with the target's rate.** Without `(1 − w) ∂_t u_exact` in the layer and the core's
    own rate the layer lagged a moving target by `|∂_t u|/ρ_max` — a
    residual of `6.5` at `4/M` and `3.9` at `8/M` at every `h`, the error
    just outside the horizon `28–44×` the resting hole's at `4/M` and `1/16`
    — and `8/M` was the remedy. With the rates (`core_follows`) the residual is `0.09–0.21`, `ℋ` just outside the horizon is
    the resting hole's from `1/24` on, and `4/M` beats the old `8/M`
    everywhere (`ℋ` over the evolved region `0.08–0.36×`). `8/M` now helps a
    moving hole only as much as a resting one: at `1/16`, where it lowers
    `ℋ` just outside `1.6–1.8×` at rest too, and not from `1/24` on.
  - **Check `r_1` against the contracted horizon**, `2√(1 − v²)` along the
    boost (`1.960` at `1/5`): the octant's `r_1 = 3/2` is `7.4` cells inside
    it at `1/16`, below the rule's 8; `23/16` is `8.4`.
  - **`1/16` is adequate** — at `4/M` `ℋ` just outside the horizon
    `1.4–1.6×` the resting hole's and the error there `1.5–2.1×` from `4 M`
    to `16 M`, every shell from `r = 3` out within `1.4×` — and `1/24`,
    `1/32` is the convergence pair (order `4.0–4.1` in every shell).
  - **The error against the analytic solution grows late, and `ℋ` does
    not**: the found center runs ahead of the analytic one by truncation
    error (`1.6·10⁻⁶` at `40 M` at `1/32`, order four), and by `40 M` the
    error is `2.6–3.4×` the resting hole's in every shell at every `h`.
  - The core must follow its target (`core_follows`): with
    `du = 0` the run died at `r_0/v = 3.1 M`.
  - **Along its spin** (measured at `a = 1/2`): the rotating
    quadrant, and everything above holds unchanged — `ℋ` the resting hole's,
    order four, the late error the position (`z − c_z = 1.2·10⁻⁶` at `40 M` at
    `1/32`). Take `r_0` outside the ring and `r_1` against the contracted
    *pole*, `r₊ √(1 − v²)`. A `:fitted` core surface must clear the ring on the
    equator, `m + n_L` cells below `√(r₊² + a²)`: at `m = 20`, `n_L = 18` and
    `a = 1/2` that rules out `1/24`. The spin's drift is the resting hole's.
  - **Choose `r_1` for the fastest speed the run will reach** (from the aligned campaign). Put it at least 8 cells, at the
    coarsest `h` of the family, below the contracted pole `r₊ √(1 − v²)` at
    that speed, and keep the radii fixed in `M` across `v` and `h`. At
    `a ≤ 1/2` one layer then serves every speed to `1/2`: `r_0 = 1/2`,
    `r_1 = 19/16` at `a = 0` (`8.7` cells at `1/16`), `r_0 = 3/4`,
    `r_1 = 5/4` at `a = 1/2` (`8.8` at `1/24`). At `a = 9/10` it does not.
    The contracted pole at `v = 1/2` is `1.244`, so `r_1 ≤ 9/8` at `1/72`,
    and that layer, `22` cells below the resting pole, grows at rest; the
    resting `r_1 = 5/4` holds `v ≤ 1/5` only. Until a layer holds both, pick
    the `a = 9/10` layer by the speed, and do not run `9/8` at rest.
    **Across the spin** the motion contracts the equator
    and not the pole, and `r_1 = 5/4` serves `a = 9/10` at every speed to
    `1/2`; read the margin from `horizon_min_radius`, which is exact there.
  - **Across its spin** (the bitant): the same layers, at
    twice the points of the quadrants. Expect an excess near the horizon at
    the coarser spacings that the aligned hole does not have: at `a = 1/2` it
    is `4–8×` the aligned hole's error at `1/24` and `1.6–2.5×` at `1/32`.
    Read a convergence claim at `v = 1/2` from `1/32` on, and from a pair
    finer than `1/72 → 1/80` at `a = 9/10`, which needs several GPUs.
  - **The moving core relaxes at `max(ρ_max, 12 |v|/r_0)`** (`031ab30`), with
    its target's rate mapped in ADM variables (`281a639`), both automatic.
    Without them `a = 0` at `v = 1/2` failed in its first chunk, then held a
    residual of `4.7` at every `h`. A small `r_0` raises the core's rate, and
    `chunk_interior` refuses one above `1/dt`. What is left at `v = 1/2` is a
    first-order residual at the point the core has just released; at
    `a = 1/2` it rises again at `1/40`.
  - **Production cadence: `chunk = M`, no split** ([The analysis
    cadence on an H200](#the-analysis-cadence-on-an-h200)):
    `2.5×` cheaper than `M/2` with the trailing/leading split. Turn the split
    on to diagnose a moving layer's two sides.
- **Screen a new `:fitted` depth** before a long run: the same tracked
  geometry with the exact target (`variant = :damped`) for `6 M` at the
  run's `h`, about an hour and a half at `1/96` — a bad depth grows from
  `1 M`. Only the scratch copy on Symmetry (`spin-sync/`) can run it today:
  `test/octant_runs.jl` needs its `variant=` option, and the singular-set
  check, which tests the core's least radius against `a` and so refuses a
  polar radius below the equatorial ring, needs to know the ring's plane.

  **Inconsistent:** `CLAUDE.md`'s "Commands" says that `test/octant_runs.jl` takes `variant=` and `trail_ramp=` for a tracked layer, while this item says only the scratch copy `spin-sync/` can run the screen because the script needs its `variant=` option.

- **What would firm this up**, in order:
  - depth screens at `a = 0.7` and `0.8`;
  - `fit_cont = 2` with spin, at `3/5`.
- **What it costs** on one H200 ([earlier runs](HISTORY.md#single-black-holes-recommended-settings)). On the capsule of the moving `a = 1/2`
  hole on the rotating quadrant, `1/48` is `94.7 M` points, about `141 GB` for `:damped` and `171 GB` for
  `:fitted` extrapolated linearly: not one H200. The aligned campaign (`chunk = M`, no split,
  TreeAMR 0.2.4) cost `21–34 ns` per point and step at its finest spacings:
  `75 s/M` at `a = 0`, `1/32` (`21.6 M` points, `44 GiB`), `252 s/M` at
  `a = 1/2`, `v = 1/2`, `1/40` (`58.4 M`, `92 GiB`), `587–605 s/M` at
  `a = 9/10`, `1/128` at rest (`63.4 M`, `101 GiB`), and `628 s/M` at
  `a = 9/10`, `v = 1/2`, `1/96` (`69.9 M`, `114 GiB`). That is `20 M` in
  `0.45`, `1.44`, `3.4–3.9` and `3.5 h`. A moving `a = 9/10` capsule at
  `1/128` is `106–166 M` points, about `170–260 GiB`: several GPUs.
- **A convergence family fits one H200 only with thin outer levels**
  (`SINGULARITY_HANDLING.md`, "`a = 9/10` `:damped`
  converges at `1/96`"). Each coarse level of a nested-cube mesh is at least
  the 56-block band 2:1 balance allows, `6.2 M` points at `N = 48`; with
  `N = 24` and `radii = 32,16,8,4,4,3` on `L = 64` it is `0.77 M`, and the
  `a = 9/10` mesh is `32.9 M` points at `1/96` (`322 s/M`) and `78.0 M` at
  `1/128` (`803 s/M`, `123 GB`). The finest cube `[0, 3]³` is what is left,
  and `1/192` would not fit. The thin levels cost the exterior's error
  (`2–17×` B's outside the horizon at `1/96`), not the interior's.
- **The finest cube must be a whole number of its parent level's blocks**.
  With `N = 96` the `1/48` level's blocks are two units
  wide, so a `1/96` cube of radius `3` refined all of `[0, 4]³`, and 2:1
  balance cascaded a level out to the root: 304 M points, out of an H200's
  memory. `roots = 4`, `N = 48` gives the same spacings with half-unit blocks
  at `1/96`.

**The ring rule limits `:fitted` to `h ≤ 1/32` at `a = 1/2`**.
  The `:fitted` core surface is `m + n_L` cells below the horizon, and on the
  equator (`√(r₊² + a²) = 1.932`) it must clear the ring `r = a`: at `m = 20`,
  `n_L = 18` it is at `0.74` at `1/32` and `0.35 < a` at `1/24`; the rules'
  least `16 + 18` cells would put it at `0.52`, on the ring. Spin raises the
  coarsest spacing `:fitted` can run before it raises `:damped`'s, whose `r_0`
  is in `M`.

### The analysis cadence on an H200

A moving-hole run on one H200 cost `71–175 ns` per point and step, against the right-hand-side kernel's `1.1 ns` a
point ([below](#the-right-hand-side-on-an-h200)). This
benchmark asked where that goes. The setup is the spinning
hole's `1/32` row of `SINGULARITY_HANDLING.md`'s "A spinning hole along its
axis": `a = 1/2`, `v = 1/5` along `z` on the rotating quadrant's capsule,
`28.05 M` points, `112` steps per `M`, `:damped` at `4/M`, run to `6 M`. Six
runs varied the chunk, the trailing/leading split (`split=`) and the finder
(`finder=`), on TreeAMR 0.2.4, jobs 580683–580688 in `boosted-ks-5`. The logs
are in `.claude/data/benchmark-cadence/b1`…`b6`, and `s/M` is the wall time
from `4 M` to `6 M` over two:

| run | chunk | split | finder | `s/M` |
|---|---|---|---|---|
| `b1` | `M/2` | on | on | `239` |
| `b2` | `M/2` | off | on | `130` |
| `b6` | `M/2` | off | off | `129` |
| `b3` | `M` | off | on | `97` |
| `b4` | `2M` | off | on | `80.5` |
| `b5` | `2M` | off | off | `80.1` |

- **The split costs `109 s/M`** at `chunk = M/2`, `55 s` a chunk (`b1 − b2`).
  It runs the ADM monitor and the error kernel over the whole mesh once per
  half of each of the six shells, twelve times a chunk, so about `4.5 s` per
  pair.
- **The rest of the analysis row costs about `33 s` a chunk** (`b2 − b3`,
  and twice `b3 − b4`). That row runs the ADM monitor and the error kernel
  seven times (globally and once per shell), and the GH monitor, the per-level
  norms and the record's own analysis besides. About `4.7 s` per pair, the
  same as the split's.
- **The finder costs under a second a find**: `1.1 s/M` at two finds per `M`
  (`b2 − b6`), `0.35 s/M` at one per `2M` (`b4 − b5`), with the spin, at
  `N_ah = 12`.
- **The evolution alone is about `64 s/M`**: `b3` less one chunk's analysis.
  That is `0.14 s` per right-hand side with its ghost fill and RK4's
  arithmetic, and `20 ns` per point and step.

So on the H200 the analysis, not the finder or the evolution, was most of a
run's cost: `175` of `239 s/M` in `b1`. [Analysis
quantities](#analysis-quantities) prices the ADM monitor at "about one RHS".
By the inference above, a pair of the ADM monitor and the error kernel is
about `30` right-hand sides here (`4.5` against `0.14 s`); the kernels were
not timed one by one.

**The rule for a production run**: `chunk = M` and
`split=0`, `97 s/M` here, two and a half times cheaper than `b1`. The
analysis is then a third of the run. Turn the split on only to diagnose a
moving layer's two sides, for a run that needs them. `chunk = 2M` saves
another `17 %`, at half the time resolution of the record and of the
checkpoints. The aligned campaign of
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#aligned-spin-and-velocity-which-holes-the-algorithm-holds-2026-10-22)
ran at `chunk = M` without the split, at `21–34 ns` per point and step at
its finest spacings and `72–78 ns` on its smallest meshes (`2.7–3.7 M`
points), where the analysis weighs most.

**The measurements above predate the folds:** the row no longer re-runs a monitor per shell or
half-shell. `test/octant_runs.jl`'s `observe` runs the GH monitor, the ADM
monitor and the error kernel once each and folds three times (`mesh_fold`,
[Analysis quantities](#analysis-quantities)), so the split costs one larger
accumulator and not twelve monitor pairs, and the rest of the row about one
pair where it was seven — by the inference above about `4.5 s` a chunk on
this mesh in place of `33 s`, and `55 s` of the split's gone. That is a
prediction from the table, not a measurement; the H200 timing is still to
be made. On the development machine (four threads, the quadrant smoke run
of `CLAUDE.md`, 0.24 M points) the second chunk's wall time went from `22.1`
to `19.9 s` with the split, and the CSV is the old one to its ten digits.

### The right-hand side on an H200

An investigation of how the right-hand side can be evaluated faster on a GPU.
Scope:

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
- **Where the code is:** the prototypes are raw CUDA.jl kernels and KA kernels in
  `bench/rhs_lab.jl`, whose modes `baseline`, `variants` and `round3` … `round9` are
  the rounds in [`HISTORY.md`](HISTORY.md#the-right-hand-side-on-an-h200). It runs from a copy of the package with CUDA added to its
  `Project.toml` (on Symmetry `rhs-gpu-lab`, jobs 570186–570203). Its companions:
  - `bench/rhs_lab_source.jl`, the lean source, shared by the GPU and CPU scripts;
  - `bench/rhs_lab_cpu.jl`, the source's CPU check and timing, in the package's own
    environment;
  - `bench/sass_stats.jl`, which counts a SASS dump's instructions.

**On a device the kernel is not inlined.** KA launches it with a
default `CUDABackend()`, whose `always_inline = false` leaves GPUCompiler to Julia's
inlining heuristics. These keep closures as real functions ([the ones the kernel
had](HISTORY.md#the-right-hand-side-on-an-h200)).
A device call passes its `SVector` and `SArray` arguments through the stack, and a
closure that indexes an `SVector` with its argument indexes memory.

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

**The fused kernel's order matters once the source is lean.** The fast kernel keeps
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

**The order of the work.** Items 1–3 are done ([history](HISTORY.md#the-right-hand-side-on-an-h200)), the
kernel without `Val(N)`: static strides are worth 4 %, and a kernel
specialised on the block size would be compiled again at every `N` the suite
runs. Half of item 5 is done. What
remains is **(proposed)**:

4. **TreeAMR's scatter and transfer kernels at bandwidth:** 0.74 → ~0.25 ns/pt at
   `N = 32`, and more at `N = 16`. Upstream. **(The first change TreeAMR should
   make:** [Upstream prerequisites](#upstream-prerequisites), item 5.**)**
5. **`metric_derivatives` and the algebraic gauge source made lean** like the
   source. **(Half done:** `metric_divergences` forms the contraction
   alone; the algebraic source has lost its closures but not its spills.**)**
6. **Then, if its 10–15 % is wanted, a native stepper** with a ghosted state and the
   stage update fused into the right-hand side.

Not worth doing, measured ([history](HISTORY.md#the-right-hand-side-on-an-h200)): shared-memory staging of the principal part, `maxregs`,
one component per thread, further workgroup tuning.

**Implemented.** Items 1–3, and half of 5, are in `src/`.

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

**The symptom of a closure that a device left as a call** is a `CALL` to `julia_…` in the SASS
(`bench/rhs_lab.jl mode=baseline` writes it, `bench/sass_stats.jl` counts it).
On the CPU the same closures cost 7× (`bench/rhs_lab_cpu.jl`).

### The right-hand side on a CPU

The H200 study's question, asked of Symmetry's AMD nodes: where does an evaluation go
on a CPU, and what would SIMD **across grid points** buy? Scope: the H200 study's — `q = 4`, no interior,
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
  scalar code's operations in its order. That is the gauge wave's, not the kernel's. On a hole the lanes
  differ from the scalar kernel in the last bits — up to `1.2e−14` on the step-5
  fixture, 1.0–2.0 eps of the size of the terms at evolved points (`rhs_scale`) —
  because StaticArrays forms `inv(g4) * h` with
  `muladd`, and LLVM fuses each into an FMA or not by the code around it, which the
  lanes change; the gauge wave's zero shift hides it. So the claim is roundoff, as for
  every two call sites of one body; bit-identity across thread counts is unaffected.
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

**What SIMD in `src/` takes** ([the plan](HISTORY.md#the-right-hand-side-on-a-cpu)).

1. **A lane type.** SIMD.jl's `Vec{W,Float64}` works through the package's algebra
   unchanged but for the three methods above; SIMD.jl has no dependencies of its own. SIMD.jl is a dependency; `pointwise.jl` has `_scale`,
   `_anynonzero` and `_select`, and `_η4` is spelled with `one(T)` and `zero(T)`; no
   method is added to a type the package does not own.
2. **The launch**, unchanged on a device
   (`W = 1`), with `W` chosen where `GHProblem` builds its `Val`s. The suite's `N = 10` at `q = 6` is not a
   multiple of four. The launch stays `map_blocks!`'s, and the
   item at the start of each group of `W` does the group — `is_lane_leader`, the other
   items do nothing. A row that `W` does not divide ends in a group that **overlaps**
   the one before it, in place of a scalar tail or masks: no lane is
   ever outside the row, and the shared points are stored twice by the same thread.
   `W` is `default_simd_width`'s: 32 bytes of lanes, 64 with AVX-512, at most `N`.
3. **No `DomainError` from a lane.** SIMD.jl's `sqrt` is the instruction: a degenerate
   metric gives a `NaN`, which the record's `finite` reads, where the scalar kernel
   throws (`CLAUDE.md`, the outer part of the layer). The layer
   and the core keep the scalar code and its `DomainError`, and `simd_width = 1`
   restores it everywhere.
4. **The interior and the gauge sources.** A group
   takes the lanes when all its points are outside the layer and the scalar code point
   by point otherwise; `Lanes` reads `Hsrc` by its cartesian index; the algebraic
   source's two clamps are `_select`s.
5. **The zero weights** need the weights in the type: `derivative_weights` returning a
   type whose `axis_stencil` and `mixed_stencil` are generated without the zero terms.
   **(Not done:** independent of the lanes, and left for a decision.**)**

**Implemented.** The kernel above is the package's (`src/lanes.jl`,
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
  counts at eight lanes as at four). That settles `default_simd_width`'s 64 bytes.
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

**Not measured.** The monitor kernels, which stay scalar, and hardware counters: `perf` is not open to users on Symmetry
(`perf_event_paranoid = 4`), so instructions a cycle and pipe occupancy above are
inferred from the native code and the clock.

**On the CPU the kernel's algebra runs on SIMD lanes.** Bit-identity across thread counts is unaffected — a block is one thread's.

## Binary black holes (proposed)

*Status: a design for review. Nothing in this section is built or
measured.* It extends the proof of concept to two black holes that orbit,
merge and ring down, with the gravitational waves they emit as the result.
`PLAN.md`'s steps B1–B9 are the work breakdown, and the two milestones are
under [Milestones G7 and G8](PLAN.md#milestones-g7-and-g8): two holes colliding
head-on (G7), and an equal-mass binary through inspiral, merger and ringdown
(G8). What it needs Erik to decide is under [What this asks Erik to
decide](#what-this-asks-erik-to-decide). A second route for the orbiting
pair — the holes at rest on a grid that rotates with them, the damped
harmonic gauge and `:fitted` — is under [A comoving grid frame for orbiting
holes](#a-comoving-grid-frame-for-orbiting-holes-proposed). A third choice
of gauge, with no source at all until the merger, is under [Harmonic gauge
for the inspiral](#harmonic-gauge-for-the-inspiral-proposed).

**Units.** The total mass is `M = 1`, so each hole of an equal-mass pair has
`M_A = 1/2`. Every single-hole setting in [Single black holes: recommended
settings](#single-black-holes-recommended-settings) is in
units of the hole's own mass and scales by `M_A`: the spacing at the hole,
the layer's radii, `ρ_max = 4/M_A = 8/M`, `γ0`'s `1/M_A`. The
convergence pair `1/24`, `1/32` of the moving `:damped` hole is `1/48`,
`1/64` here. The orbital speed of each hole is about `½ √(M/d)` at
separation `d` (Newtonian): `0.16` at `d = 10 M`, `0.29` at `d = 3 M`.
That is inside what the single moving holes measured.

### What the single holes hand over

Everything a binary needs *per hole* has been run on one hole that moves:

- **The moving `:damped` layer.** With its target's rate and a core that
  follows it holds a boosted Kerr-Schild hole at the
  resting hole's level and converges at order four outside the horizon.
  The aligned campaign extends this to `v = 1/2` and `a ≤ 9/10` along the
  spin. The bitant campaign moved holes across their spin, which is a
  binary's own configuration: spins along `z`, motion in the plane `z = 0`.
  Every row there was stable and converged at order four outside the
  horizon, except at `v = 1/2`, which is faster than a binary's holes move
  before merger.
- **The `:fitted` layer** holds every moving hole it was given. But `ℋ`
  just outside the horizon is `13–15×` `:damped`'s at `a = 0` and `43–176×`
  at `a = 1/2`, and it does not converge at a margin fixed in cells.
- **The tracked geometry**, the finder along a trajectory, coasting and the
  jump refusal.
- **The algebraic Kerr-Schild source**, which moves with its hole.
- **Checkpoints and job chains**, MPI with the analysis on rank 0, and
  SimWatch.
- **From TreeAMR 0.2.6**: the half turn (M14), `interpolate` that folds
  points through mirrors and seams and returns second derivatives, and the
  relative boundary hook (`CellBoundary(Relative(f))`), which reads the
  block's own interior.

What does not carry over is the object every case is built around:
`GHCase`'s `background`.

### A case without a background

One analytic background supplies seven things to a single-hole case. A
binary has no analytic solution, so each needs a replacement:

| what | from the background today | for two holes |
|---|---|---|
| the initial data | `state(background, 0, x)` | superposed Kerr-Schild, then XCTS ([Initial data](#initial-data)) |
| the Dirichlet data | `state(background, t, x)` at the boundary | the initial data, frozen, at a far boundary ([The outer boundary](#the-outer-boundary-and-the-wave-zone)) |
| the gauge source | `−Γ_a` sampled, or `KerrSchildSource` | a composite of per-hole sources ([The gauge](#the-gauge)) |
| the layer's target | `u_exact` | per hole, from its track or a fit ([One interior per hole](#one-interior-per-hole)) |
| the error reference | `u_exact` | none: constraints, self-convergence and the horizons |
| the center | `HoleCenter`, a line | per hole, a track ([The tracks](#the-tracks)) |
| the radii and the seed | `horizon_min_radius`, `analytic_horizon_radius` | per hole, from the hole's boosted Kerr-Schild parameters, then from the finds |

**(Proposed.)** A hole becomes a value of its own: `HoleSpec{T}`, `isbits`,
holding the mass, the spin vector, the initial center and velocity, and the
interior's spec (an `Interior`'s radii or a `FittedSpec`). `GHCase` takes
`holes::NTuple{K,HoleSpec}`, with `K` a type parameter so that the kernel's
composition below is unrolled. `background` stays, renamed for what it
becomes: the *data*, read at `t = 0` and by the frozen boundary hook, and
never as the truth. A single-hole case is `holes = (spec,)`, built from its
background as now, and is the case it was **bit for bit**: that is the first
claim of step B3. `error_norms` refuses a case whose data are not a
solution, and says why.

### The gauge

The gauge source must:

- make the initial data nearly stationary near each hole, so that the run
  starts with the data's own junk and no gauge transient on top;
- move with the holes;
- be harmless far away, where the waves are extracted;
- survive the merger;
- stay algebraic in `g`, so that the principal part is still GHSO2's and
  no field set has to be re-sampled.

There are two candidates, and both are algebraic.

**The tracked Kerr-Schild source (proposed first).**
`H_a = Σ_A w_A(x, t) H^A_a(g)`. `H^A` is `KerrSchildSource`'s closed form
with hole `A`'s `M_A`, `S_A` and its 4-velocity `u_A(t)` from the track. The
weights are a partition of unity,
`w_A = Π_{B≠A} r_B^p / Σ_C Π_{D≠C} r_D^p` with `r_A = |x − c_A(t)|` and
`p = 4` **(proposed)**. Then `w_A = 1 − O((r_A/d)^p)` on hole `A`'s
horizon, `10⁻⁴` at `d = 10`, and with one hole the source is
`KerrSchildSource` bit for bit.

- **What it buys.** Near each hole the data are a boosted Kerr-Schild hole
  in its own gauge, up to the companion's tidal field. That is the gauge
  every single-hole number was measured in, and the gauge the per-hole
  `:damped` target is written in.
- **What it adds to `∂_a H_b`.** Two terms the single hole does not have:
  `∂w_A`, analytic in `x`, with `∂_t w_A` through the track's velocity; and
  `(∂H^A/∂u_A) du_A/dt`, through the track's acceleration. Both are small,
  `|du_A/dt| = M/(2d²)`: `0.005/M` at `d = 10` and `0.06/M` at `d = 3`. Both
  must be in the kernel anyway. If `∂_t H` is not the time derivative of the
  `H` in use, the system is not the generalized harmonic system for any `H`,
  and the difference shows up in `C_a`.
- **At merger** the source becomes the remnant's: one hole with the common
  horizon's `M` and `S` and its track's velocity, blended in time over
  `10 M` **(proposed)**.
- **What is not known** is whether the composite survives the last orbit.
  There the weights change over the holes' own size, and neither hole's
  `H^A(g)` sees only its own hole.

**The damped harmonic gauge**, rolled on from the tracked source. This is
Szilágyi–Lindblom–Scheel 2009, in SpECTRE's form as written under [Open
questions](#open-questions):
`H = f(t) H_KS(g) + (1 − f(t)) H_DH(g)` with `f = e^{−((t − t_0)/σ_g)⁴}` and
`σ_g = 15–25 M`, as SpEC uses.

- **Why it rolls on from the tracked source.** SpEC rolls on from the
  initial `H(x)` frozen in space. That is right in SpEC's comoving frame and
  wrong here, where the holes cross the grid: `8 M` in `50 M` at `d = 10`.
  The tracked source moves with them.
- **What it buys.** It is the gauge with a record through merger, and it
  knows one rate and nothing else about the holes.
- **What it costs.** After the roll-on, the analytic per-hole target is in
  the wrong gauge, so this gauge pairs with `:fitted`.

**The pairing is the decision.** The two coherent choices are:

- **(α)** the tracked Kerr-Schild source with `:damped` per hole: accurate
  per hole, and untested through a merger;
- **(β)** the damped harmonic gauge with `:fitted` per hole: SpEC's gauge
  with the weaker interior.

Step B1 builds both sources and runs each on a single moving hole. Step B7's
head-on collision is the first test of either through a merger.
**(Proposed:)** build (α) first, and keep (β) as the measured alternative.

A third pairing is noted and not planned: a Kerr-Schild source near each
hole, with the damped harmonic gauge between and beyond the holes. It is the
same composite with a third weight.

**A third candidate has no source**: `H ≡ 0` on superposed harmonic data,
with the damped harmonic gauge rolled on from zero before the merger. It
has no rest frame, so nothing has to move with the holes. It is not
expected to survive the merger on its own, hence the roll-on, and it costs
a level at the holes: [Harmonic gauge for the
inspiral](#harmonic-gauge-for-the-inspiral-proposed).

**The initial momentum follows from the gauge.** Whatever the source, the
data are `(γ_ij, K_ij)` with a lapse and a shift, and `Π` needs `∂_t α` and
`∂_t β^i` as well. Those two are free. Choosing them so that
`C_a = H_a + Γ_a = 0` at `t = 0` is four linear equations per point:
`Γ_a` is linear in `∂_t g_tb`, with `g^{tt} = −1/α²` on the diagonal
(**proposed**: `gauge_consistent_momentum`). For one boosted Kerr-Schild
hole under its own source the result must be the analytic `Π` to roundoff,
and that is its test.

### One interior per hole

**The composition.** Each hole carries its own geometry and its own target.
The kernel takes a tuple `(I_1, …, I_K)` and composes it point by point:

- `w = min_A w_A` and `ρ = max_A ρ_A`;
- the core is the union of the holes' cores;
- the target is the target of the hole whose `ρ_A` is largest at the point.

Before merger the layers are disjoint: each lies inside its own horizon,
and the apparent horizons of two separate holes do not overlap. So the
composition decides nothing until the handover below.

Everything that asks "is this point evolved?" takes the composite mask, in
which a point is evolved when it is outside every offset surface: every
norm, the indicator, the projection's gate, and the finder's footprint guard.

**The profiles.**

- `γ0`: SpEC's three Gaussians **(proposed)**. One on each track, at the
  single hole's `1/M_A` with width `3 M_A`; one on the origin, of width
  `d`, for the remnant; and `1/(10 M)` far away.
- `ε_KO`: a `HorizonDissipation` per hole.

**The radius checks** apply per hole. Each is checked against its hole's
contracted horizon at its current speed, from its boosted Kerr-Schild
parameters, and then from the finds. (`horizon_min_radius` is exact across the spin, which a binary's spinning
holes need.)

**The target, two ways:**

- **Tracked `:damped`.** The target is the analytic boosted Kerr-Schild
  hole with the hole's input `M_A` and `S_A`, on the track's center and
  velocity. `Interior` already holds a `target` metric; here it is the
  track's hole, rebuilt every piece. The core follows its target with the
  target's rate (`core_follows`), as the single moving hole's does.

  This target is not the solution near a hole of a binary. It misses the
  companion's tidal field over the layer, `M_B r²/d³`: `5·10⁻⁴` at
  `d = 10` and `0.02` at `d = 3`. It also misses the orbit's acceleration
  across the layer: `0.005` and `0.06`. The interior's calibration rule may make that
  enough: a smooth layer at `4/M` with a ramp of at least `4G` cells holds
  an inexact target and leaves the exterior what the exact one leaves.
  Step B2 measures it.
- **`:fitted`.** Nothing is new but the composition and the tracks; each
  fit is one hole's own.

### The tracks

**Prediction.** `HorizonTrack` carries the center along a line,
`c_find + v_est (t − t_find)`, with `v_est` the difference of the last two
finds. On an orbit that lags, since the velocity is the chord's at the
previous interval's midpoint. The prediction is then off by about `a Δt²`,
with `a = M/(2d²)` the centripetal acceleration:

- at `chunk = M` and `d = 10` that is `0.005 M`, a third of a cell at
  `h = 1/64`;
- at `d = 3` it is `0.06 M`, four cells.

Nothing checks it: the jump refusal is on `r_min`, not on the center. At
the next find the layer and the gauge source move by as much.

**(Proposed:)**

- **A quadratic prediction**: a least-squares fit in time over the last
  four finds. What remains is the jerk term, under a cell at `d = 3`.
- **A find cadence set by the predicted error.** Shorten the interval until
  `a Δt² ≤ G h/4`. A find costs under a second, and it runs at a piece
  boundary without the analysis row.
- **A center-jump check** beside the `r_min` one: the found origin must lie
  within `G h/2` of the prediction.
- **Continuity in time.** Each new prediction is blended into the old one
  over the next interval, so that `c_A(t)` and `u_A(t)` are `C¹`. The gauge
  source and the target both read them.

**Pieces.** A moving `:fitted` chunk is already solved in pieces of
`h/(4|v|)`. The tracked target and the gauge source are rebuilt per piece,
each on that piece's line, exactly as the single moving hole's target
moves.

**One track per hole.** Under the half turn there is one track, and its
image is the other hole ([The symmetric
domains](#the-symmetric-domains)).

### The common horizon and the handover

**Finding it.** The finder looks for a common horizon at every find once
the tracks are closer than `d_c` **(proposed `4 M`**, before formation in
every equal-mass simulation**)**.

- It is seeded on a sphere about the center of mass that encloses both
  individual horizons.
- Before formation the find fails. A failed common find is recorded, never
  thrown and never coasted: there is no common track until the first
  success.
- It needs `lmax ≈ 24`: ThreePunctures measured that `KorzynskiSpin` needs
  that on a just-formed horizon.
- **Formation** is three successive finds with a nondecreasing area that
  enclose both individual horizons: every collocation point of each
  individual surface lies at positive depth below the common one
  **(proposed)**.

**The handover (proposed).** It waits until the common *core* surface,
`m + n_L` cells below the common horizon, encloses both individual core
surfaces. Then:

1. The common layer, `:fitted`, joins the composite. It is `:fitted`
   whatever the individual variants were, since no analytic target exists
   for a hole that has just merged.
2. At the next chunk the individual layers are dropped and their tracks
   retired. Their finds stop, and losing one after formation ends nothing.

The composition makes this continuous outside the common core: those
points are under the common layer before and after. Inside the common core
`F` is not evaluated, so the individual holes' singular regions are relaxed
toward the common fit at `ρ_max`.

**What is not known** is whether a just-formed common core can hold both
individual cores at an affordable `h`. The common horizon appears around
the pair with a gap that narrows at the neck, and `m + n_L ≈ 36` cells is
`0.56 M` at `1/64`. The head-on run measures the gap. If it is too small,
the handover waits, with both individual layers active inside the common
horizon until it fits.

**The shape at `lmax = 24`** costs about `0.6 µs` per layer point, up from
`79 ns` at `lmax = 8` as `(lmax + 1)²`, but only over the common layer's
points **(predicted)**.

**After the ringdown**, about `30 M` after formation, the common horizon is
Kerr to the finder's accuracy. Under (α) the common layer may then switch to
`:damped` toward the remnant's boosted Kerr-Schild hole, through the
existing `handover`, to recover the analytic layer's accuracy for the
ringdown's tail **(proposed, optional)**.

### Initial data

Three sources, in the order the steps use them.

**1. Superposed boosted Kerr-Schild, unsolved** (steps B3–B7).
`g = η + Σ_A (g_A − η)`, each `g_A` a boosted, translated, rotated
`KerrSchild`. `Π` comes from the superposition's own `∂_t g`, or from
`gauge_consistent_momentum`.

- The data violate the constraints where the two holes' fields overlap.
  The constraint damping removes the violation, and it radiates as junk.
  These data serve the head-on collision and every development test, but
  not a waveform.
- XCTS has this metric (`SuperposedKerrSchild` in its `src/freedata.jl`).
  But it holds `Vector`s and so cannot be a kernel argument, and XCTS's
  dependencies are local paths, which this package does not take.
- **(Proposed:)** an `isbits` superposition belongs in `SpacetimeMetrics`
  upstream, where the backgrounds live. Until then it is written here, with
  the attenuation option `w_A = exp(−|x − c_A|²/ℓ²)` that XCTS's free data
  use.

**2. XCTS** (step B8, for the inspiral). It solves the five XCTS fields on
superposed Kerr-Schild free data (SpEC's direct-K variant), compactified to
`i⁰`, to `10⁻¹⁰`. Three things stand between its solution and this
package's grid:

- **XCTS's production data stop at the horizon.** Its isolated-horizon
  conditions (`horizon_bc = true`) put the apparent horizon on the excision
  sphere. This package evolves `m` cells inside the horizon before its
  layer begins, so those data would end too early.
  **(Proposed:)** use XCTS's default inner condition instead,
  `horizon_bc = false`: Dirichlet to the free data, on excision spheres
  inside each hole's core surface. `solve_xcts` takes it on the two-hole
  mesh as on the one-hole one. Everything this package evolves with `F` is
  then solved data, and the horizon is wherever the solution puts it,
  found afterwards. Inside the sphere the free data take over, and their
  derivatives jump at the sphere, which lies where the core relaxes without
  `F`. SolveConstraints, the Gridap prototype XCTS replaced, excised the
  same way, a short distance inside each horizon.
  The alternative keeps the isolated-horizon data and extends them inward
  by a regular fit (`fit.jl`'s solid harmonics). That is a turducken: the
  region between the horizon and the core violates the constraints, hidden
  causally.
- **Its solution lives on spectral elements**, in a package with local-path
  sources. **(Proposed:)** a host tool, run in an environment that has both
  packages (as `bench/`'s CUDA copies do), evaluates XCTS's solution at
  TreeAMR's points and writes the forest and the state through TreeIOHDF5.
  `evolve!` then starts from that file. The mesh is chosen on the
  superposed data of the same holes, because the initial-data cycle cannot
  evaluate XCTS in a kernel. The two data sets differ by smooth
  corrections.
- **Its frame is inertial.** XCTS ignores `orbital_omega` and carries the
  orbit in per-hole boosts, so the boosts must be the quasi-circular
  velocities. Eccentricity reduction comes later.

XCTS supplies the lapse and the shift; `gauge_consistent_momentum` supplies
`∂_t α` and `∂_t β`.

**3. Not ThreePunctures.** Its slices are maximal and quasi-isotropic:

- Each horizon is at a coordinate radius of about `M_A/2 = 1/4`, against
  Kerr-Schild's `2 M_A = 1`. The `m + n_L ≈ 36` cells this package's layer
  needs inside it would mean `h ≈ 1/150` instead of `1/48–1/64`.
- Inside the throat is a second asymptotic end, not a singularity that a
  layer could hide.

Its data are TreeBSSN's. The two codes are compared on the physics (the
waveform and the remnant), not on the same slice.

### The outer boundary and the wave zone

The Dirichlet hook writes the analytic solution, and a binary has none.
There are three options:

1. **Freeze the initial data at a far boundary** **(proposed)**, at
   `L > T + R_ext`, where `T` is the run's length and `R_ext` the outermost
   extraction radius. Then nothing from the boundary reaches an extraction
   sphere during the run.
   - The hook becomes time-independent: the superposed data at `t = 0`.
   - XCTS's correction to those data at the boundary is of order
     `(M_ADM − Σ M_A)/L`, about `10⁻⁵` at `L ≈ 1000 M`.
   - What it costs is outer levels ([What it
     costs](#what-it-costs-predicted)).
2. **The relative hook** (TreeAMR's `Relative`). A ghost becomes the frozen
   data plus the deviation at the nearest evolved point, which is a zero
   gradient on the deviation. An outgoing wave then leaves with a
   first-order reflection instead of Dirichlet's total one, which allows a
   smaller `L`. Step B5 measures it against (1) on a Teukolsky wave.
3. **Constraint-preserving radiative conditions** (Kreiss–Winicour 2006;
   Rinne 2006; Lindblom et al. 2006) need the characteristic fields at the
   boundary point. Not planned.

**The wave zone** gets a level floor **(proposed)**: a ball about the
origin of radius `R_ext` plus a margin, at spacing `h_w = M/2`.

- **Why that spacing.** At ringdown the `(2, 2)` mode has `Mω ≈ 0.55`, a
  wavelength of about `11 M`, which is 22 points at `q = 4`. In the
  inspiral at `d = 10` the wavelength is about `100 M`.
- **Why a floor is needed.** Without it, the Löhner indicator coarsens the
  wave zone: its global floor is set by the near zone's amplitude, about a
  thousand times the wave's.
- **The extraction radii** are `R = 50, 60, 75, 100 M` **(proposed)**.
- **Beyond `R_ext`**, each level is as thin as 2:1 balance allows, two
  blocks wide.

### The symmetric domains

| configuration | symmetry | domain | share of the box |
|---|---|---|---|
| head-on, equal masses, no spin, along `z` | three mirrors | the reflecting octant (`octant = true`): one hole on the `z` axis, the other its image in `z = 0` | `1/8` |
| head-on, equal spins along the axis | quarter turn about `z`, mirror in `z` | the rotating octant | `1/8` |
| equal masses, equal spins along `z`, orbiting in `z = 0` | half turn about `z`, mirror in `z` | `[0, L] × [−L, L] × [0, L]`, `halfturn = (1, 2)`, root brick `(r, 2r, r)` | `1/4` |
| unequal masses, or unequal spins along `z` | mirror in `z` | the bitant | `1/2` |
| precessing spins | none | the box | `1` |

**The half turn is new here.** TreeAMR 0.2.5's M14, `halfturn = (d1, d2)`,
glues the low face of `x` to itself, reversed along `y`.

- **The state's map** is the tensor's: a component changes sign by `(−1)^k`,
  `k` the number of its indices in `{x, y}`. So `h_tx`, `h_ty`, `h_xz` and
  `h_yz` are odd, the other six even, and the same for `Π`
  (`state_halfturn`).
- **TreeAMR refuses** an odd `roots[2]` and a reflecting face in `x` or
  `y`; the mirror in `z` is allowed.
- **Each hole crosses the seam twice an orbit.** The domain holds hole 1,
  hole 2, or parts of both. So the kernel's tuple holds both geometries,
  the second the image of the first under the turn: center and velocity
  `(−x, −y, z)`, and the shape's real coefficients with odd `m` negated
  **(proposed)**. There is one track, of hole 1 in the whole plane.
- **The finder needs nothing new.** TreeAMR 0.2.6's `interpolate` folds a
  point beyond the seam onto its image and turns the state, so a find of
  hole 1 at `x < 0` reads hole 2. The fit's sampler works the same way.
- **The seam plane** holds each point and its image, both evolved.
  `add_noise!` must leave them alone, as on the rotating octant.
- **Compare on uniform meshes only.** A refined half-turn domain is not the
  refined box's restriction (the octant's lesson). And a moving hole's box
  agrees with its symmetric domain only to the amplified roundoff of the
  target's rate, as `boosted_tests.jl` bounds it.

### Refinement for two holes

- **Per-hole floors.** Each hole gets a level floor from its core surface,
  widened by its travel per chunk; the mesh takes the union. Under the half
  turn the image hole has its floor too.
- **The wave-zone floor** from [the outer
  boundary](#the-outer-boundary-and-the-wave-zone).
- **The common geometry's floor**, from formation on.
- **The ceiling** at the outer boundary, as now.
- **The indicator** reads the composite mask.

A regrid moves a block by at most one level, and near merger a hole moves
`0.3 M` a chunk, 19 cells at `h = 1/64`. The travel margin has to carry the
finest level ahead of the hole, as it does for a single hole.

### The waveform

**`Ψ₄` at the extraction points, not on the mesh** **(proposed)**.
TreeAMR 0.2.6's `interpolate` returns `h`, `∂h`, `∂∂h`, `Π` and `∂Π` at a
batch of points (Lagrange `q + 2`, whose second derivative is order `q`).
From those, pointwise algebra forms:

- `γ`, `K` and `∂K`, as `adm_from_metric` and the ADM monitor do;
- the spatial Ricci tensor;
- `E_ij = R_ij + K K_ij − K_ik K^k_j` and `B_ij = ε_(i^{kl} D_k K_{l)j}`;
- `Ψ₄ = (E_ij − i B_ij) m̄^i m̄^j`, in the coordinate radial tetrad
  (Gram–Schmidt from `r̂`, as the Einstein Toolkit's WeylScal4 builds it).

The sign and the normalisation are fixed once, against the Teukolsky wave's
analytic strain, and written down. An extraction is a few hundred points
per sphere, which is far less than a right-hand side **(predicted)**.

**The modes.** `ash_transform(grid, R Ψ₄, −2)` on an
`AbstractSphericalHarmonics` grid, `ℓ ≤ 8` **(proposed)**.

**The balance laws.** The same spheres give the ADM mass, momentum and
angular momentum as surface integrals. The modes give the radiated energy,
momentum and angular momentum by integration in time (Ruiz et al. 2008).

**The cadence.** Every `M/4` **(proposed)**, at the boundaries of the
chunk's pieces, after a ghost fill with that time's hook. It runs on rank 0
through `interpolate_state`, outside the analysis row, whose monitors take
`33 s` a chunk on one H200.

**The output** is a CSV per radius, `(t, ℓ, m, Re, Im)`. Extrapolation to
infinity and the strain, by fixed-frequency integration (Reisswig–Pollney
2011), are post-processing scripts in `test/`, as `octant_study.jl` is.
`Ψ₄` on the mesh, for slices, is an extension of the ADM monitor and waits
for step 9's viewers.

### The record

- **Per hole:** the track's center and velocity, `r_min` and `r_max`, the
  horizon's `A`, `M_irr`, `J` and `M_ch`, and the fit's validity.
- **For the pair:** the separation and the orbital phase.
- **The common horizon:** its rows from its first attempt, with
  `success = false` until formation.
- **The run:** the handover's state, the constraints under the composite
  mask, and the gauge roll-on's `f(t)`.
- **The extraction** goes to its own file.

**The checkpoint** carries everything the loop now carries: the tracks,
the common track, the handover state, the predictions' history and the
extraction's running integrals.

### What it costs (predicted)

**The mesh.** An equal-mass, non-spinning binary from `d = 10 M` at
`h = 1/64` at the holes, on the half-turn quarter domain, with balls for
levels:

| region | spacings and radii | points (millions) |
|---|---|---|
| the hole (one per quarter domain) | `1/64` to `2 M`, `1/32` to `4 M`, `1/16` to `8 M` (the single hole's `4, 8, 16 M_A`) | 12 |
| about the pair | `1/8` to `16 M`, `1/4` to `32 M` | 4 |
| the wave zone | `1/2` to `100 M` | 8 |
| the far region | `1, 2, 4, 8` to `1060 M`, each level two blocks wide | 11 |
| **total** | | **35** |

Block granularity and cubes can double that. At `1.6–2.0 kB` a point that is
`55–140 GB`: one H200 at the low end, two at the high end.

**The run.** The quadrupole formula gives about `800 M` from `d = 10` to
merger, `5 d⁴/(256 ν M³)` with `ν = 1/4`. Adding the ringdown and the
wave's travel to `R_ext` makes `T ≈ 1000 M`, so `L ≈ 1100 M`.

**The time.** With `cfl = 1/2` there are 128 steps a `M`. At the aligned
campaign's `21–34 ns` per point and step, that is `100–300 s` a `M`, or
`30–85 h` on one H200 at `1/64`. The convergence triple `1/48, 1/64, 1/80`
costs `0.56 + 1 + 1.8` times that: about `100–290` H200-hours.

**The head-on collision** on the octant, from rest at `d = 10`, merges after
a free fall of about `35 M`. It needs about `200 M` and `L ≈ 300 M`: a few
hours on one H200.

**The levers:**

- **Subcycling**, which TreeAMR does not do. The outer levels step at the
  holes' time step, and with subcycling the run would cost about what its
  finest three levels cost, `4×` less. This is not a prerequisite.
- **The far boundary** is `30 %` of the points. The relative hook at
  `L ≈ 300 M` would drop the two outermost levels, about an eighth of the
  points.

### Validation

- **Self-convergence** at three resolutions: the constraints, and `Ψ₄`'s
  `(2, 2)` phase and amplitude.
- **The extraction against a known wave**: a linearised Teukolsky
  quadrupole wave (Teukolsky 1982), whose `Ψ₄` is analytic.
- **The head-on ringdown** against the Schwarzschild quasinormal mode,
  `M_f ω = 0.3737 − 0.0890 i` for `ℓ = 2`, with `M_f` from the common
  horizon.
- **The equal-mass, non-spinning remnant** against Scheel et al. 2009:
  `M_f/M = 0.9516` and `χ_f = 0.6865`.
- **The balance laws**: `M_ADM − E_rad = M_f` and `J_ADM − J_rad = J_f`, to
  the junk radiation's uncertainty.
- **The waveform against an SXS waveform** of the same mass ratio, aligned
  in time and phase in the inspiral.
- **TreeBSSN on the same physical configuration**, which is what TreeBSSN
  was built for.

### What this asks Erik to decide

All six are **(open)**.

1. **The gauge and the interior together**: (α) the tracked Kerr-Schild
   source with `:damped` per hole, built first, or (β) the damped harmonic
   gauge with `:fitted`. Steps B1 and B2 measure both on one moving hole
   before step B3 commits the composition.
2. **The case**: `GHCase` with `holes` and the background demoted to data,
   or a separate `BinaryCase`. The first keeps one driver and makes the
   single hole a binary with one hole.
3. **The superposed metric**: upstream in `SpacetimeMetrics` (not yet
   requested), or kept here.
4. **XCTS**: its default Dirichlet inner boundary on spheres inside the
   cores, or its isolated-horizon data extended inward. And the file
   handoff from a host tool.
5. **The outer boundary**: frozen data at `L > T + R_ext`, or the relative
   hook at a smaller `L` if step B5 shows its reflection is small enough.
6. **Subcycling in TreeAMR**: the largest cost lever, about `4×`, and not
   needed for G7 or G8.

## A comoving grid frame for orbiting holes (proposed)

*Status: a design for review. Nothing in this section is built or
measured; every number it leans on was measured elsewhere, and says where.*
It is a second route to [Binary black holes](#binary-black-holes-proposed)'s
milestones for two holes that **orbit**: the holes are held at rest on the
grid, the equations are given nothing that knows where a hole is, and the
control loop steers the grid and nothing else. `PLAN.md`'s steps C1–C5 are
the work breakdown. A grid that rotates rigidly is still one Cartesian grid:
one box, one global rotation and translation, no blocks fitted to the holes
and no shape maps.

### Why the tracked target is delicate

Pairing (α) — the tracked Kerr-Schild source with the tracked `:damped`
target — holds one moving hole at the resting hole's level. What makes it
delicate for two is not the layer. It is that three lines are defined
separately and must agree to a fraction of a cell: the gauge source's, the
layer target's, and the hole's own. And the target between them, the
superposition, is not a solution. Measured on the branches that built those
pieces (none of them on `main`):

| finding | where it is measured |
|---|---|
| a tracked target one cell off its hole costs `24×` in `ℋ` just outside the horizon, four cells `286×` | step I2's quadrant rows Q1 and Q2, branch `claude/step-I2-held-interior` |
| a source fed by the tracks ends every head-on row within `1 M`: the source carries the hole, the coordinates follow, the track follows the horizon, and the next piece's source carries it further | "What step B3 built and measured", [eschnett/TreeGeneralizedHarmonic.jl#18](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/18) |
| a PID law on the source's line holds the pair to `6 M`, with the line within `1.3` cells of the track and a layer residual of `5.6` that every feed shares — the superposed target's part | "The controlled source line", the same pull request |
| a regular interior left to the equations collapses within `M/4`; held by a fixed source it is as good as the layer, and ends within `4 M` when that source is `1/16` of a cell off | step I1, [eschnett/TreeGeneralizedHarmonic.jl#19](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/19); step I2, as above |

A puncture does not carry over either. BSSN's conformal factor absorbs the
puncture and its slicing freezes it; this system evolves `g_ab`, whose
spatial part grows as `1/r²` on a trumpet slice, so there is no finite
state for the grid to hold.

So the rule this design follows: **the equations never read anything derived
from a found horizon to better than a few cells, and never read an analytic
hole.** Three choices give it.

### The three parts

1. **The frame.** The grid rotates and translates rigidly with the pair, so
   each hole is at rest on it and a quasi-circular binary is quasi-stationary
   there. The mesh, the layers and the cached fits stop moving.
2. **The gauge.** The damped harmonic gauge: it holds no mass, spin,
   velocity or center.
3. **The interior.** The `:fitted` layer per hole, whose target is the
   evolved state's own continuation, so that it matches wherever its surface
   sits.

The evidence that the three fit together is the single hole **at rest**
under the damped harmonic gauge with `:fitted`: it settles, with `ℋ` and
`C_a` at `1.6×` and `1.3×` the Kerr-Schild hole's in `[3, 5)` and `1.2×` and
`1.3×` in `[5, 8)`, and the same `M_irr`. The same hole **moving** at
`v = 1/5` across the grid does not: `6–31×` at `1/32`, with a lag of
`0.49 M` by `40 M` ("What step B1 built and measured", pull request 18
above). The frame's purpose is to make every hole of a binary the first of
those two rows. The frame alone does that for the grid and not for the
gauge, which is not covariant and stays the inertial chart's: [The gauge in
the frame](#the-gauge-in-the-frame) says what that leaves, and which shift
the gauge damps is open.

### The frame

Grid coordinates are `x̄ = R(t)ᵀ (x − b(t))`, with `R` a rotation about `z`
by `Φ(t)`, `Ω = Φ̇`, and `b` the pair's center. `x` are the inertial
coordinates in which the solution is asymptotically Minkowski; they stay
the coordinates of the physics.

**The components are taken in the co-rotating basis (proposed).** The time
index stays the inertial `∂_t`; each spatial index is rotated by `Rᵀ`. This
is a basis that turns with the grid, not a rotating coordinate system:
`g_tt` and `g_ti` stay bounded far away, where a rotating chart's would grow
as `Ω² r²`. The alternative is SpEC's, inertial components on the moving
grid. It needs the first and second derivatives rotated at every point, and
it leaves a co-rotating state oscillating at `Ω` and `2Ω` at each grid
point. In the co-rotating basis that state is constant in time.

**The equations.** With `ū` the state in that basis as a function of
`(t, x̄)`,

    ∂_t ū = F(ū)  +  V̄^k ∂̄_k ū  −  Ω × ū,        V̄ = Ω × x̄ + Rᵀ ḃ       (FRAME)

- `F` is `(EXPANDED)`'s right-hand side, **unchanged**, with the grid's own
  stencils. The system is form-invariant under a rigid rotation: `η`, `√γ`
  and `α` are, and `Γ_a` is a covector under linear maps.
- `V̄^k ∂̄_k ū` is the grid's motion through the inertial coordinates. The
  kernel has `∂̄h` and `∂̄Π` at the point already.
- `Ω × ū` turns each spatial index: `(Ω × h_t·)_i` for `h_ti`, and one such
  term per index for `h_ij`; the same for `Π`. It is algebraic.
- The `∂_t g` inside `S0` is the inertial one, `β^i ∂_i h + (α/√γ) Π`, as
  now. Only the two evolution equations gain the terms.

A hole at rest on the grid has `∂_t ū = 0`, which is `∂_t u = −V·∂u` in the
inertial frame: a hole moving at `V`.

**The frame derivative rule.** Whatever is prescribed on the grid as
`Q̄(x̄, t)` in the grid's basis — a gauge source's weights and parameters,
Dirichlet data, a target — has the inertial time derivative

    ∂_t Q |_x  =  ∂_t Q̄ |_x̄  −  V̄^k ∂̄_k Q̄  +  Ω × Q̄

and that is the one the kernel must be given wherever it asks for `∂_t H`.
It is the existing rule that a gauge source's `∂_t H` must be its own time
derivative, stated for the frame.

**What reads the map.**

| reader | what changes |
|---|---|
| the right-hand side | `(FRAME)`'s two terms, behind a `Val` built with the problem; without a frame the kernel is today's bit for bit |
| the speed and the time step | the characteristic speeds shift by `−V̄` |
| initial data and the Dirichlet hook | the data sampled at `x = R x̄ + b`, its components rotated |
| a prescribed gauge source | the frame derivative rule |
| the horizon finder | nothing: `(γ_ij, K_ij)` in the grid's basis at the grid's points is valid data; `J`'s direction is rotated back for the record |
| `Ψ₄`'s modes | spheres about the grid's origin; the inertial mode is the grid's times `e^{−imΦ(t)}` |
| the symmetric domains | nothing: the mirror in `z` and the half turn about `z` commute with a rotation about `z` (the half turn needs `b = 0`). **No hole crosses the half turn's seam**, so the tuple of interiors never holds an image |
| TreeGRRMHD | the vertex metric is in the grid's basis, and the consumer needs `V̄` and `Ω` for its own advection **(open)** |

**The far zone.** `|V̄| = Ω r` exceeds the speed of light beyond `r = 1/Ω`.
The step is global and set at the holes, so level `ℓ`, of spacing `h_ℓ` out
to radius `R_ℓ`, costs nothing while `Ω R_ℓ / h_ℓ ≲ λ_f / h_f`, the finest
level's own bound. Nested levels of about a hundred cells a radius, with
`h_f = 1/32` and `λ_f ≈ 1.5`, give room to `Ω ≈ 0.4/M` **(predicted)**; a
binary's orbital rate at merger is about half that. On a cube the outer
faces then carry a normal grid velocity: half of each face is pure inflow
and half pure outflow. The frozen Dirichlet data are supplied on all of it,
which is right on the inflow half and over-specified on the outflow half
**(open**, measured by step C3**)**.

### The gauge in the frame

**The gauge conditions are not covariant, and the gauge stays the inertial
chart's (proposed).** `H_a = −Γ_a` fixes the coordinates, and `Γ_a` is a
covector only under affine maps of spacetime. The grid's map
`x = R(t) x̄ + b(t)` is not one. With `Γ̄` the comoving chart's and
`A = R̈ x̄ + b̈`,

    Γ^t = Γ̄^t,     Γ^i = R^i_j Γ̄^j + V^i Γ̄^t + (A^i − 2 (Ω × (β + V))^i) / α²

so a chart that is harmonic on the grid is not harmonic in the inertial
chart: the two differ by a centrifugal and a Coriolis term. The 3+1 pieces
differ too. The slices are the same, so `α` and `γ_ij` are; the comoving
chart's shift is `β + V`. A gauge that is "the same, only comoving" is
therefore another gauge. This design does not take it: the coordinates of
the physics stay the inertial ones, whose components are asymptotically
Minkowski, and the grid is where the points sit and which basis the
components are stored in.

**How the source is evaluated.** `DampedHarmonicSource`'s closed form is
invariant under one thing the frame uses: a rotation that does not depend
on time. `α` and `√γ` are scalars under it, `β^i` a vector, `t_a` a
covector, and the Gaussian is about the origin. At one instant the stored
components are the inertial ones in axes turned by `R(t)`, so the same
closed form on them returns `Rᵀ H`. `∂_a H_b` is the chain rule through
`∂̄h` and the inertial `∂_t g` from `Π`, both tensors in the grid's basis;
nothing is prescribed on the grid, so the frame derivative rule adds
nothing. The closed form is **not** invariant under a boost, a Galilean
shift or a rotation that depends on time (the shift changes by `V`), nor
under a scaling (`√γ` changes).

**What follows: the grid moves and the gauge does not.** A hole at rest on
the grid moves at `V` through the gauge's frame. Near it the inertial shift
is the resting hole's less `V`, and the gauge damps that toward zero. In
the continuum this is step B1's moving hole, whichever grid it is computed
on: its lag stays, and the frame removes only what the grid added.

**Which shift is damped (open).** Two candidates, both run by step C2:

- **The inertial shift**, `−μ_S γ_ai β^i/α`: SpECTRE's form as built, with
  the consequence above.
- **The comoving shift near each hole**: `β^i + V̄_A^i` under a weight a few
  `M_A` wide about hole `A`, and `β^i` elsewhere. Near each hole this is the
  gauge "the same, only comoving", and nowhere else. It is the resting row's
  exact image only for a constant `V` and with the covector's time
  component, `H_t − V̄_A^i H_i`, which the step that builds it derives and
  tests. It reads each hole's position through a weight several `M_A` wide,
  so it stays inside the rule above; it makes the gauge read the map's
  velocity for the whole run.
- **Not a candidate: the comoving shift everywhere.** It drives `β → −V`
  through the whole near zone, which turns the inertial chart into a
  rotating one inside the Gaussian.

For a constant `V` and no rotation the map is affine, and the full image of
the resting gauge is exact everywhere. Its far field is Minkowski in a
Galilean chart, `g_ti → −V_i`, not a boosted hole's, so it is a test of the
frame's terms and not a physical case.

**In harmonic gauge the question has no object.** With `H ≡ 0` nothing is
damped and nothing is evaluated on the grid: [Harmonic gauge for the
inspiral](#harmonic-gauge-for-the-inspiral-proposed), "On which grid".

**The damped harmonic gauge is rolled on (proposed).**

- **It rolls on from the composite Kerr-Schild source on grid-fixed
  lines.** Each hole's center is its grid point and its velocity is `V̄`
  there. With `Ω` steady that source is constant on the grid, and the frame
  derivative rule supplies its `∂_t H`. After `5.2 σ_g` its weight has
  underflowed, and from then on no hole parameter is in the equations.
- **Far away the data's own source stays (proposed).** The damped harmonic
  gauge is harmonic in the far field, and the Kerr-Schild Dirichlet data are
  not: `C_a` grows at the outer faces (step B1's finding). A radial weight
  keeps the roll-on's `from` source beyond `r_far`.
- **The alternative start**: the damped harmonic gauge from `t = 0` with
  `Π` from `gauge_consistent_momentum`. Then no hole-specific source exists
  at any time, and the price is the gauge's transient on the initial data
  **(open)**.

**What step B1's moving row says about the two.** It lags by the same
`0.49 M` at both spacings, which is the continuum's and so the gauge's,
while its constraint excess falls at order `5.5–10`, which is the grid's.
Under the inertial shift the frame removes the second and keeps the first:
the hole drifts on the grid until the gauge has found its own equilibrium,
and the map's control follows it. Under the comoving shift near the hole
both should go **(predicted)**.

### The interior at rest

**`:fitted` per hole, on the tracked geometry (proposed).** A `HoleSpec`
holds a `FittedSpec`; the composition, the masks and the tracks are those of
[One interior per hole](#one-interior-per-hole). On the grid each hole is
the resting `:fitted` hole, with the companion's tide in its fit.

- **The margin is fixed in `M`, not in cells (proposed).** `:fitted` does
  not converge outside the horizon at a margin fixed in cells: its target is
  off the solution by the fit's truncation, which does not fall with `h`,
  and what reaches the exterior is the grid-scale part of that mismatch,
  attenuated per cell. At a fixed physical depth the mismatch is resolved
  better and lies more cells away as `h` falls, so the excess converges
  **(predicted**; step C2 measures the order**)**.
- **No rate and no trailing side** while the holes are at rest on the grid.
- **Excision** from the separate excision project replaces this part
  without touching the other two, and a boundary at rest on the grid is its
  easiest case.

### The control acts on the map

The map follows the found centers by the PID law built for the source line
(`SourceLaw`): the angle of `c̄_2 − c̄_1` from the `x̄` axis steers `Φ`, and
the centers' mean steers `b`, with `τ = 10–20 M` **(proposed)**. `Φ` and
`b` are `C²` in time and handed to each piece about the piece's time, as a
`TrackedHole` is.

- **After the roll-on the map is not in the continuum problem.** A control
  error moves the holes on the grid and costs truncation error. Its
  tolerance is the width of the fine region and the tracked layers' reach,
  cells to an `M`, not a fraction of a cell.
- **During the roll-on the source reads the map.** That is the source
  line's loop, whose gain is at most one at zero frequency and which the
  slow law holds; it ends with the roll-on.
- `Ω(0)` is the data's orbital rate.

### The plunge and the merger

A rotation does not remove the radial motion. Each hole creeps inward along
`x̄` at `(32/5) ν (M/d)³` (the quadrupole formula; `ν = 1/4` for equal
masses): `0.0016` at `d = 10 M`, a finest cell every `10 M` at `h = 1/64`,
and `0.013` at `d = 5 M`. The level floors follow the tracks as they do
now. The plunge is a `:fitted` hole moving at up to about `0.2` for about
`20 M` **(predicted)**, with `trail_ramp = 9/10`; `:fitted` has held `0.3`
for `13 M` at `2×` the resting hole's error, with a drift of the spinning
hole's `J` ([the moving
hole](SINGULARITY_HANDLING.md#the-interiors-questions-opened-in-step-5-and-closed-through-step-8)).

The handover is [The common horizon and the
handover](#the-common-horizon-and-the-handover)'s, with one part less: there
is no source to blend into a remnant's. After the common layer has taken
over, `Ω` is held, or relaxed to zero over `20 M` **(proposed)**; the
remnant spins about `z` and is at rest on a grid that turns about `z` at
any rate.

**A scale factor**, `x = a(t) R x̄ + b` with the spacing `a h̄`, would
remove the creep and the plunge's motion too. It is noted and not planned:
the holes grow on the grid as `a` falls, so the fine region and the step
both pay for it.

### What the frame retires and what it keeps

For an orbiting pair it retires what exists to keep three lines together:
the tracked analytic target and its rates, the source fed by the tracks or
by a controlled line beyond the roll-on, the horizon offset's role in the
target, and the quadratic prediction's cell-fraction accuracy. They stay
for single holes, where the analytic layer is the accurate one, and as the
controls of steps C2 and C4.

It keeps the damped harmonic source and the roll-on, `HoleSpec` and the
composition, the tracks and the finder, the PID law, `:fitted` with its
cache, and the symmetric domains.

### What the frame costs (predicted)

- **The kernel:** two contractions with derivatives it has formed and twenty
  algebraic terms, under `2 %` of a right-hand side.
- **The step:** none, by the bound under "The far zone".
- **The mesh:** static on the grid but for the creep, so the regrid's work
  falls to the level floors' slow motion.
- **The code:** a map type, the kernel's `Val` and terms, the speed, the
  hooks and the frame derivative rule in the gauge sources; a `FittedSpec`
  in `HoleSpec`.

### The ladder

`PLAN.md`'s steps C1–C5, each a claim that can fail:

1. **The frame terms are the equations**: `(FRAME)` against automatic
   differentiation of analytic data in the grid's basis; the gauge wave and
   Kerr at the origin on a rotating grid against the same runs at rest.
2. **A comoving grid makes a moving hole a resting one**: the boosted
   `a = 0` hole at `v = 1/5` on a translating grid, damped harmonic and
   `:fitted`, against step B1's two rows. This is the run that decides the
   design.
3. **The far zone and the faces hold a wave** on a grid that rotates faster
   than light at its boundary.
4. **Two holes at rest on the grid**: the head-on pair, with the map
   following the centers.
5. **The orbit**: a quasi-circular pair at `d = 10 M`.

### What is not known

All **(open)**.

- Whether the moving hole's excess under the damped harmonic gauge is the
  grid's or the gauge's, and which shift the gauge should damp (step C2).
- A cube's outer faces under a superluminal rotation, and the far-field
  weight.
- Whether `:fitted` at a physical margin converges, and at which order.
- The handover on the common horizon, which no route has run.
- TreeGRRMHD on a moving grid.

### What the frame asks Erik to decide

1. **The basis**: co-rotating components **(proposed)**, or inertial
   components as in SpEC.
2. **The start**: the roll-on from the grid-fixed composite source
   **(proposed)**, or the damped harmonic gauge from `t = 0`.
3. **The shift the gauge damps**: the inertial one, or the comoving one
   near each hole. Step C2 runs both. Neither, if the inspiral is in
   harmonic gauge.
4. **Which route steps B4–B9 take**: this one from step C2's result on, or
   pairing (α) with the controlled source line.
5. **The interior**: `:fitted` now and excision when the excision project
   delivers **(proposed)**, or wait for excision.

## Harmonic gauge for the inspiral (proposed)

*Status: a design for review. Steps H1 and H2, the harmonic hole at rest
and moving, are measured ([What the package has measured in this
chart](#what-the-package-has-measured-in-this-chart)); nothing else in this
section is built, and every other number it leans on was measured elsewhere,
and says where.*
It is a third choice of gauge for [Binary black
holes](#binary-black-holes-proposed), beside the two under [The
gauge](#the-gauge): **no gauge source at all**. The data are superposed Kerr
holes in harmonic coordinates, `H ≡ 0` holds until shortly before the
merger, and there the damped harmonic gauge is rolled on from zero. The
interior is `:fitted` per hole. The route runs on the inertial grid or on
[the comoving one](#a-comoving-grid-frame-for-orbiting-holes-proposed), and
on the comoving grid it leaves that section's open question about the gauge
without an object. `PLAN.md`'s steps H1–H5 are the work breakdown; the first
three need nothing that is not on `main`.

### What harmonic gauge removes

[Why the tracked target is delicate](#why-the-tracked-target-is-delicate)
counts the lines that must agree to a fraction of a cell. Per route:

| | pairing (α) | damped harmonic, `:fitted` | harmonic, `:fitted` |
|---|---|---|---|
| the gauge source's line | the track, or a controlled line | the composite's, during the roll-on | none |
| the layer target's line | the track | none | none |
| the gauge's rest frame | each hole's, through its track | the inertial frame | none |

- **Nothing is prescribed.** There is no source function, so there is no
  source line and no control loop on the equations. The kernel is the one
  compiled without a source: no `Hsrc`, no closed form and no `∂_a H_b`.
  With `:fitted` the equations read no analytic hole at any time, the
  frozen data at the outer faces apart.
- **The gauge has no rest frame.** `□x^a = 0` is Lorentz covariant, so a
  hole moving at a constant velocity is the boosted solution, in this
  gauge, exactly. The damped harmonic gauge damps the shift toward zero in
  one frame, and a hole that moves through that frame lags behind the
  boosted hole's line: `0.49 M` by `40 M` at both spacings ("What step B1
  built and measured",
  [eschnett/TreeGeneralizedHarmonic.jl#18](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/pull/18)).
  Here there is no frame to lag against.
- **A moving hole has an exact reference**, `boost(Harmonic(M, a), v)`, so
  the one-hole steps read an error and not only constraints.
- **The faces agree with the gauge.** Frozen harmonic data at the outer
  faces are harmonic. Step B1's `C_a` there, `2.4·10⁻⁴` and growing, is
  Kerr-Schild data under a gauge that is harmonic far away.
- **The chart is post-Newtonian theory's.** A quasi-circular inspiral is
  quasi-stationary in harmonic coordinates in the co-rotating frame, and
  the tracks compare with post-Newtonian orbits directly **(predicted)**.

### What the package has measured in this chart

Step H1 repeated the Kerr-Schild hole's octant campaign at `a = 0` —
fourth order, an H200, to `128 M` — and step H2 its quadrant campaign, the
hole moving at `v = 1/5` and `1/2`; they are the first two rows below. The
other harmonic runs are early ones: on CPUs, to `13 M` at most, the moving
ones at `q = 2` with a thin layer (`m = 4`, `n_L = 8`). The spinning holes'
campaigns were not repeated here.

| run | what it shows | where |
|---|---|---|
| `Harmonic(1, 0)` at rest on the octant, `q = 4`, `1/24` to `1/64`, to `24 M` and `128 M` | order four outside the horizon from `1/24` (`:damped`) and `1/40` (`:fitted`); `:fitted` at `m = 16`, `n_L = 20` equal to `:damped` outside from `1/48`; stationary near the hole to `128 M`, the mass drift truncation error; Kerr-Schild's `1/24` accuracy at `1/56–1/78` | [a harmonic hole at rest](SINGULARITY_HANDLING.md#a-harmonic-hole-at-rest-step-h1-2026-11-11) |
| `boost(Harmonic(1, 0), v)` on the quadrant, `v = 1/5` to `40 M` (`1/24`–`1/48`) and `v = 1/2` to `20 M` (`1/32`, `1/40`), against resting rows | at `1/5` the resting hole from `5/4` out (`ℋ` `1.0–1.1×`, the error `1.1–1.6×` from `1/32`), no lag (`10⁻⁵ M`), the contraction to `10⁻⁶`, `:damped` order four or better; just outside the horizon a trailing excess in `ℋ`, `1.6–3.0×` for `:damped` and `12–18×` for `:fitted`, which does not converge against rest; at `1/2` `:damped` `10–46×` in `ℋ` and `3–4×` in the error on the trailing side, converging at `4–6`, and `:fitted` unstable; a fine level that follows the hole gives the capsule's numbers at `0.6–0.8×` the cost, and at `1/48` needs two H200s | [a moving harmonic hole](SINGULARITY_HANDLING.md#a-moving-harmonic-hole-step-h2-2026-11-12) |
| `Harmonic(1, 0)` at rest, `h = 5/128`, to `10 M` | `:fitted` and `:damped` agree to `5 %`; the error is the chart's, steep between the layer and the horizon, and grows slowly | [the measurement matrix](SINGULARITY_HANDLING.md#the-generic-interior-the-measurement-matrix-step-8f) |
| `Harmonic(1, 7/10)` at rest, `h = 5/256`, `:fitted`, to `10 M` | every fit valid; `M_irr` `0.30 %` low, `J` `1.0 %` high; ends at `0.5–3.5 M` at `5/128` | the same |
| the same hole boosted at `0.3` across its spin, to `13 M` | the horizon's contraction `0.9534` against `0.95394`; no failed find; with `trail_ramp = 9/10` the error is `2.05×` the resting hole's and grows at `0.11` a `M` against `0.02`; `J` drifts by `0.013` a `M` | [the moving hole](SINGULARITY_HANDLING.md#the-moving-hole-step-8), [the trailing side](SINGULARITY_HANDLING.md#the-trailing-side-step-8) |
| `Harmonic(1, 9/10)` | refused, or a fit that is not good enough | [The spinning harmonic chart](#the-spinning-harmonic-chart) |

What the rows leave open is listed under [What is not known on this
route](#what-is-not-known-on-this-route).

### What harmonic gauge costs

- **The finest level is one and a half to three times as fine.** The
  spacing at which a layer runs follows from the horizon's smallest
  coordinate radius ([the three
  radii](#the-interior-a-pointwise-damping-layer)): `2 M` for
  Kerr-Schild at `a = 0`, `M` here; `1.71 M` against `0.71 M` at
  `a = 7/10`. Measured at `a = 0` (step H1): `:damped` at `r_0 = 2/5`,
  `r_1 = 3/4` converges from `1/24`, and `:fitted` at `m = 16`, `n_L = 20`
  runs from `1/40` — `1.5–1.7×` finer than Kerr-Schild's `1/16` and
  `1/24`, and coarser than the `1/32` and `1/48` the scaled table
  predicted. But the same accuracy as Kerr-Schild's at `1/24`, shell against
  shell at the same areal radius, takes `1/56–1/65` in `ℋ` and `1/65–1/78`
  in the error relative to the fields: `2.3–3.3×` the cells per `M`, not
  `2×`. In a binary's units that is `1/48` and `1/80` for a layer to run,
  and `1/112–1/156` at the holes for Kerr-Schild's accuracy at `1/48`.
- **With spin the room is the equator's for `:fitted` and the pole's for
  `:damped`.** The offset surface lies between the horizon's equator at `M`
  and the disk at `a`: `M/2` at `a = 1/2` and `0.3 M` at `a = 7/10`, against
  `M` less the core at `a = 0`. A spherical layer lies between the disk and
  the pole `√(M² − a²)`: `0.37 M` at `a = 1/2` and `0.014 M` at `7/10`.
  Measured (step H3, [a spinning harmonic hole](SINGULARITY_HANDLING.md#a-spinning-harmonic-hole-step-h3-2026-11-12-to-2026-11-13)): at `a = 1/2`
  `:damped` converges from `1/48` and `:fitted` from `1/64`, twice and `1.6×`
  as fine as at `a = 0`, with `J`'s drift at the Kerr-Schild rows' level from
  `1/64`. `a = 7/10` runs only as `:fitted`, from `1/128` on two H200s, and
  its error outside the horizon grows from `10 M`: `1.7` levels finer than
  `:fitted` at `a = 0`, against the one predicted. A spinning hole moving across its spin
  is `276 M` points at `1/96` on the bitant, four H200s at `529 s/M`; at
  `a = 7/10` and `1/128` it would be `654 M`, more than one node of eight
  H200s **(estimated)**.
- **Outside the horizon the spacing buys accuracy too (measured; it was
  predicted to buy nothing).** The prediction was that the chart's fields
  vary there on the scale `M`, as Kerr-Schild's do, and that only the room
  inside a horizon of half the radius needs the cells. At the same `h` the
  error in `[2, 4)` is `90–120×` the Kerr-Schild hole's in `[3, 5)`, the same
  areal radii, and `50–65×` relative to the fields, which are `1.8×`
  Kerr-Schild's there and `10×` just outside the horizon. They vary on the
  harmonic radius, not the areal one.
- **In points and steps:** one more level within about `M_A` of each hole,
  and a step that halves with the spacing, since the step is global. The
  chart's coordinate speeds cost nothing: the step is `h/(2 · 1.71)` against
  Kerr-Schild's `h/(2 · 1.75)`, and a point and step costs `21–27 ns` on an
  H200 in both. On the octant at rest the harmonic hole costs `112 s/M` at
  `1/48` and `309 s/M` at `1/64` (`32` and `75 M` points), against
  Kerr-Schild's `59 s/M` at `1/24`: `5.2×` at the spacing that matches its
  `ℋ` (measured, step H1). For the binary **(predicted)**: with `s` the
  finest level's share of a Kerr-Schild run's points, a level that follows
  the holes has as many points as the one below it, and the cost of one
  level more is `2 (1 + s)` times that run's: `2–4×`. On a capsule along a
  path the new level has twice the points of the one below it,
  `2 (1 + 2s)`, up to `6×`, and a ring along the orbit is longer than any
  path run so far. That is for one level more. At Kerr-Schild's accuracy
  step H1 measured `2.3–3.3×` in spacing, `1.2–1.7` levels more, and its
  `5.2×` on the octant at rest.
- **So the fine level must stay with the holes**: by the tracked floors of
  [Refinement for two holes](#refinement-for-two-holes) on the inertial
  grid, or because the holes rest on the comoving grid. Measured for one
  moving hole (step H2): the indicator's mesh regridded every chunk with its
  floor on the tracked horizon gives the frozen capsule's numbers near the
  hole at `0.81×` its cost at `1/24` and `0.61×` at `1/40`; at `1/48` it is
  `70–95 M` points and needs two H200s.
- **The spin is limited to `a ≤ 7/10`.** The chart's singular set is the
  equatorial disk of radius `a`, and the horizon's polar radius is
  `√(M² − a²)`; the two meet at `a = M/√2`. SpEC's superposed harmonic data
  have the same limit (Varma, Scheel and Pfeiffer 2018). Its extension to
  `0.8–0.9` is a chart that is harmonic in time only (Ma, Giesler, Scheel
  and Varma 2021), which has a spatial gauge source and so is not this
  route.

### The data

**Superposed boosted harmonic holes (proposed).** `g = η + Σ_A (g_A − η)`
with each `g_A` a boosted, translated `Harmonic(M_A, a_A)`: item 1 of
[Initial data](#initial-data) with another chart.

- **`Π` comes from `C_a = 0`.** Each `g_A` has `Γ_a = 0` and the sum does
  not, at second order in the fields. `gauge_consistent_momentum` with
  `H = 0` chooses `∂_t α` and `∂_t β^i` so that `Γ_a = 0` at `t = 0`.
- **The Einstein constraints** are violated where the fields overlap, as
  the superposed Kerr-Schild data's are, and the constraint damping removes
  that as junk. SpEC reports less junk from harmonic free data than from
  Kerr-Schild's, on solved data.
- **Solved data (open).** XCTS's free data are superposed Kerr-Schild.
  Harmonic free data are a change there. Any solved lapse and shift can
  start a harmonic evolution, since `Γ_a = 0` fixes only their time
  derivatives; whether the result is as quiet as the superposed chart's is
  not known.

### The merger

**Pure harmonic gauge is not expected to survive the merger (predicted).**
Between merging horizons the lapse collapses and `√γ/α` grows without a
bound; the damped harmonic gauge exists to prevent that (Lindblom and
Szilágyi 2009; Szilágyi, Lindblom and Scheel 2009). Whether a
finite-difference code with a relaxed interior meets the same failure, and
when, is what step H4 measures first.

**The damped harmonic gauge is rolled on before it (proposed).**
`H = (1 − f(t)) H_DH(g)` with `f = 1` up to `t_0` and
`f = e^{−((t − t_0)/σ_g)⁴}` after it: the roll-on of [The gauge](#the-gauge)
with nothing to roll on from.

- **It still holds no line.** `H_DH` reads the state and a Gaussian about
  the origin, where an equal-mass pair merges.
- **It starts exact.** At `t_0` the source and its first three time
  derivatives are zero.
- **`t_0` is an event**: the first row at which the tracks' coordinate
  separation is below `d_g`. It is read from found horizons to the accuracy
  of a row, which is inside the rule of [Why the tracked target is
  delicate](#why-the-tracked-target-is-delicate). It belongs to the run
  state and the checkpoint.
- **`d_g = 6 M` and `σ_g = 15 M` (proposed).** A quasi-circular equal-mass
  pair at `d = 6 M` has about `100 M` to its merger (the quadrupole
  formula), and the roll-on is within `1 %` of complete after `1.5 σ_g`.
  A head-on pair from rest at `d = 8 M` falls in about `25 M` (Newtonian),
  so there `t_0 = 0`.
- **The gauge's width `σ_r`** is small against the box, so that the faces
  stay harmonic.
- **What the roll-on brings back.** For its last tens of `M` the holes move
  at `0.2–0.4` through the damped harmonic gauge's frame, which is step
  B1's moving hole. There its excess over the resting hole fell at order
  `5.5–10` from `1/24` to `1/32`; this route's holes have twice those cells
  across their horizons **(predicted** to be enough; not measured**)**.
- **The remnant is at rest in that frame**, which is step B1's resting
  hole: `ℋ` and `C_a` at `1.2–1.6×` the Kerr-Schild hole's, flat from
  `20 M`.

The handover is [The common horizon and the
handover](#the-common-horizon-and-the-handover)'s, with no source to blend.

**The alternative: the roll-on at `t_0 = 0`** on the same data, which is
close to how SpEC evolves such data. It needs no event. It keeps the
gauge's rest frame for the whole inspiral, and so step B1's moving hole for
the whole run and not for its last orbit.

### On which grid

The route needs no frame. What a frame adds is unchanged, and what it asks
is less:

- **On the inertial grid** each hole is a moving `:fitted` hole for the
  whole run: the layer takes its target's rate, has a trailing side, and
  the fine level follows the track. Step H2 measured it at `q = 4`: at
  `v = 1/5`, a pair's speed at `d ≈ 6 M`, the moving hole is the resting
  one from `5/4 M` out, with `:fitted`'s `ℋ` just outside the horizon
  `12–18×` the resting hole's on the trailing side; at `v = 1/2` `:fitted`
  does not hold ([a moving harmonic hole](SINGULARITY_HANDLING.md#a-moving-harmonic-hole-step-h2-2026-11-12)).
  With spin (step H3, [a spinning harmonic hole](SINGULARITY_HANDLING.md#a-spinning-harmonic-hole-step-h3-2026-11-12-to-2026-11-13)) `:damped` at `1/96` is the resting hole within
  `2×` in the error at `a = 1/2` and `v = 1/5`, along the spin and across it,
  and `:fitted` at `1/64` is not (`1.6–350×`).
- **On the comoving grid** each hole is the resting `:fitted` hole, and the
  fine level is static but for the creep. `(FRAME)`'s terms are kinematics
  for any gauge.
- **[The gauge in the frame](#the-gauge-in-the-frame) has nothing to
  decide.** `H ≡ 0` is the inertial chart's condition and nothing is
  evaluated on the grid, so no shift is damped until the roll-on. From then
  on the gauge damps the inertial shift, and its rest frame is the
  remnant's.
- **Step C2 becomes a test of the frame's terms alone**: the boosted
  harmonic hole on a grid translating with it has an exact reference, and
  must be this route's resting row to truncation.

Steps H1–H3 run on the inertial grid, and step H2's ratio of the moving hole
to the resting one at `40 M` was to decide whether the pair needs the frame.
**Its numbers favour the inertial grid with a following fine level for the
inspiral** to the roll-on at `d_g = 6 M`, where each hole moves at about
`1/5`. They leave the plunge, at `0.3–0.5`, without a `:fitted` hole on that
grid. **(Open, Erik's decision:)** the inertial grid throughout, the comoving
grid throughout, or the inertial grid with `:fitted` measured first between
`v = 1/5` and `1/2`.

### The ladder of this route

`PLAN.md`'s steps H1–H5, each a claim that can fail:

1. **A harmonic hole at rest runs as the Kerr-Schild hole does**: fourth
   order outside the horizon and stationary to `128 M`, at twice the cells
   per `M`. *Measured (step H1):* it does, with `:fitted` equal to `:damped`
   outside the horizon from `1/48`, but at `2.3–3.3×` the cells per `M` for
   the same accuracy, not `2×`.
2. **A moving harmonic hole is a resting one**: within `2×` the resting
   row to `40 M` at `v = 1/5`, which is step B1's moving row without its
   gauge. *Measured (step H2):* from `5/4 M` out it is, for both variants,
   with no lag and the contraction to `10⁻⁶`; just outside the horizon the
   trailing side is `1.6–3.0×` in `ℋ` for `:damped` and `12–18×` for
   `:fitted`. At `v = 1/2` `:damped` holds at `3–46×` and `:fitted` does not.
3. **Spin up to `a = 7/10`**, at rest, along the motion and across it, and
   whether the early rows' drift of `J` is still there at fourth order.
   *Measured (step H3):* at `a = 1/2` the hole at rest is stationary with
   order four from `1/48` and `J` at the Kerr-Schild rows' level, and
   `:fitted` equals `:damped` with its margin fixed in `M`. Moving at
   `v = 1/5`, `:damped` at `1/96` is within `2×` in the error along the spin
   and across it, with `ℋ` `4.3×` just outside the horizon across it and an
   outgoing pulse; every excess converges at order `7–11`. `:fitted` moving
   is `1.6–350×` at `1/64`. The drift of `J` is resolution: `6000×` below the
   early rows' at `1/64` and converging at order nine. `a = 7/10` runs at
   rest at `1/128` on two H200s, not stationary by `24 M`; its moving rows
   were priced, not run.
4. **Two holes with no source**: the head-on pair in pure harmonic gauge to
   wherever it ends, then with the roll-on through the merger.
5. **The orbit**, on the grid chosen from step H2's ratio.

### What is not known on this route

All **(open)**, except where answered.

- *Answered (step H1):* a harmonic hole at rest is stationary for `128 M`
  where the hole is, as the Kerr-Schild hole is; its mass drifts at a falling
  rate, as truncation error at order four.
- *Answered for `a = 1/2` with `:damped` (step H3):* the moving spinning
  hole's drift of `J` is resolution — across the spin at `v = 3/10` it is
  `2.1·10⁻⁶/M` at `1/64` and `5·10⁻⁸` at `1/96`, order nine. Open: whether
  `:fitted` holds a moving spinning hole at a finer spacing or another
  margin (at `1/64` it does not); the pulse that leaves the moving hole at
  about `4 M`, converging at order `7–10`; the error's ratio across the spin
  rising over `20 M`; and whether `a = 7/10`'s growth at rest saturates.
- *Answered for `a = 0` at `v = 1/5` (step H2):* a moving harmonic hole has
  no lag in this gauge and is the resting hole outside `5/4 M`. Open:
  `:fitted`'s trailing excess just outside the horizon, which does not
  converge against the resting row; the error's late rise outside the
  horizon over `32–40 M` (`1.3–1.8×`, converging at order five); and the
  speed between `1/5` and `1/2` at which `:fitted` stops holding.
- *Answered for the harmonic hole at rest (step H1):* `:fitted` converges
  outside the horizon at a margin fixed in `M` — faster than order four
  while the margin gains cells, then at four. Open on any other chart and
  for a moving hole.
- How pure harmonic gauge fails at a merger in this code, and whether the
  roll-on's `d_g` and `σ_g` leave it time.
- Whether superposed data without a solve settle in a gauge that has no
  damping of its own.
- Reflections of the gauge's own waves from frozen Dirichlet faces, which
  is [The outer boundary and the wave
  zone](#the-outer-boundary-and-the-wave-zone)'s question on every route.

### What this route asks Erik to decide

1. **The gauge of the inspiral**: harmonic **(proposed)**, the damped
   harmonic gauge from `t = 0` on the same data, or pairing (α).
2. **The merger**: the roll-on at `d_g` **(proposed)**, decided again from
   step H4's pure harmonic run.
3. **The spin**: `a ≤ 7/10` for G7 and G8 on this route **(proposed)**,
   with higher spins left to a chart that has a source. Step H3 measured
   `a = 1/2` holding with `:damped` and `a = 7/10` running at rest only at
   `1/128`, where it is not stationary by `24 M`.
4. **The grid**: step H2 has reported ([On which grid](#on-which-grid)):
   the inertial grid with a following fine level holds the inspiral to
   `v = 1/5`, and `:fitted` on it fails at `v = 1/2`.
5. **The order**: steps H1–H3 now, since they need only `main`
   **(proposed)**, or after the binary steps B1–B3 have landed.

## Possible extensions

What separates the proof of concept from a production code, listed with
the design note that would start each:

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
  the sonic surface applies unchanged. **Priced:** a fill that keeps the `∂²` stencil's order must reproduce
  the Taylor polynomial to degree `q + 1`, so a block-local fill needs a
  `3G` halo — stored volume ×4–7 at `N = 8`, where the vertex invariant
  `N ≥ 6G + 2` refuses it, ×1.8–2.7 at `N = 32` — or a second ghost
  exchange per evaluation, and degree `q + 1` amplifies grid-scale noise
  15–320× at depth 2. Excised leaves are block-granular and serve static
  holes only.
- **The damped harmonic gauge driver** (Lindblom–Szilágyi 2009;
  Szilágyi–Lindblom–Scheel 2009): `H_a` algebraic in `g` and
  `log(√γ/α)`, its gradient by the chain rule through `∂_a g`, no extra
  variables, no extra exchange; the fix for the gauge drift G4 records,
  and the prerequisite for binaries. **(Designed, proposed,** under [The
  gauge](#the-gauge), rolled on from a tracked Kerr-Schild source, and
  paired with the interior there.**)**
- **A radiative outer boundary** for solutions not known at the
  boundary: the standard `1/r` outgoing extrapolation into the ghosts
  along the ray, needing a device-capable hook that reads the block's
  interior (TreeAMR's `CODE.md` names the gap). (TreeAMR's
  relative hook, `CellBoundary(Relative(f))`, is that hook for a
  zero-gradient deviation from frozen data; [The outer
  boundary](#the-outer-boundary-and-the-wave-zone) proposes measuring it
  against a far frozen boundary.)
- **Time-dependent prescribed gauge sources** evaluated in the kernel,
  which would admit boosted Kerr-Schild; priced at about one RHS per
  evaluation by GHSO2's `gauge_source_grad` timing.
- **Binaries**: superposed Kerr-Schild as the approximate data, then
  constraint-satisfying data from `XCTS` / `SolveConstraints` through
  the same `(h, Π)` callback on the host; needs excision or an interior
  layer per hole, the gauge driver and the radiative boundary.
  **(Designed, proposed,** under [Binary black
  holes](#binary-black-holes-proposed): an interior layer per
  hole, a handover to a common one at merger, and XCTS data through a file
  rather than a callback.**)**
- **Gravitational-wave extraction**: `Ψ_4` on extraction spheres, modes
  through spin-weighted harmonics, needing the interpolation of
  prerequisite 1. **(Designed, proposed,** under [The
  waveform](#the-waveform): `Ψ_4` at the spheres' points from TreeAMR
  0.2.6's second-derivative interpolation, the modes by
  `AbstractSphericalHarmonics`' spin-weighted transform.**)**
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
  the kernel is the first thing to change. **(Investigated**, in [The
  right-hand side on an H200](#the-right-hand-side-on-an-h200).**)**
  - Shared-memory staging is slower than L1.
  - The source's register schedule is the lever, and a hand-written spelling
    suffices.
  - The kernel comes first; then TreeAMR's scatter and transfer kernels; the fused
    stage update last, at 10–15 % of a step.
- **SIMD across grid points on the CPU** (implemented,
  [The right-hand side on a CPU](#the-right-hand-side-on-a-cpu)):
  the package's own kernel on four SIMD.jl lanes is 2.0–2.4× the scalar one on
  Symmetry's EPYC nodes, the same `du` to roundoff, and the stencils without their
  zero weights a further 4–13 % (not implemented). It is 1.3× a step on its own: the
  ghost fill and the stage arithmetic are what is left, and the fill is TreeAMR's.
- **A truncation-error indicator** in place of Löhner's — Richardson
  between a block and its restriction, which TreeAMR's operators make
  cheap — if the calibration in G4 shows Löhner's thresholds to be
  fragile across resolutions.
- **A hyperboloidal outer layer**, which this author's electrodynamics
  packages rehearse and which would retire the outer boundary question.
- **Matter**, through the source slot: TreeHydro's scheme on this
  spacetime is the GRHD code the two packages together rehearse.

## Open questions

**The interior's questions** — the spherical core against harmonic Kerr at
`a = 9/10`, the generic interior, the price of that chart, the fitted target and the
moving layer's trailing side, all closed or measured — are in
[`SINGULARITY_HANDLING.md`](SINGULARITY_HANDLING.md#the-interiors-questions-opened-in-step-5-and-closed-through-step-8).

Still proposed, to be confirmed or amended by the milestones that
first touch them:

1. **The damping layer `(INTERIOR)` as the default interior**, against
   the hard paste and the pure mask (G4, G5). The argument for it is
   under [The interior](#the-interior-a-pointwise-damping-layer); the
   numbers are not yet.

   **Inconsistent:** item 1 says the damping layer's numbers "are not yet";
   this file's "What the single holes hand over" (under Binary black holes)
   reports the moving `:damped` layer measured and converging at order four,
   and `SINGULARITY_HANDLING.md` holds the interior variants' runs.

2. **The Löhner indicator on `h` with a global floor**, with the mask,
   the level floor around the horizon and the ceiling at the boundary
   (G4): whether calibrated thresholds carry from the static to the
   moving hole, and whether the floor ever binds.
3. **The streaming-order fused kernel** against the stencil/algebra
   split: answered, the fused kernel stays ([The right-hand side on an
   H200](#the-right-hand-side-on-an-h200); the
   question's [history](HISTORY.md#open-questions)).
4. **In-kernel evaluation of `SpacetimeMetrics`** on the H200 (G6); the
   structure does not depend on it, the cost does.
5. **The defaults** — `q = 4`, `N = 32`, `cfl = 1/4`, `ε_KO = 0.5`,
   `γ0 = 1/M` near the hole, `m = 8`, a layer of `2(G + 1)` spacings,
   `ρ_max = 4/M`, the indicator's
   thresholds — are starting values for G4–G6 to confirm or move. Three
   of them survived G2 on flat space and on a gauge wave: `cfl = 1/4` (no
   run needed less), `ε_KO = 0.5` (the noise test, and no order lost) and
   `q = 4` as the development order (`q = 2, 6, 8` all run, at 0.5×, 1.3×
   and 1.9× the cost of `q = 4`). None of that is yet a statement about a hole. `m` is
   split into a stencil margin and a leakage margin; `m = 8` attenuates
   grid-scale content from `r_1` by `e^{−2.9}` to `e^{−4.6}` on the fixture
   (see [The interior](#the-interior-a-pointwise-damping-layer)), and stays
   the default: a smooth layer's wrong target leaves the exterior what the
   exact one leaves. `ρ_max = 4/M` with a ramp of `4G` cells, measured for an
   inexact target and for the analytic one, where it cuts the `50 M` error
   sixfold, is the code's default for every variant, the analytic `:damped`
   layer included; the grid rate `ρ_max · dt = 1` is the option
   `ρ_max_factor`. The ramp is not part of that decision: it is the case's
   `r_0`, `r_1` and `ρ_ramp`, and the suite's fixture keeps step 5's layer, a
   ramp of `4.8` cells, on which the exact target was measured at `4/M` to
   `50 M` ([history](HISTORY.md#open-questions)).

**Gauge sources that know less about the hole.** The sampled `H_a(x)` is taken
from the background ([Gauge and constraint
damping](#gauge-and-constraint-damping)), so it knows the hole's mass, spin,
position and velocity, and with it a moving Kerr-Schild hole is refused because its
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

**Built**: the source above, `KerrSchildSource` behind
`Val(:algebraic)` — see [Gauge and constraint
damping](#gauge-and-constraint-damping) — and measured first on the static
hole on the octant ([single holes on the
octant](SINGULARITY_HANDLING.md#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07)).

The boosted Kerr-Schild hole at `a = 0`, `v = 1/5`
([history](HISTORY.md#open-questions)) was not compared with G5's rows, which differ in chart, speed (`0.3`),
order (`q = 2`) and spin. A moving spinning Kerr-Schild hole, `a = 9/10`
above all, is still open.

**Inconsistent:** the sentence above calls a moving spinning Kerr-Schild
hole, `a = 9/10` above all, still open; "What the single holes hand over"
(under Binary black holes, this file) reports the aligned campaign at
`a ≤ 9/10` and `v ≤ 1/2` and the bitant campaign across the spin, stable and
converging at order four outside the horizon.
