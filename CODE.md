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
with nothing in that decision knowing about blocks or levels; the
proof-of-concept target is a **single black hole with nonzero boost and
spin**; time integration through OrdinaryDiffEq; the analysis quantities
(constraints, horizon location, area, mass and spin) are part of the
deliverable; refinement is driven by an **error indicator**, not by
prescribed spheres; `Float64` on Symmetry's H200 is the device
requirement and `Float32` on a device is desirable, not required; no
checkpointing; GPU kernel efficiency is a later research project; the
inherited documents live in `notes/`.

## Goals

- **A proof of concept of a black-hole GH code on TreeAMR.** A boosted,
  spinning Kerr black hole crossing the domain, its interior driven to
  the analytic solution, refinement following it because an error
  indicator says so, the constraints and the horizon's location, area,
  mass and spin recorded as time series; correct in `Float64`, running
  in `Float64` on Symmetry's H200, bit-identical across thread counts.
  Milestones G4–G6 define the proof. What a production code adds on top
  — binaries, excision, a dynamical gauge, radiative boundaries,
  checkpointing, an efficient GPU kernel — is under
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
- **No singularity handling** (decided). Nothing is excised. Where a
  run has a black hole, the right-hand side inside the horizon is
  modified point by point: driven toward the analytic solution in a
  layer well inside the horizon, switched off around the singularity
  (see [The interior](#the-interior-a-pointwise-damping-layer)). This
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
  under [Open questions](#open-questions).
- **Analytic initial data, boundary data and gauge sources only.**
  Everything `SpacetimeMetrics` provides: Minkowski, the gauge wave,
  shifted Minkowski, Kerr-Schild, Kerr in harmonic coordinates, and
  translations, rotations and boosts of these. Constraint-satisfying
  binary data from an elliptic solver is an extension.
- **No checkpoint and restart** (decided in review). A proof of concept
  runs to completion; checkpointing arrives with TreeAMR's M9 or as a
  production extension.
- **No mesh machinery.** Trees, ghosts, interpolation, regrid transfer
  and reductions are TreeAMR's. The one thing this package writes that
  arguably belongs upstream — point interpolation from a field set, for
  the horizon finder — is written as a stopgap and listed under
  [Upstream prerequisites](#upstream-prerequisites).
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
RK4 from OrdinaryDiffEq, the chunked regrid-and-restart driver with
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
| SAT penalties for boundaries; excision faces | ghost-filling boundary hooks; a pointwise damping layer inside the horizon instead of excision |
| `ε μ⁻⁵ D⁶` dissipation normalised by the SBP spectral radius | the standard Kreiss–Oliger operator of order `q + 2` on a uniform block |
| one mesh for the whole run | error-driven regridding that follows the hole; a fresh problem per chunk |
| Tsit5 / Vern6–9 matched to the element order; a native stepper | RK4 from OrdinaryDiffEq, fixed step (decided) |
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
none. `ε` is a case parameter.

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
with it is under [Measured results](#measured-results), step 8a.

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

### The time step

    dt = cfl · minimum_spacing(forest) / λ_max,
    λ_max = max over owned points of  α √(tr γ^{ij}) + |β|

GHSO2's conservative bound on the coordinate characteristic speed, taken
once per chunk from a speed slot in `diag` (a kernel writes it,
`block_mapreduce(max)` reduces it) and re-checked at the chunk's end as
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

### The interior: a pointwise damping layer

**No excision** (decided). Inside the horizon the solution is not left
to the Einstein equations alone: in a layer well inside the horizon it
is *driven to the analytic solution*, and around the singularity the
evolution is *switched off*. Both are decisions made **point by point**,
as functions of the distance `r = |x − c(t)|` to the hole's analytic
center `c(t) = c_0 + v t`, and neither knows anything about blocks,
ghost zones or refinement levels (decided in review). The right-hand
side at every point is

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
under [Measured results](#measured-results), step 8a). On the step-5 hole at
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

**The profiles and their parameters.** `w` and `ρ` are `C²` smoothstep
polynomials of `r`, `isbits` closures over `(c(t), r_0, r_1, ρ_max)` and
the ramp widths, evaluated per point in the kernel. `ρ_max` is bounded
by the explicit integrator: RK4 is stable on the negative real axis to
about `2.8/dt`, and `ρ_max · dt = 1` **(proposed)** relaxes by a factor
`e` per step, which is as strong as it needs to be; the driver derives
`ρ_max` from `dt` each chunk.

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
projection, from the stage limiter, is the third. Both limiters are
`solve` keywords (`stage_limiter`, `step_limiter`), not `RK4(; …)`
arguments, whose constructor form is deprecated. The projection also runs
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
results](#the-range-projection-step-8b): step 5's `N = 6` `:damped` and
`N = 8` `:pasted`.

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
        solve(ODEProblem(gh_rhs!, u, (t, stop), p), RK4(); dt, adaptive = false)
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
  is `1/dt` and `dt` is measured per chunk, so the interior the kernel
  closes over is rebuilt at the top of every chunk (`with_interior`),
  which shares the field sets, the schedule and the **sampled gauge
  source** rather than rebuilding the problem — re-sampling `H_a` is the
  most expensive setup phase there is and nothing about a new `ρ_max`
  invalidates it.
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

## Time integration

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
on the negative real axis, and `ρ_max · dt = 1` keeps it well inside;
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
| interior residual | `|u − u_exact|` inside the layer, L∞, the layer's own health | every chunk |
| mesh statistics | leaf count per level, finest spacing, the indicator's `τ_max`, the refinement centroid against the analytic center | every chunk |
| range projection | `bounds_hits` (points moved, summed over every stage-limiter call of the chunk), `bounds_nonfinite` (of those, points with a non-finite component), `bounds_r_max` (the outermost radius it fired at, `−1` where it did not); `nothing` for a case without bounds (added in step 8b) | every chunk |
| validity monitor | over the layer `r_0 ≤ r < r_1` and over the `G` points outside it: `min_detγ`, `min_α` (the *signed* lapse, negative where `g^{tt} > 0`), `max_h`, `max_Π` (the largest component magnitudes) — `min_detγ_layer` … `max_Π_shell` (added in step 8b) | every chunk |

**Constraints.** Both kernels mask the interior `r < r_1` and write zero
inside it; the modified region is not a numerical solution. Norms are
`block_mapreduce` partials weighted by each block's `h³`, combined in
block order, so they are bit-identical across thread counts.

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
[Upstream prerequisites](#upstream-prerequisites). The horizon lies
outside the layer by the margin `m`, and the provider throws if a query
point's interpolation footprint reaches `r_1`. Each find is seeded with
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
few finest spacings has found a bug, not a feature to track.

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
  of the grid's own quadrature.
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
  compared to.

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
  [Possible extensions](#possible-extensions).

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
2. **A device reduce-to-scalar** (TreeAMR `TODO.md`) would let the
   per-chunk norms stay on the device; today `block_mapreduce` copies
   one value per block back, which is fine at chunk frequency.
3. **MPI (M7) and I/O (M9)** are on TreeAMR's roadmap and this package
   is written so they arrive transparently: no host loop over blocks
   assumes all blocks are local, and every reduction is a
   `block_mapreduce` or a `firing_boxes`.
4. **A launch-configuration knob on `map_blocks!`** (G6): today it
   launches with KernelAbstractions' default workgroup, and GHAccel
   measured the workgroup shape of a 3D stencil kernel as worth about
   10 % of peak. A `workgroupsize` keyword passed through to the launch
   is the whole request.

Two further items are *not* needed for the proof of concept and are
recorded with the extensions that would need them: an interior-reading
device boundary hook (radiative boundaries, excision) and excised leaves
(excision).

## File layout

| file | contents |
|---|---|
| `notes/` | verbatim copies of the inherited documents, with provenance; see `notes/README.md` |
| `src/TreeGeneralizedHarmonic.jl` | module shell: `using`s, exports, includes |
| `src/precision.jl`, `src/device.jl` | copied from TreeWave, with TreeHydro's `hostcopy!` split so that the copying half is exercised host to host (amended in step 0) |
| `src/pointwise.jl` | GHSO2's pointwise algebra (ported from `notes/pointwise-ghso2.jl`), plus the expanded form's coefficient derivatives `metric_derivatives`, the assembled `gh_node_rhs_expanded`, and `gh_node_source` (all added in step 1), and `metric_derivatives_along` — the same chain rule along **one** direction, returning `∂√γ` and `∂γ^{jk}` rather than the assembled `∂(α√γγ^{jk})`, which is what the constraint monitors need along *time* (added in step 4) — and `_pairindex`, the packed slot of a symmetric index pair, so that the file has one packing convention used on two index pairs; `SVector{10}` state, `SMatrix{4,4}` tensors. `gh_node_source` is a **second copy** of the reduced source and the damping, written out of `gh_node_rhs` character for character rather than factored out of it: the port stays diffable against `notes/pointwise-ghso2.jl`, which is what makes it the validated reference, and `test/pointwise_identity_tests.jl` asserts the copy still matches it — to roundoff, because two spellings of one expression are not bit-identical (see "Measured results") |
| `src/stencils.jl` | rational finite-difference and Kreiss–Oliger weights at order `q` (added in step 2): `derivative_weights`, `dissipation_weights`, both `@generated` over `(T, Val(q), Val(m))` and returning `SVector`s of `T` for unit spacing; `lagrange_derivative_weights` and the two `rational_*` constructors behind them, exposed unexported so that the exactness claims can be asserted in `Rational` rather than through a tolerance; `dissipation_rank(Val(q)) = Val(q/2 + 1)`, one spelling of `2r = q + 2` **(proposed in step 2)**; the host-side `apply_stencil` and `apply_mixed_stencil`, which are the reference contractions the tests measure with and the definitions step 3's streaming kernel has to agree with |
| `src/evolution.jl` | the fused RHS kernel in streaming order (added in step 3), the linear-index stencil contractions it evaluates, `GHProblem` with the **five** `Val`s and the per-chunk geometry, `gh_rhs!`, the speed kernel, `max_speed`, `gh_dt`, and `convergence_rate` — TreeWave's, in the file TreeWave keeps it in. Step 5 split the streaming body out of the kernel into `gh_rhs_at_point`, an `@inline` plain function, because `F` must not be evaluated in the frozen core and **KernelAbstractions refuses a `return` statement anywhere in a kernel body** — so the core branch cannot be an early exit and has to be an `if` around the whole computation; and added `gh_paste_kernel!` with `gh_step_limiter!` and `paste_interior!`, the `:pasted` variant's one write to the state |
| `src/gauge.jl` | sampling prescribed sources into `Hsrc` and reading them back at a point (`gauge_at`, the kernel's half of the packing); `isharmonic` as a table over the background types and `isstatic` as an exact measurement, with the reason each is what it is (added in step 3) |
| `src/boundaries.jl` | the time-dependent Dirichlet hook |
| `src/bounds.jl` | the range projection (added in step 8b): `StateBounds` and the proposed `default_bounds`/`default_gate`, `check_bounds_gate`, the pointwise `bounds_project` over an explicit-scalar ADM split and a Jacobi `sym_eigen3`, `gh_bounds_kernel!`, `BoundsAccounting`, `apply_bounds!` and `gh_stage_limiter!`; the validity monitor (`state_validity`, `validity_rows`); and `evolved_nonfinite`, the masked finiteness check. Included after `interior.jl` and before `initialdata.jl`, whose `GHCase` carries a `StateBounds` |
| `src/interior.jl` | the profiles `w(r)`, `ρ(r)`, the core rule, the radius checks, the masks; added in step 5. Also `HoleCenter` — `c(t) = c₀ + v t` as two vectors and a line, which is what "the center is a function of `t`, never a mutated field" means as code — the horizon's analytic coordinate radii, and `layer_spacing`, the coarsest spacing among the blocks the sphere `r_1` passes through, which is the one number in the file that looks at a mesh (and looks at it only to *assert*). The `:pasted` limiter is in `evolution.jl` instead **(amended in step 5)**, beside the kernel it launches and the state layout it writes |
| `src/initialdata.jl` | backgrounds, `GHCase` and the case constructors (here rather than in `driver.jl`, amended in step 3), the forest builders — uniform, with one root block refined for the frozen two-level hierarchy the interface study needs (`refined = true`, added in step 4), or `hole_forest`'s nested shells around a hole (added in step 5, **here rather than in `interior.jl`**, since a forest builder belongs with the other forest builder) — the `(h, Π)` callback with the core rule, the `SpacetimeMetrics` index conversion and nowhere else |
| `src/refinement.jl` | the Löhner indicator with its global floor, the mask, the level floor and ceiling, the four marks, the buffer; TreeWave's `refinement.jl` ported |
| `src/constraints.jl` | the gauge-constraint kernel (state and first derivatives) and the ADM one (every second derivative of `g_ab`, the `∂_t` blocks from the evolution equations, the four-dimensional Ricci tensor assembled rather than reduced), the masks they take — `AllPoints` and the `is_evolved` predicate step 5's interior adds a method to — `masked_norms` and `constraint_norms`, and `adm_constraints_at_node`, the pointwise curvature assembly the tests check against `ddmetric` (added in step 4) |
| `src/horizon.jl` | the interpolating ADM provider for `ApparentHorizonFinder`; location, shape, area, `M_irr`, `J`, `M_ch`. Added in step 7, in the order the numbers are produced: `locate_block` and `interpolate`/`interpolate_grad` (the stopgap of [Upstream prerequisites](#upstream-prerequisites), item 1, with the footprint guard that refuses a query reaching inside `r_1`), `GHADMProvider` (batched, `Float64` out whatever the run computes in, with a one-entry cache keyed on the identity of the query array because `KorzynskiSpin.surface_geometry` asks for `γ` and `K` in two calls with the same points), `find_gh_horizon`, and `Horizon` — the cadence and resolution the case carries |
| `src/driver.jl` | `evolve!`, the analysis record per chunk, `observer`, `check_cfl`, `horizon_shell`, `forest_levels`, and `discrete_gradient_momentum!` — GHSO2's `Π` post-pass, which lives here because it runs once on the initial data and is the driver's option, not the initial data's (added in step 5). `GHCase` is in `initialdata.jl`, amended in step 3 |
| `src/io.jl` | the analysis time series, slice output |
| `src/benchmark.jl` | per-phase timings in TreeWave's format |
| `test/` | one `*_tests.jl` per section above, `prerequisite_tests.jl` (what the two pinned dependencies must still provide; added in step 0), `type_tests.jl`, `threading_tests.jl`, `device_tests.jl`, the standalone `thread_workload.jl`, and `evolution_cases.jl` — a *helper*, the runs the convergence and noise studies are made of, which lives in `test/` because what it wraps is the integrator loop and `driver.jl` is step 5's (added in step 3, after TreeAMR's `test/wave.jl`). `pointwise.jl`'s tests are **two** files over a shared `pointwise_backgrounds.jl` — `pointwise_tests.jl` for the algebra as a function of the state, `pointwise_identity_tests.jl` for the two identities that need derivatives of it — because between them they compile the metric library's nested dual passes for six backgrounds at two precisions (amended in step 1). The interface-order table has a file of its own, `interface_tests.jl`, rather than a testset in `convergence_tests.jl` **(proposed in step 4)**: it is fifteen evolutions on a mesh where the ghost fill costs four times what it costs on a uniform one, and separating it keeps the cheap order study cheap. Step 5 adds `interior_tests.jl` (the profiles, the core rule, the masks, the radius assertions, and one right-hand-side evaluation on a mesh), `driver_tests.jl` (the runs), the hole fixture in `evolution_cases.jl`, and the **standalone** `test/hole_runs.jl` — the `t = 50 M` runs, `q = 4`, and the two harmonic charts, which are minutes rather than seconds and are run by hand with their numbers recorded here **(proposed in step 5**, following `PLAN.md`'s instruction to put what cannot fit a test file in a script under `test/`**)**. Step 8b adds `bounds_tests.jl` (the projection on synthetic states and on every background, its `Float32` row, planted failures on the fixture's mesh, and the bitwise control) and `hole_runs.jl`'s `bounds` section |
| `bin/` | `gh.jl` (the CLI, after GHSO2's `gh3d.jl`), viewers, `benchmark.jl`, `backend.jl`, own `Project.toml` |

Dependencies: `TreeAMR` and `SpacetimeMetrics` (both unregistered, both
pinned to GitHub `main` by `[sources]`, which puts the Julia floor at
1.11 as in the siblings), `KernelAbstractions`, `StaticArrays`
(kernel-safe, and what `SpacetimeMetrics` speaks),
`OrdinaryDiffEqLowOrderRK` and `SciMLBase`, `LinearAlgebra` (`det`, `dot`
and `tr` on `StaticArrays`, which the pointwise algebra uses; a standard
library, added in step 1 and not listed when `PLAN.md` enumerated step 0's
`Project.toml` **(proposed in step 1)**), `HDF5` (from G6),
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
adding a local-path source would hide that instead of stating it. Tests add
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
  [Measured results](#measured-results). **G4b (step 6) is done with it**:
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
  (`a = 0.9`) Kerr in harmonic coordinates crossing the box.
  *Accept:* the indicator's refinement follows the hole, its centroid
  within a few finest spacings of the analytic center at every chunk;
  the layer follows the analytic center with the radius assertions
  holding at every regrid; the time-dependent Dirichlet data exact at
  the boundary; the masked error stays at the static run's level over
  the crossing and converges at order `q` on the frozen hierarchy; the
  adaptive run matches the uniform-fine reference at fewer points; the
  interior residual at truncation, points released by the core relaxed
  within `1/ρ_max`; `:frozen` measured to fail as predicted; the
  horizon found along the trajectory with its area, mass, spin and the
  boost's contraction recovered.
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
belongs to.

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

**G4a (step 5), the interior and the driver.** The suite is
**2486 assertions in 11m51** at one thread and **8m33** at four,
on the development machine (Apple silicon, 12 CPU threads, Julia 1.13.0),
up from step 4's 2144 in 8m40. A clean archive with no `Manifest.toml`
resolves the two pins from GitHub `main`, instantiates and passes with the
same count. Where the growth went:

| file | 1 thread | what it pays for |
|---|---|---|
| `driver_tests.jl` | **2m08** | five static-hole runs: the record, the order sweep at three resolutions, the three variants, the drift, and the `Π` post-pass |
| `interior_tests.jl` | **15 s** | the profiles, the core rule, the masks and the radius refusals (1.5 s), and one right-hand-side evaluation with the layer on a mesh (13.5 s) |
| `type_tests.jl` | **+23 s** | the whole hole pipeline again at `Float32` |

`driver_tests.jl` is now the suite's most expensive file and the third
whose cost is *arithmetic* rather than compilation. That is a hole being a
hole: `CODE.md`'s two radius requirements set the spacing from the
horizon's smallest coordinate radius, and a mesh that satisfies them at
`q = 2` is 61 440 points before anything is evolved. Before adding a run,
price it — and prefer `test/hole_runs.jl`, which is not in the suite.

**What a hole costs, before anything else.** The two radius requirements
together are `r_h,min ≥ (m + 2G + 2)·h + r_0`, so the finest spacing is
set by the horizon's *smallest coordinate radius* and by nothing about
the exterior. At the default margin `m = 8` the coefficient
`m + 2G + 2` is **14** at `q = 2` and **16** at `q = 4`, and the finest
level has also to cover the sphere `r_1` whole. Taking `r_0` where
`|h| ≈ 10`, that is, at `q = 2`: `h ≲ M/8` for Kerr-Schild at `a = 0`,
`h ≲ M/23` for harmonic Kerr at `a = 0`, and `h ≲ M/36` for harmonic Kerr
at `a = 9/10`. That factor of four and a half between the first and the
last — a factor of ninety in points — is why the suite's hole is
Kerr-Schild — which is
also the one with a *sampled gauge source*, so the cheap case is the one
that puts `Hsrc` under a hole. The amplitude, sampled along a ray that misses
the equatorial plane, runs the other way: `|h|` reaches 10 at `r = 0.2 M`
in Kerr-Schild at `a = 0`, at `0.4 M` in the harmonic chart at `a = 0`,
and **nowhere** at `a = 9/10`, where along such a ray it stays between 10
and 12 from `2 M` in to `0.05 M`. That last number is a trap and is the
reason the paragraph says "along a ray": at `a = 9/10` the chart is not
singular at the origin, it is singular on the **equatorial disk of
coordinate radius `a`**, and a ray that misses the disk never sees it.
On a grid, which does not miss it, `r_0` has to be larger than `|a|` —
which is the finding under
[The interior](#the-interior-a-pointwise-damping-layer) and what stops
the harmonic chart at `a = 9/10` altogether.

**The order of the scheme with the layer in place.** Kerr-Schild `a = 0`
in a Dirichlet box of half-width `5/2 M`, on the frozen hierarchy of
`hole_forest` — 120 leaves, 56 at level 2 around 64 at level 3, so the
sphere `r_1` lies wholly inside the finest level and there is a
coarse-fine face between it and the outer half of the box — with
`r_0 = 2/5`, `r_1 = 23/20`, the default margin `m = 8`, `ε_KO = 1/2`,
`γ0` the Gaussian profile and the `:damped` interior. `N` is raised with
the block layout held fixed, so every spacing shrinks and nothing else
moves. The error is the **masked** one: over `r ≥ r_1`, where the
equations are the Einstein equations and nothing else.

| `q` | `t_end` | `N` | `h` | masked L2 | masked L∞ | `C_a` L2 | layer residual |
|---|---|---|---|---|---|---|---|
| 2 | `3/20 M` | 6 | 5/48 | 6.381e−3 | 7.870e−2 | 6.290e−3 | 1.380e−1 |
| 2 | | 8 | 5/64 | 3.362e−3 | 3.549e−2 | 3.527e−3 | 6.210e−2 |
| 2 | | 10 | 1/16 | 2.115e−3 | 1.942e−2 | 2.256e−3 | 3.288e−2 |
| | | **rate** | | **2.16** | **2.74** | **2.01** | **2.81** |
| 4 | `1/4 M` | 8 | 5/64 | 5.464e−4 | 1.098e−2 | 1.630e−4 | 3.483e−2 |
| 4 | | 10 | 1/16 | 1.996e−4 | 3.774e−3 | 6.387e−5 | 1.950e−2 |
| 4 | | 12 | 5/96 | 9.021e−5 | 1.605e−3 | 3.033e−5 | 1.162e−2 |
| | | **rate** | | **4.45** | **4.74** | **4.15** | **2.70** |

Both rows are order `q` in the error and in the constraint, which is the
claim; the `q = 2` row is the suite's (`test/driver_tests.jl`) and the
`q = 4` row is `test/hole_runs.jl`'s. Two things in the table are not the
headline and are worth naming. The **layer residual** converges at about
`2.7`–`2.8` at *both* orders, which is what it should do: it is not a
truncation error of the scheme but the balance `ρ · residual ≈ w · F`,
and `ρ_max = 1/dt ∝ 1/h` while `F`'s truncation error is `O(h^q)` — the
`q = 2` row is `q + 1` because of the `1/h` and the `q = 4` row is short
of `q + 1` because at `h = 5/96` the layer is resolving a metric whose
own gradients are the steepest thing on the mesh. And the **masked
against unmasked** error: at `q = 2`, `N = 8`, masking the interior takes
the L2 from `5.052e−3` to `3.362e−3` and the L∞ from `6.210e−2` to
`3.549e−2` — the worst point of the whole domain is inside the layer,
every time, which is the arithmetic reason the mask is not optional.

**What the layer costs: `CODE.md`'s "a few percent of an RHS",
confirmed.** The cost is one forward-mode dual pass through the
background per *layer* point per evaluation, and it is measured on the
same mesh and the same state with and without the interior
(`INT = :none`, which evaluates `F` everywhere and `u_exact` nowhere):

| `q` | `N` | points | in the layer | ns/point without | with | share |
|---|---|---|---|---|---|---|
| 4 | 12 | 207 360 | 43 162 (20.8 %) | 1142 | 1193 | **4.5 %** |
| 2 | 8 | 61 440 | 12 714 (20.7 %) | — | — | **5.9 %** |

— 246 ns per layer point at `q = 4` on four threads, which is GHSO2's
"about a microsecond on a CPU" divided by the cores. The share is
slightly *larger* at `q = 2` than at `q = 4` and for the obvious reason:
the numerator is the same dual pass at both orders and the denominator is
a right-hand side whose cost per point grows with `q`. Both are a few
percent, and a fifth of the points paying for the analytic solution is
what both numbers are.

**The three interior variants.** `CODE.md` names three and predicts that
`:damped` and `:pasted` both hold the static hole with constraints at
truncation outside `r_1`, `:damped` with the smaller violation in the `G`
points outside `r_1`, and that `:frozen` piles compressed features up
against the freezing radius. On the fixture above (`q = 2`, `N = 8`,
`t = 1/10 M`), with the shell of `G = 2` spacings just outside `r_1` —
6104 points — read through a `ShellMask`:

| variant | layer residual (L∞) | `C_a` L2 in the shell | `C_a` L∞ in the shell | masked error L2 |
|---|---|---|---|---|
| `:damped` | **6.204e−2** | 1.1054e−2 | 6.956e−2 | 2.2688e−3 |
| `:pasted` | **0** (by construction) | 1.1360e−2 | 7.083e−2 | 2.3681e−3 |
| `:frozen` | **6.753e−1** | 1.1062e−2 | 6.955e−2 | 2.2583e−3 |

**The prediction is right about the residual and nearly silent about the
constraints.** `:frozen`'s residual is **11 times** `:damped`'s after a
tenth of an `M` and grows linearly with time — `7.8e−1, 1.63, 2.56, 3.56`
at `t = 0.05, 0.1, 0.15, 0.2 M` on a coarser mesh, against `:damped`'s
`2.42e−1, 2.56e−1, 2.57e−1, 2.57e−1`, which *saturates* after the first
chunk. That
is exactly the difference between a sticky wall and a sink, and it is the
measurement the default rests on. The constraint norms in the `G` points
outside `r_1`, on the other hand, separate the three by **3 %**:
`:damped` is below `:pasted` as predicted, and `:frozen` is
indistinguishable from `:damped` there over this time. So the acceptance
criterion "their constraint norms in the `G` points outside `r_1`" is
measured and recorded, and it is *not* what chooses the default — the
residual is (amended in step 5). **`:damped` is confirmed as the
default.**

**To `t = 50 M`, and the prediction that was wrong.** The same
configuration at `q = 2`, `N = 8` (`h = 5/64`, 120 leaves, `cfl = 1/5`,
`chunk = 1 M`, 5350 steps, 19 minutes at four threads), run to `t = 50 M`
for each variant:

| variant | reaches | masked L2 at the end | `C_a` L2 | layer residual |
|---|---|---|---|---|
| `:damped` | **`50 M`** | 1.604e−1 | 3.951e−2 | 1.387 |
| `:pasted` | `17 M`, then a degenerate metric | — | — | — |
| `:frozen` | `13 M`, then a degenerate metric | — | — | — |

`CODE.md` predicted that "`:damped` and `:pasted` both hold the static
hole to `t = 50 M` … `:frozen` holds the static hole only with
`ε_KO ≈ 0.5` and a wide ramp". **Half of that is wrong and the half that
matters is right (measured in step 5):** with `ε_KO = 1/2` and the
Gaussian `γ0`, *only* `:damped` reaches `t = 50 M`. `:pasted` fails at
`17 M` and `:frozen` at `13 M`, both by the same mechanism — `√(det γ)`
of a state that is no longer a metric — and both inside the layer. That
is the strongest evidence for the default there is: the hard paste's
truncation-order mismatch at a *surface* is not a small perturbation of
the smooth layer, it is a kink that the stencils straddling it feed back
into the interior until the metric degenerates. The same mesh one
resolution coarser (`N = 6`, `h = 5/48`) does not hold even `:damped`,
which fails at `21 M`: **26 points per `M` holds this hole at `q = 2` and
19 does not**, and that is a resolution statement about a second-order
scheme and not about the layer.

What "reaches `t = 50 M`" does *not* mean is that the answer is good. The
masked L2 error at the end is `1.6e−1` against `3.4e−3` at `t = 0.15 M`,
and the interior residual is `1.4` — this is `q = 2` at 26 points per `M`,
where fifty crossings of accumulated truncation error is a large number.
The claim the row supports is *stability*, and the order claims are the
table above.

**The other two charts.** The suite's hole is Kerr-Schild at `a = 0`
because it is the cheapest; `test/hole_runs.jl` runs the two the
resolution argument above says are expensive, at `q = 2` to `t = 1/5 M`:

| background | `r_h,min` | `r_sing` | `r_0` | `r_1` | leaves | `h` | masked L2 | `C_a` L2 | residual |
|---|---|---|---|---|---|---|---|---|---|
| `Harmonic(1, 0)` | 1.000 | 0 | 0.20 | 0.67 | 120 | 5/64 | 1.679e−1 | 1.736e−3 | 7.92e+1 |
| `KerrSchild(1, 9/10)` | 1.436 | 0.900 | 0.95 | 1.25 | 1128 | 5/128 | 2.438e−3 | 1.118e−3 | 2.31e−1 |
| `Harmonic(1, 9/10)` | 0.436 | 0.900 | — | — | — | — | **refused** | | |

Both runs stay finite and both constraint norms are at the same `1e−3` as
the suite's, which is the claim. The **harmonic hole at `a = 0` is badly
under-resolved at this spacing and says so**: its `r_0` has to sit at
`0.2 M`, where `|h| ≈ 29` rather than `CODE.md`'s `≲ 10`, because
`r_h,min = M` leaves no room for a larger one at `h = 5/64` — and the
layer's residual is `79`, three orders above the evolved region's error.
That is the guidance in "The interior" being right: `r_0` where the
solution is still moderate is not a nicety, and the harmonic chart at
`a = 0` needs `h ≈ M/23` before it has one. The spinning hole in
Kerr-Schild, at `h = 5/128` and 1128 leaves, has a residual of `0.23` and
is the well-resolved row of the three.

**The gauge drift.** GHSO2 measured a slow, constraint-preserving drift
of the excised hole under prescribed sources at `≈ 0.14/M`, and `CODE.md`
predicts that exact interior and boundary data lower it without removing
it. Measured as the L∞ of `|h_tt − h_tt,exact|` over the shell from the
horizon's smallest coordinate radius to its largest plus the layer's
width — `[2 M, 2.75 M]` for Kerr-Schild `a = 0` with this layer — against
time, with the initial data exact so the fit goes through the origin: at
`q = 2`, `N = 8`, over `t = 0 … 1/4 M`, the rate is **1.95e−3 / M**.
Over the `t = 50 M` run (`:damped`, `q = 2`, `N = 8`) the drift reaches
`4.077e−3` at the end, a rate of about **8e−5 / M** averaged over fifty
crossings — two thousand times below GHSO2's `0.14/M` on the excised
hole. **`CODE.md`'s prediction is confirmed in its strong form
(measured in step 5):** exact interior *and* boundary data do not merely
lower the drift, they leave it at the truncation error's level, and the
number that is measured here is the truncation error and not a gauge
mode. The short-run rate is larger than the long-run one because the
drift saturates rather than growing, which is what a gauge mode would not
do.

**`Float32`.** The same static-hole run — `q = 2`, `N = 8`, `t = 1/10 M`,
the `:damped` layer, the sampled gauge source, the Gaussian `γ0` — runs
end to end at `Float32` on `CPU()` and reaches the `Float64` answer to
**five significant figures**: masked L2 `2.268762e−3` against
`2.268849e−3`, masked L∞ `2.659585e−2` against `2.659472e−2`, the layer
residual `6.204157e−2` against `6.204212e−2`, `λ_max` `1.6709520` against
`1.6709517`, and the same step count. The gauge constraint — a difference
of large terms near a hole, which is where `Float32` has least to give —
agrees to `3.5e−3` against `3.5e−3`. This is the sharpest `Float32`
result in the package, and it is the offset identities of
[G1](#milestones) earning their keep: `metric_quantities` takes
`g^{ab} − η^{ab}` from GHSO2's identity rather than from `inv(g) − η`,
whose `Float32` error is `4.2e−3` against the identity's `1.4e−7`.

**What the record is.** Every number the driver writes down is
`Float64` at every element type (`precision.jl`'s `tofloat64`), so a
`Float32` run's analysis time series is comparable with a `Float64` one
without a conversion at every call site; the *arithmetic* is in the type
the caller named, which the agreement above is the test of.

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
`5/64`:

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
move.

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
the `:damped` layer) to `t = 10 M`, the horizon found at every chunk:

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

**Step 8a, the expectations: what crosses the horizon from inside it.** No
`src` change. The suite is **3463 assertions in 13m01** at one thread and
**10m01** at four (15m08 in a first four-thread run while a sibling step's
suite shared the machine), the nineteen new ones being
`stencils_tests.jl`'s Nyquist-slope testset. The frozen-coefficient table
is under [Kreiss–Oliger dissipation](#kreissoliger-dissipation) and the
two margin rules it leads to under [The
interior](#the-interior-a-pointwise-damping-layer); this is the 3D
measurement they are held against (`test/hole_runs.jl leakage`, one
`amddebugq` node, sixteen groups as subprocesses, **45m47**; each of the
88 runs `2 M` long and 350–480 s at three or five threads).

*The experiment.* A radial ripple `A (1 − s²)³ cos(2π(r − r_c)/λ)`,
`s = (r − r_c)/2h`, `A = 1e−3`, in `h_tt` with `Π` untouched, centered `d`
cells inside the horizon (`r_c = r_h − d h`), on `hole_fixture` (`:damped`,
`r_1 = 23/20`, the layer at `ρ_max = 1/dt`) at `h = 5/64`; `A_k` is the
largest `|δh|` over the ten `h` components against the same run without
the ripple, in the shell `r_h + k h ≤ r < r_h + (k+1) h` (`ShellMask`'s
membership), over the chunk boundaries at every `1/20 M`. Four decisions
the step was silent on, all **(proposed in step 8a)** and argued in the
section's header: the mesh is the fixture's **finest level everywhere** — a
uniform level-3 forest of 512 blocks, since the 120-block hierarchy puts
the whole margin at `5/32` along the axes, where a `2h` ripple is a
constant; the ripple is in **`h_tt` alone**, which excites both branches
equally; `d` is the depth of the window's **center**, with a half-width of
**two cells**, which is what fits `d = 2` inside the horizon and `d = 8`
outside `r_1`; and the runs are **`2 M`** rather than `PLAN.md`'s `1 M`,
because the dispersion analysis puts the slow modes three to fourteen `M`
from depth `d` to the outer shells. The ripple enters through `evolve!`'s
observer at `t = 0`, which is handed the state vector the first chunk
integrates from; every perturbed run took the same 200 steps as its
reference and stayed finite, and a run whose ripple had not reached the
evolution would have been refused.

*At the default `ε_KO = 1/2`*, the largest `A_0/A` over `λ = 2h, 4h, 8h`
(the first shell outside the horizon) and `A_8/A`, against `PLAN.md`'s
prediction `e^{−(d+k)/ℓ_max}` with `ℓ_max` at the source's radius, the path
integral of `test/dispersion.jl` over the modes arriving by `2 M`, and its
one-dimensional model at `2 M` and `10 M`:

| `q` | `d` | 3D `A_0/A`, `2 M` | `e^{−d/ℓ_max}` | path, `2 M` | 1D, `2 M` | 1D, `10 M` | 3D `A_8/A` | `e^{−(d+8)/ℓ_max}` |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 0.144 | 0.773 | 0.805 | 0.166 | 0.410 | 8.9e−3 | 0.276 |
| 2 | 4 | 0.078 | 0.385 | 0.437 | 0.092 | 0.136 | 1.9e−3 | 0.057 |
| 2 | 8 | 9.9e−3 | 0.039 | 0.047 | 8.9e−3 | 0.058 | 9.6e−5 | 1.6e−3 |
| 4 | 2 | 0.166 | 0.657 | 0.715 | 0.117 | 0.348 | 8.4e−3 | 0.122 |
| 4 | 4 | 0.077 | 0.324 | 0.397 | 0.044 | 0.177 | 3.3e−3 | 0.034 |
| 4 | 8 | 8.3e−3 | 0.060 | 0.086 | 0.011 | 0.037 | 3.8e−4 | 3.6e−3 |

and the attenuation per cell — the least-squares slope of `log A_0` against
`d = 2, 4, 8`, over the three `λ` — against `1/ℓ_max(r_1)`, what a constant
`ℓ` at the layer's radius would give:

| `ε_KO` | 3D `q = 2`, `2 M` | 3D `q = 4`, `2 M` | 1D `q = 2`, `10 M` | 1D `q = 4`, `10 M` | `1/ℓ_max(r_1)`, `q = 2, 4` |
|---|---|---|---|---|---|
| 0 | 0.07–0.22 | 0.03–0.08 | −0.03–0.03 | 0.01–0.08 | 0, 0 |
| 1/4 | 0.30–0.40 | 0.28–0.38 | 0.07–0.21 | 0.17–0.31 | 0.24, 0.19 |
| 1/2 | 0.40–0.47 | 0.42–0.51 | 0.16–0.31 | 0.25–0.37 | 0.48, 0.37 |
| 1 | 0.50–0.53 | 0.51–0.55 | 0.33–0.39 | 0.36–0.43 | 0.96, 0.75 |

What agrees and what does not — the discrepancy `PLAN.md` asks to have
recorded:

- **The depth dependence is right at the default; the level is not.**
  `PLAN.md`'s `e^{−d/ℓ_max}` overstates the `2 M` transmission 4–7× at
  every depth — the ripple is broadband and splits between two branches,
  and only the part near the least-damped `θ` crosses — while its slope
  between `d = 2` and `8`, `0.50` e-folds per cell at `q = 2` and `0.40` at
  `q = 4`, is within 20 % of the measured. The path integral does no better
  on the level (the measurement is `0.18–0.21` of it at `q = 2`).
- **The measurement is not `∝ ε`, and the prediction is.** From
  `ε_KO = 1/4` to `1` the measured slope moves from about `0.35` to `0.52`
  e-folds per cell, against `0.24 → 0.96` predicted, and doubling `ε_KO`
  from `1/2` to `1` lowers `A_0` from depth 8 by only `1.7–2.5×`. The
  one-dimensional model saturates the same way, so this is a property of
  the principal part with coefficients that vary along the path, not of the
  source terms or of the other directions; the likeliest reading is the
  refraction the frozen model leaves out — a packet on a stationary
  background conserves `ω`, not `θ`, and slides toward the less-damped
  small `θ` as it nears the horizon — and it is a reading, not a
  measurement.
- **`2 M` is a lower bound.** At `q = 2` the transmission from depth 8 is
  still rising at `2 M` (`3.5e−3` at `1.25 M`, `6.4e−3` at `2 M` for
  `λ = 2h`); at `q = 4` it has levelled by `1.25 M`. The one-dimensional
  model follows the 3D numbers to within a factor 2.3 at `λ ≥ 4h` and
  `ε_KO > 0` — it undershoots them `1.2–3.8×` at `λ = 2h`, a ripple defined
  on the radius that off the axes is not a Nyquist mode of the lattice, and
  up to `6×` at `ε_KO = 0` — and it puts the long-time transmission from
  depth 8 at **4–6 %**: above `PLAN.md`'s prediction at `q = 2` (`0.058`
  against `0.039`) and below it at `q = 4` (`0.037` against `0.060`). As a
  long-time estimate the frozen-coefficient formula is within a factor two.
- **Outside the horizon it falls faster.** `A_8` is 3–11 % of
  `e^{−(d+8)/ℓ_max}`; the decay outside is `0.30–0.59` e-folds per cell at
  `ε_KO = 1/2` (`0.21–0.54` at `1/4`, `0.32–0.55` at `1`), since there
  every mode is outgoing and the grid-scale ones are damped at their own
  `σ`, and the Dirichlet face at `r_h + 6.4 h` along the axes reflects
  whatever reaches it.
- **Without dissipation nothing holds it back**, as `ℓ = ∞` says: at
  `ε_KO = 0`, `A_0/A = 0.13–0.88` at `q = 2` and `0.48–1.10` at `q = 4`,
  where it exceeds the source — the grid-scale growth step 3 measured on
  flat space (×9.2 in a thousand steps) acting on it.
- **The axis is where it leaks.** The cone within 18° of a grid axis holds
  the shell's maximum in 44 of the 72 rows and never less than `0.56` of it,
  and the diagonal cone is below the axis cone in 50 to 69 of the 72 rows
  at every `k`, as the diagonal numbers under [Kreiss–Oliger
  dissipation](#kreissoliger-dissipation) predict.

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

## Possible extensions

What separates the proof of concept from a production code, listed with
the design note that would start each:

- **Checkpoint and restart**: the forest's leaf keys, the time, the
  case and the state vector to HDF5, read back into a fresh forest
  (`Forest`, `refine!` to the keys, `balance!`, `scatter!`); bit-identical
  to an uninterrupted run, since the layer and the hook depend on
  `(x, t)` only. A few dozen lines, or TreeAMR's M9.
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
  holes only.
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
  the kernel is the first thing to change.
- **A truncation-error indicator** in place of Löhner's — Richardson
  between a block and its restriction, which TreeAMR's operators make
  cheap — if the calibration in G4 shows Löhner's thresholds to be
  fragile across resolutions.
- **Smoothing the interior** for a spacetime without an analytic
  solution — the interior layer relaxing toward something other than
  the exact solution. **(Became PLAN.md step 8e on 2026-09-23**: the
  fitted target; see [Open questions](#open-questions).**)**
- **A hyperboloidal outer layer**, which this author's electrodynamics
  packages rehearse and which would retire the outer boundary question.
- **Matter**, through the source slot: TreeHydro's scheme on this
  spacetime is the GRHD code the two packages together rehearse.

## Open questions

Settled in review on 2026-09-16: the expanded form (the flux form is not
implemented on the mesh); three dimensions only; no excision, the
interior treated pointwise by profiles of the distance to the center;
the proof-of-concept target a single boosted, spinning hole;
OrdinaryDiffEq's RK4 for time integration; the analysis quantities as
part of the deliverable; an error indicator for refinement; `Float64`
on the H200 as the device requirement; no checkpointing; the inherited
documents copied into `notes/`.

**Opened in step 5, and the one thing that stands between this package
and its own proof of concept: a spherical frozen core cannot be used with
`Harmonic(M, 9/10)`.** The chart's singular set is the equatorial disk of
coordinate radius `a`, the horizon's smallest coordinate radius is
`√(M² − a²)`, and a ball fits between them only where `a < M/√2 ≈ 0.707`.
The two candidate answers — key the interior on the chart's own
spheroidal radius `R`, or run G5 at `a = 0.7` — are written out under
[The interior](#the-interior-a-pointwise-damping-layer). Step 8 has to
pick one before it can run the case this document is named for; nothing
in G4 depends on it, and Kerr-Schild at `a = 9/10` runs today.

**Answered in design review on 2026-09-23 — neither, and PLAN.md's steps
8a–8g build and measure the answer.** The layer is keyed on the *found*
horizon's offset surface `r_1(n̂) = r_h(n̂) − m h` — a sphere on the
*tracked* radii does not contain the disk either, since `0.436 − m h` is
far inside it, while the offset surface does once `m h < 0.1 M` on the
equator — and it relaxes toward a regular fit of the evolved state
instead of the analytic solution, so no singular set has to be contained
and no analytic interior is needed: the same treatment serves a spinning,
a moving and a newly found horizon. The review found three more things
the steps rest on and measure: `ρ_max = 1/dt` is a *grid* rate (about
`107/M` on the suite's fixture), which makes today's layer a paste two
cells inside `r_1` that survives to `50 M` only because its target is
exact — a generic target needs a thick ramp at a physical rate; the
Lorentzian metrics are not convex in `g_ab` (the angular mean of
Kerr-Schild `g_ab` inside the horizon has Euclidean signature), so every
blend and clamp is made in ADM variables; and the discrete scheme's
grid-scale modes have *outgoing* group velocity inside the horizon (every
centered first-derivative stencil annihilates the Nyquist mode, so the
shift advection does not act on it), attenuated only by dissipation —
step 8a computes their penetration length for this package's stencils
before anything is built, and it is what the margin `m` is measured
against. `a = 7/10` stays the fallback for G5 if the last row of step
8f's matrix does not fit a node.

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
   split it would be measured against is 3:1 and not 1:1.
4. **In-kernel evaluation of `SpacetimeMetrics`** on the H200 (G6); the
   structure does not depend on it, the cost does.
5. **The defaults** — `q = 4`, `N = 32`, `cfl = 1/4`, `ε_KO = 0.5`,
   `γ0 = 1/M` near the hole, `m = 8`, a layer of `2(G + 1)` spacings,
   `ρ_max · dt = 1`, the indicator's thresholds — are starting values
   for G4–G6 to confirm or move. Three of them survived G2 on flat space
   and on a gauge wave: `cfl = 1/4` (no run needed less), `ε_KO = 0.5`
   (the noise test, and no order lost) and `q = 4` as the development
   order (`q = 2, 6, 8` all run, at 0.5×, 1.3× and 1.9× the cost of
   `q = 4`). None of that is yet a statement about a hole. Step 8a split
   `m` into a stencil margin and a leakage margin, measured that `m = 8`
   attenuates grid-scale content from `r_1` by `e^{−2.9}` to `e^{−4.6}` on
   the fixture, and left the default where it is until step 8c says what
   amplitude it has to hold back **(proposed in step 8a**; see [The
   interior](#the-interior-a-pointwise-damping-layer)**)**.
