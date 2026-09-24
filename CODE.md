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
   reaches `5 M` at `4/M` ("The generic interior: the measurement matrix"
   under [Measured results](#measured-results)).

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

#### Step 5's layer: the analytic control

**No excision** (decided). Inside the horizon the solution is not left
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
results](#the-layer-against-an-inexact-target-step-8c) decide:

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
   The table of the screens is in [Measured results](#measured-results),
   "The moving hole (step 8)".

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
| fitted target | for a `:fitted` case (added in step 8e; `nothing` otherwise), of the fit built from this row's state: **`fit_valid`** (every point of the fit's validity sweep a Lorentzian metric — `fit_valid(fit)`; a fit that is not does not become the target), **`fit_residual`** (`fit_residual(fit).overall`, the fit's worst relative residual block by block against the data it was fitted to), and beside them the sweep's worst **`fit_min_detγ`**, `fit_min_α`, `fit_min_λ`, `fit_hits` (swept points the target's ranges would move) and `fit_refills` (the cache's mid-chunk refills in the chunk this row ends) **(proposed in step 8e)**; for `:fitted` the row `residual` is the layer's distance from its target (the cache), not from the analytic solution | every chunk |

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
| `src/gauge.jl` | sampling prescribed sources into `Hsrc` and reading them back at a point (`gauge_at`, the kernel's half of the packing); `isharmonic` as a table over the background types and `isstatic` as an exact measurement, with the reason each is what it is (added in step 3); the two `γ0` profiles (step 5) and the `ε_KO(r)` profile `HorizonDissipation`, with `dissipation_rate` the identity on a number (step 8c) |
| `src/boundaries.jl` | the time-dependent Dirichlet hook |
| `src/bounds.jl` | the range projection (added in step 8b): `StateBounds` and the proposed `default_bounds`/`default_gate`, `check_bounds_gate`, the pointwise `bounds_project` over an explicit-scalar ADM split and a Jacobi `sym_eigen3`, `gh_bounds_kernel!`, `BoundsAccounting`, `apply_bounds!` and `gh_stage_limiter!`; the validity monitor (`state_validity`, `validity_rows`); and `evolved_nonfinite`, the masked finiteness check. Included after `interior.jl` and before `initialdata.jl`, whose `GHCase` carries a `StateBounds` |
| `src/interior.jl` | the profiles `w(r)`, `ρ(r)`, the core rule, the radius checks, the masks; added in step 5. Also `HoleCenter` — `c(t) = c₀ + v t` as two vectors and a line, which is what "the center is a function of `t`, never a mutated field" means as code — the horizon's analytic coordinate radii and the hole's mass (`hole_mass`, added in step 8c′), and `layer_spacing`, the coarsest spacing among the blocks the sphere `r_1` passes through, which is the one number in the file that looks at a mesh (and looks at it only to *assert*). The `:pasted` limiter is in `evolution.jl` instead **(amended in step 5)**, beside the kernel it launches and the state layout it writes. **Step 8d adds the tracked geometry's kernel side**: the real harmonics (`real_harmonic_index`, `real_from_complex`/`complex_from_real`, the recurrence `shape_series`, `shape_bounds`), the analytic horizon of the seed (`analytic_horizon_radius`), `FittedSpec` — what a case holds — and `FittedInterior` — the kernel argument — with `interior_point`, `fitted_geometry`, `core_position`, `ShapeMask`, `ShapeBand`, `geometry_spacing` and its `check_interior_radii`; and the protocol both geometries speak (`in_layer(int, t, x)`, `interior_point`, `is_outside`, `geometry_radii`, `layer_radii`, `layer_mask`, `shell_mask`) |
| `src/initialdata.jl` | backgrounds, `GHCase` and the case constructors (here rather than in `driver.jl`, amended in step 3), the forest builders — uniform, with one root block refined for the frozen two-level hierarchy the interface study needs (`refined = true`, added in step 4), or `hole_forest`'s nested shells around a hole (added in step 5, **here rather than in `interior.jl`**, since a forest builder belongs with the other forest builder) — the `(h, Π)` callback with the core rule, the `SpacetimeMetrics` index conversion and nowhere else |
| `src/refinement.jl` | the Löhner indicator with its global floor, the mask, the level floor and ceiling, the four marks, the buffer; TreeWave's `refinement.jl` ported |
| `src/constraints.jl` | the gauge-constraint kernel (state and first derivatives) and the ADM one (every second derivative of `g_ab`, the `∂_t` blocks from the evolution equations, the four-dimensional Ricci tensor assembled rather than reduced), the masks they take — `AllPoints` and the `is_evolved` predicate step 5's interior adds a method to — `masked_norms` and `constraint_norms`, and `adm_constraints_at_node`, the pointwise curvature assembly the tests check against `ddmetric` (added in step 4) |
| `src/horizon.jl` | the interpolating ADM provider for `ApparentHorizonFinder`; location, shape, area, `M_irr`, `J`, `M_ch`. Added in step 7, in the order the numbers are produced: `locate_block` and `interpolate`/`interpolate_grad` (the stopgap of [Upstream prerequisites](#upstream-prerequisites), item 1, with the footprint guard that refuses a query reaching inside `r_1`), `GHADMProvider` (batched, `Float64` out whatever the run computes in, with a one-entry cache keyed on the identity of the query array because `KorzynskiSpin.surface_geometry` asks for `γ` and `K` in two calls with the same points), `find_gh_horizon`, and `Horizon` — the cadence and resolution the case carries |
| `src/tracking.jl` | the tracked horizon, host-side (added in step 8d, after `horizon.jl` and before `driver.jl`): the conversions from the finder's `hlm` (`real_shape`) and of the analytic horizon (`analytic_shape`) into real coefficients, `HorizonTrack` with `seed_track`, `update_track`, `track_center` and `TrackLostError`, `fitted_interior` — the kernel argument from a track and a mesh — `surface_shift` (the gauge source's re-sample rule), and `axis_dispersion`/`margin_efolds`, step 8a's leakage e-folds moved in from `test/dispersion.jl` |
| `src/fit.jl` | the fitted target (added in step 8e, after `tracking.jl` and before `driver.jl`): the fit's variables (`fit_variables`, `state_from_fit`), the real solid harmonics (`_solid_harmonic_fold`, `real_solid_harmonics`, `fit_directions`), the two samplers (`state_sampler`, `analytic_sampler`), the least squares (`solve_fit`, `fit_row_weights`), the validity sweep (`fit_sweep`), `FitParams` and `InteriorFit` with `build_fit`, `fit_residual` and `fit_valid`, the kernel-callable evaluator `fit_variables_at`/`fit_state`, and the kernel half (8e-ii): `derive_target_bounds`, the 40-variable cache (`target_cache`, `fit_target_kernel!`, `fill_target!`) and the initial data's `fitted_state_kernel!`. The variant's branch is in `evolution.jl` (`GHProblem`'s `target`/`fits`/`t_target`, `refill_target`), its residual in `constraints.jl`'s error kernel, its flow in `driver.jl` (`refit!`, the refill and the pieces of a moving chunk) |
| `src/driver.jl` | `evolve!`, the analysis record per chunk, `observer`, `check_cfl`, `horizon_shell`, `forest_levels`, `default_relaxation_rate` — the layer's default `4/M`, the one place the number is written (added in step 8c′) — and `discrete_gradient_momentum!` — GHSO2's `Π` post-pass, which lives here because it runs once on the initial data and is the driver's option, not the initial data's (added in step 5). `GHCase` is in `initialdata.jl`, amended in step 3 |
| `src/io.jl` | the analysis time series, slice output |
| `src/benchmark.jl` | per-phase timings in TreeWave's format |
| `test/` | one `*_tests.jl` per section above, `prerequisite_tests.jl` (what the two pinned dependencies must still provide; added in step 0), `type_tests.jl`, `threading_tests.jl`, `device_tests.jl`, the standalone `thread_workload.jl`, and `evolution_cases.jl` — a *helper*, the runs the convergence and noise studies are made of, which lives in `test/` because what it wraps is the integrator loop and `driver.jl` is step 5's (added in step 3, after TreeAMR's `test/wave.jl`). `pointwise.jl`'s tests are **two** files over a shared `pointwise_backgrounds.jl` — `pointwise_tests.jl` for the algebra as a function of the state, `pointwise_identity_tests.jl` for the two identities that need derivatives of it — because between them they compile the metric library's nested dual passes for six backgrounds at two precisions (amended in step 1). The interface-order table has a file of its own, `interface_tests.jl`, rather than a testset in `convergence_tests.jl` **(proposed in step 4)**: it is fifteen evolutions on a mesh where the ghost fill costs four times what it costs on a uniform one, and separating it keeps the cheap order study cheap. Step 5 adds `interior_tests.jl` (the profiles, the core rule, the masks, the radius assertions, and one right-hand-side evaluation on a mesh), `driver_tests.jl` (the runs), the hole fixture in `evolution_cases.jl`, and the **standalone** `test/hole_runs.jl` — the `t = 50 M` runs, `q = 4`, and the two harmonic charts, which are minutes rather than seconds and are run by hand with their numbers recorded here **(proposed in step 5**, following `PLAN.md`'s instruction to put what cannot fit a test file in a script under `test/`**)**. Step 8b adds `bounds_tests.jl` (the projection on synthetic states and on every background, its `Float32` row, planted failures on the fixture's mesh, and the bitwise control) and `hole_runs.jl`'s `bounds` section. Step 8c adds the E3 target `CurvatureTarget` to `evolution_cases.jl`, its claims to `interior_tests.jl` and `driver_tests.jl`, and `hole_runs.jl`'s `calibration` section. Step 8d adds `tracking_tests.jl` — the real harmonics against `AbstractSphericalHarmonics`, the seed against the charts' quartic, the depth of an oblate spheroid, the footprint guard on a non-spherical surface, the e-folds against `dispersion.jl`, the bit-identity of a fitted sphere with step 5's layer, one find of the fixture, and the tracked runs — and makes `evolution_cases.jl`'s shell norms the interior's own `shell_mask`. Step 8e adds `fit_tests.jl` — the fit's harmonics against `shape_series` and `ash_evaluate`, the whole ansatz recovered by `solve_fit`, the static hole's fit against its truncation and the interpolation order, the validity sweep on three holes and the `g_ab` mean control, the evaluator against the model at `Float64` and `Float32`, and the fit of the tracked run's final state; and, for 8e-ii, the `:fitted` right-hand side against `:damped`'s bit for bit, the fixture's `:fitted` run to `0.15 M` and one chunk of it at `Float32` — and moves `fitted_fixture` into `evolution_cases.jl` beside `tracked_fixture_run`, the one tracked run `tracking_tests.jl` and `fit_tests.jl` share; `hole_runs.jl` gains a `fitted` section (the fixture's initial-data study to `1 M`, the boosted seed, harmonic `a = 9/10`) |
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
  Kerr in harmonic coordinates crossing the box — **at `a = 7/10`
  (decided 2026-09-23**; `a = 9/10` waits with its price written down,
  [Open questions](#open-questions)**)**, on **the tracked geometry with
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
  measurement matrix (8f) under [Measured results](#measured-results);
  excision (8g) is not needed, since `:fitted` reaches `50 M` on the
  matrix's first row. What G5 inherits from 8f: G5's chart at `h = 5/256`
  is 2472 blocks and `630 s` of a node per `M` when the hole sits still; at
  `5/128` it does not survive; `:fitted` holds a boosted hole at `4/M`
  across `1.5 M` of the box, while the analytic `:damped` control on the
  boosted `a = 0` hole needs `ρ_max ≳ 20/M`, its frozen core released on the
  trailing side after about `M` at `v = 0.3`; and a moving hole's step is
  sized for the speed it will have, since the fastest speed grows by
  0.1–0.3 % a chunk and the CFL recheck otherwise stops the run.
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
`γ0` the Gaussian profile and the `:damped` interior — **at the grid rate
`1/dt`, the default until 2026-09-23**, as is every table of this step
(`ρ_max_factor = 1` reproduces them; the suite's rows at the default `4/M`
follow each, **(measured in step 8c′)**). `N` is raised with
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

**The suite's `q = 2` row at the default `4/M` (measured in step 8c′)**,
the same fixture, runs and time:

| `q` | `t_end` | `N` | `h` | masked L2 | masked L∞ | `C_a` L2 | layer residual |
|---|---|---|---|---|---|---|---|
| 2 | `3/20 M` | 6 | 5/48 | 6.322e−3 | 7.770e−2 | 6.288e−3 | 1.433 |
| 2 | | 8 | 5/64 | 3.336e−3 | 3.501e−2 | 3.530e−3 | 7.965e−1 |
| 2 | | 10 | 1/16 | 2.103e−3 | 1.917e−2 | 2.257e−3 | 5.075e−1 |
| | | **rate** | | **2.16** | **2.74** | **2.01** | **2.03** |

The error is the grid rate's to 1.5 % at every `N`, and lower, the
constraint to `0.1 %`, and their rates agree to the second digit. The
residual is ten to fifteen times larger and one order slower, and both are
the balance above with a fixed `ρ`: `residual ≈ w F/ρ_max` is `O(h^q)` when
`ρ_max` does not grow as `1/h`.
Masking now takes the L2 from `3.536e−2` to `3.336e−3` and the L∞ from
`7.965e−1` — which is the residual — to `3.501e−2`.

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
6104 points — read through a `ShellMask`, at the grid rate `1/dt`, the
default until 2026-09-23:

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

**The three variants at the default `4/M` (measured in step 8c′).**
`:pasted` and `:frozen` do not read the rate — the paste freezes the ball
`r < r_1`, and `:frozen` has `ρ ≡ 0` — so their rows above are unchanged to
every digit. `:damped` at `t = 1/10 M`, the same fixture and shell:

| variant | layer residual (L∞) | `C_a` L2 in the shell | `C_a` L∞ in the shell | masked error L2 |
|---|---|---|---|---|
| `:damped`, `4/M` | 5.513e−1 | 1.1062e−2 | 6.954e−2 | 2.2588e−3 |

The shell is where it was — `:damped` and `:frozen` now agree there to
`1e−4` relative and `:pasted` is 2.7 % above both — but **the residual claim
is not true at `1/10 M` any more**: `:frozen`'s residual is 1.22 times
`:damped`'s, not 11, because the sink relaxes in `M/4` rather than in one
step and has not yet saturated. Against time, every `1/20 M` of the same
run:

| `t/M` | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `:damped`, `4/M` | 0.289 | 0.551 | 0.796 | 1.009 | 1.181 | 1.306 | 1.385 | 1.424 | 1.433 | 1.422 | 1.457 |
| `:frozen` | 0.320 | 0.675 | 1.080 | 1.508 | 1.930 | 2.304 | 2.593 | 2.765 | 3.164 | 3.589 | 6.266 |
| `:damped`, `1/dt` | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.077 | 0.100 | 0.195 |

`:damped` saturates by `0.4 M`, `1.6/ρ_max`, and `:frozen` grows linearly:
the sticky wall against the sink is the same distinction it was, arriving
a relaxation time later. So **the suite makes the residual claim at
`t = 2/ρ_max = 1/2 M` (amended in step 8c′)**, where `:frozen`'s is 2.52
times `:damped`'s, and keeps the shell claim at `1/10 M`, where step 5 made
it: by `1/2 M` `:pasted`'s shell `C_a` L2 is `2.02e−2` against `:damped`'s
`1.24e−2` (`1.21e−2` at the grid rate) and `:frozen`'s `1.25e−2` — the
paste's kink growing toward its `17 M` failure at either rate, which is a
claim about the paste and not the one the testset makes. The price is the
two runs to `1/2 M`, 50 steps each instead of 10; the costs are under
[the default rate](#the-default-rate-step-8c).

**To `t = 50 M`, and the prediction that was wrong.** The same
configuration at `q = 2`, `N = 8` (`h = 5/64`, 120 leaves, `cfl = 1/5`,
`chunk = 1 M`, 5350 steps, 19 minutes at four threads), run to `t = 50 M`
for each variant, at the grid rate `1/dt`, the default until 2026-09-23:

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
of a state that is no longer a metric — and both inside the layer
**(corrected in step 8b** for `:pasted` and for the `N = 6` `:damped` row
below: the failing square root is the lapse's `√(−g^{tt})`, reached by the
right-hand side at a stage vector, and the state degenerates in the
*evolved* shell just outside `r_1` — at `r = 1.18` and `1.27`, against
`r_1 = 1.15` — while the layer inside the projection's gate never moves;
see [the range projection's
measurements](#the-range-projection-step-8b)**)**. That
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

**The default's `50 M` number is step 8c's**, and it is not re-run here
(step 8c′): the exact target at `4/M` on this fixture's own layer, the same
configuration with the range projection on, reaches `50 M` with the masked
error L2 at **`0.027`** against this table's `0.160`, the `G`-point shell's
`C_a` L2 at **`0.029`** against `0.181` (step 8c's control of this row; the
table's `C_a` column is the whole evolved region's), and the finder's
`M_irr` at **`0.9972`** against `0.9955` ([the layer against an inexact
target](#the-layer-against-an-inexact-target-step-8c)). The `:pasted` and
`:frozen` rows do not read the rate and stand; the `N = 6` `:damped` run that
ends at `21 M` has not been run at `4/M`.

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
`q = 2`, `N = 8`, over `t = 0 … 1/4 M`, the rate is **1.95e−3 / M**
(at the grid rate; the suite's run at the default `4/M` gives `1.9512407e−3`
against `1.9512409e−3`, the same to seven digits — the shell starts at the
horizon, `10.9` cells outside `r_1`, and the layer's rate does not reach it in
a fifth of an `M` **(measured in step 8c′)**).
Over the `t = 50 M` run (`:damped`, `q = 2`, `N = 8`) the drift reaches
`4.077e−3` at the end (at the grid rate; step 8c's runs at `4/M` end at
`2.7–3.3e−3`), a rate of about **8e−5 / M** averaged over fifty
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
`1.6709517`, and the same step count — at the grid rate. At the default
`4/M` **(measured in step 8c′)** it is the same agreement on different
numbers: masked L2 `2.258749e−3` against `2.258833e−3`, L∞ `2.637039e−2`
against `2.636953e−2`, the layer residual `5.512611e−1` against
`5.512648e−1`, the gauge constraint `3.481216e−3` against `3.481208e−3`,
`λ_max` and the step count unchanged. The gauge constraint — a difference
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
`r_1 = 23/20`, the layer at `ρ_max = 1/dt`, the default until 2026-09-23 —
a rerun of the section after step 8c′ runs it at `4/M`, while
`test/dispersion.jl`'s one-dimensional model below stays at the `1/dt` these
numbers were measured at) at `h = 5/64`; `A_k` is the
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


### The range projection (step 8b)

`src/bounds.jl`, `test/bounds_tests.jl` and `test/hole_runs.jl bounds`; the
design is under [The interior](#the-interior-a-pointwise-damping-layer).

**The suite.** **3723 assertions in 13m00 at one thread and 9m33 at
four** on the development machine (Apple silicon, Julia 1.13.0), up from
step 7's 3444 in 12m31 and 8m38 — on a machine a sibling agent's runs
shared (load average 4–8), so the wall clock is an upper bound.
`test/bounds_tests.jl` is **279** of them — 278 in the file and the slot
map's one more in `constraints_tests.jl` — in **22.6 s / 10.1 s**: the
pointwise claims `2.4 s`, the planted failures and the gate `≈ 6 s`, and
the control, two `3/20 M` runs of the fixture, the rest. That control is
the one short run the step adds, paid twice because the claim is a
comparison.

**The map, on synthetic states.** The six states `PLAN.md` names — one
negative eigenvalue of `γ`; two negative eigenvalues with `det γ > 0` (which
a determinant floor would pass); `−g^{tt} < 0` with `γ` healthy (on which
`metric_quantities` throws a `DomainError`); a `NaN` in `γ_xy`; an `Inf` in
`Π_tx`; and Kerr-Schild at the fixture's core edge `r_0 = 2/5` — and three
more, one per remaining range (`|β| = 12`, a positive lapse of `10⁻³`,
`(α/√γ)Π = 500`). On each, at `Float64` and `Float32`, the projection
returns a state `metric_quantities` accepts with every ADM quantity in its
range; the blocks it moves are exactly the offending ones, **bit for bit**
(`===` on each of `h_tt`, `h_ti`, `γ_ij` and `Π`); a raised lapse keeps
`αΠ` to `1e−12`; a second application returns the same bits and does not
fire. It is the identity, bit for bit, on all six backgrounds of
`pointwise_backgrounds.jl` at their test points and on Kerr-Schild at
`r = 2/5, 1/2, 23/20` in three directions. On random states (20 000 per
row, `|h_ab|` and `|Π_ab|` uniform in `[−s_h, s_h]`, `[−s_Π, s_Π]`):

| `(s_h, s_Π)` | fired | re-fired before the re-test | after | `Float32`: outside a range |
|---|---|---|---|---|
| `(1/2, 1)` | 318 | 0 | 0 | 0 |
| `(2, 10)` | 18 202 | 0 | 0 | 0 |
| `(5, 100)` | 19 672 | **18** | 0 | 0 (but `metric_quantities` throws on 2 and is off by > 1 % in `α` on 14) |
| `(50, 1000)` | 19 936 | **402** | 0 | 3456 (`metric_quantities` throws on 60) |

The re-fires are the shift test on an ill-conditioned `γ`, and the
self-verifying re-test is what removed them; the `Float32` column is what a
24-bit mantissa can represent — `λ_min = 1/100` beside eigenvalues of `10³`
is a condition number `Float32` cannot resolve — recorded and not fixed
(`Float64` on the H200 is the requirement).

**The mesh, and the control.** On the fixture's mesh (`N = 8`, 120 leaves)
the kernel repairs exactly the points planted inside the gate — a `NaN` in
the core and a `γ_xx = −1/2` at `r < r_gate` — counts them (`hits = 2`,
`nonfinite = 1`, `r_max` the outer one's radius), leaves a `NaN` planted in
the outer layer (`r_gate ≤ r < r_1`) and one at an evolved point alone, and
changes no other bit of the state; `evolved_nonfinite` sees the evolved one
and not the layer's. **The control holds bit for bit**: `:damped` to
`3/20 M` with the projection on makes `4 · nsteps + 1` calls, fires on none,
and ends `isequal` to the run without it, with every record row
identical. **Across a regrid too (measured in step 8b, once, not in the
suite)**: `refinement_tests.jl`'s moving-mesh run (the adaptive fixture from
a hand-built 127-block mesh, one regrid to 120, `t = 1/10 M`, gate `0.625`)
with the projection on makes `26 = 4 · 6 steps + 1 + 1 regrid` calls — the
post-regrid application included — hands the run's `BoundsAccounting` to
the rebuilt problem, fires on none, and ends `isequal` to the run without
it. It is not in the suite because the suite's budget for this step is one
short run, and the control above is it.

**What it costs** (`hole_runs.jl bounds=cost`, the fixture's `N = 8` mesh,
four threads on the development machine; prediction `0.4 %` of a step):

| | time | per point | per gated point |
|---|---|---|---|
| right-hand side | 35.3 ms | 575 ns | |
| stage limiter, nothing fires | 0.16 ms | 2.6 ns | 31 ns |
| stage limiter, every gated point fires | 0.81 ms | | 158 ns |

`5137` of the `61 440` points (`8.4 %`) are inside the gate `0.8375`. With
four limiter calls against five right-hand sides per step (four stages and
the FSAL refresh a non-trivial step limiter asks for) the projection is
**0.36 % of a step (measured in step 8b; 0.41 % at two threads)** —
the prediction, confirmed. Where it fires everywhere it costs `5.1×` its
healthy call, still under 2 % of a step. It allocates nothing per point
(7 KB per launch, the launch itself).

**The two runs that end** (`hole_runs.jl bounds=damped6,pasted8`, on
Symmetry, one `amddebugq` node at 64 threads; Kerr-Schild `a = 0`, `q = 2`,
`cfl = 1/5`, `chunk = 1 M`, `ε_KO = 1/2`, the layer at the grid rate `1/dt` —
step 5's, which the section asks for by name from step 8c′ **(proposed in
step 8c′)**, since both rows are replays of step 5's table and its autopsy
rebuilds the fatal chunk at that rate). Each is run three times —
without the projection, at the proposed gate `r_1 − 2Gh`, and at the widest
gate the assertion allows, `r_1 − Gh`:

| row | `h` | gate (widest) | ends | projection hits | fatal square root | where the state degenerates |
|---|---|---|---|---|---|---|
| `N = 6` `:damped` | `0.1042` | `0.733` (`0.942`) | step 78 of 81 in the chunk to `22 M` (`t ≈ 21.96`), all three | **0** | the lapse's `√(1 − q_1)` of `−57.59` | `r = 1.27`, the `G` points outside `r_1 = 1.15` |
| `N = 8` `:pasted` | `0.0781` | `0.838` (`0.994`) | step 106 of 107 in the chunk to `18 M` (`t ≈ 17.99`), all three | **0** | the lapse's `√(1 − q_1)` of `−0.1976` | `r = 1.18`–`1.31`, outside `r_1` |

Both square roots are `metric_quantities`' `α = 1/√(−g^{tt})`, reached by
the right-hand-side kernel's evolved-or-layer branch on a stage vector —
for `:pasted` only points at `r ≥ r_1` take that branch at all.

**The prediction was wrong, and in the direction that matters (measured in
step 8b).** `PLAN.md` predicted hits "deep, several `M` before the crash".
There are none, at either gate: the state inside `r_gate` never leaves its
range — in the `N = 6` run the inner layer's error sits at `0.136` from the
first chunk to the last, `min α` there at `0.415`, `max |Π|` at `55.3` —
and all three runs of each row end in the same step with the **same**
`DomainError` argument to the last digit, which is the suite's bitwise
control at run length (`21 M` and `17 M`). What fails is the **evolved shell just outside `r_1`**: its error
against the analytic solution grows from the first chunk — `0.28` at
`1 M`, `1.6` at `10 M`, `3.6` at `19 M` and `22.5` at `21 M` for `N = 6`,
`0.39`, `1.4`, `3.5` and `10.0` at `1, 10, 15, 17 M` for `N = 8` — while the
record's shell rows (`min_α_shell` from `0.604` to `0.539`) move slowly, and
then the metric degenerates within `0.1 M`: in the last eight steps of
`N = 6` the shell's `min α` falls `0.28 → 0.20` at `r = 1.27`, `det γ`
`1.8 → 0.33`, `max |Π|` `1.7e3 → 8.3e3`; the outer layer (`r_gate ≤ r < r_1`)
follows it (`α` `0.35 → 0.27` at `r = 1.08`) rather than leading it.

So both of step 5's failures are **surface failures at `r_1`**, not
interior ones — the kink `PLAN.md`'s finding 1 describes (`ρ_max = 1/dt` is a
paste two cells deep, and a paste is a surface the stencils straddle), and
for `:pasted` literally so. No projection gated below `r_1` can reach them,
by construction — the gate exists so that the clamp's own kink is not
read by an evolved stencil — and none should: the failing points are
evolved by the Einstein equations. The instrument's value here is the
negative result and the validity rows, which show the approach from the
first chunk. **Nothing reached the horizon from the clamp**, because there
was no clamp: in all nine shells `[r_h + kh, r_h + (k+1)h]` the state
difference between the runs with and without the projection is exactly
`0` at every chunk, and the gauge constraint there grows the same in all
three runs (`L∞` in the innermost shell `1.6e−2 → 9.3e−2` over `21 M` at
`N = 6`, `1.0e−2 → 3.1e−2` over `17 M` at `N = 8`). For step 8c: the
thing to calibrate is the transition at `r_1`, and the numbers to watch are
the validity monitor's shell rows and the shell error, not the projection's
hit count.

### The layer against an inexact target (step 8c)

`src/interior.jl` (the `target`), `src/gauge.jl` (`HorizonDissipation`),
`src/driver.jl` (`ρ_max_fixed`), `test/evolution_cases.jl`
(`CurvatureTarget`) and `test/hole_runs.jl calibration`; the rule it
decides is under [The interior](#the-interior-a-pointwise-damping-layer),
"The layer for an inexact target".

**The suite.** **3798 assertions in 13m25 at one thread and 10m12 at
four** on the development machine (Apple silicon, Julia 1.13.0), up from
step 8b's 3723. The 56 new ones cost **11.8 s / 7.2 s**: the profile's
algebra and its refusals and the target's refusals (`0.5 s`), one
right-hand side with the target equal to the background — `isequal` to
the one without it — and one with `KerrSchild(6/5, 0)`, which moves `du` at
every layer point and at no other (`1.7 s`), one with a constant profile
equal to the number — `isequal` again — and one rising to `4`, which moves
`du` exactly at `r_0 ≤ r < r_h` (`2.3 s`), and a `1/20 M` run whose record's
`ρ_max` is the fixed rate, with the two refusals (`7.3 s`). The right-hand
side of a case that uses none of the three is bit for bit the one before
this step (checked once against the tree at `70bd96c`, not in the suite).

**How the study ran.** Every configuration is `hole_fixture` — Kerr-Schild
`a = 0`, `q = 2`, the 120-leaf hierarchy, `h = 5/64` at `N = 8`, `r_1 =
23/20`, `m = 8`, `cfl = 1/5`, `ε_KO = 1/2` unless named — with the range
projection on (`default_bounds`, `default_gate`) and the finder's `M_irr`
every other chunk (`N_ah = 12`, no spin) from the observer rather than
from a case's `Horizon`, which the fixture does not carry — the observer
writes the record's rows too, so that a run that throws keeps its approach
**(proposed in step 8c)**. Screens are `5 M` at
`chunk = 1/2 M`, the survivors `50 M` at `1 M` as step 5's table was.
Thirteen `amddebugq` jobs, **5.6 node-hours**: four screen jobs (11–24
minutes, 138 runs; a fixture screen is 149–357 s at four threads, sixteen
to a node), eight `50 M` jobs (29–36 minutes, 62 runs; one that finishes is
1403–1973 s at eight threads, eight to a node) and the E0 profile pairs (17
minutes; an E0 run on the 512-block mesh is 390–440 s at sixteen threads).
Of the 120 fixture screens, 99 reached `5 M`; which 56 of them went on to
`50 M` is **(proposed in step 8c)** and is listed in the script
(`CAL_LONG`): the whole constant-`ε` scan, the profile on the two thickest
ramps at `4/M` and `1/M` and at the grid rate at `ε_in = 4`, E1 and E2 at
`n_L = 6` (`4/M`), `8` and `12`, every `δ = 4h` survivor, the four
controls, and — added after them — the exact target on the scan's layers.

**The order in the shell, `3/20 M`** (the `G = 2` points outside `r_1`,
`gh_outside_shell_norms`; the fixture's own layer, `r_0 = 2/5`,
`ρ_ramp = 1/2`, and `N = 6, 8, 10`):

| target | `ρ_max` | shell `C_a` L2, `N = 6, 8, 10` | rate L2 | rate L∞ | masked error rate |
|---|---|---|---|---|---|
| exact | `1/dt` | 1.970e−2, 1.118e−2, 6.271e−3 | **2.23** | 1.89 | 2.17 |
| exact | `4/M` | 1.958e−2, 1.120e−2, 6.298e−3 | **2.21** | 1.87 | 2.16 |
| E3 | `1/dt` | 4.238e−2, 3.790e−2, 3.027e−2 | **0.65** | 0.46 | 1.51 |
| E3 | `4/M` | 2.278e−2, 1.235e−2, 6.724e−3 | **2.38** | 2.00 | 2.07 |

Finding 1, measured: pinned at the grid rate, a target whose curvature is
wrong makes a right-hand-side error at the innermost evolved points that
does not converge — `0.65` where the scheme is `2` — and at a physical rate
the same target costs the shell nothing it can measure.

**The scan: the `G`-point shell's `C_a` L2 at `50 M`** (in parentheses the
`5 M` screen of a configuration not run long; `†` the time a run ended —
by `metric_quantities`' `DomainError` on a stage vector wherever the root
cause was read (every `50 M` run, and the screens rerun locally), and in
one screen, E2 at `δ = h` on 6 cells at the grid rate, by the CFL recheck;
rows are the ramp width `n_L` in cells, `nd` the fixture's own layer —
step 5's, a ramp of `4.8` cells in a layer of `9.6`; `ρ(Gh)/ρ_max`, the
relaxation at the innermost evolved stencil's reach, is `0.35` for `nd`,
`0.50, 0.21, 0.10, 0.036` for `n_L = 4, 6, 8, 12`):

| target | `n_L` | `1/dt` (`107/M`) | `10/M` | `4/M` | `1/M` |
|---|---|---|---|---|---|
| exact | `nd` | 0.181 (step 5's layer) | | **0.029** | |
| | 8 | 0.091 | | **0.029** | |
| | 12 | 0.037 | 0.029 | **0.029** | |
| E3 | `nd` | † 7 M | | | |
| | 4 | † 6 M | 0.152 | 0.119 | 0.075 |
| | 6 | † 7 M | 0.144 | 0.111 | 0.080 |
| | 8 | † 15 M | 0.060 | **0.033** | 0.049 |
| | 12 | 0.143 | 0.032 | **0.029** | 0.031 |
| E1 | `nd` | † 2.5 M | | | |
| | 4 | † 2.0 M | (0.598) | (0.275) | (0.264) |
| | 6 | † 3.0 M | (0.258) | 0.115 | (0.291) |
| | 8 | † 4.0 M | 0.106 | **0.038** | 0.087 |
| | 12 | † 18 M | 0.032 | **0.029** | † 15 M |
| E2, `δ = h` | `nd` | † 3.0 M | | | |
| | 4 | † 3.0 M | (0.193) | (0.085) | (0.076) |
| | 6 | † 3.5 M | (0.077) | 0.039 | (0.082) |
| | 8 | † 5 M | 0.036 | **0.029** | 0.044 |
| | 12 | 0.141 | 0.029 | **0.029** | † 6 M |
| E2, `δ = 4h` | `nd` | † 1.0 M | | | |
| | 4 | † 1.0 M | † 2.0 M | † 4.5 M | † 2.5 M |
| | 6 | † 1.0 M | † 6 M | 0.160 | † 2.0 M |
| | 8 | † 1.5 M | 0.123 | **0.041** | † 2.0 M |
| | 12 | † 2.5 M | ‡ | ‡ | ‡ |

`‡`: the target's singular point `(4h, 0, 0)` is a grid point of the
layer (`r_0 = 0.21`), and the run throws at `t = 0` — a configuration, not
a result. The masked error L2 at `50 M` follows the shell: `0.027` on every
bold entry but E1 at `n_L = 8` (`0.032`) and E2 at `4h` (`0.037`); `0.160`
for step 5's layer, `0.057` and `0.029` for the
exact target at the grid rate on `n_L = 8` and `12`. Every run that lives
is **flat from `10 M` on** — the shell's `C_a` and the masked error change in
the third digit over the last forty `M` — so the survivors do not approach
anything; the failures announce themselves, at the grid rate, by a shell
error that grows from the first chunk (E3 at `n_L = 8`: its L∞ `1.4` at
`3 M`, `2.7` at `9 M`, `4.4` at `12 M`, `20` at `15 M`, as step 8b saw step
5's), and at `1/M` on `n_L = 12` not at all in the shell (`min α` there
`0.607` to the end): for E2 the layer's `|Π|` climbs from `1022` to `1233` in
the last three `M`, the projection fires at `t = 5.85 M`, `r = 0.55` (122
hits), and the run ends inside.

**Finding 1's prediction** `n_L ≳ G (10 ρ_max M)^{1/3}` — `6.8`, `9.3`, `4.3`
and `20` cells at `4/M`, `10/M`, `1/M` and the grid rate — is **confirmed at
`ρ_max M = 4` and `10`** (the thinnest ramp holding every target is `8` and
`12`; `6` at `4/M` and `8` at `10/M` cost 1.3–4×), confirmed at the grid rate
(no inexact target holds near the rule's error at `12` cells, where it asks
for `20`), and **wrong at `1/M`**,
where it allows `4.3` and the scan has the 4–6-cell ramps at 2.5× the best
and the 12-cell one failing from inside — which is the lower bound
`ρ_max ≳ 4/M` of the rule.

**The `ε_KO(r)` profile** (`ε_out = 1/2` at and outside `r_h = 2 M`, rising
`C²` to `ε_in` at `r_1` and held inside), the same shell at `50 M`:

| E3, `n_L`, `ρ_max` | constant `1/2` | `ε_in = 1` | `2` | `4` |
|---|---|---|---|---|
| 12, `4/M` | **0.029** | 0.032 | 0.045 | 0.065 |
| 12, `1/M` | 0.031 | 0.032 | 0.036 | 0.047 |
| 8, `4/M` | **0.033** | 0.047 | 0.083 | 0.165 |
| 8, `1/M` | 0.049 | 0.035 | 0.057 | 0.110 |
| 12, `1/dt` | 0.143 | (0.146) | (0.214) | 0.238 |
| 8, `1/dt` | † 15 M | (0.240) | (0.272) | † 22 M |

and at `5 M` over the whole scan (48 screens) the profile is worse than the
constant in 35 — on every ramp that holds, at every `ε_in`. The thirteen
exceptions are transitions that are wrong already: eight at the grid rate
(`n_L = 4, 6` at every `ε_in`, taking a shell error of `0.43–0.60` to
`0.27–0.42`, and `8`, `12` at `ε_in = 1`, by 2–4 %), and five on the
marginal ramps (`n_L = 6` at `ε_in = 1` at every physical rate and at
`ε_in = 4` at `1/M`; `8` at `1/M`, `ε_in = 1`: `0.046 → 0.031`).
Step 8a's one-dimensional model said the profile cuts what crosses the margin
4–15×; what it measurably does in 3D to a smooth transition is add the
Kreiss–Oliger term's own `O(ε h^{q+1} ∂^{q+2} u)` where the solution is at its
steepest, and nothing a smooth target sends needs cutting. The masked
error says the same (worse in 37 of the 48 screens), and no survival
changes where the ramp is right; the drift of `h_tt` at the horizon falls at
`ε_in = 4` (`3.0e−3 → 2.2e−3` at `n_L = 12`, `4/M`), which is the
dissipation damping the solution and not a better one.

**E0: the hard step** (`:pasted` onto `KerrSchild(6/5, 0)`, 8a's uniform
512-block mesh at `h = 5/64`, each run against `:pasted` onto the exact
target with the same `ε_KO`; `A_0/A` is the largest `|δh_ab|` in the first
shell outside the horizon, `[r_h, r_h + h)`, over the step's `|δh| = 0.348`
at `r_1`, `10.9` cells below it):

| `ε_KO` | reached | `A_0/A` at `1 M` | `2 M` | `5 M` | e-folds/cell outside | masked error L∞ at the end | min `α` in the shell |
|---|---|---|---|---|---|---|---|
| 1/4 | † 1.0 M | 4.2e−4 | | | 1.70 | 1.1e3 | 0.354 |
| 1/2 | † 1.75 M | 2.1e−4 | 2.5e−3 (`1.5 M`) | | 0.92 | 1.9e3 | 0.287 |
| 1 | `5 M` | 3.0e−4 | 4.2e−3 | **2.2e−2** | 0.39 | 244 | 0.369 |
| 1/2 → 2 | `5 M` | 3.5e−4 | 2.6e−3 | 1.6e−2 | 0.37 | 8.1 | 0.576 |
| 1/2 → 4 | `5 M` | 5.1e−4 | 3.5e−3 | 9.5e−3 | 0.35 | 6.3 | 0.576 |

The three references reach `5 M` with a masked error L2 of `0.009–0.017`; the
e-folds of the two that end are of their last peaks, before the slow modes
arrive. Against step 8a: its ripple at depth 8, `ε_KO = 1/2`, reached the
first shell at `9.9e−3` by `2 M`, and its measured `0.40–0.47` e-folds per
cell of depth continued to `10.9` give `2.6–3.1e−3` — E0 at `ε_KO = 1/2` has
`2.5e−3` at `1.5 M`, the step behaving as a broadband source of the same
content; the frozen-coefficient `e^{−d/ℓ_max(r_1)}` is `5e−3`, `3e−5` at
`ε_KO = 1/2, 1`. The transmitted fraction is still rising at `5 M` —
the source never stops — and has reached the level 8a's long-time
one-dimensional rates give for `10.9` cells at `ε_KO = 1`, `1.4–2.7e−2`.

**The controls, with the projection on**, against step 5's table:

| run | step 5 | step 8c | at the end: masked error L2, shell `C_a` L2, drift, `M_irr` |
|---|---|---|---|
| `:damped`, `N = 8` | `50 M`, `1.604e−1` | **`50 M`** | 1.60e−1, 0.181, 4.08e−3, 0.9955 |
| `:damped`, `N = 6` | `21 M` | `21 M` | in the same chunk, with step 8b's `DomainError` to nine digits (`−57.5936359…`): 8b's ran on an AVX-512 node and this on an AVX2 one |
| `:pasted`, `N = 8` | `17 M` | `17 M` | step 8b's `DomainError`, `−0.19763566226561102`, to the last digit (both AVX2 nodes) |
| `:frozen`, `N = 8` | `13 M` | `13 M` | |

Zero projection hits in all four, and in every run of the study but two
families: E2 at `δ = 4h` on the layers its singular point comes within a
cell of (`1652` hits from `t = 0.046 M` at `r = 0.406` on the fixture's own
layer, `26124` from `t = 0.005 M` at `r = 0.398` on `n_L = 12`, both at the
grid rate and both ending in `1–2.5 M`), and E2 at `δ = h`, `n_L = 12`,
`1/M` above. The projection is not what any surviving run leans on.

**The horizon.** The finder's `M_irr` is `1.00029` at `t = 0` (its
truncation at `N_ah = 12`, `h = 5/64`) and `0.9972 ± 0.0002` at `50 M` on
every run at `4/M` or `10/M` with `n_L ≥ 8`, exact target or wrong — the
horizon's area does not see the target either — against `0.9955` for step
5's layer; the gauge drift of `h_tt` at the horizon is `2.7–3.3e−3` against
step 5's `4.08e−3` (which this study reproduces to three digits).

**The recommendation: proceed to steps 8d–8f, not to 8g (proposed in step
8c).** Forty-two of the 52 runs with a wrong target that went to `50 M` got
there: 39 of the 42 at a physical rate (the three that did not are `1/M` on
12 cells and `δ = 4h` at `10/M` on 6), 3 of the 10 at the grid rate, all on
12 cells and at five times the error. On the rule's layers (`ρ_max = 4/M`, `n_L ≥ 8`) a 20 % mass error, a
curvature error of `4/M²` and a displacement of `h` each leave the exterior
exactly where the exact target leaves it — the shell's `C_a` `0.029–0.039`,
the masked error `0.027–0.032`, `M_irr` `0.9972`, flat for forty `M`, and no
projection hit — which is what step 8e's fitted target needs: it will be
wrong by less than any of these. The layer does not need the target to be a
solution or to be close; it needs it smooth (E0's step gets out at a
percent and ends the run at `ε_KO ≤ 1/2`), a metric (finding 2; E3's other
sign is not one), regular on the layer (E2 at `4h`, `n_L = 12`), and centered
to about a cell (E2 at `4h` holds near the rule's error on one layer
only). Those are the three
things for 8d and 8e to guarantee — the tracked center to `≲ h`, the fit
valid at every swept point and smooth in time as well as in space, as 8e's
linear interpolation between two fits already is — and nothing measured
here says excision is needed. What this does *not* show: the rule at `q = 4`,
at the finer `h` the spinning holes need (the proposed scaling of rule 3),
or on a surface that is not a sphere; step 8f's matrix is where those are
measured.

### The default rate (step 8c′)

`src/interior.jl` (`hole_mass`), `src/driver.jl` (`default_relaxation_rate`
and the rule in `evolve!`), `test/driver_tests.jl`, `test/interior_tests.jl`
and `test/hole_runs.jl`; the decision — `ρ_max = 4/M` for every run,
**decided 2026-09-23** — is under [The
interior](#the-interior-a-pointwise-damping-layer), "The profiles and their
parameters". No long run: the default's `50 M` number is step 8c's exact
target at `4/M` on the fixture's own layer (masked error L2 `0.027`, shell
`C_a` L2 `0.029`, `M_irr` `0.9972`), cited beside step 5's table.

**The suite.** **3857 assertions in 14m33 at one thread and 10m30 at four**
on the development machine (Apple silicon, Julia 1.13.0), shared with
sibling agents (load average 5–9), against **3798 in 13m45** at one thread
for the tree before this step, measured the same afternoon (step 8c
recorded 13m25 and 10m12). The 59 new assertions are `interior_tests.jl`'s
`hole_mass` through every wrapper and the default's value in the case's type
(46, `0.4 s`), `driver_tests.jl`'s testset for the grid-rate option, the
default's refusal and a Minkowski case through `evolve!` with no rate
(11, `6.2 s` / `3.0 s`), and one more row each in the record's claim, which
now covers `t = 0`, and in the variants'. What the time went on is the
variants: **`26.4 s` → `82.3 s` at one thread** (`30.6 s` at four), the two
runs to `1/2 M` the saturated residual needs — so `driver_tests.jl` is
`3m00` / `1m08` against `1m56` before.

**The claims that encoded the grid rate, amended and not loosened** (a
decided change of the spec): the record's `r.ρ_max * r.dt ≈ 1` is now
`r.ρ_max == 4/M` on every row, `t = 0` included, with `ρ_max_factor = 1`
giving `ρ_max · dt ≈ 1` in a testset of its own; the variants' residual
claim is made at `2/ρ_max = 1/2 M` and their shell claim at `1/10 M`, for
the reason under step 5's table. No other assertion changed. **`ρ_max_factor
= 1` reproduces step 5's `t = 1/10 M` row to every digit this document
records** (measured once, not in the suite): `:damped`'s residual
`6.2042e−2`, shell `C_a` L2 `1.1054e−2` and L∞ `6.956e−2`, masked error
`2.2688e−3`; `:pasted` and `:frozen` do not read the rate and are
bit-identical. The calibration's `:grid` rows, now asked for by name,
reproduce step 8c's sweep (`1.118e−2` and `3.790e−2` in the shell at
`N = 8`), and the `bounds` section's `damped6` replay step 8b's layer and
shell errors (`0.136`, `0.28` at `1 M`).

**Every suite number that moved**, before (the grid rate) and after (the
default), from the two suites' logs:

| claim (file) | grid rate `1/dt` | default `4/M` |
|---|---|---|
| order sweep, `3/20 M`, rates L2 / L∞ / `C_a` / residual (`driver`) | 2.165 / 2.740 / 2.008 / 2.806 | 2.157 / 2.741 / 2.006 / 2.032 |
| layer residual at `N = 6, 8, 10` (`driver`) | 1.380e−1, 6.210e−2, 3.288e−2 | 1.433, 7.965e−1, 5.075e−1 |
| unmasked L2 / L∞ at `N = 8` (`driver`) | 5.052e−3 / 6.210e−2 | 3.536e−2 / 7.965e−1 |
| `:damped` at `1/10 M`: residual, shell `C_a` L2, masked L2 (`driver`) | 6.204e−2, 1.1054e−2, 2.2688e−3 | 5.513e−1, 1.1062e−2, 2.2588e−3 |
| `:frozen`/`:damped` residual (`driver`) | 10.9 at `1/10 M` | 1.22 at `1/10 M`, **2.52 at `1/2 M`** |
| gauge drift rate over `1/5 M` (`driver`) | 1.9512409e−3 | 1.9512407e−3 |
| `Float32` / `Float64`, masked L2 (`type`) | 2.268762e−3 / 2.268849e−3 | 2.258749e−3 / 2.258833e−3 |
| `Float32` / `Float64`, residual (`type`) | 6.204157e−2 / 6.204212e−2 | 5.512611e−1 / 5.512648e−1 |
| adaptive static run at `3/20 M`: masked L2, `C_a` L2, residual, `τ_max` (`refinement`) | 1.929e−3, 1.719e−3, 0.799, 0.5295 | 1.881e−3, 1.711e−3, 6.001, 0.5301 |
| the regrid that moves the mesh: masked L2, `τ_max` (`refinement`) | 1.360e−3, 0.5312 | 1.341e−3, 0.5314 |
| bounds control at `3/20 M`: `max_Π_shell`, `min_α_layer` (`bounds`) | 2.6345, 0.410764 | 2.6339, 0.410764 |

The `Π` post-pass is unchanged — its test sets `ρ_max = 10` on the problem
itself, not through the driver — and the horizon claims pass as they did
(they are asserted to `1e−3` against Kerr and not logged). Outside the
layer the masked errors moved by at most 2.5 %, every one of them down, and
the constraints by under 0.5 % either way; the residual, the layer's
distance from the truth, grew seven to fifteen times at these times and
twenty-three times once saturated, as `τ/ρ_max` against `τ · dt` says it
must.

**The guard `ρ_max · dt ≤ 1` at the default** (the refusal of a fixed rate
above the grid rate, now the default's too): `4/M · dt`, over the chunks
and the `t = 0` row, is `0.040–0.047` on the fixture at `N = 8`,
`0.036–0.037` at `N = 10` and `0.057–0.062` at `N = 6`, and `0.067–0.092`
on the refinement's fixture at `h = 5/32` — eleven to twenty-eight times
inside it, on every hole the suite evolves.

**What the adaptive fixture's layer does at `4/M` (measured in step 8c′,
once, not in the suite).** It is six cells of `h = 5/32` from `r_1 = 5/4`
down to `r_0 = 3/10`, where `|Π|` reaches `131`, and at the grid rate
(`60/M` there) its residual saturates at `0.80` by `1/10 M`; at `4/M` it
rises through `6.0` at `3/20 M` and saturates at **`14.7`** by `1 M`, while
the layer stays a metric — `min α` there `0.3676` and `min det γ` `2.61` to
`2 M` at both rates — and the evolved region is better at `4/M` than at the
grid rate throughout (masked L2 `6.03e−3` against `7.13e−3`, `C_a` L2
`1.99e−3` against `2.55e−3`, at `2 M`). Step 8c's rule asks for a ramp of
at least `4G = 8` cells at `4/M`, and this layer is under it; the default
holds it anyway over the suite's times. A case whose layer is both coarse
and deep is where the default's larger residual is largest, and step 8d's
tracked geometry is the next place it is read.

**`hole_runs.jl` at the new default.** `order`, `long`, `charts`,
`indicator`, `horizon` and `leakage` run at `4/M` without edits; their
recorded tables are at the grid rate and are annotated as such where they
stand, and none has been re-run. `calibration`'s `:grid` rows pass
`ρ_max_factor = 1`, as `PLAN.md` asks, and so do the `bounds` section's two
replays of step 5's failing rows, whose autopsy rebuilds the fatal chunk at
the grid rate by construction **(proposed in step 8c′)**. `leakage` is the
one section whose rerun no longer reproduces its table — step 8a's layer
was at `1/dt` — and `test/dispersion.jl`'s one-dimensional model of it stays
at `1/dt` too, as the model of the runs it was compared against.

### The tracked geometry (step 8d)

`src/tracking.jl`, the second half of `src/interior.jl`, the tracked chunk
in `src/driver.jl`, and `test/tracking_tests.jl`; the design is under [The
interior](#the-interior-a-pointwise-damping-layer), "The tracked geometry".
No long run is required by the step, and none is in `hole_runs.jl`.

**The suite.** **4410 assertions in 15m28 at one thread and 11m21 at
four** on the development machine, shared with sibling agents (load
average 6–7), against step 8c′'s 3857 in 14m33 and 10m30. The 553 new ones
are `tracking_tests.jl`'s, **`54.7 s` at one thread and `32.4 s` at
four**: the host-side part `2.3 s` (464 assertions, 415 of them the seed on
the charts' quartic), one right-hand side per variant and one find `12 s`,
and the two runs — the lost track with the trigger, and the tracked hole
against the sphere — `40 s` / `18 s`, most of it the compilation of the
tracked geometry's kernels (`1.3 s` each once the fixture's are compiled).
No other testset's claims changed; the protocol change touched only
`interior_tests.jl`'s one call of `in_layer`.

**The acceptance list, with its numbers** (all measured in step 8d on the
development machine, Julia 1.13):

| claim | number |
|---|---|
| real harmonics against `sYlm` at 200 random directions, largest error of the function | `3.7` / `6.4` eps at `lmax = 4` / `8` (`Float64`), `1.6` / `5.3` eps (`Float32`) |
| the same against `ash_evaluate` on `EquiangularGrid(6)` | below `16` eps |
| truncation of the analytic seed, Kerr-Schild `a = 9/10`, largest radius error | `1.2e−3`, `8.1e−6`, `5.5e−8`, `3.8e−10` at `lmax = 4, 8, 12, 16` |
| the same, harmonic Kerr `a = 9/10` | `4.6e−2`, `7.2e−3`, `1.1e−3`, `1.7e−4` (`2.4`, `0.37`, `0.057`, `0.009` cells at `h = 5/256`) |
| a fitted sphere against step 5's `Interior`, one right-hand side and the paste | `isequal` for `:damped`, `:frozen`, `:pasted` |
| one find of the fixture's initial data (`N_ah = 12`): the tracked center | `1.4e−5 M` from the analytic one, **`track_offset = 1.8e−4` cells** |
| the same: `r_min`, `r_max` about the found origin, against `r₊ = 2` | `2.00043`, `2.00068` |
| a find with `r_min` moved by `G h` | refused (`ArgumentError`); by `G h/4`, accepted |
| a finder disabled after `1/10 M` (chunk `1/20`, `max_misses = 2`) | `:coasting` at `3/20 M`, `TrackLostError` at `1/5 M` carrying the five rows, "0.1 M before t = 0.2" |
| the lapse-collapse trigger at `α_trigger = 1` with `every = 100` | every find after the first forced, `track_trigger = true` on those rows |
| a `:damped` tracked run of the fixture to `0.15 M` against step 5's sphere with the same layer (`r_1 = 2 − 10 h`, `n_L = 8`, `ρ_ramp = 1`): masked error L2 | `3.110274e−3` against `3.110271e−3` |
| the same: the `G`-point shell's `C_a` L2, the layer residual | `1.0745462e−2` against `1.0745448e−2`; `0.203` against `0.206` |
| the same: projection hits, `track_offset`, prediction error, re-samples | `0` and `0`; at most `3.3e−4` cells; `1.2–1.8e−4` cells per find; none |
| `margin_efolds` on the sphere `r_h = 2`, `m = 8`, `h = 5/64`, `ε_KO = 1/2` | `1.810` (`q = 2`), `2.073` (`q = 4`) — `test/dispersion.jl`'s `1.81`, `2.07` |
| the same on the tracked fixture, `m = 10` through the blocks the path crosses | `1.45` (a uniform `5/64` would give `2.68`) |
| the footprint guard on an `a = 9/10` shape, 2000 random footprints | agrees with the definition on every one |
| cost per point: `shape_series` at `lmax = 4`, `8`; one analytic `u_exact` | `21 ns`, `79 ns`; `96 ns` |
| cost of the tracked geometry: one right-hand side on the fixture, a `0.15 M` run at four threads | `+1.1 %`; `3.16 s` against `3.00 s` (`+5 %`) |
| host side per chunk: `fitted_interior`, `margin_efolds` | `0.04 ms`, `0.6 ms` |

**The same comparison to `5 M`** (`julia --project=. --threads=4
test/hole_runs.jl tracked`, the section added in step 8d and not in the
default list; chunks of `M/4`, the finder every chunk; measured once, not in
the suite). The tracked layer follows a found horizon that shrinks with the
solution's own drift — `r_in` from `2.00042` to `1.99564` by `5 M`, as
`M_irr` falls from `1.00029` to `0.99760` — and stays step 5's layer to a
percent:

| `t/M` | masked L2, tracked | sphere | shell `C_a` L2, tracked | sphere | `M_irr`, tracked | sphere | `track_offset` (cells) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 1.0796e−2 | 1.0796e−2 | 1.0002880 | 1.0002880 | 1.8e−4 |
| 1 | 1.4002e−2 | 1.3991e−2 | 1.2401e−2 | 1.2400e−2 | 0.9991516 | 0.9991516 | 7.4e−4 |
| 2 | 1.4177e−2 | 1.4147e−2 | 1.4550e−2 | 1.4508e−2 | 0.9983334 | 0.9983334 | 2.9e−4 |
| 3 | 1.6146e−2 | 1.6090e−2 | 1.8469e−2 | 1.8408e−2 | 0.9980183 | 0.9980182 | 7.4e−4 |
| 4 | 1.8755e−2 | 1.8595e−2 | 2.1214e−2 | 2.1107e−2 | 0.9977899 | 0.9977897 | 1.8e−3 |
| 5 | 2.0436e−2 | 2.0233e−2 | 2.2995e−2 | 2.2864e−2 | 0.9976053 | 0.9976048 | 3.0e−3 |

The difference grows to `1.0 %` in the masked error and `0.6 %` in the
shell by `5 M`, in the tracked run's disfavour, and `M_irr` agrees to
`4e−7`; neither run's projection fires, nothing is re-sampled (the core
surface moves by `5e−3 M`, a sixteenth of a cell, in `5 M`), and the
margin is `1.47` e-folds at the end. **`track_offset` grows, `3.0e−3` cells
at `5 M`** — the velocity estimate is two finds' difference, and the
finder's `1e−4`-cell noise over a chunk of `M/4` is a velocity of `4e−4`
cells per `M` that the prediction integrates — while the per-find
prediction error stays below `6.8e−4` cells. A static hole's track would be
better held without a velocity at all; a moving one's is G5's question,
and a velocity smoothed over several finds is the first remedy to try
**(proposed in step 8d)**.

### The fitted target, host half (step 8e-i)

`src/fit.jl`, `hole_velocity` in `src/interior.jl`, and `test/fit_tests.jl`;
the design is under [The interior](#the-interior-a-pointwise-damping-layer),
"The fitted target". No kernel evaluates the fit yet (8e-ii), and no long
run is required; the explorations behind the harmonic and boosted rows are
host-side and a few seconds each.

**The suite.** **4578 assertions in 15m09 at one thread and 10m52 at
four** on the development machine (load average 6–8, shared with sibling
agents), against step 8d's 4410 in 15m28 and 11m21. (A first pair on the
tree before the documentation commit measured 14m59 at one thread and
37m25 at four — the latter under a load of 13, user time 18m29, its excess
in `constraints_tests.jl`'s and `interface_tests.jl`'s compilation-heavy
testsets, none of which this step touches.) The 168 new assertions are
`fit_tests.jl`'s 146 — **`12.6 s` at one thread and `12.9 s` at four**, the
tracked run it fits shared with `tracking_tests.jl` (`tracked_fixture_run`),
whose `The tracked hole` testset is `40.2 s` / `18.0 s` with the run in it —
and `interior_tests.jl`'s 22 for the boost sign (`0.4 s`).

**The acceptance list, with its numbers** (all measured in step 8e on the
development machine, Julia 1.13, `Float64` unless said):

| claim | number |
|---|---|
| the fit's solid harmonics at unit vectors against `shape_series`, and `S(2ξ) = 2^l S(ξ)` | `1.8`, `14` eps at `L = 4, 8` (`Float64`); `1.3`, `16` eps (`Float32`) |
| a real field from random coefficients through `ash_evaluate`, recovered by least squares on `EquiangularGrid(L)` | cond `2.2`, `2.9` at `L = 4, 8`; the coefficients to `2.9–4.6` eps, both precisions |
| the whole ansatz from random coefficients on a non-spherical surface, recovered by `solve_fit` | cond `99`, `1.5e3`, `6.0e3` (`L, cont = 4,1; 4,2; 6,2`); coefficients to `0.03–0.19 κ eps` |
| the analytic sampler's `∂_r h` against `background_state`'s gradient on the fixture's surface | `1.6e−8` |
| the chain rule against the converted samples differenced along the rays | `5.7e−10` (slopes), `7.3e−11` (curvatures) |
| Kerr-Schild `a = 0`, `cont = 1, 2`, the state on the fixture's offset surface (`r_1 = 1.22`) | `1.10` at `L = 1`; `5.1e−15`–`3.1e−13` at `L = 2, 4, 8` |
| the same from the state sampler on the exact solution, `L = 8`, `cont = 1` | `2.2e−4` (`N = 8`), `8.3e−6` (`N = 16`): rate `4.65` (`3.9` from 16 to 32); samples `2.6e−4`, `9.2e−6` |
| the angular mean of Kerr-Schild `g_ab` on `r = 1.15 M` | `g_tt = +0.739`, eigenvalues `(0.739, 1.580, 1.580, 1.580)`, signed `α = −0.86`: **not a metric** |
| the same mean of `(log α, β^i, γ_ij, Π_ab)`, reassembled | `α = 0.60`, `det γ = 3.94`: a metric |
| the sweep, Kerr-Schild `a = 0`, `L = 8` (`1225` points) | valid, `min λ(γ) = 1`, `min α = 0.527` / `0.480` (`cont = 1` / `2`), no hit |
| the sweep, Kerr-Schild `a = 9/10` on its oblate surface (`h = 5/128`, `m = 8`) | valid at `L = 4, 8, 12, 16`, both orders, `min λ ≥ 0.945`, no hit |
| its truncation, the state on the surface, `cont = 1` (`cont = 2`) | `0.26` (`0.32`), `0.051` (`0.075`), `9.1e−3` (`1.4e−2`) at `L = 4, 8, 12` |
| harmonic `a = 9/10`, `h = 5/256`, `m = 4`: `cont = 1` | valid for `L ≥ 8` (`min λ = 2.27`, value residual `1.9e−3` at `L = 8`); invalid at `L = 4` |
| the same, `cont = 2` | **invalid at every `L` from 4 to 16** (`min λ(γ) = −73`, 953 of 1225 points at `L = 8`); refused by `build_fit` |
| the same at `h = 5/512` | `cont = 1` valid from `L = 4`; `cont = 2` first valid at `L = 16` |
| harmonic `a = 9/10`, `m = 8`, `h = 5/256` | refused at the sample: the equatorial offset radius `0.843` is inside the disk |
| harmonic `a = 9/10`'s samples against `default_bounds` | `max |(α/√γ)Π| = 366/M` against `K_max = 100/M`, at 17 of 153 points |
| a boosted hole's shift, `boost(Harmonic(1, 0), 0.3 x̂)`, `L = 8`, `cont = 1`: without and with `shift_constant` | the state on the surface `1.1e−3` → `1.7e−5`; the shift's slope rows `22×` → `1.4e−4` of their scale |
| the evaluator against the least-squares model at the collocation points | `28` eps (`Float64`), `25` eps (`Float32`) of the largest variable; the projection the identity bit for bit |
| at the center | `β = 0` exactly, a valid metric; `FitParams` `isbits`; `fit_state` allocation-free |
| the tracked `:damped` run's state at `0.15 M` fitted, against the analytic solution's fit | `0.031` on the surface to `0.016` at the center; samples `0.034` off, masked error `0.035` L∞ / `3.1e−3` L2 |
| `build_fit` at `L = 8`: analytic `cont = 1`, `2`; state sampler | `3.6 ms`, `5.2 ms`; `7.0 ms` (`0.4–0.6 ms` at `L = 4`, `20–25 ms` at `12`) |
| `fit_state` per point at `L = 4, 8, 12`, `cont = 1` (`cont = 2`) | `0.64` (`0.88`), `2.2` (`2.9`), `4.3` (`6.0`) µs, the recurrence `20`, `85`, `165 ns` of it; one `u_exact` `91–96 ns` |
| `boost(KerrSchild(1, 0), 0.3 x̂)` at `t = 1` | `|g_tt| = 1.0e17` at `x = −0.3`, `4.9` at `+0.3`; `hole_velocity = −v` |

### The fitted variant, kernel half (step 8e-ii)

The cache, the `:fitted` branch, the driver's flow and the initial data;
the design is "The fitted target", pieces 8–12, and the decisions of the
review of 8e-i are marked there. The long rows are `hole_runs.jl fitted`
(`fitted=fixture`, `boosted`, `harmonic`).

**The suite.** **4616 assertions in 14m21 at one thread and 10m41 at
four** (load 6–8, rising to 13 during the four-thread run), against 8e-i's
4578 in 15m09 and 10m52. The 38 new claims are `fit_tests.jl`'s "The fitted
variant" — `47.7 s` at one thread and `39.5 s` at four, of which the
bit-identity testset is `6.3 s` and the fixture's `:fitted` run to `0.15 M`
with its one `Float32` chunk `41.5 s`: the step's one short run, and the
`Float32` chunk beside it — plus the 8e-i testsets' sixteen for the shift
constant's two paths.

**The acceptance list, with its numbers** (measured in step 8e):

| claim | number |
|---|---|
| one right-hand side, `:fitted` against `:damped` on the same geometry and rate, outside the offset surface | bit for bit at all 45 545 points; the core `−ρ_max (u − u_fit)` exactly; the layer's difference `ρ (u_fit − u_exact)` to `1.4e−14` |
| the cache against the host evaluator | bit for bit |
| a right-hand side, `:fitted` and `:damped` layers, fixture, one thread | `111 ms`, `107 ms` (again `106.7`, `107.6`; in the suite `114`, `118`) |
| a fill of the cache from one fit, from two | `44 ms`, `75 ms` — `0.4`, `0.7` of a right-hand side, once a chunk |
| the fixture `:fitted` to `0.15 M`: masked error L2, L∞, against `:damped` | `1.33e−2`, `0.177` against `3.11e−3`, `0.035` (`4.3×`, the initial data's kink; asserted `≤ 6×`) |
| the same: `fit_valid`, projection hits, `track_offset`, fit failures, mid-chunk refills | every row; `0`; below `3.3e−4` cells; `0`; `0` |
| the same to `1 M` | `1.74e−2` against `1.40e−2` (`1.24×`); the fit below the core surface `1.56e−2`, `cont = 2` `1.66e−2` |
| one chunk at `Float32` (samples in `Float64`) | masked error `9.76982e−3` against `9.76988e−3` (`7e−6`) |
| the moving seed, `boost(Harmonic(1, 0), 0.3 x̂)`, 848 blocks, `h = 5/128`, `0.1 M` | every find succeeds; track `−0.01492`, `−0.02985` against `−0.015`, `−0.03`; `track_offset ≤ 3.9e−3` cells; two pieces a chunk, one refill; masked error `3.26e−2` against `:damped`'s `2.03e−2` |
| harmonic `a = 9/10`, `m = 4`, `h = 5/256`, 2472 blocks: the initial data | finite and a metric at all 1 265 664 points; `min det γ = 0.648`, `min α = 0.208`, `min λ(γ) = 0.5` |
| the same: one right-hand side (four threads) | `0.71 s`, finite, `max |du| = 1.1e9` at the offset surface over the disk |
| the same: the run | ends in its first chunk (`2.5e−3 M`), a degenerate metric at the equatorial offset surface |
| the kink at the first evolved point, harmonic `a = 9/10`, `L = 8`: axis, 45°, equator (analytic second difference) | `9.5e5` (`416`), `1.8e7` (`111`), `9.4e7` (`6.4e8`); with `Π̃ = (α/√γ)Π`: `2.6e4`, `7.5e5`, `7.1e7`; with `Π̃` at `L = 12`: `39`, `1.3e4`, `7.4e7` |
| the same kink, Kerr-Schild `a = 0` fixture, `a = 9/10` equator, harmonic `a = 7/10` 45° | `7.7` (`15.9`), `46` (`61`), `6000` (`334`) |

### The generic interior: the measurement matrix (step 8f)

`test/hole_runs.jl generic` (its header says how the rows are grouped and
run), the decisions of the 8e hand-over it took (below), `fit_tilde` in
`src/fit.jl`, and `handover`, `target_source` and the `:fitted` mesh cycle
in `src/driver.jl`; the design it decides is the summary at the head of
[The interior](#the-interior-a-pointwise-damping-layer). Every row is `q =
2`, `cfl = 1/4` (`1/5` on the moving rows), `ε_KO = 1/2`, `ρ_max = 4/M`
unless named, the ramp by step 8c's rule (`n_L = 8`), the finder every
chunk with the Korzyński spin; five Symmetry jobs, measured once
(2026-09-23/24), on `amdq` (`ks0`, `ks9`, `h7`) and in `amddebugq` hours with a
deadline (`harm`, `boost`: `budget=3300`; `h7` reached its deadline at
`6 M` there and was rerun to `10 M` on `amdq`; the boosted rows ran a
second and a third time, at `cfl = 1/5` and with the moving-step sizing,
after the CFL recheck stopped them at `1.5 M` and `1 M`).

**The suite.** **4632 assertions in 20m59 at one thread and 12m22 at four**
on the development machine (Apple silicon, Julia 1.13.0) under a load of
8–15 from other work (a four-thread run the evening before, on the same
tests without the moving-step change, measured 11m22 at a load of 5), against
step 8e's 4616 in 14m21 and 10m41. The sixteen new assertions are
`fit_tests.jl`'s: three for `Π̃`'s chain rule and round trip in "the static
hole's fit is the hole" (host-side, below a second), and "the snapshot, the
hand-over and the mesh cycle (step 8f)", **`25.9 s` at one thread and
`16.6 s` at four** — the snapshot cache and one right-hand side, the
hand-over run to `0.15 M`, and the adaptive fixture's mesh cycle as
`:fitted` with one regridding chunk. One assertion changed its expression
and not its claim: the evaluator's bit-identity with the reassembly now
reassembles with the fit's own `tilde` flag, the new default.

**The matrix** (the last row of each run: masked error L2 and L∞ over the
evolved region, the `G`-point shell's `C_a` L2 above the offset surface,
`C_a` L2 in step 8a's shells `[r_h + kh, r_h + (k+1)h)` outside the tracked
horizon, the layer residual — against the truth for the analytic variants,
against the target for `:fitted` — the drift of `h_tt` at the horizon's
evolved points, the finder's `M_irr`, `J` and `M_ch`, the largest track
offset from the analytic center in cells; no row fired the range projection
unless its hits are given, and every fit of every `:fitted` row was valid):

| case | row | reached | masked L2 / L∞ | shell `C_a` | `C_a` at `r_h` + 0, 2, 4 `h` | residual | drift | `M_irr` / `J` / `M_ch` | offset |
|---|---|---|---|---|---|---|---|---|---|
| Kerr-Schild `a = 0`, fixture, `m = 10` | `:damped` (control) | `50 M` | `2.36e−2` / `0.177` | `2.62e−2` | `4.2e−3`, `2.7e−3`, `1.9e−3` | `0.61` | `3.1e−3` | `0.99716` / `3.6e−6` / `0.99716` | `7.1e−3` |
| | `:fitted`, analytic data to the core surface | `50 M` | `4.59e−2` / `0.562` | `7.45e−2` | `4.8e−3`, `3.1e−3`, `2.2e−3` | `5.7` | `3.4e−3` | `0.99676` / `3.1e−6` / `0.99676` | `8.3e−3` |
| | `:fitted`, the decided data (fit below `r_1`) | `50 M` | `4.59e−2` / `0.562` | `7.45e−2` | the same | `5.7` | `3.4e−3` | `0.99676` | `7.8e−3` |
| | `:fitted`, fitting `Π` (not `Π̃`) | `50 M` | `4.53e−2` / `0.553` | `7.34e−2` | the same | `5.8` | `3.5e−3` | `0.99676` | `8.2e−3` |
| | the snapshot target | † `8 M` | `0.230` / `4.56` at `8 M` | `0.456` | `6.8e−3`, … | `190` | | `0.99805` | |
| | the finder's Kerr target (`M_ch = 1.00029`, `J = 2e−6`) | `50 M` | `2.36e−2` / `0.177` | `2.62e−2` | as `:damped` | `0.62` | `3.1e−3` | `0.99716` | `7.1e−3` |
| | hand-over: `:damped` to `5 M`, then `:fitted` | `50 M` | `4.59e−2` / `0.562` | `7.45e−2` | as `:fitted` | `5.7` | `3.4e−3` | `0.99676` | `1.0e−2` |
| Kerr-Schild `a = 9/10`, `h = 5/128`, 1632 blocks | `:damped`, `m = 5` (control) | `20 M` | `4.77e−3` / `4.54e−2` | `2.49e−3` | `1.1e−3`, `8.8e−4`, `9.5e−4` | `0.52` | `3.8e−3` | `0.84744` / `0.8978` / `0.9994` | `1.7e−3` |
| | `:fitted`, `m = 5`, `L = 12` | `20 M` | `0.196` / `2.46` | `0.100` | `2.5e−2`, `2.4e−2`, `1.5e−2` | `6.1` | `4.6e−2` | `0.8436` / `0.840` / `0.980` | `7.6e−3` |
| | `:fitted`, `m = 8`, `L = 12` | `20 M` | `0.258` / `4.32` | `0.109` | `9.7e−3`, `8.4e−3`, `8.4e−3` | `12.9` | `6.7e−2` | `0.8481` / `0.906` / `1.002` | `6.1e−3` |
| harmonic `a = 0`, `h = 5/128`, 960 blocks | `:damped` (control) | `10 M` | `0.234` / `7.9` | `7.5e−3` | `1.4e−3`, `9.5e−4`, `7.4e−4` | `41` | `1.3e−3` | `0.99754` | `2.6e−3` |
| | `:fitted` | `10 M` | `0.247` / `7.3` | `7.5e−3` | `1.8e−3`, `8.3e−4`, `5.9e−4` | `95` | `1.3e−3` | `0.99741` | `2.6e−3` |
| harmonic `a = 7/10`, `m = 4`, `lmax_shape = L = 12` | `:fitted`, `h = 5/256`, 2472 blocks | `10 M` | `3.60` / `113` | `0.168` | `3.7e−2`, `3.1e−2`, `2.9e−2` | `284` | `2.1e−2` | `0.9230` / `0.7073` / `0.9994` | `1.5e−3` |
| | `:fitted`, `h = 5/128`, 512 blocks, analytic data to `3h` | † `0.5 M` | `20` / `846` | `0.38` | | | | | |
| | `:fitted`, `h = 5/128`, the decided data | † `3.5 M` | `55` / `2570` | `2.35` | | | | `0.9187` | |
| `boost(Harmonic(1, 0), 0.3 x̂)` from `x = 0.75`, `h = 5/128`, 1128 blocks, `cfl = 1/5` | `:fitted`, tracked | `5 M` | `0.390` / `29` | `6.6e−2` | `1.8e−2`, `4.7e−3`, `4.8e−3` | `319` | `9.9e−3` | `0.99916` / `7e−6` / `0.99916` | `2.2e−2` |
| | `:fitted` at `20/M` (`n_L = 12`) | `5 M` | `0.428` / `20` | `7.1e−2` | `1.9e−2`, `1.4e−2`, `5.8e−3` | `125` | `7.3e−3` | `0.99934` | `1.8e−2` |
| | `:damped`, tracked, `4/M` | † `1.5 M` | `0.454` / `72` | `8.2e−2` | | `4.0e4` | | `1.0008` | |
| | `:damped`, tracked, `20/M` | `5 M` | `0.362` / `33` | `3.2e−2` | `3.5e−3`, `2.5e−3`, `1.1e−3` | `3.3e4` | `9.2e−3` | `0.99932` | `1.9e−2` |
| | `:damped`, tracked, the grid rate | `5 M` | `0.432` / `25` | `4.4e−2` | `7.7e−3`, `2.4e−3`, `2.7e−3` | `971` | `7.1e−3` | `0.99939` | `1.9e−2` |
| | `:damped`, step 5's sphere about the analytic center, `4/M` | † `1.0 M` | `0.288` / `52` | `4.4e−2` | | `8.6e4` | | `1.0009` | |
| | coasting: `:fitted`, the finder off from chunk 4 (`t = 1 M`), `max_misses = 6` | lost at `2.25 M` | `0.278` / `20` | `2.9e−2` | | `284` | | — | |

`†` a degenerate metric (`metric_quantities`' `DomainError`) in the evolved
shell. Kerr's values: `M_irr = 1`, `J = 0` for `a = 0`; `M_irr = 0.84744`,
`J = 0.9`, `M_ch = 1` for Kerr-Schild `a = 9/10`; `M_irr = 0.92580`, `J =
0.7`, `M_ch = 1` for harmonic `a = 7/10`.

What it says, row by row (**all measured in step 8f**):

- **The generic layer on the case that needs nothing costs a factor two.**
  On the static Kerr-Schild hole every `:fitted` row reaches `50 M` flat
  from `10 M` on, at `1.94×` the control's masked error and `2.8×` its shell
  `C_a`, with `M_irr` `4e−4` lower; the horizon shells outside it are
  within 15 %. The four `:fitted` rows — the analytic data to the core
  surface, the decided data, `Π` for `Π̃`, and the hand-over — all end on
  **the same state to three digits**: the steady error is the target's, not
  the initial data's and not the momentum variable's, and a fit that starts
  from evolved data at `5 M` (the hand-over; its projection fired 1764 times
  at the switch, at `r ≤ 0.175`, in the core, and never again) arrives at it
  within `5 M`. Step 8c's wrong-but-smooth analytic targets left the
  exterior at the exact target's error; a fit of the evolved state does not,
  because it is not a solution anywhere inside the offset surface and is
  wrong by `5.7` there against the analytic layer's `0.6`.
- **The finder's Kerr target is the analytic target** to four digits: the
  finder's `M_ch` and origin are `M` and the center to its truncation, so
  idea 4 is the analytic control by another road — useful after a merger,
  not here.
- **The snapshot target fails, at `8 M`.** Holding the evolved state still
  as the target (no fit, no regularity) grows the shell's error from the
  first chunk — `2.5e−2`, `0.11`, `0.24`, `0.46` at `2, 4, 6, 8 M` — until a
  degenerate metric ends the run: what the fit buys is its regularity, and
  a target that carries the state's own grid-scale content feeds it back.
- **On the spinning Kerr-Schild hole the fit is the error.** At `L = 12`
  the target's value residual is `2.5e−2` of the data (8e measured `L = 8`
  at 3–5 %), and the run carries it out: the masked error `41×` the
  control's, `J` `7 %` low and `M_ch` `2 %` low at `m = 5`, still slowly
  rising at `20 M`; a wider margin (`m = 8`) keeps the horizon's numbers at
  Kerr's (`0.906`, `1.002`) and the horizon shells 2.5× cleaner while the
  masked error, which includes the points between the offset surface and
  the horizon, is larger. The analytic target is the one to use on this
  chart, which admits it.
- **The harmonic chart at `a = 0` does not see the target**: `:fitted` and
  `:damped` agree to 5 % in the masked error and to 1 % in the shell. Both
  errors are the chart's: harmonic Schwarzschild at `h = 5/128` is steep
  between the offset surface and the horizon (the masked L∞ `7–8` sits
  there), and grows slowly to `10 M`.
- **G5's chart, harmonic `a = 7/10`, runs at `h = 5/256` and not at
  `5/128`.** At `5/128` the offset surface's equator is `3.6` cells from the
  ring and the run ends at `0.5 M` (analytic data to `3h`) or `3.5 M` (the
  decided data); at `5/256` it reaches `10 M` (`6293 s` on a node, `amdq`;
  the first hour-long attempt stopped at its deadline at `6 M` with the same
  numbers), every fit valid, the track within `1.5e−3` cells, with the
  horizon near Kerr's values (`M_irr` `0.30 %` low, `J` `1.0 %` high, `M_ch`
  `6e−4` low) and the masked error growing and slowing (`0.75`, `1.53`,
  `2.38`, `2.88`, `3.27`, `3.60` at `1, 2, 4, 6, 8, 10 M`; shell `C_a`
  `0.041` to `0.168`) — there is no analytic control on this chart to split
  the target's share from the chart's.

- **The moving hole: the fitted target holds it at `4/M`, and the analytic
  layer does not.** The boosted harmonic hole crosses `1.5 M` of the box in
  `5 M`; its track stays within `0.022` cells of the analytic center, every
  find succeeds, every fit is valid, `M_irr` within `8e−4` of `1`. The
  analytic `:damped` layer at the default rate ends at `1.5 M` on the
  tracked geometry and at `1.0 M` on step 5's sphere, with the masked L∞
  growing from the first chunk at the layer's *trailing* edge (measured
  locally: `3.9`, `10.5`, `36.5` at `1/4`, `1/2`, `3/4 M`, at the first
  evolved point on the `+x` side): its core is frozen (`w = ρ = 0`), a point
  crosses the eight-cell layer at `v = 0.3` in about `M`, and at `4/M` the
  ramp relaxes what the core held by only `e^{−2}` before releasing it —
  CODE.md's "points the core releases are relaxed within `1/ρ_max`" was
  written for the grid rate. At `20/M` (the rule's ramp, twelve cells) or
  at the grid rate the same layer reaches `5 M`, with the projection firing
  in the core throughout (`2.2e5` and `4.0e4` hits, at `r ≤ 0.37`). The
  `:fitted` core is not frozen — it relaxes toward the fit, which moves with
  the track — and reaches `5 M` at `4/M` without a hit, at the masked error
  of the rescued analytic rows (`0.39` against `0.36`–`0.43`) and `1.5–2×`
  their shell. **So a moving analytic layer needs `ρ_max ≳ 20/M` (proposed
  in step 8f); the fitted one does not.** Every row's masked error grows
  with the crossing, `≈ 0.08/M`, the same in all five survivors: the chart's
  truncation at `5/128` near the moving hole, as on the static harmonic
  hole.
- **Coasting costs nothing for five chunks.** With the finder off from
  `t = 1 M` the geometry is carried on the track's `v_est` for `1.25 M`,
  and the masked error is the tracked run's to 1 % (`0.2585` against
  `0.2608` at `2 M`); the sixth miss ends the run with its record by
  `TrackLostError` at `max_misses = 6`, as designed (the row was meant to
  coast five chunks and ran the window one chunk long).

The kink table and the price of harmonic `a = 9/10` are the probe's
(`generic=probe`, four threads here, three minutes), recorded under [Open
questions](#open-questions); the kink at the first evolved point,
`|Δ²(composite) − Δ²(analytic)|` against `|Δ²(analytic)|`, maximum over the
twenty components, the fit of the analytic solution (`cont = 1`):

| chart | `L = 8`, `Π` | `8`, `Π̃` | `12`, `Π` | `12`, `Π̃` (axis / 45° / equator) |
|---|---|---|---|---|
| harmonic `a = 9/10`, `5/256` (analytic `420`, `107`, `2.4e9`) | `8.8e5`, `2.0e7`, `1.6e9` | `2.3e4`, `8.7e5`, `1.5e9` | `1.1e4`, `8.9e5`, `1.6e9` | `366`, `2.5e4`, `1.5e9` (`L = 16`: `552`, `3.1e3`, `1.4e9`) |
| harmonic `a = 7/10`, `5/256` (`232`, `334`, `4.7e4`) | `7.9e3`, `3.7e4`, `1.3e4` | `2.2e3`, `1.3e4`, `1.0e4` | `727`, `5.0e3`, `1.4e4` | `193`, `1.4e3`, `1.0e4` |
| harmonic `a = 7/10`, `5/128` (`300`, `305`, `2.8e5`) | `8.1e3`, `4.8e4`, `1.0e5` | `1.7e3`, `1.3e4`, `8.4e4` | `1.2e3`, `8.5e3`, `1.1e5` | `241`, `1.9e3`, `9.2e4` |
| Kerr-Schild `a = 9/10`, `5/128` (`3.5`, `6.8`, `103`) | `17`, `27`, `55` | `8.9`, `16`, `46` | `3.4`, `2.6`, `42` | `2.2`, `2.1`, `35` |
| harmonic `a = 0`, `5/128` (`4050`) | `865` | `561` | `865` | `561` |
| Kerr-Schild `a = 0`, fixture (`15.4`) | `5.3`, `3.3`, `5.3` | `3.7`, `2.4`, `3.7` | as `L = 8` | as `L = 8` |

(The point is `r_1 + h/2`, half a cell out of step 8e's probe, whose
numbers it reproduces to a factor of two off the equator.) `Π̃` shrinks
every kink, 1.5× on the static holes and 2–40× off the spinning holes'
equators, and `L = 12` another 3–30× on the spinning ones; neither touches
a static hole's, which is radial. In the evolution `Π̃` changes nothing
measurable (the Kerr-Schild `a = 0` rows above).

**The decisions of the 8e hand-over (proposed in step 8f):**

- **`Π̃ = (α/√γ)Π` is the fitted momentum by default** (`FittedSpec`'s
  `fit_tilde = true`): the table above, and `fit_tests.jl`'s chain rule
  (`8.1e−8`, `1.2e−8` against the differenced `Π̃`, the stencil's `δ⁴`).
- **`lmax_fit` stays `8` by default and the spinning rows carry `12`**, as
  `lmax_shape` does: a static hole's fit is `l ≤ 2`, and the evaluator's
  cost doubles from `8` to `12` (`2.2` to `4.3 µs` a point, once a chunk
  in the cache fill: `17 s` of fills over the `ks9` row's 40 chunks against
  its `7300 s`).
- **The regrid path is built, and the moving rows use a wider fine region
  instead.** `adapt = true` on a `:fitted` case chooses the mesh on the
  analytic `:damped` data of the same geometry and fills the fitted data on
  the settled mesh (one pass to 288 blocks on the adaptive fixture, and a
  regridding chunk, in the suite), and refuses a chart whose analytic core
  surface meets its singular set — G5's own, where step 8 needs the
  indicator to flag on something else (the fitted data through a callback
  that reads the cache, or the state after the first chunk). The boosted
  rows cross a fixed capsule of fine blocks, which measures the layer and
  not the regrid.
- **The hand-over row is built** (`evolve!(…; handover)`) and measured
  above; **coasting** is the boosted row with the finder off for chunks
  4–9.

**What the matrix costs.** On a Symmetry node shared by seven rows of nine
threads, the Kerr-Schild fixture is `140 s` a `M` a row (seven `50 M` rows
in `7057 s`); Kerr-Schild `a = 9/10` on 1632 blocks `360–410 s` a `M` at 21
threads; harmonic `a = 0` on 960 blocks `195 s` a `M` at 16 threads; G5's
chart on 2472 blocks `630 s` a `M` at 64 threads. The fits are `16 ms` and
a cache fill `47 ms` on the fixture (101 and 100 of them in a `50 M` row),
and `46 ms` and `420 ms` on the `ks9` mesh. Before the workers were given
one BLAS thread each, OpenBLAS's pools spinning after every fit put a node
at twice its cores and the rows at ten times this machine's time per `M`.

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
exact — a generic target needs a thick ramp at a physical rate (step 8c
measured it, and from step 8c′ `4/M` is the default, decided 2026-09-23);
the Lorentzian metrics are not convex in `g_ab` (the angular mean of
Kerr-Schild `g_ab` inside the horizon has Euclidean signature), so every
blend and clamp is made in ADM variables; and the discrete scheme's
grid-scale modes have *outgoing* group velocity inside the horizon (every
centered first-derivative stencil annihilates the Nyquist mode, so the
shift advection does not act on it), attenuated only by dissipation —
step 8a computes their penetration length for this package's stencils
before anything is built, and it is what the margin `m` is measured
against. `a = 7/10` stays the fallback for G5 if the last row of step
8f's matrix does not fit a node.

**The offset surface is built (step 8d).** A case whose interior is a
`FittedSpec` keys its layer on the depth below the tracked horizon's offset
surface, with the track's center, velocity and shape (to `lmax_shape`)
updated from every find — under [The
interior](#the-interior-a-pointwise-damping-layer), "The tracked geometry";
on the static hole it is the analytic layer to six digits. What it does
not do yet is the second half of the answer: with an analytic target the
core surface must still contain the chart's singular set, which harmonic
Kerr at `a = 9/10` refuses at every `h` — so the proof-of-concept case waits
for step 8e's `:fitted` target, and its shape needs `lmax = 12` for a tenth
of a cell at `h = 5/256` where Kerr-Schild's needs `4` (measured in step
8d).

**The fitted target is built, and the proof-of-concept chart's initial data
exists (step 8e).** The `:fitted` variant relaxes the tracked layer toward
a cached fit of the evolved state (`CODE.md`'s "The fitted target", pieces
8–12): on the static fixture it is `:damped`'s to `1.24×` by `1 M` — `4.3×`
at `0.15 M`, the initial data's curvature kink at `r_1` — and a boosted hole
tracks to `4e−3` cells. Harmonic Kerr at `a = 9/10`, `m = 4`, `h = 5/256`
has, for the first time, initial data that is finite and a metric at every
point; **its run ends in the first chunk**, because the offset surface's
data (the ring `0.02 M` inside `r_1` at the equator, `|Π| ~ 4e4` there
against `30` on the axis) is not representable by a polynomial of degree
`L + 2` to the accuracy the evolved stencils need. What would unblock it —
`Π̃ = (α/√γ)Π` as the fitted momentum, `L ≥ 12`, a finer equator — is step
8f's to measure, and `a = 7/10` and 8g's excision remain the fallbacks.

**Closed in step 8f: G5 runs at `a = 7/10`, and harmonic Kerr at `a = 9/10`
waits with its price written down (decided 2026-09-23** by the orchestrating
session, with the decision delegated by Erik, before step 8f started; step
8f's host-side probe, `hole_runs.jl generic=probe`, measured the numbers
below**).** The chart is blocked by resolution before it is blocked by the
interior:

- **The measured spacing.** At `h = 5/256` and `m = 4` — the finest the
  proof-of-concept mesh of 2472 blocks reaches, and the spacing at which the
  offset surface's equator `r_1 = 0.921` first lies outside the ring at
  `0.9` — the analytic solution's own second difference at the first evolved
  point on the equator is `2.4e9` against `420` on the axis, its length scale
  `0.008 M`, under half a cell (step 8e); a spacing that resolves it by five
  cells is `h ≲ 5/1024` on the equator.
- **What step 8f changed does not rescue `5/256`.** Fitting `Π̃ = (α/√γ)Π`
  and `L = 12` brings the fit's kink at the first evolved point from `8.8e5`
  to `366` on the axis — the analytic `420` — and from `2.0e7` to `2.5e4` at
  45° (`3.1e3` at `L = 16`), against an analytic `107`: still 30–230 times
  off the equator, and on the equator itself the data are not resolved by
  any fit (`1.5e9` against `2.4e9`). The initial data at `L = 12`, `Π̃` is
  finite and a metric at all 1 265 664 points (`min det γ = 1.6`, `min α =
  0.219`), one right-hand side is finite (`max |du| = 7.9e8`, at the offset
  surface over the ring, `|z| = 0.08`), and the run ends in its first chunk,
  before `t = 1/400 M`, in a degenerate metric — as step 8e's did at `L = 8`
  (measured in step 8f).
- **The node run's price** (measured cost per point, `2.5 µs` a thread on
  this machine and twice that on a Symmetry core; `dt = 5.4e−3 M` at
  `5/256`, `λ_max = 0.906`, a quarter of it at `5/1024`): with `5/1024` on
  an equatorial band (`0.75 ≤ ρ ≤ 1.05`, `|z| ≤ 0.2`) and `5/256` elsewhere,
  **23 080 blocks, 1.2e7 points, 28 GB, one right-hand side `0.93 s` on 64
  node threads, `0.76 h` a `M`, `38 h` for `50 M`**; with `5/1024` inside the
  whole sphere `|x| ≤ 1`, 90 112 blocks, 111 GB, `3.0 h` a `M` and `149 h`
  for `50 M` (measured in step 8f). No checkpointing exists (`Possible
  extensions`), so either is one uninterrupted job longer than any queue's
  day — and it presumes a fit good enough off the equator, which none of
  step 8f's is, or a margin that depends on direction.

So `a = 9/10` in the harmonic chart is a research item of its own — a
direction-dependent margin or a finer equator, a fit that holds 45°, and
checkpointing for a multi-day run — and not a row of the matrix. **G5 runs
at `a = 7/10`** (`CODE.md`'s fallback since step 5: `√(M² − a²) = 0.714 >
a`), measured by step 8f's `h7` row at `h = 5/256`.

**The fitted target's host half is built (step 8e-i).** `src/fit.jl` fits
`(log α, β^i, γ_ij, Π_ab)` on the offset surface with a polynomial of degree
`L + 2 cont`, one QR for all twenty variables, and sweeps the result for
validity inward to the center — under [The
interior](#the-interior-a-pointwise-damping-layer), "The fitted target". It
reproduces the static hole to roundoff from `L = 2` and to the
interpolation order from the state, is a metric on both Kerr-Schild holes,
and on harmonic `a = 9/10` at `h = 5/256` is a metric fitted to value and
slope (`cont = 1`, `L ≥ 8`) and **not** fitted to curvature (`cont = 2`, at
every `L` to 16) — which is what 8e-ii's plan for that chart's initial data
asks for (measured in step 8e). Two findings for 8e-ii and G5: a moving
hole's shift needs its constant term (`shift_constant = true`), and the
evaluator costs `2.2 µs` a point at `L = 8`, twenty-three `u_exact`s.

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
