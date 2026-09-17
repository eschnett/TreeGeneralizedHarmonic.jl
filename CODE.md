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

*Status: design, reviewed three times; nothing implemented, nothing
measured.* Markers: **(decided)** is inherited from TreeAMR,
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
  excision milestone would be measured against.
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
- **No subcycling, one global `dt`** — TreeAMR's permanent commitment,
  which suits a black-hole run badly in principle (the coarse outer
  levels are advanced at the horizon's time step) and is accepted here
  for the reasons TreeAMR gives: no time interpolation at interfaces,
  one state vector for the integrator, simpler everything.

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

`q` is the finite-difference order (below), and `G = q/2 + 1` because
the Kreiss–Oliger operator of order `q + 2` reaches one point further
than the derivatives do. TreeAMR's vertex invariant `N ≥ 2G + 2` then
puts `N ≥ 8, 10, 12` at `q = 4, 6, 8`; the tests use `N = 8`, the
proof-of-concept runs `N = 16` or `32`. The working array is
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

**(predicted, G3)** On TreeAMR's M3 two-level periodic mesh the gauge
wave converges at rate `min(q, p − 1)`: at `q = 4`, rates **3 and 4**
for prolongation orders 4 and 6, at any restriction order and any `ε`;
the unrefined control converges at 4 throughout. This is TreeWave's
table for a fourth-order scheme, and it is what makes `p = 6` a
requirement rather than a choice.

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
per evaluation.

The kernel is *block-local*: it reads its own block's stored points and
nothing else, so it runs on every backend unchanged. Per-block spacings
and origins travel to the backend once per chunk, as TreeWave's spacings
do; the origins are new here, because the interior profiles, the masks
and the damping profile need positions inside the kernel.

The RHS never mutates `u`, and it is a pure function of `(u, t)`
(TreeAMR's contract): the interior treatment below is a term *in* the
right-hand side, not a write to the state.

### The time step

    dt = cfl · minimum_spacing(forest) / λ_max,
    λ_max = max over owned points of  α √(tr γ^{ij}) + |β|

GHSO2's conservative bound on the coordinate characteristic speed, taken
once per chunk from a speed slot in `diag` (a kernel writes it,
`block_mapreduce(max)` reduces it) and re-checked at the chunk's end as
TreeHydro does — throw, do not warn, if the step used violated the bound.
`cfl = 1/4` **(proposed** default, GHSO2's**)**.

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

**Constraint damping** (decided): the Gundlach–Pretorius term `Z_ab`,
with `γ0(x)` a *function of position* — a Gaussian of width a few `M`
around the hole's analytic center, tapered to a small value in the wave
zone, an `isbits` closure over `(center(t), M)` in the problem — and a
constant `γ2 > −1`. `γ0 ≈ 1/M` near the hole is GHSO2's measured
requirement for a stable evolution with the horizon in the domain.

## Boundaries

### Periodic

Free, through the tree. The gauge wave, shifted Minkowski and the
robust-stability tests are periodic in every dimension.

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
`ρ_max` from `dt` each chunk. `r_0` is chosen where the analytic
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
| shifted Minkowski | `ShiftedMinkowski(A, w)` | sampled | yes | a nonzero shift; the gauge-source path |
| Kerr-Schild | `KerrSchild(M, a)` | sampled | yes | a hole with a non-harmonic gauge source |
| harmonic Kerr | `Harmonic(M, a)` | 0 | yes | a hole in harmonic gauge; spin |
| **boosted harmonic Kerr** | `boost(Harmonic(M, a), v)` | 0 | no | **the proof-of-concept case**: a spinning hole crossing the mesh |

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
region at `t = 0`. For a `1/r` field the ratio itself decreases like
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
   loose `refine_tol`.
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

**Convergence studies on a frozen hierarchy.** An error-driven mesh
changes with the resolution, which muddles a convergence study. The
claims of order `q` in G4 and G5 are therefore made on the hierarchy the
indicator chose at `t = 0` for the coarsest run, *held fixed in space*
and refined uniformly by doubling `N` — the block layout is unchanged
and every spacing halves — with regridding off; the adaptive runs are
then compared against the uniform-fine reference in TreeWave's manner
(the tracked pulse against `uniform_pulse`). Both are needed: the frozen
hierarchy measures the scheme, the adaptive run measures the indicator.

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

The relaxation rate of the interior layer is bounded by RK4's stability
on the negative real axis, and `ρ_max · dt = 1` keeps it well inside;
the `:pasted` variant uses RK4's `step_limiter!(u, integrator, p, t)`
hook, the one place the state may be written outside the RHS. Adaptive
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

**Constraints.** Both kernels mask the interior `r < r_1` and write zero
inside it; the modified region is not a numerical solution. Norms are
`block_mapreduce` partials weighted by each block's `h³`, combined in
block order, so they are bit-identical across thread counts.

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
  compares digests.
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
  H200.
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
   TreeAMR's `TODO.md` lists "generic interpolation".
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
| `src/pointwise.jl` | GHSO2's pointwise algebra (ported from `notes/pointwise-ghso2.jl`), plus the expanded form's coefficient derivatives `metric_derivatives`, the assembled `gh_node_rhs_expanded`, and `gh_node_source` (all added in step 1); `SVector{10}` state, `SMatrix{4,4}` tensors. `gh_node_source` is a **second copy** of the reduced source and the damping, written out of `gh_node_rhs` character for character rather than factored out of it: the port stays diffable against `notes/pointwise-ghso2.jl`, which is what makes it the validated reference, and `test/pointwise_identity_tests.jl` asserts the copy still matches it — to roundoff, because two spellings of one expression are not bit-identical (see "Measured results") |
| `src/stencils.jl` | rational finite-difference and Kreiss–Oliger weights at order `q` (added in step 2): `derivative_weights`, `dissipation_weights`, both `@generated` over `(T, Val(q), Val(m))` and returning `SVector`s of `T` for unit spacing; `lagrange_derivative_weights` and the two `rational_*` constructors behind them, exposed unexported so that the exactness claims can be asserted in `Rational` rather than through a tolerance; `dissipation_rank(Val(q)) = Val(q/2 + 1)`, one spelling of `2r = q + 2` **(proposed in step 2)**; the host-side `apply_stencil` and `apply_mixed_stencil`, which are the reference contractions the tests measure with and the definitions step 3's streaming kernel has to agree with |
| `src/evolution.jl` | the fused RHS kernel in streaming order, `GHProblem`, `gh_rhs!`, the speed kernel, `gh_dt` |
| `src/gauge.jl` | sampling prescribed sources into `Hsrc`; the harmonic/static checks |
| `src/boundaries.jl` | the time-dependent Dirichlet hook |
| `src/interior.jl` | the profiles `w(r)`, `ρ(r)`, the core rule, the radius checks, the `:pasted` limiter |
| `src/initialdata.jl` | backgrounds, the `(h, Π)` callback with the core rule, the `SpacetimeMetrics` index conventions |
| `src/refinement.jl` | the Löhner indicator with its global floor, the mask, the level floor and ceiling, the four marks, the buffer; TreeWave's `refinement.jl` ported |
| `src/constraints.jl` | GH and ADM constraint kernels, masked norms |
| `src/horizon.jl` | the interpolating ADM provider for `ApparentHorizonFinder`; location, shape, area, `M_irr`, `J`, `M_ch` |
| `src/driver.jl` | `GHCase`, `evolve!`, the analysis record per chunk, `observer` |
| `src/io.jl` | the analysis time series, slice output |
| `src/benchmark.jl` | per-phase timings in TreeWave's format |
| `test/` | one `*_tests.jl` per section above, `prerequisite_tests.jl` (what the two pinned dependencies must still provide; added in step 0), `type_tests.jl`, `threading_tests.jl`, `device_tests.jl`, the standalone `thread_workload.jl`. `pointwise.jl`'s tests are **two** files over a shared `pointwise_backgrounds.jl` — `pointwise_tests.jl` for the algebra as a function of the state, `pointwise_identity_tests.jl` for the two identities that need derivatives of it — because between them they compile the metric library's nested dual passes for six backgrounds at two precisions (amended in step 1) |
| `bin/` | `gh.jl` (the CLI, after GHSO2's `gh3d.jl`), viewers, `benchmark.jl`, `backend.jl`, own `Project.toml` |

Dependencies: `TreeAMR` and `SpacetimeMetrics` (both unregistered, both
pinned to GitHub `main` by `[sources]`, which puts the Julia floor at
1.11 as in the siblings), `KernelAbstractions`, `StaticArrays`
(kernel-safe, and what `SpacetimeMetrics` speaks),
`OrdinaryDiffEqLowOrderRK` and `SciMLBase`, `LinearAlgebra` (`det`, `dot`
and `tr` on `StaticArrays`, which the pointwise algebra uses; a standard
library, added in step 1 and not listed when `PLAN.md` enumerated step 0's
`Project.toml` **(proposed in step 1)**), `HDF5` (from G6),
`ApparentHorizonFinder` and `KorzynskiSpin` (from G4). Tests add
`MultiFloats` and `ForwardDiff` — the latter because the checks on the
expanded form differentiate the analytic solution one layer above the one
`SpacetimeMetrics` takes internally (added in step 1). `bin/` adds
`CairoMakie` and `SixelTerm` in its own environment. The compat bounds
follow TreeWave's, including
`OrdinaryDiffEqLowOrderRK = "2.2.5"` **(proposed in step 0**, which is
the one bound `PLAN.md` left unstated**)**. The two pins resolved in step 0
to TreeAMR v0.1.0 and SpacetimeMetrics v1.6.0; a `[compat]` entry on a
`[sources]` dependency is a floor on what `main` may become, not a
selection, which is what `prerequisite_tests.jl` exists to notice.

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
- **G2 — The RHS on a uniform periodic mesh.** `evolution.jl`,
  `initialdata.jl`, `gauge.jl`; the gauge wave and shifted Minkowski
  cases; RK4. *Accept:* Minkowski and shifted Minkowski are stationary
  to roundoff (the RHS is exactly zero on data the stencils reproduce);
  the gauge wave converges at order `q` for `q = 2, 4, 6` on
  `N = 8`, roots `2 … 8`; white noise on flat space stays bounded for
  a thousand steps with `ε_KO > 0` and its growth without is recorded;
  the RHS is pure and never mutates `u`; the fused kernel's throughput
  per point recorded.
- **G3 — Coarse-fine faces, static mesh.** TreeAMR's two-level mesh;
  `constraints.jl`. *Accept:* the interface-order table (predicted rates
  3 and 4 at `p = 4, 6` for `q = 4`, control at 4), independent of the
  restriction order and of `ε_KO`; both constraint monitors converge on
  the gauge wave across the interface; the thread-workload digests
  identical at one and four threads; G2 and G3 on `CPU()` in `Float32`.
- **G4 — A black hole.** Kerr-Schild (`a = 0`, sampled `H`) and harmonic
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
KernelAbstractions launches, at `Float64` and `Float32`. This is the file to copy when a later step needs
a cheap test.

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
  the sonic surface applies unchanged.
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
  the exact solution — a research question about GH gauges.
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
   milestone.
4. **In-kernel evaluation of `SpacetimeMetrics`** on the H200 (G6); the
   structure does not depend on it, the cost does.
5. **The defaults** — `q = 4`, `N = 32`, `cfl = 1/4`, `ε_KO = 0.5`,
   `γ0 = 1/M` near the hole, `m = 8`, a layer of `2(G + 1)` spacings,
   `ρ_max · dt = 1`, the indicator's thresholds — are starting values
   for G4–G6 to confirm or move.
