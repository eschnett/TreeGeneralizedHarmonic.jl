> Copied verbatim from `/Users/eschnett/src/jl/GeneralizedHarmonicSecondOrder2/METHODS.md` on 2026-09-16, repository at commit `8fd820a`,
> with uncommitted local changes in the working tree. The original repository is unpublished; this copy is the
> citable reference for TreeGeneralizedHarmonic's `CODE.md`. Do not edit it here;
> amend `CODE.md` instead.

# Numerical methods

This file records the formulation and the numerical choices made in
`GeneralizedHarmonicSecondOrder2`. It is the companion to `GOALS.md`
(what we want) — this is what we actually do, and why.

## Formulation

### The system

We evolve the vacuum Einstein equations in the **generalized harmonic
(GH) formulation, second order in space and first order in time**. The
evolved state per grid node is 20 fields: the metric offset and a
densitised momentum,

    g_ab − η_ab                                   (10 components)
    Π_ab = (√γ/α) (∂_t − β^i ∂_i) g_ab = √|g| n^μ ∂_μ g_ab   (10)

in the packed component order `(tt, tx, ty, tz, xx, xy, xz, yy, yz,
zz)`. No spatial-derivative auxiliaries `Φ_iab` are evolved — the
second-order-in-space form halves the state relative to second-order
BSSN/Z4c and is 2.5× smaller than the first-order GH reduction
(LSKOR/SpEC, 50 fields), which is the point: the target is
memory-bandwidth-limited GPU evolution.

The gauge condition is the generalized harmonic constraint

    C^a ≡ Γ^a + H^a = 0,      Γ^a = g^{bc} Γ^a_{bc},

with a **prescribed gauge source** `H_a(x)` (lowered index stored,
together with its gradient `∂_a H_b`). `H` is sampled from the analytic
solution being evolved (`H^a = −Γ^a[g_exact]`), so every background in
the catalog is a stationary point of the discrete system; harmonic
backgrounds (Minkowski, the AwA gauge wave, Kerr in harmonic
coordinates) have `H = 0` exactly. A dynamical damped-harmonic gauge
driver (Szilágyi–Lindblom–Scheel) is *not* implemented yet; it would be
the natural extension for dynamical mergers, and it keeps the principal
part untouched because it is algebraic in `g`.

### Evolution equations (flux-conservative recast)

Per metric component the system is a flux-conservative scalar wave with
a source (see Garfinkle 2002, and FORMULATION.md of the predecessor
package `GeneralizedHarmonicSecondOrder` for the derivation and the
symmetrizer):

    ∂_t g_ab = β^i ∂_i g_ab + (α/√γ) Π_ab
    ∂_t Π_ab = ∂_i F^i_ab − α√γ S0_ab
    F^i_ab   = β^i Π_ab + α√γ γ^{ij} ∂_j g_ab
    S0_ab    = C2sym_ab − 2 Γ2_ab − 2 ∇_(a H_b) − Γ^ν ∂_ν g_ab + Z_ab

with `C2sym_ab = C_a^{μν}C_{μνb} + C_b^{μν}C_{μνa}` (`C_abc = ∂_a
g_bc`), `Γ2_ab = Γ^γ_{να}Γ^ν_{γβ}`, `2∇_(aH_b) = ∂_aH_b + ∂_bH_a −
2Γ^c_{ab}H_c`, and the densitisation correction `−Γ^ν ∂_ν g_ab` that
converts the densitised operator realised by the conservative flux,
`□_dens = (1/√|g|) ∂_μ(√|g| g^{μν} ∂_ν ·)`, back to the bare reduced
operator `g^{cd}∂_c∂_d`. The source enters `∂_tΠ` with a minus sign
because `Π = √|g| n^μ∂_μ g ⇒ √|g| g^{tν}∂_ν g = −Π`.

Two deliberate choices, both inherited from the validated predecessor
package:

* **Lie-advected momentum.** The naive `(g, ∂_t g)` choice of momentum
  is unstable for nonzero shift; absorbing the advection `β^i∂_i g`
  into the momentum definition makes the principal part ten decoupled
  shifted scalar-wave operators and the semidiscrete energy estimate
  shift-independent. Hyperbolicity only requires spacelike slices
  (`α > 0`), so superluminal shift costs nothing — there is no
  Bona–Massó-type gauge cone.
* **Flux-conservative spatial operator.** The divergence-of-flux form
  reuses the energy-stable conservative SBP first-derivative pair
  (gradient/divergence are discrete adjoints), giving the same interior
  skew-adjointness as the scalar-wave testbed (`WaveToySecondOrder`).

The sign conventions and the exact source algebra were taken verbatim
from the predecessor's symbolically-generated `calc_rhs` and are
re-validated pointwise against `SpacetimeMetrics` automatic
differentiation (see Tests).

### Constraint damping

Gundlach–Calabrese–Hinder–Martín-García / Pretorius damping, written in
the lowered form with the foliation normal one-form `t_a = −α δ_a^t`:

    Z_ab = γ0 [ t_a C_b + t_b C_a − (1 + γ2) g_ab t^c C_c ],
    C_a = g_ab C^b = Γ_a + H_a,

added to `S0` (so γ0 > 0 damps; γ2 corresponds to the trace parameter
`p` of Gundlach et al., with γ2 > −1 required at the continuum level).
The damping term is proportional to the gauge constraint, so exact
solutions are unaffected; because (modulo the discrete evolution) the
ADM Hamiltonian and momentum constraints are combinations of `C_a` and
its derivatives, this one mechanism services both the gauge and the
physical constraint sectors. Practically `γ0 ~ O(1/M)` near a black
hole, tapered in the wave zone.

### Floating-point hygiene

The state stores `g − η`. All cancellation-sensitive derived
quantities are computed as offsets, never by subtracting O(1) values:

    g^{ab} − η^{ab} = −g^{ac} h_{cd} η^{db}          (exact identity)
    det(g) + 1 = −(e1 + e2 + e3 + e4)(η·h)           (elementary
                                                      symmetric polys)
    det(γ) − 1 = −d1 − q1 + d1·q1,  d1 = det(g)+1, q1 = g^{tt}+1
    α = 1/√(1 − q1),   √γ = √(1 + (det(γ)−1))

A unit test verifies full relative precision of the offsets at
‖h‖ ~ 1e−13.

## Discretization

### Mesh and operators

Unstructured conforming hexahedral meshes from **HexMeshes** (uniform
box, warped box, cubed-cube, inflated-cube ≡ cubed-sphere ball, radial
shell), with tensor-product GLL spectral elements and SBP-SAT operators
from **HexSBPSAT** (N = 4 or 8 nodes per dimension recommended for
GPUs). Spatial derivatives:

* **Affine meshes** (`make_uniform_hex`): per-axis SBP first derivative
  `apply_D!` with centred-flux SAT at interior faces — the assembled
  `H·D` is exactly skew.
* **Curvilinear meshes**: the free-stream-preserving conservative split
  form `apply_gradient3d!` / `apply_divergence3d!` with metric terms in
  conservative-curl form (`make_metric_terms3d`), so the discrete
  metric identities hold to round-off (verified: flat-space RHS is
  exactly zero on warped, cubed-cube, and inflated-cube meshes).

The second-order operator is realised as divergence∘(coefficient ×
gradient) — a wide-stencil "D-of-D" composition with centred-flux SATs
on both passes. The inter-element coupling therefore inherits the
energy estimate of the conservative pair; no LDG auxiliary variable is
ever stored (gradients live only in the RHS workspace, recomputed every
evaluation). The narrow-stencil SIPG Laplacian of HexSBPSAT
(`apply_laplacian!`) is not used because the GH coefficient
`α√γ γ^{ij}(g)` varies with the solution.

### One RHS evaluation

1. `∂_i g_ab`: 10 scalar gradients (two-pass KernelAbstractions
   kernels: face-trace gather, then volume + SAT).
2. **Pointwise kernel** (single KA kernel, one workitem per node):
   builds `g4`, `g^{ab}`, ADM quantities (via the offset identities),
   Christoffels, and emits `∂_t g`, the three fluxes `F^i`, and
   `msrc = −α√γ(S0 + Z)`; reads the static `H, ∂H` fields when the
   gauge source is nonzero.
3. `∂_t Π = ∂_i F^i + msrc`: 10 divergences plus an accumulation
   kernel.
4. Optional **Kreiss–Oliger dissipation** `ε_KO μ⁻⁵ D⁶` per axis on all
   20 fields, where `μ` is the spectral radius of the assembled SBP
   first derivative (power iteration at setup). `D` is skew ⇒ `D⁶` is
   negative semidefinite; the `μ⁻⁵` normalisation pins the
   highest-mode damping rate at `ε_KO μ` so the CFL limit is not
   tightened for `ε_KO ≲ 1` (time step additionally capped at
   `1.4/(ε_KO μ)`).
5. Optional **boundary SAT** (below).

All kernels are type-generic (Float32 / Float64 / MultiFloats
`Float64x2`) and run unchanged on CPU, Metal, and CUDA backends; there
is no separate hand-written CPU loop. Every output node is written by
exactly one workitem (gather form, no scatter races).

State layout: one flat array `(N, N, N, Ne, 2·NC)`; per-component
slices are contiguous and feed the HexSBPSAT scalar kernels directly.

### Boundary conditions

Faces are classified from the characteristic speeds along the
metric-unit outward normal, `c± = −b_n ± α` with `b_n = n_i β^i`
(eigenvalues only — no eigenvector projection):

| class       | condition   | treatment                              |
|-------------|-------------|----------------------------------------|
| subluminal  | `|b_n| < α` | field-radiation SAT (Dirichlet/Sommerfeld) |
| outflow     | `b_n < −α`  | excision: no boundary term             |
| inflow      | `b_n > α`   | full-state Dirichlet                   |
| sonic       | `|c±| ≈ 0`  | rejected (ill-posed)                   |

**Dirichlet (the current default)** imposes the scalar field-radiation
condition per component on the Π equation: the normalised ingoing
residual

    r_ab = Π_ab + ((b_n + a_n)/a_div) ∂_n g_ab,
    a_n = α √(γ^{ij} n_i n_j),  a_div = α/√γ,

is penalised toward the residual of the boundary data,
`Π̇_ab += −σ |a_n + b_n| wt (r_ab − r^data_ab)`, with `σ = 1` (full
upwind) and the face weight `wt = J_F/(H1_row · det J)` (reduces to
`invjac/H1` on affine meshes). Sommerfeld is the same penalty with
zero target. Full-state Dirichlet pins both `g` and `Π` with
`τ = σ(|a_n−b_n| + |a_n+b_n|) wt`. Excision faces (mesh tag 8) receive
no SAT — the one-sided SBP stencil is already correct pure outflow.

The local coefficients (`α`, `β`, `γ^{ij}`) are read from the evolved
metric at the face node; the data targets `(g, Π, r)^data` are sampled
from the analytic solution at setup with *analytic* gradients and
stored as static face-indexed arrays — valid for stationary
backgrounds. Time-dependent boundary data is not supported yet (use
periodic meshes for the gauge wave). This boundary treatment is not
constraint-preserving; constraint-preserving/radiative outer conditions
(Lindblom et al., Rinne) are a planned refinement, per GOALS the
current scope is periodic + Dirichlet.

**The "sonic-surface instability", resolved** (full investigation:
`scratch/sonic/NOTES.md`): the growing mode reported near stalled
characteristics by the predecessor package is two stacked phenomena.
(1) A *continuum* instability of the linearized gauge-fixed system on
**non-stationary** (moving-grid) backgrounds: non-normal compression of
the conservatively transported Π-sector, converted into true point
spectrum by the system's own lower-order couplings, rate
≈ (1–1.5)·|∂(speed)| — absolute at sonic points, convective elsewhere;
static backgrounds are continuum-stable. This layer is independent of
the discretization (reproduced, converged, by an independent Fourier
code), is immune to constraint damping at testbed strength, and is an
artifact *of the steep test backgrounds*: at a black-hole horizon its
rate is the surface-gravity scale κ = 1/(4M). (2) A grid-scale
discretization layer (wide-stencil + variable coefficients), present
even on static backgrounds — **cured by Kreiss–Oliger dissipation**
(ε_KO ≈ 0.5 restores the continuum spectrum exactly). Measured on an
excised Kerr-Schild shell with the horizon inside the domain: γ0 = 0
blows up at rate ≈ κ; **γ0 = 1/M is stable with constraints flat at
truncation**. Practical recipe for excision runs: ε_KO ≈ 0.5 and
γ0 ≳ a few/M; do not calibrate stability on moving-grid testbeds with
|V′| ≫ κ. What then remains on the excised hole is a *slow,
constraint-preserving* drift (≈ 0.14/M at the tested resolution,
constraints flat to t ≈ 40M) — a gauge-sector phenomenon for which the
damped-harmonic gauge driver is the indicated next step.

### Time integration

Explicit RK, order matched to the element order: Tsit5 (N ≤ 4), Vern6
(N = 5), Vern7 (N = 6), Vern8 (N = 7), Vern9 (N ≥ 8). Fixed time step
by default (reproducible convergence studies):

    dt = cfl · dx_min / max_speed,   max_speed = max(α√(tr γ^{ij}) + |β|),

with `cfl = 1/4` default, and the KO cap when `ε_KO ≠ 0`. The driver
aborts cleanly on NaN/Inf, RHS exceptions, or dt underflow, and
reports `aborted/abort_time/abort_reason`.

Stepping is done by the **native fixed-dt RK stepper**
(`src/timestepper.jl`, `stepper = :native` default; `:diffeq` falls
back to OrdinaryDiffEq, as do adaptive or GPU-array runs). It is
generic over Butcher tableaus (`RKTableau` — any explicit RK method of
the same structure, with any number of stages, plugs in via its
`(a, b)` coefficients); the tableaus are extracted at runtime from the
OrdinaryDiffEq coefficient structs by field-name reflection, with
order-condition guards, so they are never transcribed by hand. Wins
over the generic integrator on the production path: the stage buffers
are NUMA-first-touched; each stage update is ONE fused chunk-parallel
pass (`u + Σ dt·a_ij k_j` evaluated in a single read of each operand)
instead of a broadcast chain; the per-step instability check is the
chunk-parallel `tany_notfinite`; and with fixed dt the FSAL/embedded
stages are dropped (Tsit5: 6 RHS evaluations/step instead of 7, Vern7:
9 instead of 10). The final step truncates to land exactly on `t1`,
matching OrdinaryDiffEq.

**Note (future, kernel fusion):** when the RHS is fused into
element-local kernels, fold the stage linear combinations that produce
the next state vector into the fused kernels as well — the per-element
pass should emit `utmp = u + Σ dt·a_ij k_j` (and the final
`u + Σ dt·b_i k_i`) while the element is hot in cache, so the stage
arrays never round-trip main memory between the RHS and the update.

## Initial data and backgrounds

Sampled on the host from `SpacetimeMetrics` analytic solutions
(`dmetric` forward-mode AD), then copied to the device:

| background           | metric                          | static | H = 0 |
|----------------------|---------------------------------|--------|-------|
| `:minkowski`         | `Minkowski()`                   | yes    | yes   |
| `:gauge_wave`        | `GaugeWave(A, d)` (AwA)         | no     | yes   |
| `:shifted_minkowski` | `ShiftedMinkowski(A, w)`        | yes    | no    |
| `:kerr_schild`       | `KerrSchild(M, a)`              | yes    | no    |
| `:harmonic_bh`       | `Harmonic(M, a)` (□x^μ = 0)     | yes    | yes   |

`Π` initial data uses the *discrete* spatial gradients
(`momentum_from_dtg!`), so static analytic solutions satisfy
`∂_t g = 0` exactly, not just to truncation. Boundary targets use
analytic gradients (exact data). `ic = :noise` adds white noise for
robust-stability tests.

## Diagnostics

* **Gauge constraint** `C^a = Γ^a + H^a` per node (first derivatives
  only).
* **ADM Hamiltonian and momentum constraints** from the covariant
  Einstein tensor projected on the foliation normal; the second
  derivatives are formed by applying the SBP gradient to the
  first-derivative fields, with the `∂_tt` block reconstructed from the
  reduced evolution equation — so the monitor is consistent with the
  discrete dynamics. Host-side, evaluated at sample times.
* **L² error** against the analytic solution in the `Hphys`
  (mass-weighted) norm, all 20 components.

## Validation (test suite)

ReTestItems, one item per file, parallel across worker processes; GPU
items auto-skip without functional hardware (Metal on Apple silicon,
CUDA on Linux).

* `pointwise_tests` / `pointwise_bh_tests` — the complete pointwise
  algebra against SpacetimeMetrics AD: ADM extraction, offset accuracy
  at ‖h‖ ~ 1e−13, the conservation identity `∂_tΠ − ∂_iF^i = msrc`
  via finite differences of the analytic solution (flat, gauge wave,
  shifted Minkowski, Kerr-Schild with/without spin, harmonic Kerr),
  vanishing of the gauge constraint and of the damping term on exact
  data.
* `spectrum_tests` — dense linearisation (central differences) of the
  full discrete RHS on a periodic element; max Re λ < 1e−5 for flat
  space, flat + KO + damping, and superluminal shift V = 2 (the
  excision regime), with and without dissipation.
* `convergence_tests` — RHS consistency converging at >3×/doubling
  (affine, warped-curvilinear, Kerr-Schild box), full gauge-wave
  evolution converging at >3×/doubling (measured ≈ 9.6×), stationary
  Kerr-Schild on a Dirichlet box.
* `noise_tests` — robust stability: white noise on flat space, periodic
  and Dirichlet, affine and warped, with KO; bounded over crossings.
* `constraints_tests` — constraint monitors converge on analytic data;
  γ0 > 0 reduces the constraint violation of a noise-perturbed
  evolution.
* `bc_tests` — face classification, admissibility validation
  (superluminal Dirichlet rejected), boundary SAT consistency on exact
  static data.
* `multiblock_tests` — exact free-stream on cubed-cube and
  inflated-cube meshes; Dirichlet evolution on the spherical outer
  boundary.
* `gpu_metal_tests` / `gpu_cuda_tests` — CPU/GPU agreement of full
  evolutions (gauge wave, KO + damping, Dirichlet Kerr-Schild,
  curvilinear).
* `types_tests` — Float32 and Float64x2 (MultiFloats) pointwise
  agreement and short evolutions.

## Production infrastructure (outer boundary ≥ 100M, 64-thread CPU)

### Channel-batched operators and threading

All per-component operator applications go through HexSBPSAT's
channel-batched kernels (`apply_D_batch!`, `apply_gradient3d_batch!`,
`apply_divergence3d_batch!`; fields `(N,N,N,Ne,C)`, one workgroup per
(element, channel) on a 2-D group grid). One RHS evaluation is ~9
launches without KO and ~43 with KO — down from ~60/~763 scalar
launches — which removes the KernelAbstractions CPU launch barrier
(task-spawn + sync ≈ 25–35 µs each) as a scaling obstacle at high
thread counts. Details that matter: element-fastest workgroup order and
a channels-LAST face-trace indexing (`reshape` of the shared workspace
pool) keep all hot-loop accesses streaming — with the channels-first
trace layout the batched gather degrades ~3× at production sizes. The
divergence kernel fuses the `+ msrc` source accumulation; the final KO
pass accumulates `du += ε μ⁻⁵ D⁶ u` in-kernel; no serial broadcasts
remain in the RHS.

The RK stage updates run through FastBroadcast's threaded path
(`Tsit5/Vern*(thread = Threaded())`, CPU backend only) — without this
the serial stage broadcasts rival the RHS cost at high thread counts.
Host-side diagnostics (`l2_error`, `_max_speed`) are threaded with
chunked per-task partials. Measured locally (8 threads, Ne = 432,
N = 6): RHS 131 → 89 ms, full step 1381 → 927 ms versus the scalar
implementation; `bin/bench_rhs.jl` (+ `bin/hpc_bench_sweep.sh`)
produces the per-section/threads scaling table for the HPC node.

### CPU fast path of the batch operators (HexSBPSAT ≥ 1.5)

The KernelAbstractions CPU backend executes the batch kernels one node
per workitem — a scalar N-point dot with no SIMD, localmem staging
split by `@synchronize`, stride-9 `invjac[d,d,…]` gathers, branchy
per-node SATs — measured at ~20 GB/s effective vs ~300 GB/s STREAM
(instruction-bound), with no scaling past one socket. HexSBPSAT 1.5
adds a plain-Julia threaded CPU fast path inside the same public
functions (`backend isa CPU` dispatch; GPU keeps the KA kernels):
SIMD pencil loops across *output* nodes (the reduction index stays
serial), a contiguous per-axis Jacobian diagonal `geom.invjacd`,
direct neighbour face reads (no gather pass), `:static` chunks over
the flattened `(e, c)` index consistent with the NUMA first-touch
split. The fast path is **bitwise-identical** to the KA path (same
per-output reduction order and expression trees) and CI pins the two
together with `@test ==` across all patch kinds; results of existing
runs are unchanged. `HEXSBPSAT_NO_CPU_FASTPATH=1` forces the KA path
(benchmarking only). Measured (EPYC 7532 ×2, 64t, M=6/M_r=20/N=6,
single-job A/B): one KO sweep 12× faster (KO 419 → 34 ms/RHS), KO
scaling 32→64t restored (2.1×), RHS 519 → 134 ms with the D path
alone; the gradient/divergence pencil ports take the remaining
`rhs_noko` down further.

### NUMA placement (multi-socket nodes; `src/numa.jl`)

Julia's allocator is not NUMA-aware: under Linux' first-touch policy a
page lands on the domain of the thread that first writes it, so a
serially-`fill!`ed workspace puts ~820 MB of RHS scratch on the main
thread's domain and all 64 cores then stream through that domain's two
memory channels (1/8 of node bandwidth on a 2-socket NPS4 EPYC, plus
cross-socket latency). Two coupled fixes:

  * **Parallel first-touch** (`first_touch!`): all big buffers (the
    `GHWorkspace` arrays, `u0`, the bench's `u/du`) are zeroed with one
    contiguous chunk per thread on a `:static` schedule — the same
    `divrem(n, nthreads)` split KernelAbstractions uses — so each
    thread's pages land on its own domain. `GH_NUMA_SERIAL=1` reverts
    to the serial fill for placement A/B benchmarks. (HexSBPSAT's
    face-trace pool is allocated untouched, so its pages are placed by
    their first kernel write — correct under static scheduling.)
  * **Static kernel scheduling** (`CPU(static = true)`, the default for
    `--backend cpu`; `cpu-dynamic` gives the old behavior): the dynamic
    KA backend `@spawn`s the same contiguous workgroup ranges onto
    arbitrary threads, so locality cannot persist between the ~43
    launches per RHS; static binds range ↔ thread id permanently. The
    backend is now threaded through `gh_rhs!` and the HexSBPSAT batch
    operators (`backend` kwarg) — `get_backend(::Array)` always returns
    the *dynamic* `CPU()`, so deriving it from the arrays would
    silently discard the choice.

The per-step serial full-array scans (the integrator's
`unstable_check = any(!isfinite, u)`, the progress-line and sample
`maximum(abs, …)`) read 150 MB from mostly-remote pages each; they are
replaced by the chunk-parallel `tany_notfinite`/`tmax_abs`. Pin
threads (`JULIA_EXCLUSIVE=1` or ThreadPinning) or none of the above
holds. `numactl --interleave=all` is the zero-code fallback (uniform
bandwidth, no locality).

### Radiative outer boundary (`BC_RADIATIVE` / `bc = :radiative`)

The field-radiation SAT augmented with the 1/r falloff term for
asymptotically ~1/r fields:

    r_ab = Π_ab + cr·∂_n g_ab + (cr/|x|)(g_ab − g^data_ab),
    cr = (b_n + a_n)/a_div,

driven to the static-data residual target. On exact static data the
falloff term vanishes identically, so exactness and admissibility
(subluminal faces) match the Dirichlet kind; the data targets make it
correct on curved backgrounds where plain zero-target Sommerfeld is not
(driving `r → 0` on Kerr-Schild pushes the solution off the
background).

### Batched constraint monitors (`gh_monitor`, `monitor_every`)

Both constraint families are evaluated with the same batched-kernel
machinery and per-element partial reductions (no atomics, CPU/GPU):

  * gauge `C^a` — state + first derivatives only (one batched gradient
    + two node kernels): a small fraction of an RHS evaluation;
  * ADM ℋ/ℳ — four further batched gradient applications for the
    second derivatives plus an immutable-StaticArrays port of the
    Einstein-tensor node evaluation (`∂_tt g` reconstructed from the
    reduced equation): ≈ 2 RHS evaluations.

Monitors run every `monitor_every`-th sample (`monitor = :gauge | :all`)
and agree with the host reference `constraint_rms` to round-off
(monitor_tests.jl). Caveat: the ADM node kernel's unrolled 4⁴ tensor
algebra costs minutes of *first-compile* time per session; a
PrecompileTools workload is the designated fix if this becomes
irritating.

### VTKHDF output (`src/io.jl` → HexVTKHDF package)

Time-series output in the VTKHDF 2.x format (pure HDF5, ParaView ≥
5.12) is provided by the standalone **HexVTKHDF** package
(`~/src/jl/HexVTKHDF`): static unstructured geometry — all GLL nodes as
points, `(N−1)³` linear `VTK_HEXAHEDRON` sub-cells per element — plus
temporal `Steps` bookkeeping with point data appended along the
unlimited dimension. `src/io.jl` here is a thin adapter: it re-exports
`VTKHDFWriter`/`write_step!`/`vtkhdf_finalize!` and adds the GH naming
(`GH_FIELD_NAMES`, `write_gh_step!`). The driver passes the `Mesh` to
the writer, so output files carry a `/Discretization` group (schema
`format_version = 1`: scalar type, `N`, the complete mesh state
including patch descriptors, package versions) from which HexVTKHDF's
reader (`VTKHDFFile`, `read_mesh`, `discretization`,
`f[name, step] :: MeshField`) reconstructs the mesh, the SBP element,
and the geometry without any GH code — plus slice extraction
(`uniform_slice`) and Makie plotting (`plotslice`) for post-processing.
Reproducibility: the full run-parameter set (including the RNG seed),
`Pkg.dependencies()` versions, and the Julia version are stored as a
TOML string in `/Metadata`. Driver kwargs `output_path`,
`output_every`, `output_metadata`; default field set = the 20 state
components + `C_gauge` when monitoring is on. Writer/reader internals
are tested in HexVTKHDF; the driver integration (write → reconstruct →
analyse) is tested here (io_tests.jl). One manual ParaView check on the
HPC side is still advised. No checkpointing/restart yet (per scope).

The writer **flushes after every step** (and after creating the static
geometry), so a killed job loses at most the step in flight, and it
runs in HDF5 **SWMR mode** by default (`swmr = true`; entered after the
first step, once all datasets exist): a *running* job's file can be
opened live with `h5open(path, "r"; swmr = true)`. Two SWMR-imposed
conventions: readers count completed steps as the length of
`Steps/Values` (the spec's `NSteps` attribute cannot be rewritten under
SWMR — it stays frozen at 1 during the run and is finalised by
`close`), and a plain non-SWMR open of the live file still fails (the
"bad object header version number" failure mode) — pass the SWMR flag,
or take a snapshot copy (`cp run.vtkhdf snap.vtkhdf`) for tools like
ParaView that open plainly. `vtkhdf_finalize!(path)` repairs `NSteps`
for runs killed before close. Caveats: SWMR assumes POSIX-coherent
visibility (fine on a node; cross-node on parallel filesystems like
BeeGFS usually works but is not guaranteed — the snapshot copy is the
bulletproof fallback), and a file from the pre-flush version of the
writer that died mid-run contains only the superblock and is
unrecoverable.

### Apparent horizons and spin (`src/horizon.jl`)

`find_gh_horizon` wraps ApparentHorizonFinder.jl (≥ 2.0, its default
stall-detection convergence — no `atol`/`maxiters` tuning), keeping its
single-point AoS interface: `AHInterpolator` answers each query by
analytic point location (`HexMeshes.locate_point`), tensor-product
Lagrange interpolation of the 20 state components — values and
reference gradients (`tensor_interp_grad`, new in HexMeshes) — physical
gradients through the analytic per-point element Jacobian
(`element_point_and_jac`, new), and the pointwise ADM extraction
`adm_vars_from_state` (γ_ij, ∂_kγ_ij, and K_ij with ∂_t g from the
evolution relation). Measured accuracy on Kerr-Schild (random off-node
points): worst-case γ/K error 7.7e-3 at N = 4, 6.4e-4 at N = 6, 3.4e-5
at N = 8 (K one order behind γ, as a derivative quantity); the horizon
of sampled Kerr-Schild data is found at r = 2M to interpolation
accuracy from a displaced initial guess. Driver kwargs
`horizon_every/horizon_r0/horizon_N` append a `horizon_trace`.
Out-of-mesh queries (excised region, beyond the outer boundary) throw a
descriptive error rather than returning NaNs.

After a successful find, the **Korzyński quasi-local spin**
(KorzynskiSpin.jl) is computed on the same collocation grid from the
interpolated `γ_ij`/`K_ij`: the result gains the proper `area`,
`M_irr = √(area/16π)`, the spin `J` with its coordinate-space
`spin_axis`, and the Christodoulou mass
`M_ch = √(M_irr² + J²/(4M_irr²))`; the progress line and the gh3d
summary report `J`. Disable with `spin = false`; a failed spin
computation leaves `J = NaN` without failing the find. Validated on
sampled Kerr-Schild data (N = 5, M = 3, M_r = 4): a = 0 gives
|J| < 1e-3 and M_irr = M_ch = M to 5e-3; a = 0.6 recovers J = Ma and
M_ch = M to 5e-3 with the spin axis along ±ẑ to 1e-3.

### Setup-time engineering

Production-size setup (geometry + analytic sampling) was originally
serial and dominated start-up (≈ 68 s at Ne = 4320/N = 6 locally; the
user's first 100M-domain attempt spent ~4 minutes before the first
step). Three fixes brought it to ≈ 6 s (8 threads; scales with cores):

1. `HexSBPSAT.make_geometry` element loops threaded (11 s → 0.6 s) —
   pure per-element analytic-map evaluation, disjoint writes.
2. The GH sampling loops (`sample_metric_state`, `sample_gauge_source`,
   `make_boundary_targets`) threaded over elements.
3. **An allocation bug in SpacetimeMetrics**: rebinding a variable that
   a generator closure captures (`Γ = SArray((Γ[…]+Γ[…])/2 …)` in the
   symmetrize steps of `ChristoffelSymbols`/`dChristoffelSymbols`/
   `RiemannTensor`) heap-boxes the captured variable — ~6 KB and ~40×
   slowdown per call, multiplying under the nested duals of
   `gauge_source_grad` (37 KB/call, and GC contention capped thread
   scaling at ~3×). Fresh names + manual dual seeding (replacing the
   `ForwardDiff.jacobian` closure, same pattern as `dmetric`) make the
   whole gauge-source path allocation-free: 32.7 → 1.86 µs/call.
   The evolution hot loop (`gh_node_rhs`) was verified allocation-free
   (0 B, BenchmarkTools) — it never had the rebinding pattern.

The driver's `verbose` landmarks time-stamp every setup phase, so any
future regression of this kind is visible immediately.

### HPC deliverables

`bin/hpc_bench_sweep.sh` (thread-scaling table, with pinning notes) and
`bin/hpc_bh_run.sbatch` (the R = 100M excised-hole production template:
radiative boundary, γ0 = 1/M + ε_KO = 0.5, gauge monitor every sample,
horizons every 5th, VTKHDF every 10th; constant radial spacing per the
project decision, M_r sized by the node budget). Run t1 = 50M first;
the gauge trace should show only the boundary transient at t ≈ 2·R2,
and the layer-3 slow drift (see the sonic-instability notes) sets the
practical run length until the gauge driver lands.

## Performance notes / future work

* The per-component gradient/divergence calls launch 10 scalar kernels
  each; batching the component dimension into the HexSBPSAT kernels
  (one launch per operator) is the next bandwidth/latency optimisation,
  as is fusing the flux divergence with the source accumulation.
* Kreiss–Oliger via six `apply_D!` passes per axis/field is launch-
  heavy; a fused D⁶ kernel would cut this 6×. KO is off by default
  (smooth subluminal runs are stable without it).
* The damped-harmonic gauge driver and constraint-preserving /
  Sommerfeld outer boundaries are the planned refinements after the
  Dirichlet stage.
* Excision needs a cure for the sonic-surface gauge-constraint mode
  (see the predecessor package's METHODS.md §"Sub-horizon excision");
  candidate routes: first-order LSKOR reduction near the horizon, or a
  sonic-surface-aware constraint scheme.
