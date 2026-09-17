> Copied verbatim from `/Users/eschnett/src/jl/GeneralizedHarmonicSecondOrder/METHODS.md` on 2026-09-16, repository at commit `37f4414`,
> with uncommitted local changes in the working tree. The original repository is unpublished; this copy is the
> citable reference for TreeGeneralizedHarmonic's `CODE.md`. Do not edit it here;
> amend `CODE.md` instead.

# Numerical methods

This file records the numerical choices in `GeneralizedHarmonicSecondOrder`.
The mathematical formulation and its well-posedness are derived in
`FORMULATION.md`; this file documents the discretization, mirroring
`WaveToySecondOrder/METHODS.md` (the scalar testbed the design follows).

## Formulation (conservative, first-order-in-time)

Vacuum Einstein equations in generalized-harmonic (wave-coordinate) form.
**Second order in space, first order in time.** State `(g_ab, Π_ab)`:
the spacetime metric (stored offset `g − η` for accuracy) and the
**densitised Lie derivative of `g` along the foliation normal**

    Π_ab = (√γ/α)(∂_t − ℒ_β) g_ab = √γ · n^μ ∂_μ g_ab,

NOT `∂_t g_ab`. This is the scalar testbed's `Π = √γ·n^μ∂_μΦ` lifted to
the 10 metric components. In harmonic gauge each component satisfies the
flux-conservative scalar wave with a per-component source:

    ∂_t g_ab = β^i ∂_i g_ab + (α/√γ) Π_ab
    ∂_t Π_ab = ∂_i( β^i Π_ab + α√γ γ^{ij} ∂_j g_ab ) + α√γ · S0_ab
    S0_ab    = C2sym_ab − 2 Γ2_ab    (the reduced Einstein source, 2Q_ab)

`S0` is the existing Christoffel/source algebra of the reduced GH system
(`gh.jl::make_rhs`, Garfinkle 2002 eq. 9); `calc_rhs(g, Π, ∂_x g, ∂_y g,
∂_z g)` returns `(∂_t g, Fⁱ_ab, S0_ab)` per node. The principal part is
the same conservative wave operator the scalar testbed uses, so the shift
enters as physical transport and the `Π`-equation is a flux divergence —
energy-bounded and **stable for constant, large, and superluminal shift**
(see `FORMULATION.md` §2; this cured the `∂_t g` instability that the old
second-order-in-time scheme had for any nonzero shift).

**Why not second-order-in-time / `∂_t g`:** with the `∂_t g` momentum the
shift appears as the un-absorbed central advection `2 g^{tk} ∂_k(∂_t g)`,
which has no symmetrizer and is unstable for any nonzero shift (measured
max Re λ ∝ β: 0.5→2.0, 2.0→8.8). The Lie-derivative momentum is the fix.

**Self-gravitating background:** `(α, β^i, √γ, γ^{ij})` are read from the
*evolving* metric `g` each RHS call via the ADM inversion
`adm_from_metric` (`α=1/√(−g^{tt})`, `β^i=−g^{ti}/g^{tt}`, `γ_ij=g_ij`) —
not from a fixed background.

## Discretization

* Meshes: `HexMeshes` — affine `make_uniform_hex` (periodic / cubic
  Dirichlet) and curvilinear `make_warped_uniform_hex` / inflated-cube /
  `make_radial_shell_mesh` (BH excision). `N = 4` GLL element.
* Operators (`HexSBPSAT`): the spatial gradient `∂_i g` and the flux
  divergence `∂_i Fⁱ` are SBP-SAT, applied per metric component.
  - **Affine:** the per-axis `apply_D!` (skew `H·D`).
  - **Curvilinear:** the free-stream-preserving conservative
    `apply_gradient3d!` / `apply_divergence3d!` with the conservative-curl
    mesh metric `make_metric_terms3d` (the GCL identity ⇒ constant metrics
    are preserved to round-off). The *mesh* metric `∂x/∂ξ` (fixed) and the
    *spacetime* metric `(α,β,γ)` (evolving) are orthogonal.
* RHS `gh_conservative_rhs!` + `GHWorkspace` (allocation-free per call;
  per-component gradient → `calc_rhs` → divergence + `α√γ·S0`). The old
  second-order-in-time element/mesh kernels are superseded.
* Time integration: first-order explicit RK on `(g, Π)` (the system is
  not Hamiltonian; symplectic integrators are not used).

## Kreiss–Oliger dissipation

`ε·μ⁻⁵·D⁶` per axis on both `g` and `Π` (kwarg `ε_KO`, default 0), with
`μ` the SBP-SAT first-derivative spectral radius (power iteration, stored
in `GHWorkspace`). The conservative operator has a clean spectrum
(max Re λ ≈ round-off) but is non-normal: generic noise has a *bounded*
transient (saturates ~15×, verified to 3000 steps) that KO damps (~15→7×).
KO is for high-frequency noise / sonic points, not the shift stability
(which the formulation already provides).

## Boundary conditions

Per-component **field-radiation** SAT on the `Π`-equation (eigenvalue-only,
no eigenvector projection — the testbed policy). Faces are classified
(`boundaries_gh.jl::classify_face_gh`) from the GH normal characteristic
speeds `c± = −b_n ± α`, `c⁰ = −b_n` (`b_n = n_i β^i`):
* **subluminal** → Sommerfeld (absorbing) or Dirichlet — residual
  `r_ab = Π_ab + ((βn+a_n)/a)·n̂·∂_n g_ab`, penalty `−σ|s_in|·wt·(r−tgt)`;
* **superluminal outflow** (`b_n<−α`, e.g. inside an excised horizon) →
  excision (no SAT — the one-sided stencil is the BC);
* **superluminal inflow** (`b_n>α`) → full-state Dirichlet (pin `g`, `Π`).
The physical outward normal is axis-aligned (affine) or from the Jacobian
columns × handedness (curvilinear), with weight `wt = JF/(H[row]·detjac)`
(mirrors `WaveToySecondOrder/boundaries3d.jl`). Like the scalar testbed,
field-radiation is **not constraint-preserving** (small-shift / short-run
accurate); constraint-preserving BCs are out of scope.

## Constraint damping

Gundlach–Pretorius (`γ0`/`γ2`, default 0): `Δ_ab = −γ0[t_a C_b + t_b C_a −
(1+γ2) g_ab t^c C_c]` added to the reduced source `S0`, with the harmonic
constraint `C_a` (from `gauge_constraint_at_node`) and foliation normal
`t_a = −α δ_a^t`. Δ ∝ C_a vanishes on the constraint surface (analytic
solutions untouched); `γ0>0` damps violations. Shifts the gauge speed to
`c⁰ = −(1+γ1)b_n`.

## Sub-horizon excision: the sonic-surface gauge-constraint instability

The capstone (validation item 7) and the flat 1D testbed (`test_gh_excision_1d.jl`)
exhibit a growing mode when the excised region is **inside** the horizon. A 2026-06
linear-stability study (on the corrected RHS; dense Jacobian via
`test/gh_harness.jl::gh_dense_operator`) pins it down — and, contrary to the earlier
"no-SAT excision boundary closure" framing, it is **not** a boundary-closure artifact:

* **It tracks the sonic surface, not the excision face.** On the flat MovingGrid testbed
  (`βˣ = V(x)`, `α = 1`; physical normal speeds `c± = −b_n ± α`, so at the −x face
  `c₋ = V − 1`), the **sonic surface** `c₋ = 0` sits where `V = 1` (the sub↔superluminal
  transition). The unstable eigenmode *follows that surface* as it is moved through the
  grid (it is not pinned to the excision face); with **no** sonic surface in the domain —
  a varying *but everywhere-superluminal* shift, −x outflow + +x full-Dirichlet correctly
  classified — the system is **stable** (max Re λ ≈ 0). This unifies with the BH: excision
  **outside** the horizon (no transition in the domain) is ≈stable; **inside** (the domain
  straddles the horizon/sonic surface) is unstable.

* **It is genuine, and GH-specific.** max Re λ stays bounded away from 0 under refinement
  (≈1.5–2.3 at M = 2, 4, 8) — a real mode, not under-resolution. The **scalar** conservative
  wave (`WaveToySecondOrder.wave3d_curved_rhs!`) on the *identical* MovingGrid excision
  background is **stable** at M = 2–32, with or without a sonic surface; the linear
  Lorenz-gauge Maxwell analog (`ElectrodynamicsSecondOrder`) is likewise stable. So the
  shared conservative-wave principal part is sound — the instability needs GH's gauge sector.

* **It is a gauge-constraint mode.** The growing eigenvector violates **only** the harmonic
  constraint `C^a = Γ^a + H^a` (response ≈ 0.6–1.1 per unit amplitude; the ADM Hamiltonian
  `ℋ` and momentum `ℳ_i` are untouched), is pure-`Π`, and is dominated by `g_tt/g_xx/g_tx`.
  The gauge family propagates **off the light cone** at `c⁰ = −b_n`; EM (whose constraint
  rides the light cone `c±`) has no such family and is stable, whereas GH's off-cone gauge
  constraint is pumped at the physical sonic surface (where `c₋ → 0`, the physical mode
  stalls, and the reduced source `2Q` couples the sectors).

* **It resists the local cures.** Tested on the dense-Jacobian gate, all failing to reach
  max Re λ ≤ 0: a maximally-dissipative characteristic outflow SAT on the `Π`-equation
  (`|B| = R|Λ|R⁻¹`, both light-cone modes), a gauge-family (`c⁰`) SAT on the `g`-equation and
  the combination, bulk *and* boundary-localised Gundlach–Pretorius `γ0`/`γ2`, Kreiss–Oliger,
  and the skew-symmetric (split-form) shift/flux discretisation. Each plateaus above zero or
  worsens — they target either the wrong location (the boundary) or the wrong sector (the
  principal part).

**Implication.** A constraint-preserving *boundary* SAT cannot cure this: the obstruction is
a bulk GH gauge-constraint mode at the sonic surface, not a boundary closure. The indicated
routes are the **fully-first-order GH system** (`Φ_iab`, linearly degenerate, with
constraint-preserving characteristic boundary conditions — the standard way to evolve across
horizons/sonic surfaces) or a **sonic-surface-aware constraint scheme** (e.g. Z4-style
constraint propagation that moves the gauge-constraint modes onto the light cone). Ordinary
constraint damping is insufficient.

### Z4-like constraint relocation: investigated, does NOT cure (2026-06-09)

A dynamical-Z4 cure was prototyped in the second-order `(g, Π)` form (standalone dense-Jacobian
harness on the R7 testbed, no production code changed): augment the state with a Z4 four-vector
`Z_a` and its densitised momentum `Π^Z_a`, evolved as four conservative scalar waves on the same
background, with (i) a back-coupling `+2ζ∇_(aZ_b)` injected into the reduced source `S0` (the
`H_a` gauge-source term shape, with `H→Z`), (ii) a constraint drive `±α√γ·κs·C^a` seeding `Z`
from the gauge-constraint violation, (iii) a friction `−μf·Π^Z_a`, and (iv) a **tunable Z
propagation speed `c_Z`** (flux `c_Z²·α√γ γ^{ij}∂_jZ_a`): at `c_Z>1` the Z characteristics
`−b_n ± c_Z α` do not vanish at the metric sonic surface `b_n=−α`, so `Z` can still drain the
violation where the metric mode `c₋→0` stalls. The background `(Z=0, Π^Z=0, C^a≈0)` is an exact
equilibrium, so the augmented Jacobian about it is clean (the decoupled Z block reproduces the
metric instability exactly: 1.78 at M=2, 2.33 at M=4, z-fraction 0 — no spurious Z mode).

Findings (max Re λ on the varying-V testbed `V0=1.6,w=0.5,xc=0.7`):

* **The existing Gundlach–Pretorius / Z4 damping (`γ0,γ2` ARE the Z4 `κ1,κ2`) plateaus** —
  re-confirmed on the corrected RHS: max Re λ falls only to ≈0.47 (`γ0=5,γ2=−1`), never ≤0.
* **Only the opposite-relative-sign coupling `ζ·κs<0` stabilizes**; `ζ·κs>0` blows up
  (max Re λ→31 at `κ=32`). Drive-only or feedback-only does nothing (the metric mode is unchanged).
* **The active lever is the propagation speed `c_Z`, not the damping rate** (friction `μf` is
  inert). Increasing `c_Z` pushes max Re λ down — qualitatively unlike rate-damping, and a direct
  confirmation that the obstruction is the **sonic zero-speed degeneracy**, not the decay rate.
* **But it plateaus and is not a cure.** At M=2 the optimum (`κ≈32`, `c_Z≈16`, a flat valley
  along `κ∝c_Z`) reaches max Re λ ≈ 0.36 (5× below baseline) and goes no lower; pushed further a
  competing Z-sector mode appears. Crucially it is **not resolution-robust**: at M=4 the
  re-optimized floor is only ≈1.6 (vs baseline 2.33 — ~30% reduction, cf. ~80% at M=2) and the
  `c_Z` lever inverts (it *helps* at M=2 but *worsens* at M=4), the signature of an
  under-resolution effect. And it is **not generalizable**: the M=2 reduction requires the
  idealized `Z=0` Dirichlet SAT at the excision face (the superluminal `Z` has inflow
  characteristics there); without it — the realistic unknown-interior case — max Re λ→4.0,
  *worse* than baseline.

**Conclusion.** Dynamical Z4 with superluminal constraint cleaning does relocate/damp the
violation and its speed lever confirms the diagnosis, but it yields **no genuine,
resolution-robust, generalizable cure in the second-order `(g, Π)` form**: it plateaus above
zero, collapses under refinement, and depends on knowing the excised interior. The obstruction is
the characteristic *degeneracy* at the sonic surface (a zero-speed family), not a damping rate one
can outrun. Removing the off-light-cone gauge family altogether — the first-order LSKOR `γ1=−1`
linear-degeneracy choice — requires the independent first-derivative variables `Φ_iab` and is
therefore a formulation change, out of scope for the second-order system. (Prototype preserved in
`scratch/z4/`; see its `README.md`.)

## Validation ladder (tests)

The GOALS test types, on the simple→superluminal→BH progression:
1. **Spectrum** (`gh_dense_operator`, first-order `n×n` Jacobian):
   max Re λ ≤ 1e-4·|λ|max about Minkowski, and **constant/superluminal
   shift** (β=0.5, 2.0) — the make-or-break gate (3e-7 vs the old 2.0/8.8).
2. **Free-stream / static preservation**: Minkowski and constant /
   superluminal flat metrics preserved to round-off (affine and on a
   warped curved mesh — the GCL identity).
3. **Convergence**: gauge-wave `∂_t Π → ∂_tt g` at interior SBP order.
4. **Robust stability**: superluminal-shift evolution bounded to 1000–3000
   steps; KO damps the noise transient.
5. **Boundaries**: Minkowski Sommerfeld spectrum stable; the classifier
   auto-picks excision/full-Dirichlet/Sommerfeld and preserves the flat
   superluminal static solution.
6. **Constraint damping**: `γ0>0` reduces constraint growth; vanishes on
   constraint-satisfying data.
7. **Capstone**: harmonic-coordinate Schwarzschild on a radial shell with
   the excision sphere **inside the horizon** (harmonic `r_h=1`, `R1=0.8`)
   — inner excision + outer Sommerfeld + KO — evolves **finite and
   bounded** (the configuration the old scheme NaN'd on). Coarse,
   short-time smoke; a production run (t~10M, convergence,
   inhomogeneous-Dirichlet outer, constraint damping) is future work.

## Status / future work

The conservative recast, stabilizers (KO, constraint damping), curvilinear
operators, boundary suite, the sub-horizon excision capstone (smoke), and
the first-order `gh_evolve3d` driver (with an optional harmonic-constraint
RMS monitor, `record_constraints` → `constraint_trace`) are in and
validated; the apps (`bin/harmonic_bh_demo.jl`, `bin/cubed_sphere_demo.jl`)
use the conservative `(g, Π)` API. The dead second-order kernels
(`gh_rhs_*`, `_calc_rhs_at`, `step`/`call_step`) and the stale
second-order tests have been retired; `rhs_element.jl` now holds only the
affine SBP helpers and the constraint diagnostics. The `GOALS.md`
Formulation update (state vector = Lie-derivative momentum; why
second-order-in-time is not used) has been applied; remaining suggestions
(reconciling the stale boundary text, an ADM/energy monitor) are collected
in `GOALS-comments.md`. Remaining work: a production capstone (long-time
`t ~ 10 M` harmonic-Schwarzschild run with h/p-convergence and constraint
damping) and the GPU/KernelAbstractions port.

## References

See `FORMULATION.md` (Choquet-Bruhat 1952; Friedrich 1985; Garfinkle 2002;
Pretorius 2005; Gundlach et al. 2005; Lindblom et al. 2006; Kreiss-Ortiz;
Nagy-Ortiz-Reula; Gundlach-Martín-García).
