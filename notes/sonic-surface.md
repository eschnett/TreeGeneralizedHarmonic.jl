> Copied verbatim from `/Users/eschnett/src/jl/GeneralizedHarmonicSecondOrder2/scratch/sonic/NOTES.md` on 2026-09-16, repository at commit `8fd820a`,
> clean at that commit. The original repository is unpublished; this copy is the
> citable reference for TreeGeneralizedHarmonic's `CODE.md`. Do not edit it here;
> amend `CODE.md` instead.

# The "sonic-surface instability": numerical investigation and verdict

Question: the second-order conservative GH system was reported (by the
predecessor package) to have a resolution-robust growing mode tied to
sonic surfaces (stalled characteristics) under sub-horizon excision —
is it a discretization artifact, and what cures it? Analyzed from
scratch in this package on boundary-free and excision configurations,
with two structurally different discretizations (the package's SBP-SAT
operators; an independent single-domain Fourier collocation code
reusing the same pointwise algebra — `scratch/sonic/common.jl`).
Spectra are dense central-difference Jacobians about sampled analytic
states.

## Executive summary

The reported instability is **two distinct phenomena stacked**:

1. **A genuine continuum instability of the linearized, gauge-fixed GH
   system on *non-stationary* (moving-grid) backgrounds**, growth rate
   ≈ (1–1.5)·|V′| set by the background's shift gradient. Mechanism:
   conservative transport of the Π-sector through regions of varying
   characteristic speed is *non-normally* (transiently) amplified —
   exactly neutral in spectrum for the frozen-coefficient scalar model —
   and the GH system's lower-order couplings (the quadratic source
   S0(∂g,∂g) and, independently, the metric self-coupling of the
   transport coefficients) close the feedback loop into true unstable
   point spectrum. At a sonic point the instability is *absolute*
   (local, boundary-independent); without one it is convective and
   becomes a global mode only when the domain recirculates it
   (periodicity). **Static backgrounds are continuum-stable** (the
   static shifted-Minkowski control converges to max Re λ → 0). No
   discretization choice can remove this layer; no constraint damping
   either (λ saturates at ~6.7 as γ0 → ∞ on the |V′| = 6.3 testbed,
   non-monotonically).

2. **A grid-scale discretization layer** (Im λ tracking the resolved
   band edge, Re λ up to ~2–3× the local coefficient-gradient scale,
   slowly *increasing* under refinement, present even on static
   backgrounds where the continuum is provably stable, identical in
   the full-3D and 1D-symmetric subspaces). This is the wide-stencil
   (D-of-D) + variable-coefficient mechanism. **Kreiss–Oliger
   dissipation removes it**: ε_KO = 0.5 collapses the Mx = 16
   moving-grid spectrum from +19.1 to +9.33 — exactly the
   Fourier-converged continuum value — and reduces the static-background
   layer from +5.8 to +0.78 (a mid-band residual that D⁶ reaches only
   weakly; γ0 = 4 further reduces it to +0.30).

**For black holes the continuum layer is benign at practical damping.**
The testbeds' |V′| (6–8/M) overstate the physical case by ~25×: at a
Kerr-Schild horizon the relevant gradient is the surface-gravity scale
κ = 1/(4M). Measured on an excised Kerr-Schild shell (R = [1.5, 4]M,
horizon inside, noise-perturbed): γ0 = 0 blows up at t ≈ 10–16M with
growth ≈ 0.24/M ≈ κ; **γ0 = 1/M is stable** with constraints flat at
truncation. The predecessor's conclusion ("no local cure exists")
conflated the artificially steep testbed — where indeed nothing wins —
with the physical regime, and lacked the KO-curable grid layer /
continuum split.

## Practical recipe (implemented defaults / documentation)

* Always run with Kreiss–Oliger dissipation near steep shift gradients
  or sonic surfaces: ε_KO ≈ 0.5 (removes the discretization layer; CFL
  cap handled by the driver).
* Constraint damping γ0 ≳ a few × the local characteristic-speed
  gradient (≈ a few/M for black holes; γ2 = 0 is fine).
* Do not benchmark excision stability on moving-grid testbeds with
  V′ ≫ κ: their continuum instability is real but unrepresentative,
  and undampable by design.

## Evidence table

| experiment (phase) | configuration | result |
|---|---|---|
| 1a | Fourier, periodic, Vmax=2 (sonic), Nx=32–256 | unstable, λ → +9.4, real, converged ⇒ continuum |
| 1a | Fourier, periodic, Vmax=0.5 (NO sonic) | unstable, λ → +1.19 ⇒ not sonic-specific in periodic domains |
| 1b | SBP quasi-1D, same backgrounds | λ ≈ +2.3/+24–26 with Im ~ band edge ⇒ extra discrete layer |
| 2 | eigenvector (Fourier, Vmax=2) | Π-dominated, 93% gauge-constraint-violating, localized at a sonic point, width ∝ 1/Nx ⇒ stagnation-point spectrum |
| 3a | source surgery | no single S0 block responsible (each removal worsens); principal-only (coefficient feedback) still +6.35 ≈ \|V′(x_s)\| |
| 3b | γ0 scan ≤ 256 | λ ≥ +6.7 always; non-monotone ⇒ damping cannot cure layer 1 |
| 6 | linear scalar conservative wave, same V | max Re λ = 0 exactly (both orientations) ⇒ bare transport only non-normal |
| 7 | frozen-coefficient GH | frozen+no-sources: 0; frozen+sources: +8.8; live+no-sources: +6.3 ⇒ either lower-order channel closes the loop |
| 4 | tanh excision, sonic inside | λ → +7.4 ≈ \|V′(x_s)\| = 7.5 at the sonic surface; **same closure, no sonic: stable (−0.005)** ⇒ excision face innocent |
| 5A/B | monotone stretch (no caustic/recirculation) | sonic: unstable; subluminal: grid layer only (+2.34, Im ~ band) |
| 5C | static shifted Minkowski (Fourier) | λ → 0 under refinement ⇒ static backgrounds continuum-stable |
| 5D | symmetric subspace ≡ full 3D SBP | grid layer is 1D, not transverse |
| KO | SBP Mx=16, Vmax=2, ε_KO=0.5 | +19.1 → **+9.33 = Fourier value** ⇒ grid layer cured |
| KO | static background layer | +5.8 → +0.78 (KO) → +0.30 (KO + γ0=4): mid-band residual, small |
| KS | excised Kerr-Schild shell, γ0=0 | blow-up at t ≈ 10–16M, rate ≈ 0.24/M ≈ κ |
| KS | same, γ0=1/M (and +KO) | stable, constraints flat at truncation |

## Relation to the predecessor's findings

All of its observations reproduce and are explained: resolution-robust
rate (continuum layer, converged); "violates only C^a" (93% here);
"scalar wave stable on the same background" (the scalar problem is
non-normal-neutral; GH's self-coupling closes the loop); "constant V
stable / sonic surface drives it" (no gradient ⇒ no pumping; phase 4);
"γ0, KO, Z4 relocation fail" (true for layer 1 at testbed strength —
but layer 2 is KO-curable and the physical-κ version of layer 1 is
γ0-curable, which the steep testbed masked).

## Post-script: what remains on the excised black hole

With the recipe applied (γ0 = 1/M, ε_KO = 0.5, M = 3/M_r = 5 shell),
the excised Kerr-Schild run keeps its **constraints flat at truncation
(≈ 2.6e-2) out to t ≈ 40M** — the constraint-sector instability that
killed the γ0 = 0 runs by t ≈ 10–16M is gone — but the solution
*drifts off the stationary background* at a slow, constraint-preserving
rate ≈ 0.14/M and the run fails near t ≈ 45M. This third phenomenon is
not the sonic-surface instability: it lives in the constraint-satisfying
(gauge/physical) sector. Prime suspects, in order: pure-gauge drift
(static H provides no restoring force on the gauge sector — the damped
harmonic gauge driver is the standard cure), the static-Dirichlet
boundary-data truncation mismatch acting as a secular source, and
under-resolution of the near-horizon region. Next investigation round.

### Harmonic-coordinate control (H ≡ 0)

To test whether the slow drift involves the static gauge-source sector,
the twin run was repeated with `Harmonic(1, 0)` (Kerr in fully harmonic
coordinates, `H ≡ 0`; horizon at coordinate radius R = M, shell
R = 0.75…4). Findings:

* **The harmonic chart is numerically much stiffer near the horizon**
  (it degenerates at areal r = M; metric functions carry
  (r+M)/(r−M)-type structure). At the resolution where Kerr-Schild
  holds constraints flat for 40M (M_r = 6 radial elements), the
  harmonic run dies on the first step (metric loses signature) — not a
  CFL effect (cfl/8 identical) and not the inner-radius choice. It
  takes M_r = 12 to evolve at all, and even then the initial truncation
  transient is ~9× larger (L² ≈ 2.5 vs 0.28). Empirical confirmation
  that Kerr-Schild-type slicings are the right excision substrate even
  for a *harmonic* code.
* At M_r = 12 the harmonic run survives to t ≈ 14.7M. Its gauge
  constraint grows at ≈ 0.26/M ≈ κ — i.e. γ0 = 1 does **not** hold the
  constraint sector in this chart at this (still marginal) resolution;
  the chart's steeper near-horizon gradients raise the coordinate
  pumping rate and the resolution continuously seeds the unstable
  sector.
* In its quasi-steady middle segment (t = 2…10) the constraint-
  *preserving* drift proceeds at ≈ 0.13/M — the same rate as the
  Kerr-Schild run's drift with flat constraints. **Weak evidence that
  the slow drift does not originate in the static gauge-source sector**
  (it appears with H ≡ 0 too), though the resolution confound keeps
  this short of conclusive. The gauge-driver hypothesis for layer 3
  stands.

## Open items

* The mid-band residual of the discrete layer (+0.3–0.8 on the steep
  static testbed) survives KO + γ0; candidate fixes: upwind SBP pairs
  for the β-advection, narrow-stencil variable-coefficient D2, or
  band-targeted filtering. Low priority at physical gradients.
* A direct 3D spectrum (Krylov) of the excised Kerr-Schild operator to
  attribute the γ0=0 blow-up rate (≈ κ) cleanly to the constraint
  sector.
* Long-horizon (≫ 50M) excised-BH runs and the damped-harmonic gauge
  driver for dynamical spacetimes.
