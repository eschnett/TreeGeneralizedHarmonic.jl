# TreeGeneralizedHarmonic.jl

[![CI](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/eschnett/TreeGeneralizedHarmonic.jl/graph/badge.svg?token=T3096JWXD3)](https://codecov.io/gh/eschnett/TreeGeneralizedHarmonic.jl)

`TreeGeneralizedHarmonic` solves the vacuum Einstein equations in the
generalized harmonic formulation — second order in space, first order in
time — on [TreeAMR](https://github.com/eschnett/TreeAMR.jl)'s octree of
uniform blocks. A coupled matter code can supply a stress-energy source
through `gh_rhs!`'s `matter` keyword, at this package's vertices
(`CODE.md`, "The coupling contract with TreeGRRMHD").

That is `∂ₜh_ab = β^i ∂_i h_ab + (α/√γ) Π_ab` and the second-order
reduction of `R_ab = 0` for `Π_ab`, with `h_ab = g_ab − η_ab` the offset
metric and `Π_ab = (√γ/α)(∂ₜ − β^i ∂_i) g_ab = √γ n^μ ∂_μ g_ab` the
densitised, Lie-advected momentum — not `∂ₜh_ab`, which is the first
equation — plus a prescribed gauge source `H_a`, constraint damping,
Kreiss–Oliger dissipation and RK4 in time. [TreeWave](https://github.com/eschnett/TreeWave.jl) shows the same
mesh under the scalar wave equation and
[TreeHydro](https://github.com/eschnett/TreeHydro.jl) under a conservative
finite-volume scheme; this package is general relativity, and the first
step toward a production code.

The proof of concept is a single **boosted, spinning black hole** crossing
an adaptively refined box, with **no excision**: inside the horizon the
right-hand side is modified by two smooth profiles of the distance to the
hole's analytic center — a relaxation toward the analytic solution in a
layer, and a switch-off around the singularity — and both depend on
position and time and on nothing about blocks, levels or ghost widths.
Initial data, boundary data, the interior's reference solution and the
error reference all come from one analytic background in
[SpacetimeMetrics](https://github.com/eschnett/SpacetimeMetrics.jl),
evaluated inside the kernels.

Every driver takes the floating-point type to compute in as a leading
positional argument, defaulting to `Float64`, and the KernelAbstractions
`backend` to run on as a keyword, defaulting to `CPU()`. `Float64` on an
H200 is the device requirement; no device package is a dependency of this
one.

**Contents**

- [Installation](#installation)
- [Example](#example)
- [Status](#status)
- [Documentation](#documentation)

## Installation

Its `Project.toml` pins five dependencies
to their GitHub repositories through `[sources]` and takes TreeAMR from the
General registry, so a checkout resolves with `Pkg.instantiate()`; the
supported Julia version is in [`CLAUDE.md`](CLAUDE.md), "Repository
facts".

## Example

The test suite, serially and at four threads:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                              # the suite
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])' # and threaded
```

## Status

**G0–G4 are done — there is a black hole on a mesh the code chose, and the
code knows where its horizon is: the interior damping layer, the driver
with its per-chunk analysis record and its regrid branch, the masked
Löhner indicator with its interior mask, its derived level floor and its
boundary ceiling, the masked error converging at order `q` on a frozen
hierarchy, and the apparent horizon with its area, `M_irr`, Korzyński `J`
and `M_ch` at Kerr's values in both charts and at `a = 9/10`. Before the
moving hole (G5) come PLAN.md's steps 8a–8g (added 2026-09-23), the
generic interior: step 5's layer needs an analytic center and an analytic
interior, and its spherical core cannot hold harmonic Kerr's singular
disk at `a = 9/10`, which is G5's case. Steps 8a (the leakage margin),
8b (the range projection), 8c (the calibration of the layer for an
inexact target), 8c′ (`ρ_max = 4/M` the default, decided 2026-09-23) and
8d (the tracked horizon geometry), 8e (the fitted target and the
`:fitted` variant, with the boost-sign fix) and 8f (the measurement matrix)
are done, and the generic interior is marked *(Done.)* under G5: `:fitted`
reaches `50 M` on the static Kerr-Schild hole at twice `:damped`'s error, so
step 8g (excision) is not needed; the analytic layer stays the default where
a chart admits it and `:fitted` is for G5's chart. **G5 runs at `a = 7/10`
(decided 2026-09-23)**, which runs at `h = 5/256` (2472 blocks, `10 M` in
1¾ node-hours) and not at `5/128`; harmonic `a = 9/10` waits with its price
written down in `SINGULARITY_HANDLING.md`'s "The interior's questions"
(`h ≲ 5/1024` on the equator,
23 000 blocks, `38 h` a `50 M` run, and a fit that holds 45° first). Step 8,
the moving hole, has run (2026-09-24) and G5 is **not** done: the mesh
follows the hole (the `:fitted` cycle on the case's own data, the floor from
the core surface widened by the travel), the Dirichlet data are exact, the
frozen hierarchy converges at order 2, the adaptive run is the uniform one
at the hole with 6–9× fewer points, the horizon is found along the
trajectory with Kerr's numbers and the boost's contraction, `:frozen` fails
— but on G5's chart the moving layer's error is `3.3×` the static hole's
at `7 M` and growing, carried out through the layer's trailing side
(`CODE.md`'s G5 and `SINGULARITY_HANDLING.md`'s "The moving hole (step 8)").
Step 8′ (2026-09-25) found
the lever for the trailing side — `evolve!(…; trail_ramp = 9/10)`, `ρ`'s ramp
narrowed where grid points leave the layer — which takes the crossing's
excess over the resting hole from `5.0×` to `2.05×` at `13 M` and makes the
two sides of the layer equal; G5 is still open on the uniform growth that is
left and on a drift of the horizon's `J` (`0.846` at `13 M`) that the moving
spinning hole shows with and without it (`SINGULARITY_HANDLING.md`, "The
trailing side
(step 8′)" and "The interior's questions").**
`PLAN.md` breaks the
milestones G0–G6 into steps 0–10, each a brief for one agent with a fresh
context (see its "Running a step as an agent"); step 8 (a hole that
moves) is next.

**Inconsistent:** the status says that step 8 (a hole that moves) is
next, and in the same paragraph that step 8, the moving hole, has run
(2026-09-24), and step 8′ (2026-09-25) after it.

## Documentation

See [CODE.md](CODE.md) for the design and the measured results,
[PLAN.md](PLAN.md) for the work breakdown, and [HISTORY.md](HISTORY.md)
for how the package got here. Checkpoint and restart are described in
`CODE.md`, "Checkpoint and restart", and the inherited GHSO2 documents
are in [`notes/`](notes/README.md).
