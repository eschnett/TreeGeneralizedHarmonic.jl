# TreeGeneralizedHarmonic.jl

[![CI](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/eschnett/TreeGeneralizedHarmonic.jl/graph/badge.svg?token=T3096JWXD3)](https://codecov.io/gh/eschnett/TreeGeneralizedHarmonic.jl)

`TreeGeneralizedHarmonic` solves the vacuum Einstein equations in the
generalized harmonic formulation — second order in space, first order in
time — on [TreeAMR](https://github.com/eschnett/TreeAMR.jl)'s octree of
uniform blocks.

That is `∂ₜh_ab = β^i ∂_i h_ab + (α/√γ) Π_ab` and the second-order
reduction of `R_ab = 0` for `Π_ab`, with `h_ab = g_ab − η_ab` the offset
metric and `Π_ab = (√γ/α)(∂ₜ − β^i ∂_i) g_ab = √|g| n^μ ∂_μ g_ab` the
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

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                              # the suite
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])' # and threaded
```

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

The formulation and the pointwise algebra are inherited from
`GeneralizedHarmonicSecondOrder2`, where they were validated on SBP-SAT
spectral elements; that repository is unpublished, so the documents this
package rests on are copied verbatim into [`notes/`](notes/README.md).

See [CODE.md](CODE.md) for the design and the measured results, and
[PLAN.md](PLAN.md) for the work breakdown.
