# TreeGeneralizedHarmonic.jl

[![CI](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeGeneralizedHarmonic.jl/actions/workflows/CI.yml)

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

**Status: scaffolding only — milestone G0 is done and G1 is next.** What
exists is the module shell, the `Base` bridges for software floating-point
types, the host-copy helpers, and the tests that say the pinned TreeAMR
still provides what the scheme is written against and that a
`SpacetimeMetrics` background — including the boosted, spinning one the
proof of concept runs — compiles and runs as a kernel argument. There are
no equations yet.

The formulation and the pointwise algebra are inherited from
`GeneralizedHarmonicSecondOrder2`, where they were validated on SBP-SAT
spectral elements; that repository is unpublished, so the documents this
package rests on are copied verbatim into [`notes/`](notes/README.md).

See [CODE.md](CODE.md) for the design and the measured results, and
[PLAN.md](PLAN.md) for the work breakdown.
