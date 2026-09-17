"""
    TreeGeneralizedHarmonic

The vacuum Einstein equations in the generalized harmonic formulation —
second order in space, first order in time — on
[TreeAMR](https://github.com/eschnett/TreeAMR.jl)'s octree of uniform
blocks:

    ∂ₜh_ab = β^i ∂_i h_ab + (α/√γ) Π_ab
    ∂ₜΠ_ab = (the second-order reduction of  R_ab = 0,  expanded)

with `h_ab = g_ab − η_ab` the offset metric and

    Π_ab = (√γ/α)(∂ₜ − β^i ∂_i) g_ab = √|g| n^μ ∂_μ g_ab

the **densitised, Lie-advected momentum** — the derivative along the unit
normal, weighted by `√|g|`, and *not* `∂ₜ h_ab`, which is the first line.
`α`, `β^i` and `√γ` are read off `h` pointwise. A prescribed gauge source
`H_a` and the usual constraint damping close the system; see "The
equations" in `CODE.md`.

TreeAMR supplies the mesh and its operations and deliberately contains no
physics, so this package is where the physics lives.
[TreeWave](https://github.com/eschnett/TreeWave.jl) shows the same mesh
under the scalar wave equation and
[TreeHydro](https://github.com/eschnett/TreeHydro.jl) under a conservative
finite-volume scheme; this one is general relativity, and the first step
toward a production code. The formulation and the pointwise algebra are
inherited from `GeneralizedHarmonicSecondOrder2`, where they were
validated on SBP-SAT spectral elements; the documents that record them are
copied verbatim into `notes/`.

The proof of concept is a single **boosted, spinning black hole** crossing
an adaptively refined box, with no excision: inside the horizon the
right-hand side is modified by two smooth profiles of the distance to the
hole's analytic center — a relaxation toward the analytic solution in a
layer, and a switch-off around the singularity — both functions of
position and time and of nothing about blocks, levels or ghost widths.

Every driver takes the floating-point type it computes in as a leading
positional argument (default `Float64`) and the KernelAbstractions
`backend` it runs on as a keyword (default `CPU()`), so the same study
runs at `Float32` on a device as at `Float64` on the host. No device
package is a dependency of this one.

See `CODE.md` in the package root for the design document — what is here,
why, and every measured number — and `PLAN.md` for the work breakdown.
"""
module TreeGeneralizedHarmonic

using TreeAMR

import SpacetimeMetrics
using KernelAbstractions: @Const, @index, @kernel
using KernelAbstractions: Backend, CPU, allocate, get_backend
using LinearAlgebra: det, dot, tr
using SpacetimeMetrics: AbstractMetric, GaugeWave, Harmonic, KerrSchild,
                        Minkowski, ShiftedMinkowski, dmetric, gauge_source_grad
using StaticArrays: SArray, SMatrix, SVector

# Devices
export hostcopy

# Pointwise algebra
export pack_g, pack_sym
export metric_quantities, metric_derivatives
export adm_from_metric, adm_vars_from_state, gauge_constraint_at_node
export gh_node_rhs, gh_node_source, gh_node_rhs_expanded

# Stencils
export derivative_weights, dissipation_weights, dissipation_rank
export apply_stencil, apply_mixed_stencil

# Gauge sources
export isharmonic, isstatic, sample_gauge_source!

# Cases, backgrounds and initial data
export GHCase, gh_forest
export minkowski_case, gauge_wave_case, shifted_minkowski_case
export background_state, state_tuple, state_callback, fill_exact!

# Boundaries
export dirichlet

# Evolution
export GHProblem, gh_rhs!, gh_dt, max_speed, convergence_rate

include("precision.jl")
include("device.jl")
include("pointwise.jl")
include("stencils.jl")
include("gauge.jl")
include("initialdata.jl")
include("boundaries.jl")
include("evolution.jl")

end
