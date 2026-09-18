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
using ApparentHorizonFinder: ADMVars, find_horizon, horizon_grid,
                             horizon_points
using KernelAbstractions: @Const, @index, @kernel
using KernelAbstractions: Backend, CPU, allocate, get_backend
using KorzynskiSpin: horizon_spin
using LinearAlgebra: det, dot, tr
using OrdinaryDiffEqLowOrderRK: RK4
using SciMLBase: ODEProblem, solve
using SpacetimeMetrics: AbstractMetric, GaugeWave, Harmonic, KerrSchild,
                        Minkowski, ShiftedMinkowski, dmetric, gauge_source_grad
using StaticArrays: SArray, SMatrix, SVector

# Devices
export hostcopy

# Pointwise algebra
export pack_g, pack_sym
export metric_quantities, metric_derivatives, metric_derivatives_along
export adm_from_metric, adm_vars_from_state, gauge_constraint_at_node
export gh_node_rhs, gh_node_source, gh_node_rhs_expanded

# Stencils
export derivative_weights, dissipation_weights, dissipation_rank
export apply_stencil, apply_mixed_stencil

# Gauge sources
export isharmonic, isstatic, sample_gauge_source!

# Cases, backgrounds and initial data
export GHCase, with_interior, with_refinement, with_horizon
export gh_forest, hole_forest
export minkowski_case, gauge_wave_case, shifted_minkowski_case
export hole_case, kerr_schild_case, harmonic_kerr_case
export background_state, state_tuple, case_state_tuple, state_callback
export fill_exact!

# Gauge and constraint damping, as a function of position
export ConstantDamping, GaussianDamping, damping_rate, damping_bounds

# The interior: the damping layer and the frozen core
export HoleCenter, center_at, Interior, with_ρ_max, interior_variant
export interior_profiles, is_frozen, in_layer, core_position, smoothstep
export InteriorMask, ShellMask, interior_mask
export horizon_min_radius, horizon_max_radius, singular_radius
export layer_spacing, check_interior_radii

# Boundaries
export dirichlet

# Evolution
export GHProblem, gh_rhs!, gh_dt, max_speed, convergence_rate
export gh_step_limiter!, paste_interior!

# Constraints and the masks their norms take
export AllPoints, is_evolved, adm_constraints_at_node
export gh_constraint!, adm_constraint!, gh_error!
export masked_counts, masked_norms, constraint_norms, error_norms

# Refinement: the masked Löhner indicator, its marks and its bounds
export Refinement, lohner, cell_tau, field_scales, field_scale
export gh_tau!, tau_max, gh_indicator!, indicator_flags, refine_flags
export LevelBounds, level_bounds, block_level_bounds, horizon_floor_level
export refinement_buffer, refinement_centroid

# The horizon: the stopgap interpolator, the ADM provider, the find
export Horizon, locate_block, interpolate, interpolate_grad
export GHADMProvider, gh_adm_provider, find_gh_horizon, horizon_radii

# The driver
export evolve!, check_cfl, forest_levels, horizon_shell
export discrete_gradient_momentum!

include("precision.jl")
include("device.jl")
include("pointwise.jl")
include("stencils.jl")
include("interior.jl")
include("gauge.jl")
include("initialdata.jl")
include("boundaries.jl")
include("evolution.jl")
include("constraints.jl")
# After `constraints.jl` and before `driver.jl`: the indicator dispatches on
# `GHCase` and `GHProblem` and reuses the monitors' scatter-and-fill
# preamble, and the driver is what calls it.
include("refinement.jl")
# After `evolution.jl` (it reads a `GHProblem`'s field set, `q` and
# interior) and before `driver.jl`, which calls it once every `k`-th chunk.
include("horizon.jl")
include("driver.jl")

end
