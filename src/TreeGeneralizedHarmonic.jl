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

using KernelAbstractions: Backend, CPU, allocate, get_backend

# Devices
export hostcopy

include("precision.jl")
include("device.jl")

end
