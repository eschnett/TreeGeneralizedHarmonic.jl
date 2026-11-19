# Data solved by XCTS as a SpacetimeMetrics metric (step S1; `PLAN.md`, "Step S1 —
# Solved data for two harmonic holes"; `CODE.md`, "Solved data for the pair").
#
# `test/binary_runs.jl xcts=<file>` and `test/octant_runs.jl xcts=<file>` include this
# file and hand `load_xcts_metric(file)` to `evolve!` as `initial_data`, with
# `t_end = 0` and a checkpoint: the initial-data cycle, the fills, the holes' finds and
# their fits at `t = 0` are then made from the solved data, and the file that comes out
# is continued by the same command without `xcts=` (a restart), on a host that has no
# XCTS. The file is what `examples/harmonic_holes.jl save=<file>` (XCTS's branch
# `claude/harmonic-free-data`) writes by `Serialization`.
#
# **It runs in an environment of its own** that has both packages (`PLAN.md`, "Sharp
# edges for steps S1–S2"): this package's `Project.toml` never names XCTS. Stack it,
#
#     JULIA_LOAD_PATH="@:<env with XCTS>:@stdlib" julia --project=. test/binary_runs.jl …
#
# so that SpacetimeMetrics is this package's and XCTS's at once (one copy loaded).
#
# **The metric.** `dmetric` returns the 4-metric `g_ab` from `α = Q/ψ`, `β^i` and
# `γ_ij = ψ⁴ γ̃_ij`, its spatial derivatives from XCTS's spectral basis (exact
# derivatives of the element's polynomial, not differences), `∂_t g_ij = −2αK_ij +
# D_iβ_j + D_jβ_i` from the solved `K_ij`, and `∂_t g_tb` from `C_a = 0` at `H = 0` —
# the harmonic gauge's momentum, as `_gauge_consistent_background` makes it for a
# superposition. The data are a snapshot: the metric is the same at every `t`, which is
# all `t = 0` asks of it. It is a host object (it holds the solution's arrays), so it
# runs on the CPU backend only, which is where the run's start is written.

using XCTS: XCTS
using HexCGSEM: ReferenceElement
using Serialization: deserialize
using StaticArrays: SVector
using TreeGeneralizedHarmonic: TreeGeneralizedHarmonic, center_at
import SpacetimeMetrics as SM

struct XCTSMetric{M,U,D,H,R,G} <: SM.AbstractMetric
    m::M          # the free data
    U::U          # the solution, field-major [ψ; β¹; β²; β³; Q]
    dof::D
    mesh::H
    refel::R
    gauge::G      # the metric the lapse and shift come from, or `nothing` (the solution's)
    R1::Float64   # the radius of the inner spheres, inside which the data are the free data
    centers::Vector{SVector{3,Float64}}  # the spheres' centers
    info::String  # what was solved, for `nameof` and the log
end

Base.nameof(x::XCTSMetric) = "XCTS-solved data ($(x.info))"
Base.show(io::IO, x::XCTSMetric) = print(io, nameof(x))

"""
    load_xcts_metric(path; gauge = nothing) -> XCTSMetric

The solution `examples/harmonic_holes.jl save=<path>` wrote, as a metric. With a
`gauge` metric, the lapse, the shift `β^i` and their spatial derivatives are that
metric's, and only `γ_ij`, `∂_k γ_ij` and `K_ij` are the solution's (`gauge_3p1`).

**Why a gauge.** The constraints are `γ_ij` and `K_ij`; the lapse and the shift are a
free choice on the slice. XCTS's fixed lapse (`lapse = :fixed`) takes the densitized
lapse `α̃ = αψ⁻⁶` from free data that its attenuation makes flat far from the holes, so
its lapse grows outward as `ψ⁶ ≈ 1 + 3M/r` (`1.097` at `|x| = 32` for the `d = 8` pair):
against Dirichlet data from the case's superposition, a jump of `0.27` in `g_ab` at the
outer faces. Taking the lapse and the shift from the case's background removes it, and
gives the evolution the gauge its boundary data and its layer targets carry.
"""
function load_xcts_metric(path::AbstractString; gauge = nothing)
    d = deserialize(path)
    info = "$(length(d.centers)) hole(s), d = $(d.d), mass $(d.Mass) each, R1 = $(d.R1), " *
           "p = $(d.p), $(d.dof.ndofs) DOFs/field, from $(basename(path))" *
           (gauge === nothing ? "" : "; lapse and shift from $(nameof(gauge))")
    return XCTSMetric(d.m, d.U, d.dof, d.mesh, ReferenceElement(Float64, d.p), gauge,
                      Float64(d.R1), [SVector{3,Float64}(c) for c in d.centers], info)
end

"""
    check_xcts_reach(x::XCTSMetric, geometry, depth)

Refuse a start whose layer reads the data inside XCTS's inner spheres. Inside a sphere
`|x − c| < R1` the solution is Dirichlet to the free data, so the data there are the
unsolved superposition, joined to the solution with a jump in their derivatives at
`R1`. A `:fitted` start reads the data from the offset surface `r_1(n̂)` down to the
switch depth (`fit_initial_depth`), and the fit's extrapolation below it: the least
radius read about a hole is `r_in − offset − depth`, about the layer's center, which a
companion displaces from the sphere's. So every hole needs

    r_in − offset − depth − |c_layer − c_sphere| ≥ R1,

which `fit_depth = 0` (the decision) meets by the margin, and `fit_depth = n_L h` meets
only with the sphere inside the core surface (`PLAN.md`, "Sharp edges for steps
S1–S2"). `geometry` is the run's (`evolve!`'s result's `geometry`): a
`FittedInterior`, or a `HoleInteriors` of them.
"""
function check_xcts_reach(x::XCTSMetric, geometry, depth)
    geoms = hasproperty(geometry, :holes) ? geometry.holes : (geometry,)
    for g in geoms
        c = SVector{3,Float64}(Tuple(center_at(g.center, zero(g.offset))))
        cX = x.centers[argmin([sum(abs2, c - cx) for cx in x.centers])]
        reach = Float64(g.r_in - g.offset - depth) - sqrt(sum(abs2, c - cX))
        reach ≥ x.R1 || throw(ArgumentError(
            "the :fitted start reads the data down to $reach from XCTS's sphere at " *
            "$(Tuple(cX)), inside its R1 = $(x.R1), where the data are the free data: " *
            "use fit_depth = 0, or a solution with a smaller R1 (r_in = $(g.r_in), " *
            "offset = $(g.offset), depth = $depth)."))
    end
    return nothing
end

function SM.dmetric(x::XCTSMetric, p::AbstractVector)
    f = XCTS.solved_spacetime(x.m, x.U, x.dof, x.mesh, x.refel, SVector{3,Float64}(p[2], p[3], p[4]))
    x.gauge === nothing && return TreeGeneralizedHarmonic.metric_from_3p1(f)
    g = TreeGeneralizedHarmonic.gauge_3p1(x.gauge, p)
    return TreeGeneralizedHarmonic.metric_from_3p1((; f.γ, f.∂γ, f.K, g.α, g.dα, g.β, g.dβ))
end

SM.metric(x::XCTSMetric, p::AbstractVector) = SM.dmetric(x, p)[1]
