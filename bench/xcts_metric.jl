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
using StaticArrays: SVector, SMatrix, SArray
using LinearAlgebra: inv
import SpacetimeMetrics as SM

struct XCTSMetric{M,U,D,H,R,G} <: SM.AbstractMetric
    m::M          # the free data
    U::U          # the solution, field-major [ψ; β¹; β²; β³; Q]
    dof::D
    mesh::H
    refel::R
    gauge::G      # the metric the lapse and shift come from, or `nothing` (the solution's)
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
    return XCTSMetric(d.m, d.U, d.dof, d.mesh, ReferenceElement(Float64, d.p), gauge, info)
end

function SM.dmetric(x::XCTSMetric, p::AbstractVector)
    f = XCTS.solved_spacetime(x.m, x.U, x.dof, x.mesh, x.refel, SVector{3,Float64}(p[2], p[3], p[4]))
    x.gauge === nothing && return metric_from_3p1(f)
    g = gauge_3p1(x.gauge, p)
    return metric_from_3p1((; f.γ, f.∂γ, f.K, g.α, g.dα, g.β, g.dβ))
end

"""
    gauge_3p1(gauge, p) -> (; α, dα, β, dβ)

The lapse `α`, its gradient `dα[k] = ∂_kα`, the contravariant shift `β^i` and
`dβ[i, k] = ∂_kβ^i` of the metric `gauge` at the event `p`, from its `g_ab` and
`∂_c g_ab`: `β_i = g_ti`, `β^i = γ^{ij}β_j`, `α² = −g_tt + β^iβ_i`, with
`∂_kγ^{ij} = −γ^{ia}γ^{jb}∂_kγ_ab`.
"""
function gauge_3p1(gauge, p::AbstractVector)
    g, dg = SM.dmetric(gauge, SVector{4,Float64}(p[1], p[2], p[3], p[4]))
    γ = SMatrix{3,3,Float64}(g[i + 1, j + 1] for i in 1:3, j in 1:3)
    γu = inv(γ)
    βl = SVector{3,Float64}(g[1, i + 1] for i in 1:3)
    β = γu * βl
    α = sqrt(-g[1, 1] + sum(β[i] * βl[i] for i in 1:3))
    ∂γ(a, b, k) = dg[a + 1, b + 1, k + 1]
    ∂γu(i, j, k) = -sum(γu[i, a] * γu[j, b] * ∂γ(a, b, k) for a in 1:3, b in 1:3)
    ∂βl(i, k) = dg[1, i + 1, k + 1]
    dβ = SMatrix{3,3,Float64}(sum(∂γu(i, j, k) * βl[j] + γu[i, j] * ∂βl(j, k) for j in 1:3)
                              for i in 1:3, k in 1:3)
    dα = SVector{3,Float64}((-dg[1, 1, k + 1] +
                             sum(dβ[i, k] * βl[i] + β[i] * ∂βl(i, k) for i in 1:3)) / (2α)
                            for k in 1:3)
    return (; α, dα, β, dβ)
end

"""
    metric_from_3p1(f) -> (g, dg)

`g_ab` and `dg[a, b, c] = ∂_c g_ab` from the 3+1 fields `f` — `α`, `dα`, `β`, `dβ`,
`γ`, `∂γ`, `K` as `XCTS.solved_spacetime` returns them — with `∂_t g_ij` from `K_ij` and
`∂_t g_tb` from `C_a = 0` at `H = 0`. Separate from the solution so that it can be
checked on exact data.
"""
function metric_from_3p1(f)
    α, β, γ, ∂γ, dβ, dα, K = f.α, f.β, f.γ, f.∂γ, f.dβ, f.dα, f.K
    βl = γ * β                                                  # β_i
    # ∂_k β_i = ∂_k γ_ij β^j + γ_ij ∂_k β^j
    dβl = SMatrix{3,3,Float64}(sum(∂γ[i, j, k] * β[j] + γ[i, j] * dβ[j, k] for j in 1:3)
                               for i in 1:3, k in 1:3)          # dβl[i,k] = ∂_k β_i
    gtt = -α^2 + sum(βl[i] * β[i] for i in 1:3)
    g = SMatrix{4,4,Float64}(a == 1 && b == 1 ? gtt :
                             a == 1 ? βl[b - 1] : b == 1 ? βl[a - 1] : γ[a - 1, b - 1]
                             for a in 1:4, b in 1:4)
    # The spatial derivatives, `ds[a, b, k] = ∂_k g_ab`.
    dgtt(k) = -2α * dα[k] + sum(dβ[i, k] * βl[i] + β[i] * dβl[i, k] for i in 1:3)
    ds = SArray{Tuple{4,4,3},Float64}(a == 1 && b == 1 ? dgtt(k) :
                                      a == 1 ? dβl[b - 1, k] : b == 1 ? dβl[a - 1, k] :
                                      ∂γ[a - 1, b - 1, k]
                                      for a in 1:4, b in 1:4, k in 1:3)
    # ∂_t γ_ij = −2αK_ij + D_iβ_j + D_jβ_i, D_iβ_j = ∂_iβ_j − Γ^l_ij β_l.
    γu = inv(γ)
    Γl(l, i, j) = (∂γ[l, j, i] + ∂γ[l, i, j] - ∂γ[i, j, l]) / 2  # Γ_lij
    Dβ(i, j) = dβl[j, i] - sum(γu[l, m] * Γl(m, i, j) * βl[l] for l in 1:3, m in 1:3)
    dtγ = SMatrix{3,3,Float64}(-2α * K[i, j] + Dβ(i, j) + Dβ(j, i) for i in 1:3, j in 1:3)
    # ∂_t g_tb from C_a = Γ_a = 0 (H = 0): Γ_a = g^{bc} ∂_b g_ac − ½ g^{bc} ∂_a g_bc,
    # with X_b = ∂_t g_tb entering as Γ_t = Γ⁰_t + ½ g^{tt} X_t and
    # Γ_i = Γ⁰_i + g^{tt} X_i (`gauge_consistent_state`'s four decoupled equations).
    gu = inv(g)
    dz(a, b, c) = c == 1 ? ((a == 1 || b == 1) ? 0.0 : dtγ[a - 1, b - 1]) : ds[a, b, c - 1]
    Γ0 = SVector{4,Float64}(sum(gu[b, c] * dz(a, c, b) for b in 1:4, c in 1:4) -
                            sum(gu[b, c] * dz(b, c, a) for b in 1:4, c in 1:4) / 2
                            for a in 1:4)
    Xt = -2 * Γ0[1] / gu[1, 1]
    Xi = SVector{3,Float64}(-Γ0[i + 1] / gu[1, 1] for i in 1:3)
    dt(a, b) = a == 1 && b == 1 ? Xt : a == 1 ? Xi[b - 1] : b == 1 ? Xi[a - 1] :
               dtγ[a - 1, b - 1]
    dg = SArray{Tuple{4,4,4},Float64}(c == 1 ? dt(a, b) : ds[a, b, c - 1]
                                      for a in 1:4, b in 1:4, c in 1:4)
    return g, dg
end

SM.metric(x::XCTSMetric, p::AbstractVector) = SM.dmetric(x, p)[1]
