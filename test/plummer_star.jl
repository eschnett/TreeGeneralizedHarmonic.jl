# The Plummer star and its matter, held fixed (GH-3, added 2026-11-09; split out
# of `matter_tests.jl` on 2026-11-11 so that `test/star_runs.jl` uses the same
# fixture): the metric of a static star that is not a vacuum, and a matter adapter
# that answers its `(ρ, S_i, S_ij)` at TreeGH's owned vertices. Included by
# `matter_tests.jl` and by `star_runs.jl`; it defines no case, since each caller
# chooses its own `γ0`.

using LinearAlgebra: dot
using StaticArrays: SMatrix, SVector
import SpacetimeMetrics

# --- a static star ----------------------------------------------------------------

# A static star, spherically symmetric in areal coordinates: Plummer's mass
# function `m(r) = M r³/(r² + R²)^{3/2}` — whose density `ρ = 3MR²/(4π(r² +
# R²)^{5/2})` is positive everywhere — and Plummer's potential in the lapse,
# `e^{2ν} = 1 − 2M/√(r² + R²)`. Smooth everywhere, the centre included (`2m/r`
# and `m/r³` are functions of `r²`), with an anisotropic pressure, and its exact
# solution is itself, so an evolution with its matter has an error to measure.
#
# `λ` tilts the slicing, `t → t + λ√(r² + R²)`: the same star, still static (no
# `t` in the metric), but its slices are no longer orthogonal to the fluid, so the
# shift, `K_ij` and the momentum density `S_i` are not zero, and every term of
# `T_ab` and of the momentum constraint is exercised. `λ = 0` is the star's rest
# frame. One type for both, because `SpacetimeMetrics`' `EinsteinTensor` costs
# 75 s to compile per metric type (three layers of duals; measured 2026-11-09).
#
# `isbits`, generic in the number type so that the duals differentiate it, and not
# harmonic, so the problem samples its gauge source.
struct PlummerStar{T} <: SpacetimeMetrics.AbstractMetric
    M::T
    R::T
    λ::T
end

function SpacetimeMetrics.metric(m::PlummerStar, x::AbstractVector)
    X, Y, Z = x[2], x[3], x[4]
    s2 = X * X + Y * Y + Z * Z + m.R * m.R
    s = sqrt(s2)
    s3 = s2 * s
    e2ν = 1 - 2 * m.M / s
    # γ_ij = δ_ij + x_i x_j f, with f = (2M/s³)/(1 − 2m/r) and 2m/r = 2M r²/s³;
    # the tilt adds g_ti = e^{2ν} λ x_i/s and −e^{2ν} λ² x_i x_j/s² to g_ij.
    f = (2 * m.M / s3) / (1 - 2 * m.M * (s2 - m.R * m.R) / s3) -
        e2ν * m.λ * m.λ / s2
    k = e2ν * m.λ / s
    return SMatrix{4,4}(-e2ν, k * X, k * Y, k * Z,
                        k * X, 1 + X * X * f, X * Y * f, X * Z * f,
                        k * Y, X * Y * f, 1 + Y * Y * f, Y * Z * f,
                        k * Z, X * Z * f, Y * Z * f, 1 + Z * Z * f)
end
Base.nameof(::PlummerStar) = "Plummer star"

# --- a matter adapter, as TreeGRRMHD's would be ---------------------------------

# The matter of a metric, `T_ab = G_ab/8π` from `SpacetimeMetrics`' Einstein
# tensor (forward-mode duals, exact to roundoff), projected at every owned vertex
# into a 10-variable `G = 0` field set: `ρ`, `S_i`, `S_ij` (lower triangle). It is
# sampled once and then held: that is what makes an integration with it the
# anti-Cowling setting.
struct SampledMatter{F,W}
    forest::F
    work::W
end

function SampledMatter(::Type{T}, forest, bg; t=zero(T)) where {T}
    fs = FieldSet{T}(forest, 10; G=0, centering=vertexcentered(3))
    N = forest.N
    c8π = 8 * T(π)
    for b in 1:nblocks(fs), k in 1:N, j in 1:N, i in 1:N
        x = coordinates(fs, b, (i, j, k))
        p = SVector{4,T}(T(t), x[1], x[2], x[3])
        g = SpacetimeMetrics.metric(bg, p)
        Tab = SpacetimeMetrics.EinsteinTensor(bg, p) / c8π
        α, β, _ = SpacetimeMetrics.adm_decompose(g)
        nu = SVector{4,T}(1 / α, -β[1] / α, -β[2] / α, -β[3] / α)
        ρ = dot(nu, Tab * nu)
        # γ_i^a = δ_i^a, since n_i = 0: S_i = −T_ib n^b, S_ij = T_ij.
        S = -(Tab * nu)
        vals = (ρ, S[2], S[3], S[4], Tab[2, 2], Tab[3, 2], Tab[4, 2], Tab[3, 3],
                Tab[4, 3], Tab[4, 4])
        for v in 1:10
            fs.work[i, j, k, v, b] = vals[v]
        end
    end
    return SampledMatter(forest, fs.work)
end

# The kernel side: the array and nothing else.
struct BoundSampledMatter{W}
    work::W
end
TreeGeneralizedHarmonic.Adapt.@adapt_structure BoundSampledMatter

function (m::SampledMatter)(U::FieldSet)
    U.forest === m.forest || throw(ArgumentError(
        "this matter was sampled on another forest"))
    all(c -> c === :vertex, U.centering) || throw(ArgumentError(
        "the matter answers at vertices"))
    return BoundSampledMatter(m.work)
end

@inline function (at::BoundSampledMatter)(n::NTuple{3,Int}, b::Int)
    w = at.work
    i, j, k = n
    @inbounds begin
        ρ = w[i, j, k, 1, b]
        S = SVector(w[i, j, k, 2, b], w[i, j, k, 3, b], w[i, j, k, 4, b])
        s11, s21, s31 = w[i, j, k, 5, b], w[i, j, k, 6, b], w[i, j, k, 7, b]
        s22, s32, s33 = w[i, j, k, 8, b], w[i, j, k, 9, b], w[i, j, k, 10, b]
    end
    return (; ρ, S, Sij=SMatrix{3,3}(s11, s21, s31, s21, s22, s32, s31, s32, s33))
end

