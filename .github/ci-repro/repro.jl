# Standalone reproducer: `adm_vars_from_state` (TreeGeneralizedHarmonic's
# ADM extraction, copied verbatim with its helpers) in Float64 against the
# same code in BigFloat, on deterministic random states.
#
#   julia --project=. --code-coverage=@$(pwd) --check-bounds=yes repro.jl
#
# prints the worst relative error of γ, ∂γ and K and exits non-zero if any
# exceeds 1e-10 (Float64 roundoff here is ~1e-15).
using LinearAlgebra: det, tr
using StaticArrays
using Random: MersenneTwister

const NC = 10

@inline _η4(::Type{T}) where {T} =
    SMatrix{4,4,T}(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)

@inline function _sym4(v::SVector{NC,T}) where {T}
    return SMatrix{4,4,T}(v[1], v[2], v[3], v[4],
                          v[2], v[5], v[6], v[7],
                          v[3], v[6], v[8], v[9],
                          v[4], v[7], v[9], v[10])
end

@inline function metric_quantities(h::SMatrix{4,4,T}) where {T}
    η = _η4(T)
    g4 = η + h
    # Inverse-metric offset: inv(A) − inv(B) = −inv(A)(A−B)inv(B) with
    # A = η+h, B = η (and inv(η) = η) gives gu − η = −gu·h·η. Evaluating
    # the right side with the directly-computed inverse keeps the offset
    # accurate to a relative eps even when ‖h‖ ≪ 1.
    guo = -(inv(g4) * h) * η
    gu4 = η + guo
    gutt = gu4[1, 1]
    q1 = guo[1, 1]                       # g^{tt} + 1
    α = inv(sqrt(one(T) - q1))           # 1/√(−g^{tt})
    β = SVector{3,T}(-gu4[1,2] / gutt, -gu4[1,3] / gutt, -gu4[1,4] / gutt)
    γu = SMatrix{3,3,T}(
        gu4[1+1,1+1] - gu4[1,1+1]*gu4[1,1+1]/gutt,
        gu4[2+1,1+1] - gu4[1,2+1]*gu4[1,1+1]/gutt,
        gu4[3+1,1+1] - gu4[1,3+1]*gu4[1,1+1]/gutt,
        gu4[1+1,2+1] - gu4[1,1+1]*gu4[1,2+1]/gutt,
        gu4[2+1,2+1] - gu4[1,2+1]*gu4[1,2+1]/gutt,
        gu4[3+1,2+1] - gu4[1,3+1]*gu4[1,2+1]/gutt,
        gu4[1+1,3+1] - gu4[1,1+1]*gu4[1,3+1]/gutt,
        gu4[2+1,3+1] - gu4[1,2+1]*gu4[1,3+1]/gutt,
        gu4[3+1,3+1] - gu4[1,3+1]*gu4[1,3+1]/gutt)
    # det(g)+1 from the elementary symmetric polynomials of B = η·h:
    # det(η+h) = det(η)·det(I+B) = −(1 + e1 + e2 + e3 + e4).
    B = η * h
    trB = tr(B)
    B2 = B * B
    trB2 = tr(B2)
    trB3 = tr(B2 * B)
    e1 = trB
    e2 = (trB*trB - trB2) / 2
    e3 = (trB*trB*trB - 3*trB*trB2 + 2*trB3) / 6
    e4 = det(B)
    d1 = -(e1 + e2 + e3 + e4)            # det(g4) + 1
    # det(γ) = det(g4)·g^{tt} ⇒ det(γ) − 1 = −d1 − q1 + d1·q1.
    detγm1 = -d1 - q1 + d1*q1
    sqrtγ = sqrt(one(T) + detγm1)
    return g4, gu4, α, β, γu, sqrtγ, guo
end

@inline function adm_vars_from_state(h::SVector{NC,T}, Pi::SVector{NC,T},
                                     dxh::SVector{NC,T}, dyh::SVector{NC,T},
                                     dzh::SVector{NC,T}) where {T}
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    dtg = β[1]*dxh + β[2]*dyh + β[3]*dzh + (α / sqrtγ) * Pi
    dt4 = _sym4(dtg)
    dx4 = _sym4(dxh); dy4 = _sym4(dyh); dz4 = _sym4(dzh)
    γ = SMatrix{3,3,T}(g4[i+1, j+1] for i in 1:3, j in 1:3)
    ∂γ = SArray{Tuple{3,3,3},T}(
        (k == 1 ? dx4[i+1, j+1] : k == 2 ? dy4[i+1, j+1] : dz4[i+1, j+1])
        for i in 1:3, j in 1:3, k in 1:3)
    # Lowered shift and its derivatives: β_j = g_{tj}.
    βl = SVector{3,T}(g4[1, 2], g4[1, 3], g4[1, 4])
    dβl = SMatrix{3,3,T}(
        (i == 1 ? dx4[1, j+1] : i == 2 ? dy4[1, j+1] : dz4[1, j+1])
        for i in 1:3, j in 1:3)                     # dβl[i,j] = ∂_i β_j
    # 3-Christoffels Γ^k_{ij} = ½ γ^{kl}(∂_i γ_lj + ∂_j γ_li − ∂_l γ_ij).
    Γ3 = SArray{Tuple{3,3,3},T}(
        sum(γu[k, l] * (∂γ[l, j, i] + ∂γ[l, i, j] - ∂γ[i, j, l]) / 2
            for l in 1:3)
        for k in 1:3, i in 1:3, j in 1:3)
    K = SMatrix{3,3,T}(
        -(dt4[i+1, j+1] - dβl[i, j] - dβl[j, i] +
          2 * sum(Γ3[k, i, j] * βl[k] for k in 1:3)) / (2α)
        for i in 1:3, j in 1:3)
    return γ, ∂γ, (K + K') / 2
end


function states(n)
    rng = MersenneTwister(1)
    r() = 2 * rand(rng) - 1
    out = []
    for _ in 1:n
        # A Lorentzian metric near η: offsets of order 0.2.
        h = SVector{NC,Float64}(ntuple(_ -> r() / 5, NC))
        Π = SVector{NC,Float64}(ntuple(_ -> r(), NC))
        dh = ntuple(_ -> SVector{NC,Float64}(ntuple(_ -> r(), NC)), 3)
        push!(out, (h, Π, dh))
    end
    return out
end

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1)

function main()
    setprecision(BigFloat, 256)
    worst = zeros(3)
    nbad = 0
    for (h, Π, dh) in states(200)
        γ, ∂γ, K = adm_vars_from_state(h, Π, dh[1], dh[2], dh[3])
        B(v) = SVector{NC,BigFloat}(big.(v))
        γb, ∂γb, Kb = adm_vars_from_state(B(h), B(Π), B(dh[1]), B(dh[2]), B(dh[3]))
        e = (relerr(γ, γb), relerr(∂γ, ∂γb), relerr(K, Kb))
        e = map(x -> isnan(x) ? Inf : Float64(x), e)
        worst .= max.(worst, collect(e))
        nbad += any(>(1e-10), e)
    end
    println("REPRO julia=", VERSION, " target=", Sys.CPU_NAME,
            " coverage=", Base.JLOptions().code_coverage,
            " check_bounds=", Base.JLOptions().check_bounds,
            " worst rel err γ=", worst[1], " ∂γ=", worst[2], " K=", worst[3],
            " bad states=", nbad, "/200")
    return nbad == 0
end

main() || exit(1)
