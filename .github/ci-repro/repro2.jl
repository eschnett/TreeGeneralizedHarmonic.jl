# Which part of `K` in `adm_vars_from_state` goes wrong? Each piece, verbatim,
# on random inputs of its own, Float64 against BigFloat.
#
#   julia --project=. --code-coverage=@$(pwd) --check-bounds=yes repro2.jl
using StaticArrays
using Random: MersenneTwister

christoffel(γu::SMatrix{3,3,T}, ∂γ::SArray{Tuple{3,3,3},T}) where {T} =
    SArray{Tuple{3,3,3},T}(
        sum(γu[k, l] * (∂γ[l, j, i] + ∂γ[l, i, j] - ∂γ[i, j, l]) / 2
            for l in 1:3)
        for k in 1:3, i in 1:3, j in 1:3)

contract(Γ3::SArray{Tuple{3,3,3},T}, βl::SVector{3,T}) where {T} =
    SMatrix{3,3,T}(sum(Γ3[k, i, j] * βl[k] for k in 1:3) for i in 1:3, j in 1:3)

function kassemble(dt4::SMatrix{4,4,T}, dβl::SMatrix{3,3,T},
                   Γ3::SArray{Tuple{3,3,3},T}, βl::SVector{3,T}, α::T) where {T}
    K = SMatrix{3,3,T}(
        -(dt4[i+1, j+1] - dβl[i, j] - dβl[j, i] +
          2 * sum(Γ3[k, i, j] * βl[k] for k in 1:3)) / (2α)
        for i in 1:3, j in 1:3)
    return (K + K') / 2
end

# The same K with the contraction written out, no generator inside a generator.
function kassemble_flat(dt4::SMatrix{4,4,T}, dβl::SMatrix{3,3,T},
                        Γ3::SArray{Tuple{3,3,3},T}, βl::SVector{3,T}, α::T) where {T}
    K = SMatrix{3,3,T}(
        -(dt4[i+1, j+1] - dβl[i, j] - dβl[j, i] +
          2 * (Γ3[1, i, j] * βl[1] + Γ3[2, i, j] * βl[2] + Γ3[3, i, j] * βl[3])) / (2α)
        for i in 1:3, j in 1:3)
    return (K + K') / 2
end

relerr(a, b) = Float64(maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1))

function main()
    setprecision(BigFloat, 256)
    rng = MersenneTwister(2)
    r() = 2 * rand(rng) - 1
    worst = Dict{String,Float64}()
    bump!(k, e) = (worst[k] = max(get(worst, k, 0.0), isnan(e) ? Inf : e))
    B(x) = big.(x)
    for _ in 1:200
        γu = SMatrix{3,3,Float64}(ntuple(_ -> r(), 9))
        ∂γ = SArray{Tuple{3,3,3},Float64}(ntuple(_ -> r(), 27))
        Γ3 = SArray{Tuple{3,3,3},Float64}(ntuple(_ -> r(), 27))
        βl = SVector{3,Float64}(ntuple(_ -> r(), 3))
        dt4 = SMatrix{4,4,Float64}(ntuple(_ -> r(), 16))
        dβl = SMatrix{3,3,Float64}(ntuple(_ -> r(), 9))
        α = 1 + rand(rng)
        bump!("christoffel", relerr(christoffel(γu, ∂γ), christoffel(B(γu), B(∂γ))))
        bump!("contract", relerr(contract(Γ3, βl), contract(B(Γ3), B(βl))))
        bump!("kassemble", relerr(kassemble(dt4, dβl, Γ3, βl, α),
                                  kassemble(B(dt4), B(dβl), B(Γ3), B(βl), big(α))))
        bump!("kassemble_flat", relerr(kassemble_flat(dt4, dβl, Γ3, βl, α),
                                       kassemble_flat(B(dt4), B(dβl), B(Γ3), B(βl), big(α))))
    end
    println("REPRO2 target=", Sys.CPU_NAME, " coverage=", Base.JLOptions().code_coverage,
            " check_bounds=", Base.JLOptions().check_bounds, " ",
            join(("$k=$(worst[k])" for k in sort(collect(keys(worst)))), " "))
end
main()
