# The lean source on the CPU (added 2026-10-05; `CODE.md`, "The right-hand side on
# an H200"): both spellings of `bench/rhs_lab_source.jl` against `gh_node_source` on
# random Lorentzian states, and what a call costs on one thread. It runs in the
# package's own environment:
#
#     julia --project=. bench/rhs_lab_cpu.jl
#
# The states are `h = 0.08 randn` (half of them scaled by `10⁻³`, where the offset
# identities matter), random `∂g`, `H_a` and `∂_a H_b`, `γ0 = 1.3`, `γ2 = 0.2`. The
# timing is the minimum over 50 sweeps of 4096 states, the coefficient set formed
# beforehand, as the kernel forms it before the source.
using TreeGeneralizedHarmonic, StaticArrays, LinearAlgebra, Random, Printf
const TGH = TreeGeneralizedHarmonic
const T = Float64
const NC = TGH.NC
include(joinpath(@__DIR__, "rhs_lab_source.jl"))

# The three as the kernel calls them: inlined into the loop. Without the `@inline` the
# call itself — the coefficient set passed in, the packed result returned — costs
# `lean_source_inl` almost as much again (167 ns against 90 ns per call, measured).
@inline lean(g4, gu4, α, sqrtγ, D, Hl, dHl, γ0, γ2) =
    lean_source(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ, D, Hl, dHl, γ0, γ2)
@inline lean_inl(g4, gu4, α, sqrtγ, D, Hl, dHl, γ0, γ2) =
    lean_source_inl(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ, D, Hl, dHl, γ0, γ2)
@inline pkg(g4, gu4, α, sqrtγ, D, Hl, dHl, γ0, γ2) =
    gh_node_source(g4, gu4, α, sqrtγ, TGH._dg4(D[1], (D[2], D[3], D[4])), Hl, dHl, γ0, γ2)

function check()
    Random.seed!(1)
    worst = Dict(:lean => 0.0, :lean_inl => 0.0)
    for trial in 1:200
        h = SVector{NC,T}(T(0.08) .* randn(T, NC)) .* (trial <= 100 ? one(T) : T(1e-3))
        g4, gu4, α, β, γu, sqrtγ = metric_quantities(TGH._sym4(h))
        D = ntuple(_ -> SVector{NC,T}(randn(T, NC)), 4)
        Hl = SVector{4,T}(randn(T, 4))
        dHl = SMatrix{4,4,T}(randn(T, 16))
        ref = pkg(g4, gu4, α, sqrtγ, D, Hl, dHl, T(1.3), T(0.2))
        for (name, f) in ((:lean, lean), (:lean_inl, lean_inl))
            new = f(g4, gu4, α, sqrtγ, D, Hl, dHl, T(1.3), T(0.2))
            worst[name] = max(worst[name], maximum(abs.(new .- ref)) / maximum(abs.(ref)))
        end
    end
    for name in (:lean, :lean_inl)
        @printf("%-16s worst relative difference from gh_node_source over 200 states: %.1e\n",
                name, worst[name])
    end
    return nothing
end

function sweep(f, pre, Ds, Hl, dHl)
    s = zero(SVector{NC,T})
    @inbounds for i in eachindex(pre)
        g4, gu4, α, β, γu, sqrtγ = pre[i]
        s += f(g4, gu4, α, sqrtγ, Ds[i], Hl, dHl, one(T), zero(T))
    end
    return s
end

function timing()
    Random.seed!(2)
    n = 4096
    pre = [metric_quantities(TGH._sym4(SVector{NC,T}(T(0.08) .* randn(T, NC)))) for _ in 1:n]
    Ds = [ntuple(_ -> SVector{NC,T}(randn(T, NC)), 4) for _ in 1:n]
    Hl = SVector{4,T}(randn(T, 4))
    dHl = SMatrix{4,4,T}(randn(T, 16))
    for (name, f) in (("gh_node_source", pkg), ("lean_source", lean),
                      ("lean_source_inl", lean_inl))
        sweep(f, pre, Ds, Hl, dHl)
        best = minimum(1:50) do _
            t0 = time_ns()
            sweep(f, pre, Ds, Hl, dHl)
            time_ns() - t0
        end
        @printf("%-16s %8.1f ns per call, %d bytes allocated a sweep\n", name, best / n,
                @allocated(sweep(f, pre, Ds, Hl, dHl)))
    end
    return nothing
end

@printf("# %s, Julia %s, %d thread(s)\n", gethostname(), VERSION, Threads.nthreads())
check()
timing()
