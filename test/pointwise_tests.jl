# What `src/pointwise.jl` claims, at one point, against the exact solution.
#
# This file covers the algebra that is a *function* of the state: the
# packing, the ADM extraction, the offset identities, the closed-form
# coefficient derivatives, the gauge constraint, and that all of it runs
# inside a kernel and allocates nothing.
# `pointwise_identity_tests.jl` covers the two identities that need
# derivatives of the algebra — GHSO2's `∂_tΠ − ∂_iF^i = msrc` and the
# expanded form against the flux form — and is split off because between
# them the two halves compile the metric library's nested dual passes for
# six backgrounds at two precisions, which is where this suite's time goes.
#
# The failure these tests exist to catch is the quiet one. The equations
# here were validated in GHSO2 on a different mesh; a port that transposed
# an index pair, dropped a symmetrisation or picked the wrong derivative
# convention still runs, still looks like general relativity, and is wrong
# in a way that only shows up as a convergence rate that is not `q` three
# milestones later. Every claim below is therefore against `SpacetimeMetrics`
# — a second implementation of the same geometry — and not against a stored
# number.

using KernelAbstractions: CPU, @Const, @index, @kernel, synchronize
using LinearAlgebra: det
using Random: MersenneTwister
using StaticArrays: SArray, SMatrix, SVector
using Test
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: NC, _dg4, _sym4, _η4

isdefined(@__MODULE__, :gh_backgrounds) || include("pointwise_backgrounds.jl")

# ---------------------------------------------------------------------------

@testset "The packed component order is the one every other file assumes" begin
    # Guards the convention `CLAUDE.md` names: `(tt, tx, ty, tz, xx, xy, xz,
    # yy, yz, zz)`, the column-major lower triangle. Nothing outside this
    # file indexes a component by a literal, so if `_sym4` and `_pack10`
    # ever disagreed with that sentence every stored field would be
    # transposed and every test that only round-trips would still pass.
    # Spelled with distinguishable entries rather than with a metric.
    M = SMatrix{4,4,Int}(11, 12, 13, 14,
                         12, 22, 23, 24,
                         13, 23, 33, 34,
                         14, 24, 34, 44)
    @test pack_sym(M) === SVector{10,Int}(11, 12, 13, 14, 22, 23, 24, 33, 34, 44)
    @test _sym4(pack_sym(M)) === M

    # `pack_g` is the same packing of `g − η`, and the offset is the one
    # `CODE.md` stores: `h = g − η` with `η = diag(−1, 1, 1, 1)`.
    T = Float64
    g = SMatrix{4,4,T}(M)
    @test pack_g(g) === pack_sym(g - _η4(T))
    @test _sym4(pack_g(g)) + _η4(T) === g

    # `prerequisite_tests.jl` fills the state through this packing; the two
    # spellings were separate only until `pointwise.jl` existed.
    @test collect(pack_sym(SMatrix{4,4,T}(M))) ==
          collect(SVector{10,T}(11, 12, 13, 14, 22, 23, 24, 33, 34, 44))
end

@testset "ADM extraction matches SpacetimeMetrics: T=$T" for T in (Float64, Float32)
    # Guards the 3+1 split, which the horizon finder and the time step both
    # read. `adm_from_metric` inverts the 4-metric and reads `α`, `β^i`,
    # `γ_ij` off it; `adm_decompose` builds the same three from the ADM
    # relations directly (`α = √(−g_tt + β^iβ_i)`, `β^i = γ^{ij}g_tj`).
    # They are different arithmetic for the same quantities, so agreement
    # is a real check of the sign of `β` and of the lapse's branch.
    # `metric_quantities` computes the same three through the offset
    # identities, and it is what the kernel uses.
    for (name, bg, _) in gh_backgrounds(T), x in gh_points(T, 2)
        p = SVector{4,T}(T(GH_TIME), x[1], x[2], x[3])
        g = metric(bg, p)
        α, β, γ = adm_from_metric(g)
        αr, βr, γr = SpacetimeMetrics.adm_decompose(g)
        @test α ≈ αr rtol = 64 * eps(T)
        @test β ≈ βr rtol = 64 * eps(T) atol = 64 * eps(T)
        @test γ ≈ γr rtol = 64 * eps(T)

        h = pack_g(g)
        _, _, αq, βq, γuq, sqrtγq = metric_quantities(_sym4(h))
        @test αq ≈ αr rtol = 64 * eps(T)
        @test βq ≈ βr rtol = 64 * eps(T) atol = 64 * eps(T)
        @test γuq ≈ inv(γr) rtol = 64 * eps(T)
        @test sqrtγq ≈ sqrt(det(γr)) rtol = 64 * eps(T)

        # The extrinsic curvature the horizon finder consumes, from the
        # evolved state rather than from the metric: `K` here comes through
        # `Π` and the evolution relation, and `ExtrinsicCurvature` comes
        # from `∂_t γ` directly. A `Π` that was really `∂_t g` — the
        # stand-in `prerequisite_tests.jl` uses, and the mistake that
        # `CODE.md` warns about — would fail this wherever the shift or the
        # densitisation is not trivial.
        hs, Π, ∂h = gh_state(bg, T(GH_TIME), x)
        γs, ∂γ, K = adm_vars_from_state(hs, Π, ∂h[1], ∂h[2], ∂h[3])
        Kr = SpacetimeMetrics.ExtrinsicCurvature(bg, p)
        @test γs ≈ γr rtol = 64 * eps(T)
        @test absdiff(K, Kr) <
              256 * eps(T) * max(maximum(abs, Kr), one(T))     # 0.2 eps
        # `∂γ[i, j, k] = ∂_k γ_ij`: the derivative axis is *last* here,
        # because this is what the finder wants, and that is the opposite
        # of the convention the source term uses two functions away.
        for k in 1:3
            @test SMatrix{3,3,T}(∂γ[i, j, k] for i in 1:3, j in 1:3) ≈
                  SMatrix{3,3,T}(_sym4(∂h[k])[i + 1, j + 1] for i in 1:3, j in 1:3)
        end
    end
end

# The offset scale at which GHSO2 measured the cancellation-free identities:
# `‖h‖ ~ 1e−13` at `Float64` (`notes/methods-ghso2.md`, "Floating-point
# hygiene"), and `1e−5` at `Float32`, a couple of decades above that type's
# `eps` in the same way. Below `eps` the offset is not representable at all
# and there is nothing to measure; far above it, nothing cancels and the
# naive route looks fine. A method per type, because the number *is* the
# type's, and a step further down belongs to whoever adds `Float32x2`.
offset_scale(::Type{Float64}) = 1.0e-13
offset_scale(::Type{Float32}) = 1.0f-5

@testset "The offset identities keep relative precision at ‖h‖ ≪ 1: T=$T" for T in
                                                                            (Float64,
                                                                             Float32)
    # Guards the reason the state stores `h = g − η` and not `g`. Near
    # Minkowski every derived quantity is a difference of numbers that agree
    # to almost every bit, and the naive route loses all of the offset's
    # precision: at `‖h‖ ~ 1e−13` in `Float64`, `inv(g) − η` is accurate to
    # `eps` *absolute*, which is three digits relative. The identity
    # `g^{ab} − η^{ab} = −g^{ac}h_{cd}η^{db}` keeps all of them. If this
    # ever regresses, the wave zone of every run — where `h` is small and
    # the physics is — silently loses its accuracy while the strong field
    # near the hole still looks fine.
    rng = MersenneTwister(20260916)
    scale = offset_scale(T)
    worst_offset = zero(T)
    worst_naive = zero(T)
    for _ in 1:8
        s = SMatrix{4,4,T}(2 * rand(rng, T) - 1 for _ in 1:4, _ in 1:4)
        h4 = scale * (s + s')
        g4 = _η4(T) + h4

        # The reference, in a precision where nothing cancels — and from
        # `η + h` computed *there*, not from the rounded `g4` above. The
        # sum `1 + h_ab` is itself inexact at `‖h‖ ~ eps`, so a reference
        # built by inverting the stored `g4` would differ from the offset
        # identity by exactly the rounding the identity exists to avoid,
        # and would report the identity as no better than the naive route.
        ηb = Matrix{BigFloat}(_η4(T))
        gb = ηb + Matrix{BigFloat}(h4)
        gub = inv(gb)
        guo_ref = gub - ηb
        α_ref = 1 / sqrt(-gub[1, 1])
        γb = gb[2:4, 2:4]
        sqrtγ_ref = sqrt(det(γb))
        β_ref = [-gub[1, j] / gub[1, 1] for j in 2:4]

        _, gu4, α, β, γu, sqrtγ, guo = metric_quantities(h4)

        # The offsets, to a relative eps.
        e = reldiff(Matrix{BigFloat}(guo), guo_ref)
        worst_offset = max(worst_offset, T(e))
        @test e < 64 * eps(T)
        @test reldiff(Vector{BigFloat}(β), β_ref) < 64 * eps(T)

        # The values built from them, likewise.
        @test abs(BigFloat(α) - α_ref) / α_ref < 64 * eps(T)
        @test abs(BigFloat(sqrtγ) - sqrtγ_ref) / sqrtγ_ref < 64 * eps(T)
        @test reldiff(Matrix{BigFloat}(γu), inv(γb)) < 64 * eps(T)

        # And the contrast that makes the identity worth its comment: the
        # direct difference of two O(1) inverses keeps no relative
        # precision at all at this scale.
        naive = Matrix{BigFloat}(inv(g4)) - ηb
        worst_naive = max(worst_naive, T(reldiff(naive, guo_ref)))

        # `guo` is the offset of the inverse, so `η + guo` must invert `g`.
        @test gu4 ≈ inv(g4) rtol = 64 * eps(T)
        @test guo ≈ guo' rtol = 64 * eps(T)
    end
    # The naive route's error is `eps / ‖h‖`; the assertion is loose because
    # the point is the order of magnitude, not the constant.
    @test worst_naive > 1024 * worst_offset
    @test worst_naive > sqrt(eps(T))
    @info "offset identities at ‖h‖ = $scale ($T): relative error " *
          "$(worst_offset) by the identity, $(worst_naive) naively"
end

@testset "metric_derivatives matches a forward-mode dual pass: T=$T" for T in
                                                                        (Float64,
                                                                         Float32)
    # Guards the closed-form chain rule that the expanded momentum equation
    # rests on. `∂_iβ^i` and `∂_i(α√γγ^{ij})` are what the product rule
    # bought in exchange for not differentiating a computed flux, so a sign
    # or a term dropped here is a wrong equation that is still a
    # well-posed-looking wave equation — it would converge, to the wrong
    # solution. The independent derivative is ForwardDiff straight through
    # `metric_quantities`, which knows nothing about the chain rule written
    # in `metric_derivatives`.
    for (name, bg, _) in gh_backgrounds(T), x in gh_points(T, 2)
        # The coefficients as a function of position, flattened: α, β^j,
        # and A^{jk} = α√γγ^{jk}.
        function coeffs(y::SVector{3,D}) where {D}
            hy = pack_g(metric(bg, SVector{4,D}(D(GH_TIME), y[1], y[2], y[3])))
            _, _, α, β, γu, sqrtγ = metric_quantities(_sym4(hy))
            A = (α * sqrtγ) * γu
            return SVector{13,D}(α, β[1], β[2], β[3],
                                 A[1, 1], A[2, 1], A[3, 1],
                                 A[1, 2], A[2, 2], A[3, 2],
                                 A[1, 3], A[2, 3], A[3, 3])
        end
        J = ForwardDiff.jacobian(coeffs, x)          # J[v, i] = ∂_i coeffs[v]

        h, _, ∂h = gh_state(bg, T(GH_TIME), x)
        dα, dβ, dA = metric_derivatives(h, ∂h)

        # There are two methods, and the streaming kernel of step 3 will
        # call the other one: the coefficient set is formed once per point,
        # so `metric_derivatives` takes it rather than rebuilding it with a
        # second `inv`, determinant and square root. The `(h, ∂h)` method
        # is the wrapper that does rebuild it, and it is what the tests and
        # the diagnostics call.
        #
        # **To roundoff, not `isequal`** — which is worth a sentence,
        # because the wrapper forwards into the same body with what ought
        # to be the same bits, and `isequal` *fails*, on 3 of 12 points at
        # `Float64` and 6 of 12 at `Float32` **(measured in step 1)**. The
        # two call sites inline that body into different surrounding code,
        # and the compiler fuses a multiply and an add in one and not the
        # other. So it is not that two *spellings* of an expression are not
        # bit-identical (`CLAUDE.md`): one spelling, at two call sites, is
        # already enough. The disagreement is at most `0.03 eps` of the
        # scale below at `Float64` and `0.19 eps` at `Float32`; a swapped
        # argument or a dropped term would be O(1) and is what this is for.
        _, gu4, αq, βq, γuq, sqrtγq = metric_quantities(_sym4(h))
        dα2, dβ2, dA2 = metric_derivatives(gu4, αq, βq, γuq, sqrtγq, ∂h)

        # The three coefficients are derivatives of quantities of order
        # one, so the size they are measured against is the size of the
        # gradients that produce them — never their own, which is zero for
        # `∂_iβ^j` on a background with no shift and would turn two
        # roundoff-sized numbers into a ratio of order one.
        scale = max(maximum(maximum(abs, ∂h[i]) for i in 1:3), one(T))
        tol = 256 * eps(T) * scale
        @test absdiff(dα, dα2) < 8 * eps(T) * scale
        @test absdiff(dβ, dβ2) < 8 * eps(T) * scale
        @test absdiff(dA, dA2) < 8 * eps(T) * scale
        @test absdiff(dα, SVector{3,T}(J[1, i] for i in 1:3)) < tol
        @test absdiff(dβ, SMatrix{3,3,T}(J[1 + j, i] for i in 1:3, j in 1:3)) <
              tol
        @test absdiff(dA,
                      SArray{Tuple{3,3,3},T}(J[4 + (k - 1) * 3 + j, i]
                                             for i in 1:3, j in 1:3, k in 1:3)) <
              tol
    end
end

@testset "C_a and Z_ab vanish on exact data: T=$T" for T in (Float64, Float32)
    # Guards the two things that must be zero on a solution, and the pair of
    # them is what says the gauge-source path and the damping term are wired
    # the way the formulation says.
    #
    # `C^μ = Γ^μ + H^μ` is the gauge constraint. A harmonic background has
    # `H ≡ 0` and `Γ^μ = 0` outright — Minkowski, the gauge wave, harmonic
    # Kerr and its boost, which is *why* the proof-of-concept case is a
    # boosted harmonic Kerr and not a boosted Kerr-Schild. A non-harmonic
    # one has `H^μ = −Γ^μ` sampled from the background, and the cancellation
    # is then between two independent computations of the contracted
    # Christoffel: this package's, from `dmetric`'s output by hand, and
    # `SpacetimeMetrics`', through `ChristoffelSymbols`.
    #
    # `Z_ab` is proportional to `C_a`, so switching the damping on must not
    # move the right-hand side of an exact solution by more than roundoff.
    # If it did, `γ0` would be a source rather than a damping term and every
    # run would drift away from the background it was started on.
    for (name, bg, harmonic) in gh_backgrounds(T), x in gh_points(T, 2)
        p = SVector{4,T}(T(GH_TIME), x[1], x[2], x[3])
        g, dg = dmetric(bg, p)                    # dg[a,b,c] = ∂_c g_ab
        Hup = harmonic ? zero(SVector{4,T}) : gauge_source(bg, p)
        C = gauge_constraint_at_node(g, dg, Hup)
        # The scale to measure against is the size of the terms that cancel,
        # not 1: `Γ^μ` is built from `g^{ab}∂_c g_ab`, all of which are O(1)
        # here.
        scale = max(maximum(abs, dg), one(T))
        @test maximum(abs, C) < 256 * eps(T) * scale           # ≤ 1.5 eps
        if harmonic
            # `H ≡ 0`, and `CODE.md` compiles the kernel without the
            # gauge-source terms on the strength of it — so the tolerance
            # here is a few `eps` and not a physics-sized number.
            Hl, dHl = gh_gauge(bg, T(GH_TIME), x)
            @test maximum(abs, Hl) < 64 * eps(T) * scale       # ≤ 2.4 eps
            @test maximum(abs, dHl) < 64 * eps(T) * scale
        end

        h, Π, ∂h = gh_state(bg, T(GH_TIME), x)
        Hl, dHl = gh_gauge(bg, T(GH_TIME), x)
        _, _, _, _, msrc0 = gh_node_rhs(h, Π, ∂h[1], ∂h[2], ∂h[3], Hl, dHl,
                                        zero(T), zero(T))
        _, _, _, _, msrcγ = gh_node_rhs(h, Π, ∂h[1], ∂h[2], ∂h[3], Hl, dHl,
                                        gh_γ0(T), gh_γ2(T))
        @test maximum(abs, msrcγ - msrc0) <
              256 * eps(T) * max(maximum(abs, msrc0), one(T))  # ≤ 0.3 eps
    end
end

@testset "The second derivatives are packed the way the RHS reads them" begin
    # Guards the one index convention `gh_node_rhs_expanded` owns that no
    # other function shares: `∂∂h` is an `NTuple{6}` in the column-major
    # lower-triangular order `(xx, xy, xz, yy, yz, zz)`, and the tests build
    # it by differentiating `∂h` a second time. `SpacetimeMetrics` will
    # build the same six from its own second-derivative pass, whose
    # derivative axes are the two trailing ones — a different route to the
    # same tensor, so a transposed or reordered pair shows up here and
    # nowhere else.
    #
    # One background and one precision: the claim is about an index order,
    # which is the same on every background, and the second nested dual
    # pass costs a compilation per background that buys nothing further.
    # Kerr-Schild is the row chosen because it depends on all three
    # coordinates and is the cheapest of those to compile — it is a closed
    # form rather than a pullback, and asking the proof-of-concept case for
    # the same claim cost twenty seconds more **(measured in step 1)**.
    T = Float64
    bg = gh_backgrounds(T)[4].bg               # Kerr-Schild
    for x in gh_points(T, 2)
        _, ∂∂h, _ = gh_derivatives(bg, T(GH_TIME), x)
        ref = gh_second_derivatives(bg, T(GH_TIME), x)
        scale = max(maximum(maximum(abs, r) for r in ref), one(T))
        for n in 1:6
            @test absdiff(∂∂h[n], ref[n]) < 256 * eps(T) * scale   # 0 eps
        end
    end
end

# ---------------------------------------------------------------------------
# Everything at one point, in one call, so that the kernel below is trivial
# and the host loop it is compared against runs exactly the same arithmetic.
# The claim is about the *launch*, not about the algebra, which the testsets
# above cover; hence one shared function and a bit-for-bit comparison.

# `h`, `Π`, `∂h`, `∂Π`, `∂∂h`, `H_b`, `∂_a H_b` in, and out the two halves
# of the expanded right-hand side, the flux form's five returns, the
# coefficient set, its three derivatives, the ADM triple, the gauge
# constraint, the ADM split of the metric and the source through its own
# entry point — every exported function of `pointwise.jl`, once.
const GH_NIN = 2 * NC + 3 * NC + 3 * NC + 6 * NC + 4 + 16              # 160
const GH_NOUT = 2 * NC + 5 * NC + 5 + 3 + 9 + 27 + 9 + 27 + 9 + 4 + 13 + NC  # 186

function gh_pointwise_all(h::SVector{NC,T}, Π::SVector{NC,T},
                          ∂h::NTuple{3,SVector{NC,T}},
                          ∂Π::NTuple{3,SVector{NC,T}},
                          ∂∂h::NTuple{6,SVector{NC,T}},
                          Hl::SVector{4,T}, dHl::SMatrix{4,4,T},
                          γ0::T, γ2::T) where {T}
    ∂ₜh, ∂ₜΠ = gh_node_rhs_expanded(h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl, γ0, γ2)
    dtg, Fx, Fy, Fz, msrc = gh_node_rhs(h, Π, ∂h[1], ∂h[2], ∂h[3], Hl, dHl,
                                        γ0, γ2)
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    dα, dβ, dA = metric_derivatives(h, ∂h)
    γ3, ∂γ, K = adm_vars_from_state(h, Π, ∂h[1], ∂h[2], ∂h[3])
    # `gauge_constraint_at_node` speaks SpacetimeMetrics' index order, so
    # the transpose happens here — at the call site, as it will on the mesh.
    dgl = _dg4(∂ₜh, ∂h)
    dgsm = SArray{Tuple{4,4,4},T}(dgl[c, a, b] for a in 1:4, b in 1:4, c in 1:4)
    C = gauge_constraint_at_node(g4, dgsm, gu4 * Hl)
    αa, βa, γa = adm_from_metric(g4)
    msrc2 = gh_node_source(g4, gu4, α, sqrtγ, dgl, Hl, dHl, γ0, γ2)
    return (∂ₜh..., ∂ₜΠ...,
            dtg..., Fx..., Fy..., Fz..., msrc...,
            α, sqrtγ, β...,
            dα..., dβ..., dA...,
            γ3..., ∂γ..., K...,
            C...,
            αa, βa..., γa...,
            msrc2...)
end

@inline function gh_unpack_inputs(read, ::Type{T}) where {T}
    h = SVector{NC,T}(ntuple(v -> read(v), Val(NC)))
    Π = SVector{NC,T}(ntuple(v -> read(NC + v), Val(NC)))
    ∂h = ntuple(i -> SVector{NC,T}(ntuple(v -> read(2NC + (i - 1) * NC + v),
                                          Val(NC))), Val(3))
    ∂Π = ntuple(i -> SVector{NC,T}(ntuple(v -> read(5NC + (i - 1) * NC + v),
                                          Val(NC))), Val(3))
    ∂∂h = ntuple(n -> SVector{NC,T}(ntuple(v -> read(8NC + (n - 1) * NC + v),
                                           Val(NC))), Val(6))
    Hl = SVector{4,T}(ntuple(v -> read(14NC + v), Val(4)))
    dHl = SMatrix{4,4,T}(ntuple(v -> read(14NC + 4 + v), Val(16)))
    return h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl
end

@kernel function gh_pointwise_kernel!(out, @Const(inp), γ0, γ2)
    p = @index(Global)
    T = eltype(out)
    h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl = gh_unpack_inputs(v -> inp[v, p], T)
    vals = gh_pointwise_all(h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl, γ0, γ2)
    for v in 1:length(vals)
        out[v, p] = vals[v]
    end
end

@testset "The pointwise algebra runs as a kernel on CPU(): T=$T" for T in
                                                                    (Float64,
                                                                     Float32)
    # Guards the requirement under which every one of these functions is
    # written: the right-hand side is one fused KernelAbstractions kernel
    # per owned point (`CODE.md`, "One right-hand-side evaluation"), so
    # nothing here may allocate, mutate, capture a `Type` or close over a
    # host array. A function that does all four still works when a test
    # calls it on the host and fails to compile — or silently runs on one
    # thread — when it is a kernel. Bit-for-bit rather than a tolerance,
    # because the host loop and the launch run the same arithmetic and
    # anything else is a difference that would put a floor under every
    # measured error.
    #
    # One background, not the table: what is under test is the *launch*,
    # and the algebra is measured against every row of the table by the
    # testsets above and by `pointwise_identity_tests.jl`. The one chosen is
    # the proof-of-concept case, which is the expensive one to compile and
    # the only one with a non-trivial value in every slot of the state.
    bgs = (gh_backgrounds(T)[end],)
    xs = gh_points(T, 2)
    npts = length(bgs) * length(xs)
    inp = zeros(T, GH_NIN, npts)
    p = 0
    for (name, bg, _) in bgs, x in xs
        p += 1
        h, Π, ∂h = gh_state(bg, T(GH_TIME), x)
        ∂Π, ∂∂h, _ = gh_derivatives(bg, T(GH_TIME), x)
        Hl, dHl = gh_gauge(bg, T(GH_TIME), x)
        vals = (h..., Π..., ∂h[1]..., ∂h[2]..., ∂h[3]...,
                ∂Π[1]..., ∂Π[2]..., ∂Π[3]...,
                ∂∂h[1]..., ∂∂h[2]..., ∂∂h[3]...,
                ∂∂h[4]..., ∂∂h[5]..., ∂∂h[6]...,
                Hl..., dHl...)
        for v in 1:GH_NIN
            inp[v, p] = vals[v]
        end
    end

    @test length(gh_pointwise_all(gh_unpack_inputs(v -> inp[v, 1], T)...,
                                  gh_γ0(T), gh_γ2(T))) == GH_NOUT

    out = fill(T(NaN), GH_NOUT, npts)
    gh_pointwise_kernel!(CPU(), 4)(out, inp, gh_γ0(T), gh_γ2(T); ndrange=npts)
    synchronize(CPU())

    ref = fill(T(NaN), GH_NOUT, npts)
    for q in 1:npts
        h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl = gh_unpack_inputs(v -> inp[v, q], T)
        vals = gh_pointwise_all(h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl, gh_γ0(T), gh_γ2(T))
        for v in 1:GH_NOUT
            ref[v, q] = vals[v]
        end
    end

    @test isequal(out, ref)
    @test all(isfinite, out)                  # no hole is inside the shell
    @test eltype(out) === T
end

@testset "The pointwise algebra allocates nothing: T=$T" for T in (Float64, Float32)
    # Guards the property that makes the functions above kernel-safe in the
    # first place. A single heap allocation inside the right-hand side is
    # roughly the whole evaluation on a GPU and is not even expressible
    # there; on the host it would show up only as a run that got slower.
    # `sum(generator)` inside a `StaticArrays` constructor is the classic
    # way to acquire one, and it is spelled that way throughout GHSO2's
    # algebra, so this is not hypothetical.
    bg = gh_backgrounds(T)[end].bg             # the proof-of-concept case
    x = gh_points(T, 1)[1]
    h, Π, ∂h = gh_state(bg, T(GH_TIME), x)
    ∂Π, ∂∂h, _ = gh_derivatives(bg, T(GH_TIME), x)
    Hl, dHl = gh_gauge(bg, T(GH_TIME), x)
    γ0, γ2 = gh_γ0(T), gh_γ2(T)

    call_expanded() = gh_node_rhs_expanded(h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl, γ0, γ2)
    call_flux() = gh_node_rhs(h, Π, ∂h[1], ∂h[2], ∂h[3], Hl, dHl, γ0, γ2)
    call_derivs() = metric_derivatives(h, ∂h)
    call_adm() = adm_vars_from_state(h, Π, ∂h[1], ∂h[2], ∂h[3])
    for f in (call_expanded, call_flux, call_derivs, call_adm)
        f()                                     # compile before measuring
        @test @allocated(f()) == 0
    end
end
