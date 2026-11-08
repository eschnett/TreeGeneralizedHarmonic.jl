# What `adm_vars` claims (added 2026-11-05): a downstream code reading this
# package's metric at a vertex — TreeGRRMHD's GH-1 — gets `α`, `β^i`, `γ_ij`,
# `√γ`, `γ^{ij}` and `K_ij` from the working array without knowing its layout,
# the packing or the stencils.
#
# The failures these tests exist to catch are a misread layout (a block, a
# variable or a ghost offset out by one gives the metric of a neighbouring
# point, which looks like a metric), a curvature that is a metric's but not
# this one's (a sign, a transposed derivative), and a function that works on
# the host but allocates or forms a closure, which a device refuses. Most
# claims are made on a synthetic working array of a boosted, spinning
# Kerr-Schild hole — two blocks of stored points, every one of them filled
# analytically, so ghosts are points like any other — and the layout once on a
# real field set filled by `fill_exact!`.

using KernelAbstractions: CPU, @Const, @index, @kernel, synchronize
using LinearAlgebra: det, inv
using StaticArrays: SVector, SMatrix
using Test
using TreeAMR
using TreeGeneralizedHarmonic
import SpacetimeMetrics as SMa

const TGHa = TreeGeneralizedHarmonic

# A hole spinning about `z` and moving across the spin, so that every
# component of `β` and `K` is nonzero away from the axes.
admvars_background(::Type{T}) where {T} =
    SMa.boost(SMa.KerrSchild(one(T), T(1 // 2)), SVector{3,T}(-1 // 5, 1 // 10, 0))

# A working array `(n, n, n, 20, 2)` of stored points at spacing `h`: block
# `b`'s point `I` is at `lo[b] + (I − 1) h`. Two blocks, so that a wrong block
# stride reads the other block's metric.
function admvars_work(bg, ::Type{T}, t, lo, h, n) where {T}
    work = zeros(T, n, n, n, 20, length(lo))
    for b in eachindex(lo), k in 1:n, j in 1:n, i in 1:n
        x = lo[b] + h * SVector{3,T}(i - 1, j - 1, k - 1)
        work[i, j, k, :, b] .= state_tuple(bg, t, x)
    end
    return work
end

admvars_point(lo, h, b, I) = lo[b] + h * SVector(I[1] - 1, I[2] - 1, I[3] - 1)

# The RHS's `∂_i h` at a stored point, formed by the test with the kernel's own
# stencil and the same spelling.
function admvars_rhs_gradient(work::AbstractArray{T}, I, b, inv_h, ::Val{q}) where {T,q}
    st, sv, sb = TGHa.work_strides(work)
    var = 1 + (b - 1) * sb + (I[1] - 1) * st[1] + (I[2] - 1) * st[2] + (I[3] - 1) * st[3]
    w1 = derivative_weights(T, Val(q), Val(1))
    return ntuple(d -> inv_h * SVector{10,T}(ntuple(v ->
        TGHa.axis_stencil(w1, work, var + (v - 1) * sv, st[d]), 10)), 3)
end

@kernel function admvars_test_kernel!(out, @Const(work), inv_h, r, qv)
    i, j, k = @index(Global, NTuple)
    I = (i + r, j + r, k + r)
    m = adm_vars(work, I, 1)
    a = adm_vars(work, I, 1, inv_h, qv)
    out[i, j, k, 1] = m.α
    out[i, j, k, 2] = m.sqrtγ
    out[i, j, k, 3] = m.γu[1, 2]
    out[i, j, k, 4] = a.K[1, 2]
    out[i, j, k, 5] = a.K[3, 3]
end

# What a consumer does with the contract's adapter: one `map_blocks!` over the
# owned vertices of a 16-variable set of its own, at its own `G`.
@kernel function admmetric_test_kernel!(mw, at, ::Val{GM}) where {GM}
    i, j, k, b = @index(Global, NTuple)
    a = at((i, j, k), b)
    I = (i + GM, j + GM, k + GM)
    mw[I..., 1, b] = a.α
    mw[I..., 2, b] = a.β[1]
    mw[I..., 3, b] = a.β[2]
    mw[I..., 4, b] = a.β[3]
    mw[I..., 5, b] = a.γ[1, 1]
    mw[I..., 6, b] = a.γ[1, 2]
    mw[I..., 7, b] = a.γ[1, 3]
    mw[I..., 8, b] = a.γ[2, 2]
    mw[I..., 9, b] = a.γ[2, 3]
    mw[I..., 10, b] = a.γ[3, 3]
    mw[I..., 11, b] = a.K[1, 1]
    mw[I..., 12, b] = a.K[1, 2]
    mw[I..., 13, b] = a.K[1, 3]
    mw[I..., 14, b] = a.K[2, 2]
    mw[I..., 15, b] = a.K[2, 3]
    mw[I..., 16, b] = a.K[3, 3]
end

# An adaptor that turns every array into a view, so that a test on the CPU can
# see `Adapt` reach the arrays a device would convert.
struct AdmToView end
TreeGeneralizedHarmonic.Adapt.adapt_storage(::AdmToView, a::Array) =
    view(a, ntuple(_ -> Colon(), ndims(a))...)

@testset verbose = true "adm_vars: the vertex metric in ADM variables" begin
    T = Float64
    bg = admvars_background(T)
    t = T(1 // 2)
    # Both blocks at least `2.4 M` from the hole, which is near the origin.
    lo = (SVector{3,T}(2, 1 // 2, 1), SVector{3,T}(-4, -1, 3 // 2))

    @testset "the metric is SpacetimeMetrics' ADM split at every stored vertex" begin
        # Guards the layout: the block stride, the variable stride, `h` in
        # `1:10`, and the stored (not owned) index — a ghost is a stored point,
        # and the first method claims every one of them. Also guards `√γ` and
        # `γ^{ij}`, which the caller densitizes and raises with.
        n, h = 7, T(1 // 8)
        work = admvars_work(bg, T, t, lo, h, n)
        worst = zero(T)
        for b in 1:2, k in 1:n, j in 1:n, i in 1:n
            I = (i, j, k)
            m = adm_vars(work, I, b)
            α, β, γ = SMa.adm_decompose(bg, SVector{4,T}(t, admvars_point(lo, h, b, I)...))
            worst = max(worst, abs(m.α - α), maximum(abs, m.β - β), maximum(abs, m.γ - γ),
                        abs(m.sqrtγ - sqrt(det(γ))), maximum(abs, m.γu - inv(γ)))
        end
        @test worst < 64eps(T)
        @test isbits(adm_vars(work, (1, 1, 1), 2))
        @test keys(adm_vars(work, (1, 1, 1), 2)) == (:α, :β, :γ, :sqrtγ, :γu)
    end

    @testset "K converges to ExtrinsicCurvature at order q" begin
        # Guards the curvature's sign, its index order and the `1/h`: each is
        # still a symmetric matrix of the right size, and only the exact one
        # converges to SpacetimeMetrics' at the stencil's order.
        x = SVector{3,T}(5 // 2, 3 // 2, 1)
        Kx = SMa.ExtrinsicCurvature(bg, SVector{4,T}(t, x...))
        for q in (2, 4)
            r = q ÷ 2
            n = 2r + 1
            errs = map((1 // 8, 1 // 16, 1 // 32)) do hr
                h = T(hr)
                c = r + 1                                 # `x` is the center point
                work = admvars_work(bg, T, t, (x .- h * r,), h, n)
                a = adm_vars(work, (c, c, c), 1, inv(h), Val(q))
                maximum(abs, a.K - Kx)
            end
            rates = log2.(errs[1:2] ./ errs[2:3])
            @test all(q - 0.3 .< rates .< q + 0.5)
            @test errs[3] < (q == 2 ? 1e-3 : 1e-6)
        end
        @test maximum(abs, Kx) > 1 // 50                  # a test of something
    end

    @testset "K is adm_vars_from_state's of the RHS's gradient" begin
        # Guards the claim that `K` is the horizon finder's extraction of the
        # same `∂h` the right-hand side forms: same stencil, same `1/h`, same
        # algebra (`_extrinsic_curvature`, shared). `metric_quantities` is
        # inlined into two call sites, so the two agree to the roundoff of the
        # terms, not to the bit (`CLAUDE.md`, "Two spellings…").
        q, h = 4, T(1 // 16)
        n = 9
        work = admvars_work(bg, T, t, lo, h, n)
        worst, scale, dmetric, γsame = zero(T), zero(T), zero(T), true
        for b in 1:2, k in 3:7, j in 3:7, i in 3:7
            I = (i, j, k)
            ∂h = admvars_rhs_gradient(work, I, b, inv(h), Val(q))
            hv = SVector{10,T}(work[i, j, k, 1:10, b])
            Πv = SVector{10,T}(work[i, j, k, 11:20, b])
            Kref = adm_vars_from_state(hv, Πv, ∂h...)[3]
            a = adm_vars(work, I, b, inv(h), Val(q))
            m = adm_vars(work, I, b)
            worst = max(worst, maximum(abs, a.K - Kref))
            scale = max(scale, maximum(abs, Kref), maximum(maximum.(abs, ∂h)),
                        maximum(abs, Πv))
            # The second method's metric is the first method's.
            dmetric = max(dmetric, abs(a.α - m.α), maximum(abs, a.β - m.β),
                          maximum(abs, a.γu - m.γu), abs(a.sqrtγ - m.sqrtγ))
            γsame &= a.γ == m.γ
        end
        @test worst <= 32eps(T) * scale
        @test dmetric <= 8eps(T)
        @test γsame
        @test isbits(adm_vars(work, (5, 5, 5), 1, inv(h), Val(q)))
    end

    @testset "the stencil's reach is refused at the edge of the stored range" begin
        # Guards the silent read: `work` is linear, so a stencil past the
        # lower `i` edge reads the previous row's points, which are finite.
        q, h, n = 4, T(1 // 16), 9
        work = admvars_work(bg, T, t, lo, h, n)
        r = q ÷ 2
        @test adm_vars(work, (r + 1, r + 1, n - r), 2, inv(h), Val(q)).α > 0
        @test_throws BoundsError adm_vars(work, (r, 5, 5), 1, inv(h), Val(q))
        @test_throws BoundsError adm_vars(work, (5, n - r + 1, 5), 1, inv(h), Val(q))
        @test_throws BoundsError adm_vars(work, (5, 5, 5), 3, inv(h), Val(q))
        @test_throws BoundsError adm_vars(work, (0, 5, 5), 1)
        @test_throws BoundsError adm_vars(work[:, :, :, 1:10, :], (5, 5, 5), 1, inv(h),
                                          Val(q))
        # A spacing of another precision than the array's is refused, not
        # promoted into a mixed-precision `K`.
        @test_throws MethodError adm_vars(Float32.(work), (5, 5, 5), 1, inv(h), Val(q))
    end

    @testset "both methods run in a kernel and allocate nothing" begin
        # Guards the device discipline (`CLAUDE.md`, "On a device a closure is
        # a call"): a closure or a heap value shows here as an allocation, and
        # a kernel's result as a disagreement with the host's.
        q, h, n = 4, T(1 // 16), 9
        r = q ÷ 2
        work = admvars_work(bg, T, t, lo, h, n)
        # Everything a call takes is an argument: a captured `q` would make
        # `Val(q)` a dynamic dispatch, and its allocation the test's.
        f1(w, I) = adm_vars(w, I, 2)
        f2(w, I, ih, qv) = adm_vars(w, I, 2, ih, qv)
        I0, ih, qv = (5, 5, 5), inv(h), Val(q)
        f1(work, I0); f2(work, I0, ih, qv)
        @test @allocated(f1(work, I0)) == 0
        @test @allocated(f2(work, I0, ih, qv)) == 0
        m = n - 2r
        out = zeros(T, m, m, m, 5)
        admvars_test_kernel!(CPU())(out, work, inv(h), r, Val(q); ndrange=(m, m, m))
        synchronize(CPU())
        ok = true
        for k in 1:m, j in 1:m, i in 1:m
            I = (i + r, j + r, k + r)
            a = adm_vars(work, I, 1, inv(h), Val(q))
            ok &= out[i, j, k, 1] ≈ a.α && out[i, j, k, 2] ≈ a.sqrtγ &&
                  out[i, j, k, 3] ≈ a.γu[1, 2] && out[i, j, k, 4] ≈ a.K[1, 2] &&
                  out[i, j, k, 5] ≈ a.K[3, 3]
        end
        @test ok
    end

    @testset "a field set filled by fill_exact! is read in its own layout" begin
        # Guards the contract with TreeAMR's working array: `fill_exact!`
        # writes owned points at stored index `owned + G`, and `adm_vars` at
        # that stored index is the metric at `coordinates(fs, b, I)`.
        q = 4
        G = q ÷ 2 + 1
        case = kerr_schild_case(T; a=T(1 // 2), halfwidth=T(4), r_0=T(5 // 8),
                                r_1=T(23 // 16), chunk=T(1 // 2))
        forest = gh_forest(T, case; N=8, roots=2)
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        fill_exact!(fs, case, zero(T))
        worst, npts = zero(T), 0
        for b in 1:nblocks(fs), k in 1:8, j in 1:8, i in 1:8
            I = (i + G, j + G, k + G)
            x = coordinates(fs, b, I)
            sqrt(sum(abs2, x)) > 5 // 2 || continue
            m = adm_vars(fs.work, I, b)
            α, β, γ = SMa.adm_decompose(case.background, SVector{4,T}(0, x...))
            worst = max(worst, abs(m.α - α), maximum(abs, m.β - β), maximum(abs, m.γ - γ))
            npts += 1
        end
        @test npts > 1000
        @test worst < 64eps(T)
    end

    @testset "ADMMetric is adm_vars at the owned vertices of a two-level mesh" begin
        # Guards the contract's adapter (`CODE.md`, "The coupling contract
        # with TreeGRRMHD"): the owned index plus this package's `G` (an offset
        # by one reads the neighbouring vertex, which looks like a metric), each
        # block's own `1/h` on a mesh with two levels (a root spacing on the
        # fine block halves its `∂h`), the scatter and fill the adapter does
        # before it reads, and a consumer's kernel over a set of its own `G`.
        # `N = 10`: TreeAMR's vertex invariant `N ≥ 2G + 2` refuses the
        # consumer's `G = 4` at `N = 8`.
        q, N = 4, 10
        G = q ÷ 2 + 1
        case = kerr_schild_case(T; a=T(1 // 2), halfwidth=T(4), r_0=T(5 // 8),
                                r_1=T(23 // 16), chunk=T(1 // 2))
        forest = gh_forest(T, case; N=N, roots=2, refined=true)
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        # The radius checks would refuse this coarse mesh's layer, which no
        # claim here reads.
        p = GHProblem(fs, GhostSchedule(fs, ops), case; q=q, margin_check=false)
        ref = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        fill_exact!(ref, case, zero(T))
        u = statevector(ref)
        gather!(u, ref)
        fs.work .= T(NaN)                       # the adapter must fill what it reads
        M = FieldSet{T}(forest, 16; G=4, centering=vertexcentered(3))
        at = ADMMetric(p, u, zero(T))(M)
        hs = block_spacings(forest, T)
        @test length(unique(hs)) == 2           # a test of the per-block `1/h`
        same, worst, worstK, scaleK, npts = true, zero(T), zero(T), zero(T), 0
        for b in 1:nblocks(fs), k in 1:N, j in 1:N, i in 1:N
            n = (i, j, k)
            I = (i + G, j + G, k + G)
            a = at(n, b)
            same &= isequal(a, adm_vars(fs.work, I, b, inv(hs[b]), Val(q)))
            x = coordinates(fs, b, I)
            sqrt(sum(abs2, x)) > 5 // 2 || continue
            pt = SVector{4,T}(0, x...)
            α, β, γ = SMa.adm_decompose(case.background, pt)
            Kx = SMa.ExtrinsicCurvature(case.background, pt)
            worst = max(worst, abs(a.α - α), maximum(abs, a.β - β), maximum(abs, a.γ - γ))
            worstK = max(worstK, maximum(abs, a.K - Kx))
            scaleK = max(scaleK, maximum(abs, Kx))
            npts += 1
        end
        @test same
        @test npts > 1000
        @test worst < 64eps(T)
        @test worstK < scaleK / 10
        @test keys(at((1, 1, 1), 1)) == (:α, :β, :γ, :sqrtγ, :γu, :K)

        # In the consumer's kernel the metric is the host's bit for bit; `K`
        # is formed at another call site, which contracts differently, and
        # agrees to roundoff of its scale (measured: 5.6 eps; `CLAUDE.md`,
        # "Two spellings…").
        mw = fill!(similar(M.work), T(NaN))
        map_blocks!(admmetric_test_kernel!, M, mw, at, Val(4))
        msame, dK, sK = true, zero(T), zero(T)
        for b in 1:nblocks(M), k in 1:N, j in 1:N, i in 1:N
            a = at((i, j, k), b)
            v = mw[i + 4, j + 4, k + 4, :, b]
            msame &= v[1:10] == [a.α; a.β; a.γ[1, 1]; a.γ[1, 2]; a.γ[1, 3];
                                 a.γ[2, 2]; a.γ[2, 3]; a.γ[3, 3]]
            dK = max(dK, maximum(abs, v[11:16] - [a.K[1, 1], a.K[1, 2], a.K[1, 3],
                                                  a.K[2, 2], a.K[2, 3], a.K[3, 3]]))
            sK = max(sK, maximum(abs, a.K))
        end
        @test msame
        @test dK <= 32eps(T) * sK
    end

    @testset "ADMMetric refuses another forest, another centering, a ghost" begin
        # Guards the binding, which is the only check that an owned index
        # names the point the consumer thinks, and the owned range, which is
        # all the contract promises; and the device discipline: no
        # allocation, and `Adapt` reaching both arrays.
        q, N = 4, 8
        case = kerr_schild_case(T; a=T(1 // 2), halfwidth=T(4), r_0=T(5 // 8),
                                r_1=T(23 // 16), chunk=T(1 // 2))
        forest = gh_forest(T, case; N=N, roots=2)
        fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
        p = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                      restriction=q + 2)),
                      case; q=q, margin_check=false)
        fill_exact!(fs, case, zero(T))
        adm = ADMMetric(p)
        other = gh_forest(T, case; N=N, roots=2)
        @test_throws ArgumentError adm(FieldSet{T}(other, 16; G=4,
                                                   centering=vertexcentered(3)))
        @test_throws ArgumentError adm(FieldSet{T}(forest, 16; G=3,
                                                   centering=cellcentered(3)))
        at = adm(FieldSet{T}(forest, 16; G=1, centering=vertexcentered(3)))
        @test at((1, 1, N), 1).α > 0
        @test_throws BoundsError at((0, 1, 1), 1)
        @test_throws BoundsError at((1, N + 1, 1), 1)
        @test_throws BoundsError at((1, 1, 1), nblocks(fs) + 1)
        f(a, n, b) = a(n, b)
        f(at, (2, 3, 4), 5)
        @test @allocated(f(at, (2, 3, 4), 5)) == 0
        av = TreeGeneralizedHarmonic.Adapt.adapt(AdmToView(), at)
        @test av.work isa SubArray && av.spacings isa SubArray
        @test isequal(av((2, 3, 4), 5), at((2, 3, 4), 5))
    end
end
