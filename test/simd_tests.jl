# The right-hand-side kernel on SIMD lanes (added 2026-10-05).
#
# `CODE.md`, "The right-hand side on a CPU": on the CPU the kernel evaluates `W`
# neighbouring points along the first axis at once, as SIMD.jl's `Vec{W,T}`, through
# the package's own algebra. What has to hold is that this is the scalar kernel —
# every point written, the same numbers — on every branch the kernel has: no hole, a
# hole whose layer some groups straddle, the sampled and the algebraic gauge sources,
# a row whose length `W` does not divide (the overlapping last group), `Float32`.
#
# "The same numbers" is roundoff, not bits. Each lane does its point's scalar
# operations in the scalar order, but StaticArrays forms a matrix product with
# `muladd`, and LLVM fuses each into an FMA or not by the code around it, which the
# lanes change: on the gauge wave the two kernels agree bit for bit (its shift is
# zero), on the holes by 1.0–2.0 eps of the terms that cancel into `du`. So the
# comparison is the kernel test's own: against `rhs_scale`, the size of those terms
# (`evolution_tests.jl`), at points the kernel evolves.

using KernelAbstractions: CPU, Backend
using Random: Xoshiro
using SIMD: Vec
using StaticArrays: SVector
using TreeGeneralizedHarmonic: check_simd_width, chunk_interior, interior_point,
                               is_lane_leader, is_outside, point_position,
                               simd_vector_bytes

# A backend that is not the CPU, for the width rule; nothing is launched on it.
struct NotTheCPU <: Backend end

@testset "The lane width follows the backend, the type and the block" begin
    # Guards the rule that decides who gets lanes. A device given lanes would compile
    # SIMD.jl's vectors into a GPU kernel; a software type given lanes would fail to
    # build a `Vec`; a width above the row would reach past it.
    w64 = simd_vector_bytes() ÷ 8
    @test w64 ∈ (4, 8)
    @test default_simd_width(Float64, CPU(), 16) == w64
    @test default_simd_width(Float32, CPU(), 16) == 2w64
    @test default_simd_width(Float32, CPU(), 6) == 4          # halved to fit the row
    @test default_simd_width(Float64, CPU(), 2) == 2
    @test default_simd_width(Float64, NotTheCPU(), 16) == 1
    @test default_simd_width(BigFloat, CPU(), 16) == 1
    @test check_simd_width(1, BigFloat, NotTheCPU(), 4) == 1
    @test check_simd_width(4, Float64, CPU(), 4) == 4
    @test_throws "power of two" check_simd_width(3, Float64, CPU(), 16)
    @test_throws "only the CPU kernel" check_simd_width(4, Float64, NotTheCPU(), 16)
    @test_throws "hardware floating-point" check_simd_width(4, BigFloat, CPU(), 16)
    @test_throws "wider than a block's row" check_simd_width(16, Float64, CPU(), 8)
end

@testset "The groups of lanes cover every row exactly once or overlap at its end" begin
    # Guards the overlapping tail. A point no group covers keeps whatever `du` held;
    # a group that starts past `N − W + 1` reads and writes the next row.
    for N in (2, 4, 6, 8, 10, 12, 16, 17), W in (1, 2, 4, 8)
        W ≤ N || continue
        starts = filter(i -> is_lane_leader(i, N, W), 1:N)
        covered = sort(unique(reduce(vcat, [collect(s:s + W - 1) for s in starts])))
        @test covered == 1:N
        @test all(s -> s + W - 1 ≤ N, starts)
        @test length(starts) == cld(N, W)
    end
end

# `du` from the scalar kernel and from the host's lanes on the same filled state, with
# the size of the terms at a sample of points in every block.
function lanes_against_scalar(::Type{T}, case, forest; q) where {T}
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                     parity=state_parity(forest))
    fill_exact!(fs, case, zero(T); interior=case.interior)
    u = statevector(fs)
    gather!(u, fs)
    probs = map((1, nothing)) do W
        p = GHProblem(fs, GhostSchedule(fs, ops), case; q=q, simd_width=W)
        case.interior === nothing && return p
        dt = gh_dt(p, u; cfl=T(1 // 4))
        return with_interior(p, chunk_interior(case, dt, nothing,
                                               default_relaxation_rate(case);
                                               default=true, interior=case.interior))
    end
    dus = map(probs) do p
        du = fill!(similar(u), T(NaN))
        gh_rhs!(du, u, p, zero(T))
        du
    end
    # The scale at points the kernel evolves: inside a hole's layer the stale core
    # data has steep differences that would make any bound loose.
    N = forest.N
    int = probs[1].interior
    evolved(b, o) = int === nothing ||
        is_outside(int, interior_point(int, zero(T),
                                       point_position(probs[1].origins, probs[1].spacings,
                                                      b, o)))
    S = maximum(rhs_scale(fs, b, ntuple(d -> o[d] + fs.G[d], Val(3)), q)
                for b in 1:nblocks(fs),
                    o in ((1, 1, 1), (N ÷ 2, N ÷ 2, N ÷ 2), (N, 2, N - 1)) if evolved(b, o))
    return (; W=probs[2].valsimd, scalar=dus[1], lanes=dus[2], S)
end

@testset "The lanes are the scalar kernel, every point written: $label" for
    (label, T, q, make) in (
        ("gauge wave, N = 8", Float64, 4,
         T -> (c = gauge_wave_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=T(-1 // 2));
               (c, gh_forest(T, c; N=8, roots=2)))),
        ("gauge wave, N = 10 (an overlapping group)", Float64, 4,
         T -> (c = gauge_wave_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=T(-1 // 2));
               (c, gh_forest(T, c; N=10, roots=2)))),
        ("gauge wave, Float32, N = 12", Float32, 4,
         T -> (c = gauge_wave_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=T(-1 // 2));
               (c, gh_forest(T, c; N=12, roots=2)))),
        ("the hole fixture: sampled source, layer, core, N = 10", Float64, 2,
         T -> (c = hole_fixture(T; q=2); (c, hole_fixture_forest(T, c; N=10)))),
        ("the octant hole, algebraic source", Float64, 2,
         T -> (c = kerr_schild_case(T; halfwidth=T(5 // 2), r_0=T(2 // 5),
                                    r_1=T(23 // 20), chunk=T(1 // 10), octant=true,
                                    gauge_source=:algebraic);
               (c, gh_forest(T, c; N=12, roots=2)))))
    # Guards the lane kernel on every branch it takes: a lane read at the wrong
    # offset, a group that skips or overruns its row's end, a hole's group sent down
    # the lanes with a point inside the layer, a gauge source read for the wrong
    # point. Each of those is an error of the size of a term, not of its roundoff.
    case, forest = make(T)
    r = lanes_against_scalar(T, case, forest; q=q)
    @test r.W === Val(default_simd_width(T, CPU(), forest.N))
    @test r.W !== Val(1)
    @test !any(isnan, r.scalar)
    @test !any(isnan, r.lanes)
    worst = maximum(abs.(r.lanes .- r.scalar)) / r.S
    @info "lanes against the scalar kernel" label W = r.W bitwise = isequal(r.lanes, r.scalar) worst_in_eps = worst / eps(T)
    @test worst ≤ 64 * eps(T)
end

@testset "The algebraic source on lanes is the scalar source, lane by lane" begin
    # Guards `_select` in the closed form: a lane whose root is clamped beside a lane
    # whose root is not. A branch taken for all lanes at once would clamp the wrong
    # ones; the root of a negative argument would be a `NaN` where a zero belongs.
    T = Float64
    src = KerrSchildSource(T; M=1, spin=(0, 0, 9 // 10), velocity=(1 // 5, 0, 0))
    rng = Xoshiro(3)
    rad(h) = (w = TreeGeneralizedHarmonic._sym4(h) * src.u;
              src.M^2 - sum(src.S .* w)^2)
    hs = SVector{10,T}[]
    while length(hs) < 4
        h = SVector{10,T}((length(hs) < 2 ? 1 // 10 : 3) .* randn(rng, T, 10))
        (rad(h) > 0) == (length(hs) < 2) && push!(hs, h)
    end
    ds = [ntuple(_ -> SVector{10,T}(randn(rng, T, 10)), 4) for _ in 1:4]
    lane(xs, c) = Vec(ntuple(l -> xs[l][c], 4))
    hv = SVector{10}(ntuple(c -> lane(hs, c), 10))
    dv = ntuple(a -> SVector{10}(ntuple(c -> lane([d[a] for d in ds], c), 10)), 4)
    Hv, dHv = algebraic_gauge_source(src, hv, dv[1], (dv[2], dv[3], dv[4]))
    for l in 1:4
        Hs, dHs = algebraic_gauge_source(src, hs[l], ds[l][1], (ds[l][2], ds[l][3], ds[l][4]))
        scale = 1 + maximum(abs, Hs) + maximum(abs, dHs)
        @test maximum(abs.(map(x -> x[l], Hv) .- Hs)) ≤ 16 * eps(T) * scale
        @test maximum(abs.(map(x -> x[l], dHv) .- dHs)) ≤ 16 * eps(T) * scale
    end
end
