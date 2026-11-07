# What a wider ghost width claims (added 2026-11-05, TreeGRRMHD's GH-5): the
# state may be stored with a halo wider than the scheme's reach `q/2 + 1`, so
# that a coupled code whose own field set is wider reads the metric at every
# one of its stored points through `adm_vars`, and the run is the run at the
# scheme's own width.
#
# The failure these tests exist to catch is a place that reads the stored
# width `U.G` where it means the reach: the interior's margin and layer rules,
# the level floor, the shell band of the validity rows. A wider halo would
# then quietly change the physics, and nothing would fail. TreeAMR's fill,
# prolongation, restriction and regrid write the same values into the inner
# ghost layers whatever the stored width is (each window is fixed by the
# physical offset, and `G ≥ p/2 − 1`, `N ≥ 2G + 2` keep the clamp from
# firing), so the two runs agree bit for bit and the claim is `isequal`.

using KernelAbstractions: CPU
using StaticArrays: SVector
using Test
using TreeAMR
using TreeGeneralizedHarmonic
import SpacetimeMetrics as SMg

const TGHg = TreeGeneralizedHarmonic

isdefined(@__MODULE__, :TEST_TIER) || include("tiers.jl")  # `@full_tier`

@testset verbose = true "A ghost width wider than the scheme's reach" begin
    T = Float64
    q = 2
    reach = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)

    @testset "the reach is q/2 + 1, and a narrower halo is refused" begin
        # Guards the relaxation going too far: a halo narrower than the
        # dissipation's reach reads a ghost no fill wrote.
        @test stencil_reach(2) == 2 && stencil_reach(4) == 3 && stencil_reach(6) == 4
        case = minkowski_case(T; L=one(T), ε_KO=T(1 // 2), γ0=one(T),
                              γ2=T(-1 // 2))
        forest = gh_forest(T, case; N=8, roots=1)
        narrow = FieldSet{T}(forest, 20; G=1, centering=vertexcentered(3))
        @test_throws ArgumentError GHProblem(narrow, GhostSchedule(narrow, ops), case;
                                             q=q)
        wide = FieldSet{T}(forest, 20; G=3, centering=vertexcentered(3))
        p = GHProblem(wide, GhostSchedule(wide, ops), case; q=q)
        @test stencil_reach(p) == reach
        hole = hole_fixture(T; q=q)
        @test_throws "narrower than the reach" evolve!(
            T, hole; forest=hole_fixture_forest(T, hole; N=8), q=q, ops=ops,
            t_end=T(1 // 10), G=1)
    end

    # The run that reaches every piece that could read the stored width as
    # the reach: `refinement_tests.jl`'s mesh that moves (a regrid that
    # coarsens a block, so prolongation, restriction and the transfer), the
    # indicator and its level floor, the interior's radii checked on the new
    # mesh, and the record's validity rows with their shell band.
    @full_tier @testset "a run with a wider halo is the run at the scheme's width" begin
        function moved_mesh_run(G)
            case = adaptive_hole_fixture(T)
            forest = gh_forest(T, case; N=8, roots=4)
            targets = filter(forest.leaves) do k
                ext = block_extent(T, forest, k)
                inner = all(d -> -T(5 // 2) ≤ ext[d][1] && ext[d][2] ≤ T(5 // 2), 1:3)
                corner = all(d -> ext[d][1] ≥ T(5 // 2), 1:3)
                inner || corner
            end
            refine!(forest, targets)
            balance!(forest)
            return evolve!(T, case; forest=forest, q=q, ops=ops, t_end=T(1 // 10),
                           regrid=true, G=G)
        end
        a = moved_mesh_run(nothing)
        b = moved_mesh_run(3)
        @test a.U.G == (2, 2, 2) && b.U.G == (3, 3, 3)
        @test a.nregrids == b.nregrids == 1
        @test a.forest.leaves == b.forest.leaves
        @test length(a.records) == length(b.records)
        # Every row, key by key, so that a failure names the row that moved.
        for (ra, rb) in zip(a.records, b.records)
            @test keys(ra) == keys(rb)
            differ = [k for k in keys(ra) if !isequal(ra[k], rb[k])]
            @test isempty(differ)
            isempty(differ) || @info("record rows that moved with G", t=ra.t, differ)
        end
        @test isequal(a.interior, b.interior)
        # The state at every owned point.
        same = true
        for blk in 1:nblocks(a.U), v in 1:20
            same &= isequal(interiorview(a.U, blk, v), interiorview(b.U, blk, v))
        end
        @test same
    end

    # GH-5 itself: at `q = 2` and `G = 3`, a filled state gives the metric at
    # every stored vertex, the third ghost layer included, and `K` wherever
    # its stencil reaches, `q/2 < I ≤ n − q/2` — two ghost layers here. On a
    # uniform mesh a ghost is a copy of a neighbour's owned point or the
    # Dirichlet hook's exact value, so the metric is SpacetimeMetrics' to
    # roundoff and `K` at a ghost is bit for bit `K` at the owned point it
    # copies.
    @testset "adm_vars reads every stored vertex of a G = 3 state at q = 2" begin
        G = 3
        N = 8
        case = kerr_schild_case(T; a=T(1 // 2), halfwidth=T(4), r_0=T(5 // 8),
                                r_1=T(23 // 16), chunk=T(1 // 2))
        forest = gh_forest(T, case; N=N, roots=2)
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        fill_exact!(fs, case, zero(T))
        fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=TGHg.dirichlet(case, zero(T)))
        n = N + 2G + 1
        h = T(4) / N                                    # roots = 2 over [−4, 4]
        key(x) = Tuple(round.(Int, x ./ h))

        # The owned points' `K`, by position.
        owned = Dict{NTuple{3,Int},Any}()
        for b in 1:nblocks(fs), k in G+1:G+N+1, j in G+1:G+N+1, i in G+1:G+N+1
            I = (i, j, k)
            owned[key(coordinates(fs, b, I))] = adm_vars(fs.work, I, b, inv(h), Val(q)).K
        end

        worst, nmetric, nghostK, nK, same = zero(T), 0, 0, 0, true
        r = q ÷ 2
        for b in 1:nblocks(fs), k in 1:n, j in 1:n, i in 1:n
            I = (i, j, k)
            x = coordinates(fs, b, I)
            if sqrt(sum(abs2, x)) > 5 // 2
                m = adm_vars(fs.work, I, b)
                α, β, γ = SMg.adm_decompose(case.background, SVector{4,T}(0, x...))
                worst = max(worst, abs(m.α - α), maximum(abs, m.β - β),
                            maximum(abs, m.γ - γ))
                nmetric += any(d -> I[d] == 1 || I[d] == n, 1:3)
            end
            all(d -> r < I[d] ≤ n - r, 1:3) || continue
            K = adm_vars(fs.work, I, b, inv(h), Val(q)).K
            nK += 1
            ghost = any(d -> I[d] ≤ G || I[d] > G + N + 1, 1:3)
            Ko = get(owned, key(x), nothing)
            if ghost && Ko !== nothing
                same &= isequal(K, Ko)
                nghostK += any(d -> I[d] == 2 || I[d] == n - 1, 1:3)
            end
        end
        @test worst < 64eps(T)
        @test nmetric > 1000                            # the outermost layer, read
        @test nghostK > 1000                            # `K` in the second ghost layer
        @test same
        @test_throws BoundsError adm_vars(fs.work, (1, 5, 5), 1, inv(h), Val(q))
    end

    # A restart reads the stored width from the file and must be asked for
    # the same one, as for every other part of the layout.
    @testset "a restart with another ghost width is refused" begin
        case = hole_fixture(T; q=q)
        pre = joinpath(mktempdir(), "run")
        common = (q=q, ops=ops, checkpoint_path_prefix=pre, checkpoint_every_chunks=1)
        out = evolve!(T, case; forest=hole_fixture_forest(T, case; N=8), common...,
                      t_end=T(1 // 10), G=3)
        @test out.finished
        f = latest_checkpoint(pre)
        @test f !== nothing
        @test_throws "ghost widths (3, 3, 3)" evolve!(T, case; common...,
                                                      t_end=T(2 // 10), restart_file=f)
        out2 = evolve!(T, case; common..., t_end=T(2 // 10), restart_file=f, G=3)
        @test out2.finished && out2.U.G == (3, 3, 3)
    end
end
