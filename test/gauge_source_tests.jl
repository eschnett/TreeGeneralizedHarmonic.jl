# The algebraic Kerr-Schild gauge source (added 2026-10-02): `CODE.md`, "Open
# questions", "Gauge sources that know less about the hole". A closed form in
# `g` and three constants of the run — `M`, the spin 4-vector, the 4-velocity
# — that makes every boosted, spinning Kerr-Schild hole exact; evaluated in
# the kernels instead of the sampled `Hsrc` field set.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using StaticArrays: SVector
import SpacetimeMetrics as SM

@testset verbose = true "The algebraic Kerr-Schild gauge source" begin
    T = Float64

    @testset "the closed form is the background's own −Γ_a, and its gradient" begin
        # Guards the formula, its sign, the boost of the spin 4-vector, the
        # `boost(m, v)`-moves-at-`−v` convention and the layout `∂_c H_a` of
        # the gradient — each of which a sampled-source run never sees — against
        # SpacetimeMetrics' `gauge_source_grad`, at twelve points per hole
        # between 2.5 M and 6 M from it.
        for (bg, src) in [
            (SM.KerrSchild(1.0, 0.0), KerrSchildSource(T; M=1)),
            (SM.KerrSchild(2.0, 0.0), KerrSchildSource(T; M=2)),
            (SM.KerrSchild(1.0, 0.9), KerrSchildSource(T; M=1, spin=(0, 0, 0.9))),
            (SM.translate(SM.KerrSchild(1.0, 0.0), SVector(0.0, 1.0, -2.0, 0.5)),
             KerrSchildSource(T; M=1)),
            (SM.boost(SM.KerrSchild(1.0, 0.5), SVector(0.3, 0.0, 0.0)),
             KerrSchildSource(T; M=1, spin=(0, 0, 0.5), velocity=(-0.3, 0, 0))),
            (SM.boost(SM.KerrSchild(1.0, 0.7), SVector(0.2, 0.3, -0.1)),
             KerrSchildSource(T; M=1, spin=(0, 0, 0.7), velocity=(-0.2, -0.3, 0.1)))]
            @test check_gauge_source(bg, src, T; center=(0.0, 0.0, 0.0),
                                     scale=src.M) ≤ 1e-13   # measured 2–4e−15
        end
    end

    @testset "a source with the wrong constants is refused" begin
        # Guards the check a case runs at construction: the wrong mass, the
        # boost's velocity with the wrong sign, a spin the hole does not have,
        # and a background that is not Kerr-Schild.
        ks = SM.KerrSchild(1.0, 0.0)
        msg = "does not make the background exact"
        @test_throws msg check_gauge_source(ks, KerrSchildSource(T; M=1.1), T;
                                            center=(0.0, 0.0, 0.0), scale=1.0)
        @test_throws msg check_gauge_source(
            SM.boost(ks, SVector(0.3, 0.0, 0.0)),
            KerrSchildSource(T; M=1, velocity=(0.3, 0, 0)), T;
            center=(0.0, 0.0, 0.0), scale=1.0)
        @test_throws msg check_gauge_source(
            ks, KerrSchildSource(T; M=1, spin=(0, 0, 0.5)), T;
            center=(0.0, 0.0, 0.0), scale=1.0)
        @test_throws msg check_gauge_source(
            SM.Harmonic(1.0, 0.0), KerrSchildSource(T; M=1), T;
            center=(0.0, 0.0, 0.0), scale=1.0)
        @test_throws msg kerr_schild_case(T; M=2, halfwidth=5, r_0=3 // 2, r_1=3,
                                          chunk=1, gauge_source=KerrSchildSource(T; M=1))
        @test_throws "Kerr-Schild hole at rest" harmonic_kerr_case(
            T; halfwidth=5, r_0=1 // 4, r_1=1 // 2, chunk=1, gauge_source=:algebraic)
        @test_throws "speed of light" KerrSchildSource(T; M=1, velocity=(1, 0, 0))
    end

    @testset "a static hole runs on it as on the sampled source" begin
        # Guards the kernel path — `Val(:algebraic)` with the source itself
        # where the `Hsrc` array was, in the right-hand side, the gauge
        # constraint and the ADM monitor — on the suite's octant fixture
        # (`q = 2`, `h = 5/48`) to `1/5 M`: the same hole and the same exact
        # initial data, so the two runs differ only in how the source is
        # linearised, at the truncation error's size.
        q = 2
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        kw = (; halfwidth=T(5 // 2), r_0=T(2 // 5), r_1=T(23 // 20), chunk=T(1 // 10),
              octant=true)
        runs = map((nothing, :algebraic)) do gs
            case = kerr_schild_case(T; kw..., gauge_source=gs)
            evolve!(T, case; forest=gh_forest(T, case; N=12, roots=2), q=q, ops=ops,
                    t_end=T(1 // 5), adm_every=1)
        end
        s, a = runs
        @test s.problem.valH === Val(true) && s.problem.Hsrc isa FieldSet
        @test a.problem.valH === Val(:algebraic) && a.problem.Hsrc isa KerrSchildSource
        # The initial data are exact for both, so their monitors agree to
        # roundoff (the ADM monitor reaches `∂H` through each source's own
        # arithmetic).
        @test a.records[1].gauge_l2 ≈ s.records[1].gauge_l2 rtol = 1e-10
        @test a.records[1].ham_l2 ≈ s.records[1].ham_l2 rtol = 1e-10
        ra, rs = a.records[end], s.records[end]
        @info "sampled against algebraic at 1/5 M: err $(rs.err_l2) / $(ra.err_l2), " *
              "C $(rs.gauge_l2) / $(ra.gauge_l2), ℋ $(rs.ham_l2) / $(ra.ham_l2)"
        @test ra.finite && rs.finite
        @test 1 / 2 < ra.err_l2 / rs.err_l2 < 2
        @test 1 / 2 < ra.gauge_l2 / rs.gauge_l2 < 2
        @test 1 / 2 < ra.ham_l2 / rs.ham_l2 < 2
    end
end
