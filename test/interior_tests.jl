# The interior: the profiles, the core rule, the mask, and the two radius
# assertions.
#
# `CODE.md`, "The interior: a pointwise damping layer". Everything here is
# a function of position and time; the one thing that touches a mesh is
# `check_interior_radii`, which is an assertion *about* the placement and
# not an input to it. The evolutions the layer takes part in are in
# `driver_tests.jl`; this file is the algebra, the predicates and the
# fixture, and it is deliberately cheap.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using KernelAbstractions: CPU
using StaticArrays: SVector
import ForwardDiff
import SpacetimeMetrics as SM

@testset verbose = true "The interior" begin
    T = Float64

    # A profile that is only `C¹` leaves a jump in `∂²w` that the compact
    # second derivative reads as a delta, and the layer would then show up
    # in every convergence study as a feature of its own.
    @testset "the profiles are C² and exactly flat outside the layer" begin
        @test smoothstep(-one(T)) === zero(T)
        @test smoothstep(zero(T)) === zero(T)
        @test smoothstep(one(T)) === one(T)
        @test smoothstep(T(2)) === one(T)
        @test smoothstep(T(1 // 2)) ≈ T(1 // 2)
        # value, first and second derivative all vanish at both ends: the
        # quintic's defining property, checked by differences rather than
        # asserted from the coefficients.
        # The difference quotients of a quintic whose first *three* terms
        # vanish at the join are `O(δ²)` and `O(δ)`, and the bounds below
        # are those rates and not round numbers: `S(δ) = 10δ³` gives
        # `d1 = 5δ²` and `d2 = 10δ` exactly on the `s = 0` side, and a
        # `C¹`-only ramp would give `d2 = O(1/δ)` instead.
        δ = T(1e-4)
        for s0 in (zero(T), one(T))
            d1 = (smoothstep(s0 + δ) - smoothstep(s0 - δ)) / (2δ)
            d2 = (smoothstep(s0 + δ) - 2 * smoothstep(s0) +
                  smoothstep(s0 - δ)) / δ^2
            @test abs(d1) ≤ 10 * δ^2
            @test abs(d2) ≤ 20 * δ
        end

        int = Interior(T; center=(0, 0, 0), r_0=T(2 // 5), r_1=T(23 // 20),
                       ρ_max=T(7))
        # Outside: untouched, and *exactly* so — the evolved region is
        # multiplied by a hard 1 and a hard 0.
        for r in (int.r_1, T(2), T(10))
            w, ρ = interior_profiles(int, r)
            @test w === one(T)
            @test ρ === zero(T)
        end
        # Inside the core: switched off, at full relaxation.
        for r in (zero(T), int.r_0 / 2, int.r_0)
            w, ρ = interior_profiles(int, r)
            @test w === zero(T)
            @test ρ === int.ρ_max
        end
        # In between: monotone, bounded, and complementary — no point is
        # both frozen (`w = 0`) and undamped (`ρ = 0`).
        rs = range(int.r_0, int.r_1; length=41)
        ws = [interior_profiles(int, T(r))[1] for r in rs]
        ρs = [interior_profiles(int, T(r))[2] for r in rs]
        @test all(0 .≤ ws .≤ 1)
        @test all(0 .≤ ρs .≤ int.ρ_max)
        @test issorted(ws)
        @test issorted(ρs; rev=true)
        @test all(i -> ws[i] > 0 || ρs[i] > 0, eachindex(ws))

        # `:frozen` is the pure mask: the same `w`, and ρ identically zero
        # — read off the variant and not off a zero `ρ_max`, so that a
        # misconfigured `:damped` is not silently the same run.
        fr = Interior(T; center=(0, 0, 0), r_0=T(2 // 5), r_1=T(23 // 20),
                      ρ_max=T(7), variant=:frozen)
        for r in range(zero(T), T(2); length=17)
            @test interior_profiles(fr, T(r))[1] ===
                  interior_profiles(int, T(r))[1]
            @test interior_profiles(fr, T(r))[2] === zero(T)
        end
    end

    # The analytic solution is singular at the center; a core filled from
    # it directly would be `Inf` or `NaN`, and `0 · NaN = NaN` would then
    # reach every norm and every stencil that touches the core.
    @testset "the core rule gives finite data where the solution is singular" begin
        int = Interior(T; center=(0, 0, 0), r_0=T(2 // 5), r_1=T(23 // 20))
        bg = SM.KerrSchild(one(T), zero(T))

        # Outside the core it is the identity, bit for bit.
        x = (T(3 // 4), T(1 // 2), T(-1 // 4))
        @test core_position(int, zero(T), x) === x
        @test core_position(nothing, zero(T), x) === x
        # Inside it, the radial projection onto the sphere `r_0`.
        y = core_position(int, zero(T), (T(1 // 10), zero(T), zero(T)))
        @test sqrt(sum(abs2, y)) ≈ int.r_0
        @test y[1] > 0 && y[2] == 0 && y[3] == 0
        # At the center the ray is undefined and `+ẑ` is taken: the axis is
        # the regular direction for both hole charts.
        z = core_position(int, zero(T), (zero(T), zero(T), zero(T)))
        @test z === (zero(T), zero(T), int.r_0)

        # Finite everywhere inside, including at the center — which is
        # where the raw background is not.
        @test all(isfinite, case_state_tuple(bg, int, zero(T),
                                             (zero(T), zero(T), zero(T))))
        @test !all(isfinite, state_tuple(bg, zero(T),
                                         (zero(T), zero(T), zero(T))))
        # Continuous at `r_0`: the rule and the background agree there.
        n = SVector{3,T}(T(3 // 5), T(12 // 25), T(16 // 25))
        xin = Tuple(int.r_0 * n * (1 - 1e-9))
        xout = Tuple(int.r_0 * n * (1 + 1e-9))
        a = case_state_tuple(bg, int, zero(T), xin)
        b = case_state_tuple(bg, int, zero(T), xout)
        @test maximum(abs, a .- b) < 1e-6 * maximum(abs, b)
    end

    # `CODE.md` asks for the two placement bounds at every regrid, and
    # `CLAUDE.md` says what to do when one fires: raise the resolution,
    # shrink the layer, or widen the floor — never lower `m`. So both have
    # to *fire*, and the test is that they do.
    @testset "the radius assertions hold, and fire when they should" begin
        q = 2
        G = q ÷ 2 + 1
        case = hole_fixture(T; q=q)
        forest = hole_fixture_forest(T, case; N=8)
        h = minimum_spacing(T, forest)
        info = check_interior_radii(forest, case.interior, case.background, G)
        @test info.h == h                     # the layer sits at the finest level
        @test info.nblocks > 0
        @test info.r_h_min ≈ 2                # r₊ = 2M for Kerr-Schild, a = 0
        @test case.interior.r_1 ≤ info.allowed
        @test info.thickness ≥ info.needed

        # (a) the layer creeping out toward the horizon.
        toofar = Interior(T; center=(0, 0, 0), r_0=T(2 // 5),
                          r_1=T(19 // 10), margin=8)
        @test_throws ArgumentError check_interior_radii(forest, toofar,
                                                        case.background, G)
        # (b) the layer too thin for the stencils.
        toothin = Interior(T; center=(0, 0, 0), r_0=T(11 // 10),
                           r_1=T(23 // 20), margin=8)
        @test_throws ArgumentError check_interior_radii(forest, toothin,
                                                        case.background, G)
        # (c) a margin below the floor `G + 1`, which is the one remedy
        # `CLAUDE.md` forbids.
        small = Interior(T; center=(0, 0, 0), r_0=T(2 // 5), r_1=T(23 // 20),
                         margin=G)
        @test_throws ArgumentError check_interior_radii(forest, small,
                                                        case.background, G)
        # (d) a hole outside the mesh entirely.
        away = Interior(T; center=(100, 0, 0), r_0=T(2 // 5),
                        r_1=T(23 // 20), margin=8)
        @test_throws ArgumentError check_interior_radii(forest, away,
                                                        case.background, G)
        # (e) a background with no horizon this package knows.
        @test_throws ArgumentError horizon_min_radius(SM.Minkowski())
        # (f) a core that does not contain the chart's singular set. Both
        # Kerr charts are singular on the equatorial disk `|x| ≤ |a|`, not
        # at a point, so `r_0 > |a|` is a requirement and not a formality —
        # and in the *harmonic* chart at `a = 9/10` it cannot be met at
        # all, because the disk's `0.9` is larger than the horizon's
        # smallest coordinate radius `√(M² − a²) = 0.436`. That is the one
        # configuration `CODE.md`'s proof of concept asks for, and it is
        # refused by name rather than discovered as a `NaN`.
        spun = kerr_schild_case(T; M=1, a=T(9 // 10), halfwidth=T(5 // 2),
                                r_0=T(1 // 2), r_1=T(23 // 20),
                                chunk=T(1 // 10), margin=4)
        @test_throws ArgumentError check_interior_radii(forest, spun.interior,
                                                        spun.background, G)
        harm = harmonic_kerr_case(T; M=1, a=T(9 // 10), halfwidth=T(5 // 2),
                                  r_0=T(19 // 20), r_1=T(1), chunk=T(1 // 10),
                                  margin=2)
        @test_throws ArgumentError check_interior_radii(forest, harm.interior,
                                                        harm.background, G)
        # The arithmetic behind it, stated so that a change to either
        # radius has to face it: a ball fits between the disk and the
        # horizon only where `√(M² − a²) > |a|`, that is `a < M/√2`.
        @test singular_radius(SM.KerrSchild(one(T), T(9 // 10))) ≈ T(9 // 10)
        @test singular_radius(SM.Harmonic(one(T), T(9 // 10))) ≈ T(9 // 10)
        @test singular_radius(SM.KerrSchild(one(T), zero(T))) == 0
        @test horizon_min_radius(SM.Harmonic(one(T), T(9 // 10))) <
              singular_radius(SM.Harmonic(one(T), T(9 // 10)))
        @test horizon_min_radius(SM.KerrSchild(one(T), T(9 // 10))) >
              singular_radius(SM.KerrSchild(one(T), T(9 // 10)))
        @test horizon_min_radius(SM.Harmonic(one(T), T(7 // 10))) >
              singular_radius(SM.Harmonic(one(T), T(7 // 10)))

        # And the problem constructor is where it runs, since a fresh
        # problem is built after every regrid.
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                         backend=CPU())
        sched = GhostSchedule(fs, Operators(prolongation=q + 2,
                                            restriction=q + 2))
        @test GHProblem(fs, sched, case; q=q) isa GHProblem
        @test_throws ArgumentError GHProblem(fs, sched, case; q=q,
                                             interior=toofar)
    end

    # The horizon's coordinate radii are what the layer is placed inside
    # of; getting `r_h,min` wrong is getting every placement wrong.
    @testset "the horizon radii are the analytic ones" begin
        for M in (one(T), T(2))
            @test horizon_min_radius(SM.KerrSchild(M, zero(T))) ≈ 2M
            @test horizon_max_radius(SM.KerrSchild(M, zero(T))) ≈ 2M
            @test horizon_min_radius(SM.Harmonic(M, zero(T))) ≈ M
            @test horizon_max_radius(SM.Harmonic(M, zero(T))) ≈ M
        end
        a = T(9 // 10)
        @test horizon_min_radius(SM.Harmonic(one(T), a)) ≈ sqrt(1 - a^2)
        @test horizon_min_radius(SM.Harmonic(one(T), a)) ≈ 0.4358898943540673
        @test horizon_max_radius(SM.Harmonic(one(T), a)) ≈ one(T)
        rp = 1 + sqrt(1 - a^2)
        @test horizon_min_radius(SM.KerrSchild(one(T), a)) ≈ rp
        @test horizon_max_radius(SM.KerrSchild(one(T), a)) ≈ sqrt(rp^2 + a^2)
        # A translation and a rotation move the surface without changing a
        # radius; a boost contracts the minimum by √(1 − v²) and leaves the
        # maximum, which is the transverse extent.
        base = SM.Harmonic(one(T), a)
        @test horizon_min_radius(SM.translate(base, SVector{4,T}(0, 1, 2, 3))) ≈
              horizon_min_radius(base)
        @test horizon_min_radius(SM.rotate(base, T(1 // 5), T(2 // 5),
                                           T(3 // 5))) ≈
              horizon_min_radius(base)
        v = T(3 // 10)
        bst = SM.boost(base, SVector{3,T}(v, 0, 0))
        @test horizon_min_radius(bst) ≈ horizon_min_radius(base) * sqrt(1 - v^2)
        @test horizon_max_radius(bst) ≈ horizon_max_radius(base)
    end

    # The frozen hierarchy is what the convergence protocol is stated on:
    # if the block layout moved with `N`, a rate measured on it would be a
    # rate of two different meshes.
    @testset "hole_forest nests, and doubling N leaves the layout alone" begin
        case = hole_fixture(T)
        f8 = hole_fixture_forest(T, case; N=8)
        f16 = hole_fixture_forest(T, case; N=16)
        @test f8.leaves == f16.leaves
        @test nleaves(f8) == 120
        @test forest_levels(f8) == [0, 0, 56, 64]
        @test minimum_spacing(T, f16) ≈ minimum_spacing(T, f8) / 2
        # The sphere `r_1` lies wholly inside the finest level, which is
        # what makes `layer_spacing` report that level's `h`.
        h, nb = layer_spacing(f8, case.interior, zero(T))
        @test h == minimum_spacing(T, f8)
        @test nb > 0
        # Nesting, and the refusals.
        @test_throws ArgumentError hole_forest(T, case; N=8, roots=1,
                                               radii=(T(1), T(3)))
        @test_throws ArgumentError hole_forest(T, case; N=8, roots=1,
                                               radii=(T(3), T(3)), levels=3)
        @test_throws ArgumentError hole_forest(T, case; N=8, roots=1,
                                               center=(T(100), zero(T),
                                                       zero(T)),
                                               radii=(T(3), T(3)))
        # Shells wide enough to catch every block are a *uniform* mesh, and
        # the fixture's third shell is what makes it a hierarchy: the
        # coarse-fine face is the difference.
        f2 = hole_forest(T, case; N=8, roots=1, radii=(T(3), T(3)))
        @test forest_levels(f2) == [0, 0, 64]
        @test maxlevel(f8) == 3
        @test forest_levels(f8)[3] > 0 && forest_levels(f8)[4] > 0
    end

    # `CODE.md`: the modified region is not a numerical solution and must
    # not be reported as one. The mask is the sentence that says so.
    @testset "the mask excludes exactly the ball r < r_1" begin
        int = Interior(T; center=(1, 0, 0), r_0=T(2 // 5), r_1=T(23 // 20))
        m = interior_mask(int, zero(T))
        @test m isa InteriorMask
        @test interior_mask(nothing, zero(T)) === AllPoints()
        @test is_evolved(AllPoints(), (zero(T), zero(T), zero(T)))
        @test !is_evolved(m, (one(T), zero(T), zero(T)))          # the center
        @test !is_evolved(m, (one(T) + int.r_1 * (1 - 1e-9), zero(T), zero(T)))
        @test is_evolved(m, (one(T) + int.r_1, zero(T), zero(T)))
        @test is_evolved(m, (T(10), zero(T), zero(T)))
        # It follows the center, which is a function of `t` and not a field.
        moving = Interior(T; center=HoleCenter(T, (0, 0, 0), (1, 0, 0)),
                          r_0=T(2 // 5), r_1=T(23 // 20))
        @test !is_evolved(interior_mask(moving, T(3)),
                          (T(3), zero(T), zero(T)))
        @test is_evolved(interior_mask(moving, zero(T)),
                         (T(3), zero(T), zero(T)))
        # A shell counts only its own points.
        sh = ShellMask{T}(SVector{3,T}(0, 0, 0), one(T), T(2))
        @test !is_evolved(sh, (T(1 // 2), zero(T), zero(T)))
        @test is_evolved(sh, (T(3 // 2), zero(T), zero(T)))
        @test !is_evolved(sh, (T(3), zero(T), zero(T)))
    end

    # Everything the kernels close over is a kernel argument, so all of it
    # must be `isbits`: no `Type`, no host array, no mutated field.
    @testset "the interior and the case travel into kernels" begin
        case = hole_fixture(T)
        @test isbitstype(typeof(case))
        @test isbitstype(typeof(case.interior))
        @test isbitstype(typeof(case.γ0))
        @test isbitstype(typeof(interior_mask(case.interior, zero(T))))
        @test isbitstype(typeof(ShellMask{T}(SVector{3,T}(0, 0, 0), one(T),
                                             T(2))))
        # A case without a hole is `isbits` too, which is why the absent
        # interior is a concrete `Nothing` field and not a `Union`.
        flat = minkowski_case(T; L=one(T), ε_KO=zero(T), γ0=one(T),
                              γ2=zero(T))
        @test isbitstype(typeof(flat))
        @test flat.interior === nothing
        @test interior_variant(flat.interior) === :none
        @test interior_variant(case.interior) === :damped
    end

    # The constraint-damping rate is a function of position from step 5 on;
    # a constant one has to stay exactly the number it was, or every flat
    # -space result of steps 3 and 4 moves.
    @testset "the damping profile is GHSO2's recipe near the hole" begin
        c = ConstantDamping(T, T(3 // 2))
        @test damping_rate(c, zero(T), (T(7), T(-2), T(5))) === T(3 // 2)
        @test damping_bounds(c) == (T(3 // 2), T(3 // 2))
        g = GaussianDamping(T; near=one(T), far=T(1 // 10), width=T(3),
                            center=HoleCenter(T, (0, 0, 0)))
        @test damping_rate(g, zero(T), (zero(T), zero(T), zero(T))) ≈ one(T)
        @test damping_rate(g, zero(T), (T(100), zero(T), zero(T))) ≈ T(1 // 10)
        @test damping_bounds(g) == (T(1 // 10), one(T))
        # Monotone in `r`, and it follows a moving center.
        rs = range(zero(T), T(20); length=41)
        @test issorted([damping_rate(g, zero(T), (T(r), zero(T), zero(T)))
                        for r in rs]; rev=true)
        gm = GaussianDamping(T; near=one(T), far=T(1 // 10), width=T(3),
                             center=HoleCenter(T, (0, 0, 0), (1, 0, 0)))
        @test damping_rate(gm, T(5), (T(5), zero(T), zero(T))) ≈ one(T)
        @test_throws ArgumentError GaussianDamping(T; near=one(T),
                                                   far=-one(T), width=T(3),
                                                   center=HoleCenter(T,
                                                                     (0, 0, 0)))
        @test_throws ArgumentError GHCase(T, SM.Minkowski();
                                          box=ntuple(_ -> (zero(T), one(T)),
                                                     3),
                                          periodic=(true, true, true),
                                          ε_KO=zero(T), γ0=-one(T),
                                          γ2=zero(T))
    end

    # Step 8c's `ε_KO(r)`: a profile that is only `C¹` at either join puts a
    # delta into the second difference of the dissipation, and one that is
    # not *exactly* the exterior's number outside the horizon changes the
    # exterior's arithmetic — which is what every earlier result was
    # measured with. A number must stay the number, bit for bit.
    @testset "the dissipation profile is C², exact outside r_h and inside r_1" begin
        @test dissipation_rate(T(1 // 2), T(3), (T(7), T(-2), T(5))) === T(1 // 2)
        @test dissipation_bounds(T(1 // 2)) == (T(1 // 2), T(1 // 2))
        @test has_dissipation(T(1 // 2)) && !has_dissipation(zero(T))
        d = HorizonDissipation(T; ε_out=T(1 // 2), ε_in=T(4), r_1=T(23 // 20),
                               r_h=T(2), center=HoleCenter(T, (0, 0, 0)))
        ε(r) = dissipation_rate(d, zero(T), (r, zero(r), zero(r)))
        for r in (T(2), T(5 // 2), T(10))
            @test ε(r) === T(1 // 2)
        end
        for r in (zero(T), T(1 // 2), T(23 // 20))
            @test ε(r) === T(4)
        end
        @test issorted([ε(T(r)) for r in range(T(1), T(21 // 10); length=45)];
                       rev=true)
        @test dissipation_bounds(d) == (T(1 // 2), T(4))
        @test has_dissipation(d)
        # C² at both joins: the first and second derivatives along the ray
        # are those of the quintic's triple root, `O(δ²)` and `O(δ)` with
        # the constants `30|Δε|/L` and `60|Δε|/L²` (`L = r_h − r_1`), and
        # exactly zero on the constant side.
        L = T(2) - T(23 // 20)
        Δ = T(7 // 2)
        d1(r) = ForwardDiff.derivative(ε, r)
        d2(r) = ForwardDiff.derivative(d1, r)
        δ = L / 1000
        for (r_j, side) in ((T(23 // 20), 1), (T(2), -1))
            @test d1(r_j - side * δ) === zero(T)
            @test d2(r_j - side * δ) === zero(T)
            @test abs(d1(r_j + side * δ)) ≤ 1.01 * 30 * Δ * (δ / L)^2 / L
            @test abs(d2(r_j + side * δ)) ≤ 1.01 * 60 * Δ * (δ / L) / L^2
        end
        # It follows a moving center, as the damping profile does.
        dm = HorizonDissipation(T; ε_out=T(1 // 2), ε_in=T(2), r_1=one(T),
                                r_h=T(2), center=HoleCenter(T, (0, 0, 0),
                                                            (1, 0, 0)))
        @test dissipation_rate(dm, T(5), (T(5), zero(T), zero(T))) === T(2)
        # The refusals: RK4's real-axis limit, a sign, and the geometry.
        hd(; kw...) = HorizonDissipation(T; ε_out=T(1 // 2), ε_in=T(2),
                                         r_1=one(T), r_h=T(2),
                                         center=(0, 0, 0), kw...)
        @test_throws "capped at 4" hd(ε_in=T(5))
        @test_throws ArgumentError hd(ε_out=-one(T))
        @test_throws "0 < r_1 < r_h" hd(r_1=T(2))
        # A case takes a number or a profile about its own hole, and
        # `horizon_dissipation` builds that one from the case.
        case = hole_fixture(T)
        prof = horizon_dissipation(case; ε_in=T(4))
        @test prof.r_1 === case.interior.r_1
        @test prof.r_h === T(horizon_min_radius(case.background))
        @test prof.ε_out === case.ε_KO
        @test with_dissipation(case, prof).ε_KO === prof
        @test isbitstype(typeof(with_dissipation(case, prof)))
        @test_throws "centered on" with_dissipation(
            case, HorizonDissipation(T; ε_out=T(1 // 2), ε_in=T(2), r_1=one(T),
                                     r_h=T(2), center=(1, 0, 0)))
        @test_throws ArgumentError with_dissipation(case, :loud)
    end

    # Step 8c's layer target: a metric the kernel evaluates in place of the
    # background, so it has to be one and it has to be `isbits`; and a case
    # with no layer has nowhere to put it.
    @testset "the layer's target is a metric or nothing, and nothing is the background" begin
        bg = SM.KerrSchild(one(T), zero(T))
        int = Interior(T; center=(0, 0, 0), r_0=T(2 // 5), r_1=T(23 // 20))
        @test int.target === nothing
        @test layer_target(int, bg) === bg
        wrong = SM.KerrSchild(T(6 // 5), zero(T))
        intw = Interior(T; center=(0, 0, 0), r_0=T(2 // 5), r_1=T(23 // 20),
                        target=wrong)
        @test layer_target(intw, bg) === wrong
        @test isbitstype(typeof(intw))
        @test with_ρ_max(intw, T(3)).target === wrong
        @test_throws "SpacetimeMetrics metric" Interior(
            T; center=(0, 0, 0), r_0=T(2 // 5), r_1=T(23 // 20), target=one(T))
        @test hole_fixture(T; target=wrong).interior.target === wrong
        @test_throws "no interior" GHCase(T, SM.Minkowski();
                                          box=ntuple(_ -> (zero(T), one(T)), 3),
                                          periodic=(true, true, true),
                                          ε_KO=zero(T), γ0=one(T), γ2=zero(T),
                                          target=wrong)
    end

    @testset "the interior's constructor refuses what it cannot place" begin
        @test_throws ArgumentError Interior(T; center=(0, 0, 0), r_0=zero(T),
                                            r_1=one(T))
        @test_throws ArgumentError Interior(T; center=(0, 0, 0), r_0=one(T),
                                            r_1=T(1 // 2))
        @test_throws ArgumentError Interior(T; center=(0, 0, 0),
                                            r_0=T(1 // 10), r_1=one(T),
                                            variant=:excised)
        @test_throws ArgumentError Interior(T; center=(0, 0, 0),
                                            r_0=T(1 // 10), r_1=one(T),
                                            w_ramp=zero(T))
        @test_throws ArgumentError Interior(T; center=(0, 0, 0),
                                            r_0=T(1 // 10), r_1=one(T),
                                            ρ_max=-one(T))
        # `ρ_max` is the driver's, once per chunk, and replacing it changes
        # nothing else.
        int = Interior(T; center=(0, 0, 0), r_0=T(1 // 10), r_1=one(T))
        int2 = with_ρ_max(int, T(17))
        @test int2.ρ_max === T(17)
        @test int2.r_0 === int.r_0 && int2.r_1 === int.r_1
        @test interior_variant(int2) === interior_variant(int)
    end
end

# The interior as the kernel sees it: one right-hand-side evaluation, and
# the two claims that can only be made on a mesh — that `(INTERIOR)` is
# the term `CODE.md` writes, and that `F` is not evaluated in the core.
@testset verbose = true "The (INTERIOR) term on a mesh" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    case = hole_fixture(T; q=q)
    forest = hole_fixture_forest(T, case; N=8)
    fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                     backend=CPU())
    sched = GhostSchedule(fs, Operators(prolongation=q + 2, restriction=q + 2))
    ρ_max = T(37 // 5)
    p_damped = GHProblem(fs, sched, case; q=q,
                         interior=with_ρ_max(case.interior, ρ_max))
    p_none = with_interior(p_damped, nothing)
    int = p_damped.interior

    # Exact data, then a perturbation, so that the relaxation term is not
    # identically zero and the comparison below has something to compare.
    fill_exact!(fs, case, zero(T))
    u = statevector(fs)
    gather!(u, fs)
    u0 = copy(u)
    for i in eachindex(u)
        u[i] += T(1 // 1000) * sin(T(i))
    end
    uin = copy(u)

    du_i = similar(u)
    du_n = similar(u)
    gh_rhs!(du_i, u, p_damped, zero(T))
    gh_rhs!(du_n, u, p_none, zero(T))

    # The exact solution on the state's own layout, for the relaxation's
    # reference; `fill_exact!` applies the core rule, as the kernel does.
    exact = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                        backend=CPU())
    fill_exact!(exact, case, zero(T))
    ue = statevector(exact)
    gather!(ue, exact)

    A_i = statearray(du_i, fs)
    A_n = statearray(du_n, fs)
    A_u = statearray(uin, fs)
    A_e = statearray(ue, fs)

    ncore = 0
    nlayer = 0
    nout = 0
    worst = zero(T)
    scale = zero(T)
    core_zero = true
    out_identical = true
    for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N, i in 1:forest.N
        x = coordinates(fs, b, (i + G, j + G, k + G))
        r = sqrt(sum(abs2, x))
        if r < int.r_0
            ncore += 1
            core_zero &= all(v -> A_i[i, j, k, v, b] === zero(T), 1:20)
        elseif r ≥ int.r_1
            nout += 1
            out_identical &= all(v -> A_i[i, j, k, v, b] === A_n[i, j, k, v, b],
                                 1:20)
        else
            nlayer += 1
            w, ρ = interior_profiles(int, r)
            for v in 1:20
                want = w * A_n[i, j, k, v, b] -
                       ρ * (A_u[i, j, k, v, b] - A_e[i, j, k, v, b])
                worst = max(worst, abs(A_i[i, j, k, v, b] - want))
                scale = max(scale, abs(want))
            end
        end
    end

    # Each region has to be non-empty, or the claims below are vacuous.
    @testset "the mesh actually has a core, a layer and an evolved region" begin
        @test ncore > 0
        @test nlayer > 0
        @test nout > 0
    end

    # `CODE.md`: `∂_t u = w(r) F(u) − ρ(r)(u − u_exact)`, with `w = 1` and
    # `ρ = 0` outside `r_1` — which has to be the *same numbers* as a run
    # with no interior at all, not merely close ones.
    @testset "du is w F − ρ (u − u_exact), and F itself outside r_1" begin
        @test out_identical
        @test core_zero
        @test worst ≤ 1e-12 * scale
    end

    # `CLAUDE.md`: the core holds finite but stale data on which `F` may be
    # `NaN`, and `0 · NaN = NaN`. Plant a *degenerate* metric there — one
    # `metric_quantities` cannot survive — and the evolved region must not
    # notice, which is the layer's thickness requirement doing its job.
    @testset "F is not evaluated in the frozen core" begin
        upoison = copy(uin)
        A_p = statearray(upoison, fs)
        planted = 0
        for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N,
            i in 1:forest.N

            x = coordinates(fs, b, (i + G, j + G, k + G))
            sqrt(sum(abs2, x)) < int.r_0 || continue
            planted += 1
            # h = −η makes g = 0: √γ = 0 and every coefficient is NaN.
            A_p[i, j, k, 1, b] = one(T)
            for v in 2:10
                A_p[i, j, k, v, b] = v in (5, 8, 10) ? -one(T) : zero(T)
            end
        end
        @test planted == ncore
        du_p = similar(u)
        gh_rhs!(du_p, upoison, p_damped, zero(T))
        A_pd = statearray(du_p, fs)
        allzero = true
        same = true
        for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N,
            i in 1:forest.N

            x = coordinates(fs, b, (i + G, j + G, k + G))
            r = sqrt(sum(abs2, x))
            if r < int.r_0
                allzero &= all(v -> A_pd[i, j, k, v, b] === zero(T), 1:20)
            elseif r ≥ int.r_1
                same &= all(v -> A_pd[i, j, k, v, b] === A_i[i, j, k, v, b],
                            1:20)
            end
        end
        @test allzero
        @test same
        @test upoison !== u                     # and `u` is still untouched

        # And the *time step* does not notice either. `CODE.md` lists the
        # speed kernel among the ones that mask the interior, and this is
        # why: a degenerate metric in the core gives a `NaN` speed, and an
        # unmasked `max_speed` would hand it to `gh_dt` as the bound for
        # the whole hierarchy.
        scatter!(fs, upoison)
        λ_poisoned = max_speed(p_damped)
        # Unmasked, the same state either throws on the way to a speed or
        # returns something that is not the answer — which is the disaster
        # the mask prevents, and the reason it is a branch and not a
        # multiplication by zero.
        λ_unmasked = try
            max_speed(p_damped; mask=AllPoints())
        catch
            T(NaN)
        end
        scatter!(fs, uin)
        λ_clean = max_speed(p_damped)
        @test isfinite(λ_poisoned)
        @test λ_poisoned === λ_clean
        @test !(isfinite(λ_unmasked) && λ_unmasked == λ_clean)
    end

    # The right-hand side is a pure function of `(u, t)` — TreeAMR's
    # contract, and the reason `(INTERIOR)` is a term and not a write.
    @testset "the right-hand side never mutates u" begin
        @test uin == u
        du2 = similar(u)
        gh_rhs!(du2, u, p_damped, zero(T))
        @test du2 == du_i
    end

    # `:pasted` is the one variant that writes the state, and it writes
    # exactly the ball `r < r_1` and nothing else.
    @testset ":pasted writes the ball r < r_1 and nothing outside it" begin
        pasted = hole_fixture(T; q=q, variant=:pasted)
        p_p = GHProblem(fs, sched, pasted; q=q,
                        interior=with_ρ_max(pasted.interior, ρ_max))
        up = copy(uin)
        paste_interior!(p_p, up, zero(T))
        A_p = statearray(up, fs)
        A_before = statearray(uin, fs)
        inside_exact = true
        outside_same = true
        for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N,
            i in 1:forest.N

            x = coordinates(fs, b, (i + G, j + G, k + G))
            r = sqrt(sum(abs2, x))
            if r < int.r_1
                # To roundoff, not `===`: `A_e` and `A_p` reach the
                # analytic solution by two different call sites, whose
                # results are bit-identical only by luck of inlining. See
                # the same claim in `driver_tests.jl`. `outside_same`
                # below stays exact, because there the claim *is* that
                # nothing was written.
                inside_exact &= all(v -> isapprox(A_p[i, j, k, v, b],
                                                  A_e[i, j, k, v, b];
                                                  rtol=100 * eps(T),
                                                  atol=100 * eps(T)), 1:20)
            else
                outside_same &= all(v -> A_p[i, j, k, v, b] ===
                                         A_before[i, j, k, v, b], 1:20)
            end
        end
        @test inside_exact
        @test outside_same
        # And the other three variants write nothing at all.
        for other in (p_damped, p_none)
            uq = copy(uin)
            paste_interior!(other, uq, zero(T))
            @test uq == uin
        end
        # `du = 0` inside `r_1` for `:pasted`, which is what makes the
        # limiter the whole of the interior's treatment there.
        du_p = similar(u)
        gh_rhs!(du_p, uin, p_p, zero(T))
        A_pd = statearray(du_p, fs)
        allzero = true
        for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N,
            i in 1:forest.N

            x = coordinates(fs, b, (i + G, j + G, k + G))
            sqrt(sum(abs2, x)) < int.r_1 || continue
            allzero &= all(v -> A_pd[i, j, k, v, b] === zero(T), 1:20)
        end
        @test allzero
    end

    # Step 8c's two claims on the kernel. A target equal to the case's own
    # background must be *that* background to the last bit — or every
    # comparison of a wrong target against it compares a recompilation — and
    # a wrong target must move the right-hand side only where `(INTERIOR)`
    # reads `u_exact`: the layer, and nowhere an evolved stencil sits.
    @testset "the layer target reaches the layer and nothing else" begin
        own = Interior(T; center=(0, 0, 0), r_0=int.r_0, r_1=int.r_1,
                       ρ_max=ρ_max, target=case.background)
        du_own = similar(u)
        gh_rhs!(du_own, u, with_interior(p_damped, own), zero(T))
        @test isequal(du_own, du_i)
        wrong = Interior(T; center=(0, 0, 0), r_0=int.r_0, r_1=int.r_1,
                         ρ_max=ρ_max, target=SM.KerrSchild(T(6 // 5), zero(T)))
        du_w = similar(u)
        gh_rhs!(du_w, u, with_interior(p_damped, wrong), zero(T))
        A_w = statearray(du_w, fs)
        same_outside = true
        nmoved = 0
        for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N,
            i in 1:forest.N

            x = coordinates(fs, b, (i + G, j + G, k + G))
            r = sqrt(sum(abs2, x))
            if in_layer(int, r)
                nmoved += any(v -> A_w[i, j, k, v, b] != A_i[i, j, k, v, b],
                              1:20)
            else
                same_outside &= all(v -> A_w[i, j, k, v, b] ===
                                         A_i[i, j, k, v, b], 1:20)
            end
        end
        @test same_outside
        @test nmoved == nlayer
    end

    # And the `ε_KO(r)` profile: one whose two values are the case's number
    # is that number at every point, so the kernel's arithmetic must be the
    # number's bit for bit; one that rises from the horizon must move `du`
    # only inside it.
    @testset "the dissipation profile is the number where it equals it" begin
        flat = HorizonDissipation(T; ε_out=case.ε_KO, ε_in=case.ε_KO,
                                  r_1=int.r_1,
                                  r_h=horizon_min_radius(case.background),
                                  center=case.center)
        p_flat = GHProblem(fs, sched, with_dissipation(case, flat); q=q,
                           interior=with_ρ_max(case.interior, ρ_max))
        du_f = similar(u)
        gh_rhs!(du_f, u, p_flat, zero(T))
        @test isequal(du_f, du_i)
        rising = horizon_dissipation(case; ε_in=T(4))
        p_rise = GHProblem(fs, sched, with_dissipation(case, rising); q=q,
                           interior=with_ρ_max(case.interior, ρ_max))
        du_r = similar(u)
        gh_rhs!(du_r, u, p_rise, zero(T))
        A_r = statearray(du_r, fs)
        same_outside = true
        nmoved = 0
        ninside = 0
        for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N,
            i in 1:forest.N

            x = coordinates(fs, b, (i + G, j + G, k + G))
            r = sqrt(sum(abs2, x))
            if r ≥ rising.r_h || r < int.r_0
                same_outside &= all(v -> A_r[i, j, k, v, b] ===
                                         A_i[i, j, k, v, b], 1:20)
            else
                ninside += 1
                nmoved += any(v -> A_r[i, j, k, v, b] != A_i[i, j, k, v, b],
                              1:20)
            end
        end
        @test same_outside
        @test nmoved == ninside > 0
    end

    # `CODE.md` prices the layer at "a few percent of an RHS" and asks G4
    # to measure it: one dual pass through the background per layer point
    # per evaluation. Recorded, not asserted — it is a cost, not a claim.
    @testset "the layer's share of an evaluation is recorded" begin
        reps = 3
        gh_rhs!(du_i, u, p_damped, zero(T))
        t0 = time()
        for _ in 1:reps
            gh_rhs!(du_i, u, p_damped, zero(T))
        end
        t_int = (time() - t0) / reps
        gh_rhs!(du_n, u, p_none, zero(T))
        t0 = time()
        for _ in 1:reps
            gh_rhs!(du_n, u, p_none, zero(T))
        end
        t_none = (time() - t0) / reps
        npts = nblocks(fs) * forest.N^3
        @info("the layer's share of an RHS evaluation", q, npts, nlayer,
              ncore, t_none, t_int, share=(t_int - t_none) / t_none)
        @test t_int > 0 && t_none > 0
    end
end
