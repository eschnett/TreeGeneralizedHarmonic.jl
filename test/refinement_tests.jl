# The refinement indicator: the Löhner ratio and its global floor, the four
# marks, the interior mask, the level floor around the horizon and the level
# ceiling at the outer boundary, and the initial-data cycle they drive.
#
# `CODE.md`, "Refinement and regridding". Three kinds of claim live here and
# they cost three different amounts, which is why they are separated:
#
#   * the **algebra** (`lohner`, `cell_tau`) — pure functions, microseconds;
#   * the **marks** — `refine_flags` on a τ field written by hand, so that
#     the four cases, the box's threshold and the two level bounds are
#     tested as the logic they are and not through a hole's metric;
#   * the **criterion on a hole** — a flagging pass and the initial-data
#     cycle, which fill initial data but evaluate no right-hand side, and
#     one short adaptive *run*, which does.
#
# The calibration table the thresholds come from, and the adaptive run at
# the calibrated cap, are in `test/hole_runs.jl`: at `refine_tol = 2/5` the
# indicator asks for 848 blocks around this hole, and a test file is not
# where that belongs. See `CODE.md`, "Measured results".

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: DIAG_TAU, NDIAG
using KernelAbstractions: CPU
using StaticArrays: SVector

@testset verbose = true "The refinement indicator" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)

    # Bounds with neither floor nor ceiling: the marks of the indicator
    # alone, which is what the four-mark logic is about.
    plain_bounds(forest, cap) =
        LevelBounds{T,typeof(forest)}(forest, SVector{3,T}(0, 0, 0), zero(T),
                                      -one(T), 0,
                                      ntuple(_ -> (-T(5), T(5)), 3),
                                      (false, false, false), zero(T), cap, cap)

    # Refinement is for *resolution*, not amplitude: τ is a ratio of
    # undivided differences, so it falls with `h` on smooth data — which is
    # what makes a fixed threshold terminate refinement — and does not move
    # when the data is scaled.
    @testset "τ measures the mesh: it falls with h and is amplitude-blind" begin
        @test lohner(one(T), one(T), one(T), one(T)) == 0          # constant
        @test lohner(zero(T), one(T), T(2), one(T)) == 0           # linear
        # A ratio in [0, 1] by construction: the second difference is
        # bounded by the two first differences.
        for (um, u0, up) in ((T(0), T(1), T(0)), (T(-1), T(2), T(5)),
                             (T(3), T(-1), T(3)))
            τ = lohner(um, u0, up, one(T))
            @test 0 ≤ τ ≤ 1
        end
        # `f = 2M/r` sampled around `r = 6/5`, at `h` and at `h/2`: τ falls
        # with the spacing, which is the whole point of undivided
        # differences and what makes a fixed threshold terminate.
        #
        # **Between linear and quadratic, and both ends matter.** The
        # numerator is `O(h²)` and the denominator is `2|f′|h + 4ε·scale`,
        # so τ halves while the gradient term dominates and quarters once
        # the floor does — the floor is what stops a smooth region from
        # scoring a constant τ forever, and it is why refinement terminates
        # rather than merely slowing down.
        f(r) = 2 / r
        τ_of(h) = lohner(f(T(6 // 5) - h), f(T(6 // 5)), f(T(6 // 5) + h),
                         T(2))
        @test τ_of(T(1 // 10)) > τ_of(T(1 // 20)) > τ_of(T(1 // 40))
        for h in (T(1 // 10), T(1 // 20))
            @test 1 // 4 ≤ τ_of(h / 2) / τ_of(h) ≤ 1 // 2
        end
        # Scaling the data and its reference together leaves τ alone.
        s = T(1e3)
        @test lohner(s * f(T(11 // 10)), s * f(T(6 // 5)), s * f(T(13 // 10)),
                     s * T(2)) ≈
              lohner(f(T(11 // 10)), f(T(6 // 5)), f(T(13 // 10)), T(2))
    end

    # TreeWave's measured failure, kept as a negative control: with the
    # floor referred to the *local* values — which is Löhner's classic form
    # and what a per-component reference of zero degenerates to — numerical
    # dust ten orders of magnitude below the data scores τ ≈ 1, and the
    # whole domain refines.
    @testset "the noise floor is global, or the dust refines everything" begin
        um, u0, up = T(3.9e-16), T(3.6e-17), T(3.6e-17)
        @test lohner(um, u0, up, abs(u0)) > 0.9        # the local floor
        @test lohner(um, u0, up, zero(T)) > 0.9        # no floor at all
        @test lohner(um, u0, up, one(T)) < 1e-13       # a global amplitude
        # And the floor does not blunt a real feature at the same scale:
        # a kink of the *data's* own size still scores.
        @test lohner(zero(T), one(T), zero(T), one(T)) > 1 // 10
    end

    # `CODE.md`: `U_ref` is "the largest |h| in the *evolved* region". The
    # frozen core holds the analytic solution at `r_0`, where `|h| = 2M/r_0`
    # is several times its largest evolved value, so an unmasked reference
    # would inflate the floor by a factor that depends on `r_0` — a
    # parameter of the layer, not a property of the solution.
    @testset "the reference amplitude is masked, and shared by the ten" begin
        case = adaptive_hole_fixture(T)
        forest = gh_forest(T, case; N=8, roots=2)
        U = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        fill_exact!(U, case, zero(T))
        origins = TreeGeneralizedHarmonic.to_backend(CPU(),
                                                     block_origins(forest, T))
        spacings = TreeGeneralizedHarmonic.to_backend(CPU(),
                                                      block_spacings(forest, T))
        masked = field_scales(U, interior_mask(case.interior, zero(T)), origins,
                              spacings)
        unmasked = field_scales(U, AllPoints(), origins, spacings)
        @test maximum(masked) < maximum(unmasked)
        # `h_tt = 2M/r` at the layer's edge against the core's sphere.
        @test maximum(masked) ≈ 2 / case.interior.r_1 rtol = 1 // 4
        @test maximum(unmasked) ≈ 2 / case.interior.r_0 rtol = 1 // 4
        @test field_scale(U, interior_mask(case.interior, zero(T)), origins,
                          spacings) == maximum(masked)

        # The trap the shared amplitude exists for: in the gauge wave's
        # chart six of the ten components are **identically zero**, so a
        # per-component reference is exactly zero and the component's own
        # dust would be measured against no floor at all.
        gw = gauge_wave_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T))
        gwforest = gh_forest(T, gw; N=8, roots=1)
        GW = FieldSet{T}(gwforest, 20; G=G, centering=vertexcentered(3))
        fill_exact!(GW, gw, zero(T))
        go = TreeGeneralizedHarmonic.to_backend(CPU(),
                                                block_origins(gwforest, T))
        gs = TreeGeneralizedHarmonic.to_backend(CPU(),
                                                block_spacings(gwforest, T))
        gwscales = field_scales(GW, AllPoints(), go, gs)
        @test count(iszero, gwscales) ≥ 6
        @test field_scale(GW, AllPoints(), go, gs) > 0
    end

    # The four marks of `CODE.md`, on a τ field written by hand: the logic
    # is what is under test, and a hole's metric would only make it harder
    # to read which case fired.
    @testset "the four marks, with the box keyed on coarsen_tol" begin
        case = adaptive_hole_fixture(T; maxlevel_cap=2)
        forest = gh_forest(T, case; N=8, roots=4, refined=true)
        ref = Refinement(T; refine_tol=T(2 // 5), coarsen_tol=T(1 // 10),
                         maxlevel_cap=2, floor_margin=zero(T))
        lb = plain_bounds(forest, 2)
        τfs = FieldSet{T}(forest, NDIAG; G=0, centering=vertexcentered(3))
        levels = [level(forest.leaves[b]) for b in 1:nblocks(τfs)]
        b0 = findfirst(==(0), levels)          # a coarse block
        b1 = findfirst(==(1), levels)          # a block at the cap below
        @test b0 !== nothing && b1 !== nothing

        # Everything quiet: a bare `Coarsen` above level 0, a bare `Keep`
        # at it. A quiet `Keep` must stay bare, or every quiescent block
        # would recruit its neighbours and coarsening would die globally.
        fill!(τfs.work, zero(T))
        marks = refine_flags(τfs, lb, ref).flags
        @test marks[b0] === Keep
        @test all(b -> marks[b] === (levels[b] > 0 ? Coarsen : Keep),
                  1:nblocks(τfs))

        # One block under-resolved, below the cap: `(Refine, box)`, and the
        # box is the bounding box of the cells above **coarsen_tol** — a
        # wider set than the cells above refine_tol, which is the point.
        v = interiorview(τfs, b0, DIAG_TAU)
        fill!(v, zero(T))
        v[3, 4, 5] = T(9 // 10)                      # above refine_tol
        v[2, 4, 5] = v[6, 4, 5] = T(1 // 5)          # above coarsen_tol only
        marks = refine_flags(τfs, lb, ref).flags
        @test marks[b0] isa Tuple && first(marks[b0]) === Refine
        @test last(marks[b0]) == (2:6, 4:4, 5:5)

        # The same block at the cap: `(Keep, box)`, the equal-level margin
        # that travels with the feature. Keying the box on `refine_tol`
        # instead would make this block report nothing, and the margin
        # would be unreachable in the one case it exists for.
        capped = Refinement(T; refine_tol=T(2 // 5), coarsen_tol=T(1 // 10),
                            maxlevel_cap=0, floor_margin=zero(T))
        marks = refine_flags(τfs, plain_bounds(forest, 0), capped).flags
        @test marks[b0] isa Tuple && first(marks[b0]) === Keep
        @test last(marks[b0]) == (2:6, 4:4, 5:5)

        # Between the thresholds, below the cap: the feature is there and
        # resolved, so the block is held with its box and not refined.
        fill!(v, zero(T))
        v[3, 4, 5] = T(1 // 5)
        marks = refine_flags(τfs, lb, ref).flags
        @test marks[b0] isa Tuple && first(marks[b0]) === Keep
        @test last(marks[b0]) == (3:3, 4:4, 5:5)

        # The gap between the thresholds is the dead band, and a
        # `Refinement` without one is refused rather than accepted as a
        # degenerate case.
        @test_throws ArgumentError Refinement(T; refine_tol=T(1 // 10),
                                              coarsen_tol=T(1 // 10),
                                              maxlevel_cap=1,
                                              floor_margin=zero(T))
        @test_throws ArgumentError Refinement(T; refine_tol=T(2),
                                              coarsen_tol=T(1 // 10),
                                              maxlevel_cap=1,
                                              floor_margin=zero(T))
        @test_throws ArgumentError Refinement(T; refine_tol=T(2 // 5),
                                              coarsen_tol=T(1 // 10),
                                              maxlevel_cap=1,
                                              floor_margin=zero(T),
                                              ε=zero(T))
    end

    # The level floor is derived from the interior's own radii, so that
    # `check_interior_radii` holds on the mesh the indicator settles at by
    # construction rather than by the indicator's mood.
    @testset "the level floor is the level the interior's radii need" begin
        case = adaptive_hole_fixture(T)
        int = case.interior
        forest = gh_forest(T, case; N=8, roots=4)
        ℓ = horizon_floor_level(forest, int, case.background, G)
        @test ℓ == 1
        h = spacing(T, forest, ℓ)
        # Both requirements hold at the floor's spacing and fail one level
        # coarser — which is what "the coarsest level that satisfies them"
        # means.
        @test int.r_1 ≤ horizon_min_radius(case.background) - int.margin * h
        @test int.r_1 - int.r_0 ≥ 2 * (G + 1) * h
        hc = spacing(T, forest, ℓ - 1)
        @test !(int.r_1 ≤ horizon_min_radius(case.background) -
                          int.margin * hc &&
                int.r_1 - int.r_0 ≥ 2 * (G + 1) * hc)
        # A cap below it is refused by name: the refinement would be asked
        # for a level it may not reach, and the interior's assertion would
        # fire on the mesh it settled at.
        @test_throws ArgumentError level_bounds(
            with_refinement(case, Refinement(T; refine_tol=T(2 // 5),
                                             coarsen_tol=T(1 // 10),
                                             maxlevel_cap=0,
                                             floor_margin=zero(T))),
            forest, zero(T), G)
    end

    # `CODE.md` predicts the floor never binds at calibrated thresholds and
    # asks for the test that it *can* to use a deliberately loose
    # `refine_tol`. Both halves are here, on one flagging pass, by comparing
    # the marks with the floor against the marks without it.
    @testset "the floor does not bind when calibrated, and binds when loose" begin
        case = adaptive_hole_fixture(T)
        pass = gh_flagging_pass(T, case; N=8, roots=4, q=q)
        forest = pass.forest
        lb = level_bounds(case, forest, zero(T), G)
        @test lb.floor_level == 1
        τfs = FieldSet{T}(forest, NDIAG; G=0, centering=vertexcentered(3))
        origins = TreeGeneralizedHarmonic.to_backend(CPU(),
                                                     block_origins(forest, T))
        spacings = TreeGeneralizedHarmonic.to_backend(CPU(),
                                                      block_spacings(forest, T))
        mask = interior_mask(case.interior, zero(T))
        gh_tau!(τfs, pass.U, origins, spacings, mask;
                scale=field_scale(pass.U, mask, origins, spacings),
                ε=case.refinement.ε)
        nofloor = LevelBounds{T,typeof(forest)}(forest, lb.center, lb.floor_lo,
                                                lb.floor_hi, 0, lb.box,
                                                lb.periodic, lb.ceiling_margin,
                                                lb.ceiling_level,
                                                lb.maxlevel_cap)
        tight = refine_flags(τfs, lb, case.refinement).flags
        without = refine_flags(τfs, nofloor, case.refinement).flags
        @test tight == without

        # Loosened past what the data can reach, the indicator asks for
        # nothing and the floor is the only thing refining: the blocks
        # meeting the shell `r_1 ≤ r ≤ r_h,max` still come back `Refine`,
        # and they are exactly the blocks the shell meets.
        loose = Refinement(T; refine_tol=T(99 // 100), coarsen_tol=T(9 // 10),
                           maxlevel_cap=1, floor_margin=zero(T))
        floored = refine_flags(τfs, lb, loose).flags
        flat = refine_flags(τfs, nofloor, loose).flags
        @test floored != flat
        @test all(m -> m === Keep, flat)     # level 0 everywhere, nothing fires
        nref = count(m -> m === Refine || (m isa Tuple && first(m) === Refine),
                     floored)
        @test nref > 0
        @test nref == count(1:nleaves(forest)) do b
            first(block_level_bounds(lb, forest.leaves[b])) > 0
        end
    end

    # The interior is masked out of the indicator: the layer and the frozen
    # core are not a numerical solution, and the core's stale data has steep
    # differences that mean nothing.
    @testset "the indicator is zero inside r_1, and the core cannot fire" begin
        case = adaptive_hole_fixture(T)
        pass = gh_flagging_pass(T, case; N=8, roots=4, q=q)
        forest = pass.forest
        τfs = FieldSet{T}(forest, NDIAG; G=0, centering=vertexcentered(3))
        origins = TreeGeneralizedHarmonic.to_backend(CPU(),
                                                     block_origins(forest, T))
        spacings = TreeGeneralizedHarmonic.to_backend(CPU(),
                                                      block_spacings(forest, T))
        mask = interior_mask(case.interior, zero(T))
        gh_tau!(τfs, pass.U, origins, spacings, mask;
                scale=field_scale(pass.U, mask, origins, spacings),
                ε=case.refinement.ε)
        inside = 0
        worst_outside = zero(T)
        for b in 1:nblocks(τfs)
            v = interiorview(τfs, b, DIAG_TAU)
            k = forest.leaves[b]
            h = spacing(T, forest, k)
            o = block_origin(T, forest, k)
            for I in CartesianIndices(v)
                x = ntuple(d -> o[d] + (Tuple(I)[d] - 1) * h, 3)
                r = sqrt(sum(abs2, x))
                if r < case.interior.r_1
                    inside += 1
                    @test v[I] == 0
                else
                    worst_outside = max(worst_outside, v[I])
                end
            end
        end
        @test inside > 0
        @test worst_outside > 0
        @test tau_max(τfs) == worst_outside
        # Without the mask the core's stale interior would score, which is
        # the number the mask exists to keep out of the verdict.
        unmasked = FieldSet{T}(forest, NDIAG; G=0, centering=vertexcentered(3))
        gh_tau!(unmasked, pass.U, origins, spacings, AllPoints();
                scale=field_scale(pass.U, mask, origins, spacings),
                ε=case.refinement.ε)
        @test tau_max(unmasked) > tau_max(τfs)
    end

    # The cycle is `CODE.md`'s: fill, flag, regrid without transferring,
    # re-evaluate on the new mesh. What it must converge to is nested shells
    # around the hole with the boundary left coarse.
    @testset "the initial-data cycle converges to shells around the hole" begin
        case = adaptive_hole_fixture(T)
        cyc = gh_adapt_cycle(T, case; N=8, roots=4, q=q)
        forest = cyc.forest
        @test cyc.converged
        @test cyc.passes ≤ 4
        @test nleaves(forest) == 120
        @test forest_levels(forest) == [56, 64]
        @test minimum_spacing(T, forest) ≈ T(5 // 32)

        # Nested and centred: every finest block is within a horizon radius
        # or so of the hole, and every block at the boundary is coarse.
        lb = level_bounds(case, forest, zero(T), G)
        for k in forest.leaves
            ext = block_extent(T, forest, k)
            near = sqrt(sum(d -> max(ext[d][1], -ext[d][2], zero(T))^2, 1:3))
            if level(k) == maxlevel(forest)
                @test near < 2 * horizon_max_radius(case.background)
            end
            # `CODE.md`'s ceiling: the blocks within a few coarse cells of
            # the outer boundary sit at the coarsest level.
            touching = any(1:3) do d
                ext[d][1] ≤ case.box[d][1] + lb.ceiling_margin ||
                    ext[d][2] ≥ case.box[d][2] - lb.ceiling_margin
            end
            if touching
                @test level(k) == lb.ceiling_level
            end
        end

        # The interior's radius requirements hold on the mesh the indicator
        # chose — which is what the level floor is for, and what a fresh
        # `GHProblem` asserts after every regrid.
        info = check_interior_radii(forest, case.interior, case.background, G)
        @test info.h ≈ T(5 // 32)
        U = cyc.U
        p = GHProblem(U, cyc.schedule, case; q=q)
        @test p.interior === case.interior

        # A converged hierarchy is a fixed point: flagging it again moves
        # nothing. `CODE.md`: "regrids during a static run should change
        # nothing".
        before = copy(forest.leaves)
        moved = regrid!(forest, U => cyc.schedule; flags=cyc.flags, buffer=0,
                        boundary=dirichlet(case, zero(T)))
        @test !moved
        @test forest.leaves == before

        # And the centroid of what fired is the hole, to the bias the
        # coarsest firing blocks impose (`refinement_centroid`).
        @test cyc.centroid !== nothing
        @test sqrt(sum(abs2, cyc.centroid)) < 4 * minimum_spacing(T, forest)
        @info("the indicator's mesh on the static hole", passes=cyc.passes,
              leaves=nleaves(forest), levels=forest_levels(forest),
              τ_max=cyc.τ_max, scale=cyc.scale, nfiring=cyc.nfiring,
              centroid=cyc.centroid,
              offset_in_h=sqrt(sum(abs2, cyc.centroid)) /
                          minimum_spacing(T, forest))
    end

    # The margin the hole travels, and the cadence it implies.
    @testset "the travelling margin is derived, and refuses a long chunk" begin
        case = adaptive_hole_fixture(T)
        forest = gh_forest(T, case; N=8, roots=4)
        # A static hole still gets one cell: TreeAMR measured that a margin
        # narrower than the motion it covers is worse than none at all.
        @test refinement_buffer(forest, 1, zero(T)) == 1
        h = spacing(T, forest, 1)
        @test refinement_buffer(forest, 1, 3 * h) == 4
        # More than one finest block per regrid is refused by name: TreeAMR
        # recruits exactly one ring of neighbours.
        @test_throws ArgumentError refinement_buffer(forest, 1, 9 * h)
    end

    # The whole loop: adapt, evolve, flag, regrid — on a static hole, where
    # the answer is that nothing moves. This is the only testset here that
    # evaluates a right-hand side.
    @testset "an adaptive static run regrids and changes nothing" begin
        case = adaptive_hole_fixture(T)
        forest = gh_forest(T, case; N=8, roots=4)
        out = evolve!(T, case; forest=forest, q=q, ops=ops, t_end=T(3 // 20),
                      adapt=true, regrid=true)
        @test out.converged
        @test out.nregrids == 0                 # the mesh is a fixed point
        @test out.nblocks == 120
        @test out.levels == [56, 64]
        @test out.buffer == 1
        @test length(out.records) == 4
        for r in out.records
            @test r.finite
            @test r.τ_max !== nothing && isfinite(r.τ_max)
            @test r.centroid !== nothing
            @test r.centroid_offset < 4 * out.h
            @test r.nblocks == 120
        end
        # `CODE.md`'s mesh statistics: τ_max is a property of the mesh and
        # the solution, and on a static hole it does not drift.
        τs = [r.τ_max for r in out.records]
        @test maximum(τs) - minimum(τs) < T(1 // 100) * maximum(τs)
        # The masked error is the fixed-hierarchy run's: the indicator
        # bought the same finest spacing around the hole.
        last_r = out.records[end]
        @test last_r.err_l2 > 0
        @test last_r.residual > 0
        @info("the adaptive static run", leaves=out.nblocks, levels=out.levels,
              h=out.h, passes=out.passes, nregrids=out.nregrids,
              τ_max=last_r.τ_max, centroid=last_r.centroid,
              offset_in_h=last_r.centroid_offset / out.h,
              err_l2=last_r.err_l2, gauge_l2=last_r.gauge_l2,
              residual=last_r.residual)
    end

    # The criterion is generic in the element type like everything else
    # here, and the one way it could stop being is a `Float64` literal in
    # the noise floor — which would drag every `τ` into `Float64` however
    # the field is stored. `lohner`'s `ε` is a rational converted with
    # `oftype`, and this is what says so.
    @testset "the indicator computes in the type the case is stated in" begin
        S = Float32
        case = adaptive_hole_fixture(S)
        @test case.refinement isa Refinement{S}
        pass = gh_flagging_pass(S, case; N=8, roots=4, q=q)
        @test eltype(pass.τfs.work) === S
        @test pass.τ_max isa S
        @test pass.scale isa S
        @test pass.centroid isa SVector{3,S}
        @test lohner(S(1), S(2), S(3.5), S(2)) isa S
        # The same mesh as the `Float64` pass, to `Float32`'s accuracy in
        # the threshold comparisons: the verdict is a set of marks and it
        # is the *same* set.
        ref64 = gh_flagging_pass(T, adaptive_hole_fixture(T); N=8, roots=4,
                                 q=q)
        @test pass.flags == ref64.flags
        @test pass.τ_max ≈ ref64.τ_max rtol = sqrt(eps(S))
    end

    # The driver has one refinement mechanism, and says so.
    @testset "the driver refuses to regrid without an indicator" begin
        case = hole_fixture(T; q=q)              # no `Refinement`
        forest = hole_fixture_forest(T, case; N=8)
        @test_throws ArgumentError evolve!(T, case; forest=forest, q=q, ops=ops,
                                           t_end=T(1 // 20), regrid=true)
        @test_throws ArgumentError evolve!(T, case; forest=forest, q=q, ops=ops,
                                           t_end=T(1 // 20), adapt=true)
        @test_throws ArgumentError indicator_flags(
            FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3)), case,
            zero(T); G=G)
    end
end
