# The moving hole (step 8, G5): what a hole crossing the mesh needs that a
# static one does not — the Dirichlet data of a boosted spinning hole at the
# time of the evaluation, a seed track whose smallest radius is the boosted
# surface's, a level floor that covers the whole tracked layer and the
# distance it travels before the next regrid, and an initial-data cycle for
# a `:fitted` case that flags on the data the run starts from, so that G5's
# own chart — harmonic Kerr at `a = 7/10`, whose analytic core cuts the
# singular disk — has one.
#
# `CODE.md`, milestone G5, "Refinement and regridding" (what follows for the
# hole) and "Boundaries: Dirichlet"; `PLAN.md`'s step 8. The runs that cross
# the box are `test/hole_runs.jl moving`, minutes to hours on a node, and not
# here. Every claim below is host-side or one initial-data cycle; the one
# background this file evaluates on a mesh besides the fixtures' is
# `boost(Harmonic(1, a), 0.3 x̂)`, the type `prerequisite_tests.jl` already
# compiles, and harmonic Kerr at rest.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using KernelAbstractions: CPU
using StaticArrays: SVector
import SpacetimeMetrics as SM

const TGHm = TreeGeneralizedHarmonic

@testset verbose = true "The moving hole (step 8)" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    v = SVector{3,T}(3 // 10, 0, 0)
    boosted = SM.boost(SM.Harmonic(one(T), T(7 // 10)), v)

    # Guards the outer boundary of a hole that moves: the hook is built with
    # the evaluation's `t` (`CLAUDE.md`, "Hooks depend on time"), and for G5's
    # boosted spinning hole the analytic state at the boundary is a function
    # of time — a hook built at `t = 0`, or one that forgot the boost, would
    # write the wrong numbers into exactly the ghosts the scheme reads.
    @testset "the Dirichlet data of the boosted hole is the analytic state at t" begin
        case = GHCase(T, boosted; box=ntuple(_ -> (-T(5 // 4), T(5 // 4)), 3),
                      periodic=(false, false, false), ε_KO=zero(T), γ0=zero(T),
                      γ2=zero(T))
        @test case.center.v == -v                  # `hole_velocity`'s sign
        forest = gh_forest(T, case; N=8, roots=1)
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        sched = GhostSchedule(fs, ops)
        t = T(7 // 10)
        fill!(fs.work, zero(T))
        fill_ghosts!(fs, sched; boundary=dirichlet(case, t))
        worst = zero(T)
        for idx in ((1, 5, 5), (2, 5, 5), (G, 5, 5), (5, 1, 5), (5, 5, 2),
                    (8 + G + 1, 5, 5), (5, 8 + 2G + 1, 5), (8 + 2G + 1, 8 + 2G + 1, 1))
            vals = state_tuple(boosted, t, coordinates(fs, 1, idx))
            for w in 1:20
                worst = max(worst, abs(fs.work[idx..., w, 1] - vals[w]))
            end
        end
        @test worst == 0
        # The claim is not a tautology: the boundary data moves by far more
        # than roundoff between t = 0 and t.
        x = coordinates(fs, 1, (1, 5, 5))
        @test maximum(abs, state_tuple(boosted, t, x) .- state_tuple(boosted, zero(T), x)) >
              T(1 // 1000)
    end

    # Guards the first find of a boosted spinning hole: `horizon_min_radius`
    # of a boost is the rest frame's times `√(1 − v²)`, a bound that is exact
    # only when the smallest radius lies along the boost. G5's hole has it on
    # the spin axis, which a boost along x̂ does not contract; seeded with the
    # bound, the first find (0.716) is refused as a jump of 0.034 at
    # h = 5/256 (measured in step 8). A hole at rest must keep its seed.
    @testset "the seed of a boosted hole is its analytic surface" begin
        spec = FittedSpec(T; variant=:fitted, margin=4, lmax_shape=12)
        cb = hole_case(T, boosted; halfwidth=T(5 // 2), chunk=T(1 // 4), interior=spec,
                       horizon=Horizon(T; every=1, N=12, spin=false))
        tr = seed_track(cb, 0)
        @test horizon_min_radius(boosted) ≈ sqrt(T(51 // 100)) * sqrt(T(91 // 100))
        @test tr.r_min ≈ sqrt(T(51 // 100)) rtol = 1e-12   # the spin axis
        @test tr.r_min > horizon_min_radius(boosted) + T(3 // 100)
        @test tr.r_max == horizon_max_radius(boosted)
        for bg in (SM.Harmonic(one(T), T(7 // 10)), SM.KerrSchild(one(T), T(9 // 10)))
            cs = hole_case(T, bg; halfwidth=T(5 // 2), chunk=T(1 // 4), interior=spec,
                           horizon=Horizon(T; every=1, N=12, spin=false))
            @test seed_track(cs, 0).r_min === T(horizon_min_radius(bg))
        end
    end

    # Guards the level floor along a trajectory: a tracked geometry reads its
    # spacing over the whole layer, so a floor that starts at the offset
    # surface lets the blocks inside it coarsen and the next geometry double
    # its spacing — a mesh that loses a level per regrid; and a floor
    # evaluated at the regrid's `t` must cover where the layer will be at the
    # next one. A static hole's bounds must not move.
    @testset "the floor covers the tracked layer and its travel" begin
        ref = Refinement(T; refine_tol=T(2 // 5), coarsen_tol=T(1 // 10),
                         maxlevel_cap=3, floor_margin=T(1 // 10), ceiling_cells=1)
        case = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10),
                                interior=FittedSpec(T; margin=10), refinement=ref)
        forest = hole_fixture_forest(T, case; N=8)
        geom = fitted_interior(case.interior, seed_track(case, 0), forest, G; t=0,
                               n_L=8)
        lb0 = level_bounds(case, forest, 0, G; interior=geom)
        @test lb0.floor_lo == layer_radii(geom)[1]
        @test lb0.floor_hi == geom.r_out + T(1 // 10)
        lb1 = level_bounds(case, forest, 0, G; interior=geom, travel=T(3 // 40))
        @test lb1.floor_lo == layer_radii(geom)[1] - T(3 // 40)
        @test lb1.floor_hi == geom.r_out + T(1 // 10) + T(3 // 40)
        @test lb1.floor_level == lb0.floor_level
        # Step 5's sphere keeps its floor from r_1.
        ca = adaptive_hole_fixture(T)
        la = level_bounds(ca, gh_forest(T, ca; N=8, roots=4), 0, G)
        @test la.floor_lo == ca.interior.r_1
    end

    # Guards the target's rate (step 8): a point relaxing toward a moving
    # target at `ρ` lags it by `|∂_t u_fit|/ρ` unless the target's rate is fed
    # forward, and the rate the kernel reads is the cache's slope — which
    # must then be the fit's rate at a fixed point, its translation with the
    # track included. A fit at rest must fill the cache it filled before, and
    # the kernel with the rate must be the kernel without it where the rate
    # is zero.
    @testset "the cache's slope is the moving fit's rate at a point" begin
        cf = fitted_fixture(T; variant=:fitted)
        forest = hole_fixture_forest(T, cf; N=8)
        tr = seed_track(cf, 0)
        gf = with_ρ_max(fitted_interior(cf.interior, tr, forest, G; t=0, n_L=8), T(4))
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        bd = derive_target_bounds(T, cf.background, gf; t=0)
        fit = build_fit(analytic_sampler(cf.background, 0.0; δ=gf.h / 8), gf,
                        cf.interior; cont=1, bounds=bd)
        @test iszero(fit.params.center.v)
        origins = block_origins(forest, T)
        spacings = block_spacings(forest, T)
        c0 = target_cache(fs)
        c1 = target_cache(fs)
        fill_target!(c0, origins, spacings, gf, (fit, nothing), zero(T))
        fill_target!(c1, origins, spacings, gf, (fit, nothing), zero(T); rate=true)
        @test isequal(c0.work, c1.work)           # a fit at rest: no rate
        # The same fit, moving at 0.3 x̂: the slope against the cache's own
        # values filled a little later, at every filled point.
        p0 = fit.params
        pv = FitParams{T}(p0.lmax, p0.cont, p0.rbar,
                          HoleCenter(T, p0.center.c0, (T(3 // 10), 0, 0)),
                          p0.bounds, p0.tilde)
        mv = InteriorFit(pv, fit.coeffs, fit.host, fit.t, fit.c, fit.points,
                         fit.model, fit.residual, fit.conditioning, fit.valid,
                         fit.sweep)
        δ = T(1 // 1000)
        cr = target_cache(fs)
        cp = target_cache(fs)
        cm = target_cache(fs)
        fill_target!(cr, origins, spacings, gf, (mv, nothing), zero(T); rate=true)
        fill_target!(cp, origins, spacings, gf, (mv, nothing), δ)
        fill_target!(cm, origins, spacings, gf, (mv, nothing), -δ)
        worst = scale = zero(T)
        for b in 1:nblocks(fs), k in 1:8, j in 1:8, i in 1:8, v in 1:20
            d = (cp.work[i, j, k, v, b] - cm.work[i, j, k, v, b]) / (2δ)
            worst = max(worst, abs(cr.work[i, j, k, 20 + v, b] - d))
            scale = max(scale, abs(d))
        end
        # The two differences have steps an eighth of a cell and a
        # thousandth of `M` of travel; they agree to the first one's
        # truncation, `(h/8)²` relative (measured: 1.2e−4).
        @test scale > 1
        @test worst ≤ 1e-3 * scale
        # The kernel with the rate, on a cache whose slope is zero, is the
        # kernel without it.
        u = statevector(fs)
        fill_exact!(fs, cf, zero(T); interior=with_variant(gf, :damped))
        gather!(u, fs)
        pa = GHProblem(fs, GhostSchedule(fs, ops), cf; q=q, interior=gf,
                       target=c0, fits=(fit, nothing))
        pb = with_interior(pa, gf; target_rate=true)
        da, db = similar(u), similar(u)
        gh_rhs!(da, u, pa, zero(T))
        gh_rhs!(db, u, pb, zero(T))
        @test da == db
        @info "the cache's slope against its own difference (step 8)" worst scale
    end

    # Guards the cycle G5 needs: on a chart whose analytic interior is
    # singular — harmonic Kerr at a = 7/10, the disk inside the offset
    # surface — the analytic `:damped` geometry is refused, and the `:fitted`
    # cycle must still converge on data that is finite everywhere, the
    # analytic solution outside the offset surface bit for bit, the fit inside
    # it, with the floor holding the layer at the level its geometry was built
    # at.
    @testset "a :fitted case's cycle flags on its own data, on G5's chart" begin
        spec = FittedSpec(T; variant=:fitted, margin=4, lmax_shape=12)
        # The box of step 8f's refusal test, half-width 5/4 around a horizon
        # whose equator is at 1: too small for a ceiling (the floor's shell
        # meets the boundary blocks), so the ceiling is lifted to the cap.
        ref = Refinement(T; refine_tol=T(2 // 5), coarsen_tol=T(1 // 10),
                         maxlevel_cap=3, floor_margin=zero(T), ceiling_cells=1,
                         ceiling_level=3)
        c7 = hole_case(T, SM.Harmonic(one(T), T(7 // 10)); halfwidth=T(5 // 4),
                       chunk=T(1 // 20), interior=spec,
                       horizon=Horizon(T; every=1, N=12, spin=false), refinement=ref)
        f7 = hole_forest(T, c7; N=8, roots=1, radii=(T(10), T(8 // 5), T(13 // 10)))
        tr = seed_track(c7, 0)
        geometry(f) = fitted_interior(spec, tr, f, G; t=0, n_L=8)
        @test_throws "singular" check_interior_radii(f7, with_variant(geometry(f7), :damped),
                                                     c7.background, G; t=0,
                                                     center=c7.center)
        U = FieldSet{T}(f7, 20; G=G, centering=vertexcentered(3))
        sched, passes, converged, geom = adapt_fitted_initial_data!(
            U, ops, c7, geometry; G=G, buffer=1)
        @test converged && passes ≥ 1
        @test geom.h == T(5 // 128)
        u = statevector(U)
        gather!(u, U)
        @test all(isfinite, u)
        A = statearray(u, U)
        exact = true
        nout = nin = 0
        for b in 1:nblocks(U), k in 1:8, j in 1:8, i in 1:8
            x = coordinates(U, b, (i + G, j + G, k + G))
            g = interior_point(geom, zero(T), x)
            if g.r ≥ g.r_1
                nout += 1
                vals = state_tuple(c7.background, zero(T), x)
                exact &= all(w -> A[i, j, k, w, b] === vals[w], 1:20)
            else
                nin += 1
            end
        end
        @test exact && nout > 0 && nin > 0
        check_interior_radii(f7, geom, c7.background, G; t=0, center=c7.center)
        lb = level_bounds(c7, f7, 0, G; interior=geom)
        @test all(k -> level(k) ≥ first(block_level_bounds(lb, k)), f7.leaves)
        @info "G5's chart: the :fitted cycle (step 8)" passes nblocks = nleaves(f7) levels = forest_levels(f7)
    end

    # Guards the equivalence the cycle rests on: the indicator is masked
    # inside the offset surface, so on a chart that has analytic data to flag
    # on, the cycle on the fitted data and the cycle on the analytic layer's
    # data choose the same mesh — only the one point of Löhner's stencil
    # that reaches inside reads the fit.
    @testset "on the adaptive fixture it chooses the analytic cycle's mesh" begin
        ca = adaptive_hole_fixture(T; interior=FittedSpec(T; variant=:fitted,
                                                          margin=4, n_L=6),
                                   r_0=nothing, r_1=nothing,
                                   horizon=Horizon(T; every=1, N=12, spin=false))
        tr = seed_track(ca, 0)
        geometry(f) = fitted_interior(ca.interior, tr, f, G; t=0, n_L=6)
        fa = hole_forest(T, ca; N=8, roots=4, radii=(T(5 // 2),))
        fb = hole_forest(T, ca; N=8, roots=4, radii=(T(5 // 2),))
        Ua = FieldSet{T}(fa, 20; G=G, centering=vertexcentered(3))
        _, pa, ok, _ = adapt_fitted_initial_data!(Ua, ops, ca, geometry; G=G, buffer=1)
        Ub = FieldSet{T}(fb, 20; G=G, centering=vertexcentered(3))
        g0 = with_variant(geometry(fb), :damped)
        _, pb, okb = adapt_to_initial_data!(
            Ub, ops; initial=state_callback(ca, zero(T); interior=g0),
            flags=fs -> indicator_flags(fs, ca, zero(T); G=G, buffer=1,
                                        interior=with_variant(geometry(fs.forest),
                                                              :damped)).flags,
            buffer=0, boundary=dirichlet(ca, zero(T)))
        @test ok && okb
        @test fa.leaves == fb.leaves
        @info "the fitted and the analytic cycles (step 8)" passes = (pa, pb) nblocks = nleaves(fa)
    end
end
