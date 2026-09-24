# The tracked geometry: the real harmonics, the shape and its depth, the
# kernels on a surface, and the track that places it.
#
# `CODE.md`, "The interior" — "The tracked geometry" (step 8d) — and
# `PLAN.md`'s step 8d. The file is in three parts, cheapest first. The first
# is host-side algebra and evaluates no background on a mesh: the real
# harmonics against `AbstractSphericalHarmonics`, the analytic horizon against
# the charts' own quartic, the depth of an oblate spheroid, the footprint
# guard on a surface that is not a sphere, the leakage e-folds against step
# 8a's script. The second is one right-hand side on the step-5 fixture — the
# tracked geometry holding step 5's sphere is bit for bit step 5's layer —
# and one find of its initial data. The third is the runs: the track coasting
# and lost, the lapse-collapse trigger, and a tracked hole against the
# analytic one. Every run is on `hole_fixture`'s mesh; its margin is `m = 10`
# rather than the default `8` because the fixture's finest level is the cube
# `|x|_∞ ≤ 5/4`, and a tracked offset surface has to lie inside it, as step
# 5's `r_1 = 23/20` does (see `fitted_fixture`).

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using KernelAbstractions: CPU
using Random: MersenneTwister
using StaticArrays: SVector
import AbstractSphericalHarmonics as ASH
import SpacetimeMetrics as SM

# The tracked case, `fitted_fixture`, and its one `3/20 M` run,
# `tracked_fixture_run`, are in `evolution_cases.jl` (moved there in step
# 8e, whose fit reads the same geometry and the same run).

# Random real coefficients in the canonical complex layout: `c_l0` real and
# `c_{l,−m} = (−1)^m c̄_lm`, the reality condition of a real function.
function real_function_coefficients(rng, L)
    c = zeros(ComplexF64, (L + 1)^2)
    for l in 0:L
        c[real_harmonic_index(l, 0)] = randn(rng)
        for m in 1:l
            z = complex(randn(rng), randn(rng))
            c[real_harmonic_index(l, m)] = z
            c[real_harmonic_index(l, -m)] = (-1)^m * conj(z)
        end
    end
    return c
end

unit(v) = v / sqrt(sum(abs2, v))

# Nanoseconds per point of `f` over `pts`, behind a function barrier so that
# what is timed is the call and not the test's own dispatch.
function ns_per_point(f, pts, reps)
    s = 0.0
    for p in pts
        s += f(p)
    end
    t0 = time_ns()
    for _ in 1:reps, p in pts
        s += f(p)
    end
    return (time_ns() - t0) / (reps * length(pts)), s
end

@testset verbose = true "The tracked geometry: shape, depth, guard" begin
    # A wrong sign or a wrong √2 in the real harmonics is a horizon rotated
    # or rescaled by a mode — a shape that looks right on the axis and is
    # wrong everywhere else — and step 8e's fit reads the same convention.
    @testset "the real harmonics are the complex ones, to roundoff" begin
        rng = MersenneTwister(20260923)
        for L in (4, 8), T in (Float64, Float32)
            c = real_function_coefficients(rng, L)
            a = real_from_complex(c, L)
            sv = SVector{(L + 1)^2,T}(T.(a))
            worst = 0.0
            scale = 0.0
            for _ in 1:200
                θ = acos(2 * rand(rng) - 1)
                φ = 2π * rand(rng)
                ref = sum(c[real_harmonic_index(l, m)] * ASH.sYlm(0, l, m, θ, φ)
                          for l in 0:L for m in (-l):l)
                n = SVector{3,T}(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
                worst = max(worst, abs(Float64(shape_series(sv, L, n)) - real(ref)))
                scale = max(scale, abs(ref))
            end
            # Measured: 6.4 eps (lmax 8) and 3.7 eps (lmax 4) at Float64,
            # 5.3 and 1.6 eps at Float32, of the largest value.
            @test worst ≤ 16 * eps(T) * scale
            # The round trip through the complex layout is the identity.
            @test maximum(abs.(complex_from_real(a, L) .- c)) ≤ 1e-15 * maximum(abs.(c))
        end
        # And on the grid `ash_evaluate` samples: the same function.
        L = 6
        c = real_function_coefficients(rng, L)
        grid = ASH.EquiangularGrid(L)
        f = ASH.ash_evaluate(grid, c, 0)
        sv = SVector{(L + 1)^2}(real_from_complex(c, L))
        worst = maximum(CartesianIndices(f)) do ij
            θ, φ = ASH.ash_point_coord(grid, ij)
            n = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
            abs(shape_series(sv, L, n) - real(f[ij]))
        end
        @test worst ≤ 16 * eps() * maximum(abs, f)
        # The layout is the canonical one: slot `l² + l + m + 1`.
        @test real_harmonic_index(0, 0) == 1
        @test real_harmonic_index(2, -2) == 5
        @test real_harmonic_index(2, 2) == 9
        @test Int(ASH.ash_mode_index(grid, 0, 3, -1)[1]) == real_harmonic_index(3, -1)
    end

    # The seed is the surface the first chunk's layer is an offset of: a
    # seed that is not the horizon puts the layer's margin somewhere else.
    # `SpacetimeMetrics` exposes no horizon, so the check is the charts' own
    # quartic for their radial coordinate, evaluated on the seed's surface.
    @testset "the analytic horizon is on the charts' own spheroid" begin
        ks_r(x, a) = (s = x[1]^2 + x[2]^2 + x[3]^2 - a^2;
                      sqrt((s + sqrt(s^2 + 4 * a^2 * x[3]^2)) / 2))
        rng = MersenneTwister(3)
        dirs = [unit(SVector{3}(randn(rng, 3))) for _ in 1:50]
        for a in (0.0, 0.5, 0.9)
            ks = SM.KerrSchild(1.0, a)
            ha = SM.Harmonic(1.0, a)
            for n in dirs
                @test ks_r(analytic_horizon_radius(ks, n) * n, a) ≈
                      horizon_min_radius(ks) rtol = 1e-12
                @test ks_r(analytic_horizon_radius(ha, n) * n, a) ≈
                      horizon_min_radius(ha) rtol = 1e-12
            end
            @test analytic_horizon_radius(ks, SVector(0.0, 0.0, 1.0)) ≈
                  horizon_min_radius(ks)
            @test analytic_horizon_radius(ks, SVector(1.0, 0.0, 0.0)) ≈
                  horizon_max_radius(ks)
            @test analytic_horizon_radius(ha, SVector(0.0, 0.0, 1.0)) ≈
                  horizon_min_radius(ha)
            @test analytic_horizon_radius(ha, SVector(0.0, 1.0, 0.0)) ≈
                  horizon_max_radius(ha)
        end
        ks = SM.KerrSchild(1.0, 0.9)
        # Rotated: the surface is turned, so the rotated axis is where the
        # smallest radius is.
        rot = SM.rotate(ks, 0.3, 0.7, -0.4)
        axis = SVector(rot.R[2, 4], rot.R[3, 4], rot.R[4, 4])
        @test analytic_horizon_radius(rot, axis) ≈ horizon_min_radius(ks)
        for n in dirs
            nold = SVector{3}(ntuple(j -> sum(rot.R[1 + i, 1 + j] * n[i] for i in 1:3),
                                     3))
            @test analytic_horizon_radius(rot, n) ≈ analytic_horizon_radius(ks, nold)
        end
        # Boosted: contracted along v by √(1 − v²), unchanged across it, and
        # in the rest frame the boosted surface is the horizon again.
        v = SVector(0.3, 0.0, 0.0)
        bst = SM.boost(ks, v)
        γ = 1 / sqrt(1 - 0.09)
        @test analytic_horizon_radius(bst, SVector(1.0, 0.0, 0.0)) ≈
              horizon_max_radius(ks) / γ
        @test analytic_horizon_radius(bst, SVector(0.0, 0.0, 1.0)) ≈
              horizon_min_radius(ks)
        for n in dirs
            x = analytic_horizon_radius(bst, n) * n
            xrest = SVector(γ * x[1], x[2], x[3])
            @test ks_r(xrest, 0.9) ≈ horizon_min_radius(ks) rtol = 1e-12
        end
    end

    # `PLAN.md`: "the depth of an oblate spheroid recovers its axis and
    # equator". The depth is what the layer is keyed on, so a shape whose
    # truncation moves the surface by a cell moves the layer by a cell.
    @testset "the depth of an oblate spheroid recovers its axis and equator" begin
        T = Float64
        errs = Dict{Tuple{String,Int},Float64}()
        for (name, bg) in (("KerrSchild", SM.KerrSchild(1.0, 0.9)),
                           ("Harmonic", SM.Harmonic(1.0, 0.9)))
            R_ax = horizon_min_radius(bg)
            R_eq = horizon_max_radius(bg)
            for L in (4, 8)
                shape = analytic_shape(bg, L)
                off = R_ax / 8
                int = FittedInterior(T; center=(0, 0, 0), shape=shape, lmax=L,
                                     offset=off, thickness=R_ax / 4)
                # The depth at a point on the axis and on the equator, read
                # back as the horizon's radius there: `r_h = d + offset + r`.
                r = R_ax / 2
                g_ax = fitted_geometry(int, zero(T), (0.0, 0.0, r))
                g_eq = fitted_geometry(int, zero(T), (r, 0.0, 0.0))
                e_ax = abs(g_ax.d + off + r - R_ax)
                e_eq = abs(g_eq.d + off + r - R_eq)
                # And the bounding radii are the extremes to the same order.
                e_in = abs(int.r_in - R_ax)
                e_out = abs(int.r_out - R_eq)
                # The worst direction, which for a truncated series need not
                # be the axis or the equator.
                e_all = maximum(shape_sample_directions(2L)) do n
                    abs(shape_radius(int, SVector{3,T}(n)) -
                        analytic_horizon_radius(bg, n))
                end
                errs[(name, L)] = e_all
                @test g_ax.d > 0 && g_eq.d > 0
                # `|min f − min g| ≤ sup|f − g|`, with the sup sampled.
                @test max(e_ax, e_eq, e_in, e_out) ≤ 3 * e_all / 2 + 1e-14
            end
        end
        # Measured (step 8d): Kerr-Schild `a = 9/10` is `1.2e−3` at lmax 4 and
        # `8.1e−6` at lmax 8; harmonic Kerr at `a = 9/10`, whose spheroid is
        # four times as oblate in the ratio that matters (`a²/R² = 4.3`
        # against `0.39`), `4.6e−2` and `7.2e−3` — a third of a cell at
        # `h = 5/256` even at lmax 8.
        @test errs[("KerrSchild", 4)] ≤ 1.5e-3
        @test errs[("KerrSchild", 8)] ≤ 1.0e-5
        @test errs[("Harmonic", 4)] ≤ 5.0e-2
        @test errs[("Harmonic", 8)] ≤ 8.0e-3
        @info "shape truncation, max over directions (step 8d)" errs
    end

    # The guard's reason to exist is that a query *outside* the layer can
    # read *inside* it; on a surface that is not a sphere the per-axis
    # nearest lattice point is not the one that decides, so the guard is
    # checked against the definition — every lattice point classified.
    @testset "the footprint guard is exact on a surface that is not a sphere" begin
        T = Float64
        bg = SM.KerrSchild(1.0, 0.9)
        int = FittedInterior(T; center=(0.1, -0.2, 0.05),
                             shape=analytic_shape(bg, 4), lmax=4,
                             offset=0.2, thickness=0.3)
        m = interior_mask(int, zero(T))
        rng = MersenneTwister(11)
        h = 0.05
        n = 4
        disagree = 0
        nrefused = 0
        for _ in 1:2000
            x0 = (0.1 + (2 * rand(rng) - 1) * 1.8, -0.2 + (2 * rand(rng) - 1) * 1.8,
                  0.05 + (2 * rand(rng) - 1) * 1.8)
            want = all(is_evolved(m, (x0[1] + i * h, x0[2] + j * h, x0[3] + k * h))
                       for i in 0:(n - 1), j in 0:(n - 1), k in 0:(n - 1))
            got = TreeGeneralizedHarmonic.footprint_evolved(m, x0, h, Val(n))
            disagree += got != want
            nrefused += !got
        end
        @test disagree == 0
        @test 0 < nrefused < 2000
    end

    # A margin is worth what it attenuates, not what it measures; the
    # function the driver records it with must be the script's.
    @testset "the margin's e-folds are step 8a's" begin
        T = Float64
        R = 2.0
        sphere = FittedInterior(T; center=(0, 0, 0),
                                shape=[R * sqrt(4π); zeros(24)], lmax=4,
                                offset=8 * T(5 // 64), thickness=8 * T(5 // 64))
        ks = SM.KerrSchild(1.0, 0.0)
        e2 = margin_efolds(ks, sphere, 2; t=0.0, ε_KO=0.5, spacing=5 / 64, per=32)
        e4 = margin_efolds(ks, sphere, 4; t=0.0, ε_KO=0.5, spacing=5 / 64, per=32)
        # `CODE.md`'s table (test/dispersion.jl, section 1c): 1.81 and 2.07
        # e-folds across eight cells of `h = 5/64` at `q = 2` and `4`.
        @test e2.min ≈ 1.81 atol = 0.005
        @test e4.min ≈ 2.07 atol = 0.005
        # ε enters linearly, and no dissipation buys nothing.
        e2b = margin_efolds(ks, sphere, 2; t=0.0, ε_KO=1.0, spacing=5 / 64, per=32)
        @test e2b.min ≈ 2 * e2.min rtol = 1e-12
        @test margin_efolds(ks, sphere, 2; t=0.0, ε_KO=0.0, spacing=5 / 64).min == 0
    end

    # The two cases a tracked geometry must refuse by name: a case with
    # nothing to track it with, and a mesh too coarse for its margin.
    @testset "what a tracked case cannot mean is refused" begin
        T = Float64
        @test_throws "takes no r_0" kerr_schild_case(
            T; halfwidth=2, chunk=T(1 // 10), interior=FittedSpec(T), r_0=0.4,
            r_1=1.2)
        @test_throws "needs its two radii" kerr_schild_case(
            T; halfwidth=2, chunk=T(1 // 10), interior=:damped)
        bare = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10),
                                interior=FittedSpec(T))
        forest = hole_fixture_forest(T, bare; N=8)
        @test_throws "carries no horizon finder" evolve!(
            T, bare; forest=forest, q=2,
            ops=Operators(prolongation=4, restriction=4), t_end=T(1 // 10))
        fs = FieldSet{T}(forest, 20; G=2, centering=vertexcentered(3))
        @test_throws "FittedSpec" GHProblem(fs, GhostSchedule(fs,
                                                                Operators(prolongation=4,
                                                                          restriction=4)),
                                            bare; q=2)
        @test_throws "FittedSpec" state_callback(bare, 0)
        # `m = 8` on the fixture's mesh: at `h = 5/64` the offset surface is
        # at `1.375`, outside the level-3 cube, so the layer's blocks are
        # `h = 5/32`, where margin, ramp and core need `18 h = 2.81` of a
        # horizon of radius `2`.
        tr = seed_track(bare, 0)
        @test_throws "cannot hold the layer" fitted_interior(bare.interior, tr,
                                                             forest, 2; t=0,
                                                             n_L=8)
        ok = fitted_interior(fitted_fixture(T).interior, tr, forest, 2; t=0,
                             n_L=8)
        @test ok.h == T(5 // 64)
        @test ok.offset == 10 * T(5 // 64)
        @test check_interior_radii(forest, ok, bare.background, 2).h == T(5 // 64)
        @test layer_cells(2, 4, 1) == 8
        @test layer_cells(3, 4, 1) == 12
    end

    # A floor read from the analytic radii around a tracked layer would be a
    # statement about a surface nobody measured; and a floor below the
    # geometry's own level would let the layer's blocks coarsen until
    # `check_interior_radii` fires at the next chunk.
    @testset "the level floor reads the tracked horizon" begin
        T = Float64
        ref = Refinement(T; refine_tol=T(2 // 5), coarsen_tol=T(1 // 10),
                         maxlevel_cap=3, floor_margin=zero(T), ceiling_cells=1)
        case = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10),
                                interior=FittedSpec(T; margin=10),
                                refinement=ref)
        forest = hole_fixture_forest(T, case; N=8)
        geom = fitted_interior(case.interior, seed_track(case, 0), forest, 2;
                               t=0, n_L=8)
        @test geometry_radii(geom, case.background) == (geom.r_in, geom.r_out)
        @test layer_radii(geom) == ((geom.r_in - geom.offset) - geom.thickness,
                                    geom.r_in - geom.offset)
        @test horizon_floor_level(forest, geom, case.background, 2) == 3
        lb = level_bounds(case, forest, 0, 2; interior=geom)
        @test lb.floor_level == 3
        # From the core surface, not the offset surface (amended in step 8:
        # the geometry's spacing is read over the whole layer, and a floor
        # that left its inner blocks free let them coarsen).
        @test lb.floor_lo == layer_radii(geom)[1]
        @test lb.floor_hi == geom.r_out
        @test_throws "fitted_interior" level_bounds(case, forest, 0, 2)
    end
end

@testset verbose = true "The tracked geometry on a mesh" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    fixture = hole_fixture(T; q=q)
    forest = hole_fixture_forest(T, fixture; N=8)
    fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3), backend=CPU())
    sched = GhostSchedule(fs, ops)
    fill_exact!(fs, fixture, zero(T))
    u0 = statevector(fs)
    gather!(u0, fs)

    # `PLAN.md`: "the kernels are bit-identical between an `Interior` and a
    # `FittedInterior` holding the same sphere with the same profiles". It
    # is the claim every comparison of the two geometries rests on: if it
    # were a tolerance, a difference between a tracked run and an analytic
    # one could be the arithmetic and not the geometry.
    @testset "a fitted sphere is step 5's layer, bit for bit (:$variant)" for variant in
                                                                               (:damped,
                                                                                :frozen,
                                                                                :pasted)
        case = hole_fixture(T; q=q, variant=variant)
        int = with_ρ_max(case.interior, T(37 // 5))
        p = GHProblem(fs, sched, case; q=q, interior=int)
        # The monopole whose series is `R = 2` *exactly*: the series is
        # `a_00 q_00`, so the coefficient is `R / q_00` or the float next to
        # it, and the test says which rather than trusting the division.
        R = T(2)
        q00 = inv(sqrt(4 * T(π)))
        a00 = R / q00
        for k in (1, -1, 2, -2)
            a00 * q00 == R && break
            a00 = k > 0 ? nextfloat(R / q00, k) : prevfloat(R / q00, -k)
        end
        @test a00 * q00 == R
        fit = FittedInterior(T; center=case.center, shape=[a00; zeros(T, 24)],
                             lmax=4, offset=R - int.r_1,
                             thickness=int.r_1 - int.r_0, ρ_max=int.ρ_max,
                             variant=variant, w_ramp=int.w_ramp,
                             ρ_ramp=int.ρ_ramp, margin=int.margin)
        # The preconditions, stated: the two surfaces are the sphere's radii
        # to the last bit.
        @test layer_radii(fit) === (int.r_0, int.r_1)
        u = copy(u0)
        for i in eachindex(u)
            u[i] += T(1 // 1000) * sin(T(i))
        end
        du_s = similar(u)
        du_f = similar(u)
        gh_rhs!(du_s, u, p, zero(T))
        gh_rhs!(du_f, u, with_interior(p, fit), zero(T))
        @test isequal(du_s, du_f)
        # The paste writes the same bits, and the masks count the same points.
        us = copy(u)
        uf = copy(u)
        paste_interior!(p, us, zero(T))
        paste_interior!(with_interior(p, fit), uf, zero(T))
        @test isequal(us, uf)
        xs = [coordinates(fs, b, (i + G, j + G, k + G))
              for b in 1:nblocks(fs) for k in 1:8 for j in 1:8 for i in 1:8]
        @test all(x -> in_layer(int, 0, x) == in_layer(fit, 0, x), xs)
        @test all(x -> is_evolved(interior_mask(int, 0), x) ==
                       is_evolved(interior_mask(fit, 0), x), xs)
        @test all(x -> core_position(int, 0, x) === core_position(fit, 0, x), xs)
        @test count(x -> in_layer(fit, 0, x), xs) > 0
    end

    # The shape evaluation is a series at every layer point of every
    # evaluation; `CODE.md` prices the layer's `u_exact` and this is the
    # second price. Recorded, not asserted — a cost, not a claim.
    @testset "the shape evaluation's cost is recorded" begin
        bg = SM.KerrSchild(1.0, 0.9)
        rng = MersenneTwister(5)
        dirs = [unit(SVector{3}(randn(rng, 3))) for _ in 1:2000]
        xs = [Tuple(1.2 * n) for n in dirs]
        ns = Dict{String,Float64}()
        for L in (4, 8)
            sv = SVector{(L + 1)^2}(analytic_shape(bg, L))
            t, s = ns_per_point(n -> shape_series(sv, L, n), dirs, 20)
            ns["shape_series, lmax $L"] = t
            @test isfinite(s)
        end
        t, s = ns_per_point(x -> background_state(bg, 0.0, x)[1][1], xs, 5)
        ns["background_state (u_exact)"] = t
        @test isfinite(s)
        @info "the tracked geometry's cost per point, ns (step 8d)" ns
    end

    # `PLAN.md`: "the tracked geometry of the static Kerr-Schild hole agrees
    # with the analytic one to interpolation accuracy after one find". The
    # seed is the analytic answer; one find replaces it with the found one,
    # which must be the same hole to the finder's accuracy — or every layer
    # the track places is displaced by the finder's error.
    @testset "one find of the static hole is the analytic hole" begin
        case = fitted_fixture(T)
        tr0 = seed_track(case, 0)
        @test tr0.source === :analytic
        @test tr0.r_min == horizon_min_radius(case.background) == 2
        @test tr0.r_max == horizon_max_radius(case.background) == 2
        @test tr0.shape[1] ≈ 2 * sqrt(4π) rtol = 1e-14
        @test maximum(abs, tr0.shape[2:end]) < 1e-14
        @test center_at(track_center(tr0), 0.7) == SVector(0.0, 0.0, 0.0)
        p = GHProblem(fs, sched, fixture; q=q)
        hz = find_gh_horizon(p, u0, zero(T); N=12, spin=false)
        @test hz.success
        h = minimum_spacing(T, forest)
        tr1 = update_track(tr0, hz, zero(T); G=G, h=h)
        @test tr1.source === :found
        @test tr1.nfinds == 1
        @test tr1.v_est == tr0.v_est              # one find is no velocity
        offset = sqrt(sum(abs2, center_at(track_center(tr1), zero(T)))) / h
        # Measured: 1.8e−4 cells, the finder's origin at 1.4e−5 M.
        @test offset < 1e-2
        @test tr1.r_min ≈ 2 atol = 1e-3
        @test tr1.r_max ≈ 2 atol = 1e-3
        @test tr1.r_min ≤ tr1.r_max
        # The radii about the found origin and about the analytic center are
        # the same surface, a fraction of the offset apart.
        @test abs(hz.origin_r_min - hz.r_min) ≤ 2 * hz.center_offset
        geom = fitted_interior(case.interior, tr1, forest, G; t=0, n_L=8)
        @test geom.r_in ≈ 2 atol = 1e-3
        @test geom.r_out ≈ 2 atol = 1e-3
        @test check_interior_radii(forest, geom, case.background, G;
                                   center=case.center).singular < 1e-3
        @info "one find on the fixture's initial data (step 8d)" offset_cells = offset r_min = tr1.r_min r_max = tr1.r_max origin = hz.origin
        # Refused: a find whose smallest radius moved by a stencil reach.
        # `PLAN.md`: "a find whose r_min is perturbed by G h is refused".
        bad = merge(hz, (origin_r_min=tr1.r_min + G * h,))
        @test_throws "G h/2" update_track(tr1, bad, T(1 // 10); G=G, h=h)
        # Half of that is accepted: the refusal is a threshold, not a veto.
        fine = merge(hz, (origin_r_min=tr1.r_min + G * h / 4,))
        @test update_track(tr1, fine, T(1 // 10); G=G, h=h).source === :found
        # A second find differences the first: the velocity of a static hole
        # is the finder's noise over the interval.
        tr2 = update_track(tr1, hz, T(1 // 10); G=G, h=h)
        @test tr2.nfinds == 2
        @test all(iszero, tr2.v_est)             # the same find, so no motion
        # A miss coasts; `max_misses` of them are a lost track, and the
        # message says how old the geometry is.
        miss = (success=false, note="injected")
        tr3 = update_track(tr2, miss, T(2 // 10); G=G, h=h, max_misses=2)
        @test tr3.source === :coasting
        @test tr3.misses == 1
        @test tr3.c_find == tr2.c_find
        err = try
            update_track(tr3, miss, T(3 // 10); G=G, h=h, max_misses=2)
        catch e
            e
        end
        @test err isa TrackLostError
        @test occursin("missed 2 consecutive finds", err.msg)
        @test occursin("before t = 0.3", err.msg)
        @test occursin("injected", err.msg)
        @test update_track(tr3, (success=nothing,), T(3 // 10); G=G, h=h) === tr3
        # The gauge source is re-sampled only once the core moved: two
        # geometries a tenth of a cell apart are a tenth of a cell apart.
        moved = fitted_interior(case.interior,
                                HorizonTrack{T}(tr1.t_find,
                                                tr1.c_find + SVector(h / 10, 0, 0),
                                                tr1.v_est, tr1.r_min, tr1.r_max,
                                                tr1.shape, tr1.lmax, tr1.hlm,
                                                tr1.grid, :found, 0, 1),
                                forest, G; t=0, n_L=8)
        @test surface_shift(geom, geom, 0) == 0
        @test surface_shift(geom, moved, 0) ≈ h / 10 rtol = 1e-10
    end
end

@testset verbose = true "The tracked hole" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)

    # `PLAN.md`: "a run whose finder is disabled after 0.1 M coasts and
    # records :coasting" — and a lost track ends the run *with* its record.
    # The finder is disabled by the `find` keyword, a wrapper that throws
    # after `t = 1/10` (proposed in step 8d: the observer is called after
    # the find and cannot reach it). The cadence is `every = 100`, so every
    # find after the first is the lapse-collapse trigger's: `α_trigger = 1`
    # lies above every lapse in Kerr-Schild, and each row forces the next
    # row's find — which is how the trigger is checked in the same run.
    @testset "a track coasts through a disabled finder, and is lost" begin
        case = fitted_fixture(T; chunk=T(1 // 20), every=100, N_ah=10,
                              max_misses=2, α_trigger=one(T))
        disabled(p, u, t; kw...) =
            t > T(1 // 10) ? throw(ErrorException("the finder is disabled")) :
            find_gh_horizon(p, u, t; kw...)
        forest = hole_fixture_forest(T, case; N=8)
        err = try
            evolve!(T, case; forest=forest, q=q, ops=ops, t_end=T(2 // 5),
                    find=disabled)
        catch e
            e
        end
        @test err isa TrackLostError
        recs = err.records
        @test length(recs) == 5                  # t = 0, 1/20, …, 1/5
        # The trigger: the cadence asks for the first find only, and the
        # rows after it found anyway, saying why.
        @test recs[1].track_trigger === false
        @test recs[1].horizon_success === true
        @test all(r -> r.track_trigger === true, recs[2:end])
        @test all(r -> r.min_α_evolved < 1, recs)
        @test recs[2].horizon_success === true && recs[3].horizon_success === true
        # The coast: the first miss keeps the geometry of `t = 1/10`, the
        # second ends the run with the record of both.
        @test [r.track_source for r in recs] ==
              [:found, :found, :found, :coasting, :coasting]
        @test [r.track_misses for r in recs] == [0, 0, 0, 1, 2]
        @test recs[4].horizon_success === false
        @test occursin("disabled", recs[4].horizon_note)
        # Coasting is the prediction: the last find's center carried along
        # its velocity for the chunk, and nothing else.
        @test all(k -> isapprox(recs[4].track_center[k],
                                recs[3].track_center[k] +
                                recs[3].track_velocity[k] / 20; atol=1e-14), 1:3)
        @test recs[4].track_velocity == recs[3].track_velocity
        @test occursin("0.1 M before t = 0.2", err.msg)
        @test all(r -> r.track_offset < 1, recs)
        @test all(r -> r.finite, recs)
    end

    # `PLAN.md`: a tracked run of the fixture to `0.15 M` at the analytic
    # geometry's level. The comparison is against step 5's sphere with the
    # *same* layer — the offset surface's `r_1 = 2 − 10 h`, the ramp `n_L = 8`
    # below it, step 8c's ramps — so the difference is the tracking and
    # nothing else.
    @testset "a tracked hole is the analytic one, to the tracking" begin
        # The run is shared with `fit_tests.jl` (step 8e), which fits the
        # state it ends in; nothing below writes the state.
        (; case, out) = tracked_fixture_run()
        h = T(5 // 64)
        sphere = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10),
                                  r_0=2 - 18h, r_1=2 - 10h, margin=10,
                                  w_ramp=T(1 // 2), ρ_ramp=one(T),
                                  bounds=default_bounds(T; M=1,
                                                        r_gate=T(9 // 10)))
        ref = evolve!(T, sphere; forest=hole_fixture_forest(T, sphere; N=8),
                      q=q, ops=ops, t_end=T(3 // 20))
        rf, rs = out.records[end], ref.records[end]
        sf = gh_outside_shell_norms(out)
        ss = gh_outside_shell_norms(ref)
        # Every row: found, a thousandth of a cell from the analytic center,
        # the projection never firing, and the geometry built by the rule.
        @test all(r -> r.track_source === :found, out.records)
        @test all(r -> r.track_offset < 1, out.records)
        @test all(r -> r.bounds_hits == 0, out.records)
        @test all(r -> r.horizon_success === true, out.records)
        @test all(r -> r.layer_h == h && r.layer_offset == 10h &&
                       r.layer_thickness == 8h, out.records)
        @test out.n_L == 8
        @test out.nresamples == 0
        @test ref.records[end].bounds_hits == 0
        # Measured (step 8d): the masked error `3.11027e−3` for both, the
        # shell `C_a` `1.074546e−2` against `1.074545e−2`, the residual
        # `0.203` against `0.206`, and the track `3.3e−4` cells off — so the
        # factor asserted is a thousandth, where the measured one is 10⁻⁶.
        @test rf.err_l2 ≈ rs.err_l2 rtol = 1e-3
        @test sf.gauge_l2 ≈ ss.gauge_l2 rtol = 1e-3
        @test sf.err_l2 ≈ ss.err_l2 rtol = 1e-3
        @test rf.residual ≈ rs.residual rtol = 5e-2
        @test sf.npoints == ss.npoints
        @test all(r -> r.margin_efolds > 1, out.records)
        @info "the tracked hole against the analytic one at t = 0.15 M (step 8d)" err_l2 = (rf.err_l2, rs.err_l2) shell_gauge_l2 = (sf.gauge_l2, ss.gauge_l2) residual = (rf.residual, rs.residual) track_offset = maximum(r -> r.track_offset, out.records) prediction = [r.track_prediction for r in out.records] margin_efolds = out.records[end].margin_efolds
    end
end
