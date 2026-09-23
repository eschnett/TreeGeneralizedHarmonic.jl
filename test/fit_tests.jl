# The fitted target: the fit's variables and harmonics, the least squares,
# the validity sweep, the evaluator, and the fit of an evolved state.
#
# `CODE.md`, "The interior" — "The fitted target (added in step 8e)" — and
# `PLAN.md`'s step 8e-i. Everything here is host-side and against analytic
# data, except the last claim, which fits the state the suite's one tracked
# run ends in (`tracked_fixture_run`, in `evolution_cases.jl`, shared with
# `tracking_tests.jl` so that the suite pays for the run once). No
# right-hand side is evaluated: the kernel and the `:fitted` variant are
# step 8e-ii's.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using KernelAbstractions: CPU
using LinearAlgebra: Symmetric, cond, eigvals, qr
using Random: MersenneTwister
using StaticArrays: SMatrix, SVector
import AbstractSphericalHarmonics as ASH
import SpacetimeMetrics as SM

unitvec(v) = v / sqrt(sum(abs2, v))

# The collocation points of a geometry at time `t`, and their rays.
function surface_points(int::FittedInterior{T}, L, t) where {T}
    c = center_at(int.center, T(t))
    ns = [SVector{3,T}(n) for n in fit_directions(L)]
    xs = [c + (shape_radius(int, n) - int.offset) * n for n in ns]
    return xs, ns
end

# The largest error of the fit's state against the analytic one, over the
# collocation points and `nrand` random directions on the offset surface,
# relative to the largest component there.
function surface_error(fit, int::FittedInterior{T}, bg, t; nrand=200) where {T}
    xs, _ = surface_points(int, fit.params.lmax, t)
    rng = MersenneTwister(8)
    c = center_at(int.center, T(t))
    for _ in 1:nrand
        n = SVector{3,T}(unitvec(randn(rng, 3)))
        push!(xs, c + (shape_radius(int, n) - int.offset) * n)
    end
    err = 0.0
    scale = 0.0
    for x in xs
        ex = SVector{20}(state_tuple(bg, t, Tuple(x)))
        h, Π = fit_state(fit, x, t)
        err = max(err, maximum(abs, vcat(h, Π) - ex))
        scale = max(scale, maximum(abs, ex))
    end
    return err / scale
end

# The ansatz of `InteriorFit` along one ray, written out from the solid
# harmonics: `(f, ρ ∂_ρ f, ρ² ∂_ρ² f)` at `ξ` for the coefficients `C`.
function ansatz_along(C, L, cont, ξ)
    S = real_solid_harmonics(L, ξ)
    ρ² = sum(abs2, ξ)
    f = zeros(size(C, 3))
    f1 = zeros(size(C, 3))
    f2 = zeros(size(C, 3))
    for slot in 1:((L + 1)^2), k in 0:cont
        n = isqrt(slot - 1) + 2k
        w = S[slot] * ρ²^k
        f .+= w .* C[slot, k + 1, :]
        f1 .+= (n * w) .* C[slot, k + 1, :]
        f2 .+= (n * (n - 1) * w) .* C[slot, k + 1, :]
    end
    return f, f1, f2
end

@testset verbose = true "The fitted target" begin
    # The fit shares 8d's real harmonics so that its evaluator is the shape's
    # recurrence; a sign or a √2 off is a target rotated by a mode, and a
    # least squares that does not invert its own model is a target that is
    # not the fit it reports. Both at `Float64` and at `Float32`.
    @testset "the fit's harmonics are 8d's, and its least squares inverts them ($T)" for T in
                                                                                        (Float64,
                                                                                         Float32)
        rng = MersenneTwister(20260923)
        tol = 32 * eps(T)
        # Solid harmonics at unit vectors are `shape_series`'s harmonics, and
        # scale as `ρ^l`; at the center only `ỹ_00` survives.
        for L in (4, 8)
            worst = 0.0
            for _ in 1:20
                n = SVector{3,T}(unitvec(randn(rng, 3)))
                S = real_solid_harmonics(L, n)
                S2 = real_solid_harmonics(L, 2n)
                for slot in 1:((L + 1)^2)
                    e = SVector{(L + 1)^2,T}(ntuple(i -> i == slot ? one(T) : zero(T),
                                                    (L + 1)^2))
                    worst = max(worst, abs(S[slot] - shape_series(e, L, n)),
                                abs(S2[slot] / 2^isqrt(slot - 1) - S[slot]))
                end
            end
            # Measured: 1.8 and 14 eps (Float64), 1.3 and 16 eps (Float32)
            # at L = 4 and 8.
            @test worst ≤ tol
        end
        S0 = real_solid_harmonics(8, zero(SVector{3,T}))
        @test S0[1] ≈ inv(sqrt(4 * T(π)))
        @test all(iszero, S0[2:end])

        # The angular round trip: a real field synthesised from random
        # coefficients by `ash_evaluate` (an independent transform) is
        # recovered by least squares on the equiangular grid, to roundoff
        # times the conditioning of the harmonic matrix.
        conds = Float64[]
        for L in (4, 8)
            a = randn(rng, (L + 1)^2)
            f = real.(ASH.ash_evaluate(ASH.EquiangularGrid(L), complex_from_real(a, L), 0))
            dirs = fit_directions(L)
            A = reduce(vcat, [permutedims(real_solid_harmonics(L, SVector{3,T}(n)))
                              for n in dirs])
            # `fit_directions` is the grid's `CartesianIndices` order, which
            # is `vec`'s.
            b = T.(vec(f))
            @test length(b) == length(dirs)
            â = qr(A) \ b
            κ = cond(Float64.(A))
            push!(conds, κ)
            @test maximum(abs, â .- T.(a)) ≤ 8 * eps(T) * κ * maximum(abs, a)
        end
        # Measured: cond 2.2 at L = 4 and 2.9 at L = 8, and the coefficients
        # back to 2.9–4.6 eps — the equiangular grid is a well-posed
        # collocation of the real harmonics.
        @test all(<(10), conds)

        # The whole ansatz: random coefficients (the shift's constant zero)
        # on a surface that is not a sphere, their values and radial
        # derivatives written out, and `solve_fit` recovers them — the
        # column order, the shift's leading block of the one QR, and the
        # coefficient layout, to roundoff times the conditioning.
        for (L, cont) in ((4, 1), (4, 2), (6, 2))
            nb = cont + 1
            C = randn(rng, (L + 1)^2, nb, 20)
            C[1, 1, 2:4] .= 0
            dirs = fit_directions(L)
            ρs = [1 + (n[3]^2 - 1 / 3) / 5 + n[1] / 10 for n in dirs]
            ξs = [SVector{3,T}(ρ * n) for (ρ, n) in zip(ρs, dirs)]
            rows = [ansatz_along(C, L, cont, Float64.(ξ)) for ξ in ξs]
            rbar = T(3 // 2)
            samples = ntuple(b -> [SVector{20,T}(rows[p][b] ./ (rbar^(b - 1) * ρs[p]^(b - 1)))
                                   for p in eachindex(ξs)], nb)
            coeffs, res, cnd, model = solve_fit(ξs, samples, L, cont, rbar)
            # Measured: κ = 99, 1.5e3, 6.0e3 for the three rows, and the
            # coefficients back to 0.03–0.19 of `κ eps |C|`.
            κ = max(cnd.scalars, cnd.shift)
            @test maximum(abs, coeffs .- T.(C)) ≤ 64 * eps(T) * κ * maximum(abs, C)
            @test all(iszero, coeffs[1, 1, 2:4])
            @test res.overall ≤ 64 * eps(T) * κ
            @test all(p -> maximum(abs, model[p] - samples[1][p]) ≤
                           64 * eps(T) * κ * maximum(abs, C), eachindex(ξs))
            # The written-out derivative is the evaluator's, differenced along
            # the ray (checked once, at `Float64`).
            if T === Float64 && L == 4 && cont == 2
                params = FitParams{T}(L, cont, rbar, HoleCenter(T, (0, 0, 0)),
                                      default_bounds(T; M=1, r_gate=1))
                p = 7
                n = SVector{3}(dirs[p])
                δ = 1e-3
                g(s) = fit_variables_at(params, coeffs, rbar * (ρs[p] + s) * n, 0.0)
                d1 = ((g(-2δ) - g(2δ)) + 8 * (g(δ) - g(-δ))) / (12δ)
                @test maximum(abs, d1 .- samples[2][p] .* rbar) ≤
                      1e-8 * maximum(abs, samples[2][p] .* rbar)
            end
        end
        @test FittedSpec(T).lmax_fit == 8
        @test_throws "lmax_fit" FittedSpec(T; lmax_fit=0)
    end

    # `PLAN.md`: "the cont = 2 fit of analytic Kerr-Schild data on r_1
    # reproduces it there to L truncation and to interpolation order". A fit
    # that does not reproduce the state it was fitted to is a target with an
    # error of its own at the layer's outer edge — which step 8c measured the
    # exterior to tolerate at 20 %, but which should be the data's error and
    # not the fit's.
    @testset "the static hole's fit is the hole: its truncation and its interpolation order" begin
        T = Float64
        q = 2
        G = q ÷ 2 + 1
        case = fitted_fixture(T)
        bg = case.background
        spec = case.interior
        bd = case.bounds
        forest = hole_fixture_forest(T, case; N=8)
        geom = fitted_interior(spec, seed_track(case, 0), forest, G; t=0, n_L=8)
        @test geom.h == T(5 // 64)
        analytic = analytic_sampler(bg, zero(T); δ=geom.h / 8)

        # Kerr-Schild `a = 0` on a sphere has `l ≤ 2` structure in the fit's
        # variables (`γ_ij − δ_ij ∝ n_i n_j`, `β^i ∝ n^i`, `α(r)`, and `Π`
        # likewise), so `L = 2` is where the truncation vanishes; `L = 1`
        # cannot hold the quadrupole.
        errs = Dict{Tuple{Int,Int},Float64}()
        for L in (1, 2, 4, 8), cont in (1, 2)
            fit = build_fit(analytic, geom, spec; cont=cont, bounds=bd, L=L)
            errs[(L, cont)] = surface_error(fit, geom, bg, zero(T))
        end
        @test errs[(1, 1)] > 0.1 && errs[(1, 2)] > 0.1
        # Measured: 1.10 at L = 1; 5.1e−15 … 3.1e−13 at L = 2, 4, 8, the
        # curvature rows' second-difference roundoff being the largest.
        @test all(errs[(L, cont)] ≤ 1e-11 for L in (2, 4, 8), cont in (1, 2))

        # The analytic sampler's derivatives, and the chain rule against the
        # converted samples differenced along the same rays: two routes to
        # the fit variables' slopes and curvatures that must agree to the
        # stencil's `δ⁴`.
        xs, ns = surface_points(geom, 8, zero(T))
        u0, u1, u2 = analytic(xs, ns)
        δ = analytic.δ
        worst_h = 0.0
        worst_1 = 0.0
        worst_2 = 0.0
        for i in eachindex(xs)
            _, _, ∂h = background_state(bg, zero(T), Tuple(xs[i]))
            exact = ns[i][1] * ∂h[1] + ns[i][2] * ∂h[2] + ns[i][3] * ∂h[3]
            worst_h = max(worst_h, maximum(abs, u1[i][1:10] - exact) /
                                   maximum(abs, exact))
            v, v1, v2 = fit_variables(u0[i], u1[i], u2[i])
            vk = [fit_variables(SVector{20}(state_tuple(bg, zero(T),
                                                        Tuple(xs[i] + k * δ * ns[i]))))
                  for k in -2:2]
            d1 = ((vk[1] - vk[5]) + 8 * (vk[4] - vk[2])) / (12δ)
            d2 = (16 * (vk[2] + vk[4]) - (vk[1] + vk[5]) - 30 * vk[3]) / (12δ^2)
            worst_1 = max(worst_1, maximum(abs, v1 - d1) / maximum(abs, v1))
            worst_2 = max(worst_2, maximum(abs, v2 - d2) / maximum(abs, v2))
        end
        # Measured: 1.6e−8 for ∂_r h against the analytic gradient (the
        # stencil's `δ⁴`, `δ = h/8`); the chain rule and the differenced
        # variables agree to 5.7e−10 (slope) and 7.3e−11 (curvature).
        @test worst_h ≤ 1e-7
        @test worst_1 ≤ 1e-7
        @test worst_2 ≤ 1e-6

        # The state sampler on a field set holding the exact solution: the
        # fit reproduces the interpolated data, so its error on the surface
        # is the interpolation's, `O(h^{q+2})` — on the *same* surface at
        # both resolutions (the N = 8 geometry, whose core rule both field
        # sets are filled with).
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        state_errs = Float64[]
        sample_errs = Float64[]
        for N in (8, 16)
            f = hole_fixture_forest(T, case; N=N)
            fs = FieldSet{T}(f, 20; G=G, centering=vertexcentered(3), backend=CPU())
            fill_exact!(fs, case, zero(T); interior=geom)
            fill_ghosts!(fs, GhostSchedule(fs, ops); boundary=dirichlet(case, zero(T)))
            sampler = state_sampler(fs, q; t=zero(T))
            fit = build_fit(sampler, geom, spec; cont=1, bounds=bd)
            @test fit.valid
            @test fit.sweep.hits == 0
            push!(state_errs, surface_error(fit, geom, bg, zero(T)))
            us, _ = sampler(xs, ns)
            push!(sample_errs,
                  maximum(i -> maximum(abs, us[i] - u0[i]), eachindex(xs)) /
                  maximum(u -> maximum(abs, u), u0))
            N == 8 && @test_throws "serves cont = 1" build_fit(sampler, geom, spec;
                                                                cont=2, bounds=bd)
        end
        rate = log2(state_errs[1] / state_errs[2])
        # Measured: 2.2e−4 and 8.3e−6 (a rate of 4.65; 3.9 from N = 16 to
        # 32, measured outside the suite), against samples off by 2.6e−4
        # and 9.2e−6: the fit adds nothing to the interpolant's error.
        @test state_errs[1] ≤ 5e-4
        @test rate ≥ 3.5
        @test all(state_errs .≤ 1.5 .* sample_errs)
        @info "the static hole's fit on its offset surface (step 8e)" analytic = errs state = state_errs samples = sample_errs rate chain_rule = (worst_1, worst_2) sampler_gradient = worst_h
    end

    # `PLAN.md`'s finding 2 and the acceptance list: the fit is a valid
    # metric at every swept point for `a = 0` and `a = 9/10`, and the
    # angular mean of `g_ab` — the fit made in the wrong variables — is not
    # a metric at all. A fit that passed a sweep of `g_ab` values would be a
    # fit of a Euclidean metric.
    @testset "the fits are metrics where the mean of g_ab is not" begin
        T = Float64
        bd = default_bounds(T; M=1, r_gate=T(9 // 10))
        # The control: Kerr-Schild `a = 0`'s `g_ab` averaged over the sphere
        # `r = 1.15 M` (quadrature exact to degree 16) has `g_tt = +0.74`
        # and four positive eigenvalues: Euclidean signature, a negative
        # `α²`. The same average of the fit's variables is a metric.
        ks = SM.KerrSchild(one(T), zero(T))
        grid = ASH.EquiangularGrid(16)
        gmean = zero(SMatrix{4,4,T})
        vmean = zero(SVector{20,T})
        wsum = zero(T)
        for ij in CartesianIndices(ASH.ash_grid_size(grid))
            θ, φ = ASH.ash_point_coord(grid, ij)
            dθ, dφ = ASH.ash_point_delta(grid, ij)
            w = sin(θ) * dθ * dφ
            x = T(23 // 20) * SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
            gmean += w * SM.metric(ks, SVector(zero(T), x...))
            vmean += w * fit_variables(SVector{20}(state_tuple(ks, zero(T), Tuple(x))))
            wsum += w
        end
        gmean /= wsum
        vmean /= wsum
        @test wsum ≈ 4π rtol = 1e-12
        @test gmean[1, 1] ≈ -1 + 2 / T(23 // 20) rtol = 1e-12     # +0.739
        @test all(>(0), eigvals(Symmetric(Matrix(gmean))))
        _, αg, _, _ = state_validity(pack_g(gmean), zero(SVector{10,T}))
        @test αg < 0                                          # −0.86
        hv, Πv = state_from_fit(vmean)
        detγ, αv, _, _ = state_validity(hv, Πv)
        @test detγ > 0 && αv > 0                              # 3.94, 0.60

        # Kerr-Schild `a = 0` on the fixture's tracked geometry, both orders.
        case = fitted_fixture(T)
        forest = hole_fixture_forest(T, case; N=8)
        geom = fitted_interior(case.interior, seed_track(case, 0), forest, 2;
                               t=0, n_L=8)
        for cont in (1, 2)
            fit = build_fit(analytic_sampler(case.background, zero(T); δ=geom.h / 8),
                            geom, case.interior; cont=cont, bounds=bd)
            @test fit.valid
            @test fit.sweep.npoints == 8 * 153 + 1
            @test fit.sweep.hits == 0
            # Measured: min λ(γ) = 1 (the transverse eigenvalue), min α =
            # 0.527 (cont = 1) and 0.480 (cont = 2), at the center.
            @test fit.sweep.min_λ ≈ 1 rtol = 1e-9
            @test fit.sweep.min_α > 0.4
        end

        # Kerr-Schild `a = 9/10` on its own oblate offset surface — `h =
        # 5/128`, `m = 8`, `n_L = 8`, the shape to `lmax_shape = 4` — which
        # has every even multipole: valid at both orders, with the
        # truncation falling with `L`.
        ks9 = SM.KerrSchild(one(T), T(9 // 10))
        h = T(5 // 128)
        spec9 = FittedSpec(T; lmax_shape=4)
        int9 = FittedInterior(T; center=(0, 0, 0), shape=analytic_shape(ks9, 4),
                              lmax=4, offset=8h, thickness=8h, h=h, margin=8,
                              n_L=8)
        trunc = Dict{Tuple{Int,Int},Float64}()
        for L in (4, 8, 12), cont in (1, 2)
            fit = build_fit(analytic_sampler(ks9, zero(T); δ=h / 8), int9, spec9;
                            cont=cont, bounds=bd, L=L)
            @test fit.valid
            @test fit.sweep.hits == 0
            trunc[(L, cont)] = surface_error(fit, int9, ks9, zero(T))
        end
        # Measured (cont = 1, 2): 0.26, 0.32 at L = 4; 0.051, 0.075 at
        # L = 8; 9.1e−3, 1.4e−2 at L = 12 — of the largest component on the
        # surface, the worst being `Π_tx` and `Π_ty`; every sweep valid,
        # `min λ(γ) ≥ 0.945`, no projection hit.
        @test trunc[(8, 1)] < trunc[(4, 1)] && trunc[(12, 1)] < trunc[(8, 1)]
        @test trunc[(8, 1)] ≤ 0.1
        @test trunc[(12, 2)] ≤ 0.03

        # The sweep is what catches a fit that is not a metric: harmonic Kerr
        # at `a = 9/10`, `h = 5/256`, `m = 4` — finding 3's configuration —
        # fitted to its curvature (`cont = 2`) has `γ` with negative
        # eigenvalues inside the surface, and the same fit to value and slope
        # does not. And a margin that puts the offset surface's equator
        # inside the singular disk (`m = 8`: `r_1 = 0.84 < 0.9`) is refused at
        # the sample, before any fit.
        ha = SM.Harmonic(one(T), T(9 // 10))
        hh = T(5 // 256)
        specH = FittedSpec(T; lmax_shape=12)
        harm(m) = FittedInterior(T; center=(0, 0, 0), shape=analytic_shape(ha, 12),
                                 lmax=12, offset=m * hh, thickness=8hh, h=hh,
                                 margin=m, n_L=8)
        sH = analytic_sampler(ha, zero(T); δ=hh / 8)
        @test_throws "not a valid metric" build_fit(sH, harm(4), specH; cont=2,
                                                    bounds=bd)
        bad = build_fit(sH, harm(4), specH; cont=2, bounds=bd, check=false)
        @test !bad.valid
        @test bad.sweep.min_λ < 0
        good = build_fit(sH, harm(4), specH; cont=1, bounds=bd)
        @test good.valid
        @test_throws "not finite" build_fit(sH, harm(8), specH; cont=1, bounds=bd)

        # A moving hole's shift has an `l = 0` part on its offset surface — the
        # boost's — which a shift without a constant term matches in value
        # through `ρ² ỹ_00` and cannot match in slope. `shift_constant =
        # true` fits it; both are metrics, and on the static hole the two
        # are the same fit to roundoff (above, the fixture's `β(0) = 0`).
        bh = SM.boost(SM.Harmonic(one(T), zero(T)), SVector{3,T}(T(3 // 10), 0, 0))
        hb = T(5 // 128)
        intB = FittedInterior(T; center=HoleCenter(T, (0, 0, 0), hole_velocity(bh)),
                              shape=analytic_shape(bh, 8), lmax=8, offset=8hb,
                              thickness=8hb, h=hb, margin=8, n_L=8)
        sB = analytic_sampler(bh, zero(T); δ=hb / 8)
        without = build_fit(sB, intB, specH; cont=1, bounds=bd)
        with = build_fit(sB, intB, specH; cont=1, bounds=bd, shift_constant=true)
        @test without.valid && with.valid
        eB0 = surface_error(without, intB, bh, zero(T))
        eB1 = surface_error(with, intB, bh, zero(T))
        # Measured: 1.1e−3 and 1.7e−5 on the surface; the shift's slope rows
        # 22 times their own scale without the constant, 1.4e−4 with it;
        # `|β(0)| = 0.29` with it, where the boost's shift is.
        @test eB0 > 5e-4
        @test eB1 < 1e-4
        @test maximum(without.residual.per_variable[2:4]) > 1
        @test maximum(with.residual.per_variable[2:4]) < 1e-3
        @test without.sweep.max_β_center == 0
        @test with.sweep.max_β_center > 0.1
        @info "the fits' validity and truncation (step 8e)" ks9_truncation = trunc harmonic_cont2 = bad.sweep harmonic_cont1 = good.sweep harmonic_residual = good.residual.value boosted = (eB0, eB1)
    end

    # The evaluator is what step 8e-ii's kernel will call: if it is not the
    # least-squares model at the points the model was fitted at, the target
    # the layer relaxes toward is not the fit the record reports. At the
    # center the ansatz makes the shift vanish, so the center is a metric
    # whenever `α` and `γ` are (finding 2).
    @testset "the evaluator is the model, and the center is a metric with β = 0 ($T)" for T in
                                                                                         (Float64,
                                                                                          Float32)
        case = fitted_fixture(T)
        forest = hole_fixture_forest(T, case; N=8)
        geom = fitted_interior(case.interior, seed_track(case, 0), forest, 2;
                               t=0, n_L=8)
        for cont in (1, 2)
            fit = build_fit(analytic_sampler(case.background, zero(T); δ=geom.h / 8),
                            geom, case.interior; cont=cont, bounds=case.bounds)
            @test fit.valid
            scale = maximum(v -> maximum(abs, v), fit.model)
            t = zero(T)
            worst = zero(T)
            same = true
            for (x, m) in zip(fit.points, fit.model)
                v = fit_variables_at(fit.params, fit.host, x, t)
                worst = max(worst, maximum(abs, v - m))
                # The projection is the identity on the fit, bit for bit.
                same &= isequal(fit_state(fit, x, t), state_from_fit(v))
            end
            # Measured: 11 and 28 eps (Float64), 10 and 25 eps (Float32) of
            # the largest variable at cont = 1, 2 — the design matrix's
            # product and the fold's sum, two spellings of one polynomial of
            # 162 and 243 terms.
            @test worst ≤ 64 * eps(T) * scale
            @test same
            c = center_at(geom.center, t)
            v0 = fit_variables_at(fit.params, fit.host, c, t)
            @test all(iszero, v0[2:4])
            h0, Π0 = fit_state(fit, c, t)
            @test all(iszero, h0[2:4])
            detγ, α, _, _ = state_validity(h0, Π0)
            @test detγ > 0 && α > 0
            @test fit.sweep.max_β_center == 0
            # An `isbits` kernel argument beside a coefficient array, and no
            # allocation: a kernel can call it (`CLAUDE.md`).
            @test isbits(fit.params)
            x = fit.points[1]
            fit_state(fit.params, fit.host, x, t)
            @test (@allocated fit_state(fit.params, fit.host, x, t)) == 0
        end
    end

    # `PLAN.md`: "the fit of the evolved state at the end of a :damped run
    # agrees with the fit of the analytic solution to the run's masked
    # error". The evolved state's fit is what 8e-ii's layer will relax
    # toward; if it differed from the analytic fit by more than the state
    # differs from the analytic solution, the fit would be adding an error
    # of its own.
    @testset "the evolved state's fit is the analytic fit, to the run's error" begin
        T = Float64
        (; case, q, out) = tracked_fixture_run()
        p = out.problem
        rec = out.records[end]
        t = T(rec.t)
        geom = out.geometry
        scatter!(p.U, out.u)
        fill_ghosts!(p.U, p.schedule; boundary=dirichlet(case, t))
        fe = build_fit(state_sampler(p.U, q; t=t), geom, case.interior; cont=1,
                       bounds=case.bounds)
        fa = build_fit(analytic_sampler(case.background, t; δ=geom.h / 8), geom,
                       case.interior; cont=1, bounds=case.bounds)
        @test fe.valid && fa.valid
        @test fe.sweep.hits == 0
        # The two fits' states over the sweep — the collocation surface and
        # eight radii inward along every ray, and the center — as the error
        # kernel measures a state: the Euclidean magnitude over the twenty
        # components.
        c = center_at(geom.center, t)
        xs, ns = surface_points(geom, fe.params.lmax, t)
        diff = zeros(9)
        for (x, n) in zip(xs, ns), s in 0:8
            y = c + (T(s) / 8) * (x - c)
            a = vcat(fit_state(fa, y, t)...)
            e = vcat(fit_state(fe, y, t)...)
            diff[s + 1] = max(diff[s + 1], sqrt(sum(abs2, e - a)))
        end
        us, _ = state_sampler(p.U, q; t=t)(xs, ns)
        sample = maximum(i -> sqrt(sum(abs2, us[i] -
                                          SVector{20}(state_tuple(case.background, t,
                                                                  Tuple(xs[i]))))),
                         eachindex(xs))
        # Measured at t = 3/20 M: the fits differ by 0.031 on the surface,
        # falling to 0.016 at the center, where the samples differ from the
        # truth by 0.034 and the run's masked error is 0.035 in L∞ (3.1e−3
        # in L2). The fit is a least-squares projection of the samples —
        # linear in them to first order — whose amplification on the surface
        # is measured at 0.92, and the inward extrapolation halves it; the
        # factor 2 is that amplification's headroom, not a tolerance on the
        # physics.
        @test maximum(diff) ≤ 2 * rec.err_linf
        @test maximum(diff) ≤ 1.5 * sample
        @test diff[1] ≤ diff[end]
        @info "the evolved state's fit against the analytic fit at t = 0.15 M (step 8e)" diff_by_radius = diff sample_error = sample err_linf = rec.err_linf err_l2 = rec.err_l2 residual = (fe.residual.value, fa.residual.value)
    end

    # A fit is built once per chunk in step 8e-ii, and its evaluator runs at
    # every layer point of every evaluation: both prices are recorded, not
    # asserted — a cost, not a claim.
    @testset "what a fit costs is recorded" begin
        T = Float64
        case = fitted_fixture(T)
        forest = hole_fixture_forest(T, case; N=8)
        geom = fitted_interior(case.interior, seed_track(case, 0), forest, 2;
                               t=0, n_L=8)
        analytic = analytic_sampler(case.background, zero(T); δ=geom.h / 8)
        fs = FieldSet{T}(forest, 20; G=2, centering=vertexcentered(3), backend=CPU())
        fill_exact!(fs, case, zero(T); interior=geom)
        fill_ghosts!(fs, GhostSchedule(fs, Operators(prolongation=4, restriction=4));
                     boundary=dirichlet(case, zero(T)))
        state = state_sampler(fs, 2; t=zero(T))
        ms = Dict{String,Float64}()
        for (name, s, cont) in (("analytic, cont 1", analytic, 1),
                                ("analytic, cont 2", analytic, 2),
                                ("state, cont 1", state, 1))
            build_fit(s, geom, case.interior; cont=cont, bounds=case.bounds)
            t0 = time_ns()
            n = 5
            for _ in 1:n
                build_fit(s, geom, case.interior; cont=cont, bounds=case.bounds)
            end
            ms["build_fit L = 8, " * name] = (time_ns() - t0) / (n * 1e6)
        end
        fit = build_fit(analytic, geom, case.interior; cont=1, bounds=case.bounds)
        xs = [fit.points[i] * (T(j) / 8) for i in eachindex(fit.points) for j in 1:8]
        acc = 0.0
        for x in xs
            acc += fit_state(fit.params, fit.host, x, zero(T))[1][1]
        end
        t0 = time_ns()
        for _ in 1:5, x in xs
            acc += fit_state(fit.params, fit.host, x, zero(T))[1][1]
        end
        ns_point = (time_ns() - t0) / (5 * length(xs))
        t0 = time_ns()
        for _ in 1:5, x in xs
            acc += background_state(case.background, 0.0, Tuple(x))[1][1]
        end
        ns_exact = (time_ns() - t0) / (5 * length(xs))
        @test isfinite(acc)
        @test all(isfinite, values(ms))
        @info "what a fit costs (step 8e)" ms fit_state_ns = ns_point u_exact_ns = ns_exact
    end
end
