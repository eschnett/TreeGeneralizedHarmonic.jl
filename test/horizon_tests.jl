# The horizon: the stopgap interpolator, the guard that keeps it out of the
# layer, and the find with its area, mass and spin.
#
# `CODE.md`, "Analysis quantities" (the horizon rows) and "Upstream
# prerequisites" (point interpolation);
# `notes/methods-ghso2.md`, "Apparent horizons and spin".
#
# The file is in two halves. The first is about the *interpolator* and is
# cheap: it evaluates no background at all where it can help it, and its
# claims are exactness on polynomials, a rate on the analytic metric, and
# the refusal of a query that would read the layer. The second finds the
# horizon of the suite's static hole and checks it against Kerr — and then
# against the analysis record, which is where `CODE.md` says these numbers
# belong. The `a = 9/10` chart and the `t = 50 M` trace are in
# `test/hole_runs.jl`, because the resolution the spinning hole's layer
# needs (`h ≲ 0.05 M`) is a mesh of a thousand blocks and not a test file.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using KernelAbstractions: CPU
using StaticArrays: SVector
import SpacetimeMetrics as SM

@testset verbose = true "The horizon" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)

    # A field set over the hole fixture's box carrying `nvars` polynomials,
    # filled everywhere a stencil or an interpolation window can reach: the
    # owned points by `fill_by_coordinates!`, the physical ghosts by a
    # boundary hook of the *same* polynomial, the interface ghosts by
    # prolongation — which is exact at order `p = q + 2` for a polynomial
    # of degree `≤ q + 1`. So the whole stored array holds the polynomial
    # exactly, and any error the interpolator shows is its own.
    polys = (x -> 3 + x[1] - 2x[2] + x[3] / 2,                     # degree 1
             x -> 1 + x[1]^3 - 2x[2]^2 * x[3] + x[3]^3 - x[1] * x[2] * x[3])
    function poly_fieldset(f::Function; N, roots, refined=false)
        case = hole_fixture(T; q=q)
        forest = gh_forest(T, case; N=N, roots=roots, refined=refined)
        fs = FieldSet{T}(forest, 2; G=G, centering=vertexcentered(3))
        fill_by_coordinates!(AllVariables(x -> (f(x), 2 * f(x) - 1)), fs)
        sched = GhostSchedule(fs, ops)
        fill_ghosts!(fs, sched;
                     boundary=CellBoundary(AllVariables((x, δ) ->
                                                            (f(x), 2 * f(x) - 1))))
        return forest, fs
    end

    # A point located in the wrong block is interpolated from data that is
    # merely *near* it, and on a refined mesh the wrong block is a level
    # away — an error that looks like a plausible interpolation error and
    # is not one.
    @testset "locate_block finds the leaf whose extent contains the point" begin
        forest, _ = poly_fieldset(polys[1]; N=8, roots=2, refined=true)
        @test maxlevel(forest) == 1              # the two-level mesh
        pts = [SVector{3,T}(x, y, z)
               for x in (-2.4, -1.1, 0.0, 0.7, 2.49),
                   y in (-2.3, -0.6, 1.9), z in (-1.7, 0.3, 2.2)]
        for x in pts
            b = locate_block(forest, x)
            @test b !== nothing
            ext = block_extent(T, forest, forest.leaves[b])
            @test all(d -> ext[d][1] ≤ x[d] < ext[d][2], 1:3)
        end
        # Outside the box there is no leaf, and the interpolator says so
        # rather than extrapolating from the nearest one.
        @test locate_block(forest, SVector{3,T}(2.51, 0, 0)) === nothing
        @test locate_block(forest, SVector{3,T}(0, -2.6, 0)) === nothing
        _, fs = poly_fieldset(polys[1]; N=8, roots=2, refined=true)
        @test_throws ArgumentError interpolate(fs, [SVector{3,T}(0, 0, 3)];
                                               q=q)
    end

    # Order `q + 2` interpolation reproduces a polynomial of degree `q + 1`
    # exactly and one of degree `q + 2` only to truncation; an off-by-one
    # in the window's width or in its centering shows up as the first of
    # those failing and nowhere else.
    @testset "interpolation is exact on polynomials of degree ≤ q + 1" begin
        for refined in (false, true)
            _, fs = poly_fieldset(polys[2]; N=8, roots=2, refined=refined)
            xs = [SVector{3,T}(x, y, z)
                  for x in (-1.93, -0.37, 0.61, 1.49),
                      y in (-1.11, 0.23, 1.77), z in (-0.89, 0.05, 2.13)]
            vals = interpolate(fs, xs; q=q)
            exact = [polys[2](x) for x in xs]
            @test maximum(abs.(getindex.(vals, 1) .- exact)) < 1e-12
            @test maximum(abs.(getindex.(vals, 2) .-
                               (2 .* exact .- 1))) < 1e-12
            # The gradient of the same interpolant, one order behind and
            # still exact on this degree.
            _, grads = interpolate_grad(fs, xs; q=q)
            δ = 1e-5
            for (i, x) in enumerate(xs), d in 1:3
                e = SVector{3,T}(ntuple(k -> k == d ? δ : zero(T), 3))
                fd = (polys[2](x + e) - polys[2](x - e)) / (2δ)
                @test abs(grads[i][d][1] - fd) < 1e-7
            end
        end
    end

    # The same window on a polynomial one degree higher is *not* exact —
    # a window wider than it advertises would pass the claim above for the
    # wrong reason.
    @testset "and not on one of degree q + 2" begin
        _, fs = poly_fieldset(x -> x[1]^4, N=8, roots=2)
        xs = [SVector{3,T}(0.61, -1.11, 0.05)]
        v = interpolate(fs, xs; q=q)[1][1]
        @test abs(v - 0.61^4) > 1e-6
    end

    # The rate is the claim `CODE.md` makes about the order: `q + 2` for
    # the value and `q + 1` for the gradient, which is what makes `K_ij`
    # one order behind `γ_ij` in the ADM data and the horizon's location
    # the *solution's* error rather than the interpolant's.
    @testset "interpolation converges at order q + 2 on the analytic metric" begin
        case = hole_fixture(T; q=q)
        xs = [SVector{3,T}(1.9 * sin(θ) * cos(φ), 1.9 * sin(θ) * sin(φ),
                           1.9 * cos(θ))
              for θ in (0.4, 1.1, 2.3), φ in (0.2, 1.7, 3.4, 5.1)]
        exact = map(xs) do x
            h, Π, ∂h = background_state(case.background, zero(T), Tuple(x))
            (h, ∂h)
        end
        hs = T[]
        ev = T[]
        eg = T[]
        # The fixture's nested hierarchy and not a uniform mesh: a uniform
        # `N = 8` over a box of half-width `5/2` has `h = 5/8`, on which the
        # window around a query at `r = 1.9` reaches the layer and the
        # guard refuses it — which is the guard being right about a mesh
        # too coarse to have a horizon on it at all. Doubling `N` halves
        # every spacing and leaves the layout alone, which is what makes
        # the ratio a rate (`CODE.md`'s frozen-hierarchy protocol).
        for N in (6, 12)
            forest = hole_fixture_forest(T, case; N=N)
            fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
            fill_exact!(fs, case, zero(T))
            sched = GhostSchedule(fs, ops)
            fill_ghosts!(fs, sched; boundary=dirichlet(case, zero(T)))
            vals, grads = interpolate_grad(fs, xs; q=q,
                                           mask=interior_mask(case.interior,
                                                              zero(T)))
            push!(hs, minimum_spacing(T, forest))
            push!(ev, maximum(i -> maximum(abs.(vals[i][1:10] .- exact[i][1])),
                              eachindex(xs)))
            push!(eg, maximum(eachindex(xs)) do i
                      maximum(d -> maximum(abs.(grads[i][d][1:10] .-
                                                exact[i][2][d])), 1:3)
                  end)
        end
        rate_v = log2(ev[1] / ev[2])
        rate_g = log2(eg[1] / eg[2])
        @test rate_v > q + 2 - 1 // 2
        @test rate_g > q + 1 - 1 // 2
        @info "interpolation on Kerr-Schild: value $(ev) rate $(rate_v), " *
              "gradient $(eg) rate $(rate_g)"
    end

    # The layer and the frozen core are not a numerical solution, so a
    # query whose *footprint* reaches them must refuse rather than report
    # a horizon of data the equations never produced. The check is on the
    # footprint and not on the query point, which is the whole difference:
    # a point outside `r_1` can still read inside it.
    @testset "the provider refuses a query whose footprint reaches r_1" begin
        case = hole_fixture(T; q=q)
        forest = hole_fixture_forest(T, case; N=8)
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        sched = GhostSchedule(fs, ops)
        p = GHProblem(fs, sched, case; q=q)
        fill_exact!(fs, case, zero(T))
        u = statevector(fs)
        gather!(u, fs)
        mask = interior_mask(case.interior, zero(T))
        h = minimum_spacing(T, forest)
        r_1 = case.interior.r_1
        # Far outside: answered, and with the analytic value.
        far = SVector{3,T}(1.9, 0, 0)
        hexact, _, _ = background_state(case.background, zero(T), Tuple(far))
        @test maximum(abs.(interpolate(fs, [far]; q=q,
                                       mask=mask)[1][1:10] .- hexact)) < 1e-4
        # Just outside `r_1` by less than the window's reach: refused,
        # although the point itself is evolved.
        near = SVector{3,T}(r_1 + h / 2, 0, 0)
        @test_throws ArgumentError interpolate(fs, [near]; q=q, mask=mask)
        @test_throws ArgumentError interpolate(fs, [SVector{3,T}(0, 0, 0)];
                                               q=q, mask=mask)
        # And with no guard the same query is answered, so the refusal is
        # the mask's and not the mesh's.
        @test all(isfinite, interpolate(fs, [near]; q=q)[1])
        # A seed sphere inside the layer is refused by the same guard,
        # through the provider.
        @test_throws ArgumentError find_gh_horizon(p, u, zero(T); N=8,
                                                   r_seed=0.8, spin=false)
    end

    # The measurement `CODE.md` and GHSO2 both state: Kerr's horizon
    # recovered from sampled data, from a *displaced* guess, with the
    # surface enclosing the layer by the margin `m`. The reference values
    # are `A = 4π(r₊² + a²)`, `M_irr = √(A/16π)`, `J = M a` and `M_ch = M`.
    @testset "the static hole's horizon is Kerr's, from a displaced guess" begin
        case = hole_fixture(T; q=q)
        forest = hole_fixture_forest(T, case; N=8)
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        sched = GhostSchedule(fs, ops)
        p = GHProblem(fs, sched, case; q=q)
        fill_exact!(fs, case, zero(T))
        u = statevector(fs)
        gather!(u, fs)
        h = minimum_spacing(T, forest)
        out = find_gh_horizon(p, u, zero(T); N=12,
                              origin=SVector{3,T}(0.1, -0.05, 0.08),
                              r_seed=1.7)
        @test out.success
        @test out.center_offset < 10h^3          # the interpolation's order
        for r in (out.r_min, out.r_mean, out.r_max)
            @test isapprox(r, 2; atol=1e-3)
        end
        @test isapprox(out.area, 4π * 4; rtol=1e-3)
        @test isapprox(out.M_irr, 1; atol=1e-3)
        @test abs(out.J) < 1e-4
        @test isapprox(out.M_ch, 1; atol=1e-3)
        # It encloses the layer by the margin the interior was placed with
        # — which is the statement that makes the whole construction
        # consistent: the finder reads only evolved data, and there are
        # `m` spacings of it between `r_1` and the surface.
        @test out.r_min ≥ case.interior.r_1 + case.interior.margin * h
        @info "horizon of the static hole: r = ($(out.r_min), " *
              "$(out.r_mean), $(out.r_max)), area $(out.area), " *
              "M_irr $(out.M_irr), J $(out.J), M_ch $(out.M_ch), " *
              "offset $(out.center_offset), iters $(out.iters)"

        # Seeded from the previous shape, as the driver seeds every find
        # after the first: the same surface, in fewer iterations.
        seeded = find_gh_horizon(p, u, zero(T); N=12, hlm=out.hlm, spin=false)
        @test seeded.success
        @test seeded.iters < out.iters
        @test isapprox(seeded.r_mean, out.r_mean; rtol=1e-10)
    end

    # A run that finishes without its analysis quantities is not a result
    # (`CLAUDE.md`), and the horizon rows are part of them. The cadence is
    # the case's `k`, counted on the record, so `every = 2` leaves the odd
    # chunks empty rather than repeating the previous answer.
    @testset "the analysis record carries the horizon rows at the cadence" begin
        case = with_horizon(hole_fixture(T; q=q),
                            Horizon(T; every=2, N=10, r_seed=T(9 // 5)))
        out = gh_hole_run(T, case; N=8, q=q, t_end=T(1 // 5))
        @test length(out.records) == 3           # t = 0, 1/10, 1/5
        # `every = 2` is the cadence, counted on the record: its first and
        # third entries carry the horizon and its second carries `nothing`
        # — and not a copy of the previous answer, which is what a row
        # meaning "the last horizon we saw" would be.
        @test out.records[2].horizon_success === nothing
        @test out.records[2].area === nothing
        @test out.records[2].hlm === nothing
        for r in (out.records[1], out.records[3])
            @test r.horizon_success === true
            @test r.horizon_note === nothing
            @test isapprox(r.r_mean, 2; atol=1e-3)
            @test isapprox(r.area, 4π * 4; rtol=1e-3)
            @test isapprox(r.M_irr, 1; atol=1e-3)
            @test isapprox(r.M_ch, 1; atol=1e-3)
            @test abs(r.J) < 1e-4
            @test r.center_offset < 1e-3
            @test length(r.hlm) == 100           # EquiangularGrid(9)
            @test all(isfinite, r.origin)
            @test all(isfinite, r.spin_axis)
        end
        # The hole is static, so the horizon at the last chunk is the one
        # at the first — to the *solution's* truncation error over the run
        # and not to roundoff: at `q = 2` and `h = 5/64` the state has
        # moved by `~1e−4` by `t = 1/5`, and the area is a functional of
        # it. The claim that pins the number is the comparison with Kerr
        # above, at both ends.
        @test isapprox(out.records[3].area, out.records[1].area; rtol=2e-3)

    end

    # `PLAN.md` asks for the horizon of the step-5 *and* step-6 runs: this
    # is the second, the mesh the indicator chose for itself rather than
    # one written down by hand. Nothing about the find changes — which is
    # the claim: the interpolator asks `find_leaf` where a point is, so a
    # mesh of two levels chosen by `τ` is the same mesh to it as three
    # shells chosen by a fixture.
    @testset "the horizon is found on the mesh the indicator chose" begin
        case = with_horizon(adaptive_hole_fixture(T),
                            Horizon(T; every=1, N=12))
        cyc = gh_adapt_cycle(T, case; N=8, roots=4, q=q)
        @test cyc.converged
        p = GHProblem(cyc.U, cyc.schedule, case; q=q)
        u = statevector(cyc.U)
        gather!(u, cyc.U)
        h = minimum_spacing(T, cyc.forest)
        out = find_gh_horizon(p, u, zero(T); N=12,
                              origin=SVector{3,T}(-0.12, 0.09, 0.05))
        @test out.success
        @test isapprox(out.r_mean, 2; atol=1e-2)
        @test isapprox(out.area, 4π * 4; rtol=1e-2)
        @test isapprox(out.M_irr, 1; atol=1e-2)
        @test abs(out.J) < 1e-3
        @test out.r_min ≥ case.interior.r_1 + case.interior.margin * h
        @info "horizon on the indicator's mesh ($(nleaves(cyc.forest)) " *
              "blocks, h = $h): r_mean $(out.r_mean), area $(out.area), " *
              "M_irr $(out.M_irr), J $(out.J), offset $(out.center_offset)"
    end

    # A find that throws must not end the run: the horizon is a
    # diagnostic, and what the record holds instead is the failure beside
    # the chunk it happened at (proposed in step 7).
    @testset "a failed find is recorded rather than thrown" begin
        case = with_horizon(hole_fixture(T; q=q),
                            Horizon(T; every=1, N=8, r_seed=T(4 // 5)))
        out = gh_hole_run(T, case; N=8, q=q, t_end=T(1 // 10))
        @test out.records[1].horizon_success === false
        @test occursin("r_1", out.records[1].horizon_note)
        @test out.records[1].area === nothing
        @test out.records[end].finite
    end
end
