# The common horizon of a pair (step B7): which surface a common find accepts,
# what it records when there is none, and when the run says it has formed.
#
# `CODE.md`, "The common horizon and the handover". The finder cannot tell a
# missing horizon from a slow one: at `atol = 0` a stall is a success. So the
# claims here are about this package's own checks on top of it — a positive
# tolerance, the enclosure of both holes' surfaces, the origin between them —
# and that a failure is a row and never an exception. No evolution: the states
# are `fill_exact!`'s.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: common_horizon_row, common_state0, common_state_after,
                               common_record, horizon_row
using KernelAbstractions: CPU
using StaticArrays: SVector
using LinearAlgebra: norm
using TreeGeneralizedHarmonic: StateBounds, fitted_interior

isdefined(@__MODULE__, :TEST_TIER) || include("tiers.jl")  # `@full_tier`
isdefined(@__MODULE__, :hole_fixture) || include("evolution_cases.jl")

@testset verbose = true "The common horizon (step B7)" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)

    # A single hole of mass one stands in for a pair that has merged: its horizon,
    # the sphere `r = 2`, is the common horizon of any two surfaces inside it. The
    # holes' "own" surfaces are the found horizon's shape scaled and moved — the
    # finder's grid and the depth test are the ones a pair's finds give.
    case = hole_fixture(T; q=q)
    forest = hole_fixture_forest(T, case; N=8)
    fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
    sched = GhostSchedule(fs, ops)
    p = GHProblem(fs, sched, case; q=q)
    fill_exact!(fs, case, zero(T))
    u = statevector(fs)
    gather!(u, fs)
    own = find_gh_horizon(p, u, zero(T); N=12, spin=false)
    fake(x, r) = (success=true, origin=(x, 0.0, 0.0), origin_r_max=r,
                  H_norm=own.H_norm, grid=own.grid, hlm=own.hlm * (r / 2))

    # Guards the acceptance of a true common horizon: from a sphere about the
    # holes' center, the finder converges at the tolerance the holes' own
    # residuals set, the surface encloses both holes' surfaces, its origin lies
    # between them, and only then is the spin computed.
    @testset "a surface that encloses both holes is accepted, with its spin" begin
        ch = CommonHorizon(T; N=12, enclose=1 // 2)
        inds = (fake(-0.8, 0.5), fake(0.8, 0.5))
        row = common_horizon_row(p, u, zero(T), inds, (1, 1), nothing, nothing, ch)
        @test row.attempted && row.success === true && row.note === nothing
        @test row.from === :sphere
        @test row.atol == 1e-9 && row.H_norm ≤ row.atol
        @test isapprox(row.area, 16π; rtol=1e-3)
        @test isapprox(row.r_mean, 2; atol=1e-3)
        @test all(d -> isapprox(d, 2 - 1.3; atol=1e-2), row.depth)
        @test row.J !== nothing && abs(row.J) < 1e-4
        @test row.hlm !== nothing
        # Seeded from it, the next attempt starts from the last surface.
        again = common_horizon_row(p, u, zero(T), inds, (1, 1), nothing, row.hlm, ch)
        @test again.success === true && again.from === :last
        @test again.iters < row.iters
        @test isapprox(again.area, row.area; rtol=1e-8)
        # Without the spin, `J` is not computed.
        nospin = common_horizon_row(p, u, zero(T), inds, (1, 1), nothing, nothing,
                                    CommonHorizon(T; N=12, enclose=1 // 2, spin=false))
        @test nospin.success === true && nospin.J === nothing
        @info "a common find on the fixture (step B7)" iters = row.iters seeded = again.iters H = row.H_norm atol = row.atol area = row.area depth = row.depth time = row.time
    end

    # Guards the enclosure check: a horizon that is a horizon but not the holes'
    # common one — here each hole's surface reaches outside it — is refused by
    # name, with its depths, and is not an exception.
    @testset "a horizon that does not enclose a hole is refused, not thrown" begin
        ch = CommonHorizon(T; N=12, enclose=0)
        inds = (fake(-1.6, 0.6), fake(1.6, 0.6))
        row = common_horizon_row(p, u, zero(T), inds, (1, 1), nothing, nothing, ch)
        @test row.attempted && row.success === false
        @test occursin("enclose", row.note)
        @test all(<(0), row.depth)
        @test row.J === nothing && row.hlm === nothing
        # A hole whose own find failed leaves nothing to enclose: no attempt.
        miss = common_horizon_row(p, u, zero(T), (fake(-0.8, 0.5),
                                                  merge(fake(0.8, 0.5), (success=false,))),
                                  (1, 1), nothing, nothing, ch)
        @test !miss.attempted && occursin("hole 2", miss.note)
    end

    # Guards a pair that has no common horizon, its two holes found first: the
    # attempt fails, and the failure is a row with its note, never an exception
    # and never a success. On step H4's pair the attempts stalled, which at `atol
    # = 0` the finder calls a success, and at the holes' tolerance is a failure;
    # on the suite's Kerr-Schild pair at `d = 8` the flow shrinks the sphere onto
    # a hole's layer and the footprint guard refuses it (measured 2026-11-21).
    @testset "a pair with no common horizon records a failure" begin
        unit = TrackedSpec(T; r_0=T(3 // 5), r_1=T(7 // 5), margin=4)
        pair = binary_case(T; holes=head_on_holes(T; d=8, masses=(1, 1), interior=unit),
                           halfwidth=8, chunk=T(1 // 10),
                           horizon=Horizon(T; every=1, N=12, spin=false, r_seed=T(2)))
        pf = holes_forest(T, pair; N=8, roots=2, radii=(T(4), T(3), T(8 // 5)))
        U = FieldSet{T}(pf, 20; G=G, centering=vertexcentered(3))
        geom = with_ρ_max(holes_interior(pair, seed_tracks(pair, 0), zero(T)), T(4))
        fill_exact!(U, pair, zero(T); interior=geom)
        v = statevector(U)
        gather!(v, U)
        pp = GHProblem(U, GhostSchedule(U, ops), pair; q=q, interior=geom,
                       margin_check=false)
        inds = [horizon_row(pp, v, zero(T), nothing, 0; force=true, origin=h.center,
                            center=h.center) for h in pair.holes]
        @test all(r -> r.success === true, inds)
        ch = CommonHorizon(T; N=12)
        row = common_horizon_row(pp, v, zero(T), inds, (1, 1), geom, nothing, ch)
        @test row.attempted && row.success === false && row.note !== nothing
        @test row.atol > 0 && (row.H_norm === nothing || !(row.H_norm ≤ row.atol))
        @test row.J === nothing && row.hlm === nothing
        @info "no common horizon at d = 8 (step B7)" note = row.note iters = row.iters H = row.H_norm atol = row.atol time = row.time
    end

    # Guards formation: `streak` successive accepted finds whose area does not
    # decrease, dated at the first of them; a failure or a smaller area restarts
    # the streak, and a formed horizon stays formed.
    @testset "formation is three accepted finds of nondecreasing area" begin
        ch = CommonHorizon(T)
        ok(a) = (success=true, area=a, hlm=ComplexF64[a], r_min=1.0)
        no = (success=false, area=nothing, hlm=nothing)
        st = common_state0()
        for (t, r, streak, formed) in ((1, ok(10.0), 1, nothing), (2, ok(9.0), 1, nothing),
                                       (3, ok(9.5), 2, nothing), (4, no, 0, nothing),
                                       (5, ok(9.6), 1, nothing), (6, ok(9.7), 2, nothing),
                                       (7, ok(9.7), 3, 5.0), (8, no, 0, 5.0),
                                       (9, ok(9.8), 1, 5.0))
            st = common_state_after(st, r, t, ch)
            @test st.streak == streak
            @test isequal(st.t_formed, formed)
            @test isequal(st.hlm, r.hlm)
        end
        rec = common_record(merge(TreeGeneralizedHarmonic.NO_COMMON, (hlm=[1.0im],)), st)
        @test !haskey(rec, :hlm) && rec.streak == 1 && rec.t_formed == 5.0
    end

    # Guards the remnant's pieces at the handover: its track starts at the common
    # surface's origin with its radii and shape, moving with the holes' mass-
    # weighted velocity and acceleration (a head-on pair's cancel) and predicting
    # along the quadratic; its ranges are the union of the holes'; and its finds
    # are paced so that its least radius moves at most `G h/4` between two.
    @testset "the remnant's track, ranges and find cadence" begin
        unit = TrackedSpec(T; r_0=T(3 // 5), r_1=T(7 // 5), margin=4)
        hs = head_on_holes(T; d=8, masses=(1, 1), interior=unit)
        pair = binary_case(T; holes=hs, halfwidth=16, chunk=T(1 // 4),
                           horizon=Horizon(T; every=1, N=12, spin=false))
        trs = seed_tracks(pair, T(1 // 2))
        row = (origin=(0.0, 0.0, 0.0), r_min=1.1, r_max=2.6, hlm=own.hlm, grid=own.grid)
        tr = TreeGeneralizedHarmonic.remnant_track(0.5, row, trs, (1, 1); lmax=8)
        @test tr.c_find == zero(SVector{3,T}) && tr.t_find == 1 // 2
        @test tr.r_min == 1.1 && tr.r_max == 2.6 && tr.nfinds == 1 && tr.quadratic
        @test norm(tr.v_est) < 1e-12 && norm(tr.pred.a) < 1e-12
        @test norm(trs[1].pred.a) > 0      # each hole falls; the pair's center does not
        @test isapprox(tr.shape[1], own.hlm[1] |> real; rtol=1e-12)
        b1 = StateBounds{T}(0.1, 3, 0.2, 4, 1, 10, 1)
        b2 = StateBounds{T}(0.2, 5, 0.1, 3, 2, 5, 1)
        geom = fitted_interior(FittedSpec(T; variant=:fitted, margin=3, n_L=4), tr,
                               forest, G; t=T(1 // 2), n_L=4)
        rb = TreeGeneralizedHarmonic.remnant_bounds((b1, b2), geom)
        @test (rb.α_min, rb.α_max, rb.λ_min, rb.λ_max, rb.β_max, rb.K_max) ==
              (0.1, 5, 0.1, 4, 2, 10)
        @test rb.r_gate == geom.r_in - geom.offset
        # The spacing cap (the remnant's at the handover): the geometry the mesh
        # will have once its floor has refined the layer's blocks, which this
        # mesh fails until then and passes at the capped spacing.
        spec6 = FittedSpec(T; variant=:fitted, margin=3, n_L=2 * (G + 1))
        geomc = fitted_interior(spec6, tr, forest, G; t=T(1 // 2), n_L=2 * (G + 1),
                                h_cap=geom.h / 2)
        @test geomc.h == geom.h / 2 && geomc.offset == 3 * geom.h / 2
        @test fitted_interior(spec6, tr, forest, G; t=T(1 // 2), n_L=2 * (G + 1),
                              h_cap=2 * geom.h).h == geom.h
        @test_throws "not far enough inside" check_interior_radii(forest, geomc, nothing,
                                                                  G; t=T(1 // 2))
        @test check_interior_radii(forest, geomc, nothing, G; t=T(1 // 2),
                                   h_cap=geomc.h).h == geomc.h
        # The depth diagnostic: twice the cells at half the spacing is the same
        # layer.
        sp2 = TreeGeneralizedHarmonic.deepened_spec(spec6, 2)
        @test sp2.margin == 6 && sp2.n_L == 4 * (G + 1) && sp2.core_min == spec6.core_min
        @test TreeGeneralizedHarmonic.deepened_spec(spec6, 1) === spec6
        gd = fitted_interior(sp2, tr, forest, G; t=T(1 // 2), n_L=4 * (G + 1),
                             h_cap=geom.h / 2)
        @test gd.h == geom.h / 2 && gd.offset ≈ 3 * geom.h
        gf = TreeGeneralizedHarmonic.growth_finds
        @test gf(0.6, 3, 1 / 48, 1 / 4) == ceil(Int, 0.25 * 0.6 / (3 / 48 / 4))
        @test gf(0.0, 3, 1 / 48, 1 / 4) == 1
    end

    # Guards the handover's code path and its run state, on the harmonic smoke
    # pair with a stand-in for the physics: the common surface, accepted from the
    # second row on, is a sphere of radius 3.3 about a point `M` from hole 1 (no
    # pair cheap enough for the suite forms a common horizon, `CODE.md`, "The
    # common horizon and the handover"), and the remnant's finds return it. Its
    # layer reaches blocks of the pair's mesh at `h = 1/4`, which cannot hold it,
    # as on H4's pair: the handover is at the row the common surface is first
    # accepted all the same, at the holes' spacing scaled by the masses (`1/8`),
    # and the regrid after it refines those blocks. Every later row has one hole,
    # found, its fit valid, the state finite; and a chain of one-chunk restarts
    # across the handover — one of them from that row's own checkpoint, on the
    # pair's mesh — is the run bit for bit, the records compared without the
    # attempts' wall times.
    @full_tier @testset "the pair is handed over to one remnant at the first accepted row, and restarts bit for bit" begin
        unitf = FittedSpec(T; variant=:fitted, margin=4, n_L=0, fit_cont=2, lmax_fit=8)
        hc = binary_case(T; holes=head_on_holes(T; d=8, masses=(1, 1), interior=unitf,
                                                chart=:harmonic),
                         halfwidth=8, chunk=T(1 // 20), ε_KO=T(1 // 2), γ2=zero(T),
                         horizon=Horizon(T; every=1, N=12, spin=false), gauge_source=:holes,
                         refinement=Refinement(T; refine_tol=T(2 // 5),
                                               coarsen_tol=T(1 // 10), maxlevel_cap=4,
                                               floor_margin=zero(T), ceiling_cells=0))
        hf = holes_forest(T, hc; N=8, roots=2, radii=(T(4), T(3), T(2), T(6 // 5)))
        fake = Ref{Any}(nothing)
        R = 3.3
        function sphere_out(f)
            tm = f.template
            s = R / tm.r_mean
            return (success=true, iters=1, origin=f.c, center=f.c, center_offset=0.0,
                    r_min=tm.r_min * s, r_mean=R, r_max=tm.r_max * s,
                    origin_r_min=tm.r_min * s, origin_r_mean=R,
                    origin_r_max=tm.r_max * s, area=4π * R^2, M_irr=R / 2, J=NaN,
                    spin_axis=SVector(NaN, NaN, NaN), M_ch=NaN, hlm=tm.hlm .* s,
                    grid=tm.grid, H_norm=1e-12, spin_success=false)
        end
        function fake_find(p, u, t; origin=nothing, kwargs...)
            f = fake[]
            f !== nothing && origin !== nothing && abs(origin[3] - f.c[3]) < 0.01 &&
                return sphere_out(f)
            return find_gh_horizon(p, u, t; origin=origin, kwargs...)
        end
        function fake_common(p, u, t, inds, masses, geom, seed, ch)
            t < 1 / 20 - 1e-12 &&
                return merge(TreeGeneralizedHarmonic.NO_COMMON,
                             (attempted=true, success=false, note="not yet", iters=1,
                              H_norm=1.0, atol=1e-9, from=:sphere, time=0.0))
            if fake[] === nothing
                i = inds[1]
                fake[] = (c=SVector{3,Float64}(i.origin) + SVector(1, 0, 0.05),
                          template=(hlm=i.hlm, grid=i.grid, r_mean=i.r_mean,
                                    r_min=i.r_min, r_max=i.r_max))
            end
            o = sphere_out(fake[])
            return (attempted=true, success=true, note=nothing, iters=1, H_norm=1e-12,
                    atol=1e-9, from=:sphere, origin=Tuple(o.origin),
                    r_min=o.origin_r_min, r_mean=R, r_max=o.origin_r_max, area=o.area,
                    M_irr=o.M_irr, J=nothing, M_ch=nothing, depth=(0.5, 0.5),
                    core_gap=(0.1, 0.1), time=0.0, hlm=o.hlm, grid=o.grid)
        end
        kw = (q=q, ops=ops, chunk=T(1 // 20), cfl=T(1 // 2), t_end=T(3 // 20),
              common_horizon=CommonHorizon(T; d_c=10, N=12, spin=false, streak=1,
                                           handover=true),
              common_find=fake_common, find=fake_find, checkpoint_sync_to_disk=false,
              regrid=true)
        a = evolve!(T, hc; forest=deepcopy(hf), kw...)
        r0, r1 = a.records[1], a.records[2]
        @test length(r0.holes) == 2 && r0.common.handover == "the common horizon has not formed"
        @test length(r1.holes) == 2 && r1.common.handover == "handed over" &&
              r1.merged == T(1 // 20)
        @test all(r -> length(r.holes) == 1 && r.holes[1].success === true &&
                       r.holes[1].fit_valid === true && r.finite && r.merged == T(1 // 20) &&
                       r.separation === nothing && !haskey(r, :common),
                  a.records[3:end])
        @test length(a.records) == 4 && a.track isa Vector && length(a.track) == 1
        @test a.interior isa HoleInteriors{1}
        # The remnant's spacing is the holes' `1/16` times the mass ratio, the
        # pair's mesh across its layer was a level coarser, and the regrid
        # refined it.
        rem = a.interior.holes[1]
        @test rem.h == 1 // 8
        @test first(geometry_spacing(hf, rem, T(3 // 20))) == 1 // 4
        @test first(geometry_spacing(a.forest, rem, T(3 // 20))) == 1 // 8
        fake[] = nothing
        pre = joinpath(mktempdir(), "handover")
        b = evolve!(T, hc; forest=deepcopy(hf), kw..., checkpoint_path_prefix=pre,
                    max_walltime_seconds=1e-9)
        jobs = 1
        while !b.finished
            b = evolve!(T, hc; kw..., checkpoint_path_prefix=pre,
                        max_walltime_seconds=1e-9, restart_file=latest_checkpoint(pre))
            jobs += 1
        end
        @test jobs == 3
        untimed(r) = haskey(r, :common) ? merge(r, (common=merge(r.common, (time=nothing,)),)) : r
        @test isequal(a.u, b.u)
        @test isequal(untimed.(a.records), untimed.(b.records))
        @test isequal(repr(a.track), repr(b.track))
    end

    # Guards the settings' refusals: a common find needs a positive tolerance and
    # a case of two holes.
    @testset "the settings refuse what cannot work" begin
        @test_throws "positive" CommonHorizon(T; d_c=0)
        @test_throws "stall" CommonHorizon(T; atol_factor=0)
        @test_throws "at least 1" CommonHorizon(T; remnant_depth=0)
        @test_throws "case of two holes" evolve!(T, case; forest=deepcopy(forest), q=q,
                                                  ops=ops, t_end=T(1 // 10),
                                                  common_horizon=CommonHorizon(T))
        # The handover is built for fitted harmonic holes: the suite's Kerr-Schild
        # pair of tracked targets is refused before anything is built.
        unit = TrackedSpec(T; r_0=T(3 // 5), r_1=T(7 // 5), margin=4)
        ks = binary_case(T; holes=head_on_holes(T; d=8, masses=(1, 1), interior=unit),
                         halfwidth=8, chunk=T(1 // 10),
                         horizon=Horizon(T; every=1, N=12, spin=false))
        @test_throws "not fitted" evolve!(T, ks;
                                          forest=holes_forest(T, ks; N=8, roots=2,
                                                              radii=(T(4),)),
                                          q=q, ops=ops, t_end=T(1 // 10),
                                          common_horizon=CommonHorizon(T; handover=true))
    end
end
