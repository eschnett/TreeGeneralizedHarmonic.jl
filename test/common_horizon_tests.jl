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
        ok(a) = (success=true, area=a, hlm=ComplexF64[a])
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

    # Guards the settings' refusals: a common find needs a positive tolerance and
    # a case of two holes.
    @testset "the settings refuse what cannot work" begin
        @test_throws "positive" CommonHorizon(T; d_c=0)
        @test_throws "stall" CommonHorizon(T; atol_factor=0)
        @test_throws "case of two holes" evolve!(T, case; forest=deepcopy(forest), q=q,
                                                  ops=ops, t_end=T(1 // 10),
                                                  common_horizon=CommonHorizon(T))
    end
end
