# The prescribed gauge source: which backgrounds have one, which ones are
# refused, and whether what is sampled into `Hsrc` comes back out of it.
#
# `CODE.md`, "Gauge and constraint damping". Three claims, and each of
# them is a `Val`, a refusal or a packing order that nothing else in the
# package checks.

using SpacetimeMetrics: GaugeWave, Harmonic, KerrSchild, Minkowski,
                        ShiftedMinkowski, boost, gauge_source,
                        gauge_source_grad, rotate, translate
using StaticArrays: SMatrix, SVector

# The measurement `isharmonic` may not be: `H^a` of a harmonic background
# evaluates to roundoff, not to zero (measured below), and the two classes
# are eight orders apart, so a threshold separates them here while being
# no basis at all for a `Val` that has to be exactly right.
const GAUGE_ZERO = 1e-8

function measured_gauge_scale(bg)
    s = 0.0
    for x in ((2.3, 1.7, -3.1), (-1.9, 2.7, 1.3), (3.7, -2.1, 0.9))
        for t in (0.0, 0.37)
            H = gauge_source(bg, SVector{4,Float64}(t, x[1], x[2], x[3]))
            s = max(s, maximum(abs, H))
        end
    end
    return s
end

@testset "isharmonic's table agrees with the gauge source it stands for" begin
    # Guards the table in `gauge.jl` drifting from the physics it claims —
    # including the case where one of `SpacetimeMetrics`' *unexported*
    # wrapper types is renamed on `main`, after which the fallback
    # `isharmonic(::AbstractMetric) = false` would quietly give a harmonic
    # background a sampled source (slow, and a refusal if it also moves).
    # The table is a `Val` parameter of the kernel, so it has to be right
    # rather than nearly right; this is the measurement it answers to.
    T = Float64
    rows = (("Minkowski", Minkowski(), true),
            ("gauge wave", GaugeWave(T(1 // 20), one(T)), true),
            ("shifted Minkowski", ShiftedMinkowski(T(1 // 2), one(T)), false),
            ("Kerr-Schild", KerrSchild{T}(1, 0), false),
            ("harmonic Kerr", Harmonic{T}(1, 9 // 10), true),
            ("boosted harmonic Kerr",
             boost(Harmonic{T}(1, 9 // 10), SVector{3,T}(3 // 10, 0, 0)), true),
            ("translated Kerr-Schild",
             translate(KerrSchild{T}(1, 0), SVector{4,T}(0, 1, 0, 0)), false),
            ("rotated harmonic Kerr",
             rotate(Harmonic{T}(1, 9 // 10), T(1 // 5), T(2 // 5), T(3 // 5)),
             true))
    for (name, bg, harmonic) in rows
        @test isharmonic(bg) == harmonic
        scale = measured_gauge_scale(bg)
        if harmonic
            @test scale < GAUGE_ZERO
        else
            @test scale > GAUGE_ZERO
        end
    end
end

@testset "isstatic is exact, and it is exact because ∂_t g is" begin
    # Guards the one input to the refusal below. A static background's
    # metric expression does not mention `t`, so the `t` partial of
    # `dmetric`'s forward-mode pass is an identical zero — which is why
    # this property may be measured where harmonicity may not.
    T = Float64
    @test isstatic(Minkowski())
    @test isstatic(ShiftedMinkowski(T(1 // 2), one(T)))
    @test isstatic(KerrSchild{T}(1, 0))
    @test isstatic(Harmonic{T}(1, 9 // 10))
    @test !isstatic(GaugeWave(T(1 // 20), one(T)))
    @test !isstatic(boost(Harmonic{T}(1, 9 // 10), SVector{3,T}(3 // 10, 0, 0)))
    # A boost by zero is not a moving background, and the test that says so
    # is the one that would catch an `isstatic` written as "is it wrapped
    # in a boost?" rather than as a statement about ∂_t g.
    @test isstatic(boost(Harmonic{T}(1, 9 // 10), zero(SVector{3,T})))
end

@testset "A moving non-harmonic background is refused, and says what to use" begin
    # Guards `CODE.md`'s one unsupported configuration: a time-dependent
    # prescribed gauge source, which a per-chunk sample cannot represent.
    # Boosted Kerr-Schild is exactly it, and the message has to name the
    # case that works instead — this is the refusal `CLAUDE.md` says not to
    # weaken.
    T = Float64
    moving_ks = boost(KerrSchild{T}(1, 0), SVector{3,T}(3 // 10, 0, 0))
    @test !isharmonic(moving_ks)
    @test !isstatic(moving_ks)
    @test_throws "boost(Harmonic(M, a), v)" GHCase(
        T, moving_ks; box=ntuple(_ -> (T(-8), T(8)), 3),
        periodic=(false, false, false), ε_KO=zero(T), γ0=one(T), γ2=zero(T))

    # The two neighbours of that case are both allowed: harmonic and
    # moving, and non-harmonic and static.
    @test GHCase(T, boost(Harmonic{T}(1, 9 // 10), SVector{3,T}(3 // 10, 0, 0));
                 box=ntuple(_ -> (T(-8), T(8)), 3), periodic=(false, false, false),
                 ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T)) isa GHCase
    @test GHCase(T, KerrSchild{T}(1, 0); box=ntuple(_ -> (T(-8), T(8)), 3),
                 periodic=(false, false, false), ε_KO=T(1 // 2), γ0=one(T),
                 γ2=zero(T)) isa GHCase
end

@testset "The damping parameters are checked where they are given" begin
    # Guards the two continuum conditions on the Gundlach–Pretorius term:
    # `γ0 ≥ 0` (a negative rate drives what it should damp) and `γ2 > −1`.
    # Both are properties of the case and both would otherwise be found as
    # a run that blows up in a way that looks like the physics.
    T = Float64
    mk(; γ0, γ2) = GHCase(T, Minkowski(); box=ntuple(_ -> (zero(T), one(T)), 3),
                          periodic=(true, true, true), ε_KO=zero(T), γ0=γ0, γ2=γ2)
    @test_throws "γ0 ≥ 0" mk(γ0=-one(T), γ2=zero(T))
    @test_throws "γ2 > −1" mk(γ0=one(T), γ2=-one(T))
    @test mk(γ0=one(T), γ2=T(-1 // 2)) isa GHCase
end

@testset "A sampled gauge source comes back out of Hsrc as it went in" begin
    # Guards the packing of the 20 slots — `H_b` in 1:4 and `∂_a H_b` in
    # 5:20, column-major — which `sample_gauge_source!` writes and the
    # kernel's `gauge_at` reads. The two are the only places that know the
    # order, and a transposed `dHl` would be a source term that is wrong by
    # the antisymmetric part of ∂H and right wherever it vanishes.
    T = Float64
    bg = ShiftedMinkowski(T(1 // 2), one(T))
    case = shifted_minkowski_case(T; ε_KO=zero(T), γ0=zero(T), γ2=zero(T))
    forest = gh_forest(T, case; N=8, roots=1)
    Hsrc = FieldSet{T}(forest, 20; G=0, centering=vertexcentered(3))
    sample_gauge_source!(Hsrc, bg, zero(T))

    for b in 1:nblocks(Hsrc), idx in ((1, 1, 1), (3, 5, 2), (8, 8, 8))
        x = coordinates(Hsrc, b, idx)
        Hl, dHl = gauge_source_grad(bg, SVector{4,T}(zero(T), x...))
        got = TreeGeneralizedHarmonic.gauge_at(T, Hsrc.work, idx, b, Val(true))
        @test isequal(got[1], Hl)
        @test isequal(got[2], dHl)
    end

    # ∂_a H_b of this background is genuinely asymmetric in (a, b) — a
    # transposition would be visible — and its time row is zero because the
    # background is static.
    x = coordinates(Hsrc, 1, (4, 4, 4))
    _, dHl = gauge_source_grad(bg, SVector{4,T}(zero(T), x...))
    @test maximum(abs, dHl - dHl') > 1e-3
    @test all(iszero, dHl[1, :])

    # The field set's shape is part of the claim: no ghosts, 20 variables.
    thin = FieldSet{T}(forest, 20; G=1, centering=vertexcentered(3))
    @test_throws "G = 0" sample_gauge_source!(thin, bg, zero(T))
    wrong = FieldSet{T}(forest, 10; G=0, centering=vertexcentered(3))
    @test_throws "20" sample_gauge_source!(wrong, bg, zero(T))
end

@testset "A harmonic case carries no gauge-source field set at all" begin
    # Guards the `Val(false)` path: on a harmonic background there is no
    # `Hsrc`, the kernel is compiled without the source terms, and nothing
    # is sampled. A `GHProblem` that allocated one anyway would be a
    # 20-variable field set per chunk and a lie about `CODE.md`'s table.
    T = Float64
    q = 4
    harmonic = gauge_wave_case(T; ε_KO=zero(T), γ0=zero(T), γ2=zero(T))
    sourced = shifted_minkowski_case(T; ε_KO=zero(T), γ0=zero(T), γ2=zero(T))
    for (case, hassource) in ((harmonic, false), (sourced, true))
        forest = gh_forest(T, case; N=8, roots=1)
        fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
        prob = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                         restriction=q + 2)),
                         case; q=q)
        @test (prob.Hsrc !== nothing) == hassource
        @test prob.valH === Val(hassource)
    end
end
