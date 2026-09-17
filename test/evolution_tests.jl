# The right-hand side itself: what the fused kernel computes, that it
# computes nothing else, and what it costs.
#
# `CODE.md`, "One right-hand-side evaluation". The kernel is a *streaming*
# assembly of the same equation `gh_node_rhs_expanded` writes in one
# expression — it forms the stencils component by component and never
# builds them all — so the claim that matters is that the two agree on
# real data, on every background, at every order. That is the first
# testset, and it is what makes the convergence studies in
# `convergence_tests.jl` measurements of the *scheme* rather than of
# whatever the kernel happens to do.
#
# The two are not bit-identical and are not asked to be: one body reached
# from two call sites is contracted into fused multiply-adds differently
# (`CODE.md`, "Measured results"), so every comparison here is to roundoff
# against the size of the terms that produced the number.

using KernelAbstractions: CPU
using SpacetimeMetrics: GaugeWave, Harmonic, KerrSchild, Minkowski,
                        ShiftedMinkowski
using StaticArrays: SMatrix, SVector
using TreeGeneralizedHarmonic: _dg4, _sym4, gauge_at, gauge_work

# The right-hand side at one stored point, assembled on the host out of
# `apply_stencil` and `gh_node_rhs_expanded` — the two reference
# definitions of step 2 and step 1. It reads the *same working array* the
# kernel read, ghosts and all, so it is a claim about the assembly and not
# about the ghost fill.
#
# The dissipation is where the reference has to say what the kernel does
# rather than ask the reference function: `CODE.md`'s streaming order
# gives the source the accumulated `∂ₜh`, dissipation included, and
# `gh_node_rhs_expanded` has no dissipation at all. So the reference adds
# `Q_d` to both halves and corrects the source by the difference two calls
# to `gh_node_source` make — which is exactly, and only, the design choice
# spelled out.
function host_rhs_at(fs, prob, case, q, b, idx::NTuple{3,Int})
    T = eltype(fs.work)
    work = fs.work
    w1 = derivative_weights(T, Val(q), Val(1))
    w2 = derivative_weights(T, Val(q), Val(2))
    wD = dissipation_weights(T, dissipation_rank(Val(q)))
    h = spacing(T, fs.forest, blockkey(fs, b))

    line(v, d) = ξ -> work[Base.setindex(idx, ξ, d)..., v, b]
    ∂(v, d) = apply_stencil(w1, line(v, d), idx[d], 1) / h
    ∂∂(v, d) = apply_stencil(w2, line(v, d), idx[d], 1) / h^2
    function ∂∂mixed(v, d1, d2)
        f = (ξ, η) -> work[Base.setindex(Base.setindex(idx, ξ, d1), η, d2)...,
                           v, b]
        return apply_mixed_stencil(w1, f, idx[d1], idx[d2], 1, 1) / h^2
    end
    Q(v) = (case.ε_KO / h) * sum(apply_stencil(wD, line(v, d), idx[d], 1)
                                 for d in 1:3)

    hv = SVector{10,T}(ntuple(v -> work[idx..., v, b], Val(10)))
    Πv = SVector{10,T}(ntuple(v -> work[idx..., 10 + v, b], Val(10)))
    ∂h = ntuple(d -> SVector{10,T}(ntuple(v -> ∂(v, d), Val(10))), Val(3))
    ∂Π = ntuple(d -> SVector{10,T}(ntuple(v -> ∂(10 + v, d), Val(10))), Val(3))
    pairs = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    ∂∂h = ntuple(Val(6)) do n
        d1, d2 = pairs[n]
        SVector{10,T}(ntuple(Val(10)) do v
            d1 == d2 ? ∂∂(v, d1) : ∂∂mixed(v, d1, d2)
        end)
    end

    owned = ntuple(d -> idx[d] - fs.G[d], Val(3))
    Hl, dHl = prob.Hsrc === nothing ?
              (zero(SVector{4,T}), zero(SMatrix{4,4,T})) :
              gauge_at(T, prob.Hsrc.work, owned, b, Val(true))

    ∂ₜh, ∂ₜΠ = gh_node_rhs_expanded(hv, Πv, ∂h, ∂Π, ∂∂h, Hl, dHl,
                                    case.γ0, case.γ2)
    iszero(case.ε_KO) && return ∂ₜh, ∂ₜΠ

    Qh = SVector{10,T}(ntuple(v -> Q(v), Val(10)))
    QΠ = SVector{10,T}(ntuple(v -> Q(10 + v), Val(10)))
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    Δsrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh + Qh, ∂h), Hl, dHl,
                          case.γ0, case.γ2) -
           gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, case.γ0,
                          case.γ2)
    return ∂ₜh + Qh, ∂ₜΠ + QΠ + Δsrc
end

# The size the comparison is measured against: the terms that build `∂ₜΠ`,
# not the number that is left of them. Half of what is compared here
# cancels — a static background's `∂ₜΠ` is zero while `α√γγ^{ij}∂_i∂_j h`
# is not — so a relative difference against the result would call a
# correct answer a failure (`pointwise_backgrounds.jl` says the same
# thing).
function rhs_scale(fs, b, idx, q)
    T = eltype(fs.work)
    h = spacing(T, fs.forest, blockkey(fs, b))
    w2 = derivative_weights(T, Val(q), Val(2))
    s = zero(T)
    for v in 1:20, d in 1:3
        line = ξ -> fs.work[Base.setindex(idx, ξ, d)..., v, b]
        s = max(s, abs(apply_stencil(w2, line, idx[d], 1)) / h^2,
                abs(fs.work[idx..., v, b]))
    end
    return max(s, one(T))
end

# The cases this file evaluates the kernel on: one harmonic and moving,
# one non-harmonic and static (so the `Hsrc` path runs), one with a hole
# in the box (so the coefficients are far from flat). The hole sits
# outside the box, which has no interior treatment yet — that is step 5.
function rhs_cases(::Type{T}, which; ε_KO) where {T}
    all = (gaugewave=("gauge wave",
                      gauge_wave_case(T; ε_KO=ε_KO, γ0=one(T), γ2=T(-1 // 2)),
                      (N=8, roots=2)),
           shifted=("shifted Minkowski",
                    shifted_minkowski_case(T; ε_KO=ε_KO, γ0=one(T),
                                           γ2=T(-1 // 2)), (N=8, roots=1)),
           kerr=("harmonic Kerr off centre",
                 GHCase(T, Harmonic{T}(1, 9 // 10);
                        box=ntuple(_ -> (T(2), T(4)), 3),
                        periodic=(false, false, false), ε_KO=ε_KO, γ0=one(T),
                        γ2=T(-1 // 2)), (N=8, roots=1)))
    return map(k -> all[k], which)
end

# Which rows are run: all three cases at `q = 4`, with the dissipation off
# and on, and the gauge wave alone at the neighbouring orders. The narrower
# rows are a **budget** decision (proposed in step 3): a row costs a kernel
# specialisation and a background's dual passes to compile, this file was
# 71 s with the full cross product, and what the extra rows would add is
# the claim that the *stencil widths* are right at `q = 2` and `6` — which
# the convergence study measures directly.
const RHS_ROWS = ((2, 0.5, (:gaugewave,)),
                  (4, 0.0, (:gaugewave, :shifted, :kerr)),
                  (4, 0.5, (:gaugewave, :shifted, :kerr)),
                  (6, 0.5, (:gaugewave,)))

@testset "The kernel evaluates the expanded right-hand side, q=$q, ε_KO=$ε" for
    (q, ε, which) in RHS_ROWS

    # Guards the whole of the streaming order against the one-expression
    # reference of step 1: a component formed at the wrong offset, a
    # coefficient contracted on the wrong index, a mixed derivative with
    # its axes swapped, the dissipation scaled by the wrong power of `h`.
    # Every one of those still converges to *something*, and several of
    # them still converge at order `q` on a diagonal test.
    T = Float64
    N = q == 6 ? 10 : 8
    for (name, case, geom) in rhs_cases(T, which; ε_KO=T(ε))
        forest = gh_forest(T, case; N=N, roots=geom.roots)
        fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
        prob = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                         restriction=q + 2)),
                         case; q=q)
        t = T(1 // 4)
        fill_exact!(fs, case, t)
        u = statevector(fs)
        gather!(u, fs)
        du = similar(u)
        gh_rhs!(du, u, prob, t)
        da = statearray(du, fs)

        worst = zero(T)
        for b in 1:nblocks(fs), owned in ((1, 1, 1), (2, N ÷ 2, N),
                                          (N, N, N), (N ÷ 2, 3, N - 1))
            idx = ntuple(d -> owned[d] + fs.G[d], Val(3))
            ∂ₜh, ∂ₜΠ = host_rhs_at(fs, prob, case, q, b, idx)
            scale = rhs_scale(fs, b, idx, q)
            for v in 1:10
                worst = max(worst, abs(da[owned..., v, b] - ∂ₜh[v]) / scale,
                            abs(da[owned..., 10 + v, b] - ∂ₜΠ[v]) / scale)
            end
        end
        @test worst < 1e-12
    end
end

@testset "Minkowski is stationary to roundoff: du is exactly zero" begin
    # Guards the sharpest statement available about the kernel. Flat space
    # is `h = Π = 0` at every point, so every stencil sums weights against
    # zeros, every coefficient is exactly its flat value, and the source is
    # built from a vanishing `∂ g`. Anything that leaked a term — a stray
    # `+ h`, a coefficient built from `g` rather than from the offset, a
    # dissipation weight that does not sum to zero — shows up here as a
    # number that is not zero, at *every* resolution and order.
    T = Float64
    for q in (2, 4, 6), ε in (zero(T), T(1 // 2))
        N = q == 6 ? 10 : 8
        case = minkowski_case(T; L=one(T), ε_KO=ε, γ0=one(T), γ2=T(-1 // 2))
        forest = gh_forest(T, case; N=N, roots=2)
        fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
        prob = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                         restriction=q + 2)),
                         case; q=q)
        fill_exact!(fs, case, zero(T))
        u = statevector(fs)
        gather!(u, fs)
        @test all(iszero, u)
        du = fill!(similar(u), T(NaN))
        gh_rhs!(du, u, prob, T(3 // 7))
        @test all(iszero, du)
    end
end

@testset "The right-hand side is pure: u is untouched and du is repeatable" begin
    # Guards TreeAMR's contract (`CODE.md`: "The RHS never mutates `u`").
    # The working array is scratch and the state is the integrator's; a
    # kernel that wrote into `work` where it should have written `du` would
    # still converge on the first step and drift on the second. And two
    # evaluations at the same `(u, t)` must give the same `du` bit for bit:
    # the same compiled code at the same call site, which is the only
    # bit-identity this package claims.
    T = Float64
    q = 4
    case = shifted_minkowski_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=T(-1 // 2))
    forest = gh_forest(T, case; N=8, roots=1)
    fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
    prob = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                     restriction=q + 2)),
                     case; q=q)
    fill_exact!(fs, case, zero(T))
    u = statevector(fs)
    gather!(u, fs)
    u0 = copy(u)
    du1 = similar(u)
    du2 = similar(u)
    gh_rhs!(du1, u, prob, T(1 // 8))
    @test isequal(u, u0)
    gh_rhs!(du2, u, prob, T(1 // 8))
    @test isequal(u, u0)
    @test isequal(du1, du2)
    @test any(!iszero, du1)
    @test all(isfinite, du1)
end

@testset "The Dirichlet hook fills the outer ghosts with the analytic state" begin
    # Guards the hook and its time: it is built inside the right-hand side
    # at every evaluation with *that* evaluation's `t` (`CLAUDE.md`, "Hooks
    # depend on time"). A hook built once at `t = 0` would pass on a static
    # background and fail on a moving one, so the check is made on a case
    # that moves, at a time that is not zero, through the ghost points the
    # scheme actually reads.
    T = Float64
    q = 4
    G = q ÷ 2 + 1
    bg = GaugeWave(T(1 // 20), T(2))
    case = GHCase(T, bg; box=ntuple(_ -> (zero(T), T(2)), 3),
                  periodic=(false, false, false), ε_KO=zero(T), γ0=zero(T),
                  γ2=zero(T))
    forest = gh_forest(T, case; N=8, roots=1)
    fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
    prob = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                     restriction=q + 2)),
                     case; q=q)
    @test prob.hasdirichlet
    @test dirichlet(case, T(1 // 3)) !== nothing
    @test dirichlet(gauge_wave_case(T; ε_KO=zero(T), γ0=zero(T), γ2=zero(T)),
                    zero(T)) === nothing

    t = T(1 // 3)
    fill_exact!(fs, case, t)
    u = statevector(fs)
    gather!(u, fs)
    du = similar(u)
    gh_rhs!(du, u, prob, t)

    # The ghost planes below the domain, and the shared upper plane, which
    # belongs to no owned range and is the hook's too.
    worst = zero(T)
    for b in 1:nblocks(fs), idx in ((1, 5, 5), (2, 5, 5), (G, 5, 5),
                                    (5, 1, 5), (5, 5, 2),
                                    (8 + G + 1, 5, 5), (5, 8 + 2G + 1, 5))
        vals = state_tuple(bg, t, coordinates(fs, b, idx))
        for v in 1:20
            worst = max(worst, abs(fs.work[idx..., v, b] - vals[v]))
        end
    end
    @test worst == 0                       # the hook evaluates the same function

    # And a hook built at a *different* time would not have been zero: the
    # solution moves, so this is a claim and not a tautology.
    stale = state_tuple(bg, zero(T), coordinates(fs, 1, (1, 5, 5)))
    fresh = state_tuple(bg, t, coordinates(fs, 1, (1, 5, 5)))
    @test maximum(abs, fresh .- stale) > T(1 // 100)
end

@testset "The time step is the CFL bound of the measured speed" begin
    # Guards `CODE.md`'s "The time step": `λ = α√(tr γ^{ij}) + |β|`, the
    # maximum over owned points, and `dt = cfl · h_min / λ`. On flat space
    # the speed is exactly `√3` — the conservative bound, not the physical
    # speed 1 — and getting that wrong scales every run's step by a
    # constant nobody would notice until a stability limit.
    T = Float64
    q = 4
    for (case, geom) in ((minkowski_case(T; L=one(T), ε_KO=zero(T), γ0=zero(T),
                                         γ2=zero(T)), (N=8, roots=2)),
                         (shifted_minkowski_case(T; ε_KO=zero(T), γ0=zero(T),
                                                 γ2=zero(T)), (N=8, roots=1)))
        forest = gh_forest(T, case; N=geom.N, roots=geom.roots)
        fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
        prob = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                         restriction=q + 2)),
                         case; q=q)
        fill_exact!(fs, case, zero(T))
        u = statevector(fs)
        gather!(u, fs)
        λ = (scatter!(fs, u); max_speed(prob))
        dt = gh_dt(prob, u; cfl=T(1 // 4))
        h = minimum_spacing(T, forest)
        @test dt ≈ T(1 // 4) * h / λ
        if case.background isa Minkowski
            @test λ ≈ sqrt(T(3))
        else
            @test λ > sqrt(T(3))           # a shift adds to the bound
        end
    end
end

@testset "The dissipation switch is exactly ε_KO = 0" begin
    # Guards the `Val(DISS)` parameter: a problem with `ε_KO = 0` compiles
    # the kernel without the Kreiss–Oliger stencils at all, and must give
    # the same numbers as one that computes them and multiplies by zero.
    # If it does not, the switch is not a switch but a second scheme.
    T = Float64
    q = 4
    bg = GaugeWave(T(1 // 20), one(T))
    box = ntuple(_ -> (zero(T), one(T)), 3)
    off = GHCase(T, bg; box=box, periodic=(true, true, true), ε_KO=zero(T),
                 γ0=one(T), γ2=zero(T))
    tiny = GHCase(T, bg; box=box, periodic=(true, true, true),
                  ε_KO=T(1 // 10)^300, γ0=one(T), γ2=zero(T))
    dus = map((off, tiny)) do case
        forest = gh_forest(T, case; N=8, roots=1)
        fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
        prob = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                         restriction=q + 2)),
                         case; q=q)
        @test prob.valdiss === Val(!iszero(case.ε_KO))
        fill_exact!(fs, case, zero(T))
        u = statevector(fs)
        gather!(u, fs)
        du = similar(u)
        gh_rhs!(du, u, prob, zero(T))
        du
    end
    @test maximum(abs, dus[1] - dus[2]) ≤ 1e-290
end

@testset "The right-hand side costs what CODE.md records" begin
    # Not a performance gate — a measurement, recorded in `CODE.md` under
    # "Measured results" so that a later change shows up as a changed
    # number rather than as a test that still passes. The assertion is
    # loose on purpose: what it catches is an accidental fall back to a
    # dynamic dispatch or a heap allocation per point, which is orders of
    # magnitude and not percent.
    T = Float64
    q = 4
    case = gauge_wave_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T))
    forest = gh_forest(T, case; N=8, roots=2)
    fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
    prob = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                     restriction=q + 2)),
                     case; q=q)
    fill_exact!(fs, case, zero(T))
    u = statevector(fs)
    gather!(u, fs)
    du = similar(u)
    gh_rhs!(du, u, prob, zero(T))
    best = Inf
    for _ in 1:3
        best = min(best, @elapsed gh_rhs!(du, u, prob, zero(T)))
    end
    points = nleaves(forest) * forest.N^3
    ns = best / points * 1e9
    @info "RHS throughput" q threads = Threads.nthreads() points ns_per_point = ns
    @test ns < 50_000
end
