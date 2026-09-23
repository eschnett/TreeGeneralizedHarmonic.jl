# The range projection and the validity monitor (step 8b).
#
# `CODE.md`, "The interior" (the range projection) and "Analysis
# quantities" (the validity rows). The pattern is TreeHydro's atmosphere
# reset, and so are the claims: a pointwise map that repairs what it is for
# and moves nothing else, bit for bit; idempotent on the state *and* on its
# flag; the identity on every healthy state; and — the claim every later
# experiment with it rests on — a run on which it never fires is bit for bit
# the run without it.
#
# **What this file costs.** Everything but the last testset is pointwise or
# one kernel launch on the step-5 fixture's mesh. The last is two short
# runs of that fixture, which is the suite's "one short run" `PLAN.md` asks
# the step to add, paid twice because the claim is a comparison.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: NC, _sym4, _pack10, _adm_split, _pairindex
using KernelAbstractions: CPU
using LinearAlgebra: Diagonal, I, Symmetric, eigvals, normalize
using StaticArrays: SMatrix, SVector
import SpacetimeMetrics as SM

isdefined(@__MODULE__, :gh_backgrounds) || include("pointwise_backgrounds.jl")
isdefined(@__MODULE__, :hole_fixture) || include("evolution_cases.jl")

# A synthetic state from its ADM parts: the spectrum of `γ` in a fixed,
# generic rotation (so that no eigenvector is a coordinate axis and every
# component of `γ` is nonzero), the lowered shift `β_i = h_ti`, `h_tt`, and
# `Π`. Built from `Rational`s, so that the `Float32` row is the same state
# rounded once.
function synthetic_state(::Type{T}; λ, htt, β,
                         Π=ntuple(v -> T(v - 5) / 20, NC)) where {T}
    a, b, c = T(1 // 3), T(2 // 5), T(3 // 7)
    Rz = SMatrix{3,3,T}(cos(a), sin(a), 0, -sin(a), cos(a), 0, 0, 0, 1)
    Ry = SMatrix{3,3,T}(cos(b), 0, -sin(b), 0, 1, 0, sin(b), 0, cos(b))
    Rx = SMatrix{3,3,T}(1, 0, 0, 0, cos(c), sin(c), 0, -sin(c), cos(c))
    R = Rz * Ry * Rx
    γ = R * Diagonal(SVector{3,T}(λ)) * R'
    γ = (γ + γ') / 2
    o = one(T)
    hm = SMatrix{4,4,T}(htt, β[1], β[2], β[3],
                        β[1], γ[1, 1] - o, γ[2, 1], γ[3, 1],
                        β[2], γ[1, 2], γ[2, 2] - o, γ[3, 2],
                        β[3], γ[1, 3], γ[2, 3], γ[3, 3] - o)
    return _pack10(hm), SVector{NC,T}(Π)
end

# The three ADM blocks of a packed `h`, as bit patterns: `===` on these is
# "not moved", which is the claim — not `≈`.
_blocks(h) = (hm = _sym4(h);
              (tt=hm[1, 1], β=(hm[2, 1], hm[3, 1], hm[4, 1]),
               γ=(hm[2, 2], hm[3, 2], hm[4, 2], hm[3, 3], hm[4, 3], hm[4, 4])))

function moved(h, h′, Π, Π′)
    a, b = _blocks(h), _blocks(h′)
    m = Symbol[]
    a.tt === b.tt || push!(m, :tt)
    a.β === b.β || push!(m, :β)
    a.γ === b.γ || push!(m, :γ)
    Π === Π′ || push!(m, :Π)
    return m
end

# The projection's output is a metric `metric_quantities` accepts — it does
# not throw and returns a real, positive lapse and volume element — and every
# ADM quantity is inside its range, to the relative tolerance `rtol` that the
# type's rounding of the reassembled state allows.
function in_ranges(h, Π, bd; rtol)
    T = eltype(h)
    _, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    λ = eigvals(Symmetric(Matrix(_sym4(h)[2:4, 2:4] + I)))
    s = _adm_split(_sym4(h))
    a = sqrt(s.α²) / sqrt(s.detγ)
    return isfinite(α) && isfinite(sqrtγ) && α > 0 && sqrtγ > 0 &&
           gu4[1, 1] < 0 &&
           bd.α_min * (1 - rtol) ≤ α ≤ bd.α_max * (1 + rtol) &&
           bd.λ_min * (1 - rtol) ≤ minimum(λ) &&
           maximum(λ) ≤ bd.λ_max * (1 + rtol) &&
           sqrt(s.bb) ≤ bd.β_max * (1 + rtol) &&
           maximum(x -> abs(a * x), Π) ≤ bd.K_max * (1 + rtol)
end

# The six states `PLAN.md` names, and three more — one per remaining range —
# each with the blocks the projection must move and no others.
function bounds_states(::Type{T}) where {T}
    ks = SM.KerrSchild(one(T), zero(T))
    # A healthy Kerr-Schild point outside the horizon, where one component
    # set to zero still leaves `γ` well inside its range.
    xb = SVector{3,T}(T(2), T(1), T(-1))
    hb, Πb, _ = background_state(ks, zero(T), xb)
    hnan = let m = _sym4(hb)
        _pack10(SMatrix{4,4,T}(ntuple(k -> k in (7, 10) ? T(NaN) : m[k],
                                      Val(16))))          # γ_xy = γ_yx
    end
    Πinf = let m = _sym4(Πb)
        _pack10(SMatrix{4,4,T}(ntuple(k -> k in (2, 5) ? T(Inf) : m[k],
                                      Val(16))))          # Π_tx = Π_xt
    end
    zero_at(v, bad) = map((x, y) -> isfinite(y) ? x : zero(x), v, bad)
    # The core's edge of the step-5 fixture, `r_0 = 2/5` — the deepest state
    # the fixture's core ever holds.
    x0 = T(2 // 5) * normalize(SVector{3,T}(1, 1, 1))
    h0, Π0, _ = background_state(ks, zero(T), x0)
    big = ntuple(v -> v == 6 ? T(500) : T(v - 5) / 20, NC)
    return (
        (name="one negative eigenvalue of γ",
         state=synthetic_state(T; λ=(T(-3 // 10), T(6 // 5), T(2)),
                               htt=T(1 // 5),
                               β=(T(1 // 10), T(-1 // 5), T(1 // 20))),
         moves=[:γ]),
        (name="two negative eigenvalues of γ, det γ > 0",
         state=synthetic_state(T; λ=(T(-2 // 5), T(-7 // 10), T(3 // 2)),
                               htt=T(1 // 5),
                               β=(T(1 // 20), T(1 // 20), T(-1 // 20))),
         moves=[:γ]),
        (name="−g^{tt} < 0 with γ healthy",
         state=synthetic_state(T; λ=(T(4 // 5), one(T), T(6 // 5)),
                               htt=T(3 // 2), β=(T(1 // 10), zero(T), zero(T))),
         moves=[:tt]),
        (name="a NaN in one component of h",
         state=(hnan, Πb), moves=[:γ], expect=(zero_at(hnan, hnan), Πb)),
        (name="an Inf in one component of Π",
         state=(hb, Πinf), moves=[:Π], expect=(hb, zero_at(Πinf, Πinf))),
        (name="healthy Kerr-Schild at the core's edge",
         state=(h0, Π0), moves=Symbol[]),
        (name="|β| above β_max",
         state=synthetic_state(T; λ=(one(T), one(T), one(T)), htt=-one(T),
                               β=(T(12), zero(T), zero(T))),
         moves=[:β]),
        (name="a positive lapse below α_min",
         state=synthetic_state(T; λ=(one(T), one(T), one(T)),
                               htt=one(T) - T(1 // 10^6),
                               β=(zero(T), zero(T), zero(T))),
         moves=[:tt, :Π]),
        (name="(α/√γ)Π above K_max",
         state=synthetic_state(T; λ=(one(T), one(T), one(T)), htt=zero(T),
                               β=(zero(T), zero(T), zero(T)), Π=big),
         moves=[:Π]),
    )
end

@testset verbose = true "The range projection" begin
    T = Float64
    bd = default_bounds(T; M=1, r_gate=1)

    # A range that excludes flat space makes the non-finite repair — which
    # writes Minkowski's components — something the next check moves again;
    # a bound at infinity or at zero is a switch spelled as a number.
    @testset "StateBounds refuses what cannot be a range" begin
        good = (α_min=T(1 // 50), α_max=T(50), λ_min=T(1 // 100),
                λ_max=T(1000), β_max=T(10), K_max=T(100), r_gate=one(T))
        @test StateBounds(T; good...) isa StateBounds{T}
        @test isbitstype(StateBounds{T})
        for bad in ((α_min=T(2),), (α_max=T(1 // 2),), (α_min=zero(T),),
                    (λ_min=T(3 // 2),), (λ_max=T(1 // 2),), (β_max=zero(T),),
                    (K_max=zero(T),), (r_gate=zero(T),), (λ_max=T(Inf),))
            @test_throws ArgumentError StateBounds(T; merge(good, bad)...)
        end
        @test default_bounds(T; M=2, r_gate=1).K_max == 50
    end

    # A projection that "repairs" a state by moving more than the quantity
    # that is wrong is a reset in disguise; one whose second application
    # moves anything is not a projection; one that returns a state the
    # right-hand side's own algebra refuses has repaired nothing.
    @testset "each synthetic state is repaired, and only what is wrong moves" begin
        for (; name, state, moves) in bounds_states(T)
            h, Π = state
            h′, Π′, hit, nonfinite = bounds_project(h, Π, bd)
            @testset "$name" begin
                @test hit == !isempty(moves)
                @test nonfinite == (!all(isfinite, h) | !all(isfinite, Π))
                @test sort(moved(h, h′, Π, Π′)) == sort(moves)
                @test in_ranges(h′, Π′, bd; rtol=1e-9)
                # Bitwise idempotent, and the flag with it.
                h″, Π″, hit2, nf2 = bounds_project(h′, Π′, bd)
                @test h″ === h′
                @test Π″ === Π′
                @test !hit2
                @test !nf2
            end
        end
        rows = bounds_states(T)
        # The premises, so that each row is the state its name says.
        s1 = _adm_split(_sym4(rows[2].state[1]))
        @test s1.detγ > 0                      # det γ does not see it
        λ2 = eigvals(Symmetric(Matrix(_sym4(rows[2].state[1])[2:4, 2:4] + I)))
        @test count(<(0), λ2) == 2
        @test _adm_split(_sym4(rows[3].state[1])).α² < 0
        @test_throws DomainError metric_quantities(_sym4(rows[3].state[1]))
        # The non-finite rows go to Minkowski in the offending component and
        # leave every other bit where it was.
        for k in (4, 5)
            h′, Π′, _, _ = bounds_project(rows[k].state..., bd)
            @test h′ === rows[k].expect[1]
            @test Π′ === rows[k].expect[2]
        end
        # Raising a positive lapse keeps `(α/√γ)Π`, the term `∂_t h` sees.
        h, Π = rows[8].state
        h′, Π′, _, _ = bounds_project(h, Π, bd)
        α0 = sqrt(_adm_split(_sym4(h)).α²)
        α1 = sqrt(_adm_split(_sym4(h′)).α²)
        @test α1 ≈ bd.α_min
        @test α1 .* Π′ ≈ α0 .* Π rtol = 1e-12
        # And a cap keeps the direction of Π.
        h, Π = rows[9].state
        _, Π′, _, _ = bounds_project(h, Π, bd)
        @test maximum(abs, Π′) ≈ bd.K_max
        @test Π′ ≈ Π * (bd.K_max / maximum(abs, Π)) rtol = 1e-14
    end

    # A range drawn too tight — or a test with no slack — would move a
    # healthy hole's data and make every run with the projection on a
    # different run from the one without it.
    @testset "the identity on every background off its singular set" begin
        for (; name, bg) in gh_backgrounds(T), x in gh_points(T, 2)
            h, Π, _ = gh_state(bg, GH_TIME, x)
            h′, Π′, hit, _ = bounds_project(h, Π, bd)
            @test !hit
            @test h′ === h
            @test Π′ === Π
        end
        # And inside the fixture's layer and core, down to `r_0`, where the
        # Kerr-Schild data is steepest: the defaults are far outside it.
        ks = SM.KerrSchild(one(T), zero(T))
        for r in (T(2 // 5), T(1 // 2), T(23 // 20)),
            n in (SVector{3,T}(0, 0, 1), SVector{3,T}(1, 1, 1),
                  SVector{3,T}(-2, 1, 3))

            h, Π, _ = background_state(ks, zero(T), r * normalize(n))
            h′, Π′, hit, _ = bounds_project(h, Π, bd)
            @test !hit
            @test h′ === h && Π′ === Π
        end
    end

    # `CODE.md`: `Float32` is desirable and recorded either way. The algebra
    # is the same code at another type; what a `Float32` state can represent
    # is not, and the tolerance says so.
    @testset "Float32: the same repairs, as bitwise idempotent" begin
        S = Float32
        bs = default_bounds(S; M=1, r_gate=1)
        for (; name, state, moves) in bounds_states(S)
            h, Π = state
            h′, Π′, hit, _ = bounds_project(h, Π, bs)
            @test hit == !isempty(moves)
            @test sort(moved(h, h′, Π, Π′)) == sort(moves)
            @test in_ranges(h′, Π′, bs; rtol=1e-3)
            h″, Π″, hit2, _ = bounds_project(h′, Π′, bs)
            @test h″ === h′ && Π″ === Π′ && !hit2
        end
    end

    # The eigensolver has to be accurate where the spectrum is degenerate —
    # Kerr-Schild's `γ = δ + (2M/r) l l` has a double eigenvalue, and two
    # eigenvalues clamped to one bound produce another — which is where a
    # closed-form 3×3 solver loses half its digits.
    @testset "sym_eigen3 is accurate at a double eigenvalue" begin
        for λ in ((one(T), one(T), T(6)), (T(1 // 100), T(1 // 100), T(3 // 2)),
                  (T(-3), T(1 // 7), T(1000)))
            h, _ = synthetic_state(T; λ=λ, htt=zero(T),
                                   β=(zero(T), zero(T), zero(T)))
            A = SMatrix{3,3,T}(_sym4(h)[2:4, 2:4] + I)
            μ, V = sym_eigen3(A)
            scale = maximum(abs, λ)
            @test sort(μ) ≈ sort(collect(λ)) atol = 64 * eps(T) * scale
            @test V' * V ≈ I atol = 64 * eps(T)
            @test V * Diagonal(μ) * V' ≈ A atol = 64 * eps(T) * scale
        end
    end
end

@testset verbose = true "The range projection on a mesh" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    case0 = hole_fixture(T; q=q)
    forest = hole_fixture_forest(T, case0; N=8)
    r_gate = default_gate(case0.interior, forest, q)
    bounds = default_bounds(T; M=1, r_gate=r_gate)
    case = with_bounds(case0, bounds)
    int = case.interior

    # A gate an evolved stencil can read puts the clamp's kink into the
    # equations outside the layer; a gate outside `r_1` clamps the evolved
    # region itself; a gate with no hole has no center.
    @testset "the gate is placed below every evolved stencil, or refused" begin
        h, _ = layer_spacing(forest, int, zero(T))
        @test r_gate ≈ int.r_1 - 2 * G * h
        @test check_bounds_gate(forest, int, bounds, q).allowed ≈ int.r_1 - G * h
        shallow = default_bounds(T; M=1, r_gate=int.r_1 - h)
        @test_throws ArgumentError check_bounds_gate(forest, int, shallow, q)
        @test_throws ArgumentError with_bounds(case0,
                                               default_bounds(T; M=1,
                                                              r_gate=2 * int.r_1))
        @test_throws ArgumentError hole_case(T, SM.KerrSchild(one(T), zero(T));
                                             halfwidth=T(5 // 2), r_0=T(2 // 5),
                                             r_1=T(23 // 20), chunk=T(1 // 10),
                                             interior=nothing, bounds=bounds)
        @test_throws ArgumentError with_bounds(case0,
                                               default_bounds(Float32; M=1,
                                                              r_gate=1))
        @test with_bounds(case, nothing).bounds === nothing
    end

    # The kernel must repair what is inside the gate and nothing else,
    # write nothing where it did not fire, and count what it did; and the
    # driver's finiteness check must count the evolved region only, or a
    # `NaN` in the core — the projection's business — ends the run.
    @testset "planted failures: repaired inside the gate, counted, and nothing else moves" begin
        U = FieldSet{T}(forest, 2NC; G=G, centering=vertexcentered(3),
                        backend=CPU())
        acc = BoundsAccounting()
        p = GHProblem(U, GhostSchedule(U, ops), case; q=q, accounting=acc)
        fill_exact!(U, case, zero(T))
        u = statevector(U)
        gather!(u, U)
        A = statearray(u, U)
        N = forest.N
        pts = [(b, i, j, k) for b in 1:nblocks(U), k in 1:N, j in 1:N, i in 1:N]
        rad(pt) = sqrt(sum(abs2, coordinates(U, pt[1],
                                             (pt[2] + G, pt[3] + G, pt[4] + G))))
        pick(lo, hi) = pts[findfirst(pt -> lo < rad(pt) < hi, pts)]
        core = pick(int.r_0 / 4, int.r_0)
        gate = pick((int.r_0 + r_gate) / 2, r_gate)
        layer = pick(r_gate, int.r_1)
        evolved = pick(int.r_1, int.r_1 + 1)
        # A NaN in the core, a γ with a negative eigenvalue inside the gate,
        # and a NaN in the layer outside the gate and one in the evolved
        # region, which the projection must not touch.
        tt, ty, xx = _pairindex(1, 1), _pairindex(1, 3), _pairindex(2, 2)
        A[core[2:4]..., ty, core[1]] = T(NaN)
        A[gate[2:4]..., xx, gate[1]] = T(-3 // 2)     # γ_xx = −1/2
        A[layer[2:4]..., tt, layer[1]] = T(NaN)
        A[evolved[2:4]..., NC + tt, evolved[1]] = T(NaN)
        before = copy(u)
        n = apply_bounds!(p, u, zero(T))
        @test n == 2
        @test (acc.calls, acc.hits, acc.nonfinite) == (1, 2, 1)
        @test acc.r_max ≈ rad(gate)
        @test acc.first_t == 0 && acc.first_r ≈ rad(gate)
        @test all(isfinite, A[core[2:4]..., :, core[1]])
        @test in_ranges(SVector{NC,T}(A[gate[2:4]..., 1:NC, gate[1]]),
                        SVector{NC,T}(A[gate[2:4]..., NC+1:2NC, gate[1]]),
                        bounds; rtol=1e-9)
        @test isnan(A[layer[2:4]..., tt, layer[1]])
        @test isnan(A[evolved[2:4]..., NC + tt, evolved[1]])
        # Every other value keeps its bits: 20 values at each of the two
        # points it fired at may differ, and nothing else does.
        changed = count(i -> !isequal(before[i], u[i]), eachindex(u))
        @test 1 ≤ changed ≤ 2 * 2NC
        B = statearray(before, U)
        for pt in (core, gate)
            B[pt[2:4]..., :, pt[1]] .= A[pt[2:4]..., :, pt[1]]
        end
        @test isequal(before, u)
        # The masked finiteness check sees the evolved NaN and not the
        # layer's; the unmasked one would have seen both.
        @test evolved_nonfinite(p, u, zero(T)) == 1
        @test count(!isfinite, u) == 2
        # A second application finds nothing: the kernel's own idempotence.
        after = copy(u)
        @test apply_bounds!(p, u, zero(T)) == 0
        @test isequal(after, u)
        @test take_chunk!(acc) == (hits=2, nonfinite=1, r_max=acc.r_max)
        @test take_chunk!(acc) == (hits=0, nonfinite=0, r_max=-1.0)
        # A case without bounds is a no-op, and the stage limiter is the
        # same call.
        p0 = GHProblem(U, GhostSchedule(U, ops), case0; q=q)
        A[gate[2:4]..., xx, gate[1]] = T(-3 // 2)
        @test apply_bounds!(p0, u, zero(T)) == 0
        @test A[gate[2:4]..., xx, gate[1]] == T(-3 // 2)
        gh_stage_limiter!(u, nothing, p, zero(T))
        @test acc.hits == 3
    end

    # The control. A projection that wrote back where it did not fire —
    # `g′ − η` for every point, or a flag slot read as a state — would make
    # every run with it on a different run from the one without it by an
    # ulp per point per stage, and every later experiment that compares the
    # two would be comparing roundoff.
    @testset "the control: nothing fires, and the run is bit for bit the run without it" begin
        out0 = gh_hole_run(T, case0; N=8, q=q, t_end=T(3 // 20))
        out1 = gh_hole_run(T, case; N=8, q=q, t_end=T(3 // 20))
        @test out0.bounds === nothing
        acc = out1.bounds
        @test acc isa BoundsAccounting
        @test acc.hits == 0 && acc.nonfinite == 0 && acc.r_max == -1
        # Four stage vectors per step, and the initial data once: the hook
        # is installed in `solve` and runs where RK4 says it does.
        @test out1.nsteps == out0.nsteps
        @test acc.calls == 4 * out1.nsteps + 1
        @test isequal(out1.u, out0.u)
        for (r0, r1) in zip(out0.records, out1.records)
            @test r0.bounds_hits === nothing && r0.bounds_r_max === nothing
            @test r1.bounds_hits == 0 && r1.bounds_nonfinite == 0
            @test r1.bounds_r_max == -1
            @test r0.finite && r1.finite
            for k in (:err_l2, :gauge_l2, :residual, :min_detγ_layer,
                      :min_α_layer, :max_h_layer, :max_Π_layer,
                      :min_detγ_shell, :min_α_shell, :max_h_shell,
                      :max_Π_shell)
                @test getfield(r0, k) === getfield(r1, k)
            end
        end
        # The validity rows are the Kerr-Schild numbers they should be: the
        # layer reaches in to `r_0 = 2/5`, where `α = 1/√6` and `|h| = 5`,
        # and out to `r_1`, where `det γ = 1 + 2/r_1`; the shell is the `G`
        # points outside `r_1`.
        r = out1.records[end]
        @info("the validity rows of the fixture at t = 3/20 M",
              r.min_detγ_layer, r.min_α_layer, r.max_h_layer, r.max_Π_layer,
              r.min_detγ_shell, r.min_α_shell, r.max_h_shell, r.max_Π_shell)
        @test r.min_α_layer ≈ 1 / sqrt(1 + 2 / int.r_0) rtol = 0.1
        @test r.max_h_layer ≈ 2 / int.r_0 rtol = 0.1
        @test r.min_detγ_layer ≈ 1 + 2 / int.r_1 rtol = 0.05
        @test r.min_α_shell ≈ 1 / sqrt(1 + 2 / int.r_1) rtol = 0.05
        @test r.min_detγ_shell ≈ 1 + 2 / (int.r_1 + G * out1.h) rtol = 0.05
        @test r.max_h_shell < r.max_h_layer
    end
end
