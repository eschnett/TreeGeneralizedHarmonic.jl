# Step I1: a layer-free interior (`CODE.md`, "A layer-free interior"). The
# smoothed hole, the steered source's signs and derivatives, the frozen
# correction, the data's `C_a = 0`, the rules, and one fixture on a small mesh:
# the octant `[0, 4]³` at `h = 1/8` (`N = 16`, `2³` roots), the steering's ramp
# `[1, 3/2)` four cells inside the horizon (`margin = 4`; the screen's eight need
# `h = 1/16`), `ε = 1` eight cells across. Its price: the right-hand side at the
# new gauge-source kind `:corrected` (and `:anchored`'s, once), the two cheap
# monitors at it, and the smoothed hole's dual passes.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using StaticArrays: SArray, SMatrix, SVector
using LinearAlgebra: det
using Random: Xoshiro
using ForwardDiff
using SIMD: Vec
import SpacetimeMetrics as SM

const TGHl = TreeGeneralizedHarmonic

const ηI1 = SMatrix{4,4,Float64}(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)

# The ADM split of a 4-metric by its inverse, the reference's spelling: `α`, `β^i`,
# the lowered shift and `√γ`.
function adm_ref(g)
    gi = inv(g)
    α = 1 / sqrt(-gi[1, 1])
    β = SVector(gi[1, 2], gi[1, 3], gi[1, 4]) * α^2
    βl = SVector(g[1, 2], g[1, 3], g[1, 4])
    sqrtγ = sqrt(det(g[SVector(2, 3, 4), SVector(2, 3, 4)]))
    return α, β, βl, sqrtγ
end

# The steered source's closed form, written apart from `src/layerfree.jl`'s: the
# Kerr-Schild source's value, and the steering from the 4-metric `g` and the
# smoothed hole's own metric at `x` (not `smoothed_logs`).
function steered_closed(g, x, src, bg)
    D = eltype(g)
    z = zero(SVector{10,D})
    Hk = first(algebraic_gauge_source(src.ks, pack_g(g), z, (z, z, z)))
    α, β, βl, sqrtγ = adm_ref(g)
    gs = SM.metric(bg, SVector{4,D}(0, x[1], x[2], x[3]))
    αs, _, _, sqrtγs = adm_ref(gs)
    r = sqrt(sum(abs2, x))
    ψ = 1 - smoothstep((r - src.r_s0) / (src.r_s1 - src.r_s0))
    Λ = src.logκ + log(αs) + src.p * (log(sqrtγ) - log(sqrtγs)) - log(α)
    B2 = sum(βl .* β)
    Ht = -α * src.μ_α * ψ * Λ - src.μ_β * ψ * B2 / α
    Hi = -src.μ_β * ψ * βl / α
    return Hk + SVector(Ht, Hi[1], Hi[2], Hi[3])
end

# A random smooth metric near the smoothed hole's: its own plus `ε` times a field
# of sines and cosines.
function near_smoothed(bg, rng; ε=1 // 200)
    S = [(A = randn(rng, 4, 4); SMatrix{4,4}(A + A')) for _ in 1:3]
    k = [SVector{4}(randn(rng, 4)) for _ in 1:2]
    return p -> SM.metric(bg, p) +
                ε * (S[1] + sin(sum(k[1] .* p)) * S[2] + cos(sum(k[2] .* p)) * S[3])
end

# `H` and `∂_c H_a` of `src` at `p0` against the closed form and its ForwardDiff
# Jacobian, as the worst error over their size.
function steered_against_closed(src, bg, gfun, p0)
    hfun(p) = pack_g(gfun(p))
    J = ForwardDiff.jacobian(hfun, p0)
    h = hfun(p0)
    ∂ₜh = SVector{10}(J[:, 1])
    ∂h = ntuple(i -> SVector{10}(J[:, i + 1]), 3)
    H, dH = algebraic_gauge_source(src, (p0[2], p0[3], p0[4]), p0[1], h, ∂ₜh, ∂h)
    Hfun(p) = steered_closed(gfun(p), SVector(p[2], p[3], p[4]), src, bg)
    Href = Hfun(p0)
    dHref = ForwardDiff.jacobian(Hfun, p0)'
    scale = max(maximum(abs, Href), maximum(abs, dHref))
    return max(maximum(abs, H - Href), maximum(abs, dH - dHref)) / scale
end

# `C_a` of a gauge-consistent solve, by the package's own monitor at a node.
function node_constraint(res)
    g = TGHl._sym4(res.h) + ηI1
    dgs = (res.∂ₜh, res.∂h...)
    dg = SArray{Tuple{4,4,4}}(TGHl._sym4(dgs[c])[a, b] for a in 1:4, b in 1:4, c in 1:4)
    C = g * gauge_constraint_at_node(g, dg, inv(g) * res.Hl)
    return maximum(abs, C) / (1 + maximum(abs, res.Hl) + maximum(abs, res.∂ₜh))
end

@testset verbose = true "Step I1: a layer-free interior" begin
    T = Float64
    rng = Xoshiro(20261030)
    bg = SmoothedKerrSchild(T; M=1, ε=1)
    ks = SM.KerrSchild(1.0, 0.0)
    srcV = SteeredSource(T; M=1, ε=1, r_s0=1, r_s1=3 // 2, steering=:volume, μ_α=4)

    @testset "the smoothed hole is Kerr-Schild outside ε and a metric inside, and its ranges are the brief's" begin
        # Guards the port of TreeBSSN's `SmoothedKerrSchild` (a = 0): outside ε it
        # must be the Kerr-Schild hole the exterior is compared with, inside a
        # Lorentzian metric with a positive lapse, at rest and static.
        @test isstatic(bg) && !isharmonic(bg)
        @test horizon_min_radius(bg) == 2 && singular_radius(bg) == 0
        for _ in 1:8
            n = SVector{3}(randn(rng, 3))
            x = (1 + 4 * rand(rng)) * n / sqrt(sum(abs2, n))
            p = SVector(0.0, x...)
            @test SM.metric(bg, p) ≈ SM.metric(ks, p) rtol = 1e-14 atol = 1e-15
        end
        @test smoothed_radius(0.0, 1.0) == 35 / 128
        # The data's extremes over the h = 1/16 lattice of r ≤ 3/2 (measured in
        # step I1; the brief's host values between the points are 0.347, 4.88,
        # 1.68 and 8.44).
        αmin, λmax, βmax, Kmax = Inf, 0.0, 0.0, 0.0
        for i in 0:24, j in 0:i, k in 0:j
            x = SVector(i, j, k) / 16
            sum(abs2, x) ≤ 9 / 4 || continue
            h, Π, _ = background_state(bg, 0.0, x)
            s = TGHl._adm_split(TGHl._sym4(h))
            @test s.detγ > 0 && s.α² > 0
            α = sqrt(s.α²)
            γ = SMatrix{3,3}(s.γ[1], s.γ[2], s.γ[3], s.γ[2], s.γ[4], s.γ[5], s.γ[3], s.γ[5],
                             s.γ[6])
            λ, _ = sym_eigen3(γ)
            αmin = min(αmin, α)
            λmax = max(λmax, maximum(λ))
            βmax = max(βmax, sqrt(s.bb))
            Kmax = max(Kmax, maximum(abs, (α / sqrt(s.detγ)) * Π))
        end
        @info "the smoothed data's ranges on the 1/16 lattice, ε = 1" αmin λmax βmax Kmax
        @test αmin ≈ 1 / sqrt(1 + 2 * 128 / 35) rtol = 1e-12
        @test 4.8 < λmax < 4.9 && 1.6 < βmax < 1.7 && 8 < Kmax < 8.5
        # Every range of the guard holds them, so the projection does not fire on them.
        bd = layerfree_bounds(T; r_gate=3 // 2)
        @test bd.α_min < αmin && λmax < bd.λ_max && βmax < bd.β_max && Kmax < bd.K_max
    end

    @testset "the steering drives the lapse to its target and the shift to zero" begin
        # Guards the signs, which no derivative test sees: on a slice with γ fixed
        # (K = 0) and the shift small, C_a = 0 gives the steering's share of ∂_t α as
        # α² μ_α ψ log(α_*/α) — the lapse rising below its target and falling above
        # it — and of ∂_t β^i as −α μ_β ψ β^i. Measured against the same solve without
        # the steering, so the Kerr-Schild source's share cancels.
        I3 = SMatrix{3,3,T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        z3 = zero(SMatrix{3,3,T})
        z = zero(SVector{3,T})
        x = SVector(0.3, 0.2, 0.1)                    # ψ = 1
        lα, lv, _, _ = smoothed_logs(1.0, 1.0, sum(abs2, x))
        srcF = SteeredSource(T; M=1, ε=1, r_s0=1, r_s1=3 // 2, steering=:fixed, κ=1 // 3,
                             μ_α=4, μ_β=1)
        srcN = SteeredSource(T; M=1, ε=1, r_s0=1, r_s1=3 // 2, steering=:none)
        for (src, α_star) in ((srcV, exp(lα - lv)),       # √γ = 1: α_* = α_s/√γ_s
                              (srcF, exp(lα) / 3))
            for α in (0.5 * α_star, 2 * α_star), β in (z, SVector(0.02, -0.01, 0.005))
                r = gauge_consistent_momentum(src, x, 0.0, I3, z3, α, β, (z3, z3, z3), z,
                                              (z, z, z))
                r0 = gauge_consistent_momentum(srcN, x, 0.0, I3, z3, α, β, (z3, z3, z3), z,
                                               (z, z, z))
                δα = r.∂ₜα - r0.∂ₜα
                @test sign(δα) == sign(α_star - α)
                iszero(β) &&
                    @test δα ≈ α^2 * src.μ_α * log(α_star / α) rtol = 1e-12
                if !iszero(β) && src.μ_β > 0
                    δβ = r.∂ₜβ - r0.∂ₜβ
                    @test all(δβ .* β .< 0)
                    @test δβ ≈ -α * src.μ_β * β rtol = 1e-6
                end
            end
        end
        # Outside r_s1 the source is the Kerr-Schild hole's, bit for bit.
        h = pack_g(SM.metric(bg, SVector(0.0, 1.0, 1.2, 0.3)))
        zz = zero(SVector{10,T})
        @test algebraic_gauge_source(srcV, (1.0, 1.2, 0.3), 0.0, h, zz, (zz, zz, zz)) ==
              algebraic_gauge_source(srcV.ks, h, zz, (zz, zz, zz))
    end

    @testset "H and ∂H of the steered source are the forward-mode derivatives of its closed form" begin
        # Guards every chain rule of the steering: ψ's gradient, the target's through
        # α_s and √γ_s in closed form, the state's α, β and √γ through `_adm_rates`,
        # for V at p = 1 and 2, F, and the shift steering. Random smooth metrics near
        # the smoothed data, in the core and on the ramp.
        srcs = (srcV,
                SteeredSource(T; M=1, ε=1, r_s0=1, r_s1=3 // 2, p=2, μ_α=1),
                SteeredSource(T; M=1, ε=3 // 4, r_s0=1, r_s1=3 // 2, steering=:fixed,
                              κ=1 // 3, μ_α=4, μ_β=1))
        worst = 0.0
        for src in srcs
            bgs = SmoothedKerrSchild(T; M=1, ε=src.ε)
            gfun = near_smoothed(bgs, rng)
            for _ in 1:8
                n = SVector{3}(randn(rng, 3))
                x = (0.2 + 1.25 * rand(rng)) * n / sqrt(sum(abs2, n))
                worst = max(worst, steered_against_closed(src, bgs, gfun,
                                                          SVector(rand(rng), x...)))
            end
        end
        @info "the steered source against its closed form" worst
        @test worst ≤ 1e-12
    end

    @testset "the steered source on SIMD lanes is the scalar one, lane by lane" begin
        # Guards the generic spelling the CPU kernel runs on lanes: the lane choice
        # inside and outside ε and the ramp, `min`/`max` on `Vec`, the logarithms.
        gfun = near_smoothed(bg, rng)
        xs = [SVector(0.1, 0.0, 0.05), SVector(0.9, 0.3, 0.1), SVector(1.3, 0.2, 0.0),
              SVector(1.6, 0.4, 0.2)]               # core, ramp, ramp, outside
        hs = [pack_g(gfun(SVector(0.0, x...))) for x in xs]
        dh = [ForwardDiff.jacobian(p -> pack_g(gfun(p)), SVector(0.0, x...)) for x in xs]
        V = Vec{4,T}
        lane(f) = V(ntuple(l -> f(l), 4))
        hv = SVector{10,V}(ntuple(v -> lane(l -> hs[l][v]), 10))
        tv = SVector{10,V}(ntuple(v -> lane(l -> dh[l][v, 1]), 10))
        ∂v = ntuple(i -> SVector{10,V}(ntuple(v -> lane(l -> dh[l][v, i + 1]), 10)), 3)
        xv = (lane(l -> xs[l][1]), lane(l -> xs[l][2]), lane(l -> xs[l][3]))
        Hv, dHv = algebraic_gauge_source(srcV, xv, 0.0, hv, tv, ∂v)
        for l in 1:4
            H, dH = algebraic_gauge_source(srcV, Tuple(xs[l]), 0.0, hs[l],
                                           SVector{10}(dh[l][:, 1]),
                                           ntuple(i -> SVector{10}(dh[l][:, i + 1]), 3))
            scale = max(maximum(abs, H), maximum(abs, dH))
            @test maximum(abs, SVector(ntuple(a -> Hv[a][l], 4)) - H) ≤ 16 * eps() * scale
            @test maximum(abs, SMatrix{4,4}(ntuple(k -> dHv[k][l], 16)) - dH) ≤
                  64 * eps() * scale
        end
    end

    @testset "ΔH is exactly zero outside ε, and the smoothed data satisfy C_a = 0 to roundoff" begin
        # Guards the frozen correction: outside ε the exterior's source must be the
        # baseline's bit for bit (zero through a branch, not a product), and inside
        # the algebraic source plus ΔH must be the smoothed metric's −Γ_a, so that the
        # static data start with C_a = 0 — here pointwise, with the analytic ∂g, at
        # the lattice points of the fixture's mesh.
        c = SVector(0.0, 0.0, 0.0)
        for _ in 1:20
            n = SVector{3}(randn(rng, 3))
            x = (1 + 3 * rand(rng)) * n / sqrt(sum(abs2, n))
            @test all(iszero, frozen_correction_tuple(bg, srcV.ks, 0.0, x, c, 1.0))
        end
        @test all(iszero, frozen_correction_tuple(bg, srcV.ks, 0.0, SVector(1.0, 0, 0), c,
                                                  1.0))
        worst = 0.0
        largest = 0.0
        for i in 0:8, j in 0:i, k in 0:j
            x = SVector(i, j, k) / 8
            sum(abs2, x) < 1 || continue
            h, Π, ∂h = background_state(bg, 0.0, x)
            z = zero(SVector{10,T})
            Hk = first(algebraic_gauge_source(srcV.ks, h, z, (z, z, z)))
            Δ = frozen_correction_tuple(bg, srcV.ks, 0.0, x, c, 1.0)
            Hl = Hk + SVector(Δ[1], Δ[2], Δ[3], Δ[4])
            largest = max(largest, maximum(abs, SVector(Δ[1], Δ[2], Δ[3], Δ[4])))
            worst = max(worst, node_constraint((h=h, ∂ₜh=z, ∂h=∂h, Hl=Hl)))
        end
        @info "C_a of the smoothed data with ΔH, and ΔH's largest value" worst largest
        @test worst ≤ 1e-13
        @test largest > 1                       # the brief's |C_a| = 52 without it
    end

    @testset "pre-collapsed and uncorrected data satisfy C_a = 0, and V vanishes on the data" begin
        # Guards the initial data of the screen's R6 (the lapse scaled by
        # 1 − (1 − κ)ψ, Π from C_a = 0 under the steered source plus ΔH) and R9 (no
        # ΔH: Π from C_a = 0 under the algebraic source alone, a gauge transient of
        # ∂_t α), and that V's steering is zero on the smoothed data, so R3 starts
        # with the baseline's gauge.
        srcF = SteeredSource(T; M=1, ε=1, r_s0=1, r_s1=3 // 2, steering=:fixed, κ=1 // 3)
        src9 = SteeredSource(T; M=1, ε=1, r_s0=1, r_s1=3 // 2, correction=false)
        intF = LayerFreeInterior(T; r_s0=1, r_s1=3 // 2, κ=1 // 3)
        int1 = LayerFreeInterior(T; r_s0=1, r_s1=3 // 2)
        worst = 0.0
        dtα9 = 0.0
        vworst = 0.0
        z = zero(SVector{10,T})
        for i in 0:12, j in 0:i, k in 0:j
            x = SVector(i, j, k) / 8
            sum(abs2, x) < (3 // 2)^2 || continue
            rF = layerfree_momentum(bg, srcF, intF, 0.0, x)
            r9 = layerfree_momentum(bg, src9, int1, 0.0, x)
            worst = max(worst, node_constraint(rF), node_constraint(r9))
            dtα9 = max(dtα9, abs(r9.∂ₜα))
            h, _, ∂h = background_state(bg, 0.0, x)
            HV = first(algebraic_gauge_source(srcV, Tuple(x), 0.0, h, z, (z, z, z)))
            HK = first(algebraic_gauge_source(srcV.ks, h, z, (z, z, z)))
            vworst = max(vworst, maximum(abs, HV - HK) / (1 + maximum(abs, HK)))
        end
        # The pre-collapsed lapse at the center is κ times the smoothed one.
        rc = layerfree_momentum(bg, srcF, intF, 0.0, SVector(0.0, 0.0, 0.0))
        αc = sqrt(TGHl._adm_split(TGHl._sym4(rc.h)).α²)
        @info "C_a of the R6 and R9 data, R9's largest ∂_t α, V on the data" worst dtα9 vworst
        # Roundoff of the solve's inversions at the center, where the smoothed
        # lapse is 0.35 and the collapsed one 0.12 (2.4·10⁻¹³ measured on an
        # EPYC 7532, step I1).
        @test worst ≤ 1e-12
        @test αc ≈ (1 / sqrt(1 + 2 * 128 / 35)) / 3 rtol = 1e-12
        @test dtα9 > 1                          # the brief's 3.4/M, at points
        @test vworst ≤ 1e-13
    end

    @testset "the rules: the gate, the ramp's margin and ε's cells are refused by name" begin
        # Guards the replacements of the layer's radius checks and of
        # `check_bounds_gate`'s rule: r_s1 and the gate eight cells (here `margin`)
        # inside the horizon, ε eight cells across, the gate's rule lifted only by
        # name, and the constructors' refusals.
        mk(; kw...) = kerr_schild_case(T; smoothing=1, halfwidth=4, chunk=1 // 8,
                                       octant=true, gauge_source=srcV, kw...)
        bd = layerfree_bounds(T; r_gate=3 // 2)
        f8 = nothing
        case = mk(interior=LayerFreeInterior(T; r_s0=1, r_s1=3 // 2, margin=4), bounds=bd)
        f8 = gh_forest(T, case; N=16, roots=2)                 # h = 1/8
        @test check_interior_radii(f8, case.interior, case.background, 3).allowed == 3 // 2
        @test check_bounds_gate(f8, case.interior, bd, 4).allowed == 3 // 2
        wide = mk(interior=LayerFreeInterior(T; r_s0=1, r_s1=13 // 8, margin=4), bounds=bd)
        @test_throws "not far enough inside the horizon" check_interior_radii(
            f8, wide.interior, wide.background, 3)
        deep = mk(interior=LayerFreeInterior(T; r_s0=1, r_s1=3 // 2), bounds=bd)
        @test_throws "not far enough inside the horizon" check_interior_radii(
            f8, deep.interior, deep.background, 3)               # margin 8 at h = 1/8
        thin = kerr_schild_case(T; smoothing=1 // 2, halfwidth=4, chunk=1 // 8, octant=true,
                                gauge_source=SteeredSource(T; M=1, ε=1 // 2, r_s0=1,
                                                           r_s1=3 // 2),
                                interior=LayerFreeInterior(T; r_s0=1, r_s1=3 // 2,
                                                           margin=4))
        @test_throws "smoothed ball is not resolved" check_interior_radii(
            f8, thin.interior, thin.background, 3)
        far = layerfree_bounds(T; r_gate=7 // 4)
        @test_throws "gate is too close to the horizon" check_bounds_gate(
            f8, case.interior, far, 4)
        open_gate = mk(interior=LayerFreeInterior(T; r_s0=1, r_s1=3 // 2, margin=4,
                                                  gate_rule=false),
                       bounds=layerfree_bounds(T; r_gate=16))
        @test check_bounds_gate(f8, open_gate.interior, open_gate.bounds, 4) === nothing
        @test_throws "0 < r_s0 < r_s1" LayerFreeInterior(T; r_s0=2, r_s1=1)
        @test_throws "0 < κ ≤ 1" LayerFreeInterior(T; r_s0=1, r_s1=2, κ=2)
        @test_throws "0 < ε < r₊ = 2M" SmoothedKerrSchild(T; M=1, ε=2)
        @test_throws "p ≥ 1 and no factor" SteeredSource(T; ε=1, r_s0=1, r_s1=2, κ=1 // 2)
        @test_throws "a = 0 at rest only" kerr_schild_case(T; smoothing=1, a=1 // 2,
                                                           halfwidth=4, chunk=1)
        @test_throws "not smoothed" kerr_schild_case(
            T; halfwidth=4, chunk=1, octant=true, gauge_source=:algebraic,
            interior=LayerFreeInterior(T; r_s0=1, r_s1=3 // 2, r_core=1 // 4, ρ_a=1))
        @test_throws "singular there" kerr_schild_case(
            T; halfwidth=4, chunk=1, octant=true, gauge_source=:algebraic,
            interior=LayerFreeInterior(T; r_s0=1, r_s1=3 // 2))
        @test_throws "no level floor" mk(interior=LayerFreeInterior(T; r_s0=1, r_s1=3 // 2),
                                         refinement=Refinement(T; refine_tol=1 // 10,
                                                               coarsen_tol=1 // 100,
                                                               maxlevel_cap=2, floor_margin=1))
    end

    # The fixture: the screen's base point (R3) on the small mesh, with and without
    # the guard.
    function fixture(; bounds=layerfree_bounds(T; r_gate=3 // 2), src=srcV, ρ_a=0)
        case = kerr_schild_case(T; smoothing=1, halfwidth=4, chunk=1 // 32, octant=true,
                                gauge_source=src,
                                interior=LayerFreeInterior(T; r_s0=1, r_s1=3 // 2,
                                                           margin=4, ρ_a=ρ_a),
                                bounds=bounds)
        return case, gh_forest(T, case; N=16, roots=2)
    end
    q = 4
    ops = Operators(prolongation=q + 2, restriction=q + 2)

    @testset "a run on which the guard never fires is the run without it, bit for bit" begin
        # Guards the claim every row of the screen is read against: with no hit,
        # the range projection is the identity, so R3 and R3 without its guard are
        # the same run. Two chunks of 1/32 M: on this mesh the smoothed ball's
        # volume falls ninefold by 0.09 M and the run ends at the CFL recheck at
        # 0.125 M (measured in step I1), before the projection fires. Also the
        # record's layer-free rows.
        case, forest = fixture()
        out = evolve!(T, case; forest=forest, q=q, ops=ops, t_end=1 // 16, cfl=1 // 2)
        case0, forest0 = fixture(bounds=nothing)
        out0 = evolve!(T, case0; forest=forest0, q=q, ops=ops, t_end=1 // 16, cfl=1 // 2)
        @test all(r -> r.bounds_hits == 0, out.records)
        @test isequal(out.u, out0.u)
        r = out.records[1]
        @test r.bounds_kinds == (nonfinite=0, spectrum=0, shift=0, lapse=0, momentum=0)
        @test length(r.bounds_hist) == TGHl.BOUNDS_HIST_BINS
        @test r.min_α_layer ≈ 1 / sqrt(1 + 2 * 128 / 35) rtol = 1e-12  # the origin
        @test r.min_α_band !== nothing && r.min_α_band > r.min_α_layer
        @info "the fixture's lapse at 1/16 M: the ball, the ramp, the band" out.records[end].min_α_layer out.records[end].min_α_shell out.records[end].min_α_band
    end

    @testset "on the smoothed data the diagnostics read the data, and C_a is the stencils'" begin
        # Guards `layerfree_diagnostics!`: on the initial data the volume ratio
        # √γ/√γ_s is one and the least lapse is the smoothed hole's, shell by shell;
        # and the gauge monitor at t = 0 reads the stencils' truncation of Γ_a, the
        # same with ΔH (static data under −Γ[g_s]) and without (Π from C_a = 0).
        case, forest = fixture()
        U = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                        parity=state_parity(forest), rotation=state_rotation(forest))
        p = GHProblem(U, GhostSchedule(U, ops), case; q=q)
        fill_exact!(U, case, zero(T))
        u = statevector(U)
        gather!(u, U)
        rows = layerfree_diagnostics!(layerfree_aux(p), similar(u), p, u, zero(T);
                                      edges=[0, 1, 3 // 2, 2, 3, Inf])
        @test all(r -> abs(r.vol_min - 1) ≤ 1e-13, rows)
        @test rows[1].α_min ≈ 1 / sqrt(1 + 2 * 128 / 35) rtol = 1e-12
        @test rows[1].points > 0 && rows[end].stat_l2 < 1e-2
        @info "the diagnostics on the data" rows[1] rows[4]
        gh_constraint!(p, u, zero(T))
        c = constraint_norms(p)
        case9, _ = fixture(src=SteeredSource(T; M=1, ε=1, r_s0=1, r_s1=3 // 2,
                                             correction=false))
        U9 = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                         parity=state_parity(forest), rotation=state_rotation(forest))
        p9 = GHProblem(U9, GhostSchedule(U9, ops), case9; q=q)
        fill_exact!(U9, case9, zero(T))
        u9 = statevector(U9)
        gather!(u9, U9)
        gh_constraint!(p9, u9, zero(T))
        c9 = constraint_norms(p9)
        @info "C_a at t = 0 on the 1/8 fixture, with ΔH and without" c.gauge_l2 c.gauge_linf c9.gauge_l2
        @test maximum(c.gauge_l2) ≈ maximum(c9.gauge_l2) rtol = 1e-8
        @test u9 != u                           # R9's Π is not the static data's
    end

    @testset "the anchor is −ρ_a χ (u − u_s) inside ε and nothing outside" begin
        # Guards the `:anchored` kernel branch: F is evaluated everywhere, and the
        # anchor's term is the only difference from `:none`, toward the background.
        case, forest = fixture(ρ_a=1 // 10)
        case0, _ = fixture()
        @test interior_variant(case.interior) === :anchored
        U = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                        parity=state_parity(forest), rotation=state_rotation(forest))
        sched = GhostSchedule(U, ops)
        fill_exact!(U, case, zero(T))
        u = statevector(U)
        gather!(u, U)
        u .+= 1e-3 .* sin.(1:length(u))       # off the smoothed state
        pa = GHProblem(U, sched, case; q=q)
        pn = GHProblem(U, sched, case0; q=q)
        da, dn = similar(u), similar(u)
        gh_rhs!(da, u, pa, zero(T))
        gh_rhs!(dn, u, pn, zero(T))
        A, N0, S = statearray(da, U), statearray(dn, U), statearray(u, U)
        worst = 0.0
        moved = 0
        for b in 1:size(A, 5), k in 1:16, j in 1:16, i in 1:16
            x = coordinates(U, b, (i + 3, j + 3, k + 3))
            χ = anchor_weight(pa.interior, 0.0, x)
            h, Π, _ = background_state(case.background, 0.0, SVector(x...))
            us = vcat(h, Π)
            for v in 1:20
                want = N0[i, j, k, v, b] - (1 // 10) * χ * (S[i, j, k, v, b] - us[v])
                worst = max(worst, abs(A[i, j, k, v, b] - want) /
                                   (1 + abs(N0[i, j, k, v, b])))
            end
            moved += χ > 0
        end
        @test moved > 0
        @test worst ≤ 1e-12
    end
end
