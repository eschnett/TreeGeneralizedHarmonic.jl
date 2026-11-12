# The matter term (GH-3, added 2026-11-09).
#
# `CODE.md`, "What this package takes (GH-3)": TreeGRRMHD hands this package a
# matter adapter that, bound to the state's field set, answers `(ρ, S_i, S_ij)` at
# an owned vertex; the right-hand side adds `16π α√γ (T_ab − g_ab T/2)` to `∂ₜΠ`,
# and the ADM monitor subtracts `16πρ` and `8πS_i`. What has to hold is what the
# section's acceptance says: on a static metric that is not a vacuum, with the
# matter its Einstein tensor asks for, the right-hand side and the two constraints
# vanish at order `q`, and without the matter they do not. Then the physical test:
# held fixed, that matter keeps a star static through an evolution ("anti-Cowling":
# the matter frozen and the spacetime evolved, where Cowling's approximation does
# the reverse).
#
# The vacuum kernel being the one of before, bit for bit, is not a claim this file
# can make — it has no "before" — and was checked once by hashing `du` before and
# after the change (`CODE.md`, "What this package takes (GH-3)", Measured).
using KernelAbstractions: CPU
using LinearAlgebra: dot
using StaticArrays: SMatrix, SVector
import SpacetimeMetrics

# The star (`PlummerStar`) and its matter (`SampledMatter`) are `plummer_star.jl`'s.
isdefined(Main, :PlummerStar) || include("plummer_star.jl")

# The suite's star: `M = 1/10`, `R = 1`, in a box of `±2R`, at `γ0 = 1`. That is
# `1/(10M)`, or `γ0 R = 1`, below the onset of a growing mode at `γ0 R ≈ 5–10`
# (`CODE.md`, "Gauge and constraint damping"): flat to `500 M`.

star_case(::Type{T}; λ, ε_KO=zero(T)) where {T} =
    GHCase(T, PlummerStar{T}(1 // 10, 1, λ); box=ntuple(_ -> (T(-2), T(2)), 3),
           periodic=(false, false, false), ε_KO=ε_KO, γ0=one(T), γ2=zero(T))

# --- the runs -------------------------------------------------------------------

# `du` of the exact state with and without the matter, and the ADM monitor's two
# constraints with and without it.
function matter_rhs_run(::Type{T}, case; N, roots, q, simd_width=nothing) where {T}
    forest = gh_forest(T, case; N=N, roots=roots)
    fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    p = GHProblem(fs, GhostSchedule(fs, ops), case; q=q, simd_width=simd_width)
    matter = SampledMatter(T, forest, case.background)
    fill_exact!(fs, case, zero(T))
    u = statevector(fs)
    gather!(u, fs)
    du = fill!(similar(u), T(NaN))
    gh_rhs!(du, u, p, zero(T); matter=matter)
    du0 = fill!(similar(u), T(NaN))
    gh_rhs!(du0, u, p, zero(T))
    adm_constraint!(p, u, zero(T); matter=matter)
    with = constraint_norms(p)
    adm_constraint!(p, u, zero(T))
    without = constraint_norms(p)
    return (; h=minimum_spacing(T, forest), fs, du, du0,
            rhs=volume_weighted_norm(fs, du), rhs0=volume_weighted_norm(fs, du0),
            ham=with.ham_l2, mom=maximum(with.mom_l2),
            ham0=without.ham_l2, mom0=maximum(without.mom_l2))
end

@testset "With its matter, a static star's right-hand side and ADM constraints vanish at order q, and without it they do not" begin
    # Guards the term itself: its sign, its factor of 16π, the assembly of `T_ab`
    # from `(ρ, S_i, S_ij)` with the point's lapse and shift, the packed order, and
    # the adapter's index (an owned index read as a stored one, or the wrong
    # block, is an error of the size of the term). Each of those leaves a residual
    # of the matter's size that no resolution reduces. And the monitor's half:
    # `ℋ − 16πρ`, `ℳ_i − 8πS_i`, and the matter's term in the `∂ₜΠ` the monitor
    # rebuilds `∂ₜ∂ₜg` from — a monitor that forgot that would converge to the
    # term's contribution to `ℋ`, not to zero. The tilted slicing (`λ = 3/10`) is
    # what makes `S_i`, the shift and the momentum constraint nonzero.
    T = Float64
    q = 4
    rows = [matter_rhs_run(T, star_case(T; λ=T(3 // 10)); N=N, roots=2, q=q)
            for N in (8, 12, 16)]
    hs = [r.h for r in rows]
    @test all(r -> !any(isnan, r.du), rows)
    for key in (:rhs, :ham, :mom)
        vals = [getfield(r, key) for r in rows]
        vals0 = [getfield(r, Symbol(key, 0)) for r in rows]
        @info "GH-3: $key of the exact star with and without its matter" hs vals vals0 rate = convergence_rate(hs, vals) rate0 = convergence_rate(hs, vals0)
        # q − 1/2, not q − 1/4: at h = 1/4 … 1/8 across a core of radius 1 the rates
        # are still rising toward q — `du` 3.76, 3.87, 3.92 pair by pair from N = 8
        # to 20, `ℋ` 3.75, 3.87, 3.92, `ℳ` 3.71, 3.85, 3.91 (measured 2026-11-09).
        @test convergence_rate(hs, vals) ≥ q - 1 // 2
        @test abs(convergence_rate(hs, vals0)) < 1 // 4   # the matter's own size
        @test vals0[end] > 100 * vals[end]
    end
end

@testset "With matter, the SIMD lanes are the scalar kernel" begin
    # Guards the lanes' adapter call: one call per lane at the lane's own point,
    # packed into the right lane. A lane given its neighbour's matter is an error
    # of the size of the matter's gradient times `h`, not of roundoff.
    T = Float64
    q = 4
    case = star_case(T; λ=T(3 // 10), ε_KO=T(1 // 2))
    sc = matter_rhs_run(T, case; N=10, roots=2, q=q, simd_width=1)
    ln = matter_rhs_run(T, case; N=10, roots=2, q=q)
    N = sc.fs.forest.N
    S = max(maximum(abs, sc.du0),
            maximum(rhs_scale(sc.fs, b, ntuple(d -> o[d] + sc.fs.G[d], Val(3)), q)
                    for b in 1:nblocks(sc.fs),
                        o in ((1, 1, 1), (N ÷ 2, N ÷ 2, N ÷ 2), (N, 2, N - 1))))
    worst = maximum(abs.(ln.du .- sc.du)) / S
    @info "GH-3: lanes against the scalar kernel with matter" W = default_simd_width(T, CPU(), N) bitwise = isequal(ln.du, sc.du) worst_in_eps = worst / eps(T)
    @test default_simd_width(T, CPU(), N) > 1
    @test worst ≤ 64 * eps(T)
end

# The anti-Cowling run: the star's exact data evolved to `t_end` with its matter
# held fixed, or with none, against the exact solution — the star itself.
function anti_cowling_run(::Type{T}, case; N, roots, q, t_end, with_matter=true) where {T}
    forest = gh_forest(T, case; N=N, roots=roots)
    fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    p = GHProblem(fs, GhostSchedule(fs, ops), case; q=q)
    matter = with_matter ? SampledMatter(T, forest, case.background) : nothing
    fill_exact!(fs, case, zero(T))
    u0 = statevector(fs)
    gather!(u0, fs)
    t_end = T(t_end)
    dt = gh_dt(p, u0; cfl=T(1 // 4))
    nsteps = max(1, ceilint(t_end / dt))
    u1 = gh_solve(p, u0, (zero(T), t_end); dt=t_end / nsteps, matter=matter)
    exact = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
    fill_exact!(exact, case, t_end)
    ue = statevector(exact)
    gather!(ue, exact)
    adm_constraint!(p, u1, t_end; matter=matter)
    c = constraint_norms(p)
    return (; h=minimum_spacing(T, forest), nsteps,
            err=volume_weighted_norm(fs, u1 .- ue), finite=all(isfinite, u1),
            ham=c.ham_l2, mom=maximum(c.mom_l2))
end

@testset "Anti-Cowling: a static star held up by fixed matter stays static, at order q" begin
    # Guards the matter in an evolution — through `gh_solve`'s `matter`, at every
    # stage of every step — and the physics of it: the star's matter is what
    # holds its metric static. Without it the same data fall away from the star
    # at a rate set by `16πρ`, which no resolution reduces; with it the run stays
    # on the star to its truncation error, which converges at `q`.
    #
    # At the recipe's `ε_KO = 1/2`; the error's rate is 3.81 and 4.00 pair by pair
    # from N = 8 to 16, and the constraints' as in the testset above (measured
    # 2026-11-09).
    T = Float64
    q = 4
    case = star_case(T; λ=zero(T), ε_KO=T(1 // 2))    # the star's rest frame
    t_end = 4                                    # two light-crossing times of 2R
    rows = [anti_cowling_run(T, case; N=N, roots=2, q=q, t_end=t_end)
            for N in (8, 12, 16)]
    vac = anti_cowling_run(T, case; N=16, roots=2, q=q, t_end=t_end, with_matter=false)
    hs = [r.h for r in rows]
    errs = [r.err for r in rows]
    @info "GH-3 anti-Cowling: the Plummer star to t = $t_end" hs errs nsteps = [r.nsteps for r in rows] rate = convergence_rate(hs, errs) ham = [r.ham for r in rows] mom = [r.mom for r in rows] vacuum_err = vac.err
    @test all(r -> r.finite, rows)
    @test convergence_rate(hs, errs) ≥ q - 1 // 2
    @test convergence_rate(hs, [r.ham for r in rows]) ≥ q - 1 // 2
    @test convergence_rate(hs, [r.mom for r in rows]) ≥ q - 1 // 2
    @test vac.finite
    @test vac.err > 100 * rows[end].err
end

@testset "The matter adapter is bound once, to the state's own field set" begin
    # Guards the binding: a matter sampled on another mesh answers for points of
    # that mesh, so the adapter's own check refuses it — before any kernel runs.
    # And a bound call allocates nothing, or a device could not run it.
    T = Float64
    case = star_case(T; λ=T(3 // 10))
    forest = gh_forest(T, case; N=8, roots=1)
    other = gh_forest(T, case; N=8, roots=1)
    fs = FieldSet{T}(forest, 20; G=3, centering=vertexcentered(3))
    p = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=6, restriction=6)),
                  case; q=4)
    u = statevector(fs)
    fill_exact!(fs, case, zero(T))
    gather!(u, fs)
    @test_throws ArgumentError gh_rhs!(similar(u), u, p, zero(T);
                                       matter=SampledMatter(T, other, case.background))
    at = SampledMatter(T, forest, case.background)(fs)
    at((1, 1, 1), 1)
    @test (@allocated at((2, 3, 4), 1)) == 0
end
