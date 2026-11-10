# The verbs a script that owns the time loop calls (GH-4, added 2026-11-09).
#
# `CODE.md`, "The verbs (GH-4)": TreeGRRMHD's coupled run is a script that builds one
# IMEXRungeKutta integrator over a state vector of several field sets — this
# package's and the matter's — and calls each package's `prepare!`, `rhs!`,
# limiters and speed on its own segment of it. What has to hold is that those verbs
# are this package's run: the script's integrator, over a segment at an offset in a
# longer vector, is `gh_solve`'s bit for bit, and nothing the verbs do reaches
# outside their segment.
#
# The star and its matter adapter are `matter_tests.jl`'s, so that no metric type is
# compiled twice (its `EinsteinTensor` costs 75 s).

using KernelAbstractions: CPU
using Random: Xoshiro
using TreeGeneralizedHarmonic: NC, _pairindex, ceilint
const GHV = TreeGeneralizedHarmonic
isdefined(Main, :star_case) || include("matter_tests.jl")

# The script's four callbacks, as TreeGRRMHD's step-11 sketch writes them, with a
# stand-in for the other package: a second field set whose right-hand side is zero
# and which has no limiter, so that its segment must come out with the bits it went
# in with. `p.s` is the two segments, this package's second, so that its segment
# starts at an offset.
function verbs_f_exp!(du, u, p, t)
    GHV.prepare!(p.gh, view(u, p.s[2]), t)
    GHV.rhs!(view(du, p.s[2]), p.gh, t; matter=p.matter)
    fill!(view(du, p.s[1]), zero(eltype(du)))
    return nothing
end
verbs_stage_limit!(u, _, p, t) = GHV.stage_limit!(view(u, p.s[2]), p.gh, t)
verbs_step_limit!(u, _, p, t) = GHV.step_limit!(view(u, p.s[2]), p.gh, t)

@testset "A script over two sets, calling the verbs on this package's segment, is gh_solve's run" begin
    # Guards GH-4's verbs as a script calls them: `prepare!` and `rhs!` on a
    # segment view at an offset in a longer state vector (the scatter's source, the
    # kernel's `du` and its SIMD stores through a pointer into the view), the
    # matter reaching `rhs!`, and the limiters and the speed on the segment. A
    # verb that wrote outside its segment, or a kernel that read a view's parent at
    # the wrong offset, changes bits here; a split of `gh_rhs!` that is not its two
    # halves changes them too.
    T = Float64
    q = 4
    IRK = GHV.IRK
    case = star_case(T; λ=T(3 // 10), ε_KO=T(1 // 2))
    forest = gh_forest(T, case; N=8, roots=2)
    fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    gh = GHProblem(fs, GhostSchedule(fs, ops), case; q=q)
    matter = SampledMatter(T, forest, case.background)

    other = FieldSet{T}(forest, 3; G=1, centering=cellcentered(3))
    sets = (other, GHV.evolved_sets(gh)...)
    @test GHV.evolved_sets(gh) === (fs,)
    s = (staterange(sets, 1), staterange(sets, 2))
    @test first(s[2]) > 1
    u = statevector(sets)
    view(u, s[1]) .= rand(Xoshiro(20261109), T, length(s[1]))
    other0 = copy(view(u, s[1]))
    @test GHV.initial_state!(view(u, s[2]), gh, zero(T)) isa SubArray

    # The reference: the same problem on a vector of its own.
    fill_exact!(fs, case, zero(T))
    u0 = statevector(fs)
    gather!(u0, fs)
    @test isequal(view(u, s[2]), u0)

    # The two halves on the segment are `gh_rhs!` on its own vector, and `rhs!`
    # writes its segment of `du` only.
    @test GHV.prepare!(gh, view(u, s[2]), zero(T)) === gh
    w = copy(fs.work)
    du = fill!(similar(u), T(NaN))
    GHV.rhs!(view(du, s[2]), gh, zero(T); matter=matter)
    @test all(isnan, view(du, s[1]))
    du0 = similar(u0)
    gh_rhs!(du0, u0, gh, zero(T); matter=matter)
    @test isequal(view(du, s[2]), du0)
    @test isequal(fs.work, w)                    # gh_rhs!'s fill is prepare!'s
    # The metric adapter of a prepared state is `ADMMetric(p, u, t)`'s.
    adm = ADMMetric(GHV.prepare!(gh, view(u, s[2]), zero(T)))
    @test adm.work === fs.work && isequal(fs.work, w)

    # The speed of the segment, and the step count a script makes of it.
    λ = max_speed(gh, view(u, s[2]), zero(T))
    @test λ == max_speed(gh, u0, zero(T))
    t_end = T(1 // 2)
    nsteps = ceilint(t_end * λ / (T(1 // 4) * minimum_spacing(T, forest)))
    dt = t_end / nsteps

    p = (; gh, matter, s)
    integ = IRK.init(IRK.IMEXProblem(verbs_f_exp!, nothing, u, (zero(T), t_end), p),
                     IRK.RK4(); dt=dt, stage_limiter=verbs_stage_limit!,
                     step_limiter=verbs_step_limit!, alias_u0=true)
    IRK.solve!(integ)
    @test integ.u === u
    ref = gh_solve(gh, u0, (zero(T), t_end); dt=dt, matter=matter)
    @test all(isfinite, ref) && ref != u0
    @test isequal(view(u, s[2]), ref)
    @test isequal(view(u, s[1]), other0)
end

@testset "On a segment, the limiters, the speed and the initial state are a vector of its own's" begin
    # Guards the integrator-free limiters where they write: the range projection
    # (a kernel over `statearray` of the segment, and a fold over `diag`) and the
    # `:pasted` variant's paste, on planted failures in a segment at an offset;
    # `step_limit!` being `stage_limit!`; the speed's refusal of a non-finite
    # evolved value in the segment; and `initial_state!` through the problem's own
    # core rule and limiter. The hole fixture at `q = 2`, which the suite has
    # compiled already; nothing is evolved.
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    case0 = hole_fixture(T; q=q, variant=:pasted)
    forest = hole_fixture_forest(T, case0; N=8)
    r_gate = default_gate(case0.interior, forest, q)
    case = with_bounds(case0, default_bounds(T; M=1, r_gate=r_gate))
    int = case.interior
    U = FieldSet{T}(forest, 2NC; G=G, centering=vertexcentered(3))
    p = GHProblem(U, GhostSchedule(U, ops), case; q=q)

    other = FieldSet{T}(forest, 3; G=1, centering=cellcentered(3))
    sets = (other, U)
    s = staterange(sets, 2)
    u = statevector(sets)
    view(u, staterange(sets, 1)) .= rand(Xoshiro(4), T, length(staterange(sets, 1)))
    other0 = copy(view(u, staterange(sets, 1)))
    useg = view(u, s)
    GHV.initial_state!(useg, p, zero(T))

    # The reference, as `evolve!` makes it: the exact state through the problem's
    # core rule, then the limiter of a vector of its own.
    fill_exact!(U, case, zero(T); interior=p.interior)
    u1 = statevector(U)
    gather!(u1, U)
    gh_limiter!(u1, nothing, p, zero(T))
    @test isequal(useg, u1)
    exact = copy(u1)

    # Planted: a `γ` with a negative eigenvalue inside the gate, and a value off
    # the exact state in the paste's ball, in both vectors.
    N = forest.N
    pts = [(b, i, j, k) for b in 1:nblocks(U), k in 1:N, j in 1:N, i in 1:N]
    rad(pt) = sqrt(sum(abs2, coordinates(U, pt[1], (pt[2] + G, pt[3] + G, pt[4] + G))))
    pick(lo, hi) = pts[findfirst(pt -> lo < rad(pt) < hi, pts)]
    gate = pick((int.r_0 + r_gate) / 2, r_gate)
    ball = pick(r_gate, int.r_1)
    evolved = pick(int.r_1 + 1 // 2, int.r_1 + 1)
    xx, tt = _pairindex(2, 2), _pairindex(1, 1)
    for v in (useg, u1)
        A = statearray(v, U)
        A[gate[2:4]..., xx, gate[1]] = T(-3 // 2)              # γ_xx = −1/2
        A[ball[2:4]..., NC + tt, ball[1]] += T(1 // 10)
    end
    GHV.stage_limit!(useg, p, zero(T))
    gh_limiter!(u1, nothing, p, zero(T))
    @test isequal(useg, u1)
    @test statearray(useg, U)[gate[2:4]..., xx, gate[1]] != T(-3 // 2)
    @test statearray(useg, U)[ball[2:4]..., NC + tt, ball[1]] ≈
          statearray(exact, U)[ball[2:4]..., NC + tt, ball[1]]
    @test isequal(view(u, staterange(sets, 1)), other0)
    for v in (useg, u1)
        statearray(v, U)[ball[2:4]..., NC + tt, ball[1]] += T(1 // 10)
    end
    GHV.step_limit!(useg, p, zero(T))
    GHV.stage_limit!(u1, p, zero(T))
    @test isequal(useg, u1)

    # The speed of the segment is the vector's, and a non-finite value at an
    # evolved point of the segment is refused before a step is made of it.
    @test max_speed(p, useg, zero(T)) == max_speed(p, u1, zero(T))
    statearray(useg, U)[evolved[2:4]..., tt, evolved[1]] = T(NaN)
    @test_throws ArgumentError max_speed(p, useg, zero(T))
    @test isequal(view(u, staterange(sets, 1)), other0)
end
