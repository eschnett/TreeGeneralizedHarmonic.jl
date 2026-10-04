# The rotating octant (added 2026-10-04): TreeAMR's M12 seam under this
# package's state. TreeAMR proves that a seam copies, prolongs and restricts
# with the signed map it is given; what is claimed here is that the map this
# package gives is the tensor's, that the case, the forest and the problem
# agree on the seam, and that a spinning hole on the rotating octant is the
# hole in the whole box it is the symmetric part of.
#
# Everything here is a spinning Kerr-Schild hole, `a = 3/10`: its `h_tx` and
# `h_ty`, `h_xz` and `h_yz` are nonzero and turn into each other, so a wrong
# entry of the map is a kink at the seam; and no mirror in `x` or `y` is a
# symmetry of it, which is why the octant has to turn rather than reflect. Its
# right-hand side is the kernel `reflection_tests.jl` already compiled.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using Random: Xoshiro
using StaticArrays: SVector
import SpacetimeMetrics as SM

@testset verbose = true "Rotating seam" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    # The suite's hole fixture, at a spin whose ring `r = a` is inside the
    # core and whose horizon `r₊ = 1.954` keeps `r_1` eight spacings inside.
    a = T(3 // 10)
    r_0 = T(7 // 20)
    kw = (; halfwidth=T(5 // 2), r_0=r_0, r_1=T(11 // 10), chunk=T(1 // 10), a=a)
    oct = kerr_schild_case(T; kw..., octant=:rotating)
    state(forest; rotation=state_rotation(forest)) =
        FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                    parity=state_parity(forest), rotation=rotation)

    # The owned point of a uniform field set at position `x`: its block and its
    # stored index.
    function point(U, x)
        f = U.forest
        for b in 1:nleaves(f)
            o = block_origin(T, f, f.leaves[b])
            i = ntuple(d -> round(Int, (x[d] - o[d]) / spacing(T, f, f.leaves[b])), 3)
            all(d -> 0 ≤ i[d] < f.N, 1:3) && return b, i .+ (1 .+ U.G)
        end
        return nothing
    end

    @testset "the state's rotation is the tensor's" begin
        # Guards the table every seam transfer turns the state with: a
        # component sent to the wrong partner, or with the wrong sign, is a
        # solution with a kink at the seam. Checked as a table and against
        # the solution itself, `u(Rp) = Q u(p)` for the spinning hole.
        f = gh_forest(T, oct; N=8, roots=1)
        r = state_rotation(f)
        #         tt  tx  ty tz  xx  xy  xz  yy  yz  zz
        @test r[1:10] == [1, -3, 2, 4, 8, -6, -9, 5, 7, 10]
        @test r[11:20] == sign.(r[1:10]) .* (abs.(r[1:10]) .+ 10)
        @test state_rotation(f; copies=4)[31:40] == sign.(r[1:10]) .* (abs.(r[1:10]) .+ 30)
        @test identity_rotation(f, 3) == [1, 2, 3]
        # Without a seam there is nothing to declare.
        plain = gh_forest(T, minkowski_case(T; L=1, ε_KO=0, γ0=1, γ2=0); N=8, roots=1)
        @test state_rotation(plain) === nothing && identity_rotation(plain, 3) === nothing
        rng = Xoshiro(4)
        worst = zero(T)
        for _ in 1:20
            p = SVector(2rand(rng, T) + 1 // 2, 2rand(rng, T) + 1 // 2, 2rand(rng, T) - 1)
            u = TreeGeneralizedHarmonic.state_tuple(oct.background, zero(T), p)
            uR = TreeGeneralizedHarmonic.state_tuple(oct.background, zero(T),
                                                     SVector(-p[2], p[1], p[3]))
            worst = max(worst, maximum(v -> abs(uR[v] - sign(r[v]) * u[abs(r[v])]), 1:20))
        end
        @test worst ≤ 16 * eps(T)
    end

    @testset "the case, the forest and the problem agree on the seam" begin
        # Guards the ways a seam can be declared wrong: on a periodic or
        # reflecting dimension, between unequal widths, about a hole that is
        # not its own image, and on a mesh the case does not describe.
        mk(; kw...) = GHCase(T, SM.Minkowski(); periodic=(false, false, false),
                             ε_KO=0, γ0=1, γ2=0, kw...)
        cube = ntuple(_ -> (zero(T), one(T)), 3)
        @test_throws "dimension 1 is periodic" GHCase(
            T, SM.Minkowski(); box=cube, periodic=(true, false, false),
            rotating=(1, 2), ε_KO=0, γ0=1, γ2=0)
        @test_throws "dimension 2 is reflecting" mk(
            box=cube, rotating=(1, 2),
            reflecting=((false, false), (true, false), (false, false)))
        @test_throws "need one width" mk(box=((0, 1), (0, 2), (0, 1)), rotating=(1, 2))
        @test_throws "two different dimensions" mk(box=cube, rotating=(1, 1))
        @test mk(box=cube).rotating == (0, 0)
        @test oct.rotating == (1, 2)
        @test oct.reflecting == ((false, false), (false, false), (true, false))
        @test oct.box == ntuple(_ -> (zero(T), T(5 // 2)), 3)
        f = gh_forest(T, oct; N=12, roots=2)
        @test f.rotating == (1, 2) && f.reflecting == oct.reflecting
        # The rotating octant is for the axisymmetric hole, and the mirrored
        # one now names it to a spinning hole.
        @test_throws "spinning about z" kerr_schild_case(
            T; kw..., octant=:rotating, center=(T(1 // 4), 0, 0))
        @test_throws "octant = :rotating" kerr_schild_case(T; kw..., octant=true)
        @test_throws "octant is false, true" kerr_schild_case(T; kw..., octant=:yes)
        # A case with the octant's box and mirror but no seam, on its mesh.
        U = state(f)
        @test_throws "rotating seam (1, 2) and the case none" GHProblem(
            U, GhostSchedule(U, ops), mk(box=oct.box, reflecting=oct.reflecting); q=q)
        # Noise leaves both seam planes alone: they are the same points. (The
        # core is excluded, as the octant runs exclude it: the core rule's data
        # at the corner is not of definite parity in z, and the mirror's
        # projection would change it.)
        fill_exact!(U, oct, zero(T))
        V = deepcopy(U)
        add_noise!(V, Xoshiro(2); amplitude=1e-8, exclude=x -> sum(abs2, x) < r_0^2)
        moved, seam = zero(T), zero(T)
        for b in 1:nleaves(f), I in CartesianIndices(interiorview(U, b, 1))
            J = Tuple(I) .+ G
            x = coordinates(U, b, J)
            δ = maximum(abs, V.work[J..., :, b] .- U.work[J..., :, b])
            x[1] == 0 || x[2] == 0 ? (seam = max(seam, δ)) : (moved = max(moved, δ))
        end
        @test seam == 0 && moved > 9e-9
    end

    @testset "the seam's ghosts are the solution a quarter turn away" begin
        # Guards the map where TreeAMR applies it, before any evolution: every
        # stored point of every block — owned, mirrored across z = 0, turned
        # across the seam, or the hook's — holds the case's exact state at its
        # own position after one ghost fill, to the roundoff of the solution's
        # own symmetry. The control declares every
        # variable to turn into itself, which TreeAMR accepts (it is a signed
        # permutation whose fourth power is the identity, and parity-
        # consistent): the seam's ghosts then hold `h_tx` where `−h_ty` is.
        f = gh_forest(T, oct; N=12, roots=2)
        ghost_error(U) = begin
            fill_exact!(U, oct, zero(T))
            fill_ghosts!(U, GhostSchedule(U, ops); boundary=dirichlet(oct, zero(T)))
            worst, scale = zero(T), zero(T)
            for b in 1:nleaves(f), J in CartesianIndices(size(U.work)[1:3])
                x = coordinates(U, b, Tuple(J))
                sum(abs2, x) < r_0^2 && continue       # the core rule's own data
                e = TreeGeneralizedHarmonic.case_state_tuple(oct.background,
                                                             oct.interior, zero(T), x)
                worst = max(worst, maximum(v -> abs(U.work[J, v, b] - e[v]), 1:20))
                scale = max(scale, maximum(abs, e))
            end
            worst / scale
        end
        good = ghost_error(state(f))
        bad = ghost_error(state(f; rotation=identity_rotation(f, 20)))
        @info "seam ghosts against the exact state: $good relative ($bad with the identity)"
        # Measured 1.3e−14 (60 eps), at the points just outside the core next
        # to the ring singularity, where the solution is 350 and is its own
        # image under the turn only to that roundoff.
        @test good ≤ 256 * eps(T)
        @test bad > 1e-3
    end

    @testset "a spinning Kerr-Schild hole on the rotating octant is the hole in the box" begin
        # Guards the whole chain at once — the map, the seam's two owned
        # planes evolved side by side, the axis turned into itself, the mirror
        # at z = 0, the hook at the outer faces only — together with the hole's
        # machinery: the frozen core and the layer about the axis, the sampled
        # gauge source, the Gaussian `γ0`, the interior radii checked on the
        # octant's blocks, and the horizon finder's sphere turned into the
        # quadrant. The suite's fixture, uniform at `h = 5/48`, as
        # `hole_case(; octant = :rotating)` and as the full box, compared
        # point by point outside the core after two chunks. The box's evolved
        # planes `x = −5/2` and `y = −5/2`, the octant's Dirichlet planes, are
        # five units from every point compared.
        holes = map((:rotating, false)) do octant
            # The horizon on the octant only: TreeAMR's `interpolate` turns the
            # finder's sphere into the quadrant, and the find writes no state.
            hz = octant === false ? nothing : Horizon(T; every=1, N=12, spin=true)
            case = kerr_schild_case(T; kw..., octant=octant, horizon=hz)
            f = gh_forest(T, case; N=12, roots=octant === false ? 4 : 2)
            r = evolve!(T, case; forest=f, q=q, ops=ops, t_end=T(1 // 5))
            scatter!(r.U, r.u)
            r
        end
        o, b = holes
        @test o.U.forest.rotating == (1, 2)
        @test o.nsteps == b.nsteps && o.records[end].dt == b.records[end].dt
        diff, scale = zero(T), zero(T)
        for k in 1:nleaves(o.U.forest), I in CartesianIndices(interiorview(o.U, k, 1))
            J = Tuple(I) .+ o.U.G
            x = coordinates(o.U, k, J)
            sum(abs2, x) < r_0^2 && continue        # the core: stale by design
            kb, Jb = point(b.U, x)
            diff = max(diff, maximum(abs, o.U.work[J..., :, k] .- b.U.work[Jb..., :, kb]))
            scale = max(scale, maximum(abs, o.U.work[J..., :, k]))
        end
        @info "spinning hole on the rotating octant against the box at t = 1/5: " *
              "max|Δ| = $diff, max|u| = $scale"
        @test diff ≤ 64 * eps(T) * scale            # measured 2.6 eps relative
        # The layer's residual is a maximum over the same points by symmetry.
        @test o.records[end].residual ≈ b.records[end].residual rtol = 1e-12
        # And the horizon found through the seam is Kerr's: `J = a M` and
        # `M_irr = √((r₊² + a²)/4)`. Measured at `h = 5/48`, `q = 2`, `N = 12`:
        # `J` 5e−6 off at `t = 0` and 6e−5 at `1/5`, `M_irr` 6e−5 and 2.3e−4 —
        # the box's own find to 1e−10, so the drift is the coarse hole's.
        rp = 1 + sqrt(1 - a^2)
        @test all(r -> r.horizon_success === true, o.records)
        @test all(r -> isapprox(r.J, a; atol=2e-4), o.records)
        @test all(r -> isapprox(r.M_irr, sqrt((rp^2 + a^2) / 4); atol=5e-4), o.records)
    end
end
