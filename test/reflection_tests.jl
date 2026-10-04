# Reflecting faces (added 2026-10-02): TreeAMR's M10 mirrors under this
# package's state. TreeAMR proves that a mirror copies, prolongs and restricts
# with the parity it is given; what is claimed here is that the parities this
# package gives are the tensor's, that the case, the forest and the Dirichlet
# hook agree on which faces are which, and that a run on the octant is the
# run on the whole box it is the symmetric part of.
#
# Everything here is flat space with noise — `add_noise!`'s robust-stability
# data — because noise has every component nonzero, so a wrong parity in any
# one of the twenty variables shows; and because flat space's right-hand side
# is one kernel specialisation for every case below.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using Random: Xoshiro
import SpacetimeMetrics as SM

@testset verbose = true "Reflecting faces" begin
    T = Float64
    q = 4
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    oct = minkowski_octant_case(T; L=4, ε_KO=1 // 2, γ0=1, γ2=0)
    state(forest) = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                                parity=state_parity(forest))
    dt_of(forest) = minimum_spacing(T, forest) / (4 * sqrt(T(3)))

    # The largest |value| of the odd (and of the even) variables on the owned
    # wall planes `x_d = 0`.
    function wall_values(U; parity=U.parity)
        f = U.forest
        odd, even = zero(T), zero(T)
        for b in 1:nleaves(f), v in 1:U.nvars, d in 1:3
            block_origin(T, f, f.leaves[b])[d] == 0 || continue
            m = maximum(abs, selectdim(interiorview(U, b, v), d, 1))
            parity[v][d] === OddParity ? (odd = max(odd, m)) : (even = max(even, m))
        end
        return (odd=odd, even=even)
    end

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

    @testset "the state's parities are the tensor's" begin
        # Guards the table every mirror multiplies by: a component odd in a
        # dimension it is even in (or the reverse) is a solution with a kink
        # at the wall, which the runs below would show only as a number that
        # is somewhat off.
        f = gh_forest(T, oct; N=8, roots=1)
        p = state_parity(f)
        @test length(p) == 20 && p[1:10] == p[11:20]
        E, O = EvenParity, OddParity
        #          tt         tx         ty         tz         xx
        @test p[1:5] == [(E, E, E), (O, E, E), (E, O, E), (E, E, O), (E, E, E)]
        #          xy         xz         yy         yz         zz
        @test p[6:10] == [(O, O, E), (O, E, O), (E, E, E), (E, O, O), (E, E, E)]
        @test state_parity(f; copies=4) == repeat(p[1:10], 4)
        @test even_parity(f, 3) == fill((E, E, E), 3)
        # Without a reflecting face there is nothing to declare, and a field
        # set built with `nothing` is exactly what it was before.
        plain = gh_forest(T, minkowski_case(T; L=1, ε_KO=0, γ0=1, γ2=0); N=8, roots=1)
        @test state_parity(plain) === nothing && even_parity(plain, 3) === nothing
    end

    @testset "the case, the forest and the hook agree on the faces" begin
        # Guards the two ways a face can be the wrong kind: a mirror over a
        # periodic dimension, and a mesh that mirrors where the case asks for
        # Dirichlet data (or the reverse).
        @test_throws "both periodic and reflecting" GHCase(
            T, SM.Minkowski(); box=ntuple(_ -> (zero(T), one(T)), 3),
            periodic=(true, false, false),
            reflecting=((true, false), (false, false), (false, false)),
            ε_KO=0, γ0=1, γ2=0)
        @test oct.reflecting == ntuple(_ -> (true, false), 3)
        @test has_outer_face(oct)
        @test !has_outer_face(minkowski_case(T; L=1, ε_KO=0, γ0=1, γ2=0))
        @test dirichlet(oct, zero(T)) !== nothing
        # A box reflecting at both ends of every dimension has no outer face.
        closed = GHCase(T, SM.Minkowski(); box=ntuple(_ -> (zero(T), one(T)), 3),
                        periodic=(false, false, false),
                        reflecting=ntuple(_ -> (true, true), 3), ε_KO=0, γ0=1, γ2=0)
        @test !has_outer_face(closed) && dirichlet(closed, zero(T)) === nothing
        f = gh_forest(T, oct; N=8, roots=1)
        @test f.reflecting == oct.reflecting
        full = GHCase(T, SM.Minkowski(); box=ntuple(_ -> (zero(T), T(4)), 3),
                      periodic=(false, false, false), ε_KO=1 // 2, γ0=1, γ2=0)
        U = state(f)
        @test_throws "the mesh reflects" GHProblem(U, GhostSchedule(U, ops), full; q=q)
    end

    @testset "add_noise! is bounded, seeded and projected onto the parity" begin
        # Guards the robust-stability data: an amplitude it exceeds, a seed it
        # ignores, and an odd component left nonzero on its own wall — a mode
        # the mirrored problem does not have, which TreeAMR does not remove.
        f = hole_forest(T, oct; N=8, roots=2, center=(0, 0, 0), radii=[2], shape=:cube)
        U = state(f)
        add_noise!(U, Xoshiro(7); amplitude=1e-8)
        u = statevector(U)
        gather!(u, U)
        @test maximum(abs, u) ≤ 1e-8
        @test maximum(abs, u) > 9e-9
        # Uniform noise of amplitude A has rms A/√3: here `h_tt`, even in every
        # dimension and so never projected, on 7680 draws (6σ is 3 %).
        htt = reduce(vcat, vec(interiorview(U, b, 1)) for b in 1:nleaves(f))
        @test isapprox(sqrt(sum(abs2, htt) / length(htt)), 1e-8 / sqrt(3); rtol=3e-2)
        w = wall_values(U)
        @test w.odd == 0 && w.even > 9e-9
        V = state(f)
        add_noise!(V, Xoshiro(7); amplitude=1e-8)
        @test isequal(U.work, V.work)
    end

    @testset "the octant is the mirrored box, to roundoff" begin
        # Guards the whole chain at once — the parities, the mirrors, the
        # hook only at the outer faces, the wall plane evolved — against the
        # one problem whose answer is known without a mirror: the box
        # `[−4, 4]³` with Dirichlet data everywhere and the octant's noise
        # extended by parity. The noise is put inside `[0, 1]³` only, so that
        # the box's evolved plane `x = −4`, which the octant has as its
        # Dirichlet plane `x = +4`, stays outside the data's reach. Uniform
        # meshes only: at a refinement boundary vertex centering is not
        # mirror-symmetric (the fine level owns `x = −2`, the coarse one
        # `x = +2`), so a refined octant is a different discretization of
        # the same problem.
        f8 = gh_forest(T, oct; N=8, roots=2)
        boxcase = GHCase(T, SM.Minkowski(); box=ntuple(_ -> (T(-4), T(4)), 3),
                         periodic=(false, false, false), ε_KO=1 // 2, γ0=1, γ2=0)
        fb = gh_forest(T, boxcase; N=8, roots=4)
        Uo = state(f8)
        Ub = FieldSet{T}(fb, 20; G=G, centering=vertexcentered(3))
        add_noise!(Uo, Xoshiro(1); amplitude=1e-8)
        for b in 1:nleaves(f8), I in CartesianIndices(interiorview(Uo, b, 1))
            J = Tuple(I) .+ G
            maximum(coordinates(Uo, b, J)) > 1 && (Uo.work[J..., :, b] .= 0)
        end
        par = Uo.parity
        for b in 1:nleaves(fb), I in CartesianIndices(interiorview(Ub, b, 1))
            J = Tuple(I) .+ G
            x = coordinates(Ub, b, J)
            maximum(abs, x) ≥ 4 && continue
            bo, Jo = point(Uo, abs.(x))
            for v in 1:20
                s = prod(d -> x[d] < 0 && par[v][d] === OddParity ? -1 : 1, 1:3)
                Ub.work[J..., v, b] = s * Uo.work[Jo..., v, bo]
            end
        end
        uo, ub = statevector(Uo), statevector(Ub)
        gather!(uo, Uo)
        gather!(ub, Ub)
        po = GHProblem(Uo, GhostSchedule(Uo, ops), oct; q=q)
        pb = GHProblem(Ub, GhostSchedule(Ub, ops), boxcase; q=q)
        dt = dt_of(f8)
        # 32 steps, `t ≈ 1.15`: the data spread from `[0, 1]³` at speed √3
        # stay inside `[0, 3]³`.
        uo1 = gh_solve(po, uo, (zero(T), 32dt); dt=dt)
        ub1 = gh_solve(pb, ub, (zero(T), 32dt); dt=dt)
        scatter!(Uo, uo1)
        scatter!(Ub, ub1)
        diff = zero(T)
        for b in 1:nleaves(f8), I in CartesianIndices(interiorview(Uo, b, 1))
            J = Tuple(I) .+ G
            bb, Jb = point(Ub, coordinates(Uo, b, J))
            diff = max(diff, maximum(abs, Uo.work[J..., :, b] .- Ub.work[Jb..., :, bb]))
        end
        scale = maximum(abs, uo1)
        @info "octant against the mirrored box: max|Δ| = $diff, max|u| = $scale"
        @test scale > 1e-8                      # the data did not decay away
        @test diff ≤ 64 * eps(T) * scale        # measured 1.1e-15 relative
    end

    @testset "on a refined octant the odd components stay zero on the walls" begin
        # Guards the parities on a hierarchy, where the mirrored ghosts of a
        # block at a refinement boundary are prolonged: an odd variable that
        # starts zero on its wall stays zero there to roundoff, because the
        # mirrored ghosts give `u(−h) = −u(h)` and the centred stencils
        # respect it (not exactly zero: a fused multiply-add keeps a mirrored
        # pair from cancelling bit for bit). The control is the same run with
        # every variable declared even — the mirror then has no sign, and the
        # odd variables leave their walls at the data's own size.
        f = hole_forest(T, oct; N=8, roots=2, center=(0, 0, 0), radii=[2, 1],
                        shape=:cube)
        @test length(forest_levels(f)) == 3
        noisy_run(U) = begin
            add_noise!(U, Xoshiro(3); amplitude=1e-8)
            u = statevector(U)
            gather!(u, U)
            p = GHProblem(U, GhostSchedule(U, ops), oct; q=q)
            u1 = gh_solve(p, u, (zero(T), 20dt_of(f)); dt=dt_of(f))
            scatter!(U, u1)
            maximum(abs, u1)
        end
        U = state(f)
        scale = noisy_run(U)
        w = wall_values(U)
        @info "refined octant after 20 steps: odd on walls $(w.odd), even $(w.even), max|u| $scale"
        @test w.even > 1e-8
        @test w.odd ≤ 16 * eps(T) * scale       # measured 7e-18 relative
        wrong = FieldSet{T}(f, 20; G=G, centering=vertexcentered(3),
                            parity=even_parity(f, 20))
        noisy_run(wrong)
        @test wall_values(wrong; parity=state_parity(f)).odd > 1e-9
    end

    @testset "point-weighted norms weight every point once, and add up by level" begin
        # Guards the norm the robust-stability run reads: on a hierarchy a
        # volume weight is all but the coarsest level's, and `:points` must be
        # the plain mean over owned points — checked against a host loop over
        # `diag` — with the levels partitioning it.
        f = hole_forest(T, oct; N=8, roots=2, center=(0, 0, 0), radii=[2], shape=:cube)
        U = state(f)
        add_noise!(U, Xoshiro(5); amplitude=1e-8)
        u = statevector(U)
        gather!(u, U)
        p = GHProblem(U, GhostSchedule(U, ops), oct; q=q)
        adm_constraint!(p, u, zero(T))
        hostsum, hostmax, npts = zero(T), zero(T), 0
        for b in 1:nleaves(f)
            c = interiorview(p.diag, b, TreeGeneralizedHarmonic.DIAG_HAM)
            hostsum += sum(abs2, c)
            hostmax = max(hostmax, maximum(abs, c))
            npts += length(c)
        end
        n = constraint_norms(p; weighting=:points)
        @test evolved_volume(p; weighting=:points) == npts
        @test n.ham_l2 ≈ sqrt(hostsum / npts) rtol = 1e-12
        @test n.ham_linf == hostmax
        v = constraint_norms(p)
        @test !(v.ham_l2 ≈ n.ham_l2)            # the two weights differ here
        lv = level_constraint_norms(p)
        @test [x.level for x in lv] == [0, 1]
        @test sum(x.points for x in lv) == npts
        @test sum(x.ham_l2^2 * x.points for x in lv) ≈ n.ham_l2^2 * npts rtol = 1e-12
        @test maximum(x.ham_linf for x in lv) == n.ham_linf
    end

    @testset "a noisy octant run restarts bit for bit" begin
        # Guards the parity through the checkpoint file and the perturbation
        # through the restart: the noise is the first call's initial data, and
        # a restart must take its state from the file and never perturb again.
        f = gh_forest(T, oct; N=8, roots=1)
        perturb(U) = add_noise!(U, Xoshiro(11); amplitude=1e-8)
        common = (q=q, ops=ops, t_end=T(3 // 20), chunk=T(1 // 20), perturb=perturb,
                  adm_every=1)
        ref = evolve!(T, oct; forest=deepcopy(f), common...)
        pre = joinpath(mktempdir(), "oct")
        ck = (checkpoint_path_prefix=pre, max_walltime_seconds=1e-9,
              checkpoint_sync_to_disk=false)
        out = evolve!(T, oct; forest=deepcopy(f), common..., ck...)
        jobs = 1
        while !out.finished
            out = evolve!(T, oct; common..., ck..., restart_file=latest_checkpoint(pre))
            jobs += 1
        end
        @test jobs == 3
        @test isequal(out.u, ref.u)
        @test isequal(out.records, ref.records)
        @test ref.records[end].ham_l2 > 0
    end

    @testset "a Kerr-Schild hole on the octant is the hole in the box" begin
        # Guards the hole's own machinery under the mirrors: the frozen core
        # and the damping layer straddling the three symmetry planes, the
        # sampled gauge source, the Gaussian `γ0` about the corner, the
        # interior radii checked on an octant's blocks. The suite's fixture
        # (`q = 2`, `r_0 = 2/5`, `r_1 = 23/20`, half-width `5/2`), uniform at
        # `h = 5/48`, as `hole_case(; octant = true)` and as the full box,
        # compared point by point outside the core after two chunks. The
        # box's evolved plane `x = −5/2`, the octant's Dirichlet plane
        # `x = +5/2`, is five units from every point compared, so it is not
        # what is measured.
        q2 = 2
        ops2 = Operators(prolongation=q2 + 2, restriction=q2 + 2)
        kw = (; halfwidth=T(5 // 2), r_0=T(2 // 5), r_1=T(23 // 20), chunk=T(1 // 10))
        holes = map((true, false)) do octant
            case = kerr_schild_case(T; kw..., octant=octant)
            f = gh_forest(T, case; N=12, roots=octant ? 2 : 4)
            r = evolve!(T, case; forest=f, q=q2, ops=ops2, t_end=T(1 // 5))
            scatter!(r.U, r.u)
            r
        end
        o, b = holes
        # A hole that is not its own mirror image has no octant.
        @test_throws "its own mirror image" kerr_schild_case(T; kw..., a=T(1 // 2),
                                                              octant=true)
        @test_throws "its own mirror image" kerr_schild_case(T; kw..., octant=true,
                                                              center=(T(1 // 4), 0, 0))
        @test o.U.forest.reflecting == ntuple(_ -> (true, false), 3)
        @test o.nsteps == b.nsteps && o.records[end].dt == b.records[end].dt
        diff, scale = zero(T), zero(T)
        for k in 1:nleaves(o.U.forest), I in CartesianIndices(interiorview(o.U, k, 1))
            J = Tuple(I) .+ o.U.G
            x = coordinates(o.U, k, J)
            sum(abs2, x) < (2 // 5)^2 && continue      # the core: stale by design
            kb, Jb = point(b.U, x)
            diff = max(diff, maximum(abs, o.U.work[J..., :, k] .- b.U.work[Jb..., :, kb]))
            scale = max(scale, maximum(abs, o.U.work[J..., :, k]))
        end
        @info "Kerr-Schild octant against the box at t = 1/5: max|Δ| = $diff, max|u| = $scale"
        @test diff ≤ 64 * eps(T) * scale
        # The layer's residual is a maximum over the same points by symmetry.
        @test o.records[end].residual ≈ b.records[end].residual rtol = 1e-12
    end
end
