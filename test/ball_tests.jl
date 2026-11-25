# The cube and the shells (step B5): the state on TreeMultiPatch's
# thirteen-patch ball, interpolated across its glues, with the high glued face
# planes in the state's appendix (TreeAMR's M20). `CODE.md`, "The cube and the
# shells".

# TreeMultiPatch and IMEXRungeKutta through the package's own bindings, as
# `stepping_tests.jl` takes the integrator: both are dependencies of the package
# and not of the test environment.
using KernelAbstractions: @kernel, @index
using ForwardDiff: ForwardDiff
using SpacetimeMetrics: Minkowski
using StaticArrays: SVector
using TreeGeneralizedHarmonic: _background_state, background_state, gh_rhs_kernel!,
                               gh_rhs_kernel_args
const TMP_B5 = TreeGeneralizedHarmonic.TreeMultiPatch
const IRK_B5 = TreeGeneralizedHarmonic.IRK

# The small ball every testset runs on: the cube `[−1, 1]³` of two roots a side,
# one root along the radius in each shell, the middle sphere at `5/2` (its
# corner ghosts at `N = 8` reach `√3 (1 + 6/16) = 2.38`) and the outer one at `4`.
ball_layout_b5(T) = TMP_B5.ThirteenPatchBall{T}(1, 5 // 2, 4, (2, 1, 1))

# The points of a state on the ball, its appendix included, with their patch:
# `f(patch, value)` for every owned entry.
function foreach_owned(f, p, u)
    a = statearray(u, p.U; appendix=true)
    N = p.U.forest.N
    pat = block_patches(p.U.forest)
    for b in 1:nblocks(p.U), v in 1:p.U.nvars, I in CartesianIndices((N + 1, N + 1, N + 1))
        isowned(a, Tuple(I)..., v, b) && f(pat[b], a[Tuple(I)..., v, b])
    end
    return nothing
end

@kernel function ball_owner_kernel!(tid)
    I = @index(Global, NTuple)
    tid[I[end]] = Threads.threadid()
end

@testset verbose = true "The cube and the shells" begin
    T = Float64

    # Guards the integrator's partition on a state with an appendix: the
    # appendix follows every core, so a partition of the cores alone misses it
    # (IMEXRungeKutta refuses that) and one that gave it to a single thread
    # would combine every block's high planes off its owner. Each thread owns
    # its blocks' cores and their appendix, exactly once.
    @testset "the partition gives each block's appendix to its owner" begin
        layout = TMP_B5.ThirteenPatchBall{T}(1, 5 // 2, 4, (2, 1, 1))
        forest = TMP_B5.layout_forest(T, layout; N=8)
        fs = FieldSet{T}(forest, 20; G=3, centering=vertexcentered(3),
                         glue=:interpolate)
        u = statevector(fs)
        L = forest.N^3 * fs.nvars
        ncore = L * nblocks(fs)
        @test length(u) > ncore
        part = state_partition(fs, u)
        @test length(part) == Threads.nthreads()
        @test sort(reduce(vcat, [collect(r) for p in part for r in p])) == 1:length(u)

        # Every appendix entry of block `b` lies in the range of `b`'s thread:
        # write `b` into each and read the ranges back.
        a = statearray(u, fs; appendix=true)
        N = forest.N
        for b in 1:nblocks(fs), v in 1:fs.nvars,
            I in CartesianIndices((N + 1, N + 1, N + 1))
            any(==(N + 1), Tuple(I)) || continue
            isowned(a, Tuple(I)..., v, b) && (a[Tuple(I)..., v, b] = b)
        end
        chunks = TreeAMR.threadchunks(nblocks(fs))
        for (c, r) in enumerate(chunks)
            @test part[c][1] == (first(r) - 1) * L + 1:last(r) * L
            @test all(b -> b in r, u[part[c][2]])
        end
        tid = zeros(Int, nblocks(fs))
        map_blocks!(ball_owner_kernel!, fs, tid; appendix=false)
        offset = Threads.threadpoolsize(:interactive)
        if Threads.nthreads() > 1
            for (c, r) in enumerate(chunks), b in r
                @test tid[b] == offset + c
            end
        end

        # IMEXRungeKutta takes it, and its stage arithmetic by owner is the
        # broadcast's to the bit.
        u .= range(-1, 1; length=length(u))
        f!(du, u, p, t) = (du .= -u .+ sin(t); nothing)
        run(partition) = (integ = IRK_B5.init(IRK_B5.IMEXProblem(f!, nothing, copy(u),
                                                                 (zero(T), one(T)), nothing),
                                              IRK_B5.RK4(); dt=T(1 // 8),
                                              partition=partition);
                          IRK_B5.solve!(integ); integ.u)
        @test isequal(run(part), run(nothing))
    end

    # Guards the plumbing of a run on the ball: the positions through the atlas,
    # the hook on the outer sphere, the appendix in the right-hand side and the
    # integrator, and the time step. Flat space is `h = Π = 0` in Cartesian
    # components on every patch, and every operator here is linear in a state
    # that is zero, so anything but exact zeros is a term that does not vanish.
    @testset "Minkowski stays Minkowski on every patch, the appendix included" begin
        layout = ball_layout_b5(T)
        case = ball_case(T, Minkowski(); layout=layout, ε_KO=T(1 // 2), γ0=one(T),
                         γ2=zero(T))
        p = ball_problem(T, case, layout; N=8, q=4)
        @test p.ball isa BallGeometry
        u = statevector(p.U)
        u .= NaN
        TreeGeneralizedHarmonic.initial_state!(u, p, zero(T))
        @test all(iszero, u)
        du = similar(u)
        du .= NaN
        gh_rhs!(du, u, p, zero(T))
        @test all(iszero, du)
        dt = gh_dt(p, u; cfl=T(1 // 4))
        # The shells' logical spacing over their logical speed binds, not the
        # cube's spacing `1/8`, beside the cube's corners.
        @test 0 < dt_spacing(p) < T(1 // 8)
        @test dt ≈ dt_spacing(p) / (4 * sqrt(T(3)))
        u1 = gh_solve(p, u, (zero(T), 10dt); dt=dt)
        worst = zeros(T, 13)
        foreach_owned((q, v) -> (worst[q] = max(worst[q], abs(v))), p, u1)
        @test all(iszero, worst)
    end

    # Guards the norms on the ball: each point counted once, across the glues'
    # closed ownership, and weighted by its physical volume `√g h³`; a constant
    # is its own norm, and the volume is the ball's to the first order a point
    # sum without face weights has.
    @testset "the norms weigh each point by √g h³ and count it once" begin
        layout = ball_layout_b5(T)
        case = ball_case(T, Minkowski(); layout=layout, ε_KO=T(1 // 2), γ0=one(T),
                         γ2=zero(T))
        for N in (8, 16)
            p = ball_problem(T, case, layout; N=N, q=4)
            p.diag.work .= 1
            r = ball_norms(p, TreeGeneralizedHarmonic.DIAG_ERR)
            @test r.l2 ≈ 1
            @test r.linf == 1
            @test isapprox(r.volume, 4 * T(π) * 4^3 / 3; rtol=N == 8 ? 0.15 : 0.08)
            u = statevector(p.U)
            TreeGeneralizedHarmonic.initial_state!(u, p, zero(T))
            ball_error!(p, u, zero(T))
            @test ball_norms(p, TreeGeneralizedHarmonic.DIAG_ERR).linf == 0
        end
    end

    # Guards the refinement rule on the ball: TreeAMR refuses a refinement
    # boundary at a glue, so no flag may ask for one — every shell block, and
    # every cube block within one of its widths of the cube's faces, keeps its
    # level; the cube's middle does not. A brick's flags pass unchanged.
    @testset "refine_flags keeps every block within reach of a glue" begin
        layout = TMP_B5.ThirteenPatchBall{T}(1, 3, 4, (6, 1, 1))
        forest = TMP_B5.layout_forest(T, layout; N=8)
        frozen = [glue_frozen(forest, k) for k in forest.leaves]
        cube = [patch(forest, k) == 1 for k in forest.leaves]
        @test all(frozen[.!cube])
        # The cube's 6³ roots: the 2³ in the middle are free.
        @test count(.!frozen) == 8
        flags = fill(Refine, nblocks(FieldSet{T}(forest, 1; G=0)))
        kept = TreeGeneralizedHarmonic.freeze_glue_flags(forest, flags)
        @test count(==(Refine), kept) == 8
        brick = Forest{T}((2, 2, 2); N=8)
        bflags = fill(Refine, nleaves(brick))
        @test TreeGeneralizedHarmonic.freeze_glue_flags(brick, bflags) === bflags
    end

    # The Teukolsky wave on the ball at `t = 3/10`, amplitude `10⁻⁶` (so that its
    # `O(a²)` violation of the equations is far below the truncation error) and
    # width `2`, sixteen of the cube's spacings at `N = 8`, and the problem of it
    # at `N` and SIMD width `W`. At width `1` the cube's glued faces are
    # pre-asymptotic at `N = 8 … 20`: their ghosts, interpolated in the inner
    # shells, converge at 4.1–5.8 there on their way to 6 (`CODE.md`, "The cube
    # and the shells").
    teukolsky_b5(N, W) = begin
        layout = ball_layout_b5(T)
        bg = TeukolskyWave(T; amplitude=1e-6, width=2)
        case = ball_case(T, bg; layout=layout, ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T))
        p = ball_problem(T, case, layout; N=N, q=4, simd_width=W)
        u = statevector(p.U)
        fill_exact!(p.U, case, T(3 // 10); layout=layout)
        gather!(u, p.U)
        (; p, u, bg, layout)
    end

    # Guards the per-block branch of the ball's kernel: the cube's blocks must be
    # the brick kernel's bits — `gh_rhs_kernel!` launched on the same working
    # array with the cube's physical origins and spacings — on one point at a
    # time and on the CPU's lanes. A second call site that the compiler
    # contracted differently would show here, in the last bit.
    @testset "the cube's blocks are the brick kernel's bits (W = $W)" for W in (1, 4)
        (; p, u) = teukolsky_b5(8, W)
        t = T(3 // 10)
        du = similar(u)
        TreeGeneralizedHarmonic.prepare!(p, u, t)
        TreeGeneralizedHarmonic.rhs!(du, p, t)
        dk = similar(u)
        dk .= NaN
        args = gh_rhs_kernel_args(p, dk, t)
        map_blocks!(gh_rhs_kernel!, p.U, args[1:3]..., p.ball.corigins,
                    p.ball.cspacings, args[6:end]...; appendix=false)
        cube = findall(==(1), block_patches(p.U.forest))
        @test isequal(statearray(du, p.U)[:, :, :, :, cube],
                      statearray(dk, p.U)[:, :, :, :, cube])
    end

    # Guards the shells' kernel on the CPU's lanes: the map is evaluated per
    # lane and packed, and the chain rule runs on `Vec`s, which must be the
    # scalar kernel's numbers to roundoff (`CODE.md`, "The right-hand side on a
    # CPU").
    @testset "the shells' lanes are the scalar kernel" begin
        a = teukolsky_b5(8, 1)
        b = teukolsky_b5(8, 4)
        t = T(3 // 10)
        da, db = similar(a.u), similar(b.u)
        gh_rhs!(da, a.u, a.p, t)
        gh_rhs!(db, b.u, b.p, t)
        @test maximum(abs, da .- db) ≤ 1e-13 * maximum(abs, da)
    end

    # Guards the shells' chain rule, their dissipation along the logical axes,
    # the interpolated ghosts at every kind of glue and the hook: on every patch
    # the right-hand side of the Teukolsky wave is its time derivative, `∂_t g`
    # from `dmetric` and `∂_t Π` by a dual in `t`, to a truncation error that
    # falls at order four — the points on a glue, beside a patch corner and in
    # the appendix included. A wrong term of the chain rule leaves an error
    # that does not converge; a ghost interpolated at too low an order, one
    # that converges at order two.
    @testset "the right-hand side of the Teukolsky wave converges at order 4" begin
        t = T(3 // 10)
        errs = map((8, 16)) do N
            (; p, u, bg, layout) = teukolsky_b5(N, nothing)
            du = similar(u)
            gh_rhs!(du, u, p, t)
            rate(ξ, q) = begin
                x = TreeAMR.chart_map(layout, ξ, q)
                _, _, _, ∂ₜh = _background_state(bg, t, x)
                ∂ₜΠ = ForwardDiff.derivative(t) do τ
                    SVector(background_state(bg, τ, (x[1] + 0τ, x[2] + 0τ,
                                                     x[3] + 0τ))[2])
                end
                (Tuple(∂ₜh)..., Tuple(∂ₜΠ)...)
            end
            fill_by_coordinates!(WithPatch(AllVariables(rate)), p.U)
            ex = statevector(p.U)
            gather!(ex, p.U)
            worst = zeros(T, 13)
            foreach_owned((q, v) -> (worst[q] = max(worst[q], abs(v))), p, du .- ex)
            worst ./ maximum(abs, ex)
        end
        order = log2.(errs[1] ./ errs[2])
        @info "Teukolsky right-hand side on the ball: error by patch at N = 8, 16, " *
              "relative to the largest ∂ₜu, and the order" errs order
        @test all(>(3.7), order)
        @test all(<(1e-4), errs[2])
    end

    # Guards the outer sphere's frozen hook (`CODE.md`, "The outer boundary and
    # the wave zone", option 1): its plane and the ghosts beyond hold the data
    # at `t = 0` whatever the time of the fill, through the atlas, while the
    # live hook holds the wave at that time.
    @testset "the frozen sphere holds the data at t = 0" begin
        layout = ball_layout_b5(T)
        bg = TeukolskyWave(T; amplitude=1e-3, width=2)
        case = ball_case(T, bg; layout=layout, ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T))
        G = 3
        N = 8
        t = T(3)
        for frozen in (true, false)
            p = ball_problem(T, case, layout; N=N, q=4, frozen=frozen)
            u = statevector(p.U)
            TreeGeneralizedHarmonic.initial_state!(u, p, zero(T))
            TreeGeneralizedHarmonic.prepare!(p, u, t)
            forest = p.U.forest
            og, sp = block_origins(forest, T), block_spacings(forest, T)
            worst0, worstt = zero(T), zero(T)
            for b in 1:nblocks(p.U)
                q = patch(forest, blockkey(p.U, b))
                q ≥ 8 || continue
                k = TMP_B5.radial_dimension(layout, q)
                ext = block_extent(T, forest, blockkey(p.U, b))
                ext[k][2] == layout.roots[3] || continue
                # The sphere's plane and the ghosts beyond it, along the radius.
                for j in (G + N + 1):(G + N + 1 + G), i2 in (G + 1):(G + N),
                    i3 in (G + 1):(G + N)
                    I = k == 1 ? (j, i2, i3) : k == 2 ? (i2, j, i3) : (i2, i3, j)
                    ξ = ntuple(d -> og[b][d] + (I[d] - G - 1) * sp[b], Val(3))
                    x = TreeAMR.chart_map(layout, ξ, q)
                    e0 = state_tuple(bg, zero(T), x)
                    et = state_tuple(bg, t, x)
                    for v in 1:20
                        worst0 = max(worst0, abs(p.U.work[I..., v, b] - e0[v]))
                        worstt = max(worstt, abs(p.U.work[I..., v, b] - et[v]))
                    end
                end
            end
            if frozen
                @test worst0 ≤ 1e-16
                @test worstt > 1e-6
            else
                @test worstt ≤ 1e-16
                @test worst0 > 1e-6
            end
        end
    end
end
