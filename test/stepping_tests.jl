# The time integrator (added 2026-09-26): IMEXRungeKutta's RK4 with its
# stage arithmetic by block owner, `src/stepping.jl`.
#
# What the rest of the suite cannot say about it: every convergence rate,
# every hole and the thread workload already run *through* it, so they
# claim the results; what they do not claim is that the partition is the
# ownership partition (a partition that covers the state but hands a
# block's entries to another thread is invisible in every value), that the
# by-owner path is bitwise its own broadcast, and that the swappable
# integrator `evolve!` refills a moving hole's target through is bitwise
# the plain one — the suite runs no moving hole through `evolve!`, so the
# swap is claimed here and nowhere else.

# IMEXRungeKutta through the package's own binding: it is a dependency of the
# package and not of the test environment, which does not need a second
# name for it.
const IRK = TreeGeneralizedHarmonic.IRK
using KernelAbstractions: @kernel, @index

@kernel function stepping_owner_kernel!(tid)
    I = @index(Global, NTuple)
    tid[I[end]] = Threads.threadid()
end

@testset verbose = true "The integrator (IMEXRungeKutta's RK4 by owner)" begin
    T = Float64
    q = 4
    case = gauge_wave_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T))
    forest, fs, p = gh_setup(T, case; N=8, roots=2, q=q)
    fill_exact!(fs, case, zero(T))
    u0 = statevector(fs)
    gather!(u0, fs)
    dt = gh_dt(p, u0; cfl=T(1 // 4))
    nsteps = 6

    # Guards a partition that covers the state exactly — so IMEXRungeKutta
    # accepts it and every result is still right — but gives a block's
    # entries to another thread than the one `map_blocks!` runs the block
    # on, which is the cross-core migration the ownership exists to remove.
    @testset "the partition is TreeAMR's block ownership" begin
        part = state_partition(fs, u0)
        @test length(part) == Threads.nthreads()
        @test reduce(vcat, collect.(part)) == 1:length(u0)
        L = fs.forest.N^3 * fs.nvars
        @test all(r -> isempty(r) || (first(r) - 1) % L == 0 && length(r) % L == 0,
                  part)
        tid = zeros(Int, nblocks(fs))
        map_blocks!(stepping_owner_kernel!, fs, tid)
        offset = Threads.threadpoolsize(:interactive)
        if Threads.nthreads() > 1
            for (c, r) in enumerate(part), b in unique(cld.(r, L))
                @test tid[b] == offset + c
            end
        end
        # A device state takes the broadcast path, which a partition would
        # be refused on; anything but an `Array` stands for it here.
        @test state_partition(fs, view(u0, :)) === nothing
    end

    # Guards a stage arithmetic that differs by path — a partition that
    # drops, doubles or reorders an entry — and the limiters' mapping,
    # which both paths share.
    @testset "by owner is bitwise the broadcast" begin
        owner = gh_solve(p, copy(u0), (zero(T), nsteps * dt); dt=dt)
        ib = IRK.init(IRK.IMEXProblem(gh_rhs!, nothing, copy(u0),
                                      (zero(T), nsteps * dt), p), IRK.RK4();
                      dt=dt, stage_limiter=gh_limiter!,
                      step_limiter=gh_limiter!, partition=nothing)
        IRK.solve!(ib)
        @test isequal(owner, ib.u)
        @test all(isfinite, owner) && owner != u0
    end

    # Guards the moving hole's refill path in `evolve!`: one integrator per
    # chunk, the refilled problem swapped in between pieces through a
    # `ProblemRef`. A swap that is not read (the old problem kept), a step
    # count off by one, or a limiter bypassed by the wrapper would all
    # change the bits; swapping in an equal problem must not.
    @testset "a swapped problem is read from the next step on" begin
        plain = gh_solve(p, copy(u0), (zero(T), nsteps * dt); dt=dt)
        w = copy(u0)
        integ = gh_integrator(p, w, (zero(T), nsteps * dt); dt=dt, alias_u0=true,
                              swappable=true)
        @test integ.p isa ProblemRef && integ.u === w
        for _ in 1:(nsteps ÷ 2)
            IRK.step!(integ)
        end
        # A problem built afresh on the same field set: its own `diag`, its
        # own schedule, the same contents — what a refill hands over.
        p′ = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                       restriction=q + 2)),
                       case; q=q)
        @test p′ !== p
        integ.p.p = p′
        while integ.nstep < integ.nsteps
            IRK.step!(integ)
        end
        @test integ.t == nsteps * dt
        @test isequal(w, plain)
        # And a different problem is what the remaining steps see: the same
        # wave without dissipation (a different right-hand side — the
        # background itself never enters `F` on a periodic mesh with no
        # gauge source, so a different metric would not do) makes the second
        # step differ from the plain run's.
        integ2 = gh_integrator(p, copy(u0), (zero(T), 2dt); dt=dt, swappable=true)
        IRK.step!(integ2)
        pz = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2,
                                                       restriction=q + 2)),
                       gauge_wave_case(T; ε_KO=zero(T), γ0=one(T), γ2=zero(T));
                       q=q)
        integ2.p.p = pz
        IRK.step!(integ2)
        @test integ2.u != gh_solve(p, copy(u0), (zero(T), 2dt); dt=dt)
    end

    # Guards the scratch reuse (IMEXRungeKutta 1.2's `reuse`): an integrator
    # that takes over the previous one's scratch must not allocate it again —
    # the point of passing it — and must step exactly as one with fresh
    # scratch does, since no scratch value carries over between steps.
    @testset "the next chunk takes the previous chunk's scratch" begin
        w1 = copy(u0)
        i1 = gh_integrator(p, w1, (zero(T), 2dt); dt=dt, alias_u0=true)
        IRK.solve!(i1)
        fresh = gh_solve(p, copy(w1), (2dt, 4dt); dt=dt)
        nbytes = sizeof(u0)
        mk() = gh_integrator(p, w1, (2dt, 4dt); dt=dt, alias_u0=true, reuse=i1)
        mk()
        @test @allocated(mk()) < nbytes
        @test @allocated(gh_integrator(p, w1, (2dt, 4dt); dt=dt, alias_u0=true)) >
              4nbytes
        i2 = mk()
        IRK.solve!(i2)
        @test isequal(i2.u, fresh)
        # Another mesh's state does not fit, and says so rather than
        # allocating behind the caller's back.
        forest2, fs2, p2 = gh_setup(T, case; N=8, roots=1, q=q)
        v2 = statevector(fs2)
        @test_throws ArgumentError gh_integrator(p2, v2, (zero(T), dt); dt=dt,
                                                 alias_u0=true, reuse=i1)
    end

    # Guards the driver's use of the integrator against the plain one: the
    # same case, the same steps, through `evolve!` (two chunks, so that the
    # second takes over the first's scratch; no regrid, no interior) and
    # through `gh_solve` chunk by chunk on the same mesh.
    @testset "evolve! steps the state as gh_solve does" begin
        out = evolve!(T, case; forest=deepcopy(forest), q=q,
                      ops=Operators(prolongation=q + 2, restriction=q + 2),
                      t_end=8dt, chunk=4dt)
        @test out.nchunks == 2
        u_plain = copy(u0)
        for c in 1:2
            r = out.records[c + 1]
            u_plain = gh_solve(p, u_plain, (T(out.records[c].t), T(r.t));
                               dt=T(r.dt), alias_u0=true)
        end
        @test maximum(abs, out.u .- u_plain) ≤ 100 * eps(T) * maximum(abs, u_plain)
    end
end
