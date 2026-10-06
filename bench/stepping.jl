# Where a step of TreeGeneralizedHarmonic goes, by integrator and backend
# (added 2026-09-26; its numbers are in `CODE.md`, "Time integration").
#
#     julia --project=. -t 64 bench/stepping.jl
#     BENCH_MODE=driver BENCH_CASE=wave,hole julia --project=. -t 64 bench/stepping.jl
#
#   BENCH_MODE     step | driver | scan   (default step)
#   BENCH_CASE     wave | hole | wave,hole (default wave)
#   BENCH_BACKEND  cpu | cuda | metal     (default cpu)
#   BENCH_T        Float64 | Float32      (default Float64)
#   BENCH_N        points per block edge  (default 16)
#   BENCH_ROOTS_WAVE, BENCH_ROOTS_HOLE    roots per edge (default 8 and 2:
#                                         512 blocks either way)
#   BENCH_REPS     timed repetitions      (default 5)
#   BENCH_TAG      label printed on every row
#   BENCH_SIMD     the kernel's SIMD width (`simd_width`; 1 is the scalar kernel;
#                  unset: the host's default, and nothing is passed — so the script
#                  still runs on a version of the package without the keyword)
#
# step: one right-hand side; IMEXRungeKutta's RK4 by owner and by broadcast,
# each integrator built once and stepped; `gh_solve` over four steps (`init`
# included); `init` alone, and with the previous integrator's scratch
# (`reuse`, IMEXRungeKutta 1.2); and, when the environment has
# `OrdinaryDiffEqLowOrderRK` and `SciMLBase` (the package does not — add them
# to a scratch copy), OrdinaryDiffEq's RK4 stepped and its `solve` per four
# steps (the driver's pattern until 2026-09-26) — *last*, and the owner step
# once more after them, to see whether their serial passes leave the state
# where they read it. scan: the right-hand side and one RK4 step over block
# sizes `BENCH_SCAN_N` (default 8,12,16,24,32) and roots per edge
# `BENCH_SCAN_ROOTS` (default 2,4,8 for the wave — 8, 64 and 512 blocks — and
# 1,2,4 for the hole), skipping sizes above `BENCH_MAX_POINTS` (default 4e7)
# and any the hole's radius checks refuse; one row per size, in ns per point.
# driver: `evolve!` for three chunks after one to compile;
# it runs on any version of the package, which is how `main` is compared.
# `cuda` and `metal` need the device package in the environment (again a
# scratch copy: neither is a dependency of this package).
using TreeAMR, TreeGeneralizedHarmonic, KernelAbstractions, Printf, Statistics
include(joinpath(pkgdir(TreeGeneralizedHarmonic), "test", "evolution_cases.jl"))

const MODE = get(ENV, "BENCH_MODE", "step")
const CASES = split(get(ENV, "BENCH_CASE", "wave"), ",")
const BK = get(ENV, "BENCH_BACKEND", "cpu")
const T = get(ENV, "BENCH_T", "Float64") == "Float32" ? Float32 : Float64
const TAG0 = get(ENV, "BENCH_TAG", "$(BK)-$(T)-$(MODE)")
const N = parse(Int, get(ENV, "BENCH_N", "16"))
roots_for(c) = parse(Int, get(ENV, "BENCH_ROOTS_" * uppercase(c), c == "wave" ? "8" : "2"))
TAG = ""
const REPS = parse(Int, get(ENV, "BENCH_REPS", "5"))
const SIMDKW = haskey(ENV, "BENCH_SIMD") ? (; simd_width=parse(Int, ENV["BENCH_SIMD"])) : (;)
const q = 4

if BK == "cuda"
    @eval using CUDA
    CUDA.allowscalar(false)
    const backend = CUDABackend()
elseif BK == "metal"
    @eval using Metal
    Metal.allowscalar(false)
    const backend = MetalBackend()
else
    const backend = CPU()
end
const HAVE_IRK = isdefined(TreeGeneralizedHarmonic, :gh_integrator)
const HAVE_ODE = BK == "cpu" && Base.find_package("OrdinaryDiffEqLowOrderRK") !== nothing
if HAVE_IRK
    @eval import IMEXRungeKutta as IRK
end
if HAVE_ODE
    @eval import OrdinaryDiffEqLowOrderRK, SciMLBase
end
sync() = KernelAbstractions.synchronize(backend)

function build(CASE; N=N, ROOTS=roots_for(CASE))
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    if CASE == "wave"
        case = gauge_wave_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T))
        forest = gh_forest(T, case; N=N, roots=ROOTS)
    else
        case = hole_fixture(T; q=q, halfwidth=T(5))
        forest = hole_forest(T, case; N=N, roots=ROOTS,
                             radii=(T(6), T(3), T(3 // 2)))
    end
    return case, forest, ops
end

function timeit(f; n=REPS)
    f(); sync()
    ts = Float64[]
    for _ in 1:n
        t0 = time_ns(); f(); sync(); push!(ts, (time_ns() - t0) / 1e9)
    end
    return (min=minimum(ts), med=median(ts))
end
row(name, t) = (@printf("%s\t%s\t%.3f\t%.3f\tms\n", TAG, name, 1e3t.min, 1e3t.med); flush(stdout))

function header(forest, U)
    npts = nblocks(U) * N^3
    @printf("# %s  threads %d  blocks %d  points %.3e  state %.0f MB  host %s  julia %s\n",
            TAG, Threads.nthreads(), nblocks(U), npts, 20npts * sizeof(T) / 2^20,
            gethostname(), VERSION)
    flush(stdout)
end

function step_mode(CASE)
    case, forest, ops = build(CASE)
    U = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3), backend=backend)
    interior = case.interior
    fill_exact!(U, case, zero(T); interior=interior)
    u = statevector(U); gather!(u, U)
    p = GHProblem(U, GhostSchedule(U, ops), case; q=q, t=zero(T), interior=interior,
                  SIMDKW...)
    dt = gh_dt(p, u; cfl=T(1 // 4))
    if interior !== nothing
        p = with_interior(p, TreeGeneralizedHarmonic.chunk_interior(
            case, dt, nothing, default_relaxation_rate(case); default=true,
            interior=interior))
    end
    header(forest, U)
    du = similar(u)
    far = T(10^6) * dt
    trhs = timeit(() -> gh_rhs!(du, u, p, zero(T))); row("rhs", trhs)
    if HAVE_IRK
        if backend isa CPU
            io = gh_integrator(p, copy(u), (zero(T), far); dt=dt)
            row("imex_owner_step", timeit(() -> IRK.step!(io)))
        end
        ib = IRK.init(IRK.IMEXProblem(gh_rhs!, nothing, copy(u), (zero(T), far), p),
                      IRK.RK4(); dt=dt, stage_limiter=gh_limiter!,
                      step_limiter=gh_limiter!, partition=nothing)
        row("imex_bcast_step", timeit(() -> IRK.step!(ib)))
        w = copy(u)
        t4 = timeit(() -> gh_solve(p, w, (zero(T), 4dt); dt=dt, alias_u0=true); n=3)
        row("gh_solve4_per_step", (min=t4.min / 4, med=t4.med / 4))
        row("init_only", timeit(() -> gh_integrator(p, w, (zero(T), 4dt); dt=dt,
                                                   alias_u0=true); n=3))
        prev = gh_integrator(p, w, (zero(T), 4dt); dt=dt, alias_u0=true)
        row("init_reuse", timeit(() -> gh_integrator(p, w, (zero(T), 4dt); dt=dt,
                                                    alias_u0=true, reuse=prev); n=3))
    end
    if HAVE_ODE
        oi = SciMLBase.init(SciMLBase.ODEProblem(gh_rhs!, copy(u), (zero(T), far), p),
                            OrdinaryDiffEqLowOrderRK.RK4(); dt=dt, adaptive=false,
                            save_everystep=false, stage_limiter=gh_stage_limiter!,
                            step_limiter=gh_step_limiter!)
        row("ode_step", timeit(() -> SciMLBase.step!(oi)))
        t4 = timeit(() -> SciMLBase.solve(SciMLBase.ODEProblem(gh_rhs!, u, (zero(T), 4dt), p),
                                          OrdinaryDiffEqLowOrderRK.RK4(); dt=dt,
                                          adaptive=false, save_everystep=false,
                                          stage_limiter=gh_stage_limiter!,
                                          step_limiter=gh_step_limiter!); n=3)
        row("ode_solve4_per_step", (min=t4.min / 4, med=t4.med / 4))
        if HAVE_IRK
            io2 = gh_integrator(p, copy(u), (zero(T), far); dt=dt)
            row("imex_owner_step_after_ode", timeit(() -> IRK.step!(io2)))
            row("rhs_after_ode", timeit(() -> gh_rhs!(du, u, p, zero(T))))
        end
    end
end

function driver_mode(CASE)
    case, forest, ops = build(CASE)
    chunk = CASE == "wave" ? T(1 // 50) : T(1 // 10)
    stamps = Float64[]
    obs(p, t, u) = (sync(); push!(stamps, time()))
    U0 = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3), backend=backend)
    header(forest, U0)
    # One chunk to compile, then three timed.
    evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops, t_end=chunk, chunk=chunk,
            backend=backend, SIMDKW...)
    t0 = time()
    out = evolve!(T, case; forest=forest, q=q, ops=ops, t_end=3chunk, chunk=chunk,
                  backend=backend, observer=obs, SIMDKW...)
    total = time() - t0
    per_chunk = diff(stamps)
    @printf("%s\tevolve_total\t%.3f\t\ts  (%d steps, %d chunks)\n", TAG, total,
            out.nsteps, out.nchunks)
    @printf("%s\tevolve_per_step\t%.3f\t\tms\n", TAG, 1e3 * sum(per_chunk) / out.nsteps)
    @printf("%s\tevolve_chunks\t%s\t\ts\n", TAG, join((@sprintf("%.2f", x) for x in per_chunk), " "))
    r = out.records[end]
    @printf("%s\tevolve_check\terr_l2=%.6e gauge_l2=%.6e finite=%s\n", TAG, r.err_l2,
            r.gauge_l2, r.finite)
end

function scan_mode(CASE)
    Ns = parse.(Int, split(get(ENV, "BENCH_SCAN_N", "8,12,16,24,32"), ","))
    Rs = parse.(Int, split(get(ENV, "BENCH_SCAN_ROOTS", CASE == "wave" ? "2,4,8" : "1,2,4"), ","))
    maxpts = parse(Float64, get(ENV, "BENCH_MAX_POINTS", "4e7"))
    @printf("# %s  threads %d  host %s  julia %s\n", TAG, Threads.nthreads(),
            gethostname(), VERSION)
    @printf("%s\tscan\tN\troots\tblocks\tpoints\trhs_ms\tstep_ms\trhs_ns_pt\tstep_ns_pt\n", TAG)
    for n in Ns, r in Rs
        try
            case, forest, ops = build(CASE; N=n, ROOTS=r)
            npts = nleaves(forest) * n^3
            if npts > maxpts
                @printf("%s\tscan\t%d\t%d\t%d\t%.3e\tskip: above BENCH_MAX_POINTS\n",
                        TAG, n, r, nleaves(forest), npts)
                continue
            end
            U = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                            backend=backend)
            interior = case.interior
            fill_exact!(U, case, zero(T); interior=interior)
            u = statevector(U); gather!(u, U)
            p = GHProblem(U, GhostSchedule(U, ops), case; q=q, t=zero(T),
                          interior=interior, SIMDKW...)
            dt = gh_dt(p, u; cfl=T(1 // 4))
            if interior !== nothing
                p = with_interior(p, TreeGeneralizedHarmonic.chunk_interior(
                    case, dt, nothing, default_relaxation_rate(case); default=true,
                    interior=interior))
            end
            du = similar(u)
            trhs = timeit(() -> gh_rhs!(du, u, p, zero(T)); n=3)
            integ = gh_integrator(p, copy(u), (zero(T), T(10^6) * dt); dt=dt)
            tstep = timeit(() -> IRK.step!(integ); n=3)
            @printf("%s\tscan\t%d\t%d\t%d\t%.3e\t%.2f\t%.2f\t%.2f\t%.2f\n", TAG, n, r,
                    nblocks(U), npts, 1e3trhs.min, 1e3tstep.min, 1e9trhs.min / npts,
                    1e9tstep.min / npts)
            flush(stdout)
        catch e
            e isa InterruptException && rethrow()
            @printf("%s\tscan\t%d\t%d\tskip: %s\n", TAG, n, r,
                    first(split(sprint(showerror, e), '\n')))
        end
        GC.gc(true)
        BK == "cuda" && CUDA.reclaim()
    end
end

for c in CASES
    global TAG = TAG0 * "-" * c
    MODE == "driver" ? driver_mode(c) : MODE == "scan" ? scan_mode(c) : step_mode(c)
end
