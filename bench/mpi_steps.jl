# The cost of a time step over MPI (added 2026-10-19): what several GPUs buy
# the evolution itself, apart from the per-chunk analysis — the horizon find,
# the fit and a run script's observer, which run on rank 0 or on every rank
# alike and do not shrink with the rank count.
#
#     mpiexec -n P julia --project=<env> bench/mpi_steps.jl mpi=true backend=cuda N=32
#     julia --project=<env> bench/mpi_steps.jl backend=cuda N=32          # no MPI
#
# The case is `test/octant_runs.jl`'s Kerr-Schild hole on the reflecting
# octant, with step 5's analytic `:damped` layer and the algebraic gauge
# source, so that there is no finder and no fit: a box `[0, L]³` of `roots³`
# root blocks of `N³` points, refined in the cubes `[0, R]³` of `radii`.
# `evolve!` runs one chunk of `steps_short` steps and then one of
# `steps_long`, each a single chunk with no observer; the difference of the
# two wall clocks over the difference of their step counts is the cost of a
# step without compilation, the initial data and the record row, which both
# runs pay alike. Rank 0 prints one line:
#
#     ranks blocks points step_ms points_per_ns
#
# Options (`key=value`): L=32 N=32 roots=4 radii=16,8,4 q=4 cfl=1/2
# steps_short=4 steps_long=24 backend=cpu|cuda mpi=false deviceaware=false

using TreeAMR
using TreeGeneralizedHarmonic
using Printf

const OPTIONS = Dict{String,String}()
for arg in ARGS
    k, v = occursin('=', arg) ? split(arg, '='; limit=2) : (arg, "")
    OPTIONS[String(k)] = String(v)
end
opt(k, default) = get(OPTIONS, k, default)
rat(s) = (p = split(s, '/'); length(p) == 2 ? parse(Int, p[1]) // parse(Int, p[2]) :
                            parse(Int, s))

use_mpi = opt("mpi", "false") in ("true", "1")
if use_mpi
    @eval using MPI
    MPI.Init()
end
backend = if opt("backend", "cpu") == "cuda"
    @eval using CUDA
    if use_mpi
        node = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED,
                                   MPI.Comm_rank(MPI.COMM_WORLD))
        Base.invokelatest(() -> CUDA.device!(MPI.Comm_rank(node) % length(CUDA.devices())))
        MPI.free(node)
    end
    Base.invokelatest(() -> CUDA.CUDABackend())
else
    TreeGeneralizedHarmonic.CPU()
end
comm = use_mpi ? communicator(MPI.COMM_WORLD;
                              deviceaware=opt("deviceaware", "false") in ("true", "1")) :
       nothing
rank0 = !use_mpi || MPI.Comm_rank(MPI.COMM_WORLD) == 0
nranks = use_mpi ? MPI.Comm_size(MPI.COMM_WORLD) : 1

const T = Float64
L = rat(opt("L", "32"))
N = parse(Int, opt("N", "32"))
roots = parse(Int, opt("roots", "4"))
radii = [rat(r) for r in split(opt("radii", "16,8,4"), ',')]
q = parse(Int, opt("q", "4"))
cfl = rat(opt("cfl", "1/2"))
ns = parse(Int, opt("steps_short", "4"))
nl = parse(Int, opt("steps_long", "24"))

case0 = kerr_schild_case(T; halfwidth=L, r_0=T(3 // 4), r_1=T(3 // 2), ρ_ramp=1,
                         chunk=T(1), octant=:reflecting, gauge_source=:algebraic)
ops = Operators(prolongation=q + 2, restriction=q + 2)
mesh() = hole_forest(T, case0; N=N, roots=roots, center=(0, 0, 0), radii=radii,
                     shape=:cube, comm=comm)
h = minimum_spacing(T, mesh())
# The step `evolve!` will take, from the initial data's speed; a chunk of
# `n` such steps is `t_end = n dt`, and one chunk is the whole run.
λ = 2.0                     # a bound: the step is sized inside evolve! anyway
function run(n)
    dt = cfl * h / λ
    t_end = n * dt
    t0 = time()
    out = evolve!(T, case0; forest=mesh(), q=q, ops=ops, t_end=T(t_end),
                  chunk=T(t_end), cfl=T(cfl), backend=backend, comm=comm)
    use_mpi && MPI.Barrier(MPI.COMM_WORLD)
    return time() - t0, out.nsteps, out
end

run(ns)                                  # compilation
t_s, n_s, _ = run(ns)
t_l, n_l, out = run(nl)
step = (t_l - t_s) / (n_l - n_s)
points = out.nblocks * N^3
rank0 && @printf("ranks=%d blocks=%d points=%d steps=%d,%d step_ms=%.3f points_per_ns=%.3f\n",
                 nranks, out.nblocks, points, n_s, n_l, 1e3 * step, points / (step * 1e9))
