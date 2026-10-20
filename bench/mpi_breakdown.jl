# Where a time step goes over MPI, rank by rank (added 2026-10-19): the
# question `bench/mpi_steps.jl` raised — a step on the nested octant mesh does
# not strong-scale on H200s — taken apart.
#
#     mpiexec -n P julia --project=<env> bench/mpi_breakdown.jl mpi=true backend=cuda
#     julia --project=<env> bench/mpi_breakdown.jl backend=cuda           # no MPI
#
# The case and the mesh are `bench/mpi_steps.jl`'s (the Kerr-Schild hole on
# the reflecting octant with the analytic `:damped` layer and the algebraic
# source, a box `[0, L]³` of `roots³` roots of `N³` points refined in the cubes
# `[0, R]³` of `radii`), and the problem and state are those of a short
# `evolve!`. Then, each timed on every rank by itself (a barrier before, a
# device synchronize after) as the minimum of `reps` repetitions:
#
#   * `scatter`, the state into the working array;
#   * `fill`, a whole `fill_ghosts!` with the case's Dirichlet hook;
#   * per stage of the ghost schedule — phase 1 (copies and restrictions) and
#     one prolongation stage per target level — its local groups, its packs
#     and its unpacks alone, through TreeAMR's internals as its
#     `bench/mpi.jl` does; what a fill costs beyond their sum is the messages,
#     the host staging, the boundary hook and the synchronizations;
#   * `kernel`, the fused right-hand-side kernel alone;
#   * `rhs`, a whole `gh_rhs!`, and `step`, a whole RK4 step.
#
# Rank 0 prints each rank's time for each (ms), and each rank's load: blocks
# per level, the launches of a fill, and the transfers and the source points
# (target points times stencil taps) its local groups and its packs compute,
# by kind — a sender computes what it sends, prolongation included.
#
# Options (`key=value`): L=32 N=32 roots=4 radii=16,8,4 q=4 cfl=1/2 reps=5
# backend=cpu|cuda mpi=false deviceaware=false

using TreeAMR
using TreeGeneralizedHarmonic
using Printf

const TGH = TreeGeneralizedHarmonic

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
    TGH.CPU()
end
comm = use_mpi ? communicator(MPI.COMM_WORLD;
                              deviceaware=opt("deviceaware", "false") in ("true", "1")) :
       communicator(nothing)
rank = TreeAMR.commrank(comm)
nranks = TreeAMR.commsize(comm)

const T = Float64
L = rat(opt("L", "32"))
N = parse(Int, opt("N", "32"))
roots = parse(Int, opt("roots", "4"))
radii = [rat(r) for r in split(opt("radii", "16,8,4"), ',')]
q = parse(Int, opt("q", "4"))
cfl = rat(opt("cfl", "1/2"))
reps = parse(Int, opt("reps", "5"))

case = kerr_schild_case(T; halfwidth=L, r_0=T(3 // 4), r_1=T(3 // 2), ρ_ramp=1,
                        chunk=T(1), octant=:reflecting, gauge_source=:algebraic)
ops = Operators(prolongation=q + 2, restriction=q + 2)
forest = hole_forest(T, case; N=N, roots=roots, center=(0, 0, 0), radii=radii,
                     shape=:cube, comm=use_mpi ? comm : nothing)
h = minimum_spacing(T, forest)
dt = T(cfl) * h / 2
out = evolve!(T, case; forest=forest, q=q, ops=ops, t_end=2dt, chunk=2dt, cfl=T(cfl),
              backend=backend, comm=use_mpi ? comm : nothing)
p, u = out.problem, out.u
t = T(out.t)
U, schedule = p.U, p.schedule
be = TreeAMR.get_backend(U.work)
sync() = TreeAMR.synchronize(be)
barrier() = use_mpi ? MPI.Barrier(MPI.COMM_WORLD) : nothing
du = similar(u)
bnd = TGH.dirichlet(case, t)

# This rank's time for `f`, the least of `reps` after one untimed call,
# each between a barrier and a synchronize; every rank's, gathered.
function timed(f)
    f()
    sync()
    best = Inf
    for _ in 1:reps
        barrier()
        t0 = time_ns()
        f()
        sync()
        best = min(best, (time_ns() - t0) / 1e6)
    end
    return TreeAMR.allgather(comm, best)
end

row(name, ts) = rank == 0 &&
    println(rpad(name, 28), join([@sprintf("%9.3f", x) for x in ts], " "),
            @sprintf("   max %9.3f", maximum(ts)))

rank == 0 && println("# ranks=$nranks blocks=$(nleaves(forest)) points=",
                     nleaves(forest) * N^3, " N=$N L=$L radii=$radii backend=",
                     nameof(typeof(be)))
rank == 0 && println(rpad("# phase (ms)", 28), join([lpad("rank $r", 9) for r in 0:(nranks - 1)], " "))

row("scatter", timed(() -> scatter!(U, u)))
row("fill", timed(() -> (scatter!(U, u); fill_ghosts!(U, schedule; boundary=bnd))))
# The stages by their message tag, which every rank agrees on; a rank may
# hold no stage for a level its blocks do not reach, and times nothing there.
# A stage's level is its tag's (`prolongation_tag`); a rank's stage list
# also holds the levels it only sends or receives at, which `levels` omits.
taglevel(tag) = something(findfirst(l -> TreeAMR.prolongation_tag(l) == tag, 0:30), 1) - 1
mine = Dict(st.tag => (i == 1 ? 0 : taglevel(st.tag), st)
            for (i, st) in enumerate(schedule.stages))
pairs = TreeAMR.allgatherv(comm, [(tag, lv) for (tag, (lv, _)) in mine])
for (tag, lv) in sort!(unique(pairs); by=first)
    st = haskey(mine, tag) ? mine[tag][2] : nothing
    name = lv == 0 && tag == first(schedule.stages).tag ? "phase1" : "prolong L$lv"
    row("  $name locals", timed(() -> st === nothing ? nothing :
                                      TreeAMR.run_phase!(U.work, U.work, st.locals,
                                                         U.nvars, be;
                                                         factors=U.factors,
                                                         rotvars=U.rotvars)))
    r = st === nothing ? nothing : st.remote
    any(TreeAMR.allgather(comm, r !== nothing)) || continue
    bufs = r === nothing ? nothing : TreeAMR.stagebuffers(r, U.nvars, be, forest)
    row("  $name packs", timed(() -> r === nothing ? nothing :
                                     TreeAMR.pack_stage!(U, r, bufs, be)))
    row("  $name unpacks", timed(() -> r === nothing ? nothing :
                                       TreeAMR.unpack_stage!(U, r, bufs, be)))
end
row("kernel", timed(() -> map_blocks!(TGH.gh_rhs_kernel!, U,
                                      TGH.gh_rhs_kernel_args(p, du, t)...)))
row("rhs", timed(() -> gh_rhs!(du, u, p, t)))
integ = gh_integrator(p, copy(u), (t, t + 1000dt); dt=dt)
row("step", timed(() -> TGH.IRK.step!(integ)))

# The load, rank by rank.
function load()
    levels = zeros(Int, 8)
    for b in 1:nblocks(U)
        levels[level(blockkey(U, b)) + 1] += 1
    end
    launches = sum(st -> length(st.locals) +
                         (st.remote === nothing ? 0 : length(st.remote.packs) +
                                                      length(st.remote.unpacks)),
                   schedule.stages)
    work = Dict{Symbol,Vector{Int}}()     # kind => [transfers, target points, taps]
    groups = [g for st in schedule.stages
              for g in vcat(st.locals, st.remote === nothing ? eltype(st.locals)[] :
                                       st.remote.packs)]
    for g in groups
        n = TreeAMR.ntransfers(g)
        pts = prod(length(s.srcstart) for s in g.stencils)
        taps = prod(size(s.weights, 1) for s in g.stencils)
        w = get!(work, g.kind, zeros(Int, 3))
        w .+= (n, n * pts, n * pts * taps)
    end
    w(k) = get(work, k, zeros(Int, 3))
    return (Tuple(levels[1:4])..., launches, w(:copy)..., w(:restrict)...,
            w(:prolong)...)
end
loads = TreeAMR.allgather(comm, load())
if rank == 0
    println("# load per rank: blocks at levels 0..3 | launches a fill | copy " *
            "transfers/points/taps | restrict … | prolong …")
    for (r, l) in enumerate(loads)
        println(@sprintf("rank %d: %s | %d | %s | %s | %s", r - 1, join(l[1:4], " "),
                         l[5], join(l[6:8], "/"), join(l[9:11], "/"),
                         join(l[12:14], "/")))
    end
end

# How many CPUs each rank may run on (Linux): Open MPI's `mpiexec` binds a
# rank to one core by default, which a rank's host threads then share.
function allowed_cpus()
    Sys.islinux() || return -1
    line = something(findfirst(startswith("Cpus_allowed_list"),
                               readlines("/proc/self/status")), 0)
    line == 0 && return -1
    list = strip(split(readlines("/proc/self/status")[line], ':')[2])
    return sum(split(list, ',')) do r
        a = parse.(Int, split(r, '-'))
        length(a) == 1 ? 1 : a[2] - a[1] + 1
    end
end
cpus = TreeAMR.allgather(comm, (allowed_cpus(), Threads.nthreads()))
rank == 0 && println("# CPUs allowed / Julia threads per rank: ",
                     join(["$(c[1])/$(c[2])" for c in cpus], ", "))
