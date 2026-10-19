# The workload behind the test that a distributed run is the serial run
# (added 2026-10-19, on TreeAMR's M7; TreeHydro's `test/mpi_workload.jl` is
# the pattern).
#
# Run under MPI —
#
#     mpiexec -n 3 julia --threads=1 --project=<env> test/mpi_workload.jl OUTDIR
#
# — it runs every case below three times in one launch, so that one
# compilation serves three rank counts: over all three ranks; then over a
# communicator of ranks 0 and 1, while rank 2 runs the same cases over a
# communicator of itself alone and then once more with no communicator at
# all. Each group's first rank writes the group's lines to `OUTDIR/n<k>.txt`
# (`n0.txt` being the serial run), and `mpi_tests.jl` compares the four files.
#
# Two kinds of line. A line that does not start with `~` is a claim of
# **bit-identity**: the state vector gathered in curve order, the leaves,
# every step, chunk and regrid count, every speed and every maximum, the
# horizon's numbers, the track and the fits — none of which is a
# floating-point sum, and all of which TreeAMR's exchange, its
# interpolation and this package's reductions make independent of the rank
# count, the find and the fit running on rank 0 alone. A line that starts
# with `~` holds sums — the L2 norms and the refinement centroid — which a
# partition reassociates, and is compared to roundoff. A line that starts
# with `#` depends on the rank count (the part files of a checkpoint) and is
# not compared at all; the workload asserts what it must say itself.
#
# Its digests are a fold of `hash` over **every** element, not `hash` of the
# vector, which samples a long one.

using MPI
using Test
using TreeAMR
using TreeGeneralizedHarmonic
using Random: Xoshiro

include(joinpath(@__DIR__, "evolution_cases.jl"))

const TGHm = TreeGeneralizedHarmonic

# Every element, in curve order: over a distributed forest the ranks' state
# vectors concatenated in rank order are the serial state vector, because
# each rank holds a contiguous run of the curve.
function mpi_digest(comm, u::AbstractVector)
    full = TreeAMR.allgatherv(comm, Array(u))
    return string(foldl((h, x) -> hash(x, h), full; init=UInt(0x5eed)); base=16,
                  pad=16)
end

mpi_leaves(forest) =
    string(foldl((h, k) -> hash(k, h), forest.leaves; init=UInt(0x1eaf)); base=16,
           pad=16)

# The record's rows that are sums: the L2 norms, which `mesh_mapreduce`
# combines per rank, and the refinement centroid, whose two sums are combined
# over the ranks. Everything else in a row is a maximum, a count, or a
# number computed on rank 0 from interpolated values, which are exact.
const MPI_SUM_KEYS = (:gauge_l2, :ham_l2, :mom_l2, :err_l2, :centroid,
                      :centroid_offset)

function mpi_rows(lines, tag, records)
    for (i, r) in enumerate(records)
        exact = [string(k, "=", repr(v)) for (k, v) in pairs(r) if !(k in MPI_SUM_KEYS)]
        sums = [string(k, "=", repr(v)) for (k, v) in pairs(r) if k in MPI_SUM_KEYS]
        push!(lines, string(tag, " row=", i, " ", join(exact, " ")))
        push!(lines, string("~", tag, " row=", i, " ", join(sums, " ")))
    end
    return nothing
end

mpi_fit(f) = f === nothing ? "nothing" :
             string(hash(f.host), "/", repr(f.params), "/", f.valid, "/", repr(f.t))

# Everything `evolve!` returns that is a claim about the run. The fits'
# costs are timings, and only their counts are compared.
function mpi_result(lines, tag, r)
    comm = r.forest.comm
    push!(lines, string(tag, " finished=", r.finished, " t=", repr(r.t),
                        " u=", mpi_digest(comm, r.u), " leaves=", mpi_leaves(r.forest),
                        " blocks=", r.nblocks, " levels=", repr(r.levels)))
    push!(lines, string(tag, " nsteps=", r.nsteps, " nchunks=", r.nchunks,
                        " regrids=", r.nregrids, " passes=", r.passes,
                        " resamples=", r.nresamples, " λ_initial=", repr(r.λ_initial),
                        " written=", length(r.checkpoints_written)))
    push!(lines, string(tag, " track=", repr(r.track), " bounds=", repr(r.bounds)))
    fits = r.fits === nothing ? "nothing" : join(map(mpi_fit, r.fits), ",")
    push!(lines, string(tag, " fits=", fits, " initial=", mpi_fit(r.fit_initial)))
    r.fit_cost === nothing ||
        push!(lines, string(tag, " fit_counts=", r.fit_cost.nbuild, ",",
                            r.fit_cost.nfill, ",", r.fit_cost.nrefills, ",",
                            r.fit_cost.nfailed))
    mpi_rows(lines, tag, r.records)
    return r
end

# The observer: the state and the mesh at every row, on every rank.
mpi_watch(lines, tag) = (p, t, u) ->
    push!(lines, string(tag, " t=", repr(t), " u=", mpi_digest(p.U.forest.comm, u),
                        " leaves=", mpi_leaves(p.U.forest)))

const Q = 2
const OPS = Operators(prolongation=Q + 2, restriction=Q + 2)

mpi_wave(T) = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=T(1 // 2),
                              γ0=one(T), γ2=T(-1 // 2))

# The static hole with its range projection, as `checkpoint_tests.jl` has it.
function mpi_static_hole(T, comm)
    case0 = hole_fixture(T; q=Q, chunk=T(1 // 20))
    forest = hole_fixture_forest(T, case0; N=8)
    gate = default_gate(case0.interior, forest, Q)
    case = with_bounds(case0, default_bounds(T; M=1, r_gate=gate))
    return case, hole_forest(T, case; N=8, roots=1, radii=(T(3), T(3), one(T)),
                             comm=comm)
end

"""
The cases, over `comm` (`nothing` for the serial run), as lines; `dir` is
where this group writes its checkpoints and `from` the directory of the
three-rank group's, which the others restart from.

- **The gauge wave on one block, with noise**: at two and three ranks most
  ranks hold no block for the whole run — the reductions, the integrator and
  the noise on an empty rank — and the noise is drawn per block.
- **The gauge wave on eight blocks**: the periodic exchange.
- **The adaptive hole through the initial-data cycle**, flagging at every
  chunk: the travelling margin's gathered marks, the level bounds, the
  centroid.
- **The tracked `:fitted` hole that regrids**: the find, the track and the
  fit on rank 0, the other ranks serving its fills and interpolations, a
  regrid that moves data between ranks, and the geometry rebuilt on the new
  mesh.
- **The static hole with checkpoints**, uninterrupted, and restarted from the
  three-rank group's checkpoint at chunk 2.
- **A wall-time stop**, decided from each rank's own clock and agreed.
- **Rotation**, with one I/O process per rank so that a checkpoint has part
  files, and only the newest index and its own parts left behind.
- **A refusal on some ranks only**, which must be a refusal on all.
"""
function mpi_cases(comm, dir, from)
    T = Float64
    lines = String[]
    rank = TreeAMR.commrank(TreeAMR.communicator(comm))

    wave = mpi_wave(T)
    one_block = gh_forest(T, wave; N=8, roots=1, comm=comm)
    mpi_result(lines, "noise",
               evolve!(T, wave; forest=one_block, q=Q, ops=OPS, t_end=T(1 // 10),
                       chunk=T(1 // 20),
                       perturb=U -> add_noise!(U, Xoshiro(5); amplitude=1e-8),
                       observer=mpi_watch(lines, "noise")))

    eight = gh_forest(T, wave; N=8, roots=2, comm=comm)
    mpi_result(lines, "wave",
               evolve!(T, wave; forest=eight, q=Q, ops=OPS, t_end=T(1 // 10),
                       chunk=T(1 // 20), observer=mpi_watch(lines, "wave")))

    adaptive = adaptive_hole_fixture(T)
    mpi_result(lines, "adapt",
               evolve!(T, adaptive; forest=gh_forest(T, adaptive; N=8, roots=4,
                                                     comm=comm),
                       q=Q, ops=OPS, t_end=T(1 // 10), adapt=true, regrid=true,
                       observer=mpi_watch(lines, "adapt")))

    tracked = adaptive_hole_fixture(T; interior=FittedSpec(T; variant=:fitted,
                                                           margin=4, n_L=6),
                                    r_0=nothing, r_1=nothing,
                                    horizon=Horizon(T; every=1, N=12, spin=false))
    tforest = hole_forest(T, tracked; N=8, roots=4, radii=(T(5 // 2),), comm=comm)
    corner = filter(k -> all(d -> block_extent(T, tforest, k)[d][1] ≥ T(5 // 2), 1:3),
                    tforest.leaves)
    refine!(tforest, corner)
    balance!(tforest)
    mpi_result(lines, "tracked",
               evolve!(T, tracked; forest=tforest, q=Q, ops=OPS, t_end=T(1 // 10),
                       regrid=true, observer=mpi_watch(lines, "tracked")))

    # The checkpointed static hole, uninterrupted. Every chunk is written,
    # none deleted, so that the other groups find chunk 2's file in `from`;
    # one I/O process per rank, so that the three-rank group's checkpoint is
    # an index and three part files, which the others then read.
    case, forest = mpi_static_hole(T, comm)
    prefix = joinpath(dir, "hole")
    mpi_result(lines, "hole",
               evolve!(T, case; forest=forest, q=Q, ops=OPS, t_end=T(3 // 20),
                       checkpoint_path_prefix=prefix, checkpoint_every_chunks=1,
                       num_checkpoints_keep=100, checkpoint_io=:all,
                       checkpoint_sync_to_disk=false))
    # And continued from the three-rank group's checkpoint at chunk 2: its
    # lines must be the uninterrupted run's above, whichever group wrote it.
    second = TGHm.checkpoint_files(joinpath(from, "hole"))[2][2]
    mpi_result(lines, "hole",
               evolve!(T, case; q=Q, ops=OPS, t_end=T(3 // 20), restart_file=second,
                       comm=comm))

    # Stopped by the wall clock after its first chunk, on every rank: a
    # limit no chunk fits under, so that the decision is the same whatever
    # each clock reads — and agreed, so that it would be the same if not.
    case, forest = mpi_static_hole(T, comm)
    stopped = evolve!(T, case; forest=forest, q=Q, ops=OPS, t_end=T(3 // 20),
                      checkpoint_path_prefix=joinpath(dir, "stop"),
                      max_walltime_seconds=1e-9, checkpoint_sync_to_disk=false)
    push!(lines, string("stop finished=", stopped.finished, " t=", repr(stopped.t),
                        " u=", mpi_digest(stopped.forest.comm, stopped.u),
                        " written=", length(stopped.checkpoints_written)))

    # Rotation with part files: one I/O process per rank, one file kept.
    case, forest = mpi_static_hole(T, comm)
    rot = joinpath(dir, "rot")
    evolve!(T, case; forest=forest, q=Q, ops=OPS, t_end=T(3 // 20),
            checkpoint_path_prefix=rot, checkpoint_every_chunks=1,
            num_checkpoints_keep=1, checkpoint_io=:all, checkpoint_sync_to_disk=false)
    if rank == 0
        names = readdir(dir)
        indexes = filter(n -> occursin(r"^rot\.it\d+\.h5$", n), names)
        parts = filter(n -> occursin(r"^rot\.it\d+\.h5\.[0-9a-f]{32}\.\d+\.h5$", n),
                       names)
        stale = filter(n -> !startswith(n, only(indexes) * "."), parts)
        push!(lines, string("rotation indexes=", length(indexes), " stale=",
                            length(stale)))
        push!(lines, string("# rotation parts=", length(parts)))
    end

    # A restart file that exists on rank 0 and not elsewhere: the ranks that
    # refuse it and the one that does not must all stop, and nobody wait. A
    # group of one has no other rank to disagree with, and says so.
    if TreeAMR.commsize(TreeAMR.communicator(comm)) > 1
        bad = rank == 0 ? second : joinpath(dir, "absent.h5")
        refused = try
            evolve!(T, case; q=Q, ops=OPS, t_end=T(3 // 20), restart_file=bad,
                    comm=comm)
            false
        catch err
            err isa ArgumentError || rethrow()
            true
        end
        push!(lines, "refusal agreed=$refused")
    else
        push!(lines, "refusal agreed=true")
    end
    return lines
end

# The lines of a group, written by its first rank.
function write_lines(path, lines)
    open(path, "w") do io
        foreach(l -> println(io, l), lines)
    end
    return nothing
end

function mpi_main(out)
    MPI.Init()
    world = MPI.COMM_WORLD
    rank = MPI.Comm_rank(world)
    MPI.Comm_size(world) == 3 || error("the workload runs at three ranks")
    ckpt(n) = mkpath(joinpath(out, "ckpt$n"))
    rank == 0 && foreach(ckpt, 0:3)
    MPI.Barrier(world)

    lines = mpi_cases(world, ckpt(3), ckpt(3))
    rank == 0 && write_lines(joinpath(out, "n3.txt"), lines)

    # Ranks 0 and 1 together, and rank 2 alone; the three-rank group's
    # checkpoints are complete, its collective calls having returned.
    sub = MPI.Comm_split(world, rank < 2 ? 0 : 1, rank)
    n = MPI.Comm_size(sub)
    lines = mpi_cases(sub, ckpt(n), ckpt(3))
    MPI.Comm_rank(sub) == 0 && write_lines(joinpath(out, "n$n.txt"), lines)
    if rank == 2
        write_lines(joinpath(out, "n0.txt"), mpi_cases(nothing, ckpt(0), ckpt(3)))
    end
    MPI.Barrier(world)
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        mpi_main(only(ARGS))
    catch err
        # One rank's exception would leave the others waiting in their next
        # collective: take the job down instead.
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
end
