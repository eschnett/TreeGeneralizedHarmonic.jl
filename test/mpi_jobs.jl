# The `mpiexec` job of the MPI test (added 2026-10-19), as a helper rather
# than a test: `runtests.jl` includes it before the first test, and
# `mpi_tests.jl` includes it if it was not. TreeHydro's `test/mpi_jobs.jl`
# is the pattern, and TreeAMR's before it.
#
# The job runs `mpi_workload.jl` under `mpiexec -n 3` and is
# compilation-bound — three ranks compiling the driver, the finder and the
# fit. Where the machine has room for three more processes beside the suite
# it starts here, at the start of the suite, and compiles on otherwise idle
# cores while the rest runs; elsewhere — a CI runner — it runs here to the
# end *before* the suite, and `mpi_tests.jl` compares what it wrote. Until
# 2026-10-30 it ran there in `mpi_tests.jl`, after the suite, beside a test
# process that had compiled every kernel of it: on the threaded Linux cell
# (four cores, 16 GB) the runner went down under the two in five of its
# thirteen attempts from 2026-10-23 to 2026-10-30, every time while the
# ranks ran ("lost communication", "received a shutdown signal"), and
# always after `type_tests.jl`. The same launcher is in TreeAMR
# and TreeHydro, and may move into TreeAMR (noted 2026-10-19; `CODE.md`,
# "Upstream prerequisites").

using MPI: MPI

const MPI_WORKLOAD = joinpath(@__DIR__, "mpi_workload.jl")

# Whether the job may run at all: three ranks of about 3 GB each, the
# package compiled in every one, beside the suite's own process. A machine
# with less than 24 GB skips it — GitHub's macOS runners have 7 GB and the
# job ran past its hour there swapping, and a 16 GB Linux runner went down
# under it at one thread (2026-10-19). `TREEGH_TEST_MPI=1` or `0` decides it
# by hand; CI forces it on for its threaded Linux cell.
function mpi_runnable()
    choice = get(ENV, "TREEGH_TEST_MPI", "")
    choice == "1" && return true
    choice == "0" && return false
    return Sys.total_memory() >= 24 * 2^30
end

# Whether the job may run beside the suite: three single-threaded ranks,
# each compiling the workload. `TREEGH_TEST_MPI_CONCURRENT=0` or `1`
# decides it by hand.
function mpi_concurrent()
    choice = get(ENV, "TREEGH_TEST_MPI_CONCURRENT", "")
    choice == "1" && return true
    choice == "0" && return false
    return Sys.CPU_THREADS >= 8 && Sys.total_memory() >= 24 * 2^30
end

# The workload under `mpiexec -n 3`, started and not waited for. The
# command is built from `MPI.mpiexec()` with its environment set on it:
# interpolating `mpiexec()` into a larger command drops the library paths
# it carries (TreeAMR's finding). The *active* project, not `test/`: under
# `Pkg.test` the tests run in a sandbox.
function launch_mpi_workload(; timeout=3600)
    out = mktempdir()
    mpi = MPI.mpiexec()
    project = Base.active_project()
    cmd = `$mpi -n 3 $(Base.julia_cmd()) --threads=1 --project=$project
           $MPI_WORKLOAD $out`
    err = IOBuffer()
    proc = run(pipeline(setenv(cmd, mpi.env); stdout=devnull, stderr=err); wait=false)
    return (; proc, out, err, start=time(), timeout)
end

# The job's four files of lines, with the deadline from its start: a rank
# waiting for a message never sent would otherwise hang the suite.
function finish_mpi_workload(job)
    (; proc, out, err, start, timeout) = job
    left = max(1.0, timeout - (time() - start))
    if timedwait(() -> process_exited(proc), left) !== :ok
        kill(proc)
        error("the MPI workload did not finish in $timeout s; its stderr:\n" *
              String(take!(err)))
    end
    success(proc) || error("the MPI workload failed; its stderr:\n" *
                           String(take!(err)))
    return Dict(n => readlines(joinpath(out, "n$n.txt")) for n in 0:3)
end

const MPI_JOB = Ref{Any}(nothing)

# The job, started beside the suite, or run to its end before it where the
# machine has no room for both: the ranks then share it with a process that
# holds the package and nothing else yet. Waiting throws nothing — a job that
# failed or overran is reported by `finish_mpi_workload` in `mpi_tests.jl`,
# inside the suite's testset, as before.
function start_mpi_job!()
    mpi_runnable() || return nothing
    job = launch_mpi_workload()
    MPI_JOB[] = job
    atexit(() -> process_running(job.proc) && kill(job.proc))
    if !mpi_concurrent()
        @info "the MPI workload runs before the suite: this machine has no " *
              "room for its three ranks beside it (TREEGH_TEST_MPI_CONCURRENT=1 " *
              "runs them together)"
        timedwait(() -> process_exited(job.proc), job.timeout)
        @info "the MPI workload ran for $(round(Int, time() - job.start)) s"
    end
    return nothing
end

# The job, started now if it was not, or `nothing` where it may not run.
function take_mpi_job!()
    job = MPI_JOB[]
    MPI_JOB[] = nothing
    job === nothing || return job
    return mpi_runnable() ? launch_mpi_workload() : nothing
end
