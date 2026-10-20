# Rank-count independence (added 2026-10-19, on TreeAMR's M7).
#
# The claim extends the thread one: **a run distributed over MPI is the
# serial run** — its state, its mesh, its step, chunk and regrid counts, every
# speed and every maximum, the horizon's numbers, the track and the fits, bit
# for bit at any rank count — **and its sums agree to roundoff**, exactly at
# one rank. TreeAMR makes the exchange, the regrid, the interpolation and its
# own reductions so; what can break it here is this package's, and each
# failure mode is one a serial suite cannot see:
#
#   * a number combined on the host from per-block values — the refinement
#     centroid, the firing count, the initial data's finiteness — that is a
#     rank's own over a distributed forest;
#   * a flag vector that must be the whole forest's — the travelling
#     margin's — handed one rank's blocks;
#   * a host analysis — the find, the track, the fit — that runs on rank 0
#     while the other ranks miss one of its collective steps, and hang;
#   * a decision taken from a rank's own clock or file system — when to
#     checkpoint, when to stop, whether a file exists — that sends some ranks
#     into a collective and the others past it;
#   * a run state that differs between ranks, which TreeIOHDF5's checkpoint
#     refuses on every rank;
#   * a rank with no blocks, on which a reduction, a launch, the noise or a
#     check reads a block that is not there;
#   * a checkpoint that does not restart at another rank count, or restarts
#     into another run, or a rotation that leaves part files behind.
#
# Every line of the workload that is not a sum is compared with `==`; a
# line of sums is compared number by number to roundoff, and its text
# exactly.

isdefined(@__MODULE__, :take_mpi_job!) || include("mpi_jobs.jl")

# The numbers in a line, and the line with each of them replaced by `#`,
# so that two lines of sums are the same line exactly when their skeletons
# are equal and their numbers agree.
const MPI_NUMBER = r"[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+|NaN|Inf)"

mpi_skeleton(line) = replace(line, MPI_NUMBER => "#")
mpi_numbers(line) = [parse(Float64, m.match) for m in eachmatch(MPI_NUMBER, line)]

# Roundoff: the sums are L2 norms and a centroid, each a total over the mesh
# that a partition reassociates once per rank boundary — a few ulp of the
# total. The norms are of order one or below and the centroid's coordinates
# a few `M`, so `rtol` is a few thousand ulp and `atol` covers a centroid
# coordinate whose exact value is zero.
mpi_agree(a, b) = isapprox(a, b; rtol=1e-12, atol=1e-13)

# The comparison itself, on the four files' lines.
function mpi_compare(files)
    serial = files[0]
    # Every case ran, the serial run included, and is what it claims to be:
    # the adaptive hole went through the initial-data cycle, the tracked hole
    # regridded and found its horizon and fitted at every row, the wall-clock
    # stop stopped after one chunk with its checkpoint, and the restarted hole
    # finished.
    @test length(serial) ≥ 60
    adapt = only(filter(l -> startswith(l, "adapt nsteps="), serial))
    @test !occursin("passes=0", adapt)
    @test !occursin("regrids=0", only(filter(l -> startswith(l, "tracked nsteps="),
                                             serial)))
    tracked = filter(l -> startswith(l, "tracked row="), serial)
    @test length(tracked) ≥ 2
    @test all(l -> occursin("horizon_success=true", l), tracked[2:end])
    @test all(l -> occursin("fit_valid=true", l), tracked[2:end])
    @test count(l -> startswith(l, "hole finished=true"), serial) == 2
    stop = only(filter(startswith("stop "), serial))
    @test occursin("finished=false", stop) && occursin("written=1", stop)

    for n in 1:3
        lines = files[n]
        @test length(lines) == length(serial)
        @test only(filter(startswith("rotation "), lines)) == "rotation indexes=1 stale=0"
        @test only(filter(startswith("refusal "), lines)) == "refusal agreed=true"
        # The part files of the last checkpoint kept: one per rank, the
        # rotation's I/O processes, and none serially or at one rank.
        @test only(filter(startswith("# rotation parts="), lines)) ==
              "# rotation parts=$(n == 1 ? 0 : n)"
        for (a, b) in zip(serial, lines)
            startswith(a, "#") && (@test startswith(b, "#"); continue)
            if startswith(a, "~")
                ok = mpi_skeleton(a) == mpi_skeleton(b) &&
                     all(map(mpi_agree, mpi_numbers(a), mpi_numbers(b)))
                # At one rank nothing is reassociated, and the sums are the
                # serial bits too.
                n == 1 && (ok &= a == b)
            else
                ok = a == b
            end
            @test ok
            ok || @info "a line differs at $n rank(s)" serial = a distributed = b
        end
    end
    return nothing
end

@testset "A distributed run is the serial run, at every rank count" begin
    job = take_mpi_job!()
    job === nothing && @info "the MPI workload is skipped: this machine has " *
                             "less than 24 GB for its three ranks " *
                             "(TREEGH_TEST_MPI=1 runs it anyway)"
    job === nothing && @test_skip "a distributed run is the serial run"
    job === nothing || mpi_compare(finish_mpi_workload(job))
end
