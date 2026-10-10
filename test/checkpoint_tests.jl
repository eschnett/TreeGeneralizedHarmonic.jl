# Checkpoint and restart (added 2026-10-01, on TreeAMR 0.1.4's M9a):
# `src/checkpoint.jl` and `evolve!`'s checkpoint keywords. The claim every
# testset below serves is TreeHydro's — a chain of restarts is the
# uninterrupted run bit for bit, its state, its mesh and every number
# `evolve!` returns — on this package's run state, which carries history a
# hydro run does not: the horizon track, the two fits, the finder's seed, the
# gauge source's sample, the speed growth that sizes a moving hole's step. And
# one claim TreeHydro does not make: the checkpoint is written *before* the
# regrid, so a restart may change the regridding criterion and regrid with it
# first (decided 2026-10-01).
#
# `CODE.md`, "Checkpoint and restart". The chains run one chunk per call, the
# way a job chain does: `max_walltime_seconds` below any chunk stops every
# call after its first chunk, with its checkpoint.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using MultiFloats: Float32x2
using StaticArrays: SVector
import SpacetimeMetrics as SM

const TGHc = TreeGeneralizedHarmonic

@testset verbose = true "Checkpoint and restart" begin
    T = Float64
    q = 2
    ops = Operators(prolongation=q + 2, restriction=q + 2)

    # The uninterrupted run, and the same run as a chain of jobs of one chunk
    # each: the first from the initial data, every later one from
    # `latest_checkpoint`, as `CODE.md`'s job-chain idiom is written.
    function chain(case, forest, t_end; kw...)
        pre = joinpath(mktempdir(), "run")
        common = (q=q, ops=ops, t_end=t_end, checkpoint_path_prefix=pre,
                  max_walltime_seconds=1e-9, checkpoint_sync_to_disk=false)
        out = evolve!(T, case; forest=deepcopy(forest), common..., kw...)
        jobs = 1
        while !out.finished
            out = evolve!(T, case; common..., restart_file=latest_checkpoint(pre),
                          kw...)
            jobs += 1
        end
        return out, jobs, pre
    end

    # What "the same run" means: the state and the mesh bit for bit, the
    # record row for row, the counters, and the run state that comes back —
    # the track, the projection's accounting and the fits' coefficients
    # (their diagnostics come back as plain `Float64` tuples, which nothing
    # the run computes reads).
    function test_same(a, b)
        @test isequal(a.u, b.u)
        @test a.forest.leaves == b.forest.leaves
        @test isequal(a.records, b.records)
        @test (a.nsteps, a.nregrids, a.nresamples) == (b.nsteps, b.nregrids, b.nresamples)
        @test isequal(repr(a.track), repr(b.track))
        @test isequal(repr(a.bounds), repr(b.bounds))
        if b.fits === nothing
            @test a.fits === nothing
        else
            for (fa, fb) in zip(a.fits, b.fits)
                @test (fa === nothing) == (fb === nothing)
                (fa === nothing || fb === nothing) && continue
                @test isequal(fa.host, fb.host) && isequal(fa.params, fb.params) &&
                      isequal(fa.t, fb.t) && isequal(fa.c, fb.c) &&
                      fa.valid == fb.valid && isequal(fa.points, fb.points) &&
                      isequal(fa.model, fb.model)
            end
        end
    end

    # Guards the reals of a `Float32x2` run, which TreeIOHDF5's plain data refuse
    # as scalars, and the structs of the run state, which they refuse
    # outright: either read back as anything but the bits they were would make
    # every later number of a restarted run a different one.
    @testset "the run state's reals and structs read back bit for bit" begin
        xs = Float32x2[Float32x2(1) / 3, -Float32x2(2) / 7]
        @test isequal(TGHc.from_plain_reals(Float32x2, TGHc.plain_reals(xs)), xs)
        @test TGHc.plain_reals(1 / 3) == [1 / 3]

        case = fitted_fixture(T)
        tr = seed_track(case, zero(T))
        p = TGHc.to_plain(tr)
        @test isequal(repr(TGHc.from_plain(HorizonTrack{T}, p)), repr(tr))
        @test TGHc.from_plain(HorizonTrack{T}, p).c_find isa SVector{3,T}
        bd = default_bounds(T; M=1, r_gate=T(9 // 10))
        @test TGHc.from_plain(StateBounds{T}, TGHc.to_plain(bd)) === bd
        acc = BoundsAccounting()
        acc.hits, acc.r_max = 3, T(7 // 10)
        back = TGHc.from_plain(BoundsAccounting, TGHc.to_plain(acc))
        @test (back.hits, back.r_max, back.calls) == (3, T(7 // 10), 0)
        @test isnan(back.first_t)

        # A struct the run state does not know is refused with its path, not
        # stored as something that reads back differently.
        @test_throws "run.background" TGHc.to_plain(SM.KerrSchild(one(T), zero(T));
                                                    path="run.background")
        @test_throws "run.track" TGHc.from_plain(HorizonTrack{T}, TGHc.to_plain(bd);
                                                 path="run.track")
    end

    # Guards the file names a job chain relies on: the newest by iteration,
    # never TreeIOHDF5's partial file or another prefix's, and a rotation that
    # keeps the file just written.
    @testset "names, rotation and latest_checkpoint are TreeHydro's" begin
        dir = mktempdir()
        pre = joinpath(dir, "g5")
        @test latest_checkpoint(pre) === nothing
        for it in (5, 40, 12)
            touch(TGHc.checkpoint_filename(pre, it))
        end
        touch(TGHc.checkpoint_filename(pre, 99) * ".partial")
        touch(TGHc.checkpoint_filename(joinpath(dir, "g5x"), 100))
        @test basename(TGHc.checkpoint_filename(pre, 12)) == "g5.it0000000012.h5"
        @test latest_checkpoint(pre) == TGHc.checkpoint_filename(pre, 40)
        removed = TGHc.rotate_checkpoints!(pre, 2; keep=TGHc.checkpoint_filename(pre, 12))
        @test sort(basename.(removed)) == ["g5.it0000000005.h5"]
        @test first.(TGHc.checkpoint_files(pre)) == [12, 40]
    end

    # Guards a job script's mistakes, which must cost a second and not the
    # queue wait: every refusal comes before the initial data are built.
    @testset "the checkpoint keywords are refused before anything is built" begin
        case = hole_fixture(T; q=q)
        forest = hole_fixture_forest(T, case; N=8)
        go(; kw...) = evolve!(T, case; q=q, ops=ops, t_end=T(1 // 10), kw...)
        @test_throws "does not exist" go(; forest=forest,
                                          checkpoint_path_prefix="/no/such/dir/run",
                                          checkpoint_every_chunks=1)
        @test_throws "without checkpoint_path_prefix" go(; forest=forest,
                                                          checkpoint_every_chunks=1)
        @test_throws "nothing would" go(; forest=forest,
                                         checkpoint_path_prefix=joinpath(mktempdir(), "r"))
        @test_throws "needs a forest" go()
        f = joinpath(mktempdir(), "x.h5")
        touch(f)
        @test_throws "Drop the forest" go(; forest=forest, restart_file=f)
        @test_throws "does not exist" go(; restart_file=f * ".missing")
    end

    # The suite's static hole, with the range projection on so that its
    # accounting is carried, chained one chunk per job; then a refusal of
    # every parameter that differs, and the extension of a finished run whose
    # end lies on a chunk boundary — which is what a restart point after the
    # last chunk is for.
    @testset "a chain on the static hole is the uninterrupted run" begin
        case0 = hole_fixture(T; q=q, chunk=T(1 // 20))
        forest = hole_fixture_forest(T, case0; N=8)
        case = with_bounds(case0, default_bounds(T; M=1,
                                                 r_gate=default_gate(case0.interior,
                                                                     forest, q)))
        ref = evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                      t_end=T(3 // 20))
        out, jobs, pre = chain(case, forest, T(3 // 20))
        @test jobs == 3
        @test out.finished && out.restart_file !== nothing
        @test out.criterion_changed == Symbol[]
        test_same(out, ref)

        # Two parameters changed, both named, in one refusal.
        f1 = latest_checkpoint(pre)
        other = with_bounds(hole_fixture(T; q=q, chunk=T(1 // 10)), case.bounds)
        err = try
            evolve!(T, other; q=q, ops=ops, t_end=T(3 // 20), cfl=T(1 // 5),
                    restart_file=f1)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("`chunk`", err.msg) && occursin("`cfl`", err.msg)
        @test_throws "nothing left to run" evolve!(T, case; q=q, ops=ops,
                                                   t_end=T(1 // 10), restart_file=f1)

        # A finished run at `t_end = 1/10 = 2 · (1/20)` leaves its last chunk
        # as a restart point; continued to `3/20` it is the longer run.
        pre2 = joinpath(mktempdir(), "short")
        short = evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                        t_end=T(1 // 10), checkpoint_path_prefix=pre2,
                        checkpoint_every_chunks=1, checkpoint_sync_to_disk=false)
        @test short.finished && length(short.checkpoints_written) == 2
        longer = evolve!(T, case; q=q, ops=ops, t_end=T(3 // 20),
                         restart_file=latest_checkpoint(pre2))
        test_same(longer, ref)
        @info "the static hole's chain (2026-10-01)" jobs nsteps = ref.nsteps
    end

    # The adaptive fixture's mesh that the first regrid moves (127 → 120
    # blocks, `refinement_tests.jl`): the checkpoint is written before that
    # regrid, so the restart regrids first — and with a changed criterion it
    # regrids differently, which is the point of saving there.
    @testset "a restart regrids first, with the criterion it is given" begin
        case = adaptive_hole_fixture(T)
        forest = gh_forest(T, case; N=8, roots=4)
        targets = filter(forest.leaves) do k
            ext = block_extent(T, forest, k)
            all(d -> -T(5 // 2) ≤ ext[d][1] && ext[d][2] ≤ T(5 // 2), 1:3) ||
                all(d -> ext[d][1] ≥ T(5 // 2), 1:3)
        end
        refine!(forest, targets)
        balance!(forest)
        ref = evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                      t_end=T(3 // 20), regrid=true)
        @test ref.nregrids == 1 && ref.nblocks == 120
        out, jobs, pre = chain(case, forest, T(3 // 20); regrid=true)
        test_same(out, ref)

        # The same first job, then a restart that no longer coarsens: the
        # hand-refined corner stays, the change is named, and the rows before
        # the restart are the uninterrupted run's.
        pre2 = joinpath(mktempdir(), "crit")
        job1 = evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                        t_end=T(3 // 20), regrid=true, checkpoint_path_prefix=pre2,
                        max_walltime_seconds=1e-9, checkpoint_sync_to_disk=false)
        @test !job1.finished && job1.nblocks == 127
        loose = with_refinement(case, Refinement(T; refine_tol=T(2 // 5),
                                                 coarsen_tol=T(1 // 10^6),
                                                 maxlevel_cap=1,
                                                 floor_margin=zero(T),
                                                 ceiling_cells=1))
        changed = evolve!(T, loose; q=q, ops=ops, t_end=T(3 // 20), regrid=true,
                          restart_file=latest_checkpoint(pre2))
        @test changed.criterion_changed == [:coarsen_tol]
        @test changed.nregrids == 0 && changed.nblocks == 127
        @test isequal(changed.records[1:2], ref.records[1:2])
    end

    # The tracked `:fitted` hole on the adaptive fixture, with a corner the
    # first regrid coarsens: the track, both fits, the finder's seed, the
    # gauge source's sample (Kerr-Schild is not harmonic) and the geometry
    # rebuilt on the new mesh all cross the restarts.
    @testset "a chain on a tracked :fitted hole that regrids is the run" begin
        case = adaptive_hole_fixture(T; interior=FittedSpec(T; variant=:fitted,
                                                            margin=4, n_L=6),
                                     r_0=nothing, r_1=nothing,
                                     horizon=Horizon(T; every=1, N=12, spin=false))
        forest = hole_forest(T, case; N=8, roots=4, radii=(T(5 // 2),))
        corner = filter(k -> all(d -> block_extent(T, forest, k)[d][1] ≥ T(5 // 2),
                                 1:3), forest.leaves)
        refine!(forest, corner)
        balance!(forest)
        ref = evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                      t_end=T(1 // 10), regrid=true)
        @test ref.nregrids == 1
        @test all(r -> r.fit_valid === true && r.horizon_success === true,
                  ref.records[2:end])
        out, jobs, _ = chain(case, forest, T(1 // 10); regrid=true)
        @test jobs == 2
        test_same(out, ref)
    end

    # A boosted hole whose layer follows it: the speed growth that sizes the
    # next chunk's step, the track's velocity that splits a chunk into
    # refills, and step 8′'s trailing ramp.
    @testset "a chain on a moving :fitted hole is the run" begin
        bg = SM.boost(SM.Harmonic(one(T), zero(T)), SVector{3,T}(T(3 // 10), 0, 0))
        case = hole_case(T, bg; halfwidth=T(5 // 2), chunk=T(1 // 20),
                         interior=FittedSpec(T; variant=:fitted, margin=4, n_L=6),
                         horizon=Horizon(T; every=1, N=12, spin=false))
        forest = hole_fixture_forest(T, case; N=8)
        kw = (trail_ramp=T(9 // 10),)
        ref = evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                      t_end=T(3 // 20), kw...)
        @test all(r -> r.horizon_success === true, ref.records)
        out, jobs, _ = chain(case, forest, T(3 // 20); kw...)
        @test jobs == 3
        test_same(out, ref)
    end
end
