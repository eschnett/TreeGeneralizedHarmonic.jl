# Thread-count independence (G3).
#
# There is no switch to test. TreeAMR's kernels and host-side passes
# thread themselves, this package adds no loop of its own that does, and
# every reduction it takes goes through `block_mapreduce` with the
# per-block partials combined **in block order**. What has to be guarded
# is the *invariant* that makes all of that safe to rely on — the answer
# does not move when the thread count does, to the last bit — because
# nothing else in the suite would notice if it did (`CODE.md`, "Precision,
# threads, devices"; `CLAUDE.md`, "Never thread anything a TreeAMR
# callback can reach").
#
# Every assertion here is exact equality, never `≈`. Roundoff-level
# agreement is what a *reassociated* sum gives, and a reassociated sum is
# exactly the bug.
#
# What this bit-identity does and does not mean is worth saying, because
# `CODE.md`'s "Measured results" spends a paragraph on the other half: the
# same compiled code, at the same call site, at a different thread count,
# and nothing more. Two spellings of one expression — even one body
# reached from two call sites — are *not* bit-identical here, and no test
# in this package asks them to be.

include("thread_workload.jl")

@testset "A run is bit-identical across thread counts" begin
    # The acceptance test for G3's third claim. The thread count is a
    # command-line argument to Julia and cannot be changed from inside a
    # running session, so the comparison is against a subprocess started at
    # a different count. A reduction partitioned by thread rather than by
    # block would pass every other test in this suite and fail here.
    reference = thread_digests()
    # Setup, two chunks, two regrids, and the state the last regrid moved.
    @test length(reference) == 6
    # A regrid that actually changed the mesh, and a chunk that ran on one
    # with a coarse-fine face — without both, the digests would be a claim
    # about a uniform mesh.
    @test any(l -> occursin("changed true", l), reference)
    @test any(l -> occursin("schedule", l) && occursin("[1]", l), reference)

    other = Threads.nthreads() == 1 ? max(2, min(4, Sys.CPU_THREADS)) : 1
    script = joinpath(@__DIR__, "thread_workload.jl")
    # The *active* project, not `test/`: under `Pkg.test` the tests run in
    # a sandbox and `test/Project.toml` has no manifest of its own.
    project = Base.active_project()
    out = read(`$(Base.julia_cmd()) --threads=$other --project=$project $script`,
               String)

    @test split(chomp(out), '\n') == reference
end

@testset "The masked norms are a fold in block order" begin
    # The cheap, local half of the claim above, and the one that fails
    # legibly: the subprocess test says only that two long outputs differ,
    # while this names the function. `masked_norms` is the one reduction
    # this package writes itself — `block_mapreduce` returns the per-block
    # partials and the *combination* is ours — so it is the one place a
    # running total split across tasks could appear. Checked against a
    # serial loop over blocks, with exact equality.
    T = Float64
    q = 4
    case = shifted_minkowski_case(T; ε_KO=zero(T), γ0=one(T), γ2=T(-1 // 2))
    forest, fs, prob = gh_setup(T, case; N=8, roots=2, q=q)
    fill_exact!(fs, case, zero(T))
    u = statevector(fs)
    gather!(u, fs)
    gh_constraint!(prob, u, zero(T))

    for v in (TreeGeneralizedHarmonic.DIAG_CGH,
              TreeGeneralizedHarmonic.DIAG_CGH + 2)
        num = zero(T)
        den = zero(T)
        peak = zero(T)
        for b in 1:nblocks(prob.diag)
            cellvolume = spacing(T, forest, blockkey(prob.diag, b))^3
            block = interiorview(prob.diag, b, v)
            num += cellvolume * sum(x -> x * x, block)
            den += cellvolume * length(block)
            peak = max(peak, maximum(abs, block))
        end
        n = masked_norms(prob, v)
        @test n.linf === peak
        @test n.l2 === sqrt(num / den)
    end

    # And `max_speed`, the other reduction on the per-chunk path.
    scatter!(fs, u)
    λ = max_speed(prob)
    speeds = block_mapreduce(identity, max, zero(T), prob.diag;
                             vars=TreeGeneralizedHarmonic.DIAG_SPEED)
    @test λ === maximum(speeds)
    @test λ > sqrt(T(3))                   # this case has a shift
end
