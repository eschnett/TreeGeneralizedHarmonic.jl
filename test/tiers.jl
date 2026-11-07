# The suite's tiers (added 2026-11-06), as a helper rather than a test:
# `runtests.jl` includes it first, and every file that marks a testset
# includes it if it was not.
#
# Evolving a black hole is what this package is for, and it is what a
# workstation or Symmetry does, not a free GitHub runner: there the serial
# suite took 72 minutes, 46 of them in a few dozen testsets that evolve a
# hole (four cores, about 1.7× slower than the development machine, and
# `--check-bounds=yes` another 1.65×). So the suite has two tiers:
#
#   * `full`, the default — every testset. This is what `Pkg.test()` runs on
#     a workstation before a pull request and on Symmetry, and what
#     `ci/full-suite.sh` runs and reports as a commit status.
#   * `quick` — what GitHub CI runs (`TREEGH_TEST_TIER=quick`): every testset
#     but the hole evolutions marked `@full_tier`. It keeps one short hole run
#     through `evolve!` (the record's), the kernels on every branch, the
#     algebra, the gauge wave's convergence and the thread invariance — what
#     CI adds there is a clean environment, x86-64 and bounds checks, not the
#     evolutions.
#   * `mpi` — the MPI comparison alone, which CI runs as a job of its own
#     beside the other two cells.
#
# A testset left out by the quick tier is still a testset there, holding one
# skipped test, so the summary counts it as "Broken" and names it: a green
# tick never claims more than it ran.

const TEST_TIER = let tier = get(ENV, "TREEGH_TEST_TIER", "full")
    tier in ("full", "quick", "mpi") ||
        error("TREEGH_TEST_TIER=$tier: the tiers are full (the default), " *
              "quick (GitHub CI) and mpi (the MPI comparison alone)")
    Symbol(tier)
end

# The names of the testsets the quick tier left out, which `runtests.jl`
# lists at the end.
const FULL_TIER_SKIPPED = String[]

# `@full_tier @testset "name" begin … end`: the testset in the full tier, and
# in the quick tier a testset of the same name holding one skipped test. The
# name must be a literal string, so that the skip can name it without
# evaluating anything the testset would.
macro full_tier(ex)
    ex isa Expr && ex.head === :macrocall && ex.args[1] === Symbol("@testset") ||
        error("@full_tier marks a `@testset`")
    i = findfirst(a -> a isa String, ex.args)
    i === nothing &&
        error("@full_tier needs a testset whose name is a literal string")
    name = ex.args[i]
    return quote
        if TEST_TIER === :full
            $(esc(ex))
        else
            push!(FULL_TIER_SKIPPED, $name)
            @testset $name begin
                @test_skip "evolves a hole: full tier only (TREEGH_TEST_TIER=full)"
            end
        end
    end
end
