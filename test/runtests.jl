using Test
using TreeAMR
using TreeGeneralizedHarmonic

# The suite is expected to pass, with identical numbers, at any thread
# count; CI runs it at one and at four. `Pkg.test` does not inherit `-t`,
# so the thread count has to be passed explicitly — see `CLAUDE.md`.
@info "Running the tests on $(Threads.nthreads()) thread(s)"

# `verbose = true` so that the per-testset times are printed even when
# everything passes (added in step 1). The suite's cost is now compilation
# of `SpacetimeMetrics`' dual passes, and which testset pays it is a thing
# a reader of the CI log should be able to see without a failure first.
@testset verbose = true "TreeGeneralizedHarmonic.jl" begin
    include("precision_tests.jl")
    include("prerequisite_tests.jl")
    include("pointwise_tests.jl")
    include("pointwise_identity_tests.jl")
    include("stencils_tests.jl")
    include("gauge_tests.jl")
    include("initialdata_tests.jl")
    include("evolution_tests.jl")
    # `evolution_cases.jl` is a helper, not a test file: the two studies
    # below are runs, and the runs are written once (TreeAMR includes
    # `test/wave.jl` the same way).
    include("evolution_cases.jl")
    include("convergence_tests.jl")
    include("noise_tests.jl")
    # Step 4's four. They come after `evolution_cases.jl` because all of
    # them are runs; `threading_tests.jl` is last because it starts a
    # subprocess, and a failure anywhere above is cheaper to read than a
    # diff of two long outputs.
    include("constraints_tests.jl")
    include("interface_tests.jl")
    # Step 5's two. `interior_tests.jl` is the algebra, the predicates and
    # one right-hand-side evaluation; `driver_tests.jl` is the runs, and
    # it comes after every other run file for the same reason
    # `threading_tests.jl` comes last — a hole is the suite's most
    # expensive thing, and a failure above it is cheaper to read.
    include("interior_tests.jl")
    include("driver_tests.jl")
    # Step 6's. After `driver_tests.jl` because its last testset is a run
    # through `evolve!` with the regrid branch on, and a failure in the
    # driver itself is cheaper to read than one in the loop that drives it.
    include("refinement_tests.jl")
    # Step 7's. After `driver_tests.jl` and `refinement_tests.jl` because
    # it finds the horizon of both of their meshes — the fixture's frozen
    # hierarchy and the one the indicator chose — and because a failure in
    # either is cheaper to read than one in the analysis on top of them.
    include("horizon_tests.jl")
    include("bounds_tests.jl")      # step 8b: the range projection, and its control
    include("tracking_tests.jl")    # step 8d: the tracked geometry, and its runs
    include("fit_tests.jl")         # step 8e: the fitted target (shares 8d's run)
    include("moving_tests.jl")      # step 8: the moving hole's floor, seed and cycle
    include("type_tests.jl")
    include("threading_tests.jl")
end
