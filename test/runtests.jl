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
    include("type_tests.jl")
    include("threading_tests.jl")
end
