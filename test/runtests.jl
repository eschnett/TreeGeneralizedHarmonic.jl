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
end
