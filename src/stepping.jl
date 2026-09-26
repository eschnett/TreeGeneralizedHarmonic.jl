# The time integrator: IMEXRungeKutta's classical RK4, its stage arithmetic
# run by block owner (amended 2026-09-26, replacing OrdinaryDiffEq's RK4;
# `CODE.md`, "Time integration").
#
# Why a second package and not OrdinaryDiffEq: on a 64-core node
# OrdinaryDiffEq's RK4 forms its stage vectors with a serial broadcast on
# one core (12–14 % of a step, and the cross-core migration TreeAMR's
# ownership policy removed from everything else), copies and allocates its
# buffers serially at every `solve`, and evaluates the right-hand side once
# more per `solve` for its FSAL start (measured 2026-09-25, Symmetry job
# 563749; TreeAMR's `CODE.md`, "Open questions"). IMEXRungeKutta's explicit
# `RK4()` forms each combination as one pass over the state, split by the
# partition below so that every block's entries are combined on the thread
# `map_blocks!` runs that block on, and its scratch arrays are first-touched
# through the same partition. The result is bitwise the same on every path
# and at every thread count.

"""
    state_partition(U::FieldSet, u) -> Vector{UnitRange{Int}} or nothing

The block-ownership partition of the state vector `u` of `U`, as
IMEXRungeKutta's `partition` keyword takes it: element `c` holds the entries
of the blocks in chunk `c` of TreeAMR's `threadchunks(nblocks(U))` — the
chunk `launch_by_owner!` and `threaded_chunks` run on default-pool thread
`c` — one element per thread, padded with empty ranges where there are
fewer blocks than threads. A block's entries are contiguous in the state
vector (`statearray`'s last index is the block), so each thread owns one
range.

`nothing` for a state that is not a CPU `Array`, which is IMEXRungeKutta's
broadcast path: a device array is combined by one fused broadcast on the
device, and the partition would be refused there.

TreeAMR's test suite carries the same helper (`test/imex_tests.jl`,
`state_partition`) as a candidate for its own API; this is the copy this
package runs until it is named there.
"""
function state_partition(U::FieldSet{T,D}, u) where {T,D}
    u isa Array || return nothing
    L = U.forest.N^D * U.nvars
    length(u) == L * nblocks(U) || throw(DimensionMismatch(
        "the state vector has $(length(u)) entries but the field set's blocks " *
        "hold $(L * nblocks(U)): a partition of one mesh's state cannot be " *
        "applied to another's."))
    parts = UnitRange{Int}[(first(r) - 1) * L + 1:last(r) * L
                           for r in TreeAMR.threadchunks(nblocks(U))]
    while length(parts) < Threads.nthreads()
        push!(parts, 1:0)
    end
    return parts
end

"""
    gh_limiter!(u, integrator, p, t)

The one limiter [`gh_integrator`](@ref) passes, **as both the stage and the
step limiter** (decided 2026-09-26, Erik: the limiter applies to every state
vector): the range projection ([`gh_stage_limiter!`](@ref)), then the
`:pasted` variant's paste ([`gh_step_limiter!`](@ref)). IMEXRungeKutta calls
the stage limiter on the three stage values of an RK4 step that the
right-hand side reads, and the step limiter once on the step's result, which
the next step's first stage reads; so every right-hand-side input and every
stored state has been projected and pasted — the README's rule for a
correction that must reach every right-hand-side input. Both writers are
no-ops where they do not apply (no bounds; any variant but `:pasted`), and
on a static hole pasting a stage value changes no bit: inside `r_1` the
`:pasted` right-hand side is zero, so the stage value there is `uⁿ`'s paste,
and the analytic solution does not depend on the stage's time.
"""
function gh_limiter!(u, integrator, p, t)
    gh_stage_limiter!(u, integrator, p, t)
    gh_step_limiter!(u, integrator, p, t)
    return nothing
end

"""
    ProblemRef(p)

A mutable holder for a [`GHProblem`](@ref), the `p` of an integrator built
with `swappable = true` ([`gh_integrator`](@ref)): the integrator's own `p`
is a constant, and a moving hole's chunk refills its target — a new
`GHProblem` — between pieces of the same chunk (`CODE.md`, "The fitted
target"). Assigning `integ.p.p = p′` between steps is how the refilled
problem reaches the remaining steps on the same integrator — its step count,
its time and its scratch — instead of a new one per piece. The field is untyped, since a refill can
change the problem's type (its `fits`), which costs one dynamic dispatch per
right-hand side and per limiter call.
"""
mutable struct ProblemRef
    p::Any
end

_gh_rhs_ref!(du, u, r::ProblemRef, t) = gh_rhs!(du, u, r.p, t)
_gh_limiter_ref!(u, integ, r::ProblemRef, t) = gh_limiter!(u, integ, r.p, t)

"""
    gh_integrator(p::GHProblem, u, tspan; dt, alias_u0 = false, swappable = false)

A fixed-step RK4 integrator for `p` from `u` over `tspan = (t0, t1)`, in
steps of at most `dt` — IMEXRungeKutta's `init(IMEXProblem(gh_rhs!, nothing,
u, tspan, p), RK4(); …)` with this package's limiter and the ownership
partition of [`state_partition`](@ref). Advance it with IMEXRungeKutta's
`step!` or `solve!`; its `u` is then the state and its `t` the time.

**The step.** IMEXRungeKutta takes `nsteps = ⌈(t1 − t0)/dt⌉` (with a few
ulps' tolerance, so that a span meant to be a whole number of steps is not
given one more) and `Δt = (t1 − t0)/nsteps`.

**The limiter** (`CODE.md`, "The interior": the state's second and third
writers). [`gh_limiter!`](@ref) — the range projection, then the paste — is
passed as both the stage and the step limiter, so every state vector the
integrator forms is limited: the three stage values of a step that the
right-hand side reads, and the result, which the next step's first stage
reads. Under OrdinaryDiffEq's RK4 only the projection reached the stage
values and the paste was a step limiter alone, after the FSAL evaluation of
the unpasted result; on a static hole the difference is nothing (see
`gh_limiter!`), on a moving one the ball is pasted at every stage's time.

**`alias_u0 = true`** makes the integrator step `u` itself, in place, which
is what [`evolve!`](@ref) does: `u` is the run's state vector, first-touched
by owner, and no copy of it is made. The scratch — four state-sized arrays
for RK4 — is allocated and first-touched through the partition, unless
`reuse` is given.

**`reuse = integ′`**, an earlier integrator on the same mesh, makes this one
take over `integ′`'s scratch instead (IMEXRungeKutta 1.2's `init(…; reuse)`,
used from 2026-09-26): no allocation and no first touch, which was 0.13–0.36 s
at 64 threads on 320 MB. IMEXRungeKutta refuses, by name, scratch that does
not fit — another length, array type or partition — so a caller passes it
only while the mesh is the one `integ′` was built on, and `nothing` after a
regrid. The two share the scratch and must not step at the same time.

**`swappable = true`** gives the integrator a [`ProblemRef`](@ref) as its
`p` instead of `p` itself, so that a caller may replace the problem between
steps (`integ.p.p = p′`) — how [`evolve!`](@ref) refills a moving hole's
target within one chunk on one integrator. `p′` must be a problem on the
same field set.
"""
function gh_integrator(p::GHProblem{T}, u, tspan; dt, alias_u0::Bool=false,
                       swappable::Bool=false, reuse=nothing) where {T}
    t0, t1 = T(tspan[1]), T(tspan[2])
    part = state_partition(p.U, u)
    if swappable
        return IRK.init(IRK.IMEXProblem(_gh_rhs_ref!, nothing, u, (t0, t1),
                                        ProblemRef(p)), IRK.RK4();
                        dt=T(dt), stage_limiter=_gh_limiter_ref!,
                        step_limiter=_gh_limiter_ref!, partition=part,
                        alias_u0=alias_u0, reuse=reuse)
    end
    return IRK.init(IRK.IMEXProblem(gh_rhs!, nothing, u, (t0, t1), p), IRK.RK4();
                    dt=T(dt), stage_limiter=gh_limiter!, step_limiter=gh_limiter!,
                    partition=part, alias_u0=alias_u0, reuse=reuse)
end

"""
    gh_solve(p::GHProblem, u, tspan; dt, alias_u0 = false) -> u′

Integrate `p` from `u` over `tspan` with [`gh_integrator`](@ref) and return
the state at `tspan[2]`: a fresh vector, or `u` itself, updated in place,
with `alias_u0 = true`.
"""
function gh_solve(p::GHProblem, u, tspan; dt, alias_u0::Bool=false)
    integ = gh_integrator(p, u, tspan; dt=dt, alias_u0=alias_u0)
    IRK.solve!(integ)
    return integ.u
end
