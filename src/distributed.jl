# Running over MPI (added 2026-10-19, on TreeAMR's M7).
#
# TreeAMR distributes a forest's *blocks* over the ranks of a communicator
# and keeps the forest itself on every rank, and everything it does across
# ranks — the ghost exchange, the interface restriction, the regrid, the
# point interpolation, the checkpoints, `mesh_mapreduce` — it does itself.
# What it cannot do is make a number *this package* combines from per-block
# values the same on every rank, agree on a decision taken from a rank's own
# clock or file system, or decide where the host-side analysis runs. Those
# are what this file is for, and nothing in it is mesh machinery: it is the
# application's half of "What an application must make global itself" in
# TreeAMR's `CODE.md`. The first five functions are TreeHydro's
# (`src/distributed.jl` there, 2026-10-02), restated.
#
# The verbs are TreeAMR's — `commrank`, `commsize`, `allgather`,
# `allgatherv`, `bcast` — reached through `forest.comm`, which is a
# `TreeAMR.Communicator` and a `SerialCommunicator` on a forest built
# without one. They are unexported upstream, so `test/prerequisite_tests.jl`
# checks them by name. This package never names MPI: a serial run takes
# every function here through its one-rank branch, which returns its
# argument untouched and communicates nothing, so every serial number in
# `CODE.md` is where it was. See "Running distributed" in `CODE.md`.

using Base.ScopedValues: ScopedValue, with
using Serialization: deserialize, serialize

# The communicator of whatever a reduction is over.
rank_comm(comm::TreeAMR.Communicator) = comm
rank_comm(forest::Forest) = forest.comm
rank_comm(fs::FieldSet) = fs.forest.comm

isdistributed_over(over) = TreeAMR.commsize(rank_comm(over)) > 1

"""
    rank_reduce(op, over, value; present = true)

`value` combined over every rank of `over` — a forest, a field set or a
`TreeAMR.Communicator` — with `op`, **in rank order**, the same on every
rank: collective. `present = false` says this rank has nothing to
contribute, and its `value` is then left out rather than folded in, which is
TreeAMR's rule for `mesh_mapreduce`; where no rank is present the result is
`value`, which every rank then passes the same.

One `allgather` of a `(present, value)` pair, folded in rank order, so the
association is the package's and every rank gets the same bits. On one rank
it returns `value` itself and gathers nothing. `value` must be `isbits`.
TreeHydro's.
"""
function rank_reduce(op, over, value; present::Bool=true)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return value
    parts = TreeAMR.allgather(comm, (present, value))::Vector{Tuple{Bool,typeof(value)}}
    acc, found = value, false
    for (has, v) in parts
        has || continue
        acc = found ? op(acc, v) : v
        found = true
    end
    return acc
end

# A tuple of decisions, `true` on every rank where it is `true` on any: how a
# checkpoint taken by the wall clock is agreed, since a rank that entered the
# collective `save_checkpoint` while another went on to its regrid would wait
# forever.
agree_any(over, flags::NTuple{N,Bool}) where {N} =
    rank_reduce((a, b) -> map(|, a, b), over, flags)

# Whether this rank is the one that does what only one rank may — remove a
# file, print a line, write a status file.
isroot(over) = TreeAMR.commrank(rank_comm(over)) == 0

"""
    agree_refusal(f, over)

Run the checks `f()` on every rank, and if one of them throws an
`ArgumentError` on any rank, throw on **every** rank: the rank that refused
its own error, and the others one naming the ranks that refused. For checks
that can come out differently on different ranks — a directory's existence,
or a per-block check over each rank's own blocks — since a refusal on one
rank alone would leave the others waiting in their next collective. Any
other exception is a bug and is rethrown at once. Serially this is `f()`.
TreeHydro's.
"""
function agree_refusal(f, over)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return f()
    err = nothing
    result = nothing
    try
        result = f()
    catch e
        e isa ArgumentError || rethrow()
        err = e
    end
    refused = TreeAMR.allgather(comm, err !== nothing)::Vector{Bool}
    err === nothing || throw(err)
    any(refused) && throw(ArgumentError(
        "rank(s) $(join(findall(refused) .- 1, ", ")) refused the arguments, and " *
        "this rank did not: a check that reads the file system, or this rank's " *
        "own blocks, can come out differently on different ranks, and the run " *
        "stops on every rank rather than leave the others waiting for the one " *
        "that refused. Its error is printed by that rank."))
    return result
end

"""
    rank_bcast(over, x) -> x

Rank 0's `x` on every rank: collective. Any value `Serialization` can write
— a host struct with arrays, a named tuple of records, an exception — goes
as its bytes through TreeAMR's `bcast`; the other ranks pass anything (their
`x` is ignored) and get rank 0's back. On one rank it returns `x` itself.

The bytes are read by the same build of the same packages on every rank, so
what comes back is rank 0's value bit for bit — which is the point: a value
computed on rank 0 alone and broadcast is the same on every rank by
construction, whatever each rank's BLAS or vector width would have made of
it.
"""
function rank_bcast(over, x)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return x
    bytes = if TreeAMR.commrank(comm) == 0
        io = IOBuffer()
        serialize(io, x)
        take!(io)
    else
        UInt8[]
    end
    got = TreeAMR.bcast(comm, bytes, 0)
    TreeAMR.commrank(comm) == 0 && return x
    return deserialize(IOBuffer(got))
end

# --- the host analysis on rank 0 ---------------------------------------------
#
# The horizon finder and the fitted target's fit are host computations with
# an iteration count nobody knows in advance — the fast flow's, the spin's —
# that read the state through TreeAMR's `interpolate`, which is collective.
# They run on **rank 0 alone** (decided 2026-10-19), and the other ranks
# *serve* them: every collective step rank 0 takes inside the analysis is
# announced first by a broadcast request, which the others answer by taking
# the same step with no points of their own. Two steps exist, and both go
# through one function each, which is what makes the protocol hold:
#
#   * `(:fill, t)` — scatter the state and fill the ghosts at `t`
#     ([`fill_state!`](@ref)), which every interpolation needs first;
#   * `(:interp, q, derivs, vars, mask)` — one call of TreeAMR's
#     `interpolate` ([`interpolate_state`](@ref)), whose arguments every
#     rank must pass alike: TreeAMR hashes the basis, the derivatives, the
#     variables and the excluded region and refuses a disagreement.
#
# When the analysis returns, rank 0 broadcasts `:stop` and then its result,
# which every rank adopts. **Any new collective inside the analysis must go
# through a request, or the other ranks hang** ("Things that will bite").

# What rank 0 is serving while an analysis runs: the field set the requests
# are about, and the communicator they go over. `nothing` outside one, and
# always `nothing` on the other ranks.
struct ServedAnalysis{F,C}
    fs::F
    comm::C
end

const SERVED = ScopedValue{Union{Nothing,ServedAnalysis}}(nothing)

# Announce a collective step of the analysis to the serving ranks: a no-op
# outside a served analysis, and so serially and on every rank but 0.
function served_request!(fs::FieldSet, request)
    s = SERVED[]
    s === nothing && return nothing
    s.fs === fs || error(
        "internal error: an analysis served on rank 0 made a collective call " *
        "on a field set other than the one the other ranks serve; they would " *
        "answer it on the wrong one, or hang.")
    rank_bcast(s.comm, request)
    return nothing
end

# The marker a failed analysis sends in place of its result.
struct RootFailure
    exception::Any
end

"""
    on_root(f, over; serve = nothing)

`f()` on rank 0, and its result on every rank: collective. Serially `f()`.

`serve = (p, u)` — a [`GHProblem`](@ref) and this rank's state vector —
makes the other ranks answer `f`'s collective steps while it runs (see the
comment above [`served_request!`](@ref)): what [`find_gh_horizon`](@ref) and
the state sampler of [`build_fit`](@ref) need. Without it the other ranks
only wait for the result, which is what a computation that reads no state —
the fit of the analytic solution — needs.

A failure of `f` is every rank's: rank 0 rethrows its exception, and the
others throw the same exception, broadcast (or, if it cannot be serialized,
an `ErrorException` with its message), so that a `catch` that tests the
exception's type does the same thing on every rank.
"""
function on_root(f, over; serve=nothing)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return f()
    if TreeAMR.commrank(comm) == 0
        result = try
            serve === nothing ? f() :
            with(f, SERVED => ServedAnalysis(serve[1].U, comm))
        catch e
            serve === nothing || rank_bcast(comm, (:stop,))
            rank_bcast(comm, RootFailure(sendable_exception(e)))
            rethrow()
        end
        serve === nothing || rank_bcast(comm, (:stop,))
        rank_bcast(comm, result)
        return result
    end
    serve === nothing || serve_requests(serve[1], serve[2], comm)
    got = rank_bcast(comm, nothing)
    got isa RootFailure && throw(got.exception)
    return got
end

# The exception itself where it can be serialized, and its message otherwise
# (one that holds a task or a closure over a device array cannot be).
function sendable_exception(e)
    try
        serialize(IOBuffer(), e)
        return e
    catch
        return ErrorException("rank 0 failed: " * sprint(showerror, e))
    end
end

# The other ranks' half: answer requests until rank 0 says `:stop`. A refusal
# TreeAMR agreed across the ranks — a point outside the domain — reached
# rank 0 too, whose analysis decides what it means (the finder records it as
# a failed find), so it is swallowed here and the loop waits for the next
# request.
function serve_requests(p, u, comm)
    T = eltype(p.U.work)
    while true
        req = rank_bcast(comm, nothing)
        kind = first(req)
        kind === :stop && return nothing
        if kind === :fill
            fill_state!(p, u, req[2])
        elseif kind === :interp
            _, q, derivs, vars, mask = req
            try
                interpolate_state(p.U, SVector{3,T}[], q, mask, derivs; vars=vars)
            catch e
                e isa ArgumentError || rethrow()
            end
        else
            error("internal error: rank 0 sent the unknown request $(repr(kind)).")
        end
    end
end
