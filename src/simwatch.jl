# SimWatch status files (added 2026-10-02).
#
# SimWatch (https://github.com/eschnett/simwatch) is a terminal viewer that
# finds a small `simwatch.toml` in each run directory and shows the runs'
# progress, Slurm state and diagnostics; its `FORMAT.md` is the reference for
# the file. This file started as that repository's drop-in Julia writer,
# `writers/julia/SimWatchStatus.jl` at commit 1b92e6f, and is owned here: it
# depends only on the standard libraries, runs on the host between chunks,
# and never touches the state.
#
# It is layered so that each piece can be used alone:
#
#   simwatch_document(; …)  the document, a `Dict`, from keywords — pure;
#   write_simwatch(dir, doc) the atomic write, which never throws;
#   SimWatchWriter           the state a run carries: its start, the rate
#                            limit, and the spacing of its calls, from which
#                            `update_interval` is reported.
#
# Differences from the original, each a property a long run needs: an I/O
# error is caught and warned about once rather than thrown into the run;
# `update_interval` is the *expected* time to the next write — the larger of
# the minimum spacing and 1.5 times the spacing the caller has actually been
# calling at — so that a run whose chunks take longer than the minimum is not
# shown as stale between them; problem-specific keys may extend the known
# tables (`progress`, `resources`, `slurm`) instead of being overwritten by
# them; `missing`, like `nothing`, is omitted rather than written as a
# placeholder; and the job's start, its wall-time limit and its GPUs are read
# from Slurm's environment when it sets them.
#
# `summary` and `[history]` (the headline keys, and the short windows SimWatch
# draws as sparklines) were ported from the same writer at commit 94c0efe on
# 2026-10-19, with one addition: `simwatch_history!` records a point without
# writing, so that a restarted run can refill its window from its own record.
#
# Brought to SimWatch 1.3.0 on 2026-10-21 (the reference writer at 11cbbeb,
# unchanged since 3fe6f62; the file has no format version): `group`,
# `progress.fraction`, a `slurm` keyword, the earlier jobs of a chain read from
# the status file a previous job left (`slurm.previous_job_ids`, with a
# `next_job_id` that never wrote, as `simwatch_queued` in SimWatch's
# `writers/simwatch.sh` sets it), and an update that never throws. 1.3.0's
# change is to `[[images]]`: PNG, JPEG, SVG, PPM/PGM and WebP, recognised by
# their contents, on a white, black or transparent background that SimWatch
# adapts to its user's terminal (`image_theme`); `test/simwatch_plots.jl`
# writes SVG.

"""
    simwatch_document(; name, status, kw...) -> Dict{String,Any}

The contents of a `simwatch.toml` as a `Dict` ready for `TOML.print`, from
keywords — every one optional, and `nothing` or `missing` omitted:

- top level: `name`, `status` (`"starting"`, `"running"`, `"stopped"`,
  `"finished"`, `"failed"`), `message` (its first line), `updated` and
  `started` (`DateTime`s in UTC), `update_interval` (seconds), `code`,
  `host`, `pid`, `group` (the runs that belong together, e.g. a study's);
- `progress`, `resources`, `slurm`: `NamedTuple`s or `Dict`s of
  `FORMAT.md`'s keys for those tables (`iteration`, `time`, `time_end`,
  `time_unit`, `fraction`, `walltime`, …; `nodes`, `threads`, `gpus`, …;
  `job_id`, `previous_job_ids`, `next_job_id`, …);
- `black_holes`, `images`: vectors of `NamedTuple`s or `Dict`s — an image is
  `(file = "plots/ham.svg", title = …, description = …)`, its path relative
  to the run directory;
- `summary`: the keys of the headline values, dotted (`"shells.r2.ham_l2"`), at
  most six, or tables `(key = …, label = …)`;
- `history`: a `Dict` of equally long vectors, `"time"` and one per series,
  oldest first, each named like the key it tracks;
- `extra`: any further keys and tables, merged in — a table named like a
  known one extends it.
"""
function simwatch_document(; name=nothing, status=nothing, message=nothing,
                           updated=nothing, started=nothing, update_interval=nothing,
                           code=nothing, host=nothing, pid=nothing, group=nothing,
                           progress=nothing, resources=nothing, slurm=nothing,
                           black_holes=nothing, images=nothing, summary=nothing,
                           history=nothing, extra=nothing)
    doc = Dict{String,Any}()
    extra === nothing || merge!(doc, _sw_todict(extra))
    _sw_setkey!(doc, "name", name)
    _sw_setkey!(doc, "status", status)
    _sw_setkey!(doc, "message",
                message === nothing ? nothing : first(split(string(message), '\n')))
    _sw_setkey!(doc, "updated", updated)
    _sw_setkey!(doc, "started", started)
    _sw_setkey!(doc, "update_interval", update_interval)
    _sw_setkey!(doc, "code", code)
    _sw_setkey!(doc, "host", host)
    _sw_setkey!(doc, "pid", pid)
    _sw_setkey!(doc, "group", group)
    for (key, table) in (("progress", progress), ("resources", resources),
                         ("slurm", slurm))
        table === nothing && continue
        old = get(doc, key, nothing)
        t = old isa AbstractDict ? merge!(old, _sw_todict(table)) : _sw_todict(table)
        isempty(t) || (doc[key] = t)
    end
    black_holes === nothing || (doc["black_holes"] = [_sw_todict(b) for b in black_holes])
    images === nothing || (doc["images"] = [_sw_todict(i) for i in images])
    summary === nothing ||
        (doc["summary"] = [s isa AbstractString ? String(s) : _sw_todict(s) for s in summary])
    history === nothing || isempty(history) || (doc["history"] = _sw_todict(history))
    return doc
end

"""
    write_simwatch(dir, doc) -> Bool

Write `doc` to `dir/simwatch.toml` atomically — to `simwatch.toml.tmp`, then
renamed over it, so that a reader never sees half a file — and return
whether it worked. **It never throws** (but for an `InterruptException`): a
status file must not stop a run, so an I/O error is returned as `false`, the
temporary file removed, and the caller decides whether to say so
([`SimWatchWriter`](@ref) warns once).
"""
function write_simwatch(dir::AbstractString, doc::AbstractDict)
    path = joinpath(dir, "simwatch.toml")
    tmp = path * ".tmp"
    try
        mkpath(dir)
        open(tmp, "w") do io
            TOML.print(io, doc; sorted=true)
        end
        # `rename` replaces the target atomically; `mv(; force = true)` would
        # delete it first.
        Base.Filesystem.rename(tmp, path)
        return true
    catch err
        err isa InterruptException && rethrow()
        try
            rm(tmp; force=true)
        catch
        end
        return false
    end
end

"""
    SimWatchWriter(dir; name = basename(dir), code = nothing, group = nothing,
                   summary = nothing, interval = 60, startup_interval = 900,
                   history_length = 100, comm = nothing)

The status file of one run in `dir`, and the state its writes need: when the
run started (the Slurm job's start if Slurm says, else now), when it last
wrote, how far apart its calls have been, and its simulation time at the
first call (from which SimWatch computes the average speed). `interval` is
the least number of seconds between two writes by [`simwatch_update!`](@ref);
`startup_interval` is the `update_interval` reported before the second call,
which has to cover loading and compilation. `group` and `summary` are written
into every document ([`simwatch_document`](@ref)); `history_length` is how
many points of each `history` series are kept.

A run that is one of a chain of Slurm jobs (added 2026-10-21, from SimWatch's
writer): if `dir` already holds a status file written by another job, that
job, the jobs it lists in `slurm.previous_job_ids` and a `next_job_id` that
never wrote (an attempt that died early) are this run's `previous_job_ids`,
oldest first, at most 50 — so the status file is the simulation's and not
the job's. A second run within the same job keeps the jobs the file lists.

`comm` (added 2026-10-19) makes the writer one run's over MPI: every rank
makes it and calls [`simwatch_update!`](@ref) at the same points, which is
then collective — whether to write is agreed from every rank's clock, the
memory is summed over the ranks (the peak too: the job's, not one
process's), and rank 0 alone reads the old file and writes the new one.
`nothing`, the default, is one process. Every rank keeps the history window;
rank 0's is the one written.
"""
mutable struct SimWatchWriter
    dir::String
    name::String
    code::Union{Nothing,String}
    group::Union{Nothing,String}
    interval::Float64
    startup_interval::Float64
    started::DateTime
    t0::Float64                 # `time()` at the start
    last_write::Float64         # `time()` of the last write
    last_call::Float64          # `time()` of the last call
    gap::Float64                # the spacing of the last two calls
    time_start::Union{Nothing,Float64}
    summary::Any
    history_length::Int
    history_time::Vector{Float64}
    history::Dict{String,Vector{Float64}}
    previous_job_ids::Vector{String}
    warned::Bool
    comm::TreeAMR.Communicator
end

function SimWatchWriter(dir::AbstractString; name::AbstractString=basename(abspath(dir)),
                        code=nothing, group=nothing, summary=nothing, interval::Real=60,
                        startup_interval::Real=900, history_length::Integer=100,
                        comm=nothing)
    t = time()
    job = _sw_envfloat("SLURM_JOB_START_TIME")
    c = communicator(comm)
    # Rank 0's start, so that every rank reports the same wall time.
    t0 = rank_bcast(c, job === nothing ? t : job)
    started = Dates.unix2datetime(t0)
    previous = isroot(c) ? _sw_previous_jobs(dir, get(ENV, "SLURM_JOB_ID", nothing)) :
               String[]
    return SimWatchWriter(String(dir), String(name), _sw_maybe_string(code),
                          _sw_maybe_string(group), Float64(interval),
                          Float64(startup_interval), started, t0, -Inf, -Inf, NaN,
                          nothing, summary, Int(history_length), Float64[],
                          Dict{String,Vector{Float64}}(), previous, false, c)
end

"""
    simwatch_update!(sw::SimWatchWriter; force = false, status = "running",
                     message, iteration, time, time_end, time_unit, fraction,
                     speed, walltime_limit, checkpoint, progress, resources,
                     slurm, black_holes, images, history, extra) -> Bool

Write the status file if `sw.interval` seconds have passed since the last
write, or if `force`; return whether it was written. Cheap to call after
every step or chunk: every call is timed, and the reported `update_interval`
is `max(interval, 1.5 × the last spacing of calls)` — or `startup_interval`
before there is one — so that a run that calls every seven minutes is not
shown as stale after three. The keywords are [`simwatch_document`](@ref)'s;
`progress`, `resources` and `slurm` add keys to those tables, and `fraction`
(0 to 1) is the progress of a run that is not measured in simulation time.
`walltime_limit` is the Slurm job's own unless given. `history =
Dict("constraints.ham_l2" => x, …)` adds one point per *written* update at the
current `time` (none without a `time`), as [`simwatch_history!`](@ref) does;
SimWatch shows the window as sparklines with their growth rates. **It never
throws** (but for an `InterruptException`): a failure to build or write the
file is warned about once and returned as `false`.
"""
function simwatch_update!(sw::SimWatchWriter; force::Bool=false,
                          status::AbstractString="running", message=nothing,
                          iteration=nothing, time=nothing, time_end=nothing,
                          time_unit=nothing, fraction=nothing, speed=nothing,
                          walltime_limit=nothing, checkpoint=nothing, progress=nothing,
                          resources=nothing, slurm=nothing, black_holes=nothing,
                          images=nothing, history=nothing, extra=nothing)
    t = Base.time()
    isfinite(sw.last_call) && (sw.gap = t - sw.last_call)
    sw.last_call = t
    if sw.time_start === nothing && time !== nothing
        sw.time_start = Float64(time)
    end
    due = force || t - sw.last_write ≥ sw.interval
    (due,) = agree_any(sw.comm, (due,))
    due || return false
    sw.last_write = t
    time === nothing || history === nothing || simwatch_history!(sw, time, history)
    # Every rank's memory, summed (Linux only, where the resident set is
    # read; a rank that cannot read it contributes nothing). These are the
    # collective calls, made on every rank before rank 0 alone goes on.
    rss = _sw_current_rss()
    rss = rank_reduce(+, sw.comm, something(rss, 0); present=rss !== nothing)
    rss = rss == 0 ? _sw_current_rss() : rss
    peak = rank_reduce(+, sw.comm, Int(Sys.maxrss()))
    isroot(sw.comm) || return true
    try
        expected = isnan(sw.gap) ? max(sw.interval, sw.startup_interval) :
                   max(sw.interval, 3 * sw.gap / 2)
        limit = walltime_limit !== nothing ? walltime_limit :
                _sw_slurm_walltime_limit()
        prog = Dict{String,Any}()
        for (k, v) in (("iteration", iteration), ("time", time),
                       ("time_start", time === nothing ? nothing : sw.time_start),
                       ("time_end", time_end), ("time_unit", time_unit),
                       ("fraction", fraction), ("walltime", t - sw.t0),
                       ("walltime_limit", limit), ("speed", speed),
                       ("checkpoint", checkpoint))
            _sw_setkey!(prog, k, v)
        end
        progress === nothing || merge!(prog, _sw_todict(progress))
        res = Dict{String,Any}()
        for (k, v) in (("nodes", _sw_envint("SLURM_JOB_NUM_NODES")),
                       ("tasks", _sw_envint("SLURM_NTASKS")),
                       ("threads", Threads.nthreads()), ("gpus", _sw_gpus()),
                       ("memory_bytes", rss),
                       ("memory_peak_bytes", peak))
            _sw_setkey!(res, k, v)
        end
        resources === nothing || merge!(res, _sw_todict(resources))
        sl = Dict{String,Any}()
        for (k, v) in (("job_id", get(ENV, "SLURM_JOB_ID", nothing)),
                       ("job_name", get(ENV, "SLURM_JOB_NAME", nothing)),
                       ("partition", get(ENV, "SLURM_JOB_PARTITION", nothing)),
                       ("previous_job_ids",
                        isempty(sw.previous_job_ids) ? nothing : sw.previous_job_ids))
            _sw_setkey!(sl, k, v)
        end
        slurm === nothing || merge!(sl, _sw_todict(slurm))
        doc = simwatch_document(; name=sw.name, status=status, message=message,
                                updated=Dates.now(Dates.UTC), started=sw.started,
                                update_interval=round(expected), code=sw.code,
                                host=gethostname(), pid=getpid(), group=sw.group,
                                progress=prog, resources=res,
                                slurm=isempty(sl) ? nothing : sl,
                                black_holes=black_holes, images=images,
                                summary=sw.summary, history=_sw_history_document(sw),
                                extra=extra)
        ok = write_simwatch(sw.dir, doc)
        ok || _sw_warn_once(sw, "could not write $(joinpath(sw.dir, "simwatch.toml"))")
        return ok
    catch err
        err isa InterruptException && rethrow()
        _sw_warn_once(sw, sprint(showerror, err))
        return false
    end
end

"""
    simwatch_finish!(sw::SimWatchWriter; status = "finished", kw...) -> Bool

Write the final status now: `"finished"`, `"stopped"` (at a wall-time limit,
to be continued) or `"failed"` with the error's first line as `message`.
"""
simwatch_finish!(sw::SimWatchWriter; status::AbstractString="finished", kw...) =
    simwatch_update!(sw; force=true, status=status, kw...)

function _sw_warn_once(sw::SimWatchWriter, what)
    sw.warned && return nothing
    sw.warned = true
    @warn "SimWatch: $what; the run continues without a status file"
    return nothing
end

# --- history --------------------------------------------------------------------

"""
    simwatch_history!(sw::SimWatchWriter, time, vals) -> sw

Add one point at simulation time `time` to the writer's `[history]` window,
without writing: `vals` maps series names to numbers (six significant
digits are kept; anything else is `NaN`). A series missing from `vals` is
padded with `NaN`, the window keeps the last `history_length` points, and a
series with no finite value left in it is dropped. [`simwatch_update!`](@ref)
calls it for its own `history`; call it directly to refill the window after a
restart.
"""
function simwatch_history!(sw::SimWatchWriter, time::Real, vals)
    n = length(sw.history_time)
    push!(sw.history_time, Float64(time))
    for (k, v) in pairs(vals)
        x = v isa Real && !(v isa Bool) ? round(Float64(v); sigdigits=6) : NaN
        push!(get!(() -> fill(NaN, n), sw.history, string(k)), x)
    end
    for series in values(sw.history)
        length(series) == n && push!(series, NaN)
    end
    drop = length(sw.history_time) - sw.history_length
    if drop > 0
        deleteat!(sw.history_time, 1:drop)
        foreach(s -> deleteat!(s, 1:drop), values(sw.history))
    end
    filter!(((_, s),) -> any(isfinite, s), sw.history)
    return sw
end

function _sw_history_document(sw::SimWatchWriter)
    isempty(sw.history) && return nothing
    doc = Dict{String,Any}("time" => copy(sw.history_time))
    for (k, s) in sw.history
        doc[k] = copy(s)
    end
    return doc
end

# --- TOML values ----------------------------------------------------------------

_sw_setkey!(d::AbstractDict, k, ::Nothing) = d
_sw_setkey!(d::AbstractDict, k, ::Missing) = d
_sw_setkey!(d::AbstractDict, k, v) = (d[k] = _sw_value(v); d)

_sw_value(x::Bool) = x
_sw_value(x::Integer) = Int64(x)
_sw_value(x::Real) = Float64(x)       # `nan` and `±inf` are TOML
_sw_value(x::AbstractString) = String(x)
_sw_value(x::Symbol) = String(x)
# Julia's TOML writer cannot write an offset, so a time is an RFC 3339 string
# in UTC, which `FORMAT.md` accepts.
_sw_value(x::DateTime) = Dates.format(x, Dates.dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
_sw_value(x::Union{AbstractVector,Tuple}) =
    Any[_sw_value(y) for y in x if y !== nothing && y !== missing]
_sw_value(x::Union{NamedTuple,AbstractDict}) = _sw_todict(x)
_sw_value(x) = string(x)

function _sw_todict(x::Union{NamedTuple,AbstractDict})
    d = Dict{String,Any}()
    for (k, v) in pairs(x)
        _sw_setkey!(d, string(k), v)
    end
    return d
end

_sw_maybe_string(x) = x === nothing ? nothing : String(x)

# --- what the environment says ------------------------------------------------

_sw_envint(name) = tryparse(Int, get(ENV, name, ""))
_sw_envfloat(name) = tryparse(Float64, get(ENV, name, ""))

# The Slurm job's wall-time limit in seconds, where Slurm sets its start and
# end times (it does from 22.05 on), else `nothing`.
function _sw_slurm_walltime_limit()
    s, e = _sw_envfloat("SLURM_JOB_START_TIME"), _sw_envfloat("SLURM_JOB_END_TIME")
    return s === nothing || e === nothing ? nothing : e - s
end

# The GPUs the job was given: Slurm's count, else the visible CUDA devices.
function _sw_gpus()
    n = _sw_envint("SLURM_GPUS_ON_NODE")
    n === nothing || return n
    v = strip(get(ENV, "CUDA_VISIBLE_DEVICES", ""))
    (isempty(v) || v == "NoDevFiles" || v == "-1") && return nothing
    return count(==(','), v) + 1
end

# The resident memory in bytes (Linux only).
function _sw_current_rss()
    Sys.islinux() || return nothing
    try
        pages = parse(Int, split(read("/proc/self/statm", String))[2])
        return pages * ccall(:getpagesize, Cint, ())
    catch
        return nothing
    end
end

# The earlier Slurm jobs of the simulation in `dir`, oldest first, at most 50
# (SimWatch's writer): those its existing status file lists, then the job that
# wrote it, then a `next_job_id` that never wrote; none if there is no file or
# it cannot be read. If the file is the current job's own (a second run in one
# job), only the jobs it lists — where SimWatch's writer forgets them.
function _sw_previous_jobs(dir, current)
    try
        path = joinpath(dir, "simwatch.toml")
        isfile(path) && filesize(path) ≤ 128 * 1024 || return String[]
        slurm = get(TOML.parsefile(path), "slurm", nothing)
        slurm isa AbstractDict || return String[]
        id(x) = x isa Union{AbstractString,Integer} ? String(strip(string(x))) : nothing
        old = id(get(slurm, "job_id", nothing))
        own = old !== nothing && old == current
        jobs = String[]
        prev = get(slurm, "previous_job_ids", [])
        prev isa AbstractVector && append!(jobs, filter(!isnothing, id.(prev)))
        own || old === nothing || push!(jobs, old)
        next = id(get(slurm, "next_job_id", nothing))
        own || next === nothing || push!(jobs, next)
        jobs = unique(filter(!=(current), jobs))
        return jobs[max(1, end - 49):end]
    catch err
        err isa InterruptException && rethrow()
        return String[]
    end
end
