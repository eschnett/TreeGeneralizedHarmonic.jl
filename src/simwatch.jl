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

"""
    simwatch_document(; name, status, kw...) -> Dict{String,Any}

The contents of a `simwatch.toml` as a `Dict` ready for `TOML.print`, from
keywords — every one optional, and `nothing` or `missing` omitted:

- top level: `name`, `status` (`"starting"`, `"running"`, `"stopped"`,
  `"finished"`, `"failed"`), `message` (its first line), `updated` and
  `started` (`DateTime`s in UTC), `update_interval` (seconds), `code`,
  `host`, `pid`;
- `progress`, `resources`, `slurm`: `NamedTuple`s or `Dict`s of
  `FORMAT.md`'s keys for those tables (`iteration`, `time`, `time_end`,
  `time_unit`, `walltime`, …; `nodes`, `threads`, `gpus`, …; `job_id`, …);
- `black_holes`, `images`: vectors of `NamedTuple`s or `Dict`s;
- `extra`: any further keys and tables, merged in — a table named like a
  known one extends it.
"""
function simwatch_document(; name=nothing, status=nothing, message=nothing,
                           updated=nothing, started=nothing, update_interval=nothing,
                           code=nothing, host=nothing, pid=nothing, progress=nothing,
                           resources=nothing, slurm=nothing, black_holes=nothing,
                           images=nothing, extra=nothing)
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
    for (key, table) in (("progress", progress), ("resources", resources),
                         ("slurm", slurm))
        table === nothing && continue
        t = merge!(get(doc, key, Dict{String,Any}()), _sw_todict(table))
        isempty(t) || (doc[key] = t)
    end
    black_holes === nothing || (doc["black_holes"] = [_sw_todict(b) for b in black_holes])
    images === nothing || (doc["images"] = [_sw_todict(i) for i in images])
    return doc
end

"""
    write_simwatch(dir, doc) -> Bool

Write `doc` to `dir/simwatch.toml` atomically — to `simwatch.toml.tmp`, then
renamed over it, so that a reader never sees half a file — and return
whether it worked. **It never throws**: a status file must not stop a run,
so an I/O error is returned as `false`, and the caller decides whether to
say so ([`SimWatchWriter`](@ref) warns once).
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
    catch
        return false
    end
end

"""
    SimWatchWriter(dir; name = basename(dir), code = nothing, interval = 60,
                   startup_interval = 900)

The status file of one run in `dir`, and the state its writes need: when the
run started (the Slurm job's start if Slurm says, else now), when it last
wrote, how far apart its calls have been, and its simulation time at the
first call (from which SimWatch computes the average speed). `interval` is
the least number of seconds between two writes by [`simwatch_update!`](@ref);
`startup_interval` is the `update_interval` reported before the second call,
which has to cover loading and compilation.
"""
mutable struct SimWatchWriter
    dir::String
    name::String
    code::Union{Nothing,String}
    interval::Float64
    startup_interval::Float64
    started::DateTime
    t0::Float64                 # `time()` at the start
    last_write::Float64         # `time()` of the last write
    last_call::Float64          # `time()` of the last call
    gap::Float64                # the spacing of the last two calls
    time_start::Union{Nothing,Float64}
    warned::Bool
end

function SimWatchWriter(dir::AbstractString; name::AbstractString=basename(abspath(dir)),
                        code=nothing, interval::Real=60, startup_interval::Real=900)
    t = time()
    job = _sw_envfloat("SLURM_JOB_START_TIME")
    t0 = job === nothing ? t : job
    started = Dates.unix2datetime(t0)
    return SimWatchWriter(String(dir), String(name),
                          code === nothing ? nothing : String(code), Float64(interval),
                          Float64(startup_interval), started, t0, -Inf, -Inf, NaN,
                          nothing, false)
end

"""
    simwatch_update!(sw::SimWatchWriter; force = false, status = "running",
                     message, iteration, time, time_end, time_unit, speed,
                     walltime_limit, checkpoint, progress, resources,
                     black_holes, images, extra) -> Bool

Write the status file if `sw.interval` seconds have passed since the last
write, or if `force`; return whether it was written. Cheap to call after
every step or chunk: every call is timed, and the reported `update_interval`
is `max(interval, 1.5 × the last spacing of calls)` — or `startup_interval`
before there is one — so that a run that calls every seven minutes is not
shown as stale after three. The keywords are [`simwatch_document`](@ref)'s;
`progress` and `resources` add keys to those tables. `walltime_limit` is the
Slurm job's own unless given.
"""
function simwatch_update!(sw::SimWatchWriter; force::Bool=false,
                          status::AbstractString="running", message=nothing,
                          iteration=nothing, time=nothing, time_end=nothing,
                          time_unit=nothing, speed=nothing, walltime_limit=nothing,
                          checkpoint=nothing, progress=nothing, resources=nothing,
                          black_holes=nothing, images=nothing, extra=nothing)
    t = Base.time()
    isfinite(sw.last_call) && (sw.gap = t - sw.last_call)
    sw.last_call = t
    if sw.time_start === nothing && time !== nothing
        sw.time_start = Float64(time)
    end
    force || t - sw.last_write ≥ sw.interval || return false
    sw.last_write = t
    expected = isnan(sw.gap) ? max(sw.interval, sw.startup_interval) :
               max(sw.interval, 3 * sw.gap / 2)
    limit = walltime_limit !== nothing ? walltime_limit :
            _sw_slurm_walltime_limit()
    prog = Dict{String,Any}()
    for (k, v) in (("iteration", iteration), ("time", time),
                   ("time_start", time === nothing ? nothing : sw.time_start),
                   ("time_end", time_end), ("time_unit", time_unit),
                   ("walltime", t - sw.t0), ("walltime_limit", limit),
                   ("speed", speed), ("checkpoint", checkpoint))
        _sw_setkey!(prog, k, v)
    end
    progress === nothing || merge!(prog, _sw_todict(progress))
    res = Dict{String,Any}()
    for (k, v) in (("nodes", _sw_envint("SLURM_JOB_NUM_NODES")),
                   ("tasks", _sw_envint("SLURM_NTASKS")),
                   ("threads", Threads.nthreads()), ("gpus", _sw_gpus()),
                   ("memory_bytes", _sw_current_rss()),
                   ("memory_peak_bytes", Sys.maxrss()))
        _sw_setkey!(res, k, v)
    end
    resources === nothing || merge!(res, _sw_todict(resources))
    slurm = Dict{String,Any}()
    for (k, v) in (("job_id", get(ENV, "SLURM_JOB_ID", nothing)),
                   ("job_name", get(ENV, "SLURM_JOB_NAME", nothing)),
                   ("partition", get(ENV, "SLURM_JOB_PARTITION", nothing)))
        _sw_setkey!(slurm, k, v)
    end
    doc = simwatch_document(; name=sw.name, status=status, message=message,
                            updated=Dates.now(Dates.UTC), started=sw.started,
                            update_interval=round(expected), code=sw.code,
                            host=gethostname(), pid=getpid(), progress=prog,
                            resources=res, slurm=isempty(slurm) ? nothing : slurm,
                            black_holes=black_holes, images=images, extra=extra)
    ok = write_simwatch(sw.dir, doc)
    if !ok && !sw.warned
        sw.warned = true
        @warn "SimWatch: could not write $(joinpath(sw.dir, "simwatch.toml")); " *
              "the run continues without a status file"
    end
    return ok
end

"""
    simwatch_finish!(sw::SimWatchWriter; status = "finished", kw...) -> Bool

Write the final status now: `"finished"`, `"stopped"` (at a wall-time limit,
to be continued) or `"failed"` with the error's first line as `message`.
"""
simwatch_finish!(sw::SimWatchWriter; status::AbstractString="finished", kw...) =
    simwatch_update!(sw; force=true, status=status, kw...)

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
    v = get(ENV, "CUDA_VISIBLE_DEVICES", "")
    return isempty(v) ? nothing : count(==(','), v) + 1
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
