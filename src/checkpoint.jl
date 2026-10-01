# Checkpoint and restart for `evolve!` (added 2026-10-01, on TreeAMR 0.1.4's
# M9a; Erik's decision of that day reversed `CODE.md`'s "no checkpoint and
# restart").
#
# Everything about the *file* is upstream: TreeAMR's `save_checkpoint` writes
# the forest, the evolved field set and an application's plain data, and
# writes them atomically, durably, with element types as limbs where they are
# not HDF5 natives and with the provenance of the writer; its
# `load_checkpoint` rebuilds a forest and a field set through their own
# validating constructors. Both live in the package extension
# `TreeAMRHDF5Ext`, which this package's `import HDF5` loads — HDF5 is a hard
# dependency here (decided 2026-10-01), where TreeHydro leaves it to the
# caller.
#
# The mechanism is TreeHydro's (`src/checkpoint.jl` there, 2026-09-29) on
# purpose, so that the applications of TreeAMR checkpoint alike — the same
# keywords, the same file names and rotation, the same refusal of a restart
# with other parameters — with **one difference: the checkpoint is written
# before the regrid** (decided 2026-10-01), so that a restart may regrid with
# a changed criterion as the first thing it does, before the next step. What
# is left for this file is what only the application knows:
#
#   * **its own run state** — the horizon track, the two fits, the finder's
#     seed, the target's ranges, the speed growth that sizes a moving hole's
#     step, the projection's accounting and the analysis record, without
#     which a restarted run would be a different run (`CODE.md`, "Checkpoint
#     and restart");
#   * **the recipe** — every parameter that decides a number, so that a
#     restart with a different one is refused by name — and, apart from it,
#     **the criterion**, the regridding parameters a restart may change;
#   * **the names** of the files, and their rotation.

const CHECKPOINT_APPLICATION = "TreeGeneralizedHarmonic.jl"
const CHECKPOINT_VERSION = 1

# --- names and rotation ------------------------------------------------------
#
# TreeHydro's, verbatim in behaviour: see its `src/checkpoint.jl`.

# The file for the checkpoint taken after `iteration` time steps since
# `t = 0`. The step count is monotonic across restarts, and it is what a
# reader of a directory listing wants to see — how far the run got — and the
# Cactus convention. Ten digits sort lexically up to 10¹⁰ steps; the pattern
# below reads any number of them.
checkpoint_filename(prefix, iteration::Integer) =
    "$prefix.it$(lpad(iteration, 10, '0')).h5"

# `s` quoted for a regular expression, so that a prefix with a `.` or a `+`
# in it matches itself and nothing else.
regex_quote(s::AbstractString) = replace(s, r"[\\^$.|?*+()\[\]{}]" => s"\\\0")

"""
    checkpoint_files(prefix) -> Vector{Tuple{Int,String}}

The checkpoint files of `prefix`, as `(iteration, path)` pairs sorted by
iteration: the files in `prefix`'s directory whose names are exactly
`"<basename>.it<digits>.h5"`. Nothing else matches — in particular not
TreeAMR's `"….h5.partial"`, the file a write in progress or a failed one
leaves, and not another prefix that merely starts with this one. A
directory that does not exist holds no checkpoints.
"""
function checkpoint_files(prefix::AbstractString)
    dir, base = splitdir(prefix)
    found = Tuple{Int,String}[]
    isempty(base) && return found
    listed = isempty(dir) ? "." : dir
    isdir(listed) || return found
    pattern = Regex("^" * regex_quote(base) * raw"\.it(\d+)\.h5$")
    for name in readdir(listed)
        m = match(pattern, name)
        m === nothing && continue
        iteration = tryparse(Int, m.captures[1])
        iteration === nothing && continue
        path = isempty(dir) ? name : joinpath(dir, name)
        isfile(path) && push!(found, (iteration, path))
    end
    return sort!(found)
end

"""
    latest_checkpoint(prefix)

The checkpoint file of `prefix` with the highest iteration, or `nothing` if
there is none — which is what makes a job chain one command for every job,
the first included:

```julia
r = evolve!(case; …, checkpoint_path_prefix = prefix,
            max_walltime_seconds = 23.5 * 3600,
            restart_file = latest_checkpoint(prefix))
r.finished || exit(3)          # resubmit
```

The first job finds nothing and starts from the initial data; every later
one continues from where the previous one stopped. See "Checkpoint and
restart" in `CODE.md`. The same function as TreeHydro's.
"""
function latest_checkpoint(prefix::AbstractString)
    files = checkpoint_files(prefix)
    return isempty(files) ? nothing : last(files)[2]
end

# Delete every checkpoint file of `prefix` but `keep` and the newest
# `num_keep − 1` others, and return the paths deleted. Run only after a write
# has succeeded. The file just written survives even when it is not the
# newest — a run restarted from an older checkpoint while newer ones exist —
# because it is the only one this run can vouch for; it is recognised by its
# *name*, since every file listed is in `prefix`'s directory and a path string
# is not a file's identity (TreeHydro found `run//sedov` listing as
# `run/sedov`, and a comparison of paths deleting the file just written).
function rotate_checkpoints!(prefix::AbstractString, num_keep::Integer;
                             keep::AbstractString)
    others = [path for (_, path) in checkpoint_files(prefix)
              if basename(path) != basename(keep)]
    removed = others[1:max(0, length(others) - (num_keep - 1))]
    foreach(path -> rm(path; force=true), removed)
    return removed
end

# --- exact reals as plain data -------------------------------------------------
#
# TreeAMR's `write_plain` refuses a MultiFloat scalar, and a run's `t`, its
# track and its fits are in `T`. A native float is stored as itself and any
# other `isbits` real made of one native float throughout (`Float32x2`) as the
# matrix of its limbs — TreeHydro's rule, which is TreeAMR's for a field set's
# element type, restated because that function is internal to the extension.

const NativeFloat = Union{Float16,Float32,Float64}

function limb_type(::Type{T}) where {T}
    T <: Union{NativeFloat,Base.BitInteger} && return T
    (isstructtype(T) && isconcretetype(T) && fieldcount(T) > 0) || return nothing
    F = nothing
    size = 0
    for i in 1:fieldcount(T)
        S = fieldtype(T, i)
        L = limb_type(S)
        (L === nothing || (F !== nothing && L !== F)) && return nothing
        F = L
        size += sizeof(S)
    end
    return size == sizeof(T) ? F : nothing
end

function limbs_of(::Type{R}) where {R}
    F = isbitstype(R) ? limb_type(R) : nothing
    F === nothing && throw(ArgumentError(
        "a checkpoint cannot store the reals of $R exactly: it is neither a " *
        "native float nor an isbits type made of one native type throughout " *
        "with no padding, which is stored as its limbs (Float32x2 as two " *
        "Float32). A file stores bits, and no other type could be read back " *
        "bit for bit."))
    return (F, sizeof(R) ÷ sizeof(F))
end

"""
    plain_reals(xs)

Values of the run's real type as plain data that read back **bit for bit**:
a vector of a native float as itself, and a vector of any other `isbits` real
made of one native float throughout (MultiFloats' `Float32x2`) as the matrix
of its limbs, `(nlimbs, n)`. A scalar is stored as a one-element vector and a
tuple as a vector. [`from_plain_reals`](@ref) is the inverse. TreeHydro's.
"""
plain_reals(x::Real) = plain_reals([x])
plain_reals(xs::Tuple) = plain_reals(collect(xs))
function plain_reals(xs::AbstractVector{R}) where {R}
    R <: NativeFloat && return collect(xs)
    F, n = limbs_of(R)
    return collect(reshape(reinterpret(F, collect(xs)), n, length(xs)))
end

"""
    from_plain_reals(R, a) -> Vector{R}

The inverse of [`plain_reals`](@ref): the vector of `R` that `a` stores,
refused if `a` is not what `plain_reals` makes of an `R`.
"""
function from_plain_reals(::Type{R}, a) where {R}
    if R <: NativeFloat
        a isa AbstractVector{R} || throw(ArgumentError(
            "a checkpoint value is a $(typeof(a)) where a vector of $R was " *
            "expected: the file was written by a run in another type, or damaged."))
        return collect(a)
    end
    F, n = limbs_of(R)
    (a isa AbstractMatrix{F} && size(a, 1) == n) || throw(ArgumentError(
        "a checkpoint value is a $(typeof(a)) of size $(size(a)) where the " *
        "$n $F limbs of a vector of $R were expected: the file was written by " *
        "a run in another type, or damaged."))
    return collect(reinterpret(R, vec(a)))
end

from_plain_scalar(::Type{R}, a) where {R} = only(from_plain_reals(R, a))

# A type's name as a module importing nothing but Base prints it, and not
# `string(T)`, which qualifies a name or not according to what the writer had
# imported into `Main` (TreeHydro's, and TreeAMR's extension's).
module TypeNames end
type_name(::Type{T}) where {T} = sprint(show, T; context=:module => TypeNames)

# --- the run state's own structs -------------------------------------------------
#
# The run carries this package's structs — the `HorizonTrack`, the
# `InteriorFit`s with their `FitParams`, the `StateBounds`, the
# `BoundsAccounting` — and `write_plain` refuses a struct. They are stored
# field by field and rebuilt **from their declared field types**, which is
# what brings an `SVector{3,T}` or a `T` back as itself rather than as the
# `Vector{Float64}` a plain read gives: `to_plain(x)` is the type's name, its
# field names and its field values, and `from_plain(S, p)` refuses a `p`
# whose type or field names are not `S`'s. Only this package's own structs
# are taken apart — anything else is refused with the path of the field that
# holds it, so that a struct added to the run state later is a refusal at the
# first save of a test, not a silent loss.

"""
    to_plain(x; path = "run") -> plain data

`x`, one of this package's run-state structs or a value one of them holds,
as plain data `write_plain` accepts, from which [`from_plain`](@ref) rebuilds
it bit for bit. `path` names the value in a refusal.
"""
to_plain(x::AbstractFloat; path="run") = plain_reals(x)
to_plain(x::Union{Bool,Base.BitInteger,Symbol,String,Nothing}; path="run") = x
to_plain(x::SVector{N,R}; path="run") where {N,R<:AbstractFloat} = plain_reals(collect(x))
to_plain(x::Vector{R}; path="run") where {R<:Union{NativeFloat,Int,ComplexF64}} = copy(x)
function to_plain(x::Vector{SVector{N,R}}; path="run") where {N,R<:AbstractFloat}
    return (; n=length(x), values=plain_reals(R[v[i] for v in x for i in 1:N]))
end
function to_plain(x::Array{R,3}; path="run") where {R<:AbstractFloat}
    return (; size=collect(Int, size(x)), values=plain_reals(vec(x)))
end
# The finder's grid, carried by a coasting track and read by nobody: the one
# kind `find_horizon` makes.
to_plain(x::EquiangularGrid; path="run") = (; kind="EquiangularGrid", lmax=x.lmax)
function to_plain(x::T; path="run") where {T}
    (isstructtype(T) && parentmodule(T) === @__MODULE__) || throw(ArgumentError(
        "the run state at $path is a $T, which a checkpoint cannot store: " *
        "TreeAMR's plain data hold numbers, strings, symbols, tuples and named " *
        "tuples, and this package takes apart only its own structs. Extend " *
        "`to_plain` and `from_plain` for it (src/checkpoint.jl)."))
    names = fieldnames(T)
    values = ntuple(i -> to_plain(getfield(x, i); path="$path.$(names[i])"),
                    fieldcount(T))
    return (; kind=String(nameof(T)), names=collect(String.(names)), values=values)
end

"""
    from_plain(S, p; path = "run") -> S

The inverse of [`to_plain`](@ref) for the type `S` the run state declares,
refused with the field's path where `p` is not what `to_plain` makes of an
`S`.
"""
function from_plain(::Type{F}, v; path="run") where {F}
    # `Union{Nothing,X}`: a field that holds a value or nothing.
    if F isa Union && Nothing <: F
        v === nothing && return nothing
        return from_plain(Base.nonnothingtype(F), v; path=path)
    end
    F === Nothing && (v === nothing ? (return nothing) : _plain_refusal(F, v, path))
    F <: AbstractFloat && return from_plain_scalar(F, v)
    F === Bool && return v isa Bool ? v : _plain_refusal(F, v, path)
    F <: Base.BitInteger && return v isa Integer ? F(v) : _plain_refusal(F, v, path)
    (F === Symbol || F === String) && return v isa F ? v : _plain_refusal(F, v, path)
    if F <: SVector
        N, R = length(F), eltype(F)
        x = from_plain_reals(R, v)
        length(x) == N || _plain_refusal(F, v, path)
        return SVector{N,R}(x)
    end
    if F <: Vector && eltype(F) <: SVector
        N, R = length(eltype(F)), eltype(eltype(F))
        x = from_plain_reals(R, v.values)
        length(x) == N * v.n || _plain_refusal(F, v, path)
        return [SVector{N,R}(ntuple(i -> x[(k - 1) * N + i], Val(N))) for k in 1:v.n]
    end
    if F <: Vector && eltype(F) <: Union{NativeFloat,Int,ComplexF64}
        v isa AbstractVector{eltype(F)} || _plain_refusal(F, v, path)
        return collect(eltype(F), v)
    end
    if F <: Array{<:AbstractFloat,3}
        x = from_plain_reals(eltype(F), v.values)
        return reshape(x, Tuple(v.size))
    end
    if F <: SphereGrid
        (v isa NamedTuple && get(v, :kind, "") == "EquiangularGrid") ||
            _plain_refusal(F, v, path)
        return EquiangularGrid(v.lmax)
    end
    (isstructtype(F) && isconcretetype(F) && parentmodule(F) === @__MODULE__) ||
        _plain_refusal(F, v, path)
    (v isa NamedTuple && get(v, :kind, "") == String(nameof(F)) &&
     v.names == collect(String.(fieldnames(F)))) || throw(ArgumentError(
        "the checkpoint's run state at $path is not a $(nameof(F)) with the " *
        "fields $(collect(fieldnames(F))): the file was written by another " *
        "version of this package, or is damaged."))
    args = ntuple(i -> from_plain(fieldtype(F, i), v.values[i];
                                  path="$path.$(fieldname(F, i))"),
                  fieldcount(F))
    return F(args...)
end

_plain_refusal(F, v, path) = throw(ArgumentError(
    "the checkpoint's run state at $path is a $(typeof(v)) where a $F was " *
    "expected: the file was written by a run in another type or by another " *
    "version of this package, or is damaged."))

# An `InteriorFit` (step 8e) holds its coefficients twice — on the backend,
# which the cache fill reads, and on the host — and three diagnostic named
# tuples. The host copy is stored and the backend's is made from it as
# `build_fit` makes it; the diagnostics are stored with every `SVector` as a
# tuple and every real as a `Float64` (`_plain_loose`), since nothing the run
# computes reads them back.
function to_plain(f::InteriorFit; path="run")
    return (; kind="InteriorFit", params=to_plain(f.params; path="$path.params"),
            host=to_plain(f.host; path="$path.host"), t=to_plain(f.t),
            c=to_plain(f.c), points=to_plain(f.points), model=to_plain(f.model),
            residual=_plain_loose(f.residual), conditioning=_plain_loose(f.conditioning),
            valid=f.valid, sweep=_plain_loose(f.sweep))
end

function fit_from_plain(::Type{T}, v; backend=CPU(), path="run") where {T}
    v === nothing && return nothing
    (v isa NamedTuple && get(v, :kind, "") == "InteriorFit") || throw(ArgumentError(
        "the checkpoint's run state at $path is not an InteriorFit: the file " *
        "was written by another version of this package, or is damaged."))
    params = from_plain(FitParams{T}, v.params; path="$path.params")
    host = from_plain(Array{T,3}, v.host; path="$path.host")
    coeffs = to_backend(backend, host)
    return InteriorFit{T,typeof(coeffs),typeof(host)}(
        params, coeffs, host, from_plain(T, v.t), from_plain(SVector{3,T}, v.c),
        from_plain(Vector{SVector{3,T}}, v.points),
        v.model.n == 0 ? SVector{NFIT,T}[] :
        from_plain(Vector{SVector{NFIT,T}}, v.model; path="$path.model"),
        v.residual, v.conditioning, v.valid, v.sweep)
end

_plain_loose(x::NamedTuple) = map(_plain_loose, x)
_plain_loose(x::Tuple) = map(_plain_loose, x)
_plain_loose(x::SVector) = Tuple(map(_plain_loose, x))
_plain_loose(x::AbstractVector{<:Number}) = map(_plain_loose, collect(x))
_plain_loose(x::NativeFloat) = x
_plain_loose(x::AbstractFloat) = tofloat64(x)
_plain_loose(x::Union{Bool,Base.BitInteger,Symbol,String,Nothing}) = x
_plain_loose(x) = repr(x)

# --- the recipe and the criterion --------------------------------------------------

"""
    run_recipe(T, case; q, ops, chunk, cfl, adapt, adm_every, ρ_max_factor,
               ρ_max_fixed, fit_initial_cont, fit_initial_depth, handover,
               target_source, target_rate, fit_initial_blend, trail_ramp,
               target_exact, refill_cells)

Every parameter of an [`evolve!`](@ref) call that decides a number, as plain
data, **except the regridding criterion** ([`run_criterion`](@ref)), which a
restart may change: the working type by name, the scheme's order and the
operators, the case — its background, box, dissipation, damping, center,
interior, horizon finder and bounds, each as its `repr`, which prints every
real in full — and the keywords that shape the run. `t_end` is not in it,
because a restart may move it; nor are `backend`, `maxpasses`, `find` or the
observer, which decide no number of the run once its initial data exist. The
forest is not in it either: a restart takes the forest from the file.

Every real keyword goes through `T` first and then [`plain_reals`](@ref), so
a `1//4` given to one call and a `T(1//4)` to the next compare equal, as they
are the same run. TreeHydro's `run_recipe`, for this package's parameters.
"""
function run_recipe(::Type{T}, case::GHCase; q, ops, chunk, cfl, adapt, adm_every,
                    ρ_max_factor, ρ_max_fixed, fit_initial_cont, fit_initial_depth,
                    handover, target_source, target_rate, fit_initial_blend,
                    trail_ramp, target_exact, refill_cells) where {T}
    r(x) = x === nothing ? nothing : plain_reals(T(x))
    return (; float_type=type_name(T), q=Int(q),
            ops=(; family=Symbol(ops.family), prolongation=Int(ops.prolongation),
                 restriction=Int(ops.restriction)),
            background=repr(case.background),
            box=plain_reals([x for ext in case.box for x in ext]),
            periodic=case.periodic, epsilon_KO=repr(case.ε_KO), gamma0=repr(case.γ0),
            gamma2=r(case.γ2), center=repr(case.center), interior=repr(case.interior),
            horizon=repr(case.horizon), bounds=repr(case.bounds),
            chunk=r(chunk), cfl=r(cfl), adapt=Bool(adapt), adm_every=Int(adm_every),
            rho_max_factor=r(ρ_max_factor), rho_max_fixed=r(ρ_max_fixed),
            fit_initial_cont=Int(fit_initial_cont),
            fit_initial_depth=r(fit_initial_depth), handover=r(handover),
            target_source=Symbol(target_source), target_rate=Bool(target_rate),
            fit_initial_blend=Bool(fit_initial_blend), trail_ramp=r(trail_ramp),
            target_exact=Bool(target_exact), refill_cells=r(refill_cells))
end

"""
    run_criterion(T, case; regrid, buffer)

The parameters a restart **may** change (decided 2026-10-01): the
regridding criterion — every field of the case's `Refinement`, whether the
run regrids at all, and the travelling margin `buffer` — as plain data. The
checkpoint is written before the regrid, so a restart with a changed
criterion regrids with it first thing, before its next step; it is reported
field by field and returned as `criterion_changed`.
"""
function run_criterion(::Type{T}, case::GHCase; regrid, buffer) where {T}
    ref = case.refinement
    fields = ref === nothing ? (;) :
             NamedTuple{(:refine_tol, :coarsen_tol, :maxlevel_cap, :floor_margin,
                         :ceiling_cells, :ceiling_level, :epsilon)}(
                 (plain_reals(T(ref.refine_tol)), plain_reals(T(ref.coarsen_tol)),
                  Int(ref.maxlevel_cap), plain_reals(T(ref.floor_margin)),
                  Int(ref.ceiling_cells), Int(ref.ceiling_level),
                  plain_reals(T(ref.ε))))
    return (; refinement=ref !== nothing, fields..., regrid=Bool(regrid),
            buffer=buffer === nothing ? nothing : Int(buffer))
end

# A plain value as a message prints it: a one-element vector — how a scalar
# real is stored — as its element.
describe_plain(x::AbstractVector) = length(x) == 1 ? repr(only(x)) : repr(x)
describe_plain(x) = repr(x)

# The fields of two plain named tuples that differ, each as a sentence.
function plain_differences(saved, current)
    diffs = Tuple{Symbol,String}[]
    for k in unique((keys(saved)..., keys(current)...))
        a = haskey(saved, k) ? saved[k] : missing
        b = haskey(current, k) ? current[k] : missing
        isequal(a, b) && continue
        was = a === missing ? "absent" : describe_plain(a)
        is = b === missing ? "absent" : describe_plain(b)
        push!(diffs, (k, "`$k` is $was in the checkpoint and $is in this call"))
    end
    return diffs
end

"""
    check_recipe(saved, current, path)

Refuse a restart whose parameters differ from the checkpoint's, with one
`ArgumentError` that names **every** field that differs and both of its
values — so that a job script with two wrong keywords is fixed in one round
and not two. Equality is `isequal` on the plain forms, which for reals is
equality of the bits in the run's type. TreeHydro's.
"""
function check_recipe(saved, current, path)
    diffs = plain_differences(saved, current)
    isempty(diffs) || throw(ArgumentError(
        "restart_file $(repr(path)) was written by a run with other parameters: " *
        join(last.(diffs), "; ") * ". A restart continues the saved run, and a " *
        "run continued with another parameter would be a different experiment " *
        "that looks like the old one, so it must be called with the same case " *
        "and the same keywords — only t_end and the regridding criterion (the " *
        "case's Refinement, `regrid` and `buffer`) may change, and backend, " *
        "maxpasses, find and the observer, which decide no number of the run."))
    return nothing
end

# --- writing and reading -------------------------------------------------------------

"""
    save_run(path, forest, U, u; recipe, criterion, run, filters = (), sync = true)

One checkpoint: the forest, the state field set `U` with its state vector
`u`, and this package's plain data `(; recipe, criterion, run)`, through
TreeAMR's `save_checkpoint` — atomically, so a failed write leaves the
previous file alone. `u` and not `U.work`: the checkpoint is taken after the
analysis row, whose horizon find, fit and indicator have scattered into
`U.work` and filled its ghosts, and after a chunk without a regrid the
working array holds the integrator's last stage, not the state.
"""
function save_run(path, forest, U, u; recipe, criterion, run, filters=(),
                  sync::Bool=true)
    return save_checkpoint(path, forest; fieldsets=("state" => (U, u),),
                           application=CHECKPOINT_APPLICATION => CHECKPOINT_VERSION,
                           data=(; recipe=recipe, criterion=criterion, run=run),
                           filters=filters, sync=sync)
end

"""
    load_run(path, T; backend = CPU()) -> (; forest, U, u, recipe, criterion, run)

Read a checkpoint written by [`save_run`](@ref), refusing one that is not
this package's — another application's, or a format version other than
$(CHECKPOINT_VERSION) — with the reason. The field set comes back in the type
it was saved in; whether that is `T` is the recipe's to say, so that the
refusal names it with the rest.
"""
function load_run(path::AbstractString, ::Type{T}; backend=CPU()) where {T}
    ck = load_checkpoint(path; backend=backend, types=(T,))
    name, version = ck.application
    written = "It was written by TreeAMR " *
              "$(something(ck.provenance.treeamr_version, "(unknown version)")) " *
              "on $(ck.provenance.created)."
    name == CHECKPOINT_APPLICATION || throw(ArgumentError(
        "$(repr(path)) is a checkpoint of the application $(repr(name)), not of " *
        "$(CHECKPOINT_APPLICATION): its run state is that application's, and this " *
        "package cannot continue a run it did not write. $written"))
    version == CHECKPOINT_VERSION || throw(ArgumentError(
        "$(repr(path)) stores TreeGeneralizedHarmonic's run state in format " *
        "version $version, and this version of the package reads version " *
        "$(CHECKPOINT_VERSION) only. A file from a newer version is read by that " *
        "version — TreeAMR's `checkpoint_environment(path, dir)` writes the " *
        "environment that wrote it. $written"))
    (ck.data isa NamedTuple && haskey(ck.data, :recipe) && haskey(ck.data, :run) &&
     haskey(ck.data, :criterion) && haskey(ck.fieldsets, "state")) ||
        throw(ArgumentError(
            "$(repr(path)) names $(CHECKPOINT_APPLICATION) version $version but " *
            "holds no recipe, no criterion, no run state or no field set " *
            "\"state\": the file is damaged, or was not written by `evolve!`. " *
            "$written"))
    U = ck.fieldsets["state"].fieldset
    u = ck.fieldsets["state"].state
    return (; forest=ck.forest, U=U, u=u, recipe=ck.data.recipe,
            criterion=ck.data.criterion, run=ck.data.run)
end

# --- the keywords ------------------------------------------------------------------------

# The checkpoint keywords of `evolve!`, refused up front — before the
# initial-data cycle — so that a job script's mistake costs a second and not
# the queue wait and the hours before the first write. TreeHydro's, less its
# check that HDF5 is loaded: here it always is.
function check_checkpoint_keywords(; checkpoint_path_prefix, checkpoint_every_chunks,
                                   checkpoint_interval_seconds, max_walltime_seconds,
                                   num_checkpoints_keep, restart_file)
    prefix = checkpoint_path_prefix
    triggers = (checkpoint_every_chunks, checkpoint_interval_seconds,
                max_walltime_seconds)
    if prefix === nothing
        all(isnothing, triggers) || throw(ArgumentError(
            "checkpoint_every_chunks, checkpoint_interval_seconds or " *
            "max_walltime_seconds was given without checkpoint_path_prefix: the " *
            "run would be asked to write a checkpoint, or to stop and leave one, " *
            "with nowhere to write it. Pass checkpoint_path_prefix, such as " *
            "\"run/g5\" for files run/g5.it0000001234.h5."))
    else
        prefix isa AbstractString || throw(ArgumentError(
            "checkpoint_path_prefix must be a string, got $(repr(prefix))."))
        dir, base = splitdir(prefix)
        isempty(base) && throw(ArgumentError(
            "checkpoint_path_prefix $(repr(prefix)) ends in a directory separator: " *
            "it is the start of each file's name, not a directory, so it needs a " *
            "stem — \"run/g5\" writes run/g5.it0000001234.h5."))
        isempty(dir) || isdir(dir) || throw(ArgumentError(
            "the directory of checkpoint_path_prefix, $(repr(dir)), does not " *
            "exist: the first checkpoint would fail to be written hours into the " *
            "run. Create it first."))
        any(!isnothing, triggers) || throw(ArgumentError(
            "checkpoint_path_prefix was given but no checkpoint_every_chunks, " *
            "checkpoint_interval_seconds or max_walltime_seconds: nothing would " *
            "ever write a checkpoint. Pass at least one of them."))
    end
    checkpoint_every_chunks === nothing ||
        (checkpoint_every_chunks isa Integer && checkpoint_every_chunks ≥ 1) ||
        throw(ArgumentError(
            "checkpoint_every_chunks must be an integer of at least 1, got " *
            "$(repr(checkpoint_every_chunks)): it is how many chunks lie between " *
            "two checkpoints."))
    checkpoint_interval_seconds === nothing ||
        (checkpoint_interval_seconds isa Real && checkpoint_interval_seconds ≥ 0) ||
        throw(ArgumentError(
            "checkpoint_interval_seconds must be a non-negative number, got " *
            "$(repr(checkpoint_interval_seconds)): it is the wall-clock time " *
            "between two checkpoints, and 0 writes one at every chunk boundary."))
    max_walltime_seconds === nothing ||
        (max_walltime_seconds isa Real && max_walltime_seconds > 0) ||
        throw(ArgumentError(
            "max_walltime_seconds must be a positive number, got " *
            "$(repr(max_walltime_seconds)): it is the job's wall-time limit, " *
            "from the call to evolve!, before which the run writes a checkpoint " *
            "and stops."))
    (num_checkpoints_keep isa Integer && num_checkpoints_keep ≥ 1) ||
        throw(ArgumentError(
            "num_checkpoints_keep must be an integer of at least 1, got " *
            "$(repr(num_checkpoints_keep)): the newest checkpoint is the one a " *
            "restart needs, so it is always kept."))
    if restart_file !== nothing
        restart_file isa AbstractString || throw(ArgumentError(
            "restart_file must be a path or nothing, got $(repr(restart_file))."))
        isfile(restart_file) || throw(ArgumentError(
            "restart_file $(repr(restart_file)) does not exist. To start from the " *
            "initial data when there is no checkpoint yet, pass " *
            "`restart_file = latest_checkpoint(prefix)`, which is nothing then."))
    end
    return nothing
end
