# Runs of a pair of holes (added in step B3): the head-on collision from rest of
# `PLAN.md`'s step B3 — two Kerr-Schild holes of mass `M_A = 1/2` at separation
# `d`, on the full box `[−L, L]³` with Dirichlet data frozen at `t = 0`, each
# hole's interior step B2's tracked `:damped` target, and the composite tracked
# Kerr-Schild gauge source (pairing (α), decided by Erik 2026-10-26;
# `CODE.md`, "Binary black holes").
#
# A standalone script, not part of the suite: one run per call, `key=value`
# options. The mesh is `holes_forest`'s nested balls about each hole's initial
# center, `h = 2L/(roots N)` at the outer boundary and halved inside each ball of
# `radii`. Each chunk writes one CSV row: the constraints over the evolved region
# (`C_a` every chunk, `ℋ` with `adm=1`), point-weighted and in bands about the
# nearer hole's track, the layers' residual, and per hole the track and the
# horizon; then the pair's separation from the tracks and from the found
# origins against Newtonian free fall from rest, `d̈ = −M/d²`; and the cost —
# the wall time, the points and, on a device, its memory.
#
#     julia --project=. --threads=4 test/binary_runs.jl d=8 L=8 N=8 roots=4 radii=2,3/2,5/4 q=2 r_0=3/5 r_1=7/5 margin=4 t_end=1 chunk=1/10 out=out/pair
#
#     julia --project=<env with CUDA> test/binary_runs.jl backend=cuda d=8 L=32 N=24 roots=8 radii=16,8,4,2 t_end=2 chunk=1/4 out=out/b3h48
#
# Options (`key=value`; the defaults are step B3's head-on row at `1/48`):
#
#   d=8 L=32 N=24 roots=8          the separation, the box [−L, L]³, the blocks
#   radii=16,8,4,2                 the balls about each hole, one per level (in M)
#   t_end=2 chunk=1/4 cfl=1/2 q=4  the run
#   eps=1/2 gamma2=0               ε_KO and γ2 (γ0: holes_damping's three Gaussians)
#   r_0=5/8 r_1=23/16 margin=8     each hole's layer, stated for a hole of mass one
#                                  (each hole takes it times M_A): step B2's
#                                  production :damped sphere
#   find_interval=0                the longest interval between two finds for a
#                                  hole of mass one, in M (0: the chunk)
#   target=superposed|own          each layer relaxes toward the superposition of
#                                  the holes on their tracks, or toward its own hole
#   feed=tracks|seed|controlled    the gauge source's holes follow the tracks
#                                  (pairing (α)), their seeds' Newtonian quadratics,
#                                  or (added in step B3) the controlled source lines
#   control_time=10                feed=controlled only: the loop's time for a hole
#                                  of mass one, in M (each hole takes it times M_A)
#   momentum=superposition|gauge   the initial Π: the data's own, or from C_a = 0
#   companion=4 attenuation=       the data's companion attenuation λ_B/M_B (0:
#                                  none) and XCTS's own one (a length; none)
#   adm=1                          the ADM monitor every this many chunks (0: never)
#   finder_N=16 spin=1             the finder's resolution, and the Korzyński J
#   backend=cpu|cuda mpi=false     as test/octant_runs.jl's
#   out=<dir>                      where `binary.csv`, `records.txt` and the log go
#   checkpoint=<dir> walltime=<s>  a chain of jobs; checkpoint_every=4 keep=2
#
# A run that stops at `walltime` exits with status 3, so a job script can resubmit;
# the same command continues from the newest checkpoint and appends to the CSV.

using Printf
using TreeAMR
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: DIAG_CGH, DIAG_HAM, DIAG_MOM, DIAG_MASK
using StaticArrays: SVector

const OPTIONS = Dict{String,String}()
for arg in ARGS
    k, v = occursin('=', arg) ? split(arg, '='; limit=2) : (arg, "")
    OPTIONS[String(k)] = String(v)
end
opt(k, default) = get(OPTIONS, k, default)
rat(s) = (p = split(s, '/'); length(p) == 2 ? parse(Int, p[1]) // parse(Int, p[2]) :
                            occursin(r"[.eE]", s) ? parse(Float64, s) : parse(Int, s))

const T = Float64
d_sep = rat(opt("d", "8"))
L = rat(opt("L", "32"))
N = parse(Int, opt("N", "24"))
roots = parse(Int, opt("roots", "8"))
radii = [T(rat(r)) for r in split(opt("radii", "16,8,4,2"), ',')]
t_end = rat(opt("t_end", "2"))
chunk = rat(opt("chunk", "1/4"))
cfl = rat(opt("cfl", "1/2"))
q = parse(Int, opt("q", "4"))
ε_KO = rat(opt("eps", "1/2"))
γ2 = rat(opt("gamma2", "0"))
adm_every = parse(Int, opt("adm", "1"))
outdir = mkpath(abspath(opt("out", "out/binary")))
ckdir = haskey(OPTIONS, "checkpoint") ? mkpath(abspath(OPTIONS["checkpoint"])) : nothing
walltime = haskey(OPTIONS, "walltime") ? parse(Float64, OPTIONS["walltime"]) : nothing
ckevery = parse(Int, opt("checkpoint_every", "4"))
ckkeep = parse(Int, opt("keep", "2"))

use_mpi = opt("mpi", "false") in ("true", "1")
if use_mpi
    @eval using MPI
    MPI.Init()
end
isrank0 = !use_mpi || MPI.Comm_rank(MPI.COMM_WORLD) == 0
function abort_all(err)
    use_mpi || return nothing
    showerror(stderr, err, catch_backtrace())
    println(stderr)
    MPI.Abort(MPI.COMM_WORLD, 1)
end
ondevice = opt("backend", "cpu") == "cuda"
backend = if ondevice
    @eval using CUDA
    if use_mpi
        node = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED,
                                   MPI.Comm_rank(MPI.COMM_WORLD))
        Base.invokelatest(() -> CUDA.device!(MPI.Comm_rank(node) % length(CUDA.devices())))
        MPI.free(node)
    end
    Base.invokelatest(() -> CUDA.CUDABackend())
else
    TreeGeneralizedHarmonic.CPU()
end
comm = use_mpi ? communicator(MPI.COMM_WORLD;
                              deviceaware=opt("deviceaware", "false") in ("true", "1")) :
       nothing
# The device's memory in use, in GB, or `NaN` on the host.
function device_gb()
    ondevice || return NaN
    # `CUDA.memory_info()` is `(free, total)` in bytes; the job log's `nvidia-smi`
    # lines are the fallback where this CUDA.jl has another name for it.
    try
        free, total = Base.invokelatest(() -> CUDA.memory_info())
        return (total - free) / 1e9
    catch
        return NaN
    end
end

# The pair: each hole's layer stated for a hole of mass one and scaled by its own.
unit = TrackedSpec(T; r_0=T(rat(opt("r_0", "5/8"))), r_1=T(rat(opt("r_1", "23/16"))),
                   margin=parse(Int, opt("margin", "8")),
                   max_interval=T(rat(opt("find_interval", "0"))),
                   control_time=opt("feed", "tracks") != "controlled" ? zero(T) :
                                lowercase(opt("control_time", "10")) == "inf" ? T(Inf) :
                                T(rat(opt("control_time", "10"))))
holes = head_on_holes(T; d=d_sep, interior=unit)
companion = rat(opt("companion", "4"))
attenuation = haskey(OPTIONS, "attenuation") ? T(rat(OPTIONS["attenuation"])) : nothing
horizon = Horizon(T; every=1, N=parse(Int, opt("finder_N", "16")),
                  spin=opt("spin", "1") in ("1", "true"))
case = binary_case(T; holes=holes, halfwidth=L, chunk=chunk, ε_KO=ε_KO, γ2=γ2,
                   horizon=horizon, companion=companion, attenuation=attenuation,
                   hole_target=Symbol(opt("target", "superposed")),
                   source_feed=Symbol(opt("feed", "tracks")),
                   momentum=Symbol(opt("momentum", "superposition")))
M = sum(h -> h.mass, holes)

# The mesh, or the checkpoint's.
ckw = if ckdir === nothing
    (;)
else
    pre = joinpath(ckdir, "binary")
    rf = latest_checkpoint(pre)
    rf === nothing || !isrank0 || println("restarting from $rf")
    (checkpoint_path_prefix=pre, checkpoint_every_chunks=ckevery, restart_file=rf,
     max_walltime_seconds=walltime, num_checkpoints_keep=ckkeep)
end
restarting = get(ckw, :restart_file, nothing) !== nothing
forest0 = restarting ? nothing :
          holes_forest(T, case; N=N, roots=roots, radii=radii, comm=comm)
if forest0 !== nothing && isrank0
    lv = forest_levels(forest0)
    npts = nleaves(forest0) * N^3
    @printf("mesh: %d blocks, levels %s, h = %.6g at the holes, %.4g M owned points (%.3g GB of state)\n",
            nleaves(forest0), string(lv), minimum_spacing(T, forest0), npts / 1e6,
            npts * 20 * 8 / 1e9)
end

# --- the row -------------------------------------------------------------------------
#
# One fold over `diag`'s `C_a`, `ℋ`, `ℳ_i` and the mask that the row's monitors
# left, binned by the nearer hole's track center and the distance from it:
# `bands` edges in M.
const BANDS = (1.0, 1.5, 2.0, 3.0, 4.0, Inf)
const NBAND = length(BANDS) - 1
const NV = DIAG_MASK - DIAG_CGH + 1
const VM = NV                       # the mask is the last of the slots
const NQ = NV - 1
@inline slotq(u, i) = u[i]
zbin(::Type{T}) where {T} = (n=Sum(zero(T)), s2=ntuple(_ -> Sum(zero(T)), Val(NQ)),
                             mx=ntuple(_ -> Max(zero(T)), Val(NQ)))
@inline qbin(u) = (n=Sum(u[VM]), s2=ntuple(i -> Sum(u[i] * u[i]), Val(NQ)),
                   mx=ntuple(i -> Max(abs(u[i])), Val(NQ)))
struct BandFold{T}
    c1::SVector{3,T}
    c2::SVector{3,T}
end
@inline function (f::BandFold{T})(acc, p) where {T}
    q = qbin(p.u)
    x = SVector{3,T}(p.x[1], p.x[2], p.x[3])
    r1 = sqrt(sum(abs2, x - f.c1))
    r2 = sqrt(sum(abs2, x - f.c2))
    A = r1 ≤ r2 ? 0 : 1
    r = min(r1, r2)
    k = sum(ntuple(i -> (BANDS[i] ≤ r) & (r < BANDS[i + 1]) ? i : 0, Val(NBAND)); init=0)
    return (all=TreeAMR.combine(acc.all, q),
            band=k == 0 ? acc.band : addto(acc.band, A * NBAND + k, q))
end
function bandnorms(b)
    n = b.n.value
    l2(i) = n > 0 ? sqrt(b.s2[i].value / n) : NaN
    return (points=n, gauge_l2=maximum(l2(a) for a in 1:4),
            gauge_linf=maximum(b.mx[a].value for a in 1:4), ham_l2=l2(5),
            ham_linf=b.mx[5].value)
end

# Newtonian free fall from rest at `d₀`: `d̈ = −M/d²`, RK4 at `M/1000`.
function newton_separation(d0, M, t)
    y = (T(d0), zero(T))
    f((d, v)) = (v, -M / d^2)
    n = max(1, ceil(Int, t * 1000))
    dt = t / n
    for _ in 1:n
        k1 = f(y)
        k2 = f(y .+ (dt / 2) .* k1)
        k3 = f(y .+ (dt / 2) .* k2)
        k4 = f(y .+ dt .* k3)
        y = y .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return y[1]
end

csv = joinpath(outdir, "binary.csv")
hcols = ["succ", "cx", "cy", "cz", "vx", "ox", "rmin", "rmax", "area", "Mirr", "J", "Mch",
         "pred", "finds", "offx", "srcx", "srcvx", "srcax"]
bcols = [("h$(A)_" * b * "_" * f) for A in 1:2 for b in ("r$(BANDS[k])" for k in 1:NBAND)
         for f in ("ham_l2", "gauge_l2")]
header = vcat(["t", "wall", "steps", "dt", "nblocks", "points", "gauge_l2", "gauge_linf",
               "ham_l2", "ham_linf", "residual", "finite"],
              ["h$(A)_$c" for A in 1:2 for c in hcols],
              ["sep_track", "sep_found", "sep_newton", "gpu_gb", "maxrss_gb"], bcols)
if isrank0 && !(restarting && isfile(csv))
    open(io -> println(io, join(header, ',')), csv, "w")
end
t_wall0 = time()
num(x) = x === nothing ? NaN : x
function observe(p, t, u, rec)
    c1 = SVector{3,T}(rec.holes[1].track_center)
    c2 = SVector{3,T}(rec.holes[2].track_center)
    b0 = zbin(T)
    init = (all=b0, band=ntuple(_ -> b0, Val(2 * NBAND)))
    rf = mesh_fold(BandFold{T}(c1, c2), TreeAMR.combine, init, p.diag;
                   vars=DIAG_CGH:DIAG_MASK)
    all = bandnorms(rf.all)
    isrank0 || return nothing
    o = [r.origin === nothing ? SVector(NaN, NaN, NaN) : SVector{3,Float64}(r.origin)
         for r in rec.holes]
    sep_found = sqrt(sum(abs2, o[2] - o[1]))
    row = Any[rec.t, time() - t_wall0, rec.steps, rec.dt, rec.nblocks,
              rec.nblocks * N^3, rec.gauge_l2, rec.gauge_linf, num(rec.ham_l2),
              num(rec.ham_linf), rec.residual, rec.finite]
    for h in rec.holes
        append!(row, (h.success === true ? 1 : 0, h.track_center..., h.track_velocity[1],
                      h.origin === nothing ? NaN : h.origin[1], num(h.track_r_min),
                      num(h.track_r_max), num(h.area), num(h.M_irr), num(h.J), num(h.M_ch),
                      num(h.track_prediction_max), h.track_finds, h.horizon_offset[1],
                      h.source_center[1], h.source_velocity[1],
                      h.source_acceleration[1]))
    end
    append!(row, (rec.separation, sep_found, newton_separation(d_sep, M, rec.t),
                  device_gb(), Sys.maxrss() / 1e9))
    for k in 1:(2 * NBAND)
        b = bandnorms(rf.band[k])
        append!(row, (b.ham_l2, b.gauge_l2))
    end
    open(io -> println(io, join(row, ',')), csv, "a")
    @printf("t = %7.4f  wall %8.1f s  C_a %.3e/%.3e  ℋ %.3e  res %.3e  sep %.6f (found %.6f, newton %.6f)  areas %.5f %.5f  pred %.2e %.2e  gpu %.1f GB\n",
            rec.t, time() - t_wall0, rec.gauge_l2, rec.gauge_linf, num(rec.ham_l2),
            rec.residual, rec.separation, sep_found, newton_separation(d_sep, M, rec.t),
            num(rec.holes[1].area), num(rec.holes[2].area),
            num(rec.holes[1].track_prediction_max), num(rec.holes[2].track_prediction_max),
            device_gb())
    flush(stdout)
    return nothing
end

out = try
    evolve!(T, case; forest=forest0, q=q, ops=Operators(prolongation=q + 2,
                                                         restriction=q + 2),
            t_end=t_end, chunk=chunk, cfl=cfl, backend=backend, observer=observe,
            adm_every=adm_every, comm=comm, ckw...)
catch err
    abort_all(err)
    rethrow()
end
if isrank0
    open(joinpath(outdir, "records.txt"), "w") do io
        for r in out.records
            println(io, repr(r))
        end
    end
    @printf("done: t = %.4f, %d steps, finished = %s, wall %.1f s\n", out.t, out.nsteps,
            out.finished, time() - t_wall0)
end
out.finished || exit(3)
