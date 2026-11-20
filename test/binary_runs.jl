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
# Step H4 (`chart=harmonic`): two harmonic Kerr holes with no gauge source,
# `H ≡ 0`, or with the damped harmonic gauge rolled on from zero at `t0` or at the
# separation `d_g`; the layer's defaults are then the harmonic chart's, `r_0 =
# 2/5`, `r_1 = 3/4`, and the momentum from `C_a = 0`. A local smoke run at `q = 2`
# (a unit harmonic hole needs `h = 1/16` there):
#
#     julia --project=. --threads=4 test/binary_runs.jl chart=harmonic d=8 L=8 N=8 roots=2 radii=4,3,2,6/5 q=2 masses=1,1 r_0=7/20 r_1=37/50 margin=4 t_end=1/10 chunk=1/20 finder_N=12 spin=0 adm=0 out=out/hpair
#
# Options (`key=value`; the defaults are step B3's head-on row at `1/48`):
#
#   d=8 L=32 N=24 roots=8          the separation, the box [−L, L]³, the blocks
#   radii=16,8,4,2                 the balls about each hole, one per level (in M)
#   path=0                         sweep each ball this far toward the center of
#                                  mass: the capsule a head-on pair falls through
#   mesh=fixed|follow              the nested balls frozen, or (step H4) the mesh
#                                  that follows the holes: the indicator's,
#                                  regridded every chunk, each hole's floor on its
#                                  tracked horizon out to floor_margin (in M), with
#                                  refine_tol=2/5 coarsen_tol=1/10
#   t_end=2 chunk=1/4 cfl=1/2 q=4  the run
#   eps=1/2 gamma2=0               ε_KO and γ2 (γ0: holes_damping's three Gaussians)
#   r_0=5/8 r_1=23/16 margin=8     each hole's layer, stated for a hole of mass one
#                                  (each hole takes it times M_A): step B2's
#                                  production :damped sphere
#   find_interval=0                the longest interval between two finds for a
#                                  hole of mass one, in M (0: the chunk); for
#                                  fitted holes the pair's, in M (step H4)
#   chart=kerr_schild|harmonic     the holes' chart (step H4: harmonic runs with no
#                                  source, its layer defaults r_0=2/5 r_1=3/4)
#   masses=1/2,1/2                 the holes' masses
#   seed=newton|rest               the tracks' seeds fall freely, or start at rest
#                                  and learn the acceleration from the finds (the
#                                  harmonic pair's coordinates do not fall at first)
#   interior=damped|fitted         each hole's layer: step B2's tracked :damped
#                                  target (r_0, r_1), or step 8e's :fitted target
#                                  about its own track (step H4: margin, n_L=0
#                                  for step 8c's rule, fit_cont=2, lmax_fit=8,
#                                  fit_depth=0 the initial data's switch depth)
#   t0= d_g= sigma_g=10 sigma_r=32 (harmonic) roll the damped harmonic gauge on from
#                                  zero at t0, or when the tracks are d_g apart,
#                                  over σ_g; σ_r its far-field Gaussian (none: H ≡ 0)
#   target=superposed|own          each layer relaxes toward the superposition of
#                                  the holes on their tracks, or toward its own hole
#   feed=tracks|seed               the gauge source's holes follow the tracks
#                                  (pairing (α)) or their seeds' Newtonian quadratics
#   momentum=superposition|gauge   the initial Π: the data's own, or from C_a = 0
#   companion=4 attenuation=       the data's companion attenuation λ_B/M_B (0:
#                                  none) and XCTS's own one (a length; none)
#   adm=1                          the ADM monitor every this many chunks (0: never)
#   finder_N=16 spin=1             the finder's resolution, and the Korzyński J
#   backend=cpu|cuda mpi=false     as test/octant_runs.jl's
#   out=<dir>                      where `binary.csv`, `records.txt` and the log go
#   name=<label> group=<label>     the run's SimWatch name and group (step H4): every
#                                  run keeps `simwatch.toml` in `out`
#   checkpoint=<dir> walltime=<s>  a chain of jobs; checkpoint_every=4 keep=2
#   xcts=<file>                    (step S1) the initial data solved by XCTS
#                                  (`examples/harmonic_holes.jl save=<file>`): the
#                                  run's start — the cycle, the finds and the fits
#                                  on those data — written to checkpoint=<dir> with
#                                  t_end = 0, on one CPU process, from an environment
#                                  that has XCTS (bench/xcts_metric.jl); the same
#                                  command without xcts= then continues the run; a
#                                  start that read the data inside XCTS's inner
#                                  spheres (fit_depth too deep) is refused
#   xcts_gauge=case|solved         the lapse and shift of the XCTS start: the case's
#                                  background's (XCTS's fixed lapse grows outward,
#                                  bench/xcts_metric.jl), or the solution's own
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
# Data solved by XCTS (step S1): the run's start from them, written as a checkpoint
# that the same command without `xcts=` continues. The file needs XCTS loaded, which
# this package's environment does not have (`bench/xcts_metric.jl`).
const XCTS_FILE = get(OPTIONS, "xcts", nothing)
# The outer faces' mismatch (step S1's acceptance): the run starts from the solved data,
# and its Dirichlet data are the case's own background. On a 17 × 17 grid of each face
# of [−L, L]³, the largest |g_ab| and |∂_c g_ab| of their difference, and the
# background's own largest |g_ab − η_ab| for scale.
function face_mismatch(background, data, L)
    η = SVector{4,Float64}(-1, 1, 1, 1)
    dg_max = 0.0; ddg_max = 0.0; dev_max = 0.0
    s = range(-L, L; length=17)
    for ax in 1:3, side in (-1, 1), a in s, b in s
        xs = ax == 1 ? (side * L, a, b) : ax == 2 ? (a, side * L, b) : (a, b, side * L)
        x = SVector{4,Float64}(0, xs...)
        g1, dg1 = TreeGeneralizedHarmonic.SpacetimeMetrics.dmetric(background, x)
        g2, dg2 = TreeGeneralizedHarmonic.SpacetimeMetrics.dmetric(data, x)
        dg_max = max(dg_max, maximum(abs, g2 - g1))
        ddg_max = max(ddg_max, maximum(abs, dg2 - dg1))
        dev_max = max(dev_max, maximum(abs(g1[i, j] - (i == j ? η[i] : 0)) for i in 1:4, j in 1:4))
    end
    @printf("outer faces (|x_i| = %g): max |Δg_ab| %.3e, max |Δ∂_c g_ab| %.3e (background's max |g_ab − η_ab| %.3e)\n",
            L, dg_max, ddg_max, dev_max)
    flush(stdout)
    return nothing
end
XCTS_FILE === nothing || include(joinpath(@__DIR__, "..", "bench", "xcts_metric.jl"))
XCTS_FILE === nothing || (t_end = 0)
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
chart = Symbol(opt("chart", "kerr_schild"))
harmonic = chart === :harmonic
fitted_holes = opt("interior", "damped") == "fitted"
unit = fitted_holes ?
       FittedSpec(T; variant=:fitted, margin=parse(Int, opt("margin", "8")),
                  n_L=parse(Int, opt("n_L", "0")),
                  fit_cont=parse(Int, opt("fit_cont", "2")),
                  lmax_fit=parse(Int, opt("lmax_fit", "8"))) :
       TrackedSpec(T; r_0=T(rat(opt("r_0", harmonic ? "2/5" : "5/8"))),
                   r_1=T(rat(opt("r_1", harmonic ? "3/4" : "23/16"))),
                   margin=parse(Int, opt("margin", "8")),
                   max_interval=T(rat(opt("find_interval", "0"))))
masses = Tuple(rat(m) for m in split(opt("masses", "1/2,1/2"), ','))
holes = head_on_holes(T; d=d_sep, masses=masses, interior=unit, chart=chart,
                      newtonian=opt("seed", "newton") == "newton")
# The roll-on from zero (step H4): at `t0`, or at the separation `d_g`, which
# leaves `t_0 = Inf` to the event.
rolling = harmonic && (haskey(OPTIONS, "t0") || haskey(OPTIONS, "d_g"))
(haskey(OPTIONS, "t0") || haskey(OPTIONS, "d_g")) && !harmonic &&
    error("t0 and d_g roll the damped harmonic gauge on from zero: chart=harmonic only.")
rollon_separation = haskey(OPTIONS, "d_g") ? T(rat(OPTIONS["d_g"])) : nothing
gauge_source = !rolling ? :holes :
               RolledGaugeSource(ZeroGaugeSource(T),
                                 DampedHarmonicSource(T; σ_r=T(rat(opt("sigma_r", "32"))));
                                 σ_g=T(rat(opt("sigma_g", "10"))),
                                 t_0=rollon_separation === nothing ?
                                     T(rat(OPTIONS["t0"])) : T(Inf))
companion = rat(opt("companion", "4"))
attenuation = haskey(OPTIONS, "attenuation") ? T(rat(OPTIONS["attenuation"])) : nothing
horizon = Horizon(T; every=1, N=parse(Int, opt("finder_N", "16")),
                  spin=opt("spin", "1") in ("1", "true"))
# The following mesh (step H4): `test/octant_runs.jl`'s `mesh=follow`, one floor
# per hole (`CODE.md`, "Refinement for two holes").
meshname = opt("mesh", "fixed")
meshname in ("fixed", "follow") || error("mesh is fixed or follow, got $meshname")
follow = meshname == "follow"
follow && haskey(OPTIONS, "path") && error("path = is the fixed mesh's capsule")
refinement = follow ?
             Refinement(T; refine_tol=T(rat(opt("refine_tol", "2/5"))),
                        coarsen_tol=T(rat(opt("coarsen_tol", "1/10"))),
                        maxlevel_cap=length(radii),
                        floor_margin=T(rat(opt("floor_margin", "0"))), ceiling_cells=1) :
             nothing
case = binary_case(T; holes=holes, halfwidth=L, chunk=chunk, ε_KO=ε_KO, γ2=γ2,
                   refinement=refinement,
                   horizon=horizon, companion=companion, attenuation=attenuation,
                   hole_target=Symbol(opt("target", "superposed")),
                   source_feed=Symbol(opt("feed", "tracks")),
                   momentum=haskey(OPTIONS, "momentum") ? Symbol(OPTIONS["momentum"]) :
                            nothing,
                   gauge_source=gauge_source)
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
initial_data = nothing
if XCTS_FILE !== nothing
    (ckdir === nothing || restarting || use_mpi || opt("backend", "cpu") != "cpu") && error(
        "xcts=<file> writes a run's start from data solved by XCTS as a checkpoint, on " *
        "one CPU process: it needs checkpoint=<dir> with no checkpoint in it yet, and " *
        "neither mpi=true nor backend=cuda. The same command without xcts= continues it.")
    initial_data = load_xcts_metric(XCTS_FILE;
                                    gauge=opt("xcts_gauge", "case") == "case" ? case.background : nothing)
    isrank0 && println("initial data: ", nameof(initial_data))
    # The outer faces' mismatch (step S1's acceptance): the run starts from the solved
    # data, and its Dirichlet data are the case's own background. On a 17 × 17 grid of
    # each face of [−L, L]³, the largest |g_ab| and |∂_c g_ab| of their difference, and
    # the background's own largest |g_ab − η_ab| for scale.
    isrank0 && face_mismatch(case.background, initial_data, T(L))
end
forest0 = restarting ? nothing :
          holes_forest(T, case; N=N, roots=roots, radii=radii, comm=comm,
                       path=T(rat(opt("path", "0"))))
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
const BANDS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, Inf)   # 0.5 and 0.75 added in step S1
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
            ham_linf=b.mx[5].value,
            mom_l2=n > 0 ? sqrt((b.s2[6].value + b.s2[7].value + b.s2[8].value) / n) : NaN)
end

# The lapse outside the horizons (step H4; `PLAN.md`'s step H4, "in pure harmonic
# gauge"): the smallest `α` and the largest `√γ/α` over the evolved points outside
# both horizons' coordinate spheres, and over the slab between the holes — `x`
# between the horizons' facing sides, `|y|, |z| ≤ d/2`.
struct LapseFold{T}
    c1::SVector{3,T}
    c2::SVector{3,T}
    r1::T
    r2::T
    w::T
end
const NOLAPSE = (out_α=Min(floatmax(T)), out_a=Max(zero(T)), btw_α=Min(floatmax(T)),
                 btw_a=Max(zero(T)))
@inline function (f::LapseFold{T})(acc, p) where {T}
    x = SVector{3,T}(p.x[1], p.x[2], p.x[3])
    out = (sqrt(sum(abs2, x - f.c1)) > f.r1) & (sqrt(sum(abs2, x - f.c2)) > f.r2)
    btw = out & (f.c1[1] + f.r1 < x[1] < f.c2[1] - f.r2) & (abs(x[2]) ≤ f.w) &
          (abs(x[3]) ≤ f.w)
    hv = SVector{10,T}(ntuple(v -> p.u[v], Val(10)))
    _, _, α, _, _, sg = TreeGeneralizedHarmonic.metric_quantities(
        TreeGeneralizedHarmonic._sym4(hv))
    a = sg / α
    q = (out_α=Min(out ? α : floatmax(T)), out_a=Max(out ? a : zero(T)),
         btw_α=Min(btw ? α : floatmax(T)), btw_a=Max(btw ? a : zero(T)))
    return TreeAMR.combine(acc, q)
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

# The SimWatch status file (added in step H4): progress, the constraints and the
# pair's separation as sparklines, each hole's horizon.
sw = SimWatchWriter(outdir; name=opt("name", basename(outdir)),
                    code="TreeGeneralizedHarmonic", group=opt("group", nothing),
                    comm=comm,
                    summary=[(key="constraints.gauge_l2", label="C_a"),
                             (key="pair.separation", label="sep"),
                             (key="lapse.min_between", label="α between")])
simwatch_update!(sw; force=true, status="starting", time_end=Float64(t_end),
                 time_unit="M", walltime_limit=walltime,
                 message="compiling and building the initial data")

csv = joinpath(outdir, "binary.csv")
hcols = ["succ", "cx", "cy", "cz", "vx", "ox", "rmin", "rmax", "area", "Mirr", "J", "Mch",
         "pred", "finds", "offx", "srcx", "srcvx", "srcax", "fit_valid", "fit_residual"]
bcols = [("h$(A)_" * b * "_" * f) for A in 1:2 for b in ("r$(BANDS[k])" for k in 1:NBAND)
         for f in ("ham_l2", "gauge_l2")]
header = vcat(["t", "wall", "steps", "dt", "nblocks", "points", "gauge_l2", "gauge_linf",
               "ham_l2", "ham_linf", "residual", "finite"],
              ["h$(A)_$c" for A in 1:2 for c in hcols],
              ["sep_track", "sep_found", "sep_newton", "gpu_gb", "maxrss_gb"], bcols,
              ["rollon_t0", "min_alpha_out", "max_sqrtg_alpha_out", "min_alpha_between",
               "max_sqrtg_alpha_between"],
              # ℳ (added in step S1), last so that earlier columns keep their places:
              # overall, then per hole and band as `bcols`
              ["mom_l2"; ["h$(A)_r$(BANDS[k])_mom_l2" for A in 1:2 for k in 1:NBAND]])
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
    # The lapse rows, from the state itself (`scatter!` writes `u` into the
    # working array; the next right-hand side scatters again).
    rA(A) = max(num(rec.holes[A].track_r_max), (fitted_holes ? 1 : unit.r_1) * holes[A].mass)
    lf = mesh_fold(LapseFold{T}(c1, c2, T(rA(1)), T(rA(2)), T(rec.separation / 2)),
                   TreeAMR.combine, NOLAPSE, p.U, u)
    lapse = map(b -> b.value, lf)
    # Everything below is the same on every rank; the CSV and the log are rank
    # 0's, and the SimWatch update is collective, so every rank makes it.
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
                      get(h, :source_center, (NaN,))[1],
                      get(h, :source_velocity, (NaN,))[1],
                      get(h, :source_acceleration, (NaN,))[1],
                      get(h, :fit_valid, nothing) === nothing ? NaN : Int(h.fit_valid),
                      num(get(h, :fit_residual, nothing))))
    end
    append!(row, (rec.separation, sep_found, newton_separation(d_sep, M, rec.t),
                  device_gb(), Sys.maxrss() / 1e9))
    for k in 1:(2 * NBAND)
        b = bandnorms(rf.band[k])
        append!(row, (b.ham_l2, b.gauge_l2))
    end
    append!(row, (num(get(rec, :rollon_t0, nothing)), lapse.out_α, lapse.out_a,
                  lapse.btw_α, lapse.btw_a))
    push!(row, all.mom_l2)
    for k in 1:(2 * NBAND)
        push!(row, bandnorms(rf.band[k]).mom_l2)
    end
    isrank0 && open(io -> println(io, join(row, ',')), csv, "a")
    hist = Dict("constraints.gauge_l2" => rec.gauge_l2, "pair.separation" => rec.separation,
                "lapse.min_between" => lapse.btw_α, "layer.residual" => rec.residual)
    bhs = [(name="BH$A", irreducible_mass=h.M_irr, mass=h.M_ch, spin=h.J,
            position=collect(h.track_center), found=h.success)
           for (A, h) in enumerate(rec.holes)]
    extra = Dict("constraints" => Dict("gauge_l2" => rec.gauge_l2,
                                       "ham_l2" => num(rec.ham_l2)),
                 "pair" => Dict("separation" => rec.separation,
                                "newton" => newton_separation(d_sep, M, rec.t)),
                 "lapse" => Dict("min_between" => lapse.btw_α,
                                 "min_outside" => lapse.out_α,
                                 "max_sqrtg_alpha_between" => lapse.btw_a),
                 "layer" => Dict("residual" => rec.residual),
                 "rollon_t0" => num(get(rec, :rollon_t0, nothing)))
    simwatch_update!(sw; force=true, history=hist, iteration=rec.steps, time=rec.t,
                     time_end=Float64(t_end), time_unit="M", walltime_limit=walltime,
                     black_holes=bhs, extra=extra,
                     message=@sprintf("t = %.3f M, separation %.4f", rec.t, rec.separation))
    isrank0 || return nothing
    @printf("t = %7.4f  wall %8.1f s  C_a %.3e/%.3e  ℋ %.3e  res %.3e  sep %.6f (found %.6f, newton %.6f)  areas %.5f %.5f  pred %.2e %.2e  gpu %.1f GB\n",
            rec.t, time() - t_wall0, rec.gauge_l2, rec.gauge_linf, num(rec.ham_l2),
            rec.residual, rec.separation, sep_found, newton_separation(d_sep, M, rec.t),
            num(rec.holes[1].area), num(rec.holes[2].area),
            num(rec.holes[1].track_prediction_max), num(rec.holes[2].track_prediction_max),
            device_gb())
    @printf("          α_min %.5f (between %.5f)  √γ/α_max %.4f (between %.4f)  t_0 %s\n",
            lapse.out_α, lapse.btw_α, lapse.out_a, lapse.btw_a,
            string(get(rec, :rollon_t0, "none")))
    flush(stdout)
    return nothing
end

out = try
    evolve!(T, case; forest=forest0, q=q, ops=Operators(prolongation=q + 2,
                                                         restriction=q + 2),
            t_end=t_end, chunk=chunk, cfl=cfl, backend=backend, observer=observe,
            adm_every=adm_every, comm=comm, rollon_separation=rollon_separation,
            regrid=follow,
            hole_find_interval=fitted_holes && haskey(OPTIONS, "find_interval") ?
                               T(rat(OPTIONS["find_interval"])) : nothing,
            fit_initial_depth=fitted_holes ? T(rat(opt("fit_depth", "0"))) : 0,
            initial_data=initial_data, ckw...)
catch err
    use_mpi || simwatch_finish!(sw; status="failed", message=sprint(showerror, err),
                                time_end=Float64(t_end), time_unit="M")
    abort_all(err)
    rethrow()
end
# The start must not have read the data inside XCTS's inner spheres
# (`check_xcts_reach`): a refused start's checkpoint is renamed, so that the
# continuation cannot take it up.
if XCTS_FILE !== nothing
    try
        check_xcts_reach(initial_data, out.geometry, T(rat(opt("fit_depth", "0"))))
    catch err
        foreach(f -> mv(f, f * ".refused"; force=true), out.checkpoints_written)
        rethrow()
    end
end
simwatch_finish!(sw; status=out.finished ? "finished" : "stopped", time=Float64(out.t),
                 time_end=Float64(t_end), time_unit="M",
                 message=out.finished ? "finished" :
                         "stopped at the wall-time limit, to be continued")
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
