# Runs on the octant (added 2026-10-02): white noise on flat space, on the
# nested hierarchy a single black hole at the origin will run on, and that
# black hole — Kerr-Schild `M = 1`, `a = 0`, with the damping layer and the
# algebraic gauge source, with or without the same noise. From 2026-10-04 the
# hole may spin about `z` (`a=`), on the rotating octant (`octant=rotating`).
#
# A standalone script, not part of the suite. The box is `[0, L]³` with
# reflecting faces through the origin — or, on the rotating octant, the seam
# gluing `x = 0` to `y = 0` by a quarter turn about `z` and a reflecting face
# at `z = 0` only — and Minkowski Dirichlet data
# (`h = Π = 0`, constant in time) at the three outer faces; the mesh is
# `hole_forest(; shape = :cube)` about the origin, `h = L/(roots N)` at the
# outer boundary and halved inside each cube `[0, R]³` of `radii`. The initial
# data are `add_noise!`'s uniform noise on every owned point and variable,
# projected onto the parity on the wall planes and left out of the seam's two
# planes on the rotating octant. Each chunk the observer runs
# both monitors and writes one CSV row: the ADM constraints `ℋ`, `ℳ_i` and
# the gauge constraints `C_a`, separately, point-weighted (every grid point
# the same weight) and volume-weighted, over the mesh and per level, and the
# state's own point-weighted norms.
#
# From 2026-10-20 the hole may move (`octant=quadrant v=1/5 x0=-4`): a boosted
# Kerr-Schild hole on the x axis, on the quadrant `[−L, L] × [0, L]²` with
# mirrors at `y = 0` and `z = 0`, on shells swept along its path. Everything
# measured about the hole — the shells, the noise's excluded core, the finder —
# is about `c(t) = (x0 + v t, 0, 0)`, and the CSV adds each shell's trailing and
# leading halves and the horizon's extents (`split`, on for the quadrant).
#
# From 2026-10-21 a spinning hole may move along its spin axis
# (`octant=rotating_quadrant a=1/2 v=1/5 x0=-1`): a boosted Kerr-Schild hole on
# the z axis, on the rotating quadrant `[0, L]² × [−L, L]` — the seam gluing
# `x = 0` to `y = 0` by a quarter turn about `z`, no mirror, Dirichlet at
# `x = L`, `y = L` and `z = ±L` — with `c(t) = (0, 0, x0 + v t)`. The motion
# axis is `x` on the quadrant and `z` on the rotating quadrant, and everything
# the quadrant measures along `x` it measures along `z` there.
#
#     julia --project=. --threads=4 test/octant_runs.jl case=ks octant=quadrant v=1/5 x0=-1 L=8 N=16 roots=1 radii=4,3,2 r_0=5/8 r_1=23/16 t_end=1 chunk=1/2 cfl=1/2 amplitude=0 out=out/quad
#
#     julia --project=. --threads=4 test/octant_runs.jl case=ks octant=rotating_quadrant a=1/2 v=1/5 x0=-1 L=8 N=24 roots=1 radii=4,3,2 r_0=3/4 r_1=23/16 t_end=1 chunk=1/2 cfl=1/2 amplitude=0 out=out/spin
#
#     julia --project=. --threads=4 test/octant_runs.jl L=16 radii=8,4 t_end=1 out=out/oct
#
# Options (`key=value`; the defaults are the production run's):
#
#   L=128 N=32 radii=64,32,16,8,4   the box and the hierarchy (roots = L/N, or roots=)
#
# For the hole the record also carries the error against the exact solution
# (point-weighted, overall and per level), the layer's residual and the
# drift of `h_tt` at the horizon, and every norm over the outermost blocks
# (`bnd`, the ones touching an outer face), which is where a boundary problem
# shows first. A `:fitted` hole finds its horizon every chunk — TreeAMR's
# `interpolate` mirrors the finder's and the fit's spheres into the octant
# with each variable's parity — and the run's own record, with the horizon's
# numbers and the track, is written to `records.csv` at the end.
#   t_end=128 chunk=1 cfl=1/4 q=4   the run
#   eps=1/2 gamma0=1 gamma2=0       ε_KO, γ0, γ2
#   case=minkowski|ks               flat space, or the Kerr-Schild hole
#   octant=reflecting|rotating|quadrant|rotating_quadrant
#                                   three mirrors (a = 0 only), or the
#                                   quarter turn about z and the mirror at z = 0
#                                   (any a), or (added 2026-10-20) the quadrant
#                                   [−L, L] × [0, L]² with mirrors at y = 0 and
#                                   z = 0, for a hole on the x axis moving along
#                                   it (ks, a = 0 only): a (2r, r, r) root brick
#                                   and the shells swept along the path; or
#                                   (added 2026-10-21) the rotating quadrant
#                                   [0, L]² × [−L, L], the quarter turn about z
#                                   and no mirror, for a hole spinning about z
#                                   and moving along it (ks, any a): a (r, r, 2r)
#                                   root brick and the shells swept along z
#   v=0 x0=0                        the quadrants only: the hole's speed along the
#                                   motion axis — x on the quadrant, z on the
#                                   rotating quadrant — and its position on that
#                                   axis at t = 0; it is at x0 + v t along it
#   path=v·t_end                    the quadrants only: the length along the
#                                   motion axis of the segment from x0 the shells
#                                   are swept along (the capsule mesh, frozen; in M)
#   split=0|1                       also each shell's trailing ((x − c(t))·v̂ < 0)
#                                   and leading (> 0) half, and the horizon's
#                                   extents and their ratio along/across the motion
#                                   (x/y; z/x on the rotating quadrant) against
#                                   √(1 − v²) times the rest frame's (1 along x;
#                                   r₊/√(r₊² + a²) = √(r₊/2) along z) (default: on
#                                   for the quadrants, v̂ = +x — +z on the rotating
#                                   quadrant — at v = 0). On the rotating quadrant
#                                   the centre and the found origin are `c_z` and
#                                   `hz_z`, and `hz_J` is the Korzyński J's
#                                   component along z, J times the spin axis's z
#                                   (the magnitude elsewhere)
#   a=0                             the hole's spin along z, in M (ks only; a ≠ 0
#                                   needs octant=rotating or rotating_quadrant, and
#                                   r_0 > a for :damped, which the ring of radius
#                                   a must be inside)
#   r_0=3/4 r_1=3/2 source=algebraic|sampled   the hole's layer and gauge source
#   interior=damped|fitted          step 5's analytic layer, or step 8's fitted
#                                   target on the tracked horizon (margin m=8,
#                                   lmax_shape=4, lmax_fit=8, the finder every
#                                   chunk with the spin, default_bounds at 9/10)
#   margin=8                        cells from the horizon to the layer's outer edge
#                                   (:damped: the check on r_1; :fitted: the offset)
#   n_L=0 lmax_fit=8                :fitted only: the ramp in cells (0: step 8c's
#                                   rule) and the fit's degree
#   fit_cont=1                      :fitted only: the fit's radial order, 1 (values
#                                   and slopes) or 2 (and curvatures), for the
#                                   evolved state's fits and the initial data's
#   fit_depth=n_L h                 :fitted only: the analytic initial data down to
#                                   this depth below the offset surface (the core
#                                   surface, step 8f's ks0 row), the fit below
#   variant=fitted                  interior=fitted only (added 2026-10-20): the
#                                   tracked geometry's variant — damped is the
#                                   same geometry with the exact target, the
#                                   screen CODE.md asks for before a new depth
#   trail_ramp=0 target_rate=1      interior=fitted only (added 2026-10-20):
#                                   evolve!'s levers for a moving :fitted layer
#                                   (step 8′ ran trail_ramp=9/10)
#   rho=                            a fixed ρ_max in 1/M (default: 4/M)
#   finder=1                        find the horizon every chunk (0: never; :fitted
#                                   always does)
#   shells=2,9/4,3,5,8              radii of the shells outside the horizon whose
#                                   norms the CSV carries (and one beyond the last;
#                                   and the evolved band inside the horizon). The
#                                   first is the horizon's equatorial radius
#                                   √(r₊² + a²) = √(2 r₊) by default, 2 at a = 0:
#                                   the KS horizon of a spinning hole is oblate, so
#                                   the band inside it also holds points outside
#                                   the horizon near the poles
#   amplitude=1e-8 seed=20261002    the noise (amplitude=0: none)
#   backend=cpu|cuda                cuda needs CUDA in the active environment
#   simd=                           the CPU kernel's SIMD width (1: the scalar
#                                   kernel a device runs, which throws where the
#                                   lanes leave a NaN; default: the host's)
#   mpi=true deviceaware=false      run over MPI.COMM_WORLD (MPI in the active
#                                   environment; launch with mpiexec), one GPU per
#                                   rank by its rank on the node; deviceaware=true
#                                   hands MPI device buffers (a CUDA-aware MPI).
#                                   Under Open MPI launch with `--bind-to none`:
#                                   it binds a rank to one core otherwise, and
#                                   rank 0's find, spin and fit run 3.8× slower
#                                   (CODE.md, "Running distributed")
#   out=<dir>                       where `octant.csv` and the log go
#   name=<label>                    the run's name in SimWatch (default: out's name)
#   group=<label>                   its SimWatch group, e.g. a study's (2026-10-21)
#   checkpoint=<dir> walltime=<s>   a chain of jobs (as hole_runs.jl's rows)
#   checkpoint_every=8 keep=2       chunks between checkpoints, and how many to keep
#                                   (keep them all to difference two runs later)
#
# Every run also keeps a SimWatch status file, `simwatch.toml` in the `out`
# directory (https://github.com/eschnett/simwatch): the progress, everything
# the CSV row carries, and the record's horizon, track and fit, rewritten at
# every chunk, with `finished`, `stopped` or `failed` at the end.
# From 2026-10-19 it also carries a summary, the last 100 chunks of the
# constraints, the error and the horizon as `[history]` (SimWatch draws them as
# sparklines), and two thumbnails in `out/plots/` (`test/simwatch_plots.jl`;
# SVG from 2026-10-21, on a transparent background that SimWatch 1.3.0 adapts
# to its user's terminal).
#
# A run that stops at `walltime` exits with status 3, so a job script can
# resubmit; the same command continues it from the newest checkpoint, and
# appends to the CSV.
#
# Over MPI (`mpi=true`, added 2026-10-19) every rank runs this script with the
# same options — `mpiexec -n 4 julia --project=<env> test/octant_runs.jl mpi=true
# backend=cuda …` — the forest's blocks are split between the ranks, each rank
# drives the GPU its rank on the node names, and rank 0 alone prints, writes
# the CSV and the records and the status file. An exception on one rank takes
# the job down (`MPI.Abort`) rather than leave the others waiting.

using Printf
using Random: Xoshiro
using TreeAMR
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: DIAG_CGH, DIAG_HAM, DIAG_MOM
using StaticArrays: SVector

include(joinpath(@__DIR__, "simwatch_plots.jl"))

const OPTIONS = Dict{String,String}()
for arg in ARGS
    k, v = occursin('=', arg) ? split(arg, '='; limit=2) : (arg, "")
    OPTIONS[String(k)] = String(v)
end
opt(k, default) = get(OPTIONS, k, default)
rat(s) = (p = split(s, '/'); length(p) == 2 ? parse(Int, p[1]) // parse(Int, p[2]) :
                            occursin(r"[.eE]", s) ? parse(Float64, s) : parse(Int, s))

const T = Float64
L = rat(opt("L", "128"))
N = parse(Int, opt("N", "32"))
radii = [rat(r) for r in split(opt("radii", "64,32,16,8,4"), ',')]
t_end = rat(opt("t_end", "128"))
chunk = rat(opt("chunk", "1"))
cfl = rat(opt("cfl", "1/4"))
q = parse(Int, opt("q", "4"))
ε_KO = rat(opt("eps", "1/2"))
γ0 = rat(opt("gamma0", "1"))
γ2 = rat(opt("gamma2", "0"))
amplitude = parse(Float64, opt("amplitude", "1e-8"))
seed = parse(Int, opt("seed", "20261002"))
outdir = mkpath(abspath(opt("out", "out/octant")))
ckdir = haskey(OPTIONS, "checkpoint") ? mkpath(abspath(OPTIONS["checkpoint"])) : nothing
walltime = haskey(OPTIONS, "walltime") ? parse(Float64, OPTIONS["walltime"]) : nothing
ckevery = parse(Int, opt("checkpoint_every", "8"))
ckkeep = parse(Int, opt("keep", "2"))

# MPI (added 2026-10-19): loaded only when asked for, so that a serial run
# needs no MPI in its environment.
use_mpi = opt("mpi", "false") in ("true", "1")
if use_mpi
    @eval using MPI
    MPI.Init()
end
isrank0 = !use_mpi || MPI.Comm_rank(MPI.COMM_WORLD) == 0
# One rank's failure would leave the others waiting in their next collective.
function abort_all(err)
    use_mpi || return nothing
    showerror(stderr, err, catch_backtrace())
    println(stderr)
    MPI.Abort(MPI.COMM_WORLD, 1)
end
casename = opt("case", "minkowski")
casename in ("minkowski", "ks") || error("case is minkowski or ks, got $casename")
hole = casename == "ks"
symmetry = opt("octant", "reflecting")
symmetry in ("reflecting", "rotating", "quadrant", "rotating_quadrant") ||
    error("octant is reflecting, rotating, quadrant or rotating_quadrant, got $symmetry")
rotating = symmetry == "rotating"
quadrant = symmetry == "quadrant"
# The rotating quadrant (added 2026-10-21): the spinning hole moving along z.
rquadrant = symmetry == "rotating_quadrant"
moving_domain = quadrant || rquadrant
# The axis the hole sits and moves on: z on the rotating quadrant, x elsewhere
# (the split's `v̂` at rest on an octant, as before).
axis = rquadrant ? 3 : 1
axname = ("x", "y", "z")[axis]
# The extent across the motion the horizon's ratio divides by: y for x, x for z.
across = axis == 1 ? 2 : 1
a_spin = rat(opt("a", "0"))
hole || iszero(a_spin) || error("a = $a_spin is the hole's spin, and case = minkowski has none")
# The moving hole (added 2026-10-20): on the quadrant, which is the one
# domain a hole off the origin and moving along x is its own mirror image on,
# and (added 2026-10-21) on the rotating quadrant, the one a spinning hole
# moving along z is its own image on under the quarter turn.
v_hole = rat(opt("v", "0"))
x0_hole = rat(opt("x0", "0"))
path_x = haskey(OPTIONS, "path") ? rat(OPTIONS["path"]) : v_hole * t_end
moving_domain && !hole &&
    error("octant = $symmetry is the moving hole's domain, and needs case = ks")
moving_domain || (iszero(v_hole) && iszero(x0_hole) && !haskey(OPTIONS, "path")) ||
    error("v = $v_hole, x0 = $x0_hole and path = are a hole on an axis, which " *
          "only octant = quadrant (x) or rotating_quadrant (z) maps onto itself " *
          "(the $symmetry octant needs the hole at the origin, at rest)")
abs(v_hole) < 1 || error("v = $v_hole is the hole's speed in units of c, and must be below 1")
# The trailing/leading split and the horizon's extents (added 2026-10-20): new
# CSV columns, so on by default only for the quadrant, which has no older CSVs
# to match. `v̂` is `+x` for a hole at rest.
split_on = hole && opt("split", moving_domain ? "1" : "0") in ("1", "true")
v_dir = v_hole < 0 ? -one(T) : one(T)
contraction_expected = sqrt(1 - T(v_hole)^2)
# The Kerr-Schild horizon (`M = 1`): `r₊` at the poles and `√(r₊² + a²) = √(2 r₊)`
# at the equator.
r_plus = 1 + sqrt(1 - T(a_spin)^2)
r_equator = sqrt(2 * r_plus)
M_irr_kerr = sqrt(r_plus / 2)
# The horizon's extent along the motion over the extent across it (added
# 2026-10-21): the contraction times the rest frame's ratio, `1` along x and
# the polar over the equatorial radius, `r₊/√(r₊² + a²)`, along the spin axis z.
ratio_expected = axis == 3 ? contraction_expected * (r_plus / r_equator) :
                 contraction_expected
# The least radius of a hole moving at `v` (`a = 0`, or along the spin): `r₊`
# contracted along the boost, exactly `r₊` at rest.
r_hmin = r_plus * contraction_expected

backend = if opt("backend", "cpu") == "cuda"
    @eval using CUDA
    if use_mpi
        # One GPU per rank: the rank's place among the ranks of its node,
        # TreeAMR's `test/mpi_device_workload.jl`.
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

# One root block of `N` cells of `h = 1` per `N` units by default; `roots=`
# sets another root brick (a small box for a smoke test).
roots = haskey(OPTIONS, "roots") ? parse(Int, OPTIONS["roots"]) : Int(L ÷ N)
haskey(OPTIONS, "roots") || roots * N == L ||
    error("L = $L must be a multiple of N = $N (one root block is N cells of h = 1)")
case = if hole
    src = opt("source", "algebraic")
    src in ("algebraic", "sampled") || error("source is algebraic or sampled, got $src")
    # The layer by step 8c's rule at this order: `ρ_max = 4/M` (the default),
    # `ρ_ramp = 1`, `r_1 − r_0 ≥ n_L h`; GHSO2's `ε_KO`, the Gaussian `γ0`.
    gs = src == "algebraic" ? :algebraic : nothing
    finder = opt("finder", "1") != "0" ? Horizon(T; every=1, N=12, spin=true) : nothing
    # The hole's motion, exact (the case converts it into `T` once), passed
    # only on the quadrants so that the octants' cases are built as before, and
    # the axis only on the rotating quadrant, so the quadrant's is as before.
    motion = quadrant ? (v=v_hole, x0=x0_hole) :
             rquadrant ? (v=v_hole, x0=x0_hole, axis=3) : (;)
    if opt("interior", "damped") == "fitted"
        m = parse(Int, opt("margin", "8"))
        spec = FittedSpec(T; variant=Symbol(opt("variant", "fitted")), margin=m,
                          n_L=parse(Int, opt("n_L", "0")), lmax_shape=4,
                          lmax_fit=parse(Int, opt("lmax_fit", "8")),
                          fit_cont=parse(Int, opt("fit_cont", "1")))
        # The range projection's gate: step 8f's `9/10`, or `default_gate`'s
        # rule — the offset surface less `2G` spacings — where that is deeper
        # (a margin of 12 cells or more at `h = 1/16`).
        # Against the horizon's least radius, `r₊` at the poles — contracted
        # along the boost to `r₊ √(1 − v²)` for a moving hole (`a = 0`).
        hfine = T(L) / (roots * N) / 2^length(radii)    # the finest spacing
        r_gate = min(T(9 // 10), r_hmin - (m + 2 * (q ÷ 2 + 1)) * hfine)
        kerr_schild_case(T; a=a_spin, halfwidth=L, chunk=chunk, ε_KO=ε_KO, γ2=γ2,
                         octant=Symbol(symmetry), motion...,
                         gauge_source=gs, interior=spec,
                         horizon=Horizon(T; every=1, N=12, spin=true),
                         bounds=default_bounds(T; M=1, r_gate=r_gate))
    else
        haskey(OPTIONS, "variant") &&
            error("variant= is the tracked geometry's (interior = fitted); the " *
                  "sphere here is :damped")
        kerr_schild_case(T; a=a_spin, halfwidth=L, r_0=rat(opt("r_0", "3/4")),
                         r_1=rat(opt("r_1", "3/2")), ρ_ramp=1, chunk=chunk, ε_KO=ε_KO,
                         γ2=γ2, octant=Symbol(symmetry), motion..., gauge_source=gs,
                         margin=parse(Int, opt("margin", "8")), horizon=finder)
    end
else
    minkowski_octant_case(T; L=L, ε_KO=ε_KO, γ0=γ0, γ2=γ2, chunk=chunk,
                          rotating=rotating)
end
# The quadrant's box is twice as long in x, and its shells are swept along the
# hole's path from `(x0, 0, 0)` (added 2026-10-20); the rotating quadrant's is
# twice as long in z, swept from `(0, 0, x0)` (added 2026-10-21); the octants'
# forest is the one it was.
on_axis(s) = ntuple(d -> d == axis ? s : 0, 3)
forest0 = moving_domain ?
          hole_forest(T, case; N=N, roots=ntuple(d -> d == axis ? 2 * roots : roots, 3),
                      center=on_axis(x0_hole), radii=radii, shape=:cube,
                      path=iszero(path_x) ? nothing : on_axis(path_x), comm=comm) :
          hole_forest(T, case; N=N, roots=roots, center=(0, 0, 0), radii=radii,
                      shape=:cube, comm=comm)
nlev = length(radii) + 1
isrank0 && @printf("octant (%s): L = %s, N = %d, %d blocks, %d points, levels %d, h = %s … %s\n",
        symmetry, L, N, nleaves(forest0), nleaves(forest0) * N^3, nlev,
        maximum(k -> spacing(T, forest0, k), forest0.leaves), minimum_spacing(T, forest0))
# The exact point count by level (added 2026-10-20), `N³` owned points a block.
levels0 = level.(forest0.leaves)
points_by_level = [(ℓ, count(==(ℓ), levels0) * N^3) for ℓ in sort!(unique(levels0))]
isrank0 && println("        points by level: ",
                   join(("L$ℓ $(n) ($(n ÷ N^3) blocks)" for (ℓ, n) in points_by_level), ", "),
                   "; total ", sum(last, points_by_level))
isrank0 && @printf("        case %s, t_end = %s, chunk = %s, cfl = %s, q = %d, ε_KO = %s, γ0 = %s, γ2 = %s\n",
        casename, t_end, chunk, cfl, q, ε_KO, hole ? "Gaussian 1/M → 1/(10M)" : γ0, γ2)
quadrant && isrank0 && @printf("        moving: v = %s along x from x0 = %s (at x = %s at t_end), shells swept along %s; x/y contraction √(1 − v²) = %.6f\n",
                                v_hole, x0_hole, x0_hole + v_hole * t_end, path_x,
                                contraction_expected)
rquadrant && isrank0 && @printf("        moving: v = %s along z from z0 = %s (at z = %s at t_end), shells swept along %s; z/x extent √(1 − v²) · r₊/√(r₊² + a²) = %.6f · %.6f = %.6f\n",
                                 v_hole, x0_hole, x0_hole + v_hole * t_end, path_x,
                                 contraction_expected, r_plus / r_equator, ratio_expected)
hole && isrank0 && @printf("        hole: a = %s, %s, gauge source %s\n", a_spin,
                case.interior isa FittedSpec ?
                "fitted target, tracked, margin $(case.interior.margin)" :
                "damped layer, r_0 = $(case.interior.r_0), r_1 = $(case.interior.r_1)",
                case.gauge === nothing ? "sampled" : "algebraic")
isrank0 && @printf("        noise %.3g (seed %d), backend %s, %d threads, %d rank(s)\n",
                   amplitude, seed, typeof(backend), Threads.nthreads(),
                   use_mpi ? MPI.Comm_size(MPI.COMM_WORLD) : 1)

# --- the record: one CSV row per chunk -------------------------------------------

csvpath = joinpath(outdir, "octant.csv")
cols = ["t", "wall", "ham_l2", "ham_linf", "mom_l2", "mom_linf", "gauge_l2",
        "gauge_linf", "ham_l2_vol", "mom_l2_vol", "gauge_l2_vol", "state_l2",
        "state_linf"]
hole && append!(cols, ["err_l2", "err_linf", "residual", "drift", "err_l2_vol"])
for ℓ in 0:(nlev - 1), c in ("ham_l2", "ham_linf", "mom_l2", "mom_linf", "gauge_l2",
                             "gauge_linf")
    push!(cols, "$(c)_L$ℓ")
end
hole && append!(cols, ["err_l2_L$ℓ" for ℓ in 0:(nlev - 1)])
append!(cols, ["ham_l2_bnd", "mom_l2_bnd", "gauge_l2_bnd"])
hole && push!(cols, "err_l2_bnd")
# The shells about the hole: the evolved band inside the horizon, `[r_in, r_h)`,
# then `[s_k, s_{k+1})` outside it and one beyond the last radius. Named by
# their inner radius (`_in` for the band inside).
shell_r = !hole ? T[] : haskey(OPTIONS, "shells") ?
          [T(rat(x)) for x in split(OPTIONS["shells"], ',')] :
          vcat([r_equator], T[9 // 4, 3, 5, 8])
shell_names = hole ? vcat(["in"], [@sprintf("r%g", r) for r in shell_r]) : String[]
for nm in shell_names, c in ("ham_l2", "ham_linf", "mom_l2", "gauge_l2", "err_l2", "err_linf")
    push!(cols, "$(c)_$nm")
end
# With `split` (added 2026-10-20): each shell's trailing and leading half, then
# the analytic center, the found horizon's origin and its coordinate extents
# along x, y and z (each the diameter through the origin), and x/y.
const SPLIT_COLS = ("ham_l2", "ham_linf", "err_l2", "err_linf")
if split_on
    for nm in shell_names, side in ("trail", "lead"), c in SPLIT_COLS
        push!(cols, "$(c)_$(side)_$nm")
    end
    append!(cols, ["c_$axname", "hz_found", "hz_$axname", "hz_M_irr", "hz_J", "hz_dx",
                   "hz_dy", "hz_dz", "hz_ratio"])
end
isrank0 && !isfile(csvpath) && open(io -> println(io, join(cols, ',')), csvpath, "w")

# `mom` and `gauge` are reported as the largest component's norm: the
# components are related by the octant's symmetry, and one number per family
# is what a growth rate is fitted to.
mom2(n) = maximum(n.mom_l2)
mom∞(n) = maximum(n.mom_linf)
gauge2(n) = maximum(n.gauge_l2)
gauge∞(n) = maximum(n.gauge_linf)

# The outermost blocks: those whose extent reaches an outer face — a face
# that is neither periodic, nor reflecting, nor a rotating seam's (amended
# 2026-10-20: it read the high faces alone, which are all an octant's outer
# faces; the quadrant's low x face is one too).
const OUTER_LO = ntuple(d -> !case.periodic[d] && !case.reflecting[d][1] &&
                             d ∉ case.rotating, 3)
const OUTER_HI = ntuple(d -> !case.periodic[d] && !case.reflecting[d][2], 3)
function nearboundary(forest, k)
    ext = block_extent(T, forest, k)
    return any(d -> (OUTER_HI[d] && ext[d][2] ≥ forest.extents[d][2]) ||
                    (OUTER_LO[d] && ext[d][1] ≤ forest.extents[d][1]), 1:3)
end
const DIAG_ERR = TreeGeneralizedHarmonic.DIAG_ERR
const DIAG_MASK = TreeGeneralizedHarmonic.DIAG_MASK

# One half of a shell about the moving center (added 2026-10-20): the points
# of `lo ≤ |x − c| < hi` with `(x − c)·dir > 0` — `dir = −v̂` the trailing
# half, where the moving layer releases grid points, `+v̂` the leading one.
# Points on the plane through `c` are in neither. A `ShellMask` with a
# half-space: a snapshot of the center at the call's `t`, and `isbits`, so the
# monitors take it as a kernel argument on a device as they take theirs
# (CLAUDE.md, "The mask plumbing").
struct HalfShellMask{T}
    center::SVector{3,T}
    lo::T
    hi::T
    dir::SVector{3,T}
end
@inline function TreeGeneralizedHarmonic.is_evolved(m::HalfShellMask, x)
    d1 = x[1] - m.center[1]
    d2 = x[2] - m.center[2]
    d3 = x[3] - m.center[3]
    r² = d1 * d1 + d2 * d2 + d3 * d3
    s = d1 * m.dir[1] + d2 * m.dir[2] + d3 * m.dir[3]
    return (m.lo * m.lo ≤ r²) & (r² < m.hi * m.hi) & (s > 0)
end

# The row's point-weighted norms as one fold (added 2026-10-23, TreeAMR
# 0.2.6's `mesh_fold`). Until then each number was its own `mesh_mapreduce`,
# about 280 a row and 70 more with the split, and each shell and half-shell
# re-ran the monitors with its own mask only to select its points; the
# monitors' values at an evolved point do not depend on the mask, so one pass
# of each with the interior's mask and this fold give the same numbers (the
# sums to roundoff, the maxima exactly). The fold reads `diag`'s slots
# `DIAG_CGH:DIAG_ERR` — `C_a`, `ℋ`, `ℳ_i`, the mask and the error — and bins
# every owned point into: the whole mesh, its level, the boundary band (the
# block's `tag`), its shell about `c(t)` and, with the split, its trailing or
# leading half-shell. A bin holds the evolved point count and, for the nine
# other slots, the sum of squares and the peak, so that `mom` and `gauge` are
# still the largest component's norm. The shells select by `ShellMask`'s and
# `HalfShellMask`'s own arithmetic, so their points are the masked re-runs'
# where the band lies in the evolved region; a point inside the interior has
# a mask of zero and counts in no bin. On a `:fitted` hole whose offset surface
# is not the sphere `r_in` (a spinning one) that drops the layer points
# between the two, which the `ShellMask`-only re-run counted; at `a = 0` the
# two are the same sphere, and the CSV is the re-runs' to its ten digits.
const ROW_NV = DIAG_ERR - DIAG_CGH + 1      # slots 2:11 of `diag`
const ROW_MASK = DIAG_MASK - DIAG_CGH + 1    # the mask's position among them
const ROW_NQ = ROW_NV - 1                    # the slots with norms
@inline _row_slot(u, i) = i < ROW_MASK ? u[i] : u[i + 1]
rowbin(::Type{T}) where {T} = (n=Sum(zero(T)), s2=ntuple(_ -> Sum(zero(T)), Val(ROW_NQ)),
                               mx=ntuple(_ -> Max(zero(T)), Val(ROW_NQ)))
@inline rowbin(u) = (n=Sum(u[ROW_MASK]),
                     s2=ntuple(i -> Sum(_row_slot(u, i) * _row_slot(u, i)), Val(ROW_NQ)),
                     mx=ntuple(i -> Max(abs(_row_slot(u, i))), Val(ROW_NQ)))

struct RowFold{T,NL,NS,NH}
    center::SVector{3,T}
    edges::NTuple{NS,Tuple{T,T}}
    dir::SVector{3,T}                       # +v̂: the leading half
end

@inline function (f::RowFold{T,NL,NS,NH})(acc, p) where {T,NL,NS,NH}
    q = rowbin(p.u)
    ks = sum(ntuple(i -> is_evolved(ShellMask(f.center, f.edges[i]...), p.x) ? i : 0,
                    Val(NS)); init=0)
    kt = sum(ntuple(i -> is_evolved(HalfShellMask(f.center, f.edges[i]..., -f.dir), p.x) ?
                         i : 0, Val(NH)); init=0)
    kl = sum(ntuple(i -> is_evolved(HalfShellMask(f.center, f.edges[i]..., f.dir), p.x) ?
                         i : 0, Val(NH)); init=0)
    return (all=TreeAMR.combine(acc.all, q), lev=addto(acc.lev, p.level + 1, q),
            bnd=p.tag ? TreeAMR.combine(acc.bnd, q) : acc.bnd,
            shell=addto(acc.shell, ks, q), trail=addto(acc.trail, kt, q),
            lead=addto(acc.lead, kl, q))
end

function row_fold(p, center, edges, dir, ::Val{NL}, ::Val{NS}, ::Val{NH}) where {NL,NS,NH}
    bin = rowbin(T)
    init = (all=bin, lev=ntuple(_ -> bin, Val(NL)), bnd=bin, shell=ntuple(_ -> bin, Val(NS)),
            trail=ntuple(_ -> bin, Val(NH)), lead=ntuple(_ -> bin, Val(NH)))
    f = RowFold{T,NL,NS,NH}(center, edges, dir)
    return mesh_fold(f, TreeAMR.combine, init, p.diag; vars=DIAG_CGH:DIAG_ERR,
                     tag=k -> nearboundary(p.U.forest, k))
end

# A bin's norms, in the record's shape: the slots are `C_a` (1:4), `ℋ` (5),
# `ℳ_i` (6:8) and the error (9) among the bin's nine.
function binnorms(b)
    n = b.n.value
    l2(i) = n > 0 ? sqrt(b.s2[i].value / n) : zero(n)
    mx(i) = b.mx[i].value
    return (points=n, ham_l2=l2(5), ham_linf=mx(5),
            mom_l2=SVector(l2(6), l2(7), l2(8)), mom_linf=SVector(mx(6), mx(7), mx(8)),
            gauge_l2=SVector(l2(1), l2(2), l2(3), l2(4)),
            gauge_linf=SVector(mx(1), mx(2), mx(3), mx(4)), err_l2=l2(9), err_linf=mx(9))
end

# The state's point-weighted L2 and its L∞ over every owned value, one fold
# over the state vector (2026-10-23: two reductions and a count until then).
@inline statefold(acc, p) =
    TreeAMR.combine(acc, (n=Sum(one(p.h)), sq=Sum(sum(x -> x * x, p.u)),
                          mx=Max(maximum(abs, p.u))))

# The found horizon's coordinate extents (added 2026-10-20), as
# `test/hole_runs.jl`'s moving rows measure them: the finder's shape about its
# own origin, made real to degree 12, summed over `±n̂` for the three axes —
# the diameters through the origin. The finder sees the whole sphere, the
# quadrant's mirrors included (TreeAMR's `interpolate` mirrors its queries), so
# the shape is the full surface. `find` runs on rank 0 alone over MPI, so the
# table is rank 0's, which alone writes.
hz_extents = Dict{Float64,NTuple{3,Float64}}()
function find_with_extents(p, u, t; kw...)
    o = find_gh_horizon(p, u, t; kw...)
    try
        sh = SVector{169,Float64}(real_shape(o.hlm, o.grid, 12))
        hz_extents[Float64(t)] = ntuple(3) do d
            n = SVector{3,Float64}(d == 1, d == 2, d == 3)
            shape_series(sh, 12, n) + shape_series(sh, 12, -n)
        end
    catch err
        @warn "the horizon's extents" exception = err maxlog = 1
    end
    return o
end

t_wall0 = time()

# The SimWatch status file (added 2026-10-02): `simwatch.toml` in the run
# directory, rewritten at every chunk with everything the CSV row carries, the
# record's horizon, track, fit and step, and the progress; see
# `src/simwatch.jl` and SimWatch's `FORMAT.md`.
# The sparklines (added 2026-10-19): SimWatch's `[history]`, the last 100 chunks
# of the constraints, of `ℋ` and the error just outside the horizon, of the
# overall error and of the horizon's `M_irr` and `J`, each named like the key
# it tracks. A restart refills the window from the CSV, which lacks the
# horizon's numbers (`records.csv` is written at the end).
eq_shell = hole ? shell_names[2] : ""
history_csv = vcat(["constraints.ham_l2" => "ham_l2", "constraints.mom_l2" => "mom_l2",
                    "constraints.gauge_l2" => "gauge_l2"],
                   hole ? ["shells.$eq_shell.ham_l2" => "ham_l2_$eq_shell",
                           "shells.$eq_shell.err_l2" => "err_l2_$eq_shell",
                           "error.l2" => "err_l2"] : Pair{String,String}[],
                   # With `split` the CSV has the horizon's numbers too.
                   split_on ? ["horizon.M_irr" => "hz_M_irr", "horizon.J" => "hz_J",
                               "horizon.ratio" => "hz_ratio",
                               "shells.$eq_shell.trail.err_l2" => "err_l2_trail_$eq_shell",
                               "shells.$eq_shell.lead.err_l2" => "err_l2_lead_$eq_shell"] :
                              Pair{String,String}[])
sw = SimWatchWriter(outdir; name=opt("name", basename(outdir)),
                    code="TreeGeneralizedHarmonic", group=opt("group", nothing),
                    comm=comm,
                    # FORMAT.md allows at most 6: a moving hole's extent ratio
                    # takes ℳ's place.
                    summary=hole ? [(key="shells.$eq_shell.ham_l2", label="ℋ@horizon"),
                                    (key="constraints.ham_l2", label="ℋ"),
                                    (split_on ? [] :
                                     [(key="constraints.mom_l2", label="ℳ")])...,
                                    (key="error.l2", label="err"), "horizon.M_irr",
                                    "horizon.J",
                                    (split_on ? [(key="horizon.ratio",
                                                  label=rquadrant ? "z/x" : "x/y")] :
                                     [])...] :
                            [(key="constraints.ham_l2", label="ℋ"),
                             (key="constraints.mom_l2", label="ℳ"),
                             (key="constraints.gauge_l2", label="C")])
# Rank 0 alone writes the status file, so it alone refills the window.
if isrank0 && isfile(csvpath)
    try
        ts, ys = simwatch_csv_columns(csvpath, last.(history_csv))
        for i in max(1, length(ts) - 99):length(ts)
            simwatch_history!(sw, ts[i], Dict(k => y[i] for (k, y) in
                                                  zip(first.(history_csv), ys)))
        end
    catch err
        @warn "SimWatch: could not refill the history from $csvpath" exception = err
    end
end
time_unit = hole ? "M" : nothing
setup = Dict{String,Any}(
    "case" => casename, "octant" => symmetry, "L" => Float64(L), "N" => N, "levels" => nlev,
    "h_min" => Float64(minimum_spacing(T, forest0)), "q" => q, "cfl" => Float64(cfl),
    "eps_KO" => Float64(ε_KO), "noise" => amplitude, "options" => join(ARGS, " "))
if hole
    setup["a"] = Float64(a_spin)
    setup["M_irr_kerr"] = M_irr_kerr
    setup["interior"] = case.interior isa FittedSpec ? "fitted" : "damped"
    setup["gauge_source"] = case.gauge === nothing ? "sampled" : "algebraic"
    case.interior isa FittedSpec ? (setup["margin_cells"] = case.interior.margin) :
                                   (setup["r_0"] = Float64(case.interior.r_0);
                                    setup["r_1"] = Float64(case.interior.r_1))
    case.interior isa FittedSpec &&
        (setup["variant"] = string(TreeGeneralizedHarmonic.interior_variant(case.interior)))
    setup["points"] = sum(last, points_by_level)
    if moving_domain
        setup["v"] = Float64(v_hole)
        setup["x0"] = Float64(x0_hole)
        setup["path"] = Float64(path_x)
        setup["contraction_expected"] = contraction_expected
    end
    if rquadrant
        setup["axis"] = axname
        setup["ratio_expected"] = ratio_expected
    end
end
simwatch_update!(sw; force=true, status="starting", time_end=Float64(t_end),
                 time_unit=time_unit,
                 walltime_limit=walltime === nothing ? nothing : walltime,
                 message="compiling and building the initial data",
                 extra=Dict("setup" => setup))
steps_seen = Ref(0)
last_status = Ref{Any}((; extra=Dict{String,Any}("setup" => setup), black_holes=nothing,
                          images=nothing))

# The thumbnails (added 2026-10-19): `ℋ` and, for a hole, the error against the
# exact solution in each shell over the whole CSV (so across restarts), redrawn
# every chunk into `out/plots/`; for flat space the three constraints overall.
function draw_images()
    plotdir = joinpath(outdir, "plots")
    colours = join(("$(SW_PALETTE_NAMES[mod1(i, length(SW_PALETTE))]) $nm"
                    for (i, nm) in enumerate(shell_names)), ", ")
    specs = hole ?
            [("ham.svg", "ℋ by shell", ["ham_l2_$nm" for nm in shell_names],
              "Point-weighted L2 of ℋ against t/M, log scale: $colours."),
             ("err.svg", "error by shell", ["err_l2_$nm" for nm in shell_names],
              "Point-weighted L2 of the error against Kerr-Schild, log scale: $colours."),
             # The two halves of the shell just outside the horizon (2026-10-20).
             (split_on ? [("split.svg", "trailing and leading",
                           ["err_l2_trail_$eq_shell", "err_l2_lead_$eq_shell",
                            "ham_l2_trail_$eq_shell", "ham_l2_lead_$eq_shell"],
                           "Point-weighted L2 in the shell $eq_shell about c(t), " *
                           "log scale: orange the error on the trailing half, blue on " *
                           "the leading half, green ℋ trailing, purple ℋ leading.")] :
                         [])...] :
            [("constraints.svg", "constraints", ["ham_l2", "mom_l2", "gauge_l2"],
              "Point-weighted L2 against t, log scale: orange ℋ, blue ℳ, green C_a.")]
    images = Dict{String,Any}[]
    for (file, title, cols, desc) in specs
        ok = try
            ts, ys = simwatch_csv_columns(csvpath, cols)
            simwatch_plot(joinpath(plotdir, file), ts, ys)
        catch err
            @warn "SimWatch: could not read $csvpath" exception = err maxlog = 1
            false
        end
        ok && push!(images, Dict{String,Any}("file" => "plots/$file", "title" => title,
                                             "description" => desc))
    end
    return isempty(images) ? nothing : images
end

function observe(p, t, u, rec)
    # Each monitor once, over the evolved region; everything below is folds
    # over what they wrote (amended 2026-10-23: once per shell and half-shell
    # too, until then).
    gh_constraint!(p, u, t)
    adm_constraint!(p, u, t)
    hole && gh_error!(p, u, t; shell=horizon_shell(case, p.interior))
    vol = record_norms(p)
    # The shells about the hole: about its analytic center `c(t)`, the origin
    # for a hole at rest there (amended 2026-10-20: the origin always).
    c_t = hole ? SVector{3,T}(center_at(case.center, T(t))) : zero(SVector{3,T})
    edges = if hole
        r_in = case.interior isa FittedSpec ?
               shell_r[1] - case.interior.margin * minimum_spacing(T, p.U.forest) :
               case.interior.r_1
        vcat([r_in], shell_r, [T(Inf)])
    else
        T[]
    end
    nshell = max(length(edges) - 1, 0)
    bands = ntuple(i -> (edges[i], edges[i + 1]), nshell)
    dir = SVector{3,T}(on_axis(v_dir))
    rf = row_fold(p, c_t, bands, dir, Val(nlev), Val(nshell), Val(split_on ? nshell : 0))
    pts = binnorms(rf.all)
    st = mesh_fold(statefold, TreeAMR.combine,
                   (n=Sum(zero(T)), sq=Sum(zero(T)), mx=Max(zero(T))), p.U, u)
    s2 = sqrt(st.sq.value / (st.n.value * p.U.nvars))
    s∞ = st.mx.value
    bnd = binnorms(rf.bnd)
    row = Any[t, time() - t_wall0, pts.ham_l2, pts.ham_linf, mom2(pts), mom∞(pts),
              gauge2(pts), gauge∞(pts), vol.ham_l2, mom2(vol), gauge2(vol), s2, s∞]
    extra = Dict{String,Any}(
        "setup" => setup,
        "constraints" => Dict{String,Any}(
            "note" => "point-weighted over the evolved points" *
                      (hole ? " (outside the layer)" : ""),
            "ham_l2" => pts.ham_l2, "ham_linf" => pts.ham_linf,
            "mom_l2" => mom2(pts), "mom_linf" => mom∞(pts),
            "gauge_l2" => gauge2(pts), "gauge_linf" => gauge∞(pts),
            "volume_weighted" => Dict("ham_l2" => vol.ham_l2, "mom_l2" => mom2(vol),
                                      "gauge_l2" => gauge2(vol))),
        "state" => Dict("l2" => s2, "linf" => s∞))
    err = nothing
    if hole
        err = (pts.err_l2, pts.err_linf, vol.residual, vol.drift, vol.err_l2)
        append!(row, err)
        extra["error"] = Dict("l2" => pts.err_l2, "linf" => pts.err_linf,
                              "l2_volume" => vol.err_l2, "layer_residual" => vol.residual,
                              "drift_htt" => vol.drift)
    end
    # One set of columns per level `0:nlev-1` whether or not the mesh has
    # blocks there (amended 2026-10-20): a level the shells refine away
    # entirely — level 0 when the first radius reaches every root — is `NaN`,
    # where it shifted every later column by one.
    levels = Dict{String,Any}()
    present = Set(level.(p.U.forest.leaves))
    for ℓ in 0:(nlev - 1)
        ℓ in present || (append!(row, fill(NaN, 6)); continue)
        n = binnorms(rf.lev[ℓ + 1])
        append!(row, (n.ham_l2, n.ham_linf, mom2(n), mom∞(n), gauge2(n), gauge∞(n)))
        levels["L$ℓ"] = Dict{String,Any}(
            "points" => n.points, "ham_l2" => n.ham_l2, "ham_linf" => n.ham_linf,
            "mom_l2" => mom2(n), "gauge_l2" => gauge2(n))
    end
    if hole
        for ℓ in 0:(nlev - 1)
            ℓ in present || (push!(row, NaN); continue)
            el = binnorms(rf.lev[ℓ + 1]).err_l2
            push!(row, el)
            levels["L$ℓ"]["err_l2"] = el
        end
    end
    extra["levels"] = levels
    append!(row, (bnd.ham_l2, mom2(bnd), gauge2(bnd)))
    boundary = Dict{String,Any}("note" => "the blocks touching an outer face",
                                "ham_l2" => bnd.ham_l2, "mom_l2" => mom2(bnd),
                                "gauge_l2" => gauge2(bnd))
    if hole
        push!(row, bnd.err_l2)
        boundary["err_l2"] = bnd.err_l2
    end
    extra["boundary"] = boundary
    if hole
        shells = Dict{String,Any}()
        for i in 1:nshell
            n = binnorms(rf.shell[i])
            append!(row, (n.ham_l2, n.ham_linf, mom2(n), gauge2(n), n.err_l2, n.err_linf))
            shells[shell_names[i]] = Dict{String,Any}(
                "range" => @sprintf("[%g, %g)", edges[i], edges[i + 1]),
                "ham_l2" => n.ham_l2, "ham_linf" => n.ham_linf, "mom_l2" => mom2(n),
                "gauge_l2" => gauge2(n), "err_l2" => n.err_l2, "err_linf" => n.err_linf)
        end
        # Each shell's two halves (added 2026-10-20), into the CSV after all
        # the shells: `ℋ` and the error.
        if split_on
            for i in 1:nshell, (side, half) in (("trail", rf.trail), ("lead", rf.lead))
                n = binnorms(half[i])
                append!(row, (n.ham_l2, n.ham_linf, n.err_l2, n.err_linf))
                shells[shell_names[i]][side] = Dict{String,Any}(
                    "ham_l2" => n.ham_l2, "ham_linf" => n.ham_linf, "err_l2" => n.err_l2,
                    "err_linf" => n.err_linf)
            end
        end
        extra["shells"] = shells
    end
    # The horizon's position and extents (added 2026-10-20): `NaN` where no
    # find ran or this rank did not run it.
    # On the rotating quadrant (added 2026-10-21) along z, and `J` the
    # component along the spin, `J n̂_z` with `n̂` the finder's spin axis.
    hz_J = NaN
    if split_on
        ext = get(hz_extents, Float64(t), (NaN, NaN, NaN))
        found = rec.horizon_success === true
        found || (ext = (NaN, NaN, NaN))
        hz_ratio = ext[axis] / ext[across]
        hz_J = !rquadrant ? something(rec.J, NaN) :
               rec.J === nothing || rec.spin_axis === nothing ? NaN :
               rec.J * rec.spin_axis[3]
        append!(row, (Float64(center_at(case.center, T(t))[axis]),
                      rec.horizon_success === nothing ? NaN : Float64(found),
                      rec.origin === nothing ? NaN : Float64(rec.origin[axis]),
                      something(rec.M_irr, NaN), hz_J, ext..., hz_ratio))
    end
    isrank0 && open(csvpath, "a") do io
        println(io, join((x isa Real ? @sprintf("%.10g", x) : string(x) for x in row), ','))
    end
    isrank0 && @printf("t = %8.3f  wall %8.1f s   ℋ %.3e / %.3e   ℳ %.3e / %.3e   C %.3e / %.3e   |u| %.3e",
            t, time() - t_wall0, pts.ham_l2, pts.ham_linf, mom2(pts), mom∞(pts),
            gauge2(pts), gauge∞(pts), s2)
    err === nothing || !isrank0 || @printf("   err %.3e / %.3e  res %.3e  drift %.3e", err[1], err[2],
                               err[3], err[4])
    isrank0 && @printf("   bnd ℋ %.3e C %.3e", bnd.ham_l2, gauge2(bnd))
    split_on && isrank0 &&
        @printf("   c_%s %.4f  hz %s %.4f  %s/%s %.5f (%.5f)  trail|lead err %.3e|%.3e",
                axname, center_at(case.center, T(t))[axis], axname,
                rec.origin === nothing ? NaN : rec.origin[axis], axname,
                ("x", "y", "z")[across], hz_ratio,
                ratio_expected, extra["shells"][eq_shell]["trail"]["err_l2"],
                extra["shells"][eq_shell]["lead"]["err_l2"])
    split_on && rquadrant && isrank0 &&
        @printf("  M_irr %.6f (%.6f)  J_z %.6f (%s)", something(rec.M_irr, NaN),
                M_irr_kerr, hz_J, a_spin)
    isrank0 && println()
    flush(stdout)

    # The status file: the record's own numbers beside the ones above.
    steps_seen[] += rec.steps
    extra["step"] = Dict{String,Any}("dt" => rec.dt, "steps_this_chunk" => rec.steps,
                                     "speed_max" => rec.λ_end, "cfl" => rec.cfl,
                                     "blocks" => rec.nblocks, "finite" => rec.finite)
    bhs = nothing
    if rec.horizon_success !== nothing
        hz = Dict{String,Any}("found" => rec.horizon_success, "M_irr" => rec.M_irr,
                              "M_ch" => rec.M_ch, "J" => rec.J, "area" => rec.area,
                              "r_min" => rec.r_min, "r_mean" => rec.r_mean,
                              "r_max" => rec.r_max, "note" => rec.horizon_note)
        if split_on
            hz["extent"] = collect(ext)
            hz["ratio"] = hz_ratio
            hz["ratio_expected"] = ratio_expected
            rquadrant && (hz["J_along_spin"] = hz_J)
        end
        extra["horizon"] = hz
        spin = rec.J === nothing || rec.M_ch === nothing || rec.spin_axis === nothing ?
               nothing : collect(rec.J / rec.M_ch^2 .* rec.spin_axis)
        bhs = [(name="BH", irreducible_mass=rec.M_irr, mass=rec.M_ch, spin=spin,
                position=rec.origin === nothing ? nothing : collect(rec.origin),
                found=rec.horizon_success)]
    end
    rec.track_offset === nothing ||
        (extra["track"] = Dict("offset_cells" => rec.track_offset,
                               "r_min" => rec.track_r_min))
    rec.fit_valid === nothing ||
        (extra["fit"] = Dict("valid" => rec.fit_valid, "residual" => rec.fit_residual))
    rec.bounds_hits === nothing || (extra["bounds_hits"] = rec.bounds_hits)
    msg = if hole
        @sprintf("t = %.1f M: ℋ %.2e just outside the horizon, %.2e overall; M_irr − Kerr's = %s, J − a = %s",
                 t, extra["shells"][shell_names[2]]["ham_l2"], pts.ham_l2,
                 rec.M_irr === nothing ? "—" : @sprintf("%+.2e", rec.M_irr - M_irr_kerr),
                 rec.J === nothing ? "—" :
                 @sprintf("%+.2e", (rquadrant ? hz_J : rec.J) - a_spin)) *
        (split_on ? (rquadrant ?
                     @sprintf("; z/x = %.5f (expected %.5f)", hz_ratio, ratio_expected) :
                     @sprintf("; x/y = %.5f (√(1 − v²) = %.5f)", hz_ratio,
                              contraction_expected)) : "")
    else
        @sprintf("t = %.1f: ℋ %.2e, ℳ %.2e, C %.2e", t, pts.ham_l2, mom2(pts), gauge2(pts))
    end
    rec.finite || (msg = "non-finite values in the evolved region; " * msg)
    # Rank 0 alone writes the CSV the thumbnails are drawn from.
    imgs = isrank0 ? draw_images() : nothing
    hist = Dict{String,Any}("constraints.ham_l2" => pts.ham_l2,
                            "constraints.mom_l2" => mom2(pts),
                            "constraints.gauge_l2" => gauge2(pts))
    if hole
        hist["shells.$eq_shell.ham_l2"] = extra["shells"][eq_shell]["ham_l2"]
        hist["shells.$eq_shell.err_l2"] = extra["shells"][eq_shell]["err_l2"]
        hist["error.l2"] = extra["error"]["l2"]
        hist["horizon.M_irr"] = something(rec.M_irr, NaN)
        hist["horizon.J"] = something(rec.J, NaN)
    end
    if split_on
        hist["horizon.ratio"] = hz_ratio
        hist["shells.$eq_shell.trail.err_l2"] = extra["shells"][eq_shell]["trail"]["err_l2"]
        hist["shells.$eq_shell.lead.err_l2"] = extra["shells"][eq_shell]["lead"]["err_l2"]
    end
    last_status[] = (; extra=extra, black_holes=bhs, message=msg, images=imgs)
    simwatch_update!(sw; force=true, images=imgs, history=hist, iteration=steps_seen[], time=Float64(t),
                     time_end=Float64(t_end), time_unit=time_unit,
                     walltime_limit=walltime === nothing ? nothing : walltime,
                     checkpoint=ckdir === nothing ? nothing :
                                latest_checkpoint(joinpath(ckdir, "octant")),
                     message=msg, black_holes=bhs, extra=extra)
    return nothing
end

# --- the run ---------------------------------------------------------------------

ckw = if ckdir === nothing
    (;)
else
    pre = joinpath(ckdir, "octant")
    rf = latest_checkpoint(pre)
    rf === nothing || !isrank0 || println("restarting from $rf")
    (checkpoint_path_prefix=pre, checkpoint_every_chunks=ckevery, restart_file=rf,
     max_walltime_seconds=walltime, num_checkpoints_keep=ckkeep)
end
restarting = get(ckw, :restart_file, nothing) !== nothing

# A hole's frozen core is left alone: it is never evolved, and its stale data
# at the corner is not of definite parity (see `add_noise!`). The default
# `r_core` of a `:fitted` hole is the horizon's least radius less `5/4` (`3/4`
# at `a = 0`, as before).
core² = !hole ? zero(T) : case.interior isa FittedSpec ?
        (haskey(OPTIONS, "r_core") ? T(rat(OPTIONS["r_core"])) :
         r_hmin - 5 // 4)^2 : case.interior.r_0^2
fitkw = if hole && case.interior isa FittedSpec
    nL = case.interior.n_L > 0 ? case.interior.n_L :
         TreeGeneralizedHarmonic.layer_cells(q ÷ 2 + 1, T(4), one(T))
    depth = haskey(OPTIONS, "fit_depth") ? T(rat(OPTIONS["fit_depth"])) :
            nL * minimum_spacing(T, forest0)
    # Step 8′'s levers for a moving `:fitted` layer (added 2026-10-20), passed
    # only when asked for, as `test/hole_runs.jl`'s moving rows pass them;
    # `evolve!` applies them to the `:fitted` variant alone.
    levers = (; (haskey(OPTIONS, "trail_ramp") ?
                 (trail_ramp=T(rat(OPTIONS["trail_ramp"])),) : ())...,
              (haskey(OPTIONS, "target_rate") ?
               (target_rate=OPTIONS["target_rate"] in ("1", "true"),) : ())...)
    (fit_initial_depth=depth, fit_initial_cont=case.interior.fit_cont, levers...)
else
    any(k -> haskey(OPTIONS, k), ("trail_ramp", "target_rate")) &&
        error("trail_ramp= and target_rate= are levers of the tracked geometry " *
              "(interior = fitted)")
    (;)
end
haskey(OPTIONS, "rho") && (fitkw = (; fitkw..., ρ_max_fixed=T(rat(OPTIONS["rho"]))))
# The kernel's SIMD width (added 2026-10-22): `simd=1` is the scalar kernel a
# device runs, where a degenerate metric throws a `DomainError` instead of
# leaving a `NaN` on the lanes.
haskey(OPTIONS, "simd") && (fitkw = (; fitkw..., simd_width=parse(Int, OPTIONS["simd"])))
# The core about the hole's position at `t = 0`, the origin on an octant.
c_noise = hole ? Tuple(center_at(case.center, zero(T))) : (zero(T), zero(T), zero(T))
perturb = iszero(amplitude) ? nothing :
          U -> add_noise!(U, Xoshiro(seed); amplitude=amplitude,
                          exclude=hole ? (x -> sum(abs2, x .- c_noise) < core²) :
                                  nothing)
# The finder with the horizon's extents, where the CSV carries them.
findkw = split_on ? (find=find_with_extents,) : (;)
out = try
    evolve!(T, case; forest=restarting ? nothing : forest0, q=q,
            ops=Operators(prolongation=q + 2, restriction=q + 2), t_end=t_end,
            chunk=chunk, cfl=cfl, backend=backend, observer=observe,
            perturb=perturb, comm=comm, fitkw..., findkw..., ckw...)
catch err
    # Over MPI the status file's write is collective, and a failure may be
    # one rank's: write it serially from this rank, and take the job down.
    use_mpi || (ls = last_status[];
                simwatch_finish!(sw; status="failed", message=sprint(showerror, err),
                                 images=ls.images, time_end=Float64(t_end),
                                 time_unit=time_unit, black_holes=ls.black_holes,
                                 extra=ls.extra))
    abort_all(err)
    rethrow()
end
r = out.records[end]
let ls = last_status[]
    simwatch_finish!(sw; status=out.finished ? "finished" : "stopped", images=ls.images,
                     iteration=steps_seen[], time=Float64(out.t),
                     time_end=Float64(t_end), time_unit=time_unit,
                     message=(out.finished ? "finished: " : "stopped at the wall-time " *
                              "limit, to be continued: ") * get(ls, :message, ""),
                     checkpoint=ckdir === nothing ? nothing :
                                latest_checkpoint(joinpath(ckdir, "octant")),
                     black_holes=ls.black_holes, extra=ls.extra)
end
# The run's own record, the horizon's numbers and the track included.
fields = (:t, :steps, :err_l2, :err_linf, :residual, :drift, :gauge_l2, :horizon_success,
          :r_min, :r_mean, :r_max, :area, :M_irr, :J, :M_ch, :track_offset,
          :track_r_min, :fit_valid, :fit_residual, :bounds_hits, :variant)
isrank0 && open(joinpath(outdir, "records.csv"), "w") do io
    println(io, join(fields, ','))
    for rec in out.records
        println(io, join((string(get(rec, f, missing)) for f in fields), ','))
    end
end
isrank0 && @printf("done: t = %.4f, %d steps, dt = %.6g, finished = %s, wall %.1f s\n",
        out.t, out.nsteps, r.dt, out.finished, time() - t_wall0)
out.finished || exit(3)
