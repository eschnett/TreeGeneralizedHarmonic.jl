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
# From 2026-10-24 a spinning hole may move across its spin
# (`octant=bitant a=1/2 v=1/5 x0=-1`): a boosted Kerr-Schild hole spinning about
# z on the x axis, moving along it, on the bitant `[−L, L]² × [0, L]` with the
# mirror at `z = 0` alone, `c(t) = (x0 + v t, 0, 0)`. No half turn maps that
# hole onto itself, so the bitant is half the box and twice the quadrants'
# points. Its motion axis is `x`, as on the quadrant, and `hz_J` is `J`'s
# component along z, as on the rotating quadrant.
#
#     julia --project=. --threads=4 test/octant_runs.jl case=ks octant=quadrant v=1/5 x0=-1 L=8 N=16 roots=1 radii=4,3,2 r_0=5/8 r_1=23/16 t_end=1 chunk=1/2 cfl=1/2 amplitude=0 out=out/quad
#
#     julia --project=. --threads=4 test/octant_runs.jl case=ks octant=rotating_quadrant a=1/2 v=1/5 x0=-1 L=8 N=24 roots=1 radii=4,3,2 r_0=3/4 r_1=23/16 t_end=1 chunk=1/2 cfl=1/2 amplitude=0 out=out/spin
#
#     julia --project=. --threads=4 test/octant_runs.jl case=ks octant=bitant a=1/2 v=1/5 x0=-1 L=8 N=24 roots=1 radii=4,3,2 r_0=3/4 r_1=23/16 t_end=1 chunk=1/2 cfl=1/2 amplitude=0 out=out/bitant
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
#   octant=reflecting|rotating|quadrant|rotating_quadrant|bitant
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
#                                   root brick and the shells swept along z; or
#                                   (added 2026-10-24) the bitant [−L, L]² × [0, L],
#                                   the mirror at z = 0 alone, for a hole spinning
#                                   about z and moving along x (ks, any a): a
#                                   (2r, 2r, r) root brick and the shells swept
#                                   along x
#   v=0 x0=0                        the quadrants and the bitant only: the hole's
#                                   speed along the motion axis — x on the
#                                   quadrant and the bitant, z on the rotating
#                                   quadrant — and its position on that axis at
#                                   t = 0; it is at x0 + v t along it
#   path=v·t_end                    the same domains only: the length along the
#                                   motion axis of the segment from x0 the shells
#                                   are swept along (the capsule mesh, frozen; in M)
#   split=0|1                       also each shell's trailing ((x − c(t))·v̂ < 0)
#                                   and leading (> 0) half, and the horizon's
#                                   extents and their ratio along/across the motion
#                                   (x/y; z/x on the rotating quadrant) against
#                                   √(1 − v²) times the rest frame's (1 along x;
#                                   r₊/√(r₊² + a²) = √(r₊/2) along z) (default: on
#                                   for the quadrants and the bitant, v̂ = +x — +z
#                                   on the rotating quadrant — at v = 0). On the
#                                   rotating quadrant the centre and the found
#                                   origin are `c_z` and `hz_z`; there and on the
#                                   bitant `hz_J` is the Korzyński J's component
#                                   along z, J times the spin axis's z (the
#                                   magnitude elsewhere)
#   a=0                             the hole's spin along z, in M (ks only; a ≠ 0
#                                   needs octant=rotating, rotating_quadrant or
#                                   bitant, and
#                                   r_0 > a for :damped, which the ring of radius
#                                   a must be inside)
#   r_0=3/4 r_1=3/2 source=algebraic|sampled|composite
#                                   the hole's layer and gauge source (composite,
#                                   added in step B1: the tracked Kerr-Schild
#                                   source of one hole, algebraic's bits)
#   gauge=ks|dh                     (added in step B1) the hole's own Kerr-Schild
#                                   gauge, or the damped harmonic gauge rolled on
#                                   from it, H = f H_KS + (1 − f) H_DH with
#                                   f = exp(−((t − t0)/sigma_g)⁴) after t0:
#   sigma_g=10 t0=0                 the roll-on, in M
#   sigma_r=32 A_L1=1 A_L2=0 A_S=1  the damped harmonic gauge (SpECTRE's form):
#   e=2 (or e_L1= e_L2= e_S=)       the Gaussian about the origin and the
#                                   amplitudes (1/M) and exponents of
#                                   μ_L1 L + μ_L2 log(1/α) and μ_S
#   interior=damped|fitted|tracked  step 5's analytic layer, or step 8's fitted
#                                   target on the tracked horizon (margin m=8,
#                                   lmax_shape=4, lmax_fit=8, the finder every
#                                   chunk with the spin, default_bounds at 9/10),
#                                   or (added in step B2) the tracked analytic
#                                   target: step 5's sphere (r_0, r_1) about the
#                                   horizon track, relaxing toward the boosted
#                                   Kerr-Schild hole on the track's line, rebuilt
#                                   every piece, the finder every chunk
#   interior=layerfree              (added in step I1) no layer and no target: the
#                                   smoothed hole, the steered source with its frozen
#                                   correction ΔH, the range projection as the guard
#                                   (CODE.md, "A layer-free interior"); reflecting
#                                   octant, a = 0, at rest. Its knobs, the screen's
#                                   base point by default (SINGULARITY_HANDLING.md,
#                                   "A layer-free interior (step I1)"):
#   smooth=1                        the smoothing radius ε of SmoothedKerrSchild; 0 is
#                                   the analytic Kerr-Schild hole, its core filled by
#                                   the core rule at r_core (R1)
#   r_core=1/4                      smooth=0 only: the core rule's radius
#   correction=1                    the frozen correction ΔH (0: the algebraic source
#                                   alone, Π from C_a = 0)
#   steer=v|f|none p=1 mu_alpha=4   the lapse steering: v the volume-coupled target
#   mu_beta=0 kappa=1/3             α_s (√γ/√γ_s)^p, f the fixed κ α_s, none; its
#                                   rate μ_α, the shift's μ_β (β_* = 0), f's κ (in 1/M)
#   r_s0=1 r_s1=3/2                 the steering's ramp
#   start=plain|collapsed|rolled    the data: the smoothed hole, its lapse scaled by
#                                   1 − (1 − κ) ψ (Π from C_a = 0), or the plain data
#                                   with the steering rolled on from the unsteered
#                                   source over sigma_g (default 4) after t0
#   rho_a=0                         the anchor toward the smoothed state inside ε (1/M)
#   eps_in=                         the Kreiss–Oliger amplitude inside, rising from
#                                   the horizon to r_s0 (default: eps everywhere)
#   gamma0_near=1                   γ0 at the hole (1/M; the Gaussian's far value
#                                   stays 1/(10M))
#   alpha_min=1/1000 alpha_max=2    the guard's ranges (K_max in 1/M), and its gate:
#   lambda_min=1/1000 lambda_max=1000  a radius, or `everywhere` (no gate rule)
#   beta_max=3 K_max=100 r_gate=3/2
#                                   The CSV then adds, per shell — [0, ε), [ε, r_s1),
#                                   [r_s1, 2) and the shells= ones — |∂_t h|, the least
#                                   lapse and √γ/√γ_s, the largest λ(γ) and the
#                                   roughness, then C_a over the ramp [r_s0, r_s1), the
#                                   projection's hits by kind and the outermost hit,
#                                   the radii where α = 1/2 and 1/10 along x, M_irr and
#                                   r_min; `lapse.csv` holds α along x and the diagonal,
#                                   `hits.csv` the hits by radius in finest cells
#   prediction=quadratic|line       interior=tracked only (step B2): the track's
#                                   prediction, the quadratic of the last four
#                                   finds or step 8d's line
#   find_interval=0                 interior=tracked only: the longest interval
#                                   between two finds, in M (0: the chunk; the
#                                   cadence |a| Δt² ≤ G h/4 may ask for less)
#   target_mass=1 target_dv=0       interior=tracked only: the sensitivity
#   target_dx=0 seed_dv=0           study's perturbations — the target's mass
#                                   times target_mass, its velocity plus target_dv
#                                   and its center plus target_dx finest cells,
#                                   along the motion axis, the track unchanged;
#                                   and the seed track's velocity plus seed_dv
#                                   along it (the recovery run)
#   feed=own control_time=0         interior=tracked only (added in step B3): what
#                                   the gauge source rides on — own (the case's
#                                   analytic source), seed (the analytic line,
#                                   rebuilt every piece), tracks (the track's state
#                                   every piece, unfiltered) or controlled (the
#                                   controlled source line, a critically damped
#                                   loop of time control_time in M onto the track);
#                                   every feed but own needs source=composite;
#   control_law=pid                 feed=controlled only: the loop, pid (position,
#                                   velocity and the distance's integral; added in
#                                   step B3's review, the proposal for B4) or pd
#                                   (position and velocity, 7fea8eb's). The
#                                   CSV then ends with the track's and the source
#                                   line's center and velocity along the motion
#                                   axis and the line's acceleration
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
#   shell_center=analytic|track     (added in step B1) the shells and the split about
#                                   the analytic center c(t), or about the tracked
#                                   horizon's center (interior=fitted): under a gauge
#                                   that moves the hole off c(t) (gauge=dh) a shell
#                                   about c(t) cuts through the horizon
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
symmetry in ("reflecting", "rotating", "quadrant", "rotating_quadrant", "bitant") ||
    error("octant is reflecting, rotating, quadrant, rotating_quadrant or bitant, got $symmetry")
rotating = symmetry == "rotating"
quadrant = symmetry == "quadrant"
# The rotating quadrant (added 2026-10-21): the spinning hole moving along z.
rquadrant = symmetry == "rotating_quadrant"
# The bitant (added 2026-10-24): the spinning hole moving across its spin, along x.
bitant = symmetry == "bitant"
moving_domain = quadrant || rquadrant || bitant
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
          "only octant = quadrant or bitant (x) or rotating_quadrant (z) maps onto itself " *
          "(the $symmetry octant needs the hole at the origin, at rest)")
abs(v_hole) < 1 || error("v = $v_hole is the hole's speed in units of c, and must be below 1")
# The trailing/leading split and the horizon's extents (added 2026-10-20): new
# CSV columns, so on by default only for the quadrant, which has no older CSVs
# to match. `v̂` is `+x` for a hole at rest.
split_on = hole && opt("split", moving_domain ? "1" : "0") in ("1", "true")
shell_track = opt("shell_center", "analytic") == "track"
opt("shell_center", "analytic") in ("analytic", "track") ||
    error("shell_center is analytic or track, got $(opt("shell_center", ""))")
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
# The least radius of a hole moving at `v`: along the spin, `r₊` contracted
# along the boost, exactly `r₊` at rest; across it (from 2026-10-24) the pole's
# `r₊` or the equator's `√(r₊² + a²)` contracted, whichever is less.
r_hmin = axis == 3 ? r_plus * contraction_expected :
         min(r_plus, r_equator * contraction_expected)

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
# The layer-free interior's knobs (added in step I1).
# (`r_core` is also the `:fitted` noise's core, so it is not refused elsewhere.)
const LAYERFREE_KEYS = ("smooth", "correction", "steer", "p", "mu_alpha",
                        "mu_beta", "kappa", "r_s0", "r_s1", "start", "rho_a", "eps_in",
                        "gamma0_near", "alpha_min", "alpha_max", "lambda_min",
                        "lambda_max", "beta_max", "K_max", "r_gate")
layerfree = opt("interior", "damped") == "layerfree"
function layerfree_case(gs)
    symmetry == "reflecting" && iszero(a_spin) ||
        error("interior=layerfree is the static a = 0 hole on the reflecting octant")
    ε_s = rat(opt("smooth", "1"))
    r_s0, r_s1 = rat(opt("r_s0", "1")), rat(opt("r_s1", "3/2"))
    steer = opt("steer", "v")
    steer in ("v", "f", "none") || error("steer is v, f or none, got $steer")
    start = opt("start", "plain")
    start in ("plain", "collapsed", "rolled") ||
        error("start is plain, collapsed or rolled, got $start")
    κ = rat(opt("kappa", "1/3"))
    correction = opt("correction", "1") in ("1", "true")
    src = if iszero(ε_s)
        (steer == "none" && !correction && start == "plain") ||
            error("smooth=0 is the analytic hole with the algebraic source alone: " *
                  "steer=none correction=0 start=plain")
        gs
    else
        mk(st) = SteeredSource(T; M=1, ε=ε_s, r_s0=r_s0, r_s1=r_s1, steering=st,
                               p=st === :volume ? parse(Int, opt("p", "1")) : 1,
                               κ=st === :fixed ? κ : 1,
                               μ_α=rat(opt("mu_alpha", "4")), μ_β=rat(opt("mu_beta", "0")),
                               correction=correction)
        steered = mk(steer == "v" ? :volume : steer == "f" ? :fixed : :none)
        start == "rolled" ?
            RolledGaugeSource(mk(:none), steered; σ_g=T(rat(opt("sigma_g", "4"))),
                              t_0=T(rat(opt("t0", "0")))) : steered
    end
    everywhere = opt("r_gate", "3/2") == "everywhere"
    bd = layerfree_bounds(T; r_gate=everywhere ? 4 * L : rat(opt("r_gate", "3/2")),
                          α_min=rat(opt("alpha_min", "1/1000")),
                          α_max=rat(opt("alpha_max", "2")),
                          λ_min=rat(opt("lambda_min", "1/1000")),
                          λ_max=rat(opt("lambda_max", "1000")),
                          β_max=rat(opt("beta_max", "3")), K_max=rat(opt("K_max", "100")))
    lf = LayerFreeInterior(T; r_s0=r_s0, r_s1=r_s1,
                           r_core=iszero(ε_s) ? rat(opt("r_core", "1/4")) : 0,
                           ρ_a=rat(opt("rho_a", "0")), κ=start == "collapsed" ? κ : 1,
                           margin=parse(Int, opt("margin", "8")), gate_rule=!everywhere)
    γ0 = haskey(OPTIONS, "gamma0_near") ?
         (; γ0=GaussianDamping(T; near=T(rat(OPTIONS["gamma0_near"])), far=T(1 // 10),
                               width=T(3), center=HoleCenter(T, (0, 0, 0)))) : (;)
    c = kerr_schild_case(T; smoothing=ε_s, halfwidth=L, chunk=chunk, ε_KO=ε_KO, γ2=γ2,
                         octant=:reflecting, gauge_source=src, interior=lf,
                         horizon=opt("finder", "1") != "0" ?
                                 Horizon(T; every=1, N=12, spin=true) : nothing,
                         bounds=bd, γ0...)
    haskey(OPTIONS, "eps_in") &&
        (c = with_dissipation(c, horizon_dissipation(c; ε_in=T(rat(OPTIONS["eps_in"])))))
    return c
end

case = if hole
    src = opt("source", "algebraic")
    src in ("algebraic", "sampled", "composite") ||
        error("source is algebraic, sampled or composite, got $src")
    # The layer by step 8c's rule at this order: `ρ_max = 4/M` (the default),
    # `ρ_ramp = 1`, `r_1 − r_0 ≥ n_L h`; GHSO2's `ε_KO`, the Gaussian `γ0`.
    gs0 = src == "algebraic" ? :algebraic : src == "composite" ? :composite : nothing
    gauge_kind = opt("gauge", "ks")
    gauge_kind in ("ks", "dh") || error("gauge is ks or dh, got $gauge_kind")
    finder = opt("finder", "1") != "0" ? Horizon(T; every=1, N=12, spin=true) : nothing
    # The hole's motion, exact (the case converts it into `T` once), passed
    # only on the quadrants so that the octants' cases are built as before, and
    # the axis only on the rotating quadrant, so the quadrant's is as before.
    motion = quadrant || bitant ? (v=v_hole, x0=x0_hole) :
             rquadrant ? (v=v_hole, x0=x0_hole, axis=3) : (;)
    interior_kind = opt("interior", "damped")
    interior_kind in ("damped", "fitted", "tracked", "layerfree") ||
        error("interior is damped, fitted, tracked or layerfree, got $interior_kind")
    interior_kind == "layerfree" || !any(k -> haskey(OPTIONS, k), LAYERFREE_KEYS) ||
        error("$(join(LAYERFREE_KEYS, "=, "))= are the layer-free interior's " *
              "(interior = layerfree)")
    interior_kind == "tracked" || !any(k -> haskey(OPTIONS, k),
                                       ("prediction", "find_interval", "target_mass",
                                        "target_dv", "target_dx", "seed_dv", "feed",
                                        "control_time", "control_law")) ||
        error("prediction=, find_interval=, target_mass=, target_dv=, target_dx=, " *
              "seed_dv=, feed= and control_time= are the tracked analytic target's " *
              "(interior = tracked)")
    build_case(gs) = if interior_kind == "layerfree"
        layerfree_case(gs)
    elseif interior_kind == "fitted"
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
    elseif interior_kind == "tracked"
        # The tracked analytic target (added in step B2): the `:damped` sphere
        # of `r_0`, `r_1` about the track, its perturbations along the motion
        # axis, in finest cells for the center.
        haskey(OPTIONS, "variant") &&
            error("variant= is the tracked geometry's (interior = fitted); the " *
                  "tracked analytic target is :damped")
        pred = opt("prediction", "quadratic")
        pred in ("quadratic", "line") || error("prediction is quadratic or line, got $pred")
        hfine = T(L) / (roots * N) / 2^length(radii)
        along(s) = ntuple(d -> d == axis ? T(s) : zero(T), 3)
        spec = TrackedSpec(T; r_0=rat(opt("r_0", "3/4")), r_1=rat(opt("r_1", "3/2")),
                           ρ_ramp=1, margin=parse(Int, opt("margin", "8")),
                           quadratic=pred == "quadratic",
                           max_interval=T(rat(opt("find_interval", "0"))),
                           mass_factor=T(rat(opt("target_mass", "1"))),
                           velocity_offset=along(rat(opt("target_dv", "0"))),
                           center_offset=along(rat(opt("target_dx", "0")) * hfine),
                           seed_velocity_offset=along(rat(opt("seed_dv", "0"))),
                           source_feed=Symbol(opt("feed", "own")),
                           control_time=lowercase(opt("control_time", "0")) == "inf" ?
                                        T(Inf) : T(rat(opt("control_time", "0"))),
                           control_integral=opt("control_law", "pid") == "pid")
        opt("control_law", "pid") in ("pd", "pid") ||
            error("control_law is pd or pid, got $(opt("control_law", ""))")
        spec.source_feed == TreeGeneralizedHarmonic.FEED_OWN || gs === :composite ||
            error("feed=$(opt("feed", "own")) rebuilds the one-hole composite source " *
                  "every piece, and needs source=composite")
        kerr_schild_case(T; a=a_spin, halfwidth=L, chunk=chunk, ε_KO=ε_KO, γ2=γ2,
                         octant=Symbol(symmetry), motion..., gauge_source=gs,
                         interior=spec, horizon=Horizon(T; every=1, N=12, spin=true))
    else
        haskey(OPTIONS, "variant") &&
            error("variant= is the tracked geometry's (interior = fitted); the " *
                  "sphere here is :damped")
        kerr_schild_case(T; a=a_spin, halfwidth=L, r_0=rat(opt("r_0", "3/4")),
                         r_1=rat(opt("r_1", "3/2")), ρ_ramp=1, chunk=chunk, ε_KO=ε_KO,
                         γ2=γ2, octant=Symbol(symmetry), motion..., gauge_source=gs,
                         margin=parse(Int, opt("margin", "8")), horizon=finder)
    end
    c = build_case(gs0)
    # The damped harmonic gauge (added in step B1), rolled on from the hole's own
    # algebraic source: the case is built once for that source and once more
    # with the roll-on, which `GHCase` checks at `t = 0`, where it is the
    # hole's own source.
    if gauge_kind == "dh"
        layerfree && error("gauge=dh is not a layer-free knob")
        c.gauge === nothing &&
            error("gauge=dh rolls on from the hole's algebraic source; source=sampled has none")
        e_all = parse(Int, opt("e", "2"))
        dh = DampedHarmonicSource(T; σ_r=T(rat(opt("sigma_r", "32"))),
                                  A_L1=T(rat(opt("A_L1", "1"))),
                                  A_L2=T(rat(opt("A_L2", "0"))),
                                  A_S=T(rat(opt("A_S", "1"))),
                                  e_L1=parse(Int, opt("e_L1", string(e_all))),
                                  e_L2=parse(Int, opt("e_L2", string(e_all))),
                                  e_S=parse(Int, opt("e_S", string(e_all))))
        c = build_case(RolledGaugeSource(c.gauge, dh; σ_g=T(rat(opt("sigma_g", "10"))),
                                         t_0=T(rat(opt("t0", "0")))))
    end
    c
else
    minkowski_octant_case(T; L=L, ε_KO=ε_KO, γ0=γ0, γ2=γ2, chunk=chunk,
                          rotating=rotating)
end
# The quadrant's box is twice as long in x, and its shells are swept along the
# hole's path from `(x0, 0, 0)` (added 2026-10-20); the rotating quadrant's is
# twice as long in z, swept from `(0, 0, x0)` (added 2026-10-21); the bitant's
# twice as long in x and y, swept from `(x0, 0, 0)` (added 2026-10-24); the
# octants' forest is the one it was.
on_axis(s) = ntuple(d -> d == axis ? s : 0, 3)
forest0 = moving_domain ?
          hole_forest(T, case; N=N,
                      roots=ntuple(d -> d == axis || (bitant && d ≤ 2) ? 2 * roots : roots, 3),
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
(quadrant || bitant) && isrank0 && @printf("        moving: v = %s along x from x0 = %s (at x = %s at t_end), shells swept along %s; x/y contraction √(1 − v²) = %.6f\n",
                                v_hole, x0_hole, x0_hole + v_hole * t_end, path_x,
                                contraction_expected)
rquadrant && isrank0 && @printf("        moving: v = %s along z from z0 = %s (at z = %s at t_end), shells swept along %s; z/x extent √(1 − v²) · r₊/√(r₊² + a²) = %.6f · %.6f = %.6f\n",
                                 v_hole, x0_hole, x0_hole + v_hole * t_end, path_x,
                                 contraction_expected, r_plus / r_equator, ratio_expected)
hole && isrank0 && @printf("        hole: a = %s, %s, gauge source %s\n", a_spin,
                case.interior isa FittedSpec ?
                "fitted target, tracked, margin $(case.interior.margin)" :
                case.interior isa TrackedSpec ?
                "tracked analytic target, r_0 = $(case.interior.r_0), r_1 = " *
                "$(case.interior.r_1), $(case.interior.quadratic ? "quadratic" : "line") " *
                "prediction, find interval ≤ " *
                "$(iszero(case.interior.max_interval) ? "the chunk" : case.interior.max_interval), " *
                "target mass × $(case.interior.mass_factor), velocity + " *
                "$(Tuple(case.interior.velocity_offset)), center + " *
                "$(Tuple(case.interior.center_offset)), seed velocity + " *
                "$(Tuple(case.interior.seed_velocity_offset))" :
                case.interior isa LayerFreeInterior ?
                "layer-free interior, ε = $(case.interior.ε), r_core = " *
                "$(case.interior.r_core), steering ramp [$(case.interior.r_s0), " *
                "$(case.interior.r_s1)), data κ = $(case.interior.κ), anchor ρ_a = " *
                "$(case.interior.ρ_a), $(case.bounds), ε_KO $(case.ε_KO), γ0 $(case.γ0)" :
                "damped layer, r_0 = $(case.interior.r_0), r_1 = $(case.interior.r_1)",
                case.gauge === nothing ? "sampled" :
                layerfree ? repr(case.gauge) :
                case.gauge isa RolledGaugeSource ?
                "$(opt("source", "algebraic")), rolled on to the damped harmonic gauge " *
                "(t0 = $(case.gauge.t_0), σ_g = $(case.gauge.σ_g), σ_r = " *
                "$(case.gauge.to.σ_r), A = ($(case.gauge.to.A_L1), $(case.gauge.to.A_L2), " *
                "$(case.gauge.to.A_S)), e = ($(case.gauge.to.e_L1), $(case.gauge.to.e_L2), " *
                "$(case.gauge.to.e_S)))" : opt("source", "algebraic"))
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
# The layer-free interior's (step I1): from the center, `[0, ε)` (the smoothed
# ball, or the core), `[ε, r_s1)` and `[r_s1, s_1)`, then the `shells=` ones;
# every shell is named by its inner radius.
lf_inner = !(hole && layerfree) ? T[] :
           T[0, TreeGeneralizedHarmonic.layerfree_ball(case.interior), case.interior.r_s1]
shell_names = !hole ? String[] :
              layerfree ? [@sprintf("r%g", r) for r in vcat(lf_inner, shell_r)] :
              vcat(["in"], [@sprintf("r%g", r) for r in shell_r])
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
# A fed source (added in step B3): the track's and the source line's center and
# velocity along the motion axis, and the line's acceleration.
fed_cols = hole && case.interior isa TrackedSpec &&
           case.interior.source_feed != TreeGeneralizedHarmonic.FEED_OWN
fed_cols && append!(cols, ["track_c", "track_v", "src_c", "src_v", "src_a", "err_l2_trk",
                            "err_linf_trk", "res_trk"])
# The layer-free interior's (added in step I1), last: per shell the stationarity
# `|∂_t h|`, the least lapse and `√γ/√γ_s`, the largest `λ(γ)` and the roughness;
# then `ℋ` and `C_a` over the steering's ramp `[r_s0, r_s1)`, the chunk's hits by
# kind, the outermost hit's radius and its depth inside the horizon in finest
# cells, the radii where `α = 1/2` and `1/10` along `x`, and the finder's `M_irr`
# and `r_min`.
const LF_SHELL_COLS = ("stat_l2", "stat_linf", "alpha_min", "vol_min", "lambda_max", "rough")
if hole && layerfree
    for nm in shell_names, c in LF_SHELL_COLS
        push!(cols, "$(c)_$nm")
    end
    append!(cols, ["ham_l2_ramp", "gauge_l2_ramp", "gauge_linf_ramp", "hits",
                   "hits_nonfinite", "hits_spectrum", "hits_shift", "hits_lapse",
                   "hits_momentum", "hit_r_max", "hit_cells_inside", "r_alpha_half",
                   "r_alpha_tenth", "M_irr", "r_min"])
end
# The lapse along `x` and along the diagonal, `ax<k>` and `ad<k>` at `k` finest
# cells from the center to `3 M` (`lapse.csv`), and the hits by radius in finest cells (`hits.csv`).
lapse_path = joinpath(outdir, "lapse.csv")
hits_path = joinpath(outdir, "hits.csv")
if hole && layerfree && isrank0
    hf = minimum_spacing(T, forest0)
    isfile(lapse_path) || open(lapse_path, "w") do io
        K = round(Int, 3 / hf)
        println(io, join(vcat(["t"], ["ax$k" for k in 0:K], ["ad$k" for k in 0:K]), ','))
    end
    isfile(hits_path) || open(hits_path, "w") do io
        println(io, join(vcat(["t"], ["c$(k - 1)" for k in
                                      1:TreeGeneralizedHarmonic.BOUNDS_HIST_BINS]), ','))
    end
end
# The layer-free diagnostics' field set and the right-hand side's vector, made
# once per mesh (the screen's mesh is frozen).
const LF_AUX = Ref{Any}(nothing)

# The lapse at the points `xs` (ghosts filled), and the outermost radius along a
# ray where it is `level`, linearly between the samples.
function lapse_at(p, xs)
    vals = gh_interpolate(p.U, xs; q=q)
    # The signed lapse of `state_validity`: a collapsed point reads, not throws.
    return [state_validity(SVector{10,T}(ntuple(i -> v[i], 10)),
                           SVector{10,T}(ntuple(i -> v[10 + i], 10)))[2] for v in vals]
end
function crossing(s, α, level)
    k = findlast(a -> a ≤ level, α)
    k === nothing && return NaN
    k == length(α) && return s[end]
    return s[k] + (level - α[k]) / (α[k + 1] - α[k]) * (s[k + 1] - s[k])
end

# The layer-free row's columns (step I1), appended after every other.
function layerfree_columns!(row, extra, p, t, u, rec, c_t, dir)
    li = p.interior
    if LF_AUX[] === nothing || LF_AUX[].forest !== p.U.forest
        LF_AUX[] = TreeGeneralizedHarmonic.layerfree_aux(p)
    end
    edges = vcat(lf_inner, shell_r, [T(Inf)])
    rows = layerfree_diagnostics!(LF_AUX[], similar(u), p, u, t; edges=edges)
    lf = Dict{String,Any}()
    for (i, nm) in enumerate(shell_names)
        r = rows[i]
        append!(row, (r.stat_l2, r.stat_linf, r.α_min, r.vol_min, r.λ_max, r.rough))
        lf[nm] = Dict("stat_l2" => r.stat_l2, "stat_linf" => r.stat_linf,
                      "alpha_min" => r.α_min, "vol_min" => r.vol_min,
                      "lambda_max" => r.λ_max, "rough" => r.rough)
    end
    # `ℋ` and `C_a` over the steering's ramp: the row's fold with one band. The
    # monitors' slots are still the row's (the diagnostics write their own set).
    rr = binnorms(row_fold(p, c_t, ((li.r_s0, li.r_s1),), dir, Val(nlev), Val(1),
                           Val(0)).shell[1])
    append!(row, (rr.ham_l2, gauge2(rr), gauge∞(rr)))
    k = get(rec, :bounds_kinds, nothing)
    hits = something(rec.bounds_hits, 0)
    append!(row, (hits, (k === nothing ? (0, 0, 0, 0, 0) : Tuple(k))...))
    hf = minimum_spacing(T, p.U.forest)
    r_hit = something(rec.bounds_r_max, -1.0)
    append!(row, (r_hit, r_hit < 0 ? NaN : (r_hmin - r_hit) / hf))
    # The lapse along x and the diagonal; the ghosts are the diagnostics'.
    ss = [k * hf for k in 0:round(Int, 3 / hf)]
    ax = lapse_at(p, [SVector{3,T}(s, 0, 0) for s in ss])
    ad = lapse_at(p, [SVector{3,T}(s, s, s) / sqrt(T(3)) for s in ss])
    r_half, r_tenth = crossing(ss, ax, T(1 // 2)), crossing(ss, ax, T(1 // 10))
    append!(row, (r_half, r_tenth, something(rec.M_irr, NaN), something(rec.r_min, NaN)))
    isrank0 && open(io -> println(io, join((@sprintf("%.10g", x) for x in
                                            vcat([t], ax, ad)), ',')), lapse_path, "a")
    hist = get(rec, :bounds_hist, nothing)
    isrank0 && hist !== nothing &&
        open(io -> println(io, join(vcat([@sprintf("%.10g", t)], string.(collect(hist))),
                                    ',')), hits_path, "a")
    extra["layerfree"] = Dict{String,Any}(
        "shells" => lf, "ramp" => Dict("ham_l2" => rr.ham_l2, "gauge_l2" => gauge2(rr)),
        "hits" => hits, "hits_by_kind" => k === nothing ? nothing : Dict(pairs(k)),
        "hit_r_max" => r_hit, "r_alpha_half" => r_half, "r_alpha_tenth" => r_tenth)
    return nothing
end
# The error against the hole on the track's line, and the layer's residual
# against its own target (added in step B3): a fed source moves the hole off
# the analytic line, so `err_*` and `residual` measure the move as well. The
# error kernel again, with the piece's target as the background it measures
# against; it overwrites `diag`'s error slots, so it runs after every fold that
# reads them.
function target_norms(p, u, t)
    TG = TreeGeneralizedHarmonic
    int = p.interior
    tgt = TG.layer_target(int, p.case.background)
    scatter!(p.U, u)
    map_blocks!(TG.gh_error_kernel!, p.U, p.diag.work, p.U.work, p.origins, p.spacings,
                tgt, int, interior_mask(int, T(t)), T(t), zero(T), -one(T),
                TG.target_work(p.target), p.t_target, p.valG)
    return TG._error_rows(TG.diag_norms(p, TG.DIAG_MASK:TG.DIAG_DRIFT))
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
eq_shell = hole ? @sprintf("r%g", shell_r[1]) : ""
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
    setup["interior"] = case.interior isa FittedSpec ? "fitted" :
                        case.interior isa TrackedSpec ? "tracked" :
                        case.interior isa LayerFreeInterior ? "layerfree" : "damped"
    if case.interior isa TrackedSpec
        sp = case.interior
        setup["tracked"] = Dict("prediction" => sp.quadratic ? "quadratic" : "line",
                                "max_interval" => Float64(sp.max_interval),
                                "mass_factor" => Float64(sp.mass_factor),
                                "velocity_offset" => Float64.(collect(sp.velocity_offset)),
                                "center_offset" => Float64.(collect(sp.center_offset)),
                                "seed_velocity_offset" =>
                                    Float64.(collect(sp.seed_velocity_offset)))
    end
    setup["gauge_source"] = case.gauge === nothing ? "sampled" :
                            layerfree ? repr(case.gauge) : opt("source", "algebraic")
    if case.gauge isa RolledGaugeSource && case.gauge.to isa DampedHarmonicSource
        dh = case.gauge.to
        setup["gauge"] = "damped harmonic, rolled on"
        setup["rollon"] = Dict("t0" => Float64(case.gauge.t_0),
                               "sigma_g" => Float64(case.gauge.σ_g))
        setup["damped_harmonic"] = Dict("sigma_r" => Float64(dh.σ_r),
                                        "A" => Float64.([dh.A_L1, dh.A_L2, dh.A_S]),
                                        "e" => [dh.e_L1, dh.e_L2, dh.e_S])
    else
        setup["gauge"] = "Kerr-Schild"
    end
    setup["shell_center"] = shell_track ? "track" : "analytic"
    if case.interior isa LayerFreeInterior
        li = case.interior
        setup["layerfree"] = Dict("eps" => Float64(li.ε), "r_core" => Float64(li.r_core),
                                  "r_s0" => Float64(li.r_s0), "r_s1" => Float64(li.r_s1),
                                  "kappa_data" => Float64(li.κ), "rho_a" => Float64(li.ρ_a),
                                  "gate_rule" => li.gate_rule,
                                  "bounds" => repr(case.bounds),
                                  "eps_KO" => repr(case.ε_KO), "gamma0" => repr(case.γ0))
    elseif case.interior isa FittedSpec
        setup["margin_cells"] = case.interior.margin
    else
        setup["r_0"] = Float64(case.interior.r_0)
        setup["r_1"] = Float64(case.interior.r_1)
    end
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
    # About the tracked horizon's center with `shell_center=track` (added in step
    # B1), the analytic one otherwise.
    # For the tracked analytic target (added in step B2) the track's own
    # center, which the row carries — the sphere's is the target's, and a
    # perturbed target's center is not the hole's.
    c_t = !hole ? zero(SVector{3,T}) :
          shell_track && p.interior isa FittedInterior ?
          SVector{3,T}(center_at(p.interior.center, T(t))) :
          shell_track && case.interior isa TrackedSpec && rec.track_center !== nothing ?
          SVector{3,T}(rec.track_center) :
          SVector{3,T}(center_at(case.center, T(t)))
    edges = if hole && layerfree
        vcat(lf_inner, shell_r, [T(Inf)])
    elseif hole
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
        hz_J = !(rquadrant || bitant) ? something(rec.J, NaN) :
               rec.J === nothing || rec.spin_axis === nothing ? NaN :
               rec.J * rec.spin_axis[3]
        append!(row, (Float64(center_at(case.center, T(t))[axis]),
                      rec.horizon_success === nothing ? NaN : Float64(found),
                      rec.origin === nothing ? NaN : Float64(rec.origin[axis]),
                      something(rec.M_irr, NaN), hz_J, ext..., hz_ratio))
    end
    fed_note = ""
    if fed_cols
        get_ax(x) = x === nothing ? NaN : Float64(x[axis])
        append!(row, (get_ax(get(rec, :track_center, nothing)),
                      get_ax(get(rec, :track_velocity, nothing)),
                      get_ax(rec.source_center), get_ax(rec.source_velocity),
                      get_ax(rec.source_acceleration)))
        tn = target_norms(p, u, t)
        append!(row, (tn.err_l2, tn.err_linf, tn.residual))
        fed_note = @sprintf("   track %.6f/%.6f  src %.6f/%.6f/%.3e  err_trk %.3e  res_trk %.3e",
                           get_ax(get(rec, :track_center, nothing)),
                           get_ax(get(rec, :track_velocity, nothing)),
                           get_ax(rec.source_center), get_ax(rec.source_velocity),
                           get_ax(rec.source_acceleration), tn.err_l2, tn.residual)
    end
    hole && layerfree && layerfree_columns!(row, extra, p, t, u, rec, c_t, dir)
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
    split_on && (rquadrant || bitant) && isrank0 &&
        @printf("  M_irr %.6f (%.6f)  J_z %.6f (%s)", something(rec.M_irr, NaN),
                M_irr_kerr, hz_J, a_spin)
    isrank0 && print(fed_note)
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
            (rquadrant || bitant) && (hz["J_along_spin"] = hz_J)
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
                 @sprintf("%+.2e", (rquadrant || bitant ? hz_J : rec.J) - a_spin)) *
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
         r_hmin - 5 // 4)^2 :
        case.interior isa LayerFreeInterior ? case.interior.r_core^2 : case.interior.r_0^2
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
          :track_r_min, :fit_valid, :fit_residual, :bounds_hits, :variant,
          # The track's prediction and cadence (added in step B2): the row's
          # prediction error and the chunk's largest, in finest cells, the finds
          # the chunk made, the find interval and the predicted acceleration.
          :track_prediction, :track_prediction_max, :track_finds, :track_interval,
          :track_acceleration)
isrank0 && open(joinpath(outdir, "records.csv"), "w") do io
    # And the track's center and velocity along the motion axis (step B2).
    println(io, join(fields, ','), ",track_c,track_v")
    for rec in out.records
        tc = get(rec, :track_center, nothing)
        tv = get(rec, :track_velocity, nothing)
        println(io, join((string(get(rec, f, missing)) for f in fields), ','), ',',
                tc === nothing ? missing : tc[axis], ',',
                tv === nothing ? missing : tv[axis])
    end
end
isrank0 && @printf("done: t = %.4f, %d steps, dt = %.6g, finished = %s, wall %.1f s\n",
        out.t, out.nsteps, r.dt, out.finished, time() - t_wall0)
out.finished || exit(3)
