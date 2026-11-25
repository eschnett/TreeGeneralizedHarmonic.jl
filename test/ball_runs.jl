# Step B5's runs on the cube and the shells, one run per call: the Teukolsky
# wave through every glue and off the outer sphere, and random data on flat
# space. `CODE.md`, "The cube and the shells"; `PLAN.md`, step B5's acceptance.
#
#     julia --project=. --threads=4 test/ball_runs.jl case=teukolsky N=8 t_end=1 out=out/b5
#
# Options, `key=value`:
#
# - `case`: `teukolsky` (the wave; its error is against the wave) or `noise`
#   (uniform random data of size `noise` at every point the state holds, on flat
#   space; its "error" is the state itself);
# - the ball: `Lc` (1), `Rm` (5/2), `R1` (4), `roots` (`2,1,1`: the cube's per
#   side, the inner and the outer shells' radial), `N` (8), `q` (4);
# - the wave: `a` (`1e-6`), `width` (2); the noise: `noise` (`1e-6`), `seed` (1);
# - the run: `t_end` (1), `chunk` (`1/4`, the analysis cadence), `cfl` (`1/4`),
#   `eps_KO` (`1/2`), `gamma0` (1), `frozen` (0: the exact wave on the outer
#   sphere at every time; 1: the data at `t = 0`, `CODE.md`'s option 1);
# - output: `out` (a directory for `records.csv` and `simwatch.toml`), `group`
#   (the SimWatch group), `name`.
#
# Each chunk prints and records the time, the `√g`-weighted L2 and the L∞ of
# `|u − u_exact|` (over the twenty components) in the cube, the inner shells,
# the outer shells and the ball, and the wall time.

using Printf: @printf, @sprintf
using Random: Xoshiro
using SpacetimeMetrics: Minkowski
using TreeAMR
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: DIAG_ERR
const TMP = TreeGeneralizedHarmonic.TreeMultiPatch

const OPTS = Dict(split(a, "="; limit=2) for a in ARGS if occursin('=', a))
opt(k, d) = get(OPTS, k, d)
num(k, d) = (v = opt(k, nothing); v === nothing ? d :
             occursin('/', v) ? (let (a, b) = split(v, "/")
                                     parse(Int, a) // parse(Int, b)
                                 end) : parse(Float64, v))

const T = Float64
casename = opt("case", "teukolsky")
casename in ("teukolsky", "noise") ||
    error("case=$casename: the cases are teukolsky and noise")
N = parse(Int, opt("N", "8"))
q = parse(Int, opt("q", "4"))
roots = Tuple(parse.(Int, split(opt("roots", "2,1,1"), ",")))
Lc, Rm, R1 = T(num("Lc", 1)), T(num("Rm", 5 // 2)), T(num("R1", 4))
t_end = T(num("t_end", 1))
chunk = T(num("chunk", 1 // 4))
cfl = T(num("cfl", 1 // 4))
ε_KO = T(num("eps_KO", 1 // 2))
γ0 = T(num("gamma0", 1))
frozen = opt("frozen", "0") == "1"
outdir = opt("out", nothing)

layout = TMP.ThirteenPatchBall{T}(Lc, Rm, R1, roots)
bg = casename == "teukolsky" ?
     TeukolskyWave(T; amplitude=num("a", 1e-6), width=num("width", 2)) : Minkowski()
case = ball_case(T, bg; layout=layout, ε_KO=ε_KO, γ0=γ0, γ2=zero(T))
p = ball_problem(T, case, layout; N=N, q=q, frozen=frozen)
u = statevector(p.U)
TreeGeneralizedHarmonic.initial_state!(u, p, zero(T))
if casename == "noise"
    A = T(num("noise", 1e-6))
    rng = Xoshiro(parse(Int, opt("seed", "1")))
    u .= A .* (2 .* rand(rng, T, length(u)) .- 1)
end
dt0 = gh_dt(p, u; cfl=cfl)
npoints = statelength(p.U) ÷ p.U.nvars
@printf("ball %s: Lc=%g Rm=%g R1=%g roots=%s N=%d, %d blocks, %d points, dt=%.4g, frozen=%s\n",
        casename, Lc, Rm, R1, string(roots), N, nblocks(p.U), npoints, dt0, frozen)

regions = (("cube", 1:1), ("inner", 2:7), ("outer", 8:13), ("ball", 1:13))
header = "t," * join(["err_l2_$(r[1]),err_linf_$(r[1])" for r in regions], ",") *
         ",walltime"
sw = nothing
csv = nothing
if outdir !== nothing
    mkpath(outdir)
    csv = joinpath(outdir, "records.csv")
    open(io -> println(io, header), csv, "w")
    # Under SimLaunch the status goes where its job merges it into the run's.
    swdir = get(ENV, "SIMLAUNCH_SIMWATCH_DIR", outdir)
    sw = SimWatchWriter(swdir; name=opt("name", basename(abspath(outdir))),
                        code="TreeGeneralizedHarmonic", group=opt("group", nothing),
                        summary=[(key="error.ball_l2", label="err"),
                                 (key="error.cube_l2", label="cube"),
                                 (key="error.outer_l2", label="outer")])
    simwatch_update!(sw; force=true, status="starting", time_end=Float64(t_end),
                     message="building the ball and compiling",
                     extra=Dict("setup" => Dict("case" => casename, "N" => N,
                                                "roots" => collect(roots),
                                                "Lc" => Lc, "Rm" => Rm, "R1" => R1,
                                                "frozen" => frozen, "cfl" => cfl,
                                                "eps_KO" => ε_KO,
                                                "points" => npoints,
                                                "options" => join(ARGS, " "))))
end

function record(t, wall)
    ball_error!(p, u, t)
    rows = [ball_norms(p, DIAG_ERR; patches=r[2]) for r in regions]
    line = @sprintf("%.6g", t) * "," *
           join([@sprintf("%.6e,%.6e", r.l2, r.linf) for r in rows], ",") *
           @sprintf(",%.3f", wall)
    println(line)
    csv === nothing || open(io -> println(io, line), csv, "a")
    if sw !== nothing
        hist = Dict("error.$(regions[i][1])_l2" => rows[i].l2 for i in eachindex(rows))
        simwatch_update!(sw; time=Float64(t), time_end=Float64(t_end), history=hist)
    end
    return rows
end

println(header)
wall0 = time()
t = zero(T)
record(t, 0.0)
status = "finished"
try
    while t < t_end - chunk / 1000
        t1 = min(t + chunk, t_end)
        gh_solve(p, u, (t, t1); dt=dt0, alias_u0=true)
        global t = t1
        rows = record(t, time() - wall0)
        all(r -> isfinite(r.l2), rows) || error("the state is not finite at t = $t")
    end
catch err
    global status = "failed"
    sw === nothing || simwatch_finish!(sw; status="failed", message=sprint(showerror, err))
    rethrow()
end
sw === nothing || simwatch_finish!(sw; status=status, time=Float64(t),
                                   time_end=Float64(t_end))
