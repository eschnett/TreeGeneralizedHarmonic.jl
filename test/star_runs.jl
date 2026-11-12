# Constraint damping on a star (added 2026-11-11, for TreeGRRMHD's note GH-8):
# the Plummer star of `test/matter_tests.jl`, its matter held fixed (the
# anti-Cowling setting, GH-3), evolved for hundreds of `M` at a constant `γ0`
# and `γ2`, to see whether a mode grows and what sets its threshold.
#
# GH-8 reports, on TreeGRRMHD's TOV star (`R/M = 6.85`) with a frozen sampled
# `H_a` and `ε_KO = 1/2`, that `γ0 = 1/M` grows a mode that doubles every
# `25 M` independent of resolution, that `γ0 M ≤ 1/10` is flat, and that
# `γ2 = −1/2` suppresses it; and leaves open which scale sets the threshold
# (`M`, `R`, the light-crossing time) and the mechanism. The Plummer star is a
# one-parameter family in `R/M` (the equations are scale-free), so a scan of
# `γ0 M` at several `R/M` answers the first question: a threshold fixed in
# `γ0 M` is set by `M`, one fixed in `γ0 R` by `R`.
#
# A standalone script, not part of the suite. The box is `[−L, L]³` with
# Dirichlet data from the star itself at every face (the star is its own exact
# solution), a uniform mesh of `(2 roots)³`… `roots³` blocks of `N³` points,
# the star at rest (`λ = 0`), the gauge source sampled once and frozen, `q = 4`
# and RK4 at `cfl`. Every chunk writes one row of `out/series.csv`: the
# change of the state since `t = 0`, `Δu = u(t) − u(0)` — which for a static
# star is the error — as L∞ and point-weighted L2, over `h` and over `Π`
# separately, and at the centre; the gauge constraint `C_a` and the ADM
# constraints `ℋ − 16πρ`, `ℳ_i − 8πS_i`, volume-weighted L2 and L∞. And one row
# per variable to `out/vars.csv` (each of the twenty's L∞ of `Δu`), and one per
# radial bin to `out/profile.csv` (the L∞ of `Δh` and `ΔΠ` in shells of width
# `R/4`): which components grow, and where. `out/simwatch.toml` is the
# SimWatch status, with thumbnails in `out/plots/`. `test/star_study.jl`
# reads a directory of such runs and prints the tables. Every chunk also
# writes `out/ck.jls`, from which the same command continues after a requeue;
# a finished run leaves `out/done` and refuses to start again.
#
#     julia --project=. --threads=4 test/star_runs.jl gamma0M=1 t_end=20 out=out/star
#
# Options (`key=value`):
#
#   M=1/10 R=1         the star's mass and Plummer radius
#   L=2                the box's half-width, in the same units as R
#   N=16 roots=4       block size and root blocks per dimension: h = 2L/(roots N)
#   q=4 cfl=1/4        the order and RK4's Courant factor
#   eps=1/2            ε_KO
#   gamma0M=1          γ0 in units of 1/M (the constant profile)
#   gamma2=0           γ2
#   matter=1           0 evolves the star's data without its matter (a control)
#   t_end=500 chunk=5  in units of M
#   out=out/star name= group=   the output directory, SimWatch's name and group

using Printf
using Serialization: serialize, deserialize
using TreeAMR
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: ceilint

include(joinpath(@__DIR__, "plummer_star.jl"))
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
Mstar = rat(opt("M", "1/10"))
Rstar = rat(opt("R", "1"))
L = rat(opt("L", "2"))
N = parse(Int, opt("N", "16"))
roots = parse(Int, opt("roots", "4"))
q = parse(Int, opt("q", "4"))
cfl = rat(opt("cfl", "1/4"))
ε_KO = rat(opt("eps", "1/2"))
γ0M = rat(opt("gamma0M", "1"))
γ2 = rat(opt("gamma2", "0"))
with_matter = opt("matter", "1") in ("1", "true")
t_end = T(rat(opt("t_end", "500")) * Mstar)      # coordinate time
chunk = T(rat(opt("chunk", "5")) * Mstar)
outdir = mkpath(abspath(opt("out", "out/star")))
γ2 > -1 || error("γ2 must be > −1, got $γ2")

case = GHCase(T, PlummerStar{T}(T(Mstar), T(Rstar), zero(T));
              box=ntuple(_ -> (-T(L), T(L)), 3), periodic=(false, false, false),
              ε_KO=T(ε_KO), γ0=T(γ0M / Mstar), γ2=T(γ2))
forest = gh_forest(T, case; N=N, roots=roots)
G = q ÷ 2 + 1
fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
p = GHProblem(fs, GhostSchedule(fs, Operators(prolongation=q + 2, restriction=q + 2)),
              case; q=q)
matter = with_matter ? SampledMatter(T, forest, case.background) : nothing
fill_exact!(fs, case, zero(T))
u0 = statevector(fs)
gather!(u0, fs)
u = copy(u0)
h = minimum_spacing(T, forest)
dt0 = gh_dt(p, u0; cfl=T(cfl))
nsteps_chunk = max(1, ceilint(chunk / dt0))
dt = chunk / nsteps_chunk

# The analysis' own field set, `G = 0`, so that owned point `(i, j, k)` is
# stored at `(i, j, k)`; the radius of every point, once; the centre's point.
dfs = FieldSet{T}(forest, 20; G=0, centering=vertexcentered(3))
nb = nblocks(dfs)
rad = Array{T}(undef, N, N, N, nb)
for b in 1:nb, k in 1:N, j in 1:N, i in 1:N
    x = coordinates(dfs, b, (i, j, k))
    rad[i, j, k, b] = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
end
centre = argmin(rad)
binw = T(Rstar) / 4
nbins = ceilint(maximum(rad) / binw)

function analyse(du)
    scatter!(dfs, du)
    w = dfs.work
    vlinf = zeros(T, 20)
    s2h = zero(T)
    s2Π = zero(T)
    prof = zeros(T, nbins, 2)
    @inbounds for b in 1:nb, k in 1:N, j in 1:N, i in 1:N
        ib = min(nbins, 1 + floor(Int, rad[i, j, k, b] / binw))
        for v in 1:20
            a = abs(w[i, j, k, v, b])
            a = isfinite(a) ? a : T(Inf)
            vlinf[v] = max(vlinf[v], a)
            if v ≤ 10
                s2h += a * a
                prof[ib, 1] = max(prof[ib, 1], a)
            else
                s2Π += a * a
                prof[ib, 2] = max(prof[ib, 2], a)
            end
        end
    end
    npts = N^3 * nb
    c = centre
    return (; linf=maximum(vlinf), linf_h=maximum(vlinf[1:10]),
            linf_pi=maximum(vlinf[11:20]),
            l2=sqrt((s2h + s2Π) / (20npts)), l2_h=sqrt(s2h / (10npts)),
            l2_pi=sqrt(s2Π / (10npts)),
            centre_htt=w[c[1], c[2], c[3], 1, c[4]],
            centre_pitt=w[c[1], c[2], c[3], 11, c[4]], vlinf, prof)
end

function monitors(u, t)
    gh_constraint!(p, u, t)
    adm_constraint!(p, u, t; matter=matter)
    c = constraint_norms(p)
    l2(v) = sqrt(sum(abs2, v))
    return (; gauge_l2=l2(c.gauge_l2), gauge_linf=maximum(c.gauge_linf),
            ham_l2=c.ham_l2, ham_linf=c.ham_linf, mom_l2=l2(c.mom_l2),
            mom_linf=maximum(c.mom_linf))
end

const COLS = ["t", "t_M", "steps", "linf", "l2", "linf_h", "linf_pi", "l2_h", "l2_pi",
              "centre_htt", "centre_pitt", "gauge_l2", "gauge_linf", "ham_l2",
              "ham_linf", "mom_l2", "mom_linf", "wall"]
csvpath = joinpath(outdir, "series.csv")
varpath = joinpath(outdir, "vars.csv")
profpath = joinpath(outdir, "profile.csv")
fmt(x) = x isa Integer ? string(x) : @sprintf("%.10e", x)

# The restart (a run is hours long, and a Symmetry node failure requeues the
# job from the top): every chunk writes `(t, steps, wall, u)` to `ck.jls`
# atomically, and a run that finds one continues from it and keeps the rows of
# its CSVs up to that time — a row written after the checkpoint is dropped and
# written again. A run that has finished leaves `done` and refuses to restart.
ckpath = joinpath(outdir, "ck.jls")
donepath = joinpath(outdir, "done")
if isfile(donepath)
    @info "nothing left to run: $(readchomp(donepath))"
    exit(0)
end
function save_checkpoint(t, steps, wall)
    open(io -> serialize(io, (; t, steps, wall, u)), ckpath * ".tmp", "w")
    mv(ckpath * ".tmp", ckpath; force=true)
    return nothing
end
function trim_csv(path, tM)
    lines = readlines(path)
    keep = [first(lines); [l for l in lines[2:end]
                           if parse(Float64, first(split(l, ','; limit=2))) ≤ tM * (1 + 1e-12)]]
    open(io -> foreach(l -> println(io, l), keep), path * ".tmp", "w")
    mv(path * ".tmp", path; force=true)
    return nothing
end
restart = isfile(ckpath) ? open(deserialize, ckpath) : nothing
if restart === nothing
    for (path, header) in ((csvpath, join(COLS, ',')),
                           (varpath, "t_M," * join(("v$v" for v in 1:20), ',')),
                           (profpath, "t_M,r_lo,r_hi,linf_h,linf_pi"))
        open(io -> println(io, header), path, "w")
    end
else
    length(restart.u) == length(u) ||
        error("the checkpoint $ckpath is of another mesh: $(length(restart.u)) entries, " *
              "this run has $(length(u))")
    copyto!(u, restart.u)
    # series.csv's first column is `t`; the other two files' is `t/M`.
    trim_csv(csvpath, restart.t)
    trim_csv(varpath, restart.t / Mstar)
    trim_csv(profpath, restart.t / Mstar)
end

setup = Dict{String,Any}("M" => Float64(Mstar), "R" => Float64(Rstar),
                         "R_over_M" => Float64(Rstar / Mstar), "L" => Float64(L),
                         "N" => N, "roots" => roots, "h" => Float64(h),
                         "h_over_M" => Float64(h / Mstar), "q" => q,
                         "cfl" => Float64(cfl), "dt" => Float64(dt),
                         "eps_KO" => Float64(ε_KO), "gamma0M" => Float64(γ0M),
                         "gamma2" => Float64(γ2), "matter" => with_matter,
                         "options" => join(ARGS, " "))
open(joinpath(outdir, "setup.txt"), "w") do io
    for k in sort(collect(keys(setup)))
        println(io, k, " = ", setup[k])
    end
end
sw = SimWatchWriter(outdir; name=opt("name", basename(outdir)),
                    code="TreeGeneralizedHarmonic", group=opt("group", "star-damping"),
                    summary=[(key="change.linf", label="Δu"),
                             (key="constraints.ham_l2", label="ℋ"),
                             (key="constraints.gauge_l2", label="C")])
simwatch_update!(sw; force=true, status="starting", time=0.0,
                 time_end=Float64(t_end / Mstar), time_unit="M",
                 message="compiling and sampling the star's matter",
                 extra=Dict("setup" => setup))

function record(t, steps, wall)
    du = u .- u0
    a = analyse(du)
    m = monitors(u, t)
    row = (T(t), T(t / Mstar), steps, a.linf, a.l2, a.linf_h, a.linf_pi, a.l2_h,
           a.l2_pi, a.centre_htt, a.centre_pitt, m.gauge_l2, m.gauge_linf, m.ham_l2,
           m.ham_linf, m.mom_l2, m.mom_linf, T(wall))
    open(io -> println(io, join(fmt.(row), ',')), csvpath, "a")
    open(io -> println(io, fmt(T(t / Mstar)), ',', join(fmt.(a.vlinf), ',')), varpath,
         "a")
    open(profpath, "a") do io
        for ib in 1:nbins
            println(io, join(fmt.((T(t / Mstar), (ib - 1) * binw, ib * binw,
                                   a.prof[ib, 1], a.prof[ib, 2])), ','))
        end
    end
    return a, m
end

function status!(t, steps, a, m; final=nothing)
    plotdir = joinpath(outdir, "plots")
    imgs = Dict{String,Any}[]
    for (file, title, cols, desc) in
        (("change.svg", "change of the state", ["linf_h", "linf_pi"],
          "L∞ of u(t) − u(0) against t/M, log scale: orange h, blue Π."),
         ("constraints.svg", "constraints", ["ham_l2", "mom_l2", "gauge_l2"],
          "Volume-weighted L2 against t/M, log scale: orange ℋ, blue ℳ, green C_a."))
        ok = try
            ts, ys = simwatch_csv_columns(csvpath, ["t_M"; cols])
            simwatch_plot(joinpath(plotdir, file), ys[1], ys[2:end])
        catch err
            @warn "SimWatch: could not draw $file" exception = err maxlog = 1
            false
        end
        ok && push!(imgs, Dict{String,Any}("file" => "plots/$file", "title" => title,
                                           "description" => desc))
    end
    extra = Dict{String,Any}("setup" => setup,
                             "change" => Dict("linf" => a.linf, "l2" => a.l2,
                                              "linf_h" => a.linf_h,
                                              "linf_pi" => a.linf_pi),
                             "constraints" => Dict(string(k) => v for (k, v) in pairs(m)))
    msg = @sprintf("t = %.0f M: Δu %.2e (h %.2e, Π %.2e), ℋ %.2e, C %.2e", t / Mstar,
                   a.linf, a.linf_h, a.linf_pi, m.ham_l2, m.gauge_l2)
    hist = Dict{String,Any}("change.linf" => a.linf, "constraints.ham_l2" => m.ham_l2,
                            "constraints.gauge_l2" => m.gauge_l2)
    if final === nothing
        simwatch_update!(sw; force=true, images=imgs, history=hist, iteration=steps,
                         time=Float64(t / Mstar), time_end=Float64(t_end / Mstar),
                         time_unit="M", message=msg, extra=extra)
    else
        simwatch_finish!(sw; status=final[1], images=imgs, iteration=steps,
                         time=Float64(t / Mstar), time_end=Float64(t_end / Mstar),
                         time_unit="M", message=final[2] * msg, extra=extra)
    end
    return nothing
end

@info "star run" setup restarting_at_M = restart === nothing ? nothing : restart.t / Mstar
wall0 = restart === nothing ? 0.0 : restart.wall    # the wall time of earlier jobs
t0 = time()
t = restart === nothing ? zero(T) : T(restart.t)
steps = restart === nothing ? 0 : restart.steps
a, m = restart === nothing ? record(t, steps, 0.0) : (analyse(u .- u0), monitors(u, t))
status!(t, steps, a, m)
nchunks = ceilint(t_end / chunk - 1 // 1000)
first_chunk = round(Int, t / chunk) + 1
outcome = ("finished", "finished: ")
try
    for ic in first_chunk:nchunks
        global t, steps, a, m, outcome
        t1 = min(t_end, ic * chunk)
        n = max(1, round(Int, (t1 - t) / dt))
        gh_solve(p, u, (t, t1); dt=(t1 - t) / n, alias_u0=true, matter=matter)
        steps += n
        t = t1
        wall = wall0 + time() - t0
        save_checkpoint(t, steps, wall)
        a, m = record(t, steps, wall)
        @info @sprintf("t = %6.1f M  Δu %.3e  (h %.3e, Π %.3e)  ℋ %.3e  C %.3e  %.0f s",
                       t / Mstar, a.linf, a.linf_h, a.linf_pi, m.ham_l2, m.gauge_l2, wall)
        if !(isfinite(a.linf) && a.linf < 1)
            outcome = ("finished", @sprintf("stopped at t = %.1f M, Δu ≥ 1 or not finite: ",
                                            t / Mstar))
            break
        end
        ic < nchunks && status!(t, steps, a, m)
    end
catch err
    simwatch_update!(sw; force=true, status="failed", message=sprint(showerror, err))
    rethrow()
end
status!(t, steps, a, m; final=outcome)
open(io -> println(io, outcome[2], @sprintf("t = %.1f M", t / Mstar)), donepath, "w")
@info "done" t_M = t / Mstar wall = wall0 + time() - t0
