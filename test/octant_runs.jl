# Runs on the octant (added 2026-10-02): white noise on flat space, on the
# nested hierarchy a single black hole at the origin will run on, and that
# black hole — Kerr-Schild `M = 1`, `a = 0`, with the damping layer and the
# algebraic gauge source, with or without the same noise.
#
# A standalone script, not part of the suite. The box is `[0, L]³` with
# reflecting faces through the origin and Minkowski Dirichlet data
# (`h = Π = 0`, constant in time) at the three outer faces; the mesh is
# `hole_forest(; shape = :cube)` about the origin, `h = L/(roots N)` at the
# outer boundary and halved inside each cube `[0, R]³` of `radii`. The initial
# data are `add_noise!`'s uniform noise on every owned point and variable,
# projected onto the parity on the wall planes. Each chunk the observer runs
# both monitors and writes one CSV row: the ADM constraints `ℋ`, `ℳ_i` and
# the gauge constraints `C_a`, separately, point-weighted (every grid point
# the same weight) and volume-weighted, over the mesh and per level, and the
# state's own point-weighted norms.
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
#   rho=                            a fixed ρ_max in 1/M (default: 4/M)
#   finder=1                        find the horizon every chunk (0: never; :fitted
#                                   always does)
#   shells=2,9/4,3,5,8              radii of the shells outside the horizon whose
#                                   norms the CSV carries (and one beyond the last;
#                                   and the evolved band inside the horizon)
#   amplitude=1e-8 seed=20261002    the noise (amplitude=0: none)
#   backend=cpu|cuda                cuda needs CUDA in the active environment
#   out=<dir>                       where `octant.csv` and the log go
#   name=<label>                    the run's name in SimWatch (default: out's name)
#   checkpoint=<dir> walltime=<s>   a chain of jobs (as hole_runs.jl's rows)
#   checkpoint_every=8 keep=2       chunks between checkpoints, and how many to keep
#                                   (keep them all to difference two runs later)
#
# Every run also keeps a SimWatch status file, `simwatch.toml` in the `out`
# directory (https://github.com/eschnett/simwatch): the progress, everything
# the CSV row carries, and the record's horizon, track and fit, rewritten at
# every chunk, with `finished`, `stopped` or `failed` at the end.
#
# A run that stops at `walltime` exits with status 3, so a job script can
# resubmit; the same command continues it from the newest checkpoint, and
# appends to the CSV.

using Printf
using Random: Xoshiro
using TreeAMR
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: DIAG_CGH, DIAG_HAM, DIAG_MOM
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
casename = opt("case", "minkowski")
casename in ("minkowski", "ks") || error("case is minkowski or ks, got $casename")
hole = casename == "ks"

backend = if opt("backend", "cpu") == "cuda"
    @eval using CUDA
    Base.invokelatest(() -> CUDA.CUDABackend())
else
    TreeGeneralizedHarmonic.CPU()
end

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
    if opt("interior", "damped") == "fitted"
        m = parse(Int, opt("margin", "8"))
        spec = FittedSpec(T; variant=:fitted, margin=m,
                          n_L=parse(Int, opt("n_L", "0")), lmax_shape=4,
                          lmax_fit=parse(Int, opt("lmax_fit", "8")),
                          fit_cont=parse(Int, opt("fit_cont", "1")))
        # The range projection's gate: step 8f's `9/10`, or `default_gate`'s
        # rule — the offset surface less `2G` spacings — where that is deeper
        # (a margin of 12 cells or more at `h = 1/16`).
        hfine = T(L) / (roots * N) / 2^length(radii)    # the finest spacing
        r_gate = min(T(9 // 10), 2 - (m + 2 * (q ÷ 2 + 1)) * hfine)
        kerr_schild_case(T; halfwidth=L, chunk=chunk, ε_KO=ε_KO, γ2=γ2, octant=true,
                         gauge_source=gs, interior=spec,
                         horizon=Horizon(T; every=1, N=12, spin=true),
                         bounds=default_bounds(T; M=1, r_gate=r_gate))
    else
        kerr_schild_case(T; halfwidth=L, r_0=rat(opt("r_0", "3/4")),
                         r_1=rat(opt("r_1", "3/2")), ρ_ramp=1, chunk=chunk, ε_KO=ε_KO,
                         γ2=γ2, octant=true, gauge_source=gs,
                         margin=parse(Int, opt("margin", "8")), horizon=finder)
    end
else
    minkowski_octant_case(T; L=L, ε_KO=ε_KO, γ0=γ0, γ2=γ2, chunk=chunk)
end
forest0 = hole_forest(T, case; N=N, roots=roots, center=(0, 0, 0), radii=radii,
                      shape=:cube)
nlev = length(radii) + 1
@printf("octant: L = %s, N = %d, %d blocks, %d points, levels %d, h = %s … %s\n",
        L, N, nleaves(forest0), nleaves(forest0) * N^3, nlev,
        maximum(k -> spacing(T, forest0, k), forest0.leaves), minimum_spacing(T, forest0))
@printf("        case %s, t_end = %s, chunk = %s, cfl = %s, q = %d, ε_KO = %s, γ0 = %s, γ2 = %s\n",
        casename, t_end, chunk, cfl, q, ε_KO, hole ? "Gaussian 1/M → 1/(10M)" : γ0, γ2)
hole && @printf("        hole: %s, gauge source %s\n",
                case.interior isa FittedSpec ?
                "fitted target, tracked, margin $(case.interior.margin)" :
                "damped layer, r_0 = $(case.interior.r_0), r_1 = $(case.interior.r_1)",
                case.gauge === nothing ? "sampled" : "algebraic")
@printf("        noise %.3g (seed %d), backend %s, %d threads\n", amplitude, seed,
        typeof(backend), Threads.nthreads())

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
shell_r = hole ? [T(rat(x)) for x in split(opt("shells", "2,9/4,3,5,8"), ',')] : T[]
shell_names = hole ? vcat(["in"], [@sprintf("r%g", r) for r in shell_r]) : String[]
for nm in shell_names, c in ("ham_l2", "ham_linf", "mom_l2", "gauge_l2", "err_l2", "err_linf")
    push!(cols, "$(c)_$nm")
end
isfile(csvpath) || open(io -> println(io, join(cols, ',')), csvpath, "w")

# `mom` and `gauge` are reported as the largest component's norm: the
# components are related by the octant's symmetry, and one number per family
# is what a growth rate is fitted to.
mom2(n) = maximum(n.mom_l2)
mom∞(n) = maximum(n.mom_linf)
gauge2(n) = maximum(n.gauge_l2)
gauge∞(n) = maximum(n.gauge_linf)

# The outermost blocks: those whose extent reaches an outer face.
function nearboundary(forest, k)
    ext = block_extent(T, forest, k)
    return any(d -> ext[d][2] ≥ forest.extents[d][2], 1:3)
end
# Point-weighted L2 of `diag` slot `v` over the blocks `keep(k)` selects.
function band_l2(p, v, keep)
    f = p.U.forest
    w = k -> keep(k) ? one(T) : zero(T)
    n = mesh_mapreduce(identity, +, zero(T), p.diag; vars=TreeGeneralizedHarmonic.DIAG_MASK,
                       weight=w)
    s = mesh_mapreduce(x -> x * x, +, zero(T), p.diag; vars=v, weight=w)
    return n > 0 ? sqrt(s / n) : zero(T)
end
const DIAG_ERR = TreeGeneralizedHarmonic.DIAG_ERR
bndkeep(p) = k -> nearboundary(p.U.forest, k)
# The largest component's band norm, as for the whole mesh.
band_max(p, first, n, keep) = maximum(band_l2(p, first + i - 1, keep) for i in 1:n)

t_wall0 = time()

# The SimWatch status file (added 2026-10-02): `simwatch.toml` in the run
# directory, rewritten at every chunk with everything the CSV row carries, the
# record's horizon, track, fit and step, and the progress; see
# `src/simwatch.jl` and SimWatch's `FORMAT.md`.
sw = SimWatchWriter(outdir; name=opt("name", basename(outdir)),
                    code="TreeGeneralizedHarmonic")
time_unit = hole ? "M" : nothing
setup = Dict{String,Any}(
    "case" => casename, "L" => Float64(L), "N" => N, "levels" => nlev,
    "h_min" => Float64(minimum_spacing(T, forest0)), "q" => q, "cfl" => Float64(cfl),
    "eps_KO" => Float64(ε_KO), "noise" => amplitude, "options" => join(ARGS, " "))
if hole
    setup["interior"] = case.interior isa FittedSpec ? "fitted" : "damped"
    setup["gauge_source"] = case.gauge === nothing ? "sampled" : "algebraic"
    case.interior isa FittedSpec ? (setup["margin_cells"] = case.interior.margin) :
                                   (setup["r_0"] = Float64(case.interior.r_0);
                                    setup["r_1"] = Float64(case.interior.r_1))
end
simwatch_update!(sw; force=true, status="starting", time_end=Float64(t_end),
                 time_unit=time_unit,
                 walltime_limit=walltime === nothing ? nothing : walltime,
                 message="compiling and building the initial data",
                 extra=Dict("setup" => setup))
steps_seen = Ref(0)
last_status = Ref{Any}((; extra=Dict{String,Any}("setup" => setup), black_holes=nothing))

function observe(p, t, u, rec)
    gh_constraint!(p, u, t)
    adm_constraint!(p, u, t)
    pts = constraint_norms(p; weighting=:points)
    vol = constraint_norms(p)
    lev = level_constraint_norms(p)
    bk = bndkeep(p)
    bnd = (band_l2(p, DIAG_HAM, bk), band_max(p, DIAG_MOM, 3, bk),
           band_max(p, DIAG_CGH, 4, bk))
    npts = T(statelength(p.U)) / p.U.nvars
    s2 = sqrt(mesh_mapreduce(x -> x * x, +, zero(T), p.U, u) / (npts * p.U.nvars))
    s∞ = mesh_mapreduce(abs, max, zero(T), p.U, u)
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
        gh_error!(p, u, t; shell=horizon_shell(case, p.interior))
        e = error_norms(p)
        ep = masked_norms(p, DIAG_ERR; weighting=:points)
        err = (ep.l2, ep.linf, e.residual, e.drift, e.err_l2)
        append!(row, err)
        extra["error"] = Dict("l2" => ep.l2, "linf" => ep.linf, "l2_volume" => e.err_l2,
                              "layer_residual" => e.residual, "drift_htt" => e.drift)
    end
    levels = Dict{String,Any}()
    for n in lev
        append!(row, (n.ham_l2, n.ham_linf, mom2(n), mom∞(n), gauge2(n), gauge∞(n)))
        levels["L$(n.level)"] = Dict{String,Any}(
            "points" => n.points, "ham_l2" => n.ham_l2, "ham_linf" => n.ham_linf,
            "mom_l2" => mom2(n), "gauge_l2" => gauge2(n))
    end
    if hole
        for n in lev
            el = masked_norms(p, DIAG_ERR; weighting=:points, at_level=n.level).l2
            push!(row, el)
            levels["L$(n.level)"]["err_l2"] = el
        end
    end
    extra["levels"] = levels
    append!(row, bnd)
    boundary = Dict{String,Any}("note" => "the blocks touching an outer face",
                                "ham_l2" => bnd[1], "mom_l2" => bnd[2],
                                "gauge_l2" => bnd[3])
    if hole
        eb = band_l2(p, DIAG_ERR, bk)
        push!(row, eb)
        boundary["err_l2"] = eb
    end
    extra["boundary"] = boundary
    # The shells: each monitor once per shell, masked to it.
    if hole
        r_in = case.interior isa FittedSpec ?
               shell_r[1] - case.interior.margin * minimum_spacing(T, p.U.forest) :
               case.interior.r_1
        edges = vcat([r_in], shell_r, [T(Inf)])
        shells = Dict{String,Any}()
        for i in 1:(length(edges) - 1)
            m = ShellMask(SVector{3,T}(0, 0, 0), edges[i], edges[i + 1])
            gh_constraint!(p, u, t; mask=m)
            adm_constraint!(p, u, t; mask=m)
            n = constraint_norms(p; weighting=:points)
            gh_error!(p, u, t; mask=m)
            e = masked_norms(p, DIAG_ERR; weighting=:points)
            append!(row, (n.ham_l2, n.ham_linf, mom2(n), gauge2(n), e.l2, e.linf))
            shells[shell_names[i]] = Dict{String,Any}(
                "range" => @sprintf("[%g, %g)", edges[i], edges[i + 1]),
                "ham_l2" => n.ham_l2, "ham_linf" => n.ham_linf, "mom_l2" => mom2(n),
                "gauge_l2" => gauge2(n), "err_l2" => e.l2, "err_linf" => e.linf)
        end
        extra["shells"] = shells
    end
    open(csvpath, "a") do io
        println(io, join((x isa Real ? @sprintf("%.10g", x) : string(x) for x in row), ','))
    end
    @printf("t = %8.3f  wall %8.1f s   ℋ %.3e / %.3e   ℳ %.3e / %.3e   C %.3e / %.3e   |u| %.3e",
            t, time() - t_wall0, pts.ham_l2, pts.ham_linf, mom2(pts), mom∞(pts),
            gauge2(pts), gauge∞(pts), s2)
    err === nothing || @printf("   err %.3e / %.3e  res %.3e  drift %.3e", err[1], err[2],
                               err[3], err[4])
    @printf("   bnd ℋ %.3e C %.3e\n", bnd[1], bnd[3])
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
        @sprintf("t = %.1f M: ℋ %.2e just outside the horizon, %.2e overall; M_irr − 1 = %s",
                 t, extra["shells"][shell_names[2]]["ham_l2"], pts.ham_l2,
                 rec.M_irr === nothing ? "—" : @sprintf("%+.2e", rec.M_irr - 1))
    else
        @sprintf("t = %.1f: ℋ %.2e, ℳ %.2e, C %.2e", t, pts.ham_l2, mom2(pts), gauge2(pts))
    end
    rec.finite || (msg = "non-finite values in the evolved region; " * msg)
    last_status[] = (; extra=extra, black_holes=bhs, message=msg)
    simwatch_update!(sw; force=true, iteration=steps_seen[], time=Float64(t),
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
    rf === nothing || println("restarting from $rf")
    (checkpoint_path_prefix=pre, checkpoint_every_chunks=ckevery, restart_file=rf,
     max_walltime_seconds=walltime, num_checkpoints_keep=ckkeep)
end
restarting = get(ckw, :restart_file, nothing) !== nothing

# A hole's frozen core is left alone: it is never evolved, and its stale data
# at the corner is not of definite parity (see `add_noise!`).
core² = !hole ? zero(T) : case.interior isa FittedSpec ?
        rat(opt("r_core", "3/4"))^2 : case.interior.r_0^2
fitkw = if hole && case.interior isa FittedSpec
    nL = case.interior.n_L > 0 ? case.interior.n_L :
         TreeGeneralizedHarmonic.layer_cells(q ÷ 2 + 1, T(4), one(T))
    depth = haskey(OPTIONS, "fit_depth") ? T(rat(OPTIONS["fit_depth"])) :
            nL * minimum_spacing(T, forest0)
    (fit_initial_depth=depth, fit_initial_cont=case.interior.fit_cont)
else
    (;)
end
haskey(OPTIONS, "rho") && (fitkw = (; fitkw..., ρ_max_fixed=T(rat(OPTIONS["rho"]))))
perturb = iszero(amplitude) ? nothing :
          U -> add_noise!(U, Xoshiro(seed); amplitude=amplitude,
                          exclude=hole ? (x -> sum(abs2, x) < core²) : nothing)
out = try
    evolve!(T, case; forest=restarting ? nothing : forest0, q=q,
            ops=Operators(prolongation=q + 2, restriction=q + 2), t_end=t_end,
            chunk=chunk, cfl=cfl, backend=backend, observer=observe,
            perturb=perturb, fitkw..., ckw...)
catch err
    ls = last_status[]
    simwatch_finish!(sw; status="failed", message=sprint(showerror, err),
                     time_end=Float64(t_end), time_unit=time_unit,
                     black_holes=ls.black_holes, extra=ls.extra)
    rethrow()
end
r = out.records[end]
let ls = last_status[]
    simwatch_finish!(sw; status=out.finished ? "finished" : "stopped",
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
open(joinpath(outdir, "records.csv"), "w") do io
    println(io, join(fields, ','))
    for rec in out.records
        println(io, join((string(get(rec, f, missing)) for f in fields), ','))
    end
end
@printf("done: t = %.4f, %d steps, dt = %.6g, finished = %s, wall %.1f s\n",
        out.t, out.nsteps, r.dt, out.finished, time() - t_wall0)
out.finished || exit(3)
