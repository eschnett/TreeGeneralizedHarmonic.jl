# The aligned-motion campaign (added 2026-10-22): a Kerr-Schild hole (`M = 1`)
# of spin `a` along `z`, moving along its spin axis at speed `v` from `z = −2`
# at `t = 0` to `t = 20 M`, on the rotating quadrant `[0, 64]² × [−64, 64]`
# with a frozen capsule mesh swept along the path
# (`test/octant_runs.jl octant=rotating_quadrant`), the analytic `:damped`
# layer at `4/M` with the target rate, `q = 4`, `cfl = 1/2`, `chunk = M` and
# `split` off. The matrix is four layer families, each at three spacings, at
# the speeds `v ∈ {0, 1/5, 1/2}` listed:
#
#   a = 0      h = 1/16, 1/24, 1/32    radii 32,16,8,4        r_0 = 1/2,   r_1 = 19/16
#              v = 0, 1/5, 1/2
#   a = 1/2    h = 1/24, 1/32, 1/40    radii 32,16,8,4        r_0 = 3/4,   r_1 = 5/4
#              v = 0, 1/5, 1/2
#   a = 9/10   h = 1/72, 1/96, 1/128   radii 32,16,8,4,3,2    r_0 = 19/20, r_1 = 9/8
#              v = 0, 1/2
#   a = 9/10   the same mesh                                  r_0 = 19/20, r_1 = 5/4
#              v = 0, 1/5              (rows suffixed `r`)
#
# (`h = 1/128` at rest only: a moving `a = 9/10` hole at `1/128` does not fit
# one H200). The rows are named `a{0,5,9}v{0,2,5}h{N}`, `h = 1/N`, and the
# `r_1 = 5/4` family's `a9v{0,2}h{N}r`; a moving row is compared with the
# resting row of its own family. The data of 2026-10-22: `ks5/` the resting
# rows of `r_1 = 9/8` and of `a = 0` and `1/2`, `ks6/` `a5v2…` and the `r`
# family, `ks7/` (the core relaxing at `max(ρ_max, 12|v|/r_0)`, 031ab30) the
# other moving rows. This script
# reads each row's `octant.csv` (one row per M: the global norms, and the
# point-weighted norms in shells about the hole's current center `c(t)` — the
# first at the horizon's equatorial radius `√(r₊² + a²)`, then `9/4`, `3`, `5`,
# `8`, each to the next), its `records.csv` (the horizon's `M_irr` and `J`, the
# find's success, the steps of each chunk; written when a job ends), its
# `simwatch.toml` (the status, and for a row still running the horizon's history
# to six digits) and its job logs (the point count and `gpu memory used`), and
# prints Markdown tables for `SINGULARITY_HANDLING.md`.
#
#     julia test/aligned_study.jl [dir=<data>] [times=2,5,10,20]
#                                 [runs=a0v5h16,a9v2h72,…] [batches=ks5,ks6]
#
# `dir` holds one subdirectory per batch of jobs (`ks5/`, `ks6/`), each holding
# one directory per row and the jobs' logs `<row>-<jobid>.log` beside them. A
# batch whose name contains `obsolete` is skipped unless `batches=` names it;
# where a row is in two batches the later batch (by name) is used. A row may be
# missing, queued or partial: what it does not have is printed as `—`. A CSV
# whose `t` starts again from `0` holds a job that was rerun from scratch, and
# only the rows from the last start are read; a restart from a checkpoint (a
# `t > 0`) continues the chain, its repeated times reduced to the last row.
#
# With `split` off the CSV has no `hz_J` and no extents: `J` is `records.csv`'s,
# the Korzyński `J`'s *magnitude*, so `J − a` at `a = 0` is `|J| ≥ 0` and its
# sign carries nothing there; the extent ratio is not reported.

using Printf

rat(s) = (p = split(s, '/'); length(p) == 2 ? parse(Int, p[1]) / parse(Int, p[2]) :
                            parse(Float64, s))
opts = Dict(String(k) => String(v)
            for (k, v) in (split(a, '='; limit=2) for a in ARGS if occursin('=', a)))

default_dir = first(filter(isdir, [joinpath(@__DIR__, "..", ".claude", "data", "aligned"),
                                   joinpath(@__DIR__, "..", "..", "..", "data", "aligned"),
                                   "."]))
const DIR = normpath(abspath(get(opts, "dir", default_dir)))
const TIMES = rat.(split(get(opts, "times", "2,5,10,20"), ','))
const HTIMES = [1.0, 5.0, 10.0, 20.0]          # the horizon's table
const RTIMES = [1.0, 10.0, 20.0]               # the layer residual's
const T_END = 20.0
const PICK = haskey(opts, "runs") ? String.(split(opts["runs"], ',')) : nothing
const BATCHES = haskey(opts, "batches") ? String.(split(opts["batches"], ',')) : nothing

# The campaign's layer families: the name's digit and suffix (`code` is the
# two), the spin, the label, the spacings and the speeds that were run.
const SPINS = [(code="0", digit='0', suffix="", a=0.0, label="0", Ns=[16, 24, 32],
                vs="025"),
               (code="5", digit='5', suffix="", a=0.5, label="1/2", Ns=[24, 32, 40],
                vs="025"),
               (code="9", digit='9', suffix="", a=0.9, label="9/10, r_1 = 9/8",
                Ns=[72, 96, 128], vs="05"),
               (code="9r", digit='9', suffix="r", a=0.9, label="9/10, r_1 = 5/4",
                Ns=[72, 96, 128], vs="02")]
const SPEEDS = [(code='0', v=0.0, label="0"), (code='2', v=0.2, label="1/5"),
                (code='5', v=0.5, label="1/2")]
spin(c) = SPINS[findfirst(s -> s.code == c, SPINS)]
speed(c) = SPEEDS[findfirst(s -> s.code == c, SPEEDS)]
speeds(a) = filter(v -> v.code in a.vs, SPEEDS)
runname(a, v, N) = "a$(a.digit)v$(v.code)h$N$(a.suffix)"
# Kerr-Schild (`M = 1`): `r₊` and the irreducible mass.
r_plus(a) = 1 + sqrt(1 - a^2)
m_irr(a) = sqrt(r_plus(a) / 2)
# The regions every comparison reads: the shell at the horizon's equatorial
# radius (`r_h`), the shells from 3 and from 5, and the whole evolved region.
const REGIONS = ["r_h", "r3", "r5", "global"]

num(s) = (x = tryparse(Float64, s);
          x !== nothing ? x : s == "true" ? 1.0 : s == "false" ? 0.0 : NaN)
fmt(x) = isfinite(x) ? @sprintf("%.3e", x) : "—"
fmts(x) = isfinite(x) ? @sprintf("%+.2e", x) : "—"
fmtr(x) = isfinite(x) ? @sprintf("%.3g", x) : "—"
fmto(x) = isfinite(x) ? @sprintf("%.2f", x) : "—"
fmtt(x) = isfinite(x) ? @sprintf("%g", x) : "—"

function table(header, rows)
    println("| ", join(header, " | "), " |")
    println("|", join([i == 1 ? ":---" : "---:" for i in eachindex(header)], "|"), "|")
    for r in rows
        println("| ", join(r, " | "), " |")
    end
    println()
end

# `a9v2h96r` → (family "9r", speed '2', N = 96).
function parse_name(nm)
    m = match(r"^a([059])v([025])h(\d+)(r?)$", nm)
    m === nothing && return nothing
    return (acode=m[1] * m[4], vcode=m[2][1], N=parse(Int, m[3]))
end

# A CSV as a header and rows of numbers, short rows padded with `NaN`.
function readcsv(p)
    lines = isfile(p) ? filter(!isempty, readlines(p)) : String[]
    isempty(lines) && return String[], Vector{Float64}[]
    hdr = String.(split(lines[1], ','))
    rows = map(lines[2:end]) do l
        f = split(l, ',')
        [j <= length(f) ? num(f[j]) : NaN for j in eachindex(hdr)]
    end
    return hdr, rows
end

# The status file's `status`, its progress (`iteration`, `time`), and the
# horizon's history (six digits), which is what a row still running has of it.
function readsimwatch(d)
    f = joinpath(d, "simwatch.toml")
    s = isfile(f) ? read(f, String) : ""
    m = match(r"(?m)^status = \"(\w+)\"", s)
    status = m === nothing ? (isdir(d) ? "—" : "missing") : String(m[1])
    vec(key) = (m = match(Regex("(?m)^" * key * raw" = \[([^\]]*)\]"), s);
                m === nothing ? Float64[] : num.(strip.(split(m[1], ','))))
    pm = match(r"(?ms)^\[progress\].*?^iteration = (\d+).*?^time = ([0-9.eE+-]+)", s)
    return (; status, hist_t=vec("time"), hist_M=vec("\"horizon.M_irr\""),
            hist_J=vec("\"horizon.J\""),
            iteration=pm === nothing ? NaN : parse(Float64, pm[1]),
            ptime=pm === nothing ? NaN : parse(Float64, pm[2]))
end

# The job logs beside the row: the point count from the header and the largest
# `gpu memory used` of every job of the row.
function readlogs(batchdir, name)
    pts, mib = NaN, NaN
    for f in readdir(batchdir)
        startswith(f, name * "-") && endswith(f, ".log") || continue
        for l in eachline(joinpath(batchdir, f))
            m = match(r"^octant .*?, (\d+) points,", l)
            m === nothing || (pts = parse(Float64, m[1]))
            m = match(r"^gpu memory used: (\d+) MiB", l)
            m === nothing || (mib = max(isfinite(mib) ? mib : 0.0, parse(Float64, m[1])))
        end
    end
    return pts, mib
end

# One row directory: the CSV with a restart's repeated times reduced to the last
# row of each and the wall time made cumulative over the chain's jobs (`wall`
# starts again from zero with every job), the records by time, the status file
# and the logs.
function readrun(batch, name)
    d = joinpath(DIR, batch, name)
    info = parse_name(name)
    sw = readsimwatch(d)
    pts, mib = readlogs(joinpath(DIR, batch), name)
    hdr, raw = readcsv(joinpath(d, "octant.csv"))
    idx = Dict(h => j for (j, h) in enumerate(hdr))
    jw = get(idx, "wall", nothing)
    off, prev = 0.0, 0.0
    cum = Dict{Float64,Float64}()
    bytime = Dict{Float64,Vector{Float64}}()
    for r in raw
        isfinite(r[1]) || continue
        # A job rerun from scratch: what came before it is not this run.
        if r[1] == 0 && !isempty(bytime)
            empty!(bytime); empty!(cum)
            off, prev = 0.0, 0.0
        end
        w = jw === nothing ? NaN : r[jw]
        if isfinite(w)
            w < prev && (off += prev)
            prev = w
        end
        bytime[r[1]] = r
        cum[r[1]] = off + w
    end
    t = sort!(collect(keys(bytime)))
    # The shells' names: the first (smallest) outside the horizon is `r_h`.
    radii = sort!([parse(Float64, m[1]) for m in (match(r"^err_l2_r([0-9.]+)$", h) for h in hdr)
                   if m !== nothing])
    shell = Dict{String,String}("global" => "")
    isempty(radii) || (shell["r_h"] = @sprintf("r%g", radii[1]))
    for s in ("r3", "r5")
        haskey(idx, "err_l2_$s") && (shell[s] = s)
    end
    # The horizon, from `records.csv` or (a row still running) the status file.
    rh, rrows = readcsv(joinpath(d, "records.csv"))
    hz = Dict{Float64,NamedTuple}()
    if !isempty(rrows)
        ri = Dict(h => j for (j, h) in enumerate(rh))
        g(r, k) = haskey(ri, k) ? r[ri[k]] : NaN
        for r in rrows
            hz[r[1]] = (M_irr=g(r, "M_irr"), J=g(r, "J"), found=g(r, "horizon_success"),
                        steps=g(r, "steps"), residual=g(r, "residual"))
        end
        hsrc = "records"
    elseif !isempty(sw.hist_t)
        for (i, τ) in enumerate(sw.hist_t)
            M = i <= length(sw.hist_M) ? sw.hist_M[i] : NaN
            J = i <= length(sw.hist_J) ? sw.hist_J[i] : NaN
            hz[τ] = (M_irr=M, J=J, found=isfinite(M) ? 1.0 : 0.0, steps=NaN, residual=NaN)
        end
        hsrc = "simwatch"
    else
        hsrc = "none"
    end
    a, v = spin(info.acode), speed(info.vcode)
    return (; name, batch, label="$batch/$name", info..., a=a.a, alabel=a.label, v=v.v,
            vlabel=v.label, status=sw.status, t, rows=[bytime[τ] for τ in t], idx,
            cum=[cum[τ] for τ in t], shell, hz, hsrc, points=pts, gpu_mib=mib,
            iteration=sw.iteration, ptime=sw.ptime)
end

tlast(r) = r === nothing || isempty(r.t) ? NaN : r.t[end]
finished(r) = r !== nothing && (r.status == "finished" || tlast(r) >= T_END - 1e-6)
tindex(r, τ) = r === nothing ? nothing : findfirst(t -> abs(t - τ) < 1e-6, r.t)
has(r, τ) = tindex(r, τ) !== nothing
function at(r, name, τ)
    r === nothing && return NaN
    j, i = get(r.idx, name, nothing), tindex(r, τ)
    (j === nothing || i === nothing) && return NaN
    return r.rows[i][j]
end
# `q` (`err_l2` or `ham_l2`) in a region, at `τ`.
function reg(r, q, region, τ)
    r === nothing && return NaN
    s = get(r.shell, region, nothing)
    s === nothing && return NaN
    return at(r, region == "global" ? q : "$(q)_$s", τ)
end
hzat(r, k, τ) = r === nothing ? NaN :
                (h = get(r.hz, Float64(τ), nothing); h === nothing ? NaN : getproperty(h, k))
order(ec, ef, Nc, Nf) = log(abs(ec / ef)) / log(Nf / Nc)
hlabel(N) = "1/$N"
mark(r) = r !== nothing && r.hsrc == "simwatch" ? "†" : ""

# The least-squares slope of `y` against `t` over `a ≤ t ≤ b`.
function slope(t, y, a, b)
    k = [i for i in eachindex(t) if a - 1e-9 ≤ t[i] ≤ b + 1e-9 && isfinite(y[i])]
    length(k) < 3 && return NaN
    x̄, ȳ = sum(t[k]) / length(k), sum(y[k]) / length(k)
    return sum((t[k] .- x̄) .* (y[k] .- ȳ)) / sum((t[k] .- x̄) .^ 2)
end
# The horizon's drift over `[10, 20]` (to the last time a partial row has).
function hdrift(r, k)
    r === nothing && return NaN, NaN
    ts = sort!(collect(keys(r.hz)))
    b = min(T_END, isempty(ts) ? NaN : ts[end])
    isfinite(b) && b >= 12 || return NaN, b
    return slope(ts, [getproperty(r.hz[τ], k) for τ in ts], 10.0, b), b
end
# The growth of the global error over `[10, 20]` (`[10, t]` for a partial row
# that has passed 12): what tells a run that is stable from one whose error
# grows — at `a = 9/10` it grows in the evolved band inside the horizon, `in`,
# which only the global norm sees of the regions above.
const GROWTH_MAX = 2.0
growth(r, q="err_l2") = (b = min(T_END, tlast(r)); isfinite(b) && b >= 12 ?
                         at(r, q, b) / at(r, q, 10.0) : NaN)

# Every row of the data directory, and the one used for each (a, v, N): the
# latest batch that has data.
function readall()
    isdir(DIR) || error("no data directory $DIR")
    batches = [b for b in sort(readdir(DIR)) if isdir(joinpath(DIR, b)) &&
               (BATCHES === nothing ? !occursin("obsolete", b) : b in BATCHES)]
    all_runs = [readrun(b, n) for b in batches for n in sort(readdir(joinpath(DIR, b)))
                if isdir(joinpath(DIR, b, n)) && parse_name(n) !== nothing]
    PICK === nothing || filter!(r -> r.name in PICK || r.label in PICK, all_runs)
    chosen = Dict{Tuple{String,Char,Int},Any}()
    for r in all_runs                               # batches in name order: later wins
        isempty(r.t) && haskey(chosen, (r.acode, r.vcode, r.N)) && continue
        chosen[(r.acode, r.vcode, r.N)] = r
    end
    return all_runs, chosen
end

function main()
    all_runs, chosen = readall()
    pick(a, v, N) = get(chosen, (a.code, v.code, N), nothing)
    present(a, v) = [N for N in a.Ns if (r = pick(a, v, N)) !== nothing && !isempty(r.t)]
    wanted(name) = PICK === nothing || name in PICK
    v0 = SPEEDS[1]

    println("# Aligned motion: a spinning Kerr-Schild hole moving along its spin axis\n")
    println("data: `$DIR`. Point-weighted L2 norms; the shells are about the hole's ",
            "current center `c(t) = (0, 0, −2 + v t)`: `r_h` from the horizon's ",
            "equatorial radius `√(r₊² + a²)` (2 at `a = 0`, 1.93185 at 1/2, 1.69463 at ",
            "9/10) to 9/4, `r3` from 3 to 5, `r5` from 5 to 8; `global` is the evolved ",
            "region (outside `r_1`). `—`: missing, or not reached yet.\n")

    println("## Runs\n")
    rows = Vector{String}[]
    for a in SPINS, v in speeds(a), N in a.Ns
        nm = runname(a, v, N)
        wanted(nm) || continue
        r = pick(a, v, N)
        r === nothing && (push!(rows, [nm, "—", a.label, v.label, hlabel(N), "—", "—", "no",
                                       a.digit == '9' && v.code != '0' && N == 128 ?
                                       "not run (no fit on one H200)" : "missing", "—", "—",
                                       "—", "—"]);
                          continue)
        push!(rows, [nm, r.batch, a.label, v.label, hlabel(N),
                     isfinite(r.points) ? @sprintf("%.4g", r.points) : "—",
                     isempty(r.t) ? "—" : fmtt(tlast(r)), finished(r) ? "yes" : "no", r.status,
                     isempty(r.cum) || !isfinite(r.cum[end]) ? "—" :
                     @sprintf("%.2f", r.cum[end] / 3600),
                     isfinite(r.gpu_mib) ? @sprintf("%.1f", r.gpu_mib / 1024) : "—",
                     fmtr(growth(r)), fmtr(growth(r, "err_l2_in"))])
    end
    println("`growth` is `err_l2` at `t = 20` over `t = 10` (at the last time over ",
            "`t = 10` for a partial row past 12), globally and in the evolved band ",
            "inside the horizon (`in`, from `r_1` to `r_h`); over $(fmtt(GROWTH_MAX)) ",
            "the row is not counted as stable in the verdict.\n")
    table(["run", "batch", "a", "v", "h", "points", "t reached", "finished", "status",
           "wall [h]", "peak GPU [GiB]", "growth", "growth in"], rows)

    println("## Moving against rest\n")
    println("`moving / rest` at the same `a` and `h`, each about its own `c(t)`.\n")
    for a in SPINS, N in a.Ns
        rs = pick(a, v0, N)
        mv = [(v, pick(a, v, N)) for v in speeds(a)[2:end]]
        (rs === nothing || all(((_, r),) -> r === nothing, mv)) && continue
        for q in ("err_l2", "ham_l2")
            println("### `$q`, a = $(a.label), h = $(hlabel(N))\n")
            hdr = vcat(["t"], ["$s v=$(v.label)" for s in REGIONS[1:3] for (v, _) in mv])
            table(hdr, [vcat([fmtt(τ)], [fmtr(reg(r, q, s, τ) / reg(rs, q, s, τ))
                                         for s in REGIONS[1:3] for (_, r) in mv])
                        for τ in TIMES])
        end
    end

    println("## Convergence orders\n")
    println("`log(e_c/e_f)/log(h_c/h_f)` for adjacent spacings; 4 is expected outside ",
            "the horizon.\n")
    for a in SPINS, v in speeds(a)
        have = present(a, v)
        for (Nc, Nf) in zip(have[1:end-1], have[2:end])
            rc, rf = pick(a, v, Nc), pick(a, v, Nf)
            println("### a = $(a.label), v = $(v.label), h = $(hlabel(Nc)) → $(hlabel(Nf))\n")
            hdr = vcat(["t"], ["err $s" for s in REGIONS], ["ham $s" for s in REGIONS])
            table(hdr, [vcat([fmtt(τ)],
                             [fmto(order(reg(rc, q, s, τ), reg(rf, q, s, τ), Nc, Nf))
                                         for q in ("err_l2", "ham_l2") for s in REGIONS])
                        for τ in TIMES])
        end
    end

    println("## Horizon\n")
    println("`ΔM` is `M_irr − √(r₊/2)`, `ΔJ` is `|J| − a` (`records.csv`'s `J` is the ",
            "magnitude with `split` off), `found` the finds that succeeded of the rows ",
            "recorded. `†`: a row still running, read from the status file's history ",
            "(six digits, so `ΔM` and `ΔJ` to about `1e−6`).\n")
    for a in SPINS
        rs = filter(r -> r !== nothing && !isempty(r.hz),
                    [pick(a, v, N) for v in speeds(a) for N in a.Ns])
        isempty(rs) && continue
        println("### a = $(a.label): `√(r₊/2) = ", @sprintf("%.8f", m_irr(a.a)), "`\n")
        hdr = vcat(["run"], ["ΔM t=$(fmtt(τ))" for τ in HTIMES],
                   ["ΔJ t=$(fmtt(τ))" for τ in HTIMES], ["found"])
        nfound(r) = count(h -> h.found == 1, values(r.hz))
        table(hdr, [vcat([r.name * mark(r)],
                         [fmts(hzat(r, :M_irr, τ) - m_irr(a.a)) for τ in HTIMES],
                         [fmts(hzat(r, :J, τ) - a.a) for τ in HTIMES],
                         [@sprintf("%d/%d", nfound(r), length(r.hz))])
                    for r in rs])
    end
    println("### Drift\n")
    println("Least-squares slopes over `[10, 20]` (`[10, t]` for a partial row), per M, ",
            "and their order against the next coarser `h` at the same `a` and `v`.\n")
    rows = Vector{String}[]
    for a in SPINS, v in speeds(a)
        prev = nothing
        for N in present(a, v)
            r = pick(a, v, N)
            dM, b = hdrift(r, :M_irr)
            dJ, _ = hdrift(r, :J)
            oM = oJ = NaN
            if prev !== nothing
                oM = order(prev[2], dM, prev[1], N)
                oJ = order(prev[3], dJ, prev[1], N)
            end
            push!(rows, [r.name * mark(r), isfinite(b) && b >= 12 ? "[10, $(fmtt(b))]" : "—",
                         fmts(dM), fmto(oM), fmts(dJ), fmto(oJ)])
            prev = (N, dM, dJ)
        end
    end
    table(["run", "window", "dM_irr/dt", "order", "d|J|/dt", "order"], rows)

    println("## Layer residual\n")
    println("The `:damped` layer's distance from the analytic solution it relaxes to ",
            "(the record's `residual`); `order` is at `t = 20` against the next finer ",
            "`h`, `20/1` the growth from `t = 1`.\n")
    rows = Vector{String}[]
    for a in SPINS, v in speeds(a)
        have = present(a, v)
        for (i, N) in enumerate(have)
            r = pick(a, v, N)
            o = i == length(have) ? NaN :
                order(at(r, "residual", 20.0), at(pick(a, v, have[i+1]), "residual", 20.0),
                      N, have[i+1])
            push!(rows, vcat([r.name], [fmt(at(r, "residual", τ)) for τ in RTIMES],
                             [fmtr(at(r, "residual", 20.0) / at(r, "residual", 1.0)), fmto(o)]))
        end
    end
    table(vcat(["run"], ["t=$(fmtt(τ))" for τ in RTIMES], ["20/1", "order"]), rows)

    println("## Cost\n")
    println("`s/M` is the wall time over the last 10 M a row has (`[t − 10, t]`; ",
            "`[1, t]` if `t < 11`), monitors and finds included; `steps/M` from ",
            "`records.csv`'s steps over the same window (the status file's iteration ",
            "count over its time for a row still running); `ns/pt/step` is ",
            "`s/M ÷ (steps/M × points)`.\n")
    rows = Vector{String}[]
    for a in SPINS, v in speeds(a), N in a.Ns
        r = pick(a, v, N)
        (r === nothing || length(r.t) < 2) && continue
        t1 = tlast(r)
        t0 = t1 >= 11 ? t1 - 10 : 1.0
        i0, i1 = tindex(r, t0), tindex(r, t1)
        spm = (i0 === nothing || t1 <= t0) ? NaN : (r.cum[i1] - r.cum[i0]) / (t1 - t0)
        st = [h.steps for (τ, h) in r.hz if t0 + 1e-6 < τ <= t1 + 1e-6]
        stpm = length(st) == round(Int, t1 - t0) && all(isfinite, st) ? sum(st) / (t1 - t0) :
               r.iteration / r.ptime
        push!(rows, [r.name, @sprintf("[%g, %g]", t0, t1), fmtr(spm), fmtr(stpm),
                     isfinite(r.points) ? @sprintf("%.4g", r.points) : "—",
                     fmtr(spm / (stpm * r.points) * 1e9)])
    end
    table(["run", "window", "s/M", "steps/M", "points", "ns/pt/step"], rows)

    println("## Verdict\n")
    println("Read at `t = 20` (at the latest time both rows of the pair have, shown, for ",
            "a pair still running). **works**: every row of the cell that was run ",
            "reached `t = 20` with finite norms and a global `err_l2` that grows by ",
            "at most $(fmtt(GROWTH_MAX))× over `[10, 20]` (**grows** names the rows that ",
            "grow more, with the factor; `running` the furthest a partial row has got, ",
            "its growth read over `[10, t]`). `p` is the convergence order of `err_l2` ",
            "at the finest pair in `r3`, `r5` and globally. `mov/rest` is `err_l2` in ",
            "`r3`, moving over at rest, at the finest moving `h`. **asymptotic from**: ",
            "the coarsest `h` from which every adjacent pair up to the finest has an ",
            "order in `[3.5, 5]` for `err_l2` in both `r3` and `r5` — the coarsest spacing that is in the convergent regime ",
            "outside the horizon; `none` when the finest pair itself is not. `R` is ",
            "`e(h_c) / (e(h_f) (h_c/h_f)⁴)` in `r3` for the coarsest `h_c` against the ",
            "finest `h_f`: the coarsest row's error over the order-4 extrapolation from ",
            "the finest (`≤ 2` is within 2× of the asymptotic behaviour). `growth` is the ",
            "finest row's.\n")
    cells = Dict{Tuple{String,Char},String}()
    rows = Vector{String}[]
    for a in SPINS, v in speeds(a)
        have = present(a, v)
        isempty(have) && (cells[(a.code, v.code)] = "—"; continue)
        rs = [pick(a, v, N) for N in have]
        bad = [r for r in rs if r.status == "failed" ||
               (!isempty(r.t) && !isfinite(at(r, "err_l2", tlast(r))))]
        run_ = [r for r in rs if !finished(r)]
        grow = [r for r in rs if growth(r) > GROWTH_MAX]
        state = !isempty(bad) ? "fails ($(join([hlabel(r.N) for r in bad], ", ")))" :
                !isempty(grow) ?
                "grows ($(join(["$(hlabel(r.N)) ×$(fmtr(growth(r)))" for r in grow], ", ")))" :
                "works"
        isempty(run_) ||
            (state *= ", running ($(join(["$(hlabel(r.N)) t=$(fmtt(tlast(r)))"
                                          for r in run_], ", ")))")
        # The time of the reading: 20, or the latest the finest pair shares.
        teval(rc, rf) = (τ = min(tlast(rc), tlast(rf)); τ >= T_END - 1e-6 ? T_END : τ)
        p3 = p5 = pg = R = NaN
        τf = NaN
        if length(have) >= 2
            Nc, Nf = have[end-1], have[end]
            rc, rf = pick(a, v, Nc), pick(a, v, Nf)
            τf = teval(rc, rf)
            p3, p5, pg = (order(reg(rc, "err_l2", s, τf), reg(rf, "err_l2", s, τf), Nc, Nf)
                          for s in ("r3", "r5", "global"))
            r0 = pick(a, v, have[1])
            τR = teval(r0, rf)
            R = reg(r0, "err_l2", "r3", τR) /
                (reg(rf, "err_l2", "r3", τR) * (Nf / have[1])^4)
        end
        # Coarsest asymptotic `h`, walking down from the finest pair.
        asym = length(have) < 2 ? "—" : "none"
        for i in (length(have) - 1):-1:1
            rc, rf = pick(a, v, have[i]), pick(a, v, have[i+1])
            τ = teval(rc, rf)
            ok = all(s -> 3.5 <= order(reg(rc, "err_l2", s, τ), reg(rf, "err_l2", s, τ),
                                       have[i], have[i+1]) <= 5, ("r3", "r5"))
            ok || break
            asym = hlabel(have[i])
        end
        rf = rs[end]
        Nm = have[end]
        while v.code != '0' && pick(a, v0, Nm) === nothing && Nm != have[1]
            Nm = have[findfirst(==(Nm), have) - 1]
        end
        τm = v.code == '0' ? NaN : min(tlast(pick(a, v, Nm)), tlast(pick(a, v0, Nm)), T_END)
        mr = v.code == '0' ? NaN :
             reg(pick(a, v, Nm), "err_l2", "r3", τm) /
             reg(pick(a, v0, Nm), "err_l2", "r3", τm)
        tnote(τ) = isfinite(τ) && τ < T_END - 1e-6 ? " (t=$(fmtt(τ)))" : ""
        push!(rows, [a.label, v.label, state,
                     length(have) >= 2 ?
                     "$(hlabel(have[end-1]))→$(hlabel(have[end]))" * tnote(τf) : "—",
                     fmto(p3), fmto(p5), fmto(pg),
                     v.code == '0' ? "1" :
                     fmtr(mr) * (isfinite(mr) ? " @$(hlabel(Nm))" * tnote(τm) : ""),
                     asym, fmtr(R), fmtr(growth(rf))])
        cells[(a.code, v.code)] = "$(state); p $(fmto(p3))/$(fmto(p5))/$(fmto(pg))" * tnote(τf) *
                                  (v.code == '0' ? "" : "; ×$(fmtr(mr))") * "; from $asym"
    end
    table(["a \\ v", [v.label for v in SPEEDS]...],
          [vcat([a.label], [get(cells, (a.code, v.code), "not in the campaign")
                            for v in SPEEDS]) for a in SPINS])
    println("Each cell: state; `p` in `r3`/`r5`/global at the finest pair; `×` moving ",
            "over rest in `r3`; the coarsest asymptotic `h`.\n")
    table(["a", "v", "state", "finest pair", "p r3", "p r5", "p global", "mov/rest r3",
           "asymptotic from", "R", "growth 20/10"], rows)
    return nothing
end

main()
