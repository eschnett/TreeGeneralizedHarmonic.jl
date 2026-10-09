# The exterior-constraint study of `test/octant_runs.jl` rows (added
# 2026-10-02): for every row directory under a study directory, the shells'
# norms at the end of the run and their late slopes, the horizon's `M_irr`
# drift from the run's own record, and — for rows named in a convergence
# series — the order between successive resolutions.
#
#     julia test/octant_study.jl out/study [more dirs…] [t_from=8] [at=24] [a=3/5]
#
# A row is a directory holding `octant.csv` (one row per chunk) and
# `records.csv` (the run's record, written at the end). The convergence
# series are given as `series=dA64,dA96,dA128:16,24,32` — the labels and the
# number of cells per unit length of each — and may be repeated. From step X3
# it also prints the drift of `h_tt` at the horizon (the CSV's `drift`) and,
# for an excised hole, the outflow rows.
#
# From step X7: several study directories at once (the rows' labels must
# differ); `at=<t>` reads every row at that time rather than at its end (the
# last row at or before it), so that rows of different lengths compare at one
# time, and the late slopes are over `[t_from, at]`; a restart's repeated rows
# are dropped, keeping the last of each time; `a=<spin>` measures `M_irr`
# against Kerr's `√(r₊/2)` and prints the horizon's `J − a` and its slope;
# `intervals=8,24,40,64` prints the slopes of `J` and `M_irr` over each
# interval; the excision rows carry the frame-dragged axes, their faces and the
# flips (step X6); and a series also gives the orders of `J − a` and of the
# slopes.

using Printf

dirs = [a for a in ARGS if !occursin('=', a)]
opts = Dict{String,Vector{String}}()
for a in ARGS
    occursin('=', a) || continue
    k, v = split(a, '='; limit=2)
    push!(get!(opts, String(k), String[]), String(v))
end
ratnum(s) = (p = split(s, '/'); length(p) == 2 ? parse(Float64, p[1]) / parse(Float64, p[2]) :
                                                  parse(Float64, s))
t_from = parse(Float64, first(get(opts, "t_from", ["8"])))
t_at = haskey(opts, "at") ? parse(Float64, first(opts["at"])) : Inf
a_spin = haskey(opts, "a") ? ratnum(first(opts["a"])) : 0.0
M_irr_kerr = sqrt((1 + sqrt(1 - a_spin^2)) / 2)
intervals = haskey(opts, "intervals") ?
            parse.(Float64, split(first(opts["intervals"]), ',')) : Float64[]

function readcsv(path)
    isfile(path) || return nothing
    l = readlines(path)
    cols = split(l[1], ',')
    rows = [split(x, ',') for x in l[2:end] if !isempty(x)]
    return cols, rows
end
num(s) = (x = tryparse(Float64, s); x === nothing ? NaN : x)

# A restart appends the rows since its checkpoint again: keep the last row of
# every time, in time order.
function dedup(rows)
    bytime = Dict(num(r[1]) => r for r in rows)
    return [bytime[t] for t in sort!(collect(keys(bytime)))]
end

# The least-squares slope of `y` against `t` over `t_from ≤ t ≤ t_to`.
function slope(t, y; t_from=t_from, t_to=t_at)
    k = [i for i in eachindex(t) if t_from ≤ t[i] ≤ t_to && isfinite(y[i])]
    length(k) < 3 && return NaN
    x̄ = sum(t[k]) / length(k); ȳ = sum(y[k]) / length(k)
    return sum((t[k] .- x̄) .* (y[k] .- ȳ)) / sum((t[k] .- x̄) .^ 2)
end

# A row with no data (a run refused at its start) is skipped.
rowdir = Dict{String,String}()
for dir in dirs, d in readdir(dir)
    isfile(joinpath(dir, d, "octant.csv")) &&
        countlines(joinpath(dir, d, "octant.csv")) > 1 || continue
    haskey(rowdir, d) && error("row $d is in two of the directories")
    rowdir[d] = joinpath(dir, d)
end
labels = sort!(collect(keys(rowdir)))
shell_cols = nothing
results = Dict{String,Any}()
for lb in labels
    cols, rows = readcsv(joinpath(rowdir[lb], "octant.csv"))
    rows = dedup(rows)
    t = [num(r[1]) for r in rows]
    # The row at `at` (or the last): the last at or before it.
    iat = something(findlast(≤(t_at + 1e-9), t), 1)
    col(n) = (j = findfirst(==(n), cols); j === nothing ? fill(NaN, length(rows)) :
              [num(r[j]) for r in rows])
    names = [c[8:end] for c in cols if startswith(c, "ham_l2_") && !(c[8:end] in
             ("vol", "bnd")) && !startswith(c[8:end], "L")]
    # The shells of every row, in radial order (rows may carry different
    # sets: `shells=` adds radii); a row without a shell prints NaN there.
    global shell_cols = sort!(union(something(shell_cols, String[]), names);
                              by=nm -> nm == "in" ? -Inf : parse(Float64, nm[2:end]))
    res = Dict{String,Any}("t_end" => t[iat])
    for nm in names, q in ("ham_l2", "mom_l2", "gauge_l2", "err_l2")
        y = col("$(q)_$nm")
        res["$(q)_$nm"] = y[iat]
        res["σ$(q)_$nm"] = slope(t, log.(y))         # a rate, 1/M
    end
    # The gauge drift (step X3): the L∞ of `h_tt`'s change at the horizon, at
    # the end, its largest value and its late rate — GHSO2's excised hole drifted
    # at `0.14/M`.
    dr = col("drift")
    if any(isfinite, dr)
        res["drift"] = dr[iat]
        res["drift_max"] = maximum(filter(isfinite, dr[1:iat]))
        res["σdrift"] = slope(t, log.(dr))
    end
    # The excision rows (step X3): the least normal margin over the run, the
    # most non-finite band points, the closure axes into the excised set, and
    # the faces and inflow-like ones at the end.
    nm = col("excision_normal_min")
    if any(isfinite, nm)
        r = 1:iat
        res["exc"] = (normal_min=minimum(filter(isfinite, nm[r])),
                      nonfinite=maximum(col("excision_band_nonfinite")[r]),
                      into=maximum(col("excision_into")[r]),
                      into_min=minimum(col("excision_into")[r]),
                      faces=col("excision_faces")[iat],
                      inflow=col("excision_inflow")[iat],
                      axis_min=minimum(col("excision_axis_min")[r]),
                      band=col("excision_band")[iat],
                      dragged=col("excision_dragged")[iat],
                      faces_dragged=col("excision_faces_dragged")[iat],
                      flips=maximum(col("excision_flips")[r]))
    end
    rec = readcsv(joinpath(rowdir[lb], "records.csv"))
    if rec !== nothing
        rc, rr = rec
        rr = dedup(rr)
        jt, jm, jj = (findfirst(==(c), rc) for c in ("t", "M_irr", "J"))
        tt = [num(r[jt]) for r in rr]; mm = [num(r[jm]) for r in rr]
        jat = something(findlast(≤(t_at + 1e-9), tt), 1)
        res["M_irr"] = mm[jat]
        res["dM_irr"] = slope(tt, mm)                # per M
        if jj !== nothing
            JJ = [num(r[jj]) for r in rr]
            res["J"] = JJ[jat]
            res["dJ"] = slope(tt, JJ)
            res["finds"] = (count(isfinite, JJ[1:jat]), jat)
            res["intervals"] = [(intervals[i], intervals[i + 1],
                                 slope(tt, JJ; t_from=intervals[i], t_to=intervals[i + 1]),
                                 slope(tt, mm; t_from=intervals[i], t_to=intervals[i + 1]))
                                for i in 1:(length(intervals) - 1)
                                if intervals[i + 1] ≤ min(tt[end], t_at) + 1e-9]
        end
    end
    results[lb] = res
end

for q in ("ham_l2", "err_l2")
    println("\n$q at the end of the run, by shell (and its late rate σ, 1/M)")
    @printf("%-8s %6s", "row", "t")
    for nm in shell_cols; @printf(" %21s", nm); end
    println()
    for lb in labels
        r = results[lb]
        @printf("%-8s %6.1f", lb, r["t_end"])
        for nm in shell_cols
            @printf(" %10.3e (%+8.1e)", get(r, "$(q)_$nm", NaN), get(r, "σ$(q)_$nm", NaN))
        end
        println()
    end
end

println("\nM_irr (against Kerr's $(round(M_irr_kerr; digits=6))) and J − a (a = $a_spin): " *
        "at the end, and their slopes over t ≥ $t_from (per M); finds")
for lb in labels
    r = results[lb]
    haskey(r, "M_irr") || continue
    @printf("%-8s  M_irr − Kerr = %+.3e   dM_irr/dt = %+.3e", lb, r["M_irr"] - M_irr_kerr,
            r["dM_irr"])
    haskey(r, "J") && @printf("   J − a = %+.3e   dJ/dt = %+.3e   found %d of %d", r["J"] - a_spin,
                              r["dJ"], r["finds"]...)
    println()
    for (a, b, dj, dm) in get(r, "intervals", ())
        @printf("          [%g, %g]: dJ/dt = %+.3e   dM_irr/dt = %+.3e\n", a, b, dj, dm)
    end
end

if any(lb -> haskey(results[lb], "drift"), labels)
    println("\nthe drift of h_tt at the horizon (L∞): at the end, its largest value, " *
            "and its rate σ over t ≥ $t_from (1/M)")
    for lb in labels
        r = results[lb]
        haskey(r, "drift") || continue
        @printf("%-8s  %.3e   max %.3e   σ %+.2e\n", lb, r["drift"], r["drift_max"],
                r["σdrift"])
    end
end

if any(lb -> haskey(results[lb], "exc"), labels)
    println("\nexcision rows: least normal margin over the run, most non-finite band " *
            "points, closure axes into the excised set; band points, faces, inflow-like " *
            "faces and the least b/a at the end")
    for lb in labels
        haskey(results[lb], "exc") || continue
        e = results[lb]["exc"]
        @printf("%-8s  normal_min %+.4f  nonfinite %d  into %d–%d   band %d  faces %d  inflow %d  b/a min %.3f",
                lb, e.normal_min, e.nonfinite, e.into_min, e.into, e.band, e.faces, e.inflow,
                e.axis_min)
        isfinite(e.dragged) &&
            @printf("   dragged %d  faces dragged %d  flips %d", e.dragged, e.faces_dragged,
                    e.flips)
        println()
    end
end

for s in get(opts, "series", String[])
    lbs, ns = split(s, ':')
    lbs = split(lbs, ','); ns = parse.(Float64, split(ns, ','))
    all(l -> haskey(results, l), lbs) || continue
    println("\nconvergence order, series $(join(lbs, " → "))")
    for q in ("ham_l2", "mom_l2", "gauge_l2", "err_l2")
        @printf("  %-9s", q)
        for nm in shell_cols
            o = [log(get(results[lbs[i]], "$(q)_$nm", NaN) /
                     get(results[lbs[i + 1]], "$(q)_$nm", NaN)) /
                 log(ns[i + 1] / ns[i]) for i in 1:(length(lbs) - 1)]
            @printf(" %8s %s", nm, join([@sprintf("%5.2f", x) for x in o], "/"))
        end
        println()
    end
    for (key, name, shift) in (("dM_irr", "dM_irr/dt", 0.0), ("J", "J − a", a_spin),
                               ("dJ", "dJ/dt", 0.0))
        all(l -> haskey(results[l], key), lbs) || continue
        d = [results[l][key] - shift for l in lbs]
        o = [log(abs(d[i] / d[i + 1])) / log(ns[i + 1] / ns[i]) for i in 1:(length(lbs) - 1)]
        @printf("  %-9s %s   orders %s\n", name, join([@sprintf("%+.3e", x) for x in d], " "),
                join([@sprintf("%5.2f", x) for x in o], "/"))
    end
end
