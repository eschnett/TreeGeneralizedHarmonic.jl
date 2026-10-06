# The exterior-constraint study of `test/octant_runs.jl` rows (added
# 2026-10-02): for every row directory under a study directory, the shells'
# norms at the end of the run and their late slopes, the horizon's `M_irr`
# drift from the run's own record, and — for rows named in a convergence
# series — the order between successive resolutions.
#
#     julia test/octant_study.jl out/study [t_from=8]
#
# A row is a directory holding `octant.csv` (one row per chunk) and
# `records.csv` (the run's record, written at the end). The convergence
# series are given as `series=dA64,dA96,dA128:16,24,32` — the labels and the
# number of cells per unit length of each — and may be repeated. From step X3
# it also prints the drift of `h_tt` at the horizon (the CSV's `drift`) and,
# for an excised hole, the outflow rows.

using Printf

dir = ARGS[1]
opts = Dict{String,Vector{String}}()
for a in ARGS[2:end]
    occursin('=', a) || continue
    k, v = split(a, '='; limit=2)
    push!(get!(opts, String(k), String[]), String(v))
end
t_from = parse(Float64, first(get(opts, "t_from", ["8"])))

function readcsv(path)
    isfile(path) || return nothing
    l = readlines(path)
    cols = split(l[1], ',')
    rows = [split(x, ',') for x in l[2:end] if !isempty(x)]
    return cols, rows
end
num(s) = (x = tryparse(Float64, s); x === nothing ? NaN : x)

# The least-squares slope of `y` against `t` over `t ≥ t_from`.
function slope(t, y)
    k = [i for i in eachindex(t) if t[i] ≥ t_from && isfinite(y[i])]
    length(k) < 3 && return NaN
    x̄ = sum(t[k]) / length(k); ȳ = sum(y[k]) / length(k)
    return sum((t[k] .- x̄) .* (y[k] .- ȳ)) / sum((t[k] .- x̄) .^ 2)
end

# A row with no data (a run refused at its start) is skipped.
labels = sort!([d for d in readdir(dir) if isfile(joinpath(dir, d, "octant.csv")) &&
                countlines(joinpath(dir, d, "octant.csv")) > 1])
shell_cols = nothing
results = Dict{String,Any}()
for lb in labels
    cols, rows = readcsv(joinpath(dir, lb, "octant.csv"))
    t = [num(r[1]) for r in rows]
    col(n) = (j = findfirst(==(n), cols); j === nothing ? fill(NaN, length(rows)) :
              [num(r[j]) for r in rows])
    names = [c[8:end] for c in cols if startswith(c, "ham_l2_") && !(c[8:end] in
             ("vol", "bnd")) && !startswith(c[8:end], "L")]
    global shell_cols = names
    res = Dict{String,Any}("t_end" => t[end])
    for nm in names, q in ("ham_l2", "mom_l2", "gauge_l2", "err_l2")
        y = col("$(q)_$nm")
        res["$(q)_$nm"] = y[end]
        res["σ$(q)_$nm"] = slope(t, log.(y))         # a rate, 1/M
    end
    # The gauge drift (step X3): the L∞ of `h_tt`'s change at the horizon, at
    # the end, its largest value and its late rate — GHSO2's excised hole drifted
    # at `0.14/M`.
    dr = col("drift")
    if any(isfinite, dr)
        res["drift"] = dr[end]
        res["drift_max"] = maximum(filter(isfinite, dr))
        res["σdrift"] = slope(t, log.(dr))
    end
    # The excision rows (step X3): the least normal margin over the run, the
    # most non-finite band points, the closure axes into the excised set, and
    # the faces and inflow-like ones at the end.
    nm = col("excision_normal_min")
    if any(isfinite, nm)
        res["exc"] = (normal_min=minimum(filter(isfinite, nm)),
                      nonfinite=maximum(col("excision_band_nonfinite")),
                      into=maximum(col("excision_into")),
                      faces=col("excision_faces")[end],
                      inflow=col("excision_inflow")[end],
                      axis_min=minimum(col("excision_axis_min")),
                      band=col("excision_band")[end])
    end
    rec = readcsv(joinpath(dir, lb, "records.csv"))
    if rec !== nothing
        rc, rr = rec
        jt, jm = findfirst(==("t"), rc), findfirst(==("M_irr"), rc)
        tt = [num(r[jt]) for r in rr]; mm = [num(r[jm]) for r in rr]
        res["M_irr"] = mm[end]
        res["dM_irr"] = slope(tt, mm)                # per M
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
            @printf(" %10.3e (%+8.1e)", r["$(q)_$nm"], r["σ$(q)_$nm"])
        end
        println()
    end
end

println("\nM_irr: at the end, and its slope over t ≥ $t_from (per M)")
for lb in labels
    r = results[lb]
    haskey(r, "M_irr") || continue
    @printf("%-8s  M_irr − 1 = %+.3e   dM_irr/dt = %+.3e\n", lb, r["M_irr"] - 1, r["dM_irr"])
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
        @printf("%-8s  normal_min %+.4f  nonfinite %d  into %d   band %d  faces %d  inflow %d  b/a min %.3f\n",
                lb, e.normal_min, e.nonfinite, e.into, e.band, e.faces, e.inflow,
                e.axis_min)
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
            o = [log(results[lbs[i]]["$(q)_$nm"] / results[lbs[i + 1]]["$(q)_$nm"]) /
                 log(ns[i + 1] / ns[i]) for i in 1:(length(lbs) - 1)]
            @printf(" %8s %s", nm, join([@sprintf("%5.2f", x) for x in o], "/"))
        end
        println()
    end
    if all(l -> haskey(results[l], "dM_irr"), lbs)
        d = [results[l]["dM_irr"] for l in lbs]
        o = [log(abs(d[i] / d[i + 1])) / log(ns[i + 1] / ns[i]) for i in 1:(length(lbs) - 1)]
        @printf("  dM_irr/dt %s   orders %s\n", join([@sprintf("%+.3e", x) for x in d], " "),
                join([@sprintf("%5.2f", x) for x in o], "/"))
    end
end
