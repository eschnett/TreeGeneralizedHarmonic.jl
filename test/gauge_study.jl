# Step B1's runs (added 2026-10-25): the damped harmonic gauge rolled on, on the
# boosted and the resting `a = 0` hole on the quadrant, against the same holes in
# their own Kerr-Schild gauge — `SINGULARITY_HANDLING.md`, "The damped harmonic gauge
# rolled on". It reads `test/octant_runs.jl`'s `octant.csv` of each run directory and
# prints that section's tables. It needs no package:
#
#     julia test/gauge_study.jl dir=.claude/data/step-B1 times=1,5,10,15,20,25,30,40
#
# Options: dir= (one subdirectory per run, named b1<gauge><motion><N>: gauge d (damped
# harmonic) or k (Kerr-Schild), motion m (v = 1/5) or s (at rest), N = 24 or 32, the
# spacing 1/N at the hole; a trailing t is a run whose shells are about the tracked
# horizon's center, `shell_center=track`, which the comparisons prefer); times=;
# shells=r2,r3,r5 (the shell columns' names: [2, 9/4), [3, 5), [5, 8) at a = 0).
using Printf

const OPTS = Dict(String(k) => String(v) for (k, v) in
                  (split(a, '='; limit=2) for a in ARGS if occursin('=', a)))
dir = get(OPTS, "dir", ".claude/data/step-B1")
times = parse.(Float64, split(get(OPTS, "times", "1,5,10,15,20,25,30,40"), ','))
shells = split(get(OPTS, "shells", "r2,r3,r5"), ',')

function readcsv(path)
    lines = readlines(path)
    head = split(lines[1], ',')
    rows = [parse.(Float64, split(l, ',')) for l in lines[2:end] if !isempty(l)]
    # A restart appends: keep the last row of each time.
    bytime = Dict{Float64,Vector{Float64}}()
    for r in rows
        bytime[r[1]] = r
    end
    ts = sort(collect(keys(bytime)))
    return Dict(String(h) => [bytime[t][i] for t in ts] for (i, h) in enumerate(head))
end

runs = Dict{String,Any}()
for name in sort(readdir(dir))
    f = joinpath(dir, name, "octant.csv")
    isfile(f) || continue
    r = readcsv(f)
    isempty(r["t"]) || (runs[name] = r)       # a run that has written no row yet
end
isempty(runs) && error("no runs with an octant.csv under $dir")
at(r, col, t) = (i = findfirst(x -> abs(x - t) < 1e-9, r["t"]);
                 i === nothing || !haskey(r, col) ? NaN : r[col][i])
fmt(x) = isnan(x) ? "—" : @sprintf("%.3g", x)
row(cells) = println("| ", join(cells, " | "), " |")
sep(n) = println("|", repeat("---|", n))

println("## The runs\n")
row(["run", "last t", "wall at last t", "s/M"])
sep(4)
for name in sort(collect(keys(runs)))
    r = runs[name]
    t, w = r["t"][end], r["wall"][end]
    row([name, fmt(t), @sprintf("%.0f s", w), t > 0 ? @sprintf("%.0f", w / t) : "—"])
end

# A table of one column over the runs and times.
function table(title, col; f=(n, x, t) -> x, names=sort(collect(keys(runs))))
    println("\n## ", title, "\n")
    row(vcat(["run"], [@sprintf("%g M", t) for t in times]))
    sep(length(times) + 1)
    for n in names
        haskey(runs, n) || continue
        row(vcat([n], [fmt(f(n, at(runs[n], col, t), t)) for t in times]))
    end
end
# Damped harmonic over Kerr-Schild, the same motion and spacing.
function ratio_table(title, col)
    println("\n## ", title, "\n")
    row(vcat(["rows"], [@sprintf("%g M", t) for t in times]))
    sep(length(times) + 1)
    for m in ("m", "s"), N in ("24", "32")
        # The moving damped rows' shells about the tracked center (`b1dm<N>t`,
        # `shell_center=track`) where they exist.
        d, k = haskey(runs, "b1d$m$(N)t") ? "b1d$m$(N)t" : "b1d$m$N", "b1k$m$N"
        (haskey(runs, d) && haskey(runs, k)) || continue
        row(vcat(["$d / $k"], [fmt(at(runs[d], col, t) / at(runs[k], col, t)) for t in times]))
    end
end
# The order from 1/24 to 1/32 of one column.
function order_table(title, col)
    println("\n## ", title, "\n")
    row(vcat(["rows"], [@sprintf("%g M", t) for t in times]))
    sep(length(times) + 1)
    for g in ("d", "k"), m in ("m", "s")
        a, b = "b1$g$(m)24", "b1$g$(m)32"
        if haskey(runs, a * "t") && haskey(runs, b * "t")
            a, b = a * "t", b * "t"
        end
        (haskey(runs, a) && haskey(runs, b)) || continue
        row(vcat(["$a → $b"], [fmt(log(at(runs[a], col, t) / at(runs[b], col, t)) /
                                   log(32 / 24)) for t in times]))
    end
end

for s in shells
    table("ℋ (point-weighted L2) in the shell $s", "ham_l2_$s")
    table("C_a (the largest component's L2) in the shell $s", "gauge_l2_$s")
end
for s in shells
    ratio_table("ℋ in $s, damped harmonic over Kerr-Schild", "ham_l2_$s")
    ratio_table("C_a in $s, damped harmonic over Kerr-Schild", "gauge_l2_$s")
end
for s in shells
    order_table("The order of ℋ in $s from 1/24 to 1/32", "ham_l2_$s")
end
table("C_a over the whole evolved region", "gauge_l2")
table("C_a in the blocks touching an outer face", "gauge_l2_bnd")
table("ℋ in the blocks touching an outer face", "ham_l2_bnd")
table("M_irr − 1 (the horizon's, Kerr's is 1)", "hz_M_irr"; f=(n, x, t) -> x - 1)
table("The found horizon's x minus the analytic center's", "hz_x";
      f=(n, x, t) -> x - at(runs[n], "c_x", t))
table("The horizon's x/y extent over √(1 − v²) (1 at rest; v = 1/5 where the center moves)", "hz_ratio";
      f=(n, x, t) -> x / (runs[n]["c_x"][end] != runs[n]["c_x"][1] ? sqrt(1 - 1 / 25) : 1.0))
table("The error against the Kerr-Schild solution in r2 (the gauge's distance from it)",
      "err_l2_r2")

# The horizon's coordinate radii about its own origin, from the run's own record
# (`records.csv`, written at the end of a run): under the damped harmonic gauge the
# horizon's coordinate size changes, and a fixed shell may straddle it.
recs = Dict{String,Any}()
for name in sort(readdir(dir))
    f = joinpath(dir, name, "records.csv")
    isfile(f) || continue
    lines = readlines(f)
    head = split(lines[1], ',')
    col(c) = findfirst(==(c), head)
    rows = [split(l, ',') for l in lines[2:end] if !isempty(l)]
    num(s) = something(tryparse(Float64, s), NaN)
    recs[name] = Dict(c => [num(r[col(c)]) for r in rows]
                      for c in ("t", "r_min", "r_mean", "r_max", "bounds_hits"))
end
function rec_table(title, c)
    println("\n## ", title, "\n")
    row(vcat(["run"], [@sprintf("%g M", t) for t in times]))
    sep(length(times) + 1)
    for n in sort(collect(keys(recs)))
        r = recs[n]
        row(vcat([n], [fmt((i = findfirst(x -> abs(x - t) < 1e-9, r["t"]);
                            i === nothing ? NaN : r[c][i])) for t in times]))
    end
end
isempty(recs) || (rec_table("The horizon's least coordinate radius about its origin", "r_min");
                  rec_table("Its mean coordinate radius", "r_mean");
                  rec_table("The range projection's hits", "bounds_hits"))
