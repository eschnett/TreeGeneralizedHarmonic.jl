# Step B2's runs (added 2026-10-26): the tracked analytic target on the boosted
# `a = 0` hole on the quadrant against the analytic `:damped` layer, its sensitivity
# to the target's mass, velocity and center, and the recovery of a track seeded too
# fast — `SINGULARITY_HANDLING.md`, "The tracked analytic target". It reads
# `test/octant_runs.jl`'s `octant.csv` and `records.csv` of each run directory and
# prints that section's tables. It needs no package:
#
#     julia test/tracked_study.jl dir=.claude/data/step-B2 times=1,4,8,16,24,32,40
#
# Options: dir= (one subdirectory per run: b2a<N> the analytic layer, b2t<N> the
# tracked target, and at N = 24 the perturbations b2mp24, b2mm24 (the target's mass
# ±1 %), b2vp24, b2vm24 (its velocity ±0.02), b2xp24, b2xm24 (its center ±1 cell) and
# b2r24 (the seed track 0.05 too fast); N the spacing 1/N at the hole); times=;
# shells=r2,r3,r5 (the shell columns' names: [2, 9/4), [3, 5), [5, 8)).
using Printf

const OPTS = Dict(String(k) => String(v) for (k, v) in
                  (split(a, '='; limit=2) for a in ARGS if occursin('=', a)))
dir = get(OPTS, "dir", ".claude/data/step-B2")
times = parse.(Float64, split(get(OPTS, "times", "1,4,8,16,24,32,40"), ','))
shells = split(get(OPTS, "shells", "r2,r3,r5"), ',')

num(s) = something(tryparse(Float64, s), NaN)
function readcsv(path)
    lines = readlines(path)
    head = split(lines[1], ',')
    rows = [num.(split(l, ',')) for l in lines[2:end] if !isempty(l)]
    # A restart appends: keep the last row of each time.
    bytime = Dict{Float64,Vector{Float64}}()
    for r in rows
        bytime[r[1]] = r
    end
    ts = sort(collect(keys(bytime)))
    return Dict(String(h) => [bytime[t][i] for t in ts] for (i, h) in enumerate(head))
end

runs = Dict{String,Any}()
recs = Dict{String,Any}()
for name in sort(readdir(dir))
    f = joinpath(dir, name, "octant.csv")
    isfile(f) || continue
    r = readcsv(f)
    isempty(r["t"]) || (runs[name] = r)
    g = joinpath(dir, name, "records.csv")
    isfile(g) && (recs[name] = readcsv(g))
end
isempty(runs) && error("no runs with an octant.csv under $dir")
at(r, col, t) = (i = findfirst(x -> abs(x - t) < 1e-9, r["t"]);
                 i === nothing || !haskey(r, col) ? NaN : r[col][i])
fmt(x) = isnan(x) ? "—" : @sprintf("%.3g", x)
row(cells) = println("| ", join(cells, " | "), " |")
sep(n) = println("|", repeat("---|", n))
names_in(pre) = sort(filter(n -> startswith(n, pre), collect(keys(runs))))

println("## The runs\n")
row(["run", "last t", "wall at last t", "s/M", "finds", "projection hits"])
sep(6)
for name in sort(collect(keys(runs)))
    r = runs[name]
    t, w = r["t"][end], r["wall"][end]
    rc = get(recs, name, nothing)
    nf = rc === nothing || !haskey(rc, "track_finds") ? NaN :
         sum(x -> isnan(x) ? 0.0 : x, rc["track_finds"])
    hits = rc === nothing ? NaN : sum(x -> isnan(x) ? 0.0 : x, rc["bounds_hits"])
    row([name, fmt(t), @sprintf("%.0f s", w), t > 0 ? @sprintf("%.0f", w / t) : "—",
         fmt(nf), fmt(hits)])
end

function table(title, col; f=(n, x, t) -> x, names=sort(collect(keys(runs))), src=runs)
    println("\n## ", title, "\n")
    row(vcat(["run"], [@sprintf("%g M", t) for t in times]))
    sep(length(times) + 1)
    for n in names
        haskey(src, n) || continue
        row(vcat([n], [fmt(f(n, at(src[n], col, t), t)) for t in times]))
    end
end
function ratio_table(title, col, pairs)
    println("\n## ", title, "\n")
    row(vcat(["rows"], [@sprintf("%g M", t) for t in times]))
    sep(length(times) + 1)
    for (a, b) in pairs
        (haskey(runs, a) && haskey(runs, b)) || continue
        row(vcat(["$a / $b"], [fmt(at(runs[a], col, t) / at(runs[b], col, t)) for t in times]))
    end
end
function order_table(title, col)
    println("\n## ", title, "\n")
    row(vcat(["rows"], [@sprintf("%g M", t) for t in times]))
    sep(length(times) + 1)
    for k in ("a", "t")
        a, b = "b2$(k)24", "b2$(k)32"
        (haskey(runs, a) && haskey(runs, b)) || continue
        row(vcat(["$a → $b"], [fmt(log(at(runs[a], col, t) / at(runs[b], col, t)) /
                                   log(32 / 24)) for t in times]))
    end
end

main = ["b2a24", "b2t24", "b2a32", "b2t32"]
for s in shells
    table("ℋ (point-weighted L2) in the shell $s", "ham_l2_$s"; names=main)
    table("The error against Kerr-Schild in the shell $s", "err_l2_$s"; names=main)
end
for s in shells
    ratio_table("ℋ in $s, the tracked target over the analytic layer", "ham_l2_$s",
                [("b2t24", "b2a24"), ("b2t32", "b2a32")])
    ratio_table("The error in $s, the tracked target over the analytic layer",
                "err_l2_$s", [("b2t24", "b2a24"), ("b2t32", "b2a32")])
end
for s in shells
    order_table("The order of ℋ in $s from 1/24 to 1/32", "ham_l2_$s")
    order_table("The order of the error in $s from 1/24 to 1/32", "err_l2_$s")
end
table("The layer residual (its L∞ distance from Kerr-Schild)", "residual")
table("ℋ over the evolved region", "ham_l2"; names=main)
table("The error over the evolved region", "err_l2"; names=main)

pert = ["b2mp24", "b2mm24", "b2vp24", "b2vm24", "b2xp24", "b2xm24", "b2r24"]
for s in shells
    ratio_table("ℋ in $s, each perturbation over the tracked target", "ham_l2_$s",
                [(p, "b2t24") for p in pert])
    ratio_table("The error in $s, each perturbation over the tracked target", "err_l2_$s",
                [(p, "b2t24") for p in pert])
end
ratio_table("The layer residual, each perturbation over the tracked target", "residual",
            [(p, "b2t24") for p in pert])
ratio_table("The error over the evolved region, each perturbation over the tracked target",
            "err_l2", [(p, "b2t24") for p in pert])

tracked = sort(filter(n -> haskey(recs[n], "track_offset") &&
                           any(!isnan, recs[n]["track_offset"]), collect(keys(recs))))
table("The track's center off the analytic one, in finest cells (track_offset)",
      "track_offset"; names=tracked, src=recs)
table("The largest prediction error of the chunk's finds, in finest cells",
      "track_prediction_max"; names=tracked, src=recs)
table("The track's velocity along x (the hole's is 0.2)", "track_v"; names=tracked,
      src=recs)
table("M_irr − 1 (the horizon's, Kerr's is 1)", "M_irr"; f=(n, x, t) -> x - 1,
      src=recs)
if haskey(recs, "b2r24")
    r = recs["b2r24"]
    println("\n## The recovery run, chunk by chunk to 8 M\n")
    row(["t", "track_v", "track_offset (cells)", "prediction (cells)", "finds"])
    sep(5)
    for (i, t) in enumerate(r["t"])
        t ≤ 8 || break
        row([fmt(t), fmt(r["track_v"][i]), fmt(r["track_offset"][i]),
             fmt(r["track_prediction_max"][i]), fmt(r["track_finds"][i])])
    end
end
