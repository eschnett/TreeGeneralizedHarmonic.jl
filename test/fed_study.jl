# The tables of step B3's review (added 2026-10-26): one moving hole whose gauge
# source is fed by its own track — `test/octant_runs.jl interior=tracked
# source=composite feed=…` — against step B2's tracked target on the analytic
# source (`b2t24`). Reads each run's `octant.csv`; no package needed:
#
#     julia test/fed_study.jl dir=.claude/data/step-B3 base=<B2's .claude/data/step-B2>/b2t24 runs=b3fa24,b3fb24
#
# Prints, at `times=` (default `1,2,4,8,12,16,20`): `ℋ` in the shell `[2, 9/4)`
# just outside the horizon and its ratio to the base row's, the layer residual
# against Kerr-Schild (`residual`, the base row's measure) and against the layer's
# own target (`res_trk`), the error against the hole on its track's line
# (`err_l2_trk`), and the lags along x: the track off the analytic line
# `x0 + v t`, the source line off the track, in cells of `h=` (default `1/24`),
# and the track's and the line's velocities off `v=` (default `1/5`).
#
# Options: dir=, base=, runs=, times=, h=1/24, v=1/5, x0=-4.

using Printf

const OPTIONS = Dict{String,String}()
for arg in ARGS
    k, val = occursin('=', arg) ? split(arg, '='; limit=2) : (arg, "")
    OPTIONS[String(k)] = String(val)
end
rat(s) = (p = split(s, '/'); length(p) == 2 ? parse(Int, p[1]) / parse(Int, p[2]) :
                            parse(Float64, s))
dir = get(OPTIONS, "dir", joinpath(@__DIR__, "..", ".claude", "data", "step-B3"))
base = get(OPTIONS, "base", "")
runs = haskey(OPTIONS, "runs") ? split(OPTIONS["runs"], ',') :
       sort(filter(r -> startswith(r, "b3f") && isfile(joinpath(dir, r, "octant.csv")),
                   readdir(dir)))
times = [rat(x) for x in split(get(OPTIONS, "times", "1,2,4,8,12,16,20"), ',')]
hcell = rat(get(OPTIONS, "h", "1/24"))
vh = rat(get(OPTIONS, "v", "1/5"))
x0 = rat(get(OPTIONS, "x0", "-4"))

function readcsv(path)
    lines = filter(!isempty, readlines(path))
    head = split(lines[1], ',')
    rows = [Dict(zip(head, (x == "true" ? 1.0 : x == "false" ? 0.0 :
                            tryparse(Float64, x) === nothing ? NaN : parse(Float64, x)
                            for x in split(l, ',')))) for l in lines[2:end]]
    return rows
end
# The last row at each time asked for (a restart appends; the newest wins).
at(rows, t) = (k = findlast(r -> abs(r["t"] - t) < 1e-9, rows); k === nothing ? nothing : rows[k])
fmt(x) = x === nothing || !isfinite(x) ? "—" : @sprintf("%.3g", x)

baserows = isempty(base) ? nothing : readcsv(joinpath(base, "octant.csv"))

function table(title, f)
    println("\n#### $title\n")
    println("| run | ", join((@sprintf("%g M", t) for t in times), " | "), " |")
    println("|---|", repeat("---|", length(times)))
    if baserows !== nothing && f(nothing, nothing) !== :skipbase
        println("| base | ", join((fmt(f(at(baserows, t), nothing)) for t in times), " | "), " |")
    end
    for run in runs
        path = joinpath(dir, run, "octant.csv")
        isfile(path) || continue
        rows = readcsv(path)
        vals = String[]
        for t in times
            r = at(rows, t)
            b = baserows === nothing ? nothing : at(baserows, t)
            push!(vals, r === nothing ? "—" : fmt(f(r, b)))
        end
        println("| $run | ", join(vals, " | "), " |")
    end
end

get_(r, k) = r === nothing ? nothing : get(r, k, NaN)
table("ℋ (point-weighted L2) in the shell [2, 9/4)", (r, b) -> get_(r, "ham_l2_r2"))
table("ℋ in [2, 9/4) over the base row's", (r, b) -> r === nothing && b === nothing ? :skipbase :
                                                   r === nothing || b === nothing ? nothing :
                                                   r["ham_l2_r2"] / b["ham_l2_r2"])
table("ℋ in [3, 5) over the base row's", (r, b) -> r === nothing && b === nothing ? :skipbase :
                                                 r === nothing || b === nothing ? nothing :
                                                 r["ham_l2_r3"] / b["ham_l2_r3"])
table("The layer residual against Kerr-Schild (residual)", (r, b) -> get_(r, "residual"))
table("The layer residual against its own target (res_trk)",
      (r, b) -> r === nothing && b === nothing ? :skipbase : get_(r, "res_trk"))
table("The error against the hole on the track's line (err_l2_trk)",
      (r, b) -> r === nothing && b === nothing ? :skipbase : get_(r, "err_l2_trk"))
table("The track off the analytic line x0 + v t, in cells",
      (r, b) -> r === nothing && b === nothing ? :skipbase : r === nothing ? nothing :
                (r["track_c"] - x0 - vh * r["t"]) / hcell)
table("The source line off the track, in cells",
      (r, b) -> r === nothing && b === nothing ? :skipbase : r === nothing ? nothing :
                (r["src_c"] - r["track_c"]) / hcell)
table("The track's velocity off v",
      (r, b) -> r === nothing && b === nothing ? :skipbase : r === nothing ? nothing :
                r["track_v"] - vh)
table("The source line's velocity off v",
      (r, b) -> r === nothing && b === nothing ? :skipbase : r === nothing ? nothing :
                r["src_v"] - vh)
table("The source line's acceleration",
      (r, b) -> r === nothing && b === nothing ? :skipbase : get_(r, "src_a"))
for run in runs
    path = joinpath(dir, run, "octant.csv")
    isfile(path) || continue
    rows = readcsv(path)
    b = rows[end]
    @printf("\n%s: to t = %g, %.0f s at the last row\n", run, b["t"], b["wall"])
end
