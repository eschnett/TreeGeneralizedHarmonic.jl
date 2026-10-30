# The tables of step B3's head-on rows (added in step B3), from the CSVs
# `test/binary_runs.jl` writes: one subdirectory per row under `dir`, each with
# its `binary.csv`. No package needed:
#
#     julia test/binary_study.jl dir=.claude/data/step-B3 runs=b3s48,b3sg48
#
# Prints, per row: the time, both holes' find (1 succeeded), the constraints, the
# layers' residual, both holes' areas, the separations of the tracks and of the
# found origins against Newtonian free fall, the largest prediction error, and
# `ℋ` in the bands about the nearer hole. Then the fall interval by interval
# (amended in step B3's review): the found separation's drop over each row's
# interval against free fall's over the same interval, each hole's track off its
# seed's Newtonian quadratic `c₀ ± ½ (M_B/d²) t²` in M and in cells of `h=`
# (default `1/48`), the track's velocity off the seed's, and — where the CSV has
# them — the source line off the track. A ratio of two drops is a rate over one
# interval, never a window's.
#
# Options: dir=, runs=, h=1/48, M=1 (the pair's total mass, equal holes).

using Printf

const OPTIONS = Dict{String,String}()
for arg in ARGS
    k, v = occursin('=', arg) ? split(arg, '='; limit=2) : (arg, "")
    OPTIONS[String(k)] = String(v)
end
rat(s) = (p = split(s, '/'); length(p) == 2 ? parse(Int, p[1]) / parse(Int, p[2]) :
                            parse(Float64, s))
dir = get(OPTIONS, "dir", joinpath(@__DIR__, "..", ".claude", "data", "step-B3"))
runs = haskey(OPTIONS, "runs") ? split(OPTIONS["runs"], ',') :
       sort(filter(r -> isfile(joinpath(dir, r, "binary.csv")), readdir(dir)))
hcell = rat(get(OPTIONS, "h", "1/48"))
Mtot = rat(get(OPTIONS, "M", "1"))

function readcsv(path)
    lines = filter(!isempty, readlines(path))
    head = split(lines[1], ',')
    rows = [Dict(zip(head, (x == "true" ? 1.0 : x == "false" ? 0.0 :
                            tryparse(Float64, x) === nothing ? NaN : parse(Float64, x)
                            for x in split(l, ',')))) for l in lines[2:end]]
    return head, rows
end

for run in runs
    path = joinpath(dir, run, "binary.csv")
    isfile(path) || (println("$run: no binary.csv"); continue)
    head, rows = readcsv(path)
    isempty(rows) && (println("$run: no rows"); continue)
    println("\n### $run\n")
    println("| t | finds | C_a L2 | C_a L∞ | ℋ L2 | residual | area 1 | area 2 | sep (tracks) | sep (found) | free fall | pred (cells) | ℋ [1, 1.5) | ℋ [2, 3) |")
    println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows
        @printf("| %.2f | %d%d | %.2e | %.2e | %.2e | %.2e | %.5f | %.5f | %.6f | %.6f | %.6f | %.1e | %.2e | %.2e |\n",
                r["t"], r["h1_succ"], r["h2_succ"], r["gauge_l2"], r["gauge_linf"],
                r["ham_l2"], r["residual"], r["h1_area"], r["h2_area"], r["sep_track"],
                r["sep_found"], r["sep_newton"], max(r["h1_pred"], r["h2_pred"]),
                get(r, "h1_r1.0_ham_l2", NaN), get(r, "h1_r2.0_ham_l2", NaN))
    end

    # The fall, interval by interval, and each hole against its seed.
    a = rows[1]
    d0 = a["sep_track"]
    acc = (Mtot / 2) / d0^2           # each hole's Newtonian acceleration at t = 0
    c1, c2 = a["h1_cx"], a["h2_cx"]
    s1 = c1 < c2 ? 1 : -1             # hole 1 falls toward +x if it is on the left
    hassrc = haskey(a, "h1_srcx")
    println("\n| t | Δsep found | Δ free fall | ratio over the interval | track 1 − seed | (cells) | track 2 − seed | v₁ − v_seed | residual |",
            hassrc ? " source 1 − track 1 (cells) | source v₁ − track v₁ |" : "")
    println("|---|---|---|---|---|---|---|---|---|", hassrc ? "---|---|" : "")
    for k in eachindex(rows)
        r = rows[k]
        t = r["t"]
        seed1 = c1 + s1 * acc * t^2 / 2
        seed2 = c2 - s1 * acc * t^2 / 2
        e1, e2 = r["h1_cx"] - seed1, r["h2_cx"] - seed2
        dv = r["h1_vx"] - s1 * acc * t
        if k == 1
            @printf("| %.2f | | | | %.2e | %.2f | %.2e | %.2e | %.2f |", t, e1, abs(e1) / hcell,
                    e2, dv, r["residual"])
        else
            q = rows[k - 1]
            ds = q["sep_found"] - r["sep_found"]
            dn = q["sep_newton"] - r["sep_newton"]
            @printf("| %.2f | %.4f | %.4f | %.2f | %.2e | %.2f | %.2e | %.2e | %.2f |", t, ds, dn,
                    ds / dn, e1, abs(e1) / hcell, e2, dv, r["residual"])
        end
        hassrc && @printf(" %.2f | %.2e |", (r["h1_srcx"] - r["h1_cx"]) / hcell,
                          r["h1_srcvx"] - r["h1_vx"])
        println()
    end

    b = rows[end]
    nsteady = length(rows) > 2 ? (b["wall"] - rows[2]["wall"]) / (b["t"] - rows[2]["t"]) : NaN
    last = length(rows) > 1 ?
           (rows[end - 1]["sep_found"] - b["sep_found"]) /
           (rows[end - 1]["sep_newton"] - b["sep_newton"]) : NaN
    @printf("\nto t = %.2f: every find %s; the found separation fell %.4f, free fall %.4f; over the last interval at %.2f× free fall's rate; %.0f s per M after the first row (%.0f s); %.4g M points\n",
            b["t"], all(r -> r["h1_succ"] == 1 && r["h2_succ"] == 1, rows) ? "succeeded" : "did not succeed",
            a["sep_found"] - b["sep_found"], a["sep_newton"] - b["sep_newton"], last,
            nsteady, a["wall"], b["points"] / 1e6)
end
