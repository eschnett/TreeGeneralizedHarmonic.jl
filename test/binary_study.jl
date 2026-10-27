# The tables of step B3's head-on rows (added in step B3), from the CSVs
# `test/binary_runs.jl` writes: one subdirectory per row under `dir`, each with
# its `binary.csv`. No package needed:
#
#     julia test/binary_study.jl dir=.claude/data/step-B3 runs=b3s48,b3sg48
#
# Prints, per row: the time, both holes' find (1 succeeded), the constraints, the
# layers' residual, both holes' areas, the separations of the tracks and of the
# found origins against Newtonian free fall, the largest prediction error, and
# `ℋ` in the bands about the nearer hole; then the run's end, its cost and its
# fall against free fall.

using Printf

const OPTIONS = Dict{String,String}()
for arg in ARGS
    k, v = occursin('=', arg) ? split(arg, '='; limit=2) : (arg, "")
    OPTIONS[String(k)] = String(v)
end
dir = get(OPTIONS, "dir", joinpath(@__DIR__, "..", ".claude", "data", "step-B3"))
runs = haskey(OPTIONS, "runs") ? split(OPTIONS["runs"], ',') :
       sort(filter(r -> isfile(joinpath(dir, r, "binary.csv")), readdir(dir)))

function readcsv(path)
    lines = filter(!isempty, readlines(path))
    head = split(lines[1], ',')
    rows = [Dict(zip(head, (tryparse(Float64, x) === nothing ? NaN : parse(Float64, x)
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
    a, b = rows[1], rows[end]
    fall = a["sep_track"] - b["sep_track"]
    newton = a["sep_newton"] - b["sep_newton"]
    nsteady = length(rows) > 2 ? (b["wall"] - rows[2]["wall"]) / (b["t"] - rows[2]["t"]) : NaN
    @printf("\nto t = %.2f: every find %s; the tracks fell %.4f against free fall's %.4f (%.2f×); the found origins %.4f; %.0f s per M after the first row (%.0f s); %.4g M points\n",
            b["t"], all(r -> r["h1_succ"] == 1 && r["h2_succ"] == 1, rows) ? "succeeded" : "did not succeed",
            fall, newton, fall / newton, a["sep_found"] - b["sep_found"], nsteady, a["wall"],
            b["points"] / 1e6)
end
