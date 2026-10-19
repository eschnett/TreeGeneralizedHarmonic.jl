# Compare the CSV files of two `test/octant_runs.jl` runs of the same case
# (added 2026-10-19): the serial run against the same run over MPI.
#
#     julia bench/octant_compare.jl <reference dir> <run dir> [<run dir> …]
#
# Each directory holds `octant.csv` and `records.csv`. A column whose name
# says it is an L2 norm (`l2` anywhere in it) is a sum, which a partition
# reassociates, and is compared to roundoff; `wall` is the wall clock and is
# skipped; every other column — the maxima, the counts, the horizon's
# numbers, the track — must be the same text. Prints one line per run and
# exits 1 if any differs. Base only.

function readcsv(path)
    lines = readlines(path)
    head = split(lines[1], ',')
    return head, [split(l, ',') for l in lines[2:end]]
end

# The records' sums: the L2 norms by name.
issum(name) = occursin("l2", name)

function compare(refdir, dir)
    worst = 0.0
    bad = String[]
    for file in ("octant.csv", "records.csv")
        h1, r1 = readcsv(joinpath(refdir, file))
        h2, r2 = readcsv(joinpath(dir, file))
        h1 == h2 || (push!(bad, "$file: other columns"); continue)
        length(r1) == length(r2) ||
            (push!(bad, "$file: $(length(r1)) rows against $(length(r2))"); continue)
        for (i, (a, b)) in enumerate(zip(r1, r2)), (j, name) in enumerate(h1)
            name == "wall" && continue
            a[j] == b[j] && continue
            x, y = tryparse(Float64, a[j]), tryparse(Float64, b[j])
            if issum(name) && x !== nothing && y !== nothing
                rel = abs(x - y) / max(abs(x), abs(y), floatmin())
                worst = max(worst, rel)
                rel ≤ 1e-9 && continue
            end
            push!(bad, "$file row $i $name: $(a[j]) against $(b[j])")
        end
    end
    return bad, worst
end

function main(refdir, dirs...)
    ok = true
    for dir in dirs
        bad, worst = compare(refdir, dir)
        println(basename(dir), ": ", isempty(bad) ? "same" : "DIFFERS",
                " (largest relative difference of a sum ", worst, ")")
        foreach(l -> println("    ", l), first(bad, 10))
        ok &= isempty(bad)
    end
    ok || exit(1)
    return nothing
end

main(ARGS...)
