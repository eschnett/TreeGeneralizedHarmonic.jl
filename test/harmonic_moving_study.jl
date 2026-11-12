# The moving harmonic hole (added 2026-11-12, step H2): `Harmonic(1, 0)`
# boosted along `+x` on the quadrant `[−64, 64] × [0, 64]²`, at `v = 1/5` from
# `x = −4` for `40 M` and at `v = 1/2` from `x = −5` for `20 M`, with no gauge
# source, against the same hole at rest on the same mesh. Every row is a
# `test/octant_runs.jl` run; this script reads their `octant.csv` (one row per
# chunk, the shells about the hole's current center `c(t)`, each shell's
# trailing and leading half, the horizon's columns, and on the following mesh
# its blocks) and prints the Markdown tables of `SINGULARITY_HANDLING.md`'s "A
# moving harmonic hole (step H2)". It needs no package.
#
#     julia test/harmonic_moving_study.jl [dir=<data>] [times=1,4,8,16,20,24,32,40]
#                                         [shells=r1,r1.25,r2,r4]
#
# `dir` (default `.claude/data/harm-h2`) holds one directory per row, `h = 1/N`:
#
#   qd<N>, qd<N>s       `:damped` (`r_0 = 2/5`, `r_1 = 3/4`) on the capsule,
#                       moving at `1/5` and at rest
#   qf<N>, qf<N>s       `:fitted` (`m = 16`, `n_L = 20`, `fit_cont = 2`) the same
#   qf<N>t              `:fitted`, moving, with `trail_ramp = 9/10`
#   qd<N>f, qf<N>f      moving on the following mesh
#   qd<N>v2, qd<N>v2s   `:damped` (`r_1 = 7/10`) at `v = 1/2`, and at rest
#                       (`1/32` and `1/40`)
#   qf<N>v2f            `:fitted` at `v = 1/2` on the following mesh
#
# and `rest=` (default `.claude/data/harm-h1`) step H1's resting rows on the
# octant, `hd<N>` and `hf<N>`, which the moving rows on the following mesh are
# compared with where no resting row of their own exists.

using Printf

opts = Dict(String(k) => String(v)
            for (k, v) in (split(a, '='; limit=2) for a in ARGS if occursin('=', a)))
root = normpath(joinpath(@__DIR__, ".."))
datadir(name) = first(filter(isdir, [joinpath(root, ".claude", "data", name),
                                     joinpath(root, "..", "..", "data", name), "."]))
const DIR = normpath(abspath(get(opts, "dir", datadir("harm-h2"))))
const REST = normpath(abspath(get(opts, "rest", datadir("harm-h1"))))
const TIMES = parse.(Float64, split(get(opts, "times", "1,4,8,16,20,24,32,40"), ','))
const SHELLS = split(get(opts, "shells", "r1,r1.25,r2,r4"), ',')
const SHELL_RANGE = Dict("in" => "inside", "r1" => "[1, 5/4)", "r1.25" => "[5/4, 2)",
                         "r2" => "[2, 4)", "r4" => "[4, 7)", "r7" => "≥ 7")

# --- reading -------------------------------------------------------------------

struct Run
    name::String
    cols::Dict{String,Vector{Float64}}
end

function readrun(path, name)
    f = joinpath(path, "octant.csv")
    isfile(f) || return nothing
    lines = readlines(f)
    length(lines) ≥ 2 || return nothing
    hdr = split(lines[1], ',')
    cols = Dict(String(h) => Float64[] for h in hdr)
    for l in lines[2:end]
        v = split(l, ',')
        length(v) == length(hdr) || continue
        for (h, x) in zip(hdr, v)
            push!(cols[String(h)], something(tryparse(Float64, x), NaN))
        end
    end
    # A restart appends from its checkpoint, so a time may repeat: keep the last.
    t = cols["t"]
    keep = [i for i in eachindex(t) if i == lastindex(t) || t[i + 1] > t[i]]
    seen = Dict{Float64,Int}()
    for i in keep
        seen[t[i]] = i
    end
    idx = sort!(collect(values(seen)); by=i -> t[i])
    return Run(name, Dict(k => v[idx] for (k, v) in cols))
end

const RUNS = Dict{String,Run}()
for d in (isdir(DIR) ? readdir(DIR) : String[])
    # A failed attempt is kept beside its row as `<row>.fail<k>`; it is not a row.
    (isdir(joinpath(DIR, d)) && !occursin('.', d)) || continue
    r = readrun(joinpath(DIR, d), d)
    r === nothing || (RUNS[d] = r)
end
for d in (isdir(REST) ? readdir(REST) : String[])
    occursin(r"^h[df]\d+$", d) || continue
    r = readrun(joinpath(REST, d), d)
    r === nothing || (RUNS[d] = r)
end

has(n) = haskey(RUNS, n)
tmax(r::Run) = r.cols["t"][end]
function at(r::Run, col, t)
    haskey(r.cols, col) || return NaN
    i = findfirst(==(t), r.cols["t"])
    return i === nothing ? NaN : r.cols[col][i]
end
at(n::AbstractString, col, t) = has(n) ? at(RUNS[n], col, t) : NaN

e(x) = isnan(x) ? "—" : @sprintf("%.2e", x)
f2(x) = isnan(x) ? "—" : @sprintf("%.2f", x)
tt(t) = isinteger(t) ? string(Int(t)) : @sprintf("%g", t)
f3(x) = isnan(x) ? "—" : @sprintf("%.3f", x)
order(a, b, ha, hb) = log(a / b) / log(ha / hb)
hof(n) = 1 / parse(Int, match(r"^[qh][df](\d+)", n).captures[1])

# The cost of a row: the wall clock per M over the run, from its last restart's
# first row to its last row (a restart's compilation is not counted), and on the
# following mesh the blocks.
function cost(r::Run)
    t, w = r.cols["t"], r.cols["wall"]
    i = findlast(k -> k > 1 && w[k] < w[k - 1], eachindex(w))
    i = i === nothing ? 2 : i + 1
    i ≤ length(t) || return NaN
    return (w[end] - w[i - 1]) / (t[end] - t[i - 1])
end

# --- the tables -------------------------------------------------------------------

println("## Rows\n")
println("| row | `h` | t reached | s/M | blocks (first … last) |")
println("|---|---|---|---|---|")
for n in sort!(collect(keys(RUNS)))
    r = RUNS[n]
    b = haskey(r.cols, "blocks") ? "$(Int(r.cols["blocks"][1])) … $(Int(r.cols["blocks"][end]))" : "capsule"
    startswith(n, "h") && (b = "octant")
    println("| `$n` | 1/$(round(Int, 1 / hof(n))) | $(tmax(r)) | $(@sprintf("%.0f", cost(r))) | $b |")
end

# Moving against resting, per shell: ℋ and the error, and their ratio.
function versus(mv, rs, title; times=TIMES)
    (has(mv) && has(rs)) || return
    println("\n### $title: `$mv` against `$rs`\n")
    println("| t | " * join(("ℋ $(SHELL_RANGE[s]) | err $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " |")
    println("|---|" * repeat("---|---|", length(SHELLS)))
    for t in times
        t ≤ tmax(RUNS[mv]) || continue
        cells = String[]
        for s in SHELLS
            hm, hr = at(mv, "ham_l2_$s", t), at(rs, "ham_l2_$s", t)
            em, er = at(mv, "err_l2_$s", t), at(rs, "err_l2_$s", t)
            push!(cells, "$(e(hm)) ($(f2(hm / hr))×)", "$(e(em)) ($(f2(em / er))×)")
        end
        println("| $(tt(t)) | " * join(cells, " | ") * " |")
    end
end

println("\n## Moving against resting\n")
println("Each cell is the moving row's point-weighted L2 norm over the shell about")
println("`c(t)`, with its ratio to the resting row's in parentheses.")
for (mv, rs, title) in (("qd24", "qd24s", ":damped 1/24"), ("qd32", "qd32s", ":damped 1/32"),
                        ("qf40", "qf40s", ":fitted 1/40"), ("qf40t", "qf40s", ":fitted 1/40, trail ramp"),
                        ("qd24f", "qd24s", ":damped 1/24, following mesh"),
                        ("qf40f", "qf40s", ":fitted 1/40, following mesh"),
                        ("qf48f", "hf48", ":fitted 1/48, following mesh, against H1's octant"),
                        ("qd32v2", "qd32v2s", ":damped 1/32, v = 1/2"),
                        ("qd40v2", "qd40v2s", ":damped 1/40, v = 1/2"),
                        ("qf48v2f", "hf48", ":fitted 1/48, v = 1/2, following mesh, against H1's octant"))
    versus(mv, rs, title)
end

# The orders outside the horizon between two spacings, moving and at rest.
function orders(a, b, title)
    (has(a) && has(b)) || return
    println("\n### $title: `$a` → `$b`\n")
    println("| t | " * join(("ℋ $(SHELL_RANGE[s]) | err $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " |")
    println("|---|" * repeat("---|---|", length(SHELLS)))
    for t in TIMES
        t ≤ min(tmax(RUNS[a]), tmax(RUNS[b])) || continue
        cells = String[]
        for s in SHELLS, c in ("ham_l2_$s", "err_l2_$s")
            push!(cells, f2(order(at(a, c, t), at(b, c, t), hof(a), hof(b))))
        end
        println("| $(tt(t)) | " * join(cells, " | ") * " |")
    end
end

println("\n## Orders\n")
orders("qd24", "qd32", ":damped, moving")
orders("qd24s", "qd32s", ":damped, at rest")
orders("qf40f", "qf48f", ":fitted, moving, following mesh")
orders("qf40", "qf48f", ":fitted, moving, capsule → following mesh")
orders("qd32v2", "qd40v2", ":damped, v = 1/2, moving")
orders("qd32v2s", "qd40v2s", ":damped, v = 1/2, at rest")

# The track and the horizon: where the finder puts the hole against `c(t)`,
# and its extent along the motion over its extent across it against
# `√(1 − v²)`.
println("\n## The track and the contraction\n")
println("`hz_x − c_x`, the found horizon's origin against the boosted hole's line,")
println("and `x/y − √(1 − v²)`, the extents' ratio against the contraction.\n")
rows = filter(n -> startswith(n, "q") && !endswith(n, "s") && has(n), sort!(collect(keys(RUNS))))
println("| t | " * join(("`$n` lag | `$n` x/y" for n in rows), " | ") * " |")
println("|---|" * repeat("---|---|", length(rows)))
for t in TIMES
    cells = String[]
    for n in rows
        v = occursin("v2", n) ? 1 / 2 : 1 / 5
        push!(cells, e(at(n, "hz_x", t) - at(n, "c_x", t)),
              e(at(n, "hz_ratio", t) - sqrt(1 - v^2)))
    end
    println("| $(tt(t)) | " * join(cells, " | ") * " |")
end

# The trailing and the leading side of the shell just outside the horizon.
println("\n## Trailing against leading, `[1, 5/4)`\n")
println("| t | " * join(("`$n` err trail / lead" for n in rows), " | ") * " |")
println("|---|" * repeat("---|", length(rows)))
for t in TIMES
    cells = [@sprintf("%s / %s", e(at(n, "err_l2_trail_r1", t)), e(at(n, "err_l2_lead_r1", t)))
             for n in rows]
    println("| $(tt(t)) | " * join(cells, " | ") * " |")
end

# `ℋ` on the two sides, against the resting row's (the same on both): where a
# moving row's excess lies.
println("\n## `ℋ` trailing and leading, against the resting row's\n")
println("| row | t | " * join(("$(SHELL_RANGE[s]) trail / lead / rest" for s in ("in", SHELLS[1:2]...)), " | ") * " |")
println("|---|---|" * repeat("---|", 3))
for (mv, rs) in (("qd24", "qd24s"), ("qd32", "qd32s"), ("qf40", "qf40s"), ("qf40t", "qf40s"),
                 ("qd32v2", "qd32v2s"), ("qd40v2", "qd40v2s"))
    (has(mv) && has(rs)) || continue
    t = min(tmax(RUNS[mv]), tmax(RUNS[rs]))
    cells = [@sprintf("%s / %s / %s", e(at(mv, "ham_l2_trail_$s", t)),
                      e(at(mv, "ham_l2_lead_$s", t)), e(at(rs, "ham_l2_$s", t)))
             for s in ("in", SHELLS[1:2]...)]
    println("| `$mv` | $(tt(t)) | " * join(cells, " | ") * " |")
end

# The horizon's mass.
println("\n## `M_irr − 1`, the horizon's irreducible mass against Kerr's\n")
println("| t | " * join(("`$n`" for n in sort!(collect(keys(RUNS))) if startswith(n, "q")), " | ") * " |")
println("|---|" * repeat("---|", count(n -> startswith(n, "q"), keys(RUNS))))
for t in TIMES
    cells = [e(at(n, "hz_M_irr", t) - 1) for n in sort!(collect(keys(RUNS))) if startswith(n, "q")]
    println("| $(tt(t)) | " * join(cells, " | ") * " |")
end
