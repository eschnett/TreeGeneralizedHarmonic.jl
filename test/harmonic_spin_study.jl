# The spinning harmonic hole (added 2026-11-14, step H3): Kerr in harmonic
# coordinates at `a = 1/2` and `7/10`, spinning about `z`, with no gauge source.
# At rest on the rotating octant, moving along its spin at `v = 1/5` on the
# rotating quadrant, and moving across it on the bitant at `v = 1/5` and `3/10`,
# each moving row against the same hole at rest on the same mesh. Every row is a
# `test/octant_runs.jl` run; this script reads their `octant.csv` (one row per
# chunk: the shells about the hole's current center, each shell's trailing and
# leading half, the horizon's columns on the moving meshes) and `records.csv`
# (the horizon's `M_irr` and `J` on every mesh), and prints the Markdown tables
# of `SINGULARITY_HANDLING.md`'s "A spinning harmonic hole (step H3)". It needs
# no package.
#
#     julia test/harmonic_spin_study.jl [dir=<data>] [times=1,2,4,8,12,16,20]
#                                       [shells=r1,r1.25,r2,r4]
#
# `dir` (default `.claude/data/harm-h3`) holds one directory per row; `ROWS`
# below says what each is. The shells are the point-weighted ones of steps H1 and
# H2 about the hole's center: `r1` is `[1, 5/4)`, `r1.25` `[5/4, 2)`, `r2`
# `[2, 4)`, `r4` `[4, 7)` and `r7` beyond.

using Printf

opts = Dict(String(k) => String(v)
            for (k, v) in (split(a, '='; limit=2) for a in ARGS if occursin('=', a)))
root = normpath(joinpath(@__DIR__, ".."))
datadir(name) = first(filter(isdir, [joinpath(root, ".claude", "data", name),
                                     joinpath(root, "..", "..", "data", name), "."]))
const DIR = normpath(abspath(get(opts, "dir", datadir("harm-h3"))))
const TIMES = parse.(Float64, split(get(opts, "times", "1,2,4,8,12,16,20"), ','))
const SHELLS = split(get(opts, "shells", "r1,r1.25,r2,r4"), ',')
const SHELL_RANGE = Dict("in" => "inside", "r1" => "[1, 5/4)", "r1.25" => "[5/4, 2)",
                         "r2" => "[2, 4)", "r4" => "[4, 7)", "r7" => "≥ 7")

# What each row is: the spin, `h`, the mesh, the speed, the interior, and where
# it ran. `:damped` is `r_0 = 3/5`, `r_1 = 3/4` (`r_0 = 23/40` at `1/48`);
# `:fitted` is `fit_cont = 2`, `lmax_fit = 8`, with the margin `m` and the ramp
# `n_L` in cells. The capsules start at `−2` along the motion (`−3` at
# `v = 3/10`) and are `4 M` long (`6 M`).
const ROWS = [
    # name        a      N    mesh                 v       interior                       where
    ("s5f48m8",   1/2,  48, "rotating octant",    0.0,  "`:fitted` `m = 8`, `n_L = 12`",  "1 H200"),
    ("s5f64m8",   1/2,  64, "rotating octant",    0.0,  "`:fitted` `m = 8`, `n_L = 12`",  "1 H200"),
    ("s5f64m12",  1/2,  64, "rotating octant",    0.0,  "`:fitted` `m = 12`, `n_L = 12`", "1 H200"),
    ("s5f64m16",  1/2,  64, "rotating octant",    0.0,  "`:fitted` `m = 16`, `n_L = 12`", "1 H200"),
    ("s5f96m16",  1/2,  96, "rotating octant",    0.0,  "`:fitted` `m = 16`, `n_L = 20`", "1 H200"),
    ("s5f96m24",  1/2,  96, "rotating octant",    0.0,  "`:fitted` `m = 24`, `n_L = 12`", "1 H200"),
    ("s5d64",     1/2,  64, "rotating octant",    0.0,  "`:damped`",                      "1 H200"),
    ("s5d96",     1/2,  96, "rotating octant",    0.0,  "`:damped`",                      "1 H200"),
    ("s7f80m6",   7/10, 80, "rotating octant",    0.0,  "`:fitted` `m = 6`, `n_L = 12`",  "1 H200"),
    ("s7f96m8",   7/10, 96, "rotating octant",    0.0,  "`:fitted` `m = 8`, `n_L = 12`",  "1 H200"),
    ("s7f96m12",  7/10, 96, "rotating octant",    0.0,  "`:fitted` `m = 12`, `n_L = 12`", "1 H200"),
    ("s7f128m12", 7/10, 128, "rotating octant",   0.0,  "`:fitted` `m = 12`, `n_L = 12`", "1 H200"),
    ("s7f128m16", 7/10, 128, "rotating octant",   0.0,  "`:fitted` `m = 16`, `n_L = 12`", "1 H200"),
    ("r5d48",     1/2,  48, "rotating octant",    0.0,  "`:damped`",                      "1 H200"),
    ("r5d64",     1/2,  64, "rotating octant",    0.0,  "`:damped`",                      "1 H200"),
    ("r5d96",     1/2,  96, "rotating octant",    0.0,  "`:damped`",                      "1 H200"),
    ("r5f64",     1/2,  64, "rotating octant",    0.0,  "`:fitted` `m = 16`, `n_L = 12`", "1 H200"),
    ("r5f96",     1/2,  96, "rotating octant",    0.0,  "`:fitted` `m = 16`, `n_L = 12`", "1 H200"),
    ("r5f96m",    1/2,  96, "rotating octant",    0.0,  "`:fitted` `m = 24`, `n_L = 18`", "1 H200"),
    ("r7f128",    7/10, 128, "rotating octant",   0.0,  "`:fitted` `m = 16`, `n_L = 12`", "2 H200s"),
    ("z5d48",     1/2,  48, "rotating quadrant",  0.2,  "`:damped`",                      "1 H200"),
    ("z5d48s",    1/2,  48, "rotating quadrant",  0.0,  "`:damped`",                      "1 H200"),
    ("z5d64",     1/2,  64, "rotating quadrant",  0.2,  "`:damped`",                      "1 H200"),
    ("z5d64s",    1/2,  64, "rotating quadrant",  0.0,  "`:damped`",                      "1 H200"),
    ("z5d96",     1/2,  96, "rotating quadrant",  0.2,  "`:damped`",                      "2 H200s"),
    ("z5d96s",    1/2,  96, "rotating quadrant",  0.0,  "`:damped`",                      "2 H200s"),
    ("z5f64",     1/2,  64, "rotating quadrant",  0.2,  "`:fitted` `m = 16`, `n_L = 12`", "1 H200"),
    ("z5f64t",    1/2,  64, "rotating quadrant",  0.2,  "the same, `trail_ramp = 9/10`",  "1 H200"),
    ("z5f64s",    1/2,  64, "rotating quadrant",  0.0,  "`:fitted` `m = 16`, `n_L = 12`", "1 H200"),
    ("b5d64",     1/2,  64, "bitant",             0.2,  "`:damped`",                      "2 H200s"),
    ("b5d64s",    1/2,  64, "bitant",             0.0,  "`:damped`",                      "2 H200s"),
    ("b5d96",     1/2,  96, "bitant",             0.2,  "`:damped`",                      "4 H200s"),
    ("b5d96s",    1/2,  96, "bitant",             0.0,  "`:damped`",                      "4 H200s"),
    ("b5f64",     1/2,  64, "bitant",             0.2,  "`:fitted` `m = 16`, `n_L = 12`", "2 H200s"),
    ("b5f64t",    1/2,  64, "bitant",             0.2,  "the same, `trail_ramp = 9/10`",  "2 H200s"),
    ("b5f64s",    1/2,  64, "bitant",             0.0,  "`:fitted` `m = 16`, `n_L = 12`", "2 H200s"),
    ("b5d64v3",   1/2,  64, "bitant, `6 M`",      0.3,  "`:damped`",                      "2 H200s"),
    ("b5f64v3",   1/2,  64, "bitant, `6 M`",      0.3,  "`:fitted` `m = 16`, `n_L = 12`", "2 H200s"),
    ("b5d96v3",   1/2,  96, "bitant, `6 M`",      0.3,  "`:damped`",                      "8 H100s"),
    ("b5d96v3s",  1/2,  96, "bitant, `6 M`",      0.0,  "`:damped`",                      "8 H100s"),
]
const META = Dict(r[1] => r for r in ROWS)
spin(n) = META[n][2]
hof(n) = 1 / META[n][3]
speed(n) = META[n][5]
kerr_M_irr(a) = sqrt((1 + sqrt(1 - a^2)) / 2)

# --- reading -------------------------------------------------------------------

struct Run
    name::String
    cols::Dict{String,Vector{Float64}}
end

function readcsv(f)
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
    seen = Dict{Float64,Int}()
    for i in eachindex(t)
        seen[t[i]] = i
    end
    idx = sort!(collect(values(seen)); by=i -> t[i])
    return Dict(k => v[idx] for (k, v) in cols)
end

const RUNS = Dict{String,Run}()
for (n,) in ROWS
    c = readcsv(joinpath(DIR, n, "octant.csv"))
    c === nothing && continue
    # The horizon's `M_irr` and `J` come from `records.csv`, on every mesh;
    # `octant.csv` carries them only on the moving meshes.
    r = readcsv(joinpath(DIR, n, "records.csv"))
    if r !== nothing
        for k in ("M_irr", "J")
            c["rec_" * k] = [let i = findfirst(==(t), r["t"])
                                 i === nothing ? NaN : r[k][i]
                             end for t in c["t"]]
        end
    elseif haskey(c, "hz_J")
        c["rec_M_irr"], c["rec_J"] = c["hz_M_irr"], c["hz_J"]
    end
    RUNS[n] = Run(n, c)
end

has(n) = haskey(RUNS, n)
tmax(r::Run) = r.cols["t"][end]
tmax(n::AbstractString) = has(n) ? tmax(RUNS[n]) : NaN
function at(r::Run, col, t)
    haskey(r.cols, col) || return NaN
    i = findfirst(==(t), r.cols["t"])
    return i === nothing ? NaN : r.cols[col][i]
end
at(n::AbstractString, col, t) = has(n) ? at(RUNS[n], col, t) : NaN

e(x) = isnan(x) ? "—" : @sprintf("%.2e", x)
f2(x) = isnan(x) ? "—" : @sprintf("%.2f", x)
f1(x) = isnan(x) || isinf(x) ? "—" : @sprintf("%.1f", x)
tt(t) = isinteger(t) ? string(Int(t)) : @sprintf("%g", t)
order(a, b, ha, hb) = log(a / b) / log(ha / hb)

# The least-squares slope of a column over `t0 ≤ t ≤ t1`.
function slope(n, col, t0, t1)
    has(n) || return NaN
    t, y = RUNS[n].cols["t"], get(RUNS[n].cols, col, Float64[])
    isempty(y) && return NaN
    k = [i for i in eachindex(t) if t0 ≤ t[i] ≤ t1 && !isnan(y[i])]
    length(k) ≥ 2 || return NaN
    mt, my = sum(t[k]) / length(k), sum(y[k]) / length(k)
    return sum((t[k] .- mt) .* (y[k] .- my)) / sum((t[k] .- mt) .^ 2)
end

# The cost of a row: the wall clock per M from its last restart's first row to
# its last row (a restart's compilation is not counted).
function cost(n)
    has(n) || return NaN
    t, w = RUNS[n].cols["t"], RUNS[n].cols["wall"]
    length(t) ≥ 2 || return NaN
    i = findlast(k -> k > 1 && w[k] < w[k - 1], eachindex(w))
    i = i === nothing ? 2 : i + 1
    i ≤ length(t) || return NaN
    return (w[end] - w[i - 1]) / (t[end] - t[i - 1])
end

# --- the tables -------------------------------------------------------------------

println("## Rows\n")
println("| row | `a` | `h` | mesh | `v` | interior | on | to | `s/M` |")
println("|---|---|---|---|---|---|---|---|---|")
for (n, a, N, mesh, v, int, where) in ROWS
    has(n) || continue
    println("| `$n` | $(a == 1/2 ? "1/2" : "7/10") | `1/$N` | $mesh | $(v == 0 ? "0" : v == 0.2 ? "1/5" : "3/10") | " *
            "$int | $where | `$(tt(tmax(n))) M` | $(@sprintf("%.0f", cost(n))) |")
end

# The screens at `8 M`: the margin and the depth that fit between the horizon
# and the ring, at rest on the rotating octant.
println("\n## The screens at rest, `8 M`\n")
println("| row | ℋ [1, 5/4) | ℋ [2, 4) | err [1, 5/4) | err [2, 4) | residual | `M_irr − M_K` | `J − a` | `dJ/dt` over `[2, 8]` |")
println("|---|---|---|---|---|---|---|---|---|")
for (n,) in ROWS
    (startswith(n, "s") && has(n)) || continue
    t = tmax(n)
    t ≥ 8 || (println("| `$n` | ended at `t = $(tt(t))` | | | | | | | |"); continue)
    println("| `$n` | $(e(at(n, "ham_l2_r1", t))) | $(e(at(n, "ham_l2_r2", t))) | $(e(at(n, "err_l2_r1", t))) | " *
            "$(e(at(n, "err_l2_r2", t))) | $(e(at(n, "residual", t))) | $(e(at(n, "rec_M_irr", t) - kerr_M_irr(spin(n)))) | " *
            "$(e(at(n, "rec_J", t) - spin(n))) | $(e(slope(n, "rec_J", 2, 8))) |")
end

# At rest to `24 M`: the shells, the orders and the horizon.
const REST = ["r5d48", "r5d64", "r5d96", "r5f64", "r5f96", "r5f96m", "r7f128"]
println("\n## At rest, `24 M`\n")
println("| row | " * join(("ℋ $(SHELL_RANGE[s]) | err $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " |")
println("|---|" * repeat("---|---|", length(SHELLS)))
for n in REST
    has(n) || continue
    println("| `$n` | " * join((e(at(n, "$(q)_l2_$s", 24.0)) for s in SHELLS for q in ("ham", "err")), " | ") * " |")
end

function orders(a, b, title; times=TIMES)
    (has(a) && has(b)) || return
    println("\n### $title: `$a` → `$b`\n")
    println("| t | " * join(("ℋ $(SHELL_RANGE[s]) | err $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " |")
    println("|---|" * repeat("---|---|", length(SHELLS)))
    for t in times
        t ≤ min(tmax(a), tmax(b)) || continue
        cells = [f2(order(at(a, c, t), at(b, c, t), hof(a), hof(b)))
                 for s in SHELLS for c in ("ham_l2_$s", "err_l2_$s")]
        println("| $(tt(t)) | " * join(cells, " | ") * " |")
    end
end

const RTIMES = [1.0, 4.0, 8.0, 16.0, 24.0]
orders("r5d48", "r5d64", "`:damped` at rest"; times=RTIMES)
orders("r5d64", "r5d96", "`:damped` at rest"; times=RTIMES)
orders("r5f64", "r5f96", "`:fitted` at rest, the margin fixed in cells"; times=RTIMES)
orders("r5f64", "r5f96m", "`:fitted` at rest, the margin fixed in M"; times=RTIMES)

println("\n### `:fitted` over `:damped` at the same spacing\n")
println("| pair | t | " * join(("ℋ $(SHELL_RANGE[s]) | err $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " |")
println("|---|---|" * repeat("---|---|", length(SHELLS)))
for (f, d) in (("r5f64", "r5d64"), ("r5f96", "r5d96"), ("r5f96m", "r5d96")), t in (4.0, 24.0)
    (has(f) && has(d)) || continue
    println("| `$f / $d` | $(tt(t)) | " *
            join((f2(at(f, "$(q)_l2_$s", t) / at(d, "$(q)_l2_$s", t)) for s in SHELLS for q in ("ham", "err")), " | ") * " |")
end

# The horizon of every row: `M_irr` and `J` against Kerr's, and their drift.
println("\n## The horizon: `M_irr` and `J`\n")
println("`M_K = √(r₊/2)` is Kerr's irreducible mass. The slopes are least-squares")
println("fits per `M` over the interval.\n")
println("| row | to | `M_irr − M_K` | `J − a` | `dM_irr/dt` `[4, end]` | `dJ/dt` `[0, 4]` | `dJ/dt` `[4, end]` | `dJ/dt` `[10, end]` |")
println("|---|---|---|---|---|---|---|---|")
for (n,) in ROWS
    (has(n) && !startswith(n, "s")) || continue
    t = tmax(n)
    println("| `$n` | `$(tt(t)) M` | $(e(at(n, "rec_M_irr", t) - kerr_M_irr(spin(n)))) | $(e(at(n, "rec_J", t) - spin(n))) | " *
            "$(e(slope(n, "rec_M_irr", 4, t))) | $(e(slope(n, "rec_J", 0, 4))) | $(e(slope(n, "rec_J", 4, t))) | " *
            "$(e(slope(n, "rec_J", 10, t))) |")
end

# Moving against resting, per shell.
const PAIRS = [("z5d48", "z5d48s"), ("z5d64", "z5d64s"), ("z5d96", "z5d96s"),
               ("z5f64", "z5f64s"), ("z5f64t", "z5f64s"),
               ("b5d64", "b5d64s"), ("b5d96", "b5d96s"),
               ("b5f64", "b5f64s"), ("b5f64t", "b5f64s"),
               ("b5d64v3", "b5d64s"), ("b5d96v3", "b5d96v3s"), ("b5f64v3", "b5f64s")]

println("\n## Moving against resting\n")
println("The moving row's ratio to its resting row, shell by shell. `b5d64v3` and")
println("`b5f64v3` are compared with the `4 M` capsule's resting rows, which equal")
println("the `6 M` capsule's within `1.00×` in `[1, 5/4)` and `0.7–1.25×` beyond")
println("(`b5d96s` against `b5d96v3s`, below).\n")
println("| pair | t | " * join(("ℋ $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " | " *
        join(("err $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " |")
println("|---|---|" * repeat("---|", 2 * length(SHELLS)))
for (mv, rs) in PAIRS
    (has(mv) && has(rs)) || continue
    t = min(tmax(mv), tmax(rs))
    println("| `$mv / $rs` | $(tt(t)) | " *
            join((f2(at(mv, "$(q)_l2_$s", t) / at(rs, "$(q)_l2_$s", t)) for q in ("ham", "err") for s in SHELLS), " | ") * " |")
end

function versus(mv, rs)
    (has(mv) && has(rs)) || return
    println("\n### `$mv` over `$rs`\n")
    println("| t | " * join(("ℋ $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " | " *
            join(("err $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " |")
    println("|---|" * repeat("---|", 2 * length(SHELLS)))
    for t in TIMES
        t ≤ min(tmax(mv), tmax(rs)) || continue
        println("| $(tt(t)) | " *
                join((f2(at(mv, "$(q)_l2_$s", t) / at(rs, "$(q)_l2_$s", t)) for q in ("ham", "err") for s in SHELLS), " | ") * " |")
    end
end
for (mv, rs) in PAIRS
    versus(mv, rs)
end
versus("b5d96s", "b5d96v3s")

# The order of a moving row's excess over its resting row between two spacings.
function excess_order(m1, r1, m2, r2)
    all(has, (m1, r1, m2, r2)) || return
    println("\n### The excess `$m1 − $r1` → `$m2 − $r2`\n")
    println("| t | " * join(("ℋ $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " | " *
            join(("err $(SHELL_RANGE[s])" for s in SHELLS), " | ") * " |")
    println("|---|" * repeat("---|", 2 * length(SHELLS)))
    for t in TIMES
        t ≤ minimum(tmax, (m1, r1, m2, r2)) || continue
        cells = String[]
        for q in ("ham", "err"), s in SHELLS
            c = "$(q)_l2_$s"
            a, b = at(m1, c, t) - at(r1, c, t), at(m2, c, t) - at(r2, c, t)
            push!(cells, a > 0 && b > 0 ? f1(order(a, b, hof(m1), hof(m2))) : "—")
        end
        println("| $(tt(t)) | " * join(cells, " | ") * " |")
    end
end

println("\n## The order of the excess\n")
println("The order at which a moving row's excess over its resting row falls between")
println("two spacings; `—` where the moving row is not above the resting one.")
excess_order("z5d48", "z5d48s", "z5d64", "z5d64s")
excess_order("z5d64", "z5d64s", "z5d96", "z5d96s")
excess_order("b5d64", "b5d64s", "b5d96", "b5d96s")
excess_order("b5d64v3", "b5d64s", "b5d96v3", "b5d96v3s")

println("\n## The orders at rest on the capsules\n")
orders("z5d48s", "z5d64s", "`:damped` at rest, rotating quadrant")
orders("z5d64s", "z5d96s", "`:damped` at rest, rotating quadrant")
orders("b5d64s", "b5d96s", "`:damped` at rest, bitant")

# The trailing and the leading side just outside the horizon, against rest.
println("\n## `ℋ` trailing and leading, against the resting row's\n")
println("| row | t | " * join(("$(SHELL_RANGE[s]) trail / lead / rest" for s in ("in", SHELLS[1:2]...)), " | ") * " |")
println("|---|---|" * repeat("---|", 3))
for (mv, rs) in PAIRS
    (has(mv) && has(rs)) || continue
    t = min(tmax(mv), tmax(rs))
    cells = [@sprintf("%s / %s / %s", e(at(mv, "ham_l2_trail_$s", t)),
                      e(at(mv, "ham_l2_lead_$s", t)), e(at(rs, "ham_l2_$s", t)))
             for s in ("in", SHELLS[1:2]...)]
    println("| `$mv` | $(tt(t)) | " * join(cells, " | ") * " |")
end

# The track: where the finder puts the horizon along the motion against the
# boosted hole's line, and its shape against the resting one contracted.
println("\n## The track and the contraction\n")
println("The lag is the found horizon's origin along the motion less the boosted hole's;")
println("the shape is the extent along the motion over the extent across it, against")
println("the resting row's times `√(1 − v²)`.\n")
println("| row | t | lag | shape − contracted rest |")
println("|---|---|---|---|")
for (mv, rs) in PAIRS
    (has(mv) && has(rs)) || continue
    ax = startswith(mv, "z") ? "z" : "x"
    v = speed(mv)
    for t in unique([min(4.0, tmax(mv)), min(tmax(mv), tmax(rs))])
        println("| `$mv` | $(tt(t)) | $(e(at(mv, "hz_$ax", t) - at(mv, "c_$ax", t))) | " *
                "$(e(at(mv, "hz_ratio", t) - at(rs, "hz_ratio", t) * sqrt(1 - v^2))) |")
    end
end
