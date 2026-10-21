# The boosted Kerr-Schild study (added 2026-10-20): a Schwarzschild hole
# (`M = 1`) moving along `+x` at `v = 1/5` from `x = −4` to `x = 4` over
# `t = 0 … 40` on the quadrant `[−64, 64] × [0, 64]²`, against the same hole
# at rest at `x = −4` on the same capsule mesh, at `h = 1/16, 1/24, 1/32`:
# the analytic `:damped` layer at `ρ_max = 4/M` and `8/M`, and the `:fitted`
# layer (`m = 16`, `n_L = 20`, `fit_cont = 2`, `lmax_fit = 8`) with and
# without `trail_ramp = 9/10`. Every row is a `test/octant_runs.jl` run; this
# script reads their `octant.csv` (one row per chunk, the shells about the
# hole's current center `c(t)`, each shell's trailing and leading half, and the
# horizon's columns) and prints Markdown tables for `SINGULARITY_HANDLING.md`.
#
#     julia test/boosted_study.jl [dir=<data>] [times=1,2,4,8,16,24,32,40]
#                                 [rate=4|8|fitted|all] [runs=bk16,bf24s,…]
#                                 [shells=r2,r3,r5] [baseline=<data>]
#
# `dir` holds one subdirectory per batch of jobs (`ks1/`, `ks2/`), each holding
# one directory per row, `h = 1/N`:
#
#   bk<N>, bk<N>s      `:damped` at 4/M, moving and at rest
#   bk<N>r8, bk<N>s8   `:damped` at 8/M, moving and at rest
#   bf<N>, bf<N>s      `:fitted` with the trailing ramp (moving) and at rest
#   bf<N>n             `:fitted`, moving, without the trailing ramp
#
# A row may be missing, queued (only `simwatch.toml`) or partial; where the
# same row is in two batches, the one that got further is used. `runs`
# selects rows by name (`bk16`) or by path (`ks2/bk16`); `rate=fitted` keeps
# the two `:fitted` families. `baseline` (added 2026-10-20) is a second data
# directory of the same layout, read the same way, whose moving rows each
# moving row of `dir` is compared with — how the moving `:damped` layer's target
# rate was measured against the runs without it:
#
#     julia test/boosted_study.jl dir=.claude/data/boosted-ks-rate \
#         baseline=.claude/data/boosted-ks rate=4,8
#
# The spinning family (added 2026-10-21): a Kerr-Schild hole of spin `a` moving
# along its spin axis `z` on the rotating quadrant (`octant=rotating_quadrant`)
# is the same table with `prefix=s`: its rows are named `sk<N>`, `sk<N>s`,
# `sk<N>r8`, `sk<N>s8`, `sf<N>`, `sf<N>s` and `sf<N>n`, the letters after the
# `s` as after the `b` above, and `a=` (default `1/2` for `prefix=s`, `0` for
# `b`) and `v=` (default `1/5`) give the expected horizon: `M_irr = √(r₊/2)`,
# `J = a` (the CSV's `hz_J` is the component along `z` there) and the extent
# along the motion over the extent across it, `√(1 − v²)` times the rest frame's
# `r₊/√(r₊² + a²)` along the spin (`z/x`; `1` along `x`):
#
#     julia test/boosted_study.jl dir=.claude/data/spinning-ks prefix=s a=1/2

using Printf

rat(s) = (p = split(s, '/'); length(p) == 2 ? parse(Int, p[1]) / parse(Int, p[2]) :
                            parse(Float64, s))
opts = Dict(String(k) => String(v)
            for (k, v) in (split(a, '='; limit=2) for a in ARGS if occursin('=', a)))
# The family: `b`, the hole without spin moving along x, or `s`, the spinning
# hole moving along z (added 2026-10-21).
const PREFIX = get(opts, "prefix", "b")
PREFIX in ("b", "s") || error("prefix is b (a = 0, along x) or s (spinning, along z), got $PREFIX")
const V = rat(get(opts, "v", "1/5"))
const A = rat(get(opts, "a", PREFIX == "s" ? "1/2" : "0"))
const CONTRACTION = sqrt(1 - V^2)       # the horizon's extent along the motion
# Kerr-Schild (`M = 1`): `r₊`, the irreducible mass, and the horizon's extent
# along the motion over the extent across it at rest — polar over equatorial
# along the spin axis z, and 1 for `a = 0` along x.
const R_PLUS = 1 + sqrt(1 - A^2)
const M_IRR = sqrt(R_PLUS / 2)
const REST_RATIO = PREFIX == "s" ? R_PLUS / sqrt(R_PLUS^2 + A^2) : 1.0
const AX = PREFIX == "s" ? "z" : "x"

default_dir = first(filter(isdir, [joinpath(@__DIR__, "..", ".claude", "data", "boosted-ks"),
                                   joinpath(@__DIR__, "..", "..", "..", "data", "boosted-ks"),
                                   "."]))
const DIR = normpath(abspath(get(opts, "dir", default_dir)))
const BASE = haskey(opts, "baseline") ? normpath(abspath(opts["baseline"])) : nothing
const TIMES = parse.(Float64, split(get(opts, "times", "1,2,4,8,16,24,32,40"), ','))
# The families, in the order they are printed: `:damped` at 4/M and 8/M,
# `:fitted` with the trailing ramp and without it.
const FAMILIES = Dict("all" => ["d4", "d8", "f", "fn"], "4" => ["d4"], "8" => ["d8"],
                      "4,8" => ["d4", "d8"],
                      "fitted" => ["f", "fn"])[get(opts, "rate", "all")]
const FAMNAME = Dict("d4" => ":damped 4/M", "d8" => ":damped 8/M",
                     "f" => ":fitted", "fn" => ":fitted, no trail ramp")
const PICK = haskey(opts, "runs") ? String.(split(opts["runs"], ',')) : nothing
const SHELLS = String.(split(get(opts, "shells", "r2,r3,r5"), ','))

num(s) = (x = tryparse(Float64, s); x === nothing ? NaN : x)
fmt(x) = isfinite(x) ? @sprintf("%.3e", x) : "—"
fmts(x) = isfinite(x) ? @sprintf("%+.3e", x) : "—"
fmtr(x) = isfinite(x) ? @sprintf("%.3g", x) : "—"
fmto(x) = isfinite(x) ? @sprintf("%.2f", x) : "—"

function table(header, rows)
    println("| ", join(header, " | "), " |")
    println("|", join([i == 1 ? ":---" : "---:" for i in eachindex(header)], "|"), "|")
    for r in rows
        println("| ", join(r, " | "), " |")
    end
    println()
end

# `bk16` → (N = 16, moving, "d4"); `bk16r8`, `bk16s`, `bk16s8`, `bf24`,
# `bf24s`, `bf24n` likewise, and `sk16` … for `prefix=s`.
function parse_name(nm)
    m = match(Regex("^" * PREFIX * raw"([kf])(\d+)(s?)(r?8|n)?$"), nm)
    m === nothing && return nothing
    fam = m[1] == "f" ? (m[4] == "n" ? "fn" : "f") : (m[4] === nothing ? "d4" : "d8")
    return (N=parse(Int, m[2]), moving=isempty(m[3]), fam=fam)
end

simwatch_status(d) = (f = joinpath(d, "simwatch.toml");
                      m = isfile(f) ? match(r"(?m)^status = \"(\w+)\"", read(f, String)) : nothing;
                      m === nothing ? "—" : String(m[1]))

# One row directory: the CSV with short rows padded by `NaN`, a restart's
# repeated times reduced to the last row of each, and the wall time summed
# over the chain's jobs (`wall` starts again from zero with every job).
function readrun(batch, name; root=DIR)
    d = joinpath(root, batch, name)
    info = parse_name(name)
    base = (; label="$batch/$name", name, info..., status=simwatch_status(d))
    p = joinpath(d, "octant.csv")
    lines = isfile(p) ? filter(!isempty, readlines(p)) : String[]
    if length(lines) < 2
        return (; base..., t=Float64[], rows=Vector{Float64}[], idx=Dict{String,Int}(),
                wall=NaN)
    end
    hdr = String.(split(lines[1], ','))
    raw = map(lines[2:end]) do l
        f = split(l, ',')
        [j <= length(f) ? num(f[j]) : NaN for j in eachindex(hdr)]
    end
    jw = findfirst(==("wall"), hdr)
    wall, prev = 0.0, 0.0
    for r in raw
        w = jw === nothing ? NaN : r[jw]
        isfinite(w) || continue
        w < prev && (wall += prev)
        prev = w
    end
    wall += prev
    bytime = Dict(r[1] => r for r in raw if isfinite(r[1]))
    t = sort!(collect(keys(bytime)))
    return (; base..., t, rows=[bytime[τ] for τ in t],
            idx=Dict(h => j for (j, h) in enumerate(hdr)), wall)
end

tlast(r) = isempty(r.t) ? NaN : r.t[end]
column(r, name) = (j = get(r.idx, name, nothing);
                   j === nothing ? fill(NaN, length(r.t)) : [row[j] for row in r.rows])
function at(r, name, τ)
    r === nothing && return NaN
    j = get(r.idx, name, nothing)
    i = findfirst(t -> abs(t - τ) < 1e-6, r.t)
    (j === nothing || i === nothing) && return NaN
    return r.rows[i][j]
end
has(r, τ) = r !== nothing && any(t -> abs(t - τ) < 1e-6, r.t)
order(ec, ef, Nc, Nf) = log(abs(ec / ef)) / log(Nf / Nc)

# The least-squares slope of `y` against `t` over `a ≤ t ≤ b`.
function slope(t, y, a, b)
    k = [i for i in eachindex(t) if a - 1e-9 ≤ t[i] ≤ b + 1e-9 && isfinite(y[i])]
    length(k) < 3 && return NaN
    x̄, ȳ = sum(t[k]) / length(k), sum(y[k]) / length(k)
    return sum((t[k] .- x̄) .* (y[k] .- ȳ)) / sum((t[k] .- x̄) .^ 2)
end

regions(q) = vcat([(s, "$(q)_$s") for s in SHELLS], [("global", q)])
desc(moving, fam) = (moving ? "moving" : "rest") * ", " * FAMNAME[fam]
famrank(fam) = something(findfirst(==(fam), ["d4", "d8", "f", "fn"]), 9)

# One table of `q` over `TIMES`: for every region, each run's value, then the
# ratios `runs[i] / runs[j]` for `(i, j)` in `ratios`. `runs` pairs a column
# name with a run (or `nothing`).
function compare(runs, ratios, q)
    hdr = ["t"]
    for (s, _) in regions(q)
        append!(hdr, ["$s $nm" for (nm, _) in runs])
        append!(hdr, ["$s $(runs[i][1])/$(runs[j][1])" for (i, j) in ratios])
    end
    rows = [vcat([@sprintf("%g", τ)],
                 reduce(vcat, [vcat([fmt(at(r, c, τ)) for (_, r) in runs],
                                    [fmtr(at(runs[i][2], c, τ) / at(runs[j][2], c, τ))
                                     for (i, j) in ratios])
                               for (_, c) in regions(q)]))
            for τ in TIMES if any(((_, r),) -> has(r, τ), runs)]
    isempty(rows) ? println("(no data at the requested times)\n") : table(hdr, rows)
end
labelof(r) = r === nothing ? "—" : r.label

# Every row of a data directory, and the row of each kind that got furthest,
# across its batches.
function readall(root)
    isdir(root) || error("no data directory $root")
    all_runs = [readrun(b, n; root) for b in sort(readdir(root)) if isdir(joinpath(root, b))
                for n in sort(readdir(joinpath(root, b)))
                if isdir(joinpath(root, b, n)) && parse_name(n) !== nothing]
    PICK === nothing ||
        filter!(r -> r.name in PICK || r.label in PICK, all_runs)
    filter!(r -> r.fam in FAMILIES, all_runs)
    sort!(all_runs; by=r -> (famrank(r.fam), !r.moving, r.N, r.label))
    chosen = Dict{Tuple{Bool,String,Int},Any}()
    for r in all_runs
        isempty(r.t) && continue
        k = (r.moving, r.fam, r.N)
        (!haskey(chosen, k) || tlast(r) > tlast(chosen[k])) && (chosen[k] = r)
    end
    return all_runs, chosen
end

function main()
    all_runs, chosen = readall(DIR)
    pick(mv, fam, N) = get(chosen, (mv, fam, N), nothing)
    used = sort!(collect(values(chosen)); by=r -> (famrank(r.fam), !r.moving, r.N))
    Ns = sort!(unique([r.N for r in all_runs]))

    println(PREFIX == "b" ? "# Boosted Kerr-Schild study\n\ndata: `$DIR`\n" :
            "# Boosted spinning Kerr-Schild study (a = $A, along z)\n\ndata: `$DIR`\n")
    println("## Runs\n")
    table(["run", "kind", "h", "t reached", "finished", "status", "wall [h]", "used"],
          [[r.label, desc(r.moving, r.fam), "1/$(r.N)",
            isempty(r.t) ? "—" : @sprintf("%g", tlast(r)),
            r.status == "finished" ? "yes" : "no", r.status,
            isfinite(r.wall) ? @sprintf("%.2f", r.wall / 3600) : "—",
            any(u -> u === r, used) ? "✓" : ""] for r in all_runs])

    println("## Moving against rest\n")
    println("Point-weighted L2 in the shells about `c(t)`; `global` is the evolved region.\n")
    for fam in FAMILIES, N in Ns
        fam == "fn" && continue                  # no resting row of its own
        mv, rs = pick(true, fam, N), pick(false, fam, N)
        (mv === nothing && rs === nothing) && continue
        for q in ("err_l2", "ham_l2")
            println("### `$q`, $(FAMNAME[fam]), h = 1/$N ($(labelof(mv)) / $(labelof(rs)))\n")
            compare([("mov", mv), ("rest", rs)], [(1, 2)], q)
        end
    end

    if BASE !== nothing
        _, base = readall(BASE)
        println("## Against the baseline\n")
        println("Each moving row of `$DIR` (`new`) against the moving row of the same ",
                "kind in `$BASE` (`base`), and the resting row of `new`'s directory ",
                "(`rest`).\n")
        for fam in FAMILIES, N in Ns
            mv, bs, rs = pick(true, fam, N), get(base, (true, fam, N), nothing),
                         pick(false, fam, N)
            (mv === nothing || bs === nothing) && continue
            for q in ("err_l2", "ham_l2")
                println("### `$q`, $(FAMNAME[fam]), h = 1/$N ($(labelof(mv)) / ",
                        "$(labelof(bs)) / $(labelof(rs)))\n")
                compare([("new", mv), ("base", bs), ("rest", rs)], [(1, 2), (1, 3)], q)
            end
            println("### Layer residual, $(FAMNAME[fam]), h = 1/$N\n")
            table(["t", "new", "base", "rest"],
                  [[@sprintf("%g", τ), fmt(at(mv, "residual", τ)), fmt(at(bs, "residual", τ)),
                    fmt(at(rs, "residual", τ))]
                   for τ in vcat([0.0], TIMES) if has(mv, τ) || has(bs, τ)])
        end
    end

    println("## `:fitted` against `:damped`, moving\n")
    println("The moving `:fitted` row (`bf`) against the moving `:damped` rows at ",
            "4/M (`bk`) and 8/M (`bkr8`), then `:fitted` with the trailing ramp ",
            "(`bf`) against without it (`bfn`).\n")
    for N in Ns, q in ("err_l2", "ham_l2")
        f, d4, d8 = pick(true, "f", N), pick(true, "d4", N), pick(true, "d8", N)
        f === nothing || (d4 === nothing && d8 === nothing) ||
            (println("### `$q`, h = 1/$N ($(labelof(f)), $(labelof(d4)), $(labelof(d8)))\n");
             compare([(PREFIX * "f", f), (PREFIX * "k", d4), (PREFIX * "kr8", d8)], [(1, 2), (1, 3)], q))
    end
    for N in Ns, q in ("err_l2", "ham_l2")
        f, fn = pick(true, "f", N), pick(true, "fn", N)
        (f === nothing || fn === nothing) && continue
        println("### `$q`, trailing ramp, h = 1/$N ($(f.label) / $(fn.label))\n")
        compare([(PREFIX * "f", f), (PREFIX * "fn", fn)], [(1, 2)], q)
    end

    println("## Convergence orders\n")
    println("`log(e_c/e_f)/log(h_c/h_f)`; 4 is expected outside the horizon.\n")
    for fam in FAMILIES, mv in (true, false)
        have = [N for N in Ns if pick(mv, fam, N) !== nothing]
        for (Nc, Nf) in zip(have[1:end-1], have[2:end])
            rc, rf = pick(mv, fam, Nc), pick(mv, fam, Nf)
            println("### $(desc(mv, fam)), h = 1/$Nc → 1/$Nf ($(rc.label), $(rf.label))\n")
            hdr = vcat(["t"], ["err $s" for (s, _) in regions("err_l2")],
                       ["ham $s" for (s, _) in regions("ham_l2")])
            rows = [vcat([@sprintf("%g", τ)],
                         [fmto(order(at(rc, c, τ), at(rf, c, τ), Nc, Nf))
                          for q in ("err_l2", "ham_l2") for (_, c) in regions(q)])
                    for τ in TIMES if has(rc, τ) && has(rf, τ)]
            isempty(rows) ? println("(no common times yet)\n") : table(hdr, rows)
        end
    end

    println("## Trailing against leading\n")
    println("Ratio trailing/leading of each shell's two halves (moving runs).\n")
    for r in used
        r.moving || continue
        sh = [c[length("err_l2_trail_")+1:end] for c in keys(r.idx)
              if startswith(c, "err_l2_trail_")]
        sort!(sh; by=s -> s == "in" ? -1.0 : num(s[2:end]))
        isempty(sh) && continue
        println("### $(r.label), $(desc(r.moving, r.fam)), h = 1/$(r.N)\n")
        hdr = vcat(["t"], ["err $s" for s in sh], ["ham $s" for s in sh])
        rows = [vcat([@sprintf("%g", τ)],
                     [fmtr(at(r, "$(q)_trail_$s", τ) / at(r, "$(q)_lead_$s", τ))
                      for q in ("err_l2", "ham_l2") for s in sh])
                for τ in TIMES if has(r, τ)]
        table(hdr, rows)
    end

    println("## Horizon\n")
    if PREFIX == "b"
        println("`x/y − e` is the found horizon's x/y extent against `e = √(1 − v²) = ",
                @sprintf("%.7f", CONTRACTION), "` (moving) or 1 (rest); `x − c_x` the found ",
                "origin against the analytic center.\n")
    else
        println("`z/x − e` is the found horizon's z/x extent against `e = √(1 − v²) ",
                "r₊/√(r₊² + a²) = ", @sprintf("%.7f", CONTRACTION * REST_RATIO),
                "` (moving) or `", @sprintf("%.7f", REST_RATIO), "` (rest), at `a = $A`; ",
                "`M_irr` against `√(r₊/2) = ", @sprintf("%.7f", M_IRR), "`, `J` (along z) ",
                "against `a`; `z − c_z` the found origin against the analytic center.\n")
    end
    hdr = PREFIX == "b" ? ["t", "M_irr − 1", "J", "x/y − e", "x − c_x"] :
          ["t", "M_irr − √(r₊/2)", "J − a", "z/x − e", "z − c_z"]
    for r in used
        haskey(r.idx, "hz_M_irr") || continue
        e = (r.moving ? CONTRACTION : 1.0) * REST_RATIO
        println("### $(r.label), $(desc(r.moving, r.fam)), h = 1/$(r.N)\n")
        rows = [[@sprintf("%g", τ), fmts(at(r, "hz_M_irr", τ) - M_IRR),
                 fmts(at(r, "hz_J", τ) - A), fmts(at(r, "hz_ratio", τ) - e),
                 fmts(at(r, "hz_$AX", τ) - at(r, "c_$AX", τ))]
                for τ in TIMES if has(r, τ)]
        table(hdr, rows)
    end
    println("### Drift of `M_irr`\n")
    println("Least-squares `dM_irr/dt` (per M) over each run's last half, and over ",
            "`[T/2, T]` with `T` the furthest time every run of the family reached.\n")
    rows = Vector{String}[]
    for fam in FAMILIES, mv in (true, false)
        rs = [r for r in (pick(mv, fam, N) for N in Ns) if r !== nothing]
        isempty(rs) && continue
        T = minimum(tlast, rs)
        d = [slope(r.t, column(r, "hz_M_irr"), T / 2, T) for r in rs]
        for (i, r) in enumerate(rs)
            own = slope(r.t, column(r, "hz_M_irr"), tlast(r) / 2, tlast(r))
            o = i == 1 ? NaN : order(d[i-1], d[i], rs[i-1].N, r.N)
            push!(rows, [r.label, desc(mv, fam), "1/$(r.N)",
                         @sprintf("[%g, %g]", tlast(r) / 2, tlast(r)), fmts(own),
                         @sprintf("[%g, %g]", T / 2, T), fmts(d[i]), fmto(o)])
        end
    end
    table(["run", "kind", "h", "own window", "dM_irr/dt", "common window",
           "dM_irr/dt", "order"], rows)

    println("## Layer residual\n")
    for (fams, what) in ((["d4", "d8"], "`:damped`: the layer's distance from the " *
                          "analytic solution it relaxes to"),
                         (["f", "fn"], "`:fitted`: the layer's distance from the *truth* " *
                          "inside the offset surface, not from its own (fitted) target"))
        rs = [r for r in used if r.fam in fams]
        isempty(rs) && continue
        println("$what.\n")
        table(vcat(["t"], [r.label for r in rs]),
              [vcat([@sprintf("%g", τ)], [fmt(at(r, "residual", τ)) for r in rs])
               for τ in vcat([0.0], TIMES) if any(r -> has(r, τ), rs)])
    end
    return nothing
end

main()
