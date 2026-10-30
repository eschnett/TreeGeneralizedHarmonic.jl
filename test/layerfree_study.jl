# The layer-free interior's screen (added in step I1): reads a directory of
# `test/octant_runs.jl` rows — one subdirectory per row, `R0` the `:damped`
# baseline and `R1` … `R17` the layer-free rows of `SINGULARITY_HANDLING.md`, "A
# layer-free interior (step I1)" — and prints the screen's tables: each row's
# outcome against the pass criteria of `PLAN.md`'s step I1, the shells at a time,
# and, for promoted rows, the orders between resolutions.
#
#     julia test/layerfree_study.jl dir=out/i1 [base=R0] [times=1,2,4,8] [at=8]
#     julia test/layerfree_study.jl dir=out/i1 series=R3a16,R3a24,R3a32:16,24,32
#
# A row passes if it (PLAN.md):
#   - reaches t_end (8 M for the screen);
#   - has hits per chunk that level off, with the outermost hit at least 8 finest
#     cells inside the horizon and not moving outward;
#   - has ℋ and the error in [2, 9/4) within 3× the baseline's, flat or falling
#     from 4 M.
# It needs no package: plain Julia, like `test/aligned_study.jl`.

using Printf

opts = Dict{String,String}()
for a in ARGS
    occursin('=', a) || continue
    k, v = split(a, '='; limit=2)
    opts[String(k)] = String(v)
end
dir = get(opts, "dir", "out/i1")
base = get(opts, "base", "R0")
times = [parse(Float64, x) for x in split(get(opts, "times", "1,2,4,8"), ',')]
t_at = parse(Float64, get(opts, "at", "8"))
t_flat = parse(Float64, get(opts, "t_flat", "4"))

num(s) = (x = tryparse(Float64, s); x === nothing ? NaN : x)
function readcsv(path)
    isfile(path) || return nothing
    l = readlines(path)
    length(l) < 2 && return nothing
    cols = split(l[1], ',')
    rows = [split(x, ',') for x in l[2:end] if !isempty(x)]
    return Dict(String(c) => [num(r[j]) for r in rows] for (j, c) in enumerate(cols))
end
col(d, name) = get(d, name, fill(NaN, length(d["t"])))
# The value at the row nearest to `t` (within half a chunk), or NaN.
function at(d, name, t)
    ts = d["t"]
    k = argmin(abs.(ts .- t))
    return abs(ts[k] - t) ≤ 0.5 ? col(d, name)[k] : NaN
end
# The least-squares slope of `log y` against `t` over `t ≥ t0`.
function logslope(t, y, t0)
    k = [i for i in eachindex(t) if t[i] ≥ t0 && isfinite(y[i]) && y[i] > 0]
    length(k) < 3 && return NaN
    x = t[k]; z = log.(y[k])
    x̄ = sum(x) / length(x); z̄ = sum(z) / length(z)
    return sum((x .- x̄) .* (z .- z̄)) / sum((x .- x̄) .^ 2)
end
function slope(t, y, t0)
    k = [i for i in eachindex(t) if t[i] ≥ t0 && isfinite(y[i])]
    length(k) < 3 && return NaN
    x = t[k]; z = y[k]
    x̄ = sum(x) / length(x); z̄ = sum(z) / length(z)
    return sum((x .- x̄) .* (z .- z̄)) / sum((x .- x̄) .^ 2)
end

labels = sort!([d for d in readdir(dir) if isdir(joinpath(dir, d)) &&
                readcsv(joinpath(dir, d, "octant.csv")) !== nothing];
               by=d -> (m = match(r"^R(\d+)", d); m === nothing ? (1000, d) :
                                                    (parse(Int, m[1]), d)))
data = Dict(lb => readcsv(joinpath(dir, lb, "octant.csv")) for lb in labels)
recs = Dict(lb => readcsv(joinpath(dir, lb, "records.csv")) for lb in labels)
haskey(data, base) || @warn "no baseline row $base in $dir: the ratios are NaN"
# The shell `[2, 9/4)` is `r2` in every row's CSV.
const S1 = "r2"

println("# The layer-free screen, $dir (baseline $base)\n")
println("## Outcomes\n")
println("| row | t | ℋ [2,9/4) at $(t_at) | ×$base | err [2,9/4) | ×$base | σ ℋ, σ err from $t_flat | hits/chunk (last) | outermost hit, cells inside | moving out | dM_irr/dt | verdict |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|")
for lb in labels
    d = data[lb]
    t = d["t"]
    tend = t[end]
    h = at(d, "ham_l2_$S1", t_at)
    e = at(d, "err_l2_$S1", t_at)
    hb = haskey(data, base) ? at(data[base], "ham_l2_$S1", t_at) : NaN
    eb = haskey(data, base) ? at(data[base], "err_l2_$S1", t_at) : NaN
    σh = logslope(t, col(d, "ham_l2_$S1"), t_flat)
    σe = logslope(t, col(d, "err_l2_$S1"), t_flat)
    hits = col(d, "hits")
    cells = col(d, "hit_cells_inside")
    lasthits = isempty(hits) || all(isnan, hits) ? NaN : hits[end]
    fin = filter(isfinite, cells)
    mincells = isempty(fin) ? NaN : minimum(fin)
    # Moving outward: the outermost hit's radius over the second half against the first.
    rr = col(d, "hit_r_max")
    k2 = length(rr) ÷ 2
    moving = k2 ≥ 2 && maximum(filter(x -> x ≥ 0, vcat(rr[(k2 + 1):end], -1.0))) >
                       maximum(filter(x -> x ≥ 0, vcat(rr[1:k2], -1.0))) + 1e-12
    # Hits that level off: the last two chunks' mean not above the two before by > 20 %.
    lev = length(hits) < 5 || all(isnan, hits) ? true :
          sum(hits[(end - 1):end]) ≤ 1.2 * sum(hits[(end - 3):(end - 2)]) + 2
    r = recs[lb]
    dM = r === nothing ? NaN : slope(r["t"], r["M_irr"], t_flat)
    pass = tend ≥ t_at - 1e-9 && lev && !moving && (isnan(mincells) || mincells ≥ 8) &&
           h ≤ 3 * hb && e ≤ 3 * eb && !(σh > 0.05) && !(σe > 0.05)
    @printf("| %s | %.2f | %.3g | %.2f | %.3g | %.2f | %+.3f, %+.3f | %s | %s | %s | %.3g | %s |\n",
            lb, tend, h, h / hb, e, e / eb, σh, σe,
            isnan(lasthits) ? "—" : string(round(Int, lasthits)),
            isnan(mincells) ? "none" : @sprintf("%.1f", mincells), moving ? "yes" : "no",
            dM, lb == base ? "baseline" : pass ? "**pass**" : "fail")
end

println("\n## ℋ and the error in [2, 9/4) against time\n")
println("| row | " * join(("ℋ $(t)" for t in times), " | ") * " | " *
        join(("err $(t)" for t in times), " | ") * " |")
println("|---|" * repeat("---|", 2 * length(times)))
for lb in labels
    d = data[lb]
    println("| $lb | " * join((@sprintf("%.3g", at(d, "ham_l2_$S1", t)) for t in times),
                              " | ") * " | " *
            join((@sprintf("%.3g", at(d, "err_l2_$S1", t)) for t in times), " | ") * " |")
end

println("\n## The shells at t = $t_at\n")
for lb in labels
    d = data[lb]
    names = [c[8:end] for c in keys(d) if occursin(r"^ham_l2_r[0-9]", c)]
    sort!(names; by=n -> parse(Float64, n[2:end]))
    println("**$lb**\n")
    println("| shell | ℋ | ℳ | C_a | err | |∂_t h| | min α | min √γ/√γ_s | max λ(γ) | roughness |")
    println("|---|---|---|---|---|---|---|---|---|---|")
    for nm in names
        @printf("| %s | %.3g | %.3g | %.3g | %.3g | %.3g | %.4g | %.4g | %.4g | %.3g |\n",
                nm, at(d, "ham_l2_$nm", t_at), at(d, "mom_l2_$nm", t_at),
                at(d, "gauge_l2_$nm", t_at), at(d, "err_l2_$nm", t_at),
                at(d, "stat_l2_$nm", t_at), at(d, "alpha_min_$nm", t_at),
                at(d, "vol_min_$nm", t_at), at(d, "lambda_max_$nm", t_at),
                at(d, "rough_$nm", t_at))
    end
    if haskey(d, "hits")
        @printf("\nhits at %.4g: %d (non-finite %d, spectrum %d, shift %d, lapse %d, momentum %d); C_a over the ramp %.3g; α = 1/2 at r = %.4g, α = 1/10 at r = %.4g; M_irr %.9f, r_min %.6f\n\n",
                t_at, round(Int, at(d, "hits", t_at)), round(Int, at(d, "hits_nonfinite", t_at)),
                round(Int, at(d, "hits_spectrum", t_at)), round(Int, at(d, "hits_shift", t_at)),
                round(Int, at(d, "hits_lapse", t_at)), round(Int, at(d, "hits_momentum", t_at)),
                at(d, "gauge_l2_ramp", t_at), at(d, "r_alpha_half", t_at),
                at(d, "r_alpha_tenth", t_at), at(d, "M_irr", t_at), at(d, "r_min", t_at))
    else
        println()
    end
end

# The promotion's orders: `series=a,b,c:n_a,n_b,n_c` (cells per M), repeatable as
# `series2=` …, the shells outside the horizon at the last common time.
for (k, v) in opts
    startswith(k, "series") || continue
    lbs, ns = split(v, ':')
    lbs = split(lbs, ',')
    ns = [parse(Float64, x) for x in split(ns, ',')]
    all(lb -> haskey(data, lb), lbs) || (@warn "series $v: a row is missing"; continue)
    tc = minimum(data[lb]["t"][end] for lb in lbs)
    println("\n## Orders, $(join(lbs, ", ")) at t = $tc\n")
    println("| quantity | " * join(lbs, " | ") * " | orders |")
    println("|---|" * repeat("---|", length(lbs) + 1))
    for nm in ("r2", "r2.25", "r3", "r5", "r8"), q in ("ham_l2", "err_l2", "gauge_l2")
        ys = [at(data[lb], "$(q)_$nm", tc) for lb in lbs]
        ords = [log(ys[i] / ys[i + 1]) / log(ns[i + 1] / ns[i]) for i in 1:(length(ys) - 1)]
        println("| $q $nm | " * join((@sprintf("%.3g", y) for y in ys), " | ") * " | " *
                join((@sprintf("%.2f", o) for o in ords), ", ") * " |")
    end
    dMs = [recs[lb] === nothing ? NaN : slope(recs[lb]["t"], recs[lb]["M_irr"], 8.0)
           for lb in lbs]
    println("| dM_irr/dt (8 M on) | " * join((@sprintf("%.3g", y) for y in dMs), " | ") *
            " | " * join((@sprintf("%.2f", log(dMs[i] / dMs[i + 1]) / log(ns[i + 1] / ns[i]))
                          for i in 1:(length(dMs) - 1)), ", ") * " |")
end
