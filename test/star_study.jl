# The tables of the star-damping study (added 2026-11-11, for TreeGRRMHD's GH-8):
# reads a directory of `test/star_runs.jl` runs, one subdirectory each, and
# prints one Markdown row per run — the star (`R/M`), the damping (`γ0 M`,
# `γ2`), the resolution (`h/R`), how far it got, the L∞ of the change of the
# state at a few times, the e-folding time of that change over the run's last
# half and last quarter (`∞` where it does not grow), `ℋ` at the end, and
# which variable and which radius carry the largest change at the end.
#
#     julia test/star_study.jl dir=out/star-damping times=50,100,250,500
#
# It needs no package, so it runs without `--project`.

using Printf

const OPTIONS = Dict{String,String}()
for arg in ARGS
    k, v = occursin('=', arg) ? split(arg, '='; limit=2) : (arg, "")
    OPTIONS[String(k)] = String(v)
end
dir = get(OPTIONS, "dir", "out/star-damping")
times = parse.(Float64, split(get(OPTIONS, "times", "50,100,250,500"), ','))

function readcsv(path)
    lines = filter(!isempty, readlines(path))
    header = String.(split(first(lines), ','))
    rows = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
    return header, isempty(rows) ? zeros(0, length(header)) :
                   permutedims(reduce(hcat, rows))
end
col(h, X, name) = X[:, findfirst(==(name), h)]

function readsetup(path)
    d = Dict{String,String}()
    for l in readlines(path)
        k, v = split(l, " = "; limit=2)
        d[k] = v
    end
    return d
end

# The e-folding time of `y` over `t ∈ [t_a, t_b]`: least squares on `log y`.
function efold(t, y, ta, tb)
    sel = [ta ≤ t[i] ≤ tb && isfinite(y[i]) && y[i] > 0 for i in eachindex(t)]
    count(sel) ≥ 3 || return NaN
    x = t[sel]
    z = log.(y[sel])
    xm = sum(x) / length(x)
    zm = sum(z) / length(z)
    s = sum((x .- xm) .* (z .- zm)) / sum((x .- xm) .^ 2)
    return 1 / s
end
fmtE(τ) = isnan(τ) ? "—" : (τ ≤ 0 || τ > 1e4) ? "∞" : @sprintf("%.0f", τ)

const VARNAMES = ["h_tt", "h_tx", "h_ty", "h_tz", "h_xx", "h_xy", "h_xz", "h_yy", "h_yz",
                  "h_zz", "Π_tt", "Π_tx", "Π_ty", "Π_tz", "Π_xx", "Π_xy", "Π_xz", "Π_yy",
                  "Π_yz", "Π_zz"]

runs = []
for name in sort(readdir(dir))
    rd = joinpath(dir, name)
    isfile(joinpath(rd, "series.csv")) && isfile(joinpath(rd, "setup.txt")) || continue
    s = readsetup(joinpath(rd, "setup.txt"))
    h, X = readcsv(joinpath(rd, "series.csv"))
    size(X, 1) ≥ 2 || continue
    push!(runs, (; name, s, h, X, rd))
end
sort!(runs; by=r -> (parse(Float64, r.s["R_over_M"]), parse(Float64, r.s["gamma2"]),
                     parse(Float64, r.s["h"]) / parse(Float64, r.s["R"]),
                     parse(Float64, r.s["L"]) / parse(Float64, r.s["R"]),
                     parse(Float64, r.s["gamma0M"])))

println("| run | `R/M` | `γ0 M` | `γ0 R` | `γ2` | `h/R` | `L/R` | to `t/M` | ",
        join(("`Δu` at $(Int(t)) M" for t in times), " | "),
        " | e-fold, last ½ (`M`) | last ¼ | `ℋ` at 0 | `ℋ` at end | `C` at end | largest | at `r/R` |")
println("|", repeat("---|", 15 + length(times)))
for r in runs
    s = r.s
    tM = col(r.h, r.X, "t_M")
    du = col(r.h, r.X, "linf")
    ham = col(r.h, r.X, "ham_l2")
    gauge = col(r.h, r.X, "gauge_l2")
    RM = parse(Float64, s["R_over_M"])
    γ0M = parse(Float64, s["gamma0M"])
    R = parse(Float64, s["R"])
    tl = tM[end]
    at(t) = (i = findfirst(x -> abs(x - t) < 1e-6 * max(1, t), tM);
             i === nothing ? "—" : @sprintf("%.1e", du[i]))
    # The largest change at the end: which variable, and where.
    vh, V = readcsv(joinpath(r.rd, "vars.csv"))
    vmax = argmax(V[end, 2:end])
    ph, P = readcsv(joinpath(r.rd, "profile.csv"))
    last = P[P[:, 1] .== P[end, 1], :]
    ib = argmax(max.(last[:, 4], last[:, 5]))
    rpeak = (last[ib, 2] + last[ib, 3]) / 2 / R
    println("| ", r.name, " | ", @sprintf("%g", RM), " | ", @sprintf("%g", γ0M), " | ",
            @sprintf("%g", γ0M * RM), " | ", s["gamma2"], " | ",
            @sprintf("1/%.0f", R / parse(Float64, s["h"])), " | ",
            @sprintf("%g", parse(Float64, s["L"]) / R), " | ",
            isfinite(du[end]) ? @sprintf("%.0f", tl) : @sprintf("%.0f (NaN)", tl), " | ",
            join((at(t) for t in times), " | "), " | ",
            fmtE(efold(tM, du, tl / 2, tl)), " | ", fmtE(efold(tM, du, 3tl / 4, tl)), " | ",
            @sprintf("%.1e", ham[1]), " | ", @sprintf("%.1e", ham[end]), " | ",
            @sprintf("%.1e", gauge[end]), " | ", VARNAMES[vmax], " | ",
            @sprintf("%.2f", rpeak), " |")
end
