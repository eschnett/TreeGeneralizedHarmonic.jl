# The growth rates of `test/octant_runs.jl`'s record (added 2026-10-02).
#
#     julia test/octant_rates.jl out/octant/octant.csv [t_from=16]
#
# For every norm column, the least-squares slope of `log ‖·‖` against `t`
# over `[t_from, t_end]`, over its second half, and over its last quarter —
# a rate `σ` in `‖·‖ ∝ e^{σ t}` — with the e-folding time `1/σ` and the
# norm's total change. A transient that decays and a mode that grows show as
# a rate that changes sign between the windows; a mode that has taken over
# shows as the windows agreeing.

using Printf

path = ARGS[1]
opts = Dict(split(a, '='; limit=2) for a in ARGS[2:end] if occursin('=', a))
t_from = parse(Float64, get(opts, "t_from", "16"))

lines = readlines(path)
cols = split(lines[1], ',')
data = [parse.(Float64, split(l, ',')) for l in lines[2:end] if !isempty(l)]
# A restart appends rows; keep the last row of every time.
bytime = Dict(r[1] => r for r in data)
data = [bytime[t] for t in sort!(collect(keys(bytime)))]
t = [r[1] for r in data]
col(name) = [r[findfirst(==(name), cols)] for r in data]

function slope(ts, ys)
    keep = [i for i in eachindex(ys) if ys[i] > 0 && isfinite(ys[i])]
    length(keep) < 3 && return NaN
    x, y = ts[keep], log.(ys[keep])
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

t_end = t[end]
windows = [(t_from, t_end), ((t_from + t_end) / 2, t_end), (t_end - (t_end - t_from) / 4, t_end)]
@printf("%s: %d rows, t = %g … %g\n", path, length(t), t[1], t_end)
@printf("%-16s %11s %11s %11s %11s %11s %9s\n", "norm", "at t=0", "at t_end",
        @sprintf("σ[%g,%g]", windows[1]...), @sprintf("σ[%g,…]", windows[2][1]),
        @sprintf("σ[%g,…]", windows[3][1]), "1/σ")
for name in cols[3:end]
    y = col(name)
    σ = [slope(t[(t .≥ a) .& (t .≤ b)], y[(t .≥ a) .& (t .≤ b)]) for (a, b) in windows]
    @printf("%-16s %11.3e %11.3e %11.3e %11.3e %11.3e %9.1f\n", name, y[1], y[end],
            σ..., 1 / σ[1])
end
