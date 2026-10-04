# The difference of two `test/octant_runs.jl` runs, read from their
# checkpoints (added 2026-10-02): a run with noise against the same run
# without, which is the perturbation's own evolution — on a black hole the
# noise is orders below the truncation error, so the two runs' norms agree to
# several digits and only their difference shows whether it grows.
#
#     julia --project=. test/octant_diff.jl out/ck-noise/octant out/ck-clean/octant [r_1=3/2]
#
# For every iteration both prefixes have a checkpoint of, the point-weighted
# L2 and the L∞ of `u_a − u_b` over the evolved points `r ≥ r_1`, over all
# twenty variables, overall and per refinement level.

using Printf
using TreeAMR
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: load_run, from_plain

const T = Float64
pa, pb = ARGS[1], ARGS[2]
opts = Dict(split(a, '='; limit=2) for a in ARGS[3:end] if occursin('=', a))
r1 = let s = get(opts, "r_1", "3/2"), p = split(s, '/')
    length(p) == 2 ? parse(Float64, p[1]) / parse(Float64, p[2]) : parse(Float64, s)
end

function files(prefix)
    dir, base = dirname(prefix), basename(prefix)
    out = Dict{Int,String}()
    for name in readdir(isempty(dir) ? "." : dir)
        m = match(Regex("^" * base * raw"\.it(\d+)\.h5$"), name)
        m === nothing || (out[parse(Int, m.captures[1])] = joinpath(dir, name))
    end
    return out
end

# Per level: the sum of squares and the largest |δ| of `ua − ub` over the
# points `r ≥ r1`, and their number. A function, so that the loop is compiled.
function differences(f, ua, ub, r1)
    N = f.N
    nlev = maximum(level, f.leaves) + 1
    nv = size(ua, 4)
    s2 = zeros(nlev); mx = zeros(nlev); n = zeros(Int, nlev)
    for blk in 1:nleaves(f)
        k = f.leaves[blk]
        ℓ = level(k) + 1
        o = block_origin(T, f, k)
        h = spacing(T, f, k)
        for I in CartesianIndices((N, N, N))
            x = ntuple(d -> o[d] + (I[d] - 1) * h, 3)
            sum(abs2, x) < r1^2 && continue
            n[ℓ] += 1
            for v in 1:nv
                δ = ua[I, v, blk] - ub[I, v, blk]
                s2[ℓ] += δ * δ
                mx[ℓ] = max(mx[ℓ], abs(δ))
            end
        end
    end
    return s2, mx, n
end

fa, fb = files(pa), files(pb)
its = sort!(collect(intersect(keys(fa), keys(fb))))
isempty(its) && error("no iteration has a checkpoint under both $pa and $pb")
first_row = true
for it in its
    a = load_run(fa[it], T)
    b = load_run(fb[it], T)
    a.forest.leaves == b.forest.leaves || error("iteration $it: the two meshes differ")
    t = from_plain(T, a.run.t)
    s2, mx, n = differences(a.forest, statearray(a.u, a.U), statearray(b.u, b.U), r1)
    nlev = length(n)
    nv = a.U.nvars
    if first_row
        @printf("%8s %8s %11s %11s   %s\n", "it", "t", "L2", "Linf",
                join([@sprintf("L2_L%d", ℓ - 1) for ℓ in 1:nlev], "      "))
        global first_row = false
    end
    @printf("%8d %8.2f %11.4e %11.4e   %s\n", it, t, sqrt(sum(s2) / (sum(n) * nv)),
            maximum(mx), join([@sprintf("%.4e", sqrt(s2[ℓ] / (n[ℓ] * nv))) for ℓ in 1:nlev], "  "))
    flush(stdout)
end
