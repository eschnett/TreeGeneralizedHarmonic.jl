# The excised operator's spectrum on a patch (added in step X8).
#
# A standalone script beside `excision_model.jl`, not part of the suite. Where
# `excision_model.jl` models the principal part of one component on a line or a
# plane, this asks the package's own right-hand side: on the small rotating
# octant of `test/octant_runs.jl` (Kerr-Schild `a = 3/5`, the excised ball
# `r < r_E`, `q = 4`, `ε_KO = 1/2`, the algebraic source) — by default its
# finest cube alone, `[0, 2]³` uniform at the hole's spacing, which is the same
# operator near the surface at a third of the points — it linearizes `F`
# about the analytic state — the coefficients frozen at the background — and
# restricts the Jacobian to a box of lattice points about one point: every
# non-excised owned point within `half` cells of it (`|k − c|_∞ ≤ half`), all
# twenty variables, the state outside the box held at the background. The
# columns are one-sided differences of `gh_rhs!`, one evaluation each; the
# eigenvalues of the box's Jacobian are its frozen-coefficient spectrum, with
# the eigenvector of the rightmost one located by its largest point.
#
# It is the analysis `PLAN.md`'s step X8 asks for where the shave does not hold:
# a growing lego corner is a local mode of the discrete operator, and the
# rightmost eigenvalue of a box about the corner says whether the operator has
# one there, at what rate, and on which points — the classes and the closures
# the run itself uses, with nothing modelled. The box is a principal submatrix
# of the whole operator (Dirichlet data at its faces), so a configuration known
# to be stable is its control.
#
#     julia --project=. --threads=4 test/excision_patch.jl r_E=17/15 r_0=13/15 shave=off center=3,2,27 half=2
#
# Options (`key=value`): `r_E`, `r_0` (default `(0.6 + r_E)/2`), `shave=on|off`,
# `center=i,j,k` (the box's center in cells of the finest spacing, a lattice
# point of the octant), `half=2` (the box's half-width in cells), the mesh `L=2
# N=24 roots=2 radii=` (`h = L/(roots N)/2^#radii`, `1/24` by default; `L=8
# radii=4,2` is `octant_runs.jl`'s small octant), `a=3/5`, `eps=1/2` (`ε_KO`),
# `delta=1e-7` (the difference's step, relative), `top=8` (eigenvalues printed).

using LinearAlgebra
using Printf
using TreeAMR
using TreeGeneralizedHarmonic
using StaticArrays: SVector

const TGHp = TreeGeneralizedHarmonic
const OPTS = Dict{String,String}()
for arg in ARGS
    k, v = split(arg, '='; limit=2)
    OPTS[String(k)] = String(v)
end
opt(k, d) = get(OPTS, k, d)
rat(s) = (p = split(s, '/'); length(p) == 2 ? parse(Int, p[1]) // parse(Int, p[2]) :
                            occursin(r"[.eE]", s) ? parse(Float64, s) : parse(Int, s))

const T = Float64
q = 4
G = q ÷ 2 + 1
N = parse(Int, opt("N", "24"))
L = rat(opt("L", "2"))
roots = parse(Int, opt("roots", "2"))
radii = [rat(r) for r in split(opt("radii", ""), ',') if !isempty(r)]
a = T(rat(opt("a", "3/5")))
r_E = T(rat(opt("r_E", "17/15")))
r_0 = haskey(OPTS, "r_0") ? T(rat(OPTS["r_0"])) : (T(3 // 5) + r_E) / 2
shave = opt("shave", "on") == "on"
ctr = Tuple(parse.(Int, split(opt("center", "3,2,27"), ',')))
half = parse(Int, opt("half", "2"))
δ = parse(Float64, opt("delta", "1e-7"))
top = parse(Int, opt("top", "8"))
h = T(L) / (roots * N) / 2^length(radii)
r_plus = 1 + sqrt(1 - a^2)
m = floor(Int, (r_plus - r_E) / h + 1 // 1000)
case = kerr_schild_case(T; a=a, halfwidth=T(L), r_0=r_0, r_1=r_E, chunk=one(T),
                        ε_KO=T(rat(opt("eps", "1/2"))), γ2=zero(T), octant=:rotating,
                        gauge_source=:algebraic, interior=:excised, margin=m,
                        horizon=nothing, excision=Excision(T; shave=shave))
forest = hole_forest(T, case; N=N, roots=roots, center=(0, 0, 0), radii=radii, shape=:cube)
U = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                parity=state_parity(forest), rotation=state_rotation(forest))
fill_exact!(U, case, zero(T))
p = GHProblem(U, GhostSchedule(U, Operators(prolongation=q + 2, restriction=q + 2)),
              case; q=q)
ex = p.excision
u0 = statevector(U)
gather!(u0, U)
du0 = similar(u0)
gh_rhs!(du0, u0, p, zero(T))
lin = LinearIndices(size(statearray(u0, U)))
cls = Array(ex.classes)

# The box: the non-excised owned points within `half` cells of the center.
pts = Tuple{NTuple{3,Int},Int,NTuple{3,Int},UInt8}[]      # (k, b, I, class)
for b in 1:nblocks(U), I in CartesianIndices((N, N, N))
    abs(spacing(T, forest, forest.leaves[b]) - h) < h / 100 || continue
    x = coordinates(U, b, Tuple(I) .+ G)
    k = ntuple(d -> round(Int, x[d] / h), 3)
    maximum(abs.(k .- ctr)) ≤ half || continue
    c = cls[I[1] + G, I[2] + G, I[3] + G, b]
    c == TGHp.CLASS_EXCISED && continue
    push!(pts, (k, b, Tuple(I), c))
end
sort!(pts)
idx = [lin[I..., v, b] for (_, b, I, _) in pts for v in 1:20]
n = length(idx)
@printf("a = %s, h = 1/%d, r_E = %.4f (%.2f cells, m = %d), r_0 = %.4f, shave %s: %d shaved, %d zone points\n",
        a, round(Int, 1 / h), r_E, r_E / h, m, r_0, shave ? "on" : "off", ex.nshaved, ex.nzone)
@printf("box about %s, half-width %d: %d points (%d zone), %d unknowns\n", ctr, half,
        length(pts), count(t -> t[4] == TGHp.CLASS_ZONE, pts), n)
flush(stdout)

# The box's Jacobian, a column per (point, variable): one-sided differences.
J = zeros(T, n, n)
up = copy(u0)
dup = similar(u0)
t0 = time()
for j in 1:n
    s = max(one(T), abs(u0[idx[j]])) * δ
    up[idx[j]] = u0[idx[j]] + s
    gh_rhs!(dup, up, p, zero(T))
    up[idx[j]] = u0[idx[j]]
    for i in 1:n
        J[i, j] = (dup[idx[i]] - du0[idx[i]]) / s
    end
    j == 20 && @printf("  %.2f s a column\n", (time() - t0) / 20)
end
@printf("Jacobian in %.0f s\n", time() - t0)
F = eigen(J)
order = sortperm(real.(F.values); rev=true)
onface(j) = maximum(abs.(pts[(j - 1) ÷ 20 + 1][1] .- ctr)) == half
function describe(r)
    λ = F.values[r]
    v = abs.(F.vectors[:, r])
    jmax = argmax(v)
    k, _, _, c = pts[(jmax - 1) ÷ 20 + 1]
    face = sum(v[j]^2 for j in 1:n if onface(j)) / sum(abs2, v)
    return λ, k, c, (jmax - 1) % 20 + 1, face
end
show_mode(λ, k, c, v, face) =
    @printf("  %+.4f %+.4fi   at %s (%s), variable %d; %.0f%% on the box's faces\n",
            real(λ), imag(λ), k, c == TGHp.CLASS_ZONE ? "zone" : "centered", v, 100 * face)
@printf("the rightmost %d eigenvalues (1/M), each with the point and variable of its eigenvector's largest component:\n", top)
for r in order[1:min(top, n)]
    show_mode(describe(r)...)
end
# A mode of the box's faces is the truncation's — the box is a principal
# submatrix, with Dirichlet data at its faces — and not the operator's: the
# rightmost ones with less than a tenth of their weight on the faces.
@printf("the rightmost %d with less than 10%% on the box's faces:\n", top)
let shown = 0
    for r in order
        d = describe(r)
        d[5] < 0.1 || continue
        show_mode(d...)
        shown += 1
        shown == top && break
    end
end
