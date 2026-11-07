# The right-hand-side kernel's cost per point with each gauge source (added in step
# B1; `CODE.md`, "The gauge"). One launch of `gh_rhs_kernel!` on an already filled
# working array — the kernel alone, no scatter and no ghost fill — on a uniform mesh
# of the Kerr-Schild exterior `[2, 6]³` (no interior, so every point takes the same
# branch), `q = 4`, `Float64`, for:
#
#   none       no source: the hole in harmonic coordinates, `H ≡ 0` (`Val(false)`)
#   sampled    the sampled `Hsrc` field set (`Val(true)`)
#   ks         `KerrSchildSource`, the closed form (`Val(:algebraic)`)
#   accel      a `CompositeKerrSchildSource` of one hole with an acceleration
#   two        a `CompositeKerrSchildSource` of two holes
#   dh         `DampedHarmonicSource`
#   roll       `RolledGaugeSource` from `ks` to `dh` half way (both, and the blend)
#
# `accel`, `two`, `dh` and `roll` do not make the background exact, so their
# problems are built on a case that carries them without `GHCase`'s check; the cost
# is what is measured, not the solution.
#
#     julia --project=. -t 1 bench/gauge_source_cost.jl              # the CPU, scalar and lanes
#     julia --project=<env with CUDA> bench/gauge_source_cost.jl backend=cuda
#
# Options: N=16 roots=4 reps=20 sources=none,sampled,ks,accel,two,dh,roll
using TreeAMR, TreeGeneralizedHarmonic, KernelAbstractions, StaticArrays, Printf
import SpacetimeMetrics
const TGH = TreeGeneralizedHarmonic

const OPTS = Dict(split(a, '='; limit=2) for a in ARGS if occursin('=', a))
const T = Float64
const q = 4
const N = parse(Int, get(OPTS, "N", "16"))
const ROOTS = parse(Int, get(OPTS, "roots", "4"))
const REPS = parse(Int, get(OPTS, "reps", "20"))
const SOURCES = split(get(OPTS, "sources", "none,sampled,ks,accel,two,dh,roll"), ',')
const backend = if get(OPTS, "backend", "cpu") == "cuda"
    @eval using CUDA
    Base.invokelatest(() -> CUDA.CUDABackend())
else
    KernelAbstractions.CPU()
end
sync() = KernelAbstractions.synchronize(backend)

box = ntuple(_ -> (T(2), T(6)), 3)
base(bg) = GHCase(T, bg; box=box, periodic=(false, false, false), ε_KO=T(1 // 2),
                  γ0=one(T), γ2=zero(T))
# A case carrying `src` without the check that it makes the background exact.
function carrying(case::GHCase, src)
    fs = fieldnames(GHCase)
    vals = ntuple(i -> fs[i] === :gauge ? src : getfield(case, i), length(fs))
    P = typeof(case).parameters
    return GHCase{P[1:end-1]...,typeof(src)}(vals...)
end
ks = KerrSchildSource(T; M=1)
holes = (TrackedHole(T; M=1 // 2, spin=(0, 0, 1 // 2), center=(1, 0, 0),
                     velocity=(0, 1 // 5, 0), acceleration=(-1 // 80, 0, 0)),
         TrackedHole(T; M=1 // 2, spin=(0, 0, 1 // 2), center=(-1, 0, 0),
                     velocity=(0, -1 // 5, 0), acceleration=(1 // 80, 0, 0)))
dh = DampedHarmonicSource(T; σ_r=32)
cases = Dict(
    "none" => base(SpacetimeMetrics.Harmonic(T(1), T(0))),
    "sampled" => base(SpacetimeMetrics.KerrSchild(T(1), T(0))),
    "ks" => carrying(base(SpacetimeMetrics.KerrSchild(T(1), T(0))), ks),
    "accel" => carrying(base(SpacetimeMetrics.KerrSchild(T(1), T(0))),
                        CompositeKerrSchildSource(holes[1])),
    "two" => carrying(base(SpacetimeMetrics.KerrSchild(T(1), T(0))),
                      CompositeKerrSchildSource(holes...)),
    "dh" => carrying(base(SpacetimeMetrics.KerrSchild(T(1), T(0))), dh),
    "roll" => carrying(base(SpacetimeMetrics.KerrSchild(T(1), T(0))),
                       RolledGaugeSource(ks, dh; σ_g=1)))

widths = backend isa KernelAbstractions.CPU ? (1, nothing) : (1,)
@printf("gauge source cost: %s, %d threads, N = %d, %d blocks, %d points, q = %d\n",
        typeof(backend), Threads.nthreads(), N, ROOTS^3, ROOTS^3 * N^3, q)
for name in SOURCES, W in widths
    case = cases[name]
    f = gh_forest(T, case; N=N, roots=ROOTS)
    U = FieldSet{T}(f, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                    parity=state_parity(f), backend=backend)
    fill_exact!(U, case, zero(T))
    u = statevector(U)
    gather!(u, U)
    p = GHProblem(U, GhostSchedule(U, Operators(prolongation=q + 2, restriction=q + 2)),
                  case; q=q, simd_width=W)
    du = similar(u)
    t = T(1)                       # the roll-on half way (σ_g = 1 from t = 0)
    gh_rhs!(du, u, p, t)           # compile, and fill the ghosts once
    sync()
    args = TGH.gh_rhs_kernel_args(p, du, t)
    best = Inf
    for _ in 1:REPS
        t0 = time_ns()
        map_blocks!(TGH.gh_rhs_kernel!, p.U, args...)
        sync()
        best = min(best, (time_ns() - t0) / 1e9)
    end
    npts = ROOTS^3 * N^3
    @printf("  %-8s W = %-2d  %8.3f ms  %7.2f ns per point\n", name, something(W, -1) < 0 ?
            TGH.default_simd_width(T, backend, N) : W, 1e3 * best, 1e9 * best / npts)
end
