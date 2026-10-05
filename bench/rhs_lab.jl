# Prototype right-hand-side kernels on a GPU (added 2026-10-05). The analysis and the
# numbers are in `CODE.md`, "The right-hand side on an H200".
#
# These are experiments, not the package's kernel: raw CUDA.jl and KernelAbstractions
# kernels that recompute the work of `gh_rhs!`'s kernel at `q = 4` with no interior.
#
# - **Case:** the gauge wave (no gauge source, `ε_KO = 1/2`, `γ0 = 1`) on a uniform
#   periodic mesh of `roots³` blocks of `N³`.
# - **Checked** against `gh_rhs!` on the same working array (`err h … Π …` on a row,
#   relative to the largest `|∂ₜu|`).
# - **Timed** as the minimum of `reps` calls, in ns per owned point.
#
# CUDA is not a dependency of this package. Run it, as `bench/stepping.jl` on a device,
# from a copy whose `Project.toml` has CUDA added:
#
#     julia --project=. bench/rhs_lab.jl mode=baseline N=32 roots=8
#     julia --project=. bench/rhs_lab.jl mode=round9 N=32 roots=8
#
# On Symmetry run one H200 a job: `srun --cpu-bind=none julia --project=. bench/rhs_lab.jl …`
# under `--partition=h200debugq --gres=gpu:h200:1 --cpus-per-task=8 --mem=96G`.
#
# The modes are the investigation's rounds, in the order they were run, and keep the
# rows `CODE.md` quotes (Symmetry jobs 570186–570203):
#
#   mode=baseline  gh_rhs! and its parts; the package kernel at several workgroups,
#                  with and without always_inline; its SASS and a CUDA.@profile
#   mode=variants  the package's body launched raw: static and dynamic strides,
#                  maxregs, ablations, reordered, the stencil/algebra split
#                  (which=copy,fused,fused_dyn,fused_mr,ablate,reordered,split,alg)
#   mode=round3    stencils without the zero weights; the principal part from shared
#                  memory (which=fusedf,k1f,k2f,k2s)
#   mode=round4    the source alone: gh_node_source against the lean spelling
#   mode=round5    the pipeline around the kernel: scatter, fill, RK4 step, stage
#                  broadcasts, a static-stride scatter, a fused stage update
#   mode=round6    the lean source in the fused kernel and in the two-kernel split
#   mode=round7    the split as KernelAbstractions kernels
#   mode=round8    what the fused lean kernel needs: strides, unrolling, order
#   mode=round9    the closure-free lean source, with and without always_inline
#
#   N=, roots=     block size and roots per edge (32, 8: 512 blocks, 16.8 M points)
#   inline=1       compile the raw kernels with always_inline (variants, round3–round6)
#   sass=1         write each raw kernel's SASS to out/; bench/sass_stats.jl counts it
#   which=…        a subset of a mode's rows; `alg` adds the algebraic Kerr-Schild
#                  source (variants, round4, round6)
#   reps=, out=    timed repetitions (8) and the output directory (out)
#
# The lean source the kernels use is bench/rhs_lab_source.jl, which has no device code
# and which bench/rhs_lab_cpu.jl checks and times on the CPU.
using TreeAMR, TreeGeneralizedHarmonic, KernelAbstractions, CUDA, StaticArrays, Printf,
      LinearAlgebra
using CUDA: i32
const TGH = TreeGeneralizedHarmonic
import IMEXRungeKutta as IRK
using TreeGeneralizedHarmonic: NC, gh_rhs_at_point, metric_quantities, metric_derivatives,
                               gh_node_source, axis_stencil, mixed_stencil,
                               derivative_weights, dissipation_weights, dissipation_rank
const _sym4 = TGH._sym4
const _dg4 = TGH._dg4

const T = Float64
const Q = 4
const GG = 3

const OPTS = Dict(String(split(a, "=")[1]) => String(split(a, "=")[2])
                  for a in ARGS if occursin('=', a))
opt(k, d) = get(OPTS, k, d)
const INLINE = opt("inline", "0") == "1"
const OUT = opt("out", "out")

# --- setup -------------------------------------------------------------------

function setup(N, roots)
    case = gauge_wave_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T))
    forest = gh_forest(T, case; N=N, roots=roots)
    ops = Operators(prolongation=Q + 2, restriction=Q + 2)
    U = FieldSet{T}(forest, 2NC; G=GG, centering=vertexcentered(3), backend=CUDABackend())
    fill_exact!(U, case, zero(T))
    u = statevector(U)
    gather!(u, U)
    p = GHProblem(U, GhostSchedule(U, ops), case; q=Q, t=zero(T))
    @assert size(U.work)[1:3] == ntuple(_ -> N + 2GG + 1, 3) "stored extent $(size(U.work))"
    return case, forest, U, u, p
end

ka_args(p, du) = (statearray(du, p.U), p.U.work, TGH.gauge_work(p.Hsrc), p.origins,
                  p.spacings, p.case.background, p.case.γ0, p.case.γ2, p.case.ε_KO,
                  p.interior, zero(T), TGH.target_work(p.target), p.t_target,
                  p.target_rate, p.trail, TGH._exact_fit(p), p.valG, p.valq, p.valH,
                  p.valdiss, p.valint)

function timeit(f; n=parse(Int, opt("reps", "8")))
    f()
    CUDA.synchronize()
    best = Inf
    for _ in 1:n
        t = CUDA.@elapsed f()
        best = min(best, t)
    end
    return best
end

function compare(du, ref, N, nb)
    a = reshape(du, N^3, 2NC, nb)
    r = reshape(ref, N^3, 2NC, nb)
    e(rng) = maximum(abs.(view(a, :, rng, :) .- view(r, :, rng, :))) /
             maximum(abs.(view(r, :, rng, :)))
    return e(1:NC), e(NC+1:2NC)
end

# --- indexing for the raw kernels ------------------------------------------------
#
# grid = ((N/TX)(N/TY), N/TZ, nb·NCOMP), threads = (TX, TY, TZ). `NCOMP = 10` puts
# one Π component per thread (the split's second kernel); otherwise 1.

@inline function thread_point(::Val{N}, ::Val{NCOMP}) where {N,NCOMP}
    tx = threadIdx().x
    ty = threadIdx().y
    tz = threadIdx().z
    nxb = Int32(N) ÷ blockDim().x
    bx = blockIdx().x - 1i32
    jb = bx ÷ nxb
    ib = bx - jb * nxb
    i = ib * blockDim().x + tx
    j = jb * blockDim().y + ty
    k = (blockIdx().y - 1i32) * blockDim().z + tz
    bz = blockIdx().z - 1i32
    b = bz ÷ Int32(NCOMP)
    v = bz - b * Int32(NCOMP)
    return Int(i), Int(j), Int(k), Int(b + 1i32), Int(v + 1i32)
end

launch_dims(N, nb, wg, ncomp=1) = ((N ÷ wg[1]) * (N ÷ wg[2]), N ÷ wg[3], nb * ncomp)

@inline function strides_static(::Val{N}) where {N}
    n = N + 2GG + 1
    return (1, n, n * n), n^3, 2NC * n^3
end

@inline work_base(st, sb, i, j, k, b) =
    1 + (b - 1) * sb + (i + GG - 1) * st[1] + (j + GG - 1) * st[2] + (k + GG - 1) * st[3]
@inline du_base(::Val{N}, i, j, k, b) where {N} =
    i + N * (j - 1) + N * N * (k - 1) + 2NC * N^3 * (b - 1)
const NCOEF = 13
@inline coef_base(::Val{N}, i, j, k, b) where {N} =
    i + N * (j - 1) + N * N * (k - 1) + NCOEF * N^3 * (b - 1)

@inline function store!(du, o, N3, off, F::SVector{NC})
    ntuple(Val(NC)) do v
        @inbounds du[o + (off + v - 1) * N3] = F[v]
        nothing
    end
    return nothing
end

# --- R0: the package's per-point body, launched raw ------------------------------

function k_fused!(du, work, spacings, src, γ0, γ2, ε, ::Val{N}, ::Val{STATIC},
                  ::Val{HASH}) where {N,STATIC,HASH}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = STATIC ? strides_static(Val(N)) : TGH.work_strides(work)
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    Fh, FΠ = gh_rhs_at_point(T, wk, src, (i, j, k), b, var, st, sv, inv_h, γ0, γ2,
                             ε * inv_h, Val(Q), Val(HASH), Val(true))
    o = du_base(Val(N), i, j, k, b)
    store!(du, o, N^3, 0, Fh)
    store!(du, o, N^3, NC, FΠ)
    return nothing
end

# --- ablations: the same body with the source or the Π loop removed --------------

@inline function fused_body(wk, src, idx, b, var, st, sv, inv_h, γ0, γ2, εh,
                            ::Val{HASH}, ::Val{SRC}, ::Val{LOOP}) where {HASH,SRC,LOOP}
    inv_h² = inv_h * inv_h
    w1 = derivative_weights(T, Val(Q), Val(1))
    w2 = derivative_weights(T, Val(Q), Val(2))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    divA = SVector{3,T}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)
    acc = ntuple(Val(NC)) do v
        ∂h1 = ∂h[1][v]
        ∂h2 = ∂h[2][v]
        ∂h3 = ∂h[3][v]
        Π_v = Πv[v]
        bh = var + (v - 1) * sv
        bΠ = bh + NC * sv
        ∂ₜh_v = β[1] * ∂h1 + β[2] * ∂h2 + β[3] * ∂h3 + a_div * Π_v
        ∂ₜh_v += εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                       axis_stencil(wD, wk, bh, st[3]))
        if LOOP
            ∂Π1 = inv_h * axis_stencil(w1, wk, bΠ, st[1])
            ∂Π2 = inv_h * axis_stencil(w1, wk, bΠ, st[2])
            ∂Π3 = inv_h * axis_stencil(w1, wk, bΠ, st[3])
            ∂ₜΠ_v = β[1] * ∂Π1 + β[2] * ∂Π2 + β[3] * ∂Π3 + divβ * Π_v +
                    divA[1] * ∂h1 + divA[2] * ∂h2 + divA[3] * ∂h3
            ∂ₜΠ_v += A[1, 1] * (inv_h² * axis_stencil(w2, wk, bh, st[1])) +
                     A[2, 2] * (inv_h² * axis_stencil(w2, wk, bh, st[2])) +
                     A[3, 3] * (inv_h² * axis_stencil(w2, wk, bh, st[3]))
            ∂xy = inv_h² * mixed_stencil(w1, wk, bh, st[1], st[2])
            ∂xz = inv_h² * mixed_stencil(w1, wk, bh, st[1], st[3])
            ∂yz = inv_h² * mixed_stencil(w1, wk, bh, st[2], st[3])
            ∂ₜΠ_v += 2 * (A[1, 2] * ∂xy + A[1, 3] * ∂xz + A[2, 3] * ∂yz)
            ∂ₜΠ_v += εh * (axis_stencil(wD, wk, bΠ, st[1]) + axis_stencil(wD, wk, bΠ, st[2]) +
                           axis_stencil(wD, wk, bΠ, st[3]))
        else
            ∂ₜΠ_v = zero(T)
        end
        (∂ₜh_v, ∂ₜΠ_v)
    end
    ∂ₜh = SVector{NC,T}(ntuple(v -> acc[v][1], Val(NC)))
    ∂ₜΠ = SVector{NC,T}(ntuple(v -> acc[v][2], Val(NC)))
    if SRC
        Hl, dHl = TGH.gauge_source(T, src, idx, b, Val(HASH), hv, ∂ₜh, ∂h)
        msrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)
        return ∂ₜh, ∂ₜΠ + msrc
    else
        return ∂ₜh, ∂ₜΠ
    end
end

function k_ablate!(du, work, spacings, src, γ0, γ2, ε, ::Val{N}, ::Val{HASH}, ::Val{SRC},
                   ::Val{LOOP}) where {N,HASH,SRC,LOOP}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    Fh, FΠ = fused_body(wk, src, (i, j, k), b, var, st, sv, inv_h, γ0, γ2, ε * inv_h,
                        Val(HASH), Val(SRC), Val(LOOP))
    o = du_base(Val(N), i, j, k, b)
    store!(du, o, N^3, 0, Fh)
    store!(du, o, N^3, NC, FΠ)
    return nothing
end

# --- R1: the same arithmetic, reordered — ∂ₜh stored first, then the source, then
#     the Π stencils one component at a time, each stored as it is finished --------

function k_reordered!(du, work, spacings, src, γ0, γ2, ε, ::Val{N},
                      ::Val{HASH}) where {N,HASH}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    inv_h² = inv_h * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    w2 = derivative_weights(T, Val(Q), Val(2))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    ∂ₜh = SVector{NC,T}(ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        s = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        s + εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                  axis_stencil(wD, wk, bh, st[3]))
    end)
    store!(du, o, N3, 0, ∂ₜh)
    Hl, dHl = TGH.gauge_source(T, src, (i, j, k), b, Val(HASH), hv, ∂ₜh, ∂h)
    msrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    divA = SVector{3,T}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)
    ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        bΠ = bh + NC * sv
        ∂ₜΠ_v = pi_component(wk, bh, bΠ, st, inv_h, inv_h², εh, β, divβ, divA, A,
                             Πv[v], ∂h[1][v], ∂h[2][v], ∂h[3][v])
        @inbounds du[o + (NC + v - 1) * N3] = ∂ₜΠ_v + msrc[v]
        nothing
    end
    return nothing
end

# One component of ∂ₜΠ without the source, in the package's summation order.
@inline function pi_component(wk, bh, bΠ, st, inv_h, inv_h², εh, β, divβ, divA, A, Π_v,
                              ∂h1, ∂h2, ∂h3)
    w1 = derivative_weights(T, Val(Q), Val(1))
    w2 = derivative_weights(T, Val(Q), Val(2))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    ∂Π1 = inv_h * axis_stencil(w1, wk, bΠ, st[1])
    ∂Π2 = inv_h * axis_stencil(w1, wk, bΠ, st[2])
    ∂Π3 = inv_h * axis_stencil(w1, wk, bΠ, st[3])
    s = β[1] * ∂Π1 + β[2] * ∂Π2 + β[3] * ∂Π3 + divβ * Π_v +
        divA[1] * ∂h1 + divA[2] * ∂h2 + divA[3] * ∂h3
    s += A[1, 1] * (inv_h² * axis_stencil(w2, wk, bh, st[1])) +
         A[2, 2] * (inv_h² * axis_stencil(w2, wk, bh, st[2])) +
         A[3, 3] * (inv_h² * axis_stencil(w2, wk, bh, st[3]))
    ∂xy = inv_h² * mixed_stencil(w1, wk, bh, st[1], st[2])
    ∂xz = inv_h² * mixed_stencil(w1, wk, bh, st[1], st[3])
    ∂yz = inv_h² * mixed_stencil(w1, wk, bh, st[2], st[3])
    s += 2 * (A[1, 2] * ∂xy + A[1, 3] * ∂xz + A[2, 3] * ∂yz)
    s += εh * (axis_stencil(wD, wk, bΠ, st[1]) + axis_stencil(wD, wk, bΠ, st[2]) +
               axis_stencil(wD, wk, bΠ, st[3]))
    return s
end

# --- R2: split into two kernels ---------------------------------------------------
#
# K1, pointwise: ∂ₜh (with its dissipation) into du[1:10], the source into
# du[11:20], and the 13 coefficients the Π equation's principal part needs into a
# scratch array. K2, stencils: the Π equation's derivative terms, added to du[11:20].

function k_split1!(du, coef, work, spacings, src, γ0, γ2, ε, ::Val{N},
                   ::Val{HASH}) where {N,HASH}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    ∂ₜh = SVector{NC,T}(ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        s = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        s + εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                  axis_stencil(wD, wk, bh, st[3]))
    end)
    store!(du, o, N3, 0, ∂ₜh)
    Hl, dHl = TGH.gauge_source(T, src, (i, j, k), b, Val(HASH), hv, ∂ₜh, ∂h)
    msrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)
    store!(du, o, N3, NC, msrc)
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    c = SVector{NCOEF,T}(β[1], β[2], β[3], divβ,
                         dA[1, 1, 1] + dA[2, 2, 1] + dA[3, 3, 1],
                         dA[1, 1, 2] + dA[2, 2, 2] + dA[3, 3, 2],
                         dA[1, 1, 3] + dA[2, 2, 3] + dA[3, 3, 3],
                         A[1, 1], A[1, 2], A[1, 3], A[2, 2], A[2, 3], A[3, 3])
    oc = coef_base(Val(N), i, j, k, b)
    ntuple(Val(NCOEF)) do n
        @inbounds coef[oc + (n - 1) * N3] = c[n]
        nothing
    end
    return nothing
end

# K2. With NCOMP = 1 one thread loops over the ten components (a real loop, not
# unrolled); with NCOMP = 10 each thread does one.
function k_split2!(du, coef, work, spacings, ε, ::Val{N}, ::Val{NCOMP}) where {N,NCOMP}
    i, j, k, b, vc = thread_point(Val(N), Val(NCOMP))
    wk = Base.Experimental.Const(work)
    cf = Base.Experimental.Const(coef)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    inv_h² = inv_h * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    oc = coef_base(Val(N), i, j, k, b)
    c = SVector{NCOEF,T}(ntuple(n -> (@inbounds cf[oc + (n - 1) * N3]), Val(NCOEF)))
    β = SVector{3,T}(c[1], c[2], c[3])
    divβ = c[4]
    divA = SVector{3,T}(c[5], c[6], c[7])
    A = SMatrix{3,3,T}(c[8], c[9], c[10], c[9], c[11], c[12], c[10], c[12], c[13])
    w1 = derivative_weights(T, Val(Q), Val(1))
    vlo = NCOMP == 1 ? 1 : vc
    vhi = NCOMP == 1 ? NC : vc
    for v in vlo:vhi
        bh = var + (v - 1) * sv
        bΠ = bh + NC * sv
        ∂h1 = inv_h * axis_stencil(w1, wk, bh, st[1])
        ∂h2 = inv_h * axis_stencil(w1, wk, bh, st[2])
        ∂h3 = inv_h * axis_stencil(w1, wk, bh, st[3])
        Π_v = @inbounds wk[bΠ]
        s = pi_component(wk, bh, bΠ, st, inv_h, inv_h², εh, β, divβ, divA, A, Π_v,
                         ∂h1, ∂h2, ∂h3)
        @inbounds du[o + (NC + v - 1) * N3] = s + du[o + (NC + v - 1) * N3]
    end
    return nothing
end

# --- round 3: stencils that skip the zero weights, and a shared-memory K2 -------
#
# `axis_stencil` and `mixed_stencil` multiply the first derivative's zero center
# weight like any other — IEEE forbids dropping `0 * x` — so a fourth of the mixed
# derivative's loads and a fifth of the first derivative's are spent on zeros. These
# keep the package's left-fold order and skip the zeros (q = 4 only): bit for bit the
# package's sum whenever the data are finite, up to the sign of a zero.

@inline d1(::Val{:dense}, w, a, base, s) = axis_stencil(w, a, base, s)
@inline function d1(::Val{:sparse}, w, a, base, s)
    @inbounds ((w[1] * a[base - 2s] + w[2] * a[base - s]) + w[4] * a[base + s]) +
              w[5] * a[base + 2s]
end
@inline dmix(::Val{:dense}, w, a, base, s1, s2) = mixed_stencil(w, a, base, s1, s2)
@inline function dmix(::Val{:sparse}, w, a, base, s1, s2)
    inner(o) = @inbounds ((w[1] * a[o - 2s2] + w[2] * a[o - s2]) + w[4] * a[o + s2]) +
                         w[5] * a[o + 2s2]
    return ((w[1] * inner(base - 2s1) + w[2] * inner(base - s1)) + w[4] * inner(base + s1)) +
           w[5] * inner(base + 2s1)
end

# The h-side stencils of one component: ∂h (3), ∂∂h on the axes (3), mixed (3).
@inline function h_terms(fl, ah, bh, sth, inv_h, inv_h²)
    w1 = derivative_weights(T, Val(Q), Val(1))
    w2 = derivative_weights(T, Val(Q), Val(2))
    return (inv_h * d1(fl, w1, ah, bh, sth[1]), inv_h * d1(fl, w1, ah, bh, sth[2]),
            inv_h * d1(fl, w1, ah, bh, sth[3]),
            inv_h² * axis_stencil(w2, ah, bh, sth[1]), inv_h² * axis_stencil(w2, ah, bh, sth[2]),
            inv_h² * axis_stencil(w2, ah, bh, sth[3]),
            inv_h² * dmix(fl, w1, ah, bh, sth[1], sth[2]),
            inv_h² * dmix(fl, w1, ah, bh, sth[1], sth[3]),
            inv_h² * dmix(fl, w1, ah, bh, sth[2], sth[3]))
end
# The Π-side stencils of one component: Π, ∂Π (3), the dissipation (3).
@inline function p_terms(fl, aΠ, bΠ, stΠ, inv_h)
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    return (@inbounds(aΠ[bΠ]), inv_h * d1(fl, w1, aΠ, bΠ, stΠ[1]),
            inv_h * d1(fl, w1, aΠ, bΠ, stΠ[2]), inv_h * d1(fl, w1, aΠ, bΠ, stΠ[3]),
            axis_stencil(wD, aΠ, bΠ, stΠ[1]), axis_stencil(wD, aΠ, bΠ, stΠ[2]),
            axis_stencil(wD, aΠ, bΠ, stΠ[3]))
end
# Combined in the package's summation order.
@inline function pi_combine(ht, pt, εh, β, divβ, divA, A)
    ∂h1, ∂h2, ∂h3, ∂11, ∂22, ∂33, ∂xy, ∂xz, ∂yz = ht
    Π_v, ∂Π1, ∂Π2, ∂Π3, D1, D2, D3 = pt
    s = β[1] * ∂Π1 + β[2] * ∂Π2 + β[3] * ∂Π3 + divβ * Π_v +
        divA[1] * ∂h1 + divA[2] * ∂h2 + divA[3] * ∂h3
    s += A[1, 1] * ∂11 + A[2, 2] * ∂22 + A[3, 3] * ∂33
    s += 2 * (A[1, 2] * ∂xy + A[1, 3] * ∂xz + A[2, 3] * ∂yz)
    s += εh * (D1 + D2 + D3)
    return s
end

@inline function load_coef(cf, oc, N3)
    c = SVector{NCOEF,T}(ntuple(n -> (@inbounds cf[oc + (n - 1) * N3]), Val(NCOEF)))
    β = SVector{3,T}(c[1], c[2], c[3])
    A = SMatrix{3,3,T}(c[8], c[9], c[10], c[9], c[11], c[12], c[10], c[12], c[13])
    return β, c[4], SVector{3,T}(c[5], c[6], c[7]), A
end

# K2 from global memory (through L1), either flavor.
function k_split2f!(du, coef, work, spacings, ε, ::Val{N}, ::Val{NCOMP},
                    ::Val{FL}) where {N,NCOMP,FL}
    i, j, k, b, vc = thread_point(Val(N), Val(NCOMP))
    wk = Base.Experimental.Const(work)
    cf = Base.Experimental.Const(coef)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    inv_h² = inv_h * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    β, divβ, divA, A = load_coef(cf, coef_base(Val(N), i, j, k, b), N3)
    vlo = NCOMP == 1 ? 1 : vc
    vhi = NCOMP == 1 ? NC : vc
    for v in vlo:vhi
        bh = var + (v - 1) * sv
        ht = h_terms(Val(FL), wk, bh, st, inv_h, inv_h²)
        pt = p_terms(Val(FL), wk, bh + NC * sv, st, inv_h)
        s = pi_combine(ht, pt, εh, β, divβ, divA, A)
        @inbounds du[o + (NC + v - 1) * N3] = s + du[o + (NC + v - 1) * N3]
    end
    return nothing
end

# K2 with one variable's tile staged in shared memory at a time: per component, the
# h tile (halo G) is loaded and its nine stencils taken, then the Π tile and its
# seven. The tile is the thread block's, (TX, TY, TZ) points plus the halo.
function k_split2s!(du, coef, work, spacings, ε, ::Val{N}, ::Val{TX}, ::Val{TY},
                    ::Val{TZ}, ::Val{FL}) where {N,TX,TY,TZ,FL}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    tx = Int(threadIdx().x)
    ty = Int(threadIdx().y)
    tz = Int(threadIdx().z)
    SX = TX + 2GG
    SY = TY + 2GG
    SZ = TZ + 2GG
    SN = SX * SY * SZ
    tile = CuStaticSharedArray(T, SN)
    wk = Base.Experimental.Const(work)
    cf = Base.Experimental.Const(coef)
    st, sv, sb = strides_static(Val(N))
    # The tile's lower corner, halo included, is at the stored index equal to the
    # owned index of its first point.
    base0 = 1 + (b - 1) * sb + (i - tx) + (j - ty) * st[2] + (k - tz) * st[3]
    tid = (tx - 1) + TX * ((ty - 1) + TY * (tz - 1))
    NT = TX * TY * TZ
    sst = (1, SX, SX * SY)
    ps = 1 + (tx - 1 + GG) + SX * (ty - 1 + GG) + SX * SY * (tz - 1 + GG)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    inv_h² = inv_h * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    β, divβ, divA, A = load_coef(cf, coef_base(Val(N), i, j, k, b), N3)
    for v in 1:NC
        vb = base0 + (v - 1) * sv
        sync_threads()
        n = tid
        while n < SN
            a = n % SX
            r = n ÷ SX
            c2 = r % SY
            c3 = r ÷ SY
            @inbounds tile[n + 1] = wk[vb + a + c2 * st[2] + c3 * st[3]]
            n += NT
        end
        sync_threads()
        ht = h_terms(Val(FL), tile, ps, sst, inv_h, inv_h²)
        sync_threads()
        n = tid
        while n < SN
            a = n % SX
            r = n ÷ SX
            c2 = r % SY
            c3 = r ÷ SY
            @inbounds tile[n + 1] = wk[vb + NC * sv + a + c2 * st[2] + c3 * st[3]]
            n += NT
        end
        sync_threads()
        pt = p_terms(Val(FL), tile, ps, sst, inv_h)
        s = pi_combine(ht, pt, εh, β, divβ, divA, A)
        @inbounds du[o + (NC + v - 1) * N3] = s + du[o + (NC + v - 1) * N3]
    end
    return nothing
end

# K1 with either flavor for the 30 first derivatives.
function k_split1f!(du, coef, work, spacings, src, γ0, γ2, ε, ::Val{N}, ::Val{HASH},
                    ::Val{FL}) where {N,HASH,FL}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            d1(Val(FL), w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    ∂ₜh = SVector{NC,T}(ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        s = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        s + εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                  axis_stencil(wD, wk, bh, st[3]))
    end)
    store!(du, o, N3, 0, ∂ₜh)
    Hl, dHl = TGH.gauge_source(T, src, (i, j, k), b, Val(HASH), hv, ∂ₜh, ∂h)
    msrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)
    store!(du, o, N3, NC, msrc)
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    c = SVector{NCOEF,T}(β[1], β[2], β[3], divβ,
                         dA[1, 1, 1] + dA[2, 2, 1] + dA[3, 3, 1],
                         dA[1, 1, 2] + dA[2, 2, 2] + dA[3, 3, 2],
                         dA[1, 1, 3] + dA[2, 2, 3] + dA[3, 3, 3],
                         A[1, 1], A[1, 2], A[1, 3], A[2, 2], A[2, 3], A[3, 3])
    oc = coef_base(Val(N), i, j, k, b)
    ntuple(Val(NCOEF)) do n
        @inbounds coef[oc + (n - 1) * N3] = c[n]
        nothing
    end
    return nothing
end

# The fused kernel with sparse stencils: the package's body with `d1`/`dmix` in
# place of the dense contractions, everything else as `gh_rhs_at_point`.
function k_fusedf!(du, work, spacings, src, γ0, γ2, ε, ::Val{N}, ::Val{HASH},
                   ::Val{FL}) where {N,HASH,FL}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    inv_h² = inv_h * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            d1(Val(FL), w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    divA = SVector{3,T}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)
    acc = ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        bΠ = bh + NC * sv
        ∂ₜh_v = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        ∂ₜh_v += εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                       axis_stencil(wD, wk, bh, st[3]))
        ht = h_terms(Val(FL), wk, bh, st, inv_h, inv_h²)
        pt = p_terms(Val(FL), wk, bΠ, st, inv_h)
        (∂ₜh_v, pi_combine(ht, pt, εh, β, divβ, divA, A))
    end
    ∂ₜh = SVector{NC,T}(ntuple(v -> acc[v][1], Val(NC)))
    ∂ₜΠ = SVector{NC,T}(ntuple(v -> acc[v][2], Val(NC)))
    Hl, dHl = TGH.gauge_source(T, src, (i, j, k), b, Val(HASH), hv, ∂ₜh, ∂h)
    msrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)
    store!(du, o, N3, 0, ∂ₜh)
    store!(du, o, N3, NC, ∂ₜΠ + msrc)
    return nothing
end

# --- round 4: the source alone ------------------------------------------------
#
# K1a: everything of K1 but the source — ∂ₜh into du[1:10], the 30 first
# derivatives into a scratch array, the 13 coefficients. K1b: the source alone, from
# h (the working array), ∂h (scratch) and ∂ₜh (du), into du[11:20].

function k_split1a!(du, dh, coef, work, spacings, ε, ::Val{N}, ::Val{FL}) where {N,FL}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            d1(Val(FL), w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    ∂ₜh = SVector{NC,T}(ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        s = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        s + εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                  axis_stencil(wD, wk, bh, st[3]))
    end)
    store!(du, o, N3, 0, ∂ₜh)
    od = i + N * (j - 1) + N * N * (k - 1) + 3NC * N3 * (b - 1)
    for d in 1:3
        store!(dh, od, N3, (d - 1) * NC, ∂h[d])
    end
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    c = SVector{NCOEF,T}(β[1], β[2], β[3], divβ,
                         dA[1, 1, 1] + dA[2, 2, 1] + dA[3, 3, 1],
                         dA[1, 1, 2] + dA[2, 2, 2] + dA[3, 3, 2],
                         dA[1, 1, 3] + dA[2, 2, 3] + dA[3, 3, 3],
                         A[1, 1], A[1, 2], A[1, 3], A[2, 2], A[2, 3], A[3, 3])
    oc = coef_base(Val(N), i, j, k, b)
    ntuple(Val(NCOEF)) do n
        @inbounds coef[oc + (n - 1) * N3] = c[n]
        nothing
    end
    return nothing
end

function k_split1b!(du, dh, work, src, γ0, γ2, ::Val{N}, ::Val{HASH}) where {N,HASH}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    dk = Base.Experimental.Const(dh)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    od = i + N * (j - 1) + N * N * (k - 1) + 3NC * N3 * (b - 1)
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        SVector{NC,T}(ntuple(v -> (@inbounds dk[od + ((d - 1) * NC + v - 1) * N3]), Val(NC)))
    end
    ∂ₜh = SVector{NC,T}(ntuple(v -> (@inbounds du[o + (v - 1) * N3]), Val(NC)))
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    Hl, dHl = TGH.gauge_source(T, src, (i, j, k), b, Val(HASH), hv, ∂ₜh, ∂h)
    msrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)
    store!(du, o, N3, NC, msrc)
    return nothing
end

include(joinpath(@__DIR__, "rhs_lab_source.jl"))

function k_split1b_lean!(du, dh, work, src, γ0, γ2, ::Val{N}, ::Val{HASH}) where {N,HASH}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    dk = Base.Experimental.Const(dh)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    od = i + N * (j - 1) + N * N * (k - 1) + 3NC * N3 * (b - 1)
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        SVector{NC,T}(ntuple(v -> (@inbounds dk[od + ((d - 1) * NC + v - 1) * N3]), Val(NC)))
    end
    ∂ₜh = SVector{NC,T}(ntuple(v -> (@inbounds du[o + (v - 1) * N3]), Val(NC)))
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    Hl, dHl = TGH.gauge_source(T, src, (i, j, k), b, Val(HASH), hv, ∂ₜh, ∂h)
    msrc = lean_source(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ,
                       (∂ₜh, ∂h[1], ∂h[2], ∂h[3]), Hl, dHl, γ0, γ2)
    store!(du, o, N3, NC, msrc)
    return nothing
end

function round4(N, roots)
    case, forest, U, u, p = setup(N, roots)
    nb = nblocks(U)
    npts = nb * N^3
    @printf("# round4 N=%d roots=%d blocks=%d points=%.3e  %s  inline=%s\n", N, roots, nb,
            npts, CUDA.name(CUDA.device()), INLINE)
    du = similar(u)
    ref = similar(u)
    gh_rhs!(ref, u, p, zero(T))
    CUDA.synchronize()
    work = p.U.work
    sp = p.spacings
    γ0, γ2, ε = one(T), zero(T), T(1 // 2)
    wgs = filter(wg -> wg[1] <= N, ((32, 8, 1), (32, 4, 1), (32, 2, 2), (16, 8, 1),
                                    (64, 2, 1), (64, 4, 1), (128, 1, 1)))
    coef = CUDA.zeros(T, NCOEF * npts)
    dh = CUDA.zeros(T, 3NC * npts)
    ka = run_raw("K1a sparse", k_split1a!, (du, dh, coef, work, sp, ε, Val(N), Val(:sparse)),
                 N, nb, wgs)
    pre_a() = ka(du, dh, coef, work, sp, ε, Val(N), Val(:sparse); threads=(32, 4, 1),
                 blocks=launch_dims(N, nb, (32, 4, 1)))
    kb = run_raw("K1b source", k_split1b!, (du, dh, work, nothing, γ0, γ2, Val(N), Val(false)),
                 N, nb, wgs; pre=pre_a)
    for mr in (128, 168)
        run_raw("K1b source", k_split1b!, (du, dh, work, nothing, γ0, γ2, Val(N), Val(false)),
                N, nb, ((32, 4, 1), (32, 8, 1), (64, 2, 1), (128, 1, 1)); maxregs=mr, pre=pre_a)
    end
    if "alg" in split(opt("which", ""), ",")
        src = KerrSchildSource(T; M=1)
        run_raw("K1b source algebraic", k_split1b!, (du, dh, work, src, γ0, γ2, Val(N),
                Val(:algebraic)), N, nb, ((32, 4, 1), (32, 8, 1)); pre=pre_a)
    end
    kl = run_raw("K1b lean source", k_split1b_lean!,
                 (du, dh, work, nothing, γ0, γ2, Val(N), Val(false)), N, nb, wgs; pre=pre_a)
    for mr in (128, 168)
        run_raw("K1b lean source", k_split1b_lean!,
                (du, dh, work, nothing, γ0, γ2, Val(N), Val(false)), N, nb,
                ((32, 4, 1), (32, 8, 1), (64, 2, 1), (128, 1, 1)); maxregs=mr, pre=pre_a)
    end
    pre_l() = (pre_a(); kl(du, dh, work, nothing, γ0, γ2, Val(N), Val(false);
                           threads=(32, 4, 1), blocks=launch_dims(N, nb, (32, 4, 1))))
    run_raw("K2 sparse (after K1a+K1b lean)", k_split2f!,
            (du, coef, work, sp, ε, Val(N), Val(1), Val(:sparse)), N, nb, ((32, 4, 1),);
            pre=pre_l, ref=ref, du=du)
    pre_b() = (pre_a(); kb(du, dh, work, nothing, γ0, γ2, Val(N), Val(false);
                           threads=(32, 4, 1), blocks=launch_dims(N, nb, (32, 4, 1))))
    run_raw("K2 sparse ncomp=1 (after K1a+K1b)", k_split2f!,
            (du, coef, work, sp, ε, Val(N), Val(1), Val(:sparse)), N, nb, ((32, 4, 1), (32, 8, 1));
            pre=pre_b, ref=ref, du=du)
    return nothing
end

# --- round 5: the pipeline around the kernel, and block sizes ----------------------

# The scatter with static strides: the state's 20 values into the working array.
function k_scatter!(work, u, ::Val{N}) where {N}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    o = du_base(Val(N), i, j, k, b)
    uk = Base.Experimental.Const(u)
    ntuple(Val(2NC)) do v
        @inbounds work[var + (v - 1) * sv] = uk[o + (v - 1) * N^3]
        nothing
    end
    return nothing
end

# One RK stage's arithmetic fused with the scatter: `acc += bdt k` and the next
# stage's input `y + adt k` written straight into the working array's interior.
function k_stage!(work, acc, y, kv, adt, bdt, ::Val{N}) where {N}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    o = du_base(Val(N), i, j, k, b)
    yk = Base.Experimental.Const(y)
    kk = Base.Experimental.Const(kv)
    ntuple(Val(2NC)) do v
        n = o + (v - 1) * N^3
        @inbounds kn = kk[n]
        @inbounds acc[n] += bdt * kn
        @inbounds work[var + (v - 1) * sv] = yk[n] + adt * kn
        nothing
    end
    return nothing
end

# --- round 6: the lean source back in the two-kernel split and in the fused kernel --

@inline pick_source(::Val{:pkg}, g4, gu4, α, sqrtγ, ∂ₜh, ∂h, Hl, dHl, γ0, γ2) =
    gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)
@inline pick_source(::Val{:lean}, g4, gu4, α, sqrtγ, ∂ₜh, ∂h, Hl, dHl, γ0, γ2) =
    lean_source(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ, (∂ₜh, ∂h[1], ∂h[2], ∂h[3]),
                Hl, dHl, γ0, γ2)

function k_split1s!(du, coef, work, spacings, src, γ0, γ2, ε, ::Val{N}, ::Val{HASH},
                    ::Val{SRCK}) where {N,HASH,SRCK}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            d1(Val(:sparse), w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    ∂ₜh = SVector{NC,T}(ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        s = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        s + εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                  axis_stencil(wD, wk, bh, st[3]))
    end)
    store!(du, o, N3, 0, ∂ₜh)
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    c = SVector{NCOEF,T}(β[1], β[2], β[3], divβ,
                         dA[1, 1, 1] + dA[2, 2, 1] + dA[3, 3, 1],
                         dA[1, 1, 2] + dA[2, 2, 2] + dA[3, 3, 2],
                         dA[1, 1, 3] + dA[2, 2, 3] + dA[3, 3, 3],
                         A[1, 1], A[1, 2], A[1, 3], A[2, 2], A[2, 3], A[3, 3])
    oc = coef_base(Val(N), i, j, k, b)
    ntuple(Val(NCOEF)) do n
        @inbounds coef[oc + (n - 1) * N3] = c[n]
        nothing
    end
    Hl, dHl = TGH.gauge_source(T, src, (i, j, k), b, Val(HASH), hv, ∂ₜh, ∂h)
    msrc = pick_source(Val(SRCK), g4, gu4, α, sqrtγ, ∂ₜh, ∂h, Hl, dHl, γ0, γ2)
    store!(du, o, N3, NC, msrc)
    return nothing
end

function k_fuseds!(du, work, spacings, src, γ0, γ2, ε, ::Val{N}, ::Val{HASH},
                   ::Val{SRCK}) where {N,HASH,SRCK}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    inv_h² = inv_h * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            d1(Val(:sparse), w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    ∂ₜh = SVector{NC,T}(ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        s = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        s + εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                  axis_stencil(wD, wk, bh, st[3]))
    end)
    store!(du, o, N3, 0, ∂ₜh)
    Hl, dHl = TGH.gauge_source(T, src, (i, j, k), b, Val(HASH), hv, ∂ₜh, ∂h)
    msrc = pick_source(Val(SRCK), g4, gu4, α, sqrtγ, ∂ₜh, ∂h, Hl, dHl, γ0, γ2)
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    divA = SVector{3,T}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)
    for v in 1:NC
        bh = var + (v - 1) * sv
        ht = h_terms(Val(:sparse), wk, bh, st, inv_h, inv_h²)
        pt = p_terms(Val(:sparse), wk, bh + NC * sv, st, inv_h)
        @inbounds du[o + (NC + v - 1) * N3] = pi_combine(ht, pt, εh, β, divβ, divA, A) +
                                               msrc[v]
    end
    return nothing
end

function round6(N, roots)
    case, forest, U, u, p = setup(N, roots)
    nb = nblocks(U)
    npts = nb * N^3
    @printf("# round6 N=%d roots=%d blocks=%d points=%.3e  %s  inline=%s\n", N, roots, nb,
            npts, CUDA.name(CUDA.device()), INLINE)
    du = similar(u)
    ref = similar(u)
    gh_rhs!(ref, u, p, zero(T))
    CUDA.synchronize()
    work = p.U.work
    sp = p.spacings
    γ0, γ2, ε = one(T), zero(T), T(1 // 2)
    wgs = filter(wg -> wg[1] <= N, ((32, 4, 1), (32, 2, 2), (32, 8, 1), (16, 8, 1)))
    wg = N >= 32 ? (32, 4, 1) : (16, 8, 1)
    coef = CUDA.zeros(T, NCOEF * npts)
    dh = CUDA.zeros(T, 3NC * npts)
    for srck in (:lean, :pkg)
        run_raw("fused, $srck source", k_fuseds!,
                (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(srck)), N, nb, wgs;
                ref=ref, du=du)
    end
    k1 = run_raw("K1, lean source", k_split1s!,
                 (du, coef, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(:lean)),
                 N, nb, wgs)
    pre1() = k1(du, coef, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(:lean);
                threads=wg, blocks=launch_dims(N, nb, wg))
    run_raw("K2 after K1 lean", k_split2f!, (du, coef, work, sp, ε, Val(N), Val(1),
            Val(:sparse)), N, nb, (wg,); pre=pre1, ref=ref, du=du)
    ka = run_raw("K1a", k_split1a!, (du, dh, coef, work, sp, ε, Val(N), Val(:sparse)), N, nb,
                 (wg,))
    pre_a() = ka(du, dh, coef, work, sp, ε, Val(N), Val(:sparse); threads=wg,
                 blocks=launch_dims(N, nb, wg))
    run_raw("K1b lean", k_split1b_lean!, (du, dh, work, nothing, γ0, γ2, Val(N), Val(false)),
            N, nb, (wg,); pre=pre_a)
    if "alg" in split(opt("which", ""), ",")
        srcK = KerrSchildSource(T; M=1)
        run_raw("fused lean + algebraic source", k_fuseds!,
                (du, work, sp, srcK, γ0, γ2, ε, Val(N), Val(:algebraic), Val(:lean)), N, nb,
                (wg,))
        run_raw("K1 lean + algebraic source", k_split1s!,
                (du, coef, work, sp, srcK, γ0, γ2, ε, Val(N), Val(:algebraic), Val(:lean)),
                N, nb, (wg,))
        run_raw("K1b lean + algebraic source", k_split1b_lean!,
                (du, dh, work, srcK, γ0, γ2, Val(N), Val(:algebraic)), N, nb, (wg,);
                pre=pre_a)
    end
    run_raw("K2 (for SASS)", k_split2f!, (du, coef, work, sp, ε, Val(N), Val(1), Val(:sparse)),
            N, nb, ())
    return nothing
end

# --- round 7: the split as KernelAbstractions kernels ---------------------------
#
# The same three bodies, reached through `@index(Global, NTuple)` over `(N, N, N,
# nblocks)` — what `map_blocks!` launches — instead of raw CUDA indexing.

@inline function body_1a!(du, dh, coef, work, spacings, ε, i, j, k, b, ::Val{N}) where {N}
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            d1(Val(:sparse), w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    ∂ₜh = SVector{NC,T}(ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        s = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        s + εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                  axis_stencil(wD, wk, bh, st[3]))
    end)
    store!(du, o, N3, 0, ∂ₜh)
    od = i + N * (j - 1) + N * N * (k - 1) + 3NC * N3 * (b - 1)
    store!(dh, od, N3, 0, ∂h[1])
    store!(dh, od, N3, NC, ∂h[2])
    store!(dh, od, N3, 2NC, ∂h[3])
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    c = SVector{NCOEF,T}(β[1], β[2], β[3], divβ,
                         dA[1, 1, 1] + dA[2, 2, 1] + dA[3, 3, 1],
                         dA[1, 1, 2] + dA[2, 2, 2] + dA[3, 3, 2],
                         dA[1, 1, 3] + dA[2, 2, 3] + dA[3, 3, 3],
                         A[1, 1], A[1, 2], A[1, 3], A[2, 2], A[2, 3], A[3, 3])
    oc = coef_base(Val(N), i, j, k, b)
    store_coef!(coef, oc, N3, c)
    return nothing
end

@inline function store_coef!(coef, oc, N3, c)
    ntuple(Val(NCOEF)) do n
        @inbounds coef[oc + (n - 1) * N3] = c[n]
        nothing
    end
    return nothing
end

@inline function body_1b!(du, dh, work, γ0, γ2, i, j, k, b, ::Val{N}) where {N}
    wk = Base.Experimental.Const(work)
    dk = Base.Experimental.Const(dh)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    od = i + N * (j - 1) + N * N * (k - 1) + 3NC * N3 * (b - 1)
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        SVector{NC,T}(ntuple(v -> (@inbounds dk[od + ((d - 1) * NC + v - 1) * N3]), Val(NC)))
    end
    ∂ₜh = SVector{NC,T}(ntuple(v -> (@inbounds du[o + (v - 1) * N3]), Val(NC)))
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    msrc = lean_source(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ,
                       (∂ₜh, ∂h[1], ∂h[2], ∂h[3]), zero(SVector{4,T}),
                       zero(SMatrix{4,4,T}), γ0, γ2)
    store!(du, o, N3, NC, msrc)
    return nothing
end

@inline function body_2!(du, coef, work, spacings, ε, i, j, k, b, ::Val{N}) where {N}
    wk = Base.Experimental.Const(work)
    cf = Base.Experimental.Const(coef)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    inv_h² = inv_h * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    β, divβ, divA, A = load_coef(cf, coef_base(Val(N), i, j, k, b), N3)
    for v in 1:NC
        bh = var + (v - 1) * sv
        ht = h_terms(Val(:sparse), wk, bh, st, inv_h, inv_h²)
        pt = p_terms(Val(:sparse), wk, bh + NC * sv, st, inv_h)
        s = pi_combine(ht, pt, εh, β, divβ, divA, A)
        @inbounds du[o + (NC + v - 1) * N3] = s + du[o + (NC + v - 1) * N3]
    end
    return nothing
end

@kernel function ka_1a!(du, dh, coef, work, spacings, ε, ::Val{N}) where {N}
    I = @index(Global, NTuple)
    body_1a!(du, dh, coef, work, spacings, ε, I[1], I[2], I[3], I[4], Val(N))
end
@kernel function ka_1b!(du, dh, work, γ0, γ2, ::Val{N}) where {N}
    I = @index(Global, NTuple)
    body_1b!(du, dh, work, γ0, γ2, I[1], I[2], I[3], I[4], Val(N))
end
@kernel function ka_2!(du, coef, work, spacings, ε, ::Val{N}) where {N}
    I = @index(Global, NTuple)
    body_2!(du, coef, work, spacings, ε, I[1], I[2], I[3], I[4], Val(N))
end

# The package's `gh_rhs_at_point`, verbatim but for the source: `lean_source` in place
# of `gh_node_source`. With the package's kernel shape around it (dynamic strides from
# `size(work)`, 5-index bounds-checked stores of `du`), this is the minimal change.
@inline function rhs_at_point_lean(::Type{T}, work, inner, b::Int, var::Int, st, sv::Int,
                                   inv_h, γ0, γ2, εh) where {T}
    inv_h² = inv_h * inv_h
    w1 = derivative_weights(T, Val(Q), Val(1))
    w2 = derivative_weights(T, Val(Q), Val(2))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, work, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    divA = SVector{3,T}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)
    acc = ntuple(Val(NC)) do v
        ∂h1 = ∂h[1][v]
        ∂h2 = ∂h[2][v]
        ∂h3 = ∂h[3][v]
        Π_v = Πv[v]
        bh = var + (v - 1) * sv
        bΠ = bh + NC * sv
        ∂ₜh_v = β[1] * ∂h1 + β[2] * ∂h2 + β[3] * ∂h3 + a_div * Π_v
        ∂Π1 = inv_h * axis_stencil(w1, work, bΠ, st[1])
        ∂Π2 = inv_h * axis_stencil(w1, work, bΠ, st[2])
        ∂Π3 = inv_h * axis_stencil(w1, work, bΠ, st[3])
        ∂ₜΠ_v = β[1] * ∂Π1 + β[2] * ∂Π2 + β[3] * ∂Π3 + divβ * Π_v +
                divA[1] * ∂h1 + divA[2] * ∂h2 + divA[3] * ∂h3
        ∂ₜΠ_v += A[1, 1] * (inv_h² * axis_stencil(w2, work, bh, st[1])) +
                 A[2, 2] * (inv_h² * axis_stencil(w2, work, bh, st[2])) +
                 A[3, 3] * (inv_h² * axis_stencil(w2, work, bh, st[3]))
        ∂xy = inv_h² * mixed_stencil(w1, work, bh, st[1], st[2])
        ∂xz = inv_h² * mixed_stencil(w1, work, bh, st[1], st[3])
        ∂yz = inv_h² * mixed_stencil(w1, work, bh, st[2], st[3])
        ∂ₜΠ_v += 2 * (A[1, 2] * ∂xy + A[1, 3] * ∂xz + A[2, 3] * ∂yz)
        ∂ₜh_v += εh * (axis_stencil(wD, work, bh, st[1]) + axis_stencil(wD, work, bh, st[2]) +
                       axis_stencil(wD, work, bh, st[3]))
        ∂ₜΠ_v += εh * (axis_stencil(wD, work, bΠ, st[1]) + axis_stencil(wD, work, bΠ, st[2]) +
                       axis_stencil(wD, work, bΠ, st[3]))
        (∂ₜh_v, ∂ₜΠ_v)
    end
    ∂ₜh = SVector{NC,T}(ntuple(v -> acc[v][1], Val(NC)))
    ∂ₜΠ = SVector{NC,T}(ntuple(v -> acc[v][2], Val(NC)))
    msrc = lean_source(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ,
                       (∂ₜh, ∂h[1], ∂h[2], ∂h[3]), zero(SVector{4,T}), zero(SMatrix{4,4,T}),
                       γ0, γ2)
    return ∂ₜh, ∂ₜΠ + msrc
end

# The package kernel's shape: dynamic strides, `du[i, j, k, v, b] = …` checked.
@kernel function ka_pkgshape_lean!(du, @Const(work), @Const(spacings), γ0, γ2, ε,
                                   ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    inv_h = inv(spacings[b])
    st, sv, sb = TGH.work_strides(work)
    var = 1 + (b - 1) * sb + (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]
    Fh, FΠ = rhs_at_point_lean(Float64, work, inner, b, var, st, sv, inv_h, γ0, γ2,
                               ε * inv_h)
    ntuple(Val(NC)) do v
        du[inner..., v, b] = Fh[v]
        du[inner..., NC + v, b] = FΠ[v]
        nothing
    end
end

# --- round 8: what the fused lean kernel needs, as a KA kernel ---------------------
#
# The body of `k_fuseds!` (∂ₜh stored first, then the source, then the Π components)
# with three switches: static or dynamic strides, a runtime loop or an unrolled
# `ntuple` over the Π components, sparse or dense stencils. ORDER = :late computes the
# source after the Π components instead (the package's order): the ten ∂ₜΠ are
# written to du first and the source is added to them at the end.

@inline function body_fl!(du, work, spacings, γ0, γ2, ε, i, j, k, b, ::Val{N}, ::Val{STATIC},
                          ::Val{UNROLL}, ::Val{FL}, ::Val{ORDER}) where {N,STATIC,UNROLL,FL,ORDER}
    wk = Base.Experimental.Const(work)
    st, sv, sb = STATIC ? strides_static(Val(N)) : TGH.work_strides(work)
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    inv_h² = inv_h * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            d1(Val(FL), w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    ∂ₜh = SVector{NC,T}(ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        s = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        s + εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                  axis_stencil(wD, wk, bh, st[3]))
    end)
    store!(du, o, N3, 0, ∂ₜh)
    msrc = ORDER === :early ?
           lean_source(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ,
                       (∂ₜh, ∂h[1], ∂h[2], ∂h[3]), zero(SVector{4,T}), zero(SMatrix{4,4,T}),
                       γ0, γ2) : zero(SVector{NC,T})
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    divA = SVector{3,T}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)
    if UNROLL
        ntuple(Val(NC)) do v
            bh = var + (v - 1) * sv
            ht = h_terms(Val(FL), wk, bh, st, inv_h, inv_h²)
            pt = p_terms(Val(FL), wk, bh + NC * sv, st, inv_h)
            @inbounds du[o + (NC + v - 1) * N3] = pi_combine(ht, pt, εh, β, divβ, divA, A) +
                                                   msrc[v]
            nothing
        end
    else
        for v in 1:NC
            bh = var + (v - 1) * sv
            ht = h_terms(Val(FL), wk, bh, st, inv_h, inv_h²)
            pt = p_terms(Val(FL), wk, bh + NC * sv, st, inv_h)
            @inbounds du[o + (NC + v - 1) * N3] = pi_combine(ht, pt, εh, β, divβ, divA, A) +
                                                   msrc[v]
        end
    end
    if ORDER === :late
        msrc2 = lean_source(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ,
                            (∂ₜh, ∂h[1], ∂h[2], ∂h[3]), zero(SVector{4,T}),
                            zero(SMatrix{4,4,T}), γ0, γ2)
        ntuple(Val(NC)) do v
            @inbounds du[o + (NC + v - 1) * N3] += msrc2[v]
            nothing
        end
    end
    return nothing
end

@kernel function ka_fl!(du, work, spacings, γ0, γ2, ε, ::Val{N}, ::Val{STATIC}, ::Val{UNROLL},
                        ::Val{FL}, ::Val{ORDER}) where {N,STATIC,UNROLL,FL,ORDER}
    I = @index(Global, NTuple)
    body_fl!(du, work, spacings, γ0, γ2, ε, I[1], I[2], I[3], I[4], Val(N), Val(STATIC),
             Val(UNROLL), Val(FL), Val(ORDER))
end

function round8(N, roots)
    case, forest, U, u, p = setup(N, roots)
    nb = nblocks(U)
    npts = nb * N^3
    @printf("# round8 N=%d roots=%d blocks=%d points=%.3e  %s\n", N, roots, nb, npts,
            CUDA.name(CUDA.device()))
    du = similar(u)
    ref = similar(u)
    gh_rhs!(ref, u, p, zero(T))
    CUDA.synchronize()
    work = p.U.work
    sp = p.spacings
    γ0, γ2, ε = one(T), zero(T), T(1 // 2)
    nd = (N, N, N, nb)
    wg = (min(N, 32), 4, 1, 1)
    configs = ((true, false, :sparse, :early), (false, false, :sparse, :early),
               (true, true, :sparse, :early), (true, false, :dense, :early),
               (true, false, :sparse, :late), (false, true, :dense, :late))
    for inl in (true, false), (STATIC, UNROLL, FL, ORDER) in configs
        inl || (STATIC, UNROLL, FL, ORDER) == configs[1] || continue
        k = ka_fl!(CUDABackend(; always_inline=inl), wg)
        launch() = (k(du, work, sp, γ0, γ2, ε, Val(N), Val(STATIC), Val(UNROLL), Val(FL),
                      Val(ORDER); ndrange=nd); CUDA.synchronize())
        t = timeit(launch)
        fill!(du, 0)
        launch()
        e = compare(du, ref, N, nb)
        @printf("  KA fused lean  inl=%-5s static=%-5s unroll=%-5s %-6s %-5s  %8.3f ms  %7.3f ns/pt  err h %.1e Π %.1e\n",
                inl, STATIC, UNROLL, FL, ORDER, 1e3t, 1e9t / npts, e...)
        flush(stdout)
    end
    return nothing
end

# --- round 9: the closure-free lean source (`lean_source_inl`) -------------------

@inline function body_fl2!(du, work, spacings, γ0, γ2, ε, i, j, k, b, ::Val{N}) where {N}
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    inv_h = inv(@inbounds spacings[b])
    εh = ε * inv_h
    inv_h² = inv_h * inv_h
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    w1 = derivative_weights(T, Val(Q), Val(1))
    wD = dissipation_weights(T, dissipation_rank(Val(Q)))
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (NC + v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            d1(Val(:sparse), w1, wk, var + (v - 1) * sv, st[d])
        end)
    end
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    ∂ₜh = SVector{NC,T}(ntuple(Val(NC)) do v
        bh = var + (v - 1) * sv
        s = β[1] * ∂h[1][v] + β[2] * ∂h[2][v] + β[3] * ∂h[3][v] + a_div * Πv[v]
        s + εh * (axis_stencil(wD, wk, bh, st[1]) + axis_stencil(wD, wk, bh, st[2]) +
                  axis_stencil(wD, wk, bh, st[3]))
    end)
    store!(du, o, N3, 0, ∂ₜh)
    msrc = lean_source_inl(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ,
                           (∂ₜh, ∂h[1], ∂h[2], ∂h[3]), zero(SVector{4,T}),
                           zero(SMatrix{4,4,T}), γ0, γ2)
    A = (α * sqrtγ) * γu
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]
    divA = SVector{3,T}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)
    for v in 1:NC
        bh = var + (v - 1) * sv
        ht = h_terms(Val(:sparse), wk, bh, st, inv_h, inv_h²)
        pt = p_terms(Val(:sparse), wk, bh + NC * sv, st, inv_h)
        @inbounds du[o + (NC + v - 1) * N3] = pi_combine(ht, pt, εh, β, divβ, divA, A) +
                                               msrc[v]
    end
    return nothing
end

@kernel function ka_fl2!(du, work, spacings, γ0, γ2, ε, ::Val{N}) where {N}
    I = @index(Global, NTuple)
    body_fl2!(du, work, spacings, γ0, γ2, ε, I[1], I[2], I[3], I[4], Val(N))
end

@inline function body_1b2!(du, dh, work, γ0, γ2, i, j, k, b, ::Val{N}) where {N}
    wk = Base.Experimental.Const(work)
    dk = Base.Experimental.Const(dh)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    o = du_base(Val(N), i, j, k, b)
    N3 = N^3
    od = i + N * (j - 1) + N * N * (k - 1) + 3NC * N3 * (b - 1)
    hv = SVector{NC,T}(ntuple(v -> (@inbounds wk[var + (v - 1) * sv]), Val(NC)))
    ∂h = ntuple(Val(3)) do d
        SVector{NC,T}(ntuple(v -> (@inbounds dk[od + ((d - 1) * NC + v - 1) * N3]), Val(NC)))
    end
    ∂ₜh = SVector{NC,T}(ntuple(v -> (@inbounds du[o + (v - 1) * N3]), Val(NC)))
    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    msrc = lean_source_inl(TGH._pack10(g4), TGH._pack10(gu4), α, sqrtγ,
                           (∂ₜh, ∂h[1], ∂h[2], ∂h[3]), zero(SVector{4,T}),
                           zero(SMatrix{4,4,T}), γ0, γ2)
    store!(du, o, N3, NC, msrc)
    return nothing
end
@kernel function ka_1b2!(du, dh, work, γ0, γ2, ::Val{N}) where {N}
    I = @index(Global, NTuple)
    body_1b2!(du, dh, work, γ0, γ2, I[1], I[2], I[3], I[4], Val(N))
end

function round9(N, roots)
    case, forest, U, u, p = setup(N, roots)
    nb = nblocks(U)
    npts = nb * N^3
    @printf("# round9 N=%d roots=%d blocks=%d points=%.3e  %s\n", N, roots, nb, npts,
            CUDA.name(CUDA.device()))
    du = similar(u)
    ref = similar(u)
    gh_rhs!(ref, u, p, zero(T))
    CUDA.synchronize()
    work = p.U.work
    sp = p.spacings
    γ0, γ2, ε = one(T), zero(T), T(1 // 2)
    coef = CUDA.zeros(T, NCOEF * npts)
    dh = CUDA.zeros(T, 3NC * npts)
    nd = (N, N, N, nb)
    wg = (min(N, 32), 4, 1, 1)
    row(n, t) = @printf("  %-52s %8.3f ms  %7.3f ns/pt\n", n, 1e3t, 1e9t / npts)
    for inl in (false, true)
        be = CUDABackend(; always_inline=inl)
        k = ka_fl2!(be, wg)
        launch() = (k(du, work, sp, γ0, γ2, ε, Val(N); ndrange=nd); CUDA.synchronize())
        t = timeit(launch)
        fill!(du, 0)
        launch()
        e = compare(du, ref, N, nb)
        row(@sprintf("KA fused, closure-free lean source, inl=%-5s err Π %.1e", inl, e[2]), t)
        k1a = ka_1a!(be, wg)
        k1b = ka_1b2!(be, wg)
        l1a() = (k1a(du, dh, coef, work, sp, ε, Val(N); ndrange=nd); CUDA.synchronize())
        l1b() = (k1b(du, dh, work, γ0, γ2, Val(N); ndrange=nd); CUDA.synchronize())
        t1a = timeit(l1a)
        row("KA K1a, inl=$inl", t1a)
        row("KA K1b closure-free lean, inl=$inl", timeit(() -> (l1a(); l1b())) - t1a)
    end
    return nothing
end

function round7(N, roots)
    case, forest, U, u, p = setup(N, roots)
    nb = nblocks(U)
    npts = nb * N^3
    @printf("# round7 N=%d roots=%d blocks=%d points=%.3e  %s\n", N, roots, nb, npts,
            CUDA.name(CUDA.device()))
    du = similar(u)
    ref = similar(u)
    gh_rhs!(ref, u, p, zero(T))
    CUDA.synchronize()
    work = p.U.work
    sp = p.spacings
    γ0, γ2, ε = one(T), zero(T), T(1 // 2)
    coef = CUDA.zeros(T, NCOEF * npts)
    dh = CUDA.zeros(T, 3NC * npts)
    nd = (N, N, N, nb)
    row(n, t) = @printf("  %-44s %8.3f ms  %7.3f ns/pt\n", n, 1e3t, 1e9t / npts)
    du5 = statearray(du, p.U)
    for inl in (false, true), wg in (nothing, (32, 4, 1, 1))
        kp = ka_pkgshape_lean!(CUDABackend(; always_inline=inl))
        tp = timeit(() -> (wg === nothing ? kp(du5, work, sp, γ0, γ2, ε, Val((GG, GG, GG));
                                               ndrange=nd) :
                           kp(du5, work, sp, γ0, γ2, ε, Val((GG, GG, GG)); ndrange=nd,
                              workgroupsize=wg); CUDA.synchronize()))
        e = compare(du, ref, N, nb)
        row(@sprintf("KA package shape, lean source, %s %s  err h %.1e Π %.1e",
                     inl ? "inl" : "   ", wg === nothing ? "dyn wg" : "wg $(wg[1:3])", e...), tp)
    end
    for inl in (false, true), wg in (nothing, (32, 4, 1, 1), (32, 2, 2, 1))
        wg !== nothing && wg[1] > N && continue
        be = CUDABackend(; always_inline=inl)
        k1a = wg === nothing ? ka_1a!(be) : ka_1a!(be, wg)
        k1b = wg === nothing ? ka_1b!(be) : ka_1b!(be, wg)
        k2 = wg === nothing ? ka_2!(be) : ka_2!(be, wg)
        l1a() = (k1a(du, dh, coef, work, sp, ε, Val(N); ndrange=nd); CUDA.synchronize())
        l1b() = (k1b(du, dh, work, γ0, γ2, Val(N); ndrange=nd); CUDA.synchronize())
        l2() = (k2(du, coef, work, sp, ε, Val(N); ndrange=nd); CUDA.synchronize())
        tag = (inl ? "inl " : "    ") * (wg === nothing ? "dynamic wg" : "static wg $(wg[1:3])")
        t1a = timeit(l1a)
        t1b = timeit(() -> (l1a(); l1b())) - t1a
        t2 = timeit(() -> (l1a(); l1b(); l2())) - t1a - t1b
        l1a(); l1b(); l2()
        e = compare(du, ref, N, nb)
        row("KA K1a  " * tag, t1a)
        row("KA K1b  " * tag, t1b)
        row("KA K2   " * tag, t2)
        row(@sprintf("KA total %s  err h %.1e Π %.1e", tag, e...), t1a + t1b + t2)
    end
    return nothing
end

function round5(N, roots)
    case, forest, U, u, p = setup(N, roots)
    nb = nblocks(U)
    npts = nb * N^3
    @printf("# round5 N=%d roots=%d blocks=%d points=%.3e  %s  inline=%s\n", N, roots, nb,
            npts, CUDA.name(CUDA.device()), INLINE)
    du = similar(u)
    ref = similar(u)
    gh_rhs!(ref, u, p, zero(T))
    CUDA.synchronize()
    work = p.U.work
    sp = p.spacings
    γ0, γ2, ε = one(T), zero(T), T(1 // 2)
    row(n, t) = @printf("  %-34s %9.3f ms  %7.3f ns/pt\n", n, 1e3t, 1e9t / npts)
    row("gh_rhs! (package)", timeit(() -> gh_rhs!(du, u, p, zero(T))))
    row("scatter! (TreeAMR)", timeit(() -> scatter!(p.U, u)))
    row("fill_ghosts! (TreeAMR)", timeit(() -> fill_ghosts!(p.U, p.schedule)))
    args = ka_args(p, du)
    row("kernel, KA", timeit(() -> map_blocks!(TGH.gh_rhs_kernel!, p.U, args...)))
    kinl = TGH.gh_rhs_kernel!(CUDABackend(; always_inline=true))
    row("kernel, KA always_inline", timeit(() -> (kinl(args...; ndrange=(N, N, N, nb));
                                                  CUDA.synchronize())))
    dt = gh_dt(p, u; cfl=T(1 // 4))
    integ = gh_integrator(p, copy(u), (zero(T), T(10^6) * dt); dt=dt)
    row("RK4 step (IMEXRungeKutta, package RHS)", timeit(() -> IRK.step!(integ); n=3))
    y = copy(u)
    acc = copy(u)
    row("broadcast y + c k (one stage input)", timeit(() -> (du .= y .+ T(0.1) .* ref)))
    row("broadcast acc += c k", timeit(() -> (acc .+= T(0.1) .* ref)))
    wg = N >= 32 ? (32, 4, 1) : (16, 8, 1)
    for (name, f, a) in (("raw scatter (static strides)", k_scatter!, (work, u, Val(N))),
                         ("raw stage + scatter fused", k_stage!,
                          (work, acc, y, ref, T(0.1), T(0.2), Val(N))),
                         ("copy floor", k_copy!, (du, work, Val(N))))
        k = @cuda launch=false f(a...)
        row(name, timeit(() -> k(a...; threads=wg, blocks=launch_dims(N, nb, wg))))
    end
    # restore the working array (the stage kernel wrote into its interior)
    scatter!(p.U, u)
    fill_ghosts!(p.U, p.schedule)
    coef = CUDA.zeros(T, NCOEF * npts)
    dh = CUDA.zeros(T, 3NC * npts)
    wgs = (wg,)
    run_raw("R0 fused", k_fused!,
            (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(true), Val(false)), N, nb, wgs;
            ref=ref, du=du)
    k1 = run_raw("K1 (package source)", k_split1f!,
                 (du, coef, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(:sparse)),
                 N, nb, wgs)
    pre1() = k1(du, coef, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(:sparse);
                threads=wg, blocks=launch_dims(N, nb, wg))
    run_raw("K2 after K1", k_split2f!, (du, coef, work, sp, ε, Val(N), Val(1), Val(:sparse)),
            N, nb, wgs; pre=pre1, ref=ref, du=du)
    ka = run_raw("K1a", k_split1a!, (du, dh, coef, work, sp, ε, Val(N), Val(:sparse)), N, nb,
                 wgs)
    pre_a() = ka(du, dh, coef, work, sp, ε, Val(N), Val(:sparse); threads=wg,
                 blocks=launch_dims(N, nb, wg))
    kl = run_raw("K1b lean", k_split1b_lean!,
                 (du, dh, work, nothing, γ0, γ2, Val(N), Val(false)), N, nb, wgs; pre=pre_a)
    pre_l() = (pre_a(); kl(du, dh, work, nothing, γ0, γ2, Val(N), Val(false); threads=wg,
                           blocks=launch_dims(N, nb, wg)))
    run_raw("K2 after K1a+K1b lean", k_split2f!,
            (du, coef, work, sp, ε, Val(N), Val(1), Val(:sparse)), N, nb, wgs; pre=pre_l,
            ref=ref, du=du)
    return nothing
end

function round3(N, roots)
    case, forest, U, u, p = setup(N, roots)
    nb = nblocks(U)
    npts = nb * N^3
    @printf("# round3 N=%d roots=%d blocks=%d points=%.3e  %s  inline=%s\n", N, roots, nb,
            npts, CUDA.name(CUDA.device()), INLINE)
    du = similar(u)
    ref = similar(u)
    gh_rhs!(ref, u, p, zero(T))
    CUDA.synchronize()
    work = p.U.work
    sp = p.spacings
    γ0, γ2, ε = one(T), zero(T), T(1 // 2)
    wgs = filter(wg -> wg[1] <= N, ((32, 8, 1), (32, 4, 1), (32, 2, 2), (16, 8, 1),
                                    (16, 4, 2), (64, 2, 1), (64, 4, 1)))
    which = split(opt("which", "fusedf,k1f,k2f,k2s"), ",")
    coef = CUDA.zeros(T, NCOEF * npts)
    if "fusedf" in which
        for fl in (:dense, :sparse)
            run_raw("F fused $fl", k_fusedf!,
                    (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(fl)), N, nb,
                    wgs; ref=ref, du=du)
        end
    end
    k1 = nothing
    for fl in (:dense, :sparse)
        k = run_raw("K1 $fl", k_split1f!,
                    (du, coef, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(fl)),
                    N, nb, "k1f" in which ? wgs : ((32, 4, 1),))
        fl === :sparse && (k1 = k)
    end
    best1 = (32, 4, 1)
    launch1() = k1(du, coef, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false),
                   Val(:sparse); threads=best1, blocks=launch_dims(N, nb, best1))
    if "k2f" in which
        for fl in (:dense, :sparse), nc in (1, NC)
            run_raw("K2 $fl ncomp=$nc", k_split2f!,
                    (du, coef, work, sp, ε, Val(N), Val(nc), Val(fl)), N, nb, wgs;
                    ncomp=nc, pre=launch1, ref=ref, du=du)
        end
    end
    if "k2s" in which
        for tile in ((32, 4, 4), (32, 8, 2), (32, 4, 2), (32, 8, 1), (16, 8, 4), (16, 8, 2),
                     (16, 16, 2), (64, 4, 2), (64, 2, 2))
            tile[1] <= N || continue
            all(d -> N % tile[d] == 0, 1:3) || continue
            for fl in (:dense, :sparse)
                run_raw("K2 smem $tile $fl", k_split2s!,
                        (du, coef, work, sp, ε, Val(N), Val(tile[1]), Val(tile[2]),
                         Val(tile[3]), Val(fl)), N, nb, (tile,); pre=launch1, ref=ref,
                        du=du)
            end
        end
    end
    return nothing
end

# --- floors -------------------------------------------------------------------------

# The state's own traffic: read the 20 values at the point, write them.
function k_copy!(du, work, ::Val{N}) where {N}
    i, j, k, b, _ = thread_point(Val(N), Val(1))
    wk = Base.Experimental.Const(work)
    st, sv, sb = strides_static(Val(N))
    var = work_base(st, sb, i, j, k, b)
    o = du_base(Val(N), i, j, k, b)
    ntuple(Val(2NC)) do v
        @inbounds du[o + (v - 1) * N^3] = wk[var + (v - 1) * sv]
        nothing
    end
    return nothing
end

# --- harness --------------------------------------------------------------------

function kinfo(k)
    m = CUDA.memory(k)
    return @sprintf("regs %3d  local %5d B  maxthreads %4d", CUDA.registers(k), m.local,
                    CUDA.maxthreads(k))
end

function run_raw(name, f, args, N, nb, wgs; ncomp=1, maxregs=nothing, ref=nothing,
                 du=nothing, pre=nothing)
    compile() = maxregs === nothing ? (@cuda launch=false always_inline=INLINE f(args...)) :
                (@cuda launch=false always_inline=INLINE maxregs=maxregs f(args...))
    k = compile()
    if opt("sass", "0") == "1"
        tt = Tuple{map(a -> typeof(CUDA.cudaconvert(a)), args)...}
        txt = sprint() do io
            maxregs === nothing ? CUDA.code_sass(io, f, tt; always_inline=INLINE) :
            CUDA.code_sass(io, f, tt; always_inline=INLINE, maxregs=maxregs)
        end
        fname = OUT * "/sass-" * replace(name, r"[^A-Za-z0-9]+" => "_") *
                (maxregs === nothing ? "" : "_mr$maxregs") * "-N$N.txt"
        open(fh -> write(fh, txt), fname, "w")
    end
    @printf("%-28s %s\n", name * (maxregs === nothing ? "" : " mr$maxregs") *
            (INLINE ? " [inl]" : ""), kinfo(k))
    for wg in wgs
        prod(wg) <= CUDA.maxthreads(k) || continue
        all(d -> N % wg[d] == 0, 1:3) || continue
        blocks = launch_dims(N, nb, wg, ncomp)
        launch() = (pre === nothing || pre(); k(args...; threads=wg, blocks=blocks))
        t = timeit(launch)
        tpre = pre === nothing ? 0.0 : timeit(pre)
        npts = nb * N^3
        errs = ref === nothing ? "" : (launch(); CUDA.synchronize();
                                       e = compare(du, ref, N, nb);
                                       @sprintf("  err h %.1e Π %.1e", e...))
        @printf("    wg %-12s %8.3f ms  %7.3f ns/pt%s\n", string(wg), 1e3 * (t - tpre),
                1e9 * (t - tpre) / npts, errs)
        flush(stdout)
    end
    return k
end

function baseline(N, roots)
    case, forest, U, u, p = setup(N, roots)
    nb = nblocks(U)
    npts = nb * N^3
    @printf("# baseline N=%d roots=%d blocks=%d points=%.3e  %s\n", N, roots, nb, npts,
            CUDA.name(CUDA.device()))
    du = similar(u)
    # SASS of the first call: registers and spills of every kernel it compiles.
    sass = sprint() do io
        CUDA.@device_code_sass io=io gh_rhs!(du, u, p, zero(T))
    end
    open(joinpath(OUT, "sass-pkg-N$N.txt"), "w") do f
        write(f, sass)
    end
    t_rhs = timeit(() -> gh_rhs!(du, u, p, zero(T)))
    t_sc = timeit(() -> scatter!(p.U, u))
    t_fill = timeit(() -> fill_ghosts!(p.U, p.schedule))
    args = ka_args(p, du)
    t_k = timeit(() -> map_blocks!(TGH.gh_rhs_kernel!, p.U, args...))
    for (n, t) in (("gh_rhs!", t_rhs), ("scatter!", t_sc), ("fill_ghosts!", t_fill),
                   ("kernel (KA default wg)", t_k))
        @printf("  %-26s %9.3f ms  %7.3f ns/pt\n", n, 1e3t, 1e9t / npts)
    end
    ref = similar(u)
    gh_rhs!(ref, u, p, zero(T))
    CUDA.synchronize()
    for inl in (false, true)
    kka = TGH.gh_rhs_kernel!(CUDABackend(; always_inline=inl))
    if inl
        sass2 = sprint() do io
            CUDA.@device_code_sass io=io (kka(args...; ndrange=(N, N, N, nb));
                                          CUDA.synchronize())
        end
        open(joinpath(OUT, "sass-pkg-inl-N$N.txt"), "w") do f
            write(f, sass2)
        end
        fill!(du, 0)
        kka(args...; ndrange=(N, N, N, nb))
        CUDA.synchronize()
        @printf("  KA always_inline: err h %.1e Π %.1e\n", compare(du, ref, N, nb)...)
        t = timeit(() -> (kka(args...; ndrange=(N, N, N, nb)); CUDA.synchronize()))
        @printf("  KA inl default wg   %9.3f ms  %7.3f ns/pt\n", 1e3t, 1e9t / npts)
    end
    for wg in ((32, 8, 1, 1), (32, 4, 1, 1), (32, 2, 2, 1), (16, 8, 1, 1), (16, 4, 2, 1),
               (8, 8, 2, 1), (32, 1, 1, 1), (64, 1, 1, 1), (128, 1, 1, 1), (16, 16, 1, 1))
        all(d -> N % wg[d] == 0 || wg[d] > N, 1:3) || continue
        wg[1] <= N || continue
        t = try
            timeit(() -> (kka(args...; ndrange=(N, N, N, nb), workgroupsize=wg);
                          KernelAbstractions.synchronize(CUDABackend())))
        catch e
            @printf("  KA wg %-14s failed: %s\n", string(wg), first(sprint(showerror, e), 80))
            continue
        end
        @printf("  KA%s wg %-14s %9.3f ms  %7.3f ns/pt\n", inl ? " inl" : "", string(wg),
                1e3t, 1e9t / npts)
    end
    end
    # The device-side profile of one evaluation: launches and their times.
    if opt("profile", "1") == "1"
        prof = sprint() do io
            show(io, CUDA.@profile trace=true gh_rhs!(du, u, p, zero(T)))
        end
        open(joinpath(OUT, "profile-N$N.txt"), "w") do f
            write(f, prof)
        end
        prof2 = sprint() do io
            show(io, CUDA.@profile gh_rhs!(du, u, p, zero(T)))
        end
        println(prof2)
    end
    return nothing
end

function variants(N, roots)
    case, forest, U, u, p = setup(N, roots)
    nb = nblocks(U)
    npts = nb * N^3
    @printf("# variants N=%d roots=%d blocks=%d points=%.3e  %s\n", N, roots, nb, npts,
            CUDA.name(CUDA.device()))
    du = similar(u)
    ref = similar(u)
    gh_rhs!(ref, u, p, zero(T))          # fills the ghosts too
    CUDA.synchronize()
    work = p.U.work
    sp = p.spacings
    γ0, γ2, ε = one(T), zero(T), T(1 // 2)
    wgs = ((32, 8, 1), (32, 4, 1), (32, 2, 2), (32, 2, 1), (32, 1, 2), (16, 8, 1),
           (16, 4, 2), (16, 4, 1), (8, 8, 2), (8, 4, 4), (64, 2, 1), (64, 1, 1))
    wgs = filter(wg -> wg[1] <= N, wgs)
    which = split(opt("which", "copy,fused,fused_dyn,fused_mr,ablate,reordered,split"), ",")
    if "copy" in which
        run_raw("copy (20 in, 20 out)", k_copy!, (du, work, Val(N)), N, nb, ((32, 4, 1), (32, 8, 1), (16, 8, 1)))
    end
    if "fused" in which
        run_raw("R0 fused static", k_fused!,
                (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(true), Val(false)), N, nb,
                wgs; ref=ref, du=du)
    end
    if "fused_dyn" in which
        run_raw("R0 fused dynamic strides", k_fused!,
                (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(false)), N, nb,
                ((32, 4, 1), (32, 8, 1), (16, 8, 1)); ref=ref, du=du)
    end
    if "fused_mr" in which
        for mr in (128, 168)
            run_raw("R0 fused static", k_fused!,
                    (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(true), Val(false)), N,
                    nb, ((32, 4, 1), (32, 8, 1), (32, 2, 2), (16, 8, 1)); maxregs=mr,
                    ref=ref, du=du)
        end
    end
    if "alg" in which
        src = KerrSchildSource(T; M=1)
        run_raw("R0 fused + algebraic source", k_fused!,
                (du, work, sp, src, γ0, γ2, ε, Val(N), Val(true), Val(:algebraic)), N, nb,
                ((32, 4, 1), (32, 8, 1), (16, 8, 1)))
    end
    if "ablate" in which
        run_raw("A1 no source", k_ablate!,
                (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(false), Val(true)),
                N, nb, ((32, 4, 1), (32, 8, 1), (16, 8, 1)))
        run_raw("A2 no Π stencils", k_ablate!,
                (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(true), Val(false)),
                N, nb, ((32, 4, 1), (32, 8, 1), (16, 8, 1)))
        run_raw("A3 neither", k_ablate!,
                (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false), Val(false), Val(false)),
                N, nb, ((32, 4, 1), (32, 8, 1), (16, 8, 1)))
    end
    if "reordered" in which
        run_raw("R1 reordered", k_reordered!,
                (du, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false)), N, nb, wgs;
                ref=ref, du=du)
    end
    if "split" in which
        coef = CUDA.zeros(T, NCOEF * npts)
        k1 = run_raw("R2 K1 pointwise", k_split1!,
                     (du, coef, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false)), N, nb, wgs)
        # K2 timed alone (K1 run once before, so that du holds the source).
        best1 = (32, 4, 1)
        launch1() = k1(du, coef, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false);
                       threads=best1, blocks=launch_dims(N, nb, best1))
        launch1()
        CUDA.synchronize()
        run_raw("R2 K2 loop over v", k_split2!, (du, coef, work, sp, ε, Val(N), Val(1)),
                N, nb, wgs; pre=launch1, ref=ref, du=du)
        run_raw("R2 K2 one v per thread", k_split2!,
                (du, coef, work, sp, ε, Val(N), Val(NC)), N, nb, wgs; ncomp=NC,
                pre=launch1, ref=ref, du=du)
        for mr in (64, 96)
            run_raw("R2 K1 pointwise", k_split1!,
                    (du, coef, work, sp, nothing, γ0, γ2, ε, Val(N), Val(false)), N, nb,
                    ((32, 4, 1), (32, 8, 1), (32, 2, 2), (16, 8, 1)); maxregs=mr)
        end
        if "alg" in which
            src = KerrSchildSource(T; M=1)
            run_raw("R2 K1 + algebraic source", k_split1!,
                    (du, coef, work, sp, src, γ0, γ2, ε, Val(N), Val(:algebraic)), N, nb,
                    ((32, 4, 1), (32, 8, 1), (16, 8, 1)))
        end
    end
    return nothing
end

mkpath(OUT)
N = parse(Int, opt("N", "32"))
roots = parse(Int, opt("roots", "8"))
mode = opt("mode", "variants")
CUDA.versioninfo()
mode == "baseline" ? baseline(N, roots) : mode == "round3" ? round3(N, roots) :
    mode == "round4" ? round4(N, roots) : mode == "round5" ? round5(N, roots) :
    mode == "round6" ? round6(N, roots) : mode == "round7" ? round7(N, roots) :
    mode == "round8" ? round8(N, roots) : mode == "round9" ? round9(N, roots) :
    variants(N, roots)
