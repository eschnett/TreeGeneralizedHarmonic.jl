# One right-hand-side evaluation: the fused kernel, the problem it reads,
# and the time step.
#
# `CODE.md`, "One right-hand-side evaluation", fixes TreeAMR's contract —
# three steps and nothing between them,
#
#     scatter!(U, u)                                          # state → working array
#     fill_ghosts!(U, schedule; boundary = dirichlet(case, t))
#     map_blocks!(gh_rhs_kernel!, U, statearray(du, U), …)
#
# — and, more importantly, the *internal order* of the third step, which
# is a design commitment rather than an implementation detail: it is what
# decides whether the kernel fits in a GPU thread's registers. Written
# naively — load the block, form all of `∂_i h` (30), `∂_i Π` (30),
# `∂_i∂_j h` (60) and the dissipation, then run the algebra — a thread
# holds about 140 `Float64` values before the algebra starts, which is
# already over the 255 32-bit registers it has, and it spills. So the
# kernel is written in **streaming order**:
#
#   1. load `h` and `Π` at the point, form the 30 first derivatives
#      `∂_i h` (which are kept) and, from them and `h`, the coefficient
#      set `g^{ab}, α, β^i, √γ, γ^{ij}` and its two contracted
#      derivatives `∂_iβ^i`, `∂_i(α√γγ^{ij})` — about 30 more values;
#   2. loop over the 10 components: form `∂_iΠ_ab` (3 stencils),
#      `∂_i∂_j h_ab` (3 compact and 3 tensor products) and the
#      dissipation of `h_ab` and `Π_ab` **on the fly**, contract them
#      immediately into two accumulators, and drop them;
#   3. add the source `S0 + Z` and write 20 values of `du`.
#
# Nothing here ever builds an `SVector` of all the derivatives, and
# nothing that can be rebuilt from `h` and `∂_i h` is stored
# (`CLAUDE.md`, "The RHS kernel is written in streaming order"). The
# per-component step is a loop with a small live set because the
# principal part is the *same scalar wave operator for every component*;
# all the coupling is in the coefficients and in the source.
#
# The kernel is **block-local**: it reads its own block's stored points
# and nothing else, so it runs on every backend unchanged. Per-block
# spacings and origins travel to the backend once per chunk, as TreeWave's
# spacings do.
#
# Two things this file does *not* do, and must not start doing: it never
# mutates `u` (the right-hand side is a pure function of `(u, t)`,
# TreeAMR's contract), and it never consults the tree — the schedule is
# built once per chunk and replayed.

"""
    convergence_rate(hs, errs)

Least-squares slope of `log(err)` against `log(h)` — the observed order of
a scheme over a sequence of resolutions.

TreeWave's, character for character, because the four packages' tables are
read side by side and a different fit would make them incomparable.
"""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

# The variable slots of the `diag` field set: the characteristic speed the
# time step is taken from, the two constraint monitors of step 4, and the
# indicator that says which points a masked norm counts. Step 5 adds the
# masked error and the interior residual, step 6 the refinement indicator.
#
# `DIAG_CGH` and `DIAG_MOM` are the *first* of a contiguous run — four and
# three slots — because `block_mapreduce` reduces a contiguous range of
# variables and nothing else: a device cannot be handed an arbitrary index
# vector cell by cell.
const DIAG_SPEED = 1          # λ = α√(tr γ^{ij}) + |β|
const DIAG_CGH = 2            # C_a = Γ_a + H_a, a = t, x, y, z   (2:5)
const DIAG_HAM = 6            # the ADM Hamiltonian constraint ℋ
const DIAG_MOM = 7            # the ADM momentum constraint ℳ_i   (7:9)
const DIAG_MASK = 10          # 1 where the point is evolved, 0 inside r_1
const NDIAG = 10

# The position of an **owned** point, formed exactly as TreeAMR forms it —
# the same origin, the same spacing, the same expression in the same order
# — so that a mask, an interior profile or a Dirichlet value computed here
# lands on the value `coordinates(fs, b, idx)` gives and not merely near
# it. TreeAMR's own `coordinates_kernel!` writes
# `origin[d] + (I[d] - off[d]) * h` with `off` a whole cell along a
# vertex-like dimension, and every field set in this package is
# vertex-centered (`CODE.md`, "Field sets and layout"), which
# [`GHProblem`](@ref) checks so that this line may assume it.
@inline function point_position(origins, spacings, b::Int, I)
    h = spacings[b]
    origin = origins[b]
    off = oftype(h, 1 // 1)
    return ntuple(d -> origin[d] + (I[d] - off) * h, Val(3))
end

# --- the kernel-side stencil contractions -----------------------------------
#
# `stencils.jl`'s `apply_stencil` and `apply_mixed_stencil` are the
# host-side reference *definitions*; these are what the kernel evaluates,
# and the two have to agree. Both sum from the lowest offset to the
# highest, and the mixed one runs its inner sum along the second axis —
# which is a property of the operator and not of the implementation, since
# the other order differs in the last place (`CODE.md`, "Finite-difference
# stencils"). Nothing is materialised: each contraction is formed and
# consumed where it is written.
#
# They address the working array by a **linear index**: a base index for
# the point and one stride per axis, rather than a `(i, j, k, v, b)` tuple
# per load. The working array is dense and column-major — TreeAMR
# allocates it — so the two are the same array element, and the tests
# compare the kernel against `apply_stencil`'s cartesian spelling. The
# reason is measured (step 3): at `q = 4` the stencil half of the kernel
# costs **1376 ns** per point with the cartesian index and **573 ns** with
# the linear one, for bit-identical output. Five-dimensional index
# arithmetic at every one of the ~1600 loads a point takes is not a cost
# the compiler removes, and it is not arithmetic this scheme is about.

# A left fold over a tuple, in order. `sum` would do, but its association
# is not part of its interface, and the order of a stencil's summation is
# part of this operator's.
@inline _fold(t::Tuple) = _fold(t[1], Base.tail(t))
@inline _fold(s, t::Tuple) = _fold(s + t[1], Base.tail(t))
@inline _fold(s, ::Tuple{}) = s

# The working array's strides: `(per axis), per variable, per block`. One
# call per point, from `size` alone, so it holds for any dense array on
# any backend.
@inline function work_strides(work)
    n1, n2, n3 = size(work, 1), size(work, 2), size(work, 3)
    sv = n1 * n2 * n3
    return (1, n1, n1 * n2), sv, sv * size(work, 4)
end

# `∑_k w[k] u[base + (k − 1 − r)·stride]`: the contraction every
# one-dimensional operator here is, whatever the weights mean.
# `ntuple(…, Val(n))` unrolls it, so the offsets and the weights are
# compile-time constants and the loop leaves no index arithmetic behind.
@inline function axis_stencil(w::SVector{n,T}, work, base::Int,
                              stride::Int) where {n,T}
    r = (n - 1) ÷ 2
    return _fold(ntuple(Val(n)) do k
        @inbounds w[k] * work[base + (k - 1 - r) * stride]
    end)
end

# The mixed derivative: the tensor product of two first-derivative vectors,
# outer sum along the first axis, inner along the second. It reads the edge
# ghosts TreeAMR fills unconditionally, which is why `∂_i∂_j` needs no
# wider halo than `∂_i∂_i`.
@inline function mixed_stencil(w::SVector{n,T}, work, base::Int, s1::Int,
                               s2::Int) where {n,T}
    r = (n - 1) ÷ 2
    return _fold(ntuple(Val(n)) do a
        w[a] * _fold(ntuple(Val(n)) do e
            @inbounds w[e] * work[base + (a - 1 - r) * s1 + (e - 1 - r) * s2]
        end)
    end)
end

"""
    gh_rhs_kernel!(du, work, Hwork, spacings, γ0, γ2, ε_KO,
                   ::Val{G}, ::Val{q}, ::Val{HASH}, ::Val{DISS})

The fused right-hand side at one owned point, in `CODE.md`'s streaming
order. `du` is in **state layout** (no ghosts, so the global index is used
as it comes); `work` is the ghosted working array (so the same index plus
`G`). `Hwork` is the gauge source's working array or `nothing`.

The four `Val`s are built once per chunk in [`GHProblem`](@ref) and
resolved when the kernel compiles: the ghost width, the difference order,
whether there is a gauge source, and whether there is dissipation.
Building them per evaluation would recompile or dispatch dynamically at
every RK stage (`CLAUDE.md`).

What it computes, per point:

    ∂ₜh_ab = β^i ∂_i h_ab + (α/√γ) Π_ab                    + Q_d h_ab
    ∂ₜΠ_ab = β^i ∂_i Π_ab + (∂_iβ^i) Π_ab
           + α√γ γ^{ij} ∂_i∂_j h_ab + ∂_i(α√γ γ^{ij}) ∂_j h_ab
           − α√γ (S0_ab + Z_ab)                            + Q_d Π_ab

`(EXPANDED)` of `CODE.md`'s "The equations", plus the Kreiss–Oliger term.
The source `S0 + Z` is [`gh_node_source`](@ref) and the coefficient
derivatives are [`metric_derivatives`](@ref) — the same functions
[`gh_node_rhs_expanded`](@ref) calls, which is what makes that function
the reference this kernel is checked against on analytic data
(`test/evolution_tests.jl`). The two are not bit-identical and are not
expected to be: one body reached from two call sites is contracted into
fused multiply-adds differently (`CODE.md`, "Measured results").

**The `∂_t g` the source is given is the accumulated `∂ₜh`, dissipation
included** — the two accumulators per component that the streaming order
budgets, and the honest answer besides: it is the time derivative of the
numerical solution, which is what the reduced source's `−Γ^ν ∂_ν g_ab`
means. The difference is `O(h^{q+1})`, the dissipation's own order
**(recorded in step 3**, where `CODE.md` had said only "from `h`, `∂_i h`,
`∂_t h` and the coefficients"**)**.
"""
@kernel function gh_rhs_kernel!(du, @Const(work), Hwork, @Const(spacings),
                                γ0, γ2, ε_KO, ::Val{G}, ::Val{q},
                                ::Val{HASH}, ::Val{DISS}) where {G,q,HASH,DISS}
    I = @index(Global, NTuple)                    # (i1, i2, i3, block)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))             # state-layout index
    T = eltype(du)

    inv_h = inv(spacings[b])
    inv_h² = inv_h * inv_h
    w1 = derivative_weights(T, Val(q), Val(1))
    w2 = derivative_weights(T, Val(q), Val(2))
    wD = dissipation_weights(T, dissipation_rank(Val(q)))
    εh = ε_KO * inv_h

    # The point's linear index in the working array — the owned index plus
    # the ghost width along each axis — and the strides the stencils step
    # by. `var` is the first variable's base; variable `v` is `var + (v−1)·sv`.
    st, sv, sb = work_strides(work)
    var = 1 + (b - 1) * sb +
          (I[1] + G[1] - 1) * st[1] + (I[2] + G[2] - 1) * st[2] +
          (I[3] + G[3] - 1) * st[3]

    # (1) the state at the point, the 30 first derivatives of `h`, and the
    #     coefficients built from them once.
    hv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (v - 1) * sv]),
                              Val(NC)))
    Πv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (NC + v - 1) * sv]),
                              Val(NC)))
    ∂h = ntuple(Val(3)) do d
        inv_h * SVector{NC,T}(ntuple(Val(NC)) do v
            axis_stencil(w1, work, var + (v - 1) * sv, st[d])
        end)
    end

    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(hv))
    a_div = α / sqrtγ
    A = (α * sqrtγ) * γu                          # A^{jk} = α√γ γ^{jk}
    _, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    divβ = dβ[1, 1] + dβ[2, 2] + dβ[3, 3]                     # ∂_i β^i
    divA = SVector{3,T}(dA[1, 1, j] + dA[2, 2, j] + dA[3, 3, j] for j in 1:3)

    # (2) one component at a time: nine stencils formed, contracted and
    #     dropped, leaving two accumulators.
    acc = ntuple(Val(NC)) do v
        ∂h1 = ∂h[1][v]
        ∂h2 = ∂h[2][v]
        ∂h3 = ∂h[3][v]
        Π_v = Πv[v]
        bh = var + (v - 1) * sv                   # this component of `h`
        bΠ = bh + NC * sv                         # and of `Π`

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

        if DISS
            ∂ₜh_v += εh * (axis_stencil(wD, work, bh, st[1]) +
                           axis_stencil(wD, work, bh, st[2]) +
                           axis_stencil(wD, work, bh, st[3]))
            ∂ₜΠ_v += εh * (axis_stencil(wD, work, bΠ, st[1]) +
                           axis_stencil(wD, work, bΠ, st[2]) +
                           axis_stencil(wD, work, bΠ, st[3]))
        end
        # Not `return`: KernelAbstractions refuses a `return` statement
        # anywhere in a kernel body, closures included.
        (∂ₜh_v, ∂ₜΠ_v)
    end
    ∂ₜh = SVector{NC,T}(ntuple(v -> acc[v][1], Val(NC)))
    ∂ₜΠ = SVector{NC,T}(ntuple(v -> acc[v][2], Val(NC)))

    # (3) the source, from the state, its gradients and the coefficients —
    #     the gauge source is read at the owned point, `Hsrc` having no
    #     ghosts to read.
    Hl, dHl = gauge_at(T, Hwork, inner, b, Val(HASH))
    msrc = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)

    ntuple(Val(NC)) do v
        du[inner..., v, b] = ∂ₜh[v]
        du[inner..., NC + v, b] = ∂ₜΠ[v] + msrc[v]
        nothing
    end
end

"""
    gh_speed_kernel!(speed, work, ::Val{G})

GHSO2's conservative bound on the coordinate characteristic speed at one
owned point, `λ = α √(tr γ^{ij}) + |β|`, written into the `diag` field
set's speed slot.

It reads the point and nothing else — no stencil, no ghosts — so it is the
cheapest kernel in the package and can be run at every chunk boundary
without thinking about it. `block_mapreduce(max)` over its output is
[`max_speed`](@ref); see `CODE.md`, "The time step".
"""
@kernel function gh_speed_kernel!(speed, @Const(work), ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    c = ntuple(d -> I[d] + G[d], Val(3))
    T = eltype(speed)

    hv = SVector{NC,T}(ntuple(v -> work[c..., v, b], Val(NC)))
    _, _, α, β, γu, _ = metric_quantities(_sym4(hv))
    speed[inner..., DIAG_SPEED, b] =
        α * sqrt(γu[1, 1] + γu[2, 2] + γu[3, 3]) +
        sqrt(β[1] * β[1] + β[2] * β[2] + β[3] * β[3])
end

"""
    GHProblem(U::FieldSet, schedule, case::GHCase; q, t = 0)

Everything one right-hand-side evaluation needs, built once per chunk: the
state's field set and its ghost schedule, the gauge source sampled into
its own field set (or `nothing` where the background is harmonic), the
`diag` field set the analysis kernels write into, the per-block geometry
on whatever backend the state lives on, the case, and the four `Val`s the
kernel specialises on.

It is the `p` of SciML's `f!(du, u, p, t)`, and the application writes
`f!` itself — [`gh_rhs!`](@ref) — rather than reaching for a
`semidiscretize`-style wrapper, which is TreeAMR's contract and TreeWave's
pattern.

`q` is the finite-difference order and has no default: it fixes the ghost
width `G = q/2 + 1`, which the field set was already built with, and the
prolongation order `p = q + 2` that the schedule's `Operators` must carry
(`CODE.md`, "The interface-order rule"). The constructor checks the first
of those; TreeAMR's `check_operators` and step 4's table are what check
the second.

The gauge source is **sampled here**, at `t`, because that is what "after
every regrid" means when every chunk builds a fresh problem — and because
a background that reaches this point is static, so the time it is sampled
at cannot matter. `CODE.md`, "Gauge and constraint damping", and the
refusal in [`GHCase`](@ref) are the two halves of that sentence.
"""
struct GHProblem{T,G,q,HASH,DISS,F,S,H,D,O,V,C}
    U::F
    schedule::S
    Hsrc::H                      # the sampled gauge source, or `nothing`
    diag::D                      # speeds now; constraints and errors later
    # Per block, on the backend the field set lives on. The kernel of this
    # step reads the spacings only; the origins are what the interior
    # profiles, the masks and the damping profile of step 5 turn a cell
    # index into a position with, and they are uploaded here because that
    # is where the per-chunk metadata belongs.
    origins::O
    spacings::V
    case::C
    hasdirichlet::Bool
    valG::Val{G}
    valq::Val{q}
    valH::Val{HASH}
    valdiss::Val{DISS}
end

function GHProblem(U::FieldSet{T,3}, schedule, case::GHCase{T}; q::Integer,
                   t=zero(T)) where {T}
    q ≥ 2 && iseven(q) || throw(ArgumentError(
        "the finite-difference order must be even and at least 2, so that " *
        "the centered stencils have an integer half-width q/2 and CODE.md's " *
        "ghost width G = q/2 + 1 covers them, but q=$q"))
    U.nvars == 2NC || throw(ArgumentError(
        "the evolved state is h (1:10) and Π (11:20), $(2NC) variables " *
        "(CODE.md, \"Field sets and layout\"), but this field set has " *
        "$(U.nvars)"))
    all(g -> g == q ÷ 2 + 1, U.G) || throw(ArgumentError(
        "CODE.md fixes the ghost width at G = q/2 + 1 = $(q ÷ 2 + 1) for " *
        "q=$q — one more than the derivatives need, because the " *
        "Kreiss–Oliger operator of order q + 2 reaches one point further — " *
        "but this field set has G = $(U.G). A narrower halo makes the " *
        "dissipation read a ghost that was never filled; a wider one is " *
        "memory the scheme does not use."))
    all(c -> c === :vertex, U.centering) || throw(ArgumentError(
        "CODE.md fixes this package's field sets as vertex-centered — a " *
        "finite-difference scheme wants restriction along a stagger, which " *
        "is injection and exact for any data — but this field set is " *
        "$(U.centering). The masks and the interior profiles turn an owned " *
        "index into a position assuming it (`point_position`), so a " *
        "staggered set would be evaluated half a cell from where its " *
        "values sit."))
    backend = get_backend(U.work)

    HASH = !isharmonic(case.background)
    Hsrc = if HASH
        fs = FieldSet{T}(U.forest, 2NC; G=0, centering=U.centering,
                         backend=backend)
        sample_gauge_source!(fs, case.background, t)
        fs
    else
        nothing
    end
    diag = FieldSet{T}(U.forest, NDIAG; G=0, centering=U.centering,
                       backend=backend)

    origins = to_backend(backend, block_origins(U.forest, T))
    spacings = to_backend(backend, block_spacings(U.forest, T))
    DISS = !iszero(case.ε_KO)
    hasdirichlet = !all(case.periodic)

    return GHProblem{T,U.G,Int(q),HASH,DISS,typeof(U),typeof(schedule),
                     typeof(Hsrc),typeof(diag),typeof(origins),
                     typeof(spacings),typeof(case)}(
        U, schedule, Hsrc, diag, origins, spacings, case, hasdirichlet,
        Val(U.G), Val(Int(q)), Val(HASH), Val(DISS))
end

# The gauge source's working array, or `nothing` where there is none. The
# kernel never asks whether it has one — its `Val` already said.
gauge_work(::Nothing) = nothing
gauge_work(fs::FieldSet) = fs.work

"""
    gh_rhs!(du, u, p::GHProblem, t)

One right-hand-side evaluation, in SciML's `f!(du, u, p, t)` signature:
scatter the state into the working array, fill the ghosts (with the
Dirichlet hook of this `t`, where the case has a physical boundary), and
launch the fused kernel.

Three steps and nothing between them — TreeAMR's contract, `CODE.md`'s
"One right-hand-side evaluation". **`u` is never mutated**: the working
array is scratch, and the interior treatment of step 5 is a *term* in the
right-hand side rather than a write to the state.

The hook is rebuilt here, at every evaluation, with that evaluation's `t`.
That is not a cost worth avoiding — it is a closure over two `isbits`
values — and the alternative is a boundary that lags the solution by up to
a chunk (`CLAUDE.md`, "Hooks depend on time").
"""
function gh_rhs!(du, u, p::GHProblem, t)
    scatter!(p.U, u)
    if p.hasdirichlet
        fill_ghosts!(p.U, p.schedule; boundary=dirichlet(p.case, t))
    else
        fill_ghosts!(p.U, p.schedule)
    end
    map_blocks!(gh_rhs_kernel!, p.U, statearray(du, p.U), p.U.work,
                gauge_work(p.Hsrc), p.spacings, p.case.γ0, p.case.γ2,
                p.case.ε_KO, p.valG, p.valq, p.valH, p.valdiss)
    return nothing
end

"""
    max_speed(p::GHProblem) -> T

The largest characteristic speed over every owned point, from the state
currently in the working array.

One launch of [`gh_speed_kernel!`](@ref) into `diag`, then
`block_mapreduce(max)` and a fold over the per-block values **in block
order**, so the answer does not depend on the thread count
(`CODE.md`, "Analysis quantities").

It reads the working array, not a state vector: call it after a
`scatter!`, which is what [`gh_dt`](@ref) does.
"""
function max_speed(p::GHProblem{T}) where {T}
    map_blocks!(gh_speed_kernel!, p.U, p.diag.work, p.U.work, p.valG)
    # `zero(T)` as the identity rather than `typemin`: a characteristic
    # speed is `α√(tr γ^{ij}) + |β| ≥ 0`, and `typemin` is not defined for
    # every type this package runs in. A `NaN` still propagates, which is
    # what makes a blown-up state visible here rather than as a `dt` of
    # zero two lines later.
    return maximum(block_mapreduce(identity, max, zero(T), p.diag;
                                   vars=DIAG_SPEED))
end

"""
    gh_dt(p::GHProblem, u; cfl) -> T

The time step, `cfl · minimum_spacing(forest) / λ_max`, with `λ_max` the
largest characteristic speed of the state `u` (`CODE.md`, "The time
step"). `cfl = 1/4` is GHSO2's default and this package's.

There is no subcycling anywhere in TreeAMR, so this one number advances
the whole hierarchy and the finest spacing present is what sets it.

It scatters `u` into the working array on the way — the speed kernel reads
the working array — which is scratch and about to be overwritten by the
first evaluation anyway. `u` itself is untouched.
"""
function gh_dt(p::GHProblem{T}, u; cfl) where {T}
    scatter!(p.U, u)
    λ = max_speed(p)
    isfinite(λ) && λ > 0 || throw(ArgumentError(
        "the maximum characteristic speed α√(tr γ^{ij}) + |β| came out as " *
        "$λ, so there is no CFL-limited time step: the state is either not " *
        "a metric any more (a blown-up run) or identically zero, which no " *
        "initial data of this package produces — h = 0 is Minkowski, whose " *
        "speed is √3."))
    return T(cfl) * minimum_spacing(T, p.U.forest) / λ
end
