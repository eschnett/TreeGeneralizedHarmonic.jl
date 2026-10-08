# The right-hand side on the CPU: where an evaluation goes, and what the kernel's SIMD
# lanes buy (added 2026-10-05; `CODE.md`, "The right-hand side on a CPU"). Scope as
# for the H200 study: `q = 4`, `Float64`, the gauge wave with `ε_KO = 1/2`, `γ0 = 1`,
# no gauge source, on a uniform periodic mesh — or the step-5 hole fixture on the
# step benchmark's mesh (`LAB_CASE=hole`). It runs in the package's own environment:
#
#     julia --project=. -t 64 bench/rhs_cpu_lab.jl
#
#   LAB_MODE   breakdown, simd, asm, parts, profile; a comma list (default
#              breakdown,simd)
#   LAB_CASE   wave | hole (default wave)
#   LAB_N      points per block edge (default 16)
#   LAB_ROOTS  roots per edge (default 8: 512 blocks; the hole's default is 2)
#   LAB_W      SIMD widths for simd and asm (default 1,2,4,8; 1 is the scalar kernel)
#   LAB_REPS   timed repetitions (default 10)
#   LAB_TAG    label printed on every row
#   LAB_ASM    directory for the native code of each kernel (asm mode)
#   LAB_ZW     1: the first-derivative stencils without their zero weights (below)
#   LAB_SAVE, LAB_COMPARE   keep the scalar `du` in a file / compare with one (simd)
#
# breakdown: `gh_rhs!` at the host's default width and its three parts — TreeAMR's
# `scatter!` and `fill_ghosts!`, and the kernel through `map_blocks!` — the copy floor
# (each block's stored slab copied by its owner), and on one thread the scalar
# kernel's body called from a plain loop over the points instead of a KA launch.
#
# simd: the package's kernel at each width in LAB_W (`GHProblem(…; simd_width)`),
# checked against the scalar one on the same filled working array — every point
# written, the difference against the size of the terms — and timed, the kernel alone
# and `gh_rhs!`.
#
# asm: the native code of the kernel at each width, with counts of the instructions
# that say what it is — packed and scalar floating point, loads and stores against the
# stack (spills), calls, and divisions. The counts are x86's AT&T syntax; on aarch64
# they read zero and the files in LAB_ASM are what to read.
#
# parts: the head's algebraic pieces alone, scalar and on four lanes.
#
# profile: Julia's sampling profiler over ten seconds of `fill_ghosts!`, flat, by
# self time.
using TreeAMR, TreeGeneralizedHarmonic, KernelAbstractions, StaticArrays, Printf
using InteractiveUtils, Profile, Random, SIMD
const TGH = TreeGeneralizedHarmonic
include(joinpath(pkgdir(TreeGeneralizedHarmonic), "test", "evolution_cases.jl"))

const T = Float64
const q = 4
const MODES = split(get(ENV, "LAB_MODE", "breakdown,simd"), ",")
const CASE = get(ENV, "LAB_CASE", "wave")
const N = parse(Int, get(ENV, "LAB_N", "16"))
const ROOTS = parse(Int, get(ENV, "LAB_ROOTS", CASE == "hole" ? "2" : "8"))
const WS = parse.(Int, split(get(ENV, "LAB_W", "1,2,4,8"), ","))
const REPS = parse(Int, get(ENV, "LAB_REPS", "10"))
const TAG = get(ENV, "LAB_TAG", "cpu-$CASE-t$(Threads.nthreads())-N$N")

# --- LAB_ZW=1: the stencils without their zero weights ---------------------------------
#
# The first-derivative weights at `q = 4` are `(1/12, −2/3, 0, 2/3, −1/12)`: a fifth of
# an `axis_stencil` and 9 of the 25 products of a `mixed_stencil` multiply by zero, and
# IEEE arithmetic forbids the compiler to drop `0·x`. With the weights in the *type*
# the generated contractions can leave those terms out. For finite data the sum is the
# same number (adding `+0.0` is exact), so the result agrees bit for bit except for the
# sign of a zero. This replaces the package's first-derivative weights for the whole
# process, so it is a run of its own, compared with the plain one (LAB_COMPARE).
struct ZWeights{ws,T} end
@inline Base.getindex(::ZWeights{ws,T}, k::Int) where {ws,T} = T(numerator(ws[k])) / T(denominator(ws[k]))

if get(ENV, "LAB_ZW", "0") == "1"
    @eval TGH begin
        @inline derivative_weights(::Type{T}, ::Val{4}, ::Val{1}) where {T} =
            $ZWeights{$(Tuple(Rational{Int}.(TGH.rational_derivative_weights(4, 1)))),T}()
        @generated function axis_stencil(::$ZWeights{ws,T}, work, base::Int,
                                         stride::Int) where {ws,T}
            n = length(ws); r = (n - 1) ÷ 2
            terms = [:(T($(numerator(ws[k]))) / T($(denominator(ws[k]))) *
                       work[base + $(k - 1 - r) * stride]) for k in 1:n if ws[k] != 0]
            ex = terms[1]
            for t in terms[2:end]
                ex = :($ex + $t)
            end
            return Expr(:block, Expr(:meta, :inline), :(@inbounds $ex))
        end
        @generated function mixed_stencil(::$ZWeights{ws,T}, work, base::Int, s1::Int,
                                          s2::Int) where {ws,T}
            n = length(ws); r = (n - 1) ÷ 2
            w(k) = :(T($(numerator(ws[k]))) / T($(denominator(ws[k]))))
            outer = nothing
            for a in 1:n
                ws[a] == 0 && continue
                inner = nothing
                for e in 1:n
                    ws[e] == 0 && continue
                    t = :($(w(e)) * work[base + $(a - 1 - r) * s1 + $(e - 1 - r) * s2])
                    inner = inner === nothing ? t : :($inner + $t)
                end
                outer = outer === nothing ? :($(w(a)) * $inner) : :($outer + $(w(a)) * $inner)
            end
            return Expr(:block, Expr(:meta, :inline), :(@inbounds $outer))
        end
    end
end

# --- the case ------------------------------------------------------------------------

function the_case()
    CASE == "wave" && return gauge_wave_case(T; ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T))
    CASE == "hole" && return hole_fixture(T; q=q, halfwidth=T(5))
    error("unknown LAB_CASE $CASE")
end

function the_forest(case; N=N, roots=ROOTS)
    CASE == "wave" && return gh_forest(T, case; N=N, roots=roots)
    return hole_forest(T, case; N=N, roots=roots, radii=(T(6), T(3), T(3 // 2)))
end

# A problem at width `W` (`nothing`: the host's) on the field set `U`, with the hole's
# interior at this chunk's rate as `evolve!` builds it.
function problem(U, case, u; simd_width=nothing)
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    p = GHProblem(U, GhostSchedule(U, ops), case; q=q, simd_width=simd_width)
    case.interior === nothing && return p
    dt = gh_dt(p, u; cfl=T(1 // 4))
    return with_interior(p, TGH.chunk_interior(case, dt, nothing,
                                               default_relaxation_rate(case);
                                               default=true, interior=case.interior))
end

function setup(; N=N, roots=ROOTS)
    case = the_case()
    forest = the_forest(case; N=N, roots=roots)
    U = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
    fill_exact!(U, case, zero(T); interior=case.interior)
    u = statevector(U)
    gather!(u, U)
    du = statevector(U)                       # first touched by block owner
    return case, forest, U, problem(U, case, u), u, du
end

simd_of(p) = typeof(p.valsimd).parameters[1]
# The ghost fill as `gh_rhs!` makes it: with the case's Dirichlet hook where the box
# has outer faces (the hole), which evaluates the analytic solution at every
# boundary ghost.
fill_hook!(p) = p.hasdirichlet ?
    fill_ghosts!(p.U, p.schedule; boundary=TGH.dirichlet(p.case, zero(T))) :
    fill_ghosts!(p.U, p.schedule)
kernel!(p, du) = map_blocks!(TGH.gh_rhs_kernel!, p.U, TGH.gh_rhs_kernel_args(p, du, zero(T))...)

# The scalar kernel's body over every owned point, block by block, on one thread:
# `gh_rhs_point!` with the arguments the kernel passes it.
function plain_rhs!(p, du)
    a = TGH.gh_rhs_kernel_args(p, du, zero(T))
    # Split by `Val` lengths, not by ranges: a tuple sliced by a range is not inferred.
    head, vals = ntuple(k -> a[k], Val(16)), ntuple(k -> a[16 + k], Val(5))
    plain_loop!(head, vals)
    return nothing
end
function plain_loop!(head, vals)
    n1, n2, n3, _, nb = size(head[1])
    for b in 1:nb, k in 1:n3, j in 1:n2, i in 1:n1
        TGH.gh_rhs_point!(head..., (i, j, k, b), vals...)
    end
    return nothing
end

# --- timing -----------------------------------------------------------------------

function timeit(f; n=REPS)
    f()
    best = Inf
    for _ in 1:n
        t0 = time_ns(); f(); best = min(best, (time_ns() - t0) / 1e9)
    end
    return best
end

function row(name, t, npts)
    nth = Threads.nthreads()
    @printf("%s\t%-26s\t%9.3f ms\t%8.2f ns/pt\t%9.1f ns/pt·thread\n", TAG, name, 1e3t,
            1e9t / npts, 1e9t * nth / npts)
    flush(stdout)
end

function header(U, p)
    npts = nblocks(U) * N^3
    @printf("# %s  threads %d  blocks %d of %d³  points %.3e  default W %d  host %s  julia %s  cpu %s\n",
            TAG, Threads.nthreads(), nblocks(U), N, npts, simd_of(p), gethostname(),
            VERSION, Sys.CPU_NAME)
    flush(stdout)
    return npts
end

# The copy floor: every block's whole stored slab — owned points and ghosts, all
# twenty variables — copied by its owner thread with `unsafe_copyto!`. The ghost fill
# writes about two thirds of that slab at `16³` (`(23³ − 16³)/23³`) and reads as much,
# so this is roughly what a fill at copy bandwidth would cost.
function slab_copy!(dst::Array{Float64,5}, src::Array{Float64,5})
    len = length(src) ÷ size(src, 5)
    TreeAMR.threaded_foreach(size(src, 5)) do b
        unsafe_copyto!(dst, (b - 1) * len + 1, src, (b - 1) * len + 1, len)
    end
    return nothing
end

function breakdown()
    case, forest, U, p, u, du = setup()
    npts = header(U, p)
    row("gh_rhs!", timeit(() -> gh_rhs!(du, u, p, zero(T))), npts)
    row("scatter!", timeit(() -> scatter!(p.U, u)), npts)
    row("fill_ghosts!", timeit(() -> fill_hook!(p)), npts)
    p.hasdirichlet &&
        row("fill_ghosts!, no hook", timeit(() -> fill_ghosts!(p.U, p.schedule)), npts)
    w2 = similar(p.U.work)
    slab_copy!(w2, p.U.work)                   # first touch by owner
    row("copy floor: stored slabs", timeit(() -> slab_copy!(w2, p.U.work)), npts)
    row("kernel (map_blocks!), W = $(simd_of(p))", timeit(() -> kernel!(p, du)), npts)
    if Threads.nthreads() == 1
        p1 = problem(U, case, u; simd_width=1)
        row("scalar body, plain loop", timeit(() -> plain_rhs!(p1, du)), npts)
        row("scalar kernel (map_blocks!)", timeit(() -> kernel!(p1, du)), npts)
    end
    @printf("%s\tallocated by gh_rhs!: %d bytes\n", TAG, @allocated gh_rhs!(du, u, p, zero(T)))
    return nothing
end

function simd()
    case, forest, U, p, u, du = setup()
    npts = header(U, p)
    p1 = problem(U, case, u; simd_width=1)
    gh_rhs!(du, u, p1, zero(T))                # fills the ghosts the kernels read
    ref = copy(du)
    # LAB_SAVE=<file> keeps this process's `du`; LAB_COMPARE=<file> compares with a
    # kept one — how a `LAB_ZW=1` run is checked against the package's own stencils.
    path = get(ENV, "LAB_SAVE", "")
    isempty(path) || write(path, ref)
    path = get(ENV, "LAB_COMPARE", "")
    if !isempty(path)
        other = reinterpret(Float64, read(path))
        @printf("%s\tagainst %s: max |du − du′| = %.2e, bitwise equal: %s\n", TAG, path,
                maximum(abs.(ref .- other)), isequal(ref, other))
    end
    for W in WS
        W ≤ N || continue
        pW = problem(U, case, u; simd_width=W)
        fill!(du, NaN)
        gh_rhs!(du, u, pW, zero(T))
        @printf("%s\tW = %d: bitwise %s, max |du − scalar| = %.2e (%.0f eps), NaN left %d\n",
                TAG, W, isequal(du, ref), maximum(abs.(du .- ref)),
                maximum(abs.(du .- ref)) / eps(T), count(isnan, du))
        row("kernel, W = $W", timeit(() -> kernel!(pW, du)), npts)
        row("gh_rhs!, W = $W", timeit(() -> gh_rhs!(du, u, pW, zero(T))), npts)
    end
    return nothing
end

# --- native code -------------------------------------------------------------------

function asm_stats(io, name, text)
    lines = split(text, '\n')
    ins = filter(l -> occursin(r"^\s+[a-z]", l) && !occursin(r"^\s+\.", l), lines)
    count_re(re) = count(l -> occursin(re, l), ins)
    @printf(io, "%s\t%-10s instructions %6d  ymm-FP %5d  xmm-packed-FP %5d  scalar-FP %5d  stack loads %5d  stack stores %5d  calls %3d  div %3d  sqrt %3d\n",
            TAG, name,
            length(ins),
            count_re(r"\bv(fmadd|fmsub|fnmadd|fnmsub|add|sub|mul|div)[0-9a-z]*pd\s.*%[yz]mm"),
            count_re(r"\bv(fmadd|fmsub|fnmadd|fnmsub|add|sub|mul|div)[0-9a-z]*pd\s.*%xmm"),
            count_re(r"\bv(fmadd|fmsub|fnmadd|fnmsub|add|sub|mul|div)[0-9a-z]*sd\s"),
            count_re(r"\bvmov\w*\s+-?[0-9]*\(%rsp\)|\bvmov\w*\s+-?[0-9]*\(%rbp\)"),
            count_re(r"\bvmov\w*\s+%[xyz]mm[0-9]+,\s*-?[0-9]*\(%r[sb]p\)"),
            count_re(r"\bcall"), count_re(r"\bv?div[sp]d"), count_re(r"\bv?sqrt[sp]d"))
    return nothing
end

function asm()
    dir = get(ENV, "LAB_ASM", "")
    case, forest, U, p, u, du = setup(; roots=CASE == "hole" ? ROOTS : 1)
    gh_rhs!(du, u, p, zero(T))
    obj = TGH.gh_rhs_kernel!(CPU(; static=true))
    nd = size(statearray(du, p.U))
    nd = (nd[1], nd[2], nd[3], nd[5])
    ndrange, wgs, iterspace, dynamic = KernelAbstractions.launch_config(obj, nd, (nd[1:3]..., 1))
    ctx = KernelAbstractions.mkcontext(obj, KernelAbstractions.blocks(iterspace)[1], ndrange,
                                       iterspace, dynamic)
    for W in WS
        W ≤ N || continue
        args = TGH.gh_rhs_kernel_args(problem(U, case, u; simd_width=W), du, zero(T))
        text = sprint(io -> code_native(io, obj.f, (typeof(ctx), map(typeof, args)...);
                                        debuginfo=:none, syntax=:att))
        asm_stats(stdout, "W = $W", text)
        isempty(dir) || write(joinpath(dir, "kernel-$CASE-W$W.s"), text)
    end
    return nothing
end

# --- profile: where the ghost fill's time goes ---------------------------------------
#
# Julia's sampling profiler over repeated fills, flat, by self time: is it the copies'
# arithmetic, the launches, or the threads waiting?
function profile_fill()
    case, forest, U, p, u, du = setup()
    header(U, p)
    fill_hook!(p)
    Profile.clear()
    Profile.init(n=10^7, delay=0.0005)
    t0 = time()
    Profile.@profile while time() - t0 < 10
        fill_hook!(p)
    end
    println("# flat profile of fill_ghosts!, by self time")
    Profile.print(IOContext(stdout, :displaysize => (200, 220)); format=:flat,
                  sortedby=:overhead, mincount=50, C=false)
    return nothing
end

# --- parts: the head's pieces alone, scalar and four lanes ------------------------------
#
# What one call of each algebraic piece of `gh_rhs_head` costs on one thread, inlined
# into a loop over 4096 random states as the kernel inlines it, with every output
# folded into a checksum so that nothing is dead code: `metric_quantities` (the
# inverse metric, `α`, `β`, `γ^{ij}`, `√γ`), `gh_node_source_lean` and
# `metric_divergences`. The `Vec{4}` rows evaluate four states per call; their time
# is per state.
pack_lanes(xs::Vector{SVector{n,Float64}}, k, ::Val{W}) where {n,W} =
    SVector{n,Vec{W,Float64}}(ntuple(c -> Vec(ntuple(l -> xs[k + l - 1][c], Val(W))), Val(n)))

function parts_inputs(::Val{W}) where {W}
    rng = Xoshiro(7)
    n = 4096
    hs = [SVector{10,T}(T(0.08) .* randn(rng, T, 10)) for _ in 1:n]
    Ds = [[SVector{10,T}(randn(rng, T, 10)) for _ in 1:n] for _ in 1:4]
    if W == 1
        return [(hs[k], ntuple(a -> Ds[a][k], 4)) for k in 1:n]
    end
    return [(pack_lanes(hs, k, Val(W)), ntuple(a -> pack_lanes(Ds[a], k, Val(W)), 4))
            for k in 1:W:n]
end

@inline checksum(x::Number) = x
@inline checksum(x::Vec) = sum(x)
@inline checksum(x::StaticArray) = checksum(sum(x))

function sweep_metric(xs)
    s = 0.0
    @inbounds for (h, _) in xs
        g4, gu4, α, β, γu, sqrtγ = metric_quantities(TGH._sym4(h))
        s += checksum(gu4) + checksum(α) + checksum(β) + checksum(γu) + checksum(sqrtγ)
    end
    return s
end

function sweep_source(xs, pre)
    s = 0.0
    @inbounds for k in eachindex(xs)
        h, D = xs[k]
        g4, gu4, α, β, γu, sqrtγ = pre[k]
        E = typeof(α)
        Hl, dHl = zero(SVector{4,E}), zero(SMatrix{4,4,E})
        o = one(α)
        s += checksum(TGH.gh_node_source_lean(g4, gu4, α, sqrtγ, D[1], (D[2], D[3], D[4]),
                                              Hl, dHl, o, zero(α)))
    end
    return s
end

function sweep_div(xs, pre)
    s = 0.0
    @inbounds for k in eachindex(xs)
        h, D = xs[k]
        g4, gu4, α, β, γu, sqrtγ = pre[k]
        divβ, divA = TGH.metric_divergences(gu4, α, β, γu, sqrtγ, (D[2], D[3], D[4]))
        s += checksum(divβ) + checksum(divA)
    end
    return s
end

function parts()
    @printf("# %s  parts, one thread  host %s  cpu %s\n", TAG, gethostname(), Sys.CPU_NAME)
    for W in (1, 4)
        xs = parts_inputs(Val(W))
        pre = [Base.front(metric_quantities(TGH._sym4(h))) for (h, _) in xs]
        nstates = length(xs) * W
        for (name, f) in (("metric_quantities", () -> sweep_metric(xs)),
                          ("gh_node_source_lean", () -> sweep_source(xs, pre)),
                          ("metric_divergences", () -> sweep_div(xs, pre)))
            t = timeit(f; n=50)
            @printf("%s\t%-20s %s\t%8.1f ns per point\n", TAG, name,
                    W == 1 ? "scalar " : "Vec{$W}", 1e9t / nstates)
        end
    end
    return nothing
end

for m in MODES
    m == "breakdown" ? breakdown() : m == "simd" ? simd() : m == "asm" ? asm() :
        m == "profile" ? profile_fill() : m == "parts" ? parts() :
        error("unknown LAB_MODE $m")
end
