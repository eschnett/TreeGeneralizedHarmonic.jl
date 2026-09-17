# The backgrounds, the case they are wrapped in, and the one place where
# `SpacetimeMetrics`' derivative index convention is converted into
# GHSO2's.
#
# `CODE.md`, "Initial data and backgrounds": every case is a **background
# plus parameters**, and the same object supplies the initial data, the
# Dirichlet data, the error reference, the gauge source where there is
# one, and — from step 5 — the interior's `u_exact` and the hole's
# analytic trajectory. There is no separate "initial data" object and
# never a stored copy of the exact solution: it is a function of `(t, x)`
# and is evaluated where it is needed.
#
# **The conversion.** `SpacetimeMetrics.dmetric` returns
# `dg[a, b, c] = ∂_c g_ab`, the derivative axis *last*; GHSO2's pointwise
# algebra, which `pointwise.jl` ports, uses `dg[a, b, c] = ∂_a g_bc`, the
# derivative axis *first*. `CLAUDE.md` says what mixing them costs: a
# solution that looks right in Minkowski and is wrong everywhere else.
# [`background_state`](@ref) is the one function that converts, and
# `test/initialdata_tests.jl` checks it against an independent spelling
# rather than against itself.
#
# **`Π` is not `∂_t h`.** The evolved momentum is the densitised,
# Lie-advected derivative along the normal,
# `Π_ab = (√γ/α)(∂_t − β^i ∂_i) g_ab`. The first evolution equation,
# `∂_t h_ab = β^i ∂_i h_ab + (α/√γ) Π_ab`, is what a state built from
# `∂_t g` alone would silently be; the two agree only where the shift
# vanishes and `α = √γ`.

"""
    GHCase(T = Float64, background; box, periodic, ε_KO, γ0, γ2)

A case: the background, the box it is evolved in, and the parameters of
the scheme that are properties of the physics rather than of the mesh.

`CODE.md`'s table under "Initial data and backgrounds" is the list of
backgrounds; `box` is the domain's `(lo, hi)` per dimension and `periodic`
says which dimensions close on themselves (the others take the Dirichlet
hook of [`dirichlet`](@ref)). `ε_KO` is the Kreiss–Oliger amplitude,
`γ0 ≥ 0` and `γ2 > −1` the Gundlach–Pretorius damping parameters. None has
a default: each is a number a run is judged by, and `CODE.md` records
`ε_KO ≈ 0.5` and `γ0 ≈ 1/M` as GHSO2's *recipe near a hole*, not as
something a flat-space test should inherit silently.

**A moving non-harmonic background is refused here**, with the message
`CODE.md` asks for under "Gauge and constraint damping": such a background
has a gauge source `H_a(x − vt)` that a per-chunk sample cannot represent,
and evaluating it in the kernel would cost about one right-hand side. The
proof-of-concept case, `boost(Harmonic(M, a), v)`, is harmonic — a boost
preserves `□x^a = 0` — which is why it is the case.

The struct is `isbits` whenever the background is, because the whole case
travels into kernels: the Dirichlet hook closes over it at every ghost
fill, and from step 5 the interior's `u_exact` does at every evaluation.

**(Amended in step 3.)** `CODE.md`'s file table put `GHCase` in
`driver.jl`. It is here instead, with the backgrounds it is made of: the
right-hand side needs a case two steps before there is a driver, and a
struct cannot be defined twice. `driver.jl` adds `evolve!` and the fields
the interior and the refinement need.
"""
struct GHCase{T,B}
    background::B
    box::NTuple{3,Tuple{T,T}}
    periodic::NTuple{3,Bool}
    ε_KO::T
    γ0::T
    γ2::T
end

function GHCase(::Type{T}, background; box, periodic, ε_KO, γ0, γ2) where {T}
    isharmonic(background) || isstatic(background) || throw(ArgumentError(
        "this background is neither harmonic nor static, so its prescribed " *
        "gauge source H_a(x − vt) depends on time, and CODE.md's Hsrc field " *
        "set — sampled once per chunk as a function of position — cannot " *
        "represent it; evaluating H_a in the kernel would cost about one " *
        "right-hand side and is listed as an extension, not built. A boosted " *
        "Kerr-Schild is exactly this case. Use boost(Harmonic(M, a), v) " *
        "instead: a boost preserves the harmonic condition □x^a = 0, so the " *
        "boosted hole in harmonic coordinates has H ≡ 0 and needs no source " *
        "at all. That is why it is the proof-of-concept case."))
    γ0 ≥ 0 || throw(ArgumentError(
        "the constraint-damping rate must satisfy γ0 ≥ 0 — the term enters " *
        "∂_tΠ with a factor −α√γ, so a negative γ0 drives the constraint " *
        "violation it is there to damp — but γ0 = $γ0"))
    γ2 > -1 || throw(ArgumentError(
        "the Gundlach–Pretorius trace parameter must satisfy γ2 > −1 at the " *
        "continuum level for the damped system to stay well posed, but " *
        "γ2 = $γ2"))
    all(d -> box[d][2] > box[d][1], 1:3) || throw(ArgumentError(
        "every dimension of the box needs hi > lo, but box = $box"))
    return GHCase{T,typeof(background)}(
        background, ntuple(d -> (T(box[d][1]), T(box[d][2])), Val(3)),
        ntuple(d -> Bool(periodic[d]), Val(3)), T(ε_KO), T(γ0), T(γ2))
end

GHCase(background; kwargs...) = GHCase(Float64, background; kwargs...)

"""
    minkowski_case(T = Float64; L, ε_KO, γ0, γ2)

Flat space on a periodic box of side `L` — `CODE.md`'s first row, the
robust-stability case. Its exact solution is `h = Π = 0`, so the
right-hand side is *exactly* zero on it, which is the sharpest statement
any test in this package makes about the kernel.
"""
minkowski_case(::Type{T}=Float64; L, ε_KO, γ0, γ2) where {T} =
    GHCase(T, Minkowski(); box=ntuple(_ -> (zero(T), T(L)), Val(3)),
           periodic=(true, true, true), ε_KO=ε_KO, γ0=γ0, γ2=γ2)

"""
    gauge_wave_case(T = Float64; A = 1//20, d = 1, ε_KO, γ0, γ2)

The Apples-with-Apples gauge wave, `ds² = −H dt² + H dx² + dy² + dz²` with
`H = 1 − A sin(2π(x − t)/d)` — flat spacetime in an oscillating chart,
travelling along `x` at coordinate speed 1, `CODE.md`'s second row and the
case that measures the scheme's order.

The box is one wavelength cubed and periodic in every dimension, so the
solution is exactly periodic and the mesh needs no boundary at all; one
crossing is `t = d`. `A = 1//20` is `PLAN.md`'s amplitude, written as a
rational so that the `Float32` and `Float64` runs are the same number
rounded once.
"""
gauge_wave_case(::Type{T}=Float64; A=T(1//20), d=one(T), ε_KO, γ0,
                γ2) where {T} =
    GHCase(T, GaugeWave(T(A), T(d)); box=ntuple(_ -> (zero(T), T(d)), Val(3)),
           periodic=(true, true, true), ε_KO=ε_KO, γ0=γ0, γ2=γ2)

"""
    shifted_minkowski_case(T = Float64; A = 1//2, w = 2, halfwidth = 2,
                           ε_KO, γ0, γ2)

Flat space in time-skewed coordinates `t̂ = t + A w tanh(x/w)` — static,
with a space-varying subluminal shift and, being non-harmonic, a
**sampled gauge source**. `CODE.md`'s third row, and the case that
exercises the `Hsrc` path and a nonzero `β^i`.

The profile `ψ′(x) = A sech²(x/w)` is a function of `x` alone and is not
periodic, so the box is Dirichlet in `x` and periodic in `y` and `z`
**(amended in step 3**; `CODE.md`'s "Boundaries: Periodic" had listed this
case as periodic in every dimension, which the profile is not: it is
*even* in `x`, so closing the box would join two equal values with
opposite gradients — a kink, which no stencil resolves and which every
error norm would then report as truncation error**)**. Being exactly
independent of `y` and `z`, the other two dimensions are periodic without
any mismatch at all. The Dirichlet data is the analytic solution, so the
boundary is exact; this is the case that puts [`dirichlet`](@ref) under
test.

`w = 2` on a box of half-width `2` **(proposed in step 3)**: what a
convergence study needs is the profile *resolved* at the coarsest
resolution it uses, and `w = 1` is not, at the three the suite can
afford. Measured at `q = 4`, `N = 8`, `roots = 2 … 4`: `w = 1` converges
at **3.6** and `w = 2` at **3.93**, with the same code — the shortfall is
the sech² profile against `h = L/16`, not the scheme. The Dirichlet
boundary is exact, so a profile that is still `0.42 A` at the face costs
nothing.
"""
shifted_minkowski_case(::Type{T}=Float64; A=T(1//2), w=T(2), halfwidth=T(2),
                       ε_KO, γ0, γ2) where {T} =
    GHCase(T, ShiftedMinkowski(T(A), T(w));
           box=ntuple(_ -> (-T(halfwidth), T(halfwidth)), Val(3)),
           periodic=(false, true, true), ε_KO=ε_KO, γ0=γ0, γ2=γ2)

"""
    gh_forest(T = Float64, case::GHCase; N, roots, refined = false)

A forest of `roots³` root blocks of `N³` points each, over the case's box
and with the case's periodicity — the mesh every test in step 3 runs on,
and, with `refined = true`, the **two-level** mesh that step 4 measures the
interface-order rule on.

`N` and `roots` have no defaults: they are the two numbers a convergence
study varies. TreeAMR's blocks are cubes with one isotropic spacing, so
the box has to be one too; `h = (box width)/(roots·N)`.

`refined = true` refines the one root block that contains the point three
eighths of the way along each axis, and balances — TreeWave's
`wave_forest` pattern (added in step 4), with two differences that matter
here. It refines a **single** block rather than a middle sub-box, because
the interface study runs at `roots = 2`, where every root block *is* a
half of the domain; and the refined region is a fixed *block*, so that a
convergence study which doubles `N` leaves the block layout alone and
every spacing halves — `CODE.md`'s frozen-hierarchy protocol under
"Refinement and regridding", and the only way a rate measured on a refined
mesh means anything. One refined block in a periodic `2³` root grid has a
coarse-fine face on each of its six sides, so both directions of the
interface — prolongation into the fine ghosts, injection into the coarse
ones — are exercised.

The leading `T` is the type the whole run is computed in and reaches the
field set, the schedule and the state vector through the forest, which is
the only place it has to be said (TreeWave's `wave_forest`, which this
follows).
"""
function gh_forest(::Type{T}, case::GHCase; N, roots, refined=false) where {T}
    widths = ntuple(d -> case.box[d][2] - case.box[d][1], Val(3))
    all(w -> w ≈ widths[1], widths) || throw(ArgumentError(
        "TreeAMR's blocks are cubes with a single spacing per level, so a " *
        "uniform forest over this box would have anisotropic cells: the " *
        "widths are $widths"))
    forest = Forest{T}(ntuple(_ -> roots, Val(3)); N=N, periodic=case.periodic,
                       extents=case.box)
    refined || return forest
    xref = ntuple(d -> case.box[d][1] + 3 * widths[d] / 8, Val(3))
    targets = filter(forest.leaves) do k
        ext = block_extent(forest, k)
        all(d -> ext[d][1] ≤ xref[d] < ext[d][2], 1:3)
    end
    length(targets) == 1 || throw(ArgumentError(
        "the two-level mesh refines exactly the one root block holding " *
        "$xref, but $(length(targets)) blocks claim it — which means the " *
        "reference point landed on a block face, and a mesh that depends on " *
        "which side it lands is not a mesh a convergence study can be " *
        "frozen on"))
    refine!(forest, targets)
    balance!(forest)
    return forest
end

gh_forest(case::GHCase; kwargs...) = gh_forest(Float64, case; kwargs...)

"""
    background_state(background, t, x) -> (h, Π, ∂h)

The analytic state of a background at `(t, x)`: the offset metric
`h_ab = g_ab − η_ab`, the densitised momentum
`Π_ab = (√γ/α)(∂_t − β^i ∂_i) g_ab`, and the spatial gradients
`∂h[i] = ∂_i h_ab`, all packed in the `(tt, tx, …, zz)` order.

**This is the one place the two derivative index conventions meet.**
`dmetric` returns `dg[a, b, c] = ∂_c g_ab`, so `dg[:, :, 1]` is `∂_t g`
and `dg[:, :, i+1]` is `∂_i g`; everything downstream of here speaks
GHSO2's `∂_a g_bc` (`CLAUDE.md`, "Two derivative index conventions"). The
gradients are the *analytic* ones, not differences of the sampled state:
`CODE.md`, "Initial data and backgrounds", keeps GHSO2's discrete-gradient
`Π` as a post-pass option for G4 and measures there whether it changes a
hole's stationarity.

Generic in the element type of `x`, including `ForwardDiff` duals, so that
a test may differentiate it — which is how `test/initialdata_tests.jl`
checks the conversion against a spelling that does not use `dmetric` at
all. It is also a kernel argument's body: [`state_callback`](@ref) and
[`dirichlet`](@ref) are closures around it, evaluated per point inside
`fill_by_coordinates!` and the boundary hook.
"""
@inline function background_state(bg, t, x)
    D = typeof(x[1])
    p = SVector{4,D}(D(t), x[1], x[2], x[3])
    g, dg = dmetric(bg, p)
    h = pack_g(g)
    ∂ₜg = pack_sym(SMatrix{4,4,D}(dg[a, b, 1] for a in 1:4, b in 1:4))
    ∂h = ntuple(Val(3)) do i
        pack_sym(SMatrix{4,4,D}(dg[a, b, i + 1] for a in 1:4, b in 1:4))
    end
    _, _, α, β, _, sqrtγ = metric_quantities(_sym4(h))
    Π = (sqrtγ / α) * (∂ₜg - β[1] * ∂h[1] - β[2] * ∂h[2] - β[3] * ∂h[3])
    return h, Π, ∂h
end

"""
    state_tuple(background, t, x) -> NTuple{20}

`(h..., Π...)`, the 20 evolved values at one point — [`background_state`](@ref)
in the shape TreeAMR's `AllVariables` callbacks return.

The spatial gradients are dropped here and *not* stored: the evolution
takes them from the mesh, and keeping the analytic ones would be a second
opinion about a quantity the scheme has to produce for itself.
"""
@inline function state_tuple(bg, t, x)
    h, Π, _ = background_state(bg, t, x)
    return (Tuple(h)..., Tuple(Π)...)
end

"""
    state_callback(case::GHCase, t) -> AllVariables

The case's exact state at time `t` as the coordinate callback
`fill_by_coordinates!` and `adapt_to_initial_data!` take: `x ↦ (h…, Π…)`,
once per point, all 20 variables at once.

It closes over the background and over `t` — both `isbits`, neither a
`Type` — so it is a kernel argument like any other and runs wherever the
field set lives (`CODE.md`, "Initial data and backgrounds"; the dependency
risk it names is what `test/prerequisite_tests.jl` settles on the CPU).

Built fresh at each call with that call's time, which is the rule for
every hook in this package: the initial data at `t = 0`, the error
reference at the end of a chunk, the Dirichlet data at every ghost fill
(`CLAUDE.md`, "Hooks depend on time").
"""
function state_callback(case::GHCase{T}, t) where {T}
    bg = case.background
    tt = T(t)
    return AllVariables(x -> state_tuple(bg, tt, x))
end

"""
    fill_exact!(fs::FieldSet, case::GHCase, t)

Fill `fs` with the case's exact state at time `t` — the initial data, and
the reference an error norm is taken against.

One line over [`state_callback`](@ref), written out because it is the
call every study makes twice, and because a field set filled from a
*different* layout than the state's would sample the solution at
different points (TreeWave's note on the error field set).
"""
fill_exact!(fs::FieldSet, case::GHCase, t) =
    fill_by_coordinates!(state_callback(case, t), fs)
