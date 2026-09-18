# What the pinned dependencies have to provide before any of the
# physics can be written.
#
# `Project.toml` pins TreeAMR and SpacetimeMetrics — and, from step 7,
# `ApparentHorizonFinder` and `KorzynskiSpin` — to GitHub `main` through
# `[sources]` entries, so what these tests run against is a resolved commit
# of each of those branches and *not* the checkouts at `~/src/jl/TreeAMR`
# and `~/src/jl/SpacetimeMetrics`. Three claims are made here, and a `main`
# that lost any of them would otherwise be found by a `MethodError` in the
# middle of a later step rather than at the top of the suite:
#
#   1. a `SpacetimeMetrics` background, with its `dmetric` forward-mode
#      pass, compiles and runs *inside a KernelAbstractions kernel* and
#      fills a field set with bit-for-bit what a host loop computes; and
#   2. the pinned TreeAMR and SpacetimeMetrics still export the names this
#      package is written against; and
#   3. the pinned `ApparentHorizonFinder` still finds Kerr's horizon from
#      an *analytic* provider, and `KorzynskiSpin` still gets `J = M a`
#      from it — which is what separates "the libraries work" from "this
#      package's interpolation works", the claim `horizon_tests.jl` makes
#      on top of it (added in step 7).
#
# The first is the dependency risk `CODE.md` names under "Initial data and
# backgrounds": the background is evaluated inside kernels — the Dirichlet
# hook at every ghost fill, the interior's `u_exact` at every right-hand
# side evaluation — so it has to be a kernel argument, which means
# StaticArrays plus ForwardDiff duals have to compile as one. On the CPU
# backend that is what this file settles; on the H200 it is G6's job.
#
# Nothing here is general relativity. These are claims about the mesh and
# about the metric library, plus the two host-side helpers of `device.jl`.

# The module itself as well as the names, because the last testset asks it
# what it exports.
import SpacetimeMetrics
using SpacetimeMetrics: AbstractMetric, ExtrinsicCurvature, Harmonic,
                        KerrSchild, boost, dmetric
using StaticArrays: SArray, SMatrix, SVector
import ApparentHorizonFinder
import KorzynskiSpin

# The layout of the evolved state, exactly as `CODE.md`'s "Field sets and
# layout" table gives it at `q = 4`: 20 variables (`h` is 1:10, `Π` is
# 11:20), `G = q/2 + 1 = 3`, vertex-centered. `N = 8` is the smallest a
# vertex-centered field set allows at this `G` (TreeAMR requires
# `N ≥ 2G + 2c`), and it is what every test in this package uses.
const PREREQ_N = 8
const PREREQ_G = 3
const PREREQ_NVARS = 20

# A box that contains no black hole: the backgrounds below are singular at
# the origin, and this file is about the mesh and the compiler, not about
# the interior rule that step 5 adds. Two roots per dimension with one of
# them refined, so the blocks do not all have the same spacing — which is
# what makes the comparison against `coordinates` say something.
function prereq_forest(::Type{T}) where {T}
    forest = Forest{T}((2, 2, 2); N=PREREQ_N, periodic=(false, false, false),
                       extents=((2, 4), (2, 4), (2, 4)))
    refine!(forest, [first(forest.leaves)])
    balance!(forest)
    return forest
end

# The packed component order, `CODE.md`'s and GHSO2's: the column-major
# lower triangle `(tt, tx, ty, tz, xx, xy, xz, yy, yz, zz)`, with the offset
# `h = g − η` folded into `pack_g`. Both come from `pointwise.jl` (added in
# step 1), which is the one place that knows the order; the spelling this
# file carried before it existed is now asserted against it in
# `pointwise_tests.jl` rather than maintained twice.

# The initial state of `CODE.md`'s callback, in miniature: `h` from the
# metric, and `∂_t g` from the `dmetric` pass in the ten slots the layout
# reserves for the momentum. `dmetric` returns `dg[a, b, c] = ∂_c g_ab` —
# the derivative axis *last* — so the time derivative is `dg[:, :, 1]`.
# That is the convention this package converts in `initialdata.jl` and
# nowhere else; here it is used as it comes.
#
# **`∂_t g` is a stand-in for the `Π` slot, not `Π`.** The evolved
# momentum is `Π_ab = (√γ/α)(∂_t − β^i ∂_i) g_ab = √|g| n^μ ∂_μ g_ab`,
# densitised and Lie-advected (`CODE.md`, "The equations"); it coincides
# with `∂_t g` only where the shift vanishes and `α = √γ`, which is
# nowhere near a black hole. Nothing here evolves anything, and the claim
# under test is about the *kernel argument* — that a background with its
# forward-mode pass compiles and produces the same numbers inside a
# launch as on the host — so the stand-in is deliberate and the real `Π`
# arrives with `initialdata.jl` in step 3.
@inline function prereq_state(bg, x::NTuple{3,T}) where {T}
    g, dg = dmetric(bg, SVector{4,T}(zero(T), x[1], x[2], x[3]))
    dtg = SMatrix{4,4,T}(dg[a, b, 1] for a in 1:4, b in 1:4)
    return (pack_g(g)..., pack_sym(dtg)...)
end

# One static background and one that moves. The first is the one `CODE.md`
# names as the non-harmonic hole; the second is the proof-of-concept case
# itself, `boost(Harmonic(M, a), v)`, which is the expensive one to
# compile — nested duals through a coordinate pullback — and the one whose
# `∂_t g` is not identically zero, so that both halves of the packed state
# are compared against something.
prereq_backgrounds(::Type{T}) where {T} =
    (KerrSchild{T}(1, 0),
     boost(Harmonic{T}(1, 9 // 10), SVector{3,T}(3 // 10, 0, 0)))

@testset "A background runs as a kernel argument: T=$T" for T in (Float64, Float32)
    # Guards the dependency risk of "Initial data and backgrounds": a
    # `SpacetimeMetrics` background has to survive being a kernel argument,
    # because the Dirichlet hook and the interior's `u_exact` are evaluated
    # inside kernels and there is no host-side path for either. It guards
    # the quieter failure too: a launch that runs but fills the field set
    # with *different numbers* than the host does would put a floor under
    # every error this package measures, so the comparison is bit-for-bit
    # and not a tolerance.
    forest = prereq_forest(T)
    @test nleaves(forest) == 15                    # 8 roots, one of them refined
    @test maxlevel(forest) == 1

    for bg in prereq_backgrounds(T)
        # Everything a kernel argument closes over must be `isbits`. A
        # background is a small struct of numbers, and `boost` wraps it in
        # another one; if either grew a field that is not, the launch below
        # would still work on the CPU and fail on a device.
        @test isbits(bg)
        @test bg isa AbstractMetric

        fs = FieldSet{T}(forest, PREREQ_NVARS; G=PREREQ_G,
                         centering=vertexcentered(3))
        @test size(fs.work) == (PREREQ_N + 2 * PREREQ_G + 1, PREREQ_N + 2 * PREREQ_G + 1,
                                PREREQ_N + 2 * PREREQ_G + 1, PREREQ_NVARS,
                                nleaves(forest))

        fill_by_coordinates!(AllVariables(x -> prereq_state(bg, x)), fs)

        # The host loop the launch is compared against. It goes through
        # `coordinates`, which takes *stored* indices — owned point `i` is
        # stored at `i + G` — and which is the only positional arithmetic
        # anything in this package outside a kernel is allowed to do.
        ref = zeros(T, size(fs.work))
        for b in 1:nblocks(fs), k in 1:PREREQ_N, j in 1:PREREQ_N, i in 1:PREREQ_N
            idx = (i + PREREQ_G, j + PREREQ_G, k + PREREQ_G)
            vals = prereq_state(bg, coordinates(fs, b, idx))
            for v in 1:PREREQ_NVARS
                ref[idx..., v, b] = vals[v]
            end
        end

        # `isequal` rather than `==`: bit-for-bit, and it would report a
        # `NaN` rather than swallowing it.
        @test isequal(fs.work, ref)
        @test all(isfinite, fs.work)               # the box holds no hole
        @test any(!iszero, fs.work)                # and something was filled
        @test eltype(fs.work) === T                # no promotion on the way in
    end

    # The boosted hole moves, so its `∂_t g` is not zero anywhere; a form
    # that filled the momentum half from the wrong slice of `dg` — the
    # derivative axis is *last* in `dmetric`'s convention and *first* in
    # GHSO2's — would leave it zero and pass every test above.
    moving = last(prereq_backgrounds(T))
    fs = FieldSet{T}(forest, PREREQ_NVARS; G=PREREQ_G, centering=vertexcentered(3))
    fill_by_coordinates!(AllVariables(x -> prereq_state(moving, x)), fs)
    @test any(!iszero, @view fs.work[:, :, :, 11:20, :])
end

@testset "A field set is copied to the host without changing its layout" begin
    # Guards `hostcopy`'s two halves. On the CPU it returns the field set
    # itself, so its copying path would otherwise be dead code on every
    # machine CI runs on; `hostcopy!` is that path, called here host to
    # host. The layout is the thing to get wrong: a destination built
    # without `G` and the centering has a differently shaped working array,
    # and a `copyto!` between two arrays of equal length and unequal shape
    # transposes the data instead of failing.
    T = Float64
    forest = prereq_forest(T)
    bg = first(prereq_backgrounds(T))
    src = FieldSet{T}(forest, PREREQ_NVARS; G=PREREQ_G, centering=vertexcentered(3))
    fill_by_coordinates!(AllVariables(x -> prereq_state(bg, x)), src)
    @test hostcopy(src) === src

    dst = FieldSet{T}(src.forest, src.nvars; G=src.G, centering=src.centering)
    @test TreeGeneralizedHarmonic.hostcopy!(dst, src) === dst
    @test dst.work !== src.work
    @test isequal(dst.work, src.work)

    # A destination whose ghost width was left at something else is the
    # mistake, and it is an error rather than a transposition.
    thin = FieldSet{T}(forest, PREREQ_NVARS; G=PREREQ_G - 1,
                       centering=vertexcentered(3))
    @test_throws "same layout" TreeGeneralizedHarmonic.hostcopy!(thin, src)

    # And so is a destination over a different forest, even one with the
    # same leaves: the block order comes from the leaf array.
    other = FieldSet{T}(prereq_forest(T), PREREQ_NVARS; G=PREREQ_G,
                        centering=vertexcentered(3))
    @test_throws "same forest" TreeGeneralizedHarmonic.hostcopy!(other, src)
end

# Every name this package reaches for, in the order of `CODE.md`'s "File
# layout" table: the mesh and its geometry, the field sets and their
# coordinate callbacks, the ghost exchange and its physical-boundary hook,
# the ODE coupling, the regridding, and the two device-side reductions the
# indicator and the analysis record go through.
const TREEAMR_NAMES = (
    # Forest and geometry
    :Forest, :nleaves, :maxlevel, :generation, :find_leaf, :refine!, :coarsen!,
    :balance!, :spacing, :minimum_spacing, :block_origin, :block_origins,
    :block_spacings,
    # Field sets and coordinate callbacks
    :FieldSet, :nblocks, :blockkey, :blockview, :coordinates,
    :fill_by_coordinates!, :vertexcentered, :AllVariables,
    # Ghost exchange, interpolation operators, physical boundaries
    :Operators, :PointValue, :GhostSchedule, :fill_ghosts!, :CellBoundary,
    # ODE coupling
    :statevector, :statearray, :scatter!, :gather!, :map_blocks!,
    :block_mapreduce, :volume_weighted_norm,
    # Regridding
    :Refine, :Coarsen, :Keep, :flag_blocks, :buffered_flags, :complete_marks,
    :regrid!, :adapt_to_initial_data!,
    # Device-side flagging
    :firing_boxes,
)

# The backgrounds of `CODE.md`'s "Initial data and backgrounds" table, the
# two derivative passes the pointwise algebra is tested against, the ADM
# decomposition it is compared to, and the gauge source the non-harmonic
# cases sample.
const SPACETIMEMETRICS_NAMES = (
    :AbstractMetric, :metric, :dmetric, :ddmetric, :adm_decompose,
    :gauge_source, :gauge_source_grad,
    :translate, :rotate, :boost,
    :Minkowski, :GaugeWave, :ShiftedMinkowski, :KerrSchild, :Harmonic,
)

# The metric *wrappers* `SpacetimeMetrics` builds but does not export.
# `src/gauge.jl`'s `isharmonic` dispatches on them — `boost` and `rotate`
# preserve the harmonic condition, the gauge-wave transformation preserves
# it only over Minkowski — and its fallback is `false`, so a rename on
# `main` would not throw: it would quietly give a harmonic background a
# sampled gauge source, or refuse a moving one. Named here so that the
# rename is a failure at the top of the suite with the name in it
# (added in step 3).
const SPACETIMEMETRICS_INTERNAL = (
    :TranslatedMetric, :RotatedMetric, :BoostedMetric, :GaugeWaveMetric,
    :ShiftedMinkowskiMetric,
)

# The names `src/horizon.jl` calls, in the two libraries step 7 adds. Both
# are pinned to a moving `main` like the other two, and both are reached
# through a handful of functions whose *shape* matters as much as their
# existence: `find_horizon` returns the NamedTuple `horizon_spin` consumes,
# and `ADMVars` is the struct the interpolating provider fills.
const AHF_NAMES = (:ADMVars, :find_horizon, :horizon_points, :horizon_grid,
                   :horizon_area, :horizon_shape, :pointwise)
const KORZYNSKI_NAMES = (:horizon_spin, :SpinResult, :shape_embedding)

@testset "The pinned dependencies export the names this package calls" begin
    # A name list rather than a call: every one of these is reached for in
    # steps 1–10, and the cheapest place to find out that a `main` renamed
    # one is here, at the top of the suite, with the name in the failure
    # message rather than a `MethodError` from inside a run. The two
    # `[sources]` pins are moving branches, which is exactly why this test
    # exists — see `CLAUDE.md`, "Things that will bite".
    @test setdiff(TREEAMR_NAMES, names(TreeAMR)) == Symbol[]
    @test setdiff(SPACETIMEMETRICS_NAMES, names(SpacetimeMetrics)) == Symbol[]
    @test filter(n -> !isdefined(SpacetimeMetrics, n),
                 collect(SPACETIMEMETRICS_INTERNAL)) == Symbol[]
    @test setdiff(AHF_NAMES, names(ApparentHorizonFinder)) == Symbol[]
    @test setdiff(KORZYNSKI_NAMES, names(KorzynskiSpin)) == Symbol[]
end

# The two horizon libraries on *analytic* Cauchy data, which is the
# baseline every number in `horizon_tests.jl` is measured against: what is
# left there is this package's interpolation and nothing else. The
# provider is the one from `ApparentHorizonFinder`'s own README, written
# out here rather than reached for, because a README is not an interface.
@testset "The pinned horizon libraries find Kerr's horizon analytically" begin
    M, a = 1.0, 0.9
    ks = KerrSchild(M, a)
    function analytic_adm(p::SVector{3})
        x = SVector{4}(0.0, p[1], p[2], p[3])
        g, ∂g = dmetric(ks, x)
        K = ExtrinsicCurvature(ks, x)
        γ = SMatrix{3,3}(g[i, j] for i in 2:4, j in 2:4)
        ∂γ = SArray{Tuple{3,3,3}}(∂g[i, j, k] for i in 2:4, j in 2:4,
                                  k in 2:4)
        return ApparentHorizonFinder.ADMVars(γ, ∂γ, K)
    end
    # A displaced guess, as every find in this package uses.
    res = ApparentHorizonFinder.find_horizon(analytic_adm,
                                             SVector{3}(0.1, -0.05, 0.05),
                                             16, 1.6, 0.0, 200; verbosity=0)
    @test res.success
    rp = M + sqrt(M^2 - a^2)
    @test isapprox(res.area, 4π * (rp^2 + a^2); rtol=1e-8)
    @test isapprox(sqrt(sum(abs2, res.origin)), 0; atol=1e-8)
    spin = KorzynskiSpin.horizon_spin(res, x::SVector{3} -> analytic_adm(x).γ,
                                      x::SVector{3} -> analytic_adm(x).K)
    @test isapprox(spin.J, M * a; rtol=1e-6)
    @test isapprox(abs(spin.axis_embedding[3]), 1; atol=1e-6)
end
