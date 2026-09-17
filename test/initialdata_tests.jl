# Initial data: the conversion between the two derivative index
# conventions, the momentum that is not `∂_t g`, and the cases the mesh is
# built from.
#
# `CODE.md`, "Initial data and backgrounds". The failure this file exists
# for is `CLAUDE.md`'s: a state built from the wrong slice of `dmetric`'s
# `dg`, or from `∂_t g` in place of the densitised, Lie-advected `Π`,
# looks right in Minkowski — where the shift vanishes and `α = √γ` — and
# is wrong everywhere else.

using ForwardDiff: ForwardDiff
using SpacetimeMetrics: GaugeWave, Harmonic, KerrSchild, Minkowski,
                        ShiftedMinkowski, adm_decompose, boost, dmetric,
                        metric
using LinearAlgebra: det
using StaticArrays: SMatrix, SVector

# The state of a background, spelled without `dmetric` and without
# `metric_quantities`: the derivatives by a `ForwardDiff` jacobian of
# `metric` itself, the ADM quantities by `SpacetimeMetrics`'
# `adm_decompose`, and `√γ` by a determinant. Nothing here shares a line
# with `background_state`, which is the point — the two conventions only
# meet once in the package, so the check has to come from outside it.
function reference_state(bg, t::T, x::SVector{3,T}) where {T}
    p = SVector{4,T}(t, x[1], x[2], x[3])
    g = SMatrix{4,4,T}(metric(bg, p))
    # J[k, c] = ∂_c g[k] with `k` the column-major index of (a, b).
    J = ForwardDiff.jacobian(q -> SVector{16}(Tuple(metric(bg, q))), p)
    ∂g = c -> SMatrix{4,4,T}(J[k, c] for k in 1:16)
    α, β, γ = adm_decompose(g)
    sqrtγ = sqrt(det(γ))
    ∂ₜg = ∂g(1)
    ∂ᵢg = ntuple(i -> ∂g(i + 1), Val(3))
    Π = (sqrtγ / α) *
        (∂ₜg - β[1] * ∂ᵢg[1] - β[2] * ∂ᵢg[2] - β[3] * ∂ᵢg[3])
    return pack_g(g), pack_sym(Π), ntuple(i -> pack_sym(∂ᵢg[i]), Val(3))
end

@testset "background_state converts dmetric's index order exactly once" begin
    # Guards the two derivative index conventions (`CLAUDE.md`):
    # `dmetric` returns `dg[a, b, c] = ∂_c g_ab`, the algebra wants
    # `∂_a g_bc`. A transposition passes every static test — `∂_t g` is
    # zero — which is why the rows below include two backgrounds that move
    # and one with a shift.
    T = Float64
    t = T(1 // 4)
    rows = (("Minkowski", Minkowski()),
            ("gauge wave", GaugeWave(T(1 // 20), one(T))),
            ("shifted Minkowski", ShiftedMinkowski(T(1 // 2), T(2))),
            ("harmonic Kerr", Harmonic{T}(1, 9 // 10)),
            ("boosted harmonic Kerr",
             boost(Harmonic{T}(1, 9 // 10), SVector{3,T}(3 // 10, 0, 0))))
    for (name, bg) in rows, x in (SVector{3,T}(3 // 2, -7 // 5, 11 // 5),
                                  SVector{3,T}(-5 // 2, 9 // 5, -3 // 5))
        h, Π, ∂h = background_state(bg, t, x)
        hr, Πr, ∂hr = reference_state(bg, t, x)
        scale = max(maximum(abs, hr), maximum(abs, Πr),
                    maximum(maximum(abs, d) for d in ∂hr), one(T))
        @test maximum(abs, h - hr) ≤ 1e-13 * scale
        @test maximum(abs, Π - Πr) ≤ 1e-13 * scale
        for i in 1:3
            @test maximum(abs, ∂h[i] - ∂hr[i]) ≤ 1e-13 * scale
        end
    end
end

@testset "Π is the densitised momentum and not ∂_t g" begin
    # Guards the mistake `prerequisite_tests.jl` deliberately *made* while
    # there was nothing to evolve: filling variables 11:20 with `∂_t g`.
    # The two agree only where the shift vanishes and `α = √γ`; on a static
    # background with a shift, `∂_t g` is identically zero and `Π` is not.
    T = Float64
    bg = ShiftedMinkowski(T(1 // 2), T(2))
    x = SVector{3,T}(1 // 2, -3 // 4, 1 // 4)
    _, dg = dmetric(bg, SVector{4,T}(0, x...))
    ∂ₜg = pack_sym(SMatrix{4,4,T}(dg[a, b, 1] for a in 1:4, b in 1:4))
    _, Π, _ = background_state(bg, zero(T), x)
    @test all(iszero, ∂ₜg)
    @test maximum(abs, Π) > T(1 // 100)

    # And on Minkowski, where the two *do* agree, both are zero — so the
    # test above is a statement about the shift and not about the norm.
    _, Π0, _ = background_state(Minkowski(), zero(T), x)
    @test all(iszero, Π0)
end

@testset "state_callback fills a field set with what a host loop computes" begin
    # Guards the callback path `CODE.md` relies on everywhere: the initial
    # data, the error reference and the Dirichlet hook are all this
    # closure, evaluated inside a kernel. It captures a background and a
    # time, both `isbits`, and it must produce bit-for-bit what the same
    # function produces on the host — a launch that ran but computed
    # something slightly different would put a floor under every error this
    # package measures.
    T = Float64
    q = 4
    G = q ÷ 2 + 1
    case = gauge_wave_case(T; ε_KO=zero(T), γ0=zero(T), γ2=zero(T))
    forest = gh_forest(T, case; N=8, roots=2)
    fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
    t = T(1 // 4)
    fill_exact!(fs, case, t)

    @test isbits(state_callback(case, t).f)
    ref = zeros(T, size(fs.work))
    for b in 1:nblocks(fs), k in 1:8, j in 1:8, i in 1:8
        idx = (i + G, j + G, k + G)
        vals = state_tuple(case.background, t, coordinates(fs, b, idx))
        for v in 1:20
            ref[idx..., v, b] = vals[v]
        end
    end
    # One comparison of the whole array, not one per point: the ghosts are
    # untouched on both sides (`fill_by_coordinates!` writes the owned
    # points only), so `isequal` covers the claim — and it reports a `NaN`
    # rather than swallowing it.
    @test isequal(fs.work, ref)
    @test any(!iszero, fs.work)
end

@testset "The cases are the rows of CODE.md's table, with their boxes" begin
    # Guards the case constructors: the gauge wave's box is one wavelength
    # and closes on itself, the shifted-Minkowski box is Dirichlet in `x`
    # and periodic in the two dimensions its profile does not depend on
    # (amended in step 3), and a hole case may not be built moving and
    # non-harmonic. A box that was periodic where the solution is not would
    # report an unresolved kink as a truncation error.
    T = Float64
    gw = gauge_wave_case(T; A=T(1 // 20), d=T(2), ε_KO=zero(T), γ0=zero(T),
                         γ2=zero(T))
    @test gw.periodic == (true, true, true)
    @test gw.box == ((zero(T), T(2)), (zero(T), T(2)), (zero(T), T(2)))
    @test isharmonic(gw.background)
    @test !isstatic(gw.background)

    sm = shifted_minkowski_case(T; halfwidth=T(3), ε_KO=zero(T), γ0=zero(T),
                                γ2=zero(T))
    @test sm.periodic == (false, true, true)
    @test sm.box == ((-T(3), T(3)), (-T(3), T(3)), (-T(3), T(3)))
    @test !isharmonic(sm.background)
    @test isstatic(sm.background)

    mk = minkowski_case(T; L=T(1), ε_KO=T(1 // 2), γ0=one(T), γ2=zero(T))
    @test mk.periodic == (true, true, true)
    @test mk.ε_KO == T(1 // 2)
    @test isbits(mk)

    # The shifted-Minkowski solution is `y`- and `z`-independent, which is
    # what makes those two dimensions periodic without a mismatch, and it
    # is not `x`-periodic, which is what makes `x` Dirichlet.
    x1 = SVector{3,T}(1, -2, 3)
    x2 = SVector{3,T}(1, 5, -7)
    s1 = background_state(sm.background, zero(T), x1)
    s2 = background_state(sm.background, zero(T), x2)
    @test isequal(s1[1], s2[1]) && isequal(s1[2], s2[2])
    @test all(isequal(s1[3][i], s2[3][i]) for i in 1:3)
    # In `x` it is not periodic, and the way it fails is the one that
    # matters for a second-order scheme: `ψ′(x) = A sech²(x/w)` is *even*,
    # so the two faces hold the same values and opposite gradients. A
    # periodic box would join them into a kink — continuous data with a
    # jump in `∂_x h`, which no stencil resolves and every error norm would
    # then report as truncation error.
    lo, hi = sm.box[1]
    slo = background_state(sm.background, zero(T), SVector{3,T}(lo, 0, 0))
    shi = background_state(sm.background, zero(T), SVector{3,T}(hi, 0, 0))
    @test isequal(slo[1], shi[1])
    @test maximum(abs, slo[3][1] - shi[3][1]) > T(1 // 10)
    @test maximum(abs, slo[3][1] + shi[3][1]) ≤ 1e-14
end

@testset "gh_forest builds the case's box and refuses one it cannot" begin
    # Guards the assumption TreeAMR's geometry makes and this package
    # inherits: blocks are cubes with one spacing per level, so a box that
    # is not a cube would silently have anisotropic cells and every
    # stencil would be scaled by the wrong `h`.
    T = Float64
    case = gauge_wave_case(T; d=T(2), ε_KO=zero(T), γ0=zero(T), γ2=zero(T))
    forest = gh_forest(T, case; N=8, roots=2)
    @test nleaves(forest) == 8
    @test maxlevel(forest) == 0
    @test minimum_spacing(T, forest) ≈ T(2) / 16
    @test forest.periodic == (true, true, true)

    slab = GHCase(T, Minkowski(); box=((zero(T), one(T)), (zero(T), T(2)),
                                       (zero(T), one(T))),
                  periodic=(true, true, true), ε_KO=zero(T), γ0=zero(T),
                  γ2=zero(T))
    @test_throws "anisotropic" gh_forest(T, slab; N=8, roots=2)
end
