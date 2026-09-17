# The two constraint monitors: that they are the constraints, that they
# vanish where the solution is exact, and that they converge where it is
# not.
#
# `CODE.md`, "Analysis quantities", and milestone G3. A monitor is only
# worth recording if it is zero on a solution and order `q` on a
# discretization of one, and the two halves need different settings to say
# so:
#
#   * **zero**: flat space, where every stencil is exact and the answer is
#     `0.0` with no tolerance to argue about; and the pointwise assembly on
#     *analytic* second derivatives of every background in `CODE.md`'s
#     table, where the answer is roundoff against the size of the terms.
#     The second is the sharp one — a wrong sign or a dropped term in the
#     four-dimensional Ricci tensor is O(1) there and would still converge
#     to something on a mesh.
#   * **order `q`**: the gauge wave on the two-level mesh, where the
#     violation is the interface error and nothing else (the same case on a
#     uniform mesh has *no* violation: it depends on `x − t` alone, so the
#     temporal and spatial truncation errors cancel, which `CODE.md`
#     records in "The equations"); and harmonic Kerr on a uniform mesh,
#     where it is the bulk truncation error and nothing else.
#
# Both rows are needed. A monitor with the interface term wrong would pass
# the second; one with a stencil at the wrong width would pass neither, but
# only the second says which.

using SpacetimeMetrics: Harmonic, Minkowski, ddmetric
using StaticArrays: SArray, SMatrix, SVector
using TreeGeneralizedHarmonic: DIAG_CGH, DIAG_DRIFT, DIAG_ERR, DIAG_HAM,
                               DIAG_MASK, DIAG_MOM, DIAG_RES, DIAG_SPEED,
                               DIAG_TAU, NDIAG, _dg4_last, _pairindex,
                               _pairindex3, _sym4, gauge_at
import TreeGeneralizedHarmonic: is_evolved

isdefined(@__MODULE__, :gh_backgrounds) || include("pointwise_backgrounds.jl")

# A mask that keeps half the box — the plumbing step 5's interior will use,
# exercised here on something whose answer can be written down. It is
# `isbits` and closes over one number, which is what a kernel argument has
# to be.
struct HalfSpace{T}
    x0::T
end
@inline is_evolved(m::HalfSpace, x) = x[1] ≥ m.x0

# The analytic derivatives of a background in **GHSO2's** index order, from
# `SpacetimeMetrics`' own second-derivative pass: `ddmetric` returns
# `ddg[a, b, c, d] = ∂_d∂_c g_ab`, with both derivative axes trailing, and
# `adm_constraints_at_node` takes them leading. Transposing here rather
# than in the package is deliberate: the package never differentiates a
# background twice, so this conversion has exactly one consumer and it is a
# test.
function analytic_curvature(bg, t, x::SVector{3,T}) where {T}
    p = SVector{4,T}(T(t), x[1], x[2], x[3])
    g, dgl, ddgl = ddmetric(bg, p)
    dg = SArray{Tuple{4,4,4},T}(dgl[b, c, a] for a in 1:4, b in 1:4, c in 1:4)
    ddg = SArray{Tuple{4,4,4,4},T}(ddgl[c, d, b, a]
                                   for a in 1:4, b in 1:4, c in 1:4, d in 1:4)
    return g, dg, ddg
end

@testset "The ADM constraints vanish on an exact vacuum solution" begin
    # Guards the four-dimensional Ricci assembly itself, with no mesh and
    # no stencil anywhere in it: every background in `CODE.md`'s table
    # solves `R_ab = 0`, so `ℋ` and `ℳ_i` built from its *analytic* first
    # and second derivatives must be roundoff against the size of the terms
    # that cancelled. This is the only test in the package that would catch
    # a sign error in `∂_c Γ^c_ab − ∂_b Γ^c_ca`; on a mesh such an error
    # would leave a violation that converges at order `q` to something
    # nonzero, and every other test here would pass.
    #
    # Two rows and one precision. Kerr-Schild is the background
    # `pointwise_tests.jl` already compiles `ddmetric` for — a second
    # nested dual pass per background is twenty seconds that buys no
    # further claim (step 1's measurement, and the reason that file names
    # its row) — and the gauge wave is added because it is the one row
    # whose `∂_t∂_t g` is not identically zero.
    T = Float64
    rows = gh_backgrounds(T)[[2, 4]]           # gauge wave, Kerr-Schild
    for (name, bg, _) in rows, x in gh_points(T, 2)
        g, dg, ddg = analytic_curvature(bg, T(GH_TIME), x)
        g4, gu4, α, β, _, _ = metric_quantities(pack_g(g) |> _sym4)
        ℋ, ℳ = adm_constraints_at_node(g4, gu4, α, β, dg, ddg)
        # The curvature terms are second derivatives and squares of first
        # ones; the constraints are what is left after they cancel, so the
        # tolerance is measured against the terms and not against zero
        # (`pointwise_backgrounds.jl` says the same about every other
        # comparison here).
        scale = max(one(T), maximum(abs, ddg), maximum(abs, dg)^2)
        @test abs(ℋ) < 16 * eps(T) * scale            # ≤ 1.2 eps measured
        @test maximum(abs, ℳ) < 16 * eps(T) * scale   # ≤ 0.2 eps measured
    end

    # And the check is not vacuous: perturb the metric off the solution and
    # the constraints are O(the perturbation), not roundoff.
    name, bg, _ = rows[2]
    x = gh_points(T, 1)[1]
    g, dg, ddg = analytic_curvature(bg, T(GH_TIME), x)
    bumped = SArray{Tuple{4,4,4,4},T}(ddg[a, b, c, d] +
                                      (a == 2 && b == 2 && c == 3 && d == 3 ?
                                       T(1 // 10) : zero(T))
                                      for a in 1:4, b in 1:4, c in 1:4, d in 1:4)
    g4, gu4, α, β, _, _ = metric_quantities(pack_g(g) |> _sym4)
    ℋbad, ℳbad = adm_constraints_at_node(g4, gu4, α, β, dg, bumped)
    @test abs(ℋbad) > 1e-3
    @test maximum(abs, ℳbad) > 1e-3
end

@testset "The symmetric pair index inverts the packing it names" begin
    # Guards the one new index convention step 4 introduced. The second
    # derivatives are packed by the symmetric *pair* `(μν)` in exactly the
    # order `_pack10` packs `(ab)`, and `_pairindex` is the arithmetic that
    # finds the slot. Get it wrong and `∂_t∂_x g` lands where `∂_x∂_y g`
    # belongs — which the tests below would catch as a constraint that does
    # not vanish, but only after a run and without naming the cause.
    T = Float64
    M = SMatrix{4,4,T}(ntuple(k -> T(k), Val(16)))
    M = (M + M') / 2
    packed = TreeGeneralizedHarmonic._pack10(M)
    @test all(packed[_pairindex(a, b)] == M[a, b] for a in 1:4, b in 1:4)
    @test _pairindex(1, 1) == 1 && _pairindex(4, 4) == 10
    # And the spatial six, `(xx, xy, xz, yy, yz, zz)` — the order
    # `gh_node_rhs_expanded` takes `∂∂h` in.
    spatial = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    @test all(_pairindex3(i, j) == n for (n, (i, j)) in enumerate(spatial))
    @test all(_pairindex3(i, j) == _pairindex3(j, i) for i in 1:3, j in 1:3)
end

@testset "The one-direction chain rule is the three-direction one" begin
    # Guards the second spelling this step added. `metric_derivatives` was
    # measured in step 1 against a forward-mode dual pass and is not
    # touched; `metric_derivatives_along` is the same five closed forms for
    # one direction, returning `∂√γ` and `∂γ^{jk}` separately so that the
    # constraint monitors can take them along **time**. If the two ever
    # disagree beyond roundoff, one of them has drifted — and only this
    # says which.
    T = Float64
    for (name, bg, _) in gh_backgrounds(T)[[2, 5]], x in gh_points(T, 2)
        h, _, ∂h = gh_state(bg, T(GH_TIME), x)
        _, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
        dα, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
        scale = max(one(T), maximum(maximum(abs, ∂h[i]) for i in 1:3))
        for i in 1:3
            dαi, dβi, dsqrtγi, dγui = metric_derivatives_along(gu4, α, β, γu,
                                                               sqrtγ, ∂h[i])
            @test abs(dα[i] - dαi) < 64 * eps(T) * scale
            @test maximum(abs, SVector{3,T}(dβ[i, j] for j in 1:3) .- dβi) <
                  64 * eps(T) * scale
            # And `∂√γ`, `∂γ^{jk}` reassemble `∂(α√γγ^{jk})` by the product
            # rule, which is what `metric_derivatives` returns instead.
            reassembled = SMatrix{3,3,T}(
                dαi * sqrtγ * γu[j,k] + α * dsqrtγi * γu[j,k] +
                α * sqrtγ * dγui[j,k] for j in 1:3, k in 1:3)
            ref = SMatrix{3,3,T}(dA[i, j, k] for j in 1:3, k in 1:3)
            @test maximum(abs, reassembled .- ref) < 256 * eps(T) * scale
        end
    end
end

@testset "Both monitors are exactly zero on flat space" begin
    # Guards the sharpest statement a monitor can make. Minkowski is
    # `h = Π = 0` at every stored point, so every stencil sums weights
    # against zeros and every coefficient is exactly its flat value: the
    # constraints are `0.0`, at every order, with no tolerance. A monitor
    # that leaked a term — a Christoffel built from `g` rather than from
    # the offset, a `∂_t∂_t g` assembled with the wrong sign on `φ ∂_tΠ` —
    # would show a number here and nowhere else without argument.
    T = Float64
    for q in (2, 4)
        case = minkowski_case(T; L=one(T), ε_KO=T(1 // 2), γ0=one(T),
                              γ2=T(-1 // 2))
        r = gh_constraint_run(T, case; N=8, roots=2, q=q)
        @test r.gauge_l2 == 0
        @test r.gauge_linf == 0
        @test r.ham_l2 == 0
        @test r.ham_linf == 0
        @test r.mom_l2 == 0
        @test r.mom_linf == 0
    end
end

@testset "The gauge-constraint kernel computes what the host computes" begin
    # Guards the kernel's half of the gauge monitor against a host-side
    # assembly out of `apply_stencil` and the ported
    # `gauge_constraint_at_node` — `evolution_tests.jl`'s pattern, and for
    # the same reason: the kernel forms its stencils by a linear index and
    # its `∂_t g` from the evolution equation, and a component read at the
    # wrong offset still converges to something.
    T = Float64
    q = 4
    case = shifted_minkowski_case(T; ε_KO=zero(T), γ0=one(T), γ2=T(-1 // 2))
    forest, fs, prob = gh_setup(T, case; N=8, roots=1, q=q)
    t = T(1 // 4)
    fill_exact!(fs, case, t)
    u = statevector(fs)
    gather!(u, fs)
    gh_constraint!(prob, u, t)

    w1 = derivative_weights(T, Val(q), Val(1))
    worst = zero(T)
    scale = zero(T)
    for b in 1:nblocks(fs), owned in ((1, 1, 1), (2, 5, 8), (8, 8, 8), (4, 3, 7))
        idx = ntuple(d -> owned[d] + fs.G[d], Val(3))
        h = spacing(T, forest, blockkey(fs, b))
        line(v, d) = ξ -> fs.work[Base.setindex(idx, ξ, d)..., v, b]
        ∂(v, d) = apply_stencil(w1, line(v, d), idx[d], 1) / h
        hv = SVector{10,T}(ntuple(v -> fs.work[idx..., v, b], Val(10)))
        Πv = SVector{10,T}(ntuple(v -> fs.work[idx..., 10 + v, b], Val(10)))
        ∂h = ntuple(d -> SVector{10,T}(ntuple(v -> ∂(v, d), Val(10))), Val(3))
        g4, gu4, α, β, _, sqrtγ = metric_quantities(_sym4(hv))
        ∂ₜh = β[1] * ∂h[1] + β[2] * ∂h[2] + β[3] * ∂h[3] + (α / sqrtγ) * Πv
        Hl, _ = gauge_at(T, prob.Hsrc.work, owned, b, Val(true))
        Cl = g4 * gauge_constraint_at_node(g4, _dg4_last(∂ₜh, ∂h), gu4 * Hl)
        for a in 1:4
            worst = max(worst,
                        abs(prob.diag.work[owned..., DIAG_CGH + a - 1, b] - Cl[a]))
        end
        # The scale the difference is measured against: the first
        # derivatives the contracted Christoffel is built from, which
        # cancel into a residual far smaller than themselves.
        scale = max(scale, maximum(maximum(abs, ∂h[d]) for d in 1:3), one(T))
    end
    @test worst < 1e-12 * scale
    # And the mask indicator is one at every owned point under `AllPoints`.
    # `interiorview`, not the whole working array: a vertex-centered field
    # set stores the shared upper plane, which belongs to no owned range
    # and which no kernel launched by `map_blocks!` writes.
    @test all(b -> all(isone, interiorview(prob.diag, b, DIAG_MASK)),
              1:nblocks(prob.diag))
end

@testset "Both monitors converge at order q across a coarse-fine face" begin
    # The acceptance test for G3's second half. The gauge wave on a
    # *uniform* mesh has no constraint violation at all — it depends on
    # `x − t` alone, so its temporal and spatial truncation errors cancel,
    # and the monitors sit at roundoff — so every number here is the
    # coarse-fine interface and nothing else. Both monitors must converge,
    # and faster than the scheme: the violation lives on a set of measure
    # `~h` around the interface, which is worth half an order in an L2
    # norm.
    T = Float64
    q = 4
    case = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=zero(T), γ0=one(T),
                           γ2=zero(T))
    # The uniform control first: the violation is roundoff, which is why
    # the refined rows below are a measurement of the interface.
    flat = gh_constraint_run(T, case; N=16, roots=1, q=q, t=T(1 // 8))
    @test flat.gauge_l2 < 1e-14
    @test flat.ham_l2 < 1e-12
    @test flat.mom_l2 < 1e-14

    rows = map((8, 12, 16)) do N
        gh_constraint_run(T, case; N=N, roots=2, q=q, t=T(1 // 8), refined=true)
    end
    hs = [r.h for r in rows]
    @test all(r -> r.nblocks == 15, rows)
    for key in (:gauge_l2, :ham_l2, :mom_l2)
        vals = [getfield(r, key) for r in rows]
        rate = convergence_rate(hs, vals)
        @info "constraint across the interface: $key h=$hs vals=$vals rate=$rate"
        @test rate ≥ q - 1 // 4
        @test issorted(vals; rev=true)
        @test vals[1] > 1e-9           # far above the uniform mesh's roundoff
    end
end

@testset "Both monitors converge at order q on a curved background" begin
    # The other half of the claim, and the one the interface row cannot
    # make: on a genuinely curved background the violation is the *bulk*
    # truncation error of the stencils, so the rate is `q` rather than
    # `q + 1/2`, and it is the rate of the derivatives the monitors take
    # rather than of the interpolation at a face. Harmonic Kerr at
    # `a = 9/10` in a box off the origin is `CODE.md`'s proof-of-concept
    # geometry with no hole inside the box — there is no interior treatment
    # until step 5, so the singularity stays outside.
    T = Float64
    q = 4
    case = GHCase(T, Harmonic{T}(1, 9 // 10); box=ntuple(_ -> (T(2), T(4)), 3),
                  periodic=(false, false, false), ε_KO=zero(T), γ0=one(T),
                  γ2=zero(T))
    rows = map((8, 12, 16)) do N
        gh_constraint_run(T, case; N=N, roots=1, q=q)
    end
    hs = [r.h for r in rows]
    for key in (:gauge_l2, :ham_l2, :mom_l2)
        vals = [getfield(r, key) for r in rows]
        rate = convergence_rate(hs, vals)
        @info "constraint on harmonic Kerr: $key h=$hs vals=$vals rate=$rate"
        @test rate ≥ q - 1 // 4
        @test issorted(vals; rev=true)
    end
end

@testset "The masked norms count the evolved points and nothing else" begin
    # Guards the norms' arithmetic and the mask plumbing step 5 depends on.
    # Three claims: under `AllPoints` the norm is the volume-weighted RMS a
    # host loop computes, which is also TreeAMR's `volume_weighted_norm`;
    # a mask that excludes a region writes exact zeros there and divides by
    # the volume that is left; and both are combined in block order, so the
    # answer does not move with the thread count (`threading_tests.jl` is
    # what checks the last claim across processes).
    T = Float64
    q = 4
    case = GHCase(T, Harmonic{T}(1, 9 // 10); box=ntuple(_ -> (T(2), T(4)), 3),
                  periodic=(false, false, false), ε_KO=zero(T), γ0=one(T),
                  γ2=zero(T))
    forest, fs, prob = gh_setup(T, case; N=8, roots=2, q=q)
    fill_exact!(fs, case, zero(T))
    u = statevector(fs)
    gather!(u, fs)

    gh_constraint!(prob, u, zero(T))
    for v in (DIAG_CGH, DIAG_CGH + 1)
        num = zero(T)
        den = zero(T)
        peak = zero(T)
        for b in 1:nblocks(prob.diag)
            cellvolume = spacing(T, forest, blockkey(prob.diag, b))^3
            block = interiorview(prob.diag, b, v)
            num += cellvolume * sum(x -> x * x, block)
            den += cellvolume * length(block)
            peak = max(peak, maximum(abs, block))
        end
        n = masked_norms(prob, v)
        @test n.l2 ≈ sqrt(num / den) rtol = 1e-14
        @test n.linf == peak
        @test n.l2 > 0
    end

    # Masked: the excluded half writes exact zeros, the norm divides by the
    # half that is left, and it is not the unmasked number.
    mid = (case.box[1][1] + case.box[1][2]) / 2
    open = masked_norms(prob, DIAG_CGH)
    gh_constraint!(prob, u, zero(T); mask=HalfSpace(mid))
    closed = masked_norms(prob, DIAG_CGH)
    @test closed.l2 != open.l2
    @test closed.linf ≤ open.linf
    kept = sum(block_mapreduce(identity, +, zero(T), prob.diag; vars=DIAG_MASK))
    total = nblocks(prob.diag) * forest.N^3
    @test 0 < kept < total
    # Every excluded point holds exactly zero, in the constraint slots and
    # in the mask — one assertion over the whole mesh rather than one per
    # point, so that a failure names the claim and not the twelve-thousandth
    # index.
    zeroed = all(1:nblocks(prob.diag)) do b
        mask = interiorview(prob.diag, b, DIAG_MASK)
        gauge = interiorview(prob.diag, b, DIAG_CGH)
        all(CartesianIndices(mask)) do idx
            evolved = coordinates(prob.diag, b, Tuple(idx))[1] ≥ mid
            evolved ? mask[idx] == 1 : mask[idx] == 0 && gauge[idx] == 0
        end
    end
    @test zeroed

    # And `constraint_norms` is those numbers in the shape the record holds
    # them.
    gh_constraint!(prob, u, zero(T))
    rec = constraint_norms(prob)
    @test length(rec.gauge_l2) == 4
    @test length(rec.mom_l2) == 3
    @test rec.gauge_l2[1] == masked_norms(prob, DIAG_CGH).l2
    @test rec.ham_l2 == 0                  # no ADM pass has run into `diag`
    # The slot map, asserted rather than assumed: `block_mapreduce` reduces
    # a *contiguous* range of variables and nothing else, so `DIAG_CGH` and
    # `DIAG_MOM` are the first of a run of four and of three and must stay
    # where they are. Step 5 appended three slots (the masked error, the
    # interior residual and the gauge drift) and step 6 a fourth (the
    # refinement indicator `τ`); neither moved one (amended in steps 5
    # and 6).
    @test (DIAG_SPEED, DIAG_CGH, DIAG_HAM, DIAG_MOM, DIAG_MASK) ==
          (1, 2, 6, 7, 10)
    @test (DIAG_ERR, DIAG_RES, DIAG_DRIFT, DIAG_TAU) == (11, 12, 13, 14)
    @test NDIAG == 14
end
