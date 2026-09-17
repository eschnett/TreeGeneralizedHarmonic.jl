# Coarse-fine faces: what a ghost filled by interpolation costs a scheme
# that takes second derivatives.
#
# `CODE.md`, "The interface-order rule, and what it costs a second-order
# system", and milestone G3. TreeAMR's measured rule is that a ghost
# filled by an order-`p` operator carries an `O(h^p)` error, an `m`-th
# derivative divides it by `h^m`, and the global rate is
# `min(q, p − m + 1)`. This system takes **second** derivatives, so
# `m = 2` and `p = q + 2` is the first prolongation order that does not
# cost the scheme an order: at `q = 4`, rate **3** at `p = 4` and **4** at
# `p = 6`.
#
# That is the whole content of this file, and it is why `ops` is spelled
# out at every call site in this package and never defaulted. The claim is
# asserted as a *bracket* rather than as a floor, because "3 and not 4" is
# the statement: a floor alone would pass if the interface were somehow
# fixed, and a ceiling alone would pass if the scheme had collapsed to
# second order.
#
# The mesh is `gh_forest(…; refined = true)` — TreeAMR's two-level
# `wave_forest`, one root block of a periodic `2³` grid refined once, so
# that every one of its six faces is a coarse-fine face. The hierarchy is
# **frozen**: `N` grows, the block layout does not, every spacing shrinks
# with it, and there is no regridding (`CODE.md`, "Convergence studies on
# a frozen hierarchy"). The unrefined control is the same case on the same
# root grid with no refinement at all.
#
# This is the most expensive file in the suite, and the reason is measured
# below rather than guessed: on the two-level mesh at `p = 6` the ghost
# fill is **79 %** of a right-hand-side evaluation, against 22 % on a
# uniform mesh, because a tensor-product prolongation reads `6³ = 216`
# coarse points per fine ghost point. That cost is TreeAMR's and it is
# what `CODE.md` predicts under "what it costs"; the runs here are as
# short as a clean rate allows.

using SpacetimeMetrics: GaugeWave

# Every row of the table is the gauge wave on the two-level mesh at
# `q = 4`, over three resolutions with the layout held fixed. Half the time
# `convergence_tests.jl` runs on a uniform mesh, and coarser: the interface
# error is there from the first step, and the ghost fill is four times as
# expensive here.
const IFACE_Q = 4
const IFACE_NS = (8, 10, 12)
const IFACE_T_END = 1 // 16

function interface_rates(; p, restriction, ε_KO, refined=true)
    T = Float64
    case = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=T(ε_KO), γ0=one(T),
                           γ2=zero(T))
    ops = Operators(prolongation=p, restriction=restriction)
    results = map(IFACE_NS) do N
        gh_errors(T, case; N=N, roots=2, q=IFACE_Q, t_end=T(IFACE_T_END),
                  ops=ops, refined=refined)
    end
    hs = [r.h for r in results]
    l2 = [r.l2 for r in results]
    linf = [r.linf for r in results]
    return (hs=hs, l2=l2, linf=linf, rate_l2=convergence_rate(hs, l2),
            rate_linf=convergence_rate(hs, linf), nblocks=results[1].nblocks,
            nsteps=[r.nsteps for r in results])
end

@testset "A ghost filled at order p costs a second-derivative scheme an order" begin
    # Guards `CODE.md`'s interface-order rule, which is the reason this
    # package fixes `p = q + 2` and refuses to default `ops`. At `q = 4`
    # the rate is 3 with an order-4 prolongation and 4 with an order-6
    # one, on the same mesh, in the same run, with everything else equal.
    # A change upstream that quietly lowered the operator order — or one
    # here that passed `Operators()` somewhere — shows up as the `p = 6`
    # row falling to 3 and nothing else in the suite noticing.
    rows = (p4=interface_rates(; p=4, restriction=4, ε_KO=0),
            p6=interface_rates(; p=6, restriction=4, ε_KO=0))
    for (name, r) in pairs(rows)
        @info "interface order $name: h=$(r.hs) nsteps=$(r.nsteps) " *
              "blocks=$(r.nblocks) l2=$(r.l2) rate_l2=$(r.rate_l2) " *
              "rate_linf=$(r.rate_linf)"
    end
    # Order lost at p = 4: 3, and neither 4 nor 2.
    @test 2.5 ≤ rows.p4.rate_l2 ≤ 3.6
    @test 2.5 ≤ rows.p4.rate_linf ≤ 3.6
    # Order recovered at p = 6.
    @test rows.p6.rate_l2 ≥ 3.75
    @test rows.p6.rate_linf ≥ 3.75
    # And the errors themselves fall at every step, which a fitted rate
    # alone would not say.
    @test issorted(rows.p4.l2; rev=true)
    @test issorted(rows.p6.l2; rev=true)
    # The two meshes are the same mesh: only the ghost operator differs,
    # so the p = 6 run is strictly the more accurate one at every h.
    @test all(rows.p6.l2 .< rows.p4.l2)
end

@testset "The dissipation does not change what the interface costs" begin
    # Guards the second half of `CODE.md`'s rule: the Kreiss–Oliger term
    # is `h^{2r−1}∂^{2r}`, which contributes `O(h^{p−1})` like a *first*
    # derivative and therefore does not tighten the interface rule. If it
    # did — if the dissipation were scaled by the wrong power of `h`, or
    # read a ghost the prolongation had not filled — the `p = 6` row would
    # drop to `p − 2r + 1` and this is the only place that would say so.
    rows = (p4=interface_rates(; p=4, restriction=4, ε_KO=1 // 2),
            p6=interface_rates(; p=6, restriction=4, ε_KO=1 // 2))
    for (name, r) in pairs(rows)
        @info "interface order with dissipation $name: l2=$(r.l2) " *
              "rate_l2=$(r.rate_l2) rate_linf=$(r.rate_linf)"
    end
    @test 2.5 ≤ rows.p4.rate_l2 ≤ 3.6
    @test 2.5 ≤ rows.p4.rate_linf ≤ 3.6
    @test rows.p6.rate_l2 ≥ 3.75
    @test rows.p6.rate_linf ≥ 3.75
end

@testset "The unrefined control keeps its order at every operator" begin
    # The row that says the rates above are about the *interface* and not
    # about the case, the mesh size or the time: with no coarse-fine face
    # anywhere, the same case on the same root grid converges at 4
    # whatever the ghost operator is, because no ghost is ever
    # interpolated.
    control4 = interface_rates(; p=4, restriction=4, ε_KO=0, refined=false)
    control6 = interface_rates(; p=6, restriction=4, ε_KO=0, refined=false)
    @info "unrefined control: p=4 l2=$(control4.l2) rate=$(control4.rate_l2) " *
          "p=6 l2=$(control6.l2) rate=$(control6.rate_l2)"
    @test control4.rate_l2 ≥ 3.75
    @test control6.rate_l2 ≥ 3.75
    @test control4.nblocks == 8            # 2³ roots, nothing refined
    # An unrefined mesh never prolongates, so the operator order is not
    # merely irrelevant to the rate — it is the same computation.
    @test control4.l2 == control6.l2
end

@testset "The restriction order is not a choice on a vertex-centered mesh" begin
    # Guards the row of `CODE.md`'s table that is a *property of the
    # centering* rather than a measurement: restriction along a stagger is
    # injection — a coincident fine point copied — which is exact for
    # arbitrary data and has no order to raise. TreeWave asserts the same
    # thing the same way, and the claim is stronger than "the rates agree":
    # the two runs are bit-identical, so the rates cannot differ.
    T = Float64
    case = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=T(1 // 2),
                           γ0=one(T), γ2=zero(T))
    for p in (4, 6)
        runs = map((2, 4)) do restriction
            gh_errors(T, case; N=8, roots=2, q=IFACE_Q, t_end=T(IFACE_T_END),
                      ops=Operators(prolongation=p, restriction=restriction),
                      refined=true)
        end
        @test runs[1].l2 === runs[2].l2
        @test runs[1].linf === runs[2].linf
    end
end

@testset "The two-level mesh is the frozen hierarchy the study needs" begin
    # Guards `gh_forest(; refined = true)` itself, and the protocol it
    # exists for: the block layout must not move when `N` does, or a
    # convergence study would be measuring a different mesh at every
    # resolution rather than a smaller `h` on the same one.
    T = Float64
    case = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=zero(T), γ0=one(T),
                           γ2=zero(T))
    forests = [gh_forest(T, case; N=N, roots=2, refined=true) for N in IFACE_NS]
    @test all(f -> nleaves(f) == 15, forests)       # 8 roots, one refined
    @test all(f -> maxlevel(f) == 1, forests)
    @test all(f -> f.leaves == forests[1].leaves, forests)
    @test isbalanced(forests[1])
    # Every spacing halves with N, and the finest is the refined block's.
    hs = [minimum_spacing(T, f) for f in forests]
    @test hs ≈ [one(T) / (2 * 2 * N) for N in IFACE_NS]
    # The unrefined forest over the same box is the control's mesh.
    @test nleaves(gh_forest(T, case; N=8, roots=2)) == 8
end

@testset "A coarse-fine face is what makes an evaluation expensive" begin
    # Not a gate — the measurement `CODE.md` predicts under "what it costs
    # a second-order system", recorded so that a later change to TreeAMR's
    # prolongation shows up as a changed number. A tensor-product
    # prolongation at `p = 6` reads `6³ = 216` coarse points per fine
    # ghost point, and on this mesh that is most of a right-hand side.
    T = Float64
    q = IFACE_Q
    case = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=zero(T), γ0=one(T),
                           γ2=zero(T))
    for (p, refined) in ((6, true), (4, true), (6, false))
        forest, fs, prob = gh_setup(T, case; N=12, roots=2, q=q,
                                    ops=Operators(prolongation=p,
                                                  restriction=p),
                                    refined=refined)
        fill_exact!(fs, case, zero(T))
        u = statevector(fs)
        gather!(u, fs)
        du = similar(u)
        gh_rhs!(du, u, prob, zero(T))
        rhs = Inf
        ghosts = Inf
        for _ in 1:3
            rhs = min(rhs, @elapsed gh_rhs!(du, u, prob, zero(T)))
            ghosts = min(ghosts, @elapsed fill_ghosts!(fs, prob.schedule))
        end
        points = nleaves(forest) * forest.N^3
        @info "interface cost p=$p refined=$refined " *
              "blocks=$(nleaves(forest)) points=$points " *
              "ns_per_point=$(rhs / points * 1e9) " *
              "ghost_fraction=$(ghosts / rhs)"
        @test ghosts < rhs                 # the fill is part of the evaluation
        @test all(isfinite, du)
    end
end
