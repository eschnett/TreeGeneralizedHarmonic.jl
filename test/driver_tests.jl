# The driver on a static black hole: the record a run is judged by, the
# order of the scheme with the layer in place, and `CODE.md`'s three
# interior variants side by side.
#
# `CODE.md`, "Analysis quantities" ("a run is judged by what it records,
# not by finishing") and "Refinement and regridding" (the loop). Every
# number here is masked inside `r_1`: the damping layer and the frozen core
# are not a numerical solution and are not reported as one.
#
# **What this file costs, and why.** It is the suite's only black hole,
# and a hole is expensive for a reason that is physics and not
# implementation: `CODE.md`'s two radius requirements together need
# `r_h,min ≥ (m + 2G + 2)·h`, so the spacing is set by the horizon's
# smallest coordinate radius. Kerr-Schild (`r₊ = 2 M`) is the cheapest of
# the three holes and is what the suite runs; the harmonic charts, the
# default margin `m = 8`, `q = 4` and the long runs are in the standalone
# `test/hole_runs.jl`, whose numbers are recorded in `CODE.md`.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using KernelAbstractions: CPU

@testset verbose = true "The driver and the static hole" begin
    T = Float64
    q = 2

    # `CODE.md`, "The time step": throw, do not warn, if the step used
    # violated the bound. A pure function, so it can be exercised on
    # synthetic numbers rather than by provoking an instability.
    @testset "the CFL recheck throws, and is exact at equality" begin
        @test check_cfl(0.1, 1.0, 0.25, 2.0) ≈ 0.2
        @test check_cfl(0.125, 1.0, 0.25, 2.0) ≈ 0.25      # equality holds
        @test_throws ArgumentError check_cfl(0.2, 1.0, 0.25, 2.0)
        @test_throws ArgumentError check_cfl(0.125, 1.0, 0.25, 2.1;
                                             chunk=3, λ=2.0)
    end

    # The keywords that are refused are refused by name, so that a caller
    # is told what they asked for and why it is not on offer. (The refusal
    # of `regrid` and `adapt` on a case with no refinement parameters is
    # `refinement_tests.jl`'s, since that is where the parameters are.)
    @testset "the driver refuses what a case cannot mean" begin
        case = hole_fixture(T; q=q)
        forest = hole_fixture_forest(T, case; N=8)
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        @test_throws ArgumentError evolve!(Float32, case; forest=forest, q=q,
                                           ops=ops, t_end=T(1 // 10))
        @test_throws ArgumentError evolve!(T, case; forest=forest, q=q, ops=ops,
                                           t_end=T(1 // 10), chunk=zero(T))
        @test_throws ArgumentError evolve!(T, case; forest=forest, q=q, ops=ops,
                                           t_end=-one(T))
    end

    # Step 8c's fixed relaxation rate: the physical alternative to `1/dt`,
    # which the record must report as the rate the chunk ran at — and the
    # two ways of saying the rate are one question, so saying both is
    # refused rather than silently resolved in favour of one; a fixed rate
    # above the grid rate is refused at the chunk that would take it.
    @testset "a fixed ρ_max is the rate the record reports" begin
        case = hole_fixture(T; q=q, chunk=T(1 // 40))
        out = gh_hole_run(T, case; N=8, q=q, t_end=T(1 // 20),
                          ρ_max_fixed=T(4))
        @test length(out.records) == 3
        for r in out.records
            @test r.ρ_max === 4.0
            @test r.finite
        end
        @test out.interior.ρ_max === T(4)
        forest = hole_fixture_forest(T, case; N=8)
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        @test_throws "both ρ_max_factor" evolve!(T, case; forest=forest, q=q,
                                                 ops=ops, t_end=T(1 // 20),
                                                 ρ_max_factor=one(T),
                                                 ρ_max_fixed=T(4))
        @test_throws "above this chunk's grid rate" evolve!(
            T, case; forest=forest, q=q, ops=ops, t_end=T(1 // 20),
            ρ_max_fixed=T(1000))
        @test_throws ArgumentError evolve!(T, case; forest=forest, q=q, ops=ops,
                                           t_end=T(1 // 20),
                                           ρ_max_fixed=zero(T))
    end

    # Step 8c′: the default is the physical `4/M` (asserted on the record
    # below) and the grid rate `1/dt` — the default until 2026-09-23 — is
    # the option `ρ_max_factor`. A driver that ignored the factor, or still
    # derived the default from `dt`, would run a different layer from the
    # one its record names, and step 5's numbers, which `CODE.md` keeps as
    # history, would no longer be reproducible by asking for them. The
    # default's refusal is the mesh's and says so; a case with no hole has
    # no mass to read and no rate to set, and is run as it always was.
    @testset "ρ_max_factor is the grid rate, and a case with no hole is untouched" begin
        case = hole_fixture(T; q=q, chunk=T(1 // 40))
        out = gh_hole_run(T, case; N=8, q=q, t_end=T(1 // 20),
                          ρ_max_factor=one(T))
        @test length(out.records) == 3
        # The `t = 0` row has taken no step (its `dt` is zero), so the rate
        # it reports is `1/dt` of the step the first chunk was sized from.
        @test out.records[1].ρ_max > 4 / hole_mass(case.background)
        for r in out.records[2:end]
            @test r.ρ_max * r.dt ≈ 1
            @test r.finite
        end
        @test out.interior.ρ_max * T(out.records[end].dt) ≈ 1
        @test_throws "default relaxation rate" TreeGeneralizedHarmonic.chunk_interior(
            case, T(1 // 2), nothing, default_relaxation_rate(case); default=true)
        flat = minkowski_case(T; L=one(T), ε_KO=T(1 // 2), γ0=one(T),
                              γ2=T(-1 // 2))
        @test_throws "no hole mass" hole_mass(flat.background)
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        fout = evolve!(T, flat; forest=gh_forest(T, flat; N=8, roots=2), q=q,
                       ops=ops, t_end=T(1 // 20), chunk=T(1 // 20))
        @test fout.interior === nothing
        @test all(r -> r.ρ_max == 0 && r.finite, fout.records)
    end

    # A run that finishes without the analysis quantities is not a result
    # (`CODE.md`, "Analysis quantities"), so the first claim about the
    # driver is about its record and not about its state.
    @testset "the record holds every quantity CODE.md asks of this step" begin
        case = hole_fixture(T; q=q, chunk=T(1 // 20))
        out = gh_hole_run(T, case; N=8, q=q, t_end=T(3 // 20), adm_every=1)
        @test length(out.records) == 4              # t = 0 and three chunks
        @test out.nchunks == 3
        @test out.nsteps > 0
        first_r = out.records[1]
        last_r = out.records[end]
        @test first_r.t == 0
        @test last_r.t ≈ T(3 // 20)
        for r in out.records
            for k in (:t, :dt, :λ, :λ_end, :gauge_l2, :gauge_linf, :err_l2,
                      :err_linf, :residual, :drift, :ρ_max, :h, :nblocks,
                      :levels, :ham_l2, :ham_linf, :mom_l2, :mom_linf)
                @test haskey(r, k)
            end
            @test r.finite
            @test isfinite(r.gauge_l2) && isfinite(r.err_l2)
            @test isfinite(r.ham_l2) && isfinite(r.mom_l2)
            @test r.nblocks == 120
            @test r.levels == [0, 0, 56, 64]
            @test r.h ≈ minimum_spacing(T, out.forest)
            # The horizon rows exist on every record and are empty here,
            # so the record's *shape* does not depend on the case: this
            # fixture carries no `Horizon` (added in step 7, whose own
            # file measures them where it does).
            for k in (:horizon_success, :origin, :r_min, :r_mean, :r_max,
                      :area, :M_irr, :J, :spin_axis, :M_ch, :hlm)
                @test haskey(r, k)
                @test getfield(r, k) === nothing
            end
        end
        # The initial data is exact, so the error starts at roundoff and
        # grows; the constraints do not start at zero, because a
        # finite-difference solution of an exact metric has a
        # truncation-order violation.
        #
        # Roundoff and not `== 0`: `gh_error_kernel!` samples the reference
        # through `case_state_tuple` at a *different call site* than the
        # initial data used, and two call sites of one function are not
        # bit-identical (`CODE.md`, "Testing"). The equality held on aarch64
        # and on x86-64 uninstrumented, and failed on x86-64 at 1.13 under
        # code coverage, which perturbs the inlining that decides whether a
        # multiply and an add are fused -- measured 3.2e-17, 1.4e-15 and
        # 1.2e-13 there. A claim that turns on the CPU target and the
        # compiler's flags is not a claim about this package.
        #
        # The residual gets the looser bound because it is the only one of
        # the three read *inside* `r_1`, where the solution is steepest and
        # its roundoff largest; the other two are masked to `r ≥ r_1`.
        @test first_r.err_l2 ≤ 100 * eps(T)
        @test first_r.err_linf ≤ 100 * eps(T)
        @test first_r.residual ≤ 10_000 * eps(T)
        @test last_r.err_l2 > 0
        @test issorted([r.err_l2 for r in out.records])
        @test first_r.gauge_l2 > 0
        # `CODE.md`: the layer relaxes at `4/M` in every chunk, read from
        # the hole's mass, and the record reports it on every row, `t = 0`
        # included. **Amended in step 8c′, not loosened**: until 2026-09-23
        # the default was the grid rate and this claim was
        # `r.ρ_max * r.dt ≈ 1`; Erik's decision made `4/M` the default, and
        # the grid rate's own claim is the testset above.
        for r in out.records
            @test r.ρ_max == 4 / hole_mass(case.background)
        end
        for r in out.records[2:end]
            @test r.cfl ≤ T(1 // 4) * (1 + 1e-12)
            @test r.dt ≈ (T(1 // 20)) / r.steps
        end
        # The observer sees the same times, before anything is invalidated.
        seen = T[]
        gh_hole_run(T, case; N=8, q=q, t_end=T(1 // 10),
                    observer=(p, t, u) -> push!(seen, T(t)))
        @test seen ≈ T[0, 1 // 20, 1 // 10]
    end

    # `CODE.md`'s frozen-hierarchy protocol: the block layout is held fixed
    # and `N` is raised, so every spacing shrinks and nothing else moves.
    # The claim is the *masked* error — outside `r_1`, where the equations
    # are the Einstein equations and nothing else.
    @testset "the masked error converges at order q on the frozen hierarchy" begin
        case = hole_fixture(T; q=q)
        Ns = (6, 8, 10)
        hs = T[]
        l2 = T[]
        linf = T[]
        gauge = T[]
        res = T[]
        for N in Ns
            out = gh_hole_run(T, case; N=N, q=q, t_end=T(3 // 20))
            r = out.records[end]
            push!(hs, T(out.h))
            push!(l2, T(r.err_l2))
            push!(linf, T(r.err_linf))
            push!(gauge, T(r.gauge_l2))
            push!(res, T(r.residual))
            @test r.finite
        end
        rate_l2 = convergence_rate(hs, l2)
        rate_linf = convergence_rate(hs, linf)
        rate_gauge = convergence_rate(hs, gauge)
        rate_res = convergence_rate(hs, res)
        # Printed as strings, because `@info` abbreviates a vector to
        # "3-element Vector{Float64}: …" and a CI log has to carry the
        # numbers `CODE.md` records.
        @info("static hole, masked error on the frozen hierarchy", q, Ns,
              hs=string(hs), l2=string(l2), linf=string(linf),
              gauge=string(gauge), res=string(res), rate_l2, rate_linf,
              rate_gauge, rate_res)
        # Order `q`, with the slack step 3 measured its own rates to.
        @test rate_l2 > q - T(1 // 4)
        @test rate_linf > q - T(1 // 4)
        # The constraints are flat at truncation and converge with the
        # scheme — masked, which is the whole point.
        @test rate_gauge > q - T(1 // 4)
        # And the layer's own residual converges too: the relaxation holds
        # the layer at the analytic solution to truncation order.
        @test rate_res > q - T(1 // 4)
        # The unmasked error would be dominated by the interior, which is
        # exactly why every norm is masked.
        out = gh_hole_run(T, case; N=8, q=q, t_end=T(3 // 20))
        p = out.problem
        gh_error!(p, out.u, T(3 // 20); mask=AllPoints())
        unmasked = error_norms(p)
        gh_error!(p, out.u, T(3 // 20))
        masked = error_norms(p)
        @info "masked against unmasked" masked unmasked
        @test unmasked.err_linf > masked.err_linf
    end

    # `CODE.md` measures all three variants on the static hole and asks
    # for their constraint norms in the `G` points outside `r_1` — the only
    # points at which they can differ before the difference propagates —
    # and for the default to be confirmed or changed from them.
    #
    # **Two times, since step 8c′.** The shell is read at `t = 1/10 M`, as
    # step 5 read it, and the residual after two relaxation times of the
    # default rate, `2/ρ_max = 1/2 M`. At the grid rate the sink relaxed in
    # one step and `:damped`'s residual had saturated by the first chunk, so
    # one short run showed both; at the default `4/M` it relaxes in `M/4`,
    # and at `1/10 M` it has not — `:frozen`'s residual is then only 1.2
    # times `:damped`'s, the sticky wall and the sink not yet told apart. By
    # `1/2 M` `:damped` has saturated and `:frozen` is still growing
    # linearly (`CODE.md`, "Measured results", step 8c′). The shell is not read there as well because `:pasted`'s
    # kink has by then grown it past the other two at *either* rate — the
    # surface failure step 5 measured at `17 M`, starting — which is a claim
    # about the paste and not the one this testset makes. `:pasted`'s
    # residual is zero at every time, so it stops at the shell's time.
    @testset "the three interior variants, in the G points outside r_1" begin
        t_shell = T(1 // 10)
        t_res = 2 / default_relaxation_rate(hole_fixture(T; q=q))
        rows = NamedTuple[]
        for variant in (:damped, :pasted, :frozen)
            case = hole_fixture(T; q=q, variant=variant, chunk=T(1 // 20))
            at_shell = Ref{Any}(nothing)
            watch(p, t, u) = isapprox(t, t_shell) &&
                             (at_shell[] = gh_outside_shell_norms(p, u, t))
            out = gh_hole_run(T, case; N=8, q=q, observer=watch,
                              t_end=variant === :pasted ? t_shell : t_res)
            shell = at_shell[]
            r = out.records[end]
            push!(rows,
                  (variant=variant, shell_gauge_l2=shell.gauge_l2,
                   shell_gauge_linf=shell.gauge_linf,
                   shell_err_l2=shell.err_l2, shell_err_linf=shell.err_linf,
                   npoints=shell.npoints, t=r.t, residual=r.residual,
                   err_l2=r.err_l2, gauge_l2=r.gauge_l2, finite=r.finite))
            @test all(row -> row.finite, out.records)
            @test shell.npoints > 0
        end
        @info "the three interior variants on the static hole" rows
        damped = rows[1]
        pasted = rows[2]
        frozen = rows[3]
        # `:pasted` holds the layer at the analytic solution *exactly* —
        # that is what the limiter does — and `:damped` to truncation.
        @test pasted.residual == 0
        @test damped.residual > 0
        # `CODE.md`'s prediction: `:frozen` piles perturbations up against
        # the freezing radius instead of draining them, so its residual
        # grows where `:damped`'s saturates, and it is the reason `:damped`
        # is the default. **Re-measured in step 8c′ at `t = 1/2 M` rather
        # than `1/10 M`**, for the reason above: the claim is unchanged.
        @test damped.t ≈ t_res && frozen.t ≈ t_res
        @test frozen.residual > 2 * damped.residual
        # All three keep the evolved region at the same level: the interior
        # treatment is invisible outside `r_1` to truncation order.
        @test damped.shell_gauge_l2 ≈ pasted.shell_gauge_l2 rtol = 1 // 4
        @test damped.shell_gauge_l2 ≈ frozen.shell_gauge_l2 rtol = 1 // 4
    end

    # GHSO2 measured a slow, constraint-preserving gauge drift of the
    # excised hole under prescribed sources, `≈ 0.14/M`, and `CODE.md`
    # predicts exact interior and boundary data lower it without removing
    # it. What a run of this length can say is the rate it sees.
    @testset "the gauge drift at the horizon is measured" begin
        case = hole_fixture(T; q=q, chunk=T(1 // 20))
        out = gh_hole_run(T, case; N=8, q=q, t_end=T(1 // 5))
        ts = [T(r.t) for r in out.records]
        ds = [T(r.drift) for r in out.records]
        @test ds[1] == 0
        @test issorted(ds)
        # Least squares through the origin: the drift starts at zero
        # because the initial data is exact.
        rate = sum(ts .* ds) / sum(ts .^ 2)
        @info("gauge drift of h_tt in the horizon shell",
              shell=horizon_shell(case), ts=string(ts), ds=string(ds), rate)
        @test rate > 0
        @test isfinite(rate)
    end

    # `CODE.md` keeps GHSO2's discrete-gradient `Π` as a post-pass option
    # and asks G4 whether it changes a hole's stationarity visibly
    # **(predicted: not beyond the first chunk)**.
    @testset "the discrete-gradient Π post-pass is measured" begin
        case = hole_fixture(T; q=q)
        G = q ÷ 2 + 1
        forest = hole_fixture_forest(T, case; N=8)
        ops = Operators(prolongation=q + 2, restriction=q + 2)
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                         backend=CPU())
        sched = GhostSchedule(fs, ops)
        p = GHProblem(fs, sched, case; q=q,
                      interior=with_ρ_max(case.interior, T(10)))
        fill_exact!(fs, case, zero(T))
        u_analytic = statevector(fs)
        gather!(u_analytic, fs)
        discrete_gradient_momentum!(fs, case, zero(T), q, sched)
        u_discrete = statevector(fs)
        gather!(u_discrete, fs)
        @test all(isfinite, u_discrete)
        @test u_discrete != u_analytic
        # `h` is untouched; only `Π` changes, and by a truncation-order
        # amount.
        A_a = statearray(u_analytic, fs)
        A_d = statearray(u_discrete, fs)
        @test A_a[:, :, :, 1:10, :] == A_d[:, :, :, 1:10, :]
        # The stationarity the two buy: the worst `|du|` of the exact state
        # over the **evolved** region, split into the two evolution
        # equations, because they answer differently.
        function stationarity(u, p)
            du = similar(u)
            gh_rhs!(du, u, p, zero(T))
            A = statearray(du, fs)
            wh = zero(T)
            wΠ = zero(T)
            for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N,
                i in 1:forest.N

                x = coordinates(fs, b, (i + G, j + G, k + G))
                sqrt(sum(abs2, x)) ≥ case.interior.r_1 || continue
                wh = max(wh, maximum(abs, A[i, j, k, 1:10, b]))
                wΠ = max(wΠ, maximum(abs, A[i, j, k, 11:20, b]))
            end
            return (h=wh, Π=wΠ)
        end
        s_a = stationarity(u_analytic, p)
        s_d = stationarity(u_discrete, p)
        @info("the discrete-gradient Π post-pass (ε_KO = 1/2)",
              analytic=s_a, discrete=s_d)
        @test isfinite(s_a.h) && isfinite(s_d.h)
        @test s_d.Π > 0
        # The sharp statement, and the one GHSO2 makes: inverting the first
        # evolution equation for `Π` makes that equation's residual
        # **roundoff** on a static background, because the discrete `D_i h`
        # the post-pass subtracts is the same operator the kernel adds
        # back. It is exactly true only with the dissipation off — the
        # Kreiss–Oliger term is `O(h^{q+1})` and is not part of the
        # inversion — so the claim is made there.
        nodiss = hole_fixture(T; q=q, ε_KO=zero(T))
        p0 = GHProblem(fs, sched, nodiss; q=q,
                       interior=with_ρ_max(nodiss.interior, T(10)))
        fill_exact!(fs, nodiss, zero(T))
        ua = statevector(fs)
        gather!(ua, fs)
        discrete_gradient_momentum!(fs, nodiss, zero(T), q, sched)
        ud = statevector(fs)
        gather!(ud, fs)
        n_a = stationarity(ua, p0)
        n_d = stationarity(ud, p0)
        @info("the discrete-gradient Π post-pass (ε_KO = 0)",
              analytic=n_a, discrete=n_d)
        @test n_d.h < 1e-12
        @test n_a.h > 1e-6
        # `∂_tΠ` is the reduced equation and the post-pass does not touch
        # it, which is why the *whole* right-hand side does not fall with
        # it — `CODE.md`'s "not beyond the first chunk".
        @test n_d.Π > n_d.h
    end
end
