# The static-hole runs that do not fit a test file — run by hand, once,
# with their numbers recorded in `CODE.md`.
#
#     julia --project=. --threads=4 test/hole_runs.jl
#
# `PLAN.md`'s step 5 asks for the masked error to converge at order `q` on
# the frozen hierarchy "to `t = 50 M` at the coarser resolutions and to a
# few `M` in the suite", and the suite is where "a few `M`" has to fit
# `PLAN.md`'s rule of thumb of 30 seconds per file. What is here is
# everything that does not: the default margin `m = 8`, `q = 4`, the
# `t = 50 M` run of all three interior variants, the gauge drift measured
# over a time long enough to separate it from truncation, and the two
# harmonic charts — `a = 0`, whose horizon radius is `M`, and `a = 9/10`,
# whose `0.44 M` is what `CODE.md` names as the thing that sets the finest
# spacing this package must reach.
#
# It is a **script and not a test**: it prints a table, and where a row
# ends in a state that is no longer a metric it prints *that* and the time
# it reached rather than throwing, because a table wants the failure as a
# row. It loads no `Test`. The claims it supports are in `CODE.md` under
# "Measured results", beside the predictions they confirm or correct, and
# the suite's own hole runs (`test/driver_tests.jl`) are the regression.
#
# Each row names its own configuration, because the resolution a hole
# needs is not a free choice: `CODE.md`'s two radius requirements together
# are `r_h,min ≥ (m + 2G + 2)·h + r_0`, so a chart with a smaller
# `r_h,min` needs a proportionally finer mesh and nothing else changes.
#
# Step 6 adds the `indicator` section: the calibration `CODE.md` asks the
# thresholds to be chosen from — `τ_max` against `h` on uniform meshes, and
# the depth the initial-data cycle reaches against `refine_tol` — and the
# adaptive run at the chosen thresholds against a frozen hierarchy of the
# same finest spacing. It is here rather than in the suite because at the
# calibrated thresholds this hole asks for 848 blocks.

import Printf
using TreeAMR
using TreeGeneralizedHarmonic
using KernelAbstractions: CPU
using StaticArrays: SVector
import SpacetimeMetrics as SM

include(joinpath(@__DIR__, "evolution_cases.jl"))

const T = Float64

# `Printf.@printf` insists on a *literal* format string, which at this
# file's line width would mean 200-column lines. `Printf.Format` takes the
# string at run time instead, so the formats below can be written the way
# everything else in this package is.
say(fmt, args...) = println(Printf.format(Printf.Format(fmt), args...))

# Which sections to run; all of them by default.
const SECTIONS = isempty(ARGS) ?
                 ["order", "long", "charts", "indicator", "horizon"] : ARGS

"""
The frozen hierarchy: one shell radius per refinement level. A radius of
`10` catches every block of that level (a root block and its children all
touch the center), and the last one or two are narrowed so that the sphere
`r_1` lies wholly inside the finest level while the outer part of the box
stays coarser — which is what puts a coarse-fine face in the mesh a hole
is measured on, and what keeps the block count down.
"""
shells(case, N, radii) = hole_forest(T, case; N=N, roots=1, radii=radii)

"""
One row, with the failure reported rather than thrown. A long run of a
hole on a coarse mesh **can** end in a degenerate metric — `√(det γ)` of a
state that is no longer a metric — and when it does, what a table wants is
the time it reached and the reason, not a stack trace and no other rows.
The observer records how far it got.
"""
function row(case; N, q, t_end, radii, label, kwargs...)
    forest = shells(case, N, radii)
    t0 = time()
    reached = Ref(zero(T))
    watch(p, t, u) = (reached[] = T(t))
    out = try
        evolve!(T, case; forest=forest, q=q,
                ops=Operators(prolongation=q + 2, restriction=q + 2),
                t_end=T(t_end), observer=watch, kwargs...)
    catch e
        msg = sprint(showerror, e)
        say("%-26s q=%d N=%2d **FAILED** after t = %.4f M: %s", label, q, N,
            reached[], first(split(msg, '\n')))
        return nothing
    end
    r = out.records[end]
    wall = time() - t0
    say("%-26s q=%d N=%2d h=%8.5f blocks=%4d steps=%5d  " *
            "err_l2=%10.3e err_inf=%10.3e gauge_l2=%10.3e res=%10.3e " *
            "drift=%10.3e  finite=%s  %6.1fs",
            label, q, N, out.h, out.nblocks, out.nsteps, r.err_l2, r.err_linf,
            r.gauge_l2, r.residual, r.drift, r.finite, wall)
    return out
end

# --- (1) order at q = 4 with the default margin m = 8 -----------------------
#
# The suite runs the default `m = 8` too, but at `q = 2`; here `G = 3`, so
# `r_h,min ≥ (m + 2G + 2)h` is `h ≤ (2 − r_0)/16` and the mesh is finer.
if "order" in SECTIONS
    println("\n=== (1) q = 4, margin m = 8, Kerr-Schild a = 0 ===")
    q = 4
    case = kerr_schild_case(T; M=1, a=0, halfwidth=T(5 // 2), r_0=T(2 // 5),
                            r_1=T(23 // 20), chunk=T(1 // 10), margin=8)
    hs = T[]
    l2 = T[]
    li = T[]
    gl = T[]
    rs = T[]
    for N in (8, 10, 12)
        out = row(case; N=N, q=q, t_end=T(1 // 4),
                  radii=(T(10), T(10), one(T)), label="KS a=0, m=8")
        out === nothing && error("the q = 4 order sweep needs every row")
        r = out.records[end]
        push!(hs, T(out.h))
        push!(l2, T(r.err_l2))
        push!(li, T(r.err_linf))
        push!(gl, T(r.gauge_l2))
        push!(rs, T(r.residual))
    end
    say("rates: L2 %.3f  Linf %.3f  gauge %.3f  residual %.3f",
            convergence_rate(hs, l2), convergence_rate(hs, li),
            convergence_rate(hs, gl), convergence_rate(hs, rs))

    # The layer's share of an evaluation at `q = 4`: `CODE.md` prices it at
    # "a few percent of an RHS" and asks G4 to measure it.
    forest = shells(case, 12, (T(10), T(10), one(T)))
    G = q ÷ 2 + 1
    fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                     backend=CPU())
    sched = GhostSchedule(fs, Operators(prolongation=q + 2, restriction=q + 2))
    p = GHProblem(fs, sched, case; q=q,
                  interior=with_ρ_max(case.interior, T(10)))
    p0 = with_interior(p, nothing)
    fill_exact!(fs, case, zero(T))
    u = statevector(fs)
    gather!(u, fs)
    du = similar(u)
    npts = nblocks(fs) * forest.N^3
    local ncore = 0
    local nlayer = 0
    for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N, i in 1:forest.N
        x = coordinates(fs, b, (i + G, j + G, k + G))
        r = sqrt(sum(abs2, x))
        r < case.interior.r_0 ? (ncore += 1) :
        r < case.interior.r_1 ? (nlayer += 1) : nothing
    end
    say("  points: %d total, %d in the layer (%.1f %%), %d in the core",
        npts, nlayer, 100 * nlayer / npts, ncore)
    best = Dict{String,Float64}()
    for (name, pp) in (("without", p0), ("with", p))
        gh_rhs!(du, u, pp, zero(T))
        t = minimum(1:5) do _
            t0 = time_ns()
            gh_rhs!(du, u, pp, zero(T))
            (time_ns() - t0) / 1e9
        end
        best[name] = t
        say("  RHS %-8s the interior: %8.4f s  (%6.0f ns/point)", name, t,
            t * 1e9 / npts)
    end
    say("  the layer's share: %.1f %% of an evaluation, %.0f ns per layer " *
        "point", 100 * (best["with"] - best["without"]) / best["without"],
        (best["with"] - best["without"]) * 1e9 / max(nlayer, 1))
end

# --- (2) t = 50 M, all three interior variants ------------------------------
#
# `CODE.md`'s prediction: `:damped` and `:pasted` both hold the static hole
# to `t = 50 M` with constraints at truncation outside `r_1`, `:damped`
# with the smaller violation in the `G` points outside `r_1`; `:frozen`
# holds it only with `ε_KO ≈ 0.5` and a wide ramp, with a growing layer of
# compressed features at the freezing radius.
if "long" in SECTIONS
    println("\n=== (2) t = 50 M, the three variants, q = 2 ===")
    for (N, variant) in ((6, :damped), (8, :damped), (8, :pasted), (8, :frozen))
        # `cfl = 1/5` and not the default `1/4`: at `1/4` the end-of-chunk
        # recheck **fires** at `t = 25 M` on this mesh, which is the check
        # doing its job and is recorded in `CODE.md` — `λ_max` climbs from
        # `1.6710` toward and past flat space's `√3 = 1.7321` as the
        # solution settles, and a step sized at exactly `cfl = 1/4` from the
        # chunk's opening value has no room for it. Shortening the chunk is
        # the other remedy the message names; lowering `cfl` is the one that
        # holds for fifty crossings.
        local case = kerr_schild_case(T; M=1, a=0, halfwidth=T(5 // 2),
                                      r_0=T(2 // 5), r_1=T(23 // 20),
                                      chunk=one(T), margin=8,
                                      interior=variant)
        out = row(case; N=N, q=2, t_end=T(50), cfl=T(1 // 5),
                  radii=(T(10), T(10), one(T)), label="KS a=0 $variant")
        out === nothing && continue
        shell = gh_outside_shell_norms(out)
        ts = [T(r.t) for r in out.records]
        ds = [T(r.drift) for r in out.records]
        rate = sum(ts .* ds) / sum(ts .^ 2)
        say("    shell (G points outside r_1): gauge_l2=%10.3e " *
                "gauge_linf=%10.3e err_l2=%10.3e  npoints=%d",
                shell.gauge_l2, shell.gauge_linf, shell.err_l2,
                round(Int, shell.npoints))
        say("    gauge drift of h_tt at the horizon: %.4e / M " *
            "(GHSO2: 0.14/M)", rate)
        println("    residual every 10 M: ",
                join((round(out.records[i].residual; sigdigits=4)
                      for i in 1:10:length(out.records)), " "))
        println("    masked err_l2 every 10 M: ",
                join((round(out.records[i].err_l2; sigdigits=4)
                      for i in 1:10:length(out.records)), " "))
        println("    masked gauge_l2 every 10 M: ",
                join((round(out.records[i].gauge_l2; sigdigits=4)
                      for i in 1:10:length(out.records)), " "))
        println("    λ every 10 M: ",
                join((round(out.records[i].λ; sigdigits=6)
                      for i in 1:10:length(out.records)), " "))
    end
end

# --- (3) the other charts, at the spacing each of them needs --------------
#
# `CODE.md`: `r_h,min` is `M` for harmonic Kerr at `a = 0` and about
# `0.44 M` at `a = 9/10`, and "these bounds, not the exterior, set the
# finest spacing the refinement must reach". They do — and at `a = 9/10`
# the harmonic chart is refused for a different reason, which this section
# records rather than works around: the chart's singular disk has
# coordinate radius `a = 0.9` and the horizon's smallest coordinate radius
# is `0.436`, so no *ball* contains the one and fits inside the other. The
# spinning hole therefore runs here in **Kerr-Schild**, where `r₊ = 1.436`
# leaves room for a core of radius `0.95`.
if "charts" in SECTIONS
    println("\n=== (3) the other charts ===")
    # The refusal, first, because it is the finding.
    let bad = harmonic_kerr_case(T; M=1, a=T(9 // 10), halfwidth=T(5 // 2),
                                 r_0=T(1 // 20), r_1=T(1 // 4),
                                 chunk=T(1 // 10), margin=4)
        f = hole_forest(T, bad; N=8, roots=1,
                        radii=(T(10), T(10), T(1 // 2), T(3 // 10),
                               T(3 // 10)))
        try
            evolve!(T, bad; forest=f, q=2,
                    ops=Operators(prolongation=4, restriction=4),
                    t_end=T(1 // 10))
            println("harmonic a=0.9: UNEXPECTEDLY RAN")
        catch e
            println("harmonic a=0.9 is refused: ",
                    first(split(sprint(showerror, e), '\n')))
        end
    end
    for (bg, r_0, r_1, N, radii) in
        ((SM.Harmonic(one(T), T(0)), T(1 // 5), T(67 // 100), 8,
          (T(10), T(10), one(T))),
         (SM.KerrSchild(one(T), T(9 // 10)), T(19 // 20), T(5 // 4), 8,
          (T(10), T(10), T(8 // 5), T(5 // 4))))
        local a = bg isa SM.Harmonic ? bg.spin : bg.spin
        say("%s a=%.2f  r_h,min=%.4f  r_h,max=%.4f  r_sing=%.3f  " *
            "r_0=%.3f r_1=%.3f", string(nameof(typeof(bg))), a,
            horizon_min_radius(bg), horizon_max_radius(bg),
            singular_radius(bg), r_0, r_1)
        local case = hole_case(T, bg; M=one(T), halfwidth=T(5 // 2), r_0=r_0,
                               r_1=r_1, chunk=T(1 // 10), margin=4)
        local forest = shells(case, N, radii)
        local t0 = time()
        local out = evolve!(T, case; forest=forest, q=2,
                            ops=Operators(prolongation=4, restriction=4),
                            t_end=T(1 // 5))
        local r = out.records[end]
        say("  q=2 N=%2d h=%8.5f blocks=%4d steps=%5d err_l2=%10.3e " *
                "gauge_l2=%10.3e res=%10.3e finite=%s %6.1fs",
                N, out.h, out.nblocks, out.nsteps, r.err_l2, r.gauge_l2,
                r.residual, r.finite, time() - t0)
        h, nb = layer_spacing(out.forest, out.interior, zero(T))
        say("    layer spacing h=%.5f over %d blocks", h, nb)
        r.finite || error("$(nameof(typeof(bg))) a=$a went non-finite")
    end
end

# --- (4) the refinement indicator: calibration, and the adaptive run -------
#
# `CODE.md`, "Refinement and regridding": "Thresholds are calibrated as
# TreeWave calibrates them: `τ_max` on uniform meshes at successive `h` on
# the static hole, tabulated, thresholds chosen mid-plateau". Two tables,
# TreeWave's two, and then the run they justify.
#
# The reference configuration is `adaptive_hole_fixture`'s and is the one
# `CODE.md` records the numbers for: Kerr-Schild `a = 0` in a box of
# half-width `5 M` on a `4³` root brick with `N = 8`, `r_0 = 3/10`,
# `r_1 = 5/4`, margin `m = 4`. The box is four times the suite's step-5 one
# because the ceiling and the floor need room between them: with the
# horizon at `r = 2` and a box of half-width `5/2`, the shell that must be
# refined and the shell that must stay coarse overlap, and
# `block_level_bounds` says so.
if "indicator" in SECTIONS
    println("\n=== (4) the refinement indicator ===")
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    ref_case(; kwargs...) = adaptive_hole_fixture(T; kwargs...)

    println("\n-- τ_max against h on uniform meshes (masked, N = 8) --")
    for roots in (2, 4, 8, 16)
        case = ref_case(; maxlevel_cap=1)
        pass = gh_tau_pass(T, case; N=8, roots=roots, q=q)
        h = minimum_spacing(T, pass.forest)
        say("  roots=%2d h=%8.5f blocks=%5d points=%9d  τ_max=%.4f  " *
            "U_ref=%.4f", roots, h, nleaves(pass.forest),
            nleaves(pass.forest) * 8^3, pass.τ_max, pass.scale)
        println("     per-component U_ref: ",
                join((Printf.format(Printf.Format("%.4f"), s)
                      for s in pass.scales), " "))
    end

    println("\n-- the depth the initial-data cycle reaches, against " *
            "refine_tol (cap 3) --")
    for rt in (T(4 // 5), T(3 // 5), T(1 // 2), T(2 // 5), T(3 // 10),
               T(1 // 5))
        case = ref_case(; refine_tol=rt, coarsen_tol=rt / 4, maxlevel_cap=3)
        t0 = time()
        cyc = gh_adapt_cycle(T, case; N=8, roots=4, q=q)
        say("  refine_tol=%.2f coarsen_tol=%.3f → passes=%d converged=%s " *
            "depth=%d leaves=%4d points=%8d h=%8.5f τ_max=%.4f %.1fs",
            rt, rt / 4, cyc.passes, cyc.converged, maxlevel(cyc.forest),
            nleaves(cyc.forest), nleaves(cyc.forest) * 8^3,
            minimum_spacing(T, cyc.forest), cyc.τ_max, time() - t0)
        println("     levels ", forest_levels(cyc.forest), "  centroid ",
                cyc.centroid === nothing ? nothing :
                round.(cyc.centroid; digits=4), "  offset/h ",
                cyc.centroid === nothing ? nothing :
                round(sqrt(sum(abs2, cyc.centroid)) /
                      minimum_spacing(T, cyc.forest); digits=2))
    end

    println("\n-- the adaptive run at the calibrated thresholds, against a " *
            "frozen hierarchy of the same finest spacing --")
    # The mesh the indicator chooses, evolved with the regrid on.
    let case = ref_case(; maxlevel_cap=3, chunk=T(1 // 20))
        forest = gh_forest(T, case; N=8, roots=4)
        t0 = time()
        out = evolve!(T, case; forest=forest, q=q, ops=ops, t_end=T(1 // 2),
                      adapt=true, regrid=true)
        r = out.records[end]
        say("  adaptive : leaves=%4d levels=%s points=%8d h=%8.5f " *
            "passes=%d regrids=%d steps=%4d", out.nblocks,
            string(out.levels), out.nblocks * 8^3, out.h, out.passes,
            out.nregrids, out.nsteps)
        say("             err_l2=%10.3e err_inf=%10.3e gauge_l2=%10.3e " *
            "res=%10.3e τ_max=%.4f offset/h=%.2f  %.1fs", r.err_l2,
            r.err_linf, r.gauge_l2, r.residual, r.τ_max,
            r.centroid_offset / out.h, time() - t0)

        # The same case on a hierarchy chosen by hand, with the same finest
        # spacing: `CODE.md`'s frozen-hierarchy protocol, and what the
        # adaptive run is compared against. The shells are the radii that
        # reproduce the indicator's own layout — the ball the floor covers
        # and the region the ceiling leaves alone — so the two differ only
        # in what the indicator decided about the outside.
        frozen = hole_forest(T, case; N=8, roots=4, radii=(T(5 // 2), T(5 // 2)))
        t1 = time()
        fout = evolve!(T, case; forest=frozen, q=q, ops=ops, t_end=T(1 // 2))
        fr = fout.records[end]
        say("  frozen   : leaves=%4d levels=%s points=%8d h=%8.5f steps=%4d",
            fout.nblocks, string(fout.levels), fout.nblocks * 8^3, fout.h,
            fout.nsteps)
        say("             err_l2=%10.3e err_inf=%10.3e gauge_l2=%10.3e " *
            "res=%10.3e τ_max=%.4f  %.1fs", fr.err_l2, fr.err_linf,
            fr.gauge_l2, fr.residual, fr.τ_max, time() - t1)
        say("  ratios   : err_l2 %.3f  err_inf %.3f  gauge_l2 %.3f  " *
            "points %.3f", r.err_l2 / fr.err_l2, r.err_linf / fr.err_linf,
            r.gauge_l2 / fr.gauge_l2, out.nblocks / fout.nblocks)
    end
end

# --- (5) the horizon, in the charts a test file cannot afford -------------
#
# `CODE.md`, "Analysis quantities": for Kerr the reference values are
# `A = 4π(r₊² + a²)`, `M_irr = √(A/16π)`, `J = M a` and `M_ch = M`, and the
# suite checks them on the one hole it can afford — Kerr-Schild `a = 0` on
# the step-5 fixture. The other two rows are here, because the mesh each of
# them needs is not a mesh a test file can build:
#
#   * **Kerr-Schild `a = 9/10`**, the proof-of-concept spin, where the
#     numbers have content: `J = 0.9` rather than zero, an axis to recover,
#     `A = 4π(r₊² + a²) = 28.4` rather than `16π`, and a horizon that is
#     genuinely oblate (`r_min = 1.436`, `r_max = 1.695`). Its layer needs
#     `h ≈ 0.04 M` — `r_0 > |a|` for the chart's singular disk and
#     `r_1 + 4h ≤ r₊` for the placement — and that is 1128 blocks.
#   * **Harmonic Kerr `a = 0`**, the second chart, whose horizon is at
#     `r = M` rather than `2 M` and which carries no gauge source at all.
#
# Both are *sampled* data: the claim is about the analysis, not about an
# evolution, and `notes/methods-ghso2.md` validates the same quantities the
# same way ("Validated on sampled Kerr-Schild data"). The third row is the
# trace over a run, which is the other half of `CODE.md`'s table — the
# horizon rows at every `k`-th chunk, and what they do while the hole sits
# still.
if "horizon" in SECTIONS
    println("\n=== (5) the horizon ===")
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)

    println("\n-- Kerr's numbers from sampled data --")
    # Each row carries its own seed radius as well as its own layer: the
    # fast flow's transient dips inside the seed sphere, and a query whose
    # window reaches `r_1` is refused by design — so the seed is placed
    # outside the horizon and the layer is placed with room under it. The
    # harmonic chart needs **`h = 5/128`** for that, which is the spacing
    # `CODE.md` predicted it would need before it has an `r_0` where `|h|`
    # is still moderate (`≈ M/23`); at `h = 5/64`, the spacing the
    # "other charts" table uses, `r_0 = 0.2` forces `r_1 = 0.67` against a
    # horizon at `1.0` and the flow's first iterates reach the layer.
    for (label, bg, r_0, r_1, N, radii, N_ah, r_seed) in
        (("KS a=0", SM.KerrSchild(one(T), T(0)), T(2 // 5), T(23 // 20), 8,
          (T(10), T(10), one(T)), 16, T(17 // 10)),
         ("Harmonic a=0", SM.Harmonic(one(T), T(0)), T(3 // 10), T(3 // 5),
          8, (T(10), T(10), T(3 // 2), T(6 // 5)), 16, T(13 // 10)),
         ("KS a=0.9", SM.KerrSchild(one(T), T(9 // 10)), T(19 // 20), T(5 // 4),
          8, (T(10), T(10), T(8 // 5), T(5 // 4)), 20, T(19 // 10)))
        local M = one(T)
        local a = bg.spin
        local case = hole_case(T, bg; M=M, halfwidth=T(5 // 2), r_0=r_0,
                               r_1=r_1, chunk=T(1 // 10), margin=4)
        local forest = shells(case, N, radii)
        local fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                               backend=CPU())
        local t0 = time()
        local p = GHProblem(fs, GhostSchedule(fs, ops), case; q=q)
        fill_exact!(fs, case, zero(T))
        local u = statevector(fs)
        gather!(u, fs)
        local setup = time() - t0
        # A displaced guess, as `PLAN.md` asks: the finder recentres.
        local t1 = time()
        local o = try
            find_gh_horizon(p, u, zero(T); N=N_ah, r_seed=r_seed,
                            origin=SVector{3,T}(T(1 // 10), T(-1 // 20),
                                                T(3 // 40)))
        catch e
            say("%-13s blocks=%4d h=%7.5f  **FAILED**: %s", label,
                nleaves(forest), minimum_spacing(T, forest),
                first(split(sprint(showerror, e), '\n')))
            nothing
        end
        local find = time() - t1
        o === nothing && continue
        # Kerr's own values, for the same `M` and `a`.
        local rp = M + sqrt(M^2 - a^2)
        local area = 4π * (rp^2 + a^2)
        local M_irr = sqrt(area / (16π))
        say("%-13s blocks=%4d h=%7.5f N_ah=%2d iters=%3d  setup %5.1fs " *
            "find %5.2fs", label, nleaves(forest), minimum_spacing(T, forest),
            N_ah, o.iters, setup, find)
        say("   r_min %.6f (%.6f)  r_mean %.6f  r_max %.6f (%.6f)",
            o.r_min, horizon_min_radius(bg), o.r_mean, o.r_max,
            horizon_max_radius(bg))
        say("   area  %.6f (%.6f, rel %8.2e)   M_irr %.6f (%.6f)",
            o.area, area, abs(o.area - area) / area, o.M_irr, M_irr)
        say("   J     %.6f (%.6f)   M_ch %.6f (%.6f)   axis (%.4f, %.4f, " *
            "%.4f)", o.J, M * a, o.M_ch, M, o.spin_axis[1], o.spin_axis[2],
            o.spin_axis[3])
        say("   centre offset %.3e   |H| %.3e   success %s / spin %s",
            o.center_offset, o.H_norm, o.success, o.spin_success)
    end

    println("\n-- the horizon rows of a run: Kerr-Schild a = 0 to t = 10 M --")
    let case = with_horizon(kerr_schild_case(T; M=1, a=0, halfwidth=T(5 // 2),
                                             r_0=T(2 // 5), r_1=T(23 // 20),
                                             chunk=one(T), margin=8),
                            Horizon(T; every=1, N=16))
        local forest = shells(case, 8, (T(10), T(10), one(T)))
        local t0 = time()
        local out = evolve!(T, case; forest=forest, q=q, ops=ops,
                            t_end=T(10), cfl=T(1 // 5))
        say("  %d chunks, %d steps, %.1fs", out.nchunks, out.nsteps,
            time() - t0)
        for r in out.records
            say("   t=%5.2f  r=(%.6f, %.6f, %.6f)  area %.6f  M_irr %.6f  " *
                "J %9.2e  M_ch %.6f  offset %8.2e", r.t, r.r_min, r.r_mean,
                r.r_max, r.area, r.M_irr, r.J, r.M_ch, r.center_offset)
        end
    end
end

println("\ndone")
