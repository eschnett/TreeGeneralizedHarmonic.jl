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
#
# Step 8a adds the `leakage` section, which is **not** in the default list:
# eighty-eight evolutions on a 512-block mesh are one batch job on a 64-core
# Symmetry node (`julia --project=. --threads=64 test/hole_runs.jl leakage`;
# the `symmetry-hpc` skill has the cluster mechanics), and it runs its
# evolutions concurrently when it is given the threads for it. It asks how
# much grid-scale content made inside the horizon crosses it — `PLAN.md`'s
# finding 4 — against the predictions of `test/dispersion.jl`.

import Printf
using Serialization: deserialize, serialize
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

# Which sections to run; all of them by default. An argument of the form
# `key=value` is not a section but an option of one (added in step 8a, for
# the `leakage` section's subsets: `q=2 d=4 eps=0,1/2`), so that a subset
# can be validated locally with the same code the batch job runs.
const SECTIONS = let names = filter(a -> !occursin('=', a), ARGS)
    isempty(names) ? ["order", "long", "charts", "indicator", "horizon", "bounds"] :
    names
end
const OPTIONS = Dict(String(first(split(a, '='; limit=2))) =>
                     String(last(split(a, '='; limit=2)))
                     for a in ARGS if occursin('=', a))

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

# --- (6) leakage: what crosses the horizon from inside it -----------------
#
# `PLAN.md`'s finding 4: the discrete scheme is not causal at the grid scale.
# Every centered first derivative annihilates the Nyquist mode, so the shift
# advection that makes everything ingoing inside the horizon does not act on
# it, and grid-scale content made there can cross the horizon, attenuated
# only by the dissipation. `test/dispersion.jl` predicts how much; this
# section measures it on the step-5 fixture's hole.
#
# **The measurement.** A radial ripple is added to `h_tt` in the initial
# data — `A (1 − s²)³ cos(2π(r − r_c)/λ)`, `s = (r − r_c)/(2h)`, a window
# four cells wide centered at depth `d` cells below the horizon,
# `r_c = r_h − d h`, with `A = 1e−3` and `λ = 2h, 4h, 8h` — and the run is
# compared, point by point and chunk by chunk, with the same run without it.
# The L∞ over the ten `h` components of the difference is taken in the
# shells `r_h + k h ≤ r < r_h + (k+1) h` — `ShellMask`'s membership, one
# mask per shell — for `k = 0 … 8` outside the horizon and `k = −10 … −1`
# inside it; the largest value over the run is what the tables hold,
# divided by `A`. Three views of each shell: all of it; the points within
# 18° of a grid axis, which is the direction `test/dispersion.jl`'s
# one-dimensional model is about and the least damped one; and the points
# with every `|n_d| ≥ 0.4`, a cone about each diagonal, whose box face is
# beyond `r = 3` — the axes meet the Dirichlet face at `r = 5/2 = r_h +
# 6.4 h`, which *reflects* what reaches it, so shells `k ≥ 4` along an axis
# see incident and reflected content together.
#
# Four decisions, each **(proposed in step 8a)** in `CODE.md`:
#
#   * **The mesh is the fixture's finest level everywhere** — `hole_fixture`
#     at `h = 5/64` on a uniform level-3 forest of 512 blocks, rather than
#     its 120-block hierarchy. That hierarchy refines the cube
#     `|x|_∞ ≤ 5/4`, so the horizon at `r = 2` straddles the coarse-fine
#     face and, along the axes, all of the margin but its innermost 1.3
#     cells is at `h = 5/32`, where a ripple of wavelength `2 × 5/64` is a
#     constant. What would be measured there is the interface, not the
#     scheme.
#   * **The ripple is in `h_tt` alone, with `Π` untouched.** The principal
#     part is the same scalar operator for every component (`CODE.md`, "One
#     right-hand-side evaluation"), so one component is as good as twenty;
#     and `Π = 0` excites the two branches of the dispersion relation
#     equally, where a choice of `Π` would pick one.
#   * **`d` is the depth of the window's center.** The window is `2h` on
#     either side, so `d = 2` touches the horizon and `d = 8` clears
#     `r_1 = 23/20` by nine tenths of a cell; the gap between `r_1` and
#     `r_h` is 10.9 cells, and a four-cell window is what fits all three.
#     A consequence the tables must be read with: a window four cells wide
#     is broadband whatever `λ` is, and `λ = 8h` is half a wavelength of it.
#   * **The runs are `2 M` long, not `1 M`**, sampled every `1/20 M`.
#     `test/dispersion.jl` puts the least-attenuated modes at group
#     velocities of `0.1–0.3`, three to fourteen `M` from depth `d` to the
#     outer shells, so at `1 M` the deep sources would not have arrived; the
#     one-dimensional model carries the prediction to `10 M`, and the 3D
#     runs are what check it at `2 M`.
#
# **How it runs.** The unit of work is a *group*: one reference run and the
# ripples measured against it. With a node's worth of threads (sixteen or
# more) the groups run as subprocesses all at once, each `(q, ε)` split in
# two; with fewer, one after the other in this process. The numbers do not
# depend on which (checked on two rows at `t = 1/20 M`, one thread per
# worker against four in this process: identical to every digit printed —
# the threading invariant, met here). On one `amddebugq` node the whole
# section is 45 minutes, a run 6–8 minutes at three or five threads. A
# subset is chosen with options, e.g.
#
#     julia --project=. --threads=4 test/hole_runs.jl leakage q=2 d=4 \
#         eps=0,1/2 lambda=2 t_end=1/4
#
# The ripple is added to the initial data by the **observer at `t = 0`**:
# `evolve!` hands its observer the state vector the first chunk integrates
# from, and this is the one way a test adds to the initial data without a
# change under `src/` (which step 8a makes none of). A run in which the
# ripple did not reach the evolution would show a difference of exactly
# zero, and is refused.

const LEAK_AMPLITUDE = 1e-3
const LEAK_WINDOW = 2              # the window's half-width, in cells
const LEAK_SHELLS = -10:8          # shell k: r_h + k h ≤ r < r_h + (k+1) h
const LEAK_OUTSIDE = 0:8
const LEAK_CHUNK = 1 // 20
const LEAK_RADII = (T(10), T(10), T(10))   # uniform at level 3: 512 blocks

# `a,b,c` or `p/q` lists, for the `key=value` options.
leak_option(key, default) = !haskey(OPTIONS, key) ? default :
    [occursin('/', x) ? parse(Int, split(x, '/')[1]) // parse(Int, split(x, '/')[2]) :
     parse(Int, x) // 1 for x in split(OPTIONS[key], ',')]

leak_ripple(r, r_c, λ, h) =
    (s = (r - r_c) / (LEAK_WINDOW * h);
     abs(s) < 1 ? T(LEAK_AMPLITUDE) * (1 - s^2)^3 * cos(2π * (r - r_c) / λ) : zero(T))

"""
Every owned point of the leakage mesh that lies in one of the shells, with
its shell (by `ShellMask`'s own membership test) and its direction class.
Built once: the forest is rebuilt for every run, deterministically, and
each run checks that its block count and first position agree.
"""
function leak_geometry(U, c, r_h, h)
    N = U.forest.N
    G = first(U.G)
    masks = [ShellMask{T}(c, r_h + k * h, r_h + (k + 1) * h) for k in LEAK_SHELLS]
    pts = NTuple{4,Int}[]
    shell = Int[]
    radius = T[]
    axis = Bool[]
    diag = Bool[]
    for b in 1:nblocks(U), k in 1:N, j in 1:N, i in 1:N
        x = coordinates(U, b, (i + G, j + G, k + G))
        m = findfirst(mk -> is_evolved(mk, x), masks)
        m === nothing && continue
        d = SVector{3,T}(x) - c
        r = sqrt(sum(abs2, d))
        n = abs.(d) ./ r
        push!(pts, (i, j, k, b))
        push!(shell, m)
        push!(radius, r)
        push!(axis, maximum(n) ≥ cosd(18))
        push!(diag, minimum(n) ≥ 2 // 5)
    end
    return (points=pts, shell=shell, radius=radius, axis=axis, diag=diag,
            N=N, G=G,
            nblocks=nblocks(U), x1=coordinates(U, 1, (G + 1, G + 1, G + 1)),
            counts=[count(==(m), shell) for m in eachindex(LEAK_SHELLS)])
end

"""
One leakage run: the fixture at `q` and `ε_KO` to `t_end`, with the ripple
`(λ, d)` added at `t = 0` when `ripple` is given. With `reference =
nothing` it returns the `h` of every measured point at every chunk; with a
reference it returns the largest `|δh|` per shell, view and chunk.
"""
function leak_run(q, ε; t_end, geometry, ripple=nothing, reference=nothing)
    case = hole_fixture(T; q=q, ε_KO=T(ε), chunk=T(LEAK_CHUNK))
    r_h = T(horizon_min_radius(case.background))
    snaps = Matrix{T}[]
    nchunk = ceilint(T(t_end) / T(LEAK_CHUNK))
    amax = zeros(T, length(LEAK_SHELLS), 3, nchunk + 1)
    calls = Ref(0)
    function watch(p, t, u)
        calls[] += 1
        c = calls[]
        U = p.U
        ua = statearray(u, U)
        h = T(minimum_spacing(T, U.forest))
        if c == 1
            (nblocks(U) == geometry.nblocks &&
             coordinates(U, 1, (geometry.G + 1, geometry.G + 1, geometry.G + 1)) ==
             geometry.x1) || error("the leakage mesh is not the geometry's")
            if ripple !== nothing
                λ, d = ripple
                cc = center_at(case.center, zero(T))
                N, G = geometry.N, geometry.G
                for b in 1:nblocks(U), k in 1:N, j in 1:N, i in 1:N
                    x = coordinates(U, b, (i + G, j + G, k + G))
                    r = sqrt(sum(abs2, SVector{3,T}(x) - cc))
                    ua[i, j, k, 1, b] += leak_ripple(r, r_h - d * h, λ * h, h)
                end
            end
        end
        if reference === nothing
            c == 1 && return nothing
            snap = Matrix{T}(undef, 10, length(geometry.points))
            for (n, (i, j, k, b)) in enumerate(geometry.points), v in 1:10
                snap[v, n] = ua[i, j, k, v, b]
            end
            push!(snaps, snap)
            return nothing
        end
        for (n, (i, j, k, b)) in enumerate(geometry.points)
            # At `t = 0` the difference is the ripple itself, in `h_tt`.
            δ = if c == 1
                abs(leak_ripple(geometry.radius[n], r_h - ripple[2] * h,
                                ripple[1] * h, h))
            else
                ref = reference.snaps[c - 1]
                maximum(abs(ua[i, j, k, v, b] - ref[v, n]) for v in 1:10)
            end
            m = geometry.shell[n]
            amax[m, 1, c] = max(amax[m, 1, c], δ)
            geometry.axis[n] && (amax[m, 2, c] = max(amax[m, 2, c], δ))
            geometry.diag[n] && (amax[m, 3, c] = max(amax[m, 3, c], δ))
        end
        return nothing
    end
    t0 = time()
    out = gh_hole_run(T, case; N=8, q=q, t_end=T(t_end), radii=LEAK_RADII,
                      observer=watch)
    wall = time() - t0
    dts = [r.dt for r in out.records]
    if reference === nothing
        return (snaps=snaps, dts=dts, wall=wall, nsteps=out.nsteps, h=out.h)
    end
    dts == reference.dts ||
        @warn "the perturbed run's steps differ from the reference's" maxrel =
            maximum(abs.(dts .- reference.dts) ./ max.(reference.dts, eps()))
    # The ripple must have reached the evolution: at the first chunk the
    # difference in the source's own shells is not zero.
    maximum(amax[:, 1, 2]) > 0 ||
        error("the ripple did not reach the evolution: the difference at " *
              "the first chunk is exactly zero, so the observer's addition at " *
              "t = 0 was not the state the first chunk integrated from")
    return (amax=amax, wall=wall, nsteps=out.nsteps, h=out.h,
            sameSteps=dts == reference.dts, finite=out.records[end].finite)
end

# The geometry for one order, from a forest built the way every run builds
# its own.
function leak_geometry(q)
    case = hole_fixture(T; q=q)
    f = hole_fixture_forest(T, case; N=8, radii=LEAK_RADII)
    U = FieldSet{T}(f, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                    backend=CPU())
    return leak_geometry(U, center_at(case.center, zero(T)),
                         T(horizon_min_radius(case.background)),
                         T(minimum_spacing(T, f)))
end

"""
One **group**: the reference run at `(q, ε)` and the ripples measured
against it, one after the other, each printing a line the moment it ends.
A ripple that throws returns its exception rather than ending the group.
"""
function leak_group(q, ε, ripples; t_end)
    geometry = leak_geometry(q)
    ref = leak_run(q, ε; t_end=t_end, geometry=geometry)
    iout = [findfirst(==(k), LEAK_SHELLS) for k in LEAK_OUTSIDE]
    res = map(ripples) do (λ, d)
        r = try
            leak_run(q, ε; t_end=t_end, geometry=geometry, ripple=(λ, d),
                     reference=ref)
        catch e
            e
        end
        if r isa Exception
            say("   done q=%d ε=%.2f λ=%dh d=%d: **FAILED** %s", q, ε, λ, d,
                first(split(sprint(showerror, r), '\n')))
        else
            peak = dropdims(maximum(r.amax; dims=3); dims=3) ./ LEAK_AMPLITUDE
            say("   done q=%d ε=%.2f λ=%dh d=%d (%.0f s): A_k/A, k = 0 … 8: %s",
                q, ε, λ, d, r.wall, leak_row(peak[iout, 1]))
        end
        flush(stdout)
        r
    end
    return (keys=[(q, ε, λ, d) for (λ, d) in ripples], res=res,
            refwall=ref.wall, refsteps=ref.nsteps, counts=geometry.counts,
            npoints=length(geometry.points))
end

# `2//1` → `"2/1"`, the spelling the options take.
leak_spell(x::Rational) = "$(numerator(x))/$(denominator(x))"
leak_spell(x) = string(x)

"""
Run every group, each in a **subprocess** of its own with `threads[q]`
threads, all at once, and return their results in order. A process of its
own rather than a task, because sixteen runs sharing one process share its
garbage collector and its scheduler, and a 64-core node measured at a
third busy that way (step 8a's first batch job: 1228 % CPU in `top`, 23
cores on average over its first sixteen minutes). Each worker's output goes
to a log beside its result, and a worker that fails has its log's tail
printed.
"""
function leak_fanout(groups, threads; λs, ds, t_end)
    # Beside the batch job's own log when there is one (a job run from a
    # directory holding `out/`, where its log is), so that a job cut off by
    # its time limit leaves each worker's finished rows behind; a scratch
    # directory otherwise.
    dir = isdir("out") ? mkpath(joinpath("out", "leakage")) : mktempdir()
    println("   worker logs and results in ", abspath(dir))
    project = dirname(Base.active_project())
    procs = map(enumerate(groups)) do (n, (q, ε, part, nparts))
        out = joinpath(dir, "group-$n.jls")
        log = joinpath(dir, "group-$n.log")
        cmd = `$(Base.julia_cmd()) --project=$project --threads=$(threads[q])
               $(abspath(@__FILE__)) leakage worker=1 q=$q eps=$(leak_spell(ε))
               part=$part/$nparts lambda=$(join(λs, ',')) d=$(join(ds, ','))
               t_end=$(leak_spell(t_end)) out=$out`
        io = open(log, "w")
        (run(pipeline(cmd; stdout=io, stderr=io); wait=false), out, log, io)
    end
    return map(procs) do (proc, out, log, io)
        wait(proc)
        close(io)
        if !success(proc) || !isfile(out)
            println("   a worker failed; the end of its log ($log):")
            foreach(l -> println("     ", l), last(readlines(log), 30))
            return nothing
        end
        foreach(l -> startswith(l, "   done") && println(l), readlines(log))
        return deserialize(out)
    end
end

# Least-squares e-folds per cell of `A_k` against `k`, over the entries above
# the difference's own floor.
function leak_fit(ks, A; floor=1e-12)
    sel = [i for i in eachindex(A) if A[i] > floor]
    length(sel) ≥ 2 || return NaN
    x = T[ks[i] for i in sel]
    y = [log(A[i]) for i in sel]
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return -sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end

leak_row(A) = join((Printf.format(Printf.Format("%.2e"), a) for a in A), " ")

if "leakage" in SECTIONS
    qs = Int.(leak_option("q", [2, 4]))
    εs = leak_option("eps", [0 // 1, 1 // 4, 1 // 2, 1 // 1])
    λs = Int.(leak_option("lambda", [2, 4, 8]))
    ds = Int.(leak_option("d", [2, 4, 8]))
    t_end_q = only(leak_option("t_end", [2 // 1]))
    t_end = T(t_end_q)
    ripples_all = [(λ, d) for λ in λs for d in ds]
    if haskey(OPTIONS, "worker")
        # One group of a batch job: `q`, `eps` and `part = i/n` name it, and
        # it takes every `n`-th ripple starting at the `i`-th.
        ipart, npart = parse.(Int, split(OPTIONS["part"], '/'))
        grp = leak_group(only(qs), only(εs), ripples_all[ipart:npart:end];
                         t_end=t_end)
        serialize(OPTIONS["out"], grp)
    else
        println("\n=== (6) leakage: a ripple inside the horizon, and what " *
                "reaches the shells outside it ===")
        # With a node's worth of threads the groups run as subprocesses at
        # once, each `(q, ε)` split in two with a reference each — `q = 4`
        # given five threads and `q = 2` three, roughly their costs, so that
        # sixteen workers fill 64 cores and finish together. With a
        # workstation's, they run here one after the other.
        nt = Threads.nthreads()
        fan = nt ≥ 16
        nparts = fan ? 2 : 1
        threads = Dict(4 => max(1, round(Int, 5nt / 64)),
                       2 => max(1, round(Int, 3nt / 64)),
                       6 => max(1, round(Int, 6nt / 64)))
        groups = [(q, ε, part, nparts) for q in sort(qs; rev=true) for ε in εs
                  for part in 1:nparts]
        say("q ∈ %s, ε_KO ∈ %s, λ/h ∈ %s, d ∈ %s, t_end = %.2f M, A = %.0e; " *
            "%d groups %s", string(qs), string(Float64.(εs)), string(λs),
            string(ds), t_end, LEAK_AMPLITUDE, length(groups),
            fan ? "in subprocesses at $(join(("q=$q: $(threads[q]) threads"
                                              for q in qs), ", "))" :
            "in this process at $nt threads")
        t0 = time()
        outs = fan ? leak_fanout(groups, threads; λs=λs, ds=ds, t_end=t_end_q) :
               [leak_group(q, ε, ripples_all[part:nparts:end]; t_end=t_end)
                for (q, ε, part, nparts) in groups]
        say("%d groups in %.0f s", length(groups), time() - t0)
        good = filter(!isnothing, outs)
        isempty(good) && error("every leakage group failed")
        say("measured points: %d; per shell (k = %d … %d): %s",
            first(good).npoints, first(LEAK_SHELLS), last(LEAK_SHELLS),
            string(first(good).counts))
        for (g, o) in zip(groups, outs)
            o === nothing && continue
            say("   reference q=%d ε=%.2f part %d/%d: %.0f s, %d steps", g[1],
                g[2], g[3], g[4], o.refwall, o.refsteps)
        end
        # Back into the order of the tables.
        byKey = Dict(k => r for o in good for (k, r) in zip(o.keys, o.res))
        runkeys = [(q, ε, λ, d) for q in qs for ε in εs for λ in λs for d in ds
                   if haskey(byKey, (q, ε, λ, d))]
        res = [byKey[k] for k in runkeys]
        iout = [findfirst(==(k), LEAK_SHELLS) for k in LEAK_OUTSIDE]
        iin = [findfirst(==(k), LEAK_SHELLS) for k in first(LEAK_SHELLS):-1]
        times = [T(c) * T(LEAK_CHUNK) for c in 0:ceilint(t_end / T(LEAK_CHUNK))]
        for ((q, ε, λ, d), r) in zip(runkeys, res)
            say("\n-- q=%d ε_KO=%.2f λ=%dh d=%d --", q, ε, λ, d)
            if r isa Exception
                println("   **FAILED**: ", first(split(sprint(showerror, r), '\n')))
                continue
            end
            peak = dropdims(maximum(r.amax; dims=3); dims=3) ./ LEAK_AMPLITUDE
            say("   wall %.0f s, steps %d, same steps as the reference: %s, " *
                "finite: %s", r.wall, r.nsteps, r.sameSteps, r.finite)
            println("   A_k/A outside, k = 0 … 8 (all)  : ", leak_row(peak[iout, 1]))
            println("   A_k/A outside, k = 0 … 8 (axis) : ", leak_row(peak[iout, 2]))
            println("   A_k/A outside, k = 0 … 8 (diag) : ", leak_row(peak[iout, 3]))
            println("   A_k/A inside, k = −10 … −1 (all): ", leak_row(peak[iin, 1]))
            k0 = findfirst(==(0), LEAK_SHELLS)
            k8 = findfirst(==(8), LEAK_SHELLS)
            println("   A_0(t)/A every 0.25 M (all)     : ",
                    leak_row(r.amax[k0, 1, 1:5:end] ./ LEAK_AMPLITUDE))
            println("   A_8(t)/A every 0.25 M (all)     : ",
                    leak_row(r.amax[k8, 1, 1:5:end] ./ LEAK_AMPLITUDE))
            c0 = argmax(r.amax[k0, 1, :])
            c8 = argmax(r.amax[k8, 1, :])
            say("   peaks at t = %.2f M (k = 0) and %.2f M (k = 8); e-folds per " *
                "cell outside: all %.3f, axis %.3f, diag %.3f", times[c0], times[c8],
                leak_fit(LEAK_OUTSIDE, peak[iout, 1]),
                leak_fit(LEAK_OUTSIDE, peak[iout, 2]),
                leak_fit(LEAK_OUTSIDE, peak[iout, 3]))
        end

        # The summary `CODE.md` records.
        println("\n-- summary: A_0/A, A_4/A, A_8/A over the whole shell, the axis " *
                "cone's A_0/A, and the e-folds per cell outside (all) --")
        println("| q | ε_KO | λ/h | d | A_0/A | A_4/A | A_8/A | axis A_0/A | " *
                "e-folds/cell |")
        for ((q, ε, λ, d), r) in zip(runkeys, res)
            r isa Exception && continue
            peak = dropdims(maximum(r.amax; dims=3); dims=3) ./ LEAK_AMPLITUDE
            say("| %d | %.2f | %d | %d | %.2e | %.2e | %.2e | %.2e | %.3f |", q, ε,
                λ, d, peak[iout[1], 1], peak[iout[5], 1], peak[iout[9], 1],
                peak[iout[1], 2], leak_fit(LEAK_OUTSIDE, peak[iout, 1]))
        end
        println("\n-- the depth dependence: e-folds per cell of A_0 against d " *
                "(all, axis) --")
        for q in qs, ε in εs, λ in λs
            A = T[]
            Aax = T[]
            dd = Int[]
            for d in ds
                i = findfirst(==((q, ε, λ, d)), runkeys)
                i === nothing && continue
                r = res[i]
                r isa Exception && continue
                peak = dropdims(maximum(r.amax; dims=3); dims=3) ./ LEAK_AMPLITUDE
                push!(dd, d)
                push!(A, peak[iout[1], 1])
                push!(Aax, peak[iout[1], 2])
            end
            length(dd) ≥ 2 || continue
            say("| %d | %.2f | %d | %.3f | %.3f |", q, ε, λ, leak_fit(dd, A),
                leak_fit(dd, Aax))
        end
    end
end

# --- (7) the range projection on the two runs that end (added in step 8b) --
#
# `CODE.md`, "The interior" (the range projection) and `PLAN.md`'s step 8b.
# Step 5's table has two rows that end in a degenerate metric: `:damped` one
# resolution coarser than the suite's (`N = 6`, `21 M`) and `:pasted` at the
# suite's (`N = 8`, `17 M`). Each is run twice here, without the projection
# and with it at the proposed bounds and gate, and the two are compared
# chunk by chunk: the runs are bit for bit the same until the projection
# first fires (the suite's control is the claim that says so), so every
# difference after that is the clamp's, and the question the section answers
# is how far out it reaches — the constraint norms and the state difference
# in shells `[r_h + k h, r_h + (k+1) h]`, `k = 0 … 8`, outside the horizon
# (`r_h = 2 M`), which are step 8a's shells.
#
# The prediction to confirm or correct is `PLAN.md`'s: hits start "deep,
# several `M` before the crash". The kernel's cost is measured first, on the
# suite's `N = 8` mesh (prediction: 0.4 % of a step).
#
# Rows: `cost`, `damped6`, `pasted8`; all by default, or a subset as
# `bounds=damped6,pasted8`, which is how the section is split across batch
# jobs.
# The rows come from the `bounds=<row>,…` option (step 8a's `key=value`
# protocol, so the argument is in `OPTIONS` and not in `SECTIONS`), or all
# three when the section is named or run by default (merged in review).
const BOUNDS_ROWS = haskey(OPTIONS, "bounds") ?
                    String.(split(OPTIONS["bounds"], ',')) :
                    "bounds" in SECTIONS ? ["cost", "damped6", "pasted8"] : String[]
# `t_end` is `50 M`, the length of step 5's table; `TREEGH_BOUNDS_TEND`
# overrides it for a smoke test of the section itself.
const BOUNDS_TEND = T(parse(Float64, get(ENV, "TREEGH_BOUNDS_TEND", "50")))
# Its autopsy steps the integrator by hand, which `evolution_cases.jl` does
# not import.
using SciMLBase: init, step!

# The section is a function rather than a top-level block: it defines
# closures that assign to its own locals, which a script's soft scope makes
# ambiguous (and turns into warnings, or into an `UndefVarError`).
function bounds_section(rows)
    println("\n=== (6) the range projection ===")
    q = 2
    G = q ÷ 2 + 1
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    radii = (T(10), T(10), one(T))
    base_case(variant) = kerr_schild_case(T; M=1, a=0, halfwidth=T(5 // 2),
                                          r_0=T(2 // 5), r_1=T(23 // 20),
                                          chunk=one(T), margin=8,
                                          interior=variant)
    bounded(case, forest) =
        with_bounds(case, default_bounds(T; M=1,
                                         r_gate=default_gate(case.interior,
                                                             forest, q)))

    if "cost" in rows
        println("\n-- the kernel's cost on the suite's mesh (N = 8, " *
                "$(Threads.nthreads()) threads) --")
        case0 = base_case(:damped)
        forest = shells(case0, 8, radii)
        case = bounded(case0, forest)
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                         backend=CPU())
        p = GHProblem(fs, GhostSchedule(fs, ops), case; q=q,
                      interior=with_ρ_max(case.interior, T(10)),
                      accounting=BoundsAccounting())
        fill_exact!(fs, case, zero(T))
        u = statevector(fs)
        gather!(u, fs)
        du = similar(u)
        npts = nblocks(fs) * forest.N^3
        ngate = 0
        for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N,
            i in 1:forest.N

            x = coordinates(fs, b, (i + G, j + G, k + G))
            sqrt(sum(abs2, x)) < case.bounds.r_gate && (ngate += 1)
        end
        best(f) = (f(); minimum(1:7) do _
            t0 = time_ns()
            f()
            (time_ns() - t0) / 1e9
        end)
        trhs = best(() -> gh_rhs!(du, u, p, zero(T)))
        tlim = best(() -> gh_stage_limiter!(u, nothing, p, zero(T)))
        # One RK4 step here is four stage-limiter calls and five right-hand
        # sides: four stages, and the FSAL re-evaluation a non-trivial step
        # limiter asks for.
        say("  points %d, inside the gate r < %.4f: %d (%.1f %%)", npts,
            case.bounds.r_gate, ngate, 100 * ngate / npts)
        say("  RHS %.4f s (%.0f ns/point)   stage limiter %.5f s " *
            "(%.1f ns/point, %.0f ns/gated point)", trhs, trhs * 1e9 / npts,
            tlim, tlim * 1e9 / npts, tlim * 1e9 / max(ngate, 1))
        say("  the projection's share of a step: %.2f %% " *
            "(4 limiter calls against 5 RHS)",
            100 * 4 * tlim / (5 * trhs + 4 * tlim))
        # And what it costs where it fires: every gated point with a
        # negative eigenvalue of γ, the worst case — a Jacobi decomposition,
        # a recomposition, and the verification pass, at every one.
        xx = TreeGeneralizedHarmonic._pairindex(2, 2)
        spoiled = copy(u)
        As = statearray(spoiled, fs)
        for b in 1:nblocks(fs), k in 1:forest.N, j in 1:forest.N,
            i in 1:forest.N

            x = coordinates(fs, b, (i + G, j + G, k + G))
            sqrt(sum(abs2, x)) < case.bounds.r_gate || continue
            As[i, j, k, xx, b] = -T(3 // 2)
        end
        work = copy(spoiled)
        tfire = (copyto!(work, spoiled); gh_stage_limiter!(work, nothing, p,
                                                            zero(T));
                 minimum(1:5) do _
                     copyto!(work, spoiled)
                     t0 = time_ns()
                     gh_stage_limiter!(work, nothing, p, zero(T))
                     (time_ns() - t0) / 1e9
                 end)
        say("  firing at every gated point: %.5f s (%.0f ns/gated point, " *
            "%.1f × the healthy call)", tfire, tfire * 1e9 / max(ngate, 1),
            tfire / tlim)
    end

    # The shells outside the horizon: the gauge constraint's L2 and L∞ in
    # each, and — against the unbounded run's state at the same time — the
    # largest change of any component there.
    function shell_readings(p, t, u; rh=T(2), K=8)
        tt = T(t)
        h = minimum_spacing(T, p.U.forest)
        c = center_at(p.interior.center, tt)
        return map(0:K) do k
            mask = ShellMask{T}(c, rh + k * h, rh + (k + 1) * h)
            gh_constraint!(p, u, tt; mask=mask)
            cn = constraint_norms(p)
            (l2=Float64(maximum(cn.gauge_l2)),
             linf=Float64(maximum(cn.gauge_linf)))
        end
    end
    # Each owned point's radius and shell index, once per mesh (no regrid).
    function shell_index(U; rh=T(2), K=8)
        N = U.forest.N
        G = first(U.G)
        h = minimum_spacing(T, U.forest)
        idx = Array{Int}(undef, N, N, N, nblocks(U))
        rad = Array{T}(undef, N, N, N, nblocks(U))
        for b in 1:nblocks(U), k in 1:N, j in 1:N, i in 1:N
            r = sqrt(sum(abs2, coordinates(U, b, (i + G, j + G, k + G))))
            rad[i, j, k, b] = r
            s = floor((r - rh) / h)
            idx[i, j, k, b] = 0 ≤ s ≤ K ? Int(s) : -1
        end
        return idx, rad
    end
    function shell_diffs(U, idx, u1, u0; K=8)
        A1 = statearray(u1, U)
        A0 = statearray(u0, U)
        d = zeros(K + 1)
        N = U.forest.N
        for b in 1:nblocks(U), k in 1:N, j in 1:N, i in 1:N
            s = idx[i, j, k, b]
            s < 0 && continue
            for v in 1:20
                d[s + 1] = max(d[s + 1], abs(A1[i, j, k, v, b] - A0[i, j, k, v, b]))
            end
        end
        return d
    end

    # Radial bins of the state's validity, host-side: in each bin the
    # smallest `det γ` and signed lapse (with the radius they occur at), the
    # largest `|Π_ab|`, the largest `‖u − u_exact‖`, and how many values are
    # not finite. This is the autopsy the record cannot give once a run has
    # thrown: it says *where* the state stopped being a metric.
    function radial_bins(U, u, case, t, edges)
        N = U.forest.N
        G = first(U.G)
        A = statearray(u, U)
        nb = length(edges) - 1
        dγ = fill(Inf, nb)
        rdγ = fill(NaN, nb)
        α = fill(Inf, nb)
        rα = fill(NaN, nb)
        Πm = zeros(nb)
        err = zeros(nb)
        bad = zeros(Int, nb)
        tt = T(t)
        for b in 1:nblocks(U), k in 1:N, j in 1:N, i in 1:N
            x = coordinates(U, b, (i + G, j + G, k + G))
            r = sqrt(sum(abs2, x))
            s = clamp(searchsortedlast(edges, r), 1, nb)
            hv = SVector{10,T}(ntuple(v -> A[i, j, k, v, b], 10))
            Πv = SVector{10,T}(ntuple(v -> A[i, j, k, 10 + v, b], 10))
            nf = count(!isfinite, hv) + count(!isfinite, Πv)
            bad[s] += nf
            nf > 0 && continue
            d, a, _, pm = state_validity(hv, Πv)
            a == floatmax(T) && (a = T(NaN))
            (d < dγ[s]) && (dγ[s] = d; rdγ[s] = r)
            (a < α[s] || isnan(a)) && (α[s] = a; rα[s] = r)
            Πm[s] = max(Πm[s], pm)
            ex = case_state_tuple(case.background, case.interior, tt, x)
            e = sqrt(sum(v -> (A[i, j, k, v, b] - ex[v])^2, 1:20))
            err[s] = max(err[s], e)
        end
        return (; dγ, rdγ, α, rα, Πm, err, bad)
    end
    # The root cause of a run that threw, through the task wrappers a
    # threaded kernel puts around it, with the package's own frames of its
    # backtrace — which kernel, and which line.
    function describe_failure(e, bt)
        root = e
        while true
            if root isa TaskFailedException
                stk = Base.current_exceptions(root.task)
                isempty(stk) || (bt = stk[end].backtrace)
                root = root.task.result
            elseif root isa CompositeException
                root = first(root.exceptions)
            else
                break
            end
        end
        msg = first(split(sprint(showerror, root), '\n'))
        frames = [string(f.func, " @ ", basename(string(f.file)), ":", f.line)
                  for f in stacktrace(bt)
                  if occursin("TreeGeneralizedHarmonic", string(f.file))]
        return msg, first(unique(frames), 6)
    end
    fmt(x) = Printf.format(Printf.Format("%9.2e"), x)
    fmtr(x) = Printf.format(Printf.Format("%6.3f"), x)

    for (label, N, variant) in (("damped6", 6, :damped), ("pasted8", 8, :pasted))
        label in rows || continue
        println("\n-- $label: KS a=0, q=2, N=$N, :$variant, to $(BOUNDS_TEND) M, " *
                "without the projection, with it at the proposed gate, and at " *
                "the widest gate the assertion allows --")
        case0 = base_case(variant)
        int = case0.interior
        forest0 = shells(case0, N, radii)
        h = minimum_spacing(T, forest0)
        case1 = bounded(case0, forest0)
        widest = check_bounds_gate(forest0, int, case1.bounds, q).allowed
        case2 = with_bounds(case0, default_bounds(T; M=1, r_gate=widest))
        r_h = T(horizon_min_radius(case0.background))
        edges = T[0, int.r_0, case1.bounds.r_gate, int.r_1, int.r_1 + G * h,
                  int.r_1 + 2G * h, r_h, r_h + 4h, Inf]
        names = ["core", "in-layer", "out-layer", "r1+Gh", "r1+2Gh", "→r_h",
                 "r_h+4h", "outside"]
        say("  r_0 = %.4f  r_gate = %.4f (widest %.4f)  r_1 = %.4f  h = %.5f",
            int.r_0, case1.bounds.r_gate, widest, int.r_1, h)
        println("  radial bins: ", join((Printf.format(Printf.Format("%s [%.3f, %.3f)"),
                                                       names[k], edges[k], edges[k + 1])
                                         for k in 1:length(names)), ", "))
        refstates = Dict{Int,Vector{T}}()
        chunks = Dict{Tuple{Int,Int},Any}()
        outs = Dict{Int,Any}()
        idx = shell_index(FieldSet{T}(forest0, 20; G=G, centering=vertexcentered(3),
                                      backend=CPU()))
        runs = ((0, "no projection", case0), (1, "gate r_1 − 2Gh", case1),
                (2, "gate r_1 − Rh", case2))
        for (run, what, case) in runs
            forest = shells(case, N, radii)
            reached = Ref(zero(T))
            function watch(p, t, u)
                reached[] = T(t)
                c = round(Int, t)
                run == 0 && (refstates[c] = copy(u))
                d = run > 0 && haskey(refstates, c) ?
                    shell_diffs(p.U, idx[1], u, refstates[c]) : nothing
                chunks[(run, c)] = (hits=p.accounting === nothing ? 0 :
                                         p.accounting.hits,
                                    shells=shell_readings(p, t, u), diff=d,
                                    bins=run == 0 ? radial_bins(p.U, u, case, t, edges) :
                                         nothing)
                return nothing
            end
            t0 = time()
            local failure = nothing
            out = try
                evolve!(T, case; forest=forest, q=q, ops=ops,
                        t_end=BOUNDS_TEND, cfl=T(1 // 5), observer=watch)
            catch e
                e isa InterruptException && rethrow()
                failure = describe_failure(e, catch_backtrace())
                nothing
            end
            acc = out === nothing ? nothing : out.bounds
            outs[run] = (reached=reached[], wall=time() - t0, failure=failure,
                         acc=acc)
            say("  run %d (%s): reached %.2f M in %.1f s%s", run, what,
                reached[], time() - t0, failure === nothing ? "" :
                                        ", then threw: " * failure[1])
            failure === nothing || println("     in: ", join(failure[2], " ← "))
            if run > 0
                hs = [chunks[(run, c)].hits for c in 0:round(Int, reached[])]
                c1 = findfirst(>(0), hs)
                say("     projection hits: %d in total%s", last(hs),
                    c1 === nothing ? ", never fired" :
                    Printf.format(Printf.Format(", first in the chunk ending at t = %d M"),
                                  c1 - 1))
                acc === nothing ||
                    say("     accounting: %d calls, first hit at t = %.4f M, " *
                        "r = %.4f; outermost %.4f", acc.calls, acc.first_t,
                        acc.first_r, acc.r_max)
            end
        end

        # The run without the projection, chunk by chunk and bin by bin.
        c_end = round(Int, outs[0].reached)
        for (title, key, f) in (("min det γ", :dγ, fmt), ("min α (signed)", :α, fmt),
                                ("radius of that min α", :rα, fmtr),
                                ("max |Π_ab|", :Πm, fmt),
                                ("max ‖u − u_exact‖", :err, fmt),
                                ("non-finite values", :bad, string))
            println("  $title, by bin (", join(names, ", "), "):")
            for c in 0:c_end
                b = chunks[(0, c)].bins
                say("   t=%4d  %s", c, join((lpad(f(x), 9) for x in getfield(b, key)), " "))
            end
        end

        # Whether the clamp's discontinuity reached the horizon: the shells
        # outside it, at the chunks where a run with the projection first
        # fired and after — against the run without it.
        for run in (1, 2)
            hs = [chunks[(run, c)].hits for c in 0:round(Int, outs[run].reached)]
            c1 = findfirst(>(0), hs)
            c1 === nothing && continue
            c1 -= 1
            println("  run $run: shells [r_h + k h, r_h + (k+1) h], k = 0 … 8, " *
                    "from the chunk it first fired in")
            for c in unique(filter(c -> haskey(chunks, (run, c)),
                                   [c1, c1 + 1, c1 + 2, c1 + 5, c1 + 10, c1 + 20]))
                on = chunks[(run, c)]
                off = get(chunks, (0, c), nothing)
                say("   t = %4d  C_a L∞ on : %s", c, join((fmt(x.linf) for x in on.shells), " "))
                off === nothing ||
                    say("              C_a L∞ off: %s", join((fmt(x.linf) for x in off.shells), " "))
                on.diff === nothing ||
                    say("              max |Δu|  : %s", join((fmt(x) for x in on.diff), " "))
            end
        end

        # The autopsy: the run without the projection again, to the last
        # chunk it finished — bit for bit the same run — and then the fatal
        # chunk one step at a time, with the integrator `evolve!` uses and
        # the step it would take, until it throws. `TREEGH_BOUNDS_AUTOPSY=1`
        # forces it on the last chunk of a run that did not throw, which is
        # how the autopsy itself is smoke-tested.
        forced = get(ENV, "TREEGH_BOUNDS_AUTOPSY", "") == "1"
        if outs[0].failure !== nothing || forced
            c_a = outs[0].failure === nothing ? c_end - 1 : c_end
            println("  autopsy of the chunk $(c_a) → $(c_a + 1) M, step by step, " *
                    "without the projection:")
            forest = shells(case0, N, radii)
            out = evolve!(T, case0; forest=forest, q=q, ops=ops, t_end=T(c_a),
                          cfl=T(1 // 5))
            p = out.problem
            u = copy(out.u)
            tc = T(c_a)
            λ = TreeGeneralizedHarmonic.max_speed_of(p, u, tc)
            dt = T(1 // 5) * minimum_spacing(T, out.forest) / λ
            steps = max(1, ceil(Int, 1 / dt))
            dt_used = one(T) / steps
            p = with_interior(p, TreeGeneralizedHarmonic.chunk_interior(case0,
                                                                        dt_used,
                                                                        one(T)))
            integ = init(ODEProblem(gh_rhs!, u, (tc, tc + 1), p), RK4();
                         dt=dt_used, adaptive=false, save_everystep=false,
                         stage_limiter=gh_stage_limiter!,
                         step_limiter=gh_step_limiter!)
            history = Any[]
            fatal = nothing
            for s in 1:steps
                try
                    step!(integ)
                catch e
                    e isa InterruptException && rethrow()
                    fatal = (step=s, what=describe_failure(e, catch_backtrace()))
                    break
                end
                push!(history, (step=s, t=integ.t,
                                bins=radial_bins(p.U, integ.u, case0, integ.t, edges)))
            end
            say("   %d steps of dt = %.5f; %s", steps, dt_used,
                fatal === nothing ? "the chunk integrated, so the failure is " *
                                    "the end-of-chunk check" :
                                    "step $(fatal.step) threw: $(fatal.what[1])")
            fatal === nothing || println("     in: ", join(fatal.what[2], " ← "))
            for e in history[max(1, end - 7):end]
                b = e.bins
                say("   step %3d t=%8.4f", e.step, e.t)
                say("     min α     %s", join((lpad(fmt(x), 9) for x in b.α), " "))
                say("     at r      %s", join((lpad(fmtr(x), 9) for x in b.rα), " "))
                say("     min detγ  %s", join((lpad(fmt(x), 9) for x in b.dγ), " "))
                say("     max |Π|   %s", join((lpad(fmt(x), 9) for x in b.Πm), " "))
                say("     max err   %s", join((lpad(fmt(x), 9) for x in b.err), " "))
                say("     non-finite %s", join((lpad(string(x), 9) for x in b.bad), " "))
            end
        end
        empty!(refstates)
    end
end

isempty(BOUNDS_ROWS) || bounds_section(BOUNDS_ROWS)

println("\ndone")
