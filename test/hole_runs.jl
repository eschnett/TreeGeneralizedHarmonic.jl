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
# Step 8c adds the `calibration` section, which is not in the default list
# either: 138 screens of `5 M` and 62 runs to `50 M` were thirteen batch jobs
# and 5.6 node-hours on Symmetry, run by group
# (`hole_runs.jl calibration=scan+profile`); its own header says how.
#
# Step 8d adds the `tracked` section, which is not in the default list: the
# suite's tracked hole — the step-5 fixture's mesh, `m = 10`, the finder every
# chunk — against step 5's sphere with the same layer, to `5 M` by default
# (`hole_runs.jl tracked`, or `tracked=<t_end>`), four minutes at four
# threads. Its table is in `CODE.md`, "The tracked geometry (step 8d)".
#
# Step 8e adds the `fitted` section, which is not in the default list either:
# the `:fitted` variant's three trials, selected as `fitted=<row>,…` —
# `fixture` (the suite's tracked hole to `1 M` under four choices of initial
# data against `:damped`, two minutes at four threads), `boosted` (a moving
# seed, `boost(Harmonic(1, 0), 0.3 x̂)`, to `0.1 M` on a 5/128 mesh — the
# finder on a boosted hole and the cache's refill cadence) and `harmonic`
# (the first construction of harmonic Kerr's `a = 9/10` initial data, `m =
# 4`, `h = 5/256`, 2472 blocks: its validity, one right-hand side, and the
# first chunk). `fitted=<row>` alone runs only that row. Its numbers are in
# `CODE.md`, "The fitted target".
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
using StaticArrays: SMatrix, SVector
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
# can be validated locally with the same code the batch job runs. An option
# whose *key* is a section's name — `bounds=damped6`, `calibration=scan` —
# selects that section's subset and so names the section: then nothing runs
# by default (amended in step 8c: the review merge of step 8b made
# `hole_runs.jl bounds=damped6` alone run every default section as well,
# since no bare name was given).
const SECTION_NAMES = ["order", "long", "charts", "indicator", "horizon",
                       "bounds", "leakage", "calibration", "tracked", "fitted",
                       "generic"]
const SECTIONS = let names = filter(a -> !occursin('=', a), ARGS)
    keyed = [first(split(a, '='; limit=2)) for a in ARGS if occursin('=', a)]
    isempty(names) && !any(in(SECTION_NAMES), keyed) ?
    ["order", "long", "charts", "indicator", "horizon", "bounds"] : names
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
# **At the grid rate, asked for by name (amended in step 8c′).** Both rows
# are replays of step 5's table, which ran at `ρ_max · dt = 1` — the default
# until 2026-09-23 — and the autopsy below rebuilds the fatal chunk's
# interior at that rate explicitly, so every `evolve!` here passes
# `ρ_max_factor = 1`: at the default `4/M` the section would evolve a
# different layer from the one its autopsy steps through, and `damped6`
# would no longer be the run that ends at `21 M` **(proposed in step 8c′)**.
# `pasted8` does not read the rate at all — the paste freezes the whole
# ball `r < r_1`.
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
                        t_end=BOUNDS_TEND, cfl=T(1 // 5), observer=watch,
                        ρ_max_factor=one(T))
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
                          cfl=T(1 // 5), ρ_max_factor=one(T))
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

# --- (8) calibration: the layer against an inexact target (step 8c) --------
#
# `PLAN.md`'s step 8c and its finding 1: `ρ_max = 1/dt` is a *grid* rate —
# about `107/M` on this fixture against the surface gravity `1/(4M)` — so
# step 5's `:damped` layer is a paste two cells deep, and it survives to
# `50 M` because its target is the exact solution. A target that is not
# (a fitted one, step 8e) pinned that hard that close to the evolved
# stencils is a kink at `r_1`, and step 8b measured step 5's two failing
# runs dying exactly there. This section measures what a layer needs for a
# target that is wrong, on `hole_fixture` (Kerr-Schild `a = 0`, `q = 2`,
# `N = 8`, `h = 5/64`, `r_1 = 23/20`, `m = 8`, `cfl = 1/5`), every run with
# the range projection on (`default_bounds`, `default_gate`) as a passive
# instrument and the finder's `M_irr` at every other chunk:
#
#   * **E3** — `h_tt` of the target off by `A (r − r_1)² χ(r)`, value and
#     slope right at `r_1`, curvature wrong (`CurvatureTarget`,
#     `A = −2/M²`: the sign is argued there). The `N = 6, 8, 10` sweep to
#     `3/20 M` for the order of the `G`-point shell's `C_a` (group
#     `sweep`), then the scan `n_L × ρ_max` at `ε_KO = 1/2` (`scan`) and with
#     the `ε_KO(r)` profile at `ε_in = 1, 2, 4` (`profile`).
#   * **E1** — `KerrSchild(6/5, 0)`, a valid metric that is not a solution,
#     and **E2** — `translate(KerrSchild(1, 0), (0, δ, 0, 0))`, `δ = h, 4h`,
#     the proxy for a tracking error, over the same scan at `ε_KO = 1/2`
#     (`targets`).
#   * **E0** — `:pasted` with `KerrSchild(6/5, 0)` as its target: a 20 % hard
#     step at `r_1`, and the question whether it gets out — step 8a's shells
#     outside the horizon, on step 8a's uniform 512-block mesh, against
#     `:pasted` with the exact target, at `ε_KO = 1/4, 1/2, 1` (`e0`).
#   * **Controls** — `:frozen`, `:pasted` and `:damped` at `N = 8` and the
#     `:damped` at `N = 6` whose end step 8b replayed (`controls`), and the
#     exact target on the scan's layers (`exact`, added after the first
#     `50 M` runs). E0 again with the `ε_KO(r)` profile is `e0p`.
#
# **`n_L` is the width of the relaxation ramp in cells** (proposed in step
# 8c): finding 1's prediction `n_L ≳ G (10 ρ_max M)^{1/3}` is the width over
# which `ρ` rises from `0` at `r_1` to `ρ_max` — that is where it takes the
# quintic's `ρ(d) ≈ 10 ρ_max (d/n_L)³` at depth `d = G`. `PLAN.md` sets it
# "through `r_0` with `r_1` fixed", which needs the ramp to span the layer:
# `ρ_ramp = 1` and `r_0 = r_1 − n_L h`. `n_L = 2G = 4` is the one exception —
# `check_interior_radii` asks for `r_1 − r_0 ≥ 2(G + 1) h = 6h`, and the
# assertion stays — so it runs in a layer of `6` cells with `ρ_ramp = 2/3`.
# `w` keeps its default ramp over the inner half of the layer. The label
# `nd` is the fixture's own layer (`r_0 = 2/5`, `ρ_ramp = 1/2`: a ramp of
# `4.8` cells), which is step 5's configuration.
#
# **How it runs.** Groups (`sweep`, `e0`, `e0p`, `scan`, `profile`,
# `targets`, `controls`, `exact`, and `long1` … `long8` for the survivors to
# `50 M`) are chosen with
# `calibration=<group>,…` (or `+` between them, which is what a batch job's
# name can carry); the bare section name runs every screen group.
# With a node's worth of threads the runs go to subprocesses of four threads
# each (E0's of sixteen), batched so that each worker compiles as few kernel
# specialisations as it can; each worker prints one line per chunk, so a job
# cut off by its time limit leaves every finished row in its log. A single
# configuration is `runs=<label>,…`, and `t_end=1/2` shortens every run —
# which is how the section is validated locally:
#
#     julia --project=. --threads=4 test/hole_runs.jl calibration \
#         runs=e3-n8-r4-c t_end=1/2

const CAL_Q = 2
const CAL_G = CAL_Q ÷ 2 + 1
const CAL_R1 = T(23 // 20)
const CAL_R0 = T(2 // 5)
const CAL_A = -T(2)                     # E3: 1/M²; the sign, see `CurvatureTarget`
const CAL_NLS = (4, 6, 8, 12)           # 2G, 3G, 4G, 6G cells at q = 2
const CAL_RHOS = (:grid, 10, 4, 1)      # 1/dt, or a fixed ρ_max M
const CAL_EPS_INS = (1, 2, 4)           # the profile's ε_in; ε_out = 1/2
const CAL_EPS0 = (1 // 4, 1 // 2, 1 // 1)
const CAL_OUT_K = 0:8                   # step 8a's shells outside the horizon

# The fixture's finest spacing: level 3 of a box `5 M` wide, `N` points a block.
cal_h(N) = T(5) / (8 * N)

cal_rho_label(ρ) = ρ === :grid ? "rgrid" : "r$(ρ)"
cal_nl_label(nL) = nL === nothing ? "nd" : "n$(nL)"

"""
One configuration of the calibration: every number that distinguishes it,
and a label that names it in the logs, the options and `CODE.md`.
"""
cal_spec(label; exp, variant=:damped, target=:exact, N=8, nL=nothing,
         rho=:grid, eps=1 // 2, eps_in=nothing, mesh=:fixture, t_end=5 // 1,
         chunk=1 // 2, every=2, ref=nothing) =
    (label=label, exp=exp, variant=variant, target=target, N=N, nL=nL,
     rho=rho, eps=eps, eps_in=eps_in, mesh=mesh, t_end=t_end, chunk=chunk,
     every=every, ref=ref)

"""
Every screen configuration, by group, in a fixed order.
"""
function cal_screens()
    specs = Dict{String,Vector{Any}}()
    sw = Any[]
    for tgt in (:exact, :e3), ρ in (:grid, 4), N in (6, 8, 10)
        push!(sw, cal_spec("sw-$(tgt)-$(cal_rho_label(ρ))-N$(N)"; exp=:sweep,
                           target=tgt, N=N, rho=ρ, t_end=3 // 20,
                           chunk=1 // 20, every=0))
    end
    specs["sweep"] = sw
    scan = Any[cal_spec("e3-nd-rgrid-c"; exp=:scan, target=:e3)]
    for nL in CAL_NLS, ρ in CAL_RHOS
        push!(scan, cal_spec("e3-$(cal_nl_label(nL))-$(cal_rho_label(ρ))-c";
                             exp=:scan, target=:e3, nL=nL, rho=ρ))
    end
    specs["scan"] = scan
    prof = Any[]
    for ε in CAL_EPS_INS, nL in CAL_NLS, ρ in CAL_RHOS
        push!(prof, cal_spec("e3-$(cal_nl_label(nL))-$(cal_rho_label(ρ))-p$(ε)";
                             exp=:profile, target=:e3, nL=nL, rho=ρ, eps_in=ε))
    end
    specs["profile"] = prof
    tg = Any[]
    for tgt in (:e1, :e2h, :e2h4)
        push!(tg, cal_spec("$(tgt)-nd-rgrid-c"; exp=:targets, target=tgt))
        for nL in CAL_NLS, ρ in CAL_RHOS
            push!(tg, cal_spec("$(tgt)-$(cal_nl_label(nL))-$(cal_rho_label(ρ))-c";
                               exp=:targets, target=tgt, nL=nL, rho=ρ))
        end
    end
    specs["targets"] = tg
    specs["controls"] = Any[
        cal_spec("ctl-damped8"; exp=:controls),
        cal_spec("ctl-damped6"; exp=:controls, N=6),
        cal_spec("ctl-pasted"; exp=:controls, variant=:pasted),
        cal_spec("ctl-frozen"; exp=:controls, variant=:frozen)]
    e0 = Any[]
    for ε in CAL_EPS0
        tag = "eps$(round(Int, 100 * ε))"
        push!(e0, cal_spec("e0ref-$(tag)"; exp=:e0, variant=:pasted,
                           eps=ε, mesh=:uniform, chunk=1 // 4, every=4))
        push!(e0, cal_spec("e0-$(tag)"; exp=:e0, variant=:pasted, target=:e1,
                           eps=ε, mesh=:uniform, chunk=1 // 4, every=4,
                           ref="e0ref-$(tag)"))
    end
    specs["e0"] = e0
    # E0 again with the `ε_KO(r)` profile across the margin, `ε_out = 1/2`
    # rising to `ε_in`: step 8a's recommendation, tested on the step it was
    # made for (added after the first screens, in which the constant
    # `ε_KO = 1/2` run did not survive the step). Each against a reference
    # with the same profile, so the difference is the step's alone.
    e0p = Any[]
    for ε in (2, 4)
        push!(e0p, cal_spec("e0ref-p$(ε)"; exp=:e0, variant=:pasted,
                            eps_in=ε, mesh=:uniform, chunk=1 // 4, every=4))
        push!(e0p, cal_spec("e0-p$(ε)"; exp=:e0, variant=:pasted, target=:e1,
                            eps_in=ε, mesh=:uniform, chunk=1 // 4, every=4,
                            ref="e0ref-p$(ε)"))
    end
    specs["e0p"] = e0p
    # The exact target on the scan's layers (added after the first 50 M
    # runs): whether what a thick ramp at a physical rate buys is the rate's
    # or needs a wrong target to show — step 5's layer is the grid rate on
    # the exact solution, and the long runs of the inexact targets ended
    # with a smaller error than it.
    specs["exact"] = Any[
        cal_spec("ex-nd-r4-c"; exp=:exact, rho=4),
        cal_spec("ex-n8-r4-c"; exp=:exact, nL=8, rho=4),
        cal_spec("ex-n12-r4-c"; exp=:exact, nL=12, rho=4),
        cal_spec("ex-n12-r10-c"; exp=:exact, nL=12, rho=10),
        cal_spec("ex-n8-rgrid-c"; exp=:exact, nL=8),
        cal_spec("ex-n12-rgrid-c"; exp=:exact, nL=12)]
    return specs
end

# The survivors of the screens, run to `50 M` at `chunk = 1 M` as step 5's
# table was — one list per batch job of eight workers at eight threads,
# written in after the screens. Not every survivor (the screens left 99 of
# the 120): every configuration of the `ε_KO = 1/2` scan, the profile at the
# two thickest ramps and the two physical rates that screened best — and at
# the grid rate on the same two ramps, at `ε_in = 4` — the E1 and E2 targets
# at `n_L = 6` (`ρ_max = 4/M`), `8` and `12`, every `δ = 4h` survivor, and the
# four controls **(proposed in step 8c)**; `CODE.md` has the screens of the
# rest.
const CAL_LONG = Dict{String,Vector{String}}(
    "long1" => ["ctl-damped8", "ctl-damped6", "ctl-pasted", "ctl-frozen",
                "e3-nd-rgrid-c", "e3-n4-rgrid-c", "e3-n4-r10-c", "e3-n4-r4-c"],
    "long2" => ["e3-n4-r1-c", "e3-n6-rgrid-c", "e3-n6-r10-c", "e3-n6-r4-c",
                "e3-n6-r1-c", "e3-n8-rgrid-c", "e3-n8-r10-c", "e3-n8-r4-c"],
    "long3" => ["e3-n8-r1-c", "e3-n12-rgrid-c", "e3-n12-r10-c", "e3-n12-r4-c",
                "e3-n12-r1-c", "e3-n12-rgrid-p4", "e3-n8-rgrid-p4",
                "e3-n12-r4-p1"],
    "long4" => ["e3-n12-r4-p2", "e3-n12-r4-p4", "e3-n12-r1-p1", "e3-n12-r1-p2",
                "e3-n12-r1-p4", "e3-n8-r4-p1", "e3-n8-r4-p2", "e3-n8-r4-p4"],
    "long5" => ["e3-n8-r1-p1", "e3-n8-r1-p2", "e3-n8-r1-p4", "e1-n6-r4-c",
                "e1-n8-r10-c", "e1-n8-r4-c", "e1-n8-r1-c", "e1-n12-rgrid-c"],
    "long6" => ["e1-n12-r10-c", "e1-n12-r4-c", "e1-n12-r1-c", "e2h-n6-r4-c",
                "e2h-n8-rgrid-c", "e2h-n8-r10-c", "e2h-n8-r4-c", "e2h-n8-r1-c"],
    "long7" => ["e2h-n12-rgrid-c", "e2h-n12-r10-c", "e2h-n12-r4-c",
                "e2h-n12-r1-c", "e2h4-n6-r10-c", "e2h4-n6-r4-c", "e2h4-n8-r10-c",
                "e2h4-n8-r4-c"],
    "long8" => ["ex-nd-r4-c", "ex-n8-r4-c", "ex-n12-r4-c", "ex-n12-r10-c",
                "ex-n8-rgrid-c", "ex-n12-rgrid-c"])

cal_long(spec) = merge(spec, (t_end=50 // 1, chunk=1 // 1, every=2))

function cal_all_specs()
    all = Dict{String,Any}()
    for (_, v) in cal_screens(), sp in v
        all[sp.label] = sp
    end
    return all
end

"""
The case, the forest and the evolve! keywords of one configuration.
"""
function cal_setup(sp)
    q = CAL_Q
    G = CAL_G
    h = cal_h(sp.N)
    # The layer: the fixture's own, or a ramp of `n_L` cells through `r_0`.
    r_0, ρ_ramp = if sp.nL === nothing
        CAL_R0, T(1 // 2)
    else
        cells = max(sp.nL, 2 * (G + 1))
        CAL_R1 - cells * h, T(sp.nL) / T(cells)
    end
    bg = SM.KerrSchild(one(T), zero(T))
    target = sp.target === :exact ? nothing :
             sp.target === :e3 ? CurvatureTarget(bg, T; A=CAL_A, r_0=r_0,
                                                 r_1=CAL_R1) :
             sp.target === :e1 ? SM.KerrSchild(T(6 // 5), zero(T)) :
             sp.target === :e2h ? SM.translate(bg, SVector{4,T}(0, h, 0, 0)) :
             sp.target === :e2h4 ? SM.translate(bg, SVector{4,T}(0, 4h, 0, 0)) :
             error("unknown target $(sp.target)")
    case = hole_fixture(T; q=q, variant=sp.variant, r_0=r_0, ρ_ramp=ρ_ramp,
                        ε_KO=T(sp.eps), chunk=T(sp.chunk), target=target)
    sp.eps_in === nothing ||
        (case = with_dissipation(case, horizon_dissipation(case;
                                                           ε_in=T(sp.eps_in))))
    radii = sp.mesh === :uniform ? LEAK_RADII : (T(3), T(3), one(T))
    forest = hole_fixture_forest(T, case; N=sp.N, radii=radii)
    case = with_bounds(case, default_bounds(T; M=1,
                                            r_gate=default_gate(case.interior,
                                                                forest, q)))
    # `:grid` is asked for by name (amended in step 8c′): the driver's
    # default became `4/M` on 2026-09-23, so the grid rate is the option
    # `ρ_max_factor = 1` and no longer what an empty keyword list gets.
    kw = sp.rho === :grid ? (ρ_max_factor=one(T),) : (ρ_max_fixed=T(sp.rho),)
    return case, forest, kw
end

cal_fmt(x) = x === nothing ? "      —  " : Printf.format(Printf.Format("%9.3e"), x)

# The root cause of a run that threw, through the task wrappers a threaded
# kernel puts around it — the `DomainError` and not the `TaskFailedException`.
function cal_root_cause(e)
    while true
        if e isa TaskFailedException
            e = e.task.result
        elseif e isa CompositeException
            e = first(e.exceptions)
        else
            return first(split(sprint(showerror, e), '\n'))
        end
    end
end

"""
One run: `evolve!` with an observer that writes the record's rows it needs
— and the `G`-point shell's constraint and error, the range projection's
running totals and the finder's `M_irr` — at every chunk, into a vector
that survives the run throwing, and prints each row as it is made.
"""
function cal_run(sp; t_end=nothing, reference=nothing, geometry=nothing)
    case, forest, kw = cal_setup(sp)
    q = CAL_Q
    tend = t_end === nothing ? T(sp.t_end) : min(T(t_end), T(sp.t_end))
    rows = NamedTuple[]
    snaps = Matrix{T}[]
    amax = Matrix{T}[]
    seed = Ref{Any}(nothing)
    calls = Ref(0)
    shell = horizon_shell(case)
    function watch(p, t, u)
        calls[] += 1
        tt = T(t)
        gh_constraint!(p, u, tt)
        cn = constraint_norms(p)
        gh_error!(p, u, tt; shell=shell)
        e = error_norms(p)
        v = validity_rows(p, u, tt)
        sh = gh_outside_shell_norms(p, u, tt)
        acc = p.accounting
        mirr = nothing
        if sp.every > 0 && (calls[] - 1) % sp.every == 0
            mirr = try
                o = find_gh_horizon(p, u, tt; N=12, r_seed=T(9 // 5),
                                    hlm=seed[], spin=false)
                seed[] = o.hlm
                o.M_irr
            catch err
                err isa InterruptException && rethrow()
                nothing
            end
        end
        row = (t=Float64(tt), err_l2=e.err_l2, err_linf=e.err_linf,
               gauge_l2=Float64(maximum(cn.gauge_l2)),
               gauge_linf=Float64(maximum(cn.gauge_linf)),
               residual=e.residual, drift=e.drift, M_irr=mirr,
               hits=acc.hits, r_hit=acc.r_max, first_t=acc.first_t,
               first_r=acc.first_r, α_shell=v.min_α_shell,
               detγ_shell=v.min_detγ_shell, Π_shell=v.max_Π_shell,
               h_shell=v.max_h_shell, α_layer=v.min_α_layer,
               Π_layer=v.max_Π_layer, sh_gauge_l2=sh.gauge_l2,
               sh_gauge_linf=sh.gauge_linf, sh_err_l2=sh.err_l2,
               sh_err_linf=sh.err_linf, ρ_max=Float64(p.interior.ρ_max))
        push!(rows, row)
        say("   [%s] t=%6.2f err=%s/%s C=%s shC=%s/%s shE=%s res=%s " *
            "αsh=%s Πsh=%s hits=%d M_irr=%s", sp.label, row.t, cal_fmt(row.err_l2),
            cal_fmt(row.err_linf), cal_fmt(row.gauge_l2), cal_fmt(row.sh_gauge_l2),
            cal_fmt(row.sh_gauge_linf), cal_fmt(row.sh_err_l2),
            cal_fmt(row.residual), cal_fmt(row.α_shell), cal_fmt(row.Π_shell),
            row.hits, mirr === nothing ? "—" :
                      Printf.format(Printf.Format("%.6f"), mirr))
        flush(stdout)
        # E0: the state at step 8a's shells outside the horizon, against
        # the reference's at the same chunk.
        if geometry !== nothing
            ua = statearray(u, p.U)
            if reference === nothing
                snap = Matrix{T}(undef, 10, length(geometry.points))
                for (n, (i, j, k, b)) in enumerate(geometry.points), c in 1:10
                    snap[c, n] = ua[i, j, k, c, b]
                end
                push!(snaps, snap)
            else
                ref = reference.snaps[calls[]]
                a = zeros(T, length(LEAK_SHELLS), 3)
                for (n, (i, j, k, b)) in enumerate(geometry.points)
                    δ = maximum(abs(ua[i, j, k, c, b] - ref[c, n]) for c in 1:10)
                    m = geometry.shell[n]
                    a[m, 1] = max(a[m, 1], δ)
                    geometry.axis[n] && (a[m, 2] = max(a[m, 2], δ))
                    geometry.diag[n] && (a[m, 3] = max(a[m, 3], δ))
                end
                push!(amax, a)
            end
        end
        return nothing
    end
    t0 = time()
    failure = nothing
    out = try
        evolve!(T, case; forest=forest, q=q,
                ops=Operators(prolongation=q + 2, restriction=q + 2),
                t_end=tend, cfl=T(1 // 5), observer=watch, kw...)
    catch err
        err isa InterruptException && rethrow()
        failure = cal_root_cause(err)
        nothing
    end
    wall = time() - t0
    reached = isempty(rows) ? 0.0 : rows[end].t
    say("   done [%s] reached %.2f M of %.2f in %.0f s%s", sp.label, reached,
        tend, wall, failure === nothing ? "" : ", then threw: " * failure)
    flush(stdout)
    return (label=sp.label, spec=sp, reached=reached, t_end=Float64(tend),
            failure=failure, wall=wall,
            nsteps=out === nothing ? nothing : out.nsteps,
            h=Float64(cal_h(sp.N)), rows=rows, snaps=snaps, amax=amax)
end

# The size of E0's step at `r_1`: the largest `|δh_ab|` between the target
# and the truth on the sphere, over a few directions — what the shells'
# `A_k` are divided by, as step 8a divided by its ripple's amplitude.
function cal_step_amplitude()
    tgt = SM.KerrSchild(T(6 // 5), zero(T))
    bg = SM.KerrSchild(one(T), zero(T))
    dirs = (SVector{3,T}(0, 0, 1), SVector{3,T}(1, 0, 0),
            SVector{3,T}(1, 1, 1) / sqrt(T(3)), SVector{3,T}(3, 4, 12) / 13)
    return maximum(dirs) do n
        x = Tuple(CAL_R1 * n)
        maximum(abs.(state_tuple(tgt, zero(T), x)[1:10] .-
                     state_tuple(bg, zero(T), x)[1:10]))
    end
end

"""
The runs of one worker, in order: an E0 pair is its reference and then its
run against it; everything else is one run.
"""
function cal_worker(labels; long=false, t_end=nothing)
    all = cal_all_specs()
    res = Any[]
    geometry = nothing
    refs = Dict{String,Any}()
    for l in labels
        sp = long ? cal_long(all[l]) : all[l]
        if sp.exp === :e0
            geometry === nothing && (geometry = leak_geometry(CAL_Q))
            if sp.ref === nothing
                r = cal_run(sp; t_end=t_end, geometry=geometry)
                refs[sp.label] = r
                push!(res, merge(r, (snaps=Matrix{T}[],)))
            else
                ref = get(refs, sp.ref, nothing)
                ref === nothing && (ref = cal_run(all[sp.ref]; t_end=t_end,
                                                  geometry=geometry))
                push!(res, cal_run(sp; t_end=t_end, reference=ref,
                                   geometry=geometry))
            end
        else
            push!(res, cal_run(sp; t_end=t_end))
        end
    end
    return res
end

"""
The workers of a batch job, each a subprocess with `threads` threads, all at
once; their logs and results beside the job's own log when there is one.
"""
function cal_fanout(batches; tag, long, t_end)
    base = isdir("out") ? joinpath("out", "calibration") : mktempdir()
    dir = mkpath(joinpath(base, tag))
    println("   worker logs and results in ", abspath(dir))
    project = dirname(Base.active_project())
    procs = map(enumerate(batches)) do (n, (labels, nt))
        out = joinpath(dir, "worker-$n.jls")
        log = joinpath(dir, "worker-$n.log")
        te = t_end === nothing ? "" : "t_end=$(leak_spell(t_end))"
        cmd = `$(Base.julia_cmd()) --project=$project --threads=$nt
               $(abspath(@__FILE__)) calibration worker=1
               runs=$(join(labels, ',')) long=$(Int(long)) out=$out $te`
        io = open(log, "w")
        (run(pipeline(cmd; stdout=io, stderr=io); wait=false), out, log, io)
    end
    return map(procs) do (proc, out, log, io)
        wait(proc)
        close(io)
        if !success(proc) || !isfile(out)
            println("   a worker failed; the end of its log ($log):")
            foreach(l -> println("     ", l), last(readlines(log), 30))
            return Any[]
        end
        foreach(l -> startswith(l, "   done") && println(l), readlines(log))
        return deserialize(out)
    end
end

# The summary of one run: the last row, and the extremes over the run.
function cal_summary(r)
    isempty(r.rows) && return nothing
    last_ = r.rows[end]
    first_ = r.rows[1]
    mirr = [x.M_irr for x in r.rows if x.M_irr !== nothing]
    ext(f, key) = (vals = [getfield(x, key) for x in r.rows
                           if getfield(x, key) !== nothing];
                   isempty(vals) ? nothing : f(vals))
    return (last=last_, first=first_,
            α_shell=ext(minimum, :α_shell), detγ_shell=ext(minimum, :detγ_shell),
            Π_shell=ext(maximum, :Π_shell), α_layer=ext(minimum, :α_layer),
            sh_err_max=ext(maximum, :sh_err_linf),
            M_irr0=isempty(mirr) ? nothing : first(mirr),
            M_irr=isempty(mirr) ? nothing : last(mirr),
            M_irr_drift=isempty(mirr) ? nothing : maximum(abs.(mirr .- 1)))
end

function cal_table(results)
    println("| label | reached | end | err L2 | err L∞ | shell C L2 | shell C L∞ " *
            "| shell err L∞ (max) | residual | drift | M_irr − 1 (max) | hits " *
            "| min α shell | min det γ shell | max |Π| shell |")
    for r in results
        s = cal_summary(r)
        s === nothing && (say("| %s | 0 | %s | | | | | | | | | | | | |", r.label,
                              r.failure === nothing ? "?" : "threw");
                          continue)
        l = s.last
        say("| %s | %.2f | %s | %s | %s | %s | %s | %s (%s) | %s | %s | %s | %d " *
            "| %s | %s | %s |", r.label, r.reached,
            r.failure === nothing ? "ok" : "threw", cal_fmt(l.err_l2),
            cal_fmt(l.err_linf), cal_fmt(l.sh_gauge_l2), cal_fmt(l.sh_gauge_linf),
            cal_fmt(l.sh_err_linf), cal_fmt(s.sh_err_max), cal_fmt(l.residual),
            cal_fmt(l.drift), cal_fmt(s.M_irr_drift), l.hits,
            cal_fmt(s.α_shell), cal_fmt(s.detγ_shell), cal_fmt(s.Π_shell))
    end
end

function cal_report(results, all)
    byl = Dict(r.label => r for r in results)
    println("\n-- every run: the last chunk, and the extremes over the run --")
    cal_table(sort(results; by=r -> r.label))
    for r in results
        r.failure === nothing || say("   %s threw at %.2f M: %s", r.label,
                                     r.reached, r.failure)
    end
    # The E3 sweep: the order of the G-point shell's C_a at 3/20 M.
    sw = [r for r in results if r.spec.exp === :sweep]
    if !isempty(sw)
        println("\n-- E3 sweep to 3/20 M: the G-point shell outside r_1, and " *
                "the masked error --")
        println("| target | ρ_max | N | h | shell C L2 | shell C L∞ | shell err L2 " *
                "| masked err L2 | residual |")
        for tgt in (:exact, :e3), ρ in (:grid, 4)
            rs = sort([r for r in sw if r.spec.target === tgt && r.spec.rho == ρ
                       && !isempty(r.rows)]; by=r -> r.spec.N)
            isempty(rs) && continue
            for r in rs
                l = r.rows[end]
                say("| %s | %s | %d | %.5f | %s | %s | %s | %s | %s |", tgt, ρ,
                    r.spec.N, r.h, cal_fmt(l.sh_gauge_l2), cal_fmt(l.sh_gauge_linf),
                    cal_fmt(l.sh_err_l2), cal_fmt(l.err_l2), cal_fmt(l.residual))
            end
            if length(rs) ≥ 2
                hs = [r.h for r in rs]
                rate(k) = convergence_rate(hs, [getfield(r.rows[end], k) for r in rs])
                say("| %s | %s | **rate** | | %.2f | %.2f | %.2f | %.2f | %.2f |", tgt,
                    ρ, rate(:sh_gauge_l2), rate(:sh_gauge_linf), rate(:sh_err_l2),
                    rate(:err_l2), rate(:residual))
            end
        end
    end
    # E0: what reaches the shells outside the horizon.
    e0 = [r for r in results if r.spec.exp === :e0 && r.spec.ref !== nothing]
    if !isempty(e0)
        Astep = cal_step_amplitude()
        iout = [findfirst(==(k), LEAK_SHELLS) for k in CAL_OUT_K]
        d = (T(2) - CAL_R1) / cal_h(8)
        say("\n-- E0: :pasted onto KerrSchild(6/5, 0), the step at r_1 = %.3f " *
            "(%.2f cells below the horizon) of |δh| = %.4f; A_k/A_step in the " *
            "shells [r_h + k h, r_h + (k+1) h] --", CAL_R1, d, Astep)
        for r in e0
            isempty(r.amax) && continue
            peak = reduce((a, b) -> max.(a, b), r.amax) ./ Astep
            say("   %s (ε_KO = %.2f), reached %.2f M:", r.label, r.spec.eps,
                r.reached)
            println("     A_k/A, k = 0 … 8 (all)  : ", leak_row(peak[iout, 1]))
            println("     A_k/A, k = 0 … 8 (axis) : ", leak_row(peak[iout, 2]))
            println("     A_k/A, k = 0 … 8 (diag) : ", leak_row(peak[iout, 3]))
            k0 = iout[1]
            println("     A_0(t)/A every 1/2 M    : ",
                    leak_row([a[k0, 1] / Astep for a in r.amax[1:2:end]]))
            say("     e-folds per cell outside: %.3f (all), %.3f (axis)",
                leak_fit(CAL_OUT_K, peak[iout, 1]), leak_fit(CAL_OUT_K, peak[iout, 2]))
        end
    end
end

const CAL_ALL_SCREENS = ["sweep", "controls", "scan", "profile", "targets", "e0",
                         "e0p", "exact"]

if "calibration" in SECTIONS || haskey(OPTIONS, "calibration")
    t_end_opt = haskey(OPTIONS, "t_end") ? only(leak_option("t_end", [5 // 1])) :
                nothing
    if haskey(OPTIONS, "worker")
        labels = String.(split(OPTIONS["runs"], ','))
        res = cal_worker(labels; long=get(OPTIONS, "long", "0") == "1",
                         t_end=t_end_opt)
        serialize(OPTIONS["out"], res)
    else
        screens = cal_screens()
        all = cal_all_specs()
        groups = haskey(OPTIONS, "calibration") ?
                 String.(split(OPTIONS["calibration"], r"[,+]")) : CAL_ALL_SCREENS
        # (label, long) pairs in the order the groups name them.
        runs = Tuple{String,Bool}[]
        if haskey(OPTIONS, "runs")
            append!(runs, [(String(l), false) for l in split(OPTIONS["runs"], ',')])
        else
            for g in groups
                if haskey(CAL_LONG, g)
                    append!(runs, [(l, true) for l in CAL_LONG[g]])
                else
                    haskey(screens, g) || error("no calibration group $g")
                    append!(runs, [(sp.label, false) for sp in screens[g]])
                end
            end
        end
        println("\n=== (8) calibration: the layer against an inexact target ===")
        nt = Threads.nthreads()
        say("%d runs (%s) at %d threads%s", length(runs),
            join(unique(haskey(OPTIONS, "runs") ? ["runs"] : groups), ", "), nt,
            t_end_opt === nothing ? "" : ", t_end ≤ $(Float64(t_end_opt)) M")
        t0 = time()
        results = if nt ≥ 16
            # E0 pairs to workers of sixteen threads, one pair each; the rest
            # to workers of four, in contiguous batches of the fixed order so
            # that each worker compiles one or two kernel specialisations.
            e0pairs = [[l for (l, _) in runs if all[l].exp === :e0 &&
                        (l == lab || all[l].ref == lab)]
                       for (lab, _) in runs if all[lab].exp === :e0 &&
                       all[lab].ref === nothing]
            rest = [(l, lg) for (l, lg) in runs if all[l].exp !== :e0]
            # Four threads a screen; eight a run to `50 M`, whose hour on a
            # node would not hold it at four.
            wt = any(lg for (_, lg) in runs) ? 8 : 4
            nfixed = max(0, (nt - 16 * length(e0pairs)) ÷ wt)
            nw = max(1, min(nfixed, length(rest)))
            per = cld(length(rest), nw)
            longs = unique(lg for (_, lg) in rest)
            length(longs) ≤ 1 || error("a batch is all screens or all long runs")
            batches = [(pair, 16) for pair in e0pairs]
            for i in 1:nw
                chunk = rest[(i - 1) * per + 1:min(i * per, length(rest))]
                isempty(chunk) || push!(batches, ([l for (l, _) in chunk], wt))
            end
            say("   %d workers: %d E0 pairs at 16 threads, %d batches of ≤ %d " *
                "at %d", length(batches), length(e0pairs), length(batches) -
                length(e0pairs), per, wt)
            tag = haskey(OPTIONS, "runs") ? "runs" : join(groups, "+")
            reduce(vcat, cal_fanout(batches; tag=tag,
                                    long=!isempty(longs) && only(longs),
                                    t_end=t_end_opt); init=Any[])
        else
            longs = unique(lg for (_, lg) in runs)
            length(longs) ≤ 1 || error("a batch is all screens or all long runs")
            cal_worker([l for (l, _) in runs]; long=!isempty(longs) && only(longs),
                       t_end=t_end_opt)
        end
        say("%d runs in %.0f s", length(results), time() - t0)
        cal_report(results, all)
    end
end

# --- (9) the tracked geometry against the sphere (step 8d) -------------------
#
# The claim of `test/tracking_tests.jl`'s last testset, carried from `0.15 M`
# to a few `M`: on the static hole the layer that follows the found horizon is
# step 5's layer to the tracking, and the track stays within a cell of the
# analytic center. Both runs carry the range projection and the finder every
# chunk; the shell is the `G`-point shell of each run's own geometry.
if "tracked" in SECTIONS || haskey(OPTIONS, "tracked")
    println("\n=== (9) the tracked geometry against step 5's sphere, " *
            "Kerr-Schild a = 0, the step-5 fixture's mesh, m = 10 ===")
    let q = 2, h = T(5 // 64), chunk = T(1 // 4),
        t_end = T(only(leak_option("tracked", [5 // 1])))

        ops = Operators(prolongation=q + 2, restriction=q + 2)
        common = (halfwidth=T(5 // 2), chunk=chunk,
                  horizon=Horizon(T; every=1, N=12, spin=false),
                  bounds=default_bounds(T; M=1, r_gate=T(9 // 10)))
        cases = (tracked=kerr_schild_case(T; interior=FittedSpec(T; margin=10),
                                          common...),
                 sphere=kerr_schild_case(T; r_0=2 - 18h, r_1=2 - 10h,
                                         margin=10, w_ramp=T(1 // 2),
                                         ρ_ramp=one(T), common...))
        outs = Dict{Symbol,Any}()
        shell = Dict{Symbol,Vector{Float64}}()
        for (name, case) in pairs(cases)
            shell[name] = Float64[]
            watch(p, t, u) =
                push!(shell[name], gh_outside_shell_norms(p, u, t).gauge_l2)
            t0 = time()
            outs[name] = evolve!(T, case; forest=shells(case, 8, (T(3), T(3), one(T))),
                                 q=q, ops=ops, t_end=t_end, observer=watch)
            say("%-8s %d steps, %.0f s", String(name), outs[name].nsteps,
                time() - t0)
        end
        f, s = outs[:tracked], outs[:sphere]
        println("| t/M | masked L2, tracked | sphere | shell C_a L2, tracked | " *
                "sphere | M_irr, tracked | sphere | track_offset (cells) | " *
                "prediction (cells) | hits |")
        for i in eachindex(f.records)
            (i == 1 || iszero(mod(i - 1, 4)) || i == length(f.records)) || continue
            rf, rs = f.records[i], s.records[i]
            say("| %.2f | %.4e | %.4e | %.4e | %.4e | %.7f | %.7f | %.2e | %.2e | %d/%d |",
                rf.t, rf.err_l2, rs.err_l2, shell[:tracked][i], shell[:sphere][i],
                rf.M_irr, rs.M_irr, rf.track_offset,
                something(rf.track_prediction, NaN), rf.bounds_hits, rs.bounds_hits)
        end
        say("max track_offset %.2e cells, max prediction %.2e cells, %d re-samples; " *
            "at the end r_in = %.6f, r_out = %.6f, margin %.2f e-folds",
            maximum(r -> r.track_offset, f.records),
            maximum(r -> something(r.track_prediction, 0.0), f.records),
            f.nresamples, f.records[end].layer_r_in, f.records[end].layer_r_out,
            f.records[end].margin_efolds)
    end
end

# --- (10) the fitted variant (step 8e) --------------------------------------
#
# `CODE.md`, "The fitted target": the kernel half's three trials. Each row
# says what it is; a run that ends says when and why, as a row.
function fitted_rows()
    rows = String.(split(get(OPTIONS, "fitted", "fixture,boosted,harmonic"), ','))
    return rows
end

if "fitted" in SECTIONS || haskey(OPTIONS, "fitted")
    let
    frows = fitted_rows()
    q, G = 2, 2
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    if "fixture" in frows
        println("\n=== (10a) the fitted fixture against :damped, and its initial data ===")
        t_end = T(parse(Float64, get(OPTIONS, "t_end", "1")))
        variants = (("damped", :damped, (;)),
                          ("fitted (decided: cont 1, r_1)", :fitted, (;)),
                          ("fitted cont 2", :fitted, (fit_initial_cont=2,)),
                          ("fitted, fit below the core surface", :fitted,
                           (fit_initial_depth=8 * T(5 // 64),)),
                          ("fitted, fit below 3h", :fitted,
                           (fit_initial_depth=3 * T(5 // 64),)))
        println("| run | t/M | masked L2 | L∞ | C_a L2 | layer residual | fit_residual | hits |")
        for (label, variant, kw) in variants
            case = fitted_fixture(T; variant=variant)
            t0 = time()
            out = evolve!(T, case; forest=hole_fixture_forest(T, case; N=8), q=q,
                                ops=ops, t_end=t_end, kw...)
            for r in out.records
                (r.t == 0 || isapprox(r.t, 0.1) || isapprox(r.t, 0.5) ||
                 r === out.records[end]) || continue
                say("| %s | %.2f | %.3e | %.3e | %.3e | %.3g | %s | %d |", label, r.t,
                    r.err_l2, r.err_linf, r.gauge_l2, r.residual,
                    r.fit_residual === nothing ? "—" : Printf.format(Printf.Format("%.3g"), r.fit_residual),
                    r.bounds_hits)
            end
            sh = gh_outside_shell_norms(out)
            say("|   shell at the end | | %.3e | %.3e | C_a %.3e | | | (%.0f s) |", sh.err_l2,
                sh.err_linf, sh.gauge_l2, time() - t0)
        end
    end
    if "boosted" in frows
        println("\n=== (10b) a moving seed: boost(Harmonic(1, 0), 0.3 x̂), h = 5/128, :fitted and :damped ===")
        bg = SM.boost(SM.Harmonic(one(T), zero(T)), SVector{3,T}(T(3 // 10), 0, 0))
        for variant in (:fitted, :damped)
            case = hole_case(T, bg; halfwidth=T(5 // 2), chunk=T(1 // 20),
                             interior=FittedSpec(T; variant=variant, margin=8),
                             horizon=Horizon(T; every=1, N=12, spin=false))
            forest = hole_forest(T, case; N=8, roots=1,
                                 radii=(T(10), T(10), T(3 // 2), one(T)))
            say(":%s — mesh: %d leaves, levels %s; the case's velocity %s (hole_velocity: −v)",
                String(variant), nleaves(forest), string(forest_levels(forest)),
                string(Tuple(case.center.v)))
            t0 = time()
            out = try
                evolve!(T, case; forest=forest, q=q, ops=ops,
                        t_end=T(parse(Float64, get(OPTIONS, "t_end_boosted", "0.1"))))
            catch e
                println("the run ended: ", first(split(sprint(showerror, e), '\n')))
                nothing
            end
            out === nothing && continue
            println("| t/M | masked L2 | L∞ | C_a L2 | find | track center | v_est | offset (cells) | prediction (cells) | fit_valid | fit_residual | refills | hits |")
            for r in out.records
                say("| %.3f | %.3e | %.3e | %.3e | %s | (%.5f, %.1e, %.1e) | (%.4f, %.1e, %.1e) | %.2e | %s | %s | %s | %s | %s |",
                    r.t, r.err_l2, r.err_linf, r.gauge_l2, string(r.horizon_success),
                    r.track_center..., r.track_velocity..., r.track_offset,
                    string(r.track_prediction), string(r.fit_valid),
                    string(r.fit_residual), string(r.fit_refills),
                    string(r.bounds_hits))
            end
            say("%d steps in %.0f s; fits %s", out.nsteps, time() - t0,
                string(out.fit_cost))
        end
    end
    if "harmonic" in frows
        println("\n=== (10c) harmonic Kerr a = 9/10: the first :fitted initial data, m = 4, h = 5/256 ===")
        bg = SM.Harmonic(one(T), T(9 // 10))
        spec = FittedSpec(T; variant=:fitted, margin=4, lmax_shape=12,
                                lmax_fit=parse(Int, get(OPTIONS, "lmax_fit", "8")))
        case = hole_case(T, bg; halfwidth=T(5 // 4), chunk=T(1 // 400),
                               interior=spec,
                               horizon=Horizon(T; every=1, N=16, spin=false))
        forest = hole_forest(T, case; N=8, roots=1,
                                   radii=(T(10), T(8 // 5), T(13 // 10), one(T)))
        tr = seed_track(case, 0)
        geom = with_ρ_max(fitted_interior(spec, tr, forest, G; t=0, n_L=8), T(4))
        say("mesh: %d leaves, levels %s; h = %.5f, r_1 from %.4f (axis) to %.4f (equator), core from %.4f",
            nleaves(forest), string(forest_levels(forest)), geom.h,
            geom.r_in - geom.offset, geom.r_out - geom.offset,
            geom.r_in - geom.offset - geom.thickness)
        bd = derive_target_bounds(T, bg, geom; t=0, L=spec.lmax_fit)
        t0 = time()
        fit = build_fit(analytic_sampler(bg, 0.0; δ=geom.h / 8), geom, spec;
                              cont=1, bounds=bd)
        say("initial fit (cont = 1, L = %d): valid %s, value residual %.3g, sweep min λ(γ) %.3g, min α %.3g, %d hits; %.2f s",
            spec.lmax_fit, string(fit.valid), fit.residual.value, fit.sweep.min_λ,
            fit.sweep.min_α, fit.sweep.hits, time() - t0)
        say("target bounds: α in [%.3g, %.3g], λ in [%.3g, %.4g], |β| ≤ %.3g, K ≤ %.4g",
            bd.α_min, bd.α_max, bd.λ_min, bd.λ_max, bd.β_max, bd.K_max)
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3), backend=CPU())
        p = refill_target(GHProblem(fs, GhostSchedule(fs, ops), case; q=q, interior=geom,
                                          target=target_cache(fs), fits=(fit, nothing)),
                                zero(T))
        u = statevector(fs)
        map_blocks!(TreeGeneralizedHarmonic.fitted_state_kernel!, fs, statearray(u, fs),
                    p.target.work, p.origins, p.spacings, bg, p.interior, zero(T), zero(T),
                    zero(T))
        A = statearray(u, fs)
        worst = (detγ=Inf, α=Inf, λ=Inf)
        nbad = 0
        for b in 1:nblocks(fs), k in 1:8, j in 1:8, i in 1:8
            hv = SVector{10}(ntuple(v -> A[i, j, k, v, b], 10))
            Πv = SVector{10}(ntuple(v -> A[i, j, k, 10 + v, b], 10))
            d, α, _, _ = state_validity(hv, Πv)
            λ, _ = sym_eigen3(SMatrix{3,3}(1 + hv[5], hv[6], hv[7], hv[6], 1 + hv[8],
                                                 hv[9], hv[7], hv[9], 1 + hv[10]))
            worst = (detγ=min(worst.detγ, d), α=min(worst.α, α), λ=min(worst.λ, minimum(λ)))
            nbad += !(d > 0 && α > 0 && minimum(λ) > 0 && all(isfinite, hv) && all(isfinite, Πv))
        end
        say("initial data: %d points, %d non-finite values, %d not a metric; min det γ %.3g, min α %.3g, min λ(γ) %.3g",
            length(u) ÷ 20, count(!isfinite, u), nbad, worst.detγ, worst.α, worst.λ)
        du = similar(u)
        gh_rhs!(du, u, p, zero(T))
        t1 = time()
        gh_rhs!(du, u, p, zero(T))
        say("one right-hand side: %.2f s, %d non-finite, max |du| = %.3g", time() - t1,
            count(!isfinite, du), maximum(abs, du))
        reached = Ref(zero(T))
        out = try
            evolve!(T, case; forest=forest, q=q, ops=ops,
                    t_end=T(parse(Float64, get(OPTIONS, "t_end_harmonic", "0.05"))),
                    observer=(p, t, u) -> (reached[] = T(t)))
        catch e
            say("the run ended after t = %.4f: %s", reached[],
                first(split(sprint(showerror, TreeGeneralizedHarmonic.unwrap_task_failure(e)), '\n')))
            nothing
        end
        if out !== nothing
            for r in out.records
                say("| %.4f | masked L2 %.3e | min α layer %.3g | min det γ layer %.3g | min α shell %.3g | fit_valid %s |",
                    r.t, r.err_l2, r.min_α_layer, r.min_detγ_layer, r.min_α_shell,
                    string(r.fit_valid))
            end
        end
    end
    end
end

# --- (11) the measurement matrix: the generic interior (step 8f) -----------
#
# `PLAN.md`'s step 8f: every hole this package has, on the tracked geometry,
# with the `:fitted` target against the analytic `:damped` layer where the
# chart allows one, and the controls that say what the fit is worth — the
# snapshot target (the state at the chunk's start, no fit), a Kerr target
# built from the finder's own `M_ch`, `J` and origin, a hand-over from
# `:damped` to `:fitted`, and a track that coasts. Its table is in `CODE.md`,
# "The generic interior: the measurement matrix (step 8f)".
#
# **How it runs.** A run is a label (`ks0-fitted`, `boost-coast`, …) and a
# group of runs is a batch job: `generic=<group or label>,…` (or `+`), with
# `t_end=<t>` shortening every run — which is how each row is validated
# locally before it is sent to Symmetry:
#
#     julia --project=. --threads=4 test/hole_runs.jl generic=ks0-fitted t_end=1/2
#
# The groups are `ks0` (Kerr-Schild `a = 0` to `50 M`: six rows and the
# hand-over, 120 blocks each), `ks9` (Kerr-Schild `a = 9/10` at `h = 5/128`,
# 1128 blocks), `harm` (harmonic `a = 0` and `a = 7/10`), `boost` (a boosted
# harmonic hole crossing a fixed fine region, and the coasting track), and
# `probe` (harmonic `a = 9/10`, host-side and in-process). With sixteen or
# more threads the runs of a group go to subprocess workers of `threads=<n>`
# threads each (default: the node's threads over the runs); each worker prints
# one line per chunk, so a job cut off by its time limit leaves every finished
# row in its log.
#
# Every row records, per chunk: the masked error (L2, L∞), the masked gauge
# constraint, the `G`-point shell's `C_a` above the offset surface, the `C_a`
# in step 8a's shells `[r_h + k h, r_h + (k+1) h)` outside the tracked horizon
# (`k = 0, 2, 4`), the layer residual (against the truth for the analytic
# variants, against the target for `:fitted`), the drift, the projection's
# hits and outermost radius, `fit_valid` and `fit_residual`, the find's `A`,
# `M_irr`, `J`, `M_ch` and the track's offset from the analytic center.

const GEN_Q = 2
const GEN_G = GEN_Q ÷ 2 + 1
const GEN_HSHELLS = (0, 2, 4)

gen_spec(label; chart, variant=:fitted, t_end, chunk, spec=(;), kw=(;),
         kerr=false, coast=nothing) =
    (label=label, chart=chart, variant=variant, t_end=t_end, chunk=chunk,
     spec=spec, kw=kw, kerr=kerr, coast=coast)

# The rows, by group. `kw` goes to `evolve!`; `spec` to `FittedSpec`.
function gen_groups()
    d = Dict{String,Vector{Any}}()
    ks = (t_end=50 // 1, chunk=1 // 2)
    dn = (fit_initial_depth=8 * T(5 // 64),)
    d["ks0"] = Any[
        gen_spec("ks0-damped"; chart=:ks0, variant=:damped, ks...),
        gen_spec("ks0-fitted"; chart=:ks0, kw=dn, ks...),
        gen_spec("ks0-fitted-r1"; chart=:ks0, ks...),
        gen_spec("ks0-fitted-pi"; chart=:ks0, kw=dn, spec=(fit_tilde=false,), ks...),
        gen_spec("ks0-snapshot"; chart=:ks0, kw=(dn..., target_source=:snapshot), ks...),
        gen_spec("ks0-kerr"; chart=:ks0, variant=:damped, kerr=true, ks...),
        gen_spec("handover"; chart=:ks0, kw=(handover=5 // 1,), ks...)]
    k9 = (t_end=50 // 1, chunk=1 // 2)
    d["ks9"] = Any[
        gen_spec("ks9-damped"; chart=:ks9, variant=:damped, spec=(margin=5,), k9...),
        gen_spec("ks9-fitted"; chart=:ks9, spec=(margin=5,),
                 kw=(fit_initial_depth=8 * T(5 // 128),), k9...),
        gen_spec("ks9-fitted-m8"; chart=:ks9, spec=(margin=8,),
                 kw=(fit_initial_depth=8 * T(5 // 128),), k9...)]
    d["harm"] = Any[
        gen_spec("h0-fitted"; chart=:h0, kw=(fit_initial_depth=8 * T(5 // 128),),
                 t_end=10 // 1, chunk=1 // 4),
        gen_spec("h0-damped"; chart=:h0, variant=:damped, t_end=10 // 1, chunk=1 // 4),
        gen_spec("h7-fitted"; chart=:h7, kw=(fit_initial_depth=8 * T(5 // 256),),
                 t_end=10 // 1, chunk=1 // 4),
        gen_spec("h7c-fitted"; chart=:h7c, kw=(fit_initial_depth=3 * T(5 // 128),),
                 t_end=10 // 1, chunk=1 // 4),
        gen_spec("h7c-fitted-r1"; chart=:h7c, t_end=10 // 1, chunk=1 // 4)]
    bo = (t_end=5 // 1, chunk=1 // 4)
    d["boost"] = Any[
        gen_spec("boost-fitted"; chart=:boost, kw=(fit_initial_depth=8 * T(5 // 128),), bo...),
        gen_spec("boost-damped"; chart=:boost, variant=:damped, bo...),
        gen_spec("boost-sphere"; chart=:boost_sphere, variant=:damped, bo...),
        gen_spec("boost-coast"; chart=:boost, kw=(fit_initial_depth=8 * T(5 // 128),),
                 spec=(max_misses=6,), coast=(4, 9), bo...)]
    return d
end

function gen_all_specs()
    all = Dict{String,Any}()
    for (_, v) in gen_groups(), sp in v
        all[sp.label] = sp
    end
    return all
end

# The fine region of the boosted rows: every block of a level within `R` of
# the segment the analytic center sweeps from `t = 0` to `t_end` — a capsule
# (added in step 8f, over `regrid = true`, which a static-mesh study of the
# tracked geometry does not need).
function gen_capsule_forest(case; N, radii, t_end)
    forest = gh_forest(T, case; N=N, roots=1)
    ts = range(zero(T), T(t_end); length=33)
    cs = [Tuple(center_at(case.center, t)) for t in ts]
    for (ℓ, R) in enumerate(radii)
        targets = filter(forest.leaves) do k
            level(k) == ℓ - 1 &&
                any(c -> TreeGeneralizedHarmonic._box_meets_ball(block_extent(T, forest, k),
                                                                 c, T(R)), cs)
        end
        refine!(forest, targets)
        balance!(forest)
    end
    return forest
end

# The seed geometry of a tracked case on a mesh, for its bounds and gate.
function gen_seed_geometry(case, forest)
    spec = case.interior
    n_L = spec.n_L > 0 ? spec.n_L :
          layer_cells(GEN_G, default_relaxation_rate(case), hole_mass(case.background))
    return fitted_interior(spec, seed_track(case, zero(T)), forest, GEN_G;
                           t=zero(T), n_L=n_L)
end

# A Kerr target from the finder (`PLAN.md`'s idea 4): `M_ch`, `J` and the
# found origin of the initial data, the spin along the found axis when it is
# resolved (`|J|/M_ch² > 10⁻⁴`) and none otherwise.
function gen_kerr_target(case, forest)
    q = GEN_Q
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    geom = gen_seed_geometry(case, forest)
    fs = FieldSet{T}(forest, 20; G=GEN_G, centering=vertexcentered(3), backend=CPU())
    p = GHProblem(fs, GhostSchedule(fs, ops), case; q=q, interior=geom)
    fill_exact!(fs, case, zero(T); interior=geom)
    u = statevector(fs)
    gather!(u, fs)
    o = find_gh_horizon(p, u, zero(T); N=case.horizon.N, spin=true)
    a = o.J / o.M_ch
    abs(a) / o.M_ch > 1e-4 && error("a spinning Kerr target needs the rotation " *
                                    "to the found axis, which is not built")
    tgt = SM.translate(SM.KerrSchild(T(o.M_ch), zero(T)),
                       SVector{4,T}(0, o.origin[1], o.origin[2], o.origin[3]))
    say("   Kerr target from the find: M_ch = %.7f, J = %.2e, origin (%.2e, %.2e, %.2e)",
        o.M_ch, o.J, o.origin...)
    return tgt
end

# The range projection's ranges for a chart whose data exceed step 8b's
# Kerr-Schild proposal: `derive_target_bounds`' rule — four times the largest
# `α`, `λ(γ)`, `|β|`, `|(α/√γ)Π|`, a quarter of the smallest — over the
# analytic data on the whole layer, from the offset surface down to the core
# surface, and not on the surface alone, so that the instrument is passive on
# healthy data as it is on the fixture (a non-finite sample, a point of a
# chart's singular set, is skipped).
function gen_layer_bounds(bg, geom, gate)
    c = center_at(geom.center, zero(T))
    αlo, αhi, λlo, λhi, βhi, Khi = Inf, 0.0, Inf, 0.0, 0.0, 0.0
    for n in fit_directions(12), k in 0:ceil(Int, geom.thickness / geom.h)
        nn = SVector{3,T}(n)
        r = shape_radius(geom, nn) - geom.offset - k * geom.h
        u = SVector{20,T}(state_tuple(bg, zero(T), Tuple(c + r * nn)))
        all(isfinite, u) || continue
        hv = SVector{10,T}(ntuple(i -> u[i], 10))
        d, α, _, _ = state_validity(hv, SVector{10,T}(ntuple(i -> u[10 + i], 10)))
        (d > 0 && α > 0) || continue
        γ = SMatrix{3,3,T}(1 + hv[5], hv[6], hv[7], hv[6], 1 + hv[8], hv[9], hv[7],
                           hv[9], 1 + hv[10])
        λ, _ = sym_eigen3(γ)
        β = γ \ SVector{3,T}(hv[2], hv[3], hv[4])
        αlo, αhi = min(αlo, α), max(αhi, α)
        λlo, λhi = min(λlo, minimum(λ)), max(λhi, maximum(λ))
        βhi = max(βhi, sqrt(max(β' * γ * β, zero(T))))
        Khi = max(Khi, maximum(abs, u[11:20]) * α / sqrt(d))
    end
    return TreeGeneralizedHarmonic.StateBounds{T}(min(αlo / 4, 0.5), max(4αhi, 2.0),
                                                  min(λlo / 4, 0.5), max(4λhi, 2.0),
                                                  4βhi, 4Khi, gate)
end

"""
The case, the forest and the `evolve!` keywords of one row of the matrix.
"""
function gen_setup(sp)
    fs_kw(margin, lmax_shape, lmax_fit) =
        merge((variant=sp.variant, margin=margin, lmax_shape=lmax_shape,
               lmax_fit=lmax_fit), sp.spec)
    ch = sp.chart
    if ch === :ks0
        spec = FittedSpec(T; fs_kw(10, 4, 8)...)
        case = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(sp.chunk),
                                interior=spec,
                                horizon=Horizon(T; every=1, N=12, spin=true),
                                bounds=default_bounds(T; M=1, r_gate=T(9 // 10)))
        forest = hole_fixture_forest(T, case; N=8)
        if sp.kerr
            tgt = gen_kerr_target(case, forest)
            spec = FittedSpec(T; fs_kw(10, 4, 8)..., target=tgt)
            case = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(sp.chunk),
                                    interior=spec,
                                    horizon=Horizon(T; every=1, N=12, spin=true),
                                    bounds=default_bounds(T; M=1, r_gate=T(9 // 10)))
        end
    elseif ch in (:ks9, :h0, :h7, :h7c, :boost)
        bg, hw, radii, lsh, lfit, N_ah = ch === :ks9 ?
            (SM.KerrSchild(one(T), T(9 // 10)), T(5 // 2),
             (T(10), T(10), T(2), T(8 // 5)), 4, 12, 16) :
            ch === :h0 ? (SM.Harmonic(one(T), zero(T)), T(5 // 2),
                          (T(10), T(10), T(3 // 2), T(6 // 5)), 4, 8, 12) :
            ch === :h7 ? (SM.Harmonic(one(T), T(7 // 10)), T(5 // 4),
                          (T(10), T(8 // 5), T(13 // 10), one(T)), 12, 12, 16) :
            ch === :h7c ? (SM.Harmonic(one(T), T(7 // 10)), T(5 // 4),
                           (T(10), T(8 // 5), T(13 // 10)), 12, 12, 16) :
            (SM.translate(SM.boost(SM.Harmonic(one(T), zero(T)),
                                   SVector{3,T}(T(3 // 10), 0, 0)),
                          SVector{4,T}(0, T(3 // 4), 0, 0)), T(5 // 2),
             (T(10), T(10), T(3 // 2), T(9 // 10)), 4, 8, 12)
        margin = ch in (:h7, :h7c) ? 4 : 8
        spec = FittedSpec(T; fs_kw(margin, lsh, lfit)...)
        c0 = ch === :boost ? (T(3 // 4), zero(T), zero(T)) : (zero(T), zero(T), zero(T))
        case0 = hole_case(T, bg; halfwidth=hw, chunk=T(sp.chunk), center=c0,
                          interior=spec, horizon=Horizon(T; every=1, N=N_ah, spin=true))
        forest = ch === :boost ?
                 gen_capsule_forest(case0; N=8, radii=radii, t_end=sp.t_end) :
                 hole_forest(T, case0; N=8, roots=1, radii=radii)
        # The range projection's ranges: the fixture's for Kerr-Schild, the
        # target's own for the harmonic charts (whose data exceed them), at
        # the tracked geometry's gate.
        geom = gen_seed_geometry(case0, forest)
        gate = default_gate(geom, forest, GEN_Q)
        bd = ch === :ks9 ? default_bounds(T; M=1, r_gate=gate) :
             gen_layer_bounds(bg, geom, gate)
        case = with_bounds(case0, bd)
    elseif ch === :boost_sphere
        bg = SM.translate(SM.boost(SM.Harmonic(one(T), zero(T)),
                                   SVector{3,T}(T(3 // 10), 0, 0)),
                          SVector{4,T}(0, T(3 // 4), 0, 0))
        h = T(5 // 128)
        r_1 = T(horizon_min_radius(bg)) - 8h
        case0 = hole_case(T, bg; halfwidth=T(5 // 2), chunk=T(sp.chunk),
                          center=(T(3 // 4), zero(T), zero(T)), interior=:damped,
                          r_0=r_1 - 8h, r_1=r_1, ρ_ramp=one(T), margin=8,
                          horizon=Horizon(T; every=1, N=12, spin=true))
        forest = gen_capsule_forest(case0; N=8, radii=(T(10), T(10), T(3 // 2), T(9 // 10)),
                                    t_end=sp.t_end)
        bd = default_bounds(T; M=1, r_gate=default_gate(case0.interior, forest, GEN_Q))
        case = with_bounds(case0, bd)
    else
        error("no chart $ch")
    end
    return case, forest
end

gen_fmt(x) = x === nothing ? "     —   " :
             x isa Bool ? string(x) : Printf.format(Printf.Format("%9.3e"), x)

"""
One row of the matrix: `evolve!` with an observer that writes the shells and
prints a line per chunk, and a finder wrapper that keeps the horizon's
numbers — both into vectors that survive the run throwing — and, for the
coasting row, returns a failed find over the chunks it names.
"""
function gen_run(sp; t_end=nothing)
    case, forest = gen_setup(sp)
    q = GEN_Q
    tend = t_end === nothing ? T(sp.t_end) : min(T(t_end), T(sp.t_end))
    obs = NamedTuple[]
    hzs = Dict{Float64,Any}()
    calls = Ref(0)
    function finder(p, u, t; kw...)
        k = calls[]            # the chunk this find belongs to (0 at t = 0)
        if sp.coast !== nothing && sp.coast[1] ≤ k ≤ sp.coast[2]
            error("the finder is switched off for chunks $(sp.coast) (coasting row)")
        end
        o = find_gh_horizon(p, u, t; kw...)
        hzs[Float64(t)] = (area=o.area, M_irr=o.M_irr, J=o.J, M_ch=o.M_ch,
                           success=o.success)
        return o
    end
    function watch(p, t, u)
        calls[] += 1
        tt = T(t)
        int = p.interior
        gh_error!(p, u, tt; shell=horizon_shell(case, int))
        e = error_norms(p)
        gh_constraint!(p, u, tt)
        cn = constraint_norms(p)
        sh = gh_outside_shell_norms(p, u, tt)
        v = validity_rows(p, u, tt)
        hs = map(GEN_HSHELLS) do k
            if int isa FittedInterior
                c = center_at(int.center, tt)
                mask = TreeGeneralizedHarmonic.ShapeBand(c, int.shape, int.lmax,
                                                         int.r_in, int.r_out,
                                                         int.offset,
                                                         int.offset + k * int.h,
                                                         int.offset + (k + 1) * int.h)
            else
                hh, _ = layer_spacing(p.U.forest, int, tt)
                rh = T(horizon_min_radius(case.background))
                mask = ShellMask{T}(center_at(int.center, tt), rh + k * hh,
                                    rh + (k + 1) * hh)
            end
            gh_constraint!(p, u, tt; mask=mask)
            maximum(constraint_norms(p).gauge_l2)
        end
        acc = p.accounting
        hz = get(hzs, Float64(tt), nothing)
        row = (t=Float64(tt), err_l2=e.err_l2, err_linf=e.err_linf,
               gauge_l2=Float64(maximum(cn.gauge_l2)), residual=e.residual,
               drift=e.drift, sh_l2=sh.gauge_l2, sh_linf=sh.gauge_linf,
               hC=hs, hits=acc === nothing ? 0 : acc.hits,
               r_hit=acc === nothing ? -1.0 : Float64(acc.r_max),
               α_shell=v.min_α_shell, Π_shell=v.max_Π_shell,
               variant=interior_variant(int),
               area=hz === nothing ? nothing : hz.area,
               M_irr=hz === nothing ? nothing : hz.M_irr,
               J=hz === nothing ? nothing : hz.J,
               M_ch=hz === nothing ? nothing : hz.M_ch)
        push!(obs, row)
        say("   [%s] t=%6.3f err=%s/%s C=%s shC=%s hC0,2,4=%s,%s,%s res=%s " *
            "drift=%s hits=%d M_irr=%s J=%s M_ch=%s %s", sp.label, row.t,
            gen_fmt(row.err_l2), gen_fmt(row.err_linf), gen_fmt(row.gauge_l2),
            gen_fmt(row.sh_l2), gen_fmt(hs[1]), gen_fmt(hs[2]), gen_fmt(hs[3]),
            gen_fmt(row.residual), gen_fmt(row.drift), row.hits,
            gen_fmt(row.M_irr), gen_fmt(row.J), gen_fmt(row.M_ch),
            String(row.variant))
        flush(stdout)
        return nothing
    end
    t0 = time()
    failure = nothing
    out = try
        evolve!(T, case; forest=forest, q=q,
                ops=Operators(prolongation=q + 2, restriction=q + 2),
                t_end=tend, observer=watch, find=finder, sp.kw...)
    catch err
        err isa InterruptException && rethrow()
        failure = cal_root_cause(err)
        err isa TrackLostError && (failure = "track lost: " * failure)
        nothing
    end
    wall = time() - t0
    reached = isempty(obs) ? 0.0 : obs[end].t
    recs = out === nothing ? NamedTuple[] :
           [(t=r.t, dt=r.dt, fit_valid=r.fit_valid, fit_residual=r.fit_residual,
             fit_refills=r.fit_refills, track_offset=r.track_offset,
             track_prediction=r.track_prediction, track_source=r.track_source,
             track_center=r.track_center, track_velocity=r.track_velocity,
             margin_efolds=r.margin_efolds, bounds_r_max=r.bounds_r_max,
             center_offset=r.center_offset, horizon_success=r.horizon_success,
             nblocks=r.nblocks, h=r.h, variant=r.variant) for r in out.records]
    say("   done [%s] reached %.3f M of %.3f in %.0f s (%s steps, %d blocks)%s",
        sp.label, reached, tend, wall, out === nothing ? "?" : string(out.nsteps),
        nleaves(forest), failure === nothing ? "" : ", then threw: " * failure)
    out === nothing || say("   fits [%s]: %s", sp.label, string(out.fit_cost))
    flush(stdout)
    return (label=sp.label, spec=sp, reached=reached, t_end=Float64(tend),
            failure=failure, wall=wall,
            nsteps=out === nothing ? nothing : out.nsteps,
            nblocks=nleaves(forest), obs=obs, recs=recs)
end

function gen_fanout(batches; tag, t_end)
    base = isdir("out") ? joinpath("out", "generic") : mktempdir()
    dir = mkpath(joinpath(base, tag))
    println("   worker logs and results in ", abspath(dir))
    project = dirname(Base.active_project())
    procs = map(enumerate(batches)) do (n, (labels, nt))
        out = joinpath(dir, "worker-$n.jls")
        log = joinpath(dir, "worker-$n.log")
        te = t_end === nothing ? "" : "t_end=$(leak_spell(t_end))"
        cmd = `$(Base.julia_cmd()) --project=$project --threads=$nt
               $(abspath(@__FILE__)) generic worker=1
               runs=$(join(labels, ',')) out=$out $te`
        io = open(log, "w")
        (run(pipeline(cmd; stdout=io, stderr=io); wait=false), out, log, io)
    end
    return map(procs) do (proc, out, log, io)
        wait(proc)
        close(io)
        if !success(proc) || !isfile(out)
            println("   a worker failed; the end of its log ($log):")
            foreach(l -> println("     ", l), last(readlines(log), 30))
            return Any[]
        end
        foreach(l -> startswith(l, "   done") && println(l), readlines(log))
        return deserialize(out)
    end
end

function gen_report(results)
    println("\n-- the matrix: the last chunk of every row, and the extremes over the run --")
    println("| row | reached | end | err L2 | err L∞ | shell C_a L2 | C_a at r_h + 0, 2, 4 h " *
            "| residual | drift | hits (r_max) | fit valid (all) | fit residual (max) " *
            "| A | M_irr | J | M_ch | track offset (max, cells) | wall |")
    for r in sort(results; by=r -> r.label)
        isempty(r.obs) && (say("| %s | 0 | %s |", r.label,
                                r.failure === nothing ? "?" : r.failure); continue)
        l = r.obs[end]
        mi = [x.M_irr for x in r.obs if x.M_irr !== nothing]
        hz = findlast(x -> x.M_irr !== nothing, r.obs)
        fv = isempty(r.recs) ? nothing : all(x -> x.fit_valid !== false, r.recs)
        fr = isempty(r.recs) ? nothing :
             (v = [x.fit_residual for x in r.recs if x.fit_residual !== nothing];
              isempty(v) ? nothing : maximum(v))
        to = isempty(r.recs) ? nothing :
             (v = [x.track_offset for x in r.recs if x.track_offset !== nothing];
              isempty(v) ? nothing : maximum(v))
        rmax = isempty(r.recs) ? nothing :
               (v = [x.bounds_r_max for x in r.recs if x.bounds_r_max !== nothing];
                isempty(v) ? nothing : maximum(v))
        say("| %s | %.2f | %s | %s | %s | %s | %s, %s, %s | %s | %s | %d (%s) | %s | %s " *
            "| %s | %s | %s | %s | %s | %.0f s |", r.label, r.reached,
            r.failure === nothing ? "ok" : "threw", gen_fmt(l.err_l2),
            gen_fmt(l.err_linf), gen_fmt(l.sh_l2), gen_fmt(l.hC[1]), gen_fmt(l.hC[2]),
            gen_fmt(l.hC[3]), gen_fmt(l.residual), gen_fmt(l.drift), l.hits,
            gen_fmt(rmax), gen_fmt(fv), gen_fmt(fr),
            hz === nothing ? "—" : gen_fmt(r.obs[hz].area),
            hz === nothing ? "—" : gen_fmt(r.obs[hz].M_irr),
            hz === nothing ? "—" : gen_fmt(r.obs[hz].J),
            hz === nothing ? "—" : gen_fmt(r.obs[hz].M_ch), gen_fmt(to), r.wall)
    end
    for r in results
        r.failure === nothing || say("   %s threw at %.3f M: %s", r.label, r.reached,
                                     r.failure)
    end
    # The time series every 5 M (every M for runs shorter than 20 M).
    println("\n-- time series: masked err L2 / shell C_a L2 / M_irr --")
    for r in sort(results; by=r -> r.label)
        isempty(r.obs) && continue
        step = r.t_end ≥ 20 ? 5.0 : r.t_end ≥ 4 ? 1.0 : r.t_end / 5
        pts = [x for x in r.obs if x.t == 0 || abs(x.t / step - round(x.t / step)) < 1e-6 ||
               x === r.obs[end]]
        println("   ", r.label, ": ", join([Printf.format(Printf.Format("t=%.4g %.3e/%.3e/%s"),
                                                         x.t, x.err_l2, x.sh_l2,
                                                         x.M_irr === nothing ? "—" :
                                                         Printf.format(Printf.Format("%.6f"),
                                                                       x.M_irr))
                                              for x in pts], "  "))
    end
end

# --- the probe: harmonic Kerr at a = 9/10, host-side (decided 2026-09-23) ---
#
# `PLAN.md`'s step 8f, "Decided 2026-09-23": G5 runs at `a = 7/10` and no node
# is spent on `a = 9/10`; its row is this probe — the initial data and one
# right-hand side at `h = 5/256` as step 8e measured them, the fit's kink at
# three latitudes with whatever step 8f changed in the fit (`Π̃`, `L`), a run
# to `t_end_probe` (default `1/20 M`), and the price of the node run at
# `h = 5/1024` on the equator, written down.

# The composite data's second difference along the ray at the first evolved
# point `r_e = r_1(n̂) + h/2` — the analytic solution at `r_e` and `r_e + h`,
# the fit at `r_e − h` — against the analytic solution's own: `CODE.md`'s
# "kink" (step 8e's probe, redone in step 8f; the point is half a cell out of
# step 8e's, whose numbers it reproduces to a factor of two off the equator).
function gen_kink(bg, geom, fit, n)
    h = geom.h
    c = center_at(geom.center, zero(T))
    r1 = shape_radius(geom, SVector{3,T}(n)) - geom.offset
    re = r1 + h / 2
    an(r) = SVector{20}(state_tuple(bg, zero(T), Tuple(c + r * n)))
    ft(r) = vcat(fit_state(fit, c + r * n, zero(T))...)
    D2a = (an(re + h) - 2an(re) + an(re - h)) / h^2
    D2c = (an(re + h) - 2an(re) + ft(re - h)) / h^2
    return maximum(abs, D2c - D2a), maximum(abs, D2a)
end

const GEN_KINK_DIRS = (("axis", SVector{3,T}(0, 0, 1)),
                       ("45°", SVector{3,T}(1, 0, 1) / sqrt(T(2))),
                       ("equator", SVector{3,T}(1, 0, 0)))

# The kink table of one chart for a list of `(L, tilde)`.
function gen_kink_table(label, bg, geom, spec, Ls)
    bd = derive_target_bounds(T, bg, geom; t=0, L=8)
    for (L, tl) in Ls
        fit = build_fit(analytic_sampler(bg, 0.0; δ=geom.h / 8), geom, spec; cont=1,
                        bounds=bd, L=L, tilde=tl, check=false)
        ks = [gen_kink(bg, geom, fit, n) for (_, n) in GEN_KINK_DIRS]
        say("| %s | %d | %s | %s | %.2e | %.3g | %.3g (%.3g) | %.3g (%.3g) | %.3g (%.3g) |",
            label, L, tl ? "Π̃" : "Π", string(fit.valid), fit.residual.value,
            fit.sweep.min_λ, ks[1]..., ks[2]..., ks[3]...)
    end
end

# Leaves and points of a forest refined to `levels` levels wherever `pred(box)`.
function gen_count_forest(case, N, preds)
    forest = gh_forest(T, case; N=N, roots=1)
    for pred in preds
        ℓ = maxlevel(forest)
        targets = filter(k -> level(k) == ℓ && pred(block_extent(T, forest, k)),
                         forest.leaves)
        isempty(targets) && break
        refine!(forest, targets)
        balance!(forest)
    end
    return nleaves(forest), forest_levels(forest)
end

function gen_probe()
    println("\n=== (11p) harmonic Kerr a = 9/10: the host-side probe, m = 4, h = 5/256 ===")
    q, G = GEN_Q, GEN_G
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    bg = SM.Harmonic(one(T), T(9 // 10))
    radii = (T(10), T(8 // 5), T(13 // 10), one(T))
    mk(L, tl) = (spec = FittedSpec(T; variant=:fitted, margin=4, lmax_shape=12,
                                   lmax_fit=L, fit_tilde=tl);
                 case = hole_case(T, bg; halfwidth=T(5 // 4), chunk=T(1 // 400),
                                  interior=spec,
                                  horizon=Horizon(T; every=1, N=16, spin=false));
                 (spec, case))
    spec0, case0 = mk(8, false)
    forest = hole_forest(T, case0; N=8, roots=1, radii=radii)
    geom = with_ρ_max(fitted_interior(spec0, seed_track(case0, 0), forest, G; t=0,
                                      n_L=8), T(4))
    say("mesh: %d leaves, levels %s; h = %.5f, r_1 from %.4f (axis) to %.4f (equator), core from %.4f",
        nleaves(forest), string(forest_levels(forest)), geom.h,
        geom.r_in - geom.offset, geom.r_out - geom.offset,
        geom.r_in - geom.offset - geom.thickness)
    println("\n-- the kink at the first evolved point: |Δ²(composite) − Δ²(analytic)| (|Δ²(analytic)|), max over the packed components --")
    println("| chart | L | momentum | valid | value residual | sweep min λ(γ) | axis | 45° | equator |")
    gen_kink_table("harmonic a = 9/10", bg, geom, spec0,
                   ((8, false), (8, true), (12, false), (12, true), (16, true)))
    # The same for the charts the matrix runs, for comparison.
    for (lab, b2, hw, rad, m, lsh) in
        (("harmonic a = 7/10, 5/256", SM.Harmonic(one(T), T(7 // 10)), T(5 // 4), radii, 4, 12),
         ("harmonic a = 7/10, 5/128", SM.Harmonic(one(T), T(7 // 10)), T(5 // 4),
          (T(10), T(8 // 5), T(13 // 10)), 4, 12),
         ("Kerr-Schild a = 9/10, 5/128", SM.KerrSchild(one(T), T(9 // 10)), T(5 // 2),
          (T(10), T(10), T(8 // 5), T(5 // 4)), 5, 4),
         ("harmonic a = 0, 5/128", SM.Harmonic(one(T), zero(T)), T(5 // 2),
          (T(10), T(10), T(3 // 2), T(6 // 5)), 8, 4),
         ("Kerr-Schild a = 0, fixture", SM.KerrSchild(one(T), zero(T)), T(5 // 2),
          (T(3), T(3), one(T)), 10, 4))
        s2 = FittedSpec(T; variant=:fitted, margin=m, lmax_shape=lsh)
        c2 = hole_case(T, b2; halfwidth=hw, chunk=T(1 // 10), interior=s2,
                       horizon=Horizon(T; every=1, N=16, spin=false))
        f2 = hole_forest(T, c2; N=8, roots=1, radii=rad)
        g2 = fitted_interior(s2, seed_track(c2, 0), f2, G; t=0, n_L=8)
        gen_kink_table(lab, b2, g2, s2, ((8, false), (8, true), (12, false), (12, true)))
    end

    # The initial data and one right-hand side, at step 8e's fit and at 8f's.
    Lbest = parse(Int, get(OPTIONS, "lmax_fit", "12"))
    local p_best, u_best, fs_best, trhs_best
    println()
    for (L, tl) in ((8, false), (Lbest, true))
        spec, case = mk(L, tl)
        bd = derive_target_bounds(T, bg, geom; t=0, L=8)
        t0 = time()
        fit = build_fit(analytic_sampler(bg, 0.0; δ=geom.h / 8), geom, spec; cont=1,
                        bounds=bd, check=false)
        tfit = time() - t0
        fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3), backend=CPU())
        p = refill_target(GHProblem(fs, GhostSchedule(fs, ops), case; q=q,
                                    interior=geom, target=target_cache(fs),
                                    fits=(fit, nothing)), zero(T))
        u = statevector(fs)
        map_blocks!(TreeGeneralizedHarmonic.fitted_state_kernel!, fs, statearray(u, fs),
                    p.target.work, p.origins, p.spacings, bg, p.interior, zero(T),
                    zero(T), zero(T))
        A = statearray(u, fs)
        worst = (detγ=Inf, α=Inf, λ=Inf)
        nbad = 0
        for b in 1:nblocks(fs), k in 1:8, j in 1:8, i in 1:8
            hv = SVector{10}(ntuple(v -> A[i, j, k, v, b], 10))
            Πv = SVector{10}(ntuple(v -> A[i, j, k, 10 + v, b], 10))
            d, α, _, _ = state_validity(hv, Πv)
            λ, _ = sym_eigen3(SMatrix{3,3}(1 + hv[5], hv[6], hv[7], hv[6], 1 + hv[8],
                                            hv[9], hv[7], hv[9], 1 + hv[10]))
            worst = (detγ=min(worst.detγ, d), α=min(worst.α, α),
                     λ=min(worst.λ, minimum(λ)))
            nbad += !(d > 0 && α > 0 && minimum(λ) > 0 && all(isfinite, hv) &&
                      all(isfinite, Πv))
        end
        du = similar(u)
        gh_rhs!(du, u, p, zero(T))
        t1 = time()
        gh_rhs!(du, u, p, zero(T))
        trhs = time() - t1
        D = statearray(du, fs)
        imax = argmax(abs.(D))
        xmax = coordinates(fs, imax[5], (imax[1] + G, imax[2] + G, imax[3] + G))
        say("L = %d, %s: fit valid %s (value residual %.2e, sweep min λ %.3g, %.2f s); " *
            "initial data %d points, %d non-finite, %d not a metric, min det γ %.3g, " *
            "min α %.3g, min λ(γ) %.3g; one RHS %.2f s at %d threads, max |du| %.3g at " *
            "|x| = %.4f, z = %.4f",
            L, tl ? "Π̃" : "Π", string(fit.valid), fit.residual.value, fit.sweep.min_λ,
            tfit, length(u) ÷ 20, count(!isfinite, u), nbad, worst.detγ, worst.α,
            worst.λ, trhs, Threads.nthreads(), maximum(abs, du),
            sqrt(sum(abs2, xmax)), xmax[3])
        if L == Lbest && tl
            p_best, u_best, fs_best, trhs_best = p, u, fs, trhs
        end
    end

    # The fastest speed and the step at 5/256, for the price below.
    scatter!(fs_best, u_best)
    λ = max_speed(p_best; t=zero(T))
    dt = T(1 // 4) * geom.h / λ
    npts = length(u_best) ÷ 20
    say("λ_max = %.4g, dt = %.4g M at cfl = 1/4 (%.0f steps per M)", λ, dt, 1 / dt)

    # The run, with 8f's fit: does it survive its first chunk?
    tp = T(only(leak_option("t_end_probe", [1 // 20])))
    specb, caseb = mk(Lbest, true)
    reached = Ref(zero(T))
    t0 = time()
    rows = String[]
    watch(p, t, u) = (reached[] = T(t);
                      v = validity_rows(p, u, T(t));
                      push!(rows, Printf.format(Printf.Format(
                          "   t = %.4f: min α shell %.3g, min det γ shell %.3g, max |Π| shell %.3g, min α layer %.3g"),
                          T(t), v.min_α_shell, v.min_detγ_shell, v.max_Π_shell, v.min_α_layer));
                      println(rows[end]); flush(stdout))
    fail = try
        evolve!(T, caseb; forest=forest, q=q, ops=ops, t_end=tp, observer=watch)
        nothing
    catch e
        e isa InterruptException && rethrow()
        cal_root_cause(e)
    end
    say("the run with L = %d, Π̃: reached %.4f M of %.4f in %.0f s%s", Lbest, reached[],
        tp, time() - t0, fail === nothing ? "" : "; it ended: " * fail)

    # The price of the node run at h = 5/1024 on the equator.
    println("\n-- the node run at h = 5/1024, priced --")
    ring(box, lo, hi, zmax) = begin
        # the box meets the solid torus lo ≤ ρ ≤ hi, |z| ≤ zmax
        rmin = sqrt(sum(abs2, (max(box[1][d], min(0.0, box[2][d])) for d in 1:2)))
        rmax = maximum(sqrt(sum(abs2, (c[d] for d in 1:2)))
                       for c in Iterators.product((box[1][1], box[2][1]),
                                                  (box[1][2], box[2][2])))
        zlo = max(box[1][3], min(0.0, box[2][3]))
        rmax ≥ lo && rmin ≤ hi && abs(zlo) ≤ zmax
    end
    ball(R) = box -> TreeGeneralizedHarmonic._box_meets_ball(box, (0.0, 0.0, 0.0), R)
    base = [ball(10.0), ball(1.6), ball(1.3), ball(1.0)]
    n256, lv256 = gen_count_forest(case0, 8, base)
    nball, lvball = gen_count_forest(case0, 8, [base..., ball(1.0), ball(1.0)])
    nring, lvring = gen_count_forest(case0, 8, [base..., b -> ring(b, 0.7, 1.1, 0.25),
                                                b -> ring(b, 0.75, 1.05, 0.2)])
    # The measured cost per point and thread of one right-hand side here; a
    # Symmetry core is about twice as slow as this machine's (CLAUDE.md).
    c_pt = trhs_best * Threads.nthreads() / npts
    dt4 = dt / 4
    println("| mesh | leaves | levels | points | memory (2.4 kB/pt, 8e) | RHS on 64 node threads " *
            "| dt | wall per M | 10 M | 50 M |")
    for (lab, n, lv, d) in (("5/256 (measured)", n256, lv256, dt),
                            ("5/1024 inside |x| ≤ 1", nball, lvball, dt4),
                            ("5/1024 on the equatorial band", nring, lvring, dt4))
        pts = n * 8^3
        trhs_node = pts * c_pt * 2 / 64
        perM = trhs_node * 4 / d
        say("| %s | %d | %s | %.3g | %.3g GB | %.3g s | %.3g | %.3g h | %.3g h | %.3g h |",
            lab, n, string(lv), pts, pts * 2.4e3 / 1e9, trhs_node, d, perM / 3600,
            10perM / 3600, 50perM / 3600)
    end
    say("(cost per point and thread measured here: %.3g µs; one RHS of the 5/256 mesh " *
        "%.2f s at %d threads)", c_pt * 1e6, trhs_best, Threads.nthreads())
end

const GEN_DEFAULT = ["ks0", "ks9", "harm", "boost", "probe"]

if "generic" in SECTIONS || haskey(OPTIONS, "generic")
    t_end_opt = haskey(OPTIONS, "t_end") ? only(leak_option("t_end", [1 // 1])) : nothing
    gen_specs = gen_all_specs()
    if haskey(OPTIONS, "worker")
        labels = String.(split(OPTIONS["runs"], ','))
        res = [gen_run(gen_specs[l]; t_end=t_end_opt) for l in labels]
        serialize(OPTIONS["out"], res)
    else
        groups = gen_groups()
        names = haskey(OPTIONS, "generic") ?
                String.(split(OPTIONS["generic"], r"[,+]")) : GEN_DEFAULT
        labels = String[]
        probe = "probe" in names
        for n in names
            if n == "probe"
                continue
            elseif haskey(groups, n)
                append!(labels, [sp.label for sp in groups[n]])
            elseif haskey(gen_specs, n)
                push!(labels, n)
            else
                error("no generic group or row $n")
            end
        end
        if !isempty(labels)
            println("
=== (11) the measurement matrix: the generic interior ===")
            nt = Threads.nthreads()
            say("%d rows (%s) at %d threads%s", length(labels), join(names, ", "), nt,
                t_end_opt === nothing ? "" : ", t_end ≤ $(Float64(t_end_opt)) M")
            t0 = time()
            results = if nt ≥ 16 && length(labels) > 1
                wt = haskey(OPTIONS, "threads") ? parse(Int, OPTIONS["threads"]) :
                     max(1, nt ÷ length(labels))
                say("   %d workers of %d threads", length(labels), wt)
                reduce(vcat, gen_fanout([([l], wt) for l in labels];
                                        tag=join(names, "+"), t_end=t_end_opt);
                       init=Any[])
            else
                [gen_run(gen_specs[l]; t_end=t_end_opt) for l in labels]
            end
            say("%d rows in %.0f s", length(results), time() - t0)
            gen_report(results)
        end
        probe && gen_probe()
    end
end

println("\ndone")
