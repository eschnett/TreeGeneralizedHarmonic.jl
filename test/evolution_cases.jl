# The runs the evolution tests measure: a convergence study, a
# robust-stability study, and the fixtures both are built from.
#
# This file is a *helper*, not a test file — `runtests.jl` includes it
# before the test files that use it, as TreeAMR includes `test/wave.jl`.
# It lives here rather than in `src/` because what it wraps is the
# integrator loop, and `CODE.md`'s driver (`evolve!`, the chunked loop
# with regridding and the per-chunk analysis record) is step 5's. Writing
# half a driver now and replacing it then would leave two of them; what
# the step-3 tests need is the *shortest* thing that turns a `GHProblem`
# into an error norm.
#
# Everything here is generic in the element type and in the backend, for
# the reason `PLAN.md` gives under "Ground rules": retrofitting those is a
# rewrite, and steps 4 and 9 add the tests of the property, not the
# property.

using KernelAbstractions: CPU
using OrdinaryDiffEqLowOrderRK: RK4
using Random: MersenneTwister
using SciMLBase: ODEProblem, solve
using TreeGeneralizedHarmonic: ceilint

"""
The state field set and the problem for a case on a uniform mesh, built
the way every run in this package builds them: 20 variables, `G = q/2 + 1`,
vertex-centered, prolongation `p = q + 2`.

The operator order is spelled out at the call site and never defaulted
(`CLAUDE.md`, "The interface-order rule"): this system takes second
derivatives, so a ghost filled at order `q + 1` or below costs the scheme
an order, and a default would hide exactly that.
"""
function gh_setup(::Type{T}, case::GHCase{T}; N, roots, q,
                  ops=Operators(prolongation=q + 2, restriction=q + 2),
                  refined=false, backend=CPU()) where {T}
    forest = gh_forest(T, case; N=N, roots=roots, refined=refined)
    fs = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                     backend=backend)
    problem = GHProblem(fs, GhostSchedule(fs, ops), case; q=q)
    return forest, fs, problem
end

"""
Evolve `case` to `t_end` with fixed-step RK4 and return the
volume-weighted L2 and L∞ errors against the analytic solution, with the
spacing and the step count that produced them.

The pattern is TreeWave's `wave_errors`: sample the exact solution at
`t = 0`, integrate, sample it again at `t_end` into a field set of the
*same layout* (a different ghost width or centering would sample it at
different points), and subtract state vectors. The time step comes from
[`gh_dt`](@ref) — the CFL bound of the initial state — and is then
trimmed so that the run lands exactly on `t_end`, which is what makes a
convergence study compare solutions at one time.

`save_everystep = false`: nothing between the ends is wanted, and at 20
variables on a 3D mesh the intermediate states are the memory.
"""
function gh_errors(::Type{T}, case::GHCase{T}; N, roots, q, t_end,
                   cfl=T(1 // 4), ops=Operators(prolongation=q + 2,
                                                restriction=q + 2),
                   refined=false, backend=CPU()) where {T}
    forest, fs, problem = gh_setup(T, case; N=N, roots=roots, q=q, ops=ops,
                                   refined=refined, backend=backend)
    fill_exact!(fs, case, zero(T))
    u0 = statevector(fs)
    gather!(u0, fs)

    t_end = T(t_end)
    dt = gh_dt(problem, u0; cfl=cfl)
    nsteps = max(1, ceilint(t_end / dt))
    dt = t_end / nsteps

    sol = solve(ODEProblem(gh_rhs!, u0, (zero(T), t_end), problem), RK4();
                dt=dt, adaptive=false, save_everystep=false)

    exact = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                        backend=backend)
    fill_exact!(exact, case, t_end)
    uexact = statevector(exact)
    gather!(uexact, exact)

    err = sol.u[end] .- uexact
    return (l2=volume_weighted_norm(fs, err),
            linf=volume_weighted_norm(fs, err; p=Inf),
            h=minimum_spacing(T, forest), nsteps=nsteps,
            nblocks=nleaves(forest))
end

"""
The two constraint monitors of a case's **exact** state at time `t`, on
the mesh `N`, `roots`, `refined` describes, as the norms the analysis
record holds.

Exact data is the sharp setting for a monitor: the continuum constraints
vanish on it identically, so whatever the monitor reports is the
*discretization's* violation and nothing else — roundoff where the
stencils are exact (flat space), the interface error on a two-level mesh,
and the bulk truncation error where the background is genuinely curved.
There is no evolution here for the same reason; `gh_errors` is where a
run's accumulated error is measured.

Both kernels fill their ghosts with this `t`'s hook before they read a
stencil, so the numbers are the mesh's and not the boundary's.
"""
function gh_constraint_run(::Type{T}, case::GHCase{T}; N, roots, q, t=zero(T),
                           ops=Operators(prolongation=q + 2,
                                         restriction=q + 2),
                           refined=false, mask=AllPoints(),
                           backend=CPU()) where {T}
    forest, fs, problem = gh_setup(T, case; N=N, roots=roots, q=q, ops=ops,
                                   refined=refined, backend=backend)
    fill_exact!(fs, case, T(t))
    u = statevector(fs)
    gather!(u, fs)
    gh_constraint!(problem, u, T(t); mask=mask)
    gauge = constraint_norms(problem)
    adm_constraint!(problem, u, T(t); mask=mask)
    adm = constraint_norms(problem)
    return (gauge_l2=maximum(gauge.gauge_l2), gauge_linf=maximum(gauge.gauge_linf),
            ham_l2=adm.ham_l2, ham_linf=adm.ham_linf,
            mom_l2=maximum(adm.mom_l2), mom_linf=maximum(adm.mom_linf),
            h=minimum_spacing(T, forest), nblocks=nleaves(forest),
            problem=problem, fs=fs, u=u)
end

"""
White noise of amplitude `amplitude` on top of the case's exact state, as
a state vector.

Drawn from a seeded generator on the host, in the state vector's own
order, so the perturbation is the same at every thread count and can be
replayed from the seed — which is the only way a stability claim about
"noise" is a claim at all. It perturbs all 20 variables, `Π` as much as
`h`: a perturbation of `h` alone is a constrained initial datum in one
sense and this test is about the unconstrained ones.
"""
function gh_noisy_state(::Type{T}, fs, case::GHCase{T}; amplitude,
                        seed=20260917) where {T}
    fill_exact!(fs, case, zero(T))
    u = statevector(fs)
    gather!(u, fs)
    rng = MersenneTwister(seed)
    a = T(amplitude)
    for i in eachindex(u)
        u[i] += a * (2 * rand(rng, T) - 1)
    end
    return u
end

"""
Run `nsteps` fixed steps of RK4 on noise-perturbed data and report how the
perturbation grew: its L2 and L∞ norms at the start and at the end, and
the ratios.

`CODE.md`'s robust-stability case (`PLAN.md` step 3): white noise of
amplitude `1e−8` on flat space must stay bounded over a thousand steps
with `ε_KO = 0.5`, and what it does without dissipation is recorded rather
than asserted. The norms are of `u` itself, because the exact solution of
this case is `h = Π = 0` and the state *is* the perturbation.
"""
function gh_noise_growth(::Type{T}, case::GHCase{T}; N, roots, q, nsteps,
                         amplitude=T(1 // 10)^8, cfl=T(1 // 4),
                         ops=Operators(prolongation=q + 2, restriction=q + 2),
                         seed=20260917, backend=CPU()) where {T}
    forest, fs, problem = gh_setup(T, case; N=N, roots=roots, q=q, ops=ops,
                                   backend=backend)
    u0 = gh_noisy_state(T, fs, case; amplitude=amplitude, seed=seed)
    dt = gh_dt(problem, u0; cfl=cfl)
    t_end = nsteps * dt
    sol = solve(ODEProblem(gh_rhs!, u0, (zero(T), t_end), problem), RK4();
                dt=dt, adaptive=false, save_everystep=false)
    u1 = sol.u[end]
    l2_0 = volume_weighted_norm(fs, u0)
    l2_1 = volume_weighted_norm(fs, u1)
    linf_0 = volume_weighted_norm(fs, u0; p=Inf)
    linf_1 = volume_weighted_norm(fs, u1; p=Inf)
    return (l2_0=l2_0, l2_1=l2_1, linf_0=linf_0, linf_1=linf_1,
            l2_ratio=l2_1 / l2_0, linf_ratio=linf_1 / linf_0,
            dt=dt, t_end=t_end, nsteps=nsteps, finite=all(isfinite, u1))
end

# --- the black hole (added in step 5) ---------------------------------------
#
# From here on the runs go through `evolve!` — `driver.jl`'s one chunked
# loop — rather than through a bare `solve`, because that is what step 5
# built and what `CODE.md` judges a run by. What is written here is the
# *fixture*: the mesh and the parameters the suite's hole runs share,
# picked so that `check_interior_radii` passes at the coarsest resolution
# of the sweep and the whole sweep fits in a test file.

"""
The suite's static-hole fixture: Kerr-Schild (`a = 0`, sampled `H`) on a
box of half-width `5/2 M`, with the frozen hierarchy
[`hole_forest`](@ref) builds around the origin.

**Why these numbers.** `CODE.md`'s two radius requirements are
`r_1 ≤ r_h,min − m·h` and `r_1 − r_0 ≥ 2(G+1)·h`, so together they need
`r_h,min ≥ (m + 2G + 2)·h + r_0` — the resolution is set by the horizon's
*smallest coordinate radius*, and Kerr-Schild's `r₊ = 2 M` is the largest
of the three holes in the table (harmonic Kerr's is `M` at `a = 0` and
`0.44 M` at `a = 9/10`). That is why the cheap case is this one. At
`q = 2`, `G = 2`, the default margin `m = 8`, `r_0 = 2/5` (where
`|h| ≈ 5`, `CODE.md`'s "still moderate") and `h = 5/48` at the coarsest
`N` of the sweep (`N = 6`), the bound is `r_1 ≤ 2 − 8h = 1.167` and the
thickness `r_1 − r_0 ≥ 6h = 0.625`, both of which `r_1 = 23/20` clears —
with the default margin `m = 8` and not the floor.

The hierarchy is **three shells and genuinely nested**: the third catches
only the level-2 blocks that reach within `M` of the center, so the
sphere `r_1` lies wholly inside the finest level while the outer half of
the box stays a level coarser — 120 leaves with a coarse-fine face
between them, at every `N`. Doubling `N` halves every spacing and changes
nothing else, which is `CODE.md`'s frozen-hierarchy protocol; a fixture
whose shells caught every block would be a uniform mesh with no interface
in it, and the interface is where a hole on a refined mesh is different
from a hole on a uniform one.
"""
function hole_fixture(::Type{T}=Float64; q=2, M=one(T), a=zero(T),
                      variant=:damped, margin=8, r_0=T(2 // 5),
                      r_1=T(23 // 20), halfwidth=T(5 // 2), chunk=T(1 // 10),
                      kwargs...) where {T}
    return kerr_schild_case(T; M=M, a=a, halfwidth=halfwidth, r_0=r_0, r_1=r_1,
                            chunk=chunk, margin=margin, interior=variant,
                            kwargs...)
end

"""
The forest the fixture runs on: [`hole_forest`](@ref) with three shells.
The first two catch every block (a root block and its eight children all
touch the center), the third catches only the level-2 blocks within `M`
of it — so the mesh is `56` blocks at level 2 around `64` at level 3, and
the layout does not depend on `N`.
"""
hole_fixture_forest(::Type{T}, case::GHCase{T}; N, roots=1,
                    radii=(T(3), T(3), one(T))) where {T} =
    hole_forest(T, case; N=N, roots=roots, radii=radii)

"""
One static-hole run through [`evolve!`](@ref), returning the driver's
record together with the spacing and the block count that produced it —
the shape a convergence table is built from.
"""
function gh_hole_run(::Type{T}, case::GHCase{T}; N, q, t_end, roots=1,
                     radii=(T(3), T(3), one(T)),
                     ops=Operators(prolongation=q + 2, restriction=q + 2),
                     backend=CPU(), kwargs...) where {T}
    forest = hole_fixture_forest(T, case; N=N, roots=roots, radii=radii)
    out = evolve!(T, case; forest=forest, q=q, ops=ops, t_end=T(t_end),
                  backend=backend, kwargs...)
    return out
end

# --- the indicator's hole (added in step 6) ---------------------------------
#
# A *second* hole fixture, because the refinement needs room the step-5 one
# does not have: `CODE.md`'s level ceiling holds the blocks within a few
# coarse cells of the outer boundary at the coarsest level, and its level
# floor refines the shell from `r_1` out past the horizon — so a box whose
# half-width is only `5/4` of the horizon's coordinate radius has the two
# regions touching, and `block_level_bounds` throws saying exactly that.
# This fixture is the reference configuration of `CODE.md`'s calibration:
# the same Kerr-Schild hole in a box of half-width `5 M` on a `4³` root
# brick, at the margin `m = 4` rather than the default `8` so that the
# level the interior needs is one rather than two — which is what keeps the
# mesh at the *same* 120 blocks and 61 440 points as the step-5 fixture
# while the indicator, not a hand-written shell list, chooses them.

"""
The suite's **adaptive**-hole fixture: Kerr-Schild (`a = 0`) in a box of
half-width `5 M` with `CODE.md`'s refinement parameters attached — the
thresholds calibrated in step 6 (`refine_tol = 2/5`, `coarsen_tol = 1/10`,
mid-plateau of the table under "Measured results"), the level floor derived
from the interior's own radii, and the ceiling one coarse cell deep.

`maxlevel_cap = 1` is the suite's, not the calibration's: at the calibrated
thresholds the indicator asks for level 2 around this hole and the mesh is
848 blocks, which is `test/hole_runs.jl`'s business and not a test file's.
Capping at one level keeps the fixture at the step-5 fixture's size and has
the side effect of exercising the `(Keep, box)` mark, which is the mark a
block *at the cap* reports and which nothing else in the suite reaches.

`margin = 4` rather than `CODE.md`'s default `8`: the floor level is
`min((r_h,min − r_1)/m, (r_1 − r_0)/(2(G+1)))` in spacings, so the margin is
what decides whether this hole needs one refinement level or two. `m = 4` is
still above the floor `G + 1 = 3` the interior insists on.
"""
function adaptive_hole_fixture(::Type{T}=Float64; q=2, M=one(T), a=zero(T),
                               refine_tol=T(2 // 5), coarsen_tol=T(1 // 10),
                               maxlevel_cap=1, floor_margin=zero(T),
                               ceiling_cells=1, margin=4, r_0=T(3 // 10),
                               r_1=T(5 // 4), halfwidth=T(5),
                               chunk=T(1 // 20), kwargs...) where {T}
    ref = Refinement(T; refine_tol=refine_tol, coarsen_tol=coarsen_tol,
                     maxlevel_cap=maxlevel_cap, floor_margin=floor_margin,
                     ceiling_cells=ceiling_cells)
    return kerr_schild_case(T; M=M, a=a, halfwidth=halfwidth, r_0=r_0, r_1=r_1,
                            chunk=chunk, margin=margin, refinement=ref,
                            kwargs...)
end

"""
Fill a uniform mesh with a case's exact data at `t`, fill its ghosts with
that `t`'s hook, and evaluate the indicator into a scratch `diag` — one
pass over the criterion and no time stepping at all, which is what every
claim about `τ` itself is made on.

The ghost fill is not optional and is the reason this helper exists:
Löhner's stencil reaches one point past the block face, and `regrid!` fills
ghosts only *after* the flags are computed, for its own prolongation.

It stops short of the marks, because `τ` is defined on any mesh while the
marks are not: the level bounds refuse a `maxlevel_cap` below the level the
interior's radii need, and the calibration table sweeps `h` over meshes far
coarser than that.
"""
function gh_tau_pass(::Type{T}, case::GHCase{T}; N, roots, q=2, t=zero(T),
                     forest=nothing,
                     ops=Operators(prolongation=q + 2, restriction=q + 2),
                     backend=CPU()) where {T}
    f = forest === nothing ? gh_forest(T, case; N=N, roots=roots) : forest
    U = FieldSet{T}(f, 20; G=q ÷ 2 + 1, centering=vertexcentered(3),
                    backend=backend)
    sched = GhostSchedule(U, ops)
    fill_exact!(U, case, T(t))
    boundary = dirichlet(case, T(t))
    boundary === nothing ? fill_ghosts!(U, sched) :
    fill_ghosts!(U, sched; boundary=boundary)
    origins = TreeGeneralizedHarmonic.to_backend(backend, block_origins(f, T))
    spacings = TreeGeneralizedHarmonic.to_backend(backend, block_spacings(f, T))
    mask = interior_mask(case.interior, T(t))
    scales = field_scales(U, mask, origins, spacings)
    τfs = FieldSet{T}(f, TreeGeneralizedHarmonic.NDIAG; G=0,
                      centering=U.centering, backend=backend)
    gh_tau!(τfs, U, origins, spacings, mask; scale=maximum(scales),
            ε=case.refinement === nothing ? T(1 // 100) : case.refinement.ε)
    return (forest=f, U=U, schedule=sched, τfs=τfs, scales=scales,
            scale=maximum(scales), τ_max=tau_max(τfs),
            origins=origins, spacings=spacings, mask=mask)
end

"""
[`gh_tau_pass`](@ref) followed by the marks: the indicator's whole verdict
on one filled mesh, as `regrid!` would take it.
"""
function gh_flagging_pass(::Type{T}, case::GHCase{T}; N, roots, q=2,
                          t=zero(T), buffer=0, forest=nothing,
                          ops=Operators(prolongation=q + 2, restriction=q + 2),
                          backend=CPU()) where {T}
    pass = gh_tau_pass(T, case; N=N, roots=roots, q=q, t=t, forest=forest,
                       ops=ops, backend=backend)
    out = indicator_flags(pass.U, case, T(t); G=q ÷ 2 + 1, buffer=buffer)
    return (forest=pass.forest, U=pass.U, schedule=pass.schedule,
            τfs=pass.τfs, scales=pass.scales, out...)
end

"""
`CODE.md`'s initial-data cycle on a case: fill, flag with the indicator,
regrid without transferring, re-evaluate — until the hierarchy stops
changing. Returns the mesh it converged to together with the verdict on it.

This is [`evolve!`](@ref)`(; adapt = true)` without the evolution, which is
what a test of the *mesh* wants: no right-hand side is evaluated, so the
whole cycle costs a handful of initial-data fills.
"""
function gh_adapt_cycle(::Type{T}, case::GHCase{T}; N, roots, q=2, buffer=1,
                        maxpasses=8,
                        ops=Operators(prolongation=q + 2, restriction=q + 2),
                        backend=CPU()) where {T}
    G = q ÷ 2 + 1
    forest = gh_forest(T, case; N=N, roots=roots)
    U = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3),
                    backend=backend)
    criterion(fs) = indicator_flags(fs, case, zero(T); G=G, buffer=buffer).flags
    sched, passes, converged = adapt_to_initial_data!(
        U, ops; initial=state_callback(case, zero(T)), flags=criterion,
        buffer=0, maxpasses=maxpasses, boundary=dirichlet(case, zero(T)))
    boundary = dirichlet(case, zero(T))
    boundary === nothing ? fill_ghosts!(U, sched) :
    fill_ghosts!(U, sched; boundary=boundary)
    out = indicator_flags(U, case, zero(T); G=G, buffer=buffer)
    return (forest=forest, U=U, schedule=sched, passes=passes,
            converged=converged, out...)
end

"""
The constraint norms of a run's final state over the shell of `width`
spacings just **outside** `r_1` — `CODE.md`'s "the `G` points outside
`r_1`", the only points at which the three interior variants can differ
before the difference has had time to propagate.

The shell is a [`ShellMask`](@ref), so it is the same masked-norm
machinery every other row of the record uses.
"""
function gh_outside_shell_norms(out; width=nothing)
    p = out.problem
    T = eltype(p.U.work)
    int = p.interior
    G = first(p.U.G)
    w = width === nothing ? G : width
    h = T(out.h)
    c = center_at(int.center, T(out.records[end].t))
    mask = ShellMask{T}(c, int.r_1, int.r_1 + w * h)
    gh_constraint!(p, out.u, T(out.records[end].t); mask=mask)
    c1 = constraint_norms(p)
    gh_error!(p, out.u, T(out.records[end].t); mask=mask)
    e = error_norms(p)
    return (gauge_l2=maximum(c1.gauge_l2), gauge_linf=maximum(c1.gauge_linf),
            err_l2=e.err_l2, err_linf=e.err_linf,
            npoints=sum(masked_counts(p)))
end
