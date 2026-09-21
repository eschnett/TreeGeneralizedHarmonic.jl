# Working notes for Claude in TreeGeneralizedHarmonic.jl

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. This file is only about mechanics.

## What this package is

The third downstream application of
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl), and the first step
toward a production code: the vacuum Einstein equations in the
generalized harmonic formulation, second order in space and first order
in time, on TreeAMR's octree of uniform blocks. What `CODE.md`
specifies is a **proof of concept** — a single boosted, spinning black
hole with its interior driven to the analytic solution — and its
milestones stop there. TreeAMR is the mesh and contains no physics;
[TreeWave](https://github.com/eschnett/TreeWave.jl) is the scalar wave
equation and [TreeHydro](../TreeHydro) is Newtonian hydrodynamics; this
package is general relativity. The formulation and the pointwise
algebra come from `GeneralizedHarmonicSecondOrder2` ("GHSO2"), where
they were validated on SBP-SAT spectral elements. That repository is
unpublished, so the documents this package rests on are copied verbatim
into `notes/` — read `notes/methods-ghso2.md` for the physics and the
numbers this package compares itself against, and cite `notes/`, not
the sibling checkout.

Three rules follow from `CODE.md` and govern every change here:

- **No mesh machinery.** If a change is about trees, ghost cells,
  interpolation or reductions, it belongs upstream in TreeAMR. The one
  stopgap this package carries — point interpolation for the horizon
  finder — is marked as such in `CODE.md` and goes upstream when
  TreeAMR grows it.
- **No singularity handling, and the interior is pointwise.** There is
  no excision. Inside the horizon the right-hand side is modified by two
  smooth profiles of the distance to the hole's analytic center: a
  relaxation toward the analytic solution in a layer, and a switch-off
  around the singularity. Those profiles depend on position and time
  and on nothing about blocks, levels or ghost widths. Do not add an
  excision mask or an extrapolation into the hole; `CODE.md` lists
  excision under extensions with the design that was set aside.
- **Inherit GHSO2's algebra, do not re-derive it.** `pointwise.jl` is a
  port of `notes/pointwise-ghso2.jl`; a change to the equations there is
  a change to a validated result and needs the corresponding test
  against `SpacetimeMetrics` automatic differentiation to say so.

## Current state

**G0–G4 are done — there is a black hole on a mesh the code chose, and the
code knows where its horizon is: the interior damping layer, the driver
with its per-chunk analysis record and its regrid branch, the masked
Löhner indicator with its interior mask, its derived level floor and its
boundary ceiling, the masked error converging at order `q` on a frozen
hierarchy, and the apparent horizon with its area, `M_irr`, Korzyński `J`
and `M_ch` at Kerr's values in both charts and at `a = 9/10`. The moving
hole (G5) is next.**
`CODE.md` is complete and reviewed three times (2026-09-16): the expanded
form of the momentum equation, three dimensions only, a pointwise damping
layer instead of excision, a single boosted spinning black hole as the
proof-of-concept target, RK4 from OrdinaryDiffEq, the analysis quantities
as part of the deliverable, an error indicator for refinement, `Float64`
on Symmetry's H200 as the device requirement, no checkpointing, GPU
kernel efficiency deferred to a research project. `PLAN.md` breaks the
milestones G0–G6 into steps 0–10, each a brief for one agent with a fresh
context (see its "Running a step as an agent"); step 8 (a hole that
moves) is next. `notes/` holds the inherited documents.

What exists in `src/` is the module shell, `precision.jl` (the `Base`
bridges for software floating-point types), `device.jl` (`to_backend`,
`hostcopy`, `hostcopy!`), `pointwise.jl` — GHSO2's node-local algebra
ported from `notes/pointwise-ghso2.jl`, plus the expanded momentum
equation this package discretises (`metric_derivatives`,
`gh_node_rhs_expanded`) and `gh_node_source` — `stencils.jl` — the
centered finite-difference and Kreiss–Oliger weights at order `q`, built
in `Rational` and rounded once into `T` by a `@generated` method, plus the
host-side `apply_stencil` the tests measure them with — and, from step 3,
the mesh-side physics: `evolution.jl` (the fused right-hand-side kernel in
`CODE.md`'s streaming order, `GHProblem`, `gh_rhs!`, the speed kernel and
`gh_dt`, `convergence_rate`), `initialdata.jl` (`GHCase`, the three
cases, the uniform forest builder, and the one conversion of
`SpacetimeMetrics`' derivative index order), `gauge.jl` (`isharmonic`,
`isstatic`, the sampled `Hsrc` and the refusal of a moving non-harmonic
background) and `boundaries.jl` (the time-dependent Dirichlet hook) —
and, from step 4, `constraints.jl`: the gauge-constraint kernel, the ADM
one (every second derivative of `g_ab`, with the `∂_t` blocks
reconstructed from the evolution equations and the four-dimensional Ricci
tensor assembled rather than reduced), the `AllPoints` mask and the
`is_evolved` predicate step 5 extends, `masked_norms` and
`constraint_norms`. Step 4 also added `metric_derivatives_along` to
`pointwise.jl` (the same chain rule along **one** direction, which is how
the monitors reach `∂_t α`, `∂_t β^j` and `∂_t √γ`) and a
`refined = true` forest to `initialdata.jl`.

From step 5 there is a **black hole**: `interior.jl` (the `C²` profiles
`w` and `ρ`, the core rule, `HoleCenter`, the horizon's analytic
coordinate radii, the `InteriorMask` and `ShellMask` every norm takes,
and `check_interior_radii` — the two placement bounds, asserted wherever
a problem is built and therefore after every regrid) and `driver.jl`
(`evolve!`, the chunked loop with the CFL recheck and the per-chunk
analysis record, `observer`, and GHSO2's discrete-gradient `Π`
post-pass). `evolution.jl` grew the `(INTERIOR)` term, a fifth `Val`
carrying the interior *variant*, and the `:pasted` variant's
`step_limiter!`; `constraints.jl` grew the error kernel and
`error_norms`; `gauge.jl` grew the position-dependent `γ0` profile;
`initialdata.jl` grew `hole_forest` and the two hole cases.

From step 6 there is a **mesh the code chose**: `refinement.jl` (the
`Refinement` parameters a case carries, `lohner` with the global
amplitude, the masked `field_scales`, the `τ` kernel into `diag`'s
fourteenth slot, `tau_max`, the four marks with the box keyed on
`coarsen_tol`, the derived `horizon_floor_level`, the `LevelBounds` the
floor and the ceiling are applied through, the travelling margin,
`refinement_centroid`, and `indicator_flags`/`gh_indicator!`, the two
entry points the cycle and the driver call). `driver.jl` grew `adapt` (the
initial-data cycle) and `regrid` (the branch at every chunk boundary but
the last), and the record grew `τ_max`, the centroid and its distance from
the analytic center.

From step 7 the code **knows where the horizon is**: `horizon.jl` —
`locate_block` and `interpolate`/`interpolate_grad` (the *stopgap* point
interpolator, `find_leaf` then a tensor-product Lagrange window of
`q + 2` points, batched and threaded over a host array, with the
footprint guard that refuses a query reading inside `r_1`),
`GHADMProvider` (the batched `ADMVars` provider, `Float64` out whatever
the run computes in, with a one-entry cache because `KorzynskiSpin` asks
for `γ` and `K` in two calls with the same points), `find_gh_horizon`
(the fast flow, the proper area, `M_irr`, the Korzyński `J` with its
axis, `M_ch`) and `Horizon`, the cadence and resolution the case carries.
`initialdata.jl`'s `GHCase` grew a sixth type parameter for it and
`with_horizon`; `driver.jl`'s record grew the horizon rows, seeded from
the previous find. There is no I/O and no CLI: those are step 9.

What exists in `test/` is `precision_tests.jl`, `prerequisite_tests.jl`
(the pinned TreeAMR still exports the names the design calls, a
`SpacetimeMetrics` background compiles and runs as a kernel argument on
`CPU()` filling a field set bit-for-bit as a host loop does, and the
*unexported* metric wrappers `gauge.jl` dispatches on still exist), the
pointwise pair — `pointwise_tests.jl` and `pointwise_identity_tests.jl`
over a shared `pointwise_backgrounds.jl`, which is where the six
backgrounds of `CODE.md`'s table and the analytic data are built —
`stencils_tests.jl`, which evaluates no background at all and costs
seconds, and step 3's five: `gauge_tests.jl`, `initialdata_tests.jl`,
`evolution_tests.jl` (the kernel against `gh_node_rhs_expanded`, and the
properties nothing else can state), `convergence_tests.jl` and
`noise_tests.jl`, over the helper `evolution_cases.jl` — the runs
themselves, which live in `test/` because what they wrap is the
integrator loop and `driver.jl` is step 5's — and step 4's four:
`constraints_tests.jl` (both monitors, against `ddmetric` on analytic
data and against the mesh), `interface_tests.jl` (the interface-order
table on the two-level mesh), `type_tests.jl` (`Float32` end to end,
`Float32x2` for the algebra) and `threading_tests.jl` over the standalone
`thread_workload.jl`; and step 5's two — `interior_tests.jl` (the
profiles, the core rule, the masks, the radius assertions firing, and one
right-hand-side evaluation with the layer on a mesh) and
`driver_tests.jl` (the runs: the record, the order on the frozen
hierarchy, the three variants, the drift, the `Π` post-pass) — over the
hole fixture in `evolution_cases.jl`; and step 6's `refinement_tests.jl`
(the Löhner algebra and its global floor, the four marks on a `τ` field
written by hand, the mask, the level floor and the ceiling, the
initial-data cycle, and two short adaptive runs — one whose mesh does not
move and one whose does), over that file's second hole fixture and the
three new helpers in `evolution_cases.jl`; and step 7's
`horizon_tests.jl` (the interpolator's exactness and its rate, the
footprint guard, Kerr's horizon from a displaced guess on both the step-5
fixture and the mesh the indicator chose, and the record's horizon rows at
the case's cadence). **`test/hole_runs.jl` is a standalone script, not
part of the suite**: the `t = 50 M` runs, `q = 4`, the two harmonic
charts, the indicator's calibration and — from step 7 — the horizon
section (Kerr's numbers at `a = 9/10` and in the harmonic chart, on meshes
of a thousand blocks) are minutes rather than seconds, and its numbers are
in `CODE.md` with the command that produced them.
`Project.toml` carries the `[sources]` pins and CI is in place — but
**the CI matrix is temporarily reduced** (2026-09-19): Julia 1.11 and code
coverage are both dropped, each with the removed lines and the reason in a
comment at its site in `CI.yml`. 1.11 fails only
`pointwise_tests.jl:480` (176 bytes where zero is claimed); coverage
computes *wrong numbers* on GitHub's runners while the identical
instrumented suite passes on Symmetry. Both are owed back. There is
no `Manifest.toml` (deliberately, and permanently: it is what makes the
clean-checkout check below mean something), no `bin/`, and there is now a
remote — `git@github.com:eschnett/TreeGeneralizedHarmonic.jl.git`.

The suite is **3444 assertions in 12m31** at one thread and **8m38** at
four on the development machine (step 6 measured 2968 in 13m35 / 10m45
here and 29m32 / 20m29 on Symmetry; the wall clock went *down* while the
count went up, so read each step's numbers as that step's rather than as a
regression). Most of it is **compilation**, and the things that pay for it
are, in order: `SpacetimeMetrics`' nested forward-mode passes for six
backgrounds at two precisions (step 1's cost, unchanged); a
right-hand-side kernel per `(q, has gauge source, has dissipation,
interior variant, T)`; the ADM constraint kernel, whose first
specialisation is **18 s** and each further one about 4 s; and the four
files whose cost is *arithmetic* rather than compilation —
`driver_tests.jl` (the suite's black hole, 1m59 / 44.8 s),
`interface_tests.jl`, where the ghost fill at `p = 6` is 79 % of every
evaluation, `refinement_tests.jl` (22.1 / 12.8 s) and `horizon_tests.jl`
(28.7 / 12.8 s, of which the two evolutions are most: the interpolator's
own claims are under two seconds and a *find* is a tenth of one).
Several files are over `PLAN.md`'s 30 s rule of thumb and `CODE.md`
records each with what it buys. Before adding a row anywhere, price it: a new `q` or a new element
type is a new kernel; a new background is a new dual pass; a resolution
added to an interface sweep is `N⁴` of ghost filling. Before adding a test
that differentiates a background, look at what `pointwise_backgrounds.jl`
already computes in one pass.

## Commands

The full suite, from the package root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

and the same at four threads — `Pkg.test` does not inherit `-t`, so it
has to be passed explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

The clean-checkout check, which is what the `[sources]` pins exist for: a
tree with no `Manifest.toml` resolves TreeAMR from the registry and the
other three from GitHub, and passes. From 2026-09-21 it is a real check —
every source is public, so it works anonymously, which is what CI does:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

It runs from a git worktree as happily as from the checkout, which is how
the per-step agents work; the `[sources]` pins mean every worktree
resolves the same two branches.

The thread workload runs on its own, which is how a digest mismatch is
bisected — `test/threading_tests.jl` starts exactly this in a subprocess:

```bash
julia --project=. -t 4 test/thread_workload.jl
```

The black-hole runs that are too long for the suite — the default margin
`m = 8` at `q = 4`, the `t = 50 M` run of all three interior variants,
the two harmonic charts, and from step 6 the indicator's calibration and
its adaptive run — are a **script**, run by hand, with its numbers
recorded in `CODE.md` under "Measured results" (added in step 5). It takes
an optional list of sections (`order`, `long`, `charts`, `indicator`,
`horizon`):

```bash
julia --project=. --threads=4 test/hole_runs.jl
julia --project=. --threads=4 test/hole_runs.jl indicator
julia --project=. --threads=4 test/hole_runs.jl horizon
```

The `horizon` section (added in step 7) is Kerr's `A`, `M_irr`, `J` and
`M_ch` from sampled data in both charts and at `a = 9/10`, plus the
horizon rows of a `t = 10 M` run; its two spinning-hole meshes are about a
thousand blocks each, which is why they are here and not in the suite. It
takes about eight minutes at four threads.

**On Symmetry** (added in step 6, and step 9 writes the batch job for
real): the suite and the long studies run there as one SLURM job each on a
64-core EPYC node, which is what makes them parallel — a node *core* is
about **twice as slow** as the development machine's, so the cluster buys
throughput and not wall clock (measured in step 6: the same 2968
assertions in 29m32 / 20m29 at one and four threads there against
13m35 / 10m45 here). One quirk that
will bite a batch script: **do not export `JULIA_EXCLUSIVE=1` for the
one-thread suite.** A one-thread parent then pins itself to a single CPU,
and `test/threading_tests.jl`'s four-thread subprocess inherits the
affinity mask and aborts with `Too many threads requested for
JULIA_EXCLUSIVE option` — the environment, not the code. The `symmetry-hpc`
skill has the rest of the cluster's mechanics.

Later: the CLI (`julia --project bin/gh.jl --case=boosted_kerr …`) and
the viewers (`julia --project=bin bin/visualize.jl`) arrive in step 9,
and device tests behind
`TREEGH_TEST_BACKEND` (`cuda` on Symmetry's H200 is the requirement,
`metal` on the development machine is desirable) in an environment of
your own that has the device package, in step 9. Neither this package
nor TreeAMR depends on a device package. The `symmetry-hpc` skill has
the cluster mechanics (modules, SLURM, NUMA, precompilation).

## Things that will bite

Carried over from TreeAMR, TreeWave and TreeHydro where they apply, plus
what is specific to a GR code. Each is in `CODE.md` with its reason.

- **Three dependencies are pinned to GitHub `main`, not to the local
  checkouts; TreeAMR is not one of them any more** (2026-09-21).
  `SpacetimeMetrics`, `ApparentHorizonFinder` and `KorzynskiSpin` are what
  `Project.toml`'s `[sources]` entries resolve, so `~/src/jl/…` is *not*
  what the tests see; an unpushed change there is invisible here, and the
  local SpacetimeMetrics checkout has been behind `main` before. Read what
  Pkg installed under `~/.julia/packages/` when in doubt about an API. Say
  so rather than editing a checkout and assuming the tests see it.
  **TreeAMR now comes from the General registry at `0.1.1`**, so an
  unreleased change there is invisible too — releasing is what publishes
  it. `test/prerequisite_tests.jl` is what notices when a moving branch
  drops a name.
  **`KorzynskiSpin`'s repository became public on 2026-09-21**, so its URL
  is plain `https`, the read-only deploy key and the `ssh-agent` step and
  `JULIA_PKG_USE_CLI_GIT` are all gone from `CI.yml`, and an anonymous
  clean checkout and a fork's pull request both resolve it — the
  clean-checkout check below finally proves what it claims. Do not vendor
  a package and do not add a local-path source.
- **`[sources]` is why the Julia floor is 1.11**, and it cannot be lowered
  yet. The section was introduced in 1.11, so the floor drops only when
  every entry goes: that needs `KorzynskiSpin` registered in General (it
  is not registered at all) and `ApparentHorizonFinder` **2.1** released
  (General carries `2.0.0`, and `[compat]` here asks for `2.1`).
  `SpacetimeMetrics` is registered at the bound this package asks for, so
  its pin is a choice rather than a necessity.
- **`notes/` is read-only.** The copies carry their provenance; when
  `CODE.md` departs from them, `CODE.md` says so. Do not "fix" a copy.
- **Two derivative index conventions.** `SpacetimeMetrics.dmetric`
  returns `dg[a, b, c] = ∂_c g_ab` (derivative axis *last*); GHSO2's
  pointwise algebra uses `dg[a, b, c] = ∂_a g_bc` (derivative axis
  *first*). Convert in one place, `initialdata.jl`, and test the
  conversion; a wrong convention produces a solution that looks right
  in Minkowski and wrong everywhere else.
- **The packed component order is** `(tt, tx, ty, tz, xx, xy, xz, yy,
  yz, zz)`, column-major lower triangle, GHSO2's. `h` is variables
  `1:10`, `Π` is `11:20`. Nothing indexes a component by a literal
  outside `pointwise.jl`'s pack/unpack helpers.
- **Off by `G`.** Owned point `i` is stored at `i + G[d]`;
  `coordinates(fs, b, idx)` takes **stored** indices; a kernel launched
  by `map_blocks!` (default range) gets the owned index and adds `G`.
  Every stencil reaches `±(q/2 + 1)` and `G = q/2 + 1` is exactly that.
  Getting this wrong produces plots that look almost right. The
  right-hand-side kernel does the addition **once**, into a linear index,
  and the stencils step by a stride per axis — same element, a quarter of
  the time (measured in step 3), and one place to get it wrong instead of
  1600.
- **`N = 8` does not exist at every order.** TreeAMR's vertex invariant
  is `N ≥ 2G + 2`, so `q = 4, 6, 8` need `N ≥ 8, 10, 12`. The tests use
  `N = 10` at `q = 6`; `PLAN.md`'s "`N = 8`, `q = 2, 4, 6`" was one row
  wider than the mesh allows, and `FieldSet` says so rather than running.
- **KernelAbstractions refuses a `return` in a kernel** — anywhere in the
  body, closures included, which is what `ntuple(Val(10)) do v … end` is.
  End the block with the value instead. The error names the kernel and
  arrives at precompilation, so it is cheap; it is here because the
  package's own convention asks for an explicit `return` everywhere else.
- **The RHS never mutates `u`.** The interior layer is a term of the
  right-hand side, `du = w F(u) − ρ (u − u_exact)`. Only the `:pasted`
  variant writes the state, and only from RK4's `step_limiter!`. Do not
  add a third place.
- **`F` is never evaluated where `w = 0`.** The frozen core holds
  finite, stale data by design — the analytic solution is singular
  inside it — and `F` of that data may be `NaN`; `0 · NaN = NaN`. The
  kernel branches on the core predicate before it touches a stencil.
- **The interior is masked in every norm and in the indicator.**
  Constraint, error and speed kernels and the Löhner indicator write
  zero for `r < r_1`. If a number looks wrong near the hole, check the
  mask before the physics. **Write the masked slot through a branch, not
  as `keep * value`** (step 5): the masked region may hold a `NaN`, and
  `0 · NaN = NaN` — the same trap as the frozen core's, met in the
  monitors.
- **Every path that puts the analytic solution on the grid goes through
  the core rule**, `core_position` — the initial data, the error
  reference, the `:pasted` limiter, the Dirichlet hook *and the
  gauge-source sample*. The last is the one that was forgotten once
  (step 5): `H^a = −Γ^a[g_exact]` is sampled at every owned point
  including the center, where `KerrSchild`'s `k^i` divides by zero, and
  the `NaN` surfaces in the constraint monitors and nowhere else.
- **The mask plumbing.** A kernel takes a `mask`, asks
  `is_evolved(mask, x)`, writes zero where it says no and a `1`/`0`
  indicator into `DIAG_MASK`; `masked_norms` divides by the **evolved**
  volume, not the domain's, so masking a region out does not make the
  number smaller by diluting it with zeros. `AllPoints` is the trivial
  mask, `InteriorMask` is `r ≥ r_1` at that call's `t`, and `ShellMask`
  is a band — which is how the three interior variants are compared over
  "the `G` points outside `r_1`" and how the gauge drift is read at the
  horizon.
- **The `diag` slots are a contiguous-range interface.** `block_mapreduce`
  reduces an integer or a *contiguous* range of variables and refuses
  anything else — a device cannot be handed an arbitrary index vector cell
  by cell — so `DIAG_CGH` (four) and `DIAG_MOM` (three) are the *first* of
  a run and are indexed as `DIAG_CGH + a - 1`. Adding a slot in the middle
  of either run is how that breaks.
- **A vertex-centered field set stores the shared upper plane, and no
  kernel writes it.** The stored size is `N + 2G + 1` per axis, and the
  point at `N + 2G + 1` belongs to no owned range: `map_blocks!` never
  reaches it and the Dirichlet hook or a neighbour's ghost exchange fills
  it. So a test that reads `fs.work[:, :, :, v, :]` sees a plane of zeros
  that means nothing; use `interiorview(fs, b, v)`. `GHProblem` refuses a
  field set that is not vertex-centered, because `point_position` — the
  one place an owned index becomes a position for the masks and the
  interior profiles — assumes it.
- **The ADM monitor costs 18 s to compile**, and about 4 s for every
  further specialisation of `(G, q, has gauge source, T)`. It is a
  four-dimensional Ricci tensor out of a hundred second derivatives, which
  is the opposite of the right-hand side's streaming order and
  deliberately so: it runs once per chunk, not once per stage. Before
  putting it in a new test row or a new loop, count the specialisations.
  It is *not* in `test/thread_workload.jl` for exactly this reason.
- **A coarse-fine face is expensive in time as well as in order.** On the
  two-level mesh at `p = 6` the ghost fill is **79 %** of a
  right-hand-side evaluation (22 % on a uniform mesh), because a
  tensor-product prolongation reads `6³ = 216` coarse points per fine
  ghost point. A study on a refined mesh costs roughly four times what the
  same study costs uniform, which is why `interface_tests.jl` runs an
  eighth of a crossing on `N = 8, 10, 12` and not more.
- **A spherical frozen core cannot hide Kerr's singularity in the
  harmonic chart at `a = 0.9`** (found in step 5, and it is the
  proof-of-concept case). Both Kerr charts are singular on the equatorial
  **disk** of coordinate radius `|a|`, not at a point, so the core needs
  `r_0 > |a|`; the placement bound needs `r_0 < r_h,min`, and harmonic
  Kerr's `r_h,min = √(M²−a²) = 0.436` is smaller than `0.9`. A ball fits
  only where `a < M/√2`. `check_interior_radii` refuses it by name
  (`singular_radius`); `CODE.md`'s "Open questions" has the two ways out.
  Kerr-Schild at `a = 0.9` runs, because `r₊ = 1.436 > 0.9`.
- **The three interior radii are asserted at every regrid**: `r_1`
  inside the horizon by `m` spacings of the blocks containing it
  (`m = 8` by default, never below `G + 1`); `r_1 − r_0` at least
  `2(G + 1)` such spacings; the refinement's level floor covering the
  horizon. If one fires, raise the floor, shrink the layer or widen the
  floor's shell — do not remove the assertion. For `a = 0.9` in harmonic
  coordinates the horizon's smallest coordinate radius is about
  `0.44 M`, and these bounds are what set the finest spacing.
- **The refinement is an error indicator with three geometric
  corrections, and all four matter.** TreeWave's Löhner indicator with
  a *global* noise floor (the local floor refines `1e−16` tails), the
  box keyed on `coarsen_tol` (keying it on `refine_tol` silently
  disables the travelling margin), the mask inside `r_1` (stale core
  data has steep meaningless differences), the level *floor* around the
  horizon (it is what guarantees the interior's resolution) and the
  level *ceiling* at the outer boundary (the Dirichlet mismatch is a
  kink the indicator scores). Ghosts must be filled before flagging.
  Convergence claims are made on a hierarchy frozen at `t = 0`, not on
  the adapting mesh.
- **The reference amplitude is one number for all ten components, and
  that is not an approximation** (step 6). In the gauge wave's chart six
  components are *identically zero*, so a per-component reference is
  exactly zero, the floor with it, and that component's dust scores
  `τ ≈ 1` over the whole domain — TreeWave's blast-wave trap in a chart
  instead of in initial data.
- **The travelling margin is dilated inside `refine_flags`, and
  `regrid!` is then given `buffer = 0`** (step 6). TreeAMR's
  `buffered_flags` promotes every leaf a dilated box reaches whatever the
  application said about it, so a margin applied afterwards refines the
  blocks the ceiling just capped — measured, 12 of 64 boundary root
  blocks. Dilate, *then* clamp.
- **The floor and the ceiling together are a statement about the box.**
  `block_level_bounds` throws when a block is both in the horizon shell
  and within the boundary margin, because the region that must be
  resolved has met the region that must stay coarse. That is what a box
  of half-width `5/2 M` around a hole whose horizon is at `2 M` does:
  the refinement's reference configuration uses `5 M`, and step 5's
  fixture — which has no refinement — keeps `5/2`. The floor's *level* is
  derived from the interior's own radii, so a `maxlevel_cap` below it is
  refused rather than discovered later as a `check_interior_radii`
  failure.
- **2:1 balance overrides the ceiling, and that is TreeAMR's invariant.**
  A deep enough hierarchy in a small enough box pushes refinement out to
  the boundary through balance alone. If the boundary blocks are not at
  the coarsest level, count the levels before suspecting the ceiling.
- **`ρ_max` is bounded by RK4's stability**, about `2.8/dt` on the
  negative real axis; the driver sets `ρ_max · dt = 1` per chunk. A run
  that blows up in the layer after raising `ρ_max` has found the
  integrator, not the physics.
- **The horizon finder reads only the evolved region, and it says so by
  throwing.** The interpolation window is `q + 2` points per axis, `G` of
  them on each side of the query's cell, so a query *outside* `r_1` can
  still read *inside* it. The guard is on the footprint and not on the
  query point, and it is exact (the footprint is a lattice, so the nearest
  of its points to the center is the per-axis nearest). A find that fails
  because of it is recorded in the run's record as
  `horizon_success = false` with the message in `horizon_note`, never
  thrown out of `evolve!`: the horizon is a diagnostic. If a find fails,
  read `r_1 + (q+2)h/2` against the horizon's smallest coordinate radius
  before suspecting the finder — and remember that the fast flow's
  *transient* dips below its seed, which is how the harmonic chart at
  `h = 5/64` fails and the same chart at `5/128` does not.
- **Ghosts must be filled before anything is interpolated**, with that
  call's hook. `find_gh_horizon` scatters and fills; `gh_adm_provider` and
  `interpolate` do not, and an unfilled halo makes the metric garbage
  exactly at the block faces — which the fast flow then walks into the
  layer to escape.
- **The Korzyński spin is what a find costs**, not the interpolation:
  `16×`, `45×` and `105×` a right-hand-side evaluation at
  `N_ah = 12, 16, 20`, against `1.7×`, `2.7×` and `3.6×` for the find
  itself — one batch of 496 interpolated `ADMVars` is 0.26 ms. Lower the
  cadence or pass `spin = false` before lowering `N_ah`. Its `unif_tol` is
  `1e-8` here and not the library's `1e-13`, because interpolated data has
  a residual floor of its own (`2.2e−5` at `h = 5/64`): the tighter
  tolerance reports `success = false` on every find and returns the same
  `J`.
- **The analysis quantities are part of the deliverable**: masked
  constraint norms, the interior residual, horizon location, area,
  `M_irr`, `J`, `M_ch`, the refinement centroid, mesh statistics, at
  every chunk boundary, appended to the run's time series. A run that
  finishes without them is not a result, and a test that asserts only
  "it ran" is not a test.
- **`Float64` on the H200 is the requirement; `Float32` is a
  nice-to-have.** Record a `Float32` failure, do not fix it at
  `Float64`'s expense. The type-generic discipline stays anyway: a
  decimal literal in a `T` expression is a leak — `T(1//2)`, not `0.5`;
  `oftype(x, 2)` inside closures.
- **The RHS kernel is written in streaming order, and stays that way.**
  Coefficients and `∂_i h` once per point; then per component, stencils
  formed and consumed immediately; then the source. Never build an
  `SVector` of all derivatives — that is about 140 `Float64` values,
  over a GPU thread's 255 registers, and it spills. GHAccel's ten-field
  kernel fit only as fully scalarised generated code; spills show as
  `ld.local` in the PTX. GPU kernel *efficiency* beyond this order is a
  research project, not a milestone; G6 measures, it does not tune.
- **Hooks depend on time.** `dirichlet(case, t)` is built at each call.
  It goes to `fill_ghosts!` inside the RHS, to `regrid!`, and to
  `adapt_to_initial_data!`, each with that call's `t`. Forgetting the
  second is the bug that arrives one chunk late; passing a stale `t` is
  the bug that arrives as a boundary reflection.
- **A moving non-harmonic background is refused.** A boosted
  Kerr-Schild has a time-dependent gauge source that a per-chunk sample
  cannot represent. Use `boost(Harmonic(M, a), v)`, which is harmonic
  (`H ≡ 0`) — a boost preserves the harmonic condition. The refusal's
  message says this; do not weaken it.
- **`Val`s once per chunk.** `G`, `q`, "has gauge source", "has
  dissipation" and — from step 5 — the interior *variant* (`:none`,
  `:damped`, `:pasted`, `:frozen`, which is "has interior" and *which* in
  one parameter) are `Val` parameters built in `GHProblem`'s constructor.
  Building them per evaluation recompiles or dispatches dynamically on
  every RK stage. The price is paid at compile time instead: a test row
  at a new `q` is a new kernel, which is most of what
  `evolution_tests.jl`'s 56 s are, and each interior variant is another
  one. The interior itself is rebuilt per chunk anyway, because
  `ρ_max = 1/dt`; `with_interior` shares the field sets and the sampled
  gauge source rather than rebuilding the problem, which would re-sample
  `H_a`.
- **KernelAbstractions refuses a `return` statement anywhere in a kernel
  body**, closures included. That is why the streaming right-hand side
  lives in `gh_rhs_at_point`, a plain `@inline` function: the frozen
  core's branch has to be an `if` *around* the whole computation, and it
  cannot be an early exit.
- **A `@generated` method must not convert to the caller's type.** Its
  generator may only call methods that existed when the generated function
  was *defined*, and this package is precompiled long before a driver loads
  MultiFloats: a generator that wrote `T(w)` for a rational `w` worked at
  `Float64` and `Float32` and threw "the applicable method may be too new"
  at `Float32x2`. Emit the exact integers and let the call site divide.
  The symptom only appears when this package is loaded *before* the type's
  package, which is what `runtests.jl` does — so a test file that loads
  MultiFloats above `TreeGeneralizedHarmonic` hides it.
- **The stencil weights are for unit spacing, and the dissipation's
  factor of `h` is a single one.** `derivative_weights` is divided by
  `h^m` at the call site; `dissipation_weights` is multiplied by `ε/h_d`,
  because the `h_d^{2r−1}` of `CODE.md`'s formula cancels all but one
  power against `(D₊D₋)^r`'s own `h_d^{−2r}`. The weights already carry
  `(−1)^{r+1}`, the `2^{−2r}` and therefore the *damping* sign: Nyquist is
  damped at exactly `ε/h_d`. A run that blows up faster the larger `ε_KO`
  is has that sign backwards, and `test/stencils_tests.jl` asserts it as
  an exact rational eigenvalue.
- **The recipe near a hole is `ε_KO ≈ 0.5`, `γ0 ≈ 1/M`.** GHSO2
  measured both as requirements with the horizon in the domain
  (`notes/methods-ghso2.md`); a run that blows up at the sonic surface
  with `ε_KO = 0` is not a bug in the stencils.
- **The interface-order rule.** This system takes second derivatives,
  so prolongation order `p = q + 2` (order 6 at `q = 4`), or the global
  rate drops by one. Never give `ops` a default that hides it; TreeAMR
  refuses `G` below `p/2 − 1` and this package's `G = q/2 + 1` clears it.
- **Don't name a keyword `maxlevel`.** It shadows TreeAMR's exported
  `maxlevel(forest)` inside the function body. Use `maxlevel_cap`.
- **A callback must capture no `Type` and no host array.** Backgrounds,
  the interior profiles, the damping profile, the indicator's thresholds
  and floors become kernel arguments; they are `isbits` structs and
  tuples. The center is a function of `t`, never a mutated field.
- **Two spellings of one expression are not bit-identical — and neither
  are two call sites of one function.** The same arithmetic written twice
  — `gh_node_source` and the block it copies, `gh_fluxes` and
  `gh_node_rhs`'s fluxes — disagrees in the last place on this machine,
  because the compiler fuses a multiply and an add in one inlining context
  and not in the other. So does one body reached two ways:
  `metric_derivatives`'s wrapper and its coefficient-set method disagree
  on 3 of 12 points at `Float64` (measured in step 1). Compare any of
  these to roundoff against the scale that produced the number, never with
  `isequal`. The bit-identity that *is* an invariant is the same compiled
  code, at the same call site, at a different thread count, which is what
  `test/threading_tests.jl` will assert.
- **Never thread anything a TreeAMR callback can reach**, and never
  accumulate into shared state in a loop of your own: bit-identity
  across thread counts is the invariant, and `test/threading_tests.jl`
  is the only thing that will report a violation.
- **`Base` is not generic even though the mesh is.** MultiFloats defines
  no `rem`, no conversion to `Integer`, no `Float64(::Float32x2)`. Use
  `wrap` / `ceilint` / `floorint` / `tofloat64` from `precision.jl`.
- **Measured numbers go into `CODE.md`**, beside the prediction they
  confirm or correct, so a regression shows up as a changed number and
  not as a test that merely still passes.

## Conventions

Match TreeAMR's, since the four packages are read together:

- 4-space indent, wrap at about 80–90 columns.
- `return` on the last line of any non-trivial function.
- `ntuple(d -> f(d), Val(3))` rather than comprehensions in
  kernel-adjacent code; states are `SVector{10}`, tensors `SMatrix{4,4}`.
- Unicode in mathematical contexts (`α`, `β`, `γ`, `Π`, `∂ₜ`, `Σ`).
- Keyword-heavy driver signatures with no default for anything the caller
  must think about; `T` as a leading positional argument defaulting to
  `Float64`, `backend` a keyword defaulting to `CPU()`.
- `ArgumentError`s say *why*, not just what.
- Docstrings are prose-first: what it is, then why, pointing at `CODE.md`.
- **Testset names are claims**, each opening with a comment naming the
  failure mode it guards.
- Spec-first: when the implementation shows `CODE.md` was wrong or
  incomplete, amend it and say so in it — "(amended in step N)",
  "(measured in step N)" — rather than diverging silently.

## Repository facts

- **The remote is `git@github.com:eschnett/TreeGeneralizedHarmonic.jl.git`**,
  and the rule from the siblings applies: work on a branch, and do not
  push, open a pull request, or merge to `main` without being asked. Each
  step lands on `main` only after review.
- `TODO.md`, when it appears, is Erik's personal to-do list. **Do not
  modify it.** It is gitignored.
- `CODE.md`, `PLAN.md`, `README.md`, `notes/`, `src/`, `test/`,
  `.github/` and this file are committed. `Manifest.toml` files,
  `bin/output/` and `docs/build/` are gitignored — no `Manifest.toml` is
  tracked, which is what makes the clean-checkout check above mean
  something.
- Sibling checkouts: `~/src/jl/TreeAMR` (the mesh; read its `CLAUDE.md`
  and `CODE.md` for the API and its sharp edges), `~/src/jl/TreeWave`
  and `~/src/jl/TreeHydro` (the other applications; copy the *patterns*
  of `precision.jl`, `device.jl`, `refinement.jl`, `bin/backend.jl`, the
  viewers and the thread workload — do not depend on them),
  `~/src/jl/SpacetimeMetrics` (the backgrounds),
  `~/src/jl/ApparentHorizonFinder` and `KorzynskiSpin` (the horizon
  diagnostics, from step 7). `~/src/jl/GeneralizedHarmonicSecondOrder2`
  is the *origin* of `notes/`; the text to cite is in `notes/`.
