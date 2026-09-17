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

**G0 is done and G1 is half done — the pointwise algebra exists, the
stencils are next.**
`CODE.md` is complete and reviewed three times (2026-09-16): the expanded
form of the momentum equation, three dimensions only, a pointwise damping
layer instead of excision, a single boosted spinning black hole as the
proof-of-concept target, RK4 from OrdinaryDiffEq, the analysis quantities
as part of the deliverable, an error indicator for refinement, `Float64`
on Symmetry's H200 as the device requirement, no checkpointing, GPU
kernel efficiency deferred to a research project. `PLAN.md` breaks the
milestones G0–G6 into steps 0–10, each a brief for one agent with a fresh
context (see its "Running a step as an agent"); step 2 (the stencils) is
next. `notes/` holds the inherited documents.

What exists in `src/` is the module shell, `precision.jl` (the `Base`
bridges for software floating-point types), `device.jl` (`to_backend`,
`hostcopy`, `hostcopy!`) and `pointwise.jl` — GHSO2's node-local algebra
ported from `notes/pointwise-ghso2.jl`, plus the expanded momentum
equation this package discretises (`metric_derivatives`,
`gh_node_rhs_expanded`) and `gh_node_source`. There is still no mesh-side
physics: nothing in `src/` reads a field set.

What exists in `test/` is `precision_tests.jl`, `prerequisite_tests.jl`
(the pinned TreeAMR still exports the names the design calls, and a
`SpacetimeMetrics` background compiles and runs as a kernel argument on
`CPU()`, filling a field set bit-for-bit as a host loop does), and the
pointwise pair — `pointwise_tests.jl` and `pointwise_identity_tests.jl`
over a shared `pointwise_backgrounds.jl`, which is where the six
backgrounds of `CODE.md`'s table and the analytic data are built.
`Project.toml` carries the `[sources]` pins and CI is in place. There is
no `Manifest.toml` (deliberately, and permanently: it is what makes the
clean-checkout check below mean something), no `bin/`, and no remote.

The suite's cost is now dominated by **compiling** `SpacetimeMetrics`'
nested forward-mode passes for six backgrounds at two precisions — about
two minutes, against step 0's eight seconds, with the evaluation itself
in microseconds. Before adding a test that differentiates a background,
look at what `pointwise_backgrounds.jl` already computes in one pass:
`CODE.md`'s "Measured results" records what fusing them was worth.

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

The clean-checkout check, which is what the `[sources]` pins exist for:
a tree with no `Manifest.toml` resolves TreeAMR and SpacetimeMetrics from
GitHub and passes. This is what CI does, and a local run that passes
proves nothing about it:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

It runs from a git worktree as happily as from the checkout, which is how
the per-step agents work; the `[sources]` pins mean every worktree
resolves the same two branches.

Later: the CLI (`julia --project bin/gh.jl --case=boosted_kerr …`) and
the viewers (`julia --project=bin bin/visualize.jl`) arrive in step 9,
the thread-independence test in step 4, and device tests behind
`TREEGH_TEST_BACKEND` (`cuda` on Symmetry's H200 is the requirement,
`metal` on the development machine is desirable) in an environment of
your own that has the device package, in step 9. Neither this package
nor TreeAMR depends on a device package. The `symmetry-hpc` skill has
the cluster mechanics (modules, SLURM, NUMA, precompilation).

## Things that will bite

Carried over from TreeAMR, TreeWave and TreeHydro where they apply, plus
what is specific to a GR code. Each is in `CODE.md` with its reason.

- **TreeAMR and SpacetimeMetrics are pinned to GitHub `main`, not to the
  local checkouts.** `Project.toml`'s `[sources]` entries are what the
  tests resolve, so `~/src/jl/TreeAMR` and `~/src/jl/SpacetimeMetrics`
  are *not* what they see; an unpushed change there is invisible here,
  and the local SpacetimeMetrics checkout has been behind `main` before.
  Read what Pkg installed under `~/.julia/packages/` when in doubt about
  an API. Say so rather than editing a checkout and assuming the tests
  see it. The entries are also why the Julia floor is 1.11, and
  `test/prerequisite_tests.jl` is what notices when a moving branch drops
  a name.
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
  Getting this wrong produces plots that look almost right.
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
  mask before the physics.
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
- **`ρ_max` is bounded by RK4's stability**, about `2.8/dt` on the
  negative real axis; the driver sets `ρ_max · dt = 1` per chunk. A run
  that blows up in the layer after raising `ρ_max` has found the
  integrator, not the physics.
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
  interior" are `Val` parameters built in `GHProblem`'s constructor.
  Building them per evaluation recompiles or dispatches dynamically on
  every RK stage.
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
- **Two spellings of one expression are not bit-identical.** The same
  arithmetic written twice — `gh_node_source` and the block it was lifted
  out of, `gh_fluxes` and `gh_node_rhs`'s fluxes — disagrees in the last
  place on this machine, because the compiler fuses a multiply and an add
  in one inlining context and not in the other (measured in step 1).
  Compare such copies to roundoff, not with `isequal`. The bit-identity
  that *is* an invariant is the same compiled code at a different thread
  count, which is what `test/threading_tests.jl` will assert.
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

- **No remote yet.** When there is one, the rule from the siblings
  applies: work on a branch, and do not push, open a pull request, or
  merge to `main` without being asked. Each step lands on `main` only
  after review.
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
