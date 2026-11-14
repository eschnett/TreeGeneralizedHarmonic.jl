# Working notes for Claude in TreeGeneralizedHarmonic.jl

Read `CODE.md` first — it is the design document and states *why* things
are the way they are.

**Contents**

- [The files, and where new content goes](#the-files-and-where-new-content-goes)
- [What this package is](#what-this-package-is)
- [Commands](#commands)
- [Things that will bite](#things-that-will-bite)
- [Conventions](#conventions)
- [Repository facts](#repository-facts)

## The files, and where new content goes

- **`README.md`** — for users: what the package does, installation, a
  short example, the status, and a pointer to `CODE.md`. No design
  rationale beyond a short paragraph.
- **`CLAUDE.md`** (this file) — for agents, loaded into every session, so
  as short as it can be while complete: what to read first and this map, a
  few lines on what the package is, the commands, the conventions and
  sharp edges (each a short rule pointing at the `CODE.md` section that
  explains why), and the repository facts. This file is only about mechanics.
  No status narrative, file inventory, release history or detailed
  measurements; a rough duration that helps plan a run is fine.
- **`CODE.md`** — the current design and the reasons for it: purpose and
  requirements, the method, the code and its structure, the interfaces and
  their contracts, the tests, current measurements that support a current
  decision, known limitations, and open questions and rough ideas for the
  future. Present tense, without dates or step numbers on decisions; only
  what is not settled is marked, **(open)** or **(proposed)**. The test
  against `HISTORY.md`: would someone changing the code today need this?
  Then it belongs in `CODE.md`, rationale included.
- **`PLAN.md`** — the concrete plan: steps or milestones, each with what it
  delivers and how one knows it is done. A finished step is one line saying
  it is done and where its results went; the step numbers stay, since other
  documents cite them.
- **`HISTORY.md`** — how the package got where it is: who decided what and
  when, alternatives that were tried or rejected and why, measurements of
  code that no longer exists or that later measurements replaced, upstream
  bugs that have since been fixed, and the release history; by topic,
  following `CODE.md`'s sections, with dates. Where a reason lives there,
  `CODE.md` links to it.
- **`SINGULARITY_HANDLING.md`** — topical, the runs behind the black hole's interior — step 5's
  layer, steps 8a–8′, the single holes on the octant; the
  settings they recommend for a single hole are `CODE.md`'s "[Single black
  holes: recommended settings](CODE.md#single-black-holes-recommended-settings)".
- **`notes/`** — `notes/` holds the inherited documents, read-only:
  `GeneralizedHarmonicSecondOrder2` ("GHSO2") is unpublished, so the
  documents this package rests on are copied verbatim
  into `notes/` — read `notes/methods-ghso2.md` for the physics and the
  numbers this package compares itself against, and cite `notes/`, not
  the sibling checkout.

## What this package is

The third downstream application of
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl), and the first step
toward a production code: the vacuum Einstein equations in the
generalized harmonic formulation, second order in space and first order
in time, on TreeAMR's octree of uniform blocks. TreeAMR is the mesh and contains no physics;
[TreeWave](https://github.com/eschnett/TreeWave.jl) is the scalar wave
equation and [TreeHydro](../TreeHydro) is Newtonian hydrodynamics; this
package is general relativity. The formulation and the pointwise
algebra come from `GeneralizedHarmonicSecondOrder2` ("GHSO2"). The rest is `CODE.md`'s opening, [Goals](CODE.md#goals) and [Lineage:
what is inherited and what
changes](CODE.md#lineage-what-is-inherited-and-what-changes).

Three rules follow from `CODE.md` and govern every change here:

- **No mesh machinery.** If a change is about trees, ghost cells,
  interpolation or reductions, it belongs upstream in TreeAMR. See `CODE.md`'s [Scope and
  non-goals](CODE.md#scope-and-non-goals) and [Upstream
  prerequisites](CODE.md#upstream-prerequisites).
- **The interior is pointwise and generic, and there is no excision.**
  Do not add an excision mask or a one-sided stencil:
  excision is `CODE.md`'s fallback, priced there and decided by step
  8g's host-side test, not built. See `CODE.md`'s [The
  interior](CODE.md#the-interior-a-pointwise-damping-layer) and
  [Possible extensions](CODE.md#possible-extensions).
- **Inherit GHSO2's algebra, do not re-derive it.** `pointwise.jl` is a
  port of `notes/pointwise-ghso2.jl`; a change to the equations there is
  a change to a validated result and needs the corresponding test
  against `SpacetimeMetrics` automatic differentiation to say so. See `CODE.md`'s
  [Lineage](CODE.md#lineage-what-is-inherited-and-what-changes) and [The
  equations](CODE.md#the-equations).

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
tree with no `Manifest.toml` resolves the five pinned packages from
GitHub and TreeAMR from the General registry (TreeIOHDF5 among the five, and HDF5 with its
binary library through it), and passes. It is a real check —
every source is public, so it works anonymously, which is what CI does:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

It runs from a git worktree as happily as from the checkout, which is how
the per-step agents work; the `[sources]` pins mean every worktree
resolves the same branches.

The quick tier, which is what GitHub CI runs (`test/tiers.jl`): every testset but the hole evolutions, each of which is
then one skipped test, counted as "Broken" and listed at the end:

```bash
TREEGH_TEST_TIER=quick julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

The full suite as a commit status: `ci/full-suite.sh`
runs the clean-checkout check above on a commit at four threads and, with
`--post`, sets the commit's status `full-suite/<host>` to pending and then
to the result — the count, the time, the threads and the CPU. Posting is
public: ask Erik each time, as for a push. The commit must be on GitHub.
`ci/full-suite-symmetry.sh submit` does the same on an `amdq` node (one fixed
tree under `/home/eschnetter/claude/full-suite/`, the log and the result under
`/mnt/beegfs/eschnetter/claude/full-suite/<sha>/`), and `collect` posts its
result afterwards, since the cluster has no `gh`:

```bash
ci/full-suite.sh --post
```

```bash
ci/full-suite-symmetry.sh submit --post
```

```bash
ci/full-suite-symmetry.sh collect --post
```

The MPI workload runs on its own too, from an
environment that has MPI and the package — `test/mpi_tests.jl` starts
exactly this from the suite's sandbox, beside the suite where the machine
has room and to its end *before* the suite where it has not
(`TREEGH_TEST_MPI_CONCURRENT=0` or `1` decides it by hand; why before the
suite: `CODE.md`, "Running distributed"). It is
skipped on a machine with less than 24 GB — three ranks of about 3 GB
beside the suite, which a macOS runner (7 GB) and the serial Linux runner
could not hold — unless `TREEGH_TEST_MPI=1`, which CI sets for
its `mpi` job. That job
runs `TREEGH_TEST_TIER=mpi`, the comparison and nothing else. It
writes `OUTDIR/n0.txt` … `n3.txt`, the serial run and the three rank counts:

```bash
mpiexec -n 3 julia --threads=1 --project=<env> test/mpi_workload.jl OUTDIR
```

Where a step goes over MPI, rank by rank — the scatter, the fill and each of
its stages' local groups, packs and unpacks, the kernel, a right-hand side, an
RK4 step, and each rank's load (`CODE.md`, "Running
distributed"). On the CPU from the MPI environment, on H200s through
`bench/symmetry_mpi_breakdown.sh`:

```bash
mpiexec -n 2 julia --threads=1 --project=<env> bench/mpi_breakdown.jl mpi=true L=16 N=16 radii=8,4,2 reps=2
```

The thread workload runs on its own, which is how a digest mismatch is
bisected — `test/threading_tests.jl` starts exactly this in a subprocess:

```bash
julia --project=. -t 4 test/thread_workload.jl
```

The black-hole runs that are too long for the suite — the default margin
`m = 8` at `q = 4`, the `t = 50 M` run of all three interior variants,
the two harmonic charts, and the indicator's calibration and
its adaptive run — are a **script**, run by hand, with its numbers
recorded under "Measured results" — the indicator's and
the horizon's in `CODE.md`, the interior's in `SINGULARITY_HANDLING.md`. It takes
an optional list of sections (`order`, `long`, `charts`, `indicator`,
`horizon`, `bounds`; and `leakage`, `calibration` and `tracked`, which are
not in the default list). An option `key=value` whose key is a section's name selects
that section's subset and so names the section — `bounds=damped6` runs
the one row and nothing by default:

```bash
julia --project=. --threads=4 test/hole_runs.jl
julia --project=. --threads=4 test/hole_runs.jl indicator
julia --project=. --threads=4 test/hole_runs.jl horizon
julia --project=. --threads=4 test/hole_runs.jl bounds=cost,damped6,pasted8
```

The `horizon` section is Kerr's `A`, `M_irr`, `J` and
`M_ch` from sampled data in both charts and at `a = 9/10`, plus the
horizon rows of a `t = 10 M` run; its two spinning-hole meshes are about a
thousand blocks each, which is why they are here and not in the suite. It
takes about eight minutes at four threads.

The `leakage` section is eighty-eight `2 M` evolutions
on a 512-block mesh — a ripple inside the horizon against the same run
without it — and is **not** run by default: it is run as **one Symmetry
job**, `julia --project=. --threads=64 test/hole_runs.jl leakage` on a
64-core node, where it fans its groups out into sixteen subprocesses
(45m47 on one `amddebugq` node, inside that queue's hour with a quarter to
spare — a longer `t_end` or a third order would not fit). The
`symmetry-hpc` skill has the cluster mechanics, as for the suite below. A
subset of it runs locally through `key=value` options, one group after the
other:

```bash
julia --project=. --threads=4 test/hole_runs.jl leakage q=2 d=4 eps=0,1/2 lambda=2 t_end=1/4
```

Its predictions are a script of their own, a minute at one thread:

```bash
julia --project=. test/dispersion.jl
```

The `bounds` section has three rows, selectable as
`bounds=<row>,…` so that they split across batch jobs: `cost` (the stage
limiter against a right-hand side, seconds), and `damped6` and `pasted8` —
step 5's two failing runs, each three times (no projection, the proposed
gate, the widest gate), with radial bins of the state's validity and a
step-by-step replay of the fatal chunk. Those two took 14 and 26
minutes on one Symmetry node; `TREEGH_BOUNDS_TEND` shortens them for a
smoke test and `TREEGH_BOUNDS_AUTOPSY=1` forces the replay.

The `calibration` section is step 8c's experiments on
the step-5 fixture — the E3 sweep to `3/20 M`, the `n_L × ρ_max` scan with
an inexact target, the `ε_KO(r)` profile, the E1 and E2 targets, E0's hard
step on 8a's uniform mesh, the controls, and the survivors to `50 M` — in
groups, `calibration=<group>+<group>` (`+` or `,`): `sweep`, `scan`,
`profile`, `targets`, `controls`, `e0`, `e0p`, `exact`, and `long1` …
`long8` for the lists of `50 M` runs written into the script. With sixteen
or more threads it fans its runs out to subprocess workers (four threads a
screen, eight a `50 M` run, sixteen an E0 pair), whose logs carry one row
per chunk; each group was one `amddebugq` job — a screen group about 20
minutes, a `50 M` group of eight about 30 (`symmetry-run.sh …
"hole:calibration=long3"`). A single run, shortened, is how it is
validated locally:

```bash
julia --project=. --threads=4 test/hole_runs.jl calibration runs=e3-n8-r4-c t_end=1/2
```

The `tracked` section is not in the default list either:
the suite's tracked hole against step 5's sphere with the same layer, to
`5 M` by default or to `tracked=<t_end>`, four minutes at four threads:

```bash
julia --project=. --threads=4 test/hole_runs.jl tracked
julia --project=. --threads=4 test/hole_runs.jl tracked=1/2
```

The `generic` section is the measurement matrix and is
not in the default list: groups `ks0` (Kerr-Schild `a = 0` to `50 M`, seven
rows), `ks9` (Kerr-Schild `a = 9/10` to `20 M` on 1632 blocks), `harm`
(harmonic `a = 0`, and `a = 7/10` at `5/128`), `h7` (G5's chart at `5/256`,
a node on its own), `boost` (a boosted hole crossing a fixed fine region,
the rates, the coasting track) and `probe` (harmonic `a = 9/10`, host-side,
three minutes at four threads). A row alone is `generic=<label>`;
`t_end=<t>` shortens every row, `budget=<s>` stops them at a deadline and
records how far they got, and `tag=<name>` names the workers' directory.
With sixteen or more threads each row is a subprocess worker with one BLAS
thread (seven OpenBLAS pools spinning put a node at twice its cores). The
long groups ran on `amdq` (`ks0` about two hours, `ks9` about 2.3) and the
rest in `amddebugq` hours with `budget=3300`. They ran through a
scratch copy of `.claude/orchestration/symmetry-run.sh` with three changes
the orchestration's own script does not have: `PART` and `TLIM` for the
partition and the limit, `~` for a space in a job
(`'hole:generic=boost~budget=3300~tag=boost2'`), and **no precompile on the
login node** — its Cascade Lake image made two jobs starting together on
EPYC nodes race to rebuild the package and fail ("Unable to find compatible
target in cached code image"):

```bash
julia --project=. --threads=4 test/hole_runs.jl generic=ks0-fitted t_end=1/2
julia --project=. --threads=4 test/hole_runs.jl generic=probe
```

**Inconsistent:** the paragraph above says the orchestration's own
`.claude/orchestration/symmetry-run.sh` lacks `PART` and `TLIM`, `~` for a
space and the skipped precompile on the login node; `PLAN.md`, "Step 8 —
A hole that moves (G5)", "Symmetry mechanics", says that script now takes
them.

The `fitted` section is not in the default list either:
three rows, `fitted=fixture` (the suite's tracked hole to `1 M` under four
choices of initial data against `:damped`, two minutes at four threads),
`fitted=boosted` (a moving seed on 848 blocks, `:fitted` and `:damped`,
ninety seconds) and `fitted=harmonic` (harmonic Kerr at `a = 9/10` on 2472
blocks at `h = 5/256` — the initial data, one right-hand side, and the run,
which ends in its first chunk; three minutes and about 3 GB):

```bash
julia --project=. --threads=4 test/hole_runs.jl fitted=fixture,boosted,harmonic
```

The `moving` section is G5's runs and is not in the default
list: groups `g5` (G5's chart crossing a box of `5 M` from `x = 2` to
`x = −1.9` in `13 M`, adaptive, and the same hole at rest — two workers on
one `amdq` node, about seven hours), `g5u` (the crossing in a box of
`5/2 M` against a uniform mesh at `5/256`, 256³ points, about 90 GB and five
hours), `conv` (the frozen hierarchy, `N = 8, 12, 16`, on G5's chart and on
the boosted `a = 0` hole with the analytic sphere; the `N = 16` G5 row is
about four hours at 36 threads), `ctl` (the boosted `a = 0` hole across the
box: `:fitted`, `:damped` at `20/M`, `:frozen`, and the adaptive-against-
uniform pair at `5/128`) and the screens `g5v` … `g5z` (the moving layer's
parameters to `M/2`, one `amddebugq` hour each). `moving=mesh` runs only the
initial-data cycle of every adaptive row and prints its mesh (three minutes
at four threads); `moving=<label>` one row; `t_end`, `budget`, `tag` and
`threads` as in `generic`. Each worker prints one line per chunk: the masked
error, the error outside a fixed sphere enclosing the horizon (what the
convergence rows compare, since the masked region moves with the offset
surface), the shell's `C_a`, the layer's error off the truth by depth and by
side (trailing or leading), the finder's numbers and the horizon's extent
along the boost against the contraction. Stagger job starts, and do not
rsync the worktree while a job's workers are starting:

```bash
julia --project=. --threads=4 test/hole_runs.jl moving=mesh
julia --project=. --threads=4 test/hole_runs.jl moving=ctl-a0-fitted t_end=1/2
```

The `moving` section also has the groups `l0` (the trailing-side levers on the boosted
`a = 0` hole in the `5/2 M` box to `M/2`, ten rows of about four minutes at
four threads here — run them three at a time), `l7` (the same on G5's chart,
one `amddebugq` hour) and `g5t` (G5's crossing with `trail_ramp = 9/10`,
eleven hours alone on an `amdq` node). **Give every long Symmetry study a
remote directory of its own** (`symmetry-run.sh <worktree> <unique name>`):
the package image of a project path is one cache slot, and rsyncing changed
sources into a directory whose jobs are running, then precompiling, rewrites
the `.so` they have mapped (`SINGULARITY_HANDLING.md`, "The moving hole
(step 8)").

**A long row is a chain of jobs**: `checkpoint=<dir>`
makes every `generic` or `moving` row checkpoint each chunk to
`<dir>/<label>.it….h5` and continue from the newest file there, and
`walltime=<s>` stops it with a checkpoint before that many seconds of its
`evolve!` (leave the job's startup and compilation as margin below the
queue's limit). The same command resubmitted continues the row; a row that
is finished refuses to restart ("nothing left to run"). Give each study its
own `<dir>`, as its own remote directory:

```bash
julia --project=. --threads=4 test/hole_runs.jl moving=l0-trail-9 t_end=1/2 checkpoint=out/ck walltime=3000
```

The step benchmark is `bench/stepping.jl`, driven by
environment variables its header lists; on Symmetry it ran pinned
(`JULIA_EXCLUSIVE=1`, `srun --cpu-bind=none`, the fastest of the four
placements) in a scratch copy with `OrdinaryDiffEqLowOrderRK` and
`SciMLBase` added for its comparison rows, and on an H200 in a copy with
`CUDA` added — never in the package's own environment:

```bash
BENCH_MODE=step BENCH_CASE=wave,hole julia --project=. -t 4 bench/stepping.jl
```

The GPU right-hand-side prototypes are `bench/rhs_lab.jl`:
`key=value` options, one mode per round of `CODE.md`'s "The right-hand side on an
H200", listed in its header. Like `bench/stepping.jl` on a device, it runs from a
copy with `CUDA` added — on Symmetry `rhs-gpu-lab`, one H200 a job in `h200debugq`.
`bench/rhs_lab_source.jl` is the lean source it measures. `bench/rhs_lab_cpu.jl`
checks and times that source on the CPU in the package's own environment, and
`bench/sass_stats.jl` counts the instructions of a SASS dump it writes:

```bash
julia --project=. bench/rhs_lab.jl mode=round9 N=32 roots=8
```

```bash
julia --project=. bench/rhs_lab_cpu.jl
```

The CPU right-hand-side lab is `bench/rhs_cpu_lab.jl`, the
measurements of `CODE.md`'s "The right-hand side on a CPU": where `gh_rhs!` goes,
the kernel at each SIMD width (`simd_width`), checked against the scalar one, the
stencils without their zero weights (`LAB_ZW=1`), the native code, the head's pieces
and a profile of the ghost fill — modes and variables in its header; `LAB_CASE=hole`
for the hole fixture. It runs in the package's own environment, pinned on Symmetry as
`bench/stepping.jl` is; `BENCH_SIMD=1` gives `bench/stepping.jl` the scalar kernel:

```bash
LAB_MODE=breakdown,simd julia --project=. -t 4 bench/rhs_cpu_lab.jl
```

**On Symmetry** (step 9 writes the batch job for
real): the suite and the long studies run there as one SLURM job each on a
64-core EPYC node, which is what makes them parallel — a node *core* is
about **twice as slow** as the development machine's, so the cluster buys
throughput and not wall clock (`CODE.md`, "Measured results"). One quirk that
will bite a batch script: **do not export `JULIA_EXCLUSIVE=1` for the
one-thread suite.** A one-thread parent then pins itself to a single CPU,
and `test/threading_tests.jl`'s four-thread subprocess inherits the
affinity mask and aborts with `Too many threads requested for
JULIA_EXCLUSIVE option` — the environment, not the code. Two more: **the suite no longer fits `amddebugq`'s or `h200debugq`'s hour**
on these nodes (`CODE.md`, "The suite's tiers, and where each runs"), so submit it to `amdq` or `h200q`;
and **`mpi_tests.jl` inside a SLURM job needs `HYDRA_LAUNCHER=fork`**
(with `MPIR_CVAR_CH4_SHM_POSIX_RELEASE_GATHER_ENABLE=0`): MPICH's `mpiexec`
otherwise starts its ranks with `srun`, which waits for a step the suite
already holds ("Requested nodes are busy") until the workload's 3600 s
timeout. The `symmetry-hpc`
skill has the rest of the cluster's mechanics.

**SimWatch in a batch job**: `bench/simwatch.sh` is a verbatim
copy of SimWatch 1.3.0's `writers/simwatch.sh` (`simwatch_queued` after
`sbatch`, `simwatch_mark DIR failed "…"` on a non-zero exit; neither changed
since SimWatch 1.0.0). A new sbatch script sources the copy inside its own
rsynced directory, `. "$R/bench/simwatch.sh"` with `R` that directory — not
`spin-a06/out/simwatch.sh` or `ks-study2/out/simwatch.sh`, the older copies
the running studies' scripts name and source again at their exit, which stay
as they are.

The octant runs are `test/octant_runs.jl`, one run per
call, `key=value` options listed in its header — the case (`case=minkowski`,
`ks` or `harmonic`, `interior=damped|fitted`, margins, `fit_cont`;
`octant=reflecting|rotating` and the spin `a=`, which needs `rotating`), the mesh (`L`, `N`,
`roots`, `radii`), the run (`t_end`, `cfl`, `chunk`), noise, `backend=cuda`,
`out=<dir>` (CSV, `records.csv`, `simwatch.toml` — with
sparklines of the constraints, the error and the horizon, and thumbnails in
`out/plots/` drawn by `test/simwatch_plots.jl`, plain Julia:
SVG on a transparent background with `currentColor` axes, which SimWatch 1.3.0
shows as it is or flips for a dark terminal by its user's `image_theme`; `group=` names the SimWatch group) and
`checkpoint=<dir>`.
On Symmetry they ran one H200 each from a copy with `CUDA` added to its
`Project.toml` (`ks-octant`, `ks-study2`); `cfl = 1/2` is as accurate as
`1/4` there and half the cost:

```bash
julia --project=. --threads=4 test/octant_runs.jl case=ks interior=fitted L=8 N=16 roots=2 radii=4,2 t_end=1 chunk=1/2 cfl=1/2 amplitude=0 out=out/smoke
julia test/octant_study.jl out/study t_from=8 series=dA64,dA96,dA128:16,24,32
julia --project=. --threads=4 test/octant_runs.jl case=ks octant=rotating a=1/2 L=8 N=16 roots=1 radii=4,2,1 r_0=3/4 r_1=5/4 t_end=1/2 chunk=1/4 out=out/spin
```

`case=harmonic` is Kerr in harmonic coordinates at rest on an octant, with
no gauge source (step H1). Its defaults are the harmonic chart's radii, half
Kerr-Schild's: the layer `r_0 = 2/5`, `r_1 = 3/4` and the shells
`1, 5/4, 2, 4, 7`. Step H1's
rows ran on `L = 64`, cubes `32, 16, 8, 4`, one H200 each (`harm-h1`); the
line below is its `:fitted` row at `1/48`, two minutes a `M`:

```bash
julia --project=. --threads=4 test/octant_runs.jl case=harmonic L=8 N=32 roots=1 radii=4,4,2 t_end=1 chunk=1/4 cfl=1/2 amplitude=0 out=out/harm
julia --project=. test/octant_runs.jl backend=cuda case=harmonic L=64 roots=2 radii=32,16,8,4 N=96 interior=fitted fit_cont=2 lmax_fit=8 margin=16 n_L=20 t_end=24 chunk=1 cfl=1/2 amplitude=0 group=h1 out=out/hf48 checkpoint=out/hf48/ck checkpoint_every=4 walltime=40000
```

The harmonic hole moves too, on the quadrant (`octant=quadrant v= x0=
path=`), with no source (step H2), and **`mesh=follow`** gives a moving hole
the indicator's mesh regridded every chunk, with the floor on the tracked
horizon out to `floor_margin` (no `path`). Step H2's rows ran in `harm-h2`;
the first line is a local smoke run of the following mesh, the second the
`:fitted` row at `1/48` on two H200s of one node over MPI (from the HPC-X
environment `bench/symmetry_mpi_gpu_env.sh` builds, through `MPI.mpiexec()`
with `--bind-to none -n 2`; about `160 s/M`), and the third prints
`SINGULARITY_HANDLING.md`'s "A moving harmonic hole" from
`.claude/data/harm-h2` and step H1's `.claude/data/harm-h1`, with no package:

```bash
julia --project=. --threads=4 test/octant_runs.jl case=harmonic octant=quadrant v=1/5 x0=-1 L=8 N=24 roots=1 radii=4,3,2 margin=4 mesh=follow floor_margin=1 t_end=1/2 chunk=1/4 cfl=1/2 amplitude=0 out=out/hquad
mpiexec --bind-to none -n 2 julia --threads=8 --project=<hpcx-env> test/octant_runs.jl backend=cuda mpi=true deviceaware=false case=harmonic octant=quadrant L=64 roots=4 radii=32,16,8,4 N=48 interior=fitted fit_cont=2 lmax_fit=8 margin=16 n_L=20 v=1/5 x0=-4 mesh=follow floor_margin=3 t_end=40 chunk=1 cfl=1/2 amplitude=0 group=h2 out=out/qf48f checkpoint=out/qf48f/ck checkpoint_every=4 walltime=40000
julia test/harmonic_moving_study.jl
```

The spinning harmonic hole (step H3) is `case=harmonic a=` on the rotating
octant at rest, on the rotating quadrant along its spin and on the bitant
across it, cubes `32, 16, 8, 4, 2` about the hole. Step H3's rows ran in
`harm-h3`; the first line is the `:damped` row across the spin at `1/96`
(four H200s over MPI, about `530 s/M`), and the second prints
`SINGULARITY_HANDLING.md`'s "A spinning harmonic hole" from
`.claude/data/harm-h3`, with no package:

```bash
mpiexec --bind-to none -n 4 julia --threads=8 --project=<hpcx-env> test/octant_runs.jl backend=cuda mpi=true deviceaware=false case=harmonic L=64 radii=32,16,8,4,2 octant=bitant roots=2 x0=-2 path=4 a=1/2 N=96 r_0=3/5 r_1=3/4 margin=4 v=1/5 t_end=20 chunk=1 cfl=1/2 amplitude=0 group=h3 out=out/b5d96 checkpoint=out/b5d96/ck checkpoint_every=4 walltime=40000
julia test/harmonic_spin_study.jl
```

**On Nibi** (the `nibi-hpc` skill has the cluster): step H3's `v = 3/10` pair
ran on eight H100s of one node from an environment with Symmetry's HPC-X
environment's `Project.toml` and `Manifest.toml` and
`MPIPreferences.use_system_binary(; extra_paths = [<openmpi>/lib])` against the
system's Open MPI 5.0.10, with `OpenMPI_jll`'s library paths pointed there as
on Symmetry. Use the Alliance's Julia, not juliaup's, which cannot load that
library, and launch with `srun`:

```bash
module load StdEnv/2026 julia/1.13.0
srun --mpi=pmix --cpu-bind=none $EBROOTJULIA/bin/julia --project=<ompi-env> --threads=12 test/octant_runs.jl backend=cuda mpi=true deviceaware=false …
```

The same script runs a hole moving along `x` on the quadrant
(`octant=quadrant v= x0= path=`; `variant=` and `trail_ramp=` for a tracked
layer, `rho=` for the rate), and `test/boosted_study.jl` reads a directory of
such runs (`dir=`, default `.claude/data/boosted-ks`, one subdirectory per
batch) and prints the Markdown tables of `SINGULARITY_HANDLING.md`'s "The
boosted Kerr-Schild hole at binary speed" — moving against resting, the
orders, trailing against leading, the horizon, the layer residual. The first
line is a local smoke run; the second is the production `:damped` row at
`1/32` (on Symmetry, from a copy with `CUDA` added, one H200, about
`2.7 h`):

```bash
julia --project=. --threads=4 test/octant_runs.jl case=ks octant=quadrant v=1/5 x0=-1 L=8 N=16 roots=1 radii=4,3,2 r_0=5/8 r_1=23/16 t_end=1 chunk=1/2 cfl=1/2 amplitude=0 out=out/quad
julia --project=. test/octant_runs.jl backend=cuda case=ks octant=quadrant L=64 roots=4 radii=32,16,8,4 N=32 t_end=40 chunk=1/2 cfl=1/2 amplitude=0 r_0=5/8 r_1=23/16 v=1/5 x0=-4 path=8 out=out/bk32 checkpoint=out/bk32/ck walltime=40000
julia --project=. test/boosted_study.jl times=1,4,8,16,24,32,40
```

The production row is the default `4/M` (the moving layer's
target rate; `rho=8` is not needed), and `baseline=<dir>` compares each
moving row of `dir` with the same kind's row in another data directory — how
the rate was measured against the runs without it (`.claude/data/boosted-ks-rate`
holds the new batch `ks3/` and links to the resting controls in `rest/`):

```bash
julia --project=. test/boosted_study.jl dir=.claude/data/boosted-ks-rate baseline=.claude/data/boosted-ks rate=4,8
```

`octant=rotating_quadrant` (with `a=`) moves a spinning hole
along its spin, `z`, on the root brick `(r, r, 2r)` and a capsule along `z`;
`x0` and `path` are along `z` there. The production `:damped` row at `1/32`
(job 580105 in `boosted-ks-4`, about `2.8 h` on one H200), and `prefix=s`,
which reads the spinning family's rows (`sk<N>`, `sk<N>s`, `sf<N>`, `sf<N>s`;
`a=` sets the expected horizon, default `1/2`) and prints
`SINGULARITY_HANDLING.md`'s "A spinning hole along its axis" from
`.claude/data/boosted-ks/ks4` (the shell `r1.93185` is the equatorial horizon
radius `√(r₊² + a²)`):

```bash
julia --project=. test/octant_runs.jl backend=cuda case=ks octant=rotating_quadrant a=1/2 L=64 roots=4 radii=32,16,8,4 N=32 t_end=40 chunk=1/2 cfl=1/2 amplitude=0 r_0=3/4 r_1=23/16 v=1/5 x0=-4 path=8 out=out/sk32 checkpoint=out/sk32/ck walltime=40000
julia --project=. test/boosted_study.jl prefix=s shells=r1.93185,r3,r5 times=1,4,8,16,24,32,40
```

A production run takes **`chunk=1 split=0`**: the analysis
row every `M` and no trailing/leading split, which is `2.5×` cheaper than
`chunk=1/2` with the split (`CODE.md`, "The analysis cadence on an H200";
`finder=0` turns the finder off, which saves under a second a find). The
aligned campaign's rows are one such command each. The line below is its
`a = 9/10`, `v = 1/2`, `1/96` row (job 580900 in `boosted-ks-7`, `3.5 h` on
one H200). `test/aligned_study.jl` reads the campaign's data
(`dir=`, default `.claude/data/aligned`, one subdirectory per batch; a batch
named `*obsolete*` is skipped unless `batches=` names it; `runs=`, `times=`)
and prints `SINGULARITY_HANDLING.md`'s "Aligned spin and velocity": the runs,
moving against resting, the orders, the horizon, the layer residual, the cost
and the verdict matrix. It needs no package, so it runs without `--project`:

```bash
julia --project=. test/octant_runs.jl backend=cuda case=ks octant=rotating_quadrant a=9/10 v=1/2 x0=-2 path=10 L=64 roots=4 radii=32,16,8,4,3,2 N=24 t_end=20 chunk=1 split=0 cfl=1/2 amplitude=0 r_0=19/20 r_1=9/8 group=aligned out=out/a9v5h96 checkpoint=out/a9v5h96/ck checkpoint_every=4 walltime=40000
julia test/aligned_study.jl
```

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

- **Five dependencies are pinned to GitHub `main`, and TreeAMR comes
  from the registry**. `SpacetimeMetrics`, `ApparentHorizonFinder`,
  `KorzynskiSpin`, `IMEXRungeKutta` and `TreeIOHDF5` are
  what `Project.toml`'s `[sources]` entries resolve, and TreeAMR is
  General's release — so `~/src/jl/…` is *not*
  what the tests see; an unpushed change there is invisible here, a pushed
  change to TreeAMR's `main` is invisible too until it is *released*, and
  the local SpacetimeMetrics checkout has been behind `main` before. Read
  what Pkg installed under `~/.julia/packages/` when in doubt about an API.
  Say so rather than editing a checkout and assuming the tests see it. A
  local `Manifest.toml` resolved while TreeAMR was pinned needs
  `Pkg.update("TreeAMR")` (a plain `resolve` refuses the old version);
  `test/prerequisite_tests.jl` is what notices when a moving branch or a
  release drops a name.
  Do not vendor
  a package and do not add a local-path source.
  See `CODE.md`, "File layout".
- **Most `[sources]` pins are necessary, not chosen.** A
  pin goes only when its package is in General at the bound asked for:
  `KorzynskiSpin` is not registered at all, and
  `ApparentHorizonFinder` **2.1** is not released (General carries
  `2.0.0`, and `[compat]` here asks for `2.1`).
  See `CODE.md`, "File layout".

  **Inconsistent:** this rule names `KorzynskiSpin` and `ApparentHorizonFinder` as the necessary pins; `CODE.md`, "File layout", says three are necessary and that `TreeIOHDF5` is not in General at all, while the same paragraph and `HISTORY.md`'s "Releases and dependencies" (the status of 2026-10-09) call `TreeIOHDF5` registered (`[compat]` `0.1.0`).

- **TreeIOHDF5 `main` can be ahead of the TreeAMR it admits**.
  A copy of an older commit on
  TreeAMR 0.2.4 still needs the 0.1.3 pin.
  See `CODE.md`, "Checkpoint and restart".
- **`notes/` is read-only.** The copies carry their provenance; when
  `CODE.md` departs from them, `CODE.md` says so. Do not "fix" a copy.
  See `CODE.md`, "Lineage: what is inherited and what changes".
- **Two derivative index conventions.** `SpacetimeMetrics.dmetric`
  returns `dg[a, b, c] = ∂_c g_ab` (derivative axis *last*); GHSO2's
  pointwise algebra uses `dg[a, b, c] = ∂_a g_bc` (derivative axis
  *first*). Convert in one place, `initialdata.jl`, and test the
  conversion; a wrong convention produces a solution that looks right
  in Minkowski and wrong everywhere else.
  See `CODE.md`, "Initial data and backgrounds".
- **The packed component order is** `(tt, tx, ty, tz, xx, xy, xz, yy,
  yz, zz)`, column-major lower triangle, GHSO2's. `h` is variables
  `1:10`, `Π` is `11:20`. Nothing indexes a component by a literal
  outside `pointwise.jl`'s pack/unpack helpers.
  See `CODE.md`, "The equations".
- **Off by `G`.** Owned point `i` is stored at `i + G[d]`;
  `coordinates(fs, b, idx)` takes **stored** indices; a kernel launched
  by `map_blocks!` (default range) gets the owned index and adds `G`.
  Every stencil reaches `±(q/2 + 1)` and `G = q/2 + 1` is exactly that.
  Getting this wrong produces plots that look almost right. The
  right-hand-side kernel does the addition **once**, into a linear index,
  and the stencils step by a stride per axis — one place to get it wrong instead of
  1600.
  See `CODE.md`, "Field sets and layout"; `CODE.md`, "One right-hand-side evaluation".
- **`N = 8` does not exist at every order.** TreeAMR's vertex invariant
  is `N ≥ 2G + 2`, so `q = 4, 6, 8` need `N ≥ 8, 10, 12`.
  See `CODE.md`, "Field sets and layout".
- **A closure in a kernel must not capture a name assigned twice in that
  body**. Give each branch its own names.
  See `CODE.md`, "Precision, threads, devices".
- **KernelAbstractions refuses a `return` in a kernel** — anywhere in the
  body, closures included, which is what `ntuple(Val(10)) do v … end` is.
  End the block with the value instead. The error names the kernel and
  arrives at precompilation, so it is cheap; it is here because the
  package's own convention asks for an explicit `return` everywhere else.
  See `CODE.md`, "File layout" (`src/evolution.jl`).
- **The time integrator is IMEXRungeKutta's `RK4()`, by block owner, and
  every run goes through `gh_integrator`/`gh_solve`** (`src/stepping.jl`). `evolve!` passes `alias_u0 =
  true`: the integrator steps the run's own state vector in place, so an
  observer that keeps `u` keeps a vector that changes (none does). A regrid that moves the mesh
  sets `integ_prev = nothing`: its state vector is a new one, usually of
  another length, whose scratch IMEXRungeKutta would refuse by name (it
  checks the length, the array type and the partition). Its
  `RK4` clashes with OrdinaryDiffEq's by name: the package `import`s it
  `as IRK`.
  See `CODE.md`, "Time integration".
- **A script's verbs are `public`, not exported, and `rhs!` reads what
  `prepare!` left.** `prepare!`, `rhs!`, `stage_limit!`, `step_limit!`,
  `initial_state!` and `evolved_sets` share TreeGRRMHD's names: qualify
  them. Anything that scatters between `prepare!` and `rhs!` (`max_speed`, a
  monitor, `gh_dt`, `initial_state!`) makes `rhs!` evaluate another state,
  and nothing checks. A segment must be a contiguous view, since the CPU
  lanes store through a pointer. See `CODE.md`, "The verbs (GH-4)".
- **The RHS never mutates `u`, and the state has exactly three writers**. Both limiters
  are **integrator keywords**, passed by `gh_integrator` (`src/stepping.jl`)
  and nowhere else. **Under IMEXRungeKutta the two are
  one limiter on every state vector**: `gh_limiter!`, the
  projection and then the paste, is passed as both the stage limiter and the step limiter. Do not split them again, and do not add a fourth
  writer.
  See `CODE.md`, "The range projection"; `CODE.md`, "Time integration".
- **A limiter writes back only where it fired, and a run on which nothing
  fires is bit for bit the run without it.** `test/bounds_tests.jl`'s control asserts `isequal(u, u′)` for
  the fixture with and without bounds; it is the claim every experiment
  that compares the two rests on. A change that makes it a tolerance has
  broken the projection, whatever else it fixed.
  See `CODE.md`, "The range projection".
- **A `NaN` in the core is a hit, not the end of the run**. Read the validity rows (`min_α_shell`,
  `max_Π_shell`) and the shell error, not `bounds_hits`, for a failure at
  `r_1`.
  See `CODE.md`, "The range projection"; `CODE.md`, "Analysis quantities".
- **`ρ_max = 1/dt` is a grid rate, and a wrong target cannot survive it**: the
  calibrated layer is `4/M` with a ramp of at
  least 8 cells (`ρ_ramp = 1`, `r_0 = r_1 − n_L h`).
  See `CODE.md`, "The layer for an inexact target".
- **A layer target is evaluated wherever the layer is**, `r_0 ≤ r < r_1`,
  and at the core's sphere for `:pasted` — so its own singular set has to
  be inside `r_0` exactly as the background's does, and nothing checks it. The `residual` row is the
  layer's distance from the *background*, not from the target, and is
  large by construction for a wrong one.
  See `CODE.md`, "The layer for an inexact target".
- **A `HoleCenter` is sometimes the *tracked* trajectory**. Before comparing a radius with an analytic value, say which
  center it is about.
  See `CODE.md`, "The tracked geometry"; `CODE.md`, "Analysis quantities".
- **The fit is made in `(log α, β^i, γ_ij, Π_ab)`, never in `g_ab`**. The Lorentzian metrics are not convex in `g_ab`, and a least-squares fit is a weighted mean.
  `γ` is held as its offset `γ_ij − δ_ij` in slots 5–10, the contravariant
  shift in 2–4. Convert only through `fit_variables`/`state_from_fit`.
  See `CODE.md`, "The fitted target".
- **The fit's state sampler reads the layer, on purpose**: `state_sampler` passes `mask = AllPoints()`
  — the horizon finder's guard is exactly what it switches off. It is safe
  because `n_L ≥ 4G` holds `ρ ≤ 0.10 ρ_max` there; a thinner ramp
  would make it read a relaxed state. Ghosts must be filled first, as for
  every interpolation.
  See `CODE.md`, "The fitted target".
- **The shift has no constant term by default, and a moving hole needs
  one**. `build_fit(…; shift_constant = true)` fits it.
  See `CODE.md`, "The fitted target".

  **Inconsistent:** this rule says the shift has no constant term by default; `CODE.md`, "The fitted target" (piece 2), says `shift_constant = true` is the default and `false` the switch.

- **A fit to curvature (`cont = 2`) is not a metric on harmonic Kerr at
  `a = 9/10`** at `h = 5/256`, `m = 4`. The sweep
  catches it (`build_fit` throws with the numbers).
  See `CODE.md`, "The fitted target".
- **The fit's evaluator costs `2.2 µs` a point at `L = 8`**. That is why the
  right-hand side reads a **cache** and never calls it.
  See `CODE.md`, "The fitted target".
- **The cache is refilled when the center moves**. Change the
  geometry of a `:fitted` problem and you must refill (`refill_target`).
  See `CODE.md`, "The fitted target".
- **`:fitted` initial data are `C¹` at `r_1`, and it shows**. `fit_initial_depth = n_L h` (the switch at the core surface) removes
  it wherever the analytic solution is regular on the layer. Do not read a
  `:fitted` run's early error as the target's.
  See `CODE.md`, "The fitted target".
- **On harmonic Kerr at `a = 9/10` the fit is not good enough**.
  See `CODE.md`, "The fitted target" (piece 12); `SINGULARITY_HANDLING.md`, "The interior's questions".
- **A moving layer exports what it holds through its trailing side**. Read the `moving` rows' trailing-side
  layer error before suspecting the target.
  See `CODE.md`, "The fitted target" (piece 13).
- **Behind a moving layer, narrow the ramp** (`trail_ramp = 9/10`).
  Pass it for
  every moving `:fitted` run. The horizon's `J` of a moving spinning hole
  drifts with or without it — read it before trusting a long crossing.
  See `CODE.md`, "The fitted target".
- **`horizon_min_radius` of a boosted hole is a bound, not the radius**: the rest frame's smallest radius times `√(1 − v²)`,
  exact only when that radius lies along the boost. A moving seed takes
  its surface's least radius instead, or the first find of a boosted
  spinning hole is refused as a jump.
  See `CODE.md`, "The tracked geometry" (piece 5).

  **Inconsistent:** `CODE.md`, "The bitant: a hole moving across its spin" ("The least horizon radius across the spin"), and `HISTORY.md`'s "The bitant" (the status of 2026-10-24) say that `horizon_min_radius` of a boosted hole is now the shortest semi-axis of the contracted ellipsoid (`horizon_ellipsoid`), exact across the spin too, and the old bound only where it was exact.

- **A tracked floor starts at the core surface**.
  See `CODE.md`, "Refinement and regridding".
- **A tracked run's first find starts from the seed's shape**, not
  a sphere of the mean radius.
  See `CODE.md`, "The fitted target" (piece 10).
- **The depth replaces the radius**. `in_layer(int, t, x)`
  takes a position and a time, the
  kernel's predicates are asked of `interior_point(int, t, x)`, and a band
  about the layer is `layer_mask`/`shell_mask`, never a `ShellMask` built
  from `r_0` and `r_1` — a `FittedInterior` has neither field. A tracked case
  holds a `FittedSpec`, not a layer: `GHProblem`, `state_callback`,
  `level_bounds` and the `Π` post-pass refuse it and ask for `interior =
  fitted_interior(…)`, which `evolve!` builds every chunk.
  See `CODE.md`, "The tracked geometry".
- **The shape is a series at every layer point**. The
  real harmonics' slots are `ash_mode_index`'s, `m < 0` the sine, `ỹ^s =
  −√2 Im Y_lm`; the fit must use the same convention, and
  `real_from_complex` is the only conversion.
  See `CODE.md`, "The tracked geometry".
- **A tracked geometry's `h` is its layer's, and its margin is stated in
  it.** `fitted_interior` reads the coarsest spacing of the blocks the layer
  lives in.
  See `CODE.md`, "The tracked geometry".
- **`F` is never evaluated where `w = 0`.** The frozen core holds
  finite, stale data by design — the analytic solution is singular
  inside it — and `F` of that data may be `NaN`; `0 · NaN = NaN`. The
  kernel branches on the core predicate before it touches a stencil.
  See `CODE.md`, "The interior: a pointwise damping layer".
- **The interior is masked in every norm and in the indicator.**
  Constraint, error and speed kernels and the Löhner indicator write
  zero for `r < r_1`. If a number looks wrong near the hole, check the
  mask before the physics. **Write the masked slot through a branch, not
  as `keep * value`**: the masked region may hold a `NaN`, and
  `0 · NaN = NaN` — the same trap as the frozen core's, met in the
  monitors.
  See `CODE.md`, "The interior: a pointwise damping layer"; `CODE.md`, "Analysis quantities".
- **Every path that puts the analytic solution on the grid goes through
  the core rule**, `core_position` — the initial data, the error
  reference, the `:pasted` limiter, the Dirichlet hook *and the
  gauge-source sample*.
  See `CODE.md`, "The interior: a pointwise damping layer".
- **The mask plumbing.** A kernel takes a `mask`, asks
  `is_evolved(mask, x)`, writes zero where it says no and a `1`/`0`
  indicator into `DIAG_MASK`; `masked_norms` divides by the **evolved**
  volume, not the domain's, so masking a region out does not make the
  number smaller by diluting it with zeros. `AllPoints` is the trivial
  mask, `InteriorMask` is `r ≥ r_1` at that call's `t`, and `ShellMask`
  is a band — which is how the three interior variants are compared over
  "the `G` points outside `r_1`" and how the gauge drift is read at the
  horizon.
  See `CODE.md`, "Analysis quantities".
- **A reduction that crosses blocks is TreeAMR's `mesh_fold` or
  `mesh_mapreduce`, never a host fold of ours**. Several numbers, or one number over a region smaller
  than a block, are **one fold** with a field each, never several
  reductions or a monitor re-run with a mask to select points.
  See `CODE.md`, "Analysis quantities".
- **A checkpoint is written after the analysis row and before the regrid**. Keep the regrid one code path. Save
  `(U, u)`, never `U` alone. **Anything added to the loop's carried state goes
  into the run state too** (`driver.jl`'s `state = (; …)` and the restore
  branch), or restarts silently stop being the run;
  `test/checkpoint_tests.jl`'s chains are what notice.
  See `CODE.md`, "Checkpoint and restart".
- **The `diag` slots are a contiguous-range interface.** `DIAG_CGH` (four) and `DIAG_MOM` (three) are the *first* of
  a run and are indexed as `DIAG_CGH + a - 1`. Adding a slot in the middle
  of either run is how that breaks.
  See `CODE.md`, "Field sets and layout".
- **A vertex-centered field set stores the shared upper plane, and no
  kernel writes it.** So a test that reads `fs.work[:, :, :, v, :]` sees a plane of zeros
  that means nothing; use `interiorview(fs, b, v)`.
  See `CODE.md`, "Field sets and layout".
- **The ADM monitor costs 18 s to compile**, and about 4 s for every
  further specialisation of `(G, q, has gauge source, T)`. Before
  putting it in a new test row or a new loop, count the specialisations.
  See `CODE.md`, "Analysis quantities".
- **A coarse-fine face is expensive in time as well as in order.** A study on a refined mesh costs roughly four times what the
  same study costs uniform.
  See `CODE.md`, "The interface-order rule, and what it costs a second-order system".
- **A spherical frozen core cannot hide Kerr's singularity in the
  harmonic chart at `a = 0.9`**.
  See `CODE.md`, "The spinning harmonic chart"; `SINGULARITY_HANDLING.md`, "The interior's questions".
- **The three interior radii are asserted at every regrid**: `r_1`
  inside the horizon by `m` spacings of the blocks containing it
  (`m = 8` by default, never below `G + 1`); `r_1 − r_0` at least
  `2(G + 1)` such spacings; the refinement's level floor covering the
  horizon. If one fires, raise the floor, shrink the layer or widen the
  floor's shell — do not remove the assertion.
  See `CODE.md`, "The interior: a pointwise damping layer".

- **The refinement is an error indicator with three geometric
  corrections, and all four matter.** Ghosts must be filled before flagging.
  Convergence claims are made on a hierarchy frozen at `t = 0`, not on
  the adapting mesh (`CODE.md`, "Refinement and regridding").
- **The reference amplitude is one number for all ten components, and
  that is not an approximation** (`CODE.md`, "Refinement and regridding").
- **The travelling margin is dilated inside `refine_flags`, and
  `regrid!` is then given `buffer = 0`** (`CODE.md`, "Refinement and regridding"). Dilate, *then* clamp.
- **The floor and the ceiling together are a statement about the box.** See `CODE.md`, "Refinement and regridding".
- **2:1 balance overrides the ceiling, and that is TreeAMR's invariant.**
  If the boundary blocks are not at
  the coarsest level, count the levels before suspecting the ceiling (`CODE.md`, "Refinement and regridding").
- **`ρ_max` is `4/M` by default, read from the hole, and bounded by RK4's
  stability** (`CODE.md`, "The profiles and the rate"). A run that blows up in
  the layer after raising `ρ_max` has found the integrator, not the
  physics. **At `4/M` the layer is held to `τ/ρ_max`, not `τ · dt`**: the `:damped` residual saturates only after about `2/ρ_max = M/2`,
  so a claim about the residual's saturation needs a run that long — which
  is why `driver_tests.jl`'s variants read it at `1/2 M` and the shell at
  `1/10 M`.
- **The horizon finder reads only the evolved region, and it says so by
  throwing.** See `CODE.md`, "Analysis quantities". Do not swap in TreeAMR's `Ellipsoid`, which rounds
  differently from `is_evolved` and measured no cheaper. If a find fails,
  read `r_1 + (q+2)h/2` against the horizon's smallest coordinate radius
  before suspecting the finder.
- **Ghosts must be filled before anything is interpolated**, with that
  call's hook. `find_gh_horizon` scatters and fills; `gh_adm_provider` and
  `gh_interpolate` do not (`CODE.md`, "Analysis quantities").
- **The Korzyński spin is what a find costs**, not the interpolation (`CODE.md`, "Measured results", "What a find costs"). Lower the
  cadence or pass `spin = false` before lowering `N_ah`.
- **The analysis quantities are part of the deliverable**: masked
  constraint norms, the interior residual, horizon location, area,
  `M_irr`, `J`, `M_ch`, the refinement centroid, mesh statistics, at
  every chunk boundary, appended to the run's time series. A run that
  finishes without them is not a result, and a test that asserts only
  "it ran" is not a test (`CODE.md`, "Analysis quantities").
- **`Float64` on the H200 is the requirement; `Float32` is a
  nice-to-have.** Record a `Float32` failure, do not fix it at
  `Float64`'s expense. The type-generic discipline stays anyway: a
  decimal literal in a `T` expression is a leak — `T(1//2)`, not `0.5`;
  `oftype(x, 2)` inside closures (`CODE.md`, "Precision, threads, devices").
- **The RHS kernel is written in streaming order, and stays that way** (`CODE.md`, "One right-hand-side evaluation").

  Never build an `SVector` of all derivatives: that is about 140 `Float64`
  values, over a GPU thread's 255 registers, and it spills. Spills show as `STL`/`LDL`
  in the SASS (`bench/sass_stats.jl`).
- **On a device a closure is a call** (`CODE.md`, "The right-hand side on an H200").
  - **How to write kernel-path code instead:** `Base.Cartesian.@ntuple`/`@nexprs`
    with literal counts, small tuples spelled out, and a generated body returning
    `Expr(:block, Expr(:meta, :inline), …)`.
- **Over MPI the host analysis is rank 0's, and the other ranks serve it** (`CODE.md`, "Running distributed"). Inside `on_root(…; serve = (p, u))` — the find, the
  track's update, the fits — rank 0 announces each collective step by a
  broadcast request and the other ranks take it with no points of their own.
  The only two steps are `fill_state!` and `interpolate_state`. **A new
  collective inside the analysis must go through one of them, or a new
  request kind in `serve_requests`**: a `fill_ghosts!`, a `mesh_mapreduce`
  or an `interpolate` called directly there runs on rank 0 alone and hangs
  the job. And never assign a captured variable inside the analysis: what
  it returns is adopted on every rank after it, which is how the track, the
  geometry and the fits stay the same bits everywhere.
- **A block index is this rank's, a leaf index is the forest's** (TreeAMR's
  M7; `CODE.md`, "Running distributed"). `nblocks(fs)`, `blockkey(fs, b)`, `block_origins`, `block_mapreduce`
  and `firing_boxes` are per local block; `forest.leaves`, `nleaves` and
  `locate_point` are global. `forest.leaves[b]` with a local `b` is right on
  one rank and wrong on every other. A flag vector `buffered_flags` takes is the whole forest's.
- **Multi-GPU runs go through HPC-X with `--bind-to none`, not MPICH_jll** (`CODE.md`, "Running distributed"). `bench/symmetry_mpi_gpu_env.sh`
  builds the HPC-X environment (with `OpenMPI_jll` pointed at its libraries)
  and the `bench/symmetry_mpi_*.sh` jobs pass the flag.
- **A decision from a rank's own clock or file system must be agreed**
  (`agree_any`, `agree_refusal`), and a number combined on the host from
  per-block values must be reduced over the ranks (`rank_reduce`), or the
  ranks part ways at the next collective and the job hangs. What the
  checkpoint's `data` holds must be the same on every rank, or TreeIOHDF5
  refuses the save (`CODE.md`, "Running distributed").
- **Hooks depend on time.** `dirichlet(case, t)` is built at each call.
  It goes to `fill_ghosts!` inside the RHS, to `regrid!`, and to
  `adapt_to_initial_data!`, each with that call's `t`. See `CODE.md`, "Outer boundary: Dirichlet from the background, at the current time".
- **A moving non-harmonic background with a sampled source is refused.** See `CODE.md`, "Gauge and constraint damping". The refusal's message says this; do
  not weaken it.
- **`boost(m, v)` moves the hole at `−v`** (`CODE.md`, "The tracked geometry").
  `hole_velocity(background)` is the one place the sign is written. Do not pass `velocity = v` for a boost
  by `v` — leave it unset. **Never test a metric *on* its singular point** (`CODE.md`, "The tracked geometry").
- **A moving `:damped` core relaxes in ADM variables, never linearly in `g`** (`CODE.md`, "The layer for an inexact target", "The frozen core").
- **A tracked `:damped` core is still frozen** (`CODE.md`, "The layer for an inexact target"). Screen a moving depth for less than `r_core/v`, or with `:fitted`.
- **A relaxation toward a moving target needs the target's rate** (`CODE.md`, "Why touch the interior, and why relax"). Anything
  else that relaxes toward something that moves — a new variant, a new
  target — needs the same term in every region it relaxes in, the core
  included. The rate is a centered difference (`δ = ε^{1/3} r_0`), so a moving
  hole's quadrant and box agree to `ε^{2/3}`, not to roundoff:
  `boosted_tests.jl` says so, and so must any new mirror test of a moving
  hole.
- **Check `r_1` against the contracted horizon** (`CODE.md`, "Single black holes: recommended settings"). Choose the
  radii for the contracted horizon before submitting a run.
- **The quadrant needs a hole on the `x` axis, moving along it, with
  `a = 0`** (`CODE.md`, "The quadrant: a hole moving along `x`"). As with the octant,
  a refined quadrant is not the refined box's restriction; compare the two on
  uniform meshes only.
- **The rotating quadrant needs a hole on the `z` axis, moving along it and
  spinning about it** (`CODE.md`, "The rotating quadrant: a hole moving along its spin axis").
- **A moving hole's symmetric domain is the box only to the target rate's
  amplified roundoff** (`CODE.md`, "The rotating quadrant: a hole moving along its spin axis"). Bound any new symmetry test of a moving hole the same way, and do
  not read a growing difference there as a seam bug before checking the box's
  own asymmetry.
- **The ring rule limits `:fitted` to `h ≤ 1/32` at `a = 1/2`** (`CODE.md`, "Single black holes: recommended settings").
- **`hz_J` is a component on the rotating quadrant and a magnitude elsewhere** (`CODE.md`, "The rotating quadrant: a hole moving along its spin axis"). Say which one a comparison reads.
- **`1/48` on the moving hole's capsule does not fit one H200** (`CODE.md`, "Single black holes: recommended settings"), and in the harmonic chart a mesh that follows the hole does not either: it grows past
  one H200 at its regrids (`CODE.md`, "Time integration"). Run it on several GPUs (`mpi=true`).
- **On a GPU the analysis row, not the evolution, is the expensive part** (`CODE.md`, "The analysis cadence on an H200"). Use `chunk=1
  split=0` for production, and the split only to diagnose a moving layer's
  sides. Before adding a monitor to the
  row, price it in ADM-monitor evaluations.
- **At `a = 9/10` the layer depends on the speed** (`CODE.md`, "Single black holes: recommended settings"). A run that changes speed,
  or a resting control for a `v = 1/2` row, has no layer that is known to
  work; compare the moving `9/8` rows with the resting `5/4` ones outside
  the horizon, where the two resting layers agree to three digits by `20 M`.
- **A half turn cannot hold a hole whose spin and velocity are not parallel** (`CODE.md`, "The bitant: a hole moving across its spin"). The half turn is for a pair of holes, each the other's
  image.
- **`Val`s once per chunk.** `G`, `q`, "has gauge source", "has
  dissipation", the interior *variant* (`:none`,
  `:damped`, `:pasted`, `:frozen`, which is "has interior" and *which* in
  one parameter) and the SIMD width `W` are `Val`
  parameters built in `GHProblem`'s constructor.
  Building them per evaluation recompiles or dispatches dynamically on
  every RK stage. See `CODE.md`, "One right-hand-side evaluation". The
  seventh, `HASMATTER`, is the exception that proves it: `gh_rhs!` builds
  it per call from the matter adapter's *type* (`has_matter`), which is
  static. Never from a value. See `CODE.md`, "What this package takes (GH-3)".
- **On the CPU the kernel's algebra runs on SIMD lanes, so it must stay generic
  in its number type** (`CODE.md`, "The right-hand side on a CPU"). Everything `gh_rhs_head` and
  `gh_rhs_pi` reach — `metric_quantities`, the lean source, `metric_divergences`,
  the gauge sources, the damping and dissipation profiles' *results* — is evaluated
  with `T = Vec{W,T}`, which is not a `Number`. So: **no branch on a computed value**
  (`if x > 0`, `?:` — a lane condition has a value per lane; use `_select`, and
  `_anynonzero` for "is it zero"); **no scalar times a static array** (`a * M`;
  use `_scale`, which is also closure-free for a device); **no literal converted to
  `T`** (`SMatrix{4,4,T}(-1, 0, …)`; use `one(T)`, `zero(T)`, `T(x)`). A violation
  is a `MethodError` or a `TypeError` at the first CPU evaluation, which
  `test/simd_tests.jl` makes on every branch.
- **KernelAbstractions refuses a `return` statement anywhere in a kernel
  body**, closures included. See `CODE.md`, "File layout", `src/evolution.jl`.
- **A `@generated` method must not convert to the caller's type.** See `CODE.md`, "Measured results", "A `@generated` method may not convert to the caller's type". Emit the exact integers and let the call site divide.
- **The stencil weights are for unit spacing, and the dissipation's
  factor of `h` is a single one.** See `CODE.md`, "Finite-difference stencils" and "Kreiss–Oliger dissipation". A run that blows up faster the larger `ε_KO`
  is has that sign backwards, and `test/stencils_tests.jl` asserts it as
  an exact rational eigenvalue.
- **The recipe near a hole is `ε_KO ≈ 0.5`, `γ0 ≈ 1/M`.** GHSO2
  measured both as requirements with the horizon in the domain
  (`notes/methods-ghso2.md`); a run that blows up at the sonic surface
  with `ε_KO = 0` is not a bug in the stencils (`CODE.md`, "Lineage: what is inherited and what changes").
- **The interface-order rule.** This system takes second derivatives,
  so prolongation order `p = q + 2` (order 6 at `q = 4`), or the global
  rate drops by one. Never give `ops` a default that hides it; TreeAMR
  refuses `G` below `p/2 − 1` and this package's `G = q/2 + 1` clears it (`CODE.md`, "The interface-order rule, and what it costs a second-order system").
- **Don't name a keyword `maxlevel`.** It shadows TreeAMR's exported
  `maxlevel(forest)` inside the function body. Use `maxlevel_cap`.
- **A callback must capture no `Type` and no host array.** Backgrounds,
  the interior profiles, the damping profile, the indicator's thresholds
  and floors become kernel arguments; they are `isbits` structs and
  tuples. The center is a function of `t`, never a mutated field (`CODE.md`, "Precision, threads, devices").
- **Two spellings of one expression are not bit-identical — and neither
  are two call sites of one function.** See `CODE.md`, "Measured results". Compare any of
  these to roundoff against the scale that produced the number, never with
  `isequal`. The bit-identity that *is* an invariant is the same compiled
  code, at the same call site, at a different thread count, which is what
  `test/threading_tests.jl` will assert.
- **Never thread anything a TreeAMR callback can reach**, and never
  accumulate into shared state in a loop of your own: bit-identity
  across thread counts is the invariant, and `test/threading_tests.jl`
  is the only thing that will report a violation (`CODE.md`, "Precision, threads, devices").
- **A parallel loop of this package's own is TreeAMR's
  `threaded_foreach`/`threaded_chunks`, never `Threads.@threads` or
  `@spawn`, and a kernel is a `map_blocks!`** (`CODE.md`, "Precision, threads, devices"). The two
  helpers are unexported, so `prerequisite_tests.jl` names the one this
  package calls (`threadchunks`, for the integrator's partition), beside the `Region` extension points the footprint guard
  reaches (`inside`, `stencil_hits`, `stencil_position`).
- **`Base` is not generic even though the mesh is.** MultiFloats defines
  no `rem`, no conversion to `Integer`, no `Float64(::Float32x2)`. Use
  `wrap` / `ceilint` / `floorint` / `tofloat64` from `precision.jl` (`CODE.md`, "Precision, threads, devices").
- **Measured numbers go into `CODE.md`** when they support a current
  decision, beside the prediction they
  confirm or correct, so a regression shows up as a changed number and
  not as a test that merely still passes — those of the black hole's
  interior into `SINGULARITY_HANDLING.md`, and a recommendation that changes into
  `CODE.md`'s table.
  Numbers that later measurements replaced, and numbers of code that no
  longer exists, go into `HISTORY.md`.
- **A refined octant is not the refined full box** (`CODE.md`, "Reflecting faces: symmetry planes"). Compare an octant with its box on uniform meshes
  only (`test/reflection_tests.jl` does).
- **An octant's corner is in the core, and its stale data is not of definite
  parity** (`CODE.md`, "Reflecting faces: symmetry planes"). Never project or perturb the core; `add_noise!(; exclude)` skips it.
- **A Symmetry node failure requeues a job from scratch** (`CODE.md`, "Checkpoint and restart"): Slurm
  restarts it with no checkpoint unless the run has one, over its own log and
  appending to its CSV. Give every run longer than an hour `checkpoint=` and a
  `walltime=`. And inside a `while read` loop `ssh` eats the loop's stdin —
  use `ssh -n`. **A requeue can hit a run that has finished**. After an outage, read the CSV's last `t`
  before trusting a "failed" status. Correct the status by hand
  (`status = "finished"` and a `message` saying what happened), and keep the
  peak memory and wall time elsewhere, since the log is gone.

- **A test that evolves a hole is `@full_tier`** (`CODE.md`, "The suite's tiers, and where each runs"). GitHub CI
  runs the quick tier, so a new testset that runs `evolve!` or `gh_solve` on
  a hole for more than a few seconds gets `@full_tier @testset "…"`, with a
  literal name (a looped testset goes inside a literal-named one; see
  `tracking_tests.jl`). Keep the quick tier honest the other way too: what
  only CI can see — x86-64, bounds checks, the kernel on every branch, the
  thread invariance — stays out of it. A file that marks a testset includes
  `tiers.jl` if `runtests.jl` did not, as `mpi_tests.jl` does `mpi_jobs.jl`.

- **Run the full suite locally before opening a pull request**: a pull
  request runs the quick tier on the two Linux cells only, so that is the
  macOS run the pull request does not get (`CODE.md`, "The suite's tiers,
  and where each runs").
- **Before adding a test row anywhere, price it**: a new `q` or a new
  element type is a new kernel; a new background is a new dual pass; a
  resolution added to an interface sweep is `N⁴` of ghost filling. Before
  adding a test that differentiates a background, look at what
  `pointwise_backgrounds.jl` already computes in one pass (`CODE.md`,
  "What the suite costs, and where", and "Measured results", G3).

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
  incomplete, fix `CODE.md` to describe the current state,
  and record in `HISTORY.md` what changed, when and why — rather than
  diverging silently.
- `README.md`, `CLAUDE.md`, `CODE.md`, `PLAN.md` and `HISTORY.md` each
  open with a **table of contents** of links — the `##` and `###`
  headings for `CODE.md`, `PLAN.md` and `HISTORY.md`, the `##` headings
  only for `README.md` and `CLAUDE.md` — kept current. A heading added
  or renamed goes into it, linked by GitHub's anchor: lowercase, backticks and
  punctuation dropped (an en dash too), spaces to hyphens.

## Repository facts

- **The remote is `git@github.com:eschnett/TreeGeneralizedHarmonic.jl.git`**,
  and the rule from the siblings applies: work on a branch, and do not
  push, open a pull request, or merge to `main` without being asked. Each
  step lands on `main` only after review.
- `TODO.md`, when it appears, is Erik's personal to-do list. **Do not
  modify it.** It is gitignored.
- `CODE.md`, `HISTORY.md`, `SINGULARITY_HANDLING.md`, `PLAN.md`, `README.md`, `notes/`, `src/`,
  `test/`, `.github/` and this file are committed. `Manifest.toml` files,
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
  diagnostics). `~/src/jl/GeneralizedHarmonicSecondOrder2`
  is the *origin* of `notes/`; the text to cite is in `notes/`.

- **Julia 1.13 is the floor and the only version CI runs**; the 1.11
  cells are not owed back (`CODE.md`, "File layout").
- **There is no `Manifest.toml`** (deliberately, and permanently: it is
  what makes the clean-checkout check mean something), and no `bin/`.
- **Keep coverage off the Linux cells** until a 1.13 release is shown to
  carry the fix (`CODE.md`, "The suite's tiers, and where each runs").
