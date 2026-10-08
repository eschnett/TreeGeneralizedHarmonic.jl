# Singularity handling — the measured results

Moved here from `CODE.md` on 2026-10-08, verbatim. `CODE.md` keeps the
design of the interior — [The interior: a pointwise damping
layer](CODE.md#the-interior-a-pointwise-damping-layer), the variants, the
profiles, the range projection, the tracked geometry and the fitted target —
and the settings these runs recommend for a single hole, [Single black holes:
recommended settings](CODE.md#single-black-holes-recommended-settings-added-2026-10-08).
**(Amended 2026-10-08**, in the merge of `main` into the excision round: the
excision round's runs, steps X1–X7, are here too, and `CODE.md` keeps their
design, [Excision](CODE.md#excision-added-2026-10-05).**)**
This file keeps the runs that made them: what was measured, with which jobs
and scripts, and what each result decided. Prose that names a section
("under …", "above", "below") names one of `CODE.md`'s unless the section is
here; links have been pointed at the right file.

- [Step 5: the static hole's interior and the driver (G4a)](#step-5-the-static-holes-interior-and-the-driver-g4a)
- [Step 8a: what crosses the horizon from inside it](#step-8a-what-crosses-the-horizon-from-inside-it)
- [Steps 8b–8′: the generic interior and the moving hole](#steps-8b8-the-generic-interior-and-the-moving-hole)
- [Single holes on the octant, `a = 0` to `9/10` (2026-10-02 to 2026-10-07)](#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07)
- [Excision, steps X1–X7 (2026-10-05 to 2026-10-08)](#excision-steps-x1x7-2026-10-05-to-2026-10-08)
- [The interior's questions, opened in step 5 and closed through step 8′](#the-interiors-questions-opened-in-step-5-and-closed-through-step-8)

## Step 5: the static hole's interior and the driver (G4a)

From `CODE.md`'s "Measured results", milestone G4a.

**G4a (step 5), the interior and the driver.** The suite is
**2486 assertions in 11m51** at one thread and **8m33** at four,
on the development machine (Apple silicon, 12 CPU threads, Julia 1.13.0),
up from step 4's 2144 in 8m40. A clean archive with no `Manifest.toml`
resolves the two pins from GitHub `main`, instantiates and passes with the
same count. Where the growth went:

| file | 1 thread | what it pays for |
|---|---|---|
| `driver_tests.jl` | **2m08** | five static-hole runs: the record, the order sweep at three resolutions, the three variants, the drift, and the `Π` post-pass |
| `interior_tests.jl` | **15 s** | the profiles, the core rule, the masks and the radius refusals (1.5 s), and one right-hand-side evaluation with the layer on a mesh (13.5 s) |
| `type_tests.jl` | **+23 s** | the whole hole pipeline again at `Float32` |

`driver_tests.jl` is now the suite's most expensive file and the third
whose cost is *arithmetic* rather than compilation. That is a hole being a
hole: `CODE.md`'s two radius requirements set the spacing from the
horizon's smallest coordinate radius, and a mesh that satisfies them at
`q = 2` is 61 440 points before anything is evolved. Before adding a run,
price it — and prefer `test/hole_runs.jl`, which is not in the suite.

**What a hole costs, before anything else.** The two radius requirements
together are `r_h,min ≥ (m + 2G + 2)·h + r_0`, so the finest spacing is
set by the horizon's *smallest coordinate radius* and by nothing about
the exterior. At the default margin `m = 8` the coefficient
`m + 2G + 2` is **14** at `q = 2` and **16** at `q = 4`, and the finest
level has also to cover the sphere `r_1` whole. Taking `r_0` where
`|h| ≈ 10`, that is, at `q = 2`: `h ≲ M/8` for Kerr-Schild at `a = 0`,
`h ≲ M/23` for harmonic Kerr at `a = 0`, and `h ≲ M/36` for harmonic Kerr
at `a = 9/10`. That factor of four and a half between the first and the
last — a factor of ninety in points — is why the suite's hole is
Kerr-Schild — which is
also the one with a *sampled gauge source*, so the cheap case is the one
that puts `Hsrc` under a hole. The amplitude, sampled along a ray that misses
the equatorial plane, runs the other way: `|h|` reaches 10 at `r = 0.2 M`
in Kerr-Schild at `a = 0`, at `0.4 M` in the harmonic chart at `a = 0`,
and **nowhere** at `a = 9/10`, where along such a ray it stays between 10
and 12 from `2 M` in to `0.05 M`. That last number is a trap and is the
reason the paragraph says "along a ray": at `a = 9/10` the chart is not
singular at the origin, it is singular on the **equatorial disk of
coordinate radius `a`**, and a ray that misses the disk never sees it.
On a grid, which does not miss it, `r_0` has to be larger than `|a|` —
which is the finding under
[The interior](CODE.md#the-interior-a-pointwise-damping-layer) and what stops
the harmonic chart at `a = 9/10` altogether.

**The order of the scheme with the layer in place.** Kerr-Schild `a = 0`
in a Dirichlet box of half-width `5/2 M`, on the frozen hierarchy of
`hole_forest` — 120 leaves, 56 at level 2 around 64 at level 3, so the
sphere `r_1` lies wholly inside the finest level and there is a
coarse-fine face between it and the outer half of the box — with
`r_0 = 2/5`, `r_1 = 23/20`, the default margin `m = 8`, `ε_KO = 1/2`,
`γ0` the Gaussian profile and the `:damped` interior — **at the grid rate
`1/dt`, the default until 2026-09-23**, as is every table of this step
(`ρ_max_factor = 1` reproduces them; the suite's rows at the default `4/M`
follow each, **(measured in step 8c′)**). `N` is raised with
the block layout held fixed, so every spacing shrinks and nothing else
moves. The error is the **masked** one: over `r ≥ r_1`, where the
equations are the Einstein equations and nothing else.

| `q` | `t_end` | `N` | `h` | masked L2 | masked L∞ | `C_a` L2 | layer residual |
|---|---|---|---|---|---|---|---|
| 2 | `3/20 M` | 6 | 5/48 | 6.381e−3 | 7.870e−2 | 6.290e−3 | 1.380e−1 |
| 2 | | 8 | 5/64 | 3.362e−3 | 3.549e−2 | 3.527e−3 | 6.210e−2 |
| 2 | | 10 | 1/16 | 2.115e−3 | 1.942e−2 | 2.256e−3 | 3.288e−2 |
| | | **rate** | | **2.16** | **2.74** | **2.01** | **2.81** |
| 4 | `1/4 M` | 8 | 5/64 | 5.464e−4 | 1.098e−2 | 1.630e−4 | 3.483e−2 |
| 4 | | 10 | 1/16 | 1.996e−4 | 3.774e−3 | 6.387e−5 | 1.950e−2 |
| 4 | | 12 | 5/96 | 9.021e−5 | 1.605e−3 | 3.033e−5 | 1.162e−2 |
| | | **rate** | | **4.45** | **4.74** | **4.15** | **2.70** |

Both rows are order `q` in the error and in the constraint, which is the
claim; the `q = 2` row is the suite's (`test/driver_tests.jl`) and the
`q = 4` row is `test/hole_runs.jl`'s. Two things in the table are not the
headline and are worth naming. The **layer residual** converges at about
`2.7`–`2.8` at *both* orders, which is what it should do: it is not a
truncation error of the scheme but the balance `ρ · residual ≈ w · F`,
and `ρ_max = 1/dt ∝ 1/h` while `F`'s truncation error is `O(h^q)` — the
`q = 2` row is `q + 1` because of the `1/h` and the `q = 4` row is short
of `q + 1` because at `h = 5/96` the layer is resolving a metric whose
own gradients are the steepest thing on the mesh. And the **masked
against unmasked** error: at `q = 2`, `N = 8`, masking the interior takes
the L2 from `5.052e−3` to `3.362e−3` and the L∞ from `6.210e−2` to
`3.549e−2` — the worst point of the whole domain is inside the layer,
every time, which is the arithmetic reason the mask is not optional.

**The suite's `q = 2` row at the default `4/M` (measured in step 8c′)**,
the same fixture, runs and time:

| `q` | `t_end` | `N` | `h` | masked L2 | masked L∞ | `C_a` L2 | layer residual |
|---|---|---|---|---|---|---|---|
| 2 | `3/20 M` | 6 | 5/48 | 6.322e−3 | 7.770e−2 | 6.288e−3 | 1.433 |
| 2 | | 8 | 5/64 | 3.336e−3 | 3.501e−2 | 3.530e−3 | 7.965e−1 |
| 2 | | 10 | 1/16 | 2.103e−3 | 1.917e−2 | 2.257e−3 | 5.075e−1 |
| | | **rate** | | **2.16** | **2.74** | **2.01** | **2.03** |

The error is the grid rate's to 1.5 % at every `N`, and lower, the
constraint to `0.1 %`, and their rates agree to the second digit. The
residual is ten to fifteen times larger and one order slower, and both are
the balance above with a fixed `ρ`: `residual ≈ w F/ρ_max` is `O(h^q)` when
`ρ_max` does not grow as `1/h`.
Masking now takes the L2 from `3.536e−2` to `3.336e−3` and the L∞ from
`7.965e−1` — which is the residual — to `3.501e−2`.

**What the layer costs: `CODE.md`'s "a few percent of an RHS",
confirmed.** The cost is one forward-mode dual pass through the
background per *layer* point per evaluation, and it is measured on the
same mesh and the same state with and without the interior
(`INT = :none`, which evaluates `F` everywhere and `u_exact` nowhere):

| `q` | `N` | points | in the layer | ns/point without | with | share |
|---|---|---|---|---|---|---|
| 4 | 12 | 207 360 | 43 162 (20.8 %) | 1142 | 1193 | **4.5 %** |
| 2 | 8 | 61 440 | 12 714 (20.7 %) | — | — | **5.9 %** |

— 246 ns per layer point at `q = 4` on four threads, which is GHSO2's
"about a microsecond on a CPU" divided by the cores. The share is
slightly *larger* at `q = 2` than at `q = 4` and for the obvious reason:
the numerator is the same dual pass at both orders and the denominator is
a right-hand side whose cost per point grows with `q`. Both are a few
percent, and a fifth of the points paying for the analytic solution is
what both numbers are.

**The three interior variants.** `CODE.md` names three and predicts that
`:damped` and `:pasted` both hold the static hole with constraints at
truncation outside `r_1`, `:damped` with the smaller violation in the `G`
points outside `r_1`, and that `:frozen` piles compressed features up
against the freezing radius. On the fixture above (`q = 2`, `N = 8`,
`t = 1/10 M`), with the shell of `G = 2` spacings just outside `r_1` —
6104 points — read through a `ShellMask`, at the grid rate `1/dt`, the
default until 2026-09-23:

| variant | layer residual (L∞) | `C_a` L2 in the shell | `C_a` L∞ in the shell | masked error L2 |
|---|---|---|---|---|
| `:damped` | **6.204e−2** | 1.1054e−2 | 6.956e−2 | 2.2688e−3 |
| `:pasted` | **0** (by construction) | 1.1360e−2 | 7.083e−2 | 2.3681e−3 |
| `:frozen` | **6.753e−1** | 1.1062e−2 | 6.955e−2 | 2.2583e−3 |

**The prediction is right about the residual and nearly silent about the
constraints.** `:frozen`'s residual is **11 times** `:damped`'s after a
tenth of an `M` and grows linearly with time — `7.8e−1, 1.63, 2.56, 3.56`
at `t = 0.05, 0.1, 0.15, 0.2 M` on a coarser mesh, against `:damped`'s
`2.42e−1, 2.56e−1, 2.57e−1, 2.57e−1`, which *saturates* after the first
chunk. That
is exactly the difference between a sticky wall and a sink, and it is the
measurement the default rests on. The constraint norms in the `G` points
outside `r_1`, on the other hand, separate the three by **3 %**:
`:damped` is below `:pasted` as predicted, and `:frozen` is
indistinguishable from `:damped` there over this time. So the acceptance
criterion "their constraint norms in the `G` points outside `r_1`" is
measured and recorded, and it is *not* what chooses the default — the
residual is (amended in step 5). **`:damped` is confirmed as the
default.**

**The three variants at the default `4/M` (measured in step 8c′).**
`:pasted` and `:frozen` do not read the rate — the paste freezes the ball
`r < r_1`, and `:frozen` has `ρ ≡ 0` — so their rows above are unchanged to
every digit. `:damped` at `t = 1/10 M`, the same fixture and shell:

| variant | layer residual (L∞) | `C_a` L2 in the shell | `C_a` L∞ in the shell | masked error L2 |
|---|---|---|---|---|
| `:damped`, `4/M` | 5.513e−1 | 1.1062e−2 | 6.954e−2 | 2.2588e−3 |

The shell is where it was — `:damped` and `:frozen` now agree there to
`1e−4` relative and `:pasted` is 2.7 % above both — but **the residual claim
is not true at `1/10 M` any more**: `:frozen`'s residual is 1.22 times
`:damped`'s, not 11, because the sink relaxes in `M/4` rather than in one
step and has not yet saturated. Against time, every `1/20 M` of the same
run:

| `t/M` | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `:damped`, `4/M` | 0.289 | 0.551 | 0.796 | 1.009 | 1.181 | 1.306 | 1.385 | 1.424 | 1.433 | 1.422 | 1.457 |
| `:frozen` | 0.320 | 0.675 | 1.080 | 1.508 | 1.930 | 2.304 | 2.593 | 2.765 | 3.164 | 3.589 | 6.266 |
| `:damped`, `1/dt` | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.077 | 0.100 | 0.195 |

`:damped` saturates by `0.4 M`, `1.6/ρ_max`, and `:frozen` grows linearly:
the sticky wall against the sink is the same distinction it was, arriving
a relaxation time later. So **the suite makes the residual claim at
`t = 2/ρ_max = 1/2 M` (amended in step 8c′)**, where `:frozen`'s is 2.52
times `:damped`'s, and keeps the shell claim at `1/10 M`, where step 5 made
it: by `1/2 M` `:pasted`'s shell `C_a` L2 is `2.02e−2` against `:damped`'s
`1.24e−2` (`1.21e−2` at the grid rate) and `:frozen`'s `1.25e−2` — the
paste's kink growing toward its `17 M` failure at either rate, which is a
claim about the paste and not the one the testset makes. The price is the
two runs to `1/2 M`, 50 steps each instead of 10; the costs are under
[the default rate](#the-default-rate-step-8c).

**To `t = 50 M`, and the prediction that was wrong.** The same
configuration at `q = 2`, `N = 8` (`h = 5/64`, 120 leaves, `cfl = 1/5`,
`chunk = 1 M`, 5350 steps, 19 minutes at four threads), run to `t = 50 M`
for each variant, at the grid rate `1/dt`, the default until 2026-09-23:

| variant | reaches | masked L2 at the end | `C_a` L2 | layer residual |
|---|---|---|---|---|
| `:damped` | **`50 M`** | 1.604e−1 | 3.951e−2 | 1.387 |
| `:pasted` | `17 M`, then a degenerate metric | — | — | — |
| `:frozen` | `13 M`, then a degenerate metric | — | — | — |

`CODE.md` predicted that "`:damped` and `:pasted` both hold the static
hole to `t = 50 M` … `:frozen` holds the static hole only with
`ε_KO ≈ 0.5` and a wide ramp". **Half of that is wrong and the half that
matters is right (measured in step 5):** with `ε_KO = 1/2` and the
Gaussian `γ0`, *only* `:damped` reaches `t = 50 M`. `:pasted` fails at
`17 M` and `:frozen` at `13 M`, both by the same mechanism — `√(det γ)`
of a state that is no longer a metric — and both inside the layer
**(corrected in step 8b** for `:pasted` and for the `N = 6` `:damped` row
below: the failing square root is the lapse's `√(−g^{tt})`, reached by the
right-hand side at a stage vector, and the state degenerates in the
*evolved* shell just outside `r_1` — at `r = 1.18` and `1.27`, against
`r_1 = 1.15` — while the layer inside the projection's gate never moves;
see [the range projection's
measurements](#the-range-projection-step-8b)**)**. That
is the strongest evidence for the default there is: the hard paste's
truncation-order mismatch at a *surface* is not a small perturbation of
the smooth layer, it is a kink that the stencils straddling it feed back
into the interior until the metric degenerates. The same mesh one
resolution coarser (`N = 6`, `h = 5/48`) does not hold even `:damped`,
which fails at `21 M`: **26 points per `M` holds this hole at `q = 2` and
19 does not**, and that is a resolution statement about a second-order
scheme and not about the layer.

What "reaches `t = 50 M`" does *not* mean is that the answer is good. The
masked L2 error at the end is `1.6e−1` against `3.4e−3` at `t = 0.15 M`,
and the interior residual is `1.4` — this is `q = 2` at 26 points per `M`,
where fifty crossings of accumulated truncation error is a large number.
The claim the row supports is *stability*, and the order claims are the
table above.

**The default's `50 M` number is step 8c's**, and it is not re-run here
(step 8c′): the exact target at `4/M` on this fixture's own layer, the same
configuration with the range projection on, reaches `50 M` with the masked
error L2 at **`0.027`** against this table's `0.160`, the `G`-point shell's
`C_a` L2 at **`0.029`** against `0.181` (step 8c's control of this row; the
table's `C_a` column is the whole evolved region's), and the finder's
`M_irr` at **`0.9972`** against `0.9955` ([the layer against an inexact
target](#the-layer-against-an-inexact-target-step-8c)). The `:pasted` and
`:frozen` rows do not read the rate and stand; the `N = 6` `:damped` run that
ends at `21 M` has not been run at `4/M`.

**The other two charts.** The suite's hole is Kerr-Schild at `a = 0`
because it is the cheapest; `test/hole_runs.jl` runs the two the
resolution argument above says are expensive, at `q = 2` to `t = 1/5 M`:

| background | `r_h,min` | `r_sing` | `r_0` | `r_1` | leaves | `h` | masked L2 | `C_a` L2 | residual |
|---|---|---|---|---|---|---|---|---|---|
| `Harmonic(1, 0)` | 1.000 | 0 | 0.20 | 0.67 | 120 | 5/64 | 1.679e−1 | 1.736e−3 | 7.92e+1 |
| `KerrSchild(1, 9/10)` | 1.436 | 0.900 | 0.95 | 1.25 | 1128 | 5/128 | 2.438e−3 | 1.118e−3 | 2.31e−1 |
| `Harmonic(1, 9/10)` | 0.436 | 0.900 | — | — | — | — | **refused** | | |

Both runs stay finite and both constraint norms are at the same `1e−3` as
the suite's, which is the claim. The **harmonic hole at `a = 0` is badly
under-resolved at this spacing and says so**: its `r_0` has to sit at
`0.2 M`, where `|h| ≈ 29` rather than `CODE.md`'s `≲ 10`, because
`r_h,min = M` leaves no room for a larger one at `h = 5/64` — and the
layer's residual is `79`, three orders above the evolved region's error.
That is the guidance in "The interior" being right: `r_0` where the
solution is still moderate is not a nicety, and the harmonic chart at
`a = 0` needs `h ≈ M/23` before it has one. The spinning hole in
Kerr-Schild, at `h = 5/128` and 1128 leaves, has a residual of `0.23` and
is the well-resolved row of the three.

**The gauge drift.** GHSO2 measured a slow, constraint-preserving drift
of the excised hole under prescribed sources at `≈ 0.14/M`, and `CODE.md`
predicts that exact interior and boundary data lower it without removing
it. Measured as the L∞ of `|h_tt − h_tt,exact|` over the shell from the
horizon's smallest coordinate radius to its largest plus the layer's
width — `[2 M, 2.75 M]` for Kerr-Schild `a = 0` with this layer — against
time, with the initial data exact so the fit goes through the origin: at
`q = 2`, `N = 8`, over `t = 0 … 1/4 M`, the rate is **1.95e−3 / M**
(at the grid rate; the suite's run at the default `4/M` gives `1.9512407e−3`
against `1.9512409e−3`, the same to seven digits — the shell starts at the
horizon, `10.9` cells outside `r_1`, and the layer's rate does not reach it in
a fifth of an `M` **(measured in step 8c′)**).
Over the `t = 50 M` run (`:damped`, `q = 2`, `N = 8`) the drift reaches
`4.077e−3` at the end (at the grid rate; step 8c's runs at `4/M` end at
`2.7–3.3e−3`), a rate of about **8e−5 / M** averaged over fifty
crossings — two thousand times below GHSO2's `0.14/M` on the excised
hole. **`CODE.md`'s prediction is confirmed in its strong form
(measured in step 5):** exact interior *and* boundary data do not merely
lower the drift, they leave it at the truncation error's level, and the
number that is measured here is the truncation error and not a gauge
mode. The short-run rate is larger than the long-run one because the
drift saturates rather than growing, which is what a gauge mode would not
do.

**`Float32`.** The same static-hole run — `q = 2`, `N = 8`, `t = 1/10 M`,
the `:damped` layer, the sampled gauge source, the Gaussian `γ0` — runs
end to end at `Float32` on `CPU()` and reaches the `Float64` answer to
**five significant figures**: masked L2 `2.268762e−3` against
`2.268849e−3`, masked L∞ `2.659585e−2` against `2.659472e−2`, the layer
residual `6.204157e−2` against `6.204212e−2`, `λ_max` `1.6709520` against
`1.6709517`, and the same step count — at the grid rate. At the default
`4/M` **(measured in step 8c′)** it is the same agreement on different
numbers: masked L2 `2.258749e−3` against `2.258833e−3`, L∞ `2.637039e−2`
against `2.636953e−2`, the layer residual `5.512611e−1` against
`5.512648e−1`, the gauge constraint `3.481216e−3` against `3.481208e−3`,
`λ_max` and the step count unchanged. The gauge constraint — a difference
of large terms near a hole, which is where `Float32` has least to give —
agrees to `3.5e−3` against `3.5e−3`. This is the sharpest `Float32`
result in the package, and it is the offset identities of
[G1](CODE.md#milestones) earning their keep: `metric_quantities` takes
`g^{ab} − η^{ab}` from GHSO2's identity rather than from `inv(g) − η`,
whose `Float32` error is `4.2e−3` against the identity's `1.4e−7`.

**What the record is.** Every number the driver writes down is
`Float64` at every element type (`precision.jl`'s `tofloat64`), so a
`Float32` run's analysis time series is comparable with a `Float64` one
without a conversion at every call site; the *arithmetic* is in the type
the caller named, which the agreement above is the test of.

## Step 8a: what crosses the horizon from inside it

From `CODE.md`'s "Measured results"; the dispersion analysis it rests on
is under [Kreiss–Oliger dissipation](CODE.md#kreissoliger-dissipation).

**Step 8a, the expectations: what crosses the horizon from inside it.** No
`src` change. The suite is **3463 assertions in 13m01** at one thread and
**10m01** at four (15m08 in a first four-thread run while a sibling step's
suite shared the machine), the nineteen new ones being
`stencils_tests.jl`'s Nyquist-slope testset. The frozen-coefficient table
is under [Kreiss–Oliger dissipation](CODE.md#kreissoliger-dissipation) and the
two margin rules it leads to under [The
interior](CODE.md#the-interior-a-pointwise-damping-layer); this is the 3D
measurement they are held against (`test/hole_runs.jl leakage`, one
`amddebugq` node, sixteen groups as subprocesses, **45m47**; each of the
88 runs `2 M` long and 350–480 s at three or five threads).

*The experiment.* A radial ripple `A (1 − s²)³ cos(2π(r − r_c)/λ)`,
`s = (r − r_c)/2h`, `A = 1e−3`, in `h_tt` with `Π` untouched, centered `d`
cells inside the horizon (`r_c = r_h − d h`), on `hole_fixture` (`:damped`,
`r_1 = 23/20`, the layer at `ρ_max = 1/dt`, the default until 2026-09-23 —
a rerun of the section after step 8c′ runs it at `4/M`, while
`test/dispersion.jl`'s one-dimensional model below stays at the `1/dt` these
numbers were measured at) at `h = 5/64`; `A_k` is the
largest `|δh|` over the ten `h` components against the same run without
the ripple, in the shell `r_h + k h ≤ r < r_h + (k+1) h` (`ShellMask`'s
membership), over the chunk boundaries at every `1/20 M`. Four decisions
the step was silent on, all **(proposed in step 8a)** and argued in the
section's header: the mesh is the fixture's **finest level everywhere** — a
uniform level-3 forest of 512 blocks, since the 120-block hierarchy puts
the whole margin at `5/32` along the axes, where a `2h` ripple is a
constant; the ripple is in **`h_tt` alone**, which excites both branches
equally; `d` is the depth of the window's **center**, with a half-width of
**two cells**, which is what fits `d = 2` inside the horizon and `d = 8`
outside `r_1`; and the runs are **`2 M`** rather than `PLAN.md`'s `1 M`,
because the dispersion analysis puts the slow modes three to fourteen `M`
from depth `d` to the outer shells. The ripple enters through `evolve!`'s
observer at `t = 0`, which is handed the state vector the first chunk
integrates from; every perturbed run took the same 200 steps as its
reference and stayed finite, and a run whose ripple had not reached the
evolution would have been refused.

*At the default `ε_KO = 1/2`*, the largest `A_0/A` over `λ = 2h, 4h, 8h`
(the first shell outside the horizon) and `A_8/A`, against `PLAN.md`'s
prediction `e^{−(d+k)/ℓ_max}` with `ℓ_max` at the source's radius, the path
integral of `test/dispersion.jl` over the modes arriving by `2 M`, and its
one-dimensional model at `2 M` and `10 M`:

| `q` | `d` | 3D `A_0/A`, `2 M` | `e^{−d/ℓ_max}` | path, `2 M` | 1D, `2 M` | 1D, `10 M` | 3D `A_8/A` | `e^{−(d+8)/ℓ_max}` |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 0.144 | 0.773 | 0.805 | 0.166 | 0.410 | 8.9e−3 | 0.276 |
| 2 | 4 | 0.078 | 0.385 | 0.437 | 0.092 | 0.136 | 1.9e−3 | 0.057 |
| 2 | 8 | 9.9e−3 | 0.039 | 0.047 | 8.9e−3 | 0.058 | 9.6e−5 | 1.6e−3 |
| 4 | 2 | 0.166 | 0.657 | 0.715 | 0.117 | 0.348 | 8.4e−3 | 0.122 |
| 4 | 4 | 0.077 | 0.324 | 0.397 | 0.044 | 0.177 | 3.3e−3 | 0.034 |
| 4 | 8 | 8.3e−3 | 0.060 | 0.086 | 0.011 | 0.037 | 3.8e−4 | 3.6e−3 |

and the attenuation per cell — the least-squares slope of `log A_0` against
`d = 2, 4, 8`, over the three `λ` — against `1/ℓ_max(r_1)`, what a constant
`ℓ` at the layer's radius would give:

| `ε_KO` | 3D `q = 2`, `2 M` | 3D `q = 4`, `2 M` | 1D `q = 2`, `10 M` | 1D `q = 4`, `10 M` | `1/ℓ_max(r_1)`, `q = 2, 4` |
|---|---|---|---|---|---|
| 0 | 0.07–0.22 | 0.03–0.08 | −0.03–0.03 | 0.01–0.08 | 0, 0 |
| 1/4 | 0.30–0.40 | 0.28–0.38 | 0.07–0.21 | 0.17–0.31 | 0.24, 0.19 |
| 1/2 | 0.40–0.47 | 0.42–0.51 | 0.16–0.31 | 0.25–0.37 | 0.48, 0.37 |
| 1 | 0.50–0.53 | 0.51–0.55 | 0.33–0.39 | 0.36–0.43 | 0.96, 0.75 |

What agrees and what does not — the discrepancy `PLAN.md` asks to have
recorded:

- **The depth dependence is right at the default; the level is not.**
  `PLAN.md`'s `e^{−d/ℓ_max}` overstates the `2 M` transmission 4–7× at
  every depth — the ripple is broadband and splits between two branches,
  and only the part near the least-damped `θ` crosses — while its slope
  between `d = 2` and `8`, `0.50` e-folds per cell at `q = 2` and `0.40` at
  `q = 4`, is within 20 % of the measured. The path integral does no better
  on the level (the measurement is `0.18–0.21` of it at `q = 2`).
- **The measurement is not `∝ ε`, and the prediction is.** From
  `ε_KO = 1/4` to `1` the measured slope moves from about `0.35` to `0.52`
  e-folds per cell, against `0.24 → 0.96` predicted, and doubling `ε_KO`
  from `1/2` to `1` lowers `A_0` from depth 8 by only `1.7–2.5×`. The
  one-dimensional model saturates the same way, so this is a property of
  the principal part with coefficients that vary along the path, not of the
  source terms or of the other directions; the likeliest reading is the
  refraction the frozen model leaves out — a packet on a stationary
  background conserves `ω`, not `θ`, and slides toward the less-damped
  small `θ` as it nears the horizon — and it is a reading, not a
  measurement.
- **`2 M` is a lower bound.** At `q = 2` the transmission from depth 8 is
  still rising at `2 M` (`3.5e−3` at `1.25 M`, `6.4e−3` at `2 M` for
  `λ = 2h`); at `q = 4` it has levelled by `1.25 M`. The one-dimensional
  model follows the 3D numbers to within a factor 2.3 at `λ ≥ 4h` and
  `ε_KO > 0` — it undershoots them `1.2–3.8×` at `λ = 2h`, a ripple defined
  on the radius that off the axes is not a Nyquist mode of the lattice, and
  up to `6×` at `ε_KO = 0` — and it puts the long-time transmission from
  depth 8 at **4–6 %**: above `PLAN.md`'s prediction at `q = 2` (`0.058`
  against `0.039`) and below it at `q = 4` (`0.037` against `0.060`). As a
  long-time estimate the frozen-coefficient formula is within a factor two.
- **Outside the horizon it falls faster.** `A_8` is 3–11 % of
  `e^{−(d+8)/ℓ_max}`; the decay outside is `0.30–0.59` e-folds per cell at
  `ε_KO = 1/2` (`0.21–0.54` at `1/4`, `0.32–0.55` at `1`), since there
  every mode is outgoing and the grid-scale ones are damped at their own
  `σ`, and the Dirichlet face at `r_h + 6.4 h` along the axes reflects
  whatever reaches it.
- **Without dissipation nothing holds it back**, as `ℓ = ∞` says: at
  `ε_KO = 0`, `A_0/A = 0.13–0.88` at `q = 2` and `0.48–1.10` at `q = 4`,
  where it exceeds the source — the grid-scale growth step 3 measured on
  flat space (×9.2 in a thousand steps) acting on it.
- **The axis is where it leaks.** The cone within 18° of a grid axis holds
  the shell's maximum in 44 of the 72 rows and never less than `0.56` of it,
  and the diagonal cone is below the axis cone in 50 to 69 of the 72 rows
  at every `k`, as the diagonal numbers under [Kreiss–Oliger
  dissipation](CODE.md#kreissoliger-dissipation) predict.

## Steps 8b–8′: the generic interior and the moving hole

From `CODE.md`'s "Measured results", one subsection per step.

### The range projection (step 8b)

`src/bounds.jl`, `test/bounds_tests.jl` and `test/hole_runs.jl bounds`; the
design is under [The interior](CODE.md#the-interior-a-pointwise-damping-layer).

**The suite.** **3723 assertions in 13m00 at one thread and 9m33 at
four** on the development machine (Apple silicon, Julia 1.13.0), up from
step 7's 3444 in 12m31 and 8m38 — on a machine a sibling agent's runs
shared (load average 4–8), so the wall clock is an upper bound.
`test/bounds_tests.jl` is **279** of them — 278 in the file and the slot
map's one more in `constraints_tests.jl` — in **22.6 s / 10.1 s**: the
pointwise claims `2.4 s`, the planted failures and the gate `≈ 6 s`, and
the control, two `3/20 M` runs of the fixture, the rest. That control is
the one short run the step adds, paid twice because the claim is a
comparison.

**The map, on synthetic states.** The six states `PLAN.md` names — one
negative eigenvalue of `γ`; two negative eigenvalues with `det γ > 0` (which
a determinant floor would pass); `−g^{tt} < 0` with `γ` healthy (on which
`metric_quantities` throws a `DomainError`); a `NaN` in `γ_xy`; an `Inf` in
`Π_tx`; and Kerr-Schild at the fixture's core edge `r_0 = 2/5` — and three
more, one per remaining range (`|β| = 12`, a positive lapse of `10⁻³`,
`(α/√γ)Π = 500`). On each, at `Float64` and `Float32`, the projection
returns a state `metric_quantities` accepts with every ADM quantity in its
range; the blocks it moves are exactly the offending ones, **bit for bit**
(`===` on each of `h_tt`, `h_ti`, `γ_ij` and `Π`); a raised lapse keeps
`αΠ` to `1e−12`; a second application returns the same bits and does not
fire. It is the identity, bit for bit, on all six backgrounds of
`pointwise_backgrounds.jl` at their test points and on Kerr-Schild at
`r = 2/5, 1/2, 23/20` in three directions. On random states (20 000 per
row, `|h_ab|` and `|Π_ab|` uniform in `[−s_h, s_h]`, `[−s_Π, s_Π]`):

| `(s_h, s_Π)` | fired | re-fired before the re-test | after | `Float32`: outside a range |
|---|---|---|---|---|
| `(1/2, 1)` | 318 | 0 | 0 | 0 |
| `(2, 10)` | 18 202 | 0 | 0 | 0 |
| `(5, 100)` | 19 672 | **18** | 0 | 0 (but `metric_quantities` throws on 2 and is off by > 1 % in `α` on 14) |
| `(50, 1000)` | 19 936 | **402** | 0 | 3456 (`metric_quantities` throws on 60) |

The re-fires are the shift test on an ill-conditioned `γ`, and the
self-verifying re-test is what removed them; the `Float32` column is what a
24-bit mantissa can represent — `λ_min = 1/100` beside eigenvalues of `10³`
is a condition number `Float32` cannot resolve — recorded and not fixed
(`Float64` on the H200 is the requirement).

**The mesh, and the control.** On the fixture's mesh (`N = 8`, 120 leaves)
the kernel repairs exactly the points planted inside the gate — a `NaN` in
the core and a `γ_xx = −1/2` at `r < r_gate` — counts them (`hits = 2`,
`nonfinite = 1`, `r_max` the outer one's radius), leaves a `NaN` planted in
the outer layer (`r_gate ≤ r < r_1`) and one at an evolved point alone, and
changes no other bit of the state; `evolved_nonfinite` sees the evolved one
and not the layer's. **The control holds bit for bit**: `:damped` to
`3/20 M` with the projection on makes `4 · nsteps + 1` calls, fires on none,
and ends `isequal` to the run without it, with every record row
identical. **Across a regrid too (measured in step 8b, once, not in the
suite)**: `refinement_tests.jl`'s moving-mesh run (the adaptive fixture from
a hand-built 127-block mesh, one regrid to 120, `t = 1/10 M`, gate `0.625`)
with the projection on makes `26 = 4 · 6 steps + 1 + 1 regrid` calls — the
post-regrid application included — hands the run's `BoundsAccounting` to
the rebuilt problem, fires on none, and ends `isequal` to the run without
it. It is not in the suite because the suite's budget for this step is one
short run, and the control above is it.

**What it costs** (`hole_runs.jl bounds=cost`, the fixture's `N = 8` mesh,
four threads on the development machine; prediction `0.4 %` of a step):

| | time | per point | per gated point |
|---|---|---|---|
| right-hand side | 35.3 ms | 575 ns | |
| stage limiter, nothing fires | 0.16 ms | 2.6 ns | 31 ns |
| stage limiter, every gated point fires | 0.81 ms | | 158 ns |

`5137` of the `61 440` points (`8.4 %`) are inside the gate `0.8375`. With
four limiter calls against five right-hand sides per step (four stages and
the FSAL refresh a non-trivial step limiter asks for) the projection is
**0.36 % of a step (measured in step 8b; 0.41 % at two threads)** —
the prediction, confirmed. Where it fires everywhere it costs `5.1×` its
healthy call, still under 2 % of a step. It allocates nothing per point
(7 KB per launch, the launch itself).

**The two runs that end** (`hole_runs.jl bounds=damped6,pasted8`, on
Symmetry, one `amddebugq` node at 64 threads; Kerr-Schild `a = 0`, `q = 2`,
`cfl = 1/5`, `chunk = 1 M`, `ε_KO = 1/2`, the layer at the grid rate `1/dt` —
step 5's, which the section asks for by name from step 8c′ **(proposed in
step 8c′)**, since both rows are replays of step 5's table and its autopsy
rebuilds the fatal chunk at that rate). Each is run three times —
without the projection, at the proposed gate `r_1 − 2Gh`, and at the widest
gate the assertion allows, `r_1 − Gh`:

| row | `h` | gate (widest) | ends | projection hits | fatal square root | where the state degenerates |
|---|---|---|---|---|---|---|
| `N = 6` `:damped` | `0.1042` | `0.733` (`0.942`) | step 78 of 81 in the chunk to `22 M` (`t ≈ 21.96`), all three | **0** | the lapse's `√(1 − q_1)` of `−57.59` | `r = 1.27`, the `G` points outside `r_1 = 1.15` |
| `N = 8` `:pasted` | `0.0781` | `0.838` (`0.994`) | step 106 of 107 in the chunk to `18 M` (`t ≈ 17.99`), all three | **0** | the lapse's `√(1 − q_1)` of `−0.1976` | `r = 1.18`–`1.31`, outside `r_1` |

Both square roots are `metric_quantities`' `α = 1/√(−g^{tt})`, reached by
the right-hand-side kernel's evolved-or-layer branch on a stage vector —
for `:pasted` only points at `r ≥ r_1` take that branch at all.

**The prediction was wrong, and in the direction that matters (measured in
step 8b).** `PLAN.md` predicted hits "deep, several `M` before the crash".
There are none, at either gate: the state inside `r_gate` never leaves its
range — in the `N = 6` run the inner layer's error sits at `0.136` from the
first chunk to the last, `min α` there at `0.415`, `max |Π|` at `55.3` —
and all three runs of each row end in the same step with the **same**
`DomainError` argument to the last digit, which is the suite's bitwise
control at run length (`21 M` and `17 M`). What fails is the **evolved shell just outside `r_1`**: its error
against the analytic solution grows from the first chunk — `0.28` at
`1 M`, `1.6` at `10 M`, `3.6` at `19 M` and `22.5` at `21 M` for `N = 6`,
`0.39`, `1.4`, `3.5` and `10.0` at `1, 10, 15, 17 M` for `N = 8` — while the
record's shell rows (`min_α_shell` from `0.604` to `0.539`) move slowly, and
then the metric degenerates within `0.1 M`: in the last eight steps of
`N = 6` the shell's `min α` falls `0.28 → 0.20` at `r = 1.27`, `det γ`
`1.8 → 0.33`, `max |Π|` `1.7e3 → 8.3e3`; the outer layer (`r_gate ≤ r < r_1`)
follows it (`α` `0.35 → 0.27` at `r = 1.08`) rather than leading it.

So both of step 5's failures are **surface failures at `r_1`**, not
interior ones — the kink `PLAN.md`'s finding 1 describes (`ρ_max = 1/dt` is a
paste two cells deep, and a paste is a surface the stencils straddle), and
for `:pasted` literally so. No projection gated below `r_1` can reach them,
by construction — the gate exists so that the clamp's own kink is not
read by an evolved stencil — and none should: the failing points are
evolved by the Einstein equations. The instrument's value here is the
negative result and the validity rows, which show the approach from the
first chunk. **Nothing reached the horizon from the clamp**, because there
was no clamp: in all nine shells `[r_h + kh, r_h + (k+1)h]` the state
difference between the runs with and without the projection is exactly
`0` at every chunk, and the gauge constraint there grows the same in all
three runs (`L∞` in the innermost shell `1.6e−2 → 9.3e−2` over `21 M` at
`N = 6`, `1.0e−2 → 3.1e−2` over `17 M` at `N = 8`). For step 8c: the
thing to calibrate is the transition at `r_1`, and the numbers to watch are
the validity monitor's shell rows and the shell error, not the projection's
hit count.

### The layer against an inexact target (step 8c)

`src/interior.jl` (the `target`), `src/gauge.jl` (`HorizonDissipation`),
`src/driver.jl` (`ρ_max_fixed`), `test/evolution_cases.jl`
(`CurvatureTarget`) and `test/hole_runs.jl calibration`; the rule it
decides is under [The interior](CODE.md#the-interior-a-pointwise-damping-layer),
"The layer for an inexact target".

**The suite.** **3798 assertions in 13m25 at one thread and 10m12 at
four** on the development machine (Apple silicon, Julia 1.13.0), up from
step 8b's 3723. The 56 new ones cost **11.8 s / 7.2 s**: the profile's
algebra and its refusals and the target's refusals (`0.5 s`), one
right-hand side with the target equal to the background — `isequal` to
the one without it — and one with `KerrSchild(6/5, 0)`, which moves `du` at
every layer point and at no other (`1.7 s`), one with a constant profile
equal to the number — `isequal` again — and one rising to `4`, which moves
`du` exactly at `r_0 ≤ r < r_h` (`2.3 s`), and a `1/20 M` run whose record's
`ρ_max` is the fixed rate, with the two refusals (`7.3 s`). The right-hand
side of a case that uses none of the three is bit for bit the one before
this step (checked once against the tree at `70bd96c`, not in the suite).

**How the study ran.** Every configuration is `hole_fixture` — Kerr-Schild
`a = 0`, `q = 2`, the 120-leaf hierarchy, `h = 5/64` at `N = 8`, `r_1 =
23/20`, `m = 8`, `cfl = 1/5`, `ε_KO = 1/2` unless named — with the range
projection on (`default_bounds`, `default_gate`) and the finder's `M_irr`
every other chunk (`N_ah = 12`, no spin) from the observer rather than
from a case's `Horizon`, which the fixture does not carry — the observer
writes the record's rows too, so that a run that throws keeps its approach
**(proposed in step 8c)**. Screens are `5 M` at
`chunk = 1/2 M`, the survivors `50 M` at `1 M` as step 5's table was.
Thirteen `amddebugq` jobs, **5.6 node-hours**: four screen jobs (11–24
minutes, 138 runs; a fixture screen is 149–357 s at four threads, sixteen
to a node), eight `50 M` jobs (29–36 minutes, 62 runs; one that finishes is
1403–1973 s at eight threads, eight to a node) and the E0 profile pairs (17
minutes; an E0 run on the 512-block mesh is 390–440 s at sixteen threads).
Of the 120 fixture screens, 99 reached `5 M`; which 56 of them went on to
`50 M` is **(proposed in step 8c)** and is listed in the script
(`CAL_LONG`): the whole constant-`ε` scan, the profile on the two thickest
ramps at `4/M` and `1/M` and at the grid rate at `ε_in = 4`, E1 and E2 at
`n_L = 6` (`4/M`), `8` and `12`, every `δ = 4h` survivor, the four
controls, and — added after them — the exact target on the scan's layers.

**The order in the shell, `3/20 M`** (the `G = 2` points outside `r_1`,
`gh_outside_shell_norms`; the fixture's own layer, `r_0 = 2/5`,
`ρ_ramp = 1/2`, and `N = 6, 8, 10`):

| target | `ρ_max` | shell `C_a` L2, `N = 6, 8, 10` | rate L2 | rate L∞ | masked error rate |
|---|---|---|---|---|---|
| exact | `1/dt` | 1.970e−2, 1.118e−2, 6.271e−3 | **2.23** | 1.89 | 2.17 |
| exact | `4/M` | 1.958e−2, 1.120e−2, 6.298e−3 | **2.21** | 1.87 | 2.16 |
| E3 | `1/dt` | 4.238e−2, 3.790e−2, 3.027e−2 | **0.65** | 0.46 | 1.51 |
| E3 | `4/M` | 2.278e−2, 1.235e−2, 6.724e−3 | **2.38** | 2.00 | 2.07 |

Finding 1, measured: pinned at the grid rate, a target whose curvature is
wrong makes a right-hand-side error at the innermost evolved points that
does not converge — `0.65` where the scheme is `2` — and at a physical rate
the same target costs the shell nothing it can measure.

**The scan: the `G`-point shell's `C_a` L2 at `50 M`** (in parentheses the
`5 M` screen of a configuration not run long; `†` the time a run ended —
by `metric_quantities`' `DomainError` on a stage vector wherever the root
cause was read (every `50 M` run, and the screens rerun locally), and in
one screen, E2 at `δ = h` on 6 cells at the grid rate, by the CFL recheck;
rows are the ramp width `n_L` in cells, `nd` the fixture's own layer —
step 5's, a ramp of `4.8` cells in a layer of `9.6`; `ρ(Gh)/ρ_max`, the
relaxation at the innermost evolved stencil's reach, is `0.35` for `nd`,
`0.50, 0.21, 0.10, 0.036` for `n_L = 4, 6, 8, 12`):

| target | `n_L` | `1/dt` (`107/M`) | `10/M` | `4/M` | `1/M` |
|---|---|---|---|---|---|
| exact | `nd` | 0.181 (step 5's layer) | | **0.029** | |
| | 8 | 0.091 | | **0.029** | |
| | 12 | 0.037 | 0.029 | **0.029** | |
| E3 | `nd` | † 7 M | | | |
| | 4 | † 6 M | 0.152 | 0.119 | 0.075 |
| | 6 | † 7 M | 0.144 | 0.111 | 0.080 |
| | 8 | † 15 M | 0.060 | **0.033** | 0.049 |
| | 12 | 0.143 | 0.032 | **0.029** | 0.031 |
| E1 | `nd` | † 2.5 M | | | |
| | 4 | † 2.0 M | (0.598) | (0.275) | (0.264) |
| | 6 | † 3.0 M | (0.258) | 0.115 | (0.291) |
| | 8 | † 4.0 M | 0.106 | **0.038** | 0.087 |
| | 12 | † 18 M | 0.032 | **0.029** | † 15 M |
| E2, `δ = h` | `nd` | † 3.0 M | | | |
| | 4 | † 3.0 M | (0.193) | (0.085) | (0.076) |
| | 6 | † 3.5 M | (0.077) | 0.039 | (0.082) |
| | 8 | † 5 M | 0.036 | **0.029** | 0.044 |
| | 12 | 0.141 | 0.029 | **0.029** | † 6 M |
| E2, `δ = 4h` | `nd` | † 1.0 M | | | |
| | 4 | † 1.0 M | † 2.0 M | † 4.5 M | † 2.5 M |
| | 6 | † 1.0 M | † 6 M | 0.160 | † 2.0 M |
| | 8 | † 1.5 M | 0.123 | **0.041** | † 2.0 M |
| | 12 | † 2.5 M | ‡ | ‡ | ‡ |

`‡`: the target's singular point `(4h, 0, 0)` is a grid point of the
layer (`r_0 = 0.21`), and the run throws at `t = 0` — a configuration, not
a result. The masked error L2 at `50 M` follows the shell: `0.027` on every
bold entry but E1 at `n_L = 8` (`0.032`) and E2 at `4h` (`0.037`); `0.160`
for step 5's layer, `0.057` and `0.029` for the
exact target at the grid rate on `n_L = 8` and `12`. Every run that lives
is **flat from `10 M` on** — the shell's `C_a` and the masked error change in
the third digit over the last forty `M` — so the survivors do not approach
anything; the failures announce themselves, at the grid rate, by a shell
error that grows from the first chunk (E3 at `n_L = 8`: its L∞ `1.4` at
`3 M`, `2.7` at `9 M`, `4.4` at `12 M`, `20` at `15 M`, as step 8b saw step
5's), and at `1/M` on `n_L = 12` not at all in the shell (`min α` there
`0.607` to the end): for E2 the layer's `|Π|` climbs from `1022` to `1233` in
the last three `M`, the projection fires at `t = 5.85 M`, `r = 0.55` (122
hits), and the run ends inside.

**Finding 1's prediction** `n_L ≳ G (10 ρ_max M)^{1/3}` — `6.8`, `9.3`, `4.3`
and `20` cells at `4/M`, `10/M`, `1/M` and the grid rate — is **confirmed at
`ρ_max M = 4` and `10`** (the thinnest ramp holding every target is `8` and
`12`; `6` at `4/M` and `8` at `10/M` cost 1.3–4×), confirmed at the grid rate
(no inexact target holds near the rule's error at `12` cells, where it asks
for `20`), and **wrong at `1/M`**,
where it allows `4.3` and the scan has the 4–6-cell ramps at 2.5× the best
and the 12-cell one failing from inside — which is the lower bound
`ρ_max ≳ 4/M` of the rule.

**The `ε_KO(r)` profile** (`ε_out = 1/2` at and outside `r_h = 2 M`, rising
`C²` to `ε_in` at `r_1` and held inside), the same shell at `50 M`:

| E3, `n_L`, `ρ_max` | constant `1/2` | `ε_in = 1` | `2` | `4` |
|---|---|---|---|---|
| 12, `4/M` | **0.029** | 0.032 | 0.045 | 0.065 |
| 12, `1/M` | 0.031 | 0.032 | 0.036 | 0.047 |
| 8, `4/M` | **0.033** | 0.047 | 0.083 | 0.165 |
| 8, `1/M` | 0.049 | 0.035 | 0.057 | 0.110 |
| 12, `1/dt` | 0.143 | (0.146) | (0.214) | 0.238 |
| 8, `1/dt` | † 15 M | (0.240) | (0.272) | † 22 M |

and at `5 M` over the whole scan (48 screens) the profile is worse than the
constant in 35 — on every ramp that holds, at every `ε_in`. The thirteen
exceptions are transitions that are wrong already: eight at the grid rate
(`n_L = 4, 6` at every `ε_in`, taking a shell error of `0.43–0.60` to
`0.27–0.42`, and `8`, `12` at `ε_in = 1`, by 2–4 %), and five on the
marginal ramps (`n_L = 6` at `ε_in = 1` at every physical rate and at
`ε_in = 4` at `1/M`; `8` at `1/M`, `ε_in = 1`: `0.046 → 0.031`).
Step 8a's one-dimensional model said the profile cuts what crosses the margin
4–15×; what it measurably does in 3D to a smooth transition is add the
Kreiss–Oliger term's own `O(ε h^{q+1} ∂^{q+2} u)` where the solution is at its
steepest, and nothing a smooth target sends needs cutting. The masked
error says the same (worse in 37 of the 48 screens), and no survival
changes where the ramp is right; the drift of `h_tt` at the horizon falls at
`ε_in = 4` (`3.0e−3 → 2.2e−3` at `n_L = 12`, `4/M`), which is the
dissipation damping the solution and not a better one.

**E0: the hard step** (`:pasted` onto `KerrSchild(6/5, 0)`, 8a's uniform
512-block mesh at `h = 5/64`, each run against `:pasted` onto the exact
target with the same `ε_KO`; `A_0/A` is the largest `|δh_ab|` in the first
shell outside the horizon, `[r_h, r_h + h)`, over the step's `|δh| = 0.348`
at `r_1`, `10.9` cells below it):

| `ε_KO` | reached | `A_0/A` at `1 M` | `2 M` | `5 M` | e-folds/cell outside | masked error L∞ at the end | min `α` in the shell |
|---|---|---|---|---|---|---|---|
| 1/4 | † 1.0 M | 4.2e−4 | | | 1.70 | 1.1e3 | 0.354 |
| 1/2 | † 1.75 M | 2.1e−4 | 2.5e−3 (`1.5 M`) | | 0.92 | 1.9e3 | 0.287 |
| 1 | `5 M` | 3.0e−4 | 4.2e−3 | **2.2e−2** | 0.39 | 244 | 0.369 |
| 1/2 → 2 | `5 M` | 3.5e−4 | 2.6e−3 | 1.6e−2 | 0.37 | 8.1 | 0.576 |
| 1/2 → 4 | `5 M` | 5.1e−4 | 3.5e−3 | 9.5e−3 | 0.35 | 6.3 | 0.576 |

The three references reach `5 M` with a masked error L2 of `0.009–0.017`; the
e-folds of the two that end are of their last peaks, before the slow modes
arrive. Against step 8a: its ripple at depth 8, `ε_KO = 1/2`, reached the
first shell at `9.9e−3` by `2 M`, and its measured `0.40–0.47` e-folds per
cell of depth continued to `10.9` give `2.6–3.1e−3` — E0 at `ε_KO = 1/2` has
`2.5e−3` at `1.5 M`, the step behaving as a broadband source of the same
content; the frozen-coefficient `e^{−d/ℓ_max(r_1)}` is `5e−3`, `3e−5` at
`ε_KO = 1/2, 1`. The transmitted fraction is still rising at `5 M` —
the source never stops — and has reached the level 8a's long-time
one-dimensional rates give for `10.9` cells at `ε_KO = 1`, `1.4–2.7e−2`.

**The controls, with the projection on**, against step 5's table:

| run | step 5 | step 8c | at the end: masked error L2, shell `C_a` L2, drift, `M_irr` |
|---|---|---|---|
| `:damped`, `N = 8` | `50 M`, `1.604e−1` | **`50 M`** | 1.60e−1, 0.181, 4.08e−3, 0.9955 |
| `:damped`, `N = 6` | `21 M` | `21 M` | in the same chunk, with step 8b's `DomainError` to nine digits (`−57.5936359…`): 8b's ran on an AVX-512 node and this on an AVX2 one |
| `:pasted`, `N = 8` | `17 M` | `17 M` | step 8b's `DomainError`, `−0.19763566226561102`, to the last digit (both AVX2 nodes) |
| `:frozen`, `N = 8` | `13 M` | `13 M` | |

Zero projection hits in all four, and in every run of the study but two
families: E2 at `δ = 4h` on the layers its singular point comes within a
cell of (`1652` hits from `t = 0.046 M` at `r = 0.406` on the fixture's own
layer, `26124` from `t = 0.005 M` at `r = 0.398` on `n_L = 12`, both at the
grid rate and both ending in `1–2.5 M`), and E2 at `δ = h`, `n_L = 12`,
`1/M` above. The projection is not what any surviving run leans on.

**The horizon.** The finder's `M_irr` is `1.00029` at `t = 0` (its
truncation at `N_ah = 12`, `h = 5/64`) and `0.9972 ± 0.0002` at `50 M` on
every run at `4/M` or `10/M` with `n_L ≥ 8`, exact target or wrong — the
horizon's area does not see the target either — against `0.9955` for step
5's layer; the gauge drift of `h_tt` at the horizon is `2.7–3.3e−3` against
step 5's `4.08e−3` (which this study reproduces to three digits).

**The recommendation: proceed to steps 8d–8f, not to 8g (proposed in step
8c).** Forty-two of the 52 runs with a wrong target that went to `50 M` got
there: 39 of the 42 at a physical rate (the three that did not are `1/M` on
12 cells and `δ = 4h` at `10/M` on 6), 3 of the 10 at the grid rate, all on
12 cells and at five times the error. On the rule's layers (`ρ_max = 4/M`, `n_L ≥ 8`) a 20 % mass error, a
curvature error of `4/M²` and a displacement of `h` each leave the exterior
exactly where the exact target leaves it — the shell's `C_a` `0.029–0.039`,
the masked error `0.027–0.032`, `M_irr` `0.9972`, flat for forty `M`, and no
projection hit — which is what step 8e's fitted target needs: it will be
wrong by less than any of these. The layer does not need the target to be a
solution or to be close; it needs it smooth (E0's step gets out at a
percent and ends the run at `ε_KO ≤ 1/2`), a metric (finding 2; E3's other
sign is not one), regular on the layer (E2 at `4h`, `n_L = 12`), and centered
to about a cell (E2 at `4h` holds near the rule's error on one layer
only). Those are the three
things for 8d and 8e to guarantee — the tracked center to `≲ h`, the fit
valid at every swept point and smooth in time as well as in space, as 8e's
linear interpolation between two fits already is — and nothing measured
here says excision is needed. What this does *not* show: the rule at `q = 4`,
at the finer `h` the spinning holes need (the proposed scaling of rule 3),
or on a surface that is not a sphere; step 8f's matrix is where those are
measured.

### The default rate (step 8c′)

`src/interior.jl` (`hole_mass`), `src/driver.jl` (`default_relaxation_rate`
and the rule in `evolve!`), `test/driver_tests.jl`, `test/interior_tests.jl`
and `test/hole_runs.jl`; the decision — `ρ_max = 4/M` for every run,
**decided 2026-09-23** — is under [The
interior](CODE.md#the-interior-a-pointwise-damping-layer), "The profiles and their
parameters". No long run: the default's `50 M` number is step 8c's exact
target at `4/M` on the fixture's own layer (masked error L2 `0.027`, shell
`C_a` L2 `0.029`, `M_irr` `0.9972`), cited beside step 5's table.

**The suite.** **3857 assertions in 14m33 at one thread and 10m30 at four**
on the development machine (Apple silicon, Julia 1.13.0), shared with
sibling agents (load average 5–9), against **3798 in 13m45** at one thread
for the tree before this step, measured the same afternoon (step 8c
recorded 13m25 and 10m12). The 59 new assertions are `interior_tests.jl`'s
`hole_mass` through every wrapper and the default's value in the case's type
(46, `0.4 s`), `driver_tests.jl`'s testset for the grid-rate option, the
default's refusal and a Minkowski case through `evolve!` with no rate
(11, `6.2 s` / `3.0 s`), and one more row each in the record's claim, which
now covers `t = 0`, and in the variants'. What the time went on is the
variants: **`26.4 s` → `82.3 s` at one thread** (`30.6 s` at four), the two
runs to `1/2 M` the saturated residual needs — so `driver_tests.jl` is
`3m00` / `1m08` against `1m56` before.

**The claims that encoded the grid rate, amended and not loosened** (a
decided change of the spec): the record's `r.ρ_max * r.dt ≈ 1` is now
`r.ρ_max == 4/M` on every row, `t = 0` included, with `ρ_max_factor = 1`
giving `ρ_max · dt ≈ 1` in a testset of its own; the variants' residual
claim is made at `2/ρ_max = 1/2 M` and their shell claim at `1/10 M`, for
the reason under step 5's table. No other assertion changed. **`ρ_max_factor
= 1` reproduces step 5's `t = 1/10 M` row to every digit this document
records** (measured once, not in the suite): `:damped`'s residual
`6.2042e−2`, shell `C_a` L2 `1.1054e−2` and L∞ `6.956e−2`, masked error
`2.2688e−3`; `:pasted` and `:frozen` do not read the rate and are
bit-identical. The calibration's `:grid` rows, now asked for by name,
reproduce step 8c's sweep (`1.118e−2` and `3.790e−2` in the shell at
`N = 8`), and the `bounds` section's `damped6` replay step 8b's layer and
shell errors (`0.136`, `0.28` at `1 M`).

**Every suite number that moved**, before (the grid rate) and after (the
default), from the two suites' logs:

| claim (file) | grid rate `1/dt` | default `4/M` |
|---|---|---|
| order sweep, `3/20 M`, rates L2 / L∞ / `C_a` / residual (`driver`) | 2.165 / 2.740 / 2.008 / 2.806 | 2.157 / 2.741 / 2.006 / 2.032 |
| layer residual at `N = 6, 8, 10` (`driver`) | 1.380e−1, 6.210e−2, 3.288e−2 | 1.433, 7.965e−1, 5.075e−1 |
| unmasked L2 / L∞ at `N = 8` (`driver`) | 5.052e−3 / 6.210e−2 | 3.536e−2 / 7.965e−1 |
| `:damped` at `1/10 M`: residual, shell `C_a` L2, masked L2 (`driver`) | 6.204e−2, 1.1054e−2, 2.2688e−3 | 5.513e−1, 1.1062e−2, 2.2588e−3 |
| `:frozen`/`:damped` residual (`driver`) | 10.9 at `1/10 M` | 1.22 at `1/10 M`, **2.52 at `1/2 M`** |
| gauge drift rate over `1/5 M` (`driver`) | 1.9512409e−3 | 1.9512407e−3 |
| `Float32` / `Float64`, masked L2 (`type`) | 2.268762e−3 / 2.268849e−3 | 2.258749e−3 / 2.258833e−3 |
| `Float32` / `Float64`, residual (`type`) | 6.204157e−2 / 6.204212e−2 | 5.512611e−1 / 5.512648e−1 |
| adaptive static run at `3/20 M`: masked L2, `C_a` L2, residual, `τ_max` (`refinement`) | 1.929e−3, 1.719e−3, 0.799, 0.5295 | 1.881e−3, 1.711e−3, 6.001, 0.5301 |
| the regrid that moves the mesh: masked L2, `τ_max` (`refinement`) | 1.360e−3, 0.5312 | 1.341e−3, 0.5314 |
| bounds control at `3/20 M`: `max_Π_shell`, `min_α_layer` (`bounds`) | 2.6345, 0.410764 | 2.6339, 0.410764 |

The `Π` post-pass is unchanged — its test sets `ρ_max = 10` on the problem
itself, not through the driver — and the horizon claims pass as they did
(they are asserted to `1e−3` against Kerr and not logged). Outside the
layer the masked errors moved by at most 2.5 %, every one of them down, and
the constraints by under 0.5 % either way; the residual, the layer's
distance from the truth, grew seven to fifteen times at these times and
twenty-three times once saturated, as `τ/ρ_max` against `τ · dt` says it
must.

**The guard `ρ_max · dt ≤ 1` at the default** (the refusal of a fixed rate
above the grid rate, now the default's too): `4/M · dt`, over the chunks
and the `t = 0` row, is `0.040–0.047` on the fixture at `N = 8`,
`0.036–0.037` at `N = 10` and `0.057–0.062` at `N = 6`, and `0.067–0.092`
on the refinement's fixture at `h = 5/32` — eleven to twenty-eight times
inside it, on every hole the suite evolves.

**What the adaptive fixture's layer does at `4/M` (measured in step 8c′,
once, not in the suite).** It is six cells of `h = 5/32` from `r_1 = 5/4`
down to `r_0 = 3/10`, where `|Π|` reaches `131`, and at the grid rate
(`60/M` there) its residual saturates at `0.80` by `1/10 M`; at `4/M` it
rises through `6.0` at `3/20 M` and saturates at **`14.7`** by `1 M`, while
the layer stays a metric — `min α` there `0.3676` and `min det γ` `2.61` to
`2 M` at both rates — and the evolved region is better at `4/M` than at the
grid rate throughout (masked L2 `6.03e−3` against `7.13e−3`, `C_a` L2
`1.99e−3` against `2.55e−3`, at `2 M`). Step 8c's rule asks for a ramp of
at least `4G = 8` cells at `4/M`, and this layer is under it; the default
holds it anyway over the suite's times. A case whose layer is both coarse
and deep is where the default's larger residual is largest, and step 8d's
tracked geometry is the next place it is read.

**`hole_runs.jl` at the new default.** `order`, `long`, `charts`,
`indicator`, `horizon` and `leakage` run at `4/M` without edits; their
recorded tables are at the grid rate and are annotated as such where they
stand, and none has been re-run. `calibration`'s `:grid` rows pass
`ρ_max_factor = 1`, as `PLAN.md` asks, and so do the `bounds` section's two
replays of step 5's failing rows, whose autopsy rebuilds the fatal chunk at
the grid rate by construction **(proposed in step 8c′)**. `leakage` is the
one section whose rerun no longer reproduces its table — step 8a's layer
was at `1/dt` — and `test/dispersion.jl`'s one-dimensional model of it stays
at `1/dt` too, as the model of the runs it was compared against.

### The tracked geometry (step 8d)

`src/tracking.jl`, the second half of `src/interior.jl`, the tracked chunk
in `src/driver.jl`, and `test/tracking_tests.jl`; the design is under [The
interior](CODE.md#the-interior-a-pointwise-damping-layer), "The tracked geometry".
No long run is required by the step, and none is in `hole_runs.jl`.

**The suite.** **4410 assertions in 15m28 at one thread and 11m21 at
four** on the development machine, shared with sibling agents (load
average 6–7), against step 8c′'s 3857 in 14m33 and 10m30. The 553 new ones
are `tracking_tests.jl`'s, **`54.7 s` at one thread and `32.4 s` at
four**: the host-side part `2.3 s` (464 assertions, 415 of them the seed on
the charts' quartic), one right-hand side per variant and one find `12 s`,
and the two runs — the lost track with the trigger, and the tracked hole
against the sphere — `40 s` / `18 s`, most of it the compilation of the
tracked geometry's kernels (`1.3 s` each once the fixture's are compiled).
No other testset's claims changed; the protocol change touched only
`interior_tests.jl`'s one call of `in_layer`.

**The acceptance list, with its numbers** (all measured in step 8d on the
development machine, Julia 1.13):

| claim | number |
|---|---|
| real harmonics against `sYlm` at 200 random directions, largest error of the function | `3.7` / `6.4` eps at `lmax = 4` / `8` (`Float64`), `1.6` / `5.3` eps (`Float32`) |
| the same against `ash_evaluate` on `EquiangularGrid(6)` | below `16` eps |
| truncation of the analytic seed, Kerr-Schild `a = 9/10`, largest radius error | `1.2e−3`, `8.1e−6`, `5.5e−8`, `3.8e−10` at `lmax = 4, 8, 12, 16` |
| the same, harmonic Kerr `a = 9/10` | `4.6e−2`, `7.2e−3`, `1.1e−3`, `1.7e−4` (`2.4`, `0.37`, `0.057`, `0.009` cells at `h = 5/256`) |
| a fitted sphere against step 5's `Interior`, one right-hand side and the paste | `isequal` for `:damped`, `:frozen`, `:pasted` |
| one find of the fixture's initial data (`N_ah = 12`): the tracked center | `1.4e−5 M` from the analytic one, **`track_offset = 1.8e−4` cells** |
| the same: `r_min`, `r_max` about the found origin, against `r₊ = 2` | `2.00043`, `2.00068` |
| a find with `r_min` moved by `G h` | refused (`ArgumentError`); by `G h/4`, accepted |
| a finder disabled after `1/10 M` (chunk `1/20`, `max_misses = 2`) | `:coasting` at `3/20 M`, `TrackLostError` at `1/5 M` carrying the five rows, "0.1 M before t = 0.2" |
| the lapse-collapse trigger at `α_trigger = 1` with `every = 100` | every find after the first forced, `track_trigger = true` on those rows |
| a `:damped` tracked run of the fixture to `0.15 M` against step 5's sphere with the same layer (`r_1 = 2 − 10 h`, `n_L = 8`, `ρ_ramp = 1`): masked error L2 | `3.110274e−3` against `3.110271e−3` |
| the same: the `G`-point shell's `C_a` L2, the layer residual | `1.0745462e−2` against `1.0745448e−2`; `0.203` against `0.206` |
| the same: projection hits, `track_offset`, prediction error, re-samples | `0` and `0`; at most `3.3e−4` cells; `1.2–1.8e−4` cells per find; none |
| `margin_efolds` on the sphere `r_h = 2`, `m = 8`, `h = 5/64`, `ε_KO = 1/2` | `1.810` (`q = 2`), `2.073` (`q = 4`) — `test/dispersion.jl`'s `1.81`, `2.07` |
| the same on the tracked fixture, `m = 10` through the blocks the path crosses | `1.45` (a uniform `5/64` would give `2.68`) |
| the footprint guard on an `a = 9/10` shape, 2000 random footprints | agrees with the definition on every one |
| cost per point: `shape_series` at `lmax = 4`, `8`; one analytic `u_exact` | `21 ns`, `79 ns`; `96 ns` |
| cost of the tracked geometry: one right-hand side on the fixture, a `0.15 M` run at four threads | `+1.1 %`; `3.16 s` against `3.00 s` (`+5 %`) |
| host side per chunk: `fitted_interior`, `margin_efolds` | `0.04 ms`, `0.6 ms` |

**The same comparison to `5 M`** (`julia --project=. --threads=4
test/hole_runs.jl tracked`, the section added in step 8d and not in the
default list; chunks of `M/4`, the finder every chunk; measured once, not in
the suite). The tracked layer follows a found horizon that shrinks with the
solution's own drift — `r_in` from `2.00042` to `1.99564` by `5 M`, as
`M_irr` falls from `1.00029` to `0.99760` — and stays step 5's layer to a
percent:

| `t/M` | masked L2, tracked | sphere | shell `C_a` L2, tracked | sphere | `M_irr`, tracked | sphere | `track_offset` (cells) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 1.0796e−2 | 1.0796e−2 | 1.0002880 | 1.0002880 | 1.8e−4 |
| 1 | 1.4002e−2 | 1.3991e−2 | 1.2401e−2 | 1.2400e−2 | 0.9991516 | 0.9991516 | 7.4e−4 |
| 2 | 1.4177e−2 | 1.4147e−2 | 1.4550e−2 | 1.4508e−2 | 0.9983334 | 0.9983334 | 2.9e−4 |
| 3 | 1.6146e−2 | 1.6090e−2 | 1.8469e−2 | 1.8408e−2 | 0.9980183 | 0.9980182 | 7.4e−4 |
| 4 | 1.8755e−2 | 1.8595e−2 | 2.1214e−2 | 2.1107e−2 | 0.9977899 | 0.9977897 | 1.8e−3 |
| 5 | 2.0436e−2 | 2.0233e−2 | 2.2995e−2 | 2.2864e−2 | 0.9976053 | 0.9976048 | 3.0e−3 |

The difference grows to `1.0 %` in the masked error and `0.6 %` in the
shell by `5 M`, in the tracked run's disfavour, and `M_irr` agrees to
`4e−7`; neither run's projection fires, nothing is re-sampled (the core
surface moves by `5e−3 M`, a sixteenth of a cell, in `5 M`), and the
margin is `1.47` e-folds at the end. **`track_offset` grows, `3.0e−3` cells
at `5 M`** — the velocity estimate is two finds' difference, and the
finder's `1e−4`-cell noise over a chunk of `M/4` is a velocity of `4e−4`
cells per `M` that the prediction integrates — while the per-find
prediction error stays below `6.8e−4` cells. A static hole's track would be
better held without a velocity at all; a moving one's is G5's question,
and a velocity smoothed over several finds is the first remedy to try
**(proposed in step 8d)**.

### The fitted target, host half (step 8e-i)

`src/fit.jl`, `hole_velocity` in `src/interior.jl`, and `test/fit_tests.jl`;
the design is under [The interior](CODE.md#the-interior-a-pointwise-damping-layer),
"The fitted target". No kernel evaluates the fit yet (8e-ii), and no long
run is required; the explorations behind the harmonic and boosted rows are
host-side and a few seconds each.

**The suite.** **4578 assertions in 15m09 at one thread and 10m52 at
four** on the development machine (load average 6–8, shared with sibling
agents), against step 8d's 4410 in 15m28 and 11m21. (A first pair on the
tree before the documentation commit measured 14m59 at one thread and
37m25 at four — the latter under a load of 13, user time 18m29, its excess
in `constraints_tests.jl`'s and `interface_tests.jl`'s compilation-heavy
testsets, none of which this step touches.) The 168 new assertions are
`fit_tests.jl`'s 146 — **`12.6 s` at one thread and `12.9 s` at four**, the
tracked run it fits shared with `tracking_tests.jl` (`tracked_fixture_run`),
whose `The tracked hole` testset is `40.2 s` / `18.0 s` with the run in it —
and `interior_tests.jl`'s 22 for the boost sign (`0.4 s`).

**The acceptance list, with its numbers** (all measured in step 8e on the
development machine, Julia 1.13, `Float64` unless said):

| claim | number |
|---|---|
| the fit's solid harmonics at unit vectors against `shape_series`, and `S(2ξ) = 2^l S(ξ)` | `1.8`, `14` eps at `L = 4, 8` (`Float64`); `1.3`, `16` eps (`Float32`) |
| a real field from random coefficients through `ash_evaluate`, recovered by least squares on `EquiangularGrid(L)` | cond `2.2`, `2.9` at `L = 4, 8`; the coefficients to `2.9–4.6` eps, both precisions |
| the whole ansatz from random coefficients on a non-spherical surface, recovered by `solve_fit` | cond `99`, `1.5e3`, `6.0e3` (`L, cont = 4,1; 4,2; 6,2`); coefficients to `0.03–0.19 κ eps` |
| the analytic sampler's `∂_r h` against `background_state`'s gradient on the fixture's surface | `1.6e−8` |
| the chain rule against the converted samples differenced along the rays | `5.7e−10` (slopes), `7.3e−11` (curvatures) |
| Kerr-Schild `a = 0`, `cont = 1, 2`, the state on the fixture's offset surface (`r_1 = 1.22`) | `1.10` at `L = 1`; `5.1e−15`–`3.1e−13` at `L = 2, 4, 8` |
| the same from the state sampler on the exact solution, `L = 8`, `cont = 1` | `2.2e−4` (`N = 8`), `8.3e−6` (`N = 16`): rate `4.65` (`3.9` from 16 to 32); samples `2.6e−4`, `9.2e−6` |
| the angular mean of Kerr-Schild `g_ab` on `r = 1.15 M` | `g_tt = +0.739`, eigenvalues `(0.739, 1.580, 1.580, 1.580)`, signed `α = −0.86`: **not a metric** |
| the same mean of `(log α, β^i, γ_ij, Π_ab)`, reassembled | `α = 0.60`, `det γ = 3.94`: a metric |
| the sweep, Kerr-Schild `a = 0`, `L = 8` (`1225` points) | valid, `min λ(γ) = 1`, `min α = 0.527` / `0.480` (`cont = 1` / `2`), no hit |
| the sweep, Kerr-Schild `a = 9/10` on its oblate surface (`h = 5/128`, `m = 8`) | valid at `L = 4, 8, 12, 16`, both orders, `min λ ≥ 0.945`, no hit |
| its truncation, the state on the surface, `cont = 1` (`cont = 2`) | `0.26` (`0.32`), `0.051` (`0.075`), `9.1e−3` (`1.4e−2`) at `L = 4, 8, 12` |
| harmonic `a = 9/10`, `h = 5/256`, `m = 4`: `cont = 1` | valid for `L ≥ 8` (`min λ = 2.27`, value residual `1.9e−3` at `L = 8`); invalid at `L = 4` |
| the same, `cont = 2` | **invalid at every `L` from 4 to 16** (`min λ(γ) = −73`, 953 of 1225 points at `L = 8`); refused by `build_fit` |
| the same at `h = 5/512` | `cont = 1` valid from `L = 4`; `cont = 2` first valid at `L = 16` |
| harmonic `a = 9/10`, `m = 8`, `h = 5/256` | refused at the sample: the equatorial offset radius `0.843` is inside the disk |
| harmonic `a = 9/10`'s samples against `default_bounds` | `max |(α/√γ)Π| = 366/M` against `K_max = 100/M`, at 17 of 153 points |
| a boosted hole's shift, `boost(Harmonic(1, 0), 0.3 x̂)`, `L = 8`, `cont = 1`: without and with `shift_constant` | the state on the surface `1.1e−3` → `1.7e−5`; the shift's slope rows `22×` → `1.4e−4` of their scale |
| the evaluator against the least-squares model at the collocation points | `28` eps (`Float64`), `25` eps (`Float32`) of the largest variable; the projection the identity bit for bit |
| at the center | `β = 0` exactly, a valid metric; `FitParams` `isbits`; `fit_state` allocation-free |
| the tracked `:damped` run's state at `0.15 M` fitted, against the analytic solution's fit | `0.031` on the surface to `0.016` at the center; samples `0.034` off, masked error `0.035` L∞ / `3.1e−3` L2 |
| `build_fit` at `L = 8`: analytic `cont = 1`, `2`; state sampler | `3.6 ms`, `5.2 ms`; `7.0 ms` (`0.4–0.6 ms` at `L = 4`, `20–25 ms` at `12`) |
| `fit_state` per point at `L = 4, 8, 12`, `cont = 1` (`cont = 2`) | `0.64` (`0.88`), `2.2` (`2.9`), `4.3` (`6.0`) µs, the recurrence `20`, `85`, `165 ns` of it; one `u_exact` `91–96 ns` |
| `boost(KerrSchild(1, 0), 0.3 x̂)` at `t = 1` | `|g_tt| = 1.0e17` at `x = −0.3`, `4.9` at `+0.3`; `hole_velocity = −v` |

### The fitted variant, kernel half (step 8e-ii)

The cache, the `:fitted` branch, the driver's flow and the initial data;
the design is "The fitted target", pieces 8–12, and the decisions of the
review of 8e-i are marked there. The long rows are `hole_runs.jl fitted`
(`fitted=fixture`, `boosted`, `harmonic`).

**The suite.** **4616 assertions in 14m21 at one thread and 10m41 at
four** (load 6–8, rising to 13 during the four-thread run), against 8e-i's
4578 in 15m09 and 10m52. The 38 new claims are `fit_tests.jl`'s "The fitted
variant" — `47.7 s` at one thread and `39.5 s` at four, of which the
bit-identity testset is `6.3 s` and the fixture's `:fitted` run to `0.15 M`
with its one `Float32` chunk `41.5 s`: the step's one short run, and the
`Float32` chunk beside it — plus the 8e-i testsets' sixteen for the shift
constant's two paths.

**The acceptance list, with its numbers** (measured in step 8e):

| claim | number |
|---|---|
| one right-hand side, `:fitted` against `:damped` on the same geometry and rate, outside the offset surface | bit for bit at all 45 545 points; the core `−ρ_max (u − u_fit)` exactly; the layer's difference `ρ (u_fit − u_exact)` to `1.4e−14` |
| the cache against the host evaluator | bit for bit |
| a right-hand side, `:fitted` and `:damped` layers, fixture, one thread | `111 ms`, `107 ms` (again `106.7`, `107.6`; in the suite `114`, `118`) |
| a fill of the cache from one fit, from two | `44 ms`, `75 ms` — `0.4`, `0.7` of a right-hand side, once a chunk |
| the fixture `:fitted` to `0.15 M`: masked error L2, L∞, against `:damped` | `1.33e−2`, `0.177` against `3.11e−3`, `0.035` (`4.3×`, the initial data's kink; asserted `≤ 6×`) |
| the same: `fit_valid`, projection hits, `track_offset`, fit failures, mid-chunk refills | every row; `0`; below `3.3e−4` cells; `0`; `0` |
| the same to `1 M` | `1.74e−2` against `1.40e−2` (`1.24×`); the fit below the core surface `1.56e−2`, `cont = 2` `1.66e−2` |
| one chunk at `Float32` (samples in `Float64`) | masked error `9.76982e−3` against `9.76988e−3` (`7e−6`) |
| the moving seed, `boost(Harmonic(1, 0), 0.3 x̂)`, 848 blocks, `h = 5/128`, `0.1 M` | every find succeeds; track `−0.01492`, `−0.02985` against `−0.015`, `−0.03`; `track_offset ≤ 3.9e−3` cells; two pieces a chunk, one refill; masked error `3.26e−2` against `:damped`'s `2.03e−2` |
| harmonic `a = 9/10`, `m = 4`, `h = 5/256`, 2472 blocks: the initial data | finite and a metric at all 1 265 664 points; `min det γ = 0.648`, `min α = 0.208`, `min λ(γ) = 0.5` |
| the same: one right-hand side (four threads) | `0.71 s`, finite, `max |du| = 1.1e9` at the offset surface over the disk |
| the same: the run | ends in its first chunk (`2.5e−3 M`), a degenerate metric at the equatorial offset surface |
| the kink at the first evolved point, harmonic `a = 9/10`, `L = 8`: axis, 45°, equator (analytic second difference) | `9.5e5` (`416`), `1.8e7` (`111`), `9.4e7` (`6.4e8`); with `Π̃ = (α/√γ)Π`: `2.6e4`, `7.5e5`, `7.1e7`; with `Π̃` at `L = 12`: `39`, `1.3e4`, `7.4e7` |
| the same kink, Kerr-Schild `a = 0` fixture, `a = 9/10` equator, harmonic `a = 7/10` 45° | `7.7` (`15.9`), `46` (`61`), `6000` (`334`) |

### The generic interior: the measurement matrix (step 8f)

`test/hole_runs.jl generic` (its header says how the rows are grouped and
run), the decisions of the 8e hand-over it took (below), `fit_tilde` in
`src/fit.jl`, and `handover`, `target_source` and the `:fitted` mesh cycle
in `src/driver.jl`; the design it decides is the summary at the head of
[The interior](CODE.md#the-interior-a-pointwise-damping-layer). Every row is `q =
2`, `cfl = 1/4` (`1/5` on the moving rows), `ε_KO = 1/2`, `ρ_max = 4/M`
unless named, the ramp by step 8c's rule (`n_L = 8`), the finder every
chunk with the Korzyński spin; five Symmetry jobs, measured once
(2026-09-23/24), on `amdq` (`ks0`, `ks9`, `h7`) and in `amddebugq` hours with a
deadline (`harm`, `boost`: `budget=3300`; `h7` reached its deadline at
`6 M` there and was rerun to `10 M` on `amdq`; the boosted rows ran a
second and a third time, at `cfl = 1/5` and with the moving-step sizing,
after the CFL recheck stopped them at `1.5 M` and `1 M`).

**The suite.** **4632 assertions in 20m59 at one thread and 12m22 at four**
on the development machine (Apple silicon, Julia 1.13.0) under a load of
8–15 from other work (a four-thread run the evening before, on the same
tests without the moving-step change, measured 11m22 at a load of 5), against
step 8e's 4616 in 14m21 and 10m41. The sixteen new assertions are
`fit_tests.jl`'s: three for `Π̃`'s chain rule and round trip in "the static
hole's fit is the hole" (host-side, below a second), and "the snapshot, the
hand-over and the mesh cycle (step 8f)", **`25.9 s` at one thread and
`16.6 s` at four** — the snapshot cache and one right-hand side, the
hand-over run to `0.15 M`, and the adaptive fixture's mesh cycle as
`:fitted` with one regridding chunk. One assertion changed its expression
and not its claim: the evaluator's bit-identity with the reassembly now
reassembles with the fit's own `tilde` flag, the new default.

**The matrix** (the last row of each run: masked error L2 and L∞ over the
evolved region, the `G`-point shell's `C_a` L2 above the offset surface,
`C_a` L2 in step 8a's shells `[r_h + kh, r_h + (k+1)h)` outside the tracked
horizon, the layer residual — against the truth for the analytic variants,
against the target for `:fitted` — the drift of `h_tt` at the horizon's
evolved points, the finder's `M_irr`, `J` and `M_ch`, the largest track
offset from the analytic center in cells; no row fired the range projection
unless its hits are given, and every fit of every `:fitted` row was valid):

| case | row | reached | masked L2 / L∞ | shell `C_a` | `C_a` at `r_h` + 0, 2, 4 `h` | residual | drift | `M_irr` / `J` / `M_ch` | offset |
|---|---|---|---|---|---|---|---|---|---|
| Kerr-Schild `a = 0`, fixture, `m = 10` | `:damped` (control) | `50 M` | `2.36e−2` / `0.177` | `2.62e−2` | `4.2e−3`, `2.7e−3`, `1.9e−3` | `0.61` | `3.1e−3` | `0.99716` / `3.6e−6` / `0.99716` | `7.1e−3` |
| | `:fitted`, analytic data to the core surface | `50 M` | `4.59e−2` / `0.562` | `7.45e−2` | `4.8e−3`, `3.1e−3`, `2.2e−3` | `5.7` | `3.4e−3` | `0.99676` / `3.1e−6` / `0.99676` | `8.3e−3` |
| | `:fitted`, the decided data (fit below `r_1`) | `50 M` | `4.59e−2` / `0.562` | `7.45e−2` | the same | `5.7` | `3.4e−3` | `0.99676` | `7.8e−3` |
| | `:fitted`, fitting `Π` (not `Π̃`) | `50 M` | `4.53e−2` / `0.553` | `7.34e−2` | the same | `5.8` | `3.5e−3` | `0.99676` | `8.2e−3` |
| | the snapshot target | † `8 M` | `0.230` / `4.56` at `8 M` | `0.456` | `6.8e−3`, … | `190` | | `0.99805` | |
| | the finder's Kerr target (`M_ch = 1.00029`, `J = 2e−6`) | `50 M` | `2.36e−2` / `0.177` | `2.62e−2` | as `:damped` | `0.62` | `3.1e−3` | `0.99716` | `7.1e−3` |
| | hand-over: `:damped` to `5 M`, then `:fitted` | `50 M` | `4.59e−2` / `0.562` | `7.45e−2` | as `:fitted` | `5.7` | `3.4e−3` | `0.99676` | `1.0e−2` |
| Kerr-Schild `a = 9/10`, `h = 5/128`, 1632 blocks | `:damped`, `m = 5` (control) | `20 M` | `4.77e−3` / `4.54e−2` | `2.49e−3` | `1.1e−3`, `8.8e−4`, `9.5e−4` | `0.52` | `3.8e−3` | `0.84744` / `0.8978` / `0.9994` | `1.7e−3` |
| | `:fitted`, `m = 5`, `L = 12` | `20 M` | `0.196` / `2.46` | `0.100` | `2.5e−2`, `2.4e−2`, `1.5e−2` | `6.1` | `4.6e−2` | `0.8436` / `0.840` / `0.980` | `7.6e−3` |
| | `:fitted`, `m = 8`, `L = 12` | `20 M` | `0.258` / `4.32` | `0.109` | `9.7e−3`, `8.4e−3`, `8.4e−3` | `12.9` | `6.7e−2` | `0.8481` / `0.906` / `1.002` | `6.1e−3` |
| harmonic `a = 0`, `h = 5/128`, 960 blocks | `:damped` (control) | `10 M` | `0.234` / `7.9` | `7.5e−3` | `1.4e−3`, `9.5e−4`, `7.4e−4` | `41` | `1.3e−3` | `0.99754` | `2.6e−3` |
| | `:fitted` | `10 M` | `0.247` / `7.3` | `7.5e−3` | `1.8e−3`, `8.3e−4`, `5.9e−4` | `95` | `1.3e−3` | `0.99741` | `2.6e−3` |
| harmonic `a = 7/10`, `m = 4`, `lmax_shape = L = 12` | `:fitted`, `h = 5/256`, 2472 blocks | `10 M` | `3.60` / `113` | `0.168` | `3.7e−2`, `3.1e−2`, `2.9e−2` | `284` | `2.1e−2` | `0.9230` / `0.7073` / `0.9994` | `1.5e−3` |
| | `:fitted`, `h = 5/128`, 512 blocks, analytic data to `3h` | † `0.5 M` | `20` / `846` | `0.38` | | | | | |
| | `:fitted`, `h = 5/128`, the decided data | † `3.5 M` | `55` / `2570` | `2.35` | | | | `0.9187` | |
| `boost(Harmonic(1, 0), 0.3 x̂)` from `x = 0.75`, `h = 5/128`, 1128 blocks, `cfl = 1/5` | `:fitted`, tracked | `5 M` | `0.390` / `29` | `6.6e−2` | `1.8e−2`, `4.7e−3`, `4.8e−3` | `319` | `9.9e−3` | `0.99916` / `7e−6` / `0.99916` | `2.2e−2` |
| | `:fitted` at `20/M` (`n_L = 12`) | `5 M` | `0.428` / `20` | `7.1e−2` | `1.9e−2`, `1.4e−2`, `5.8e−3` | `125` | `7.3e−3` | `0.99934` | `1.8e−2` |
| | `:damped`, tracked, `4/M` | † `1.5 M` | `0.454` / `72` | `8.2e−2` | | `4.0e4` | | `1.0008` | |
| | `:damped`, tracked, `20/M` | `5 M` | `0.362` / `33` | `3.2e−2` | `3.5e−3`, `2.5e−3`, `1.1e−3` | `3.3e4` | `9.2e−3` | `0.99932` | `1.9e−2` |
| | `:damped`, tracked, the grid rate | `5 M` | `0.432` / `25` | `4.4e−2` | `7.7e−3`, `2.4e−3`, `2.7e−3` | `971` | `7.1e−3` | `0.99939` | `1.9e−2` |
| | `:damped`, step 5's sphere about the analytic center, `4/M` | † `1.0 M` | `0.288` / `52` | `4.4e−2` | | `8.6e4` | | `1.0009` | |
| | coasting: `:fitted`, the finder off from chunk 4 (`t = 1 M`), `max_misses = 6` | lost at `2.25 M` | `0.278` / `20` | `2.9e−2` | | `284` | | — | |

`†` a degenerate metric (`metric_quantities`' `DomainError`) in the evolved
shell. Kerr's values: `M_irr = 1`, `J = 0` for `a = 0`; `M_irr = 0.84744`,
`J = 0.9`, `M_ch = 1` for Kerr-Schild `a = 9/10`; `M_irr = 0.92580`, `J =
0.7`, `M_ch = 1` for harmonic `a = 7/10`.

What it says, row by row (**all measured in step 8f**):

- **The generic layer on the case that needs nothing costs a factor two.**
  On the static Kerr-Schild hole every `:fitted` row reaches `50 M` flat
  from `10 M` on, at `1.94×` the control's masked error and `2.8×` its shell
  `C_a`, with `M_irr` `4e−4` lower; the horizon shells outside it are
  within 15 %. The four `:fitted` rows — the analytic data to the core
  surface, the decided data, `Π` for `Π̃`, and the hand-over — all end on
  **the same state to three digits**: the steady error is the target's, not
  the initial data's and not the momentum variable's, and a fit that starts
  from evolved data at `5 M` (the hand-over; its projection fired 1764 times
  at the switch, at `r ≤ 0.175`, in the core, and never again) arrives at it
  within `5 M`. Step 8c's wrong-but-smooth analytic targets left the
  exterior at the exact target's error; a fit of the evolved state does not,
  because it is not a solution anywhere inside the offset surface and is
  wrong by `5.7` there against the analytic layer's `0.6`.
- **The finder's Kerr target is the analytic target** to four digits: the
  finder's `M_ch` and origin are `M` and the center to its truncation, so
  idea 4 is the analytic control by another road — useful after a merger,
  not here.
- **The snapshot target fails, at `8 M`.** Holding the evolved state still
  as the target (no fit, no regularity) grows the shell's error from the
  first chunk — `2.5e−2`, `0.11`, `0.24`, `0.46` at `2, 4, 6, 8 M` — until a
  degenerate metric ends the run: what the fit buys is its regularity, and
  a target that carries the state's own grid-scale content feeds it back.
- **On the spinning Kerr-Schild hole the fit is the error.** At `L = 12`
  the target's value residual is `2.5e−2` of the data (8e measured `L = 8`
  at 3–5 %), and the run carries it out: the masked error `41×` the
  control's, `J` `7 %` low and `M_ch` `2 %` low at `m = 5`, still slowly
  rising at `20 M`; a wider margin (`m = 8`) keeps the horizon's numbers at
  Kerr's (`0.906`, `1.002`) and the horizon shells 2.5× cleaner while the
  masked error, which includes the points between the offset surface and
  the horizon, is larger. The analytic target is the one to use on this
  chart, which admits it.
- **The harmonic chart at `a = 0` does not see the target**: `:fitted` and
  `:damped` agree to 5 % in the masked error and to 1 % in the shell. Both
  errors are the chart's: harmonic Schwarzschild at `h = 5/128` is steep
  between the offset surface and the horizon (the masked L∞ `7–8` sits
  there), and grows slowly to `10 M`.
- **G5's chart, harmonic `a = 7/10`, runs at `h = 5/256` and not at
  `5/128`.** At `5/128` the offset surface's equator is `3.6` cells from the
  ring and the run ends at `0.5 M` (analytic data to `3h`) or `3.5 M` (the
  decided data); at `5/256` it reaches `10 M` (`6293 s` on a node, `amdq`;
  the first hour-long attempt stopped at its deadline at `6 M` with the same
  numbers), every fit valid, the track within `1.5e−3` cells, with the
  horizon near Kerr's values (`M_irr` `0.30 %` low, `J` `1.0 %` high, `M_ch`
  `6e−4` low) and the masked error growing and slowing (`0.75`, `1.53`,
  `2.38`, `2.88`, `3.27`, `3.60` at `1, 2, 4, 6, 8, 10 M`; shell `C_a`
  `0.041` to `0.168`) — there is no analytic control on this chart to split
  the target's share from the chart's.

- **The moving hole: the fitted target holds it at `4/M`, and the analytic
  layer does not.** The boosted harmonic hole crosses `1.5 M` of the box in
  `5 M`; its track stays within `0.022` cells of the analytic center, every
  find succeeds, every fit is valid, `M_irr` within `8e−4` of `1`. The
  analytic `:damped` layer at the default rate ends at `1.5 M` on the
  tracked geometry and at `1.0 M` on step 5's sphere, with the masked L∞
  growing from the first chunk at the layer's *trailing* edge (measured
  locally: `3.9`, `10.5`, `36.5` at `1/4`, `1/2`, `3/4 M`, at the first
  evolved point on the `+x` side): its core is frozen (`w = ρ = 0`), a point
  crosses the eight-cell layer at `v = 0.3` in about `M`, and at `4/M` the
  ramp relaxes what the core held by only `e^{−2}` before releasing it —
  CODE.md's "points the core releases are relaxed within `1/ρ_max`" was
  written for the grid rate. At `20/M` (the rule's ramp, twelve cells) or
  at the grid rate the same layer reaches `5 M`, with the projection firing
  in the core throughout (`2.2e5` and `4.0e4` hits, at `r ≤ 0.37`). The
  `:fitted` core is not frozen — it relaxes toward the fit, which moves with
  the track — and reaches `5 M` at `4/M` without a hit, at the masked error
  of the rescued analytic rows (`0.39` against `0.36`–`0.43`) and `1.5–2×`
  their shell. **So a moving analytic layer needs `ρ_max ≳ 20/M` (proposed
  in step 8f); the fitted one does not.** Every row's masked error grows
  with the crossing, `≈ 0.08/M`, the same in all five survivors: the chart's
  truncation at `5/128` near the moving hole, as on the static harmonic
  hole.
- **Coasting costs nothing for five chunks.** With the finder off from
  `t = 1 M` the geometry is carried on the track's `v_est` for `1.25 M`,
  and the masked error is the tracked run's to 1 % (`0.2585` against
  `0.2608` at `2 M`); the sixth miss ends the run with its record by
  `TrackLostError` at `max_misses = 6`, as designed (the row was meant to
  coast five chunks and ran the window one chunk long).

The kink table and the price of harmonic `a = 9/10` are the probe's
(`generic=probe`, four threads here, three minutes), recorded under [Open
questions](CODE.md#open-questions); the kink at the first evolved point,
`|Δ²(composite) − Δ²(analytic)|` against `|Δ²(analytic)|`, maximum over the
twenty components, the fit of the analytic solution (`cont = 1`):

| chart | `L = 8`, `Π` | `8`, `Π̃` | `12`, `Π` | `12`, `Π̃` (axis / 45° / equator) |
|---|---|---|---|---|
| harmonic `a = 9/10`, `5/256` (analytic `420`, `107`, `2.4e9`) | `8.8e5`, `2.0e7`, `1.6e9` | `2.3e4`, `8.7e5`, `1.5e9` | `1.1e4`, `8.9e5`, `1.6e9` | `366`, `2.5e4`, `1.5e9` (`L = 16`: `552`, `3.1e3`, `1.4e9`) |
| harmonic `a = 7/10`, `5/256` (`232`, `334`, `4.7e4`) | `7.9e3`, `3.7e4`, `1.3e4` | `2.2e3`, `1.3e4`, `1.0e4` | `727`, `5.0e3`, `1.4e4` | `193`, `1.4e3`, `1.0e4` |
| harmonic `a = 7/10`, `5/128` (`300`, `305`, `2.8e5`) | `8.1e3`, `4.8e4`, `1.0e5` | `1.7e3`, `1.3e4`, `8.4e4` | `1.2e3`, `8.5e3`, `1.1e5` | `241`, `1.9e3`, `9.2e4` |
| Kerr-Schild `a = 9/10`, `5/128` (`3.5`, `6.8`, `103`) | `17`, `27`, `55` | `8.9`, `16`, `46` | `3.4`, `2.6`, `42` | `2.2`, `2.1`, `35` |
| harmonic `a = 0`, `5/128` (`4050`) | `865` | `561` | `865` | `561` |
| Kerr-Schild `a = 0`, fixture (`15.4`) | `5.3`, `3.3`, `5.3` | `3.7`, `2.4`, `3.7` | as `L = 8` | as `L = 8` |

(The point is `r_1 + h/2`, half a cell out of step 8e's probe, whose
numbers it reproduces to a factor of two off the equator.) `Π̃` shrinks
every kink, 1.5× on the static holes and 2–40× off the spinning holes'
equators, and `L = 12` another 3–30× on the spinning ones; neither touches
a static hole's, which is radial. In the evolution `Π̃` changes nothing
measurable (the Kerr-Schild `a = 0` rows above).

**The decisions of the 8e hand-over (proposed in step 8f):**

- **`Π̃ = (α/√γ)Π` is the fitted momentum by default** (`FittedSpec`'s
  `fit_tilde = true`): the table above, and `fit_tests.jl`'s chain rule
  (`8.1e−8`, `1.2e−8` against the differenced `Π̃`, the stencil's `δ⁴`).
- **`lmax_fit` stays `8` by default and the spinning rows carry `12`**, as
  `lmax_shape` does: a static hole's fit is `l ≤ 2`, and the evaluator's
  cost doubles from `8` to `12` (`2.2` to `4.3 µs` a point, once a chunk
  in the cache fill: `17 s` of fills over the `ks9` row's 40 chunks against
  its `7300 s`).
- **The regrid path is built, and the moving rows use a wider fine region
  instead.** `adapt = true` on a `:fitted` case chooses the mesh on the
  analytic `:damped` data of the same geometry and fills the fitted data on
  the settled mesh (one pass to 288 blocks on the adaptive fixture, and a
  regridding chunk, in the suite), and refuses a chart whose analytic core
  surface meets its singular set — G5's own, where step 8 needs the
  indicator to flag on something else (the fitted data through a callback
  that reads the cache, or the state after the first chunk). The boosted
  rows cross a fixed capsule of fine blocks, which measures the layer and
  not the regrid.
- **The hand-over row is built** (`evolve!(…; handover)`) and measured
  above; **coasting** is the boosted row with the finder off for chunks
  4–9.

**What the matrix costs.** On a Symmetry node shared by seven rows of nine
threads, the Kerr-Schild fixture is `140 s` a `M` a row (seven `50 M` rows
in `7057 s`); Kerr-Schild `a = 9/10` on 1632 blocks `360–410 s` a `M` at 21
threads; harmonic `a = 0` on 960 blocks `195 s` a `M` at 16 threads; G5's
chart on 2472 blocks `630 s` a `M` at 64 threads. The fits are `16 ms` and
a cache fill `47 ms` on the fixture (101 and 100 of them in a `50 M` row),
and `46 ms` and `420 ms` on the `ks9` mesh. Before the workers were given
one BLAS thread each, OpenBLAS's pools spinning after every fit put a node
at twice its cores and the rows at ten times this machine's time per `M`.

### The moving hole (step 8)

`test/hole_runs.jl moving` (its header says how the rows are grouped and
run), `test/moving_tests.jl`, and in `src/` the `:fitted` cycle
(`adapt_fitted_initial_data!`, `fill_fitted_initial!`), the tracked floor
from the core surface widened by the travel, the moving seed's `r_min`, the
1 % step margin, the target's rate and the blended initial data. Every row
is `q = 2`, `cfl = 1/5`, `ε_KO = 1/2`, `ρ_max = 4/M` unless named, `chunk
= M/4`, the finder every chunk with the Korzyński spin (`N_ah = 16` on G5's
chart, `12` on `a = 0`), `boost(·, 0.3 x̂)` moving the hole at `−0.3 x̂`.
G5's chart is harmonic Kerr `a = 7/10` with `m = 4`, `n_L = 8`,
`lmax_shape = lmax_fit = 12`, finest `h = 5/256`, the initial data blended
from the analytic solution at the offset surface into the fit at the core
surface; the `a = 0` hole has `m = 8`, `n_L = 8` (`12` at `20/M`), `lmax =
4, 8`, finest `5/128`. Symmetry jobs on `amdq` and in `amddebugq` hours
(the screens, `budget=3300`), 2026-09-24.

**The suite.** **4659 assertions in 17m16 at one thread and 11m36 at four**
on the development machine (Apple silicon, Julia 1.13.0, load 7–13),
against step 8f's 4632 in 20m59 and 12m22 under a heavier load. The 28 new
assertions are `test/moving_tests.jl`, **`19.9 s` at one thread and
`16.9 s` at four**, of which `13.2 s` is the target's rate (the cache fill
with its two extra evaluations compiles anew); one of step 8f's assertions,
the refusal of G5's chart by the `:fitted` cycle, went with the refusal,
and one of step 8d's changed its claim with the floor
(`lb.floor_lo == layer_radii(geom)[1]`).

**The `a = 0` hole across the box (`moving=ctl`, measured in step 8).**
From `x = 2` to `x = −1.9` in `13 M`, box `5 M` on a `4³` root brick, the
indicator's mesh re-chosen at every chunk (the adaptive rows), against the
same hole in a box of `5/2 M` on the capsule-free adaptive mesh and on a
uniform mesh at the finest spacing:

| row | reached | masked L2 / L∞ at the end | error outside `r_h,max + 3h`, L2 / L∞ | shell `C_a` | trailing / leading outer quarter of the layer, off the truth | projection hits | centroid offset, max (finest `h`) | `M_irr` / `J` at the end | wall |
|---|---|---|---|---|---|---|---|---|---|
| `:fitted`, tracked | `13 M` | `0.174` / `33.8` | `1.66e−2` / `1.08` | `0.103` | `27.9` / `16.9` | 0 | `7.3` | `0.9965` / `1.6e−5` | `5947 s` |
| `:damped` at `20/M`, tracked | `13 M` | `0.153` / `40.6` | `1.55e−2` / `0.85` | `2.6e−2` | `33.6` / `12.2` | 677 214 | `1.4` | `0.9966` / `2.9e−6` | `5776 s` |
| `:frozen`, tracked | † `1.0 M` | `0.360` / `178` at `1 M` | `3.8e−3` / `0.22` | `0.174` | `345` / `15.8` | 0 | — | `1.001` | `744 s` |
| `:fitted`, box `5/2`, adaptive (820–1464 blocks) | `5 M` | `0.391` / `29.2` | `2.35e−2` / `0.59` | `6.6e−2` | `25.3` / `12.7` | 0 | `7.3` | `0.9993` / `6.7e−6` | `1840 s` |
| `:fitted`, box `5/2`, uniform `5/128` (4096 blocks of `8³`) | `5 M` | `0.434` / `36.9` | `1.66e−2` / `0.77` | `5.1e−2` | `27.4` / `11.4` | 0 | — | — | `4693 s` |

`†` a degenerate metric (`metric_quantities`' `DomainError`) in the evolved
shell, the frozen core's stale data released on the trailing side.
40 regrids in 52 chunks, 1128–1912 blocks; the track within `0.022` (`:fitted`)
and `0.028` (`:damped`) cells of the analytic center; the found horizon's
extent along the boost over its extent across it `0.9542` and `0.9533`
against the contraction `√(1 − v²) = 0.95394` at `13 M`.

**The frozen hierarchy on the `a = 0` hole (`moving=conv`, measured in step
8).** The analytic sphere about the moving analytic center at `20/M`, `r_1 =
r_h,min − m h`, `m = N` and a ramp of `12 N/8` cells — the same surfaces at
every `N` — on a capsule of fine blocks around the trajectory to `1/2 M`
(1492 blocks), `N = 8, 12, 16` (finest `5/128`, `5/192`, `5/256`):

| `t` | masked L2 (`N = 8, 12, 16`) | rates | error outside `r_h,max + 3h₈`, L2 | rates |
|---|---|---|---|---|
| `M/8` | `3.20e−2`, `1.33e−2`, `7.26e−3` | `2.17`, `2.10` | `1.05e−3`, `5.50e−4`, `2.51e−4` | `1.59`, `2.73` |
| `M/2` | `0.106`, `4.46e−2`, `2.45e−2` | `2.14`, `2.08` | `3.60e−3`, `1.89e−3`, `8.55e−4` | `1.59`, `2.76` |

— order `q = 2` in the masked norm over the crossing's first `0.15 M` of
travel, and on average in the far field (whose `N = 8` row has the coarse
levels' error in it). The projection fired in the frozen core (4820, 10 208
and 2896 hits at `N = 8, 12, 16`), never outside it.

**G5's chart moving: what the layer needed (the screens, measured in step
8).** Box `5/2 M` on a `2³` root brick, from `x = 0.3`, adaptive
(3536 blocks at `t = 0`, 4880 after the first regrid), masked L2 / L∞ at
`M/4` and `M/2`, and the largest error off the truth in the outer quarter of
the layer on the trailing side (where the moving core releases points):

| row | masked at `M/4` | at `M/2` | error outside `r_h,max + 3h` at `M/2`, L2 / L∞ | trailing outer quarter at `M/4` |
|---|---|---|---|---|
| static, the step at the core surface | `3.23e−2` / `2.36` | `9.32e−2` / `13.5` | `9.4e−3` / `0.53` | — |
| static, blended | `7.47e−2` / `9.55` | `0.157` / `20.1` | `9.4e−3` / `0.54` | — |
| moving, the step (step 8f's data), rate on | `0.492` / `334` | `1.48` / `533` | `1.2e−2` / `5.5` (`0.116` / `41.5` at `M`) | `1425` |
| moving, the step, rate off | `0.494` / `334` | `1.51` / `544` | `1.2e−2` / `5.6` | `1417` |
| moving, the step, `w_ramp = 1` | `0.132` / `54.5` | `2.37` / `1087` | `1.1e−2` / `2.5` | `544` |
| moving, the step, `w_ramp = 1`, rate off | `0.172` / `64.9` | `2.51` / `1137` | `1.1e−2` / `2.6` | `518` |
| moving, the step, `w_ramp = 1`, `20/M` | `0.136` / `60.8` | `0.363` / `98.6` | `1.1e−2` / `1.6` | `191` |
| moving, the step, `w_ramp = 1`, `n_L = 16` | `0.309` / `225` | `1.24` / `707` | `1.1e−2` / `3.0` | `1973` |
| moving, the step, `w_ramp = 1`, `m = 6` (fit from `6h`) | `2.11` / `1413` | `5.40` / `2347` | `1.1e−2` / `4.6` | `2692` |
| moving, the fit from the offset surface, `m = 8` | `1.66` / `151` | `1.64` / `169` | `1.1e−2` / `0.76` | `953` |
| **moving, blended** (G5's rows) | **`0.115` / `18.1`** | **`0.351` / `121`** | `1.1e−2` / `0.98` | **`53`** |
| moving, blended, `w_ramp = 1` | `0.321` / `110` | `1.26` / `278` | `1.0e−2` / `1.7` | `170` |
| moving, blended, `ρ_ramp = 1/2` | `0.162` / `19.3` | `0.295` / `34.5` | `1.2e−2` / `1.2` | `78` |
| static, blended, `ρ_ramp = 1/2` | `0.110` / `10.0` | `0.214` / `16.4` | `9.3e−3` / `0.51` | — |
| moving, blended, `20/M` (`n_L = 12`, fit from `8h`) | `0.204` / `46.2` | `0.541` / `108` | `1.1e−2` / `1.5` | `297` |
| moving, blended, `m = 6` (fit from `6h`) | `0.478` / `112` | `0.864` / `243` | `1.0e−2` / `0.58` | `247` |

What it says (**measured in step 8**): the moving `:fitted` layer on G5's
chart is dominated by what its **initial data** carry through the trailing
side, not by the target's lag — the target's rate changes the first chunk by
`0.2 %` on the default ramp; the step at the core surface, where the data
jump from the analytic solution to the fit by `25×` the solution, is what
moves outward in depth with the hole into the evolved part of the layer
(`Π_xx = +2650` four cells deep on the trailing equator at `M/4` against a
solution of `1100` and a fit `580` below it). Blending the data into the fit
across the layer takes the first chunk from `0.49` / `334` to `0.115` / `18`
and the trailing layer from `1425` to `53` off the truth; the step's cost on
the static hole is a factor two in the first chunk. The wider margins lose,
because their initial data have to switch to the fit where the ring is
close: the analytic solution is singular on a core surface deeper than
`7h` on the equator at `5/256`. `ρ_ramp = 1/2` trades the first chunk for a
slower growth, `0.162` → `0.295` against `0.115` → `0.351`, and costs the
static hole more; G5's rows keep step 8c's ramps.

**G5's crossing (`moving=g5`, measured in step 8).** Harmonic Kerr `a =
7/10` boosted at `0.3` from `x = 2` toward `x = −1.9`, box `5 M` on a `4³`
root brick, the indicator's mesh chosen by the `:fitted` cycle and
re-chosen at every chunk boundary (3956–5216 blocks, two passes to 4040 at
`t = 0`), against the same hole **at rest** at the center on the same
machinery (3872 blocks throughout). The rows are read from the workers'
per-chunk lines; the jobs were still running when this was written (the
tunnel to Symmetry closed at `7 M` and `8.25 M`, see the end of this entry):

| `t` | moving: masked L2 / L∞ | error outside `r_h,max + 3h`, L2 / L∞ | at rest: masked L2 / L∞ | outside, L2 / L∞ |
|---|---|---|---|---|
| `M/2` | `0.121` / `126` | `4.4e−3` / `1.2` | `5.5e−2` / `20.1` | `3.3e−3` / `0.54` |
| `1.25 M` | `0.385` / `153` | `1.7e−2` / `13.4` | `0.139` / `45.1` | `5.5e−3` / `0.73` |
| `1.75 M` | `0.462` / `190` | `3.4e−2` / `12.7` | `0.181` / `55.5` | `6.4e−3` / `1.2` |
| `2.5 M` | `0.610` / `227` | `4.0e−2` / `16.7` | `0.228` / `64.0` | `9.0e−3` / `3.2` |
| `6.25 M` | `1.107` / `478` | `7.3e−2` / `22.0` | `0.347` / `89.8` | `2.1e−2` / `4.8` |
| `7 M` | `1.229` / `547` | `7.6e−2` / `23.2` | `0.368` / `95.9` | `2.3e−2` / `4.9` |

The moving run's first chunk is `4.05e−2` / `17.5` (outside `2.5e−3` /
`0.48`). No projection hit, no
failed find; the horizon along the trajectory at `1.5 M`: `M_irr = 0.9254`
(Kerr `0.92580`), `J = 0.6979` (`0.7`), `M_ch = 0.9993`, the found
surface's extent along the boost over its extent across it **`0.9534`
against `√(1 − v²) = 0.95394`** (`0.9539` in the first chunks); at rest
`M_irr = 0.9255`, `J = 0.6980`, `M_ch = 0.9994` at `1.75 M`, extent ratio
`1.000`. The refinement centroid of this mesh at `t = 0`, before anything
moves, is **`0.13 M` = 6.7 finest spacings** behind the center along the
boost (`0.29` on the boosted `a = 0` hole, `0.08` on the hole at rest): the
boosted horizon-penetrating chart is not symmetric along the boost, and a
spinning one much less so, so the Löhner field the centroid weights is
displaced, at the coarse levels' spacing (step 6's bias note).

What it says (**measured in step 8**): **the moving hole's masked error is
not at the static run's level.** It is `3.3×` it in L2 and `5.7×` in L∞ at
`7 M`, and grows at `0.14` a `M` against the static `0.03` from `2 M` on; the
far field outside the horizon, `3.3×` and `4.7×`. At `M/2` it is `2.2×`,
so the excess is made during the crossing, and the trailing side of the
layer says where: the error off the truth in the layer's outer quarter is `52` / `57`
(trailing / leading) at `M/4`, `148` / `104` at `1.5 M`, `196` / `107` at
`2.5 M`, where the static hole's is `50`–`60` — **points released by the
moving layer are not relaxed within `1/ρ_max`** at `4/M`: a point crosses
the eight-cell ramp at `v = 0.3` in `0.52 M`, two e-folds at `4/M`, and
`20/M` (the `g5z` screen, which also needs a thicker ramp and fits its data
from `8h`) did not do better in the first `M/2`.

**The adaptive run against the uniform mesh (`moving=g5u`, measured in step
8).** The crossing in a box of `5/2 M` from `x = 0.3`, adaptive (3536–5020
blocks of `8³`, `1.8–2.6 × 10⁶` points) against a uniform mesh at the finest
spacing `5/256` (4096 blocks of `16³`, `1.68 × 10⁷` points):

| `t` | adaptive: masked L2 / L∞ | uniform | adaptive: outside `r_h,max + 3h` L2 | uniform |
|---|---|---|---|---|
| `M/4` | `0.1152` / `18.12` | `0.1152` / `18.12` | `5.9e−3` | `2.5e−3` |
| `M/2` | `0.3505` / `121.2` | `0.3504` / `121.2` | `1.1e−2` | `4.7e−3` |
| `3M/4` | `0.6773` / `161.0` | `0.6772` / `161.0` | `2.0e−2` | `6.2e−3` |
| `M` | `0.9462` / `156.9` | `0.9471` / `156.9` | `3.0e−2` | `8.0e−3` |
| `1.25 M` | `1.101` / `153.7` | `1.102` / `153.8` | `4.7e−2` | `1.8e−2` |
| `1.5 M` | `1.185` / `162.2` | `1.193` / `164.4` | `7.7e−2` | `3.1e−2` |
| `1.75 M` | `1.314` / `187.6` | `1.323` / `188.2` | `9.7e−2` | `3.9e−2` |
| `2 M` | `1.478` / `207.9` | `1.483` / `212.4` | `0.104` | `5.3e−2` |

— the adaptive run is the uniform one to three digits where the error is,
at the hole, with **6.5–9.3 times fewer points**; outside the horizon its
error is 2.3–3.7 times the uniform run's, the coarse levels' truncation
(the `a = 0` pair to `5 M` above: `1.4×` outside, the masked error within
`10 %`). **The uniform run reached `2 M` in step 8′** (rows `1.25`–`2 M`), in
`17 739 s` on 64 threads against the adaptive run's `5707 s` on 12: within
`0.3 %` in the masked L2 and `2 %` in L∞ to the end, the far field twice the
uniform run's by then. In step 8 both attempts ended at `0.75 M` and
`1.0 M` with a `SIGBUS` inside Julia's allocator at 76–88 GB of resident
memory (240 GB allowed), and the `N = 16` row of the frozen hierarchy
(`2 × 10⁷` points) met the same once. **The `SIGBUS` diagnosis (step 8′,
PLAN.md's two attempts)**: both attempts — `--heap-size-hint=160G`, and
`OPENBLAS_NUM_THREADS=1` with `JULIA_NUM_GC_THREADS=8` — were run from a
remote directory of their own (`step-8-sigbus`), and the first reached
`2 M` (the second too, the same numbers to the last digit printed). Neither setting is therefore
shown to matter; what changed is the directory. Julia's package images on
Symmetry are cached in one slot per package and project path, and an rsync
of changed sources into a running study's directory followed by another
job's precompile rewrites that slot's `.so` in place — the `.so` for the
step-8 path was rewritten at 09:15 and 19:46 on 2026-09-24 and none of the
day's other compiles survives — so a long-running process that pages in its
image after that can take a `SIGBUS`. **Proposed in step 8′**: a batch study
that runs for hours gets a remote directory of its own
(`symmetry-run.sh <worktree> <unique name>`); the kill times are not all
explained by an rsync (the 18:20 pair followed none), so this is the likely
mechanism and not a proof.

**The frozen hierarchy on G5's chart (`moving=conv`, measured in step 8).**
The capsule around the trajectory to `M/2` (4908 blocks), `N = 8, 12, 16`
(finest `5/256`, `5/384`, `5/512`), `m = N/2` and `n_L = N` so that the
offset and core surfaces are the same surfaces, the data blended:

| `t` | masked L2 (`N = 8, 12, 16`) | rates | outside `r_h,max + 3h₈`, L2 | rates |
|---|---|---|---|---|
| `M/8` | `5.13e−2`, `2.02e−2`, `1.02e−2` | `2.30`, `2.37` | `2.81e−3`, `1.48e−3`, `6.65e−4` | `1.59`, `2.77` |
| `M/4` | `0.114`, `5.34e−2`, `2.19e−2` | `1.87`, `3.10` | `5.31e−3`, `2.79e−3`, `1.25e−3` | `1.59`, `2.80` |
| `3M/8` | `0.202`, `7.48e−2`, `4.59e−2` | `2.44`, `1.70` | `7.53e−3`, `3.98e−3`, `1.78e−3` | `1.57`, `2.80` |
| `M/2` | `0.351`, `9.00e−2`, `8.44e−2` | `3.36`, `0.22` | `1.01e−2`, `5.04e−3`, `2.29e−3` | `1.71`, `2.74` |

— **order `q = 2` over the first `0.15 M` of travel** in the masked norm
(`2.3` over both intervals at `M/8`), and outside the horizon at every row
(`2.1` averaged over the three resolutions, the `N = 8` row's far field
carrying the coarse levels' error as on the `a = 0` hole); **but the masked
norm stops converging by `M/2`** (`N = 12` to `16`: `0.22`, the L∞ `13.7`
against `24.5`): the moving layer's excess is not truncation error and does
not shrink with `h` (the `N = 16` row, completed after the tunnel returned;
recorded in step 8′).

**Where the runs stood (2026-09-24, 18:50), and what they recorded
(read in step 8′).** The tunnel to Symmetry closed while `g5-adaptive`,
`g5-static` and `conv-h7-N16` were running; it returned at 19:40. The
per-chunk record of the finished adaptive G5 row (`g5b-adaptive`, the `5/2`
box, 9 rows to `2 M`) and of the `a = 0` rows:

| row | centroid offset from the analytic center, finest `h`: `t = 0` / median / max | track offset, max (cells) | fits valid | finds | blocks |
|---|---|---|---|---|---|
| G5's chart, `g5b-adaptive` | `6.8` / `27.5` / `29.8` | `0.060` | 9 of 9 | 9 of 9 | 3536–5020 |
| `a = 0`, `:fitted`, box `5 M` | `0.29` / `4.8` / `7.3` | `0.022` | 53 of 53 | 53 of 53 | 1128–1912 |
| `a = 0`, `:damped` at `20/M` | `0.29` / `0.49` / `1.4` | `0.028` | 53 of 53 | 53 of 53 | 1128–1912 |
| `a = 0`, `:fitted`, box `5/2` | `0.29` / `3.3` / `7.3` | `0.022` | 21 of 21 | 21 of 21 | 820–1464 |

So on G5's chart **the refinement centroid is not within a few finest
spacings** once the hole moves: `28` of them, `0.55 M`, from the first
moving chunk on (**measured in step 8′**). The indicator fires on what the
moving layer exports — the fitted `a = 0` row is five times the analytic
one's offset for the same reason — and the centroid measures the refined
region's asymmetry, not a failure to follow: the track stays within `0.06`
cells, every find succeeds and every fit is valid. **The crossing in the `5 M` box completed (`g5-2`, `28 434 s` on 40
threads, 51 regrids, 3886–5580 blocks)**: every fit valid (53 of 53
rows), every find successful, the track within `0.215` cells of the analytic
center; the refinement centroid `27.5`–`32.3` finest spacings off from the
first moving chunk on (`0.12` on the hole at rest, whose track stays within
`0.0067` cells). The masked error at `13 M` is `2.59` / `1328` against the
resting hole's `0.517` / `142` — `5.0×`, and still growing faster
(`0.35` a `M` over the last two against `0.022`) — and **the horizon
degrades after `4 M`**: `J = 0.852`, `M_ch = 1.030`, `M_irr = 0.9199` at
`13 M` (Kerr `0.7`, `1`, `0.92580`; the resting hole `0.7102`, `0.9985`,
`0.9211`), the extent ratio `0.949` from `8 M` against `0.954`. Step 8's
table above stops at `7 M`; its time series every `M`: masked L2 `0.333`,
`0.517`, `0.678`, `0.811`, `0.949`, `1.079`, `1.229`, `1.358`, `1.540`,
`1.733`, `1.961`, `2.233`, `2.586` against `0.115`, `0.199`, `0.248`,
`0.284`, `0.316`, `0.340`, `0.368`, `0.392`, `0.421`, `0.449`, `0.474`,
`0.498`, `0.517` at rest (recorded in step 8′).
### The trailing side (step 8′)

`PLAN.md`'s step 8′: three levers on the side a moving `:fitted` layer
leaves, each an `evolve!` keyword that is the unchanged code path when off
(the kernel branches on `trail = 0` and on the exact target's argument being
`nothing` before it touches anything, so a run without them is bit for bit
the run before them; `test/moving_tests.jl` asserts that the ramp narrows
only behind a moving layer and that the exact target moves the layer and
nothing outside the offset surface) — **all proposed in step 8′**:

- **`trail_ramp = σ`**: `ρ`'s ramp fraction narrowed to `ρ_ramp (1 − σ ζ)`,
  `ζ = max(0, −n̂ · v̂)` about the tracked center, so that on the trailing
  side — where the depth of a grid point decreases at `|n̂ · v|` and points
  leave the layer — `ρ` reaches `ρ_max` nearer the offset surface; `w` is
  unchanged.
- **`refill_cells = f`**: the target refilled every `f` cells of the track's
  travel instead of step 8e's `1/4`.
- **`target_exact = true`**: the target in the layer and the core is the
  latest fit evaluated in the kernel at `(x, t)` — carried by its tracked
  center exactly — instead of the cache's linear continuation between
  refills (my reading of "the target's evolved continuation, the cache
  advected with `F` off", **proposed in step 8′**: the cache advected
  exactly is the fit evaluated where the hole has moved it).

**The screen on the boosted `a = 0` hole (`moving=l0`, measured in step
8′**, locally at four threads, `3.7 min` a row: box `5/2 M`, from
`x = 3/4`, adaptive, finest `5/128`, to `M/2`; masked L2 / L∞, the excess
of the masked L2 over the hole at rest, and the largest error off the truth
in the layer's outer quarter on the trailing and the leading side**)**:

| row | masked L2 / L∞ | excess over rest | trailing / leading outer quarter |
|---|---|---|---|
| at rest | `5.70e−2` / `1.90` | — | — / `4.4` |
| `:damped` at `20/M` (the analytic control) | `8.30e−2` / `6.11` | `2.6e−2` | `15.8` / `6.4` |
| `:fitted` | `0.1296` / `24.4` | `7.3e−2` | `88.6` / `5.0` |
| `trail_ramp = 1/2` | `0.1119` / `21.2` | `5.5e−2` | `71.9` / `5.0` |
| **`trail_ramp = 3/4`** | **`8.92e−2` / `12.0`** | **`3.2e−2`** | `69.0` / `5.0` |
| **`trail_ramp = 9/10`** | **`8.60e−2` / `8.99`** | **`2.9e−2`** | `67.6` / `5.0` |
| `trail_ramp = 3/4` at `8/M` | `0.1117` / `5.83` | `5.5e−2` | `42.0` / `9.7` |
| `refill_cells = 1/16` | `0.1295` / `24.4` | `7.2e−2` | `88.5` / `5.0` |
| `target_exact` | `0.1296` / `24.4` | `7.3e−2` | `88.5` / `5.0` |
| `trail_ramp = 3/4` and `target_exact` | `8.77e−2` / `11.3` | `3.1e−2` | `66.1` / `4.9` |

**On G5's chart (`moving=l7`, measured in step 8′**, the `g5y` screen's
configuration: box `5/2 M`, from `x = 0.3`, finest `5/256`, the data
blended, to `M/2`, four workers of 16 threads in one `amddebugq` hour**)**:

| row | masked at `M/4` | at `M/2` | excess over rest at `M/2` | trailing / leading outer quarter at `M/2` |
|---|---|---|---|---|
| at rest (blended, `g5y`) | `7.47e−2` / `9.55` | `0.157` / `20.1` | — | — / `32.9` |
| `:fitted` (`g5y`) | `0.115` / `18.1` | `0.351` / `121` | `0.194` | `157` / `80` |
| `trail_ramp = 3/4` | `0.139` / `24.7` | `0.225` / `22.9` | `0.068` | `113` / `80` |
| **`trail_ramp = 9/10`** | `0.129` / `24.3` | **`0.216` / `22.5`** | **`0.059`** | `104` / `80` |
| `target_exact` | `0.115` / `18.1` | `0.352` / `122` | `0.195` | `157` / `80` |
| `trail_ramp = 3/4` and `target_exact` | `0.139` / `24.8` | `0.224` / `23.0` | `0.067` | `95` / `80` |

What they say (**measured in step 8′**): **the side-dependent ramp is the
lever, and the other two are nothing.** On both holes it cuts the masked
excess over the hole at rest by more than half at `M/2` — `a = 0` from
`7.3e−2` to `2.9e−2`, the analytic control's `2.6e−2`; G5's chart from
`0.194` to `0.059`, the L∞ from `121` to `22.5` against the resting hole's
`20.1` — at the price of a larger first chunk on G5's chart (`0.129`
against `0.115` at `M/4`: a steeper ramp behind the hole is a harder paste
there). The trailing side's outer quarter is still `1.3×` the leading
side's on G5's chart (`104` against `80`) and `13×` on `a = 0` (`68`
against `5`), where the analytic layer at `20/M` has `2.5×`; so points
released behind the hole are relaxed better and not yet as well as ahead of
it. Refilling sixteen times as often, or evaluating the fit exactly in the
kernel, changes the fourth digit: the target's representation between
refills is not what the trailing side suffers from — what the layer does to
the points that cross it is. A faster rate with the narrow ramp (`8/M`)
trades L2 for L∞.

**G5's crossing with the trailing ramp (`moving=g5t`, measured in step
8′**: `trail_ramp = 9/10`, otherwise `g5-adaptive`'s row — box `5 M`, from
`x = 2`, `13 M`, alone on an `amdq` node, `39 183 s`, 51 regrids, 3872–5356
blocks**)**, against the same crossing without it and the hole at rest:

| `t` | with the ramp: masked L2 / L∞ | without | at rest | with / at rest |
|---|---|---|---|---|
| `M` | `0.138` / `41.8` | `0.333` / — | `0.115` / `39.6` | `1.20` |
| `2 M` | `0.245` / `60.0` | `0.517` / `208` | `0.199` / — | `1.23` |
| `4 M` | `0.380` / `75.0` | `0.811` / `327` | `0.284` / `71.7` | `1.34` |
| `7 M` | `0.540` / `104` | `1.229` / `547` | `0.368` / `95.9` | `1.47` |
| `10 M` | `0.757` / `156` | `1.733` / — | `0.449` / — | `1.69` |
| `13 M` | `1.062` / `262` | `2.586` / `1328` | `0.517` / `142` | `2.05` |

The excess of the masked L2 over the resting hole is **`0.545` against
`2.069` at `13 M`** (`0.096` against `0.527` at `4 M`); the error outside
`r_h,max + 3h` is `0.105` / `19.1` against `0.120` / `30.0` (the resting hole
`3.4e−2` / `5.8`). **The trailing side's excess is gone**: the layer's outer
quarter off the truth is `268` behind the hole and `277` ahead of it at
`13 M` (`2729` / `2942` without the ramp, where both sides had grown; `1.6`
and `1.3` at `M` and `7 M`, `1.0` from `9 M` on), against the resting hole's
`157`. The refinement centroid is `5.3`–`14.5` finest spacings off (without
the ramp `27`–`32`), every fit valid, every find successful, the track
within `0.40` cells. **What the ramp does not touch is the horizon's
drift**: `J = 0.846`, `M_ch = 1.030`, `M_irr = 0.9227` and the extent ratio
`0.948` at `13 M`, as without it (`0.852`, `1.030`, `0.9199`, `0.949`) —
`J` rises by `0.013` a `M` from `2 M` on in both runs and not at rest
(`0.710`). That is a second, separate failure of the moving spinning hole,
and not the layer's export: on the boosted `a = 0` hole `J` stays at
`10⁻⁵`.

## Single holes on the octant, `a = 0` to `9/10` (2026-10-02 to 2026-10-07)

From `CODE.md`'s "Robust stability on the octant", whose flat-space noise
run stays there; everything from the Kerr-Schild hole on is here. "The same
octant" in the first paragraph is that run's, `[0, 128]³` with reflecting
faces through the origin.

**The Kerr-Schild hole on the octant (measured 2026-10-02**, jobs 568189 and
568190, `ks-octant`**)**: `M = 1`, `a = 0` at the origin of the same octant,
one level fewer — `h = 1` to `1/16` in `[0, 8]³` (`R = 64, 32, 16, 8`, 9.4 M
points) — the algebraic source, the `:damped` layer at `r_0 = 3/4`,
`r_1 = 3/2` (`n_L = 12` cells, `ρ_ramp = 1`, `ρ_max = 4/M`), the Gaussian
`γ0`, `cfl = 1/2`, to `t = 128 M`: 7168 steps, `57 min` on one H200 (`26 s`
per `M`), once clean and once with the noise (`10⁻⁸`, the frozen core
excluded). Masked to `r ≥ r_1`, point-weighted:

- **The hole region is stationary.** After a transient to `t ≈ 6 M` — the
  layer saturating — `ℋ = 7.24·10⁻⁶`, `ℳ_i = 5.55·10⁻⁶`, `C_a = 1.64·10⁻⁶`
  and the error `5.56·10⁻⁶` hold to three digits at every row to `128 M`
  (late rates `≤ 4·10⁻⁶/M`); the layer residual is `0.101`; the drift of
  `h_tt` at the horizon peaks at `3.5·10⁻⁶` near `40 M` and falls to
  `2.8·10⁻⁶`.
- **The outer levels settle slowly.** The error against the exact solution
  on levels 0–3 grows from zero roughly linearly and decelerating (level 3:
  `1.0, 1.45, 1.77, 2.03, 2.27, 2.50·10⁻⁷` at `t = 48 … 128`) — the hole's
  truncation-level stationary state, a slightly different hole than the exact
  one, spreading outward at the speed of light; their constraints stay at
  `10⁻¹²–10⁻⁹`. At the outermost blocks `ℋ` rises from `4·10⁻¹³` to
  `1.3·10⁻¹²` and the error reaches `3.8·10⁻¹⁰`: nothing severe comes from
  the Dirichlet boundary in `128 M`, though the settling front reaches it at
  about that time.
- **The noise decays.** The two runs' norms agree to three or four digits,
  the noise being three orders below the truncation error; their difference,
  from paired checkpoints (`test/octant_diff.jl`), falls on every level and
  in every interval — `2.94, 2.06, 1.60, 1.23, 1.14, 1.05, 0.91,
  0.78·10⁻⁹` at `t = 16, 32, …, 128` (L∞ `1.6·10⁻⁷` to `2.3·10⁻⁸`), about
  `−0.012/M`, the hole's own level from `5.1·10⁻⁹` to `1.0·10⁻⁹`.
- **The same hole with the `:fitted` target (job 568294)**: the tracked
  geometry at `m = 8` (offset surface at `r ≈ 1.5`, `n_L = 12`),
  `lmax_shape = 4`, `lmax_fit = 8`, the finder every chunk with the spin,
  `default_bounds` gated at `9/10`, the analytic initial data down to the core
  surface (`fit_initial_depth = 3/4`; the default `0`, the fit right below the
  offset surface, starts at `ℋ = 4·10⁻²`); `61 min`. Stable to `128 M` and
  stationary from `t ≈ 8 M`, but at a higher level: `ℋ = 2.27·10⁻⁴`,
  `ℳ_i = 1.11·10⁻⁴`, `C_a = 2.28·10⁻⁵`, error `6.35·10⁻⁵` — `31×`, `20×`,
  `14×` and `11×` the `:damped` run — with the layer's residual against the
  truth `2.05` (`:damped`: `0.10`) and the fit's residual a constant
  `0.0165–0.0173`. All 129 finds succeed (`J = 1.06·10⁻⁹`, the track within
  `10⁻⁸` cells of the origin) and every fit is valid; the range projection
  never fires. **The horizon grows linearly**: `M_irr` from `0.9999984` at
  `16 M` to `1.0000090` at `128 M`, `+9.4·10⁻⁸/M` with constant increments,
  `r_min` with it — a slightly heavier hole, which the outer levels' error
  shows too (level 3 `1.35·10⁻⁶` against `:damped`'s `2.5·10⁻⁷`, about
  `δM/r` there), and the outermost blocks' `ℋ` steps from `1.2·10⁻¹²` to
  `3.5·10⁻¹²` over `112–128 M`, when that front arrives at the boundary.

**The constraints outside the horizon: convergence, depth and the layer's
parameters (measured 2026-10-02**, jobs 568343–568363 and the reruns
568719–568720, `ks-octant/out/study` and `ks-study2`**)**. Kerr-Schild
`a = 0` on the octant `[0, 64]³`, root brick `2³`, cubes `R = 32, 16, 8`, so
the finest level `[0, 8]³` has `h = 4/N`; the algebraic source, `q = 4`,
`cfl = 1/2`, `24 M`, the finder every chunk, one H200 a row
(`test/octant_runs.jl`, analysed by `test/octant_study.jl`). The norms are
point-weighted in shells about the hole: the evolved band inside the horizon
(`in`), then `[2, 2.25)`, `[2.25, 3)`, `[3, 5)`, `[5, 8)` and `r ≥ 8`; the
values below are at `24 M`, and `dM_irr/dt` is the slope over `8–24 M`.

- **`:damped` converges at order four outside the horizon, the horizon's mass
  drift included** (`r_1 = 1.5`, `r_0 = 0.75` fixed in `M`, so `m = 8, 12,
  16` and `n_L = 12, 18, 24` cells at `h = 1/16, 1/24, 1/32`):

  | `h` | `ℋ [2, 2.25)` | `ℋ [2.25, 3)` | error `[2, 2.25)` | `M_irr − 1` | `dM_irr/dt` |
  |---|---|---|---|---|---|
  | `1/16` | `2.42·10⁻⁵` | `2.33·10⁻⁶` | `1.08·10⁻⁵` | `1.45·10⁻⁶` | `2.09·10⁻⁸` |
  | `1/24` | `1.46·10⁻⁶` | `4.17·10⁻⁷` | `1.65·10⁻⁶` | `2.89·10⁻⁷` | `4.05·10⁻⁹` |
  | `1/32` | `4.30·10⁻⁷` | `1.31·10⁻⁷` | `5.24·10⁻⁷` | `8.7·10⁻⁸` | `1.27·10⁻⁹` |

  Orders: every shell from `2.25` out `4.0` for `ℋ`, `ℳ`, `C_a` and the
  error; the first shell `6.9/4.25` (`ℋ`) and `4.6/4.0` (error), a layer term
  that falls faster than the bulk; `dM_irr/dt` `4.05/4.04`. The mass drift is
  the scheme's truncation error and nothing else: at `h = 1/32`, `1.3·10⁻⁹/M`.
- **Depth helps `:damped` only to about 8–12 cells** (`h = 1/16`, ramp 12
  cells): `ℋ [2, 2.25)` is `5.5·10⁻⁵`, `2.4·10⁻⁵`, `1.1·10⁻⁵`, `1.2·10⁻⁵` at
  `m = 4, 8, 12, 16`, the shell `[2.25, 3)` is `2.1–2.7·10⁻⁶` at every depth,
  every shell from `r = 3` out is the same to two digits, and `dM_irr/dt` is
  `2.1–2.2·10⁻⁸` at `m = 8, 12, 16` (at `m = 4` the finder's footprint reaches
  the layer and the horizon is not found).
- **`:fitted` converges faster than order four at a fixed physical depth**
  (`m = 8, 12, 16`, `n_L = 12, 18, 24` at `h = 1/16, 1/24, 1/32`), because
  its layer gains cells: `ℋ [2, 2.25)` `8.5·10⁻⁴`, `3.5·10⁻⁵`, `5.8·10⁻⁶`
  (orders `7.9/6.2`), the error there `1.7·10⁻⁴`, `6.2·10⁻⁶`, `6.6·10⁻⁷`
  (`8.2/7.8`), `dM_irr/dt` `7.0·10⁻⁸`, `5.5·10⁻⁹`, `1.4·10⁻⁹`. At
  `h = 1/32` it is `:damped`'s outside the first shell — the error in
  `[2, 2.25)` `1.3×`, from `2.25` out equal — and only `ℋ` just outside the
  horizon stays `13×`; inside, in the band and the layer, it is far larger.
- **Depth helps `:fitted` exponentially, about a factor `e` per 2.5–3
  cells, up to where it fails**: at `h = 1/16`, `ℋ [2, 2.25)` is `2.1·10⁻³`,
  `8.5·10⁻⁴`, `2.0·10⁻⁴` at `m = 6, 8, 12` (the error `4.9·10⁻⁴`, `1.7·10⁻⁴`,
  `5.5·10⁻⁵`); `m = 4` loses its track (the finder's footprint), and `m = 16`
  — the core surface at `r = 0.25` — degenerates in its first chunk
  (`DomainError` in the kernel, the layer's unguarded outer part), where
  `:damped` at the same depth runs.
- **The layer's parameters** (`h = 1/16`, `m = 8`, against the rows above):
  for `:damped`, `ρ_max = 8/M` and a ramp of 18 cells each lower
  `ℋ [2, 2.25)` about `3×` (`8.6·10⁻⁶`, `6.8·10⁻⁶`) and `2/M` raises it `3×`;
  for `:fitted`, `4/M` is the best of the three (`2/M` and `8/M`: `1.4` and
  `1.6·10⁻³`), the 18-cell ramp lowers it `4×` (`2.0·10⁻⁴`), and
  **`lmax_fit = 12` is `lmax_fit = 8` to three digits** — the hole is
  spherical, so the fit's angular degree is not what limits it; for `l = 0`
  the `cont = 1` ansatz is a quadratic in `r` below the offset surface, which
  is the representation to suspect next. `ε_KO = 1/4` is worse for both, by
  `10×` for `:damped` and `3.5×` for `:fitted`.
- **The transient leaves as a pulse.** The constraint violation the start-up
  emits near the hole travels outward: in `[5, 8)` it peaks at `t ≈ 12–16`
  (`:damped` at `h = 1/16` `1.1·10⁻⁸`, `:fitted` `2.0·10⁻⁷`) and falls, in
  `r ≥ 8` it arrives at `16–20 M`; level 0 and the boundary band are flat to
  `24 M`. The positive "rates" of the outer shells over `8–24 M` are that
  pulse passing and, for the error, the slightly different hole's `δM/r`
  settling outward — not growth. **So `24 M` is long enough** for everything
  near the hole: its shells saturate by about `8 M`, and the mass drift is
  linear from there with a rate that converges at order four; what a short
  run cannot show is the pulse and the settling front meeting the outer
  boundary.

**A setup for `:fitted` (measured 2026-10-03**, jobs 569587–569594 and
569644, `ks-study2/out/study`**)**: the same octant and analysis, rows chosen
from the study above — the depth bound, a fit that also matches curvatures
(`FittedSpec`'s `fit_cont = 2`, added the same day: the state sampler's
`order = 2`, which differences the interpolated radial gradient because
TreeAMR 0.1.4 interpolates no second derivatives), and the proposed setups.
At `24 M`, against `:damped` at the same `h` (`ℋ` and the error in
`[2, 2.25)`, the mass drift):

| `h` | `m` / `n_L` / `fit_cont` | `ℋ [2, 2.25)` | error `[2, 2.25)` | `dM_irr/dt` |
|---|---|---|---|---|
| `1/16` | `:damped` | `2.4·10⁻⁵` | `1.1·10⁻⁵` | `2.1·10⁻⁸` |
| `1/16` | 8 / 12 / 1 | `8.5·10⁻⁴` | `1.7·10⁻⁴` | `7.0·10⁻⁸` |
| `1/16` | 8 / 12 / 2 | `3.2·10⁻⁴` | `8.1·10⁻⁵` | `5.4·10⁻⁹` |
| `1/16` | 12 / 14 / 1 | `7.3·10⁻⁵` | `1.9·10⁻⁵` | `1.6·10⁻⁸` |
| `1/24` | `:damped` | `1.46·10⁻⁶` | `1.65·10⁻⁶` | `4.05·10⁻⁹` |
| `1/24` | 12 / 18 / 1 | `3.5·10⁻⁵` | `6.2·10⁻⁶` | `5.5·10⁻⁹` |
| `1/24` | 12 / 18 / 2 | `1.47·10⁻⁵` | `3.3·10⁻⁶` | `3.9·10⁻⁹` |
| `1/24` | 18 / 18 / 1 | `5.4·10⁻⁶` | `2.2·10⁻⁶` | `4.3·10⁻⁹` |
| `1/24` | 24 / 12 / 1 | `7.3·10⁻⁶` | `3.4·10⁻⁶` | `4.2·10⁻⁹` |
| `1/24` | 16 / 20 / 1 | `8.3·10⁻⁶` | `2.2·10⁻⁶` | `3.7·10⁻⁹` |
| **`1/24`** | **16 / 20 / 2** | **`2.3·10⁻⁶`** | **`1.8·10⁻⁶`** | **`4.0·10⁻⁹`** |
| `1/32` | `:damped` | `4.3·10⁻⁷` | `5.2·10⁻⁷` | `1.3·10⁻⁹` |
| `1/32` | 16 / 24 / 1 | `5.8·10⁻⁶` | `6.6·10⁻⁷` | `1.4·10⁻⁹` |
| `1/32` | 20 / 24 / 1 | `1.1·10⁻⁶` | `4.8·10⁻⁷` | `1.26·10⁻⁹` |

The last row's first attempt ended at `18 M` in a node failure (both H200
nodes down); it was rerun with checkpoints (job 569644), and its shells at
`18 M` were its shells at `24 M` to two digits. From `r = 2.25` out every row
is `:damped`'s to two digits.

- **The layer that failed at `h = 1/16` runs at `1/24`**: the offset surface
  at `r = 1.0` with the core surface at `0.5` (`m = 24`, `n_L = 12`) is stable
  to `24 M`, so the bound is cells, not the radius. Past 16–18 cells more
  margin does not help (`m = 24` is worse than `m = 18`), a thicker ramp does.
- **`fit_cont = 2` wins every pair**, `2.3–3.6×` in `ℋ` just outside the
  horizon and `1.2–2×` in the error, and at `h = 1/16` it takes the mass
  drift from `7.0·10⁻⁸` to `5.4·10⁻⁹`. Its fit residual is `15–17×`
  `cont = 1`'s and flat in time (`0.017` against `0.0011` for the `1/24`,
  16 / 20 rows): the ansatz cannot match values, slopes and curvatures at
  once — a representation limit, two to three orders above the sampled
  curvature's own error (`3·10⁻⁵` relative at `1/24`), so interpolating the
  second derivative directly would not change these rows.
- **The setup: `h = 1/24` at the hole, `m = 16`, `n_L = 20`,
  `fit_cont = 2`**, `ρ_max = 4/M`, `ε_KO = 1/2`, `lmax_fit = 8` (12 for a
  spinning hole), `fit_initial_depth = n_L h`: `ℋ` just outside the horizon
  `1.6×` `:damped`'s, the error `1.1×`, the mass drift the same, and everything
  from `r = 2.25` out equal. At `h = 1/32`, `m = 20` with `cont = 1` already
  matches `:damped`: the error `0.92×`, the mass drift to 1 %, `ℋ` just
  outside the horizon `2.6×`.

**Proposed next (2026-10-04): a spinning hole on a bitant.** Written for the
session that picks this up; nothing of it is built. **(Amended 2026-10-04:
the static spinning hole runs on the *rotating octant* instead —
`hole_case(; octant = :rotating)`, TreeAMR 0.1.7's quarter-turn seam with
the mirror at `z = 0`, a quarter of the bitant; see "The rotating octant".
The bitant remains the domain of a hole that moves in the plane.)**

- **A bitant, not an octant.** A spin along `z` keeps only the `z → −z`
  mirror, and so does a boost in the `x`–`y` plane: the domain is
  `[−L, L]² × [0, L]`, reflecting at `z = 0` only — four times the octant —
  and G5's moving spinning hole fits it too. `state_parity` is already right
  (Kerr-Schild's `g_tz` is odd in `z`). Needed: a `bitant` option to
  `hole_case` that accepts a spin along `z` and a velocity and center in the
  plane and refuses anything else, and a non-cubic root brick (`(2, 2, 1)`)
  in `gh_forest`/`hole_forest`, which today refuse unequal box widths.
- **The source is ready.** `KerrSchildSource(; spin = (0, 0, a))` is exact at
  `a = 9/10` to `2.5·10⁻¹⁵` and lifts the refusal of a moving Kerr-Schild
  hole. Its `M`, `S` and `u` are constants of the run, so a drift of the
  evolved hole's parameters is a drift of the coordinates without a
  constraint violation: record `M_irr` and `J` from the start.
- **The interior is tighter.** At `a = 9/10` the horizon's smallest radius is
  `1.436` (poles) and the singular ring has radius `0.9`. Step 5's sphere needs
  `r_0 > 0.9` and `r_1 ≤ 1.436 − m h`, so a 12-cell ramp wants `h ≈ 1/48`
  rather than `1/24` (`check_interior_radii` will say). The tracked shape is
  oblate, but with `fit_initial_depth = n_L h` the analytic initial data must
  stay outside the ring, which bounds `m + n_L` at the equator: the `a = 0`
  recipe above does not carry over, and `lmax_fit` matters now (step 8f used
  12). Step 8f's `ks9` rows (`q = 2`) had `:fitted` `40×` `:damped`; whether
  `fit_cont = 2` and depth repair that is the first measurement.
- **An order.** `:damped` at `a = 9/10`, `h = 1/48`, `24 M`, with a
  convergence pair, as the reference; `:fitted` on the same mesh, depth and
  `fit_cont` as above; then whether a static spinning hole's `J` drifts, as
  G5's moving one does. `test/octant_runs.jl` and its analysis scripts carry
  over once they take the bitant. TreeAMR `main`'s second-derivative
  interpolation can replace the state sampler's differenced gradient when it
  is released; nothing here needs it sooner.

**A spinning hole on the rotating octant, `a = 3/5` (measured 2026-10-04**,
jobs 569778 and 569779, `spin-a06`**)**. Erik's first spinning reference: not
`a = 9/10` at `h ≈ 1/48`, but `a = 3/5` at `h = 1/32`, `:damped` and
`:fitted` side by side, everything else the `a = 0` rows `dA128` and `fR32`
— the octant `[0, 64]³`, cubes `32, 16, 8`, `N = 128` (60.8 M points),
`q = 4`, `cfl = 1/2`, the algebraic source, `24 M`, the finder with the spin
every chunk — on the rotating octant (`test/octant_runs.jl octant=rotating
a=3/5`). Kerr-Schild `a = 3/5` has `r₊ = 1.8` at the poles, `√(2 r₊) = 1.897`
on the equator, and its ring at `0.6`. Two settings follow from the ring:

- `:damped` keeps `r_0 = 3/4`, `r_1 = 3/2` — the ring inside `r_0`, `r_1`
  9.6 cells inside the horizon at the poles.
- `:fitted` keeps `m = 20` but takes `n_L = 18`, not `fR32`'s 24: the
  analytic initial data fill the layer down to the core surface, and
  `m + n_L = 44` cells below the equator's `1.897` is `0.52`, inside the
  ring; at 38 cells the core surface is `0.71` there. `lmax_fit = 12` (the
  spinning hole's, above), `fit_cont = 1`.

Both run to `24 M` in `2 h 35` on one H200 each. At `24 M`, point-weighted
L2 in the shells of `test/octant_runs.jl` (the first from the equatorial
radius; the band `in` is the evolved region inside it):

| | `ℋ` in | `ℋ [1.90, 2.25)` | `ℋ [2.25, 3)` | error in | error `[1.90, 2.25)` | error `[2.25, 3)` |
|---|---|---|---|---|---|---|
| `:damped` | `3.95·10⁻⁶` | `6.41·10⁻⁷` | `1.55·10⁻⁷` | `5.0·10⁻⁶` | `1.35·10⁻⁶` | `4.75·10⁻⁷` |
| `:fitted` | `1.43·10⁻³` | `2.04·10⁻⁶` | `1.39·10⁻⁷` | `2.7·10⁻⁴` | `1.41·10⁻⁶` | `4.41·10⁻⁷` |
| `dA128` (`a = 0`) | `1.77·10⁻⁶` | `4.30·10⁻⁷` | `1.31·10⁻⁷` | `2.2·10⁻⁶` | `5.24·10⁻⁷` | `1.89·10⁻⁷` |
| `fR32` (`a = 0`) | `1.65·10⁻⁴` | `1.11·10⁻⁶` | `1.34·10⁻⁷` | `1.7·10⁻⁵` | `4.82·10⁻⁷` | `1.89·10⁻⁷` |

- **The spin changes nothing qualitative outside the horizon.** Both runs'
  constraints are stationary from `t ≈ 6–8 M` to `64 M` (both rows were
  continued from their `24 M` checkpoints, jobs 569811 and 569812): `ℋ` just
  outside the horizon `6.41 → 6.47·10⁻⁷` (`:damped`) and `2.04·10⁻⁶`
  throughout (`:fitted`, `3.2×` — `a = 0`: `2.6×`), and from `2.25` out the
  two agree to two digits. Every find succeeds (65 of 65), every fit is
  valid, the projection never fires, and `:fitted`'s band inside the horizon
  and its fit residual (`6.9·10⁻³`) are constant from `t = 2 M`. The spinning
  hole's own error is `2–2.6×` the `a = 0` hole's from `2.25` to `8`, and
  `ℋ` just outside the horizon `1.5×`.
- **The spin drifts, linearly, as truncation error at order four.**
  `:damped`'s `J` rises from `a + 8·10⁻⁹` at a constant rate to `64 M` —
  `5.42`, `5.11`, `5.16·10⁻⁸/M` over `8–24`, `24–40`, `40–64 M`
  (`+3.6·10⁻⁶` at `64 M`) — and `M_irr` falls with it, about `−0.2 δJ` at
  constant mass (`−2.7·10⁻⁷` at `64 M`). The convergence rows `h = 1/16` and
  `1/24` (`N = 64, 96`, the same radii in `M`, `margin = 4` for the check;
  jobs 569813 and 569814) give the same constant rates, `1.1–1.4·10⁻⁶` and
  `1.6–1.75·10⁻⁷`, and orders `3.99–4.07` from `1/24` to `1/32` in every
  interval, `4.16` for `J − a` at `24 M` and `4.02` for `dM_irr/dt`; every
  shell's `ℋ` and error from `2.25` out converge at `3.98–4.17`. `1/16` is not
  in the asymptotic regime (orders `4.6–8.7` from it, and `145×` the `1/32`
  `ℋ` just outside the horizon by `8 M`): `r_1 = 3/2` is only 4.8 of its cells
  inside the horizon at the poles. The drift is the scheme's, as `a = 0`'s
  mass drift is (`1.3·10⁻⁹/M`); it is four orders below G5's moving hole's.
  `:fitted`'s `J` wanders by `±2·10⁻⁷` to `15 M` and then drifts at
  `4.1·10⁻⁸/M`, `0.8×` `:damped`'s.
- **The error grows with the drift, so the variants' error ratio is the ratio
  of their drifts.** The error outside the horizon grows linearly in every
  run, the slightly different hole spreading outward, at order four: in the
  first shell `:fitted`/`:damped` goes `1.65×` (`8 M`), `1.04×` (`24 M`),
  `0.88×` (`40 M`), `0.81×` (`64 M`), toward the drifts' `0.80`. So a
  comparison of the two by their error must name its time; by `ℋ` it need not.
- **`:fitted` is worse inside than at `a = 0`.** The evolved band inside
  the horizon is `9×` `fR32`'s `ℋ`, and the fit's residual `9×` `fR32`'s
  `7.5·10⁻⁴` — the shorter ramp, the band reaching deeper (from `1.27`
  rather than `1.375`) and an oblate hole in the `cont = 1` ansatz, not
  separated here.
- **The far field settles by `64 M`**: the start-up pulse passes `r ≥ 8` at
  about `24 M` for `:fitted` (`ℋ 1.2·10⁻¹⁰`, then `1.0·10⁻¹⁰`) and both rows
  end at `1.0–1.3·10⁻¹⁰` there, three orders below the first shell.

**`a = 9/10` at `h = 1/48` (measured 2026-10-04**, jobs 569826, 569855,
569856, `spin-a06`**)**. Kerr-Schild `a = 9/10` has `r₊ = 1.436` at the poles,
`1.695` on the equator and its ring at `0.9`, which leaves `0.79` below the
equator for a `:fitted` margin and ramp — 25 cells at `h = 1/32` — so the rows
run at `h = 1/48` on one more cube (`32, 16, 8, 4`, the finest level
`[0, 4]³`, `N = 96`, 31.9 M points, `2 h` for `24 M` on an H200).

- **`:fitted` at `m = 16`, `n_L = 18` (the core surface at `0.986` on the
  equator, 4 cells outside the ring), `lmax_fit = 12` is unstable, at either
  `fit_cont`.** With `fit_cont = 2` the band inside the horizon grows `2–4×`
  per `M` from `t = 0` (`ℋ` `3.9·10⁻³` at `1 M`, `0.65` at `7 M`), the fit's
  residual with it (`0.034 → 0.46`), and it crosses the horizon — the first
  shell outside `1600×` in `6 M` — and ends in a `DomainError` in the kernel
  at `7.5 M`; every fit stays valid, every find succeeds and the projection
  never fires. `fit_cont = 1` grows the same way (`ℋ 0.14`, L∞ `178` at `8 M`;
  cancelled). So the curvature fit is not the cause.
- **`:damped` runs** (`r_0 = 19/20`, the ring inside the core; `r_1 = 5/4`,
  8.9 cells inside the poles' horizon, a 14-cell ramp), stationary from `8 M`
  to `24 M` — but at a high level: `ℋ` `3.7·10⁻³` in the band inside the
  horizon and `2.4·10⁻⁵` just outside it (`37×` the `a = 3/5` hole's at
  `h = 1/32`), the error there `5.4·10⁻⁶` (`4×`), the layer's residual against
  the truth `5.2` (`a = 3/5`: `0.15`). `J` starts `7·10⁻⁶` above `a` (the
  finder at this resolution) and drifts at `7.8·10⁻⁸/M`; all 25 finds succeed.
  The layer next to the ring is under-resolved at `1/48`: on the equator it
  spans `0.95–1.25`, where Kerr-Schild's `H` falls from `3.3` to `1.3`.
- **Nothing at `1/48` rescues `:fitted`** (2026-10-05, half-`M` screens on
  `[0, 4]³` and jobs 570123, 570124): the band's `ℋ` scales with `ρ_max`
  (`2/M`: `0.55×`, `8/M`: `2×`) and a thicker ramp at the same depth
  (`m = 12`, `n_L = 22`) lowers it `2.6×`, but both rows run away outside the
  horizon by `6–8 M`; the layer pulled away from the ring (`m = 10`,
  `n_L = 14`) and the analytic data stopped 6 cells below the offset surface
  (`fit_depth = 1/8`) are worse from the start.

**`a = 9/10` at `h = 1/96` (measured 2026-10-05**, jobs 570115 and 570116**)**:
one more level, the finest box `[0, 3]³` so that the whole hole is at one
resolution — `roots = 4`, `N = 48` (the box needs one-unit blocks one level
up), cubes `32, 16, 8, 4, 3`, 477 blocks, 52.8 M points, `dt = 1/333`,
about `15 min` per `M` on an H200.

- **`:damped` is clean** (`r_0 = 19/20`, `r_1 = 5/4`, `a09d48`'s radii in
  `M`): stationary to `24 M`, `ℋ` `1.3·10⁻⁶` overall, `8·10⁻⁶` in the band
  inside the horizon and `2.9·10⁻⁸` just outside it — `1000×` below `1/48`,
  far more than the scheme's order, so `1/48` was outside the convergent
  regime next to the ring.
- **`:fitted` (`m = 24`, `n_L = 36`, the core surface 16 cells outside the
  ring, `fit_cont = 1`, `lmax_fit = 12`) holds for `18 M` and then grows on
  the axis.** Its band inside the horizon saturates at `ℋ ≈ 2.3·10⁻³` and the
  first shell outside at `8.3·10⁻⁶` (`280×` `:damped`, a third of `:damped`
  at `1/48`) by `7 M`; from `t ≈ 18 M` the gauge constraint and then `ℋ` grow
  at the first evolved points on the `z` axis — `ℋ`'s L∞ `0.03`, `0.07`,
  `0.82`, `2.95` at `20, 22, 23, 24 M` — while every shell, the horizon
  (`J = 0.900007`, `M_irr` at Kerr's to `2·10⁻⁸`) and the far field are
  unchanged. From the `22` and `24 M` checkpoints, with both constraints
  evaluated everywhere and binned by depth below the horizon and polar angle:
  at `22 M` the largest `|C_a|` (`2.6·10⁻³`) is a ring `0.15` off the axis
  at the offset surface (`|cos θ| = 0.991`, depth 24 cells); at `24 M` it is
  on the axis there (`x = y = 0`, `z = 1.19`, `|C| = 0.097`, `ℋ = 2.95`),
  `37×` larger, and every bin with `|cos θ| < 0.75` is unchanged. **The
  seam's interpolation is not the cause**: the exact state's value and
  gradient interpolated beyond `x = 0`, beyond `y = 0` and beyond both, near
  the pole and around the azimuth, have the same error as inside the
  quadrant to three digits (`2.3·10⁻⁶`, `2.3·10⁻⁴` at `h = 1/24`). Left
  open: the fit's or the tracked shape's representation at the pole of an
  oblate `a = 9/10` horizon (`a = 3/5` `:fitted` ran to `64 M` on the same
  octant), against the `:fitted` layer's outer edge on the axis.

**The axis instability narrowed down (2026-10-05/06**, jobs 570249–570520,
`spin-sync/` — a scratch copy of the package and of TreeAMR with experimental
switches**)**. Each row is run A's mesh at `h = 1/96`, changed in one thing:

| row | change | result |
|---|---|---|
| A2 | the fit's collocation closed under the quarter turn | A to 7 digits, the onset at `17–18 M` |
| A3 | the two seam planes made identical after every stage (a TreeAMR `sync_seam!`), the axis projected | A to 7 digits |
| A4 | both | A to 7 digits |
| A5 | the target frozen at the initial fit, no refits | the same onset at `18 M` |
| A6 | A's tracked geometry with the exact target (`variant = :damped`) | grows on the axis at the offset surface from `1 M` |
| G5 | as A6, `n_L = 24` (core surface at the pole `0.94`) | grows in the band inside the horizon from `1 M` |
| G4 | a spherical `:damped` layer with its edge at A's polar radius (`r_1 = 19/16`) | stable (noisy next to the ring) |
| G2 | as A6, `m = 12` (edge `1.31` at the pole) | clean to `6 M`: `ℋ` `3.7·10⁻⁸` just outside the horizon |
| A7 | `:fitted`, `m = 12`, `n_L = 36` | stable to `24 M`, no axis growth |

- **Not the symmetry.** The fit was asymmetric — `EquiangularGrid(12)` has 25
  longitudes, so the evolved state's grid-induced azimuthal content at
  `m = 24, 28` aliases into `m = 1, 3`, and the fit of A's state broke the
  quarter turn at `1.2·10⁻⁴` (the exact, axisymmetric data's fit: `5·10⁻¹⁴`);
  closing the collocation set under the turn (`fit_directions(L; turns = 4)`,
  automatic for a state on a rotating forest) brings it to `5·10⁻¹¹`. And A's
  two seam planes, which must be each other's images, had drifted apart by
  `O(1)` near the axis by `22 M`. But forcing either symmetry, or both, leaves
  the growth unchanged: it is a symmetric mode, and the drift its consequence.
- **Not the fit.** A frozen target grows the same way, and the exact target on
  A's geometry grows sooner.
- **The tracked, oblate layer `24` cells (`0.25 M`) below the horizon is
  unstable at `a = 9/10`; `12` cells (`0.125 M`) is not.** A sphere with the
  same polar edge is stable, so it is neither the polar radius nor the depth
  of the core surface alone. Why `a = 9/10` is sensitive is open; one
  suspect is the near-null inward speed of the outgoing characteristic between
  the horizons, `|Δ|/(r² + a² + 2Mr) ≈ 0.03` at `r ≈ 1.2` (`a = 0`, `r = 1.5`:
  `0.14`), which makes the layer's edge barely an outflow boundary.
- **But `:fitted` at `m = 12` leaks.** A7 is stationary from `10 M` and stable
  to `24 M`, but its fit target's mismatch reaches across a horizon only 12
  cells away: `ℋ` `1.85·10⁻⁴` just outside it (B: `4.2·10⁻⁸`), the error there
  `4.7·10⁻⁵` (B: `1.1·10⁻⁷`), and `J` drifts at `−4.4·10⁻⁷/M` (B:
  `2.5·10⁻⁹/M`, `31×` below `1/48`'s `7.8·10⁻⁸`, order five). The tracked
  `:damped` layer at `m = 12` (G2) has none of this. So at `a = 9/10`
  `:fitted` has no margin that works so far: `24` cells is unstable on the
  axis, `12` leaks; whether one in between does is the next screen.
  (It does — `16` to `20`, below; amended 2026-10-08.)

**The margin screen, `m = 16–20` (measured 2026-10-06/07**, jobs 570817–570822
and 570861/2, `spin-sync/` with its experimental switches off — the committed
`e1bb8eb`'s code**)**. A's mesh at `h = 1/96`, `n_L = 36`, `lmax_fit = 12`,
`fit_cont = 1`; each margin once as `:fitted` to `32 M` (`8 h` on an H200) and
once as the same tracked geometry with the exact target (`variant = :damped`,
as G2 and A6) to `6 M`, `18` and `20` continued to `12 M`. "Outside" is the
first shell, `1.69 ≤ r < 2.25`, at the run's last time; `J` is read against B's
`0.9000069` (the finder at this resolution). `octant_runs.jl`'s band inside the
horizon starts at the equatorial radius less `m h` and so misses the evolved
points near the poles, where A's mode lives: the global L∞ and the location
script (`locate.jl` in `spin-a06-cpu/out/study`, which bins the finest level by
depth and polar angle) are what see it.

| `m` | the geometry, exact target | `:fitted`: `ℋ` outside | its error | `J − J_B` | `dJ/dt` from `16 M` |
|---|---|---|---|---|---|
| 12 (G2, A7) | clean | `1.85·10⁻⁴` | `4.7·10⁻⁵` | `−2.8·10⁻⁵` | `−1.6·10⁻⁷/M` |
| 16 | clean, `ℋ∞` `4.7·10⁻⁶` | `8.8·10⁻⁵` | `9.9·10⁻⁶` | `+3.3·10⁻⁶` | `−6·10⁻⁹/M` |
| 18 | clean, `4.4·10⁻⁶` to `12 M` | `4.9·10⁻⁵` | `7.8·10⁻⁶` | `−3.9·10⁻⁶` | `−2·10⁻⁹/M` |
| 20 | a bounded ring at the pole, `6.4·10⁻⁵` | `2.5·10⁻⁵` | `5.0·10⁻⁶` | `+2.5·10⁻⁶` | `+9·10⁻⁹/M` |
| 24 (A6, A) | grows `10×` per `M` from `1 M` | `8.3·10⁻⁶`, the axis from `18 M` | `1.3·10⁻⁶` | `+2·10⁻⁷` | — |
| B (sphere) | — | `4.2·10⁻⁸` | `1.1·10⁻⁷` | `0` | `2.4·10⁻⁹/M` |

- **All three `:fitted` rows are stable to `32 M`**, stationary in every norm
  from `12–16 M`: through A's onset at `18 M` the gauge constraint's L∞ stays
  at `4.4–4.7·10⁻⁴` (A's rose from `3.65·10⁻⁴` at `16 M`), every find
  succeeds, and `M_irr` is B's to `10⁻⁷`. Located from the `12` and `32 M`
  checkpoints, their largest violation is on the equator at the offset
  surface — the target's mismatch, `|C_a| ≈ 6·10⁻⁴`, as in A7 — and near the
  pole (`|cos θ| ≥ 0.9`) at that depth it is a quarter of that (`|C_a|`
  `1.6`, `1.4`, `1.2·10⁻⁴` for `m = 16, 18, 20`), every bin the same at `12`
  and `32 M` to two digits; A's precursor, a ring just off the axis, never
  appears.
- **The geometry's polar mode switches on between `18` and `20` cells.** With
  the exact target `m = 16` and `18` are G2's clean layer (near the pole at the
  offset surface `ℋ` `1.4–1.6·10⁻⁷`, `|C_a|` `0.7–1.1·10⁻⁸`, the largest
  violation the ordinary one on the equator), while `m = 20` grows a ring
  `0.11` off the axis at the offset surface (`z = 1.22`, `|cos θ| = 0.995`) —
  A's precursor at `22 M` — to `ℋ` `6.3·10⁻⁵` by `6 M`, `400×` its
  neighbours', and then holds it there to `12 M`; A6 at `24` grows without
  bound. In the `:fitted` row at `m = 20` that ring is far below the target's
  own mismatch in the same bin (`ℋ` `5.7·10⁻³`), and nothing grows to `32 M`.
- **The leak falls `1.8–2×` for every two cells of margin from `16` to `20`**
  (`2.1×` from `12` to `16`, `3×` from `20` to `24`), and `J` stops
  drifting: after an offset of `±3·10⁻⁶` set in the first `8 M` (not
  monotonic in `m`), it is flat to `|dJ/dt| ≤ 9·10⁻⁹/M` from `16 M`, where
  A7's still drifts at `−1.6·10⁻⁷/M`. So at `a = 9/10`, `h = 1/96`, `:fitted`
  works at `m = 16–20` — `18` is the deepest margin whose geometry is clean —
  at a price outside the horizon that no margin here removes: `ℋ` `600–2100×`
  B's and the error `50–90×`.

## Excision, steps X1–X7 (2026-10-05 to 2026-10-08)

From `CODE.md`'s "Measured results", moved here on 2026-10-08 when `main` was
merged into the excision round's integration branch, verbatim but for the
links; X4's entry, which stood first there (beside the H200 study it merged),
is in the order of the steps here. `CODE.md` keeps the design — [Excision
(added 2026-10-05)](CODE.md#excision-added-2026-10-05), with each step's "What
step X… built" and "… measured" — and the stencil provider under [One
right-hand-side evaluation](CODE.md#one-right-hand-side-evaluation); its
"Single black holes: recommended settings" has the excised rows. What the merge
itself measured is in `CODE.md` too, under "One right-hand-side evaluation".

### Excision: the analysis (step X1)

Host-side, no kernel change: the closure weights in `src/stencils.jl` (under
[Excision](CODE.md#excision-added-2026-10-05), "The closures as built") and the
script `test/excision_model.jl`, whose three sections are below. Every model
is a sparse matrix assembled from the package's own closure weights and the
background's coefficients, read through `background_state`,
`metric_quantities` and `metric_derivatives` as the kernel reads them; its
self-checks throw (Kerr-Schild's closed form, `dispersion.jl`'s recorded
layer numbers to their printed digits, the parity sectors against the full
operator to `10⁻¹²`). Measured on the development machine (Apple silicon, 12
CPU threads, Julia 1.13.1), shared with the suite and other work at a load
of 10–40.

**Margins** (`excision_model.jl margins`, 20 s). The seed's offset surfaces
`r_E(n̂) = r_h(n̂) − m h` at the octant runs' `h = 1/16 … 1/48`; *normal* is
the least `b_n/a_n − 1` over the surface along its true normal (outflow
`> 0`); *b/a* the per-axis ratio over the lego surface's closure faces (a
face is an evolved point with an excised axis neighbour, `b = −s β^d` toward
the excised side `s`, `a = α√γ^{dd}`), its least value and 1 % quantile;
*inflow* the fraction of faces with `b/a < 1`, and *`b/a < 0`* of those
whose shift points into the excised set; *clearance* the least distance
from the surface to the chart's singular disk. Representative
rows at `h = 1/32` (the full table has every `h` and `m = 1 … 48`):

| chart | `m` | `r_E` | normal | faces | `b/a` min, 1 % | inflow | `b/a < 0` | `r_E/(2M)` | clearance (cells) |
|---|---|---|---|---|---|---|---|---|---|
| KS `a = 0` | 4 | 1.875 | `+0.067` | 67 662 | `0.037`, `0.099` | 0.947 | 0 | 0.938 | — |
| | 16 | 1.500 | `+0.333` | 43 254 | `0.055`, `0.110` | 0.752 | 0 | 0.750 | — |
| | 32 | 1.000 | `+1.000` | 19 230 | `0.072`, `0.145` | 0.495 | 0 | 0.500 | — |
| KS `a = 3/5` | 8 | 1.55–1.65 | `+0.121` | 50 342 | `−0.286`, `−0.079` | 0.827 | 0.021 | — | 33.5 |
| | 32 | 0.80–0.90 | `+0.456` | 14 446 | `−0.866`, `−0.494` | 0.529 | 0.071 | — | 9.5 |
| KS `a = 9/10` | 8 | 1.19–1.45 | `+0.065` | 35 502 | `−0.676`, `−0.335` | 0.785 | 0.063 | — | 17.4 |
| | 16 | 0.94–1.20 | `+0.065` | 23 638 | `−1.183`, `−0.594` | 0.734 | 0.092 | — | 9.4 |
| | 24 | 0.69–0.95 | **`−0.250`** | 14 182 | `−4.117`, `−1.366` | 0.769 | 0.148 | — | 1.4 |
| harmonic `a = 7/10` | 4 | 0.59–0.88 | `+0.136` | 11 550 | `−1.150`, `−0.887` | 0.659 | 0.138 | — | 5.6 |
| | 8 | 0.46–0.75 | `+0.298` | 8 134 | `−1.668`, `−1.300` | 0.459 | 0.172 | — | 1.6 |
| harmonic `a = 9/10` | 1 | 0.41–0.97 | `+0.027` | 11 054 | `−1.395`, `−1.183` | 0.816 | 0.173 | — | 2.2 |
| | 2 | 0.37–0.94 | `+0.049` | 10 198 | `−1.503`, `−1.290` | 0.762 | 0.176 | — | 1.2 |

Four findings **(measured in step X1)**:

- **The lego staircase's inflow-like fraction is `r_E/(2M)`**, as predicted
  for Kerr-Schild `a = 0`: `0.947, 0.880, 0.752, 0.622, 0.495` against
  `0.938, 0.875, 0.750, 0.625, 0.500` at `h = 1/32`, within `0.03` at every
  `h` and depth, and the least per-axis ratio is near zero (`0.008–0.45`):
  some face is always nearly tangent.
- **A spinning hole's lego surface has faces where the shift points *into*
  the excised set along the axis, `b/a < 0`** — frame dragging gives `β` an
  azimuthal part — down to `−0.4` (`a = 3/5`) and `−1.2` (`a = 9/10`)
  within half an `M` of the horizon (below `−2` near the inner horizons),
  and below `−1` (both characteristics entering) on harmonic Kerr. Kerr-Schild `a = 0` has none (`β` is radial, so
  `b = β^r cos θ ≥ 0`), and none of the models below covers them except
  the frozen line, which finds them unstable.
- **Normal outflow holds everywhere between the horizon and Kerr-Schild's
  inner horizon** — the spheroid `R = r₋`, at the equator `√(r₋² + a²)` —
  and fails below it: the margin is positive from `m = 1` cell at every `h`
  in every chart down to where it ends, `2M/r_E − 1` for `a = 0`; for `a = 9/10` it peaks at
  `+0.079` half an `M` down and is `−0.058` at `0.667 M` and `−0.25` at
  `0.75 M`, the inner horizon being `0.633 M` below the outer one at the
  equator (`1.265 M` at `a = 3/5`). So **Kerr-Schild `a = 9/10` has room:
  a depth window of `0.63 M` at the equator, which the ring (`0.9`) does
  not bind** (the ring is inside the inner horizon, `1.062`); it is
  `≤ 9` cells at `h = 1/16` and `≤ 28` at `1/48`. The harmonic chart ends
  at its disk before the inner horizon: `0.3 M` at `a = 7/10` (a surface
  with three cells' clearance from `h = 1/24` on, at `m ≤ 8` at `1/48`)
  and `0.1 M` at `a = 9/10` — `m = 1, 2` at `h = 1/48` only, where `m = 4`
  already fails normal outflow (`−0.032`).
- **The answer per chart**, the shallowest surface with normal outflow and
  the deepest scanned that has it and clears the disk by three cells: `m = 1`
  and any depth to `r_E ≈ M/2` (Kerr-Schild `a = 0`); `m = 1` and `1.0 M`
  (`a = 3/5`); `m = 1` and `0.5 M` (`a = 9/10`); `m = 1` and `0.17 M`
  (harmonic `a = 7/10`, from `h = 1/24`); `m = 1` at `h = 1/48` only
  (harmonic `a = 9/10`).

**The frozen line** (`model1d=frozen`, 20 s): `dispersion.jl`'s
constant-coefficient system, `a = 1/2`, on 200 points closed at the left,
Dirichlet ghosts on the right. Entries are the dense spectrum's largest
`Re λ` in `1/h` away from zero (`< 0` to `10⁻¹⁰`, `(b)` for a mode with
half its norm on the first ten points), then `0` for a zero mode and `0²`
for a Jordan block at zero (a mode growing linearly in time), and the RK4
step the spectrum allows as `dt (a + |b|)/h`; at `ε_KO = 1/2`, the default
closure (reach `G`, `:msn`):

| `b/a` | `q = 2` | cfl | `q = 4` | cfl | Dirichlet ghosts' cfl, `q = 2, 4` | lopsided `q = 2, 4` | its cfl |
|---|---|---|---|---|---|---|---|
| `−1.25` | **`+1.2e−1` (b)**, `0²` | 2.38 | **`+1.9e−1` (b)**, `0` | 1.90 | 2.38, 1.90 | **`+6.8e−2`**, **`+1.6e−1`** | 1.08, 1.26 |
| `−0.5` | **`+3.2e−2` (b)**, `0` | 1.78 | **`+5.0e−2` (b)**, `0²` | 1.61 | 1.78, 1.61 | **`+3.1e−2`**, **`+4.8e−2`** | 1.17, 1.21 |
| `0` | `< 0`, `0²` | 1.19 | `< 0`, `0²` | 1.07 | 1.19, 1.07 | `< 0`, `< 0` | 1.19, 1.07 |
| `0.5` | `< 0`, `0` | 1.80 | `< 0`, `0²` | 1.64 | 1.80, 1.65 | `< 0`, `< 0` | 1.23, 1.27 |
| `0.95` | `< 0`, `0` | 2.28 | `< 0`, `0²` | 1.88 | 2.28, 1.88 | `< 0`, `< 0` | 1.22, 1.41 |
| `1.05` | `< 0` | 2.35 | `< 0` | 1.89 | 2.35, 1.89 | `< 0`, `< 0` | 1.20, 1.44 |
| `2` | `< 0` | 2.56 | `< 0` | 1.96 | 2.56, 1.96 | `< 0`, `< 0` | 1.13, 1.50 |

- **Outflow (`b/a > 1`) is stable** at every `q`, `ε_KO ∈ {0, 1/4, 1/2,
  1}`, reach (`G − 1 … G + 2`) and dissipation closure, and so is a
  Dirichlet-ghost left end.
- **An inflow-like face with the shift still pointing out of the excised
  set (`0 ≤ b/a < 1`) is marginal, not unstable**: no eigenvalue to the
  right of zero, and a zero mode — the static `u = c(x − L)` an inflow
  boundary without data admits — that is a Jordan block at `q = 4` under
  `:msn` (linear growth; a simple zero under the reduced rank and the
  one-sided closure) and at `ε_KO = 0` for `q = 2`.
- **A face with the shift pointing into the excised set (`b/a < 0`) is
  unstable**, with a mode on the closure growing at `0.03–0.19/h` — at
  every `ε_KO` from `1/4` to `1` under `:msn`, under the other two
  closures at `ε_KO = 1/2`, at every reach but one row (reach 4 at
  `q = 2`, `b/a = −1.25`), and with the lopsided advection (whose upwind
  side is then the excised one, so it falls back to the closure); only the
  reduced rank and the one-sided closure at `ε_KO = 1` escape some rows.
  That is what the spinning charts' lego surfaces have, above.
- **The closures do not shorten RK4's step**: the closed line allows what
  the Dirichlet-ghost line and the open line allow, to three digits. The
  lopsided advection does, from `1.8–2.6` to `1.1–1.5` here — at the
  octant runs' `cfl = 1/2` a margin of two.

**What the closure reflects** (`model1d=reflect`, 30 s). A pure discrete
branch-1 packet (the fast ingoing branch, projected in Fourier space) into
the closure; `R` the largest `|u|` off the closure's points after it has
arrived, against its own. At `ε_KO = 0` the closure turns `0.03–1.4 %` of a
smooth packet (`λ = 16h`) and `1–25 %` of a `λ = 8h` one into grid-scale
outgoing content, carried at the open line's branch-2 group velocity of its
phase (`θ = 0.55–0.97 π`; at `q = 2`, `b/a = 0.8`, measured `+0.386` against
`+0.389` cells per unit time). At `ε_KO = 1/2` that content is damped within
cells and `R` is `0.6–4 × 10⁻⁴` (`λ = 16h`) and `0.9–83 × 10⁻⁴` (`8h`).
With the lopsided advection nothing comes back: `R ≤ 5 × 10⁻⁵`, and what
remains moves inward (the packet's own mismatch with the lopsided branch,
`v_g < 0`).

**Into the surface and out through the horizon** (`model1d=radial`, 20 s).
`dispersion.jl`'s radial line of Kerr-Schild `a = 0` at `h = 5/64`, `ε_KO =
1/2`; the largest `A_0/A` in the first shell outside the horizon, over
`λ = 2h, 4h, 8h` for the ripple at depth `d`, and for a Gaussian pulse
(half-width two cells) halfway between the surface and the horizon; and
the line's spectrum at `h = 5/64, 5/128, 5/256`:

| interior | `q` | ripple `d = 4`: `2 M`, `10 M` | `d = 8`: `2 M`, `10 M` | pulse: `2 M`, `10 M` | largest `Re λ` (`1/M`) |
|---|---|---|---|---|---|
| the layer (`r_1 = 1.15`, control) | 2 | `9.2e−2`, `0.136` | `8.9e−3`, `5.8e−2` | `4.6e−2`, `0.112` | `−0.08`, `−0.15`, `−0.13` |
| excised at `r_E = 1.15` | 2 | `9.2e−2`, `0.137` | `8.9e−3`, `6.1e−2` | `4.6e−2`, `0.115` | `−0.12`, `−0.13`, `−0.13` |
| excised, lopsided | 2 | `4.6e−2`, `8.1e−2` | `3.7e−4`, `8.5e−4` | `2.7e−2`, `4.5e−2` | `−0.13` |
| the layer (control) | 4 | `4.4e−2`, `0.177` | `1.1e−2`, `3.7e−2` | `9.2e−3`, `3.2e−2` | `−0.08`, `−0.17`, `−0.13` |
| excised at `r_E = 1.15` | 4 | `4.4e−2`, `0.174` | `1.0e−2`, `2.6e−2` | `8.8e−3`, `1.5e−2` | `−0.13` |
| excised, lopsided | 4 | `3.0e−2`, `8.8e−2` | `7.1e−4`, `3.4e−3` | `4.2e−3`, `1.4e−2` | `−0.13` |

- **The closure at the layer's outer radius transmits what the layer
  does**: the same to two digits from depths 2 and 4, within 6 % (`q = 2`)
  and 30 % lower (`q = 4`) from depth 8 by `10 M`, and the pulse likewise
  (half the layer's at `q = 4` by `10 M`).
  So the leakage margin of step 8a ([The margin](CODE.md#the-margin)) carries over
  to excision unchanged: excision makes no more grid-scale content than the
  layer's outer surface does.
- **The lopsided advection is the lever on the leakage**: from depth 8 it
  cuts what crosses the horizon `24×` and `72×` at `q = 2` (`2 M`,
  `10 M`) and `15×` and `7.6×` at `q = 4`; from depth 4 by `1.5–2.0×`;
  from depth 2 — inside its ramp — hardly at all.
- A deeper surface (`r_E = 0.75`) changes nothing for content made above it,
  and a pulse made between it and the horizon starts deeper and leaks
  `5.7×` (`q = 2`) and `5.4×` (`q = 4`) less at `2 M`.

**The go/no-go: the plane** (`model2d`). One component's evolution on
Kerr-Schild `a = 0`'s equatorial plane, `[−5/2, 5/2]²` vertex-centered about
the hole with zero Dirichlet ghosts, in the kernel's form with the
coefficient gradients over the plane — the wave equation of a stationary
2+1 metric with Kerr-Schild's radial speeds and horizon, whose continuum has
no growing or static mode **(decided in step X1:** the first version took
the 3D divergences at `z = 0`, which leave lower-order `∂_z` terms with no
energy behind them; under them the `:damped` control grew at `+0.04/M` and
the excised runs settled onto plateaus**)**. The lego circle `r < r_E` is
excised. *axis*: per-axis closures (`:msn`); *extrap*: centered stencils
with each excised tap replaced by the quadratic extrapolation along the
lattice direction nearest the normal, from sources in the point's `G`-box;
*lop*: the advection lopsided from one cell below the horizon, full at
five. Entries are the largest `Re λ` in `1/M` over the four parity sectors
(the operator commutes with both reflections to `10⁻¹²`), and the RK4
`cfl = dt λ_max/h` the whole spectrum allows. At `h = 5/64` (`65²` points),
`ε_KO = 1/2`:

| `r_E/M` | `r_E/h` | faces | inflow | `q = 2`: axis | axis, lop | extrap | extrap, lop | `q = 4`: axis | axis, lop | extrap | extrap, lop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 6.4 | 52 | 0.154 | `−0.35` / 2.05 | `−0.34` / 1.23 | `−0.35` / 2.05 | `−0.34` / 1.21 | `−0.34` / 1.86 | `−0.34` / 1.56 | `−0.34` / 1.86 | `−0.34` / 1.56 |
| 0.75 | 9.6 | 76 | 0.211 | `−0.35` / 2.05 | `−0.34` / 1.26 | `−0.35` / 2.05 | `−0.34` / 1.24 | `−0.34` / 1.86 | `−0.34` / 1.56 | `−0.34` / 1.86 | `−0.34` / 1.55 |
| 1.00 | 12.8 | 100 | 0.240 | `−0.35` / 2.05 | `−0.34` / 1.28 | `−0.35` / 2.05 | `−0.34` / 1.26 | `−0.34` / 1.86 | `−0.34` / 1.54 | `−0.34` / 1.86 | `−0.34` / 1.53 |
| 1.25 | 16.0 | 124 | 0.387 | `−0.35` / 2.05 | `−0.34` / 1.32 | `−0.36` / 2.05 | `−0.34` / 1.29 | `−0.34` / 1.86 | `−0.34` / 1.57 | `−0.35` / 1.86 | `−0.34` / 1.52 |
| 1.50 | 19.2 | 156 | 0.513 | `−0.35` / 2.05 | `−0.34` / 1.50 | `−0.37` / 2.05 | `−0.34` / 1.38 | `−0.34` / 1.86 | `−0.34` / 1.79 | `−0.34` / 1.86 | `−0.34` / 1.55 |
| 1.75 | 22.4 | 180 | 0.667 | `−0.32` / 2.05 | `−0.33` / 2.05 | `−0.31` / 2.05 | `−0.31` / 2.05 | `−0.34` / 1.86 | `−0.34` / 1.86 | `−0.34` / 1.86 | `−0.34` / 1.86 |

The `:damped` layer of the octant runs (`r_0 = 3/4`, `r_1 = 3/2`, `4/M`) on
the same plane: `−0.27` / 2.05 (`q = 2`) and `−0.34` / 1.86 (`q = 4`). At
`h = 5/48` (`49²`, `r_E = 4.8 … 16.8` cells, faces 36 … 132, inflow
`0 … 0.727`) and at `ε_KO = 1` every entry is again between `−0.30` and
`−0.36/M`; `ε_KO = 1` takes the step to `1.65` (`1.52`), and with the
lopsided advection to `0.93–1.67` (`1.20–1.54`). **No configuration has an
eigenvalue to the right of `−0.30/M`** — the rightmost is the box's own
slowest decay, the same with and without a hole's closure. The inflow-like
fraction on the circle is `1 − √(1 − r_E/(2M))` in the continuum rather
than the sphere's `r_E/(2M)` (`0.29` against `0.5` at `r_E = M`), so the
plane reaches a fraction of `0.73` at `r_E = 7M/4`, where the sphere has
`0.875`; what carries
over is the kind of face, `0 ≤ b/a < 1` in both, which the frozen line
found marginal.

*The controls* (`model2d=controls`, 13 min), the same spectrum at `h = 5/64`
(`5/48` in parentheses) for what the go is read against:

| variation | `q = 2`, `r_E = M` | `r_E = 3M/2` | `q = 4`, `r_E = M` | `r_E = 3M/2` |
|---|---|---|---|---|
| the `:damped` layer, `ε_KO = 0` | **`+0.108`** (`+0.106`) | | **`+0.136`** (`+0.131`) | |
| axis, `ε_KO = 0` | **`+0.144`** (`+0.134`) | **`+0.085`** (`+0.079`) | **`+0.143`** (`+0.127`) | **`+0.094`** (`+0.088`) |
| extrap, `ε_KO = 0` | **`+1.79` (s)** (`+1.52`) | **`+1.35` (s)** (`+1.12`) | **`+3.82` (s)** (`+3.56`) | **`+2.99` (s)** (`+2.68`) |
| axis, reduced rank | `−0.35` | `−0.34` | `−0.34` | `−0.33` |
| axis, one-sided | `−0.35` | `−0.35` | `−0.34` | `−0.35` |
| extrap, degree `≤ q` | `−0.35` | `−0.37` | `−0.35` | `−0.34` |
| extrap, degree 1 | `−0.35` | `−0.36` | `−0.34` | `−0.34` |
| bare frozen core (Dirichlet) | `−0.17` | `−0.029` (**`+0.041` (s)**) | `−0.24` | `−0.066` (`−0.008`) |
| bare frozen core, `ε_KO = 0` | **`+0.241`** | **`+0.191`** | **`+0.285`** | **`+0.228`** |

`(s)`: the mode has half its norm within three cells of the surface.
Without dissipation the per-axis closures grow at about the rate of the
interior itself (the layer has no surface at all) and almost independently
of `h`; the extrapolation grows on the surface ten to thirty times faster,
and faster at the finer `h`. The three dissipation closures and the
extrapolation's degree make no difference at `ε_KO = 1/2`; the bare core
barely decays at `r_E = 3M/2` and grows on its surface at `q = 2`,
`h = 5/48`.

*Noise* (`model2d=noise`, 4 min at `129²`; `model2d=fine`, 36 min at
`257²`): uniform noise on every unknown, RK4 at `cfl = 1/2`, `ε_KO = 1/2`,
to `100 M`. Every configuration — the four families at every `r_E` from
`M/2` to `7M/4`, `q = 2, 4`, and the layer — falls to `0.08–0.75` of its
start by `10 M` and to `10⁻¹⁶–10⁻¹⁴` by `100 M`, at the spectrum's
rightmost rate. The late rate (`1/M`, over `60–100 M`), as a range over
`r_E`:

| `q`, grid | `:damped` layer | axis | axis, lop | extrap | extrap, lop |
|---|---|---|---|---|---|
| 2, `129²` (`h = 5/128`) | `−0.34` | `−0.35 … −0.36` | `−0.35 … −0.36` | `−0.35 … −0.36` | `−0.35 … −0.36` |
| 4, `129²` | `−0.35` | `−0.36` | `−0.35 … −0.37` | `−0.36` | `−0.35 … −0.37` |
| 2, `257²` (`h = 5/256`) | `−0.36` | `−0.34 … −0.36` | `−0.34 … −0.37` | `−0.32 … −0.36` | `−0.34 … −0.37` |
| 4, `257²` | `−0.34` | `−0.34 … −0.35` | `−0.34 … −0.36` | `−0.34 … −0.35` | `−0.34 … −0.36` |

So the closures change nothing the evolution can see, at four resolutions
and to `100 M`: the box's content falls into the hole or out through its
face at the same rate with a layer, a closure or an extrapolation inside.

**The least depth, per chart** (from the margins, the frozen line and the
plane; `q = 4`):

| chart | normal outflow | ends at (equator) | faces with `b/a < 0` | per-axis closures | least depth |
|---|---|---|---|---|---|
| Kerr-Schild `a = 0` | from `m = 1` | the singularity | none | **go** (`r_E = M/2 … 7M/4`) | `⌈√3 G⌉ = 6` cells (`0.375 M` at `h = 1/16`, `0.19 M` at `1/32`) |
| Kerr-Schild `a = 3/5` | from `m = 1` | inner horizon, `1.26 M` down | 2–14 % | not covered | (6 cells) |
| Kerr-Schild `a = 9/10` | from `m = 1` | inner horizon, `0.63 M` down | 5–10 % | not covered | (6 cells; the window is 10 cells at `h = 1/16`) |
| harmonic `a = 7/10` | from `m = 1` | the disk, `0.3 M` down (3 cells' clearance: `m ≤ 8` at `1/48`) | 12–17 % | not covered | (6 cells: fits from `h = 1/48`) |
| harmonic `a = 9/10` | from `m = 1` | the disk, `0.1 M` down (`m ≤ 2` at `1/48`) | 17–18 % | not covered | no room |

**What it cost.** The script's parts on the development machine at a load of
10–40 (the suite and three parts at once): `margins` 20 s, `model1d` 70 s
(`frozen` 20 s, `reflect` 30 s, `radial` 20 s), `model2d=eig` 32 min,
`controls` 13 min, `noise` 4 min, `fine` 36 min. The suite went from
**4787 assertions in 19m23** (one thread, load 5–10) and **4795 in 15m59**
(four, load 5–18) to **6458 in 22m46** (one thread, load 6–22, the first
half beside the plane's runs) and **6466 in 15m12** (four, load 6–11). The
`1671` new ones are `stencils_tests.jl`'s, `13.8 s` and `7.6 s` of it, most
of that compiling the closure tables' `SArray`s at `q = 6, 8`.

### Excision: the variant (step X2b)

The `:excised` variant as built ([Excision](CODE.md#excision-added-2026-10-05),
"What step X2b built"), measured on the development machine (Apple silicon,
12 CPU threads, Julia 1.13.1, TreeAMR 0.1.7), shared with other sessions at
loads of 8 to over 100.

**The suite's fixture** (`test/excision_tests.jl`): Kerr-Schild `a = 0` on
step 5's fixture mesh at `q = 2`, `N = 8` — 120 blocks, 61 440 points,
`h = 5/64` at the hole — with the ball `r < r_E = 3/4` excised (`m = 8`,
`W = 2h`) **(measured in step X2b)**:
- **3743 excised points, 2192 zone points in 32 of the 64 fine blocks, 55 505
  centered.** Every stored point's class is the masks' predicate at its
  position, ghosts and outer faces included, and the zone is the
  enumeration of the right-hand side's taps.
- **The outflow rows at `t = 0`**: 1758 faces, 648 of them inflow-like —
  `0.369` against step X1's `r_E/(2M) = 0.375` — least `b/a = 0.290`, the
  normal margin `b_n/a_n − 1 = 1.237` (`2M/r − 1` at the band), and no closure
  axis whose shift points into the excised set.
- **Bit for bit, on Apple silicon:** the centered points' `du` is the `:none`
  kernel's on the same state at all 55 505 (claimed to `512 eps`); the closure
  provider's whole `F` at 4508 centered points next to the band is the
  centered provider's (claimed to `64 eps`), and each of its contractions is
  `isequal` (claimed so); a `FittedInterior` holding the sphere gives the same
  classes and `du`; `h = −η`, `Π = NaN` planted on every excised point leaves
  every non-excised `du` `isequal` and every excised `du` zero. Every zone
  point's `du` differs from the `:none` kernel's by at least `0.137` — the
  closures replace reads of the core rule's data.
- **The closures are exact** on polynomials of degree `q/2 + 1` at the faces,
  edges and corners of excised half-spaces, `q = 2, 4`, to `10⁻¹²` of the
  data, and `:msn` annihilates degree `< G` there.
- **The lopsided blend** `upwind = (1, 4)`: every point above `r_h − h` keeps
  the blend-free `du` bit for bit; all 29 858 centered points with weight
  `≥ 1/100` move; 1910 of the 2192 zone points move, and the other 282 are
  faces where at `q = 2` the lopsided row *is* the one-sided closure (both
  start at the face). The two providers' lopsided derivatives are `isequal`.
- **A run to `M/5`** (two chunks, 18 steps, `cfl = 1/4`): finite, the normal
  margin `1.2367 → 1.2349`, nothing in the band non-finite. The masked error
  is `1.20e−2` at `M/10` and `1.77e−2` at `M/5`, its L∞ `0.44` and `0.50` at
  the band — the surface's one-sided truncation error at `q = 2`, which the
  masked norm now counts — while **outside the `:damped` fixture's own
  layer, `r ≥ 23/20`, the error is the `:damped` run's: `4.390e−3` against
  `4.395e−3`, and the gauge constraint `3.582e−3` against `3.581e−3`.** A
  chain of two one-chunk jobs is the run bit for bit.
- **`Float32`**: one right-hand side within `8.0e−5` of the `Float64` one's
  scale (`eps(Float32)/h²` of the near-hole data), the same classes.
- **On a device**: on Metal at `Float32` (a scratch environment with
  `Metal` 1.11.1, as `CLAUDE.md` asks) the classes, the census, the zone
  kernel, the lopsided blend and the outflow monitor compile and run: the
  classes are the CPU's, the outflow rows the CPU's to `Float32` roundoff,
  and one right-hand side is within `6.5e−5` of the CPU's scale (`1.4e−4`
  with the blend) — against `2.5e−4` for the `:damped` layer's own CPU–Metal
  difference. The H200 at `Float64` is X3's.
- The tracked geometry (`FittedSpec(; variant = :excised, margin = 16, n_L =
  4)`, its surface at `r ≈ 3/4`): frozen, the found horizon `16.0` cells out
  at every row.

**Unchanged elsewhere** (against the integration branch's `8fa6658` in a
scratch copy with the same manifest): `test/thread_workload.jl`'s six lines
are identical, character for character, and its seventh — the excised
right-hand side, classes, zone kernel and monitors — is identical at one and
four threads. The one-chunk `test/octant_runs.jl` of X2a's check (`case=ks
L=8 N=16 roots=2 radii=4,2 t_end=1/2 chunk=1/2 cfl=1/2`, four threads), for
`:damped` with the default noise and for `:fitted` without: `octant.csv`
identical in every column but `wall`, and `records.csv` identical in its 21
columns, beside the eight new `excision_*` ones, which are `nothing`. The legacy `repr` of every interior without excision
parameters is the base's, byte for byte (`excision_tests.jl` pins three).

**The smoke run** (`test/octant_runs.jl case=ks interior=excised L=8 N=16
roots=2 radii=4,2 t_end=1`, four threads, 22 blocks, 90 112 points, `h =
1/16` at the hole, `q = 4`, the default sphere `r_E = 1`, `m = 16`, the
algebraic source and `10⁻⁸` noise): finished, 110 steps; the band 1387
points, none non-finite; the normal margin `0.7032`; 642 faces, least `b/a =
0.217`, 291 inflow-like (`0.45` against `r_E/(2M) = 0.5`), none into the
excised set; `M_irr − 1 = 4.0e−6` at `t = 1`; `simwatch.toml` carries
`[extra.excision]` and the setup's geometry, `r_E`, margin, band width,
blend and closure. The tracked geometry with the blend on (`geometry=tracked
upwind=1,4 t_end=1/2 chunk=1/2 cfl=1/2`, 28 steps): the seed's surface at
`r ≈ 1` frozen, `m = 16`, 1387 band points, 648 faces (least `b/a = 0.072`,
297 inflow-like), the normal margin `0.684`, and the found horizon
`16.00` cells outside the surface at both rows.

**What it costs.** `bench/stepping.jl` (`BENCH_MODE=step`, `N = 16`, 512
blocks, 2.1 million points, `q = 4`, `Float64`), minimum of five, in ms. The
excised case is the hole's mesh with `r < 3/4` excised: 29 423 excised
points, **14 162 zone points** in 32 of the 512 blocks.

| row | right-hand side, 4 threads | zone kernel alone | per zone point | share of the right-hand side |
|---|---|---|---|---|
| `:damped` hole (the base row, last pair below) | 1444.5–1451.5 | — | — | — |
| `:excised` | **1364.3** | **8.77** | **620 ns** | **0.64 %** |
| `:excised`, `upwind = (1, 4)` | 1368.9 | 9.79 | 691 ns | 0.72 % |

and at one thread (right-hand side, zone kernel, per zone point, share):
`:damped` `4943.3`; `:excised` `4901.2`, `33.18`, **`2343 ns`**, **`0.68 %`**;
with the blend `4918.6`, `37.77`, `2667 ns`, `0.77 %` (load 3–5).

So **a zone point costs about what a whole right-hand side costs per point**
— `2.34 µs` at one thread either way, of which the main kernel is about two
thirds — for generic loops, the codes scanned from the class array and the
nested mixed derivative's inner codes read per outer node; at `0.7 %` of a
right-hand side it is not worth tuning, and on a device it is a launch over
every block of which `32/512` do work. The excised right-hand side is
`0.9 %` cheaper than the `:damped` layer's at one thread (no analytic
`u_exact` in a layer, nothing evaluated at the excised points) and `6 %` at
four, where the layer's expensive points sit in few blocks and so on few
owner threads. **The lopsided blend costs `+0.3 %`** at one and four
threads: here its shell, from the surface to one cell below the horizon, is
494 476 points, a quarter of the mesh, at about `9 ns` each at four threads
— the centered `d1/h` of `Π` still formed and a second contraction of
`q + 1` points per advected derivative.

**No other run moved in cost either.** The wave and the `:damped` hole on
`8fa6658` and on the branch, interleaved base–branch–branch–base twice, four
threads, minimum `rhs` and `imex_owner_step` in ms, the load falling from
about 40 to 7 over the hour (the first branch run shared the machine with a
Metal compile):

| run | load | wave `rhs` | wave step | hole `rhs` | hole step |
|---|---|---|---|---|---|
| base | 20–64 | 1050.3 | 4029.0 | 1883.8 | 7254.6 |
| branch | 14–39 | 1184.6 | 4787.4 | 2164.5 | 8320.9 |
| branch | 25–40 | 1108.6 | 4274.9 | 1614.5 | 6530.5 |
| base | 11–31 | 1062.1 | 4069.9 | 1723.5 | 7048.3 |
| base | 25–30 | 1044.7 | 4256.0 | 1465.7 | 6133.1 |
| branch | 11–24 | 957.4 | 3870.4 | 1461.7 | 5913.8 |
| branch | 9–19 | 938.3 | 3805.2 | 1451.5 | 5875.4 |
| base | 7–16 | 937.7 | 3805.8 | 1444.5 | 5827.0 |

The last, quietest pair differs by `+0.06 %` (wave) and `+0.5 %` (hole);
every other difference follows the load. The arithmetic is the same — the
digests say so — and the kernel takes two more arguments that are
`nothing`.

**The suite.** **6579 assertions in 21m11** at one thread and **6587 in
17m39** at four, the two at once at a load of 7–25 (6488 and 6496 after step
X2a). The 91 new claims: `excision_tests.jl`'s 84, `44.6 s` at one thread and
`32.8 s` at four in the suite (`1m41` alone, which compiles what the suite
shares) — inside `PLAN.md`'s estimate of about 60 s, the tracked run and the
restarted run being most of it; the
`Float32` excised right-hand side in `type_tests.jl` (6, `6.1 s`); and in
`interior_tests.jl` `:excised` accepted beside the refusal of an unknown
variant, which until this step was asserted of `:excised`. The first
four-thread run found that one: 6585 passed and it failed.

### Excision on the static hole (step X3)

The `:excised` variant ([Excision](CODE.md#excision-added-2026-10-05)) on Symmetry's
H200s, one GPU and eight CPU threads a row (`h200q`; CUDA.jl 6.4.2,
KernelAbstractions 0.9.43, TreeAMR 0.1.7, IMEXRungeKutta 1.3.0, Julia 1.13.1,
driver 595.45), from the study's own copy `excision-x3` with `CUDA` added to
its `Project.toml`. Every row is `test/octant_runs.jl` with `PLAN.md`'s
command: Kerr-Schild `a = 0` on the exterior study's octant `[0, 64]³` (root
brick `2³`, cubes `32, 16, 8`, `h = 4/N` in `[0, 8]³`), the algebraic source,
`q = 4`, `cfl = 1/2`, `ε_KO = 1/2`, the Gaussian `γ0`, `:msn`, the finder
every chunk with the spin, uniform noise of `10⁻⁸` outside the excised set;
the excised ball `r < r_E` about the origin, the core rule's `r_0 = r_E/2`,
the margin `m = ⌊(2 − r_E)/h⌋` cells. The comparison is the exterior study's
`:damped` and `:fitted` rows under [Single holes on the
octant](#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07) (in
`CODE.md`'s "Robust stability on the octant" until 2026-10-08), which had no
noise. Each row's `octant.csv`, `records.csv` and `simwatch.toml` are in
`excision-x3/out/x3/{scan,prod}/<row>` on Symmetry, with the job scripts and
their logs; `test/octant_study.jl` reads them (amended in this step to print
the drift of `h_tt` and the outflow rows).

**On the H200 at `Float64` (measured in step X3**, jobs 570330 and 570336**)**:
- **The variant compiles and runs as built.** Nothing in the package needed a
  change: the `UInt8` classes, the closure table as a `NamedTuple` of
  `CuArray`s, the zone kernel, the lopsided blend and the outflow monitor.
- **One right-hand side against the CPU's**, on the smoke's octant (`L = 8,
  N = 16`, 22 blocks) and on the scan's (`L = 64, N = 64`, 29 blocks, 7.6 M
  points) at `r_E = 1`: the largest difference of `du` is `1.48·10⁻¹²`
  against a scale of `0.31` (`1.49·10⁻¹²` with the blend; `:damped`'s own
  `1.52·10⁻¹²` of `0.28`), every value finite; the classes are the CPU's
  exactly — 2443 excised, 1387 zone, the rest centered; the outflow rows are
  the CPU's (the normal margin `0.7031881453195168` against `…177`).
- **The smoke run** (`L=8 N=16 roots=2 radii=4,2 t_end=1 chunk=1/2`, 56
  steps) and its CPU twin on the same node: every cell of `octant.csv` but
  the wall clock within `1.7·10⁻⁹` relative (218 of 255 identical in ten
  digits), `records.csv` the same to fifteen digits, `M_irr − 1 = 3.999·10⁻⁶`
  at `t = 1` in both. 146 s on the H200 against 152 s for eight CPU threads,
  compilation (about two minutes) being most of both.
- **Registers and spills**, from `ptxas`'s own report (`JULIA_DEBUG=CUDACore`
  prints it for every kernel compiled):

  | kernel | registers | stack frame | spill stores / loads | `ld.local` / `st.local` in the PTX | call sites in the PTX |
  |---|---|---|---|---|---|
  | `gh_rhs_kernel!`, `:damped`, `q = 4`, algebraic source | 255 | 10 640 B | 296 / 320 B | 361 / 582 | 143 |
  | the same before step X2a (`5dcddab`) | 255 | 10 528 B | 280 / 304 B | | |
  | `gh_rhs_kernel!`, `:excised` | 255 | 8 800 B | 192 / 192 B | 311 / 495 | 131 |
  | `gh_rhs_kernel!`, `:excised`, blend | 255 | 9 112 B | 316 / 316 B | 311 / 519 | 135 |
  | `gh_zone_kernel!` | 255 | 9 984 B | 172 / 172 B | 311 / 607 | 112 |
  | `gh_zone_kernel!`, blend | 255 | 10 072 B | 260 / 260 B | 311 / 607 | 116 |

  Every right-hand-side kernel is at the 255-register ceiling with a stack
  frame of 9–11 kB. The frame is not the spills — those are 170–320 bytes —
  but, as far as the PTX shows, the device calls that are not inlined (about
  110–140 call sites: the `ntuple` closures, `mixed_stencil`,
  `gh_node_source`, the providers' methods), whose `SVector` arguments pass
  through local memory. So excision adds no register pressure of its own:
  its main kernel's frame is *smaller* than `:damped`'s (no `u_exact` in a
  layer), and the zone kernel's is about `:damped`'s.
- **X2a's check on the device.** Before step X2a (`5dcddab`), after it
  (`8fa6658`) and after X2b (this branch), on one H200 (job 570337): the
  `:damped` octant run's `octant.csv` (but the wall clock) and `records.csv`
  are identical in all three, so the provider is invisible in the values on
  the device too; but it added 112 B to the frame and 16 B to each spill
  direction, and **it costs 5 %**: `bench/stepping.jl`'s hole (`BENCH_CASE=
  hole`, 512 blocks of `16³`, minimum of ten) `rhs` 28.84 and 28.79 ms before
  it against 30.28, 30.29 (`8fa6658`) and 30.32, 30.30 ms (this branch),
  interleaved base–X2a–head–head–X2a–base; the RK4 step 116.7 and 116.4
  against 122.4–122.6 ms. On the CPU X2a measured `±3 %` with changing sign
  and read it as code layout; on the H200 it is `+5.2 %`, reproducible, and
  X2b changed nothing further **(proposed in step X3:** recorded, not
  fixed — the kernel's device efficiency is the research project
  [Precision, threads, devices](CODE.md#precision-threads-devices) defers, and
  forcing the calls inline would remove the provider's frame with the
  rest**)**.
- **What the zone kernel and the blend cost on the H200.** One right-hand side
  on the scan's octant (`N = 64`, 7.6 M points, minimum of ten): `:damped`
  86.8 ms (`11.4 ns` a point), `:excised` 84.8 ms (`11.2 ns`), of which the
  zone kernel is **0.54 ms, `0.63 %`** (1387 zone points, `387 ns` each);
  with the blend 102.0 ms (`13.4 ns`), **`+20 %`**, the zone kernel 0.60 ms.
  On `bench/stepping.jl`'s excised case (512 blocks of `16³`, 14 162 zone
  points in 32 blocks) the zone kernel is 2.00 ms of a 32.04 ms right-hand
  side, **`6.3 %`** (`141 ns` a zone point), against the `:damped` hole's
  30.38 ms, and the blend adds `10.5 %` (35.41 ms). So on the device the
  zone kernel's share is what its launch over every block makes it — a few
  zone points in large blocks cost nothing, many small blocks cost their
  number — and **the blend is not the `0.3 %` it is on the CPU** but 10–20 %
  of every right-hand side: its specialisation of the main kernel is
  dearer at every point, not only in its shell (the shell is 0.2 % of the
  points here).

**The depth scan (measured in step X3**, jobs 570332–570335 to `10 M` and
570347–570348 on to `24 M` from their `t = 8` checkpoints**)**: `h = 1/16`,
`N = 64`, X1's window `r_E = M/2 … 13M/8` with and without the blend
(`upwind=1,4`), twelve rows. At `24 M`; the band is `[r_E, r_E + 3h)`,
*faces* X1's per-axis closure faces, *inflow-like* those with `b/a < 1`
(`r_E/(2M)` predicted: 0.25, 0.38, 0.50, 0.63, 0.75, 0.81); *band ℋ* the
evolved shell `[r_E + 3h, 2)` inside the horizon (the CSV's `in`); the drift
is `h_tt`'s L∞ over `[2, 2 + r_E/2]`:

| `r_E` | `m` | band points | faces, inflow-like | normal margin | ℋ `[2, 2.25)` | with the blend | error `[2, 2.25)` | with the blend | band ℋ | with the blend | drift of `h_tt` | with the blend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `1/2` | 24 | 427 | 168, 39 (0.23) | `1.997` | `8.3·10⁻⁶` | `9.1·10⁻⁶` | `9.4·10⁻⁶` | `1.1·10⁻⁵` | `7.1·10⁻³` | `3.1·10⁻³` | `4.7·10⁻⁶` | `4.9·10⁻⁶` |
| `3/4` | 20 | 839 | 363, 114 (0.31) | `1.177` | `1.4·10⁻⁵` | `9.1·10⁻⁶` | `7.3·10⁻⁶` | `1.1·10⁻⁵` | `2.6·10⁻³` | `9.8·10⁻⁴` | `3.7·10⁻⁶` | `4.9·10⁻⁶` |
| `1` | 16 | 1387 | 642, 291 (0.45) | `0.703` | `3.4·10⁻⁵` | `9.1·10⁻⁶` | `1.7·10⁻⁵` | `1.1·10⁻⁵` | `1.6·10⁻³` | `5.8·10⁻⁴` | `6.5·10⁻⁶` | `4.9·10⁻⁶` |
| `5/4` | 12 | 2065 | 993, 594 (0.60) | `0.402` | `1.4·10⁻⁴` | `8.0·10⁻⁶` | `3.4·10⁻⁵` | `1.0·10⁻⁵` | `1.5·10⁻³` | `4.3·10⁻⁴` | `5.3·10⁻⁶` | `4.8·10⁻⁶` |
| `3/2` | 8 | 2885 | 1413, 1050 (0.74) | `0.192` | `4.5·10⁻⁴` | `5.7·10⁻⁵` | `1.2·10⁻⁴` | `1.1·10⁻⁵` | `2.0·10⁻³` | `4.4·10⁻⁴` | `1.4·10⁻⁵` | `2.5·10⁻⁶` |
| `13/8` | 6 | 3352 | 1659, 1341 (0.81) | `0.108` | `7.2·10⁻⁴` | `3.0·10⁻⁴` | `2.3·10⁻⁴` | `1.0·10⁻⁴` | `2.3·10⁻³` | `8.3·10⁻⁴` | `3.9·10⁻⁵` | `2.1·10⁻⁵` |

against `:damped` at its own depths (`r_1 = 1, 5/4, 3/2, 7/4`, ramps of
12 cells, the exterior study's rows): ℋ `[2, 2.25)` `1.2·10⁻⁵`, `1.1·10⁻⁵`,
`2.4·10⁻⁵`, `5.5·10⁻⁵`, the error there `1.1–1.2·10⁻⁵` (`2.8·10⁻⁵` at
`7/4`), the drift `3.4–8.0·10⁻⁶`.

- **There is no stability boundary in the window.** All twelve rows run to
  `24 M` with nothing non-finite in the band, no closure axis whose shift
  points into the excised set, and outflow rows constant to four digits.
  Everything near the hole saturates by `3–10 M` and is stationary after:
  over `12–24 M` the band's ℋ and error change at `|σ| ≤ 3·10⁻⁴/M`, ℋ just
  outside the horizon at `≤ 10⁻³/M`, and the drift at `0.005–0.015/M`,
  decelerating (`:damped`'s, `0.018–0.023/M` over `8–24 M`, peaks near
  `40 M`). The marginal inflow-like faces of X1's frozen line — up to 81 %
  of the closures at `13M/8` — are harmless in 3D too.
- **The boundary is accuracy, set by leakage, and the blend removes it.**
  Without the blend ℋ just outside the horizon grows as the surface nears
  it, `8.3·10⁻⁶` at 24 cells deep to `7.2·10⁻⁴` at 6 — about a factor `e`
  per 3–5 cells in the middle of the window, the grid-scale content of the
  closure surface penetrating the horizon as step 8a's analysis and X1's
  radial line predicted ([The margin](CODE.md#the-margin)) — and it is 3–19× the
  `:damped` layer's at the same surface radius (`r = 1`, `5/4`, `3/2`). With the blend the
  exterior is **independent of the depth** from `M/2` to `5M/4`: `9.1·10⁻⁶`
  at `M/2`, `3M/4` and `M` to three digits (the lopsided advection carries
  nothing up from below its full region), `8.0·10⁻⁶` at `5M/4`, and it
  degrades only where the surface reaches into the blend's ramp (`3M/2`,
  three cells below its full region: `5.7·10⁻⁵`; `13M/8`: `3.0·10⁻⁴`).
  From 20 cells down (`r_E ≤ 3M/4`) the surface without the blend is
  already below `:damped`'s `2.4·10⁻⁵`, and at 24 cells the blend does not
  help.
- The band inside the horizon carries the closures' lower order and the
  steeper data near `r_E`: `4·10⁻⁴` to `7·10⁻³`, against the `:damped`
  layer's evolved `[r_1, 2)` `2·10⁻⁴` at `r_1 = 3/2` — largest at the
  deepest surface, and 2–4.5× smaller with the blend.
- The mass and the drift: `M_irr − 1 = 1.36–1.49·10⁻⁶` at `24 M` and
  `dM_irr/dt = 2.07–2.14·10⁻⁸/M` over `8–24 M` wherever ℋ just outside is
  below `4·10⁻⁵` (`:damped`: `1.45·10⁻⁶`, `2.09·10⁻⁸`); the shallow rows
  without the blend drift by their leakage (`3/2`: `5.7·10⁻⁸/M`; `13/8`:
  `M_irr − 1 = 5.4·10⁻⁶`).

**The production rows (measured in step X3**, jobs 570341–570346, 570359,
570363–570366 and 570368–570369**)**: `h = 1/16, 1/24, 1/32` (`N = 64, 96,
128`) to `24 M`, two surfaces:
- **A**, `r_E = M` with the blend (`upwind=1,4`; 16, 24, 32 cells), the
  middle of the window, where the scan's exterior was the same with the
  blend at every depth;
- **B**, `r_E = M/2` without it (24, 36, 48 cells), the deepest.

Each ran with the hand-over's noise and, at `h = 1/24` and `1/32`, again
without (`amplitude=0`) **(proposed in step X3**: the noise's constraint
content falls only as a power of `t`, to `ℋ ≈ 8·10⁻⁸` at `24 M` in every
shell — above the truncation error from `r = 3` out at `h = 1/24`, and
comparable to it from `r = 2.25` at `1/32` — while the reference rows have
none; at `h = 1/16` A without
noise agrees with A to three digits inside `r = 3`**)**. Two diagnostics at
`h = 1/24` without noise: **C**, `r_E = M` without the blend, and **D**,
`r_E = M` with the blend from four cells below the horizon
(`upwind=4,4`). And A's surface as a tracked geometry (`geometry=tracked
margin=16`, the seed's offset surface, frozen) at `h = 1/16` is A to four
digits, with the found horizon `16.00` cells outside it at every row. At
`24 M` (without noise, but B at `h = 1/16`, whose noise is invisible inside
`r = 3`); *band ℋ* is the evolved shell inside the horizon — for excision
`[r_E + 3h, 2)`, for the layers `[r_1, 2)` or the offset surface's — and
the last column the H200's seconds per `M` of evolution, the analysis every
chunk included:

| `h` | interior | ℋ `[2, 2.25)` | ℋ `[2.25, 3)` | error `[2, 2.25)` | error `[2.25, 3)` | band ℋ | `M_irr − 1` | `dM_irr/dt` | drift of `h_tt` | s per `M` |
|---|---|---|---|---|---|---|---|---|---|---|
| `1/16` | `:damped` | `2.42·10⁻⁵` | `2.33·10⁻⁶` | `1.08·10⁻⁵` | `3.28·10⁻⁶` | `2.16·10⁻⁴` | `1.45·10⁻⁶` | `2.09·10⁻⁸` | `3.38·10⁻⁶` | 29 |
| `1/16` | `:fitted`, m/n_L = 8/12 | `8.50·10⁻⁴` | `3.51·10⁻⁵` | `1.69·10⁻⁴` | `8.84·10⁻⁶` | `6.78·10⁻³` | `−7.61·10⁻⁷` | `6.99·10⁻⁸` | `1.72·10⁻⁵` | 30 |
| `1/16` | `:fitted`, best (12/14) | `7.29·10⁻⁵` | `2.55·10⁻⁶` | `1.94·10⁻⁵` | `3.72·10⁻⁶` | `1.78·10⁻³` | `1.60·10⁻⁶` | `1.63·10⁻⁸` | `4.65·10⁻⁶` | 30 |
| `1/16` | A: r_E = M, blend | `9.07·10⁻⁶` | `2.08·10⁻⁶` | `1.07·10⁻⁵` | `3.14·10⁻⁶` | `5.75·10⁻⁴` | `1.46·10⁻⁶` | `2.08·10⁻⁸` | `4.91·10⁻⁶` | 35 |
| `1/16` | B: r_E = M/2 | `8.30·10⁻⁶` | `2.13·10⁻⁶` | `9.44·10⁻⁶` | `3.11·10⁻⁶` | `7.05·10⁻³` | `1.41·10⁻⁶` | `2.13·10⁻⁸` | `4.73·10⁻⁶` | 31 |
| `1/24` | `:damped` | `1.46·10⁻⁶` | `4.16·10⁻⁷` | `1.65·10⁻⁶` | `6.06·10⁻⁷` | `8.66·10⁻⁶` | `2.89·10⁻⁷` | `4.05·10⁻⁹` | `8.57·10⁻⁷` | 140 |
| `1/24` | `:fitted`, 12/18 | `3.50·10⁻⁵` | `7.45·10⁻⁷` | `6.19·10⁻⁶` | `5.98·10⁻⁷` | `9.50·10⁻⁴` | `2.42·10⁻⁷` | `5.54·10⁻⁹` | `5.51·10⁻⁷` | 144 |
| `1/24` | `:fitted`, best (16/20, cont 2) | `2.32·10⁻⁶` | `4.08·10⁻⁷` | `1.79·10⁻⁶` | `6.11·10⁻⁷` | `1.12·10⁻⁴` | `2.94·10⁻⁷` | `3.95·10⁻⁹` | `8.77·10⁻⁷` | 145 |
| `1/24` | A | `2.29·10⁻⁶` | `4.09·10⁻⁷` | `2.13·10⁻⁶` | `6.07·10⁻⁷` | `2.44·10⁻⁴` | `2.97·10⁻⁷` | `4.01·10⁻⁹` | `9.76·10⁻⁷` | 162 |
| `1/24` | B | `1.35·10⁻⁶` | `4.14·10⁻⁷` | `1.70·10⁻⁶` | `6.06·10⁻⁷` | `2.93·10⁻³` | `2.90·10⁻⁷` | `4.06·10⁻⁹` | `8.74·10⁻⁷` | 139 |
| `1/24` | C: r_E = M, no blend | `2.16·10⁻⁶` | `4.20·10⁻⁷` | `1.73·10⁻⁶` | `5.98·10⁻⁷` | `6.93·10⁻⁴` | `2.90·10⁻⁷` | `4.15·10⁻⁹` | `9.08·10⁻⁷` | 143 |
| `1/24` | D: r_E = M, blend from 4 cells | `1.72·10⁻⁶` | `4.17·10⁻⁷` | `1.61·10⁻⁶` | `6.10·10⁻⁷` | `2.44·10⁻⁴` | `2.89·10⁻⁷` | `4.06·10⁻⁹` | `8.52·10⁻⁷` | 166 |
| `1/32` | `:damped` | `4.30·10⁻⁷` | `1.31·10⁻⁷` | `5.24·10⁻⁷` | `1.89·10⁻⁷` | `1.77·10⁻⁶` | `8.68·10⁻⁸` | `1.27·10⁻⁹` | `2.72·10⁻⁷` | 351 |
| `1/32` | `:fitted`, 16/24 | `5.84·10⁻⁶` | `1.58·10⁻⁷` | `6.63·10⁻⁷` | `1.81·10⁻⁷` | `2.55·10⁻⁴` | `7.89·10⁻⁸` | `1.44·10⁻⁹` | `2.35·10⁻⁷` | 373 |
| `1/32` | `:fitted`, best (20/24) | `1.11·10⁻⁶` | `1.34·10⁻⁷` | `4.82·10⁻⁷` | `1.89·10⁻⁷` | `1.65·10⁻⁴` | `8.49·10⁻⁸` | `1.26·10⁻⁹` | `2.49·10⁻⁷` | 375 |
| `1/32` | A | `9.27·10⁻⁷` | `1.29·10⁻⁷` | `6.91·10⁻⁷` | `1.90·10⁻⁷` | `1.49·10⁻⁴` | `8.89·10⁻⁸` | `1.25·10⁻⁹` | `3.10·10⁻⁷` | 434 |
| `1/32` | B | `4.29·10⁻⁷` | `1.31·10⁻⁷` | `5.25·10⁻⁷` | `1.89·10⁻⁷` | `1.72·10⁻³` | `8.68·10⁻⁸` | `1.27·10⁻⁹` | `2.72·10⁻⁷` | 374 |

- **B is `:damped` outside the horizon.** ℋ in `[2, 2.25)` is `0.34×`,
  `0.92×` and `1.00×` the layer's, the error there `0.87×`, `1.03×` and
  `1.00×`, every shell from `2.25` out equal to two or three digits,
  `dM_irr/dt` within 2 %, and the drift of `h_tt` `1.40×`, `1.02×` and
  `1.00×`. At `h = 1/32` the two runs agree to three or four digits in every
  shell from `r = 2`, and at `1/16` excision is the better: its surface is
  so far below the horizon that what it leaks is under the layer's own
  leakage at `r_1 = 3/2`. Its orders over the three resolutions
  (`test/octant_study.jl … series=xB64,xB96q,xB128q:16,24,32`): ℋ in
  `[2, 2.25)` `4.48/3.98`, in `[2.25, 3)` `4.04/4.00`, the error there
  `4.24/4.08` and `4.03/4.05`, `dM_irr/dt` `4.09/4.05` — `:damped`'s are
  `6.93/4.25`, `4.24/4.01`, `4.63/3.99`, `4.17/4.05` and `4.05/4.04` (the
  layer's first shell carries a term that falls faster); from `r = 5` out
  the first orders read the noise of the `h = 1/16` row, the second are
  `4.0`.
- **A, C and D: at `r_E = M` the closures, or the blend, leak through the
  first shell at fine `h`.** At `h = 1/16` A is `0.37×` `:damped` in
  ℋ just outside; at `1/24` A is `1.57×`, C (no blend, 24 cells) `1.48×` and
  D (the blend from four cells) `1.18×`; at `1/32` A is `2.16×`, its ℋ
  just outside converging at order `3.39/3.15` against B's `4.48/3.98`
  (the error there `3.97/3.92`) — while every shell from `2.25` out, the
  mass drift and the drift of `h_tt` stay within 15 % of `:damped`'s.
  Twenty-four cells were enough at `1/16` and are not at `1/24`: the
  leakage falls with the depth in cells, the truncation error with `h⁴`, so
  the depth has to grow with the resolution — a fixed radius does that.
- **`:fitted` loses to both.** Against its best setups B is `8.8×`,
  `1.7×` and `2.6×` lower in ℋ just outside the horizon. Inside the horizon
  both are far from the layer: every `:fitted` row's band is `8–140×` the
  layer's, B's `33×` (`1/16`) to `970×` (`1/32`) — the closures' second
  order at the surface, which nothing outside sees.
- **The band** converges at order 2: B `7.05·10⁻³`, `2.93·10⁻³`,
  `1.72·10⁻³` (orders `2.17/1.85`), A `5.75·10⁻⁴`, `2.44·10⁻⁴`,
  `1.49·10⁻⁴` (`2.11/1.71`) — the closures' `∂²` is second order at the
  first evolved point.

**The gauge drift did not return (measured in step X3**, job 570346: `h =
1/16` to `50 M`, A and B**)**. GHSO2's excised hole drifted off the
stationary background at `≈ 0.14/M` — a factor `38` between `24` and `50 M`
— and failed near `45 M` (`notes/sonic-surface.md`). Here, between `24` and
`50 M`:
- the drift of `h_tt` at the horizon rises by 2 % (A: `4.91 → 5.01·10⁻⁶`)
  and 3 % (B: `4.73 → 4.86·10⁻⁶`), peaking at `37` and `41 M` and falling
  after — the shape of the `:damped` layer's own, which on its `128 M` run
  (the `L = 128` octant, the same `h = 1/16` at the hole) rises from
  `3.38·10⁻⁶` at `24 M` to its peak `3.49·10⁻⁶` at `40 M` and is
  `3.46·10⁻⁶` at `50 M`;
- `M_irr` peaks at `26 M` (`1 + 1.46·10⁻⁶`, B `1 + 1.42·10⁻⁶`) and falls at
  `−5.0·10⁻⁹/M` (B `−3.7·10⁻⁹/M`) to `50 M`, the horizon found at every row
  with `J ≤ 1.5·10⁻⁹`;
- ℋ just outside the horizon, the gauge constraint, and the band's ℋ and
  error inside it change by less than 0.4 %, the error in `[2, 2.25)` by
  `−1.3 %` and in `[2.25, 3)` by `+2–3 %`; only the error in `[3, 5)` grows
  (`×1.39`), the settling front of the slightly different stationary hole
  moving outward, as on the `:damped` run's outer levels.

So **the excised hole holds the gauge as the layer does, at the truncation
error**: the drift at the horizon is `1.4×` the layer's at `h = 1/16`,
`1.0–1.1×` at `1/24` and `1.00×` at `1/32` (B), and converges with it.
Whatever drove GHSO2's — its Dirichlet boundary at `R = 5 M`, its
resolution, its elements — is not in this scheme: the static gauge source
holds the gauge without a layer inside.

**The outflow rows** (every row, every chunk): constant to four digits for
the whole run, no non-finite value in the band, no closure axis whose shift
points into the excised set. The normal margin `b_n/a_n − 1` at the band is
`2M/r − 1` at its outermost points (`β` is radial) — A `0.703`, `0.788`,
`0.835` and B `1.997`, `2.266`, `2.406` at `h = 1/16, 1/24, 1/32` — and the
inflow-like fraction of
X1's per-axis faces is A `291/642 = 0.45`, `0.49`, `0.49` and B
`39/168 = 0.23`, `0.21`, `0.24`, against `r_E/(2M) = 0.5` and `0.25`; the
least `b/a` falls with `h` to `0.072` (A at `1/32`) — some face is always
nearly tangent — and is never negative.

**What it costs on the H200.** Per `M` of evolution, the analysis included:
B `31, 139, 374 s` at `h = 1/16, 1/24, 1/32` against `:damped`'s `29, 140,
351 s` (`+0–8 %`); A with the blend `35, 162, 434 s` (`+16–24 %`), D the
same; `:fitted` `30, 144–145, 373–375 s`. The `50 M` rows took `30 min`
(A) and `27 min` (B) on one H200, the `1/32` rows `2 h 34` (B) and
`2 h 57` (A) — `:damped`'s `2 h 24`. The eight GPUs of a node share its
CPUs and its host memory with the other jobs there, which moves these by a
few per cent.

**The suite.** No source changed in this step (`test/octant_study.jl`, which
is not in it, prints the drift of `h_tt` and the outflow rows): **6579
assertions in 17m07** at one thread (load 4–5.5) and **6587 in 12m41** at
four (load 2.4–4), X2b's counts, run after every row above had finished.

### The merge with `main` and the rotating octant (step X4)

Step X4 merged `main` (`cbe3662`, the spill-free right-hand side of
`CODE.md`'s [The right-hand side on an
H200](CODE.md#the-right-hand-side-on-an-h200-measured-2026-10-05)) and the
rotating octant (`0a96f27`) into the excision round's integration branch, and
unified the kernel: the stencil provider is
the argument of `main`'s `gh_rhs_head` and `gh_rhs_pi` ([One right-hand-side
evaluation](CODE.md#one-right-hand-side-evaluation), "One design after `main`'s
rewrite"; [Excision](CODE.md#excision-added-2026-10-05), "What step X4 changed").
Compared against `main` and against the integration branch before the merge
(`b80f4ed`), with one manifest (TreeAMR 0.1.7, KernelAbstractions 0.9.43,
Julia 1.13.1; on the H200 CUDA.jl 6.4.2, driver 595.45, X3's).

**On the CPU** (development machine, Apple silicon, four threads):
- `test/thread_workload.jl` prints `main`'s six lines character for
  character; the seventh, the excised right-hand side, has the base's zone
  (2192), excised (3743) and zone-block (32) counts, its classes' digest, its
  outflow rows and its gauge constraint to every printed digit, and another
  `du` digest (`main`'s source is summed in another order).
- The one-chunk octant runs (`case=ks L=8 N=16 roots=2 radii=4,2 t_end=1/2
  chunk=1/2 cfl=1/2`) — `:damped` with the default noise, `:fitted` without
  — give `main`'s `octant.csv` in every cell but the wall clock, and
  `main`'s `records.csv` in every column `main` has.
- The excised fixture: the centered points' `du` is the `:none` kernel's at
  55 505 of 55 505 points; with the lopsided blend, the 25 165 points beyond
  its start are the blend-free run's bit for bit (496 were not before the
  main kernel took the `:none` call there); the closure provider at the 4508
  centered points next to the band is `Centered`'s at 2826, and within 103
  eps of each variable's largest `|du|` at the rest.

**On one H200** (Symmetry `cn111`, jobs 570646 and 570647, `h200q`, eight CPU
threads; `excision-x4/{main,base,x4}`, each a copy with `CUDA` added). The
octant is X3's scan mesh (`L = 64`, `N = 64`, 29 blocks, 7.6 M points,
`r_E = 1`); the bench is `bench/stepping.jl`'s 512 blocks of `16³`, minimum
of ten, interleaved base–main–X4–X4–main–base and base–X4–X4–base:

| kernel, `q = 4`, `Float64` | registers | stack frame | spill stores / loads | `ld.local` / `st.local` | call sites (PTX) |
|---|---|---|---|---|---|
| `:damped`, `main` | 255 | 6576 B | 5700 / 8480 B | 50 / 119 | 47 |
| `:damped`, step X4 | 255 | 6576 B | 5700 / 8480 B | 50 / 119 | 47 |
| `:damped`, before the merge | 255 | 10 640 B | 296 / 320 B | 361 / 582 | 143 |
| `:excised`, step X4 | 255 | 1544 B | 2692 / 3700 B | 1 / 10 | 5 |
| `:excised`, before | 255 | 8800 B | 192 / 192 B | 311 / 495 | 131 |
| `:excised` with the blend, step X4 | 255 | 2104 B | 5672 / 8212 B | 2 / 20 | 11 |
| `:excised` with the blend, before | 255 | 9112 B | 316 / 316 B | 311 / 519 | 135 |
| zone kernel, step X4 | 255 | 1512 B | 2140 / 3740 B | 1 / 10 | 6 |
| zone kernel, before | 255 | 9984 B | 172 / 172 B | 311 / 607 | 112 |
| zone kernel with the blend, step X4 | 255 | 1688 B | 2780 / 6128 B | 1 / 10 | 10 |

| right-hand side | `main` | before the merge | step X4 |
|---|---|---|---|
| `:damped`, octant, ns a point (kernel alone) | 5.29 (3.96) | 11.40 (10.07) | 5.28 (3.96) |
| `:excised`, octant (kernel; zone kernel) | — | 11.14 (9.73; 0.53 ms, 382 ns a zone point) | **2.88** (1.50; 0.33 ms, 238 ns, 1.5 %) |
| `:excised` with the blend, octant | — | 13.39 (`+20.2 %`) | 2.95 (`+2.7 %`) |
| hole, bench, ms | 17.66, 17.66 | 30.35, 30.35 | 17.71, 17.72 |
| excised, bench, ms (zone kernel) | — | 31.91, 31.89 (2.0; 140 ns a zone point) | **14.80**, 14.81 (1.0; 70.7 ns) |
| excised with the blend, bench, ms | — | 35.37, 35.35 (`+10.8 %`) | 16.43, 16.28 (`+10.5 %`) |
| excised, bench, RK4 step, ms | — | 129.0 | 60.4 |

- **The `:damped` kernel is `main`'s**: the same `ptxas` report and PTX
  statistics, the same time to 0.3 %. **X2a's `+5 %` is gone**: `main`'s head
  and Π components compiled through the slim `Centered` are `main`'s code.
  (`main`'s `:damped` branch still collects `F` as two vectors and spills
  5.7 kB a thread; that is `main`'s, recorded and not changed here.)
- **The excised right-hand side is 3.9× faster on the octant** and 2.2× on the
  bench — the main kernel stores through `main`'s spill-free path, the zone
  kernel through the same head and Π components — and the blend's cost fell
  from a fifth of a right-hand side to `2.7 %` on the octant.
- **The device agrees with the CPU** as before: the largest difference of
  `du` is `1.49·10⁻¹²` against a scale of `0.31` (`:damped` `1.50·10⁻¹²` of
  `0.28`, the same as `main`'s), the classes are the CPU's exactly.

**The excised hole on the rotating octant** (`test/octant_runs.jl
case=ks interior=excised L=8 N=16 roots=2 radii=4,2 t_end=1 chunk=1/2
cfl=1/2 amplitude=0`, `octant=reflecting` against `octant=rotating`, `a = 0`,
`r_E = 1`, `m = 16`; the finder every chunk): the classes and the census are
the mirror octant's at every stored point, the first row is identical, and
at `t = 1` the record agrees to `10⁻⁷` in every `L2` norm, to `9.5·10⁻⁵` in
the `L∞` norms of the shells outside the horizon and to `1.4·10⁻³` in those
inside it — the nested mixed derivative's `x ↔ y` asymmetry, `3.0·10⁻³` at
zone points on both octants, which the rotating seam turns into a difference
between them ([Excision](CODE.md#excision-added-2026-10-05), "What step X4
changed"). With the mixed derivative averaged over both nestings (a scratch
copy, proposed for X6) the two records agree to the CSV's ten digits. The two
seam planes stay one state to `1.6·10⁻¹³` (`3.6·10⁻¹⁴` at zone points) as
built. At `a = 3/5` the build refuses the case by name: 72 (zone point,
closure axis) pairs with the shift into the excised set, the least `b/a =
−0.499` (SimWatch's status `failed`, with the message).

**The suite**: **6727 assertions in 21m10** at one thread and **6735 in
16m29** at four (the two at once, the machine loaded 7–17 by step X5's
models), all green — X3's counts, `main`'s 112, the rotating octant's 29 and
the seam's 7. Two claims changed with `main`'s head, each saying so in its
test: `evolution_tests.jl`'s probe sees `d1` of `h` asked twice per component
and axis (270 requests, not 240), and `excision_tests.jl`'s closure provider
at a point with no excised tap is `Centered`'s to 512 eps of each variable's
largest `|du|` (measured 103), its every contraction still `isequal`.

### Excision: the frame-dragged faces (step X5)

Host-side, no kernel change: `test/excision_model.jl`'s `margins=window` and
the `model2d=spin…` parts, and the extrapolation's weights in
`src/stencils.jl` (under [Excision](CODE.md#excision-added-2026-10-05), "The
frame-dragged faces (step X5)"). Every model is X1's — the same sparse
assembly from the package's closure and extrapolation weights and the
metric's coefficients as the kernel reads them — on Kerr-Schild `a = 3/5`:
`r₊ = 1.8` at the poles, `1.897` on the equator, the ring at `ρ = 0.6`, the
inner horizon at `0.632` on the equator. Its self-checks throw: the turned
coefficients against the metric, the operator against the turns it commutes
with, and the general operator against X1's at `a = 0` (to `10⁻¹³`). The
local parts ran on the development machine (Apple silicon, 12 threads) at a
load of 10–30, beside step X4's suites; the spectra at `5/64`, `5/96`, the
controls and the noise on Symmetry's EPYC nodes (`amdq`), one process per
`(q, ε_KO, r_E)` or family.

**The window** (`margins=window`, 20 s). The sphere `r < r_E` at the
spinning round's spacings; *normal* the least `b_n/a_n − 1` along its
radial normal; the depth below the horizon in cells; *ring room* the cells
between the ring and the surface, where the core rule's sphere must go; the
*frame-dragged* closure axes (`k_s < G`, `q = 4`, the shift pointing into
the excised set: X2b's refusal, X5's rule) and their fraction of the closure
axes; their least `b/a`; the fraction of X1's faces with both
characteristics entering (`b/a < −1`):

| `r_E` | normal | poles: cells at `1/24, 1/32, 1/48` | equator | ring room at `1/24` | frame-dragged axes at `1/24, 1/32, 1/48` (fraction) | least `b/a` | `b/a < −1` at `1/48` |
|---|---|---|---|---|---|---|---|
| 0.65 | `+0.052` | 27.6, 36.8, 55.2 | 29.9, 39.9, 59.9 | 1.2 | 1472, 2844, 6824 (0.107–0.124) | `−2.57, −2.92, −3.94` | 0.021 |
| 0.70 | `+0.224` | 26.4, 35.2, 52.8 | 28.7, 38.3, 57.5 | 2.4 | 1504, 2888, 6812 (0.094–0.107) | `−1.50, −2.02, −2.13` | 0.012 |
| 0.75 | `+0.322` | 25.2, 33.6, 50.4 | 27.5, 36.7, 55.1 | 3.6 | 1452, 2752, 6712 (0.080–0.092) | `−1.21, −1.32, −1.43` | 0.006 |
| 0.80 | `+0.381` | 24.0, 32.0, 48.0 | 26.3, 35.1, 52.7 | 4.8 | 1488, 2836, 6828 (0.072–0.082) | `−1.38, −1.17, −1.39` | 0.004 |
| 0.90 | `+0.429` | 21.6, 28.8, 43.2 | 23.9, 31.9, 47.9 | 7.2 | 1564, 2840, 6888 (0.059–0.065) | `−0.96, −0.82, −1.09` | 0.001 |
| 1.00 | `+0.422` | 19.2, 25.6, 38.4 | 21.5, 28.7, 43.1 | 9.6 | 1424, 2828, 6752 (0.044–0.052) | `−0.57, −0.65, −0.73` | 0 |
| 1.133 | `+0.368` | 16.0, 21.3, 32.0 | 18.3, 24.5, 36.7 | 12.8 | 1452, 2956, 6800 (0.035–0.041) | `−0.58, −0.61, −0.61` | 0 |
| 1.30 | `+0.268` | 12.0, 16.0, 24.0 | 14.3, 19.1, 28.7 | 16.8 | 1488, 2840, 6972 (0.027–0.032) | `−0.44, −0.48, −0.49` | 0 |
| 1.50 | `+0.149` | 7.2, 9.6, 14.4 | 9.5, 12.7, 19.1 | 21.6 | 1408, 2748, 6592 (0.019–0.023) | `−0.29, −0.30, −0.31` | 0 |
| 1.70 | `+0.046` | 2.4, 3.2, 4.8 | 4.7, 6.3, 9.5 | 26.4 | 1464, 2792, 6908 (0.016–0.018) | `−0.28, −0.26, −0.28` | 0 |

- **Normal outflow** holds on the sphere from `r_E = 0.641` (`−0.0007` at
  `0.64`, `−0.051` at `0.632`) up to the horizon, largest near `r_E = 0.9`;
  on the tracked offset surface at every `m` scanned, from `4` down to the
  inner horizon (`+0.04` to `+0.46`; the deepest rows, `m = 28, 40, 56` at
  `h = 1/24, 1/32, 1/48` with the equator at `0.731, 0.647, 0.731`, have
  `+0.32, +0.05, +0.32`).
- **The frame-dragged axes are a fixed number at each `h`, not a fraction**:
  about `1450`, `2800` and `6800` at every depth on both geometries, so
  their share of the closure axes grows inward, from 2 % near the horizon to
  12 % at `r_E = 0.65`. They are along `x` and `y` only — equal numbers of
  each and none along `z`, whose faces see the radial part of `β` (counted
  at four `(h, r_E)`). The tracked offset surface has the same counts, and
  least ratios close to those of the sphere through its equator (`m = 24`
  at `1/24`, the equator at `0.897`: `−0.94` against the sphere's `−0.96`
  at `0.9`).
- **X5's rule always finds its sources in 3D**: every excised tap of a
  frame-dragged axis's advective stencil (`1612–1732`, `2928–3228` and
  `6708–7360` taps) has three non-excised points beyond it along the nearest
  of the 26 lattice directions inside the point's `G`-box, except 8–124 that
  have two (all at `r_E ≤ 0.8`, at most 1.7 %); none has fewer, the first
  is at most two steps out and the last at most three (`k₀ + n − 1 ≤ G`, what
  `extrapolation_table` holds).

**The plane's faces** (`model2d=spinfaces`, 10 s): the lego circle on the
equatorial plane — faces, their least `b/a`, how many have `b/a < 0` and
`b/a < −1`, and the closure axes the rule extrapolates:

| `r_E` | `n = 24` (`h = 5/48`) | 32 (`5/64`) | 48 (`5/96`) | 64 (`5/128`) | 128 (`5/256`) |
|---|---|---|---|---|---|
| 0.65 | 52, `−2.57`, 12, 8, 24 | 68, `−2.19`, 16, 8, 36 | 100, `−2.57`, 28, 12, 64 | 132, `−2.78`, 36, 16, 92 | 268, `−3.76`, 76, 36, 208 |
| 0.75 | 60, `−1.31`, 8, 4, 20 | 76, `−0.92`, 12, 0, 24 | 116, `−1.31`, 20, 4, 48 | 156, `−1.69`, 32, 8, 76 | 308, `−1.69`, 60, 12, 160 |
| 0.90 | 68, `−0.34`, 4, 0, 8 | 92, `−0.55`, 8, 0, 20 | 140, `−0.76`, 16, 0, 40 | 188, `−1.04`, 24, 4, 60 | 372, `−1.09`, 48, 4, 132 |
| 1.10 | 84, `−0.22`, 4, 0, 8 | 116, `−0.54`, 8, 0, 20 | 172, `−0.54`, 12, 0, 32 | 228, `−0.60`, 20, 0, 52 | 452, `−0.60`, 36, 0, 100 |
| 1.30 | 100, `−0.16`, 4, 0, 8 | 132, `−0.18`, 4, 0, 12 | 196, `−0.21`, 8, 0, 20 | 268, `−0.35`, 16, 0, 40 | 532, `−0.37`, 28, 0, 80 |
| 1.50 | 116, `−0.12`, 4, 0, 8 | 156, `−0.24`, 8, 0, 16 | 228, `−0.15`, 8, 0, 16 | 308, `−0.24`, 12, 0, 32 | 612, `−0.24`, 24, 0, 64 |
| 1.70 | 132, `−0.09`, 4, 0, 8 | 172, `−0.07`, 4, 0, 8 | 260, `−0.12`, 8, 0, 16 | 348, `−0.17`, 12, 0, 28 | 700, `−0.28`, 24, 0, 64 |

The plane's frame-dragged faces are 3–30 % of its faces, more than the
sphere's (the equator is where frame dragging lies along the surface), and
reach the sphere's least ratios from `n = 64` on.

**The go/no-go: the spinning plane's spectrum** (`model2d=spineig`). X1's
model with the coefficients of `a = 3/5`, whose operator commutes with the
half turn `P → −P` only — the spin breaks the mirrors, and the nested
mixed derivative the quarter turn — so its spectrum is computed in two
sectors (the quarter turn's four where a family allows it). Entries: the
rightmost `Re λ` in `1/M` over the seven `r_E = 0.65, 0.75, 0.90, 1.10,
1.30, 1.50, 1.70` (bold above `10⁻⁹`, with the unstable `r_E`; every bold
mode has more than half its norm within three cells of the surface) / the
least RK4 `cfl = dt λ_max/h` over them; *lop* the advection lopsided from
one cell below `r₊ = 1.8` over four:

| `q`, `ε_KO`, `h` | `:damped` layer | axis | axis, lop | extrap | extrap, lop | hybrid | hybrid, lop | hybrid-adv | hybrid-adv, lop |
|---|---|---|---|---|---|---|---|---|---|
| 4, 1/2, 5/48 | `−0.079` / 1.89 | **`+4.81`** (0.65, 0.75) / 1.88 | **`+4.98`** (0.65, 0.75) / 1.56 | **`+2.87`** (0.65) / 1.89 | `−0.087` / 1.58 | `−0.078` / 1.88 | `−0.087` / 1.58 | `−0.078` / 1.88 | `−0.087` / 1.57 |
| 4, 1/2, 5/64 | `−0.082` / 1.88 | **`+3.28`** (0.65) / 1.87 | **`+3.48`** (0.65) / 1.52 | `−0.082` / 1.87 | `−0.086` / 1.52 | `−0.082` / 1.87 | `−0.086` / 1.53 | `−0.082` / 1.87 | `−0.086` / 1.53 |
| 4, 1/2, 5/96 | `−0.084` / 1.86 | **`+3.64`** (0.65) / 1.86 | | | | `−0.083` / 1.86 | | `−0.083` / 1.86 | |
| 4, 1, 5/48 | `−0.103` / 1.54 | **`+4.31`** (0.65, 0.75) / 1.54 | **`+4.50`** (0.65, 0.75) / 1.22 | **`+3.20`** (0.65) / 1.54 | `−0.108` / 1.22 | `−0.103` / 1.54 | `−0.108` / 1.22 | `−0.103` / 1.54 | `−0.108` / 1.22 |
| 4, 1, 5/64 | `−0.106` / 1.52 | **`+2.45`** (0.65) / 1.52 | **`+2.75`** (0.65) / 1.18 | `−0.106` / 1.52 | `−0.108` / 1.16 | `−0.106` / 1.52 | `−0.108` / 1.18 | `−0.106` / 1.52 | `−0.108` / 1.19 |
| 2, 1/2, 5/48 | `−0.077` / 2.08 | **`+2.69`** (0.65, 0.75) / 2.08 | **`+2.17`** (0.65) / 1.21 | **`+2.26`** (0.65) / 2.08 | `−0.099` / 1.21 | `−0.075` / 2.08 | `−0.099` / 1.21 | `−0.075` / 2.08 | `−0.099` / 1.21 |
| 2, 1/2, 5/64 | `−0.081` / 2.06 | **`+1.20`** (0.65) / 2.06 | **`+0.26`** (0.65) / 1.17 | `−0.081` / 2.06 | `−0.096` / 1.16 | `−0.081` / 2.06 | `−0.096` / 1.17 | `−0.081` / 2.06 | `−0.096` / 1.17 |
| 2, 1, 5/48 | `−0.108` / 1.68 | **`+1.87`** (0.65) / 1.68 | **`+1.63`** (0.65) / 0.95 | **`+1.97`** (0.65) / 1.68 | `−0.128` / 0.94 | `−0.107` / 1.68 | `−0.128` / 0.95 | `−0.107` / 1.68 | `−0.128` / 0.95 |
| 2, 1, 5/64 | `−0.112` / 1.66 | `−0.114` / 1.66 | `−0.091` / 0.92 | `−0.114` / 1.66 | `−0.126` / 0.91 | `−0.114` / 1.66 | `−0.126` / 0.92 | `−0.114` / 1.66 | `−0.126` / 0.92 |

- **The per-axis closures fail at `r_E = 0.65` at every resolution (but for
  `q = 2`, `ε_KO = 1` at `5/64`), and at `0.75` on the coarsest plane**,
  with a surface mode of `+0.02` to `+5/M`; the lopsided blend does not
  save them (it falls back to
  the closure where the upwind side is excised). Elsewhere — every `r_E ≥
  0.9`, and `0.75` at `5/64` and `5/96` — they are stable, though every row
  has frame-dragged faces: on the plane the failure needs faces well below
  `b/a = −1` (`−1.31` breaks `5/48` and not `5/96`; `−2.2` to `−2.6` breaks
  every resolution), where the frozen line found every `b/a < 0` unstable
  (`+1.1·10⁻⁴/h` at `−0.02` to `+5.0·10⁻²/h` at `−0.5`, `q = 4`, `ε_KO =
  1/2`, `model1d=frozen ratios=…`).
- **X1's extrapolation fails at `r_E = 0.65` on the coarsest plane** at every
  `(q, ε_KO)` (`+2.0` to `+3.2/M`) and is stable everywhere else and with
  the blend.
- **At X7's proposed depth, `r_E = 0.80`** (`spineig rE=0.8`, the same
  configurations, 6 min on one node), every family is stable at `5/48` and
  `5/64` and the three computed at `5/96`, with faces down to `b/a = −0.46`,
  `−0.96` and `−1.06`; the hybrids are within `0.002/M` of the layer. So
  the per-axis closures happen to hold there on the plane — but the 3D
  sphere at `0.80` has faces at `−1.17` to `−1.39`, where the plane has
  already broken them once (`−1.31` at `5/48`), and the rule costs nothing
  where it is not needed.
- **Both hybrids are stable in every configuration**, their rightmost
  eigenvalue within `0.002/M` of the layer's at `r_E = 0.65` and below it
  where the hole is larger, and RK4's step the layer's to `0.01`. They agree
  with each other to the three digits printed in every cell of the
  per-`r_E` tables — extrapolating the advection alone is the whole fix.
  The blend costs the step (`1.88 → 1.53–1.58` at `q = 4`, `ε_KO = 1/2`,
  at the worst `r_E`) and buys nothing here.

*The controls* (`model2d=spincontrols`), the rightmost `Re λ` at `r_E =
0.75` and `1.10`, `ε_KO = 1/2` unless stated:

| variation | `q = 4`, `5/48` | `q = 4`, `5/64` | `q = 2`, `5/48` | `q = 2`, `5/64` |
|---|---|---|---|---|
| the `:damped` layer, `ε_KO = 0` | **`+0.13`** | **`+0.14`** | **`+0.11`** | **`+0.12`** |
| axis, `ε_KO = 0` | **`+2.88` (s)**, **`+0.12`** | **`+0.60` (s)**, **`+0.36` (s)** | **`+1.45` (s)**, **`+0.13`** | **`+0.60` (s)**, **`+0.13`** |
| extrap, `ε_KO = 0` | **`+3.10` (s)**, **`+2.21` (s)** | **`+3.22` (s)**, **`+2.18` (s)** | **`+1.49` (s)**, **`+1.12` (s)** | **`+1.61` (s)**, **`+1.25` (s)** |
| hybrid, `ε_KO = 0` | **`+0.22` (s)**, **`+0.12`** | **`+0.20`**, **`+0.13`** | **`+0.22`**, **`+0.13`** | **`+0.22`**, **`+0.14`** |
| hybrid-adv, `ε_KO = 0` | **`+0.23` (s)**, **`+0.12`** | **`+0.20`**, **`+0.13`** | **`+0.22`**, **`+0.13`** | **`+0.22`**, **`+0.13`** |
| axis, reduced rank; one-sided | `−0.023`, `−0.080`; `−0.078`, `−0.081` | `−0.082` | `−0.075`, `−0.080` | `−0.081` |
| hybrid, reduced rank; hybrid-adv, one-sided | `−0.078`, `−0.081` | `−0.082` | `−0.075`, `−0.081` | `−0.081` |
| hybrid-adv, extrapolation of degree 1; `≤ q` | `−0.078`, `−0.081` | `−0.082` | `−0.075`, `−0.082` | `−0.081`, `−0.082` |
| hybrid-adv where `b/a < 1/2` | `−0.078`, `−0.081` | `−0.082` | `−0.075`, `−0.082` | `−0.081`, `−0.082` |
| hybrid-adv **at every closure axis** | **`+7.07` (s)**, **`+13.6` (s)** | **`+13.5` (s)**, **`+18.2` (s)** | **`+2.64` (s)**, **`+5.02` (s)** | **`+6.43` (s)**, **`+6.55` (s)** |
| extrap, degree 1 | `−0.078`, `−0.079` | `−0.082` | `−0.075`, `−0.079` | `−0.081` |
| bare frozen core (Dirichlet) | `−0.078`, `−0.077` | `−0.082` | `−0.075`, `−0.067` | `−0.081`, `−0.055` |
| bare frozen core, `ε_KO = 0` | **`+0.34`**, **`+0.23`** | **`+0.34`**, **`+0.24`** | **`+0.29`**, **`+0.20`** | **`+0.37`**, **`+0.22`** |

Without dissipation the hybrids grow as the layer does (the interior's own
grid-scale growth, `notes/sonic-surface.md`), the per-axis closures up to
twenty times faster at `r_E = 0.75` and the extrapolation ten to twenty-five
times: `ε_KO > 0` stays required. The rule's details — its threshold (`0` or
`1/2`), the extrapolation's degree, the dissipation's closure — change
nothing at `ε_KO = 1/2`; extrapolating the advection where the shift
points *out* of the excised set too is violently unstable on every plane.
The bare frozen core is stable here at `ε_KO = 1/2`, but it over-specifies
an outflow surface (X1's reflection), and it is not a candidate.

*Noise* (`model2d=spinnoise`, `129²`, `h = 5/128`; `model2d=spinfine`,
`257²`, `h = 5/256`): uniform noise on every unknown, RK4 at `cfl = 1/2`,
`ε_KO = 1/2`, to `100 M`; the late rate (`60–100 M`) as a range over `r_E =
0.65 … 1.7`:

| `q`, grid | `:damped` layer | axis | axis, lop | extrap | extrap, lop | hybrid | hybrid, lop | hybrid-adv | hybrid-adv, lop |
|---|---|---|---|---|---|---|---|---|---|
| 4, `129²` | `−0.088` | **blows up at `5 M`** (0.65); `−0.083 … −0.090` | **blows up at `5 M`** (0.65); `−0.083 … −0.089` | `−0.083 … −0.090` | `−0.083 … −0.089` | `−0.084 … −0.090` | `−0.084 … −0.089` | `−0.083 … −0.090` | `−0.084 … −0.089` |
| 2, `129²` | `−0.091` | **blows up at `26 M`** (0.65); `−0.089 … −0.099` | `−0.090 … −0.094` | `−0.090 … −0.099` | `−0.090 … −0.094` | `−0.089 … −0.099` | `−0.090 … −0.094` | `−0.089 … −0.099` | `−0.090 … −0.094` |
| 4, `257²` | `−0.074` | **blows up at `5 M`** (0.65); `−0.074 … −0.075` | **blows up at `6 M`** (0.65); `−0.073 … −0.075` | `−0.074 … −0.075` | `−0.073 … −0.075` | `−0.074 … −0.075` | `−0.073 … −0.075` | `−0.074 … −0.075` | `−0.073 … −0.075` |
| 2, `257²` | `−0.086` | `−0.084 … −0.088` | `−0.083 … −0.088` | `−0.084 … −0.088` | `−0.083 … −0.087` | `−0.084 … −0.088` | `−0.083 … −0.087` | `−0.084 … −0.088` | `−0.083 … −0.087` |

Every hybrid run falls to `10⁻⁴`–`10⁻⁵` of its start by `100 M` (`6·10⁻⁴`
at `257²`, `q = 4`) at the layer's rate, which is itself slower on the finer
plane (`−0.074/M`); the per-axis run at `r_E = 0.75`, `q = 4`, `129²` first
grows to twice its start by `10 M` (the faces at `b/a = −1.69`) and then
decays, and at `q = 2` the per-axis closures survive `r_E = 0.65` at `257²`
after blowing up at `129²` — the staircase again.

**What it cost.** `margins=window` 20 s and `spinfaces` 10 s locally;
`spineig` at `n = 24` 22 min locally at four threads (the four `(q,
ε_KO)` one after the other); on Symmetry, `n = 32` 8 min for its 28
processes at two threads, `n = 48` (three families, `q = 4`, `ε_KO = 1/2`)
14 min for 21 processes at three, the controls 10 min for 8 processes at
four, `spinnoise` 5 min and `spinfine` 23 min at 64 threads, `r_E = 0.80`
6 min. **The suite**: before the step **6579 assertions in 24m30** at one
thread (load 8–30; X3 recorded the same sources at four threads, 6587 in
12m41, and that run was not repeated); after it **6804 in 24m14** at one
thread and **6812 in 18m31** at four, the two at once beside step X4's
suites (load 8–32). The 225 new claims are `stencils_tests.jl`'s
extrapolation, `1.3 s` of it.

### Excision: the frame-dragged faces in the zone kernel (step X6)

What step X6 built is under [Excision](CODE.md#excision-added-2026-10-05), "What step
X6 built": step X5's rule (`hybrid-adv`) in a second launch of the zone
kernel, from rule bits and direction codes built once with the problem; the
zone points' mixed derivative symmetric in its two axes; the spinning hole
on the rotating octant end to end. On the development machine (Apple silicon,
12 threads, Julia 1.13.1, TreeAMR 0.1.7) at a load of 4–10 from other
sessions, and on one H200 (Symmetry `cn111`, job 570816, `h200debugq`, eight
CPU threads; CUDA.jl 6.4.2, driver 595.45) from the copy `excision-x6/x6` with
`CUDA` added.

**The spinning hole runs.** `test/octant_runs.jl case=ks octant=rotating a=3/5
interior=excised L=8 N=16 roots=2 radii=4,2 r_E=1 r_0=0.8 t_end=1 chunk=1/2
cfl=1/2` — `h = 1/16` at the hole, `m = 12` cells below the poles, `q = 4`,
the algebraic source, noise `10⁻⁸`, 56 steps:
- the build finds **72 frame-dragged (zone point, axis) pairs** — the 72 X4
  refused, least `b/a = −0.499` — every one along `y` (on the rotating
  octant's quadrant frame dragging points the shift into the ball along `y`
  only; the `x` ones are the other quadrants'), with 33 of the band's 642
  faces on them; no excised tap without a source; 1387 band points, 2443
  excised;
- at every row the band is finite, the normal margin `+0.348`, the
  frame-dragged axes 72, **the flips 0**; at `t = 1` ℋ just outside the
  horizon is `7.29·10⁻⁵` (`2.87·10⁻³` over the mesh, the band's), `M_irr` is
  Kerr's `+1.19·10⁻⁶`, `J` is `a + 5.07·10⁻⁶`, the horizon found every chunk;
  105 s of wall clock at four threads, most of it compilation. SimWatch
  writes `simwatch.toml` with the rows under `[excision]`;
- **on the H200** the classes and the codes are the CPU's exactly, `du` is
  within `1.08·10⁻¹²` of its scale `1.95` (X3: `1.5·10⁻¹²` at `a = 0`), the
  outflow rows the CPU's; the same run on CUDA gives an `octant.csv` within
  `3.2·10⁻⁹` relative of the CPU's (72 of 89 columns identical to ten digits)
  and a `records.csv` within `1.5·10⁻¹¹`.

**The symmetric mixed derivative.** What it changes, and what it buys:
- **The two octants evolve as one.** X4's comparison (`a = 0`, `r_E = 1`, the
  same mesh, `amplitude=0 cfl=1/2`, to `t = 1`), the rotating octant against
  the mirror octant: with steps X2b–X5's nesting (`mixed=nested`) they differ
  by `1.43·10⁻³` in the `L∞` norms inside the horizon and `9.5·10⁻⁵` in
  ℋ's `L∞` in `[2, 2.25)` — X4's numbers, reproduced; **symmetric, by at most
  `8.5·10⁻¹⁰`, the CSV's tenth digit** (75 of 89 columns identical), and the
  record by `7·10⁻¹³`. In the suite, four RK4 steps of each octant on its own
  (`q = 2`, `h = 5/64`): states `0.77` eps apart and their `du` `2.9·10³` eps
  of each variable's largest value (symmetric), `1.7·10⁻⁶` and `1.5·10⁻³`
  (nested). Host-side, `½(D_x D_y + D_y D_x)` at a point and at its image
  across the diagonal is bit for bit the same on 253 (`q = 2`) and 668
  (`q = 4`) points next to a tilted surface, where the nested sum differs at
  every one.
- **`mixed = :nested` is the old operator bit for bit**: X3's smoke row
  (`case=ks interior=excised L=8 N=16 roots=2 radii=4,2 t_end=1 chunk=1/2`, the
  default noise and `cfl = 1/4`) gives the base's (`5da40da`) `octant.csv` in
  every column but the wall clock and its `records.csv` in every column, and
  the fixture's run to `M/5` the base's state (one digest).
- **Against X2b's fixture record** (`q = 2`, `r_E = 3/4`, to `M/5`): the
  error outside `r = 23/20` is `4.389940·10⁻³` in both (X2b recorded
  `4.390·10⁻³`; the two differ by `7·10⁻⁸` relative); in the band
  `[3/4, 23/20)` the error falls from `0.0892` to `0.0869` in `L2` and from
  `0.500` to `0.422` in `L∞`, and the masked error, which counts the band,
  from `0.01768` to `0.01726`; the gauge constraint moves by `6·10⁻⁵`.
- **Against X3's smoke row** (above), symmetric against the base at `t = 1`:
  every shell from `r = 3` out the same to seven digits; `[2, 2.25)` and
  `[2.25, 3)` within `3.5·10⁻⁵` in the `L2` norms (`1.8·10⁻⁴` in ℋ's `L∞`
  just outside); the band inside the horizon within 1.2 %; the masked error
  1.4 % in `L2`, 4.3 % in `L∞`; `M_irr − 1 = 3.99991·10⁻⁶` against
  `3.99910·10⁻⁶` (X3 recorded `3.999·10⁻⁶`).

**What the rule and the symmetry cost.** On the CPU, `bench/stepping.jl
BENCH_CASE=excised` at four threads — the hole's mesh, 512 blocks of `16³`
(2.1 M points), the ball `r < 3/4` excised at `h = 5/128`, 14 162 zone
points in 32 blocks — and with `BENCH_A=3/5` the same hole spinning (core
rule at `0.675`): base (`5da40da`) and step X6 interleaved base–X6–X6
(`a = 3/5`)–X6–base–X6 (`a = 3/5`), minimum of ten, at a load of 8–12:

| row | base | step X6, `a = 0` | step X6, `a = 3/5` |
|---|---|---|---|
| right-hand side | 1002.8, 991.6 ms | 994.1, 995.8 ms | 963.7, 1002.7 ms |
| zone kernel | 7.16, 7.11 ms (505, 502 ns a zone point; 0.71 %) | 9.24, 9.21 ms (652, 651 ns; 0.93 %) | 9.11, 9.35 ms |
| the rule's launch | — | — (not launched) | 2.48, 2.29 ms: 1700 frame-dragged points in 24 blocks, 1458, 1346 ns each; **0.26, 0.23 %** |

The symmetric mixed derivative costs the zone kernel `+29 %` — a second
nested sum wherever a point's box meets the excised set — which is `+0.2 %`
of a right-hand side; the rule costs a quarter of a per cent. On the H200:

| kernel, `q = 4`, `Float64` | registers | stack frame | spill stores / loads | `ld.local` / `st.local` | call sites (PTX) |
|---|---|---|---|---|---|
| zone kernel, step X6 (symmetric) | 255 | 1504 B | 2164 / 3828 B | 1 / 10 | 6 |
| zone kernel, step X4 | 255 | 1512 B | 2140 / 3740 B | 1 / 10 | 6 |
| frame-dragged kernel | 255 | 1872 B | 2668 / 5504 B | 265 / 37 | 7 |

| octant, `a = 3/5`, `r_E = 1` | points | right-hand side | zone kernel | the rule's launch |
|---|---|---|---|---|
| the smoke's (`L = 8`, `N = 16`, 22 blocks) | 90 112 | 5.87 ms | 0.390 ms (281 ns a zone point, 6.6 %) | 0.397 ms (72 points, 6.8 %) |
| X3's scan (`L = 64`, `N = 64`, 29 blocks) | 7.6 M | 22.6 ms (2.97 ns a point) | 0.379 ms (273 ns, 1.7 %) | 0.466 ms (72 points, 2.1 %) |

against X4's `2.88` ns a point and `238` ns a zone point for the `a = 0`
excised octant. On the device the zone kernel and the rule are latency, not
throughput: a few dozen to a few thousand threads that each form a whole
`F`, about `0.4–0.5 ms` a launch whatever the mesh, so their share falls as
the mesh grows — `2.1 %` at 7.6 M points, and **(predicted)** `0.6 %` at
X7's `h = 1/24` (25.7 M points) and `0.3 %` at `1/32` (60.8 M). The
frame-dragged kernel's 265 `ld.local` are, as far as the PTX shows, its
run-time indexing of the extrapolation table and its tap loop (not measured
further: they run at a few dozen points).

**The suite.** Before the step, on its base (`5da40da`, X4 with X5), **6952 assertions in
19m31** at one thread and **6960 in 14m39** at four (the two at once, at a
load of 4–9); after it, **7001 in 20m18** and **7009 in 15m42** (the two at
once, a load of 7–10). The 49 new claims are `excision_tests.jl`'s, whose
file went from `58.2 s` to `1m42` at one thread and from `49.3 s` to `1m24`
at four: the new testsets are `9.7 s` / `6.0 s` (the spinning run and its
chain of two jobs `6.5 s` / `2.9 s` of it), the seam's four steps on each
octant in both nestings `+4.3 s` / `+2.8 s`, the coarse octant's refusal
`+1.3 s` / `+1.1 s`, and the rest — about `28 s` at one thread — is the
first compilation of the spinning problem's build and of the frame-dragged
kernel, outside any testset. `test/thread_workload.jl`'s eighth line, the
spinning hole's right-hand side, made the thread test `+5.9 s` / `+3.5 s`.
Two strings changed, each saying so: the `repr` of an `:excised` interior
carries the mixed derivative's field (the nested one prints as before), and
X2b's refusal of the shift into the excised set became a build that counts
frame-dragged axes plus the refusal of a tap without a source.

### Excision of the static spinning hole (step X7)

The `:excised` variant with step X5's rule as X6 built it (under
[Excision](CODE.md#excision-added-2026-10-05), "What step X7 measured"), on
Symmetry's H200s at `Float64`, one GPU and eight CPU threads a row (CUDA.jl
6.4.2, KernelAbstractions 0.9.43, TreeAMR 0.1.7, IMEXRungeKutta 1.3.0, Julia
1.13.1, driver 595.45), from the study's own copy `excision-x7` with `CUDA`
added to its `Project.toml`, through X3's `rows.sbatch` (a file of rows, a
chain on exit 3). Every row is `test/octant_runs.jl backend=cuda case=ks
octant=rotating a=3/5 interior=excised geometry=sphere L=64 roots=2
chunk=1 cfl=1/2`: Kerr-Schild `a = 3/5` (`r₊ = 1.8` at the poles, `1.897` on
the equator, the ring at `0.6`) on the rotating octant `[0, 64]³`, root brick
`2³`, cubes `32, 16, 8` (and `4` at `h = 1/48`), the algebraic source, `q =
4`, `ε_KO = 1/2`, the Gaussian `γ0`, `:msn`, the symmetric mixed derivative,
no blend, the finder every chunk with the spin; the ball `r < r_E`, the core
rule's `r_0 = (0.6 + r_E)/2`, the margin `⌊(r₊ − r_E)/h⌋` cells. The reference
is the rotating octant's `a = 3/5` record (`a06d16`, `a06d24`, `a06d32`,
`a06f32`), whose job scripts say `amplitude=0`: the production rows have no
noise, and the depth scan ran both ways **(proposed in step X7)**. Each row's
`octant.csv`, `records.csv`, `simwatch.toml` and log are in the worktree's
ignored `bin/output/x7/{scan,prod}`, the reference rows' in
`bin/output/x7/ref`; `test/octant_study.jl` reads them (amended in this step:
several directories, `at=`, `a=`, `intervals=`, the frame-dragged rows).

**The depth scan** (jobs 570834, 570844–570852, `h200debugq`): `h = 1/24`
(`N = 96`, 29 blocks, 25.7 M points), `10 M` a row, X5's window. At `10 M`
without noise; *band ℋ* is the evolved shell `[r_E + 3h, 1.90)` inside the
horizon (the layer's: `[3/2, 1.90)`), *L∞* the largest error of the run
(in the closure band), *dragged* the frame-dragged axes and their faces:

| `r_E` | `m` | band points | faces, inflow-like | dragged | least `b/a` | normal | ℋ `[1.90, 2.25)` | error there | ℋ `[2.25, 3)` | band ℋ | L∞ | `J − a` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `:damped`, `r_1 = 3/2` | (7) | | | | | | `2.56·10⁻⁶` | `2.69·10⁻⁶` | `4.57·10⁻⁷` | `5.90·10⁻⁵` | `1.8·10⁻⁴` | `2.65·10⁻⁶` |
| `7/10` | 26 | 1519 | 714, 363 | 197, 77 | `−1.50` | `+0.229` | `5.42·10⁻⁶` | `2.74·10⁻⁶` | `4.58·10⁻⁷` | `1.19·10⁻¹` | `14.3` | `2.32·10⁻⁶` |
| `4/5` | 24 | 1909 | 924, 475 | 194, 77 | `−1.38` | `+0.381` | `5.50·10⁻⁶` | `2.63·10⁻⁶` | `5.21·10⁻⁷` | `2.96·10⁻²` | `2.06` | `1.97·10⁻⁶` |
| `9/10` | 21 | 2380 | 1170, 624 | 202, 76 | `−0.95` | `+0.419` | `9.74·10⁻⁶` | `3.64·10⁻⁶` | `5.65·10⁻⁷` | `9.97·10⁻³` | `0.37` | `0.75·10⁻⁶` |
| `1` | 19 | 2885 | 1413, 784 | 182, 73 | `−0.57` | `+0.378` | `1.30·10⁻⁵` | `4.76·10⁻⁶` | `5.74·10⁻⁷` | `5.08·10⁻³` | `0.11` | `2.83·10⁻⁶` |
| `17/15` | 16 | 3647 | 1824, 1111 | 187, 75 | `−0.58` | `+0.296` | **blows up at `4.5 M`**: `2.94·10⁻⁵` at `4 M` | `6.75·10⁻⁶` | `8.79·10⁻⁷` | `4.5·10⁻³` | `4.45` at `4 M` | — |

- **Four depths are stable; the shallowest is not.** At `r_E ≤ 1` the band's
  norms are constant to three digits from `2 M`, ℋ just outside the horizon
  peaks at `3–5 M` (the start-up) and settles, no value in the band is ever
  non-finite, and the frame-dragged axes keep their bits (`excision_flips =
  0` at every row). `r_E = 17/15` is the same until `2 M`; then its largest
  error grows about `1.9/M` (`0.21`, `0.52`, `1.36`, `4.45` at `2.5`, `3`,
  `3.5`, `4 M`) and the kernel throws a `DomainError` at `4.5 M`.
- **The exterior wants depth, the band wants room from the ring.** ℋ just
  outside the horizon is `2.1×` the layer's at `0.70` and `0.80` and rises
  to `3.8×` and `5.1×` at `0.90` and `1`; the error there is the layer's at
  `0.70` and `0.80` and `1.35×`, `1.77×` at `0.90`, `1`; in `[2.25, 3)` ℋ is
  `1.00–1.26×` and the error `0.96–1.08×` the layer's. The band
  goes the other way: its ℋ falls `4×` from `0.70` to `0.80` and `3×` to
  `0.90`, and its `L∞` from `14` (`0.1 M` outside the ring on the equator)
  to `2.1`. Hence production at `4/5` (X6's proposal).
- **The noise changes nothing near the hole**: the rows with `10⁻⁸` agree
  with these to three or four digits inside `r = 2.25` and fail the same way
  at `17/15`; in `[2.25, 3)` they add `6·10⁻⁸` to ℋ (`5.84·10⁻⁷` at `4/5`),
  X3's noise floor.

**The failure at `r_E = 17/15`, located (diagnosed in step X7)**, on the CPU
with a scratch script (`diag_surface.jl`, the case of `test/octant_runs.jl` on
the small rotating octant `L = 8`, `N = 24`, cubes `4, 2`, so `h = 1/24` at
the hole, printing the points of largest `|u − u_exact|` with their class,
rule bits and variable every half or quarter `M`; the script, its logs and
the scratch copies' patches are in `bin/output/x7/local`). It reproduces the
H200 row's largest error to four digits (`0.5238` at `3 M`). The growing points
are `(3h, 2h, 27h)` and `(2h, 3h, 27h)`, `0.04` cells outside the sphere on
the rim of the excised ball's topmost layer (`z = 27 h`, `x² + y² ≤ 10.8 h²`),
with `−x`, `−y` and `−z` excised and no rule bit; the error is in `Π_xy`,
`Π_yy`, `Π_yz`. The shift's component along `y` there is nearly cancelled by
frame dragging, `b/a = +0.011`. Each row below changes one thing (the largest
error of the run; *grows* means the same points):

| row | change | result |
|---|---|---|
| as built | — | grows from `2.25 M`, `4.5` at `4 M`, `48` at `4.5 M` |
| nested | `mixed = :nested` | grows, `7.2` at `4 M` |
| `ε_KO = 1` | | grows from `3.5 M` at about half the rate, `0.36` at `5 M` |
| blend | `upwind = 1,4` | grows, `7.6` at `4 M` |
| rule to `b/a < 0.02` | the rule's threshold, a scratch copy | the corner takes a bit and grows faster, `19` at `4 M` |
| rule to `b/a < 0.05` | | `8.1` at `3.5 M` |
| rule to `b/a < 1/2` | | `8.2` at `1 M`, at the equator's frame-dragged points |
| `a = 0` | the same lattice and radius (20 cells below `r₊ = 2`), symmetric or nested | `0.012`, stationary to `5 M` |
| `h = 1/32` | the same radius | `0.048`, stationary to `5 M` |
| `r_E = 21/20` | the same topmost disk, `x² + y² ≤ 10.0 h²` at `z = 25 h` | grows from `4.5 M`, `0.87` at `6 M` |
| `r_E = 11/10`, `23/20` | topmost disks of radius `4.6`, `5.7` cells | `0.085`, `0.046`, stationary to `6 M` |
| shave | a scratch copy excising every point whose three inward axis neighbours are excised (217 points of the open octant at `17/15`, 198 at `21/20`, 108 at `4/5`) | `17/15`: `0.117`, `21/20`: `0.154`, stationary to `8 M`; `4/5`: `2.07`, as built |

**The production rows** (jobs 570855 (`p24`), 570863 (`p32`), 570854 and
570879 (`p48`, two jobs of a chain)), and a `:damped` row at `h = 1/48` for
the comparison there, which the reference set has not (`d48`, jobs 570887 and
570898; `r_0 = 3/4`, `r_1 = 3/2`, the reference rows' options otherwise)
**(proposed in step X7)**: `r_E = 4/5`, `r_0 = 7/10`, no noise,
`h = 1/24, 1/32, 1/48` (`N = 96`, `128`, and `96` with the fourth cube, 36
blocks, 31.9 M points, `[0, 4]³` at `1/48` and `[4, 8)` at `1/24`), with a
shell edge added at `r = 3/2` (`shells=3/2,√(2r₊),9/4,3,5,8`) so that
`[3/2, 1.90)` is the region the `:damped` layer evolves inside the horizon
**(proposed in step X7)**. At `24 M`; the band is `[r_E + 3h, 3/2)`; `J − a`,
`M_irr` and their slopes over `8–24 M` from the record; the last column the
H200's seconds per `M`, the analysis every chunk included (the reference rows
`a06d*` ran on the kernel before X4's merge, `d48` on this one):

| `h` | interior | band ℋ | ℋ `[3/2, 1.90)` | ℋ `[1.90, 2.25)` | ℋ `[2.25, 3)` | error `[3/2, 1.90)` | error `[1.90, 2.25)` | error `[2.25, 3)` | `J − a` | `dJ/dt` | `M_irr − M_Kerr` | `dM_irr/dt` | drift of `h_tt` | s per `M` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `1/16` | `:damped` | | `1.26·10⁻³` | `9.13·10⁻⁵` | `4.42·10⁻⁶` | `3.38·10⁻⁴` | `5.24·10⁻⁵` | `1.25·10⁻⁵` | `2.08·10⁻⁵` | `1.14·10⁻⁶` | `+6.8·10⁻⁷` | `−5.7·10⁻⁸` | `1.79·10⁻⁵` | 29 |
| `1/24` | `:damped` | | `5.90·10⁻⁵` | `2.65·10⁻⁶` | `4.88·10⁻⁷` | `1.98·10⁻⁵` | `4.36·10⁻⁶` | `1.50·10⁻⁶` | `5.06·10⁻⁶` | `1.75·10⁻⁷` | `+4.6·10⁻⁸` | `−8.8·10⁻⁹` | `3.00·10⁻⁶` | 138 |
| `1/32` | `:damped` | | `3.95·10⁻⁶` | `6.41·10⁻⁷` | `1.55·10⁻⁷` | `5.02·10⁻⁶` | `1.35·10⁻⁶` | `4.75·10⁻⁷` | `1.53·10⁻⁶` | `5.42·10⁻⁸` | `+1.1·10⁻⁸` | `−2.8·10⁻⁹` | `9.29·10⁻⁷` | 354 |
| `1/32` | `:fitted` | | | `2.04·10⁻⁶` | `1.39·10⁻⁷` | | `1.41·10⁻⁶` | `4.41·10⁻⁷` | `3.06·10⁻⁷` | `2.54·10⁻⁸` | `−8.0·10⁻⁹` | `−3.6·10⁻⁹` | `1.07·10⁻⁶` | |
| `1/24` | excised | `4.52·10⁻²` | `2.91·10⁻⁴` | `5.56·10⁻⁶` | `4.80·10⁻⁷` | `6.21·10⁻⁵` | `3.80·10⁻⁶` | `1.44·10⁻⁶` | `4.15·10⁻⁶` | `1.57·10⁻⁷` | `+3.5·10⁻⁸` | `−9.5·10⁻⁹` | `3.01·10⁻⁶` | 67 |
| `1/32` | excised | `2.75·10⁻²` | `4.31·10⁻⁵` | `7.37·10⁻⁷` | `1.58·10⁻⁷` | `7.47·10⁻⁶` | `1.43·10⁻⁶` | `4.84·10⁻⁷` | `1.57·10⁻⁶` | `5.56·10⁻⁸` | `+1.3·10⁻⁸` | `−2.7·10⁻⁹` | `9.25·10⁻⁷` | 148 |
| `1/48` | `:damped` | | `5.69·10⁻⁷` | `1.26·10⁻⁷` | `3.03·10⁻⁸` | `9.76·10⁻⁷` | `2.74·10⁻⁷` | `1.00·10⁻⁷` | `3.07·10⁻⁷` | `1.07·10⁻⁸` | `+1.0·10⁻⁸` | `−1.5·10⁻¹⁰` | `1.96·10⁻⁷` | 165 |
| `1/48` | excised | `1.43·10⁻²` | `1.18·10⁻⁶` | `1.26·10⁻⁷` | `3.03·10⁻⁸` | `9.80·10⁻⁷` | `2.73·10⁻⁷` | `1.00·10⁻⁷` | `3.06·10⁻⁷` | `1.07·10⁻⁸` | `+1.0·10⁻⁸` | `−1.5·10⁻¹⁰` | `1.96·10⁻⁷` | 112 |

- **Against the layer at the same `h`**, excised over `:damped`: ℋ just
  outside the horizon `2.09×` (`1/24`) and `1.15×` (`1/32`), `ℳ` there
  `2.43×`, `1.89×`, the gauge constraint `1.00×`, `1.04×`, the error `0.87×`,
  `1.06×`; from `r = 2.25` out every constraint and the error within 5 % at
  `1/24` to `r = 5` (ℋ `1.17×` in `[5, 8)` and `1.55×` beyond, the start-up
  pulse still passing) and within 5 % at `1/32` in every shell, most within
  2 %. At `1/48` the excised row is `d48` to three digits in every shell
  from `r = 1.90` out, in `J`, `M_irr` and the drift of `h_tt`; only
  `[3/2, 1.90)`, next to the closures, is `2.07×` in ℋ (the error there
  `1.00×`).
- **Orders**, `1/24 → 1/32 → 1/48` (`test/octant_study.jl … series=p24,p32,
  p48:24,32,48`): ℋ just outside the horizon `7.02/4.36` (the layer's
  `1/24 → 1/32`: `4.94`), the error there `3.39/4.08`, in `[2.25, 3)` ℋ
  `3.87/4.07` and the error `3.79/3.88`; `[3, 5)` straddles the `1/48`
  mesh's `r = 4` and the shells beyond are at `1/24` on it, so their second
  orders mean nothing (`[3, 5)`: `3.91/3.39`). Inside: `[3/2, 1.90)` ℋ
  `6.65/8.88`, the error `7.37/5.01`; the band `1.72/1.61` (ℋ) and
  `2.18/2.41` (error); the run's `L∞`, in the band, `2.06`, `1.10`, `0.60`.
  `J − a` `3.37/4.03` and `dJ/dt` `3.61/4.06`, against the layer's
  (`a06d24 → a06d32 → d48`) `4.16/3.96` and `4.07/3.99`, whose ℋ just
  outside the horizon goes `4.94/4.01`.
- **`M_irr` at `1/48`** is Kerr's to `1.0·10⁻⁸` and flat to `1.5·10⁻¹⁰/M`
  in both rows: below what the finder resolves at `N_ah = 12` (the layer at
  `1/32` has `1.1·10⁻⁸`), so its slope there is not a drift.

**To `64 M` at `h = 1/32`** (job 570863, one job of `2 h 57`, 7104 steps),
against `a06d32` continued to `64 M`; the excised row at `24`, `40` and
`64 M`, the layer's in brackets:

| `t` | ℋ `[1.90, 2.25)` | ℋ `[2.25, 3)` | error `[1.90, 2.25)` | error `[2.25, 3)` | `J − a` | `dJ/dt` since the last row | `M_irr − M_Kerr` | drift of `h_tt` |
|---|---|---|---|---|---|---|---|---|
| `24` | `7.37·10⁻⁷` (`6.41`) | `1.58·10⁻⁷` (`1.55`) | `1.43·10⁻⁶` (`1.35`) | `4.84·10⁻⁷` (`4.75`) | `1.57·10⁻⁶` (`1.53`) | `5.56·10⁻⁸` (`5.42`), over `8–24` | `+1.3·10⁻⁸` (`+1.1`) | `9.25·10⁻⁷` (`9.29`) |
| `40` | `7.41·10⁻⁷` (`6.46`) | `1.61·10⁻⁷` (`1.58`) | `2.02·10⁻⁶` (`1.93`) | `8.17·10⁻⁷` (`8.00`) | `2.41·10⁻⁶` (`2.35`) | `5.25·10⁻⁸` (`5.11`) | `−8.9·10⁻⁸` (`−8.9`) | `1.19·10⁻⁶` (`1.19`) |
| `64` | `7.43·10⁻⁷` (`6.48`) | `1.61·10⁻⁷` (`1.59`) | `2.99·10⁻⁶` (`2.88`) | `1.35·10⁻⁶` (`1.32`) | `3.69·10⁻⁶` (`3.58`) | `5.30·10⁻⁸` (`5.16`) | `−2.71·10⁻⁷` (`−2.67`) | `1.57·10⁻⁶` (`1.56`) |

The hole region is stationary from `3 M` to `64 M`: the band's ℋ is
`2.75·10⁻²` to three digits at every row, ℋ just outside the horizon rises
by 1 % in `40 M` as the layer's does, and only the errors grow — the slightly
different hole spreading outward with the drift of `J`, `1.03×` the layer's
in every interval. All 65 finds succeed, and `excision_flips = 0` at all 65
rows.

**What it costs on the H200** (job 570857, `h200debugq`; `x7_device.jl`,
X6's device script reduced to the timings, minimum of ten, the `:damped`
layer at `r_0 = 3/4`, `r_1 = 3/2` on the same mesh):

| mesh | points | zone points | frame-dragged axes | right-hand side | zone kernel | the rule's launch | `:damped` right-hand side |
|---|---|---|---|---|---|---|---|
| `1/24` (`N = 96`, 29 blocks) | 25.7 M | 1909 | 194 | 64.4 ms (2.51 ns a point) | 1.23 ms (1.9 %, 647 ns a zone point) | 1.62 ms (2.5 %) | 136.1 ms (5.31 ns; excised `0.47×`) |
| `1/32` (`N = 128`, 29 blocks) | 60.8 M | 3272 | 365 | 136.3 ms (2.24 ns) | 2.16 ms (1.6 %, 661 ns) | 3.89 ms (2.9 %) | 283.6 ms (4.66 ns; `0.48×`) |
| `1/48` (`N = 96`, 36 blocks) | 31.9 M | 6967 | 870 | 82.1 ms (2.58 ns) | 2.51 ms (3.1 %, 360 ns) | 2.63 ms (3.2 %) | 169.9 ms (5.34 ns; `0.48×`) |

The excised right-hand side is X4's `2.9 ns` a point; the `:damped` one is
X4's `5.3 ns` (`main`'s `:damped` branch still spills), so on this kernel an
excised right-hand side costs about half a `:damped` one. The zone kernel and
the rule's launch together are 4.4 %, 4.5 % and 6.3 % — **X6 predicted 0.6 % and 0.3 %
for the rule** at `1/24` and `1/32` from a `0.47 ms` launch at 72 points: the
launch is not a fixed latency but grows with the frame-dragged points and the
`96³` or `128³` block that holds them. Per `M` of evolution, the analysis
every chunk included: `62` (scan) and `67 s` (production, one shell more) at
`1/24`, `148 s` at `1/32`, `112 s` at `1/48`, against X6's `~140`, `~380`,
`~350 s`, and `165 s` for `d48`, the `:damped` layer at `1/48` on the same
kernel: an excised run is `0.68×` a layer's. The start-up is about `125 s`,
a checkpoint chunk `+60–130 s`. The
whole study took 8.2 H200-hours (`sacct`), the `1/32` row 3 of them.

**The suite.** No source changed in this step (`test/octant_study.jl`, which
is not in it, was extended): **7001 assertions in 17m42** at one thread and
**7009 in 13m41** at four (the two at once, at a load of 4–7, after every row
had finished), X6's counts; the same counts in 17m43 and 13m46 at the step's
start.

## The interior's questions, opened in step 5 and closed through step 8′

From `CODE.md`'s "Open questions", whose other questions stay there.

**Opened in step 5, and the one thing that stands between this package
and its own proof of concept: a spherical frozen core cannot be used with
`Harmonic(M, 9/10)`.** The chart's singular set is the equatorial disk of
coordinate radius `a`, the horizon's smallest coordinate radius is
`√(M² − a²)`, and a ball fits between them only where `a < M/√2 ≈ 0.707`.
The two candidate answers — key the interior on the chart's own
spheroidal radius `R`, or run G5 at `a = 0.7` — are written out under
[The interior](CODE.md#the-interior-a-pointwise-damping-layer). Step 8 has to
pick one before it can run the case this document is named for; nothing
in G4 depends on it, and Kerr-Schild at `a = 9/10` runs today.

**Answered in design review on 2026-09-23 — neither, and PLAN.md's steps
8a–8g build and measure the answer.** The layer is keyed on the *found*
horizon's offset surface `r_1(n̂) = r_h(n̂) − m h` — a sphere on the
*tracked* radii does not contain the disk either, since `0.436 − m h` is
far inside it, while the offset surface does once `m h < 0.1 M` on the
equator — and it relaxes toward a regular fit of the evolved state
instead of the analytic solution, so no singular set has to be contained
and no analytic interior is needed: the same treatment serves a spinning,
a moving and a newly found horizon. The review found three more things
the steps rest on and measure: `ρ_max = 1/dt` is a *grid* rate (about
`107/M` on the suite's fixture), which makes today's layer a paste two
cells inside `r_1` that survives to `50 M` only because its target is
exact — a generic target needs a thick ramp at a physical rate (step 8c
measured it, and from step 8c′ `4/M` is the default, decided 2026-09-23);
the Lorentzian metrics are not convex in `g_ab` (the angular mean of
Kerr-Schild `g_ab` inside the horizon has Euclidean signature), so every
blend and clamp is made in ADM variables; and the discrete scheme's
grid-scale modes have *outgoing* group velocity inside the horizon (every
centered first-derivative stencil annihilates the Nyquist mode, so the
shift advection does not act on it), attenuated only by dissipation —
step 8a computes their penetration length for this package's stencils
before anything is built, and it is what the margin `m` is measured
against. `a = 7/10` stays the fallback for G5 if the last row of step
8f's matrix does not fit a node.

**The offset surface is built (step 8d).** A case whose interior is a
`FittedSpec` keys its layer on the depth below the tracked horizon's offset
surface, with the track's center, velocity and shape (to `lmax_shape`)
updated from every find — under [The
interior](CODE.md#the-interior-a-pointwise-damping-layer), "The tracked geometry";
on the static hole it is the analytic layer to six digits. What it does
not do yet is the second half of the answer: with an analytic target the
core surface must still contain the chart's singular set, which harmonic
Kerr at `a = 9/10` refuses at every `h` — so the proof-of-concept case waits
for step 8e's `:fitted` target, and its shape needs `lmax = 12` for a tenth
of a cell at `h = 5/256` where Kerr-Schild's needs `4` (measured in step
8d).

**The fitted target is built, and the proof-of-concept chart's initial data
exists (step 8e).** The `:fitted` variant relaxes the tracked layer toward
a cached fit of the evolved state (`CODE.md`'s "The fitted target", pieces
8–12): on the static fixture it is `:damped`'s to `1.24×` by `1 M` — `4.3×`
at `0.15 M`, the initial data's curvature kink at `r_1` — and a boosted hole
tracks to `4e−3` cells. Harmonic Kerr at `a = 9/10`, `m = 4`, `h = 5/256`
has, for the first time, initial data that is finite and a metric at every
point; **its run ends in the first chunk**, because the offset surface's
data (the ring `0.02 M` inside `r_1` at the equator, `|Π| ~ 4e4` there
against `30` on the axis) is not representable by a polynomial of degree
`L + 2` to the accuracy the evolved stencils need. What would unblock it —
`Π̃ = (α/√γ)Π` as the fitted momentum, `L ≥ 12`, a finer equator — is step
8f's to measure, and `a = 7/10` and 8g's excision remain the fallbacks.

**Closed in step 8f: G5 runs at `a = 7/10`, and harmonic Kerr at `a = 9/10`
waits with its price written down (decided 2026-09-23** by the orchestrating
session, with the decision delegated by Erik, before step 8f started; step
8f's host-side probe, `hole_runs.jl generic=probe`, measured the numbers
below**).** The chart is blocked by resolution before it is blocked by the
interior:

- **The measured spacing.** At `h = 5/256` and `m = 4` — the finest the
  proof-of-concept mesh of 2472 blocks reaches, and the spacing at which the
  offset surface's equator `r_1 = 0.921` first lies outside the ring at
  `0.9` — the analytic solution's own second difference at the first evolved
  point on the equator is `2.4e9` against `420` on the axis, its length scale
  `0.008 M`, under half a cell (step 8e); a spacing that resolves it by five
  cells is `h ≲ 5/1024` on the equator.
- **What step 8f changed does not rescue `5/256`.** Fitting `Π̃ = (α/√γ)Π`
  and `L = 12` brings the fit's kink at the first evolved point from `8.8e5`
  to `366` on the axis — the analytic `420` — and from `2.0e7` to `2.5e4` at
  45° (`3.1e3` at `L = 16`), against an analytic `107`: still 30–230 times
  off the equator, and on the equator itself the data are not resolved by
  any fit (`1.5e9` against `2.4e9`). The initial data at `L = 12`, `Π̃` is
  finite and a metric at all 1 265 664 points (`min det γ = 1.6`, `min α =
  0.219`), one right-hand side is finite (`max |du| = 7.9e8`, at the offset
  surface over the ring, `|z| = 0.08`), and the run ends in its first chunk,
  before `t = 1/400 M`, in a degenerate metric — as step 8e's did at `L = 8`
  (measured in step 8f).
- **The node run's price** (measured cost per point, `2.5 µs` a thread on
  this machine and twice that on a Symmetry core; `dt = 5.4e−3 M` at
  `5/256`, `λ_max = 0.906`, a quarter of it at `5/1024`): with `5/1024` on
  an equatorial band (`0.75 ≤ ρ ≤ 1.05`, `|z| ≤ 0.2`) and `5/256` elsewhere,
  **23 080 blocks, 1.2e7 points, 28 GB, one right-hand side `0.93 s` on 64
  node threads, `0.76 h` a `M`, `38 h` for `50 M`**; with `5/1024` inside the
  whole sphere `|x| ≤ 1`, 90 112 blocks, 111 GB, `3.0 h` a `M` and `149 h`
  for `50 M` (measured in step 8f). No checkpointing exists (`Possible
  extensions`), so either is one uninterrupted job longer than any queue's
  day **(amended 2026-10-01**: it exists now, so either is a chain of
  jobs**)** — and it presumes a fit good enough off the equator, which none of
  step 8f's is, or a margin that depends on direction.

So `a = 9/10` in the harmonic chart is a research item of its own — a
direction-dependent margin or a finer equator, a fit that holds 45°, and
checkpointing for a multi-day run (there from 2026-10-01) — and not a row of
the matrix. **G5 runs
at `a = 7/10`** (`CODE.md`'s fallback since step 5: `√(M² − a²) = 0.714 >
a`), measured by step 8f's `h7` row at `h = 5/256`.

**The fitted target's host half is built (step 8e-i).** `src/fit.jl` fits
`(log α, β^i, γ_ij, Π_ab)` on the offset surface with a polynomial of degree
`L + 2 cont`, one QR for all twenty variables, and sweeps the result for
validity inward to the center — under [The
interior](CODE.md#the-interior-a-pointwise-damping-layer), "The fitted target". It
reproduces the static hole to roundoff from `L = 2` and to the
interpolation order from the state, is a metric on both Kerr-Schild holes,
and on harmonic `a = 9/10` at `h = 5/256` is a metric fitted to value and
slope (`cont = 1`, `L ≥ 8`) and **not** fitted to curvature (`cont = 2`, at
every `L` to 16) — which is what 8e-ii's plan for that chart's initial data
asks for (measured in step 8e). Two findings for 8e-ii and G5: a moving
hole's shift needs its constant term (`shift_constant = true`), and the
evaluator costs `2.2 µs` a point at `L = 8`, twenty-three `u_exact`s.

**The moving layer's trailing side (opened in step 8, answered in step
8′).** On G5's chart the `:fitted` layer moving at `0.3` exported error
through the side it leaves. Step 8′ tried the three levers step 8 named,
each measured as the masked L2's excess over the hole at rest (`a = 0` at
`M/2`; G5's chart at `M/2` and on the crossing):

| lever | `a = 0`, `M/2` | G5, `M/2` | G5, `13 M` |
|---|---|---|---|
| none | `7.3e−2` | `0.194` | `2.069` |
| the side-dependent ramp, `trail_ramp = 9/10` | `2.9e−2` | `0.059` | `0.545` |
| the refill every `1/16` cell | `7.2e−2` | (not run: no effect on `a = 0`) | — |
| the exact target (the fit in the kernel) | `7.3e−2` | `0.195` | — |
| the analytic `:damped` layer at `20/M` (control) | `2.6e−2` | (no analytic layer) | — |

**The ramp answers it** (proposed in step 8′: `trail_ramp = 9/10` for every
moving `:fitted` run, off by default so that a static one is unchanged): it
takes the `a = 0` excess to the analytic control's, and on G5's chart
removes the side asymmetry entirely and cuts the crossing's excess by
`3.8×`. **It does not make G5's error the static run's**: at `13 M` the
moving hole is `2.05×` the resting one and still growing faster (`0.11` a
`M` against `0.02`), symmetrically on both sides of the layer, and the
horizon's `J` drifts by `0.013` a `M` with and without the ramp. So the
remaining problem is not an export through the trailing side, and excision
(step 8g) — which replaces the layer, not a side of it — is not indicated
by it. **Recommendation (proposed in step 8′): more interior work before
excision's price**, in this order: the spin drift of the moving spinning
hole (it is absent on `a = 0` and at rest; the fit's `L = 12` and the
target's rate are the first suspects, and a boosted hole at rest in its
own frame is the cheapest test), then the uniform growth — which the
resting `:fitted` hole shows too at a fifth of the rate. Step 8g's
host-side test is the right next step only if the spin drift turns out to
be the layer's.
