# Singularity handling — the measured results

Moved here from `CODE.md` on 2026-10-08, verbatim. `CODE.md` keeps the
design of the interior — [The interior: a pointwise damping
layer](CODE.md#the-interior-a-pointwise-damping-layer), the variants, the
profiles, the range projection, the tracked geometry and the fitted target —
and the settings these runs recommend for a single hole, [Single black holes:
recommended settings](CODE.md#single-black-holes-recommended-settings-added-2026-10-08).
This file keeps the runs that made them: what was measured, with which jobs
and scripts, and what each result decided. Prose that names a section
("under …", "above", "below") names one of `CODE.md`'s unless the section is
here; links have been pointed at the right file.

- [Step 5: the static hole's interior and the driver (G4a)](#step-5-the-static-holes-interior-and-the-driver-g4a)
- [Step 8a: what crosses the horizon from inside it](#step-8a-what-crosses-the-horizon-from-inside-it)
- [Steps 8b–8′: the generic interior and the moving hole](#steps-8b8-the-generic-interior-and-the-moving-hole)
- [Single holes on the octant, `a = 0` to `9/10` (2026-10-02 to 2026-10-07)](#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07)
- [The boosted Kerr-Schild hole at binary speed (2026-10-20)](#the-boosted-kerr-schild-hole-at-binary-speed-2026-10-20)
- [The moving `:damped` layer's target rate (2026-10-21)](#the-moving-damped-layers-target-rate-2026-10-21)
- [A spinning hole along its axis (2026-10-21)](#a-spinning-hole-along-its-axis-2026-10-21)
- [Aligned spin and velocity: which holes the algorithm holds (2026-10-22)](#aligned-spin-and-velocity-which-holes-the-algorithm-holds-2026-10-22)
- [Spin across the motion: the bitant (2026-10-24)](#spin-across-the-motion-the-bitant-2026-10-24)
- [The damped harmonic gauge rolled on (step B1, 2026-10-25)](#the-damped-harmonic-gauge-rolled-on-step-b1-2026-10-25)
- [The tracked analytic target (step B2, 2026-10-26)](#the-tracked-analytic-target-step-b2-2026-10-26)
- [A layer-free interior (step I1)](#a-layer-free-interior-step-i1)
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
  (Amended 2026-10-16: moving `r_0` to `1`, away from the ring, makes `1/48`
  worse, not convergent — "`a = 9/10` to `128 M`", below.)
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

**`a = 9/10` to `128 M`, and the ring rule tested (measured 2026-10-14/16**,
jobs 574817, 574819 → 575441 and 575609 → 575990, `a09-L128-T128` — `main` at
`fbfd416` with CUDA added; run from the TreeExcision session, checked here
against its CSVs, which are in `.claude/data/a09-L128-T128`**)**. Three rows on
the rotating octant grown to `L = 128` (one more coarse level, `h = 2/3`), to
`128 M`, everything else the rows above: `q = 4`, `cfl = 1/2`, `ε_KO = 1/2`,
`ρ_max = 4/M`, the algebraic source, the finder every `M`, one H200 each,
checkpointed every `2 M` and resubmitted at the wall-time limit.

- `d48r1`: `:damped` at `h = 1/48`, `r_0 = 1`, `r_1 = 5/4` — the
  recommended-settings row, which the ring rule predicted convergent. The
  ring's length `L = (R² − a²)/R` at `R = 1` is `9.1` cells, against `4.7` at
  `a09d48`'s `r_0 = 19/20`; the ramp is `12` cells, against `14.4`. `roots = 2`,
  `N = 96`, cubes `64, 32, 16, 8, 4`, 43 blocks, 38.0 M points, `220 s/M`.
- `d96r1`: the same layer at `h = 1/96`, A's and B's mesh one level wider:
  `roots = 4`, `N = 48`, cubes `64, 32, 16, 8, 4, 3`, 533 blocks, 58.9 M
  points, `458 s/M`. Its first attempt (575014), with `N = 96`, ran out of
  GPU memory at 304 M points. At `N = 96` the `1/48` level's blocks are two
  units wide, so the cube `3` refined all of `[0, 4]³`, and 2:1 balance
  carried a level out to the root. **The finest cube must be a whole number
  of its parent level's blocks.**
- `f96m16L`: `:fitted` at `m = 16`, `n_L = 36`, `lmax_fit = 12`,
  `fit_cont = 1` (570817's row, continued past `32 M`), on `d96r1`'s mesh,
  `457 s/M`.

At `128 M`, point-weighted L2 in `test/octant_runs.jl`'s shells (the first from
the equatorial radius `1.695`):

| | `ℋ` in | `ℋ [1.69, 2.25)` | `ℋ [2.25, 3)` | `ℋ [3, 5)` | error `[1.69, 2.25)` | `J` | `M_irr` |
|---|---|---|---|---|---|---|---|
| `d48r1` | `1.48·10⁻²` | `8.94·10⁻⁵` | `2.53·10⁻⁷` | `1.86·10⁻⁸` | `2.20·10⁻⁵` | `0.9000036` | `0.8473145` |
| `d96r1` | `1.02·10⁻⁴` | `4.63·10⁻⁸` | `5.04·10⁻⁹` | `3.02·10⁻⁹` | `4.39·10⁻⁷` | `0.9000072` | `0.8473164` |
| `f96m16L` | `2.11·10⁻³` | `8.83·10⁻⁵` | `7.75·10⁻⁸` | `7.23·10⁻⁹` | `1.01·10⁻⁵` | `0.9000092` | `0.8473168` |
| `a09d48` (`24 M`) | `3.70·10⁻³` | `2.38·10⁻⁵` | `8.77·10⁻⁸` | `8.29·10⁻⁹` | `5.44·10⁻⁶` | `0.9000093` | `0.8473164` |
| B, `a09d96` (`24 M`) | `8.11·10⁻⁶` | `4.16·10⁻⁸` | `4.28·10⁻⁹` | `2.82·10⁻⁹` | `1.06·10⁻⁷` | `0.9000069` | `0.8473165` |

- **The ring rule is refuted at `a = 9/10`.** `r_0 = 1` at `1/48` is not
  convergent. Against `a09d48`'s `r_0 = 19/20` it is worse: `3.7×` just outside
  the horizon and `4×` in the band inside it, at `8 M` and still at `24 M`.
  The deeper `L = 128` box does not explain that, since `f96m16L` is 570817
  to three digits in every shell. What did change is the ramp, `12` cells
  against `14.4`, and a thinner ramp costs `:damped` about as much at `a = 0`
  (12 → 18 cells lowered `ℋ` `3×` there).
  - `1/48 → 1/96` at fixed radii is `1935×` in `ℋ` just outside the horizon,
    and `50×` in `[2.25, 3)` and in the error just outside. Fourth order
    would be `16×`.
  - Outside `r = 3` the two meshes have the same spacing, so their ratios
    there (`6.2`, `2.4`, `0.9`) measure only what reaches them from inside.
  - So the coarsest `:damped` resolution at `a = 9/10` is `1/96`, and the
    ring's 8 cells are not enough.
- **At `1/96` the layer's position does not matter outside the horizon.**
  `d96r1` and B agree:
  - to `7 %` just outside the horizon at `24 M` (`4.45` against
    `4.16·10⁻⁸`);
  - to three digits from `2.25` out;
  - to `0.5 %` in the error just outside the horizon.

  Inside the horizon `r_0 = 1` is worse: the evolved band's `ℋ` is `13×` B's
  (`1.02·10⁻⁴` against `8.1·10⁻⁶`), and the layer residual is `0.87`
  against `1.50`. Whether `1/96` itself is in the convergent regime needs a
  finer partner, which neither study has run. (Amended 2026-10-19: it is,
  at fourth order outside the horizon — "`a = 9/10` `:damped` converges at
  `1/96`", below.)
- **All three rows are stationary to `128 M`**, and none of them grows. All 387
  finds succeed, `:fitted`'s fits are valid at every row (residual
  `0.0023`), and the projection never fires.
  - `d48r1`: `ℋ` just outside the horizon is `8.94·10⁻⁵` at `8 M` and at
    `128 M`.
  - `d96r1`: it saturates at `4.6·10⁻⁸` from `32 M`, and the shells from `3`
    out are flat to two digits.
  - `f96m16L`: `ℋ` just outside the horizon is flat at `8.83·10⁻⁵` from
    `16 M`. The leak 570817 showed spreading outward between `16` and `32 M`
    settles: `[2.25, 3)` is `7.2–7.7·10⁻⁸` and `[3, 5)` is `4–7·10⁻⁹` from
    `24 M` to `128 M`.
  - The far field `r ≥ 8` is `1.3·10⁻¹⁰` in every row at every time.
- **`:damped`'s error grows linearly at `1/96`, as at `a = 3/5`.**
  - Just outside the horizon it goes `5.2·10⁻⁸` (`8 M`) → `2.3·10⁻⁷`
    (`64 M`) → `4.4·10⁻⁷` (`128 M`), `3.3·10⁻⁹/M`.
  - `J` drifts at a constant `+2.5·10⁻⁹/M`, B's rate, and `M_irr` falls with
    it at `−9·10⁻¹⁰/M`. This is the slightly different hole spreading outward.
  - `d48r1`'s `J` wanders: `−7·10⁻⁶` below B's to `24 M`, then rising at
    `4·10⁻⁸/M`.
  - `f96m16L`'s `J` sits `+2·10⁻⁶` above B's and from `32 M` falls at
    `−1·10⁻⁸/M`, while its `M_irr` rises at `+4.5·10⁻⁹/M` from `64 M`.
- **`:fitted` against `:damped` at `1/96`, at `128 M`.**
  - `ℋ` just outside the horizon is `1900×`, the same at every time.
  - The error just outside it is `23×`, against `95×` at `24 M`: `:damped`'s
    error grows and `:fitted`'s stays at `1.0·10⁻⁵` from `16 M`. A comparison
    by the error must name its time, as at `a = 3/5`.
- **What it costs with `main`'s kernel**:
  - `220 s/M` at `1/48` (38.0 M points) and `457–460 s/M` at `1/96`
    (58.9 M), for both variants.
  - Per point that is `1.6×` and `2.2×` faster than the same rows before the
    rewrite (`5` and `15 min/M` on 31.9 and 52.8 M points).
  - That is not the gauge wave's `8×`: a hole's step is the layer's kernel,
    which was not rewritten, and the prolongation on a refined mesh.

For scale, the same session's Z4c puncture in TreeBSSN (job 574812) holds `ℋ`
at `5–8·10⁻⁶` in every shell by `128 M`, with `J = 0.89998`. That is a
different chart, slice and normalisation, at `h = 1/256`, so it is not a
like-for-like comparison. Its horizon sits at coordinate radius `0.41–0.54`.

**`a = 9/10` `:damped` converges at `1/96` (measured 2026-10-19**, jobs
578411–578413, `a09-conv` — `main` at `59d4c6f` with CUDA added and the
SimWatch thumbnails; the CSVs are in `.claude/data/a09-conv`**)**. Three
resolutions of one mesh, everything scaled together, B's layer and settings
otherwise: `r_0 = 19/20`, `r_1 = 5/4`, `q = 4`, `cfl = 1/2`, `ε_KO = 1/2`,
`ρ_max = 4/M`, the algebraic source, the finder every `M`, no noise, the
rotating octant `L = 64`, to `32 M`, one H200 each.

- **The mesh is B's near the hole and thinner outside it.** `roots = 4`,
  `radii = 32,16,8,4,4,3`, 2381 blocks: `h_fine` in `[0, 3]³`, `2 h_fine` in
  `[0, 4]³`, and every coarser level the thinnest band TreeAMR's 2:1 balance
  allows — four blocks a side less the two of the next level, 56 blocks. B's
  `N = 48` makes each such band `6.2 M` points, `31` of its `52.8 M`. With
  `N = 24` the bands are `0.77 M`, each level ends at half B's radius outside
  `r ≈ 5.5` (`h/r ≈ 1/96` against `1/192`), and the mesh is `32.9 M` points at
  `1/96`, `0.62×` B's. The resolutions differ only in `N`, so the blocks, and
  the regions each level covers, are the same in all three.
- **Why these three.** The finest cube alone is `24 M` points at `1/96` and
  would be `191 M` at `1/192`; one H200 holds about `100 M` here (`c128`
  used `123 GB` of `141`), and the package has no MPI. `1/72 → 1/96 → 1/128`
  are two steps of `4/3`, `3.16×` each at fourth order.

| | `h` | `N` | points | GPU memory | `s/M` | ns per point and step |
|---|---|---|---|---|---|---|
| `c72` | `1/72` | 18 | `13.9 M` | `37 GB` | `137` | `39` |
| `c96` | `1/96` | 24 | `32.9 M` | `64 GB` | `322` | `29` |
| `c128` | `1/128` | 32 | `78.0 M` | `123 GB` | `803` | `23` |

Smaller blocks cost more per point (the ghost fill and the prolongation),
but `c96` is still `1.43×` faster than `d96r1` (`460 s/M`) at the same
spacing near the hole. All 99 finds succeed and all three runs are
stationary.

The convergence orders, `1/72 → 1/96` and `1/96 → 1/128`, of the
point-weighted L2 in `test/octant_runs.jl`'s shells (the first from the
equatorial radius `1.695`):

| | error, `8 M` | error, `32 M` | `ℋ`, `8 M` | `ℋ`, `32 M` |
|---|---|---|---|---|
| band inside the horizon | `5.65`, `4.15` | `5.29`, `4.13` | `10.3`, `8.3` | `10.2`, `8.3` |
| `[1.69, 2.25)` | `4.07`, `4.03` | `4.03`, `4.03` | `6.90`, `4.02` | `6.86`, `4.02` |
| `[2.25, 3)` | `4.00`, `4.02` | `4.03`, `4.03` | `4.03`, `4.00` | `4.03`, `4.01` |
| `[3, 5)` | `4.01`, `4.02` | `4.03`, `4.03` | `3.93`, `3.87` | `3.85`, `3.79` |
| `[5, 8)` | `4.02`, `4.02` | `4.04`, `4.04` | `4.05`, `4.03` | `4.05`, `4.03` |
| `r ≥ 8` | `4.01`, `4.01` | `4.05`, `4.04` | `4.10`, `4.08` | `4.10`, `4.08` |

- **`1/96` is in the convergent regime.** From `1/96` to `1/128` every shell
  outside the horizon converges at order `4.0` (`3.8–4.1`) at every time
  from `1 M` to `32 M`, in the error against Kerr-Schild and in `ℋ`. So does
  the drift of the horizon: `M_irr(32 M) − M_irr(1 M)` is `4.80`, `1.50` and
  `0.47·10⁻⁷` (orders `4.03`, `4.02`), and `J`'s is `12.8`, `4.09` and
  `1.26·10⁻⁸` (`3.96`, `4.09`). `J − a = +6.83·10⁻⁶` at `1 M` in all three
  runs: that offset is the finder's (`N_ah = 12`), not the evolution's.
- **`1/72` is in it too, except just outside the horizon.** There `ℋ` at
  `1/72` is too large from about `4 M` on (`2.3·10⁻⁷` against the
  `1.0·10⁻⁷` fourth order would give), while the error there converges at
  `4.0`. The excess comes from inside, where `ℋ` converges faster than fourth
  order (below): `1/48` (`d48r1`, `a09d48`) is outside the regime, and
  `1/72` is at its edge.
- **Inside the horizon the evolved band is not yet asymptotic in `ℋ`.** It
  falls `19×` and then `11×` a step (orders `10` and `8.3`), from
  `1.5·10⁻⁴` at `1/72` to `7.5·10⁻⁷` at `1/128`. The band's error settles
  toward fourth order (`5.3–5.7`, then `4.13–4.15`). The layer residual —
  the layer's distance from the truth, large by construction — falls at
  order `1.8` (`2.51`, `1.50`, `0.88` at `32 M`).
- **The error grows linearly at every resolution**, `3.7·10⁻⁹/M` just
  outside the horizon at `1/128` (`2.0·10⁻⁸` at `8 M`, `1.1·10⁻⁷` at
  `32 M`), and converges at fourth order at every time: it is accumulated
  truncation error, not a resolution-independent drift.
- **`c96` against B** (same spacing near the hole, B's outer levels twice as
  wide, `L = 64`): the band inside the horizon is the same to `2 %` and the
  layer residual to three digits; `ℋ` just outside the horizon is `0.76–0.92×`
  B's. The error is not: B's coarser outer levels in `c96` raise it `1.25×`
  just outside the horizon at `8 M` and `2.4×` at `24 M`, `2.9–8.7×` in
  `[2.25, 8)` and `12–17×` beyond `r = 8`, where the factor `16` is the
  halved levels' `2⁴`. The error at a given radius is set by the mesh
  outside it as the run goes on; for the interior's question the mesh can be
  thin outside, for the exterior's accuracy it cannot.

## The boosted Kerr-Schild hole at binary speed (2026-10-20)

This is `CODE.md`'s proposal of 2026-10-02 — once the algebraic source
exists, run a boosted Kerr-Schild hole across the mesh with the analytic
layer — made at the speed of one member of a binary a few orbits before
merger, with the static hole's settings from the octant above. It is the first boosted Kerr-Schild
hole this package has evolved: step 8's moving holes were all in the boosted
harmonic chart (`v = 0.3`, `q = 2`, on CPUs, `13 M` at most).

**The speed, `v = 1/5`.** An equal-mass binary of two `M = 1` holes
(`M_tot = 2`) at a separation of `d = 8–10 M_tot` is `3.6–6.3` orbits before
merger by the quadrupole formula. The Newtonian speed of each hole is
`½√(M_tot/d) = 0.158–0.177`, and quasi-circular puncture data at `D = 10` and
`8 M_tot` have `p/M_tot ≈ 0.095` and `0.109`, that is `v ≈ 0.19–0.21`. `1/5`
is exact in `Rational`. The contraction is `√(1 − v²) = 0.97980`, so the
horizon's coordinate extent along the boost is `1.960`.

**The setup** (commit `1b8a4ea`, `test/octant_runs.jl`):

- Kerr-Schild `a = 0`, translated and boosted, `kerr_schild_case(; v = 1/5,
  x0 = −4)` — the hole at `(−4 + t/5, 0, 0)` — with the algebraic source at
  the hole's velocity (`gauge_source = :algebraic`).
- The quadrant `[−64, 64] × [0, 64]²` (`octant = :quadrant`: mirrors at
  `y = 0` and `z = 0`, Dirichlet at the other four faces), root brick
  `(8, 4, 4)`.
- A frozen capsule mesh, `hole_forest(; shape = :cube, path = (8, 0, 0))`:
  cubes of radius `32, 16, 8, 4` swept along the segment `x ∈ [−4, 4]`, 856
  blocks. `N = 16, 24, 32` give `h = 1/16, 1/24, 1/32` at the hole and
  `3.51`, `11.83` and `28.05 M` points. The hole never leaves its finest
  level, and nothing regrids.
- `q = 4`, `cfl = 1/2` (`56`, `84` and `112` steps per `M`: the step is set by
  the hole's fastest speed, `1.75`), `ε_KO = 1/2`, the Gaussian `γ0` about
  `c(t)`, chunk `M/2`, the finder every chunk (`N_ah = 12`), no noise. From
  `t = 0` to `40 M`, `x = −4` to `x = +4`: two horizon diameters.
- **The control** is the same hole at rest at `x = −4` on the same mesh.
- **`:damped`**: `r_0 = 5/8`, `r_1 = 23/16`, fixed in `M`. The octant's
  `r_1 = 3/2` is only `7.4` cells inside the *contracted* horizon at `1/16`;
  `23/16` is `8.4`, with a ramp of `13` cells (`19.5` and `26` at `1/24` and
  `1/32`). `ρ_max = 4/M` (the default) or `8/M`.
- **`:fitted`**: the octant's setup, `m = 16`, `n_L = 20`, `fit_cont = 2`,
  `lmax_fit = 8`, with `trail_ramp = 9/10` (`bf24`, `bf32`) or without it
  (`bf24n`).
- One H200 a run on Symmetry, from a copy with `CUDA` added: `boosted-ks` (the
  first moving run, job 579652, and the resting controls at `4/M`,
  579656–579657) and `boosted-ks-2` (everything after the fix below,
  579664–579698). The CSVs, logs and SimWatch files are in
  `.claude/data/boosted-ks/{ks1,ks2}`, and `julia --project=.
  test/boosted_study.jl times=1,4,8,16,24,32,40` prints every table below
  from them.

The norms are point-weighted L2 in shells about the hole's *current* center
`c(t)` — `[2, 2.25)`, `[3, 5)` and `[5, 8)` — and in the evolved band inside
the horizon, each also split into the trailing half (`(x − c(t)) · v̂ < 0`) and
the leading one. "Error" is the distance from the boosted analytic solution,
`ℋ` the Hamiltonian constraint, and "global" the whole evolved region.

**The first moving run died, and why (measured 2026-10-20**, job 579652**)**.
`:damped` at `4/M`, `h = 1/16`. The layer residual grew about linearly —
`6.3`, `10.9`, `14.8`, `19.1`, `30.0`, `37.8` at `t = 1/2, 1, …, 3`, against
`0.12` at rest — and the trailing half carried `16×` the leading half's error
in the band inside the horizon and `21×` in `[2, 2.25)` at `3 M`. At `3.2 M` a
`DomainError` in a kernel on the GPU ended it. The analytic core had `du = 0`.
It kept the solution about `c(0)` and released it into the layer on its
trailing side, off the current solution by an amount that grew with the
distance travelled, and the run died when the points that were at the center
at `t = 0` left the core, at `r_0/v = 3.1 M`. That is step 8's `:frozen`
failure, met by `:damped`. **The fix** (commit `ae3e7e8`; `CODE.md`, the
frozen core's paragraph under "The interior"): a moving `:damped` core relaxes
at `ρ_max` toward the layer's target at the core rule's point, the metric in
the ADM variables (`core_relaxation_rate`) and `Π` linearly. Relaxing `g_ab`
linearly ended the run in its first chunk: the straight path between the
targets on the two sides of the sphere `r_0` crosses a Euclidean metric. A hole
at rest is unchanged bit for bit (`core_follows`), and the fixed run reached
`40 M`.

**Every run after the fix reached `40 M`, and every find succeeded**
(81 a run). What a run costs on
one H200 (the wall time from the CSV, `:damped` at either rate and `:fitted`
alike to within 10 %):

| `h` | points | GPU memory | steps per `M` | `s/M` | `40 M` |
|---|---|---|---|---|---|
| `1/16` | `3.51 M` | `21 GB` | `56` | `34` | `0.37–0.41 h` |
| `1/24` | `11.83 M` | `33 GB` (`:fitted` `37`) | `84` | `125` | `1.34–1.44 h` |
| `1/32` | `28.05 M` | `54 GB` (`:fitted` `64`) | `112` | `234–250` | `2.63–2.81 h` |

That is `175`, `126` and `75 ns` per point and step, `2–4×` what the
`a = 9/10` family above costs per point at a similar size; where the
difference goes (the finder every `M/2`, the smaller mesh) was not measured.
**(2026-10-22:** not the finder, which is under a second a find, but the
analysis row every `M/2` and the trailing/leading split, which together are
about three quarters of such a run's cost (`175` of `239 s/M` for the
spinning hole's `1/32` row below, on TreeAMR 0.2.4); the evolution alone is about `20 ns` per
point and step — `CODE.md`, [The analysis cadence on an
H200](CODE.md#the-analysis-cadence-on-an-h200-measured-2026-10-22).**)**

**`:damped` at `8/M` puts the moving hole at the resting hole's level
(measured 2026-10-20).** Moving over resting, at `ρ_max = 8/M`:

| `h` | `t` | error `[2, 2.25)` | `[3, 5)` | `[5, 8)` | `ℋ` `[2, 2.25)` | `[3, 5)` | `[5, 8)` |
|---|---|---|---|---|---|---|---|
| `1/16` | `4 M` | `2.76` | `1.24` | `1.23` | `4.66` | `1.02` | `0.98` |
| `1/16` | `16 M` | `2.21` | `1.45` | `1.31` | `5.65` | `1.05` | `0.95` |
| `1/16` | `40 M` | `3.18` | `3.38` | `2.79` | `5.65` | `1.05` | `1.05` |
| `1/24` | `4 M` | `1.83` | `1.22` | `1.24` | `3.34` | `0.96` | `0.99` |
| `1/24` | `16 M` | `1.35` | `1.36` | `1.32` | `3.22` | `1.02` | `0.86` |
| `1/24` | `40 M` | `2.74` | `3.15` | `2.58` | `3.52` | `1.07` | `1.05` |
| `1/32` | `4 M` | `1.27` | `1.22` | `1.24` | `1.19` | `0.96` | `0.99` |
| `1/32` | `16 M` | `1.32` | `1.34` | `1.32` | `1.12` | `1.00` | `0.82` |
| `1/32` | `40 M` | `2.84` | `3.35` | `2.70` | `1.14` | `1.09` | `1.08` |

The convergence orders, `1/16 → 1/24` and `1/24 → 1/32`, at `8/M`:

| | `t` | error `[2, 2.25)` | `[3, 5)` | `[5, 8)` | `ℋ` `[2, 2.25)` | `[3, 5)` | `[5, 8)` |
|---|---|---|---|---|---|---|---|
| moving | `4 M` | `5.06`, `5.31` | `4.05`, `4.03` | `3.98`, `3.99` | `4.99`, `7.60` | `4.16`, `4.01` | `3.98`, `3.97` |
| moving | `16 M` | `5.29`, `4.12` | `4.20`, `4.10` | `4.03`, `4.03` | `5.44`, `7.69` | `4.09`, `4.09` | `4.26`, `4.17` |
| moving | `40 M` | `4.43`, `3.92` | `4.22`, `3.81` | `4.24`, `3.87` | `5.23`, `7.93` | `3.99`, `3.93` | `4.02`, `3.90` |
| at rest | `4 M` | `4.05`, `4.05` | `4.02`, `4.02` | `3.99`, `3.99` | `4.17`, `4.01` | `4.02`, `4.01` | `4.00`, `3.99` |
| at rest | `16 M` | `4.07`, `4.05` | `4.05`, `4.03` | `4.05`, `4.03` | `4.06`, `4.02` | `4.03`, `4.01` | `4.01`, `4.01` |
| at rest | `40 M` | `4.07`, `4.04` | `4.04`, `4.03` | `4.05`, `4.03` | `4.06`, `4.02` | `4.03`, `4.01` | `4.02`, `4.01` |

- **At `1/32` the moving hole is the resting one**: the error `1.13–1.34×`
  the resting hole's in every shell to `16 M`, `ℋ` `0.82–1.19×` at every time
  to `40 M`.
- **The constraints converge at order four, and faster just outside the
  horizon.** From `[3, 5)` out, error and `ℋ` alike, the orders are `3.8–4.3`
  at every time (one exception, `ℋ` in `[3, 5)` at `8 M` from `1/16`, `4.9`).
  In `[2, 2.25)` `ℋ` goes at `5.0–5.6`, then `7.5–7.9`: the moving hole's
  excess there is a layer term that falls much faster than the bulk,
  `4.7–5.8×` the resting hole's `ℋ` at `1/16`, `3.1–3.5×` at `1/24`,
  `1.1–1.2×` at `1/32`. So `1/16` is adequate at `8/M` for anything but `ℋ`
  just outside the horizon, and `1/24`, `1/32` is a convergence pair.
- **The error's ratio outside `r = 3` is the same at every `h`**: `1.22–1.24`
  at `4 M` (`1.13–1.17` at `1 M`), `1.31–1.45` at `16 M`, `2.6–3.4` at
  `40 M`. A ratio that does not depend on `h`, in a norm that converges at
  order four, is the moving solution's own truncation error, not a layer term
  — a contracted hole has more structure on the grid. Its growth late in the
  run is the hole's position (below), and `ℋ`, which does not see a position,
  stays flat.
- **`8/M` costs a hole at rest nothing**: from `1/24` on the resting runs at
  `4/M` and `8/M` agree to about 1 % in every shell, and at `1/16` `8/M`
  lowers `ℋ` just outside the horizon `1.6–1.8×`. `ρ_max dt = 0.14` at
  `1/16`, far from RK4's bound.

**At `4/M` the analytic layer lags its moving target (measured 2026-10-20).**
Moving over resting just outside the horizon, `[2, 2.25)`, at `t = 4`, `16`
and `40 M`:

| `h` | error, `4/M` | error, `8/M` | `ℋ`, `4/M` | `ℋ`, `8/M` |
|---|---|---|---|---|
| `1/16` | `36.9`, `34.2`, `28.4` | `2.76`, `2.21`, `3.18` | `35.1`, `32.6`, `36.7` | `4.66`, `5.65`, `5.65` |
| `1/24` | `12.5`, `14.0`, `11.7` | `1.83`, `1.35`, `2.74` | `30.1`, `30.7`, `34.9` | `3.34`, `3.22`, `3.52` |
| `1/32` | `2.62`, `3.67`, `4.10` | `1.27`, `1.32`, `2.84` | `8.43`, `7.57`, `8.83` | `1.19`, `1.12`, `1.14` |

The layer residual, the layer's distance from the analytic solution it relaxes
to, from `1 M` to `40 M`:

| | `1/16` | `1/24` | `1/32` |
|---|---|---|---|
| moving, `4/M` | `6.3–6.6` | `6.4–6.6` | `6.5–6.6` |
| moving, `8/M` | `3.8–4.0` | `3.8–4.0` | `3.9–4.0` |
| at rest, `4/M` | `0.121` | `0.067` | `0.040` |
| at rest, `8/M` | `0.065` | `0.035` | `0.020` |

- **The excess at `4/M` is large at `1/16` and superconvergent.** Just
  outside the horizon the error is `28–44×` the resting hole's at `1/16`,
  `11.7–18.5×` at `1/24` and `2.6–4.6×` at `1/32` (orders `6.3–6.7`, then
  `7.7–9.5`); `ℋ` is `30–37×`, `29–35×` and `7.5–10×` (orders `5.3–5.7`, then
  `8.4–9.0`). It spreads outward with the start-up pulse — `ℋ` in `[3, 5)` is
  `17×` the resting hole's at `8 M` at `1/16` — and over the evolved region
  the error is `46–56×` the resting hole's at `1/16` and `1/32` and `91–107×`
  at `1/24`.
- **The moving residual does not depend on `h`, and falls with `ρ_max`**:
  `6.5` at `4/M` and `3.9` at `8/M` at all three spacings, where the resting
  hole's falls with `h` (orders `1.5` and `1.8`) and is `50–200×` smaller. A
  residual set by the rate and not by the grid is the lag of a relaxation
  behind a moving target, `|∂_t u_exact|/ρ_max`: the analytic layer has no
  `∂_t u_exact` term (the `:fitted` layer's `target_rate` is that term), so at
  `v = 1/5` it trails the solution by an amount of order one. Doubling the
  rate takes it down `1.7×`, not the `2×` a pure lag would. What leaks from it
  through a ramp whose cell count grows as `1/h` (`13`, `19.5`, `26`) is what
  falls faster than any power above.

**Trailing against leading (measured 2026-10-20).** The ratio of the trailing
half's norm to the leading half's, from `4 M` to `40 M` (at `1 M` every moving
run has `1.7–2.6` in every shell outside the horizon, whatever its layer: the
start-up transient of a moving hole, not the layer):

| row | band inside the horizon, error | `ℋ` | `[2, 2.25)`, error | `ℋ` |
|---|---|---|---|---|
| `:damped` `4/M`, `1/16` | `22–26` | `10–11` | `6.2–35` | `16–22` |
| `:damped` `4/M`, `1/24` | `84–91` | `31–34` | `12–30` | `31–47` |
| `:damped` `4/M`, `1/32` | `34–77` | `48–50` | `1.5–7.3` | `9.8–13` |
| `:damped` `8/M`, `1/16` | `2.8–3.2` | `1.45–1.49` | `1.45–3.6` | `3.1–4.1` |
| `:damped` `8/M`, `1/24` | `5.2–8.1` | `3.7–3.8` | `1.07–2.8` | `4.1–4.8` |
| `:damped` `8/M`, `1/32` | `1.7–3.5` | `3.3–3.4` | `0.98–1.78` | `1.15–1.30` |
| `:fitted`, `1/24` | `38–73` | `48–54` | `5.2–23` | `34–57` |
| `:fitted`, `1/32` | `40–57` | `36–40` | `2.8–57` | `54–84` |
| `:fitted` without the ramp, `1/24` | `81–89` | `31–33` | `5.7–15` | `19–43` |

At `8/M` and `1/32` the two sides just outside the horizon are equal in the
error by `32 M` (within `1.8×` before) and within `1.3×` in `ℋ`; the band
inside the horizon, which holds the points the layer released last, keeps
`1.7–3.5×` in the error and `3.3×` in `ℋ`. At `4/M` the trailing side is the whole excess, `10–90×` inside the
horizon at every `h`.

**The horizon is the contracted Schwarzschild horizon, and its center drifts
at order four (measured 2026-10-20).** At `40 M`, `:damped` at `8/M`, moving
against resting (`x/y` is the found horizon's extent along the boost over its
extent across it; `c_x` is the analytic center):

| `h` | `M_irr − 1` | `J` | `x/y − √(1 − v²)` | `x − c_x` | `dM_irr/dt`, `[20, 40] M` |
|---|---|---|---|---|---|
| `1/16` | `2.21e−6` / `2.40e−6` | `1.98e−9` / `1.11e−9` | `−4.6e−7` | `2.52e−5` / `2.06e−6` | `1.51e−8` / `2.16e−8` |
| `1/24` | `4.15e−7` / `4.80e−7` | `2.65e−10` / `1.49e−10` | `−6.1e−8` | `4.74e−6` / `4.33e−7` | `2.74e−9` / `4.13e−9` |
| `1/32` | `1.33e−7` / `1.46e−7` | `6.33e−11` / `3.55e−11` | `−2.8e−8` | `1.60e−6` / `1.41e−7` | `5.37e−10` / `1.28e−9` |

- **`M_irr`, `J` and the contraction are Schwarzschild's to the finder's
  accuracy.** `M_irr − 1` is the resting hole's within 15 %, `J` is constant
  in time, at rest as moving, and converges at order five: it is the finder's
  on the initial data, not the evolution's. The
  extent ratio is `0.97980` to `3·10⁻⁸` at `1/32`. The mass drift converges
  at order `4.2`, then `5.7` (at rest `4.1`, `4.1`). At `4/M` its sign changes
  with `h` (`−1.1e−7`, `+1.2e−8`, `−1.2e−10`) and it has no order.
- **The found center runs ahead of `c(t)` by truncation error.**
  `x − c_x` at `40 M` falls at order `4.1`, then `3.8`, and grows faster than
  linearly in time (at `1/32`: `1.5e−7`, `4.6e−7`, `9.5e−7`, `1.6e−6` at `16`,
  `24`, `32`, `40 M`); at rest the hole drifts too, `11×` less. A hole
  displaced by `1.6e−6` differs from the analytic one by about that times
  `∂_r g ≈ 0.45` at `r = 2.1`, `7e−7`, the same order as the moving error's
  late excess there (`1.25e−6`). That is why the error against the analytic
  solution grows from `16 M` on in every shell and at every `h` while `ℋ`
  stays flat: the moving hole is a good solution in a slightly wrong place.

**`:fitted` holds the moving hole, at `:damped`'s `4/M` level and not its
`8/M` one (measured 2026-10-20).** The moving `:fitted` run over the moving `:damped` runs, just
outside the horizon, at `t = 4`, `16` and `40 M`:

| `h` | error, over `4/M` | error, over `8/M` | `ℋ`, over `4/M` | `ℋ`, over `8/M` |
|---|---|---|---|---|
| `1/24` | `1.19`, `0.87`, `0.99` | `8.1`, `9.0`, `4.2` | `1.64`, `1.36`, `1.35` | `14.7`, `13.1`, `13.5` |
| `1/32` | `13.8`, `4.9`, `4.0` | `28.4`, `13.7`, `5.8` | `16.4`, `18.7`, `15.6` | `116`, `126`, `120` |

- **At `1/24` it is `:damped` at `4/M` in the error just outside the horizon**
  (`0.87–0.99×` from `8 M` on) with `1.35–1.64×` its `ℋ`, and `4–15×` worse
  than `8/M`. Over the evolved region its error is `17–22×` `8/M`'s and `ℋ`
  `36–41×`. Its layer residual against the truth is `3.1–3.4` moving and
  `1.94` at rest (`2.7–3.3` and `1.51` at `1/32`), against `:damped`'s `3.9`
  at `8/M`.
- **At `1/32` `ℋ` just outside the horizon does not converge, even at rest**:
  `2.29e−6` at `1/24` and `1.94e−6` at `1/32` at rest at `16 M` (order
  `0.6`, and `0.59` to `40 M`), while the error there converges at
  `4.3–4.6`;
  moving, `ℋ`'s order is `−0.2–0.4` and the error's `0.95–3.1`. That is the
  fit's leak with the margin fixed in cells: `m = 16` puts the offset surface
  `0.67 M` below the horizon at `1/24` and `0.5 M` at `1/32`, while the octant's
  `:fitted` converged faster than fourth order at a fixed physical depth,
  where its layer gains cells. The resting `1/24` run reproduces the octant's
  setup row (`ℋ` `2.30e−6` against `2.3·10⁻⁶` at `24 M`, the error `2.08e−6`
  against `1.8·10⁻⁶`).
- **The trailing ramp does not help here.** With `trail_ramp = 9/10` the
  error just outside the horizon is `1.5–2.4×` the run without it and `ℋ`
  there `1.05–3.0×`; over the evolved region the error is `0.72–0.91×` and `ℋ`
  `1.5–1.7×`. Inside the horizon it halves the trailing side's excess (`37–40`
  against `81–88` from `8 M` on) and lowers its residual (`3.1–3.4` against
  `3.9–5.0`), and what it pastes harder there reaches the horizon. Step 8′'s
  lever was measured on the harmonic chart at `q = 2`, `5/128`–`5/256`; at
  `v = 1/5`, `q = 4` and `1/24` it is not one.

**What it says.** A non-spinning hole at binary speed runs on the analytic
layer at the resting hole's level, provided the layer relaxes at `8/M`, its
core follows the target, and `r_1` is measured against the contracted
horizon. The analytic layer at the default `4/M` lags a moving target by an
amount the grid does not reduce, and the evolved region pays for it — the
error just outside the horizon `28–44×` the resting hole's at `1/16`, still
`2.6–4.6×` at `1/32`. What is left at `8/M` is the hole's
position, which drifts by truncation error at order four, and a layer term in
`ℋ` just outside the horizon that converges at order `7.5–7.9`. `:fitted`, the
variant a binary needs, holds the moving hole stably for `40 M` but is
`:damped` at `4/M` at best, and its `ℋ` just outside does not converge at a
margin fixed in cells.

**What is open** (2026-10-20):

- **A target rate for `:damped`.** Adding `(1 − w) ∂_t u_exact` in the layer —
  the analytic counterpart of the `:fitted` layer's `target_rate` — should
  remove the lag at `4/M`; whether `8/M` is then still better is the
  measurement. **(Done 2026-10-21:** it does, with the core's rate besides,
  and `8/M` is then no better for a moving hole than for a resting one —
  [the next section](#the-moving-damped-layers-target-rate-2026-10-21).**)**
- **`:fitted`'s leak at a margin fixed in cells.** A convergence pair at a
  fixed physical margin (`m = 16` at `1/24`, `21`–`22` at `1/32`), and the
  margin screen of the octant (`m = 16–20`) for a moving hole.
- **The trailing ramp** helped step 8′'s harmonic `v = 0.3` hole and hurts
  here; which of chart, order, resolution or speed decides is unmeasured.
- **Circular orbits.** A straight line at constant speed is the easy case:
  an orbit adds an acceleration (`v²/(d/2) ≈ 0.003–0.004/M` at
  `d = 10 M_tot`), a rotating direction of motion, and of the quadrant's two
  mirrors only `z = 0`.
- **The indicator's mesh instead of a capsule.** The capsule costs the whole
  path at the finest level; a run that regrids around the moving hole is step
  8's machinery and has not been run in Kerr-Schild.
- **A moving `a = 9/10` hole.** At rest it needs `1/96`; moving, a spin
  along `z` keeps only the mirror `z = 0`, so it runs on a half-box, twice
  the quadrant's points at the spacing the resting hole already needs.
  **(2026-10-21:** moving along its spin it runs on the rotating quadrant
  instead, a quarter of the box — [a spinning hole along its
  axis](#a-spinning-hole-along-its-axis-2026-10-21), at `a = 1/2`.**)**

## The moving `:damped` layer's target rate (2026-10-21)

The previous section's first open question. At `4/M` the analytic layer trailed
the moving hole by `|∂_t u_exact|/ρ_max`, because `(INTERIOR)` is `w F(u) −
ρ (u − u_exact)` and `w F(u_exact)` falls short of `∂_t u_exact` by `(1 − w)
∂_t u_exact` wherever `w < 1` (`CODE.md`, "Why touch the interior"). Commit
`ae19a4a` adds that term to the layer of a moving `:damped` sphere and, as it
turned out, the corresponding term to its core.

**What was built** (`CODE.md`, "The interior", amended 2026-10-20):

- **The layer**: `∂_t u = w F(u) + (1 − w) ∂_t u_exact − ρ (u − u_exact)`.
  `∂_t h` is `dmetric`'s time column from the pass the layer already makes;
  `∂_t Π` is the centered difference of that pass at `t ± δ`, `δ = ε^{1/3} r_0`
  (`background_state_rate`). It agrees with ForwardDiff's `−v · ∇u` to
  `2–6·10⁻¹⁰` of `|∂_t Π|`. A difference rather than `ddmetric` (a second
  nested dual pass compiled into every `:damped` kernel, the resting hole's
  included) or `−v · ∇u` (the same second derivatives, for a rigidly
  translating target only).
- **The core**: in its outer half, `s(r) ∂_t u_exact(t, c(t) + r_0 n̂)` beside
  the ADM relaxation, `s` rising `C²` from 0 at `r_0/4` to 1 at `r_0/2`
  (`core_target_rate`, `core_rate_weight`). The layer's rate alone was not
  enough: on the suite's quadrant (`q = 2`, `h = 5/48`, `4/M`) the layer's
  residual after 8, 16 and 32 RK4 steps was `18.2`, `27.3`, `34.6` without
  either rate, `1.31`, `2.36`, `12.3` with the layer's, and `3.33` at 32 with
  both, against the resting hole's `1.09`, `1.74`, `2.61`. The jump at 32 steps
  sat at `r = r_0` on the trailing-oblique side, where the core releases its
  points, and the core itself was `10–92` off its target there: the core lags
  its own moving target by `|∂_t u|/ρ_max`.
- A hole at rest takes the old branches and is unchanged bit for bit (its
  resting runs above are the controls here). The moving quadrant agrees with
  the moving box to `3.8·10⁻¹¹` instead of roundoff: the difference's roundoff,
  `ε^{2/3} |Π|`, is not the same at a point and at its mirror image.

**On the CPU first** (`L = 8`, `N = 16`, radii `4, 3, 2`, `h = 1/16`, `4/M`,
the hole from `x = −1` to `−1/5`, `.claude/data/boosted-ks-rate-cpu`): to `4 M`
the layer residual was `0.12–0.21` against `6.3–6.6` without the rates and
`0.121` at rest; the masked error at `4 M` was `6.7·10⁻⁵` against
`1.58·10⁻³` and `3.8·10⁻⁵`; just outside the horizon the moving hole was
`1.12×` the resting one in the error and `1.07×` in `ℋ`, where it had been
`5.3×` and `7.0×`.

**The H200 family** (`boosted-ks-3`, jobs 579817–579821, the previous
section's setup and resting controls, `.claude/data/boosted-ks-rate`; `julia
test/boosted_study.jl dir=.claude/data/boosted-ks-rate
baseline=.claude/data/boosted-ks rate=4,8` prints the tables, `baseline=`
added for this). Every run reached `40 M` and every find succeeded; a run cost
what it did without the rate (`0.38`, `1.34`, `2.63 h`). Moving over resting
just outside the horizon, `[2, 2.25)`, at `t = 4`, `16` and `40 M`:

| `h` | error, `4/M`, with | without | `ℋ`, `4/M`, with | without | error, `8/M`, with | `ℋ`, `8/M`, with |
|---|---|---|---|---|---|---|
| `1/16` | `2.09`, `1.47`, `2.79` | `36.9`, `34.2`, `28.4` | `1.62`, `1.36`, `1.41` | `35.1`, `32.6`, `36.7` | `1.35`, `1.32`, `2.85` | `1.03`, `1.08`, `1.09` |
| `1/24` | `1.26`, `1.35`, `2.86` | `12.5`, `14.0`, `11.7` | `1.03`, `1.02`, `1.03` | `30.1`, `30.7`, `34.9` | `1.27`, `1.36`, `2.86` | `0.995`, `1.00`, `1.01` |
| `1/32` | `1.27`, `1.36`, `2.86` | `2.62`, `3.67`, `4.10` | `0.996`, `1.01`, `1.01` | `8.43`, `7.57`, `8.83` | — | — |

(`8/M` against the resting hole at `8/M`.) The layer residual from `1 M` to
`40 M`:

| | `1/16` | `1/24` | `1/32` |
|---|---|---|---|
| moving, `4/M`, with the rates | `0.12–0.21` | `0.12–0.15` | `0.091–0.12` |
| moving, `4/M`, without | `6.3–6.6` | `6.4–6.6` | `6.5–6.6` |
| moving, `8/M`, with the rates | `0.080–0.16` | `0.11–0.13` | — |
| at rest, `4/M` | `0.121` | `0.067` | `0.040` |

- **The lag is gone.** `ℋ` just outside the horizon is the resting hole's to
  `1–3 %` from `1/24` on, at every time to `40 M`, where it was `30–35×` at
  `1/24` and `7.5–8.8×` at `1/32`; over the evolved region `ℋ` is `1.11–1.13×`
  the resting hole's at `1/24` and `1.00–1.01×` at `1/32`, where it was
  `182–187×` and `123–124×`. The residual is `1–3×` the resting hole's and falls,
  slowly, with `h`.
- **The moving hole converges at order four at `4/M`**: from `1/24` to `1/32`
  the error is `4.0–4.1` in every shell and `4.1–4.3` over the evolved
  region, `ℋ` `4.0–4.1` in the shells and `4.55` over the evolved region, at
  `4`, `16` and `40 M`. Without the rate the excess just outside the horizon
  went at `6–9`, which was the layer term falling away.
- **`1/16` keeps a layer term at `4/M`**: `ℋ` just outside `1.36–1.62×` the
  resting hole's, the evolved region's `ℋ` `1.41–1.49×`, the trailing half of
  the shell `2.5–2.9×` the leading one (`0.94–1.03` from `1/24` on); from
  `1/16` to `1/24` `ℋ` there goes at `6.2–6.4`.
- **`8/M` now buys a moving hole nothing a resting one does not get.** At
  `1/24` `8/M` and `4/M` agree to `3 %` in `ℋ` just outside the horizon and to
  `1 %` in the error; at `1/16` `8/M` is lower (`7.6·10⁻⁶` against `1.75·10⁻⁵`
  in `ℋ` there at `16 M`), as it is at rest (`7.0·10⁻⁶` against
  `1.29·10⁻⁵`), so against its own resting hole it is `1.03–1.09×`.
- **`4/M` with the rates beats `8/M` without them**, the previous section's
  recommendation, everywhere: `ℋ` just outside the horizon `0.43–0.55×` at
  `1/16`, `0.30–0.34×` at `1/24`, `0.84–0.90×` at `1/32`; over the evolved
  region `0.27–0.36×`, `0.078–0.079×` and `0.154–0.158×`; the error just
  outside `0.62–1.04×`.
- **The late error is still the hole's position.** Every row's error grows
  from `16 M` on in every shell, to `2.6–3.4×` the resting hole's at `40 M` at
  every `h` and either rate, and `ℋ` does not; the found center at `40 M` is
  `x − c_x = 1.61·10⁻⁶` at `1/32` (`1.60·10⁻⁶` without the rate). `M_irr`, `J`
  and the contraction are the previous section's to its digits.
- **The two sides of the layer are equal**: the band inside the horizon has a
  trailing-to-leading error ratio of `1.0–2.0` at `1/24` and `1/32` (`34–91`
  without the rate), `3.1–4.2` at `1/16`.

**What it says.** A non-spinning hole at binary speed runs on the analytic
layer at the default `4/M` at the resting hole's level from `1/24` on, and
converges at order four; the rate is what was missing, and `ρ_max` goes back to
being a choice that does not depend on the motion. `CODE.md`'s moving
`:damped` row now reads `4/M`. What it leaves open is the previous section's
list without its first item, and one new question: whether `:fitted`'s
`target_rate`, which is on by default, has the same trouble in its core —
its core relaxes toward the cached target with the cache's slope, which is
the target's rate, so it should not.

## A spinning hole along its axis (2026-10-21)

The two sections above move a hole without spin. A Kerr hole spinning about
`z` and moving along `z` has no mirror plane — the motion breaks `z → −z`,
the spin `x → −x` and `y → −y` — but it is its own image under a quarter turn
about `z` at every time, so it runs on the **rotating quadrant**, a quarter
of the box (commit `8efc5fe`; `CODE.md`, [The rotating quadrant: a hole moving
along its spin axis](CODE.md#the-rotating-quadrant-a-hole-moving-along-its-spin-axis-added-2026-10-21)).
It is the first spinning hole this package has moved in Kerr-Schild
coordinates, and it is not G5's case: G5's hole moves *across* its spin, in
the harmonic chart.

**The setup** (`test/octant_runs.jl octant=rotating_quadrant`):

- Kerr-Schild, `M = 1`, `a = 1/2`, the spin along `z`, moving along `z` at
  `v = 1/5`: `kerr_schild_case(; a = 1/2, v = 1/5, x0 = −4, axis = 3)`, the
  hole at `(0, 0, −4 + t/5)`, from `z = −4` at `t = 0` to `z = +4` at
  `40 M`, with the algebraic source at its spin and velocity.
- The rotating quadrant `[0, 64]² × [−64, 64]`: the seam `(1, 2)`, a quarter
  turn about `z` gluing `x = 0` to `y = 0`, no mirror, Dirichlet data at
  `x = 64`, `y = 64` and `z = ±64`. Root brick `(4, 4, 8)`, and the frozen
  capsule of the sections above turned along `z`: cubes of radius `32, 16, 8,
  4` swept along `z ∈ [−4, 4]`, 856 blocks on five levels. `N = 24, 32, 40`
  give `h = 1/24, 1/32, 1/40` at the hole and `11.83`, `28.05` and
  `54.78 M` points. `1/48` would be `94.7 M` (below).
- Common to every row: `q = 4`, `cfl = 1/2` (`84`, `112` and `140` steps per
  `M`), `ε_KO = 1/2`, the Gaussian `γ0` about `c(t)`, chunk `M/2`, the finder
  every chunk with the spin, no noise. **The control** is the same hole at rest
  at `z = −4` on the same mesh.
- **`:damped`**: `r_0 = 3/4`, with the ring `r = a = 1/2` inside it (the ring
  rule's length `(r_0² − a²)/r_0 = 0.42` is `10` cells at `1/24`), and
  `r_1 = 23/16`, `9.4` cells inside the contracted pole `r₊ √(1 − v²) = 1.828`
  at `1/24` (`12.5` and `15.6` at `1/32` and `1/40`); the ramp is `16.5`, `22`
  and `27.5` cells. `ρ_max = 4/M` with the target's rate of commit `ae19a4a`
  (the previous section). Rows `sk24`, `sk32`, `sk40`, and at rest `sk24s`,
  `sk32s`, `sk40s`.
- **`:fitted`**: `m = 20`, `n_L = 18`, `fit_cont = 1`, `lmax_fit = 12` —
  `CODE.md`'s `a = 3/5` row — without the trailing ramp. `1/32` is the
  coarsest spacing at which its core surface clears the ring on the equator:
  `m + n_L = 38` cells below the equatorial horizon `√(r₊² + a²) = 1.932` is
  `r = 0.74` at `1/32` and `0.35 < a` at `1/24`. Rows `sf32`, `sf40`, and at
  rest `sf32s`, `sf40s`.
- One H200 a run on Symmetry, from a copy with `CUDA` added, in `boosted-ks-4`:
  jobs 580104–580109 (`:damped`) and 580129–580132 (`:fitted`), each to `40 M`
  in one job. The CSVs, logs, batch scripts and SimWatch files are in
  `.claude/data/boosted-ks/ks4` (the checkpoints are deleted), and

  ```bash
  julia --project=. test/boosted_study.jl prefix=s shells=r1.93185,r3,r5 times=1,4,8,16,24,32,40
  ```

  prints every table below from them.

The norms are the previous sections': point-weighted L2 about the hole's
current center `c(t)`, in `[1.932, 2.25)` ("just outside the horizon", from
the equatorial horizon radius), `[3, 5)` and `[5, 8)`, and in the evolved
band below `1.932` (from `r_1` for `:damped`, from the offset surface's
equatorial radius `1.932 − 20 h` for `:fitted`), each also split into its
trailing half (`(z − c_z(t)) < 0`) and its leading one. "Global" is the whole
evolved region, which reaches deeper for `:fitted`. Kerr's numbers are
`M_irr = √(r₊/2) = 0.965926`, `J = a`, and the horizon's coordinate extent
along `z` over its extent across, `√(1 − v²) r₊/√(r₊² + a²) = 0.946410`
moving and `0.965926` at rest.

**Every run reached `40 M` and every find succeeded** (81 a run). What a run
costs on one H200 (the wall time from the CSV; the memory the largest
`nvidia-smi` sample of the log, taken every five minutes):

| `h` | points | GPU memory | steps per `M` | `:damped`, moving / at rest | `:fitted`, moving / at rest | `s/M`, `:damped` moving |
|---|---|---|---|---|---|---|
| `1/24` | `11.83 M` | `33 GB` | `84` | `1.45` / `1.46 h` | — | `130` |
| `1/32` | `28.05 M` | `54 GB` (`:fitted` `64`) | `112` | `2.82` / `2.79 h` | `2.70` / `2.68 h` | `254` |
| `1/40` | `54.78 M` | `89 GB` (`:fitted` `107`) | `140` | `6.03` / `5.96 h` | `5.84` / `5.80 h` | `543` |

That is `131`, `81` and `71 ns` per point and step, about what the `a = 0`
hole cost on the `x` quadrant at `1/24` and `1/32` (`126` and `75`). Linear in the points from
`1/32` to `1/40`, `1/48` would need about `141 GB` for `:damped` and `171 GB`
for `:fitted`, against the H200's `144 GB` (`143771 MiB`): the first does not
leave the margin a run needs, the second does not fit.

**`:damped` at `4/M` puts the moving spinning hole at the resting hole's level
(measured 2026-10-21).** Moving over resting:

| `h` | `t` | error `[1.93, 2.25)` | `[3, 5)` | `[5, 8)` | `ℋ` `[1.93, 2.25)` | `[3, 5)` | `[5, 8)` |
|---|---|---|---|---|---|---|---|
| `1/24` | `4 M` | `1.15` | `1.19` | `1.20` | `1.02` | `0.968` | `0.979` |
| `1/24` | `16 M` | `1.12` | `1.14` | `1.20` | `0.965` | `1.00` | `0.814` |
| `1/24` | `40 M` | `1.51` | `1.56` | `1.56` | `0.971` | `1.05` | `1.01` |
| `1/32` | `4 M` | `1.16` | `1.19` | `1.21` | `1.00` | `0.968` | `0.985` |
| `1/32` | `16 M` | `1.16` | `1.14` | `1.20` | `1.01` | `1.00` | `0.813` |
| `1/32` | `40 M` | `1.51` | `1.58` | `1.58` | `1.02` | `1.06` | `1.01` |
| `1/40` | `4 M` | `1.16` | `1.19` | `1.21` | `0.998` | `0.968` | `0.989` |
| `1/40` | `16 M` | `1.15` | `1.14` | `1.20` | `1.01` | `0.999` | `0.813` |
| `1/40` | `40 M` | `1.50` | `1.58` | `1.58` | `1.01` | `1.06` | `1.01` |

The convergence orders, `1/24 → 1/32` and `1/32 → 1/40`:

| | `t` | error `[1.93, 2.25)` | `[3, 5)` | `[5, 8)` | global | `ℋ` `[1.93, 2.25)` | `[3, 5)` | `[5, 8)` | global |
|---|---|---|---|---|---|---|---|---|---|
| moving | `4 M` | `4.06`, `4.03` | `4.02`, `4.02` | `3.99`, `3.99` | `6.07`, `4.15` | `4.51`, `4.02` | `4.00`, `3.99` | `3.97`, `3.96` | `9.55`, `6.95` |
| moving | `16 M` | `3.78`, `4.05` | `4.06`, `4.03` | `4.04`, `4.03` | `5.70`, `4.12` | `4.23`, `4.03` | `4.01`, `4.00` | `4.01`, `4.00` | `9.67`, `6.86` |
| moving | `40 M` | `3.91`, `4.04` | `4.02`, `4.03` | `4.03`, `4.03` | `4.88`, `4.08` | `4.21`, `4.04` | `3.99`, `4.01` | `4.00`, `4.00` | `9.64`, `6.84` |
| at rest | `4 M` | `4.11`, `4.03` | `4.02`, `4.01` | `4.00`, `4.00` | `6.28`, `4.23` | `4.45`, `4.00` | `4.00`, `3.99` | `3.99`, `3.98` | `9.22`, `7.36` |
| at rest | `16 M` | `3.89`, `4.03` | `4.06`, `4.03` | `4.03`, `4.03` | `5.89`, `4.17` | `4.40`, `4.02` | `4.01`, `4.00` | `4.01`, `4.00` | `9.33`, `7.28` |
| at rest | `40 M` | `3.90`, `4.03` | `4.05`, `4.03` | `4.06`, `4.03` | `5.29`, `4.12` | `4.37`, `4.02` | `4.00`, `4.00` | `4.00`, `4.00` | `9.33`, `7.26` |

- **`ℋ` is the resting hole's at every `h` and every time to `40 M`**:
  `0.96–1.02×` just outside the horizon, `0.97–1.06×` in `[3, 5)` and
  `0.81–1.01×` in `[5, 8)` (the dip to `0.81` at `16 M` is the same at every
  `h`, the moving solution's own structure), and `0.90–1.08×` over the evolved
  region.
- **Order four outside the horizon, moving and at rest**: `3.96–4.08` in
  `[3, 5)` and `[5, 8)` in both pairs at every time, the error and `ℋ` alike;
  just outside the horizon `3.8–4.1` in the error and `4.0–4.5` in `ℋ` from
  `1/24`, `4.00–4.05` in both from `1/32`. From `4 M` on the evolved region's
  `ℋ` goes at `9.2–9.7`, then `6.8–7.4`, at rest as moving: the resting
  layer's own term falling away.
- **The error's excess grows late, by the same ratio at every `h`**:
  `1.10–1.24×` to `16 M`, `1.50–1.58×` at `40 M` in every shell. As on the `x`
  quadrant, that is the hole's position: the found origin runs ahead of `c(t)`
  by truncation error, `z − c_z = 1.17·10⁻⁶` at `40 M` at `1/32` (`9×` the
  resting hole's `1.33·10⁻⁷`), order four, and faster than linearly in time —
  `1.2`, `3.4`, `6.9`, `11.7·10⁻⁷` at `16`, `24`, `32`, `40 M`, about as
  `(t − 4 M)²`. A displacement of `1.17·10⁻⁶` is an error of about half that
  at `r ≈ 2`, the size of the moving error's excess just outside the horizon
  at `40 M` (`7.4·10⁻⁷`). `ℋ`, which does not see a position, stays flat.

**The horizon is Kerr's, and its spin drifts only as the resting hole's
(measured 2026-10-21).** `:damped`, moving / at rest, at `40 M` (the drifts
fitted over `[20, 40] M`):

| `h` | `M_irr − √(r₊/2)` | `dM_irr/dt` | `J − a` | `dJ/dt` | `z/x − e` | `z − c_z` |
|---|---|---|---|---|---|---|
| `1/24` | `2.2e−8` / `6.5e−8` | `−9.1e−9` / `−7.5e−9` | `5.22e−6` / `5.29e−6` | `1.245e−7` / `1.280e−7` | `−6.9e−7` / `−6.6e−7` | `3.71e−6` / `4.1e−7` |
| `1/32` | `1.2e−8` / `3.0e−8` | `−2.9e−9` / `−2.3e−9` | `1.82e−6` / `1.82e−6` | `4.04e−8` / `4.05e−8` | `−2.3e−7` / `−2.1e−7` | `1.17e−6` / `1.3e−7` |
| `1/40` | `7.3e−9` / `1.2e−8` | `−1.2e−9` / `−9.4e−10` | `7.39e−7` / `7.37e−7` | `1.64e−8` / `1.65e−8` | `−9.2e−8` / `−8.6e−8` | `4.76e−7` / `5.6e−8` |
| orders | | `4.02`, `4.04` / `4.12`, `4.04` | `3.66`, `4.05` / `3.72`, `4.04` | `3.92`, `4.04` / `4.00`, `4.03` | `3.88`, `4.05` / `3.96`, `4.01` | `4.01`, `4.03` / `3.88`, `3.88` |

- **`M_irr` and the contraction are Kerr's to truncation error.** `M_irr` is
  within `2.6·10⁻⁷` of `√(r₊/2)` at every chunk at `1/24`, `8.5·10⁻⁸` at
  `1/32` and `3.1·10⁻⁸` at `1/40`, and its drift converges at order four. The extent ratio
  is `0.946410` to `2.3·10⁻⁷` at `1/32`; it drifts by as much at rest, and
  converges at order four.
- **`J` drifts linearly, at the resting hole's rate, at order four.** `J − a`
  rises at a nearly constant rate — `4.04·10⁻⁸/M` at `1/32` over `[20, 40]
  M`, `1.8·10⁻⁶` at `40 M` — and the moving hole's rate is the resting hole's
  to `0.4–3 %` at every `h`; orders `3.9–4.0`, then `4.0`. That is the resting
  octant's finding at `a = 3/5` (`5.1–5.4·10⁻⁸/M` at `1/32`, order four,
  [Single holes on the
  octant](#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07)),
  and the motion along the spin adds nothing to it. `J` here is the Korzyński
  `J`'s component along `z` (`hz_J`); its magnitude agrees with it to the
  CSV's ten digits, so the axis does not tilt.
- **G5's drift is something else.** On G5's chart — harmonic Kerr at
  `a = 7/10`, boosted at `v = 0.3` *across* the spin, `:fitted`, `q = 2` —
  `J` rose by `0.013` a `M` while the resting hole kept it ([the trailing side
  (step 8′)](#the-trailing-side-step-8)), measured at one resolution: `3·10⁵`
  times this hole's drift at `1/32`, which is the same moving and at rest. A
  spinning hole moving along its spin does not drift, on either layer; what
  G5 has and this run does not — the boost across the spin, the harmonic
  chart, `q = 2`, `v = 0.3`, `a = 7/10`, its `:fitted` layer at `m = 4` — is
  where the drift must come from. That is an elimination, not a test of any of
  them.

**Trailing against leading (measured 2026-10-21).** The trailing half's norm
over the leading half's, from `4 M` to `40 M` (at `1 M` every moving row,
`:fitted` included, has `1.5–2.5` in the error in every shell outside the
horizon: the start-up transient):

| row | band inside, error | `ℋ` | `[1.93, 2.25)`, error | `ℋ` | `[3, 5)`, error | `[5, 8)`, error |
|---|---|---|---|---|---|---|
| `:damped`, `1/24` | `1.18–1.36` | `1.10–1.21` | `1.01–1.29` | `0.98–1.04` | `1.04–1.51` | `1.00–2.33` |
| `:damped`, `1/32` | `1.02–1.18` | `0.95–1.01` | `1.00–1.26` | `0.92–0.96` | `1.04–1.52` | `1.00–2.33` |
| `:damped`, `1/40` | `1.04–1.21` | `0.84–0.85` | `1.01–1.27` | `0.92–0.96` | `1.04–1.52` | `1.00–2.34` |
| `:fitted`, `1/32` | `11.7–12.5` | `9.4–10.0` | `2.9–23.9` | `13–46` | `0.64–3.5` | `0.71–3.2` |
| `:fitted`, `1/40` | `15.7–16.5` | `17.8–19.3` | `2.3–39` | `10–96` | `0.47–12.3` | `0.30–8.8` |

`:damped`'s two sides are equal inside the horizon and just outside it to
`1.4×` at every `h`, as the `a = 0` hole's were with the rate (`1.0–2.0`
inside). Outside `r = 3` the error's ratio, up to `2.3` in `[5, 8)` to
`16 M` and `1.0` at `40 M`, is the same at all three spacings: the moving
solution's own asymmetry, not a layer term. `:fitted`'s trailing side holds
`12–16×` the leading side's error inside the horizon, more at the finer
spacing.

The **layer residual** from `1 M` to `40 M` (`:damped`: off the analytic
solution it relaxes to; `:fitted`: off the truth inside the offset surface,
not off its own target):

| | `1/24` | `1/32` | `1/40` |
|---|---|---|---|
| `:damped`, moving | `0.151–0.153` | `0.0839–0.0847` | `0.0552–0.0553` |
| `:damped`, at rest | `0.157–0.160` | `0.0889–0.0904` | `0.0593–0.0603` |
| `:fitted`, moving | — | `6.58–6.77` | `5.16–5.39` |
| `:fitted`, at rest | — | `5.24–5.68` | `3.58–3.76` |

The moving `:damped` layer is `0.92–0.96×` the resting one and falls at order
about `2` with `h`; the target's rate holds a spinning hole's moving target as it
held the `a = 0` hole's (`0.091–0.12` at `1/32` there).

**`:fitted` holds the moving spinning hole to `40 M`, far above `:damped`,
and does not converge at `m = 20` (measured 2026-10-21).** The moving
`:fitted` row over the moving `:damped` row:

| `h` | `t` | error `[1.93, 2.25)` | `ℋ` `[1.93, 2.25)` | `ℋ` `[3, 5)` | `ℋ` `[5, 8)` | error, global | `ℋ`, global |
|---|---|---|---|---|---|---|---|
| `1/32` | `4 M` | `17.3` | `73` | `1.24` | `1.00` | `494` | `1290` |
| `1/32` | `8 M` | `12.4` | `61` | `15.3` | `1.00` | `457` | `1270` |
| `1/32` | `16 M` | `4.6` | `43` | `11.1` | `13.4` | `393` | `1270` |
| `1/32` | `24 M` | `3.4` | `43` | `2.95` | `8.2` | `344` | `1260` |
| `1/32` | `32 M` | `2.8` | `44` | `1.40` | `2.00` | `296` | `1250` |
| `1/32` | `40 M` | `2.3` | `43` | `1.50` | `1.29` | `248` | `1250` |
| `1/40` | `4 M` | `98` | `522` | `2.2` | `1.00` | `1630` | `10 100` |
| `1/40` | `16 M` | `19` | `180` | `58` | `76` | `1290` | `9710` |
| `1/40` | `40 M` | `7.1` | `176` | `4.2` | `2.8` | `805` | `9650` |

- **A pulse leaves the layer in the first `4 M`**: `ℋ` just outside the
  horizon peaks at `4 M` (`3.7·10⁻⁵` at `1/32`), crosses `[3, 5)` at
  `8–16 M` (`15×` `:damped`'s) and `[5, 8)` at `16–24 M` (`13×`), and is gone
  by `40 M` (`1.3–1.5×`). What stays is `ℋ` just outside the horizon `43×`
  `:damped`'s at `1/32` from `16 M` on, and over the evolved region — which
  reaches `0.625 M` below the horizon at `1/32`, where `:damped`'s stops at
  `r_1`, `0.39–0.49 M` below it — the error `250–500×` and `ℋ` `1250–1290×`.
- **It does not converge from `1/32` to `1/40`**: just outside the horizon
  the orders are `−1.1` to `−3.7` in the error and `−2.2` to `−4.8` in `ℋ`
  from `4 M` on, over the evolved region `−1.2` and `−2.3`; in `[3, 5)` and
  `[5, 8)` it is negative while the pulse passes, positive before it
  arrives, and at `40 M` the error's `3.6` and `1.4`, `ℋ`'s `−0.6` and `0.5`.
  At rest `[3, 5)` and `[5, 8)` converge (`3.5–6.5` from `4 M` on), but just
  outside the horizon `ℋ` goes at `−1.5` to `−1.9` and over the evolved region
  at `−0.7`. That is the `a = 0` hole's finding two sections above: the fit's
  leak with the margin fixed in cells, `m = 20` being `0.625 M` below the
  horizon at `1/32` and `0.5 M` at `1/40`.
- **The motion costs `:fitted` what it does not cost `:damped`**: moving over
  resting, `ℋ` just outside the horizon is `12×` at `1/32` and `13×` at `1/40`
  from `16 M` on, over the evolved region `10.5×` and `15×`. At rest at
  `1/32` `:fitted` is `3.6–4.0×` `:damped`'s `ℋ` just outside the horizon from
  `8 M` on, the error there `0.94–1.25×` (`CODE.md`'s `a = 3/5` row: `3.2×`
  and equal).
- Its horizon holds and does not converge: over every chunk
  `|M_irr − √(r₊/2)| ≤ 7.7·10⁻⁷` and `J − a` between `−7.6·10⁻⁷` and
  `1.6·10⁻⁶` at `1/32` (`1.7·10⁻⁶`, and `−2.5·10⁻⁶` to `3.6·10⁻⁶`, at
  `1/40`), `|z − c_z|` up to `1.1·10⁻⁶` and `2.5·10⁻⁶`.

**What it says.** A spinning hole moving along its spin axis at binary speed
runs on the rotating quadrant on the analytic layer, at the default `4/M` with
the target's rate, at the resting hole's level from `1/24` on — `ℋ` within
`0.81–1.08×` the resting hole's in every shell and over the evolved region at
every time to `40 M` — and converges at order four outside the horizon. Its
horizon is Kerr's: `M_irr`, the contraction and `J` converge at order four,
and `J`'s linear drift is the resting hole's to a few per cent. What grows
late is again the hole's position, at order four. `:fitted` holds the moving
hole for `40 M`, with a start-up pulse that crosses the domain by `40 M`, but
at `m = 20` fixed in cells it does not converge from `1/32` to `1/40`, even at
rest, and moving costs it `10–15×` its resting self. G5's spin drift is not a
property of a moving spinning hole as such.

**What is open** (2026-10-21):

- **A spin not parallel to the motion — G5's case.** A spin along `z` with
  the motion along `x` keeps only the mirror `z = 0`: a half-box, which
  `hole_case` does not offer (the full box is four times this quadrant's
  points). That run, in Kerr-Schild at `q = 4`, is what decides whether G5's
  `J` drift comes from the transverse boost or from G5's chart and order.
- **Circular orbits.** The previous sections' item, unchanged; an orbit in the
  `xy` plane with spins along `z` keeps the mirror `z = 0` and no quarter
  turn.
- **Regridding instead of the capsule.** At `1/40` the capsule is `54.8 M`
  points, and `1/48` does not fit one H200; a mesh that follows the hole (step
  8's regrid) or several GPUs (`test/octant_runs.jl mpi=true`) is what a finer
  or longer run needs. Neither has been run on a moving Kerr-Schild hole.
- **Higher spin.** `a = 9/10` needs `1/96` at rest; along its spin it would
  run on the rotating quadrant, not on the half-box the `a = 0` section
  assumed, but a capsule at `1/96` is `(96/40)³ ≈ 14` times `1/40`'s points. `:fitted`'s
  ring bound grows with `a` too: at `a = 1/2` it already rules out `1/24`.
  **(2026-10-22:** with the thin outer levels of the resting `a = 9/10`
  campaign the capsule is `44.5 M` points at `1/96` and `v = 1/5`, and
  `:damped` holds the hole at `1/72` and `1/96` — [the next
  section](#aligned-spin-and-velocity-which-holes-the-algorithm-holds-2026-10-22).**)**
- **`:fitted` at a fixed physical margin**, as for `a = 0`: a pair with `m`
  scaled with `1/h` (`20` at `1/32`, `25` at `1/40`), and why the motion costs
  `:fitted` `10–15×` and `:damped` nothing.

## Aligned spin and velocity: which holes the algorithm holds (2026-10-22)

The sections above moved two holes at one speed: `a = 0` along `x` and
`a = 1/2` along its spin, both at `v = 1/5`. This campaign asks which holes the
analytic layer holds when the spin and the velocity are aligned, over the
matrix `a ∈ {0, 1/2, 9/10}` × `v ∈ {0, 1/5, 1/2}`, each cell at three spacings
and each moving row against the same hole at rest. `v = 1/2` is two and a half
times a binary member's speed a few orbits out: a stress test of the moving
layer and its core. `a = 9/10` is the highest spin the octant has held, at
`1/96` ([Single holes on the
octant](#single-holes-on-the-octant-a--0-to-910-2026-10-02-to-2026-10-07)).
The campaign forced two fixes to the moving core, and it found one layer that
grows at rest.

**The setup** (`test/octant_runs.jl octant=rotating_quadrant`):

- Kerr-Schild, `M = 1`, the spin `a` along `z`, moving along `z` at `v` from
  `z = −2` at `t = 0` to `20 M`, where it is at `z = −2`, `2` or `8`, with the
  algebraic source at its spin and velocity.
- The rotating quadrant `[0, 64]² × [−64, 64]`, root brick `(4, 4, 8)`, and
  the frozen capsule along the path, `path = 20 v` (`0`, `4`, `10 M`): radii
  `32, 16, 8, 4` at `a = 0` and `1/2`, `32, 16, 8, 4, 3, 2` at `a = 9/10`
  (thin outer levels, as in the resting `a = 9/10` campaign). From `2.7 M`
  points (`a = 0`, at rest, `1/16`) to `69.9 M` (`a = 9/10`, `v = 1/2`,
  `1/96`).
- `:damped` at `ρ_max = 4/M` with the target's rate, and from `031ab30` the
  moving core at `max(ρ_max, 12 |v|/r_0)` (below). `q = 4`, `cfl = 1/2`,
  `ε_KO = 1/2`, the Gaussian `γ0` about `c(t)`, the finder every chunk
  (`N_ah = 12`, with the spin), no noise, a checkpoint every four chunks.
- **`chunk = M`, without the trailing/leading split** (`chunk=1 split=0`).
  On the previous section's `1/32` row that is `97 s/M` against `239` at
  `chunk = M/2` with the split, at the same step, and the evolution alone is
  `64 s/M`. The benchmark is in `CODE.md`, [The analysis cadence on an
  H200](CODE.md#the-analysis-cadence-on-an-h200-measured-2026-10-22).
  Without the split the CSV has no extents and no `hz_J`. `J` here is
  `records.csv`'s, the Korzyński `J`'s magnitude, so at `a = 0` it is
  `|J| ≥ 0`.
- **The radii are fixed in `M` across `v`**, chosen so that `r_1` is at least
  8 cells inside the contracted pole `r₊ √(1 − v²)` at `v = 1/2` and the
  coarsest `h`:

| `a` | `r_0` | `r_1` | `h` | `r₊`; at `v = 1/2` | `r_1` below it, coarsest `h` (at rest) | ramp | ring length `(r_0² − a²)/r_0` |
|---|---|---|---|---|---|---|---|
| `0` | `1/2` | `19/16` | `1/16`, `1/24`, `1/32` | `2`; `1.732` | `8.7` cells (`13.0`) | `11` cells | — |
| `1/2` | `3/4` | `5/4` | `1/24`, `1/32`, `1/40` | `1.866`; `1.616` | `8.8` (`14.8`) | `12` | `10` cells |
| `9/10` | `19/20` | `9/8` | `1/72`, `1/96`, `1/128` | `1.436`; `1.244` | `8.5` (`22.4`) | `12.6` | `7.0` |
| `9/10`, the `r` rows | `19/20` | `5/4` | the same | the same | `11.3` at `v = 1/5` (`13.4`) | `21.6` | `7.0` |

`r_1 = 5/4` is the resting `a = 9/10` campaign's layer. It lies outside the
contracted pole at `v = 1/2` (`1.244`) and cannot hold that speed; the `r`
rows were added when `9/8` grew at rest (below). `1/128` fits one H200 only at
rest: moving, the capsule is `105.6 M` points at `v = 1/5` and `165.7 M` at
`1/2`, about `170` and `260 GiB` extrapolated linearly from the largest
rows, against the H200's `140 GiB`.

- One H200 a run on Symmetry, from copies with `CUDA` added, on TreeAMR 0.2.4:
  `boosted-ks-5` (jobs 580693–580713, at `78e60ae`: every resting row but the
  `r` family's), `boosted-ks-6` (580742–580744, `a = 1/2`, `v = 1/5`, at
  `281a639`; 580822–580826, the `r` family) and `boosted-ks-7` (580890–580900,
  at `031ab30`: the other moving rows). That is 28 runs and `29.7` GPU-hours.
  The rows are named `a{0,5,9}v{0,2,5}h{N}`, `h = 1/N`, and `a9v{0,2}h{N}r`.
  The CSVs, logs, batch scripts and SimWatch files are in
  `.claude/data/aligned/{ks5,ks6,ks7}`; the `*-obsolete` directories hold the
  runs the fixes replaced, and the checkpoints are deleted. This command
  prints every table below from them:

  ```bash
  julia test/aligned_study.jl
  ```

The norms are point-weighted L2 about the hole's current center
`c(t) = (0, 0, −2 + v t)`, in three shells. `r_h` runs from the equatorial
horizon radius `√(r₊² + a²)` (`2`, `1.932`, `1.695`) to `9/4`, `r3` is
`[3, 5)`, and `r5` is `[5, 8)`. "In" is the evolved band between `r_1` and the
equatorial radius, and "global" the whole evolved region. "Error" is the
distance from the moving analytic solution, `ℋ` the Hamiltonian constraint.

**Two fixes the campaign forced** (`CODE.md`, the frozen core's paragraph
under "The interior", amended twice on 2026-10-22). Both leave a hole at rest
bit for bit:

- **`281a639`, the core's target rate mapped in ADM variables.** The first
  `a = 0`, `v = 1/2` rows, at `1/16` and `1/32`, ended in their first chunk.
  At a core point `(2h, h, −3h)` from the center, `α²` reached
  `−1.2·10⁻³`. The target's rate had been added in `h`, which maps it at the
  target and not at the state, and near the path, where the target turns at
  `v r_⊥/r²`, the mismatch drove `α²` through zero.
- **`031ab30`, the core relaxing at `max(ρ_max, 12 |v|/r_0)`.** With the
  first fix alone the `v = 1/2` rows had a layer residual of about `4.7` at
  every `h`. That is the core target's jump where the path crosses a point,
  released on the trailing side after `r_0/v` with `e^{−4}` of it left.
  What the layer did not absorb crossed `r_1`, and moving over resting
  outside the horizon grew to `20–50×` by `20 M`. The core's rate is now
  `12/M` at `a = 0`, `v = 1/2`, `4.8/M` at `a = 0`, `v = 1/5`, `8/M` at
  `a = 1/2`, `v = 1/2` and `6.3/M` at `a = 9/10`, `v = 1/2`. At `a = 1/2`
  and `9/10` with `v = 1/5` it would be `3.2` and `2.5/M`, below `ρ_max`.
- The moving rows were rerun after each fix. The resting rows are bit for bit
  unchanged by both. The `a = 1/2`, `v = 1/5` rows are unchanged by the
  second, and so, their core rate being below `ρ_max`, are the `r` family's
  `v = 1/5` rows.

**Every run reached `20 M` and every find succeeded** (21 a run).

**The verdict (measured 2026-10-22).** "Works": every row of the cell
reached `20 M` with finite norms and a global error that grows by at most
`2×` over `[10, 20] M`. "From": the coarsest `h` from which every adjacent
pair converges at order `3.5–5` in the error in `r3` and `r5`. "`×`": the
error in `r3`, moving over resting, at the finest moving `h` at `20 M`.

| `a` \ `v` | `0` | `1/5` | `1/2` |
|---|---|---|---|
| `0` | works, from `1/16` | works, from `1/16`; `×1.38` | works, from `1/24`; `×4.9` |
| `1/2` | works, from `1/24` | works, from `1/24`; `×1.13` | works, from `1/24`; `×2.79` |
| `9/10`, `r_1 = 9/8` | **grows** inside the horizon from `10 M`: `×382`, `×36`, `×3.4` over `[10, 20] M` at `1/72`, `1/96`, `1/128` | not run | works, from `1/72`; `×2.06` |
| `9/10`, `r_1 = 5/4` | works, from `1/72` | works, from `1/72`; `×0.86` | `r_1` outside the contracted pole |

The orders at the finest pair at `20 M`, and the finest row's growth of the
global error over `[10, 20] M`:

| `a` | `v` | pair | error `r3` | `r5` | global | `ℋ` `r3` | `ℋ` `r5` | growth |
|---|---|---|---|---|---|---|---|---|
| `0` | `0` | `1/24 → 1/32` | `4.04` | `4.04` | `4.51` | `4.01` | `4.01` | `1.03` |
| `0` | `1/5` | `1/24 → 1/32` | `4.06` | `4.05` | `5.49` | `4.03` | `4.03` | `1.02` |
| `0` | `1/2` | `1/24 → 1/32` | `4.14` | `4.01` | `4.64` | `4.21` | `4.32` | `1.01` |
| `1/2` | `0` | `1/32 → 1/40` | `4.03` | `4.03` | `8.45` | `3.99` | `4.00` | `1.05` |
| `1/2` | `1/5` | `1/32 → 1/40` | `4.03` | `4.04` | `8.04` | `4.00` | `4.01` | `1.07` |
| `1/2` | `1/2` | `1/32 → 1/40` | `4.04` | `3.99` | `3.32` | `2.96` | `2.80` | `1.01` |
| `9/10`, `9/8` | `0` | `1/96 → 1/128` | `4.03` | `4.04` | `16.3` | `3.87` | `4.00` | `3.36` |
| `9/10`, `9/8` | `1/2` | `1/72 → 1/96` | `4.04` | `4.06` | `5.68` | `4.01` | `3.97` | `1.08` |
| `9/10`, `5/4` | `0` | `1/96 → 1/128` | `4.03` | `4.04` | `4.09` | `3.87` | `4.00` | `1.44` |
| `9/10`, `5/4` | `1/5` | `1/72 → 1/96` | `4.05` | `4.05` | `4.34` | `3.96` | `4.02` | `1.41` |

The error outside the horizon converges at order four in every cell that was
run, and the coarsest `h` of each family is within `1.8×` of the order-4
extrapolation from the finest (within `1.08×` in every cell but `a = 0`,
`v = 1/2`). Two exceptions to fourth order:

- **`a = 1/2`, `v = 1/2`: `ℋ` converges at only `2.5–3.0`** from `1/32` to
  `1/40`: in `r3` from `10 M` on (`2.48`, `2.96` at `10` and `20 M`), in
  `r5` at `20 M` (`2.80`). The global error goes at `3.3`, and the layer
  residual does not fall there either (below).
- **`a = 9/10`, `r_1 = 9/8`, at rest: the band inside the horizon grows.**
  The global error, which is dominated by that band, is flat to `10 M` and
  then rises at every `h`: at `1/72` it is `5.3·10⁻⁴` at `10 M`,
  `6.2·10⁻³` at `16 M` and `0.20` at `20 M`. It falls about tenfold per step
  in `h` (`×382`, `×36`, `×3.4` over `[10, 20] M`) but is there at every
  `h`. By `20 M` it has not reached `r3`, whose error is the `r_1 = 5/4`
  row's to three digits. The `r_1 = 5/4` rows grow by `1.22–1.44×`
  instead, the linear growth of accumulated truncation error the resting
  campaign measured.

**Moving over resting (measured 2026-10-22)**, at the finest moving `h`, at
`5 M` and `20 M`:

| `a` | `v` | `h` | error `r_h` | `r3` | `r5` | `ℋ` `r_h` | `r3` | `r5` |
|---|---|---|---|---|---|---|---|---|
| `0` | `1/5` | `1/32` | `1.26`, `1.39` | `1.08`, `1.38` | `0.98`, `1.35` | `1.00`, `1.01` | `0.93`, `1.01` | `0.70`, `0.74` |
| `0` | `1/2` | `1/32` | `3.99`, `5.06` | `3.33`, `4.90` | `2.22`, `3.84` | `1.38`, `1.22` | `1.33`, `1.96` | `1.17`, `1.25` |
| `1/2` | `1/5` | `1/40` | `1.14`, `1.16` | `1.04`, `1.13` | `0.96`, `1.12` | `1.00`, `1.01` | `0.95`, `1.00` | `0.71`, `0.73` |
| `1/2` | `1/2` | `1/40` | `3.54`, `3.20` | `3.25`, `2.79` | `2.18`, `2.74` | `2.04`, `1.20` | `1.43`, `2.10` | `1.23`, `1.50` |
| `9/10`, `5/4` | `1/5` | `1/96` | `0.97`, `1.00` | `0.72`, `0.86` | `0.67`, `1.13` | `0.98`, `0.98` | `0.60`, `0.65` | `0.49`, `0.36` |
| `9/10`, `9/8` | `1/2` | `1/96` | `2.08`, `1.98` | `1.67`, `2.06` | `0.93`, `1.97` | `2.12`, `1.89` | `1.19`, `1.26` | `0.45`, `0.36` |

- **The error's excess is the same at every `h`.** At `a = 0`, `v = 1/2`
  the error in `r3` at `20 M` is `5.03×` the resting hole's at `1/24` and
  `4.90×` at `1/32`; at `a = 1/2`, `v = 1/2` it is `2.71×`, `2.79×` and
  `2.79×` at `1/24`, `1/32`, `1/40`. In a norm that converges at order four
  that is the moving solution's own truncation error, together with the
  hole's position, as at `v = 1/5` before. It is not a layer term. `ℋ`,
  which does not see a position, is `0.8–2.7×` the resting hole's at the
  finest `h` at any time.
- **At `v = 1/5` the moving hole is the resting one** in `ℋ`
  (`0.92–1.01×` at `a = 0` and `1/2` in `r_h` and `r3`). At `a = 9/10` it is
  below it (`0.36–0.98×`): the moving hole's shells are another part of the
  same mesh, and the ratio is not a quality.
- **At `1/16` the `a = 0`, `v = 1/2` hole keeps a layer term**: the error in
  `r_h` is `13.4×` the resting hole's at `5 M` (`3.9–4.0×` from `1/24` on),
  and the orders from `1/16` to `1/24` there are `4.6–7.7`. That is why its
  cell is asymptotic only from `1/24`.

**The horizon (measured 2026-10-22).** `M_irr − √(r₊/2)` and `|J| − a` at
`20 M`, at `1 M` (where the finder sees the initial data), and the drift
`d|J|/dt` over `[10, 20] M` with its order against the next coarser `h`, at
the finest `h`:

| `a` | `v` | `h` | `ΔM_irr`, `1 M` / `20 M` | `|J| − a`, `1 M` / `20 M` | `d|J|/dt` (order) | `dM_irr/dt` (order) |
|---|---|---|---|---|---|---|
| `0` | `0` | `1/32` | `2.3e−9` / `1.25e−7` | `3.6e−11` / `3.6e−11` | — | `3.3e−9` (`4.04`) |
| `0` | `1/5` | `1/32` | `1.1e−8` / `1.25e−7` | `5.4e−11` / `6.1e−11` | — | `2.8e−9` (`3.99`) |
| `0` | `1/2` | `1/32` | `6.5e−8` / `2.47e−7` | `7.6e−10` / `7.9e−10` | — | `3.5e−9` (`4.64`) |
| `1/2` | `0` | `1/40` | `−1.0e−9` / `3.24e−8` | `2.7e−8` / `4.09e−7` | `1.75e−8` (`3.98`) | `3.3e−10` (`3.89`) |
| `1/2` | `1/5` | `1/40` | `−1.3e−9` / `3.04e−8` | `3.0e−8` / `4.10e−7` | `1.74e−8` (`4.08`) | `1.7e−10` (`3.74`) |
| `1/2` | `1/2` | `1/40` | `2.23e−7` / `2.71e−7` | `7.03e−7` / `1.09e−6` | `1.77e−8` (`4.16`) | `6.6e−11` (`15`) |
| `9/10`, `5/4` | `0` | `1/128` | `1.87e−7` / `1.89e−7` | `6.83e−6` / `6.94e−6` | `6.7e−9` (`4.06`) | `−8.0e−10` (`4.25`) |
| `9/10`, `5/4` | `1/5` | `1/96` | `3.68e−7` / `3.68e−7` | `1.07e−5` / `1.10e−5` | `1.90e−8` (`4.08`) | `−1.9e−9` (`4.37`) |
| `9/10`, `9/8` | `1/2` | `1/96` | `6.67e−6` / `6.74e−6` | `1.63e−4` / `1.63e−4` | `1.71e−8` (`4.75`) | `6.0e−9` (`1.03`) |

- **The drift is the evolution's, and it converges at order four.** `J`'s
  drift at `a = 1/2` and `1/40` is the same at all three speeds,
  `1.74–1.77·10⁻⁸/M`, at order `4.0–4.2`, and at `a = 9/10` it goes at
  `4.06–4.08` with the `r_1 = 5/4` layer. `M_irr`'s drift goes at order
  `3.6–4.6` at `a = 0`, at `3.7–3.9` at `a = 1/2` from `1/32` to `1/40` at
  rest and at `v = 1/5`, and at `4.3–4.4` at `a = 9/10` with `r_1 = 5/4`. At
  `a = 1/2`, `v = 1/2` it is too small to have an order (`6.6·10⁻¹¹/M`), and
  at `a = 9/10`, `v = 1/2` it goes at `1.0`.
- **The offset at `1 M` is the finder's, and it grows with `a` and `v`.**
  It is the same at every `h` and constant in time on top of the drift.
  `|J| − a` is `6.83·10⁻⁶` at `a = 9/10` at rest, as in the resting
  campaign, which named it the finder's at `N_ah = 12`. Here it is
  `1.07·10⁻⁵` at `v = 1/5` and `1.63·10⁻⁴` at `v = 1/2`, and `M_irr`'s is
  `1.9`, `3.7` and `67·10⁻⁷`. At `a = 1/2` and `v = 1/2` the offset is
  `7.0–8.8·10⁻⁷` in `J` and `1.8–2.2·10⁻⁷` in `M_irr` at every `h`. A faster,
  higher-spin hole has a less round, contracted horizon on the finder's
  fixed `N_ah = 12` grid; that this is the cause is likely, not measured.
- At `a = 0` `|J|` stays below `10⁻⁹` from `1/24` on at every speed
  (`2·10⁻⁸` at most at `1/16`).

The **layer residual** (the `:damped` layer's distance from the analytic
solution it relaxes to) at `20 M` at the three spacings, and its orders:

| `a` | `v = 0` | `v = 1/5` | `v = 1/2` |
|---|---|---|---|
| `0` | `0.319`, `0.189`, `0.101` (`1.3`, `2.2`) | `0.403`, `0.365`, `0.279` (`0.24`, `0.94`) | `2.56`, `1.75`, `1.25` (`0.93`, `1.17`) |
| `1/2` | `0.373`, `0.218`, `0.146` (`1.9`, `1.8`) | `0.357`, `0.202`, `0.134` (`2.0`, `1.9`) | `0.364`, `0.263`, `0.346` (`1.1`, `−1.2`) |
| `9/10`, `9/8` | `11.4`, `6.76`, `4.08` (`1.8`, `1.8`) | — | `4.67`, `2.78` (`1.8`) |
| `9/10`, `5/4` | `2.51`, `1.50`, `0.882` (`1.8`, `1.8`) | `1.60`, `0.955` (`1.8`) | — |

- **At `v = 1/2` what is left is first order.** At `a = 0` it is the
  released point of `CODE.md`'s amendment, the point that has just crossed
  `r_0` on the trailing side: `2.56` at `1/16`, as the `cfl = 1/2` row
  there, and order `0.9–1.2`. At `a = 1/2` it rises again at `1/40`
  (`0.26` at `1/32`, `0.35` at `1/40`, slowly decreasing in time), which is the cell whose `ℋ`
  converges at only `2.5–3.0`.
- At rest and at `v = 1/5`, from `a = 1/2` on, it converges at order `1.75–2.0`,
  as the resting campaigns' did. The resting `r` rows reproduce the resting
  `a = 9/10` campaign's `2.51`, `1.50`, `0.88`.
- **The moving `a = 9/10`, `r_1 = 9/8` rows have less than half the resting
  rows' residual** (`4.67` against `11.4` at `1/72`, `2.78` against `6.76`
  at `1/96`) and do not grow (`1.01×` and `1.08×` over `[10, 20] M`). Their
  core relaxes at `6.3/M`; at rest the core does not relax at all
  (`du = 0`). That the core's rate is what holds the moving layer is
  suggestive, not shown: the moving band is also `8.5` cells deep at `1/72`,
  where the resting one is `22.4`.

**What a run costs** (measured 2026-10-22) on one H200, `chunk = M`, no
split, at the finest `h`. `s/M` is over `[10, 20] M`, monitors and finds
included; the wall time and the peak memory (the largest `nvidia-smi` sample)
are from the log:

| row | points | steps per `M` | `s/M` | `20 M` | GPU memory | ns per point and step |
|---|---|---|---|---|---|---|
| `a = 0`, `v = 0`, `1/32` | `21.6 M` | `111` | `75` | `0.45 h` | `44 GiB` | `31` |
| `a = 0`, `v = 1/2`, `1/32` | `29.9 M` | `112` | `113` | `0.68 h` | `55 GiB` | `34` |
| `a = 1/2`, `v = 0`, `1/40` | `42.2 M` | `139` | `177` | `1.05 h` | `71 GiB` | `30` |
| `a = 1/2`, `v = 1/2`, `1/40` | `58.4 M` | `140` | `252` | `1.44 h` | `92 GiB` | `31` |
| `a = 9/10`, `v = 0`, `1/128`, both layers | `63.4 M` | `443` | `587–605` | `3.4–3.9 h` | `101 GiB` | `21` |
| `a = 9/10`, `v = 1/5`, `1/96` (`r`) | `44.5 M` | `336` | `446` | `2.77 h` | `78 GiB` | `30` |
| `a = 9/10`, `v = 1/2`, `1/96` | `69.9 M` | `336` | `628` | `3.53 h` | `114 GiB` | `27` |

Over all 28 rows a point and step cost `21–78 ns`. The small meshes pay the
per-chunk analysis most (`72–78 ns` at `1/16`, `2.7–3.7 M` points), the
finest rows `21–34 ns`, against the bare evolution's `20 ns` of the
benchmark. That is `2–3×` cheaper per point than the sections above at the
same spacings, which ran `chunk = M/2` with the split (`71–175 ns`). The
memory is about `15 GiB` plus `1.3–1.5 GiB` per million points. The one row without a memory figure, `a9v2h72r`, lost its log
to a requeue (below).

**Operations (2026-10-22).**

- **TreeIOHDF5 `main` broke the first `boosted-ks-7` batch.** TreeIOHDF5
  0.1.4 reads TreeAMR 0.2.5's `FieldSet.halfturn`, but its `[compat]` admits
  TreeAMR `=0.2.2, …, =0.2.5`. The package's `[sources]` pin to `main`
  resolved it beside TreeAMR 0.2.4, and ten of the eleven jobs
  (580867–580876) ended at their first checkpoint, at `4 M`, in TreeIOHDF5's
  `set_layout` ("type FieldSet has no field `halfturn`"); the eleventh,
  580877, stopped at `3 M`. `boosted-ks-7` was pinned to TreeIOHDF5 0.1.3 and
  all eleven were resubmitted (580890–580900). `boosted-ks-5` and `-6`
  had resolved before 0.1.4 appeared.
- **A Slurm controller outage requeued a run that had finished.** `a9v2h72r`
  (job 580825) reached `20 M`; the requeued job restarted from its last
  checkpoint, `t = 20`, and `evolve!` refused ("there is nothing left to
  run"). It exited non-zero, so the batch script marked the SimWatch status
  "failed", and the restart overwrote the job's log. The CSV and
  `records.csv` were complete. The status was corrected by hand to
  "finished", with a message saying what happened.
- TreeAMR 0.2.5, the half-turn seam, is out; the campaign stayed on 0.2.4.

**What it says.** Spin and velocity aligned along `z`, the analytic layer at
the default `4/M`, with the target's rate and the core at its transit rate,
holds every cell that fits one H200 to `20 M` and converges at order four
outside the horizon. That is `a = 0` and `1/2` at `v` up to `1/2`, and
`a = 9/10` at rest and at `v = 1/5` with `r_1 = 5/4`, or at `v = 1/2` with
`r_1 = 9/8`. The coarsest spacing in the convergent regime is the resting
hole's at every speed but one: `1/16` at `a = 0` (`1/24` at `v = 1/2`),
`1/24` at `1/2`, `1/72` at `9/10`. The motion costs the error outside the
horizon a factor that does not depend on `h`: `1.1–1.4` at `v = 1/5`
(`0.86` at `a = 9/10`), `2–5` at `1/2`. The horizon's drift converges at order four, and `J`'s at
`a = 1/2` does not depend on the speed. What does not work is one layer at
`a = 9/10` for every speed. The contracted pole at `v = 1/2` needs
`r_1 ≤ 9/8`, and `r_1 = 9/8` grows inside the horizon at rest. The finder's
offset grows to `1.6·10⁻⁴` in `J` at `a = 9/10`, `v = 1/2`.

**What is open** (2026-10-22):

- **`a = 9/10` at `v = 1/2` with a layer that does not grow at rest.** `r_1`
  must stay `8` cells below `1.244`, and `r_0` outside the ring and the ring
  rule, so the ramp is at most about `0.2 M`. Which of the resting `9/8`
  layer's properties makes it grow is unmeasured: its depth (`22` cells
  below the pole at `1/72`, where `5/4` is `13`), its ramp (`12.6` cells
  against `21.6`) or its frozen core. A hole whose speed changes, as a
  binary member's does, needs one layer for every speed it passes through.
- **The first-order residual at the released point.** At `v = 1/2` it
  converges at order one at `a = 0`, and at `a = 1/2` not at all from `1/32`
  to `1/40`, where `ℋ` outside goes at `2.5–3.0`. `CODE.md`'s untried remedy
  is a core target that matches `∂_r u_exact` at `r_0`.
- **The finder's offset.** An `N_ah` scan on the initial data, host-side and
  cheap, would tell the finder's part of `|J| − a` from the evolution's at
  `a = 9/10` and `v = 1/2`.
- **A spin not parallel to the motion**, G5's case; unchanged from the
  previous section. (Measured in the Kerr-Schild chart on 2026-10-25: [spin
  across the motion](#spin-across-the-motion-the-bitant-2026-10-24).)
- **Circular orbits**; unchanged.
- **Several GPUs for `a = 9/10` at `1/128` moving.** `105.6 M` and
  `165.7 M` points, about two and three H200s (`test/octant_runs.jl
  mpi=true`), would give the moving `a = 9/10` cells a third spacing.

## Spin across the motion: the bitant (2026-10-24)

The previous section moved each hole along its spin. This one turns the
velocity through a right angle: the spin along `z`, the motion along `x`, over
the same matrix `a ∈ {0, 1/2, 9/10}` × `v ∈ {0, 1/5, 1/2}`, with the same
layers and the same spacings. It is the geometry of G5's case, a spin that
is not parallel to the motion, but in the Kerr-Schild chart and with the
analytic `:damped` layer (G5 is the harmonic chart with `:fitted`). Each
transverse row is compared with the aligned row of the same `a`, `v` and `h`.
The rows at rest are the same physics on another domain, and they are
the bitant's check against the rotating quadrant.

**No rotation holds this hole** (`CODE.md`, [The
bitant](CODE.md#the-bitant-a-hole-moving-across-its-spin-added-2026-10-24)).
The request was for TreeAMR 0.2.5's half-turn seam. A half turn keeps only
vectors along its axis: one about `x` reverses the spin and one about `z`
reverses the velocity. The hole's one symmetry is the mirror `z → −z`, so its
domain is the **bitant** `[−L, L]² × [0, L]`, half the box and twice the
quadrants' points. The half turn is for a pair of holes, each the other's
image.

**The setup** (`test/octant_runs.jl octant=bitant`):

- Kerr-Schild, `M = 1`, the spin `a` along `z`, moving along `x` at `v` from
  `x = −2` at `t = 0` to `20 M`, with the algebraic source at its spin and
  velocity.
- The bitant `[−64, 64]² × [0, 64]`, root brick `(8, 8, 4)`. The frozen
  capsule runs along the path, `path = 20 v` along `x`, with the aligned
  campaign's radii (`32, 16, 8, 4`, and `32, 16, 8, 4, 3, 2` at
  `a = 9/10`). The meshes run from `5.4 M` points (`a = 0`, at rest, `1/16`)
  to `89.1 M` (`a = 9/10`, `v = 1/5`, `1/96`).
- As in the aligned campaign: `:damped` at `ρ_max = 4/M` with the target's
  rate, and the moving core at `max(ρ_max, 12 |v|/r_0)`. `q = 4`,
  `cfl = 1/2`, `ε_KO = 1/2`, the Gaussian `γ0` about `c(t)`, and the finder
  every chunk (`N_ah = 12`, with the spin). No noise, `chunk = M`, no split,
  and a checkpoint every `2 M` (every `M` at `a = 9/10`).
- **Across the spin one layer serves every speed.** The least horizon radius
  is now exact for a boosted spinning hole (`horizon_ellipsoid`, `CODE.md`).
  The motion along `x` contracts the equatorial radius `√(r₊² + a²)` and
  leaves the pole `r₊` alone. At `a = 9/10`, `v = 1/2` the least radius is
  the pole's `1.436`, where the aligned hole's contracted pole is `1.244`. So
  the resting `r_1 = 5/4`, which the aligned campaign could not run at
  `v = 1/2`, is `13.4` cells inside at `1/72` at every speed. At `a = 0` and
  `1/2` the radii are the aligned ones.

| `a` | `r_0` | `r_1` | `h` | least radius at rest; at `v = 1/2` | `r_1` below it, coarsest `h`, `v = 1/2` | ramp |
|---|---|---|---|---|---|---|
| `0` | `1/2` | `19/16` | `1/16`, `1/24`, `1/32` | `2`; `1.732` | `8.7` cells | `11` cells |
| `1/2` | `3/4` | `5/4` | `1/24`, `1/32`, `1/40` | `1.866`; `1.673` (the contracted equator) | `10.2` | `12` |
| `9/10` | `19/20` | `5/4` | `1/72`, `1/80`, `1/96` | `1.436`; `1.436` (the pole) | `13.4` | `21.6` |

`1/80` at `a = 9/10` was added at rest and at `v = 1/2`, because `1/96` at
`v = 1/2` does not fit one H200. Six rows were not run for the same reason;
the estimates below are linear in the points from the largest rows, at about
`16 GiB` plus `1.3 GiB` per million points, against the H200's `140 GiB`:

| row | points | memory |
|---|---|---|
| `a = 1/2`, `v = 1/5`, `1/40` | `102 M` | about `150 GiB` |
| `a = 1/2`, `v = 1/2`, `1/40` | `117 M` | about `170 GiB` |
| `a = 9/10`, `v = 1/2`, `1/96` | `140 M` | about `200 GiB` |
| `a = 9/10`, `1/128`, at rest | `127 M` | about `180 GiB` |
| `a = 9/10`, `1/128`, `v = 1/5` | `211 M` | about `290 GiB` |
| `a = 9/10`, `1/128`, `v = 1/2` | `331 M` | about `450 GiB` |

- One H200 a run on Symmetry, `transverse-1`, from a copy with `CUDA` added,
  on TreeAMR 0.2.6 at `495313f` plus the bitant's changes. The smoke run was
  job 582097. The 23 rows were jobs 582102–582124, on `h200q,h200preq`
  (`h200q`'s group limit was full). They are named `b{0,5,9}v{0,2,5}h{N}`,
  `h = 1/N`.
- `h200preq` pre-empted 14 jobs. Each batch script resubmitted itself on a
  pre-emption (exit `143`) or at its wall-time limit (exit `3`), and the
  successor continued from the newest checkpoint; the last was job 582419.
  The 23 rows took `29.1` GPU-hours of evolution, not counting the work lost
  to the pre-emptions or the startups.
- The CSVs, logs, batch scripts and SimWatch files are in
  `.claude/data/transverse/tr1`. The checkpoints (`275 GB`) stayed on
  Symmetry. This command prints every table below from them, with the
  aligned rows from `.claude/data/aligned`:

  ```bash
  julia test/aligned_study.jl campaign=transverse
  ```

The norms and shells are the previous section's, about
`c(t) = (−2 + v t, 0, 0)`. `r_h` is `[√(r₊² + a²), 9/4)`, `r3` is `[3, 5)`,
`r5` is `[5, 8)`, "in" is the evolved band between `r_1` and the equatorial
radius, and "global" is the whole evolved region.

**Every run reached `20 M` and every find succeeded** (21 a run).

**The verdict (measured 2026-10-25)**, with the previous section's
definitions. "From": the coarsest `h` from which every adjacent pair
converges at order `3.5–5` in the error in `r3` and `r5`. "`×`": the error in
`r3`, moving over resting, at the finest moving `h` at `20 M`.

| `a` \ `v` | `0` | `1/5` | `1/2` |
|---|---|---|---|
| `0` | works, from `1/16` | works, from `1/16`; `×1.39` | works, from `1/24`; `×4.95` |
| `1/2` | works, from `1/24` | works, from `1/24`; `×1.32` | works, **not yet asymptotic** at `1/24 → 1/32` (orders `6.8`, `5.7`); `×4.55` |
| `9/10` | works, from `1/72` | works, from `1/72`; `×0.87` | works, from `1/72` (one pair, `1/72 → 1/80`); `×1.96`; the band inside the horizon does not converge |

The orders at the finest pair at `20 M`, and the finest row's growth of the
global error over `[10, 20] M`:

| `a` | `v` | pair | error `r3` | `r5` | global | `ℋ` `r3` | `ℋ` `r5` | growth |
|---|---|---|---|---|---|---|---|---|
| `0` | `0` | `1/24 → 1/32` | `4.03` | `4.02` | `4.49` | `4.00` | `4.00` | `1.02` |
| `0` | `1/5` | `1/24 → 1/32` | `4.05` | `4.03` | `5.46` | `4.02` | `4.02` | `1.02` |
| `0` | `1/2` | `1/24 → 1/32` | `4.13` | `4.00` | `4.62` | `4.22` | `4.33` | `1.01` |
| `1/2` | `0` | `1/32 → 1/40` | `4.03` | `4.02` | `8.49` | `3.99` | `4.00` | `1.05` |
| `1/2` | `1/5` | `1/24 → 1/32` | `4.78` | `4.20` | `8.27` | `5.59` | `6.47` | `1.00` |
| `1/2` | `1/2` | `1/24 → 1/32` | `6.79` | `5.65` | `8.52` | `7.96` | `8.27` | `1.01` |
| `9/10` | `0` | `1/80 → 1/96` | `3.99` | `4.01` | `5.21` | `3.91` | `4.00` | `1.41` |
| `9/10` | `1/5` | `1/72 → 1/96` | `3.99` | `4.00` | `6.12` | `3.95` | `4.01` | `1.08` |
| `9/10` | `1/2` | `1/72 → 1/80` | `4.00` | `4.01` | `1.21` | `4.08` | `4.42` | `1.00` |

**Transverse against aligned (measured 2026-10-25)**, at the same `a`, `v`
and `h`, at `20 M`; at `a = 9/10`, `v = 1/2` against the aligned `r_1 = 9/8`
row:

| `a` | `v` | `h` | error `r_h` | `r3` | `r5` | global | `ℋ` `r_h` | `r3` | `r5` |
|---|---|---|---|---|---|---|---|---|---|
| `0` | all | all | `0.98–1.00` | `0.97–1.00` | `0.97–0.99` | `0.98–0.99` | `0.96–1.00` | `0.98–1.00` | `0.98–1.00` |
| `1/2` | `0` | `1/24`–`1/40` | `1.00–1.02` | `0.99` | `0.98` | `1.02–1.04` | `1.00–1.02` | `0.98` | `0.99` |
| `1/2` | `1/5` | `1/24` | `4.10` | `1.37` | `1.23` | `1.75` | `3.93` | `1.63` | `2.05` |
| `1/2` | `1/5` | `1/32` | `1.59` | `1.15` | `1.17` | `1.73` | `2.00` | `1.06` | `1.04` |
| `1/2` | `1/2` | `1/24` | `7.51` | `3.47` | `2.16` | `3.71` | `4.06` | `4.68` | `6.02` |
| `1/2` | `1/2` | `1/32` | `2.54` | `1.62` | `1.43` | `2.94` | `3.06` | `2.47` | `3.74` |
| `9/10` | `0` | `1/72`, `1/96` | `0.95–0.96` | `0.94–0.95` | `0.94–0.95` | `1.00` | `0.82` | `0.92` | `0.99` |
| `9/10` | `1/5` | `1/72` | `0.82` | `0.95` | `1.00` | `2.50` | `3.00` | `0.74` | `0.91` |
| `9/10` | `1/5` | `1/96` | `0.81` | `0.97` | `1.01` | `1.50` | `0.77` | `0.74` | `0.91` |
| `9/10` | `1/2` | `1/72` | `13.9` | `0.90` | `0.95` | `0.43` | `36.1` | `0.71` | `0.91` |

- **Without spin the bitant is the rotating quadrant.** An `a = 0` hole
  moving along `x` is the aligned hole moving along `z`, turned. The two
  domains agree to `0–3 %` in every shell at every `h` and speed, and in
  `M_irr` to `1–15 %` of its offset, which is `10⁻⁷`. The spinning holes at
  rest agree to `0–8 %`, except `ℋ` just outside the `a = 9/10` horizon,
  which is `0.82×` on the bitant.
- **A spinning hole moving across its spin has an excess near the horizon,
  and it falls fast with `h`.** At `a = 1/2` the error in `r_h` is `4.1×`
  the aligned row's at `1/24` and `1.6×` at `1/32` (`v = 1/5`), and `7.5×`
  and `2.5×` at `v = 1/2`. Moving over resting in `r_h` it is `4.3×` and
  `1.9×` at `v = 1/5` (aligned `1.2×` at `1/32`), and `16×` and `7.5×` at
  `v = 1/2` (aligned `3.0×`). The orders in `r_h` from `1/24` to `1/32` are
  `7.8` and `7.6`, and they carry `r3`'s to `4.8` and `6.8`. So the term
  that dominates at `1/24` is not order four. Whether `1/40` makes the
  transverse rows the aligned ones is not measured: those two rows need
  `150` and `170 GiB`.
- **At `a = 9/10` the same excess appears at `v = 1/2` and not at `1/5`.** At
  `v = 1/5` the bitant equals the aligned row outside the horizon
  (`0.95–1.01×` in `r3` and `r5`). The global error is `2.5×` and `1.5×` the
  aligned row's at `1/72` and `1/96`, in the band inside the horizon. At
  `v = 1/2` the error in `r_h` is `28×` the resting hole's at `1/72` and
  `19×` at `1/80` (`ℋ` `140×` and `93×`), order `8.0` there. Yet `r3` and
  `r5` are at or below the aligned `9/8` row's at `1/72`, and they converge
  at order four.
- **At `a = 9/10`, `v = 1/2` the band inside the horizon does not converge.**
  Its L2 error goes from `4.35·10⁻⁴` at `1/72` to `3.87·10⁻⁴` at `1/80`
  (order `1.1`). Its L∞ rises, from `0.0156` to `0.0228`, and the layer
  residual rises with it (below). The band holds that level from `10 M` to
  `20 M` (growth `1.00–1.02`). It dominates the global norm, whose order is
  `1.2` (`ℋ` `−0.2`). The aligned `9/8` band is `2.2×` larger in L2 and
  `4.9×` in L∞ at `1/72`, and converges at `5.7` to `1/96`. `1/72 → 1/80` is a pair `1.11×` apart, so
  each order here is uncertain by a few tenths.
- **At `a = 9/10` at rest the error grows by `1.21`, `1.23` and `1.41`** over
  `[10, 20] M` at `1/72`, `1/80` and `1/96`. Those are the aligned `r` rows'
  `1.22` and `1.43`: the same layer, and the linear growth the resting
  campaign measured.

**The horizon (measured 2026-10-25)**, at the finest `h`; the drift is the
slope over `[10, 20] M` with its order against the next coarser `h`:

| `a` | `v` | `h` | `ΔM_irr` `20 M` | `|J| − a` `20 M` (aligned) | `d|J|/dt` (order) (aligned at the same `h`) |
|---|---|---|---|---|---|
| `0` | `1/2` | `1/32` | `2.09e−7` | `2.1e−9` (`7.9e−10`) | — |
| `1/2` | `0` | `1/40` | `3.19e−8` | `4.09e−7` (`4.09e−7`) | `1.74e−8` (`3.98`) (`1.75e−8`) |
| `1/2` | `1/5` | `1/32` | `6.78e−8` | `1.30e−6` (`1.04e−6`) | `5.19e−8` (`3.77`) (`4.32e−8`) |
| `1/2` | `1/2` | `1/32` | `2.91e−7` | `2.17e−6` (`1.71e−6`) | `1.00e−7` (`6.47`) (`4.48e−8`) |
| `9/10` | `0` | `1/96` | `1.85e−7` | `7.18e−6` (`7.19e−6`) | `2.07e−8` (`4.03`) (`2.15e−8`) |
| `9/10` | `1/5` | `1/96` | `1.68e−7` | `7.84e−6` (`1.10e−5`) | `1.25e−8` (`4.03`) (`1.90e−8`) |
| `9/10` | `1/2` | `1/80` | `4.17e−7` | `6.08e−6` (`1.64e−4` at `1/72`, `9/8`) | `5.14e−8` (—) |

- **At `a = 1/2` the drift of `J` is the excess again.** It is the aligned
  row's at rest, `1.2×` at `v = 1/5` and `2.2×` at `v = 1/2` at `1/32`. At
  `1/24` and `v = 1/2` it ran the other way, `−6.4·10⁻⁷/M`, `|J| − a`
  reaching `−2.1·10⁻⁵` by `20 M`; at `1/32` it is `+1.0·10⁻⁷/M`.
- **Across the spin the finder's offset does not grow with the speed.** At
  `a = 9/10`, `v = 1/2` `|J| − a` is `1.9·10⁻⁶` at `1/72` and `6.1·10⁻⁶` at
  `1/80`. That is the resting hole's `7.2–8.0·10⁻⁶` and `27×` below the
  aligned row's `1.6·10⁻⁴`, and `M_irr`'s offset is `4–6·10⁻⁷` against
  `6.9·10⁻⁶`. The previous section's guess was a less round horizon on the
  finder's `N_ah = 12` grid. These rows narrow it: across the spin the
  contraction acts on the equator, which is already the longer axis, and the
  offset stays the resting hole's. Along the spin it shortens the pole.
- At `a = 9/10`, `v = 1/2` `J`'s drift has no order (`−1.9·10⁻⁸` and
  `+5.1·10⁻⁸/M` at `1/72` and `1/80`). At rest and at `v = 1/5` it converges
  at order four, as along the spin.

The **layer residual** at `20 M` at each spacing, with the aligned rows':

| `a` | `v = 0` | `v = 1/5` | `v = 1/2` |
|---|---|---|---|
| `0` | `0.319`, `0.189`, `0.101` | `0.403`, `0.365`, `0.279` | `2.56`, `1.75`, `1.25` |
| `1/2` | `0.373`, `0.218`, `0.146` | `0.452`, `0.259` (aligned `0.357`, `0.202`) | `0.971`, `0.998` (aligned `0.364`, `0.263`) |
| `9/10` | `2.51`, `2.08`, `1.50` | `2.40`, `1.58` (aligned `1.60`, `0.955`) | `4.37`, `6.20` at `1/72`, `1/80` (aligned `9/8`: `4.67` at `1/72`) |

- At `a = 0` and at rest these are the aligned rows' to three digits.
- **At `v = 1/2` across the spin the residual does not fall**: at `a = 1/2`
  it is `0.97` and `1.00` at `1/24` and `1/32`, `4×` the aligned row's, and
  at `a = 9/10` it rises `1.4×` from `1/72` to `1/80`. At `v = 1/5` it
  converges at order `1.9` (`a = 1/2`) and `1.5` (`a = 9/10`), from
  `1.3–1.5×` the aligned rows'. The previous section traced the residual at
  `v = 1/2` to the point the moving core has just released on its trailing
  side. Across the spin the ring lies in the plane of the motion and is
  contracted along it, which the core's target sees and the aligned core's
  does not. That this is the cause is a guess. The trailing/leading split
  (`split=1`) is the measurement that would test it.

**What a run costs** (measured 2026-10-25) on one H200, `chunk = M`, no
split; `s/M` over `[10, 20] M`:

| row | points | steps per `M` | `s/M` | `20 M` | GPU memory | ns per point and step |
|---|---|---|---|---|---|---|
| `a = 0`, `v = 0`, `1/32` | `43.3 M` | `111` | `139` | `0.80 h` | `71 GiB` | `29` |
| `a = 0`, `v = 1/2`, `1/32` | `59.8 M` | `112` | `195` | `1.08 h` | `93 GiB` | `29` |
| `a = 1/2`, `v = 0`, `1/40` | `84.5 M` | `139` | `307` | `1.88 h` | `122 GiB` | `26` |
| `a = 1/2`, `v = 1/2`, `1/32` | `59.8 M` | `112` | `201` | `1.11 h` | `93 GiB` | `30` |
| `a = 9/10`, `v = 0`, `1/96` | `53.5 M` | `333` | `482` | `2.66 h` | `88 GiB` | `27` |
| `a = 9/10`, `v = 1/5`, `1/96` | `89.1 M` | `336` | `870` | `4.85 h` | `136 GiB` | `29` |
| `a = 9/10`, `v = 1/2`, `1/80` | `80.9 M` | `280` | `688` | `3.83 h` | `129 GiB` | `30` |

A point and step cost `26–39 ns` from `1/24` on and `53–56 ns` at `1/16`.
That is the aligned rows' cost per point, and a row costs about twice the
aligned row of the same `a`, `v` and `h` because the bitant has twice the
points.

**What it says.** A spinning hole moving across its spin, with the analytic
layer at the default `4/M`, the target's rate and the core at its transit
rate, holds every cell that fits one H200 to `20 M` without growth, and its
error outside the horizon converges at order four. At `a = 1/2`, `v = 1/2`
the finest pair is not yet in that regime. Across the spin one layer serves
every speed at every spin, including `r_1 = 5/4` at `a = 9/10`, which along
the spin cannot run at `v = 1/2`. What the motion across the spin adds is a
term near the horizon that is large at the coarser spacings. At `1/24` it is
`4–8×` the aligned hole's error there at `a = 1/2`, and it falls at order
`7–8`, to `1.6–2.5×` at `1/32`. With it come a layer residual that does not
fall at `v = 1/2`, and at `a = 9/10`, `v = 1/2` a band inside the horizon
that holds steady but does not converge. The finder's offset across the spin
is the resting hole's.

**What is open** (2026-10-25):

- **Whether the excess at `a = 1/2` is gone at `1/40`.** `a = 1/2` at
  `v = 1/5` and `1/2` at `1/40` (`102 M` and `117 M` points) need two H200s
  each (`test/octant_runs.jl mpi=true`). So does `a = 9/10`, `v = 1/2` at
  `1/96` (`140 M`), the only way to a second pair for that cell and a test of
  its band inside the horizon.
- **Where the excess comes from.** Run the trailing/leading split on
  `a = 1/2`, `v = 1/2` at `1/24`. That says whether the residual sits at the
  released point, as along the spin. A core target that matches
  `∂_r u_exact` at `r_0`, `CODE.md`'s untried remedy for the first-order
  residual, would be tested on these rows too.
- **G5's case itself**, the harmonic chart with `:fitted`, is unchanged.
  These rows show that the Kerr-Schild chart's analytic layer holds the same
  geometry.
- **Circular orbits**; unchanged.

## The damped harmonic gauge rolled on (step B1, 2026-10-25)

`PLAN.md`'s step B1 and `CODE.md`'s [The gauge](CODE.md#the-gauge): the damped
harmonic gauge of Szilágyi, Lindblom and Scheel in SpECTRE's form, rolled on from
the hole's own Kerr-Schild source, on the boosted `a = 0` hole of [the boosted
Kerr-Schild hole at binary
speed](#the-boosted-kerr-schild-hole-at-binary-speed-2026-10-20) and on the same hole
at rest, against both in their own Kerr-Schild gauge. It is pairing (β)'s first
run: the gauge with `:fitted`, because once the gauge has moved the analytic
`:damped` target is in the wrong one. Single holes only; the binary is step B3's.

**The setup** (commit `a22122e`, `test/octant_runs.jl gauge=dh`):

- **The gauge.** `RolledGaugeSource(KerrSchildSource, DampedHarmonicSource(σ_r = 32,
  A_L1 = A_S = 1, A_L2 = 0, e = 2); σ_g = 10, t_0 = 0)` **(proposed in step B1:**
  SpEC rolls on over `15–25 M`; `10 M` leaves half of a `40 M` run with the roll-on
  done, `f(15) = 0.006` and `f(20) = 1.1·10⁻⁷`**)**. `f` underflows only at `52 M`,
  so these runs evaluate both sources to the end.
- **Everything else is `bf24n`'s** (the `:fitted` row of the boosted hole without
  `trail_ramp`): Kerr-Schild `a = 0`, at `x = −4 + t/5` or at rest at `x = −4`, on
  the quadrant `[−64, 64] × [0, 64]²` and its capsule of radii `32, 16, 8, 4` swept
  along `x ∈ [−4, 4]` (856 blocks; `N = 24, 32`: `h = 1/24, 1/32` at the hole,
  `11.83 M` and `28.05 M` points); `:fitted` with `m = 16`, `n_L = 20`,
  `fit_cont = 2`, `lmax_fit = 8`; `q = 4`, `cfl = 1/2`, `ε_KO = 1/2`, the Gaussian
  `γ0`; the finder every chunk; to `40 M`. **New:** `chunk = M` with the split
  (the folds of 2026-10-23 made it cheap), `checkpoint_every = 4`.
- **The runs**, one H200 each, in `step-B1-gauge` on Symmetry (TreeAMR 0.2.6, CUDA
  added): `b1dm24` (damped harmonic, moving), `b1ds24`, `b1ds32` (damped
  harmonic, at rest), `b1km24`, `b1km32`, `b1ks24`, `b1ks32` (Kerr-Schild, the
  controls), and `b1dm24t`, `b1dm32t`, the moving damped harmonic rows again
  with the shells about the tracked horizon (`shell_center=track`, below) — jobs
  582577 (`b1km32`), 582578 (`b1ds32`), 582579 (`b1ks32`), 582580 (`b1dm24`),
  582581 (`b1km24`), 582582 and 582643 (`b1ds24`, pre-empted at its start and
  resubmitted), 582583 (`b1ks24`), 582625 (`b1dm24t`), 582624, 582634 and 582663
  (`b1dm32t`, pre-empted twice and continued from its checkpoints), all on
  `h200preq`; `b1dm32` (582574, 582585) was pre-empted at its start and then
  cancelled for `b1dm32t`. The CSVs and logs are in `.claude/data/step-B1`, and `julia
  test/gauge_study.jl dir=.claude/data/step-B1` prints every table below.

The norms are point-weighted L2 in shells — `[2, 2.25)` just outside the
horizon, `[3, 5)`, `[5, 8)` — about the analytic center `c(t)`, or, in
`b1dm24t` and `b1dm32t`, about the tracked horizon's; `C_a` is the largest
component's. Under the damped harmonic gauge the analytic center is the
Kerr-Schild hole's, not the evolved hole's: the gauge moves the coordinates, and
the found horizon's origin is the hole's position.

**Every run reached `40 M`, every find succeeded, and the range projection never
fired.** On one H200 at `chunk = M` with the split: `32–37 s/M` at `1/24` and
`72–81 s/M` at `1/32` — the damped harmonic rows about `12 %` dearer than the
Kerr-Schild ones (`37` against `33 s/M` moving at `1/24`, `81` against `72` at
rest at `1/32`), the roll-on evaluating both sources to the end. A `40 M` row is
`22–25 min` at `1/24` and `48–55 min` at `1/32` on one H200, uninterrupted.

**The gauge moves the hole, and grows its horizon in coordinates.** The horizon's
least coordinate radius about its own origin goes from `2.00` (at rest) and `1.96`
(moving, the boost's contraction) to `2.06` in both by `20 M`, and its mean radius
to `2.07`: under this gauge the moving hole's horizon is no longer contracted
(least over mean `0.995` against Kerr-Schild's `0.985`). The moving hole falls
behind the analytic center `c(t) = −4 + t/5`, by `0.11`, `0.34`, `0.44` and
`0.49 M` at `10`, `20`, `30` and `40 M`, its coordinate speed back to `0.196` at
the end; the hole at rest drifts `+0.010 M` along `x`, towards the Gaussian's
center at the origin, slowing (`0.0069`, `0.0092`, `0.0099` at `20`, `30`, `40 M`).
So two things in the record are not what they were for the Kerr-Schild runs:

- a shell about `c(t)` is not about the moving hole — `b1dm24`, with the shells
  about `c(t)`, read `ℋ` `1.2·10⁻²` in `[2, 2.25)` at `40 M` against `2.4·10⁻³`
  about the hole (`b1dm24t`, `shell_center=track`; the evolution is the same bit
  for bit, the record's `M_irr` and positions agree to every digit) — and
- `[2, 2.25)` straddles the horizon at `2.06`, a quarter of it inside, so its
  norms mix the horizon's inside with its outside. `[3, 5)` and `[5, 8)` are
  outside it in either gauge, and are what the comparisons below read.

**The hole at rest settles, at the Kerr-Schild hole's level outside `r = 3`.**
Damped harmonic over Kerr-Schild at `1/32`, the hole at rest:

| shell | `ℋ` at `10, 20, 30, 40 M` | `C_a` at `10, 20, 30, 40 M` |
|---|---|---|
| `[2, 2.25)` (straddling) | `1.23`, `3.52`, `3.27`, `3.26` | `1.32`, `2.90`, `2.74`, `2.76` |
| `[3, 5)` | `1.24`, `1.59`, `1.60`, `1.60` | `1.07`, `1.31`, `1.32`, `1.29` |
| `[5, 8)` | `1.09`, `1.17`, `1.19`, `1.18` | `1.01`, `1.25`, `1.26`, `1.30` |

- Every ratio is flat from `20 M` on, and the state's distance from the
  Kerr-Schild solution in `[2, 2.25)` is `0.343`, `0.333`, `0.339`, `0.338` at
  `15`, `20`, `30`, `40 M`: the solution has moved to another gauge and stopped
  there — the time-independent state Lindblom and Szilágyi saw a hole reach.
- `M_irr − 1` is `1.73·10⁻⁷` at `40 M` against the Kerr-Schild hole's
  `1.46·10⁻⁷`: the horizon is the same hole.
- At `1/24` the ratios are the same (`[3, 5)` `1.63` and `1.23`, `[5, 8)` `1.20`
  and `1.25` at `40 M`), and from `1/24` to `1/32` `ℋ` converges at order
  `3.96–4.04` in `[3, 5)` and `[5, 8)`, as the Kerr-Schild hole's does
  (`3.97–4.03`): the excess is the new chart's own truncation error, not a
  transient. The drift, `+0.00995 M` at both spacings, is the gauge's, and
  `M_irr` converges at order `3.8` (`1 + 5.2·10⁻⁷` at `1/24`). Just outside the
  horizon, in the straddling `[2, 2.25)`, neither gauge converges (`0.5–0.8`),
  the `:fitted` layer's known floor at `m = 16`.

**The moving hole does not, at `1/24`.** Damped harmonic (`b1dm24t`) over
Kerr-Schild (`b1km24`), the moving hole, the shells about the hole:

| shell | `ℋ` at `10, 20, 30, 40 M` | `C_a` at `10, 20, 30, 40 M` |
|---|---|---|
| `[2, 2.25)` (straddling) | `0.78`, `28`, `40`, `44` | `1.2`, `49`, `86`, `100` |
| `[3, 5)` | `1.0`, `1.7`, `61`, `118` | `1.0`, `1.4`, `55`, `195` |
| `[5, 8)` | `0.92`, `1.0`, `13`, `72` | `1.0`, `1.2`, `7.4`, `69` |

- It starts when the roll-on ends: the ratios are about one to `10 M`, and from
  `15 M` (`f = 0.006`) the excess grows at the horizon and travels out, reaching
  `[5, 8)` by `30 M`, still growing at `40 M`.
- `M_irr − 1` is `+2.2·10⁻⁷` at `15 M`, `−4.8·10⁻⁶` at `20 M` and `−1.3·10⁻⁵` at
  `40 M`, against the Kerr-Schild hole's `+4.5·10⁻⁷`: the horizon itself is
  `30×` worse, and that does not depend on where a shell is.
- **At `1/32` the excess is much smaller, and falls faster than the scheme's
  order.** Damped harmonic over Kerr-Schild at `1/32` (`b1dm32t` over `b1km32`),
  at `20`, `30`, `40 M`: `ℋ` in `[3, 5)` `1.16`, `6.7`, `6.3` and `C_a` `1.26`,
  `9.7`, `15`; `ℋ` in `[5, 8)` `1.03`, `1.7`, `9.9` and `C_a` `1.03`, `1.9`, `31`;
  in the straddling `[2, 2.25)` `7.8`, `12`, `13`. From `1/24` to `1/32`, from
  `30 M` on, `ℋ` falls at order `7.8–10` in `[3, 5)` and `5.5–7.6` in `[5, 8)`
  (`3.2–3.6` in `[2, 2.25)`), and `M_irr − 1` is `−1.0·10⁻⁶` at `40 M` against
  `−1.3·10⁻⁵`: an under-resolved feature of the moving hole under the new gauge,
  not a gauge that fails. The lag is the same at both spacings to every printed
  digit, so it is the gauge's, as the resting hole's drift is. Whether `1/48`
  puts the moving hole at the Kerr-Schild hole's level is the next measurement;
  on this capsule it does not fit one H200 with `:fitted` (about `171 GB`,
  `CODE.md`'s recommended settings), so it needs `mpi=true` or a mesh that
  follows the hole.

**The faces: the harmonic far field against Kerr-Schild Dirichlet data.** `C_a`
in the blocks touching an outer face is `2.0·10⁻⁵`, `9.5·10⁻⁵`, `1.6·10⁻⁴`,
`2.4·10⁻⁴` at `10`, `20`, `30`, `40 M` under the damped harmonic gauge, at rest and
moving alike, against `10⁻¹¹` under Kerr-Schild's (`ℋ` there `5·10⁻⁵` against
`4·10⁻¹²`), still growing linearly at `40 M`; over the whole evolved region the
faces are most of the damped harmonic runs' `C_a` (`7.4·10⁻⁵` at rest at `40 M`).
`2M/r²` is `4.9·10⁻⁴` at `64 M` ([What step B1 built and
measured](CODE.md#what-step-b1-built-and-measured-2026-10-25)). It does not reach
the shells about the hole by `40 M`.

**What this says about the pairings** (`CODE.md`, "The gauge"): the damped harmonic
gauge with `:fitted` holds a hole at rest at the Kerr-Schild level outside
`r = 3` — `1.2–1.6×`, converging at order four — and its horizon at the
Kerr-Schild hole's `M_irr`. A moving hole it holds less well: from the end of
the roll-on its constraints grow, `70–200×` the same hole's in its own gauge by
`40 M` at `1/24` and `6–31×` at `1/32`, falling with `h` faster than the
scheme's order. The analytic `:damped` layer does not survive the gauge moving
at all (the smoke runs below). Pairing (α) — the tracked Kerr-Schild source with
`:damped` — is the accurate one on everything measured here; (β) needs a finer
moving hole than one H200 holds on this capsule to show whether it reaches the
same level, and a boundary condition of its own.


**The local smoke runs (2026-10-25, the development machine, six threads).** The
moving and the resting hole at `h = 1/16` on a quadrant of half-width `16`
(`radii = 8,6,4,3`), with the `:damped` layer that is the production layer at that
spacing (`r_0 = 5/8`, `r_1 = 23/16`, `4/M` with the target's rate), a fast roll-on
(`σ_g = 1`), to `4 M`:

| run | `ℋ` `[2, 2.25)` at `1, 2, 3, 4 M` | `C_a` `[2, 2.25)` | `C_a` at the faces |
|---|---|---|---|
| Kerr-Schild, moving | `6.2·10⁻⁶`, `8.6·10⁻⁶`, `1.5·10⁻⁵`, `1.9·10⁻⁵` | `2.0·10⁻⁶` … `4.3·10⁻⁶` | `2.8·10⁻⁹` |
| damped harmonic, at rest | `6.5·10⁻⁶`, `1.5·10⁻⁵`, `1.3·10⁻⁴`, `2.7·10⁻⁴` | `1.8·10⁻⁶` … `2.7·10⁻⁵` | `1.3·10⁻⁴` … `1.4·10⁻³` |
| damped harmonic, moving | `6.5·10⁻⁶`, `8.3·10⁻⁵`, `5.3·10⁻⁴`, — | `1.9·10⁻⁶` … `9.8·10⁻⁵` | `1.3·10⁻⁴` … `1.1·10⁻³` |

- **The analytic `:damped` layer does not survive the gauge moving.** The moving
  hole's layer residual — its distance from the Kerr-Schild target it relaxes to —
  went `0.73`, `6.7`, `36`, `100`, `350`, `1616` at `t = 1/2 … 3`, and a `sqrt` of a
  negative number ended it after `3 M`. At rest the residual rose to `14` by `3 M`
  and stayed there (the Kerr-Schild hole's: `0.12–0.21`), and `ℋ` just outside the
  horizon was `15×` the Kerr-Schild hole's at `4 M`. The layer
  holds the interior in Kerr-Schild's gauge while the source moves the exterior
  out of it. That is `CODE.md`'s reason to pair this gauge with `:fitted`, measured.
- **The faces' `C_a` is the gauge's far field**, `10⁵×` the Kerr-Schild run's:
  the harmonic far field against Kerr-Schild Dirichlet data, `2M/r²` at a box of
  `16 M` ([What step B1 built and
  measured](CODE.md#what-step-b1-built-and-measured-2026-10-25)).
- A `:fitted` smoke run (`h = 1/24` on a box of `8 M`, `m = 16`, `n_L = 20`,
  `σ_g = 1`, moving) reached `2 M`, the found center `0.074 M` behind the analytic
  one.

## The tracked analytic target (step B2, 2026-10-26)

`PLAN.md`'s step B2 and `CODE.md`'s [The tracks](CODE.md#the-tracks) and [One
interior per hole](CODE.md#one-interior-per-hole): the moving `:damped` layer of [the
moving `:damped` layer's target rate](#the-moving-damped-layers-target-rate-2026-10-21)
— the accurate interior — with its target and its center taken from the horizon
track instead of the analytic background, which a binary does not have. The sphere is
the same (`r_0 = 5/8`, `r_1 = 23/16`, `4/M` with the target's rate, its core
following its target); what changes is that the target is the boosted Kerr-Schild hole
the track says is there, rebuilt at every piece's midpoint, and the track predicts
along the quadratic of its last four finds ([What step B2 built and
measured](CODE.md#what-step-b2-built-and-measured-2026-10-26)). One hole, in its own
Kerr-Schild gauge, where it moves at exactly `v`: with exact finds the tracked target
is the analytic one, so what is measured is the price of the finds' noise, and of
each perturbation of the target's parameters a binary's target will carry —
its mass, its velocity, its center — and whether a track started wrong recovers.

**The setup** (commit `8012032`, `test/octant_runs.jl interior=tracked`): the moving
`:damped` production row of [the moving `:damped` layer's target
rate](#the-moving-damped-layers-target-rate-2026-10-21) — Kerr-Schild `a = 0` at
`x = −4 + t/5` on the quadrant `[−64, 64] × [0, 64]²` and its capsule of radii `32,
16, 8, 4` swept along `x ∈ [−4, 4]` (856 blocks; `N = 24, 32`: `h = 1/24, 1/32` at
the hole, `11.83 M` and `28.05 M` points), `r_0 = 5/8`, `r_1 = 23/16`, `ρ_ramp = 1`,
`4/M` with the target's rate, `q = 4`, `cfl = 1/2`, `ε_KO = 1/2`, the Gaussian `γ0`,
the algebraic source, the finder every chunk with the spin — at `chunk = M` without
the split (the production cadence of 2026-10-22), to `40 M`, `checkpoint_every = 4`.
The tracked rows predict along the quadratic; the hole's acceleration is zero, so the
cadence is one find a chunk, the row's, and the target is rebuilt at the midpoints of
`20` pieces a chunk at `1/24` (`26` at `1/32`).

- **The runs**, one H200 each, in `step-B2-tracked` on Symmetry (TreeAMR 0.2.6, CUDA
  added), all on `h200preq`: the analytic `:damped` control `b2a24`, `b2a32` (jobs
  582992, 582995) — rerun on this commit and cadence, so that no comparison crosses
  either — and the tracked target `b2t24`, `b2t32` (582988, 582994); at `1/24` the
  target's mass times `1.01` and `0.99` (`b2mp24`, `b2mm24`: 582996, 582997), its
  velocity `+0.02` and `−0.02` along `x` (`b2vp24`, `b2vm24`: 582998, pre-empted
  while it wrote its `40 M` checkpoint and continued from `36 M` as 583116 for its
  record; 583000), its
  center one cell (`1/24`) ahead and behind along `x` (`b2xp24`, `b2xm24`: 583001;
  583003 pre-empted at `10 M` and continued from its `8 M` checkpoint as 583088),
  and the seed track `0.05` too fast (`b2r24`: 583004, pre-empted at `6 M` and
  continued from `4 M` as 583091, pre-empted again at `17 M` and continued from
  `16 M` as 583105). A pre-empted job here ended `FAILED` and was not requeued
  (`CLAUDE.md`, "Things that will bite"), so each was resubmitted by hand; every
  continued row wrote again the rows its predecessor had written after the
  checkpoint, identical to the CSV's ten digits but for the wall clock — the chain
  is the run on an H200 too. The CSVs
  and logs are in `.claude/data/step-B2`, and `julia test/tracked_study.jl
  dir=.claude/data/step-B2` prints every table below.
- **Every run reached `40 M`, every find succeeded, the range projection never
  fired**, and the tracked target costs what the analytic layer does: `37–40 s/M` at
  `1/24` and `88–89 s/M` at `1/32` on one H200, a `40 M` row `25 min` and `59 min`.

**The tracked target is the analytic layer, to the tracking.** The tracked rows over the
analytic ones at `1, 2, 4, 8, 16, 24, 32, 40 M`, in the shells just outside the horizon
`[2, 2.25)`, `[3, 5)` and `[5, 8)` and over the evolved region:

| | `1/24` | `1/32` |
|---|---|---|
| `ℋ` in `[2, 2.25)` | `0.9992–1.0009` | `1.0000` |
| the error in `[2, 2.25)` | `0.9998–1.0003` | `1.0000` |
| `ℋ` and the error in `[3, 5)` and `[5, 8)` | `0.9999–1.0001` | `1.0000` |
| `ℋ` over the evolved region | `0.9965–1.0026` | `0.9997–1.0000` |
| the error over the evolved region | `0.9986–1.0017` | `0.9992–1.0001` |
| the layer residual (an L∞ over the layer) | `0.89–1.12` (`0.12–0.15` both) | `0.81–1.00` (`0.09–0.12`) |

- **Order four**, from `1/24` to `1/32`, the tracked target's and the analytic layer's
  to the printed digit: `ℋ` `4.01–4.15` in `[2, 2.25)`, `4.00–4.02` in `[3, 5)`,
  `3.97–4.01` in `[5, 8)`; the error `3.97–4.05`, `4.00–4.04` and `3.99–4.04`.
- **The track**: `1.2·10⁻⁴` cells off the analytic center at `40 M` at `1/24` and
  `5.2·10⁻⁵` at `1/32` — the found center running ahead by the truncation error the
  analytic rows' finder already showed (`1.6·10⁻⁶ M` at `1/32`, order four) — and every
  prediction within `6·10⁻⁷` cells of its find; its velocity `0.2` to `3·10⁻⁷`.
  `M_irr − 1` is the analytic rows' to every printed digit (`4.3·10⁻⁷` and
  `1.3·10⁻⁷` at `40 M`).

**The sensitivity.** Each perturbation over the tracked target at `1/24`, at
`1, 2, 4, 8, 16, 24, 32, 40 M`:

| perturbation | `ℋ` in `[2, 2.25)` | the error there | `ℋ` in `[3, 5)` | the layer residual | the error over the evolved region |
|---|---|---|---|---|---|
| mass `+1 %` | `1.07`, `2.95`, `7.71`, `7.63`, `7.42`, `7.32`, `7.30`, `7.39` | `1.01` → `2.73` at `4 M` → `1.22` | `1.00` → `1.19` at `8 M` → `0.97` | `6.5–8.4` | `10.9` → `3.8` |
| mass `−1 %` | `1.03`, `3.19`, `7.76`, `7.88`, `7.65`, `7.56`, `7.56`, `7.61` | `1.01` → `2.46` at `8 M` → `1.17` | `1.00` → `1.23` at `8 M` → `1.06` | `7.0–8.3` | `10.5` → `3.6` |
| velocity `+0.02` | `1.28`, `7.65`, `19.9`, `18.3`, `17.6`, `17.4`, `17.4`, `17.6` | `1.14` → `8.20` at `4 M` → `2.32` | `1.00` → `2.52` at `8 M` → `1.09` | `14.1–17.3` | `32.7` → `11.7` |
| velocity `−0.02` | `1.35`, `7.61`, `18.7`, `18.0`, `17.3`, `17.1`, `17.1`, `17.2` | `1.13` → `7.60` at `4 M` → `2.27` | `1.00` → `2.54` at `8 M` → `1.24` | `13.7–16.4` | `31.2` → `10.9` |
| center `+1` cell | `1.20`, `6.29`, `16.6`, `16.3`, `15.8`, `15.8`, `15.7`, `15.9` | `1.05` → `5.00` at `4 M` → `1.62` | `1.00` → `1.92` at `8 M` → `1.07` | `69–84` | `26.2` → `9.1` |
| center `−1` cell | `2.02`, `11.7`, `28.9`, `26.9`, `26.2`, `26.1`, `26.1`, `26.2` | `1.20` → `7.59` at `4 M` → `1.60` | `1.00` → `3.08` at `8 M` → `1.25` | `53–65` | `28.3` → `10.5` |
| seed velocity `+0.05` | `2.00`, `12.5`, `35.6`, `6.40`, `1.42`, `1.02`, `1.00`, `1.00` | `1.44` → `13.6` at `4 M` → `1.00` from `32 M` | `1.00` → `14.5` at `8 M` → `1.00` at `32 M` | `66.7`, `33.7`, then `1.00–1.13` from `4 M` | `100` → `1.00` from `32 M` |

- **A wrong target is a steady excess just outside the horizon, and nowhere else.**
  Each perturbation raises `ℋ` in `[2, 2.25)` within `4 M` to a level it then keeps to
  `40 M` — `7.4–7.9×` for the mass, `17–20×` for the velocity — the two signs
  alike — and `16×` for the center a cell ahead, `26×` a cell behind, on the side
  the layer releases its points. Farther out the excess is the transient the
  start sends out, one pulse through `[3, 5)` at `8 M` and `[5, 8)` at `16 M`, and
  `ℋ` is back at the tracked target's (`0.96–1.25×`) by `24–40 M`; the error just
  outside the horizon decays the same way, to `1.2–2.3×` at `40 M`. (Behind the hole
  the target holds back the late error of the hole's position — the found center's
  truncation-order lead — and the error in `[3, 5)` and `[5, 8)` is `0.40` and `0.53`
  of the tracked target's at `40 M`.) Step 8c's rule — a layer at `4/M` on a ramp
  of at least `4G` cells leaves the exterior what the exact target leaves — holds
  from `r = 3` out, and not in the band between the
  layer and the horizon's `2 M`, which a wrong target holds `7–26×` above the exact
  one's level.
- **The scale, per unit of error.** The layer residual — its L∞ distance from the
  true solution — is `7–8×` the exact target's for `1 %` of mass, `14–17×` for `0.02`
  of velocity and `53–84×` for a cell of center, and `ℋ` just outside the horizon
  `7.5×`, `18×` and `16–26×`: as the residual for the mass and the velocity, and much
  less for the center, whose error is largest deep in the layer, near the core,
  where the residual's L∞ lives. What a binary's target misses — the companion's
  tidal field over the layer, `M_B r²/d³` (`5·10⁻⁴` at `d = 10`, `0.02` at
  `d = 3`), and the orbit's acceleration across it (`0.005` and `0.06`, `CODE.md`, "One interior per
  hole") — is at `d = 10` a twentieth of the `1 %` the mass row carries, and at
  `d = 3` twice it.
- **The recovery is the track's, in one find, and the run's, in one crossing.** The
  first find, at `t = 0`, keeps the seed's velocity, which carries the target
  `0.05 M` ahead of the hole by the second, at `1 M` — `1.2` cells, inside the
  check's `1.5` at `1/24` — and that find's chord is the hole's `0.2000`; from
  `2 M` on the track is the tracked target's — `10⁻⁶–10⁻⁵` cells off the analytic
  center to `16 M`, `1.2·10⁻⁴` at `40 M` — and its predictions within `7·10⁻⁶` cells of the finds. The layer
  residual is `8.2` and `5.0` at `1` and `2 M` and the exact target's from `4 M`. What the first two `M` left in the layer travels out — `ℋ`
  `36×` just outside the horizon at `4 M`, `14.5×` in `[3, 5)` at `8 M`, `14×` in
  `[5, 8)` at `16 M` — and leaves: by `40 M` every shell and the evolved region are
  the tracked target's to `0.6 %` (`[5, 8)` still `1.3×` in `ℋ` at `32 M`), and
  `M_irr` is its to two digits (`4.30·10⁻⁷` against `4.28·10⁻⁷`). No refusal; the
  row was pre-empted twice (above) and continued from its checkpoints at `4` and
  `16 M`.

**The local smoke runs (2026-10-26, the development machine, three threads each,
loaded 13–28 by other work).** B1's smoke setup — the moving hole at `h = 1/16` on a
quadrant of half-width `16` (`radii = 8,6,4,3`, `x0 = −1`, `v = 1/5`, `chunk = M/2`,
`cfl = 1/2`), the production layer at that spacing — to `4 M`, each run about
`13–15 min`; `.claude/data/step-B2-cpu`. The analytic layer (`s16a`), the tracked
target (`s16t`), its center one cell (`0.0625 M`) ahead along `x` (`s16x`,
`target_dx=1`), and the track seeded `0.05` too fast (`s16r`, `seed_dv=1/20`); each
over the analytic layer at `1, 2, 3, 4 M`:

| run | `ℋ` in `[2, 2.25)` | the error there | `ℋ` in `[3, 5)` | the layer residual | `ℋ` over the evolved region |
|---|---|---|---|---|---|
| tracked | `1.000`, `1.001`, `1.000`, `1.000` | `1.000`, `1.001`, `1.000`, `1.000` | `1.000` at every time | `1.000` (`0.21`, `0.12`, `0.13`, `0.16`) | `1.001`, `1.000`, `1.000`, `1.002` |
| center `+1` cell | `2.3`, `9.4`, `9.7`, `11.0` | `1.5`, `4.6`, `6.7`, `8.4` | `1.00`, `1.00`, `1.07`, `1.52` | `78`, `121`, `127`, `106` (`15–17`) | `32`, `15`, `11`, `11` |
| seed `+0.05` | `3.5`, `9.0`, `4.6`, `3.0` | `2.9`, `7.4`, `4.5`, `2.9` | `1.00`, `1.00`, `1.64`, `5.4` | `20`, `2.5`, `0.99`, `1.00` | `40`, `13`, `5.4`, `2.2` |

- **The tracked target is the analytic layer.** Every number agrees to `0.2 %`, the
  track `≤ 9·10⁻⁶` cells off the analytic center and every prediction within
  `5·10⁻⁶` cells of the find; `[5, 8)` agrees to every printed digit in all four
  runs.
- **A target one cell off is not held.** The layer relaxes toward a hole a cell
  ahead of the one it holds — a residual of `16`, a hundred times the exact
  target's — and the mismatch leaves through `r_1`: `ℋ` just outside the horizon
  `10×` the analytic layer's after `2 M`, and on its way out by `4 M`. The tracked
  center is good to `10⁻⁵` cells; a cell is the scale of a find that has converged
  to the wrong surface, which the center-jump check refuses.
- **The recovery is the track's, in two finds**: the first, at `t = 0`, keeps the
  seed's velocity, the second, at `t = 1/2`, finds the hole `0.025 M` (`0.4` cells)
  off the prediction, and the chord of the two is the hole's `0.2000`; from `t = 1`
  the track is within `3·10⁻⁵` cells, the residual the analytic layer's from `3 M`. What the half `M` of a target running ahead left in
  the layer travels out — `ℋ` in `[3, 5)` `5.4×` at `4 M` — which the `40 M` row
  follows.

**What it says.** The tracked analytic target is the accurate moving `:damped` layer
without the analytic background: on one hole in its own gauge it is the analytic layer
to the tracking, `10⁻³` and order four at both spacings, at the analytic layer's cost,
and a track started wrong recovers in two finds without a refusal. A target whose
parameters are wrong is the layer's business and the band's just outside the horizon:
`1 %` of mass costs `7.5×` there, `0.02` of velocity `18×`, a cell of center `16–26×`,
steadily, and nothing from `r = 3` out once the start's transient has passed. For
pairing (α) that puts the open question where `CODE.md`'s "One interior per hole"
put it — whether a target without the companion's tidal field is good enough late
in the inspiral — with a scale: at `d = 10` its error is a twentieth of the mass
row's, at `d = 3` twice it. Beside step B1's (β), `6–31×` the Kerr-Schild hole outside
`r = 3` at `1/32` and still growing, the numbers favour (α).

## A layer-free interior (step I1)

**Not yet run (2026-10-30).** The design is `CODE.md`'s [A layer-free interior
(proposed in step I1)](CODE.md#a-layer-free-interior-proposed-in-step-i1), the
brief `PLAN.md`'s step I1. Step I1's implementation job built the code, the
tests and the run tools and ran no row of the screen; the lines below are what
the next session launches, one H200 a row, each study in its own remote
directory (`CLAUDE.md`: never rsync into a directory whose jobs are running), from
a copy with `CUDA` added to its `Project.toml`. The rows share the baseline's
case and mesh — Kerr-Schild `a = 0` on the reflecting octant `[0, 64]³`, root
brick `2³`, cubes `R = 32, 16, 8`, `h = 1/16` on `[0, 8]³` (9.4 M points),
`q = 4`, `cfl = 1/2`, `ε_KO = 1/2`, the Gaussian `γ0`, `chunk = 1`, the finder
every chunk — and the base point R3 is every layer-free knob's default
(`test/octant_runs.jl`'s header): `ε = 1`, `ΔH` on, V with `p = 1`,
`μ_α = 4/M`, `μ_β = 0`, the ramp `[1, 3/2)`, the guard's ranges gated at `3/2`,
no drivers. `R0` is the `:damped` baseline on this commit, to `24 M`; every
other row is `8 M`.

```bash
B="julia --project=. test/octant_runs.jl backend=cuda case=ks L=64 roots=2 radii=32,16,8 N=64 chunk=1 cfl=1/2 amplitude=0 group=step-I1"
$B t_end=24 r_0=3/4 r_1=3/2 out=out/i1/R0
$B t_end=8 interior=layerfree smooth=0 r_core=1/4 steer=none correction=0 out=out/i1/R1
$B t_end=8 interior=layerfree steer=none out=out/i1/R2
$B t_end=8 interior=layerfree out=out/i1/R3
$B t_end=8 interior=layerfree p=2 out=out/i1/R4
$B t_end=8 interior=layerfree mu_alpha=1 out=out/i1/R5
$B t_end=8 interior=layerfree steer=f kappa=1/3 start=collapsed out=out/i1/R6
$B t_end=8 interior=layerfree steer=f kappa=1/3 start=rolled sigma_g=4 out=out/i1/R7
$B t_end=8 interior=layerfree mu_beta=1 out=out/i1/R8
$B t_end=8 interior=layerfree correction=0 out=out/i1/R9
$B t_end=8 interior=layerfree smooth=3/4 out=out/i1/R10
$B t_end=8 interior=layerfree smooth=5/4 out=out/i1/R11
$B t_end=8 interior=layerfree eps_in=2 out=out/i1/R12
$B t_end=8 interior=layerfree gamma0_near=1/10 out=out/i1/R13
$B t_end=8 interior=layerfree rho_a=1/10 out=out/i1/R14
$B t_end=8 interior=layerfree alpha_min=1e-6 out=out/i1/R15
$B t_end=8 interior=layerfree alpha_min=17/100 lambda_min=1/2 lambda_max=10 K_max=17 out=out/i1/R16
$B t_end=8 interior=layerfree r_gate=everywhere out=out/i1/R17
julia test/layerfree_study.jl dir=out/i1 times=1,2,4,8 at=8
```

| row | what it is |
|---|---|
| R0 | `:damped`, `r_0 = 3/4`, `r_1 = 3/2`, the baseline, to `24 M`: must give `ℋ [2, 9/4) = 2.42·10⁻⁵` and the error there `1.08·10⁻⁵` to two digits |
| R1 | analytic Kerr-Schild through the core rule at `r_0 = 1/4`, the algebraic source, the guard only |
| R2 | smoothed data and `ΔH`, no steering |
| R3 | **the base point** |
| R4, R5 | `p = 2`; `μ_α = 1/M` |
| R6 | F, `κ = 1/3`, the data pre-collapsed |
| R7 | F, `κ = 1/3`, the plain data, the steering rolled on over `σ_g = 4 M` |
| R8 | the shift steering, `μ_β = 1/M`, `β_* = 0` |
| R9 | without `ΔH` (`Π` from `C_a = 0`, `∂_t α` up to `3.4/M`) |
| R10, R11 | `ε = 3/4`; `ε = 5/4` |
| R12, R13 | `ε_in = 2`; `γ0` near the hole `1/(10M)` |
| R14 | the anchor, `ρ_a = 1/(10M)` |
| R15 | `α_min = 10⁻⁶` |
| R16 | tight ranges at `2×` the data's extremes |
| R17 | the gate everywhere |

**A row passes** (`PLAN.md`) if it reaches `8 M`, its hits per chunk level off
with the outermost hit at least 8 cells inside the horizon and not moving out,
and its `ℋ` and error in `[2, 9/4)` are within `3×` R0's, flat or falling from
`4 M`; `test/layerfree_study.jl` prints that verdict beside the numbers. The
promotion rows (at most three passing rows at `N = 96` and `128`, `1/24` and
`1/32`, to `24 M` with `checkpoint=` and `walltime=`) and the best row at
`128 M` on `ks-octant`'s `[0, 128]³` mesh (`L=128 radii=64,32,16,8 N=32`
gives `h = 1/16` there) are the same lines; `series=` gives their orders.

**What the start does, measured before the screen** (step I1, on the laptop, on
`[0, 4]³` with Dirichlet data at `4 M`, chunks of `M/32`): the base point's
smoothed ball collapses before the steering acts. At `h = 1/16` the least
`det γ` in the ball is `1, 0.840, 0.460, 0.146, 0.0247, 0.0023` at
`t = 0, 1/32, …, 5/32 M` and the least lapse `0.347, 0.341, 0.302, 0.230,
0.142, 0.066`; at `3/16 M` the projection fires on six points, `C_a`'s L2 jumps
from `0.11` to `25`, the fastest speed from `1.69` to `204`, and the CFL recheck
ends the run. At `h = 1/8` the same ends at `1/8 M` (`det γ = 0.11` at
`0.094 M`). **Expect R3 and its neighbours to end in their first chunk at
`1/16`**; a row that does is the failure signature "the singularity re-forms
before the gauge collapses" rather than a leak, and the rows that change it —
F pre-collapsed (R6), the anchor (R14), the rate — are the screen's answer.
A layer-free run sizes its step as a moving hole's (`1 %` and the previous
chunk's growth squared), so a slow speed growth does not end it; a fast one
still does, through the recheck.

**The smoke** (job 586053, `amddebugq`, an EPYC 7532, CPU): each line above,
with `backend=cuda`, `L`, `roots`, `radii`, `N`, `t_end` and `chunk` replaced by
the small mesh `L=4 roots=2 radii=2 N=16` (`h = 1/16` on `[0, 2]³`, Dirichlet at
`4 M`) and one chunk of `1/16 M`, parsed, built its case and wrote its `t = 0`
row; every row but R1 finished the chunk (`6–8 min` a row with 18 at once on 64
cores, compilation included), its CSV's 147 columns matching its header, and
`test/layerfree_study.jl` read the directory. **R1 ended at the end of its first
chunk**: the analytic hole's core-rule data inside `r_core = 1/4` are not a
solution (`ℋ` L∞ `4.9·10⁵` at `t = 0`), and at the chunk's end the speed
kernel's `metric_quantities` took `√(det γ)` of `−1.65·10⁻⁷` at some point (a
`DomainError` on the CPU; a `NaN`, and the driver's refusal, on a device) — the
bare conjecture's failure signature on this mesh, not diagnosed further. **A
caution for every layer-free row:** `metric_quantities` forms `det γ` as
`det g · g^{tt}`, so its roundoff grows as `1/α²` — about `10⁻⁶` absolute at
the lapse floor `α = 10⁻³` with `|h| ~ 10` — and the guard's spectrum floor
`λ ≥ 10⁻³` does not protect a quantity computed that way; a row that collapses
its lapse to the floor may end the same way. That is the parse and the start,
not a result.

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
