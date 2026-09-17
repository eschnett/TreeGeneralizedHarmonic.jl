# Inherited documents

Verbatim copies of the documents `CODE.md` builds on. Their originals live
in this author's unpublished repositories under `~/src/jl/`, which are
neither stable nor citable, so the text is kept here. Each file opens with
its source path, the source repository's commit, and whether that file had
uncommitted changes when it was copied (2026-09-16). Do not edit these
copies; when `CODE.md` departs from them, `CODE.md` says so.

| file | source | contents |
|---|---|---|
| `formulation.md` | `GeneralizedHarmonicSecondOrder/FORMULATION.md` | the second-order-in-space GH system, its symmetrizer and well-posedness, the characteristic structure |
| `methods-ghso1.md` | `GeneralizedHarmonicSecondOrder/METHODS.md` | the predecessor's numerical choices, and the first account of the sonic-surface instability with the cures that failed |
| `methods-ghso2.md` | `GeneralizedHarmonicSecondOrder2/METHODS.md` | the formulation as implemented, constraint damping, floating-point hygiene, boundaries, the resolved sonic-surface findings, the horizon analysis, the recipe `ε_KO ≈ 0.5`, `γ0 ≈ 1/M` |
| `sonic-surface.md` | `GeneralizedHarmonicSecondOrder2/scratch/sonic/NOTES.md` | the full sonic-surface investigation and verdict |
| `pointwise-ghso2.jl` | `GeneralizedHarmonicSecondOrder2/src/pointwise.jl` | the pointwise algebra `src/pointwise.jl` is ported from |
| `ghaccel-bench.jl` | `GeneralizedHarmonicAccel/bin/bench.jl` | the GPU roofline measurements (A40, H200) the RHS kernel is compared against |
