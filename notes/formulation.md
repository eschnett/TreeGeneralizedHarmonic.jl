> Copied verbatim from `/Users/eschnett/src/jl/GeneralizedHarmonicSecondOrder/FORMULATION.md` on 2026-09-16, repository at commit `37f4414`,
> clean at that commit. The original repository is unpublished; this copy is the
> citable reference for TreeGeneralizedHarmonic's `CODE.md`. Do not edit it here;
> amend `CODE.md` instead.

# Formulation and well-posedness

This note answers, *before any implementation* (Milestone A1), whether the
conservative first-order-in-time `(g_ab, Π_ab)` system this package evolves is a
genuine, well-posed formulation of the Einstein equations — or merely a scalar-wave
analogy. Short answer: it **is** the generalized-harmonic (wave-coordinate) Einstein
system, it is **symmetric hyperbolic / symmetrizable** for any lapse `α>0` and any
shift (including superluminal) on spacelike slices, and the chosen
second-order-in-space form is well-posed **without** auxiliary first-derivative
variables. The per-component "scalar wave" structure is exact, not analogy: that is
the content of "generalized *harmonic*."

Notation: spacetime metric `g_ab` (indices `a,b,… = t,x,y,z`); ADM split
`ds² = −α²dt² + γ_ij(dx^i+β^idt)(dx^j+β^jdt)`; future unit normal
`n^μ = (1/α)(1,−β^i)`, `n_μ=(−α,0,0,0)`; `√|g| = α√γ`.

---

## 1. It is the Einstein equations (reduced generalized harmonic), not an analogy

Define the gauge source functions `H^a` (here `H^a ≡ 0`, pure harmonic) and the
**harmonic constraint**

    C^a := H^a + Γ^a = 0,    Γ^a := g^{bc} Γ^a_{bc}   (contracted Christoffel).

The Ricci tensor obeys the identity

    R_ab = −½ g^{cd}∂_c∂_d g_ab + ∇_{(a} Γ_{b)} + (terms quadratic in ∂g),

so the **reduced** equation, obtained by substituting the constraint
`Γ_a = −H_a` (and here `H_a=0`) and setting the physical `R_ab = 0`, is the
quasilinear system of wave equations

    g^{cd} ∂_c ∂_d g_ab = 2 Q_ab(g, ∂g),                              (★)

with `Q_ab` algebraic and quadratic in first derivatives. Written out
(Garfinkle 2002, eq. 9; the form `src/gh.jl::make_rhs` implements):

    −g^{tt}∂_tt g_ab = 2g^{tk}∂_t∂_k g_ab + g^{ik}∂_i∂_k g_ab
                       − C_a^{μν}C_{μνb} − C_b^{μν}C_{μνa} + 2Γ^γ_{να}Γ^ν_{γβ},
    C_aμν := ∂_a g_μν.

Collecting the principal terms, `(★)` reads `g^{cd}∂_c∂_d g_ab = C2sym_ab − 2Γ2_ab`
with `C2sym_ab = C_a^{μν}C_{μνb}+C_b^{μν}C_{μνa}` and `Γ2_ab = Γ^γ_{να}Γ^ν_{γβ}` —
exactly the algebraic quantities the symbolic generator already builds. Hence the
**reduced source** `S0_ab := C2sym_ab − 2Γ2_ab = 2Q_ab` is genuine Einstein content,
not invented.

This is **Choquet-Bruhat's wave-coordinate form of the vacuum Einstein equations**
(Fourès-Bruhat 1952), the basis of the classical local-existence theorem: in
harmonic/generalized-harmonic coordinates the principal part of `R_ab=0` *is* the
curved scalar-wave operator `g^{cd}∂_c∂_d` acting component-wise on `g_ab`. The 10
metric components form a coupled system, but the coupling is only through (i) the
shared principal symbol `g^{cd}(g)` and (ii) the lower-order source `Q(g,∂g)`. Treating
each component with the same scalar-wave machinery is therefore **exact**.

**Relation to full GR (constraint propagation).** A solution of the reduced system
`(★)` solves the full vacuum equations `R_ab=0` iff it also satisfies `C^a=0`. The
contracted Bianchi identity forces `C^a` to obey a homogeneous wave equation
`g^{cd}∂_c∂_d C^a = (lower order)·C`, so `C^a=0` and `∂_t C^a=0` initially ⇒ `C^a≡0`
(continuum constraint propagation). Discretely `C^a` drifts at truncation order; this
is what the kept diagnostics `gh_gauge_constraint!` (`C^μ=g^{αβ}Γ^μ_{αβ}`) and
`gh_physical_constraints!` (Einstein/Hamiltonian–momentum) monitor, and what
constraint damping (Milestone R5) controls.

---

## 2. First-order-in-time conservative reduction and its symmetrizer

The momentum is the **densitised Lie derivative of `g_ab` along the normal**,

    Π_ab := (√γ/α)(∂_t − ℒ_β) g_ab = √γ · n^μ ∂_μ g_ab,                (def. Π)

exactly as the scalar testbed `WaveToySecondOrder` carries `Π = √γ·n^μ∂_μΦ`
(`WaveToySecondOrder/METHODS.md`). The conservative flux form realizes the **densitised**
operator `□_dens := (1/√|g|)∂_μ(√|g|g^{μν}∂_ν·) = g^{cd}∂_c∂_d − Γ^ν∂_ν` (since
`(1/√|g|)∂_μ(√|g|g^{μν}) = −Γ^ν`, the contracted Christoffel `Γ^ν = g^{cd}Γ^ν_{cd}`). The
reduced equation `(★)` uses the **bare** operator `g^{cd}∂_c∂_d`, so they differ by
`Γ^ν∂_ν g_ab`. Rewriting `(★)` as `□_dens g_ab = 2Q_ab − Γ^ν∂_ν g_ab` and recasting per
component:

    ∂_t g_ab = β^i ∂_i g_ab + (α/√γ) Π_ab
    ∂_t Π_ab = ∂_i( β^i Π_ab + α√γ γ^{ij} ∂_j g_ab ) − 𝒮_ab,
    𝒮_ab = α√γ ( C2sym_ab − 2Γ2_ab − Γ^ν ∂_ν g_ab ).                   (CONS)

Two facts about `(CONS)` are part of the result, not free:

**(i) Minus sign on the source.** Multiplying `□_dens g = 2Q − Γ^ν∂g` by `√|g|`, the time
term is `∂_t(√|g| g^{tν}∂_ν g)`. The densitised momentum is `Π = √γ·n^μ∂_μ g =
(√γ/α)(∂_t−β^i∂_i)g`, and `√|g| g^{tν}∂_ν g = −Π` (with `g^{tt}=−1/α²`, `g^{ti}=β^i/α²`).
So `∂_t(−Π) + ∂_i Fⁱ = α√γ·(2Q − Γ^ν∂g)`, i.e. `∂_t Π = ∂_i Fⁱ − α√γ·(C2sym−2Γ2−Γ^ν∂g)`.

**(ii) The `−Γ^ν∂_ν g` densitisation term.** It corrects `□_dens` (what the flux gives)
back to the bare `g^{cd}∂_c∂_d` of `(★)`. It **vanishes in harmonic gauge** (`Γ^ν=0`), so
flat / harmonic data are untouched; for a gauge-sourced (non-harmonic) solution `Γ^ν=−H^ν≠0`
on the constraint surface, and it is an **O(1)** term — NOT `O(C^a)` — without which the
RHS does not reproduce the solution. (Both `(i)` and the `−Γ^ν∂g` term were absent in an
earlier version; the `(i)` minus was caught by the harmonic-Schwarzschild RHS-convergence
gate `test_gh_rhs_curved.jl`, the `−Γ^ν∂g` term by the gauge-sourced MovingGrid gate.)

**Symmetrizer / energy.** For each component `ψ = g_ab` the quadratic form

    E[ψ] = ∫_Σ ½ [ Π²/√γ + √γ γ^{ij} ∂_i ψ ∂_j ψ ] d³x                 (sym.)

is **positive definite for `α>0` and `γ_ij` positive definite** — i.e. precisely when
the slice `Σ` is spacelike. `E` is the symmetrizer of the principal symbol of
`(CONS)`: differentiating along `(CONS)` with `𝒮=0` gives `dE/dt =` a pure boundary
flux (the `∂_i(·)` term integrates against the skew pairing of `∂_i` with the `√γ`
measure), so `dE/dt = 0` on a closed/periodic domain. With the lower-order source and
boundary SATs, `dE/dt ≤ C·E`, the standard a-priori energy estimate ⇒ well-posedness.

**Crucially, `E` does not depend on the sign or magnitude of `β`.** The shift enters
`(CONS)` only as the advective transport `β^i∂_i` of the field (lower order w.r.t. the
energy norm; its contribution to `dE/dt` is controlled by `∂_iβ`). A **superluminal**
shift (`|β|>α`) still has spacelike slices (`α>0`), so `E` is still positive definite
and the estimate still closes — it merely tilts the numerical domain of dependence
(possibly fully one-sided, which is exactly the excision/full-Dirichlet boundary
situation). This is why the scalar testbed is stable at constant `β=2`, and is the
direct cure for the `∂_t g_ab` instability (`max Re λ ∝ β`) diagnosed in the old
second-order-in-time scheme: there the shift appeared as an *un-absorbed* advection of
the velocity `2g^{tk}∂_k(∂_t g)`, which has no such symmetrizer.

**Full 10-component system.** The principal symbol of `(CONS)` is block-diagonal: every
component carries the *same* scalar symbol `g^{cd}ξ_cξ_d`, and the components couple
only through the lower-order `𝒮`. The symmetrizer is the sum `Σ_{a≤b} E[g_ab]`, so the
block system is symmetric hyperbolic with this block-diagonal symmetrizer. **Symmetric
hyperbolic / symmetrizable: yes**, for any `α>0` and any shift on spacelike slices.

This is consistent with — and a second-order-in-space specialization of — the
first-order symmetric-hyperbolic generalized-harmonic system of Friedrich (1985) and
Lindblom–Scheel–Kidder–Owen–Rinne (2006).

---

## 3. Second-order-in-space vs fully-first-order — the go/no-go (PASS)

The chosen formulation keeps only `(g_ab, Π_ab)` and computes spatial derivatives
`∂_i g_ab` by the SBP operator each step; it does **not** evolve auxiliary
`Φ_iab := ∂_i g_ab` as independent fields. Is *this* form well-posed?

For second-order-in-space systems the relevant criterion (Kreiss–Ortiz 2002;
Nagy–Ortiz–Reula 2004; Gundlach–Martín-García 2004, 2006) is that the associated
first-order pencil be diagonalizable with real characteristic speeds and a bounded
symmetrizer — equivalently that the energy `(sym.)` closes when the spatial gradient
is treated as a derived quantity. For `(CONS)` the principal symbol per component is
the **scalar wave symbol** `g^{cd}ξ_cξ_d` (positive-definite spatial part
`α√γ γ^{ij}` in `E`), which is the textbook well-posed case. The consistency condition
`∂_t(∂_i ψ) = ∂_i(∂_t ψ)` is maintained automatically — in the continuum trivially, and
discretely because the same SBP derivative produces both `∂_i ψ` and the divergence,
so the discrete energy estimate mirrors `(sym.)` (this is exactly the
skew-adjoint `H·D` SBP-SAT structure validated in `WaveToySecondOrder`).

**Conclusion: PASS.** The second-order-in-space `(g, Π)` form is well-posed and
symmetric hyperbolic; the auxiliary `Φ_iab` are **not required**. This matches the
classical Choquet-Bruhat result, which is genuinely second order (no first-order
reduction). The fully-first-order Lindblom et al. (2006) system (which *does* carry
`Φ_iab`) is an alternative chosen for cleaner characteristic boundary conditions and
constraint control, **not** a well-posedness necessity. → No pivot; proceed to R0.

**Conservative vs reduced operator.** The reduced (bare) operator `g^{cd}∂_c∂_d` and the
conservative (densitised) operator `(1/√|g|)∂_μ(√|g|g^{μν}∂_ν·)` differ by the contracted
Christoffel term `Γ^ν∂_ν g_ab`, `Γ^ν = g^{cd}Γ^ν_{cd}`. This is included exactly in `(CONS)`
via the `−Γ^ν∂_ν g` source term (§2), so the discrete scheme reproduces the bare reduced
equation `(★)` for **any** gauge, on or off the constraint surface. (Earlier this difference
was treated as `−√|g|·C^a∂_a g_ab = O(C^a)` and dropped as constraint-order; but
`Γ^ν = C^ν − H^ν`, so for a gauge source `H^a≠0` the `−H^ν∂_ν g` part is `O(1)` and does
NOT vanish on the constraint surface — dropping it left the non-harmonic RHS inconsistent.)

---

## 4. Characteristic structure (matches the boundary classifier)

The principal symbol is `g^{cd}ξ_cξ_d`. Along a spatial unit normal `n_i`
(`γ^{ij}n_in_j=1`) with `ξ=(−s,n_i)`, the characteristic condition
`g^{tt}s² − 2g^{ti}n_i s + g^{ij}n_in_j = 0` (using `g^{tt}=−1/α²`,
`g^{ti}=β^i/α²`, `g^{ij}=γ^{ij}−β^iβ^j/α²`) gives the **physical (light-cone)** speeds

    c± = −b_n ± α,        b_n := n_i β^i,

and the field/gauge variables (the `g_ab` values, transported with the coordinate
observer) propagate at the **gauge speed**

    c⁰ = −b_n      ( → −(1+γ₁) b_n with Gundlach–Pretorius constraint damping γ₁ ).

These are exactly the speeds the boundary classifier `classify_face_gh`
(`src/boundaries_gh.jl`, Milestone M1) uses: subluminal `|b_n|<α` (one physical mode
in, one out), superluminal outflow `b_n<−α` (both out → excision), superluminal inflow
`b_n>α` (both in → full-state Dirichlet), with the near-zero-speed gauge family pinned
to data. The boundary treatment is thus grounded in the proven hyperbolic structure,
not heuristics.

**Sonic surface.** Where a physical speed vanishes, `c₋ = −b_n − α = 0` (the
sub↔superluminal transition; the classifier flags it `FACE_SONIC`), the discrete GH
evolution develops a growing **gauge-constraint** mode (`C^a≠0`, with `ℋ`, `ℳ` clean)
sitting on that surface. It is genuine and GH-specific: the scalar conservative wave
(`WaveToySecondOrder`) and the linear Lorenz-gauge Maxwell analog
(`ElectrodynamicsSecondOrder`) on the identical superluminal-shift background are stable,
because neither carries an *off-light-cone* gauge family (EM's constraint rides `c±`); GH's
gauge family at `c⁰=−b_n` does, and the reduced source `2Q` pumps it where the physical
mode stalls (`c₋→0`). This is what makes sub-horizon excision unstable while
outside-horizon excision is fine, and it is not curable by a boundary SAT or ordinary
constraint damping — see `METHODS.md`, "Sub-horizon excision: the sonic-surface
gauge-constraint instability".

---

## Summary

1. **Einstein equations:** yes — the reduced generalized-harmonic / wave-coordinate
   system `(★)`, `g^{cd}∂_c∂_d g_ab = 2Q_ab`, with `2Q = C2sym − 2Γ2` (genuine
   Einstein source). Per-component scalar-wave structure is exact (Choquet-Bruhat;
   Garfinkle 2002; Friedrich 1985).
2. **Well-posed / symmetric hyperbolic / symmetrizable:** yes — symmetrizer = the
   per-component wave energy `(sym.)`, positive definite for `α>0` on spacelike slices,
   for **any** shift including superluminal. The conservative momentum
   `Π=√γ n^μ∂_μ g` is what makes this hold; `∂_t g` does not.
3. **Second-order-in-space (no `Φ_iab`):** well-posed (PASS) — proceed; fully-first-order
   is an option, not a requirement.
4. **Characteristics:** `c±=−b_n±α`, `c⁰=−b_n` (`−(1+γ₁)b_n` damped), matching the M1
   boundary classifier.

## References

- Y. Fourès-Bruhat (Choquet-Bruhat), *Théorème d'existence pour certains systèmes
  d'équations aux dérivées partielles non linéaires*, Acta Math. **88**, 141 (1952) —
  local existence for vacuum Einstein in harmonic coordinates.
- H. Friedrich, *On the hyperbolicity of Einstein's and other gauge field equations*,
  Commun. Math. Phys. **100**, 525 (1985) — generalized harmonic gauge, symmetric
  hyperbolicity.
- D. Garfinkle, *Harmonic coordinate method for simulating generic singularities*,
  Phys. Rev. D **65**, 044029 (2002), gr-qc/0110013 — the form implemented in `gh.jl`.
- F. Pretorius, *Evolution of binary black hole spacetimes*, Phys. Rev. Lett. **95**,
  121101 (2005); *Numerical relativity using a generalized harmonic decomposition*,
  Class. Quantum Grav. **22**, 425 (2005), gr-qc/0407110.
- C. Gundlach, J. M. Martín-García, G. Calabrese, I. Hinder, *Constraint damping in
  the Z4 formulation and harmonic gauge*, Class. Quantum Grav. **22**, 3767 (2005),
  gr-qc/0504114.
- L. Lindblom, M. A. Scheel, L. E. Kidder, R. Owen, O. Rinne, *A new generalized
  harmonic evolution system*, Class. Quantum Grav. **23**, S447 (2006), gr-qc/0512093 —
  first-order, linearly degenerate, symmetric-hyperbolic GH; constraint damping;
  constraint-preserving BCs.
- H.-O. Kreiss, O. E. Ortiz, *Some mathematical and numerical questions connected with
  first and second order time-dependent systems of partial differential equations*,
  Lect. Notes Phys. **604**, 359 (2002).
- G. Nagy, O. E. Ortiz, O. A. Reula, *Strongly hyperbolic second order Einstein's
  evolution equations*, Phys. Rev. D **70**, 044012 (2004), gr-qc/0402123.
- C. Gundlach, J. M. Martín-García, *Hyperbolicity of second-order in space systems of
  evolution equations*, Class. Quantum Grav. **23**, S387 (2006), gr-qc/0506037.
