# Element-type genericity: the scheme must run in the type the caller
# names, with no `Float64` left in the arithmetic.
#
# `CODE.md`, "Precision, threads, devices", and `PLAN.md`'s ground rule
# that steps 4 and 9 add the *tests* of this property and not the property
# — TreeWave records that retrofitting it was a rewrite. The failure mode
# is code that is generic in name only: computing in `Float64` and
# converting at the end, which is invisible in a `Float64` run and fatal
# on a device with no hardware fp64.
#
# Each non-default type catches a different fault, and TreeWave's reading
# of them carries over:
#
#   `Float32`    is the **leak detector**. A stray `Float64` operand widens
#                the result, so a returned `Float64` names the leak. It is
#                also `CODE.md`'s nice-to-have precision — a hole at
#                `Float32` is a real test of the offset identities — and a
#                failure here would be *recorded*, not fixed at `Float64`'s
#                expense.
#   `Float32x2`  is the **off the beaten path** detector: a software float
#                built from two `Float32` limbs, which no `Float64` fast
#                path can serve. It cannot detect leaks — MultiFloats
#                promotes `Float64` downward — but it says that nothing
#                depends on a hardware float at all. It is also what found
#                step 2's `@generated` trap, which is why `runtests.jl`
#                loads this file *after* the package rather than the other
#                way round.
#
# Convergence **orders** are asserted at `Float64` everywhere else in the
# suite and only at the coarsest resolutions here, for the reason
# `CODE.md` gives under "the window": the roundoff floor of a second
# derivative is `eps/h²`, which at `Float32` and `h = 1/24` is `2e−6` —
# the size of the truncation error the `Float64` study measures there. A
# `Float32` sweep over three resolutions would be measuring `eps/h²`, and
# moving the meshes until it did not would be fitting the test to the
# answer.
#
# No background is evaluated at `Float32x2`: `SpacetimeMetrics`' metrics
# are built on `sin`, `tanh` and `sqrt` of expressions MultiFloats does not
# implement transcendentally, and this package does not call
# `MultiFloats.use_bigfloat_transcendentals()` any more than TreeAMR does.
# What is run at that type is the pointwise algebra, on a state written
# down as rationals — which is the part that would break on a type without
# a hardware float underneath it.

using MultiFloats: Float32x2
using StaticArrays: SArray, SMatrix, SVector
using TreeGeneralizedHarmonic: NC, _dg4, _pairindex, _sym4, tofloat64

const TYPE_FLOATS = (Float64, Float32, Float32x2)

# A state written down as rationals, so that every type gets the same
# numbers rounded once — the discipline `CODE.md` asks for in every `T`
# expression, applied to the test's own data. The offsets are a few
# hundredths, which is a metric well inside the range where `α` and `√γ`
# are real and the algebra is not near a coordinate singularity.
type_state(::Type{T}) where {T} =
    SVector{NC,T}(T(3 // 100), T(1 // 50), T(-1 // 100), T(1 // 25),
                  T(7 // 100), T(-3 // 200), T(1 // 40), T(9 // 100),
                  T(-1 // 50), T(11 // 100))

type_momentum(::Type{T}) where {T} =
    SVector{NC,T}(T(1 // 20), T(-1 // 30), T(1 // 60), T(-1 // 25),
                  T(3 // 100), T(1 // 80), T(-1 // 100), T(2 // 50),
                  T(1 // 45), T(-3 // 100))

# Gradients that are not all alike, so that a transposed index or a
# dropped direction cannot hide.
type_gradients(::Type{T}) where {T} =
    ntuple(i -> SVector{NC,T}(ntuple(v -> T((-1)^(v + i) * (v + 2i) // 500),
                                     Val(NC))), Val(3))

type_second(::Type{T}) where {T} =
    ntuple(n -> SVector{NC,T}(ntuple(v -> T((-1)^(v * n) * (v + n) // 800),
                                     Val(NC))), Val(6))

@testset "The pointwise algebra computes in the type it is given: T=$T" for T in
                                                                           TYPE_FLOATS
    # Guards every function `constraints.jl` and the right-hand side call
    # per point. At `Float32` a returned `Float64` names a leak; at
    # `Float32x2` a `MethodError` would have arrived long before this line,
    # which is the whole point of having a software float in the suite.
    h = type_state(T)
    Π = type_momentum(T)
    ∂h = type_gradients(T)
    ∂Π = map(v -> v / 2, type_gradients(T))
    ∂∂h = type_second(T)
    Hl = zero(SVector{4,T})
    dHl = zero(SMatrix{4,4,T})
    γ0, γ2 = one(T), T(-1 // 2)

    g4, gu4, α, β, γu, sqrtγ = metric_quantities(_sym4(h))
    @test α isa T && sqrtγ isa T
    @test α > 0 && sqrtγ > 0
    @test eltype(β) === T && eltype(γu) === T

    dα, dβ, dA = metric_derivatives(gu4, α, β, γu, sqrtγ, ∂h)
    @test eltype(dα) === T && eltype(dβ) === T && eltype(dA) === T
    for i in 1:3
        dαi, dβi, dsqrtγi, dγui = metric_derivatives_along(gu4, α, β, γu,
                                                           sqrtγ, ∂h[i])
        @test dαi isa T && dsqrtγi isa T
        @test eltype(dβi) === T && eltype(dγui) === T
        @test isapprox(dαi, dα[i]; rtol=64 * eps(T), atol=64 * eps(T))
    end

    ∂ₜh, ∂ₜΠ = gh_node_rhs_expanded(h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl, γ0, γ2)
    @test eltype(∂ₜh) === T && eltype(∂ₜΠ) === T
    @test all(isfinite, ∂ₜh) && all(isfinite, ∂ₜΠ)

    src = gh_node_source(g4, gu4, α, sqrtγ, _dg4(∂ₜh, ∂h), Hl, dHl, γ0, γ2)
    @test eltype(src) === T

    # The step-4 additions: the four-dimensional curvature assembly, on the
    # second derivatives packed by the symmetric pair.
    dd = (∂ₜΠ / 100, ∂h[1] / 50, ∂h[2] / 50, ∂h[3] / 50, ∂∂h[1], ∂∂h[2],
          ∂∂h[3], ∂∂h[4], ∂∂h[5], ∂∂h[6])
    ddg = SArray{Tuple{4,4,4,4},T}(_sym4(dd[_pairindex(μ, ν)])[a, b]
                                   for μ in 1:4, ν in 1:4, a in 1:4, b in 1:4)
    ℋ, ℳ = adm_constraints_at_node(g4, gu4, α, β, _dg4(∂ₜh, ∂h), ddg)
    @test ℋ isa T
    @test eltype(ℳ) === T
    @test isfinite(ℋ) && all(isfinite, ℳ)

    Cup = gauge_constraint_at_node(g4, _dg4(∂ₜh, ∂h))
    @test eltype(Cup) === T
    γ3, ∂γ3, K = adm_vars_from_state(h, Π, ∂h[1], ∂h[2], ∂h[3])
    @test eltype(γ3) === T && eltype(∂γ3) === T && eltype(K) === T
end

@testset "The algebra agrees across types on the same rational data" begin
    # The claim `Float32x2` is in the suite for: the answer is the *same*
    # answer, not merely one of the right type. The inputs are rationals,
    # so each type sees the same numbers rounded once, and the comparison
    # is against the `Float64` result at the precision the narrower type
    # can carry. A `Float64` fast path hiding inside a `Float32` run would
    # pass this; a term that only exists at one precision would not.
    ref = nothing
    for T in TYPE_FLOATS
        h = type_state(T)
        Π = type_momentum(T)
        ∂h = type_gradients(T)
        ∂Π = map(v -> v / 2, type_gradients(T))
        ∂∂h = type_second(T)
        ∂ₜh, ∂ₜΠ = gh_node_rhs_expanded(h, Π, ∂h, ∂Π, ∂∂h, zero(SVector{4,T}),
                                        zero(SMatrix{4,4,T}), one(T),
                                        T(-1 // 2))
        # `tofloat64` and not `Float64`: MultiFloats defines a conversion
        # only to its own limb type, so `Float64(::Float32x2)` is a
        # `MethodError` — which is exactly the gap `precision.jl` exists to
        # bridge, seen from a test rather than from a driver.
        values = (tofloat64.(Tuple(∂ₜh))..., tofloat64.(Tuple(∂ₜΠ))...)
        if ref === nothing
            ref = values
        else
            # `Float32x2` carries 48 bits of significand, `Float32` 24; the
            # tolerance is the type's own, and the scale is the size of the
            # terms rather than of the result.
            scale = maximum(abs, ref)
            for (a, b) in zip(values, ref)
                @test abs(a - b) < 512 * tofloat64(eps(T)) * scale
            end
        end
    end
end

@testset "A run computes in the type it is given: T=$T" for T in (Float64, Float32)
    # Guards the whole of step 3 and step 4 at reduced precision: the
    # forest, the field set, the schedule, the state vector, the fused
    # kernel, the time step, the integrator and both monitors. Every number
    # that comes back has to carry the caller's type — a `Float64` here
    # names a promotion inside the run, which on a device with no hardware
    # fp64 is not a precision question but a refusal to compile.
    q = 4
    case = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=T(1 // 2),
                           γ0=one(T), γ2=T(-1 // 2))
    r = gh_errors(T, case; N=8, roots=2, q=q, t_end=T(1 // 8))
    @test r.l2 isa T
    @test r.linf isa T
    @test r.h isa T
    @test isfinite(r.l2) && r.l2 > 0

    c = gh_constraint_run(T, case; N=8, roots=2, q=q, t=T(1 // 8))
    @test c.gauge_l2 isa T
    @test c.ham_l2 isa T
    @test c.mom_linf isa T
    @test isfinite(c.gauge_l2) && isfinite(c.ham_l2)

    # And on the two-level mesh, where the ghosts come from an
    # interpolation whose weights are built in `Rational` and rounded into
    # `T` once.
    ct = gh_constraint_run(T, case; N=8, roots=2, q=q, t=T(1 // 8),
                           refined=true)
    @test ct.nblocks == 15
    @test ct.gauge_l2 isa T
    @test ct.gauge_l2 > c.gauge_l2         # the interface is the violation
end

@testset "Float32 reproduces the gauge wave's rate at coarse resolution" begin
    # `CODE.md`'s nice-to-have, measured. Two resolutions and no more: the
    # roundoff floor of the second derivative is `eps/h²`, which at
    # `Float32` and `h = 1/24` is already the size of the `Float64`
    # truncation error there (step 2's warning about the window), so a
    # third point would measure the arithmetic rather than the scheme. At
    # `h = 1/8` and `1/16` the truncation error is still a factor of four
    # above the floor, and the rate is the scheme's.
    q = 4
    rows = map((Float64, Float32)) do T
        case = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=zero(T),
                               γ0=one(T), γ2=zero(T))
        results = map((1, 2)) do roots
            gh_errors(T, case; N=8, roots=roots, q=q, t_end=T(1 // 8))
        end
        hs = [Float64(r.h) for r in results]
        l2 = [Float64(r.l2) for r in results]
        (hs=hs, l2=l2, rate=convergence_rate(hs, l2))
    end
    @info "gauge wave rate: Float64 $(rows[1].l2) rate=$(rows[1].rate); " *
          "Float32 $(rows[2].l2) rate=$(rows[2].rate)"
    @test rows[2].rate ≥ 3.5
    # And it is the *same* error, not merely one that falls at the same
    # rate: the arithmetic has not started to dominate at these spacings.
    for (a, b) in zip(rows[2].l2, rows[1].l2)
        @test a ≈ b rtol = 0.05
    end
end
