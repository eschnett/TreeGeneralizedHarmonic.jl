# The two identities that say `src/pointwise.jl` is the right equation, not
# merely a self-consistent one.
#
# `pointwise_tests.jl` checks the algebra that is a function of the state,
# against `SpacetimeMetrics`. What it cannot check is whether the reduced
# source is the source of the *Einstein* equations: `S0` is a contraction of
# Christoffels that no second implementation of the geometry computes, so
# the only way to find a wrong sign in it is to evolve an exact solution for
# an instant and watch it not move. That is what the last two testsets do,
# at one point and with the derivatives taken exactly rather than by a
# time step:
#
#   1. **GHSO2's conservation identity** `∂_tΠ − ∂_iF^i = msrc` on analytic
#      data, with the divergence a central difference of the analytic flux
#      and `∂_tΠ` a central difference of the analytic momentum. This is the
#      statement that the flux form *is* the vacuum Einstein equation in this
#      gauge — if a term of `S0` were wrong, the residual would sit at the
#      size of that term instead of at the difference's truncation error.
#   2. **The expanded form equals the flux form.** `CODE.md` discretises
#      `(EXPANDED)`, which is `(FLUX)` with the product rule applied to the
#      divergence, so the two must agree to roundoff when the flux's
#      divergence is taken exactly. This is what validates
#      `metric_derivatives` in the place it is used, and it is the check
#      that the coefficient derivatives carry the signs the chain rule gives
#      them.
#
# The first testset below belongs to them rather than to the algebra: two
# expressions of `gh_node_rhs` are written down a second time so that the
# identities can be taken without compiling the reduced source under dual
# numbers, and a copy that drifted would make both of the checks above
# measure the copy instead of the port.
#
# Split out of `pointwise_tests.jl` because between them these evaluate the
# six backgrounds under nested dual numbers at two precisions, and GHSO2's
# rule of thumb is 30 s per test file.

using StaticArrays: SVector
using Test
using TreeGeneralizedHarmonic
using TreeGeneralizedHarmonic: _dg4, _sym4

isdefined(@__MODULE__, :gh_backgrounds) || include("pointwise_backgrounds.jl")

# A fourth-order central difference, built in `Rational` and rounded once
# into `T` — the discipline `CODE.md` prescribes for the real stencils,
# which arrive in step 2 and are not needed here. This one is host-side,
# one-dimensional and applied to a closure, which is all a consistency check
# of the continuum equation requires.
const CD4_NODES = (-2, -1, 1, 2)
const CD4_WEIGHTS = (1 // 12, -2 // 3, 2 // 3, -1 // 12)

function central_difference(f, ε::T) where {T}
    s = T(CD4_WEIGHTS[1]) * f(T(CD4_NODES[1]) * ε)
    for k in 2:4
        s = s + T(CD4_WEIGHTS[k]) * f(T(CD4_NODES[k]) * ε)
    end
    return s / ε
end

# The step at which the difference is taken. At `Float64` the fourth-order
# truncation error `~ε⁴` and the roundoff floor `~eps/ε` cross well below
# the residual this test would notice, so `1/64` and its half both sit in
# the truncation-dominated regime and the ratio between them is the order.
# At `Float32` there is no such window — `eps/ε` is already `10⁻⁶` at
# `ε = 1/8` — so the step is as large as the fourth-order error allows and
# the assertion is correspondingly loose. It still says what it is for: an
# algebra with a wrong term would miss by O(1), not by `10⁻³`.
fd_step(::Type{Float64}) = 1 / 64
fd_step(::Type{Float32}) = 1.0f0 / 8

# The residual the fourth-order difference leaves, with room. The largest
# measured is `3.8e−8` at `Float64` and `1.8e−3` at `Float32`, both on the
# shifted Minkowski row, whose `tanh(x/w)` profile has the largest fifth
# derivative in the table **(measured in step 1)**.
identity_tolerance(::Type{Float64}) = 1.0e-6
identity_tolerance(::Type{Float32}) = 1.0f-1

@testset "The source and the flux have one spelling each: T=$T" for T in (Float64,
                                                                        Float32)
    # Guards the two places where an expression from `gh_node_rhs` is
    # written down a second time, which is twice more than anyone would
    # like. `gh_node_source` is the ported function's source block lifted
    # out so that the expanded form does not carry a second transcription
    # of the reduced source; `gh_fluxes` in `pointwise_backgrounds.jl` is
    # its flux, written short so that automatic differentiation does not
    # have to compile the source under dual numbers. If either drifts from
    # the port, the identity tests would be measuring the copy instead of
    # the thing this package inherited, and would still pass.
    #
    # **To roundoff and not bit for bit.** The expressions are identical
    # character for character, and `isequal` still fails on two of the
    # twenty-four cases: the compiler is free to contract a multiply and an
    # add into a fused multiply-add in one inlining context and not in the
    # other, and on this machine it does. The disagreement is one unit in
    # the last place of `msrc` itself and under `0.4 eps` of the O(1) scale
    # below — a property of the code generator, not of the algebra
    # **(measured in step 1)**. A dropped or mis-signed term would be O(1)
    # relative and would still be caught.
    for (name, bg, _) in gh_backgrounds(T), x in gh_points(T, 2)
        h, Π, ∂h = gh_state(bg, T(GH_TIME), x)
        Hl, dHl = gh_gauge(bg, T(GH_TIME), x)
        for (γ0, γ2) in ((zero(T), zero(T)), (gh_γ0(T), gh_γ2(T)))
            dtg, Fx, Fy, Fz, msrc = gh_node_rhs(h, Π, ∂h[1], ∂h[2], ∂h[3],
                                                Hl, dHl, γ0, γ2)
            g4, gu4, α, _, _, sqrtγ = metric_quantities(_sym4(h))
            msrc2 = gh_node_source(g4, gu4, α, sqrtγ, _dg4(dtg, ∂h),
                                   Hl, dHl, γ0, γ2)
            @test absdiff(msrc, msrc2) <
                  64 * eps(T) * max(maximum(abs, msrc), one(T))

            F = gh_fluxes(bg, T(GH_TIME), x)
            for (n, Fn) in enumerate((Fx, Fy, Fz))
                @test absdiff(F[n], Fn) <
                      64 * eps(T) * max(maximum(abs, Fn), one(T))
            end
        end
    end
end

@testset "The flux identity ∂_tΠ − ∂_iF^i = msrc holds on exact data: T=$T" for T in
                                                                               (Float64,
                                                                                Float32)
    # Guards the reduced source, which is the only part of the formulation
    # that no other implementation of the geometry can be compared against
    # term by term. GHSO2 validated it this way and this package inherits
    # it verbatim; the identity is what says the port did not lose a
    # symmetrisation, a factor of two or the densitisation correction
    # `−Γ^ν ∂_ν g_ab`, each of which is O(1) off harmonic gauge and
    # invisible in Minkowski.
    #
    # `F^i` does not involve the gauge source or the damping, so it is
    # differenced with neither; `msrc` carries both.
    ε = T(fd_step(T))
    for (name, bg, _) in gh_backgrounds(T), x in gh_points(T, 2)
        t = T(GH_TIME)
        h, Π, ∂h = gh_state(bg, t, x)
        Hl, dHl = gh_gauge(bg, t, x)
        _, _, _, _, msrc = gh_node_rhs(h, Π, ∂h[1], ∂h[2], ∂h[3], Hl, dHl,
                                       gh_γ0(T), gh_γ2(T))

        residual(step) = begin
            ∂ₜΠ = central_difference(δ -> gh_state(bg, t + δ, x)[2], step)
            divF = central_difference(
                       δ -> gh_fluxes(bg, t, x + SVector{3,T}(δ, 0, 0))[1], step) +
                   central_difference(
                       δ -> gh_fluxes(bg, t, x + SVector{3,T}(0, δ, 0))[2], step) +
                   central_difference(
                       δ -> gh_fluxes(bg, t, x + SVector{3,T}(0, 0, δ))[3], step)
            return maximum(abs, ∂ₜΠ - divF - msrc), maximum(abs, divF)
        end

        r1, scale = residual(ε)
        r2, _ = residual(ε / 2)
        s = max(scale, maximum(abs, msrc), one(T))

        # The residual is the difference's truncation error and nothing
        # else, measured against the size of the terms that cancel. What
        # the tolerance has to separate is a residual at the difference's
        # error from one at the size of a missing term, and those are six
        # decades apart.
        @test r1 < identity_tolerance(T) * s

        # And it is *fourth* order in the step, wherever the step is still
        # above the roundoff floor. The gauge wave is the exception by
        # construction: it depends on `x − t` alone, so the temporal and the
        # spatial truncation errors cancel against each other and the
        # residual sits at roundoff at every step.
        floor = 1024 * eps(T) * s
        @test r2 ≤ max(r1 / 8, floor)
    end
end

@testset "The expanded form is the flux form: T=$T" for T in (Float64, Float32)
    # Guards `CODE.md`'s decision to discretise `(EXPANDED)` rather than
    # `(FLUX)`. The two are the same equation with the product rule applied
    # to `∂_i F^i`, so on analytic data — where the flux's divergence can be
    # taken exactly by automatic differentiation rather than by a stencil —
    # they must agree to roundoff. A sign lost in `metric_derivatives` or a
    # term dropped from the expansion would show here as an O(1)
    # disagreement, while the evolution it produced would still look like a
    # wave equation and still converge, to the wrong solution.
    #
    # The comparison also pins the two shapes of the second derivative: the
    # off-diagonal entries of `∂∂h` carry a factor of two in the contraction
    # because the packing is lower-triangular, and dropping it is a change
    # in the principal part.
    for (name, bg, _) in gh_backgrounds(T), x in gh_points(T, 2)
        t = T(GH_TIME)
        h, Π, ∂h = gh_state(bg, t, x)
        # ∂Π, ∂∂h and the *exact* ∂_iF^i, from one dual pass.
        ∂Π, ∂∂h, divF = gh_derivatives(bg, t, x)
        Hl, dHl = gh_gauge(bg, t, x)
        γ0, γ2 = gh_γ0(T), gh_γ2(T)

        ∂ₜh, ∂ₜΠ = gh_node_rhs_expanded(h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl, γ0, γ2)
        dtg, _, _, _, msrc = gh_node_rhs(h, Π, ∂h[1], ∂h[2], ∂h[3], Hl, dHl,
                                         γ0, γ2)

        # The size to measure the disagreement against is the size of the
        # two terms, not of their sum: on every static background in the
        # table the sum is zero — that is what "stationary" means — and
        # `∂_tΠ` and `∂_iF^i + msrc` are each O(1) numbers that cancel.
        scale = max(maximum(abs, divF), maximum(abs, msrc),
                    maximum(abs, ∂ₜΠ), one(T))
        @test absdiff(∂ₜΠ, divF + msrc) < 256 * eps(T) * scale

        # The first evolution equation is one expression in both — same
        # characters, and the same number to a fused multiply-add's worth
        # of rounding.
        @test absdiff(∂ₜh, dtg) <
              64 * eps(T) * max(maximum(abs, dtg), one(T))

        # And the same on the constraint surface with the damping off, so
        # that the agreement is not an artefact of `Z_ab` being tiny.
        ∂ₜh0, ∂ₜΠ0 = gh_node_rhs_expanded(h, Π, ∂h, ∂Π, ∂∂h, Hl, dHl,
                                          zero(T), zero(T))
        _, _, _, _, msrc0 = gh_node_rhs(h, Π, ∂h[1], ∂h[2], ∂h[3], Hl, dHl,
                                        zero(T), zero(T))
        @test absdiff(∂ₜh0, dtg) <
              64 * eps(T) * max(maximum(abs, dtg), one(T))
        @test absdiff(∂ₜΠ0, divF + msrc0) < 256 * eps(T) * scale
    end
end
