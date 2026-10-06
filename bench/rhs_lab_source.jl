# The register-lean spelling of `gh_node_source` that `bench/rhs_lab.jl` measures,
# and nothing that needs a device (added 2026-10-05; `CODE.md`, "The right-hand side
# on an H200"). Included by `bench/rhs_lab.jl` and `bench/rhs_lab_cpu.jl`, which
# define `T` (the element type), `NC` and `TGH` (the package) before including it.
#
# Two spellings of the same arithmetic:
#
# - `lean_source`, whose loops are `ntuple` do-blocks. It is fast only where every
#   closure is inlined: on a device with `always_inline`, and nowhere else.
# - `lean_source_inl`, whose loops `Base.Cartesian.@ntuple` expands at macro time
#   with literal indices. It needs no inliner.
#
# Both agree with `gh_node_source` to roundoff (bench/rhs_lab_cpu.jl).

# --- a register-lean source -------------------------------------------------------
#
# `gh_node_source`'s algebra with every symmetric tensor held by its unique
# components (`P(a, b)` is the packed slot) and the work ordered by phase, so that the
# live set is bounded: first `C2 + C2ᵀ`, one first index `a` at a time (`G D_a G`, ten
# values); then the forty Christoffel symbols and everything that reads them. Same
# terms as the package; summed in another order, so equal to roundoff.

const P = TGH._pairindex

@inline symget(v, a, b) = @inbounds v[P(a, b)]

@inline function lean_source(g4p, Gp, α, sqrtγ, D, Hl, dHl, γ0, γ2)
    # D[a] :: SVector{10}, ∂_a g packed; Gp, g4p :: SVector{10}, g^{ab} and g_ab packed.
    # Every loop is an `ntuple` over a `Val`, so that it is unrolled and every index
    # is a constant: a tuple indexed at run time would live in local memory.
    # (1) S = C2 + C2ᵀ, one first index a at a time.
    S = _sumtuple(ntuple(Val(4)) do a
        Da = D[a]
        E = ntuple(Val(16)) do n
            μ = (n - 1) % 4 + 1
            y = (n - 1) ÷ 4 + 1
            _sumtuple(ntuple(x -> symget(Gp, μ, x) * symget(Da, x, y), Val(4)))
        end
        Cuu = SVector{NC,T}(ntuple(Val(NC)) do n
            μ, ν = _unpair(n)
            _sumtuple(ntuple(y -> E[μ + 4 * (y - 1)] * symget(Gp, y, ν), Val(4)))
        end)
        C2a = ntuple(bb -> c2(Cuu, D, bb), Val(4))          # C2[a, b], b = 1:4
        SVector{NC,T}(ntuple(Val(NC)) do n
            c, d = _unpair(n)
            (c == a ? C2a[d] : zero(T)) + (d == a ? C2a[c] : zero(T))
        end)
    end)
    # (2) Γ^a_{bc} = Σ_x G[a, x] Γ_{x,bc}, Γ_{x,bc} = ½(D_b[x,c] + D_c[x,b] − D_x[b,c]).
    Γ = ntuple(Val(4)) do a
        SVector{NC,T}(ntuple(Val(NC)) do n
            b, c = _unpair(n)
            _sumtuple(ntuple(x -> symget(Gp, a, x) *
                                  ((symget(D[b], x, c) + symget(D[c], x, b) -
                                    symget(D[x], b, c)) / 2), Val(4)))
        end)
    end
    Γup = SVector{4,T}(ntuple(Val(4)) do c
        _sumtuple(ntuple(n -> symget(Gp, (n - 1) % 4 + 1, (n - 1) ÷ 4 + 1) *
                              symget(Γ[c], (n - 1) % 4 + 1, (n - 1) ÷ 4 + 1), Val(16)))
    end)
    S = S + SVector{NC,T}(ntuple(Val(NC)) do n
        a, b = _unpair(n)
        t2 = _sumtuple(ntuple(m -> symget(Γ[(m - 1) % 4 + 1], (m - 1) ÷ 4 + 1, a) *
                                   symget(Γ[(m - 1) ÷ 4 + 1], (m - 1) % 4 + 1, b), Val(16)))
        t4 = Γ[1][n] * Hl[1] + Γ[2][n] * Hl[2] + Γ[3][n] * Hl[3] + Γ[4][n] * Hl[4]
        t5 = Γup[1] * D[1][n] + Γup[2] * D[2][n] + Γup[3] * D[3][n] + Γup[4] * D[4][n]
        -2 * t2 - (dHl[a, b] + dHl[b, a]) + 2 * t4 - t5
    end)
    if γ0 != 0
        GH = SVector{4,T}(ntuple(c -> _sumtuple(ntuple(x -> symget(Gp, c, x) * Hl[x], Val(4))),
                                 Val(4)))
        Cup = Γup + GH
        Cl = SVector{4,T}(ntuple(c -> _sumtuple(ntuple(x -> symget(g4p, c, x) * Cup[x], Val(4))),
                                 Val(4)))
        tC = -α * _sumtuple(ntuple(x -> symget(Gp, 1, x) * Cl[x], Val(4)))
        tl1 = -α
        S = S + SVector{NC,T}(ntuple(Val(NC)) do n
            a, b = _unpair(n)
            γ0 * ((a == 1 ? tl1 * Cl[b] : zero(T)) + (b == 1 ? tl1 * Cl[a] : zero(T)) -
                  (1 + γ2) * g4p[n] * tC)
        end)
    end
    return -(α * sqrtγ) * S
end

@inline _sumtuple(t::Tuple) = TGH._fold(t)

# C2[a, b] = Σ_{μν} Cuu[μ, ν] D_μ[ν, b], for the a whose Cuu is given.
@inline c2(Cuu, D, b) =
    _sumtuple(ntuple(m -> symget(Cuu, (m - 1) % 4 + 1, (m - 1) ÷ 4 + 1) *
                          symget(D[(m - 1) % 4 + 1], (m - 1) ÷ 4 + 1, b), Val(16)))

# The (row, column) of packed slot n, row ≥ column: the inverse of `_pairindex`.
@inline function _unpair(n)
    n <= 4 && return (n, 1)
    n <= 7 && return (n - 3, 2)
    n <= 9 && return (n - 5, 3)
    return (4, 4)
end

# The lean source with no closures: every loop is expanded by `Base.Cartesian.@ntuple`
# at macro time, so each index is a literal and nothing depends on the inliner taking
# a closure — on the CPU or on a device. Same arithmetic, same order as `lean_source`.
using Base.Cartesian: @ntuple

@inline _sum4(t) = ((t[1] + t[2]) + t[3]) + t[4]
@inline _sum16(t) = TGH._fold(t)

@inline function lean_source_inl(g4p, Gp, α, sqrtγ, D, Hl, dHl, γ0, γ2)
    S = _cuu_sum(Gp, D)
    Γ = @ntuple 4 a -> _gamma_row(Gp, D, Val(a))
    Γup = SVector{4,T}(@ntuple 4 c -> TGH._fold(@ntuple 16 n -> symget(Gp, (n - 1) % 4 + 1,
                                                                       (n - 1) ÷ 4 + 1) *
                                                               symget(Γ[c], (n - 1) % 4 + 1,
                                                                      (n - 1) ÷ 4 + 1)))
    S = S + SVector{NC,T}(@ntuple 10 n -> _gamma_terms(Γ, Γup, D, Hl, dHl, Val(n)))
    if γ0 != 0
        GH = SVector{4,T}(@ntuple 4 c -> _sum4(@ntuple 4 x -> symget(Gp, c, x) * Hl[x]))
        Cup = Γup + GH
        Cl = SVector{4,T}(@ntuple 4 c -> _sum4(@ntuple 4 x -> symget(g4p, c, x) * Cup[x]))
        tC = -α * _sum4(@ntuple 4 x -> symget(Gp, 1, x) * Cl[x])
        tl1 = -α
        S = S + SVector{NC,T}(@ntuple 10 n -> _damping_term(Cl, tC, tl1, g4p, γ0, γ2, Val(n)))
    end
    return -(α * sqrtγ) * S
end

# S = C2 + C2ᵀ, one first index a at a time.
@inline function _cuu_sum(Gp, D)
    return _cuu_a(Gp, D, Val(1)) + _cuu_a(Gp, D, Val(2)) + _cuu_a(Gp, D, Val(3)) +
           _cuu_a(Gp, D, Val(4))
end
@inline function _cuu_a(Gp, D, ::Val{a}) where {a}
    Da = D[a]
    E = @ntuple 16 n -> _sum4(@ntuple 4 x -> symget(Gp, (n - 1) % 4 + 1, x) *
                                             symget(Da, x, (n - 1) ÷ 4 + 1))
    Cuu = SVector{NC,T}(@ntuple 10 n -> _sum4(@ntuple 4 y -> E[_unpair(n)[1] + 4 * (y - 1)] *
                                                             symget(Gp, y, _unpair(n)[2])))
    C2a = @ntuple 4 bb -> TGH._fold(@ntuple 16 m -> symget(Cuu, (m - 1) % 4 + 1, (m - 1) ÷ 4 + 1) *
                                                     symget(D[(m - 1) % 4 + 1], (m - 1) ÷ 4 + 1, bb))
    return SVector{NC,T}(@ntuple 10 n -> (_unpair(n)[1] == a ? C2a[_unpair(n)[2]] : zero(T)) +
                                         (_unpair(n)[2] == a ? C2a[_unpair(n)[1]] : zero(T)))
end
@inline function _gamma_row(Gp, D, ::Val{a}) where {a}
    return SVector{NC,T}(@ntuple 10 n -> _sum4(@ntuple 4 x ->
        symget(Gp, a, x) * ((symget(D[_unpair(n)[1]], x, _unpair(n)[2]) +
                             symget(D[_unpair(n)[2]], x, _unpair(n)[1]) -
                             symget(D[x], _unpair(n)[1], _unpair(n)[2])) / 2)))
end
@inline function _gamma_terms(Γ, Γup, D, Hl, dHl, ::Val{n}) where {n}
    a, b = _unpair(n)
    t2 = TGH._fold(@ntuple 16 m -> symget(Γ[(m - 1) % 4 + 1], (m - 1) ÷ 4 + 1, a) *
                                   symget(Γ[(m - 1) ÷ 4 + 1], (m - 1) % 4 + 1, b))
    t4 = Γ[1][n] * Hl[1] + Γ[2][n] * Hl[2] + Γ[3][n] * Hl[3] + Γ[4][n] * Hl[4]
    t5 = Γup[1] * D[1][n] + Γup[2] * D[2][n] + Γup[3] * D[3][n] + Γup[4] * D[4][n]
    return -2 * t2 - (dHl[a, b] + dHl[b, a]) + 2 * t4 - t5
end
@inline function _damping_term(Cl, tC, tl1, g4p, γ0, γ2, ::Val{n}) where {n}
    a, b = _unpair(n)
    return γ0 * ((a == 1 ? tl1 * Cl[b] : zero(T)) + (b == 1 ? tl1 * Cl[a] : zero(T)) -
                 (1 + γ2) * g4p[n] * tC)
end
