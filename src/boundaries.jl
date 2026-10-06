# The outer boundary: Dirichlet data from the background, at the current
# time.
#
# `CODE.md`, "Boundaries": this is the only outer boundary the package
# has. It is exact for every background here, static or moving — the
# analytic solution is known everywhere at every time, so the boundary
# data *is* the solution and a wave leaving through the boundary is
# absorbed to truncation order. A radiative condition, for a solution that
# is not known at the boundary, needs a hook that reads the block's
# interior, which TreeAMR's device form cannot do; it is under "Possible
# extensions".
#
# The whole file is one function, and the thing to get right about it is
# *when* it is built. The hook depends on time, so it is built at each
# call with that call's `t` — inside the right-hand side, at every ghost
# fill; at `regrid!`; and at `adapt_to_initial_data!` (`CLAUDE.md`, "Hooks
# depend on time": forgetting the second is the bug that arrives one chunk
# late, and passing a stale `t` is the bug that arrives as a boundary
# reflection).

"""
    dirichlet(case::GHCase, t) -> CellBoundary or nothing

The physical-boundary hook for `case` at time `t`: the analytic state of
the background at each outward-facing ghost point, or `nothing` when
every face of the case is periodic or reflecting and there is no physical
boundary at all. A reflecting face is never the hook's: TreeAMR's schedule
mirrors it (M10).

    boundary = CellBoundary(AllVariables(x -> state(background, t, x)))

exactly as `CODE.md` writes it. It is a `CellBoundary`, so it runs as a
kernel on every backend and cannot read the block's interior; a condition
defined by position alone is precisely this shape. It closes over the
background and `t`, both `isbits`.

For a vertex-centered field set the domain's upper boundary plane belongs
to no block's owned range and the hook fills it; the lower boundary points
are owned and evolved, with the ghosts below them filled from here.

Returning `nothing` for a fully periodic case is what lets the
right-hand side branch once, on a `Bool` it stores, instead of handing
`fill_ghosts!` an argument whose type changes with the case.
"""
function dirichlet(case::GHCase{T}, t) where {T}
    has_outer_face(case) || return nothing
    bg = case.background
    int = boundary_interior(case.interior)
    tt = T(t)
    return CellBoundary(AllVariables((x, δ) -> case_state_tuple(bg, int, tt, x)))
end

# The interior whose core rule the hook applies: the case's own, where it is
# the identity (the core is nowhere near the boundary) — and `nothing` for a
# tracked case, whose `FittedSpec` is a rule and not a geometry, and whose
# core is just as far from the boundary (added in step 8d). The identity is
# then written as the identity rather than computed.
boundary_interior(int) = int
boundary_interior(::FittedSpec) = nothing

"""
    has_outer_face(case::GHCase) -> Bool

Whether any face of the case's box is an outer face — neither periodic nor
reflecting — and so needs [`dirichlet`](@ref)'s hook (added 2026-10-02,
when reflecting faces made "not every dimension periodic" a weaker
statement than "there is a physical boundary").
"""
has_outer_face(case::GHCase) =
    any(d -> !case.periodic[d] && !all(case.reflecting[d]), 1:3)

# --- reflecting faces (added 2026-10-02) ---------------------------------------
#
# TreeAMR's M10 mirrors the ghosts across a reflecting face and multiplies
# each variable by its parity, which it requires on every field set over a
# forest with such a face: whether a variable changes sign in a mirror is
# physics. The state's parities follow from its packing — `h_ab` and `Π_ab`
# are tensors, so component `ab` is odd across the plane `x^d = 0` when the
# index `d` occurs an odd number of times in `ab` — and nothing else here
# needs a parity of its own: every other field set of the package has
# `G = 0`, is never ghost-filled and never transferred by a regrid, so it
# is declared even, which is what TreeAMR asks for and nothing reads.
#
# Over a forest without a reflecting face both helpers return `nothing`, so
# that every field set of a run without one — its checkpoints included — is
# exactly what it was before.

_reflects(forest) = any(r -> r[1] || r[2], forest.reflecting)

# The parity of packed component `c` (`_pack10`'s order) across the plane
# normal to spatial dimension `d`: the spacetime index of `x^d` is `d + 1`.
function _component_parity(c::Integer, d::Integer)
    for a in 1:4, b in 1:a
        _pairindex(a, b) == c || continue
        n = (a == d + 1) + (b == d + 1)
        return isodd(n) ? OddParity : EvenParity
    end
    throw(ArgumentError("there is no packed component $c of a symmetric 4×4"))
end

"""
    state_parity(forest; copies = 2) -> Vector or nothing

The parities of the state's variables under TreeAMR's mirrors, the
`parity` keyword of a `FieldSet` over `forest`: `h_ab` (variables `1:10`)
and `Π_ab` (`11:20`) are symmetric tensors, so component `ab` is
`OddParity` in dimension `d` exactly when `x^d` occurs an odd number of
times in `ab` — `h_tx` is odd in `x`, `h_xy` in `x` and `y`, `h_tt` and the
diagonal are even everywhere. `copies` repeats the `NC`-component pattern,
`2` for the state and `4` for the target cache (the target and its slope).
`nothing` over a forest without a reflecting face.
"""
function state_parity(forest; copies::Integer=2)
    _reflects(forest) || return nothing
    one = [ntuple(d -> _component_parity(c, d), Val(3)) for c in 1:NC]
    return repeat(one, copies)
end

"""
    even_parity(forest, nvars) -> Vector or nothing

`EvenParity` for every variable, or `nothing` over a forest without a
reflecting face: the declaration of a `G = 0` field set, which TreeAMR
requires over a reflecting forest and which nothing ever reads, since such
a field set has no ghosts to mirror and is never transferred.
"""
even_parity(forest, nvars::Integer) =
    _reflects(forest) ? fill(ntuple(_ -> EvenParity, Val(3)), Int(nvars)) : nothing

# --- the rotating seam (added 2026-10-04) --------------------------------------
#
# TreeAMR's M12 fills the ghosts across a rotating seam from the real data a
# quarter turn away and turns the variables with a signed map, which it
# requires on every field set over a forest with a seam, for the reason it
# requires a parity: how a variable turns is physics. The quarter turn `R`
# takes `e_{d1}` to `e_{d2}` and `e_{d2}` to `−e_{d1}`, and a covariant tensor
# turns index by index, `T′_ab(Rp) = R_a^c R_b^d T_cd(p)`: the index `d1` of
# the turned tensor is minus the index `d2` of the original, `d2` is plus
# `d1`, and `t` and the third dimension stay. So component `ab` of `h` or `Π`
# is plus or minus one other component, and the four turns compose to the
# identity. The other field sets are declared with the identity, as they are
# declared even under a mirror: they have `G = 0` and are neither filled nor
# transferred.
#
# Over a forest without a seam both helpers return `nothing`, so every field
# set of a run without one — its checkpoints included — is what it was.

# The packed component and the sign that component `c` (`_pack10`'s order)
# reads a quarter turn away, about the seam `(d1, d2)`: the spacetime index of
# `x^d` is `d + 1`.
function _component_rotation(c::Integer, d1::Integer, d2::Integer)
    turn(a) = a == d1 + 1 ? (d2 + 1, -1) : a == d2 + 1 ? (d1 + 1, 1) : (a, 1)
    for a in 1:4, b in 1:a
        _pairindex(a, b) == c || continue
        (a′, sa), (b′, sb) = turn(a), turn(b)
        return sa * sb * _pairindex(a′, b′)
    end
    throw(ArgumentError("there is no packed component $c of a symmetric 4×4"))
end

"""
    state_rotation(forest; copies = 2) -> Vector{Int} or nothing

How the state's variables turn under the quarter turn of TreeAMR's rotating
seam (M12), the `rotation` keyword of a `FieldSet` over `forest`: variable
`v` at `Rp` is `sign(r[v])` times variable `abs(r[v])` at `p`. `h_ab` and
`Π_ab` are covariant tensors, and with the seam `(d1, d2) = (1, 2)` — `R`
takes `x` to `y` and `y` to `−x` — `h_tx` turns into `−h_ty`, `h_ty` into
`h_tx`, `h_xx` and `h_yy` into each other, `h_xy` into `−h_xy`, `h_xz` into
`−h_yz`, and `h_tt`, `h_tz` and `h_zz` into themselves. `copies` repeats the
`NC`-component pattern, offset by `NC` each time, as
[`state_parity`](@ref)'s does. `nothing` over a forest without a seam
(added 2026-10-04).
"""
function state_rotation(forest; copies::Integer=2)
    seam = seam_dims(forest)
    seam === nothing && return nothing
    one = [_component_rotation(c, seam...) for c in 1:NC]
    return reduce(vcat, [sign.(one) .* (abs.(one) .+ k * NC) for k in 0:(copies - 1)])
end

"""
    identity_rotation(forest, nvars) -> Vector{Int} or nothing

`1:nvars`, every variable turning into itself, or `nothing` over a forest
without a rotating seam: the declaration of a `G = 0` field set, which TreeAMR
requires over a rotating forest and which nothing reads, as
[`even_parity`](@ref) is under a mirror (added 2026-10-04).
"""
identity_rotation(forest, nvars::Integer) =
    seam_dims(forest) === nothing ? nothing : collect(1:Int(nvars))
