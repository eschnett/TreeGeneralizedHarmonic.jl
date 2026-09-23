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
the background at each outward-facing ghost point, or `nothing` when the
case is periodic in every dimension and there is no physical boundary at
all.

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
    all(case.periodic) && return nothing
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
