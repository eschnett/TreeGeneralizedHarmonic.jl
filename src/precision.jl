# The operations this package performs that a *software* floating-point type
# does not provide, written so that they need only what every type here has.
#
# The package is generic in its element type `T`, and TreeAMR's mesh is too
# (see "Precision, threads, devices" in `CODE.md`). What is not generic is
# `Base`: MultiFloats.jl implements `floor` and `ceil` returning a float, but
# no conversion to `Integer` and no `rem`, so `ceil(Int, x)` and `mod(x, y)` --
# both of which close through those -- are `MethodError`s at `Float32x2` while
# working at `Float32` and `Float64`. Neither is a physics decision, so neither
# belongs spelled out at the call site.
#
# Copied from TreeWave rather than depended on: the four packages are read
# together, but a sample application does not depend on another sample
# application. See `CLAUDE.md`, "Things that will bite".
#
# Nothing here is exported. They are spellings, not concepts.

"""
    wrap(x, L)

`x` reduced into `[0, L)` for positive `L` — what `mod(x, L)` means on a
periodic box.

Spelled `x - L·floor(x/L)` rather than `mod`, because `Base.mod` on floats
goes through `rem`, which MultiFloats.jl does not define. The two agree
wherever `L > 0`, which is the only case a box produces.

Its consumer here is the gauge wave, whose exact solution is periodic in
`x - t` and has to be brought back into the box before it is evaluated; the
boosted hole's analytic center is *not* wrapped, because its box is
Dirichlet and the hole crosses it once.
"""
wrap(x, L) = x - L * floor(x / L)

"""
    ceilint(x)
    floorint(x)

`ceil(Int, x)` and `floor(Int, x)`, for a type that may not define
`Int(::AbstractFloat)`.

Both `ceil(Int, x)` and `floor(Int, x)` close through a conversion to
`Integer` that MultiFloats.jl does not provide — and neither does a detour
through `Float64`, since `Float64(::Float32x2)` is not defined either. What
*every* `AbstractFloat` in Julia converts to is `BigFloat`, so that is the
fallback, taken only once the value is already an exact integer and therefore
only ever exact. A hardware float never reaches it.

The fallback allocates, which is why these are confined to what they are used
for: host-side control flow evaluated a handful of times per run — the number
of chunks a run takes, the number of steps in a chunk, the width of the
refinement buffer in cells — and never per point.
"""
ceilint(x) = _toint(ceil(x))
floorint(x) = _toint(floor(x))

# `y` is an exact integer value by construction, so both branches are exact.
_toint(y::Base.IEEEFloat) = Int(y)
_toint(y) = Int(BigFloat(y))
roundint(x) = _toint(round(x))

"""
    chunk_count(t_end, chunk)

The number of chunks to `t_end`: `⌈t_end / chunk⌉`, except that a quotient
within a few ulp of an integer `m` gives `m`, so that a run meant to be a
whole number of chunks is not given one more by rounding. The tolerance is
the rounding a quotient of two values of type `T` can carry — a few ulp of
the quotient, plus a few ulp of `t_end` in units of `chunk` — so it never
absorbs a genuine remainder. TreeHydro's `chunk_count` (its step 12), which
is IMEXRungeKutta's `step_count` rule applied to the chunk cadence.

`ceilint(t_end / chunk)`, which [`evolve!`](@ref) used until 2026-10-09,
overcounts whenever the quotient rounds just above an integer: `0.33/0.03`
is `11.000000000000002` at `Float64`, which gave a twelfth chunk from
`0.32999999999999996` to `0.33` with a step, a row, a regrid and a
checkpoint of its own; and `0.07/0.01 = 7.000000000000001` gave an empty
eighth chunk, so the real last one regridded after its row. Pair it with
[`chunk_bounds`](@ref), whose last chunk ends at `t_end` exactly.
"""
function chunk_count(t_end::T, chunk::T) where {T}
    r = t_end / chunk
    m = round(r)
    tol = 4 * (eps(r) + eps(t_end) / chunk)
    return m >= 1 && abs(r - m) <= tol ? roundint(m) : ceilint(r)
end

"""
    chunk_bounds(c, nchunks, t_end, chunk) -> (tstart, stop)

The span of chunk `c` of [`chunk_count`](@ref)`(t_end, chunk) == nchunks`:
`(c − 1) · chunk` to `c · chunk`, both capped at `t_end`, the last ending at
`t_end` exactly — `nchunks · chunk` can round an ulp below it
(`40 · 0.005f0 == 0.19999999f0`). TreeHydro's spans.
"""
function chunk_bounds(c::Integer, nchunks::Integer, t_end::T, chunk::T) where {T}
    tstart = min((c - 1) * chunk, t_end)
    stop = c == nchunks ? t_end : min(c * chunk, t_end)
    return tstart, stop
end

"""
    tofloat64(x)

`x` as a `Float64` — the bridge to everything that is a `Float64` whatever
the run's type is: the analysis time series, the horizon finder's host-side
solve, and the numbers a test compares against `CODE.md`.

`Float64(x)` is not universal either: MultiFloats.jl defines a conversion
only to its own *limb* type, so `Float64(::Float32x2)` is a `MethodError`
while `Float32(::Float32x2)` is not. `BigFloat` is again the common currency.
Host-side, and at chunk frequency — never per point.
"""
tofloat64(x::Base.IEEEFloat) = Float64(x)
tofloat64(x) = Float64(BigFloat(x))
