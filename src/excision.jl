# Excision: the `:excised` interior variant (added in step X2b).
#
# `CODE.md`, "Excision (added 2026-10-05)". Points beyond the **excision
# surface** — the sphere `r = r_1` of step 5's layer, or the tracked offset
# surface `d = 0`, frozen for the run — are not evolved: `du = 0`, `F` never
# evaluated, their data finite and never read. The evolved points whose
# stencils would reach them take **closures** (`src/stencils.jl`, step X1):
# per axis and side the point counts `k⁻, k⁺ ∈ 0…G` consecutive non-excised
# points, and every stencil that would reach past them reads the closure on
# the nodes `−k⁻ … k⁺` instead — the mixed derivative nested, the
# dissipation Mattsson–Svärd–Nordström's.
#
# Four pieces, in the order a problem meets them:
#
#   1. **The classes** ([`build_excision`](@ref)), built once per problem: a
#      `UInt8` per stored point — centered, zone or excised — from the masks'
#      own predicate `is_evolved(interior_mask(int, t), x)` on owned points,
#      one ghost exchange of that bit through TreeAMR (so every ghost is its
#      owner's and an octant's walls mirror it), and a pass over the stored
#      points that marks a **zone** point: one some stencil of the right-hand
#      side would read an excised value through. Within a problem the classes
#      are the single source of truth for what is excised, in every kernel.
#   2. **The main kernel** (`evolution.jl`'s `gh_rhs_kernel!`) computes `F`
#      with the centered stencils at centered points — today's operator, bit
#      for bit — writes `0` at excised ones and nothing at zone points. With
#      the lopsided advection on, its centered points take the blend through
#      [`Lopsided`](@ref), a provider whose `adv` is the only difference.
#   3. **The zone kernel** ([`gh_zone_kernel!`](@ref)) computes `F` at zone
#      points with [`ClosureProvider`](@ref): the same physics,
#      `gh_rhs_at_point`, with the closures' stencils (step X2a's provider).
#   4. **The outflow monitor** ([`excision_rows`](@ref)), at record time: the
#      characteristic margins at the band, from the evolved state.
#
# No kernel reads an excised value, not even with weight zero (`0 · NaN =
# NaN`): every contraction here runs over a closure's own nodes, which lie in
# `[−k⁻, k⁺]`. And there is no fourth state writer: the `:excised` step
# limiter is a no-op, and the right-hand side writes `du` and nothing else.

# The per-point classes. Ghost points carry only the excised bit: their
# zone/centered distinction is never read.
const CLASS_CENTERED = 0x00
const CLASS_ZONE = 0x01
const CLASS_EXCISED = 0x02

# The monitor field set's variables: what the build's census and the record's
# outflow monitor write, one launch each (a field set of the excision's own,
# so that no other run's `diag` grows; proposed in step X2b).
const EXM_BAND = 1          # 1 at a zone point
const EXM_NONFINITE = 2     # census: 1 at an excised owned point; record: non-finite values
const EXM_NORMAL = 3        # census: inadmissible; record: b_n/a_n − 1 (floatmax off the band)
const EXM_AXIS = 4          # least per-axis b/a (census: over closure axes; record: over faces)
const EXM_FACES = 5         # census: least ε_KO; record: faces at the point
const EXM_INFLOW = 6        # record: faces with b/a < 1
const EXM_INTO = 7          # closure axes whose shift points into the excised set, b < 0
const NEXM = 7

"""
    excision_band_cells(q) -> Int

`W/h = max(G, ⌈√2 q/2⌉)`: how far, in cells, a stencil of the right-hand
side or of a monitor reaches from its point — the dissipation's `G` along an
axis, the mixed derivative's diagonal `√2 q/2`, strictly less than its
ceiling since `2(q/2)²` is never a square. The band `[r_E, r_E + W)` is
what the validity monitor reads as `:excised`'s layer, and the excised set
widened by `W` is the stencil monitors' mask ([`monitor_mask`](@ref)).
"""
function excision_band_cells(q::Integer)
    r = q ÷ 2
    return max(r + 1, isqrt(2 * r * r) + 1)
end

# --- the lopsided advection's blend ------------------------------------------

"""
    ExcisionBlend{T,NM}

Where the lopsided shift advection is switched on, as a function of position
(added in step X2b; `CODE.md`, "Excision"): `λ = smoothstep((d/h − start) /
width)` with `d = r_h(n̂) − r` the depth below the horizon `r_h(n̂)` about the
hole's static center — zero above `start` cells, full below `start + width`
— `h` the surface's spacing. The horizon is the tracked geometry's seed
shape, or for the sphere the background's smallest horizon radius as a
degree-0 shape with `r_in = r_out`, which the clamp makes exact. `isbits`; a
kernel argument of the main and the zone kernel, which evaluate it the same
way.
"""
struct ExcisionBlend{T,NM}
    center::SVector{3,T}
    shape::SVector{NM,T}
    lmax::Int
    r_in::T
    r_out::T
    h::T
    start::T
    width::T
end

@inline blend_weight(::Nothing, x) = zero(x[1])

@inline function blend_weight(bl::ExcisionBlend{T}, x) where {T}
    d1 = x[1] - bl.center[1]
    d2 = x[2] - bl.center[2]
    d3 = x[3] - bl.center[3]
    r = sqrt(d1 * d1 + d2 * d2 + d3 * d3)
    # Above every horizon radius less `start` cells: zero, without the series.
    r ≥ bl.r_out - bl.start * bl.h && return zero(T)
    n = iszero(r) ? SVector{3,T}(zero(T), zero(T), one(T)) :
        SVector{3,T}(d1 / r, d2 / r, d3 / r)
    rh = _surface_radius(bl.shape, bl.lmax, bl.r_in, bl.r_out, n)
    return smoothstep(((rh - r) / bl.h - bl.start) / bl.width)
end

# The blend for an interior and the surface's spacing, or `nothing` when the
# lopsided advection is off.
function excision_blend(int::Interior{T,:excised}, background, h) where {T}
    ex = int.excision
    upwind_on(ex) || return nothing
    rh = T(horizon_min_radius(background))
    c = center_at(int.center, zero(T))
    return ExcisionBlend{T,1}(c, SVector{1,T}(rh), 0, rh, rh, T(h),
                              ex.upwind_start, ex.upwind_width)
end

function excision_blend(int::FittedInterior{T,:excised,NM}, background,
                        h) where {T,NM}
    ex = int.excision
    upwind_on(ex) || return nothing
    c = center_at(int.center, zero(T))
    return ExcisionBlend{T,NM}(c, int.shape, int.lmax, int.r_in, int.r_out, T(h),
                               ex.upwind_start, ex.upwind_width)
end

# --- the stencil providers -------------------------------------------------------

"""
    Lopsided(T, ::Val{q}, st, inv_h, λ) -> Lopsided

The centered stencils with the lopsided shift advection blended in at weight
`λ` (added in step X2b): the main kernel's provider at the `:excised`
variant's centered points when the blend is on. `d1`, `d2`, `dmix` and `ko`
are [`Centered`](@ref)'s; `adv` returns the derivative it is handed **through
a branch** where `λ = 0` — so the exterior beyond the blend shell is today's
operator bit for bit — and otherwise `(1 − λ) ∂f_d + λ L`, `L` the order-`q`
lopsided derivative ([`lopsided_centered_weights`](@ref)) on the side the
shift points to, scaled by `1/h` itself.

**(Amended in step X4:** it holds the centered provider, `1/h` and `λ` and no
weights — the two lopsided rows are `@generated` constants formed in `adv`, as
[`Centered`](@ref)'s are in its methods — and the lopsided contraction is
generated term by term, so that the main kernel's `:excised` branch forms no
closure, which a device compiles as a call.**)**
"""
struct Lopsided{T,q} <: StencilProvider
    c::Centered{T,q}
    inv_h::T
    λ::T
end

@inline Lopsided(::Type{T}, ::Val{q}, st::NTuple{3,Int}, inv_h, λ) where {T,q} =
    Lopsided{T,q}(Centered(T, Val(q), st), T(inv_h), T(λ))

@inline d1(S::Lopsided, work, base::Int, d::Int) = d1(S.c, work, base, d)
@inline d2(S::Lopsided, work, base::Int, d::Int) = d2(S.c, work, base, d)
@inline dmix(S::Lopsided, work, base::Int, i::Int, j::Int) =
    dmix(S.c, work, base, i, j)
@inline ko(S::Lopsided, work, base::Int, d::Int) = ko(S.c, work, base, d)

# `∑_k w[k] u[base + (lo + k − 1)·stride]`, left to right from the first
# product — the left fold the table's contraction in the zone kernel forms, so
# the two kernels' blends are one operator. Generated with an explicit
# `:inline` meta, as `axis_stencil` is (amended in step X4: it was an
# `ntuple(Val(n)) do … end` folded by `_fold`, the same arithmetic, a closure).
@generated function _shifted_stencil(w::SVector{n}, work, base::Int, stride::Int,
                                     lo::Int) where {n}
    ex = :(w[1] * work[base + lo * stride])
    for k in 2:n
        ex = :($ex + w[$k] * work[base + (lo + $(k - 1)) * stride])
    end
    return Expr(:block, Expr(:meta, :inline), :(@inbounds $ex))
end

@inline function adv(S::Lopsided{T,q}, β_d, ∂f_d, work, base::Int,
                     d::Int) where {T,q}
    iszero(S.λ) && return ∂f_d
    stride = S.c.st[d]
    L = β_d ≥ 0 ?
        S.inv_h * _shifted_stencil(lopsided_centered_weights(T, Val(q), Val(1)), work,
                                   base, stride, lopsided_first(Val(q), Val(1))) :
        S.inv_h * _shifted_stencil(lopsided_centered_weights(T, Val(q), Val(-1)), work,
                                   base, stride, lopsided_first(Val(q), Val(-1)))
    return (one(T) - S.λ) * ∂f_d + S.λ * L
end

"""
    ClosureProvider

The stencils of an evolved point next to the excision surface (added in step
X2b; `CODE.md`, "Excision", "How the closure provider plugs in"): built per
zone point by [`closure_provider`](@ref), it holds the point's codes `k⁻, k⁺`
along the three axes, the class array and the point's linear index in it —
the class array has the working array's spatial strides, so a neighbour at
offset `a` along `i` is that index plus `a·st[i]` — the closure table (device
arrays), `1/h` and the lopsided blend's weight.

- `d1`, `d2` and `ko` contract the table's row `[·, k⁻ + 1, k⁺ + 1]` of the
  axis over its own nodes `lo:hi`, in ascending order, from the first
  product — `axis_stencil`'s order, so that the centered rows (the table's
  are `derivative_weights` bit for bit) give the centered contraction bit for
  bit;
- `dmix` runs the outer sum along `i` over the point's `i`-closure and, at
  each outer node `x + a e_i`, the inner sum along `j` over **that node's**
  `j`-closure, its codes read from the class array — `mixed_stencil`'s order;
- `adv` returns `∂f_d` where the blend is zero, and otherwise blends it with
  the table's lopsided row for the side the shift points to.

Every node a contraction reads is in `[−k⁻, k⁺]`: no excised value is read.
"""
struct ClosureProvider{T,G,C,TB} <: StencilProvider
    st::NTuple{3,Int}
    cls::C
    cbase::Int
    km::NTuple{3,Int}
    kp::NTuple{3,Int}
    tab::TB
    inv_h::T
    λ::T
end

# The number of consecutive non-excised points from `c` in steps of `step`,
# capped at `G`.
@inline function _run(cls, c::Int, step::Int, ::Val{G}) where {G}
    k = 0
    while k < G && (@inbounds cls[c + (k + 1) * step]) != CLASS_EXCISED
        k += 1
    end
    return k
end

"""
    closure_provider(T, ::Val{G}, st, cls, cbase, tab, inv_h, λ) -> ClosureProvider

The [`ClosureProvider`](@ref) of the point at linear index `cbase` of the
class array `cls`, with its codes `k±` read from it along the three axes.
"""
@inline function closure_provider(::Type{T}, ::Val{G}, st::NTuple{3,Int}, cls,
                                  cbase::Int, tab, inv_h, λ) where {T,G}
    # Spelled out rather than `ntuple(d -> …, Val(3))`: no closure (step X4).
    km = (_run(cls, cbase, -st[1], Val(G)), _run(cls, cbase, -st[2], Val(G)),
          _run(cls, cbase, -st[3], Val(G)))
    kp = (_run(cls, cbase, st[1], Val(G)), _run(cls, cbase, st[2], Val(G)),
          _run(cls, cbase, st[3], Val(G)))
    return ClosureProvider{T,G,typeof(cls),typeof(tab)}(st, cls, cbase, km, kp,
                                                        tab, T(inv_h), T(λ))
end

# `∑_{j = lo}^{hi} w[j + G + 1, k⁻ + 1, k⁺ + 1] u[base + j·stride]`, left to
# right from the first product; zero where the closure has no nodes.
@inline function _contract(w, lo::Int, hi::Int, km::Int, kp::Int, G::Int, work,
                           base::Int, stride::Int)
    lo > hi && return zero(eltype(w)) * zero(eltype(work))
    s = (@inbounds w[lo + G + 1, km + 1, kp + 1]) *
        (@inbounds work[base + lo * stride])
    j = lo + 1
    while j ≤ hi
        s += (@inbounds w[j + G + 1, km + 1, kp + 1]) *
             (@inbounds work[base + j * stride])
        j += 1
    end
    return s
end

@inline function _contract4(w, lo::Int, hi::Int, km::Int, kp::Int, iu::Int,
                            G::Int, work, base::Int, stride::Int)
    lo > hi && return zero(eltype(w)) * zero(eltype(work))
    s = (@inbounds w[lo + G + 1, km + 1, kp + 1, iu]) *
        (@inbounds work[base + lo * stride])
    j = lo + 1
    while j ≤ hi
        s += (@inbounds w[j + G + 1, km + 1, kp + 1, iu]) *
             (@inbounds work[base + j * stride])
        j += 1
    end
    return s
end

@inline function _closure_axis(w, lot, hit, S::ClosureProvider{T,G}, work,
                               base::Int, d::Int) where {T,G}
    km = S.km[d]
    kp = S.kp[d]
    lo = Int(@inbounds lot[km + 1, kp + 1])
    hi = Int(@inbounds hit[km + 1, kp + 1])
    return _contract(w, lo, hi, km, kp, G, work, base, S.st[d])
end

@inline d1(S::ClosureProvider, work, base::Int, d::Int) =
    _closure_axis(S.tab.d1, S.tab.d_lo, S.tab.d_hi, S, work, base, d)
@inline d2(S::ClosureProvider, work, base::Int, d::Int) =
    _closure_axis(S.tab.d2, S.tab.d_lo, S.tab.d_hi, S, work, base, d)
@inline ko(S::ClosureProvider, work, base::Int, d::Int) =
    _closure_axis(S.tab.ko, S.tab.ko_lo, S.tab.ko_hi, S, work, base, d)

# One outer node `a` of the nested mixed derivative: the point's `i`-weight at
# `a` times the `j`-closure of the point `x + a e_i`, with that point's own
# codes along `j`.
@inline function _dmix_term(S::ClosureProvider{T,G}, work, base::Int, i::Int,
                            j::Int, a::Int, kmi::Int, kpi::Int) where {T,G}
    ca = S.cbase + a * S.st[i]
    kmj = _run(S.cls, ca, -S.st[j], Val(G))
    kpj = _run(S.cls, ca, S.st[j], Val(G))
    loj = Int(@inbounds S.tab.d_lo[kmj + 1, kpj + 1])
    hij = Int(@inbounds S.tab.d_hi[kmj + 1, kpj + 1])
    inner = _contract(S.tab.d1, loj, hij, kmj, kpj, G, work, base + a * S.st[i],
                      S.st[j])
    return (@inbounds S.tab.d1[a + G + 1, kmi + 1, kpi + 1]) * inner
end

@inline function dmix(S::ClosureProvider{T,G}, work, base::Int, i::Int,
                      j::Int) where {T,G}
    kmi = S.km[i]
    kpi = S.kp[i]
    lo = Int(@inbounds S.tab.d_lo[kmi + 1, kpi + 1])
    hi = Int(@inbounds S.tab.d_hi[kmi + 1, kpi + 1])
    lo > hi && return zero(T) * zero(eltype(work))
    acc = _dmix_term(S, work, base, i, j, lo, kmi, kpi)
    a = lo + 1
    while a ≤ hi
        acc += _dmix_term(S, work, base, i, j, a, kmi, kpi)
        a += 1
    end
    return acc
end

@inline function adv(S::ClosureProvider{T,G}, β_d, ∂f_d, work, base::Int,
                     d::Int) where {T,G}
    iszero(S.λ) && return ∂f_d
    iu = β_d ≥ 0 ? 2 : 1
    km = S.km[d]
    kp = S.kp[d]
    lo = Int(@inbounds S.tab.lop_lo[km + 1, kp + 1, iu])
    hi = Int(@inbounds S.tab.lop_hi[km + 1, kp + 1, iu])
    L = S.inv_h * _contract4(S.tab.lop, lo, hi, km, kp, iu, G, work, base, S.st[d])
    return (one(T) - S.λ) * ∂f_d + S.λ * L
end

"""
    closure_arrays(T, ::Val{q}, dissipation, backend) -> NamedTuple

[`closure_table`](@ref)`(T, Val(q); dissipation)` as arrays on `backend` —
the zone kernel's argument (added in step X2b). Step X1 measured the
`isbits` table at 4624 bytes at `q = 4`, `Float64`, above CUDA's classic
4 kB kernel-parameter limit, so it travels as device arrays; a `NamedTuple`
of them, which KernelAbstractions adapts field by field.
"""
function closure_arrays(::Type{T}, ::Val{q}, dissipation::Symbol,
                        backend) where {T,q}
    ct = closure_table(T, Val(q); dissipation=dissipation)
    up(a) = to_backend(backend, Array(a))
    return (d1=up(ct.d1), d2=up(ct.d2), ko=up(ct.ko), lop=up(ct.lop),
            d_lo=up(ct.d_lo), d_hi=up(ct.d_hi), ko_lo=up(ct.ko_lo),
            ko_hi=up(ct.ko_hi), lop_lo=up(ct.lop_lo), lop_hi=up(ct.lop_hi))
end

# --- the zone kernel ---------------------------------------------------------------

"""
    gh_zone_kernel!(du, work, Hwork, origins, spacings, damping, γ2, ε_KO, t,
                    cls, zoneblocks, tab, blend, ::Val{G}, ::Val{q},
                    ::Val{HASH}, ::Val{DISS})

`F(u)` at the **zone** points of an `:excised` problem with the closures
(added in step X2b): [`gh_rhs_store!`](@ref)'s provider form with a
[`ClosureProvider`](@ref), everything else — `γ0`, `ε_KO/h`, the gauge source
— computed exactly as [`gh_rhs_kernel!`](@ref) computes it. Launched right
after the main kernel, over every owned point of every block, with a
block-uniform early exit through `zoneblocks` (TreeAMR has no launch over a
subset of blocks), and writing `du` at zone points only — the main kernel
wrote every other point.

**(Amended in step X4:** the store, not the two-vector form: `main`'s head and
Π components through the closure provider, each component stored as it is
finished, and no closure in the kernel's own body. The provider's contractions
stay generic — run-time loops over the table's rows — since the zone is a
shell of `10⁴`–`10⁵` points.**)**
"""
@kernel function gh_zone_kernel!(du, @Const(work), Hwork, @Const(origins),
                                 @Const(spacings), damping, γ2, ε_KO, t,
                                 @Const(cls), @Const(zoneblocks), tab, blend,
                                 ::Val{G}, ::Val{q}, ::Val{HASH},
                                 ::Val{DISS}) where {G,q,HASH,DISS}
    I = @index(Global, NTuple)
    b = I[4]
    if zoneblocks[b]
        st, sv, sb = work_strides(work)
        # The point in the class array: the working array's spatial strides,
        # one variable.
        cb = 1 + (b - 1) * sv + (I[1] + G[1] - 1) * st[1] +
             (I[2] + G[2] - 1) * st[2] + (I[3] + G[3] - 1) * st[3]
        if cls[cb] == CLASS_ZONE
            T = eltype(du)
            inner = (I[1], I[2], I[3])
            inv_h = inv(spacings[b])
            var = 1 + (b - 1) * sb + (I[1] + G[1] - 1) * st[1] +
                  (I[2] + G[2] - 1) * st[2] + (I[3] + G[3] - 1) * st[3]
            x = point_position(origins, spacings, b, I)
            γ0 = damping_rate(damping, t, x)
            εh = dissipation_rate(ε_KO, t, x) * inv_h
            S = closure_provider(T, Val(G[1]), st, cls, cb, tab, inv_h,
                                 blend_weight(blend, x))
            o, sd = state_offset(du, I)
            gh_rhs_store!(du, o, sd, S, T, work, Hwork, inner, b, var, sv, inv_h, γ0,
                          γ2, εh, Val(HASH), Val(DISS))
        end
    end
end

# --- the classes ------------------------------------------------------------------

# Pass 1: the excised bit at every owned point, from the masks' predicate.
@kernel function _excised_bit_kernel!(bits, @Const(origins), @Const(spacings),
                                      mask, ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    c = ntuple(d -> I[d] + G[d], Val(3))
    T = eltype(bits)
    bits[c..., 1, b] = is_evolved(mask, point_position(origins, spacings, b, I)) ?
                       zero(T) : one(T)
end

@inline _bit(bits, i::Int) = (@inbounds bits[i]) > one(eltype(bits)) / 2

# Whether a stencil of the right-hand side at the point `base` reads an
# excised point: the dissipation's `±G` along each axis, and the mixed
# derivatives' plane boxes of half-width `q/2`.
@inline function _reads_excised(bits, base::Int, st::NTuple{3,Int}, ::Val{G},
                                ::Val{q}) where {G,q}
    found = false
    for d in 1:3, a in 1:G
        found |= _bit(bits, base + a * st[d]) | _bit(bits, base - a * st[d])
    end
    r = q ÷ 2
    for (i, j) in ((1, 2), (1, 3), (2, 3)), a in (-r):r, e in (-r):r
        found |= _bit(bits, base + a * st[i] + e * st[j])
    end
    return found
end

# Pass 3, over every stored point: excised from the exchanged bit; at an owned
# point zone where a stencil reads an excised point, centered otherwise; a
# ghost that is not excised is marked centered, a placeholder never read.
@kernel function _class_kernel!(cls, @Const(bits), ::Val{G}, ::Val{q},
                                ::Val{N}) where {G,q,N}
    S = @index(Global, NTuple)                 # a stored index
    b = S[4]
    st, sv, _ = work_strides(bits)
    base = 1 + (b - 1) * sv + (S[1] - 1) * st[1] + (S[2] - 1) * st[2] +
           (S[3] - 1) * st[3]
    owned = (G[1] < S[1] ≤ G[1] + N) & (G[2] < S[2] ≤ G[2] + N) &
            (G[3] < S[3] ≤ G[3] + N)
    cls[base] = _bit(bits, base) ? CLASS_EXCISED :
                owned && _reads_excised(bits, base, st, Val(G[1]), Val(q)) ?
                CLASS_ZONE : CLASS_CENTERED
end

# The census at the build, over the owned points (one launch): the zone and
# excised indicators, the inadmissible zone points, and at every zone point
# the shift's component toward the excised side along every closure axis
# (`k_s < G`) as `b/a`, `b = −s β^d`, `a = α√γ^{dd}` — the least, and the
# count of negative ones — and the dissipation there.
@kernel function _census_kernel!(out, @Const(cls), @Const(work), @Const(origins),
                                 @Const(spacings), ε_KO, t,
                                 ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    T = eltype(out)
    st, sv, sb = work_strides(work)
    # The class array has the working array's spatial layout, one variable.
    cb = 1 + (b - 1) * sv + (I[1] + G[1] - 1) * st[1] +
         (I[2] + G[2] - 1) * st[2] + (I[3] + G[3] - 1) * st[3]
    cl = cls[cb]
    out[inner..., EXM_BAND, b] = cl == CLASS_ZONE ? one(T) : zero(T)
    out[inner..., EXM_NONFINITE, b] = cl == CLASS_EXCISED ? one(T) : zero(T)
    if cl == CLASS_ZONE
        var = 1 + (b - 1) * sb + (I[1] + G[1] - 1) * st[1] +
              (I[2] + G[2] - 1) * st[2] + (I[3] + G[3] - 1) * st[3]
        hv = SVector{NC,T}(ntuple(v -> (@inbounds work[var + (v - 1) * sv]),
                                  Val(NC)))
        x = point_position(origins, spacings, b, I)
        bad, rmin, nneg = _census_at(cls, cb, st, hv, Val(G[1]))
        out[inner..., EXM_NORMAL, b] = bad
        out[inner..., EXM_AXIS, b] = rmin
        out[inner..., EXM_FACES, b] = dissipation_rate(ε_KO, t, x)
        out[inner..., EXM_INTO, b] = nneg
    else
        out[inner..., EXM_NORMAL, b] = zero(T)
        out[inner..., EXM_AXIS, b] = floatmax(T)
        out[inner..., EXM_FACES, b] = floatmax(T)
        out[inner..., EXM_INTO, b] = zero(T)
    end
    out[inner..., EXM_INFLOW, b] = zero(T)
end

@inline function _census_at(cls, cb::Int, st::NTuple{3,Int}, hv::SVector{NC,T},
                            ::Val{G}) where {T,G}
    _, _, α, β, γu, _ = metric_quantities(_sym4(hv))
    bad = zero(T)
    rmin = floatmax(T)
    nneg = zero(T)
    for d in 1:3
        km = _run(cls, cb, -st[d], Val(G))
        kp = _run(cls, cb, st[d], Val(G))
        max(km, kp) ≥ G || (bad = one(T))
        a = α * sqrt(γu[d, d])
        for (s, k) in ((-1, km), (1, kp))
            k < G || continue
            ratio = -s * β[d] / a
            rmin = min(rmin, ratio)
            ratio < 0 && (nneg += one(T))
        end
    end
    return bad, rmin, nneg
end

# --- the outflow monitor --------------------------------------------------------

"""
    excision_normal(interior, t, x) -> SVector{3}

The outward unit normal covector of the excision surface's level set through
`x` (added in step X2b): radial for the sphere; for the tracked geometry the
gradient of `|y − c| − r_h(n̂(y))`, by central differences of relative step
`∛eps`, which for a diagnostic margin is ample.
"""
@inline function excision_normal(int::Interior{T}, t, x) where {T}
    c = center_at(int.center, t)
    d = SVector{3,T}(x[1] - c[1], x[2] - c[2], x[3] - c[3])
    r = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    return iszero(r) ? SVector{3,T}(zero(T), zero(T), one(T)) : d / r
end

@inline function excision_normal(int::FittedInterior{T}, t, x) where {T}
    c = center_at(int.center, t)
    p = SVector{3,T}(x[1] - c[1], x[2] - c[2], x[3] - c[3])
    r = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3])
    iszero(r) && return SVector{3,T}(zero(T), zero(T), one(T))
    δ = cbrt(eps(T)) * r
    F(y) = (ry = sqrt(y[1] * y[1] + y[2] * y[2] + y[3] * y[3]);
            ry - shape_radius(int, y / ry))
    g = SVector{3,T}(ntuple(Val(3)) do k
        e = SVector{3,T}(ntuple(l -> l == k ? δ : zero(T), Val(3)))
        (F(p + e) - F(p - e)) / (2δ)
    end)
    return g / sqrt(g[1] * g[1] + g[2] * g[2] + g[3] * g[3])
end

# The record's numbers at one band point: the non-finite count, the normal
# margin `b_n/a_n − 1`, and over the faces (`(d, s)` with the immediate
# neighbour excised, step X1's definition) the least `b/a`, their number and
# the inflow-like ones (`b/a < 1`); over the closure axes (`k_s < G`) the ones
# whose shift points into the excised set (`b < 0`).
@inline function _outflow_at(cls, cb::Int, st::NTuple{3,Int}, hv::SVector{NC,T},
                             Πv::SVector{NC,T}, n, ::Val{G}) where {T,G}
    nf = zero(T)
    for v in 1:NC
        isfinite(hv[v]) || (nf += one(T))
        isfinite(Πv[v]) || (nf += one(T))
    end
    _, _, α, β, γu, _ = metric_quantities(_sym4(hv))
    bn = β[1] * n[1] + β[2] * n[2] + β[3] * n[3]
    an = α * sqrt(n[1] * (γu[1, 1] * n[1] + γu[1, 2] * n[2] + γu[1, 3] * n[3]) +
                  n[2] * (γu[2, 1] * n[1] + γu[2, 2] * n[2] + γu[2, 3] * n[3]) +
                  n[3] * (γu[3, 1] * n[1] + γu[3, 2] * n[2] + γu[3, 3] * n[3]))
    normal = bn / an - one(T)
    rmin = floatmax(T)
    faces = zero(T)
    inflow = zero(T)
    into = zero(T)
    for d in 1:3
        a = α * sqrt(γu[d, d])
        for s in (-1, 1)
            ratio = -s * β[d] / a
            if (@inbounds cls[cb + s * st[d]]) == CLASS_EXCISED
                faces += one(T)
                rmin = min(rmin, ratio)
                ratio < 1 && (inflow += one(T))
            end
            _run(cls, cb, s * st[d], Val(G)) < G && ratio < 0 && (into += one(T))
        end
    end
    return nf, normal, rmin, faces, inflow, into
end

@kernel function _outflow_kernel!(out, @Const(state), @Const(cls), @Const(origins),
                                  @Const(spacings), interior, t,
                                  ::Val{G}) where {G}
    I = @index(Global, NTuple)
    b = I[4]
    inner = ntuple(d -> I[d], Val(3))
    T = eltype(out)
    n1, n2, n3 = size(cls, 1), size(cls, 2), size(cls, 3)
    st = (1, n1, n1 * n2)
    cb = 1 + (b - 1) * (n1 * n2 * n3) + (I[1] + G[1] - 1) * st[1] +
         (I[2] + G[2] - 1) * st[2] + (I[3] + G[3] - 1) * st[3]
    if cls[cb] == CLASS_ZONE
        hv = SVector{NC,T}(ntuple(v -> state[inner..., v, b], Val(NC)))
        Πv = SVector{NC,T}(ntuple(v -> state[inner..., NC + v, b], Val(NC)))
        x = point_position(origins, spacings, b, I)
        nf, normal, rmin, faces, inflow, into =
            _outflow_at(cls, cb, st, hv, Πv, excision_normal(interior, t, x),
                        Val(G[1]))
        out[inner..., EXM_BAND, b] = one(T)
        out[inner..., EXM_NONFINITE, b] = nf
        out[inner..., EXM_NORMAL, b] = normal
        out[inner..., EXM_AXIS, b] = rmin
        out[inner..., EXM_FACES, b] = faces
        out[inner..., EXM_INFLOW, b] = inflow
        out[inner..., EXM_INTO, b] = into
    else
        out[inner..., EXM_BAND, b] = zero(T)
        out[inner..., EXM_NONFINITE, b] = zero(T)
        out[inner..., EXM_NORMAL, b] = floatmax(T)
        out[inner..., EXM_AXIS, b] = floatmax(T)
        out[inner..., EXM_FACES, b] = zero(T)
        out[inner..., EXM_INFLOW, b] = zero(T)
        out[inner..., EXM_INTO, b] = zero(T)
    end
end

# --- the problem's excision -------------------------------------------------------

"""
    ExcisionData

What an `:excised` [`GHProblem`](@ref) carries (added in step X2b), built
once by [`build_excision`](@ref): the per-point classes (`UInt8`, one per
stored point, the working array's spatial layout), a device `Bool` per block
— whether it holds a zone point — the closure table as device arrays, the
lopsided blend (`nothing` when off), a ghost-free field set of
[`NEXM`](@ref) variables the census and the outflow monitor write into, the
geometry it was built for, the band's width `W`, the surface's spacing `h`,
and the counts over the owned points.
"""
struct ExcisionData{T,C,Z,TB,BL,M,I}
    classes::C
    zoneblocks::Z
    table::TB
    blend::BL
    monitor::M
    interior::I
    W::T
    h::T
    q::Int
    nexcised::Int
    nzone::Int
    ncentered::Int
    nzoneblocks::Int
    min_ratio::Float64              # the least b/a over the closure axes at the build
end

excision_classes(::Nothing) = nothing
excision_classes(ex::ExcisionData) = ex.classes
excision_blend(::Nothing) = nothing
excision_blend(ex::ExcisionData) = ex.blend

"""
    check_excision_mesh(forest, interior, q; t = 0) -> h

`CODE.md`'s third check (added in step X2b): every leaf within `(G + q + 2)
h` of the excision surface, on either side, is on **one level**, so that no
prolongation reads excised data into a ghost an evolved stencil reads — a
fine ghost at a coarse-fine face is interpolated from `(q + 2)/2` coarse
points around it. `h` is the spacing of the blocks containing the surface,
which it returns. It throws, naming the levels and the remedy.
"""
function check_excision_mesh(forest::Forest{3}, int::Interior{T,:excised}, q::Integer;
                             t=zero(T)) where {T}
    h, _ = layer_spacing(forest, int, T(t))
    w = (q ÷ 2 + 1 + q + 2) * h
    return _check_one_level(forest, center_at(int.center, T(t)), int.r_1 - w,
                            int.r_1 + w, w, h)
end

function check_excision_mesh(forest::Forest{3}, int::FittedInterior{T,:excised},
                             q::Integer; t=zero(T)) where {T}
    c = center_at(int.center, T(t))
    h, _ = _annulus_spacing(forest, T, c, int.r_in - int.offset,
                            int.r_out - int.offset)
    w = (q ÷ 2 + 1 + q + 2) * h
    return _check_one_level(forest, c, (int.r_in - int.offset) - w,
                            (int.r_out - int.offset) + w, w, h)
end

function _check_one_level(forest, c, lo, hi, w, h)
    T = typeof(h)
    levels = Int[]
    for k in forest.leaves
        _box_meets_annulus(block_extent(T, forest, k), c, lo, hi) || continue
        push!(levels, level(k))
    end
    isempty(levels) && throw(ArgumentError(
        "no block of this forest meets the excision surface's neighbourhood " *
        "[$lo, $hi] about $(Tuple(c)): the surface is outside the domain."))
    allequal(levels) || throw(ArgumentError(
        "the blocks within (G + q + 2) h = $w of the excision " *
        "surface lie on levels $(sort!(unique(levels))): every leaf meeting " *
        "[$lo, $hi] about $(Tuple(c)) must be on one level (CODE.md, " *
        "\"Excision\", \"Checks\"). A coarse-fine face there prolongs from " *
        "(q + 2)/2 coarse points around each fine ghost, and near the surface " *
        "those include excised data, which would reach an evolved stencil. Put " *
        "the surface inside one refinement level — a hole_forest shell that " *
        "covers the annulus, or a deeper r_1."))
    return h
end

"""
    check_excision_case(case, interior, q)

The refusals of an `:excised` problem that the case decides (added in step
X2b; `CODE.md`, "Excision", and `PLAN.md`'s hand-over from step X1), each
saying why:

- **`ε_KO > 0`**: without dissipation the closures grow as the interior
  itself does (`+0.08` to `+0.14/M` on X1's plane), and the extrapolation
  family at `+1–4/M` on the surface; a profile that vanishes at the surface
  is refused by [`build_excision`](@ref), which reads it at every zone point;
- **a static hole**: the geometry is frozen for the run, and a point that
  leaves the excised set would need values — the moving round's;
- **no range projection**: the excised set is never read and the band is
  evolved, so there is nothing for a clamp to guard, and a fourth reason to
  write the state is not wanted;
- **`m ≥ ⌈√3 G⌉` with a `Horizon`**: the finder's footprint reaches `√3 G h`
  from its query on a diagonal and must not touch the excised set — `6`
  cells at `q = 4`, `4` at `q = 2`.
"""
function check_excision_case(case::GHCase{T}, int, q::Integer) where {T}
    G = q ÷ 2 + 1
    has_dissipation(case.ε_KO) || throw(ArgumentError(
        "an :excised hole needs Kreiss–Oliger dissipation at its surface, and " *
        "this case has ε_KO = $(case.ε_KO): without it the per-axis closures " *
        "grow as the interior itself does (+0.08 to +0.14/M on step X1's " *
        "plane) and the extrapolation family at +1 to +4/M on the surface " *
        "(CODE.md, \"Excision: the analysis (step X1)\"). Use GHSO2's recipe, " *
        "ε_KO ≈ 1/2."))
    iszero(case.center.v) || throw(ArgumentError(
        "an :excised hole is static in this round (CODE.md, \"Excision\"): its " *
        "geometry is frozen for the run, and a hole that moves leaves points " *
        "behind it that were excised and have no values — the moving round's " *
        "problem. This case moves at $(Tuple(case.center.v))."))
    case.bounds === nothing || throw(ArgumentError(
        "an :excised hole takes no range projection: its excised set is never " *
        "read and its band is evolved by the equations, so a clamp has nothing " *
        "to guard and would be a fourth writer of the state. Leave `bounds = " *
        "nothing`."))
    need = ceil(Int, sqrt(3) * G)
    case.horizon === nothing || int.margin ≥ need || throw(ArgumentError(
        "this :excised case finds its horizon, and the finder's interpolation " *
        "footprint reaches √3 G h = $(sqrt(3) * G) h from a query on a " *
        "diagonal: with the surface m = $(int.margin) cells inside the " *
        "horizon it would read the excised set, so m ≥ ⌈√3 G⌉ = $need at q = " *
        "$q (CODE.md, \"Excision\", \"The least depth\")."))
    return nothing
end

"""
    build_excision(U, schedule, case, interior; q, t = 0) -> ExcisionData

The classes of an `:excised` problem and everything its kernels need (added
in step X2b), in three passes over the mesh:

1. the excised bit at every owned point, `!is_evolved(interior_mask(int, t),
   x)` — the masks' own predicate, so that the classes, the norms, the speed
   and the horizon guard exclude the same set;
2. one `fill_ghosts!` of a one-variable field set with even parity, so that
   every ghost is its owner's bit and an octant's walls mirror it (the outer
   faces' ghosts take the predicate at their own positions);
3. a pass over every stored point writing the class — excised, zone (an
   owned point some stencil of the right-hand side would read an excised
   point through: the dissipation's `±G` along an axis, the mixed
   derivatives' boxes of half-width `q/2`) or centered.

Then a census of the owned points, and the refusals that need the mesh or
the state: [`check_excision_case`](@ref)'s, the one level at the surface
([`check_excision_mesh`](@ref)), a zone point with no admissible closure
(excised on both sides of one axis within reach, which a convex excised set
never makes), the dissipation vanishing at a zone point, and — **the
refusal of the spinning holes, which is the physics and not the spin** — a
closure axis along which the shift of the state in `U`'s working array
points into the excised set (`b/a < 0`), where X1's frozen line found the
closure unstable at `0.03–0.19/h`.

`U`'s working array must hold the state at its owned points; the ghosts are
not read.
"""
function build_excision(U::FieldSet{T,3}, schedule, case::GHCase{T}, int;
                        q::Integer, t=zero(T)) where {T}
    interior_variant(int) === :excised || throw(ArgumentError(
        "build_excision builds the classes of an :excised interior; this one " *
        "is :$(interior_variant(int))."))
    check_excision_case(case, int, q)
    h = check_excision_mesh(U.forest, int, q; t=T(t))
    G = q ÷ 2 + 1
    backend = get_backend(U.work)
    forest = U.forest
    origins = to_backend(backend, block_origins(forest, T))
    spacings = to_backend(backend, block_spacings(forest, T))
    mask = interior_mask(int, T(t))

    # (1) and (2): the bit, exchanged.
    bits = FieldSet{T}(forest, 1; G=U.G, centering=U.centering,
                       parity=even_parity(forest, 1), backend=backend)
    fill!(bits.work, zero(T))
    map_blocks!(_excised_bit_kernel!, bits, bits.work, origins, spacings, mask,
                Val(U.G))
    if has_outer_face(case)
        fill_ghosts!(bits, schedule;
                     boundary=CellBoundary(AllVariables(
                         (x, δ) -> (is_evolved(mask, x) ? zero(x[1]) : one(x[1]),))))
    else
        fill_ghosts!(bits, schedule)
    end

    # (3): the classes, over every stored point.
    n = size(U.work)
    classes = allocate(backend, UInt8, (n[1], n[2], n[3], nblocks(U)))
    map_blocks!(_class_kernel!, bits, classes, bits.work, Val(U.G), Val(Int(q)),
                Val(forest.N); stored=true)

    # The census.
    monitor = FieldSet{T}(forest, NEXM; G=0, centering=U.centering,
                          parity=even_parity(forest, NEXM), backend=backend)
    map_blocks!(_census_kernel!, U, monitor.work, classes, U.work, origins,
                spacings, case.ε_KO, T(t), Val(U.G))
    total(v) = round(Int, tofloat64(mesh_mapreduce(identity, +, zero(T), monitor;
                                                   vars=v)))
    least(v) = mesh_mapreduce(identity, min, floatmax(T), monitor; vars=v)
    nzone = total(EXM_BAND)
    nexcised = total(EXM_NONFINITE)
    nbad = total(EXM_NORMAL)
    ninto = total(EXM_INTO)
    rmin = least(EXM_AXIS)
    εmin = least(EXM_FACES)
    perblock = block_mapreduce(identity, +, zero(T), monitor; vars=EXM_BAND)
    zb = Bool[x > 0 for x in perblock]

    nzone > 0 || throw(ArgumentError(
        "the excision surface of this :excised interior has no zone point on " *
        "this mesh — no evolved stencil reads the excised set — so nothing is " *
        "excised where it matters: $nexcised owned points are excised. The " *
        "surface is outside the domain, or smaller than a cell."))
    nbad == 0 || throw(ArgumentError(
        "$nbad zone points of this :excised interior have excised points on " *
        "both sides of one axis within the reach G = $G, so no closure of the " *
        "per-axis family applies to them (closure_admissible; CODE.md, " *
        "\"Excision\"). A convex excised set never makes such a point: the " *
        "surface is not convex at this resolution — refine, or move it."))
    εmin > 0 || throw(ArgumentError(
        "the Kreiss–Oliger amplitude vanishes at a zone point of this :excised " *
        "interior (its least value there is $εmin): the closures need " *
        "dissipation at the surface, without which they grow as the interior " *
        "itself does (CODE.md, \"Excision: the analysis (step X1)\"). Give the " *
        "case a profile that is positive at the surface."))
    ninto == 0 || throw(ArgumentError(
        "at $ninto (zone point, closure axis) pairs of this :excised interior " *
        "the shift points into the excised set — b/a = −s β^d/(α√γ^{dd}) < 0, " *
        "the least $(tofloat64(rmin)) — and on step X1's frozen line the " *
        "per-axis closure is unstable there, at 0.03–0.19/h, under every " *
        "dissipation closure and with or without the lopsided advection " *
        "(CODE.md, \"Excision\"). Frame dragging makes such faces on a spinning " *
        "hole's lego surface (2–18 % of them); this round covers the static " *
        "Kerr-Schild a = 0 hole, whose shift points out of the excised set at " *
        "every face."))

    tab = closure_arrays(T, Val(Int(q)), excision_closure(int.excision), backend)
    W = T(excision_band_cells(q)) * h
    ntotal = nleaves(forest) * forest.N^3
    zoneblocks = to_backend(backend, zb)
    blend = excision_blend(int, case.background, h)
    return ExcisionData{T,typeof(classes),typeof(zoneblocks),typeof(tab),
                        typeof(blend),typeof(monitor),typeof(int)}(
        classes, zoneblocks, tab, blend, monitor, int, W, h, Int(q), nexcised,
        nzone, ntotal - nzone - nexcised, count(zb), tofloat64(rmin))
end

"""
    monitor_mask(p::GHProblem, t) -> mask

The mask of the monitors that take stencils — [`gh_constraint!`](@ref),
[`adm_constraint!`](@ref) and the indicator ([`gh_indicator!`](@ref)) — by
default (added in step X2b): the interior's own mask
([`interior_mask`](@ref)) for every variant but `:excised`, and for it the
excised set widened by the band `W` ([`excision_monitor_mask`](@ref)), so
that no monitor's stencil reads an excised value. The error, the speed, the
non-finite count, the validity monitor and the horizon guard read points,
not stencils, and keep `interior_mask`, which counts the band.
"""
monitor_mask(p, t) = _monitor_mask(p.excision, p.interior, t)
_monitor_mask(::Nothing, int, t) = interior_mask(int, t)
_monitor_mask(ex::ExcisionData, int, t) = excision_monitor_mask(int, t, ex.W)

# The validity monitor's two bands: the interior's layer and the shell outside
# it, or for `:excised` the band `[r_E, r_E + W)` and the `width` beyond it.
_validity_bands(::Nothing, int, t, width) =
    (layer_mask(int, t), shell_mask(int, t, width))
_validity_bands(ex::ExcisionData, int, t, width) =
    (layer_mask(int, t; band=ex.W), shell_mask(int, t, width; band=ex.W))

"""
    excision_rows(p::GHProblem, u, t) -> NamedTuple

The outflow monitor's rows of the analysis record (added in step X2b;
`CODE.md`, "Excision", "Record rows"), from the state `u` at the band's
points — the zone points — and `nothing` in every row for a problem that
is not `:excised`:

- `excision_band`, `excision_band_nonfinite`: the band's points and the
  non-finite values among their states;
- `excision_normal_min`: the least `b_n/a_n − 1` along the surface's normal
  (`b_n = β^i n_i`, `a_n = α√(γ^{ij} n_i n_j)`), which must stay positive —
  normal outflow;
- `excision_faces`, `excision_axis_min`, `excision_inflow`: step X1's faces —
  a band point with an excised immediate neighbour along an axis — their
  number, their least per-axis `b/a` (`b = −s β^d` toward the excised side
  `s`, `a = α√γ^{dd}`) and the inflow-like ones, `b/a < 1`;
- `excision_into`: the (band point, closure axis) pairs whose shift points
  into the excised set, `b < 0` — the build's refusal, counted every chunk.
"""
function excision_rows(p, u, t)
    ex = p.excision
    ex === nothing && return (excision_band=nothing, excision_band_nonfinite=nothing,
                              excision_normal_min=nothing, excision_faces=nothing,
                              excision_axis_min=nothing, excision_inflow=nothing,
                              excision_into=nothing)
    T = eltype(p.U.work)
    map_blocks!(_outflow_kernel!, p.U, ex.monitor.work, statearray(u, p.U),
                ex.classes, p.origins, p.spacings, p.interior, T(t), p.valG)
    total(v) = round(Int, tofloat64(mesh_mapreduce(identity, +, zero(T), ex.monitor;
                                                   vars=v)))
    least(v) = (x = mesh_mapreduce(identity, min, floatmax(T), ex.monitor; vars=v);
                x == floatmax(T) ? nothing : tofloat64(x))
    return (excision_band=total(EXM_BAND), excision_band_nonfinite=total(EXM_NONFINITE),
            excision_normal_min=least(EXM_NORMAL), excision_faces=total(EXM_FACES),
            excision_axis_min=least(EXM_AXIS), excision_inflow=total(EXM_INFLOW),
            excision_into=total(EXM_INTO))
end

"""
    gh_zone!(du, p::GHProblem, t)

The zone kernel's launch for an `:excised` problem, right after the main
kernel in [`gh_rhs!`](@ref), on the working array the main kernel read.
"""
function gh_zone!(du, p, t)
    ex = p.excision
    map_blocks!(gh_zone_kernel!, p.U, statearray(du, p.U), p.U.work,
                gauge_work(p.Hsrc), p.origins, p.spacings, p.case.γ0, p.case.γ2,
                p.case.ε_KO, eltype(p.U.work)(t), ex.classes, ex.zoneblocks,
                ex.table, ex.blend, p.valG, p.valq, p.valH, p.valdiss)
    return nothing
end
