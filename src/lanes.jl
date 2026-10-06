# SIMD lanes for the right-hand side on the CPU (added 2026-10-05).
#
# `CODE.md`, "The right-hand side on a CPU": the kernel's algebra is generic in its
# number type, so on the CPU it is evaluated on SIMD.jl's `Vec{W,T}` — `W`
# neighbouring points along the first axis at once — which is 2.0–2.4× the scalar
# kernel on Symmetry's EPYC nodes and the same `du` bit for bit. What that takes
# lives here: the width, the array accessor that turns one index into `W`
# neighbouring elements, which work items lead a group, and the `Vec` methods of
# the three helpers `pointwise.jl` gives the algebra (`_scale` needs none). Nothing
# here adds a method for a type this package does not own: SIMD.jl's `Vec` gets
# methods of the package's own helpers only.

"""
    default_simd_width(T, backend, N) -> Int

How many points the right-hand-side kernel evaluates at once: `1` unless the
backend is the CPU and `T` is a type SIMD.jl has lanes for (`Float32`, `Float64`),
and otherwise as many `T` as one vector register holds — 64 bytes with AVX-512,
32 bytes otherwise (AVX2; on aarch64 four `Float64` lanes measured 1.3× faster than
one, though a NEON register holds two) — halved until it is at most the block size
`N`. A device runs the scalar kernel: its threads are the lanes.
"""
function default_simd_width(::Type{T}, backend, N::Integer) where {T}
    backend isa CPU || return 1
    (T === Float64 || T === Float32) || return 1
    W = simd_vector_bytes() ÷ sizeof(T)
    while W > N
        W ÷= 2
    end
    return W
end

# The width of the host's vector registers in bytes, as the kernel uses them.
function simd_vector_bytes()
    Sys.ARCH === :x86_64 || return 32
    CPUID = Base.BinaryPlatforms.CPUID
    return CPUID.test_cpu_feature(CPUID.JL_X86_avx512f) ? 64 : 32
end

# A width a caller asks for, refused where the kernel could not honour it.
function check_simd_width(W::Integer, ::Type{T}, backend, N::Integer) where {T}
    W ≥ 1 && ispow2(W) || throw(ArgumentError(
        "the SIMD width is the number of points the kernel evaluates at once, a " *
        "power of two, but simd_width = $W"))
    W == 1 && return Int(W)
    backend isa CPU || throw(ArgumentError(
        "simd_width = $W asks for SIMD lanes, which only the CPU kernel has; on a " *
        "device every point is a thread of its own. Pass simd_width = 1, or leave " *
        "it to default_simd_width."))
    (T === Float64 || T === Float32) || throw(ArgumentError(
        "simd_width = $W asks for SIMD lanes of $T, and SIMD.jl has lanes only for " *
        "hardware floating-point types. Pass simd_width = 1 for $T."))
    W ≤ N || throw(ArgumentError(
        "simd_width = $W is wider than a block's row of N = $N points: a group of " *
        "lanes would reach past the row it belongs to."))
    return Int(W)
end

"""
    Lanes{W}(a)

The dense array `a` read and written `W` consecutive elements at a time: `l[i]` is
the `Vec{W}` of `a[i] … a[i + W − 1]`, and `l[i] = v` stores them. A stencil along
the first axis then reads `W` neighbouring points, and one along another axis `W`
neighbouring rows, with the linear indices the scalar kernel forms. The cartesian
`l[i, j, k, v, b]` is the same for the sampled gauge source, which is read at the
owned index.
"""
struct Lanes{W,A<:DenseArray}
    a::A
end
@inline Lanes{W}(a::A) where {W,A<:DenseArray} = Lanes{W,A}(a)

@inline function Base.getindex(l::Lanes{W}, i::Int) where {W}
    a = l.a
    return GC.@preserve a vload(Vec{W,eltype(a)}, pointer(a, i))
end
@inline function Base.getindex(l::Lanes{W}, I::Vararg{Int,M}) where {W,M}
    return @inbounds l[LinearIndices(l.a)[I...]]
end
@inline function Base.setindex!(l::Lanes{W}, v::Vec{W}, i::Int) where {W}
    a = l.a
    GC.@preserve a vstore(v, pointer(a, i))
    return l
end

# An array argument of the kernel as the lanes see it; anything else — `nothing`,
# an algebraic gauge source — as it is.
@inline lanes(::Val{W}, a::DenseArray) where {W} = Lanes{W}(a)
@inline lanes(::Val{W}, x) where {W} = x
# KernelAbstractions passes a `@Const` argument on the CPU as Julia's read-only
# `Const` wrapper, which has no `pointer`; its array is its one field.
@inline lanes(::Val{W}, a::Base.Experimental.Const) where {W} = Lanes{W}(a.a)

"""
    is_lane_leader(i, N, W) -> Bool

Whether the work item at owned index `i` along the first axis starts a group of `W`
points: every `W`-th from the first, as far as a whole group fits, and the group
that ends at the row's last point `N` — which overlaps the one before when `W` does
not divide `N`. Every point is in some group, and no group reaches past the row.
"""
@inline function is_lane_leader(i::Int, N::Int, W::Int)
    last = N - W + 1
    return (i ≤ last && (i - 1) % W == 0) || i == last
end

# The helpers of `pointwise.jl` on lanes: a damping rate is nonzero if it is in
# any lane, and a choice is made lane by lane.
@inline _anynonzero(x::Vec) = any(x != zero(x))
@inline _select(c::Vec{W,Bool}, x, y) where {W} = vifelse(c, x, y)
