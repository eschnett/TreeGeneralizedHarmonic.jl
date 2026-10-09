# The bridges over `Base`'s gaps at a software floating-point type.
#
# The package is generic in its element type, and so is TreeAMR's mesh; what
# is not generic is `Base`. `mod`, `ceil(Int, ·)` and `Float64(·)` are all
# `MethodError`s at a MultiFloat, and every one of them sits on a path the
# drivers will take -- the gauge wave's periodic exact solution, the count of
# chunks and of steps in a chunk, the conversion into the `Float64` analysis
# record and the `Float64` horizon solve. Without the four functions below
# the package is `Float32`-and-`Float64`-only, and the symptom is a
# `MethodError` from inside a run rather than anything the type system warns
# about.
#
# The three types each catch a different fault:
#
#   Float64    is the baseline, and the type every measured number in
#              `CODE.md` is taken at. It is also the device requirement:
#              `Float64` on Symmetry's H200 is what the proof of concept
#              runs in.
#   Float32    is the *leak detector*. A stray Float64 operand widens the
#              result, so a returned Float64 names the leak. It is
#              desirable rather than required here -- a `Float32` failure
#              is recorded, not fixed at `Float64`'s expense.
#   Float32x2  is the *off the beaten path* detector: a software type built
#              from two Float32 limbs, which no Float64 fast path can serve.
#              It cannot detect leaks -- MultiFloats promotes Float64
#              *downward* -- it tests instead that nothing depends on a
#              hardware float at all.
#
# See "Precision, threads, devices" in `CODE.md`.

using MultiFloats: Float32x2

const FLOATTYPES = (Float64, Float32, Float32x2)

@testset "Base's gaps at a software float are bridged: T=$T" for T in FLOATTYPES
    # `wrap` is `mod` without `rem`: the gauge wave's exact solution is
    # periodic in `x - t` and has to come back into the box before it is
    # evaluated, and `Base.mod` on floats closes through `rem`, which
    # MultiFloats does not define.
    @test TreeGeneralizedHarmonic.wrap(T(9//4), one(T)) ≈ T(1//4)
    @test TreeGeneralizedHarmonic.wrap(-T(1//4), one(T)) ≈ T(3//4)
    @test TreeGeneralizedHarmonic.wrap(T(9//4), one(T)) isa T
    @test TreeGeneralizedHarmonic.wrap(T(3), T(2)) isa T

    # `ceil(Int, ·)` and `floor(Int, ·)` close through a conversion to
    # `Integer` that MultiFloats does not provide either. The results are
    # counts -- of chunks, of steps, of buffer cells -- so they are `Int`
    # and the identity is asserted with `===`.
    @test TreeGeneralizedHarmonic.ceilint(T(5//2)) === 3
    @test TreeGeneralizedHarmonic.floorint(T(5//2)) === 2
    @test TreeGeneralizedHarmonic.ceilint(-T(5//2)) === -2
    @test TreeGeneralizedHarmonic.floorint(-T(5//2)) === -3
    @test TreeGeneralizedHarmonic.ceilint(T(2)) === 2         # already integral

    # The chunk count of a run meant to be a whole number of chunks, where
    # the quotient rounds an ulp off the integer (`0.33/0.03` is
    # `11.000000000000002` at Float64, `0.15f0/0.005f0` is `30.000002f0`):
    # `ceilint` gave one chunk more. A genuine remainder still adds one.
    TGH = TreeGeneralizedHarmonic
    @test TGH.chunk_count(T(33//100), T(3//100)) == 11
    @test TGH.chunk_count(T(7//100), T(1//100)) == 7
    @test TGH.chunk_count(T(3//20), T(1//200)) == 30
    @test TGH.chunk_count(T(1//5), T(1//200)) == 40
    @test TGH.chunk_count(T(1//5), T(1//5)) == 1
    @test TGH.chunk_count(T(1//5), T(3//40)) == 3
    @test TGH.chunk_count(T(1//10), T(1//3)) == 1
    for (te, ch, n) in ((T(1//5), T(1//200), 40), (T(33//100), T(3//100), 11))
        @test TGH.chunk_bounds(n, n, te, ch)[2] === te
        @test all(c -> TGH.chunk_bounds(c, n, te, ch)[2] > TGH.chunk_bounds(c, n, te, ch)[1],
                  1:n)
    end
    @test TreeGeneralizedHarmonic.floorint(T(2)) === 2

    # `Float64(x)` is not universal: MultiFloats defines a conversion only
    # to its own limb type, so `Float64(::Float32x2)` is a `MethodError`
    # while `Float32(::Float32x2)` is not. This is the bridge into the
    # analysis time series and the horizon finder, both host `Float64`.
    @test TreeGeneralizedHarmonic.tofloat64(T(1//2)) === 0.5
    @test TreeGeneralizedHarmonic.tofloat64(T(-3)) === -3.0

    # And `wrap` really is `mod` where `mod` exists, which is what lets the
    # `Float64` numbers in `CODE.md` stay put when a driver goes generic.
    if T <: Base.IEEEFloat
        @test TreeGeneralizedHarmonic.wrap(T(9//4), one(T)) === mod(T(9//4), one(T))
        @test TreeGeneralizedHarmonic.wrap(-T(1//4), one(T)) === mod(-T(1//4), one(T))
    end
end
