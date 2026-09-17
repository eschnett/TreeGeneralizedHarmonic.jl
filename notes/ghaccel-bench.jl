# Copied verbatim from `/Users/eschnett/src/jl/GeneralizedHarmonicAccel/bin/bench.jl` on 2026-09-16, repository at commit `4980f2e`,
# with uncommitted local changes in the working tree. The original repository is unpublished; this copy is the
# citable reference for TreeGeneralizedHarmonic's `CODE.md`. Do not edit it here;
# amend `CODE.md` instead.

using CUDA
using GeneralizedHarmonicAccel

################################################################################

# nsys profile --output=report1 julia --project=@. --optimize bin/bench.jl
# ncu --export=report1 --set=full julia --project=@. --optimize bin/bench.jl

function bench()
    println("GeneralizedHarmonicAccel")
    println("Written 2026-01-10 by Erik Schnetter <schnetter@gmail.com>")

    # - CUDA (Nvidia A40), 512^3 × 32:
    #   Memory:        120 Byte   696   GByte/sec   172.41e-12 seconds/cell update
    #   Arithmetic:   2604 Flop    37.4 TFlop/sec    69.63e-12 seconds/cell update
    #
    #   - theoretical best:                  172.41e-12 seconds/cell update   (100.0% of peak)
    #   - div:                  1.797144 s   418.43e-12 seconds/cell update   ( 41.2% of peak)
    #   - int32, arrays, div:   1.817580 s   423.19e-12 seconds/cell update   ( 40.7% of peak)
    #   - block size:           1.491878 s   347.35e-12 seconds/cell update   ( 49.6% of peak)

    # - CUDA (Nvidia H200), 512^3 × 32:
    #   Memory:        120 Byte    4.8 TByte/sec   25.00e-12 seconds/cell update
    #   Arithmetic:   2604 Flop   67   TFlop/sec   38.87e-12 seconds/cell update
    #
    #   - block size:           0.647970 s   150.87e-12 seconds/cell update   ( 25.8% of peak)

    # GH = GeneralizedHarmonicAccel.GHCPU
    # GH = GeneralizedHarmonicAccel.GHParallelStencil
    GH = GeneralizedHarmonicAccel.GHCUDA1
    # GH = GeneralizedHarmonicAccel.GHCUDA2

    T = Float64   # Float32
    # Use these for benchmarking:
    ni, nj, nk = 512, 512, 512
    nsteps = 32
    # # Use these for ncu profiling:
    # ni, nj, nk = 64, 64, 64   # 128, 128, 128
    # nsteps = 0

    println("Setup...")
    step = 0
    state, state_prev, dt, dx, dy, dz = GH.setup(T, ni, nj, nk)

    println("Initialize...")
    GH.init!(state, state_prev, step, dt, dx, dy, dz)

    println("Step...")
    state_prev2 = similar(state)
    state, state_prev, state_prev2 = state_prev2, state, state_prev
    GH.step!(state, state_prev, state_prev2, dt, dx, dy, dz)

    println("Re-initialize...")
    GH.init!(state, state_prev, step, dt, dx, dy, dz)

    println("Evolve...")
    CUDA.synchronize()
    @time begin
        for _ in 1:nsteps
            state, state_prev, state_prev2 = state_prev2, state, state_prev
            step += 1
            GH.step!(state, state_prev, state_prev2, dt, dx, dy, dz)
        end
        CUDA.synchronize()
    end

    println("Done.")

    return nothing
end

bench()
