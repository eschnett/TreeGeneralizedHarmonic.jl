# The cube and the shells (step B5): the state on TreeMultiPatch's
# thirteen-patch ball, interpolated across its glues, with the high glued face
# planes in the state's appendix (TreeAMR's M20). `CODE.md`, "The cube and the
# shells".

# TreeMultiPatch and IMEXRungeKutta through the package's own bindings, as
# `stepping_tests.jl` takes the integrator: both are dependencies of the package
# and not of the test environment.
using KernelAbstractions: @kernel, @index
const TMP_B5 = TreeGeneralizedHarmonic.TreeMultiPatch
const IRK_B5 = TreeGeneralizedHarmonic.IRK

@kernel function ball_owner_kernel!(tid)
    I = @index(Global, NTuple)
    tid[I[end]] = Threads.threadid()
end

@testset verbose = true "The cube and the shells" begin
    T = Float64

    # Guards the integrator's partition on a state with an appendix: the
    # appendix follows every core, so a partition of the cores alone misses it
    # (IMEXRungeKutta refuses that) and one that gave it to a single thread
    # would combine every block's high planes off its owner. Each thread owns
    # its blocks' cores and their appendix, exactly once.
    @testset "the partition gives each block's appendix to its owner" begin
        layout = TMP_B5.ThirteenPatchBall{T}(1, 5 // 2, 4, (2, 1, 1))
        forest = TMP_B5.layout_forest(T, layout; N=8)
        fs = FieldSet{T}(forest, 20; G=3, centering=vertexcentered(3),
                         glue=:interpolate)
        u = statevector(fs)
        L = forest.N^3 * fs.nvars
        ncore = L * nblocks(fs)
        @test length(u) > ncore
        part = state_partition(fs, u)
        @test length(part) == Threads.nthreads()
        @test sort(reduce(vcat, [collect(r) for p in part for r in p])) == 1:length(u)

        # Every appendix entry of block `b` lies in the range of `b`'s thread:
        # write `b` into each and read the ranges back.
        a = statearray(u, fs; appendix=true)
        N = forest.N
        for b in 1:nblocks(fs), v in 1:fs.nvars,
            I in CartesianIndices((N + 1, N + 1, N + 1))
            any(==(N + 1), Tuple(I)) || continue
            isowned(a, Tuple(I)..., v, b) && (a[Tuple(I)..., v, b] = b)
        end
        chunks = TreeAMR.threadchunks(nblocks(fs))
        for (c, r) in enumerate(chunks)
            @test part[c][1] == (first(r) - 1) * L + 1:last(r) * L
            @test all(b -> b in r, u[part[c][2]])
        end
        tid = zeros(Int, nblocks(fs))
        map_blocks!(ball_owner_kernel!, fs, tid; appendix=false)
        offset = Threads.threadpoolsize(:interactive)
        if Threads.nthreads() > 1
            for (c, r) in enumerate(chunks), b in r
                @test tid[b] == offset + c
            end
        end

        # IMEXRungeKutta takes it, and its stage arithmetic by owner is the
        # broadcast's to the bit.
        u .= range(-1, 1; length=length(u))
        f!(du, u, p, t) = (du .= -u .+ sin(t); nothing)
        run(partition) = (integ = IRK_B5.init(IRK_B5.IMEXProblem(f!, nothing, copy(u),
                                                                 (zero(T), one(T)), nothing),
                                              IRK_B5.RK4(); dt=T(1 // 8),
                                              partition=partition);
                          IRK_B5.solve!(integ); integ.u)
        @test isequal(run(part), run(nothing))
    end
end
