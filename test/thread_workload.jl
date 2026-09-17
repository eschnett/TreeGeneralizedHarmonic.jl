# The workload behind the thread-count independence test (G3).
#
# Run as a standalone script —
#
#     julia -t N --project=. test/thread_workload.jl
#
# — it prints a digest of everything a full cycle of this package
# produces: the state vector at every chunk, the norms, both constraint
# monitors, the characteristic speed, and the mesh a regrid left behind.
# Two runs at different thread counts must print the same lines,
# **character for character**. That is stricter than "agrees to roundoff",
# and it is the property the code has: every parallel loop in TreeAMR
# writes its own slot, every combination of partials happens in block
# order, and this package adds no loop of its own that does otherwise
# (`CODE.md`, "Precision, threads, devices").
#
# It is a script rather than a testset because the thread count is a
# command-line argument to Julia and cannot be changed from inside a
# running session. Three rules keep it usable as one:
#
#   * **Nothing outside `Base` and the two packages.** No ODE package —
#     `rk4!` below is fifteen lines — so a subprocess starts in seconds
#     rather than compiling `OrdinaryDiffEq`; `hash` rather than a
#     cryptographic digest; `repr`, which round-trips a `Float64` exactly,
#     so that a difference in the last bit shows as different characters.
#   * **Everything that threads is on the path.** The initial-data cycle
#     with its flagging pass (`firing_boxes`, a reduction per block), the
#     ghost fill at a coarse-fine face, the fused right-hand side, a
#     regrid that moves data, `block_mapreduce` through `max_speed`, the
#     masked constraint norms and — from step 6 — the indicator's own
#     reference amplitude and its `τ_max`, and `volume_weighted_norm`.
#   * **No randomness and no timing.** Every number printed is a function
#     of the case and the mesh.
#
# What it would catch: a reduction partitioned by thread rather than by
# block, a host loop over blocks that accumulated into shared state, a
# flagging pass whose per-task lists were merged by anything but block
# order. Every one of those passes the rest of the suite.

using TreeAMR
using TreeGeneralizedHarmonic

digest(u::Vector{<:Real}) = string(hash(u); base=16, pad=16)
digest(s::AbstractString) = string(hash(s); base=16, pad=16)

# Plain fixed-step RK4, so the workload needs no ODE package. The same
# four stages `OrdinaryDiffEqLowOrderRK`'s `RK4()` takes, which is what
# the rest of the suite integrates with; what is being compared here is
# the *mesh's* determinism, and a stepper of our own keeps the subprocess
# cheap.
function rk4!(u, problem, t, dt, nsteps)
    k1, k2, k3, k4, tmp = (similar(u) for _ in 1:5)
    for _ in 1:nsteps
        gh_rhs!(k1, u, problem, t)
        @. tmp = u + (dt / 2) * k1
        gh_rhs!(k2, tmp, problem, t + dt / 2)
        @. tmp = u + (dt / 2) * k2
        gh_rhs!(k3, tmp, problem, t + dt / 2)
        @. tmp = u + dt * k3
        gh_rhs!(k4, tmp, problem, t + dt)
        @. u += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    end
    return u, t
end

"""
One adapt / evolve / regrid / evolve cycle on the gauge wave, reduced to
printed lines.

The gauge wave is the case that moves: its refined region has to follow
the wave, so the regrid in the middle really does transfer data between
levels rather than confirming the mesh it already had. `q = 4` and
`ε_KO = 1/2` are the package's development defaults, and the prolongation
order is `q + 2` because anything less costs the scheme an order
(`CODE.md`, "The interface-order rule") — the workload runs the
configuration a real run would.
"""
function gh_workload(; N=8, roots=2, q=4, chunks=2, steps=6, buffer=1,
                     threshold=1 // 40, refine_tol=1 // 2, coarsen_tol=1 // 8)
    T = Float64
    base = gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=T(1 // 2),
                           γ0=one(T), γ2=T(-1 // 2))
    # The real refinement parameters are attached to the case so that the
    # **indicator's own reductions** — the masked reference amplitude
    # (`block_mapreduce` per component), `τ_max`, the firing counts and the
    # centroid — are evaluated and digested at every chunk below. They are
    # the newest per-block folds in the package and they belong under the
    # bit-identity claim. No interior means no level floor and a periodic
    # box means no ceiling, so what they produce here is the indicator's
    # verdict and nothing else's.
    case = with_refinement(base,
                           Refinement(T; refine_tol=T(refine_tol),
                                      coarsen_tol=T(coarsen_tol),
                                      maxlevel_cap=1, floor_margin=zero(T),
                                      ceiling_cells=0))
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    G = q ÷ 2 + 1
    forest = gh_forest(T, case; N=N, roots=roots)
    fs = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))

    # **What drives the mesh here is not the indicator, and that is
    # deliberate (step 6).** One wavelength on a `2³` root brick puts a
    # crest in *every* block, so the indicator — which scores the crests and
    # nothing else — refines the whole domain uniformly, leaves no
    # coarse-fine face, and never changes the mesh again; a partial mesh
    # would need four root blocks per axis, which is four and a half times
    # the arithmetic in a workload that runs twice per suite. So the mesh
    # is still driven by the signed threshold below, whose slab travels
    # along `x` and drags the refinement with it, and the indicator is
    # *measured* at every chunk instead. What the digest then covers is
    # every reduction either of them makes.
    #
    # The threshold is **signed**, which picks one crest out of the two a
    # wavelength has: with `abs` the two crests refine symmetric halves of
    # the root grid and the mesh never changes again. `1/40` against an
    # amplitude of `1/20` puts the slab's leading edge one cell inside the
    # first block's `+x` face, so the buffer dilation reaches the second
    # block a few steps in and the regrid below really transfers data.
    thr = T(threshold)
    fires(work, idx, b, x) = work[idx..., 1, b] > thr
    function flags_now(f)
        return map(enumerate(firing_boxes(fires, f))) do (b, (n, box))
            k = f.forest.leaves[b]
            n == 0 && return level(k) > 0 ? Coarsen : Keep
            return level(k) ≥ 1 ? (Keep, box) : (Refine, box)
        end
    end

    schedule, passes, converged =
        adapt_to_initial_data!(fs, ops; initial=state_callback(case, zero(T)),
                               flags=flags_now, buffer=buffer, maxpasses=6)

    lines = String[]
    push!(lines, string("setup passes ", passes, " ", converged, " leaves ",
                        nleaves(forest), " ", digest(string(forest.leaves))))

    t = zero(T)
    for chunk in 1:chunks
        problem = GHProblem(fs, schedule, case; q=q, t=t)
        u = statevector(fs)
        gather!(u, fs)
        dt = gh_dt(problem, u; cfl=T(1 // 4))
        u, t = rk4!(u, problem, t, dt, steps)

        # The monitors, at the one place where a thread-dependent reduction
        # would show. The **gauge** constraint only: the ADM kernel is the
        # same `map_blocks!` launch reduced by the same `masked_norms` fold,
        # so it adds no parallel structure, and compiling it costs eighteen
        # seconds in each of the two processes this test runs (measured in
        # step 4). `constraints_tests.jl` is where it is checked.
        gh_constraint!(problem, u, t)
        gauge = constraint_norms(problem)
        scatter!(fs, u)
        push!(lines,
              string("chunk ", chunk, " t ", repr(t), " dt ", repr(dt),
                     " u ", digest(u),
                     " l2 ", repr(volume_weighted_norm(fs, u)),
                     " linf ", repr(volume_weighted_norm(fs, u; p=Inf)),
                     " λ ", repr(max_speed(problem)),
                     " C ", repr(maximum(gauge.gauge_l2)),
                     " ", repr(maximum(gauge.gauge_linf)),
                     " blocks ", nblocks(fs),
                     " schedule ", length(schedule.phase1), " ",
                     string(schedule.levels)))

        # A regrid that moves data, with the hook of *this* time. The state
        # field set is the only one transferred: `GHProblem` builds its own
        # `diag` (and, on a non-harmonic background, its own `Hsrc`) on
        # whatever forest it is handed, which is `CODE.md`'s
        # "`diag => nothing`, then re-sampled" spelled as a constructor.
        fill_ghosts!(fs, schedule)
        # The indicator on the same filled ghosts, digested but not acted
        # on: `τ_max` and the reference amplitude are `block_mapreduce`
        # folds, the firing counts come out of `firing_boxes`, and the
        # centroid and the marks are host passes in block order — every one
        # of them a place a thread-dependent reduction would show.
        ind = indicator_flags(fs, case, t; G=G, buffer=buffer)
        changed = regrid!(forest, fs => schedule; flags=flags_now(fs),
                          buffer=buffer, boundary=dirichlet(case, t))
        changed && (schedule = GhostSchedule(fs, ops))
        push!(lines, string("regrid ", chunk, " changed ", changed, " leaves ",
                            nleaves(forest), " ", digest(string(forest.leaves)),
                            " maxlevel ", maxlevel(forest),
                            " τ ", repr(ind.τ_max), " U_ref ", repr(ind.scale),
                            " firing ", ind.nfiring, " marks ",
                            digest(string(ind.flags)), " centroid ",
                            ind.centroid === nothing ? "none" :
                            digest(string(ind.centroid))))
    end

    # The state the last regrid *transferred*, digested after the fact.
    # Without this line the run ends on a regrid whose result nothing reads,
    # and a thread-dependent prolongation during the transfer would not
    # reach any printed number.
    u = statevector(fs)
    gather!(u, fs)
    push!(lines, string("final u ", digest(u), " l2 ",
                        repr(volume_weighted_norm(fs, u)), " linf ",
                        repr(volume_weighted_norm(fs, u; p=Inf)), " blocks ",
                        nblocks(fs)))
    return lines
end

"""
The digest lines the threading test compares, as a `Vector{String}`.

One workload, not two: unlike TreeWave's pair of cases this package has a
single right-hand side, and what varies between them would be the physics
rather than the parallel structure. The cycle above already visits every
loop that threads.
"""
thread_digests() = gh_workload()

abspath(PROGRAM_FILE) == abspath(@__FILE__) && foreach(println, thread_digests())
