# Excision: the `:excised` interior variant (added in step X2b).
#
# `CODE.md`, "Excision", and `PLAN.md`, step X2b. The claims, on the step-5
# fixture at `q = 2`, `N = 8` (`h = 5/64` at the hole) with the excision
# surface at `r_E = 3/4`: the per-point classes are the geometry's; the
# closures are exact where they claim to be; where no tap is excised the
# operator is today's, bit for bit; no kernel reads an excised value; the two
# geometries are one operator; the monitors read no excised value; the
# refusals fire, each by name; and a run goes, and restarts, as the run.
#
# From step X8 the fixture's excised set is shaved — the ball and its 216
# lego corners — and the step's claims are the testsets after the first: the
# shaved set is the rule's enumeration, the shave changes nothing beyond its
# points' reach (and nothing at all on a sphere with no corner), and every
# mask, the monitors and the horizon guard agree with the classes.
#
# The run file of the excision round, so it prices its claims: the fixture's
# problem is built once and shared by the testsets that only evaluate on it.

using Test
using TreeAMR
using TreeGeneralizedHarmonic
using Random: Xoshiro
using StaticArrays: SVector

const TGHx = TreeGeneralizedHarmonic

@testset verbose = true "Excision (step X2b)" begin
    T = Float64
    q = 2
    G = q ÷ 2 + 1
    N = 8
    ops = Operators(prolongation=q + 2, restriction=q + 2)
    r_E = T(3 // 4)
    case = hole_fixture(T; q=q, variant=:excised, r_1=r_E)
    forest = hole_fixture_forest(T, case; N=N)

    # A problem on the fixture's mesh with the case's initial data, and its
    # state vector. Its names are `local`: `U`, `p` and `u` are also this
    # file's own, and a closure that assigns a name of its enclosing scope
    # rebinds it — every `setup` until step X6 replaced the fixture's
    # problem with the one it built, or with a half-built field set where it
    # refused (no testset read them afterwards until X6's).
    function setup(c; interior=c.interior, perturb=nothing, simd_width=nothing)
        local U = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        fill_exact!(U, c, zero(T); interior=interior)
        perturb === nothing || perturb(U)
        local p = GHProblem(U, GhostSchedule(U, ops), c; q=q, interior=interior,
                            simd_width=simd_width)
        local u = statevector(U)
        gather!(u, U)
        return p, u
    end
    p, u = setup(case)
    ex = p.excision
    U = p.U
    du = similar(u)
    gh_rhs!(du, u, p, zero(T))
    A = statearray(du, U)
    owned = CartesianIndices((N, N, N))
    # The class of owned point `I` of block `b`.
    cls(ex, I, b) = ex.classes[I[1] + G, I[2] + G, I[3] + G, b]
    # The excised set as the classes hold it: from step X8 the geometry's with
    # its lego corners shaved, the problem's `evolved_mask` (the fixture's
    # sphere `r < 3/4` has 216 corners, 27 in each octant).
    mask = evolved_mask(p, zero(T))
    @test mask isa ShavedMask && ex.nshaved == 216

    @testset "the classes are the geometry's, the ghosts their owners', the zone the enumeration's" begin
        # Guards the single source of truth. A class array that disagreed with
        # the masks' predicate anywhere — an owned point, a ghost the exchange
        # filled, an outer face the hook filled — would make the kernels and
        # the norms exclude different sets; a zone narrower than the
        # enumeration would leave a centered stencil reading an excised value,
        # one wider than it would put a closure where today's operator runs.
        nbad = 0
        for b in 1:nblocks(U), S in CartesianIndices(size(ex.classes)[1:3])
            x = coordinates(U, b, Tuple(S))
            want = !is_evolved(mask, x)
            nbad += (ex.classes[S, b] == TGHx.CLASS_EXCISED) != want
        end
        @test nbad == 0
        h = minimum_spacing(T, forest)
        tapped(x) = any(!is_evolved(mask, x .+ off) for off in taps)
        taps = NTuple{3,Float64}[]
        for d in 1:3, a in (-G):G
            a == 0 || push!(taps, ntuple(k -> k == d ? a * h : 0.0, 3))
        end
        for (i, j) in ((1, 2), (1, 3), (2, 3)), a in -(q ÷ 2):(q ÷ 2),
            e in -(q ÷ 2):(q ÷ 2)
            push!(taps, ntuple(k -> k == i ? a * h : k == j ? e * h : 0.0, 3))
        end
        nzone = 0
        nexc = 0
        wrong = 0
        for b in 1:nblocks(U), I in owned
            x = coordinates(U, b, Tuple(I) .+ G)
            c = cls(ex, I, b)
            if !is_evolved(mask, x)
                nexc += 1
            else
                z = tapped(x)
                nzone += z
                wrong += z != (c == TGHx.CLASS_ZONE)
            end
        end
        @test wrong == 0
        @test (ex.nzone, ex.nexcised) == (nzone, nexc)
        @test ex.nzone + ex.nexcised + ex.ncentered == nblocks(U) * N^3
        # The numbers on the record: a ball of radius 3/4 at h = 5/64 with
        # its 216 corners shaved (step X8; without the shave 3743, 2192, 32,
        # the testset below), and its band; the zone lives in 32 of the 64
        # fine blocks.
        @test (ex.nexcised, ex.nzone, ex.nzoneblocks) == (3959, 2312, 32)
        @test ex.W == 2 * h && ex.h == h
        @test count(Array(ex.zoneblocks)) == ex.nzoneblocks
    end

    # --- step X8: the lego corners shaved (`CODE.md`, "Excision", "What step
    # X8 built"). A point the geometry leaves outside is excised too when its
    # three neighbours one step toward the center are inside it; one pass,
    # decided once with the classes from a predicate every mask evaluates. The
    # fixture's problem above is shaved (the default); `p0` is the same
    # fixture without the shave, steps X2b–X7's excised set.
    p0, u0 = setup(hole_fixture(T; q=q, variant=:excised, r_1=r_E,
                                excision=Excision(T; shave=false)))
    ex0 = p0.excision
    # The fixture's level-3 lattice is `x = k h`, `h = 5/64`, about the center:
    # inside the sphere `r < 3/4` exactly when `400 |k|² < 36864`, in integers.
    inball(k) = 400 * (k[1]^2 + k[2]^2 + k[3]^2) < 36864
    lattice(x) = ntuple(d -> round(Int, x[d] / ex.h), 3)
    inward(k, d) = Base.setindex(k, k[d] - sign(k[d]), d)
    corner(E, k) = !E(k) && all(d -> k[d] != 0 && E(inward(k, d)), 1:3)

    @testset "the shaved set is the rule's enumeration on the host, one pass, and without the shave the excised set is X2b's" begin
        # Guards the shave's definition (step X8). A shaved set that were not
        # the rule — an inward neighbour taken on the wrong side, a center
        # plane shaved, a neighbour read on another lattice, the predicate
        # iterated — would excise other points than the ones step X7's scratch
        # copy cured; a build without the shave that were not steps
        # X2b–X7's would not be the comparison the switch exists for. At every
        # owned point, from the integer lattice alone: the shaved classes'
        # excised set is the ball and its corners, and the unshaved one the
        # ball. One pass: the corners of the shaved set itself — a point whose
        # three inward neighbours are excised, one of them shaved — are left
        # (iterated, the rule would shave them and their neighbours on).
        nsh = 0
        bad = 0
        bad0 = 0
        second = 0
        shavedset = Set{NTuple{3,Int}}()
        for b in 1:nblocks(U), I in owned
            k = lattice(coordinates(U, b, Tuple(I) .+ G))
            sh = corner(inball, k)
            sh && push!(shavedset, k)
            nsh += sh
            bad += (cls(ex, I, b) == TGHx.CLASS_EXCISED) != (inball(k) || sh)
            bad0 += (cls(ex0, I, b) == TGHx.CLASS_EXCISED) != inball(k)
        end
        for b in 1:nblocks(U), I in owned
            k = lattice(coordinates(U, b, Tuple(I) .+ G))
            second += corner(k -> inball(k) || k in shavedset, k)
        end
        @info "the fixture's shave: $nsh corners shaved; $second corners of the " *
              "shaved set left by the one pass"
        @test bad == 0 && bad0 == 0
        @test nsh == ex.nshaved == 216 && ex0.nshaved == 0
        @test second > 0
        @test (ex0.nexcised, ex0.nzone, ex0.nzoneblocks) == (3743, 2192, 32)
        @test ex.nexcised == ex0.nexcised + ex.nshaved
        @test evolved_mask(p0, zero(T)) isa InteriorMask
        # Every shaved point lies within `h/√3` of the sphere (step X8's bound).
        @test all(k -> sqrt(sum(abs2, k)) * ex.h - r_E < ex.h / sqrt(3), shavedset)
    end

    @testset "the shave changes nothing beyond the shaved points' reach, and nothing at all where it shaves nothing" begin
        # Guards "the exterior is unchanged" for the shave (step X8). Where it
        # shaves, the classes change at the shaved points (excised for zone)
        # and at the points whose stencils read one (zone for centered); every
        # other point keeps its class and — its stencils' reads of the classes
        # reaching `G` along the axes and the `G`-box in each plane — its `du`
        # bit for bit. And a sphere with no corner: every lattice sphere of
        # 3.6 cells or more has some (enumerated from 1 to 40 cells in steps
        # of 0.01), so the claim is made on one of 3.5 cells, `r_E = 35/128`,
        # where the shaved and the unshaved problem must be one problem — the
        # classes, the codes, `du`, the masks and every monitor's numbers.
        du0 = similar(u)
        gh_rhs!(du0, u, p0, zero(T))
        A0 = statearray(du0, U)
        reach = NTuple{3,Int}[]
        for d in 1:3, a in (-G):G
            a == 0 || push!(reach, ntuple(l -> l == d ? a : 0, 3))
        end
        for (i, j) in ((1, 2), (1, 3), (2, 3)), a in (-G):G, e in (-G):G
            push!(reach, ntuple(l -> l == i ? a : l == j ? e : 0, 3))
        end
        shaved_k = Set(lattice(coordinates(U, b, Tuple(I) .+ G))
                       for b in 1:nblocks(U), I in owned
                       if cls(ex, I, b) == TGHx.CLASS_EXCISED &&
                          cls(ex0, I, b) != TGHx.CLASS_EXCISED)
        @test length(shaved_k) == ex.nshaved
        nfar = 0
        same = 0
        clsame = 0
        nnear = 0
        changed = 0
        for b in 1:nblocks(U), I in owned
            (cls(ex, I, b) == TGHx.CLASS_EXCISED || cls(ex0, I, b) == TGHx.CLASS_EXCISED) &&
                continue
            k = lattice(coordinates(U, b, Tuple(I) .+ G))
            if any(o -> (k .+ o) in shaved_k, reach)
                nnear += 1
                changed += any(v -> !isequal(A[I, v, b], A0[I, v, b]), 1:20)
            else
                nfar += 1
                clsame += cls(ex, I, b) == cls(ex0, I, b)
                same += all(v -> isequal(A[I, v, b], A0[I, v, b]), 1:20)
            end
        end
        @info "the shave on the fixture: $same of $nfar points beyond the shaved " *
              "points' reach bit for bit; $changed of $nnear within it changed"
        @test clsame == nfar && same == nfar && nfar > 50_000
        @test changed > 0
        # The corner-free sphere.
        tiny(sh) = hole_fixture(T; q=q, variant=:excised, r_0=T(1 // 8),
                                r_1=T(35 // 128), excision=Excision(T; shave=sh))
        (ps, us), (pn, un) = setup(tiny(true)), setup(tiny(false))
        @test ps.excision.nshaved == 0 && ps.excision.nexcised == pn.excision.nexcised > 0
        @test ps.excision.classes == pn.excision.classes && ps.excision.codes == pn.excision.codes
        @test evolved_mask(ps, zero(T)) == evolved_mask(pn, zero(T))
        @test monitor_mask(ps, zero(T)) == monitor_mask(pn, zero(T))
        dus, dun = similar(us), similar(un)
        gh_rhs!(dus, us, ps, zero(T))
        gh_rhs!(dun, un, pn, zero(T))
        @test isequal(dus, dun)
        rows(pp, uu) = (gh_constraint!(pp, uu, zero(T)); adm_constraint!(pp, uu, zero(T));
                        gh_error!(pp, uu, zero(T)); (constraint_norms(pp), error_norms(pp),
                                                     validity_rows(pp, uu, zero(T)),
                                                     excision_rows(pp, uu, zero(T))))
        @test isequal(rows(ps, us), rows(pn, un))
    end

    @testset "the masks agree with the classes point by point, the monitors read no shaved point, and the horizon guard refuses one" begin
        # Guards "everything that asks 'is it excised' agrees with the
        # classes" (step X8). The shave is a predicate every mask evaluates
        # (`ShavedMask`), so: at every stored point — the ghosts the exchange
        # filled included — the problem's mask says excised exactly where the
        # class does; the monitors' mask, widened by `W + h` where the shave
        # excised a point, leaves no stencil of theirs (the axes to `q/2` and
        # the plane boxes of half-width `q/2`) reading an excised point; the
        # validity monitor's band holds no shaved point; the noise exclusion's
        # mask is the problem's; and the horizon finder's guard refuses a
        # footprint whose only excised point is a shaved one, which the
        # sphere's own mask would have read.
        em = evolved_mask(p, zero(T))
        bad = 0
        for b in 1:nblocks(U), S in CartesianIndices(size(ex.classes)[1:3])
            x = coordinates(U, b, Tuple(S))
            bad += (!is_evolved(em, x)) != (ex.classes[S, b] == TGHx.CLASS_EXCISED)
        end
        @test bad == 0
        mm = monitor_mask(p, zero(T))
        @test mm == excision_monitor_mask(p.interior, zero(T), ex.W + ex.h)
        excised_k = Set(lattice(coordinates(U, b, Tuple(I) .+ G))
                        for b in 1:nblocks(U), I in owned
                        if cls(ex, I, b) == TGHx.CLASS_EXCISED)
        taps = NTuple{3,Int}[]
        r = q ÷ 2
        for d in 1:3, a in (-r):r
            push!(taps, ntuple(l -> l == d ? a : 0, 3))
        end
        for (i, j) in ((1, 2), (1, 3), (2, 3)), a in (-r):r, e in (-r):r
            push!(taps, ntuple(l -> l == i ? a : l == j ? e : 0, 3))
        end
        nmon = 0
        reads = 0
        inband = 0
        lmask, _ = TGHx._validity_bands(ex, p.interior, zero(T), ex.W)
        for b in 1:nblocks(U), I in owned
            x = coordinates(U, b, Tuple(I) .+ G)
            k = lattice(x)
            if is_evolved(mm, x)
                nmon += 1
                reads += any(o -> (k .+ o) in excised_k, taps)
            end
            inband += is_evolved(lmask, x) && cls(ex, I, b) == TGHx.CLASS_EXCISED
        end
        @test nmon > 0 && reads == 0 && inband == 0
        @test excised_mask(p.interior, zero(T), ex.h) == em
        # The guard: a corner `P` with positive coordinates, and a query whose
        # `(q + 2)³` window runs from `P` outward — no point of the sphere in
        # it, only `P`.
        TGHx.scatter!(U, u)
        fill_ghosts!(U, p.schedule; boundary=dirichlet(case, zero(T)))
        k = first(k for k in sort!(collect(excised_k)) if all(>(0), k) && !inball(k))
        xq = [SVector{3,T}((k .+ T(3 // 2)) .* ex.h)]
        @test_throws ArgumentError gh_interpolate(U, xq; q=q, mask=em)
        @test all(isfinite, first(gh_interpolate(U, xq; q=q,
                                                 mask=interior_mask(case.interior,
                                                                    zero(T)))))
        msg = try
            gh_interpolate(U, xq; q=q, mask=em)
            ""
        catch err
            sprint(showerror, err)
        end
        @test occursin("lego corners shaved", msg)
    end

    @testset "the closures are exact on polynomials across faces, edges and corners" begin
        # Guards the closure provider's indexing — a table read at the wrong
        # `[slot, k⁻ + 1, k⁺ + 1]`, a node range off by one, a mixed derivative
        # whose inner closure is the outer point's rather than the outer
        # node's. On one block of a host working array with an excised
        # half-space — `i ≤ c` (faces), `i + j ≤ c` (edges: two axis neighbours
        # excised), `i + j + k ≤ c` (corners: three) — every evolved point
        # within reach of it evaluates, on a polynomial of the degree every
        # closure is exact to (`q/2 + 1`), `d1`, `d2`, the nested `dmix` and the
        # lopsided `adv` exactly, and `ko` annihilates the degree below `G`
        # that `:msn` keeps near the surface.
        for qq in (2, 4)
            GG = qq ÷ 2 + 1
            n = 4GG + 5
            deg = qq ÷ 2 + 1
            rng = Xoshiro(qq)
            mons = [(a, b, c) for a in 0:deg for b in 0:deg for c in 0:deg
                    if a + b + c ≤ deg]
            coef = [T(rand(rng, -9:9)) / 8 for _ in mons]
            # `∂^e P` exactly, `e` the derivative count per axis.
            ff(a, e) = e == 0 ? 1 : e == 1 ? a : a * (a - 1)
            dpoly(x, e) = sum(coef[m] * prod(ff(p[k], e[k]) == 0 ? 0.0 :
                                             ff(p[k], e[k]) * x[k]^(p[k] - e[k])
                                             for k in 1:3)
                              for (m, p) in enumerate(mons))
            P(x) = dpoly(x, (0, 0, 0))
            unit(d, k=1) = ntuple(l -> l == d ? k : 0, 3)
            # A polynomial of degree below `G`, which `:msn` annihilates.
            linp(x) = 1.5 - 0.25 * x[1] + 0.75 * x[2] - 0.5 * x[3] +
                     (GG > 2 ? 0.125 * x[1] * x[2] - 0.375 * x[3]^2 : 0.0)
            tab = TGHx.closure_arrays(T, Val(qq), :msn, TreeGeneralizedHarmonic.CPU())
            for shape in (:face, :edge, :corner)
                c0 = 2GG + 3 + (shape === :face ? 0 : shape === :edge ? n ÷ 2 : n)
                excised(I) = shape === :face ? I[1] ≤ c0 :
                             shape === :edge ? I[1] + I[2] ≤ c0 : I[1] + I[2] + I[3] ≤ c0
                work = Array{T}(undef, n, n, n, 20, 1)
                cl = Array{UInt8}(undef, n, n, n, 1)
                for I in CartesianIndices((n, n, n))
                    x = T.(Tuple(I))
                    for v in 1:20
                        work[I, v, 1] = v == 1 ? P(x) : v == 2 ? linp(x) : T(NaN)
                    end
                    cl[I, 1] = excised(Tuple(I)) ? TGHx.CLASS_EXCISED : TGHx.CLASS_ZONE
                    excised(Tuple(I)) && (work[I, 1, 1] = work[I, 2, 1] = T(NaN))
                end
                wst, wsv, _ = TGHx.work_strides(work)
                npts = 0
                worst = 0.0
                kos = 0.0
                for I in CartesianIndices((n, n, n))
                    t = Tuple(I)
                    (all(d -> GG < t[d] ≤ n - GG, 1:3) && !excised(t)) || continue
                    # Within reach of the excised set along some axis.
                    any(excised(Base.setindex(t, t[d] + s * a, d))
                        for d in 1:3, s in (-1, 1), a in 1:GG) || continue
                    base = 1 + (t[1] - 1) * wst[1] + (t[2] - 1) * wst[2] +
                           (t[3] - 1) * wst[3]
                    S = TGHx.closure_provider(T, Val(GG), wst, cl, base, tab, one(T),
                                              one(T))
                    x = T.(t)
                    for d in 1:3
                        worst = max(worst, abs(TGHx.d1(S, work, base, d) -
                                               dpoly(x, unit(d))))
                        worst = max(worst, abs(TGHx.d2(S, work, base, d) -
                                               dpoly(x, unit(d, 2))))
                        for β_d in (-one(T), one(T))
                            worst = max(worst, abs(TGHx.adv(S, β_d, zero(T), work,
                                                            base, d) -
                                                   dpoly(x, unit(d))))
                        end
                        kos = max(kos, abs(TGHx.ko(S, work, base + wsv, d)))
                    end
                    # The mixed derivative symmetric in its two axes (the
                    # default from step X6) and nested along the lower axis
                    # (steps X2b–X5, `mixed = :nested`): both exact.
                    Sn = TGHx.closure_provider(T, Val(GG), wst, cl, base, tab, one(T),
                                               one(T), Val(false))
                    for (i, j) in ((1, 2), (1, 3), (2, 3))
                        worst = max(worst, abs(TGHx.dmix(S, work, base, i, j) -
                                               dpoly(x, unit(i) .+ unit(j))))
                        worst = max(worst, abs(TGHx.dmix(Sn, work, base, i, j) -
                                               dpoly(x, unit(i) .+ unit(j))))
                    end
                    npts += 1
                end
                scale = maximum(abs, filter(isfinite, work[:, :, :, 1:2, 1]))
                @test npts > 0
                @test worst ≤ 1e-12 * scale
                @test kos ≤ 1e-12 * scale
            end
        end
    end

    # The working array of `p` with the ghosts of `u` filled, and a point's
    # linear indices into it and into the class array.
    TGHx.scatter!(U, u)
    fill_ghosts!(U, p.schedule; boundary=dirichlet(case, zero(T)))
    st, sv, sb = TGHx.work_strides(U.work)
    lin(I, b) = (1 + (b - 1) * sb + (I[1] + G - 1) * st[1] + (I[2] + G - 1) * st[2] +
                 (I[3] + G - 1) * st[3],
                 1 + (b - 1) * sv + (I[1] + G - 1) * st[1] + (I[2] + G - 1) * st[2] +
                 (I[3] + G - 1) * st[3])

    @testset "with no excised tap the closure provider is the centered one" begin
        # Guards "the exterior is unchanged bit for bit" at the provider: the
        # closure table's centered rows contracted over their nodes in
        # ascending order must be `axis_stencil`'s and `mixed_stencil`'s
        # arithmetic, so that a centered point reached by the closure provider
        # — the inner closures of a nested mixed derivative, a zone point's
        # far side — is today's. Each contraction is claimed `isequal`. The
        # whole `F` is one body compiled for two providers (`CLAUDE.md`, "Two
        # spellings of one expression"): it was `isequal` on Apple silicon
        # until step X4, and since `main`'s head (merged in step X4) the two
        # heads — one around the closure provider's loops over its table, one
        # around the centered provider's unrolled stencils — fuse
        # `metric_quantities`' StaticArrays `muladd`s differently (`β` by 2.8
        # eps, `A^{ij}` by 6.8). In a stationary background `F` is the small
        # difference of `O(1)` terms, so that is up to 238 eps of `F`'s own
        # largest value at these points: the whole `F` is held to roundoff as
        # the suite's other comparisons of two specialisations are, 512 eps of
        # each variable's largest `|du|` on the mesh (measured: 103, at 1682
        # of the 4508 points; amended in step X4).
        C = TGHx.Centered(T, Val(q), st)
        nc = 0
        each = true
        bitwise = true
        nbitwise = 0
        dmax = zeros(20)
        for b in 1:nblocks(U), I in owned
            cls(ex, I, b) == TGHx.CLASS_CENTERED || continue
            # Centered points next to the band: their taps are evolved points
            # with codes below `G` on one side, which the closure provider
            # reads through its own tables.
            x = coordinates(U, b, Tuple(I) .+ G)
            sqrt(sum(abs2, x)) < r_E + 4 * ex.h || continue
            var, cb = lin(I, b)
            S = TGHx.closure_provider(T, Val(G), st, ex.classes, cb, ex.table,
                                      inv(spacing(T, forest, forest.leaves[b])),
                                      zero(T))
            for v in (1, 6, 11, 18), d in 1:3
                bv = var + (v - 1) * sv
                each &= isequal(TGHx.d1(S, U.work, bv, d), TGHx.d1(C, U.work, bv, d))
                each &= isequal(TGHx.d2(S, U.work, bv, d), TGHx.d2(C, U.work, bv, d))
                each &= isequal(TGHx.ko(S, U.work, bv, d), TGHx.ko(C, U.work, bv, d))
            end
            for v in (1, 6), (i, j) in ((1, 2), (1, 3), (2, 3))
                bv = var + (v - 1) * sv
                each &= isequal(TGHx.dmix(S, U.work, bv, i, j),
                                TGHx.dmix(C, U.work, bv, i, j))
            end
            inv_h = inv(spacing(T, forest, forest.leaves[b]))
            γ0 = damping_rate(case.γ0, zero(T), x)
            εh = dissipation_rate(case.ε_KO, zero(T), x) * inv_h
            Hw = TGHx.gauge_work(p.Hsrc)
            F1 = TGHx.gh_rhs_at_point(S, T, U.work, Hw, Tuple(I), b, var, sv, inv_h,
                                      γ0, case.γ2, εh, Val(true), Val(true))
            F0 = TGHx.gh_rhs_at_point(T, U.work, Hw, Tuple(I), b, var, st, sv, inv_h,
                                      γ0, case.γ2, εh, Val(q), Val(true), Val(true))
            bitwise &= isequal(F1, F0)
            nbitwise += isequal(F1, F0)
            dmax .= max.(dmax, abs.(vcat(F1[1], F1[2]) .- vcat(F0[1], F0[2])))
            nc += 1
        end
        vscale = [maximum(abs, A[:, :, :, v, :]) for v in 1:20]
        worst = maximum(dmax ./ vscale)
        @info "the closure provider's F at $nc centered points next to the band: " *
              (bitwise ? "bit for bit" :
               "$nbitwise bit for bit, the rest within $(worst / eps(T)) eps of " *
               "each variable's largest |du|") * " the centered one's"
        @test nc > 1000
        @test each
        @test bitwise || worst ≤ 512 * eps(T)
    end

    @testset "no kernel reads an excised value" begin
        # Guards `0 · NaN = NaN` and every way of reading the excised set: a
        # stencil with a zero weight on it, a closure that reaches past its
        # codes, a ghost an exchange copied from an excised point. A
        # degenerate metric (`h = −η`, so `g = 0`, and `Π = NaN`) planted on
        # every excised owned point must leave every non-excised `du` bit for
        # bit the clean state's, and every excised `du` exactly zero.
        u2 = copy(u)
        S2 = statearray(u2, U)
        η = (1, 0, 0, 0, -1, 0, 0, -1, 0, -1)
        for b in 1:nblocks(U), I in owned
            cls(ex, I, b) == TGHx.CLASS_EXCISED || continue
            for v in 1:10
                S2[I, v, b] = η[v]
                S2[I, 10 + v, b] = T(NaN)
            end
        end
        du2 = similar(u)
        gh_rhs!(du2, u2, p, zero(T))
        B = statearray(du2, U)
        same = 0
        zero_ = 0
        for b in 1:nblocks(U), I in owned
            if cls(ex, I, b) == TGHx.CLASS_EXCISED
                zero_ += all(v -> A[I, v, b] === zero(T) && B[I, v, b] === zero(T), 1:20)
            else
                same += all(v -> isequal(A[I, v, b], B[I, v, b]), 1:20)
            end
        end
        @test same == ex.nzone + ex.ncentered
        @test zero_ == ex.nexcised
    end

    @testset "the centered points' du is the unexcised operator's, and the zone points' is not" begin
        # Guards the main kernel's `:excised` branch: at a centered point it
        # must be today's `F` — the `:none` kernel's, on the same state — and
        # at a zone point the zone kernel's closures, which differ from the
        # centered operator reading the core rule's data by far more than
        # roundoff. To `512 eps` (two kernel specialisations, `CLAUDE.md`).
        none = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10),
                                interior=nothing)
        Un = FieldSet{T}(forest, 20; G=G, centering=vertexcentered(3))
        pn = GHProblem(Un, GhostSchedule(Un, ops), none; q=q)
        dun = similar(u)
        gh_rhs!(dun, u, pn, zero(T))
        C = statearray(dun, Un)
        worst = 0.0
        scale = 0.0
        nbit = 0
        zmin = Inf
        for b in 1:nblocks(U), I in owned
            c = cls(ex, I, b)
            d = maximum(v -> abs(A[I, v, b] - C[I, v, b]), 1:20)
            if c == TGHx.CLASS_CENTERED
                worst = max(worst, d)
                scale = max(scale, maximum(v -> abs(A[I, v, b]), 1:20))
                nbit += all(v -> isequal(A[I, v, b], C[I, v, b]), 1:20)
            elseif c == TGHx.CLASS_ZONE
                zmin = min(zmin, d)
            end
        end
        @info "centered points: $nbit of $(ex.ncentered) bit for bit the :none " *
              "kernel's, worst $(worst / (eps(T) * scale)) eps; zone points differ " *
              "by at least $zmin"
        @test worst ≤ 512 * eps(T) * scale
        @test zmin > 1e6 * 512 * eps(T) * scale
    end

    @testset "the excised right-hand side runs scalar on the CPU, whatever the problem's SIMD width" begin
        # Guards the routing of the merge with `main`'s SIMD lanes (proposed
        # in the main merge, 2026-10-08): an `:excised` problem keeps the
        # host's lane width like any other, and the kernel must still give
        # every point of it the scalar path, `W = 1`'s. Sent down the lanes,
        # a group that straddles the surface would load excised values into
        # its lanes and the `:none` branch's operator onto zone points, and
        # every centered group would differ from the scalar code in the last
        # bits (StaticArrays' `muladd`s fuse differently on lanes,
        # `test/simd_tests.jl`: 1–207 eps of each variable's largest `|du|`
        # here). So the claim is the `W = 1` problem's `du` — bit for bit on
        # the development machine, and held to 64 eps of each variable's
        # largest `|du|` where two specialisations of one body round apart
        # (`CLAUDE.md`, "Two spellings of one expression").
        @test p.valsimd === Val(default_simd_width(T, TGHx.CPU(), N))
        @test p.valsimd !== Val(1)
        p1, u1 = setup(case; simd_width=1)
        @test p1.valsimd === Val(1) && isequal(u1, u)
        @test p1.excision.classes == ex.classes
        du1 = fill!(similar(u), T(NaN))
        gh_rhs!(du1, u, p1, zero(T))
        A1 = statearray(du1, p1.U)
        vscale = [maximum(abs, A[:, :, :, v, :]) for v in 1:20]
        worst = 0.0
        zero_ = 0
        for b in 1:nblocks(U), I in owned
            if cls(ex, I, b) == TGHx.CLASS_EXCISED
                zero_ += all(v -> A[I, v, b] === zero(T) && A1[I, v, b] === zero(T), 1:20)
            else
                worst = max(worst, maximum(v -> abs(A[I, v, b] - A1[I, v, b]) / vscale[v],
                                           1:20))
            end
        end
        bitwise = isequal(du1, du)
        @info "the excised kernel at W = $(typeof(p.valsimd).parameters[1]) against " *
              "W = 1: " * (bitwise ? "bit for bit" :
                           "within $(worst / eps(T)) eps of each variable's largest |du|")
        @test !any(isnan, du1)
        @test zero_ == ex.nexcised
        @test bitwise || worst ≤ 64 * eps(T)
    end

    @testset "an excised sphere and a FittedInterior holding it are one operator" begin
        # Guards the two geometries' protocol: the tracked geometry's excised
        # set is `d > 0` below its offset surface, the sphere's `r < r_1`, and
        # a `FittedInterior` whose surface *is* that sphere (a degree-0 shape
        # with `r_in = r_out = 2`, offset `2 − 3/4`) must give the same classes
        # and the same `du` — `isequal` where the platform gives it.
        fint = FittedInterior(T; center=(0, 0, 0),
                              shape=SVector(2 * sqrt(4 * T(π))), lmax=0,
                              offset=2 - r_E, thickness=r_E - T(2 // 5),
                              variant=:excised, margin=8,
                              h=minimum_spacing(T, forest), r_in=2, r_out=2)
        pf, uf = setup(case; interior=fint)
        @test isequal(uf, u)
        @test pf.excision.classes == ex.classes
        @test (pf.excision.nzone, pf.excision.nexcised) == (ex.nzone, ex.nexcised)
        duf = similar(u)
        gh_rhs!(duf, u, pf, zero(T))
        bitwise = isequal(duf, du)
        @info "the tracked geometry's du is " *
              (bitwise ? "bit for bit" : "$(maximum(abs, duf - du) / (eps(T) * maximum(abs, du))) eps of") *
              " the sphere's"
        @test bitwise || maximum(abs, duf - du) ≤ 64 * eps(T) * maximum(abs, du)
    end

    @testset "the lopsided blend is off above its shell, and one operator in both kernels" begin
        # Guards the blend's two promises (`CODE.md`, "Excision"): beyond the
        # depth where it starts it is not there at all — `adv` returns its
        # argument through a branch, so the exterior is today's operator bit
        # for bit — and the main kernel's `Lopsided` and the zone kernel's
        # closure provider are one lopsided derivative, the table's centered
        # row being `lopsided_centered_weights` bit for bit. With the blend
        # from one cell below the horizon, full at five: every point above
        # `r_h − h` keeps the blend-free `du`, and the points below it where
        # the blend's weight is at least a hundredth do not.
        up = hole_fixture(T; q=q, variant=:excised, r_1=r_E,
                          excision=Excision(T; upwind=(1, 4)))
        pu, _ = setup(up)
        @test pu.excision.classes == ex.classes
        duu = similar(u)
        gh_rhs!(duu, u, pu, zero(T))
        B = statearray(duu, U)
        bl = pu.excision.blend
        above = 0
        same = 0
        below = 0
        moved = 0
        zmoved = 0
        for b in 1:nblocks(U), I in owned
            c = cls(ex, I, b)
            c == TGHx.CLASS_EXCISED && continue
            x = coordinates(U, b, Tuple(I) .+ G)
            λ = blend_weight(bl, x)
            if λ == 0
                above += 1
                same += all(v -> isequal(A[I, v, b], B[I, v, b]), 1:20)
            elseif c == TGHx.CLASS_ZONE
                zmoved += any(v -> A[I, v, b] != B[I, v, b], 1:20)
            elseif λ ≥ 1 // 100
                # (Just inside the start the blend's weight is `O(s³)` and can
                # round away; a hundredth of the lopsided derivative cannot.)
                below += 1
                moved += any(v -> A[I, v, b] != B[I, v, b], 1:20)
            end
        end
        @test same == above && above > 0
        @test moved == below && below > ex.nzone
        # At `q = 2` the lopsided derivative of a first evolved point *is* its
        # one-sided closure (`lopsided_weights` starts at `−k_down = 0`), so a
        # zone point whose advected axes are all such faces does not move:
        # 194 of the 2312 here (282 of 2192 without the shave, until step X8).
        # The rest do — the zone kernel blends too.
        @test zmoved == ex.nzone - 194
        # The two providers' lopsided derivative at a centered point, at full
        # weight, and at zero weight the argument itself.
        I, b = first((I, b) for b in 1:nblocks(U), I in owned
                     if cls(ex, I, b) == TGHx.CLASS_CENTERED &&
                        sqrt(sum(abs2, coordinates(U, b, Tuple(I) .+ G))) < 1)
        var, cb = lin(I, b)
        ih = inv(spacing(T, forest, forest.leaves[b]))
        Lp = TGHx.Lopsided(T, Val(q), st, ih, one(T))
        Cp = TGHx.closure_provider(T, Val(G), st, ex.classes, cb, ex.table, ih, one(T))
        agree = all(isequal(TGHx.adv(Lp, β, zero(T), U.work, var + (v - 1) * sv, d),
                            TGHx.adv(Cp, β, zero(T), U.work, var + (v - 1) * sv, d))
                    for v in (1, 7, 13, 20), d in 1:3, β in (-one(T), one(T)))
        @test agree
        L0 = TGHx.Lopsided(T, Val(q), st, ih, zero(T))
        @test isequal(TGHx.adv(L0, one(T), T(NaN), U.work, var, 1), T(NaN))
    end

    @testset "the tracked geometry is frozen, and the found horizon stays m h outside it" begin
        # Guards the driver's tracked `:excised` path: the geometry built once
        # from the seed (no rebuild per chunk, none on a find), the finder run
        # every chunk for the record, and the assertion's row — the found
        # horizon's least distance from the frozen surface, `m` cells for the
        # seed. `m = 16` puts the surface at `3/4`, inside the fixture's fine
        # cube with its `(G + q + 2) h`; `n_L = 4` is the core rule's depth.
        tracked = kerr_schild_case(T; halfwidth=T(5 // 2), chunk=T(1 // 10),
                                   interior=FittedSpec(T; variant=:excised,
                                                       margin=16, n_L=4),
                                   horizon=Horizon(T; every=1, N=12, spin=false))
        out = evolve!(T, tracked; forest=deepcopy(forest), q=q, ops=ops,
                      t_end=T(1 // 10))
        @test out.geometry isa FittedInterior{T,:excised}
        @test out.problem.excision.interior === out.geometry
        @test all(r -> r.finite && r.horizon_success === true, out.records)
        @test all(r -> 15.9 < r.excision_horizon_margin < 16.1, out.records)
        @test all(r -> r.excision_into == 0 && r.excision_normal_min > 1, out.records)
        @test abs(out.problem.excision.nzone - ex.nzone) ≤ ex.nzone ÷ 50
    end

    @testset "the monitors read no excised value, and the horizon guard refuses one" begin
        # Guards `monitor_mask` and the masks the pointwise monitors keep:
        # with `NaN` in every excised owned point the gauge and ADM
        # constraints (which take stencils, masked to the excised set widened
        # by `W`), the error, the speed, the validity monitor, the outflow
        # monitor and the non-finite count (which read points, masked by the
        # interior's own mask, the band included) must all be finite — and
        # TreeAMR's interpolation, through the horizon finder's guard, must
        # refuse a footprint that reaches an excised point.
        u3 = copy(u)
        S3 = statearray(u3, U)
        for b in 1:nblocks(U), I in owned
            cls(ex, I, b) == TGHx.CLASS_EXCISED || continue
            for v in 1:20
                S3[I, v, b] = T(NaN)
            end
        end
        gh_constraint!(p, u3, zero(T))
        c = constraint_norms(p)
        @test all(isfinite, c.gauge_l2) && all(isfinite, c.gauge_linf)
        adm_constraint!(p, u3, zero(T))
        a = constraint_norms(p)
        @test isfinite(a.ham_l2) && isfinite(a.ham_linf) && all(isfinite, a.mom_linf)
        gh_error!(p, u3, zero(T); shell=horizon_shell(case, p.interior))
        e = error_norms(p)
        @test isfinite(e.err_l2) && isfinite(e.err_linf) && e.residual == 0
        @test evolved_nonfinite(p, u3, zero(T)) == 0
        @test isfinite(TGHx.max_speed_of(p, u3, zero(T)))
        val = validity_rows(p, u3, zero(T))
        @test all(x -> x === nothing || isfinite(x), values(val))
        r = excision_rows(p, u3, zero(T))
        @test r.excision_band == ex.nzone && r.excision_band_nonfinite == 0
        @test r.excision_normal_min > 0 && r.excision_into == 0
        # The guard: a query half a cell outside the surface reads 1.5 cells
        # inside it (`q + 2` points), one well outside reads nothing excised.
        h = minimum_spacing(T, forest)
        TGHx.scatter!(U, u3)
        fill_ghosts!(U, p.schedule; boundary=dirichlet(case, zero(T)))
        @test_throws ArgumentError gh_interpolate(U, [SVector(r_E + h / 2, h / 3, h / 5)];
                                                  q=q, mask=mask)
        @test all(isfinite, first(gh_interpolate(U, [SVector(r_E + 3h, h / 3, h / 5)];
                                                 q=q, mask=mask)))
    end

    @testset "an interior without excision parameters prints as it did" begin
        # Guards every checkpoint written before step X2b: the recipe holds
        # `repr(case.interior)`, and the `excision` field the variant added to
        # `Interior` and `FittedSpec` would otherwise change it for every
        # layer variant and refuse their restarts. The strings are the base
        # branch's (8fa6658), printed in the same session; an `:excised`
        # interior prints its parameters, which the recipe then carries.
        @test repr(Interior(T; center=(0, 0, 0), r_0=0.4, r_1=1.15)) ==
              "Interior{Float64, :damped, Nothing}(HoleCenter{Float64}([0.0, 0.0, 0.0], " *
              "[0.0, 0.0, 0.0]), 0.4, 1.15, 0.0, 0.5, 0.5, 8, Val{:damped}(), nothing)"
        @test repr(Interior(Float32; center=(0, 0, 0), r_0=0.4, r_1=1.15)) ==
              "Interior{Float32, :damped, Nothing}(HoleCenter{Float32}(Float32[0.0, " *
              "0.0, 0.0], Float32[0.0, 0.0, 0.0]), 0.4f0, 1.15f0, 0.0f0, 0.5f0, " *
              "0.5f0, 8, Val{:damped}(), nothing)"
        @test repr(FittedSpec(T; variant=:fitted,
                              target_bounds=default_bounds(T; M=1, r_gate=0.9))) ==
              "FittedSpec{Float64, :fitted, Nothing, StateBounds{Float64}}(8, 0, 2, 4, " *
              "8, 0.0, 0.5, 1.0, 3, 0.1, Val{:fitted}(), nothing, " *
              "StateBounds{Float64}(0.02, 50.0, 0.01, 1000.0, 10.0, 100.0, 0.9), " *
              "true, 1)"
        # From step X6 the parameters include the zone points' mixed derivative,
        # `mixed = :symmetric` by default, which prints as a fourth field;
        # `mixed = :nested`, the operator of steps X2b–X5, prints as the struct
        # did before the field, so that a checkpoint those steps wrote restarts
        # under its own operator when it is asked for (and its recipe refuses
        # the new default).
        # From step X8 they include the shave too, `shave = true` by default,
        # printed as a fifth field; `shave = false` prints as the struct did
        # before it — steps X6–X7's layout, and with `mixed = :nested` steps
        # X2b–X5's — so that the checkpoints of every earlier step restart
        # under their own excised set when it is asked for.
        @test occursin("Excision{Float64, :msn}(1.0, 4.0, Val{:msn}(), true, true)",
                       repr(Interior(T; center=(0, 0, 0), r_0=0.4, r_1=0.75,
                                     variant=:excised,
                                     excision=Excision(T; upwind=(1, 4)))))
        @test occursin("Excision{Float64, :msn}(1.0, 4.0, Val{:msn}(), true))",
                       repr(Interior(T; center=(0, 0, 0), r_0=0.4, r_1=0.75,
                                     variant=:excised,
                                     excision=Excision(T; upwind=(1, 4), shave=false))))
        @test occursin("Excision{Float64, :msn}(1.0, 4.0, Val{:msn}()))",
                       repr(Interior(T; center=(0, 0, 0), r_0=0.4, r_1=0.75,
                                     variant=:excised,
                                     excision=Excision(T; upwind=(1, 4),
                                                       mixed=:nested, shave=false))))
        @test repr(Excision(Float32; mixed=:nested, shave=false)) ==
              "Excision{Float32, :msn}(0.0f0, 0.0f0, Val{:msn}())"
        @test repr(Excision(Float32; shave=false)) ==
              "Excision{Float32, :msn}(0.0f0, 0.0f0, Val{:msn}(), true)"
        @test repr(Excision(Float32; mixed=:nested)) ==
              "Excision{Float32, :msn}(0.0f0, 0.0f0, Val{:msn}(), false, true)"
        @test excision_mixed(Excision(T)) === :symmetric && excision_shave(Excision(T))
        @test_throws ArgumentError Excision(T; mixed=:outer)
        @test_throws ArgumentError Interior(T; center=(0, 0, 0), r_0=0.4, r_1=1.15,
                                            excision=Excision(T))
    end

    @testset "the refusals, each by name" begin
        # Guards the checks the design rests on, each where it is asserted:
        # the surface on one level, the singular set inside the core rule's
        # sphere, the margin, `ε_KO > 0`, `m ≥ ⌈√3 G⌉` with a horizon finder,
        # an excised tap of a frame-dragged axis with no source (until step
        # X6: the shift into the excised set, a spinning hole), the range
        # projection, and the driver's keywords and the Π post-pass.
        msg(f) = try
            f()
            ""
        catch err
            sprint(showerror, err)
        end
        mk(c) = () -> setup(c)
        @test occursin("one level",
                       msg(mk(hole_fixture(T; q=q, variant=:excised, r_1=T(11 // 10)))))
        @test occursin("singular set",
                       msg(mk(hole_fixture(T; q=q, a=T(3 // 5), variant=:excised,
                                           r_0=T(1 // 2), r_1=r_E))))
        @test occursin("not far enough inside the horizon",
                       msg(mk(hole_fixture(T; q=q, variant=:excised, r_1=T(3 // 2)))))
        @test occursin("needs Kreiss–Oliger dissipation",
                       msg(mk(hole_fixture(T; q=q, variant=:excised, r_1=r_E,
                                           ε_KO=zero(T)))))
        @test occursin("⌈√3 G⌉",
                       msg(mk(hole_fixture(T; q=q, variant=:excised, r_1=r_E, margin=3,
                                           horizon=Horizon(T; every=1, N=12)))))
        # With the shave the excised set reaches `h/√3` beyond the surface, and
        # at `q = 2` the finder's margin is 5 cells where it was 4 (step X8).
        hz(sh) = mk(hole_fixture(T; q=q, variant=:excised, r_1=r_E, margin=4,
                                 horizon=Horizon(T; every=1, N=12),
                                 excision=Excision(T; shave=sh)))
        @test occursin("⌈√3 G + 1/√3⌉ = 5", msg(hz(true)))
        @test hz(false)()[1].excision.nshaved == 0
        # The shift pointing into the excised set — X2b's refusal of a spinning
        # hole — is from step X6 a frame-dragged axis with a rule of its own,
        # and what is refused is what the rule does not cover: an excised tap
        # with no source along its direction inside the point's `G`-box. On a
        # coarse rotating octant (`q = 4`, `h = 1/4`, `r_E = 17/25`, under
        # three cells) the lopsided blend's row, which reaches `G` into the
        # excised side of a frame-dragged axis, has two such taps; the same
        # surface without the blend builds.
        @test mk(hole_fixture(T; q=q, a=T(3 // 5), variant=:excised, r_0=T(31 // 50),
                              r_1=T(19 // 25)))()[1].excision.ndragged > 0
        coarse(up) = () -> begin
            local c = kerr_schild_case(T; a=T(3 // 5), halfwidth=T(5 // 2), r_0=T(16 // 25),
                                 r_1=T(17 // 25), chunk=T(1 // 10), interior=:excised,
                                 octant=:rotating, margin=4,
                                 excision=Excision(T; upwind=up))
            local f = gh_forest(T, c; N=10, roots=1)
            local Uc = FieldSet{T}(f, 20; G=3, centering=vertexcentered(3),
                             parity=state_parity(f), rotation=state_rotation(f))
            fill_exact!(Uc, c, zero(T))
            GHProblem(Uc, GhostSchedule(Uc, Operators(prolongation=6, restriction=6)),
                      c; q=4).excision
        end
        @test occursin("there is no non-excised point along the lattice direction",
                       msg(coarse((0, 1))))
        @test coarse(nothing)().ndragged == 2
        @test occursin("no range projection",
                       msg(() -> hole_fixture(T; q=q, variant=:excised, r_1=r_E,
                                              bounds=default_bounds(T; M=1,
                                                                    r_gate=T(1 // 2)))))
        ev(; kw...) = () -> evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                                    t_end=T(1 // 10), kw...)
        @test occursin("regrid = true is refused", msg(ev(; regrid=true)))
        @test occursin("adapt = true is refused", msg(ev(; adapt=true)))
        @test occursin("no relaxation rate", msg(ev(; ρ_max_fixed=T(4))))
        @test occursin("refused for an :excised hole",
                       msg(() -> discrete_gradient_momentum!(U, case, zero(T), q,
                                                             p.schedule)))
    end

    @testset "the classes agree across the rotating octant's seam and with the mirror octant, and the two octants evolve as one" begin
        # Guards the excision on TreeAMR 0.1.7's rotating seam (step X4). The
        # class field set's bit is a scalar — even under a mirror, turned into
        # itself by a quarter turn — and without a rotation map TreeAMR
        # refuses the field set over a seam; with the wrong sign the ghosts
        # across the seam would hold minus the excised bit, which reads as
        # evolved, and the points next to the seam would take centered
        # stencils through excised values. On the rotating octant `[0, 5/2]³`
        # (uniform, `h = 5/64`, the fixture's spacing at the hole; `a = 0`)
        # with the ball `r < 3/4` excised:
        #   * every stored point whose image under the quarter turns about `z`
        #     and the mirror at `z = 0` is an owned point — the ghosts across
        #     the seam, the wall's and the blocks' own — has that owned point's
        #     excised bit, found by position;
        #   * the two owned seam planes `x = 0` and `y = 0`, the same points a
        #     quarter turn apart, have the same class — excised, zone or
        #     centered — point by point;
        #   * at `a = 0` the mirror octant is the same problem: its classes are
        #     the rotating octant's at every stored point, its census is the
        #     same, and one right-hand side agrees at every owned point — to
        #     roundoff, 512 eps of each variable's largest `|du|`, since the
        #     ghosts across the seam and across the mirror are copied from
        #     different owned points of the same solution.
        mk(oct; mixed=:symmetric) =
            kerr_schild_case(T; halfwidth=T(5 // 2), r_0=T(2 // 5), r_1=r_E,
                             chunk=T(1 // 10), interior=:excised, octant=oct,
                             excision=Excision(T; mixed=mixed))
        function octant_problem(c)
            f = gh_forest(T, c; N=N, roots=4)
            Uo = FieldSet{T}(f, 20; G=G, centering=vertexcentered(3),
                             parity=state_parity(f), rotation=state_rotation(f))
            fill_exact!(Uo, c, zero(T))
            po = GHProblem(Uo, GhostSchedule(Uo, ops), c; q=q)
            uo = statevector(Uo)
            gather!(uo, Uo)
            return po, uo
        end
        pr, ur = octant_problem(mk(:rotating))
        pm, um = octant_problem(mk(:reflecting))
        Ur, er = pr.U, pr.excision
        @test TGHx.seam_dims(Ur.forest) == (1, 2)
        hs = minimum_spacing(T, Ur.forest)
        # The owned points' classes by their integer position `x/h`.
        owned_cls = Dict{NTuple{3,Int},UInt8}()
        for b in 1:nblocks(Ur), I in owned
            x = coordinates(Ur, b, Tuple(I) .+ G)
            owned_cls[ntuple(d -> round(Int, x[d] / hs), 3)] =
                er.classes[I[1] + G, I[2] + G, I[3] + G, b]
        end
        # The image of a position in the evolved octant: the quarter turn
        # `(x, y) → (y, −x)` until both are non-negative, then the mirror in z.
        function image(k)
            for _ in 1:4
                (k[1] ≥ 0 && k[2] ≥ 0) && break
                k = (k[2], -k[1], k[3])
            end
            return (k[1], k[2], abs(k[3]))
        end
        nseam = 0
        nexc = 0
        bad = 0
        for b in 1:nblocks(Ur), S in CartesianIndices(size(er.classes)[1:3])
            x = coordinates(Ur, b, Tuple(S))
            k = ntuple(d -> round(Int, x[d] / hs), 3)
            c = get(owned_cls, image(k), nothing)
            c === nothing && continue           # beyond the outer faces
            excised = er.classes[S, b] == TGHx.CLASS_EXCISED
            bad += excised != (c == TGHx.CLASS_EXCISED)
            if k[1] < 0 || k[2] < 0
                nseam += 1
                nexc += excised
            end
        end
        @test bad == 0
        @test nseam > 1000 && nexc > 100        # the seam's ghosts reach the ball
        planes = 0
        same = 0
        zones = 0
        for (k, c) in owned_cls
            k[1] == 0 && k[2] > 0 || continue
            planes += 1
            same += owned_cls[(k[2], 0, k[3])] == c
            zones += c == TGHx.CLASS_ZONE
        end
        @test planes > 0 && same == planes && zones > 0
        em = pm.excision
        # (From step X8 the excised set is shaved: its corners are on both
        # octants, and through the seam and the mirror.)
        @test er.nshaved == em.nshaved > 0
        @test er.classes == em.classes
        @test (er.nzone, er.nexcised, er.ncentered, er.nzoneblocks) ==
              (em.nzone, em.nexcised, em.ncentered, em.nzoneblocks)
        dur = similar(ur)
        gh_rhs!(dur, ur, pr, zero(T))
        dum = similar(um)
        gh_rhs!(dum, um, pm, zero(T))
        Ar = statearray(dur, Ur)
        Am = statearray(dum, pm.U)
        nbit = count(((I, b),) -> all(v -> isequal(Ar[I, v, b], Am[I, v, b]), 1:20),
                     Iterators.product(owned, 1:nblocks(Ur)))
        worst = maximum(v -> maximum(abs, Ar[:, :, :, v, :] - Am[:, :, :, v, :]) /
                             maximum(abs, Am[:, :, :, v, :]), 1:20)
        @info "the rotating octant's excised du: $nbit of $(N^3 * nblocks(Ur)) " *
              "points bit for bit the mirror octant's, worst $(worst / eps(T)) eps"
        @test worst ≤ 512 * eps(T)
        # **The two octants evolve as one (step X6).** On the analytic state
        # above both octants' ghosts hold the same solution, so one right-hand
        # side cannot tell an operator that is symmetric in `x ↔ y` from one
        # that is not; an evolution can, since the rotating seam forces the
        # quarter turn's symmetry on the state and the mirror octant does not.
        # With the zone points' mixed derivative symmetric (the default from
        # X6) the two octants, each stepped four times on its own, hold the
        # same state to roundoff (measured `0.77` eps of each variable's
        # largest value) and the same right-hand side on it (`2.9·10³` eps:
        # the states' roundoff over `h²`); with steps X2b–X5's single nesting
        # (`mixed = :nested`) they differ at the truncation level, `1.7·10⁻⁶`
        # and `1.5·10⁻³` — X4's `1.4·10⁻³` at `t = 1` in its making.
        rel(A, B) = maximum(v -> maximum(abs, A[:, :, :, v, :] - B[:, :, :, v, :]) /
                                 maximum(abs, B[:, :, :, v, :]), 1:20)
        dt = gh_dt(pm, um; cfl=T(1 // 4))
        function evolved(po, uo)
            v = gh_solve(po, uo, (zero(T), 4dt); dt=dt)
            dv = similar(v)
            gh_rhs!(dv, v, po, 4dt)
            return statearray(v, po.U), statearray(dv, po.U)
        end
        (sr, fr), (sm, fm) = evolved(pr, ur), evolved(pm, um)
        su, sf = rel(sr, sm), rel(fr, fm)
        (snr, fnr), (snm, fnm) = evolved(octant_problem(mk(:rotating; mixed=:nested))...),
                                 evolved(octant_problem(mk(:reflecting; mixed=:nested))...)
        nu, nf = rel(snr, snm), rel(fnr, fnm)
        @info "four steps on each octant: the states differ by $(su / eps(T)) eps and " *
              "their du by $(sf / eps(T)) eps (symmetric); by $nu and $nf (nested)"
        @test su ≤ 64 * eps(T) && sf ≤ 2^15 * eps(T)
        @test nu > 1e-8 && nf > 1e-5
    end

    @testset "a run to M/5 is finite with a positive normal margin, and restarts as the run" begin
        # Guards the driver's `:excised` path end to end — the frozen
        # geometry, the problem rebuilt per chunk carrying its classes, the
        # outflow rows — and its checkpoint: an `:excised` interior's
        # parameters are in its `repr`, which the recipe holds, and a restart
        # rebuilds the classes from the file's state. Two chunks of `M/10`;
        # the chain of two one-chunk jobs must be the run bit for bit.
        run = evolve!(T, case; forest=deepcopy(forest), q=q, ops=ops,
                      t_end=T(1 // 5))
        rows = run.records
        @test length(rows) == 3
        @test all(r -> r.finite && r.variant === :excised, rows)
        @test all(r -> r.excision_band == ex.nzone && r.excision_band_nonfinite == 0,
                  rows)
        @test all(r -> r.excision_normal_min > 1 && r.excision_into == 0, rows)
        @test all(r -> r.excision_shaved == ex.nshaved > 0, rows)     # step X8
        # Kerr-Schild `a = 0` has no frame-dragged axis (step X6).
        @test all(r -> r.excision_dragged == 0 && r.excision_faces_dragged == 0 &&
                       r.excision_flips == 0, rows)
        @test all(r -> r.residual == 0, rows)
        # Outside the step-5 fixture's own layer, `r ≥ 23/20`, the error is
        # the `:damped` fixture's: `4.390e−3` against `4.395e−3` at `M/5`
        # (measured in step X2b, `CODE.md`, "Excision"); a closure that
        # leaked more than the layer would show here first.
        far = ShellMask(SVector{3,T}(0, 0, 0), T(23 // 20), T(100))
        gh_error!(run.problem, run.u, T(1 // 5); mask=far)
        @test error_norms(run.problem).err_l2 < 4.5e-3
        @info "the excised fixture to M/5" err_l2 = [r.err_l2 for r in rows] gauge_l2 =
            [r.gauge_l2 for r in rows] normal = [r.excision_normal_min for r in rows]
        pre = joinpath(mktempdir(), "run")
        common = (q=q, ops=ops, t_end=T(1 // 5), checkpoint_path_prefix=pre,
                  max_walltime_seconds=1e-9, checkpoint_sync_to_disk=false)
        first_ = evolve!(T, case; forest=deepcopy(forest), common...)
        @test !first_.finished
        second = evolve!(T, case; common..., restart_file=latest_checkpoint(pre))
        @test second.finished
        @test isequal(second.u, run.u)
        @test isequal(second.records, run.records)
    end

    # --- step X6: the frame-dragged faces in the zone kernel, and the mixed
    # derivative symmetric in its two axes (`CODE.md`, "Excision", "What step
    # X6 built"). The spinning hole is Kerr-Schild `a = 3/5` on the rotating
    # octant `[0, 5/2]³` (uniform, `h = 5/64`, the fixture's spacing at the
    # hole) with the ball `r < 19/25` excised and the core rule's sphere at
    # `31/50`, between the ring and the surface: 27 frame-dragged axes with
    # the lego corners shaved (24 corners on the octant; 33 axes without the
    # shave, until step X8).

    @testset "the symmetric mixed derivative takes one value across the diagonal, bit for bit, and the nested one does not" begin
        # Guards the symmetrization (step X6, proposed in step X4). Near the
        # surface the `x`- and `y`-closures differ, so `D_x D_y ≠ D_y D_x` at
        # the truncation level, and an operator with one nesting is not
        # equivariant under the reflection `x ↔ y` — nor under the rotating
        # octant's quarter turn. `½(D_x D_y + D_y D_x)` at a point is the same
        # arithmetic as at its mirror image on the mirrored data, `a + b ==
        # b + a`, so it must agree bit for bit; the nested sum must not. On one
        # host block with an excised half-space that is not symmetric across
        # the diagonal (`2i + j + k ≤ c`) and smooth data that is not either,
        # at every evolved point whose `(x, y)` box meets the excised set; and
        # `∂_x∂_z` at a point is `∂_y∂_z` at its image in both nestings.
        for qq in (2, 4)
            GG = qq ÷ 2 + 1
            n = 4GG + 9
            r = qq ÷ 2
            tab = TGHx.closure_arrays(T, Val(qq), :msn, TreeGeneralizedHarmonic.CPU())
            c0 = 2n
            excised(I) = 2I[1] + I[2] + I[3] ≤ c0
            f(x) = sin(0.4 * x[1] - 0.3 * x[2] + 0.2 * x[3]) + 0.01 * x[1]^2 * x[2]
            sw(I) = (I[2], I[1], I[3])
            work = (Array{T}(undef, n, n, n, 1, 1), Array{T}(undef, n, n, n, 1, 1))
            cl = (Array{UInt8}(undef, n, n, n, 1), Array{UInt8}(undef, n, n, n, 1))
            for I in CartesianIndices((n, n, n)), (k, J) in ((1, Tuple(I)), (2, sw(Tuple(I))))
                work[k][I, 1, 1] = excised(J) ? T(NaN) : f(T.(J))
                cl[k][I, 1] = excised(J) ? TGHx.CLASS_EXCISED : TGHx.CLASS_ZONE
            end
            wst, _, _ = TGHx.work_strides(work[1])
            base(I) = 1 + (I[1] - 1) * wst[1] + (I[2] - 1) * wst[2] + (I[3] - 1) * wst[3]
            npts = 0
            sym = true
            nested_diff = 0
            others = true
            for I in CartesianIndices((n, n, n))
                t = Tuple(I)
                (all(d -> GG < t[d] ≤ n - GG, 1:3) && !excised(t)) || continue
                any(excised((t[1] + a, t[2] + b, t[3])) for a in (-r):r, b in (-r):r) ||
                    continue
                for SYM in (true, false)
                    SA = TGHx.closure_provider(T, Val(GG), wst, cl[1], base(t), tab, one(T),
                                               zero(T), Val(SYM))
                    SB = TGHx.closure_provider(T, Val(GG), wst, cl[2], base(sw(t)), tab,
                                               one(T), zero(T), Val(SYM))
                    xy = isequal(TGHx.dmix(SA, work[1], base(t), 1, 2),
                                 TGHx.dmix(SB, work[2], base(sw(t)), 1, 2))
                    SYM ? (sym &= xy) : (nested_diff += !xy)
                    others &= isequal(TGHx.dmix(SA, work[1], base(t), 1, 3),
                                      TGHx.dmix(SB, work[2], base(sw(t)), 2, 3))
                end
                npts += 1
            end
            @info "q = $qq: the (x, y) mixed derivative at $npts points next to the " *
                  "surface; the nested one differs from its diagonal image at " *
                  "$nested_diff of them"
            @test npts > 100
            @test sym && others
            @test nested_diff > npts ÷ 2
        end
    end

    @testset "the frame-dragged rule's advection is exact on polynomials to its degree, and reads no excised value" begin
        # Guards `DraggedProvider` (step X6, step X5's rule): along a
        # frame-dragged axis the advection is the centered `D₁` — with the
        # lopsided blend at full weight the open lopsided row — with each
        # excised tap extrapolated along its direction code from up to three
        # sources inside the point's `G`-box. A tap read at the wrong index, a
        # source outside the box or on the excised set, weights of the wrong
        # `(k₀, n)` or a stencil row of the wrong offset would show on a
        # quadratic, which three sources extrapolate exactly (at `q = 4`; at
        # `q = 2` there are two at most), and on a linear function where a tap
        # has two; a cubic, which they do not extrapolate, must show that the
        # extrapolation is used. On one host
        # block with an excised ball off the lattice's symmetry, the direction
        # codes from its radial normal, every closure axis's bit set, every
        # excised value `NaN`.
        for qq in (2, 4)
            GG = qq ÷ 2 + 1
            n = 6GG + 9
            tab = TGHx.closure_arrays(T, Val(qq), :msn, TreeGeneralizedHarmonic.CPU())
            ext = TGHx.extrapolation_table(T, Val(qq))
            ctr = (n / 2 + 0.37, n / 2 - 0.21, n / 2 + 0.13)
            R = 2GG + 1.3
            excised(I) = sum(abs2, I .- ctr) < R^2
            rng = Xoshiro(10 + qq)
            mons = [(a, b, c) for a in 0:3 for b in 0:3 for c in 0:3 if a + b + c ≤ 3]
            coef = [T(rand(rng, -9:9)) / 8 for _ in mons]
            P(x, deg) = sum(coef[m] * prod(x[k]^p[k] for k in 1:3)
                            for (m, p) in enumerate(mons) if sum(p) ≤ deg)
            dP(x, deg, d) = sum(coef[m] * p[d] * x[d]^(p[d] - 1) *
                                prod(k == d ? 1 : x[k]^p[k] for k in 1:3)
                                for (m, p) in enumerate(mons) if sum(p) ≤ deg && p[d] > 0)
            work = Array{T}(undef, n, n, n, 3, 1)
            cl = Array{UInt8}(undef, n, n, n, 1)
            codes = zeros(UInt8, n, n, n, 1)
            for I in CartesianIndices((n, n, n))
                t = Tuple(I)
                ex_ = excised(t)
                cl[I, 1] = ex_ ? TGHx.CLASS_EXCISED : TGHx.CLASS_ZONE
                for (v, deg) in enumerate((1, 2, 3))
                    work[I, v, 1] = ex_ ? T(NaN) : P(T.(t), deg)
                end
                if ex_
                    nr = (t .- ctr) ./ sqrt(sum(abs2, t .- ctr))
                    codes[I, 1] = TGHx.direction_code(T.(nr)...)
                end
            end
            wst, wsv, _ = TGHx.work_strides(work)
            scale = maximum(abs, filter(isfinite, work))
            nsrc = zeros(Int, 4)
            nstencil = zeros(Int, 3)               # by the fewest sources of a tap
            worst2 = 0.0
            worst1 = 0.0
            cubic = 0.0
            for I in CartesianIndices((n, n, n))
                t = Tuple(I)
                (all(d -> GG < t[d] ≤ n - GG, 1:3) && !excised(t)) || continue
                b0 = 1 + (t[1] - 1) * wst[1] + (t[2] - 1) * wst[2] + (t[3] - 1) * wst[3]
                for λ in (zero(T), one(T))
                    C0 = TGHx.closure_provider(T, Val(GG), wst, cl, b0, tab, one(T), λ)
                    any(d -> max(C0.km[d], C0.kp[d]) < GG, 1:3) && continue  # inadmissible
                    rule = sum((min(C0.km[d], C0.kp[d]) < GG) << (d - 1) for d in 1:3)
                    S = dragged_provider(C0, codes, ext, rule)
                    for d in 1:3, β in (-one(T), one(T))
                        ks, s = C0.km[d] < GG ? (C0.km[d], -1) : (C0.kp[d], 1)
                        # The taps the rule reads on the excised side: the
                        # centered `D₁`'s to `q/2`, and at full blend the
                        # lopsided row's too — to `G` when `β` points there
                        # (its upwind side), to `q/2 − 1` when it does not.
                        reach = !iszero(λ) && β * s > 0 ? GG : qq ÷ 2
                        ks < reach || continue
                        fewest = 3
                        for j in (ks + 1):reach
                            cQ = b0 + s * j * wst[d]
                            cl[cQ] == TGHx.CLASS_EXCISED || continue
                            o = s * j
                            _, m, _ = TGHx._tap_sources(cl, codes, cQ, wst, d == 1 ? o : 0,
                                                        d == 2 ? o : 0, d == 3 ? o : 0,
                                                        Val(GG), Val(3))
                            nsrc[m + 1] += 1
                            fewest = min(fewest, m)
                        end
                        fewest == 0 && continue          # the build refuses these
                        nstencil[fewest] += 1
                        x = T.(t)
                        for (v, deg) in enumerate((1, 2, 3))
                            # (`∂f_d`, the closure's own derivative, enters
                            # only where the centered `D₁` reaches no excised
                            # tap, and then at the blend's weight, zero here.)
                            got = TGHx.adv(S, β, zero(T), work, b0 + (v - 1) * wsv, d)
                            err = abs(got - dP(x, deg, d))
                            deg == 1 && fewest ≥ 2 && (worst1 = max(worst1, err))
                            deg == 2 && fewest == 3 && (worst2 = max(worst2, err))
                            deg == 3 && (cubic = max(cubic, err))
                        end
                    end
                end
            end
            @info "q = $qq: excised taps with 0, 1, 2, 3 sources $(nsrc); stencils by " *
                  "their fewest $(nstencil); a cubic's worst error $(cubic / scale)"
            # At `q = 2` the sources stop at `G = 2` steps (`k₀ + n − 1 ≤ G`,
            # `extrapolation_table`), so two at most and the rule is linear.
            @test nstencil[min(GG, 3)] > 100 && (GG == 2 || nstencil[2] > 0)
            @test worst1 ≤ 1e-12 * scale && worst2 ≤ 1e-12 * scale
            @test cubic > 1e-6 * scale
        end
    end

    c35 = kerr_schild_case(T; a=T(3 // 5), halfwidth=T(5 // 2), r_0=T(31 // 50),
                           r_1=T(19 // 25), chunk=T(1 // 10), interior=:excised,
                           octant=:rotating)
    f35 = gh_forest(T, c35; N=N, roots=4)
    function spin_problem(c)
        Us = FieldSet{T}(f35, 20; G=G, centering=vertexcentered(3),
                         parity=state_parity(f35), rotation=state_rotation(f35))
        fill_exact!(Us, c, zero(T))
        ps = GHProblem(Us, GhostSchedule(Us, ops), c; q=q)
        us = statevector(Us)
        gather!(us, Us)
        return ps, us
    end
    p35, u35 = spin_problem(c35)
    e35 = p35.excision
    U35 = p35.U
    du35 = similar(u35)
    gh_rhs!(du35, u35, p35, zero(T))
    A35 = statearray(du35, U35)
    codes35 = Array(e35.codes)
    cls35(I, b) = e35.classes[I[1] + G, I[2] + G, I[3] + G, b]
    rule35(I, b) = Int(codes35[I[1] + G, I[2] + G, I[3] + G, b])

    @testset "a spinning hole builds: its frame-dragged axes are the state's, along x and y only, and their taps have sources" begin
        # Guards the build of step X6: the rule bits and direction codes are
        # built once, from the state and the frozen geometry, and the kernel
        # reads nothing else. A bit that is not the census's criterion — the
        # shift of the state pointing into the excised set along a closure
        # axis, `b/a < 0` on the side whose run is shorter than `G` — would
        # change the operator where X5 did not ask; a direction code that is
        # not the lattice direction nearest the surface's normal would
        # extrapolate along another line. Enumerated on the host at every
        # owned zone point and every stored excised point. X5's numbers: no
        # `z` axis is frame-dragged at `a = 3/5`, and on this octant's
        # quadrant the shift's frame-dragged part points into the ball along
        # `y` only (the `x` ones are the other quadrants').
        st35, _, _ = TGHx.work_strides(U35.work)
        S35 = statearray(u35, U35)
        cl = Array(e35.classes)
        bad = 0
        peraxis = zeros(Int, 3)
        nzone = 0
        for b in 1:nblocks(U35), I in owned
            cls35(I, b) == TGHx.CLASS_ZONE || continue
            nzone += 1
            cb = 1 + (b - 1) * prod(size(cl)[1:3]) + (I[1] + G - 1) * st35[1] +
                 (I[2] + G - 1) * st35[2] + (I[3] + G - 1) * st35[3]
            hv = SVector{10,T}(ntuple(v -> S35[I, v, b], 10))
            _, _, α, β, γu, _ = TGHx.metric_quantities(TGHx._sym4(hv))
            want = 0
            for d in 1:3, s in (-1, 1)
                k = TGHx._run(cl, cb, s * st35[d], Val(G))
                k < G && -s * β[d] / (α * sqrt(γu[d, d])) < 0 && (want |= 1 << (d - 1))
            end
            bad += want != rule35(I, b)
            for d in 1:3
                peraxis[d] += (want >> (d - 1)) & 1
            end
        end
        @test bad == 0 && nzone == e35.nzone
        @test peraxis == [0, e35.ndragged, 0] && e35.ndragged == 27
        @test e35.nshaved == 24
        @test e35.drag !== nothing
        badcode = 0
        nexc = 0
        for b in 1:nblocks(U35), S in CartesianIndices(size(cl)[1:3])
            cl[S, b] == TGHx.CLASS_EXCISED || continue
            nexc += 1
            x = coordinates(U35, b, Tuple(S))
            nr = TGHx.excision_normal(p35.interior, zero(T), x)
            badcode += codes35[S, b] != TGHx.direction_code(nr[1], nr[2], nr[3])
        end
        @test badcode == 0 && nexc > e35.nexcised
        r = excision_rows(p35, u35, zero(T))
        @test r.excision_dragged == e35.ndragged == r.excision_into
        @test r.excision_flips == 0 && r.excision_band == e35.nzone
        @test r.excision_shaved == e35.nshaved > 0                     # step X8
        @test 0 < r.excision_faces_dragged < r.excision_faces
        @test r.excision_normal_min > 0 && r.excision_axis_min < 0
        @test all(isfinite, du35)
    end

    @testset "the frame-dragged rule changes nothing where no axis is frame-dragged" begin
        # Guards "the exterior is unchanged" one level in (step X6): the rule
        # is a second launch over the zone points with a bit set, and a zone
        # point with none must keep exactly the `du` the zone kernel gave it.
        # At `a = 0` the build finds no frame-dragged axis and launches no
        # rule; launched all the same, over every zone block, it must change
        # nothing. At `a = 3/5`, without the rule, every zone point whose bits
        # are clear must keep its `du` bit for bit, and so must a
        # frame-dragged one whose centered `D₁` reaches no excised point
        # (`k_s = q/2`); the rest change. The rule being a launch of its own,
        # the zone kernel is the same compiled code with and without it, so
        # these are claims about which points the second launch writes — not
        # about how the compiler contracts the head around a second provider
        # (`CLAUDE.md`, "A provider changes the code around the head").
        @test ex.drag === nothing && ex.ndragged == 0
        duA = similar(u)
        gh_rhs!(duA, u, p, zero(T))
        duB = copy(duA)
        TGHx.gh_zone!(duB, p, zero(T);
                      drag=(codes=ex.codes, ext=TGHx.extrapolation_table(T, Val(q)),
                            blocks=ex.zoneblocks))
        @test isequal(duA, duB)
        dA = similar(u35)
        gh_rhs!(dA, u35, p35, zero(T))
        @test isequal(dA, du35)
        dB = copy(dA)
        TGHx.gh_zone!(dB, p35, zero(T); drag=nothing)
        Bn = statearray(dB, U35)
        st35, _, _ = TGHx.work_strides(U35.work)
        cl = Array(e35.classes)
        clear_same = 0
        nclear = 0
        reach_changed = 0
        nreach = 0
        noreach_same = 0
        nnoreach = 0
        other_same = 0
        nother = 0
        for b in 1:nblocks(U35), I in owned
            same = all(v -> isequal(A35[I, v, b], Bn[I, v, b]), 1:20)
            if cls35(I, b) != TGHx.CLASS_ZONE
                nother += 1
                other_same += same
                continue
            end
            rb = rule35(I, b)
            if rb == 0
                nclear += 1
                clear_same += same
                continue
            end
            cb = 1 + (b - 1) * prod(size(cl)[1:3]) + (I[1] + G - 1) * st35[1] +
                 (I[2] + G - 1) * st35[2] + (I[3] + G - 1) * st35[3]
            reaches = any(d -> (rb >> (d - 1)) & 1 == 1 &&
                               min(TGHx._run(cl, cb, -st35[d], Val(G)),
                                   TGHx._run(cl, cb, st35[d], Val(G))) < q ÷ 2, 1:3)
            if reaches
                nreach += 1
                reach_changed += !same
            else
                nnoreach += 1
                noreach_same += same
            end
        end
        @info "a = 3/5 without the rule: $clear_same of $nclear zone points with no " *
              "frame-dragged axis unchanged, $noreach_same of $nnoreach whose D₁ reaches " *
              "no excised point, $reach_changed of $nreach others changed"
        @test clear_same == nclear > 200 && other_same == nother
        @test noreach_same == nnoreach && reach_changed == nreach > 0
    end

    @testset "no kernel reads an excised value on the spinning hole, and nothing but the integrator writes its state" begin
        # Guards the rule's extrapolation against reading the excised set: a
        # degenerate metric planted on every excised owned point — and through
        # the ghost exchange on every excised ghost, across the seam too — must
        # leave every non-excised `du` bit for bit, and every excised one zero.
        # And against a fourth writer: the right-hand side with the rule's
        # launch leaves its state alone, and so does the `:excised` limiter.
        w = copy(u35)
        dw = similar(u35)
        gh_rhs!(dw, w, p35, zero(T))
        @test isequal(w, u35) && isequal(dw, du35)
        gh_limiter!(w, nothing, p35, zero(T))
        @test isequal(w, u35)
        u2 = copy(u35)
        S2 = statearray(u2, U35)
        η = (1, 0, 0, 0, -1, 0, 0, -1, 0, -1)
        for b in 1:nblocks(U35), I in owned
            cls35(I, b) == TGHx.CLASS_EXCISED || continue
            for v in 1:10
                S2[I, v, b] = η[v]
                S2[I, 10 + v, b] = T(NaN)
            end
        end
        du2 = similar(u35)
        gh_rhs!(du2, u2, p35, zero(T))
        B = statearray(du2, U35)
        same = 0
        zero_ = 0
        for b in 1:nblocks(U35), I in owned
            if cls35(I, b) == TGHx.CLASS_EXCISED
                zero_ += all(v -> B[I, v, b] === zero(T), 1:20)
            else
                same += all(v -> isequal(A35[I, v, b], B[I, v, b]), 1:20)
            end
        end
        @test same == e35.nzone + e35.ncentered
        @test zero_ == e35.nexcised
    end

    @testset "the spinning hole's centered points are the unexcised operator's" begin
        # Guards the main kernel at `a = 3/5`: the rule lives in the zone
        # kernel and nowhere else, so every centered point is the `:none`
        # kernel's `F` on the same state (to 512 eps, two specialisations).
        none = kerr_schild_case(T; a=T(3 // 5), halfwidth=T(5 // 2), chunk=T(1 // 10),
                                interior=nothing, octant=:rotating)
        Un = FieldSet{T}(f35, 20; G=G, centering=vertexcentered(3),
                         parity=state_parity(f35), rotation=state_rotation(f35))
        pn = GHProblem(Un, GhostSchedule(Un, ops), none; q=q)
        dun = similar(u35)
        gh_rhs!(dun, u35, pn, zero(T))
        C = statearray(dun, Un)
        worst = 0.0
        scale = 0.0
        nbit = 0
        for b in 1:nblocks(U35), I in owned
            cls35(I, b) == TGHx.CLASS_CENTERED || continue
            worst = max(worst, maximum(v -> abs(A35[I, v, b] - C[I, v, b]), 1:20))
            scale = max(scale, maximum(v -> abs(A35[I, v, b]), 1:20))
            nbit += all(v -> isequal(A35[I, v, b], C[I, v, b]), 1:20)
        end
        @info "a = 3/5: $nbit of $(e35.ncentered) centered points bit for bit the " *
              ":none kernel's, worst $(worst / (eps(T) * scale)) eps"
        @test worst ≤ 512 * eps(T) * scale
    end

    @testset "the spinning hole runs to M/5 with the frame-dragged rows, and without a flip restarts as the run" begin
        # Guards the driver's path with the rule (step X6): the rule bits are
        # the build state's and a restart rebuilds them from the file's state,
        # so a chain is the run exactly when no axis's shift has changed its
        # sign (`excision_flips == 0` at the checkpoint's row). Two chunks of
        # `M/10` on the rotating octant, and the chain of two one-chunk jobs.
        # **(Amended in step X8:** on this fixture's lattice the shave makes
        # `(6, 5, 7) h` a corner of the shaved set — `k⁻ = 0` along all three
        # axes — whose `y` axis has `b/a = +1.7·10⁻⁴` at the build and flips
        # its sign by `M/10` (`excision_flips = 1` there, 0 at `0` and
        # `M/5`), so its chain would not be the run: X6's claim is made on
        # the same hole without the shave, which has no flip, and the shaved
        # hole's run is claimed finite with its rows and its one flip, the
        # staircase's lottery step X7 met, in miniature.**)**
        c35u = kerr_schild_case(T; a=T(3 // 5), halfwidth=T(5 // 2), r_0=T(31 // 50),
                                r_1=T(19 // 25), chunk=T(1 // 10), interior=:excised,
                                octant=:rotating, excision=Excision(T; shave=false))
        shaved = evolve!(T, c35; forest=deepcopy(f35), q=q, ops=ops, t_end=T(1 // 5))
        srows = shaved.records
        @test length(srows) == 3 && all(r -> r.finite && r.excision_band_nonfinite == 0,
                                        srows)
        @test all(r -> r.excision_dragged == e35.ndragged &&
                       r.excision_shaved == e35.nshaved, srows)
        @test [r.excision_flips for r in srows] == [0, 1, 0]
        common = (forest=deepcopy(f35), q=q, ops=ops, t_end=T(1 // 5))
        run = evolve!(T, c35u; common...)
        rows = run.records
        e35u = run.problem.excision
        @test e35u.ndragged == 33 && e35u.nshaved == 0
        @test length(rows) == 3
        @test all(r -> r.finite && r.excision_band_nonfinite == 0, rows)
        @test all(r -> r.excision_dragged == e35u.ndragged == r.excision_into &&
                       r.excision_flips == 0, rows)
        @test all(r -> r.excision_faces_dragged == rows[1].excision_faces_dragged > 0,
                  rows)
        @test all(r -> r.excision_normal_min > 0 && r.excision_shaved == 0, rows)
        @info "the spinning hole to M/5" err_l2 = [r.err_l2 for r in rows] gauge_l2 =
            [r.gauge_l2 for r in rows] axis_min = [r.excision_axis_min for r in rows]
        pre = joinpath(mktempdir(), "spin")
        ck = (q=q, ops=ops, t_end=T(1 // 5), checkpoint_path_prefix=pre,
              max_walltime_seconds=1e-9, checkpoint_sync_to_disk=false)
        first_ = evolve!(T, c35u; forest=deepcopy(f35), ck...)
        @test !first_.finished
        second = evolve!(T, c35u; ck..., restart_file=latest_checkpoint(pre))
        @test second.finished
        @test isequal(second.u, run.u)
        @test isequal(second.records, run.records)
    end
end
