# Stage 2: pointwise_tests.jl:97 alone — `K` from the package's
# `adm_vars_from_state` on `gh_state` against `SpacetimeMetrics`'
# `ExtrinsicCurvature`, on the six test backgrounds. Run from test/.
using StaticArrays: SVector
import SpacetimeMetrics
include(joinpath(pwd(), "pointwise_backgrounds.jl"))
function stage2()
    T = Float64
    nbad = 0; ntot = 0; worst = 0.0
    for (name, bg, _) in gh_backgrounds(T), x in gh_points(T, 2)
        p = SVector{4,T}(T(GH_TIME), x[1], x[2], x[3])
        hs, Π, ∂h = gh_state(bg, T(GH_TIME), x)
        γs, ∂γ, K = TreeGeneralizedHarmonic.adm_vars_from_state(hs, Π, ∂h[1], ∂h[2], ∂h[3])
        Kr = SpacetimeMetrics.ExtrinsicCurvature(bg, p)
        e = absdiff(K, Kr) / max(maximum(abs, Kr), one(T))
        e = isnan(e) ? Inf : e
        worst = max(worst, e); ntot += 1; nbad += e > 256 * eps(T)
    end
    println("STAGE2 julia=", VERSION, " target=", Sys.CPU_NAME,
            " coverage=", Base.JLOptions().code_coverage,
            " check_bounds=", Base.JLOptions().check_bounds,
            " worst K rel err=", worst, " bad=", nbad, "/", ntot)
end
stage2()
