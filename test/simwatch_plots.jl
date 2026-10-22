# Thumbnails for SimWatch (added 2026-10-19): time series from a run's CSV
# drawn as small log-scale line plots, which `simwatch.toml`'s `[[images]]`
# point at and SimWatch shows in its detail view.
#
# SVG from 2026-10-21, by SimWatch 1.3.0's `FORMAT.md` (until then PNGs drawn
# pixel by pixel here, dark on light): SVG text written by hand, with no
# plotting package, which SimWatch draws with its own built-in font (DejaVu
# Sans, so `−`, Greek and superscripts are fine) at 512 pixels on the longer
# side. What the format asks of it, and what is done:
#
# - **self-contained**: lines, `<text>`, no `<image>` and no other file;
# - **the background**: white, black or transparent — here *transparent*,
#   with the axes and labels in `currentColor` (black unless an embedding
#   page says otherwise). SimWatch reads a transparent image with black ink as
#   drawn for white and shows it as it is or flips it for a dark terminal, as
#   its user's `image_theme` says; the plot itself no longer chooses. The
#   grid is `currentColor` at low opacity, so that SimWatch's detection of the
#   background does not count it as light ink;
# - **colours**: the series in saturated hues of middling lightness (the PNGs'
#   palette, with brown for its black), which read on white and keep their
#   hue when SimWatch flips the image;
# - **size**: `400 × 300` user units (`FORMAT.md`'s "about 400×300 is best"),
#   far below its 1 MiB limit — a series is thinned to at most 1000 points;
# - **atomic**: written to `<file>.tmp` and renamed over the old file, before
#   the status file that names it is rewritten.
#
# Included by `test/octant_runs.jl` and `test/simwatch_tests.jl`. Nothing here
# may stop a run — `simwatch_plot` returns `false` instead of throwing.

const SW_PALETTE = ["#d55e00", "#0072b2", "#009e73", "#963caa", "#8c564b", "#8c8c8c",
                    "#c81e1e"]
const SW_PALETTE_NAMES = ["orange", "blue", "green", "purple", "brown", "grey", "red"]

# A coordinate with one decimal: a tenth of a user unit is a seventh of a pixel.
_sw_fmt(x) = string(round(x; digits=1))

# A tick label's number, to three digits: `0.25`, `12.5`, `130`.
_sw_tick(x) = (r = round(x; sigdigits=3); isinteger(r) ? string(Int(r)) : string(r))

"""
    simwatch_plot(path, t, series; width = 400, height = 300) -> Bool

A log-scale line plot of the vectors in `series` against `t`, one colour of
`SW_PALETTE` each in order, written to `path` as SVG (SimWatch 1.3.0's
`FORMAT.md`; see this file's header): decades labelled `10⁻⁶` on the left,
the time axis below, a transparent background with the axes in
`currentColor`. Non-positive and non-finite values break their line. Returns
whether it worked; it never throws.
"""
function simwatch_plot(path, t::AbstractVector, series; width=400, height=300)
    try
        vals = [log10(v) for s in series for v in s if isfinite(v) && v > 0]
        (length(t) ≥ 1 && !isempty(vals)) || return false
        xl, xr, yt, yb = 46.0, width - 10.0, 10.0, height - 26.0
        lo, hi = floor(minimum(vals)), ceil(maximum(vals))
        hi == lo && (hi += 1)
        tf = [Float64(x) for x in t if isfinite(x)]
        isempty(tf) && return false
        t0, t1 = minimum(tf), maximum(tf)
        t1 == t0 && (t1 = t0 + 1)
        px(tt) = xl + (xr - xl) * (tt - t0) / (t1 - t0)
        py(lv) = yb - (yb - yt) * (lv - lo) / (hi - lo)
        io = IOBuffer()
        println(io, """<?xml version="1.0" encoding="UTF-8"?>""")
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" """ *
                    """height="$height" viewBox="0 0 $width $height" """ *
                    """font-family="DejaVu Sans, sans-serif" font-size="12">""")
        # The grid: the decades and four time ticks.
        println(io, """<g stroke="currentColor" stroke-opacity="0.18" stroke-width="1">""")
        for d in Int(lo):Int(hi)
            y = _sw_fmt(py(d))
            println(io, """<line x1="$(_sw_fmt(xl))" y1="$y" x2="$(_sw_fmt(xr))" y2="$y"/>""")
        end
        for k in 1:4
            x = _sw_fmt(px(t0 + k * (t1 - t0) / 4))
            println(io, """<line x1="$x" y1="$(_sw_fmt(yt))" x2="$x" y2="$(_sw_fmt(yb))"/>""")
        end
        println(io, "</g>")
        # The axes and their labels: the decades as powers of ten, where
        # there is room, and the time at five ticks.
        println(io, """<g fill="currentColor">""")
        step = max(1, ceil(Int, (hi - lo) / 8))
        for d in Int(lo):Int(hi)
            (d - Int(lo)) % step == 0 || continue
            e = d < 0 ? "−$(-d)" : string(d)
            println(io, """<text x="$(_sw_fmt(xl - 4))" y="$(_sw_fmt(py(d) + 4))" """ *
                        """text-anchor="end">10<tspan dy="-5" font-size="9">$e</tspan></text>""")
        end
        for k in 0:4
            x = px(t0 + k * (t1 - t0) / 4)
            anchor = k == 0 ? "start" : k == 4 ? "end" : "middle"
            println(io, """<text x="$(_sw_fmt(x))" y="$(_sw_fmt(yb + 16))" """ *
                        """text-anchor="$anchor">$(_sw_tick(t0 + k * (t1 - t0) / 4))</text>""")
        end
        println(io, "</g>")
        println(io, """<path d="M$(_sw_fmt(xl)),$(_sw_fmt(yt))V$(_sw_fmt(yb))""" *
                    """H$(_sw_fmt(xr))" fill="none" stroke="currentColor" """ *
                    """stroke-width="1.2"/>""")
        # The series, each a run of polylines broken where a value is missing,
        # and a dot for a point alone. At most 1000 points of each.
        stride = cld(length(t), 1000)
        for (i, s) in enumerate(series)
            color = SW_PALETTE[mod1(i, length(SW_PALETTE))]
            runs = Vector{Vector{String}}()
            run = String[]
            for j in 1:stride:min(length(t), length(s))
                tt, v = t[j], s[j]
                if isfinite(tt) && isfinite(v) && v > 0
                    push!(run, "$(_sw_fmt(px(Float64(tt)))),$(_sw_fmt(py(log10(v))))")
                elseif !isempty(run)
                    push!(runs, run)
                    run = String[]
                end
            end
            isempty(run) || push!(runs, run)
            for r in runs
                if length(r) == 1
                    cx, cy = split(only(r), ',')
                    println(io, """<circle cx="$cx" cy="$cy" r="1.6" fill="$color"/>""")
                else
                    println(io, """<polyline points="$(join(r, ' '))" fill="none" """ *
                                """stroke="$color" stroke-width="1.6" """ *
                                """stroke-linejoin="round" stroke-linecap="round"/>""")
                end
            end
        end
        println(io, "</svg>")
        mkpath(dirname(path))
        tmp = path * ".tmp"
        write(tmp, take!(io))
        # `rename` replaces the old file atomically; `mv(; force = true)`
        # would delete it first.
        Base.Filesystem.rename(tmp, path)
        return true
    catch err
        err isa InterruptException && rethrow()
        @warn "SimWatch: could not draw $path" exception = err maxlog = 1
        return false
    end
end

"""
    simwatch_csv_columns(csvpath, names) -> (t, columns)

The `t` column and the named columns of a run's CSV, as `Float64` vectors
(`NaN` where a cell does not parse, and for a name the header lacks).
"""
function simwatch_csv_columns(csvpath, names)
    lines = readlines(csvpath)
    header = split(first(lines), ',')
    idx(n) = findfirst(==(n), header)
    rows = [split(l, ',') for l in lines[2:end] if !isempty(l)]
    col(i) = i === nothing ? fill(NaN, length(rows)) :
             [i ≤ length(r) ? something(tryparse(Float64, r[i]), NaN) : NaN for r in rows]
    return col(idx("t")), [col(idx(n)) for n in names]
end
